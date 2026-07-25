import asyncio
import io
import json
import os
import sys
import tempfile
import uuid
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# In-memory job store
jobs: dict = {}


@app.post("/api/process")
async def process(
    model_files: list[UploadFile] = File(...),
    mapping_files: list[UploadFile] = File(...),
    expr_file: UploadFile = File(...),
    batch_count: int = Form(1000),
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "messages": [],
        "output_zip": None,
        "error": None,
        "file_count": batch_count,
        "species_count": len(model_files),
    }

    # Save uploaded files to temp dir
    tmp = tempfile.mkdtemp()

    model_paths = []
    for f in model_files:
        content = await f.read()
        path = os.path.join(tmp, f.filename)
        with open(path, "wb") as out:
            out.write(content)
        model_paths.append(path)

    mapping_paths = []
    for f in mapping_files:
        content = await f.read()
        path = os.path.join(tmp, f.filename)
        with open(path, "wb") as out:
            out.write(content)
        mapping_paths.append(path)

    expr_content = await expr_file.read()
    expr_path = os.path.join(tmp, expr_file.filename)
    with open(expr_path, "wb") as out:
        out.write(expr_content)

    species_prefixes = [Path(f.filename).stem.split("_")[0] for f in model_files]

    asyncio.create_task(
        run_bootstrap(job_id, model_paths, mapping_paths, expr_path, species_prefixes, batch_count, tmp)
    )

    return {"job_id": job_id}


async def run_bootstrap(job_id, model_paths, mapping_paths, expr_path, species_prefixes, batch_count, tmp):
    job = jobs[job_id]

    def add_msg(text, type_=""):
        job["messages"].append({"text": text, "type": type_})

    try:
        add_msg(f"Starting bootstrap for: {', '.join(species_prefixes)}", "info")
        job["progress"] = 10

        output_dir = os.path.join(tmp, "output")
        os.makedirs(output_dir, exist_ok=True)

        captured = io.StringIO()

        def run_sync():
            sys.path.insert(0, "/app")
            from utils.bootstrap_genes import bootstrap_genes
            with redirect_stdout(captured):
                bootstrap_genes(
                    model_pre_filenames=model_paths,
                    mapping_filenames=mapping_paths,
                    species_prefixes=species_prefixes,
                    combined_geneExpr_filename=expr_path,
                    geneExpr_folder=output_dir,
                    batch_count=batch_count,
                )

        job["progress"] = 20
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_sync)
        job["progress"] = 80

        # Relay only key status lines — skip all Warning lines
        for line in captured.getvalue().strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("Warning"):
                continue
            if "Write to" in line:
                add_msg("Writing output files…", "ok")
            elif line.startswith("Read") or line.startswith("Species") or line.startswith("Bootstrap"):
                add_msg(line, "info")

        add_msg("Compressing output files…", "info")
        job["progress"] = 90

        zip_path = os.path.join(tmp, "geneExpr_bootstrapped.zip")
        output_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".csv")])
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in output_files:
                zf.write(os.path.join(output_dir, fname), arcname=fname)

        job["output_zip"] = zip_path
        job["file_count"] = batch_count
        job["species_count"] = len(species_prefixes)
        job["progress"] = 100
        job["status"] = "done"
        add_msg(f"Done! {batch_count} files generated.", "ok")

    except Exception as e:
        import traceback
        job["status"] = "error"
        job["error"] = str(e)
        add_msg(f"Error: {e}", "err")
        print(traceback.format_exc(), flush=True)


@app.get("/api/progress/{job_id}")
async def progress_stream(job_id: str):
    async def event_stream():
        last_idx = 0
        while True:
            job = jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'status': 'error', 'error': 'Job not found'})}\n\n"
                break

            new_msgs = job["messages"][last_idx:]
            last_idx = len(job["messages"])

            for msg in new_msgs:
                payload = {
                    "message": msg["text"],
                    "type": msg["type"],
                    "progress": job["progress"],
                    "label": msg["text"],
                }
                yield f"data: {json.dumps(payload)}\n\n"

            if job["status"] == "done":
                yield f"data: {json.dumps({'status': 'done', 'progress': 100, 'file_count': job.get('file_count'), 'species_count': job.get('species_count')})}\n\n"
                break
            elif job["status"] == "error":
                yield f"data: {json.dumps({'status': 'error', 'error': job.get('error', 'Unknown error')})}\n\n"
                break

            if not new_msgs:
                yield f"data: {json.dumps({'progress': job['progress'], 'label': 'Processing…'})}\n\n"

            await asyncio.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    job = jobs.get(job_id, {})
    zip_path = job.get("output_zip")
    if not zip_path or not os.path.exists(zip_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(
        zip_path,
        filename="geneExpr_bootstrapped.zip",
        media_type="application/zip",
    )


# Serve frontend — must be last
app.mount("/", StaticFiles(directory="/app/frontend", html=True), name="frontend")
