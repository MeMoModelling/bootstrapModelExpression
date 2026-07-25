import argparse
import ast
import libsbml
import os
import re
import numpy as np
import pandas as pd
import sys
from utils.utils import *

def extract_genes_from_gpa(gpa_str):
    """Extract all gene IDs from a gpaAssociation string (e.g. 'G_A or (G_B and G_C)')."""
    if pd.isna(gpa_str) or str(gpa_str).strip() == "":
        return []
    tokens = re.findall(r'[A-Za-z_]\w*', str(gpa_str))
    keywords = {'or', 'and', 'not'}
    return [t for t in tokens if t.lower() not in keywords]

def parse_genes(gene_value):
    """Parse either a stringified gene list or a boolean gene expression."""
    if pd.isna(gene_value) or str(gene_value).strip() == "":
        return []

    if isinstance(gene_value, (list, tuple, set)):
        return list(gene_value)

    try:
        parsed = ast.literal_eval(str(gene_value))
    except (ValueError, SyntaxError):
        return extract_genes_from_gpa(gene_value)

    if isinstance(parsed, (list, tuple, set)):
        return list(parsed)
    return extract_genes_from_gpa(parsed)

def read_combined_geneExpr(combined_geneExpr_filename):
    if not os.path.isfile(combined_geneExpr_filename):
        raise FileNotFoundError(f"Missing combined normalized count file, expected at {combined_geneExpr_filename}")
    combined_geneExpr_df = pd.read_csv(combined_geneExpr_filename, index_col=0)
    return combined_geneExpr_df

def is_real_gene_with_mapping(gene):
    return not gene.startswith(("unmapped", "unknown", "Spontaneous", "Exchange", "Sink", "Diffusion"))

def read_system_gene(model_pre_filename):
    rxn_df = read_model_excel(model_pre_filename, "Reactions")

    # Support both stringified lists and boolean expressions in 'genes'.
    if "genes" in rxn_df.columns:
        gene_col = "genes"
    elif "gpaAssociation" in rxn_df.columns:
        gene_col = "gpaAssociation"
    else:
        raise ValueError(
            "Reactions sheet must have a 'genes' or 'gpaAssociation' column — "
            f"columns found: {list(rxn_df.columns)}"
        )

    system_genes_dict = {}
    missing_gene_system_dict = {}
    all_genes_set = set()
    for system, gene_list_raw in zip(rxn_df["system"], rxn_df[gene_col]):
        gene_list = parse_genes(gene_list_raw)

        if system != "" and system not in system_genes_dict:
            system_genes_dict[system] = set()

        for gene in gene_list:
            if system != "" and is_real_gene_with_mapping(gene):
                system_genes_dict[system].add(gene)
            if gene.startswith(("unmapped", "unknown")):
                missing_gene_system_dict[gene] = system
            if is_real_gene_with_mapping(gene):
                all_genes_set.add(gene)

    missing_gene_system_dict = dict(sorted(missing_gene_system_dict.items()))
    return system_genes_dict, missing_gene_system_dict, all_genes_set

def map_genes(mapping_dict, system_genes_dict, missing_gene_system_dict, all_genes_set):
    """Map model gene tags to expression IDs and retain unmapped tags for bootstrapping."""
    mapped_system_genes = {}
    missing_genes = dict(missing_gene_system_dict)

    for system, gene_set in system_genes_dict.items():
        mapped_genes = set()
        for gene in gene_set:
            if gene in mapping_dict:
                mapped_genes.add(mapping_dict[gene])
            else:
                missing_genes.setdefault(gene, system)
        if mapped_genes:
            mapped_system_genes[system] = mapped_genes

    mapped_all_genes = {mapping_dict[gene] for gene in all_genes_set if gene in mapping_dict}
    return mapped_system_genes, dict(sorted(missing_genes.items())), mapped_all_genes

def filter_geneExpr_df(combined_geneExpr_df, all_genes_set):
    geneExpr_df = combined_geneExpr_df.loc[combined_geneExpr_df.index.isin(all_genes_set)]
    return geneExpr_df

def get_system_gene_counts(system_genes_dict, gene_count_dict):
    system_gene_counts_dict = {}
    missing_genes = set()
    for system, gene_set in system_genes_dict.items():
        gene_counts = []
        for gene in sorted(gene_set):
            if gene in gene_count_dict:
                gene_counts.append(gene_count_dict[gene])
            else:
                missing_genes.add(gene)
        system_gene_counts_dict[system] = gene_counts
    if missing_genes:
        print("Warning   : These genes do not have normalized count values - {}".format(", ".join(missing_genes)))
    return system_gene_counts_dict

def initialize_df(geneExpr_sample, columns):
    new_geneExpr_df_sample = pd.concat([geneExpr_sample.rename(col) for col in columns], axis=1)
    return new_geneExpr_df_sample

def bootstrap_missing_genes(missing_gene_system_dict, system_gene_counts_dict, columns):
    rng = np.random.default_rng(seed=0)
    sample_pool_all_genes = [gc for gene_counts in system_gene_counts_dict.values() for gc in gene_counts]
    new_geneExpr_df_sample_missing = pd.DataFrame(columns=columns)
    for missing_gene, system in missing_gene_system_dict.items():
        if system in system_gene_counts_dict:
            sample_pool = system_gene_counts_dict[system]
            new_geneExpr_df_sample_missing.loc[missing_gene] = rng.choice(sample_pool, len(columns))
        else:
            new_geneExpr_df_sample_missing.loc[missing_gene] = rng.choice(sample_pool_all_genes, len(columns))
    return new_geneExpr_df_sample_missing

def filter_by_batch(new_geneExpr_df, start, end, columns):
    target_columns = [f"{col}_{i}" for col in columns for i in range(start, end)]
    new_geneExpr_df_batch_df = new_geneExpr_df[target_columns]
    return new_geneExpr_df_batch_df

def bootstrap_genes(model_pre_filenames, mapping_filenames, species_prefixes, combined_geneExpr_filename, geneExpr_folder, batch_count=1000):
    print("Read models from", ", ".join(model_pre_filenames))
    print("Read mapping tables from", ", ".join(mapping_filenames))
    print("Species prefixes:", ", ".join(species_prefixes))
    print("Read gene normalized counts from", combined_geneExpr_filename)
    os.makedirs(geneExpr_folder, exist_ok=True)

    total_to_sample = batch_count

    # column: sample_name
    # row: real genes for all species
    combined_geneExpr_df = read_combined_geneExpr(combined_geneExpr_filename)

    # column: A_1, A_2, ..., B_1, B_2, ...
    # row: (real genes with mapping + unmapped real genes + unknown genes) + Exchange + Sink for all species
    new_geneExpr_df = pd.DataFrame()
    
    for model_pre_filename, mapping_filename, species in zip(model_pre_filenames, mapping_filenames, species_prefixes): # loop by species
        print(f"Bootstrapping genes for {species}...")
        # mapping_dict: {model_tag in the model file: gene_id in the geneExpr file}
        mapping_dict = read_mapping(mapping_filename)
        
        # system_genes_dict: {system: set of real genes with mapping with that system}
        # missing_gene_system_dict: {unmapped real gene & unknown gene: system of the gene}
        # all_genes_set: set of all the real genes with mapping used in the model of that species
        system_genes_dict, missing_gene_system_dict, all_genes_set = read_system_gene(model_pre_filename)

        # Map model tags to the annotation IDs used in the expression file.
        system_genes_dict, missing_gene_system_dict, all_genes_set = map_genes(
            mapping_dict, system_genes_dict, missing_gene_system_dict, all_genes_set
        )

        # filter to get only geneExprs for that species
        geneExpr_df = filter_geneExpr_df(combined_geneExpr_df, all_genes_set)

        # column: A_1, A_2, ..., B_1, B_2, ...
        # row: (real genes with mapping + unmapped real genes + unknown genes) for one species
        new_geneExpr_df_species = pd.DataFrame()
        for sample in geneExpr_df.columns:
            # gene_count_dict: {gene: gene count}
            gene_count_dict = geneExpr_df[sample].to_dict()
            # system_gene_counts_dict: {system: list of gene counts of the genes with that system}
            system_gene_counts_dict = get_system_gene_counts(system_genes_dict, gene_count_dict)

            # column: (one sample) A_1, A_2, ..
            columns = [f"{sample}_{i+1}" for i in range(total_to_sample)]
            # row: real genes with mapping (same as geneExpr file)
            new_geneExpr_df_sample = initialize_df(geneExpr_df[sample], columns)
            # row: unmapped real genes + unknown genes 
            # (randomly bootstrap from the gene counts of the same species and the same sample and the same system)
            # (if the system of the missing gene do not have gene counts, randomly bootstrap from the gene counts of the same species and the same sample)
            new_geneExpr_df_sample_missing = bootstrap_missing_genes(missing_gene_system_dict, system_gene_counts_dict, columns)
            # concat by rows: real genes with mapping + unmapped real genes + unknown genes
            new_geneExpr_df_sample = pd.concat([new_geneExpr_df_sample, new_geneExpr_df_sample_missing], axis=0)

            # concat by columns: A_1, A_2, ..., B_1, B_2, ...
            new_geneExpr_df_species = pd.concat([new_geneExpr_df_species, new_geneExpr_df_sample], axis=1)
            
        # concat by rows: species_1 + species_2, ...
        new_geneExpr_df = pd.concat([new_geneExpr_df, new_geneExpr_df_species], axis=0)

    new_geneExpr_df.loc["Spontaneous"] = 0
    new_geneExpr_df.loc["Exchange"] = 0
    new_geneExpr_df.loc["Sink"] = 0
    new_geneExpr_df.loc["Growth"] = 0
    new_geneExpr_df.loc["Diffusion"] = 0

    # separate into batch to different files
    print("Writing to files...")
    new_geneExpr_df_batch_filename = os.path.join(geneExpr_folder, "geneExpr")
    for i in range(batch_count):
        # first file: A_1, B_1, ..
        # second file: A_2, B_2, 
        new_geneExpr_df_batch_df = filter_by_batch(new_geneExpr_df, i+1, (i+1)+1, combined_geneExpr_df.columns)
        new_geneExpr_df_batch_df.to_csv(new_geneExpr_df_batch_filename + f"_{i+1}.csv")    
    print(f"Write to {new_geneExpr_df_batch_filename}_<1_{batch_count}>.csv")

if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pre_filenames", nargs="+", required=True, help="List of parsed model files from construct_modularized_community_model tool")
    parser.add_argument("--mapping_filenames", nargs="+", required=True, help="List of mapping files to map gene in model to gene in annotation, from identifiers_mapping tool")
    parser.add_argument("--species_prefixes", nargs="+", required=True, help="List of species prefixes for the models, in the same order as input files")
    parser.add_argument("--combined_geneExpr_filename", required=True, help="The gene expression values for all species and for all samples")
    parser.add_argument("--geneExpr_folder", required=True, help="Folder containing gene expression files (geneExpr_<1-batch_count>.csv) with bootstrapped values added for unmapped and unknown genes")
    parser.add_argument("--batch_count", type=int, default=1000, help="Number of batch to bootstrap, write each batch to a file")
    args = parser.parse_args()

    # read arguments
    model_pre_filenames = args.model_pre_filenames
    mapping_filenames = args.mapping_filenames
    species_prefixes = args.species_prefixes
    combined_geneExpr_filename = args.combined_geneExpr_filename
    geneExpr_folder = args.geneExpr_folder
    batch_count = args.batch_count
    
    bootstrap_genes(model_pre_filenames, mapping_filenames, species_prefixes, combined_geneExpr_filename, geneExpr_folder, batch_count)
