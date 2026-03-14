#!/usr/bin/env python3
"""
Glutamate and GABA Gene Expression Analysis
============================================

Analyzes expression of glutamate and GABA metabolism genes across multiple
single-cell RNA-seq datasets from mouse and human brain.

Cell Type Strategy:
    - Neurons: Combined category (Glutamatergic + GABAergic neurons)
    - Astrocytes: Identified via subclass/supercluster annotations
    - Microglia: Identified via subclass/supercluster annotations

Datasets:
    1. Zeng Whole Mouse Brain (Allen Brain Atlas)
    2. Zeng Aging Mouse Brain (Allen Brain Atlas)
    3. Siletti Human Brain (Allen Brain Atlas)
    4. ASAP Human Brain (Parkinson's disease study)

Technical Implementation:
    - Memory-efficient batch processing with configurable chunk size
    - Parallel processing using multiprocessing (14 CPU cores)
    - Backed/memory-mapped h5ad file reading to minimize RAM usage

Output:
    - CSV file with expression statistics per cell type, condition, and gene
    - Metrics: n_cells, n_expressing, pct_expressing, mean_all, mean_expressing
    - NOTE: Allen Brain Atlas source data is in log2(CPM+1) units (counts per
      million, not raw counts). mean_all and mean_expressing are computed
      directly on these log2(CPM+1) values and should be interpreted accordingly.

Manuscript figures: Fig 2A, Fig 5G,I
"""

import os
import gc
import warnings
from typing import Dict, List, Optional, Tuple, Any
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

N_WORKERS = 14
CHUNK_SIZE = 100_000
BASE_DIR = "/Users/cjsogn/Glutamate_GABA_Expression_Analysis"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# =============================================================================
# GENE DEFINITIONS
# =============================================================================

MOUSE_GENE_MAP: Dict[str, str] = {
    # Glutamate metabolism
    'Gls': 'ENSMUSG00000026103',       # Glutaminase
    'Glul': 'ENSMUSG00000026473',      # Glutamine synthetase
    # Glutamate transporters
    'Slc1a3': 'ENSMUSG00000005360',    # EAAT1/GLAST
    'Slc1a2': 'ENSMUSG00000005089',    # EAAT2/GLT-1
    'Slc38a1': 'ENSMUSG00000023169',   # SNAT1
    # GABA metabolism
    'Gad1': 'ENSMUSG00000070880',      # GAD67
    'Gad2': 'ENSMUSG00000026787',      # GAD65
    'Abat': 'ENSMUSG00000057880',      # GABA transaminase
    # GABA transporters
    'Slc6a1': 'ENSMUSG00000030310',    # GAT1
    'Slc6a11': 'ENSMUSG00000030307',   # GAT3
}

HUMAN_GENE_MAP: Dict[str, str] = {
    # Glutamate metabolism
    'GLS': 'ENSG00000115419',
    'GLUL': 'ENSG00000135821',
    # Glutamate transporters
    'SLC1A3': 'ENSG00000079215',
    'SLC1A2': 'ENSG00000110436',
    'SLC38A1': 'ENSG00000111371',
    # GABA metabolism
    'GAD1': 'ENSG00000128683',
    'GAD2': 'ENSG00000136750',
    'ABAT': 'ENSG00000183044',
    # GABA transporters
    'SLC6A1': 'ENSG00000157103',
    'SLC6A11': 'ENSG00000132164',
}

GENE_GROUPS: Dict[str, List[str]] = {
    'Glutamate_Enzymes': ['Gls', 'Glul', 'GLS', 'GLUL'],
    'Glutamate_Transporters': ['Slc1a3', 'Slc1a2', 'Slc38a1', 'SLC1A3', 'SLC1A2', 'SLC38A1'],
    'GABA_Enzymes': ['Gad1', 'Gad2', 'Abat', 'GAD1', 'GAD2', 'ABAT'],
    'GABA_Transporters': ['Slc6a1', 'Slc6a11', 'SLC6A1', 'SLC6A11'],
}

GENE_DISPLAY_NAMES: Dict[str, str] = {
    'Gls': 'GLS', 'GLS': 'GLS',
    'Glul': 'GLUL', 'GLUL': 'GLUL',
    'Slc1a3': 'SLC1A3 (EAAT1)', 'SLC1A3': 'SLC1A3 (EAAT1)',
    'Slc1a2': 'SLC1A2 (EAAT2)', 'SLC1A2': 'SLC1A2 (EAAT2)',
    'Slc38a1': 'SLC38A1', 'SLC38A1': 'SLC38A1',
    'Gad1': 'GAD1', 'GAD1': 'GAD1',
    'Gad2': 'GAD2', 'GAD2': 'GAD2',
    'Abat': 'ABAT', 'ABAT': 'ABAT',
    'Slc6a1': 'SLC6A1 (GAT1)', 'SLC6A1': 'SLC6A1 (GAT1)',
    'Slc6a11': 'SLC6A11 (GAT3)', 'SLC6A11': 'SLC6A11 (GAT3)',
}

# Human neuron superclusters (Siletti/ASAP datasets)
HUMAN_NEURON_SUPERCLUSTERS: List[str] = [
    'CGE interneuron', 'MGE interneuron', 'LAMP5-LHX6 and Chandelier',
    'Midbrain-derived inhibitory', 'Cerebellar inhibitory',
    'Upper-layer intratelencephalic', 'Deep-layer intratelencephalic',
    'Deep-layer corticothalamic and 6b', 'Deep-layer near-projecting',
    'Hippocampal CA1-3', 'Hippocampal CA4', 'Hippocampal dentate gyrus',
    'Thalamic excitatory', 'Amygdala excitatory', 'Upper rhombic lip',
    'Lower rhombic lip', 'Mammillary body', 'Medium spiny neuron',
    'Eccentric medium spiny neuron',
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_gene_group(gene: str) -> str:
    """Return the functional group for a given gene."""
    for group, genes in GENE_GROUPS.items():
        if gene in genes:
            return group
    return 'Other'


def create_accumulator() -> Dict[str, float]:
    """Create a new expression accumulator dictionary."""
    return {
        'sum_expr': 0.0,
        'n_cells': 0,
        'n_expressing': 0,
        'sum_expressing': 0.0,
    }


def merge_accumulators(
    acc1: Dict[Tuple, Dict[str, float]],
    acc2: Dict[Tuple, Dict[str, float]]
) -> Dict[Tuple, Dict[str, float]]:
    """Merge two accumulator dictionaries."""
    merged = acc1.copy()
    for key, values in acc2.items():
        if key not in merged:
            merged[key] = create_accumulator()
        merged[key]['sum_expr'] += values['sum_expr']
        merged[key]['n_cells'] += values['n_cells']
        merged[key]['n_expressing'] += values['n_expressing']
        merged[key]['sum_expressing'] += values['sum_expressing']
    return merged


def compute_expression_stats(
    expr: np.ndarray,
    accumulator: Dict[str, float]
) -> Dict[str, float]:
    """
    Update accumulator with expression statistics from an array.

    Parameters
    ----------
    expr : np.ndarray
        Expression values for cells
    accumulator : dict
        Dictionary to accumulate statistics into

    Returns
    -------
    dict
        Updated accumulator
    """
    expressing_mask = expr > 0
    accumulator['sum_expr'] += np.sum(expr)
    accumulator['n_cells'] += len(expr)
    accumulator['n_expressing'] += np.sum(expressing_mask)
    accumulator['sum_expressing'] += np.sum(expr[expressing_mask])
    return accumulator


def accumulator_to_results(
    accumulators: Dict[Tuple, Dict[str, float]],
    dataset: str,
    has_condition: bool = False
) -> List[Dict[str, Any]]:
    """
    Convert accumulators to result dictionaries.

    Parameters
    ----------
    accumulators : dict
        Dictionary mapping (cell_type, [condition,] gene) to statistics
    dataset : str
        Dataset name for output
    has_condition : bool
        Whether accumulators include condition in key

    Returns
    -------
    list
        List of result dictionaries
    """
    results = []
    for key, acc in accumulators.items():
        if has_condition:
            cell_type, condition, gene = key
        else:
            cell_type, gene = key
            condition = 'All'

        n_cells = acc['n_cells']
        n_expressing = acc['n_expressing']

        results.append({
            'dataset': dataset,
            'cell_type': cell_type,
            'condition': condition,
            'gene': gene,
            'gene_display': GENE_DISPLAY_NAMES.get(gene, gene),
            'gene_group': get_gene_group(gene),
            'n_cells': n_cells,
            'n_expressing': n_expressing,
            'pct_expressing': (n_expressing / n_cells * 100) if n_cells > 0 else 0,
            'mean_all': acc['sum_expr'] / n_cells if n_cells > 0 else 0,
            'mean_expressing': acc['sum_expressing'] / n_expressing if n_expressing > 0 else 0,
        })
    return results


def get_gene_indices(
    var_names: pd.Index,
    gene_map: Dict[str, str]
) -> Dict[str, int]:
    """
    Get column indices for genes of interest.

    Parameters
    ----------
    var_names : pd.Index
        Variable names from AnnData object
    gene_map : dict
        Mapping of gene symbols to Ensembl IDs

    Returns
    -------
    dict
        Mapping of gene symbols to column indices
    """
    var_names_list = list(var_names)
    indices = {}
    for gene_symbol, ensembl_id in gene_map.items():
        if ensembl_id in var_names:
            indices[gene_symbol] = var_names_list.index(ensembl_id)
    return indices


# =============================================================================
# CELL TYPE CLASSIFICATION FUNCTIONS
# =============================================================================

def classify_zeng_mouse_cell(row: pd.Series) -> Optional[str]:
    """
    Classify cell type for Zeng whole mouse brain dataset.

    Neurons (Glutamatergic and GABAergic) are combined into single category.
    """
    subclass = str(row.get('subclass', ''))
    cell_class = str(row.get('class', ''))

    if '334 Microglia' in subclass:
        return 'Microglia'
    elif 'Astro' in subclass and 'Astro-Epen' in cell_class:
        return 'Astrocytes'
    elif 'Glut' in cell_class or 'GABA' in cell_class:
        return 'Neurons'
    return None


def classify_zeng_aging_cell(row: pd.Series) -> Optional[str]:
    """
    Classify cell type for Zeng aging mouse brain dataset.

    Neurons (Glutamatergic and GABAergic) are combined into single category.
    """
    subclass = str(row['subclass_name'])
    cell_class = str(row['class_name'])

    if '334 Microglia' in subclass or 'Microglia' in subclass:
        return 'Microglia'
    elif 'Astro' in subclass and 'Astro-Epen' in cell_class:
        return 'Astrocytes'
    elif 'Glut' in cell_class or 'GABA' in cell_class:
        return 'Neurons'
    return None


def classify_human_cell(supercluster: str) -> Optional[str]:
    """
    Classify cell type for human datasets (Siletti/ASAP).

    Neurons are identified by membership in neuron supercluster list.
    """
    if pd.isna(supercluster):
        return None
    if supercluster == 'Microglia':
        return 'Microglia'
    elif supercluster == 'Astrocyte':
        return 'Astrocytes'
    elif supercluster in HUMAN_NEURON_SUPERCLUSTERS:
        return 'Neurons'
    return None


# =============================================================================
# PARALLEL PROCESSING WORKER FUNCTIONS
# =============================================================================

def process_zeng_mouse_region(
    args: Tuple[str, str, pd.DataFrame, Dict[str, str]]
) -> Dict[Tuple, Dict[str, float]]:
    """
    Process a single brain region file for Zeng mouse dataset.

    Worker function for parallel processing.

    Parameters
    ----------
    args : tuple
        (region_path, region_name, metadata, gene_map)

    Returns
    -------
    dict
        Accumulators for this region
    """
    region_path, region_name, meta, gene_map = args

    if not os.path.exists(region_path):
        return {}

    accumulators = {}

    try:
        adata = sc.read_h5ad(region_path, backed='r')
        cell_labels = adata.obs.index.tolist()
        gene_indices = get_gene_indices(adata.var_names, gene_map)

        if len(gene_indices) == 0:
            adata.file.close()
            return {}

        n_cells = adata.n_obs

        for chunk_start in range(0, n_cells, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, n_cells)
            chunk_cells = cell_labels[chunk_start:chunk_end]
            chunk_meta = meta.loc[meta.index.isin(chunk_cells)]

            if len(chunk_meta) == 0:
                continue

            X_chunk = adata.X[chunk_start:chunk_end, :]
            if hasattr(X_chunk, 'toarray'):
                X_chunk = X_chunk.toarray()

            chunk_cell_to_idx = {c: i for i, c in enumerate(chunk_cells)}

            for cell_type in chunk_meta['cell_type'].unique():
                matching_cells = chunk_meta[chunk_meta['cell_type'] == cell_type].index.tolist()
                cell_indices = [chunk_cell_to_idx[c] for c in matching_cells if c in chunk_cell_to_idx]

                if len(cell_indices) == 0:
                    continue

                for gene_symbol, gene_idx in gene_indices.items():
                    expr = X_chunk[cell_indices, gene_idx]
                    key = (cell_type, gene_symbol)

                    if key not in accumulators:
                        accumulators[key] = create_accumulator()
                    compute_expression_stats(expr, accumulators[key])

            del X_chunk
            gc.collect()

        adata.file.close()

    except Exception as e:
        print(f"  Error processing {region_name}: {e}")
        return {}

    return accumulators


# =============================================================================
# DATASET ANALYSIS FUNCTIONS
# =============================================================================

def analyze_zeng_mouse() -> pd.DataFrame:
    """
    Analyze Zeng whole mouse brain dataset.

    Processes multiple brain region files in parallel.
    Combines Glutamatergic and GABAergic neurons into single 'Neurons' category.

    Returns
    -------
    pd.DataFrame
        Expression statistics per cell type and gene
    """
    print("\n" + "=" * 80)
    print("ANALYZING: Zeng Whole Mouse Brain (Neurons Combined)")
    print("=" * 80)

    data_dir = "/Users/cjsogn/Documents/data/Whole mouse brain RNA Zeng"
    metadata_path = os.path.join(data_dir, "cell_metadata_with_cluster_annotation.csv")

    region_files = [
        "WMB-10Xv3-CB-log2.h5ad", "WMB-10Xv3-CTXsp-log2.h5ad",
        "WMB-10Xv3-HPF-log2.h5ad", "WMB-10Xv3-HY-log2.h5ad",
        "WMB-10Xv3-Isocortex-1-log2.h5ad", "WMB-10Xv3-Isocortex-2-log2.h5ad",
        "WMB-10Xv3-MB-log2.h5ad", "WMB-10Xv3-MY-log2.h5ad",
        "WMB-10Xv3-OLF-log2.h5ad", "WMB-10Xv3-P-log2.h5ad",
        "WMB-10Xv3-PAL-log2.h5ad", "WMB-10Xv3-STR-log2.h5ad",
        "WMB-10Xv3-TH-log2.h5ad",
    ]

    # Load and prepare metadata
    print("Loading metadata...")
    meta = pd.read_csv(metadata_path, usecols=['cell_label', 'class', 'subclass'])
    meta['cell_type'] = meta.apply(classify_zeng_mouse_cell, axis=1)
    meta = meta.dropna(subset=['cell_type'])
    meta = meta.set_index('cell_label')

    print(f"Cell type distribution:")
    print(meta['cell_type'].value_counts())

    # Prepare arguments for parallel processing
    worker_args = [
        (
            os.path.join(data_dir, f),
            f.replace('WMB-10Xv3-', '').replace('-log2.h5ad', ''),
            meta,
            MOUSE_GENE_MAP
        )
        for f in region_files
    ]

    # Process regions in parallel
    print(f"\nProcessing {len(region_files)} brain regions using {N_WORKERS} workers...")

    with Pool(processes=N_WORKERS) as pool:
        region_results = pool.map(process_zeng_mouse_region, worker_args)

    # Merge results from all regions
    accumulators = {}
    for region_acc in region_results:
        accumulators = merge_accumulators(accumulators, region_acc)

    results = accumulator_to_results(accumulators, 'Zeng_Mouse', has_condition=False)
    df = pd.DataFrame(results)

    print(f"\nCell counts: {df.groupby('cell_type')['n_cells'].first().to_dict()}")
    return df


def analyze_zeng_aging() -> pd.DataFrame:
    """
    Analyze Zeng aging mouse brain dataset.

    Compares gene expression between adult and aged mice.
    Combines Glutamatergic and GABAergic neurons into single 'Neurons' category.

    Returns
    -------
    pd.DataFrame
        Expression statistics per cell type, condition, and gene
    """
    print("\n" + "=" * 80)
    print("ANALYZING: Zeng Aging Mouse Brain (Neurons Combined)")
    print("=" * 80)

    adata_path = "/Users/cjsogn/Documents/data/Whole aging mouse brain RNA Zeng/Zeng-Aging-Mouse-10Xv3-log2.h5ad"

    # Load and merge metadata
    print("Loading metadata...")
    cluster_mapping = pd.read_csv(
        "/Users/cjsogn/Documents/data/Whole aging mouse brain RNA Zeng/cell_cluster_mapping_annotations.csv",
        usecols=['cell_label', 'class_name', 'subclass_name']
    )
    cell_meta = pd.read_csv(
        "/Users/cjsogn/Documents/data/Whole aging mouse brain RNA Zeng/cell_metadata.csv",
        usecols=['cell_label', 'donor_age_category']
    )

    meta = cluster_mapping.merge(cell_meta, on='cell_label', how='inner')
    meta['cell_type'] = meta.apply(classify_zeng_aging_cell, axis=1)
    meta['condition'] = meta['donor_age_category'].map({'adult': 'Control', 'aged': 'Aged'})
    meta = meta.dropna(subset=['cell_type', 'condition'])
    meta = meta.set_index('cell_label')

    print(f"Cell type distribution:")
    print(meta['cell_type'].value_counts())
    print(f"\nCondition distribution:")
    print(pd.crosstab(meta['cell_type'], meta['condition']))

    # Load expression data
    adata = sc.read_h5ad(adata_path, backed='r')
    print(f"Shape: {adata.shape}")

    gene_indices = get_gene_indices(adata.var_names, MOUSE_GENE_MAP)

    # Process in chunks
    accumulators = {}
    n_cells = adata.n_obs
    cell_labels = adata.obs.index.tolist()

    for chunk_start in range(0, n_cells, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_cells)
        if chunk_start % 200_000 == 0:
            print(f"  Processing cells {chunk_start:,} to {chunk_end:,}...")

        chunk_cells = cell_labels[chunk_start:chunk_end]
        chunk_meta = meta.loc[meta.index.isin(chunk_cells)]

        if len(chunk_meta) == 0:
            continue

        X_chunk = adata.X[chunk_start:chunk_end, :]
        if hasattr(X_chunk, 'toarray'):
            X_chunk = X_chunk.toarray()

        chunk_cell_to_idx = {c: i for i, c in enumerate(chunk_cells)}

        for cell_type in chunk_meta['cell_type'].unique():
            for condition in chunk_meta['condition'].unique():
                mask = (chunk_meta['cell_type'] == cell_type) & (chunk_meta['condition'] == condition)
                matching_cells = chunk_meta[mask].index.tolist()
                cell_indices = [chunk_cell_to_idx[c] for c in matching_cells if c in chunk_cell_to_idx]

                if len(cell_indices) == 0:
                    continue

                for gene_symbol, gene_idx in gene_indices.items():
                    expr = X_chunk[cell_indices, gene_idx]
                    key = (cell_type, condition, gene_symbol)

                    if key not in accumulators:
                        accumulators[key] = create_accumulator()
                    compute_expression_stats(expr, accumulators[key])

        del X_chunk
        gc.collect()

    adata.file.close()

    results = accumulator_to_results(accumulators, 'Zeng_Aging', has_condition=True)
    return pd.DataFrame(results)


def analyze_siletti() -> pd.DataFrame:
    """
    Analyze Siletti human brain dataset.

    Processes separate neuron and non-neuron files.
    Combines all neuron superclusters into single 'Neurons' category.

    Returns
    -------
    pd.DataFrame
        Expression statistics per cell type and gene
    """
    print("\n" + "=" * 80)
    print("ANALYZING: Siletti Human Brain (Neurons Combined)")
    print("=" * 80)

    data_dir = "/Users/cjsogn/Documents/data/Whole human RNA Siletti"

    # Load and prepare metadata
    print("Loading metadata...")
    meta = pd.read_csv(
        os.path.join(data_dir, "cell_metadata.csv"),
        usecols=['cell_label', 'cluster_alias']
    )

    cluster_mapping = pd.read_csv(os.path.join(data_dir, "cluster_to_cluster_annotation_membership.csv"))
    supercluster = cluster_mapping[cluster_mapping['cluster_annotation_term_set_name'] == 'supercluster']
    cluster_to_supercluster = dict(zip(supercluster['cluster_alias'], supercluster['cluster_annotation_term_name']))

    meta['supercluster'] = meta['cluster_alias'].map(cluster_to_supercluster)
    meta['cell_type'] = meta['supercluster'].apply(classify_human_cell)
    meta = meta.dropna(subset=['cell_type'])
    meta = meta.set_index('cell_label')

    print(f"Cell type distribution:")
    print(meta['cell_type'].value_counts())

    accumulators = {}

    # Process Neurons file
    print("\nProcessing Neurons file...")
    adata = sc.read_h5ad(os.path.join(data_dir, "WHB-10Xv3-Neurons-log2.h5ad"), backed='r')
    print(f"Shape: {adata.shape}")

    gene_indices = get_gene_indices(adata.var_names, HUMAN_GENE_MAP)
    n_cells = adata.n_obs
    cell_labels = adata.obs.index.tolist()

    for chunk_start in range(0, n_cells, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_cells)
        if chunk_start % 500_000 == 0:
            print(f"  Processing neurons {chunk_start:,} to {chunk_end:,}...")

        chunk_cells = cell_labels[chunk_start:chunk_end]
        chunk_meta = meta.loc[meta.index.isin(chunk_cells)]

        if len(chunk_meta) == 0:
            continue

        X_chunk = adata.X[chunk_start:chunk_end, :]
        if hasattr(X_chunk, 'toarray'):
            X_chunk = X_chunk.toarray()

        chunk_cell_to_idx = {c: i for i, c in enumerate(chunk_cells)}

        # Only process neurons from neurons file
        matching_cells = chunk_meta[chunk_meta['cell_type'] == 'Neurons'].index.tolist()
        cell_indices = [chunk_cell_to_idx[c] for c in matching_cells if c in chunk_cell_to_idx]

        if len(cell_indices) > 0:
            for gene_symbol, gene_idx in gene_indices.items():
                expr = X_chunk[cell_indices, gene_idx]
                key = ('Neurons', gene_symbol)

                if key not in accumulators:
                    accumulators[key] = create_accumulator()
                compute_expression_stats(expr, accumulators[key])

        del X_chunk
        gc.collect()

    adata.file.close()

    # Process Non-neurons file
    print("\nProcessing Non-neurons file...")
    adata = sc.read_h5ad(os.path.join(data_dir, "WHB-10Xv3-Nonneurons-log2.h5ad"), backed='r')
    print(f"Shape: {adata.shape}")

    gene_indices = get_gene_indices(adata.var_names, HUMAN_GENE_MAP)
    n_cells = adata.n_obs
    cell_labels = adata.obs.index.tolist()

    for chunk_start in range(0, n_cells, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_cells)
        if chunk_start % 200_000 == 0:
            print(f"  Processing non-neurons {chunk_start:,} to {chunk_end:,}...")

        chunk_cells = cell_labels[chunk_start:chunk_end]
        chunk_meta = meta.loc[meta.index.isin(chunk_cells)]

        if len(chunk_meta) == 0:
            continue

        X_chunk = adata.X[chunk_start:chunk_end, :]
        if hasattr(X_chunk, 'toarray'):
            X_chunk = X_chunk.toarray()

        chunk_cell_to_idx = {c: i for i, c in enumerate(chunk_cells)}

        for cell_type in ['Microglia', 'Astrocytes']:
            matching_cells = chunk_meta[chunk_meta['cell_type'] == cell_type].index.tolist()
            cell_indices = [chunk_cell_to_idx[c] for c in matching_cells if c in chunk_cell_to_idx]

            if len(cell_indices) == 0:
                continue

            for gene_symbol, gene_idx in gene_indices.items():
                expr = X_chunk[cell_indices, gene_idx]
                key = (cell_type, gene_symbol)

                if key not in accumulators:
                    accumulators[key] = create_accumulator()
                compute_expression_stats(expr, accumulators[key])

        del X_chunk
        gc.collect()

    adata.file.close()

    results = accumulator_to_results(accumulators, 'Siletti_Human', has_condition=False)
    return pd.DataFrame(results)


def analyze_asap() -> pd.DataFrame:
    """
    Analyze ASAP human brain dataset (Parkinson's disease study).

    Compares gene expression between control and Parkinson's disease patients.
    Combines all neuron superclusters into single 'Neurons' category.

    Returns
    -------
    pd.DataFrame
        Expression statistics per cell type, condition, and gene
    """
    print("\n" + "=" * 80)
    print("ANALYZING: ASAP Human Brain (Neurons Combined)")
    print("=" * 80)

    data_dir = "/Users/cjsogn/Documents/data/ASAP"

    # Load and merge metadata
    print("Loading metadata...")
    cell_meta = pd.read_csv(os.path.join(data_dir, "cell_metadata.csv"))
    sample_info = pd.read_csv(os.path.join(data_dir, "sample.csv"), usecols=['sample_label', 'donor_label'])
    donor_info = pd.read_csv(os.path.join(data_dir, "donor.csv"), usecols=['donor_label', 'primary_diagnosis'])
    mmc = pd.read_csv(os.path.join(data_dir, "mmc_results_siletti_whb.csv"), usecols=['cell_label', 'supercluster_name'])

    cell_meta = cell_meta.merge(sample_info, on='sample_label', how='left')
    cell_meta = cell_meta.merge(donor_info, on='donor_label', how='left')
    cell_meta = cell_meta.merge(mmc, on='cell_label', how='left')

    # Classify conditions
    control_diagnoses = ['Healthy Control', 'No PD nor other neurological disorder']
    pd_diagnoses = ["Parkinson's disease", "Idiopathic Parkinson's disease", "Prodromal motor Parkinson's disease"]

    cell_meta['condition'] = 'Other'
    cell_meta.loc[cell_meta['primary_diagnosis'].isin(control_diagnoses), 'condition'] = 'Control'
    cell_meta.loc[cell_meta['primary_diagnosis'].isin(pd_diagnoses), 'condition'] = 'PD'

    # Classify cell types
    cell_meta['cell_type'] = cell_meta['supercluster_name'].apply(classify_human_cell)
    cell_meta = cell_meta[cell_meta['condition'].isin(['Control', 'PD'])]
    cell_meta = cell_meta.dropna(subset=['cell_type'])
    cell_meta = cell_meta.set_index('cell_label')

    print(f"Cell type distribution:")
    print(cell_meta['cell_type'].value_counts())
    print(f"\nCross-tabulation:")
    print(pd.crosstab(cell_meta['cell_type'], cell_meta['condition']))

    # Load expression data
    adata = sc.read_h5ad(os.path.join(data_dir, "ASAP-PMDBS-10X-log2.h5ad"), backed='r')
    print(f"Shape: {adata.shape}")

    gene_indices = get_gene_indices(adata.var_names, HUMAN_GENE_MAP)

    # Process in chunks
    accumulators = {}
    n_cells = adata.n_obs
    cell_labels = adata.obs.index.tolist()

    for chunk_start in range(0, n_cells, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_cells)
        if chunk_start % 500_000 == 0:
            print(f"  Processing cells {chunk_start:,} to {chunk_end:,}...")

        chunk_cells = cell_labels[chunk_start:chunk_end]
        chunk_meta = cell_meta.loc[cell_meta.index.isin(chunk_cells)]

        if len(chunk_meta) == 0:
            continue

        X_chunk = adata.X[chunk_start:chunk_end, :]
        if hasattr(X_chunk, 'toarray'):
            X_chunk = X_chunk.toarray()

        chunk_cell_to_idx = {c: i for i, c in enumerate(chunk_cells)}

        for cell_type in chunk_meta['cell_type'].unique():
            for condition in ['Control', 'PD']:
                mask = (chunk_meta['cell_type'] == cell_type) & (chunk_meta['condition'] == condition)
                matching_cells = chunk_meta[mask].index.tolist()
                cell_indices = [chunk_cell_to_idx[c] for c in matching_cells if c in chunk_cell_to_idx]

                if len(cell_indices) == 0:
                    continue

                for gene_symbol, gene_idx in gene_indices.items():
                    expr = X_chunk[cell_indices, gene_idx]
                    key = (cell_type, condition, gene_symbol)

                    if key not in accumulators:
                        accumulators[key] = create_accumulator()
                    compute_expression_stats(expr, accumulators[key])

        del X_chunk
        gc.collect()

    adata.file.close()

    results = accumulator_to_results(accumulators, 'ASAP_Human', has_condition=True)
    return pd.DataFrame(results)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Run complete expression analysis pipeline.

    Analyzes all four datasets and combines results into single output file.
    """
    print("=" * 80)
    print("GLUTAMATE AND GABA GENE EXPRESSION ANALYSIS")
    print("Neurons Combined (Glutamatergic + GABAergic)")
    print(f"Using {N_WORKERS} CPU cores for parallel processing")
    print("=" * 80)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {}

    # Analyze each dataset
    print("\n[1/4] Zeng Whole Mouse Brain...")
    results['zeng_mouse'] = analyze_zeng_mouse()

    print("\n[2/4] Zeng Aging Mouse Brain...")
    results['zeng_aging'] = analyze_zeng_aging()

    print("\n[3/4] Siletti Human Brain...")
    results['siletti'] = analyze_siletti()

    print("\n[4/4] ASAP Human Brain...")
    results['asap'] = analyze_asap()

    # Combine and save results
    print("\n" + "=" * 80)
    print("COMBINING RESULTS")
    print("=" * 80)

    all_dfs = [df for df in results.values() if df is not None and len(df) > 0]

    if len(all_dfs) > 0:
        combined = pd.concat(all_dfs, ignore_index=True)
        output_file = os.path.join(RESULTS_DIR, "expression_results.csv")
        combined.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
        print(f"Total rows: {len(combined)}")

        # Print summary
        print("\n--- Cell Counts Summary ---")
        for dataset in combined['dataset'].unique():
            print(f"\n{dataset}:")
            sub = combined[combined['dataset'] == dataset]
            for condition in sub['condition'].unique():
                print(f"  {condition}:")
                sub2 = sub[sub['condition'] == condition]
                for ct in sub2['cell_type'].unique():
                    n = sub2[sub2['cell_type'] == ct]['n_cells'].iloc[0]
                    print(f"    {ct}: {n:,}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
