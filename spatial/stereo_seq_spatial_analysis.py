#!/usr/bin/env python3
"""
Microglia Glutamate/GABA Transporter Expression Analysis

Analyzes microglial expression of glutamate and GABA transporters/enzymes
in relation to spatial proximity to glutamatergic and GABAergic neurons
using Han et al. Stereo-seq mouse brain atlas data.

Key questions addressed:
1. What is the baseline expression of Glu/GABA transporters in microglia vs astrocytes?
2. Is microglial expression correlated with proximity to GABAergic vs Glutamatergic neurons?
3. Is there regional enrichment of these genes across brain sections?

Manuscript figures: Fig 3 (Stereo-seq proximity to neuron types, fold enrichment)
"""

import argparse
import gzip
import os
import pickle
import sys
from collections import defaultdict
from glob import glob
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Gene definitions with common names and gene symbols
GLUTAMATE_GENES = {
    'EAAT1': 'Slc1a3',   # Excitatory amino acid transporter 1
    'EAAT2': 'Slc1a2',   # Excitatory amino acid transporter 2
    'GLS': 'Gls',        # Glutaminase
    'GLUL': 'Glul',      # Glutamine synthetase
    'SNAT1': 'Slc38a1',  # Sodium-coupled neutral amino acid transporter 1
}

GABA_GENES = {
    'GAT1': 'Slc6a1',    # GABA transporter 1
    'GAT3': 'Slc6a11',   # GABA transporter 3
    'GAD1': 'Gad1',      # Glutamate decarboxylase 1 (GAD67)
    'GAD2': 'Gad2',      # Glutamate decarboxylase 2 (GAD65)
    'ABAT': 'Abat',      # 4-aminobutyrate aminotransferase
}

ALL_TARGET_GENES = set(GLUTAMATE_GENES.values()) | set(GABA_GENES.values())

# Neuron classification patterns (based on cluster naming conventions)
NEURON_PATTERNS = {
    'Glutamatergic': '_GLU_',
    'GABAergic': '_GABA_',
}

# Analysis parameters
N_CPUS = 14
BATCH_SIZE = 28  # Sections per batch (2 per CPU)
MIN_MICROGLIA_PER_SECTION = 5
MIN_NEURONS_PER_TYPE = 5


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def classify_neuron(cell_cluster: str) -> str:
    """
    Classify a neuron by its neurotransmitter type based on cluster name.

    Parameters
    ----------
    cell_cluster : str
        Cell cluster annotation string

    Returns
    -------
    str
        Neuron type: 'Glutamatergic', 'GABAergic', or 'Other'
    """
    if pd.isna(cell_cluster):
        return 'Other'

    cluster_upper = str(cell_cluster).upper()
    for neuron_type, pattern in NEURON_PATTERNS.items():
        if pattern in cluster_upper:
            return neuron_type
    return 'Other'


def load_cell_annotations(stereo_dir: str, mouse_id: str = 'mouse1') -> Dict:
    """
    Load cell type annotations from Stereo-seq metadata.

    Parameters
    ----------
    stereo_dir : str
        Path to Stereo-seq data directory
    mouse_id : str
        Mouse identifier to filter (default: 'mouse1')

    Returns
    -------
    dict
        Dictionary mapping (section_id, cell_id) to cell metadata
    """
    print(f"Loading cell type annotations for {mouse_id}...")

    annotation_file = os.path.join(stereo_dir, 'stereoseq.celltypeTransfer.2mice.all.tsv')
    df = pd.read_csv(annotation_file, sep='\t')
    df = df[df['mouse'] == mouse_id]

    cell_lookup = {}
    for _, row in df.iterrows():
        key = (row['section_id'], row['cell_id'])
        cell_lookup[key] = {
            'cell_class': row['cell_class'],
            'cell_cluster': row['cell_cluster']
        }

    print(f"  Loaded annotations for {len(cell_lookup):,} cells")
    return cell_lookup


def process_section(args: Tuple[str, str]) -> Optional[List[Dict]]:
    """
    Process a single tissue section to extract microglia expression and proximity data.

    For each microglia:
    - Computes centroid coordinates
    - Extracts expression of target genes
    - Finds nearest glutamatergic and GABAergic neurons
    - Records distances and nearest neuron type

    Parameters
    ----------
    args : tuple
        (section_file_path, cell_lookup_pickle_path)

    Returns
    -------
    list or None
        List of dictionaries containing cell-level results, or None if insufficient data
    """
    section_file, cell_lookup_file = args

    # Load cell lookup from pickle (for multiprocessing compatibility)
    with open(cell_lookup_file, 'rb') as f:
        cell_lookup = pickle.load(f)

    section_id = os.path.basename(section_file).split('_')[2]

    # Accumulate per-cell data
    cell_data = defaultdict(lambda: {
        'x_sum': 0.0, 'y_sum': 0.0, 'umi_weight_sum': 0,
        'total_umi': 0,
        'gene_umi': defaultdict(int)
    })

    try:
        # Parse expression file
        with gzip.open(section_file, 'rt') as f:
            _ = f.readline()  # Skip header

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue

                gene = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                umi = int(parts[3])
                cell_label = int(parts[4])

                if cell_label == 0:  # Skip background
                    continue

                # Accumulate for UMI-weighted centroid calculation.
                # Weighting by UMI count gives more accurate centroids, as UMI
                # density is higher near the actual cell center.
                cell_data[cell_label]['x_sum'] += x * umi
                cell_data[cell_label]['y_sum'] += y * umi
                cell_data[cell_label]['umi_weight_sum'] += umi
                cell_data[cell_label]['total_umi'] += umi

                # Track target gene expression
                if gene in ALL_TARGET_GENES:
                    cell_data[cell_label]['gene_umi'][gene] += umi

        # Classify cells and compute centroids
        microglia = []
        neurons = {'Glutamatergic': [], 'GABAergic': [], 'Other': []}
        astrocytes = []

        for cell_label, data in cell_data.items():
            if data['umi_weight_sum'] == 0:
                continue

            # UMI-weighted centroid: positions with more UMIs (closer to cell
            # center) contribute proportionally more to the centroid estimate
            cx = data['x_sum'] / data['umi_weight_sum']
            cy = data['y_sum'] / data['umi_weight_sum']

            key = (section_id, cell_label)
            if key not in cell_lookup:
                continue

            cell_info = cell_lookup[key]
            cell_class = cell_info['cell_class']

            cell_entry = {
                'section_id': section_id,
                'cell_id': cell_label,
                'x': cx,
                'y': cy,
                'total_umi': data['total_umi'],
                'gene_umi': dict(data['gene_umi'])
            }

            if cell_class == 'Microglia':
                microglia.append(cell_entry)
            elif cell_class == 'Neurons':
                neuron_type = classify_neuron(cell_info['cell_cluster'])
                neurons[neuron_type].append([cx, cy])
            elif 'Astrocytes' in cell_class:
                astrocytes.append(cell_entry)

        # Check minimum cell counts
        if len(microglia) < MIN_MICROGLIA_PER_SECTION:
            return None

        # Build KD-trees for spatial queries
        neuron_trees = {}
        for neuron_type, coords in neurons.items():
            if len(coords) >= MIN_NEURONS_PER_TYPE:
                neuron_trees[neuron_type] = cKDTree(np.array(coords))

        # Process each microglia
        results = []
        for mg in microglia:
            mg_coord = np.array([[mg['x'], mg['y']]])

            # Find distance to nearest neuron of each type
            distances = {}
            for neuron_type, tree in neuron_trees.items():
                dist, _ = tree.query(mg_coord, k=1)
                distances[neuron_type] = dist[0]

            # Determine nearest neuron type (Glu vs GABA only)
            nearest_type = None
            if 'Glutamatergic' in distances and 'GABAergic' in distances:
                nearest_type = 'Glutamatergic' if distances['Glutamatergic'] < distances['GABAergic'] else 'GABAergic'
            elif 'Glutamatergic' in distances:
                nearest_type = 'Glutamatergic'
            elif 'GABAergic' in distances:
                nearest_type = 'GABAergic'

            results.append({
                'section_id': section_id,
                'cell_id': mg['cell_id'],
                'cell_class': 'Microglia',
                'total_umi': mg['total_umi'],
                'gene_umi': mg['gene_umi'],
                'nearest_neuron_type': nearest_type,
                'dist_to_glu': distances.get('Glutamatergic', np.nan),
                'dist_to_gaba': distances.get('GABAergic', np.nan),
                'n_glu_neurons': len(neurons['Glutamatergic']),
                'n_gaba_neurons': len(neurons['GABAergic'])
            })

        # Also collect astrocyte data for comparison
        for ast in astrocytes:
            results.append({
                'section_id': section_id,
                'cell_id': ast['cell_id'],
                'cell_class': 'Astrocytes',
                'total_umi': ast['total_umi'],
                'gene_umi': ast['gene_umi'],
                'nearest_neuron_type': None,
                'dist_to_glu': np.nan,
                'dist_to_gaba': np.nan,
                'n_glu_neurons': len(neurons['Glutamatergic']),
                'n_gaba_neurons': len(neurons['GABAergic'])
            })

        return results

    except Exception as e:
        print(f"  Error processing section {section_id}: {e}")
        return None


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analysis_expression_by_cell_type(df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """
    Analysis 1: Compare gene expression between microglia and astrocytes.

    Parameters
    ----------
    df : pd.DataFrame
        Expression data with cell_class column

    Returns
    -------
    tuple
        (glutamate_results, gaba_results) as lists of dictionaries
    """
    print("\n" + "=" * 100)
    print("ANALYSIS 1: GLUTAMATE/GABA TRANSPORTER EXPRESSION BY CELL TYPE")
    print("=" * 100)

    mg_df = df[df['cell_class'] == 'Microglia']
    ast_df = df[df['cell_class'] == 'Astrocytes']

    print(f"\nMicroglia: n = {len(mg_df):,}")
    print(f"Astrocytes: n = {len(ast_df):,}")

    # Glutamate pathway genes
    print("\n--- GLUTAMATE PATHWAY GENES ---")
    print(f"{'Gene':<10} {'Symbol':<12} {'MG %det':<12} {'MG CP10K':<12} {'AST %det':<12} {'AST CP10K':<12}")
    print("-" * 75)

    glu_results = []
    for name, gene in GLUTAMATE_GENES.items():
        mg_pct = (mg_df[gene] > 0).mean() * 100
        mg_cp10k = mg_df[f'{gene}_cp10k'].mean()
        ast_pct = (ast_df[gene] > 0).mean() * 100 if len(ast_df) > 0 else 0
        ast_cp10k = ast_df[f'{gene}_cp10k'].mean() if len(ast_df) > 0 else 0

        print(f"{name:<10} {gene:<12} {mg_pct:<12.2f} {mg_cp10k:<12.4f} {ast_pct:<12.2f} {ast_cp10k:<12.4f}")
        glu_results.append({
            'gene_name': name,
            'gene_symbol': gene,
            'microglia_pct_detected': mg_pct,
            'microglia_mean_cp10k': mg_cp10k,
            'astrocyte_pct_detected': ast_pct,
            'astrocyte_mean_cp10k': ast_cp10k
        })

    # GABA pathway genes
    print("\n--- GABA PATHWAY GENES ---")
    print(f"{'Gene':<10} {'Symbol':<12} {'MG %det':<12} {'MG CP10K':<12} {'AST %det':<12} {'AST CP10K':<12}")
    print("-" * 75)

    gaba_results = []
    for name, gene in GABA_GENES.items():
        mg_pct = (mg_df[gene] > 0).mean() * 100
        mg_cp10k = mg_df[f'{gene}_cp10k'].mean()
        ast_pct = (ast_df[gene] > 0).mean() * 100 if len(ast_df) > 0 else 0
        ast_cp10k = ast_df[f'{gene}_cp10k'].mean() if len(ast_df) > 0 else 0

        print(f"{name:<10} {gene:<12} {mg_pct:<12.2f} {mg_cp10k:<12.4f} {ast_pct:<12.2f} {ast_cp10k:<12.4f}")
        gaba_results.append({
            'gene_name': name,
            'gene_symbol': gene,
            'microglia_pct_detected': mg_pct,
            'microglia_mean_cp10k': mg_cp10k,
            'astrocyte_pct_detected': ast_pct,
            'astrocyte_mean_cp10k': ast_cp10k
        })

    return glu_results, gaba_results


def analysis_expression_by_proximity(df: pd.DataFrame) -> List[Dict]:
    """
    Analysis 2: Compare microglial gene expression by nearest neuron type.

    Tests whether microglia near glutamatergic neurons have different expression
    of Glu/GABA pathway genes compared to microglia near GABAergic neurons.

    Parameters
    ----------
    df : pd.DataFrame
        Expression data with nearest_neuron_type column

    Returns
    -------
    list
        Results as list of dictionaries with statistical tests
    """
    print("\n" + "=" * 100)
    print("ANALYSIS 2: MICROGLIAL EXPRESSION BY NEAREST NEURON TYPE")
    print("=" * 100)

    mg_df = df[df['cell_class'] == 'Microglia']
    mg_near_glu = mg_df[mg_df['nearest_neuron_type'] == 'Glutamatergic']
    mg_near_gaba = mg_df[mg_df['nearest_neuron_type'] == 'GABAergic']

    print(f"\nMicroglia nearest to Glutamatergic neurons: n = {len(mg_near_glu):,}")
    print(f"Microglia nearest to GABAergic neurons: n = {len(mg_near_gaba):,}")

    results = []

    # Glutamate pathway genes
    print("\n--- GLUTAMATE PATHWAY: Expression by Nearest Neuron Type ---")
    print(f"{'Gene':<10} {'Near GLU':<14} {'Near GABA':<14} {'Fold (G/B)':<14} {'p-value':<12} {'Sig'}")
    print("-" * 75)

    for name, gene in GLUTAMATE_GENES.items():
        expr_col = f'{gene}_cp10k'
        glu_vals = mg_near_glu[expr_col].values
        gaba_vals = mg_near_gaba[expr_col].values

        glu_mean = np.mean(glu_vals)
        gaba_mean = np.mean(gaba_vals)
        fold = glu_mean / gaba_mean if gaba_mean > 0 else np.nan

        _, pval = stats.mannwhitneyu(glu_vals, gaba_vals, alternative='two-sided')
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

        print(f"{name:<10} {glu_mean:<14.4f} {gaba_mean:<14.4f} {fold:<14.2f} {pval:<12.2e} {sig}")

        results.append({
            'gene_name': name,
            'gene_symbol': gene,
            'pathway': 'Glutamate',
            'mean_near_glu': glu_mean,
            'mean_near_gaba': gaba_mean,
            'fold_change_glu_vs_gaba': fold,
            'pvalue': pval
        })

    # GABA pathway genes
    print("\n--- GABA PATHWAY: Expression by Nearest Neuron Type ---")
    print(f"{'Gene':<10} {'Near GLU':<14} {'Near GABA':<14} {'Fold (B/G)':<14} {'p-value':<12} {'Sig'}")
    print("-" * 75)

    for name, gene in GABA_GENES.items():
        expr_col = f'{gene}_cp10k'
        glu_vals = mg_near_glu[expr_col].values
        gaba_vals = mg_near_gaba[expr_col].values

        glu_mean = np.mean(glu_vals)
        gaba_mean = np.mean(gaba_vals)
        fold = gaba_mean / glu_mean if glu_mean > 0 else np.nan

        _, pval = stats.mannwhitneyu(glu_vals, gaba_vals, alternative='two-sided')
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

        print(f"{name:<10} {glu_mean:<14.4f} {gaba_mean:<14.4f} {fold:<14.2f} {pval:<12.2e} {sig}")

        results.append({
            'gene_name': name,
            'gene_symbol': gene,
            'pathway': 'GABA',
            'mean_near_glu': glu_mean,
            'mean_near_gaba': gaba_mean,
            'fold_change_gaba_vs_glu': fold,
            'pvalue': pval
        })

    return results


def analysis_regional_enrichment(df: pd.DataFrame) -> List[Dict]:
    """
    Analysis 3: Examine regional enrichment of gene expression across brain sections.

    Tests correlation between gene expression and the local ratio of
    GABAergic to glutamatergic neurons.

    Parameters
    ----------
    df : pd.DataFrame
        Expression data with section_id column

    Returns
    -------
    list
        Regional correlation results
    """
    print("\n" + "=" * 100)
    print("ANALYSIS 3: REGIONAL ENRICHMENT")
    print("=" * 100)
    print("\nCorrelation between gene expression and regional GABA/GLU neuron ratio")

    mg_df = df[df['cell_class'] == 'Microglia']

    # Aggregate by section
    section_stats = mg_df.groupby('section_id').agg({
        **{gene: 'sum' for gene in ALL_TARGET_GENES},
        **{f'{gene}_cp10k': 'mean' for gene in ALL_TARGET_GENES},
        'total_umi': 'sum',
        'cell_id': 'count',
        'n_glu_neurons': 'first',
        'n_gaba_neurons': 'first'
    }).reset_index()
    section_stats = section_stats.rename(columns={'cell_id': 'n_microglia'})

    # Calculate GABA/GLU ratio per section
    section_stats['gaba_glu_ratio'] = section_stats['n_gaba_neurons'] / (section_stats['n_glu_neurons'] + 1)

    print("\n--- Correlation with Regional GABA/GLU Neuron Ratio ---")
    print(f"{'Gene':<10} {'Spearman r':<14} {'p-value':<14} {'Interpretation'}")
    print("-" * 70)

    results = []
    all_genes = {**GLUTAMATE_GENES, **GABA_GENES}

    for name, gene in all_genes.items():
        expr_col = f'{gene}_cp10k'
        valid = section_stats[section_stats[expr_col] > 0]

        if len(valid) > 10:
            r, p = stats.spearmanr(valid['gaba_glu_ratio'], valid[expr_col])

            if r > 0.2 and p < 0.05:
                interp = "Higher in GABA-rich regions"
            elif r < -0.2 and p < 0.05:
                interp = "Higher in GLU-rich regions"
            else:
                interp = "No regional bias"

            print(f"{name:<10} {r:<14.3f} {p:<14.2e} {interp}")

            results.append({
                'gene_name': name,
                'gene_symbol': gene,
                'spearman_r': r,
                'pvalue': p,
                'interpretation': interp
            })

    return results


def analysis_distance_correlation(df: pd.DataFrame) -> List[Dict]:
    """
    Analysis 4: Correlate gene expression with distance to nearest neurons.

    Tests whether expression of glutamate/GABA genes correlates with
    physical distance to the corresponding neuron types.

    Parameters
    ----------
    df : pd.DataFrame
        Expression data with distance columns

    Returns
    -------
    list
        Distance correlation results
    """
    print("\n" + "=" * 100)
    print("ANALYSIS 4: DISTANCE-BASED CORRELATION")
    print("=" * 100)
    print("\nCorrelation between gene expression and distance to nearest neuron type")

    mg_df = df[df['cell_class'] == 'Microglia']
    results = []

    # Glutamate genes vs distance to glutamatergic neurons
    print("\n--- Distance to Glutamatergic Neurons ---")
    print(f"{'Gene':<10} {'Spearman r':<14} {'p-value':<14} {'Interpretation'}")
    print("-" * 70)

    valid_glu = mg_df[mg_df['dist_to_glu'].notna()]

    for name, gene in GLUTAMATE_GENES.items():
        expr_col = f'{gene}_cp10k'
        if len(valid_glu) > 100:
            r, p = stats.spearmanr(valid_glu['dist_to_glu'], valid_glu[expr_col])

            if r < -0.05 and p < 0.05:
                interp = "HIGHER near GLU neurons"
            elif r > 0.05 and p < 0.05:
                interp = "LOWER near GLU neurons"
            else:
                interp = "No correlation"

            print(f"{name:<10} {r:<14.4f} {p:<14.2e} {interp}")

            results.append({
                'gene_name': name,
                'gene_symbol': gene,
                'neuron_type': 'Glutamatergic',
                'spearman_r': r,
                'pvalue': p,
                'interpretation': interp
            })

    # GABA genes vs distance to GABAergic neurons
    print("\n--- Distance to GABAergic Neurons ---")
    print(f"{'Gene':<10} {'Spearman r':<14} {'p-value':<14} {'Interpretation'}")
    print("-" * 70)

    valid_gaba = mg_df[mg_df['dist_to_gaba'].notna()]

    for name, gene in GABA_GENES.items():
        expr_col = f'{gene}_cp10k'
        if len(valid_gaba) > 100:
            r, p = stats.spearmanr(valid_gaba['dist_to_gaba'], valid_gaba[expr_col])

            if r < -0.05 and p < 0.05:
                interp = "HIGHER near GABA neurons"
            elif r > 0.05 and p < 0.05:
                interp = "LOWER near GABA neurons"
            else:
                interp = "No correlation"

            print(f"{name:<10} {r:<14.4f} {p:<14.2e} {interp}")

            results.append({
                'gene_name': name,
                'gene_symbol': gene,
                'neuron_type': 'GABAergic',
                'spearman_r': r,
                'pvalue': p,
                'interpretation': interp
            })

    return results


def print_summary(df: pd.DataFrame):
    """
    Print summary statistics for the analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Expression data
    """
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    mg_df = df[df['cell_class'] == 'Microglia'].copy()

    # Calculate pathway sums
    glu_genes = list(GLUTAMATE_GENES.values())
    gaba_genes = list(GABA_GENES.values())

    mg_df['glu_pathway_sum'] = mg_df[[f'{g}_cp10k' for g in glu_genes]].sum(axis=1)
    mg_df['gaba_pathway_sum'] = mg_df[[f'{g}_cp10k' for g in gaba_genes]].sum(axis=1)

    print(f"\nTotal microglia analyzed: {len(mg_df):,}")
    print(f"\nMean Glutamate Pathway Expression (CP10K sum): {mg_df['glu_pathway_sum'].mean():.4f}")
    print(f"Mean GABA Pathway Expression (CP10K sum): {mg_df['gaba_pathway_sum'].mean():.4f}")

    # Compare by nearest neuron type
    mg_near_glu = mg_df[mg_df['nearest_neuron_type'] == 'Glutamatergic']
    mg_near_gaba = mg_df[mg_df['nearest_neuron_type'] == 'GABAergic']

    print(f"\n--- Pathway Expression by Nearest Neuron Type ---")

    print(f"\nGlutamate pathway in microglia near GLU neurons: {mg_near_glu['glu_pathway_sum'].mean():.4f}")
    print(f"Glutamate pathway in microglia near GABA neurons: {mg_near_gaba['glu_pathway_sum'].mean():.4f}")
    _, p = stats.mannwhitneyu(mg_near_glu['glu_pathway_sum'], mg_near_gaba['glu_pathway_sum'])
    print(f"  Mann-Whitney U p-value: {p:.2e}")

    print(f"\nGABA pathway in microglia near GLU neurons: {mg_near_glu['gaba_pathway_sum'].mean():.4f}")
    print(f"GABA pathway in microglia near GABA neurons: {mg_near_gaba['gaba_pathway_sum'].mean():.4f}")
    _, p = stats.mannwhitneyu(mg_near_glu['gaba_pathway_sum'], mg_near_gaba['gaba_pathway_sum'])
    print(f"  Mann-Whitney U p-value: {p:.2e}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main analysis pipeline."""

    parser = argparse.ArgumentParser(
        description='Analyze microglial Glu/GABA transporter expression in Stereo-seq data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python microglia_glu_gaba_analysis_v2.py --stereo-dir /path/to/stereoseq --output-dir ./results
  python microglia_glu_gaba_analysis_v2.py -s /Volumes/T9/han_brain_stereo_seq -o ./output
        """
    )
    parser.add_argument('-s', '--stereo-dir', required=True,
                        help='Path to Stereo-seq data directory')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='Output directory for results')
    parser.add_argument('--mouse', default='mouse1',
                        help='Mouse identifier (default: mouse1)')
    parser.add_argument('--cpus', type=int, default=N_CPUS,
                        help=f'Number of CPUs for parallel processing (default: {N_CPUS})')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.stereo_dir):
        sys.exit(f"Error: Stereo-seq directory not found: {args.stereo_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 100)
    print("MICROGLIA GLUTAMATE/GABA TRANSPORTER EXPRESSION ANALYSIS")
    print("=" * 100)
    print(f"\nStereo-seq directory: {args.stereo_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mouse: {args.mouse}")
    print(f"CPUs: {args.cpus}")

    # Load cell annotations
    cell_lookup = load_cell_annotations(args.stereo_dir, args.mouse)

    # Save cell lookup for multiprocessing
    pickle_file = os.path.join(args.output_dir, 'cell_lookup_temp.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(cell_lookup, f)

    # Find section files
    section_pattern = os.path.join(args.stereo_dir, f'total_gene_T*_{args.mouse}*.txt.gz')
    section_files = sorted(glob(section_pattern))
    print(f"\nFound {len(section_files)} section files")

    if len(section_files) == 0:
        sys.exit(f"Error: No section files found matching pattern: {section_pattern}")

    # Process sections in batches
    print(f"\nProcessing sections with {args.cpus} CPUs...")
    all_results = []
    n_batches = (len(section_files) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(section_files))
        batch_files = section_files[start_idx:end_idx]

        print(f"  Batch {batch_idx + 1}/{n_batches} (sections {start_idx + 1}-{end_idx})...")

        args_list = [(sf, pickle_file) for sf in batch_files]

        with Pool(args.cpus) as pool:
            batch_results = pool.map(process_section, args_list)

        for result in batch_results:
            if result:
                all_results.extend(result)

    # Clean up temp file
    os.remove(pickle_file)

    print(f"\nCollected data for {len(all_results):,} cells")

    # Build results DataFrame
    rows = []
    for r in all_results:
        row = {
            'section_id': r['section_id'],
            'cell_id': r['cell_id'],
            'cell_class': r['cell_class'],
            'total_umi': r['total_umi'],
            'nearest_neuron_type': r['nearest_neuron_type'],
            'dist_to_glu': r['dist_to_glu'],
            'dist_to_gaba': r['dist_to_gaba'],
            'n_glu_neurons': r['n_glu_neurons'],
            'n_gaba_neurons': r['n_gaba_neurons']
        }
        # Add gene expression columns
        for gene in ALL_TARGET_GENES:
            row[gene] = r['gene_umi'].get(gene, 0)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Calculate CP10K (counts per 10,000 UMI)
    for gene in ALL_TARGET_GENES:
        df[f'{gene}_cp10k'] = df[gene] / df['total_umi'] * 10000

    # Run analyses
    glu_celltype, gaba_celltype = analysis_expression_by_cell_type(df)
    proximity_results = analysis_expression_by_proximity(df)
    regional_results = analysis_regional_enrichment(df)
    distance_results = analysis_distance_correlation(df)
    print_summary(df)

    # Save results
    print("\n" + "=" * 100)
    print("SAVING RESULTS")
    print("=" * 100)

    # Main expression data
    output_file = os.path.join(args.output_dir, 'microglia_glu_gaba_expression.csv')
    df.to_csv(output_file, index=False)
    print(f"  Expression data: {output_file}")

    # Cell type comparison
    celltype_df = pd.DataFrame(glu_celltype + gaba_celltype)
    output_file = os.path.join(args.output_dir, 'expression_by_celltype.csv')
    celltype_df.to_csv(output_file, index=False)
    print(f"  Cell type comparison: {output_file}")

    # Proximity analysis
    output_file = os.path.join(args.output_dir, 'expression_by_proximity.csv')
    pd.DataFrame(proximity_results).to_csv(output_file, index=False)
    print(f"  Proximity analysis: {output_file}")

    # Regional analysis
    output_file = os.path.join(args.output_dir, 'regional_enrichment.csv')
    pd.DataFrame(regional_results).to_csv(output_file, index=False)
    print(f"  Regional enrichment: {output_file}")

    # Distance correlation
    output_file = os.path.join(args.output_dir, 'distance_correlation.csv')
    pd.DataFrame(distance_results).to_csv(output_file, index=False)
    print(f"  Distance correlation: {output_file}")

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
