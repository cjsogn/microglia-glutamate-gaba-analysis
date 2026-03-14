#!/usr/bin/env python3
"""
Decontamination Validation of Microglial Gene Expression

Uses principled contamination correction approaches (conceptually similar to
SoupX/DecontX) to evaluate whether GABA and glutamate gene expression in
microglia survives ambient RNA removal.

Two complementary methods:
  Method 1 - Ambient Profile Subtraction (SoupX-like):
    Estimate ambient RNA profile from source cells (neurons/astrocytes),
    estimate per-cell contamination fraction from known markers, subtract
    estimated contamination, and evaluate residual expression.

  Method 2 - Regression-based Decontamination:
    For each target gene, regress expression on contamination marker
    expression.  Residuals represent decontaminated signal.
    Test whether residual expression is significantly > 0.

The data is in log2(CPM+1) units (counts per million, not raw counts);
we reverse this to linear CPM scale before analysis.

Manuscript figures: Supplementary Fig S2
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, sparse
from sklearn.linear_model import LinearRegression
import warnings
import os
import time
from multiprocessing import Pool

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = "/Users/cjsogn/Documents/Artikkel 1/supp/Decontamination_Validation"
DATA_DIR = "/Users/cjsogn/Documents/data/Whole aging mouse brain RNA Zeng"
H5AD_PATH = os.path.join(DATA_DIR, "Zeng-Aging-Mouse-10Xv3-log2.h5ad")
METADATA_PATH = os.path.join(DATA_DIR, "cell_cluster_mapping_annotations.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

N_CORES = 14

# Target genes
GENE_INFO = {
    'Gad1':   {'name': 'GAD1',  'category': 'GABA',     'source': 'Neuron'},
    'Gad2':   {'name': 'GAD2',  'category': 'GABA',     'source': 'Neuron'},
    'Slc6a1': {'name': 'GAT1',  'category': 'GABA',     'source': 'Neuron'},
    'Slc6a11':{'name': 'GAT3',  'category': 'GABA',     'source': 'Astro'},
    'Abat':   {'name': 'ABAT',  'category': 'GABA',     'source': 'Both'},
    'Gls':    {'name': 'GLS',   'category': 'Glutamate', 'source': 'Neuron'},
    'Glul':   {'name': 'GLUL',  'category': 'Glutamate', 'source': 'Astro'},
    'Slc1a3': {'name': 'EAAT1', 'category': 'Glutamate', 'source': 'Astro'},
    'Slc1a2': {'name': 'EAAT2', 'category': 'Glutamate', 'source': 'Astro'},
    'Slc38a1':{'name': 'SNAT1', 'category': 'Glutamate', 'source': 'Neuron'},
}
TARGET_GENES = list(GENE_INFO.keys())

# Contamination markers (expressed in source cells, not in microglia)
NEURONAL_MARKERS = ['Snap25', 'Syt1', 'Rbfox3', 'Tubb3', 'Nefl', 'Nefm', 'Map2']
ASTRO_MARKERS = ['Gfap', 'Aqp4', 'Aldh1l1', 'Sox9']
ALL_CONTAM_MARKERS = NEURONAL_MARKERS + ASTRO_MARKERS

# Microglial identity markers
MICRO_MARKERS = ['P2ry12', 'Cx3cr1', 'Hexb', 'Csf1r', 'Tmem119', 'Aif1', 'Itgam']

ALL_GENES_NEEDED = list(set(TARGET_GENES + ALL_CONTAM_MARKERS + MICRO_MARKERS))

# Nature journal style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})


# ============================================================================
# Step 1: Load metadata and identify cell types
# ============================================================================

def load_metadata():
    """Load cell metadata and identify microglia, neurons, astrocytes."""
    print("Loading cell metadata...")
    t0 = time.time()

    meta = pd.read_csv(METADATA_PATH, usecols=['cell_label', 'class_name', 'subclass_name'])
    print(f"  Loaded {len(meta):,} cells in {time.time()-t0:.1f}s")

    # Identify cell types
    microglia_mask = meta['subclass_name'].str.contains('Microglia', case=False, na=False)
    neuron_mask = meta['class_name'].str.contains('GABA|Glut', case=False, na=False)
    astro_mask = meta['class_name'].str.contains('Astro', case=False, na=False)

    print(f"  Microglia: {microglia_mask.sum():,}")
    print(f"  Neurons:   {neuron_mask.sum():,}")
    print(f"  Astrocytes: {astro_mask.sum():,}")

    return meta, microglia_mask, neuron_mask, astro_mask


# ============================================================================
# Step 2: Extract gene expression from h5ad (memory-efficient)
# ============================================================================

def extract_expression(meta, microglia_mask, neuron_mask, astro_mask):
    """
    Read the h5ad file using h5py and extract expression for needed genes
    and needed cells only. Returns expression DataFrames for each cell type.
    """
    print("\nReading h5ad file (extracting needed genes)...")
    t0 = time.time()

    # First, map cell_labels to h5ad row indices
    with h5py.File(H5AD_PATH, 'r') as f:
        # Read cell labels from h5ad
        cl_group = f['obs']['cell_label']
        if 'categories' in cl_group:
            cats = np.array([c.decode() if isinstance(c, bytes) else c
                           for c in cl_group['categories'][:]])
            codes = cl_group['codes'][:]
            h5_cell_labels = cats[codes]
        else:
            h5_cell_labels = np.array([c.decode() if isinstance(c, bytes) else c
                                       for c in cl_group[:]])

        # Read gene symbols
        gs_group = f['var']['gene_symbol']
        if 'categories' in gs_group:
            gs_cats = np.array([c.decode() if isinstance(c, bytes) else c
                               for c in gs_group['categories'][:]])
            gs_codes = gs_group['codes'][:]
            h5_gene_symbols = gs_cats[gs_codes]
        else:
            h5_gene_symbols = np.array([c.decode() if isinstance(c, bytes) else c
                                        for c in gs_group[:]])

        print(f"  H5AD: {len(h5_cell_labels):,} cells x {len(h5_gene_symbols):,} genes")

        # Find gene column indices
        gene_col_map = {}
        for gene in ALL_GENES_NEEDED:
            matches = np.where(h5_gene_symbols == gene)[0]
            if len(matches) > 0:
                gene_col_map[gene] = matches[0]
            else:
                print(f"  WARNING: {gene} not found in h5ad")
        gene_cols = sorted(gene_col_map.values())
        gene_names_ordered = [h5_gene_symbols[c] for c in gene_cols]
        print(f"  Found {len(gene_col_map)}/{len(ALL_GENES_NEEDED)} genes")

        # Map metadata cell labels to h5ad row indices
        # Build index: cell_label -> h5ad row
        h5_label_to_idx = {label: i for i, label in enumerate(h5_cell_labels)}

        # Get indices for each cell type
        meta_labels = meta['cell_label'].values
        micro_labels = meta_labels[microglia_mask.values]
        neuron_labels = meta_labels[neuron_mask.values]
        astro_labels = meta_labels[astro_mask.values]

        micro_h5_idx = np.array([h5_label_to_idx[l] for l in micro_labels
                                 if l in h5_label_to_idx])
        neuron_h5_idx = np.array([h5_label_to_idx[l] for l in neuron_labels
                                  if l in h5_label_to_idx])
        astro_h5_idx = np.array([h5_label_to_idx[l] for l in astro_labels
                                 if l in h5_label_to_idx])

        print(f"  Matched: {len(micro_h5_idx):,} microglia, "
              f"{len(neuron_h5_idx):,} neurons, {len(astro_h5_idx):,} astrocytes")

        # Read the sparse matrix (CSR format)
        print("  Reading sparse matrix...")
        data = f['X']['data'][:]
        indices = f['X']['indices'][:]
        indptr = f['X']['indptr'][:]
        shape = tuple(f['X'].attrs['shape'])

    print(f"  Sparse matrix loaded in {time.time()-t0:.1f}s")
    print(f"  Reconstructing CSR matrix...")

    X = sparse.csr_matrix((data, indices, indptr), shape=shape)

    # Extract only needed gene columns (dramatically reduces memory)
    print(f"  Extracting {len(gene_cols)} gene columns...")
    X_subset = X[:, gene_cols].toarray()  # n_cells x n_genes_needed (small)
    del X, data, indices, indptr  # free memory

    # Reverse log2(CPM+1) transform: 2^x - 1 yields CPM values (not raw counts)
    print("  Reversing log2 transform...")
    X_linear = np.power(2.0, X_subset) - 1.0
    X_linear[X_linear < 0] = 0  # clip numerical noise
    del X_subset

    # Create DataFrames for each cell type
    df_all = pd.DataFrame(X_linear, columns=gene_names_ordered)

    df_micro = df_all.iloc[micro_h5_idx].reset_index(drop=True)
    df_neuron = df_all.iloc[neuron_h5_idx].reset_index(drop=True)
    df_astro = df_all.iloc[astro_h5_idx].reset_index(drop=True)
    del df_all, X_linear

    print(f"  Extraction complete in {time.time()-t0:.1f}s")
    print(f"  Microglia: {df_micro.shape}, Neurons: {df_neuron.shape}, Astro: {df_astro.shape}")

    return df_micro, df_neuron, df_astro, gene_names_ordered


# ============================================================================
# Step 3: Method 1 - Ambient Profile Subtraction
# ============================================================================

def ambient_profile_correction(df_micro, df_neuron, df_astro):
    """
    SoupX-like ambient correction:
    1. Estimate ambient RNA profile from source cells
    2. Estimate per-cell contamination fraction using known markers
    3. Subtract estimated contamination
    4. Evaluate residual expression
    """
    print("\n" + "=" * 60)
    print("METHOD 1: Ambient Profile Subtraction")
    print("=" * 60)

    results = {}

    for gene in TARGET_GENES:
        info = GENE_INFO[gene]
        source = info['source']

        # Get microglial expression of this gene
        micro_expr = df_micro[gene].values

        # Determine source cell ambient profile for this gene
        if source == 'Neuron':
            source_expr = df_neuron[gene].values
            contam_markers = NEURONAL_MARKERS
        elif source == 'Astro':
            source_expr = df_astro[gene].values
            contam_markers = ASTRO_MARKERS
        else:  # Both
            source_expr_n = df_neuron[gene].values
            source_expr_a = df_astro[gene].values
            source_expr = np.concatenate([source_expr_n, source_expr_a])
            contam_markers = ALL_CONTAM_MARKERS

        # Ambient expression level for this gene (mean across source cells)
        ambient_level = np.mean(source_expr)

        # Estimate per-cell contamination fraction in microglia
        # Using contamination markers: average expression of contamination markers
        # normalized by their average in source cells
        available_markers = [m for m in contam_markers if m in df_micro.columns]

        if source == 'Neuron':
            source_marker_df = df_neuron
        elif source == 'Astro':
            source_marker_df = df_astro
        else:  # Both - use appropriate source for each marker type
            source_marker_df = None

        # For each marker, compute: (microglia expression) / (source cell mean)
        # Average across markers = contamination fraction estimate
        contam_fractions = np.zeros(len(df_micro))
        n_markers_used = 0
        for marker in available_markers:
            # For 'Both' source genes, use the appropriate source cell type
            if source_marker_df is not None:
                source_mean = np.mean(source_marker_df[marker].values)
            elif marker in NEURONAL_MARKERS:
                source_mean = np.mean(df_neuron[marker].values)
            else:
                source_mean = np.mean(df_astro[marker].values)
            if source_mean > 0:
                contam_fractions += df_micro[marker].values / source_mean
                n_markers_used += 1

        if n_markers_used > 0:
            contam_fractions /= n_markers_used
        # Clip to [0, 1]
        contam_fractions = np.clip(contam_fractions, 0, 1)

        # Corrected expression: observed - contamination_fraction * ambient_level
        corrected_expr = micro_expr - contam_fractions * ambient_level
        corrected_expr = np.maximum(corrected_expr, 0)

        # Evaluate: expressing cells before and after
        expressing_before = np.sum(micro_expr > 0)
        expressing_after = np.sum(corrected_expr > 0)
        mean_before = np.mean(micro_expr)
        mean_after = np.mean(corrected_expr)
        retention_fraction = mean_after / mean_before if mean_before > 0 else 0

        # Test whether corrected expression retains signal above zero.
        # Use all cells with positive corrected expression (not just originally
        # expressing cells) to avoid biasing the test toward significance.
        expr_nonzero = corrected_expr[corrected_expr > 0]
        if len(expr_nonzero) >= 10:
            stat, pval = stats.wilcoxon(expr_nonzero, alternative='greater')
        else:
            stat, pval = 0, 1.0

        # Median contamination fraction
        median_contam = np.median(contam_fractions[micro_expr > 0]) if expressing_before > 0 else 0

        results[gene] = {
            'name': info['name'],
            'category': info['category'],
            'source': source,
            'n_micro_total': len(df_micro),
            'expressing_before': int(expressing_before),
            'expressing_after': int(expressing_after),
            'pct_retained_cells': expressing_after / expressing_before * 100 if expressing_before > 0 else 0,
            'mean_before': mean_before,
            'mean_after': mean_after,
            'retention_fraction': retention_fraction,
            'ambient_level': ambient_level,
            'median_contam_fraction': median_contam,
            'wilcoxon_stat': stat,
            'wilcoxon_p': pval,
            'corrected_expr': corrected_expr,
            'original_expr': micro_expr,
            'contam_fractions': contam_fractions,
        }

        verdict = "SURVIVES" if pval < 0.01 and retention_fraction > 0.1 else "REDUCED" if pval < 0.05 else "REMOVED"
        print(f"  {info['name']:6s}: retention={retention_fraction:.1%}, "
              f"cells {expressing_before:,}->{expressing_after:,} "
              f"({results[gene]['pct_retained_cells']:.1f}%), "
              f"p={pval:.2e} -> {verdict}")

    return results


# ============================================================================
# Step 4: Method 2 - Regression-based Decontamination
# ============================================================================

def _regression_worker(args):
    """
    Worker for parallel regression.

    Correct approach: fit gene_expr = intercept + sum(beta_i * marker_i) + epsilon.
    The contamination-explained component = sum(beta_i * marker_i) (no intercept).
    Decontaminated expression = observed - max(0, contamination_component).
    The intercept represents baseline expression when all markers are zero.
    We test: (a) is the intercept significantly > 0, and
             (b) is the decontaminated expression significantly > 0.
    """
    gene, micro_expr, contam_matrix, marker_names = args
    n_cells = len(micro_expr)

    # Only use cells that have some expression
    mask = micro_expr > 0

    if mask.sum() < 10:
        return gene, {
            'coef_intercept': np.nan,
            'intercept_p': 1.0,
            'r_squared': np.nan,
            'decontam_retention': np.nan,
            'decontam_pos_frac': np.nan,
            'p_decontam_gt_zero': 1.0,
            'marker_coefficients': {},
            'decontam_expr': np.zeros(n_cells),
        }

    X = contam_matrix[mask]
    y = micro_expr[mask]
    n = len(y)
    k = X.shape[1]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r_sq = model.score(X, y)

    # Compute intercept p-value (standard OLS formula)
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (n - k - 1)
    X_with_intercept = np.column_stack([np.ones(n), X])
    try:
        cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        se_intercept = np.sqrt(cov_matrix[0, 0])
        t_intercept = model.intercept_ / se_intercept
        # One-sided: intercept > 0
        intercept_p = 1.0 - stats.t.cdf(t_intercept, df=n - k - 1)
    except np.linalg.LinAlgError:
        intercept_p = 1.0

    # Decontaminated expression:
    # contamination_component = sum(beta_i * marker_i) for each cell
    # (the part explained by markers, excluding intercept)
    contam_component = X @ model.coef_
    # Only subtract positive contamination contributions
    contam_component_clipped = np.maximum(contam_component, 0)
    decontam_expr_subset = y - contam_component_clipped
    decontam_expr_subset = np.maximum(decontam_expr_subset, 0)

    # Full array
    decontam_expr = np.zeros(n_cells)
    decontam_expr[mask] = decontam_expr_subset

    # Retention
    mean_before = np.mean(y)
    mean_after = np.mean(decontam_expr_subset)
    retention = mean_after / mean_before if mean_before > 0 else 0

    # Fraction of expressing cells with positive decontaminated expression
    decontam_pos_frac = np.mean(decontam_expr_subset > 0)

    # Wilcoxon test: is decontaminated expression > 0?
    if np.any(decontam_expr_subset > 0):
        stat, p_decontam = stats.wilcoxon(decontam_expr_subset, alternative='greater')
    else:
        p_decontam = 1.0

    marker_coefs = {marker_names[i]: model.coef_[i] for i in range(len(marker_names))}

    return gene, {
        'coef_intercept': model.intercept_,
        'intercept_p': intercept_p,
        'r_squared': r_sq,
        'decontam_retention': retention,
        'decontam_pos_frac': decontam_pos_frac,
        'p_decontam_gt_zero': p_decontam,
        'marker_coefficients': marker_coefs,
        'decontam_expr': decontam_expr,
    }


def regression_decontamination(df_micro):
    """
    Regression-based decontamination:
    For each target gene, regress on all contamination markers.
    The intercept and residuals represent expression not explained
    by contamination.
    """
    print("\n" + "=" * 60)
    print("METHOD 2: Regression-based Decontamination")
    print("=" * 60)

    available_markers = [m for m in ALL_CONTAM_MARKERS if m in df_micro.columns]
    contam_matrix = df_micro[available_markers].values
    print(f"  Using {len(available_markers)} contamination markers")

    # Prepare parallel jobs
    jobs = []
    for gene in TARGET_GENES:
        micro_expr = df_micro[gene].values
        jobs.append((gene, micro_expr, contam_matrix, available_markers))

    # Run in parallel
    print(f"  Running regressions on {N_CORES} cores...")
    with Pool(min(N_CORES, len(jobs))) as pool:
        raw_results = pool.map(_regression_worker, jobs)

    results = {}
    for gene, res in raw_results:
        results[gene] = res
        info = GENE_INFO[gene]
        p_int = res['intercept_p']
        p_dec = res['p_decontam_gt_zero']
        verdict = "SURVIVES" if p_dec < 0.01 and res['decontam_retention'] > 0.1 else \
                  "MARGINAL" if p_dec < 0.05 else "REMOVED"
        print(f"  {info['name']:6s}: intercept={res['coef_intercept']:.1f} (p={p_int:.2e}), "
              f"R²={res['r_squared']:.3f}, "
              f"retention={res['decontam_retention']:.1%}, "
              f"p_decontam={p_dec:.2e} -> {verdict}")

    return results


# ============================================================================
# Step 5: Figures
# ============================================================================

def create_figures(ambient_results, regression_results, df_micro, df_neuron, df_astro):
    """Create publication-quality figures."""
    print("\nGenerating figures...")

    gene_names = [GENE_INFO[g]['name'] for g in TARGET_GENES]

    # Sort genes by ambient retention
    sorted_genes = sorted(TARGET_GENES,
                         key=lambda g: ambient_results[g]['retention_fraction'],
                         reverse=True)
    sorted_names = [GENE_INFO[g]['name'] for g in sorted_genes]

    # ======================================================================
    # Figure 1: Ambient Profile Correction Results (3 panels)
    # ======================================================================
    fig = plt.figure(figsize=(7.2, 9.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.2, 1.0],
                          hspace=0.45, wspace=0.35,
                          left=0.10, right=0.95, top=0.95, bottom=0.06)

    # ------ Panel A: Expression retention bar chart ------
    ax_a = fig.add_subplot(gs[0, 0])

    retention_vals = [ambient_results[g]['retention_fraction'] * 100 for g in sorted_genes]
    bar_colors = ['#C0392B' if GENE_INFO[g]['category'] == 'GABA' else '#2471A3'
                  for g in sorted_genes]

    y_pos = np.arange(len(sorted_genes))
    bars = ax_a.barh(y_pos, retention_vals, color=bar_colors, alpha=0.85,
                     edgecolor='black', linewidth=0.3, height=0.7)

    for i, (val, gene) in enumerate(zip(retention_vals, sorted_genes)):
        p = ambient_results[gene]['wilcoxon_p']
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        ax_a.text(val + 1, i, f'{val:.1f}% {stars}', va='center', fontsize=5)

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(sorted_names, fontsize=6.5)
    ax_a.set_xlabel('Expression Retained After\nAmbient Correction (%)', fontsize=6.5)
    ax_a.set_xlim(0, 115)
    ax_a.axvline(50, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax_a.invert_yaxis()
    ax_a.set_title('A', loc='left', fontsize=10, fontweight='bold', x=-0.15, y=1.02)

    # ------ Panel B: Cells retained ------
    ax_b = fig.add_subplot(gs[0, 1])

    cells_retained = [ambient_results[g]['pct_retained_cells'] for g in sorted_genes]
    bars_b = ax_b.barh(y_pos, cells_retained, color=bar_colors, alpha=0.85,
                       edgecolor='black', linewidth=0.3, height=0.7)

    for i, (val, gene) in enumerate(zip(cells_retained, sorted_genes)):
        before = ambient_results[gene]['expressing_before']
        after = ambient_results[gene]['expressing_after']
        ax_b.text(val + 1, i, f'{after:,}/{before:,}', va='center', fontsize=4.5)

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(sorted_names, fontsize=6.5)
    ax_b.set_xlabel('Expressing Cells Retained\nAfter Correction (%)', fontsize=6.5)
    ax_b.set_xlim(0, 115)
    ax_b.axvline(50, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax_b.invert_yaxis()
    ax_b.set_title('B', loc='left', fontsize=10, fontweight='bold', x=-0.15, y=1.02)

    # ------ Panel C: Before vs After distributions (selected genes) ------
    ax_c = fig.add_subplot(gs[1, :])

    # Show top 5 and bottom 2 genes
    show_genes = sorted_genes[:5] + sorted_genes[-2:]
    n_show = len(show_genes)

    positions = np.arange(n_show)
    width = 0.35

    means_before = [ambient_results[g]['mean_before'] for g in show_genes]
    means_after = [ambient_results[g]['mean_after'] for g in show_genes]

    bars_before = ax_c.bar(positions - width/2, means_before, width,
                           label='Before correction', color='#E8E8E8',
                           edgecolor='black', linewidth=0.3)
    bars_after = ax_c.bar(positions + width/2, means_after, width,
                          label='After correction',
                          color=['#C0392B' if GENE_INFO[g]['category'] == 'GABA' else '#2471A3'
                                 for g in show_genes],
                          alpha=0.85, edgecolor='black', linewidth=0.3)

    ax_c.set_xticks(positions)
    ax_c.set_xticklabels([GENE_INFO[g]['name'] for g in show_genes], fontsize=6.5)
    ax_c.set_ylabel('Mean Expression\n(linear CPM scale)', fontsize=6.5)
    ax_c.legend(fontsize=6, framealpha=0.8, edgecolor='none')
    ax_c.set_title('C', loc='left', fontsize=10, fontweight='bold', x=-0.06, y=1.02)

    # Add retention % labels
    for i, gene in enumerate(show_genes):
        ret = ambient_results[gene]['retention_fraction'] * 100
        ax_c.text(i, max(means_before[i], means_after[i]) * 1.05,
                  f'{ret:.0f}%', ha='center', fontsize=5.5, fontweight='bold')

    # ------ Panel D: Regression R² and residual significance ------
    ax_d = fig.add_subplot(gs[2, 0])

    r_sq_vals = [regression_results[g]['r_squared'] for g in sorted_genes]
    bars_d = ax_d.barh(y_pos, r_sq_vals, color=bar_colors, alpha=0.85,
                       edgecolor='black', linewidth=0.3, height=0.7)

    for i, (val, gene) in enumerate(zip(r_sq_vals, sorted_genes)):
        p = regression_results[gene]['intercept_p']
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        ax_d.text(val + 0.01, i, f'R²={val:.3f} {stars}', va='center', fontsize=5)

    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(sorted_names, fontsize=6.5)
    ax_d.set_xlabel('Variance Explained by\nContamination Markers (R²)', fontsize=6.5)
    ax_d.set_xlim(0, 1.0)
    ax_d.invert_yaxis()
    ax_d.set_title('D', loc='left', fontsize=10, fontweight='bold', x=-0.15, y=1.02)

    # ------ Panel E: Combined verdict ------
    ax_e = fig.add_subplot(gs[2, 1])

    # X-axis: ambient retention, Y-axis: regression residual positive fraction
    for gene in TARGET_GENES:
        x = ambient_results[gene]['retention_fraction'] * 100
        y = regression_results[gene]['decontam_pos_frac'] * 100
        color = '#C0392B' if GENE_INFO[gene]['category'] == 'GABA' else '#2471A3'

        # Marker size by significance (both methods)
        p1 = ambient_results[gene]['wilcoxon_p']
        p2 = regression_results[gene]['p_decontam_gt_zero']
        if p1 < 0.01 and p2 < 0.01:
            ms = 60
            marker = 'o'
        elif p1 < 0.05 or p2 < 0.05:
            ms = 40
            marker = 's'
        else:
            ms = 25
            marker = '^'

        ax_e.scatter(x, y, s=ms, c=color, marker=marker, alpha=0.85,
                     edgecolor='black', linewidth=0.4, zorder=3)
        ax_e.annotate(GENE_INFO[gene]['name'], (x, y),
                      fontsize=5, ha='left', va='bottom',
                      xytext=(3, 3), textcoords='offset points')

    ax_e.axvline(50, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax_e.axhline(50, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax_e.set_xlabel('Ambient Correction: Expression Retained (%)', fontsize=6)
    ax_e.set_ylabel('Regression: Cells with\nPositive Residual (%)', fontsize=6)
    ax_e.set_xlim(0, 105)
    ax_e.set_ylim(0, 105)

    # Quadrant labels
    ax_e.text(75, 95, 'Both methods\nsupport', fontsize=5, ha='center',
              color='darkgreen', alpha=0.6)
    ax_e.text(25, 5, 'Both methods\nreject', fontsize=5, ha='center',
              color='darkred', alpha=0.6)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=6, label='Both p<0.01'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=5, label='One p<0.05'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markersize=5, label='Neither sig.'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#C0392B',
               markersize=5, label='GABA'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2471A3',
               markersize=5, label='Glutamate'),
    ]
    ax_e.legend(handles=legend_elements, fontsize=5, loc='lower right',
                framealpha=0.8, edgecolor='none')

    ax_e.set_title('E', loc='left', fontsize=10, fontweight='bold', x=-0.15, y=1.02)

    plt.savefig(os.path.join(FIGURES_DIR, "decontamination_validation.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: decontamination_validation.png")

    # ======================================================================
    # Figure 2: Expression distributions before/after (violin plots)
    # ======================================================================
    fig2, axes = plt.subplots(2, 5, figsize=(7.2, 4.5))
    fig2.subplots_adjust(hspace=0.5, wspace=0.4)

    for idx, gene in enumerate(sorted_genes):
        ax = axes[idx // 5, idx % 5]
        info = GENE_INFO[gene]

        orig = ambient_results[gene]['original_expr']
        corr = ambient_results[gene]['corrected_expr']

        # Only plot expressing cells
        orig_expr = orig[orig > 0]
        corr_expr = corr[corr > 0]

        color = '#C0392B' if info['category'] == 'GABA' else '#2471A3'

        data_to_plot = []
        labels = []
        if len(orig_expr) > 0:
            # Subsample for violin plot if too many cells
            if len(orig_expr) > 5000:
                rng = np.random.default_rng(42)
                orig_sub = rng.choice(orig_expr, 5000, replace=False)
            else:
                orig_sub = orig_expr
            data_to_plot.append(orig_sub)
            labels.append('Before')

        if len(corr_expr) > 0:
            if len(corr_expr) > 5000:
                rng = np.random.default_rng(42)
                corr_sub = rng.choice(corr_expr, 5000, replace=False)
            else:
                corr_sub = corr_expr
            data_to_plot.append(corr_sub)
            labels.append('After')

        if data_to_plot:
            parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=False)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor('#CCCCCC' if i == 0 else color)
                pc.set_alpha(0.7)
            parts['cmeans'].set_color('black')
            parts['cmeans'].set_linewidth(0.8)
            for key in ['cbars', 'cmins', 'cmaxes']:
                parts[key].set_linewidth(0.5)

            ax.set_xticks([1, 2] if len(data_to_plot) == 2 else [1])
            ax.set_xticklabels(labels, fontsize=5)

        ret = ambient_results[gene]['retention_fraction'] * 100
        p = ambient_results[gene]['wilcoxon_p']
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        ax.set_title(f'{info["name"]} ({ret:.0f}% {stars})', fontsize=6, fontweight='bold',
                     color=color)
        ax.tick_params(axis='y', labelsize=5)

        if idx % 5 == 0:
            ax.set_ylabel('Expression', fontsize=5.5)

    plt.savefig(os.path.join(FIGURES_DIR, "decontamination_distributions.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: decontamination_distributions.png")

    # ======================================================================
    # Figure 3: Summary table
    # ======================================================================
    fig_t, ax_t = plt.subplots(figsize=(7.2, 3.5))
    ax_t.axis('off')

    headers = ['Gene', 'Cat.', 'Source', 'Ambient\nRetention',
               'Cells\nRetained', 'Regr.\nRetention', 'Regr. R²\n(contam.)', 'Verdict']

    table_data = []
    for gene in sorted_genes:
        ar = ambient_results[gene]
        rr = regression_results[gene]

        p1 = ar['wilcoxon_p']
        p2 = rr['p_decontam_gt_zero']

        if p1 < 0.01 and ar['retention_fraction'] > 0.1 and p2 < 0.01:
            verdict = 'SURVIVES'
        elif (p1 < 0.05 or p2 < 0.05) and ar['retention_fraction'] > 0.05:
            verdict = 'PARTIALLY SURVIVES'
        else:
            verdict = 'LIKELY CONTAMINATION'

        table_data.append([
            GENE_INFO[gene]['name'],
            GENE_INFO[gene]['category'],
            GENE_INFO[gene]['source'],
            f"{ar['retention_fraction']:.1%}",
            f"{ar['pct_retained_cells']:.1f}%",
            f"{rr['decontam_retention']:.1%}",
            f"{rr['r_squared']:.3f}",
            verdict,
        ])

    table = ax_t.table(cellText=table_data, colLabels=headers,
                       cellLoc='center', loc='upper center',
                       colColours=['#E8E8E8'] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.6)

    # Color verdict cells
    verdict_colors = {
        'SURVIVES': '#90EE90',
        'PARTIALLY SURVIVES': '#FFF9C4',
        'LIKELY CONTAMINATION': '#FFCDD2',
    }
    last_col = len(headers) - 1
    for i, row in enumerate(table_data):
        v = row[-1]
        if v in verdict_colors:
            table[(i + 1, last_col)].set_facecolor(verdict_colors[v])

    plt.savefig(os.path.join(FIGURES_DIR, "decontamination_summary_table.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: decontamination_summary_table.png")


# ============================================================================
# Step 6: Save results
# ============================================================================

def save_results(ambient_results, regression_results):
    """Save CSV and text report."""

    sorted_genes = sorted(TARGET_GENES,
                         key=lambda g: ambient_results[g]['retention_fraction'],
                         reverse=True)

    # CSV
    rows = []
    for gene in sorted_genes:
        ar = ambient_results[gene]
        rr = regression_results[gene]

        p1 = ar['wilcoxon_p']
        p2 = rr['p_decontam_gt_zero']

        if p1 < 0.01 and ar['retention_fraction'] > 0.1 and p2 < 0.01:
            verdict = 'SURVIVES'
        elif (p1 < 0.05 or p2 < 0.05) and ar['retention_fraction'] > 0.05:
            verdict = 'PARTIALLY SURVIVES'
        else:
            verdict = 'LIKELY CONTAMINATION'

        rows.append({
            'gene': gene,
            'gene_name': GENE_INFO[gene]['name'],
            'category': GENE_INFO[gene]['category'],
            'expected_source': GENE_INFO[gene]['source'],
            'ambient_retention_fraction': ar['retention_fraction'],
            'ambient_cells_before': ar['expressing_before'],
            'ambient_cells_after': ar['expressing_after'],
            'ambient_pct_cells_retained': ar['pct_retained_cells'],
            'ambient_mean_before': ar['mean_before'],
            'ambient_mean_after': ar['mean_after'],
            'ambient_wilcoxon_p': ar['wilcoxon_p'],
            'ambient_level_source': ar['ambient_level'],
            'median_contam_fraction': ar['median_contam_fraction'],
            'regression_r_squared': rr['r_squared'],
            'regression_intercept': rr['coef_intercept'],
            'regression_intercept_p': rr['intercept_p'],
            'regression_decontam_retention': rr['decontam_retention'],
            'regression_decontam_pos_frac': rr['decontam_pos_frac'],
            'regression_p_decontam_gt_zero': rr['p_decontam_gt_zero'],
            'verdict': verdict,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "decontamination_validation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}")

    # Text report
    report_path = os.path.join(RESULTS_DIR, "DECONTAMINATION_REPORT.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DECONTAMINATION VALIDATION OF MICROGLIAL GENE EXPRESSION\n")
        f.write("=" * 80 + "\n\n")

        f.write("OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write("This analysis evaluates whether GABA and glutamate gene expression\n")
        f.write("in microglia persists after ambient RNA decontamination correction.\n\n")
        f.write("Two complementary methods are used:\n")
        f.write("  1. Ambient Profile Subtraction (SoupX-like):\n")
        f.write("     Estimates ambient RNA from source cells and per-cell contamination\n")
        f.write("     fraction from known markers, then subtracts estimated contamination.\n")
        f.write("  2. Regression-based Decontamination:\n")
        f.write("     Regresses each gene on contamination markers; residuals represent\n")
        f.write("     expression not explainable by contamination.\n\n")

        f.write("DATA\n")
        f.write("-" * 40 + "\n")
        n_micro = ambient_results[TARGET_GENES[0]]['n_micro_total']
        f.write(f"  Dataset: Zeng Aging Mouse Brain (10Xv3)\n")
        f.write(f"  Microglia analyzed: {n_micro:,}\n")
        f.write(f"  Neuronal markers: {', '.join(NEURONAL_MARKERS)}\n")
        f.write(f"  Astrocytic markers: {', '.join(ASTRO_MARKERS)}\n\n")

        f.write("VERDICT CRITERIA\n")
        f.write("-" * 40 + "\n")
        f.write("  SURVIVES: ambient retention >10%, both methods p<0.01\n")
        f.write("  PARTIALLY SURVIVES: either method p<0.05, retention >5%\n")
        f.write("  LIKELY CONTAMINATION: neither method significant\n\n")

        f.write("=" * 80 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for gene in sorted_genes:
            ar = ambient_results[gene]
            rr = regression_results[gene]
            info = GENE_INFO[gene]

            p1 = ar['wilcoxon_p']
            p2 = rr['p_decontam_gt_zero']

            if p1 < 0.01 and ar['retention_fraction'] > 0.1 and p2 < 0.01:
                verdict = 'SURVIVES'
            elif (p1 < 0.05 or p2 < 0.05) and ar['retention_fraction'] > 0.05:
                verdict = 'PARTIALLY SURVIVES'
            else:
                verdict = 'LIKELY CONTAMINATION'

            f.write(f"{info['name']} ({gene})\n")
            f.write(f"  Category: {info['category']}, Expected source: {info['source']}\n")
            f.write(f"  VERDICT: {verdict}\n\n")

            f.write(f"  Method 1 - Ambient Profile Subtraction:\n")
            f.write(f"    Expression retained: {ar['retention_fraction']:.1%}\n")
            f.write(f"    Expressing cells: {ar['expressing_before']:,} -> "
                    f"{ar['expressing_after']:,} ({ar['pct_retained_cells']:.1f}%)\n")
            f.write(f"    Mean expression: {ar['mean_before']:.3f} -> {ar['mean_after']:.3f}\n")
            f.write(f"    Ambient level (source cells): {ar['ambient_level']:.3f}\n")
            f.write(f"    Median contamination fraction: {ar['median_contam_fraction']:.3f}\n")
            f.write(f"    Wilcoxon p-value: {ar['wilcoxon_p']:.2e}\n\n")

            f.write(f"  Method 2 - Regression Decontamination:\n")
            f.write(f"    R² (variance explained by markers): {rr['r_squared']:.3f}\n")
            f.write(f"    Unexplained variance (1-R²): {1-rr['r_squared']:.3f}\n")
            f.write(f"    Intercept (baseline expression): {rr['coef_intercept']:.1f}\n")
            f.write(f"    Intercept p-value: {rr['intercept_p']:.2e}\n")
            f.write(f"    NOTE: A significant intercept (p < 0.05) indicates baseline expression\n")
            f.write(f"    not explained by contamination markers, supporting genuine expression.\n")
            f.write(f"    Decontaminated retention: {rr['decontam_retention']:.1%}\n")
            f.write(f"    Cells with positive decontam. expression: {rr['decontam_pos_frac']:.1%}\n")
            f.write(f"    p-value (decontam > 0): {rr['p_decontam_gt_zero']:.2e}\n")

            f.write(f"    Marker coefficients:\n")
            for marker, coef in sorted(rr['marker_coefficients'].items(),
                                       key=lambda x: abs(x[1]), reverse=True):
                f.write(f"      {marker:12s}: {coef:+.4f}\n")

            f.write("\n" + "-" * 40 + "\n\n")

        # Summary
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        for v_label in ['SURVIVES', 'PARTIALLY SURVIVES', 'LIKELY CONTAMINATION']:
            genes_in = [g for g in sorted_genes
                       if (lambda ar=ambient_results[g], rr=regression_results[g]:
                           'SURVIVES' if ar['wilcoxon_p'] < 0.01 and ar['retention_fraction'] > 0.1 and rr['p_decontam_gt_zero'] < 0.01
                           else 'PARTIALLY SURVIVES' if (ar['wilcoxon_p'] < 0.05 or rr['p_decontam_gt_zero'] < 0.05) and ar['retention_fraction'] > 0.05
                           else 'LIKELY CONTAMINATION')() == v_label]
            if genes_in:
                f.write(f"{v_label}:\n")
                for g in genes_in:
                    f.write(f"  {GENE_INFO[g]['name']:6s} "
                            f"ambient_retention={ambient_results[g]['retention_fraction']:.1%}, "
                            f"regr_retention={regression_results[g]['decontam_retention']:.1%}, "
                            f"R²={regression_results[g]['r_squared']:.3f}\n")
                f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("Analysis completed: February 2026\n")
        f.write("=" * 80 + "\n")

    print(f"  Report saved: {report_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("DECONTAMINATION VALIDATION OF MICROGLIAL GENE EXPRESSION")
    print("Ambient Profile Subtraction + Regression Decontamination")
    print("=" * 70)

    t_start = time.time()

    # Step 1: Load metadata
    meta, micro_mask, neuron_mask, astro_mask = load_metadata()

    # Step 2: Extract expression data
    df_micro, df_neuron, df_astro, gene_names = extract_expression(
        meta, micro_mask, neuron_mask, astro_mask)

    # Step 3: Method 1 - Ambient correction
    ambient_results = ambient_profile_correction(df_micro, df_neuron, df_astro)

    # Step 4: Method 2 - Regression decontamination
    regression_results = regression_decontamination(df_micro)

    # Step 5: Save results
    print("\nSaving results...")
    save_results(ambient_results, regression_results)

    # Step 6: Figures
    create_figures(ambient_results, regression_results,
                   df_micro, df_neuron, df_astro)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"ANALYSIS COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"\nOutputs in: {BASE_DIR}")
    print("  results/decontamination_validation_results.csv")
    print("  results/DECONTAMINATION_REPORT.txt")
    print("  figures/decontamination_validation.png")
    print("  figures/decontamination_distributions.png")
    print("  figures/decontamination_summary_table.png")


if __name__ == '__main__':
    main()
