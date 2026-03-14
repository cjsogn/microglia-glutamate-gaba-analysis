#!/usr/bin/env python3
"""
Contamination-Stratified Expression Analysis

Splits microglia into quartiles by contamination level (based on neuronal
and astrocytic marker expression), then tests whether target genes are
expressed even in the cleanest cells.

Logic: if cells with near-zero contamination markers still express a gene,
that expression cannot be explained by ambient RNA or doublet contamination.

Manuscript figures: Supplementary Fig S2
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, sparse
import warnings
import os
import time

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

# Contamination markers
NEURONAL_MARKERS = ['Snap25', 'Syt1', 'Rbfox3', 'Tubb3', 'Nefl', 'Nefm', 'Map2']
ASTRO_MARKERS = ['Gfap', 'Aqp4', 'Aldh1l1', 'Sox9']
ALL_CONTAM_MARKERS = NEURONAL_MARKERS + ASTRO_MARKERS
MICRO_MARKERS = ['P2ry12', 'Cx3cr1', 'Hexb', 'Csf1r', 'Tmem119', 'Aif1', 'Itgam']

ALL_GENES_NEEDED = list(set(TARGET_GENES + ALL_CONTAM_MARKERS + MICRO_MARKERS))

N_QUARTILES = 4
QUARTILE_LABELS = ['Q1\n(Cleanest)', 'Q2', 'Q3', 'Q4\n(Most contam.)']

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
# Data loading (same approach as script 01)
# ============================================================================

def load_data():
    """Load metadata and extract expression for needed genes."""
    print("Loading cell metadata...")
    t0 = time.time()

    meta = pd.read_csv(METADATA_PATH, usecols=['cell_label', 'class_name', 'subclass_name'])
    micro_mask = meta['subclass_name'].str.contains('Microglia', case=False, na=False)
    print(f"  {micro_mask.sum():,} microglia identified")

    print("Reading h5ad (extracting needed genes)...")
    with h5py.File(H5AD_PATH, 'r') as f:
        # Cell labels
        cl_group = f['obs']['cell_label']
        if 'categories' in cl_group:
            cats = np.array([c.decode() if isinstance(c, bytes) else c
                           for c in cl_group['categories'][:]])
            codes = cl_group['codes'][:]
            h5_cell_labels = cats[codes]
        else:
            h5_cell_labels = np.array([c.decode() if isinstance(c, bytes) else c
                                       for c in cl_group[:]])

        # Gene symbols
        gs_group = f['var']['gene_symbol']
        if 'categories' in gs_group:
            gs_cats = np.array([c.decode() if isinstance(c, bytes) else c
                               for c in gs_group['categories'][:]])
            gs_codes = gs_group['codes'][:]
            h5_gene_symbols = gs_cats[gs_codes]
        else:
            h5_gene_symbols = np.array([c.decode() if isinstance(c, bytes) else c
                                        for c in gs_group[:]])

        # Gene column indices
        gene_col_map = {}
        for gene in ALL_GENES_NEEDED:
            matches = np.where(h5_gene_symbols == gene)[0]
            if len(matches) > 0:
                gene_col_map[gene] = matches[0]
        gene_cols = sorted(gene_col_map.values())
        gene_names_ordered = [h5_gene_symbols[c] for c in gene_cols]
        print(f"  Found {len(gene_col_map)}/{len(ALL_GENES_NEEDED)} genes")

        # Map microglia cell labels to h5ad indices
        h5_label_to_idx = {label: i for i, label in enumerate(h5_cell_labels)}
        micro_labels = meta.loc[micro_mask, 'cell_label'].values
        micro_h5_idx = np.array([h5_label_to_idx[l] for l in micro_labels
                                 if l in h5_label_to_idx])
        print(f"  Matched {len(micro_h5_idx):,} microglia to h5ad")

        # Read sparse matrix
        print("  Reading sparse matrix...")
        data = f['X']['data'][:]
        indices = f['X']['indices'][:]
        indptr = f['X']['indptr'][:]
        shape = tuple(f['X'].attrs['shape'])

    X = sparse.csr_matrix((data, indices, indptr), shape=shape)
    del data, indices, indptr

    # Extract needed columns
    print(f"  Extracting {len(gene_cols)} gene columns for microglia...")
    X_micro = X[micro_h5_idx][:, gene_cols].toarray()
    del X

    # Reverse log2(CPM+1) transform: 2^x - 1 yields CPM values (not raw counts)
    X_linear = np.power(2.0, X_micro) - 1.0
    X_linear[X_linear < 0] = 0
    del X_micro

    df_micro = pd.DataFrame(X_linear, columns=gene_names_ordered)
    del X_linear

    print(f"  Done in {time.time()-t0:.0f}s. Shape: {df_micro.shape}")
    return df_micro


# ============================================================================
# Contamination scoring
# ============================================================================

def compute_contamination_scores(df_micro):
    """
    Compute per-cell contamination scores from marker expression.

    Returns three scores:
      - neuronal_contam: mean expression of neuronal markers
      - astro_contam: mean expression of astrocytic markers
      - total_contam: mean of all contamination markers (used for stratification)
    """
    print("\nComputing contamination scores...")

    avail_neuro = [m for m in NEURONAL_MARKERS if m in df_micro.columns]
    avail_astro = [m for m in ASTRO_MARKERS if m in df_micro.columns]

    neuro_score = df_micro[avail_neuro].mean(axis=1).values
    astro_score = df_micro[avail_astro].mean(axis=1).values
    total_score = df_micro[avail_neuro + avail_astro].mean(axis=1).values

    print(f"  Neuronal markers used: {len(avail_neuro)} ({', '.join(avail_neuro)})")
    print(f"  Astrocytic markers used: {len(avail_astro)} ({', '.join(avail_astro)})")
    print(f"  Total contamination score: mean={np.mean(total_score):.1f}, "
          f"median={np.median(total_score):.1f}, "
          f"range=[{np.min(total_score):.1f}, {np.max(total_score):.1f}]")

    return neuro_score, astro_score, total_score


# ============================================================================
# Stratified analysis
# ============================================================================

def stratified_analysis(df_micro, total_contam):
    """
    Split microglia into quartiles by contamination, analyze each gene
    within each quartile.
    """
    print("\nPerforming contamination-stratified analysis...")

    # Assign quartiles using pd.qcut (handles ties at zero correctly)
    try:
        quartile_labels_int = pd.qcut(total_contam, q=N_QUARTILES, labels=False,
                                      duplicates='drop')
        # If qcut merged bins due to ties, we may get fewer groups
        n_actual = len(np.unique(quartile_labels_int))
        if n_actual < N_QUARTILES:
            # Fallback: rank-based assignment to ensure equal groups
            ranks = stats.rankdata(total_contam, method='ordinal')
            quartile_labels_int = pd.qcut(ranks, q=N_QUARTILES, labels=False)
            n_actual = N_QUARTILES
    except ValueError:
        # Rank-based fallback
        ranks = stats.rankdata(total_contam, method='ordinal')
        quartile_labels_int = pd.qcut(ranks, q=N_QUARTILES, labels=False)
        n_actual = N_QUARTILES

    quartile_idx = quartile_labels_int.values if hasattr(quartile_labels_int, 'values') else quartile_labels_int

    # Compute actual edges from assignments
    quartile_edges = [0.0]
    for q in range(N_QUARTILES):
        q_vals = total_contam[quartile_idx == q]
        quartile_edges.append(q_vals.max())
    quartile_edges = np.array(quartile_edges)

    print(f"  Quartile boundaries (contamination score):")
    for q in range(N_QUARTILES):
        n = np.sum(quartile_idx == q)
        q_vals = total_contam[quartile_idx == q]
        print(f"    Q{q+1}: n={n:,}, contam score range=[{q_vals.min():.1f}, {q_vals.max():.1f}], "
              f"mean={q_vals.mean():.1f}")

    results = {}

    for gene in TARGET_GENES:
        info = GENE_INFO[gene]
        expr = df_micro[gene].values

        gene_result = {
            'name': info['name'],
            'category': info['category'],
            'source': info['source'],
            'quartile_data': [],
        }

        for q in range(N_QUARTILES):
            mask = quartile_idx == q
            q_expr = expr[mask]
            n_cells = len(q_expr)
            n_expressing = np.sum(q_expr > 0)
            pct_expressing = n_expressing / n_cells * 100
            mean_expr = np.mean(q_expr)
            mean_expr_positive = np.mean(q_expr[q_expr > 0]) if n_expressing > 0 else 0
            median_expr_positive = np.median(q_expr[q_expr > 0]) if n_expressing > 0 else 0

            # Per-quartile p-value removed: testing wilcoxon(q_expr[q_expr > 0])
            # is tautological since positive values are always > 0.
            # The key test is the binomial test on Q1 expression rate (below).
            p_val = np.nan

            gene_result['quartile_data'].append({
                'quartile': q + 1,
                'n_cells': n_cells,
                'n_expressing': n_expressing,
                'pct_expressing': pct_expressing,
                'mean_expr': mean_expr,
                'mean_expr_positive': mean_expr_positive,
                'median_expr_positive': median_expr_positive,
                'p_value': p_val,
            })

        # Test: is the expression rate in Q1 significantly different from 0?
        q1_mask = quartile_idx == 0
        q1_expr = expr[q1_mask]
        q1_expressing = np.sum(q1_expr > 0)
        # Binomial test: is the expressing fraction different from what we'd
        # expect under pure noise? Use a very conservative null of 1%
        # Binomial test: is Q1 expression rate significantly above a conservative
        # null of 1%? Tests whether cleanest cells express at rates exceeding noise.
        try:
            binom_result = stats.binomtest(q1_expressing, len(q1_expr), 0.01,
                                           alternative='greater')
            binom_p = binom_result.pvalue
        except AttributeError:
            binom_p = stats.binom_test(q1_expressing, len(q1_expr), 0.01,
                                       alternative='greater')

        # Trend test: does expression decrease with contamination quartile?
        # Spearman correlation between quartile and expression rate
        quartile_pcts = [gene_result['quartile_data'][q]['pct_expressing'] for q in range(N_QUARTILES)]
        if len(set(quartile_pcts)) > 1:
            trend_r, trend_p = stats.spearmanr([1, 2, 3, 4], quartile_pcts)
        else:
            trend_r, trend_p = 0, 1.0

        gene_result['q1_expressing'] = q1_expressing
        gene_result['q1_total'] = int(np.sum(q1_mask))
        gene_result['q1_pct'] = q1_expressing / np.sum(q1_mask) * 100
        gene_result['q1_binom_p'] = binom_p
        gene_result['trend_r'] = trend_r
        gene_result['trend_p'] = trend_p

        # Verdict
        if gene_result['q1_pct'] > 5 and binom_p < 0.001:
            verdict = 'AUTHENTIC'
        elif gene_result['q1_pct'] > 1 and binom_p < 0.01:
            verdict = 'LIKELY AUTHENTIC'
        else:
            verdict = 'INCONCLUSIVE'
        gene_result['verdict'] = verdict

        results[gene] = gene_result

        print(f"  {info['name']:6s}: Q1={gene_result['q1_pct']:.1f}% expressing "
              f"(n={q1_expressing:,}/{gene_result['q1_total']:,}), "
              f"binom p={binom_p:.2e}, "
              f"trend r={trend_r:+.2f} -> {verdict}")

    return results, quartile_idx, quartile_edges


# ============================================================================
# Figures
# ============================================================================

def create_figures(df_micro, results, quartile_idx, quartile_edges, total_contam):
    """Publication-quality figures."""
    print("\nGenerating figures...")

    sorted_genes = sorted(TARGET_GENES,
                         key=lambda g: results[g]['q1_pct'], reverse=True)

    # ======================================================================
    # Figure 1: Main result - Expression rate across contamination quartiles
    # ======================================================================
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 4.0),
                             gridspec_kw={'hspace': 0.55, 'wspace': 0.35})

    for idx, gene in enumerate(sorted_genes):
        ax = axes[idx // 5, idx % 5]
        info = GENE_INFO[gene]
        res = results[gene]

        pcts = [res['quartile_data'][q]['pct_expressing'] for q in range(N_QUARTILES)]
        means = [res['quartile_data'][q]['mean_expr'] for q in range(N_QUARTILES)]

        color = '#C0392B' if info['category'] == 'GABA' else '#2471A3'

        bars = ax.bar(range(N_QUARTILES), pcts, color=color, alpha=0.8,
                      edgecolor='black', linewidth=0.3, width=0.7)

        # Highlight Q1 (cleanest)
        bars[0].set_edgecolor('gold')
        bars[0].set_linewidth(1.5)

        # Annotate Q1 percentage
        ax.text(0, pcts[0] + 1.5, f'{pcts[0]:.1f}%', ha='center', fontsize=5,
                fontweight='bold', color='black')

        # Annotate counts in each bar
        for q in range(N_QUARTILES):
            n_exp = res['quartile_data'][q]['n_expressing']
            if pcts[q] > 5:
                ax.text(q, pcts[q] / 2, f'{n_exp:,}', ha='center', va='center',
                        fontsize=4, color='white', fontweight='bold')

        ax.set_xticks(range(N_QUARTILES))
        ax.set_xticklabels(['Q1\nClean', 'Q2', 'Q3', 'Q4\nContam.'], fontsize=4.5)

        # Significance stars for Q1
        p = res['q1_binom_p']
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

        verdict = res['verdict']
        ax.set_title(f'{info["name"]} {stars}', fontsize=7, fontweight='bold', color=color)

        if idx % 5 == 0:
            ax.set_ylabel('% Expressing', fontsize=6)

        # Trend arrow
        if res['trend_p'] < 0.05:
            if res['trend_r'] > 0:
                ax.annotate('', xy=(3.3, max(pcts)*0.9), xytext=(3.3, max(pcts)*0.4),
                           arrowprops=dict(arrowstyle='->', color='red', lw=1))
            else:
                ax.annotate('', xy=(3.3, max(pcts)*0.4), xytext=(3.3, max(pcts)*0.9),
                           arrowprops=dict(arrowstyle='->', color='green', lw=1))

    plt.savefig(os.path.join(FIGURES_DIR, "contamination_stratified_expression_rate.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: contamination_stratified_expression_rate.png")

    # ======================================================================
    # Figure 2: Mean expression in expressing cells across quartiles
    # ======================================================================
    fig2, axes2 = plt.subplots(2, 5, figsize=(7.2, 4.0),
                               gridspec_kw={'hspace': 0.55, 'wspace': 0.4})

    for idx, gene in enumerate(sorted_genes):
        ax = axes2[idx // 5, idx % 5]
        info = GENE_INFO[gene]
        res = results[gene]
        color = '#C0392B' if info['category'] == 'GABA' else '#2471A3'

        # Violin plot per quartile (expressing cells only)
        data_per_q = []
        for q in range(N_QUARTILES):
            mask = quartile_idx == q
            expr = df_micro[gene].values[mask]
            expr_pos = expr[expr > 0]
            # Subsample for plotting
            if len(expr_pos) > 2000:
                rng = np.random.default_rng(42 + idx)
                expr_pos = rng.choice(expr_pos, 2000, replace=False)
            data_per_q.append(expr_pos if len(expr_pos) > 0 else np.array([0]))

        parts = ax.violinplot(data_per_q, showmeans=True, showmedians=False,
                              showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            alpha = 0.9 if i == 0 else 0.5 + i * 0.1
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            if i == 0:
                pc.set_edgecolor('gold')
                pc.set_linewidth(1.0)
            else:
                pc.set_edgecolor('black')
                pc.set_linewidth(0.3)
        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(0.8)

        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['Q1\nClean', 'Q2', 'Q3', 'Q4\nContam.'], fontsize=4.5)

        ax.set_title(f'{info["name"]}', fontsize=7, fontweight='bold', color=color)
        if idx % 5 == 0:
            ax.set_ylabel('Expression (CPM)\n(expressing cells)', fontsize=5.5)

    plt.savefig(os.path.join(FIGURES_DIR, "contamination_stratified_expression_level.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: contamination_stratified_expression_level.png")

    # ======================================================================
    # Figure 3: Summary - Q1 expression rate bar chart (the key result)
    # ======================================================================
    fig3 = plt.figure(figsize=(7.2, 3.5))
    gs = fig3.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.35,
                           left=0.08, right=0.95, top=0.88, bottom=0.15)

    # Panel A: Q1 expression rates
    ax_a = fig3.add_subplot(gs[0])

    q1_pcts = [results[g]['q1_pct'] for g in sorted_genes]
    names = [GENE_INFO[g]['name'] for g in sorted_genes]
    bar_colors = ['#C0392B' if GENE_INFO[g]['category'] == 'GABA' else '#2471A3'
                  for g in sorted_genes]

    y_pos = np.arange(len(sorted_genes))
    bars = ax_a.barh(y_pos, q1_pcts, color=bar_colors, alpha=0.85,
                     edgecolor='black', linewidth=0.3, height=0.7)

    for i, (val, gene) in enumerate(zip(q1_pcts, sorted_genes)):
        n = results[gene]['q1_expressing']
        total = results[gene]['q1_total']
        p = results[gene]['q1_binom_p']
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        ax_a.text(val + 0.5, i,
                  f'{val:.1f}% ({n:,}/{total:,}) {stars}',
                  va='center', fontsize=5)

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(names, fontsize=7)
    ax_a.set_xlabel('% Expressing in Cleanest Quartile (Q1)', fontsize=7)
    ax_a.invert_yaxis()
    ax_a.set_title('A', loc='left', fontsize=10, fontweight='bold', x=-0.12, y=1.02)

    # Add text about Q1
    q1_contam_range = f"Q1 contamination score: {quartile_edges[0]:.0f}-{quartile_edges[1]:.0f}"
    ax_a.text(0.98, 0.02, q1_contam_range, transform=ax_a.transAxes,
              fontsize=5, ha='right', va='bottom', style='italic', color='gray')

    # Panel B: Expression rate ratio Q1/Q4
    ax_b = fig3.add_subplot(gs[1])

    ratios = []
    for gene in sorted_genes:
        q1 = results[gene]['quartile_data'][0]['pct_expressing']
        q4 = results[gene]['quartile_data'][3]['pct_expressing']
        ratio = q1 / q4 if q4 > 0 else np.inf
        ratios.append(ratio)

    bars_b = ax_b.barh(y_pos, ratios, color=bar_colors, alpha=0.85,
                       edgecolor='black', linewidth=0.3, height=0.7)

    for i, val in enumerate(ratios):
        if np.isfinite(val):
            ax_b.text(val + 0.02, i, f'{val:.2f}', va='center', fontsize=5.5)

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(names, fontsize=7)
    ax_b.set_xlabel('Expression Rate Ratio (Q1/Q4)', fontsize=7)
    ax_b.axvline(1.0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax_b.invert_yaxis()
    ax_b.set_title('B', loc='left', fontsize=10, fontweight='bold', x=-0.12, y=1.02)

    # Interpretation text
    ax_b.text(0.98, 0.02, 'Ratio > 1: more expression\nin clean cells',
              transform=ax_b.transAxes, fontsize=4.5, ha='right', va='bottom',
              style='italic', color='gray')

    plt.savefig(os.path.join(FIGURES_DIR, "contamination_stratified_summary.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: contamination_stratified_summary.png")

    # ======================================================================
    # Figure 4: Summary table
    # ======================================================================
    fig_t, ax_t = plt.subplots(figsize=(7.2, 3.8))
    ax_t.axis('off')

    headers = ['Gene', 'Cat.', 'Q1 (Clean)\n% Expr.', 'Q2\n% Expr.',
               'Q3\n% Expr.', 'Q4 (Contam.)\n% Expr.', 'Q1/Q4\nRatio',
               'Trend', 'Verdict']

    table_data = []
    for gene in sorted_genes:
        res = results[gene]
        pcts = [res['quartile_data'][q]['pct_expressing'] for q in range(N_QUARTILES)]
        ratio = pcts[0] / pcts[3] if pcts[3] > 0 else np.inf

        trend_str = f"r={res['trend_r']:+.2f}" if res['trend_p'] < 0.05 else 'n.s.'

        table_data.append([
            GENE_INFO[gene]['name'],
            GENE_INFO[gene]['category'],
            f"{pcts[0]:.1f}%",
            f"{pcts[1]:.1f}%",
            f"{pcts[2]:.1f}%",
            f"{pcts[3]:.1f}%",
            f"{ratio:.2f}" if np.isfinite(ratio) else "inf",
            trend_str,
            res['verdict'],
        ])

    table = ax_t.table(cellText=table_data, colLabels=headers,
                       cellLoc='center', loc='upper center',
                       colColours=['#E8E8E8'] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.6)

    # Color verdict cells
    last_col = len(headers) - 1
    for i, row in enumerate(table_data):
        v = row[-1]
        if v == 'AUTHENTIC':
            table[(i + 1, last_col)].set_facecolor('#90EE90')
        elif v == 'LIKELY AUTHENTIC':
            table[(i + 1, last_col)].set_facecolor('#C8E6C9')
        else:
            table[(i + 1, last_col)].set_facecolor('#FFF9C4')

    plt.savefig(os.path.join(FIGURES_DIR, "contamination_stratified_table.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: contamination_stratified_table.png")


# ============================================================================
# Save results
# ============================================================================

def save_results(results, quartile_edges, total_contam, quartile_idx):
    """Save CSV and text report."""

    sorted_genes = sorted(TARGET_GENES,
                         key=lambda g: results[g]['q1_pct'], reverse=True)

    # CSV
    rows = []
    for gene in sorted_genes:
        res = results[gene]
        row = {
            'gene': gene,
            'gene_name': res['name'],
            'category': res['category'],
            'expected_source': res['source'],
        }
        for q in range(N_QUARTILES):
            qd = res['quartile_data'][q]
            row[f'q{q+1}_n_cells'] = qd['n_cells']
            row[f'q{q+1}_n_expressing'] = qd['n_expressing']
            row[f'q{q+1}_pct_expressing'] = qd['pct_expressing']
            row[f'q{q+1}_mean_expr'] = qd['mean_expr']
            row[f'q{q+1}_mean_expr_positive'] = qd['mean_expr_positive']
            row[f'q{q+1}_p_value'] = qd['p_value']

        q1 = res['quartile_data'][0]['pct_expressing']
        q4 = res['quartile_data'][3]['pct_expressing']
        row['q1_q4_ratio'] = q1 / q4 if q4 > 0 else np.inf
        row['q1_binom_p'] = res['q1_binom_p']
        row['trend_r'] = res['trend_r']
        row['trend_p'] = res['trend_p']
        row['verdict'] = res['verdict']
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "contamination_stratified_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}")

    # Text report
    report_path = os.path.join(RESULTS_DIR, "CONTAMINATION_STRATIFIED_REPORT.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CONTAMINATION-STRATIFIED EXPRESSION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write("RATIONALE\n")
        f.write("-" * 40 + "\n")
        f.write("If microglial expression of a gene is driven by ambient RNA\n")
        f.write("contamination from neurons or astrocytes, then cells with the\n")
        f.write("lowest contamination marker expression should show the least\n")
        f.write("target gene expression. Conversely, if the cleanest microglia\n")
        f.write("still express the gene at substantial rates, the expression\n")
        f.write("cannot be explained by contamination.\n\n")

        f.write("METHOD\n")
        f.write("-" * 40 + "\n")
        f.write(f"Microglia were split into {N_QUARTILES} quartiles by a composite\n")
        f.write("contamination score (mean expression of neuronal + astrocytic\n")
        f.write("contamination markers).\n\n")
        f.write(f"Neuronal markers: {', '.join(NEURONAL_MARKERS)}\n")
        f.write(f"Astrocytic markers: {', '.join(ASTRO_MARKERS)}\n\n")
        f.write("Quartile boundaries (contamination score):\n")
        for q in range(N_QUARTILES):
            n = int(np.sum(quartile_idx == q))
            f.write(f"  Q{q+1}: [{quartile_edges[q]:.1f}, {quartile_edges[q+1]:.1f}] "
                    f"(n={n:,})\n")
        f.write("\n")

        f.write("VERDICT CRITERIA\n")
        f.write("-" * 40 + "\n")
        f.write("AUTHENTIC: >5% expressing in Q1 (cleanest), binomial p < 0.001\n")
        f.write("LIKELY AUTHENTIC: >1% expressing in Q1, binomial p < 0.01\n")
        f.write("INCONCLUSIVE: otherwise\n\n")

        f.write("=" * 80 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for gene in sorted_genes:
            res = results[gene]
            f.write(f"{res['name']} ({gene})\n")
            f.write(f"  Category: {res['category']}, Source: {res['source']}\n")
            f.write(f"  VERDICT: {res['verdict']}\n\n")

            f.write(f"  Expression rate by quartile:\n")
            for q in range(N_QUARTILES):
                qd = res['quartile_data'][q]
                label = "CLEANEST" if q == 0 else "MOST CONTAMINATED" if q == 3 else ""
                f.write(f"    Q{q+1}: {qd['pct_expressing']:.1f}% "
                        f"({qd['n_expressing']:,}/{qd['n_cells']:,}) "
                        f"mean={qd['mean_expr']:.1f}")
                if label:
                    f.write(f"  [{label}]")
                f.write("\n")

            q1 = res['quartile_data'][0]['pct_expressing']
            q4 = res['quartile_data'][3]['pct_expressing']
            ratio = q1 / q4 if q4 > 0 else np.inf
            f.write(f"\n  Q1/Q4 ratio: {ratio:.2f}\n")
            f.write(f"  Q1 binomial test: p={res['q1_binom_p']:.2e}\n")
            f.write(f"  Trend across quartiles: r={res['trend_r']:+.3f}, p={res['trend_p']:.3f}\n")
            f.write("\n" + "-" * 40 + "\n\n")

        # Summary
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        for v in ['AUTHENTIC', 'LIKELY AUTHENTIC', 'INCONCLUSIVE']:
            genes_in = [g for g in sorted_genes if results[g]['verdict'] == v]
            if genes_in:
                f.write(f"{v}:\n")
                for g in genes_in:
                    f.write(f"  {results[g]['name']:6s} Q1={results[g]['q1_pct']:.1f}%\n")
                f.write("\n")

        f.write("INTERPRETATION\n")
        f.write("-" * 40 + "\n")
        f.write("Genes with substantial expression in the cleanest quartile (Q1)\n")
        f.write("are unlikely to be explained solely by ambient RNA or doublet\n")
        f.write("contamination from source cells. The contamination markers in\n")
        f.write("Q1 cells are at near-zero levels, yet the target genes remain\n")
        f.write("expressed, consistent with cell-intrinsic transcription.\n\n")

        f.write("=" * 80 + "\n")
        f.write("Analysis completed: February 2026\n")
        f.write("=" * 80 + "\n")

    print(f"  Report saved: {report_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("CONTAMINATION-STRATIFIED EXPRESSION ANALYSIS")
    print("Do the cleanest microglia still express the target genes?")
    print("=" * 70)

    t_start = time.time()

    # Load data
    df_micro = load_data()

    # Compute contamination scores
    neuro_score, astro_score, total_score = compute_contamination_scores(df_micro)

    # Stratified analysis
    results, quartile_idx, quartile_edges = stratified_analysis(df_micro, total_score)

    # Save results
    print("\nSaving results...")
    save_results(results, quartile_edges, total_score, quartile_idx)

    # Figures
    create_figures(df_micro, results, quartile_idx, quartile_edges, total_score)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"ANALYSIS COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"\nOutputs:")
    print("  results/contamination_stratified_results.csv")
    print("  results/CONTAMINATION_STRATIFIED_REPORT.txt")
    print("  figures/contamination_stratified_expression_rate.png")
    print("  figures/contamination_stratified_expression_level.png")
    print("  figures/contamination_stratified_summary.png")
    print("  figures/contamination_stratified_table.png")


if __name__ == '__main__':
    main()
