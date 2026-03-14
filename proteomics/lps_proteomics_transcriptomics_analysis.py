#!/usr/bin/env python3
"""
Forest Plot: LPS Effects on Glutamate/GABA Genes (Proteomics + Transcriptomics)
================================================================================
Generates forest plots comparing LPS-induced changes in glutamate and GABA
pathway genes across transcriptomics (scRNA-seq, 3d and 30d) and proteomics
(TMT mass spectrometry, 4d) datasets.

Manuscript figures: Fig 5G-J
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D

# Style settings - even larger fonts to match violin plots (C and F)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Figure size for larger subplots
FIGURE_SIZE = (5.5, 4.5)

# Color scheme - base colors
COLORS = {
    'grid': '#ECEFF1',
    'text': '#212121',
    'sig': '#1A1A1A'
}

# GABA colors - shades of red
GABA_COLORS = {
    'acute': '#C62828',      # Dark red (LPS 3d)
    'chronic': '#EF5350',    # Light red (LPS 30d)
    'single': '#D32F2F',     # Medium red (proteomics single timepoint)
    'up_bg': '#FFEBEE',      # Light red background
    'down_bg': '#FFCDD2',    # Lighter red background
}

# Glutamate colors - shades of blue
GLUT_COLORS = {
    'acute': '#1565C0',      # Dark blue (LPS 3d)
    'chronic': '#64B5F6',    # Light blue (LPS 30d)
    'single': '#1976D2',     # Medium blue (proteomics single timepoint)
    'up_bg': '#E3F2FD',      # Light blue background
    'down_bg': '#BBDEFB',    # Lighter blue background
}

# Data paths
RNA_PATH = '/Users/cjsogn/geo_downloads/GSE307796/differential_expression_results.csv'
PROT_PATH = '/Users/cjsogn/rangaraju_microglia_proteomics/PerseusExport_proteomicRulerTransformation.txt'
OUTPUT_DIR = '/Users/cjsogn/rangaraju_microglia_proteomics/'

# Proteomics sample columns
CONTROL_COLS = ['B6.1', 'B6.2', 'B6.3']
LPS_COLS = ['LPS1a', 'LPS1b', 'LPS2']


def load_data():
    """Load transcriptomics and proteomics data"""
    rna_df = pd.read_csv(RNA_PATH)
    prot_df = pd.read_csv(PROT_PATH, sep='\t', header=0, skiprows=[1, 2])
    return rna_df, prot_df


def get_prot_data(prot_df, gene):
    """Extract proteomics data for a specific gene"""
    prot_symbol = prot_df['Symbol'].astype(str).str.lower()
    matches = prot_df[prot_symbol == gene.lower()]
    if len(matches) > 0:
        row = matches.iloc[0]
        return row[CONTROL_COLS].values.astype(float), row[LPS_COLS].values.astype(float)
    return None, None


def create_forest_plot(ax, genes, labels, data_list, title, color_scheme, legend_loc='lower right'):
    """
    Create a forest plot for gene expression changes

    Parameters:
    -----------
    ax : matplotlib axis
    genes : list of gene symbols
    labels : list of display labels
    data_list : dict with gene -> list of (log2fc, ci_low, ci_high, sig, condition, color, marker)
    title : plot title
    color_scheme : dict with color keys (acute, chronic, single, up_bg, down_bg)
    legend_loc : str, legend location (default 'lower right')
    """
    y_positions = np.arange(len(genes))[::-1]

    # Compute xlim from data so all CIs and significance labels fit
    all_vals = []
    for gene in genes:
        for (log2fc, ci_low, ci_high, sig, *_rest) in data_list[gene]:
            all_vals.extend([ci_low, ci_high])
            # Account for significance text width
            if sig and sig != 'ns':
                extra = 0.35 + 0.08 * len(sig)
                if log2fc >= 0:
                    all_vals.append(ci_high + extra)
                else:
                    all_vals.append(ci_low - extra)
    xlim = max(1.5, max(abs(v) for v in all_vals) + 0.15)

    # Background shading using color scheme
    ax.axvspan(-xlim, 0, alpha=0.15, color=color_scheme['down_bg'], zorder=0)
    ax.axvspan(0, xlim, alpha=0.15, color=color_scheme['up_bg'], zorder=0)
    ax.axvline(x=0, color=COLORS['text'], linewidth=1.5, zorder=1)
    
    # Horizontal grid lines
    for y in y_positions:
        ax.axhline(y=y, color=COLORS['grid'], linewidth=0.5, zorder=0, alpha=0.5)
    
    legend_elements = []
    legend_added = set()
    
    for i, (gene, label) in enumerate(zip(genes, labels)):
        y = y_positions[i]
        gene_data = data_list[gene]
        
        for j, (log2fc, ci_low, ci_high, sig, condition, color, marker) in enumerate(gene_data):
            y_offset = 0.15 * (j - (len(gene_data) - 1) / 2)
            
            # Confidence interval line
            ax.plot([ci_low, ci_high], [y + y_offset, y + y_offset],
                    color=color, linewidth=2, alpha=0.6, zorder=2)
            
            # Point estimate
            ax.scatter(log2fc, y + y_offset, s=150, c=color, marker=marker,
                       edgecolor='white', linewidth=1.5, zorder=3)
            
            # Significance annotation
            if sig and sig != 'ns':
                text_x = ci_high + 0.08 if log2fc >= 0 else ci_low - 0.08
                ha = 'left' if log2fc >= 0 else 'right'
                ax.text(text_x, y + y_offset, sig, fontsize=16, fontweight='bold',
                        color=COLORS['sig'], va='center', ha=ha)
            
            # Legend entry
            if condition not in legend_added:
                legend_elements.append(
                    Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,
                           markersize=10, markeredgecolor='white', markeredgewidth=1,
                           label=condition))
                legend_added.add(condition)
    
    # Axis settings - even larger fonts to match violin plots C and F
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=18, fontweight='bold')
    ax.set_xlabel('Log2 Fold Change', fontsize=16, fontweight='bold')
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-0.5, len(genes) - 0.5)
    # Title removed - will be added in composite figure for consistency
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Direction labels removed per user request

    # Legend
    ax.legend(handles=legend_elements, loc=legend_loc, frameon=True,
              fancybox=True, shadow=False, fontsize=14,
              edgecolor=COLORS['grid'], facecolor='white')
    
    return ax


def main():
    # Load data
    rna_df, prot_df = load_data()
    
    # Gene lists
    glutamate_genes_rna = ['Gls', 'Glul', 'Slc1a3', 'Slc1a2', 'Slc38a1']
    glutamate_labels_rna = ['GLS', 'GS', 'EAAT1', 'EAAT2', 'SNAT1']
    
    glutamate_genes_prot = ['Gls', 'Glul', 'Slc1a3', 'Slc1a2']
    glutamate_labels_prot = ['GLS', 'GS', 'EAAT1', 'EAAT2']
    
    gaba_genes_rna = ['Gad1', 'Gad2', 'Abat', 'Slc6a1']
    gaba_labels_rna = ['GAD67', 'GAD65', 'ABAT', 'GAT1']
    
    gaba_genes_prot = ['Gad1', 'Gad2', 'Abat', 'Slc6a1', 'Slc6a11']
    gaba_labels_prot = ['GAD67', 'GAD65', 'ABAT', 'GAT1', 'GAT3']
    
    # ============================================
    # FIGURE 1: GLUTAMATE TRANSCRIPTOMICS
    # ============================================
    fig1, ax1 = plt.subplots(figsize=FIGURE_SIZE)
    
    data_glu_rna = {}
    for gene in glutamate_genes_rna:
        row = rna_df[rna_df['gene'] == gene]
        fc_3d = row['log2FC_3d'].values[0]
        fc_30d = row['log2FC_30d'].values[0]
        sig_3d = row['sig_3d'].values[0]
        sig_30d = row['sig_30d'].values[0]
        data_glu_rna[gene] = [
            (fc_3d, fc_3d, fc_3d, sig_3d, 'LPS 3d', GLUT_COLORS['acute'], 'o'),
            (fc_30d, fc_30d, fc_30d, sig_30d, 'LPS 30d', GLUT_COLORS['chronic'], 's')
        ]

    create_forest_plot(ax1, glutamate_genes_rna, glutamate_labels_rna, data_glu_rna,
                       'Glutamate Genes — Transcriptomics', GLUT_COLORS, legend_loc='lower left')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fig1_glutamate_transcriptomics_final.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: fig1_glutamate_transcriptomics_final.png")
    
    # ============================================
    # FIGURE 2: GLUTAMATE PROTEOMICS
    # ============================================
    fig2, ax2 = plt.subplots(figsize=FIGURE_SIZE)
    
    data_glu_prot = {}
    for gene in glutamate_genes_prot:
        ctrl_vals, lps_vals = get_prot_data(prot_df, gene)
        log2fc = np.log2(np.mean(lps_vals) / np.mean(ctrl_vals))
        fc_vals = lps_vals / ctrl_vals.mean()
        log2_fc_vals = np.log2(fc_vals)
        sem = stats.sem(log2_fc_vals)
        t_crit = stats.t.ppf(0.975, df=len(log2_fc_vals) - 1)
        ci_low = log2fc - t_crit * sem
        ci_high = log2fc + t_crit * sem
        _, p = stats.ttest_ind(lps_vals, ctrl_vals)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        data_glu_prot[gene] = [
            (log2fc, ci_low, ci_high, sig, 'LPS 4d', GLUT_COLORS['single'], 'o')
        ]

    create_forest_plot(ax2, glutamate_genes_prot, glutamate_labels_prot, data_glu_prot,
                       'Glutamate Genes — Proteomics', GLUT_COLORS, legend_loc='lower left')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fig2_glutamate_proteomics_final.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: fig2_glutamate_proteomics_final.png")
    
    # ============================================
    # FIGURE 3: GABA TRANSCRIPTOMICS
    # ============================================
    fig3, ax3 = plt.subplots(figsize=FIGURE_SIZE)
    
    data_gaba_rna = {}
    for gene in gaba_genes_rna:
        row = rna_df[rna_df['gene'] == gene]
        fc_3d = row['log2FC_3d'].values[0]
        fc_30d = row['log2FC_30d'].values[0]
        sig_3d = row['sig_3d'].values[0]
        sig_30d = row['sig_30d'].values[0]
        data_gaba_rna[gene] = [
            (fc_3d, fc_3d, fc_3d, sig_3d, 'LPS 3d', GABA_COLORS['acute'], 'o'),
            (fc_30d, fc_30d, fc_30d, sig_30d, 'LPS 30d', GABA_COLORS['chronic'], 's')
        ]

    create_forest_plot(ax3, gaba_genes_rna, gaba_labels_rna, data_gaba_rna,
                       'GABA Genes — Transcriptomics', GABA_COLORS)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fig3_gaba_transcriptomics_final.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: fig3_gaba_transcriptomics_final.png")
    
    # ============================================
    # FIGURE 4: GABA PROTEOMICS
    # ============================================
    fig4, ax4 = plt.subplots(figsize=FIGURE_SIZE)
    
    data_gaba_prot = {}
    for gene in gaba_genes_prot:
        ctrl_vals, lps_vals = get_prot_data(prot_df, gene)
        log2fc = np.log2(np.mean(lps_vals) / np.mean(ctrl_vals))
        fc_vals = lps_vals / ctrl_vals.mean()
        log2_fc_vals = np.log2(fc_vals)
        sem = stats.sem(log2_fc_vals)
        t_crit = stats.t.ppf(0.975, df=len(log2_fc_vals) - 1)
        ci_low = log2fc - t_crit * sem
        ci_high = log2fc + t_crit * sem
        _, p = stats.ttest_ind(lps_vals, ctrl_vals)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        data_gaba_prot[gene] = [
            (log2fc, ci_low, ci_high, sig, 'LPS 4d', GABA_COLORS['single'], 'o')
        ]

    create_forest_plot(ax4, gaba_genes_prot, gaba_labels_prot, data_gaba_prot,
                       'GABA Genes — Proteomics', GABA_COLORS)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fig4_gaba_proteomics_final.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: fig4_gaba_proteomics_final.png")
    
    print("\n✓ All 4 figures generated!")


if __name__ == '__main__':
    main()
