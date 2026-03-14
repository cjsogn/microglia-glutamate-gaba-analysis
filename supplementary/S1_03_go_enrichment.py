#!/usr/bin/env python3
"""
GO Enrichment Analysis for Microglia Cluster DE Genes
=====================================================
Runs GO Biological Process, Cellular Component, and Molecular Function
enrichment on top 50 DE genes per cluster using Enrichr via gseapy.

Manuscript figures: Supplementary Fig S1
"""

import pandas as pd
import numpy as np
import gseapy as gp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import time

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Cluster names
CLUSTER_NAMES = {
    0: 'HM-Transcriptional',
    1: 'HM-Inflammatory',
    2: 'Spp1+/BAM',
    3: 'HM-Metabolic',
    4: 'HM-Motile',
    5: 'HM-Quiescent',
}

CLUSTER_COLORS = {
    'HM-Transcriptional': '#1f77b4',
    'HM-Inflammatory': '#ff7f0e',
    'Spp1+/BAM': '#2ca02c',
    'HM-Metabolic': '#d62728',
    'HM-Motile': '#9467bd',
    'HM-Quiescent': '#8c564b',
}

OUTPUT_DIR = '/Users/cjsogn/Documents/Artikkel 1/supp'
DATA_DIR = '/Users/cjsogn/zeng_microglia_analysis'

# Load DE results
df = pd.read_csv(f'{DATA_DIR}/marker_genes_allgenes.csv')

# Gene set libraries to query
libraries = [
    'GO_Biological_Process_2025',
    'GO_Cellular_Component_2025',
    'GO_Molecular_Function_2025',
]

# Run enrichment for each cluster (top 50 DE genes)
N_TOP = 50
all_results = []

for cluster_id in sorted(df['cluster'].unique()):
    cluster_name = CLUSTER_NAMES[cluster_id]
    cluster_genes = df[df['cluster'] == cluster_id].head(N_TOP)
    gene_list = cluster_genes['gene_symbol'].dropna().tolist()

    # Remove any non-standard gene names
    gene_list = [g for g in gene_list if isinstance(g, str) and len(g) > 0]

    print(f"\nCluster {cluster_id} ({cluster_name}): {len(gene_list)} genes")
    print(f"  Top 5: {gene_list[:5]}")

    for lib in libraries:
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=lib,
                organism='mouse',
                outdir=None,
                no_plot=True,
                cutoff=0.05,
            )
            res = enr.results.copy()
            res['cluster'] = cluster_id
            res['cluster_name'] = cluster_name
            res['library'] = lib
            all_results.append(res)

            sig = res[res['Adjusted P-value'] < 0.05]
            print(f"  {lib}: {len(sig)} significant terms (FDR < 0.05)")
            if len(sig) > 0:
                for _, row in sig.head(3).iterrows():
                    print(f"    {row['Term']}: p={row['Adjusted P-value']:.2e}, genes={row['Genes']}")
        except Exception as e:
            print(f"  ERROR with {lib}: {e}")

        # Small delay to be nice to Enrichr API
        time.sleep(0.5)

# Combine all results
if all_results:
    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(f'{OUTPUT_DIR}/go_enrichment_all_clusters.csv', index=False)
    print(f"\nSaved full results: {OUTPUT_DIR}/go_enrichment_all_clusters.csv")
    print(f"Total rows: {len(all_df)}")

    # Filter significant
    sig_df = all_df[all_df['Adjusted P-value'] < 0.05].copy()
    print(f"Significant terms (FDR < 0.05): {len(sig_df)}")

    # ========================================================================
    # Create summary figure: top 5 GO BP terms per cluster
    # ========================================================================
    bp_df = sig_df[sig_df['library'] == 'GO_Biological_Process_2025'].copy()

    if len(bp_df) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, cluster_id in enumerate(sorted(CLUSTER_NAMES.keys())):
            ax = axes[idx]
            cluster_name = CLUSTER_NAMES[cluster_id]
            cluster_data = bp_df[bp_df['cluster'] == cluster_id].sort_values('Adjusted P-value').head(10)

            if len(cluster_data) > 0:
                # Shorten term names
                terms = []
                for t in cluster_data['Term'].values:
                    # Remove GO ID suffix like "(GO:0006412)"
                    t_clean = t.split('(GO:')[0].strip()
                    if len(t_clean) > 45:
                        t_clean = t_clean[:42] + '...'
                    terms.append(t_clean)

                pvals = -np.log10(cluster_data['Adjusted P-value'].values)
                colors = [CLUSTER_COLORS[cluster_name]] * len(terms)

                y_pos = range(len(terms))
                ax.barh(y_pos, pvals, color=colors, edgecolor='none', height=0.7, alpha=0.85)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(terms, fontsize=7)
                ax.invert_yaxis()
                ax.set_xlabel('-log10(adj. P)', fontsize=9)
                ax.set_title(cluster_name, fontsize=11, fontweight='bold',
                           color=CLUSTER_COLORS[cluster_name])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Add significance threshold line
                ax.axvline(-np.log10(0.05), color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            else:
                ax.text(0.5, 0.5, 'No significant\nGO BP terms',
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=10, color='gray')
                ax.set_title(cluster_name, fontsize=11, fontweight='bold',
                           color=CLUSTER_COLORS[cluster_name])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        plt.suptitle('Gene Ontology Biological Process enrichment\n(top 50 DE genes per cluster)',
                    fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'{OUTPUT_DIR}/GO_enrichment_BP_all_clusters.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"\nSaved: {OUTPUT_DIR}/GO_enrichment_BP_all_clusters.png")

    # ========================================================================
    # Create focused figure for clusters 0 and 1 (all 3 GO categories)
    # ========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row_idx, cluster_id in enumerate([0, 1]):
        cluster_name = CLUSTER_NAMES[cluster_id]

        for col_idx, lib in enumerate(libraries):
            ax = axes[row_idx, col_idx]
            lib_short = lib.replace('GO_', '').replace('_2025', '').replace('_', ' ')

            cluster_data = sig_df[
                (sig_df['cluster'] == cluster_id) &
                (sig_df['library'] == lib)
            ].sort_values('Adjusted P-value').head(10)

            if len(cluster_data) > 0:
                terms = []
                for t in cluster_data['Term'].values:
                    t_clean = t.split('(GO:')[0].strip()
                    if len(t_clean) > 50:
                        t_clean = t_clean[:47] + '...'
                    terms.append(t_clean)

                pvals = -np.log10(cluster_data['Adjusted P-value'].values)

                y_pos = range(len(terms))
                ax.barh(y_pos, pvals, color=CLUSTER_COLORS[cluster_name],
                       edgecolor='none', height=0.7, alpha=0.85)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(terms, fontsize=7)
                ax.invert_yaxis()
                ax.set_xlabel('-log10(adj. P)', fontsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.axvline(-np.log10(0.05), color='gray', linestyle='--',
                          linewidth=0.5, alpha=0.5)
            else:
                ax.text(0.5, 0.5, 'No significant terms',
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=10, color='gray')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            if row_idx == 0:
                ax.set_title(lib_short, fontsize=10, fontweight='bold')

            # Row label
            if col_idx == 0:
                ax.annotate(cluster_name, xy=(-0.35, 0.5),
                          xycoords='axes fraction', fontsize=12,
                          fontweight='bold', color=CLUSTER_COLORS[cluster_name],
                          rotation=90, ha='center', va='center')

    plt.suptitle('GO enrichment for renamed clusters\n(top 50 DE genes)',
                fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    plt.savefig(f'{OUTPUT_DIR}/GO_enrichment_clusters_0_1.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/GO_enrichment_clusters_0_1.png")

    # ========================================================================
    # Print summary table for manuscript
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY: Top GO BP term per cluster (supports annotation naming)")
    print("="*80)
    for cluster_id in sorted(CLUSTER_NAMES.keys()):
        cluster_name = CLUSTER_NAMES[cluster_id]
        cluster_bp = bp_df[bp_df['cluster'] == cluster_id].sort_values('Adjusted P-value')
        print(f"\n{cluster_name} (Cluster {cluster_id}):")
        if len(cluster_bp) > 0:
            for _, row in cluster_bp.head(5).iterrows():
                term = row['Term'].split('(GO:')[0].strip()
                print(f"  {term}: adj.P={row['Adjusted P-value']:.2e}, genes: {row['Genes']}")
        else:
            print("  No significant GO BP terms")

else:
    print("\nNo results returned from Enrichr")
