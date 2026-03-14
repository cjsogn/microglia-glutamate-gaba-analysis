#!/usr/bin/env python3
"""
Differential Expression Analysis on All Genes
==============================================
Runs Wilcoxon rank-sum tests (one-vs-rest) on ALL 32,285 genes (not just HVGs)
for each of the 6 microglia clusters. Maps Ensembl IDs to gene symbols via
the mygene.info API.

Manuscript figures: Supplementary Fig S1

Requires internet access for the mygene.info gene symbol lookup.

Outputs:
  - marker_genes_allgenes.csv: Top 100 DE genes per cluster with
    scores, adjusted p-values, log fold changes, and gene symbols.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import requests
import warnings
import gc

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# =============================================================================
# STEP 1: Re-run DE on all genes
# =============================================================================
print("=" * 70)
print("STEP 1: Re-running differential expression on ALL 32,285 genes")
print("=" * 70)

BASE_DIR = '/Users/cjsogn/zeng_microglia_analysis'
OUTPUT_DIR = '/Users/cjsogn/Documents/Artikkel 1/supp'

# Load raw data (all genes)
print("Loading raw h5ad (all genes)...")
adata_raw = ad.read_h5ad(f'{BASE_DIR}/microglia_extracted_raw.h5ad')
print(f"  Raw shape: {adata_raw.shape}")

# Load annotated data to get cluster labels
print("Loading annotated h5ad (cluster labels)...")
adata_ann = ad.read_h5ad(f'{BASE_DIR}/microglia_scvi_annotated.h5ad', backed='r')
leiden_labels = adata_ann.obs['leiden'].copy()
adata_ann.file.close()
del adata_ann
gc.collect()

# Transfer cluster labels to raw data
# Verify cell order matches
assert (adata_raw.obs.index == leiden_labels.index).all(), "Cell indices don't match!"
adata_raw.obs['leiden'] = leiden_labels
print(f"  Cluster labels transferred")
print(f"  Clusters: {adata_raw.obs['leiden'].value_counts().sort_index().to_dict()}")

# The source data is in log2(CPM+1) units; convert back to linear CPM scale
# From methods: X = 2^y - 1
print("Converting log2(CPM+1) data to linear CPM scale...")
adata_raw.X = np.expm1(adata_raw.X * np.log(2))  # 2^y - 1 = exp(y*ln2) - 1
# NOTE: The linear values are CPM (counts per million), not raw counts.
# Re-normalizing to CP10K and log1p below makes downstream DE results
# independent of the original CPM normalization.

# Normalize and log-transform for DE
print("Normalizing for DE analysis...")
sc.pp.normalize_total(adata_raw, target_sum=1e4)
sc.pp.log1p(adata_raw)

# Run Wilcoxon rank-sum DE on ALL genes
print("Running Wilcoxon rank-sum test (all genes, one-vs-rest)...")
print("  This may take a few minutes with 32,285 genes and 81,472 cells...")
sc.tl.rank_genes_groups(adata_raw, 'leiden', method='wilcoxon', n_genes=100)

# Build marker gene dataframe with gene symbol conversion
print("Building marker gene table...")

# Collect all unique Ensembl IDs from top 100 per cluster
all_ensembl_ids = set()
for i in range(6):
    c = str(i)
    for j in range(100):
        all_ensembl_ids.add(adata_raw.uns['rank_genes_groups']['names'][c][j])

all_ensembl_ids = list(all_ensembl_ids)
print(f"  Converting {len(all_ensembl_ids)} Ensembl IDs to gene symbols...")

# Query mygene in batches
gene_mapping = {}
batch_size = 1000
for i in range(0, len(all_ensembl_ids), batch_size):
    batch = all_ensembl_ids[i:i+batch_size]
    url = "http://mygene.info/v3/query"
    params = {
        'q': ','.join(batch),
        'scopes': 'ensembl.gene',
        'fields': 'symbol',
        'species': 'mouse'
    }
    try:
        response = requests.post(url, data=params, timeout=120)
        results = response.json()
        for result in results:
            if 'symbol' in result and 'query' in result:
                gene_mapping[result['query']] = result['symbol']
    except Exception as e:
        print(f"  API error: {e}")

# Fill in missing
for eid in all_ensembl_ids:
    if eid not in gene_mapping:
        gene_mapping[eid] = eid

print(f"  Mapped {sum(1 for v in gene_mapping.values() if not v.startswith('ENSMUSG'))} / {len(gene_mapping)} to symbols")

# Build marker dataframe
markers = []
for i in range(6):
    c = str(i)
    for j in range(100):
        eid = adata_raw.uns['rank_genes_groups']['names'][c][j]
        markers.append({
            'cluster': i,
            'gene': eid,
            'gene_symbol': gene_mapping.get(eid, eid),
            'score': float(adata_raw.uns['rank_genes_groups']['scores'][c][j]),
            'pval_adj': float(adata_raw.uns['rank_genes_groups']['pvals_adj'][c][j]),
            'logfoldchange': float(adata_raw.uns['rank_genes_groups']['logfoldchanges'][c][j]),
            'pct_expressing': float(adata_raw.uns['rank_genes_groups']['pts'][c][j]) if 'pts' in adata_raw.uns['rank_genes_groups'] else np.nan,
        })

marker_df = pd.DataFrame(markers)
marker_df.to_csv(f'{BASE_DIR}/marker_genes_allgenes.csv', index=False)
print(f"  Saved marker_genes_allgenes.csv ({len(marker_df)} entries)")

# Print top 10 per cluster
print("\nTop 10 DE genes per cluster (all-gene DE):")
cluster_names = {0: 'HM-Transcriptional', 1: 'HM-Inflammatory', 2: 'Spp1+/BAM',
                 3: 'HM-Metabolic', 4: 'HM-Motile', 5: 'HM-Quiescent'}
for i in range(6):
    print(f"\n  Cluster {i} ({cluster_names[i]}):")
    top = marker_df[marker_df['cluster'] == i].head(10)
    for _, row in top.iterrows():
        print(f"    {row['gene_symbol']:15s} score={row['score']:.1f}  logFC={row['logfoldchange']:.2f}")

# Check canonical markers
print("\n\nCanonical marker presence in new DE results:")
canonical = ['Spp1', 'Fth1', 'P2ry12', 'Trem2', 'C1qa', 'Lpl', 'Ctsd', 'Ctsb',
             'Siglech', 'Cx3cr1', 'Fcrls', 'Hexb']
for gene in canonical:
    matches = marker_df[marker_df['gene_symbol'] == gene]
    if len(matches) > 0:
        for _, row in matches.iterrows():
            rank = len(marker_df[(marker_df['cluster']==row['cluster']) & (marker_df['score']>=row['score'])])
            print(f"  {gene:12s}: cluster {row['cluster']} ({cluster_names[row['cluster']]}), score={row['score']:.1f}, rank={rank}")
    else:
        print(f"  {gene:12s}: NOT in top 100 for any cluster")

# Free memory
del adata_raw
gc.collect()

print("\n" + "=" * 70)
print("DONE")
print(f"  Output: {BASE_DIR}/marker_genes_allgenes.csv")
print(f"  ({len(marker_df)} entries: top 100 DE genes x 6 clusters)")
print("=" * 70)
