#!/usr/bin/env python3
"""
Microglia Clustering and Annotation Pipeline
=============================================
Zeng Whole Mouse Brain Single-Cell RNA-seq Dataset

Extracts microglial cells from the Allen Brain Institute's Zeng whole mouse brain
atlas, performs batch-corrected clustering using scVI, annotates clusters, and
analyzes glutamate/GABA pathway expression.

NOTE: Allen Brain Atlas source files contain expression in log2(CPM+1) units,
not log2(raw counts+1). The back-transform yields CPM values. See
run_scvi_clustering() for discussion of implications for scVI model fitting.

Pipeline:
    1. Extract microglia from brain region h5ad files
    2. Preprocess and train scVI model with donor batch correction
    3. Leiden clustering and marker gene identification
    4. Cluster annotation (canonical, reference, functional signatures)
    5. Glutamate/GABA pathway analysis
    6. Generate visualizations and save results

References:
    - Lopez et al. (2018) Nature Methods - scVI
    - Hammond et al. (2019) Immunity - Microglia states
    - Keren-Shaul et al. (2017) Cell - DAM signatures

Manuscript figures: Fig 2B-D
"""

import gc
import warnings
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import stats

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Pipeline configuration."""

    # Paths - MODIFY FOR YOUR SYSTEM
    DATA_DIR = Path("/Users/cjsogn/Documents/data/Whole mouse brain RNA Zeng")
    OUTPUT_DIR = Path("/Users/cjsogn/zeng_microglia_analysis")

    # Brain region files
    H5AD_FILES = [
        "WMB-10Xv3-CB-log2.h5ad", "WMB-10Xv3-CTXsp-log2.h5ad",
        "WMB-10Xv3-HPF-log2.h5ad", "WMB-10Xv3-HY-log2.h5ad",
        "WMB-10Xv3-Isocortex-1-log2.h5ad", "WMB-10Xv3-Isocortex-2-log2.h5ad",
        "WMB-10Xv3-MB-log2.h5ad", "WMB-10Xv3-MY-log2.h5ad",
        "WMB-10Xv3-OLF-log2.h5ad", "WMB-10Xv3-P-log2.h5ad",
        "WMB-10Xv3-PAL-log2.h5ad", "WMB-10Xv3-STR-log2.h5ad",
        "WMB-10Xv3-TH-log2.h5ad",
    ]

    # scVI parameters
    N_LATENT = 30
    N_LAYERS = 2
    MAX_EPOCHS = 200
    BATCH_SIZE = 256
    N_TOP_GENES = 3000

    # Clustering
    TARGET_CLUSTERS = 6
    N_NEIGHBORS = 30
    N_JOBS = 14


# =============================================================================
# GENE DEFINITIONS
# =============================================================================

# Ensembl ID to symbol mapping for key microglia genes
ENSEMBL_TO_SYMBOL = {
    # Homeostatic
    'ENSMUSG00000036353': 'P2ry12', 'ENSMUSG00000030789': 'Tmem119',
    'ENSMUSG00000052336': 'Cx3cr1', 'ENSMUSG00000024679': 'Hexb',
    'ENSMUSG00000027447': 'Cst3', 'ENSMUSG00000024621': 'Csf1r',
    'ENSMUSG00000030468': 'Siglech', 'ENSMUSG00000037266': 'Fcrls',
    'ENSMUSG00000030579': 'Gpr34', 'ENSMUSG00000055170': 'Sall1',
    # DAM
    'ENSMUSG00000023992': 'Trem2', 'ENSMUSG00000002985': 'Apoe',
    'ENSMUSG00000052698': 'Tyrobp', 'ENSMUSG00000015396': 'Lpl',
    'ENSMUSG00000068129': 'Cst7', 'ENSMUSG00000079293': 'Clec7a',
    'ENSMUSG00000029304': 'Spp1', 'ENSMUSG00000031451': 'Lgals3',
    'ENSMUSG00000021025': 'Cd9', 'ENSMUSG00000024528': 'Cd63',
    'ENSMUSG00000030762': 'Gpnmb', 'ENSMUSG00000017009': 'Igf1',
    'ENSMUSG00000058341': 'Itgax',
    # Inflammatory
    'ENSMUSG00000018774': 'Cd68', 'ENSMUSG00000027398': 'Il1b',
    'ENSMUSG00000024401': 'Tnf', 'ENSMUSG00000038418': 'Ccl3',
    'ENSMUSG00000018930': 'Ccl4', 'ENSMUSG00000035385': 'Ccl2',
    # Proliferating
    'ENSMUSG00000031004': 'Mki67', 'ENSMUSG00000020914': 'Top2a',
    'ENSMUSG00000022594': 'Pcna', 'ENSMUSG00000027715': 'Birc5',
    # Phagocytic
    'ENSMUSG00000002111': 'Axl', 'ENSMUSG00000014361': 'Mertk',
    'ENSMUSG00000024672': 'Cd36',
    # Interferon
    'ENSMUSG00000034459': 'Ifit1', 'ENSMUSG00000074896': 'Ifit3',
    'ENSMUSG00000035692': 'Isg15', 'ENSMUSG00000000386': 'Mx1',
    'ENSMUSG00000026104': 'Stat1', 'ENSMUSG00000034329': 'Irf7',
    # Antigen presentation
    'ENSMUSG00000036594': 'H2-Aa', 'ENSMUSG00000073411': 'H2-Ab1',
    'ENSMUSG00000018446': 'Cd74', 'ENSMUSG00000037649': 'B2m',
    # Lysosomal
    'ENSMUSG00000052837': 'Ctsd', 'ENSMUSG00000044786': 'Ctsb',
    'ENSMUSG00000027508': 'Ctsl', 'ENSMUSG00000038037': 'Ctss',
    # Complement
    'ENSMUSG00000024397': 'C1qa', 'ENSMUSG00000036896': 'C1qb',
    'ENSMUSG00000036905': 'C1qc',
    # Iron/Lipid
    'ENSMUSG00000021108': 'Fth1', 'ENSMUSG00000050708': 'Ftl1',
    'ENSMUSG00000020484': 'Fabp5', 'ENSMUSG00000021453': 'Abca1',
    # Glutamate pathway
    'ENSMUSG00000026103': 'Gls', 'ENSMUSG00000026473': 'Glul',
    'ENSMUSG00000005360': 'Slc1a3', 'ENSMUSG00000005089': 'Slc1a2',
    'ENSMUSG00000023169': 'Slc38a1',
    # GABA pathway
    'ENSMUSG00000070880': 'Gad1', 'ENSMUSG00000026787': 'Gad2',
    'ENSMUSG00000057880': 'Abat', 'ENSMUSG00000030310': 'Slc6a1',
    'ENSMUSG00000030307': 'Slc6a11',
}

SYMBOL_TO_ENSEMBL = {v.lower(): k for k, v in ENSEMBL_TO_SYMBOL.items()}

# Signature gene sets
SIGNATURES = {
    'Homeostatic': ['P2ry12', 'Tmem119', 'Cx3cr1', 'Hexb', 'Cst3', 'Csf1r', 'Siglech', 'Fcrls'],
    'DAM': ['Trem2', 'Apoe', 'Lpl', 'Cst7', 'Itgax', 'Clec7a', 'Spp1', 'Gpnmb', 'Lgals3'],
    'Inflammatory': ['Cd68', 'Il1b', 'Tnf', 'Ccl2', 'Ccl3', 'Ccl4'],
    'Proliferating': ['Mki67', 'Top2a', 'Pcna', 'Birc5'],
    'Interferon': ['Ifit1', 'Ifit3', 'Isg15', 'Mx1', 'Stat1', 'Irf7'],
    'Antigen_Presenting': ['H2-Aa', 'H2-Ab1', 'Cd74', 'B2m'],
    'Lysosomal': ['Ctsd', 'Ctsb', 'Ctsl', 'Ctss', 'Cd68'],
    'Complement': ['C1qa', 'C1qb', 'C1qc'],
    'Iron_Storage': ['Fth1', 'Ftl1'],
    'Lipid': ['Apoe', 'Lpl', 'Fabp5', 'Abca1'],
}

# Glutamate/GABA pathway genes
PATHWAY_GENES = {
    'glutamate': {
        'ENSMUSG00000026103': ('Gls', 'Glutaminase'),
        'ENSMUSG00000026473': ('Glul', 'Glutamine synthetase'),
        'ENSMUSG00000005360': ('Slc1a3', 'EAAT1'),
        'ENSMUSG00000005089': ('Slc1a2', 'EAAT2'),
        'ENSMUSG00000023169': ('Slc38a1', 'SNAT1'),
    },
    'gaba': {
        'ENSMUSG00000070880': ('Gad1', 'GAD67'),
        'ENSMUSG00000026787': ('Gad2', 'GAD65'),
        'ENSMUSG00000057880': ('Abat', 'GABA-T'),
        'ENSMUSG00000030310': ('Slc6a1', 'GAT1'),
        'ENSMUSG00000030307': ('Slc6a11', 'GAT3'),
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_gene_expr(adata, ensembl_id: str) -> np.ndarray:
    """Extract expression vector for a gene."""
    idx = adata.var_names.get_loc(ensembl_id)
    X = adata.X[:, idx]
    return X.toarray().flatten() if hasattr(X, 'toarray') else X.flatten()


def genes_to_ensembl(gene_list: List[str], var_names) -> List[str]:
    """Convert gene symbols to Ensembl IDs present in data."""
    return [SYMBOL_TO_ENSEMBL[g.lower()] for g in gene_list
            if g.lower() in SYMBOL_TO_ENSEMBL and SYMBOL_TO_ENSEMBL[g.lower()] in var_names]


# =============================================================================
# EXTRACTION
# =============================================================================

def extract_microglia(config: Config) -> ad.AnnData:
    """Extract microglia from all brain region files."""
    print("=" * 70)
    print("STEP 1: EXTRACTING MICROGLIA")
    print("=" * 70)

    # Load metadata
    meta = pd.read_csv(config.DATA_DIR / "cell_metadata_with_cluster_annotation.csv")
    microglia_ids = set(meta[meta['subclass'].str.contains('Microglia', na=False)]['cell_label'])
    meta_dict = meta.set_index('cell_label')[['donor_label', 'subclass', 'supertype']].to_dict('index')
    print(f"Found {len(microglia_ids):,} microglia in metadata")

    # Extract from each region
    all_data = []
    for f in config.H5AD_FILES:
        region = f.replace("WMB-10Xv3-", "").replace("-log2.h5ad", "")
        adata = sc.read_h5ad(config.DATA_DIR / f)
        mask = adata.obs_names.isin(microglia_ids)

        if mask.sum() > 0:
            sub = adata[mask].copy()
            sub.obs['region'] = region
            sub.obs['donor_label'] = [meta_dict.get(c, {}).get('donor_label', 'Unknown') for c in sub.obs_names]
            all_data.append(sub)
            print(f"  {region}: {mask.sum():,} cells")

        del adata
        gc.collect()

    adata = ad.concat(all_data, join='outer')
    print(f"\nTotal: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    return adata


# =============================================================================
# SCVI CLUSTERING
# =============================================================================

def run_scvi_clustering(adata: ad.AnnData, config: Config) -> ad.AnnData:
    """Preprocess, run scVI, and cluster."""
    import scvi

    print("\n" + "=" * 70)
    print("STEP 2: SCVI INTEGRATION AND CLUSTERING")
    print("=" * 70)

    # NOTE: Source data is in log2(CPM+1) units, not log2(counts+1).
    # The reverse transform 2^x - 1 yields CPM values, not integer counts.
    # scVI expects discrete counts (uses negative binomial likelihood), so feeding
    # CPM values is technically incorrect. In practice, this primarily affects the
    # statistical assumptions of the generative model but the learned latent space
    # (and downstream UMAP/clustering) is relatively robust to this. If raw count
    # files become available, they should be used instead.
    print("Converting log2(CPM+1) data to linear scale (CPM)...")
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    adata.X = np.power(2, X) - 1
    adata.layers['counts'] = adata.X.copy()

    # HVG selection
    print(f"Selecting {config.N_TOP_GENES} HVGs...")
    adata_hvg = adata.copy()
    sc.pp.normalize_total(adata_hvg, target_sum=1e4)
    sc.pp.log1p(adata_hvg)
    sc.pp.highly_variable_genes(adata_hvg, n_top_genes=config.N_TOP_GENES,
                                 flavor='seurat', batch_key='donor_label')
    adata = adata[:, adata_hvg.var['highly_variable']].copy()
    del adata_hvg
    # NOTE: Subsetting to HVGs may exclude some target genes (e.g., glutamate/GABA
    # pathway genes) if they are not highly variable within the microglia subset.
    # This affects marker gene identification and pathway analysis downstream.
    # Genes not in the HVG set will not appear in rank_genes_groups results.
    print(f"After HVG filtering: {adata.n_vars} genes retained")

    # Train scVI
    print(f"Training scVI (n_latent={config.N_LATENT})...")
    scvi.model.SCVI.setup_anndata(adata, layer='counts', batch_key='donor_label')
    model = scvi.model.SCVI(adata, n_layers=config.N_LAYERS, n_latent=config.N_LATENT, gene_likelihood='nb')
    model.train(max_epochs=config.MAX_EPOCHS, early_stopping=True, batch_size=config.BATCH_SIZE)

    adata.obsm['X_scvi'] = model.get_latent_representation()
    model.save(str(config.OUTPUT_DIR / "scvi_model"), overwrite=True)

    # UMAP and clustering
    print("Computing UMAP and clustering...")
    sc.pp.neighbors(adata, use_rep='X_scvi', n_neighbors=config.N_NEIGHBORS)
    sc.tl.umap(adata)

    # Target 6 clusters based on established microglia state classifications
    # (homeostatic, inflammatory, DAM/activated, proliferating, interferon-responsive,
    # and antigen-presenting states) from Hammond et al. 2019 and Keren-Shaul et al. 2017.
    # The resolution parameter is searched to yield exactly this number.
    for res in np.arange(0.01, 1.0, 0.01):
        sc.tl.leiden(adata, resolution=res, key_added='leiden')
        if adata.obs['leiden'].nunique() == config.TARGET_CLUSTERS:
            print(f"Found {config.TARGET_CLUSTERS} clusters at resolution {res:.2f}")
            break

    print(f"\nCluster sizes:\n{adata.obs['leiden'].value_counts().sort_index()}")
    return adata


# =============================================================================
# MARKER GENES
# =============================================================================

def find_markers(adata: ad.AnnData) -> pd.DataFrame:
    """Find marker genes per cluster."""
    print("\n" + "=" * 70)
    print("STEP 3: MARKER GENE IDENTIFICATION")
    print("=" * 70)

    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    sc.tl.rank_genes_groups(adata_norm, 'leiden', method='wilcoxon', n_genes=50)
    adata.uns['rank_genes_groups'] = adata_norm.uns['rank_genes_groups']

    # Build marker dataframe
    markers = []
    for i in range(adata.obs['leiden'].nunique()):
        c = str(i)
        for j in range(50):
            markers.append({
                'cluster': i,
                'gene': adata.uns['rank_genes_groups']['names'][c][j],
                'score': adata.uns['rank_genes_groups']['scores'][c][j],
                'pval_adj': adata.uns['rank_genes_groups']['pvals_adj'][c][j]
            })

    return pd.DataFrame(markers)


# =============================================================================
# ANNOTATION
# =============================================================================

def annotate_clusters(adata: ad.AnnData, marker_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate clusters using signature scoring."""
    print("\n" + "=" * 70)
    print("STEP 4: CLUSTER ANNOTATION")
    print("=" * 70)

    # Normalize for scoring
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)

    # Score signatures
    scores = {}
    for name, genes in SIGNATURES.items():
        ensembl = genes_to_ensembl(genes, adata_norm.var_names)
        if len(ensembl) >= 2:
            sc.tl.score_genes(adata_norm, ensembl, score_name=name)
            scores[name] = adata_norm.obs.groupby('leiden')[name].mean()

    score_df = pd.DataFrame(scores)
    score_df_z = (score_df - score_df.mean()) / score_df.std()

    # Annotate based on top signature
    annotations = {}
    for cluster in score_df_z.index:
        top = score_df_z.loc[cluster].nlargest(2)
        ann = top.index[0]
        if top.iloc[1] > 0.5:
            ann = f"{ann}/{top.index[1]}"
        annotations[cluster] = ann

    adata.obs['annotation'] = adata.obs['leiden'].map(annotations)

    # Build annotation summary
    ann_df = pd.DataFrame([
        {'cluster': c, 'annotation': a,
         'n_cells': (adata.obs['leiden'] == c).sum(),
         'pct': f"{(adata.obs['leiden'] == c).mean()*100:.1f}%"}
        for c, a in annotations.items()
    ])

    print("\nAnnotations:")
    for _, r in ann_df.iterrows():
        print(f"  Cluster {r['cluster']}: {r['annotation']} ({r['n_cells']:,} cells)")

    return ann_df, score_df_z


# =============================================================================
# GLUTAMATE/GABA ANALYSIS
# =============================================================================

def analyze_pathways(adata: ad.AnnData) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyze glutamate/GABA pathway expression."""
    print("\n" + "=" * 70)
    print("STEP 5: GLUTAMATE/GABA PATHWAY ANALYSIS")
    print("=" * 70)

    # Normalize
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)

    cluster_ann = adata.obs.groupby('leiden')['annotation'].first().to_dict()
    all_genes = {**PATHWAY_GENES['glutamate'], **PATHWAY_GENES['gaba']}

    # 1. Per-gene enrichment
    enrichment = []
    for eid, (symbol, name) in all_genes.items():
        if eid not in adata_norm.var_names:
            continue
        expr = get_gene_expr(adata_norm, eid)
        global_pct = (expr > 0).mean() * 100

        for cluster in sorted(adata.obs['leiden'].unique(), key=int):
            mask = adata.obs['leiden'].values == cluster
            cluster_expr = expr[mask]
            pct = (cluster_expr > 0).mean() * 100
            enrichment.append({
                'gene_symbol': symbol, 'gene_name': name, 'ensembl_id': eid,
                'cluster': cluster, 'annotation': cluster_ann.get(cluster, ''),
                'pct_expressing': pct, 'global_pct': global_pct,
                'fold_enrichment': pct / global_pct if global_pct > 0 else 0
            })

    enrichment_df = pd.DataFrame(enrichment)

    # 2. Gene correlations
    expr_matrix = {sym: get_gene_expr(adata_norm, eid)
                   for eid, (sym, _) in all_genes.items() if eid in adata_norm.var_names}
    symbols = list(expr_matrix.keys())

    corr_matrix = pd.DataFrame(np.eye(len(symbols)), index=symbols, columns=symbols)
    for i, g1 in enumerate(symbols):
        for g2 in symbols[i+1:]:
            r, _ = stats.spearmanr(expr_matrix[g1], expr_matrix[g2])
            corr_matrix.loc[g1, g2] = corr_matrix.loc[g2, g1] = r

    # 3. Co-expression
    def count_expr(gene_dict):
        counts = np.zeros(adata_norm.n_obs)
        for eid in gene_dict:
            if eid in adata_norm.var_names:
                counts += (get_gene_expr(adata_norm, eid) > 0).astype(int)
        return counts

    glut_n = count_expr(PATHWAY_GENES['glutamate'])
    gaba_n = count_expr(PATHWAY_GENES['gaba'])
    both = (glut_n > 0) & (gaba_n > 0)
    high = (glut_n >= 3) & (gaba_n >= 2)

    coexpr = []
    for cluster in sorted(adata.obs['leiden'].unique(), key=int):
        mask = adata.obs['leiden'].values == cluster
        coexpr.append({
            'cluster': cluster, 'annotation': cluster_ann.get(cluster, ''),
            'n_cells': mask.sum(),
            'pct_both': both[mask].mean() * 100,
            'pct_high_coexpr': high[mask].mean() * 100
        })
    coexpr.append({
        'cluster': 'ALL', 'annotation': 'GLOBAL', 'n_cells': adata.n_obs,
        'pct_both': both.mean() * 100, 'pct_high_coexpr': high.mean() * 100
    })

    coexpr_df = pd.DataFrame(coexpr)

    print(f"  Cells expressing both pathways: {both.sum():,} ({both.mean()*100:.1f}%)")
    print(f"  High co-expressors (≥3 glut, ≥2 gaba): {high.sum():,} ({high.mean()*100:.1f}%)")

    return enrichment_df, corr_matrix, coexpr_df


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_figures(adata: ad.AnnData, score_df: pd.DataFrame, config: Config):
    """Generate publication figures."""
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING FIGURES")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({'font.size': 10, 'figure.dpi': 300})

    # UMAP
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sc.pl.umap(adata, color='leiden', ax=axes[0], show=False, title='Clusters', legend_loc='on data')
    sc.pl.umap(adata, color='annotation', ax=axes[1], show=False, title='Annotations')
    sc.pl.umap(adata, color='region', ax=axes[2], show=False, title='Region')
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "umap_overview.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Signature heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(score_df.T, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Signature Scores (Z-normalized)')
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "signature_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Cluster-region heatmap
    cr = pd.crosstab(adata.obs['leiden'], adata.obs['region'], normalize='index')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(cr, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
    ax.set_title('Cluster Distribution by Region')
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "cluster_region_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved figures")


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(adata, marker_df, ann_df, score_df, enrich_df, corr_df, coexpr_df, config):
    """Save all results."""
    print("\n" + "=" * 70)
    print("STEP 7: SAVING RESULTS")
    print("=" * 70)

    adata.write(config.OUTPUT_DIR / "microglia_scvi_annotated.h5ad")
    marker_df.to_csv(config.OUTPUT_DIR / "marker_genes.csv", index=False)
    ann_df.to_csv(config.OUTPUT_DIR / "cluster_annotations.csv", index=False)
    score_df.to_csv(config.OUTPUT_DIR / "signature_scores.csv")
    enrich_df.to_csv(config.OUTPUT_DIR / "glutamate_gaba_cluster_enrichment.csv", index=False)
    corr_df.to_csv(config.OUTPUT_DIR / "gene_coexpression_matrix.csv")
    coexpr_df.to_csv(config.OUTPUT_DIR / "glutamate_gaba_coexpression_summary.csv", index=False)

    # Cluster statistics
    stats = []
    for c in sorted(adata.obs['leiden'].unique(), key=int):
        mask = adata.obs['leiden'] == c
        stats.append({
            'cluster': c,
            'annotation': adata.obs.loc[mask, 'annotation'].iloc[0],
            'n_cells': mask.sum(),
            'n_donors': adata.obs.loc[mask, 'donor_label'].nunique(),
            'regions': str(adata.obs.loc[mask, 'region'].value_counts().to_dict())
        })
    pd.DataFrame(stats).to_csv(config.OUTPUT_DIR / "cluster_statistics.csv", index=False)

    print("All results saved")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run complete pipeline."""
    print("=" * 70)
    print("MICROGLIA CLUSTERING PIPELINE")
    print("=" * 70)

    config = Config()
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    sc.settings.verbosity = 2
    sc.settings.n_jobs = config.N_JOBS

    # Pipeline
    adata = extract_microglia(config)
    adata.write(config.OUTPUT_DIR / "microglia_extracted_raw.h5ad")

    adata = run_scvi_clustering(adata, config)
    adata.write(config.OUTPUT_DIR / "microglia_scvi_clustered.h5ad")

    marker_df = find_markers(adata)
    ann_df, score_df = annotate_clusters(adata, marker_df)
    enrich_df, corr_df, coexpr_df = analyze_pathways(adata)

    generate_figures(adata, score_df, config)
    save_results(adata, marker_df, ann_df, score_df, enrich_df, corr_df, coexpr_df, config)

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Cells: {adata.n_obs:,} | Clusters: {adata.obs['leiden'].nunique()} | Donors: {adata.obs['donor_label'].nunique()}")

    return adata


if __name__ == "__main__":
    adata = main()
