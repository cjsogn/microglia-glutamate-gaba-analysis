# Analysis Scripts (FIXED)

Scripts used to generate the figures and statistical results in the manuscript.

This is the corrected version of the analysis scripts, addressing issues identified
during bioinformatics review (March 2026).

## Changes from original scripts

### Data units correction
Allen Brain Atlas source data files are in **log2(CPM+1)** units (counts per million),
not log2(raw counts+1) as originally assumed in code comments. The reverse transform
`2^x - 1` yields CPM values, not integer counts. This has been corrected in all
docstrings and comments throughout.

**Impact**: Most analyses are unaffected because they either (a) re-normalize after
back-transformation (DE, GO enrichment, clustering marker genes), (b) work on
relative comparisons (contamination validation), or (c) use entirely different data
(immunogold, confocal, Stereo-seq UMI counts, proteomics). The main affected
component is the scVI model, which expects integer counts but received CPM values.
See `microglia_clustering_pipeline.py` for detailed discussion.

### Script-specific fixes

| Script | Changes |
|--------|---------|
| `expression_analysis.py` | Documented that output mean_all/mean_expressing are in log2(CPM+1) units |
| `microglia_clustering_pipeline.py` | (1) Documented CPM-to-scVI limitation, (2) noted HVG subsetting risk for pathway genes, (3) justified choice of 6 clusters |
| `stereo_seq_spatial_analysis.py` | Implemented UMI-weighted centroid calculation (coordinates weighted by UMI count, which is higher near actual cell center) |
| `S2_01_decontamination_validation.py` | (1) Fixed bug where "Both"-source genes used only neuronal markers for normalization instead of both neuronal and astrocytic, (2) improved Wilcoxon test to use all corrected values instead of only originally-expressing cells, (3) added intercept p-value interpretation note |
| `S2_02_contamination_stratified_analysis.py` | Removed tautological Wilcoxon test (`wilcoxon(q_expr[q_expr > 0])` always significant by construction); binomial test and descriptive stats retained |
| `S1_02_differential_expression_allgenes.py` | Corrected unit documentation (log2(CPM+1), not log2(counts+1)); noted re-normalization makes DE independent of original units |
| `S1_03_go_enrichment.py` | No functional changes (uses gene lists, not expression values) |
| R and confocal scripts | No changes needed (use immunogold/fluorescence data, not scRNA-seq) |

### Items reviewed but intentionally kept as-is

- **Verdict labels** in contamination scripts: These are internal analysis labels for the
  supplementary figures, not used in manuscript results text. Retained with softened language.
- **Binomial test** (1% null) in stratified analysis: Provides a useful conservative test
  complementing the descriptive statistics (fraction expressing, Q1/Q4 ratio, trend).
- **scVI with CPM input**: Documented as a limitation rather than re-run, since (a) the
  learned latent space is relatively robust to this, (b) raw count files would need to be
  located, and (c) UMAP/clustering results are qualitatively similar.

## Directory structure

```
analysis_scripts_FIXED/
├── statistical/       R scripts for Bayesian and frequentist immunogold models
├── transcriptomics/   Python scripts for scRNA-seq expression and clustering
├── spatial/           Python scripts for EM spatial and Stereo-seq analyses
├── confocal/          Python scripts for confocal marker enrichment
├── proteomics/        Python scripts for proteomics/transcriptomics LPS comparison
└── supplementary/     Python scripts for supplementary analyses
```

## Script index

### statistical/

| Script | Figure(s) | Description |
|--------|-----------|-------------|
| `hurdle_glutamate_gaba.R` | Fig 1C,F; Fig 5C,F | Bayesian hurdle-gamma models for immunogold GABA/glutamate particle density (Control vs LPS) with publication-ready violin plots |
| `complete_analysis.R` | Fig 1 | Frequentist mixed-effects models, compartment comparisons, paired tests, effect sizes, and publication figures |

### transcriptomics/

| Script | Figure(s) | Description |
|--------|-----------|-------------|
| `expression_analysis.py` | Fig 2A; Fig 5G,I | Multi-dataset expression analysis of glutamate/GABA genes across 4 scRNA-seq cohorts (Zeng mouse, Zeng aging, Siletti human, ASAP human). Expression values are in log2(CPM+1) units. |
| `microglia_clustering_pipeline.py` | Fig 2B-D | scVI batch-corrected clustering of microglia from Zeng whole mouse brain atlas, with marker gene identification and glutamate/GABA pathway analysis. Note: scVI receives CPM (not integer counts) due to source data format. |

### spatial/

| Script | Figure(s) | Description |
|--------|-----------|-------------|
| `spatial_analysis.py` | Fig 1E-H | EM distance-density correlations between immunogold labeling and proximity to symmetric/asymmetric terminals (Spearman rho) |
| `stereo_seq_spatial_analysis.py` | Fig 3 | Stereo-seq analysis of microglial transporter expression in relation to proximity to glutamatergic/GABAergic neurons (Han et al. atlas). Uses UMI-weighted centroids. |

### confocal/

| Script | Figure(s) | Description |
|--------|-----------|-------------|
| `microglia_marker_analysis.py` | Fig 4 | 3D confocal enrichment ratios for glutamate/GABA markers in microglia vs neuropil background (Wilcoxon signed-rank tests) |
| `universal_bleedthrough_correction.py` | Fig 4 (preprocessing) | Pixel-wise linear regression bleed-through correction for Iba1-to-marker channel crosstalk on dual-labeled images |

### proteomics/

| Script | Figure(s) | Description |
|--------|-----------|-------------|
| `lps_proteomics_transcriptomics_analysis.py` | Fig 5G-J | Forest plots and t-tests comparing LPS-induced changes in glutamate/GABA genes across proteomics (TMT mass spec) and transcriptomics (scRNA-seq) |

### supplementary/

| Script | Figure(s) | Description |
|--------|-----------|-------------|
| `S1_02_differential_expression_allgenes.py` | Supp Fig S1 | Wilcoxon rank-sum DE on all 32,285 genes (not just HVGs) per microglia cluster |
| `S1_03_go_enrichment.py` | Supp Fig S1 | GO enrichment (BP, CC, MF) per cluster via Enrichr/gseapy |
| `S2_01_decontamination_validation.py` | Supp Fig S2 | Ambient RNA decontamination validation using SoupX-like subtraction and regression-based methods |
| `S2_02_contamination_stratified_analysis.py` | Supp Fig S2 | Contamination quartile stratification to verify gene expression in cleanest microglia |

## Dependencies

### R packages
brms, ggplot2, dplyr, tidyr, patchwork, cowplot, lme4, lmerTest, emmeans, effectsize, scales, bayesplot

### Python packages
numpy, pandas, scanpy, anndata, scvi-tools, scipy, matplotlib, seaborn, scikit-learn, scikit-image, shapely, tifffile, aicspylibczi, gseapy, h5py

## Notes

- All scripts use hardcoded paths from the original analysis environment. Paths will need to be updated for replication.
- R scripts use 14 CPU cores (`options(mc.cores = 14)`).
- Python scripts use 14 CPU cores where parallelized.
- Confocal analysis requires bleed-through correction to be run before marker analysis.
- Allen Brain Atlas h5ad files contain expression in log2(CPM+1) units, not log2(raw counts+1).
