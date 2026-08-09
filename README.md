# Microglial glutamate and GABA handling: analysis code

Analysis code for *Context-dependent localization and expression of glutamate and GABA
and their machinery in microglia*.

The study combines post-embedding immunogold electron microscopy, confocal
immunofluorescence, single-cell and single-nucleus transcriptomics, proteomics, and
Stereo-seq spatial transcriptomics. This repository holds the code that produces the
reported statistics, tables and quantitative figure panels.

## Repository layout

| Path | What it does |
| --- | --- |
| `spatial/extract_stereoseq_cells.py` | Extracts per-cell tables from the Stereo-seq atlas: centroid, Allen region, and counts for the handling genes. Input to everything else under `spatial/`. |
| `spatial/coexpression.py` | Counts, per microglial cell, how many genes of each pathway are detected, and the joint distribution across pathways. |
| `spatial/proximity_statistics.py` | Distance from microglia to the nearest cognate neuron, whole brain and by division, under alternative definitions of the neuronal classes. Also tabulates vesicular glutamate transporter usage by region. |
| `spatial/regional_stratification.py` | Per-division effects with bootstrap confidence intervals, plus a whole-brain estimate with both groups given the same regional composition by direct standardisation. |
| `spatial/depth_stratification.py` | Repeats the proximity comparison within quintiles of total UMI, to test whether the result tracks sequencing depth. |
| `spatial/spatial_analysis.py` | Spatial relationships between microglial profiles and synaptic terminals in the electron microscopy data. |
| `confocal/universal_bleedthrough_correction.py` | Linear-regression bleed-through correction applied to the confocal stacks before analysis. |
| `confocal/microglia_marker_analysis.py` | Marker enrichment inside the Iba1-positive mask relative to surrounding neuropil. |
| `confocal/orthogonal_views.py` | Segments Iba1 and marker channels, finds colocalised sites inside the microglial volume, and writes each site for downstream use. |
| `confocal/depth_vs_surface.py` | Tests whether colocalised signal sits inside the microglial volume or on its surface, against a within-cell null. |
| `transcriptomics/expression_analysis.py` | Expression of the handling machinery across the single-cell and single-nucleus datasets. |
| `transcriptomics/microglia_clustering_pipeline.py` | scVI training, microglial clustering and subtype annotation. |
| `proteomics/lps_proteomics_transcriptomics_analysis.py` | Joint proteomic and transcriptomic response to LPS. |
| `statistical/complete_analysis.R` | Linear mixed-effects models for the immunogold compartment comparisons. |
| `statistical/hurdle_glutamate_gaba.R` | Bayesian hurdle-gamma model of immunogold density under LPS. |
| `supplementary/S1_02_differential_expression_allgenes.py` | Differential expression across all genes for the microglial clusters. |
| `supplementary/S1_03_go_enrichment.py` | GO enrichment on the cluster marker genes. |
| `supplementary/S2_01_decontamination_validation.py` | Ambient RNA correction applied to the pathway genes. |
| `supplementary/S2_02_contamination_stratified_analysis.py` | Expression re-examined in cells stratified by contamination level. |

`data/` holds the immunogold and confocal result tables underlying the statistics.

## Requirements

Python 3.11 or later:

```
anndata  gseapy  h5py  matplotlib  numpy  pandas  requests
scanpy  scipy  scvi-tools  seaborn  shapely  scikit-image
scikit-learn  tifffile  pyarrow
```

`aicspylibczi` is needed only to read raw CZI stacks.

R 4.5 or later: `brms`, `lme4`, `lmerTest`, `emmeans`, `effectsize`, `dplyr`, `tidyr`,
`ggplot2`, `cowplot`, `patchwork`, `scales`. `brms` requires a working Stan toolchain.

## Source data

None of the primary datasets are redistributed here. They are obtained from:

- Allen Brain Cell Atlas, whole mouse brain and aging mouse (Zeng)
- Human brain single-nucleus atlas (Siletti)
- ASAP platform
- Stereo-seq mouse brain atlas (Han)
- Microglial proteomics (Rangaraju)
- GEO accession GSE307796, for the independent LPS dataset

Electron microscopy and confocal images are available from the corresponding author.

## Configuration

Paths are read from environment variables so that nothing is machine-specific. Each has a
relative default, so a layout of `raw/`, `data/`, `results/` and `figures/` under the
repository root works without setting anything.

| Variable | Default | Contents |
| --- | --- | --- |
| `ZENG_MOUSE_DIR` | `raw/zeng_mouse_brain` | Allen whole mouse brain |
| `ZENG_AGING_DIR` | `raw/zeng_aging_mouse` | Allen aging mouse |
| `SILETTI_DIR` | `raw/siletti_human` | Human single-nucleus atlas |
| `ASAP_DIR` | `raw/asap` | ASAP platform data |
| `STEREOSEQ_DIR` | `raw/han_brain_stereo_seq` | Stereo-seq atlas |
| `GEO_LPS_DIR` | `raw/GSE307796` | GEO LPS dataset |
| `PROTEOMICS_DIR` | `raw/rangaraju_proteomics` | Proteomics tables |
| `CONFOCAL_STACKS` | `raw/corrected_stacks` | Bleed-through corrected stacks |
| `MARKER_STACKS` | `raw/marker_analysis` | Raw marker stacks |
| `SPATIAL_IMAGE_DIR` | `raw/spatial_images` | Electron microscopy fields |
| `MICROGLIA_DATA` | `data` | Intermediate tables |
| `MICROGLIA_RESULTS` | `results` | Result tables |
| `MICROGLIA_FIGURES` | `figures` | Figure output |
| `N_CPUS` | `14` | Worker processes for the Stereo-seq extraction |

A few scripts write to their own result subdirectories, set by `ZENG_ANALYSIS_DIR`,
`EXPRESSION_RESULTS`, `DECONTAMINATION_DIR`, `SUPP_RESULTS`, `SPATIAL_RESULTS` and
`BAYES_RESULTS`.

## Running the spatial analysis

Extraction runs once and everything else reads its output:

```bash
export STEREOSEQ_DIR=/path/to/han_brain_stereo_seq
python spatial/extract_stereoseq_cells.py      # 123 sections, writes per-section tables

python spatial/coexpression.py --tag cached
python spatial/proximity_statistics.py
python spatial/regional_stratification.py
python spatial/depth_stratification.py
```

Extraction is the expensive step. It reads the full atlas and parallelises over sections,
so it benefits from `N_CPUS` and from the atlas being on a fast disk.

The confocal pipeline runs in the order correction, then colocalisation, then the depth
test:

```bash
export CONFOCAL_STACKS=/path/to/corrected_stacks
python confocal/orthogonal_views.py
python confocal/depth_vs_surface.py
```

## Notes on the analysis

**Neuronal classes.** Neurons are typed using the transcriptome-wide cell-type
annotations supplied with the Stereo-seq atlas, not by thresholding a single marker gene.
Because no single vesicular glutamate transporter marks excitatory neurons throughout the
brain, `proximity_statistics.py` repeats the comparison under marker-based definitions as
a sensitivity analysis.

**Pooling.** Microglia expressing a pathway and those not expressing it are distributed
differently across brain divisions, and divisions differ widely in neuronal density. A
pooled whole-brain comparison therefore mixes proximity with location.
`regional_stratification.py` reports per-division effects alongside a directly
standardised whole-brain estimate in which both groups carry the same regional
composition.

**Depth of the colocalised signal.** Microglial processes are thin, so the absolute
depths available are small. `depth_vs_surface.py` compares colocalised voxels against all
Iba1-positive voxels in the same cell rather than against the axial point spread function.

## License

Released for reproduction and reuse. Please cite the paper.
