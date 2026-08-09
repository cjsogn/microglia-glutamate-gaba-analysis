#!/usr/bin/env python3
"""
Proximity to cognate neurons, stratified by sequencing depth.

Transcript detection depends on how deeply a cell was sequenced, so a microglial
cell counted as expressing a pathway may simply have been sampled better than one
counted as not expressing it. If that drove the proximity result, the difference
between the two groups would appear only where detection is easiest.

Microglia are therefore split into quintiles of total UMI count, and the
comparison between expressing and non-expressing cells is repeated within each
quintile: the median distance from the cell centroid to the nearest cognate
neuron, the reduction between the two groups, and a two-sided Mann-Whitney test.

Writes the supplementary table of depth-stratified proximity.

Usage
    python depth_stratification.py

Environment
    MICROGLIA_DATA     per-section tables from extract_stereoseq_cells.py
    MICROGLIA_RESULTS  output directory (default: results)
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

CELLS = os.path.join(os.environ.get('MICROGLIA_DATA', 'data'), 'stereoseq', 'cells')
OUT = os.environ.get('MICROGLIA_RESULTS', 'results')

GLU_GENES = ['Gls', 'Glul', 'Slc1a2', 'Slc1a3', 'Slc38a1']
GABA_GENES = ['Gad1', 'Gad2', 'Abat', 'Slc6a1', 'Slc6a11']
QUINTILES = ['Q1 (lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (highest)']

PATHWAYS = [
    ('Glutamate', GLU_GENES, 'dist_glu_annotation'),
    ('GABA', GABA_GENES, 'dist_gaba_annotation'),
]


def load():
    columns = (['total_umi'] + GLU_GENES + GABA_GENES
               + ['dist_glu_annotation', 'dist_gaba_annotation'])
    files = sorted(glob.glob(os.path.join(CELLS, 'microglia_*.parquet')))
    if not files:
        raise SystemExit(f'no microglia tables under {CELLS}; '
                         'run extract_stereoseq_cells.py first')
    df = pd.concat([pd.read_parquet(f, columns=columns) for f in files],
                   ignore_index=True)
    # quintiles over the whole population, so the same cut applies to both pathways
    df['umi_quintile'] = pd.qcut(df['total_umi'], 5, labels=QUINTILES)
    return df


def main():
    df = load()
    print(f'{len(df):,} microglia in {len(glob.glob(os.path.join(CELLS, "microglia_*.parquet")))} '
          f'sections')

    rows = []
    for pathway, genes, dist in PATHWAYS:
        expressing = df[genes].gt(0).any(axis=1)
        for q in QUINTILES:
            in_q = df['umi_quintile'] == q
            a = df.loc[in_q & expressing, dist].dropna()
            b = df.loc[in_q & ~expressing, dist].dropna()
            med_a, med_b = np.median(a), np.median(b)
            rows.append({
                'pathway': pathway,
                'umi_quintile': q,
                'n_expressing': len(a),
                'n_non_expressing': len(b),
                'median_expressing_um': round(med_a, 1),
                'median_non_expressing_um': round(med_b, 1),
                'pct_reduction': round(100.0 * (med_b - med_a) / med_b, 1),
                'p_value': mannwhitneyu(a, b, alternative='two-sided').pvalue,
            })

    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'Supplementary_Table_S4b_depth_stratification.csv')
    out.to_csv(path, index=False)

    for pathway, _, _ in PATHWAYS:
        sub = out[out['pathway'] == pathway]
        sig = (sub['p_value'] < 0.05).sum()
        print(f'  {pathway:<10} reduction {sub["pct_reduction"].min():>5.1f} to '
              f'{sub["pct_reduction"].max():>5.1f}%   significant in {sig} of 5 quintiles')
    print(f'\n-> {path}')


if __name__ == '__main__':
    main()
