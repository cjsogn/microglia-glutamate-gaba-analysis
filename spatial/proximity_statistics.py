#!/usr/bin/env python3
"""
Distance from microglia to their nearest cognate neuron.

Compares microglia that express a handling pathway with microglia that do not,
on the distance from the cell centroid to the nearest glutamatergic or
GABAergic neuron. The comparison is made for the whole brain, within each
anatomical division, and under alternative definitions of the neuronal classes,
since no single vesicular glutamate transporter marks excitatory neurons
throughout the brain. Vesicular transporter usage by region is tabulated
alongside.

Usage
    python proximity_statistics.py

Environment
    MICROGLIA_DATA     per-section tables from extract_stereoseq_cells.py
    MICROGLIA_RESULTS  output directory (default: results)
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

CELLS = os.path.join(os.environ.get('MICROGLIA_DATA', 'data'), 'stereoseq', 'cells')
REGION_LOOKUP = os.path.join(os.environ.get('MICROGLIA_DATA', 'data'), 'stereoseq',
                             'region_lookup.csv')
OUT = os.environ.get('MICROGLIA_RESULTS', 'results')

GLU_GENES = ['Gls', 'Glul', 'Slc1a2', 'Slc1a3', 'Slc38a1']
GABA_GENES = ['Gad1', 'Gad2', 'Abat', 'Slc6a1', 'Slc6a11']
VGLUT = ['Slc17a6', 'Slc17a7', 'Slc17a8']

DIVISION_NAMES = {
    'Isocortex': 'Isocortex', 'OLF': 'Olfactory areas', 'HPF': 'Hippocampal formation',
    'CTXsp': 'Cortical subplate', 'CNU': 'Cerebral nuclei', 'TH': 'Thalamus',
    'HY': 'Hypothalamus', 'MB': 'Midbrain', 'P': 'Pons', 'MY': 'Medulla',
    'CB': 'Cerebellum', 'fibertracts': 'Fibre tracts', 'VS': 'Ventricular systems',
    'CTX': 'Cerebral cortex',
}
MIN_CELLS_PER_GROUP = 200


def major_division(area_name):
    """'LCB-CBX-CBXgr' -> 'CB';  'RIsocortex' -> 'Isocortex'."""
    if not isinstance(area_name, str) or not area_name:
        return 'unknown'
    s = re.sub(r'^[LR]', '', area_name)
    tok = s.split('-')[0]
    return tok if tok in DIVISION_NAMES else tok


def load(kind, cols=None):
    files = sorted(glob.glob(os.path.join(CELLS, f'{kind}_*.parquet')))
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files],
                     ignore_index=True)


def attach_region(df, lut):
    df = df.merge(lut, on='gene_area', how='left')
    df['division'] = df['area_name'].map(major_division)
    return df


def prox_test(sub, expr_mask, dist_col):
    a = sub.loc[expr_mask, dist_col].dropna()
    b = sub.loc[~expr_mask, dist_col].dropna()
    if len(a) < MIN_CELLS_PER_GROUP or len(b) < MIN_CELLS_PER_GROUP:
        return None
    u, p = mannwhitneyu(a, b, alternative='two-sided')
    ma, mb = float(a.median()), float(b.median())
    return {'n_expr': int(len(a)), 'n_non': int(len(b)),
            'median_expr_um': ma, 'median_non_um': mb,
            'pct_reduction': float(100 * (1 - ma / mb)) if mb else np.nan,
            'p_value': float(p)}


# ---------------------------------------------------------------- A
def analysis_A(neurons, lut):
    print('\n' + '=' * 74)
    print('A. Which vGluT do glutamatergic neurons use, by region?')
    print('=' * 74)
    n = attach_region(neurons, lut)
    g = n[n['class_annotation'] == 'Glutamatergic']
    rows = []
    for div, sub in g.groupby('division'):
        if len(sub) < MIN_CELLS_PER_GROUP:
            continue
        r = {'division': div, 'label': DIVISION_NAMES.get(div, div), 'n_neurons': len(sub)}
        for gene in VGLUT:
            r[f'pct_{gene}'] = float(100 * (sub[gene] > 0).mean())
        r['pct_any_vglut'] = float(100 * (sub[VGLUT] > 0).any(axis=1).mean())
        r['pct_only_a7'] = float(100 * ((sub['Slc17a7'] > 0) &
                                        (sub['Slc17a6'] == 0) & (sub['Slc17a8'] == 0)).mean())
        r['pct_a6_not_a7'] = float(100 * ((sub['Slc17a6'] > 0) & (sub['Slc17a7'] == 0)).mean())
        rows.append(r)
    df = pd.DataFrame(rows).sort_values('pct_Slc17a7', ascending=False)
    print(f"{'region':<24}{'n':>8}{'Slc17a7':>9}{'Slc17a6':>9}{'Slc17a8':>9}"
          f"{'any':>7}{'a6 not a7':>11}")
    for _, r in df.iterrows():
        print(f"{r['label'][:23]:<24}{r['n_neurons']:>8}{r['pct_Slc17a7']:>8.1f}%"
              f"{r['pct_Slc17a6']:>8.1f}%{r['pct_Slc17a8']:>8.1f}%"
              f"{r['pct_any_vglut']:>6.1f}%{r['pct_a6_not_a7']:>10.1f}%")
    df.to_csv(f'{OUT}/vglut_by_region.csv', index=False)
    return df


# ---------------------------------------------------------------- B
def analysis_B(mg):
    print('\n' + '=' * 74)
    print('B. Proximity result under three neuron definitions')
    print('=' * 74)
    rows = []
    for defn in ['annotation', 'vglut_all', 'slc17a7']:
        for panel, genes, short in (('glutamate', GLU_GENES, 'glu'),
                                    ('GABA', GABA_GENES, 'gaba')):
            col = f'dist_{short}_{defn}'
            if col not in mg.columns:
                continue
            sub = mg.dropna(subset=[col])
            res = prox_test(sub, (sub[genes] > 0).any(axis=1), col)
            if res:
                rows.append({'definition': defn, 'pathway': panel, **res})
    df = pd.DataFrame(rows)
    print(f"{'definition':<12}{'pathway':<11}{'expr median':>13}{'non median':>12}"
          f"{'reduction':>11}{'P':>10}")
    for _, r in df.iterrows():
        print(f"{r['definition']:<12}{r['pathway']:<11}{r['median_expr_um']:>12.1f}u"
              f"{r['median_non_um']:>11.1f}u{r['pct_reduction']:>10.1f}%"
              f"{r['p_value']:>10.1e}")
    df.to_csv(f'{OUT}/proximity_by_definition.csv', index=False)
    return df


# ---------------------------------------------------------------- C
def analysis_C(mg, lut):
    print('\n' + '=' * 74)
    print('C. Region-stratified proximity (Han annotation definition)')
    print('=' * 74)
    m = attach_region(mg, lut)
    rows = []
    for div, sub in m.groupby('division'):
        if len(sub) < 2 * MIN_CELLS_PER_GROUP:
            continue
        for panel, genes, short in (('glutamate', GLU_GENES, 'glu'),
                                    ('GABA', GABA_GENES, 'gaba')):
            col = f'dist_{short}_annotation'
            s = sub.dropna(subset=[col])
            res = prox_test(s, (s[genes] > 0).any(axis=1), col)
            if res:
                rows.append({'division': div, 'label': DIVISION_NAMES.get(div, div),
                             'pathway': panel, **res})
    df = pd.DataFrame(rows)
    for panel in ['glutamate', 'GABA']:
        d = df[df['pathway'] == panel].sort_values('pct_reduction', ascending=False)
        pos = (d['pct_reduction'] > 0).sum()
        sig = ((d['pct_reduction'] > 0) & (d['p_value'] < 0.05)).sum()
        print(f'\n  {panel} pathway — {pos}/{len(d)} divisions show the expected '
              f'direction, {sig} at P < 0.05')
        print(f"    {'region':<24}{'expr':>9}{'non':>9}{'reduction':>11}{'P':>10}")
        for _, r in d.iterrows():
            flag = '' if r['p_value'] < 0.05 else '  n.s.'
            print(f"    {r['label'][:23]:<24}{r['median_expr_um']:>8.1f}u"
                  f"{r['median_non_um']:>8.1f}u{r['pct_reduction']:>10.1f}%"
                  f"{r['p_value']:>10.1e}{flag}")
    df.to_csv(f'{OUT}/proximity_by_region.csv', index=False)
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    lut = pd.read_csv(REGION_LOOKUP)[['gene_area', 'area_name']]

    print('loading neurons ...', flush=True)
    neurons = load('neurons', ['gene_area', 'class_annotation'] + VGLUT + ['Gad1', 'Gad2'])
    print(f'  {len(neurons):,} neurons')
    analysis_A(neurons, lut)
    del neurons

    print('\nloading microglia ...', flush=True)
    mg = load('microglia')
    print(f'  {len(mg):,} microglia')
    analysis_B(mg)
    analysis_C(mg, lut)
    print(f'\nresults written to {OUT}/')


if __name__ == '__main__':
    main()
