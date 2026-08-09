#!/usr/bin/env python3
"""
Co-expression of glutamate- and GABA-handling genes within single microglia.

Counts, for each microglial cell, how many genes of each pathway are detected,
and summarises the joint distribution across the two pathways. This gives the
proportion of microglia expressing at least one gene of a pathway and the
fraction expressing both, as reported in the Results.

Usage
    python coexpression.py --tag cached

Environment
    MICROGLIA_DATA     per-section tables from extract_stereoseq_cells.py
    MICROGLIA_RESULTS  output directory (default: results)
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

GLU_GENES = ['Gls', 'Glul', 'Slc1a2', 'Slc1a3', 'Slc38a1']
GABA_GENES = ['Gad1', 'Gad2', 'Abat', 'Slc6a1', 'Slc6a11']

OUT_DIR = os.environ.get('MICROGLIA_RESULTS', 'results')


def load(source):
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, 'microglia_*.parquet')))
        mg = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        mg['cell_class'] = 'Microglia'
        afiles = sorted(glob.glob(os.path.join(source, 'astrocytes_*.parquet')))
        ast = (pd.concat([pd.read_parquet(f) for f in afiles], ignore_index=True)
               if afiles else pd.DataFrame())
        if not ast.empty:
            ast['cell_class'] = 'Astrocytes'
        return pd.concat([mg, ast], ignore_index=True)
    return pd.read_csv(source)


def summarise(df, label, genes, panel):
    """Distribution of the number of detected genes in `genes` per cell."""
    present = (df[genes] > 0)
    n_genes = present.sum(axis=1)
    n_cells = len(df)

    rows = []
    for k in range(len(genes) + 1):
        c = int((n_genes == k).sum())
        rows.append({'panel': panel, 'cell_type': label, 'n_genes_detected': k,
                     'n_cells': c, 'pct_cells': 100 * c / n_cells})
    out = pd.DataFrame(rows)

    any_pct = 100 * (n_genes > 0).mean()
    multi_pct = 100 * (n_genes > 1).mean()
    # of the cells that express anything, what share express >1?
    multi_of_pos = 100 * (n_genes > 1).sum() / max((n_genes > 0).sum(), 1)

    marg = {g: float((df[g] > 0).mean()) for g in genes}
    exp_none = np.prod([1 - p for p in marg.values()])
    exp_any = 100 * (1 - exp_none)

    print(f'\n{panel} pathway — {label} (n = {n_cells:,})')
    print('  marginal detection rate per gene:')
    for g, p in sorted(marg.items(), key=lambda kv: -kv[1]):
        print(f'    {g:<9} {100*p:6.2f}%')
    print(f'  >=1 gene detected : {any_pct:6.2f}%   '
          f'(expected under independence: {exp_any:.2f}%)')
    print(f'  >=2 genes detected: {multi_pct:6.2f}%')
    print(f'  of expressing cells, share with >1 gene: {multi_of_pos:.1f}%')
    print(f'  mean genes per cell: {n_genes.mean():.3f}  '
          f'(median {int(n_genes.median())})')
    return out, {'panel': panel, 'cell_type': label, 'n_cells': n_cells,
                 'pct_any': any_pct, 'pct_any_expected_indep': exp_any,
                 'pct_multi': multi_pct, 'pct_multi_of_expressing': multi_of_pos,
                 'mean_genes_per_cell': float(n_genes.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True,
                    help='per-cell CSV, or directory of parquet files')
    ap.add_argument('--tag', default='cached')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load(args.source)
    print(f'loaded {len(df):,} cells from {args.source}')
    missing = [g for g in GLU_GENES + GABA_GENES if g not in df.columns]
    if missing:
        raise SystemExit(f'missing gene columns: {missing}')

    dists, summaries = [], []
    for label in ['Microglia', 'Astrocytes']:
        sub = df[df['cell_class'] == label]
        if sub.empty:
            print(f'\n(no {label} in this table — skipped)')
            continue
        for panel, genes in (('glutamate', GLU_GENES), ('GABA', GABA_GENES)):
            d, s = summarise(sub, label, genes, panel)
            dists.append(d)
            summaries.append(s)

    # joint glutamate x GABA table for microglia
    mg = df[df['cell_class'] == 'Microglia']
    if not mg.empty:
        g_any = (mg[GLU_GENES] > 0).any(axis=1)
        b_any = (mg[GABA_GENES] > 0).any(axis=1)
        joint = pd.crosstab(g_any.rename('glutamate_any'),
                            b_any.rename('GABA_any'), normalize=True) * 100
        print('\njoint glutamate x GABA in microglia (% of all microglia):')
        print(joint.round(2).to_string())
        joint.to_csv(f'{OUT_DIR}/coexpression_joint_{args.tag}.csv')

    pd.concat(dists, ignore_index=True).to_csv(
        f'{OUT_DIR}/coexpression_distribution_{args.tag}.csv', index=False)
    pd.DataFrame(summaries).to_csv(
        f'{OUT_DIR}/coexpression_summary_{args.tag}.csv', index=False)
    print(f'\nwritten to {OUT_DIR}/coexpression_*_{args.tag}.csv')


if __name__ == '__main__':
    main()
