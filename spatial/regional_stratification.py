#!/usr/bin/env python3
"""
Region-stratified proximity, with bootstrap intervals and direct standardisation.

Computes, for each anatomical division, the reduction in median distance to the
nearest cognate neuron for microglia expressing a handling pathway relative to
those that do not, with bootstrap confidence intervals on each estimate.

It also computes a whole-brain estimate in which both groups are given the same
regional composition by direct standardisation. The two groups do not occupy the
brain alike, and divisions differ widely in neuronal density, so the pooled
comparison mixes proximity with location. The standardised estimate separates
them, and is what the supplementary figure and table report alongside the
per-division effects.

Usage
    python regional_stratification.py

Environment
    MICROGLIA_DATA     per-section tables from extract_stereoseq_cells.py
    MICROGLIA_RESULTS  output directory (default: results)
    MICROGLIA_FIGURES  figure output directory (default: figures)
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CELLS = os.path.join(os.environ.get('MICROGLIA_DATA', 'data'), 'stereoseq', 'cells')
LUT = os.path.join(os.environ.get('MICROGLIA_DATA', 'data'), 'stereoseq',
                   'region_lookup.csv')
OUT = os.environ.get('MICROGLIA_RESULTS', 'results')
FIGDIR = os.environ.get('MICROGLIA_FIGURES', 'figures')

GLU = ['Gls', 'Glul', 'Slc1a2', 'Slc1a3', 'Slc38a1']
GABA = ['Gad1', 'Gad2', 'Abat', 'Slc6a1', 'Slc6a11']

DIV_LABEL = {
    'Isocortex': 'Isocortex', 'OLF': 'Olfactory areas', 'HPF': 'Hippocampal formation',
    'CTXsp': 'Cortical subplate', 'CNU': 'Cerebral nuclei', 'TH': 'Thalamus',
    'HY': 'Hypothalamus', 'MB': 'Midbrain', 'P': 'Pons', 'MY': 'Medulla',
    'CB': 'Cerebellum', 'fibertracts': 'Fibre tracts',
}
# anterior -> posterior, parenchyma first, then non-parenchymal compartments
# Ventricular systems are excluded from the regional analysis: microglia there
# sit in the ependymal/ventricular zone rather than in neuropil, so "distance to
# the nearest neuron" is not a meaningful proximity measure. Excluding them
# changes the whole-brain estimates by <0.3 percentage points (n = 3,103 of
# 438,666), so the primary analysis is unaffected; it does remove the largest
# positive glutamate value, which was an artefact of that compartment.
ORDER = ['Isocortex', 'OLF', 'HPF', 'CTXsp', 'CNU', 'TH', 'HY', 'MB', 'P', 'MY', 'CB',
         'fibertracts']

MIN_PER_GROUP = 200
N_BOOT = 1000
BOOT_CAP = 40_000
SEED = 20260728

GREY = '#9E9E9E'
BLUE = '#1F6FB4'      # glutamate
RED = '#C0392B'       # GABA


def load():
    files = sorted(glob.glob(os.path.join(CELLS, 'microglia_*.parquet')))
    mg = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    lut = pd.read_csv(LUT)[['gene_area', 'area_name']]
    mg = mg.merge(lut, on='gene_area', how='left')
    mg['div'] = mg['area_name'].fillna('').map(
        lambda s: re.sub(r'^[LR]', '', s).split('-')[0] if s else 'unknown')
    return mg


def pct_reduction(a, b):
    ma, mb = np.median(a), np.median(b)
    return 100.0 * (1 - ma / mb) if mb else np.nan


def boot_ci(a, b, rng, n_boot=N_BOOT):
    a = a if a.size <= BOOT_CAP else rng.choice(a, BOOT_CAP, replace=False)
    b = b if b.size <= BOOT_CAP else rng.choice(b, BOOT_CAP, replace=False)
    out = np.empty(n_boot)
    for i in range(n_boot):
        out[i] = pct_reduction(rng.choice(a, a.size, replace=True),
                               rng.choice(b, b.size, replace=True))
    return np.nanpercentile(out, [2.5, 97.5])


def group_totals(mg):
    """How many expressing / non-expressing microglia each pathway has in total.

    Needed for the composition columns: what matters is not how many cells a
    region holds, but what SHARE of each group lives there. The pooled estimate
    is driven by those two shares differing.
    """
    tot = {}
    for pathway, genes, short in (('Glutamate', GLU, 'glu'), ('GABA', GABA, 'gaba')):
        s_ = mg.dropna(subset=[f'dist_{short}_annotation'])
        expr = (s_[genes] > 0).any(axis=1)
        tot[pathway] = (int(expr.sum()), int((~expr).sum()))
    return tot


def build_table(mg):
    rng = np.random.default_rng(SEED)
    totals = group_totals(mg)
    rows = []
    for div in ORDER:
        sub = mg[mg['div'] == div]
        if sub.empty:
            continue
        for pathway, genes, short in (('Glutamate', GLU, 'glu'), ('GABA', GABA, 'gaba')):
            col = f'dist_{short}_annotation'
            s = sub.dropna(subset=[col])
            expr = (s[genes] > 0).any(axis=1)
            a = s.loc[expr, col].to_numpy()
            b = s.loc[~expr, col].to_numpy()
            if a.size < MIN_PER_GROUP or b.size < MIN_PER_GROUP:
                continue
            est = pct_reduction(a, b)
            lo, hi = boot_ci(a, b, rng)
            rows.append({
                'division': div, 'region': DIV_LABEL.get(div, div), 'pathway': pathway,
                'n_expressing': int(a.size), 'n_non_expressing': int(b.size),
                'median_expressing_um': round(float(np.median(a)), 1),
                'median_non_expressing_um': round(float(np.median(b)), 1),
                'pct_of_expressing': round(100.0 * a.size / totals[pathway][0], 1),
                'pct_of_non_expressing': round(100.0 * b.size / totals[pathway][1], 1),
                'composition_difference': round(100.0 * a.size / totals[pathway][0]
                                                - 100.0 * b.size / totals[pathway][1], 1),
                'pct_reduction': round(float(est), 1),
                'ci_low': round(float(lo), 1), 'ci_high': round(float(hi), 1),
                'ci_excludes_zero': bool(lo > 0 or hi < 0),
            })
            print(f'  {DIV_LABEL.get(div,div):<22}{pathway:<10}'
                  f'{est:>7.1f}%  [{lo:.1f}, {hi:.1f}]')
    return pd.DataFrame(rows)


def wmedian(v, w):
    o = np.argsort(v)
    v, w = np.asarray(v)[o], np.asarray(w, float)[o]
    c = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(c, 0.5)])


def standardised(mg):
    """Whole-brain estimate with both groups given the same regional composition.

    Direct standardisation. Each cell is weighted so that, within its own group,
    the distribution across divisions matches the distribution of all microglia.
    Comparing the two weighted medians therefore removes the difference in WHERE
    the two groups live, which is what the raw pooled estimate is mostly picking
    up, and leaves only the within-region difference.
    """
    out = {}
    for pathway, genes, short in (('Glutamate', GLU, 'glu'), ('GABA', GABA, 'gaba')):
        col = f'dist_{short}_annotation'
        s_ = mg.dropna(subset=[col])
        s_ = s_[s_['div'].isin(ORDER)]
        expr = (s_[genes] > 0).any(axis=1)
        target = s_['div'].value_counts(normalize=True)
        vals = {}
        for name, mask in (('e', expr), ('n', ~expr)):
            g = s_.loc[mask]
            share = g['div'].value_counts(normalize=True)
            w = g['div'].map(lambda d: target.get(d, 0.0) / share.get(d, np.nan))
            vals[name] = wmedian(g[col].to_numpy(), w.to_numpy())
        out[pathway] = (100.0 * (1 - vals['e'] / vals['n']), vals['e'], vals['n'])
    return out


def make_figure(df, whole_brain, standard=None):
    fig, allax = plt.subplots(2, 2, figsize=(9.9, 9.2))
    axes = allax[0]
    for ax, pathway, colour in zip(axes, ['Glutamate', 'GABA'], [BLUE, RED]):
        d = df[df['pathway'] == pathway].set_index('division').reindex(
            [x for x in ORDER if x in set(df['division'])]).dropna(subset=['pct_reduction'])
        y = np.arange(len(d))[::-1]

        ax.axvline(0, color='0.25', lw=1.0, zorder=1)
        wb = whole_brain[pathway]
        ax.axvline(wb, color=colour, lw=1.0, ls=(0, (4, 3)), alpha=.75, zorder=1)
        if standard is not None:
            st = standard[pathway][0]
            ax.axvline(st, color='0.25', lw=1.2, ls=(0, (1, 2)), zorder=1)

        for yi, (_, r) in zip(y, d.iterrows()):
            sig = r['ci_excludes_zero']
            c = colour if sig else GREY
            ax.plot([r['ci_low'], r['ci_high']], [yi, yi], color=c, lw=1.8,
                    solid_capstyle='round', zorder=2)
            ax.plot(r['pct_reduction'], yi, 'o', ms=6.5, color=c,
                    mec='white', mew=1.0, zorder=3)

        ax.set_yticks(y)
        ax.set_yticklabels(d['region'], fontsize=9)
        ax.set_xlabel('Reduction in distance to cognate neurons (%)', fontsize=9.5)
        ax.set_title(f'{pathway} pathway', fontsize=11, pad=8, weight='bold', color=colour)
        ax.tick_params(labelsize=8.5)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.spines['left'].set_color('0.4')
        ax.spines['bottom'].set_color('0.4')
        ax.set_axisbelow(True)

    lo = min(df['ci_low'].min(), -5) - 4
    hi = max(df['ci_high'].max(), 5) + 4
    for ax, pathway in zip(axes, ['Glutamate', 'GABA']):
        ax.set_xlim(lo, hi)
        top = ax.get_ylim()[1]
        ax.annotate(f'pooled\n{whole_brain[pathway]:.1f}%',
                    xy=(whole_brain[pathway], top), xytext=(2, -2),
                    textcoords='offset points', ha='left', va='top',
                    fontsize=7.6, color=BLUE if pathway == 'Glutamate' else RED)
        if standard is not None:
            ax.annotate(f'same\ncomposition\n{standard[pathway][0]:.1f}%',
                        xy=(standard[pathway][0], top), xytext=(-3, -2),
                        textcoords='offset points', ha='right', va='top',
                        fontsize=7.6, color='0.25')

    # ---- composition: which regions each group actually lives in ----
    for ax, pathway, colour in zip(allax[1], ['Glutamate', 'GABA'], [BLUE, RED]):
        d = df[df['pathway'] == pathway].copy()
        d = d.sort_values('median_non_expressing_um')          # sparse neurons to the right
        y = np.arange(len(d))[::-1]
        for yi, (_, r) in zip(y, d.iterrows()):
            pe, pn = r['pct_of_expressing'], r['pct_of_non_expressing']
            ax.plot([pn, pe], [yi, yi], color='0.65', lw=1.4, zorder=1)
            ax.plot(pn, yi, 'o', ms=6.0, mfc='white', mec='0.35', mew=1.3, zorder=3)
            ax.plot(pe, yi, 'o', ms=6.5, color=colour, mec='white', mew=1.0, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['region']}  ({r['median_non_expressing_um']:.0f} µm)"
                            for _, r in d.iterrows()], fontsize=8.5)
        ax.set_xlabel('Share of each group living in the region (%)', fontsize=9.5)
        ax.set_title(f'{pathway} pathway: where the two groups live',
                     fontsize=10.5, pad=8, weight='bold', color=colour)
        ax.tick_params(labelsize=8.5)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.spines['left'].set_color('0.4'); ax.spines['bottom'].set_color('0.4')
        ax.set_xlim(0, max(df['pct_of_expressing'].max(), df['pct_of_non_expressing'].max()) + 3)


    # legend markers must not imply a single pathway colour means "significant";
    # colour encodes pathway (panel title), saturation encodes significance
    handles = [
        Line2D([], [], color='0.25', lw=1.0, label='no effect'),
        Line2D([], [], color='0.45', lw=1.0, ls=(0, (4, 3)), label='pooled whole-brain estimate'),
        Line2D([], [], color='0.25', lw=1.2, ls=(0, (1, 2)),
               label='whole brain, both groups given the same regional composition'),
        Line2D([], [], marker='o', color='0.15', lw=1.8, mec='white',
               label='panel colour: 95% CI excludes 0'),
        Line2D([], [], marker='o', color=GREY, lw=1.8, mec='white',
               label='grey: 95% CI includes 0'),
        Line2D([], [], marker='o', color='0.35', mfc='white', ls='none',
               label='lower panels: open = non-expressing, filled = expressing'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Proximity of microglia to cognate neurons, stratified by brain division',
                 fontsize=12, y=0.97)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIGDIR, f'FigS_regional_stratification.{ext}'),
                    dpi=400, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    mg = load()
    print(f'{len(mg):,} microglia loaded\n')
    print('per-region effects (bootstrap 95% CI):')
    df = build_table(mg)

    whole = {}
    for pathway, genes, short in (('Glutamate', GLU, 'glu'), ('GABA', GABA, 'gaba')):
        col = f'dist_{short}_annotation'
        s = mg.dropna(subset=[col])
        expr = (s[genes] > 0).any(axis=1)
        whole[pathway] = pct_reduction(s.loc[expr, col].to_numpy(),
                                       s.loc[~expr, col].to_numpy())
    print(f"\nwhole-brain: glutamate {whole['Glutamate']:.1f}%, GABA {whole['GABA']:.1f}%")

    df.to_csv(os.path.join(OUT, 'Supplementary_Table_S4_regional_stratification.csv'),
              index=False)
    std = standardised(mg)
    for pathway in ('Glutamate', 'GABA'):
        est, ve, vn = std[pathway]
        print(f'  {pathway}: composition-standardised {est:.1f}% '
              f'(expressing {ve:.1f} um vs non-expressing {vn:.1f} um) '
              f'against raw pooled {whole[pathway]:.1f}%')
    make_figure(df, whole, std)

    for pathway in ['Glutamate', 'GABA']:
        d = df[df['pathway'] == pathway]
        pos = ((d['pct_reduction'] > 0) & d['ci_excludes_zero']).sum()
        neg = ((d['pct_reduction'] < 0) & d['ci_excludes_zero']).sum()
        print(f'{pathway}: {pos}/{len(d)} divisions positive with CI excluding 0; '
              f'{neg} negative with CI excluding 0')
    print(f'\ntable -> {OUT}/Supplementary_Table_S4_regional_stratification.csv')
    print(f'figure -> {FIGDIR}/FigS_regional_stratification.png')


if __name__ == '__main__':
    main()
