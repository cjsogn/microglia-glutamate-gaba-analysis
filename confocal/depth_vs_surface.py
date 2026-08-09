#!/usr/bin/env python3
"""
Is colocalised marker signal inside the microglial volume, or on its surface?

For every colocalised site, compares the distance-to-surface distribution of the
colocalised voxels with that of all Iba1-positive voxels in the same volume, and
computes the fraction of each lying within one lateral voxel of the surface.

The depth ratio is the median distance-to-surface of colocalised voxels divided
by that of all microglial voxels. A ratio of 1 means colocalised signal is
distributed through the microglial volume like any microglial voxel; values
below 1 would indicate surface bias, which is what close juxtaposition rather
than genuine internalisation would produce.

The comparison is made against this within-cell null rather than against the
axial point spread function, because microglial processes are thin and the
absolute depths available are correspondingly small.

Usage
    python depth_vs_surface.py

Environment
    MICROGLIA_DATA     per-site archives from orthogonal_views.py
    MICROGLIA_RESULTS  output directory (default: results)
"""

import glob
import json
import os

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu

XY_UM, Z_UM = 0.10, 0.57
IN_DIR = os.path.join(os.environ.get('MICROGLIA_DATA', 'data'), 'orthogonal')
OUT = os.environ.get('MICROGLIA_RESULTS', 'results')
SURFACE_SHELL_UM = 0.10
MAX_SAMPLE = 200_000          # subsample huge masks for the test


def unpack(npz):
    shape = tuple(npz['shape'])
    n = int(np.prod(shape))
    iba1 = np.unpackbits(npz['iba1_mask'])[:n].astype(bool).reshape(shape)
    mark = np.unpackbits(npz['marker_mask'])[:n].astype(bool).reshape(shape)
    return iba1, mark


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(IN_DIR, '*.npz'))):
        stem = os.path.basename(path)[:-4]
        marker = stem.split('_')[0]
        with np.load(path) as npz:
            iba1, mark = unpack(npz)

        if iba1.sum() == 0:
            continue
        depth = ndi.distance_transform_edt(iba1, sampling=(Z_UM, XY_UM, XY_UM))

        null_d = depth[iba1]                      # every microglial voxel
        obs_d = depth[iba1 & mark]                # colocalised voxels
        if obs_d.size < 50:
            continue

        rng = np.random.default_rng(0)
        nd = null_d if null_d.size <= MAX_SAMPLE else rng.choice(null_d, MAX_SAMPLE, replace=False)
        od = obs_d if obs_d.size <= MAX_SAMPLE else rng.choice(obs_d, MAX_SAMPLE, replace=False)
        u, p = mannwhitneyu(od, nd, alternative='two-sided')

        rows.append({
            'stack': stem, 'marker': marker,
            'n_obs': int(obs_d.size), 'n_null': int(null_d.size),
            'median_obs_um': float(np.median(obs_d)),
            'median_null_um': float(np.median(null_d)),
            'depth_ratio': float(np.median(obs_d) / max(np.median(null_d), 1e-9)),
            'shell_frac_obs': float((obs_d <= SURFACE_SHELL_UM).mean()),
            'shell_frac_null': float((null_d <= SURFACE_SHELL_UM).mean()),
            'p_value': float(p),
        })
        print(f"{marker:<6} {stem.split('_',1)[1][:24]:<26} "
              f"obs {np.median(obs_d):.3f}  null {np.median(null_d):.3f}  "
              f"ratio {np.median(obs_d)/max(np.median(null_d),1e-9):.2f}  P={p:.2g}")

    if not rows:
        raise SystemExit('no stacks analysed')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'depth_vs_null.json'), 'w') as fh:
        json.dump(rows, fh, indent=1)

    ratios = np.array([r['depth_ratio'] for r in rows])
    shell_o = np.array([r['shell_frac_obs'] for r in rows])
    shell_n = np.array([r['shell_frac_null'] for r in rows])
    print('\n' + '=' * 62)
    print(f'stacks: {len(rows)}')
    print(f'depth ratio (observed/null): median {np.median(ratios):.2f}  '
          f'range {ratios.min():.2f}-{ratios.max():.2f}')
    print(f'  stacks where colocalised signal is DEEPER than null: '
          f'{(ratios > 1).sum()}/{len(ratios)}')
    print(f'surface shell (<={SURFACE_SHELL_UM} um) fraction: '
          f'observed {shell_o.mean():.3f} vs null {shell_n.mean():.3f}')
    print('\ninterpretation: ratio ~1 and matched shell fractions = marker signal is')
    print('distributed through the microglial volume like any microglial voxel;')
    print('ratio <1 with excess shell fraction = surface-biased (juxtaposition).')


if __name__ == '__main__':
    main()
