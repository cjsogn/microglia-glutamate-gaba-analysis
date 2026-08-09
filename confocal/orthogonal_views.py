#!/usr/bin/env python3
"""
Colocalisation of microglial markers in confocal z-stacks.

Segments the Iba1 channel and the marker channel in each stack, identifies
colocalised sites lying inside the microglial volume, and writes each site as a
compressed NumPy archive together with an index of the sites found. The archives
are the input to depth_vs_surface.py, and the sites are the source of the
orthogonal XY, XZ and YZ views shown in the supplementary figure.

Voxels are 0.10 um laterally and 0.57 um axially, so the stacks are anisotropic
by a factor of 5.7; distances are computed in micrometres throughout.

Usage
    CONFOCAL_STACKS=/path/to/stacks python orthogonal_views.py

Environment
    CONFOCAL_STACKS  directory of bleed-through corrected stacks
    MICROGLIA_DATA   where per-site archives are written (default: data)
"""

import argparse
import json
import os
from glob import glob

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import ball, binary_dilation, remove_small_objects, white_tophat

# ---------------------------------------------------------------- geometry
XY_UM = 0.10
Z_UM = 0.57
ANISO = Z_UM / XY_UM

INPUT_DIR = os.environ.get('CONFOCAL_STACKS', 'raw/corrected_stacks')
OUT_DIR = os.path.join(os.environ.get('MICROGLIA_DATA', 'data'), 'orthogonal')

# identical to the published pipeline
MARKER_PARAMS = {
    'EAAT1': {'thresh_multiplier': 1.0, 'min_size': 30, 'min_intensity_percentile': 60, 'tophat_radius': 5},
    'EAAT2': {'thresh_multiplier': 1.3, 'min_size': 50, 'min_intensity_percentile': 70, 'tophat_radius': 4},
    'GAD67': {'thresh_multiplier': 1.5, 'min_size': 50, 'min_intensity_percentile': 75, 'tophat_radius': 4},
    'GAT1':  {'thresh_multiplier': 1.4, 'min_size': 50, 'min_intensity_percentile': 70, 'tophat_radius': 4},
    'GAT3':  {'thresh_multiplier': 1.4, 'min_size': 50, 'min_intensity_percentile': 70, 'tophat_radius': 4},
    'GLS':   {'thresh_multiplier': 1.1, 'min_size': 40, 'min_intensity_percentile': 65, 'tophat_radius': 5},
}

MIN_COLOC_VOXELS = 40


def segment_iba1(image):
    smoothed = ndi.gaussian_filter(image, sigma=1.0)
    nz = smoothed[smoothed > 0]
    if nz.size == 0:
        return np.zeros_like(image, dtype=bool)
    binary = smoothed > threshold_otsu(nz)
    binary = binary_dilation(binary, ball(1))
    lab, _ = ndi.label(binary)
    return remove_small_objects(lab, min_size=50) > 0


def segment_marker(image, params):
    enhanced = white_tophat(image, ball(params['tophat_radius']))
    smoothed = ndi.gaussian_filter(enhanced, sigma=0.8)
    flat = smoothed[smoothed > 0]
    if flat.size == 0:
        return np.zeros_like(image, dtype=bool)
    thresh = threshold_otsu(flat) * params['thresh_multiplier']
    binary = smoothed > thresh
    lab, _ = ndi.label(binary)
    cleaned = remove_small_objects(lab, min_size=params['min_size']) > 0
    lab, _ = ndi.label(cleaned)
    if image[image > 0].size == 0:
        return np.zeros_like(image, dtype=bool)
    min_int = np.percentile(image[image > 0], params['min_intensity_percentile'])
    out = np.zeros_like(image, dtype=bool)
    for prop in regionprops(lab, intensity_image=image):
        if prop.mean_intensity >= min_int:
            out[lab == prop.label] = True
    return out


def find_sites(iba1_mask, marker_mask):
    """
    Colocalised components ranked by how deep inside the microglia they sit.

    The distance transform is computed with real voxel spacing so `depth_um` is
    a physical depth, not a voxel count; a site whose depth exceeds the lateral
    PSF cannot be explained by surface juxtaposition.
    """
    coloc = iba1_mask & marker_mask
    if coloc.sum() < MIN_COLOC_VOXELS:
        return []

    depth = ndi.distance_transform_edt(iba1_mask, sampling=(Z_UM, XY_UM, XY_UM))
    lab, n = ndi.label(coloc)
    sites = []
    for prop in regionprops(lab):
        if prop.area < MIN_COLOC_VOXELS:
            continue
        zc, yc, xc = [int(round(v)) for v in prop.centroid]
        if not iba1_mask[zc, yc, xc]:
            continue                      # centroid fell outside a concave object
        sites.append({
            'z': zc, 'y': yc, 'x': xc,
            'n_voxels': int(prop.area),
            'volume_um3': float(prop.area * XY_UM * XY_UM * Z_UM),
            'depth_um': float(depth[zc, yc, xc]),
            'max_depth_um': float(depth[lab == prop.label].max()),
        })
    sites.sort(key=lambda s: (-s['depth_um'], -s['n_voxels']))
    return sites


def process_stack(path, marker):
    stack = tifffile.imread(path)          # (Z, C, Y, X)
    iba1 = stack[:, 0].astype(np.float32)
    mark = stack[:, 1].astype(np.float32)

    im = segment_iba1(iba1)
    mm = segment_marker(mark, MARKER_PARAMS[marker])
    sites = find_sites(im, mm)

    return {
        'file': os.path.basename(path),
        'marker': marker,
        'shape': list(stack.shape),
        'iba1_voxels': int(im.sum()),
        'marker_voxels': int(mm.sum()),
        'coloc_voxels': int((im & mm).sum()),
        'coloc_fraction_of_iba1': float((im & mm).sum() / max(im.sum(), 1)),
        'n_sites': len(sites),
        # keep every site: truncating to the deepest N biases the depth
        # distribution upward and would overstate the evidence against
        # surface juxtaposition
        'sites': sites,
    }, im, mm, iba1, mark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--markers', nargs='*', default=list(MARKER_PARAMS))
    ap.add_argument('--per-marker', type=int, default=6,
                    help='how many stacks to analyse per marker')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    summary = []
    for marker in args.markers:
        files = sorted(glob(os.path.join(INPUT_DIR, marker, '*_corrected.tif')))
        if not files:
            print(f'{marker}: no stacks found')
            continue
        # spread across regions rather than taking the first N alphabetically
        pick = files[:: max(1, len(files) // args.per_marker)][:args.per_marker]
        for path in pick:
            res, im, mm, iba1, mark = process_stack(path, marker)
            summary.append(res)
            top = res['sites'][0]['depth_um'] if res['sites'] else 0.0
            print(f"{marker:<6} {res['file']:<28} coloc={res['coloc_voxels']:7d} "
                  f"sites={res['n_sites']:4d}  best depth={top:.2f} um")
            np.savez_compressed(
                os.path.join(OUT_DIR, f'{marker}_{os.path.splitext(res["file"])[0]}.npz'),
                iba1_mask=np.packbits(im), marker_mask=np.packbits(mm),
                shape=np.array(im.shape), iba1=iba1.astype(np.uint8),
                marker=mark.astype(np.uint8))

    with open(os.path.join(OUT_DIR, 'coloc_sites.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    print(f'\nwritten to {OUT_DIR}/coloc_sites.json')


if __name__ == '__main__':
    main()
