#!/usr/bin/env python3
"""
Microglia Marker Colocalization Analysis
========================================
Analyzes marker enrichment in microglia vs surrounding neuropil from 3D
confocal Z-stacks. Computes enrichment ratios and Wilcoxon signed-rank tests.

Manuscript figures: Fig 4

Input: Two-channel confocal Z-stacks (TIFF)
  - Channel 0: Iba1 (microglia marker)
  - Channel 1: Target marker (EAAT1, EAAT2, GAD67, GAT1, GAT3, or GLS)

Output: Enrichment ratio = marker intensity in microglia / true background
  - True background = intensity outside both microglia AND marker objects

Statistical test: Wilcoxon signed-rank test (H0: median ratio = 1.0)
"""

import os
import glob
import numpy as np
import tifffile
from scipy import stats
from scipy.ndimage import gaussian_filter, label, binary_dilation
from skimage.morphology import ball, remove_small_objects, white_tophat
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from multiprocessing import Pool


# =============================================================================
# SEGMENTATION PARAMETERS (optimized per marker)
# =============================================================================
MARKER_PARAMS = {
    'EAAT1': {'thresh_multiplier': 1.0, 'min_size': 30, 'min_intensity_percentile': 60, 'tophat_radius': 5},
    'EAAT2': {'thresh_multiplier': 1.3, 'min_size': 50, 'min_intensity_percentile': 70, 'tophat_radius': 4},
    'GAD67': {'thresh_multiplier': 1.5, 'min_size': 50, 'min_intensity_percentile': 75, 'tophat_radius': 4},
    'GAT1':  {'thresh_multiplier': 1.4, 'min_size': 50, 'min_intensity_percentile': 70, 'tophat_radius': 4},
    'GAT3':  {'thresh_multiplier': 1.4, 'min_size': 50, 'min_intensity_percentile': 70, 'tophat_radius': 4},
    'GLS':   {'thresh_multiplier': 1.1, 'min_size': 40, 'min_intensity_percentile': 65, 'tophat_radius': 5},
}


# =============================================================================
# SEGMENTATION FUNCTIONS
# =============================================================================
def segment_iba1(image):
    """
    Segment microglia from Iba1 channel using Otsu thresholding.

    Parameters:
        image: 3D numpy array (Z, Y, X) of Iba1 channel

    Returns:
        Binary mask of microglia (3D boolean array)
    """
    # Gaussian smoothing to reduce noise
    smoothed = gaussian_filter(image, sigma=1.0)

    # Otsu thresholding on non-zero pixels
    thresh = threshold_otsu(smoothed[smoothed > 0])
    binary = smoothed > thresh

    # Morphological dilation to fill small gaps
    struct = ball(1)
    binary = binary_dilation(binary, struct)

    # Remove small objects (noise)
    labeled, _ = label(binary)
    return remove_small_objects(labeled, min_size=50) > 0


def segment_marker(image, params):
    """
    Segment marker puncta using white top-hat transform and Otsu thresholding.

    Parameters:
        image: 3D numpy array (Z, Y, X) of marker channel
        params: Dictionary with segmentation parameters

    Returns:
        Binary mask of marker-positive regions (3D boolean array)
    """
    # White top-hat to enhance punctate structures
    selem = ball(params['tophat_radius'])
    enhanced = white_tophat(image, selem)

    # Gaussian smoothing
    smoothed = gaussian_filter(enhanced, sigma=0.8)

    # Otsu thresholding with multiplier
    flat = smoothed[smoothed > 0]
    if len(flat) == 0:
        return np.zeros_like(image, dtype=bool)

    thresh = threshold_otsu(flat) * params['thresh_multiplier']
    binary = smoothed > thresh

    # Remove small objects
    labeled, _ = label(binary)
    cleaned = remove_small_objects(labeled, min_size=params['min_size'])

    # Filter by intensity percentile
    labeled, _ = label(cleaned > 0)
    min_intensity = np.percentile(image[image > 0], params['min_intensity_percentile'])
    props = regionprops(labeled, intensity_image=image)

    final_mask = np.zeros_like(image, dtype=bool)
    for prop in props:
        if prop.mean_intensity >= min_intensity:
            final_mask[labeled == prop.label] = True

    return final_mask


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def process_single_stack(args):
    """
    Process a single confocal stack and calculate enrichment ratio.

    Parameters:
        args: Tuple of (file_path, marker_params)

    Returns:
        Dictionary with enrichment ratio and colocalization %, or None if failed
    """
    fpath, params = args

    try:
        # Load TIFF stack
        with tifffile.TiffFile(fpath) as tif:
            stack = tif.asarray()

        # Extract channels
        iba1_raw = stack[:, 0, :, :].astype(np.float32)
        marker_raw = stack[:, 1, :, :].astype(np.float32)
        del stack

        # Segment both channels
        iba1_mask = segment_iba1(iba1_raw)
        marker_mask = segment_marker(marker_raw, params)

        # Calculate enrichment ratio
        microglia_voxels = np.sum(iba1_mask)
        if microglia_voxels == 0:
            return None

        # Signal: marker intensity inside microglia
        signal = np.mean(marker_raw[iba1_mask])

        # True background: outside BOTH microglia AND marker objects
        true_bg_mask = ~iba1_mask & ~marker_mask
        bg_voxels = np.sum(true_bg_mask)
        if bg_voxels == 0:
            return None

        background = np.mean(marker_raw[true_bg_mask])
        if background == 0:
            return None

        enrichment_ratio = signal / background

        # Calculate colocalization %
        overlap = iba1_mask & marker_mask
        overlap_voxels = np.sum(overlap)
        colocalization_pct = (overlap_voxels / microglia_voxels) * 100

        return {
            'file': os.path.basename(fpath),
            'enrichment': enrichment_ratio,
            'colocalization': colocalization_pct,
            'microglia_voxels': microglia_voxels,
            'marker_voxels': np.sum(marker_mask),
            'overlap_voxels': overlap_voxels
        }

    except Exception as e:
        print(f"Error processing {os.path.basename(fpath)}: {e}")
        return None


def analyze_marker(input_dir, marker, n_cores=14):
    """
    Analyze all stacks for a single marker.

    Parameters:
        input_dir: Path to directory containing marker subdirectories
        marker: Marker name (EAAT1, EAAT2, GAD67, GAT1, GAT3, or GLS)
        n_cores: Number of CPU cores for parallel processing

    Returns:
        Dictionary with analysis results and statistics
    """
    marker_dir = os.path.join(input_dir, marker)
    files = glob.glob(os.path.join(marker_dir, '*.tif'))

    if not files:
        print(f"No TIFF files found for {marker}")
        return None

    params = MARKER_PARAMS[marker]
    print(f"Processing {marker} ({len(files)} stacks)...")

    # Parallel processing
    with Pool(min(n_cores, len(files))) as pool:
        results = pool.map(process_single_stack, [(f, params) for f in files])

    # Filter valid results
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print(f"No valid results for {marker}")
        return None

    # Extract values
    enrichment_values = np.array([r['enrichment'] for r in valid_results])
    colocalization_values = np.array([r['colocalization'] for r in valid_results])

    # Calculate statistics
    median = np.median(enrichment_values)
    q1 = np.percentile(enrichment_values, 25)
    q3 = np.percentile(enrichment_values, 75)
    mean = np.mean(enrichment_values)
    std = np.std(enrichment_values)

    # Wilcoxon signed-rank test (H0: median = 1.0)
    try:
        _, p_value = stats.wilcoxon(enrichment_values - 1.0, alternative='two-sided')
    except ValueError:
        p_value = 1.0

    return {
        'marker': marker,
        'n': len(valid_results),
        'enrichment': {
            'median': median,
            'q1': q1,
            'q3': q3,
            'mean': mean,
            'std': std,
            'values': list(enrichment_values)
        },
        'colocalization': {
            'mean': np.mean(colocalization_values),
            'std': np.std(colocalization_values),
            'values': list(colocalization_values)
        },
        'p_value': p_value,
        'individual_results': valid_results
    }


def analyze_all_markers(input_dir, output_file=None, n_cores=14):
    """
    Analyze all markers and generate summary report.

    Parameters:
        input_dir: Path to directory containing marker subdirectories
        output_file: Optional path to save results (text file)
        n_cores: Number of CPU cores for parallel processing

    Returns:
        Dictionary with results for all markers
    """
    markers = ['EAAT1', 'EAAT2', 'GAD67', 'GAT1', 'GAT3', 'GLS']
    all_results = {}

    print("=" * 70)
    print("MICROGLIA MARKER COLOCALIZATION ANALYSIS")
    print("=" * 70)
    print()

    for marker in markers:
        result = analyze_marker(input_dir, marker, n_cores)
        if result:
            all_results[marker] = result

    # Print summary
    print()
    print("=" * 70)
    print("ENRICHMENT RESULTS (Wilcoxon signed-rank test, H0: median = 1.0)")
    print("=" * 70)
    print()
    print(f"{'Marker':<8} {'n':<4} {'Median':<10} {'IQR':<15} {'p-value':<12} {'Significant'}")
    print("-" * 65)

    for marker in markers:
        if marker in all_results:
            r = all_results[marker]
            e = r['enrichment']
            iqr_str = f"[{e['q1']:.2f}, {e['q3']:.2f}]"
            sig = "Yes ***" if r['p_value'] < 0.001 else ("Yes **" if r['p_value'] < 0.01 else ("Yes *" if r['p_value'] < 0.05 else "No"))
            print(f"{marker:<8} {r['n']:<4} {e['median']:<10.3f} {iqr_str:<15} {r['p_value']:<12.6f} {sig}")

    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write("MICROGLIA MARKER COLOCALIZATION ANALYSIS RESULTS\n")
            f.write("=" * 70 + "\n\n")

            for marker in markers:
                if marker in all_results:
                    r = all_results[marker]
                    e = r['enrichment']
                    f.write(f"{marker}:\n")
                    f.write(f"  n = {r['n']}\n")
                    f.write(f"  Enrichment: {e['median']:.3f} [{e['q1']:.3f}, {e['q3']:.3f}]\n")
                    f.write(f"  Wilcoxon p-value: {r['p_value']:.6f}\n")
                    f.write(f"  Individual values: {', '.join([f'{v:.3f}' for v in e['values']])}\n\n")

        print(f"\nResults saved to: {output_file}")

    return all_results


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    import sys

    # Default input directory
    INPUT_DIR = "/Users/cjsogn/Corrected_Stacks"

    # Parse command line arguments
    if len(sys.argv) > 1:
        INPUT_DIR = sys.argv[1]

    # Run analysis
    results = analyze_all_markers(INPUT_DIR, n_cores=14)
