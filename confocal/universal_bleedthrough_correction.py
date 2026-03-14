#!/usr/bin/env python3
"""
Universal Bleed-Through Correction for Confocal Marker Analysis
================================================================
Corrects Iba1 (Ch0) spectral bleed-through into marker channel (Ch1) for all
confocal CZI stacks using pixel-wise linear regression on dual-labeled images.
The regression slope between Ch0 and Ch1 intensities estimates the bleed-through
fraction, which is then subtracted from the marker channel.

Manuscript figures: Fig 4 (preprocessing step)

Input:  CZI files organized by marker folder
Output: Corrected TIFF stacks, JSON correction summaries, comparison PNGs
"""

import numpy as np
from aicspylibczi import CziFile
import tifffile
import matplotlib.pyplot as plt
from scipy import stats
from skimage import exposure
from pathlib import Path
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

def read_czi_channels(czi_path):
    """Read channels from CZI"""
    czi = CziFile(czi_path)
    img_tuple = czi.read_image()
    img = img_tuple[0]
    ch0 = img[0, 0, :, :, :]  # iba1
    ch1 = img[0, 1, :, :, :]  # marker
    return ch0, ch1

def estimate_bleedthrough(ch0, ch1):
    """Estimate bleed-through using linear regression"""
    ch0_flat = ch0.flatten()
    ch1_flat = ch1.flatten()
    valid = (ch0_flat > 0) & (ch1_flat > 0)

    if np.sum(valid) > 1000:
        # Sample for efficiency
        sample_size = min(50000, np.sum(valid))
        sample_idx = np.random.choice(np.sum(valid), sample_size, replace=False)
        ch0_sample = ch0_flat[valid][sample_idx]
        ch1_sample = ch1_flat[valid][sample_idx]

        # Linear regression
        slope, intercept, r_value, _, _ = stats.linregress(ch0_sample, ch1_sample)

        # Use slope as bleed-through estimate
        fraction = max(0.0, min(slope, 0.8))  # Cap at 80%
    else:
        fraction = 0.0
        intercept = 0.0
        r_value = 0.0

    return fraction, intercept, r_value**2

def apply_correction(ch0, ch1, fraction):
    """Apply linear correction"""
    ch1_corrected = ch1.astype(np.float32) - fraction * ch0.astype(np.float32)
    ch1_corrected = np.maximum(ch1_corrected, 0)
    return ch1_corrected

def calculate_pearson(ch0, ch1, phase_averaged=False):
    """Calculate Pearson correlation"""
    valid = (ch0 > 0) & (ch1 > 0)
    if np.sum(valid) > 100:
        return np.corrcoef(ch0[valid].flatten(), ch1[valid].flatten())[0, 1]
    return 0.0

def process_single_file(args):
    """Process a single file (for parallel execution)"""
    czi_path, output_dir, marker_name, visualize = args

    try:
        czi_path = Path(czi_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Read data
        ch0, ch1 = read_czi_channels(czi_path)

        # Estimate bleed-through
        fraction, intercept, r2 = estimate_bleedthrough(ch0, ch1)

        # Calculate correlation before
        corr_before = calculate_pearson(ch0, ch1)

        # Apply correction
        ch1_corrected = apply_correction(ch0, ch1, fraction)

        # Calculate correlation after
        corr_after = calculate_pearson(ch0, ch1_corrected)

        # Save corrected image
        output_name = czi_path.stem + "_corrected.tif"
        output_path = output_dir / output_name

        nz, ny, nx = ch0.shape
        output_stack = np.zeros((nz, 2, ny, nx), dtype=ch0.dtype)
        output_stack[:, 0, :, :] = ch0
        output_stack[:, 1, :, :] = ch1_corrected.astype(ch0.dtype)

        tifffile.imwrite(output_path, output_stack, imagej=True,
                         metadata={'axes': 'ZCYX'})

        # Generate visualization
        if visualize:
            z_mid = ch0.shape[0] // 2

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Top row: individual channels
            axes[0, 0].imshow(ch0[z_mid], cmap='Reds', vmax=np.percentile(ch0[z_mid], 99))
            axes[0, 0].set_title('iba1 (Ch0)')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(ch1[z_mid], cmap='Greens', vmax=np.percentile(ch1[z_mid], 99))
            axes[0, 1].set_title(f'{marker_name} (uncorrected)')
            axes[0, 1].axis('off')

            axes[0, 2].imshow(ch1_corrected[z_mid], cmap='Greens', vmax=np.percentile(ch1_corrected[z_mid], 99))
            axes[0, 2].set_title(f'{marker_name} (corrected)')
            axes[0, 2].axis('off')

            # Bottom row: overlays and analysis
            overlay_before = np.stack([
                exposure.rescale_intensity(ch0[z_mid], out_range=(0, 1)),
                exposure.rescale_intensity(ch1[z_mid], out_range=(0, 1)),
                np.zeros_like(ch0[z_mid], dtype=float)
            ], axis=-1)
            axes[1, 0].imshow(overlay_before)
            axes[1, 0].set_title(f'Before\nCorr: {corr_before:.3f}')
            axes[1, 0].axis('off')

            overlay_after = np.stack([
                exposure.rescale_intensity(ch0[z_mid], out_range=(0, 1)),
                exposure.rescale_intensity(ch1_corrected[z_mid], out_range=(0, 1)),
                np.zeros_like(ch0[z_mid], dtype=float)
            ], axis=-1)
            axes[1, 1].imshow(overlay_after)
            axes[1, 1].set_title(f'After\nCorr: {corr_after:.3f}')
            axes[1, 1].axis('off')

            # Scatter plot
            ch0_flat = ch0[z_mid].flatten()
            ch1_flat = ch1[z_mid].flatten()
            ch1_corr_flat = ch1_corrected[z_mid].flatten()
            valid = (ch0_flat > 0) & (ch1_flat > 0)

            if np.sum(valid) > 1000:
                idx = np.random.choice(np.sum(valid), min(3000, np.sum(valid)), replace=False)
                axes[1, 2].scatter(ch0_flat[valid][idx], ch1_flat[valid][idx],
                                 alpha=0.3, s=1, label='Before', color='gray')
                axes[1, 2].scatter(ch0_flat[valid][idx], ch1_corr_flat[valid][idx],
                                 alpha=0.3, s=1, label='After', color='green')
                axes[1, 2].plot([0, ch0_flat.max()], [0, fraction * ch0_flat.max()],
                              'r--', linewidth=2, label=f'Removed: {fraction:.3f}')
                axes[1, 2].set_xlabel('iba1 intensity')
                axes[1, 2].set_ylabel(f'{marker_name} intensity')
                axes[1, 2].set_title('Correction')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)

            fig.suptitle(f'{czi_path.stem} - {marker_name}\nBleed-through: {fraction*100:.2f}%',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()

            fig_path = output_dir / (czi_path.stem + "_comparison.png")
            fig.savefig(fig_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

        results = {
            'filename': czi_path.name,
            'marker': marker_name,
            'bleedthrough_fraction': float(fraction),
            'bleedthrough_percent': float(fraction * 100),
            'intercept': float(intercept),
            'r2': float(r2),
            'correlation_before': float(corr_before),
            'correlation_after': float(corr_after),
            'correlation_change': float(corr_after - corr_before),
            'ch0_mean': float(np.mean(ch0[ch0 > 0])),
            'ch1_mean_before': float(np.mean(ch1[ch1 > 0])),
            'ch1_mean_after': float(np.mean(ch1_corrected[ch1_corrected > 0])),
        }

        return results, None

    except Exception as e:
        return None, f"{czi_path.name}: {str(e)}"

def process_marker(marker_name, input_dir, base_output_dir, n_cores=14, visualize=True):
    """Process all files for a single marker"""

    input_path = Path(input_dir) / marker_name
    output_dir = Path(base_output_dir) / marker_name / 'corrected_regression'

    if not input_path.exists():
        print(f"ERROR: Directory not found: {input_path}")
        return None

    # Get CZI files
    czi_files = sorted(input_path.glob('*.czi'))

    if len(czi_files) == 0:
        print(f"ERROR: No CZI files found in {input_path}")
        return None

    print(f"\n{'='*60}")
    print(f"PROCESSING MARKER: {marker_name}")
    print(f"{'='*60}")
    print(f"Found {len(czi_files)} files")
    print(f"Output: {output_dir}")
    print(f"Using {n_cores} cores")

    output_dir.mkdir(exist_ok=True, parents=True)

    # Prepare arguments for parallel processing
    args_list = [(f, output_dir, marker_name, visualize) for f in czi_files]

    # Process in parallel
    all_results = []
    errors = []

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = {executor.submit(process_single_file, args): args[0] for args in args_list}

        for i, future in enumerate(as_completed(futures), 1):
            file_path = futures[future]
            print(f"  [{i}/{len(czi_files)}] {file_path.name}...", end=' ')

            try:
                result, error = future.result()
                if result:
                    all_results.append(result)
                    print(f"✓ {result['bleedthrough_percent']:.1f}%")
                elif error:
                    errors.append(error)
                    print(f"✗ {error}")
            except Exception as e:
                errors.append(f"{file_path.name}: {str(e)}")
                print(f"✗ {str(e)}")

    # Save summary
    if all_results:
        summary_path = output_dir / 'correction_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Print statistics
        fractions = [r['bleedthrough_fraction'] for r in all_results]
        corr_changes = [r['correlation_change'] for r in all_results]

        print(f"\n{'='*60}")
        print(f"SUMMARY: {marker_name}")
        print(f"{'='*60}")
        print(f"Files processed: {len(all_results)}/{len(czi_files)}")
        print(f"Bleed-through: {np.mean(fractions)*100:.2f}% ± {np.std(fractions)*100:.2f}%")
        print(f"  Range: {np.min(fractions)*100:.2f}% - {np.max(fractions)*100:.2f}%")
        print(f"Correlation change: {np.mean(corr_changes):.4f} ± {np.std(corr_changes):.4f}")
        print(f"Summary saved: {summary_path}")

        if errors:
            print(f"\nErrors: {len(errors)}")
            for err in errors[:5]:
                print(f"  - {err}")

        return {'marker': marker_name, 'results': all_results, 'errors': errors}
    else:
        print(f"\nERROR: No files successfully processed for {marker_name}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Universal bleed-through correction')
    parser.add_argument('--input-dir', type=str, default='/Users/cjsogn/Marker analysis',
                       help='Base directory containing marker folders')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Base output directory')
    parser.add_argument('--markers', nargs='+', default=None,
                       help='Specific markers to process (default: all)')
    parser.add_argument('--cores', type=int, default=14,
                       help='Number of cores to use (default: 14)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = input_dir

    # Default markers
    all_markers = ['EAAT1', 'GAD67', 'GAT1', 'GAT3', 'GLS']

    if args.markers:
        markers_to_process = args.markers
    else:
        markers_to_process = all_markers

    print(f"\n{'='*60}")
    print("UNIVERSAL BLEED-THROUGH CORRECTION")
    print(f"{'='*60}")
    print(f"Method: Pixel-wise Linear Regression (dual-labeled images)")
    print(f"Markers: {', '.join(markers_to_process)}")
    print(f"Cores: {args.cores}")
    print(f"Visualization: {not args.no_viz}")

    # Process each marker
    all_marker_results = {}
    for marker in markers_to_process:
        try:
            result = process_marker(marker, input_dir, base_output_dir,
                                   n_cores=args.cores, visualize=not args.no_viz)
            if result:
                all_marker_results[marker] = result
        except Exception as e:
            print(f"\nERROR processing {marker}: {e}")
            import traceback
            traceback.print_exc()

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    for marker, data in all_marker_results.items():
        n_files = len(data['results'])
        fractions = [r['bleedthrough_fraction'] for r in data['results']]
        print(f"{marker:10s}: {n_files:3d} files, {np.mean(fractions)*100:5.2f}% ± {np.std(fractions)*100:4.2f}% bleed-through")

    # Save overall summary
    overall_summary_path = base_output_dir / 'all_markers_summary.json'
    with open(overall_summary_path, 'w') as f:
        json.dump({k: {'results': v['results'], 'errors': v['errors']}
                  for k, v in all_marker_results.items()}, f, indent=2)
    print(f"\nOverall summary saved: {overall_summary_path}")

if __name__ == '__main__':
    main()
