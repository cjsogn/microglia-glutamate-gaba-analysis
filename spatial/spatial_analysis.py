#!/usr/bin/env python3
"""
Spatial Analysis of Microglia-Terminal Relationships
=====================================================
Analyzes whether GABA/glutamate immunogold labeling in microglia correlates
with proximity to symmetric (GABA) or asymmetric (glutamate) terminals
using EM distance-density correlations and Spearman rho.

Manuscript figures: Fig 1E-H
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy import stats
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = "/Users/cjsogn/2.11.15 iba1 glut gaba 86956"
OUTPUT_DIR = "/Users/cjsogn/microglia_spatial_analysis"

# Animal mapping
ANIMALS = {
    # GLUT series
    'E1 iba glut GA1 hippo': {'animal': 'GA1', 'treatment': 'Control', 'antibody': 'GLUT'},
    'E2 iba glut GA2 hippo': {'animal': 'GA2', 'treatment': 'Control', 'antibody': 'GLUT'},
    'E3 iba1 glut GA3 hippo': {'animal': 'GA3', 'treatment': 'Control', 'antibody': 'GLUT'},
    'E4 iba glut GA4 hippo': {'animal': 'GA4', 'treatment': 'Control', 'antibody': 'GLUT'},
    'F1 iba glut LPS1 hippo': {'animal': 'LPS1', 'treatment': 'LPS', 'antibody': 'GLUT'},
    'F2 iba glut LPS2 hippo': {'animal': 'LPS2', 'treatment': 'LPS', 'antibody': 'GLUT'},
    # GABA series
    'J1 iba gaba GA1 hippo': {'animal': 'GA1', 'treatment': 'Control', 'antibody': 'GABA'},
    'J2 iba gaba GA2 hippo': {'animal': 'GA2', 'treatment': 'Control', 'antibody': 'GABA'},
    'J3 iba gaba GA3 hippo': {'animal': 'GA3', 'treatment': 'Control', 'antibody': 'GABA'},
    'J4 iba gaba GA4 hippo': {'animal': 'GA4', 'treatment': 'Control', 'antibody': 'GABA'},
    'K1 iba gaba LPS1 hippo': {'animal': 'LPS1', 'treatment': 'LPS', 'antibody': 'GABA'},
    'K2 iba gaba LPS2 hippo': {'animal': 'LPS2', 'treatment': 'LPS', 'antibody': 'GABA'},
}

# Compartment mapping (folder names may vary - includes Norwegian variations)
COMPARTMENT_FOLDERS = {
    'microglia': ['mikroglia', 'Mikroglia', 'microglia', 'Microglia'],
    'gaba_terminal': ['GABA terminals', 'gaba terminals', 'GABA terminal', 'symmetric',
                      'GABA terminaler', 'gaba terminaler'],
    'glut_terminal': ['glutamate terminal', 'Glutamate terminal', 'GLUT terminal', 'asymmetric',
                      'GLUT terminaler', 'glut terminaler'],
    'spine': ['spine', 'Spine', 'spines', 'spina']
}


def parse_pd_file(filepath):
    """Parse a Point Density .pd file and extract profile data."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract image name
    image_match = re.search(r'IMAGE\s+(.+)', content)
    image_name = image_match.group(1).strip() if image_match else None

    # Extract profile ID
    profile_match = re.search(r'PROFILE_ID\s+(\d+)', content)
    profile_id = int(profile_match.group(1)) if profile_match else None

    # Extract pixel width
    pixel_match = re.search(r'PIXELWIDTH\s+([\d.]+)\s*nm', content)
    pixel_width = float(pixel_match.group(1)) if pixel_match else 1.0

    # Extract profile border coordinates
    border_match = re.search(r'PROFILE_BORDER\s*\n(.*?)END', content, re.DOTALL)
    border_coords = []
    if border_match:
        for line in border_match.group(1).strip().split('\n'):
            line = line.strip()
            if line and ',' in line:
                coords = line.split(',')
                if len(coords) >= 2:
                    try:
                        x, y = float(coords[0].strip()), float(coords[1].strip())
                        border_coords.append((x * pixel_width, y * pixel_width))  # Convert to nm
                    except ValueError:
                        pass

    # Extract particle (gold) coordinates
    points_match = re.search(r'POINTS\s*\n(.*?)END', content, re.DOTALL)
    particle_coords = []
    if points_match:
        for line in points_match.group(1).strip().split('\n'):
            line = line.strip()
            if line and ',' in line:
                coords = line.split(',')
                if len(coords) >= 2:
                    try:
                        x, y = float(coords[0].strip()), float(coords[1].strip())
                        particle_coords.append((x * pixel_width, y * pixel_width))  # Convert to nm
                    except ValueError:
                        pass

    return {
        'image': image_name,
        'profile_id': profile_id,
        'pixel_width': pixel_width,
        'border': border_coords,
        'particles': particle_coords,
        'n_particles': len(particle_coords)
    }


def normalize_image_name(image_name):
    """
    Normalize image names to match across different naming conventions.
    E.g., 'Tv10 (2).tif' and 'Tv10.tif' should match as the same underlying image.
    """
    if image_name is None:
        return None
    # Remove variations like " (2)" or ".a.a" from the name
    normalized = re.sub(r'\s*\(\d+\)', '', image_name)  # Remove " (2)" etc
    normalized = re.sub(r'\.a\.a\.', '.', normalized)   # Remove ".a.a."
    normalized = normalized.strip()
    return normalized


def find_compartment_folder(animal_path, compartment):
    """Find the actual folder name for a compartment."""
    analyse_path = os.path.join(animal_path, 'analyse')
    if not os.path.exists(analyse_path):
        analyse_path = os.path.join(animal_path, 'analysert')

    if not os.path.exists(analyse_path):
        return None

    for folder_name in COMPARTMENT_FOLDERS[compartment]:
        full_path = os.path.join(analyse_path, folder_name)
        if os.path.exists(full_path):
            return full_path
    return None


def calculate_polygon_properties(coords):
    """Calculate centroid and area of a polygon."""
    if len(coords) < 3:
        return None, None, None

    try:
        # Close the polygon if needed
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]

        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)  # Try to fix invalid polygon

        centroid = poly.centroid
        return (centroid.x, centroid.y), poly.area, poly
    except:
        return None, None, None


def calculate_min_distance(poly1, poly2):
    """Calculate minimum distance between two polygons."""
    try:
        if poly1 is None or poly2 is None:
            return np.nan

        if poly1.intersects(poly2):
            return 0.0  # Overlapping/touching

        return poly1.distance(poly2)
    except:
        return np.nan


def extract_all_profiles():
    """Extract all profiles from all animals and compartments."""
    all_profiles = []

    for folder_name, info in ANIMALS.items():
        animal_path = os.path.join(BASE_DIR, folder_name)
        if not os.path.exists(animal_path):
            print(f"Warning: Animal folder not found: {folder_name}")
            continue

        for compartment in ['microglia', 'gaba_terminal', 'glut_terminal', 'spine']:
            comp_path = find_compartment_folder(animal_path, compartment)
            if comp_path is None:
                continue

            # Find all .pd files
            pd_files = glob.glob(os.path.join(comp_path, '*.pd'))

            for pd_file in pd_files:
                try:
                    data = parse_pd_file(pd_file)

                    if data['border'] and len(data['border']) >= 3:
                        centroid, area, polygon = calculate_polygon_properties(data['border'])

                        # Calculate particle density
                        if area and area > 0:
                            density = data['n_particles'] / (area / 1e6)  # particles/μm²
                        else:
                            density = np.nan

                        all_profiles.append({
                            'folder': folder_name,
                            'animal': info['animal'],
                            'treatment': info['treatment'],
                            'antibody': info['antibody'],
                            'compartment': compartment,
                            'image': data['image'],
                            'image_normalized': normalize_image_name(data['image']),
                            'profile_id': data['profile_id'],
                            'centroid_x': centroid[0] if centroid else np.nan,
                            'centroid_y': centroid[1] if centroid else np.nan,
                            'area_nm2': area if area else np.nan,
                            'area_um2': area / 1e6 if area else np.nan,
                            'n_particles': data['n_particles'],
                            'density': density,
                            'border': data['border'],
                            'polygon': polygon,
                            'file': pd_file
                        })
                except Exception as e:
                    print(f"Error parsing {pd_file}: {e}")

    return pd.DataFrame(all_profiles)


def calculate_spatial_relationships(df):
    """Calculate distances between microglia and terminals on the same image."""
    relationships = []

    # Group by folder and normalized image name (to match across naming conventions)
    grouped = df.groupby(['folder', 'image_normalized'])

    for (folder, image), group in grouped:
        # Get microglia profiles
        microglia = group[group['compartment'] == 'microglia']
        gaba_terminals = group[group['compartment'] == 'gaba_terminal']
        glut_terminals = group[group['compartment'] == 'glut_terminal']

        for _, mg in microglia.iterrows():
            mg_poly = mg['polygon']
            if mg_poly is None:
                continue

            # Calculate distance to nearest GABA terminal
            min_dist_gaba = np.inf
            apposed_gaba = False
            for _, gt in gaba_terminals.iterrows():
                gt_poly = gt['polygon']
                if gt_poly is not None:
                    dist = calculate_min_distance(mg_poly, gt_poly)
                    if not np.isnan(dist) and dist < min_dist_gaba:
                        min_dist_gaba = dist
                        if dist == 0:
                            apposed_gaba = True

            # Calculate distance to nearest glutamate terminal
            min_dist_glut = np.inf
            apposed_glut = False
            for _, gt in glut_terminals.iterrows():
                gt_poly = gt['polygon']
                if gt_poly is not None:
                    dist = calculate_min_distance(mg_poly, gt_poly)
                    if not np.isnan(dist) and dist < min_dist_glut:
                        min_dist_glut = dist
                        if dist == 0:
                            apposed_glut = True

            # Handle infinite distances (no terminal found)
            if min_dist_gaba == np.inf:
                min_dist_gaba = np.nan
            if min_dist_glut == np.inf:
                min_dist_glut = np.nan

            # Categorize proximity
            # Apposed = touching (distance = 0)
            # Close = within 500 nm (typical synaptic cleft + some margin)
            # Distant = > 500 nm
            CLOSE_THRESHOLD = 500  # nm

            if not np.isnan(min_dist_gaba):
                if min_dist_gaba == 0:
                    gaba_proximity = 'apposed'
                elif min_dist_gaba <= CLOSE_THRESHOLD:
                    gaba_proximity = 'close'
                else:
                    gaba_proximity = 'distant'
            else:
                gaba_proximity = 'no_terminal'

            if not np.isnan(min_dist_glut):
                if min_dist_glut == 0:
                    glut_proximity = 'apposed'
                elif min_dist_glut <= CLOSE_THRESHOLD:
                    glut_proximity = 'close'
                else:
                    glut_proximity = 'distant'
            else:
                glut_proximity = 'no_terminal'

            # Determine which terminal type is closer
            if np.isnan(min_dist_gaba) and np.isnan(min_dist_glut):
                closer_terminal = 'none'
            elif np.isnan(min_dist_gaba):
                closer_terminal = 'glutamate'
            elif np.isnan(min_dist_glut):
                closer_terminal = 'gaba'
            elif min_dist_gaba < min_dist_glut:
                closer_terminal = 'gaba'
            elif min_dist_glut < min_dist_gaba:
                closer_terminal = 'glutamate'
            else:
                closer_terminal = 'equal'

            relationships.append({
                'folder': mg['folder'],
                'animal': mg['animal'],
                'treatment': mg['treatment'],
                'antibody': mg['antibody'],
                'image': mg['image'],
                'profile_id': mg['profile_id'],
                'mg_density': mg['density'],
                'mg_area_um2': mg['area_um2'],
                'mg_n_particles': mg['n_particles'],
                'dist_to_gaba_terminal': min_dist_gaba,
                'dist_to_glut_terminal': min_dist_glut,
                'apposed_gaba': apposed_gaba,
                'apposed_glut': apposed_glut,
                'gaba_proximity': gaba_proximity,
                'glut_proximity': glut_proximity,
                'closer_terminal': closer_terminal,
                'n_gaba_terminals': len(gaba_terminals),
                'n_glut_terminals': len(glut_terminals)
            })

    return pd.DataFrame(relationships)


def run_statistical_tests(df_spatial):
    """Run statistical tests on spatial relationships."""
    results = {}

    # =========================================================================
    # HYPOTHESIS 1: GABA in microglia vs proximity to symmetric (GABA) terminals
    # =========================================================================
    gaba_data = df_spatial[df_spatial['antibody'] == 'GABA'].copy()
    gaba_data = gaba_data[gaba_data['treatment'] == 'Control']  # Focus on controls first

    print("\n" + "="*70)
    print("HYPOTHESIS 1: GABA labeling in microglia vs proximity to GABA terminals")
    print("="*70)

    # Test 1a: Apposed vs non-apposed
    if 'gaba_proximity' in gaba_data.columns:
        apposed = gaba_data[gaba_data['gaba_proximity'] == 'apposed']['mg_density'].dropna()
        not_apposed = gaba_data[gaba_data['gaba_proximity'] != 'apposed']['mg_density'].dropna()

        if len(apposed) > 0 and len(not_apposed) > 0:
            print(f"\nApposed to GABA terminal (n={len(apposed)}): {apposed.mean():.2f} ± {apposed.std():.2f} particles/μm²")
            print(f"Not apposed (n={len(not_apposed)}): {not_apposed.mean():.2f} ± {not_apposed.std():.2f} particles/μm²")

            stat, p = stats.mannwhitneyu(apposed, not_apposed, alternative='two-sided')
            print(f"Mann-Whitney U test: U={stat:.1f}, p={p:.4f}")

            results['gaba_apposed_vs_not'] = {
                'apposed_n': len(apposed),
                'apposed_mean': apposed.mean(),
                'apposed_std': apposed.std(),
                'not_apposed_n': len(not_apposed),
                'not_apposed_mean': not_apposed.mean(),
                'not_apposed_std': not_apposed.std(),
                'mann_whitney_U': stat,
                'p_value': p
            }

    # Test 1b: Correlation with distance
    valid_dist = gaba_data[gaba_data['dist_to_gaba_terminal'].notna() &
                           gaba_data['mg_density'].notna()].copy()

    if len(valid_dist) > 5:
        r, p = stats.spearmanr(valid_dist['dist_to_gaba_terminal'], valid_dist['mg_density'])
        print(f"\nSpearman correlation (GABA density vs distance to GABA terminal):")
        print(f"  r = {r:.3f}, p = {p:.4f}")
        print(f"  Interpretation: {'Negative (closer = more GABA)' if r < 0 else 'Positive (farther = more GABA)'}")

        results['gaba_distance_correlation'] = {
            'n': len(valid_dist),
            'spearman_r': r,
            'p_value': p
        }

    # Test 1c: Closer to GABA vs closer to Glut terminal
    closer_gaba = gaba_data[gaba_data['closer_terminal'] == 'gaba']['mg_density'].dropna()
    closer_glut = gaba_data[gaba_data['closer_terminal'] == 'glutamate']['mg_density'].dropna()

    if len(closer_gaba) > 0 and len(closer_glut) > 0:
        print(f"\nCloser to GABA terminal (n={len(closer_gaba)}): {closer_gaba.mean():.2f} ± {closer_gaba.std():.2f}")
        print(f"Closer to GLUT terminal (n={len(closer_glut)}): {closer_glut.mean():.2f} ± {closer_glut.std():.2f}")

        stat, p = stats.mannwhitneyu(closer_gaba, closer_glut, alternative='two-sided')
        print(f"Mann-Whitney U test: U={stat:.1f}, p={p:.4f}")

        results['gaba_closer_terminal_type'] = {
            'closer_gaba_n': len(closer_gaba),
            'closer_gaba_mean': closer_gaba.mean(),
            'closer_glut_n': len(closer_glut),
            'closer_glut_mean': closer_glut.mean(),
            'mann_whitney_U': stat,
            'p_value': p
        }

    # =========================================================================
    # HYPOTHESIS 2: Glutamate in microglia vs proximity to asymmetric terminals
    # =========================================================================
    glut_data = df_spatial[df_spatial['antibody'] == 'GLUT'].copy()
    glut_data = glut_data[glut_data['treatment'] == 'Control']

    print("\n" + "="*70)
    print("HYPOTHESIS 2: Glutamate labeling in microglia vs proximity to GLUT terminals")
    print("="*70)

    # Test 2a: Apposed vs non-apposed
    if 'glut_proximity' in glut_data.columns:
        apposed = glut_data[glut_data['glut_proximity'] == 'apposed']['mg_density'].dropna()
        not_apposed = glut_data[glut_data['glut_proximity'] != 'apposed']['mg_density'].dropna()

        if len(apposed) > 0 and len(not_apposed) > 0:
            print(f"\nApposed to GLUT terminal (n={len(apposed)}): {apposed.mean():.2f} ± {apposed.std():.2f} particles/μm²")
            print(f"Not apposed (n={len(not_apposed)}): {not_apposed.mean():.2f} ± {not_apposed.std():.2f} particles/μm²")

            stat, p = stats.mannwhitneyu(apposed, not_apposed, alternative='two-sided')
            print(f"Mann-Whitney U test: U={stat:.1f}, p={p:.4f}")

            results['glut_apposed_vs_not'] = {
                'apposed_n': len(apposed),
                'apposed_mean': apposed.mean(),
                'apposed_std': apposed.std(),
                'not_apposed_n': len(not_apposed),
                'not_apposed_mean': not_apposed.mean(),
                'not_apposed_std': not_apposed.std(),
                'mann_whitney_U': stat,
                'p_value': p
            }

    # Test 2b: Correlation with distance
    valid_dist = glut_data[glut_data['dist_to_glut_terminal'].notna() &
                           glut_data['mg_density'].notna()].copy()

    if len(valid_dist) > 5:
        r, p = stats.spearmanr(valid_dist['dist_to_glut_terminal'], valid_dist['mg_density'])
        print(f"\nSpearman correlation (GLUT density vs distance to GLUT terminal):")
        print(f"  r = {r:.3f}, p = {p:.4f}")

        results['glut_distance_correlation'] = {
            'n': len(valid_dist),
            'spearman_r': r,
            'p_value': p
        }

    # Test 2c: Closer to GLUT vs closer to GABA terminal
    closer_glut = glut_data[glut_data['closer_terminal'] == 'glutamate']['mg_density'].dropna()
    closer_gaba = glut_data[glut_data['closer_terminal'] == 'gaba']['mg_density'].dropna()

    if len(closer_glut) > 0 and len(closer_gaba) > 0:
        print(f"\nCloser to GLUT terminal (n={len(closer_glut)}): {closer_glut.mean():.2f} ± {closer_glut.std():.2f}")
        print(f"Closer to GABA terminal (n={len(closer_gaba)}): {closer_gaba.mean():.2f} ± {closer_gaba.std():.2f}")

        stat, p = stats.mannwhitneyu(closer_glut, closer_gaba, alternative='two-sided')
        print(f"Mann-Whitney U test: U={stat:.1f}, p={p:.4f}")

        results['glut_closer_terminal_type'] = {
            'closer_glut_n': len(closer_glut),
            'closer_glut_mean': closer_glut.mean(),
            'closer_gaba_n': len(closer_gaba),
            'closer_gaba_mean': closer_gaba.mean(),
            'mann_whitney_U': stat,
            'p_value': p
        }

    return results


def run_animal_level_analysis(df_spatial):
    """Aggregate to animal level and run tests (proper experimental unit)."""
    print("\n" + "="*70)
    print("ANIMAL-LEVEL ANALYSIS (Proper experimental unit)")
    print("="*70)

    results = {}

    # GABA analysis - aggregate to animal level
    gaba_data = df_spatial[(df_spatial['antibody'] == 'GABA') &
                           (df_spatial['treatment'] == 'Control')].copy()

    if len(gaba_data) > 0:
        # For each animal, calculate mean density for apposed vs not apposed
        animal_summary = []

        for animal in gaba_data['animal'].unique():
            animal_data = gaba_data[gaba_data['animal'] == animal]

            apposed = animal_data[animal_data['gaba_proximity'] == 'apposed']['mg_density']
            not_apposed = animal_data[animal_data['gaba_proximity'] != 'apposed']['mg_density']

            animal_summary.append({
                'animal': animal,
                'apposed_mean': apposed.mean() if len(apposed) > 0 else np.nan,
                'apposed_n': len(apposed),
                'not_apposed_mean': not_apposed.mean() if len(not_apposed) > 0 else np.nan,
                'not_apposed_n': len(not_apposed)
            })

        animal_df = pd.DataFrame(animal_summary)
        print("\nGABA - Animal-level summary:")
        print(animal_df.to_string(index=False))

        # Paired test if both conditions present in each animal
        paired_data = animal_df.dropna()
        if len(paired_data) >= 2:
            stat, p = stats.wilcoxon(paired_data['apposed_mean'], paired_data['not_apposed_mean'])
            print(f"\nWilcoxon signed-rank test: W={stat:.1f}, p={p:.4f}")
            results['gaba_animal_level'] = {'test': 'wilcoxon', 'W': stat, 'p': p}

    # GLUT analysis
    glut_data = df_spatial[(df_spatial['antibody'] == 'GLUT') &
                           (df_spatial['treatment'] == 'Control')].copy()

    if len(glut_data) > 0:
        animal_summary = []

        for animal in glut_data['animal'].unique():
            animal_data = glut_data[glut_data['animal'] == animal]

            apposed = animal_data[animal_data['glut_proximity'] == 'apposed']['mg_density']
            not_apposed = animal_data[animal_data['glut_proximity'] != 'apposed']['mg_density']

            animal_summary.append({
                'animal': animal,
                'apposed_mean': apposed.mean() if len(apposed) > 0 else np.nan,
                'apposed_n': len(apposed),
                'not_apposed_mean': not_apposed.mean() if len(not_apposed) > 0 else np.nan,
                'not_apposed_n': len(not_apposed)
            })

        animal_df = pd.DataFrame(animal_summary)
        print("\nGLUT - Animal-level summary:")
        print(animal_df.to_string(index=False))

        paired_data = animal_df.dropna()
        if len(paired_data) >= 2:
            stat, p = stats.wilcoxon(paired_data['apposed_mean'], paired_data['not_apposed_mean'])
            print(f"\nWilcoxon signed-rank test: W={stat:.1f}, p={p:.4f}")
            results['glut_animal_level'] = {'test': 'wilcoxon', 'W': stat, 'p': p}

    return results


def create_figures(df_spatial):
    """Create publication-quality figures."""

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.2

    # Color palette
    colors = {'apposed': '#E74C3C', 'close': '#F39C12', 'distant': '#3498DB', 'no_terminal': '#95A5A6'}

    # =========================================================================
    # Figure 1: GABA in microglia by proximity to GABA terminals
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    gaba_ctrl = df_spatial[(df_spatial['antibody'] == 'GABA') &
                           (df_spatial['treatment'] == 'Control')].copy()

    # Panel A: Boxplot by proximity category
    ax = axes[0]
    order = ['apposed', 'close', 'distant']
    plot_data = gaba_ctrl[gaba_ctrl['gaba_proximity'].isin(order)]

    if len(plot_data) > 0:
        sns.boxplot(data=plot_data, x='gaba_proximity', y='mg_density',
                   order=order, palette=[colors[x] for x in order], ax=ax)
        sns.stripplot(data=plot_data, x='gaba_proximity', y='mg_density',
                     order=order, color='black', alpha=0.5, size=4, ax=ax)

    ax.set_xlabel('Proximity to GABA Terminal')
    ax.set_ylabel('GABA Density in Microglia\n(particles/μm²)')
    ax.set_title('A. GABA by Terminal Proximity')

    # Panel B: Scatter plot - distance vs density
    ax = axes[1]
    valid = gaba_ctrl[gaba_ctrl['dist_to_gaba_terminal'].notna() &
                      gaba_ctrl['mg_density'].notna()].copy()

    if len(valid) > 0:
        ax.scatter(valid['dist_to_gaba_terminal']/1000, valid['mg_density'],
                  alpha=0.6, c='#E74C3C', edgecolor='white', linewidth=0.5)

        # Add trend line
        if len(valid) > 5:
            z = np.polyfit(valid['dist_to_gaba_terminal']/1000, valid['mg_density'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid['dist_to_gaba_terminal'].min()/1000,
                                valid['dist_to_gaba_terminal'].max()/1000, 100)
            ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.5)

            r, pval = stats.spearmanr(valid['dist_to_gaba_terminal'], valid['mg_density'])
            ax.text(0.95, 0.95, f'r = {r:.2f}\np = {pval:.3f}',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Distance to Nearest GABA Terminal (μm)')
    ax.set_ylabel('GABA Density in Microglia\n(particles/μm²)')
    ax.set_title('B. GABA vs Distance')

    # Panel C: Closer to GABA vs GLUT terminal
    ax = axes[2]
    closer_data = gaba_ctrl[gaba_ctrl['closer_terminal'].isin(['gaba', 'glutamate'])]

    if len(closer_data) > 0:
        sns.boxplot(data=closer_data, x='closer_terminal', y='mg_density',
                   order=['gaba', 'glutamate'],
                   palette=['#E74C3C', '#3498DB'], ax=ax)
        sns.stripplot(data=closer_data, x='closer_terminal', y='mg_density',
                     order=['gaba', 'glutamate'], color='black', alpha=0.5, size=4, ax=ax)

    ax.set_xlabel('Closer Terminal Type')
    ax.set_ylabel('GABA Density in Microglia\n(particles/μm²)')
    ax.set_xticklabels(['GABA\n(Symmetric)', 'Glutamate\n(Asymmetric)'])
    ax.set_title('C. GABA by Nearest Terminal Type')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'Fig1_GABA_spatial.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'Fig1_GABA_spatial.pdf'), bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 2: Glutamate in microglia by proximity to GLUT terminals
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    glut_ctrl = df_spatial[(df_spatial['antibody'] == 'GLUT') &
                           (df_spatial['treatment'] == 'Control')].copy()

    # Panel A: Boxplot by proximity category
    ax = axes[0]
    order = ['apposed', 'close', 'distant']
    plot_data = glut_ctrl[glut_ctrl['glut_proximity'].isin(order)]

    if len(plot_data) > 0:
        sns.boxplot(data=plot_data, x='glut_proximity', y='mg_density',
                   order=order, palette=[colors[x] for x in order], ax=ax)
        sns.stripplot(data=plot_data, x='glut_proximity', y='mg_density',
                     order=order, color='black', alpha=0.5, size=4, ax=ax)

    ax.set_xlabel('Proximity to Glutamate Terminal')
    ax.set_ylabel('Glutamate Density in Microglia\n(particles/μm²)')
    ax.set_title('A. Glutamate by Terminal Proximity')

    # Panel B: Scatter plot
    ax = axes[1]
    valid = glut_ctrl[glut_ctrl['dist_to_glut_terminal'].notna() &
                      glut_ctrl['mg_density'].notna()].copy()

    if len(valid) > 0:
        ax.scatter(valid['dist_to_glut_terminal']/1000, valid['mg_density'],
                  alpha=0.6, c='#3498DB', edgecolor='white', linewidth=0.5)

        if len(valid) > 5:
            z = np.polyfit(valid['dist_to_glut_terminal']/1000, valid['mg_density'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid['dist_to_glut_terminal'].min()/1000,
                                valid['dist_to_glut_terminal'].max()/1000, 100)
            ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.5)

            r, pval = stats.spearmanr(valid['dist_to_glut_terminal'], valid['mg_density'])
            ax.text(0.95, 0.95, f'r = {r:.2f}\np = {pval:.3f}',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Distance to Nearest GLUT Terminal (μm)')
    ax.set_ylabel('Glutamate Density in Microglia\n(particles/μm²)')
    ax.set_title('B. Glutamate vs Distance')

    # Panel C: Closer to GLUT vs GABA terminal
    ax = axes[2]
    closer_data = glut_ctrl[glut_ctrl['closer_terminal'].isin(['gaba', 'glutamate'])]

    if len(closer_data) > 0:
        sns.boxplot(data=closer_data, x='closer_terminal', y='mg_density',
                   order=['glutamate', 'gaba'],
                   palette=['#3498DB', '#E74C3C'], ax=ax)
        sns.stripplot(data=closer_data, x='closer_terminal', y='mg_density',
                     order=['glutamate', 'gaba'], color='black', alpha=0.5, size=4, ax=ax)

    ax.set_xlabel('Closer Terminal Type')
    ax.set_ylabel('Glutamate Density in Microglia\n(particles/μm²)')
    ax.set_xticklabels(['Glutamate\n(Asymmetric)', 'GABA\n(Symmetric)'])
    ax.set_title('C. Glutamate by Nearest Terminal Type')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'Fig2_GLUT_spatial.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'Fig2_GLUT_spatial.pdf'), bbox_inches='tight')
    plt.close()

    print("\nFigures saved to:", os.path.join(OUTPUT_DIR, 'figures'))


def main():
    print("="*70)
    print("SPATIAL ANALYSIS OF MICROGLIA-TERMINAL RELATIONSHIPS")
    print("="*70)

    # Step 1: Extract all profiles
    print("\n[1/5] Extracting profiles from all animals...")
    df_profiles = extract_all_profiles()
    print(f"    Extracted {len(df_profiles)} profiles")

    # Summary by compartment
    print("\n    Profile counts by compartment:")
    for comp in ['microglia', 'gaba_terminal', 'glut_terminal', 'spine']:
        n = len(df_profiles[df_profiles['compartment'] == comp])
        print(f"      {comp}: {n}")

    # Save profile data (without polygon objects)
    df_profiles_save = df_profiles.drop(columns=['border', 'polygon', 'file'])
    df_profiles_save.to_csv(os.path.join(OUTPUT_DIR, 'data', 'all_profiles.csv'), index=False)

    # Step 2: Calculate spatial relationships
    print("\n[2/5] Calculating spatial relationships...")
    df_spatial = calculate_spatial_relationships(df_profiles)
    print(f"    Calculated relationships for {len(df_spatial)} microglia profiles")

    # Save spatial data
    df_spatial.to_csv(os.path.join(OUTPUT_DIR, 'data', 'spatial_relationships.csv'), index=False)

    # Step 3: Run statistical tests (profile level)
    print("\n[3/5] Running profile-level statistical tests...")
    profile_results = run_statistical_tests(df_spatial)

    # Step 4: Run animal-level analysis
    print("\n[4/5] Running animal-level analysis...")
    animal_results = run_animal_level_analysis(df_spatial)

    # Save results
    results_df = pd.DataFrame([
        {'analysis': k, **{f'{k2}': v2 for k2, v2 in v.items()}}
        for k, v in {**profile_results, **animal_results}.items()
    ])
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'results', 'statistical_results.csv'), index=False)

    # Step 5: Create figures
    print("\n[5/5] Creating figures...")
    create_figures(df_spatial)

    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print("  - data/all_profiles.csv")
    print("  - data/spatial_relationships.csv")
    print("  - results/statistical_results.csv")
    print("  - figures/Fig1_GABA_spatial.png/pdf")
    print("  - figures/Fig2_GLUT_spatial.png/pdf")

    return df_profiles, df_spatial


if __name__ == '__main__':
    df_profiles, df_spatial = main()
