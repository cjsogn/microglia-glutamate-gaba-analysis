#!/usr/bin/env python3
"""
Extract per-cell tables from the Han Stereo-seq mouse brain atlas.

Writes one Parquet file per section for microglia, astrocytes and neurons,
holding the cell centroid, its Allen region assignment and counts for the
glutamate- and GABA-handling genes analysed in the paper.

Neurons are typed from the transcriptome-wide cell-type annotations supplied
with the atlas rather than by thresholding a single marker. The vesicular
glutamate transporter genes are carried through so that alternative
marker-based definitions of the excitatory class can be tested downstream.

Output feeds coexpression.py, proximity_statistics.py and
regional_stratification.py.

Usage
    STEREOSEQ_DIR=/path/to/han_brain_stereo_seq python extract_stereoseq_cells.py

Environment
    STEREOSEQ_DIR   raw atlas directory
    MICROGLIA_DATA  where the per-section tables are written (default: data)
    N_CPUS          worker processes (default: 14)
"""

import argparse
import gzip
import os
import pickle
import sys
import time
from collections import defaultdict
from glob import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

STEREO_DIR = os.environ.get('STEREOSEQ_DIR', 'raw/han_brain_stereo_seq')
OUT_DIR = os.path.join(os.environ.get('MICROGLIA_DATA', 'data'), 'stereoseq')
N_CPUS = int(os.environ.get('N_CPUS', '14'))

# glutamate / GABA handling machinery (identical to the published analysis)
GLUTAMATE_GENES = {
    'EAAT1': 'Slc1a3',
    'EAAT2': 'Slc1a2',
    'GLS': 'Gls',
    'GLUL': 'Glul',
    'SNAT1': 'Slc38a1',
}
GABA_GENES = {
    'GAT1': 'Slc6a1',
    'GAT3': 'Slc6a11',
    'GAD1': 'Gad1',
    'GAD2': 'Gad2',
    'ABAT': 'Abat',
}
HANDLING_GENES = sorted(set(GLUTAMATE_GENES.values()) | set(GABA_GENES.values()))

# neuronal identity genes.  Gad1/Gad2 are already in HANDLING_GENES; the
# vesicular glutamate transporters are new for this revision.
VGLUT_GENES = ['Slc17a6', 'Slc17a7', 'Slc17a8']
IDENTITY_GENES = VGLUT_GENES + ['Gad1', 'Gad2']

ALL_GENES = sorted(set(HANDLING_GENES) | set(IDENTITY_GENES))

MIN_MICROGLIA_PER_SECTION = 5
MIN_NEURONS_PER_TYPE = 5

RETRIES = 4                  # external drive has dropped mid-run before
RETRY_WAIT_S = 20


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def split_cell_annotations(stereo_dir, out_dir, mouse_id='mouse1'):
    """
    Write one small annotation pickle per section.

    The full table is ~6M cells.  Loading it inside every worker put 14
    multi-GB copies in RAM simultaneously and exhausted a 24 GB machine, so each
    worker now loads only the ~50k rows belonging to its own section.
    """
    ann_dir = os.path.join(out_dir, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)

    done = glob(os.path.join(ann_dir, '*.pkl'))
    if done:
        print(f'  reusing {len(done)} cached per-section annotation files')
        return ann_dir

    path = os.path.join(stereo_dir, 'stereoseq.celltypeTransfer.2mice.all.tsv')
    df = pd.read_csv(path, sep='\t')
    df = df[df['mouse'] == mouse_id]
    print(f'  {len(df):,} annotated cells for {mouse_id}')

    for section_id, grp in df.groupby('section_id', sort=False):
        lookup = {int(cid): (cls, clus, sub) for cid, clus, sub, cls in
                  zip(grp['cell_id'], grp['cell_cluster'],
                      grp['cell_subclass'], grp['cell_class'])}
        with open(os.path.join(ann_dir, f'{section_id}.pkl'), 'wb') as fh:
            pickle.dump(lookup, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  wrote {df["section_id"].nunique()} per-section annotation files')
    return ann_dir


def load_region_names(stereo_dir):
    """gene_area code -> (area_name, description)."""
    path = os.path.join(stereo_dir, 'regions-mouse1-20230119.rmDup.tsv')
    df = pd.read_csv(path, sep='\t')
    return {int(a): (n, d) for a, n, d in
            zip(df['gene_area'], df['area_name'], df['description'])}


def classify_by_annotation(cell_cluster):
    """As published: Han et al. cluster naming convention."""
    if cell_cluster is None or (isinstance(cell_cluster, float) and np.isnan(cell_cluster)):
        return 'Other'
    up = str(cell_cluster).upper()
    if '_GLU_' in up:
        return 'Glutamatergic'
    if '_GABA_' in up:
        return 'GABAergic'
    return 'Other'


def classify_by_marker(gene_umi, vglut_genes):
    """
    Marker-based neuron classification.

    A neuron is called glutamatergic if it carries more vesicular glutamate
    transporter UMI than Gad1+Gad2 UMI, and GABAergic if the reverse.  Cells
    with no counts for either set are left unclassified rather than being
    forced into a class.
    """
    vglut = sum(gene_umi.get(g, 0) for g in vglut_genes)
    gad = gene_umi.get('Gad1', 0) + gene_umi.get('Gad2', 0)
    if vglut == 0 and gad == 0:
        return 'Other'
    if vglut > gad:
        return 'Glutamatergic'
    if gad > vglut:
        return 'GABAergic'
    return 'Other'      # tie


# --------------------------------------------------------------------------
# per-section worker
# --------------------------------------------------------------------------

def process_section(args):
    section_file, ann_dir = args
    section_id = os.path.basename(section_file).split('_')[2]

    ann_path = os.path.join(ann_dir, f'{section_id}.pkl')
    if not os.path.exists(ann_path):
        return section_id, None, 'no annotations for section'
    with open(ann_path, 'rb') as fh:
        cell_lookup = pickle.load(fh)          # keyed by cell_id within section

    # accumulate per cell.  UMI-weighted centroid: sum(x*umi) / sum(umi).
    cells = defaultdict(lambda: {
        'xw': 0.0, 'yw': 0.0, 'umi': 0,
        'genes': defaultdict(int),
        'areas': defaultdict(int),
    })

    gene_set = set(ALL_GENES)

    # The source files live on an external drive that has dropped off the bus
    # mid-run before ("Device not configured").  Retry rather than silently
    # skipping the section, and start the accumulator from scratch each attempt
    # so a partial read cannot corrupt the counts.
    last_exc = None
    for attempt in range(RETRIES):
        cells.clear()
        try:
            with gzip.open(section_file, 'rt') as fh:
                fh.readline()                              # header
                for line in fh:
                    p = line.rstrip('\n').split('\t')
                    if len(p) < 6:
                        continue
                    label = int(p[4])
                    if label == 0:                         # background spot
                        continue
                    umi = int(p[3])
                    c = cells[label]
                    c['xw'] += float(p[1]) * umi
                    c['yw'] += float(p[2]) * umi
                    c['umi'] += umi
                    c['areas'][int(p[5])] += umi
                    g = p[0]
                    if g in gene_set:
                        c['genes'][g] += umi
            break
        except Exception as exc:                           # noqa: BLE001
            last_exc = exc
            if attempt < RETRIES - 1:
                time.sleep(RETRY_WAIT_S * (attempt + 1))
    else:
        return section_id, None, f'read error after {RETRIES} attempts: {last_exc}'

    # ---- assemble records -------------------------------------------------
    microglia, astrocytes = [], []
    neurons = {
        'annotation': {'Glutamatergic': [], 'GABAergic': []},
        'vglut_all': {'Glutamatergic': [], 'GABAergic': []},
        'slc17a7':   {'Glutamatergic': [], 'GABAergic': []},
    }
    neuron_rows = []

    for label, c in cells.items():
        if c['umi'] == 0:
            continue
        info = cell_lookup.get(label)
        if info is None:
            continue
        cls, cluster, subclass = info

        cx = c['xw'] / c['umi']
        cy = c['yw'] / c['umi']
        area = max(c['areas'].items(), key=lambda kv: kv[1])[0]
        genes = dict(c['genes'])

        base = {
            'section_id': section_id,
            'cell_id': label,
            'x': cx, 'y': cy,
            'total_umi': c['umi'],
            'gene_area': area,
            'cell_class': cls,
            'cell_cluster': cluster,
            'cell_subclass': subclass,
        }
        for g in ALL_GENES:
            base[g] = genes.get(g, 0)

        if cls == 'Microglia':
            microglia.append(base)
        elif cls == 'Neurons':
            a = classify_by_annotation(cluster)
            if a in neurons['annotation']:
                neurons['annotation'][a].append([cx, cy])
            b = classify_by_marker(genes, VGLUT_GENES)
            if b in neurons['vglut_all']:
                neurons['vglut_all'][b].append([cx, cy])
            d = classify_by_marker(genes, ['Slc17a7'])
            if d in neurons['slc17a7']:
                neurons['slc17a7'][d].append([cx, cy])
            neuron_rows.append({
                'section_id': section_id, 'cell_id': label,
                'x': cx, 'y': cy,
                'gene_area': area, 'total_umi': c['umi'],
                'cell_cluster': cluster, 'cell_subclass': subclass,
                'class_annotation': a, 'class_vglut_all': b, 'class_slc17a7': d,
                **{g: genes.get(g, 0) for g in IDENTITY_GENES},
            })
        elif isinstance(cls, str) and 'Astrocytes' in cls:
            astrocytes.append(base)

    if len(microglia) < MIN_MICROGLIA_PER_SECTION:
        return section_id, None, 'too few microglia'

    # ---- nearest-neighbour distances under each definition ----------------
    trees = {}
    for defn, groups in neurons.items():
        for ntype, coords in groups.items():
            if len(coords) >= MIN_NEURONS_PER_TYPE:
                trees[(defn, ntype)] = cKDTree(np.asarray(coords))

    mg = pd.DataFrame(microglia)
    pts = mg[['x', 'y']].to_numpy()

    for defn in neurons:
        for ntype, short in (('Glutamatergic', 'glu'), ('GABAergic', 'gaba')):
            col = f'dist_{short}_{defn}'
            tree = trees.get((defn, ntype))
            if tree is None:
                mg[col] = np.nan
            else:
                mg[col] = tree.query(pts, k=1)[0]
            mg[f'n_{short}_{defn}'] = len(neurons[defn][ntype])

    ast = pd.DataFrame(astrocytes) if astrocytes else pd.DataFrame()
    neu = pd.DataFrame(neuron_rows) if neuron_rows else pd.DataFrame()

    os.makedirs(f'{OUT_DIR}/cells', exist_ok=True)
    mg.to_parquet(f'{OUT_DIR}/cells/microglia_{section_id}.parquet', index=False)
    if not ast.empty:
        ast.to_parquet(f'{OUT_DIR}/cells/astrocytes_{section_id}.parquet', index=False)
    if not neu.empty:
        neu.to_parquet(f'{OUT_DIR}/cells/neurons_{section_id}.parquet', index=False)

    return section_id, len(mg), None


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None,
                    help='process only the first N sections (smoke test)')
    ap.add_argument('--cpus', type=int, default=N_CPUS)
    ap.add_argument('--resume', action='store_true',
                    help='skip sections whose output already exists')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f'{OUT_DIR}/cells', exist_ok=True)

    files = sorted(f for f in glob(f'{STEREO_DIR}/total_gene_T*_mouse1-*.txt.gz')
                   if not os.path.basename(f).startswith('._'))
    if args.limit:
        files = files[:args.limit]

    if args.resume:
        before = len(files)
        files = [f for f in files if not os.path.exists(
            f'{OUT_DIR}/cells/microglia_'
            f'{os.path.basename(f).split("_")[2]}.parquet')]
        print(f'resume: {before - len(files)} sections already done, '
              f'{len(files)} remaining', flush=True)
    print(f'sections to process: {len(files)}', flush=True)

    print('preparing per-section cell annotations ...', flush=True)
    ann_dir = split_cell_annotations(STEREO_DIR, OUT_DIR)

    regions = load_region_names(STEREO_DIR)
    pd.DataFrame([{'gene_area': k, 'area_name': v[0], 'description': v[1]}
                  for k, v in regions.items()]).to_csv(
        f'{OUT_DIR}/region_lookup.csv', index=False)

    t0 = time.time()
    done = 0
    with Pool(args.cpus, maxtasksperchild=2) as pool:
        for section_id, n, err in pool.imap_unordered(
                process_section, [(f, ann_dir) for f in files], chunksize=1):
            done += 1
            el = time.time() - t0
            rate = el / done
            eta = rate * (len(files) - done)
            msg = f'[{done}/{len(files)}] {section_id}: '
            msg += f'{n} microglia' if n else f'SKIPPED ({err})'
            msg += f'  |  {el/60:.1f} min elapsed, ~{eta/60:.1f} min left'
            print(msg, flush=True)

    print(f'\ndone in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
