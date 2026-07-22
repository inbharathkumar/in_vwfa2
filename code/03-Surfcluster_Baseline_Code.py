"""
03-Surfcluster_Baseline_Code.py

HYPOTHESIS UNDER TEST (Jul 2026):
"Instead of the custom scaled-BFS / gradation clustering, plain FreeSurfer
mri_surfcluster (fixed t-value threshold + connected-component clustering on
the cortical mesh, minimum cluster size, no other tricks) would have found the
same VOTC ROIs -- including separate anterior ones -- and the existing Stage 2
scoring modules could have been run unchanged on top of it."

This script implements the mri_surfcluster ALGORITHM directly in Python
(threshold -> connected components on the surface graph -> minimum-size
filter), since the actual FreeSurfer binary is not installable in this
sandboxed environment and does not need to be: that is precisely the entirety
of what mri_surfcluster's core clustering step does. No seed selection, no
per-seed adaptive cutoff, no scaling/binning, and -- crucially -- no artificial
max-cluster-size cap: a supra-threshold blob is one cluster, however large.

Multi-session reliability (DICE) is still computed for parity with the
existing pipeline's scoring, but by *matching clusters that already exist*
across sessions via vertex overlap -- no growing, no rounds, no winner-takes-
vertices exclusion. That bookkeeping is orthogonal to the clustering-algorithm
question this script is testing.

Output schema of `iteration_winners` matches local_code_full.py exactly
(same keys used downstream by Stage 2: 'combo_union_vertices', 'dice',
'mni_y', 'combo_name', 'clusters', 'round'), so it is a drop-in swap for the
existing Stage 1 clustering cell.
"""
import os
import pickle
import time
from collections import deque
from os.path import join

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = join(ROOT, 'results', 'mri_surfcluster_baseline')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_CONTRAST = 'WordvsPER'
MIN_SIZE = 50          # same convention as the existing pipeline's min_size
THRESHOLDS = [1.65, 2.33, 3.1, 4.0]   # p<.05, p<.01, ~p<.001, stricter
SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']


# ---------------------------------------------------------------- Utilities
def _build_surface_topology(surface_faces):
    topology = {vertex: set() for vertex in np.unique(surface_faces)}
    for face in surface_faces:
        topology[face[0]].update([face[1], face[2]])
        topology[face[1]].update([face[0], face[2]])
        topology[face[2]].update([face[0], face[1]])
    return topology


def dice_coefficient(cluster1, cluster2):
    if len(cluster1) == 0 or len(cluster2) == 0:
        return 0.0
    intersection = len(set(cluster1) & set(cluster2))
    return 2.0 * intersection / (len(cluster1) + len(cluster2))


def mean_pairwise_dice(cluster_list):
    non_empty = [c for c in cluster_list if len(c) > 0]
    if len(non_empty) < 2:
        return 0.0
    d_sum, n_pair = 0.0, 0
    for i in range(len(non_empty)):
        for j in range(i + 1, len(non_empty)):
            d_sum += dice_coefficient(non_empty[i], non_empty[j])
            n_pair += 1
    return d_sum / n_pair


def mri_surfcluster(available_vertices, surface_topology, min_size):
    """The entirety of mri_surfcluster's clustering algorithm: connected
    components of an already-thresholded vertex set, dropped below min_size.
    No seed, no growth cap, no adaptive cutoff."""
    unvisited = set(available_vertices)
    clusters = []
    while unvisited:
        start = unvisited.pop()
        comp = [start]
        q = deque([start])
        while q:
            v = q.popleft()
            for nb in surface_topology.get(v, ()):
                if nb in unvisited:
                    unvisited.discard(nb)
                    comp.append(nb)
                    q.append(nb)
        if len(comp) >= min_size:
            clusters.append(np.array(comp, dtype=int))
    return clusters


def _union_find_match_across_sessions(session_clusters):
    """Group clusters from different sessions into one 'ROI' whenever they
    share at least one vertex. Pure spatial-overlap matching -- no growing,
    no priority ordering, no vertex exclusion."""
    flat = [(sess, set(c)) for sess, cl in session_clusters.items() for c in cl]
    n = len(flat)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if flat[i][0] == flat[j][0]:
                continue
            if flat[i][1] & flat[j][1]:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [[flat[i] for i in members] for members in groups.values()]


# ------------------------------------------------------------ Main routine
def run_surfcluster_stage1(df_tmaps, geometry, subjects, target_contrast,
                            tval_threshold, min_size):
    iteration_winners = {}
    for subj in subjects:
        surf_faces = geometry[subj]['faces']
        topology = _build_surface_topology(surf_faces)

        df_subj = df_tmaps[df_tmaps['subj'] == subj]
        sessions = sorted(df_subj['sess'].unique())
        searchspace = {row.vrtx_id: (row.x_pos, row.y_pos, row.z_pos)
                       for row in df_subj[['vrtx_id', 'x_pos', 'y_pos', 'z_pos']]
                       .drop_duplicates('vrtx_id').itertuples()}

        raw_maps = {}
        session_clusters = {}
        for sess in sessions:
            df_sess = df_subj[df_subj['sess'] == sess]
            raw_map = dict(zip(df_sess['vrtx_id'], df_sess[target_contrast]))
            raw_maps[sess] = raw_map
            supra = {v for v, t in raw_map.items() if t > tval_threshold}
            clusters = mri_surfcluster(supra, topology, min_size)
            clusters.sort(key=lambda c: max(raw_map[v] for v in c), reverse=True)
            session_clusters[sess] = clusters

        groups = _union_find_match_across_sessions(session_clusters)

        subject_winners = []
        for group in groups:
            clusters_by_sess = {}
            combo_clusters = []
            for sess, vset in group:
                arr = np.array(sorted(vset), dtype=int)
                clusters_by_sess.setdefault(sess, arr)
                combo_clusters.append(arr)
            combo_union = np.array(sorted(set().union(*[set(c) for c in combo_clusters])), dtype=int)
            dice = mean_pairwise_dice(combo_clusters)

            seed_sess = max(clusters_by_sess, key=lambda s: max(raw_maps[s][v] for v in clusters_by_sess[s]))
            seed_vertex = max(clusters_by_sess[seed_sess], key=lambda v: raw_maps[seed_sess][v])
            seed_val = raw_maps[seed_sess][seed_vertex]
            mni_y = searchspace[seed_vertex][1]

            subject_winners.append({
                'round': 0,  # assigned after sorting below
                'tmax': 1,
                'seed_session': seed_sess,
                'seed_vertex': int(seed_vertex),
                'seed_tval': float(seed_val),
                'seed_rawtval': f"{seed_val:.2f}",
                'mni_y': f"{mni_y:.2f}",
                'clusters': clusters_by_sess,
                'combo_union_vertices': combo_union,
                'dice': dice,
                'n_sessions_present': len(clusters_by_sess),
            })

        subject_winners.sort(key=lambda w: w['seed_tval'], reverse=True)
        for i, w in enumerate(subject_winners, start=1):
            w['round'] = i
            roi_name = w['seed_session'].replace('ses-', '')
            w['combo_name'] = f"Surfclust{i}_{roi_name}"

        iteration_winners[subj] = subject_winners
    return iteration_winners


def summarize(iteration_winners, threshold):
    rows = []
    for subj, winners in iteration_winners.items():
        for w in winners:
            rows.append({
                'threshold': threshold,
                'subj': subj,
                'roi': w['combo_name'],
                'mni_y': float(w['mni_y']),
                'size': len(w['combo_union_vertices']),
                'n_sessions_present': w['n_sessions_present'],
                'seed_tval': w['seed_tval'],
                'dice': w['dice'],
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    t0 = time.time()
    df_tmaps = pd.read_parquet(join(ROOT, 'dataframe', 'df_tmaps.parquet'))
    geometry = pd.read_pickle(join(ROOT, 'dataframe', 'geometry.pkl'))
    print(f"[DBG] data loaded t={time.time()-t0:.1f}s")

    all_summaries = []
    for thr in THRESHOLDS:
        print(f"\n===== mri_surfcluster baseline: threshold={thr}, min_size={MIN_SIZE} =====")
        winners = run_surfcluster_stage1(df_tmaps, geometry, SUBJECTS, TARGET_CONTRAST, thr, MIN_SIZE)

        with open(join(OUT_DIR, f'iteration_winners_thr{thr}.pkl'), 'wb') as f:
            pickle.dump(winners, f)

        summary = summarize(winners, thr)
        all_summaries.append(summary)
        for subj in SUBJECTS:
            sub_df = summary[summary['subj'] == subj].sort_values('mni_y', ascending=False)
            print(f"  {subj}: {len(sub_df)} ROI(s)")
            for _, r in sub_df.iterrows():
                print(f"      {r['roi']:<20s} Y={r['mni_y']:7.2f}  size={r['size']:5d}  "
                      f"n_sess={r['n_sessions_present']}  seed_t={r['seed_tval']:.2f}  DICE={r['dice']:.3f}")

    full_summary = pd.concat(all_summaries, ignore_index=True)
    full_summary.to_csv(join(OUT_DIR, 'surfcluster_summary_all_thresholds.csv'), index=False)
    print(f"\nSaved summary to {join(OUT_DIR, 'surfcluster_summary_all_thresholds.csv')}")
    print(f"[DBG] total runtime {time.time()-t0:.1f}s")
