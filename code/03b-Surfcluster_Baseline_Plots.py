"""
Companion plot generator for 03-Surfcluster_Baseline_Code.py.

Produces MNI-Y vs raw-t scatter plots (mean tmap across sessions), colored by
mri_surfcluster cluster membership at each threshold, for visual comparison
against the existing scaled-BFS / gradation Stage 1 ROI figures. Same plot
style as the existing pipeline's "1.1.2a Raw vs Scaled MNI-Y" cell.
"""
import os
from os.path import join

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = join(ROOT, 'results', 'mri_surfcluster_baseline')
TARGET_CONTRAST = 'WordvsPER'
THRESHOLDS = [1.65, 2.33, 3.1, 4.0]
SUBJECTS = ['sub-02', 'sub-04', 'sub-05', 'sub-01']

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_mod_name = '03-Surfcluster_Baseline_Code'
import importlib.util
spec = importlib.util.spec_from_file_location(
    'surfcluster_baseline', join(os.path.dirname(os.path.abspath(__file__)), '03-Surfcluster_Baseline_Code.py'))
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)


def cluster_map_for_subject(subj, df_tmaps, geometry, threshold, min_size=50):
    """Cluster the SESSION-AVERAGED raw tmap (one map per subject) for a clean
    single-panel picture; per-session results (see console output) show the
    same pattern."""
    surf_faces = geometry[subj]['faces']
    topology = sb._build_surface_topology(surf_faces)
    df_subj = df_tmaps[df_tmaps['subj'] == subj]
    mean_t = df_subj.groupby('vrtx_id')[TARGET_CONTRAST].mean()
    ypos = df_subj.groupby('vrtx_id')['y_pos'].first()
    raw_map = mean_t.to_dict()
    supra = {v for v, t in raw_map.items() if t > threshold}
    clusters = sb.mri_surfcluster(supra, topology, min_size)
    clusters.sort(key=len, reverse=True)
    vrtx_to_cluster = {}
    for ci, c in enumerate(clusters, start=1):
        for v in c:
            vrtx_to_cluster[v] = ci
    return mean_t, ypos, vrtx_to_cluster, len(clusters)


if __name__ == '__main__':
    df_tmaps = pd.read_parquet(join(ROOT, 'dataframe', 'df_tmaps.parquet'))
    geometry = pd.read_pickle(join(ROOT, 'dataframe', 'geometry.pkl'))

    for subj in SUBJECTS:
        fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=(5 * len(THRESHOLDS), 4.5), sharey=True)
        for ax, thr in zip(axes, THRESHOLDS):
            mean_t, ypos, vrtx_to_cluster, n_clusters = cluster_map_for_subject(subj, df_tmaps, geometry, thr)
            vrtx_ids = mean_t.index.values
            y = ypos.loc[vrtx_ids].values
            t = mean_t.values
            cluster_id = np.array([vrtx_to_cluster.get(v, 0) for v in vrtx_ids])

            ax.scatter(y[cluster_id == 0], t[cluster_id == 0], s=4, c='lightgray', alpha=0.4, label='below thresh / <min_size')
            cmap = plt.get_cmap('tab10')
            for ci in sorted(set(cluster_id) - {0}):
                mask = cluster_id == ci
                ax.scatter(y[mask], t[mask], s=8, color=cmap((ci - 1) % 10), label=f'cluster {ci} (n={mask.sum()})')
            ax.axhline(thr, color='red', linestyle='--', linewidth=0.8)
            ax.invert_xaxis()
            ax.set_title(f'thr={thr}  |  {n_clusters} cluster(s)')
            ax.set_xlabel('y_pos (anterior <- -> posterior)')
            ax.legend(fontsize=7, loc='upper right', markerscale=2)
        axes[0].set_ylabel('Mean raw t-value (WordvsPER)')
        fig.suptitle(f'{subj} - mri_surfcluster baseline (threshold + connected components only, min_size=50)', fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = join(OUT_DIR, f'{subj}_surfcluster_thresholds.png')
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f'Saved: {out_path}')
