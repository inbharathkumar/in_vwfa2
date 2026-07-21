# -*- coding: utf-8 -*-
"""
Gradation Clustering Pipeline - Run Until Stage 1 ROI Plots
This script runs the new Gradation-Constrained Region Growing (GCRG) algorithm
on the raw t-values of the WordvsPER contrast to isolate Stage 1 ROIs (fROIs)
for all 5 subjects, generating individual and combined PNG plots, skipping Stage 2 entirely.
"""

import os
import gc
import pickle
import warnings
import numpy as np
import pandas as pd
from collections import Counter, deque
from datetime import datetime
from os import makedirs
from os.path import join, exists, isdir
from PIL import Image, ImageChops

# Disable warnings
warnings.filterwarnings('ignore')

# Set plotting backend to Agg for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Nilearn imports
from nilearn.plotting import plot_surf_stat_map, plot_surf_contours

# =====================================================================
# PATHS AND PARAMETERS
# =====================================================================
ROOT = '/home/user/in_vwfa2'
df_path = join(ROOT, 'dataframe/df_tmaps.parquet')
geom_path = join(ROOT, 'dataframe/geometry.pkl')

target_contrast = 'WordvsPER'
tvals_type = 'raw_tval'  # Operating on raw t-values!

# Stage 1 Parameters
min_size = 50
max_size = 100
max_seeds = 2

# Gradation clustering parameters
relative_drop = 0.35  # Threshold for relative drop from seed peak
absolute_min = 1.0    # Absolute minimum t-value to consider as candidate

# Color palette for different rounds (up to 12 rounds)
ROI_COLS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324']

# Output Directory
out_dir = join(ROOT, 'results', 'stage1_gradation_only')
makedirs(out_dir, exist_ok=True)
makedirs(join(out_dir, 'labels'), exist_ok=True)

print(f"=== GRADATION CLUSTERING: RUN UNTIL STAGE 1 ROI PLOTS ===")
print(f"Output Directory: {out_dir}")
print(f"Parameters: relative_drop={relative_drop}, absolute_min={absolute_min}")

# =====================================================================
# CORE FUNCTIONS
# =====================================================================
def load_data_and_geometry():
    print("Loading df_tmaps.parquet...")
    df_tmaps = pd.read_parquet(df_path)
    print("Loading geometry.pkl...")
    geometry = pd.read_pickle(geom_path)
    return df_tmaps, geometry

def _build_surface_topology(surface_faces):
    topology = {vertex: set() for vertex in np.unique(surface_faces)}
    for face in surface_faces:
        topology[face[0]].update([face[1], face[2]])
        topology[face[1]].update([face[0], face[2]])
        topology[face[2]].update([face[0], face[1]])
    return topology

def _grow_gradation_bfs_cluster(seed_vertex, available_vertices_set, surface_topology, max_size, raw_t_map, relative_drop=0.35, absolute_min=1.0):
    if seed_vertex not in available_vertices_set:
        return np.array([], dtype=int)
    
    seed_val = raw_t_map.get(seed_vertex, 0.0)
    if seed_val <= absolute_min:
        return np.array([], dtype=int)
    
    # Adaptive relative cutoff
    cutoff_val = max(absolute_min, seed_val * relative_drop)
    
    found_cluster_vertices = []
    q = deque([seed_vertex])
    visited_vertices = {seed_vertex}
    
    while q and len(found_cluster_vertices) < max_size:
        current_vertex = q.popleft()
        found_cluster_vertices.append(current_vertex)
        
        for neighbor in surface_topology.get(current_vertex, []):
            if neighbor in available_vertices_set and neighbor not in visited_vertices:
                if raw_t_map.get(neighbor, 0.0) >= cutoff_val:
                    visited_vertices.add(neighbor)
                    q.append(neighbor)
                    
    return np.array(found_cluster_vertices, dtype=int)

def dice_coefficient(cluster1, cluster2):
    if len(cluster1) == 0 or len(cluster2) == 0:
        return 0.0
    intersection = len(np.intersect1d(cluster1, cluster2))
    return 2.0 * intersection / (len(cluster1) + len(cluster2))

def mean_pairwise_dice(cluster_list):
    non_empty = [c for c in cluster_list if len(c) > 0]
    if len(non_empty) < 2:
        return 0.0
    dices = []
    from itertools import combinations
    for c1, c2 in combinations(non_empty, 2):
        dices.append(dice_coefficient(c1, c2))
    return np.mean(dices) if dices else 0.0

def write_combo_union_label(outdir, subj, round_num, seed_session, tmax_num, union_vertices, searchspace):
    subj_dir = join(outdir, subj)
    makedirs(subj_dir, exist_ok=True)
    seed_sess_clean = seed_session.replace('ses-', '')
    out_path = join(subj_dir, f"_rnd{round_num}_tmax{tmax_num}_{seed_sess_clean}_unionROI.label")
    with open(out_path, 'w') as f:
        f.write(f"# combo union ROI (seed: {seed_session})\n{len(union_vertices)}\n")
        for v in union_vertices:
            coords = searchspace.get(v, (0.0, 0.0, 0.0))
            f.write(f"{v} {coords[0]:.4f} {coords[1]:.4f} {coords[2]:.4f} 1\n")
    return out_path

def create_montage(paths, out_path, cols=3):
    ims = [Image.open(p) for p in paths if exists(p)]
    if not ims:
        print(f"No images found to create montage: {out_path}")
        return
    w = max(i.width for i in ims)
    h = max(i.height for i in ims)
    rows = int(np.ceil(len(ims) / cols))
    montage_img = Image.new('RGB', (w * cols, h * rows), 'white')
    for k, im in enumerate(ims):
        montage_img.paste(im, ((k % cols) * w, (k // cols) * h))
    montage_img.save(out_path)
    print(f"Montage saved to: {out_path}")

# =====================================================================
# MAIN PIPELINE
# =====================================================================
def run_pipeline():
    df_tmaps, geometry = load_data_and_geometry()
    subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']
    
    # -----------------------------------------------------------------
    # STAGE 1: GRADATION CLUSTERING ON RAW T-VALUES
    # -----------------------------------------------------------------
    print("\n=== STAGE 1: GRADATION CLUSTERING ===")
    
    iteration_winners = {}
    
    for subj in subjects:
        print(f"\nProcessing Subject: {subj}...")
        surf_faces = geometry[subj]['faces']
        surface_topology = _build_surface_topology(surf_faces)
        
        df_subj = df_tmaps[df_tmaps['subj'] == subj]
        sessions = df_subj['sess'].unique()
        all_vrtx = df_subj['vrtx_id'].unique()
        
        # Build raw t-value map per session
        raw_t_maps = {}
        for sess in sessions:
            df_sess = df_subj[df_subj['sess'] == sess]
            raw_t_maps[sess] = dict(zip(df_sess['vrtx_id'], df_sess[target_contrast]))
            
        searchspace = {row.vrtx_id: (row.x_pos, row.y_pos, row.z_pos) for row in df_subj.itertuples()}
        
        iteration_claimed_vertices = set()
        subject_winners = []
        
        round_iterator = range(1, 10)  # Max 9 rounds
        for round_num in round_iterator:
            iteration_combos = []
            tmax_native_exclude = {sess: set() for sess in sessions}
            
            for tmax_num in range(1, max_seeds + 1):
                found_seed_any = False
                for seed_session in sessions:
                    raw_map = raw_t_maps[seed_session]
                    
                    available_vertices = set(all_vrtx) - iteration_claimed_vertices - tmax_native_exclude[seed_session]
                    candidates = [v for v in available_vertices if raw_map.get(v, 0.0) > absolute_min]
                    
                    if not candidates:
                        continue
                        
                    found_seed_any = True
                    seed_vertex = max(candidates, key=lambda v: raw_map[v])
                    seed_val = raw_map[seed_vertex]
                    
                    # Grow native cluster using Gradation Clustering
                    native_cluster = _grow_gradation_bfs_cluster(
                        seed_vertex, available_vertices, surface_topology, max_size, raw_map, relative_drop=relative_drop, absolute_min=absolute_min
                    )
                    
                    if len(native_cluster) < min_size:
                        native_cluster = np.array([], dtype=int)
                        
                    if len(native_cluster) > 0:
                        tmax_native_exclude[seed_session].update(native_cluster)
                    else:
                        tmax_native_exclude[seed_session].add(seed_vertex)
                        continue
                        
                    # Anchoring step
                    combo_clusters = [native_cluster]
                    clusters_by_sess = {seed_session: native_cluster}
                    
                    for target_session in sessions:
                        if target_session == seed_session:
                            continue
                        
                        tgt_map = raw_t_maps[target_session]
                        tgt_available = set(all_vrtx) - iteration_claimed_vertices
                        
                        intersection = set(native_cluster).intersection(tgt_available)
                        intersection = [v for v in intersection if tgt_map.get(v, 0.0) > 0.01]
                        
                        if not intersection:
                            anchor_cluster = np.array([], dtype=int)
                        else:
                            anchor_seed = max(intersection, key=lambda v: tgt_map[v])
                            anchor_cluster = _grow_gradation_bfs_cluster(
                                anchor_seed, tgt_available, surface_topology, max_size, tgt_map, relative_drop=relative_drop, absolute_min=absolute_min
                            )
                            if len(anchor_cluster) < min_size:
                                anchor_cluster = np.array([], dtype=int)
                                
                        combo_clusters.append(anchor_cluster)
                        clusters_by_sess[target_session] = anchor_cluster
                        
                    non_empty_count = sum(len(c) > 0 for c in combo_clusters)
                    if non_empty_count < 2:
                        continue
                        
                    # Compute mean pairwise DICE
                    dice = mean_pairwise_dice(combo_clusters)
                    combo_union = set()
                    for c in combo_clusters:
                        combo_union.update(c)
                    combo_union_vertices = np.array(sorted(combo_union), dtype=int)
                    
                    roi_name_sess = seed_session.replace('ses-', '')
                    combo_info = {
                        'round': round_num,
                        'tmax': tmax_num,
                        'seed_session': seed_session,
                        'seed_vertex': seed_vertex,
                        'seed_tval': seed_val,
                        'seed_rawtval': f"{raw_map[seed_vertex]:.2f}",
                        'mni_y': f"{df_subj[df_subj['vrtx_id'] == seed_vertex]['y_pos'].iloc[0]:.2f}",
                        'clusters': clusters_by_sess,
                        'combo_union_vertices': combo_union_vertices,
                        'dice': dice,
                        'combo_name': f"Rnd{round_num}Tmax{tmax_num}{roi_name_sess}"
                    }
                    iteration_combos.append(combo_info)
                    
            if not iteration_combos:
                print(f"    No more valid combos found. Stopping rounds at round {round_num - 1}.")
                break
                
            # Pick round winner by highest DICE
            best_combo = max(iteration_combos, key=lambda x: x['dice'])
            if best_combo['dice'] <= 0 or len(best_combo['combo_union_vertices']) == 0:
                print(f"    Best combo has 0 DICE. Stopping rounds.")
                break
                
            print(f"    Round {round_num} Winner: {best_combo['combo_name']} (DICE={best_combo['dice']:.3f}, MNI_Y={best_combo['mni_y']}, Size={len(best_combo['combo_union_vertices'])})")
            
            # Save the winning union label
            winner_union_path = write_combo_union_label(
                join(out_dir, 'labels'), subj, round_num, best_combo['seed_session'], best_combo['tmax'], best_combo['combo_union_vertices'], searchspace
            )
            
            # Claim the vertices
            winner_vertices = set()
            for c in best_combo['clusters'].values():
                winner_vertices.update(c)
            iteration_claimed_vertices.update(winner_vertices)
            
            subject_winners.append(best_combo)
            
            if len(set(all_vrtx) - iteration_claimed_vertices) < min_size:
                print("    No more available vertices.")
                break
                
        iteration_winners[subj] = subject_winners
        
    winners_pkl_path = join(out_dir, 'iteration_winners.pkl')
    with open(winners_pkl_path, 'wb') as f:
        pickle.dump(iteration_winners, f)
    print(f"Saved iteration winners to: {winners_pkl_path}")
    
    # -----------------------------------------------------------------
    # PLOTTING STAGE 1 ROIS USING HIGH-PERFORMANCE MATPLOTLIB ENGINE
    # -----------------------------------------------------------------
    print("\n=== GENERATING STAGE 1 ROI CONTOUR PLOTS ===")
    
    stage1_paths = []
    
    for s in subjects:
        print(f"  Rendering plot for {s}...")
        
        # Extract OTC ventral patch submesh (constant across all views)
        co = geometry[s]['inflated'].astype(np.float64)
        fa = geometry[s]['faces']
        cu = geometry[s]['curv']
        
        # Get active OTC vertices as selection mask, expand slightly to ensure margin
        sel = np.zeros(co.shape[0], bool)
        sel[df_tmaps[df_tmaps.subj == s].vrtx_id.unique()] = True
        for _ in range(4):
            sel[np.unique(fa[sel[fa].any(1)])] = True
        ni = np.where(sel)[0]
        im = -np.ones(co.shape[0], int)
        im[ni] = np.arange(len(ni))
        sc, sf, scv = co[ni], im[fa[sel[fa].all(1)]], cu[ni]
        
        # Load raw t-values
        raw = df_tmaps[df_tmaps.subj == s].groupby('vrtx_id')[target_contrast].mean()
        sm_raw = np.zeros(co.shape[0], dtype=np.float32)
        sm_raw[raw.index.values] = raw.values
        
        # Plot Stage 1 ROIs
        rmap_s1 = np.zeros_like(sm_raw, int)
        s1_winners = iteration_winners[s]
        roilabels = []
        for idx, winner in enumerate(s1_winners, start=1):
            rmap_s1[winner['combo_union_vertices']] = idx
            mean_tval = raw.loc[raw.index.isin(winner['combo_union_vertices'])].mean()
            roilabels.append(f"{winner['combo_name']} (y:{winner['mni_y']}|n={len(winner['combo_union_vertices'])}|T={mean_tval:.1f}|D={winner['dice']:.2f})")
            
        fig = plot_surf_stat_map((sc, sf), sm_raw[ni], hemi='left', view='ventral', bg_map=scv, cmap='YlOrRd', threshold=0.1,
                               vmax=float(raw.max()), bg_on_data=True, title=f'{s.capitalize()} - Stage 1 Gradation ROIs', engine='matplotlib', colorbar=True)
        plot_surf_contours((sc, sf), rmap_s1[ni], levels=list(range(1, len(s1_winners) + 1)), colors=ROI_COLS[:len(s1_winners)], figure=fig, legend=False)
        
        # Add legend showing all ROI details
        from matplotlib.lines import Line2D
        fig.axes[0].legend(
            handles=[Line2D([0], [0], color=ROI_COLS[i], lw=3, label=roilabels[i]) for i in range(len(roilabels))],
            loc='lower left', fontsize=6, framealpha=0.85
        )
        
        p1 = join(out_dir, f'{s}_stage1_rois.png')
        fig.savefig(p1, dpi=130, bbox_inches='tight')
        plt.close(fig)
        stage1_paths.append(p1)
        
    print("\nCreating combined 5 subjects montage...")
    combined_montage_path = join(out_dir, 'Stage1_ROI_contour_plots_5subj.png')
    create_montage(stage1_paths, combined_montage_path, cols=3)
    
    # Save a copy in the root results directory so the user can easily see/download it
    root_results_dir = join(ROOT, 'results')
    makedirs(root_results_dir, exist_ok=True)
    root_montage_path = join(root_results_dir, 'Stage1_ROI_contour_plots_5subj.png')
    create_montage(stage1_paths, root_montage_path, cols=3)
    
    print("\n=======================================================")
    print(f"STAGE 1 RUN COMPLETE!")
    print(f"All individual plots saved in: {out_dir}")
    print(f"Combined 5-subject plot saved in: {root_montage_path}")
    print("=======================================================")

if __name__ == '__main__':
    run_pipeline()
