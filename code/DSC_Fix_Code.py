"""
THIS FILE CONTAINS CODE ONLY TILL STAGE 1 WITH PERFECT CODE TO GET PLOTS AS PNGs (both for single subject and all subjects)
USE THE SAME PLOTTING CODE FOR STAGE 2 ROIs TO GENERATE INDIVIDUAL AND STITCHED SUBJECTS PLOTS. 
SEE #@title 1.2.2 Save Stage 1 ROIs (All) for the actual plotting code. 
Use this code - !pip install "plotly<6" "kaleido==0.2.1" -q 
Use this function - save_plotly_fig 
DO NOT USE ANY FUNTION ON YOUR OWN TO GENERATE PLOTS!
"""
# Setup

## ===== Imports & Functions

# @title Connect
from google.colab import drive
from google.colab.output import clear
from os import chdir, listdir, makedirs
drive.mount('/content/drive/', force_remount=False)
root = '/content/drive/My Drive/Colab/VSS/derivatives'
save_root = '/content/drive/My Drive/Colab_Output/VSS/derivatives'
chdir(root)
clear()

#@title Install packages
!pip install nilearn -q
!pip install kneed -q
!pip install tvb-gdist -q
!pip install "plotly<6" "kaleido==0.2.1" -q
#os.kill(os.getpid(), 9)

#@title Imports
from glob import glob
from google.colab import data_table
from os.path import join, isdir, getmtime
import gc
import hashlib
import json
import nibabel as nib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel.freesurfer.io as fsio
import pickle
import seaborn as sns
import warnings
from IPython.display import clear_output
from IPython.display import display, Javascript, clear_output

import gc

from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
from matplotlib.colors import ListedColormap
from collections import Counter, deque
from datetime import datetime, timedelta
from glob import glob
from gdist import compute_gdist
from kneed import KneeLocator
from math import ceil
from nilearn.plotting import view_surf
from nilearn.surface import load_surf_data, load_surf_mesh
from IPython.display import HTML, display
from itertools import combinations, count, product
from nibabel.freesurfer import read_annot, read_geometry, read_label, read_morph_data
from nibabel.gifti import GiftiDataArray
from nibabel import load, GiftiImage

from pandas import read_csv, read_json, read_parquet, to_pickle
from pathlib import Path
from PIL import Image, ImageChops

from scipy.spatial.distance import cdist
from shutil import copy2
from sklearn.preprocessing import minmax_scale
from nilearn.plotting import plot_surf_stat_map

from numpy import (
    arange, argmax, argpartition, argsort, array, asarray, concatenate, mean, std, nan, nan_to_num, clip,
    diff, eye, float16, fromiter, intersect1d, isfinite, isin, isnan, percentile, linalg, linspace, ones, squeeze, transpose, vstack, where
)
from os.path import join, basename, dirname, exists, getmtime, isdir
from pandas import read_parquet, read_pickle, read_csv

import ipywidgets as W
from ipywidgets import interact, Dropdown
from ipywidgets import widgets

from google.colab import output
output.enable_custom_widget_manager()

bold = '\033[1m'
boldred = '\033[1;31m'
boldblue = '\033[1;34m'
boldgreen = '\033[1;92m'#'\033[1;32m'
boldmag = '\033[1;35m'
#resetbold = '\033[0m'
unbold = '\033[0m'
clear()



#@title Function (I/O)

def read_FSLabel(roi_path):
  with open(roi_path, 'r') as f:
    valid_rows = [l.strip().split() for l in f if l.strip()][2:]
  vertices_ids = np.array([int(row[0]) for row in valid_rows])
  xyz_coords   = np.array([[float(coord) for coord in row[1:4]] for row in valid_rows])
  return vertices_ids, xyz_coords

def load_geometry(subj):
    ''' Usage:
    subj = 'sub-03'
    coords, coords_rot, faces, bgcurv = load_geometry(subj)
    '''
    geometry = read_pickle('dataframe/geometry.pkl')
    geom = geometry[subj]
    coords = geom['inflated']
    coords_rot = _rotate_coords(coords)
    faces = geom['faces']  # <-- make sure faces matches this subject!
    bgcurv = np.where(geom['curv'] > 0, 0.7, 0.4)
    return coords, coords_rot, faces, bgcurv

def load_geometry_pial(subj):
    geometry = read_pickle('dataframe/geometry.pkl')
    geom = geometry[subj]
    coords = geom['pial']
    coords_rot = _rotate_coords(coords)
    faces = geom['faces']  # <-- make sure faces matches this subject!
    bgcurv = np.where(geom['curv'] > 0, 0.7, 0.4)
    return coords, coords_rot, faces, bgcurv


def write_combo_union_label(outdir, subj, round, seed_session, tmax_num, union_vertices, searchspace):
    """Save union label for a combo: rnd{round}_tmax{tmax_num}_{sess}_unionROI.label"""
    subj_outdir = join(outdir, subj)
    makedirs(subj_outdir, exist_ok=True)

    # Remove 'ses-' prefix from session name
    seed_sess_clean = seed_session.replace('ses-', '')

    filename = f"rnd{round}_tmax{tmax_num}_{seed_sess_clean}_unionROI.label"
    out_lbl = join(subj_outdir, filename)

    union_vertices = np.atleast_1d(np.array(union_vertices, dtype=int))
    with open(out_lbl, 'w') as f:
        f.write(f"# Freesurfer label: {subj} rnd{round} tmax{tmax_num} {seed_sess_clean} unionROI\n")
        f.write(f"{len(union_vertices)}\n")
        for v in union_vertices:
            x, y, z = searchspace[v]
            f.write(f"{v} {x:.6f} {y:.6f} {z:.6f} 1\n")
    return out_lbl

def write_individual_roi_label(outdir, subj, round, tmax_num, root_session, roi_session, cluster, searchspace):
    """Save individual ROI label: rnd{round}_tmax{tmax_num}_{root}root_{roi}ROI.label"""
    subj_outdir = join(outdir, subj)
    makedirs(subj_outdir, exist_ok=True)

    # Remove 'ses-' prefix from session names
    root_clean = root_session.replace('ses-', '')
    roi_clean = roi_session.replace('ses-', '')

    filename = f'rnd{round}_tmax{tmax_num}_{root_clean}root_{roi_clean}ROI.label'
    out_lbl = join(subj_outdir, filename)

    with open(out_lbl, 'w') as f:
        f.write(f'# Freesurfer label: {subj} rnd{round} tmax{tmax_num} {root_clean}root {roi_clean}ROI\n')
        f.write(f'{len(cluster)}\n')
        for v in cluster:
            x, y, z = searchspace[v]
            f.write(f'{v} {x:.6f} {y:.6f} {z:.6f} 1\n')
    return out_lbl

def save_plotly_fig(fig, out_path, width=1100, height=780, scale=2, zoom_factor=1.18):
    # Scale 3D camera eye vector to step camera back and prevent wide surface meshes from clipping at WebGL viewport boundaries
    camera_eye = fig.layout.scene.camera.eye
    fig.update_scenes(
        camera_eye=dict(
            x=camera_eye.x * zoom_factor,
            y=camera_eye.y * zoom_factor,
            z=camera_eye.z * zoom_factor
        )
    )
    fig.write_html(f'{out_path}.html', include_plotlyjs='cdn')
    fig.update_layout(margin=dict(l=0, r=0, t=34, b=0), paper_bgcolor='white')
    fig.write_image(f'{out_path}.png', width=width, height=height, scale=scale)
    img = Image.open(f'{out_path}.png').convert('RGB')
    left_pos, top_pos, right_pos, bottom_pos = ImageChops.difference(img, Image.new('RGB', img.size, (255, 255, 255))).getbbox()
    img.crop((max(0, left_pos - 10), max(0, top_pos - 10), min(img.width, right_pos + 10), min(img.height, bottom_pos + 10))).save(f'{out_path}.png')
    print(f'{boldgreen}Saved: {out_path}.html + .png{unbold}')

#@title Function (Dataframe)

def create_geometry_cache(pkl_path):
  geometry = {}
  subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']
  for subj in subjects:
      print(f"Reading {subj} geometry")
      infl_coords, faces = read_geometry(f'{root}/freesurfer/{subj}/surf/lh.inflated')
      pial_coords, _ = read_geometry(f'{root}/freesurfer/{subj}/surf/lh.pial')
      curv = read_morph_data(f'{root}/freesurfer/{subj}/surf/lh.curv')
      annot_path = f'{root}/freesurfer/{subj}/label/lh.aparc.a2009s.annot'
      annot_labels, annot_ctab, annot_names = read_annot(annot_path)
      geometry[subj] = {
          'inflated': infl_coords.astype('<f4'),
          'pial': pial_coords.astype('<f4'),
          'faces': faces,
          'curv': curv.astype('<f4'),
          'annot_labels': annot_labels,
          'annot_ctab': annot_ctab,
          'annot_names': annot_names
      }

  to_pickle(geometry, pkl_path)
  print(f"{boldgreen}Geometry cache saved to: {pkl_path}{unbold}")



def create_bigLOTS_df():

    analysis_dir = join(root, 'l1_surface', 'analysis-okazaki_nosmooth')

    rows = []

    for subj in sorted(listdir(analysis_dir)):
      print(f"Reading {subj} files")
      subj_dir = join(analysis_dir, subj)

      # Read Anatomical maps
      ct_full    = load_surf_data(f'{root}/freesurfer/{subj}/surf/lh.thickness').astype('<f4')
      depth_full = load_surf_data(f'{root}/freesurfer/{subj}/surf/lh.sulc').astype('<f4')
      curv_full  = load_surf_data(f'{root}/freesurfer/{subj}/surf/lh.curv').astype('<f4')
      annot_path = glob(f'{root}/freesurfer/{subj}/label/*aparc*.annot')[0]
      annot_labels, annot_ctab, annot_names = nib.freesurfer.read_annot(annot_path)
      annot_names_decoded = [name.decode('utf-8') for name in annot_names]

      # Parse bigLOTS label
      roi_path = f'{root}/freesurfer/{subj}/label/lh.bigLOTS.label'
      vrtx_ids, coords = read_FSLabel(roi_path)

      for sess in sorted(listdir(subj_dir)):
          print(f"      Reading {subj}-{sess} files")
          sess_dir = join(subj_dir, sess)

          # Find all left‐hemisphere statmap files
          pattern = join(sess_dir,
              f'{subj}_{sess}_task-fLoc_hemi-L_space-fsnative_contrast-*_*statmap.func.gii'
          )
          for fpath in glob(pattern):
              fname = basename(fpath).split('_')
              contrast = next(p.split('-',1)[1] for p in fname if p.startswith('contrast-'))
              stat     = next(p.split('-',1)[1] for p in fname if p.startswith('stat-') and 'statmap' not in p)
              if stat not in ('effect','t','variance'):
                  continue
              # Load and mask
              arr = load_surf_data(fpath).astype('<f4')[vrtx_ids]

              # Accumulate per‐vertex rows
              for i, v in enumerate(vrtx_ids):
                  rows.append({
                      'subj':        subj,
                      'sess':        sess,
                      'contrast':    contrast,
                      'vrtx_id':     int(v),
                      'x_pos':       float(coords[i,0]),
                      'y_pos':       float(coords[i,1]),
                      'z_pos':       float(coords[i,2]),
                      f'stat_{stat}': float(arr[i]),
                      'anat_ct':     float(ct_full[v]),
                      'anat_depth':  float(depth_full[v]),
                      'anat_curv':   float(curv_full[v]),
                      'annot_labels':  int(annot_labels[v]),
                      'annot_name':   annot_names_decoded[annot_labels[v]],
                  })

    # Convert to DataFrame and pivot so each stat is its own column
    df = pd.DataFrame(rows)

    # Define the non‐stat columns that uniquely identify each vertex
    index_cols = [
        'subj','sess','contrast',
        'vrtx_id','x_pos','y_pos','z_pos',
        'anat_ct','anat_depth','anat_curv',
        'annot_name'
    ]

    # Find all the stat_ columns that were created
    stat_cols = ['stat_effect','stat_t','stat_variance']

    # Aggregate so each (subj,sess,contrast,vertex) has its stat_effect, stat_p, stat_t, stat_variance, stat_z
    df_all = (df.groupby(index_cols, as_index=False)[stat_cols].first())
    #df_all = df_all[index_cols + stat_cols]
    df_all.to_parquet(f'{root}/dataframe/df_all.parquet')
    print(f"{boldgreen}  Success! Dataframe saved to: {root}/dataframe/df_all.parquet{unbold}")



    df_pivoted = df_all.pivot_table(
        index=['subj', 'sess', 'vrtx_id', 'x_pos', 'y_pos', 'z_pos',
              'anat_ct', 'anat_depth', 'anat_curv', 'annot_name'],
        columns='contrast',
        values=['stat_effect', 'stat_t', 'stat_variance'],
        aggfunc='first'
    )

    # Flatten column names to create contrast_stat format
    df_pivoted.columns = [f'{contrast}_{stat.split("_")[1]}'
                          for stat, contrast in df_pivoted.columns]

    # Reset index to make all columns regular columns
    df_pivoted = df_pivoted.reset_index()

    # Save pivoted DataFrame
    df_pivoted.to_parquet('dataframe/df_pivoted_all.parquet')
    print(f"{boldgreen}  Success! Dataframe saved to: {root}/dataframe/df_pivoted_all.parquet{unbold}")

#@title Function (DSC)
def dice_coefficient(cluster1, cluster2):
    """Calculate Dice coefficient between two clusters"""
    if len(cluster1) == 0 or len(cluster2) == 0:
        return 0.0
    intersection = len(set(cluster1) & set(cluster2))
    return 2.0 * intersection / (len(cluster1) + len(cluster2))

def mean_pairwise_dice(cluster_list):
    """Calculate mean pairwise DICE across only the non-empty clusters.
    (FIXED 2026-07-22: previous version included empty-vs-nonempty pairs in
    the average. dice_coefficient forces those pairs to 0.0, which deflated
    the score whenever any session failed to anchor a matching cluster,
    biasing round-winner selection. This restores the original filtering
    behavior from the reference gradation_clustering_code.py.)"""
    non_empty = [c for c in cluster_list if len(c) > 0]
    if len(non_empty) < 2:
        return 0.0
    d_sum, n_pair = 0.0, 0
    for i in range(len(non_empty)):
        for j in range(i+1, len(non_empty)):
            d_sum += dice_coefficient(non_empty[i], non_empty[j])
            n_pair += 1
    return d_sum / n_pair


#@title Function (Clustering)
def _build_surface_topology(surface_faces):
    """Builds an adjacency list from surface faces for efficient neighbor lookup."""
    topology = {vertex: set() for vertex in np.unique(surface_faces)}
    for face in surface_faces:
        topology[face[0]].update([face[1], face[2]])
        topology[face[1]].update([face[0], face[2]])
        topology[face[2]].update([face[0], face[1]])
    return topology

def _grow_gradation_bfs_cluster(seed_vertex, available_vertices_set, surface_topology,
                                 max_size, raw_t_map, relative_drop=0.35, absolute_min=1.0):
    """
    Grows a single contiguous cluster from a seed vertex using Breadth-First Search (BFS),
    with an adaptive "gradation" cutoff: a neighbor is only added if its raw t-value is
    still within `relative_drop` of the seed's peak activation (and above `absolute_min`).
    This lets the growth criterion adapt to the local peak strength instead of using one
    global hard threshold (see Gradation-Constrained Region Growing, GCRG).
    """
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


def run_gradation_clustering(df_tmaps, geometry, subjects, out_dir, target_contrast='WordvsPER',
                              min_size=50, max_size=100, max_seeds=2, relative_drop=0.35,
                              absolute_min=1.0, max_rounds=9, save_all_roi=True):
    """
    Runs Gradation-Constrained Region Growing (GCRG) Stage 1 clustering on RAW t-values
    for every subject, and returns (and pickles) the iteration_winners dict consumed by
    the plotting cell ("1.2.2 Save Stage 1 ROIs (All)").

    This is a self-contained replacement for the old scaled-tval BFS clustering cell
    (1.2.1). No local/scaled thresholding (1.1) is required - the relative_drop /
    absolute_min gradation criterion adapts to each seed's local peak directly on
    raw t-values, so weak anterior peaks and strong posterior peaks are both handled
    without any pre-scaling step.

    Parameters
    ----------
    df_tmaps : DataFrame with columns [subj, sess, vrtx_id, x_pos, y_pos, z_pos, target_contrast, ...]
    geometry : dict[subj] -> {'faces': ..., 'inflated': ..., 'curv': ..., ...}
    subjects : list of subject IDs, e.g. ['sub-01', ..., 'sub-05']
    out_dir  : directory where per-round winner union .label files get saved
               (only saved if save_all_roi=True - matches "candidate ROIs shown in the
               Stage 1 plot" scope: round winners only, no per-session intermediate ROIs)
    target_contrast : column name in df_tmaps to cluster on (raw t-values)
    min_size, max_size, max_seeds : as in original GCRG params
    relative_drop, absolute_min   : GCRG adaptive threshold params
    max_rounds : safety cap on number of rounds per subject
    save_all_roi : if True, saves each round-winner's union label to
                   out_dir/labels/<subj>/_rnd{N}_winner==tmax{T}_{sess}_unionROI.label

    Returns
    -------
    iteration_winners : dict[subj] -> list of combo_info dicts (round winners, in order)
    """
    iteration_winners = {}
    labels_dir = join(out_dir, 'labels')
    if save_all_roi:
        makedirs(labels_dir, exist_ok=True)

    for subj in subjects:
        print(f"\n{boldmag}===== {subj} ====={unbold}")
        surf_faces = geometry[subj]['faces']
        surface_topology = _build_surface_topology(surf_faces)

        df_subj = df_tmaps[df_tmaps['subj'] == subj]
        sessions = df_subj['sess'].unique()
        all_vrtx = df_subj['vrtx_id'].unique()

        # Build raw t-value map per session (vrtx_id -> t-value)
        raw_t_maps = {}
        for sess in sessions:
            df_sess = df_subj[df_subj['sess'] == sess]
            raw_t_maps[sess] = dict(zip(df_sess['vrtx_id'], df_sess[target_contrast]))

        searchspace = {row.vrtx_id: (row.x_pos, row.y_pos, row.z_pos) for row in df_subj.itertuples()}

        iteration_claimed_vertices = set()
        subject_winners = []

        for round_num in range(1, max_rounds + 1):
            print(f"\n{boldblue}  --- Round {round_num} ---{unbold}")
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
                        seed_vertex, available_vertices, surface_topology, max_size,
                        raw_map, relative_drop=relative_drop, absolute_min=absolute_min
                    )

                    if len(native_cluster) < min_size:
                        native_cluster = np.array([], dtype=int)

                    if len(native_cluster) > 0:
                        tmax_native_exclude[seed_session].update(native_cluster)
                    else:
                        tmax_native_exclude[seed_session].add(seed_vertex)
                        continue

                    roi_name_sess = seed_session.replace('ses-', '')
                    native_y = df_subj.loc[df_subj['vrtx_id'] == seed_vertex, 'y_pos'].iloc[0]
                    print(f"\t\t      {seed_session} (native): {len(native_cluster)} vertices",
                          f" (seed vertex: {seed_vertex} | mni_y: {native_y:.2f}",
                          f"| raw_tval: {seed_val:.3f})")

                    # Anchoring step: find/grow the matching cluster in every other session
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
                            print(f"\t\t      {target_session} (anchored): empty (seed vertex: Not Found)")
                        else:
                            anchor_seed = max(intersection, key=lambda v: tgt_map[v])
                            anchor_cluster = _grow_gradation_bfs_cluster(
                                anchor_seed, tgt_available, surface_topology, max_size,
                                tgt_map, relative_drop=relative_drop, absolute_min=absolute_min
                            )
                            if len(anchor_cluster) < min_size:
                                anchor_cluster = np.array([], dtype=int)

                            anchor_y = df_subj.loc[df_subj['vrtx_id'] == anchor_seed, 'y_pos'].iloc[0]
                            print(f"\t\t      {target_session} (anchored): {len(anchor_cluster)} vertices",
                                  f" (seed vertex: {anchor_seed} | mni_y: {anchor_y:.2f}",
                                  f"| raw_tval: {tgt_map[anchor_seed]:.3f})")

                        combo_clusters.append(anchor_cluster)
                        clusters_by_sess[target_session] = anchor_cluster

                    non_empty_count = sum(len(c) > 0 for c in combo_clusters)
                    if non_empty_count < 2:
                        print(f"\t\t  Skipping combo (insufficient non-empty ROIs: {non_empty_count}).")
                        continue

                    # Compute mean pairwise DICE across all session clusters in this combo
                    dice = mean_pairwise_dice(combo_clusters)
                    combo_union = set()
                    for c in combo_clusters:
                        combo_union.update(c)
                    combo_union_vertices = np.array(sorted(combo_union), dtype=int)

                    combo_info = {
                        'round': round_num,
                        'tmax': tmax_num,
                        'seed_session': seed_session,
                        'seed_vertex': seed_vertex,
                        'seed_tval': seed_val,
                        'seed_rawtval': f"{seed_val:.2f}",
                        'mni_y': f"{native_y:.2f}",
                        'clusters': clusters_by_sess,
                        'combo_union_vertices': combo_union_vertices,
                        'dice': dice,
                        'combo_name': f"Rnd{round_num}Tmax{tmax_num}{roi_name_sess}"
                    }
                    iteration_combos.append(combo_info)
                    print(f"\t\t  DICE of iter{round_num}_{roi_name_sess}_Tmax{tmax_num} Combo: {dice:.3f}")

                if not found_seed_any:
                    print("\t    No candidate seeds remain in any session; stopping TMAX loop.")
                    break

            # ========== Find round winner ==========
            if not iteration_combos:
                print(f"\n{boldred}  No valid combos found in round {round_num}. Stopping.{unbold}")
                break

            best_combo = max(iteration_combos, key=lambda x: x['dice'])
            if best_combo['dice'] <= 0 or len(best_combo['combo_union_vertices']) == 0:
                print(f"\n{boldred}  No reliable ROI found (winner empty or zero-DICE). Stopping further rounds.{unbold}")
                break

            print(f"\n{boldgreen}  Round {round_num} Winner: {best_combo['combo_name']}",
                  f"(SEED_TVAL={best_combo['seed_rawtval']} |",
                  f"MNI_Y={best_combo['mni_y']} |",
                  f"SIZE={len(best_combo['combo_union_vertices'])} |",
                  f"DICE={best_combo['dice']:.3f})")

            # Save ONLY the round-winner's union label (the ROI that actually shows up
            # in the Stage 1 plot) - no per-session intermediate ROI labels are written.
            if save_all_roi:
                seed_sess_clean = best_combo['seed_session'].replace('ses-', '')
                winner_union_path = join(
                    labels_dir, subj,
                    f"_rnd{round_num}_winner==tmax{best_combo['tmax']}_{seed_sess_clean}_unionROI.label"
                )
                tmp = write_combo_union_label(
                    labels_dir, subj, round_num,
                    best_combo['seed_session'], best_combo['tmax'],
                    best_combo['combo_union_vertices'], searchspace
                )
                os.replace(tmp, winner_union_path)
                best_combo['combo_union_label'] = winner_union_path

            # Claim the vertices used by the winner so future rounds can't reuse them
            winner_vertices = set()
            for c in best_combo['clusters'].values():
                winner_vertices.update(c)
            iteration_claimed_vertices.update(winner_vertices)

            subject_winners.append(best_combo)

            if len(set(all_vrtx) - iteration_claimed_vertices) < min_size:
                print(f"\n{boldred}  No more vertices available for clustering. Stopping at round {round_num}.{unbold}")
                break

        iteration_winners[subj] = subject_winners
        print(f"\n{boldmag}  Final winners for {subj}:{unbold}")
        for winner in subject_winners:
            print(f"    Round {winner['round']}: {winner['combo_name']}",
                  f"(SEED_TVAL={winner['seed_rawtval']} |",
                  f"MNI_Y={winner['mni_y']} |",
                  f"SIZE={len(winner['combo_union_vertices'])} |",
                  f"DICE={winner['dice']:.3f})")

    winners_path = join(out_dir, 'iteration_winners.pkl')
    with open(winners_path, 'wb') as f:
        pickle.dump(iteration_winners, f)
    print(f"\n{boldgreen}Clustering complete. Saved iteration winners to: {winners_path}{unbold}")

    return iteration_winners

#@title Function (FreeView Config)
# ------ FreeView angles ------
FV_AZIM = 270
FV_ELEV = -25 # 0
FV_ROLL = 60 #60
CONVENTION = 'inv_zx'
AZIM_OFFSET = 110

# ------------- Rotation helpers -------------
def _Rx(deg):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],[0, c,-s],[0, s, c]], float)

def _Rz(deg):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[ c,-s, 0],[ s, c, 0],[ 0, 0, 1]], float)

def _R_axis_angle(axis, deg):
    theta = np.deg2rad(deg)
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n == 0: return np.eye(3)
    x, y, z = axis / n
    K = np.array([[0,-z, y],[z, 0,-x],[-y, x, 0]], float)
    I = np.eye(3)
    return I + np.sin(theta)*K + (1-np.cos(theta))*(K@K)

def _rotate_freeview_angles(coords, fv_azim, fv_elev, fv_roll=0,
                            convention='inv_zx', azim_offset=90):
    """
    Convert FreeView (azim, elev, roll) to an object rotation for view_surf.

    convention:
      - 'inv_zx': rotate object by Rz(-(azim+offset)) then Rx(-elev)
      - 'inv_xz': rotate object by Rx(-elev) then Rz(-(azim+offset))
    """
    ctr = coords.mean(axis=0, keepdims=True)
    X = coords - ctr

    az = -(fv_azim + azim_offset)
    el = -fv_elev

    if convention == 'inv_xz':
        R0 = _Rx(el) @ _Rz(az)
    else:  # 'inv_zx'
        R0 = _Rz(az) @ _Rx(el)

    # roll around camera-forward axis (use +Y after yaw/pitch)
    if fv_roll != 0:
        forward_world = R0 @ np.array([0.0, 1.0, 0.0])
        Rroll = _R_axis_angle(forward_world, fv_roll)
        R = Rroll @ R0
    else:
        R = R0

    return (X @ R.T) + ctr

def _rotate_coords(coords):
    return _rotate_freeview_angles(coords, FV_AZIM, FV_ELEV, FV_ROLL, CONVENTION, AZIM_OFFSET)

# I. Stage 1

## ===== 1.1 Stage 1 ROI

#@title 1.1 Run Gradation Clustering

# GLM-I https://nilearn.github.io/dev/modules/generated/nilearn.glm.compute_contrast.html
# Test directionality: one-tailed
df_tmaps = read_parquet('dataframe/df_tmaps.parquet')
geometry = read_pickle('dataframe/geometry.pkl')
subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']

target_contrast = 'WordvsPER'

# Gradation-Constrained Region Growing (GCRG) parameters
# Our target is to get 50-100 vertex ROIs
min_size = 50
max_size = 100
max_seeds = 2
max_rounds = 9          # safety cap on rounds per subject
relative_drop = 0.35    # threshold for relative drop from seed peak (gamma)
absolute_min = 1.0      # absolute minimum t-value to consider as a growth candidate

# Only round-winner union ROIs get saved (the ones that actually show up in the
# Stage 1 plot below) - no per-session intermediate ROI labels.
save_all_roi = True

# Set Output File Paths
timestamp = (datetime.now() + timedelta(hours=2)).strftime("%b%d_%Hh%Mm")
out_dir = join(save_root, 'results', f'{target_contrast}_gradation_{relative_drop}drop_{absolute_min}min_{timestamp}')
makedirs(out_dir, exist_ok=True)

# NOTE: if you develop a refined clustering algorithm later, you only need to swap
# out `run_gradation_clustering` for your new function here (same call signature,
# same return value: iteration_winners) - nothing below this cell needs to change.
iteration_winners = run_gradation_clustering(
    df_tmaps, geometry, subjects, out_dir,
    target_contrast=target_contrast,
    min_size=min_size, max_size=max_size, max_seeds=max_seeds,
    relative_drop=relative_drop, absolute_min=absolute_min,
    max_rounds=max_rounds, save_all_roi=save_all_roi
)

#@title 1.2.2 Save Stage 1 ROIs (All)

# 1. Setup and initialization
mapthreshold = 0.001
subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']
views = []
# Load required data mappings
roi_clusters = read_pickle(f'{out_dir}/iteration_winners.pkl')
geometry = read_pickle('dataframe/geometry.pkl')
print(f"{boldblue}Generating Stage 1 ROI Contour Plots for all subjects...{unbold}")
for subj in subjects:
    print(f" Processing {subj}...")
    # Load geometry for the current subject
    geom = geometry[subj]
    coords, curv, faces = geom['inflated'], geom['curv'], geom['faces']
    coords_rot = _rotate_coords(coords)
    bgcurv = np.where(curv > 0, 0.7, 0.4)
    # Calculate mean raw target contrast across sessions
    raw_target = df_tmaps.query(f'subj=="{subj}"').groupby('vrtx_id')[target_contrast].mean()
    vmaxunw = raw_target.max()
    # Prepare the raw T-value statistical map
    map_raw_tval = np.zeros(coords.shape[0], dtype=np.float32)
    map_raw_tval[raw_target.index.values] = raw_target.values
    map_raw_tval[map_raw_tval < 0] = 0
    # Initialize maps and labels for contours
    roilabels = []
    roimap = np.zeros(coords.shape[0], dtype=np.int32)
    # Populate ROI IDs and labels from iteration winners
    for idx, winner in enumerate(roi_clusters[subj], start=1):
        roiname = winner['combo_name']
        unionverts = winner['combo_union_vertices']
        mni_y = winner['mni_y']
        dice = winner['dice']
        mean_tval = raw_target.loc[raw_target.index.isin(unionverts)].mean()
        roimap[unionverts] = idx
        roilabels.append(f"{roiname} (y:{mni_y}|n={len(unionverts)}|T={mean_tval:.1f}|D={dice:.2f})")
    # Generate the base surface map
    plt_title = f'{subj.capitalize()} Stage1 ROIs (vmax={vmaxunw:.3f})'
    view = plot_surf_stat_map(
        (coords_rot, faces),
        stat_map=map_raw_tval,
        bg_map=bgcurv,
        cmap='YlOrRd',
        threshold=mapthreshold,
        bg_on_data=True,
        symmetric_cmap=False,
        engine='plotly',
        title=plt_title,
    )
    # Add contours and adjust layout constraints
    view.add_contours(roimap, levels=range(1, len(roilabels) + 1), labels=roilabels)
    view.figure.update_layout(
        width=1000,
        height=600,
        #margin=dict(b=100)
    )
    view.figure.update_traces(
        colorbar=dict(
            orientation='v',
            y=0.30,
            x=1.06,
            xanchor='center',
            len=0.5,
            thickness=10
        ),
        selector=dict(type='mesh3d')
    )
    views.append(view)

    # Save individual subject plot as PNG image
    save_plotly_fig(view.figure, join(out_dir, f'{subj}_stage1_rois'))

    
# 2. Stitch individual PNGs into a combined grid layout and save
plot_cols = 3
png_paths = [join(out_dir, f'{subj}_stage1_rois.png') for subj in subjects]
images = [Image.open(path) for path in png_paths]

# Find maximum dimensions across all auto-cropped images to prevent right-edge clipping
max_w = max(img.width for img in images)
max_h = max(img.height for img in images)
total_rows = (len(images) + plot_cols - 1) // plot_cols

grid_image = Image.new('RGB', (max_w * min(len(images), plot_cols), max_h * total_rows), (255, 255, 255))
for idx, img in enumerate(images):
    row_idx, col_idx = idx // plot_cols, idx % plot_cols
    offset_x = col_idx * max_w + (max_w - img.width) // 2
    offset_y = row_idx * max_h + (max_h - img.height) // 2
    grid_image.paste(img, (offset_x, offset_y))

grid_png_file = join(out_dir, f'Stage1_ROI_contour_plots_{len(subjects)}subj.png')
grid_image.save(grid_png_file)
print(f"\n{boldgreen}Success! Saved Stage 1 contour grid layout to: {grid_png_file}{unbold}")