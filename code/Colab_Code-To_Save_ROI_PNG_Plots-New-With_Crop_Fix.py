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
    """Calculate mean pairwise DICE, handling empty clusters"""
    if len(cluster_list) < 2:
        return 0.0
    d_sum, n_pair = 0.0, 0
    for i in range(len(cluster_list)):
        for j in range(i+1, len(cluster_list)):
            d_sum += dice_coefficient(cluster_list[i], cluster_list[j])
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

def _grow_contiguous_bfs_cluster(seed_vertex, supra_threshold_vertices_set, surface_topology, max_size):
    """
    Grows a single contiguous cluster from a seed vertex using Breadth-First Search (BFS).
    """
    # check if the seed vertex is valid
    if seed_vertex not in supra_threshold_vertices_set:
        return np.array([], dtype=int)

    # initialize the cluster, queue for BFS, and a set to track visited vertices
    found_cluster_vertices = []
    q = deque([seed_vertex])
    visited_vertices = {seed_vertex}

    # grow cluster until the queue is empty or max_size is reached
    while q and len(found_cluster_vertices) < max_size:
        current_vertex = q.popleft()
        found_cluster_vertices.append(current_vertex)

        # find all neighbors of the current vertex
        for neighbor in surface_topology.get(current_vertex, []):
            # add neighbor to the queue if it's a valid, unvisited vertex
            if neighbor in supra_threshold_vertices_set and neighbor not in visited_vertices:
                visited_vertices.add(neighbor)
                q.append(neighbor)

    return np.array(found_cluster_vertices, dtype=int)

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

## ===== 1.1 TMAP

#@title 1.1.0 Set Params

# GLM-I https://nilearn.github.io/dev/modules/generated/nilearn.glm.compute_contrast.html
# Test directionality: one-tailed
df_tmaps = read_parquet('dataframe/df_tmaps.parquet')
df_scaled_tmaps = df_tmaps
target_contrast = 'WordvsPER'
tvals_type = 'scaled_tval'
n_bins = 10 #
tval_threshold = 1.65 #t≈1.65|2.33(p<0.05|0.01)
local_threshold = 0.5

# Set Output File Paths
timestamp = (datetime.now() + timedelta(hours=2)).strftime("%b%d_%Hh%Mm")
tvals_prefix = tvals_type[:8]
out_dir = join(save_root,'results',f'{target_contrast}_{tvals_prefix}_{n_bins}bins_{local_threshold}thresh_{timestamp}')
makedirs(out_dir, exist_ok=True)
tmap_type = 'scaled'

# @title 1.1.1 Threshold and Scale T-values



# 1. Create bins based on y-axis
df_tmaps['y_bin'] = pd.cut(df_tmaps['y_pos'], bins=n_bins)

# 2. Group data per Subject, per Session, per Bin for local threshold and local max
grouped = df_tmaps.groupby(['subj', 'sess', 'y_bin'], observed=False)[target_contrast]

# 3. Set nth percentile of positive values in the bin as local threshold
calc_quantile = lambda x: x[x > tval_threshold].quantile(local_threshold) if (x > tval_threshold).any() else 0
df_tmaps['local_thresh'] = grouped.transform(calc_quantile)

# 4. Set maximum value in the bin as Local Max (Ceiling):
df_tmaps['local_max'] = grouped.transform('max')

# 5. Scale values (Threshold -> 0, Max ->1) (t-val - threshold) / (max - threshold)
scaled_vals = (df_tmaps[target_contrast] - df_tmaps['local_thresh']) / (df_tmaps['local_max'] - df_tmaps['local_thresh'])
df_tmaps[tvals_type] = where(df_tmaps[target_contrast] >= df_tmaps['local_thresh'], scaled_vals, 0.0)

# @title 1.1.2a Raw vs Scaled MNI-Y
subject = 'sub-01'  # @param ['sub-01','sub-02','sub-03','sub-04','sub-05']

# Aggregate data identically to the brain surface maps (mean across sessions per vertex)
subj_data = df_tmaps.query('subj == @subject').groupby('vrtx_id').mean(numeric_only=True).reset_index()

# Setup side-by-side plot layout
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

fig.suptitle(f'{subject} - {target_contrast} - MNI-Y vs T-values (Mean across sessions)', fontsize=15, y=1.05)

# --- Plot 1: Raw T-values ---
sc1 = axes[0].scatter(subj_data['y_pos'], subj_data[target_contrast],
                      c=subj_data[target_contrast], cmap='YlOrRd', alpha=0.6, s=15)
axes[0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
axes[0].invert_xaxis()
axes[0].set_title('Raw T-values', fontsize=12, pad=10)
axes[0].set_xlabel('y_pos (anterior ← → posterior)', fontsize=10)
axes[0].set_ylabel('Raw t-value', fontsize=10)
fig.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04)

# --- Plot 2: Scaled T-values ---
sc2 = axes[1].scatter(subj_data['y_pos'], subj_data[tvals_type],
                      c=subj_data[tvals_type], cmap='YlOrRd', alpha=0.6, s=15)
axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
axes[1].invert_xaxis()
axes[1].set_title('Scaled T-values', fontsize=12, pad=10)
axes[1].set_xlabel('y_pos (anterior ← → posterior)', fontsize=10)
axes[1].set_ylabel('Scaled t-value (0.0 to 1.0)', fontsize=10)
fig.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout(w_pad=12.0, rect=[0.25, 0, 1, 1])
plt.show()

#@title 1.1.2b Raw vs Scaled TMap
subject = 'sub-01'  # @param ['sub-01','sub-02','sub-03','sub-04','sub-05']

geometry = read_pickle('dataframe/geometry.pkl')

subj = subject

geom = geometry[subj]
coords, curv, faces = geom['inflated'], geom['curv'], geom['faces']
coords_rot = _rotate_coords(coords)
bg_std = np.where(curv > 0, 0.7, 0.4)

raw_target = df_tmaps.query('subj == @subj').groupby('vrtx_id')[target_contrast].mean()
scaled_target = df_tmaps.query('subj == @subj').groupby('vrtx_id')[tvals_type].mean()

raw_map = np.zeros(coords.shape[0])
scaled_map = np.zeros(coords.shape[0])

raw_map[raw_target.index.values] = raw_target.values
scaled_map[scaled_target.index.values] = scaled_target.values

raw_title = (
    f"<span style='font-size: 20px;'>{subj} - {target_contrast} - Raw T-values"
    f"<br><span style='font-size: 15px;'>T-value Range Before Scaling: {raw_target.min():.1f} (min) : {raw_target.max():.1f} (max) [mean across sessions]</span>"
)

scaled_title = (
    f"<span style='font-size: 20px;'>{subj} - {target_contrast} - Scaled T-values"
    f"<br><span style='font-size: 15px;'>&nbsp;</span>"
)

raw_view = view_surf(
    (coords_rot, faces),
    surf_map=raw_map,
    bg_map=bg_std,
    cmap='YlOrRd',
    threshold=0.1,
    symmetric_cmap=False,
    bg_on_data=True,
    colorbar_fontsize=15,
    colorbar_height=0.6,
    title=raw_title
)

scaled_view = view_surf(
    (coords_rot, faces),
    surf_map=scaled_map,
    bg_map=bg_std,
    cmap='YlOrRd',
    threshold=0.0001,
    symmetric_cmap=False,
    bg_on_data=True,
    colorbar_fontsize=15,
    colorbar_height=0.6,
    title=scaled_title
)

raw_view.resize(700, 500)
scaled_view.resize(700, 500)

combined_html = f"""
<div style="display: flex; gap: 0px;">
    <div>{raw_view._repr_html_()}</div>
    <div>{scaled_view._repr_html_()}</div>
</div>
"""

display(HTML(combined_html))

#@title 1.1.3 Save TMap (All)
# # Set Output File Paths
# timestamp = (datetime.now() + timedelta(hours=2)).strftime("%b%d_%Hh%Mm")
# tvals_prefix = tvals_type[:8]
# out_dir = join(save_root,'results',f'{target_contrast}_{tvals_prefix}_{n_bins}bins_{local_threshold}thresh_{timestamp}')
# makedirs(out_dir, exist_ok=True)
# tmap_type = 'scaled'

geometry = read_pickle('dataframe/geometry.pkl')
subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']
views = []
for subj in subjects:
    print(f"Processing {subj}...")

    geom = geometry[subj]
    coords, curv, faces = geom['inflated'], geom['curv'], geom['faces']
    coords_rot = _rotate_coords(coords)
    bg_std = np.where(curv > 0, 0.7, 0.4) #0->1 white to black sulci:gyri

    # Calculate mean across all sessions
    raw_target = df_tmaps.query('subj==@subj').groupby('vrtx_id')[target_contrast].mean()
    scaled_target = df_tmaps.query('subj==@subj').groupby('vrtx_id')[tvals_type].mean()

    surf_map = np.zeros(coords.shape[0])
    if tmap_type == 'raw':
      map_threshold = 0.1
      surf_map[raw_target.index.values] = raw_target.values
    elif tmap_type == 'scaled':
      map_threshold = 0.0001
      surf_map[scaled_target.index.values] = scaled_target.values
    else:
      raise ValueError(f"Unknown tmap_type: {tmap_type}")

    plot_title = (f"<span style='font-size: 20px;'>{subj} - {target_contrast} - Local threshold - {local_threshold*100}th percentile"
    f"<br><span style='font-size: 15px;'>Min and Max RAW T-values (mean): {raw_target.min():.1f} : {raw_target.max():.1f}</span>")

    view = view_surf((coords_rot, faces), surf_map=surf_map,
                      bg_map=bg_std,
                      cmap='YlOrRd',
                      threshold=map_threshold,
                      symmetric_cmap=False,
                      bg_on_data=True,
                      colorbar_fontsize=15,
                      colorbar_height=0.6,
                      title=plot_title)
    views.append(view.resize(850, 600))


plot_cols = 3
html_parts = [view._repr_html_() for view in views]
style = f"display: grid; grid-template-columns: repeat({plot_cols}, max-content);"
combined_html_scaled_tmap = f"<div style='{style}'>{''.join(html_parts)}</div>"

print(f"\n{bold}Uncomment to View/Save{unbold}")
# display(HTML(combined_html_scaled_tmap))


## ===== 1.2 ROI

#@title 1.2.1 Clustering

# ========== Parameters ==========
df_weighted = df_tmaps.copy()


# Our target is to get 50 vertex ROIs
min_size = 50
max_size = 100
max_seeds = 2
max_rounds = -1

# Store round winners
save_all_roi = False
iteration_winners = {}
subj_sess = df_weighted.groupby('subj')['sess'].unique().to_dict()

# ========== Subject loop ==========
for subj, sessions in subj_sess.items():
    print(f"\n{boldmag}===== {subj} ====={unbold}")

    # Load data and build topology
    #pial_path = join('freesurfer', subj, 'surf', 'lh.pial')
    #surf_verts, surf_faces = read_geometry(pial_path)
    geometry = read_pickle('dataframe/geometry.pkl')
    surf_faces = geometry[subj]['faces']
    surface_topology = _build_surface_topology(surf_faces)

    df_subj = df_weighted[df_weighted['subj'] == subj]

    # Define search space (all vertices in bigLOTS)
    vrtx_coords = df_subj[['vrtx_id', 'y_pos']].drop_duplicates().set_index('vrtx_id')
    all_vrtx = vrtx_coords.index.values
    searchspace = {row.vrtx_id: (row.x_pos, row.y_pos, row.z_pos) for row in df_subj.itertuples()}

    # Track vertices claimed across iterations
    iteration_claimed_vertices = set()
    subject_iteration_winners = []

    # ========== Rounds loop ==========
    round_iterator = count(1) if max_rounds == -1 else range(1, max_rounds + 1)
    for round_num in round_iterator:
        print(f"\n{boldblue}  --- Round {round_num} ---{unbold}")

        iteration_combos = []
        tmax_native_exclude = {sess: set() for sess in sessions}

        # ========== Tmax loop ==========
        tmax_iterator = count(1) if max_seeds == -1 else range(1, max_seeds + 1)
        for tmax_num in tmax_iterator:
            found_seed_in_tmax_iteration = False

            # ========== Session loop for seed selection ==========
            for s_idx, seed_session in enumerate(sessions, start=1):
                if s_idx == 1:
                    print(f"{bold}\t--- Seed Loop: {tmax_num} out of {max_seeds}seeds ({subj} Rnd{round_num} Tmax{tmax_num}) ---{unbold}")
                roi_name = seed_session.replace('ses-', '')
                print(f"\t\t--- ROI Loop: {roi_name} ({s_idx} out of {len(sessions)})---")
                print(f"\t\t  -------------------- tmax{tmax_num} - {roi_name} --------------------")

                # Get available vertices for seed selection
                df_seed_sess = df_subj[(df_subj['sess'] == seed_session)]
                region_mask = df_seed_sess['vrtx_id'].isin(all_vrtx)

                # Exclude previous-round winners
                available_mask = ~df_seed_sess['vrtx_id'].isin(iteration_claimed_vertices)

                # Exclude this session's prior NATIVE ROIs from earlier TMAX in this round
                available_mask &= ~df_seed_sess['vrtx_id'].isin(tmax_native_exclude[seed_session])

                # Candidate seed vertices must be supra-threshold
                candidate_mask = region_mask & available_mask & (df_seed_sess[tvals_type] > 0)
                candidate_vrtx = df_seed_sess[candidate_mask]['vrtx_id'].values
                candidate_vals = df_seed_sess[candidate_mask][tvals_type].values

                if len(candidate_vrtx) == 0:
                    print(f"\t    No available vertices for seed in {seed_session}")
                    continue

                found_seed_in_tmax_iteration = True

                # Pick highest value as seed
                seed_idx = np.argmax(candidate_vals)
                seed_vertex = candidate_vrtx[seed_idx]
                native_seed_tval = candidate_vals[seed_idx]

                # 1) Native session clustering
                native_session = seed_session
                df_native = df_subj[df_subj['sess'] == native_session]
                native_region_mask = df_native['vrtx_id'].isin(all_vrtx)

                native_available_mask = ~df_native['vrtx_id'].isin(iteration_claimed_vertices)
                native_available_mask &= ~df_native['vrtx_id'].isin(tmax_native_exclude[native_session])
                native_supra_mask = (df_native[tvals_type] > 0.01)
                native_valid_mask = native_region_mask & native_available_mask & native_supra_mask
                native_valid_vertices = set(df_native[native_valid_mask]['vrtx_id'].values)

                if seed_vertex in native_valid_vertices:
                    native_cluster = _grow_contiguous_bfs_cluster(
                        seed_vertex, native_valid_vertices, surface_topology, max_size
                    )
                    if len(native_cluster) < min_size:
                        native_cluster = np.array([], dtype=int)
                else:
                    native_cluster = np.array([], dtype=int)

                if save_all_roi: write_individual_roi_label(
                    out_dir, subj, round_num, tmax_num, seed_session, native_session,
                    native_cluster, searchspace
                )

                if len(native_cluster) > 0:
                    tmax_native_exclude[native_session].update(native_cluster)
                else:
                    tmax_native_exclude[native_session].add(seed_vertex)

                native_vertices_text = f"{len(native_cluster)} vertices" if len(native_cluster) > 0 else "empty"
                native_y = df_native.loc[df_native['vrtx_id'] == seed_vertex, 'y_pos'].values[0]
                native_rawtval = df_native.loc[df_native['vrtx_id'] == seed_vertex, target_contrast].values[0]
                print(f"\t\t      {native_session} (native): {native_vertices_text}",
                      f" (seed vertex: {seed_vertex} | mni_y: {native_y:.2f}",
                      f"| tval: {native_seed_tval:.3f} | raw_tval: {native_rawtval:.3f})")

                if len(native_cluster) == 0:
                    continue

                combo_clusters = []
                combo_info = {
                    'round': round_num,
                    'tmax': tmax_num,
                    'seed_session': seed_session,
                    'seed_vertex': seed_vertex,
                    'seed_tval': native_seed_tval,
                    'seed_rawtval': f"{native_rawtval:.2f}",
                    'mni_y': f"{native_y:.2f}",
                    'clusters': {}
                }

                combo_clusters.append(native_cluster)
                combo_info['clusters'][native_session] = native_cluster

                # 2) Anchored sessions
                native_roi_vertices = set(native_cluster.tolist())
                for target_session in sessions:
                    if target_session == native_session:
                        continue

                    df_target = df_subj[df_subj['sess'] == target_session]
                    region_mask_tgt = df_target['vrtx_id'].isin(all_vrtx)

                    target_available_mask = ~df_target['vrtx_id'].isin(iteration_claimed_vertices)
                    target_supra_mask = (df_target[tvals_type] > 0)
                    target_valid_mask = region_mask_tgt & target_available_mask & target_supra_mask
                    valid_vertices = set(df_target[target_valid_mask]['vrtx_id'].values)

                    ss_anch_tmax = list(native_roi_vertices.intersection(valid_vertices))

                    if len(ss_anch_tmax) == 0:
                        anchor_cluster = np.array([], dtype=int)
                        print(f"\t\t      {target_session} (anchored): empty (seed vertex: Not Found | tval: NA)")
                    else:
                        sub = df_target[df_target['vrtx_id'].isin(ss_anch_tmax)][['vrtx_id', tvals_type]]
                        anchor_seed_idx = int(sub.iloc[sub[tvals_type].values.argmax()]['vrtx_id'])
                        anchor_seed_tval = float(sub.loc[sub['vrtx_id'] == anchor_seed_idx, tvals_type].values[0])

                        anchor_cluster = _grow_contiguous_bfs_cluster(
                            anchor_seed_idx, valid_vertices, surface_topology, max_size
                        )

                        anchor_vertices_text = f"{len(anchor_cluster)} vertices" if len(anchor_cluster) > 0 else "empty"
                        anchor_y = df_target.loc[df_target['vrtx_id'] == anchor_seed_idx, 'y_pos'].values[0]
                        anchor_rawtval = df_target.loc[df_target['vrtx_id'] == anchor_seed_idx, target_contrast].values[0]
                        print(f"\t\t      {target_session} (anchored): {anchor_vertices_text}",
                              f" (seed vertex: {anchor_seed_idx} | mni_y: {anchor_y:.2f}",
                              f"| tval: {anchor_seed_tval:.3f} | raw_tval: {anchor_rawtval:.3f})")

                    if save_all_roi: write_individual_roi_label(
                        out_dir, subj, round_num, tmax_num, seed_session, target_session,
                        anchor_cluster, searchspace
                    )
                    combo_clusters.append(anchor_cluster)
                    combo_info['clusters'][target_session] = anchor_cluster

                combo_union_vertices = set()
                for cl in combo_clusters:
                    if len(cl) > 0:
                        combo_union_vertices.update(cl.tolist())
                combo_union_vertices = np.array(sorted(combo_union_vertices), dtype=int)

                non_empty_count = sum(len(c) > 0 for c in combo_clusters)
                if non_empty_count < 2:
                    print(f"\t\t  Skipping combo (insufficient non-empty ROIs: {non_empty_count}).")
                    continue

                if save_all_roi:
                    combo_info['combo_union_label'] = write_combo_union_label(
                        out_dir, subj, round_num, seed_session, tmax_num,
                        combo_union_vertices, searchspace
                    )

                combo_info['combo_union_vertices'] = combo_union_vertices

                combo_dice = mean_pairwise_dice(combo_clusters)
                combo_info['dice'] = combo_dice
                combo_info['combo_name'] = f"Rnd{round_num}Tmax{tmax_num}{roi_name}"
                iteration_combos.append(combo_info)
                print(f"\t\t  DICE of iter{round_num}_{roi_name}_Tmax{tmax_num} Combo: {combo_dice:.3f}")

            if not found_seed_in_tmax_iteration:
                print("\t    No candidate seeds remain in any session; stopping TMAX loop.")
                break

        # ========== Find round winner ==========
        if iteration_combos:
            best_combo = max(iteration_combos, key=lambda x: x['dice'])

            print(f"\n{boldgreen}  Round {round_num} Winner: {best_combo['combo_name']}",
                  f"(SEED_TVAL={best_combo['seed_rawtval']} |",
                  f"MNI_Y={best_combo['mni_y']} |",
                  f"SIZE={len(best_combo['combo_union_vertices'])} |",
                  f"DICE={best_combo['dice']:.3f})")

            if best_combo['dice'] <= 0 or len(best_combo.get('combo_union_vertices', [])) == 0:
                print(f"\n{boldred}  No reliable ROI found (winner empty or zero-DICE). Stopping further rounds.{unbold}")
                break

            winner_vertices = set()
            for sess_cluster in best_combo['clusters'].values():
                winner_vertices.update(sess_cluster)

            iteration_claimed_vertices.update(winner_vertices)
            subject_iteration_winners.append(best_combo)

            seed_sess_clean = best_combo['seed_session'].replace('ses-', '')
            winner_union_path = join(out_dir, subj, f"_rnd{round_num}_winner==tmax{best_combo['tmax']}_{seed_sess_clean}_unionROI.label")

            if save_all_roi and best_combo.get('combo_union_label'):
                copy2(best_combo['combo_union_label'], winner_union_path)
            else:
                # Generate temp file and rename to winner path
                tmp = write_combo_union_label(out_dir, subj, round_num,
                                              best_combo['seed_session'],
                                              best_combo['tmax'],
                                              best_combo['combo_union_vertices'],
                                              searchspace)
                os.replace(tmp, winner_union_path)

            available = set(all_vrtx) - iteration_claimed_vertices
            if len(available) < min_size:
                print(f"\n{boldred}  No more vertices available for clustering. Stopping at round {round_num}.{unbold}")
                break
        else:
            print(f"\n{boldred}  No valid combos found in round {round_num}. Stopping.{unbold}")
            break

    iteration_winners[subj] = subject_iteration_winners
    print(f"\n{boldmag}  Final winners for {subj}:{unbold}")
    for winner in subject_iteration_winners:
        print(f"    Round {winner['round']}: {winner['combo_name']}",
              f"(SEED_TVAL={winner['seed_rawtval']} |",
              f"MNI_Y={winner['mni_y']} |",
              f"SIZE={len(winner['combo_union_vertices'])} |",
              f"DICE={winner['dice']:.3f})")
    winners_path = join(out_dir, 'iteration_winners.pkl')
    with open(winners_path, 'wb') as f:
        pickle.dump(iteration_winners, f)
    print(f"  Saved iteration winners to: {winners_path}")

print(f"\n{boldgreen}Clustering complete. Results saved to: {out_dir}{unbold}")

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