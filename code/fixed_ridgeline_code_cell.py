#@title 2.2.1 View Distribution - Ridgeline (primary) + Violin (fallback) [o4]
anat_measures = [('anat_depth', 'Sulcal Depth (lh.sulc)'),
                 ('anat_ct',    'Thickness (mm)'),
                 ('anat_curv',  'Curvature')]
row_gap = 1.6       # spacing between ROI baselines
kde_height = 1.1    # max KDE height per row (< row_gap -> rows never overlap)

def rois_anterior_to_posterior(subj):
    pial_coords = geometry[subj]['pial']
    return sorted(iteration_winners[subj],
                  key=lambda w: -roi_centroid_y(pial_coords, np.array(w['combo_union_vertices'])))

subj = 'sub-01'  #@param ['sub-01','sub-02','sub-03','sub-04','sub-05']
anat_by_vrtx = df_tmaps[df_tmaps['subj'] == subj].drop_duplicates('vrtx_id').set_index('vrtx_id')
ordered_rois = rois_anterior_to_posterior(subj)
n_rows = len(ordered_rois)

# --- Ridgeline: one row per ROI (anterior top -> posterior bottom); GMM lines on sulcal depth only ---
fig, axes = plt.subplots(1, 3, figsize=(13, 0.62 * n_rows + 2.0))
for panel_idx, (col, axis_label) in enumerate(anat_measures):
    ax = axes[panel_idx]
    all_vals = np.concatenate([anat_by_vrtx.reindex(w['combo_union_vertices'])[col].values for w in ordered_rois])
    grid = np.linspace(all_vals.min(), all_vals.max(), 220)
    baselines = []
    for row_idx, winner in enumerate(ordered_rois[::-1]):
        baseline = row_idx * row_gap
        baselines.append(baseline)
        vals = anat_by_vrtx.reindex(winner['combo_union_vertices'])[col].values
        color = roi_palette[(n_rows - 1 - row_idx) % len(roi_palette)]
        if vals.std() > 1e-6:
            kde = gaussian_kde(vals)
            density_max = kde(grid).max()
            density = kde(grid) / density_max * kde_height
            ax.fill_between(grid, baseline, baseline + density, color=color, alpha=0.6, lw=0.8, edgecolor='k', zorder=3 * row_idx + 2)
            if col == 'anat_depth':
                depth_gmm, _ = fit_depth_gmm(vals)
                for mean_val in depth_gmm.means_.ravel():
                    line_height = float(kde(mean_val)[0] / density_max * kde_height)   # reach this ROI's own curve
                    ax.plot([mean_val, mean_val], [baseline, baseline + line_height], color='k', ls=':', lw=1.0, zorder=3 * row_idx + 3)
        ax.plot(vals, np.full(len(vals), baseline - 0.12), '|', color=color, ms=4, alpha=0.5, zorder=3 * row_idx + 1)
    ax.set_ylim(-0.45, (n_rows - 1) * row_gap + kde_height + 0.25)
    ax.set_xlabel(axis_label, fontsize=9)
    ax.set_title(axis_label.split(' (')[0], fontsize=10, fontweight='bold')
    ax.spines[['right', 'top']].set_visible(False)
    if panel_idx == 0:
        ax.set_yticks(baselines)
        ax.set_yticklabels([w['combo_name'] for w in ordered_rois[::-1]], fontsize=7)
        ax.tick_params(left=False)
        ax.spines['left'].set_visible(False)
    else:
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
fig.text(0.012, 0.80, 'anterior', rotation=90, fontsize=8, color='dimgray', ha='center', va='center')
fig.text(0.012, 0.50, '^', fontsize=13, color='dimgray', ha='center', va='center')
fig.text(0.012, 0.20, 'posterior', rotation=90, fontsize=8, color='dimgray', ha='center', va='center')
fig.suptitle(f'{subj} - Per-ROI anatomical distributions   (rows: anterior top -> posterior bottom;  dotted = per-ROI GMM means, sulcal depth only)', fontsize=10.5, fontweight='bold')
plt.tight_layout(rect=[0.03, 0, 1, 0.99])
plt.show()

# --- Violin (fallback) ---
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for panel_idx, (col, axis_label) in enumerate(anat_measures):
    ax = axes[panel_idx]
    data = [anat_by_vrtx.reindex(w['combo_union_vertices'])[col].values for w in ordered_rois]
    parts = ax.violinplot(data, showmedians=True)
    for body_idx, body in enumerate(parts['bodies']):
        body.set_facecolor(roi_palette[body_idx % len(roi_palette)]); body.set_alpha(0.5)
    ax.set_xticks(range(1, n_rows + 1))
    ax.set_xticklabels([w['combo_name'].split('Tmax')[0].replace('Rnd', 'R') for w in ordered_rois], rotation=60, fontsize=6)
    ax.set_ylabel(axis_label, fontsize=9); ax.set_title(axis_label.split(' (')[0], fontsize=10, fontweight='bold')
fig.suptitle(f'{subj} - Anatomical distributions (violin fallback)', fontsize=11, fontweight='bold')
plt.tight_layout(); plt.show()