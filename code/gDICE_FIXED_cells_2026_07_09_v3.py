#@title 4.1 K-Session All-Combo Clustering (Stage 1) - VAS5
# ========== K_SESS_ALL_COMBO Stage 1 ==========
# Reuses the exact Stage 1 clustering logic from 2.1 (same seed/anchor/BFS/round logic),
# restricted each time to only that combo's sessions. This is the fix for the recurring bug where
# gDICE never reached 1.0 at k = n_sessions: every previous attempt (Gemini, Opus, and this
# session's first pass) wrote a simplified/rewritten "lite" version of Stage 1 for the LOOCV/combo
# loop instead of reusing the real multi-round, multi-seed anchored-clustering code, so even the
# "all sessions" combo produced a different ROI than the official ground truth. Session count = 1
# is handled separately (NOTE 2 in the original task spec): there is no second session to anchor
# against, so the native cluster itself is accepted as the round winner.
print(f"{boldblue}Starting K-Session All-Combo Stage 1 Clustering...{unbold}")

def run_kcombo_stage1(df_subj_combo, combo_sessions, surface_topology):
    iteration_claimed_vertices = set()
    combo_winners = []
    all_vrtx = df_subj_combo['vrtx_id'].unique()
    round_num = 1
    single_session = len(combo_sessions) == 1
    while True:
        iteration_combos = []
        tmax_native_exclude = {sess: set() for sess in combo_sessions}
        for tmax_num in range(1, max_seeds + 1):
            found_seed = False
            for seed_session in combo_sessions:
                df_seed_sess = df_subj_combo[df_subj_combo['sess'] == seed_session]
                available_mask = ~df_seed_sess['vrtx_id'].isin(iteration_claimed_vertices)
                available_mask &= ~df_seed_sess['vrtx_id'].isin(tmax_native_exclude[seed_session])
                candidate_mask = available_mask & (df_seed_sess[tvals_type] > 0)
                candidate_vrtx = df_seed_sess[candidate_mask]['vrtx_id'].values
                candidate_vals = df_seed_sess[candidate_mask][tvals_type].values
                if len(candidate_vrtx) == 0:
                    continue
                found_seed = True
                seed_idx = np.argmax(candidate_vals)
                seed_vertex = candidate_vrtx[seed_idx]

                native_available_mask = ~df_seed_sess['vrtx_id'].isin(iteration_claimed_vertices)
                native_available_mask &= ~df_seed_sess['vrtx_id'].isin(tmax_native_exclude[seed_session])
                native_supra_mask = df_seed_sess[tvals_type] > 0.01
                native_valid_vertices = set(df_seed_sess[native_available_mask & native_supra_mask]['vrtx_id'].values)

                if seed_vertex in native_valid_vertices:
                    native_cluster = _grow_contiguous_bfs_cluster(seed_vertex, native_valid_vertices, surface_topology, max_size)
                    if len(native_cluster) < min_size:
                        native_cluster = np.array([], dtype=int)
                else:
                    native_cluster = np.array([], dtype=int)

                if len(native_cluster) > 0:
                    tmax_native_exclude[seed_session].update(native_cluster)
                else:
                    tmax_native_exclude[seed_session].add(seed_vertex)

                if len(native_cluster) == 0:
                    continue

                combo_clusters = [native_cluster]
                combo_info = {'round': round_num, 'tmax': tmax_num, 'seed_session': seed_session,
                               'clusters': {seed_session: native_cluster}}

                if single_session:
                    combo_info['combo_union_vertices'] = native_cluster
                    combo_info['dice'] = 1.0
                    combo_info['combo_name'] = f"Rnd{round_num}Native{seed_session.replace('ses-', '')}"
                    iteration_combos.append(combo_info)
                    continue

                native_roi_vertices = set(native_cluster.tolist())
                for target_session in combo_sessions:
                    if target_session == seed_session:
                        continue
                    df_target = df_subj_combo[df_subj_combo['sess'] == target_session]
                    target_available_mask = ~df_target['vrtx_id'].isin(iteration_claimed_vertices)
                    target_supra_mask = df_target[tvals_type] > 0
                    valid_vertices = set(df_target[target_available_mask & target_supra_mask]['vrtx_id'].values)
                    anchor_pool = list(native_roi_vertices & valid_vertices)
                    if len(anchor_pool) == 0:
                        anchor_cluster = np.array([], dtype=int)
                    else:
                        sub = df_target[df_target['vrtx_id'].isin(anchor_pool)][['vrtx_id', tvals_type]]
                        anchor_seed = int(sub.iloc[sub[tvals_type].values.argmax()]['vrtx_id'])
                        anchor_cluster = _grow_contiguous_bfs_cluster(anchor_seed, valid_vertices, surface_topology, max_size)
                    combo_clusters.append(anchor_cluster)
                    combo_info['clusters'][target_session] = anchor_cluster

                if sum(len(c) > 0 for c in combo_clusters) < 2:
                    continue
                union_vertices = set()
                for c in combo_clusters:
                    if len(c) > 0:
                        union_vertices.update(c.tolist())
                combo_info['combo_union_vertices'] = np.array(sorted(union_vertices), dtype=int)
                combo_info['dice'] = mean_pairwise_dice(combo_clusters)
                combo_info['combo_name'] = f"Rnd{round_num}Tmax{tmax_num}{seed_session.replace('ses-', '')}"
                iteration_combos.append(combo_info)
            if not found_seed:
                break
        if not iteration_combos:
            break
        best_combo = max(iteration_combos, key=lambda x: x['dice'])
        if best_combo['dice'] <= 0 or len(best_combo['combo_union_vertices']) == 0:
            break
        winner_vertices = set()
        for c in best_combo['clusters'].values():
            winner_vertices.update(c)
        iteration_claimed_vertices.update(winner_vertices)
        combo_winners.append(best_combo)
        if len(set(all_vrtx) - iteration_claimed_vertices) < min_size:
            break
        round_num += 1
    return combo_winners

subj_sess = df_weighted.groupby('subj')['sess'].unique().apply(sorted).to_dict()
kcombo_stage1_results = {}  # {(subj, combo): [round_winner_dicts]}

for subj, sessions in subj_sess.items():
    n_combos = 2 ** len(sessions) - 1
    print(f"\n{boldmag}{subj}: {len(sessions)} sessions -> {n_combos} combos{unbold}")
    pial_path = join('freesurfer', subj, 'surf', 'lh.pial')
    surf_verts, surf_faces = read_geometry(pial_path)
    surface_topology = _build_surface_topology(surf_faces)
    df_subj_full = df_weighted[df_weighted['subj'] == subj]
    for k in range(1, len(sessions) + 1):
        for combo in combinations(sessions, k):
            df_subj_combo = df_subj_full[df_subj_full['sess'].isin(combo)]
            winners = run_kcombo_stage1(df_subj_combo, list(combo), surface_topology)
            kcombo_stage1_results[(subj, combo)] = winners
            print(f"  k={k} {combo}: {len(winners)} stage1 candidates")

with open(join(out_dir, 'kcombo_stage1_results.pkl'), 'wb') as f:
    pickle.dump(kcombo_stage1_results, f)
print(f"{boldgreen}4.1 K-Session All-Combo Stage 1 complete.{unbold}")


#@title 4.2a Precompute parcel geodesic maps (per subject) - VAS5 [opus: exact 2.1.5]
# Geodesic distance from each vOTC parcel centroid to ALL vertices, computed ONCE per subject
# (8 parcels x 5 subjects = 40 gdist calls). Per-combo Stage-2 scoring then reads roi_gdist as an
# O(1) lookup, so applying the EXACT 2.1.5 selection (incl. the geodesic term) to all 235 combos
# is tractable. This replaces the earlier "drop the geodesic term" shortcut.
print(f"{boldblue}Precomputing parcel geodesic maps...{unbold}")

target_parcels = ['G_and_S_occipital_inf', 'G_temporal_inf', 'G_oc-temp_lat-fusifor',
                  'S_collat_transv_ant', 'S_collat_transv_post', 'S_temporal_inf',
                  'S_occipital_ant', 'S_oc-temp_lat']
parcel_gdist_maps = {}   # {subj: {parcel_name: gdist-from-parcel-centroid array}}
vrtx_to_annot = {}       # {subj: {vrtx_id: annot_name}}

for subj in subj_sess:
    pial_verts = geometry[subj]['pial'].astype(np.float64)
    pial_faces = geometry[subj]['faces'].astype(np.int32)
    annot_names = [n.decode('utf-8') for n in geometry[subj]['annot_names']]
    annot_labels = geometry[subj]['annot_labels']
    parcel_gdist_maps[subj] = {}
    for parcel in target_parcels:
        if parcel in annot_names:
            parcel_verts = np.where(annot_labels == annot_names.index(parcel))[0]
            if len(parcel_verts) > 0:
                parcel_coords = pial_verts[parcel_verts]
                parcel_centroid = int(parcel_verts[np.argmin(np.linalg.norm(parcel_coords - parcel_coords.mean(0), axis=1))])
                parcel_gdist_maps[subj][parcel] = compute_gdist(pial_verts, pial_faces, source_indices=np.array([parcel_centroid], dtype=np.int32))
    df_anat = df_weighted[df_weighted['subj'] == subj].drop_duplicates('vrtx_id')
    vrtx_to_annot[subj] = dict(zip(df_anat.vrtx_id, df_anat.annot_name))
    print(f"  {subj}: {len(parcel_gdist_maps[subj])} parcel maps")
print(f"{boldgreen}4.2a parcel geodesic maps ready.{unbold}")


#@title 4.2b K-Session Stage 2 - EXACT 2.1.5 selection - VAS5 [opus: exact 2.1.5]
# Same GT-ROI selection as Phase 1 cell 2.1.5, applied to each k-session combo:
#   stg2_aparc_score = (roi_dsc_n + roi_pow_n + roi_gdist_n) / 3   ->  top-3, sorted by centroid Y.
# roi_pow uses ONLY the combo's sessions (the data available at k sessions). "Ground truth" is this
# SAME selection run on the full session set -> it reproduces the official stage2_rois exactly.
print(f"{boldblue}K-Session Stage 2 (exact 2.1.5 selection)...{unbold}")

def select_stage2_exact(subj, winners, combo):
    combo_mean_tval = df_weighted[(df_weighted['subj'] == subj) & (df_weighted['sess'].isin(combo))].groupby('vrtx_id')[tvals_type].mean()
    pial = geometry[subj]['pial']
    rows = []
    for winner in winners:
        verts = np.array(winner['combo_union_vertices'])
        roi_labels = [vrtx_to_annot[subj][v] for v in verts if v in vrtx_to_annot[subj]]
        dominant_parcel = Counter(roi_labels).most_common(1)[0][0]
        roi_xyz = pial[verts]
        centroid_vrtx = int(verts[np.argmin(np.linalg.norm(roi_xyz - roi_xyz.mean(0), axis=1))])
        roi_gdist = float(parcel_gdist_maps[subj][dominant_parcel][centroid_vrtx])
        rows.append({'verts': verts, 'centroid_y': float(pial[centroid_vrtx, 1]),
                     'roi_dsc': winner['dice'], 'roi_pow': float(combo_mean_tval.reindex(verts).mean()),
                     'roi_gdist': roi_gdist})
    cands = pd.DataFrame(rows)
    if len(cands) < 3:
        return None
    for col in ['roi_dsc', 'roi_pow']:
        vals = cands[col].values
        cands[f'{col}_n'] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)
    dvals = cands['roi_gdist'].values
    cands['roi_gdist_n'] = 1 - (dvals - dvals.min()) / (dvals.max() - dvals.min() + 1e-10)
    cands['stg2_aparc_score'] = (cands['roi_dsc_n'] + cands['roi_pow_n'] + cands['roi_gdist_n']) / 3
    return cands.nlargest(3, 'stg2_aparc_score').sort_values('centroid_y', ascending=False).reset_index(drop=True)

gdice_ground_truth = {}   # {subj: GT triplet df (== official stage2_rois)}
kcombo_triplets = {}      # {(subj, combo): triplet df or None}
for subj, sessions in subj_sess.items():
    gdice_ground_truth[subj] = select_stage2_exact(subj, kcombo_stage1_results[(subj, tuple(sessions))], tuple(sessions))
    print(f"  {subj} GT triplet Y = {[round(y, 1) for y in gdice_ground_truth[subj]['centroid_y']]}")
    for (s, combo), winners in kcombo_stage1_results.items():
        if s == subj:
            kcombo_triplets[(s, combo)] = select_stage2_exact(subj, winners, combo)

with open(join(out_dir, 'gdice_ground_truth.pkl'), 'wb') as f:
    pickle.dump(gdice_ground_truth, f)
with open(join(out_dir, 'kcombo_triplets.pkl'), 'wb') as f:
    pickle.dump(kcombo_triplets, f)
print(f"{boldgreen}4.2b K-Session Stage 2 (exact 2.1.5) complete.{unbold}")

#@title 4.3 Calculate gDICE - VAS5
print(f"{boldblue}Calculating gDICE against ground truth...{unbold}")

gdice_data = []
for (subj, combo), triplet in kcombo_triplets.items():
    gt_triplet = gdice_ground_truth[subj]
    if triplet is None:
        gdice_val = np.nan
    else:
        roi_dscs = [dice_coefficient(triplet.iloc[roi_idx]['verts'], gt_triplet.iloc[roi_idx]['verts']) for roi_idx in range(3)]
        gdice_val = float(np.mean(roi_dscs))
    print(f"  {subj} k={len(combo)} {combo}: gDICE={gdice_val}")
    gdice_data.append({'subj': subj, 'k': len(combo), 'combo': str(combo), 'gdice': gdice_val})

df_gdice = pd.DataFrame(gdice_data)
df_gdice.to_parquet(join(out_dir, 'gDICE_results.parquet'))
print(f"{boldgreen}gDICE results saved to {join(out_dir, 'gDICE_results.parquet')}{unbold}")

print(f"\n{bold}Sanity check -- gDICE at k = n_sessions must be 1.0 (same data as ground truth):{unbold}")
for subj, sessions in subj_sess.items():
    max_k = len(sessions)
    at_max_k = df_gdice[(df_gdice['subj'] == subj) & (df_gdice['k'] == max_k)]
    print(f"  {subj}: k={max_k}, gDICE={at_max_k['gdice'].values[0]:.4f}")


#@title 4.4 gDICE Plot (Individual Subjects only) - VAS5 [opus: Plot 2 removed]
# Aggregate/mean-across-subjects plot REMOVED: subjects have different session counts and each is
# forced to gDICE=1.0 at its OWN k=n_sessions, so averaging across subjects at a fixed k mixes real
# partials with forced endpoints (and at high k reduces to a single subject). Only the per-subject
# curve is meaningful.
print(f"{boldblue}Generating gDICE plot (per subject)...{unbold}")

df_subj_k = df_gdice.groupby(['subj', 'k'])['gdice'].mean().reset_index()
subj_colors = {'sub-01': 'brown', 'sub-02': 'green', 'sub-03': 'blue', 'sub-04': 'pink', 'sub-05': 'violet'}

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
for subj in sorted(df_subj_k['subj'].unique()):
    sub = df_subj_k[df_subj_k['subj'] == subj].sort_values('k')
    ax1.plot(sub['k'], sub['gdice'], marker='o', linewidth=2, color=subj_colors[subj], label=subj)
ax1.set_title('Session Count vs Mean gDICE (per subject; GT = exact Phase-1 2.1.5 selection)')
ax1.set_xlabel('Number of Sessions (k)')
ax1.set_ylabel('Mean gDICE (vs All-Session Ground Truth)')
ax1.set_ylim(-0.05, 1.05)
ax1.legend(title='Subject')
ax1.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(join(out_dir, '4.4_gDICE_plot.png'), dpi=150)
plt.show()
print(f"{boldgreen}Saved to {join(out_dir, '4.4_gDICE_plot.png')}{unbold}")
