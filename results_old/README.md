# Phase 1 · Stage 2 — Sections 2.2–2.5 (cov4.8_v1, 2026-07-09)

Completes the "SCORE ROIs" stage of the VWFA delineation pipeline. Each candidate Stage‑1 ROI
is scored **solely** on one factor (so its individual impact is visible), then all factors are
combined into the final a/c/p‑VWFA selection.

- **Code**: `../../colab_code_2026_07_09.py` (Colab cells, appended under section `cov4.8_v1`; existing cells untouched).
- **Local reproducer**: `local_run_2026_07_09.py` (set `ROOT` once, run; regenerates everything here from `dataframe/`).
- Inputs used: `dataframe/df_tmaps.parquet`, `dataframe/geometry.pkl`, Stage‑1 `iteration_winners.pkl`. 38 candidate ROIs across 5 subjects.

> Note on the surface PNGs: rendered as a **zoomed ventral‑OTC patch** (bigLOTS + margin) rather
> than the whole hemisphere — same data as the interactive plotly views in the Colab cells, just
> cropped for speed and legibility. Colours: aVWFA = red, cVWFA = green, pVWFA = blue.

---

## 2.2 Cortical Surface — `2.2_cortical_surface/`
**Method.** For each candidate ROI, a Gaussian Mixture Model (components chosen by BIC, 1–4) is fit
to the vertices' **sulcal depth** (`anat_depth`), *strictly per ROI*. The ROI's `roi_depth_score`
rewards anatomical consistency = inverse of the median deviation from the **nearest GMM peak**
(so a tight uni-/bi-modal ROI scores high; a diffuse ROI scores low). Multi-peak + steepness are
captured because likelihood under a narrow peak falls off faster than under a broad one. Per-vertex
typicality (normalised GMM likelihood) is retained for later refinement.

**Why ridgeline, not violin.** `ridgeline_<subj>.png` (primary) stacks per‑ROI KDEs of depth /
thickness / curvature ordered anterior→posterior, with the actual vertices as a rug and dotted GMM
peaks. It exposes the *multi‑modal, per‑ROI* structure that a violin hides and that motivates the
GMM. `violin_<subj>.png` is kept as the requested fallback.

**Outputs.** `ridgeline_<subj>.png`, `violin_<subj>.png`, `rois_2_2_<subj>.png`, `rois_2_2_ALL.png`,
`cluster_table_anat.csv` (per-ROI depth stats + score), `stage2_2_rois.pkl`, `vertex_typicality.pkl`, `labels/`.

*Suggested caption (Fig).* "Per‑ROI distribution of cortical sulcal depth (ridgeline; dotted lines = GMM component means). The cortical‑surface criterion selects ROIs whose vertices share a consistent folding profile."

## 2.3 Non-target Contrast — `2.3_nontarget/`
**Method.** Word ROIs should not overlap Faces/Limbs‑selective cortex. Using a per‑subject **p90**
threshold on each non-target contrast (matching the existing Non-target Summary cell), a vertex is
penalised (score 0) if its `FacesvsNull` or `LimbsvsNull` t exceeds threshold; otherwise it is
inverse-scaled by its strongest non-target activation. `roi_nontarget_score` = mean vertex score
(higher = more word‑specific). Anterior ROIs (away from posterior fusiform face cortex) score higher —
as expected.

**Outputs.** `selectivity_<subj>.png` (Word‑vs‑Faces / Word‑vs‑Limbs per-vertex scatter, ROI vertices
highlighted, p90 line), `rois_2_3_<subj>.png`, `rois_2_3_ALL.png`, `cluster_table_nontarget.csv`,
`stage2_3_rois.pkl`, `labels/`.

*Suggested caption.* "Target vs non‑target selectivity. Candidate‑ROI vertices (red) are largely word‑specific (below the equality line and the p90 non‑target threshold)."

## 2.4 Session Count — `2.4_session_count/`
**Method.** Keep only vertices appearing in **> 50 %** of the subject's sessions, then enforce
contiguity by retaining the **largest connected component** of the survivors (literature: functional
ROIs are conventionally spatially contiguous, so trimming is followed by a re‑clustering step rather
than leaving fragments). Reliable early-round ROIs retain 50–75 %; noisy late-round ROIs are trimmed
to 12–30 % — the intended behaviour.

**Outputs.** `sesscount_<subj>.png` (per-vertex session-count heatmap + trimmed cores),
`sesscount_ALL.png`, `cluster_table_sessioncount.csv` (orig vs kept vs contiguous counts, fragments),
`trimmed_rois.pkl`, `labels/`.

## 2.5 Stage 2 — Combined — `2.5_combined/`
**Method.** The three per-ROI factor scores (2.1 parcellation `stg2_aparc_score`, 2.2 `roi_depth_score`,
2.3 `roi_nontarget_score`) are each normalised 0–1 within subject and averaged (equal weights, editable
at the top of the cell). Vertices are taken from the 2.4 session‑trimmed contiguous cores. The top‑3
ROIs by combined score (≥20 vertices) are sorted by centroid‑Y into **aVWFA / cVWFA / pVWFA**.

**Outputs.** `rois_final_<subj>.png`, `rois_final_ALL.png` (all 5 subjects), `factor_comparison_sub-01.png`
(2.2 vs 2.3 vs final), `final_selection.csv` (per-factor + combined breakdown), `cluster_table_combined_ALL.csv`,
`stage2_final_rois.pkl`, `labels/`.

### Final selection (per subject: ROI, n vertices, Y, factor scores)
See `2.5_combined/final_selection.csv`. Anterior→posterior ordering verified (Y decreases a→p).

---

## Reproduce
- **In your notebook**: run your existing cells through 2.1.5, then run the `cov4.8_v1` section in
  `colab_code_2026_07_09.py`. It reuses `iteration_winners`, `df_metrics`, etc. already in memory.
- **Locally / headless**: edit `ROOT` at the top of `local_run_2026_07_09.py` and run it; it rebuilds
  this whole folder from `dataframe/`. (Parcel distance uses a fast Dijkstra graph‑distance locally in
  place of `tvb-gdist`; the Colab code path is unchanged and still consumes your original `df_distance`.)
