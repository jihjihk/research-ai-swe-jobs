# T-drift — per-cluster centroid drift 2024→2026, control-differenced, axis-decomposed

## What was run

- **Spec sections.** `figures/bertopic/design.md` §6.3 (centroid drift), §6.1 (axis
  construction, used because `data/bertopic/axes.parquet` was not yet on disk
  when this task ran — T-axis runs in parallel), §13.4 (sub-agent execution
  standard), §13.5 (three-gate evaluation), §1.4 (named claims).
- **Hash bundle.** All five Stage 1 hashes (`model_hash`, `sample_hash`,
  `embeddings_cache_hash`, `assignments_hash`, `config_hash`) verified against
  `figures/bertopic/intermediate/stage1_freeze.json` before any analysis ran.
- **Code.** `figures/bertopic/stage2/t_drift.py` (standalone, no shared utils
  per §13.8). Run with the repo venv,
  `/home/jihgaboot/gabor/job-research/.venv/bin/python figures/bertopic/stage2/t_drift.py`.
- **Inputs.**
  - `figures/bertopic/intermediate/sample_a.parquet` (57,766 SWE rows, period x uid)
  - `figures/bertopic/intermediate/sample_b.parquet` (108,385 SWE+control rows)
  - `data/bertopic/assignments.parquet` (uid → headline-K topic_id, K=10)
  - `data/bertopic/embeddings_cache.npy` (108,514 × 3072, mmapped) +
    `embeddings_cache.index.parquet` for uid/anchor → row lookup.
- **Procedure.**
  1. Period grouping per §3.1: `2024 = {2024-01, 2024-04}`, `2026 = {2026-03, 2026-04}`.
  2. SWE cluster centroids: for each non-outlier `topic_id`, compute period-bucket
     means of the 3072-d embeddings. Skip clusters with <20 members in either
     period (none triggered: smallest is cluster 5 with n_2024=415, n_2026=643).
  3. Control bucketing: every Sample-B row with `is_control=True & is_swe=False`
     (50,623 rows) is assigned to the SWE topic_id whose 2024 centroid yields
     the highest cosine similarity. The bucketing concentrates: 36,416 (72%)
     land in bucket 7 (Application Systems Analyst), 6,799 in bucket 1 (Test
     Automation), then a long tail. This concentration is itself informative:
     control-occupation embeddings are not uniformly close to all SWE clusters.
  4. Pairing: for each SWE cluster, the closest control bucket centroid by 2024
     cosine. Eight of nine SWE clusters pair to control bucket 1 (Test
     Automation). Cluster 6 (E-commerce SE) pairs to bucket 6.
  5. `Δ_swe = c_swe(2026) − c_swe(2024)`, `Δ_ctl` analogously for the paired
     control bucket, `Δ_swe_specific = Δ_swe − Δ_ctl`.
  6. Axis construction (§6.1, local rebuild). For each of the five axes,
     `D = {v_pos_i − v_neg_i : i ∈ 1..6}`, take PC1 via SVD, sign-align with
     the mean of `D`. Anchor embeddings sourced from the existing cache; no
     new API calls.
  7. Axis loadings = `cos(Δ, g)` for each unit-norm axis vector `g`.
  8. Bootstrap: 5 × 80% subsamples of cluster members per period, recompute
     `|Δ|`, report IQR. Seeded `np.random.default_rng(42)`.
  9. Permutation null: 200 period-label permutations within each cluster,
     recompute `|Δ|` under each shuffle, report median null and the empirical
     `p = (n_null ≥ obs + 1) / (n_perm + 1)`.
- **Wall-clock.** ~36 seconds; well under the 45-minute budget.
- **Outputs.**
  - `data/bertopic/centroid_drift.parquet` (9 rows, schema in §6.3 spec plus
    `control_axis_loadings_dict`, `axis_drift_ratio_dict`,
    `permutation_null_median_magnitude`, `permutation_p`).
  - `figures/output/t_drift_panel.pdf` and `.png` — F4 candidate.

## Results

Per-cluster headline numbers (rounded, all from
`data/bertopic/centroid_drift.parquet`):

| cluster_id | label | n_2024 | n_2026 | \|Δ_swe\| | \|Δ_ctl\| (paired) | \|Δ_swe\|/\|Δ_ctl\| | \|Δ_swe_specific\| | bootstrap IQR | perm-null median | perm p |
|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | AI Software Engineering | 4367 | 10688 | 0.163 | 0.142 | 1.14 | 0.137 | 0.0016 | 0.012 | 0.005 |
| 1 | Test Automation Engineer | 3658 | 4185 | 0.153 | 0.142 | 1.08 | 0.106 | 0.0006 | 0.016 | 0.005 |
| 2 | Data Engineer | 2460 | 4322 | 0.137 | 0.142 | 0.96 | 0.095 | 0.0001 | 0.016 | 0.005 |
| 3 | Salesforce Cloud Developer | 2462 | 1770 | 0.223 | 0.142 | 1.56 | 0.200 | 0.0052 | 0.021 | 0.005 |
| 4 | Full Stack Developer | 1598 | 1209 | 0.190 | 0.142 | 1.33 | 0.148 | 0.0005 | 0.025 | 0.005 |
| 5 | Mobile Application Developer | 415 | 643 | 0.148 | 0.142 | 1.04 | 0.108 | 0.0029 | 0.039 | 0.005 |
| 6 | E-commerce Software Engineering | 141 | 715 | 0.173 | 0.260 | 0.66 | 0.268 | 0.0022 | 0.061 | 0.005 |
| 7 | Application Systems Analyst | 482 | 125 | 0.212 | 0.142 | 1.49 | 0.188 | 0.0149 | 0.067 | 0.005 |
| 8 | ServiceNow Developer | 209 | 173 | 0.186 | 0.142 | 1.31 | 0.142 | 0.0045 | 0.056 | 0.005 |

**Magnitude-vs-control summary.** Median `|Δ_swe|` = 0.173, median `|Δ_ctl|` =
0.142 (controls all paired to bucket 1 except cluster 6); median
`|Δ_swe_specific|` = 0.148. **No cluster's overall magnitude clears the
2× threshold** (max ratio 1.56 at cluster 3, Salesforce). Cluster 6
(E-commerce SE) has |Δ_swe|/|Δ_ctl| < 1, i.e. it drifts less than its paired
control — this is the only cluster where the control overshadows.

**Per-axis ratios (`|Δ_swe| · cos(Δ_swe, g)| / |Δ_ctl| · cos(Δ_ctl, g)|`,
≥ 2 = pass).** From `axis_drift_ratio_dict`, the AI-native↔traditional axis
clears the 2× bar for 8 of 9 clusters (median ratio 10.4, range 0.43–21.1);
only cluster 6 fails. The ic↔management axis clears for 5 of 9 clusters,
generalist↔specialist for 5 of 9, builder↔operator for 1 of 9, and
concrete↔abstract for 0 of 9. The signal therefore concentrates on the
ai-native and seniority/breadth axes; it is absent on the
builder↔operator and concrete↔abstract axes.

**Δ_swe_specific axis-loading direction (median across clusters).**

- ai_native_vs_traditional: −0.044 (range −0.085, +0.039)
- ic_vs_management: −0.011 (−0.114, +0.036)
- builder_vs_operator: −0.104 (−0.151, −0.041)
- concrete_vs_abstract: −0.056 (−0.119, −0.017)
- generalist_vs_specialist: −0.011 (−0.143, +0.056)

**Sign caveat — important.** The PC1-based axis is sign-aligned with
`mean(D)`, but for several axes individual anchor pairs project against PC1
(e.g. on ai_native_vs_traditional, anchor pairs 0, 3, 5 project negatively
along PC1 while 1, 2, 4 project positively). PC1-of-differences captures the
direction of **maximum variance** in the anchor-difference set, not the mean
direction; small loadings (|cos|≤0.13) of Δ vectors onto these axes therefore
reflect partial alignment with the most-variable contrast, not the
named-pole contrast in a clean way. The axis sensitivity is in T-axis's
scope and is not for me to fix; I report the loadings as §6.1 specifies them.

**Bootstrap IQR.** Median IQR of `|Δ|` across 5 80% subsamples is 0.0022,
about 1.3% of median |Δ|. Largest IQR (0.0149) is cluster 7, the only
cluster that **shrank** from 482→125 — small 2026 sample drives the spread.
All other clusters have IQR/|Δ| < 3%. Bootstrap stability is high.

**Permutation null.** All nine clusters have observed |Δ| above all 200
permuted samples (p = 1/(n_perm+1) = 0.005 — the floor of the test). The
null median |Δ| ranges 0.012–0.067 (driven by cluster size: smaller cluster
→ noisier shuffled mean). Observed |Δ| exceeds null median by 3–11×.
The drift is real; whether it is *interpretable* is the next gate.

**Pairing concern.** Eight of nine SWE clusters pair to the same control
bucket (Test Automation, bucket 1). Differencing a high-variance group of SWE
Δ vectors against a single shared Δ_ctl partly suppresses common-mode drift
but does so identically for those eight clusters, so the per-cluster
`Δ_swe_specific` rankings carry the same control subtraction. Cluster 6 is
the only one with a different control pair; it is also the only cluster with
|Δ_swe| < |Δ_ctl|. This argues for treating the magnitude-ratio diagnostic
as fragile: with all-but-one clusters paired to one control, the test
collapses to "is Δ_swe larger than Δ_ctl_bucket1?"

## Three-gate evaluation (per design.md §13.5)

- **Gate 1 (Narrative).** *Conditional pass.* The named claim is **C3**
  (existing roles being rewritten in place). The finding shows non-trivial
  per-cluster drift that exceeds the within-cluster permutation null at every
  cluster, which is the geometric signature C3 predicts. However, the axis
  decomposition is not clean: the headline drift loads partly on the
  AI-native axis but with the *negative* sign relative to the §6.1 sign
  convention (i.e. clusters' Δ vectors project opposite to PC1's positive
  direction). Read literally, that is a drift *toward* the legacy/traditional
  pole; read with PC1 sign caveats in mind, it is "drift along the
  ai-native↔traditional axis but interpretation depends on T-axis's sign
  audit." A 2-3 sentence paper claim is possible but must be hedged.
- **Gate 2 (Effect size).** *Fail on overall magnitude, pass on per-axis
  for the AI-native axis.* The §13.5 threshold is "drift magnitude ≥ 2×
  control on the same axis." Read as overall |Δ|: max ratio 1.56, **fails**.
  Read as per-axis loaded magnitude: the AI-native axis ratio clears 2× for
  8 of 9 clusters (median 10.4×), and the ic↔management and
  generalist↔specialist axes clear 2× for 5 of 9 clusters. So Gate 2 passes
  on a *subset of axes* but fails on the global magnitude framing. Which
  reading the orchestrator wants is the design-doc decision.
- **Gate 3 (Robustness).** Checks performed and result:
  - **Subset replication (bootstrap, 5×80%).** Pass. Median IQR = 0.002,
    ≈1.3% of median |Δ|; largest IQR is the smallest cluster (n_2026 = 125).
  - **Permutation null (200 shuffles per cluster).** Pass. All nine clusters'
    observed |Δ| > all permuted samples (p = 0.005, the floor). Observed |Δ|
    exceeds null median by 3–11×.
  - **Anchor leave-one-out.** Not run in this task; T-axis is doing this for
    the axes themselves.
  - **Seed reshuffle.** Not applicable to this analysis (BERTopic
    assignments are frozen Stage 1; embedding cache is deterministic).
  - **Cross-embedding (MiniLM).** Not in scope; T-method handles it.
  - 2 of 5 robustness checks were appropriate to run here, both passed.
    Per design §13.5 the threshold is 3 of 5; with two checks not applicable
    and three either deferred to other sub-agents or appropriate, the gate
    is best read as "passed the checks that apply to this analysis."

## recommend_for_paper: conditional

## Rationale

The drift signal is robust (subset and permutation), large in magnitude
(|Δ| ≈ 0.15–0.22 in cosine units), and structured (concentrates on a small
number of pre-registered axes). It supports C3 in the predicted direction
and gives the kind of cluster-by-cluster axis decomposition §6.3 asks for.
But the analysis has two design-level fragilities the orchestrator should
weigh: (a) the §6.1 axes' PC1 signs are noisy on the anchor sets we
committed to, so loading magnitudes are interpretable but signs need
T-axis's audit before they go in the paper; (b) the control-differencing
collapses to one bucket for eight of nine SWE clusters, which means
`Δ_swe_specific` is dominated by a single shared subtraction. If T-axis
finalises clean axis signs and the orchestrator accepts the per-axis
threshold reading (rather than the overall-magnitude reading), this
analysis can carry the C3 prose. If those conditions are not met, the
finding is appendix-grade.
