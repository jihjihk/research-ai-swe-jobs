# T-anchor — Anchor-neighborhood diffusion at four cosine thresholds

## What was run

- **Code:** `figures/bertopic/stage2/t_anchor.py` (single standalone script).
- **Inputs (hash-verified):** `data/bertopic/embeddings_cache.npy`,
  `data/bertopic/embeddings_cache.index.parquet`,
  `data/bertopic/assignments.parquet`,
  `data/bertopic/topic_info.parquet`,
  `figures/bertopic/intermediate/sample_a.parquet`,
  `figures/bertopic/intermediate/raw_fit.bertopic` (the file referenced as
  `model.bertopic` in `stage1_freeze.json`),
  `figures/bertopic/config.py`. All five SHA256 hashes match the values
  pinned in `intermediate/stage1_freeze.json`.
- **Anchors:** the five strings in `config.NEIGHBORHOOD_ANCHORS`
  (`ai_engineer`, `backend_engineer`, `frontend_engineer`,
  `legacy_specialist`, `sre`), retrieved from the cache via the
  index-key `neighborhood::<role>`. All anchor norms within
  `config.EMBEDDING_NORM_TOLERANCE = (0.99, 1.01)`.
- **Sample:** Sample A, 57,766 postings; 23,344 in 2024 and 34,422 in
  2026 (period derived from the year prefix of the period column —
  `2024-01`, `2024-04`, `2026-03`, `2026-04`).
- **Cosine** = dot product on unit-normalized 3072-d OpenAI
  `text-embedding-3-large` vectors.
- **Thresholds (pre-registered):** `{0.5, 0.6, 0.7, 0.8}`.
- **Bootstraps:** 5 × 80% draws per period per anchor; IQR of the
  resulting neighborhood-size distribution reported.
- **Threshold sensitivity:** ±0.05 perturbation reported as
  `size_threshold_plus_005` / `size_threshold_minus_005`.
- **Plot:** `figures.style.setup()` / `save()`; F10 is one panel per
  anchor, log y-axis, error bars = bootstrap IQR / 2.
- **Runtime:** 11.0 s end-to-end on the repo venv.

Outputs:

- `data/bertopic/anchor_neighborhoods.parquet` (40 rows: 5 anchors × 4
  thresholds × 2 periods; columns `anchor_id`, `anchor_string`,
  `period`, `threshold`, `neighborhood_size`, `period_total`,
  `neighborhood_share`, `top_clusters_dict` (JSON), `bootstrap_iqr_size`,
  `bootstrap_sizes` (JSON), `size_threshold_plus_005`,
  `size_threshold_minus_005`).
- `figures/output/fig_anchor_neighborhood.pdf` (+ `.png` preview).

## Results

### Cosine-similarity range — anchors are not deeply embedded in the SWE corpus

| Anchor | min cos | max cos | mean | std |
|---|---|---|---|---|
| `ai_engineer` | 0.133 | 0.554 | 0.322 | 0.042 |
| `backend_engineer` | 0.195 | 0.600 | 0.381 | 0.053 |
| `frontend_engineer` | 0.153 | 0.642 | 0.331 | 0.064 |
| `legacy_specialist` | 0.118 | 0.616 | 0.315 | 0.054 |
| `sre` | 0.160 | 0.596 | 0.337 | 0.050 |

**Across all 57,766 postings and all five anchors, no posting reaches
cos ≥ 0.65.** This is a property of the OpenAI 3072-d embedding for
short anchor sentences vs. multi-paragraph job descriptions; the
absolute scale is compressed compared to MiniLM expectations. The
pre-registered τ ∈ {0.7, 0.8} buckets are therefore empty in both
periods for every anchor.

### Neighborhood sizes (full table)

Period totals: 2024 = 23,344; 2026 = 34,422 postings. IQR is the
inter-quartile range over five 80% bootstraps.

| anchor | τ | n_2024 | share_2024 | n_2026 | share_2026 | Δshare (pp) | IQR_2024 | IQR_2026 |
|---|---|---|---|---|---|---|---|---|
| ai_engineer | 0.5 | 0 | 0.000% | 20 | 0.058% | +0.058 | 0 | 3 |
| ai_engineer | 0.6 | 0 | 0.000% | 0 | 0.000% | 0.000 | 0 | 0 |
| ai_engineer | 0.7 | 0 | 0.000% | 0 | 0.000% | 0.000 | 0 | 0 |
| ai_engineer | 0.8 | 0 | 0.000% | 0 | 0.000% | 0.000 | 0 | 0 |
| backend_engineer | 0.5 | 214 | 0.917% | 275 | 0.799% | −0.118 | 4 | 6 |
| backend_engineer | 0.6 | 0 | 0.000% | 1 | 0.003% | +0.003 | 0 | 0 |
| backend_engineer | 0.7 | 0 | 0.000% | 0 | 0.000% | 0.000 | 0 | 0 |
| backend_engineer | 0.8 | 0 | 0.000% | 0 | 0.000% | 0.000 | 0 | 0 |
| frontend_engineer | 0.5 | 410 | 1.756% | 641 | 1.862% | +0.106 | 3 | 12 |
| frontend_engineer | 0.6 | 4 | 0.017% | 10 | 0.029% | +0.012 | 1 | 1 |
| frontend_engineer | 0.7 | 0 | 0.000% | 0 | 0.000% | 0.000 | 0 | 0 |
| frontend_engineer | 0.8 | 0 | 0.000% | 0 | 0.000% | 0.000 | 0 | 0 |
| legacy_specialist | 0.5 | 136 | 0.583% | 60 | 0.174% | −0.408 | 4 | 6 |
| legacy_specialist | 0.6 | 3 | 0.013% | 3 | 0.009% | −0.004 | 1 | 1 |
| legacy_specialist | 0.7 | 0 | 0.000% | 0 | 0.000% | 0.000 | 0 | 0 |
| legacy_specialist | 0.8 | 0 | 0.000% | 0 | 0.000% | 0.000 | 0 | 0 |
| sre | 0.5 | 112 | 0.480% | 142 | 0.413% | −0.067 | 3 | 5 |
| sre | 0.6 | 0 | 0.000% | 0 | 0.000% | 0.000 | 0 | 0 |
| sre | 0.7 | 0 | 0.000% | 0 | 0.000% | 0.000 | 0 | 0 |
| sre | 0.8 | 0 | 0.000% | 0 | 0.000% | 0.000 | 0 | 0 |

### Threshold sensitivity (±0.05) at τ = 0.5

| anchor | period | n at τ=0.5 | n at τ=0.45 | n at τ=0.55 |
|---|---|---|---|---|
| ai_engineer | 2024 | 0 | 22 | 0 |
| ai_engineer | 2026 | 20 | 218 | 1 |
| backend_engineer | 2024 | 214 | 2,043 | 5 |
| backend_engineer | 2026 | 275 | 3,501 | 8 |
| frontend_engineer | 2024 | 410 | 1,126 | 88 |
| frontend_engineer | 2026 | 641 | 2,056 | 136 |
| legacy_specialist | 2024 | 136 | 352 | 60 |
| legacy_specialist | 2026 | 60 | 205 | 26 |
| sre | 2024 | 112 | 430 | 9 |
| sre | 2026 | 142 | 633 | 8 |

A 0.05 cosine perturbation moves neighborhood sizes by 1–2 orders of
magnitude in many cells. The structure-of-the-result is therefore
threshold-dependent, which is exactly the failure mode the §6.5 design
calls for cutting on.

### Composition at τ = 0.5 (the only bucket with non-trivial mass)

| anchor / period | top BERTopic clusters (share) |
|---|---|
| `ai_engineer` 2026 (n=20) | topic 0 *AI Software Engineering* 0.70; outlier 0.30 |
| `backend_engineer` 2024 (n=214) | outlier 0.43; topic 0 *AI SWE* 0.43; topic 1 *Test Automation* 0.09; topic 3 *Salesforce Cloud Dev* 0.05 |
| `backend_engineer` 2026 (n=275) | topic 0 *AI SWE* 0.47; outlier 0.36; topic 1 *Test Automation* 0.12; topic 3 *Salesforce Cloud Dev* 0.04 |
| `frontend_engineer` 2024 (n=410) | topic 4 *Full Stack Dev* 0.68; outlier 0.14; topic 0 *AI SWE* 0.12; topic 1 *Test Automation* 0.03 |
| `frontend_engineer` 2026 (n=641) | topic 4 *Full Stack Dev* 0.56; outlier 0.18; topic 0 *AI SWE* 0.18; topic 1 *Test Automation* 0.04 |
| `legacy_specialist` 2024 (n=136) | topic 7 *Application Systems Analyst* 0.88; outlier 0.08; topic 1 *Test Automation* 0.04 |
| `legacy_specialist` 2026 (n=60) | topic 7 *Application Systems Analyst* 0.82; outlier 0.10; topic 1 *Test Automation* 0.07 |
| `sre` 2024 (n=112) | topic 0 *AI SWE* 0.93; outlier 0.05; topic 1 *Test Automation* 0.02 |
| `sre` 2026 (n=142) | topic 0 *AI SWE* 0.88; outlier 0.07; topic 1 *Test Automation* 0.04 |

The composition itself is a flag: `sre` and (especially) `backend_engineer`
neighborhoods are dominated by topic 0 *AI Software Engineering*, which
suggests the K=10 mega-cluster at topic 0 is absorbing semantic mass that
the anchors are also pulled toward; the headline run already hit
`largest_cluster_share_headline = 0.261`.

### Sign / monotonicity by anchor (Gate-2 substrate)

For each anchor, the four pre-registered thresholds give the following
2024→2026 size deltas (positive = grew; 0 means both periods were 0 or
indistinguishable):

| anchor | Δsize at τ=0.5 | Δsize at τ=0.6 | Δsize at τ=0.7 | Δsize at τ=0.8 | sign set |
|---|---|---|---|---|---|
| ai_engineer | +20 | 0 | 0 | 0 | {0, +} |
| backend_engineer | +61 | +1 | 0 | 0 | {0, +} |
| frontend_engineer | +231 | +6 | 0 | 0 | {0, +} |
| legacy_specialist | −76 | 0 | 0 | 0 | {0, −} |
| sre | +30 | 0 | 0 | 0 | {0, +} |

No anchor satisfies the strict §6.5 criterion of "trend monotonic in
the same direction across all four cosine thresholds" because the
higher thresholds produce zero neighborhoods (the trend is the zero
function there, not a positive or negative trend). Even relaxing to
"non-zero buckets share the same sign," only `legacy_specialist`,
`backend_engineer`, and `frontend_engineer` have any change at τ = 0.6;
`ai_engineer` and `sre` have a single non-zero bucket each.

## Three-gate evaluation (per design.md §13.5)

### Gate 1 (Narrative)

**Conditional fail.** The §6.5 analysis is meant to test the
diffusion-vs-concentration cut for §1.4.3 ("AI vocabulary diffusion vs
concentration") and to corroborate C1 ("AI-native role families have
crystallized") via a monotonically growing AI-engineer neighborhood. At
τ = 0.5 the AI-engineer neighborhood does grow (0 → 20 postings, 0 → 0.06%
of the period), and the legacy-specialist neighborhood does shrink
(136 → 60, −0.41pp). Directionally consistent with C1 and C2. But the
absolute counts are tiny — twenty postings out of 34,422 — and three of
the four pre-registered thresholds collapse to zero in both periods.
A hostile reviewer can ask "what does this actually show?" and the
honest answer is "anchor sentences sit at cosine ~0.32 from the corpus
mean and the pre-registered thresholds are mis-scaled for this
embedding family." The narrative survives only if the anchor strings
or the threshold grid are recalibrated, which would be a paper-visible
erratum to `prereg_log.md`.

### Gate 2 (Effect size)

**Fail.** §13.5 requires "trend monotonic across all of {0.5, 0.6, 0.7,
0.8} cosine thresholds." For every anchor, τ ∈ {0.7, 0.8} is identically
zero in both periods, so the trend is undefined there. Treating zero
buckets as "no change" and asking for sign consistency among the
non-zero buckets:

- `ai_engineer`: only τ=0.5 has a non-zero delta. **Insufficient
  evidence of a monotonic trend** under the spec.
- `backend_engineer`: τ=0.5 grew +61, τ=0.6 grew +1. Non-zero buckets
  agree but at τ=0.6 the count is one posting; not robust to noise.
- `frontend_engineer`: τ=0.5 grew +231, τ=0.6 grew +6. Same caveat.
- `legacy_specialist`: τ=0.5 shrank −76, τ=0.6 flat (3=3). Only one
  non-zero non-flat bucket.
- `sre`: only τ=0.5 has a non-zero delta.

Strict reading of §6.5: **all five anchors fail the gate**. The
analysis is "threshold-dependent" exactly in the sense the design says
should cut.

### Gate 3 (Robustness)

Of the five robustness checks listed in §13.5:

- **Seed reshuffle.** N/A — cosine to a fixed anchor on cached
  embeddings is deterministic. Not applicable.
- **Anchor leave-one-out.** N/A by spec ("one anchor per role").
  Threshold-perturbation sensitivity reported instead: ±0.05 in cosine
  changes neighborhood sizes by 1–2 orders of magnitude in most cells
  (e.g. `backend_engineer` 2026 goes 8 → 275 → 3,501 across τ ∈ {0.55,
  0.50, 0.45}). **Fail under the threshold-perturbation analogue.**
- **Subset replication.** Five 80%-bootstraps per period per anchor.
  IQR ≤ 12 postings in every cell, ≤ 6 for all but `frontend_engineer`
  τ=0.5; bootstrap IQR is small relative to the period delta at τ=0.5
  for the anchors with directional movement (e.g. `frontend_engineer`
  2026 IQR=12 vs Δsize=+231; `legacy_specialist` IQR_2024=4 vs
  Δsize=−76). **Pass for τ=0.5; trivially "pass" elsewhere because
  every bootstrap returns 0 too.**
- **Permutation null.** Not run for this analysis (no permutation null
  is specified in §6.5; it is for §6.2 boundary postings).
- **Cross-embedding (MiniLM).** Out of scope for T-anchor; would
  require re-running the embedding cache on MiniLM, which is T-method's
  responsibility. **Not run.**

Summary: 1 of 5 checks (subset replication at τ=0.5 only) clearly
passes. Below the §13.5 minimum of "3 of 5."

## recommend_for_paper: no

## Rationale

The analysis ran cleanly on the frozen Stage 1 artifacts; all hashes
verified, the figure and parquet are saved at the spec-mandated paths,
and the ±0.05 threshold sensitivity and 5×80% bootstraps were both
computed and persisted. The findings, on their merits, do not support
inclusion: the anchor sentences sit at mean cosine 0.32 from Sample A
postings (max ≈ 0.55–0.64), so τ ∈ {0.7, 0.8} is empty in both periods
for every anchor and the §13.5 "monotonic across all four thresholds"
criterion is structurally unreachable. At the only non-trivial
threshold (τ = 0.5), four of five anchors move in the direction the
narrative predicts (AI/backend/frontend grow, legacy shrinks; SRE
roughly flat), but the absolute counts are 20–641 out of period
totals of 23k–34k, the τ=0.5 → τ=0.55 sensitivity collapses most
neighborhoods to single digits, and the BERTopic-cluster composition
of every neighborhood is dominated by topic 0 *AI Software
Engineering*, the headline-K mega-cluster — meaning the F10 panels are
partly reading the mega-cluster's gravity rather than five distinct
role-anchored neighborhoods. If the orchestrator wants a usable F10,
the paper-visible move is an erratum entry that recalibrates the
threshold grid (e.g. anchor-specific quantile thresholds, or a single
absolute threshold at the 99th-percentile of each anchor's similarity
distribution) and re-runs §6.5 on the recalibrated scale. As pre-registered, the result fails Gate 2 and clears only one of the
applicable Gate-3 checks.
