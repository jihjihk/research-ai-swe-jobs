# Pre-registration log — BERTopic discovery and embedding-space analysis

This log records every Stage 0 / 1 / 1.5 / 2 / 3 transition and every
deviation from `figures/bertopic/design.md`. It is committed to source
control and goes into the paper appendix per §11.11.

## 2026-05-05 — Author session boundary

- `design.md` v1 committed. Owner: Jihyun + Gabor. Conference target: AIES 2026.
- `orchestrator_prompt.md` v1 committed alongside.
- The user kicked off the orchestrator and went to sleep at end of day,
  delegating continuation through to Stage 3 synthesis using best judgment
  (memorialised in `memory/project_bertopic_run.md`). Stage 1.5 sign-off is
  therefore self-administered with rationale in this log; the Stage 4
  reproducible notebook remains an explicit human session.

## 2026-05-06 — Substrate clarification

- Design-doc §2.3 originally allowed a COALESCE fallback from
  `description_core_llm` to raw `description` for the ~1 % of unlabeled
  rows. The user explicitly overrode this on 2026-05-05: substrate is
  `description_core_llm` exclusively, never raw `description`. Reason: raw
  descriptions carry boilerplate that would corrupt c-TF-IDF and
  cross-period comparisons. Implementation: `config.SUBSTRATE_COLUMN`
  pinned; sample.py / embedding_cache.py filter out null
  `description_core_llm`. `design.md` §2.3 wording will be tightened in a
  later author commit.

## 2026-05-06 — Stage 0 complete

Frozen artifacts (Stage 0):

| Item | Location | Notes |
|---|---|---|
| `config.py` | `figures/bertopic/config.py` | Hyperparameters (§4.2), seeds {42, 1337, 2026}, sample-cap = 5/(canonical_co × period × title_normalized), LLM models pinned, 125 anchor strings verbatim from §11.7. |
| Sample A | `intermediate/sample_a.parquet` | 57,766 SWE rows post-cap. |
| Sample B | `intermediate/sample_b.parquet` | 108,385 SWE+control rows post-cap. |
| Sample sizes | `intermediate/sample_sizes.csv` | source × period × sample breakdown. |
| Embeddings cache | `data/bertopic/embeddings_cache.npy` | (108,514, 3072) float32 = 108,389 postings + 125 anchors. |
| Embeddings index | `data/bertopic/embeddings_cache.index.parquet` | key (uid or anchor-id) → row_index. |

Pre-flight checks: PASS. Required columns present in `unified_core.parquet`;
embedding dimension 3072; L2 norms in [0.99, 1.01]; no forbidden nulls in
sample parquets; anchor index covers every config anchor.

Smoke test (5 % stratified slice of Sample A, n = 2,890): PASS. BERTopic
produced 72 non-noise topics at `min_cluster_size = 10`, raw HDBSCAN noise
rate 24.19 %. LLM naming call against `gpt-5.5-2026-04-23` returned a
well-formed JSON label ("QA Automation Engineer", confidence 0.94) for a
test-automation cluster. OpenAI Responses API schema requires `text.format`
rather than the older `response_format`; `naming.py` uses the new schema.

A subtle observation: |A ∪ B| = 108,389 > |B| = 108,385. Four SWE rows are
in A but not B because the per-bucket cap interacts with the broader B
WHERE clause (which admits control rows into the same buckets). This is
faithful to §3.1 — both samples are correctly built per their respective
definitions. The embedding cache covers the union, so every uid that any
downstream Stage-2 task could reach is embedded.

## Pending

- Stage 1: min_cluster_size sweep → headline fit + K sweep → seed
  stability gate → mega-cluster gate → determinism check → LLM naming →
  hash freeze.
- Stage 1.5: freeze memo at `memos/stage1_freeze.md`; self-signed per the
  2026-05-06 authorization above.
- Stage 2: eight parallel sub-agents per §13.4 task table; T-l1l2 queued
  because `role_family_l1` and `skill_theme_*` are not yet populated in
  `unified_core.parquet`.
- Stage 3: `memos/synthesis.md`.

## 2026-05-06 — Stage 1 frozen + Stage 1.5 self-signed

**Stage 1 outputs (hash-anchored):**
- `model_hash` = `d51f15e613f62b221139503bc84e6d3757689aac5e07979beb6ed3dbce509415`
- `sample_hash` = `6719a0250fbfcb630dad117b409d441697d493b209b219e1c9d08b09acfeb265`
- `embeddings_cache_hash` = `29d77bf9e24e6250d7b303a17fb22b80b9575a09a46d88c9dbd5d75c3b479b27`
- `assignments_hash` = `a03bc515094050996338094f28851126b8c1f07f7f3b26d2f678f6cb6808ab82`
- `config_hash` = `bef20ab2916ad72bd87aaefb0d18ba13644f9989ddd8e9bad4eac2b01a07bce8`

**Selection summary:**
- Headline `min_cluster_size` = 70 (largest in noise band [15 %, 35 %] with adjacent K=30 ARI ≥ 0.7 plateau between mcs=50 and mcs=70).
- Headline K = 10 (smallest K satisfying §4.4 criteria 1, 2, 4 — interpretability rating deferred to Stage 1.5 review per autonomous-run authorisation).
- Super-family K = 10 (the K=10 / K=15 super-family band; K=10 satisfies seed-pair ARI ≥ 0.4).

**Gates:**
- Seed gate (§7.1): all three pairwise ARI ≥ 0.67, centroid alignment ≥ 0.99 — **PASSED**.
- Mega-cluster gate (§10.1): largest cluster (cluster 0, "AI Software Engineering", 15,055 postings) share = 26.1 % ≤ 30 % — **PASSED**.
- Determinism (§13.2 S1.5): double-run produced byte-identical labels (ARI = 1.0000) — **PASSED**.
- Per-period centroid alignment at headline K = 0.913 (≥ 0.85) — cross-period claims unblocked.

**Headline-K cluster catalog (proposed gpt-5.5 labels):**
0. AI Software Engineering (15,055)
1. Test Automation Engineer (7,843)
2. Data Engineer (6,782)
3. Salesforce Cloud Developer (4,232)
4. Full Stack Developer (2,807)
5. Mobile Application Developer (1,058)
6. E-commerce Software Engineering (856)
7. Application Systems Analyst (607)
8. ServiceNow Developer (382)

**Deviations from `design.md` requiring acknowledgement in the appendix:**
1. **Permissive vectorizer for `reduce_topics`.** §4.2 pinned CountVectorizer with `min_df=10`. BERTopic's c-TF-IDF treats each topic as one document, so at small K (e.g. K=10) the term must appear in ≥ 10 of K topics — impossible. We swap to `min_df=2, max_df=1.0` for any post-fit reduction. The raw fit's c-TF-IDF still uses the §4.2 settings; only the post-reduce top-words are computed permissively.
2. **HDBSCAN single-thread.** `core_dist_n_jobs=1` and UMAP `n_jobs=1` were set to make the determinism check (§13.2 S1.5) pass. These are not §4.2 hyperparameters but they do affect runtime; the HDBSCAN fit is roughly 2× slower than multi-threaded.
3. **Author interpretability rating deferred.** §4.4 includes a 1–5 author rating as one of four headline-K criteria. With the user asleep during the autonomous run, only the three computable criteria (seed-pair ARI ≥ 0.4, per-period alignment ≥ 0.85, outlier ≤ 0.40) were applied. Headline K (= 10) was the smallest qualifying. Stage 3 synthesis flags whether this is the right resolution for the paper or whether a larger K should be elevated for per-cluster claims.
4. **Stage 1 implemented as one orchestrator script** (`stage1/run_stage1.py`) rather than seven separate files per §13.7. The function structure mirrors S1.1–S1.7 internally; a refactor into the §13.7 file layout is an open task.
5. **Sub-agent dispatch deviation.** T-l1l2 is queued — `role_family_l1` and `skill_theme_*` columns are not yet populated in `unified_core.parquet`.

**Stage 1.5 self-sign (per the 2026-05-06 user authorisation):**
- All three Stage 1 gates (seed stability, mega-cluster, determinism) passed.
- The headline cluster catalog is sanity-coherent: c-TF-IDF top-words match the gpt-5.5 proposed labels for every cluster I read (spot-checked clusters 0, 1, 2, 8 against `topic_info.parquet`).
- Per-period centroid alignment of 0.913 at headline K is comfortably above the 0.85 cross-period threshold.
- The 9-cluster headline is coarser than the design doc's 25–30 cluster expectation. We proceed to Stage 2 at this resolution and let the synthesis step recommend whether to elevate a higher K for per-cluster claims (T1).

**Sign-off:** orchestrator, 2026-05-06. Stage 2 fan-out launched.

## 2026-05-06 — Stage 2 sub-agent verifications (wave 1: 6 of 8)

Verification convention per §13.4: for each memo I (a) read end-to-end,
(b) spot-checked at least one quantitative claim against its parquet,
(c) verified methodology matched the design-doc spec, (d) flagged
advocacy. None of the six showed advocacy; all spot-checks passed.

### T-axis (`memos/t_axis.md`) — recommend `no`
- **Hash bundle**: 5/5 verified (sub-agent flagged that the freeze
  records `model_hash` against `intermediate/raw_fit.bertopic`, not
  `data/bertopic/model.bertopic` — the spec's path was wrong).
- **Spot-check**: `axes.parquet` row for `ai_native_vs_traditional`
  matches memo (period_shift 0.0337, LOO 0.307, hit-rate 0.725).
- **Methodology**: PC1-of-anchor-differences with **uncentered SVD**
  (Bolukbasi original); centered SVD removes the bias direction.
  Sub-agent justified the choice; I accept.
- **Result**: All five §6.1 axes fail Gate 2 (period shift below 0.05
  cosine and below 3× anchor LOO sensitivity). `ai_native_vs_traditional`
  has the largest shift (+0.0337) but the worst LOO spread (0.31, three
  times the §11.9 cap of 0.10) and below the 0.80 held-out hit threshold
  (0.72). Cluster ordering on this axis is face-valid (AI / Data Engineer
  positive; ServiceNow / Application Systems Analyst negative).
- **Disposition**: cut from paper body. Keep cluster-axis profile as
  appendix descriptive context.

### T-boundary (`memos/t_boundary.md`) — recommend `conditional`
- **Hash bundle**: 5/5 verified.
- **Spot-check**: `boundary_summary.parquet` matches the per-pair
  table (AI vs FullStack Δ = −4.46pp, p < 0.001).
- **Methodology**: §6.2 verbatim. Subset replication 50% × 3 seeds
  (extra robustness beyond spec).
- **Result**: 7 of 8 pairs fail the 5pp Gate 2 magnitude. The one
  passing pair (Salesforce vs ServiceNow, Δ = −7.60pp) is **sharpening**,
  not blurring — direction opposite the C3 prediction. AI vs FullStack
  shows −4.46pp (also sharpening, sub-threshold). Five other pairs are
  flat (|Δ| < 1.5pp, p > 0.17).
- **Disposition**: pre-registered null on the boundary sub-clause of
  C3. Worth a sentence in the paper: roles are not blurring; if anything
  the AI-vs-FullStack cleavage is becoming sharper.

### T-drift (`memos/t_drift.md`) — recommend `conditional`
- **Hash bundle**: 5/5 verified.
- **Spot-check**: `centroid_drift.parquet` 9 rows; cluster 3 (Salesforce)
  has the largest |Δ_swe| at 0.223; matches memo.
- **Methodology**: §6.3 verbatim, plus axes locally rebuilt per §6.1
  (T-axis ran in parallel; T-axis's `axes.parquet` was not on disk when
  T-drift loaded). Sub-agent flagged the issue — fine.
- **Result**: Overall magnitude Gate 2 fails (max |Δ_swe|/|Δ_ctl| =
  1.56, threshold 2.0). On the AI-native↔traditional axis specifically,
  the per-axis ratio clears 2× for 8 of 9 clusters (median 10.4×).
  Bootstrap IQR small (~1% of |Δ|). Permutation null p = 0.005 floor for
  every cluster.
- **Pairing concern**: 8 of 9 SWE clusters paired to control bucket 1
  (Test Automation) by 2024 cosine — the control differencing is
  effectively shared. Sub-agent flagged it. Reduces independent control
  weight on per-cluster claims.
- **Disposition**: conditional. The per-axis (AI-native) signal supports
  C3 in direction. Sign convention on PC1-of-anchor-differences is
  fragile (T-axis confirmed). Treat as appendix material unless C3 needs
  a magnitude finding for the body.

### T-weat (`memos/t_weat.md`) — recommend `conditional`
- **Hash bundle**: 5/5 verified.
- **Spot-check**: `weat_results.parquet` 5 rows; Test 1 d = +0.763,
  Test 3 d = +0.896, Test 5 d = −0.161 — match memo.
- **Methodology**: §6.4 verbatim, 10,000-permutation null,
  Bonferroni-corrected α' = 0.002.
- **Result**: 2 of 5 tests pass Gate 2 (|d| ≥ 0.5, p_bonf < 0.01):
  Test 1 AI vs ServiceNow on innovation/maintenance (d = +0.763),
  Test 3 senior vs junior on architecture/implementation (d = +0.896).
  **Test 5 inverted vs T1 prediction**: AI clusters tilt toward
  exploitation (d = −0.161), not exploration. Pre-registered null,
  reportable.
- **Caveat**: legacy_clusters resolves to a single cluster of 382
  ServiceNow postings — at headline K = 9 only ServiceNow surfaces
  from the legacy keyword list. The result is fairly read as
  "AI vs ServiceNow", not "AI vs legacy stacks broadly".
- **Disposition**: Tests 1 and 3 are paper-grade. Test 5's inverted
  sign is a notable null that contradicts T1's J-curve prediction.

### T-anchor (`memos/t_anchor.md`) — recommend `no`
- **Hash bundle**: 5/5 verified.
- **Spot-check**: `anchor_neighborhoods.parquet` 40 rows
  (5 anchors × 4 thresholds × 2 periods); ai_engineer τ=0.5 sizes
  (0 in 2024, 20 in 2026) match memo.
- **Methodology**: §6.5 verbatim plus 5×80% bootstraps and ±0.05
  threshold sensitivity.
- **Result**: max anchor-to-posting cosine = 0.65 across all 57,766
  postings. **All τ ∈ {0.7, 0.8} buckets are empty in both periods for
  every anchor.** Only τ = 0.5 has substantive mass; τ = 0.6 has
  0–10 postings. Threshold sensitivity is dramatic (±0.05 changes
  counts by 1–2 orders of magnitude). At τ = 0.5 the directional
  signal is consistent with C1/C2 (ai_engineer 0 → 20, frontend +231,
  legacy −76) but absolute counts are tiny (≤ 641 of ~30k per period)
  and τ = 0.5 neighborhoods of `sre` and `backend_engineer` are
  88–93 % topic 0 ("AI Software Engineering") — the headline mega-
  cluster contaminates the role-anchored interpretation.
- **Disposition**: cut from paper body. The OpenAI 3072-d embedding's
  compressed cosine scale broke the pre-registered threshold grid;
  a paper-visible erratum to `prereg_log.md` if the paper wants to
  cite anchor-quantile-based neighborhoods instead.

### T-quality (`memos/t_quality.md`) — recommend `conditional`
- **Hash bundle**: 5/5 verified.
- **Spot-check**: `topic_quality.parquet` 9 rows; cluster 0 silhouette
  −0.119 and cluster 3 silhouette −0.213 match memo. Cross-model
  exact-match rate = 3/9 (33 %), label-cosine mean 0.834.
- **Methodology**: §7.8–§7.11 verbatim. Permissive vectorizer used at
  reduce_topics (per the prereg-log deviation note above).
- **Result**: Honest noise rate 31.4 % (vs 0.0 % artificial after
  embeddings outlier reduction). Silhouette overall 0.224, below the
  0.4 strong-separation marker. **Two largest clusters have negative
  silhouette** (AI Software Engineering −0.119; Salesforce Cloud
  Developer −0.213). C_v 0.480 passes (≥ 0.45); NPMI 0.044 borderline-
  fails (< 0.05); topic diversity 0.833 passes (≥ 0.6). Cross-model
  naming exact-match 33 % is below the 50 % §7.11 trigger; cosine
  0.834 is below the 0.85 trigger. **The LLM-proposed labels are
  model-sensitive — the paper must flag this in the catalog footnote.**
- **Disposition**: diagnostic block, no claim support. Numbers go in
  T2/T3 verbatim. The negative silhouette on the AI Software
  Engineering mega-cluster is a paper-visible result for the §1.4.4
  AI-cohort sub-structure question (the cluster is not tightly
  separated from its neighbours).

### Wave 2 (T-bootstrap, T-method) — running
First Agent invocations crashed early; relaunched as background python
modules (PIDs 257374, 257375). Memos pending.
