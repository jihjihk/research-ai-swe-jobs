# Stage 2 per-task specifications

Per-task spec strings the orchestrator drops into the prompt template at
dispatch time. Keep these tight: each sub-agent will read the design doc
sections cited, so we do not duplicate the spec here.

---

## T-axis — §6.1 semantic axis projection (45 min budget)

**Sections to read in design.md:** §6.1, §6.6, §11.7, §11.9, §13.5.

**Spec.** Implement the five pre-registered semantic axes from §11.7
(`AXIS_ANCHORS` in `config.py`). For each axis:
1. Build the axis vector by taking 6 anchor differences (positive[i] −
   negative[i]) and using PC1 of the difference set per §6.1.
2. Project every Sample A posting onto each axis (cosine).
3. Compute period-mean shift (2026 mean − 2024 mean) per axis with a
   permutation null (1,000 shuffles of period labels).
4. **Anchor leave-one-out sensitivity** per axis: drop each pole anchor
   one at a time, refit the axis, report the cosine spread of axis
   identity. Pass per §11.9 if spread ≤ 0.10.
5. **Held-out anchor validation** per axis: hold out 30 % of anchors
   (round per pole), refit on 70 %, project the held-out positives /
   negatives. Report hit rate per §11.9 (≥ 80 %).
6. **Permutation null on axis effect:** project on 1,000 random unit
   directions; report the percentile of the actual axis effect size.
7. Per-cluster axis profile (cluster mean ± IQR on each axis).

**Outputs.**
- `data/bertopic/axes.parquet` — per axis: id, name, anchor_ids,
  axis_vector_path (a sidecar npy file path), leave_one_out_spread,
  held_out_hit_rate, permutation_null_percentile, period_shift,
  permutation_null_p.
- `data/bertopic/axis_projections.parquet` — (uid, axis_id, projection).
- `data/bertopic/cluster_axis_profile.parquet` — (cluster_id, axis_id,
  mean, q25, q75).
- F9 candidate figure at `figures/output/fig_axis_projection.pdf` plus its
  png preview, via `figures.style.setup()` and `save()`. Five small
  multiples, distribution by period.
- Code at `figures/bertopic/stage2/t_axis.py`.

**Gate 2 threshold (Effect size).** Period-mean shift ≥ 0.05 cosine units
AND ≥ 3× the leave-one-out anchor sensitivity per axis.

**Memo path:** `figures/bertopic/memos/t_axis.md`.

---

## T-boundary — §6.2 cluster-difference vectors and boundary postings (30 min)

**Sections to read:** §6.2, §13.5.

**Spec.**
1. Identify cluster pairs of substantive interest from the headline-K
   topic catalog. Use `topic_info.parquet` labels: pick at minimum the
   pairs (AI-flavored cluster vs Backend-flavored), (AI-flavored vs
   Data-Scientist-flavored), (DevOps vs SRE), and any other pair where
   labels suggest semantic adjacency. List 5–8 pairs.
2. For each pair (A, B), compute the centroid-difference vector
   `δ_AB = v_A − v_B` (in posting-embedding space) and L2-normalize it.
3. Project every posting that belongs to A or B onto `δ_AB`. Boundary
   postings = `|projection| < 0.05`.
4. Compute boundary fraction per period (2024 / 2026) per pair; report
   the change.
5. Permutation null on the change: shuffle period labels 1,000 times and
   recompute Δ-boundary-fraction.

**Outputs.**
- `data/bertopic/boundary_postings.parquet` — (uid, cluster_pair,
  projection, period).
- Per-pair summary table.
- Code at `figures/bertopic/stage2/t_boundary.py`.

**Gate 2 threshold.** Boundary-fraction change ≥ 5pp AND permutation
p < 0.05.

**Memo path:** `figures/bertopic/memos/t_boundary.md`.

---

## T-drift — §6.3 centroid drift over time (45 min)

**Sections to read:** §6.3, §6.1, §13.5.

**Spec.**
1. For each headline-K cluster, compute `Δ = v_2026 − v_2024` where
   v_period = mean of cluster members' embeddings in that period.
2. Compute the analogous drift for control occupations from Sample B's
   `is_control = TRUE` rows. Pair SWE clusters with the most similar
   control occupation by centroid cosine in 2024 (use that pairing for
   differencing).
3. SWE-specific drift component = `Δ_swe − Δ_control` (vector form).
4. Project both `Δ_swe` and `Δ_swe_specific` onto the five §6.1 axes.
   Report per-cluster axis loadings.
5. Subset replication: bootstrap 80 % of cluster members per period and
   recompute Δ; report the spread of `|Δ|` across 5 bootstraps per
   cluster.

**Outputs.**
- `data/bertopic/centroid_drift.parquet` — (cluster_id, drift_magnitude,
  drift_swe_specific_magnitude, axis_loadings_dict, control_pair_id).
- Code at `figures/bertopic/stage2/t_drift.py`.

**Gate 2 threshold.** Drift magnitude ≥ 2× the control-occupation drift
on the same axis. Drift dominated by control-shared signal → cut.

**Memo path:** `figures/bertopic/memos/t_drift.md`.

---

## T-weat — §6.4 WEAT-style association tests (30 min)

**Sections to read:** §6.4, §11.7 WEAT_ATTRIBUTES, §11.7 WEAT_TESTS,
§13.5.

**Spec.** Run the five pre-registered WEAT tests from
`config.WEAT_TESTS`. For each test:
1. Build target sets X, Y from the labelled Stage 1 catalog
   (`config.WEAT_TESTS` documents the cluster-or-period semantics for
   each target name; resolve to actual uids).
2. Build attribute sets A, B from `config.WEAT_ATTRIBUTES` — average the
   six anchor embeddings in each set.
3. Compute differential cosine s(t, A, B) for each posting in X, Y;
   compute Cohen's d of the X-vs-Y differential.
4. Permutation null: 10,000 random splits of (X ∪ Y) into same-sized
   halves; compute Cohen's d for each.
5. Bonferroni-correct across the five tests (α' = 0.01 / 5 = 0.002).

**Outputs.**
- `data/bertopic/weat_results.parquet` — (test_name, X, Y, A, B, n_X,
  n_Y, cohens_d, p_value, p_bonf).
- Code at `figures/bertopic/stage2/t_weat.py`.

**Gate 2 threshold.** Cohen's d ≥ 0.5 AND Bonferroni-corrected p < 0.01
across all reported tests. Tests not clearing the bar → reported as null
in the paper, not cut entirely.

**Memo path:** `figures/bertopic/memos/t_weat.md`.

---

## T-anchor — §6.5 anchor-neighborhood diffusion (30 min)

**Sections to read:** §6.5, §13.5.

**Spec.** For each of the five §11.7 NEIGHBORHOOD_ANCHORS:
1. Pull the anchor embedding from the cache (key
   `neighborhood::<role>`).
2. Compute cosine similarity to every Sample A posting.
3. For each cosine threshold in `config.ANCHOR_NEIGHBORHOOD_THRESHOLDS`
   (= {0.5, 0.6, 0.7, 0.8}) and each period (2024 / 2026), compute:
   - Neighborhood size (count of postings ≥ threshold).
   - Composition: top-5 BERTopic clusters by share of neighborhood
     members.
4. Subset replication: re-compute neighborhood sizes on 80 %
   bootstraps of the period, 5 bootstraps per period.
5. Anchor LOO sensitivity does not apply (one anchor per role); report
   instead the sensitivity to a 0.05 perturbation in cosine threshold
   per anchor.

**Outputs.**
- `data/bertopic/anchor_neighborhoods.parquet` — (anchor_id, period,
  threshold, neighborhood_size, top_clusters_dict, bootstrap_iqr_size).
- F10 candidate figure at `figures/output/fig_anchor_neighborhood.pdf`
  via `figures.style`. Line chart, neighborhood size by period at four
  thresholds, one panel per anchor.
- Code at `figures/bertopic/stage2/t_anchor.py`.

**Gate 2 threshold.** Trend monotonic in the same direction across all
four cosine thresholds. Threshold-dependent results → cut.

**Memo path:** `figures/bertopic/memos/t_anchor.md`.

---

## T-bootstrap — §7.2 + §7.3 + §7.6 stability suite (90 min)

**Sections to read:** §7.2, §7.3, §7.6, §13.5.

**Spec.**
1. **§7.2 Bootstrap.** Three bootstrap samples of Sample A at 80 %
   without replacement, stratified by (period, source). For each, refit
   BERTopic at headline (mcs, seed=42); reduce to headline K. ARI vs
   the headline fit on the overlapping rows.
2. **§7.3 Per-period reproduction.** Refit BERTopic on 2024-only and
   2026-only subsets at headline (mcs, seed=42); reduce each to
   headline K. For every joint cluster, find the nearest period-fit
   cluster by centroid cosine (Hungarian); report mean and median.
   *The Stage 1 K-sweep already produced these fits — load them from
   `intermediate/period_fits/` rather than refitting.*
3. **§7.6 Within-2024 cross-source placebo.** Refit BERTopic separately
   on asaniczka-2024 SWE rows and arshkon-2024 SWE rows at headline
   (mcs, seed=42); reduce each to headline K. Centroid alignment via
   Hungarian.

**Outputs.**
- `data/bertopic/stability.parquet` — one row per pair / bootstrap with
  ARI, NMI, centroid alignment.
- Code at `figures/bertopic/stage2/t_bootstrap.py`.

**Gate 2 threshold (per design.md §11.9).** Per-period centroid cosine
≥ 0.85; within-2024 centroid cosine ≥ 0.85; bootstrap ARI ≥ 0.4. Per-
period and within-2024 are gates on cross-period claims; bootstrap is
informative.

**Memo path:** `figures/bertopic/memos/t_bootstrap.md`.

---

## T-method — §7.4 NMF baseline + §7.5 MiniLM cross-embedding (60 min)

**Sections to read:** §7.4, §7.5, §13.5.

**Spec.**
1. **§7.4 NMF baseline.** TF-IDF vectorize `description_core_llm` on
   Sample A using the same vectorizer settings as `config` (ngram
   (1,3), min_df 10, max_df 0.4, custom stopwords). Fit NMF with
   `n_components = headline K` (random_state = 42). Hard-assign via
   argmax. Compute ARI / NMI vs the BERTopic headline-K assignments.
2. **§7.5 MiniLM cross-embedding.** Encode Sample A docs with
   `all-MiniLM-L6-v2` (384-d). Refit BERTopic at headline (mcs,
   seed=42) on those embeddings. Reduce to headline K. ARI / NMI vs
   the OpenAI fit.
3. Spot-check 5 clusters where MiniLM and OpenAI agree on most
   members; spot-check 5 where they disagree; compare vocabulary.

**Outputs.**
- T3 row entries appended to a stability/comparison parquet
  (`data/bertopic/method_comparison.parquet`).
- Code at `figures/bertopic/stage2/t_method.py`.

**Gate decision (§7.5).** Strong agreement ARI ≥ 0.5 → cluster
structure is not embedding-specific. Weak ≤ 0.3 → name embedding in
every claim.

**Memo path:** `figures/bertopic/memos/t_method.md`.

---

## T-quality — §7.8 + §7.9 + §7.10 + §7.11 quality block (45 min)

**Sections to read:** §7.8, §7.9, §7.10, §7.11, §13.5.

**Spec.**
1. **§7.8 Coherence + diversity.** Compute NPMI, UMass, C_v on
   `description_core_llm` Sample A as reference; topic top-10 terms
   per cluster. Use `gensim.models.CoherenceModel(coherence='c_v')`.
   Topic diversity = unique tokens / total across all clusters' top-10.
2. **§7.9 Silhouette + cluster size.** Cluster-size distribution
   (median, IQR, p5/p95) at headline K. Silhouette score in 5-D UMAP
   space (re-run UMAP at the headline (mcs, seed=42) — or load the
   raw fit and pull the UMAP output if BERTopic exposes it).
3. **§7.10 Honest noise rate.** Report HDBSCAN noise rate before and
   after `reduce_outliers(strategy='embeddings')`.
4. **§7.11 Cross-model naming.** Re-run §5.1 LLM naming with
   `gpt-5.4-mini` against the `gpt-5.5` primary labels in
   `topic_info.parquet`. Compute exact-match rate and
   label-embedding cosine (embed labels via text-embedding-3-large
   per-call; you may batch). Report distribution.

**Outputs.**
- `data/bertopic/topic_quality.parquet` — (topic_id, n_members,
  npmi, umass, c_v, silhouette).
- Append to `data/bertopic/topic_info.parquet` with
  `gpt54mini_label`, `gpt54mini_confidence`, `label_cosine` columns
  (write a parallel file `topic_info_with_naming.parquet`).
- Code at `figures/bertopic/stage2/t_quality.py`.

**Memo path:** `figures/bertopic/memos/t_quality.md`.

---

## T-ablations — §8.2 secondary ablations (3 hr; runs after the others)

**Sections to read:** §8.2, §9.5, §13.5.

**Spec.** Each ablation: re-run from BERTopic fit through headline K
labels, compute ARI vs headline + qualitative description. Use the
T-bootstrap / T-method / T-quality outputs as the "headline" reference.

Ablations to run (per §8.2 table):
1. Embedding model: MiniLM-L6 (loaded from T-method), `text-embedding-
   3-small` (3072→1536; if not available locally, skip with note),
   jobBERT-v2 (skip if not on Hugging Face cache).
2. UMAP n_components: 5 (headline), 10, 15.
3. UMAP n_neighbors: 15 (headline), 30, 50.
4. Sample cap: 5/(co × period × title) (headline), 3, 10, legacy
   30/(co × period), uncapped.
5. Aggregator: include (headline) vs exclude.
6. Substrate: `description_core_llm` (headline) vs raw `description`
   — but **the user has explicitly forbidden raw description**; SKIP
   this ablation and document the skip in the memo.
7. Length floor: 200 (headline), 100, 400.
8. Outlier reduction: off (headline), on (`distributions` strategy).

For each ablation, compute the T6 sign-consistency for primary claims
C1–C4 (§1.4.1). For C1 / C2 you need to identify the AI-flavored and
legacy-stack clusters in each ablation by c-TF-IDF top-words; document
the rule. For C4 use entropy of share distribution at the headline K.

**Outputs.**
- `data/bertopic/ablations.parquet` — one row per ablation: name,
  variant, n_topics, ari_vs_headline, mean_centroid_alignment,
  noise_rate, c1_holds, c2_holds, c3_holds, c4_holds.
- T6 sign-consistency matrix as `data/bertopic/t6_robustness.parquet`.
- Code at `figures/bertopic/stage2/t_ablations.py`.

**Memo path:** `figures/bertopic/memos/t_ablations.md`.
