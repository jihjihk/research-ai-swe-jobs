"""
Compose Stage 2 sub-agent prompts.

`compose_prompt(task_id)` reads the per-task spec and the frozen Stage 1
hash bundle, then returns the self-contained prompt the orchestrator
hands to the Agent tool. Run as a script to print prompts to stdout for
copy/paste into Agent calls.

Usage:
    .venv/bin/python -m figures.bertopic.stage2.dispatch t_axis
    .venv/bin/python -m figures.bertopic.stage2.dispatch --all
"""

from __future__ import annotations

import json
import sys

from figures.bertopic import config


_TASKS: dict[str, dict] = {
    "t_axis": {
        "task_id": "T-axis",
        "sections": "§6.1, §6.6, §11.7, §11.9, §13.4, §13.5",
        "time_budget": "45 minutes",
        "spec": """\
Implement the five pre-registered semantic axes from `config.AXIS_ANCHORS`
(§11.7). For each axis:

1. Build the axis vector by taking 6 anchor differences (positive[i] −
   negative[i]) per pole-pair and using PC1 of the difference set
   (§6.1). L2-normalize the axis vector.
2. Project every Sample A posting onto each axis (cosine).
3. Compute period-mean shift (2026 mean − 2024 mean) per axis with a
   permutation null (1,000 shuffles of period labels).
4. Anchor leave-one-out sensitivity per axis: drop each pole anchor one
   at a time, refit the axis, report the cosine spread of axis identity
   (max abs cosine difference vs the full-anchor axis).
5. Held-out anchor validation per axis: hold out 30 % of anchors per
   pole (round to 2 anchors), refit on the rest, project the held-out
   anchors. Report the hit rate (held-out positives should land
   positive on the axis, held-out negatives negative).
6. Permutation null on axis effect: project on 1,000 random unit
   directions; report the percentile of the actual axis effect size.
7. Per-cluster axis profile: cluster mean ± IQR on each axis from
   `assignments.parquet`'s topic_id column.

Outputs:
- `data/bertopic/axes.parquet` — per axis: id, name, anchor_ids,
  axis_vector_path, leave_one_out_spread, held_out_hit_rate,
  permutation_null_percentile, period_shift, permutation_null_p.
- `data/bertopic/axis_projections.parquet` — (uid, axis_id, projection).
- `data/bertopic/cluster_axis_profile.parquet` — (cluster_id, axis_id,
  mean, q25, q75).
- F9 candidate at `figures/output/fig_axis_projection.pdf` via
  `figures.style.setup()` and `save()`. Five small multiples, one per
  axis: distribution of posting scores by period (2024 vs 2026 overlay).
- Code at `figures/bertopic/stage2/t_axis.py`.

Gate 2 threshold: period-mean shift ≥ 0.05 cosine units AND
≥ 3 × the leave-one-out anchor sensitivity per axis.

Memo path: `figures/bertopic/memos/t_axis.md`.
""",
    },
    "t_boundary": {
        "task_id": "T-boundary",
        "sections": "§6.2, §13.4, §13.5",
        "time_budget": "30 minutes",
        "spec": """\
1. Identify cluster pairs of substantive interest from the headline-K
   topic catalog. Read `topic_info.parquet` and pick at minimum these
   pairs (use cluster ids from your read; map by label):
   - the AI-flavored cluster vs Backend-flavored
   - the AI-flavored cluster vs Data-Scientist-flavored (if present)
   - DevOps vs SRE if both present
   - any other adjacent pair where labels suggest semantic neighborhood.
   List 5–8 pairs total.
2. For each pair (A, B), compute the centroid-difference vector
   `δ_AB = v_A − v_B` (means in 3072-d posting space) and L2-normalize.
3. Project every posting that belongs to A or B onto `δ_AB`. Boundary
   postings = `|projection| < 0.05` (cosine units).
4. Compute boundary fraction per period (2024 / 2026) per pair; report
   the change.
5. Permutation null on the change: shuffle period labels 1,000 times
   and recompute Δ-boundary-fraction.

Outputs:
- `data/bertopic/boundary_postings.parquet` — (uid, cluster_pair,
  projection, period).
- A summary parquet `data/bertopic/boundary_summary.parquet` —
  (cluster_pair, n_2024, n_2026, boundary_frac_2024, boundary_frac_2026,
  delta, permutation_p).
- Code at `figures/bertopic/stage2/t_boundary.py`.

Gate 2 threshold: boundary-fraction change ≥ 5pp AND permutation
p < 0.05.

Memo path: `figures/bertopic/memos/t_boundary.md`.
""",
    },
    "t_drift": {
        "task_id": "T-drift",
        "sections": "§6.3, §6.1, §13.4, §13.5",
        "time_budget": "45 minutes",
        "spec": """\
1. For each headline-K cluster, compute `Δ = v_2026 − v_2024` where
   v_period = mean of cluster members' embeddings in that period.
   Skip clusters with < 20 members in either period (insufficient).
2. Compute the analogous drift for control occupations from Sample B's
   `is_control = TRUE` rows. Cluster control postings into broad
   occupation buckets via their nearest BERTopic centroid (use the
   existing model — call `transform()` if needed, or assign by argmax
   cosine in 3072-d).
3. Pair each SWE cluster with the most similar control occupation by
   centroid cosine in 2024; use that pairing for differencing. Compute
   `Δ_swe_specific = Δ_swe − Δ_control`.
4. Project both `Δ_swe` and `Δ_swe_specific` onto the five §6.1 axes
   (load `axes.parquet` from T-axis if it exists; otherwise rebuild
   axes per §6.1 — the T-axis sub-agent runs in parallel so artifacts
   may not be ready when you start).
5. Subset replication: bootstrap 80 % of cluster members per period
   and recompute Δ; report the spread of `|Δ|` across 5 bootstraps
   per cluster (IQR).

Outputs:
- `data/bertopic/centroid_drift.parquet` — (cluster_id, n_2024, n_2026,
  drift_magnitude, drift_swe_specific_magnitude, axis_loadings_dict,
  axis_loadings_swe_specific_dict, control_pair_id, bootstrap_iqr).
- Code at `figures/bertopic/stage2/t_drift.py`.

Gate 2 threshold: drift magnitude ≥ 2× the control-occupation drift
on the same axis. Drift dominated by control-shared signal → cut.

Memo path: `figures/bertopic/memos/t_drift.md`.
""",
    },
    "t_weat": {
        "task_id": "T-weat",
        "sections": "§6.4, §11.7 (WEAT_*), §13.4, §13.5",
        "time_budget": "30 minutes",
        "spec": """\
Run the five pre-registered WEAT tests from `config.WEAT_TESTS`. For
each test:

1. Build target sets X, Y from the labelled Stage 1 catalog. Resolve
   each WEAT_TESTS target name as follows:
   - `ai_clusters`: clusters whose c-TF-IDF top-words contain any of
     {llm, ai, ml engineer, machine learning, rag, agent, vector,
     foundation model, generative}.
   - `legacy_clusters`: clusters whose top-words contain any of
     {.net, cobol, mainframe, php, wordpress, autosar, servicenow,
     plc, mainframe, fortran}.
   - `non_ai_clusters`: complement of `ai_clusters`.
   - `period_2024` / `period_2026`: postings filtered by period.
   - `senior_swe` / `junior_swe`: postings filtered by `seniority_final`.
   Be explicit in the memo about which clusters resolved to each set.
2. Build attribute sets A, B from `config.WEAT_ATTRIBUTES` — average
   the six anchor embeddings per set (post-L2-normalize the average).
3. Compute differential cosine s(t, A, B) = mean cos(t, A) − mean
   cos(t, B) for each posting t in X, Y. Cohen's d of the X-vs-Y
   differential.
4. Permutation null: 10,000 random splits of (X ∪ Y) into same-sized
   halves; compute Cohen's d for each. Two-sided p.
5. Bonferroni-correct across the five tests (α' = 0.01 / 5 = 0.002).

Outputs:
- `data/bertopic/weat_results.parquet` — one row per test:
  test_name, X_definition, Y_definition, A_set, B_set, n_X, n_Y,
  cohens_d, p_value, p_bonf.
- Code at `figures/bertopic/stage2/t_weat.py`.

Gate 2 threshold: Cohen's d ≥ 0.5 AND Bonferroni-corrected p < 0.01.
Tests not clearing the bar are reported as null in the paper, not
cut entirely.

Memo path: `figures/bertopic/memos/t_weat.md`.
""",
    },
    "t_anchor": {
        "task_id": "T-anchor",
        "sections": "§6.5, §13.4, §13.5",
        "time_budget": "30 minutes",
        "spec": """\
For each of the five `config.NEIGHBORHOOD_ANCHORS`:

1. Pull the anchor embedding from the cache (key
   `neighborhood::<role>`).
2. Compute cosine similarity to every Sample A posting.
3. For each cosine threshold in
   `config.ANCHOR_NEIGHBORHOOD_THRESHOLDS` (= {0.5, 0.6, 0.7, 0.8})
   and each period (2024 / 2026), compute:
   - Neighborhood size (count of postings with cos ≥ threshold).
   - Composition: top-5 BERTopic clusters by share of neighborhood
     members.
4. Subset replication: re-compute neighborhood sizes on 80 %
   bootstraps of the period, 5 bootstraps per period; report IQR of
   size.
5. Anchor LOO does not apply (one anchor per role); instead report
   sensitivity to a 0.05 perturbation in cosine threshold per anchor.
6. F10 candidate figure: line chart, neighborhood size by period at
   four thresholds, one panel per anchor. Use `figures.style.setup()`
   and `save()`.

Outputs:
- `data/bertopic/anchor_neighborhoods.parquet` — (anchor_id, period,
  threshold, neighborhood_size, top_clusters_dict, bootstrap_iqr_size).
- F10 at `figures/output/fig_anchor_neighborhood.pdf`.
- Code at `figures/bertopic/stage2/t_anchor.py`.

Gate 2 threshold: trend monotonic in the same direction across all
four cosine thresholds. Threshold-dependent results → cut.

Memo path: `figures/bertopic/memos/t_anchor.md`.
""",
    },
    "t_bootstrap": {
        "task_id": "T-bootstrap",
        "sections": "§7.2, §7.3, §7.6, §13.4, §13.5",
        "time_budget": "90 minutes",
        "spec": """\
1. §7.2 Bootstrap. Three bootstrap samples of Sample A at 80 % without
   replacement, stratified by (period, source). For each, refit
   BERTopic at headline (mcs, seed=42) using
   `figures.bertopic.stage1.pipeline.fit_topic_model`; reduce to
   headline K. ARI vs the headline fit on the overlapping rows.
2. §7.3 Per-period reproduction. The Stage 1 K-sweep already produced
   period fits at headline mcs; load them from
   `intermediate/period_fits/period_2024_mcs_<X>.bertopic` and
   `period_2026_mcs_<X>.bertopic`. Reduce each to headline K. For each
   joint-Sample-A cluster centroid, find the nearest period-fit cluster
   centroid by cosine via Hungarian matching; report mean and median
   matched cosine.
3. §7.6 Within-2024 cross-source placebo. Fit BERTopic separately on
   asaniczka-2024 SWE rows (n ≈ 19k post-cap) and arshkon-2024 SWE
   rows (n ≈ 5k post-cap) at headline (mcs, seed=42). Reduce each to
   headline K. Centroid alignment via Hungarian.

Outputs:
- `data/bertopic/stability.parquet` — one row per pair / bootstrap:
  pair_kind ('bootstrap', 'per_period', 'within_2024'), pair_label,
  ari, nmi, centroid_alignment_mean, centroid_alignment_median.
- Code at `figures/bertopic/stage2/t_bootstrap.py`.

Gate 2 thresholds (per design.md §11.9): per-period centroid cosine
≥ 0.85; within-2024 centroid cosine ≥ 0.85; bootstrap ARI ≥ 0.4.
Per-period and within-2024 are gates on cross-period claims;
bootstrap is informative.

Memo path: `figures/bertopic/memos/t_bootstrap.md`.
""",
    },
    "t_method": {
        "task_id": "T-method",
        "sections": "§7.4, §7.5, §13.4, §13.5",
        "time_budget": "60 minutes",
        "spec": """\
1. §7.4 NMF baseline. TF-IDF vectorize `description_core_llm` on
   Sample A using the same vectorizer settings as `config` (ngram
   (1,3), `min_df=10`, `max_df=0.4`, custom stopwords). Fit NMF with
   `n_components = headline K` and `random_state = 42`. Hard-assign
   via argmax. Compute ARI / NMI vs the BERTopic headline-K
   assignments from `assignments.parquet`.
2. §7.5 MiniLM cross-embedding. Encode Sample A docs with
   `all-MiniLM-L6-v2` (sentence-transformers, 384-d). Refit BERTopic
   at headline (mcs, seed=42) on those embeddings. Reduce to headline
   K. ARI / NMI vs the OpenAI fit. Decision rule: ARI ≥ 0.5 → cluster
   structure not embedding-specific; ≤ 0.3 → name embedding in every
   claim.
3. Cluster comparison: identify 5 clusters where the two methods
   roughly agree on most members and 5 where they substantially
   disagree (use centroid Hungarian alignment to compare). For each,
   list the top-words from BERTopic and the top words from NMF or
   MiniLM-BERTopic.

Outputs:
- `data/bertopic/method_comparison.parquet` — rows: (comparison,
  ari, nmi, n_overlap, notes).
- Code at `figures/bertopic/stage2/t_method.py`.

Memo path: `figures/bertopic/memos/t_method.md`.
""",
    },
    "t_ablations": {
        "task_id": "T-ablations",
        "sections": "§8.2, §9.5, §13.4, §13.5",
        "time_budget": "3 hours",
        "spec": """\
Run the §8.2 secondary ablations and produce the T6 sign-consistency
matrix (§9.5). Each ablation re-runs from BERTopic fit through
headline-K labels and reports ARI vs the headline assignments
(`assignments.parquet`). Use `figures.bertopic.stage1.pipeline`
helpers wherever possible.

Ablations to run:

1. Embedding model: `all-MiniLM-L6-v2` (load from T-method's outputs
   if it ran first; else encode fresh). Skip `text-embedding-3-small`
   and jobBERT-v2 if not locally cached and document the skips.
2. UMAP n_components: 10 and 15 (5 is headline).
3. UMAP n_neighbors: 30 and 50 (15 is headline).
4. Sample cap: 3, 10, legacy 30/(co × period), uncapped. Build the
   alternate samples in-place via DuckDB.
5. Aggregator: exclude (`is_aggregator = FALSE`).
6. Substrate: SKIP — the user has explicitly forbidden raw
   `description`. Document the skip.
7. Length floor: 100 and 400 (200 is headline).
8. Outlier reduction: on (`reduce_outliers(strategy='embeddings')`)
   vs off (headline).

For each ablation:
- Refit BERTopic at headline mcs, seed = 42.
- Reduce to headline K (where applicable).
- Compute ARI vs headline assignments on overlapping uids.
- Compute mean centroid alignment vs headline (Hungarian).
- Compute noise rate.
- Compute T6 sign-consistency for primary claims C1–C4 (§1.4.1):
  * C1 (AI cluster emerged): is there a cluster whose top-words
    contain AI vocabulary AND n_2026/n_2024 >= 2× ?
  * C2 (legacy shrunk): is there a cluster whose top-words contain
    legacy vocabulary (.NET/COBOL/PHP/etc.) AND n_2026/n_2024 < 0.6 ?
  * C3 (rewriting): are there ≥ 4 stable clusters with top-words
    set-difference >= 30 % across periods ?
  * C4 (concentration): is the entropy of share distribution lower
    in 2026 than 2024 ?
  Score each as ✓ / ✗ / partial / N/A.

Outputs:
- `data/bertopic/ablations.parquet` — one row per ablation:
  name, variant, n_clusters, ari_vs_headline, centroid_alignment,
  noise_rate, c1_holds, c2_holds, c3_holds, c4_holds, notes.
- `data/bertopic/t6_robustness.parquet` — sign-consistency matrix
  rows = headline + each ablation, cols = C1, C2, C3, C4.
- Code at `figures/bertopic/stage2/t_ablations.py`.

Time budget enforcement: if you exceed 2 hours, **pause and surface**
which ablations remain — the orchestrator can decide to drop the
remaining ones rather than have you silently run out of time.

Memo path: `figures/bertopic/memos/t_ablations.md`.
""",
    },
    "t_quality": {
        "task_id": "T-quality",
        "sections": "§7.8, §7.9, §7.10, §7.11, §13.4, §13.5",
        "time_budget": "45 minutes",
        "spec": """\
1. §7.8 Coherence + diversity. Compute NPMI, UMass, C_v on
   `description_core_llm` Sample A as reference; topic top-10 terms
   per cluster from `topic_info.parquet`. Use
   `gensim.models.CoherenceModel(coherence='c_v')`. Topic diversity =
   unique tokens / total across all clusters' top-10 lists.
2. §7.9 Silhouette + cluster size. Cluster-size distribution
   (median, IQR, p5/p95) at headline K. Silhouette score in 5-D UMAP
   space — load the saved BERTopic raw fit
   (`data/bertopic/model.bertopic`) and pull the UMAP-reduced
   embeddings if BERTopic exposes them
   (`topic_model.umap_model.embedding_`); otherwise rerun UMAP on
   the cached 3072-d embeddings with the same hyperparameters and
   seed=42. Mean per-cluster silhouette and overall mean.
3. §7.10 Honest noise rate. Report HDBSCAN noise rate before AND
   after `reduce_outliers(strategy='embeddings')`. Both numbers go
   in the memo and the paper.
4. §7.11 Cross-model naming. Re-run the §5.1 LLM naming with
   `gpt-5.4-mini` against the gpt-5.5 primary labels in
   `topic_info.parquet`. Use
   `figures.bertopic.stage1.naming.propose_label` with
   `model="gpt-5.4-mini"`. Compute exact-match label rate and
   label-embedding cosine (embed labels via text-embedding-3-large
   in a single batch; cosine ≥ 0.85 = semantically equivalent).

Outputs:
- `data/bertopic/topic_quality.parquet` — (topic_id, n_members,
  npmi, umass, c_v, silhouette).
- A second parquet `data/bertopic/topic_info_with_naming.parquet`
  appending gpt-5.4-mini labels and label_cosine to topic_info.
- Code at `figures/bertopic/stage2/t_quality.py`.

Memo path: `figures/bertopic/memos/t_quality.md`.
""",
    },
}


_TEMPLATE = """\
You are a Stage 2 sub-agent for the BERTopic discovery + embedding-space
project (paper target: AIES 2026). Your task ID is `{task_id}`.

Read these files in order before writing any code:

1. `/home/jihgaboot/gabor/job-research/figures/bertopic/design.md` —
   sections {sections} for the analysis spec; §13.4 for sub-agent
   execution standard; §13.5 for the three-gate evaluation; §1.4 for
   the named claims your finding must connect to.
2. `/home/jihgaboot/gabor/job-research/figures/bertopic/config.py` —
   single source of truth for hyperparameters, anchor sets, paths.
3. `/home/jihgaboot/gabor/job-research/figures/bertopic/intermediate/stage1_freeze.json`
   — frozen Stage 1 hash bundle. **First action after reading**:
   recompute SHA256 of the on-disk artifacts and verify they match.
   The expected hashes are:
     - `model_hash`           = {model_hash}
     - `sample_hash`          = {sample_hash}
     - `embeddings_cache_hash`= {embeddings_cache_hash}
     - `assignments_hash`     = {assignments_hash}
     - `config_hash`          = {config_hash}
   If any mismatch, **fail loud and stop** — do not proceed.
4. `/home/jihgaboot/gabor/job-research/AGENTS.md` and
   `/home/jihgaboot/gabor/job-research/figures/style.md` for repo
   conventions (DuckDB for parquet, 31 GB RAM ceiling, pyarrow chunked
   I/O, `figures/style.py` for any plot you save).

## Frozen inputs

- BERTopic model: `data/bertopic/model.bertopic`
- Sample A: `figures/bertopic/intermediate/sample_a.parquet` (57,766 rows)
- Sample B: `figures/bertopic/intermediate/sample_b.parquet` (108,385 rows)
- Embeddings cache: `data/bertopic/embeddings_cache.npy`
  + index `data/bertopic/embeddings_cache.index.parquet`
- Assignments: `data/bertopic/assignments.parquet`
  (uid → topic_id at headline K = {headline_k}; topic_label;
  is_outlier).
- Topic info: `data/bertopic/topic_info.parquet` (per-topic c-TF-IDF
  top-words, gpt-5.5 proposed labels).
- Headline mcs = {headline_mcs}; super-family K = {super_family_k}.
- Stage 1 fits cached at `figures/bertopic/intermediate/mcs_fits/`,
  `seed_fits/`, `period_fits/`, and `determinism/` for re-use.

## Task

{spec}

## Standards

- Model: `claude-opus-4-7` (you), high effort. **No early termination**
  — run every robustness check the spec lists, even if early evidence
  looks compelling.
- Time budget: {time_budget}. If you exceed 2× budget, pause and
  surface the issue rather than silently finishing partial work.
- Use the repo venv: `/home/jihgaboot/gabor/job-research/.venv/bin/python`.
- DuckDB for parquet inspection. Open every connection with
  `con.execute('PRAGMA disable_progress_bar')` — DuckDB dumps a
  multi-line progress bar to stdout otherwise that pollutes scripts.
- Substrate is `description_core_llm` only; never raw `description`
  (user clarification 2026-05-05).
- 31 GB RAM ceiling. For per-row analyses on 58k × 3072 vectors,
  prefer numpy slicing over pandas; load embeddings via the cache,
  never re-read from `unified_core.parquet`.
- Save figures via `figures.style` (`from figures.style import
  setup, save, FIGSIZE_SINGLE, FIGSIZE_DOUBLE`); call `setup()` once.
- Code lives at `figures/bertopic/stage2/<task_file>.py`. One file
  per task. Standalone — no shared "utils" file.
- **Do not refit BERTopic** beyond what the spec authorizes. The
  Stage 1 frozen fit is read-only. If your analysis seems to require
  Stage 1 retuning, stop and report it — the orchestrator decides
  whether to escalate.
- Do not advocate for inclusion of your own work. The memo's job is
  to give the orchestrator and authors what they need to decide.
- Fail loud on anomalies (hash mismatch, dim mismatch, NaNs). Do not
  silently retry.

## Memo format (mandatory)

Write the memo to the path the spec gives you (`figures/bertopic/memos/<task_id>.md`).

```markdown
# {task_id} — <one-line description>

## What was run
Exact parameters, code paths, time taken. Enough detail to reproduce
from this section alone. Point at file paths in
`figures/bertopic/stage2/` you wrote — do not paste full code.

## Results
Tables and figures with paths to generated artifacts. Quote actual
numbers, not adjectives. If a number is small or surprising, say so.

## Three-gate evaluation (per design.md §13.5)
- **Gate 1 (Narrative).** Does this finding support a named claim
  from §1.4 (C1–C4 or T1–T4)? Pass / Fail with rationale.
- **Gate 2 (Effect size).** Does it clear the §13.5 threshold for
  this analysis? Pass / Fail with the actual number vs threshold.
- **Gate 3 (Robustness).** Which of {{seed reshuffle, anchor LOO,
  subset replication, permutation null, cross-embedding}} did it
  survive? List those checked and the result for each.

## recommend_for_paper: yes / no / conditional

## Rationale
One paragraph, evidence-based, no advocacy. The orchestrator decides
inclusion; your job is to give it what it needs.
```
"""


def compose_prompt(task_key: str) -> str:
    if task_key not in _TASKS:
        raise ValueError(f"unknown task: {task_key}")
    if not config.STAGE1_FREEZE_JSON.exists():
        raise RuntimeError(
            f"missing {config.STAGE1_FREEZE_JSON}; Stage 1 has not frozen yet"
        )
    bundle = json.loads(config.STAGE1_FREEZE_JSON.read_text())
    task = _TASKS[task_key]
    return _TEMPLATE.format(
        task_id=task["task_id"],
        sections=task["sections"],
        spec=task["spec"],
        time_budget=task["time_budget"],
        headline_k=bundle["headline_k"],
        headline_mcs=bundle["headline_mcs"],
        super_family_k=bundle.get("super_family_k", "(see freeze.json)"),
        model_hash=bundle["model_hash"],
        sample_hash=bundle["sample_hash"],
        embeddings_cache_hash=bundle["embeddings_cache_hash"],
        assignments_hash=bundle["assignments_hash"],
        config_hash=bundle["config_hash"],
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: dispatch.py <task_key|--all>")
    if sys.argv[1] == "--all":
        for key in _TASKS:
            print(f"\n\n========== {key} ==========\n")
            print(compose_prompt(key))
        return
    print(compose_prompt(sys.argv[1]))


if __name__ == "__main__":
    main()
