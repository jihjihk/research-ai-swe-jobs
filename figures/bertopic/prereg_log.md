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
