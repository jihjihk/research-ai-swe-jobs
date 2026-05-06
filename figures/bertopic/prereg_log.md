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
