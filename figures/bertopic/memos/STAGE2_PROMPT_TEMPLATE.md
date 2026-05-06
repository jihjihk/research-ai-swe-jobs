# Stage 2 sub-agent prompt template

Every Stage 2 sub-agent receives a self-contained prompt of the form below.
The orchestrator fills in `<TASK_ID>`, the per-task spec, and the
hash-anchored Stage 1 inputs from `intermediate/stage1_freeze.json` at
dispatch time. Sub-agents do not see this template directly — they see
the filled-in prompt.

Memos land at `figures/bertopic/memos/<task-id>.md` and follow the §13.5
three-gate format. The orchestrator verifies each memo per §13.4 and logs
verifications in `prereg_log.md`.

---

You are a Stage 2 sub-agent for the BERTopic discovery + embedding-space
project (paper target: AIES 2026). Your task ID is `<TASK_ID>`. Read these
files in order before writing any code:

1. `/home/jihgaboot/gabor/job-research/figures/bertopic/design.md` —
   §<SECTIONS> for the analysis spec; §13.4 for sub-agent execution
   standard; §13.5 for the three-gate evaluation; §1.4 for the named claims
   you must connect findings to.
2. `/home/jihgaboot/gabor/job-research/figures/bertopic/config.py` —
   single source of truth for hyperparameters, anchor sets, paths.
3. `/home/jihgaboot/gabor/job-research/figures/bertopic/intermediate/stage1_freeze.json`
   — frozen Stage 1 hashes. Your **first action after reading** is to
   recompute SHA256 of the on-disk artifacts and verify they match:
     - `model_hash` for `data/bertopic/model.bertopic`
     - `sample_hash` for `figures/bertopic/intermediate/sample_a.parquet`
     - `embeddings_cache_hash` for `data/bertopic/embeddings_cache.npy`
     - `assignments_hash` for `data/bertopic/assignments.parquet`
     - `config_hash` for `figures/bertopic/config.py`
   If any mismatch, **fail loud and stop** — do not proceed.
4. `/home/jihgaboot/gabor/job-research/AGENTS.md` and
   `/home/jihgaboot/gabor/job-research/figures/style.md` — repo conventions
   (DuckDB for parquet inspection, 31 GB RAM ceiling, pyarrow chunked I/O,
   `figures/style.py` for any plot you save).

## Frozen inputs (verify hashes match)

- BERTopic model: `data/bertopic/model.bertopic` (hash `<MODEL_HASH>`)
- Sample A: `figures/bertopic/intermediate/sample_a.parquet` (hash
  `<SAMPLE_HASH>`) — 57,766 rows, columns include uid, source, period,
  title, description_core_llm, company_name_canonical, is_aggregator,
  metro_area, seniority_final, yoe_min_years_llm, is_swe, is_control.
- Sample B: `figures/bertopic/intermediate/sample_b.parquet` — 108,385
  rows, used for cross-occupation analyses.
- Embeddings cache: `data/bertopic/embeddings_cache.npy` (hash
  `<CACHE_HASH>`) + `data/bertopic/embeddings_cache.index.parquet`
  (key = uid for postings, key = anchor-id for §11.7 anchors).
- Assignments: `data/bertopic/assignments.parquet` (uid → topic_id at
  headline K=`<HEADLINE_K>`, with topic_label and is_outlier).
- Topic info: `data/bertopic/topic_info.parquet` (per-topic c-TF-IDF
  top-words, gpt-5.5 proposed labels).
- `config.py` hash: `<CONFIG_HASH>`.

## Task

<PER_TASK_SPEC>

## Outputs

Artifacts:
<OUTPUT_PATHS>

Memo: `/home/jihgaboot/gabor/job-research/figures/bertopic/memos/<task-id>.md`

## Memo format (mandatory)

```markdown
# <Task ID> — <One-line description>

## What was run
Exact parameters, code paths, time taken. Include enough detail that someone
could reproduce the run from this section alone. Do not paste full code
listings — point at file paths in `figures/bertopic/stage2/` you wrote.

## Results
Tables and figures with paths to the generated artifacts. Quote actual
numbers, not adjectives. If a number is small or surprising, say so plainly.

## Three-gate evaluation (per design.md §13.5)
- **Gate 1 (Narrative).** Does this finding support a named claim from §1.4
  (C1–C4 or T1–T4)? Pass / Fail with rationale.
- **Gate 2 (Effect size).** Does it clear the §13.5 threshold for this
  analysis? Pass / Fail with the actual number vs threshold.
- **Gate 3 (Robustness).** Which of {seed reshuffle, anchor LOO, subset
  replication, permutation null, cross-embedding} did it survive? List those
  checked and the result for each.

## recommend_for_paper: yes / no / conditional

## Rationale
One paragraph, evidence-based, no advocacy. The orchestrator decides
inclusion; your job is to give it what it needs, not to push for a verdict.
```

## Standards

- Model: `claude-opus-4-7` (you), high effort. No early termination — run
  every robustness check the spec lists, even if early evidence is
  compelling.
- Use the repo venv: `/home/jihgaboot/gabor/job-research/.venv/bin/python`.
- DuckDB for parquet inspection (`PRAGMA disable_progress_bar` at the top
  of every query session — it dumps to stdout otherwise).
- Substrate is `description_core_llm` only; never raw `description`.
- 31 GB RAM ceiling. For per-row analyses on 58 k × 3072 vectors, prefer
  numpy slicing over pandas; load embeddings via the cache, never re-read
  from `unified_core.parquet`.
- Save any figure via `figures.style` (`from figures.style import setup,
  save, FIGSIZE_SINGLE, FIGSIZE_DOUBLE`); call `setup()` once.
- Code lives at `figures/bertopic/stage2/<task_id>.py`. One file per task.
  Standalone — no shared "utils" file.
- **Do not refit BERTopic** or retune Stage 1 hyperparameters. The frozen
  fit is read-only. If your analysis seems to require refitting, stop and
  report it — the orchestrator decides whether to escalate.
- Do not advocate for inclusion of your own work. The memo's job is to
  give the orchestrator and authors what they need to decide.
- Fail loud on anomalies (hash mismatch, dimensional mismatch, NaNs in
  results); do not silently retry.
- Stay under your time budget (`<TIME_BUDGET>`). If you exceed 2× budget,
  pause and surface the issue rather than silently finishing partial work.
