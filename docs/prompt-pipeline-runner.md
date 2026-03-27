# Orchestrator Prompt: Run Full Preprocessing Pipeline

You are a pipeline orchestrator agent. Your job is to run the preprocessing pipeline end-to-end, monitor progress, fix small issues, and keep the long-running LLM stages moving until they finish.

The main expectations are:
- Stages 1-8 are deterministic. Run them sequentially and verify each output is healthy.
- Stages 9 and 10 use the shared engine runtime. Run them after Stage 8 with the configured engines and engine tiers.
- Use 30 workers for both LLM stages unless there is a concrete reason to change it.
- Monitor the LLM stages roughly hourly. Confirm the cache is growing unless the log clearly shows an intentional engine pause.
- If something crashes, read the error and fix small issues. If the problem is materially larger than a small repair/restart, notify the user.
- Stages 11 and 12 have not been exercised end-to-end in production yet. Expect possible bugs there and handle them carefully.
- There is no artificial 10-hour stop. Let the run continue until it is done unless you hit a materially larger issue.

## Setup

Read these files first:
1. `AGENTS.md`
2. `docs/1-research-design.md`
3. `docs/plan-preprocessing.md`
4. `preprocessing/run_pipeline.py`
5. `preprocessing/scripts/stage10_llm_classify.py`
6. `preprocessing/scripts/stage11_llm_integrate.py`
7. `preprocessing/scripts/stage12_validation.py`
8. `preprocessing/scripts/stage_final_output.py`

Use the project virtualenv for all commands:

```bash
cd /home/jihgaboot/gabor/job-research
./.venv/bin/python --version
```

## Phase 1: Run Stages 1-8

`preprocessing/run_pipeline.py` does not have a stop-at-stage flag. Do not use it for a Stage 1-8-only run, because it will continue into Stage 9-10 with default settings. Run the deterministic stages directly:

```bash
cd /home/jihgaboot/gabor/job-research
./.venv/bin/python preprocessing/scripts/stage1_ingest.py
./.venv/bin/python preprocessing/scripts/stage2_aggregators.py
./.venv/bin/python preprocessing/scripts/stage3_boilerplate.py
./.venv/bin/python preprocessing/scripts/stage4_dedup.py
./.venv/bin/python preprocessing/scripts/stage5_classification.py
./.venv/bin/python preprocessing/scripts/stage678_normalize_temporal_flags.py
```

Expected main outputs:
- `preprocessing/intermediate/stage1_unified.parquet`
- `preprocessing/intermediate/stage1_observations.parquet`
- `preprocessing/intermediate/stage2_aggregators.parquet`
- `preprocessing/intermediate/stage3_boilerplate.parquet`
- `preprocessing/intermediate/stage4_dedup.parquet`
- `preprocessing/intermediate/stage4_company_name_lookup.parquet`
- `preprocessing/intermediate/stage5_classification.parquet`
- `preprocessing/intermediate/stage8_final.parquet`

After each stage, verify the output is readable parquet, not just that the file exists. Use DuckDB through the repo venv:

```bash
./.venv/bin/python - <<'PY'
import duckdb
for path in [
    "preprocessing/intermediate/stage1_unified.parquet",
    "preprocessing/intermediate/stage1_observations.parquet",
    "preprocessing/intermediate/stage2_aggregators.parquet",
    "preprocessing/intermediate/stage3_boilerplate.parquet",
    "preprocessing/intermediate/stage4_dedup.parquet",
    "preprocessing/intermediate/stage4_company_name_lookup.parquet",
    "preprocessing/intermediate/stage5_classification.parquet",
    "preprocessing/intermediate/stage8_final.parquet",
]:
    try:
        rows = duckdb.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
        print(f"{path}: {rows:,} rows")
    except Exception as exc:
        print(f"{path}: ERROR -> {exc}")
PY
```

Known gotcha:
- `preprocessing/intermediate/stage5_classification.parquet` was previously reported as unreadable (`No magic bytes found at end of file`). Treat parquet readability as part of stage verification, especially after Stage 5.

Useful log files:
- `preprocessing/logs/stage1_ingest.log`
- `preprocessing/logs/stage2_aggregators.log`
- `preprocessing/logs/stage3_boilerplate.log`
- `preprocessing/logs/stage4_dedup.log`
- `preprocessing/logs/stage5_classification.log`
- `preprocessing/logs/stage678.log`

If a deterministic stage runs unusually long, inspect its log:

```bash
tail -50 preprocessing/logs/stage5_classification.log
```

If a deterministic stage fails:
- Read the traceback first.
- Small fix: typo, missing import, path mismatch, simple schema mismatch, stale assumption in a validation check.
- Then rerun that stage and re-verify the parquet.
- If the failure suggests a larger architectural or data-contract problem, stop and notify the user.

## Phase 1 Verification

After Stage 8 completes, run a quick sanity check:

```bash
./.venv/bin/python - <<'PY'
import duckdb
path = "preprocessing/intermediate/stage8_final.parquet"
rows = duckdb.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
cols = len(duckdb.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall())
print(f"rows={rows:,} cols={cols}")
missing = []
schema = {row[0] for row in duckdb.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()}
for col in ["is_swe", "seniority_final", "seniority_final_source", "description", "period", "source"]:
    if col not in schema:
        missing.append(col)
print("missing_columns=", missing)
PY
```

If the row count is drastically off or key columns are missing, do not continue into the LLM stages until the issue is understood.

## Phase 2: Run Stage 9

Stage 9 is now the extraction stage. It selects the LLM analysis universe, builds the deterministic control cohort, applies the short-description hard skip, runs extraction only, and materializes the posting-level cleaned-text artifact.

```bash
./.venv/bin/python preprocessing/scripts/stage9_llm_prefilter.py \
  --engines codex,claude \
  --engine-tiers codex=full,claude=non_intrusive \
  --quota-wait-hours 5 \
  --max-workers 30
```

Verify the key outputs:

```bash
./.venv/bin/python - <<'PY'
import duckdb
for path in [
    "preprocessing/intermediate/stage9_control_cohort.parquet",
    "preprocessing/intermediate/stage9_llm_extraction_candidates.parquet",
    "preprocessing/intermediate/stage9_llm_extraction_results.parquet",
    "preprocessing/intermediate/stage9_llm_cleaned.parquet",
]:
    try:
        rows = duckdb.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
        print(f"{path}: {rows:,} rows")
    except Exception as exc:
        print(f"{path}: ERROR -> {exc}")
stage8 = duckdb.execute("SELECT count(*) FROM read_parquet('preprocessing/intermediate/stage8_final.parquet')").fetchone()[0]
stage9 = duckdb.execute("SELECT count(*) FROM read_parquet('preprocessing/intermediate/stage9_llm_cleaned.parquet')").fetchone()[0]
print(f"stage8_final rows={stage8:,}")
print(f"stage9_llm_cleaned rows={stage9:,}")
PY
```

## Phase 3: Run Stage 10

Stage 10 is now the classification stage plus final posting-level integration. It reads `stage9_llm_cleaned.parquet`, routes only the rows that still need classification, executes classification on the cleaned-description fallback chain, and writes the canonical LLM-integrated posting table.

The important operational facts are:
- Engine selection is explicit via `--engines`.
- Tier assignment is explicit via `--engine-tiers`.
- A task is assigned to one engine and stays on that engine until it succeeds.
- Non-quota failures wait 1 minute and retry the same engine.
- Quota pauses are provider-scoped, not global.
- Use 30 workers.
- Progress is checkpointed to `preprocessing/cache/llm_responses.db` immediately after each successful task.
- The final Stage 10 parquets are only materialized at the end of a clean full run. Partial progress lives in the cache DB and log.

```bash
./.venv/bin/python preprocessing/scripts/stage10_llm_classify.py \
  --engines codex,claude \
  --engine-tiers codex=full,claude=non_intrusive \
  --quota-wait-hours 5 \
  --max-workers 30
```

Current defaults already match the intended settings:
- Claude model: `haiku`
- Codex model: `gpt-5.4-mini`
- Quota wait: `5.0` hours
- Retry wait after a failed task attempt: `60` seconds
- Max workers: `30`

### Stage 10 Monitoring

Check progress about once per hour.

Main log:

```bash
tail -50 preprocessing/logs/stage10_llm.log
```

Cache growth:

```bash
./.venv/bin/python - <<'PY'
import sqlite3
con = sqlite3.connect("preprocessing/cache/llm_responses.db")
total = con.execute("SELECT count(*) FROM responses").fetchone()[0]
window = con.execute("SELECT min(timestamp), max(timestamp) FROM responses").fetchone()
print(f"cached responses: {total:,}")
print(f"first timestamp: {window[0]}")
print(f"last timestamp:  {window[1]}")
for task_name, n in con.execute("SELECT task_name, count(*) FROM responses GROUP BY task_name ORDER BY task_name"):
    print(f"{task_name}: {n:,}")
con.close()
PY
```

Interpretation:
- If the cache counts and latest timestamp are moving forward, Stage 10 is healthy.
- If the cache is flat but the log shows a quota/rate-limit pause with a future UTC resume time for one engine, that is expected. Leave it alone.
- If the cache is flat for a long time and the log does not show an intentional quota pause, inspect the latest error and decide whether a small fix/restart is needed.

Quota behavior:
- Full-tier engines pause for 5 hours on quota/rate-limit errors and then retry.
- Non-intrusive engines pause until the current five-hour slot ends after a quota hit.
- Do not stop the run just because one engine is in a logged quota pause.

Failure behavior:
- If a task attempt fails without a quota pause, the runtime waits 1 minute and retries the same engine.
- The runtime does not fall back to a different engine for that task.

Crash behavior:
- If Stage 10 crashes on a small issue, read the traceback, fix it, and restart with the same command.
- The cache DB is durable, so reruns should reuse completed `(input_hash, task_name, prompt_version)` rows.
- If the issue is materially larger than a small local fix, notify the user.

Completion check:
- A clean Stage 10 finish should write:
  - `preprocessing/intermediate/stage10_llm_classification_results.parquet`
  - `preprocessing/intermediate/stage10_llm_integrated.parquet`
- Do not use the absence of those parquets alone to judge partial progress; use the cache DB and log first.

## Phase 4: Final Output and Stage 12

The redesigned pipeline ends at Stage 10 for the posting-level LLM artifact. Stage 11 is compatibility-only at most; it is not the architectural handoff. Stage 12 is still the least production-proven part of the flow, so expect possible first-run bugs there.

Proceed only once Stage 10 appears complete.

### Verify Stage 10 integrated output

```bash
./.venv/bin/python - <<'PY'
import duckdb
stage8 = duckdb.execute("SELECT count(*) FROM read_parquet('preprocessing/intermediate/stage8_final.parquet')").fetchone()[0]
stage10 = duckdb.execute("SELECT count(*) FROM read_parquet('preprocessing/intermediate/stage10_llm_integrated.parquet')").fetchone()[0]
print(f"stage8_final rows={stage8:,}")
print(f"stage10_integrated rows={stage10:,}")
schema = {row[0] for row in duckdb.execute("DESCRIBE SELECT * FROM read_parquet('preprocessing/intermediate/stage10_llm_integrated.parquet')").fetchall()}
print("description_core_llm present =", "description_core_llm" in schema)
print("selected_for_control_cohort present =", "selected_for_control_cohort" in schema)
PY
```

If Stage 10 drops rows relative to Stage 9/Stage 8, treat that as a bug.

### Final output

Run final output generation after Stage 10. `stage_final_output.py` now expects `preprocessing/intermediate/stage10_llm_integrated.parquet`.

```bash
./.venv/bin/python preprocessing/scripts/stage_final_output.py
```

Verify:

```bash
./.venv/bin/python - <<'PY'
import duckdb
for path in ["data/unified.parquet", "data/unified_observations.parquet"]:
    rows = duckdb.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
    print(f"{path}: {rows:,} rows")
PY
```

### Stage 12

Stage 12 is a non-blocking validation/reporting step. It defaults to `data/unified.parquet`, so run it after final output. If it fails, read the error, attempt a small fix if appropriate, and otherwise report the failure without blocking the main pipeline result.

```bash
./.venv/bin/python preprocessing/scripts/stage12_validation.py
```

Useful logs:
- `preprocessing/logs/stage10_llm.log`
- `preprocessing/logs/stage12_validation.log`
- `preprocessing/logs/stage12_validation_errors.jsonl`

## Completion Report

When the run finishes, report:
1. Whether Stages 1-8 completed cleanly, and any fixes required.
2. Stage 9 output sizes and whether `stage9_llm_cleaned.parquet` preserved row count from Stage 8.
3. Stage 10 status:
   - whether the full run completed
   - total cached responses and counts by task
   - any quota pauses observed
4. Whether Stage 10 integrated output preserved row count and wrote the expected parquets.
5. Whether final outputs were written, with row counts.
6. Whether Stage 12 completed or failed.
7. Any code changes made to keep the run moving.

## Rules

- Use `./.venv/bin/python` for all Python commands.
- Do not modify data artifacts manually.
- Prefer small, local fixes only. Do not turn a pipeline run into a broad refactor.
- Do not run `git commit` or `git push`.
- Respect the 31GB RAM limit.
- Do not treat a logged Stage 10 quota pause as a crash.
- If the pipeline hits a genuinely larger issue, notify the user with the traceback, the stage, and your assessment of whether the problem is local or architectural.
