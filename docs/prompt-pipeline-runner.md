# Orchestrator Prompt: Run Full Preprocessing Pipeline

You are a pipeline orchestrator agent. Your job is to run the entire preprocessing pipeline end-to-end, monitor progress, fix small issues, and manage LLM quota carefully. The user is going to sleep and will need the coding agent quota back in ~10 hours.

## Setup

Read these files first:
1. `AGENTS.md` — project context, pipeline status, architectural rules
2. `preprocessing/run_pipeline.py` — the pipeline runner (for stages 1-8)
3. `docs/plan-preprocessing-fixes.md` — fixes that were just implemented (for verification context)

The project venv is at `.venv/bin/python`.

## Phase 1: Stages 1-8 (deterministic, no LLM)

Run stages 1 through 8 using the pipeline runner. These stages are deterministic and should complete in ~15-30 minutes total.

```bash
cd /home/jihgaboot/gabor/job-research
.venv/bin/python preprocessing/run_pipeline.py --from-stage 1
```

**BUT** — the pipeline runner runs ALL stages including 9-12. Since we need to run stages 9-12 separately with special handling, you should NOT use the pipeline runner for the full run. Instead, run stages individually:

```bash
.venv/bin/python preprocessing/scripts/stage1_ingest.py
.venv/bin/python preprocessing/scripts/stage2_aggregators.py
.venv/bin/python preprocessing/scripts/stage3_boilerplate.py
.venv/bin/python preprocessing/scripts/stage4_dedup.py
.venv/bin/python preprocessing/scripts/stage5_classification.py
.venv/bin/python preprocessing/scripts/stage678_normalize_temporal_flags.py
```

Run these sequentially. After each stage, verify the output exists:

```bash
ls -la preprocessing/intermediate/stage1_unified.parquet
ls -la preprocessing/intermediate/stage2_aggregators.parquet
ls -la preprocessing/intermediate/stage3_boilerplate.parquet
ls -la preprocessing/intermediate/stage4_dedup.parquet
ls -la preprocessing/intermediate/stage5_classification.parquet
ls -la preprocessing/intermediate/stage8_final.parquet
```

**Monitoring:** Each stage writes a log file to `preprocessing/logs/`. If a stage runs for more than 20 minutes without completing, check the log for progress:

```bash
tail -20 preprocessing/logs/stage5_classification.log
```

**If a stage fails:** Read the error message. Common issues:
- Import errors → missing package, fix the import
- File not found → previous stage didn't produce output, rerun it
- Memory errors → the 31GB limit was hit, check if something loaded full parquet into pandas
- Syntax errors → a preprocessing fix introduced a bug, read the traceback and fix

If you can fix the issue with a small edit (typo, missing import, off-by-one), do so and rerun. If the issue is fundamental, stop and report.

### Phase 1 Verification

After stage 6-8 completes, run these quick checks:

```bash
.venv/bin/python -c "
import pyarrow.parquet as pq
pf = pq.ParquetFile('preprocessing/intermediate/stage8_final.parquet')
print(f'Rows: {pf.metadata.num_rows:,}')
print(f'Columns: {pf.metadata.num_columns}')
# Check key columns exist
schema_names = set(pf.schema.names)
for col in ['is_swe', 'seniority_final', 'seniority_final_source', 'description', 'period', 'source']:
    assert col in schema_names, f'Missing column: {col}'
print('All key columns present')
"
```

Row count should be ~1.2M. If it's drastically different (e.g., 0 or 100K), something went wrong.

## Phase 2: Stage 9 (LLM routing/prefilter)

Stage 9 decides which rows need LLM processing. It's fast and deterministic (no actual LLM calls).

```bash
.venv/bin/python preprocessing/scripts/stage9_llm_prefilter.py
```

Verify output:
```bash
.venv/bin/python -c "
import pyarrow.parquet as pq
pf = pq.ParquetFile('preprocessing/intermediate/stage9_llm_candidates.parquet')
print(f'LLM candidates: {pf.metadata.num_rows:,} rows')
"
```

## Phase 3: Stage 10 (LLM calls — CAREFUL)

This is the long-running stage. It makes external LLM API calls and will likely hit quota limits.

**Configuration:**
- Models: Claude uses `haiku`, Codex uses `gpt-5.4-mini` (hardcoded in the script)
- Provider order: `claude,codex` — try Claude Haiku first, fall back to Codex GPT-5.4-mini
- Quota wait: 5 hours (when quota is hit, the script pauses all calls for 5 hours, then retries)
- Max workers: 40 (parallel LLM calls)
- The cache DB at `preprocessing/cache/llm_responses.db` persists progress. If the script stops and restarts, it skips already-cached `(description_hash, task_name, prompt_version)` entries.

**Run Stage 10:**

```bash
.venv/bin/python preprocessing/scripts/stage10_llm_classify.py \
  --provider-order claude,codex \
  --quota-wait-hours 5 \
  --max-workers 40
```

**IMPORTANT — 10-hour budget:**
The user needs the coding agent back in ~10 hours. Stage 10 will likely:
1. Run for 1-3 hours processing LLM calls
2. Hit a quota limit → automatically pause for 5 hours
3. Resume and process more calls for 1-3 hours
4. Possibly hit quota again

This fits within the 10-hour window. **Do NOT manually intervene during quota pauses** — the script handles them automatically. Just let it run.

**Monitoring Stage 10:**
While Stage 10 is running, periodically check progress (every 30-60 minutes):

```bash
# Check the log for progress
tail -30 preprocessing/logs/stage10_llm_classify.log

# Check cache growth
.venv/bin/python -c "
import sqlite3
con = sqlite3.connect('preprocessing/cache/llm_responses.db')
r = con.execute('SELECT count(*) FROM responses').fetchone()
print(f'Cached responses: {r[0]}')
r2 = con.execute('SELECT task_name, count(*) FROM responses GROUP BY task_name').fetchall()
for row in r2:
    print(f'  {row[0]}: {row[1]}')
con.close()
"
```

The cache should grow over time. If it's stuck at the same number for >30 minutes (outside a quota pause), something is wrong — check the log.

**If Stage 10 hits a quota limit:**
The script will log something like: "Quota/rate limit detected... Pausing all new provider calls until [time] UTC."
This is EXPECTED. Let it wait. After 5 hours it will automatically resume.

**If Stage 10 crashes (not quota, but actual error):**
Read the error. Fix if small. Then restart with the same command — the cache DB preserves all progress, so it will skip already-completed calls and pick up where it left off.

**If Stage 10 is still running when the 10-hour window is approaching:**
The user needs quota back. If Stage 10 is in a quota pause and there's <1 hour left, it's OK to let it sit — it won't use quota while paused. If it's actively making calls and the 10-hour window is nearly up, you can Ctrl-C it. The cache preserves progress, and the user can rerun later.

**If Stage 10 finishes completely:**
It will write `preprocessing/intermediate/stage10_llm_results.parquet`. Verify:
```bash
.venv/bin/python -c "
import pyarrow.parquet as pq
pf = pq.ParquetFile('preprocessing/intermediate/stage10_llm_results.parquet')
print(f'LLM results: {pf.metadata.num_rows:,} rows, {pf.metadata.num_columns} cols')
"
```

## Phase 4: Stages 11-12 + Final (only if Stage 10 completed)

**Only proceed to these stages if Stage 10 finished writing its output parquet.**

### Stage 11: LLM Integration
```bash
.venv/bin/python preprocessing/scripts/stage11_llm_integrate.py
```

Verify:
```bash
.venv/bin/python -c "
import pyarrow.parquet as pq
pf = pq.ParquetFile('preprocessing/intermediate/stage11_llm_integrated.parquet')
print(f'Integrated: {pf.metadata.num_rows:,} rows')
assert 'description_core_llm' in pf.schema.names, 'Missing description_core_llm'
print('description_core_llm column present')
"
```

Row count must match stage8_final (~1.2M). If it doesn't, Stage 11 dropped rows — that's a bug per architectural rules.

### Stage 12: Validation (optional, smoke-test only)

Stage 12 is a validation report, not a data transformation. It's scaffolded and may not be fully production-ready. Run it but don't block on failure:

```bash
.venv/bin/python preprocessing/scripts/stage12_validation.py || echo "Stage 12 failed (non-blocking)"
```

### Final output

```bash
.venv/bin/python preprocessing/scripts/stage_final_output.py
```

Verify:
```bash
.venv/bin/python -c "
import pyarrow.parquet as pq
for f in ['data/unified.parquet', 'data/unified_observations.parquet']:
    pf = pq.ParquetFile(f)
    print(f'{f}: {pf.metadata.num_rows:,} rows, {pf.metadata.num_columns} cols')
"
```

## Completion Report

When done (or when the 10-hour window is up), report:

1. **Stages 1-8:** Did they all complete? Any fixes needed?
2. **Stage 9:** How many LLM candidates?
3. **Stage 10:** How many LLM responses cached? Did it complete, or was it interrupted?
   - If interrupted: how many total responses, how many per task? (User can resume later)
4. **Stages 11-final:** Did they run? Output sizes?
5. **Any code changes made** to fix issues (list files and what changed)

If Stage 10 did NOT complete (quota limited, timed out), that's OK — report the partial progress and note that the user can resume with the same command. The cache preserves everything.

## Rules

- Use the project venv (`.venv/bin/python`) for everything
- Never modify data files manually — only run pipeline scripts
- If a stage fails, read the error and fix small issues (typos, missing imports). Do NOT attempt large refactors.
- Do NOT run `git commit` or `git push` — the user will review changes
- Respect the 31GB RAM limit — if you see memory issues, don't try to work around them by increasing limits
- Stage 10 timeout is 24 hours in the pipeline runner config, but your effective budget is ~10 hours
- Do NOT kill Stage 10 during a quota pause — it will resume automatically
- If Stage 10 is actively running (not paused) and the 10h window is almost up, it's OK to stop it — cache is durable
