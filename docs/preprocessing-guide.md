# Preprocessing Pipeline Guide

Last updated: 2026-04-01

This document is the primary reference for the preprocessing pipeline that transforms raw job-posting data into the analysis-ready `unified.parquet` and `unified_observations.parquet` datasets. It covers architecture, operations, and development practices.

For the detailed column-by-column schema, see [`preprocessing-schema.md`](preprocessing-schema.md). For the research design that motivates this pipeline, see [`1-research-design.md`](1-research-design.md).

---

## Architecture

### Design philosophy

The pipeline has two layers:

1. **Rule-based baseline (Stages 1-8):** Deterministic, fast (~30 min end-to-end), reproducible. Produces a usable corpus with rule-based classification labels. Every column from this layer is preserved even after LLM augmentation runs.
2. **LLM augmentation (Stages 9-10):** Adds higher-quality classification and cleaned text via Codex (GPT) calls by default; Claude can be enabled via `--engines codex,claude`. Takes hours to days depending on corpus size and API quotas. Results are cached in SQLite for resumability.

The rule-based layer always runs first and its outputs serve as both the fallback labels and the cache keys for the LLM layer. The intended production dataset includes both layers, but the Stage 8 output is independently usable for exploration while LLM stages are in progress.

### Data flow

```
Raw Data (3 sources)
  │
  ├── Stage 1: Ingest & Schema Unification
  │     ├── 1a: Arshkon ingest + companion joins
  │     ├── 1b: Asaniczka ingest + description/skills joins
  │     ├── 1c: Scraped ingest (current-format, LinkedIn + Indeed)
  │     └── 1d: Concatenation to canonical 39-column format
  ├── Stage 2: Aggregator / Staffing Handling
  ├── Stage 3: Rule-Based Boilerplate Removal
  ├── Stage 4: Company Canonicalization + Dedup
  ├── Stage 5: Occupation + Seniority Classification
  ├── Stage 6: Location Normalization           ─┐
  ├── Stage 7: Temporal Alignment                │ (single script)
  ├── Stage 8: Quality Flags & Provenance       ─┘
  │
  ▼
  preprocessing/intermediate/stage8_final.parquet   ← rule-based baseline
  │
  ├── Stage 9: LLM Extraction + Cleaned Text
  ├── Stage 10: LLM Classification + Final Integration
  │
  ▼
  preprocessing/intermediate/stage10_llm_integrated.parquet
  │
  ├── Final Output Stage
  │
  ▼
  data/unified.parquet                 (one row per unique posting)
  data/unified_observations.parquet    (daily panel: one row per posting × scrape_date)
  data/quality_report.json
  data/preprocessing_log.txt
```

### Output artifacts

| Artifact | Unit of observation | Purpose |
|---|---|---|
| `unified.parquet` | One row per unique posting | Primary analysis dataset |
| `unified_observations.parquet` | One row per posting per scrape_date | Posting-duration analysis, daily panel |
| `quality_report.json` | Pipeline-level | Funnel metrics, quality summaries by source |
| `preprocessing_log.txt` | Pipeline-level | Human-readable run summary |

---

## Stage Reference

### Quick reference

| Stage | Script | Rows | Purpose |
|---|---|---|---|
| 1 | `stage1_ingest.py` | ~1.2M in | Ingest three sources, unify to canonical schema |
| 2 | `stage2_aggregators.py` | Preserved | Flag staffing agencies, extract real employers |
| 3 | `stage3_boilerplate.py` | Preserved | Rule-based boilerplate removal from descriptions |
| 4 | `stage4_dedup.py` | ~1.0M out | Company canonicalization + posting dedup |
| 5 | `stage5_classification.py` | Preserved | SWE/seniority classification, YOE extraction |
| 6-8 | `stage678_normalize_temporal_flags.py` | Preserved | Location, temporal, quality flags |
| 9 | `stage9_llm_prefilter.py` | Preserved | Control cohort, LLM extraction, cleaned text |
| 10 | `stage10_llm_classify.py` | Preserved | LLM classification + posting-level integration |
| 11 | `stage11_llm_integrate.py` | Preserved | Compatibility shim (copies Stage 10 output) |
| final | `stage_final_output.py` | Preserved | Produces `data/unified*.parquet` + reports |

"Preserved" means the stage does not change row count. Only Stage 4 (dedup) reduces rows.

### Stage details

**Stage 1 — Ingest & Schema Unification.** Loads three source datasets (Kaggle arshkon, Kaggle asaniczka, scraped current-format CSVs), applies source-specific joins and normalization, and concatenates into a canonical 39-column format. Handles seniority mapping, text normalization, date parsing, and company size parsing. Produces both the canonical posting table and a separate daily observations table for scraped data.

**Stage 2 — Aggregator Handling.** Identifies staffing agencies and job board aggregators (Dice, Lensa, Robert Half, etc.) via exact name matching and regex patterns. Extracts the real employer from description text when possible. Derives `company_name_effective` = real employer if aggregator, else original company name.

**Stage 3 — Boilerplate Removal.** Applies regex-based section header detection to strip EEO statements, benefits sections, application instructions, and company "About Us" blocks from descriptions. Produces `description_core`. Current accuracy is ~44% — superseded by LLM extraction in Stage 9 when available.

**Stage 4 — Company Canonicalization + Dedup.** The only row-reducing stage. Uses a memory-safe two-pass design: Pass 1 loads only dedup key columns to compute keep/drop decisions; Pass 2 streams full data and filters to kept rows. Dedup strategies: exact job_id duplicates, exact opening duplicates (company + title + location + description hash), and fuzzy near-duplicates (token_set_ratio >= 85%). Produces a `company_name_lookup` audit artifact.

**Stage 5 — Occupation + Seniority Classification.** The first analytical classification boundary. Assigns `is_swe`, `is_swe_adjacent`, and `is_control` via a 3-tier system (regex, curated title lookup, embedding). Runs a multi-signal seniority resolver producing `seniority_final` from title keywords, native platform labels, and description patterns. Extracts years-of-experience (YOE) from descriptions with a clause-aware parser.

**Stages 6-8 — Normalization, Temporal, Quality Flags.** A single script implementing three logical stages. Stage 6 parses locations into city/state/country, infers remote status, and assigns metro areas. Stage 7 derives `period`, `posting_age_days`, and `scrape_week`. Stage 8 adds language detection, date validation, ghost-job heuristics, and description quality flags. All three are row-preserving.

**Stage 9 — LLM Extraction + Cleaned Text.** Defines the LLM analysis universe (LinkedIn, English, has description). Selects a deterministic control cohort. Segments descriptions into sentence units and asks LLMs to identify boilerplate units for removal. Produces `description_core_llm` — the LLM-cleaned description used by Stage 10. Hard-skips descriptions under 15 words.

**Stage 10 — LLM Classification + Final Integration.** Routes eligible rows to LLM classification: SWE type, seniority, ghost-job assessment, and YOE cross-check. Skips rows where rule-based confidence is already high. Merges LLM results back to the full posting table. The output `stage10_llm_integrated.parquet` is the canonical LLM-augmented artifact.

**Stage 11 — Compatibility Shim.** Simply copies Stage 10 output to legacy paths. Not architecturally significant.

**Final Output.** Reads the latest integrated artifact (Stage 10 if available, Stage 8 otherwise), applies final column selection, and writes `data/unified.parquet`, `data/unified_observations.parquet`, `data/quality_report.json`, and `data/preprocessing_log.txt`.

---

## Data Sources

| Source | Rows | Period | Platform | Key strength | Key gap |
|---|---|---|---|---|---|
| Kaggle arshkon | 124K | Apr 2024 | LinkedIn | Entry-level labels (~385 SWE) | Small SWE count (~3,466) |
| Kaggle asaniczka | 1.35M | Jan 2024 | LinkedIn | Large volume (18K US SWE) | Zero entry-level labels |
| Scraped current-format | ~3.7K SWE/day | Mar 2026+ | LinkedIn + Indeed | Fresh data, search metadata | Growing daily |

**Platform policy:** LinkedIn is the primary analysis platform. Indeed is included for sensitivity analyses only. Both Kaggle sources are LinkedIn-only, making LinkedIn the cleanest cross-period comparison surface.

**Excluded data:** YC postings, Apify data, and old scraped format (Mar 5-18, which used 25 results/query and lacked search metadata columns).

**Sync fresh scraped data:**

```bash
aws s3 sync s3://swe-labor-research/scraped/ data/scraped/
```

---

## Running the Pipeline

### Prerequisites

All commands use the project virtualenv:

```bash
cd /home/jihgaboot/gabor/job-research
./.venv/bin/python --version
```

### Running Stages 1-8 (deterministic)

Run each stage individually and verify outputs between stages. `run_pipeline.py` does not have a stop-at-stage flag — it will continue into the LLM stages with default settings. For a deterministic-only run, invoke stage scripts directly:

```bash
./.venv/bin/python preprocessing/scripts/stage1_ingest.py
./.venv/bin/python preprocessing/scripts/stage2_aggregators.py
./.venv/bin/python preprocessing/scripts/stage3_boilerplate.py
./.venv/bin/python preprocessing/scripts/stage4_dedup.py
./.venv/bin/python preprocessing/scripts/stage5_classification.py
./.venv/bin/python preprocessing/scripts/stage678_normalize_temporal_flags.py
```

Expected runtime: ~30 minutes total. Stage 1 is the longest (~24 min) due to asaniczka's 1.35M rows.

Verify all outputs after completion:

```bash
./.venv/bin/python - <<'PY'
import duckdb
for path in [
    "preprocessing/intermediate/stage1_unified.parquet",
    "preprocessing/intermediate/stage1_observations.parquet",
    "preprocessing/intermediate/stage2_aggregators.parquet",
    "preprocessing/intermediate/stage3_boilerplate.parquet",
    "preprocessing/intermediate/stage4_dedup.parquet",
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

Quick sanity check on Stage 8 output:

```bash
./.venv/bin/python - <<'PY'
import duckdb
path = "preprocessing/intermediate/stage8_final.parquet"
rows = duckdb.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
cols = len(duckdb.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall())
print(f"rows={rows:,} cols={cols}")
schema = {row[0] for row in duckdb.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()}
for col in ["is_swe", "seniority_final", "seniority_final_source", "description", "period", "source"]:
    if col not in schema:
        print(f"MISSING: {col}")
PY
```

### Remote LLM execution

By default, stages 9 and 10 run LLM CLI commands (`codex`, `claude`) on the local machine. To run them on the remote EC2 instance via SSH instead, add `--remote`:

```bash
# Individual stage
./.venv/bin/python preprocessing/scripts/stage9_llm_prefilter.py --remote ...

# Full pipeline
./.venv/bin/python preprocessing/run_pipeline.py --remote
```

Remote mode uses a prewarmed SSH master connection (ControlMaster multiplexing) to avoid per-call SSH handshake overhead. The SSH key and host are configured in `llm_shared.py`. Remote mode requires the EC2 instance to have `codex` and/or `claude` CLI tools installed.

### Running Stage 9 (LLM extraction)

Stage 9 selects the LLM analysis universe, builds the control cohort, and runs extraction only:

```bash
./.venv/bin/python preprocessing/scripts/stage9_llm_prefilter.py \
  --engines codex \
  --quota-wait-hours 5 \
  --max-workers 20
```

To run LLM commands on the remote EC2 instance, add `--remote`:

```bash
./.venv/bin/python preprocessing/scripts/stage9_llm_prefilter.py \
  --remote \
  --engines codex \
  --quota-wait-hours 5 \
  --max-workers 20
```

Verify outputs:

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
PY
```

`stage9_llm_cleaned.parquet` should have the same row count as `stage8_final.parquet`.

### Running Stage 10 (LLM classification)

Stage 10 classifies the cleaned descriptions and writes the canonical LLM-integrated artifact:

```bash
./.venv/bin/python preprocessing/scripts/stage10_llm_classify.py \
  --engines codex \
  --quota-wait-hours 5 \
  --max-workers 20
```

Add `--remote` to run LLM commands on the remote EC2 instance (same as Stage 9).

This is the longest-running stage (hours to days). See "LLM Stage Operations" below for monitoring and recovery.

### Running Final Output

After Stage 10 completes:

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

### Using the Pipeline Runner

For a full end-to-end run including LLM stages:

```bash
./.venv/bin/python preprocessing/run_pipeline.py                  # Full run (local LLM)
./.venv/bin/python preprocessing/run_pipeline.py --remote         # Full run (remote LLM on EC2)
./.venv/bin/python preprocessing/run_pipeline.py --from-stage 3   # Resume from stage 3
./.venv/bin/python preprocessing/run_pipeline.py --dry-run        # Validate existing outputs
```

The runner validates each stage's output (row counts, required columns) before proceeding.

### S3 backup

After a successful pipeline run, back up final outputs and the LLM cache to S3 with a timestamped prefix so previous backups are never overwritten:

```bash
# Standalone (after any manual run)
./.venv/bin/python preprocessing/scripts/backup_to_s3.py

# Dry run — see what would be uploaded
./.venv/bin/python preprocessing/scripts/backup_to_s3.py --dry-run

# Integrated — automatically back up after a full pipeline run
./.venv/bin/python preprocessing/run_pipeline.py --backup
```

Uploads to `s3://swe-labor-research/backups/<YYYY-MM-DD_HHMMSS>/`:
- `unified.parquet`, `unified_observations.parquet` — final analysis datasets
- `quality_report.json`, `preprocessing_log.txt` — pipeline reports
- `llm_responses.db` — LLM cache (expensive to regenerate)

Missing files are skipped with a warning. Backup failure does not fail the pipeline run. Always back up after completing a full pipeline run, especially after LLM stages.

### Stage log files

Each stage writes to `preprocessing/logs/`:

| Log file | Stage |
|---|---|
| `stage1_ingest.log` | Stage 1 |
| `stage2_aggregators.log` | Stage 2 |
| `stage3_boilerplate.log` | Stage 3 |
| `stage4_dedup.log` | Stage 4 |
| `stage5_classification.log` | Stage 5 |
| `stage678.log` | Stages 6-8 |
| `stage9_llm.log` | Stage 9 |
| `stage10_llm.log` | Stage 10 |
| `stage12_validation.log` | Stage 12 |
| `pipeline_run.log` | Full pipeline runner |

---

## LLM Stage Operations

### Engine configuration

| Engine | Model | Invocation |
|---|---|---|
| Codex | `gpt-5.4-mini` | `codex exec --full-auto --config model=gpt-5.4-mini` |
| Claude | `haiku` | `claude -p "<prompt>" --model haiku --output-format json` |

Models are pinned in code. By default, these CLI commands run locally. Pass `--remote` to execute them on the remote EC2 instance via SSH.

The `--engines` flag controls which engines are active (default: `codex`); `--engine-tiers` assigns utilization modes:
- `full`: Standard utilization. Pauses 5 hours on quota hits, then retries.
- `non_intrusive`: Conservative slot budget. Pauses until the current 5-hour window ends after a quota hit.

### Monitoring

Check progress roughly once per hour during LLM stages.

**Tail the log:**
```bash
tail -50 preprocessing/logs/stage10_llm.log
```

**Check cache growth:**
```bash
./.venv/bin/python - <<'PY'
import sqlite3
con = sqlite3.connect("preprocessing/cache/llm_responses.db")
total = con.execute("SELECT count(*) FROM responses").fetchone()[0]
window = con.execute("SELECT min(timestamp), max(timestamp) FROM responses").fetchone()
print(f"cached responses: {total:,}")
print(f"first: {window[0]}  last: {window[1]}")
for task_name, n in con.execute("SELECT task_name, count(*) FROM responses GROUP BY task_name ORDER BY task_name"):
    print(f"  {task_name}: {n:,}")
con.close()
PY
```

**Interpretation:**
- Cache count rising + recent timestamp = healthy.
- Cache flat + log shows quota pause with future UTC resume time = expected, leave it alone.
- Cache flat + no quota pause in log = investigate the latest error.

### Cache and checkpointing

- All LLM responses are cached immediately in `preprocessing/cache/llm_responses.db`.
- Cache key: `(input_hash, task_name, model, prompt_version)`.
- Restarting a stage reuses all cached responses — no duplicate API calls.
- Final parquet outputs are only written at the end of a clean full run. Partial progress lives in the cache DB.

### Crash recovery

1. Read the traceback in the log.
2. If it is a small issue (missing import, path mismatch, schema mismatch): fix it and restart with the same command.
3. The cache DB is durable — reruns pick up where they left off.
4. If the issue is architectural or affects data contracts, stop and investigate before restarting.

### Task routing

- Each task is assigned to one engine and stays there for retries.
- Non-quota failures retry the same engine after 60 seconds.
- There is no cross-engine fallback for a given task.
- Do not stop a run because one engine is in a quota pause — the other engine continues working.

---

## Memory & Performance

### 31 GB RAM constraint

The pipeline runs on a machine with 31 GB RAM. Every stage is designed to stay within this limit.

### Chunking strategy

| Pattern | Chunk size | Used by |
|---|---|---|
| Standard processing | 200,000 rows | Stages 1, 3, 4, 5, 6-8 |
| LLM processing | 50,000 rows | Stages 9, 10 |
| Lightweight dedup pass | Full dataset, key columns only (~0.3 GB) | Stage 4 Pass 1 |

### Key patterns

- **PyArrow chunked reads:** `pq.ParquetFile(path).iter_batches(batch_size=200_000)` — avoids loading full datasets into memory.
- **Column projection:** Load only needed columns when possible (Stage 4 Pass 1 loads only dedup keys).
- **Explicit GC:** `gc.collect()` after each chunk in long-running stages.
- **Atomic writes:** All stage outputs write to `.tmp` files first, then atomically rename on success. This prevents corrupted artifacts from incomplete writes.
- **Metadata queries:** Row counts via `pq.ParquetFile(path).metadata.num_rows` — no decompression needed.

### Gotchas

- Never load a full parquet file into pandas with `pd.read_parquet()` unless the data volume is known to be safe. Use PyArrow chunked iteration instead.
- Stage 1 processing asaniczka (1.35M rows) is the peak memory point for deterministic stages.
- Stage 4 uses a two-pass design specifically to keep memory under control during dedup of 1.2M rows.
- The LLM cache DB (`llm_responses.db`) grows to hundreds of MB during long runs. This is expected and necessary for resumability.

---

## Development Guide

### Test framework

The test suite uses `pytest` with specialized markers and fixtures. Tests live in `tests/` and target the preprocessing pipeline contract.

**Running tests:**

```bash
# All unit + fixture tests (fast, default)
./.venv/bin/python -m pytest tests/ -m "unit or fixture" -v

# Single stage
./.venv/bin/python -m pytest tests/test_stage5_swe_classification.py -v

# Integration tests (slower, runs stage end-to-end on fixtures)
./.venv/bin/python -m pytest tests/ -m integration -v

# All tests
./.venv/bin/python -m pytest tests/ -v
```

### Test markers

| Marker | Scope | Speed | When to use |
|---|---|---|---|
| `unit` | Pure logic, no I/O | Fast | Regexes, normalizers, resolvers, validators |
| `fixture` | Checked-in small parquet/CSV | Fast | Schema contracts, branch coverage |
| `sampled` | Reviewed real-data rows | Fast | Messy real inputs, regression protection |
| `integration` | Full stage on temp fixtures | Medium | End-to-end stage contracts |
| `slow` | Sentence-transformers, large fixtures | Slow | Embedding-dependent tests |

### Test layers

1. **Pure-unit:** Logic tests with no file I/O. Regex patterns, normalizer functions, seniority resolvers, hash functions.
2. **Golden-fixture:** Tiny synthetic inputs with exact expected outputs. Targets branch coverage and schema contracts.
3. **Sampled-fixture:** Real rows extracted from production artifacts after manual review. Each gets a checked-in expected result.
4. **Stage-integration:** Runs a full stage end-to-end on a small temp fixture corpus. Verifies schema, row counts, null rates.

### Test file layout

```
tests/
  conftest.py                           # Shared fixtures, temp dirs
  helpers/
    imports.py                          # Dynamic module loading for stages
    stage_runner.py                     # Temp directory setup, parquet I/O
    llm_fakes.py                        # Mock LLM responses
    fixture_extractors.py               # Load fixture data
    parquet_asserts.py                  # DataFrame validation helpers
    sqlite_asserts.py                   # Cache validation
  fixtures/
    sampled/                            # Reviewed real-data fixtures
    synthetic/                          # Targeted synthetic fixtures
  test_pipeline_smoke.py                # End-to-end contract validation
  test_stage1_ingest.py                 # Stage 1 tests
  test_stage2_aggregators.py            # Stage 2 tests
  test_stage4_dedup.py                  # Stage 4 tests
  test_stage5_swe_classification.py     # Stage 5 SWE + seniority
  test_stage678_normalize_temporal_flags.py
  test_stage9_llm_prefilter.py          # LLM extraction routing
  test_stage10_llm_classify.py          # LLM classification routing
  test_llm_shared_runtime.py            # LLM runtime, providers, quota
```

### Making changes — TDD workflow

The default development mode is test-driven:

1. **Write or update the smallest high-signal test** that captures the intended behavior.
2. **Confirm it fails** for the right reason (when practical).
3. **Implement the code change.**
4. **Run the narrowest relevant test selection** first, then widen after the local contract is green.
5. **Update schema/contract tests** if the change affects stage boundaries or column definitions.

"High-signal" means logic assertions, schema/cardinality contracts, and reviewed sampled-row expectations — not broad snapshot churn.

### Stage ownership boundaries

Each stage has a strict contract about what it owns and must not do. These rules prevent scope creep between stages and protect data contracts.

**General principles:**
- Only Stage 4 reduces row count. All other stages are row-preserving.
- Do not move classification logic between stages (e.g., SWE classification belongs in Stage 5, not Stage 1).
- Schema-boundary changes need matching test updates.
- LLM columns are additive — they never erase rule-based columns, which remain as fallback and ablation baselines.
- Canonical postings and daily observations are both first-class outputs. Do not collapse them into one dataset.

**Stage-specific ownership rules:**
- **Stage 1:** Ingest, schema unification, provenance, and date handling only. Must treat approved historical sources equivalently and remain extensible to future sources. Must not define analytical occupation samples or filter rows by SWE/non-SWE/control class.
- **Stage 2:** Aggregator detection, `real_employer` extraction, and `company_name_effective` derivation. Must not own company canonicalization for dedup.
- **Stage 3:** Rule-based boilerplate removal only. Reads working `description`, preserves it unchanged, writes `description_core` plus quality flags. Must not own company-name normalization, occupation classification, source-specific text rules, or row filtering.
- **Stage 4:** Posting-level deduplication. Canonicalizes `company_name_effective` into `company_name_canonical`, handles exact/near-duplicate removal and `is_multi_location` flagging. May use normalized fields and description-derived support signals for dedup decisions, but must not redefine daily observations or analytical samples. Must not take over Stage 1, 2, or 5 responsibilities.
- **Stage 5:** First occupation-classification boundary. `is_swe`, `is_swe_adjacent`, and `is_control` belong here. Analytical samples are defined after classification, in later stages.
- **Stages 6-8:** Row-preserving enrichment only. May add normalization, temporal, quality, and provenance columns, but must not change row cardinality or define analytical samples.
- **Stages 9-12:** LLM augmentation only. Stage 10 may deduplicate LLM *calls* by cache key to reduce API volume, but that is call deduplication, not posting deduplication. Stage 11 must preserve row count. If row counts change in LLM stages, treat it as a bug.

### Regenerating reference data

If metro reference data needs updating:

```bash
./.venv/bin/python preprocessing/scripts/build_metro_city_state_reference.py
```

This rebuilds `preprocessing/reference/metro_city_state_lookup.parquet` from the CBSA reference files.

---

## Intermediate Artifacts

All intermediate outputs live in `preprocessing/intermediate/`. These are rebuilt on each pipeline run.

| Artifact | Source stage | Description |
|---|---|---|
| `stage1_unified.parquet` | 1 | Canonical postings from all three sources |
| `stage1_observations.parquet` | 1 | Daily observations for scraped data |
| `stage2_aggregators.parquet` | 2 | With aggregator flags and real employer |
| `stage3_boilerplate.parquet` | 3 | With `description_core` |
| `stage4_dedup.parquet` | 4 | After deduplication |
| `stage4_company_name_lookup.parquet` | 4 | Company canonicalization audit trail |
| `stage5_classification.parquet` | 5 | With SWE/seniority/YOE columns |
| `stage8_final.parquet` | 6-8 | Rule-based baseline with all enrichments |
| `stage9_control_cohort.parquet` | 9 | Control cohort selection |
| `stage9_llm_extraction_candidates.parquet` | 9 | Rows eligible for LLM extraction |
| `stage9_llm_extraction_results.parquet` | 9 | Raw LLM extraction outputs |
| `stage9_llm_cleaned.parquet` | 9 | Full dataset with `description_core_llm` |
| `stage10_llm_classification_results.parquet` | 10 | Raw LLM classification outputs |
| `stage10_llm_integrated.parquet` | 10 | Full dataset with all LLM columns |

---

## Known Limitations

- **Seniority unknown rate:** 71.8% in Stage 8 before LLM augmentation. Improves substantially with `seniority_llm` from Stage 10.
- **Entry-level historical data:** Only arshkon has native entry-level labels (~89 entry-level SWE postings). Asaniczka has zero.
- **Boilerplate removal accuracy:** ~44% for rule-based (`description_core`). LLM-based `description_core_llm` is the intended upgrade.
- **Ghost job detection:** Rule-based `ghost_job_risk` is very conservative (519 non-low out of 1.22M). LLM-based `ghost_assessment_llm` provides a richer signal.
- **No stop-at-stage flag:** `run_pipeline.py` runs all stages. For partial runs, invoke stage scripts directly.
- **Row count growth:** Each day of scraping adds ~34K rows before dedup. Re-run from Stage 1 after syncing new data.
