# Handoff: Preprocessing Pipeline Implementation & Validation

Date: 2026-03-21

This document is a handoff brief for a coding agent tasked with implementing, testing, and validating the preprocessing pipeline.

---

## Your task

Run the preprocessing pipeline stage by stage. At each stage: inspect inputs, run the stage, inspect outputs, validate data quality, and document what the stage actually does vs what it should do. Be deeply critical about data flow and data quality at every step.

**Do not blindly trust existing code.** The pipeline ran once on older data. The data sources, formats, and research questions have all changed since. Your job is to understand what each stage does, verify it works correctly on the current data, identify problems, and fix them.

---

## Project context

This is a research project studying how AI coding agents are restructuring SWE roles. We compare historical LinkedIn job postings (Kaggle datasets from 2024) with daily-scraped postings from March 2026 to measure changes in junior/senior role composition and requirements. Stage 1 is a pure ingest/schema-unification layer: it should normalize source schemas and provenance, but not decide occupation class or historical sample membership.

### Key documents (read these first)

1. `docs/1-research-design.md` — Canonical research design. Defines 4 research questions (RQ1-RQ4), core constructs, and empirical strategy. **Start here.**
2. `docs/plan-preprocessing.md` — The preprocessing plan (v3). Defines what each pipeline stage should do, the data inventory, schema mappings, and output specification. **This is your implementation spec.**
3. `docs/6-methods-learning.md` — Methods guidance (topic models, Fightin' Words, LLM annotation constraints).

### Research questions (from `1-research-design.md`)

- **RQ1:** Employer-side restructuring — junior share/volume, scope inflation, senior role redefinition
- **RQ2:** Task and requirement migration — which requirements moved between seniority levels
- **RQ3:** Employer-requirement / worker-usage divergence — do posting AI requirements outpace actual usage
- **RQ4:** Mechanisms — interview-based qualitative (not your concern for this task)

---

## Data sources

### Source 1: Kaggle arshkon (`data/kaggle-linkedin-jobs-2023-2024/`)

- `postings.csv`: 124K rows, April 5-20 2024, LinkedIn
- Companion files: `jobs/job_industries.csv`, `companies/companies.csv`, `companies/employee_counts.csv`, `mappings/industries.csv`
- Schema: `job_id`, `title`, `description` (inline), `formatted_experience_level` ("Mid-Senior level", "Entry level", etc.), `listed_time` (epoch ms), `company_name`, `company_id`, `location`, `formatted_work_type`
- Key strength: has entry-level labels (~385 entry-level SWE postings — the only historical source with them)
- ~2.1% of postings match SWE title patterns

### Source 2: Kaggle asaniczka (`data/kaggle-asaniczka-1.3m/`)

- `linkedin_job_postings.csv`: 1.35M rows, January 12-17 2024, LinkedIn
- `job_summary.csv`: descriptions in separate file, join on `job_link` (96.2% coverage)
- `job_skills.csv`: extracted skills, join on `job_link`
- Schema: `job_link` (URL as ID), `job_title`, `company`, `job_location`, `first_seen` (YYYY-MM-DD), `search_city`, `search_country`, `search_position`, `job_level` ("Mid senior"/"Associate" only), `job_type`
- Must filter to `search_country == "United States"` (86% of rows are US)
- **Critical gap:** NO entry-level seniority labels. Only "Mid senior" (17,045) and "Associate" (1,124) among US SWE matches
- NO salary, NO company size/industry metadata
- 18,169 US SWE title matches out of 1.15M US rows

### Source 3: Scraped data (`data/scraped/`)

- Load all matching scraped-date files in the current 41-column format. Do not hard-code a month/date ceiling.
- Ignore incompatible legacy files (for example Mar 5-18 old-format files with 35 columns and fewer results per query).
- Files: `YYYY-MM-DD_swe_jobs.csv` and `YYYY-MM-DD_non_swe_jobs.csv`
- ~3,680 SWE rows/day, ~30,888 non-SWE rows/day
- 41 columns including search metadata: `search_query`, `query_tier` (swe/adjacent/control), `search_metro_id`, `search_metro_name`, `search_metro_region`, `search_location`
- Platform-specific field availability:
  - LinkedIn: `job_level` 100%, `company_industry` 100%, `date_posted` 2.8%
  - Indeed: `date_posted` 100%, `company_num_employees` 91%, `job_level` 0%
- IDs are unique within each day. Cross-day overlap is ~40%.
- 100 results/query, 26 metros, 28 search queries

### NOT used

- YC data (ignore `*_yc_jobs.csv` and all YC state files)
- Apify data
- Old scraped format (Mar 5-18)

---

## Current pipeline state

### Pipeline runner
`preprocessing/run_pipeline.py` — runs all stages sequentially. Supports `--from-stage N` to resume.

### Scripts (in `preprocessing/scripts/`)

| Script | Stage | Purpose |
|---|---|---|
| `stage1_ingest.py` | 1 | Schema unification across 3 sources |
| `stage2_aggregators.py` | 2 | Aggregator/staffing company detection |
| `stage3_boilerplate.py` | 3 | Boilerplate removal from descriptions |
| `stage4_dedup.py` | 4 | Company canonicalization + exact/near-duplicate detection |
| `stage5_classification.py` | 5 | SWE / SWE-adjacent / control classification + seniority imputation |
| `stage678_normalize_temporal_flags.py` | 6-8 | Field normalization, temporal alignment, quality flags |
| `company_name_canonicalization.py` | helper | Company-name canonicalization utilities used by Stage 4 |
| `stage9_llm_prefilter.py` | 9 | LLM pre-filtering (NEW, may be incomplete) |
| `stage10_llm_classify.py` | 10 | LLM classification (NEW, may be incomplete) |
| `stage11_llm_integrate.py` | 11 | LLM response integration (NEW, may be incomplete) |
| `stage12_validation.py` | 12 | Three-way validation (NEW, may be incomplete) |
| `stage_final_output.py` | final | Produces `data/unified.parquet` |

### Intermediate outputs (from the last run)
Existing `.parquet` files in `preprocessing/intermediate/` are from the v1/v2 run on OLD data. They will need to be regenerated with the new data.

### Logs and validation reports
- `preprocessing/logs/` — stage logs, LLM validation report, decision log, manual review notes
- These are valuable context for understanding past decisions — read them

---

## What to do, stage by stage

### General approach for each stage

1. **Read the script** — understand what it does, what assumptions it makes
2. **Read `plan-preprocessing.md`** for what the stage SHOULD do — compare against implementation
3. **Run the stage** — check it completes without errors
4. **Inspect the output** — row counts, column values, distributions, null rates
5. **Validate quality** — spot-check random samples, check for data corruption, verify transformations
6. **Document findings** — what worked, what's wrong, what needs fixing

### Stage 1: Ingest (highest priority for changes)

This stage needs the most work because the data sources have changed:
- Asaniczka schema is completely different from what was originally coded (verify the join to `job_summary.csv` works, verify US filtering, verify `job_link` is used as ID)
- New scraped format has 6 extra columns (search metadata) that weren't in the old format
- Should load all current-format scraped files that are present, while skipping incompatible legacy files and YC
- Must NOT load YC files
- Historical Kaggle sources should be ingested without occupation filtering; keep approved source rows and defer occupation class decisions to Stage 5 and later analysis
- Preserve raw source seniority labels separately from the mapped canonical seniority field
- Preserve raw source descriptions separately from the normalized working description field
- Check: does the unified schema match what `plan-preprocessing.md` Stage 1d specifies?
- Check: are the asaniczka descriptions actually joining correctly? (96.2% expected coverage)
- Check: are seniority labels normalized consistently across all 3 sources?

### Stage 2: Aggregators

- Verify the AGGREGATOR_PATTERNS list catches the known aggregators (Lensa, Dice, Jobot, etc.)
- Check: what fraction of each dataset gets flagged? (Expected: ~9% of scraped, ~15% of Kaggle SWE)
- Spot-check: pull 10 flagged postings and 10 unflagged. Are the flags correct?

### Stage 3: Boilerplate

- The v1 boilerplate remover had ~44% accuracy (per LLM validation). It misses front-matter boilerplate.
- Check: compare `description` vs `description_core` on 20 random postings. Is the removal sensible?
- Check: are descriptions from different sources (arshkon, asaniczka, scraped) handled consistently?
- Measure: median character reduction rate per source

### Stage 4: Dedup

- Check that Stage 4 is canonicalizing `company_name_effective` into `company_name_canonical` before dedup decisions.
- Near-dedup is key-first and description-supported, with fuzzy title resolution inside company/location/description candidate sets.
- Check: what fraction of rows are removed? By source?
- Check: are cross-source duplicates being caught? (Same company+title posted on LinkedIn and Indeed)
- Check: is the dedup removing too aggressively or too leniently? Pull 10 pairs that were deduped and verify they're genuine duplicates.

### Stage 5: Classification

- Occupation classification: 3-tier (SWE / SWE-adjacent / control; regex → LLM lookup → embedding). The v1 LLM validation showed 64% raw agreement. Check current accuracy.
- Seniority imputation: multi-signal classifier. The v1 validation showed 32% agreement with LLM. This is known to be weak.
- Check: distribution of `is_swe`, `seniority_imputed` by source. Do they look reasonable?
- Check: what does the "unknown" seniority rate look like per source? (Was 21% in v2)

### Stages 6-8: Normalization, temporal, flags

- Check: are locations normalized consistently?
- Check: is `period` assignment correct (2024 vs 2026)?
- Check: ghost job risk flags — what fraction flagged? Is it reasonable?
- Check: language detection — how many non-English postings?

### Stages 9-12: LLM augmentation

These may be incomplete or untested. Assess their state:
- Can they run at all?
- If they have cached results from previous runs, are those still valid?
- Stage 10 extraction now uses a unit-ID contract: sentence-like units are labeled locally, and Stage 11 reconstructs `description_core_llm` from the selected unit IDs.
- Treat the LLM stages as a separate augmentation layer that still needs small-sample profiling and validation before any full-batch run.

### Final output

- Verify `data/unified.parquet` has the expected columns (see `plan-preprocessing.md` output specification)
- Verify `data/unified_observations.parquet` is produced (daily panel for scraped data)
- Run basic sanity checks: row counts by source, SWE counts, seniority distributions, null rates for key fields

---

## Critical things to watch for

1. **Data loss**: count rows at every stage boundary. Where do rows disappear? Is each drop justified?
2. **Schema drift**: are column types stable across stages? Do nulls appear where they shouldn't?
3. **Source bias**: do transformations affect one source differently than another? (e.g., boilerplate removal working well on arshkon but poorly on asaniczka)
4. **Seniority accuracy**: this is the most important classification for the research. Be extra critical.
5. **Memory**: 31GB RAM limit. All stages must use chunked pyarrow I/O. If any stage tries to load everything into a pandas DataFrame, it will OOM on the full dataset.
6. **Asaniczka description join**: this is new and tricky. The `job_summary.csv` file has 1.3M rows with multi-line descriptions (48M lines total). The join must be on `job_link` and should succeed for ~96% of rows.

---

## Environment

```bash
# Activate the venv
source .venv/bin/activate

# Run the full pipeline
python preprocessing/run_pipeline.py

# Run from a specific stage
python preprocessing/run_pipeline.py --from-stage 3

# Run individual stage
python preprocessing/scripts/stage1_ingest.py

# Quick data inspection
python3 -c "
import pyarrow.parquet as pq
f = pq.ParquetFile('preprocessing/intermediate/stage1_unified.parquet')
print(f'Rows: {f.metadata.num_rows:,}, Cols: {f.metadata.num_columns}')
"
```

---

## Output expectations

After this task, we should have:
1. A working pipeline that runs clean on the current data (3 sources, new scraped format only)
2. Documented findings for each stage — what works, what's broken, what needs improvement
3. A valid `data/unified.parquet` with correct schema
4. Identified issues ranked by severity for the next iteration
