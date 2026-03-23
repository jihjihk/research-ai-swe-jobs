# SWE Labor Market Research — Project Instructions

Last updated: 2026-03-23

## Project Charter

### Purpose

Research project studying how AI coding agents are restructuring SWE roles. The core comparison is historical LinkedIn postings from 2024 versus daily-scraped postings from 2026.

### Team Boundary

Two-person team:
- Partner owns scraper/infrastructure
- You own preprocessing, analysis, and related documentation in the analysis lane

### Work Areas

#### 1. Data preprocessing & analysis (`preprocessing/`, `notebooks/`, `data/`)

Pipeline that transforms raw data into analysis-ready datasets and supports statistical analysis.

- Plan docs: `docs/plan-preprocessing.md`, `docs/plan-exploration.md`, `docs/plan-analysis.md`
- Code: `preprocessing/scripts/`, `preprocessing/run_pipeline.py`, `notebooks/`
- Outputs: `data/unified.parquet`, `data/unified_observations.parquet`
- Do not touch: `scraper/`, research writing files

#### 2. Scraper & infrastructure (`scraper/`, `logs/`)

Daily scraping pipeline running on EC2.

- Docs: `docs/infrastructure-setup.md`, `docs/data-sources-and-prompts.md`
- Code: `scraper/scrape_linkedin_swe.py`, `scraper/harmonize.py`, `scraper/run_daily.sh`
- Do not touch: `preprocessing/`, `notebooks/`, research writing files

#### 3. Research writing (`research/`, `docs/1-*.md` through `docs/6-*.md`)

Academic writing, research design, literature review, interview protocol, methods, paper drafts.

- Instructions: `research/AMPLIFY.md`
- Scratch workspace: `research/`
- Canonical docs: `docs/1-research-design.md` through `docs/6-methods-learning.md`
- Do not touch: `preprocessing/`, `scraper/`, `notebooks/`

### Global Rules

- Read `docs/1-research-design.md` first. It defines RQ1-RQ4 and the empirical strategy.
- Use DuckDB for parquet/CSV inspection through the repo virtualenv. Avoid inline Python for data inspection.
- 31GB RAM limit. Use pyarrow chunked I/O for pipeline code. Never load full parquet into pandas unless the data volume is known to be safe.
- Do not overwrite large data artifacts unless the task requires it.
- After completing work, update this file if pipeline status, known issues, or priorities changed.

### Research Questions

- RQ1: Employer-side restructuring — junior share/volume, scope inflation, senior role redefinition
- RQ2: Task and requirement migration — which requirements moved between seniority levels
- RQ3: Employer-requirement / worker-usage divergence — do posting AI requirements outpace actual usage
- RQ4: Mechanisms — interview-based qualitative (reflexive thematic analysis)

### Architectural Rules

- Stage 1 is ingest, schema unification, provenance, and date handling only.
- Stage 1 should treat approved historical sources equivalently and remain extensible to future sources.
- Stage 1 must not define analytical occupation samples or filter rows by SWE/non-SWE/control class.
- Stage 2 owns aggregator detection, `real_employer` extraction, and `company_name_effective` derivation. It must not own company canonicalization for dedup.
- Stage 3 owns rule-based boilerplate removal only: it reads the working `description`, preserves it unchanged, and writes `description_core` plus Stage 3 quality flags.
- Stage 3 must not own company-name normalization, occupation classification, source-specific text rules, or row filtering.
- Stage 5 is the first occupation-classification boundary. `is_swe`, `is_swe_adjacent`, and `is_control` belong there.
- Analytical samples are defined after classification, in later preprocessing, validation, or analysis steps.
- Canonical postings and daily observations are both first-class outputs. Do not collapse them into one dataset.
- Stage 4 owns posting-level deduplication. Later stages must not silently change analytical row cardinality.
- Stage 4 owns canonicalization of `company_name_effective` into `company_name_canonical` for opening-level matching, plus exact duplicate removal, same-location near-duplicate resolution, and `is_multi_location` flagging for canonical postings.
- Stage 4 may use normalized company/title/location fields and description-derived support signals to make posting-level dedup decisions, but it must not redefine daily observations or analytical samples.
- Stage 4 must not take over Stage 1 ingest responsibilities, Stage 2 aggregator handling, or Stage 5 occupation classification.
- Stages 6-8 are row-preserving enrichment stages only. They may add normalization, temporal, quality, and provenance columns, but they must not silently change row cardinality or define analytical samples.
- Stages 9-12 own LLM augmentation only: prefiltering, LLM calling, integration, and validation.
- Stage 10 may deduplicate LLM calls by `description_hash` or another cache key to reduce API volume, but that is call deduplication only, not posting deduplication.
- Stage 10 intermediate outputs may be one row per unique cache key, as long as Stage 11 reattaches those outputs back to every matching posting row.
- Stage 11 must preserve the row count of its input posting table unless an explicit, separately documented filtering rule says otherwise.
- Stage 10/11 should never take over the responsibilities of Stage 4. If row counts change in the LLM stages, treat that as a bug unless the pipeline spec explicitly allows it.

### Key Docs

| Doc | Purpose |
|---|---|
| `docs/1-research-design.md` | Canonical research design |
| `docs/plan-preprocessing.md` | Preprocessing pipeline spec (Stages 1-12) |
| `docs/plan-exploration.md` | Validation + exploration spec (Stages 13-14) |
| `docs/plan-analysis.md` | Analysis + robustness spec (Stages 15-16) |
| `docs/handoff-preprocessing-implementation.md` | Task brief for pipeline work |
| `docs/6-methods-learning.md` | Methods guidance |
| `research/AMPLIFY.md` | Writing instructions |

### Data Source Policy

| Source | Rows | Period | Key strength | Key gap |
|---|---|---|---|---|
| Kaggle arshkon | 124K | April 2024 | Entry-level labels (~385 SWE) | Small SWE count (~3,466) |
| Kaggle asaniczka | 1.35M | Jan 2024 | Large volume (18K US SWE) | No entry-level labels |
| Scraped current-format files | ~3.7K SWE/day | March 2026 onward | Search metadata, 41 columns | Growing daily |

- Primary platform: LinkedIn only
- Indeed: sensitivity analyses only
- Do not use: YC data, Apify data, old scraped format (Mar 5-18)
- Sync fresh data with: `aws s3 sync s3://swe-labor-research/scraped/ data/scraped/`

## Current State

### Repo Structure

```text
├── AGENTS.md
├── docs/
├── preprocessing/
│   ├── run_pipeline.py
│   ├── scripts/
│   ├── intermediate/
│   └── logs/
├── scraper/
├── research/
├── data/
└── notebooks/
```

### Pipeline Status

| Stage | Script | Status | Notes |
|---|---|---|---|
| 1 | `stage1_ingest.py` | Revalidated on current data | Pure ingest/schema unification; raw vs mapped seniority preserved; raw descriptions preserved; current-format scraped files loaded, legacy files skipped |
| 2 | `stage2_aggregators.py` | Revalidated on current data | Raw `description` preserved; expanded aggregator list/patterns on current sources; derives `company_name_effective` |
| 3 | `stage3_boilerplate.py` | Needs improvement | ~44% accuracy |
| 4 | `stage4_dedup.py` | Rebuilt on current data | Canonicalizes `company_name_effective` and applies description-supported key-first dedup; needs validation sampling |
| 5 | `stage5_classification.py` | Reworked on current data; still needs validation | SWE title-key contract aligned to `title_normalized`; title lookup artifact validated/deduped; description-based fallback tightened; `seniority_final` now applies strong-title/native-backfill/title-prior resolution for SWE-adjacent and control rows; core seniority still needs follow-up calibration |
| 6-8 | `stage678_normalize_temporal_flags.py` | Revalidated on current data | Row-preserving field normalization, temporal alignment, and quality/provenance flags; `posting_age_days` computed here |
| 9-12 | LLM stages | Stage 9-11 reworked and wired into orchestration; still needs larger validation | Stage 9 now does fixed-universe forward routing with control extraction by default and technical-corpus classification skip logic; Stage 10/11 honor task-specific routing and Stage 11 feeds the final canonical output path; Stage 12 remains scaffolded / smoke-tested rather than production-validated |
| final | `stage_final_output.py` | Revalidated on current data | Produces `unified.parquet` and `unified_observations.parquet` |

Current `unified.parquet` and `unified_observations.parquet`: rebuilt on current sources after the Stage 1 redesign and salary-field removal.

### Known Issues

- [ ] Asaniczka has no entry-level seniority labels
- [ ] Seniority classifier accuracy is poor (32%)
- [ ] Boilerplate removal accuracy is poor (~44%)
- [ ] Stage 4 was rebuilt with description-supported dedup logic and still needs validation sampling on real pairs
- [ ] LLM stages (9-12) still need broader validation on real data before a full-batch run
- [ ] Stage 12 validation remains scaffolded / smoke-tested and is not yet part of the default runner

### Priorities

1. Preprocessing pipeline — stage by stage
2. Exploration/validation (Stages 13-14)
3. Analysis (Stages 15-16)
