# SWE Labor Market Research — Project Instructions

Last updated: 2026-03-27

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
- Use test-driven development for preprocessing and analysis-lane code whenever practical: write or update the smallest high-signal failing test first, then implement the code change, then rerun the targeted tests before broader validation.
- Use DuckDB for parquet/CSV inspection through the repo virtualenv. Avoid inline Python for data inspection.
- 31GB RAM limit. Use pyarrow chunked I/O for pipeline code. Never load full parquet into pandas unless the data volume is known to be safe.
- Do not overwrite large data artifacts unless the task requires it.
- Reader-facing pipeline outputs should be written to sibling temp files and atomically promoted only after the write completes; do not stream long-running writes directly to final artifact paths.
- Treat the `tests/` suite as part of the preprocessing contract. Logic changes, schema-boundary changes, and dependency-seam changes should land with matching test updates.
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
| `docs/testing/preprocessing-test-strategy.md` | Testing strategy and stage coverage model |
| `docs/testing/test-implementation-workpacks.md` | TDD execution playbook for stage test implementation |
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
| 2 | `stage2_aggregators.py` | Updated 2026-03-23 | Added Turing to aggregator list with negative lookahead for Pharmaceutical/Medical; derives `company_name_effective` |
| 3 | `stage3_boilerplate.py` | Updated 2026-03-23 | Expanded header (16 new), EEO (10 new), and benefits (22 new) patterns; ~44% accuracy still limited by rule-based approach; LLM version will supersede |
| 4 | `stage4_dedup.py` | Rebuilt on current data | Canonicalizes `company_name_effective` and applies description-supported key-first dedup; needs validation sampling |
| 5 | `stage5_classification.py` | Updated 2026-03-27 | YOE extractor now uses raw `description`, clause-aware section segmentation, richer candidate tagging, revised resolver hierarchy, audit fields (`yoe_min_extracted`, `yoe_max_extracted`, `yoe_match_count`, `yoe_resolution_rule`, `yoe_all_mentions_json`), and stronger false-positive guards; title-side SWE recall was also widened for technical-family misses (`network/security/data-science/admin/cloud/SAP/VMware/Salesforce/ServiceNow/.NET/mainframe` patterns plus adjacent-family rescue); fixed YOE regressions now live in `tests/test_stage5_yoe_extractor.py`; remaining validation risk is the long tail of mixed multi-mention rows, hard-negative tuning, and broader SWE-vs-adjacent recall/precision sampling |
| 6-8 | `stage678_normalize_temporal_flags.py` | Updated 2026-03-27 | Row-preserving field normalization, temporal alignment, and quality/provenance flags; `posting_age_days` computed here; `date_flag` is now a permissive sanity check (parseable dates with a 2020 floor, no future-date or scrape-window enforcement on `date_posted`); `ghost_job_risk` now keys off canonical `seniority_final` rather than the coarse 3-level bucket; rerun after the 2026-03-26 Stage 5 YOE changes if refreshed `stage8_final.parquet` is needed |
| 9-12 | LLM stages | Updated 2026-03-29; larger validation still needed | Stage 9 now owns the LLM analysis universe, deterministic control-cohort selection, extraction-only routing/execution, short-description hard skips, and the posting-level cleaned-text contract (`description_core_llm`, `selected_for_control_cohort`). Stage 10 now owns classification routing/execution on the cleaned-description fallback chain, task-specific cache keys (`input_hash`), and the canonical row-preserving posting artifact at `stage10_llm_integrated.parquet`. The cache DB remains `preprocessing/cache/llm_responses.db`; Codex is pinned to `gpt-5.4-mini`, Claude to `haiku`; Stage 9/10 now use shared engine scheduling via `--engines` plus `--engine-tiers`, with provider-scoped quota pauses, no cross-engine fallback, same-engine retry after 60 seconds on non-quota failures, and a default `--max-workers 30`. The temporary remote-CLI SSH hop now prewarms a single control master and reuses it via a hashed control socket path, which avoids the earlier multiplexing race under concurrent worker startup. Stage 11 is now compatibility-only at most, not the architectural posting boundary. Stage 12 remains scaffolded / smoke-tested rather than production-validated |
| final | `stage_final_output.py` | Revalidated on current data | Produces `unified.parquet` and `unified_observations.parquet` |

Current `unified.parquet` and `unified_observations.parquet`: rebuilt on current sources after the Stage 1 redesign and salary-field removal.

Operational note for future LLM reruns: rerun Stage 9 first, then run Stage 10 directly with only the runtime controls that matter operationally (for example `--engines codex,claude --engine-tiers codex=full,claude=non_intrusive --quota-wait-hours 5 --max-workers 30`). Model selection is fixed in code: Codex uses `gpt-5.4-mini` and Claude uses `haiku`. The durable checkpoint is `preprocessing/cache/llm_responses.db`; rerunning Stage 10 reuses completed `(input_hash, task_name, prompt_version)` cache entries even if the prior run stopped before writing the final Stage 10 parquet outputs.

### Known Issues

- [ ] RQ1 seniority direction not robust until seniority_llm runs (T02)
- [ ] Asaniczka has no entry-level seniority labels
- [ ] Seniority classifier recall: entry 30-61%, associate 8-15%, director 17% (T02)
- [ ] SWE title_lookup_llm tier still has elevated FP rate; needs swe_classification_llm (T04)
- [ ] Stage 5 rule-based technical-corpus recall was expanded on 2026-03-27, but a broader false-negative / false-positive validation pass is still needed before treating the new SWE-adjacent boundary as production-clean
- [ ] Rule-based boilerplate ~44% accuracy; LLM extraction/classification sequence will supersede it once the redesigned Stage 9-10 flow is validated
- [ ] YOE extractor substantially improved on 2026-03-26, but mixed multi-mention rows and clause-local hard negatives still need targeted validation before calling it production-clean
- [ ] Stage 4 dedup needs validation sampling on real pairs
- [ ] `preprocessing/intermediate/stage5_classification.parquet` is currently unreadable (`No magic bytes found at end of file`); add artifact-health checks before trusting downstream stages
- [ ] LLM stages (9-12) need broader validation before full-batch run
- [ ] Stage 9/10 artifacts need a fresh end-to-end rerun after the sequence redesign; current final outputs still look rule-based rather than LLM-augmented
- [ ] ghost_job_risk too conservative (354 non-low); needs LLM ghost_assessment_llm (T15)
- [ ] "Jobs via Dice" (130 rows): check if `real_employer` was extracted (T13)

### Exploration Phase Status

**All exploration tasks (T01-T20) are complete** as of 2026-03-24.

T25-T26 handoff update on 2026-03-24: interview elicitation artifacts were added under `exploration/artifacts/` and the consolidated analysis handoff is `exploration/reports/SYNTHESIS.md`.

Post-gate repair on 2026-03-24: T10 and T11 were rerun with `description_core`-first text, SWE-only T11 counters, and cleaned term filtering. The blocker in `exploration/reports/INDEX.md` was cleared.

- Wave 1 (T01-T07): Data audit and validation -- done
- Wave 2 (T08-T16): Exploratory analysis -- done
- Wave 3 (T17-T18): Interview artifacts and synthesis -- done

**Handoff document:** `exploration/reports/SYNTHESIS.md` -- read this first for analysis phase.

**Interview artifacts:** `exploration/artifacts/` -- 5 artifacts for RQ4 interviews.

### Priorities

1. **Fix Stage 5 native_backfill bug** (blocking for analysis; workaround available)
2. **Run seniority_llm** (Stage 10) to resolve seniority direction disagreement
3. **Run swe_classification_llm** (Stage 10) to reduce 26% FP in title_lookup_llm tier
4. **Analysis phase** (Stages 15-16) -- read `exploration/reports/SYNTHESIS.md` first
