# Preprocessing Test Strategy

Date: 2026-03-27
Owner: analysis lane
Scope: `preprocessing/` stages 1-12 plus final output publish step

## Goal

Build a test framework that catches:

- logic regressions inside each stage
- schema drift between stages
- row-count and cardinality contract violations between stages
- bad sampled decisions from real data, not just synthetic toy cases
- external dependency failures in embedding, DuckDB, SQLite, langdetect, and LLM-provider code

The target is not "100% unit coverage". The target is a fast, opinionated suite that blocks bad preprocessing publishes.

## Development Mode

The default development mode for preprocessing changes is test-driven development:

- write or update the smallest high-signal test that captures the intended behavior
- confirm it fails for the right reason when practical
- implement the code change
- rerun the narrowest relevant test selection first, then widen only after the local contract is green

For this repo, "high-signal" means logic assertions, schema/cardinality contracts, and reviewed sampled-row expectations, not broad snapshot churn.

## Scope Boundary

This strategy is intentionally not centered on the health of whatever large parquet outputs happen to exist in the workspace today. The core suite should be driven by:

- pure function behavior
- schema and row-cardinality contracts between stages
- small synthetic fixtures that target edge branches
- sampled real rows whose expected outcomes have been manually reviewed
- dependency seams where code relies on DuckDB, SQLite, langdetect, embeddings, or LLM-provider subprocess wrappers

Current large artifacts can still be used to identify good sampled rows, but they are not the main object under test.

## Framework Recommendation

Use `pytest` as the runner when implementing the full suite. The repo currently has only one `unittest` file and no `pytest` in the venv, but the planned suite needs parametrization, tmp-path fixtures, monkeypatching, and selective slow/integration markers. If the team refuses a new dev dependency, keep helpers pure enough that the same tests can be adapted to `unittest`, but `pytest` is the right default.

### Proposed layout

```text
tests/
  conftest.py
  helpers/
    parquet_asserts.py
    stage_runner.py
    sqlite_asserts.py
    llm_fakes.py
    fixture_extractors.py
  fixtures/
    sampled/
      manifest.json
      stage1/
      stage2/
      stage3/
      stage4/
      stage5/
      stage678/
      stage9/
      stage10/
      stage11/
      final/
    synthetic/
      stage1/
      stage2/
      stage3/
      stage4/
      stage5/
      stage678/
      stage9/
      stage10/
      stage11/
      stage12/
      final/
  test_pipeline_smoke.py
  test_runner_contracts.py
  test_stage1_ingest.py
  test_stage2_aggregators.py
  test_stage3_boilerplate.py
  test_stage4_dedup.py
  test_stage5_classification.py
  test_stage678_normalize_temporal_flags.py
  test_stage9_llm_prefilter.py
  test_stage10_llm_classify.py
  test_stage11_llm_integrate.py
  test_stage12_validation.py
  test_stage_final_output.py
```

### Test layers

1. `pure-unit`
   - Regexes, normalizers, resolvers, validators, quota detection, hash functions.
   - No file IO.

2. `golden-fixture`
   - Tiny synthetic input files with exact expected outputs.
   - Used for branch coverage and stage-boundary contracts.

3. `sampled-fixture`
   - Real rows extracted from current intermediate outputs after manual review.
   - Each sampled row gets a checked-in expected result file.

4. `stage-integration`
   - Run one stage end-to-end on a tiny temp fixture corpus.
   - Verify schema, row counts, null rates, and key outputs.

5. `publish-gate`
   - Final contract checks on fixture-driven outputs.
   - Schema expectations, row-preservation, and final-output compatibility.

### Marker policy

- `unit`: pure logic, default in local and CI.
- `fixture`: uses checked-in small parquet/csv fixtures.
- `integration`: stage-level temp outputs.
- `sampled`: uses reviewed real-data fixtures extracted from live artifacts.
- `slow`: sentence-transformers or larger fixture builds.

## Fixture Strategy

Use two fixture families for every stage.

### 1. Synthetic targeted fixtures

Purpose:

- hit each branch in the decision flow
- lock expected behavior at exact edge conditions
- keep failures easy to diagnose

Rules:

- keep them tiny
- build them from raw csv/parquet/json created under `tests/fixtures/synthetic/...`
- one fixture should target one logic idea
- expected outputs should live beside the input in `expected.json` or `expected.parquet`

### 2. Sampled real-data fixtures

Purpose:

- test messy real inputs the code actually sees
- convert manual review into regression protection

Rules:

- sub-agent uses DuckDB to identify candidate rows
- reviewer reads the raw input and stage output manually
- reviewer writes down the expected result and why
- snapshot that raw row or small row group into `tests/fixtures/sampled/<stage>/...`
- add the source artifact path, query, row id, and review date to `manifest.json`

### Sampled fixture candidates already identified

Stage 1:

- `asaniczka_70c0f7742486e930`: missing description after `job_summary` join
- `linkedin_li-3736674638`: scraped `seniority_raw='not applicable'` should map to native null
- `asaniczka_6664e88d0e35c7dd`: typical asaniczka row with populated `search_position` and `search_city`

Stage 2:

- Positive extraction candidates:
  - `arshkon_3887888058`: `Dice -> Avenues International`
  - `arshkon_3888032530`: `Dice -> Beacon Systems`
  - `arshkon_3888937982`: `Dice -> Maxar Technologies`
- Negative extraction candidates:
  - `arshkon_3884435199`: `Capgemini -> Vendors` should be rejected
  - `arshkon_3884801360`: `Insight Global -> Project Management` should be rejected
  - `arshkon_3885824903`: `TEKsystems -> a CHANGE AGENT that will be` should be rejected

Stage 3:

- `arshkon_921716`: over-removed sample
- `arshkon_3190494363`: `too_short`/empty-core style sample
- `arshkon_3884434746`: company/about/apply heavy description

Stage 4:

- Multi-location positives:
  - `arshkon_2147609789`
  - `arshkon_3884429858`
  - `arshkon_3884429859`
- Canonicalization artifact has method counts:
  - `passthrough=85,950`
  - `exact_normalized=1,407`
  - `fuzzy=276`
  - `alias=12`

Stage 5:

- Use the fixed YOE regression cases in `tests/test_stage5_yoe_extractor.py`
- Add sampled rows once a reviewed Stage 5 fixture corpus is assembled
- Candidate downstream rows showing suspicious YOE/junior interactions:
  - `arshkon_3884437640`
  - `arshkon_3884439345`
  - `arshkon_3884443954`

Stage 6-8:

- `arshkon_3901965000`: `date_posted_out_of_range`
- `indeed_in-00065fb90e634182`: old Indeed posting date vs scrape date
- `arshkon_3877509537`: high ghost risk
- `arshkon_3190494363`: `description_quality_flag='too_short'`

Stage 9:

- Candidate dedup example:
  - description hash `4585f85027d16b1de46db0ab1ecfb4ca000bedde16bfd24d06e40d0dea05333b`
  - title `Network Engineer`
  - company `Epic`
  - `source_row_count=221`
- Routed examples:
  - hash `87d2c6c7...` `Software Engineer`
  - hash `4d1c8d59...` `Data Architect` extraction only
- Control skip examples:
  - `arshkon_263583866` `Registered Nurse`
  - `arshkon_3619548798` `Senior Mechanical Engineer`

Stage 10-11:

- Reuse checked-in eval artifacts:
  - `preprocessing/intermediate/stage10_test_results.parquet`
  - `preprocessing/intermediate/stage11_unit_eval_12_v2.parquet`
- Cache DB statistics currently visible:
  - `responses` rows: `35,653`
  - main prompt versions:
    - classification: `11,922`
    - extraction: `23,731`

Final:

- Final observations source counts currently are:
  - `kaggle_arshkon=109,885`
  - `kaggle_asaniczka=911,633`
  - `scraped=43,313`

## Core Helpers To Build

### `tests/helpers/parquet_asserts.py`

Provide helpers for:

- `assert_parquet_readable(path)`
- `assert_row_count_equal(path_a, path_b)`
- `assert_row_count_leq(path_a, path_b)`
- `assert_has_columns(path, required_cols)`
- `assert_missing_columns(path, forbidden_cols)`
- `assert_unique(path, cols)`
- `assert_all_boolean_mask(path, expr, expected=True)`

### `tests/helpers/stage_runner.py`

Provide temp-dir stage execution wrappers so tests can run stage code on tiny fixtures without touching repo-scale outputs.

Pattern:

- monkeypatch module-level paths (`INTERMEDIATE_DIR`, `DATA_DIR`, `LOG_DIR`, etc.)
- create temp input/output paths
- call `run_stageX()` or `main()`
- assert output exists and is readable

### `tests/helpers/fixture_extractors.py`

One script to snapshot sampled rows out of available source/intermediate artifacts:

- must use DuckDB via `./.venv/bin/python`
- writes tiny parquet/csv fixture files
- writes a `manifest.json` entry with:
  - stage
  - source artifact
  - query
  - identifier
  - expected behavior summary
  - reviewer initials/date

### `tests/helpers/llm_fakes.py`

Provide:

- fake subprocess results for Codex/Claude
- fake sqlite cache builders
- payload factories for valid and invalid classification/extraction responses
- deterministic quota/rate-limit stderr samples

## Stage-by-Stage Test Matrix

## Stage 1: Ingest and Schema Unification

### Boundary

Stage 1 owns source ingest, schema normalization, provenance, and observation-vs-canonical split. It must not classify SWE/control rows.

### Key code paths to test

- `map_seniority`
- `parse_company_size`
- `normalize_date_series`
- `finalize_frame`
- `load_kaggle_arshkon`
- `load_kaggle_asaniczka`
- `list_scraped_files`
- `load_scraped`

### High-value synthetic tests

- seniority mapping:
  - `Entry level -> entry`
  - `Mid senior -> mid-senior`
  - `Not Applicable -> null`
  - unknown labels stay null
- company size parsing:
  - `"1,001-5,000"` midpoint
  - `"10,001+"` exact floor
  - empty / garbage -> NaN
- scraped file policy:
  - load `YYYY-MM-DD_{swe,non_swe}_jobs.csv`
  - skip YC files
  - skip wrong filename
  - skip wrong column count
- canonical vs observations:
  - repeated scraped `id` across scrape dates should collapse in unified but not observations
- stage boundary:
  - no `is_swe`, `is_control`, `is_swe_adjacent` columns written

### Sample-derived tests

- asaniczka row with missing summary remains present and has null description
- scraped row with `seniority_raw='not applicable'` yields null `seniority_native`
- asaniczka row preserves `search_position -> search_query` and `search_city -> search_metro_name`

### Contract tests

- `stage1_unified` row count equals `arshkon + asaniczka_us + scraped_unique_ids`
- `stage1_observations` row count equals `arshkon + asaniczka_us + scraped_observation_rows`
- required columns exactly match `OUTPUT_COLUMNS`
- `job_id == uid` contract holds

### Plan-vs-code risk to pin down

- current scraped validation only checks `len(columns) == 41`, not column names; add a failing test for a wrong 41-column schema

## Stage 2: Aggregators

### Boundary

Owns aggregator detection, `real_employer`, and `company_name_effective`. Must be row-preserving.

### Key code paths to test

- `is_aggregator`
- `_looks_like_organization_name`
- `extract_real_employer`
- `company_name_effective` fallback logic

### High-value synthetic tests

- exact aggregator matches: `Dice`, `Revature`, `Jobs via Dice`
- pattern matches: `Net2Source Inc.`, `Turing`
- negative lookahead: `Turing Pharmaceutical` should not flag
- employer extraction:
  - `our client, Maxar Technologies, is...` -> `Maxar Technologies`
  - `on behalf of Beacon Systems` -> `Beacon Systems`
  - reject role phrases, generic fragments, same-as-aggregator labels
- row preservation: output rows == input rows

### Sample-derived tests

- lock in positive real-employer samples from Dice rows
- add regression negatives for current bad extractions:
  - `Vendors`
  - `Project Management`
  - `a CHANGE AGENT that will be`

### Contract tests

- `company_name_effective` non-null whenever `company_name` is non-null
- non-aggregator rows must have `company_name_effective == company_name`
- stage adds only Stage 2 columns, does not drop rows

## Stage 3: Boilerplate Removal

### Boundary

Owns only text cleanup from `description` to `description_core` and Stage 3 quality flags. No row filtering.

### Key code paths to test

- `_strip_common_noise`
- `_is_boilerplate_paragraph`
- `_strip_eeo_safely`
- `extract_core`
- `process_chunk`

### High-value synthetic tests

- remove `Show more / Show less`
- remove standalone URL lines
- remove EEO paragraphs but keep substantive single-paragraph content
- keep responsibilities/requirements sections
- fallback path when no headers exist
- `boilerplate_flag` thresholds:
  - `over_removed`
  - `under_removed`
  - `empty_core`

### Sample-derived tests

- `arshkon_921716` remains readable but is flagged `over_removed`
- `arshkon_3190494363` becomes very short and should stay non-empty but flagged
- `arshkon_3884434746` should not collapse to empty because of apply links / company intro

### Contract tests

- output row count == input row count
- original `description` unchanged
- `core_length == len(description_core)` on non-null rows
- `boilerplate_flag` values restricted to known enum

### Plan-vs-code risk to pin down

- plan says "Stage 3 quality flags"; current code emits `boilerplate_flag`, not `description_quality_flag`

## Stage 4: Canonicalization and Dedup

### Boundary

Owns company canonicalization, exact dedup, same-location near-dup resolution, and multi-location flagging. This is the first stage allowed to change canonical row cardinality.

### Key code paths to test

- company normalization helpers
- title near-dup logic
- lookup build and passthrough fallback
- exact `job_id` dedup
- exact opening-key dedup
- fuzzy same-company/location dedup
- multi-location flagging

### High-value synthetic tests

- exact duplicate `job_id` drops later copy
- same `(company,title,location,desc_hash)` drops later copy
- near-duplicate titles with same desc hash:
  - keep more complete row
  - tie-break on longer description
- meaningful title token changes should not dedup:
  - `mechanical engineer` vs `electrical engineer`
  - `senior` vs `junior`
- multi-location same opening across 2 locations should keep both and flag both

### Sample-derived tests

- sampled multi-location rows already identified should assert `is_multi_location=True`
- sampled company lookup rows should cover `passthrough`, `exact_normalized`, and `fuzzy`

### Contract tests

- output rows < input rows
- output schema includes `company_name_canonical`, `company_name_canonical_method`, `is_multi_location`
- kept `uid`s are unique
- Stage 4 does not introduce classification columns

### Plan-vs-code risk to pin down

- no explicit duplicate-type audit column exists; if reviewers want that auditability, add it later, but tests should at least preserve funnel counts

## Stage 5: Classification

### Boundary

Owns first occupation-classification boundary plus seniority and YOE extraction.

### Key code paths to test

- SWE title normalization and regex tiers
- title-lookup artifact validation
- family inference and mutual exclusivity
- `resolve_seniority_final`
- YOE extraction pipeline
- streaming write schema

### High-value synthetic tests

- extend existing title tests for:
  - primary SWE
  - adjacent technical
  - control titles
- title-lookup artifact failures:
  - missing columns
  - invalid labels
  - conflicting normalized keys
- native backfill decision flow:
  - strong title beats native
  - native backfill wins over unknown
  - title prior only when imputed is unknown
- mutual exclusion: SWE > adjacent > control
- YOE regression cases from existing JSON
- stage-integration test: on temp fixture input, output parquet must be readable

### Sample-derived tests

- freeze sampled rows around:
  - junior + high YOE contradiction
  - title-only false positives
  - adjacent-family rescue titles

### Contract tests

- row count preserved from Stage 4
- expected columns written, including:
  - `is_swe`
  - `is_swe_adjacent`
  - `is_control`
  - `seniority_final`
  - `yoe_min_extracted`
  - `yoe_max_extracted`
  - `yoe_match_count`
  - `yoe_resolution_rule`
  - `yoe_all_mentions_json`

### Integration contract test

- on a temp Stage 4 fixture run, Stage 5 output must be readable and expose the new audit columns

## Stages 6-8: Normalize, Temporal, Flags

### Boundary

Row-preserving enrichment only.

### Key code paths to test

- `normalize_location`
- `infer_metro`
- `validate_dates`
- `detect_language`
- `detect_ghost_job`
- `assess_description_quality`
- `derive_period`

### High-value synthetic tests

- US city/state parsing
- full state name parsing
- non-US parsing
- remote inference
- metro resolution priority:
  - search metro
  - manual alias
  - city/state lookup
  - reference lookup
  - unresolved
- date range checks by source/platform
- language detection for high ASCII vs low ASCII text
- ghost job risk thresholds
- description quality enum

### Sample-derived tests

- `arshkon_3901965000` should keep `date_flag='date_posted_out_of_range'`
- `indeed_in-00065fb90e634182` should also flag date range
- `arshkon_3877509537` should be `ghost_job_risk='high'`
- `arshkon_3190494363` should be `description_quality_flag='too_short'`

### Contract tests

- row count preserved from Stage 5
- Stage 8 output contains metro columns
- final publish gate must assert those columns are not silently lost unless explicitly documented

### Plan-vs-code risks to pin down

- add a regression test for the current scraped-date behavior: `date_posted` can predate scrape collection and should not automatically be treated like `scrape_date`
- add explicit tests for short low-information descriptions (`tbd`, `PI239170402`) so language behavior is deliberate rather than accidental

## Stage 9: LLM Prefilter

### Boundary

Owns LLM analysis-universe selection, deterministic control-cohort selection, short-description hard skips, extraction routing/execution, and the posting-level cleaned-text contract. It must preserve posting rows while deduplicating extraction calls by extraction-specific `input_hash`.

### Key code paths to test

- `english_mask`
- `has_raw_description`
- `process_chunk`
- control-cohort allocation / deterministic scoring
- extraction candidate build and cleaned-table write path

### High-value synthetic tests

- routing universe:
  - only LinkedIn
  - only English
  - requires raw description
- technical corpus plus selected control cohort
- short-description hard skip writes `description_core_llm=''`
- deterministic control selection is stable across reruns
- extraction input hash uses title + company + raw description

### Sample-derived tests

- routed SWE examples from current Stage 9 artifact
- selected control-cohort examples from stable time buckets
- duplicate raw-description reuse cases for extraction input hashes

### Contract tests

- `stage9_llm_cleaned` row count equals Stage 8 input row count
- extraction results join back row-preservingly into `stage9_llm_cleaned`
- selected controls are monotone/stable for a fixed bucket target
- short-description skips are excluded from downstream classification candidates

### Plan-vs-code risk to pin down

- control-bucket redistribution must stay deterministic when some buckets lack enough controls
- Stage 9 must not silently persist a posting-level classification-input column; Stage 10 should derive that transiently

## Stage 10: LLM Task Execution

### Boundary

Owns classification routing, provider calls, classification caching, retries, final posting-level integration, and the canonical LLM artifact. It may dedup calls by classification-specific `input_hash` only.

### Key code paths to test

- payload validators
- cache schema and fetch/store
- engine-list parsing
- quota/rate-limit detection and pause logic
- same-engine retry behavior without cross-engine fallback
- synthetic fallbacks
- classifier-input fallback chain
- final posting-level integration

### High-value synthetic tests

- valid classification payload accepted
- malformed JSON / missing keys rejected
- cache hit skips subprocess call
- quota text triggers provider-scoped pause logic
- duplicate classification inputs collapse before execution
- classifier input falls back `description_core_llm -> description_core -> description`
- high-confidence technical rows are skipped while selected controls remain eligible

### External dependency strategy

- never call real providers in default tests
- stub `call_subprocess`
- use temp sqlite DB
- patch `time.sleep`
- patch model commands and stdout parsers

### Plan-vs-code risks to pin down

- cached rows must be revalidated on read, not just on write
- final integrated output must preserve Stage 9 row count exactly
- compatibility alias behavior, if retained, must not become the canonical runner dependency

## Stage 11: LLM Integration

### Boundary

Deprecated as an architectural boundary. If retained, Stage 11 is compatibility-only and must behave as a thin alias/shim over the Stage 10 integrated artifact without changing row count or schema semantics.

### Key code paths to test

- compatibility alias generation
- row-preserving copy semantics
- no new business logic

### High-value synthetic tests

- optional alias path exists only when explicitly requested
- alias output matches the Stage 10 integrated artifact bit-for-bit or schema-for-schema

### Sample-derived tests

- use small Stage 10 integrated fixtures if the alias path is kept

### Contract tests

- if the shim exists, output row count == Stage 10 integrated row count exactly
- if the shim exists, it must not mutate any non-LLM columns

## Stage 12: Validation

### Boundary

Produces validation sample and report only. Should not mutate main outputs.

### Key code paths to test

- disagreement detection
- stratified sampling reproducibility
- pairwise metrics / kappa helpers
- markdown report build

### High-value synthetic tests

- disagreement flag across SWE/seniority/ghost/boilerplate channels
- `stratified_sample` determinism with fixed seed
- no duplicate `job_id`s in sample
- report generation with missing LLM fields still produces markdown

### Plan-vs-code risks to pin down

- add a regression test for the missing `quota_wait_hours` argument in `run_full_model()`
- add a fail-fast schema test because current `data/unified.parquet` lacks the LLM columns Stage 12 expects by default

### External dependency strategy

- default tests stub `try_provider`
- no live-provider execution in the core suite

## Final Output Stage

### Boundary

Copies final canonical table and rebuilds observation panel by joining Stage 1 observations to the canonical uid set.

### High-value tests

- fail fast if the declared final-stage input for a fixture path is missing
- `build_unified_observations` keeps exactly the columns of unified output, with `scrape_date` sourced from observations
- observation row count equals join cardinality on `uid`
- `quality_report.json` fields align with actual parquet counts

### Publish-gate checks

- if final unified for a fixture-driven LLM path omits Stage 10 columns that the runner path claims to produce, fail
- if final unified drops Stage 8 metro columns without an explicit contract update, fail

## Runner and Publish-Gate Tests

Add `tests/test_pipeline_smoke.py` and `tests/test_runner_contracts.py`.

Required checks:

- runner `check_col` expectations reflect real schemas
- runner stage ordering and wrapper paths reflect the intended orchestration
- fixture-driven final publish gate checks final outputs against current stage contracts

Note: `run_pipeline.py` still describes itself as `V2`. That is cosmetic but should be covered by a low-priority consistency cleanup task, not a blocking test.

## Sub-Agent Implementation Briefs

The environment here does not expose a literal sub-agent spawn tool, so the right deliverable is a set of ready-to-send implementation briefs. Each assignee must use `pipeline-stage-review` first and `parquet-duckdb` for data sampling.

### Shared instructions for every stage implementer

Use this exact checklist:

1. Read only:
   - `docs/1-research-design.md`
   - `docs/plan-preprocessing.md`
   - the target stage script
   - `preprocessing/run_pipeline.py` only if orchestration matters
2. Build a short stage card:
   - purpose
   - inputs
   - outputs
   - columns created/changed
   - what the stage must not own
3. Compare plan vs code and write down every mismatch that changes test design.
4. Use DuckDB through `./.venv/bin/python` to:
   - inspect row counts
   - inspect null rates
   - sample random rows
   - identify 3-5 real rows that should become deterministic fixtures
5. Propose:
   - pure unit tests
   - synthetic edge-case tests
   - sampled fixture tests
   - row-count/schema contract tests
   - any external dependency test doubles needed
6. Before freezing sampled expectations, manually inspect the row and write down why the expected result is correct.

### Stage 1 implementer brief

- Focus on source ingest correctness and canonical-vs-observation split.
- You must produce:
  - mini arshkon fixture
  - mini asaniczka fixture with missing summary row
  - mini scraped multi-day fixture with repeated ids and one invalid 41-column file

### Stage 2 implementer brief

- Focus on precision, not recall, for `real_employer`.
- You must convert the bad current extractions into explicit negative tests.
- Add at least 3 positive and 3 negative sampled fixtures.

### Stage 3 implementer brief

- Focus on header classification, EEO stripping, fallback behavior, and non-destructive handling of short descriptions.
- Add sampled tests for `over_removed` and `too_short` style rows.

### Stage 4 implementer brief

- Focus on exact dedup, near-dedup, canonicalization, and multi-location behavior.
- You must create at least:
  - one exact-duplicate fixture
  - one near-duplicate keep/drop fixture
  - one "same company but meaningfully different title" negative fixture
  - one multi-location keep-both fixture

### Stage 5 implementer brief

- Focus on decision-flow coverage for SWE, adjacent, control, seniority resolution, and YOE extraction.
- Reuse and expand `tests/test_stage5_yoe_extractor.py`.
- Add a temp-fixture integration test first, because Stage 5 writes the widest contract surface.

### Stage 6-8 implementer brief

- Focus on row preservation and enrichment-only behavior.
- Add fixtures for location parsing, remote inference, date validation, ghost risk thresholds, and description quality.
- Add a contract test that explicitly documents whether metro columns survive into the final dataset.

### Stage 9 implementer brief

- Focus on deterministic control-cohort selection, short-description skips, extraction routing, and row-preserving cleaned-text integration.
- Add one sampled test from a high-reuse extraction input hash and one control-bucket stability test.

### Stage 10 implementer brief

- Focus on offline tests for classification routing, cache reuse, and final posting-level integration.
- Stub subprocesses and sqlite cache.
- Cover validators, provider fallback, quota detection, cleaned-text classifier input fallback, and row-preserving integrated output.

### Stage 11 implementer brief

- Focus on exact alias/copy behavior only if the compatibility shim is retained.
- Do not add new business logic to the shim.

### Stage 12 implementer brief

- Focus on deterministic sampling and report correctness.
- Default to fake providers.
- Keep the core suite fully offline.

### Final-output implementer brief

- Focus on publication safety.
- Add fail-fast tests for missing Stage 10 integrated input and schema mismatch between expected final unified schema and actual copied file.

## Recommended Rollout Order

1. Add Stage 5 and Stage 10 unit tests first; they carry the highest branch and dependency risk.
2. Add Stage 1-4 synthetic/golden fixtures.
3. Add Stage 6-9 row-preserving and routing tests.
4. Add Stage 10-12 integration tests, plus a Stage 11 shim test only if the alias path remains.
5. Add sampled-fixture builder and freeze reviewed real rows.
6. Add final publish-gate contract tests on fixture-driven outputs.

## Minimum Definition Of Done

The framework is "good enough" when:

- every stage has at least one row-count/schema contract test
- every stage with branchy logic has synthetic edge-case tests
- every stage has at least 3 sampled real-data fixtures or a documented reason it cannot yet
- LLM and embedding stages run offline by default
- final publish-gate fails on silent schema loss or row-cardinality drift
