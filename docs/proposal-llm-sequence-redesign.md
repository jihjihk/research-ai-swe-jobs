# Proposal: Simplify The LLM Sequence Around Cleaned Description First

Date: 2026-03-27
Status: Proposal
Scope: Redesign the Stage 9-11 data flow without changing any LLM prompt text

## Goal

Make the LLM layer easier to reason about by separating the two LLM tasks into a clean sequence:

1. Run LLM boilerplate removal first.
2. Materialize a stable posting-level cleaned description.
3. Feed that cleaned description into the classification task.
4. Keep the final posting-level LLM artifact row-preserving and analysis-facing.

The main expected benefits are:

- smaller classifier inputs
- less distracting text in the classifier input
- simpler routing logic because each task gets its own prefilter
- fewer mixed responsibilities inside one hash-level queue

## What Is Wrong With The Current Design

Current Stages 9-11 are structurally correct but harder to change than they need to be:

- Stage 9 routes extraction and classification together off the same row-level pass and the same hash-deduplicated candidate queue.
- Stage 10 executes two different tasks from one queue, even though the classification task should logically depend on the output of extraction.
- Stage 11 integrates both tasks at once, which hides the fact that extraction is really a prerequisite for the preferred classifier input.
- The current cache/dedup key is `description_hash`, but both prompts include `title` and `company_name`. That means the current key is cheaper than the true prompt input, but not semantically exact.

There is also a more practical issue:

- the current downstream contract already treats the last LLM stage as the real posting-level artifact
- so the redesign should keep one final posting-level LLM artifact, but it does not need a third LLM stage to do that

## Recommendation

Collapse the LLM layer to two stages:

- Stage 9 becomes LLM analysis-universe selection plus extraction routing, extraction execution, and cleaned-text integration
- Stage 10 becomes classification routing, classification execution, and final posting-level integration

This is cleaner than the current split because:

- Stage 9 fully owns the cleaned-text contract
- Stage 10 fully owns the classification contract
- there is no extra integration-only stage

This does mean downstream code should move from a Stage 11 posting artifact to a Stage 10 posting artifact. That migration should be done explicitly rather than keeping an extra stage only for compatibility.

## Proposed Stage Design

### Stage 9: Cohort Selection, Extraction, And Cleaned-Text Integration

Purpose:

- decide which rows should receive LLM boilerplate removal
- choose the LLM control cohort before any extraction calls are planned
- run only the extraction task
- validate extraction responses
- reconstruct posting-level cleaned text
- produce one stable description column for later classification

Input:

- `preprocessing/intermediate/stage8_final.parquet`

Row-level universe:

- `source_platform == "linkedin"`
- `is_english == True`
- raw `description` present

Default extraction corpus:

- all `is_swe == True`
- all `is_swe_adjacent == True`
- selected control cohort only

Short-description hard skip:

- if raw `description` is present but has fewer than `15` words, do not send it to extraction
- set `description_core_llm = ''`
- mark the row with a text-skip reason
- let Stage 10 inherit that exclusion when it builds classification candidates

Outputs:

- extraction queue / audit artifact
- extraction results audit artifact
- row-preserving cleaned posting table

Suggested files:

- `preprocessing/intermediate/stage9_llm_extraction_candidates.parquet`
- `preprocessing/intermediate/stage9_llm_extraction_results.parquet`
- `preprocessing/intermediate/stage9_llm_cleaned.parquet`

Suggested row-level columns added in `stage9_llm_cleaned.parquet`:

- `description_core_llm`
- `selected_for_control_cohort`

Definition:

- `description_core_llm` is the reconstructed cleaned description when extraction succeeds and validates
- `description_core_llm = ''` for raw descriptions under `15` words
- `selected_for_control_cohort` means the row was admitted to the stable control sample used for LLM labeling

Stage 9 should not persist a posting-level `description_for_classification` column. Stage 10 should derive classifier input transiently from the row using this fallback order:

- `description_core_llm` when non-empty
- else `description_core` when present
- else raw `description`
- except rows already excluded by the Stage 9 short-description rule

Detailed extraction metadata such as routing flags, skip reasons, input hashes, unit IDs, model provenance, and validator reasons should stay in the Stage 9 candidate/results audit artifacts rather than being copied onto every posting row.

### Stage 10: Classification Routing, Execution, And Final Integration

Purpose:

- decide which rows need LLM classification after cleaned text exists
- execute classification on the cleaned description variant
- integrate classification back into the final posting-level artifact

Input:

- `preprocessing/intermediate/stage9_llm_cleaned.parquet`
- cache DB

Row-level classification universe:

- `source_platform == "linkedin"`
- `is_english == True`
- row is not excluded by the Stage 9 short-description rule

Default classification corpus:

- all `is_swe == True`
- all `is_swe_adjacent == True`
- all `selected_for_control_cohort == True`

Default classification skip logic:

- keep the current Stage 9 logic for now:
- skip classification when `swe_classification_tier in {"regex", "embedding_high", "title_lookup_llm"}`
- and `seniority_source` starts with `title_`
- and `ghost_job_risk == "low"`

The critical difference is only the classifier input:

- classification prompt uses the transient classifier input derived from `description_core_llm`, `description_core`, and raw `description`
- not raw `description`

Outputs:

- optional classification queue / results parquet
- final posting-level integrated table

Suggested files:

- `preprocessing/intermediate/stage10_llm_classification_results.parquet`
- `preprocessing/intermediate/stage10_llm_integrated.parquet`

This makes Stage 10 the canonical posting-level artifact.

## Control Cohort Selection

The control cohort should be deterministic, stable across reruns, and monotone as the SWE corpus grows.

Conceptual placement:

- this is not part of the extraction prompt or the cleaned-text contract
- it is the first Stage 9 pre-step because Stage 9 defines the LLM analysis universe
- the purpose is simply to decide which control rows are allowed to enter the LLM pipeline at all

Recommendation:

- define the eligible control universe inside Stage 9 using the same base text rules as extraction:
  - LinkedIn
  - English
  - raw description present
  - raw description has at least `15` words
  - `is_control == True`
- do the selection at the extraction call unit, not the posting-row unit
- target total selected controls equal to the number of eligible `is_swe == True` rows
- preserve the SWE timeline by selecting controls within stable time buckets
- use a deterministic pseudo-random score inside each bucket so reruns keep the same selected rows unless the target grows

Recommended algorithm:

1. Build the eligible control pool at the same deduplicated unit used for extraction calls, preferably `extraction_input_hash`.
2. Build a `control_bucket` column:
   - scraped rows: `scraped|YYYY-WW` from scrape year plus ISO week
   - historical rows: `source|period`
3. Compute eligible SWE counts by `control_bucket`.
4. Compute eligible control counts by `control_bucket`.
5. Set the initial control target in each bucket to the SWE count for that bucket.
6. If a bucket does not have enough controls, take all eligible controls in that bucket and record the shortfall.
7. Redistribute any shortfall across buckets with spare control capacity, in proportion to their remaining SWE-weighted demand.
8. Compute a stable pseudo-random score for each control, such as `sha256('control-cohort-v1|' || control_bucket || '|' || extraction_input_hash)`.
9. Select the `target_n` lowest-score controls in each bucket.

Current data note:

- Stage 8 already has `period` and `scrape_week`, but the redesign should derive a dedicated `control_bucket` so scraped rows are weekly while historical rows remain source-defined and fixed

Why this works:

- reruns are stable because the pseudo-random score is deterministic
- growth is monotone because a larger target only adds lower-score controls; it does not replace existing ones
- time distribution stays aligned to the SWE corpus because allocation is bucketed before selection
- cache reuse happens automatically because previously selected controls re-enter the cohort with the same extraction input hash and therefore the same LLM cache key

Recommended implementation note:

- write a small audit artifact such as `preprocessing/intermediate/stage9_control_cohort.parquet` containing `extraction_input_hash`, `control_bucket`, `selected_for_control_cohort`, the stable score, and representative provenance fields
- do not make the cohort depend on mutable external state; deterministic ranking plus cache reuse is simpler and safer

## Cache And Dedup Recommendation

The redesign should stop using one generic `description_hash` as the task cache key.

Recommended cache key model:

- `extraction_input_hash = sha256(title, company_name, raw description)`
- `classification_input_hash = sha256(title, company_name, transient classification input)`

Use the task-specific input hash as the cache key together with `task_name` and `prompt_version`.

Why:

- both prompts already include title and company
- classification will now use cleaned text, not raw text
- task-specific keys remove the current mismatch between prompt input and dedup key
- this also removes the need for Stage 9 to choose an arbitrary `any_value(title/company)` representative for a shared raw description hash

Recommended cache schema:

- rename the current key concept from `description_hash` to a more general `input_hash`
- keep raw `description_hash` as a provenance column in row-level artifacts for lineage and debugging

## Prefilter Logic By Task

### Extraction Prefilter

Use for Stage 9 only.

Default:

- LinkedIn
- English
- raw description present
- raw description has at least `15` words
- technical corpus plus selected control cohort

Short-description policy:

- rows with raw descriptions under `15` words are not routed to extraction
- they receive `description_core_llm = ''`

No dependence on extraction output, because this is the first LLM task.

### Classification Prefilter

Use for Stage 10 only.

Default:

- LinkedIn
- English
- technical corpus plus selected control cohort
- not excluded by the Stage 9 short-description rule

Skip classification when:

- strong rule-based SWE signal
- title-driven seniority signal
- low ghost risk

This preserves the current production behavior while making the classifier input cleaner.

## Orchestration Contract

The clean mental model should be:

1. Stage 8 gives the rule-based posting table.
2. Stage 9 selects the control cohort, routes extraction, runs extraction, and writes the cleaned posting table.
3. Stage 10 routes classification from that cleaned table, runs classification, and writes the final posting table.

That is simpler than the current model because:

- each stage has one dominant responsibility
- the extraction stage fully owns the cleaned-text contract
- the classification task no longer depends on a later integration stage
- each task has its own routing logic and cache key

## Downstream Compatibility

Target interface after redesign:

- `preprocessing/intermediate/stage10_llm_integrated.parquet` is row-preserving and posting-level
- Stage 10 output contains both rule-based and LLM columns
- `uid`, `job_id`, `source`, `source_platform`, `title`, `company_name`, `description`, `description_core`, `description_core_llm`, `is_swe`, `is_swe_adjacent`, `is_control`, `seniority_final`, `seniority_llm`, `ghost_job_risk`, `ghost_assessment_llm` remain available

This requires coordinated updates to:

- `preprocessing/run_pipeline.py`
- `preprocessing/scripts/stage_final_output.py`
- `preprocessing/scripts/stage12_validation.py`
- `preprocessing/viewer/stage_viewer.py`
- smoke tests and runner-contract tests

Optional migration shim:

- for one transition cycle, Stage 10 may also write the old `stage11_llm_integrated.parquet` path as a compatibility alias
- this is optional and should not become the long-term architecture

## Testing Plan

### Required Test Coverage

1. Keep Stage 5 YOE regression tests unchanged and passing.
2. Add Stage 9 tests for:
   - short-description hard skip writes `description_core_llm = ''`
   - deterministic control-cohort selection at the call-dedup unit
   - control-cohort monotonic growth when SWE targets increase
3. Add Stage 10 tests for:
   - classification input fallback order uses `description_core_llm`, then `description_core`, then raw `description`
   - final integration stays row-preserving
   - Stage 9 short-description exclusions are honored during Stage 10 routing
4. Update runner and smoke tests for the new terminal artifact path and stage numbering assumptions.
5. Update Stage 12 validation tests so prompt rendering uses the same transient classifier input as Stage 10.
6. Replace the old Stage 11 integration test contract with a Stage 10 final-integration contract.

### Success Criteria

- classifier input length materially decreases
- Stage 10 final interface remains row-preserving
- classification volume does not increase unexpectedly
- raw-description YOE behavior does not regress
- selected control cohort is stable across reruns on the same corpus
- selected control cohort grows monotonically when the SWE target grows

## Documentation And Contract Updates

Update these documents and boundary references with the implementation:

- `AGENTS.md`
- `docs/plan-preprocessing.md` for the new two-stage LLM sequence and artifact paths
- `docs/schema-stage8-and-stage12.md` so the post-LLM schema boundary moves from Stage 11 to Stage 10
- `docs/prompt-pipeline-runner.md` and any runner-facing docs that still describe Stage 11 as the final LLM artifact
- `docs/parquet-viewer.md` if it references old LLM artifact names
- stage docstrings and inline comments in `stage9_llm_prefilter.py`, `stage10_llm_classify.py`, `stage11_llm_integrate.py`, `stage12_validation.py`, and `stage_final_output.py`

Update these code contracts with the implementation:

- `preprocessing/run_pipeline.py`
- `preprocessing/scripts/stage12_validation.py`
- `preprocessing/scripts/stage_final_output.py`
- `preprocessing/viewer/stage_viewer.py`
- `tests/test_stage10_llm_classify.py`
- `tests/test_stage11_llm_integrate.py` or its replacement if Stage 11 is removed
- `tests/test_stage_final_output.py`
- `tests/test_pipeline_smoke.py`
- `tests/test_runner_contracts.py`

## Implementation Sequence

1. Refactor cache helpers to support task-specific `input_hash`.
2. Rewrite Stage 9 to:
   - select the deterministic control cohort as a pre-step
   - apply the `<15 word` short-description skip
   - route extraction
   - execute extraction
   - integrate `description_core_llm`
   - write `selected_for_control_cohort`
3. Rewrite Stage 10 to:
   - route classification from `stage9_llm_cleaned.parquet`
   - derive classifier input transiently from the cleaned row
   - execute classification on that transient input
   - integrate final posting-level columns
4. Update Stage 12 full-model validation so classification prompts use the same transient classifier input, not raw `description`.
5. Update runner contracts, final output, viewer, and smoke tests from the Stage 11 posting artifact to the Stage 10 posting artifact.
6. Add a deterministic control-cohort audit artifact and a targeted test for cohort stability.

## Final Recommendation

Implement one clean two-stage redesign:

- Stage 9 selects the control cohort, skips unusable short descriptions, runs extraction, and materializes `description_core_llm`
- Stage 10 runs classification from that cleaned input and writes the final posting-level LLM artifact
- cache keys become task-specific
- control selection becomes deterministic, bucketed, and monotone across reruns
- no Stage 5 rule-refresh work is part of this implementation

This gets the main benefit you want, cleaner classifier input with a simpler architecture, without adding extra rule-based loops or a third LLM stage.
