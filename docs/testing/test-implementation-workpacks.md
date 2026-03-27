# Test Implementation Workpacks

Date: 2026-03-27
Scope: Stage 1 through Stage 12 plus final assembly

## Purpose

This is the execution playbook for the remaining test implementation. The goal is a high-signal, minimal test suite that catches logic regressions, schema/cardinality drift between stages, bad decisions on real rows, and failures in dependency seams.

## Non-Negotiable Style Rules

- Keep tests short and local.
- Prefer parametrized cases over repeated test bodies.
- One test should prove one idea.
- Default to TDD: add or tighten the test first, then make the implementation pass.
- Use tiny synthetic fixtures first.
- Add sampled real-data fixtures only when they add signal that synthetic data cannot.
- Never snapshot large parquet outputs.
- Keep helper layers thin. Avoid test-framework abstractions that hide logic.
- Default all dependency-heavy tests offline with stubs or temp sqlite/parquet files.
- Every sampled fixture must record why the expected answer is correct.

## Shared Implementation Protocol

Each implementation agent should follow this sequence:

1. Read only:
   - `docs/1-research-design.md`
   - `docs/plan-preprocessing.md`
   - the target stage script
   - `docs/testing/preprocessing-test-strategy.md`
   - this file
2. Build a stage card:
   - purpose
   - inputs
   - outputs
   - columns created or changed
   - what the stage must not own
3. Compare plan vs code and write down every mismatch that changes test design.
4. Add or tighten the smallest failing test first.
5. Add pure-unit tests before broader fixtures whenever possible.
6. Add tiny synthetic golden fixtures next.
7. Add sampled real-row fixtures only after the synthetic logic is pinned down.
8. Add one stage-integration or contract test for the stage boundary.
9. Keep code concise; if a helper exists, reuse it instead of inventing a second abstraction.

## Sampled Fixture Review Rules

When promoting a real row into a fixture:

- extract the smallest row or row-group that proves the behavior
- inspect both the source input and the expected stage output
- write a short rationale into `tests/fixtures/sampled/manifest.json`
- include:
  - stage
  - source artifact
  - extraction query
  - unique identifier
  - expected result summary
  - reviewer/date

## Known Logic Risks Already Discovered

These are useful targets for the first implementation pass:

- Stage 2 real-employer extraction can over-extract fragments like `Vendors` and `Project Management`.
- Stage 3 emits `boilerplate_flag`; downstream quality tests need to keep that boundary explicit.
- Stage 5 carries the widest logic surface: family classification, seniority resolution, and YOE extraction.
- Stage 6-8 date sanity was too strict on scraped `date_posted`; it has now been simplified to parseability plus a 2020 floor.
- Stage 9 queue generation uses `any_value(...)` on duplicate hashes and should be tested for deterministic behavior.
- Stage 10 cache reads trust existing payload JSON too much; revalidation needs test coverage.
- Stage 12 `run_full_model()` currently needs regression tests around its provider-call plumbing.

## Workpack A: Stage 1-4

Owner files:

- `tests/test_stage1_ingest.py`
- `tests/test_stage2_aggregators.py`
- `tests/test_stage3_boilerplate.py`
- `tests/test_stage4_dedup.py`

Required coverage:

- Stage 1:
  - `map_seniority`
  - `parse_company_size`
  - `normalize_date_series`
  - `finalize_frame`
  - canonical vs observations split
  - file-selection policy for scraped inputs
- Stage 2:
  - exact and fuzzy aggregator detection
  - precision-first real-employer extraction
  - `company_name_effective` fallback
- Stage 3:
  - header classification
  - noise stripping
  - EEO stripping
  - fallback path for no headers
  - `boilerplate_flag` thresholds
- Stage 4:
  - exact duplicate removal
  - same-opening duplicate removal
  - near-duplicate keep/drop logic
  - multi-location keep-both behavior
  - company canonicalization methods

Fixture minimums:

- synthetic:
  - Stage 1: 2 fixture sets
  - Stage 2: 2 fixture sets
  - Stage 3: 2 fixture sets
  - Stage 4: 3 fixture sets
- sampled:
  - Stage 2: at least 3 positive + 3 negative real-employer fixtures
  - Stage 3: at least 2 reviewed rows
  - Stage 4: at least 3 reviewed rows or row-groups

Acceptance criteria:

- Stage 1 tests explicitly prove no occupation-class columns are introduced.
- Stage 2 tests prove precision on negatives, not just success on positives.
- Stage 3 tests prove original `description` stays unchanged.
- Stage 4 tests prove the stage is the first allowed row-cardinality change.

## Workpack B: Stage 5

Owner files:

- `tests/test_stage5_classification.py`

Required coverage:

- title normalization
- primary SWE include/exclude behavior
- adjacent technical rescue behavior
- control-title negatives
- title-lookup artifact validation
- seniority precedence rules
- `resolve_seniority_final`
- YOE candidate rejection reasons
- YOE resolution rules
- contradiction thresholds

Fixture minimums:

- synthetic:
  - at least 10 parametrized title-family cases
  - port all rows from `tests/test_stage5_yoe_extractor.py`
  - 1 tiny parquet integration fixture
- sampled:
  - at least 5 reviewed rows covering:
    - native backfill
    - title-prior path
    - junior/high-YOE contradiction
    - adjacent-family rescue
    - `other` family no-backfill behavior

Acceptance criteria:

- tests prove `is_swe`, `is_swe_adjacent`, and `is_control` are mutually exclusive
- tests prove row-preserving behavior on the tiny parquet integration fixture
- tests keep embedding/model work stubbed

## Workpack C: Stage 6-8 and Final Assembly

Owner files:

- `tests/test_stage678_normalize_temporal_flags.py`
- `tests/test_stage_final_output.py`

Required coverage:

- location parsing
- remote inference
- metro inference precedence
- date sanity logic
- ghost-job thresholds
- description quality
- final observations join behavior
- final-output schema contract

Fixture minimums:

- synthetic:
  - at least 3 location fixtures
  - at least 3 date fixtures
  - 1 small final-assembly parquet fixture set
- sampled:
  - at least 4 reviewed rows covering:
    - LinkedIn date sanity
    - older Indeed posting date
    - ghost risk
    - short description quality

Acceptance criteria:

- Stage 6-8 tests prove row preservation
- date tests reflect the new permissive sanity policy
- final-output tests are fixture-driven and do not depend on the current repo outputs

## Workpack D: Stage 9-12

Owner files:

- `tests/test_stage9_llm_prefilter.py`
- `tests/test_stage10_llm_classify.py`
- `tests/test_stage11_llm_integrate.py`
- `tests/test_stage12_validation.py`

Required coverage:

- Stage 9:
  - routing universe
  - route-group assignment
  - hash-based queue reduction
  - deterministic behavior on duplicate hashes
- Stage 10:
  - payload validators
  - unitizer and parser behavior
  - cache schema
  - cache hit/miss behavior
  - quota/rate-limit detection
  - synthetic extraction fallbacks
- Stage 11:
  - route-respecting integration
  - extraction reconstruction
  - null-on-failure behavior
  - row preservation
- Stage 12:
  - disagreement detection
  - deterministic sampling
  - kappa helpers
  - report generation
  - provider-call regression coverage

Fixture minimums:

- synthetic:
  - Stage 9: 2 route fixtures
  - Stage 10: 3 validator/parser fixtures
  - Stage 11: 2 sqlite-backed integration fixtures
  - Stage 12: 2 report/sample fixtures
- sampled:
  - Stage 9: at least 1 reviewed high-reuse hash
  - Stage 10-11: at least 2 reviewed cache/examples from eval artifacts

Acceptance criteria:

- all tests remain offline
- subprocess calls are always stubbed
- sqlite use is via temp DBs only
- Stage 11 explicitly proves row preservation

## Cross-Cutting Publish-Gate Work

Owner files:

- `tests/test_pipeline_smoke.py`
- `tests/test_runner_contracts.py`

Required coverage:

- runner stage ordering
- expected `check_col` coverage
- stage-contract compatibility on fixture-generated outputs
- final-assembly schema compatibility

Acceptance criteria:

- no dependency on current repo-scale outputs
- tests prove declared contracts, not local artifact health

## Recommended Implementation Order

1. Workpack B: Stage 5
2. Workpack D: Stage 10 and Stage 11
3. Workpack A: Stage 2 and Stage 4
4. Workpack C: Stage 6-8
5. Workpack A: Stage 1 and Stage 3
6. Workpack D: Stage 9 and Stage 12
7. Final publish-gate tests

## What To Report Back

Each implementation agent should report:

- files changed
- tests added
- fixtures added
- any code logic issue discovered
- any contract ambiguity discovered
- any test that was intentionally deferred and why
