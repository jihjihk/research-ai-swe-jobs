# LLM Pipeline Implementation Notes

Date: 2026-03-20

These notes document implementation-specific compatibility behavior for
Stages 9-12 without changing the methodological plan in
`docs/plan-preprocessing.md`.

## Current artifact compatibility

The current `preprocessing/intermediate/stage8_final.parquet` artifact differs
slightly from the architecture document:

- It contains `lang_detected` rather than `is_english`.
- It does not contain `description_hash`.

The Stage 9-11 implementation handles this by deriving the missing fields at
runtime:

- `is_english` behavior is inferred as `lang_detected == "en"` when
  `is_english` is absent.
- `description_hash` is computed as `sha256(description)` when the column is
  absent.

This keeps Stages 1-8 unchanged while allowing the LLM augmentation layer to
run against the existing stage-8 artifact.

## Stage 10 profiling gate

The implementation enforces the requirement that a profiling run happens before
an uncached full batch:

- `--profile` writes a prompt-version marker into `preprocessing/cache/`.
- A non-profile run with uncached descriptions aborts unless that marker exists.

This is an implementation safeguard, not a methodological change.

## Cache separation

Stage 10 uses the required cache at:

- `preprocessing/cache/llm_responses.db`

Stage 12 uses a separate validation cache for GPT-5.4 full responses so the
production mini-model cache is not overwritten:

- `preprocessing/cache/llm_validation_full.db`

This separation exists because the Stage 10 cache schema uses
`description_hash` as the primary key.
