# LLM-based topic classification — pilot module

This folder pilots an LLM-based multi-label classification of SWE job postings against the same 8 topics covered by `paper/vocab_lists/vocab_lists.json`. Goal of this folder: pick a model tier and freeze a prompt that we can later promote into the preprocessing pipeline as a new stage. **Not** the formal paper-grade validation — that's a separate hand-labelled cohort, downstream.

## Files

| File | Role |
|---|---|
| `prompt_v1.md` | Frozen prompt artifact. Edit only by bumping to `prompt_v2.md`. |
| `classifier.py` | Single-posting LLM call wrapper. httpx + OpenAI Responses API + structured output. |
| `run_pilot.py` | Stratified 25-posting sample selector + multi-model multi-rep driver. Writes `pilot_sample.parquet` and `pilot_results.jsonl`. |
| `analyze.py` | Inter-model agreement (Cohen's κ), self-agreement (Jaccard across reps), LLM-vs-regex F1, confusion table. Writes `analysis.md`. |
| `spot_check.py` | Renders the 25 postings + per-model labels into a single skim-friendly markdown. Writes `spot_check.md`. |
| `stability_tests.py` | Order-shuffle and prompt-paraphrase robustness on the chosen model. Writes `stability_results.jsonl` + `stability_report.md`. |

Outputs (generated, gitignored where appropriate):
- `pilot_sample.parquet` — frozen 25 postings (uid, source, period, title, description, regex topic labels)
- `pilot_results.jsonl` — one row per (posting × model × rep) call
- `analysis.md`, `spot_check.md`, `stability_report.md` — human-readable

## Auth

Reuses the preprocessing pipeline's OpenAI auth. Reads `OPENAI_API_KEY` from the environment, falling back to `~/.config/job-research/openai.env` (override with `JOB_RESEARCH_OPENAI_ENV_FILE`). Same convention as `preprocessing/scripts/llm_shared.py`.

## How to run

```bash
# 1. Pilot (225 calls, ~$5-30 total depending on tiers)
./.venv/bin/python paper/vocab_lists/llm_classification/run_pilot.py

# 2. Analysis
./.venv/bin/python paper/vocab_lists/llm_classification/analyze.py
./.venv/bin/python paper/vocab_lists/llm_classification/spot_check.py

# 3. Stability tests (on the chosen tier)
./.venv/bin/python paper/vocab_lists/llm_classification/stability_tests.py --model gpt-5.4-mini
```

## Labels

8 binary topic labels matching the vocabulary lists, defined in `prompt_v1.md`:
- `people_management`, `orchestration`, `verification`, `mentorship`, `performance`, `process_scaffolding`, `legacy_stack`, `context_infrastructure`

Multi-label, no exclusivity. Output is a JSON array of slugs.

## Sample stratification

The 25-posting pilot is stratified by current regex labels to stress-test the LLM on easy and hard cases:

| Stratum | n | Selection rule |
|---|---:|---|
| Single-topic regex hits | 8 | 1 posting per topic where only that topic's regex fires |
| Multi-topic regex hits | 5 | postings hitting 3+ topics |
| No-topic regex hits | 5 | postings hitting zero topics |
| High-noise topics | 5 | postings hitting `performance/depth_claim` or `mentorship/Guidance` (the persistent fluff layers) |
| Random | 2 | uniform random for sanity |

## Promotion to pipeline

When validated, this work graduates to a new stage in `preprocessing/scripts/`. Until then nothing in this folder is wired into `run_pipeline.py`.
