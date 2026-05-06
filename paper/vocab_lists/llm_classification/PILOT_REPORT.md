# LLM classification pilot — final report

_Pilot run on 2026-05-06 against `data/unified_core.parquet`. 25 stratified SWE postings × 3 models (`gpt-5.4-nano`, `gpt-5.4-mini`, `gpt-5.4`) × 3 reps = 225 calls. Plus 6 stability variants × 25 postings = 150 calls on the chosen tier. All 375 calls parsed (0 parse failures). See `analysis.md`, `spot_check.md`, `stability_report.md` for raw outputs._

## Headlines

- **Prompt v1 works.** Multi-label JSON output via OpenAI Responses API + structured-output schema, all 375 calls parsed, no JSON repair needed.
- **All three model tiers give correlated labels** (macro Cohen's κ ≥ 0.81 between any two models on majority-vote labels) and tag the obvious cases identically.
- **Recommendation: `gpt-5.4-mini` with 3-rep majority voting**. Reasoning below.
- **Estimated full-corpus cost** at chosen tier ≈ ~$10-25 for 60k SWE postings × 3 reps.

## Detailed results

### Latency and tokens
| Model | Mean latency | Mean output tokens |
|---|---:|---:|
| `gpt-5.4-nano` | 1.80s | 104 |
| `gpt-5.4-mini` | 2.04s | 149 |
| `gpt-5.4` | 3.08s | 142 |

All tiers are fast enough for full-corpus runs in the pipeline. No cached-token reuse this round (cold prompt cache); production runs will benefit from cache hits since the system prompt is identical across calls.

### Self-stability (Jaccard across 3 reps, same prompt)
| Model | Mean | Min |
|---|---:|---:|
| `gpt-5.4` | **0.931** | 0.333 |
| `gpt-5.4-mini` | 0.865 | 0.333 |
| `gpt-5.4-nano` | 0.766 | 0.333 |

`gpt-5.4-nano` is meaningfully unstable — same prompt, same posting, different label set ~25% of the time on average. **Disqualifies nano** from production use without majority voting, and even with majority voting it's noisy.

### Inter-model agreement (macro Cohen's κ on 3-rep majority labels)
| Pair | Macro κ | Set Jaccard |
|---|---:|---:|
| `gpt-5.4` ↔ `gpt-5.4-mini` | **0.87** | 0.80 |
| `gpt-5.4` ↔ `gpt-5.4-nano` | 0.82 | 0.82 |
| `gpt-5.4-mini` ↔ `gpt-5.4-nano` | 0.81 | 0.83 |

All pairs in the "substantial agreement" band (κ > 0.7). Per-label, `people_management` / `verification` / `mentorship` reach κ = 1.0 across all model pairs (perfect agreement). The weakest label is `legacy_stack`: nano gets κ = 0.24-0.34 against the larger models — nano specifically gets legacy wrong (over- and under-tags). `orchestration` is the second weakest (κ = 0.59) — driven by genuine ambiguity, not by any single model.

### LLM vs regex (per-label F1)
The current 4-round-cleaned regex labels are reasonably aligned with LLM majority labels. F1 by label, mini:
- `mentorship` 0.92, `orchestration` 0.92 — strongest agreement
- `process_scaffolding` 0.67, `verification` 0.67, `performance` 0.67, `people_management` 0.67, `context_infrastructure` 0.70 — moderate
- `legacy_stack` 0.57 — weakest, driven by remaining false-positive risk on the regex side

The regex over-tags `people_management` (LLM precision = 0.50, recall = 1.00) — i.e., when regex says yes, LLM agrees, but regex says yes more often than LLM does. Consistent with the residual fluff that the v4 regex review couldn't dislodge.

### Stability tests on `gpt-5.4-mini` (chosen tier)

| Test | Mean Jaccard vs pilot reference |
|---|---:|
| ORDER alphabetical | 0.815 |
| ORDER reverse | 0.872 |
| ORDER by-regex-frequency | 0.800 |
| ORDER random-shuffle | 0.805 |
| PARAPHRASE terse | 0.780 |
| PARAPHRASE active-voice | 0.831 |

**Cross-variant mean Jaccard: 0.800.**

These numbers need careful interpretation. The "reference" is the 3-rep majority from the pilot (same model, same prompt). Each variant is a single rep. Mini's pilot self-Jaccard (run-to-run noise on the same prompt) was 0.865 — so a single variant rep's Jaccard against a 3-rep majority is **noise-floor-limited**, not purely a variant effect. The fact that all variants land in the 0.78-0.87 band means **order and paraphrase changes do not noticeably exceed the model's irreducible run-to-run noise**.

In other words: the prompt is reasonably robust to surface variation; the bigger problem is run-to-run noise at single-rep granularity. **Mitigation: 3-rep majority voting in production** (already in the pilot design and easy to keep).

The four worst-stability postings cluster on roles where the LLM genuinely vacillates between adjacent labels — e.g., DevOps Engineer (verification vs context_infrastructure vs process_scaffolding), Android Developer (orchestration: yes or no?). These are concept-boundary cases, not LLM defects.

## Why `gpt-5.4-mini`

| Criterion | Decision |
|---|---|
| Self-stability (3-rep) | mini at 0.87, full at 0.93. Both adequate with majority voting. |
| Agreement with full | mini ↔ full κ = 0.87 — substantial. |
| Legacy-stack weakness | nano fails (κ=0.24 vs mini); mini matches full closely. |
| Latency | 2.04s — comfortable for parallel batched runs. |
| Cost | ~5× nano, ~⅕ of full. Sweet spot. |
| Run-to-run noise | Mitigated by 3-rep majority. |

Nano's instability and legacy-stack failures rule it out unless the per-call cost is the binding constraint, which it isn't at our scale (~$10-25 vs ~$2-5 — both negligible relative to the paper's value).

## Open questions for the next phase (not blocking)

1. **3-rep production cost.** ~60k postings × 3 reps × mini ≈ 60-75k calls. Cost should be in the $30-75 range. Verify before scaling.
2. **Genuinely ambiguous postings.** ~10 of 25 had cross-model disagreement; these are real concept-boundary cases. The eventual hand-validation cohort should over-sample them rather than drawing uniformly random.
3. **Boundary labels.** `orchestration` (κ = 0.59 between models) and `legacy_stack` (mini ↔ nano κ = 0.24) are the weakest. Worth inspecting whether the prompt's definitions for these are too broad — but only after validation on a larger sample.
4. **Prompt cache.** The system prompt is fixed, so production runs should hit the cache. Worth verifying empirically once we run > 1k calls (cache eligibility kicks in beyond a threshold).

## Decision and next steps

- **Freeze prompt at v1.** No edits to `prompt_v1.md` going forward; future revisions create `prompt_v2.md` and a new pilot.
- **Promote to a preprocessing stage** with `gpt-5.4-mini` as default, 3-rep majority voting per posting. Stage probably lives between Stage 10 (LLM classify) and Stage 11 (embeddings), reading from `description_core_llm`.
- **Hand-validation cohort** (paper-grade ground truth) is a separate downstream task; defer until preprocessing-stage is wired and run on full corpus.

## Files in this folder

| File | Role |
|---|---|
| `prompt_v1.md` | Frozen prompt artifact. |
| `classifier.py` | LLM call wrapper (httpx + Responses API). |
| `run_pilot.py` | Sample selector + 225-call driver. |
| `analyze.py` | Pilot agreement metrics. |
| `spot_check.py` | Human-readable markdown renderer. |
| `stability_tests.py` | Order-shuffle + paraphrase variants. |
| `pilot_sample.parquet` | The 25 frozen postings. |
| `pilot_results.jsonl` | 225 raw call results. |
| `stability_results.jsonl` | 150 stability call results. |
| `analysis.md` | Pilot statistics. |
| `spot_check.md` | 25 postings × 3 models, eyeball view. |
| `stability_report.md` | Stability test results. |
| `PILOT_REPORT.md` | This file — synthesized recommendation. |
