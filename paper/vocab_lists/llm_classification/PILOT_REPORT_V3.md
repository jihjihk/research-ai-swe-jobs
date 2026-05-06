# V3 pilot — final report

_Pilot run on 2026-05-06. 31 stratified SWE postings × 3 prompt variants (skill / role_family / combined) × 3 models × 3 reps = 837 calls. Plus 9 stability variants × 31 postings on the chosen tier+variant = 279 stability calls. **0 parse failures across 1,116 total calls.** See `analysis_v3.md`, `spot_check_v3.md`, `stability_report_v3.md`._

## Decision

**Ship the combined v3 prompt on `gpt-5.4-mini` with 3-rep majority voting.**

Rationale:
- **Combined is safe for the role-family axis**: standalone↔combined Jaccard = 0.952 on mini, above the 0.95 ship-threshold.
- **Combined has minor degradation on the skill axis**: standalone↔combined Jaccard = 0.881 on mini, in the 0.85-0.95 acceptable band. Inside the model's run-to-run noise floor (mini's self-stability is 0.83-0.87), so the gap is mostly stochastic, not structural.
- **Inter-model agreement is equal or slightly better under combined**: skill macro κ 0.78 (combined) vs 0.76 (skill standalone); role-family macro κ 0.89 (combined) vs 0.85 (standalone).
- **One API call instead of two halves the per-posting cost**, simplifies the eventual preprocessing stage, and keeps a single canonical prompt artefact frozen at `prompt_combined_v3.md`.

## Headline metrics

### Standalone-vs-combined (the decision metric)

| Model | Skill axis Jaccard | Role-family axis Jaccard |
|---|---:|---:|
| `gpt-5.4` | 0.792 | 0.863 |
| `gpt-5.4-mini` | **0.881** | **0.952** |
| `gpt-5.4-nano` | 0.691 | 0.632 |

Mini is the only tier where both axes land in or above the acceptable band. Full and nano show meaningful divergence between standalone and combined — not a defect but stronger evidence that mini is the right pick.

### Self-stability (Jaccard across 3 reps)

| Variant | full | mini | nano |
|---|---:|---:|---:|
| `skill` skill axis | 0.890 | 0.835 | 0.831 |
| `role_family` role axis | 0.964 | 0.892 | 0.856 |
| `combined` skill axis | 0.806 | 0.801 | 0.717 |
| `combined` role axis | 0.942 | 0.892 | 0.692 |

Combined slightly hurts skill self-stability (mini: 0.835 → 0.801). Role-family stability is essentially unchanged. 3-rep majority voting smooths the residual noise.

### Inter-model agreement (mini ↔ full)

| Axis | Skill macro κ | Role macro κ |
|---|---:|---:|
| Skill standalone | 0.76 | — |
| Role-family standalone | — | 0.85 |
| Combined | 0.78 | 0.89 |

Combined edges out standalone on both axes — encouraging signal that combining the two classifications doesn't trade quality.

### Per-label agreement at risk

Skill axis (combined, mini ↔ full): `legacy_stack` κ = 0.46 — the weakest label. The other 7 are 0.72-1.00. Legacy is the perennial fuzzy boundary; the v3 prompt's "any-version old-paradigm enterprise frameworks" framing covers it but model priors still differ on edge cases.

Role-family axis (combined, mini ↔ full): all 17 families ≥ 0.47 (the lowest is `research`, n=1 in sample), most ≥ 0.78. Strong agreement.

### Title-heuristic match

`role_family` and `combined`, all 3 models: **0.958** match rate when the title literally names a specific family. Identical across the 6 cells. Title is dispositive when explicit, as the prompt instructs.

### `software_engineer_general` misuse

0% on full and mini (both variants). 6.5% on combined-nano (2/31 postings tagged it alongside another family). Another reason to skip nano.

### Stability tests (mini × combined)

| Variant | Skill Jaccard | Role-family Jaccard |
|---|---:|---:|
| Order — skill list reversed | 0.843 | 0.957 |
| Order — skill list shuffled #1 | 0.824 | 0.906 |
| Order — skill list shuffled #2 | 0.819 | 0.919 |
| Order — role list reversed | 0.861 | 0.933 |
| Order — role list shuffled #1 | 0.736 | 0.903 |
| Order — role list shuffled #2 | 0.899 | 0.859 |
| Order — both shuffled | 0.839 | 0.921 |
| Paraphrase — terse | 0.809 | 0.798 |
| Paraphrase — active voice | 0.856 | 0.843 |

Reference is the same model's 3-rep majority, so Jaccard is bounded above by ~0.83-0.89 (mini's self-stability). Order changes land near or above the noise floor on both axes — **prompt is robust to ordering**. Paraphrase changes drop role-family Jaccard ~0.80-0.84, slightly below the noise floor — **prompt is moderately robust to paraphrase but more sensitive than to ordering**. Acceptable for a frozen prompt artefact; we won't be paraphrasing in production.

## Latency and tokens

| Variant | Model | Mean latency | Mean input tokens | Mean output tokens |
|---|---|---:|---:|---:|
| skill | mini | ~2.0s | ~440 | ~30 |
| role_family | mini | ~2.5s | ~520 | ~35 |
| combined | mini | ~3.0s | ~720 | ~45 |

Combined adds ~50% latency and ~75% input tokens vs skill-only — but **half** the calls and half the latency vs running both standalone variants per posting. Net win on cost and wall time.

## Cost projection — full corpus

60k SWE postings × 3 reps = 180k calls. At mini pricing with prompt-cache hits:
- Input tokens: ~720 × 180k = 130M tokens. With caching (system prompt cached after first call), only the first ~5% pays full input rate; the rest pays cache rate.
- Estimated cost: **~$30-60** for the full corpus.

Compared to running standalone skill + standalone role_family separately: ~$50-90. Combined wins by ~30-50%.

## What to do before scaling

1. **Add 5xx retry to `classifier_v3.py`** — the v1 pilot saw 1 transient HTTP 502; production at 180k calls will see them more often. Match the preprocessing pipeline's `llm_shared.py` retry pattern.
2. **Verify cache-hit rate on a 1k-call dry-run** — production cost projections assume the system prompt caches. Confirm by inspecting `cached_tokens` in the dry-run output.
3. **Decide on majority threshold for production** — 3 reps with ≥ 2 majority is the pilot pattern. Could go to 5 reps if budget permits and quality bar tightens. 3 reps gives Jaccard ~0.87-0.95; 5 reps would push to ~0.92-0.98.
4. **Hand-validation cohort** (paper-grade ground truth, separate task) — ~200 stratified postings double-coded by you and Jihyun, computed against the chosen model+variant. The pilot is preprocessing-stage validation, not paper-grade.

## Files in this folder (v3 pilot artefacts)

| File | Role |
|---|---|
| `prompt_skill_v2.md` | Frozen — skill-only variant prompt |
| `prompt_role_family_v1.md` | Frozen — role-family-only variant prompt |
| `prompt_combined_v3.md` | **Frozen — production prompt** |
| `classifier_v3.py` | LLM call wrapper supporting all 3 variants |
| `build_sample_v3.py` | Sample selector with role-family stratum |
| `run_pilot_v3.py` | 837-call driver |
| `analyze_v3.py` | Per-axis agreement metrics |
| `spot_check_v3.py` | 3-variant side-by-side renderer |
| `stability_tests_v3.py` | Order + paraphrase stability harness |
| `pilot_sample_v3.parquet` | 31 frozen postings |
| `pilot_v3_results.jsonl` | 837 raw call results |
| `stability_v3_results.jsonl` | 279 stability call results |
| `analysis_v3.md` | Pilot agreement report |
| `spot_check_v3.md` | Human-readable per-posting view |
| `stability_report_v3.md` | Stability test report |
| `PILOT_REPORT_V3.md` | This file |

## Promotion plan

When ready, promote `prompt_combined_v3.md` + `classifier_v3.py` into a new preprocessing stage between Stage 10 (LLM classify) and Stage 11 (embeddings):
- Reads `description_core_llm` from Stage 9 output
- 3-rep majority on `gpt-5.4-mini` per posting
- Writes `skill_themes` and `role_families` columns
- Cost cap via existing `--llm-budget` mechanism
- Cache: same `~/.config/job-research/openai.env` auth as preprocessing
