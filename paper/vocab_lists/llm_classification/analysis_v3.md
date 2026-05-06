# V3 pilot — analysis

_Calls analyzed: 837 from 31 postings × 3 variants × 3 models × 3 reps. Parse failures: 0 (0.0%)._

## Headline metric — does combining hurt? (set-Jaccard, standalone vs combined)

Threshold: ≥0.95 means combining is safe; 0.85-0.95 minor degradation; <0.85 ship two prompts.

| Model | Skill axis (skill ↔ combined) | Role family axis (role_family ↔ combined) |
|---|---:|---:|
| `gpt-5.4` | 0.792 (min 0.000) | 0.863 (min 0.000) |
| `gpt-5.4-mini` | 0.881 (min 0.000) | 0.952 (min 0.500) |
| `gpt-5.4-nano` | 0.691 (min 0.000) | 0.632 (min 0.000) |

## Self-stability (Jaccard across 3 reps within a (variant × model))

| Variant | Model | Skill axis mean | Role-family axis mean |
|---|---|---:|---:|
| `combined` | `gpt-5.4` | 0.806 | 0.942 |
| `combined` | `gpt-5.4-mini` | 0.801 | 0.892 |
| `combined` | `gpt-5.4-nano` | 0.717 | 0.692 |
| `role_family` | `gpt-5.4` | n/a | 0.964 |
| `role_family` | `gpt-5.4-mini` | n/a | 0.892 |
| `role_family` | `gpt-5.4-nano` | n/a | 0.856 |
| `skill` | `gpt-5.4` | 0.890 | n/a |
| `skill` | `gpt-5.4-mini` | 0.835 | n/a |
| `skill` | `gpt-5.4-nano` | 0.831 | n/a |

## Inter-model agreement (per variant)

Macro κ = mean of valid per-label kappas (skips labels with no positives).

### Skill axis

| Variant | Model A | Model B | Macro κ | Set Jaccard |
|---|---|---|---:|---:|
| `combined` | `gpt-5.4` | `gpt-5.4-mini` | 0.78 | 0.66 |
| `combined` | `gpt-5.4` | `gpt-5.4-nano` | 0.55 | 0.54 |
| `combined` | `gpt-5.4-mini` | `gpt-5.4-nano` | 0.62 | 0.69 |
| `skill` | `gpt-5.4` | `gpt-5.4-mini` | 0.76 | 0.75 |
| `skill` | `gpt-5.4` | `gpt-5.4-nano` | 0.62 | 0.69 |
| `skill` | `gpt-5.4-mini` | `gpt-5.4-nano` | 0.68 | 0.76 |

### Role-family axis

| Variant | Model A | Model B | Macro κ | Set Jaccard |
|---|---|---|---:|---:|
| `combined` | `gpt-5.4` | `gpt-5.4-mini` | 0.89 | 0.88 |
| `combined` | `gpt-5.4` | `gpt-5.4-nano` | 0.67 | 0.65 |
| `combined` | `gpt-5.4-mini` | `gpt-5.4-nano` | 0.67 | 0.64 |
| `role_family` | `gpt-5.4` | `gpt-5.4-mini` | 0.85 | 0.81 |
| `role_family` | `gpt-5.4` | `gpt-5.4-nano` | 0.79 | 0.76 |
| `role_family` | `gpt-5.4-mini` | `gpt-5.4-nano` | 0.85 | 0.84 |

### Skill axis — per-label κ (`gpt-5.4` vs `gpt-5.4-mini`)

| Variant | people | orches | verifi | mentor | perfor | proces | legacy | contex |
|---|---|---|---|---|---|---|---|---|
| `combined` | 1.00 | 0.76 | 0.74 | 0.76 | 0.93 | 0.87 | 0.46 | 0.72 |
| `skill` | 1.00 | 0.71 | 0.60 | 0.83 | 0.86 | 0.80 | 0.51 | 0.72 |

### Role-family axis — per-label κ (`gpt-5.4` vs `gpt-5.4-mini`)

Sparse families may show NaN κ if neither model tags any posting.

| Variant | software | frontend | backend_ | mobile | embedded | data_eng | ml_engin | ai_llm_e | devops_s | security | qa_test | solution | legacy_s | data_ana | research | infra_op | people_m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `combined` | 1.00 | 1.00 | 0.77 | 0.65 | 1.00 | 1.00 | 0.78 | 0.64 | 0.79 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.47 | 1.00 | 1.00 |
| `role_family` | 0.00 | 0.90 | 0.42 | 0.65 | 1.00 | 1.00 | 0.78 | 0.78 | 0.87 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Title-heuristic match (role family)

For postings whose title heuristic returned a specific (non-fallback) family, the rate at which the model also tagged that family. Higher is better.

| Variant | Model | Match rate |
|---|---|---:|
| `role_family` | `gpt-5.4` | 0.958 |
| `role_family` | `gpt-5.4-mini` | 0.958 |
| `role_family` | `gpt-5.4-nano` | 0.958 |
| `combined` | `gpt-5.4` | 0.958 |
| `combined` | `gpt-5.4-mini` | 0.958 |
| `combined` | `gpt-5.4-nano` | 0.958 |

## `software_engineer_general` misuse

`software_engineer_general` is supposed to be the residual fallback — the model is instructed never to pair it with another family. Misuse rate = % of postings where it appears alongside another family. Lower is better.

| Variant | Model | Misuse rate |
|---|---|---:|
| `role_family` | `gpt-5.4` | 0.0% (0/31) |
| `role_family` | `gpt-5.4-mini` | 0.0% (0/31) |
| `role_family` | `gpt-5.4-nano` | 0.0% (0/31) |
| `combined` | `gpt-5.4` | 0.0% (0/31) |
| `combined` | `gpt-5.4-mini` | 0.0% (0/31) |
| `combined` | `gpt-5.4-nano` | 6.5% (2/31) |

## Notes

- Sample size is 31 postings; per-label kappa is statistically noisy — treat as directional.
- For sparse role families (e.g., `research`, `solutions_field`), a NaN or zero κ usually means agreement-by-mutual-zero rather than disagreement.
- Self-stability target ≥ 0.85; inter-model macro κ target ≥ 0.7.
- The headline standalone-vs-combined Jaccard is computed model-wise; if any model shows a meaningful gap, ship two prompts for that axis.
