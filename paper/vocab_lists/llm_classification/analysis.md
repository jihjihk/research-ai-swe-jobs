# LLM classification pilot — analysis

_Calls analyzed: 225 from 25 postings × 3 models × 3 reps._


## Latency and tokens per model

| Model | Mean latency (s) | Median (s) | Max (s) | Mean input tok | Mean output tok | Cached tokens (sum) |
|---|---:|---:|---:|---:|---:|---:|
| `gpt-5.4` | 3.08 | 2.81 | 6.45 | 684 | 142 | 0 |
| `gpt-5.4-mini` | 2.04 | 2.00 | 5.03 | 684 | 149 | 0 |
| `gpt-5.4-nano` | 1.80 | 1.62 | 4.79 | 684 | 104 | 0 |

## Self-agreement (Jaccard across reps within a model)

Higher is better. <0.95 = unstable on this prompt.

| Model | Mean self-Jaccard (across postings) | Min |
|---|---:|---:|
| `gpt-5.4` | 0.931 | 0.333 |
| `gpt-5.4-mini` | 0.865 | 0.333 |
| `gpt-5.4-nano` | 0.766 | 0.333 |

## Inter-model agreement

Per-label Cohen's κ on majority-vote labels (3 reps → label is positive if it appears in ≥2 reps). Higher = more agreement; >0.7 = substantial; >0.4 = moderate.

| Model A | Model B | Macro κ | Set Jaccard | peopleκ | orchesκ | verifiκ | mentorκ | perforκ | procesκ | legacyκ | contexκ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `gpt-5.4` | `gpt-5.4-mini` | 0.87 | 0.80 | 1.00 | 0.59 | 1.00 | 1.00 | 1.00 | 0.80 | 0.78 | 0.76 |
| `gpt-5.4` | `gpt-5.4-nano` | 0.82 | 0.82 | 1.00 | 0.59 | 0.91 | 1.00 | 0.88 | 0.80 | 0.34 | 1.00 |
| `gpt-5.4-mini` | `gpt-5.4-nano` | 0.81 | 0.83 | 1.00 | 0.84 | 0.91 | 1.00 | 0.88 | 0.82 | 0.24 | 0.76 |

## LLM vs regex (per-label P / R / F1)

Treats regex topic-density positives as 'true' and the model's majority labels as 'predicted'. Disagreements are interesting on both sides: the regex has known false-positive issues; the LLM is independent. Eyeball both columns rather than treating either as ground truth.

### `gpt-5.4`

| Label | P | R | F1 |
|---|---:|---:|---:|
| `people_management` | 0.50 | 1.00 | 0.67 |
| `orchestration` | 1.00 | 0.58 | 0.74 |
| `verification` | 0.67 | 0.67 | 0.67 |
| `mentorship` | 1.00 | 0.86 | 0.92 |
| `performance` | 0.67 | 0.67 | 0.67 |
| `process_scaffolding` | 1.00 | 0.60 | 0.75 |
| `legacy_stack` | 1.00 | 0.50 | 0.67 |
| `context_infrastructure` | 0.80 | 0.80 | 0.80 |

### `gpt-5.4-mini`

| Label | P | R | F1 |
|---|---:|---:|---:|
| `people_management` | 0.50 | 1.00 | 0.67 |
| `orchestration` | 0.92 | 0.92 | 0.92 |
| `verification` | 0.67 | 0.67 | 0.67 |
| `mentorship` | 1.00 | 0.86 | 0.92 |
| `performance` | 0.67 | 0.67 | 0.67 |
| `process_scaffolding` | 0.75 | 0.60 | 0.67 |
| `legacy_stack` | 0.67 | 0.50 | 0.57 |
| `context_infrastructure` | 0.62 | 0.80 | 0.70 |

### `gpt-5.4-nano`

| Label | P | R | F1 |
|---|---:|---:|---:|
| `people_management` | 0.50 | 1.00 | 0.67 |
| `orchestration` | 0.83 | 0.83 | 0.83 |
| `verification` | 0.75 | 0.67 | 0.71 |
| `mentorship` | 1.00 | 0.86 | 0.92 |
| `performance` | 0.60 | 0.50 | 0.55 |
| `process_scaffolding` | 0.88 | 0.70 | 0.78 |
| `legacy_stack` | 0.33 | 0.25 | 0.29 |
| `context_infrastructure` | 0.80 | 0.80 | 0.80 |

## Postings with cross-model disagreement

10 postings show worst pairwise set-Jaccard < 0.6.

| uid | worst pairwise Jaccard | per-model labels |
|---|---:|---|
| `arshkon_3891` | 0.00 | `gpt-5.4`=[] · `gpt-5.4-mini`=['orchestration'] · `gpt-5.4-nano`=[] |
| `arshkon_3903` | 0.00 | `gpt-5.4`=[] · `gpt-5.4-mini`=['orchestration'] · `gpt-5.4-nano`=['orchestration'] |
| `asaniczka_7b` | 0.33 | `gpt-5.4`=['people_management'] · `gpt-5.4-mini`=['legacy_stack', 'people_management', 'process_scaffolding'] · `gpt-5.4-nano`=['people_management'] |
| `linkedin_li-` | 0.33 | `gpt-5.4`=['verification'] · `gpt-5.4-mini`=['context_infrastructure', 'process_scaffolding', 'verification'] · `gpt-5.4-nano`=['process_scaffolding', 'verification'] |
| `linkedin_li-` | 0.33 | `gpt-5.4`=['performance', 'verification'] · `gpt-5.4-mini`=['context_infrastructure', 'performance', 'verification'] · `gpt-5.4-nano`=['verification'] |
| `asaniczka_3d` | 0.50 | `gpt-5.4`=['mentorship', 'people_management'] · `gpt-5.4-mini`=['mentorship', 'people_management'] · `gpt-5.4-nano`=['legacy_stack', 'mentorship', 'orchestration', 'people_management'] |
| `linkedin_li-` | 0.50 | `gpt-5.4`=['performance', 'process_scaffolding', 'verification'] · `gpt-5.4-mini`=['context_infrastructure', 'orchestration', 'performance', 'process_scaffolding', 'verification'] · `gpt-5.4-nano`=['orchestration', 'performance', 'process_scaffolding'] |
| `asaniczka_0c` | 0.50 | `gpt-5.4`=['context_infrastructure', 'legacy_stack', 'verification'] · `gpt-5.4-mini`=['context_infrastructure', 'legacy_stack', 'verification'] · `gpt-5.4-nano`=['context_infrastructure', 'process_scaffolding', 'verification'] |
| `linkedin_li-` | 0.50 | `gpt-5.4`=['mentorship'] · `gpt-5.4-mini`=['mentorship', 'orchestration'] · `gpt-5.4-nano`=['mentorship', 'orchestration'] |
| `asaniczka_00` | 0.50 | `gpt-5.4`=['context_infrastructure', 'performance'] · `gpt-5.4-mini`=['context_infrastructure', 'orchestration', 'performance'] · `gpt-5.4-nano`=['context_infrastructure', 'legacy_stack', 'orchestration', 'performance'] |

## Notes

- Cohen's κ is computed on majority-vote labels per (posting × model). Reps were 3.
- 'Set Jaccard' is `|A∩B| / |A∪B|` averaged across postings.
- LLM-vs-regex F1 is descriptive, not validation. Use spot_check.md to read example disagreements.
