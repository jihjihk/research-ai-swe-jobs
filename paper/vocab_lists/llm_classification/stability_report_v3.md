# V3 stability tests (gpt-5.4-mini, combined variant)

_Variants tested against pilot reference (3-rep majority of `gpt-5.4-mini` on `combined`)._

## Order-shuffle variants

| Variant | Skill axis Jaccard | Role axis Jaccard | Min skill | Min role |
|---|---:|---:|---:|---:|
| `order:skill_reverse` | 0.843 | 0.957 | 0.000 | 0.500 |
| `order:skill_shuffled_seed_7` | 0.824 | 0.906 | 0.000 | 0.000 |
| `order:skill_shuffled_seed_42` | 0.819 | 0.919 | 0.000 | 0.333 |
| `order:role_reverse` | 0.861 | 0.933 | 0.000 | 0.333 |
| `order:role_shuffled_seed_7` | 0.736 | 0.903 | 0.000 | 0.000 |
| `order:role_shuffled_seed_42` | 0.899 | 0.859 | 0.000 | 0.333 |
| `order:both_shuffled_seed_7` | 0.839 | 0.921 | 0.000 | 0.000 |

## Paraphrase variants

| Variant | Skill axis Jaccard | Role axis Jaccard | Min skill | Min role |
|---|---:|---:|---:|---:|
| `paraphrase:terse` | 0.809 | 0.798 | 0.000 | 0.000 |
| `paraphrase:active_voice` | 0.856 | 0.843 | 0.000 | 0.000 |

## Notes

- The reference is the 3-rep majority of the same model+variant from the v3 pilot.
- Each stability variant is one rep, so Jaccard is bounded above by the model's single-rep-vs-majority noise floor (typically ~0.85-0.90 on `mini`).
- Order changes that don't degrade beyond the noise floor → prompt is robust to ordering.
- Paraphrase changes test surface-phrasing robustness.
