# LLM classification — stability tests (gpt-5.4-mini)

_Variants tested against pilot reference (3-rep majority on `gpt-5.4-mini`)._

## Order-shuffle variants

Same prompt, different label-list order in the structured-output enum. If the model is robust, Jaccard vs. reference stays >=0.95.

| Variant | Mean Jaccard vs reference | Min | Postings perfectly stable |
|---|---:|---:|---:|
| `order:alphabetical` | 0.815 | 0.000 | 17/25 |
| `order:reverse_alphabetical` | 0.872 | 0.000 | 18/25 |
| `order:regex_frequency_desc` | 0.800 | 0.000 | 17/25 |
| `order:random_seed_7` | 0.805 | 0.000 | 17/25 |

## Paraphrase variants

Same label set, different system-prompt phrasing. If the model is robust to surface phrasing, Jaccard vs. reference stays high.

| Variant | Mean Jaccard vs reference | Min | Postings perfectly stable |
|---|---:|---:|---:|
| `paraphrase:paraphrase_terse` | 0.780 | 0.000 | 16/25 |
| `paraphrase:paraphrase_active_voice` | 0.831 | 0.000 | 17/25 |

## Cross-variant agreement

For each posting, fraction of (variant pairs) with identical label sets.

**Worst-stability postings** (lowest mean cross-variant Jaccard):

| uid | mean cross-var Jaccard | min |
|---|---:|---:|
| `linkedin_li-43` | 0.33 | 0.00 |
| `arshkon_389102` | 0.40 | 0.00 |
| `arshkon_390381` | 0.47 | 0.00 |
| `linkedin_li-43` | 0.47 | 0.00 |
| `linkedin_li-43` | 0.50 | 0.25 |
| `asaniczka_19a3` | 0.53 | 0.00 |
| `asaniczka_0c98` | 0.65 | 0.40 |
| `linkedin_li-43` | 0.73 | 0.50 |
| `asaniczka_0021` | 0.78 | 0.33 |
| `asaniczka_7b7a` | 0.79 | 0.50 |

**Overall**: mean cross-variant Jaccard = 0.800
