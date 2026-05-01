# Data sources

## Three complementary posting sources

| Source | Temporal role | Platform | Key strength | Key gap |
|---|---|---|---|---|
| **Kaggle arshkon** | Historical snapshot | LinkedIn | Has native entry-level labels | Small SWE count (4,691) |
| **Kaggle asaniczka** | Historical snapshot | LinkedIn | Large volume (18,129) | Zero native entry-level labels |
| **Scraped current-format** | Growing current window | LinkedIn + Indeed | Fresh data; search metadata | Growing daily; text coverage 30.7% |

**Platform policy:** LinkedIn is the primary analysis platform. Indeed is included for sensitivity analyses only. Both Kaggle sources are LinkedIn-only, making LinkedIn the cleanest cross-period comparison surface.

**Date ranges (T19):**

- arshkon: 2024-04-05 to 2024-04-20
- asaniczka: 2024-01-12 to 2024-01-17
- scraped: 2026-03-20 to 2026-04-14

**Excluded data:** YC postings, Apify data, and the old scraped format which used 25 results/query and lacked search metadata columns.

## Default analytical frame

Every headline finding in the exploration uses this filter:

```sql
WHERE is_swe = TRUE
  AND source_platform = 'linkedin'
  AND is_english = TRUE
  AND date_flag = 'ok'
```

SWE row counts under this frame:

| Source | SWE rows |
|---|---|
| arshkon (2024) | 4,691 |
| asaniczka (2024) | 18,129 |
| scraped (2026) | 40,881 |
| **Total** | **63,701** |

## Sample validation (T07)

- **BLS 15-1252 software-developer employment vs our sample metro counts: Pearson r = 0.97** across 18 qualifying metros (n ≥ 50 SWE per period).
- **Overlap panel sizes:** n = 240 at ≥ 3 SWE per period; n = 125 at ≥ 5 SWE per period.
- **Power analysis:** pooled-2024 entry-vs-scraped MDE = 8.2 pp (marginal); arshkon-only MDE = 11.3 pp (underpowered).
- **JOLTS info-sector openings:** dropped 29% between windows — macro cooling is the dominant backdrop for any junior-share metric. Addressed by T19 macro-robustness ratios.

## Binding constraints on the 2026 side

Three coverage gaps that determine which analyses can run on 2026 scraped data:

| Gap | Value | Affects |
|---|---|---|
| Stage 9 LLM text cleaning coverage | **30.7%** labeled | All text-based analyses on 2026 side |
| Stage 10 LLM seniority coverage | 53% `seniority_final = 'unknown'` | Denominator drift on any `of known` entry-share comparison |
| T09 archetype labels on scraped | **30.5%** coverage | All within-archetype 2026-side claims |

**Practical implication:** text-sensitive 2026 analyses cap at ~12,500 rows. Denominator-aware analyses should report "of labeled" and "of all" side by side.

## Known data confounders

Summarized from SYNTHESIS Section 6; see [Limitations](limitations.md) for severity ratings.

- Length growth is mostly style migration (not scope expansion).
- Kaggle text is unformatted prose; scraped text is markdown — instrument confound on bullet density / em-dashes.
- Asaniczka has zero native entry labels — `seniority_native` is NOT pool-able.
- Aggregator + entry-specialist intermediary contamination drives entry-share aggregates.
- Company composition shift (new-entrant wave) requires Kitagawa-style decomposition.
- Markdown-escape bug: `c\+\+`, `c\#`, `\.net` are silently dropped in scraped text. Preprocessing fix pending.

## What this page does NOT cover

- How the data is transformed → [Preprocessing pipeline](preprocessing-pipeline.md).
- Seniority validation → [Sensitivity framework](sensitivity-framework.md).
- Schema column reference → see the repository's `docs/preprocessing-schema.md`.
