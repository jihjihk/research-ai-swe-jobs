# Fact-check: Piece 10 "Where the middle narrowed"

## Method
Independently re-derived 5-bucket shares via DuckDB on `data/unified_core.parquet`.
Filter: `source_platform='linkedin' AND is_swe AND is_english AND date_flag='ok' AND llm_extraction_coverage='labeled'`.
`pooled-2024` = kaggle_arshkon + kaggle_asaniczka; `pooled-2026` = scraped. Bucketing on `yoe_min_years_llm` (NULL -> unknown).

## Pooled-2024 shares (n=22,796)
- A (0-2): 7.31%
- B (3-4): 16.03%
- C (5-7): 35.15%
- D (8-10): 17.36%
- E (11+): 4.01%

## 2026-scraped shares (n=25,650)
- A: 11.26%
- B: 17.94%
- C: 33.27%
- D: 14.03%
- E: 2.87%

## pp changes (scraped minus pooled-2024)
- A: +3.95 pp (claim +3.97)
- B: +1.91 pp (claim +1.94)
- C: -1.88 pp (claim -1.88)
- D: -3.33 pp (claim -3.32)
- E: -1.14 pp (claim -1.13)

Denominators differ from the CSV by ~15-165 rows (22,796 vs 22,811; 25,650 vs 25,815), likely a minor filter nuance (e.g. boolean NA handling on `is_english`). Shares still agree to 0.03 pp across all five buckets.

## Arshkon-only robustness pp change (re-derived)
- A +1.68 (claim +1.70), B +1.75 (+1.78), C +2.69 (+2.70), D -1.01 (-1.00), E +0.57 (+0.58). All within 0.03 pp.

## Biggest negative
Bucket D (8-10 YOE) at -3.33 pp is the largest negative change under the pooled baseline.

## Is D the only bucket negative under BOTH baselines?
Confirmed. Pooled baseline: C, D, E all negative (-1.88, -3.33, -1.14). Arshkon baseline: only D is negative (-1.01); C and E flip positive (+2.69, +0.57). D is the sole bucket negative in both -> the claim holds.

## Verdict
Matches. Headline numbers, ranking, and the "D-only under both baselines" structural claim all replicate.

## Caveat on YOE extraction
LLM-extracted `yoe_min_years_llm` is integer-rounded, so postings at bucket boundaries (e.g. "7-9 years", "10+", "8+") may land in an adjacent bucket; a small share of the -3.33 pp D drop could be boundary leakage into C or E rather than a true level shift.
