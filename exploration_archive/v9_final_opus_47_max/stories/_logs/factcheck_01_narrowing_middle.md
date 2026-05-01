# Fact-check: Piece 01 "The narrowing middle that wasn't"

**Date:** 2026-04-21
**Scope:** Headline J3 entry-share delta and supervised AUC associate↔mid-senior sharpening.

## J3 entry share (YOE ≤ 2 among labeled LLM-YOE rows)

Filter: `source_platform='linkedin' AND is_swe AND is_english AND date_flag='ok' AND llm_extraction_coverage='labeled'`, numerator `yoe_min_years_llm <= 2`, denominator rows with `yoe_min_years_llm` not null (matches T30 panel `share_of_denominator`).

| Panel | Re-derived | T30 panel CSV |
|---|---|---|
| Pooled-2024 (asaniczka+arshkon) | 0.09150 (1666/18205) | 0.09150 |
| Arshkon-only 2024 | 0.12999 (449/3454) | 0.12999 |
| Pooled-2026 scraped | 0.14192 (2889/20360) | 0.14192 |

**Computed J3 pooled 2024 share:** 9.15% (share_of_denominator).
**Computed J3 2026 pooled scraped share:** 14.19%.
**Computed delta pooled:** +5.04 pp. **Arshkon-only baseline delta:** +1.19 pp.

## Supervised AUC

T20 (`exploration/reports/T20.md` §1): associate↔mid-senior AUC 0.743 → 0.893, Δ = **+0.150**. V2_verification replicates at +0.146 (within-run noise).

## Verdicts

- **J3 claim:** matches exactly (within <0.01 pp of the +5.0 pp and +1.19 pp stated).
- **AUC +0.150 claim:** matches T20 exactly; V2 replicates at +0.146.

**Explanation:** Re-derivation against `data/unified_core.parquet` reproduces the T30 panel row-for-row, and T20 §1 plus V2_verification corroborate the associate↔mid-senior +0.150 AUC sharpening.
