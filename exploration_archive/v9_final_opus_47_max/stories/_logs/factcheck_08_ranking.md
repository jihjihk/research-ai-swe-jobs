# Fact-check: Piece 08 "The market got the pattern right"

**Date:** 2026-04-21 **Checker:** Claude Opus 4.7 (max reasoning)
**Sources:** `exploration/reports/T32.md`; `exploration/tables/T32/*.csv`; `exploration/reports/V2_verification.md`; `exploration/tables/V2/H_w3_2_did_robustness.csv`; `exploration/reports/T18.md`.

## Independently recomputed numbers

Loaded `subgroup_divergence.csv` (n=20), filtered to 16 rows with worker benchmark (`rate_any_mid` not null), and ran SciPy `spearmanr`:

| Quantity | Piece claim | T32 saved | V2 replication | My recomputation |
|---|---|---|---|---|
| Spearman(worker_mid, employer_2026) | **+0.92** | 0.9233 | 0.923 | **0.9233 (p≈3e-7)** |
| Spearman(worker_mid, gap_2026) | **+0.71** (p=0.002) | 0.7094, p=0.00208 | 0.709 | **0.7094 (p=0.00208)** |
| 16/16 positive gap 2024 any-mid | YES | 16/16 | — | **16/16 confirmed** |
| 16/16 positive gap 2026 any-mid | YES | 16/16 | — | **16/16 confirmed** |

## Specific rates

From `subgroup_rates.csv`:
- **Accountant 2026:** n=2,910, ai_strict_rate = 0.006873 = **0.69%**. Worker benchmark 0.50. Ratio 0.50 / 0.006873 = **72.75x**. Piece "72x, worker 50%, employer 0.69%" **matches exactly**.
- **Nurse 2026:** n=**6,801**, ai_strict_rate = **0.0000 (literal zero)**. Piece "0.00% on 6,801 postings" **matches exactly**.

## SWE DiD

From `tables/V2/H_w3_2_did_robustness.csv` row `primary_SWE-CTL`: DiD = 13.1043 pp, CI [12.7626, 13.4461] — this is the V1-rebuilt pattern. T18.md row 22 reports **+14.02 pp, CI [+13.67, +14.37]** under top-level `ai_strict` (0.86 precision). V2 line 34 confirms: "+13.10 pp (V1); +14.04 pp (top); Agent +14.02 pp; Δ +0.02 pp". Piece cites the top-level number.

## Verdict

**MATCHES with one qualification.** All five T32-derived claims (Spearman 0.92, Spearman 0.71, 16/16 universality, accountant 72x, nurse 0.00% / 6,801) replicate to three decimal places against the saved tables and my own computation. SWE DiD +14.02 pp [13.67, 14.37] is the **top-level pattern** number; V2 reports the corresponding V1-rebuilt figure as +13.10 pp [12.76, 13.45]. The piece's Evidence Block footnote 5 already acknowledges both patterns; footnote 3 cites the top-level DiD only — acceptable but Gate 3 has flagged that T18's report text labels the pattern as "V1-rebuilt" while code uses top-level (V2 §3a). Direction robust under both.

**Note:** The piece is numerically clean; the only caveat is that the DiD CI corresponds to the top-level `ai_strict` pattern, which V2 reconciled as a pattern-label documentation issue not a content error.
