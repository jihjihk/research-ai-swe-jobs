# Factcheck 07 — "The democratiser that asks for more experience"

Source: `exploration/tables/T21/cluster_assignments.csv` (33,693 senior rows) joined on `uid` to `data/unified.parquet` (100% join coverage; T34_run.py script uses `unified.parquet`, not `unified_core.parquet` — the latter has 77% coverage and would bias results). Verification via DuckDB 1.5.0.

| # | Claim | Computed | Match? |
|---|---|---|---|
| 1 | Cluster 0: 144 (2024) -> 2,251 (2026), 15.6x, 94% 2026 | 144 -> 2,251, 15.63x, 94.0% | yes |
| 2 | Median YOE c0=6.0 vs c1=5.0 (labeled only) | c0=6.0 (n_labeled=1,511) vs c1=5.0 (n=5,550) | yes |
| 3 | Director share c0=1.9% vs c1=1.0% (2x) | c0=46/2,395=1.921% vs c1=74/7,292=1.015% (1.89x) | yes |
| 4 | 2026 industry mix: 44.6% SWE Dev + 16.5% Financial Services | 44.57% + 16.47% (normalized within top-10, matching T34_run.py L421-422) | yes |
| 5 | 1,163 distinct firms; HHI 38.6 | 1,163 (canonical); HHI=38.63 (shares^2 x 10000) | yes |

**Verdict: 5/5 matches.**

**Material discrepancies:** None. Reproduction exact to the first decimal.

**Note:** The industry-mix denominator is the sum of the top-10 industry counts in cluster 0 2026 (n=1,445), not the full cluster 0 2026 row count (n=2,251). This is the T34 methodology (see `exploration/scripts/T34_run.py` L420-422) and should be stated explicitly in Piece 07 if readers might misread "44.6% Software Development" as a share of all cluster 0 postings (raw share is 28.6%).
