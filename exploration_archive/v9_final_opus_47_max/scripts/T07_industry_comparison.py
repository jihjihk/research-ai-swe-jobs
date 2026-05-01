"""
T07 Part B: Industry distribution comparison.

Compares our SWE (arshkon + scraped, which have company_industry) against BLS OES
industry distribution for software developers (SOC 15-1252) — using the national
industry profile from the occupation_industry file (NAICS industry × occupation).

Output:
  exploration/tables/T07/industry_distribution.csv (ours)
  exploration/tables/T07/industry_shift_arshkon_vs_scraped.csv
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
TABLES = ROOT / "exploration" / "tables" / "T07"

BASE_WHERE = (
    "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe"
)


def main() -> None:
    # 1. Our industry distribution by source (arshkon + scraped)
    ind = duckdb.sql(
        f"""
        SELECT source, company_industry,
               count(*) AS n
        FROM '{DATA}'
        WHERE {BASE_WHERE}
          AND source IN ('kaggle_arshkon','scraped')
          AND company_industry IS NOT NULL AND company_industry <> ''
        GROUP BY source, company_industry
        """
    ).df()

    # Pivot to wide with shares
    wide = ind.pivot_table(index="company_industry", columns="source", values="n", fill_value=0).reset_index()
    for col in ["kaggle_arshkon", "scraped"]:
        if col not in wide.columns:
            wide[col] = 0
    wide["share_arshkon"] = wide["kaggle_arshkon"] / wide["kaggle_arshkon"].sum()
    wide["share_scraped"] = wide["scraped"] / wide["scraped"].sum()
    wide["pp_change"] = (wide["share_scraped"] - wide["share_arshkon"]) * 100

    wide = wide.sort_values("scraped", ascending=False)
    wide.to_csv(TABLES / "industry_distribution_ours.csv", index=False)

    # Shift analysis: largest gains and largest losses
    # Consider only industries with >= 50 postings in scraped (for stability)
    stable = wide[wide["scraped"] >= 50].copy()
    gains = stable.sort_values("pp_change", ascending=False).head(15)
    losses = stable.sort_values("pp_change", ascending=True).head(15)
    out = pd.concat([
        gains.assign(category="top_gainer"),
        losses.assign(category="top_loser")
    ])
    out = out[["category", "company_industry", "kaggle_arshkon", "scraped",
               "share_arshkon", "share_scraped", "pp_change"]]
    out.to_csv(TABLES / "industry_shift_arshkon_vs_scraped.csv", index=False)
    print("Top gainers (pp share change, arshkon -> scraped):")
    print(gains[["company_industry", "kaggle_arshkon", "scraped", "share_arshkon", "share_scraped", "pp_change"]].to_string(index=False))
    print()
    print("Top losers:")
    print(losses[["company_industry", "kaggle_arshkon", "scraped", "share_arshkon", "share_scraped", "pp_change"]].to_string(index=False))

    # 2. Composition verdict
    total_ours_arshkon = wide["kaggle_arshkon"].sum()
    total_ours_scraped = wide["scraped"].sum()
    software_dev_ark = int(wide.loc[wide["company_industry"] == "Software Development", "kaggle_arshkon"].sum())
    software_dev_scp = int(wide.loc[wide["company_industry"] == "Software Development", "scraped"].sum())
    it_consulting_ark = int(wide.loc[wide["company_industry"] == "IT Services and IT Consulting", "kaggle_arshkon"].sum())
    it_consulting_scp = int(wide.loc[wide["company_industry"] == "IT Services and IT Consulting", "scraped"].sum())

    summary = pd.DataFrame(
        [
            {
                "metric": "Software Development share — arshkon",
                "value_pct": round(100 * software_dev_ark / total_ours_arshkon, 2),
                "n": software_dev_ark,
            },
            {
                "metric": "Software Development share — scraped",
                "value_pct": round(100 * software_dev_scp / total_ours_scraped, 2),
                "n": software_dev_scp,
            },
            {
                "metric": "IT Services & IT Consulting share — arshkon",
                "value_pct": round(100 * it_consulting_ark / total_ours_arshkon, 2),
                "n": it_consulting_ark,
            },
            {
                "metric": "IT Services & IT Consulting share — scraped",
                "value_pct": round(100 * it_consulting_scp / total_ours_scraped, 2),
                "n": it_consulting_scp,
            },
            {
                "metric": "Staffing/Recruiting share — arshkon",
                "value_pct": round(100 * int(wide.loc[wide["company_industry"] == "Staffing and Recruiting", "kaggle_arshkon"].sum()) / total_ours_arshkon, 2),
                "n": int(wide.loc[wide["company_industry"] == "Staffing and Recruiting", "kaggle_arshkon"].sum()),
            },
            {
                "metric": "Staffing/Recruiting share — scraped",
                "value_pct": round(100 * int(wide.loc[wide["company_industry"] == "Staffing and Recruiting", "scraped"].sum()) / total_ours_scraped, 2),
                "n": int(wide.loc[wide["company_industry"] == "Staffing and Recruiting", "scraped"].sum()),
            },
        ]
    )
    summary.to_csv(TABLES / "industry_summary.csv", index=False)
    print()
    print("Key industry share summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.max_rows", 40)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)
    main()
