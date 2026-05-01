"""
T07 Part A: Metro-level feasibility.

For each metro, count SWE postings per period × source. Classify metros as:
  - qualifies_gt50 : >= 50 SWE in both 2024 pooled and 2026 scraped
  - qualifies_gt100 : >= 100 SWE in both
  - arshkon_only_gt50 : >= 50 in arshkon 2024 and scraped 2026 (stricter)

Also reports multi-location exclusion.

Output: exploration/tables/T07/metro_feasibility.csv
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
TABLES = ROOT / "exploration" / "tables" / "T07"
TABLES.mkdir(parents=True, exist_ok=True)

BASE_WHERE = (
    "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe"
)


def main() -> None:
    # 1. Multi-location exclusion counts by source
    print("--- multi-location SWE postings excluded from per-metro rollups ---")
    mloc = duckdb.sql(
        f"""
        SELECT source,
               sum(CASE WHEN is_multi_location THEN 1 ELSE 0 END) n_multi_location,
               sum(CASE WHEN metro_area IS NULL AND NOT is_multi_location THEN 1 ELSE 0 END) n_unresolved_metro,
               sum(CASE WHEN metro_area IS NOT NULL THEN 1 ELSE 0 END) n_with_metro,
               count(*) n_total
        FROM '{DATA}'
        WHERE {BASE_WHERE}
        GROUP BY source
        ORDER BY source
        """
    ).df()
    print(mloc.to_string(index=False))
    mloc.to_csv(TABLES / "metro_exclusion_summary.csv", index=False)

    # 2. Per-metro × per-source SWE counts
    counts = duckdb.sql(
        f"""
        SELECT metro_area,
               source,
               count(*) n
        FROM '{DATA}'
        WHERE {BASE_WHERE}
          AND metro_area IS NOT NULL
        GROUP BY metro_area, source
        """
    ).df()
    wide = counts.pivot_table(index="metro_area", columns="source", values="n", fill_value=0).reset_index()
    wide["pooled_2024"] = wide.get("kaggle_arshkon", 0) + wide.get("kaggle_asaniczka", 0)
    # Handle missing columns defensively
    for col in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]:
        if col not in wide.columns:
            wide[col] = 0

    wide["qual_gt50_pooled"] = (wide["pooled_2024"] >= 50) & (wide["scraped"] >= 50)
    wide["qual_gt100_pooled"] = (wide["pooled_2024"] >= 100) & (wide["scraped"] >= 100)
    wide["qual_gt50_arshkon"] = (wide["kaggle_arshkon"] >= 50) & (wide["scraped"] >= 50)
    wide["qual_gt100_arshkon"] = (wide["kaggle_arshkon"] >= 100) & (wide["scraped"] >= 100)
    wide = wide.sort_values("scraped", ascending=False)

    wide.to_csv(TABLES / "metro_counts.csv", index=False)

    print()
    print("--- per-metro SWE counts by source (sorted by scraped desc) ---")
    print(wide.to_string(index=False))

    print()
    print("--- metro feasibility summary ---")
    summary = pd.DataFrame(
        [
            {
                "criterion": ">=50 SWE in pooled 2024 AND scraped",
                "n_metros": int(wide["qual_gt50_pooled"].sum()),
                "metros": ", ".join(wide.loc[wide["qual_gt50_pooled"], "metro_area"].tolist()),
            },
            {
                "criterion": ">=100 SWE in pooled 2024 AND scraped",
                "n_metros": int(wide["qual_gt100_pooled"].sum()),
                "metros": ", ".join(wide.loc[wide["qual_gt100_pooled"], "metro_area"].tolist()),
            },
            {
                "criterion": ">=50 SWE in arshkon-only 2024 AND scraped",
                "n_metros": int(wide["qual_gt50_arshkon"].sum()),
                "metros": ", ".join(wide.loc[wide["qual_gt50_arshkon"], "metro_area"].tolist()),
            },
            {
                "criterion": ">=100 SWE in arshkon-only 2024 AND scraped",
                "n_metros": int(wide["qual_gt100_arshkon"].sum()),
                "metros": ", ".join(wide.loc[wide["qual_gt100_arshkon"], "metro_area"].tolist()),
            },
        ]
    )
    print(summary.to_string(index=False))
    summary.to_csv(TABLES / "metro_qualification_summary.csv", index=False)


if __name__ == "__main__":
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_colwidth", 100)
    pd.set_option("display.width", 200)
    main()
