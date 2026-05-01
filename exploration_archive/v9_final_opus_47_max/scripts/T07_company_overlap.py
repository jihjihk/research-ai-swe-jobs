"""
T07 Part A: Company overlap panel feasibility.

For companies that posted SWE jobs in both arshkon (2024) and scraped (2026),
count how many meet thresholds of >=3 and >=5 SWE postings in both periods.
This sizes T16's overlap panel and T31's same-co × same-title pair count.

Also compute same-co × same-title overlap.

Output:
  exploration/tables/T07/company_overlap_summary.csv
  exploration/tables/T07/company_overlap_companies.csv
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
    # 1. Per-company SWE counts by source
    print("--- per-company SWE counts by source ---")
    per_co = duckdb.sql(
        f"""
        SELECT company_name_canonical,
               source,
               count(*) n
        FROM '{DATA}'
        WHERE {BASE_WHERE}
          AND company_name_canonical IS NOT NULL
        GROUP BY company_name_canonical, source
        """
    ).df()
    wide = per_co.pivot_table(
        index="company_name_canonical", columns="source", values="n", fill_value=0
    ).reset_index()
    for col in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]:
        if col not in wide.columns:
            wide[col] = 0
    wide["pooled_2024"] = wide["kaggle_arshkon"] + wide["kaggle_asaniczka"]

    # 2. Thresholds
    # arshkon-only vs scraped overlap
    arshkon_scraped_3 = int(((wide["kaggle_arshkon"] >= 3) & (wide["scraped"] >= 3)).sum())
    arshkon_scraped_5 = int(((wide["kaggle_arshkon"] >= 5) & (wide["scraped"] >= 5)).sum())
    # pooled 2024 vs scraped overlap
    pooled_scraped_3 = int(((wide["pooled_2024"] >= 3) & (wide["scraped"] >= 3)).sum())
    pooled_scraped_5 = int(((wide["pooled_2024"] >= 5) & (wide["scraped"] >= 5)).sum())
    any_overlap = int(((wide["pooled_2024"] >= 1) & (wide["scraped"] >= 1)).sum())

    # aggregator-excluded versions
    per_co_no_agg = duckdb.sql(
        f"""
        SELECT company_name_canonical,
               source,
               count(*) n
        FROM '{DATA}'
        WHERE {BASE_WHERE}
          AND is_aggregator = false
          AND company_name_canonical IS NOT NULL
        GROUP BY company_name_canonical, source
        """
    ).df()
    wide_no_agg = per_co_no_agg.pivot_table(
        index="company_name_canonical", columns="source", values="n", fill_value=0
    ).reset_index()
    for col in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]:
        if col not in wide_no_agg.columns:
            wide_no_agg[col] = 0
    wide_no_agg["pooled_2024"] = wide_no_agg["kaggle_arshkon"] + wide_no_agg["kaggle_asaniczka"]
    arshkon_scraped_3_na = int(((wide_no_agg["kaggle_arshkon"] >= 3) & (wide_no_agg["scraped"] >= 3)).sum())
    arshkon_scraped_5_na = int(((wide_no_agg["kaggle_arshkon"] >= 5) & (wide_no_agg["scraped"] >= 5)).sum())
    pooled_scraped_3_na = int(((wide_no_agg["pooled_2024"] >= 3) & (wide_no_agg["scraped"] >= 3)).sum())
    pooled_scraped_5_na = int(((wide_no_agg["pooled_2024"] >= 5) & (wide_no_agg["scraped"] >= 5)).sum())
    any_overlap_na = int(((wide_no_agg["pooled_2024"] >= 1) & (wide_no_agg["scraped"] >= 1)).sum())

    summary = pd.DataFrame(
        [
            {"criterion": "any overlap (arshkon+asaniczka & scraped)", "n_companies_agg_included": any_overlap, "n_companies_agg_excluded": any_overlap_na},
            {"criterion": ">=3 SWE in arshkon-only AND scraped", "n_companies_agg_included": arshkon_scraped_3, "n_companies_agg_excluded": arshkon_scraped_3_na},
            {"criterion": ">=5 SWE in arshkon-only AND scraped", "n_companies_agg_included": arshkon_scraped_5, "n_companies_agg_excluded": arshkon_scraped_5_na},
            {"criterion": ">=3 SWE in pooled 2024 AND scraped", "n_companies_agg_included": pooled_scraped_3, "n_companies_agg_excluded": pooled_scraped_3_na},
            {"criterion": ">=5 SWE in pooled 2024 AND scraped", "n_companies_agg_included": pooled_scraped_5, "n_companies_agg_excluded": pooled_scraped_5_na},
        ]
    )
    print(summary.to_string(index=False))
    summary.to_csv(TABLES / "company_overlap_summary.csv", index=False)

    # 3. Same-co × same-title overlap for T31 feasibility
    print()
    print("--- same-co x same-title overlap (for T31 feasibility) ---")
    co_title = duckdb.sql(
        f"""
        SELECT company_name_canonical,
               title_normalized,
               sum(CASE WHEN source = 'kaggle_arshkon' THEN 1 ELSE 0 END) n_arshkon,
               sum(CASE WHEN source = 'kaggle_asaniczka' THEN 1 ELSE 0 END) n_asaniczka,
               sum(CASE WHEN source = 'scraped' THEN 1 ELSE 0 END) n_scraped
        FROM '{DATA}'
        WHERE {BASE_WHERE}
          AND company_name_canonical IS NOT NULL
          AND title_normalized IS NOT NULL
        GROUP BY company_name_canonical, title_normalized
        """
    ).df()
    co_title["pooled_2024"] = co_title["n_arshkon"] + co_title["n_asaniczka"]

    t31_rows = []
    for thresh in [1, 2, 3]:
        for arshkon_only in [True, False]:
            if arshkon_only:
                mask = (co_title["n_arshkon"] >= thresh) & (co_title["n_scraped"] >= thresh)
                label = f"arshkon-only: >={thresh} pair"
            else:
                mask = (co_title["pooled_2024"] >= thresh) & (co_title["n_scraped"] >= thresh)
                label = f"pooled 2024: >={thresh} pair"
            t31_rows.append({"criterion": label, "n_pairs": int(mask.sum())})
    t31_df = pd.DataFrame(t31_rows)
    print(t31_df.to_string(index=False))
    t31_df.to_csv(TABLES / "company_title_pair_summary.csv", index=False)

    # 4. Export top-N overlap companies (sorted by min(pooled, scraped))
    wide["min_overlap"] = wide[["pooled_2024", "scraped"]].min(axis=1)
    wide = wide.sort_values("min_overlap", ascending=False)
    wide.head(200).to_csv(TABLES / "company_overlap_top200.csv", index=False)
    print()
    print(f"Top-200 overlapping companies written to company_overlap_top200.csv")


if __name__ == "__main__":
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 200)
    main()
