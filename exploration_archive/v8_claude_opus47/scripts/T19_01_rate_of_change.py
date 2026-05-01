"""T19 Step 1 — Rate-of-change table.

For each metric (J2, S1, AI-mention strict, median cleaned desc len, median tech,
org-scope density), compute:
  - value in each source window (arshkon, asaniczka, scraped_2026-03, scraped_2026-04)
  - within-2024 difference (arshkon - asaniczka) annualized given date ranges
  - cross-period change (pooled-2024 mean → scraped mean) annualized (~23 months)
  - acceleration ratio = cross_annual / within_2024_annual

Uses primarily J2 and S1 per Gate-2 locks; also includes J1 and S3 as robustness.
Loads the T18 feature parquet for AI-mention/tech/scope features.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
ART_T18 = ROOT / "exploration" / "artifacts" / "T18"
TAB = ROOT / "exploration" / "tables" / "T19"
TAB.mkdir(parents=True, exist_ok=True)


def date_midpoint(start: str, end: str) -> datetime:
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    return s + (e - s) / 2


def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")

    # Seniority
    sql = """
    SELECT uid, source, period, date_posted, scrape_date,
           is_swe,
           seniority_final, yoe_extracted,
           title_normalized
    FROM read_parquet('/home/jihgaboot/gabor/job-research/data/unified.parquet')
    WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok'
      AND is_swe=true
    """
    sen = con.execute(sql).df()
    print("SWE rows:", len(sen))

    # Load T18 features to get ai_strict, desc_len_cleaned, tech_count, org_scope
    feat = pd.read_parquet(ART_T18 / "T18_posting_features.parquet")
    feat_swe = feat[feat["group"] == "SWE"][
        ["uid", "ai_strict_binary", "ai_broad_binary", "desc_len_chars", "desc_len_cleaned",
         "tech_count", "ai_tech_count", "org_scope_count", "mgmt_strict_count", "requirement_breadth",
         "llm_extraction_coverage"]
    ]

    df = sen.merge(feat_swe, on="uid", how="left")

    # Seniority operationalizations
    df["J1"] = (df["seniority_final"] == "entry").astype(int)
    df["J2"] = df["seniority_final"].isin(["entry", "associate"]).astype(int)
    df["S1"] = df["seniority_final"].isin(["mid-senior", "director"]).astype(int)

    # Date ranges (observed mid-points from data)
    source_windows = {
        "arshkon": {"period": "2024-04", "date_min": "2024-04-05", "date_max": "2024-04-20"},
        "asaniczka": {"period": "2024-01", "date_min": "2024-01-12", "date_max": "2024-01-17"},
        "scraped_2026-03": {"period": "2026-03", "date_min": "2026-03-19", "date_max": "2026-03-30"},
        "scraped_2026-04": {"period": "2026-04", "date_min": "2026-03-31", "date_max": "2026-04-13"},
    }
    source_mid = {name: date_midpoint(w["date_min"], w["date_max"]) for name, w in source_windows.items()}

    metrics = {
        "J2_share": ("J2", "mean"),
        "S1_share": ("S1", "mean"),
        "J1_share": ("J1", "mean"),
        "ai_strict": ("ai_strict_binary", "mean"),
        "ai_broad": ("ai_broad_binary", "mean"),
        "desc_len_raw_median": ("desc_len_chars", "median"),
        "desc_len_cleaned_median": ("desc_len_cleaned_labeled", "median"),  # special: labeled only
        "tech_count_median": ("tech_count", "median"),
        "org_scope_density_median": ("org_scope_count", "median"),
        "ai_tech_count_median": ("ai_tech_count", "median"),
    }

    # Build per-source values
    rows = []
    for metric_name, (col, agg) in metrics.items():
        rec = {"metric": metric_name}
        for name, w in source_windows.items():
            sub = df[df["period"] == w["period"]]
            # For scraped, split by period; for arshkon/asaniczka, by source match
            if name == "arshkon":
                sub = sub[sub["source"] == "kaggle_arshkon"]
            if name == "asaniczka":
                sub = sub[sub["source"] == "kaggle_asaniczka"]
            if metric_name == "desc_len_cleaned_median":
                sub2 = sub[sub["llm_extraction_coverage"] == "labeled"]
                if len(sub2) == 0:
                    rec[name] = np.nan
                else:
                    rec[name] = sub2["desc_len_cleaned"].median()
            else:
                if len(sub) == 0:
                    rec[name] = np.nan
                else:
                    if agg == "mean":
                        rec[name] = sub[col].mean()
                    elif agg == "median":
                        rec[name] = sub[col].median()

        # Within-2024 change: arshkon - asaniczka divided by years between their midpoints
        delta_within = rec["arshkon"] - rec["asaniczka"]
        dt_within_days = (source_mid["arshkon"] - source_mid["asaniczka"]).days
        within_years = dt_within_days / 365.25
        rec["within_2024_delta"] = delta_within
        rec["within_2024_days"] = dt_within_days
        rec["within_2024_annualized"] = delta_within / within_years if within_years != 0 else np.nan

        # Cross-period change: scraped_2026-04 - pooled-2024 mean
        pooled_2024 = np.nanmean([rec["arshkon"], rec["asaniczka"]])
        scraped_mean = np.nanmean([rec["scraped_2026-03"], rec["scraped_2026-04"]])
        delta_cross = scraped_mean - pooled_2024
        # Midpoint diff (pooled_2024 mid ~= 2024-02-29 to 2024-04-12 average; use asaniczka+arshkon avg)
        pooled_mid = source_mid["asaniczka"] + (source_mid["arshkon"] - source_mid["asaniczka"]) / 2
        scraped_mid = source_mid["scraped_2026-03"] + (source_mid["scraped_2026-04"] - source_mid["scraped_2026-03"]) / 2
        dt_cross_days = (scraped_mid - pooled_mid).days
        cross_years = dt_cross_days / 365.25
        rec["cross_period_delta"] = delta_cross
        rec["cross_period_days"] = dt_cross_days
        rec["cross_period_annualized"] = delta_cross / cross_years if cross_years != 0 else np.nan
        rec["acceleration_ratio"] = (
            rec["cross_period_annualized"] / rec["within_2024_annualized"]
            if rec["within_2024_annualized"] not in (0, None) and not np.isnan(rec["within_2024_annualized"]) else np.nan
        )
        rows.append(rec)

    roc = pd.DataFrame(rows)
    roc.to_csv(TAB / "T19_rate_of_change.csv", index=False)
    print("\n=== Rate-of-change table (SWE only) ===")
    cols_show = [
        "metric", "asaniczka", "arshkon", "scraped_2026-03", "scraped_2026-04",
        "within_2024_annualized", "cross_period_annualized", "acceleration_ratio"
    ]
    print(roc[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
