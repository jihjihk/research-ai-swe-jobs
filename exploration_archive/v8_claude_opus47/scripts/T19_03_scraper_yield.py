"""T19 Step 3 — Scraper yield + Step 5 within-scraped-window stability.

Daily SWE counts across scrape dates. First day may be an accumulated backlog.
Also: are AI-mention, J2/S1 shares, length stable across scraped dates?
Includes day-of-week split.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
ART_T18 = ROOT / "exploration" / "artifacts" / "T18"
TAB = ROOT / "exploration" / "tables" / "T19"
FIG = ROOT / "exploration" / "figures" / "T19"
FIG.mkdir(parents=True, exist_ok=True)


def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")

    sql = """
    SELECT uid, source, period, date_posted, scrape_date, posting_age_days,
      seniority_final, is_swe,
      CASE WHEN seniority_final IN ('entry','associate') THEN 1 ELSE 0 END AS J2,
      CASE WHEN seniority_final IN ('mid-senior','director') THEN 1 ELSE 0 END AS S1
    FROM read_parquet('/home/jihgaboot/gabor/job-research/data/unified.parquet')
    WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok'
      AND is_swe=true AND source='scraped'
    """
    df = con.execute(sql).df()
    df["scrape_date"] = pd.to_datetime(df["scrape_date"])
    print("Scraped SWE rows:", len(df))

    feat = pd.read_parquet(ART_T18 / "T18_posting_features.parquet")
    feat_cols = ["uid", "ai_strict_binary", "ai_broad_binary", "desc_len_chars",
                 "tech_count", "org_scope_count", "requirement_breadth"]
    merged = df.merge(feat[feat["group"] == "SWE"][feat_cols], on="uid", how="left")

    # Daily yield
    daily = (
        merged.groupby(merged["scrape_date"].dt.date)
        .agg(
            n=("uid", "count"),
            ai_strict=("ai_strict_binary", "mean"),
            ai_broad=("ai_broad_binary", "mean"),
            J2_share=("J2", "mean"),
            S1_share=("S1", "mean"),
            desc_len=("desc_len_chars", "mean"),
            tech_count=("tech_count", "mean"),
            requirement_breadth=("requirement_breadth", "mean"),
        )
        .reset_index()
        .rename(columns={"scrape_date": "scrape_day"})
    )
    daily["dow"] = pd.to_datetime(daily["scrape_day"]).dt.day_name()
    daily.to_csv(TAB / "T19_scraper_yield_daily.csv", index=False)

    print("\n=== Scraper daily yield (SWE) ===")
    print(daily.to_string())

    # First-day check — is it a backlog?
    first = daily.iloc[0]["n"] if len(daily) else 0
    rest_mean = daily.iloc[1:]["n"].mean() if len(daily) > 1 else None
    print(f"\nFirst day SWE postings: {first}")
    print(f"Mean of rest: {rest_mean:.1f}" if rest_mean is not None else "")
    if rest_mean:
        print(f"Ratio: {first / rest_mean:.2f}×")

    # Posting age
    age = merged["posting_age_days"].dropna()
    age_dist = age.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print("\n=== posting_age_days distribution ===")
    print(age_dist)
    age_dist.to_csv(TAB / "T19_posting_age_distribution.csv")

    # Day-of-week effects
    dow = (
        merged.assign(dow=merged["scrape_date"].dt.day_name())
        .groupby("dow")
        .agg(
            n=("uid", "count"),
            ai_strict=("ai_strict_binary", "mean"),
            ai_broad=("ai_broad_binary", "mean"),
            J2_share=("J2", "mean"),
            S1_share=("S1", "mean"),
            desc_len=("desc_len_chars", "mean"),
            tech_count=("tech_count", "mean"),
        )
        .reset_index()
    )
    dow.to_csv(TAB / "T19_day_of_week.csv", index=False)
    print("\n=== Day-of-week effects ===")
    print(dow.to_string())

    # Stability summary
    stab_metrics = ["ai_strict", "ai_broad", "J2_share", "S1_share",
                    "desc_len", "tech_count", "requirement_breadth"]
    stab = daily.iloc[1:][stab_metrics].agg(["mean", "std", "min", "max"]).T
    stab["cv"] = stab["std"] / stab["mean"]
    stab["range"] = stab["max"] - stab["min"]
    stab.to_csv(TAB / "T19_within_scraped_stability.csv")
    print("\n=== Within-scraped stability (excluding first day) ===")
    print(stab.to_string())


if __name__ == "__main__":
    main()
