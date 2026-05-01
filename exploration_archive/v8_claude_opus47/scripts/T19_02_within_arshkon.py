"""T19 Step 2 — Within-arshkon stability.

Bin arshkon rows by date_posted (2024-04-05 to 2024-04-20 is ~16 days).
Question: do AI-mention, desc length, tech count vary materially across bins?
If yes, the "arshkon" data point has within-window heterogeneity that
biases within-2024 comparisons.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
ART_T18 = ROOT / "exploration" / "artifacts" / "T18"
TAB = ROOT / "exploration" / "tables" / "T19"
TAB.mkdir(parents=True, exist_ok=True)


def main():
    feat = pd.read_parquet(ART_T18 / "T18_posting_features.parquet")
    swe = feat[(feat["group"] == "SWE") & (feat["source"] == "kaggle_arshkon")].copy()
    swe["date_posted"] = pd.to_datetime(swe["date_posted"])
    swe = swe.dropna(subset=["date_posted"])

    METRICS = ["ai_strict_binary", "ai_broad_binary", "desc_len_chars",
               "tech_count", "org_scope_count", "requirement_breadth"]

    # Aggregate by day
    daily = (
        swe.assign(d=swe["date_posted"].dt.date)
        .groupby("d")[METRICS]
        .agg(["mean", "count"])
        .reset_index()
    )
    daily.columns = ["_".join([c for c in col if c]).rstrip("_") for col in daily.columns.values]
    daily.to_csv(TAB / "T19_arshkon_daily.csv", index=False)

    print("=== Arshkon SWE daily stability ===")
    disp_cols = ["d", "ai_strict_binary_mean", "ai_strict_binary_count",
                 "ai_broad_binary_mean",
                 "desc_len_chars_mean", "tech_count_mean",
                 "requirement_breadth_mean"]
    print(daily[disp_cols].to_string())

    # Stability summary — CV across days
    summary = {}
    for m in METRICS:
        col = f"{m}_mean"
        summary[m] = {
            "mean_across_days": daily[col].mean(),
            "std_across_days": daily[col].std(),
            "cv": daily[col].std() / daily[col].mean() if daily[col].mean() != 0 else None,
            "min": daily[col].min(),
            "max": daily[col].max(),
            "range": daily[col].max() - daily[col].min(),
        }
    ss = pd.DataFrame(summary).T
    ss.to_csv(TAB / "T19_arshkon_stability_summary.csv")
    print("\n=== Stability summary (across days) ===")
    print(ss.to_string())


if __name__ == "__main__":
    main()
