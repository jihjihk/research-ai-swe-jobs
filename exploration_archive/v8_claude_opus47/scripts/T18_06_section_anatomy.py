"""T18 Step 6 — Section anatomy cross-occupation.

Uses the T13 section classifier (importable) on a sampled 3000 postings per
(group, period) cell to ask: does the requirements-section shrink appear in
adjacent/control too, or is it SWE-specific?

For each posting, computes character share for each section type.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
sys.path.insert(0, str(ROOT / "exploration" / "scripts"))

from T13_section_classifier import SECTION_TYPES, classify_description  # noqa: E402

TAB = ROOT / "exploration" / "tables" / "T18"

SEED = 20260417
PER_CELL_N = 3000
PERIODS = ["2024-01", "2024-04", "2026-03", "2026-04"]


def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")

    sql = """
    SELECT uid, source, period, is_swe, is_swe_adjacent, is_control,
      description
    FROM read_parquet('/home/jihgaboot/gabor/job-research/data/unified.parquet')
    WHERE source_platform='linkedin'
      AND is_english=true
      AND date_flag='ok'
      AND (is_swe=true OR is_swe_adjacent=true OR is_control=true)
    """
    df = con.execute(sql).df()
    df["group"] = np.where(
        df["is_swe"], "SWE",
        np.where(df["is_swe_adjacent"], "adjacent",
                 np.where(df["is_control"], "control", None))
    )
    df = df[df["group"].notna()].copy()

    rng = np.random.default_rng(SEED)
    # Downsample per cell
    samples = []
    for period in PERIODS:
        for group in ["SWE", "adjacent", "control"]:
            sub = df[(df["period"] == period) & (df["group"] == group)]
            if len(sub) == 0:
                continue
            n = min(PER_CELL_N, len(sub))
            idx = rng.choice(len(sub), size=n, replace=False)
            samples.append(sub.iloc[idx])
    sub_df = pd.concat(samples, ignore_index=True)
    print("Sample size:", len(sub_df))
    print(sub_df.groupby(["group", "period"]).size())

    # Classify each description
    records = []
    for i, row in enumerate(sub_df.itertuples(index=False)):
        if i % 5000 == 0:
            print(f"  {i}/{len(sub_df)}")
        secs = classify_description(row.description or "")
        total = sum(s["chars"] for s in secs.values()) or 1
        rec = {"uid": row.uid, "group": row.group, "period": row.period,
               "total_chars": total}
        for st in SECTION_TYPES:
            rec[f"chars_{st}"] = secs[st]["chars"]
            rec[f"share_{st}"] = secs[st]["chars"] / total
        records.append(rec)

    sdf = pd.DataFrame(records)

    # Aggregate mean shares per (group, period)
    share_cols = [f"share_{st}" for st in SECTION_TYPES]
    char_cols = [f"chars_{st}" for st in SECTION_TYPES]
    agg = (
        sdf.groupby(["group", "period"], observed=False)[share_cols + char_cols + ["total_chars"]]
        .mean()
        .reset_index()
    )
    agg.to_csv(TAB / "T18_section_anatomy_by_group_period.csv", index=False)
    print("Saved section anatomy aggregate.")

    # Key question: change in requirements share per group
    pivot_rec = []
    for group in ["SWE", "adjacent", "control"]:
        pre_rec = sdf[(sdf["group"] == group) & (sdf["period"].isin(["2024-01", "2024-04"]))]
        post_rec = sdf[(sdf["group"] == group) & (sdf["period"].isin(["2026-03", "2026-04"]))]
        rec = {"group": group,
               "n_pre": len(pre_rec), "n_post": len(post_rec),
               "pre_total_chars": pre_rec["total_chars"].mean(),
               "post_total_chars": post_rec["total_chars"].mean()}
        for st in SECTION_TYPES:
            rec[f"pre_share_{st}"] = pre_rec[f"share_{st}"].mean()
            rec[f"post_share_{st}"] = post_rec[f"share_{st}"].mean()
            rec[f"delta_share_{st}"] = rec[f"post_share_{st}"] - rec[f"pre_share_{st}"]
            rec[f"pre_chars_{st}"] = pre_rec[f"chars_{st}"].mean()
            rec[f"post_chars_{st}"] = post_rec[f"chars_{st}"].mean()
            rec[f"delta_chars_{st}"] = rec[f"post_chars_{st}"] - rec[f"pre_chars_{st}"]
        pivot_rec.append(rec)
    piv = pd.DataFrame(pivot_rec)
    piv.to_csv(TAB / "T18_section_change_by_group.csv", index=False)

    print("\n=== Requirements section change ===")
    print(piv[[
        "group", "n_pre", "n_post",
        "pre_share_requirements", "post_share_requirements", "delta_share_requirements",
        "pre_chars_requirements", "post_chars_requirements", "delta_chars_requirements",
    ]].to_string())

    print("\n=== Responsibilities section change ===")
    print(piv[[
        "group", "pre_share_responsibilities", "post_share_responsibilities", "delta_share_responsibilities",
    ]].to_string())

    print("\n=== Total chars change ===")
    print(piv[["group", "pre_total_chars", "post_total_chars"]].to_string())


if __name__ == "__main__":
    main()
