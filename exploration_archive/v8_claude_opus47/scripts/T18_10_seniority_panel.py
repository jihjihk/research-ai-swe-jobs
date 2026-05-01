"""T18 Step 10 — Seniority panel per group.

For each group × period, report J1/J2/J3/J4 and S1/S2/S3/S4 shares
(share of all rows in the cell). Uses the T30 operationalizations but per
occupation group. Arshkon-only for junior per locked directive; pooled-2024
for robustness.
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
TAB = ROOT / "exploration" / "tables" / "T18"
TAB.mkdir(parents=True, exist_ok=True)

TITLE_SENIOR = re.compile(r"\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b", re.IGNORECASE)
TITLE_JUNIOR = re.compile(r"\b(junior|jr\.?|entry[- ]level|new grad|graduate|trainee|intern)\b", re.IGNORECASE)


def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    sql = """
    SELECT uid, source, period, is_swe, is_swe_adjacent, is_control,
      seniority_final, yoe_extracted, title_normalized
    FROM read_parquet('/home/jihgaboot/gabor/job-research/data/unified.parquet')
    WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok'
      AND (is_swe=true OR is_swe_adjacent=true OR is_control=true)
    """
    df = con.execute(sql).df()
    df["group"] = np.where(
        df["is_swe"], "SWE",
        np.where(df["is_swe_adjacent"], "adjacent",
                 np.where(df["is_control"], "control", None))
    )

    df["J1"] = (df["seniority_final"] == "entry").astype(int)
    df["J2"] = df["seniority_final"].isin(["entry", "associate"]).astype(int)
    df["J3"] = (df["yoe_extracted"].fillna(100) <= 2).astype(int)
    df["J4"] = (df["yoe_extracted"].fillna(100) <= 3).astype(int)
    df["J5"] = df["title_normalized"].fillna("").apply(lambda t: bool(TITLE_JUNIOR.search(t))).astype(int)

    df["S1"] = df["seniority_final"].isin(["mid-senior", "director"]).astype(int)
    df["S2"] = (df["seniority_final"] == "director").astype(int)
    df["S3"] = df["title_normalized"].fillna("").apply(lambda t: bool(TITLE_SENIOR.search(t))).astype(int)
    df["S4"] = (df["yoe_extracted"].fillna(-1) >= 5).astype(int)

    agg = (
        df.groupby(["group", "period", "source"])
        [["J1", "J2", "J3", "J4", "J5", "S1", "S2", "S3", "S4"]]
        .mean()
        .reset_index()
    )
    agg.to_csv(TAB / "T18_seniority_panel_by_group.csv", index=False)
    print(agg.to_string())


if __name__ == "__main__":
    main()
