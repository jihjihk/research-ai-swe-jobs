"""T13 supplementary: run the same readability + tone metrics on
``description_core_llm`` (LLM-cleaned text, labeled subset only).

This is the sensitivity for dim (d) text_source. The primary T13 run uses raw
``description`` because the section-anatomy boilerplate question requires raw.
This supplement reports readability and tone on the LLM-cleaned text so that
direction-consistency with the primary can be confirmed.

Output:
    tables/T13/readability_by_period_seniority_core_llm.csv
    tables/T13/tone_by_period_seniority_core_llm.csv
    tables/T13/text_source_sensitivity.csv   (raw vs core_llm side-by-side)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent.parent
sys.path.insert(0, str(THIS_DIR))

from T13_run import (  # type: ignore  # noqa: E402
    compute_metrics,
    READABILITY_COLS,
    TONE_COLS,
    SENIORITY_BUCKETS,
    seniority_mask,
)

UNIFIED = REPO / "data" / "unified.parquet"
OUT_TABLES = REPO / "exploration" / "tables" / "T13"
T13_PARQUET = REPO / "exploration" / "artifacts" / "shared" / "T13_readability_metrics.parquet"


def main():
    con = duckdb.connect()
    q = f"""
    SELECT
      uid, source,
      CASE WHEN source LIKE 'kaggle_%' THEN '2024' ELSE '2026' END AS period_year,
      seniority_final, yoe_min_years_llm, llm_classification_coverage,
      description_core_llm
    FROM read_parquet('{UNIFIED.as_posix()}')
    WHERE is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok'
      AND description_core_llm IS NOT NULL
      AND length(description_core_llm) >= 50
      AND llm_extraction_coverage='labeled'
    """
    df = con.execute(q).df()
    print(f"n core-llm labeled = {len(df):,}")

    yoe = df["yoe_min_years_llm"]
    labeled = df["llm_classification_coverage"] == "labeled"
    df["is_J3"] = (yoe <= 2) & yoe.notna() & labeled
    df["is_S4"] = (yoe >= 5) & yoe.notna() & labeled
    df["is_J1"] = df["seniority_final"].eq("entry")
    df["is_S1"] = df["seniority_final"].eq("mid-senior")

    rows = []
    t0 = time.time()
    for i, row in df.reset_index(drop=True).iterrows():
        m = compute_metrics(row["description_core_llm"])
        out = {
            "uid": row["uid"],
            "source": row["source"],
            "period_year": row["period_year"],
            "is_J3": bool(row["is_J3"]),
            "is_S4": bool(row["is_S4"]),
            "is_J1": bool(row["is_J1"]),
            "is_S1": bool(row["is_S1"]),
        }
        out.update(m)
        rows.append(out)
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(df) - i - 1) / rate
            print(f"  {i+1}/{len(df)}  {rate:.1f}/s  ETA {eta:.0f}s")
    feats = pd.DataFrame(rows)

    # Aggregate
    def agg(cols, how="mean"):
        rows = []
        for period in ("2024", "2026"):
            dp = feats[feats["period_year"] == period]
            for sen in SENIORITY_BUCKETS:
                mask = seniority_mask(dp, sen)
                dps = dp[mask]
                r = {"period": period, "seniority": sen, "n": int(len(dps))}
                for c in cols:
                    if how == "mean":
                        r[c] = float(dps[c].mean()) if len(dps) else float("nan")
                rows.append(r)
        return pd.DataFrame(rows)

    agg(READABILITY_COLS).to_csv(
        OUT_TABLES / "readability_by_period_seniority_core_llm.csv", index=False
    )
    agg(TONE_COLS).to_csv(
        OUT_TABLES / "tone_by_period_seniority_core_llm.csv", index=False
    )

    # Side-by-side sensitivity: raw vs core_llm
    raw_feats = pd.read_parquet(T13_PARQUET)
    raw_agg = raw_feats.groupby("period_year")[READABILITY_COLS + TONE_COLS].mean().reset_index()
    core_agg = feats.groupby("period_year")[READABILITY_COLS + TONE_COLS].mean().reset_index()
    raw_agg["text_source"] = "raw"
    core_agg["text_source"] = "core_llm"
    combined = pd.concat([raw_agg, core_agg]).reset_index(drop=True)
    combined.to_csv(OUT_TABLES / "text_source_sensitivity.csv", index=False)
    print("Supplement done.")


if __name__ == "__main__":
    main()
