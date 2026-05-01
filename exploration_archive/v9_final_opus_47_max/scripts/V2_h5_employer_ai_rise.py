"""V2 Phase A — H_w3_5: Independent re-derivation of T23 employer AI-strict rise 10.3×.

Claim (T23): SWE ai_strict 1.03% (2024 pooled) → 10.61% (2026 scraped) — 10.3× ratio.
V2 applies V1-validated ai_strict_v1_rebuilt on raw description.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = str(ROOT / "data" / "unified.parquet")
VALIDATED = ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json"
OUT = ROOT / "exploration" / "tables" / "V2"

with open(VALIDATED) as f:
    p = json.load(f)
AI_V1 = p["v1_rebuilt_patterns"]["ai_strict_v1_rebuilt"]["pattern"]
AI_TOP = p["ai_strict"]["pattern"]


def fetch():
    q = f"""
    SELECT uid, source, period, LOWER(description) AS txt
    FROM '{UNIFIED}'
    WHERE is_swe=TRUE AND source_platform='linkedin' AND is_english=TRUE AND date_flag='ok'
    """
    con = duckdb.connect()
    df = con.execute(q).df()
    con.close()
    df["era"] = np.where(df["source"] == "scraped", "2026", "2024")
    return df


def main():
    df = fetch()

    rx_v1 = re.compile(AI_V1, flags=re.IGNORECASE)
    rx_top = re.compile(AI_TOP, flags=re.IGNORECASE)
    texts = df["txt"].fillna("").to_numpy()
    df["ai_v1"] = np.fromiter((1 if rx_v1.search(t) else 0 for t in texts), dtype=np.int8, count=len(texts))
    df["ai_top"] = np.fromiter((1 if rx_top.search(t) else 0 for t in texts), dtype=np.int8, count=len(texts))

    out = df.groupby("era").agg(
        n=("uid", "size"),
        ai_v1=("ai_v1", "mean"),
        ai_top=("ai_top", "mean"),
    ).reset_index()
    out["ai_v1_pct"] = out["ai_v1"] * 100
    out["ai_top_pct"] = out["ai_top"] * 100

    print("\nH_w3_5 T23 rise (raw description, default SWE LinkedIn filter):")
    print(out.to_string(index=False))

    # Ratio
    p24_v1 = out.loc[out.era == "2024", "ai_v1"].iloc[0]
    p26_v1 = out.loc[out.era == "2026", "ai_v1"].iloc[0]
    p24_top = out.loc[out.era == "2024", "ai_top"].iloc[0]
    p26_top = out.loc[out.era == "2026", "ai_top"].iloc[0]

    print(f"\nRatio (V1-rebuilt): {p24_v1*100:.2f}% → {p26_v1*100:.2f}% = {p26_v1/max(p24_v1,1e-6):.2f}×")
    print(f"Ratio (top-level):  {p24_top*100:.2f}% → {p26_top*100:.2f}% = {p26_top/max(p24_top,1e-6):.2f}×")

    out.to_csv(OUT / "H_w3_5_ai_rise.csv", index=False)


if __name__ == "__main__":
    main()
