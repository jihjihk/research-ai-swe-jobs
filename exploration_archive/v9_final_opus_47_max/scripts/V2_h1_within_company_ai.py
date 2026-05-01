"""V2 Phase A — H_w3_1: Independent re-derivation of T16 within-company AI-strict rewriting.

Claim (T16): within-company AI-strict Δ = +7.65 (pooled_min5), +8.34 (arshkon_min5), +8.47 (arshkon_min3)
Uses T16's text source (description_cleaned from swe_cleaned_text.parquet) and the same pattern
T16 actually used (top-level `ai_strict` 0.86 precision — not ai_strict_v1_rebuilt as the report
text implies). V2 also reports BOTH patterns so we can flag the pattern-label gap.
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
CLEAN = str(ROOT / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet")
VALIDATED = ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json"
OUT_DIR = ROOT / "exploration" / "tables" / "V2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(VALIDATED) as f:
    p = json.load(f)
AI_STRICT_TOP = p["ai_strict"]["pattern"]
AI_STRICT_V1 = p["v1_rebuilt_patterns"]["ai_strict_v1_rebuilt"]["pattern"]

print("ai_strict (top):", AI_STRICT_TOP[:120])
print("ai_strict_v1_rebuilt:", AI_STRICT_V1[:120])


def load_joined() -> pd.DataFrame:
    q = f"""
    SELECT
        t.uid,
        t.company_name_canonical,
        t.source,
        t.period,
        t.text_source,
        CASE WHEN t.source='scraped' THEN '2026' ELSE '2024' END AS era,
        LOWER(t.description_cleaned) AS txt
    FROM read_parquet('{CLEAN}') t
    WHERE t.company_name_canonical IS NOT NULL
    """
    con = duckdb.connect()
    df = con.execute(q).df()
    con.close()
    print(f"Loaded {len(df):,} cleaned-text rows")
    return df


def apply_patterns(df: pd.DataFrame) -> pd.DataFrame:
    rx_top = re.compile(AI_STRICT_TOP, flags=re.IGNORECASE)
    rx_v1 = re.compile(AI_STRICT_V1, flags=re.IGNORECASE)
    texts = df["txt"].fillna("").to_numpy()
    df["ai_top"] = np.fromiter((1 if rx_top.search(t) else 0 for t in texts), dtype=np.int8, count=len(texts))
    df["ai_v1"] = np.fromiter((1 if rx_v1.search(t) else 0 for t in texts), dtype=np.int8, count=len(texts))
    return df


def build_panels(df: pd.DataFrame) -> dict:
    cos_24_arsh = df[df["source"] == "kaggle_arshkon"].groupby("company_name_canonical").size()
    cos_24_asan = df[df["source"] == "kaggle_asaniczka"].groupby("company_name_canonical").size()
    cos_26 = df[df["era"] == "2026"].groupby("company_name_canonical").size()

    p_a3 = set(cos_24_arsh[cos_24_arsh >= 3].index) & set(cos_26[cos_26 >= 3].index)
    p_a5 = set(cos_24_arsh[cos_24_arsh >= 5].index) & set(cos_26[cos_26 >= 5].index)
    cos_24_pooled = cos_24_arsh.add(cos_24_asan, fill_value=0)
    p_p5 = set(cos_24_pooled[cos_24_pooled >= 5].index) & set(cos_26[cos_26 >= 5].index)

    return {"arshkon_min3": p_a3, "arshkon_min5": p_a5, "pooled_min5": p_p5}


def within_between(df: pd.DataFrame, cos: set, src_24: list, col: str) -> dict:
    d = df[df["company_name_canonical"].isin(cos)].copy()
    d_24 = d[(d["era"] == "2024") & (d["source"].isin(src_24))]
    d_26 = d[d["era"] == "2026"]

    g24 = d_24.groupby("company_name_canonical").agg(n24=("uid", "size"), m24=(col, "mean"))
    g26 = d_26.groupby("company_name_canonical").agg(n26=("uid", "size"), m26=(col, "mean"))
    g = g24.join(g26, how="inner").dropna()
    g["s24"] = g["n24"] / g["n24"].sum()
    g["s26"] = g["n26"] / g["n26"].sum()
    g["sym"] = (g["s24"] + g["s26"]) / 2

    agg_24 = d_24[col].mean()
    agg_26 = d_26[col].mean()
    total = agg_26 - agg_24
    within = (g["sym"] * (g["m26"] - g["m24"])).sum()
    between = total - within
    return {
        "n_cos_overlap": int(len(g)),
        "n_2024": int(d_24.shape[0]),
        "n_2026": int(d_26.shape[0]),
        "agg_24_pct": float(agg_24 * 100),
        "agg_26_pct": float(agg_26 * 100),
        "total_delta_pp": float(total * 100),
        "within_delta_pp": float(within * 100),
        "between_delta_pp": float(between * 100),
    }


def main():
    df = apply_patterns(load_joined())
    panels = build_panels(df)

    out = []
    for pname, cos in panels.items():
        src = ["kaggle_arshkon"] if pname.startswith("arshkon") else ["kaggle_arshkon", "kaggle_asaniczka"]
        for tag, col in [("ai_strict_top", "ai_top"), ("ai_strict_v1_rebuilt", "ai_v1")]:
            r = within_between(df, cos, src, col)
            r["panel"] = pname
            r["pattern"] = tag
            out.append(r)

    df_out = pd.DataFrame(out)[
        ["panel", "pattern", "n_cos_overlap", "agg_24_pct", "agg_26_pct",
         "total_delta_pp", "within_delta_pp", "between_delta_pp"]
    ]
    print("\nH_w3_1 within-between decomposition on AI-strict (BOTH patterns reported):")
    print(df_out.to_string(index=False))
    df_out.to_csv(OUT_DIR / "H_w3_1_within_between_ai.csv", index=False)


if __name__ == "__main__":
    main()
