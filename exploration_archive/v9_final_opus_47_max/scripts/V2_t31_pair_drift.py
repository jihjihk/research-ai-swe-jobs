"""V2 Phase A — H_T31: Independent re-derivation of pair-level AI drift.

Claim (T31): same-co × same-title AI-strict drift +13.4 pp EXCEEDS T16 company-level +7.7–8.3 pp
on the cleanest n≥3 panel (arshkon_min3 n=23 pairs; pooled_min5 n=33 pairs).

V2 independent approach: build pair panel, compute per-pair AI drift, aggregate mean Δ.
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
OUT = ROOT / "exploration" / "tables" / "V2"

with open(VALIDATED) as f:
    p = json.load(f)
AI_V1 = p["v1_rebuilt_patterns"]["ai_strict_v1_rebuilt"]["pattern"]
AI_TOP = p["ai_strict"]["pattern"]


def fetch():
    # Replicate T31: raw description, not cleaned
    q = f"""
    SELECT uid, company_name_canonical, source, period, is_aggregator,
           LOWER(title) AS title_lower,
           LOWER(description) AS txt
    FROM '{UNIFIED}'
    WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE
      AND company_name_canonical IS NOT NULL
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

    # Load panel definitions from T16 overlap panel
    panel_df = pd.read_csv("/home/jihgaboot/gabor/job-research/exploration/tables/T16/overlap_panel.csv")
    arshkon_min3_cos = set(panel_df[panel_df["panel_type"] == "arshkon_min3"]["company_name_canonical"])
    arshkon_min5_cos = set(panel_df[panel_df["panel_type"] == "arshkon_min5"]["company_name_canonical"])
    pooled_min5_cos = set(panel_df[panel_df["panel_type"] == "pooled_min5"]["company_name_canonical"])
    print(f"panel sizes: arshkon_min3={len(arshkon_min3_cos)}, arshkon_min5={len(arshkon_min5_cos)}, pooled_min5={len(pooled_min5_cos)}")

    for pname, cos, src_24 in [
        ("arshkon_min3", arshkon_min3_cos, ["kaggle_arshkon"]),  # Must be arshkon-only for arshkon panels
        ("arshkon_min5", arshkon_min5_cos, ["kaggle_arshkon"]),
        ("pooled_min5", pooled_min5_cos, ["kaggle_arshkon", "kaggle_asaniczka"]),
    ]:
        d = df[df["company_name_canonical"].isin(cos)].copy()
        # Restrict 2024 to only the relevant source for pair count
        d_for_count = d[~((d["era"] == "2024") & (~d["source"].isin(src_24)))]
        k = d_for_count.groupby(["company_name_canonical", "title_lower", "era"]).size().reset_index(name="n")
        g_24 = k[(k["era"] == "2024") & (k["n"] >= 3)]
        g_26 = k[(k["era"] == "2026") & (k["n"] >= 3)]
        pairs_set = set(zip(g_24["company_name_canonical"], g_24["title_lower"])) & \
                    set(zip(g_26["company_name_canonical"], g_26["title_lower"]))

        if not pairs_set:
            print(f"\n{pname}: 0 qualifying pairs at n>=3")
            continue

        # Compute per-pair AI drift
        pair_drifts = []
        for co, t in pairs_set:
            sub_24 = d[(d["company_name_canonical"] == co) & (d["title_lower"] == t) &
                       (d["era"] == "2024") & (d["source"].isin(src_24))]
            sub_26 = d[(d["company_name_canonical"] == co) & (d["title_lower"] == t) &
                       (d["era"] == "2026")]
            if len(sub_24) < 3 or len(sub_26) < 3:
                continue
            ai24 = sub_24["ai_v1"].mean()
            ai26 = sub_26["ai_v1"].mean()
            ai24_top = sub_24["ai_top"].mean()
            ai26_top = sub_26["ai_top"].mean()
            pair_drifts.append({
                "company_name_canonical": co,
                "title_lower": t,
                "n_2024": len(sub_24),
                "n_2026": len(sub_26),
                "ai_v1_2024": ai24,
                "ai_v1_2026": ai26,
                "ai_v1_delta": ai26 - ai24,
                "ai_top_2024": ai24_top,
                "ai_top_2026": ai26_top,
                "ai_top_delta": ai26_top - ai24_top,
            })
        pdf = pd.DataFrame(pair_drifts)
        if pdf.empty:
            print(f"\n{pname}: 0 pairs post-filter")
            continue
        print(f"\n{pname}: n_pairs={len(pdf)}")
        print(f"  Mean AI-v1-rebuilt pair drift: {pdf['ai_v1_delta'].mean()*100:.2f} pp")
        print(f"  Median AI-v1 pair drift: {pdf['ai_v1_delta'].median()*100:.2f} pp")
        print(f"  Mean AI-top pair drift: {pdf['ai_top_delta'].mean()*100:.2f} pp")

        pdf.to_csv(OUT / f"H_T31_{pname}_pair_drift.csv", index=False)


if __name__ == "__main__":
    main()
