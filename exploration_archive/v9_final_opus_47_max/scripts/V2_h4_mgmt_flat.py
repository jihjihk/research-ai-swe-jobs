"""V2 Phase A — H_w3_4: Independent re-derivation of T21 management flatness.

Claim (T21): under mgmt_strict_v1_rebuilt, senior mid-senior mgmt density is FLAT:
  0.039 (2024) → 0.038 (2026), SNR 0.1, below noise.
  The T11 "management density fell" under the 0.28-precision broad pattern does NOT reproduce.

V2 independent approach: apply V1-rebuilt mgmt pattern on senior cohort, compute density per 1K chars
for 2024 and 2026.
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
MGMT_REBUILT = p["v1_rebuilt_patterns"]["mgmt_strict_v1_rebuilt"]["pattern"]
MGMT_T11 = p["mgmt_strict"]["pattern"]  # T11 original


def fetch():
    q = f"""
    SELECT t.uid, t.source, t.period, t.seniority_final, t.seniority_3level,
           t.is_aggregator, t.company_name_canonical,
           LOWER(t.description_cleaned) AS txt, length(t.description_cleaned) AS clen
    FROM read_parquet('{CLEAN}') t
    """
    con = duckdb.connect()
    df = con.execute(q).df()
    con.close()
    df["era"] = np.where(df["source"] == "scraped", "2026", "2024")
    return df


def apply_mgmt(df):
    rx1 = re.compile(MGMT_REBUILT, flags=re.IGNORECASE)
    rx2 = re.compile(MGMT_T11, flags=re.IGNORECASE)
    tx = df["txt"].fillna("").to_numpy()
    df["mgmt_v1_hits"] = np.fromiter((len(rx1.findall(t)) for t in tx), dtype=np.int32, count=len(tx))
    df["mgmt_t11_hits"] = np.fromiter((len(rx2.findall(t)) for t in tx), dtype=np.int32, count=len(tx))
    return df


def density(df, col):
    # density per 1K chars
    return df.groupby(["era", "seniority_final"]).apply(
        lambda g: (g[col].sum() / max(g["clen"].sum(), 1)) * 1000
    )


def main():
    df = apply_mgmt(fetch())

    # Senior cohort
    sr = df[df["seniority_final"].isin(["mid-senior", "director"])].copy()

    # Density per 1K chars (T21 reports this as e.g. 0.039/1K)
    for col_out, col_hits in [("v1_rebuilt_density", "mgmt_v1_hits"), ("t11_density", "mgmt_t11_hits")]:
        den = sr.groupby(["era", "seniority_final"]).apply(
            lambda g: pd.Series({
                "n": len(g),
                f"{col_out}_per_1k": (g[col_hits].sum() / max(g["clen"].sum(), 1)) * 1000,
                "mean_hits": g[col_hits].mean(),
            })
        )
        print(f"\n{col_out} — T21 claims mgmt_rebuilt mid-senior 0.039 → 0.038 / director 0.031 → 0.026")
        print(den.to_string())

    # Within-2024 SNR check: compare arshkon vs asaniczka in 2024
    s24 = sr[sr["era"] == "2024"].copy()
    s24_arsh = s24[s24["source"] == "kaggle_arshkon"]
    s24_asan = s24[s24["source"] == "kaggle_asaniczka"]
    for src_df, src_name in [(s24_arsh, "arshkon_2024"), (s24_asan, "asaniczka_2024")]:
        for col_out, col_hits in [("v1_rebuilt", "mgmt_v1_hits"), ("t11", "mgmt_t11_hits")]:
            dens = (src_df[col_hits].sum() / max(src_df["clen"].sum(), 1)) * 1000
            print(f"  {src_name} {col_out} density = {dens:.4f}")

    # Save
    out = sr.groupby(["era", "seniority_final"]).apply(
        lambda g: pd.Series({
            "n": len(g),
            "v1_rebuilt_per_1k": (g["mgmt_v1_hits"].sum() / max(g["clen"].sum(), 1)) * 1000,
            "t11_per_1k": (g["mgmt_t11_hits"].sum() / max(g["clen"].sum(), 1)) * 1000,
        })
    )
    out.to_csv(OUT / "H_w3_4_mgmt_density.csv")
    print("\nSaved", OUT / "H_w3_4_mgmt_density.csv")


if __name__ == "__main__":
    main()
