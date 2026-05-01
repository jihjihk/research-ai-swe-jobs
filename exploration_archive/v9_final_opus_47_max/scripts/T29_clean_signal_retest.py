"""T29 addendum — re-test using CLEAN authorship signal only.

Calibration showed:
- raw_sig_vocab_density: within-2024 SNR ~4.5 (CLEAN)
- raw_em_dash_density: format-confounded (scraped markdown vs 2024 plain)
- avg_sentence_length: within-2024 noise swamps signal (SNR 0.09)
- sentence_length_sd: same issue
- type_token_ratio: SNR ~1.5 (near noise)

So we restrict the authorship score to signature vocabulary density only, and
re-run the low-LLM-subset headline re-test with this cleaner signal.

Usage: ./.venv/bin/python exploration/scripts/T29_clean_signal_retest.py
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = "/home/jihgaboot/gabor/job-research"
OUT = f"{ROOT}/exploration/tables/T29"
SHARED = f"{ROOT}/exploration/artifacts/shared"

# Load authorship scores
df = pd.read_csv(f"{OUT}/authorship_scores.csv")

# Define CLEAN score = sig_vocab density z-score only
df["z_sig_clean"] = (df["raw_sig_vocab_density"] - df["raw_sig_vocab_density"].mean()) / df["raw_sig_vocab_density"].std(ddof=0)
df["authorship_clean"] = df["z_sig_clean"]

# Redo quartile within period — use rank-based to avoid tie issues from zeros
def rank_quartile(x):
    r = x.rank(method="first") / len(x)
    labels = pd.Series(["Q1_low"] * len(x), index=x.index, dtype=object)
    labels[r > 0.25] = "Q2"
    labels[r > 0.5] = "Q3"
    labels[r > 0.75] = "Q4_high"
    return labels

df["clean_quartile"] = df.groupby("period_year")["authorship_clean"].transform(rank_quartile)
print(f"Quartile distribution: {df['clean_quartile'].value_counts().to_dict()}")

# Load T11 + T13
t11 = pq.read_table(f"{SHARED}/T11_posting_features.parquet").to_pandas()[
    ["uid", "tech_count", "requirement_breadth_resid", "credential_stack_depth",
     "scope_density", "ai_binary"]]
t13 = pq.read_table(f"{SHARED}/T13_readability_metrics.parquet").to_pandas()[
    ["uid", "sec_requirements_share", "raw_length"]]
df = df.merge(t11, on="uid", how="inner").merge(t13, on="uid", how="inner")
df["is_J3"] = (df["yoe_min_years_llm"] <= 2) & df["yoe_min_years_llm"].notna()
df["is_S4"] = (df["yoe_min_years_llm"] >= 5) & df["yoe_min_years_llm"].notna()

print("=" * 72)
print("Clean-signal (sig vocab only) re-test")
print("=" * 72)

rows = []
for subset_name, mask in [
    ("ALL", pd.Series(True, index=df.index)),
    ("LOW-CLEAN-Q1", df["clean_quartile"] == "Q1_low"),
    ("HIGH-CLEAN-Q4", df["clean_quartile"] == "Q4_high"),
]:
    sub = df[mask]
    d24 = sub[sub["period_year"] == 2024]
    d26 = sub[sub["period_year"] == 2026]

    rows.append({"subset": subset_name, "metric": "raw_length",
                 "val_2024": d24["raw_length"].mean(),
                 "val_2026": d26["raw_length"].mean(),
                 "delta": d26["raw_length"].mean() - d24["raw_length"].mean(),
                 "n_24": len(d24), "n_26": len(d26)})
    rows.append({"subset": subset_name, "metric": "ai_binary_rate",
                 "val_2024": d24["ai_binary"].astype(float).mean(),
                 "val_2026": d26["ai_binary"].astype(float).mean(),
                 "delta": d26["ai_binary"].astype(float).mean() - d24["ai_binary"].astype(float).mean(),
                 "n_24": len(d24), "n_26": len(d26)})
    rows.append({"subset": subset_name, "metric": "scope_density",
                 "val_2024": d24["scope_density"].mean(),
                 "val_2026": d26["scope_density"].mean(),
                 "delta": d26["scope_density"].mean() - d24["scope_density"].mean(),
                 "n_24": len(d24), "n_26": len(d26)})
    rows.append({"subset": subset_name, "metric": "requirements_share",
                 "val_2024": d24["sec_requirements_share"].mean(),
                 "val_2026": d26["sec_requirements_share"].mean(),
                 "delta": d26["sec_requirements_share"].mean() - d24["sec_requirements_share"].mean(),
                 "n_24": len(d24), "n_26": len(d26)})
    for tier, col in [("J3", "is_J3"), ("S4", "is_S4")]:
        t24 = d24[d24[col]]
        t26 = d26[d26[col]]
        if len(t24) < 20 or len(t26) < 20:
            continue
        r24 = (t24["credential_stack_depth"] >= 5).mean()
        r26 = (t26["credential_stack_depth"] >= 5).mean()
        rows.append({"subset": subset_name, "metric": f"credential_stack>=5_{tier}",
                     "val_2024": r24, "val_2026": r26, "delta": r26 - r24,
                     "n_24": len(t24), "n_26": len(t26)})
    rows.append({"subset": subset_name, "metric": "tech_count",
                 "val_2024": d24["tech_count"].mean(),
                 "val_2026": d26["tech_count"].mean(),
                 "delta": d26["tech_count"].mean() - d24["tech_count"].mean(),
                 "n_24": len(d24), "n_26": len(d26)})
    rows.append({"subset": subset_name, "metric": "requirement_breadth_resid",
                 "val_2024": d24["requirement_breadth_resid"].mean(),
                 "val_2026": d26["requirement_breadth_resid"].mean(),
                 "delta": d26["requirement_breadth_resid"].mean() - d24["requirement_breadth_resid"].mean(),
                 "n_24": len(d24), "n_26": len(d26)})

head = pd.DataFrame(rows)
head.to_csv(f"{OUT}/step7_clean_signal_retest.csv", index=False)
print(head.round(3).to_string(index=False))

# Shrinkage ratio
piv = head.pivot_table(index="metric", columns="subset", values="delta")
if "LOW-CLEAN-Q1" in piv.columns and "ALL" in piv.columns:
    piv["low_over_all"] = piv["LOW-CLEAN-Q1"] / piv["ALL"]
    if "HIGH-CLEAN-Q4" in piv.columns:
        piv["high_over_all"] = piv["HIGH-CLEAN-Q4"] / piv["ALL"]
    print("\n=== Shrinkage ratio (low_over_all; < 1 = weakens on low-LLM) ===")
    print(piv.round(3).to_string())
    piv.to_csv(f"{OUT}/step7_clean_signal_shrinkage.csv")

# Also show sig vocab density decomposition by company
print("\n=== Top 20 companies by sig vocab density 2026, n>=20 ===")
comp = df.groupby(["company_name_canonical", "period_year"]).agg(
    n=("uid", "count"),
    mean_sig=("raw_sig_vocab_density", "mean"),
).reset_index()
tops = comp[(comp["period_year"] == 2026) & (comp["n"] >= 20)].sort_values("mean_sig", ascending=False)
print(tops.head(20).to_string(index=False))
print("\n=== Bottom 20 companies by sig vocab density 2026, n>=20 ===")
print(tops.tail(20).to_string(index=False))
tops.to_csv(f"{OUT}/step7_top_company_sig_vocab_2026.csv", index=False)

# Company change in sig vocab: 2024 -> 2026, n>=5 both
comp_piv = comp.pivot(index="company_name_canonical", columns="period_year",
                        values=["mean_sig", "n"])
comp_piv.columns = [f"{a}_{b}" for a, b in comp_piv.columns]
comp_piv = comp_piv.dropna(subset=["mean_sig_2024", "mean_sig_2026"])
comp_piv = comp_piv[(comp_piv["n_2024"] >= 5) & (comp_piv["n_2026"] >= 5)]
comp_piv["sig_delta"] = comp_piv["mean_sig_2026"] - comp_piv["mean_sig_2024"]
print(f"\nCompanies with n>=5 in both periods: {len(comp_piv)}")
comp_piv = comp_piv.sort_values("sig_delta", ascending=False)
print("Top 10 biggest sig-vocab increase (hallmark of LLM-style drift):")
print(comp_piv.head(10).round(3).to_string())
print("Bottom 10:")
print(comp_piv.tail(10).round(3).to_string())
comp_piv.to_csv(f"{OUT}/step7_company_sig_vocab_change.csv")

print("\nDone.")
