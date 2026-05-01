"""T07 Part B: Correlate our state-level SWE counts with BLS OEWS 2024 employment.

Computes Pearson and Spearman r per source (arshkon, asaniczka, scraped, pooled_2024)
vs BLS OEWS 2024 state employment. Produces scatter plot.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
OUT_TABLES = ROOT / "exploration" / "tables" / "T07"
OUT_FIGS = ROOT / "exploration" / "figures" / "T07"
OUT_FIGS.mkdir(parents=True, exist_ok=True)
BENCH = ROOT / "exploration" / "artifacts" / "T07_benchmarks"


def load_bls() -> pd.DataFrame:
    with open(BENCH / "oes_state_2024.json") as f:
        d = json.load(f)
    rows = []
    for st, emp in d["swe_15_1252"].items():
        rows.append({"state": st, "bls_swe_2024": emp})
    return pd.DataFrame(rows)


def load_our_state_counts() -> pd.DataFrame:
    """SWE counts per state by source."""
    con = duckdb.connect()
    q = f"""
    SELECT state_normalized AS state, source, COUNT(*) AS n
    FROM read_parquet('{DATA}')
    WHERE source_platform = 'linkedin'
      AND is_english = true
      AND date_flag = 'ok'
      AND is_swe = true
      AND state_normalized IS NOT NULL
      AND is_multi_location = false
    GROUP BY state_normalized, source
    """
    df = con.execute(q).df()
    # pivot
    pivot = df.pivot_table(index="state", columns="source", values="n",
                           aggfunc="sum", fill_value=0).reset_index()
    pivot["pooled_2024"] = pivot.get("kaggle_arshkon", 0) + pivot.get("kaggle_asaniczka", 0)
    return pivot


def main():
    bls = load_bls()
    ours = load_our_state_counts()

    # Merge on state. BLS uses 2-letter abbreviations; our data uses the same (verify)
    print("Our state_normalized sample:", ours["state"].head().tolist())
    print("BLS sample:", bls["state"].head().tolist())

    merged = bls.merge(ours, on="state", how="inner")
    print(f"\nMerged states: {len(merged)}")
    print(merged.head().to_string())

    # Compute Pearson / Spearman per source
    sources = [c for c in merged.columns if c not in ("state", "bls_swe_2024")]
    rows = []
    for src in sources:
        # drop rows where BLS is null or our count is 0 (states not represented)
        sub = merged.dropna(subset=["bls_swe_2024", src])
        # Require at least 10 states
        if len(sub) < 10:
            continue
        # Pearson on raw counts
        r_pear, p_pear = stats.pearsonr(sub["bls_swe_2024"], sub[src])
        # Spearman (rank-based; robust to scale differences)
        r_spear, p_spear = stats.spearmanr(sub["bls_swe_2024"], sub[src])
        # Log-Pearson (often more appropriate for employment counts)
        log_bls = np.log1p(sub["bls_swe_2024"])
        log_our = np.log1p(sub[src])
        r_logpear, p_logpear = stats.pearsonr(log_bls, log_our)
        rows.append({
            "source": src,
            "n_states": len(sub),
            "pearson_r": round(r_pear, 4),
            "pearson_p": round(p_pear, 6),
            "spearman_r": round(r_spear, 4),
            "spearman_p": round(p_spear, 6),
            "log_pearson_r": round(r_logpear, 4),
            "log_pearson_p": round(p_logpear, 6),
            "verdict": "target_met" if r_pear > 0.80 else "below_target",
        })
    corr_df = pd.DataFrame(rows)
    print("\n=== Geographic correlation (state-level SWE vs BLS OEWS 2024) ===")
    print(corr_df.to_string(index=False))
    corr_df.to_csv(OUT_TABLES / "bls_state_correlation.csv", index=False)

    # Scatter plots
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, src in zip(axes, ["kaggle_arshkon", "kaggle_asaniczka", "pooled_2024", "scraped"]):
        sub = merged.dropna(subset=["bls_swe_2024", src])
        ax.scatter(sub["bls_swe_2024"], sub[src], s=30, alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        # annotate top-5
        for _, row in sub.nlargest(5, "bls_swe_2024").iterrows():
            ax.annotate(row["state"], (row["bls_swe_2024"], row[src]),
                        fontsize=8, xytext=(3, 3), textcoords="offset points")
        r_pear, _ = stats.pearsonr(np.log1p(sub["bls_swe_2024"]), np.log1p(sub[src]))
        ax.set_title(f"{src}\nlog-Pearson r = {r_pear:.3f} (n={len(sub)})")
        ax.set_xlabel("BLS OEWS 2024 SWE employment (log)")
        ax.set_ylabel("Our postings count (log)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("State-level SWE count: our data vs BLS OEWS 2024 (SOC 15-1252)")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "bls_state_correlation.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_FIGS / 'bls_state_correlation.png'}")

    # Also save the merged per-state data for downstream use
    merged.to_csv(OUT_TABLES / "bls_state_merged.csv", index=False)


if __name__ == "__main__":
    main()
