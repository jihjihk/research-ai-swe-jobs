"""T09 Step 14: Topics over time — archetype share per period.

Uses the BERTopic model's c-TF-IDF topic labels combined with our period
stratification. We already computed archetype share per period in step 8;
here we also plot it.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "exploration/artifacts/T09"
TABLES = "exploration/tables/T09"
FIGS = "exploration/figures/T09"


def main():
    share = pd.read_csv(f"{TABLES}/archetype_period_share.csv", index_col=0)
    # Drop unused summary column for plotting
    shr = share[["2024-01", "2024-04", "2026"]].copy()

    # Plot: grouped bar chart of top 10 archetypes by size
    top_arch = shr.sum(axis=1).sort_values(ascending=False).head(12).index
    shr_top = shr.loc[top_arch]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(shr_top))
    width = 0.27
    ax.bar(x - width, shr_top["2024-01"], width, label="2024-01 (asaniczka)", color="#4a90e2")
    ax.bar(x, shr_top["2024-04"], width, label="2024-04 (arshkon)", color="#f5a623")
    ax.bar(x + width, shr_top["2026"], width, label="2026 (scraped)", color="#d0021b")
    ax.set_xticks(x)
    ax.set_xticklabels(shr_top.index, rotation=45, ha="right")
    ax.set_ylabel("Share of period sample (%)")
    ax.set_title("Archetype share of SWE postings by period (T09 sample, n=8000)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGS}/archetype_share_by_period.png", dpi=150)
    plt.close(fig)
    print("Wrote archetype_share_by_period.png")

    # Delta plot: biggest growers and shrinkers
    delta = (shr["2026"] - 0.5 * (shr["2024-01"] + shr["2024-04"])).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#d0021b" if v > 0 else "#4a90e2" for v in delta.values]
    ax.barh(range(len(delta)), delta.values, color=colors)
    ax.set_yticks(range(len(delta)))
    ax.set_yticklabels(delta.index)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Δ share 2024→2026 (pp)")
    ax.set_title("Archetype share change 2024 avg → 2026")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/archetype_delta.png", dpi=150)
    plt.close(fig)
    print("Wrote archetype_delta.png")

    # J2 entry share trajectory per archetype
    entry = pd.read_csv(f"{TABLES}/archetype_entry_j2_by_period.csv", index_col=0)
    entry_shares = entry[["2024-01", "2024-04", "2026"]].copy()
    top_arch = shr.sum(axis=1).sort_values(ascending=False).head(10).index
    fig, ax = plt.subplots(figsize=(11, 6))
    for arch in top_arch:
        if arch in entry_shares.index:
            y = entry_shares.loc[arch].values
            ax.plot(["2024-01", "2024-04", "2026"], y, marker="o", label=arch)
    ax.set_ylabel("J2 (entry+associate) share of known seniority")
    ax.set_title("Entry share (J2) by archetype × period")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/entry_share_by_archetype_period.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote entry_share_by_archetype_period.png")

    # AI mention by archetype x period
    ai = pd.read_csv(f"{TABLES}/archetype_ai_share_by_period.csv", index_col=0)
    ai_shares = ai[["2024-01", "2024-04", "2026"]].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    for arch in top_arch:
        if arch in ai_shares.index:
            y = ai_shares.loc[arch].values
            ax.plot(["2024-01", "2024-04", "2026"], y, marker="o", label=arch)
    ax.set_ylabel("AI mention binary share")
    ax.set_title("AI mention share by archetype × period")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/ai_mention_by_archetype_period.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote ai_mention_by_archetype_period.png")


if __name__ == "__main__":
    main()
