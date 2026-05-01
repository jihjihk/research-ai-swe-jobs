"""T21 — Figure generation."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa

ROOT = Path("/home/jihgaboot/gabor/job-research")
TABL = ROOT / "exploration" / "tables" / "T21"
FIG = ROOT / "exploration" / "figures" / "T21"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10})


def fig_density_bars():
    df = pd.read_csv(TABL / "density_summary_by_period_seniority.csv")
    df["period2"] = df["period2"].astype(str)
    # 2x2 grid: one subplot per density dimension
    feats = ["mgmt_rebuilt_density", "mgmt_t11_density", "orch_density", "strat_density"]
    titles = [
        "Management (V1 rebuilt, mentor + object)",
        "Management (T11 original, 0.55 precision)",
        "Technical orchestration",
        "Strategic language (note: strat 0.32 precision)",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    for i, (f, t) in enumerate(zip(feats, titles)):
        ax = axes[i]
        piv = df.pivot_table(index="seniority_final", columns="period2", values=f)
        x = np.arange(len(piv))
        w = 0.38
        ax.bar(x - w / 2, piv["2024"], width=w, color="#6c9cd2", label="2024")
        ax.bar(x + w / 2, piv["2026"], width=w, color="#d26c6c", label="2026")
        ax.set_xticks(x)
        ax.set_xticklabels(piv.index)
        ax.set_ylabel("Mentions per 1K chars")
        ax.set_title(t, fontsize=10)
        for ii, sen in enumerate(piv.index):
            delta = piv.loc[sen, "2026"] - piv.loc[sen, "2024"]
            ax.text(ii, max(piv.loc[sen, "2024"], piv.loc[sen, "2026"]) * 1.02, f"Δ{delta:+.3f}", ha="center", fontsize=8)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG / "density_by_period_seniority.png")
    plt.close(fig)


def fig_mgmt_orch_strat_scatter():
    df = pd.read_csv(TABL / "cluster_assignments.csv")
    df["period"] = df["period"].astype(str)
    # Sample 3k rows for legibility
    samp = df.sample(n=min(5000, len(df)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, period in zip(axes, ["2024", "2026"]):
        sub = samp[samp["period"] == period]
        sc = ax.scatter(
            sub["mgmt_density_v1_rebuilt"] + 1e-6,
            sub["orch_density"] + 1e-6,
            c=sub["strat_density"],
            alpha=0.4,
            s=10,
            cmap="viridis",
            vmin=0,
            vmax=3,
        )
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlabel("Management density (V1 rebuilt)")
        ax.set_ylabel("Orchestration density")
        ax.set_title(f"{period} — n={len(sub):,}")
        plt.colorbar(sc, ax=ax, label="Strategic density")
    fig.suptitle("T21 — Management × Orchestration × Strategic (sampled scatter)")
    plt.tight_layout()
    fig.savefig(FIG / "scatter_mgmt_orch_strat.png")
    plt.close(fig)


def fig_cluster_shares():
    df = pd.read_csv(TABL / "subcluster_by_period.csv")
    df["period2"] = df["period2"].astype(str)
    tot = df.groupby("period2")["n"].sum()
    df["share"] = df.apply(lambda r: r["n"] / tot[r["period2"]], axis=1)
    piv = df.pivot_table(index=["cluster_id", "cluster_name"], columns="period2", values="share")
    piv.columns = [str(c) for c in piv.columns]
    piv["delta_pp"] = (piv["2026"] - piv["2024"]) * 100
    piv = piv.reset_index().sort_values("delta_pp")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Left: Δ share bar chart
    colors = ["#6c9cd2" if d > 0 else "#d26c6c" for d in piv["delta_pp"]]
    axes[0].barh(piv["cluster_name"], piv["delta_pp"], color=colors)
    axes[0].axvline(0, color="black", linewidth=0.6)
    axes[0].set_xlabel("Share change 2024→2026 (pp)")
    axes[0].set_title("Senior cluster composition change")
    axes[0].invert_yaxis()

    # Right: stacked bar of shares per period
    piv2 = piv.set_index("cluster_name")[["2024", "2026"]]
    piv2.T.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab10", legend=True)
    axes[1].set_ylabel("Share of senior postings")
    axes[1].set_title("Senior cluster composition (stacked)")
    axes[1].legend(bbox_to_anchor=(1.0, 1.0), fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG / "cluster_shares.png")
    plt.close(fig)


def fig_cross_seniority_mgmt():
    df = pd.read_csv(TABL / "cross_seniority_mgmt.csv")
    df["period2"] = df["period2"].astype(str)
    # Focus on mgmt_rebuilt, orch, strat, ai_strict_bin
    feats = ["mgmt_rebuilt_density", "orch_density", "strat_density", "ai_strict_bin"]
    titles = ["Mgmt (V1 rebuilt)", "Orchestration", "Strategic", "AI-strict (binary)"]
    order = ["junior", "mid", "senior"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, (f, t) in enumerate(zip(feats, titles)):
        ax = axes[i]
        piv = df[df["seniority_3level"].isin(order)].pivot_table(
            index="seniority_3level", columns="period2", values=f
        )
        piv = piv.reindex(order)
        x = np.arange(len(order))
        w = 0.38
        ax.bar(x - w / 2, piv["2024"], width=w, color="#6c9cd2", label="2024")
        ax.bar(x + w / 2, piv["2026"], width=w, color="#d26c6c", label="2026")
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_title(t)
        ax.set_ylabel("density / rate")
        for ii, sen in enumerate(order):
            if sen in piv.index:
                d24 = piv.loc[sen, "2024"] if not pd.isna(piv.loc[sen, "2024"]) else 0
                d26 = piv.loc[sen, "2026"] if not pd.isna(piv.loc[sen, "2026"]) else 0
                ax.text(ii, max(d24, d26) * 1.02, f"Δ{d26-d24:+.3f}", ha="center", fontsize=7)
        ax.legend(fontsize=8)
    plt.suptitle("T21 — Cross-seniority management / orch / strat / AI comparison")
    plt.tight_layout()
    fig.savefig(FIG / "cross_seniority_mgmt.png")
    plt.close(fig)


def fig_ai_interaction():
    df = pd.read_csv(TABL / "senior_ai_interaction.csv")
    df["period2"] = df["period2"].astype(str)
    feats = ["mgmt_rebuilt_density", "orch_density", "strat_density"]
    titles = ["Mgmt density (V1 rebuilt)", "Orchestration density", "Strategic density"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, f, t in zip(axes, feats, titles):
        piv = df.pivot_table(index="period2", columns="ai_strict_bin", values=f)
        piv.columns = ["non-AI senior", "AI-mentioning senior"]
        piv.plot(kind="bar", ax=ax, color=["#888888", "#d26c6c"])
        ax.set_title(t)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
    plt.suptitle("T21 — Among senior postings, AI-mentioning vs non-AI")
    plt.tight_layout()
    fig.savefig(FIG / "ai_interaction_senior.png")
    plt.close(fig)


def fig_pattern_validation():
    df = pd.read_csv(TABL / "pattern_validation.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    w = 0.3
    ax.bar(x - w, df["precision_period_2024"], width=w, color="#6c9cd2", label="2024 precision")
    ax.bar(x, df["precision_period_2026"], width=w, color="#d26c6c", label="2026 precision")
    ax.bar(x + w, df["overall_precision"], width=w, color="#8877bb", label="Overall")
    ax.axhline(0.8, color="green", linestyle="--", label="Gate 2 threshold (0.80)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["pattern_name"], rotation=10)
    ax.set_ylabel("Measured precision")
    ax.set_title("T21 — Pattern validation (50-row sample, programmatic adjudication)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG / "pattern_validation.png")
    plt.close(fig)


def main() -> None:
    fig_density_bars()
    fig_mgmt_orch_strat_scatter()
    fig_cluster_shares()
    fig_cross_seniority_mgmt()
    fig_ai_interaction()
    fig_pattern_validation()
    print("[T21 figures] saved to", FIG)


if __name__ == "__main__":
    main()
