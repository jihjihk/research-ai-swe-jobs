"""T20 — Figure generation for the seniority boundary report."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path("/home/jihgaboot/gabor/job-research")
TABL = ROOT / "exploration" / "tables" / "T20"
FIG = ROOT / "exploration" / "figures" / "T20"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10})


def fig_auc_by_boundary():
    df = pd.read_csv(TABL / "auc_by_boundary_period.csv")
    df["period"] = df["period"].astype(str)
    # Strip junior↔senior 3level from main plot
    main = df[df["boundary"] != "junior↔senior (3level)"].copy()
    piv = main.pivot(index="boundary", columns="period", values="auc_mean")
    piv = piv.loc[["entry↔associate", "associate↔mid-senior", "mid-senior↔director"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(piv))
    w = 0.38
    ax.bar(x - w / 2, piv["2024"], width=w, label="2024", color="#6c9cd2")
    ax.bar(x + w / 2, piv["2026"], width=w, label="2026", color="#d26c6c")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=10)
    ax.set_ylabel("5-fold CV AUC")
    ax.set_title("T20 — Supervised boundary discriminability (logistic regression AUC)")
    ax.set_ylim(0.45, 1.0)
    for i, b in enumerate(piv.index):
        d_24 = piv.loc[b, "2024"]
        d_26 = piv.loc[b, "2026"]
        delta = d_26 - d_24
        arrow = "↑" if delta > 0 else "↓"
        ax.text(i, max(d_24, d_26) + 0.03, f"Δ {delta:+.3f} {arrow}", ha="center", fontsize=9)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(FIG / "auc_by_boundary.png")
    plt.close(fig)


def fig_yoe_auc_drop():
    df1 = pd.read_csv(TABL / "auc_by_boundary_period.csv")
    df2 = pd.read_csv(TABL / "auc_sensitivity_no_yoe.csv")
    df1["period"] = df1["period"].astype(str)
    df2["period"] = df2["period"].astype(str)
    # Merge
    m = df1[["period", "boundary", "auc_mean"]].merge(
        df2[["period", "boundary", "auc_mean"]], on=["period", "boundary"], suffixes=("_full", "_noyoe")
    )
    m["auc_drop"] = m["auc_mean_full"] - m["auc_mean_noyoe"]
    m = m[m["boundary"] != "junior↔senior (3level)"]
    piv = m.pivot(index="boundary", columns="period", values="auc_drop")
    piv = piv.loc[["entry↔associate", "associate↔mid-senior", "mid-senior↔director"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(piv))
    w = 0.38
    ax.bar(x - w / 2, piv["2024"], width=w, label="2024", color="#6c9cd2")
    ax.bar(x + w / 2, piv["2026"], width=w, label="2026", color="#d26c6c")
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=10)
    ax.set_ylabel("AUC drop when YOE removed")
    ax.set_title("T20 — YOE contribution to boundary AUC (full − no_YOE)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(FIG / "yoe_contribution_to_auc.png")
    plt.close(fig)


def fig_feature_heatmap():
    df = pd.read_csv(TABL / "feature_heatmap_long.csv")
    df["period"] = df["period"].astype(str)
    # Z-score within each feature across all 8 groups (2 periods × 4 levels)
    df["group"] = df["seniority_final"] + "_" + df["period"].astype(str)
    piv = df.pivot_table(index="feature", columns="group", values="mean")
    # reorder columns: entry_2024, associate_2024, mid-senior_2024, director_2024, entry_2026, ...
    order = [
        "entry_2024",
        "associate_2024",
        "mid-senior_2024",
        "director_2024",
        "entry_2026",
        "associate_2026",
        "mid-senior_2026",
        "director_2026",
    ]
    piv = piv[[c for c in order if c in piv.columns]]
    # Feature row order
    feat_order = [
        "yoe_min_years_llm",
        "requirement_breadth",
        "tech_count",
        "credential_stack_depth",
        "description_cleaned_length",
        "ai_binary",
        "scope_density",
        "mgmt_strong_density",
        "education_level",
    ]
    piv = piv.loc[[f for f in feat_order if f in piv.index]]
    # z-normalize each row (feature) for heatmap
    z = piv.sub(piv.mean(axis=1), axis=0).div(piv.std(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        z, annot=piv.round(2), fmt="", cmap="RdBu_r", center=0, cbar_kws={"label": "z-score within feature"}, ax=ax
    )
    ax.set_title("T20 — Feature profile by seniority × period (annotation = raw mean; color = row z-score)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(FIG / "feature_heatmap.png")
    plt.close(fig)


def fig_centroid_similarity():
    df = pd.read_csv(TABL / "centroid_similarity_matrix.csv")
    # groups are stringified
    order = [
        "entry_2024",
        "associate_2024",
        "mid-senior_2024",
        "director_2024",
        "entry_2026",
        "associate_2026",
        "mid-senior_2026",
        "director_2026",
    ]
    piv = df.pivot_table(index="group_i", columns="group_j", values="cosine")
    piv = piv.reindex(index=order, columns=order)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(piv, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", cbar_kws={"label": "cosine"}, ax=ax)
    ax.set_title("T20 — Cosine similarity on z-standardized structured features")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(FIG / "centroid_similarity_matrix.png")
    plt.close(fig)


def fig_gap_evolution():
    df = pd.read_csv(TABL / "gap_evolution.csv")
    sub = df[df["panel_pair"] == "J3 vs S4"].copy()
    feats = [
        "yoe_min_years_llm",
        "requirement_breadth",
        "tech_count",
        "credential_stack_depth",
        "ai_binary",
        "scope_density",
        "mgmt_strong_density",
        "description_cleaned_length",
    ]
    sub = sub[sub["feature"].isin(feats)].set_index("feature").loc[feats].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Left: ΔM_senior vs ΔM_junior
    x = np.arange(len(sub))
    w = 0.38
    axes[0].bar(x - w / 2, sub["delta_M_junior"], width=w, label="ΔM junior (J3 YOE≤2)", color="#6c9cd2")
    axes[0].bar(x + w / 2, sub["delta_M_senior"], width=w, label="ΔM senior (S4 YOE≥5)", color="#d26c6c")
    axes[0].axhline(0, color="gray", linestyle="-", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sub["feature"], rotation=30, ha="right")
    axes[0].set_ylabel("Δ mean (2026 − 2024)")
    axes[0].set_title("Same-level change by feature")
    axes[0].legend()

    # Right: attribution_senior bar
    colors = ["#d26c6c" if v > 0.55 else ("#6c9cd2" if v < 0.45 else "gray") for v in sub["attribution_senior"]]
    axes[1].bar(x, sub["attribution_senior"], color=colors)
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sub["feature"], rotation=30, ha="right")
    axes[1].set_ylabel("attribution_senior = |ΔS| / (|ΔS|+|ΔJ|)")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Share of gap movement driven by senior side")
    plt.tight_layout()
    fig.savefig(FIG / "gap_evolution.png")
    plt.close(fig)


def fig_yoe_interaction():
    df = pd.read_csv(TABL / "yoe_period_interaction.csv")
    df = df.sort_values("yoe_x_period_coef").reset_index(drop=True)
    outcomes = df["outcome"].tolist()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(df))
    ax.errorbar(
        df["yoe_x_period_coef"],
        y,
        xerr=[df["yoe_x_period_coef"] - df["yoe_x_period_ci_low"], df["yoe_x_period_ci_high"] - df["yoe_x_period_coef"]],
        fmt="o",
        color="#444",
        capsize=4,
    )
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["outcome"])
    ax.set_xlabel("YOE × Period (2026) coefficient (95% CI)")
    ax.set_title("T20 — Continuous YOE × period interaction (positive = YOE sorts more strongly in 2026)")
    plt.tight_layout()
    fig.savefig(FIG / "yoe_period_interaction.png")
    plt.close(fig)


def fig_missing_middle():
    df = pd.read_csv(TABL / "missing_middle.csv")
    df["period"] = df["period"].astype(str)
    piv = df.pivot_table(index="to", columns="period", values="euclidean_distance_z")
    piv = piv.loc[[c for c in ["entry", "mid-senior", "director"] if c in piv.index]]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(piv))
    w = 0.38
    ax.bar(x - w / 2, piv["2024"], width=w, label="2024", color="#6c9cd2")
    ax.bar(x + w / 2, piv["2026"], width=w, label="2026", color="#d26c6c")
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index)
    ax.set_ylabel("Euclidean distance (z-standardized features)")
    ax.set_title("T20 — Associate centroid distance to neighbor centroids")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG / "missing_middle.png")
    plt.close(fig)


def main() -> None:
    fig_auc_by_boundary()
    fig_yoe_auc_drop()
    fig_feature_heatmap()
    fig_centroid_similarity()
    fig_gap_evolution()
    fig_yoe_interaction()
    fig_missing_middle()
    print("[T20 figures] all saved to", FIG)


if __name__ == "__main__":
    main()
