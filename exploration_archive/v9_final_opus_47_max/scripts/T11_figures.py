"""T11 figures — requirements complexity & credential stacking."""

from __future__ import annotations

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "exploration" / "tables" / "T11"
FIG = ROOT / "exploration" / "figures" / "T11"
SHARED = ROOT / "exploration" / "artifacts" / "shared"
FIG.mkdir(parents=True, exist_ok=True)


def fig_distribution_hist():
    df = pq.read_table(SHARED / "T11_posting_features.parquet").to_pandas()
    df["bucket"] = df["source"].map(
        {"kaggle_arshkon": "arshkon_2024", "kaggle_asaniczka": "asaniczka_2024", "scraped": "scraped_2026"}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for lbl, sub in [
        ("pooled 2024", df[df["bucket"].isin(["arshkon_2024", "asaniczka_2024"])]),
        ("scraped 2026", df[df["bucket"] == "scraped_2026"]),
    ]:
        axes[0].hist(sub["requirement_breadth"], bins=40, alpha=0.55, label=lbl, density=True)
        axes[1].hist(sub["credential_stack_depth"], bins=np.arange(0, 9), alpha=0.55, label=lbl, density=True)
    axes[0].set_xlabel("requirement_breadth")
    axes[0].set_ylabel("density")
    axes[0].set_title("Requirement breadth distribution")
    axes[0].legend()
    axes[1].set_xlabel("credential_stack_depth (0-7)")
    axes[1].set_ylabel("density")
    axes[1].set_title("Credential stack depth distribution")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "distribution_hists.png", dpi=150)
    plt.close(fig)


def fig_stack_depth_panel():
    df = pd.read_csv(TAB / "credential_stack_depth_panel.csv")
    pivot = df[df["seniority_cut"].isin(["J3", "J4", "S4", "S5", "all"])].copy()
    pivot["label"] = pivot["period"] + "_" + pivot["seniority_cut"]
    fig, ax = plt.subplots(figsize=(11, 5))
    depths = list(range(1, 8))
    x = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(depths)))
    for i, d in enumerate(depths):
        col = f"share_depth_ge_{d}"
        vals = (pivot[col].values - (pivot[f"share_depth_ge_{d+1}"].values if d < 7 else np.zeros(len(pivot))))
        vals = np.clip(vals, 0, None)
        ax.bar(x, vals, bottom=bottom, color=colors[i], label=f"depth={d}")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(pivot["label"].tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Share of postings")
    ax.set_title("Credential stack depth distribution by period × seniority")
    ax.legend(ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.3), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "stack_depth_panel.png", dpi=150)
    plt.close(fig)


def fig_residualized_breadth():
    df = pd.read_csv(TAB / "distribution_panel.csv")
    df = df[df["seniority_cut"].isin(["all", "J3", "J4", "S4", "S5"])]
    df = df[df["period"].isin(["pooled_2024", "scraped_2026"])].copy()
    df["label"] = df["seniority_cut"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, ttl in [
        (axes[0], "requirement_breadth_resid_mean", "requirement_breadth (residualized)"),
        (axes[1], "credential_stack_depth_resid_mean", "credential_stack_depth (residualized)"),
    ]:
        pivot = df.pivot_table(index="label", columns="period", values=metric, aggfunc="mean")
        pivot = pivot.reindex(["all", "J3", "J4", "S4", "S5"])
        x = np.arange(len(pivot.index))
        w = 0.4
        ax.bar(x - w / 2, pivot["pooled_2024"].values, width=w, label="pooled 2024", color="steelblue")
        ax.bar(x + w / 2, pivot["scraped_2026"].values, width=w, label="scraped 2026", color="orange")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index.tolist())
        ax.set_ylabel(ttl)
        ax.set_title(ttl)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "residualized_breadth_by_seniority.png", dpi=150)
    plt.close(fig)


def fig_mgmt_terms():
    df = pd.read_csv(TAB / "mgmt_term_top10.csv")
    strong = df[df["tier"] == "strong"].copy()
    strong = strong[strong["period"].isin(["pooled_2024", "scraped_2026"])]
    broad = df[df["tier"] == "broad_extra"].copy()
    broad = broad[broad["period"].isin(["pooled_2024", "scraped_2026"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, d, ttl in [(axes[0], strong, "strong-tier mgmt terms"), (axes[1], broad, "broad-extra terms")]:
        pivot = d.pivot_table(index="term", columns="period", values="share_of_postings", aggfunc="sum").fillna(0)
        pivot = pivot.sort_values("scraped_2026", ascending=True)
        y = np.arange(len(pivot.index))
        h = 0.4
        ax.barh(y - h / 2, pivot.get("pooled_2024", pd.Series(0, index=pivot.index)).values, height=h, label="pooled 2024", color="steelblue")
        ax.barh(y + h / 2, pivot.get("scraped_2026", pd.Series(0, index=pivot.index)).values, height=h, label="scraped 2026", color="orange")
        ax.set_yticks(y)
        ax.set_yticklabels(pivot.index.tolist())
        ax.set_xlabel("Share of postings")
        ax.set_title(ttl)
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "mgmt_terms.png", dpi=150)
    plt.close(fig)


def fig_length_correlation():
    df = pd.read_csv(TAB / "length_correlation_check.csv")
    df = df[df["period"] != "all"]
    pivot = df.pivot_table(index="metric", columns="period", values="pearson_r", aggfunc="mean")
    pivot = pivot.loc[["tech_count", "soft_skill_count", "scope_count", "mgmt_broad_count", "requirement_breadth", "credential_stack_depth"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(pivot.index))
    w = 0.4
    ax.bar(x - w / 2, pivot["pooled_2024"].values, width=w, label="pooled 2024", color="steelblue")
    ax.bar(x + w / 2, pivot["scraped_2026"].values, width=w, label="scraped 2026", color="orange")
    ax.axhline(0.3, color="red", lw=0.8, linestyle="--", label="r=0.3 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Pearson r with description_cleaned_length")
    ax.set_title("Length-correlation check (residualize metrics with r > 0.3)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "length_correlation_check.png", dpi=150)
    plt.close(fig)


def fig_precision_validation():
    df = pd.read_csv(TAB / "mgmt_precision_summary.csv")
    df = df[df["period"].isin(["pooled_2024", "scraped_2026"])]
    # Color by tier
    fig, ax = plt.subplots(figsize=(11, 5))
    df = df.sort_values(["tier", "term", "period"])
    x = np.arange(len(df))
    colors = ["steelblue" if t == "strong" else "orange" for t in df["tier"]]
    ax.bar(x, df["precision"].values, color=colors)
    ax.axhline(0.80, color="red", lw=0.8, linestyle="--", label="0.80 precision threshold")
    labels = [f"{row.term}·{row.period.replace('_','-')}" for _, row in df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Heuristic semantic-context precision")
    ax.set_title("Management-pattern precision on 25-row stratified samples")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "precision_validation.png", dpi=150)
    plt.close(fig)


def main():
    fig_distribution_hist()
    fig_stack_depth_panel()
    fig_residualized_breadth()
    fig_mgmt_terms()
    fig_length_correlation()
    fig_precision_validation()
    print("figs written to", FIG)


if __name__ == "__main__":
    main()
