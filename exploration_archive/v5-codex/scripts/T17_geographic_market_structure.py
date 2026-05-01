#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from T16_T17_common import (
    TABLE_DIR,
    FIG_DIR,
    add_row_metrics,
    company_cap,
    company_period_summary,
    ensure_dirs,
    load_base_frame,
    save_csv,
    save_fig,
    validate_regexes,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_TABLE = TABLE_DIR / "T17"
OUT_FIG = FIG_DIR / "T17"

SELECTED_METROS_MIN = 50
COMPANY_CAP = 25


def metro_counts(frame: pd.DataFrame) -> pd.DataFrame:
    counts = (
        frame.dropna(subset=["metro_area"])
        .groupby(["metro_area", "year"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"2024": "n_2024", "2026": "n_2026"})
    )
    if "n_2024" not in counts:
        counts["n_2024"] = 0
    if "n_2026" not in counts:
        counts["n_2026"] = 0
    return counts


def select_metros(counts: pd.DataFrame, min_n: int = SELECTED_METROS_MIN) -> pd.Index:
    selected = counts[(counts["n_2024"] >= min_n) & (counts["n_2026"] >= min_n)]["metro_area"]
    return pd.Index(selected.tolist())


def metro_summary(frame: pd.DataFrame, metros: pd.Index) -> pd.DataFrame:
    sub = frame[frame["metro_area"].isin(metros)].copy()
    summary = company_period_summary(sub, ["metro_area", "year"])
    summary["metro_area"] = summary["metro_area"].fillna("unknown")
    return summary


def metro_change(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "n_posts",
        "entry_final_share",
        "entry_yoe_share",
        "ai_any_share",
        "scope_any_share",
        "clean_len_mean",
        "tech_count_mean",
        "requirement_breadth_mean",
        "stack_depth_mean",
        "ai_domain_share",
        "domain_entropy",
        "llm_text_share",
        "remote_share",
    ]
    wide = summary.pivot(index="metro_area", columns="year", values=metrics)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for metric in metrics:
        if f"{metric}_2024" in wide and f"{metric}_2026" in wide:
            wide[f"delta_{metric}"] = wide[f"{metric}_2026"] - wide[f"{metric}_2024"]
    return wide


def rank_metros(changes: pd.DataFrame, metric: str, n: int = 8) -> pd.DataFrame:
    col = f"delta_{metric}"
    out = changes[["metro_area", col]].copy().sort_values(col, ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    return out.head(n)


def plot_change_heatmap(changes: pd.DataFrame, metric_cols: list[str]) -> plt.Figure:
    plot_df = changes.set_index("metro_area")[[f"delta_{m}" for m in metric_cols]].copy()
    plot_df.columns = [c.replace("delta_", "").replace("_", " ") for c in plot_df.columns]
    plot_df = plot_df.sort_values(by=plot_df.columns[0], ascending=False)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.45 * len(plot_df))))
    sns.heatmap(plot_df, cmap="vlag", center=0, linewidths=0.2, linecolor="white", ax=ax)
    ax.set_title("Metro change profile: 2024 -> 2026")
    ax.set_xlabel("")
    ax.set_ylabel("")
    return fig


def plot_ai_entry_scatter(changes: pd.DataFrame) -> plt.Figure:
    plot_df = changes.copy()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.regplot(
        data=plot_df,
        x="delta_ai_any_share",
        y="delta_entry_final_share",
        scatter=False,
        ax=ax,
        color="0.35",
        line_kws={"linewidth": 1},
    )
    sns.scatterplot(data=plot_df, x="delta_ai_any_share", y="delta_entry_final_share", s=70, ax=ax)
    for _, row in plot_df.iterrows():
        ax.text(row["delta_ai_any_share"], row["delta_entry_final_share"], row["metro_area"], fontsize=7, alpha=0.8)
    ax.axhline(0, color="0.75", linewidth=0.8)
    ax.axvline(0, color="0.75", linewidth=0.8)
    ax.set_title("Metros with larger AI surges do not systematically show larger entry declines")
    ax.set_xlabel("Delta AI-tool / LLM share")
    ax.set_ylabel("Delta explicit entry share")
    return fig


def plot_archetype_family_heatmap(family_change: pd.DataFrame) -> plt.Figure:
    pivot = family_change.pivot(index="metro_area", columns="domain_family", values="delta_share_pp").fillna(0)
    pivot = pivot.sort_values(by=pivot.columns[0], ascending=False)
    fig, ax = plt.subplots(figsize=(11, max(6, 0.45 * len(pivot))))
    sns.heatmap(pivot, cmap="vlag", center=0, linewidths=0.2, linecolor="white", ax=ax)
    ax.set_title("Metro domain-family composition shift")
    ax.set_xlabel("")
    ax.set_ylabel("")
    return fig


def plot_remote_share(remote: pd.DataFrame) -> plt.Figure:
    plot_df = remote.sort_values("remote_share", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x="remote_share", y="metro_area", color="#4c78a8", ax=ax)
    ax.set_title("Remote share by metro in scraped 2026")
    ax.set_xlabel("Remote share")
    ax.set_ylabel("")
    return fig


def metro_archetype_share(frame: pd.DataFrame, metros: pd.Index) -> pd.DataFrame:
    sub = frame[frame["metro_area"].isin(metros)].copy()
    rows = []
    for (metro, year), group in sub.groupby(["metro_area", "year"], dropna=False):
        counts = group["archetype_name"].value_counts(dropna=False)
        total = counts.sum()
        for name, count in counts.items():
            rows.append(
                {
                    "metro_area": metro,
                    "year": year,
                    "archetype_name": name,
                    "share": float(count / total) if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def family_share(frame: pd.DataFrame, metros: pd.Index) -> pd.DataFrame:
    sub = frame[frame["metro_area"].isin(metros)].copy()
    rows = []
    for (metro, year), group in sub.groupby(["metro_area", "year"], dropna=False):
        counts = group["domain_family"].value_counts(dropna=False)
        total = counts.sum()
        for name, count in counts.items():
            rows.append(
                {
                    "metro_area": metro,
                    "year": year,
                    "domain_family": name,
                    "share": float(count / total) if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def sensitivity_summary(primary: pd.DataFrame, alt: pd.DataFrame, key_metrics: list[str], label: str) -> pd.DataFrame:
    rows = []
    merged = primary.set_index("metro_area").join(alt.set_index("metro_area"), lsuffix="_primary", rsuffix="_alt", how="inner")
    for metric in key_metrics:
        p = merged[f"delta_{metric}_primary"]
        a = merged[f"delta_{metric}_alt"]
        rows.append(
            {
                "spec": label,
                "metric": metric,
                "mean_abs_diff": float((a - p).abs().mean()),
                "max_abs_diff": float((a - p).abs().max()),
                "direction_match_share": float(((p >= 0) == (a >= 0)).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    validate_regexes()
    OUT_TABLE.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    frame = load_base_frame()
    frame = add_row_metrics(frame)
    frame["year"] = frame["year"].astype(str)

    counts = metro_counts(frame)
    save_csv(counts, OUT_TABLE / "T17_metro_counts.csv")
    metros = select_metros(counts, SELECTED_METROS_MIN)
    save_csv(pd.DataFrame({"metro_area": metros}), OUT_TABLE / "T17_selected_metros.csv")
    multi_location_summary = pd.DataFrame(
        [
            {
                "excluded_multi_location_rows": int(frame["is_multi_location"].fillna(False).sum()),
                "excluded_no_metro_rows": int(frame["metro_area"].isna().sum()),
            }
        ]
    )
    save_csv(multi_location_summary, OUT_TABLE / "T17_multi_location_summary.csv")

    primary_summary = metro_summary(frame, metros)
    save_csv(primary_summary, OUT_TABLE / "T17_metro_summary_primary.csv")
    primary_change = metro_change(primary_summary)
    save_csv(primary_change, OUT_TABLE / "T17_metro_change_primary.csv")

    # Rankings by absolute change for the main metrics.
    ranking_frames = []
    for metric in [
        "entry_final_share",
        "entry_yoe_share",
        "ai_any_share",
        "scope_any_share",
        "clean_len_mean",
        "tech_count_mean",
        "ai_domain_share",
        "remote_share",
    ]:
        tmp = primary_change[["metro_area", f"delta_{metric}"]].copy()
        tmp["metric"] = metric
        tmp["abs_change"] = tmp[f"delta_{metric}"].abs()
        ranking_frames.append(tmp.sort_values("abs_change", ascending=False).head(8))
    rankings = pd.concat(ranking_frames, ignore_index=True)
    save_csv(rankings, OUT_TABLE / "T17_metro_change_rankings.csv")

    # Correlations between change metrics.
    corr_metrics = [
        "delta_entry_final_share",
        "delta_entry_yoe_share",
        "delta_ai_any_share",
        "delta_scope_any_share",
        "delta_clean_len_mean",
        "delta_tech_count_mean",
        "delta_ai_domain_share",
    ]
    corr = primary_change[["metro_area"] + corr_metrics].copy()
    corr_rows = []
    for a in corr_metrics:
        for b in corr_metrics:
            if a >= b:
                continue
            pear = float(corr[a].corr(corr[b], method="pearson"))
            spear = float(corr[a].corr(corr[b], method="spearman"))
            corr_rows.append({"metric_a": a, "metric_b": b, "pearson": pear, "spearman": spear})
    corr_df = pd.DataFrame(corr_rows)
    save_csv(corr_df, OUT_TABLE / "T17_metro_change_correlations.csv")

    # Remote share on scraped 2026 only.
    remote = (
        frame[(frame["year"] == "2026") & frame["metro_area"].isin(metros)]
        .groupby("metro_area", dropna=False)
        .agg(remote_share=("is_remote_inferred", lambda s: float(s.fillna(False).mean())), n=("uid", "size"))
        .reset_index()
    )
    save_csv(remote, OUT_TABLE / "T17_remote_share_2026.csv")
    remote_corr_rows = []
    remote_base = primary_summary[primary_summary["year"] == "2026"].groupby("metro_area", dropna=False).agg(
        entry_final_share=("entry_final_share", "mean"),
        ai_any_share=("ai_any_share", "mean"),
        scope_any_share=("scope_any_share", "mean"),
        clean_len_mean=("clean_len_mean", "mean"),
        tech_count_mean=("tech_count_mean", "mean"),
    ).reset_index()
    remote_join = remote.merge(remote_base, on="metro_area", how="inner")
    for metric in ["entry_final_share", "ai_any_share", "scope_any_share", "clean_len_mean", "tech_count_mean"]:
        remote_corr_rows.append(
            {
                "metric": metric,
                "pearson": float(remote_join["remote_share"].corr(remote_join[metric], method="pearson")),
                "spearman": float(remote_join["remote_share"].corr(remote_join[metric], method="spearman")),
            }
        )
    save_csv(pd.DataFrame(remote_corr_rows), OUT_TABLE / "T17_remote_correlations.csv")

    # Archetype and domain-family geography.
    archetype_share = metro_archetype_share(frame, metros)
    save_csv(archetype_share, OUT_TABLE / "T17_metro_archetype_shares.csv")
    family_shares = family_share(frame, metros)
    save_csv(family_shares, OUT_TABLE / "T17_metro_family_shares.csv")

    archetype_change = (
        archetype_share.pivot(index=["metro_area", "archetype_name"], columns="year", values="share")
        .fillna(0)
        .reset_index()
    )
    archetype_change["delta_share_pp"] = (archetype_change.get("2026", 0) - archetype_change.get("2024", 0)) * 100
    save_csv(archetype_change, OUT_TABLE / "T17_metro_archetype_changes.csv")

    family_change = (
        family_shares.pivot(index=["metro_area", "domain_family"], columns="year", values="share")
        .fillna(0)
        .reset_index()
    )
    family_change["delta_share_pp"] = (family_change.get("2026", 0) - family_change.get("2024", 0)) * 100
    save_csv(family_change, OUT_TABLE / "T17_metro_family_changes.csv")

    # Sensitivities: no aggregators and company cap.
    noagg = add_row_metrics(frame.loc[~frame["is_aggregator"]].copy())
    noagg["year"] = noagg["year"].astype(str)
    noagg_summary = metro_summary(noagg, metros)
    noagg_change = metro_change(noagg_summary)
    save_csv(noagg_summary, OUT_TABLE / "T17_metro_summary_noagg.csv")
    save_csv(noagg_change, OUT_TABLE / "T17_metro_change_noagg.csv")

    capped = company_cap(frame.copy(), COMPANY_CAP, ["year", "metro_area", "company_name_canonical"])
    capped = add_row_metrics(capped)
    capped["year"] = capped["year"].astype(str)
    capped_summary = metro_summary(capped, metros)
    capped_change = metro_change(capped_summary)
    save_csv(capped_summary, OUT_TABLE / "T17_metro_summary_cap25.csv")
    save_csv(capped_change, OUT_TABLE / "T17_metro_change_cap25.csv")

    sensitivity = pd.concat(
        [
            sensitivity_summary(primary_change, noagg_change, [
                "entry_final_share",
                "entry_yoe_share",
                "ai_any_share",
                "scope_any_share",
                "clean_len_mean",
                "tech_count_mean",
                "ai_domain_share",
            ], "no_aggregator"),
            sensitivity_summary(primary_change, capped_change, [
                "entry_final_share",
                "entry_yoe_share",
                "ai_any_share",
                "scope_any_share",
                "clean_len_mean",
                "tech_count_mean",
                "ai_domain_share",
            ], "company_cap_25"),
        ],
        ignore_index=True,
    )
    save_csv(sensitivity, OUT_TABLE / "T17_sensitivity_summary.csv")

    # Figures
    fig = plot_change_heatmap(primary_change, [
        "entry_final_share",
        "entry_yoe_share",
        "ai_any_share",
        "scope_any_share",
        "clean_len_mean",
        "tech_count_mean",
        "ai_domain_share",
    ])
    save_fig(fig, OUT_FIG / "T17_metro_change_heatmap.png")
    plt.close(fig)

    fig = plot_ai_entry_scatter(primary_change)
    save_fig(fig, OUT_FIG / "T17_ai_entry_scatter.png")
    plt.close(fig)

    fig = plot_archetype_family_heatmap(
        family_change[family_change["domain_family"].isin(["AI / LLM", "Frontend / Mobile", "Embedded", "Data / Platform", "Backend", "DevOps", "Requirements / Workflow"])]
    )
    save_fig(fig, OUT_FIG / "T17_domain_family_heatmap.png")
    plt.close(fig)

    fig = plot_remote_share(remote)
    save_fig(fig, OUT_FIG / "T17_remote_share.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
