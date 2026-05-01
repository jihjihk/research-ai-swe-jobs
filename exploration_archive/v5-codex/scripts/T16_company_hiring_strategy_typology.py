#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from T16_T17_common import (
    TABLE_DIR,
    FIG_DIR,
    add_row_metrics,
    ai_cols_from_tech,
    company_cap,
    company_period_summary,
    ensure_dirs,
    family_count,
    load_base_frame,
    save_csv,
    save_fig,
    validate_regexes,
    SOFT_SKILL_TERMS,
    SCOPE_TERMS_STRICT,
    EDU_LEVELS,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_TABLE = TABLE_DIR / "T16"
OUT_FIG = FIG_DIR / "T16"


def sql_round(x: float, digits: int = 4) -> float:
    return float(round(float(x), digits))


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return add_row_metrics(df)


def overlap_company_panel(df: pd.DataFrame, min_n: int = 3, noagg: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    if noagg:
        work = work.loc[~work["is_aggregator"]].copy()
    counts = (
        work.groupby(["company_name_canonical", "year"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"2024": "n_2024", "2026": "n_2026"})
    )
    if "n_2024" not in counts:
        counts["n_2024"] = 0
    if "n_2026" not in counts:
        counts["n_2026"] = 0
    counts = counts[["company_name_canonical", "n_2024", "n_2026"]]
    panel = counts[(counts["n_2024"] >= min_n) & (counts["n_2026"] >= min_n)].copy()
    return panel, counts


def pooled_2024_frame() -> pd.DataFrame:
    """Load arshkon + asaniczka for 2024 and scraped for 2026 as a sensitivity baseline."""
    import duckdb

    tech_cols = None
    con = duckdb.connect()
    meta = con.execute(
        """
        SELECT
            uid, source, period, company_name_canonical, company_name_effective, company_industry,
            company_size, is_aggregator, is_remote_inferred, is_multi_location, metro_area,
            seniority_final, seniority_3level, yoe_extracted, description_length
        FROM read_parquet('data/unified.parquet') u
        WHERE u.source_platform = 'linkedin'
          AND u.is_english = true
          AND u.date_flag = 'ok'
          AND u.is_swe = true
          AND (
            (u.source IN ('kaggle_arshkon', 'kaggle_asaniczka') AND u.period IN ('2024-01', '2024-04'))
            OR (u.source = 'scraped' AND u.period IN ('2026-03', '2026-04'))
          )
        """
    ).fetchdf()
    from T14_T15_common import load_cleaned_text, load_tech_matrix, tech_columns

    tech_cols = tech_columns()
    text = load_cleaned_text(["uid", "description_cleaned", "text_source"])
    tech = load_tech_matrix(tech_cols)
    archetypes = con.execute(
        """
        SELECT uid, archetype_name
        FROM read_parquet('exploration/artifacts/shared/swe_archetype_labels.parquet')
        """
    ).fetchdf()
    frame = meta.merge(text, on="uid", how="left").merge(tech, on="uid", how="left").merge(archetypes, on="uid", how="left")
    frame["company_key"] = (
        frame["company_name_canonical"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "unknown_company")
    )
    frame["year"] = np.where(frame["period"].astype(str).str.startswith("2024"), "2024", "2026")
    from T16_T17_common import DOMAIN_FAMILY_MAP

    frame["domain_family"] = frame["archetype_name"].map(DOMAIN_FAMILY_MAP).fillna(frame["archetype_name"].fillna("Other"))
    return add_row_metrics(frame)


def company_change_table(company_period: pd.DataFrame) -> pd.DataFrame:
    pivots = []
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
    pivot = company_period.pivot(index="company_name_canonical", columns="year", values=metrics)
    pivot.columns = [f"{m}_{y}" for m, y in pivot.columns]
    pivot = pivot.reset_index()
    for metric in metrics:
        a = pivot.get(f"{metric}_2024")
        b = pivot.get(f"{metric}_2026")
        if a is None or b is None:
            continue
        pivot[f"delta_{metric}"] = b - a
    return pivot


def choose_k(X: np.ndarray, k_candidates: list[int]) -> tuple[int, pd.DataFrame]:
    rows = []
    best_k = k_candidates[0]
    best_score = -np.inf
    for k in k_candidates:
        if k >= len(X):
            continue
        labels = KMeans(n_clusters=k, random_state=42, n_init=30).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        rows.append({"k": k, "silhouette": float(score)})
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, pd.DataFrame(rows)


def cluster_typology(change_df: pd.DataFrame, metric_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = change_df.dropna(subset=metric_cols, how="all").copy()
    work = work.fillna(0.0)
    scaler = StandardScaler()
    X = scaler.fit_transform(work[metric_cols].to_numpy())
    k, selection = choose_k(X, [3, 4, 5, 6])
    model = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels = model.fit_predict(X)
    work["cluster_id"] = labels + 1
    work["cluster_label"] = ""
    centroids = pd.DataFrame(model.cluster_centers_, columns=metric_cols)
    centroids["cluster_id"] = np.arange(1, k + 1)
    centroids = centroids.merge(
        work.groupby("cluster_id")[metric_cols]
        .mean()
        .reset_index()
        .rename(columns={c: f"mean_{c}" for c in metric_cols}),
        on="cluster_id",
        how="left",
    )
    centroids["size"] = centroids["cluster_id"].map(work["cluster_id"].value_counts()).fillna(0).astype(int)

    def label_cluster(row: pd.Series) -> str:
        raw = {c.replace("delta_", ""): row[f"mean_{c}"] for c in metric_cols}
        if raw["ai_domain_share"] > 0.15 or raw["ai_any_share"] > 0.25:
            return "AI-forward recomposition"
        if raw["requirement_breadth_mean"] > 2.0 or raw["stack_depth_mean"] > 0.5:
            return "stack expansion"
        if raw["clean_len_mean"] > 700 and raw["tech_count_mean"] < 1.0:
            return "template inflation / text-heavy"
        if raw["entry_yoe_share"] > 0.02 and raw["tech_count_mean"] < 1.5:
            return "entry-heavy compact"
        if max(abs(v) for v in raw.values()) < 0.10:
            return "stable / low-change"
        top = sorted(raw.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return f"mixed ({top[0][0].replace('_share', '').replace('_mean', '')})"

    labels_by_cluster = centroids.apply(label_cluster, axis=1)
    centroids["cluster_name"] = labels_by_cluster.values
    work = work.merge(centroids[["cluster_id", "cluster_name"]], on="cluster_id", how="left")
    work["cluster_id"] = work["cluster_id"].astype(int)
    selection["chosen_k"] = k
    return work, centroids, selection


def plot_cluster_heatmap(centroids: pd.DataFrame, metric_cols: list[str]) -> plt.Figure:
    plot_df = centroids.set_index("cluster_name")[ [f"mean_{c}" for c in metric_cols] ].copy()
    plot_df.columns = [c.replace("mean_delta_", "").replace("delta_", "") for c in plot_df.columns]
    fig, ax = plt.subplots(figsize=(12, max(4, 0.55 * len(plot_df))))
    sns.heatmap(plot_df, cmap="vlag", center=0, annot=True, fmt=".2f", linewidths=0.2, linecolor="white", ax=ax)
    ax.set_title("Company strategy clusters: mean change profile")
    ax.set_xlabel("")
    ax.set_ylabel("")
    return fig


def plot_pca(change_df: pd.DataFrame, metric_cols: list[str]) -> plt.Figure:
    work = change_df.dropna(subset=metric_cols, how="all").copy().fillna(0.0)
    X = StandardScaler().fit_transform(work[metric_cols].to_numpy())
    pcs = PCA(n_components=2, random_state=42).fit_transform(X)
    work["pc1"] = pcs[:, 0]
    work["pc2"] = pcs[:, 1]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=work, x="pc1", y="pc2", hue="cluster_name", palette="tab10", s=45, alpha=0.85, ax=ax)
    ax.set_title("Company strategy typology in PCA space")
    ax.axhline(0, color="0.8", linewidth=0.8)
    ax.axvline(0, color="0.8", linewidth=0.8)
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    return fig


def plot_decomposition(decomp: pd.DataFrame) -> plt.Figure:
    plot_df = decomp.copy()
    plot_df["metric"] = plot_df["metric"].str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df["within_company"], label="Within-company", color="#4c78a8")
    ax.bar(x, plot_df["between_company"], bottom=plot_df["within_company"], label="Between-company", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["metric"], rotation=30, ha="right")
    ax.set_ylabel("Change")
    ax.set_title("Overlap-panel decomposition of 2024 -> 2026 change")
    ax.legend()
    return fig


def plot_domain_shift(domain_change: pd.DataFrame) -> plt.Figure:
    plot_df = domain_change.sort_values("delta_share_pp", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_df, x="delta_share_pp", y="domain_family", palette="vlag", ax=ax)
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set_xlabel("Share change (pp)")
    ax.set_ylabel("")
    ax.set_title("Domain-family composition shift on the overlap panel")
    return fig


def decomposition_table(company_period: pd.DataFrame, metric: str, company_col: str = "company_name_canonical") -> pd.Series:
    wide = company_period.pivot(index=company_col, columns="year", values=[metric, "n_posts"]).copy()
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    if f"{metric}_2024" not in wide or f"{metric}_2026" not in wide:
        raise KeyError(metric)
    wide = wide.fillna(0.0)
    w0 = wide["n_posts_2024"] / max(wide["n_posts_2024"].sum(), 1.0)
    w1 = wide["n_posts_2026"] / max(wide["n_posts_2026"].sum(), 1.0)
    m0 = wide[f"{metric}_2024"]
    m1 = wide[f"{metric}_2026"]
    total0 = float((w0 * m0).sum())
    total1 = float((w1 * m1).sum())
    within = float((w0 * (m1 - m0)).sum())
    between = float(((w1 - w0) * m0).sum())
    return pd.Series(
        {
            "metric": metric,
            "baseline_2024": total0,
            "value_2026": total1,
            "change": total1 - total0,
            "within_company": within,
            "between_company": between,
            "within_plus_between": within + between,
        }
    )


def domain_entry_decomposition(raw_frame: pd.DataFrame, metric_col: str = "entry_final", domain_family: str = "domain_family") -> pd.Series:
    cell = (
        raw_frame.groupby(["company_name_canonical", "year", domain_family], dropna=False)
        .agg(n_posts=("uid", "size"), metric=(metric_col, "mean"))
        .reset_index()
    )
    domain_totals = cell.groupby(["year", domain_family], dropna=False)["n_posts"].sum().reset_index(name="domain_posts")
    cell = cell.merge(domain_totals, on=["year", domain_family], how="left")
    cell["company_within_domain_weight"] = cell["n_posts"] / cell["domain_posts"].replace(0, np.nan)
    cell["year_weight"] = cell["n_posts"] / cell.groupby("year")["n_posts"].transform("sum")
    fams = sorted(cell[domain_family].dropna().unique().tolist())

    def share(year: str) -> dict[str, float]:
        sub = cell[cell["year"] == year].copy()
        out = {}
        for fam in fams:
            fam_sub = sub[sub[domain_family] == fam]
            out[fam] = float((fam_sub["company_within_domain_weight"].fillna(0) * fam_sub["metric"].fillna(0)).sum()) if not fam_sub.empty else 0.0
        return out

    base_domain_share = cell[cell["year"] == "2024"].groupby(domain_family)["domain_posts"].sum()
    new_domain_share = cell[cell["year"] == "2026"].groupby(domain_family)["domain_posts"].sum()
    base_total = float(base_domain_share.sum()) if len(base_domain_share) else 1.0
    new_total = float(new_domain_share.sum()) if len(new_domain_share) else 1.0
    d0 = {fam: float(base_domain_share.get(fam, 0.0) / base_total) for fam in fams}
    d1 = {fam: float(new_domain_share.get(fam, 0.0) / new_total) for fam in fams}
    m0 = share("2024")
    m1 = share("2026")

    within_company = 0.0
    between_company = 0.0
    between_domain = 0.0
    for fam in fams:
        fam0 = cell[(cell["year"] == "2024") & (cell[domain_family] == fam)].copy()
        fam1 = cell[(cell["year"] == "2026") & (cell[domain_family] == fam)].copy()
        w0 = fam0.set_index("company_name_canonical")["company_within_domain_weight"].fillna(0.0)
        w1 = fam1.set_index("company_name_canonical")["company_within_domain_weight"].fillna(0.0)
        m0_f = fam0.set_index("company_name_canonical")["metric"].to_dict()
        m1_f = fam1.set_index("company_name_canonical")["metric"].to_dict()
        company_keys = sorted(set(m0_f) | set(m1_f))
        base = sum(w0.get(k, 0.0) * m0_f.get(k, 0.0) for k in company_keys)
        after_within = sum(w0.get(k, 0.0) * m1_f.get(k, 0.0) for k in company_keys)
        after_company = sum(w1.get(k, 0.0) * m0_f.get(k, 0.0) for k in company_keys)
        within_company += d0[fam] * (after_within - base)
        between_company += d0[fam] * (after_company - after_within)
        between_domain += (d1[fam] - d0[fam]) * after_company

    total0 = sum(d0[fam] * m0[fam] for fam in fams)
    total1 = sum(d1[fam] * m1[fam] for fam in fams)
    return pd.Series(
        {
            "metric": metric_col,
            "baseline_2024": total0,
            "value_2026": total1,
            "change": total1 - total0,
            "within_company": within_company,
            "between_company_within_domain": between_company,
            "between_domain": between_domain,
            "within_plus_between": within_company + between_company + between_domain,
            "residual": (total1 - total0) - (within_company + between_company + between_domain),
        }
    )


def main() -> None:
    ensure_dirs()
    validate_regexes()
    OUT_TABLE.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    frame = load_base_frame()
    frame = add_metrics(frame)

    frame["year"] = frame["year"].astype(str)
    frame["period_bucket"] = frame["year"]

    company_panel3, counts = overlap_company_panel(frame, min_n=3, noagg=False)
    company_panel5, counts5 = overlap_company_panel(frame, min_n=5, noagg=False)
    company_panel3_noagg, counts_noagg = overlap_company_panel(frame, min_n=3, noagg=True)
    company_panel5_noagg, counts5_noagg = overlap_company_panel(frame, min_n=5, noagg=True)

    counts_out = counts.merge(counts5, on="company_name_canonical", how="outer", suffixes=("", "_ge5"))
    save_csv(counts_out, OUT_TABLE / "T16_company_overlap_counts.csv")
    panel_summary = pd.DataFrame(
        [
            {
                "spec": "primary",
                "min_n": 3,
                "companies": int(len(company_panel3)),
                "companies_ge5": int(len(company_panel5)),
            },
            {
                "spec": "no_aggregator",
                "min_n": 3,
                "companies": int(len(company_panel3_noagg)),
                "companies_ge5": int(len(company_panel5_noagg)),
            },
        ]
    )
    save_csv(panel_summary, OUT_TABLE / "T16_overlap_panel_summary.csv")

    company_period = company_period_summary(frame[frame["company_name_canonical"].isin(company_panel3["company_name_canonical"])], ["company_name_canonical", "year"])
    save_csv(company_period, OUT_TABLE / "T16_company_period_summary_primary.csv")

    company_change = company_change_table(company_period)
    primary_metrics = [
        "delta_entry_final_share",
        "delta_entry_yoe_share",
        "delta_ai_any_share",
        "delta_scope_any_share",
        "delta_clean_len_mean",
        "delta_tech_count_mean",
        "delta_requirement_breadth_mean",
        "delta_stack_depth_mean",
        "delta_ai_domain_share",
        "delta_domain_entropy",
    ]
    company_change = company_change.sort_values("company_name_canonical").reset_index(drop=True)
    save_csv(company_change, OUT_TABLE / "T16_company_change_profile_primary.csv")

    selected = company_change.dropna(subset=primary_metrics, how="all").copy()
    cluster_df, centroids, k_selection = cluster_typology(selected, primary_metrics)
    save_csv(cluster_df, OUT_TABLE / "T16_company_cluster_membership_primary.csv")
    save_csv(centroids, OUT_TABLE / "T16_company_cluster_centroids_primary.csv")
    save_csv(k_selection, OUT_TABLE / "T16_cluster_k_selection.csv")

    cluster_summary = cluster_df.groupby(["cluster_id", "cluster_name"], dropna=False)[primary_metrics].agg(["mean", "median", "count"]).reset_index()
    flat_cols = []
    for col in cluster_summary.columns:
        if isinstance(col, tuple):
            flat_cols.append("_".join([str(c) for c in col if c]))
        else:
            flat_cols.append(str(col))
    cluster_summary.columns = flat_cols
    save_csv(cluster_summary, OUT_TABLE / "T16_cluster_summary_primary.csv")

    # Cluster labels and top companies
    cluster_counts = cluster_df["cluster_name"].value_counts().reset_index()
    cluster_counts.columns = ["cluster_name", "n_companies"]
    save_csv(cluster_counts, OUT_TABLE / "T16_cluster_counts_primary.csv")
    cluster_tops = (
        cluster_df.assign(
            abs_ai=cluster_df["delta_ai_any_share"].abs(),
            abs_scope=cluster_df["delta_scope_any_share"].abs(),
            abs_entry=cluster_df["delta_entry_yoe_share"].abs(),
        )
        .sort_values(["cluster_name", "abs_ai", "abs_scope", "abs_entry"], ascending=[True, False, False, False])
        .groupby("cluster_name", dropna=False)
        .head(10)[
            [
                "company_name_canonical",
                "cluster_name",
                "delta_entry_final_share",
                "delta_entry_yoe_share",
                "delta_ai_any_share",
                "delta_scope_any_share",
                "delta_clean_len_mean",
                "delta_tech_count_mean",
                "delta_requirement_breadth_mean",
                "delta_stack_depth_mean",
                "delta_ai_domain_share",
                "delta_domain_entropy",
            ]
        ]
    )
    save_csv(cluster_tops, OUT_TABLE / "T16_cluster_top_companies_primary.csv")

    decomp_rows = []
    for metric in primary_metrics:
        # Percent-share metrics are kept in raw proportion units; report conversion in the memo.
        if metric not in company_period.columns and metric not in company_change.columns:
            continue
        source_metric = metric.replace("delta_", "").replace("_mean", "")
        base_metric = source_metric
        if base_metric not in company_period.columns:
            continue
    # Explicit metric decomposition on the company-period summary table.
    for metric in [
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
    ]:
        decomp_rows.append(decomposition_table(company_period, metric))
    decomp_primary = pd.DataFrame(decomp_rows)
    save_csv(decomp_primary, OUT_TABLE / "T16_decomposition_primary.csv")

    overlap_raw = frame[frame["company_name_canonical"].isin(company_panel3["company_name_canonical"])].copy()
    domain_decomp_rows = [
        domain_entry_decomposition(overlap_raw, metric_col="entry_final"),
        domain_entry_decomposition(overlap_raw, metric_col="entry_yoe"),
    ]
    domain_decomp = pd.DataFrame(domain_decomp_rows)
    save_csv(domain_decomp, OUT_TABLE / "T16_domain_decomposition_primary.csv")

    # Domain family shift table on the overlap panel.
    overlap_raw = frame[frame["company_name_canonical"].isin(company_panel3["company_name_canonical"])].copy()
    domain_family = (
        overlap_raw.groupby(["year", "domain_family"], dropna=False)
        .agg(n_posts=("uid", "size"), entry_final_share=("entry_final", "mean"), ai_any_share=("ai_any", "mean"))
        .reset_index()
    )
    domain_family["share"] = domain_family.groupby("year")["n_posts"].transform(lambda s: s / s.sum())
    domain_pivot = domain_family.pivot(index="domain_family", columns="year", values="share").fillna(0)
    domain_change = domain_pivot.copy()
    domain_change["delta_share_pp"] = (domain_pivot.get("2026", 0) - domain_pivot.get("2024", 0)) * 100
    domain_change = domain_change.reset_index().rename(columns={"2024": "share_2024", "2026": "share_2026"})
    save_csv(domain_change.sort_values("delta_share_pp", ascending=False), OUT_TABLE / "T16_domain_family_shift_primary.csv")

    new_company_names = counts.loc[(counts["n_2024"] == 0) & (counts["n_2026"] > 0), "company_name_canonical"]
    new_entrants = frame[(frame["year"] == "2026") & (frame["company_name_canonical"].isin(new_company_names))].copy()
    new_entrants = company_period_summary(new_entrants, ["company_name_canonical", "year"])
    save_csv(new_entrants, OUT_TABLE / "T16_new_entrants_profile.csv")

    # Sensitivities: no-aggregator, company cap, pooled 2024 baseline.
    noagg_frame = add_metrics(frame.loc[~frame["is_aggregator"]].copy())
    noagg_frame = noagg_frame[noagg_frame["company_name_canonical"].isin(company_panel3_noagg["company_name_canonical"])].copy()
    company_period_noagg = company_period_summary(noagg_frame, ["company_name_canonical", "year"])
    save_csv(company_period_noagg, OUT_TABLE / "T16_company_period_summary_noagg.csv")
    save_csv(
        pd.DataFrame([decomposition_table(company_period_noagg, metric) for metric in [
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
        ]]),
        OUT_TABLE / "T16_decomposition_noagg.csv",
    )

    capped = company_cap(frame.copy(), 25, ["year", "company_name_canonical"])
    capped = add_metrics(capped)
    capped_panel, _ = overlap_company_panel(capped, min_n=3, noagg=False)
    company_period_cap = company_period_summary(capped[capped["company_name_canonical"].isin(capped_panel["company_name_canonical"])], ["company_name_canonical", "year"])
    save_csv(company_period_cap, OUT_TABLE / "T16_company_period_summary_cap25.csv")
    decomp_cap = pd.DataFrame([decomposition_table(company_period_cap, metric) for metric in [
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
    ]])
    save_csv(decomp_cap, OUT_TABLE / "T16_decomposition_cap25.csv")

    pooled = pooled_2024_frame()
    pooled_panel_counts = (
        pooled.groupby(["company_name_canonical", "year"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"2024": "n_2024", "2026": "n_2026"})
    )
    pooled_panel = pooled_panel_counts[(pooled_panel_counts["n_2024"] >= 3) & (pooled_panel_counts["n_2026"] >= 3)].copy()
    pooled_company_period = company_period_summary(
        pooled[pooled["company_name_canonical"].isin(pooled_panel["company_name_canonical"])],
        ["company_name_canonical", "year"],
    )
    save_csv(pooled_panel_counts, OUT_TABLE / "T16_pooled2024_overlap_counts.csv")
    save_csv(pooled_company_period, OUT_TABLE / "T16_pooled2024_company_period_summary.csv")
    save_csv(
        pd.DataFrame([decomposition_table(pooled_company_period, metric) for metric in [
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
        ]]),
        OUT_TABLE / "T16_pooled2024_decomposition.csv",
    )

    # llm-only company-panel coverage for text-sensitivity
    llm_company_counts = (
        frame.loc[frame["text_source"] == "llm"]
        .groupby(["company_name_canonical", "year"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"2024": "n_2024", "2026": "n_2026"})
    )
    llm_company_panel = llm_company_counts[(llm_company_counts["n_2024"] >= 3) & (llm_company_counts["n_2026"] >= 3)].copy()
    save_csv(llm_company_counts, OUT_TABLE / "T16_llm_overlap_counts.csv")
    save_csv(llm_company_panel, OUT_TABLE / "T16_llm_overlap_panel.csv")

    # Figures
    fig = plot_cluster_heatmap(centroids, primary_metrics)
    save_fig(fig, OUT_FIG / "T16_cluster_heatmap.png")
    plt.close(fig)

    fig = plot_pca(cluster_df, primary_metrics)
    save_fig(fig, OUT_FIG / "T16_cluster_pca.png")
    plt.close(fig)

    fig = plot_decomposition(decomp_primary.assign(metric=lambda d: d["metric"]))
    save_fig(fig, OUT_FIG / "T16_decomposition_bars.png")
    plt.close(fig)

    fig = plot_domain_shift(domain_change)
    save_fig(fig, OUT_FIG / "T16_domain_family_shift.png")
    plt.close(fig)

    coverage = pd.DataFrame(
        [
            {
                "panel": "primary_overlap_ge3",
                "companies": int(len(company_panel3)),
                "companies_ge5": int(len(company_panel5)),
                "rows_2024": int(company_period[company_period["year"] == "2024"]["n_posts"].sum()),
                "rows_2026": int(company_period[company_period["year"] == "2026"]["n_posts"].sum()),
                "llm_text_share_2024": float(company_period.loc[company_period["year"] == "2024", "llm_text_share"].mean()),
                "llm_text_share_2026": float(company_period.loc[company_period["year"] == "2026", "llm_text_share"].mean()),
            },
            {
                "panel": "no_aggregator_overlap_ge3",
                "companies": int(len(company_panel3_noagg)),
                "companies_ge5": int(len(company_panel5_noagg)),
                "rows_2024": int(company_period_noagg[company_period_noagg["year"] == "2024"]["n_posts"].sum()),
                "rows_2026": int(company_period_noagg[company_period_noagg["year"] == "2026"]["n_posts"].sum()),
                "llm_text_share_2024": float(company_period_noagg.loc[company_period_noagg["year"] == "2024", "llm_text_share"].mean()),
                "llm_text_share_2026": float(company_period_noagg.loc[company_period_noagg["year"] == "2026", "llm_text_share"].mean()),
            },
            {
                "panel": "cap25_overlap_ge3",
                "companies": int(len(capped_panel)),
                "companies_ge5": int((capped_panel["n_2024"] >= 5).sum()),
                "rows_2024": int(company_period_cap[company_period_cap["year"] == "2024"]["n_posts"].sum()),
                "rows_2026": int(company_period_cap[company_period_cap["year"] == "2026"]["n_posts"].sum()),
                "llm_text_share_2024": float(company_period_cap.loc[company_period_cap["year"] == "2024", "llm_text_share"].mean()),
                "llm_text_share_2026": float(company_period_cap.loc[company_period_cap["year"] == "2026", "llm_text_share"].mean()),
            },
            {
                "panel": "pooled_2024_overlap_ge3",
                "companies": int(len(pooled_panel)),
                "companies_ge5": int(((pooled_panel["n_2024"] >= 5) & (pooled_panel["n_2026"] >= 5)).sum()),
                "rows_2024": int(pooled_company_period[pooled_company_period["year"] == "2024"]["n_posts"].sum()),
                "rows_2026": int(pooled_company_period[pooled_company_period["year"] == "2026"]["n_posts"].sum()),
                "llm_text_share_2024": float(pooled_company_period.loc[pooled_company_period["year"] == "2024", "llm_text_share"].mean()),
                "llm_text_share_2026": float(pooled_company_period.loc[pooled_company_period["year"] == "2026", "llm_text_share"].mean()),
            },
            {
                "panel": "llm_only_overlap_ge3",
                "companies": int(len(llm_company_panel)),
                "companies_ge5": int((llm_company_panel["n_2024"] >= 5).sum()) if not llm_company_panel.empty else 0,
                "rows_2024": int(llm_company_counts.get("n_2024", pd.Series(dtype=float)).sum()),
                "rows_2026": int(llm_company_counts.get("n_2026", pd.Series(dtype=float)).sum()),
                "llm_text_share_2024": 1.0,
                "llm_text_share_2026": 1.0,
            },
        ]
    )
    save_csv(coverage, OUT_TABLE / "T16_panel_coverage_summary.csv")


if __name__ == "__main__":
    main()
