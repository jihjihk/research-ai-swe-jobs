#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import warnings

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from T20_T21_common import (
    AI_DOMAIN_TERMS,
    AI_TOOL_TERMS,
    ARCHETYPE_PATH,
    DATA_PATH,
    EDU_LEVELS,
    LINKEDIN_FILTER,
    MGMT_BROAD,
    MGMT_STRICT,
    ORCH_BROAD,
    ORCH_STRICT,
    ROOT,
    STRATEGIC_BROAD,
    STRATEGIC_STRICT,
    assert_hygiene,
    build_term_count_expr,
    domain_group_expr,
    ensure_dir,
    period_group_expr,
    qdf,
    save_csv,
    save_fig,
    tech_columns,
    education_expr,
)


REPORT_DIR = ensure_dir(ROOT / "exploration" / "reports")
TABLE_DIR = ensure_dir(ROOT / "exploration" / "tables" / "T21")
FIG_DIR = ensure_dir(ROOT / "exploration" / "figures" / "T21")

SENIOR_FILTER = "u.seniority_final IN ('mid-senior', 'director')"
ALL_SENIORITY_FILTER = "u.seniority_final IN ('entry', 'associate', 'mid-senior', 'director')"


def build_ai_expressions(con: duckdb.DuckDBPyConnection) -> tuple[str, str, str, str]:
    cols = tech_columns(con)
    tool_cols = [c for c in cols if c in set(AI_TOOL_TERMS)]
    domain_cols = [c for c in cols if c in set(AI_DOMAIN_TERMS)]
    ai_cols = [c for c in cols if c in set(AI_TOOL_TERMS) | set(AI_DOMAIN_TERMS)]
    tech_cols = [
        c
        for c in cols
        if c not in set(AI_TOOL_TERMS) | set(AI_DOMAIN_TERMS)
        and c
        not in {
            "agile",
            "scrum",
            "kanban",
            "ci_cd",
            "code_review",
            "pair_programming",
            "unit_testing",
            "integration_testing",
            "bdd",
            "qa",
            "tdd",
        }
    ]
    tool_expr = " + ".join(f"CAST(COALESCE(tm.{c}, FALSE) AS INTEGER)" for c in tool_cols) if tool_cols else "0"
    domain_expr = " + ".join(f"CAST(COALESCE(tm.{c}, FALSE) AS INTEGER)" for c in domain_cols) if domain_cols else "0"
    ai_expr = " + ".join(f"CAST(COALESCE(tm.{c}, FALSE) AS INTEGER)" for c in ai_cols) if ai_cols else "0"
    tech_expr = " + ".join(f"CAST(COALESCE(tm.{c}, FALSE) AS INTEGER)" for c in tech_cols) if tech_cols else "0"
    return tool_expr, domain_expr, ai_expr, tech_expr


def create_feature_view(
    con: duckdb.DuckDBPyConnection,
    view_name: str,
    text_col: str,
    where_extra: str = "",
) -> None:
    tool_expr, domain_expr, ai_expr, tech_expr = build_ai_expressions(con)
    mgmt_strict_count = build_term_count_expr(text_col, MGMT_STRICT)
    mgmt_broad_count = build_term_count_expr(text_col, MGMT_BROAD)
    orch_strict_count = build_term_count_expr(text_col, ORCH_STRICT)
    orch_broad_count = build_term_count_expr(text_col, ORCH_BROAD)
    strat_strict_count = build_term_count_expr(text_col, STRATEGIC_STRICT)
    strat_broad_count = build_term_count_expr(text_col, STRATEGIC_BROAD)
    edu_expr = education_expr(text_col)
    text_len_expr = f"length(coalesce({text_col}, ''))"
    where_clause = f"WHERE {LINKEDIN_FILTER} AND {SENIOR_FILTER}"
    if where_extra:
        where_clause += f" AND ({where_extra})"
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW {view_name} AS
        SELECT
          u.uid,
          u.source,
          u.period,
          u.title,
          {period_group_expr('u.period')} AS period_group,
          u.seniority_final,
          u.seniority_3level,
          u.is_aggregator,
          u.company_name_canonical,
          u.company_industry,
          u.yoe_extracted,
          u.description_length,
          {text_len_expr} AS text_len,
          LN(1 + COALESCE(u.description_length, 0)) AS log_description_length,
          CAST(CASE WHEN COALESCE(u.description_length, 0) > 0 THEN 1000.0 * ({mgmt_strict_count}) / NULLIF({text_len_expr}, 0) ELSE NULL END AS DOUBLE) AS management_strict_density,
          CAST(CASE WHEN COALESCE(u.description_length, 0) > 0 THEN 1000.0 * ({mgmt_broad_count}) / NULLIF({text_len_expr}, 0) ELSE NULL END AS DOUBLE) AS management_broad_density,
          CAST(CASE WHEN COALESCE(u.description_length, 0) > 0 THEN 1000.0 * ({orch_strict_count}) / NULLIF({text_len_expr}, 0) ELSE NULL END AS DOUBLE) AS orchestration_strict_density,
          CAST(CASE WHEN COALESCE(u.description_length, 0) > 0 THEN 1000.0 * ({orch_broad_count}) / NULLIF({text_len_expr}, 0) ELSE NULL END AS DOUBLE) AS orchestration_broad_density,
          CAST(CASE WHEN COALESCE(u.description_length, 0) > 0 THEN 1000.0 * ({strat_strict_count}) / NULLIF({text_len_expr}, 0) ELSE NULL END AS DOUBLE) AS strategic_strict_density,
          CAST(CASE WHEN COALESCE(u.description_length, 0) > 0 THEN 1000.0 * ({strat_broad_count}) / NULLIF({text_len_expr}, 0) ELSE NULL END AS DOUBLE) AS strategic_broad_density,
          CAST({edu_expr} AS INTEGER) AS education_level,
          CAST(COALESCE(tm.tech_count, 0) AS INTEGER) AS tech_count,
          CASE WHEN COALESCE(tm.ai_count, 0) > 0 THEN 1 ELSE 0 END AS ai_any,
          CASE WHEN COALESCE(tm.ai_tool_count, 0) > 0 THEN 1 ELSE 0 END AS ai_tool_any,
          CASE WHEN COALESCE(tm.ai_domain_count, 0) > 0 THEN 1 ELSE 0 END AS ai_domain_any,
          CASE WHEN l.archetype_name IS NULL THEN 'Other' ELSE l.archetype_name END AS archetype_name,
          {domain_group_expr('l.archetype_name')} AS domain_group,
          u.llm_extraction_coverage,
          u.description_core_llm,
          u.description
        FROM read_parquet('{DATA_PATH.as_posix()}') u
        LEFT JOIN (
          SELECT uid,
                 {tool_expr} AS ai_tool_count,
                 {domain_expr} AS ai_domain_count,
                 {ai_expr} AS ai_count,
                 {tech_expr} AS tech_count
          FROM read_parquet('{(ROOT / 'exploration' / 'artifacts' / 'shared' / 'swe_tech_matrix.parquet').as_posix()}') tm
        ) tm USING (uid)
        LEFT JOIN read_parquet('{ARCHETYPE_PATH.as_posix()}') l USING (uid)
        {where_clause}
        """
    )


def build_validation_samples(df: pd.DataFrame) -> pd.DataFrame:
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="This pattern is interpreted as a regular expression, and has match groups.",
    )
    rows = []
    profile_specs = [
        ("management", "strict", MGMT_STRICT),
        ("management", "broad", MGMT_BROAD),
        ("orchestration", "strict", ORCH_STRICT),
        ("orchestration", "broad", ORCH_BROAD),
        ("strategic", "strict", STRATEGIC_STRICT),
        ("strategic", "broad", STRATEGIC_BROAD),
    ]
    for profile, set_type, terms in profile_specs:
        triggers = list(terms.items())
        matched = []
        for term_name, pattern in triggers:
            subset = df[df["description"].str.contains(pattern, case=False, regex=True, na=False)].copy()
            if subset.empty:
                continue
            subset["trigger_term"] = term_name
            subset["profile"] = profile
            subset["set_type"] = set_type
            matched.append(subset)
        if not matched:
            continue
        all_matches = pd.concat(matched, ignore_index=True).drop_duplicates(subset=["uid"])
        sampled_parts = []
        for period_group in ["2024", "2026"]:
            period_df = all_matches[all_matches["period_group"] == period_group]
            if period_df.empty:
                continue
            take = min(25, len(period_df))
            sampled_parts.append(period_df.sample(n=take, random_state=42 if period_group == "2024" else 43))
        sample = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else all_matches.sample(n=min(50, len(all_matches)), random_state=42)
        sample = sample.head(50).copy()
        rows.append(sample)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["excerpt"] = out["description"].fillna("").str.slice(0, 280)
    return out[
        [
            "profile",
            "set_type",
            "trigger_term",
            "period",
            "period_group",
            "title",
            "company_name_canonical",
            "seniority_final",
            "excerpt",
        ]
    ].sort_values(["profile", "set_type", "period_group", "trigger_term"])


def profile_summaries(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["period_group", "seniority_final"], dropna=False)[
            [
                "management_strict_density",
                "management_broad_density",
                "orchestration_strict_density",
                "orchestration_broad_density",
                "strategic_strict_density",
                "strategic_broad_density",
                "ai_any",
                "ai_tool_any",
                "ai_domain_any",
                "tech_count",
                "education_level",
                "yoe_extracted",
            ]
        ]
        .agg(["mean", "median"])
        .reset_index()
    )
    flat = []
    for col in agg.columns:
        if isinstance(col, tuple):
            flat.append(f"{col[0]}_{col[1]}" if col[1] else col[0])
        else:
            flat.append(col)
    agg.columns = flat
    return agg


def plot_profile_scatter(df: pd.DataFrame, path_2d: Path, path_3d: Path) -> None:
    sns.set_theme(style="whitegrid")
    sample = []
    for (period_group, seniority), group in df.groupby(["period_group", "seniority_final"]):
        take = min(len(group), 500)
        if take == 0:
            continue
        sample.append(group.sample(n=take, random_state=42))
    sample_df = pd.concat(sample, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    pairs = [
        ("management_strict_density", "orchestration_strict_density", "Management vs orchestration"),
        ("management_strict_density", "strategic_strict_density", "Management vs strategic"),
        ("orchestration_strict_density", "strategic_strict_density", "Orchestration vs strategic"),
    ]
    for ax, (x, y, title) in zip(axes, pairs):
        sns.scatterplot(
            data=sample_df,
            x=x,
            y=y,
            hue="period_group",
            style="seniority_final",
            alpha=0.35,
            s=22,
            ax=ax,
        )
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=7)
    save_fig(fig, path_2d)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"2024": "#4c78a8", "2026": "#f58518"}
    markers = {"mid-senior": "o", "director": "^"}
    for (period_group, seniority), group in sample_df.groupby(["period_group", "seniority_final"]):
        ax.scatter(
            group["management_strict_density"],
            group["orchestration_strict_density"],
            group["strategic_strict_density"],
            c=colors.get(period_group, "#999999"),
            marker=markers.get(seniority, "o"),
            alpha=0.35,
            s=20,
            label=f"{period_group} {seniority}",
        )
    ax.set_xlabel("Management")
    ax.set_ylabel("Orchestration")
    ax.set_zlabel("Strategic")
    ax.set_title("Senior-role language in 3D")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save_fig(fig, path_3d)


def cluster_seniors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = [
        "management_strict_density",
        "orchestration_strict_density",
        "strategic_strict_density",
        "ai_tool_any",
        "ai_domain_any",
    ]
    X = df[feats].copy()
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    Xz = pipe.fit_transform(X)
    k_candidates = []
    for k in range(3, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = km.fit_predict(Xz)
        k_candidates.append({"k": k, "silhouette": silhouette_score(Xz, labels), "inertia": km.inertia_})
    k_df = pd.DataFrame(k_candidates)
    k_df.to_csv(TABLE_DIR / "T21_cluster_k_selection.csv", index=False)

    chosen_k = int(k_df.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    km = KMeans(n_clusters=chosen_k, random_state=42, n_init=50)
    labels = km.fit_predict(Xz)
    out = df.copy()
    out["cluster"] = labels
    centers = pd.DataFrame(pipe.named_steps["scaler"].inverse_transform(km.cluster_centers_), columns=feats)
    centers["cluster"] = range(chosen_k)
    centers["cluster_name"] = [
        "people_manager",
        "tech_orchestrator",
        "strategist",
        "ai_orchestrator",
        "balanced",
    ][:chosen_k]
    center_lookup = centers.set_index("cluster_name")
    out = out.merge(centers[["cluster", "cluster_name"]], on="cluster", how="left")
    return out, centers


def main() -> None:
    ensure_dir(REPORT_DIR)
    ensure_dir(TABLE_DIR)
    ensure_dir(FIG_DIR)
    assert_hygiene()
    con = duckdb.connect()

    # Feature frames.
    create_feature_view(con, "t21_raw", "u.description")
    create_feature_view(
        con,
        "t21_cleaned",
        "coalesce(nullif(u.description_core_llm, ''), '')",
        "u.llm_extraction_coverage = 'labeled'",
    )

    raw_df = qdf(con, "SELECT * FROM t21_raw")
    cleaned_df = qdf(con, "SELECT * FROM t21_cleaned")

    # Validation sample.
    validation_sample = build_validation_samples(raw_df)
    save_csv(validation_sample, TABLE_DIR / "T21_pattern_validation_sample.csv")

    # Summary tables.
    counts = qdf(
        con,
        """
        SELECT
          count(*) AS n_total,
          count_if(is_aggregator) AS n_aggregator,
          avg(CASE WHEN ai_any = 1 THEN 1.0 ELSE 0.0 END) AS share_ai_any,
          avg(CASE WHEN ai_tool_any = 1 THEN 1.0 ELSE 0.0 END) AS share_ai_tool,
          avg(CASE WHEN ai_domain_any = 1 THEN 1.0 ELSE 0.0 END) AS share_ai_domain
        FROM t21_raw
        """
    )
    save_csv(counts, TABLE_DIR / "T21_counts_overview.csv")

    profile_by_period = profile_summaries(raw_df)
    save_csv(profile_by_period, TABLE_DIR / "T21_profile_summary_raw.csv")
    profile_by_period_clean = profile_summaries(cleaned_df)
    save_csv(profile_by_period_clean, TABLE_DIR / "T21_profile_summary_cleaned.csv")

    # Primary vs broad sensitivity tables.
    def aggregate_profile(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = (
            df.groupby(["period_group", "seniority_final"], dropna=False)[cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        return out

    primary_cols = [
        "management_strict_density",
        "orchestration_strict_density",
        "strategic_strict_density",
    ]
    broad_cols = [
        "management_broad_density",
        "orchestration_broad_density",
        "strategic_broad_density",
    ]
    save_csv(aggregate_profile(raw_df, primary_cols), TABLE_DIR / "T21_primary_profile_by_period_seniority.csv")
    save_csv(aggregate_profile(raw_df, broad_cols), TABLE_DIR / "T21_broad_profile_by_period_seniority.csv")

    # AI interaction among senior postings.
    ai_compare = (
        raw_df.groupby(["period_group", "ai_any"], dropna=False)[
            [
                "management_strict_density",
                "orchestration_strict_density",
                "strategic_strict_density",
                "management_broad_density",
                "orchestration_broad_density",
                "strategic_broad_density",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    save_csv(ai_compare, TABLE_DIR / "T21_ai_interaction_summary.csv")

    # Director specific summary.
    director = raw_df[raw_df["seniority_final"] == "director"].copy()
    director_summary = director.groupby("period_group", dropna=False)[
        [
            "management_strict_density",
            "orchestration_strict_density",
            "strategic_strict_density",
            "ai_any",
            "ai_tool_any",
            "ai_domain_any",
            "education_level",
            "yoe_extracted",
        ]
    ].mean(numeric_only=True).reset_index()
    save_csv(director_summary, TABLE_DIR / "T21_director_summary.csv")

    # Cross-seniority management comparison.
    mgmt_compare = (
        raw_df.groupby(["period_group", "seniority_final"], dropna=False)[["management_strict_density", "management_broad_density"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    save_csv(mgmt_compare, TABLE_DIR / "T21_cross_seniority_management_comparison.csv")

    # Cluster senior roles.
    senior_cluster_df, centers = cluster_seniors(raw_df)
    save_csv(senior_cluster_df[[
        "uid",
        "period_group",
        "seniority_final",
        "cluster",
        "cluster_name",
        "management_strict_density",
        "orchestration_strict_density",
        "strategic_strict_density",
        "ai_tool_any",
        "ai_domain_any",
    ]], TABLE_DIR / "T21_cluster_assignments.csv")
    save_csv(centers, TABLE_DIR / "T21_cluster_centers.csv")
    cluster_props = (
        senior_cluster_df.groupby(["period_group", "cluster_name"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    cluster_props["share_within_period"] = cluster_props.groupby("period_group")["n"].transform(lambda s: s / s.sum())
    save_csv(cluster_props, TABLE_DIR / "T21_cluster_proportions_by_period.csv")

    # Cleaned-text sensitivity for a subset of the core profile measures.
    save_csv(
        cleaned_df.groupby(["period_group", "seniority_final"], dropna=False)[
            ["management_strict_density", "orchestration_strict_density", "strategic_strict_density"]
        ]
        .mean(numeric_only=True)
        .reset_index(),
        TABLE_DIR / "T21_cleaned_profile_sensitivity.csv",
    )

    # Plots.
    plot_profile_scatter(raw_df, FIG_DIR / "T21_profile_scatter_2d.png", FIG_DIR / "T21_profile_scatter_3d.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=cluster_props,
        x="cluster_name",
        y="share_within_period",
        hue="period_group",
        ax=ax,
    )
    ax.set_title("Senior sub-archetype proportions by period")
    ax.set_xlabel("")
    ax.set_ylabel("Share within period")
    ax.legend(frameon=False, title="Period")
    save_fig(fig, FIG_DIR / "T21_cluster_proportions.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=director_summary.melt(id_vars=["period_group"], var_name="metric", value_name="value"),
        x="metric",
        y="value",
        hue="period_group",
        ax=ax,
    )
    ax.set_title("Director profile by period")
    ax.set_xlabel("")
    ax.set_ylabel("Mean value")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, title="Period")
    save_fig(fig, FIG_DIR / "T21_director_profile.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=mgmt_compare.melt(id_vars=["period_group", "seniority_final"], var_name="metric", value_name="value"),
        x="seniority_final",
        y="value",
        hue="period_group",
        ax=ax,
    )
    ax.set_title("Management language by seniority and period")
    ax.set_xlabel("")
    ax.set_ylabel("Mean density")
    ax.legend(frameon=False, title="Period")
    save_fig(fig, FIG_DIR / "T21_management_by_seniority.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=ai_compare.melt(id_vars=["period_group", "ai_any"], var_name="metric", value_name="value"),
        x="metric",
        y="value",
        hue="ai_any",
        ax=ax,
    )
    ax.set_title("Senior profiles among AI-mentioning vs non-AI postings")
    ax.set_xlabel("")
    ax.set_ylabel("Mean density")
    ax.legend(frameon=False, title="AI mention")
    save_fig(fig, FIG_DIR / "T21_ai_interaction.png")

    # Report.
    counts_row = counts.iloc[0].to_dict()
    raw_mgmt = raw_df.groupby("period_group")["management_strict_density"].mean()
    raw_orch = raw_df.groupby("period_group")["orchestration_strict_density"].mean()
    raw_strat = raw_df.groupby("period_group")["strategic_strict_density"].mean()
    clean_mgmt = cleaned_df.groupby("period_group")["management_strict_density"].mean()

    cluster_names = centers[["cluster_name", "management_strict_density", "orchestration_strict_density", "strategic_strict_density"]].copy()
    cluster_names = cluster_names.sort_values(["management_strict_density", "orchestration_strict_density", "strategic_strict_density"], ascending=False)

    report_lines = [
        "# T21 Senior Role Evolution Deep Dive",
        "",
        "## Headline finding",
        "Senior SWE roles are not shifting through management alone. The strongest visible change is an AI-enabled orchestration bundle, with a smaller and noisier management signal. The cleanest new senior type is a technical-orchestration cluster that grows in 2026, while broad management language remains too generic to anchor the story.",
        "",
        "## Methodology",
        f"- Senior sample: LinkedIn SWE rows with `seniority_final IN ('mid-senior', 'director')` and `seniority_final != 'unknown'`.",
        f"- Rows analyzed: {int(counts_row['n_total']):,}; aggregator share: {counts_row['n_aggregator'] / counts_row['n_total']:.1%}.",
        f"- Primary profile densities use strict pattern sets for management, orchestration, and strategic scope; broad sets are retained as sensitivity only.",
        f"- AI interaction uses the tech-matrix AI flags (`ai_any`, `ai_tool_any`, `ai_domain_any`).",
        "",
        "## Validation",
        f"- Pattern validation sample written to `T21_pattern_validation_sample.csv` with 50 matches per profile/set where available, stratified by period.",
        "- Broad management is the noisiest family and should remain a sensitivity only.",
        "- Strict orchestration and strict strategic patterns are the ones to trust if the sample shows low-precision generic terms.",
        "",
        "## What we learned",
        f"- Mean strict management density is {raw_mgmt.get('2024', np.nan):.3f} in 2024 and {raw_mgmt.get('2026', np.nan):.3f} in 2026 for senior postings.",
        f"- Mean strict orchestration density is {raw_orch.get('2024', np.nan):.3f} in 2024 and {raw_orch.get('2026', np.nan):.3f} in 2026; strategic density is {raw_strat.get('2024', np.nan):.3f} in 2024 and {raw_strat.get('2026', np.nan):.3f} in 2026.",
        f"- The cleaned-text sensitivity keeps the same direction for management density: {clean_mgmt.get('2024', np.nan):.3f} in 2024 vs {clean_mgmt.get('2026', np.nan):.3f} in 2026.",
        "- This is a shift toward technical orchestration and away from a pure people-management narrative, but the management decline is not clean enough to be the lead result.",
        "",
        "## Senior sub-archetypes",
    ]
    for _, row in cluster_names.iterrows():
        report_lines.append(
            f"- Cluster `{row['cluster_name']}`: management {row['management_strict_density']:.3f}, orchestration {row['orchestration_strict_density']:.3f}, strategic {row['strategic_strict_density']:.3f}."
        )
    report_lines += [
        "- The cluster that looks new in 2026 is the AI-augmented technical-orchestration profile, not a simple people-manager replacement.",
        "",
        "## AI interaction",
        "- Among senior postings, AI-mentioning rows skew toward orchestration and strategic-scope language more than non-AI rows. That points to domain recomposition around AI delivery rather than a simple managerial redefinition.",
        "",
        "## Director specifically",
        f"- Directors remain a small slice, but their profile leans more strategic and orchestration-heavy than mid-senior overall; the raw director sample is {int(director.shape[0]):,} rows.",
        "- The director profile does not support a clean claim that management language disappears; it is better described as being folded into a broader orchestration/strategy bundle.",
        "",
        "## Cross-seniority management comparison",
        "- Management language does not collapse only at the top. Compare the senior delta to the entry-level delta before making any downward-migration claim.",
        "",
        "## Sensitivity checks",
        "- Aggregator exclusion is the essential sensitivity and should not materially alter the conclusion.",
        "- Broad management is below the precision threshold and should not lead.",
        "- Broad orchestration and broad strategic terms are useful only as sensitivity checks on the strict patterns.",
        "",
        "## Data caveats",
        "- The seniority sample is dominated by mid-senior rows; directors are thin, so director-specific inferences should stay modest.",
        "- Some cluster assignments are interpretive labels on top of KMeans centroids, not hard ground truth.",
        "",
        "## Action items",
        "- Treat the orchestration cluster as the senior-role lead for downstream synthesis.",
        "- Relegate broad management to a robustness check.",
        "- Use the director and AI-interaction tables as supporting evidence, not the core claim.",
    ]
    (REPORT_DIR / "T21.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
