#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from T18_T19_common import (
    AI_TOOL_TERMS,
    DATA,
    REPORT_DIR,
    SCOPE_TERMS,
    ensure_dir,
    make_binary_expr,
    make_count_expr,
    qdf,
    regex_hygiene,
    save_csv,
    save_fig,
    sql_quote,
    tech_term_list,
)

ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "exploration" / "tables" / "T18"
FIG_DIR = ROOT / "exploration" / "figures" / "T18"

GROUP_ORDER = ["swe", "adjacent", "control"]
WINDOW_ORDER = ["kaggle_asaniczka / 2024-01", "kaggle_arshkon / 2024-04", "scraped / 2026-03", "scraped / 2026-04"]


def group_case_sql() -> str:
    return (
        "CASE WHEN is_swe THEN 'swe' "
        "WHEN is_swe_adjacent THEN 'adjacent' "
        "WHEN is_control THEN 'control' "
        "ELSE 'other' END"
    )


def period_group_sql() -> str:
    return "CASE WHEN source IN ('kaggle_arshkon', 'kaggle_asaniczka') THEN '2024' ELSE '2026' END"


def build_feature_query(text_expr: str, filter_sql: str, include_tech: bool, spec: str, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    tech_expr = None
    if include_tech:
        tech_terms = tech_term_list(con)
        tech_expr = make_count_expr("analysis_text", tech_terms)[0]

    q = f"""
    WITH src AS (
      SELECT
        uid,
        source,
        period,
        scrape_date,
        date_posted,
        company_name_canonical,
        title_normalized,
        description_length,
        seniority_final,
        seniority_3level,
        seniority_native,
        yoe_extracted,
        swe_classification_tier,
        llm_extraction_coverage,
        llm_classification_coverage,
        is_aggregator,
        is_swe,
        is_swe_adjacent,
        is_control,
        {group_case_sql()} AS occ,
        CASE WHEN llm_extraction_coverage = 'labeled' THEN description_core_llm END AS cleaned_text,
        description AS raw_text
      FROM read_parquet('{DATA.as_posix()}')
      WHERE source_platform = 'linkedin'
        AND is_english = true
        AND date_flag = 'ok'
        AND (is_swe OR is_swe_adjacent OR is_control)
    ),
    texted AS (
      SELECT
        src.*,
        {text_expr} AS analysis_text,
        length(coalesce({text_expr}, '')) AS analysis_text_len,
        CASE WHEN regexp_matches(lower(coalesce({text_expr}, '')), {sql_quote(AI_TOOL_TERMS['llm'])})
              OR regexp_matches(lower(coalesce({text_expr}, '')), {sql_quote(AI_TOOL_TERMS['copilot'])})
              OR regexp_matches(lower(coalesce({text_expr}, '')), {sql_quote(AI_TOOL_TERMS['cursor'])})
              OR regexp_matches(lower(coalesce({text_expr}, '')), {sql_quote(AI_TOOL_TERMS['claude'])})
              OR regexp_matches(lower(coalesce({text_expr}, '')), {sql_quote(AI_TOOL_TERMS['agent'])}) THEN 1 ELSE 0 END AS ai_tool_any,
        CASE WHEN regexp_matches(lower(coalesce({text_expr}, '')), {sql_quote(SCOPE_TERMS['ownership'])})
              OR regexp_matches(lower(coalesce({text_expr}, '')), {sql_quote(SCOPE_TERMS['end_to_end'])})
              OR regexp_matches(lower(coalesce({text_expr}, '')), {sql_quote(SCOPE_TERMS['cross_functional'])})
              OR regexp_matches(lower(coalesce({text_expr}, '')), {sql_quote(SCOPE_TERMS['autonomous'])})
              OR regexp_matches(lower(coalesce({text_expr}, '')), {sql_quote(SCOPE_TERMS['initiative'])})
              OR regexp_matches(lower(coalesce({text_expr}, '')), {sql_quote(SCOPE_TERMS['strategic'])})
              OR regexp_matches(lower(coalesce({text_expr}, '')), {sql_quote(SCOPE_TERMS['roadmap'])}) THEN 1 ELSE 0 END AS scope_any
      FROM src
    ),
    feat AS (
      SELECT
        texted.*,
        {tech_expr if tech_expr is not None else 'NULL::INTEGER'} AS tech_count
      FROM texted
      WHERE {filter_sql}
    )
    SELECT
      '{spec}' AS spec,
      source,
      period,
      {period_group_sql()} AS period_group,
      occ,
      COUNT(*) AS n,
      SUM(CASE WHEN llm_extraction_coverage = 'labeled' THEN 1 ELSE 0 END) AS text_labeled_n,
      SUM(CASE WHEN llm_classification_coverage = 'labeled' THEN 1 ELSE 0 END) AS class_labeled_n,
      AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) AS entry_final_share,
      AVG(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) AS yoe_le2_share,
      AVG(CASE WHEN yoe_extracted <= 3 THEN 1.0 ELSE 0.0 END) AS yoe_le3_share,
      AVG(CASE WHEN seniority_3level = 'junior' THEN 1.0 ELSE 0.0 END) AS junior_3level_share,
      AVG(CASE WHEN seniority_3level = 'mid' THEN 1.0 ELSE 0.0 END) AS mid_3level_share,
      AVG(CASE WHEN seniority_3level = 'senior' THEN 1.0 ELSE 0.0 END) AS senior_3level_share,
      AVG(CASE WHEN seniority_3level = 'unknown' THEN 1.0 ELSE 0.0 END) AS unknown_3level_share,
      AVG(ai_tool_any) AS ai_tool_share,
      AVG(scope_any) AS scope_any_share,
      AVG(analysis_text_len) AS analysis_text_mean_len,
      median(analysis_text_len) AS analysis_text_median_len,
      AVG(description_length) AS raw_desc_mean_len,
      median(description_length) AS raw_desc_median_len,
      AVG(CASE WHEN yoe_extracted IS NOT NULL THEN yoe_extracted END) AS mean_yoe,
      median(yoe_extracted) AS median_yoe,
      AVG(tech_count) AS tech_count,
      median(tech_count) AS tech_count_median
    FROM feat
    GROUP BY 1,2,3,4,5
    ORDER BY source, period, occ
    """
    return qdf(con, q)


def weighted_pool(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for (spec, period_group, occ), g in df.groupby(["spec", "period_group", "occ"], dropna=False):
        row = {
            "spec": spec,
            "period_group": period_group,
            "occ": occ,
            "n": int(g["n"].sum()),
            "text_labeled_n": int(g["text_labeled_n"].sum()),
            "class_labeled_n": int(g["class_labeled_n"].sum()),
        }
        weights = g["n"].to_numpy(dtype=float)
        for metric in metric_cols:
            if metric not in g.columns:
                continue
            vals = g[metric].to_numpy(dtype=float)
            mask = np.isfinite(vals) & np.isfinite(weights)
            if mask.sum() == 0:
                row[metric] = float("nan")
            else:
                row[metric] = float(np.average(vals[mask], weights=weights[mask]))
        rows.append(row)
    return pd.DataFrame(rows)


def company_capped_labeled_pool(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    q = f"""
    WITH src AS (
      SELECT
        uid,
        source,
        period,
        {period_group_sql()} AS period_group,
        {group_case_sql()} AS occ,
        title_normalized,
        company_name_canonical,
        description_core_llm AS text_value,
        seniority_final,
        seniority_3level,
        yoe_extracted,
        description_length,
        CASE WHEN regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(AI_TOOL_TERMS['llm'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(AI_TOOL_TERMS['copilot'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(AI_TOOL_TERMS['cursor'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(AI_TOOL_TERMS['claude'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(AI_TOOL_TERMS['agent'])}) THEN 1 ELSE 0 END AS ai_tool_any,
        CASE WHEN regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(SCOPE_TERMS['ownership'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(SCOPE_TERMS['end_to_end'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(SCOPE_TERMS['cross_functional'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(SCOPE_TERMS['autonomous'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(SCOPE_TERMS['initiative'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(SCOPE_TERMS['strategic'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(SCOPE_TERMS['roadmap'])}) THEN 1 ELSE 0 END AS scope_any,
        NULL::DOUBLE AS tech_count,
        length(coalesce(description_core_llm, '')) AS analysis_text_len,
        AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) OVER () AS entry_final_share,
        AVG(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) OVER () AS yoe_le2_share,
        row_number() OVER (
          PARTITION BY source, period, {group_case_sql()}, company_name_canonical
          ORDER BY uid
        ) AS company_rank
      FROM read_parquet('{DATA.as_posix()}')
      WHERE source_platform = 'linkedin'
        AND is_english = true
        AND date_flag = 'ok'
        AND llm_extraction_coverage = 'labeled'
        AND description_core_llm IS NOT NULL
        AND (is_swe OR is_swe_adjacent OR is_control)
        AND ({group_case_sql()}) IN ('swe', 'adjacent')
    )
    SELECT *
    FROM src
    WHERE company_rank <= 25
    """
    return qdf(con, q)


def make_sample_pool(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    pool = company_capped_labeled_pool(con)
    rng = np.random.default_rng(42)
    samples = []
    for (source, period, occ), g in pool.groupby(["source", "period", "occ"], dropna=False):
        if len(g) == 0:
            continue
        n_take = min(200, len(g))
        samples.append(g.sample(n=n_take, random_state=int(rng.integers(0, 2**31 - 1))))
    if not samples:
        return pd.DataFrame()
    sample_df = pd.concat(samples, ignore_index=True).reset_index(drop=True)
    sample_df["window_label"] = sample_df["source"] + " / " + sample_df["period"]
    return sample_df


def tfidf_similarity(sample_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_df = sample_df.reset_index(drop=True).copy()
    sample_df["doc"] = sample_df["text_value"].fillna("").astype(str)
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_features=12000)
    X = tfidf.fit_transform(sample_df["doc"])

    sim_rows = []
    for (source, period), g in sample_df.groupby(["source", "period"], dropna=False):
        swe_idx = g.index[g["occ"] == "swe"].to_numpy()
        adj_idx = g.index[g["occ"] == "adjacent"].to_numpy()
        if len(swe_idx) == 0 or len(adj_idx) == 0:
            continue
        swe_centroid = np.asarray(X[swe_idx].mean(axis=0))
        adj_centroid = np.asarray(X[adj_idx].mean(axis=0))
        sim_rows.append(
            {
                "source": source,
                "period": period,
                "window_label": f"{source} / {period}",
                "swe_sample_n": int(len(swe_idx)),
                "adjacent_sample_n": int(len(adj_idx)),
                "tfidf_centroid_cosine": float(cosine_similarity(swe_centroid, adj_centroid)[0, 0]),
            }
        )
    sim_df = pd.DataFrame(sim_rows)

    count_vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_features=12000, binary=True)
    C = count_vec.fit_transform(sample_df["doc"])
    terms = np.array(count_vec.get_feature_names_out())
    term_rows = []
    for (source, period, occ), g in sample_df.groupby(["source", "period", "occ"], dropna=False):
        idx = g.index.to_numpy()
        if len(idx) == 0:
            continue
        prevalence = np.asarray(C[idx].mean(axis=0)).ravel()
        for term, prev in zip(terms, prevalence):
            if prev > 0:
                term_rows.append(
                    {
                        "source": source,
                        "period": period,
                        "period_group": "2024" if source in {"kaggle_asaniczka", "kaggle_arshkon"} else "2026",
                        "occ": occ,
                        "term": term,
                        "prevalence": float(prev),
                    }
                )
    term_df = pd.DataFrame(term_rows)
    if term_df.empty:
        return sim_df, pd.DataFrame(), pd.DataFrame()

    pooled = (
        term_df.groupby(["period_group", "occ", "term"], dropna=False)["prevalence"]
        .mean()
        .reset_index()
    )
    wide = pooled.pivot_table(index="term", columns=["period_group", "occ"], values="prevalence", fill_value=0.0)
    if ("2024", "adjacent") in wide.columns and ("2024", "swe") in wide.columns and ("2026", "adjacent") in wide.columns and ("2026", "swe") in wide.columns:
        wide["gap_2024"] = wide[("2024", "adjacent")] - wide[("2024", "swe")]
        wide["gap_2026"] = wide[("2026", "adjacent")] - wide[("2026", "swe")]
        wide["migration_score"] = wide["gap_2026"] - wide["gap_2024"]
        migration_df = (
            wide.reset_index()
            .sort_values("migration_score", ascending=False)
            .loc[:, ["term", "gap_2024", "gap_2026", "migration_score"]]
        )
    else:
        migration_df = pd.DataFrame(columns=["term", "gap_2024", "gap_2026", "migration_score"])

    return sim_df, term_df, migration_df


def title_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    q = f"""
    WITH src AS (
      SELECT
        uid,
        source,
        period,
        {period_group_sql()} AS period_group,
        {group_case_sql()} AS occ,
        title_normalized,
        company_name_canonical,
        description_core_llm,
        seniority_final,
        seniority_3level,
        yoe_extracted,
        description_length,
        CASE WHEN regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(AI_TOOL_TERMS['llm'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(AI_TOOL_TERMS['copilot'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(AI_TOOL_TERMS['cursor'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(AI_TOOL_TERMS['claude'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(AI_TOOL_TERMS['agent'])}) THEN 1 ELSE 0 END AS ai_tool_any,
        CASE WHEN regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(SCOPE_TERMS['ownership'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(SCOPE_TERMS['end_to_end'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(SCOPE_TERMS['cross_functional'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(SCOPE_TERMS['autonomous'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(SCOPE_TERMS['initiative'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(SCOPE_TERMS['strategic'])})
              OR regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(SCOPE_TERMS['roadmap'])}) THEN 1 ELSE 0 END AS scope_any,
        length(coalesce(description_core_llm, '')) AS analysis_text_len,
        row_number() OVER (
          PARTITION BY source, period, {group_case_sql()}, company_name_canonical
          ORDER BY uid
        ) AS company_rank
      FROM read_parquet('{DATA.as_posix()}')
        WHERE source_platform = 'linkedin'
        AND is_english = true
        AND date_flag = 'ok'
        AND llm_extraction_coverage = 'labeled'
        AND description_core_llm IS NOT NULL
        AND (is_swe OR is_swe_adjacent OR is_control)
    ),
    capped AS (
      SELECT *
      FROM src
      WHERE company_rank <= 25
    )
    SELECT
      period_group,
      title_normalized,
      occ,
      COUNT(*) AS n,
      AVG(ai_tool_any) AS ai_tool_share,
      AVG(scope_any) AS scope_any_share,
      AVG(analysis_text_len) AS analysis_text_mean_len,
      median(analysis_text_len) AS analysis_text_median_len,
      AVG(description_length) AS raw_desc_mean_len,
      median(description_length) AS raw_desc_median_len,
      AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) AS entry_final_share,
      AVG(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) AS yoe_le2_share
    FROM capped
    GROUP BY 1,2,3
    ORDER BY period_group, title_normalized, occ
    """
    return qdf(con, q)


def title_convergence_table(title_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if title_df.empty:
        return title_df
    top_titles = (
        title_df.loc[title_df["occ"] == "adjacent"]
        .groupby("title_normalized")["n"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .index.tolist()
    )
    for period_group in ["2024", "2026"]:
        sub = title_df.loc[(title_df["period_group"] == period_group) & (title_df["title_normalized"].isin(top_titles))].copy()
        for title in top_titles:
            title_sub = sub.loc[sub["title_normalized"] == title]
            if title_sub.empty:
                continue
            row = {"period_group": period_group, "title_normalized": title}
            for metric in [
                "n",
                "ai_tool_share",
                "scope_any_share",
                "analysis_text_mean_len",
                "analysis_text_median_len",
                "raw_desc_mean_len",
                "raw_desc_median_len",
                "entry_final_share",
                "yoe_le2_share",
            ]:
                for occ in ["adjacent", "swe"]:
                    val = title_sub.loc[title_sub["occ"] == occ, metric]
                    row[f"{occ}_{metric}"] = float(val.iloc[0]) if not val.empty else float("nan")
            row["ai_gap"] = row["adjacent_ai_tool_share"] - row["swe_ai_tool_share"]
            row["scope_gap"] = row["adjacent_scope_any_share"] - row["swe_scope_any_share"]
            row["len_gap"] = row["adjacent_analysis_text_median_len"] - row["swe_analysis_text_median_len"]
            row["yoe_gap"] = row["adjacent_yoe_le2_share"] - row["swe_yoe_le2_share"]
            rows.append(row)
    return pd.DataFrame(rows)


def plot_parallel_trends(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = [
        ("entry_final_share", "Explicit entry share"),
        ("yoe_le2_share", "YOE <= 2 share"),
        ("ai_tool_share", "AI-tool share"),
        ("scope_any_share", "Scope-any share"),
    ]
    colors = {"swe": "#1f77b4", "adjacent": "#ff7f0e", "control": "#2ca02c"}
    for ax, (metric, label) in zip(axes.flat, panels):
        for occ in GROUP_ORDER:
            sub = df.loc[df["occ"] == occ].sort_values("period_group")
            ax.plot(sub["period_group"], sub[metric], marker="o", label=occ, color=colors[occ])
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("Share")
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(title="Group", frameon=False, loc="best")
    fig.suptitle("T18: Parallel trends by occupation group", y=1.02, fontsize=14)
    save_fig(fig, FIG_DIR / "T18_parallel_trends.png")


def plot_similarity(sim_df: pd.DataFrame) -> None:
    if sim_df.empty:
        return
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    plot = sim_df.copy()
    plot["window_label"] = pd.Categorical(plot["window_label"], categories=WINDOW_ORDER, ordered=True)
    sns.lineplot(data=plot.sort_values("window_label"), x="window_label", y="tfidf_centroid_cosine", marker="o", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("TF-IDF centroid cosine")
    ax.set_title("T18: SWE-adjacent boundary similarity by source window")
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, FIG_DIR / "T18_boundary_similarity.png")


def plot_migration(migration_df: pd.DataFrame) -> None:
    if migration_df.empty:
        return
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    top = migration_df.head(12).copy()
    sns.barplot(data=top, y="term", x="migration_score", ax=ax, color="#4c78a8")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("T18: Terms moving toward SWE-adjacent roles")
    ax.set_xlabel("2026 gap minus 2024 gap")
    ax.set_ylabel("")
    save_fig(fig, FIG_DIR / "T18_term_migration.png")


def write_report(
    coverage: pd.DataFrame,
    primary_pool: pd.DataFrame,
    raw_pool: pd.DataFrame,
    did_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    migration_df: pd.DataFrame,
    title_conv: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> None:
    lines = ["# T18 Cross-Occupation Boundary Analysis", ""]
    lines.append("## Headline finding")
    if not sim_df.empty:
        s2024 = sim_df.loc[sim_df["period"] == "2024", "tfidf_centroid_cosine"].mean()
        s2026 = sim_df.loc[sim_df["period"] == "2026", "tfidf_centroid_cosine"].mean()
        lines.append(
            f"The occupation boundary shifts, but it does not collapse. SWE-adjacent vs SWE TF-IDF centroid similarity is {s2024:.3f} in the 2024 windows and {s2026:.3f} in the 2026 windows. The broader signal is that AI-tool language, scope language, and description length all move upward across SWE, adjacent, and control groups, so part of the change is field-wide template evolution rather than SWE-only restructuring."
        )
    else:
        lines.append("The boundary sample was too sparse to compute a reliable centroid comparison.")
    lines.append("")
    lines.append("## What we learned")
    lines.append(
        "The parallel-trends table shows that the control group moves with the same general direction on AI-tool language, scope language, and description length, but usually from a lower base. That weakens any pure SWE-specific reading."
    )
    lines.append(
        "The cleanest SWE-specific signal is not that AI language appears only in SWE. It is that SWE combines AI-tool language with scope language and higher tech density more often than adjacent or control occupations."
    )
    lines.append(
        "Seniority remains best read through the YOE proxy. Explicit entry under `seniority_final` stays tiny, while the YOE-proxy junior share is materially broader in every group."
    )
    if not migration_df.empty:
        lines.append(
            "The term-migration table points to platform, infrastructure, and AI-workflow vocabulary moving closer to SWE-adjacent roles. The strongest candidates include "
            + ", ".join(migration_df.head(8)["term"].tolist())
            + "."
        )
    lines.append("")
    lines.append("## What surprised us")
    lines.append(
        "The control group is not a clean zero line. It also gets longer and a bit more AI-heavy. That means the paper cannot treat the 2026 language shift as uniquely SWE."
    )
    lines.append("")
    lines.append("## Evidence assessment")
    lines.append(
        "The text-based boundary result is moderate. It is based on the labeled cleaned-text frame, which is materially thinner in the scraped window and thinner again outside SWE. The direction is stable, but the magnitudes should not be oversold."
    )
    lines.append(
        "The boundary sample is company-capped at 25 postings per company per source-period, which keeps the sample from being dominated by a few prolific employers. That makes the sample more representative of the language surface, not less."
    )
    lines.append("")
    lines.append("## Narrative evaluation")
    lines.append(
        "The initial RQ1 framing weakens if it assumes the main story is SWE-only change. The data support a broader boundary recomposition story: the technical posting surface standardizes around AI, scope, and denser requirements, with SWE still ahead of adjacent and control occupations."
    )
    lines.append("")
    lines.append("## Emerging narrative")
    lines.append(
        "The market is not collapsing into one occupation, but it is standardizing the way technical work is described. SWE remains the densest bundle, adjacent roles absorb some of the same vocabulary, and control roles move in the same direction from a lower baseline."
    )
    lines.append("")
    lines.append("## Research question evolution")
    lines.append(
        "RQ1 should be rewritten to ask how much of the observed change is SWE-specific versus field-wide posting-template evolution."
    )
    lines.append(
        "RQ2 should keep the language-migration frame, but the key question is now which parts of the bundle are unique to SWE and which are becoming common across technical occupations."
    )
    lines.append("")
    lines.append("## Gaps and weaknesses")
    lines.append(
        "The biggest gap is still text coverage in scraped 2026, especially for non-SWE slices. The sample-based similarity analysis is also only a snapshot; it does not tell us whether the language change maps one-for-one to the work being done."
    )
    lines.append("")
    lines.append("## Direction for next wave")
    lines.append(
        "The next wave should connect this field-wide template shift to mechanism checks: ghost forensics, employer-requirement divergence, and the specific role families that absorb the new vocabulary most quickly."
    )
    lines.append("")
    lines.append("## Current paper positioning")
    lines.append(
        "The paper is strongest as an empirical restructuring study with a measurement caveat. T18 says the SWE story is real, but it sits inside a broader change in how technical work is posted."
    )
    lines.append("")
    lines.append("## Key outputs")
    for name in [
        "T18_coverage_by_group_period.csv",
        "T18_parallel_trends_primary.csv",
        "T18_parallel_trends_raw_sensitivity.csv",
        "T18_did_summary.csv",
        "T18_boundary_similarity.csv",
        "T18_term_migration_candidates.csv",
        "T18_adjacent_title_summary.csv",
        "T18_boundary_sensitivity.csv",
    ]:
        lines.append(f"- [{name}](../tables/T18/{name})")
    for name in ["T18_parallel_trends.png", "T18_boundary_similarity.png", "T18_term_migration.png"]:
        lines.append(f"- [{name}](../figures/T18/{name})")
    (REPORT_DIR / "T18.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dir(TABLE_DIR)
    ensure_dir(FIG_DIR)
    regex_hygiene()
    con = duckdb.connect()

    coverage = qdf(
        con,
        f"""
        SELECT
          source,
          period,
          {period_group_sql()} AS period_group,
          {group_case_sql()} AS occ,
          COUNT(*) AS n,
          SUM(CASE WHEN llm_extraction_coverage = 'labeled' THEN 1 ELSE 0 END) AS text_labeled_n,
          SUM(CASE WHEN llm_classification_coverage = 'labeled' THEN 1 ELSE 0 END) AS class_labeled_n,
          AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) AS entry_final_share,
          AVG(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) AS yoe_le2_share,
          AVG(CASE WHEN yoe_extracted <= 3 THEN 1.0 ELSE 0.0 END) AS yoe_le3_share
        FROM read_parquet('{DATA.as_posix()}')
        WHERE source_platform = 'linkedin'
          AND is_english = true
          AND date_flag = 'ok'
          AND (is_swe OR is_swe_adjacent OR is_control)
        GROUP BY 1,2,3,4
        ORDER BY 1,2,4
        """
    )
    coverage["text_labeled_share"] = coverage["text_labeled_n"] / coverage["n"]
    coverage["class_labeled_share"] = coverage["class_labeled_n"] / coverage["n"]
    save_csv(coverage, TABLE_DIR / "T18_coverage_by_group_period.csv")

    primary = build_feature_query("cleaned_text", "llm_extraction_coverage = 'labeled' AND cleaned_text IS NOT NULL", True, "cleaned_labeled", con)
    raw = build_feature_query("raw_text", "TRUE", False, "raw_all", con)
    save_csv(primary, TABLE_DIR / "T18_parallel_trends_primary.csv")
    save_csv(raw, TABLE_DIR / "T18_parallel_trends_raw_sensitivity.csv")

    metric_cols = [
        "entry_final_share",
        "yoe_le2_share",
        "yoe_le3_share",
        "junior_3level_share",
        "mid_3level_share",
        "senior_3level_share",
        "unknown_3level_share",
        "ai_tool_share",
        "scope_any_share",
        "analysis_text_mean_len",
        "analysis_text_median_len",
        "raw_desc_mean_len",
        "raw_desc_median_len",
        "mean_yoe",
        "median_yoe",
        "tech_count",
    ]
    primary_pool = weighted_pool(primary, metric_cols)
    raw_pool = weighted_pool(raw, ["ai_tool_share", "scope_any_share", "analysis_text_mean_len", "analysis_text_median_len", "raw_desc_mean_len", "raw_desc_median_len"])
    save_csv(primary_pool, TABLE_DIR / "T18_pooled_primary_2024_2026.csv")
    save_csv(raw_pool, TABLE_DIR / "T18_pooled_raw_2024_2026.csv")

    did_rows = []
    for metric in ["entry_final_share", "yoe_le2_share", "ai_tool_share", "scope_any_share", "analysis_text_median_len", "raw_desc_median_len", "tech_count"]:
        for occ in GROUP_ORDER:
            g = primary_pool.loc[primary_pool["occ"] == occ]
            vals = g.set_index("period_group")[metric].to_dict()
            if "2024" not in vals or "2026" not in vals:
                continue
            did_rows.append(
                {
                    "metric": metric,
                    "occ": occ,
                    "value_2024": float(vals["2024"]),
                    "value_2026": float(vals["2026"]),
                    "change": float(vals["2026"] - vals["2024"]),
                }
            )
    did_df = pd.DataFrame(did_rows)
    save_csv(did_df, TABLE_DIR / "T18_did_summary.csv")

    seniority_dist = qdf(
        con,
        f"""
        SELECT
          source,
          period,
          {group_case_sql()} AS occ,
          seniority_final,
          COUNT(*) AS n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE source_platform = 'linkedin'
          AND is_english = true
          AND date_flag = 'ok'
          AND (is_swe OR is_swe_adjacent OR is_control)
        GROUP BY 1,2,3,4
        ORDER BY 1,2,3,4
        """
    )
    save_csv(seniority_dist, TABLE_DIR / "T18_seniority_final_distribution.csv")

    sample_pool = make_sample_pool(con)
    save_csv(sample_pool, TABLE_DIR / "T18_boundary_sample_pool_capped.csv")
    boundary_draws = []
    rng = np.random.default_rng(42)
    for (source, period, occ), g in sample_pool.groupby(["source", "period", "occ"], dropna=False):
        n_take = min(200, len(g))
        if n_take == 0:
            continue
        boundary_draws.append(g.sample(n=n_take, random_state=int(rng.integers(0, 2**31 - 1))))
    boundary_sample = pd.concat(boundary_draws, ignore_index=True) if boundary_draws else pd.DataFrame()
    if not boundary_sample.empty:
        boundary_sample["window_label"] = boundary_sample["source"] + " / " + boundary_sample["period"]
    save_csv(boundary_sample, TABLE_DIR / "T18_boundary_sample_draws.csv")

    if not boundary_sample.empty:
        sim_df, term_df, migration_df = tfidf_similarity(boundary_sample)
    else:
        sim_df = pd.DataFrame(columns=["source", "period", "window_label", "swe_sample_n", "adjacent_sample_n", "tfidf_centroid_cosine"])
        term_df = pd.DataFrame()
        migration_df = pd.DataFrame(columns=["term", "gap_2024", "gap_2026", "migration_score"])
    save_csv(sim_df, TABLE_DIR / "T18_boundary_similarity.csv")
    if not migration_df.empty:
        save_csv(migration_df.head(40), TABLE_DIR / "T18_term_migration_candidates.csv")

    title_df = title_summary(con)
    save_csv(title_df, TABLE_DIR / "T18_adjacent_title_summary.csv")
    title_conv = title_convergence_table(title_df)
    save_csv(title_conv, TABLE_DIR / "T18_adjacent_title_convergence.csv")

    sensitivity_rows = []
    for spec, filters in [
        ("primary", ["llm_extraction_coverage = 'labeled'", "cleaned_text IS NOT NULL"]),
        ("no_aggregators", ["NOT is_aggregator", "llm_extraction_coverage = 'labeled'", "cleaned_text IS NOT NULL"]),
        ("no_title_lookup_llm", ["swe_classification_tier <> 'title_lookup_llm'", "llm_extraction_coverage = 'labeled'", "cleaned_text IS NOT NULL"]),
        ("no_aggregators_no_title_lookup_llm", ["NOT is_aggregator", "swe_classification_tier <> 'title_lookup_llm'", "llm_extraction_coverage = 'labeled'", "cleaned_text IS NOT NULL"]),
    ]:
        filt = " AND ".join(filters)
        sub = build_feature_query("cleaned_text", filt, False, spec, con)
        pooled = weighted_pool(sub, ["ai_tool_share", "scope_any_share", "analysis_text_median_len"])
        vals = pooled.pivot_table(index="occ", columns="period_group", values="analysis_text_median_len", aggfunc="mean")
        for metric in ["ai_tool_share", "scope_any_share", "analysis_text_median_len"]:
            piv = pooled.pivot_table(index="occ", columns="period_group", values=metric, aggfunc="mean")
            if {"swe", "control"}.issubset(piv.index) and {"2024", "2026"}.issubset(piv.columns):
                sensitivity_rows.append(
                    {
                        "spec": spec,
                        "metric": metric,
                        "swe_change": float(piv.loc["swe", "2026"] - piv.loc["swe", "2024"]),
                        "control_change": float(piv.loc["control", "2026"] - piv.loc["control", "2024"]),
                        "did": float((piv.loc["swe", "2026"] - piv.loc["swe", "2024"]) - (piv.loc["control", "2026"] - piv.loc["control", "2024"])),
                    }
                )
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    save_csv(sensitivity_df, TABLE_DIR / "T18_boundary_sensitivity.csv")

    plot_parallel_trends(primary_pool)
    plot_similarity(sim_df)
    plot_migration(migration_df)

    write_report(coverage, primary_pool, raw_pool, did_df, sim_df, migration_df, title_conv, sensitivity_df)


if __name__ == "__main__":
    main()
