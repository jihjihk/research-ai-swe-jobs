#!/usr/bin/env python3
"""T16 company hiring strategy typology.

Memory posture:
- DuckDB projects only derived row features for the default LinkedIn SWE frame.
- Company/metro-sized aggregates are the only objects materialized in pandas.
- LLM-text metrics are kept separate from raw/binary metrics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
SHARED = ROOT / "exploration" / "artifacts" / "shared"
TECH = SHARED / "swe_tech_matrix.parquet"
TAXONOMY = SHARED / "tech_taxonomy.csv"
ARCHETYPES = SHARED / "swe_archetype_labels.parquet"
SENIORITY_PANEL = SHARED / "seniority_definition_panel.csv"
COMPLEXITY = ROOT / "exploration" / "tables" / "T11" / "posting_complexity_features.parquet"
TABLE_DIR = ROOT / "exploration" / "tables" / "T16"
FIG_DIR = ROOT / "exploration" / "figures" / "T16"
SUMMARY_PATH = TABLE_DIR / "summary.json"

BROAD_AI_COLS = [
    "agents",
    "anthropic_api",
    "chatgpt",
    "claude",
    "claude_api",
    "codex",
    "copilot",
    "cursor",
    "evals",
    "fine_tuning",
    "gemini",
    "generative_ai",
    "llm",
    "mcp",
    "openai_api",
    "prompt_engineering",
    "rag",
    "langchain",
    "llamaindex",
    "hugging_face",
    "chroma",
    "pinecone",
    "weaviate",
    "vector_databases",
    "machine_learning",
    "deep_learning",
    "nlp",
    "computer_vision",
    "mlops",
    "pytorch",
    "tensorflow",
]

METRIC_DENOMS = {
    "j1_entry_share": "postings",
    "j2_entry_associate_share": "postings",
    "j3_low_yoe_share": "yoe_known_n",
    "j4_low_yoe_share": "yoe_known_n",
    "broad_ai_share": "postings",
    "ai_tool_strict_share": "postings",
    "raw_description_length_mean": "description_nonnull_n",
    "tech_count_mean": "postings",
    "platform_tech_count_mean": "postings",
    "org_scope_count_mean_llm": "llm_text_n",
    "requirement_breadth_mean_llm": "llm_text_n",
    "label_yoe_divergence": "yoe_known_n",
}

PRIMARY_DECOMP_METRICS = [
    "j1_entry_share",
    "j3_low_yoe_share",
    "broad_ai_share",
    "ai_tool_strict_share",
    "raw_description_length_mean",
    "requirement_breadth_mean_llm",
    "tech_count_mean",
    "platform_tech_count_mean",
]

CLUSTER_FEATURES = [
    "delta_j1_entry_share",
    "delta_j3_low_yoe_share",
    "delta_broad_ai_share",
    "delta_ai_tool_strict_share",
    "delta_raw_description_length_mean",
    "delta_tech_count_mean",
    "delta_platform_tech_count_mean",
    "delta_org_scope_count_mean_llm",
    "delta_requirement_breadth_mean_llm",
    "delta_label_yoe_divergence",
]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "exploration" / "artifacts" / "tmp_duckdb").mkdir(parents=True, exist_ok=True)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=1")
    con.execute("PRAGMA temp_directory='exploration/artifacts/tmp_duckdb'")
    return con


def sum_expr(cols: Iterable[str], prefix: str = "tm") -> str:
    parts = [f"COALESCE({prefix}.{c}::INT, 0)" for c in cols]
    return " + ".join(parts) if parts else "0"


def any_expr(cols: Iterable[str], prefix: str = "tm") -> str:
    return f"(({sum_expr(cols, prefix)}) > 0)"


def load_taxonomy() -> tuple[list[str], list[str], list[str], list[str]]:
    tax = pd.read_csv(TAXONOMY)
    tech_cols = tax["column"].tolist()
    broad_ai = [c for c in BROAD_AI_COLS if c in tech_cols]
    ai_tool_strict = tax.loc[
        tax["category"].eq("ai_tool") & ~tax["column"].isin(["mcp"]), "column"
    ].tolist()
    platform_cols = tax.loc[
        tax["category"].isin(["cloud", "devops", "ops", "architecture", "tooling"]),
        "column",
    ].tolist()
    return tech_cols, broad_ai, ai_tool_strict, platform_cols


def create_row_view(con: duckdb.DuckDBPyConnection) -> None:
    tech_cols, broad_ai_cols, ai_tool_cols, platform_cols = load_taxonomy()
    if not COMPLEXITY.exists():
        raise FileNotFoundError(
            f"Required T11 feature artifact not found: {COMPLEXITY}. "
            "T16 uses it for validated org-scope and requirement-breadth metrics."
        )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW row_features AS
        SELECT
            u.uid,
            u.source,
            CASE
                WHEN u.source = 'kaggle_arshkon' THEN 'arshkon'
                WHEN u.source = 'kaggle_asaniczka' THEN 'asaniczka'
                WHEN u.source = 'scraped' THEN 'scraped_2026'
                ELSE u.source
            END AS source_group,
            CASE
                WHEN u.source IN ('kaggle_arshkon', 'kaggle_asaniczka') THEN '2024'
                WHEN u.source = 'scraped' THEN '2026'
                ELSE u.period
            END AS period_group,
            u.period,
            NULLIF(u.company_name_canonical, '') AS company_name_canonical,
            u.company_industry,
            COALESCE(u.is_aggregator, false) AS is_aggregator,
            u.seniority_final,
            u.seniority_3level,
            u.seniority_final_source,
            u.yoe_extracted,
            u.description_length,
            u.swe_classification_tier,
            {sum_expr(tech_cols)}::DOUBLE AS tech_count,
            {sum_expr(platform_cols)}::DOUBLE AS platform_tech_count,
            {any_expr(broad_ai_cols)} AS broad_ai_any,
            {any_expr(ai_tool_cols)} AS ai_tool_strict_any,
            cf.text_source,
            cf.char_len AS cleaned_char_len,
            cf.org_scope_count,
            cf.requirement_breadth,
            ar.archetype_name
        FROM read_parquet('{DATA.as_posix()}') AS u
        JOIN read_parquet('{TECH.as_posix()}') AS tm USING (uid)
        LEFT JOIN read_parquet('{COMPLEXITY.as_posix()}') AS cf USING (uid)
        LEFT JOIN read_parquet('{ARCHETYPES.as_posix()}') AS ar USING (uid)
        WHERE u.source_platform = 'linkedin'
          AND u.is_english = true
          AND u.date_flag = 'ok'
          AND u.is_swe = true
        """
    )


def company_metrics_sql(group_expr: str, where_extra: str = "TRUE") -> str:
    return f"""
        SELECT
            company_name_canonical,
            {group_expr} AS group_key,
            COUNT(*)::INTEGER AS postings,
            SUM(is_aggregator::INT)::INTEGER AS aggregator_postings,
            AVG(is_aggregator::INT)::DOUBLE AS aggregator_share,
            SUM(CASE WHEN description_length IS NOT NULL THEN 1 ELSE 0 END)::INTEGER AS description_nonnull_n,
            SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END)::INTEGER AS yoe_known_n,
            SUM(CASE WHEN text_source = 'llm' THEN 1 ELSE 0 END)::INTEGER AS llm_text_n,
            SUM(CASE WHEN archetype_name IS NOT NULL THEN 1 ELSE 0 END)::INTEGER AS archetype_labeled_n,
            AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) AS j1_entry_share,
            AVG(CASE WHEN seniority_final IN ('entry', 'associate') THEN 1.0 ELSE 0.0 END) AS j2_entry_associate_share,
            SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END)::DOUBLE
              / NULLIF(SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END), 0) AS j3_low_yoe_share,
            SUM(CASE WHEN yoe_extracted <= 3 THEN 1 ELSE 0 END)::DOUBLE
              / NULLIF(SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END), 0) AS j4_low_yoe_share,
            AVG(broad_ai_any::INT)::DOUBLE AS broad_ai_share,
            AVG(ai_tool_strict_any::INT)::DOUBLE AS ai_tool_strict_share,
            AVG(description_length)::DOUBLE AS raw_description_length_mean,
            AVG(tech_count)::DOUBLE AS tech_count_mean,
            AVG(platform_tech_count)::DOUBLE AS platform_tech_count_mean,
            AVG(CASE WHEN text_source = 'llm' THEN cleaned_char_len END)::DOUBLE AS cleaned_length_mean_llm,
            AVG(CASE WHEN text_source = 'llm' THEN org_scope_count END)::DOUBLE AS org_scope_count_mean_llm,
            AVG(CASE WHEN text_source = 'llm' THEN requirement_breadth END)::DOUBLE AS requirement_breadth_mean_llm,
            AVG(CASE WHEN seniority_final = 'unknown' THEN 1.0 ELSE 0.0 END) AS unknown_seniority_share,
            AVG(CASE WHEN seniority_final = 'entry' AND yoe_extracted > 3 THEN 1.0 ELSE 0.0 END) AS entry_high_yoe_share_all
        FROM row_features
        WHERE company_name_canonical IS NOT NULL
          AND {where_extra}
        GROUP BY 1, 2
    """


def load_company_metrics(
    con: duckdb.DuckDBPyConnection, group_expr: str, where_extra: str = "TRUE"
) -> pd.DataFrame:
    out = con.execute(company_metrics_sql(group_expr, where_extra)).fetchdf()
    out["label_yoe_divergence"] = out["j3_low_yoe_share"] - out["j1_entry_share"]
    return out


def common_companies(metrics: pd.DataFrame, left: str, right: str, min_postings: int = 3) -> set[str]:
    wide = metrics.pivot_table(
        index="company_name_canonical", columns="group_key", values="postings", fill_value=0
    )
    if left not in wide or right not in wide:
        return set()
    return set(wide[(wide[left] >= min_postings) & (wide[right] >= min_postings)].index)


def flatten_columns(cols: pd.Index) -> list[str]:
    flattened: list[str] = []
    for col in cols:
        if isinstance(col, tuple):
            flattened.append("_".join(str(c) for c in col if c != ""))
        else:
            flattened.append(str(col))
    return flattened


def make_change_features(
    metrics: pd.DataFrame, left: str, right: str, companies: set[str], suffix: str
) -> pd.DataFrame:
    keep_metrics = list(dict.fromkeys([
        "postings",
        "yoe_known_n",
        "llm_text_n",
        "archetype_labeled_n",
        "aggregator_share",
        *METRIC_DENOMS.keys(),
        "j4_low_yoe_share",
        "j2_entry_associate_share",
        "unknown_seniority_share",
        "cleaned_length_mean_llm",
        "entry_high_yoe_share_all",
    ]))
    wide = (
        metrics[metrics["company_name_canonical"].isin(companies)]
        .pivot(index="company_name_canonical", columns="group_key", values=keep_metrics)
        .reset_index()
    )
    wide.columns = flatten_columns(wide.columns)
    rows = wide.copy()
    for metric in keep_metrics:
        lcol = f"{metric}_{left}"
        rcol = f"{metric}_{right}"
        if lcol in rows and rcol in rows:
            rows[f"delta_{metric}"] = rows[rcol] - rows[lcol]
    rows["comparison_spec"] = suffix
    return rows


def midpoint_decomposition(
    metrics: pd.DataFrame,
    left: str,
    right: str,
    companies: set[str],
    metric: str,
    spec_name: str,
    exclude_aggregators: bool,
) -> dict[str, object]:
    denom_col = METRIC_DENOMS[metric]
    work = metrics[metrics["company_name_canonical"].isin(companies)].copy()
    val = work.pivot(index="company_name_canonical", columns="group_key", values=metric)
    den = work.pivot(index="company_name_canonical", columns="group_key", values=denom_col)
    needed = [left, right]
    if any(col not in val or col not in den for col in needed):
        return {
            "spec": spec_name,
            "metric": metric,
            "aggregator_excluded": exclude_aggregators,
            "companies_used": 0,
        }
    tmp = pd.DataFrame(
        {
            "y0": val[left],
            "y1": val[right],
            "n0": den[left],
            "n1": den[right],
        }
    ).dropna()
    tmp = tmp[(tmp["n0"] > 0) & (tmp["n1"] > 0)]
    if tmp.empty:
        return {
            "spec": spec_name,
            "metric": metric,
            "aggregator_excluded": exclude_aggregators,
            "companies_used": 0,
        }
    w0 = tmp["n0"] / tmp["n0"].sum()
    w1 = tmp["n1"] / tmp["n1"].sum()
    y0 = tmp["y0"]
    y1 = tmp["y1"]
    total0 = float((w0 * y0).sum())
    total1 = float((w1 * y1).sum())
    wbar = (w0 + w1) / 2
    ybar = (y0 + y1) / 2
    within = float((wbar * (y1 - y0)).sum())
    between = float(((w1 - w0) * ybar).sum())
    total = total1 - total0
    return {
        "spec": spec_name,
        "metric": metric,
        "aggregator_excluded": exclude_aggregators,
        "companies_in_pool": len(companies),
        "companies_used": int(len(tmp)),
        "left_group": left,
        "right_group": right,
        "left_weighted_value": total0,
        "right_weighted_value": total1,
        "total_change": total,
        "within_company_component": within,
        "between_company_reweighting_component": between,
        "residual": total - within - between,
        "within_share_of_abs_components": abs(within) / (abs(within) + abs(between))
        if (abs(within) + abs(between)) > 0
        else np.nan,
    }


def run_decompositions(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    specs = [
        ("arshkon_only", "source_group", "arshkon", "scraped_2026"),
        ("pooled_2024", "period_group", "2024", "2026"),
    ]
    for spec_name, group_expr, left, right in specs:
        for exclude in [False, True]:
            where = "NOT is_aggregator" if exclude else "TRUE"
            metrics = load_company_metrics(con, group_expr, where)
            companies = common_companies(metrics, left, right, min_postings=3)
            for metric in PRIMARY_DECOMP_METRICS:
                rows.append(
                    midpoint_decomposition(metrics, left, right, companies, metric, spec_name, exclude)
                )
            for definition, metric in [
                ("J1", "j1_entry_share"),
                ("J2", "j2_entry_associate_share"),
                ("J3", "j3_low_yoe_share"),
                ("J4", "j4_low_yoe_share"),
            ]:
                result = midpoint_decomposition(metrics, left, right, companies, metric, spec_name, exclude)
                result["definition"] = definition
                result["agreement_verdict"] = (
                    "up" if result.get("total_change", np.nan) > 0 else "down"
                    if result.get("total_change", np.nan) < 0
                    else "flat"
                )
                panel_rows.append(result)
    decomp = pd.DataFrame(rows)
    panel = pd.DataFrame(panel_rows)
    decomp.to_csv(TABLE_DIR / "within_between_decomposition_all_specs.csv", index=False)
    panel.to_csv(TABLE_DIR / "seniority_panel_company_overlap.csv", index=False)
    decomp[(decomp["spec"] == "arshkon_only") & (~decomp["aggregator_excluded"])].to_csv(
        TABLE_DIR / "within_between_decomposition_arshkon_only.csv", index=False
    )
    decomp[(decomp["spec"] == "pooled_2024") & (~decomp["aggregator_excluded"])].to_csv(
        TABLE_DIR / "within_between_decomposition_pooled_2024.csv", index=False
    )
    return decomp, panel


def cluster_trajectories(change: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = change.copy()
    X = work[CLUSTER_FEATURES].copy()
    missing_rates = X.isna().mean().rename("missing_rate").reset_index()
    missing_rates.columns = ["feature", "missing_rate"]
    n = len(work)
    k_candidates = [k for k in range(3, min(7, n)) if k < n]
    sil_rows: list[dict[str, object]] = []
    best_k = 5 if n >= 80 else max(3, min(4, n - 1))
    best_score = -np.inf
    for k in k_candidates:
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        X_scaled = pipe.fit_transform(X)
        labels = KMeans(n_clusters=k, random_state=42, n_init=50).fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan
        sil_rows.append({"k": k, "silhouette": score})
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_k = k
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    X_scaled = pipe.fit_transform(X)
    model = KMeans(n_clusters=best_k, random_state=42, n_init=50)
    labels = model.fit_predict(X_scaled)
    work["cluster_id"] = labels

    summary = (
        work.groupby("cluster_id")
        .agg(
            companies=("company_name_canonical", "size"),
            median_2024_postings=("postings_arshkon", "median"),
            median_2026_postings=("postings_scraped_2026", "median"),
            **{f"mean_{f}": (f, "mean") for f in CLUSTER_FEATURES},
        )
        .reset_index()
    )

    def name_cluster(row: pd.Series) -> str:
        if row["mean_delta_j3_low_yoe_share"] <= -0.25:
            return "low-YOE retreat with AI broadening"
        if row["mean_delta_j3_low_yoe_share"] >= 0.30 and row["mean_delta_j1_entry_share"] < -0.05:
            return "label/YOE divergence spike"
        if row["mean_delta_ai_tool_strict_share"] >= 0.30:
            return "AI-heavy scope expanders"
        if row["mean_delta_requirement_breadth_mean_llm"] >= 3.0 and row["mean_delta_tech_count_mean"] >= 3.0:
            return "stack-breadth expanders"
        if row["mean_delta_j3_low_yoe_share"] >= 0.06 and row["mean_delta_tech_count_mean"] < 0:
            return "low-YOE growth, tool-light shift"
        if row["mean_delta_requirement_breadth_mean_llm"] >= 2.0 and row["mean_delta_org_scope_count_mean_llm"] >= 0.20:
            return "requirement and scope broadening"
        if row["mean_delta_j3_low_yoe_share"] >= 0.08 or row["mean_delta_j1_entry_share"] >= 0.04:
            return "entry/low-YOE expanding"
        if row["mean_delta_j1_entry_share"] < -0.02 and row["mean_delta_j3_low_yoe_share"] < -0.02:
            return "junior reducing"
        if row["mean_delta_raw_description_length_mean"] < -100 and row["mean_delta_tech_count_mean"] < 0.5:
            return "shorter/tool-light"
        return "mixed incumbent"

    summary["trajectory_name"] = summary.apply(name_cluster, axis=1)
    work = work.merge(summary[["cluster_id", "trajectory_name"]], on="cluster_id", how="left")
    top = (
        work.sort_values("postings_scraped_2026", ascending=False)
        .groupby("cluster_id")
        .head(8)[
            [
                "cluster_id",
                "trajectory_name",
                "company_name_canonical",
                "postings_arshkon",
                "postings_scraped_2026",
                *CLUSTER_FEATURES,
            ]
        ]
    )
    missing_rates.to_csv(TABLE_DIR / "cluster_feature_missingness.csv", index=False)
    pd.DataFrame(sil_rows).to_csv(TABLE_DIR / "cluster_silhouette_candidates.csv", index=False)
    work.to_csv(TABLE_DIR / "company_change_features.csv", index=False)
    summary.to_csv(TABLE_DIR / "company_cluster_summary.csv", index=False)
    top.to_csv(TABLE_DIR / "cluster_top_companies.csv", index=False)
    return work, summary, pd.DataFrame(sil_rows)


def mode_nonnull(series: pd.Series) -> str | None:
    vals = series.dropna()
    if vals.empty:
        return None
    return str(vals.value_counts().index[0])


def new_entrant_tables(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profile = con.execute(
        """
        WITH hist AS (
            SELECT DISTINCT company_name_canonical
            FROM row_features
            WHERE period_group = '2024' AND company_name_canonical IS NOT NULL
        ),
        tagged AS (
            SELECT
                r.*,
                NOT EXISTS (
                    SELECT 1 FROM hist h
                    WHERE h.company_name_canonical = r.company_name_canonical
                ) AS new_entrant_vs_2024
            FROM row_features r
            WHERE r.source_group = 'scraped_2026'
              AND r.company_name_canonical IS NOT NULL
        )
        SELECT
            new_entrant_vs_2024,
            COUNT(DISTINCT company_name_canonical)::INTEGER AS companies,
            COUNT(*)::INTEGER AS postings,
            AVG(is_aggregator::INT)::DOUBLE AS aggregator_share,
            SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END)::INTEGER AS yoe_known_n,
            SUM(CASE WHEN text_source = 'llm' THEN 1 ELSE 0 END)::INTEGER AS llm_text_n,
            AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) AS j1_entry_share,
            SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END)::DOUBLE
              / NULLIF(SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END), 0) AS j3_low_yoe_share,
            AVG(CASE WHEN seniority_final = 'unknown' THEN 1.0 ELSE 0.0 END) AS unknown_seniority_share,
            AVG(broad_ai_any::INT)::DOUBLE AS broad_ai_share,
            AVG(ai_tool_strict_any::INT)::DOUBLE AS ai_tool_strict_share,
            AVG(tech_count)::DOUBLE AS tech_count_mean,
            AVG(platform_tech_count)::DOUBLE AS platform_tech_count_mean,
            AVG(CASE WHEN text_source = 'llm' THEN requirement_breadth END)::DOUBLE AS requirement_breadth_mean_llm,
            AVG(CASE WHEN text_source = 'llm' THEN org_scope_count END)::DOUBLE AS org_scope_count_mean_llm
        FROM tagged
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    total_scraped = profile["postings"].sum()
    profile["share_of_scraped_postings"] = profile["postings"] / total_scraped
    profile.to_csv(TABLE_DIR / "new_entrant_profile.csv", index=False)

    top = con.execute(
        """
        WITH hist AS (
            SELECT DISTINCT company_name_canonical
            FROM row_features
            WHERE period_group = '2024' AND company_name_canonical IS NOT NULL
        ),
        tagged AS (
            SELECT r.*
            FROM row_features r
            WHERE r.source_group = 'scraped_2026'
              AND r.company_name_canonical IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM hist h
                  WHERE h.company_name_canonical = r.company_name_canonical
              )
        ),
        company AS (
            SELECT
                company_name_canonical,
                COUNT(*)::INTEGER AS postings,
                MAX(company_industry) AS industry_sample,
                AVG(is_aggregator::INT)::DOUBLE AS aggregator_share,
                AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) AS j1_entry_share,
                SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END)::DOUBLE
                  / NULLIF(SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END), 0) AS j3_low_yoe_share,
                AVG(broad_ai_any::INT)::DOUBLE AS broad_ai_share,
                AVG(ai_tool_strict_any::INT)::DOUBLE AS ai_tool_strict_share,
                AVG(tech_count)::DOUBLE AS tech_count_mean,
                AVG(platform_tech_count)::DOUBLE AS platform_tech_count_mean,
                SUM(CASE WHEN text_source = 'llm' THEN 1 ELSE 0 END)::INTEGER AS llm_text_n,
                AVG(CASE WHEN text_source = 'llm' THEN requirement_breadth END)::DOUBLE AS requirement_breadth_mean_llm
            FROM tagged
            GROUP BY 1
        ),
        arch AS (
            SELECT company_name_canonical, archetype_name, COUNT(*) AS n,
                   ROW_NUMBER() OVER (
                       PARTITION BY company_name_canonical
                       ORDER BY COUNT(*) DESC, archetype_name
                   ) AS rn
            FROM tagged
            WHERE archetype_name IS NOT NULL
            GROUP BY 1, 2
        )
        SELECT c.*, a.archetype_name AS top_archetype, a.n AS top_archetype_labeled_n
        FROM company c
        LEFT JOIN arch a
          ON c.company_name_canonical = a.company_name_canonical
         AND a.rn = 1
        ORDER BY c.postings DESC, c.company_name_canonical
        LIMIT 75
        """
    ).fetchdf()
    top.to_csv(TABLE_DIR / "new_entrant_top_employers.csv", index=False)

    industries = con.execute(
        """
        WITH hist AS (
            SELECT DISTINCT company_name_canonical
            FROM row_features
            WHERE period_group = '2024' AND company_name_canonical IS NOT NULL
        )
        SELECT
            COALESCE(company_industry, 'Unknown') AS company_industry,
            COUNT(*)::INTEGER AS postings,
            COUNT(DISTINCT company_name_canonical)::INTEGER AS companies,
            AVG(broad_ai_any::INT)::DOUBLE AS broad_ai_share,
            AVG(ai_tool_strict_any::INT)::DOUBLE AS ai_tool_strict_share,
            AVG(tech_count)::DOUBLE AS tech_count_mean
        FROM row_features r
        WHERE r.source_group = 'scraped_2026'
          AND r.company_name_canonical IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM hist h
              WHERE h.company_name_canonical = r.company_name_canonical
          )
        GROUP BY 1
        ORDER BY postings DESC
        LIMIT 40
        """
    ).fetchdf()
    industries.to_csv(TABLE_DIR / "new_entrant_industry_profile.csv", index=False)
    return profile, top, industries


def aggregator_profile(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    out = con.execute(
        """
        SELECT
            source_group,
            period_group,
            is_aggregator,
            COUNT(*)::INTEGER AS postings,
            COUNT(DISTINCT company_name_canonical)::INTEGER AS companies,
            SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END)::INTEGER AS yoe_known_n,
            SUM(CASE WHEN text_source = 'llm' THEN 1 ELSE 0 END)::INTEGER AS llm_text_n,
            AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) AS j1_entry_share,
            SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END)::DOUBLE
              / NULLIF(SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END), 0) AS j3_low_yoe_share,
            AVG(broad_ai_any::INT)::DOUBLE AS broad_ai_share,
            AVG(ai_tool_strict_any::INT)::DOUBLE AS ai_tool_strict_share,
            AVG(description_length)::DOUBLE AS raw_description_length_mean,
            AVG(tech_count)::DOUBLE AS tech_count_mean,
            AVG(platform_tech_count)::DOUBLE AS platform_tech_count_mean,
            AVG(CASE WHEN text_source = 'llm' THEN requirement_breadth END)::DOUBLE AS requirement_breadth_mean_llm
        FROM row_features
        GROUP BY 1, 2, 3
        ORDER BY 1, 3
        """
    ).fetchdf()
    out.to_csv(TABLE_DIR / "aggregator_direct_profile.csv", index=False)
    return out


def domain_tables(con: duckdb.DuckDBPyConnection, common_arsh: set[str]) -> pd.DataFrame:
    if not common_arsh:
        out = pd.DataFrame()
        out.to_csv(TABLE_DIR / "domain_metrics_overlap_archetype.csv", index=False)
        return out
    company_values = ",".join("'" + c.replace("'", "''") + "'" for c in sorted(common_arsh))
    out = con.execute(
        f"""
        SELECT
            source_group,
            archetype_name,
            COUNT(*)::INTEGER AS labeled_n,
            COUNT(DISTINCT company_name_canonical)::INTEGER AS companies,
            AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) AS j1_entry_share,
            SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END)::DOUBLE
              / NULLIF(SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END), 0) AS j3_low_yoe_share,
            AVG(broad_ai_any::INT)::DOUBLE AS broad_ai_share,
            AVG(ai_tool_strict_any::INT)::DOUBLE AS ai_tool_strict_share,
            AVG(requirement_breadth)::DOUBLE AS requirement_breadth_mean_llm,
            AVG(tech_count)::DOUBLE AS tech_count_mean
        FROM row_features
        WHERE company_name_canonical IN ({company_values})
          AND source_group IN ('arshkon', 'scraped_2026')
          AND archetype_name IS NOT NULL
          AND text_source = 'llm'
        GROUP BY 1, 2
        ORDER BY archetype_name, source_group
        """
    ).fetchdf()
    out.to_csv(TABLE_DIR / "domain_metrics_overlap_archetype.csv", index=False)

    changes = []
    for arch, sub in out.groupby("archetype_name", dropna=False):
        wide = sub.set_index("source_group")
        if {"arshkon", "scraped_2026"}.issubset(wide.index):
            row = {"archetype_name": arch}
            for metric in [
                "j1_entry_share",
                "j3_low_yoe_share",
                "broad_ai_share",
                "ai_tool_strict_share",
                "requirement_breadth_mean_llm",
                "tech_count_mean",
            ]:
                row[f"delta_{metric}"] = wide.loc["scraped_2026", metric] - wide.loc["arshkon", metric]
                row[f"arshkon_{metric}"] = wide.loc["arshkon", metric]
                row[f"scraped_2026_{metric}"] = wide.loc["scraped_2026", metric]
            row["arshkon_labeled_n"] = int(wide.loc["arshkon", "labeled_n"])
            row["scraped_2026_labeled_n"] = int(wide.loc["scraped_2026", "labeled_n"])
            changes.append(row)
    changes_df = pd.DataFrame(changes).sort_values("delta_broad_ai_share", ascending=False)
    changes_df.to_csv(TABLE_DIR / "domain_metric_changes_overlap_archetype.csv", index=False)
    return out


def sample_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    out = con.execute(
        """
        SELECT
            source_group,
            period_group,
            COUNT(*)::INTEGER AS eligible_n,
            COUNT(DISTINCT company_name_canonical)::INTEGER AS companies,
            SUM(CASE WHEN company_name_canonical IS NULL THEN 1 ELSE 0 END)::INTEGER AS missing_company_n,
            SUM(is_aggregator::INT)::INTEGER AS aggregator_n,
            SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END)::INTEGER AS yoe_known_n,
            SUM(CASE WHEN text_source = 'llm' THEN 1 ELSE 0 END)::INTEGER AS llm_labeled_text_n,
            SUM(CASE WHEN archetype_name IS NOT NULL THEN 1 ELSE 0 END)::INTEGER AS archetype_labeled_n,
            AVG(broad_ai_any::INT)::DOUBLE AS broad_ai_share,
            AVG(ai_tool_strict_any::INT)::DOUBLE AS ai_tool_strict_share,
            AVG(tech_count)::DOUBLE AS tech_count_mean
        FROM row_features
        GROUP BY 1, 2
        ORDER BY 1
        """
    ).fetchdf()
    out["llm_labeled_text_share"] = out["llm_labeled_text_n"] / out["eligible_n"]
    out.to_csv(TABLE_DIR / "sample_counts.csv", index=False)
    return out


def overall_change_specs(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    specs = [
        ("primary_pooled", "TRUE", "period_group"),
        ("aggregator_excluded_pooled", "NOT is_aggregator", "period_group"),
        (
            "swe_tier_exclude_title_lookup_pooled",
            "swe_classification_tier <> 'title_lookup_llm'",
            "period_group",
        ),
        ("primary_arshkon", "source_group IN ('arshkon', 'scraped_2026')", "source_group"),
        (
            "aggregator_excluded_arshkon",
            "source_group IN ('arshkon', 'scraped_2026') AND NOT is_aggregator",
            "source_group",
        ),
    ]
    rows = []
    for spec, where, group_col in specs:
        label0, label1 = ("2024", "2026") if group_col == "period_group" else ("arshkon", "scraped_2026")
        query = f"""
            SELECT
                {group_col} AS group_key,
                COUNT(*) AS n,
                SUM(CASE WHEN text_source = 'llm' THEN 1 ELSE 0 END) AS llm_text_n,
                AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) AS j1_entry_share,
                SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END)::DOUBLE
                  / NULLIF(SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END), 0) AS j3_low_yoe_share,
                AVG(broad_ai_any::INT)::DOUBLE AS broad_ai_share,
                AVG(ai_tool_strict_any::INT)::DOUBLE AS ai_tool_strict_share,
                AVG(description_length)::DOUBLE AS raw_description_length_mean,
                AVG(tech_count)::DOUBLE AS tech_count_mean,
                AVG(platform_tech_count)::DOUBLE AS platform_tech_count_mean,
                AVG(CASE WHEN text_source = 'llm' THEN requirement_breadth END)::DOUBLE AS requirement_breadth_mean_llm,
                AVG(CASE WHEN text_source = 'llm' THEN org_scope_count END)::DOUBLE AS org_scope_count_mean_llm
            FROM row_features
            WHERE {where}
            GROUP BY 1
        """
        data = con.execute(query).fetchdf().set_index("group_key")
        if label0 not in data.index or label1 not in data.index:
            continue
        for metric in [
            "j1_entry_share",
            "j3_low_yoe_share",
            "broad_ai_share",
            "ai_tool_strict_share",
            "raw_description_length_mean",
            "tech_count_mean",
            "platform_tech_count_mean",
            "requirement_breadth_mean_llm",
            "org_scope_count_mean_llm",
        ]:
            rows.append(
                {
                    "spec": spec,
                    "metric": metric,
                    "left_group": label0,
                    "right_group": label1,
                    "left_value": data.loc[label0, metric],
                    "right_value": data.loc[label1, metric],
                    "delta": data.loc[label1, metric] - data.loc[label0, metric],
                    "left_n": data.loc[label0, "n"],
                    "right_n": data.loc[label1, "n"],
                    "left_llm_text_n": data.loc[label0, "llm_text_n"],
                    "right_llm_text_n": data.loc[label1, "llm_text_n"],
                }
            )
    out = pd.DataFrame(rows)
    primary = out[out["spec"].eq("primary_pooled")][["metric", "delta"]].rename(
        columns={"delta": "primary_pooled_delta"}
    )
    out = out.merge(primary, on="metric", how="left")
    out["effect_size_change_vs_primary"] = (
        (out["delta"] - out["primary_pooled_delta"]).abs()
        / out["primary_pooled_delta"].abs().replace(0, np.nan)
    )
    out["direction_flip_vs_primary"] = np.sign(out["delta"]) != np.sign(out["primary_pooled_delta"])
    out["material_sensitivity_vs_primary"] = (
        out["effect_size_change_vs_primary"].gt(0.30) | out["direction_flip_vs_primary"]
    )
    out.to_csv(TABLE_DIR / "overall_change_sensitivity_specs.csv", index=False)
    return out


def write_plots(cluster_summary: pd.DataFrame, decomp: pd.DataFrame, entrant_profile: pd.DataFrame) -> None:
    if not cluster_summary.empty:
        heat_cols = [
            "mean_delta_j1_entry_share",
            "mean_delta_j3_low_yoe_share",
            "mean_delta_broad_ai_share",
            "mean_delta_ai_tool_strict_share",
            "mean_delta_tech_count_mean",
            "mean_delta_requirement_breadth_mean_llm",
        ]
        plot = cluster_summary.set_index("trajectory_name")[heat_cols].copy()
        fig, ax = plt.subplots(figsize=(11, 4.8))
        im = ax.imshow(plot.to_numpy(), aspect="auto", cmap="RdBu_r")
        ax.set_yticks(range(len(plot.index)))
        ax.set_yticklabels(plot.index, fontsize=8)
        ax.set_xticks(range(len(heat_cols)))
        ax.set_xticklabels([c.replace("mean_delta_", "").replace("_", "\n") for c in heat_cols], fontsize=8)
        ax.set_title("T16 company trajectory cluster mean deltas")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "company_cluster_delta_heatmap.png", dpi=150)
        plt.close(fig)

    primary = decomp[(decomp["spec"].eq("arshkon_only")) & (~decomp["aggregator_excluded"])].copy()
    primary = primary[primary["metric"].isin(PRIMARY_DECOMP_METRICS)]
    if not primary.empty:
        fig, ax = plt.subplots(figsize=(11, 4.8))
        x = np.arange(len(primary))
        ax.bar(x - 0.18, primary["within_company_component"], width=0.36, label="within company")
        ax.bar(x + 0.18, primary["between_company_reweighting_component"], width=0.36, label="between/reweighting")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(primary["metric"].str.replace("_", "\n"), fontsize=8)
        ax.set_title("T16 common-company decomposition, arshkon to scraped")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "common_company_decomposition.png", dpi=150)
        plt.close(fig)

    if not entrant_profile.empty:
        metrics = ["j1_entry_share", "j3_low_yoe_share", "broad_ai_share", "ai_tool_strict_share"]
        plot = entrant_profile.set_index("new_entrant_vs_2024")[metrics].copy()
        plot.index = ["returning" if not bool(i) else "new entrant" for i in plot.index]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot.plot(kind="bar", ax=ax)
        ax.set_ylim(0, max(0.35, float(plot.max().max()) * 1.25))
        ax.set_ylabel("Share")
        ax.set_title("T16 scraped 2026 entrants vs returning companies")
        ax.legend(fontsize=8)
        plt.xticks(rotation=0)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "new_entrant_profile.png", dpi=150)
        plt.close(fig)


def main() -> None:
    ensure_dirs()
    con = connect()
    create_row_view(con)
    sample = sample_counts(con)

    source_metrics = load_company_metrics(con, "source_group")
    source_metrics.to_csv(TABLE_DIR / "company_source_period_metrics.csv", index=False)
    arsh_common = common_companies(source_metrics, "arshkon", "scraped_2026", 3)
    change = make_change_features(source_metrics, "arshkon", "scraped_2026", arsh_common, "arshkon_only")
    clustered, cluster_summary, sil = cluster_trajectories(change)

    pooled_metrics = load_company_metrics(con, "period_group")
    pooled_common = common_companies(pooled_metrics, "2024", "2026", 3)
    pooled_change = make_change_features(pooled_metrics, "2024", "2026", pooled_common, "pooled_2024")
    pooled_change.to_csv(TABLE_DIR / "company_change_features_pooled_2024.csv", index=False)

    decomp, panel = run_decompositions(con)
    entrant_profile, entrant_top, industries = new_entrant_tables(con)
    agg = aggregator_profile(con)
    domain = domain_tables(con, arsh_common)
    sensitivity = overall_change_specs(con)
    if SENIORITY_PANEL.exists():
        pd.read_csv(SENIORITY_PANEL).to_csv(TABLE_DIR / "t30_panel_loaded_for_reference.csv", index=False)

    write_plots(cluster_summary, decomp, entrant_profile)

    summary = {
        "eligible_rows": sample.to_dict(orient="records"),
        "arshkon_scraped_overlap_ge3_companies": len(arsh_common),
        "pooled2024_scraped_overlap_ge3_companies": len(pooled_common),
        "cluster_count": int(cluster_summary["cluster_id"].nunique()) if not cluster_summary.empty else 0,
        "cluster_silhouette": sil.to_dict(orient="records"),
        "new_entrant_profile": entrant_profile.to_dict(orient="records"),
        "primary_decomposition": decomp[
            (decomp["spec"].eq("arshkon_only")) & (~decomp["aggregator_excluded"])
        ].to_dict(orient="records"),
        "seniority_panel": panel[
            (panel["spec"].eq("arshkon_only")) & (~panel["aggregator_excluded"])
        ].to_dict(orient="records"),
        "material_sensitivities": sensitivity[
            sensitivity["material_sensitivity_vs_primary"].fillna(False)
        ].to_dict(orient="records"),
        "notes": [
            "AI broad prevalence uses the V1/T08 broad list and includes MCP.",
            "AI-tool-specific prevalence uses shared taxonomy category ai_tool, excluding MCP.",
            "Requirement breadth and org-scope means use T11 features and are restricted to text_source='llm'.",
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote T16 tables to {TABLE_DIR}")
    print(f"Wrote T16 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
