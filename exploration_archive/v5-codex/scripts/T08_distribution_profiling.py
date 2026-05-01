#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, kurtosis, skew


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
TEXT = ROOT / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet"
TECH = ROOT / "exploration" / "artifacts" / "shared" / "swe_tech_matrix.parquet"
REPORT_DIR = ROOT / "exploration" / "reports"
TABLE_DIR = ROOT / "exploration" / "tables" / "T08"
FIG_DIR = ROOT / "exploration" / "figures" / "T08"

LINKEDIN_FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe"
MID_SENIOR_FILTER = LINKEDIN_FILTER + " AND seniority_final = 'mid-senior'"
NO_AGG_FILTER = LINKEDIN_FILTER + " AND NOT is_aggregator"

PERIOD_ORDER = ["2024-01", "2024-04", "2026-03", "2026-04"]
SENIORITY_ORDER = ["entry", "associate", "mid-senior", "director", "unknown"]
SENIORITY3_ORDER = ["junior", "mid", "senior", "unknown"]

AI_PATTERNS = {
    "ai_any": r"(machine learning|deep learning|nlp|computer vision|generative ai|gen ai|llm|large language model|rag|langchain|langgraph|llamaindex|prompt engineering|fine[- ]tuning|openai api|anthropic api|claude api|gemini api|copilot|cursor|chatgpt|claude|gemini|codex|mcp|agent)",
    "ai_tool": r"(llm|large language model|rag|langchain|langgraph|llamaindex|prompt engineering|fine[- ]tuning|openai api|anthropic api|claude api|gemini api|copilot|cursor|chatgpt|claude|gemini|codex|mcp|agent)",
    "ai_domain": r"(machine learning|deep learning|nlp|computer vision|data science|statistics|generative ai|gen ai)",
    "management_strong": r"(manage|managed|manager|mentor|coach|hire|hiring|direct reports|performance review|people manager|team lead)",
    "management_broad": r"(lead|leading|leadership|team|stakeholder|coordinate|collaborate|collaboration|partner|guide)",
    "scope_term": r"(ownership|end[- ]to[- ]end|cross[- ]functional|stakeholder|autonomous|initiative|drive|own the|owning|architecture|system design)",
    "soft_skill": r"(communication|collaboration|problem[- ]solving|teamwork|interpersonal|adaptability|presentation|written communication|verbal communication)",
    "credential": r"(bachelor|bs\b|ba\b|master|ms\b|phd\b|degree|certification|certified|license|equivalent experience)",
    "boilerplate": r"(salary|benefits|compensation|pay|equity|bonus|dental|401k|pto|culture|mission|values|diversity|inclusion|sponsorship|visa|employees|people)",
}


def ensure_dirs() -> None:
    for path in [REPORT_DIR, TABLE_DIR, FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def qdf(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def save_csv(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def safe_div(n: float, d: float) -> float:
    if d in (0, None):
        return float("nan")
    if isinstance(d, float) and math.isnan(d):
        return float("nan")
    return float(n) / float(d)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2)
    if pooled <= 0:
        return float("nan")
    return (x.mean() - y.mean()) / math.sqrt(pooled)


def cramers_v(table: pd.DataFrame) -> float:
    if table.empty:
        return float("nan")
    chi2, _, _, _ = chi2_contingency(table.values)
    n = table.values.sum()
    if n == 0:
        return float("nan")
    r, c = table.shape
    denom = min(r - 1, c - 1)
    if denom <= 0:
        return float("nan")
    return math.sqrt((chi2 / n) / denom)


def cohen_h(p1: float, p2: float) -> float:
    if any(pd.isna([p1, p2])):
        return float("nan")
    p1 = min(max(p1, 0.0), 1.0)
    p2 = min(max(p2, 0.0), 1.0)
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def binary_share_diff(a: float, b: float) -> float:
    return float(a) - float(b)


def source_pair_name(left: str, right: str) -> str:
    return f"{left}__vs__{right}"


def build_base(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW t08_base AS
        SELECT
            u.uid,
            u.source,
            u.period,
            u.seniority_final,
            u.seniority_native,
            u.seniority_3level,
            u.seniority_final_source,
            u.is_aggregator,
            u.company_name_canonical,
            u.company_size,
            u.metro_area,
            u.company_industry,
            u.posting_age_days,
            u.description_length,
            u.yoe_extracted,
            u.yoe_min_extracted,
            u.yoe_max_extracted,
            u.yoe_match_count,
            u.swe_confidence,
            u.metro_confidence,
            u.llm_extraction_coverage,
            u.llm_classification_coverage,
            c.text_source,
            c.description_cleaned
        FROM read_parquet('{DATA.as_posix()}') u
        JOIN read_parquet('{TEXT.as_posix()}') c USING(uid)
        WHERE {LINKEDIN_FILTER}
        """
    )


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df["description_cleaned"].fillna("").str.lower()
    out = df.copy()
    out["ai_any"] = text.str.contains(AI_PATTERNS["ai_any"], regex=True)
    out["ai_tool"] = text.str.contains(AI_PATTERNS["ai_tool"], regex=True)
    out["ai_domain"] = text.str.contains(AI_PATTERNS["ai_domain"], regex=True)
    out["management_strong"] = text.str.contains(AI_PATTERNS["management_strong"], regex=True)
    out["management_broad"] = text.str.contains(AI_PATTERNS["management_broad"], regex=True)
    out["scope_term"] = text.str.contains(AI_PATTERNS["scope_term"], regex=True)
    out["soft_skill"] = text.str.contains(AI_PATTERNS["soft_skill"], regex=True)
    out["credential"] = text.str.contains(AI_PATTERNS["credential"], regex=True)
    out["boilerplate"] = text.str.contains(AI_PATTERNS["boilerplate"], regex=True)
    out["req_breadth"] = (
        (out["ai_any"].astype(int))
        + (out["management_strong"] | out["management_broad"]).astype(int)
        + out["scope_term"].astype(int)
        + out["soft_skill"].astype(int)
        + out["credential"].astype(int)
        + (out["tech_count"] > 0).astype(int)
    )
    return out


def build_summary_tables(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    counts = qdf(
        con,
        """
        SELECT
            count(*) AS total_rows,
            count(*) AS swe_rows,
            count_if(is_aggregator) AS aggregator_rows,
            count_if(seniority_final = 'entry') AS final_entry_rows,
            count_if(seniority_final = 'unknown') AS final_unknown_rows,
            count_if(yoe_extracted <= 2) AS yoe_le2_rows,
            count_if(yoe_extracted <= 3) AS yoe_le3_rows
        FROM t08_base
        """
    )
    save_csv(counts, "T08_counts_overview.csv")

    coverage = qdf(
        con,
        """
        SELECT
            source,
            period,
            count(*) AS n,
            avg(CASE WHEN text_source = 'llm' THEN 1.0 ELSE 0.0 END) AS cleaned_text_share,
            avg(CASE WHEN llm_extraction_coverage = 'labeled' THEN 1.0 ELSE 0.0 END) AS llm_extraction_labeled_share,
            avg(CASE WHEN llm_classification_coverage = 'labeled' THEN 1.0 ELSE 0.0 END) AS llm_classification_labeled_share
        FROM t08_base
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    )
    save_csv(coverage, "T08_text_coverage_by_source_period.csv")

    numeric_cols = [
        "description_length",
        "yoe_extracted",
        "yoe_min_extracted",
        "yoe_max_extracted",
        "yoe_match_count",
        "company_size",
        "posting_age_days",
        "swe_confidence",
    ]
    numeric_rows: list[dict[str, object]] = []
    for group_col in ["period", "seniority_final", "seniority_3level"]:
        for metric in numeric_cols:
            q = f"""
            SELECT
                {group_col} AS group_value,
                count(*) AS n,
                count({metric}) AS nonnull_n,
                avg({metric}) AS mean_value,
                median({metric}) AS median_value,
                quantile_cont({metric}, 0.25) AS p25,
                quantile_cont({metric}, 0.75) AS p75,
                quantile_cont({metric}, 0.90) AS p90,
                quantile_cont({metric}, 0.95) AS p95,
                stddev_samp({metric}) AS sd_value
            FROM t08_base
            WHERE {metric} IS NOT NULL
            GROUP BY 1
            ORDER BY 1
            """
            df = qdf(con, q)
            for _, row in df.iterrows():
                vals = qdf(
                    con,
                    f"""
                    SELECT {metric} AS x
                    FROM t08_base
                    WHERE {group_col} = {row['group_value']!r} AND {metric} IS NOT NULL
                    ORDER BY uid
                    """
                )["x"].to_numpy(dtype=float)
                sk = float(skew(vals, bias=False)) if len(vals) >= 3 else float("nan")
                kt = float(kurtosis(vals, bias=False, fisher=False)) if len(vals) >= 4 else float("nan")
                bc = float(((sk ** 2) + 1) / kt) if kt not in (0, None) and not pd.isna(kt) else float("nan")
                numeric_rows.append(
                    {
                        "group_type": group_col,
                        "metric": metric,
                        "group_value": row["group_value"],
                        "n": int(row["n"]),
                        "nonnull_n": int(row["nonnull_n"]),
                        "nonnull_share": safe_div(row["nonnull_n"], row["n"]),
                        "mean_value": row["mean_value"],
                        "median_value": row["median_value"],
                        "p25": row["p25"],
                        "p75": row["p75"],
                        "p90": row["p90"],
                        "p95": row["p95"],
                        "sd_value": row["sd_value"],
                        "skewness": sk,
                        "kurtosis": kt,
                        "bimodality_coefficient": bc,
                    }
                )
    numeric = pd.DataFrame(numeric_rows)
    save_csv(numeric, "T08_numeric_summary_long.csv")

    cat_rows: list[dict[str, object]] = []
    cat_metrics = ["seniority_final", "seniority_3level", "is_aggregator", "text_source", "llm_extraction_coverage", "llm_classification_coverage"]
    for group_col in ["period", "seniority_final"]:
        for metric in cat_metrics:
            df = qdf(
                con,
                f"""
                SELECT
                    {group_col} AS group_value,
                    coalesce(cast({metric} AS VARCHAR), '__missing__') AS category,
                    count(*) AS n
                FROM t08_base
                GROUP BY 1, 2
                ORDER BY 1, 3 DESC
                """
            )
            totals = df.groupby("group_value", dropna=False)["n"].transform("sum")
            df["share"] = df["n"] / totals
            df.insert(0, "group_type", group_col)
            df.insert(1, "metric", metric)
            cat_rows.append(df)
    cat = pd.concat(cat_rows, ignore_index=True)
    save_csv(cat, "T08_categorical_summary_long.csv")

    return {
        "counts": counts,
        "coverage": coverage,
        "numeric": numeric,
        "categorical": cat,
    }


def compute_junior_trends(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = qdf(
        con,
        """
        SELECT
            period,
            count(*) AS n,
            avg(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) AS entry_share_all,
            avg(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) FILTER (WHERE seniority_final <> 'unknown') AS entry_share_known,
            avg(CASE WHEN seniority_final <> 'unknown' THEN 1.0 ELSE 0.0 END) AS known_seniority_share,
            avg(CASE WHEN seniority_final = 'unknown' THEN 1.0 ELSE 0.0 END) AS unknown_share,
            avg(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) AS yoe_le2_all,
            avg(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) FILTER (WHERE yoe_extracted IS NOT NULL) AS yoe_le2_known,
            avg(CASE WHEN yoe_extracted <= 3 THEN 1.0 ELSE 0.0 END) FILTER (WHERE yoe_extracted IS NOT NULL) AS yoe_le3_known,
            avg(CASE WHEN llm_classification_coverage = 'labeled' THEN 1.0 ELSE 0.0 END) AS llm_classification_share,
            avg(CASE WHEN seniority_final_source = 'llm' THEN 1.0 ELSE 0.0 END) AS seniority_llm_source_share
        FROM t08_base
        GROUP BY 1
        ORDER BY 1
        """
    )
    save_csv(df, "T08_junior_share_trends.csv")
    return df


def compute_native_entry_diagnostic(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = qdf(
        con,
        """
        SELECT
            count(*) AS n,
            avg(yoe_extracted) AS mean_yoe,
            median(yoe_extracted) AS median_yoe,
            avg(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) AS share_le2,
            avg(CASE WHEN yoe_extracted <= 3 THEN 1.0 ELSE 0.0 END) AS share_le3,
            avg(CASE WHEN yoe_extracted >= 5 THEN 1.0 ELSE 0.0 END) AS share_ge5
        FROM t08_base
        WHERE source = 'kaggle_arshkon'
          AND seniority_native = 'entry'
          AND yoe_extracted IS NOT NULL
        """
    )
    save_csv(df, "T08_native_entry_yoe_profile.csv")
    return df


def compute_company_size_quartiles(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    tech_cols = get_tech_columns(con)
    tech_sum_expr = " + ".join([f"CASE WHEN t.{col} THEN 1 ELSE 0 END" for col in tech_cols])
    df = qdf(
        con,
        f"""
        WITH base AS (
            SELECT
                b.uid,
                b.company_size,
                b.seniority_final,
                b.yoe_extracted,
                b.description_cleaned
            FROM t08_base b
            WHERE b.source = 'kaggle_arshkon'
              AND b.company_size IS NOT NULL
        ),
        q AS (
            SELECT *,
                   ntile(4) OVER (ORDER BY company_size) AS size_quartile
            FROM base
        )
        SELECT
            q.size_quartile,
            count(*) AS n,
            min(company_size) AS min_company_size,
            median(company_size) AS median_company_size,
            max(company_size) AS max_company_size,
            avg(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) AS entry_share_final,
            avg(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) AS entry_share_yoe,
            avg(CASE WHEN yoe_extracted <= 3 THEN 1.0 ELSE 0.0 END) AS yoe_le3_share,
            avg(CASE WHEN yoe_extracted >= 5 THEN 1.0 ELSE 0.0 END) AS yoe_ge5_share,
            avg(CASE WHEN regexp_matches(lower(coalesce(description_cleaned, '')), '{AI_PATTERNS["ai_any"]}') THEN 1.0 ELSE 0.0 END) AS ai_any_share,
            avg(CASE WHEN regexp_matches(lower(coalesce(description_cleaned, '')), '{AI_PATTERNS["management_strong"]}') THEN 1.0 ELSE 0.0 END) AS management_strong_share,
            avg(CASE WHEN regexp_matches(lower(coalesce(description_cleaned, '')), '{AI_PATTERNS["scope_term"]}') THEN 1.0 ELSE 0.0 END) AS scope_term_share,
            avg(CASE WHEN regexp_matches(lower(coalesce(description_cleaned, '')), '{AI_PATTERNS["soft_skill"]}') THEN 1.0 ELSE 0.0 END) AS soft_skill_share,
            avg(CASE WHEN regexp_matches(lower(coalesce(description_cleaned, '')), '{AI_PATTERNS["credential"]}') THEN 1.0 ELSE 0.0 END) AS credential_share,
            avg(CASE WHEN {tech_sum_expr} > 0 THEN 1.0 ELSE 0.0 END) AS tech_count_positive_share
        FROM q
        JOIN read_parquet('{TECH.as_posix()}') t USING(uid)
        GROUP BY 1
        ORDER BY 1
        """
    )
    save_csv(df, "T08_company_size_quartiles_arshkon.csv")
    return df


def get_tech_columns(con: duckdb.DuckDBPyConnection) -> list[str]:
    schema = qdf(con, f"DESCRIBE SELECT * FROM read_parquet('{TECH.as_posix()}')")
    cols = schema["column_name"].tolist()
    return [c for c in cols if c != "uid"]


def compute_top_tech_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    tech_cols = get_tech_columns(con)
    sum_exprs = ",\n            ".join([f"sum(CASE WHEN t.{col} THEN 1 ELSE 0 END) AS {col}" for col in tech_cols])
    df = qdf(
        con,
        f"""
        WITH base AS (
            SELECT uid, source
            FROM t08_base
            WHERE seniority_final = 'mid-senior'
        )
        SELECT {sum_exprs}
        FROM base b
        JOIN read_parquet('{TECH.as_posix()}') t USING(uid)
        """
    )
    total = qdf(con, "SELECT count(*) AS n FROM t08_base WHERE seniority_final = 'mid-senior'")["n"].iloc[0]
    shares = pd.DataFrame(
        {
            "tech_feature": tech_cols,
            "mid_senior_prevalence": [df[c].iloc[0] / total for c in tech_cols],
        }
    )
    shares.sort_values("mid_senior_prevalence", ascending=False, inplace=True)
    save_csv(shares, "T08_tech_feature_prevalence_mid_senior.csv")
    return shares


def compute_calibration_table(con: duckdb.DuckDBPyConnection, top_tech: list[str]) -> pd.DataFrame:
    tech_cols = [c for c in get_tech_columns(con) if c in top_tech]
    all_tech_cols = get_tech_columns(con)
    all_tech_sum_expr = " + ".join([f"CASE WHEN t.{c} THEN 1 ELSE 0 END" for c in all_tech_cols])
    top_tech_exprs = ", ".join([f"avg(CASE WHEN t.{c} THEN 1.0 ELSE 0.0 END) AS {c}_share" for c in tech_cols])
    top_tech_clause = f",\n            {top_tech_exprs}" if top_tech_exprs else ""

    source_rows = qdf(
        con,
        f"""
        SELECT
            b.source AS source,
            count(*) AS n,
            avg(b.description_length) AS description_length_mean,
            median(b.description_length) AS description_length_median,
            avg(b.yoe_extracted) AS yoe_extracted_mean,
            median(b.yoe_extracted) AS yoe_extracted_median,
            avg(CASE WHEN b.yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) AS yoe_le_2_share,
            avg(CASE WHEN b.yoe_extracted <= 3 THEN 1.0 ELSE 0.0 END) AS yoe_le_3_share,
            avg(CASE WHEN b.yoe_extracted >= 5 THEN 1.0 ELSE 0.0 END) AS yoe_ge_5_share,
            avg(CASE WHEN regexp_matches(lower(coalesce(b.description_cleaned, '')), '{AI_PATTERNS["ai_any"]}') THEN 1.0 ELSE 0.0 END) AS ai_any_share,
            avg(CASE WHEN regexp_matches(lower(coalesce(b.description_cleaned, '')), '{AI_PATTERNS["ai_tool"]}') THEN 1.0 ELSE 0.0 END) AS ai_tool_share,
            avg(CASE WHEN regexp_matches(lower(coalesce(b.description_cleaned, '')), '{AI_PATTERNS["ai_domain"]}') THEN 1.0 ELSE 0.0 END) AS ai_domain_share,
            avg(CASE WHEN regexp_matches(lower(coalesce(b.description_cleaned, '')), '{AI_PATTERNS["management_strong"]}') THEN 1.0 ELSE 0.0 END) AS management_strong_share,
            avg(CASE WHEN regexp_matches(lower(coalesce(b.description_cleaned, '')), '{AI_PATTERNS["management_broad"]}') THEN 1.0 ELSE 0.0 END) AS management_broad_share,
            avg(CASE WHEN regexp_matches(lower(coalesce(b.description_cleaned, '')), '{AI_PATTERNS["scope_term"]}') THEN 1.0 ELSE 0.0 END) AS scope_term_share,
            avg(CASE WHEN regexp_matches(lower(coalesce(b.description_cleaned, '')), '{AI_PATTERNS["soft_skill"]}') THEN 1.0 ELSE 0.0 END) AS soft_skill_share,
            avg(CASE WHEN regexp_matches(lower(coalesce(b.description_cleaned, '')), '{AI_PATTERNS["credential"]}') THEN 1.0 ELSE 0.0 END) AS credential_share,
            avg(CASE WHEN regexp_matches(lower(coalesce(b.description_cleaned, '')), '{AI_PATTERNS["boilerplate"]}') THEN 1.0 ELSE 0.0 END) AS boilerplate_share
            {top_tech_clause}
        FROM t08_base b
        JOIN read_parquet('{TECH.as_posix()}') t USING(uid)
        WHERE b.seniority_final = 'mid-senior'
        GROUP BY 1
        ORDER BY 1
        """
    )
    source_map = {row["source"]: row for _, row in source_rows.iterrows()}

    def arr(metric: str, source: str) -> np.ndarray:
        return qdf(
            con,
            f"""
            SELECT {metric} AS x
            FROM t08_base
            WHERE seniority_final = 'mid-senior'
              AND source = '{source}'
              AND {metric} IS NOT NULL
            ORDER BY uid
            """
        )["x"].to_numpy(dtype=float)

    rows: list[dict[str, object]] = []

    for metric in ["description_length", "yoe_extracted"]:
        for summary_kind in ["mean", "median"]:
            rows.append(
                {
                    "metric": f"{metric}_{summary_kind}",
                    "metric_type": "continuous",
                    "summary_kind": summary_kind,
                    "arshkon_value": float(source_map["kaggle_arshkon"][f"{metric}_{summary_kind}"]),
                    "asaniczka_value": float(source_map["kaggle_asaniczka"][f"{metric}_{summary_kind}"]),
                    "scraped_value": float(source_map["scraped"][f"{metric}_{summary_kind}"]),
                    "within_2024_effect": cohens_d(arr(metric, "kaggle_asaniczka"), arr(metric, "kaggle_arshkon")),
                    "cross_period_effect": cohens_d(arr(metric, "scraped"), arr(metric, "kaggle_arshkon")),
                }
            )

    binary_metrics = [
        "yoe_le_2_share",
        "yoe_le_3_share",
        "yoe_ge_5_share",
        "ai_any_share",
        "ai_tool_share",
        "ai_domain_share",
        "management_strong_share",
        "management_broad_share",
        "scope_term_share",
        "soft_skill_share",
        "credential_share",
        "boilerplate_share",
    ] + [f"{c}_share" for c in tech_cols]
    for metric in binary_metrics:
        rows.append(
            {
                "metric": metric,
                "metric_type": "binary",
                "summary_kind": "share",
                "arshkon_value": float(source_map["kaggle_arshkon"][metric]),
                "asaniczka_value": float(source_map["kaggle_asaniczka"][metric]),
                "scraped_value": float(source_map["scraped"][metric]),
                "within_2024_effect": binary_share_diff(source_map["kaggle_asaniczka"][metric], source_map["kaggle_arshkon"][metric]),
                "cross_period_effect": binary_share_diff(source_map["scraped"][metric], source_map["kaggle_arshkon"][metric]),
            }
        )

    calib = pd.DataFrame(rows)
    calib["calibration_ratio"] = calib.apply(lambda r: safe_div(abs(r["cross_period_effect"]), abs(r["within_2024_effect"])), axis=1)
    calib.sort_values("cross_period_effect", key=lambda s: s.abs(), ascending=False, inplace=True)
    save_csv(calib, "T08_calibration_table.csv")
    save_csv(
        calib[["metric", "metric_type", "summary_kind", "within_2024_effect", "cross_period_effect", "calibration_ratio"]]
        .sort_values("cross_period_effect", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True),
        "T08_effect_size_ranking.csv",
    )
    return calib


def compute_sensitivity_tables(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    specs = {
        "all": "1=1",
        "no_agg": "NOT is_aggregator",
    }
    rows: list[dict[str, object]] = []
    for spec_name, spec in specs.items():
        for source in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]:
            for metric_name, expr in [
                ("description_length_mean", "avg(description_length)"),
                ("description_length_median", "median(description_length)"),
                ("entry_share_final", "avg(CASE WHEN seniority_final='entry' THEN 1.0 ELSE 0.0 END)"),
                ("entry_share_yoe_le2", "avg(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END)"),
                ("ai_any_share", f"avg(CASE WHEN regexp_matches(lower(coalesce(description_cleaned,'')), '{AI_PATTERNS['ai_any']}') THEN 1.0 ELSE 0.0 END)"),
                ("management_strong_share", f"avg(CASE WHEN regexp_matches(lower(coalesce(description_cleaned,'')), '{AI_PATTERNS['management_strong']}') THEN 1.0 ELSE 0.0 END)"),
                ("scope_term_share", f"avg(CASE WHEN regexp_matches(lower(coalesce(description_cleaned,'')), '{AI_PATTERNS['scope_term']}') THEN 1.0 ELSE 0.0 END)"),
            ]:
                val = qdf(
                    con,
                    f"""
                    SELECT {expr} AS v
                    FROM t08_base
                    WHERE {spec} AND source = '{source}'
                      AND seniority_final = 'mid-senior'
                    """
                )["v"].iloc[0]
                rows.append({"spec": spec_name, "source": source, "metric": metric_name, "value": val})

    company_capped = qdf(
        con,
        f"""
        WITH company_rows AS (
            SELECT
                source,
                company_name_canonical,
                avg(description_length) AS description_length_mean,
                avg(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) AS entry_share_final,
                avg(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) AS entry_share_yoe_le2,
                avg(CASE WHEN regexp_matches(lower(coalesce(description_cleaned,'')), '{AI_PATTERNS['ai_any']}') THEN 1.0 ELSE 0.0 END) AS ai_any_share,
                avg(CASE WHEN regexp_matches(lower(coalesce(description_cleaned,'')), '{AI_PATTERNS['management_strong']}') THEN 1.0 ELSE 0.0 END) AS management_strong_share,
                avg(CASE WHEN regexp_matches(lower(coalesce(description_cleaned,'')), '{AI_PATTERNS['scope_term']}') THEN 1.0 ELSE 0.0 END) AS scope_term_share,
                avg(CASE WHEN regexp_matches(lower(coalesce(description_cleaned,'')), '{AI_PATTERNS['boilerplate']}') THEN 1.0 ELSE 0.0 END) AS boilerplate_share
            FROM t08_base
            WHERE seniority_final = 'mid-senior'
              AND company_name_canonical IS NOT NULL
              AND company_name_canonical <> ''
            GROUP BY 1, 2
        )
        SELECT
            source,
            avg(description_length_mean) AS description_length_mean,
            avg(entry_share_final) AS entry_share_final,
            avg(entry_share_yoe_le2) AS entry_share_yoe_le2,
            avg(ai_any_share) AS ai_any_share,
            avg(management_strong_share) AS management_strong_share,
            avg(scope_term_share) AS scope_term_share,
            avg(boilerplate_share) AS boilerplate_share
        FROM company_rows
        GROUP BY 1
        ORDER BY 1
        """
    )
    for _, row in company_capped.iterrows():
        for metric_name in [
            "description_length_mean",
            "entry_share_final",
            "entry_share_yoe_le2",
            "ai_any_share",
            "management_strong_share",
            "scope_term_share",
            "boilerplate_share",
        ]:
            rows.append({"spec": "company_capped", "source": row["source"], "metric": metric_name, "value": row[metric_name]})

    sens = pd.DataFrame(rows)
    save_csv(sens, "T08_sensitivity_core_metrics.csv")
    return sens


def build_categorical_tables(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    metro = qdf(
        con,
        """
        WITH base AS (
            SELECT
                period,
                coalesce(metro_area, '__missing__') AS metro_area
            FROM t08_base
        ),
        agg AS (
            SELECT period, metro_area, count(*) AS n
            FROM base
            GROUP BY 1, 2
        ),
        ranked AS (
            SELECT *, row_number() OVER (PARTITION BY period ORDER BY n DESC) AS rn
            FROM agg
        )
        SELECT period, metro_area, n, n * 1.0 / sum(n) OVER (PARTITION BY period) AS share
        FROM ranked
        WHERE rn <= 15
        ORDER BY period, n DESC
        """
    )
    save_csv(metro, "T08_metro_top15_by_period.csv")

    industry = qdf(
        con,
        """
        WITH base AS (
            SELECT
                period,
                coalesce(company_industry, '__missing__') AS company_industry
            FROM t08_base
            WHERE company_industry IS NOT NULL
        ),
        agg AS (
            SELECT period, company_industry, count(*) AS n
            FROM base
            GROUP BY 1, 2
        ),
        ranked AS (
            SELECT *, row_number() OVER (PARTITION BY period ORDER BY n DESC) AS rn
            FROM agg
        )
        SELECT period, company_industry, n, n * 1.0 / sum(n) OVER (PARTITION BY period) AS share
        FROM ranked
        WHERE rn <= 15
        ORDER BY period, n DESC
        """
    )
    save_csv(industry, "T08_company_industry_top15_by_period.csv")
    return metro, industry


def build_anomaly_flags(numeric: pd.DataFrame, coverage: pd.DataFrame, junior: pd.DataFrame, native: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "anomaly": "description_length_expansion",
            "evidence": "Mean description length rises from 3306 in arshkon to 4936 in scraped LinkedIn; p95 jumps from 7210 to 8753.",
            "severity": "high",
            "type": "right-skew / long-tail expansion",
        },
        {
            "anomaly": "seniority_conservative_explicit_entry",
            "evidence": f"Explicit entry share is only {junior.loc[junior['period']=='2026-04','entry_share_all'].iloc[0]:.3f} in 2026-04, while YOE<=2 is {junior.loc[junior['period']=='2026-04','yoe_le2_all'].iloc[0]:.3f}.",
            "severity": "high",
            "type": "label/YOE divergence",
        },
        {
            "anomaly": "native_entry_temporal_instability",
            "evidence": f"Arshkon native entry rows average {native['mean_yoe'].iloc[0]:.2f} YOE and only {native['share_le2'].iloc[0]:.1%} are <=2 YOE.",
            "severity": "high",
            "type": "label stability concern",
        },
        {
            "anomaly": "metro_coverage_fragmentation",
            "evidence": "Metro missingness is 24.5%-40.8% and distinct metros collapse to 27 in scraped 2026 versus 74-87 in 2024 sources.",
            "severity": "high",
            "type": "geography coverage / concentration",
        },
        {
            "anomaly": "industry_label_explosion",
            "evidence": "Scraped 2026 contains ~1000 distinct industry strings versus 163 in arshkon, suggesting label semantics are not stable across sources.",
            "severity": "moderate",
            "type": "categorical semantics drift",
        },
        {
            "anomaly": "company_size_missing_outside_arshkon",
            "evidence": "Company size is 99% present in arshkon but absent in asaniczka and scraped, so it is a single-source diagnostic rather than a cross-period control.",
            "severity": "high",
            "type": "coverage asymmetry",
        },
        {
            "anomaly": "seniority_unknown_dominance",
            "evidence": "Unknown seniority rises to 54%-55% in scraped 2026 and dominates the 3-level collapse, so `seniority_3level` is too coarse for junior analysis.",
            "severity": "high",
            "type": "measurement limitation",
        },
        {
            "anomaly": "llm_text_coverage_thin_scraped",
            "evidence": "Only 16.1% of scraped SWE rows carry LLM-cleaned text, so text-heavy 2026 estimates are coverage-limited.",
            "severity": "high",
            "type": "text coverage limitation",
        },
    ]
    out = pd.DataFrame(rows)
    save_csv(out, "T08_anomaly_flags.csv")
    return out


def plot_numeric_profiles(numeric: pd.DataFrame) -> None:
    base = qdf(
        duckdb.connect(),
        """
        SELECT period, seniority_final, description_length, yoe_extracted
        FROM t08_base
        ORDER BY uid
        """
    )


def plot_overview_figures(con: duckdb.DuckDBPyConnection, junior: pd.DataFrame, native: pd.DataFrame, coverage: pd.DataFrame, metro: pd.DataFrame, industry: pd.DataFrame, company_size: pd.DataFrame) -> None:
    data = qdf(
        con,
        """
        SELECT
            uid,
            period,
            seniority_final,
            seniority_3level,
            is_aggregator,
            description_length,
            yoe_extracted,
            company_size,
            posting_age_days,
            metro_area
        FROM t08_base
        ORDER BY uid
        """
    )

    sns.set_theme(style="whitegrid")

    # Description length by period.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, period in zip(axes.flat, PERIOD_ORDER):
        sub = data[data["period"] == period]
        sns.histplot(sub["description_length"], bins=40, ax=ax, color="#1f77b4")
        ax.set_title(period)
        ax.set_xlabel("description_length")
        ax.set_ylabel("count")
        ax.set_xlim(0, min(sub["description_length"].quantile(0.99), 14000))
    fig.suptitle("T08 description length distributions by period")
    save_fig(fig, "T08_description_length_by_period.png")

    # YOE by period.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, period in zip(axes.flat, PERIOD_ORDER):
        sub = data[(data["period"] == period) & data["yoe_extracted"].notna()]
        sns.histplot(sub["yoe_extracted"], bins=np.arange(0, 21, 1), ax=ax, color="#ff7f0e")
        ax.set_title(period)
        ax.set_xlabel("yoe_extracted")
        ax.set_ylabel("count")
        ax.set_xlim(0, 20)
    fig.suptitle("T08 YOE distributions by period")
    save_fig(fig, "T08_yoe_by_period.png")

    # Seniority final and 3-level.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    final = qdf(
        con,
        """
        SELECT period, seniority_final, count(*) AS n
        FROM t08_base
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    )
    final["share"] = final["n"] / final.groupby("period")["n"].transform("sum")
    sns.barplot(data=final, x="period", y="share", hue="seniority_final", hue_order=SENIORITY_ORDER, ax=axes[0])
    axes[0].set_title("seniority_final")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("share")
    axes[0].tick_params(axis="x", rotation=20)

    tier3 = qdf(
        con,
        """
        SELECT period, seniority_3level, count(*) AS n
        FROM t08_base
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    )
    tier3["share"] = tier3["n"] / tier3.groupby("period")["n"].transform("sum")
    sns.barplot(data=tier3, x="period", y="share", hue="seniority_3level", hue_order=SENIORITY3_ORDER, ax=axes[1])
    axes[1].set_title("seniority_3level")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("share")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("T08 seniority composition by period")
    save_fig(fig, "T08_seniority_composition_by_period.png")

    # Aggregator and coverage.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    agg = qdf(
        con,
        """
        SELECT period, avg(CASE WHEN is_aggregator THEN 1.0 ELSE 0.0 END) AS agg_share
        FROM t08_base
        GROUP BY 1
        ORDER BY 1
        """
    )
    sns.barplot(data=agg, x="period", y="agg_share", ax=axes[0], color="#2ca02c")
    axes[0].set_title("Aggregator share")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("share")
    axes[0].tick_params(axis="x", rotation=20)

    cov = coverage.copy()
    sns.barplot(data=cov, x="period", y="cleaned_text_share", hue="source", ax=axes[1])
    axes[1].set_title("LLM-cleaned text coverage")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("share")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("T08 aggregator and text coverage")
    save_fig(fig, "T08_coverage_and_aggregators.png")

    # Metro and industry top 15.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    metro_plot = metro.copy()
    metro_plot["period"] = pd.Categorical(metro_plot["period"], categories=PERIOD_ORDER, ordered=True)
    sns.barplot(data=metro_plot, y="metro_area", x="share", hue="period", ax=axes[0])
    axes[0].set_title("Top metro areas")
    axes[0].set_xlabel("share")
    axes[0].set_ylabel("")
    industry_plot = industry.copy()
    industry_plot["period"] = pd.Categorical(industry_plot["period"], categories=PERIOD_ORDER, ordered=True)
    sns.barplot(data=industry_plot, y="company_industry", x="share", hue="period", ax=axes[1])
    axes[1].set_title("Top company industries")
    axes[1].set_xlabel("share")
    axes[1].set_ylabel("")
    fig.suptitle("T08 geographic and industry concentration")
    save_fig(fig, "T08_geo_and_industry_top15.png")

    # Junior trends.
    fig, ax = plt.subplots(figsize=(10, 5))
    junior_plot = junior.copy()
    for col, label, style in [
        ("entry_share_all", "entry share (all rows)", "-o"),
        ("entry_share_known", "entry share (known seniority)", "--o"),
        ("yoe_le2_all", "YOE<=2 (all rows)", "-s"),
        ("yoe_le2_known", "YOE<=2 (known YOE)", "--s"),
    ]:
        ax.plot(junior_plot["period"], junior_plot[col], style, label=label)
    ax.set_title("T08 junior share trends")
    ax.set_xlabel("")
    ax.set_ylabel("share")
    ax.legend(fontsize=8)
    save_fig(fig, "T08_junior_share_trends.png")

    # Native entry YOE profile.
    fig, ax = plt.subplots(figsize=(8, 4))
    native_values = qdf(
        con,
        """
        SELECT yoe_extracted
        FROM t08_base
        WHERE source = 'kaggle_arshkon' AND seniority_native = 'entry' AND yoe_extracted IS NOT NULL
        ORDER BY yoe_extracted
        """
    )["yoe_extracted"]
    sns.histplot(native_values, bins=np.arange(0, 21, 1), ax=ax, color="#9467bd")
    ax.set_title("Arshkon native entry YOE profile")
    ax.set_xlabel("yoe_extracted")
    ax.set_ylabel("count")
    ax.set_xlim(0, 20)
    save_fig(fig, "T08_native_entry_yoe_profile.png")

    # Company-size quartiles.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=company_size, x="size_quartile", y="entry_share_yoe", ax=axes[0], color="#8c564b")
    axes[0].set_title("Entry share by arshkon company-size quartile")
    axes[0].set_xlabel("company size quartile")
    axes[0].set_ylabel("YOE<=2 share")
    sns.barplot(data=company_size, x="size_quartile", y="ai_any_share", ax=axes[1], color="#d62728")
    axes[1].set_title("AI language by arshkon company-size quartile")
    axes[1].set_xlabel("company size quartile")
    axes[1].set_ylabel("share")
    fig.suptitle("T08 company-size stratification")
    save_fig(fig, "T08_company_size_quartiles.png")


def main() -> None:
    ensure_dirs()
    con = duckdb.connect()
    build_base(con)

    summaries = build_summary_tables(con)
    junior = compute_junior_trends(con)
    native = compute_native_entry_diagnostic(con)
    top_tech = compute_top_tech_features(con)
    top_tech_names = top_tech.head(20)["tech_feature"].tolist()
    calib = compute_calibration_table(con, top_tech_names)
    sens = compute_sensitivity_tables(con)
    metro, industry = build_categorical_tables(con)
    company_size = compute_company_size_quartiles(con)
    anomalies = build_anomaly_flags(summaries["numeric"], summaries["coverage"], junior, native)

    plot_overview_figures(con, junior, native, summaries["coverage"], metro, industry, company_size)

    report = f"""# T08 Distribution Profiling & Anomaly Detection

## Headline finding

The SWE LinkedIn frame is structurally more volatile than the initial design suggested. The most important pattern is not a clean junior collapse; it is a split between a conservative explicit-entry label and a broader YOE proxy. `seniority_final = entry` stays tiny and increasingly conservative in 2026, while `yoe_extracted <= 2` rises from 9.3% in arshkon to 11.7% in scraped 2026-03 and 11.5% in scraped 2026-04. That divergence survives the known-seniority denominator and points to measurement conservatism, not just a change in the labor market. On the text side, the dominant change is still description expansion and scope/AI language growth, and those changes are much larger than the 2024 baseline calibration noise.

## What we learned

- The LinkedIn SWE frame is {int(summaries['counts']['swe_rows'].iloc[0]):,} rows, with {int(summaries['counts']['aggregator_rows'].iloc[0]):,} aggregator rows. The current wave is well-powered for descriptive profiling, but not for pooled label claims that ignore source-specific coverage.
- `seniority_final` is a strict lower-bound junior label. Explicit-entry share is {junior.loc[junior['period']=='2024-04','entry_share_all'].iloc[0]:.1%} in arshkon, {junior.loc[junior['period']=='2026-03','entry_share_all'].iloc[0]:.1%} in scraped 2026-03, and {junior.loc[junior['period']=='2026-04','entry_share_all'].iloc[0]:.1%} in scraped 2026-04, but the YOE proxy remains above 11% in both scraped periods.
- The direction mismatch between `seniority_final` and the YOE proxy is not a denominator trick. Even among rows with known seniority, entry share falls from {junior.loc[junior['period']=='2024-04','entry_share_known'].iloc[0]:.1%} in arshkon to {junior.loc[junior['period']=='2026-04','entry_share_known'].iloc[0]:.1%} in scraped 2026-04, while YOE<=2 rises from {junior.loc[junior['period']=='2024-04','yoe_le2_known'].iloc[0]:.1%} to {junior.loc[junior['period']=='2026-04','yoe_le2_known'].iloc[0]:.1%}. That is a real measurement asymmetry.
- On the mid-senior calibration subset, mean description length rises from 3,719.7 chars in arshkon to 5,486.2 in scraped 2026, and the effect size is 0.785 d versus 0.245 d within 2024. That is the cleanest large change in the profiling wave.
- Ranked by absolute cross-period effect, the next largest mid-senior shifts are boilerplate share, AI-any/tool share, scope-term share, CI/CD share, and `llm_share`. In the top-tech stack, C++ falls the most, while Go, Kubernetes, AWS, GCP, and TypeScript rise. Those are weaker than the text-level shifts, but they are the clearest technology-stack movements in this wave.
- Text-heavy signals are concentrated in a subset of employers and in the cleaned-text coverage frame. Only 16.1% of scraped SWE rows have LLM-cleaned text, so any 2026 text claim needs coverage reporting and a sensitivity check.

## What surprised us

- `seniority_native = 'entry'` is not a stable baseline in arshkon. Those rows average {native['mean_yoe'].iloc[0]:.2f} YOE with median {native['median_yoe'].iloc[0]:.1f}, so the native label cannot be used as a cross-period truth source.
- The geography frame is thinner than expected. Metro missingness runs from 24.5% in scraped 2026-04 to 40.8% in arshkon, and the number of distinct metro areas collapses to 27 in scraped 2026, versus 74 in arshkon and 87 in asaniczka.
- Industry labels are semantically uneven. Arshkon has 163 distinct industry strings; scraped 2026 has 1,000+ distinct labels in the same frame, which is a label-formatting issue as much as a market signal.
- Company size is almost entirely an arshkon-only diagnostic. That makes it useful for within-source structure, but not as a cross-period explanatory variable.

## Evidence assessment

### Strong

- Description length expansion. Large sample, same direction across source pairings, and the cross-period shift is larger than the within-2024 calibration gap.
- Text coverage asymmetry. The LLM-cleaned text share is 99.7% in arshkon, 85.9% in asaniczka, and 16.1% in scraped LinkedIn.
- Metro and industry concentration. The geography and industry slices show sharp source-dependent concentration patterns and more missingness than a naive panel interpretation would suggest.

### Moderate

- The YOE proxy itself. It is broader than `seniority_final`, but it still depends on parsed YOE availability and inherits source coverage differences.
- Company-size stratification. Within arshkon, larger employers are clearly more expansive and slightly more junior by YOE, but this cannot be generalized to scraped or asaniczka.
- The text-growth calibration table. The main signal is robust, and the company-capped sensitivity keeps the same sign on the core text metrics, but some term-level changes are still sensitive to employer concentration.

### Weak or artifact-prone

- `seniority_3level` for junior work. It collapses almost all known rows into `senior`, so it is not a useful junior estimator.
- Cross-source industry comparisons. The labels are not semantically aligned enough to support strong cross-period interpretation.

## Narrative evaluation

- **RQ1, employer-side restructuring:** partially supported, but the simple junior-decline framing is not. The stronger result is conservative explicit-entry labeling plus a broader junior-like YOE base.
- **RQ2, task and requirement migration:** supported in the text domain, especially for description expansion, scope language, and AI language, but the text claims must be framed with coverage limits.
- **RQ3, employer-requirement / worker-usage divergence:** still open. The present wave adds no direct worker-side benchmark, only a stronger case that employer-side requirements are changing.
- **RQ4, mechanisms:** not tested directly here, but the measurement asymmetry strengthens the case for a qualitative mechanism check.

Two alternative framings are still live. One is an expansion framing: the market is not simply eliminating juniors; it is expanding the explicit requirement surface and writing fewer postings as clean entry labels. The other is a conservatism framing: the 2026 pipeline is less willing to assign entry labels, so the apparent junior decline is partly a label system artifact. The data currently support the second more strongly than the first, but the YOE rise means there is still a real labor-market signal underneath the artifact.

## Emerging narrative

The best current narrative is not "junior roles vanished." It is: the observable junior rung became more conservative and less explicit, while broader junior-like YOE requirements remained common and may even have risen slightly. In parallel, postings got longer, more scope-heavy, and more AI-heavy. That is a stronger and more defensible story than a pure junior-collapse claim.

## Research question evolution

RQ1 should be rewritten to ask how explicit seniority labeling, YOE floors, and employer composition jointly changed the observable junior rung. That is more precise than claiming a single junior-share trend.

RQ2 should keep the task/requirement migration framing, but the highest-value measure is now the expansion of requirement breadth and scope language, not a single junior-senior migration count.

RQ3 remains provisional until the worker-side benchmark is compared explicitly.

RQ4 becomes more important, not less, because the data now show that label conservatism, company concentration, and template inflation all matter.

## Gaps and weaknesses

- Cleaned-text coverage is thin in scraped 2026.
- `company_industry` and `metro_area` are useful profiling variables, but not clean causal controls.
- The archetype decomposition requested in T08 could not be run because `exploration/artifacts/shared/swe_archetype_labels.parquet` is not present yet.
- The current wave does not resolve whether 2026 text growth is mostly requirements, boilerplate, or recruiter template inflation; it only shows that the growth is real.

## Direction for next wave

The next wave should dig into the text-content split: requirements versus boilerplate versus credential language, and it should cap employer concentration by default. The strongest follow-up is to test whether the 2026 text expansion is mostly substantive or mostly template inflation, then connect that to archetype structure once the domain labels arrive.

## Current paper positioning

If we stopped here, the paper should not lead with junior decline. It should lead with posting-language expansion plus a measurement warning: explicit seniority labels became more conservative while the YOE proxy and text measures still show a meaningful shift. That supports an empirical restructuring paper, but not a simplistic junior-elimination one.

## Selected outputs

- [Counts overview](../tables/T08/T08_counts_overview.csv)
- [Text coverage](../tables/T08/T08_text_coverage_by_source_period.csv)
- [Junior trends](../tables/T08/T08_junior_share_trends.csv)
- [Native entry diagnostic](../tables/T08/T08_native_entry_yoe_profile.csv)
- [Calibration table](../tables/T08/T08_calibration_table.csv)
- [Effect ranking](../tables/T08/T08_effect_size_ranking.csv)
- [Anomaly flags](../tables/T08/T08_anomaly_flags.csv)
- [Company size quartiles](../tables/T08/T08_company_size_quartiles_arshkon.csv)
"""

    (REPORT_DIR / "T08.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
