#!/usr/bin/env python
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
UNIFIED = ROOT / "data" / "unified.parquet"
CLEANED = ROOT / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet"
TECH_MATRIX = ROOT / "exploration" / "artifacts" / "shared" / "swe_tech_matrix.parquet"

REPORT_DIR = ROOT / "exploration" / "reports"
TABLE_DIR = ROOT / "exploration" / "tables" / "T11"
FIG_DIR = ROOT / "exploration" / "figures" / "T11"

LINKEDIN_FILTER = "source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe"

SOFT_SKILL_TERMS = {
    "communication": r"\bcommunication(s| skill(s)?)?\b",
    "collaboration": r"\bcollaboration|collaborative\b",
    "problem_solving": r"problem[- ]solving",
    "teamwork": r"\bteamwork\b",
    "presentation": r"\bpresentation(s)?\b",
    "interpersonal": r"\binterpersonal\b",
    "adaptability": r"\badaptability\b",
    "detail_oriented": r"detail[- ]oriented|attention to detail",
    "customer_focus": r"\bcustomer[- ]?facing|customer[- ]?focused\b",
}

SCOPE_TERMS = {
    "ownership": r"\bownership\b",
    "end_to_end": r"end[- ]to[- ]end|\be2e\b",
    "cross_functional": r"cross[- ]functional",
    "autonomous": r"\bautonomous(ly)?\b",
    "initiative": r"\binitiative\b",
    "strategic": r"\bstrategic\b",
    "roadmap": r"\broadmap\b",
}

MGMT_STRONG_TERMS = {
    "manage": r"\bmanage(d|r|rs|ing)?\b",
    "mentor": r"\bmentor(ship|ing)?\b",
    "coach": r"\bcoach(ing|es|ed)?\b",
    "hire": r"\bhire(d|s|ing)?\b",
    "direct_reports": r"direct reports?",
    "performance_review": r"performance review(s)?",
    "headcount": r"\bheadcount\b",
    "supervise": r"\bsupervis(e|ion|or|ory)\b",
    "people_manager": r"\bpeople manager\b",
}

MGMT_BROAD_TERMS = {
    "lead": r"\blead(s|er|ing)?\b",
    "team": r"\bteam(s)?\b",
    "stakeholder": r"\bstakeholder(s)?\b",
    "coordinate": r"\bcoordinate(d|s|ing)?\b",
    "collaborate": r"\bcollaborat(e|es|ed|ing|ion|ive)\b",
    "partner": r"\bpartner(s|ed|ing)?\b",
}

EDU_LEVELS = {
    "phd": r"\b(ph\.?d\.?|phd|doctorate|doctoral)\b",
    "ms": r"\b(master'?s|masters|m\.?s\.?|m\.?sc\.?|ms\b)\b",
    "bs": r"\b(bachelor'?s|bachelors|b\.?s\.?|b\.?a\.?|b\.?sc\.?|bs\b|ba\b)\b",
}

MANAGEMENT_SAMPLE_TERMS = [
    "manage",
    "mentor",
    "coach",
    "hire",
    "direct_reports",
    "performance_review",
    "headcount",
    "supervise",
    "people_manager",
    "lead",
    "team",
    "stakeholder",
    "coordinate",
    "collaborate",
    "partner",
]


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


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def regex_hygiene() -> None:
    # Validate patterns with non-word boundaries and common false-positive traps.
    manage = re.compile(MGMT_STRONG_TERMS["manage"], re.I)
    mentor = re.compile(MGMT_STRONG_TERMS["mentor"], re.I)
    review = re.compile(MGMT_STRONG_TERMS["performance_review"], re.I)
    lead = re.compile(MGMT_BROAD_TERMS["lead"], re.I)
    team = re.compile(MGMT_BROAD_TERMS["team"], re.I)
    scope = re.compile(SCOPE_TERMS["cross_functional"], re.I)
    assert manage.search("manage a team")
    assert mentor.search("mentor junior engineers")
    assert review.search("performance reviews and headcount planning")
    assert lead.search("lead a team and coordinate with stakeholders")
    assert team.search("team collaboration and ownership")
    assert scope.search("cross-functional ownership")
    assert not manage.search("management consultant")
    assert not lead.search("leadership principles")


def build_sum_expr(text_col: str, terms: dict[str, str]) -> str:
    parts = [
        f"CASE WHEN regexp_matches(lower(coalesce({text_col}, '')), {sql_quote(pattern)}) THEN 1 ELSE 0 END"
        for pattern in terms.values()
    ]
    return " + ".join(parts) if parts else "0"


def education_expr(text_col: str) -> str:
    return (
        f"CASE "
        f"WHEN regexp_matches(lower(coalesce({text_col}, '')), {sql_quote(EDU_LEVELS['phd'])}) THEN 'phd' "
        f"WHEN regexp_matches(lower(coalesce({text_col}, '')), {sql_quote(EDU_LEVELS['ms'])}) THEN 'ms' "
        f"WHEN regexp_matches(lower(coalesce({text_col}, '')), {sql_quote(EDU_LEVELS['bs'])}) THEN 'bs' "
        f"ELSE 'none' END"
    )


def fetch_schema_columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    return [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')").fetchall()]


def summarize_metric(df: pd.DataFrame, metric: str, group_cols: list[str]) -> pd.DataFrame:
    q = (
        df.groupby(group_cols, dropna=False)[metric]
        .agg(n="size", mean="mean", median="median", p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75))
        .reset_index()
    )
    q.insert(len(group_cols), "metric", metric)
    return q


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    x = x.dropna().astype(float)
    y = y.dropna().astype(float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2)
    if pooled <= 0:
        return float("nan")
    return float((x.mean() - y.mean()) / np.sqrt(pooled))


def main() -> None:
    ensure_dirs()
    regex_hygiene()
    con = duckdb.connect()

    tech_cols_all = fetch_schema_columns(con, TECH_MATRIX)
    ai_candidates = {
        "ai",
        "machine_learning",
        "deep_learning",
        "data_science",
        "statistics",
        "nlp",
        "computer_vision",
        "generative_ai",
        "tensorflow",
        "pytorch",
        "scikit_learn",
        "mlflow",
        "kubeflow",
        "ray",
        "hugging_face",
        "openai_api",
        "anthropic_api",
        "claude_api",
        "gemini_api",
        "langchain",
        "langgraph",
        "llamaindex",
        "rag",
        "vector_db",
        "pinecone",
        "weaviate",
        "chroma",
        "milvus",
        "faiss",
        "prompt_engineering",
        "fine_tuning",
        "mcp",
        "llm",
        "copilot",
        "cursor",
        "chatgpt",
        "claude",
        "gemini",
        "codex",
        "agent",
    }
    method_exclude = {
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
    ai_cols = [c for c in tech_cols_all if c in ai_candidates]
    tech_cols = [c for c in tech_cols_all if c not in ai_candidates and c not in method_exclude and c != "uid"]
    if "uid" in tech_cols:
        tech_cols.remove("uid")

    # Build the joined feature view.
    tech_sum_expr = " + ".join([f"CAST(COALESCE(tm.{c}, false) AS INTEGER)" for c in tech_cols]) if tech_cols else "0"
    ai_sum_expr = " + ".join([f"CAST(COALESCE(tm.{c}, false) AS INTEGER)" for c in ai_cols]) if ai_cols else "0"
    soft_sum_expr = build_sum_expr("c.description_cleaned", SOFT_SKILL_TERMS)
    scope_sum_expr = build_sum_expr("c.description_cleaned", SCOPE_TERMS)
    mgmt_strong_sum_expr = build_sum_expr("c.description_cleaned", MGMT_STRONG_TERMS)
    mgmt_broad_sum_expr = build_sum_expr("c.description_cleaned", MGMT_BROAD_TERMS)

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW feature_base AS
        SELECT
          u.uid,
          u.source,
          u.period,
          u.title,
          u.title_normalized,
          u.company_name_canonical,
          u.seniority_final,
          u.seniority_3level,
          u.yoe_extracted,
          u.is_aggregator,
          c.text_source,
          c.description_cleaned,
          length(coalesce(c.description_cleaned, '')) AS text_len,
          {tech_sum_expr} AS tech_count,
          {ai_sum_expr} AS ai_count,
          {soft_sum_expr} AS soft_skill_count,
          {scope_sum_expr} AS scope_count,
          {mgmt_strong_sum_expr} AS management_strong_count,
          {mgmt_broad_sum_expr} AS management_broad_count,
          {education_expr('c.description_cleaned')} AS education_level,
          CASE WHEN {education_expr('c.description_cleaned')} != 'none' THEN 1 ELSE 0 END AS education_flag,
          CASE WHEN u.yoe_extracted IS NOT NULL THEN 1 ELSE 0 END AS yoe_flag
        FROM read_parquet('{UNIFIED.as_posix()}') u
        JOIN read_parquet('{CLEANED.as_posix()}') c USING (uid)
        LEFT JOIN read_parquet('{TECH_MATRIX.as_posix()}') tm USING (uid)
        WHERE {LINKEDIN_FILTER}
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW feature_capped AS
        SELECT *,
               row_number() OVER (
                 PARTITION BY source, period, company_name_canonical
                 ORDER BY uid
               ) AS company_rank
        FROM feature_base
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW feature_primary AS
        SELECT *,
               tech_count + soft_skill_count + scope_count + management_strong_count + ai_count + education_flag + yoe_flag AS requirement_breadth,
               (CASE WHEN tech_count > 0 THEN 1 ELSE 0 END)
                 + (CASE WHEN soft_skill_count > 0 THEN 1 ELSE 0 END)
                 + (CASE WHEN scope_count > 0 THEN 1 ELSE 0 END)
                 + (CASE WHEN management_strong_count > 0 THEN 1 ELSE 0 END)
                 + (CASE WHEN ai_count > 0 THEN 1 ELSE 0 END)
                 + education_flag
                 + yoe_flag AS credential_stack_depth,
               (CASE WHEN management_broad_count > 0 THEN 1 ELSE 0 END) AS management_broad_any,
               (CASE WHEN management_strong_count > 0 THEN 1 ELSE 0 END) AS management_strong_any,
               (CASE WHEN ai_count > 0 THEN 1 ELSE 0 END) AS ai_any,
               (CASE WHEN scope_count > 0 THEN 1 ELSE 0 END) AS scope_any
        FROM feature_capped
        WHERE company_rank <= 25
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW feature_primary_llm AS
        SELECT * FROM feature_primary WHERE text_source = 'llm'
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW feature_primary_raw AS
        SELECT * FROM feature_primary WHERE text_source = 'raw'
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW feature_nocap_llm AS
        SELECT *,
               tech_count + soft_skill_count + scope_count + management_strong_count + ai_count + education_flag + yoe_flag AS requirement_breadth,
               (CASE WHEN tech_count > 0 THEN 1 ELSE 0 END)
                 + (CASE WHEN soft_skill_count > 0 THEN 1 ELSE 0 END)
                 + (CASE WHEN scope_count > 0 THEN 1 ELSE 0 END)
                 + (CASE WHEN management_strong_count > 0 THEN 1 ELSE 0 END)
                 + (CASE WHEN ai_count > 0 THEN 1 ELSE 0 END)
                 + education_flag
                 + yoe_flag AS credential_stack_depth,
               (CASE WHEN management_broad_count > 0 THEN 1 ELSE 0 END) AS management_broad_any,
               (CASE WHEN management_strong_count > 0 THEN 1 ELSE 0 END) AS management_strong_any,
               (CASE WHEN ai_count > 0 THEN 1 ELSE 0 END) AS ai_any,
               (CASE WHEN scope_count > 0 THEN 1 ELSE 0 END) AS scope_any
        FROM feature_base
        WHERE text_source = 'llm'
        """
    )

    metric_names = [
        "tech_count",
        "ai_count",
        "soft_skill_count",
        "scope_count",
        "management_strong_count",
        "management_broad_count",
        "education_flag",
        "yoe_flag",
        "requirement_breadth",
        "credential_stack_depth",
        "text_len",
        "tech_density",
        "scope_density",
    ]

    # Density fields need to be materialized via a small helper view.
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW feature_primary_density AS
        SELECT *,
               1000.0 * tech_count / NULLIF(text_len, 0) AS tech_density,
               1000.0 * scope_count / NULLIF(text_len, 0) AS scope_density
        FROM feature_primary
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW feature_primary_llm_density AS
        SELECT *,
               1000.0 * tech_count / NULLIF(text_len, 0) AS tech_density,
               1000.0 * scope_count / NULLIF(text_len, 0) AS scope_density
        FROM feature_primary_llm
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW feature_primary_raw_density AS
        SELECT *,
               1000.0 * tech_count / NULLIF(text_len, 0) AS tech_density,
               1000.0 * scope_count / NULLIF(text_len, 0) AS scope_density
        FROM feature_primary_raw
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW feature_nocap_llm_density AS
        SELECT *,
               1000.0 * tech_count / NULLIF(text_len, 0) AS tech_density,
               1000.0 * scope_count / NULLIF(text_len, 0) AS scope_density
        FROM feature_nocap_llm
        """
    )

    # Primary distribution summaries by source, period, and seniority.
    summary_rows = []
    metric_source = "feature_primary_llm_density"
    for metric in metric_names:
        if metric in {"tech_density", "scope_density"}:
            metric_sql = metric
        else:
            metric_sql = metric
        summary_rows.append(
            qdf(
                con,
                f"""
                SELECT source, period, seniority_final,
                       count(*) AS n,
                       avg({metric_sql}) AS mean,
                       median({metric_sql}) AS median,
                       quantile_cont({metric_sql}, 0.25) AS p25,
                       quantile_cont({metric_sql}, 0.75) AS p75
                FROM {metric_source}
                GROUP BY 1,2,3
                ORDER BY 1,2,3
                """
            ).assign(metric=metric, text_source="llm", company_cap="25")
        )
    summary_df = pd.concat(summary_rows, ignore_index=True)
    save_csv(summary_df, "T11_complexity_summary_primary_llm_cap25.csv")

    # Raw fallback sensitivity on the same company-capped frame.
    raw_rows = []
    metric_source = "feature_primary_raw_density"
    for metric in metric_names:
        raw_rows.append(
            qdf(
                con,
                f"""
                SELECT source, period, seniority_final,
                       count(*) AS n,
                       avg({metric}) AS mean,
                       median({metric}) AS median,
                       quantile_cont({metric}, 0.25) AS p25,
                       quantile_cont({metric}, 0.75) AS p75
                FROM {metric_source}
                GROUP BY 1,2,3
                ORDER BY 1,2,3
                """
            ).assign(metric=metric, text_source="raw", company_cap="25")
        )
    raw_summary_df = pd.concat(raw_rows, ignore_index=True)
    save_csv(raw_summary_df, "T11_complexity_summary_raw_cap25.csv")

    # No-cap sensitivity for llm text.
    nocap_rows = []
    for metric in metric_names:
        nocap_rows.append(
            qdf(
                con,
                f"""
                SELECT source, period, seniority_final,
                       count(*) AS n,
                       avg({metric}) AS mean,
                       median({metric}) AS median,
                       quantile_cont({metric}, 0.25) AS p25,
                       quantile_cont({metric}, 0.75) AS p75
                FROM feature_nocap_llm_density
                GROUP BY 1,2,3
                ORDER BY 1,2,3
                """
            ).assign(metric=metric, text_source="llm", company_cap="none")
        )
    nocap_summary_df = pd.concat(nocap_rows, ignore_index=True)
    save_csv(nocap_summary_df, "T11_complexity_summary_primary_llm_nocap.csv")

    # Aggregator-exclusion sensitivity on llm text.
    noagg_rows = []
    for metric in metric_names:
        noagg_rows.append(
            qdf(
                con,
                f"""
                SELECT source, period, seniority_final,
                       count(*) AS n,
                       avg({metric}) AS mean,
                       median({metric}) AS median,
                       quantile_cont({metric}, 0.25) AS p25,
                       quantile_cont({metric}, 0.75) AS p75
                FROM feature_primary_llm_density
                WHERE NOT is_aggregator
                GROUP BY 1,2,3
                ORDER BY 1,2,3
                """
            ).assign(metric=metric, text_source="llm", company_cap="25", aggregator_excluded=True)
        )
    noagg_summary_df = pd.concat(noagg_rows, ignore_index=True)
    save_csv(noagg_summary_df, "T11_complexity_summary_primary_llm_noagg.csv")

    # Text-source coverage by period and seniority.
    coverage = qdf(
        con,
        """
        SELECT source, period, text_source, seniority_final, count(*) AS n
        FROM feature_base
        GROUP BY 1,2,3,4
        ORDER BY 1,2,3,4
        """
    )
    save_csv(coverage, "T11_text_source_coverage.csv")

    # Explicit entry vs YOE proxy comparison. Pool scraped 2026 as the current window.
    entry_comp = qdf(
        con,
        """
        WITH base AS (
          SELECT source, period, seniority_final, yoe_extracted,
                 requirement_breadth, credential_stack_depth, tech_count, tech_density, scope_density
          FROM feature_primary_llm_density
          WHERE source IN ('kaggle_arshkon', 'scraped')
        )
        SELECT 'explicit_entry' AS entry_definition,
               CASE WHEN source = 'kaggle_arshkon' THEN '2024-arshkon' ELSE '2026-scraped' END AS period_group,
               count(*) AS n,
               avg(requirement_breadth) AS mean_requirement_breadth,
               avg(credential_stack_depth) AS mean_stack_depth,
               avg(tech_count) AS mean_tech_count,
               avg(tech_density) AS mean_tech_density,
               avg(scope_density) AS mean_scope_density
        FROM base
        WHERE seniority_final = 'entry'
        GROUP BY 1,2
        UNION ALL
        SELECT 'yoe_proxy_le2' AS entry_definition,
               CASE WHEN source = 'kaggle_arshkon' THEN '2024-arshkon' ELSE '2026-scraped' END AS period_group,
               count(*) AS n,
               avg(requirement_breadth) AS mean_requirement_breadth,
               avg(credential_stack_depth) AS mean_stack_depth,
               avg(tech_count) AS mean_tech_count,
               avg(tech_density) AS mean_tech_density,
               avg(scope_density) AS mean_scope_density
        FROM base
        WHERE yoe_extracted <= 2
        GROUP BY 1,2
        ORDER BY 1,2
        """
    )
    save_csv(entry_comp, "T11_entry_complexity_comparison.csv")

    # Within-2024 calibration.
    calib = []
    baseline = qdf(
        con,
        """
        SELECT *
        FROM feature_primary_llm_density
        WHERE source = 'kaggle_arshkon'
        """
    )
    alt = qdf(
        con,
        """
        SELECT *
        FROM feature_primary_llm_density
        WHERE source = 'kaggle_asaniczka'
        """
    )
    current = qdf(
        con,
        """
        SELECT *
        FROM feature_primary_llm_density
        WHERE source = 'scraped'
        """
    )
    for metric in metric_names:
        calib.append(
            {
                "metric": metric,
                "baseline_mean": baseline[metric].mean(),
                "alt_2024_mean": alt[metric].mean(),
                "current_mean": current[metric].mean(),
                "within_2024_diff": alt[metric].mean() - baseline[metric].mean(),
                "cross_period_diff": current[metric].mean() - baseline[metric].mean(),
                "within_2024_d": cohens_d(alt[metric], baseline[metric]),
                "cross_period_d": cohens_d(current[metric], baseline[metric]),
            }
        )
    calib_df = pd.DataFrame(calib)
    calib_df["signal_to_noise"] = calib_df["cross_period_d"].abs() / calib_df["within_2024_d"].abs().replace(0, np.nan)
    save_csv(calib_df, "T11_calibration_metrics.csv")

    # Education level distribution.
    edu = qdf(
        con,
        """
        SELECT source, period, seniority_final, education_level, count(*) AS n
        FROM feature_primary_llm_density
        GROUP BY 1,2,3,4
        ORDER BY 1,2,3,4
        """
    )
    save_csv(edu, "T11_education_level_distribution.csv")

    # Management term breakdown by period and tier.
    management_term_rows = []
    for tier_name, terms in [("management_strong", MGMT_STRONG_TERMS), ("management_broad", MGMT_BROAD_TERMS)]:
        for term_name, pattern in terms.items():
            term_counts = qdf(
                con,
                f"""
                SELECT source, period, count(*) AS n
                FROM feature_primary_llm_density
                WHERE regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(pattern)})
                GROUP BY 1,2
                ORDER BY 1,2
                """
            )
            term_counts["tier"] = tier_name
            term_counts["term"] = term_name
            management_term_rows.append(term_counts)
    management_terms = pd.concat(management_term_rows, ignore_index=True)
    total_by_source_period = coverage[coverage.text_source == "llm"].groupby(["source", "period"], dropna=False)["n"].sum().reset_index().rename(columns={"n": "total_n"})
    management_terms = management_terms.merge(total_by_source_period, on=["source", "period"], how="left")
    management_terms["share"] = management_terms["n"] / management_terms["total_n"]
    save_csv(management_terms, "T11_management_term_breakdown.csv")

    # Validation sample, stratified by period and sampled from all management matches.
    validation_sample = qdf(
        con,
        f"""
        WITH matches AS (
          SELECT
            uid,
            source,
            period,
            title,
            company_name_canonical,
            text_source,
            description_cleaned,
            CASE
              WHEN regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_STRONG_TERMS['manage'])}) THEN 'management_strong:manage'
              WHEN regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_STRONG_TERMS['mentor'])}) THEN 'management_strong:mentor'
              WHEN regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_STRONG_TERMS['coach'])}) THEN 'management_strong:coach'
              WHEN regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_STRONG_TERMS['hire'])}) THEN 'management_strong:hire'
              WHEN regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_STRONG_TERMS['direct_reports'])}) THEN 'management_strong:direct_reports'
              WHEN regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_STRONG_TERMS['performance_review'])}) THEN 'management_strong:performance_review'
              WHEN regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_STRONG_TERMS['headcount'])}) THEN 'management_strong:headcount'
              WHEN regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_STRONG_TERMS['supervise'])}) THEN 'management_strong:supervise'
              WHEN regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_STRONG_TERMS['people_manager'])}) THEN 'management_strong:people_manager'
              WHEN regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_BROAD_TERMS['lead'])}) THEN 'management_broad:lead'
              WHEN regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_BROAD_TERMS['team'])}) THEN 'management_broad:team'
              WHEN regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_BROAD_TERMS['stakeholder'])}) THEN 'management_broad:stakeholder'
              WHEN regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_BROAD_TERMS['coordinate'])}) THEN 'management_broad:coordinate'
              WHEN regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_BROAD_TERMS['collaborate'])}) THEN 'management_broad:collaborate'
              WHEN regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_BROAD_TERMS['partner'])}) THEN 'management_broad:partner'
              ELSE NULL
            END AS trigger_term
          FROM feature_primary_llm_density
          WHERE
            regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_STRONG_TERMS['manage'])})
            OR regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_STRONG_TERMS['mentor'])})
            OR regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_STRONG_TERMS['coach'])})
            OR regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_STRONG_TERMS['hire'])})
            OR regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_STRONG_TERMS['direct_reports'])})
            OR regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_STRONG_TERMS['performance_review'])})
            OR regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_STRONG_TERMS['headcount'])})
            OR regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_STRONG_TERMS['supervise'])})
            OR regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_STRONG_TERMS['people_manager'])})
            OR regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_BROAD_TERMS['lead'])})
            OR regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_BROAD_TERMS['team'])})
            OR regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_BROAD_TERMS['stakeholder'])})
            OR regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_BROAD_TERMS['coordinate'])})
            OR regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_BROAD_TERMS['collaborate'])})
            OR regexp_matches(lower(coalesce(description_cleaned, '')), {sql_quote(MGMT_BROAD_TERMS['partner'])})
        )
        SELECT *,
               left(coalesce(description_cleaned, ''), 300) AS excerpt,
               row_number() OVER (PARTITION BY period ORDER BY random()) AS rn
        FROM matches
        WHERE trigger_term IS NOT NULL
        QUALIFY rn <= 25
        ORDER BY period, rn
        """
    )
    save_csv(validation_sample, "T11_management_validation_sample.csv")

    # Outlier analysis: top 1% by requirement breadth.
    threshold = qdf(
        con,
        """
        SELECT quantile_cont(requirement_breadth, 0.99) AS p99
        FROM feature_primary_llm_density
        """
    )["p99"].iloc[0]
    outliers = qdf(
        con,
        f"""
        SELECT
          uid,
          source,
          period,
          title,
          company_name_canonical,
          seniority_final,
          yoe_extracted,
          text_source,
          text_len,
          tech_count,
          ai_count,
          soft_skill_count,
          scope_count,
          management_strong_count,
          management_broad_count,
          education_level,
          requirement_breadth,
          credential_stack_depth,
          tech_density,
          scope_density,
          left(coalesce(description_cleaned, ''), 240) AS excerpt
        FROM feature_primary_llm_density
        WHERE requirement_breadth >= {threshold}
        ORDER BY requirement_breadth DESC, text_len DESC
        """
    )
    save_csv(outliers, "T11_outlier_postings_top1pct.csv")

    # Figures.
    sns.set_theme(style="whitegrid")

    # Figure 1: metric distributions across the primary llm-capped sample.
    metric_fig_df = summary_df[summary_df.metric.isin(["requirement_breadth", "credential_stack_depth", "tech_count", "scope_density"])].copy()
    metric_fig_df["group"] = metric_fig_df["source"] + " " + metric_fig_df["period"]
    order = ["kaggle_asaniczka 2024-01", "kaggle_arshkon 2024-04", "scraped 2026-03", "scraped 2026-04"]
    metric_fig_df["group"] = pd.Categorical(metric_fig_df["group"], categories=order, ordered=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("requirement_breadth", "Requirement breadth"),
        ("credential_stack_depth", "Credential stack depth"),
        ("tech_count", "Distinct technologies"),
        ("scope_density", "Scope density per 1k chars"),
    ]
    for ax, (metric, title) in zip(axes.flat, panels):
        subset = metric_fig_df[metric_fig_df.metric == metric].sort_values("group")
        sns.barplot(data=subset, x="group", y="median", hue="seniority_final", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Median")
        ax.tick_params(axis="x", rotation=20)
        if ax.get_legend() is not None:
            ax.legend_.remove()
    axes[0, 0].legend(frameon=False, title="Seniority", loc="upper right", fontsize=8)
    fig.suptitle("Primary llm-cleaned complexity metrics by source, period, and seniority", y=1.02)
    save_fig(fig, "T11_complexity_distributions.png")

    # Figure 2: entry-level complexity comparison.
    entry_plot = entry_comp.copy()
    entry_plot["period_group"] = pd.Categorical(entry_plot["period_group"], categories=["2024-arshkon", "2026-scraped"], ordered=True)
    entry_metric = entry_plot.melt(
        id_vars=["entry_definition", "period_group", "n"],
        value_vars=["mean_requirement_breadth", "mean_stack_depth", "mean_tech_count", "mean_scope_density"],
        var_name="metric",
        value_name="value",
    )
    entry_metric["metric"] = entry_metric["metric"].map(
        {
            "mean_requirement_breadth": "Requirement breadth",
            "mean_stack_depth": "Credential stack depth",
            "mean_tech_count": "Tech count",
            "mean_scope_density": "Scope density",
        }
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=entry_metric, x="period_group", y="value", hue="entry_definition", ax=ax)
    ax.set_title("Entry-level complexity under explicit entry vs YOE proxy")
    ax.set_xlabel("")
    ax.set_ylabel("Mean value")
    ax.legend(frameon=False, title="Entry definition")
    save_fig(fig, "T11_entry_complexity_comparison.png")

    # Figure 3: management trigger breakdown.
    mgmt_plot = management_terms.copy()
    mgmt_plot["period_label"] = mgmt_plot["source"].map(
        {"kaggle_arshkon": "arshkon", "kaggle_asaniczka": "asaniczka", "scraped": "scraped"}
    ) + " " + mgmt_plot["period"]
    mgmt_plot = mgmt_plot[mgmt_plot["source"].isin(["kaggle_arshkon", "scraped"])]
    top_terms = (
        mgmt_plot.groupby(["tier", "term"])["n"].sum().reset_index().sort_values(["tier", "n"], ascending=[True, False]).groupby("tier").head(10)
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    top_terms["label"] = top_terms["tier"] + ": " + top_terms["term"]
    sns.barplot(data=top_terms.sort_values("n"), y="label", x="n", hue="tier", dodge=False, ax=ax)
    ax.set_title("Top management triggers in the primary llm-capped sample")
    ax.set_xlabel("Matched postings")
    ax.set_ylabel("")
    ax.legend(frameon=False)
    save_fig(fig, "T11_management_triggers.png")

    # Figure 4: outlier complexity profile.
    outlier_plot = outliers.head(20).copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=outlier_plot.sort_values("requirement_breadth"), y="title", x="requirement_breadth", hue="source", dodge=False, ax=ax)
    ax.set_title("Most complex postings by requirement breadth")
    ax.set_xlabel("Requirement breadth")
    ax.set_ylabel("")
    ax.legend(frameon=False, title="Source")
    save_fig(fig, "T11_outlier_examples.png")


if __name__ == "__main__":
    main()
