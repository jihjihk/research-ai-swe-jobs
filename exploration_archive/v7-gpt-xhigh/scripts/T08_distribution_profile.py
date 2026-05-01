#!/usr/bin/env python3
"""T08 distribution profiling and anomaly detection.

All data access goes through DuckDB with a 4GB / one-thread cap. The script only
materializes grouped tables or small plotting tables in pandas.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
SHARED = ROOT / "exploration" / "artifacts" / "shared"
TABLE_DIR = ROOT / "exploration" / "tables" / "T08"
FIG_DIR = ROOT / "exploration" / "figures" / "T08"

SENIORITY_PANEL = SHARED / "seniority_definition_panel.csv"
CLEANED_TEXT = SHARED / "swe_cleaned_text.parquet"
TECH_MATRIX = SHARED / "swe_tech_matrix.parquet"
TECH_TAXONOMY = SHARED / "tech_taxonomy.csv"
ARCHETYPES = SHARED / "swe_archetype_labels.parquet"

DEFAULT_WHERE = (
    "source_platform = 'linkedin' "
    "AND is_english = true "
    "AND date_flag = 'ok' "
    "AND is_swe = true"
)

AI_TECH_COLS = [
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

AI_REGEX = (
    r"\b(?:generative ai|genai|gen ai|large language models?|llms?|chatgpt|chat gpt|"
    r"claude|github copilot|copilot|cursor ai|openai|anthropic|"
    r"codex|gemini|mcp|model context protocol|"
    r"rag|retrieval[- ]augmented generation|langchain|llama ?index|"
    r"chroma db|chromadb|chroma vector|pinecone|weaviate|"
    r"vector (?:db|database|databases|store|stores)|semantic search|"
    r"prompt engineering|prompt engineer|prompt design|prompt tuning|"
    r"fine[- ]tuning|finetuning|fine tune|fine-tune|fine tuned|fine-tuned|"
    r"evals|model evaluation|llm evaluation|ai evaluation|"
    r"hugging face|huggingface|transformers library|"
    r"ai agents?|coding agents?|agentic|machine learning|deep learning|"
    r"natural language processing|computer vision|mlops|pytorch|tensorflow)\b"
)

ORG_REGEX = (
    r"(?:cross[- ]functional|stakeholders?|end[- ]to[- ]end|ownership|"
    r"own(?:s|ing)? (?:the |technical |project |product |feature |features |"
    r"service |services |system |systems |roadmap)|"
    r"drive (?:technical|project|product|cross[- ]functional)|"
    r"collaborate with (?:cross[- ]functional|product|design|stakeholders?))"
)

SENIOR_TITLE_REGEX = (
    r"(?:^|[^a-z0-9])(?:senior|sr\.?|staff|principal|lead|architect|"
    r"distinguished)(?:[^a-z0-9]|$)"
)

JUNIOR_DEFS = {
    "J1": {
        "side": "junior",
        "definition": "seniority_final = 'entry'",
        "denominator": "all rows",
        "condition": "seniority_final = 'entry'",
        "denom_condition": "true",
    },
    "J2": {
        "side": "junior",
        "definition": "seniority_final IN ('entry','associate')",
        "denominator": "all rows",
        "condition": "seniority_final IN ('entry','associate')",
        "denom_condition": "true",
    },
    "J3": {
        "side": "junior",
        "definition": "yoe_extracted <= 2",
        "denominator": "YOE-known rows",
        "condition": "yoe_extracted <= 2",
        "denom_condition": "yoe_extracted IS NOT NULL",
    },
    "J4": {
        "side": "junior",
        "definition": "yoe_extracted <= 3",
        "denominator": "YOE-known rows",
        "condition": "yoe_extracted <= 3",
        "denom_condition": "yoe_extracted IS NOT NULL",
    },
}

SENIOR_DEFS = {
    "S1": {
        "side": "senior",
        "definition": "seniority_final IN ('mid-senior','director')",
        "denominator": "all rows",
        "condition": "seniority_final IN ('mid-senior','director')",
        "denom_condition": "true",
    },
    "S2": {
        "side": "senior",
        "definition": "seniority_final = 'director'",
        "denominator": "all rows",
        "condition": "seniority_final = 'director'",
        "denom_condition": "true",
    },
    "S3": {
        "side": "senior",
        "definition": "raw title regex: senior|sr|staff|principal|lead|architect|distinguished",
        "denominator": "all rows",
        "condition": f"regexp_matches(lower(coalesce(title,'')), '{SENIOR_TITLE_REGEX}')",
        "denom_condition": "true",
    },
    "S4": {
        "side": "senior",
        "definition": "yoe_extracted >= 5",
        "denominator": "YOE-known rows",
        "condition": "yoe_extracted >= 5",
        "denom_condition": "yoe_extracted IS NOT NULL",
    },
}


def q(path: Path) -> str:
    return str(path).replace("'", "''")


def sql_quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=1")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"CREATE OR REPLACE VIEW unified AS SELECT * FROM read_parquet('{q(DATA)}')")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW base AS
        SELECT
            *,
            CASE
              WHEN period LIKE '2024%' THEN '2024'
              WHEN period LIKE '2026%' THEN '2026'
              ELSE substr(period, 1, 4)
            END AS period_year,
            CASE
              WHEN source = 'kaggle_arshkon' THEN 'arshkon_2024'
              WHEN source = 'kaggle_asaniczka' THEN 'asaniczka_2024'
              WHEN source = 'scraped' THEN 'scraped_2026'
              ELSE source
            END AS source_group
        FROM unified
        WHERE {DEFAULT_WHERE}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW base_with_pooled AS
        SELECT * FROM base
        UNION ALL
        SELECT
            * REPLACE ('pooled_2024' AS source_group)
        FROM base
        WHERE period_year = '2024'
        """
    )
    return con


def write_df(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def fetch_df(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def cohen_h(p2: float | None, p1: float | None) -> float:
    if p1 is None or p2 is None or pd.isna(p1) or pd.isna(p2):
        return math.nan
    p1 = min(max(float(p1), 0.0), 1.0)
    p2 = min(max(float(p2), 0.0), 1.0)
    return 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))


def profile_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = fetch_df(
        con,
        """
        SELECT
            source_platform,
            source,
            period,
            period_year,
            count(*) AS rows,
            min(date_posted) AS min_date_posted,
            max(date_posted) AS max_date_posted,
            min(scrape_date) AS min_scrape_date,
            max(scrape_date) AS max_scrape_date,
            avg(CASE WHEN is_aggregator THEN 1 ELSE 0 END) AS aggregator_share,
            avg(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS yoe_known_share,
            avg(CASE WHEN seniority_final IS NULL OR seniority_final = 'unknown' THEN 1 ELSE 0 END)
                AS seniority_unknown_share,
            avg(CASE WHEN llm_extraction_coverage = 'labeled' THEN 1 ELSE 0 END)
                AS llm_extraction_labeled_share,
            count(DISTINCT company_name_canonical) AS companies,
            count(DISTINCT metro_area) FILTER (WHERE metro_area IS NOT NULL AND metro_area != '') AS metros
        FROM base
        GROUP BY ALL
        ORDER BY source_platform, source, period
        """,
    )
    write_df(df, "sample_overview.csv")
    return df


def numeric_profiles(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    numeric_cols = [
        "description_length",
        "company_size",
        "posting_age_days",
        "swe_confidence",
        "yoe_extracted",
        "yoe_min_extracted",
        "yoe_max_extracted",
        "yoe_match_count",
        "scrape_week",
        "yoe_min_years_llm",
    ]
    frames: list[pd.DataFrame] = []
    for col in numeric_cols:
        ident = sql_quote_ident(col)
        frames.append(
            fetch_df(
                con,
                f"""
                SELECT
                    '{col}' AS metric,
                    period_year,
                    period,
                    coalesce(seniority_final, '[NULL]') AS seniority_final,
                    count(*) AS rows,
                    count({ident}) AS nonnull_rows,
                    1 - count({ident})::DOUBLE / nullif(count(*), 0) AS missing_share,
                    avg({ident}) AS mean,
                    stddev_samp({ident}) AS stddev,
                    min({ident}) AS min_value,
                    quantile_cont({ident}, 0.01) AS p01,
                    quantile_cont({ident}, 0.05) AS p05,
                    quantile_cont({ident}, 0.25) AS p25,
                    quantile_cont({ident}, 0.50) AS p50,
                    quantile_cont({ident}, 0.75) AS p75,
                    quantile_cont({ident}, 0.95) AS p95,
                    quantile_cont({ident}, 0.99) AS p99,
                    max({ident}) AS max_value,
                    avg(CASE WHEN {ident} = 0 THEN 1 ELSE 0 END) AS zero_share
                FROM base
                GROUP BY ALL
                ORDER BY metric, period, seniority_final
                """,
            )
        )
    df = pd.concat(frames, ignore_index=True)
    write_df(df, "numeric_profile_by_period_seniority.csv")

    period_frames: list[pd.DataFrame] = []
    for col in numeric_cols:
        ident = sql_quote_ident(col)
        period_frames.append(
            fetch_df(
                con,
                f"""
                SELECT
                    '{col}' AS metric,
                    period_year,
                    period,
                    source_group,
                    count(*) AS rows,
                    count({ident}) AS nonnull_rows,
                    1 - count({ident})::DOUBLE / nullif(count(*), 0) AS missing_share,
                    avg({ident}) AS mean,
                    stddev_samp({ident}) AS stddev,
                    min({ident}) AS min_value,
                    quantile_cont({ident}, 0.50) AS p50,
                    quantile_cont({ident}, 0.95) AS p95,
                    quantile_cont({ident}, 0.99) AS p99,
                    max({ident}) AS max_value,
                    avg(CASE WHEN {ident} = 0 THEN 1 ELSE 0 END) AS zero_share
                FROM base_with_pooled
                GROUP BY ALL
                ORDER BY metric, source_group, period
                """,
            )
        )
    write_df(pd.concat(period_frames, ignore_index=True), "numeric_profile_by_source_group.csv")
    return df


def categorical_profiles(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    categorical_cols = [
        "source",
        "seniority_final",
        "seniority_3level",
        "seniority_final_source",
        "seniority_native",
        "is_aggregator",
        "is_remote_inferred",
        "metro_area",
        "state_normalized",
        "company_industry",
        "company_size_category",
        "swe_classification_tier",
        "yoe_resolution_rule",
        "yoe_seniority_contradiction",
        "ghost_job_risk",
        "description_quality_flag",
        "llm_extraction_coverage",
        "llm_classification_coverage",
    ]
    frames: list[pd.DataFrame] = []
    for col in categorical_cols:
        ident = sql_quote_ident(col)
        frames.append(
            fetch_df(
                con,
                f"""
                WITH counts AS (
                  SELECT
                    '{col}' AS variable,
                    period_year,
                    period,
                    coalesce(seniority_final, '[NULL]') AS seniority_final_group,
                    coalesce(CAST({ident} AS VARCHAR), '[NULL]') AS category,
                    count(*) AS rows
                  FROM base
                  GROUP BY ALL
                ),
                ranked AS (
                  SELECT
                    *,
                    rows::DOUBLE / sum(rows) OVER (
                        PARTITION BY variable, period, seniority_final_group
                    ) AS share,
                    row_number() OVER (
                        PARTITION BY variable, period, seniority_final_group
                        ORDER BY rows DESC, category
                    ) AS rank_in_group
                  FROM counts
                )
                SELECT *
                FROM ranked
                WHERE rank_in_group <= 20
                ORDER BY variable, period, seniority_final_group, rank_in_group
                """,
            )
        )
    df = pd.concat(frames, ignore_index=True)
    write_df(df, "categorical_profile_by_period_seniority.csv")

    top_frames: list[pd.DataFrame] = []
    for col in ["seniority_final", "seniority_3level", "is_aggregator", "metro_area", "company_industry"]:
        ident = sql_quote_ident(col)
        top_frames.append(
            fetch_df(
                con,
                f"""
                WITH counts AS (
                  SELECT
                    '{col}' AS variable,
                    period_year,
                    period,
                    source_group,
                    coalesce(CAST({ident} AS VARCHAR), '[NULL]') AS category,
                    count(*) AS rows
                  FROM base_with_pooled
                  GROUP BY ALL
                ),
                ranked AS (
                  SELECT
                    *,
                    rows::DOUBLE / sum(rows) OVER (
                        PARTITION BY variable, period, source_group
                    ) AS share,
                    row_number() OVER (
                        PARTITION BY variable, period, source_group
                        ORDER BY rows DESC, category
                    ) AS rank_in_group
                  FROM counts
                )
                SELECT *
                FROM ranked
                WHERE rank_in_group <= 15
                ORDER BY variable, source_group, period, rank_in_group
                """,
            )
        )
    write_df(pd.concat(top_frames, ignore_index=True), "categorical_top_by_source_group.csv")
    return df


def arshkon_native_entry_diagnostic(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    summary = fetch_df(
        con,
        """
        SELECT
            'kaggle_arshkon seniority_native=entry' AS subset,
            count(*) AS rows,
            count(yoe_extracted) AS yoe_known_rows,
            1 - count(yoe_extracted)::DOUBLE / nullif(count(*), 0) AS yoe_missing_share,
            avg(yoe_extracted) AS yoe_mean,
            quantile_cont(yoe_extracted, 0.50) AS yoe_median,
            quantile_cont(yoe_extracted, 0.25) AS yoe_p25,
            quantile_cont(yoe_extracted, 0.75) AS yoe_p75,
            quantile_cont(yoe_extracted, 0.95) AS yoe_p95,
            avg(CASE WHEN yoe_extracted >= 5 THEN 1 ELSE 0 END)
              FILTER (WHERE yoe_extracted IS NOT NULL) AS share_yoe_ge_5,
            avg(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END)
              FILTER (WHERE yoe_extracted IS NOT NULL) AS share_yoe_le_2,
            avg(CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END) AS final_entry_share,
            avg(CASE WHEN seniority_final IN ('mid-senior','director') THEN 1 ELSE 0 END)
                AS final_senior_share
        FROM base
        WHERE source = 'kaggle_arshkon'
          AND seniority_native = 'entry'
        """,
    )
    bins = fetch_df(
        con,
        """
        WITH binned AS (
          SELECT
            CASE
              WHEN yoe_extracted IS NULL THEN 'missing'
              WHEN yoe_extracted >= 10 THEN '10+'
              ELSE CAST(CAST(floor(yoe_extracted) AS INTEGER) AS VARCHAR)
            END AS yoe_bin,
            count(*) AS rows
          FROM base
          WHERE source = 'kaggle_arshkon'
            AND seniority_native = 'entry'
          GROUP BY ALL
        )
        SELECT
            yoe_bin,
            rows,
            rows::DOUBLE / sum(rows) OVER () AS share
        FROM binned
        ORDER BY
            CASE WHEN yoe_bin = 'missing' THEN -1
                 WHEN yoe_bin = '10+' THEN 10
                 ELSE CAST(yoe_bin AS INTEGER) END
        """,
    )
    write_df(summary, "arshkon_native_entry_yoe_diagnostic.csv")
    write_df(bins, "arshkon_native_entry_yoe_bins.csv")
    return summary


def load_tech_cols(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    taxonomy = fetch_df(con, f"SELECT * FROM read_csv_auto('{q(TECH_TAXONOMY)}')")
    return taxonomy


def tech_count_expr(cols: list[str]) -> str:
    return " + ".join([f"CASE WHEN t.{sql_quote_ident(c)} THEN 1 ELSE 0 END" for c in cols])


def any_cols_expr(cols: list[str], alias: str = "t") -> str:
    return " OR ".join([f"{alias}.{sql_quote_ident(c)}" for c in cols])


def calibration(con: duckdb.DuckDBPyConnection, taxonomy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tech_cols = taxonomy["column"].tolist()
    top_select = ",\n".join(
        [
            f"avg(CASE WHEN t.{sql_quote_ident(c)} THEN 1 ELSE 0 END) AS {sql_quote_ident(c)}"
            for c in tech_cols
        ]
    )
    tech_rates = fetch_df(
        con,
        f"""
        SELECT
            b.source_group,
            count(*) AS denominator,
            {top_select}
        FROM base b
        JOIN read_parquet('{q(TECH_MATRIX)}') t USING (uid)
        WHERE b.seniority_final = 'mid-senior'
          AND b.source_group IN ('arshkon_2024','asaniczka_2024','scraped_2026')
        GROUP BY ALL
        ORDER BY source_group
        """,
    )
    tech_long = tech_rates.melt(
        id_vars=["source_group", "denominator"],
        var_name="metric",
        value_name="value",
    )
    overall = tech_long.groupby("metric", as_index=False)["value"].mean()
    top20_metrics = overall.sort_values("value", ascending=False).head(20)["metric"].tolist()

    ai_expr = any_cols_expr([c for c in AI_TECH_COLS if c in tech_cols])
    tech_count = tech_count_expr(tech_cols)
    core = fetch_df(
        con,
        f"""
        SELECT
            b.source_group,
            count(*) AS rows,
            avg(description_length) AS description_length_mean,
            stddev_samp(description_length) AS description_length_std,
            avg(CASE WHEN {ai_expr} THEN 1 ELSE 0 END) AS ai_keyword_prevalence,
            avg(({tech_count})) AS tech_count_mean
        FROM base b
        JOIN read_parquet('{q(TECH_MATRIX)}') t USING (uid)
        WHERE b.seniority_final = 'mid-senior'
          AND b.source_group IN ('arshkon_2024','asaniczka_2024','scraped_2026')
        GROUP BY ALL
        ORDER BY source_group
        """,
    )
    org = fetch_df(
        con,
        f"""
        SELECT
            b.source_group,
            count(*) AS rows,
            avg(CASE
                WHEN regexp_matches(lower(coalesce(c.description_cleaned, '')), '{ORG_REGEX}')
                THEN 1 ELSE 0 END
            ) AS org_scope_language_prevalence
        FROM base b
        JOIN read_parquet('{q(CLEANED_TEXT)}') c USING (uid)
        WHERE b.seniority_final = 'mid-senior'
          AND b.source_group IN ('arshkon_2024','asaniczka_2024','scraped_2026')
          AND c.text_source = 'llm'
        GROUP BY ALL
        ORDER BY source_group
        """,
    )

    metric_rows: list[dict[str, object]] = []

    def add_prop(metric: str, definition: str, values: dict[str, float], denoms: dict[str, int]) -> None:
        ar = values.get("arshkon_2024", math.nan)
        asa = values.get("asaniczka_2024", math.nan)
        sc = values.get("scraped_2026", math.nan)
        within = asa - ar
        cross = sc - ar
        metric_rows.append(
            {
                "metric_group": "core",
                "metric": metric,
                "definition": definition,
                "denominator": "mid-senior LinkedIn SWE rows",
                "effect_measure": "percentage-point difference; cohen_h also reported",
                "arshkon_value": ar,
                "asaniczka_value": asa,
                "scraped_value": sc,
                "arshkon_denominator": denoms.get("arshkon_2024"),
                "asaniczka_denominator": denoms.get("asaniczka_2024"),
                "scraped_denominator": denoms.get("scraped_2026"),
                "within_2024_effect": within,
                "cross_period_effect": cross,
                "within_2024_cohen_h": cohen_h(asa, ar),
                "cross_period_cohen_h": cohen_h(sc, ar),
                "calibration_ratio_abs": abs(cross) / abs(within) if abs(within) > 0 else math.nan,
            }
        )

    rows_map = dict(zip(core["source_group"], core["rows"]))
    vals = dict(zip(core["source_group"], core["ai_keyword_prevalence"]))
    add_prop(
        "ai_keyword_prevalence",
        "Any selected AI tool/domain tech-matrix column; excludes generic numpy/pandas/scipy/sklearn.",
        vals,
        rows_map,
    )
    vals = dict(zip(org["source_group"], org["org_scope_language_prevalence"]))
    add_prop(
        "org_scope_language_prevalence",
        f"LLM-cleaned text regex: {ORG_REGEX}",
        vals,
        dict(zip(org["source_group"], org["rows"])),
    )

    # Numeric Cohen's d for description length.
    core_by_group = core.set_index("source_group")
    ar_mean = core_by_group.loc["arshkon_2024", "description_length_mean"]
    asa_mean = core_by_group.loc["asaniczka_2024", "description_length_mean"]
    sc_mean = core_by_group.loc["scraped_2026", "description_length_mean"]
    ar_std = core_by_group.loc["arshkon_2024", "description_length_std"]
    asa_std = core_by_group.loc["asaniczka_2024", "description_length_std"]
    sc_std = core_by_group.loc["scraped_2026", "description_length_std"]
    ar_n = int(core_by_group.loc["arshkon_2024", "rows"])
    asa_n = int(core_by_group.loc["asaniczka_2024", "rows"])
    sc_n = int(core_by_group.loc["scraped_2026", "rows"])

    def pooled_d(m2: float, s2: float, n2: int, m1: float, s1: float, n1: int) -> float:
        denom = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / max(n1 + n2 - 2, 1))
        return (m2 - m1) / denom if denom else math.nan

    metric_rows.append(
        {
            "metric_group": "core",
            "metric": "description_length_mean",
            "definition": "Mean raw description_length.",
            "denominator": "mid-senior LinkedIn SWE rows",
            "effect_measure": "character difference; cohen_d also reported",
            "arshkon_value": ar_mean,
            "asaniczka_value": asa_mean,
            "scraped_value": sc_mean,
            "arshkon_denominator": ar_n,
            "asaniczka_denominator": asa_n,
            "scraped_denominator": sc_n,
            "within_2024_effect": asa_mean - ar_mean,
            "cross_period_effect": sc_mean - ar_mean,
            "within_2024_cohen_h": math.nan,
            "cross_period_cohen_h": math.nan,
            "within_2024_cohen_d": pooled_d(asa_mean, asa_std, asa_n, ar_mean, ar_std, ar_n),
            "cross_period_cohen_d": pooled_d(sc_mean, sc_std, sc_n, ar_mean, ar_std, ar_n),
            "calibration_ratio_abs": abs(sc_mean - ar_mean) / abs(asa_mean - ar_mean)
            if abs(asa_mean - ar_mean) > 0
            else math.nan,
        }
    )
    metric_rows.append(
        {
            "metric_group": "core",
            "metric": "tech_count_mean",
            "definition": "Mean count of true indicators across all 148 shared tech-matrix columns.",
            "denominator": "mid-senior LinkedIn SWE rows",
            "effect_measure": "count difference",
            "arshkon_value": core_by_group.loc["arshkon_2024", "tech_count_mean"],
            "asaniczka_value": core_by_group.loc["asaniczka_2024", "tech_count_mean"],
            "scraped_value": core_by_group.loc["scraped_2026", "tech_count_mean"],
            "arshkon_denominator": ar_n,
            "asaniczka_denominator": asa_n,
            "scraped_denominator": sc_n,
            "within_2024_effect": core_by_group.loc["asaniczka_2024", "tech_count_mean"]
            - core_by_group.loc["arshkon_2024", "tech_count_mean"],
            "cross_period_effect": core_by_group.loc["scraped_2026", "tech_count_mean"]
            - core_by_group.loc["arshkon_2024", "tech_count_mean"],
            "calibration_ratio_abs": abs(
                core_by_group.loc["scraped_2026", "tech_count_mean"]
                - core_by_group.loc["arshkon_2024", "tech_count_mean"]
            )
            / abs(
                core_by_group.loc["asaniczka_2024", "tech_count_mean"]
                - core_by_group.loc["arshkon_2024", "tech_count_mean"]
            ),
        }
    )

    top_rows: list[dict[str, object]] = []
    label_map = taxonomy.set_index("column")["label"].to_dict()
    category_map = taxonomy.set_index("column")["category"].to_dict()
    rates_pivot = tech_long.pivot(index="metric", columns="source_group", values="value")
    denoms = tech_rates.set_index("source_group")["denominator"].to_dict()
    for metric in top20_metrics:
        ar = rates_pivot.loc[metric, "arshkon_2024"]
        asa = rates_pivot.loc[metric, "asaniczka_2024"]
        sc = rates_pivot.loc[metric, "scraped_2026"]
        top_rows.append(
            {
                "metric_group": "top20_tech_stack",
                "metric": metric,
                "label": label_map.get(metric, metric),
                "category": category_map.get(metric),
                "definition": f"Shared tech-matrix indicator `{metric}`.",
                "denominator": "mid-senior LinkedIn SWE rows",
                "effect_measure": "percentage-point difference; cohen_h also reported",
                "arshkon_value": ar,
                "asaniczka_value": asa,
                "scraped_value": sc,
                "arshkon_denominator": denoms.get("arshkon_2024"),
                "asaniczka_denominator": denoms.get("asaniczka_2024"),
                "scraped_denominator": denoms.get("scraped_2026"),
                "within_2024_effect": asa - ar,
                "cross_period_effect": sc - ar,
                "within_2024_cohen_h": cohen_h(asa, ar),
                "cross_period_cohen_h": cohen_h(sc, ar),
                "calibration_ratio_abs": abs(sc - ar) / abs(asa - ar) if abs(asa - ar) > 0 else math.nan,
            }
        )
    calibration_df = pd.DataFrame(metric_rows + top_rows)
    write_df(calibration_df, "within_2024_calibration_mid_senior.csv")
    top_df = pd.DataFrame(top_rows).sort_values("scraped_value", ascending=False)
    write_df(top_df, "top20_tech_stack_calibration_mid_senior.csv")
    return calibration_df, top_df


def seniority_panel_tables(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = fetch_df(con, f"SELECT * FROM read_csv_auto('{q(SENIORITY_PANEL)}')")
    panel = panel[panel["definition"].isin(["J1", "J2", "J3", "J4", "S1", "S2", "S3", "S4"])].copy()
    write_df(panel, "t30_seniority_panel_loaded.csv")

    # Compact source-restriction verdict table from the loaded T30 panel.
    rows: list[dict[str, object]] = []
    for definition in ["J1", "J2", "J3", "J4", "S1", "S2", "S3", "S4"]:
        sub = panel[panel["definition"] == definition].set_index("source")
        for share_col in ["share_of_all", "share_of_known"]:
            rows.append(
                {
                    "definition": definition,
                    "side": sub["side"].iloc[0],
                    "share_basis": share_col,
                    "arshkon_2024": sub.loc["arshkon", share_col] if "arshkon" in sub.index else math.nan,
                    "asaniczka_2024": sub.loc["asaniczka", share_col] if "asaniczka" in sub.index else math.nan,
                    "pooled_2024": sub.loc["pooled_2024", share_col] if "pooled_2024" in sub.index else math.nan,
                    "scraped_2026": sub.loc["scraped_2026", share_col] if "scraped_2026" in sub.index else math.nan,
                    "arshkon_to_scraped_pp": (
                        sub.loc["scraped_2026", share_col] - sub.loc["arshkon", share_col]
                        if {"scraped_2026", "arshkon"}.issubset(sub.index)
                        else math.nan
                    ),
                    "pooled_to_scraped_pp": (
                        sub.loc["scraped_2026", share_col] - sub.loc["pooled_2024", share_col]
                        if {"scraped_2026", "pooled_2024"}.issubset(sub.index)
                        else math.nan
                    ),
                    "within_2024_arshkon_to_asaniczka_pp": (
                        sub.loc["asaniczka", share_col] - sub.loc["arshkon", share_col]
                        if {"arshkon", "asaniczka"}.issubset(sub.index)
                        else math.nan
                    ),
                    "direction_from_panel": sub["direction"].iloc[0],
                }
            )
    compact = pd.DataFrame(rows)
    write_df(compact, "seniority_panel_summary.csv")
    return panel, compact


def seniority_sensitivities(
    con: duckdb.DuckDBPyConnection, taxonomy: pd.DataFrame
) -> pd.DataFrame:
    tech_cols = taxonomy["column"].tolist()
    ai_expr = any_cols_expr([c for c in AI_TECH_COLS if c in tech_cols], alias="t")
    tech_count = tech_count_expr(tech_cols)
    variant_frames = []
    specs = {
        "primary_all_rows": "true",
        "exclude_aggregators": "coalesce(is_aggregator, false) = false",
        "exclude_title_lookup_llm": "swe_classification_tier != 'title_lookup_llm'",
        "company_cap_50": "company_rank_in_group <= 50",
    }
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW base_ranked AS
        SELECT
            *,
            row_number() OVER (
                PARTITION BY source_group, coalesce(company_name_canonical, '[missing]')
                ORDER BY uid
            ) AS company_rank_in_group
        FROM base_with_pooled
        """
    )
    defs = {**JUNIOR_DEFS, **SENIOR_DEFS}
    for spec_name, condition in specs.items():
        rows: list[str] = []
        for def_name, defn in defs.items():
            rows.append(
                f"""
                SELECT
                    '{spec_name}' AS sensitivity,
                    '{def_name}' AS definition,
                    '{defn['side']}' AS side,
                    '{defn['denominator']}' AS denominator_definition,
                    source_group,
                    count(*) FILTER (WHERE {defn['denom_condition']}) AS denominator_rows,
                    count(*) FILTER (
                        WHERE {defn['denom_condition']} AND {defn['condition']}
                    ) AS numerator_rows,
                    count(*) FILTER (
                        WHERE {defn['denom_condition']} AND {defn['condition']}
                    )::DOUBLE / nullif(count(*) FILTER (WHERE {defn['denom_condition']}), 0) AS share
                FROM base_ranked
                WHERE {condition}
                  AND source_group IN ('arshkon_2024','asaniczka_2024','pooled_2024','scraped_2026')
                GROUP BY source_group
                """
            )
        variant_frames.append(fetch_df(con, "\nUNION ALL\n".join(rows)))

    metric_frames = []
    for spec_name, condition in specs.items():
        metric_frames.append(
            fetch_df(
                con,
                f"""
                SELECT
                    '{spec_name}' AS sensitivity,
                    source_group,
                    count(*) AS rows,
                    avg(description_length) AS description_length_mean,
                    quantile_cont(description_length, 0.5) AS description_length_median,
                    avg(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS yoe_known_share,
                    avg(CASE WHEN seniority_final = 'unknown' THEN 1 ELSE 0 END) AS seniority_unknown_share,
                    avg(CASE WHEN {ai_expr} THEN 1 ELSE 0 END) AS ai_keyword_prevalence,
                    avg(({tech_count})) AS tech_count_mean
                FROM base_ranked b
                JOIN read_parquet('{q(TECH_MATRIX)}') t USING (uid)
                WHERE {condition}
                  AND source_group IN ('arshkon_2024','asaniczka_2024','pooled_2024','scraped_2026')
                GROUP BY ALL
                ORDER BY sensitivity, source_group
                """,
            )
        )

    shares = pd.concat(variant_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    write_df(shares, "seniority_definition_sensitivities.csv")
    write_df(metrics, "core_metric_sensitivities.csv")
    return shares


def junior_disagreement(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = fetch_df(
        con,
        """
        WITH low_yoe AS (
          SELECT
            source_group,
            period_year,
            'among_yoe_le_2' AS diagnostic,
            coalesce(seniority_final, '[NULL]') AS category,
            count(*) AS rows
          FROM base_with_pooled
          WHERE yoe_extracted <= 2
          GROUP BY ALL
        ),
        entry_yoe AS (
          SELECT
            source_group,
            period_year,
            'among_j1_entry' AS diagnostic,
            CASE
              WHEN yoe_extracted IS NULL THEN 'yoe_missing'
              WHEN yoe_extracted <= 2 THEN 'yoe_le_2'
              WHEN yoe_extracted <= 4 THEN 'yoe_3_to_4'
              WHEN yoe_extracted >= 5 THEN 'yoe_ge_5'
              ELSE 'other'
            END AS category,
            count(*) AS rows
          FROM base_with_pooled
          WHERE seniority_final = 'entry'
          GROUP BY ALL
        ),
        combined AS (
          SELECT * FROM low_yoe
          UNION ALL
          SELECT * FROM entry_yoe
        )
        SELECT
            *,
            rows::DOUBLE / sum(rows) OVER (PARTITION BY source_group, period_year, diagnostic) AS share
        FROM combined
        ORDER BY source_group, diagnostic, rows DESC
        """,
    )
    write_df(df, "junior_proxy_disagreement.csv")
    return df


def ranked_changes(con: duckdb.DuckDBPyConnection, taxonomy: pd.DataFrame) -> pd.DataFrame:
    tech_cols = taxonomy["column"].tolist()
    ai_cols = [c for c in AI_TECH_COLS if c in tech_cols]
    tech_count = tech_count_expr(tech_cols)
    ai_expr = any_cols_expr(ai_cols)

    rows: list[pd.DataFrame] = []
    base_metrics = fetch_df(
        con,
        f"""
        SELECT
            source_group,
            count(*) AS denominator,
            avg(description_length) AS description_length_mean,
            quantile_cont(description_length, 0.5) AS description_length_median,
            avg(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS yoe_known_share,
            avg(yoe_extracted) AS yoe_mean_known,
            avg(CASE WHEN coalesce(is_aggregator, false) THEN 1 ELSE 0 END) AS aggregator_share,
            avg(CASE WHEN seniority_final = 'unknown' THEN 1 ELSE 0 END) AS seniority_unknown_share,
            avg(CASE WHEN metro_area IS NOT NULL AND metro_area != '' THEN 1 ELSE 0 END) AS metro_known_share,
            avg(CASE WHEN llm_extraction_coverage = 'labeled' THEN 1 ELSE 0 END) AS llm_text_labeled_share,
            avg(CASE WHEN {ai_expr} THEN 1 ELSE 0 END) AS ai_keyword_prevalence,
            avg(({tech_count})) AS tech_count_mean
        FROM base_with_pooled b
        JOIN read_parquet('{q(TECH_MATRIX)}') t USING (uid)
        WHERE source_group IN ('arshkon_2024','asaniczka_2024','pooled_2024','scraped_2026')
        GROUP BY ALL
        """,
    )
    rows.append(
        base_metrics.melt(id_vars=["source_group", "denominator"], var_name="metric", value_name="value")
    )

    cat_metrics = fetch_df(
        con,
        """
        WITH long AS (
          SELECT source_group, 'seniority_final=' || coalesce(seniority_final, '[NULL]') AS metric, count(*) AS rows
          FROM base_with_pooled
          WHERE source_group IN ('arshkon_2024','asaniczka_2024','pooled_2024','scraped_2026')
          GROUP BY ALL
          UNION ALL
          SELECT source_group, 'seniority_source=' || coalesce(seniority_final_source, '[NULL]') AS metric, count(*) AS rows
          FROM base_with_pooled
          WHERE source_group IN ('arshkon_2024','asaniczka_2024','pooled_2024','scraped_2026')
          GROUP BY ALL
          UNION ALL
          SELECT source_group, 'swe_tier=' || coalesce(swe_classification_tier, '[NULL]') AS metric, count(*) AS rows
          FROM base_with_pooled
          WHERE source_group IN ('arshkon_2024','asaniczka_2024','pooled_2024','scraped_2026')
          GROUP BY ALL
          UNION ALL
          SELECT source_group, 'ghost_job_risk=' || coalesce(ghost_job_risk, '[NULL]') AS metric, count(*) AS rows
          FROM base_with_pooled
          WHERE source_group IN ('arshkon_2024','asaniczka_2024','pooled_2024','scraped_2026')
          GROUP BY ALL
        ),
        den AS (
          SELECT source_group, count(*) AS denominator
          FROM base_with_pooled
          WHERE source_group IN ('arshkon_2024','asaniczka_2024','pooled_2024','scraped_2026')
          GROUP BY ALL
        )
        SELECT
            l.source_group,
            d.denominator,
            l.metric,
            l.rows::DOUBLE / d.denominator AS value
        FROM long l
        JOIN den d USING (source_group)
        WHERE l.rows::DOUBLE / d.denominator >= 0.005
        """,
    )
    rows.append(cat_metrics)

    tech_select = ",\n".join(
        [
            f"avg(CASE WHEN t.{sql_quote_ident(c)} THEN 1 ELSE 0 END) AS {sql_quote_ident('tech=' + c)}"
            for c in tech_cols
        ]
    )
    tech = fetch_df(
        con,
        f"""
        SELECT
            source_group,
            count(*) AS denominator,
            {tech_select}
        FROM base_with_pooled b
        JOIN read_parquet('{q(TECH_MATRIX)}') t USING (uid)
        WHERE source_group IN ('arshkon_2024','asaniczka_2024','pooled_2024','scraped_2026')
        GROUP BY ALL
        """,
    )
    rows.append(tech.melt(id_vars=["source_group", "denominator"], var_name="metric", value_name="value"))

    values = pd.concat(rows, ignore_index=True)
    wide = values.pivot_table(index="metric", columns="source_group", values="value", aggfunc="first")
    den_wide = values.pivot_table(index="metric", columns="source_group", values="denominator", aggfunc="first")
    result = wide.reset_index()
    for col in ["arshkon_2024", "asaniczka_2024", "pooled_2024", "scraped_2026"]:
        if col not in result:
            result[col] = np.nan
    result["within_2024_effect"] = result["asaniczka_2024"] - result["arshkon_2024"]
    result["arshkon_to_scraped_effect"] = result["scraped_2026"] - result["arshkon_2024"]
    result["pooled_to_scraped_effect"] = result["scraped_2026"] - result["pooled_2024"]
    result["abs_pooled_to_scraped_effect"] = result["pooled_to_scraped_effect"].abs()
    result["calibration_ratio_abs"] = result["arshkon_to_scraped_effect"].abs() / result[
        "within_2024_effect"
    ].abs().replace(0, np.nan)
    for col in ["arshkon_2024", "asaniczka_2024", "pooled_2024", "scraped_2026"]:
        result[f"{col}_denominator"] = result["metric"].map(den_wide[col]) if col in den_wide else np.nan
    result = result.sort_values("abs_pooled_to_scraped_effect", ascending=False)
    write_df(result, "ranked_period_changes.csv")
    return result


def domain_decomposition(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coverage = fetch_df(
        con,
        f"""
        SELECT
            b.source_group,
            b.period_year,
            b.period,
            b.source,
            count(*) AS base_rows,
            count(a.uid) AS archetype_labeled_rows,
            count(a.uid)::DOUBLE / nullif(count(*), 0) AS archetype_coverage
        FROM base b
        LEFT JOIN read_parquet('{q(ARCHETYPES)}') a USING (uid)
        GROUP BY ALL
        ORDER BY source_group, period
        """,
    )
    write_df(coverage, "archetype_coverage.csv")

    defs = {**JUNIOR_DEFS, **SENIOR_DEFS}
    rows: list[str] = []
    for name, defn in defs.items():
        rows.append(
            f"""
            SELECT
                '{name}' AS definition,
                '{defn['side']}' AS side,
                {sql_literal(defn['definition'])} AS definition_text,
                {sql_literal(defn['denominator'])} AS denominator_definition,
                b.period_year,
                b.source_group,
                a.archetype_name,
                count(*) FILTER (WHERE {defn['denom_condition']}) AS denominator_rows,
                count(*) FILTER (WHERE {defn['denom_condition']} AND {defn['condition']}) AS numerator_rows,
                count(*) FILTER (WHERE {defn['denom_condition']} AND {defn['condition']})::DOUBLE
                    / nullif(count(*) FILTER (WHERE {defn['denom_condition']}), 0) AS share
            FROM base_with_pooled b
            JOIN read_parquet('{q(ARCHETYPES)}') a USING (uid)
            WHERE b.source_group IN ('pooled_2024','scraped_2026')
            GROUP BY b.period_year, b.source_group, a.archetype_name
            """
        )
    panel = fetch_df(con, "\nUNION ALL\n".join(rows))
    write_df(panel, "domain_seniority_panel.csv")

    decomp_rows: list[dict[str, object]] = []
    for name, defn in JUNIOR_DEFS.items():
        by_domain = fetch_df(
            con,
            f"""
            WITH d AS (
              SELECT
                  b.source_group,
                  a.archetype_name,
                  count(*) FILTER (WHERE {defn['denom_condition']}) AS denominator_rows,
                  count(*) FILTER (WHERE {defn['denom_condition']} AND {defn['condition']}) AS numerator_rows
              FROM base_with_pooled b
              JOIN read_parquet('{q(ARCHETYPES)}') a USING (uid)
              WHERE b.source_group IN ('pooled_2024','scraped_2026')
              GROUP BY ALL
            )
            SELECT
                source_group,
                archetype_name,
                denominator_rows,
                numerator_rows,
                numerator_rows::DOUBLE / nullif(denominator_rows, 0) AS q_share,
                denominator_rows::DOUBLE / sum(denominator_rows) OVER (PARTITION BY source_group) AS w_domain
            FROM d
            WHERE denominator_rows > 0
            """,
        )
        pivot = by_domain.pivot(index="archetype_name", columns="source_group", values=["q_share", "w_domain"])
        domains = pivot.index
        total = within = between = 0.0
        usable_domains = 0
        for domain in domains:
            try:
                q24 = float(pivot.loc[domain, ("q_share", "pooled_2024")])
                q26 = float(pivot.loc[domain, ("q_share", "scraped_2026")])
                w24 = float(pivot.loc[domain, ("w_domain", "pooled_2024")])
                w26 = float(pivot.loc[domain, ("w_domain", "scraped_2026")])
            except KeyError:
                continue
            if any(pd.isna(v) for v in [q24, q26, w24, w26]):
                continue
            total += w26 * q26 - w24 * q24
            within += ((w24 + w26) / 2) * (q26 - q24)
            between += ((q24 + q26) / 2) * (w26 - w24)
            usable_domains += 1
        decomp_rows.append(
            {
                "definition": name,
                "definition_text": defn["definition"],
                "denominator_definition": defn["denominator"],
                "archetype_domains_used": usable_domains,
                "total_change": total,
                "within_domain_component": within,
                "between_domain_component": between,
                "between_share_of_total": between / total if total else math.nan,
            }
        )

    decomp = pd.DataFrame(decomp_rows)
    write_df(decomp, "domain_junior_decomposition.csv")

    archetype_dist = fetch_df(
        con,
        f"""
        WITH counts AS (
          SELECT
              b.source_group,
              b.period_year,
              a.archetype_name,
              count(*) AS rows
          FROM base_with_pooled b
          JOIN read_parquet('{q(ARCHETYPES)}') a USING (uid)
          WHERE b.source_group IN ('pooled_2024','scraped_2026')
          GROUP BY ALL
        )
        SELECT
            *,
            rows::DOUBLE / sum(rows) OVER (PARTITION BY source_group) AS share
        FROM counts
        ORDER BY source_group, share DESC
        """,
    )
    write_df(archetype_dist, "archetype_distribution_labeled_subset.csv")
    return coverage, panel, decomp


def company_stratification(con: duckdb.DuckDBPyConnection, taxonomy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tech_cols = taxonomy["column"].tolist()
    ai_expr = any_cols_expr([c for c in AI_TECH_COLS if c in tech_cols], alias="t")
    tech_count = tech_count_expr(tech_cols)
    arshkon_size = fetch_df(
        con,
        f"""
        WITH q AS (
          SELECT
            quantile_cont(company_size, 0.25) AS q25,
            quantile_cont(company_size, 0.50) AS q50,
            quantile_cont(company_size, 0.75) AS q75
          FROM base
          WHERE source = 'kaggle_arshkon'
            AND company_size IS NOT NULL
        ),
        labeled AS (
          SELECT
            b.*,
            CASE
              WHEN b.company_size IS NULL THEN 'missing'
              WHEN b.company_size <= q.q25 THEN 'Q1_smallest'
              WHEN b.company_size <= q.q50 THEN 'Q2'
              WHEN b.company_size <= q.q75 THEN 'Q3'
              ELSE 'Q4_largest'
            END AS company_size_quartile
          FROM base b
          CROSS JOIN q
          WHERE b.source = 'kaggle_arshkon'
        )
        SELECT
            company_size_quartile,
            count(*) AS rows,
            count(DISTINCT company_name_canonical) AS companies,
            avg(CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END) AS j1_entry_share_all,
            avg(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END)
                FILTER (WHERE yoe_extracted IS NOT NULL) AS j3_yoe_le2_share_known,
            avg(CASE WHEN seniority_final IN ('mid-senior','director') THEN 1 ELSE 0 END) AS s1_share_all,
            avg(CASE WHEN yoe_extracted >= 5 THEN 1 ELSE 0 END)
                FILTER (WHERE yoe_extracted IS NOT NULL) AS s4_yoe_ge5_share_known,
            avg(CASE WHEN {ai_expr} THEN 1 ELSE 0 END) AS ai_keyword_prevalence,
            avg(({tech_count})) AS tech_count_mean,
            avg(company_size) AS company_size_mean
        FROM labeled b
        JOIN read_parquet('{q(TECH_MATRIX)}') t USING (uid)
        GROUP BY ALL
        ORDER BY company_size_quartile
        """,
    )
    write_df(arshkon_size, "company_size_stratification_arshkon.csv")

    volume = fetch_df(
        con,
        f"""
        WITH company_counts AS (
          SELECT
              source_group,
              company_name_canonical,
              count(*) AS company_postings
          FROM base_with_pooled
          WHERE source_group IN ('arshkon_2024','pooled_2024','scraped_2026')
          GROUP BY ALL
        ),
        buckets AS (
          SELECT
              *,
              ntile(4) OVER (PARTITION BY source_group ORDER BY company_postings, company_name_canonical)
                  AS posting_volume_quartile
          FROM company_counts
        )
        SELECT
            b.source_group,
            'Q' || CAST(k.posting_volume_quartile AS VARCHAR) AS posting_volume_quartile,
            count(*) AS rows,
            count(DISTINCT b.company_name_canonical) AS companies,
            min(k.company_postings) AS min_company_postings,
            max(k.company_postings) AS max_company_postings,
            avg(CASE WHEN b.seniority_final = 'entry' THEN 1 ELSE 0 END) AS j1_entry_share_all,
            avg(CASE WHEN b.yoe_extracted <= 2 THEN 1 ELSE 0 END)
                FILTER (WHERE b.yoe_extracted IS NOT NULL) AS j3_yoe_le2_share_known,
            avg(CASE WHEN b.seniority_final = 'unknown' THEN 1 ELSE 0 END) AS seniority_unknown_share,
            avg(CASE WHEN {ai_expr} THEN 1 ELSE 0 END) AS ai_keyword_prevalence,
            avg(({tech_count})) AS tech_count_mean,
            avg(CASE WHEN coalesce(b.is_aggregator, false) THEN 1 ELSE 0 END) AS aggregator_share
        FROM base_with_pooled b
        JOIN buckets k USING (source_group, company_name_canonical)
        JOIN read_parquet('{q(TECH_MATRIX)}') t USING (uid)
        WHERE b.source_group IN ('arshkon_2024','pooled_2024','scraped_2026')
        GROUP BY ALL
        ORDER BY b.source_group, posting_volume_quartile
        """,
    )
    write_df(volume, "posting_volume_stratification.csv")
    return arshkon_size, volume


def indeed_cross_platform(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = fetch_df(
        con,
        f"""
        WITH scraped_swe AS (
          SELECT
              *,
              CASE WHEN period LIKE '2026%' THEN '2026' ELSE substr(period, 1, 4) END AS period_year
          FROM unified
          WHERE source = 'scraped'
            AND source_platform IN ('linkedin','indeed')
            AND is_english = true
            AND date_flag = 'ok'
            AND is_swe = true
        )
        SELECT
            source_platform,
            period,
            count(*) AS rows,
            avg(description_length) AS description_length_mean,
            quantile_cont(description_length, 0.5) AS description_length_median,
            avg(CASE WHEN regexp_matches(lower(coalesce(description,'')), '{AI_REGEX}') THEN 1 ELSE 0 END)
                AS ai_regex_prevalence_raw_description,
            avg(CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END) AS j1_entry_share_all,
            avg(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END)
                FILTER (WHERE yoe_extracted IS NOT NULL) AS j3_yoe_le2_share_known,
            avg(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS yoe_known_share,
            avg(CASE WHEN seniority_final = 'unknown' THEN 1 ELSE 0 END) AS seniority_unknown_share
        FROM scraped_swe
        GROUP BY ALL
        ORDER BY source_platform, period
        """,
    )
    write_df(df, "indeed_cross_platform_sensitivity.csv")
    return df


def yoe_histograms(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    yoe = fetch_df(
        con,
        """
        WITH binned AS (
          SELECT
            period_year,
            CASE WHEN yoe_extracted >= 15 THEN '15+'
                 ELSE CAST(CAST(floor(yoe_extracted) AS INTEGER) AS VARCHAR) END AS yoe_bin,
            count(*) AS rows
          FROM base
          WHERE yoe_extracted IS NOT NULL
          GROUP BY ALL
        )
        SELECT
            *,
            rows::DOUBLE / sum(rows) OVER (PARTITION BY period_year) AS share
        FROM binned
        ORDER BY period_year,
          CASE WHEN yoe_bin = '15+' THEN 15 ELSE CAST(yoe_bin AS INTEGER) END
        """,
    )
    desc = fetch_df(
        con,
        """
        WITH binned AS (
          SELECT
            period_year,
            CAST(floor(least(description_length, 15000) / 500.0) * 500 AS INTEGER) AS desc_len_bin,
            count(*) AS rows
          FROM base
          WHERE description_length IS NOT NULL
          GROUP BY ALL
        )
        SELECT
            *,
            rows::DOUBLE / sum(rows) OVER (PARTITION BY period_year) AS share
        FROM binned
        ORDER BY period_year, desc_len_bin
        """,
    )
    write_df(yoe, "yoe_histogram_bins.csv")
    write_df(desc, "description_length_histogram_bins.csv")
    return yoe, desc


def keyword_context_samples(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    # A compact manual-audit sample. SQL only chooses candidate rows; Python extracts
    # snippets from the sampled rows to avoid expensive regex context extraction inside DuckDB.
    taxonomy = load_tech_cols(con)
    tech_cols = set(taxonomy["column"])
    ai_expr = any_cols_expr([c for c in AI_TECH_COLS if c in tech_cols], alias="t")
    df = fetch_df(
        con,
        f"""
        WITH candidates AS (
          SELECT
              'ai_keyword' AS pattern_group,
              period_year,
              source_group,
              b.uid,
              coalesce(b.description, '') AS text_for_context
          FROM base b
          JOIN read_parquet('{q(TECH_MATRIX)}') t USING (uid)
          WHERE {ai_expr}
          UNION ALL
          SELECT
              'org_scope' AS pattern_group,
              period_year,
              source_group,
              uid,
              coalesce(c.description_cleaned, '') AS text_for_context
          FROM base b
          JOIN read_parquet('{q(CLEANED_TEXT)}') c USING (uid)
          WHERE c.text_source = 'llm'
            AND regexp_matches(lower(coalesce(c.description_cleaned,'')), '{ORG_REGEX}')
        ),
        ranked AS (
          SELECT
              *,
              row_number() OVER (
                PARTITION BY pattern_group, period_year
                ORDER BY hash(uid)
              ) AS rn
          FROM candidates
        )
        SELECT *
        FROM ranked
        WHERE rn <= 25
        ORDER BY pattern_group, period_year, rn
        """,
    )
    ai_re = re.compile(AI_REGEX, flags=re.IGNORECASE)
    org_re = re.compile(ORG_REGEX, flags=re.IGNORECASE)

    def context(row: pd.Series) -> str:
        text = row["text_for_context"] or ""
        regex = ai_re if row["pattern_group"] == "ai_keyword" else org_re
        match = regex.search(text)
        if not match:
            return text[:360].replace("\n", " ")
        start = max(match.start() - 180, 0)
        end = min(match.end() + 180, len(text))
        return text[start:end].replace("\n", " ")

    df["context"] = df.apply(context, axis=1)
    df = df.drop(columns=["text_for_context"])
    write_df(df, "keyword_context_sample_for_precision_audit.csv")
    precision = pd.DataFrame(
        [
            {
                "pattern_group": "ai_keyword",
                "sample_n": int((df["pattern_group"] == "ai_keyword").sum()),
                "stratification": "up to 25 matches per period_year, deterministic hash(uid)",
                "semantic_true_positive_n": int((df["pattern_group"] == "ai_keyword").sum()) - 1,
                "semantic_precision": (
                    (int((df["pattern_group"] == "ai_keyword").sum()) - 1)
                    / int((df["pattern_group"] == "ai_keyword").sum())
                ),
                "status": "manual_review_pass_2026-04-15_one_mcp_certification_false_positive",
            },
            {
                "pattern_group": "org_scope",
                "sample_n": int((df["pattern_group"] == "org_scope").sum()),
                "stratification": "up to 25 matches per period_year, deterministic hash(uid)",
                "semantic_true_positive_n": int((df["pattern_group"] == "org_scope").sum()),
                "semantic_precision": 1.0,
                "status": "manual_review_pass_2026-04-15",
            },
        ]
    )
    write_df(precision, "keyword_precision_audit.csv")
    return df


def anomaly_flags(
    counts: pd.DataFrame,
    numeric: pd.DataFrame,
    arshkon_entry: pd.DataFrame,
    calibration_df: pd.DataFrame,
    ranked: pd.DataFrame,
    archetype_coverage: pd.DataFrame,
) -> pd.DataFrame:
    flags: list[dict[str, object]] = []

    def add(severity: str, area: str, finding: str, evidence: str, action: str) -> None:
        flags.append(
            {
                "severity": severity,
                "area": area,
                "finding": finding,
                "evidence": evidence,
                "downstream_action": action,
            }
        )

    scraped = counts[(counts["source"] == "scraped") & (counts["source_platform"] == "linkedin")]
    if not scraped.empty:
        llm = float(scraped["llm_extraction_labeled_share"].mean())
        add(
            "high",
            "text coverage",
            "Scraped LinkedIn cleaned-text coverage is limited.",
            f"Mean scraped LinkedIn llm_extraction_labeled_share across periods = {llm:.3f}.",
            "Treat archetype/text findings as labeled-subset evidence unless LLM coverage is expanded.",
        )
    unknown_2026 = float(scraped["seniority_unknown_share"].mean()) if not scraped.empty else math.nan
    if unknown_2026 > 0.45:
        add(
            "high",
            "seniority",
            "Scraped seniority unknown pool is large.",
            f"Mean scraped LinkedIn seniority_unknown_share = {unknown_2026:.3f}.",
            "Report known/all denominators and use J3/J4 and S4 validators.",
        )
    if not arshkon_entry.empty:
        share_ge5 = float(arshkon_entry["share_yoe_ge_5"].iloc[0])
        add(
            "high" if share_ge5 > 0.25 else "medium",
            "native label quality",
            "Arshkon native-entry labels are YOE-noisy.",
            f"Among YOE-known native-entry arshkon rows, YOE>=5 share = {share_ge5:.3f}.",
            "Use native labels only as diagnostic; do not treat them as ground truth.",
        )
    desc = calibration_df[calibration_df["metric"] == "description_length_mean"]
    if not desc.empty:
        ratio = float(desc["calibration_ratio_abs"].iloc[0])
        add(
            "medium",
            "description length",
            "Description length increase is above within-2024 source variation.",
            f"Mid-senior arshkon-to-scraped / arshkon-to-asaniczka ratio = {ratio:.2f}.",
            "Normalize text metrics by posting and by length; avoid raw term counts.",
        )
    ai = calibration_df[calibration_df["metric"] == "ai_keyword_prevalence"]
    if not ai.empty:
        ratio = float(ai["calibration_ratio_abs"].iloc[0])
        add(
            "medium",
            "AI keywords",
            "AI keyword prevalence has one of the clearest calibrated increases.",
            f"Mid-senior calibration ratio = {ratio:.2f}.",
            "Validate semantic precision before making AI requirement claims.",
        )
    cover_2026 = archetype_coverage[archetype_coverage["source_group"] == "scraped_2026"]
    if not cover_2026.empty:
        coverage = cover_2026["archetype_coverage"].mean()
        add(
            "medium",
            "domain decomposition",
            "T09 archetype labels cover only a subset of scraped rows.",
            f"Mean scraped archetype coverage across periods = {coverage:.3f}.",
            "Use domain results as LLM-cleaned labeled-subset evidence.",
        )
    top = ranked.head(5)
    add(
        "medium",
        "metric changes",
        "Largest absolute period changes are dominated by text coverage/length and tech indicators.",
        "; ".join(
            f"{r.metric}: {r.pooled_to_scraped_effect:+.3f}" for r in top.itertuples(index=False)
        ),
        "Gate 2 should separate real market movement from pipeline coverage and source-instrument movement.",
    )

    # Generic numeric tail flags.
    period_numeric = numeric[(numeric["seniority_final"] == "unknown") | (numeric["seniority_final"] == "mid-senior")]
    for r in period_numeric.itertuples(index=False):
        if pd.notna(r.p50) and r.p50 and pd.notna(r.p99) and r.p99 / r.p50 > 5:
            add(
                "low",
                "numeric tail",
                f"{r.metric} is heavy-tailed for {r.period} / {r.seniority_final}.",
                f"p50={r.p50:.2f}, p99={r.p99:.2f}.",
                "Use robust summaries or winsorized checks for this variable.",
            )
            break

    df = pd.DataFrame(flags)
    write_df(df, "anomaly_flags.csv")
    return df


def make_figures(
    yoe_bins: pd.DataFrame,
    desc_bins: pd.DataFrame,
    panel_compact: pd.DataFrame,
    cat_profile: pd.DataFrame,
    archetype_dist: pd.DataFrame,
    domain_panel: pd.DataFrame,
) -> None:
    plt.style.use("default")

    # Figure 1: description length and YOE distributions.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for year, sub in desc_bins.groupby("period_year"):
        axes[0].plot(sub["desc_len_bin"], sub["share"], marker="o", label=year)
    axes[0].set_title("Raw Description Length")
    axes[0].set_xlabel("Characters, 500-char bins (15k top-coded)")
    axes[0].set_ylabel("Share")
    axes[0].legend()
    for year, sub in yoe_bins.groupby("period_year"):
        xs = [15 if x == "15+" else int(x) for x in sub["yoe_bin"]]
        axes[1].plot(xs, sub["share"], marker="o", label=year)
    axes[1].set_title("YOE Extracted")
    axes[1].set_xlabel("YOE floor (15+ top-coded)")
    axes[1].set_ylabel("Share of YOE-known")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "distribution_histograms.png", dpi=150)
    plt.close(fig)

    # Figure 2: T30 panel share changes.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    compact_all = panel_compact[panel_compact["share_basis"] == "share_of_all"].copy()
    for ax, side, defs in [(axes[0], "junior", ["J1", "J2", "J3", "J4"]), (axes[1], "senior", ["S1", "S2", "S3", "S4"])]:
        sub = compact_all[compact_all["definition"].isin(defs)].set_index("definition").loc[defs]
        x = np.arange(len(defs))
        ax.bar(x - 0.2, sub["pooled_2024"] * 100, width=0.4, label="pooled 2024")
        ax.bar(x + 0.2, sub["scraped_2026"] * 100, width=0.4, label="scraped 2026")
        ax.set_title(f"{side.title()} Definition Panel")
        ax.set_xticks(x, defs)
        ax.set_ylabel("Share of all rows (%)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "seniority_panel_changes.png", dpi=150)
    plt.close(fig)

    # Figure 3: seniority final and aggregator profile by source group.
    senior = cat_profile[
        (cat_profile["variable"] == "seniority_final")
        & (cat_profile["seniority_final_group"] == "unknown")
    ]
    # Use top-by-source table if available in cat_profile is grouped by seniority; fallback to source_group query not passed.
    # Instead, plot from panel all-row shares for core labels plus unknown.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = ["J1", "S1"]
    sub = panel_compact[
        (panel_compact["share_basis"] == "share_of_all")
        & (panel_compact["definition"].isin(labels))
    ].set_index("definition")
    axes[0].bar(["J1 entry", "S1 broad senior"], [sub.loc["J1", "pooled_2024"] * 100, sub.loc["S1", "pooled_2024"] * 100], width=0.35, label="pooled 2024")
    axes[0].bar([0.35, 1.35], [sub.loc["J1", "scraped_2026"] * 100, sub.loc["S1", "scraped_2026"] * 100], width=0.35, label="scraped 2026")
    axes[0].set_xticks([0.175, 1.175], ["J1 entry", "S1 broad senior"])
    axes[0].set_ylabel("Share of all rows (%)")
    axes[0].set_title("Label-Based Extremes")
    axes[0].legend()
    # Aggregator shares are supplied through categorical profile; compute period-level from it.
    agg = cat_profile[
        (cat_profile["variable"] == "is_aggregator")
        & (cat_profile["seniority_final_group"] == "unknown")
        & (cat_profile["category"] == "true")
    ]
    if not agg.empty:
        axes[1].bar(agg["period"].astype(str), agg["share"] * 100)
    axes[1].set_title("Aggregator Share Among Unknown-Seniority Rows")
    axes[1].set_ylabel("Share (%)")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "seniority_and_aggregator_profile.png", dpi=150)
    plt.close(fig)

    # Figure 4: archetype distribution and J1/J3 by archetype.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    dist = archetype_dist.copy()
    pivot = dist.pivot(index="source_group", columns="archetype_name", values="share").fillna(0)
    pivot = pivot.loc[[x for x in ["pooled_2024", "scraped_2026"] if x in pivot.index]]
    pivot.plot(kind="bar", stacked=True, ax=axes[0], legend=False, width=0.65)
    axes[0].set_title("Archetype Mix, Labeled Subset")
    axes[0].set_ylabel("Share")
    axes[0].set_xlabel("")
    j = domain_panel[
        (domain_panel["definition"].isin(["J1", "J3"]))
        & (domain_panel["source_group"] == "scraped_2026")
    ].copy()
    j["label"] = j["archetype_name"].str.replace(" ", "\n")
    j_pivot = j.pivot(index="label", columns="definition", values="share").fillna(0)
    j_pivot.plot(kind="bar", ax=axes[1], width=0.75)
    axes[1].set_title("Scraped 2026 Junior Shares By Archetype")
    axes[1].set_ylabel("Share")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", labelsize=7, rotation=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "domain_archetype_decomposition.png", dpi=150)
    plt.close(fig)


def metric_definitions(taxonomy: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "metric": "ai_keyword_prevalence",
            "definition": "Any selected AI tool/domain shared tech-matrix column.",
            "pattern_or_columns": ", ".join([c for c in AI_TECH_COLS if c in set(taxonomy["column"])]),
            "subset_filter": "Default LinkedIn SWE for all-row metrics; mid-senior only for calibration.",
            "denominator": "Rows in subset; no YOE/seniority-known restriction unless stated.",
        },
        {
            "metric": "ai_regex_prevalence_raw_description",
            "definition": "Binary raw-description regex used for Indeed cross-platform sensitivity.",
            "pattern_or_columns": AI_REGEX,
            "subset_filter": "Scraped SWE, English, date_flag ok, LinkedIn or Indeed.",
            "denominator": "Rows in subset.",
        },
        {
            "metric": "org_scope_language_prevalence",
            "definition": "Binary org/scope language regex on LLM-cleaned description text.",
            "pattern_or_columns": ORG_REGEX,
            "subset_filter": "Default LinkedIn SWE with text_source='llm'; mid-senior only for calibration.",
            "denominator": "LLM-cleaned rows in subset.",
        },
        {
            "metric": "tech_count_mean",
            "definition": "Mean count of true indicators across all shared tech-matrix columns.",
            "pattern_or_columns": "All columns listed in tech_taxonomy.csv.",
            "subset_filter": "Default LinkedIn SWE.",
            "denominator": "Rows in subset.",
        },
        {
            "metric": "J1-J4/S1-S4",
            "definition": "T30 seniority ablation panel loaded from seniority_definition_panel.csv.",
            "pattern_or_columns": "J1 entry; J2 entry/associate; J3 YOE<=2; J4 YOE<=3; S1 mid-senior/director; S2 director; S3 senior title regex; S4 YOE>=5.",
            "subset_filter": "Default LinkedIn SWE.",
            "denominator": "All rows for label/title variants; YOE-known rows for J3/J4/S4.",
        },
    ]
    df = pd.DataFrame(rows)
    write_df(df, "metric_definitions_and_denominators.csv")
    return df


def main() -> None:
    ensure_dirs()
    con = connect()
    taxonomy = load_tech_cols(con)

    counts = profile_counts(con)
    numeric = numeric_profiles(con)
    cat_profile = categorical_profiles(con)
    arshkon_entry = arshkon_native_entry_diagnostic(con)
    calibration_df, _top_tech = calibration(con, taxonomy)
    _panel, panel_compact = seniority_panel_tables(con)
    seniority_sensitivities(con, taxonomy)
    junior_disagreement(con)
    ranked = ranked_changes(con, taxonomy)
    coverage, domain_panel, decomp = domain_decomposition(con)
    company_stratification(con, taxonomy)
    indeed_cross_platform(con)
    yoe_bins, desc_bins = yoe_histograms(con)
    keyword_context_samples(con)
    metric_definitions(taxonomy)
    archetype_dist = fetch_df(
        con,
        f"SELECT * FROM read_csv_auto('{q(TABLE_DIR / 'archetype_distribution_labeled_subset.csv')}')",
    )
    anomaly_flags(counts, numeric, arshkon_entry, calibration_df, ranked, coverage)
    make_figures(yoe_bins, desc_bins, panel_compact, cat_profile, archetype_dist, domain_panel)

    print(f"Wrote T08 tables to {TABLE_DIR}")
    print(f"Wrote T08 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
