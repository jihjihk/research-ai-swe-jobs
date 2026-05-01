#!/usr/bin/env python3
"""Generate missing T06 outputs with DuckDB-only aggregate queries.

This companion script avoids the pandas-heavy T06 script that can exhaust RAM.
It scans `data/unified.parquet` through DuckDB under a 4GB memory cap, stores
only compact aggregate tables in DuckDB, and writes small CSV deliverables.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
TABLE_DIR = ROOT / "exploration" / "tables" / "T06"
SHARED_DIR = ROOT / "exploration" / "artifacts" / "shared"
PANEL = SHARED_DIR / "seniority_definition_panel.csv"
SPECIALIST_PATH = SHARED_DIR / "entry_specialist_employers.csv"
SUMMARY_PATH = TABLE_DIR / "summary.json"

SOURCE_SORT = """
CASE source_key
  WHEN 'kaggle_arshkon' THEN 1
  WHEN 'kaggle_asaniczka' THEN 2
  WHEN 'scraped_linkedin' THEN 3
  WHEN 'scraped_indeed' THEN 4
  ELSE 99
END
"""

AI_REGEX = (
    r"\b(ai|a\.i\.|artificial intelligence|machine learning|ml|llm|large language model|"
    r"generative ai|genai|gpt|chatgpt|openai|anthropic|claude|copilot|cursor|rag|"
    r"agentic|ai agent|prompt engineering)\b"
)

TECH_PATTERNS = [
    r"\bpython\b",
    r"\bjava\b",
    r"\bjavascript\b|\bjs\b",
    r"\btypescript\b|\bts\b",
    r"\breact\b|\breactjs\b",
    r"\bangular\b",
    r"\bvue\b|\bvuejs\b",
    r"\bnode\.?js\b",
    r"(^|[^a-z0-9])c#([^a-z0-9]|$)|\bc sharp\b",
    r"(^|[^a-z0-9])c\+\+([^a-z0-9]|$)",
    r"\bgolang\b",
    r"\brust\b",
    r"\bruby\b",
    r"\bphp\b",
    r"\bswift\b",
    r"\bkotlin\b",
    r"\bsql\b",
    r"\bpostgres\b|\bpostgresql\b",
    r"\bmysql\b",
    r"\bmongodb\b|\bmongo\b",
    r"\baws\b|amazon web services",
    r"\bazure\b",
    r"\bgcp\b|google cloud",
    r"\bdocker\b",
    r"\bkubernetes\b|\bk8s\b",
    r"\bterraform\b",
    r"\bci/cd\b|\bcontinuous integration\b|\bcontinuous delivery\b",
    r"\bgit\b|\bgithub\b|\bgitlab\b",
    r"\blinux\b",
    r"\bspark\b|\bapache spark\b",
    r"\bkafka\b",
    r"\bredis\b",
    r"\bgraphql\b",
    r"\brestful\b|\brest api\b|\brest\b",
    r"\bmicroservices\b|\bmicroservice\b",
    r"\bspring boot\b|\bspring\b",
    r"\bdjango\b",
    r"\bflask\b",
    r"(^|[^a-z0-9])\.net([^a-z0-9]|$)|\bdotnet\b",
]


def sql_lit(value: str) -> str:
    return value.replace("'", "''")


def copy_to_csv(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    con.execute(f"COPY ({query}) TO '{path.as_posix()}' (HEADER, DELIMITER ',')")


def tech_count_sql(text_expr: str) -> str:
    parts = [
        f"CASE WHEN regexp_matches({text_expr}, '{sql_lit(pattern)}') THEN 1 ELSE 0 END"
        for pattern in TECH_PATTERNS
    ]
    return " + ".join(parts)


def category_case(company_expr: str) -> str:
    return f"""
    CASE
      WHEN regexp_matches(lower({company_expr}), '(revature|synergisticit|skillstorm|mthree|accenture|deloitte|cognizant|infosys|wipro|tata consultancy services|capgemini|five cubes|aaratech|aara technologies|aaratechnologies|impetusit|emonics|innovit|guidehouse|tech consulting)')
        THEN 'bulk-posting consulting'
      WHEN regexp_matches(lower({company_expr}), '(genesis10|dice|lensa|robert half|motion recruitment|teksystems|tek systems|insight global|kforce|apex systems|radley james|skyrocket ventures|staff|staffing|recruit|talent|randstad|cybercoders|jobot|experis|judge group|collabera|hire)')
        THEN 'staffing firm'
      WHEN regexp_matches(lower({company_expr}), '(handshake|ripplematch|wayup|college|university)')
        THEN 'college-jobsite intermediary'
      WHEN regexp_matches(lower({company_expr}), '(amazon|amazon web services|google|microsoft|meta|apple|oracle|ibm|nvidia)')
        THEN 'tech-giant intern pipeline'
      WHEN regexp_matches(lower({company_expr}), '(consulting|consultants|solutions|systems integrator)')
        THEN 'bulk-posting consulting'
      ELSE 'direct employer'
    END
    """


def setup_base(con: duckdb.DuckDBPyConnection) -> int:
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=1")
    con.execute("PRAGMA preserve_insertion_order=false")

    panel_rows = con.execute(
        f"""
        SELECT count(*)
        FROM read_csv_auto('{PANEL.as_posix()}')
        WHERE definition IN ('J1', 'J2', 'J3', 'J4')
        """
    ).fetchone()[0]
    if panel_rows < 12:
        raise RuntimeError(f"T30 seniority panel appears incomplete: {panel_rows} J1-J4 rows")

    lower_desc = "lower(coalesce(description, ''))"
    source_key = "CASE WHEN source = 'scraped' THEN source || '_' || source_platform ELSE source END"
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW t06_base AS
        SELECT
          uid,
          {source_key} AS source_key,
          source,
          source_platform,
          company_name_canonical,
          company_industry,
          coalesce(is_aggregator, false) AS is_aggregator,
          description_hash,
          description_length,
          seniority_final,
          yoe_extracted,
          CASE WHEN seniority_final IS NOT NULL AND seniority_final <> 'unknown' THEN 1 ELSE 0 END AS known_seniority,
          CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END AS j1_entry,
          CASE WHEN seniority_final IN ('entry', 'associate') THEN 1 ELSE 0 END AS j2_entry_associate,
          CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END AS yoe_known,
          CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END AS j3_yoe_le2,
          CASE WHEN yoe_extracted <= 3 THEN 1 ELSE 0 END AS j4_yoe_le3,
          CASE WHEN regexp_matches({lower_desc}, '{sql_lit(AI_REGEX)}') THEN 1 ELSE 0 END AS ai_mention,
          ({tech_count_sql(lower_desc)})::INTEGER AS tech_count
        FROM read_parquet('{DATA.as_posix()}')
        WHERE is_english = true
          AND date_flag = 'ok'
          AND is_swe = true
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE t06_company_metrics AS
        WITH grouped AS (
          SELECT
            source_key,
            company_name_canonical,
            count(*) AS postings,
            sum(CASE WHEN is_aggregator THEN 1 ELSE 0 END) AS aggregator_postings,
            sum(known_seniority) AS known_seniority_n,
            sum(j1_entry) AS j1_entry_n,
            sum(j2_entry_associate) AS j2_entry_associate_n,
            sum(yoe_known) AS yoe_known_n,
            sum(j3_yoe_le2) AS j3_yoe_le2_n,
            sum(j4_yoe_le3) AS j4_yoe_le3_n,
            sum(CASE WHEN description_length IS NOT NULL THEN 1 ELSE 0 END) AS description_length_known_n,
            avg(description_length) AS mean_description_length,
            avg(ai_mention::DOUBLE) AS ai_mention_share,
            avg(tech_count::DOUBLE) AS mean_tech_count
          FROM t06_base
          WHERE company_name_canonical IS NOT NULL
            AND company_name_canonical <> ''
          GROUP BY 1, 2
        )
        SELECT
          *,
          aggregator_postings > 0 AS any_aggregator,
          aggregator_postings::DOUBLE / postings AS aggregator_posting_share,
          j1_entry_n::DOUBLE / postings AS j1_share_all,
          j2_entry_associate_n::DOUBLE / postings AS j2_share_all,
          j1_entry_n::DOUBLE / nullif(known_seniority_n, 0) AS j1_share_known_seniority,
          j2_entry_associate_n::DOUBLE / nullif(known_seniority_n, 0) AS j2_share_known_seniority,
          j3_yoe_le2_n::DOUBLE / nullif(yoe_known_n, 0) AS j3_share_yoe_known,
          j4_yoe_le3_n::DOUBLE / nullif(yoe_known_n, 0) AS j4_share_yoe_known
        FROM grouped
        """
    )
    return int(panel_rows)


def write_duplicate_audit(con: duckdb.DuckDBPyConnection) -> None:
    copy_to_csv(
        con,
        f"""
        WITH company_hash AS (
          SELECT source_key, company_name_canonical, description_hash, count(*) AS hash_n
          FROM t06_base
          WHERE company_name_canonical IS NOT NULL
            AND company_name_canonical <> ''
            AND description_hash IS NOT NULL
          GROUP BY 1, 2, 3
        ),
        company_stats AS (
          SELECT
            source_key,
            company_name_canonical,
            sum(hash_n) AS postings,
            count(*) AS distinct_description_hashes,
            sum(hash_n)::DOUBLE / count(*) AS postings_per_distinct_hash,
            max(hash_n) AS largest_hash_n,
            max(hash_n)::DOUBLE / sum(hash_n) AS largest_hash_share
          FROM company_hash
          GROUP BY 1, 2
          HAVING sum(hash_n) >= 5
        ),
        ranked AS (
          SELECT
            *,
            row_number() OVER (
              PARTITION BY source_key
              ORDER BY postings_per_distinct_hash DESC, postings DESC, company_name_canonical
            ) AS source_rank
          FROM company_stats
        )
        SELECT
          source_key,
          source_rank,
          company_name_canonical,
          postings,
          distinct_description_hashes,
          postings_per_distinct_hash,
          largest_hash_n,
          largest_hash_share
        FROM ranked
        WHERE source_rank <= 10
        ORDER BY {SOURCE_SORT}, source_rank
        """,
        TABLE_DIR / "duplicate_template_top10_by_source.csv",
    )


def write_company_entry_outputs(con: duckdb.DuckDBPyConnection) -> None:
    copy_to_csv(
        con,
        f"""
        SELECT *
        FROM t06_company_metrics
        ORDER BY {SOURCE_SORT}, postings DESC, company_name_canonical
        """,
        TABLE_DIR / "company_entry_metrics.csv",
    )

    copy_to_csv(
        con,
        f"""
        SELECT
          source_key,
          count(*) AS companies_total,
          sum(CASE WHEN j1_entry_n > 0 THEN 1 ELSE 0 END) AS companies_with_any_j1_entry,
          avg(CASE WHEN j1_entry_n > 0 THEN 1.0 ELSE 0.0 END) AS companies_with_any_j1_entry_share,
          sum(CASE WHEN j3_yoe_le2_n > 0 THEN 1 ELSE 0 END) AS companies_with_any_yoe_le2,
          avg(CASE WHEN j3_yoe_le2_n > 0 THEN 1.0 ELSE 0.0 END) AS companies_with_any_yoe_le2_share,
          sum(CASE WHEN postings >= 5 THEN 1 ELSE 0 END) AS companies_ge5,
          sum(CASE WHEN postings >= 5 AND j1_entry_n = 0 THEN 1 ELSE 0 END) AS ge5_zero_j1_entry,
          sum(CASE WHEN postings >= 5 AND j1_entry_n = 0 THEN 1.0 ELSE 0.0 END)
            / nullif(sum(CASE WHEN postings >= 5 THEN 1 ELSE 0 END), 0) AS ge5_zero_j1_entry_share,
          sum(CASE WHEN postings >= 5 AND j3_yoe_le2_n = 0 THEN 1 ELSE 0 END) AS ge5_zero_yoe_le2,
          sum(CASE WHEN postings >= 5 AND j3_yoe_le2_n = 0 THEN 1.0 ELSE 0.0 END)
            / nullif(sum(CASE WHEN postings >= 5 THEN 1 ELSE 0 END), 0) AS ge5_zero_yoe_le2_share,
          quantile_cont(j1_share_all, 0.5) FILTER (WHERE j1_entry_n > 0) AS j1_entry_poster_share_all_median,
          quantile_cont(j1_share_all, 0.9) FILTER (WHERE j1_entry_n > 0) AS j1_entry_poster_share_all_p90,
          max(j1_share_all) FILTER (WHERE j1_entry_n > 0) AS j1_entry_poster_share_all_max,
          quantile_cont(j1_share_known_seniority, 0.5) FILTER (WHERE j1_entry_n > 0 AND known_seniority_n > 0) AS j1_entry_poster_share_known_median,
          quantile_cont(j1_share_known_seniority, 0.9) FILTER (WHERE j1_entry_n > 0 AND known_seniority_n > 0) AS j1_entry_poster_share_known_p90,
          quantile_cont(j3_share_yoe_known, 0.5) FILTER (WHERE j3_yoe_le2_n > 0 AND yoe_known_n > 0) AS yoe_le2_poster_share_yoe_known_median,
          quantile_cont(j3_share_yoe_known, 0.9) FILTER (WHERE j3_yoe_le2_n > 0 AND yoe_known_n > 0) AS yoe_le2_poster_share_yoe_known_p90,
          max(j3_share_yoe_known) FILTER (WHERE j3_yoe_le2_n > 0 AND yoe_known_n > 0) AS yoe_le2_poster_share_yoe_known_max
        FROM t06_company_metrics
        GROUP BY 1
        ORDER BY {SOURCE_SORT}
        """,
        TABLE_DIR / "entry_posting_concentration.csv",
    )


def write_decomposition(con: duckdb.DuckDBPyConnection) -> None:
    copy_to_csv(
        con,
        f"""
        WITH common AS (
          SELECT company_name_canonical
          FROM t06_company_metrics
          WHERE source_key IN ('kaggle_arshkon', 'scraped_linkedin')
          GROUP BY 1
          HAVING
            max(CASE WHEN source_key = 'kaggle_arshkon' AND postings >= 5 THEN 1 ELSE 0 END) = 1
            AND max(CASE WHEN source_key = 'scraped_linkedin' AND postings >= 5 THEN 1 ELSE 0 END) = 1
        ),
        metric_values AS (
          SELECT 'entry_j1' AS metric, source_key, company_name_canonical, postings::DOUBLE AS denominator, j1_share_all AS value
          FROM t06_company_metrics WHERE source_key IN ('kaggle_arshkon', 'scraped_linkedin')
          UNION ALL
          SELECT 'entry_j2', source_key, company_name_canonical, postings::DOUBLE, j2_share_all
          FROM t06_company_metrics WHERE source_key IN ('kaggle_arshkon', 'scraped_linkedin')
          UNION ALL
          SELECT 'entry_j3', source_key, company_name_canonical, yoe_known_n::DOUBLE, j3_share_yoe_known
          FROM t06_company_metrics WHERE source_key IN ('kaggle_arshkon', 'scraped_linkedin')
          UNION ALL
          SELECT 'entry_j4', source_key, company_name_canonical, yoe_known_n::DOUBLE, j4_share_yoe_known
          FROM t06_company_metrics WHERE source_key IN ('kaggle_arshkon', 'scraped_linkedin')
          UNION ALL
          SELECT 'ai_mention_prevalence', source_key, company_name_canonical, postings::DOUBLE, ai_mention_share
          FROM t06_company_metrics WHERE source_key IN ('kaggle_arshkon', 'scraped_linkedin')
          UNION ALL
          SELECT 'description_length_mean', source_key, company_name_canonical, description_length_known_n::DOUBLE, mean_description_length
          FROM t06_company_metrics WHERE source_key IN ('kaggle_arshkon', 'scraped_linkedin')
          UNION ALL
          SELECT 'tech_count_mean', source_key, company_name_canonical, postings::DOUBLE, mean_tech_count
          FROM t06_company_metrics WHERE source_key IN ('kaggle_arshkon', 'scraped_linkedin')
        ),
        full_agg AS (
          SELECT
            metric,
            source_key,
            sum(value * denominator) / nullif(sum(denominator), 0) AS full_value,
            sum(denominator) AS full_denominator
          FROM metric_values
          WHERE denominator > 0 AND value IS NOT NULL
          GROUP BY 1, 2
        ),
        paired AS (
          SELECT
            a.metric,
            a.company_name_canonical,
            a.denominator AS n0,
            s.denominator AS n1,
            a.value AS y0,
            s.value AS y1
          FROM metric_values a
          JOIN metric_values s
            ON a.metric = s.metric
           AND a.company_name_canonical = s.company_name_canonical
          JOIN common c
            ON a.company_name_canonical = c.company_name_canonical
          WHERE a.source_key = 'kaggle_arshkon'
            AND s.source_key = 'scraped_linkedin'
            AND a.denominator > 0
            AND s.denominator > 0
            AND a.value IS NOT NULL
            AND s.value IS NOT NULL
        ),
        weighted AS (
          SELECT
            *,
            n0 / sum(n0) OVER (PARTITION BY metric) AS w0,
            n1 / sum(n1) OVER (PARTITION BY metric) AS w1
          FROM paired
        ),
        components AS (
          SELECT
            metric,
            count(*) AS companies_used,
            sum(n0) AS arshkon_common_denominator,
            sum(n1) AS scraped_common_denominator,
            sum(w0 * y0) AS arshkon_common_value,
            sum(w1 * y1) AS scraped_common_value,
            sum(w1 * y1) - sum(w0 * y0) AS common_panel_total_change,
            sum(((w0 + w1) / 2.0) * (y1 - y0)) AS within_company_component,
            sum((w1 - w0) * ((y0 + y1) / 2.0)) AS between_reweighting_component
          FROM weighted
          GROUP BY 1
        ),
        common_count AS (
          SELECT count(*) AS common_company_pool_n FROM common
        )
        SELECT
          c.metric,
          cc.common_company_pool_n,
          c.companies_used,
          c.arshkon_common_denominator,
          c.scraped_common_denominator,
          c.arshkon_common_value,
          c.scraped_common_value,
          c.common_panel_total_change,
          c.within_company_component,
          c.between_reweighting_component,
          c.common_panel_total_change - c.within_company_component - c.between_reweighting_component AS midpoint_residual,
          f0.full_value AS arshkon_full_value,
          f1.full_value AS scraped_full_value,
          f0.full_denominator AS arshkon_full_denominator,
          f1.full_denominator AS scraped_full_denominator,
          f1.full_value - f0.full_value AS full_total_change,
          (f1.full_value - f0.full_value) - c.common_panel_total_change AS noncommon_or_entrant_exit_residual
        FROM components c
        CROSS JOIN common_count cc
        LEFT JOIN full_agg f0
          ON c.metric = f0.metric AND f0.source_key = 'kaggle_arshkon'
        LEFT JOIN full_agg f1
          ON c.metric = f1.metric AND f1.source_key = 'scraped_linkedin'
        ORDER BY
          CASE c.metric
            WHEN 'entry_j1' THEN 1
            WHEN 'entry_j2' THEN 2
            WHEN 'entry_j3' THEN 3
            WHEN 'entry_j4' THEN 4
            WHEN 'ai_mention_prevalence' THEN 5
            WHEN 'description_length_mean' THEN 6
            WHEN 'tech_count_mean' THEN 7
            ELSE 99
          END
        """,
        TABLE_DIR / "within_between_decomposition_common_arshkon_scraped_linkedin.csv",
    )


def write_entry_specialists(con: duckdb.DuckDBPyConnection) -> int:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t06_entry_specialists AS
        WITH scored AS (
          SELECT
            *,
            greatest(
              coalesce(j1_share_all, -1),
              coalesce(j2_share_all, -1),
              coalesce(j3_share_yoe_known, -1),
              coalesce(j4_share_yoe_known, -1)
            ) AS max_junior_share
          FROM t06_company_metrics
          WHERE postings >= 5
        ),
        variants AS (
          SELECT
            *,
            CASE
              WHEN max_junior_share = coalesce(j1_share_all, -1) THEN 'J1'
              WHEN max_junior_share = coalesce(j2_share_all, -1) THEN 'J2'
              WHEN max_junior_share = coalesce(j3_share_yoe_known, -1) THEN 'J3'
              WHEN max_junior_share = coalesce(j4_share_yoe_known, -1) THEN 'J4'
              ELSE NULL
            END AS max_junior_variant
          FROM scored
          WHERE max_junior_share > 0.60
        )
        SELECT
          row_number() OVER (ORDER BY max_junior_share DESC, postings DESC, company_name_canonical) AS rank,
          source_key,
          company_name_canonical,
          postings,
          known_seniority_n,
          yoe_known_n,
          j1_entry_n,
          j2_entry_associate_n,
          j3_yoe_le2_n,
          j4_yoe_le3_n,
          j1_share_all,
          j2_share_all,
          j1_share_known_seniority,
          j2_share_known_seniority,
          j3_share_yoe_known,
          j4_share_yoe_known,
          max_junior_share,
          max_junior_variant,
          any_aggregator,
          aggregator_posting_share,
          {category_case('company_name_canonical')} AS manual_category,
          row_number() OVER (ORDER BY max_junior_share DESC, postings DESC, company_name_canonical) <= 20 AS top20_manual_reviewed
        FROM variants
        """
    )
    copy_to_csv(
        con,
        "SELECT * FROM t06_entry_specialists ORDER BY rank",
        SPECIALIST_PATH,
    )
    copy_to_csv(
        con,
        "SELECT * FROM t06_entry_specialists ORDER BY rank LIMIT 50",
        TABLE_DIR / "entry_specialist_employers_top50.csv",
    )
    copy_to_csv(
        con,
        """
        SELECT
          manual_category,
          any_aggregator,
          count(*) AS flagged_companies,
          sum(postings) AS flagged_postings
        FROM t06_entry_specialists
        GROUP BY 1, 2
        ORDER BY manual_category, any_aggregator
        """,
        TABLE_DIR / "entry_specialist_category_by_aggregator.csv",
    )
    return con.execute("SELECT count(*) FROM t06_entry_specialists").fetchone()[0]


def write_aggregator_profile(con: duckdb.DuckDBPyConnection) -> None:
    copy_to_csv(
        con,
        f"""
        WITH totals AS (
          SELECT source_key, count(*) AS source_n
          FROM t06_base
          GROUP BY 1
        )
        SELECT
          b.source_key,
          b.is_aggregator,
          count(*) AS postings,
          count(*)::DOUBLE / max(t.source_n) AS share_of_source_swe,
          avg(b.description_length) AS mean_description_length,
          avg(b.yoe_extracted) FILTER (WHERE b.yoe_known = 1) AS mean_yoe,
          sum(b.yoe_known) AS yoe_known_n,
          sum(b.known_seniority) AS known_seniority_n,
          sum(b.j1_entry)::DOUBLE / count(*) AS entry_share_of_all_postings,
          sum(b.j1_entry)::DOUBLE / nullif(sum(b.known_seniority), 0) AS entry_share_of_known_seniority,
          sum(b.j3_yoe_le2)::DOUBLE / nullif(sum(b.yoe_known), 0) AS yoe_le2_share_of_yoe_known,
          avg(b.ai_mention::DOUBLE) AS ai_mention_share,
          avg(b.tech_count::DOUBLE) AS mean_tech_count
        FROM t06_base b
        JOIN totals t USING (source_key)
        GROUP BY 1, 2
        ORDER BY {SOURCE_SORT}, b.is_aggregator
        """,
        TABLE_DIR / "aggregator_profile.csv",
    )
    copy_to_csv(
        con,
        f"""
        SELECT
          source_key,
          is_aggregator,
          coalesce(seniority_final, 'null') AS seniority_final,
          count(*) AS n,
          count(*)::DOUBLE / sum(count(*)) OVER (PARTITION BY source_key, is_aggregator) AS share
        FROM t06_base
        GROUP BY 1, 2, 3
        ORDER BY {SOURCE_SORT}, is_aggregator, seniority_final
        """,
        TABLE_DIR / "aggregator_seniority_distribution.csv",
    )


def write_new_entrants(con: duckdb.DuckDBPyConnection) -> None:
    copy_to_csv(
        con,
        f"""
        WITH hist AS (
          SELECT DISTINCT company_name_canonical
          FROM t06_base
          WHERE source_key IN ('kaggle_arshkon', 'kaggle_asaniczka')
            AND company_name_canonical IS NOT NULL
            AND company_name_canonical <> ''
        ),
        scraped AS (
          SELECT
            b.*,
            h.company_name_canonical IS NULL AS new_entrant_vs_2024
          FROM t06_base b
          LEFT JOIN hist h USING (company_name_canonical)
          WHERE b.source_key IN ('scraped_linkedin', 'scraped_indeed')
            AND b.company_name_canonical IS NOT NULL
            AND b.company_name_canonical <> ''
        ),
        totals AS (
          SELECT source_key, count(*) AS source_n
          FROM scraped
          GROUP BY 1
        )
        SELECT
          s.source_key,
          s.new_entrant_vs_2024,
          count(DISTINCT s.company_name_canonical) AS companies,
          count(*) AS postings,
          count(*)::DOUBLE / max(t.source_n) AS share_of_scraped_source_postings,
          avg(s.description_length) AS mean_description_length,
          avg(s.yoe_extracted) FILTER (WHERE s.yoe_known = 1) AS mean_yoe,
          sum(s.j1_entry)::DOUBLE / count(*) AS entry_share_of_all_postings,
          sum(s.j1_entry)::DOUBLE / nullif(sum(s.known_seniority), 0) AS entry_share_of_known_seniority,
          sum(s.j3_yoe_le2)::DOUBLE / nullif(sum(s.yoe_known), 0) AS yoe_le2_share_of_yoe_known,
          avg(s.ai_mention::DOUBLE) AS ai_mention_share,
          avg(s.tech_count::DOUBLE) AS mean_tech_count
        FROM scraped s
        JOIN totals t USING (source_key)
        GROUP BY 1, 2
        ORDER BY {SOURCE_SORT}, s.new_entrant_vs_2024
        """,
        TABLE_DIR / "new_entrant_profile.csv",
    )

    copy_to_csv(
        con,
        f"""
        WITH hist AS (
          SELECT DISTINCT company_name_canonical
          FROM t06_base
          WHERE source_key IN ('kaggle_arshkon', 'kaggle_asaniczka')
            AND company_name_canonical IS NOT NULL
            AND company_name_canonical <> ''
        ),
        new_rows AS (
          SELECT b.*
          FROM t06_base b
          LEFT JOIN hist h USING (company_name_canonical)
          WHERE b.source_key IN ('scraped_linkedin', 'scraped_indeed')
            AND b.company_name_canonical IS NOT NULL
            AND b.company_name_canonical <> ''
            AND h.company_name_canonical IS NULL
        ),
        grouped AS (
          SELECT
            source_key,
            company_name_canonical,
            count(*) AS postings,
            avg(ai_mention::DOUBLE) AS ai_mention_share,
            avg(description_length) AS mean_description_length,
            avg(tech_count::DOUBLE) AS mean_tech_count
          FROM new_rows
          GROUP BY 1, 2
        ),
        industry_counts AS (
          SELECT
            source_key,
            company_name_canonical,
            company_industry,
            count(*) AS industry_n,
            row_number() OVER (
              PARTITION BY source_key, company_name_canonical
              ORDER BY count(*) DESC, company_industry
            ) AS rn
          FROM new_rows
          WHERE company_industry IS NOT NULL AND company_industry <> ''
          GROUP BY 1, 2, 3
        ),
        ranked AS (
          SELECT
            g.*,
            i.company_industry AS industry_mode,
            row_number() OVER (
              PARTITION BY g.source_key
              ORDER BY g.postings DESC, g.company_name_canonical
            ) AS source_rank
          FROM grouped g
          LEFT JOIN industry_counts i
            ON g.source_key = i.source_key
           AND g.company_name_canonical = i.company_name_canonical
           AND i.rn = 1
        )
        SELECT
          source_key,
          source_rank,
          company_name_canonical,
          postings,
          industry_mode,
          ai_mention_share,
          mean_description_length,
          mean_tech_count
        FROM ranked
        WHERE source_rank <= 20
        ORDER BY {SOURCE_SORT}, source_rank
        """,
        TABLE_DIR / "new_entrant_top20_by_scraped_source.csv",
    )


def write_prediction_table(con: duckdb.DuckDBPyConnection) -> None:
    top20_share, top50_share = con.execute(
        f"""
        SELECT top_20_share, top_50_share
        FROM read_csv_auto('{(TABLE_DIR / 'concentration_metrics.csv').as_posix()}')
        WHERE source_key = 'scraped_linkedin'
          AND aggregator_excluded = false
        """
    ).fetchone()
    ge5_zero_j1_share = con.execute(
        f"""
        SELECT ge5_zero_j1_entry_share
        FROM read_csv_auto('{(TABLE_DIR / 'entry_posting_concentration.csv').as_posix()}')
        WHERE source_key = 'scraped_linkedin'
        """
    ).fetchone()[0]
    specialist_count = con.execute(
        "SELECT count(*) FROM t06_entry_specialists WHERE source_key = 'scraped_linkedin'"
    ).fetchone()[0]
    max_dup_share = con.execute(
        f"""
        SELECT max(largest_hash_share)
        FROM read_csv_auto('{(TABLE_DIR / 'duplicate_template_top10_by_source.csv').as_posix()}')
        WHERE source_key = 'scraped_linkedin'
        """
    ).fetchone()[0]

    rows = [
        {
            "analysis_category": "entry share",
            "concentration_risk": "high",
            "evidence": (
                f"scraped_linkedin top-20 company share={top20_share:.3f}; "
                f">=5-posting companies with zero J1 rows={ge5_zero_j1_share:.3f}; "
                f"entry-specialist flagged companies={specialist_count}"
            ),
            "recommended_default": "report J1-J4; add company-weighted estimate; exclude entry_specialist_employers as a required sensitivity",
        },
        {
            "analysis_category": "AI mention rate",
            "concentration_risk": "medium",
            "evidence": (
                f"scraped_linkedin top-50 company share={top50_share:.3f}; "
                "AI is binary raw-description prevalence, so repeated templates and prolific employers can move rates"
            ),
            "recommended_default": "use row-level rate plus company-clustered/company-weighted sensitivity; cap prolific firms for corpus summaries",
        },
        {
            "analysis_category": "description length",
            "concentration_risk": "medium",
            "evidence": (
                f"scraped_linkedin top-20 company share={top20_share:.3f}; "
                "length differs by aggregator status, new-entrant status, and source composition"
            ),
            "recommended_default": "use row-level distribution tests, then report company-weighted median/mean and aggregator-excluded sensitivity",
        },
        {
            "analysis_category": "term frequencies",
            "concentration_risk": "high",
            "evidence": (
                f"scraped_linkedin top-50 company share={top50_share:.3f}; "
                f"largest residual exact-template share among top duplicate-template employers={max_dup_share:.3f}"
            ),
            "recommended_default": "cap at 20-50 postings per company and deduplicate exact description_hash within company before term ranking",
        },
        {
            "analysis_category": "topic models",
            "concentration_risk": "high",
            "evidence": "topic models over postings will learn employer/template clusters when prolific firms are uncapped",
            "recommended_default": "deduplicate exact templates; cap per company; inspect topic employer concentration before interpretation",
        },
        {
            "analysis_category": "co-occurrence networks",
            "concentration_risk": "high",
            "evidence": "skill co-occurrence edges are sensitive to repeated employer stacks and template duplication",
            "recommended_default": "construct company-capped network and require terms/edges to appear across >=20 distinct companies",
        },
    ]
    path = TABLE_DIR / "concentration_prediction_table.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(con: duckdb.DuckDBPyConnection, panel_rows: int, specialist_count: int) -> None:
    generated_files = [
        "duplicate_template_top10_by_source.csv",
        "company_entry_metrics.csv",
        "entry_posting_concentration.csv",
        "within_between_decomposition_common_arshkon_scraped_linkedin.csv",
        "entry_specialist_employers_top50.csv",
        "entry_specialist_category_by_aggregator.csv",
        "aggregator_profile.csv",
        "aggregator_seniority_distribution.csv",
        "new_entrant_profile.csv",
        "new_entrant_top20_by_scraped_source.csv",
        "concentration_prediction_table.csv",
    ]
    summary = {
        "memory_discipline": {
            "engine": "duckdb",
            "memory_limit": "4GB",
            "threads": 1,
            "python_full_result_materialization": False,
            "pandas_used": False,
        },
        "t30_panel_rows_loaded_for_j1_j4": panel_rows,
        "note": (
            "The T30 panel CSV is an aggregate reference; company-level J1-J4 membership "
            "was computed from the same definitions under DuckDB aggregates."
        ),
        "generated_missing_tables": generated_files,
        "entry_specialist_count": int(specialist_count),
        "shared_entry_specialist_artifact": SPECIALIST_PATH.as_posix(),
    }
    summary["source_rows"] = [
        {"source_key": row[0], "n_swe_rows": row[1]}
        for row in con.execute(
            f"SELECT source_key, count(*) FROM t06_base GROUP BY 1 ORDER BY {SOURCE_SORT}"
        ).fetchall()
    ]
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SHARED_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    panel_rows = setup_base(con)
    write_duplicate_audit(con)
    write_company_entry_outputs(con)
    write_decomposition(con)
    specialist_count = write_entry_specialists(con)
    write_aggregator_profile(con)
    write_new_entrants(con)
    write_prediction_table(con)
    write_summary(con, panel_rows, specialist_count)
    print(f"Wrote missing T06 outputs under {TABLE_DIR}")
    print(f"Wrote shared entry-specialist artifact to {SPECIALIST_PATH}")


if __name__ == "__main__":
    main()
