#!/usr/bin/env python3
"""T04 SWE classification audit.

Writes aggregate classification tables, boundary-case profiles, and deterministic
spot-check samples for manual review.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


DATA = "data/unified.parquet"
OUT = Path("exploration/tables/T04")
OUT.mkdir(parents=True, exist_ok=True)

BASE_FILTER = """
  source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
"""


def main() -> None:
    con = duckdb.connect()

    swe_tier = con.execute(
        f"""
        WITH t AS (
          SELECT source, period, swe_classification_tier, count(*) AS n
          FROM read_parquet('{DATA}')
          WHERE {BASE_FILTER}
            AND is_swe = true
          GROUP BY source, period, swe_classification_tier
        )
        SELECT
          source,
          period,
          swe_classification_tier,
          n,
          n::DOUBLE / sum(n) OVER (PARTITION BY source, period) AS share
        FROM t
        ORDER BY source, period, n DESC
        """
    ).df()
    swe_tier.to_csv(OUT / "swe_rows_by_classification_tier.csv", index=False)

    swe_tier_overall = con.execute(
        f"""
        WITH t AS (
          SELECT swe_classification_tier, count(*) AS n
          FROM read_parquet('{DATA}')
          WHERE {BASE_FILTER}
            AND is_swe = true
          GROUP BY swe_classification_tier
        )
        SELECT swe_classification_tier, n, n::DOUBLE / sum(n) OVER () AS share
        FROM t
        ORDER BY n DESC
        """
    ).df()
    swe_tier_overall.to_csv(OUT / "swe_rows_by_classification_tier_overall.csv", index=False)

    borderline_swe_condition = """
      is_swe = true
      AND (
        (swe_confidence BETWEEN 0.3 AND 0.7)
        OR swe_classification_tier = 'title_lookup_llm'
      )
    """
    borderline_swe_population = con.execute(
        f"""
        SELECT source, period, swe_classification_tier, count(*) AS n
        FROM read_parquet('{DATA}')
        WHERE {BASE_FILTER}
          AND {borderline_swe_condition}
        GROUP BY source, period, swe_classification_tier
        ORDER BY source, period, n DESC
        """
    ).df()
    borderline_swe_population.to_csv(OUT / "borderline_swe_population.csv", index=False)

    borderline_swe_sample = con.execute(
        f"""
        SELECT
          uid,
          source,
          period,
          title,
          company_name,
          swe_confidence,
          swe_classification_tier,
          seniority_final,
          is_swe_adjacent,
          is_control,
          left(coalesce(description, ''), 240) AS description_excerpt
        FROM read_parquet('{DATA}')
        WHERE {BASE_FILTER}
          AND {borderline_swe_condition}
        ORDER BY hash(uid)
        LIMIT 50
        """
    ).df()
    borderline_swe_sample.to_csv(OUT / "sample_borderline_swe_50.csv", index=False)

    borderline_non_swe_condition = """
      is_swe = false
      AND regexp_matches(
        coalesce(title_normalized, ''),
        '\\b(engineer|developer|software)\\b'
      )
    """
    borderline_non_swe_population = con.execute(
        f"""
        SELECT
          source,
          period,
          is_swe_adjacent,
          is_control,
          swe_classification_tier,
          count(*) AS n
        FROM read_parquet('{DATA}')
        WHERE {BASE_FILTER}
          AND {borderline_non_swe_condition}
        GROUP BY source, period, is_swe_adjacent, is_control, swe_classification_tier
        ORDER BY source, period, n DESC
        """
    ).df()
    borderline_non_swe_population.to_csv(OUT / "borderline_non_swe_population.csv", index=False)

    borderline_non_swe_sample = con.execute(
        f"""
        SELECT
          uid,
          source,
          period,
          title,
          company_name,
          swe_confidence,
          swe_classification_tier,
          is_swe_adjacent,
          is_control,
          seniority_final,
          left(coalesce(description, ''), 240) AS description_excerpt
        FROM read_parquet('{DATA}')
        WHERE {BASE_FILTER}
          AND {borderline_non_swe_condition}
        ORDER BY hash(uid)
        LIMIT 50
        """
    ).df()
    borderline_non_swe_sample.to_csv(OUT / "sample_borderline_non_swe_50.csv", index=False)

    group_counts = con.execute(
        f"""
        WITH flags AS (
          SELECT
            source,
            period,
            CASE
              WHEN is_swe THEN 'swe'
              WHEN is_swe_adjacent THEN 'swe_adjacent'
              WHEN is_control THEN 'control'
              ELSE 'other_non_swe'
            END AS group_label,
            swe_classification_tier,
            count(*) AS n
          FROM read_parquet('{DATA}')
          WHERE {BASE_FILTER}
          GROUP BY source, period, group_label, swe_classification_tier
        )
        SELECT
          source,
          period,
          group_label,
          swe_classification_tier,
          n,
          n::DOUBLE / sum(n) OVER (PARTITION BY source, period, group_label) AS share_within_group
        FROM flags
        ORDER BY source, period, group_label, n DESC
        """
    ).df()
    group_counts.to_csv(OUT / "classification_group_counts_by_source_period.csv", index=False)

    top_titles = con.execute(
        f"""
        WITH grouped AS (
          SELECT
            CASE
              WHEN is_swe THEN 'swe'
              WHEN is_swe_adjacent THEN 'swe_adjacent'
              WHEN is_control THEN 'control'
              ELSE 'other_non_swe'
            END AS group_label,
            title_normalized,
            count(*) AS n
          FROM read_parquet('{DATA}')
          WHERE {BASE_FILTER}
            AND title_normalized IS NOT NULL
          GROUP BY group_label, title_normalized
        ),
        ranked AS (
          SELECT
            *,
            row_number() OVER (PARTITION BY group_label ORDER BY n DESC, title_normalized) AS rn
          FROM grouped
        )
        SELECT group_label, title_normalized, n
        FROM ranked
        WHERE rn <= 30
        ORDER BY group_label, n DESC, title_normalized
        """
    ).df()
    top_titles.to_csv(OUT / "top_titles_by_classification_group.csv", index=False)

    dual_flags = con.execute(
        f"""
        SELECT
          count(*) AS denominator_default_linkedin_rows,
          sum(CASE WHEN (
            (CASE WHEN is_swe THEN 1 ELSE 0 END) +
            (CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) +
            (CASE WHEN is_control THEN 1 ELSE 0 END)
          ) > 1 THEN 1 ELSE 0 END) AS dual_flag_violations,
          sum(CASE WHEN (
            (CASE WHEN is_swe THEN 1 ELSE 0 END) +
            (CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) +
            (CASE WHEN is_control THEN 1 ELSE 0 END)
          ) = 0 THEN 1 ELSE 0 END) AS no_flag_rows
        FROM read_parquet('{DATA}')
        WHERE {BASE_FILTER}
        """
    ).df()
    dual_flags.to_csv(OUT / "dual_flag_violations.csv", index=False)

    boundary_patterns = {
        "ml_engineer": r"\b(machine learning engineer|ml engineer|ai engineer|artificial intelligence engineer)\b",
        "data_engineer": r"\bdata engineer\b",
        "devops_sre_platform": r"\b(devops|site reliability|sre|platform engineer|infrastructure engineer)\b",
        "qa_test_sdet": r"\b(qa engineer|quality assurance|test automation|sdet)\b",
        "sales_solutions_engineer": r"\b(sales engineer|solutions engineer|solution engineer)\b",
        "support_engineer": r"\b(support engineer|technical support engineer)\b",
        "generic_software_engineer": r"\b(software engineer|software developer|developer)\b",
    }

    boundary_rows = []
    for name, pattern in boundary_patterns.items():
        df = con.execute(
            f"""
            SELECT
              '{name}' AS boundary_case,
              source,
              period,
              is_swe,
              is_swe_adjacent,
              is_control,
              count(*) AS n,
              avg(swe_confidence) AS avg_swe_confidence
            FROM read_parquet('{DATA}')
            WHERE {BASE_FILTER}
              AND regexp_matches(coalesce(title_normalized, ''), ?)
            GROUP BY source, period, is_swe, is_swe_adjacent, is_control
            """,
            [pattern],
        ).df()
        boundary_rows.append(df)
    pd.concat(boundary_rows, ignore_index=True).to_csv(OUT / "boundary_case_profiles.csv", index=False)

    print(f"Wrote T04 tables to {OUT}")


if __name__ == "__main__":
    main()
