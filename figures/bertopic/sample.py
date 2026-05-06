"""
Build Sample A (SWE) and Sample B (SWE + control) for the BERTopic project.

Sample A: SWE postings from kaggle_asaniczka (2024-01) + kaggle_arshkon
(2024-04) + scraped (2026-03, 2026-04). Headline BERTopic fit; primary paper
claims about SWE role-landscape change come from this sample.

Sample B: Sample A plus control occupations from arshkon + scraped. Used by
the cross-occupation tests (§3.3) and by §6.3 control-differenced drift.

Filters (§2.4): substrate non-null, substrate length >= 200, date_flag ok,
has_llm_classification, embedding non-null. Substrate is `description_core_llm`
exclusively — never raw `description` (user clarification 2026-05-05).

Cap (§3.1): 5 per (canonical_co, period, title_normalized), where
title_normalized = lowercase + strip-non-alphanumeric + collapse-whitespace.
Within-bucket selection is hash-on-uid so the kept rows are not biased by
posting-time clustering.

Outputs:
  - intermediate/sample_a.parquet
  - intermediate/sample_b.parquet
  - intermediate/sample_sizes.csv  (one row per source × period × sample)
"""

from __future__ import annotations

import csv
import duckdb

from figures.bertopic import config


def title_normalized_sql(col: str) -> str:
    """SQL fragment that lowercases and reduces a title to alphanumeric tokens.

    Lowercase + collapse non-alphanumeric runs to a single space + trim. Keeps
    `Senior SWE III` distinct from `Senior SWE II`; collapses
    `Senior Software Engineer` and `senior software engineer` to one bucket.
    """
    return (
        f"trim(regexp_replace(lower({col}), '[^a-z0-9]+', ' ', 'g'))"
    )


_BASE_FILTER = (
    "description_core_llm IS NOT NULL "
    "AND length(description_core_llm) >= "
    f"{config.SUBSTRATE_MIN_LENGTH} "
    "AND date_flag = 'ok' "
    "AND has_llm_classification "
    "AND job_description_embedding IS NOT NULL"
)


def _build_capped_query(*, where_class: str) -> str:
    title_norm = title_normalized_sql("title")
    # Embedding column is intentionally excluded — it lives in the cache
    # under uid keys and re-storing it here would waste ~1 GB.
    return f"""
    WITH base AS (
        SELECT
            uid, source, period, title, description_core_llm,
            company_name_canonical, is_aggregator, metro_area,
            seniority_final, yoe_min_years_llm,
            is_swe, is_control,
            {title_norm} AS title_normalized
        FROM '{config.UNIFIED_CORE_PATH}'
        WHERE {_BASE_FILTER} AND ({where_class})
    ),
    ranked AS (
        SELECT
            *,
            row_number() OVER (
                PARTITION BY company_name_canonical, period, title_normalized
                ORDER BY hash(uid)
            ) AS rn_within_bucket
        FROM base
    )
    SELECT
        uid, source, period, title, title_normalized,
        description_core_llm,
        company_name_canonical, is_aggregator, metro_area,
        seniority_final, yoe_min_years_llm,
        is_swe, is_control
    FROM ranked
    WHERE rn_within_bucket <= {config.PER_BUCKET_CAP}
    """


def build_samples() -> tuple[int, int]:
    """Build Sample A and Sample B parquet files; return (n_a, n_b)."""
    config.INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")

    # Sample A: SWE only.
    sample_a_query = _build_capped_query(where_class="is_swe")
    n_a = con.execute(
        f"COPY ({sample_a_query}) TO '{config.SAMPLE_A_PATH}' "
        f"(FORMAT PARQUET, COMPRESSION ZSTD)"
    ).fetchone()
    n_a_count = con.execute(
        f"SELECT count(*) FROM '{config.SAMPLE_A_PATH}'"
    ).fetchone()[0]

    # Sample B: SWE OR control.
    sample_b_query = _build_capped_query(where_class="is_swe OR is_control")
    con.execute(
        f"COPY ({sample_b_query}) TO '{config.SAMPLE_B_PATH}' "
        f"(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    n_b_count = con.execute(
        f"SELECT count(*) FROM '{config.SAMPLE_B_PATH}'"
    ).fetchone()[0]

    return n_a_count, n_b_count


def write_sample_sizes() -> None:
    """Write the source × period × sample row counts to sample_sizes.csv."""
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")

    rows: list[tuple[str, str, str, int, int, int]] = []
    for sample_name, path in (("A", config.SAMPLE_A_PATH), ("B", config.SAMPLE_B_PATH)):
        per_period = con.execute(f"""
            SELECT source, period,
                   count(*) AS n_total,
                   sum(CASE WHEN is_swe THEN 1 ELSE 0 END) AS n_swe,
                   sum(CASE WHEN is_control THEN 1 ELSE 0 END) AS n_control
            FROM '{path}'
            GROUP BY source, period
            ORDER BY source, period
        """).fetchall()
        for source, period, n_total, n_swe, n_control in per_period:
            rows.append((sample_name, source, period, n_total, n_swe, n_control))

    with config.SAMPLE_SIZES_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "source", "period", "n_total", "n_swe", "n_control"])
        for r in rows:
            w.writerow(r)


def main() -> None:
    n_a, n_b = build_samples()
    write_sample_sizes()
    print(f"sample_a rows: {n_a}")
    print(f"sample_b rows: {n_b}")
    print(f"sample_sizes.csv → {config.SAMPLE_SIZES_CSV}")


if __name__ == "__main__":
    main()
