"""T03 — Seniority label audit.

Produces:
  - seniority_final_source profile (by source x period), SWE only
  - seniority_final vs seniority_native cross-tabs with Cohen's kappa
    (arshkon SWE, scraped LinkedIn SWE)
  - Junior-share by period under three operationalizations:
      (a) seniority_final = 'entry'
      (b) seniority_native = 'entry' (arshkon-only)
      (c) yoe_extracted <= 2
  - Sample of 100 LLM-routed SWE rows whose titles contain weak seniority
    markers (I/II/III, junior, senior) for routing-error spot-check

All outputs written to exploration/tables/T03/.

Default SQL filters: source_platform='linkedin', is_english=true, date_flag='ok'.
SWE restriction: is_swe=true.
"""

from __future__ import annotations

import csv
import os
import json
from pathlib import Path

import duckdb

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
OUT = ROOT / "exploration" / "tables" / "T03"
OUT.mkdir(parents=True, exist_ok=True)

BASE_FILTER = (
    "is_english = TRUE AND date_flag = 'ok' AND source_platform = 'linkedin' "
    "AND is_swe = TRUE"
)


def write_csv(path: Path, header, rows) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def q(con, sql):
    return con.execute(sql).fetchall()


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute(f"CREATE VIEW swe AS SELECT * FROM '{DATA}' WHERE {BASE_FILTER}")

    # ------------------------------------------------------------------
    # 1. seniority_final_source distribution by source x period (SWE)
    # ------------------------------------------------------------------
    rows = q(
        con,
        """
        SELECT source, period,
               COALESCE(seniority_final_source,'<null>') AS sfs,
               COUNT(*) AS n
        FROM swe
        GROUP BY 1,2,3
        ORDER BY 1,2,3
        """,
    )
    write_csv(OUT / "01_final_source_profile.csv",
              ["source", "period", "seniority_final_source", "n"], rows)

    # Pivot friendly table: share of each source within (source, period)
    rows2 = q(
        con,
        """
        WITH t AS (
          SELECT source, period,
                 COALESCE(seniority_final_source,'<null>') AS sfs,
                 COUNT(*) AS n
          FROM swe
          GROUP BY 1,2,3
        ),
        totals AS (
          SELECT source, period, SUM(n) AS n_total FROM t GROUP BY 1,2
        )
        SELECT t.source, t.period, t.sfs, t.n, totals.n_total,
               ROUND(100.0 * t.n / totals.n_total, 2) AS pct
        FROM t JOIN totals USING (source, period)
        ORDER BY 1,2,3
        """,
    )
    write_csv(
        OUT / "01_final_source_profile_with_pct.csv",
        ["source", "period", "seniority_final_source", "n", "n_total_cell", "pct_of_cell"],
        rows2,
    )

    # ------------------------------------------------------------------
    # 2. seniority_final distribution by source x period (SWE)
    # ------------------------------------------------------------------
    rows = q(
        con,
        """
        SELECT source, period,
               COALESCE(seniority_final,'<null>') AS sen,
               COUNT(*) AS n
        FROM swe
        GROUP BY 1,2,3
        ORDER BY 1,2,3
        """,
    )
    write_csv(
        OUT / "02_final_distribution.csv",
        ["source", "period", "seniority_final", "n"],
        rows,
    )

    rows = q(
        con,
        """
        SELECT source, period,
               COALESCE(seniority_native,'<null>') AS sen,
               COUNT(*) AS n
        FROM swe
        GROUP BY 1,2,3
        ORDER BY 1,2,3
        """,
    )
    write_csv(
        OUT / "02_native_distribution.csv",
        ["source", "period", "seniority_native", "n"],
        rows,
    )

    # ------------------------------------------------------------------
    # 3. Cross-tabs: seniority_final vs seniority_native
    #    (arshkon SWE + scraped LinkedIn SWE)
    # ------------------------------------------------------------------
    def crosstab(subset_pred: str, tag: str) -> None:
        ct = q(
            con,
            f"""
            SELECT COALESCE(seniority_native,'<null>') AS native,
                   COALESCE(seniority_final,'<null>') AS final,
                   COUNT(*) AS n
            FROM swe
            WHERE {subset_pred}
            GROUP BY 1,2
            ORDER BY 1,2
            """,
        )
        write_csv(
            OUT / f"03_crosstab_{tag}.csv",
            ["seniority_native", "seniority_final", "n"],
            ct,
        )

    crosstab("source = 'kaggle_arshkon'", "arshkon")
    crosstab("source = 'scraped'", "scraped_linkedin")

    # ------------------------------------------------------------------
    # 4. Junior share by period under three operationalizations
    #    All rows are SWE LinkedIn defaults.
    # ------------------------------------------------------------------
    # (a) seniority_final = 'entry'
    rows_a = q(
        con,
        """
        SELECT period,
               COUNT(*) AS n_total,
               SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END) AS n_entry_final,
               SUM(CASE WHEN seniority_final NOT IN ('unknown') AND seniority_final IS NOT NULL
                        THEN 1 ELSE 0 END) AS n_known_final,
               ROUND(100.0 * SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END) /
                     NULLIF(COUNT(*),0), 2) AS pct_entry_of_all,
               ROUND(100.0 * SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END) /
                     NULLIF(SUM(CASE WHEN seniority_final NOT IN ('unknown') AND seniority_final IS NOT NULL
                                     THEN 1 ELSE 0 END), 0), 2) AS pct_entry_of_known
        FROM swe
        GROUP BY 1
        ORDER BY 1
        """,
    )
    write_csv(
        OUT / "04a_junior_share_seniority_final.csv",
        ["period", "n_total", "n_entry_final", "n_known_final",
         "pct_entry_of_all", "pct_entry_of_known"],
        rows_a,
    )

    # (b) seniority_native = 'entry' (arshkon only)
    rows_b = q(
        con,
        """
        SELECT period,
               COUNT(*) AS n_total,
               SUM(CASE WHEN seniority_native='entry' THEN 1 ELSE 0 END) AS n_entry_native,
               SUM(CASE WHEN seniority_native IS NOT NULL THEN 1 ELSE 0 END) AS n_known_native,
               ROUND(100.0 * SUM(CASE WHEN seniority_native='entry' THEN 1 ELSE 0 END) /
                     NULLIF(COUNT(*),0), 2) AS pct_entry_of_all,
               ROUND(100.0 * SUM(CASE WHEN seniority_native='entry' THEN 1 ELSE 0 END) /
                     NULLIF(SUM(CASE WHEN seniority_native IS NOT NULL THEN 1 ELSE 0 END), 0), 2)
                 AS pct_entry_of_known
        FROM swe
        WHERE source='kaggle_arshkon'
        GROUP BY 1
        ORDER BY 1
        """,
    )
    write_csv(
        OUT / "04b_junior_share_seniority_native_arshkon.csv",
        ["period", "n_total", "n_entry_native", "n_known_native",
         "pct_entry_of_all", "pct_entry_of_known"],
        rows_b,
    )

    # (c) yoe_extracted <= 2 (label-independent)
    rows_c = q(
        con,
        """
        SELECT period,
               COUNT(*) AS n_total,
               SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS n_with_yoe,
               SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) AS n_yoe_le2,
               SUM(CASE WHEN yoe_extracted <= 3 THEN 1 ELSE 0 END) AS n_yoe_le3,
               ROUND(100.0 * SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) /
                     NULLIF(COUNT(*),0),2) AS pct_yoe_le2_of_all,
               ROUND(100.0 * SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) /
                     NULLIF(SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END), 0), 2)
                 AS pct_yoe_le2_of_yoe_known,
               ROUND(100.0 * SUM(CASE WHEN yoe_extracted <= 3 THEN 1 ELSE 0 END) /
                     NULLIF(SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END), 0), 2)
                 AS pct_yoe_le3_of_yoe_known
        FROM swe
        GROUP BY 1
        ORDER BY 1
        """,
    )
    write_csv(
        OUT / "04c_junior_share_yoe_proxy.csv",
        ["period", "n_total", "n_with_yoe", "n_yoe_le2", "n_yoe_le3",
         "pct_yoe_le2_of_all", "pct_yoe_le2_of_yoe_known",
         "pct_yoe_le3_of_yoe_known"],
        rows_c,
    )

    # Same three operationalizations, but collapsing to 2024 vs 2026
    # (pooling all 2024 LinkedIn SWE, all 2026 LinkedIn SWE). For (b) restrict to arshkon.
    def period_bucket_sql():
        return (
            "CASE WHEN period LIKE '2024%' THEN '2024' "
            "WHEN period LIKE '2026%' THEN '2026' ELSE period END"
        )

    rows = q(
        con,
        f"""
        SELECT {period_bucket_sql()} AS period_bucket,
               COUNT(*) AS n,
               SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END) AS n_entry_final,
               ROUND(100.0*SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END)/
                     NULLIF(COUNT(*),0),2) AS pct_entry_final,
               ROUND(100.0*SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END)/
                     NULLIF(SUM(CASE WHEN seniority_final NOT IN ('unknown') AND seniority_final IS NOT NULL THEN 1 ELSE 0 END),0),2) AS pct_entry_final_of_known,
               SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) AS n_yoe_le2,
               ROUND(100.0*SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END)/
                     NULLIF(COUNT(*),0),2) AS pct_yoe_le2,
               ROUND(100.0*SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END)/
                     NULLIF(SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END),0),2) AS pct_yoe_le2_of_yoe_known
        FROM swe
        GROUP BY 1 ORDER BY 1
        """,
    )
    write_csv(
        OUT / "04_three_ops_pooled_periods.csv",
        ["period_bucket", "n", "n_entry_final", "pct_entry_final",
         "pct_entry_final_of_known", "n_yoe_le2", "pct_yoe_le2",
         "pct_yoe_le2_of_yoe_known"],
        rows,
    )

    rows = q(
        con,
        f"""
        SELECT {period_bucket_sql()} AS period_bucket,
               COUNT(*) AS n,
               SUM(CASE WHEN seniority_native='entry' THEN 1 ELSE 0 END) AS n_entry_native,
               ROUND(100.0*SUM(CASE WHEN seniority_native='entry' THEN 1 ELSE 0 END)/
                     NULLIF(COUNT(*),0),2) AS pct_entry_native
        FROM swe
        WHERE source='kaggle_arshkon'
        GROUP BY 1 ORDER BY 1
        """,
    )
    write_csv(
        OUT / "04b_pooled_periods_native_arshkon.csv",
        ["period_bucket", "n", "n_entry_native", "pct_entry_native"],
        rows,
    )

    # Source-level breakdown for (a) to show which source contributes
    rows = q(
        con,
        f"""
        SELECT {period_bucket_sql()} AS period_bucket, source,
               COUNT(*) AS n,
               SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END) AS n_entry_final,
               ROUND(100.0*SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0),2)
                 AS pct_entry_final,
               SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) AS n_yoe_le2,
               ROUND(100.0*SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0),2)
                 AS pct_yoe_le2
        FROM swe
        GROUP BY 1,2 ORDER BY 1,2
        """,
    )
    write_csv(
        OUT / "04_three_ops_by_source.csv",
        ["period_bucket", "source", "n", "n_entry_final", "pct_entry_final",
         "n_yoe_le2", "pct_yoe_le2"],
        rows,
    )

    # ------------------------------------------------------------------
    # 5. Sample 100 LLM-routed rows with weak seniority markers for
    #    routing-error spot-check. Balance across sources where possible.
    # ------------------------------------------------------------------
    # Use LIKE patterns instead of regex — some rows contain invalid UTF-8
    # and duckdb's regex engine raises on them. LIKE is byte-level safe.
    MARKER_CASE = """
        CASE
          WHEN t LIKE '%junior%' OR t LIKE '%entry level%' OR t LIKE '%entry-level%'
               OR t LIKE '%intern %' OR t LIKE '% intern' OR t LIKE '%interns%'
               OR t LIKE '%jr.%' OR t LIKE '% jr %' OR t LIKE ' jr %' OR t LIKE '%new grad%'
               OR t LIKE '%new-grad%' OR t LIKE '%early career%' OR t LIKE '%early-career%'
            THEN 'junior_marker'
          WHEN t LIKE '%senior%' OR t LIKE '%staff %' OR t LIKE '% staff'
               OR t LIKE '%principal%' OR t LIKE '%distinguished%'
               OR t LIKE '% lead %' OR t LIKE '%lead ' OR t LIKE 'lead %'
               OR t LIKE '%sr.%' OR t LIKE '% sr %' OR t LIKE '%director%'
            THEN 'senior_marker'
          WHEN t LIKE '% ii' OR t LIKE '% iii' OR t LIKE '% iv' OR t LIKE '% i '
               OR t LIKE '% ii %' OR t LIKE '% iii %' OR t LIKE '% iv %'
            THEN 'roman_marker'
          ELSE NULL
        END
    """
    rows = q(
        con,
        f"""
        WITH base AS (
          SELECT source, period, title, title_normalized,
                 seniority_final, seniority_final_source, seniority_native,
                 yoe_extracted, lower(COALESCE(title,'')) AS t
          FROM swe
          WHERE seniority_final_source='llm'
        ),
        tagged AS (
          SELECT *, {MARKER_CASE} AS marker_kind FROM base
        ),
        picked AS (
          SELECT *, row_number() OVER (PARTITION BY source, marker_kind
                                       ORDER BY hash(title_normalized)) AS rn
          FROM tagged
          WHERE marker_kind IS NOT NULL
        )
        SELECT source, period, marker_kind, title_normalized,
               seniority_final, seniority_native, yoe_extracted
        FROM picked
        WHERE rn <= 17
        ORDER BY source, marker_kind, rn
        """,
    )
    # Sanitize title_normalized (may contain invalid utf-8 bytes) — replace any
    # non-ASCII printable byte with '?'.
    def _clean(s):
        if s is None:
            return ""
        return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in s)
    rows = [
        (r[0], r[1], r[2], _clean(r[3]), r[4], r[5], r[6])
        for r in rows
    ]
    write_csv(
        OUT / "05_llm_routed_weak_marker_sample.csv",
        ["source", "period", "marker_kind", "title_normalized",
         "seniority_final", "seniority_native", "yoe_extracted"],
        rows,
    )

    # Counts of LLM-routed rows with each marker kind (so we know the denominator)
    rows = q(
        con,
        f"""
        WITH base AS (
          SELECT source, seniority_final, lower(COALESCE(title,'')) AS t
          FROM swe WHERE seniority_final_source='llm'
        )
        SELECT source,
               COALESCE({MARKER_CASE}, 'none') AS marker_kind,
               seniority_final,
               COUNT(*) AS n
        FROM base
        GROUP BY 1,2,3 ORDER BY 1,2,3
        """,
    )
    write_csv(
        OUT / "05_llm_routed_marker_counts.csv",
        ["source", "marker_kind", "seniority_final", "n"],
        rows,
    )

    # ------------------------------------------------------------------
    # 6. Sanity: LLM-routed rows whose seniority_final='unknown'
    #    (should be a small valid category per schema)
    # ------------------------------------------------------------------
    rows = q(
        con,
        """
        SELECT source, period, seniority_final, COUNT(*) AS n
        FROM swe
        WHERE seniority_final_source='llm'
        GROUP BY 1,2,3 ORDER BY 1,2,3
        """,
    )
    write_csv(
        OUT / "06_llm_source_final_by_source_period.csv",
        ["source", "period", "seniority_final", "n"],
        rows,
    )

    print("Wrote tables to", OUT)


if __name__ == "__main__":
    main()
