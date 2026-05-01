"""T03 — Native-label quality diagnostics.

- YOE distribution of native='entry' rows per source x period
- How native='entry' rows get mapped to seniority_final by path
  (title_keyword, llm, unknown)
- Entry-row agreement: restrict native=entry to rows that made it into the
  LLM frame and see how the LLM re-labeled them.
"""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
OUT = ROOT / "exploration" / "tables" / "T03"

BASE_FILTER = (
    "is_english = TRUE AND date_flag = 'ok' AND source_platform = 'linkedin' "
    "AND is_swe = TRUE"
)


def write_csv(path, header, rows):
    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    # 1. YOE distribution of native='entry' rows
    rows = con.execute(
        f"""
        SELECT source, period,
               COUNT(*) AS n_native_entry,
               ROUND(AVG(yoe_extracted),2) AS mean_yoe,
               MEDIAN(yoe_extracted) AS median_yoe,
               ROUND(100.0*AVG(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END),2) AS pct_with_yoe,
               ROUND(100.0*AVG(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END),2) AS pct_yoe_le2,
               ROUND(100.0*AVG(CASE WHEN yoe_extracted <= 3 THEN 1 ELSE 0 END),2) AS pct_yoe_le3,
               ROUND(100.0*AVG(CASE WHEN yoe_extracted >= 5 THEN 1 ELSE 0 END),2) AS pct_yoe_ge5,
               ROUND(100.0*AVG(CASE WHEN yoe_extracted >= 7 THEN 1 ELSE 0 END),2) AS pct_yoe_ge7
        FROM '{DATA}'
        WHERE {BASE_FILTER} AND seniority_native='entry'
        GROUP BY 1,2 ORDER BY 1,2
        """,
    ).fetchall()
    write_csv(
        OUT / "07_native_entry_yoe_profile.csv",
        ["source", "period", "n_native_entry", "mean_yoe", "median_yoe",
         "pct_with_yoe", "pct_yoe_le2", "pct_yoe_le3", "pct_yoe_ge5", "pct_yoe_ge7"],
        rows,
    )

    # 2. native=entry → final mapping broken by path
    rows = con.execute(
        f"""
        SELECT source, period,
               COALESCE(seniority_final_source,'<null>') AS path,
               seniority_final, COUNT(*) AS n
        FROM '{DATA}'
        WHERE {BASE_FILTER} AND seniority_native='entry'
        GROUP BY 1,2,3,4 ORDER BY 1,2,3,4
        """,
    ).fetchall()
    write_csv(
        OUT / "07_native_entry_to_final_by_path.csv",
        ["source", "period", "seniority_final_source", "seniority_final", "n"],
        rows,
    )

    # 3. Do the same for native='mid-senior'
    rows = con.execute(
        f"""
        SELECT source, period,
               COALESCE(seniority_final_source,'<null>') AS path,
               seniority_final, COUNT(*) AS n
        FROM '{DATA}'
        WHERE {BASE_FILTER} AND seniority_native='mid-senior'
        GROUP BY 1,2,3,4 ORDER BY 1,2,3,4
        """,
    ).fetchall()
    write_csv(
        OUT / "07_native_midsenior_to_final_by_path.csv",
        ["source", "period", "seniority_final_source", "seniority_final", "n"],
        rows,
    )

    # 4. Agreement only among LLM-routed rows where native is available
    # (excludes rows where title_keyword fired, which would bias kappa toward
    # the obvious title-matches).
    rows = con.execute(
        f"""
        SELECT source, period,
               seniority_native,
               seniority_final,
               COUNT(*) AS n
        FROM '{DATA}'
        WHERE {BASE_FILTER}
          AND seniority_final_source='llm'
          AND seniority_native IS NOT NULL
        GROUP BY 1,2,3,4 ORDER BY 1,2,3,4
        """,
    ).fetchall()
    write_csv(
        OUT / "08_llm_vs_native_crosstab.csv",
        ["source", "period", "seniority_native", "seniority_final", "n"],
        rows,
    )

    # 5. By path: coverage of the LLM path in each source x period
    rows = con.execute(
        f"""
        SELECT source, period,
               COALESCE(seniority_final_source,'<null>') AS path,
               COUNT(*) AS n
        FROM '{DATA}'
        WHERE {BASE_FILTER}
        GROUP BY 1,2,3 ORDER BY 1,2,3
        """,
    ).fetchall()
    write_csv(
        OUT / "09_path_coverage_by_source_period.csv",
        ["source", "period", "seniority_final_source", "n"],
        rows,
    )

    # 6. Junior share sanity: what if we require YOE-known rows only?
    rows = con.execute(
        f"""
        SELECT source, period,
               COUNT(*) AS n_yoe_known,
               SUM(CASE WHEN yoe_extracted<=2 THEN 1 ELSE 0 END) AS n_yoe_le2,
               ROUND(100.0*AVG(CASE WHEN yoe_extracted<=2 THEN 1 ELSE 0 END),2) AS pct_yoe_le2,
               SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END) AS n_final_entry,
               ROUND(100.0*AVG(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END),2) AS pct_final_entry
        FROM '{DATA}'
        WHERE {BASE_FILTER} AND yoe_extracted IS NOT NULL
        GROUP BY 1,2 ORDER BY 1,2
        """,
    ).fetchall()
    write_csv(
        OUT / "10_junior_share_yoe_known_subset.csv",
        ["source", "period", "n_yoe_known", "n_yoe_le2", "pct_yoe_le2",
         "n_final_entry", "pct_final_entry"],
        rows,
    )

    print("Wrote diagnostics to", OUT)


if __name__ == "__main__":
    main()
