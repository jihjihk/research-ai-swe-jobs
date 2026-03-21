#!/usr/bin/env python3
"""
Stage Final: produce analysis-ready outputs for the rule-based pipeline.

Outputs:
  - data/unified.parquet
  - data/unified_observations.parquet
  - data/quality_report.json
  - data/preprocessing_log.txt
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).parent.parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
DATA_DIR = PROJECT_ROOT / "data"

UNIFIED_INPUT = INTERMEDIATE_DIR / "stage8_final.parquet"
OBS_INPUT = INTERMEDIATE_DIR / "stage1_observations.parquet"
UNIFIED_OUTPUT = DATA_DIR / "unified.parquet"
OBS_OUTPUT = DATA_DIR / "unified_observations.parquet"
QUALITY_OUTPUT = DATA_DIR / "quality_report.json"
LOG_OUTPUT = DATA_DIR / "preprocessing_log.txt"

STAGE_INPUTS = {
    "stage1_unified": INTERMEDIATE_DIR / "stage1_unified.parquet",
    "stage1_observations": INTERMEDIATE_DIR / "stage1_observations.parquet",
    "stage2": INTERMEDIATE_DIR / "stage2_aggregators.parquet",
    "stage3": INTERMEDIATE_DIR / "stage3_boilerplate.parquet",
    "stage4": INTERMEDIATE_DIR / "stage4_dedup.parquet",
    "stage5": INTERMEDIATE_DIR / "stage5_classification.parquet",
    "stage8": INTERMEDIATE_DIR / "stage8_final.parquet",
    "stage9_candidates": INTERMEDIATE_DIR / "stage9_llm_candidates.parquet",
}


def parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def parquet_columns(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_columns


def build_unified_observations() -> int:
    unified_cols = pq.ParquetFile(UNIFIED_INPUT).schema.names

    select_exprs = []
    for col in unified_cols:
        if col == "scrape_date":
            select_exprs.append("o.scrape_date AS scrape_date")
        else:
            select_exprs.append(f"u.{col}")

    sql = f"""
    COPY (
      SELECT {", ".join(select_exprs)}
      FROM read_parquet('{OBS_INPUT}') AS o
      INNER JOIN read_parquet('{UNIFIED_INPUT}') AS u USING (uid)
    )
    TO '{OBS_OUTPUT}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """

    duckdb.execute(sql)
    return parquet_rows(OBS_OUTPUT)


def compute_quality_report() -> dict:
    q = f"""
    WITH unified AS (
      SELECT * FROM read_parquet('{UNIFIED_OUTPUT}')
    )
    SELECT
      count(*) AS total_rows,
      sum(CASE WHEN is_swe THEN 1 ELSE 0 END) AS total_swe,
      sum(CASE WHEN is_control THEN 1 ELSE 0 END) AS total_control,
      sum(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) AS total_adjacent,
      sum(CASE WHEN seniority_imputed = 'unknown' THEN 1 ELSE 0 END) AS seniority_unknown,
      sum(CASE WHEN is_aggregator THEN 1 ELSE 0 END) AS aggregators,
      sum(CASE WHEN date_flag != 'ok' THEN 1 ELSE 0 END) AS date_flagged,
      sum(CASE WHEN is_english = false THEN 1 ELSE 0 END) AS non_english,
      sum(CASE WHEN ghost_job_risk != 'low' THEN 1 ELSE 0 END) AS ghost_flagged
    FROM unified
    """
    totals = duckdb.execute(q).fetchone()

    source_counts = duckdb.execute(
        f"""
        SELECT source, source_platform, count(*) AS n
        FROM read_parquet('{UNIFIED_OUTPUT}')
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchall()

    swe_source_counts = duckdb.execute(
        f"""
        SELECT source, source_platform, count(*) AS n
        FROM read_parquet('{UNIFIED_OUTPUT}')
        WHERE is_swe
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchall()

    seniority_swe = duckdb.execute(
        f"""
        SELECT seniority_imputed, count(*) AS n
        FROM read_parquet('{UNIFIED_OUTPUT}')
        WHERE is_swe
        GROUP BY 1
        ORDER BY n DESC, seniority_imputed
        """
    ).fetchall()

    date_flags = duckdb.execute(
        f"""
        SELECT source, source_platform, date_flag, count(*) AS n
        FROM read_parquet('{UNIFIED_OUTPUT}')
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, n DESC
        """
    ).fetchall()

    return {
        "pipeline_version": "3.0",
        "run_date": time.strftime("%Y-%m-%d"),
        "rule_based_only": True,
        "row_counts": {
            name: parquet_rows(path) for name, path in STAGE_INPUTS.items() if path.exists()
        },
        "columns": {
            "unified": parquet_columns(UNIFIED_OUTPUT),
            "unified_observations": parquet_columns(OBS_OUTPUT),
        },
        "funnel": {
            "final_unified": totals[0],
            "final_observations": parquet_rows(OBS_OUTPUT),
            "final_swe": totals[1],
            "final_control": totals[2],
            "final_adjacent": totals[3],
        },
        "classification_rates": {
            "seniority_unknown_rate_rules": round(totals[4] / totals[0], 4) if totals[0] else None,
        },
        "quality_flags": {
            "aggregators": totals[5],
            "date_flagged": totals[6],
            "non_english": totals[7],
            "ghost_flagged": totals[8],
        },
        "source_counts": [
            {
                "source": source,
                "source_platform": platform,
                "rows": n,
            }
            for source, platform, n in source_counts
        ],
        "swe_by_source": [
            {
                "source": source,
                "source_platform": platform,
                "rows": n,
            }
            for source, platform, n in swe_source_counts
        ],
        "seniority_distribution_swe": [
            {"seniority_imputed": seniority, "rows": n}
            for seniority, n in seniority_swe
        ],
        "date_flags": [
            {
                "source": source,
                "source_platform": platform,
                "date_flag": flag,
                "rows": n,
            }
            for source, platform, flag, n in date_flags
        ],
    }


def write_log(report: dict) -> None:
    lines = [
        "PREPROCESSING PIPELINE LOG",
        "==========================",
        f"Pipeline version: {report['pipeline_version']}",
        f"Run date: {report['run_date']}",
        "",
        "ROW COUNTS",
        "----------",
    ]

    for name, count in report["row_counts"].items():
        lines.append(f"{name:<20} {count:>10,}")

    lines.extend(
        [
            "",
            "FINAL OUTPUTS",
            "-------------",
            f"unified.parquet rows:             {report['funnel']['final_unified']:>10,}",
            f"unified_observations.parquet rows:{report['funnel']['final_observations']:>10,}",
            f"SWE postings:                     {report['funnel']['final_swe']:>10,}",
            f"Control postings:                 {report['funnel']['final_control']:>10,}",
            f"SWE-adjacent postings:            {report['funnel']['final_adjacent']:>10,}",
            "",
            "QUALITY FLAGS",
            "-------------",
            f"Seniority unknown rate (rules): {report['classification_rates']['seniority_unknown_rate_rules']}",
            f"Aggregators:                   {report['quality_flags']['aggregators']:>10,}",
            f"Date flagged:                  {report['quality_flags']['date_flagged']:>10,}",
            f"Non-English:                   {report['quality_flags']['non_english']:>10,}",
            f"Ghost flagged:                 {report['quality_flags']['ghost_flagged']:>10,}",
            "",
            "SOURCE COUNTS",
            "-------------",
        ]
    )

    for row in report["source_counts"]:
        lines.append(
            f"{row['source']} | {row['source_platform']:<9} {row['rows']:>10,}"
        )

    lines.extend(["", "SWE BY SOURCE", "-------------"])
    for row in report["swe_by_source"]:
        lines.append(
            f"{row['source']} | {row['source_platform']:<9} {row['rows']:>10,}"
        )

    lines.extend(["", "SENIORITY (SWE ONLY)", "--------------------"])
    for row in report["seniority_distribution_swe"]:
        lines.append(f"{row['seniority_imputed']:<12} {row['rows']:>10,}")

    LOG_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FINAL OUTPUT: unified.parquet + unified_observations.parquet")
    print("=" * 70)

    print(f"[1/4] Copying {UNIFIED_INPUT} -> {UNIFIED_OUTPUT}")
    shutil.copy2(UNIFIED_INPUT, UNIFIED_OUTPUT)

    print(f"[2/4] Building {OBS_OUTPUT.name}")
    obs_rows = build_unified_observations()
    print(f"  Observation rows written: {obs_rows:,}")

    print(f"[3/4] Writing {QUALITY_OUTPUT.name}")
    report = compute_quality_report()
    QUALITY_OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"[4/4] Writing {LOG_OUTPUT.name}")
    write_log(report)

    elapsed = time.time() - t0
    print(f"Complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
