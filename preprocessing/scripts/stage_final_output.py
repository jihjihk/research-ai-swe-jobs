#!/usr/bin/env python3
"""
Stage Final: produce analysis-ready outputs from the LLM-integrated pipeline.

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

from io_utils import (
    cleanup_temp_file,
    parquet_columns,
    parquet_rows,
    prepare_temp_output,
    promote_temp_file,
)


PROJECT_ROOT = Path(__file__).parent.parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
DATA_DIR = PROJECT_ROOT / "data"

UNIFIED_INPUT = INTERMEDIATE_DIR / "stage10_llm_integrated.parquet"
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
    "stage9_extraction_candidates": INTERMEDIATE_DIR / "stage9_llm_extraction_candidates.parquet",
    "stage9_extraction_results": INTERMEDIATE_DIR / "stage9_llm_extraction_results.parquet",
    "stage9_cleaned": INTERMEDIATE_DIR / "stage9_llm_cleaned.parquet",
    "stage9_control_cohort": INTERMEDIATE_DIR / "stage9_control_cohort.parquet",
    "stage10_classification_results": INTERMEDIATE_DIR / "stage10_llm_classification_results.parquet",
    "stage10_integrated": INTERMEDIATE_DIR / "stage10_llm_integrated.parquet",
}

def build_unified_observations(unified_path: Path, output_path: Path) -> int:
    unified_cols = duckdb.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{unified_path}')"
    ).fetchall()

    select_exprs = []
    for col_name, *_ in unified_cols:
        if col_name == "scrape_date":
            select_exprs.append("o.scrape_date AS scrape_date")
        else:
            select_exprs.append(f"u.{col_name}")

    sql = f"""
    COPY (
      SELECT {", ".join(select_exprs)}
      FROM read_parquet('{OBS_INPUT}') AS o
      INNER JOIN read_parquet('{unified_path}') AS u USING (uid)
    )
    TO '{output_path}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """

    duckdb.execute(sql)
    return parquet_rows(output_path)

def compute_quality_report(unified_path: Path, observations_path: Path) -> dict:
    q = f"""
    WITH unified AS (
      SELECT * FROM read_parquet('{unified_path}')
    )
    SELECT
      count(*) AS total_rows,
      sum(CASE WHEN is_swe THEN 1 ELSE 0 END) AS total_swe,
      sum(CASE WHEN is_control THEN 1 ELSE 0 END) AS total_control,
      sum(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) AS total_adjacent,
      sum(CASE WHEN seniority_imputed = 'unknown' THEN 1 ELSE 0 END) AS seniority_unknown,
      sum(CASE WHEN seniority_llm = 'unknown' THEN 1 ELSE 0 END) AS seniority_unknown_llm,
      sum(CASE WHEN seniority_llm IS NOT NULL THEN 1 ELSE 0 END) AS seniority_llm_non_null,
      sum(CASE WHEN description_core_llm IS NOT NULL THEN 1 ELSE 0 END) AS extraction_non_null,
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
        FROM read_parquet('{unified_path}')
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchall()

    swe_source_counts = duckdb.execute(
        f"""
        SELECT source, source_platform, count(*) AS n
        FROM read_parquet('{unified_path}')
        WHERE is_swe
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchall()

    seniority_swe = duckdb.execute(
        f"""
        SELECT seniority_imputed, count(*) AS n
        FROM read_parquet('{unified_path}')
        WHERE is_swe
        GROUP BY 1
        ORDER BY n DESC, seniority_imputed
        """
    ).fetchall()

    date_flags = duckdb.execute(
        f"""
        SELECT source, source_platform, date_flag, count(*) AS n
        FROM read_parquet('{unified_path}')
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, n DESC
        """
    ).fetchall()

    return {
        "pipeline_version": "3.0",
        "run_date": time.strftime("%Y-%m-%d"),
        "rule_based_only": False,
        "llm_augmented": True,
        "row_counts": {
            name: parquet_rows(path) for name, path in STAGE_INPUTS.items() if path.exists()
        },
        "columns": {
            "unified": parquet_columns(unified_path),
            "unified_observations": parquet_columns(observations_path),
        },
        "funnel": {
            "final_unified": totals[0],
            "final_observations": parquet_rows(observations_path),
            "final_swe": totals[1],
            "final_control": totals[2],
            "final_adjacent": totals[3],
        },
        "classification_rates": {
            "seniority_unknown_rate_rules": round(totals[4] / totals[0], 4) if totals[0] else None,
            "seniority_unknown_rate_llm": round(totals[5] / totals[6], 4) if totals[6] else None,
            "seniority_llm_coverage": round(totals[6] / totals[0], 4) if totals[0] else None,
            "description_core_llm_coverage": round(totals[7] / totals[0], 4) if totals[0] else None,
        },
        "quality_flags": {
            "aggregators": totals[8],
            "date_flagged": totals[9],
            "non_english": totals[10],
            "ghost_flagged": totals[11],
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


def build_log_text(report: dict) -> str:
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
            f"Seniority unknown rate (LLM):   {report['classification_rates']['seniority_unknown_rate_llm']}",
            f"Seniority LLM coverage:         {report['classification_rates']['seniority_llm_coverage']}",
            f"Description-core LLM coverage:  {report['classification_rates']['description_core_llm_coverage']}",
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

    return "\n".join(lines) + "\n"


def main() -> None:
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FINAL OUTPUT: unified.parquet + unified_observations.parquet")
    print("=" * 70)

    tmp_unified_output = prepare_temp_output(UNIFIED_OUTPUT)
    tmp_obs_output = prepare_temp_output(OBS_OUTPUT)
    tmp_quality_output = prepare_temp_output(QUALITY_OUTPUT)
    tmp_log_output = prepare_temp_output(LOG_OUTPUT)

    try:
        print(f"[1/4] Copying {UNIFIED_INPUT} -> {tmp_unified_output}")
        shutil.copy2(UNIFIED_INPUT, tmp_unified_output)

        print(f"[2/4] Building {OBS_OUTPUT.name}")
        obs_rows = build_unified_observations(tmp_unified_output, tmp_obs_output)
        print(f"  Observation rows written: {obs_rows:,}")

        print(f"[3/4] Writing {QUALITY_OUTPUT.name}")
        report = compute_quality_report(tmp_unified_output, tmp_obs_output)
        tmp_quality_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        print(f"[4/4] Writing {LOG_OUTPUT.name}")
        tmp_log_output.write_text(build_log_text(report), encoding="utf-8")

        promote_temp_file(tmp_unified_output, UNIFIED_OUTPUT)
        promote_temp_file(tmp_obs_output, OBS_OUTPUT)
        promote_temp_file(tmp_quality_output, QUALITY_OUTPUT)
        promote_temp_file(tmp_log_output, LOG_OUTPUT)
    except Exception:
        cleanup_temp_file(tmp_unified_output)
        cleanup_temp_file(tmp_obs_output)
        cleanup_temp_file(tmp_quality_output)
        cleanup_temp_file(tmp_log_output)
        raise

    elapsed = time.time() - t0
    print(f"Complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
