#!/usr/bin/env python3
"""
Build unified_core.parquet from data/unified.parquet.

unified_core.parquet is the analysis-ready subset of unified.parquet:
  rows: selected_for_llm_frame = TRUE (Stage 9 balanced core frame)
  cols: the columns that are actually used in analysis; audit/routing
        internals are dropped.

Can be called in-process from stage_final_output.py or run standalone
to rebuild the core without re-running the final stage:

    ./.venv/bin/python preprocessing/scripts/build_unified_core.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

from io_utils import (
    cleanup_temp_file,
    prepare_temp_output,
    promote_temp_file,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_UNIFIED = DATA_DIR / "unified.parquet"
DEFAULT_OBSERVATIONS = DATA_DIR / "unified_observations.parquet"
DEFAULT_CORE = DATA_DIR / "unified_core.parquet"
DEFAULT_CORE_OBS = DATA_DIR / "unified_core_observations.parquet"

# ---------------------------------------------------------------------------
# Selection contract
# ---------------------------------------------------------------------------

CORE_ROW_FILTER_COLUMN = "selected_for_llm_frame"
CORE_ROW_FILTER = f"{CORE_ROW_FILTER_COLUMN} = TRUE"

# Columns kept in unified_core.parquet, grouped by semantic purpose.
# This list is the single source of truth — the schema doc and tests read it.
CORE_COLUMNS: list[str] = [
    # --- Identity ---
    "uid",
    # --- Source & time ---
    "source",
    "source_platform",
    "period",
    "date_posted",
    "scrape_date",
    # --- Job content ---
    "title",
    "description",
    "description_core_llm",
    "description_length",
    # --- Company ---
    "company_name",
    "company_name_effective",
    "company_name_canonical",
    "is_aggregator",
    "company_industry",
    "company_size",
    # --- Occupation / SWE classification ---
    "is_swe",
    "is_swe_adjacent",
    "is_control",
    "analysis_group",
    "swe_classification_tier",
    "swe_classification_llm",
    # --- Seniority ---
    "seniority_final",
    "seniority_final_source",
    "seniority_3level",
    "seniority_rule",
    "seniority_rule_source",
    "seniority_native",
    # --- Years of experience ---
    "yoe_extracted",
    "yoe_min_years_llm",
    # --- Geography ---
    "location",
    "city_extracted",
    "state_normalized",
    "metro_area",
    "is_remote_inferred",
    "is_multi_location",
    # --- Quality / ghost ---
    "is_english",
    "date_flag",
    "ghost_job_risk",
    "ghost_assessment_llm",
    # --- LLM coverage (residual, rows inside the core can still be
    # `deferred` / `skipped_short`; filter to `labeled` when reading LLM cols) ---
    "llm_extraction_coverage",
    "llm_classification_coverage",
]


def has_core_filter_column(unified_path: Path) -> bool:
    cols = {
        row[0]
        for row in duckdb.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{unified_path}')"
        ).fetchall()
    }
    return CORE_ROW_FILTER_COLUMN in cols


def validate_columns_present(unified_path: Path) -> None:
    available = {
        row[0]
        for row in duckdb.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{unified_path}')"
        ).fetchall()
    }
    missing = [c for c in CORE_COLUMNS if c not in available]
    if missing:
        raise ValueError(
            f"unified parquet at {unified_path} is missing CORE_COLUMNS: {missing}. "
            "Update CORE_COLUMNS or rerun the pipeline through stage 10."
        )
    if CORE_ROW_FILTER_COLUMN not in available:
        raise ValueError(
            f"unified parquet at {unified_path} is missing the filter column "
            f"{CORE_ROW_FILTER_COLUMN!r}."
        )


def _observations_select_exprs(core_cols: list[str]) -> str:
    parts = []
    for col_name in core_cols:
        if col_name == "scrape_date":
            parts.append("o.scrape_date AS scrape_date")
        else:
            parts.append(f"u.{col_name}")
    return ", ".join(parts)


def build_core(
    unified_path: Path,
    observations_path: Path,
    core_path: Path,
    core_observations_path: Path,
) -> dict:
    """Write the core files at the exact paths given. Caller handles tmp/promote."""
    validate_columns_present(unified_path)

    col_list = ", ".join(CORE_COLUMNS)

    duckdb.execute(
        f"""
        COPY (
          SELECT {col_list}
          FROM read_parquet('{unified_path}')
          WHERE {CORE_ROW_FILTER}
        )
        TO '{core_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    select_exprs = _observations_select_exprs(CORE_COLUMNS)
    duckdb.execute(
        f"""
        COPY (
          SELECT {select_exprs}
          FROM read_parquet('{observations_path}') AS o
          INNER JOIN read_parquet('{core_path}') AS u USING (uid)
        )
        TO '{core_observations_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    core_rows = duckdb.execute(
        f"SELECT count(*) FROM read_parquet('{core_path}')"
    ).fetchone()[0]
    core_obs_rows = duckdb.execute(
        f"SELECT count(*) FROM read_parquet('{core_observations_path}')"
    ).fetchone()[0]

    return {
        "row_filter": CORE_ROW_FILTER,
        "columns": list(CORE_COLUMNS),
        "n_columns": len(CORE_COLUMNS),
        "core_rows": core_rows,
        "core_observations_rows": core_obs_rows,
    }


def build_core_atomic(
    unified_path: Path,
    observations_path: Path,
    core_path: Path,
    core_observations_path: Path,
) -> dict:
    """Wrap build_core with tmp/promote for atomic writes."""
    tmp_core = prepare_temp_output(core_path)
    tmp_core_obs = prepare_temp_output(core_observations_path)
    try:
        summary = build_core(
            unified_path,
            observations_path,
            tmp_core,
            tmp_core_obs,
        )
        promote_temp_file(tmp_core, core_path)
        promote_temp_file(tmp_core_obs, core_observations_path)
    except Exception:
        cleanup_temp_file(tmp_core)
        cleanup_temp_file(tmp_core_obs)
        raise
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build unified_core.parquet from data/unified.parquet"
    )
    parser.add_argument("--unified", type=Path, default=DEFAULT_UNIFIED)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--out-core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--out-core-obs", type=Path, default=DEFAULT_CORE_OBS)
    args = parser.parse_args()

    for label, path in [
        ("unified", args.unified),
        ("observations", args.observations),
    ]:
        if not path.exists():
            print(f"error: {label} input not found: {path}", file=sys.stderr)
            return 1

    summary = build_core_atomic(
        args.unified,
        args.observations,
        args.out_core,
        args.out_core_obs,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
