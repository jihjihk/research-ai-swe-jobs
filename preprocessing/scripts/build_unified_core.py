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
# The analysis frame is the intersection of:
#   1. The Stage 9 deterministic balanced core frame (selected_for_llm_frame).
#   2. Rows with a confirmed cohort label: either the LLM classified them as
#      SWE / SWE-adjacent, or the Stage 5 rule labelled them as control.
# Rows in the core frame that the LLM rejected as NOT_SWE on the rule-SWE side
# are dropped from unified_core (they remain in unified.parquet for audit).
CORE_ROW_FILTER = (
    f"{CORE_ROW_FILTER_COLUMN} = TRUE "
    "AND (is_swe_combined_llm = TRUE OR is_control = TRUE)"
)

# Columns kept in unified_core.parquet, grouped by semantic purpose.
# Each entry is (output_name, source_expr). source_expr is either a bare
# column name from unified.parquet (pass-through) or a SQL expression
# referencing one or more unified.parquet columns. This list is the single
# source of truth — the schema doc and tests read it.
#
# Naming note: in unified.parquet, `is_swe` is the Stage 5 narrow rule column
# (SWE only, excludes adjacent). In unified_core.parquet, `is_swe` is the LLM
# combined verdict (SWE union SWE_ADJACENT). The two files use the same name
# for different semantics; analyses mixing the files must adjust accordingly.
# Pass-through entries: (output_name, source_column_in_unified).
# The only rename is is_swe_combined_llm → is_swe (cleaner name in core where
# the narrow Stage 5 is_swe column has been dropped).
CORE_PASSTHROUGH: list[tuple[str, str]] = [
    # --- Identity ---
    ("uid", "uid"),
    # --- Source & time ---
    ("source", "source"),
    ("period", "period"),
    ("date_posted", "date_posted"),
    ("scrape_date", "scrape_date"),
    # --- Job content ---
    ("title", "title"),
    ("description", "description"),
    ("description_core_llm", "description_core_llm"),
    # --- Company ---
    ("company_name_effective", "company_name_effective"),
    ("company_name_canonical", "company_name_canonical"),
    ("is_aggregator", "is_aggregator"),
    ("company_industry", "company_industry"),
    ("company_size", "company_size"),
    # --- Occupation / SWE classification ---
    # is_swe here is the LLM combined verdict.
    ("is_swe", "is_swe_combined_llm"),
    ("is_control", "is_control"),
    # --- Seniority ---
    ("seniority_final", "seniority_final"),
    ("seniority_final_source", "seniority_final_source"),
    ("seniority_rule", "seniority_rule"),
    ("seniority_rule_source", "seniority_rule_source"),
    ("seniority_native", "seniority_native"),
    # --- Years of experience ---
    ("yoe_extracted", "yoe_extracted"),
    ("yoe_min_years_llm", "yoe_min_years_llm"),
    # --- Geography ---
    ("location", "location"),
    ("city_extracted", "city_extracted"),
    ("state_normalized", "state_normalized"),
    ("metro_area", "metro_area"),
    ("is_remote_inferred", "is_remote_inferred"),
    ("is_multi_location", "is_multi_location"),
    # --- Quality / ghost ---
    ("date_flag", "date_flag"),
    ("ghost_assessment_llm", "ghost_assessment_llm"),
]

# Computed entries: (output_name, sql_expression_template, source_columns).
# The expression uses {col} placeholders so we can qualify column references
# (e.g. with `u.`) when joining for the observations file.
CORE_COMPUTED: list[tuple[str, str, list[str]]] = [
    (
        "has_llm_extraction",
        "{llm_extraction_coverage} = 'labeled'",
        ["llm_extraction_coverage"],
    ),
    (
        "has_llm_classification",
        "{llm_classification_coverage} = 'labeled'",
        ["llm_classification_coverage"],
    ),
]

# Output column order — the schema doc and tests read this.
CORE_COLUMNS: list[str] = (
    [name for name, _ in CORE_PASSTHROUGH]
    + [name for name, _, _ in CORE_COMPUTED]
)

# Source columns from unified.parquet that the projection depends on.
SOURCE_COLUMNS_REQUIRED: set[str] = (
    {src for _, src in CORE_PASSTHROUGH}
    | {col for _, _, cols in CORE_COMPUTED for col in cols}
)


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
    missing = sorted(SOURCE_COLUMNS_REQUIRED - available)
    if missing:
        raise ValueError(
            f"unified parquet at {unified_path} is missing source columns: {missing}. "
            "Update CORE_COLUMNS or rerun the pipeline through stage 10."
        )
    if CORE_ROW_FILTER_COLUMN not in available:
        raise ValueError(
            f"unified parquet at {unified_path} is missing the filter column "
            f"{CORE_ROW_FILTER_COLUMN!r}."
        )


def _projection_select_list(qualifier: str = "") -> str:
    """Build SELECT expressions for the core projection.

    qualifier: optional column-reference prefix (e.g. "u." for joins). Empty
    when selecting from a single table.
    """
    parts = []
    for output_name, source_col in CORE_PASSTHROUGH:
        parts.append(f"{qualifier}{source_col} AS {output_name}")
    for output_name, expr_template, src_cols in CORE_COMPUTED:
        substitutions = {col: f"{qualifier}{col}" for col in src_cols}
        expr = expr_template.format(**substitutions)
        parts.append(f"({expr}) AS {output_name}")
    return ", ".join(parts)


def _observations_select_exprs() -> str:
    """Build SELECT expressions for the joined observations projection.

    The core file is already projected (output column names); scrape_date
    comes from the observations table (alias `o`), everything else from
    the core file (alias `u`).
    """
    parts = []
    for output_name in CORE_COLUMNS:
        if output_name == "scrape_date":
            parts.append("o.scrape_date AS scrape_date")
        else:
            parts.append(f"u.{output_name} AS {output_name}")
    return ", ".join(parts)


def build_core(
    unified_path: Path,
    observations_path: Path,
    core_path: Path,
    core_observations_path: Path,
) -> dict:
    """Write the core files at the exact paths given. Caller handles tmp/promote."""
    validate_columns_present(unified_path)

    core_select = _projection_select_list()

    duckdb.execute(
        f"""
        COPY (
          SELECT {core_select}
          FROM read_parquet('{unified_path}')
          WHERE {CORE_ROW_FILTER}
        )
        TO '{core_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    select_exprs = _observations_select_exprs()
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
