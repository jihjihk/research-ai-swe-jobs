"""
Stage 0 pre-flight checks (§13.1 S0.4).

Validates that everything Stage 1 needs is in place: required columns
exist, embedding dimension is right, L2 norms are unit-ish, no nulls
where forbidden, sample sizes sit within ±20 % of expectation. Fails
loud on any mismatch.
"""

from __future__ import annotations

import sys

import duckdb
import numpy as np

from figures.bertopic import config
from figures.bertopic.embedding_cache import load_cache


REQUIRED_UNIFIED_CORE_COLUMNS = (
    "uid", "source", "period", "title",
    "description_core_llm", "job_description_embedding",
    "is_swe", "is_control",
    "seniority_final", "yoe_min_years_llm",
    "company_name_canonical", "is_aggregator", "metro_area",
    "date_flag", "has_llm_extraction", "has_llm_classification",
)

# §3.1: pre-cap counts give Sample A ≈ 50–55k post-cap. We allow ±20 %.
EXPECTED_SAMPLE_A_BOUNDS = (40_000, 70_000)
EXPECTED_SAMPLE_B_BOUNDS = (80_000, 140_000)


def _check_required_columns(con: duckdb.DuckDBPyConnection) -> None:
    schema = {
        r[0] for r in con.execute(
            f"DESCRIBE SELECT * FROM '{config.UNIFIED_CORE_PATH}'"
        ).fetchall()
    }
    missing = [c for c in REQUIRED_UNIFIED_CORE_COLUMNS if c not in schema]
    if missing:
        raise RuntimeError(
            f"unified_core.parquet missing required columns: {missing}"
        )


def _check_embedding_dim_and_norm(matrix: np.ndarray) -> None:
    if matrix.shape[1] != config.EMBEDDING_DIMS:
        raise RuntimeError(
            f"Cache has shape {matrix.shape}; "
            f"expected (_, {config.EMBEDDING_DIMS})"
        )
    if matrix.dtype != np.float32:
        raise RuntimeError(
            f"Cache dtype {matrix.dtype} != float32"
        )
    norms = np.linalg.norm(matrix, axis=1)
    lo, hi = config.EMBEDDING_NORM_TOLERANCE
    bad = np.where((norms < lo) | (norms > hi))[0]
    if bad.size:
        raise RuntimeError(
            f"{bad.size} cache rows have non-unit L2 norm "
            f"(min={norms.min():.4f}, max={norms.max():.4f})"
        )


def _check_sample_sizes(con: duckdb.DuckDBPyConnection) -> None:
    n_a = con.execute(
        f"SELECT count(*) FROM '{config.SAMPLE_A_PATH}'"
    ).fetchone()[0]
    n_b = con.execute(
        f"SELECT count(*) FROM '{config.SAMPLE_B_PATH}'"
    ).fetchone()[0]

    a_lo, a_hi = EXPECTED_SAMPLE_A_BOUNDS
    if not (a_lo <= n_a <= a_hi):
        raise RuntimeError(
            f"Sample A size {n_a} out of expected range [{a_lo}, {a_hi}]"
        )
    b_lo, b_hi = EXPECTED_SAMPLE_B_BOUNDS
    if not (b_lo <= n_b <= b_hi):
        raise RuntimeError(
            f"Sample B size {n_b} out of expected range [{b_lo}, {b_hi}]"
        )
    print(f"  Sample A: {n_a:,} rows (in [{a_lo}, {a_hi}])")
    print(f"  Sample B: {n_b:,} rows (in [{b_lo}, {b_hi}])")


def _check_no_forbidden_nulls(con: duckdb.DuckDBPyConnection) -> None:
    for path, name in (
        (config.SAMPLE_A_PATH, "Sample A"),
        (config.SAMPLE_B_PATH, "Sample B"),
    ):
        nulls = con.execute(f"""
            SELECT
                sum(CASE WHEN uid IS NULL THEN 1 ELSE 0 END) AS uid_null,
                sum(CASE WHEN description_core_llm IS NULL THEN 1 ELSE 0 END) AS sub_null,
                sum(CASE WHEN job_description_embedding IS NULL THEN 1 ELSE 0 END) AS emb_null,
                sum(CASE WHEN length(description_core_llm) < {config.SUBSTRATE_MIN_LENGTH} THEN 1 ELSE 0 END) AS short
            FROM '{path}'
        """).fetchone()
        if any(nulls):
            raise RuntimeError(
                f"{name} has forbidden values: "
                f"uid_null={nulls[0]}, sub_null={nulls[1]}, "
                f"emb_null={nulls[2]}, short_substrate={nulls[3]}"
            )


def _check_anchor_index(matrix: np.ndarray, key_to_row: dict[str, int]) -> None:
    expected_anchors = config.all_anchor_strings()
    missing = [k for k in expected_anchors if k not in key_to_row]
    if missing:
        raise RuntimeError(
            f"Anchor cache missing {len(missing)} keys: {missing[:3]}..."
        )
    n_total = matrix.shape[0]
    bad = [k for k, idx in key_to_row.items() if not (0 <= idx < n_total)]
    if bad:
        raise RuntimeError(f"Index has out-of-range row_index: {bad[:3]}")


def main() -> None:
    print("Pre-flight (§13.1 S0.4):")

    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")

    print("  - required columns in unified_core.parquet…")
    _check_required_columns(con)

    print("  - sample sizes within expected bounds…")
    _check_sample_sizes(con)

    print("  - no forbidden nulls in sample parquets…")
    _check_no_forbidden_nulls(con)

    print("  - embeddings cache dimension and L2 norms…")
    matrix, key_to_row = load_cache()
    _check_embedding_dim_and_norm(matrix)
    print(f"    cache shape: {matrix.shape}, dtype: {matrix.dtype}")

    print("  - anchor index covers every config anchor…")
    _check_anchor_index(matrix, key_to_row)

    print("Pre-flight OK.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Pre-flight FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
