from pathlib import Path

import duckdb
import pyarrow.parquet as pq


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema.names


def assert_parquet_readable(path: Path) -> None:
    pq.ParquetFile(path)


def assert_has_columns(path: Path, required_cols: list[str]) -> None:
    cols = set(parquet_columns(path))
    missing = [col for col in required_cols if col not in cols]
    assert not missing, f"Missing columns in {path}: {missing}"


def assert_row_count_equal(left: Path, right: Path) -> None:
    left_rows = pq.ParquetFile(left).metadata.num_rows
    right_rows = pq.ParquetFile(right).metadata.num_rows
    assert left_rows == right_rows, f"Row count mismatch: {left}={left_rows}, {right}={right_rows}"


def assert_unique(path: Path, columns: list[str]) -> None:
    cols = ", ".join(columns)
    row = duckdb.execute(
        f"""
        SELECT COUNT(*) AS total_rows, COUNT(DISTINCT ({cols})) AS distinct_rows
        FROM read_parquet('{path}')
        """
    ).fetchone()
    assert row is not None
    assert row[0] == row[1], f"Expected unique rows on {columns} in {path}, got {row[0]} vs {row[1]}"
