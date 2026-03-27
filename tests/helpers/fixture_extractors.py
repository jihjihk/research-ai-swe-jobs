"""Utilities for promoting reviewed real rows into sampled fixtures.

Keep these helpers tiny. Query with DuckDB, extract the smallest useful row set,
and record why the expected answer is correct in the manifest.
"""

from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


def extract_parquet_query(*, source_path: Path, query: str, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cursor = duckdb.execute(query.format(source_path=str(source_path)))
    rows = cursor.fetchall()
    if not rows:
        return 0
    columns = [desc[0] for desc in cursor.description]
    table = pa.Table.from_pylist([dict(zip(columns, row)) for row in rows])
    pq.write_table(table, output_path)
    return table.num_rows
