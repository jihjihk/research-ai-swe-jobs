from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def temp_path_for(path: Path, suffix: str = ".tmp") -> Path:
    return path.with_name(path.name + suffix)


def prepare_temp_output(final_path: Path, suffix: str = ".tmp") -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = temp_path_for(final_path, suffix=suffix)
    if tmp_path.exists():
        tmp_path.unlink()
    return tmp_path


def cleanup_temp_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def promote_temp_file(tmp_path: Path, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.replace(final_path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_text_atomic(text: str, final_path: Path, encoding: str = "utf-8") -> None:
    tmp_path = prepare_temp_output(final_path)
    try:
        tmp_path.write_text(text, encoding=encoding)
        promote_temp_file(tmp_path, final_path)
    except Exception:
        cleanup_temp_file(tmp_path)
        raise


def write_parquet_atomic(
    df: pd.DataFrame,
    final_path: Path,
    *,
    compression: str | None = None,
    chunk_size: int = 200_000,
) -> None:
    tmp_path = prepare_temp_output(final_path)
    try:
        if len(df) <= chunk_size:
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, tmp_path, compression=compression)
        else:
            # Chunked write: convert and flush one slice at a time so that
            # pandas and Arrow never both hold the full dataset.
            first_chunk = pa.Table.from_pandas(df.iloc[:chunk_size], preserve_index=False)
            schema = promote_null_schema(first_chunk.schema)
            writer = pq.ParquetWriter(tmp_path, schema, compression=compression)
            writer.write_table(first_chunk.cast(schema))
            del first_chunk
            for start in range(chunk_size, len(df), chunk_size):
                chunk = pa.Table.from_pandas(
                    df.iloc[start : start + chunk_size], preserve_index=False
                )
                writer.write_table(chunk.cast(schema))
                del chunk
            writer.close()
        promote_temp_file(tmp_path, final_path)
    except Exception:
        cleanup_temp_file(tmp_path)
        raise


def promote_null_schema(schema: pa.Schema, extra_fields: list[pa.Field] | None = None) -> pa.Schema:
    """Return *schema* with null-typed fields promoted to string.

    When reading parquet in batches, columns that are all-NA in a chunk get
    inferred as ``pa.null()`` which causes ParquetWriter schema mismatches
    between chunks.  Call this once on the input schema to build a stable
    output schema, optionally appending *extra_fields*.
    """
    fields = []
    for field in schema:
        if field.type == pa.null():
            fields.append(pa.field(field.name, pa.string()))
        else:
            fields.append(field)
    if extra_fields:
        fields.extend(extra_fields)
    return pa.schema(fields)


def parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def parquet_columns(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_columns
