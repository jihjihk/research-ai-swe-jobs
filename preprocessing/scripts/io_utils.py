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
) -> None:
    tmp_path = prepare_temp_output(final_path)
    try:
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, tmp_path, compression=compression)
        promote_temp_file(tmp_path, final_path)
    except Exception:
        cleanup_temp_file(tmp_path)
        raise


def parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def parquet_columns(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_columns
