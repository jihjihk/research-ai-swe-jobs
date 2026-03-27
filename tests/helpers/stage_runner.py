from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def patch_stage_dirs(monkeypatch, module, dirs: dict[str, Path]) -> None:
    if hasattr(module, "PROJECT_ROOT"):
        monkeypatch.setattr(module, "PROJECT_ROOT", dirs["root"])
    if hasattr(module, "DATA_DIR"):
        monkeypatch.setattr(module, "DATA_DIR", dirs["data"])
    if hasattr(module, "INTERMEDIATE_DIR"):
        monkeypatch.setattr(module, "INTERMEDIATE_DIR", dirs["intermediate"])
    if hasattr(module, "LOG_DIR"):
        monkeypatch.setattr(module, "LOG_DIR", dirs["logs"])
    if hasattr(module, "CACHE_DIR"):
        monkeypatch.setattr(module, "CACHE_DIR", dirs["cache"])


def write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
