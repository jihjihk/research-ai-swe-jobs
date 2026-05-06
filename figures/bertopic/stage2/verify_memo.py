"""
Helper for the orchestrator to verify a Stage 2 sub-agent memo.

`verify(task_id)` reads `memos/<task-id>.md` and the artifacts it claims
to have produced (per `STAGE2_TASK_SPECS.md`), then prints:
  - which artifacts exist and their schemas
  - first few rows of each parquet for spot-checking
  - hash bundle from stage1_freeze.json for cross-checking sub-agent
    quotes

The orchestrator's role is to read the memo, run this helper to surface
the supporting artifacts, then write a verification entry to
`prereg_log.md`. The actual judgment (advocacy language, methodology
match, claim spot-check) is the orchestrator's, not this script's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

from figures.bertopic import config


_TASK_ARTIFACTS: dict[str, list[Path]] = {
    "t-axis": [
        config.BERTOPIC_DATA_DIR / "axes.parquet",
        config.BERTOPIC_DATA_DIR / "axis_projections.parquet",
        config.BERTOPIC_DATA_DIR / "cluster_axis_profile.parquet",
    ],
    "t-boundary": [
        config.BERTOPIC_DATA_DIR / "boundary_postings.parquet",
    ],
    "t-drift": [
        config.BERTOPIC_DATA_DIR / "centroid_drift.parquet",
    ],
    "t-weat": [
        config.BERTOPIC_DATA_DIR / "weat_results.parquet",
    ],
    "t-anchor": [
        config.BERTOPIC_DATA_DIR / "anchor_neighborhoods.parquet",
    ],
    "t-bootstrap": [
        config.BERTOPIC_DATA_DIR / "stability.parquet",
    ],
    "t-method": [
        config.BERTOPIC_DATA_DIR / "method_comparison.parquet",
    ],
    "t-quality": [
        config.BERTOPIC_DATA_DIR / "topic_quality.parquet",
    ],
    "t-ablations": [
        config.BERTOPIC_DATA_DIR / "ablations.parquet",
        config.BERTOPIC_DATA_DIR / "t6_robustness.parquet",
    ],
}


def verify(task_id: str) -> None:
    norm = task_id.lower().replace("_", "-")
    if norm not in _TASK_ARTIFACTS:
        raise SystemExit(f"unknown task id: {task_id}")

    memo_path = config.MEMOS_DIR / f"{norm.replace('-', '_')}.md"
    print(f"== Memo: {memo_path}")
    if memo_path.exists():
        size = memo_path.stat().st_size
        print(f"  exists, {size} bytes")
    else:
        print(f"  MISSING")

    print("\n== Frozen Stage 1 hash bundle:")
    if config.STAGE1_FREEZE_JSON.exists():
        bundle = json.loads(config.STAGE1_FREEZE_JSON.read_text())
        for key in ("config_hash", "sample_hash", "embeddings_cache_hash",
                    "model_hash", "assignments_hash"):
            print(f"  {key} = {bundle.get(key, '(missing)')}")
        print(f"  headline_k = {bundle.get('headline_k')}")
    else:
        print("  stage1_freeze.json missing — Stage 2 should not have run yet")

    print("\n== Artifacts claimed by spec:")
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")
    for path in _TASK_ARTIFACTS[norm]:
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        size = path.stat().st_size
        print(f"  {path} ({size} bytes)")
        if path.suffix == ".parquet":
            n = con.execute(f"SELECT count(*) FROM '{path}'").fetchone()[0]
            print(f"    rows: {n:,}")
            schema = con.execute(
                f"DESCRIBE SELECT * FROM '{path}'"
            ).fetchall()
            print("    columns:")
            for col_name, col_type, *_ in schema:
                print(f"      {col_name}: {col_type}")
            sample = con.execute(f"SELECT * FROM '{path}' LIMIT 3").fetchall()
            print("    head:")
            for row in sample:
                print(f"      {row}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_memo.py <task-id>")
    verify(sys.argv[1])
