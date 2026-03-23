#!/usr/bin/env python3
"""
V2 Preprocessing Pipeline Runner

Executes all stages in dependency order, validates intermediate outputs,
and produces the final data/unified.parquet.

Usage:
    python preprocessing/run_pipeline.py                  # Full run
    python preprocessing/run_pipeline.py --from-stage 3   # Resume from stage 3

Each stage reads the previous stage's output and writes its own.
All stages use chunked pyarrow I/O (200K rows/batch) to stay under 31GB RAM.
"""

import argparse
import subprocess
import sys
import time
import json
import shutil
import logging
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "preprocessing" / "scripts"
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline_run.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Pipeline stage definitions
STAGES = [
    {
        "num": 1,
        "name": "Ingest & Schema Unification",
        "script": "stage1_ingest.py",
        "output": "stage1_unified.parquet",
        "min_rows": 1_200_000,
    },
    {
        "num": 2,
        "name": "Aggregator Handling",
        "script": "stage2_aggregators.py",
        "output": "stage2_aggregators.parquet",
        "min_rows": 1_200_000,
        "check_col": "company_name_effective",
    },
    {
        "num": 3,
        "name": "Boilerplate Removal",
        "script": "stage3_boilerplate.py",
        "output": "stage3_boilerplate.parquet",
        "min_rows": 1_200_000,
        "check_col": "description_core",
    },
    {
        "num": 4,
        "name": "Company Canonicalization + Deduplication",
        "script": "stage4_dedup.py",
        "output": "stage4_dedup.parquet",
        "min_rows": 1_000_000,
        "check_col": "company_name_canonical",
    },
    {
        "num": 5,
        "name": "Classification (SWE + Seniority)",
        "script": "stage5_classification.py",
        "output": "stage5_classification.parquet",
        "min_rows": 1_000_000,
        "check_col": "seniority_final",
    },
    {
        "num": "6-8",
        "name": "Normalization + Temporal + Quality Flags",
        "script": "stage678_normalize_temporal_flags.py",
        "output": "stage8_final.parquet",
        "min_rows": 1_000_000,
        "check_col": "period",
    },
    {
        "num": 9,
        "name": "LLM Routing / Pre-filtering",
        "script": "stage9_llm_prefilter.py",
        "output": "stage9_llm_candidates.parquet",
        "min_rows": 1,
        "check_col": "needs_llm_extraction",
    },
    {
        "num": 10,
        "name": "LLM Task Execution",
        "script": "stage10_llm_classify.py",
        "output": "stage10_llm_results.parquet",
        "min_rows": 1,
        "check_col": "description_hash",
    },
    {
        "num": 11,
        "name": "LLM Response Integration",
        "script": "stage11_llm_integrate.py",
        "output": "stage11_llm_integrated.parquet",
        "min_rows": 1_000_000,
        "check_col": "description_core_llm",
    },
    {
        "num": "final",
        "name": "Final Output Generation",
        "script": "stage_final_output.py",
        "output": None,  # Writes to data/unified.parquet directly
    },
]


def validate_output(stage: dict) -> bool:
    """Check that a stage's output exists and meets minimum requirements."""
    if stage["output"] is None:
        # Final stage writes to data/
        output = DATA_DIR / "unified.parquet"
    else:
        output = INTERMEDIATE_DIR / stage["output"]

    if not output.exists():
        log.error(f"  Output not found: {output}")
        return False

    try:
        pf = pq.ParquetFile(output)
        rows = pf.metadata.num_rows
        cols = pf.metadata.num_columns

        min_rows = stage.get("min_rows", 0)
        if rows < min_rows:
            log.error(f"  Too few rows: {rows:,} (expected >= {min_rows:,})")
            return False

        check_col = stage.get("check_col")
        if check_col and check_col not in pf.schema.names:
            log.error(f"  Missing expected column: {check_col}")
            return False

        log.info(f"  Validated: {rows:,} rows, {cols} columns")
        return True

    except Exception as e:
        log.error(f"  Validation failed: {e}")
        return False


def run_stage(stage: dict) -> bool:
    """Run a single pipeline stage."""
    script = SCRIPTS_DIR / stage["script"]
    if not script.exists():
        log.error(f"  Script not found: {script}")
        return False

    log.info(f"  Running {script.name}...")
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True,
        timeout=3600,  # 1 hour max per stage
    )

    elapsed = time.time() - t0

    if result.returncode != 0:
        log.error(f"  FAILED (exit code {result.returncode}) after {elapsed:.0f}s")
        log.error(f"  stderr: {result.stderr[-1000:]}")
        return False

    log.info(f"  Completed in {elapsed:.0f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run V2 preprocessing pipeline")
    parser.add_argument("--from-stage", type=str, default="1",
                        help="Start from this stage number (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just validate existing outputs, don't run stages")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("V2 PREPROCESSING PIPELINE")
    log.info("=" * 60)

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    start_from = args.from_stage
    started = False
    total_t0 = time.time()

    for stage in STAGES:
        stage_num = str(stage["num"])

        # Skip stages before --from-stage
        if not started:
            if stage_num == start_from:
                started = True
            else:
                log.info(f"\nStage {stage_num}: {stage['name']} — SKIPPED (--from-stage={start_from})")
                continue

        log.info(f"\n{'='*60}")
        log.info(f"Stage {stage_num}: {stage['name']}")
        log.info(f"{'='*60}")

        if args.dry_run:
            validate_output(stage)
            continue

        # Run
        if not run_stage(stage):
            log.error(f"\nPIPELINE FAILED at Stage {stage_num}")
            return 1

        # Validate
        if not validate_output(stage):
            log.error(f"\nPIPELINE FAILED: Stage {stage_num} output validation failed")
            return 1

    total_elapsed = time.time() - total_t0
    log.info(f"\n{'='*60}")
    log.info(f"PIPELINE COMPLETE in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    log.info(f"{'='*60}")

    # Final summary
    if not args.dry_run:
        final = DATA_DIR / "unified.parquet"
        if final.exists():
            pf = pq.ParquetFile(final)
            log.info(f"\nFinal output: {final}")
            log.info(f"  Rows: {pf.metadata.num_rows:,}")
            log.info(f"  Columns: {pf.metadata.num_columns}")
            log.info(f"  Size: {final.stat().st_size / 1e9:.2f} GB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
