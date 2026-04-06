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

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from backup_to_s3 import run_backup

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "preprocessing" / "scripts"
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

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
        "outputs": [
            {"path": INTERMEDIATE_DIR / "stage1_unified.parquet", "kind": "parquet", "min_rows": 1_200_000},
            {"path": INTERMEDIATE_DIR / "stage1_observations.parquet", "kind": "parquet", "min_rows": 1_200_000},
        ],
        "timeout_seconds": 2 * 3600,
    },
    {
        "num": 2,
        "name": "Aggregator Handling",
        "script": "stage2_aggregators.py",
        "outputs": [
            {
                "path": INTERMEDIATE_DIR / "stage2_aggregators.parquet",
                "kind": "parquet",
                "min_rows": 1_200_000,
                "check_col": "company_name_effective",
            }
        ],
        "timeout_seconds": 2 * 3600,
    },
    {
        "num": 3,
        "name": "Boilerplate Removal",
        "script": "stage3_boilerplate.py",
        "outputs": [
            {
                "path": INTERMEDIATE_DIR / "stage3_boilerplate.parquet",
                "kind": "parquet",
                "min_rows": 1_200_000,
                "check_col": "description_core",
            }
        ],
        "timeout_seconds": 2 * 3600,
    },
    {
        "num": 4,
        "name": "Company Canonicalization + Deduplication",
        "script": "stage4_dedup.py",
        "outputs": [
            {
                "path": INTERMEDIATE_DIR / "stage4_dedup.parquet",
                "kind": "parquet",
                "min_rows": 1_000_000,
                "check_col": "company_name_canonical",
            },
            {
                "path": INTERMEDIATE_DIR / "stage4_company_name_lookup.parquet",
                "kind": "parquet",
                "min_rows": 1,
                "check_col": "company_name_canonical",
            },
        ],
        "timeout_seconds": 2 * 3600,
    },
    {
        "num": 5,
        "name": "Classification (SWE + Seniority)",
        "script": "stage5_classification.py",
        "outputs": [
            {
                "path": INTERMEDIATE_DIR / "stage5_classification.parquet",
                "kind": "parquet",
                "min_rows": 1_000_000,
                "check_col": "seniority_final",
            }
        ],
        "timeout_seconds": 2 * 3600,
    },
    {
        "num": "6-8",
        "name": "Normalization + Temporal + Quality Flags",
        "script": "stage678_normalize_temporal_flags.py",
        "outputs": [
            {
                "path": INTERMEDIATE_DIR / "stage8_final.parquet",
                "kind": "parquet",
                "min_rows": 1_000_000,
                "check_col": "period",
            }
        ],
        "timeout_seconds": 2 * 3600,
    },
    {
        "num": 9,
        "name": "LLM Extraction + Cleaned Text Integration",
        "script": "stage9_llm_prefilter.py",
        "outputs": [
            {
                "path": INTERMEDIATE_DIR / "stage9_llm_extraction_candidates.parquet",
                "kind": "parquet",
                "min_rows": 1,
                "check_col": "extraction_input_hash",
            },
            {
                "path": INTERMEDIATE_DIR / "stage9_llm_extraction_results.parquet",
                "kind": "parquet",
                "min_rows": 1,
                "check_col": "extraction_response_json",
            },
            {
                "path": INTERMEDIATE_DIR / "stage9_llm_cleaned.parquet",
                "kind": "parquet",
                "min_rows": 1_000_000,
                "check_col": "description_core_llm",
            },
            {
                "path": INTERMEDIATE_DIR / "stage9_control_cohort.parquet",
                "kind": "parquet",
                "min_rows": 1,
                "check_col": "selected_for_control_cohort",
            },
        ],
        "timeout_seconds": 2 * 3600,
    },
    {
        "num": 10,
        "name": "LLM Classification + Final Integration",
        "script": "stage10_llm_classify.py",
        "outputs": [
            {
                "path": INTERMEDIATE_DIR / "stage10_llm_classification_results.parquet",
                "kind": "parquet",
                "min_rows": 1,
                "check_col": "classification_input_hash",
            },
            {
                "path": INTERMEDIATE_DIR / "stage10_llm_integrated.parquet",
                "kind": "parquet",
                "min_rows": 1_000_000,
                "check_col": "description_core_llm",
            }
        ],
        "timeout_seconds": 24 * 3600,
    },
    {
        "num": "final",
        "name": "Final Output Generation",
        "script": "stage_final_output.py",
        "outputs": [
            {"path": DATA_DIR / "unified.parquet", "kind": "parquet", "min_rows": 1_000_000},
            {"path": DATA_DIR / "unified_observations.parquet", "kind": "parquet", "min_rows": 1_000_000},
            {"path": DATA_DIR / "quality_report.json", "kind": "text"},
            {"path": DATA_DIR / "preprocessing_log.txt", "kind": "text"},
        ],
        "timeout_seconds": 6 * 3600,
    },
]


def validate_output(stage: dict) -> bool:
    """Check that a stage's outputs exist and meet minimum requirements."""
    for output_spec in stage["outputs"]:
        output = output_spec["path"]
        if not output.exists():
            log.error(f"  Output not found: {output}")
            return False

        if output_spec["kind"] != "parquet":
            log.info(f"  Validated: {output}")
            continue

        try:
            pf = pq.ParquetFile(output)
            rows = pf.metadata.num_rows
            cols = pf.metadata.num_columns

            min_rows = output_spec.get("min_rows", 0)
            if rows < min_rows:
                log.error(f"  Too few rows in {output.name}: {rows:,} (expected >= {min_rows:,})")
                return False

            check_col = output_spec.get("check_col")
            if check_col and check_col not in pf.schema.names:
                log.error(f"  Missing expected column in {output.name}: {check_col}")
                return False

            log.info(f"  Validated {output.name}: {rows:,} rows, {cols} columns")
        except Exception as e:
            log.error(f"  Validation failed for {output.name}: {e}")
            return False

    return True


LLM_STAGES = {9, 10}


def run_stage(stage: dict, extra_args: list[str] | None = None) -> bool:
    """Run a single pipeline stage."""
    script = SCRIPTS_DIR / stage["script"]
    if not script.exists():
        log.error(f"  Script not found: {script}")
        return False

    log.info(f"  Running {script.name}...")
    t0 = time.time()
    timeout_seconds = stage.get("timeout_seconds", 3600)

    cmd = [sys.executable, str(script)] + (extra_args or [])
    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
        timeout=timeout_seconds,
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
    parser.add_argument("--remote", action="store_true", default=False,
                        help="Run LLM commands on the remote EC2 instance via SSH (stages 9-10)")
    parser.add_argument("--llm-budget", type=int, default=None,
                        help=(
                            "Max new LLM calls per LLM stage (stages 9-10). "
                            "REQUIRED if running stages 9 or 10. Use 0 for cache-only."
                        ))
    parser.add_argument("--llm-budget-split", type=str, default=None,
                        help=(
                            "Budget split swe,swe_adjacent,control (default: 0.4,0.3,0.3). "
                            "Only applies when --llm-budget is provided."
                        ))
    parser.add_argument("--backup", action="store_true", default=False,
                        help="Back up final outputs and LLM cache to S3 after successful run")
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
        extra_args: list[str] = []
        if stage["num"] in LLM_STAGES:
            if args.remote:
                extra_args.append("--remote")
            if args.llm_budget is None:
                log.error(
                    f"Stage {stage_num} requires --llm-budget (no default). "
                    "Pass --llm-budget N to run_pipeline.py."
                )
                return 1
            extra_args.extend(["--llm-budget", str(args.llm_budget)])
            if args.llm_budget_split is not None:
                extra_args.extend(["--llm-budget-split", args.llm_budget_split])
        if not run_stage(stage, extra_args=extra_args or None):
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

        # S3 backup
        if args.backup:
            log.info(f"\n{'='*60}")
            log.info("BACKING UP TO S3")
            log.info(f"{'='*60}")
            if not run_backup():
                log.error("S3 backup failed (pipeline outputs are still valid locally)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
