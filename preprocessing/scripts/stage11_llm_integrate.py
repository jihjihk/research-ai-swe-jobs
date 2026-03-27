#!/usr/bin/env python3
"""
Stage 11 compatibility shim.

The redesigned LLM flow materializes the posting-level artifact in Stage 10.
This script exists only to copy that Stage 10 output onto the legacy Stage 11
path for one transition cycle.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

from io_utils import cleanup_temp_file, prepare_temp_output, promote_temp_file


PROJECT_ROOT = Path(__file__).parent.parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

DEFAULT_INPUT_PATH = INTERMEDIATE_DIR / "stage10_llm_integrated.parquet"
DEFAULT_OUTPUT_PATH = INTERMEDIATE_DIR / "stage11_llm_integrated.parquet"


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "stage11_llm_integrate.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def yoe_contradiction(yoe_extracted: float | None, seniority_llm, seniority_imputed) -> bool:
    if yoe_extracted is None or yoe_extracted < 5:
        return False
    if seniority_llm == "entry":
        return True
    if seniority_llm is None and seniority_imputed == "entry":
        return True
    return False


def run_stage11(input_path: Path = DEFAULT_INPUT_PATH, output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    log = configure_logging()
    t0 = time.time()
    tmp_output_path = prepare_temp_output(output_path)
    try:
        shutil.copy2(input_path, tmp_output_path)
    except Exception:
        cleanup_temp_file(tmp_output_path)
        raise
    promote_temp_file(tmp_output_path, output_path)
    log.info("Stage 11 compatibility copy complete in %.1fs", time.time() - t0)
    log.info("Copied %s -> %s", input_path, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 11 compatibility copy")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage11(input_path=args.input, output_path=args.output)
