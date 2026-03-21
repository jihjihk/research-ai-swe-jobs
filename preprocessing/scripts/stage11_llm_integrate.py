#!/usr/bin/env python3
"""
Stage 11: Integrate cached LLM responses into unified.parquet.

Input:
  - preprocessing/intermediate/stage8_final.parquet
  - preprocessing/cache/llm_responses.db

Output:
  - data/unified.parquet

Architectural contract:
  - Stage 11 preserves the row cardinality of the input posting table.
  - It joins Stage 10 cached outputs back onto every row by `description_hash`.
  - Stage 10 may deduplicate LLM calls; Stage 11 must not deduplicate postings.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None

from stage10_llm_classify import (
    CLASSIFICATION_PROMPT_VERSION,
    CLASSIFICATION_TASK_NAME,
    EXTRACTION_PROMPT_VERSION,
    EXTRACTION_TASK_NAME,
    fetch_cached_rows,
    join_selected_blocks,
    open_cache,
    segment_description_into_blocks,
)


PROJECT_ROOT = Path(__file__).parent.parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
CACHE_DIR = PROJECT_ROOT / "preprocessing" / "cache"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_INPUT_PATH = INTERMEDIATE_DIR / "stage8_final.parquet"
DEFAULT_CACHE_DB = CACHE_DIR / "llm_responses.db"
DEFAULT_OUTPUT_PATH = DATA_DIR / "unified.parquet"
DEFAULT_VERBATIM_LOG = LOG_DIR / "stage11_verbatim_failures.jsonl"

CHUNK_SIZE = 200_000
SIMILARITY_THRESHOLD = 0.99


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "stage11_llm_integrate.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def compute_description_hash(text) -> str:
    if pd.isna(text):
        text = ""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def normalize_ws(text: str) -> str:
    return " ".join((text or "").split())


def fuzzy_similarity(left: str, right: str) -> float:
    left_norm = normalize_ws(left)
    right_norm = normalize_ws(right)
    if not left_norm and not right_norm:
        return 1.0
    if fuzz is not None:
        return float(fuzz.ratio(left_norm, right_norm)) / 100.0
    return SequenceMatcher(a=left_norm, b=right_norm).ratio()


def validate_extraction_payload(
    original_description: str,
    payload: dict,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    blocks = segment_description_into_blocks(original_description)
    keep_block_ids = payload.get("keep_block_ids", []) or []
    selected_text = payload.get("selected_text", "") or ""
    all_boilerplate = bool(payload.get("all_boilerplate", False))

    normalized_ids = []
    invalid_ids = []
    for value in keep_block_ids:
        if isinstance(value, bool):
            invalid_ids.append(value)
            continue
        try:
            block_id = int(value)
        except (TypeError, ValueError):
            invalid_ids.append(value)
            continue
        if block_id < 1 or block_id > len(blocks):
            invalid_ids.append(block_id)
            continue
        normalized_ids.append(block_id)

    ordered_unique_ids = []
    for block_id in normalized_ids:
        if block_id not in ordered_unique_ids:
            ordered_unique_ids.append(block_id)

    reconstructed_text = join_selected_blocks(blocks, ordered_unique_ids)
    similarity = fuzzy_similarity(selected_text, reconstructed_text)

    reason = "ok"
    passed = True
    if invalid_ids:
        passed = False
        reason = "invalid_block_ids"
    elif all_boilerplate and ordered_unique_ids:
        passed = False
        reason = "all_boilerplate_with_keep_ids"
    elif not ordered_unique_ids and normalize_ws(selected_text):
        passed = False
        reason = "selected_text_without_keep_ids"
    elif similarity < similarity_threshold:
        passed = False
        reason = "selected_text_similarity_below_threshold"

    return {
        "passed": passed,
        "reason": reason,
        "block_count": len(blocks),
        "keep_block_ids": ordered_unique_ids,
        "invalid_block_ids": invalid_ids,
        "selected_text": selected_text,
        "reconstructed_text": reconstructed_text,
        "similarity": similarity,
        "all_boilerplate": all_boilerplate,
    }


def yoe_contradiction(yoe_extracted: float | None, seniority_llm, seniority_imputed) -> bool:
    if yoe_extracted is None or yoe_extracted < 5:
        return False
    if seniority_llm == "entry":
        return True
    if seniority_llm is None and seniority_imputed == "entry":
        return True
    return False


def run_stage11(
    input_path: Path = DEFAULT_INPUT_PATH,
    cache_db: Path = DEFAULT_CACHE_DB,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    verbatim_log_path: Path = DEFAULT_VERBATIM_LOG,
) -> None:
    log = configure_logging()
    t0 = time.time()

    log.info("=" * 70)
    log.info("Stage 11: Integrate LLM outputs")
    log.info("=" * 70)
    log.info("Input: %s", input_path)
    log.info("Cache DB: %s", cache_db)
    log.info("Output: %s", output_path)
    log.info("Row-cardinality rule: Stage 11 preserves input posting rows and reattaches cached LLM outputs by description_hash.")

    conn = open_cache(cache_db)
    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    writer = None
    rows_written = 0
    classification_hits = 0
    classification_misses = 0
    extraction_hits = 0
    extraction_misses = 0
    extraction_passes = 0
    extraction_failures = 0

    for chunk_num, batch in enumerate(pf.iter_batches(batch_size=CHUNK_SIZE), start=1):
        t_chunk = time.time()
        chunk = pa.Table.from_batches([batch]).to_pandas()

        if "description_hash" not in chunk.columns:
            chunk["description_hash"] = chunk["description"].map(compute_description_hash)

        hashes = chunk["description_hash"].astype(str).tolist()
        classification_cache = fetch_cached_rows(
            conn,
            hashes,
            CLASSIFICATION_TASK_NAME,
            CLASSIFICATION_PROMPT_VERSION,
        )
        extraction_cache = fetch_cached_rows(
            conn,
            hashes,
            EXTRACTION_TASK_NAME,
            EXTRACTION_PROMPT_VERSION,
        )

        swe_values = []
        seniority_values = []
        ghost_values = []
        description_core_values = []
        class_model_values = []
        extract_model_values = []
        class_prompt_values = []
        extract_prompt_values = []
        extraction_ids_values = []
        extraction_similarity_values = []
        extraction_validated_values = []

        for row in chunk.itertuples(index=False):
            description_hash = str(row.description_hash)
            classification_row = classification_cache.get(description_hash)
            extraction_row = extraction_cache.get(description_hash)

            if classification_row is None:
                classification_misses += 1
                class_payload = None
            else:
                classification_hits += 1
                class_payload = json.loads(classification_row["response_json"])

            if extraction_row is None:
                extraction_misses += 1
                extraction_validation = None
            else:
                extraction_hits += 1
                extraction_payload = json.loads(extraction_row["response_json"])
                extraction_validation = validate_extraction_payload(
                    "" if row.description is None else str(row.description),
                    extraction_payload,
                )
                if extraction_validation["passed"]:
                    extraction_passes += 1
                else:
                    extraction_failures += 1
                    append_jsonl(
                        verbatim_log_path,
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "job_id": row.job_id,
                            "description_hash": description_hash,
                            "reason": extraction_validation["reason"],
                            "similarity": extraction_validation["similarity"],
                            "keep_block_ids": extraction_validation["keep_block_ids"],
                            "invalid_block_ids": extraction_validation["invalid_block_ids"],
                            "selected_text_preview": extraction_validation["selected_text"][:1200],
                            "reconstructed_text_preview": extraction_validation["reconstructed_text"][:1200],
                            "description_preview": ("" if row.description is None else str(row.description))[:1200],
                        },
                    )

            swe_values.append(None if class_payload is None else class_payload["swe_classification"])
            seniority_values.append(None if class_payload is None else class_payload["seniority"])
            ghost_values.append(None if class_payload is None else class_payload["ghost_assessment"])
            description_core_values.append(
                None
                if extraction_validation is None or not extraction_validation["passed"]
                else extraction_validation["reconstructed_text"]
            )
            class_model_values.append(None if classification_row is None else classification_row["model"])
            extract_model_values.append(None if extraction_row is None else extraction_row["model"])
            class_prompt_values.append(
                None if classification_row is None else classification_row["prompt_version"]
            )
            extract_prompt_values.append(
                None if extraction_row is None else extraction_row["prompt_version"]
            )
            extraction_ids_values.append(
                None
                if extraction_validation is None
                else json.dumps(extraction_validation["keep_block_ids"], ensure_ascii=False)
            )
            extraction_similarity_values.append(
                None if extraction_validation is None else extraction_validation["similarity"]
            )
            extraction_validated_values.append(
                None if extraction_validation is None else extraction_validation["passed"]
            )

        chunk["swe_classification_llm"] = swe_values
        chunk["seniority_llm"] = seniority_values
        chunk["ghost_assessment_llm"] = ghost_values
        chunk["description_core_llm"] = description_core_values
        chunk["llm_model_classification"] = class_model_values
        chunk["llm_model_extraction"] = extract_model_values
        chunk["llm_prompt_version_classification"] = class_prompt_values
        chunk["llm_prompt_version_extraction"] = extract_prompt_values
        chunk["llm_extraction_block_ids"] = extraction_ids_values
        chunk["llm_extraction_similarity"] = pd.Series(extraction_similarity_values, dtype="float64")
        chunk["llm_extraction_validated"] = pd.Series(extraction_validated_values, dtype="boolean")

        out_table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, out_table.schema)
        writer.write_table(out_table)
        rows_written += len(chunk)

        elapsed = time.time() - t_chunk
        log.info(
            "Chunk %s done in %.1fs (%s/%s rows)",
            chunk_num,
            elapsed,
            f"{rows_written:,}",
            f"{total_rows:,}",
        )

        del chunk, out_table, batch
        gc.collect()

    if writer is not None:
        writer.close()
    conn.close()

    elapsed_total = time.time() - t0
    extraction_total = extraction_passes + extraction_failures
    extraction_pass_rate = extraction_passes / extraction_total if extraction_total else 1.0

    log.info("=" * 70)
    log.info("Stage 11 complete in %.1fs", elapsed_total)
    log.info("=" * 70)
    log.info("  Rows written: %s", f"{rows_written:,}")
    log.info("  Row-cardinality preserved from input: %s", f"{total_rows:,}")
    log.info("  Classification hits / misses: %s / %s", f"{classification_hits:,}", f"{classification_misses:,}")
    log.info("  Extraction hits / misses: %s / %s", f"{extraction_hits:,}", f"{extraction_misses:,}")
    log.info("  Extraction passes: %s", f"{extraction_passes:,}")
    log.info("  Extraction failures: %s", f"{extraction_failures:,}")
    log.info("  Extraction pass rate: %.2f%%", extraction_pass_rate * 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 11 LLM integration")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--cache-db", type=Path, default=DEFAULT_CACHE_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--verbatim-log", type=Path, default=DEFAULT_VERBATIM_LOG)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage11(
        input_path=args.input,
        cache_db=args.cache_db,
        output_path=args.output,
        verbatim_log_path=args.verbatim_log,
    )
