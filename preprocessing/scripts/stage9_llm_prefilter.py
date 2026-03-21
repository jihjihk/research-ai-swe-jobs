#!/usr/bin/env python3
"""
Stage 9: LLM pre-filtering.

Inputs:
  - preprocessing/intermediate/stage8_final.parquet

Outputs:
  - preprocessing/intermediate/stage9_llm_candidates.parquet
      Unique-description subset that should be sent to the LLM.
  - preprocessing/intermediate/stage9_skip_reasons.parquet
      Full row-level output with llm_candidate / llm_skip_reason added.

Compatibility note:
  The current stage8 artifact uses `lang_detected` instead of `is_english` and
  does not include `description_hash`. This stage derives compatible values
  without modifying stages 1-8.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).parent.parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
CACHE_DIR = PROJECT_ROOT / "preprocessing" / "cache"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

DEFAULT_INPUT_PATH = INTERMEDIATE_DIR / "stage8_final.parquet"
DEFAULT_CANDIDATES_PATH = INTERMEDIATE_DIR / "stage9_llm_candidates.parquet"
DEFAULT_SKIP_REASONS_PATH = INTERMEDIATE_DIR / "stage9_skip_reasons.parquet"

CHUNK_SIZE = 200_000


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "stage9_llm_prefilter.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def normalize_text_for_hash(text) -> str:
    if pd.isna(text):
        return ""
    return str(text)


def compute_description_hash(text) -> str:
    return hashlib.sha256(normalize_text_for_hash(text).encode("utf-8")).hexdigest()


def english_mask(df: pd.DataFrame) -> pd.Series:
    if "is_english" in df.columns:
        return df["is_english"].fillna(False).astype(bool)
    if "lang_detected" in df.columns:
        return df["lang_detected"].fillna("").astype(str).eq("en")
    return pd.Series(True, index=df.index)


def process_chunk(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "description_hash" not in out.columns:
        out["description_hash"] = out["description"].map(compute_description_hash)

    is_english = english_mask(out)
    obvious_non_swe = (
        out["swe_classification_tier"].fillna("").eq("regex")
        & ~out["is_swe"].fillna(False).astype(bool)
        & ~out["is_swe_adjacent"].fillna(False).astype(bool)
    )
    title_seniority = out["seniority_source"].fillna("").astype(str).str.startswith("title_")
    high_confidence_rules = (
        out["swe_classification_tier"].fillna("").eq("regex")
        & title_seniority
        & out["ghost_job_risk"].fillna("").eq("low")
    )

    out["llm_skip_reason"] = "send_to_llm"
    out.loc[~is_english, "llm_skip_reason"] = "non_english"
    out.loc[is_english & obvious_non_swe, "llm_skip_reason"] = "obvious_non_swe"
    out.loc[
        is_english & ~obvious_non_swe & high_confidence_rules,
        "llm_skip_reason",
    ] = "high_confidence_rules"

    out["llm_candidate"] = out["llm_skip_reason"].eq("send_to_llm")
    return out


def write_table_chunk(
    df: pd.DataFrame,
    output_path: Path,
    writer: pq.ParquetWriter | None,
) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema)
    writer.write_table(table)
    del table
    return writer


def run_stage9(
    input_path: Path = DEFAULT_INPUT_PATH,
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    skip_reasons_path: Path = DEFAULT_SKIP_REASONS_PATH,
) -> None:
    log = configure_logging()
    t0 = time.time()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Stage 9: LLM pre-filtering")
    log.info("=" * 70)
    log.info("Input: %s", input_path)
    log.info("Candidates output: %s", candidates_path)
    log.info("Row-level output: %s", skip_reasons_path)

    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    log.info("Input rows: %s", f"{total_rows:,}")

    skip_writer = None
    candidate_writer = None
    seen_hashes: set[str] = set()

    reason_counts = {
        "non_english": 0,
        "obvious_non_swe": 0,
        "high_confidence_rules": 0,
        "send_to_llm": 0,
    }
    candidate_rows = 0
    unique_candidate_rows = 0
    rows_written = 0

    candidate_cols = [
        "job_id",
        "source",
        "source_platform",
        "title",
        "company_name",
        "description",
        "description_hash",
        "llm_candidate",
        "llm_skip_reason",
    ]

    for chunk_num, batch in enumerate(pf.iter_batches(batch_size=CHUNK_SIZE), start=1):
        t_chunk = time.time()
        chunk = pa.Table.from_batches([batch]).to_pandas()
        chunk = process_chunk(chunk)
        n = len(chunk)

        for reason, count in chunk["llm_skip_reason"].value_counts().items():
            reason_counts[reason] = reason_counts.get(reason, 0) + int(count)

        candidate_chunk = chunk.loc[chunk["llm_candidate"], candidate_cols].copy()
        candidate_rows += len(candidate_chunk)

        if not candidate_chunk.empty:
            is_new = ~candidate_chunk["description_hash"].isin(seen_hashes)
            unique_candidate_chunk = candidate_chunk.loc[is_new].copy()
            if not unique_candidate_chunk.empty:
                seen_hashes.update(unique_candidate_chunk["description_hash"].tolist())
                unique_candidate_rows += len(unique_candidate_chunk)
                candidate_writer = write_table_chunk(
                    unique_candidate_chunk,
                    candidates_path,
                    candidate_writer,
                )
                del unique_candidate_chunk

        skip_writer = write_table_chunk(chunk, skip_reasons_path, skip_writer)
        rows_written += n

        elapsed = time.time() - t_chunk
        log.info(
            "Chunk %s done in %.1fs (%s/%s rows, %s unique LLM candidates so far)",
            chunk_num,
            elapsed,
            f"{rows_written:,}",
            f"{total_rows:,}",
            f"{unique_candidate_rows:,}",
        )

        del chunk, candidate_chunk, batch
        gc.collect()

    if skip_writer is not None:
        skip_writer.close()
    if candidate_writer is not None:
        candidate_writer.close()

    elapsed_total = time.time() - t0
    log.info("=" * 70)
    log.info("COMPLETE in %.1fs", elapsed_total)
    log.info("=" * 70)
    for reason in [
        "non_english",
        "obvious_non_swe",
        "high_confidence_rules",
        "send_to_llm",
    ]:
        log.info("  %s: %s", reason, f"{reason_counts.get(reason, 0):,}")
    log.info("  Candidate rows before dedup: %s", f"{candidate_rows:,}")
    log.info("  Unique descriptions to send: %s", f"{unique_candidate_rows:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 9 LLM pre-filtering")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--candidates-output", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--skip-reasons-output", type=Path, default=DEFAULT_SKIP_REASONS_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage9(
        input_path=args.input,
        candidates_path=args.candidates_output,
        skip_reasons_path=args.skip_reasons_output,
    )
