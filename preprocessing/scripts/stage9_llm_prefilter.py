#!/usr/bin/env python3
"""
Stage 9: LLM routing / pre-filtering.

Inputs:
  - preprocessing/intermediate/stage8_final.parquet

Outputs:
  - preprocessing/intermediate/stage9_llm_candidates.parquet
      Unique-description queue with task flags OR-ed by `description_hash`.
  - preprocessing/intermediate/stage9_skip_reasons.parquet
      Full row-level output with routing flags and reasons added.

Compatibility note:
  Stage 8 now writes `is_english` and `description_hash` directly. The fallback
  logic here remains only to support older artifacts that may still carry
  `lang_detected` or omit `description_hash`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import time
from pathlib import Path

import duckdb
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
LINKEDIN_PLATFORM = "linkedin"
CLASSIFICATION_STRONG_TIERS = {"regex", "embedding_high", "embedding_llm"}
DEFAULT_INCLUDE_CONTROL_EXTRACTION = False


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


def has_raw_description(df: pd.DataFrame) -> pd.Series:
    return ~df["description"].isna() & df["description"].astype(str).str.strip().ne("")


def process_chunk(
    df: pd.DataFrame,
    include_control_extraction: bool = DEFAULT_INCLUDE_CONTROL_EXTRACTION,
) -> pd.DataFrame:
    out = df.copy()

    if "description_hash" not in out.columns:
        out["description_hash"] = out["description"].map(compute_description_hash)

    is_english = english_mask(out)
    is_linkedin = out["source_platform"].fillna("").astype(str).str.lower().eq(LINKEDIN_PLATFORM)
    has_description = has_raw_description(out)
    is_swe = out["is_swe"].fillna(False).astype(bool)
    is_swe_adjacent = out["is_swe_adjacent"].fillna(False).astype(bool)
    is_control = out["is_control"].fillna(False).astype(bool)
    in_default_universe = is_linkedin & is_english & has_description

    title_seniority = out["seniority_source"].fillna("").astype(str).str.startswith("title_")
    strong_occupation_signal = out["swe_classification_tier"].fillna("").isin(CLASSIFICATION_STRONG_TIERS)
    high_confidence_tech = (
        (is_swe | is_swe_adjacent)
        & strong_occupation_signal
        & title_seniority
        & out["ghost_job_risk"].fillna("").eq("low")
    )

    needs_extraction = in_default_universe & (is_swe | is_swe_adjacent)
    if include_control_extraction:
        needs_extraction = needs_extraction | (in_default_universe & is_control)
    needs_classification = in_default_universe & (is_swe | is_swe_adjacent) & ~high_confidence_tech

    out["needs_llm_classification"] = needs_classification
    out["needs_llm_extraction"] = needs_extraction
    out["llm_candidate"] = needs_classification | needs_extraction

    out["llm_route_group"] = "not_routed"
    out.loc[in_default_universe & is_control, "llm_route_group"] = "control_not_routed"
    if include_control_extraction:
        out.loc[in_default_universe & is_control, "llm_route_group"] = "control_extraction_only"
    out.loc[
        in_default_universe & (is_swe | is_swe_adjacent) & ~needs_classification,
        "llm_route_group",
    ] = "technical_extraction_only"
    out.loc[
        in_default_universe & (is_swe | is_swe_adjacent) & needs_classification,
        "llm_route_group",
    ] = "technical_classification_and_extraction"

    out["llm_classification_reason"] = "not_routed"
    out.loc[needs_classification, "llm_classification_reason"] = "routed"
    out.loc[in_default_universe & is_control, "llm_classification_reason"] = "controls_not_routed"
    out.loc[
        in_default_universe & (is_swe | is_swe_adjacent) & ~needs_classification,
        "llm_classification_reason",
    ] = "high_confidence_technical_rules"

    out["llm_extraction_reason"] = "not_routed"
    out.loc[needs_extraction, "llm_extraction_reason"] = "routed"
    out.loc[in_default_universe & is_control & ~needs_extraction, "llm_extraction_reason"] = (
        "controls_disabled_by_default"
    )

    out["llm_skip_reason"] = "outside_default_scope"
    out.loc[~is_linkedin, "llm_skip_reason"] = "non_linkedin_platform"
    out.loc[is_linkedin & ~is_english, "llm_skip_reason"] = "non_english"
    out.loc[is_linkedin & is_english & ~has_description, "llm_skip_reason"] = "missing_raw_description"
    out.loc[in_default_universe & is_control & ~needs_extraction, "llm_skip_reason"] = "controls_disabled_by_default"
    out.loc[in_default_universe & ~(is_swe | is_swe_adjacent | is_control), "llm_skip_reason"] = (
        "outside_default_classification_scope"
    )
    out.loc[needs_extraction | needs_classification, "llm_skip_reason"] = "routed"
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


def build_candidate_queue(skip_reasons_path: Path, candidates_path: Path, log: logging.Logger) -> dict[str, int]:
    if candidates_path.exists():
        candidates_path.unlink()

    con = duckdb.connect()
    source = str(skip_reasons_path)
    dest = str(candidates_path)

    con.execute(
        f"""
        COPY (
          WITH candidate_rows AS (
            SELECT
              description_hash,
              job_id,
              source,
              source_platform,
              title,
              company_name,
              description,
              CAST(needs_llm_classification AS INTEGER) AS needs_llm_classification,
              CAST(needs_llm_extraction AS INTEGER) AS needs_llm_extraction
            FROM read_parquet('{source}')
            WHERE coalesce(llm_candidate, false)
          )
          SELECT
            description_hash,
            any_value(job_id) AS job_id,
            any_value(source) AS source,
            any_value(source_platform) AS source_platform,
            any_value(title) AS title,
            any_value(company_name) AS company_name,
            any_value(description) AS description,
            max(needs_llm_classification) = 1 AS needs_llm_classification,
            max(needs_llm_extraction) = 1 AS needs_llm_extraction,
            CASE
              WHEN max(needs_llm_classification) = 1 AND max(needs_llm_extraction) = 1
                THEN 'classification_and_extraction'
              WHEN max(needs_llm_extraction) = 1
                THEN 'extraction_only'
              WHEN max(needs_llm_classification) = 1
                THEN 'classification_only'
              ELSE 'not_routed'
            END AS llm_route_group,
            count(*) AS source_row_count
          FROM candidate_rows
          GROUP BY description_hash
        )
        TO '{dest}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    stats = con.execute(
        f"""
        SELECT
          count(*) AS candidate_hashes,
          sum(CASE WHEN needs_llm_classification THEN 1 ELSE 0 END) AS classification_hashes,
          sum(CASE WHEN needs_llm_extraction THEN 1 ELSE 0 END) AS extraction_hashes
        FROM read_parquet('{dest}')
        """
    ).fetchone()
    con.close()

    result = {
        "candidate_hashes": int(stats[0] or 0),
        "classification_hashes": int(stats[1] or 0),
        "extraction_hashes": int(stats[2] or 0),
    }
    log.info("Candidate queue written: %s", candidates_path)
    return result


def run_stage9(
    input_path: Path = DEFAULT_INPUT_PATH,
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    skip_reasons_path: Path = DEFAULT_SKIP_REASONS_PATH,
    include_control_extraction: bool = DEFAULT_INCLUDE_CONTROL_EXTRACTION,
) -> None:
    log = configure_logging()
    t0 = time.time()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Stage 9: LLM routing / pre-filtering")
    log.info("=" * 70)
    log.info("Input: %s", input_path)
    log.info("Candidates output: %s", candidates_path)
    log.info("Row-level output: %s", skip_reasons_path)
    log.info("Include control extraction: %s", include_control_extraction)

    if candidates_path.exists():
        candidates_path.unlink()
    if skip_reasons_path.exists():
        skip_reasons_path.unlink()

    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    log.info("Input rows: %s", f"{total_rows:,}")

    skip_writer = None
    route_counts: dict[str, int] = {}
    skip_reason_counts: dict[str, int] = {}
    classification_rows = 0
    extraction_rows = 0
    rows_written = 0

    for chunk_num, batch in enumerate(pf.iter_batches(batch_size=CHUNK_SIZE), start=1):
        t_chunk = time.time()
        chunk = pa.Table.from_batches([batch]).to_pandas()
        chunk = process_chunk(chunk, include_control_extraction=include_control_extraction)

        for route_group, count in chunk["llm_route_group"].value_counts().items():
            route_counts[route_group] = route_counts.get(route_group, 0) + int(count)
        for reason, count in chunk["llm_skip_reason"].value_counts().items():
            skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + int(count)

        classification_rows += int(chunk["needs_llm_classification"].sum())
        extraction_rows += int(chunk["needs_llm_extraction"].sum())

        skip_writer = write_table_chunk(chunk, skip_reasons_path, skip_writer)
        rows_written += len(chunk)

        elapsed = time.time() - t_chunk
        log.info(
            "Chunk %s done in %.1fs (%s/%s rows, class rows=%s, extraction rows=%s)",
            chunk_num,
            elapsed,
            f"{rows_written:,}",
            f"{total_rows:,}",
            f"{classification_rows:,}",
            f"{extraction_rows:,}",
        )

        del chunk, batch
        gc.collect()

    if skip_writer is not None:
        skip_writer.close()

    candidate_counts = build_candidate_queue(skip_reasons_path, candidates_path, log)

    elapsed_total = time.time() - t0
    log.info("=" * 70)
    log.info("COMPLETE in %.1fs", elapsed_total)
    log.info("=" * 70)
    for route_group in sorted(route_counts):
        log.info("  route[%s]: %s", route_group, f"{route_counts[route_group]:,}")
    for reason in sorted(skip_reason_counts):
        log.info("  skip_reason[%s]: %s", reason, f"{skip_reason_counts[reason]:,}")
    log.info("  Rows requesting classification: %s", f"{classification_rows:,}")
    log.info("  Rows requesting extraction: %s", f"{extraction_rows:,}")
    log.info("  Unique hashes routed: %s", f"{candidate_counts['candidate_hashes']:,}")
    log.info("  Unique hashes needing classification: %s", f"{candidate_counts['classification_hashes']:,}")
    log.info("  Unique hashes needing extraction: %s", f"{candidate_counts['extraction_hashes']:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 9 LLM routing / pre-filtering")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--candidates-output", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--skip-reasons-output", type=Path, default=DEFAULT_SKIP_REASONS_PATH)
    parser.add_argument(
        "--include-control-extraction",
        action="store_true",
        help="Route control rows for LLM extraction as a sensitivity run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage9(
        input_path=args.input,
        candidates_path=args.candidates_output,
        skip_reasons_path=args.skip_reasons_output,
        include_control_extraction=args.include_control_extraction,
    )
