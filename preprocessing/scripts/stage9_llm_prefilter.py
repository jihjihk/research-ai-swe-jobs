#!/usr/bin/env python3
"""
Stage 9: control-cohort selection, extraction routing, extraction execution,
and posting-level cleaned-text integration.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from io_utils import cleanup_temp_file, prepare_temp_output, promote_temp_file
from llm_shared import (
    DEFAULT_BUDGET_SPLIT,
    DEFAULT_ENGINE_TIMEZONE,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CODEX_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_QUOTA_WAIT_HOURS,
    EXTRACTION_PROMPT_VERSION,
    EXTRACTION_STATUS_CANNOT_COMPLETE,
    EXTRACTION_TASK_NAME,
    LLMEngineRuntime,
    MAX_EXTRACTION_UNIT_COUNT,
    SINGLE_UNIT_WARNING_CHARS,
    SUPPORTED_PROVIDERS,
    build_engine_configs,
    build_progress_checkpoints,
    chunked,
    compute_description_hash,
    compute_extraction_input_hash,
    configure_remote_execution,
    execute_task_with_runtime,
    format_engine_labels,
    log_budget_plan,
    log_sampled_llm_response,
    open_cache,
    parse_budget_split,
    parse_engine_tiers,
    parse_engine_list,
    render_extraction_prompt,
    segment_description_into_units,
    select_rows_with_budget,
    store_cached_row,
    try_provider,
    validate_extraction_payload,
    validate_extraction_selection,
    fetch_cached_row,
    fetch_cached_rows,
)


PROJECT_ROOT = Path(__file__).parent.parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
CACHE_DIR = PROJECT_ROOT / "preprocessing" / "cache"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

DEFAULT_INPUT_PATH = INTERMEDIATE_DIR / "stage8_final.parquet"
DEFAULT_CANDIDATES_PATH = INTERMEDIATE_DIR / "stage9_llm_extraction_candidates.parquet"
DEFAULT_RESULTS_PATH = INTERMEDIATE_DIR / "stage9_llm_extraction_results.parquet"
DEFAULT_CLEANED_PATH = INTERMEDIATE_DIR / "stage9_llm_cleaned.parquet"
DEFAULT_CONTROL_COHORT_PATH = INTERMEDIATE_DIR / "stage9_control_cohort.parquet"
DEFAULT_CACHE_DB = CACHE_DIR / "llm_responses.db"
DEFAULT_ERROR_LOG = LOG_DIR / "llm_errors.jsonl"

CHUNK_SIZE = 50_000
MIN_DESCRIPTION_WORDS = 15


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "stage9_llm_prefilter.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def should_cache_extraction_result(result: dict) -> bool:
    return str(result.get("model") or "") != "synthetic-provider-failed"


def english_mask(df: pd.DataFrame) -> pd.Series:
    if "is_english" in df.columns:
        return df["is_english"].astype("boolean").fillna(False).astype(bool)
    if "lang_detected" in df.columns:
        return df["lang_detected"].fillna("").astype(str).eq("en")
    return pd.Series(True, index=df.index)


def has_raw_description(df: pd.DataFrame) -> pd.Series:
    return ~df["description"].isna() & df["description"].astype(str).str.strip().ne("")


def build_control_bucket(df: pd.DataFrame) -> pd.Series:
    scraped = df["source"].fillna("").astype(str).eq("scraped")
    scrape_dates = pd.to_datetime(
        df.get("scrape_date", pd.Series(index=df.index, dtype="object")),
        errors="coerce",
    )
    iso = scrape_dates.dt.isocalendar()
    buckets = (
        df["source"].fillna("unknown").astype(str)
        + "|"
        + df.get("period", pd.Series(index=df.index, dtype="object")).fillna("unknown").astype(str)
    )
    scraped_buckets = (
        "scraped|"
        + iso["year"].fillna(0).astype(int).astype(str)
        + "-"
        + iso["week"].fillna(0).astype(int).astype(str).str.zfill(2)
    )
    buckets.loc[scraped] = scraped_buckets.loc[scraped]
    return buckets


def stable_control_score(control_bucket: str, extraction_input_hash: str) -> str:
    seed = f"control-cohort-v1|{control_bucket}|{extraction_input_hash}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def annotate_stage9_chunk(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "description_hash" not in out.columns:
        out["description_hash"] = out["description"].map(compute_description_hash)
    out["is_english"] = english_mask(out)
    out["has_raw_description"] = has_raw_description(out)
    out["is_linkedin"] = out["source_platform"].fillna("").astype(str).str.lower().eq("linkedin")
    raw_description = out["description"].fillna("").astype(str)
    out["raw_description_word_count"] = raw_description.str.split().str.len().astype("Int64")
    out["short_description_skip"] = out["raw_description_word_count"].fillna(0).lt(MIN_DESCRIPTION_WORDS)
    out["control_bucket"] = build_control_bucket(out)
    eligible_text = out["is_linkedin"] & out["is_english"] & out["has_raw_description"]
    out["eligible_swe_extraction"] = eligible_text & out["is_swe"].fillna(False).astype(bool) & ~out["short_description_skip"]
    out["eligible_control_extraction"] = (
        eligible_text & out["is_control"].fillna(False).astype(bool) & ~out["short_description_skip"]
    )
    out["eligible_control_unit"] = out["eligible_control_extraction"]
    out["eligible_for_extraction"] = eligible_text & (
        out["is_swe"].fillna(False).astype(bool) | out["is_swe_adjacent"].fillna(False).astype(bool)
    ) & ~out["short_description_skip"]
    out["llm_text_skip_reason"] = None
    out.loc[out["short_description_skip"], "llm_text_skip_reason"] = "short_description_under_15_words"
    out["description_core_llm"] = None
    out.loc[out["short_description_skip"], "description_core_llm"] = ""
    out["extraction_input_hash"] = [
        compute_extraction_input_hash(title, company, description) if eligible_text else None
        for title, company, description, eligible_text in zip(
            out.get("title", pd.Series(index=out.index, dtype="object")),
            out.get("company_name", pd.Series(index=out.index, dtype="object")),
            out.get("description", pd.Series(index=out.index, dtype="object")),
            out["has_raw_description"],
        )
    ]
    return out


def annotate_chunk(df: pd.DataFrame) -> pd.DataFrame:
    return annotate_stage9_chunk(df)


def _select_control_cohort(annotated: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    eligible_swe = annotated.loc[annotated["eligible_swe_extraction"]]
    eligible_controls = annotated.loc[annotated["eligible_control_extraction"]]

    swe_counts = (
        eligible_swe[["control_bucket", "extraction_input_hash"]]
        .drop_duplicates()
        .groupby("control_bucket")
        .size()
        .to_dict()
    )

    control_records = (
        eligible_controls[
            [
                "control_bucket",
                "extraction_input_hash",
                "job_id",
                "source",
                "source_platform",
                "title",
                "company_name",
                "description_hash",
            ]
        ]
        .drop_duplicates(subset=["control_bucket", "extraction_input_hash"])
        .copy()
    )
    if control_records.empty:
        return control_records.assign(selected_for_control_cohort=False), set()

    control_records["stable_score"] = [
        stable_control_score(bucket, input_hash)
        for bucket, input_hash in zip(control_records["control_bucket"], control_records["extraction_input_hash"])
    ]

    controls_by_bucket = {
        bucket: bucket_df.sort_values("stable_score").copy()
        for bucket, bucket_df in control_records.groupby("control_bucket", sort=True)
    }
    targets = {
        bucket: min(int(swe_counts.get(bucket, 0)), len(bucket_df))
        for bucket, bucket_df in controls_by_bucket.items()
    }
    shortfall = sum(
        max(int(swe_counts.get(bucket, 0)) - len(bucket_df), 0) for bucket, bucket_df in controls_by_bucket.items()
    )

    while shortfall > 0:
        spare_buckets = [
            bucket for bucket, bucket_df in controls_by_bucket.items() if len(bucket_df) > targets[bucket]
        ]
        if not spare_buckets:
            break
        for bucket in sorted(spare_buckets, key=lambda item: (-int(swe_counts.get(item, 0)), item)):
            if shortfall <= 0 or len(controls_by_bucket[bucket]) <= targets[bucket]:
                continue
            targets[bucket] += 1
            shortfall -= 1

    selected_hashes: set[str] = set()
    frames = []
    for bucket, bucket_df in controls_by_bucket.items():
        target_n = targets.get(bucket, 0)
        bucket_df = bucket_df.copy()
        bucket_df["selected_for_control_cohort"] = False
        if target_n > 0:
            bucket_df.loc[bucket_df.index[:target_n], "selected_for_control_cohort"] = True
            selected_hashes.update(bucket_df.loc[bucket_df.index[:target_n], "extraction_input_hash"].astype(str))
        frames.append(bucket_df)

    return pd.concat(frames, ignore_index=True), selected_hashes


def select_control_cohort(annotated: pd.DataFrame) -> pd.DataFrame:
    control_cohort_df, _ = _select_control_cohort(annotated)
    return control_cohort_df


def build_extraction_candidates(annotated: pd.DataFrame, control_cohort: pd.DataFrame) -> pd.DataFrame:
    selected_control_hashes = set(
        control_cohort.loc[control_cohort["selected_for_control_cohort"], "extraction_input_hash"].astype(str)
    )
    routed = annotated.copy()
    routed["selected_for_control_cohort"] = routed["extraction_input_hash"].astype(str).isin(selected_control_hashes)
    routed = routed.loc[
        (~routed["short_description_skip"])
        & (
            routed["eligible_for_extraction"]
            | routed["selected_for_control_cohort"]
        )
    ].copy()
    if routed.empty:
        return pd.DataFrame()
    routed["llm_route_group"] = "technical_extraction"
    routed.loc[routed["selected_for_control_cohort"] & ~routed["eligible_for_extraction"], "llm_route_group"] = (
        "control_extraction"
    )
    agg_spec = dict(
        job_id=("job_id", "first"),
        source=("source", "first"),
        source_platform=("source_platform", "first"),
        title=("title", "first"),
        company_name=("company_name", "first"),
        description_hash=("description_hash", "first"),
        llm_route_group=("llm_route_group", "first"),
        selected_for_control_cohort=("selected_for_control_cohort", "max"),
        source_row_count=("job_id", "count"),
    )
    if "description" in routed.columns:
        agg_spec["description"] = ("description", "first")
    return routed.groupby("extraction_input_hash", as_index=False).agg(**agg_spec)


def process_chunk(df: pd.DataFrame, selected_control_hashes: set[str] | None = None) -> pd.DataFrame:
    out = annotate_stage9_chunk(df)
    selected_control_hashes = selected_control_hashes or set()
    out["selected_for_control_cohort"] = out["extraction_input_hash"].astype(str).isin(selected_control_hashes)
    extraction_scope = (
        out["is_swe"].fillna(False).astype(bool)
        | out["is_swe_adjacent"].fillna(False).astype(bool)
        | out["selected_for_control_cohort"]
    )
    out["needs_llm_extraction"] = (
        out["is_linkedin"] & out["is_english"] & out["has_raw_description"] & extraction_scope & ~out["short_description_skip"]
    )
    out["llm_extraction_reason"] = "not_routed"
    out.loc[out["short_description_skip"] & extraction_scope, "llm_extraction_reason"] = "short_description"
    out.loc[out["needs_llm_extraction"], "llm_extraction_reason"] = "routed"
    return out


def _promote_null_fields(schema: pa.Schema) -> pa.Schema:
    """Replace null-typed fields with string so later batches can write values."""
    fields = []
    for field in schema:
        if pa.types.is_null(field.type):
            fields.append(pa.field(field.name, pa.string()))
        else:
            fields.append(field)
    return pa.schema(fields, metadata=schema.metadata)


def write_parquet_rows(rows: list[dict], output_path: Path) -> None:
    if not rows:
        pd.DataFrame(rows).to_parquet(output_path, index=False)
        return
    writer = None
    schema = None
    try:
        for batch in chunked(rows, CHUNK_SIZE):
            table = pa.Table.from_pylist(batch)
            if writer is None:
                schema = _promote_null_fields(table.schema)
                writer = pq.ParquetWriter(output_path, schema)
            table = table.cast(schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def call_task_with_engine(
    *,
    prompt: str,
    input_hash: str,
    error_log_path: Path,
    log: logging.Logger,
    runtime: LLMEngineRuntime | None = None,
    codex_model: str = DEFAULT_CODEX_MODEL,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    timeout_seconds: int = 180,
    max_retries: int = 3,
    enabled_engines: tuple[str, ...] = SUPPORTED_PROVIDERS,
    claude_model: str = DEFAULT_CLAUDE_MODEL,
    quota_wait_hours: float = DEFAULT_QUOTA_WAIT_HOURS,
    engine_tiers: dict[str, str] | None = None,
    engine_timezone: str = DEFAULT_ENGINE_TIMEZONE,
) -> dict | None:
    runtime = runtime or LLMEngineRuntime(
        build_engine_configs(
            enabled_engines,
            codex_model=codex_model,
            claude_model=claude_model,
            openai_model=openai_model,
            engine_tiers=engine_tiers,
        ),
        slot_timezone=engine_timezone,
    )
    return execute_task_with_runtime(
        runtime=runtime,
        task_name=EXTRACTION_TASK_NAME,
        prompt=prompt,
        input_hash=input_hash,
        error_log_path=error_log_path,
        log=log,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        payload_validator=validate_extraction_payload,
        quota_wait_hours=quota_wait_hours,
    )


def process_candidate_row(
    row: dict,
    *,
    runtime: LLMEngineRuntime,
    timeout_seconds: int,
    max_retries: int,
    error_log_path: Path,
    log: logging.Logger,
    quota_wait_hours: float,
) -> dict:
    prompt, units = render_extraction_prompt(row["title"], row["company_name"], row["description"])
    if not units:
        payload = {
            "task_status": EXTRACTION_STATUS_CANNOT_COMPLETE,
            "boilerplate_unit_ids": [],
            "uncertain_unit_ids": [],
            "reason": "empty_description",
        }
        return {
            "model": "synthetic-empty-description",
            "response_json": json.dumps(payload, ensure_ascii=False),
            "tokens_used": None,
            "prompt_version": EXTRACTION_PROMPT_VERSION,
        }
    if len(units) == 1 and len(units[0]["text"]) >= SINGLE_UNIT_WARNING_CHARS:
        payload = {
            "task_status": EXTRACTION_STATUS_CANNOT_COMPLETE,
            "boilerplate_unit_ids": [],
            "uncertain_unit_ids": [],
            "reason": "single_unit_description",
        }
        return {
            "model": "synthetic-single-unit-fallback",
            "response_json": json.dumps(payload, ensure_ascii=False),
            "tokens_used": None,
            "prompt_version": EXTRACTION_PROMPT_VERSION,
        }
    if len(units) > MAX_EXTRACTION_UNIT_COUNT:
        payload = {
            "task_status": EXTRACTION_STATUS_CANNOT_COMPLETE,
            "boilerplate_unit_ids": [],
            "uncertain_unit_ids": [],
            "reason": "too_many_units",
        }
        return {
            "model": "synthetic-too-many-units-fallback",
            "response_json": json.dumps(payload, ensure_ascii=False),
            "tokens_used": None,
            "prompt_version": EXTRACTION_PROMPT_VERSION,
        }
    result = call_task_with_engine(
        runtime=runtime,
        prompt=prompt,
        input_hash=row["extraction_input_hash"],
        error_log_path=error_log_path,
        log=log,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        quota_wait_hours=quota_wait_hours,
    )
    if result is None:
        payload = {
            "task_status": EXTRACTION_STATUS_CANNOT_COMPLETE,
            "boilerplate_unit_ids": [],
            "uncertain_unit_ids": [],
            "reason": "provider_failed",
        }
        return {
            "model": "synthetic-provider-failed",
            "response_json": json.dumps(payload, ensure_ascii=False),
            "tokens_used": None,
            "prompt_version": EXTRACTION_PROMPT_VERSION,
        }
    result["prompt_version"] = EXTRACTION_PROMPT_VERSION
    return result


def build_candidate_records(input_path: Path, selected_control_hashes: set[str]) -> list[dict]:
    pf = pq.ParquetFile(input_path)
    lookup: dict[str, dict] = {}
    for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
        chunk = pa.Table.from_batches([batch]).to_pandas()
        prepared = process_chunk(chunk, selected_control_hashes=selected_control_hashes)
        routed = prepared.loc[prepared["needs_llm_extraction"]].copy()
        if routed.empty:
            continue
        # Canonicalize scrape_date to YYYY-MM-DD strings for daily bucketing.
        if "scrape_date" in routed.columns:
            routed["scrape_date"] = (
                pd.to_datetime(routed["scrape_date"], errors="coerce")
                .dt.strftime("%Y-%m-%d")
                .fillna("unknown")
            )
        else:
            routed["scrape_date"] = "unknown"
        grouped = (
            routed[
                [
                    "extraction_input_hash",
                    "job_id",
                    "source",
                    "source_platform",
                    "title",
                    "company_name",
                    "description",
                    "description_hash",
                    "scrape_date",
                    "is_swe",
                    "is_swe_adjacent",
                    "selected_for_control_cohort",
                ]
            ]
            .drop_duplicates(subset=["extraction_input_hash"])
            .to_dict("records")
        )
        for row in grouped:
            lookup.setdefault(str(row["extraction_input_hash"]), row)
    return list(lookup.values())


def resolve_description_core_llm(row, cached_rows: dict[str, dict]) -> str | None:
    if row.short_description_skip and (
        row.is_swe or row.is_swe_adjacent or row.selected_for_control_cohort
    ):
        return ""
    if not row.needs_llm_extraction or row.extraction_input_hash is None:
        return None

    cached = cached_rows.get(str(row.extraction_input_hash))
    if cached is None:
        return None

    payload = json.loads(cached["response_json"])
    validation = validate_extraction_selection(
        "" if row.description is None else str(row.description),
        payload,
    )
    if validation["passed"]:
        return validation["reconstructed_text"]
    if validation["reason"] == "all_units_dropped":
        return ""
    return None


def summarize_stage9_routing(
    prepared: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    selected_control_count: int,
    cached_task_count: int,
    fresh_task_count: int,
) -> dict[str, int]:
    reason_counts = Counter(prepared["llm_extraction_reason"].fillna("unknown").astype(str))
    route_group_counts = (
        Counter(candidate_summary["llm_route_group"].fillna("unknown").astype(str))
        if not candidate_summary.empty
        else Counter()
    )
    routed_row_count = int(candidate_summary["source_row_count"].sum()) if not candidate_summary.empty else 0
    unique_task_count = int(len(candidate_summary))
    return {
        "total_rows": int(len(prepared)),
        "linkedin_rows": int(prepared["is_linkedin"].fillna(False).sum()),
        "english_rows": int(prepared["is_english"].fillna(False).sum()),
        "technical_scope_rows": int(
            (prepared["is_swe"].fillna(False).astype(bool) | prepared["is_swe_adjacent"].fillna(False).astype(bool)).sum()
        ),
        "control_pool_rows": int(prepared["eligible_control_extraction"].fillna(False).sum()),
        "selected_control_rows": int(selected_control_count),
        "short_skip_rows": int(reason_counts.get("short_description", 0)),
        "routed_rows": int(reason_counts.get("routed", 0)),
        "not_routed_rows": int(reason_counts.get("not_routed", 0)),
        "unique_tasks": unique_task_count,
        "technical_tasks": int(route_group_counts.get("technical_extraction", 0)),
        "control_tasks": int(route_group_counts.get("control_extraction", 0)),
        "duplicate_rows_collapsed": int(routed_row_count - unique_task_count),
        "cached_tasks": int(cached_task_count),
        "fresh_tasks": int(fresh_task_count),
    }


def log_stage9_plan(log: logging.Logger, summary: dict[str, int], *, max_workers: int, runtime: LLMEngineRuntime) -> None:
    log.info(
        "Execution plan | workers=%s | engines=%s",
        max_workers,
        format_engine_labels(runtime.engines),
    )
    log.info(
        "Routing summary | rows=%s | linkedin=%s | english=%s | technical_scope=%s | control_pool=%s | selected_controls=%s",
        f"{summary['total_rows']:,}",
        f"{summary['linkedin_rows']:,}",
        f"{summary['english_rows']:,}",
        f"{summary['technical_scope_rows']:,}",
        f"{summary['control_pool_rows']:,}",
        f"{summary['selected_control_rows']:,}",
    )
    log.info(
        "Extraction volume | routed_rows=%s | short_skips=%s | not_routed=%s | unique_tasks=%s | technical_tasks=%s | control_tasks=%s | deduped_rows=%s",
        f"{summary['routed_rows']:,}",
        f"{summary['short_skip_rows']:,}",
        f"{summary['not_routed_rows']:,}",
        f"{summary['unique_tasks']:,}",
        f"{summary['technical_tasks']:,}",
        f"{summary['control_tasks']:,}",
        f"{summary['duplicate_rows_collapsed']:,}",
    )
    log.info(
        "Cache plan | cached=%s | fresh=%s",
        f"{summary['cached_tasks']:,}",
        f"{summary['fresh_tasks']:,}",
    )


def log_stage9_sample_response(log: logging.Logger, *, completed: int, row: dict, result: dict) -> None:
    log_sampled_llm_response(
        log,
        stage_label="Stage 9",
        completed=completed,
        input_hash=str(row.get("extraction_input_hash") or ""),
        job_id=str(row.get("job_id") or ""),
        model=str(result.get("model") or ""),
        response_json=result.get("response_json"),
        extra_fields={
            "route_group": row.get("llm_route_group"),
            "source_row_count": row.get("source_row_count"),
        },
    )


def run_stage9(
    *,
    llm_budget: int,
    llm_budget_split: dict[str, float],
    input_path: Path = DEFAULT_INPUT_PATH,
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    results_path: Path = DEFAULT_RESULTS_PATH,
    cleaned_path: Path = DEFAULT_CLEANED_PATH,
    control_cohort_path: Path = DEFAULT_CONTROL_COHORT_PATH,
    cache_db: Path = DEFAULT_CACHE_DB,
    error_log_path: Path = DEFAULT_ERROR_LOG,
    codex_model: str = DEFAULT_CODEX_MODEL,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    timeout_seconds: int = 180,
    max_retries: int = 3,
    max_workers: int = 30,
    enabled_engines: tuple[str, ...] = SUPPORTED_PROVIDERS,
    claude_model: str = DEFAULT_CLAUDE_MODEL,
    quota_wait_hours: float = DEFAULT_QUOTA_WAIT_HOURS,
    engine_tiers: dict[str, str] | None = None,
    engine_timezone: str = DEFAULT_ENGINE_TIMEZONE,
) -> None:
    if llm_budget < 0:
        raise ValueError(f"llm_budget must be >= 0, got {llm_budget}")
    log = configure_logging()
    t0 = time.time()
    runtime = LLMEngineRuntime(
        build_engine_configs(
            enabled_engines,
            codex_model=codex_model,
            claude_model=claude_model,
            openai_model=openai_model,
            engine_tiers=engine_tiers,
        ),
        slot_timezone=engine_timezone,
    )

    # ---- Pass 1: stream chunks, keep only lightweight columns for control
    # cohort selection and summary logging.  Text columns (description, etc.)
    # are dropped after annotation to avoid holding the full dataset in RAM.
    _LIGHTWEIGHT_COLS = [
        "job_id", "source", "source_platform", "title", "company_name",
        "description_hash", "is_english", "is_swe", "is_swe_adjacent",
        "is_control", "has_raw_description", "is_linkedin",
        "short_description_skip", "control_bucket", "scrape_date",
        "eligible_swe_extraction", "eligible_control_extraction",
        "eligible_control_unit", "eligible_for_extraction",
        "extraction_input_hash", "raw_description_word_count",
    ]
    pf = pq.ParquetFile(input_path)
    lightweight_frames = []
    for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
        chunk = pa.Table.from_batches([batch]).to_pandas()
        annotated_chunk = annotate_chunk(chunk)
        cols_to_keep = [c for c in _LIGHTWEIGHT_COLS if c in annotated_chunk.columns]
        lightweight_frames.append(annotated_chunk[cols_to_keep])
    annotated_light = pd.concat(lightweight_frames, ignore_index=True) if lightweight_frames else pd.DataFrame()
    del lightweight_frames

    control_cohort_df, selected_control_hashes = _select_control_cohort(annotated_light)

    # Build the logging summary directly from the lightweight frame.
    # This replicates the flag logic from process_chunk() without needing
    # the description column (all input flags are already present).
    annotated_light["selected_for_control_cohort"] = (
        annotated_light["extraction_input_hash"].astype(str).isin(selected_control_hashes)
    )
    extraction_scope = (
        annotated_light["is_swe"].fillna(False).astype(bool)
        | annotated_light["is_swe_adjacent"].fillna(False).astype(bool)
        | annotated_light["selected_for_control_cohort"]
    )
    annotated_light["needs_llm_extraction"] = (
        annotated_light["is_linkedin"]
        & annotated_light["is_english"]
        & annotated_light["has_raw_description"]
        & extraction_scope
        & ~annotated_light["short_description_skip"]
    )
    annotated_light["llm_extraction_reason"] = "not_routed"
    annotated_light.loc[
        annotated_light["short_description_skip"] & extraction_scope,
        "llm_extraction_reason",
    ] = "short_description"
    annotated_light.loc[
        annotated_light["needs_llm_extraction"],
        "llm_extraction_reason",
    ] = "routed"
    candidate_summary = build_extraction_candidates(annotated_light, control_cohort_df)
    prepared_for_logging = annotated_light
    del annotated_light

    # ---- Pass 2: build candidate records (streams from parquet again)
    candidate_rows = build_candidate_records(input_path, selected_control_hashes)

    conn = open_cache(cache_db)
    hashes = [str(row["extraction_input_hash"]) for row in candidate_rows]
    cached_rows = fetch_cached_rows(
        conn,
        hashes,
        EXTRACTION_TASK_NAME,
        EXTRACTION_PROMPT_VERSION,
        exclude_retryable_failures=True,
    )
    cached_hash_set = set(cached_rows.keys())

    # Budget-aware selection: all candidates go through the same budget pool.
    rows_to_process, category_allocation, uncached_per_category = select_rows_with_budget(
        candidates=candidate_rows,
        cached_hashes=cached_hash_set,
        budget=llm_budget,
        split=llm_budget_split,
        hash_key="extraction_input_hash",
    )
    deferred = sum(uncached_per_category.values()) - len(rows_to_process)

    log_stage9_plan(
        log,
        summarize_stage9_routing(
            prepared_for_logging,
            candidate_summary,
            int(len(selected_control_hashes)),
            cached_task_count=len(cached_rows),
            fresh_task_count=len(rows_to_process),
        ),
        max_workers=max_workers,
        runtime=runtime,
    )
    log_budget_plan(
        log,
        budget=llm_budget,
        split=llm_budget_split,
        uncached_per_category=uncached_per_category,
        category_allocation=category_allocation,
        deferred=deferred,
    )
    del prepared_for_logging, candidate_summary

    if rows_to_process:
        progress_checkpoints = set(build_progress_checkpoints(len(rows_to_process)))
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    process_candidate_row,
                    row,
                    runtime=runtime,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    error_log_path=error_log_path,
                    log=log,
                    quota_wait_hours=quota_wait_hours,
                ): row
                for row in rows_to_process
            }
            for future in as_completed(future_map):
                row = future_map[future]
                result = future.result()
                completed += 1
                log_stage9_sample_response(log, completed=completed, row=row, result=result)
                if should_cache_extraction_result(result):
                    store_cached_row(
                        conn,
                        input_hash=row["extraction_input_hash"],
                        task_name=EXTRACTION_TASK_NAME,
                        model=result["model"],
                        prompt_version=result["prompt_version"],
                        response_json=result["response_json"],
                        tokens_used=result["tokens_used"],
                    )
                    cached_rows[str(row["extraction_input_hash"])] = fetch_cached_row(
                        conn,
                        input_hash=row["extraction_input_hash"],
                        task_name=EXTRACTION_TASK_NAME,
                        prompt_version=EXTRACTION_PROMPT_VERSION,
                        exclude_retryable_failures=True,
                    )
                else:
                    cached_rows.pop(str(row["extraction_input_hash"]), None)
                if completed in progress_checkpoints:
                    log.info(
                        "Stage 9 progress | completed=%s/%s fresh extraction tasks (%.1f%%)",
                        f"{completed:,}",
                        f"{len(rows_to_process):,}",
                        100.0 * completed / len(rows_to_process),
                    )

    tmp_candidates_path = prepare_temp_output(candidates_path)
    tmp_results_path = prepare_temp_output(results_path)
    tmp_cleaned_path = prepare_temp_output(cleaned_path)
    tmp_control_cohort_path = prepare_temp_output(control_cohort_path)

    try:
        write_parquet_rows(candidate_rows, tmp_candidates_path)
        write_parquet_rows(
            [
                {
                    "extraction_input_hash": row["extraction_input_hash"],
                    "job_id": row["job_id"],
                    "source": row["source"],
                    "source_platform": row["source_platform"],
                    "title": row["title"],
                    "company_name": row["company_name"],
                    "description_hash": row["description_hash"],
                    "llm_model_extraction": None
                    if cached_rows.get(str(row["extraction_input_hash"])) is None
                    else cached_rows[str(row["extraction_input_hash"])]["model"],
                    "llm_prompt_version_extraction": None
                    if cached_rows.get(str(row["extraction_input_hash"])) is None
                    else cached_rows[str(row["extraction_input_hash"])]["prompt_version"],
                    "extraction_response_json": None
                    if cached_rows.get(str(row["extraction_input_hash"])) is None
                    else cached_rows[str(row["extraction_input_hash"])]["response_json"],
                }
                for row in candidate_rows
            ],
            tmp_results_path,
        )
        control_cohort_df.to_parquet(tmp_control_cohort_path, index=False)

        writer = None
        try:
            for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
                chunk = pa.Table.from_batches([batch]).to_pandas()
                prepared = process_chunk(chunk, selected_control_hashes=selected_control_hashes)
                prepared["description_core_llm"] = [
                    resolve_description_core_llm(row, cached_rows)
                    for row in prepared.itertuples(index=False)
                ]
                # llm_extraction_coverage: sequential overrides starting from "not_routed".
                in_scope = (
                    prepared["is_swe"].fillna(False).astype(bool)
                    | prepared["is_swe_adjacent"].fillna(False).astype(bool)
                    | prepared["selected_for_control_cohort"].fillna(False).astype(bool)
                )
                routed = prepared["needs_llm_extraction"].fillna(False).astype(bool)
                has_result = routed & prepared["extraction_input_hash"].astype(str).isin(cached_rows)
                coverage = pd.Series("not_routed", index=prepared.index, dtype="object")
                coverage.loc[in_scope & prepared["short_description_skip"].fillna(False)] = "skipped_short"
                coverage.loc[routed] = "deferred"
                coverage.loc[has_result] = "labeled"
                prepared["llm_extraction_coverage"] = coverage
                cleaned = prepared.drop(
                    columns=[
                        "has_raw_description",
                        "is_linkedin",
                        "raw_description_word_count",
                        "short_description_skip",
                        "eligible_swe_extraction",
                        "eligible_control_extraction",
                        "needs_llm_extraction",
                    ],
                    errors="ignore",
                )
                out_table = pa.Table.from_pandas(cleaned, preserve_index=False)
                if writer is None:
                    cleaned_schema = _promote_null_fields(out_table.schema)
                    writer = pq.ParquetWriter(tmp_cleaned_path, cleaned_schema)
                out_table = out_table.cast(cleaned_schema)
                writer.write_table(out_table)
        finally:
            if writer is not None:
                writer.close()
    except Exception:
        cleanup_temp_file(tmp_candidates_path)
        cleanup_temp_file(tmp_results_path)
        cleanup_temp_file(tmp_cleaned_path)
        cleanup_temp_file(tmp_control_cohort_path)
        conn.close()
        raise

    conn.close()
    promote_temp_file(tmp_candidates_path, candidates_path)
    promote_temp_file(tmp_results_path, results_path)
    promote_temp_file(tmp_cleaned_path, cleaned_path)
    promote_temp_file(tmp_control_cohort_path, control_cohort_path)

    log.info("Stage 9 complete in %.1fs", time.time() - t0)
    log.info("Selected control hashes: %s", f"{len(selected_control_hashes):,}")
    log.info("Extraction candidates: %s", f"{len(candidate_rows):,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 9 extraction routing and cleaned-text integration")
    parser.add_argument(
        "--llm-budget",
        type=int,
        required=True,
        help=(
            "Max new LLM calls (REQUIRED, no default). "
            "Use 0 for cache-only. Counts unique extraction tasks."
        ),
    )
    parser.add_argument(
        "--llm-budget-split",
        type=str,
        default=DEFAULT_BUDGET_SPLIT,
        help=(
            "Fractional split of the budget across categories: "
            "swe,swe_adjacent,control. Default '0.4,0.3,0.3'. Values are "
            "normalized to sum to 1.0."
        ),
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--candidates-output", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--cleaned-output", type=Path, default=DEFAULT_CLEANED_PATH)
    parser.add_argument("--control-cohort-output", type=Path, default=DEFAULT_CONTROL_COHORT_PATH)
    parser.add_argument("--cache-db", type=Path, default=DEFAULT_CACHE_DB)
    parser.add_argument("--error-log", type=Path, default=DEFAULT_ERROR_LOG)
    parser.add_argument("--codex-model", type=str, default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--openai-model", type=str, default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=30)
    parser.add_argument("--engines", type=str, default="codex")
    parser.add_argument("--claude-model", type=str, default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--quota-wait-hours", type=float, default=DEFAULT_QUOTA_WAIT_HOURS)
    parser.add_argument("--engine-tiers", type=str, default=None)
    parser.add_argument("--engine-timezone", type=str, default=DEFAULT_ENGINE_TIMEZONE)
    parser.add_argument("--remote", action="store_true", default=False,
                        help="Run LLM commands on the remote EC2 instance via SSH")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_remote_execution(args.remote)
    enabled_engines = parse_engine_list(args.engines)
    if args.remote and "openai" in enabled_engines:
        raise SystemExit("--remote is not supported when using the openai engine")
    run_stage9(
        llm_budget=args.llm_budget,
        llm_budget_split=parse_budget_split(args.llm_budget_split),
        input_path=args.input,
        candidates_path=args.candidates_output,
        results_path=args.results_output,
        cleaned_path=args.cleaned_output,
        control_cohort_path=args.control_cohort_output,
        cache_db=args.cache_db,
        error_log_path=args.error_log,
        codex_model=args.codex_model,
        openai_model=args.openai_model,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        max_workers=args.max_workers,
        enabled_engines=enabled_engines,
        claude_model=args.claude_model,
        quota_wait_hours=args.quota_wait_hours,
        engine_tiers=parse_engine_tiers(args.engine_tiers, enabled_engines),
        engine_timezone=args.engine_timezone,
    )
