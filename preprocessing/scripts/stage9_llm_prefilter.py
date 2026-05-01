#!/usr/bin/env python3
"""
Stage 9: deterministic frame selection, extraction execution, and cleaned-text
integration for the Stage 9 LLM sample.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
    ANALYSIS_GROUP_PRIORITY,
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
    select_fresh_call_tasks,
    select_sticky_task_frame,
    segment_description_into_units,
    split_budget_by_category,
    store_cached_row,
    try_provider,
    validate_extraction_payload,
    validate_extraction_selection,
    fetch_cached_row,
    fetch_cached_rows,
    derive_analysis_group,
    derive_date_bin,
    write_core_frame_manifest,
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
DEFAULT_CORE_FRAME_MANIFEST_PATH = INTERMEDIATE_DIR / "stage9_core_frame_manifest.json"
DEFAULT_CACHE_DB = CACHE_DIR / "llm_responses.db"
DEFAULT_ERROR_LOG = LOG_DIR / "llm_errors.jsonl"

CHUNK_SIZE = 50_000
MIN_DESCRIPTION_WORDS = 15


def _format_top_counts(mapping: dict[str, int] | None, *, limit: int = 8) -> str:
    if not mapping:
        return "none"
    items = sorted(mapping.items(), key=lambda item: (-int(item[1]), item[0]))
    rendered = [f"{key}={value:,}" for key, value in items[:limit] if int(value) > 0]
    return ", ".join(rendered) if rendered else "none"


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
    out["analysis_group"] = [
        derive_analysis_group(row)
        for row in out[["is_swe", "is_swe_adjacent", "is_control"]].to_dict("records")
    ]
    out["selection_date_bin"] = [
        derive_date_bin(
            {
                "source": source,
                "scrape_date": scrape_date,
                "date_posted": date_posted,
            }
        )
        for source, scrape_date, date_posted in zip(
            out.get("source", pd.Series(index=out.index, dtype="object")),
            out.get("scrape_date", pd.Series(index=out.index, dtype="object")),
            out.get("date_posted", pd.Series(index=out.index, dtype="object")),
        )
    ]
    eligible_text = out["is_linkedin"] & out["is_english"] & out["has_raw_description"]
    out["analysis_in_scope"] = out["analysis_group"].notna()
    out["eligible_for_extraction"] = eligible_text & out["analysis_group"].notna() & ~out["short_description_skip"]
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


def build_extraction_candidates(annotated: pd.DataFrame, selected_frame_hashes: set[str]) -> pd.DataFrame:
    routed = annotated.copy()
    routed["selected_for_llm_frame"] = routed["extraction_input_hash"].astype(str).isin(selected_frame_hashes)
    routed = routed.loc[routed["selected_for_llm_frame"] & ~routed["short_description_skip"]].copy()
    if routed.empty:
        return pd.DataFrame()
    agg_spec = dict(
        job_id=("job_id", "first"),
        source=("source", "first"),
        source_platform=("source_platform", "first"),
        title=("title", "first"),
        company_name=("company_name", "first"),
        description_hash=("description_hash", "first"),
        analysis_group=("analysis_group", "first"),
        selection_date_bin=("selection_date_bin", "first"),
        selected_for_llm_frame=("selected_for_llm_frame", "max"),
        selected_for_control_cohort=("is_control", "max"),
        source_row_count=("job_id", "count"),
    )
    if "description" in routed.columns:
        agg_spec["description"] = ("description", "first")
    return routed.groupby("extraction_input_hash", as_index=False).agg(**agg_spec)


def process_chunk(
    df: pd.DataFrame,
    selected_frame_hashes: set[str] | None = None,
    supplemental_cached_hashes: set[str] | None = None,
) -> pd.DataFrame:
    if {"eligible_for_extraction", "extraction_input_hash", "analysis_group"}.issubset(df.columns):
        out = df.copy()
    else:
        out = annotate_stage9_chunk(df)
    selected_frame_hashes = selected_frame_hashes or set()
    supplemental_cached_hashes = supplemental_cached_hashes or set()
    extraction_hashes = out["extraction_input_hash"].astype(str)
    eligible_for_extraction = out["eligible_for_extraction"].fillna(False).astype(bool)
    out["selected_for_llm_frame"] = extraction_hashes.isin(selected_frame_hashes)
    out["selected_for_control_cohort"] = out["selected_for_llm_frame"] & out["is_control"].fillna(False).astype(bool)
    out["llm_extraction_sample_tier"] = "none"
    out.loc[
        eligible_for_extraction & extraction_hashes.isin(supplemental_cached_hashes),
        "llm_extraction_sample_tier",
    ] = "supplemental_cache"
    out.loc[out["selected_for_llm_frame"], "llm_extraction_sample_tier"] = "core"
    out["needs_llm_extraction"] = out["selected_for_llm_frame"]
    out["llm_extraction_reason"] = "not_selected"
    out.loc[out["short_description_skip"] & out["analysis_group"].notna(), "llm_extraction_reason"] = "short_description"
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


def build_candidate_records(input_path: Path) -> list[dict]:
    pf = pq.ParquetFile(input_path)
    lookup: dict[str, dict] = {}
    for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
        chunk = pa.Table.from_batches([batch]).to_pandas()
        prepared = annotate_stage9_chunk(chunk)
        frame_mask = (
            prepared["is_linkedin"]
            & prepared["is_english"]
            & prepared["has_raw_description"]
            & prepared["analysis_group"].notna()
            & ~prepared["short_description_skip"]
        )
        routed = prepared.loc[frame_mask].copy()
        if routed.empty:
            continue
        grouped = (
            routed.groupby("extraction_input_hash", as_index=False)
            .agg(
                job_id=("job_id", "first"),
                source=("source", "first"),
                source_platform=("source_platform", "first"),
                title=("title", "first"),
                company_name=("company_name", "first"),
                description=("description", "first"),
                description_hash=("description_hash", "first"),
                date_posted=("date_posted", "first"),
                scrape_date=("scrape_date", "first"),
                selection_date_bin=("selection_date_bin", "first"),
                analysis_group=("analysis_group", "first"),
                is_swe=("is_swe", "max"),
                is_swe_adjacent=("is_swe_adjacent", "max"),
                is_control=("is_control", "max"),
                source_row_count=("job_id", "count"),
            )
            .to_dict("records")
        )
        for row in grouped:
            key = str(row["extraction_input_hash"])
            if key not in lookup:
                lookup[key] = row
                continue
            lookup[key]["source_row_count"] = int(lookup[key].get("source_row_count", 0)) + int(
                row.get("source_row_count", 0)
            )
    return list(lookup.values())


def resolve_description_core_llm(row, cached_rows: dict[str, dict]) -> str | None:
    if row.short_description_skip and row.analysis_group is not None:
        return ""
    sample_tier = getattr(row, "llm_extraction_sample_tier", None)
    if (
        not bool(getattr(row, "selected_for_llm_frame", False))
        and sample_tier != "supplemental_cache"
    ) or row.extraction_input_hash is None:
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
    cached_task_count: int,
    fresh_task_count: int,
) -> dict[str, int]:
    reason_counts = Counter(prepared["llm_extraction_reason"].fillna("unknown").astype(str))
    analysis_group_counts = (
        Counter(candidate_summary["analysis_group"].fillna("unknown").astype(str))
        if not candidate_summary.empty
        else Counter()
    )
    routed_row_count = int(candidate_summary["source_row_count"].sum()) if not candidate_summary.empty else 0
    unique_task_count = int(len(candidate_summary))
    return {
        "total_rows": int(len(prepared)),
        "linkedin_rows": int(prepared["is_linkedin"].fillna(False).sum()),
        "english_rows": int(prepared["is_english"].fillna(False).sum()),
        "analysis_scope_rows": int(prepared["analysis_group"].notna().sum()),
        "control_scope_rows": int(prepared["is_control"].fillna(False).sum()),
        "short_skip_rows": int(reason_counts.get("short_description", 0)),
        "routed_rows": int(reason_counts.get("routed", 0)),
        "not_selected_rows": int(reason_counts.get("not_selected", 0)),
        "unique_tasks": unique_task_count,
        "swe_combined_tasks": int(analysis_group_counts.get("swe_combined", 0)),
        "control_tasks": int(analysis_group_counts.get("control", 0)),
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
        "Routing summary | rows=%s | linkedin=%s | english=%s | analysis_scope=%s | control_scope=%s",
        f"{summary['total_rows']:,}",
        f"{summary['linkedin_rows']:,}",
        f"{summary['english_rows']:,}",
        f"{summary['analysis_scope_rows']:,}",
        f"{summary['control_scope_rows']:,}",
    )
    log.info(
        "Extraction volume | routed_rows=%s | short_skips=%s | not_selected=%s | unique_tasks=%s | swe_combined_tasks=%s | control_tasks=%s | deduped_rows=%s",
        f"{summary['routed_rows']:,}",
        f"{summary['short_skip_rows']:,}",
        f"{summary['not_selected_rows']:,}",
        f"{summary['unique_tasks']:,}",
        f"{summary['swe_combined_tasks']:,}",
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
            "analysis_group": row.get("analysis_group"),
            "selection_date_bin": row.get("selection_date_bin"),
            "source_row_count": row.get("source_row_count"),
        },
    )


def log_stage9_selection_debug(
    log: logging.Logger,
    *,
    selection_summary: dict[str, object],
    effective_selection_target: int,
    selected_rows: list[dict],
    preexisting_core_cached_hashes: set[str],
    supplemental_cached_hashes: set[str],
    fresh_candidates: list[dict],
    rows_to_process: list[dict],
    deferred: int,
) -> None:
    top_up_summary = selection_summary.get("top_up_summary", {}) if isinstance(selection_summary, dict) else {}
    top_up_selected = int(top_up_summary.get("selected_count", 0)) if isinstance(top_up_summary, dict) else 0
    log.info(
        "Frame plan | selection_target=%s | selected_tasks=%s | retained=%s | top_up=%s | top_up_selected=%s | fresh_candidates=%s | deferred=%s | reset=%s",
        f"{effective_selection_target:,}",
        f"{len(selected_rows):,}",
        f"{int(selection_summary.get('retained_count', 0)):,}",
        f"{int(selection_summary.get('top_up_count', 0)):,}",
        f"{top_up_selected:,}",
        f"{len(fresh_candidates):,}",
        f"{deferred:,}",
        bool(selection_summary.get("reset", False)),
    )
    log.info(
        "Manifest state | path=%s | prior_selected=%s | retained=%s | final_selected=%s",
        selection_summary.get("manifest_path", DEFAULT_CORE_FRAME_MANIFEST_PATH),
        f"{int(selection_summary.get('manifest_selected_count', 0)):,}",
        f"{int(selection_summary.get('manifest_retained_count', 0)):,}",
        f"{len(selection_summary.get('selected_hashes', [])):,}",
    )
    log.info(
        "Selection targets | source_group=%s",
        _format_top_counts(selection_summary.get("source_group_targets")),
    )
    log.info(
        "Top-up targets | source_group=%s | cells=%s",
        _format_top_counts(top_up_summary.get("source_group_targets") if isinstance(top_up_summary, dict) else None),
        _format_top_counts(top_up_summary.get("cell_targets") if isinstance(top_up_summary, dict) else None, limit=10),
    )
    log.info(
        "Selection resolution pool | core_cached=%s | supplemental_cached=%s | fresh_selected=%s",
        f"{len(preexisting_core_cached_hashes):,}",
        f"{len(supplemental_cached_hashes):,}",
        f"{len(rows_to_process):,}",
    )


def run_stage9(
    *,
    llm_budget: int,
    llm_budget_split: dict[str, float],
    selection_target: int | None = None,
    selection_targets: dict[str, int] | None = None,
    input_path: Path = DEFAULT_INPUT_PATH,
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    results_path: Path = DEFAULT_RESULTS_PATH,
    cleaned_path: Path = DEFAULT_CLEANED_PATH,
    control_cohort_path: Path = DEFAULT_CONTROL_COHORT_PATH,
    core_frame_manifest_path: Path = DEFAULT_CORE_FRAME_MANIFEST_PATH,
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
    reset_core_frame: bool = False,
) -> None:
    if llm_budget < 0:
        raise ValueError(f"llm_budget must be >= 0, got {llm_budget}")
    if selection_target is not None and selection_target < 0:
        raise ValueError(f"selection_target must be >= 0, got {selection_target}")
    if selection_targets is not None:
        bad = {g: v for g, v in selection_targets.items() if v < 0}
        if bad:
            raise ValueError(f"selection_targets values must be >= 0, got {bad}")
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

    # ---- Pass 1: stream chunks, keep only lightweight columns for summary
    # logging and row-level frame annotation. Text columns are dropped after
    # annotation to avoid holding the full dataset in RAM.
    _LIGHTWEIGHT_COLS = [
        "job_id", "source", "source_platform", "title", "company_name",
        "description_hash", "is_english", "is_swe", "is_swe_adjacent",
        "is_control", "has_raw_description", "is_linkedin",
        "short_description_skip", "scrape_date",
        "date_posted", "analysis_group", "selection_date_bin",
        "eligible_for_extraction",
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

    # ---- Pass 2: build candidate records (streams from parquet again)
    candidate_rows = build_candidate_records(input_path)
    if selection_targets is not None:
        effective_selection_target = int(sum(selection_targets.values()))
    else:
        effective_selection_target = llm_budget if selection_target is None else selection_target
    selected_rows, selection_summary = select_sticky_task_frame(
        candidate_rows,
        selection_target=effective_selection_target if selection_targets is None else 0,
        selection_targets=selection_targets,
        hash_key="extraction_input_hash",
        manifest_path=core_frame_manifest_path,
        groups=ANALYSIS_GROUP_PRIORITY,
        reset=reset_core_frame,
    )
    selected_frame_hashes = {str(row["extraction_input_hash"]) for row in selected_rows}
    conn = open_cache(cache_db)
    hashes = [str(row["extraction_input_hash"]) for row in candidate_rows]
    cached_rows = fetch_cached_rows(
        conn,
        hashes,
        EXTRACTION_TASK_NAME,
        EXTRACTION_PROMPT_VERSION,
        exclude_retryable_failures=True,
    )
    preexisting_cached_hashes = set(cached_rows.keys())
    preexisting_core_cached_hashes = preexisting_cached_hashes & selected_frame_hashes
    supplemental_cached_hashes = preexisting_cached_hashes - selected_frame_hashes

    fresh_candidates = [
        row for row in selected_rows
        if str(row["extraction_input_hash"]) not in preexisting_core_cached_hashes
    ]
    uncached_per_category = {
        group: sum(1 for row in fresh_candidates if row.get("analysis_group") == group)
        for group in ANALYSIS_GROUP_PRIORITY
    }
    category_targets = split_budget_by_category(llm_budget, uncached_per_category, llm_budget_split)
    rows_to_process = []
    for analysis_group in ANALYSIS_GROUP_PRIORITY:
        group_rows = [row for row in fresh_candidates if row.get("analysis_group") == analysis_group]
        if not group_rows:
            continue
        selected_group_rows, _fresh_summary = select_fresh_call_tasks(
            group_rows,
            llm_budget=category_targets.get(analysis_group, 0),
            hash_key="extraction_input_hash",
            groups=(analysis_group,),
        )
        rows_to_process.extend(selected_group_rows)
    category_allocation = {
        group: sum(1 for row in rows_to_process if row.get("analysis_group") == group)
        for group in ANALYSIS_GROUP_PRIORITY
    }
    deferred = len(fresh_candidates) - len(rows_to_process)

    prepared_for_logging = process_chunk(
        annotated_light,
        selected_frame_hashes=selected_frame_hashes,
        supplemental_cached_hashes=supplemental_cached_hashes,
    )
    candidate_summary = pd.DataFrame(selected_rows)
    control_cohort_df = pd.DataFrame(
        [
            {
                "extraction_input_hash": row["extraction_input_hash"],
                "selection_date_bin": row["selection_date_bin"],
                "selected_for_control_cohort": True,
                "job_id": row["job_id"],
                "source": row["source"],
                "source_platform": row["source_platform"],
                "title": row["title"],
                "company_name": row["company_name"],
                "description_hash": row["description_hash"],
                "source_row_count": row.get("source_row_count", 1),
            }
            for row in selected_rows
            if bool(row.get("is_control"))
        ]
    )
    del annotated_light

    log_stage9_plan(
        log,
        summarize_stage9_routing(
            prepared_for_logging,
            candidate_summary,
            cached_task_count=len(preexisting_core_cached_hashes),
            fresh_task_count=len(rows_to_process),
        ),
        max_workers=max_workers,
        runtime=runtime,
    )
    log_stage9_selection_debug(
        log,
        selection_summary=selection_summary,
        effective_selection_target=effective_selection_target,
        selected_rows=selected_rows,
        preexisting_core_cached_hashes=preexisting_core_cached_hashes,
        supplemental_cached_hashes=supplemental_cached_hashes,
        fresh_candidates=fresh_candidates,
        rows_to_process=rows_to_process,
        deferred=deferred,
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

    fresh_hashes = {str(row["extraction_input_hash"]) for row in rows_to_process}

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
        write_parquet_rows(
            [
                {
                    **row,
                    "selected_for_llm_frame": str(row["extraction_input_hash"]) in selected_frame_hashes,
                    "selected_for_control_cohort": (
                        str(row["extraction_input_hash"]) in selected_frame_hashes and bool(row.get("is_control"))
                    ),
                    "llm_extraction_sample_tier": (
                        "core"
                        if str(row["extraction_input_hash"]) in selected_frame_hashes
                        else "supplemental_cache"
                        if str(row["extraction_input_hash"]) in supplemental_cached_hashes
                        else "none"
                    ),
                }
                for row in candidate_rows
            ],
            tmp_candidates_path,
        )
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
                    "selected_for_llm_frame": str(row["extraction_input_hash"]) in selected_frame_hashes,
                    "selected_for_control_cohort": (
                        str(row["extraction_input_hash"]) in selected_frame_hashes and bool(row.get("is_control"))
                    ),
                    "llm_extraction_sample_tier": (
                        "core"
                        if str(row["extraction_input_hash"]) in selected_frame_hashes
                        else "supplemental_cache"
                        if str(row["extraction_input_hash"]) in supplemental_cached_hashes
                        else "none"
                    ),
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
                prepared = process_chunk(
                    chunk,
                    selected_frame_hashes=selected_frame_hashes,
                    supplemental_cached_hashes=supplemental_cached_hashes,
                )
                prepared["description_core_llm"] = [
                    resolve_description_core_llm(row, cached_rows)
                    for row in prepared.itertuples(index=False)
                ]
                # llm_extraction_coverage: frame-aware resolution state.
                routed = prepared["needs_llm_extraction"].fillna(False).astype(bool)
                extraction_hashes = prepared["extraction_input_hash"].astype(str)
                has_result = extraction_hashes.isin(cached_rows)
                coverage = pd.Series("not_selected", index=prepared.index, dtype="object")
                coverage.loc[prepared["analysis_group"].notna() & prepared["short_description_skip"].fillna(False)] = "skipped_short"
                coverage.loc[routed] = "deferred"
                coverage.loc[has_result] = "labeled"
                prepared["llm_extraction_coverage"] = coverage
                resolution = pd.Series("not_selected", index=prepared.index, dtype="object")
                resolution.loc[prepared["analysis_group"].notna() & prepared["short_description_skip"].fillna(False)] = "skipped_short"
                resolution.loc[routed] = "deferred"
                resolution.loc[has_result] = "cached_llm"
                resolution.loc[has_result & extraction_hashes.isin(fresh_hashes)] = "fresh_llm"
                prepared["llm_extraction_resolution"] = resolution
                cleaned = prepared.drop(
                    columns=[
                        "has_raw_description",
                        "is_linkedin",
                        "raw_description_word_count",
                        "short_description_skip",
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
    write_core_frame_manifest(
        core_frame_manifest_path,
        selected_hashes=selection_summary.get("selected_hashes", []),
        hash_key="extraction_input_hash",
    )

    log.info("Stage 9 complete in %.1fs", time.time() - t0)
    log.info("Selected frame tasks: %s", f"{len(selected_rows):,}")
    log.info("Stage 9 candidates: %s", f"{len(candidate_rows):,}")


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
            "Fractional split of the fresh-call budget across categories: "
            "swe_combined,control. Default '0.7,0.3'."
        ),
    )
    parser.add_argument(
        "--selection-target",
        type=int,
        default=None,
        help=(
            "Minimum unique-task size for the persisted Stage 9 core frame, "
            "split evenly across analysis groups. Defaults to llm-budget when "
            "omitted. Mutually exclusive with --selection-targets."
        ),
    )
    parser.add_argument(
        "--selection-targets",
        type=str,
        default=None,
        help=(
            "Per-group selection targets, e.g. 'swe_combined=75000,control=55000'. "
            "Overrides --selection-target when provided."
        ),
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--candidates-output", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--cleaned-output", type=Path, default=DEFAULT_CLEANED_PATH)
    parser.add_argument("--control-cohort-output", type=Path, default=DEFAULT_CONTROL_COHORT_PATH)
    parser.add_argument("--core-frame-manifest", type=Path, default=DEFAULT_CORE_FRAME_MANIFEST_PATH)
    parser.add_argument("--reset-core-frame", action="store_true", default=False)
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
    parsed_selection_targets: dict[str, int] | None = None
    if args.selection_targets:
        parsed_selection_targets = {}
        for piece in args.selection_targets.split(","):
            piece = piece.strip()
            if not piece:
                continue
            if "=" not in piece:
                raise SystemExit(f"--selection-targets entry must be 'group=N': {piece!r}")
            key, value = piece.split("=", 1)
            parsed_selection_targets[key.strip()] = int(value.strip())
    run_stage9(
        llm_budget=args.llm_budget,
        llm_budget_split=parse_budget_split(args.llm_budget_split),
        selection_target=args.selection_target,
        selection_targets=parsed_selection_targets,
        input_path=args.input,
        candidates_path=args.candidates_output,
        results_path=args.results_output,
        cleaned_path=args.cleaned_output,
        control_cohort_path=args.control_cohort_output,
        core_frame_manifest_path=args.core_frame_manifest,
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
        reset_core_frame=args.reset_core_frame,
    )
