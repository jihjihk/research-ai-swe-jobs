#!/usr/bin/env python3
"""
Stage 10: LLM classification and final posting-level integration.

Input:
  - preprocessing/intermediate/stage9_llm_cleaned.parquet

Outputs:
  - preprocessing/cache/llm_responses.db
  - preprocessing/intermediate/stage10_llm_classification_results.parquet
  - preprocessing/intermediate/stage10_llm_integrated.parquet
  - optionally preprocessing/intermediate/stage11_llm_integrated.parquet
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from io_utils import cleanup_temp_file, prepare_temp_output, promote_temp_file
from llm_shared import (
    ANALYSIS_GROUP_PRIORITY,
    CLASSIFICATION_PROMPT_VERSION,
    CLASSIFICATION_TASK_NAME,
    DEFAULT_BUDGET_SPLIT,
    DEFAULT_ENGINE_TIMEZONE,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CODEX_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_QUOTA_WAIT_HOURS,
    LLMEngineRuntime,
    SENIORITY_3LEVEL,
    SUPPORTED_PROVIDERS,
    build_engine_configs,
    build_progress_checkpoints,
    call_subprocess,
    chunked,
    compute_classification_input_hash,
    configure_remote_execution,
    derive_classification_input,
    execute_task_with_runtime,
    fetch_cached_row,
    fetch_cached_rows,
    format_engine_labels,
    log_budget_plan,
    log_sampled_llm_response,
    open_cache,
    parse_budget_split,
    parse_engine_tiers,
    parse_engine_list,
    render_classification_prompt,
    select_fresh_call_tasks,
    segment_description_into_units,
    split_budget_by_category,
    store_cached_row,
    try_provider,
    validate_classification_payload,
    validate_extraction_payload,
)


PROJECT_ROOT = Path(__file__).parent.parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
CACHE_DIR = PROJECT_ROOT / "preprocessing" / "cache"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

DEFAULT_INPUT_PATH = INTERMEDIATE_DIR / "stage9_llm_cleaned.parquet"
DEFAULT_RESULTS_PATH = INTERMEDIATE_DIR / "stage10_llm_classification_results.parquet"
DEFAULT_INTEGRATED_PATH = INTERMEDIATE_DIR / "stage10_llm_integrated.parquet"
DEFAULT_COMPAT_OUTPUT_PATH = INTERMEDIATE_DIR / "stage11_llm_integrated.parquet"
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
            logging.FileHandler(LOG_DIR / "stage10_llm.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def prepare_classification_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw_description = out["description"].fillna("").astype(str)
    raw_word_count = raw_description.str.split().str.len()
    is_linkedin = out["source_platform"].fillna("").astype(str).str.lower().eq("linkedin")
    is_english = out["is_english"].fillna(False).astype(bool)
    if "selected_for_llm_frame" not in out.columns:
        raise ValueError(
            "Stage 10 requires selected_for_llm_frame from Stage 9 and will not widen the core frame via fallback."
        )
    in_scope = out["selected_for_llm_frame"].fillna(False).astype(bool)
    short_skip = raw_word_count.lt(MIN_DESCRIPTION_WORDS)
    classification_eligible = is_linkedin & is_english & ~short_skip

    classification_input = [
        derive_classification_input(core_llm, desc)
        for core_llm, desc in zip(
            out.get("description_core_llm", pd.Series(index=out.index, dtype="object")),
            out.get("description", pd.Series(index=out.index, dtype="object")),
        )
    ]
    out["classification_input"] = classification_input
    out["needs_llm_classification"] = in_scope & classification_eligible
    out["classification_input_hash"] = [
        compute_classification_input_hash(title, company, text) if needs else None
        for title, company, text, needs in zip(
            out.get("title", pd.Series(index=out.index, dtype="object")),
            out.get("company_name", pd.Series(index=out.index, dtype="object")),
            out["classification_input"],
            classification_eligible,
        )
    ]
    out["llm_classification_sample_tier"] = "none"
    out.loc[in_scope, "llm_classification_sample_tier"] = "core"
    out["llm_classification_reason"] = "not_selected"
    out.loc[in_scope & short_skip, "llm_classification_reason"] = "short_description_excluded_by_stage9"
    out.loc[out["needs_llm_classification"], "llm_classification_reason"] = "routed"
    return out


def has_uncached_rows(df: pd.DataFrame, conn) -> bool:
    candidate_hashes = (
        df.loc[df["needs_llm_classification"], "classification_input_hash"].dropna().astype(str).unique().tolist()
    )
    if not candidate_hashes:
        return False
    cached = fetch_cached_rows(
        conn,
        candidate_hashes,
        CLASSIFICATION_TASK_NAME,
        CLASSIFICATION_PROMPT_VERSION,
    )
    return any(candidate_hash not in cached for candidate_hash in candidate_hashes)


def call_task_with_engine(
    task_name: str,
    prompt: str,
    input_hash: str,
    error_log_path: Path,
    log: logging.Logger,
    payload_validator,
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
        task_name=task_name,
        prompt=prompt,
        input_hash=input_hash,
        error_log_path=error_log_path,
        log=log,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        payload_validator=payload_validator,
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
) -> dict | None:
    prompt = render_classification_prompt(
        row["title"],
        row["company_name"],
        row["classification_input"],
    )
    return call_task_with_engine(
        task_name=CLASSIFICATION_TASK_NAME,
        runtime=runtime,
        prompt=prompt,
        input_hash=row["classification_input_hash"],
        error_log_path=error_log_path,
        log=log,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        payload_validator=validate_classification_payload,
        quota_wait_hours=quota_wait_hours,
    )


def build_results_row(row_meta: dict, classification_input_hash: str, classification_cached: dict | None) -> dict:
    payload = json.loads(classification_cached["response_json"]) if classification_cached is not None else {}
    return {
        "classification_input_hash": classification_input_hash,
        "title": row_meta.get("title"),
        "company_name": row_meta.get("company_name"),
        "source": row_meta.get("source"),
        "source_platform": row_meta.get("source_platform"),
        "job_id": row_meta.get("job_id"),
        "classification_row_count": row_meta.get("classification_row_count", 1),
        "llm_model_classification": None if classification_cached is None else classification_cached["model"],
        "llm_prompt_version_classification": (
            None if classification_cached is None else classification_cached["prompt_version"]
        ),
        "classification_response_json": (
            None if classification_cached is None else classification_cached["response_json"]
        ),
        "classification_tokens_used": (
            None
            if classification_cached is None or classification_cached["tokens_used"] is None
            else float(classification_cached["tokens_used"])
        ),
        "swe_classification_llm": payload.get("swe_classification"),
        "is_swe_combined_llm": (
            None
            if payload.get("swe_classification") is None
            else payload.get("swe_classification") in ("SWE", "SWE_ADJACENT")
        ),
        "ghost_assessment_llm": payload.get("ghost_assessment"),
        "yoe_min_years_llm": payload.get("yoe_min_years"),
    }


def integrate_chunk(
    chunk: pd.DataFrame,
    classification_cache: dict[str, dict],
    *,
    fresh_hashes: set[str] | None = None,
) -> pd.DataFrame:
    out = prepare_classification_rows(chunk)
    in_scope = out["selected_for_llm_frame"].fillna(False).astype(bool)
    has_cached_result = out["classification_input_hash"].notna() & out["classification_input_hash"].astype(str).isin(
        classification_cache
    )
    supplemental_cached_mask = ~in_scope & has_cached_result
    out.loc[supplemental_cached_mask, "llm_classification_sample_tier"] = "supplemental_cache"

    swe_values = []
    llm_seniority_values = []
    ghost_values = []
    yoe_values = []
    model_values = []
    prompt_values = []

    for row in out.itertuples(index=False):
        input_hash = row.classification_input_hash
        cached = classification_cache.get(str(input_hash)) if input_hash is not None else None
        payload = json.loads(cached["response_json"]) if cached is not None else {}
        swe_values.append(payload.get("swe_classification"))
        llm_seniority_values.append(payload.get("seniority"))
        ghost_values.append(payload.get("ghost_assessment"))
        yoe_values.append(payload.get("yoe_min_years"))
        model_values.append(None if cached is None else cached["model"])
        prompt_values.append(None if cached is None else cached["prompt_version"])

    out["swe_classification_llm"] = swe_values
    out["is_swe_combined_llm"] = pd.Series(
        [
            None if value is None else (value in ("SWE", "SWE_ADJACENT"))
            for value in swe_values
        ],
        index=out.index,
        dtype="boolean",
    )
    out["ghost_assessment_llm"] = ghost_values
    out["yoe_min_years_llm"] = pd.Series(yoe_values, dtype="Int64")
    out["llm_model_classification"] = model_values
    out["llm_prompt_version_classification"] = prompt_values

    # llm_classification_coverage: map routing reason → coverage, upgrade to
    # "labeled" when a cache hit exists for a routed row.
    reason = out["llm_classification_reason"].fillna("not_selected").astype(str)
    coverage = reason.map({
        "not_selected": "not_selected",
        "short_description_excluded_by_stage9": "skipped_short",
        "routed": "deferred",
    }).fillna("not_selected")
    has_core_result = reason.eq("routed") & has_cached_result
    coverage.loc[has_core_result | supplemental_cached_mask] = "labeled"
    out["llm_classification_coverage"] = coverage
    resolution = coverage.copy()
    if fresh_hashes is None:
        resolution.loc[resolution.eq("labeled")] = "cached_llm"
    else:
        fresh_mask = has_core_result & out["classification_input_hash"].astype(str).isin(fresh_hashes)
        cached_mask = coverage.eq("labeled") & ~fresh_mask
        resolution.loc[fresh_mask] = "fresh_llm"
        resolution.loc[cached_mask] = "cached_llm"
    out["llm_classification_resolution"] = resolution

    # Overwrite seniority_final with the LLM result for labeled rows. The
    # rule-based snapshot lives in seniority_rule / seniority_rule_source.
    llm_seniority_series = pd.Series(llm_seniority_values, index=out.index, dtype="object")
    labeled_mask = coverage.eq("labeled") & llm_seniority_series.notna()
    out.loc[labeled_mask, "seniority_final"] = llm_seniority_series[labeled_mask]
    out.loc[labeled_mask, "seniority_final_source"] = "llm"
    # Recompute seniority_3level since seniority_final may have changed.
    out["seniority_3level"] = out["seniority_final"].map(SENIORITY_3LEVEL).fillna("unknown")

    return out.drop(columns=["classification_input"], errors="ignore")


def summarize_stage10_routing(
    *,
    total_rows: int,
    routed_rows: int,
    selected_control_rows: int,
    reason_counts: Counter,
    unique_task_count: int,
    duplicate_rows_collapsed: int,
    cached_task_count: int,
    fresh_task_count: int,
) -> dict[str, int]:
    return {
        "total_rows": int(total_rows),
        "routed_rows": int(routed_rows),
        "selected_control_rows": int(selected_control_rows),
        "short_skip_rows": int(reason_counts.get("short_description_excluded_by_stage9", 0)),
        "not_routed_rows": int(reason_counts.get("not_selected", 0)),
        "unique_tasks": int(unique_task_count),
        "duplicate_rows_collapsed": int(duplicate_rows_collapsed),
        "cached_tasks": int(cached_task_count),
        "fresh_tasks": int(fresh_task_count),
    }


def log_stage10_plan(log: logging.Logger, summary: dict[str, int], *, max_workers: int, runtime: LLMEngineRuntime) -> None:
    log.info(
        "Execution plan | workers=%s | engines=%s",
        max_workers,
        format_engine_labels(runtime.engines),
    )
    log.info(
        "Routing summary | rows=%s | selected_controls=%s | routed_rows=%s | short_skips=%s | not_routed=%s",
        f"{summary['total_rows']:,}",
        f"{summary['selected_control_rows']:,}",
        f"{summary['routed_rows']:,}",
        f"{summary['short_skip_rows']:,}",
        f"{summary['not_routed_rows']:,}",
    )
    log.info(
        "Classification volume | unique_tasks=%s | deduped_rows=%s",
        f"{summary['unique_tasks']:,}",
        f"{summary['duplicate_rows_collapsed']:,}",
    )
    log.info(
        "Cache plan | cached=%s | fresh=%s",
        f"{summary['cached_tasks']:,}",
        f"{summary['fresh_tasks']:,}",
    )


def log_stage10_sample_response(log: logging.Logger, *, completed: int, row: dict, result: dict) -> None:
    log_sampled_llm_response(
        log,
        stage_label="Stage 10",
        completed=completed,
        input_hash=str(row.get("classification_input_hash") or ""),
        job_id=str(row.get("job_id") or ""),
        model=str(result.get("model") or ""),
        response_json=result.get("response_json"),
        extra_fields={
            "row_count": row.get("classification_row_count"),
            "source_platform": row.get("source_platform"),
        },
    )


def log_stage10_selection_debug(
    log: logging.Logger,
    *,
    candidate_rows: list[dict],
    supplemental_candidate_rows: list[dict],
    cached_rows: dict[str, dict],
    supplemental_cached_rows: dict[str, dict],
    fresh_candidates: list[dict],
    rows_to_process: list[dict],
    deferred: int,
) -> None:
    def _group(row: dict) -> str:
        if row.get("is_swe") or row.get("is_swe_adjacent"):
            return "swe_combined"
        return "control"

    core_by_group = Counter(_group(row) for row in candidate_rows)
    supplemental_by_group = Counter(_group(row) for row in supplemental_candidate_rows)
    fresh_by_group = Counter(_group(row) for row in fresh_candidates)
    log.info(
        "Frame inheritance | core_tasks=%s | core_cached=%s | supplemental_candidates=%s | supplemental_cached=%s",
        f"{len(candidate_rows):,}",
        f"{len(cached_rows):,}",
        f"{len(supplemental_candidate_rows):,}",
        f"{len(supplemental_cached_rows):,}",
    )
    log.info(
        "Core scope by group | %s",
        _format_top_counts(dict(core_by_group), limit=3),
    )
    log.info(
        "Supplemental cache by group | candidates=%s | cached=%s",
        _format_top_counts(dict(supplemental_by_group), limit=3),
        _format_top_counts(
            dict(
                Counter(
                    _group(row)
                    for row in supplemental_candidate_rows
                    if str(row.get("classification_input_hash")) in supplemental_cached_rows
                )
            ),
            limit=3,
        ),
    )
    log.info(
        "Fresh-call pool | unresolved_core=%s | selected_fresh=%s | deferred=%s | by_group=%s",
        f"{len(fresh_candidates):,}",
        f"{len(rows_to_process):,}",
        f"{deferred:,}",
        _format_top_counts(dict(fresh_by_group), limit=3),
    )


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


def run_stage10(
    *,
    llm_budget: int,
    llm_budget_split: dict[str, float],
    input_path: Path = DEFAULT_INPUT_PATH,
    results_path: Path = DEFAULT_RESULTS_PATH,
    integrated_path: Path = DEFAULT_INTEGRATED_PATH,
    compat_output_path: Path | None = DEFAULT_COMPAT_OUTPUT_PATH,
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

    log.info("=" * 70)
    log.info("Stage 10: LLM classification + final integration")
    log.info("=" * 70)
    log.info("Input: %s", input_path)
    log.info("Results output: %s", results_path)
    log.info("Integrated output: %s", integrated_path)
    if compat_output_path is not None:
        log.info("Compatibility alias output: %s", compat_output_path)

    pf = pq.ParquetFile(input_path)
    candidate_lookup: dict[str, dict] = {}
    supplemental_candidate_lookup: dict[str, dict] = {}
    total_rows = pf.metadata.num_rows
    reason_counts: Counter = Counter()
    routed_row_count = 0
    selected_control_rows = 0

    for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
        chunk = pa.Table.from_batches([batch]).to_pandas()
        prepared = prepare_classification_rows(chunk)
        reason_counts.update(prepared["llm_classification_reason"].fillna("unknown").astype(str))
        routed_row_count += int(prepared["needs_llm_classification"].fillna(False).sum())
        selected_control_rows += int(
            (
                prepared["selected_for_llm_frame"].fillna(False).astype(bool)
                & prepared["is_control"].fillna(False).astype(bool)
            ).sum()
        )
        supplemental_candidates = prepared.loc[
            ~prepared["selected_for_llm_frame"].fillna(False).astype(bool)
            & prepared["classification_input_hash"].notna()
        ].copy()
        if not supplemental_candidates.empty:
            grouped_supplemental = (
                supplemental_candidates.groupby("classification_input_hash", dropna=True, as_index=False)
                .agg(
                    job_id=("job_id", "first"),
                    source=("source", "first"),
                    source_platform=("source_platform", "first"),
                    title=("title", "first"),
                    company_name=("company_name", "first"),
                    classification_input=("classification_input", "first"),
                    classification_row_count=("job_id", "count"),
                )
                .to_dict("records")
            )
            for row in grouped_supplemental:
                input_hash = str(row["classification_input_hash"])
                if input_hash in candidate_lookup:
                    continue
                supplemental_candidate_lookup.setdefault(input_hash, row)
        candidates = prepared.loc[prepared["needs_llm_classification"]].copy()
        if candidates.empty:
            continue
        # Ensure category flags exist for budget allocation.
        for col in ("is_swe", "is_swe_adjacent", "is_control"):
            if col not in candidates.columns:
                candidates[col] = False
        grouped = (
            candidates.groupby("classification_input_hash", dropna=True, as_index=False)
            .agg(
                job_id=("job_id", "first"),
                source=("source", "first"),
                source_platform=("source_platform", "first"),
                title=("title", "first"),
                company_name=("company_name", "first"),
                classification_input=("classification_input", "first"),
                classification_row_count=("job_id", "count"),
                date_posted=("date_posted", "first"),
                scrape_date=("scrape_date", "first"),
                selection_date_bin=("selection_date_bin", "first"),
                is_swe=("is_swe", "max"),
                is_swe_adjacent=("is_swe_adjacent", "max"),
                is_control=("is_control", "max"),
            )
            .to_dict("records")
        )
        for row in grouped:
            candidate_lookup.setdefault(str(row["classification_input_hash"]), row)

    candidate_rows = list(candidate_lookup.values())
    supplemental_candidate_rows = list(supplemental_candidate_lookup.values())

    conn = open_cache(cache_db)
    hashes = [str(row["classification_input_hash"]) for row in candidate_rows]
    cached_rows = fetch_cached_rows(conn, hashes, CLASSIFICATION_TASK_NAME, CLASSIFICATION_PROMPT_VERSION)
    cached_hash_set = set(cached_rows.keys())
    supplemental_hashes = [str(row["classification_input_hash"]) for row in supplemental_candidate_rows]
    supplemental_cached_rows = fetch_cached_rows(
        conn,
        supplemental_hashes,
        CLASSIFICATION_TASK_NAME,
        CLASSIFICATION_PROMPT_VERSION,
    )

    fresh_candidates = [
        row for row in candidate_rows
        if str(row["classification_input_hash"]) not in cached_hash_set
    ]

    def _row_group(row: dict) -> str:
        if row.get("is_swe") or row.get("is_swe_adjacent"):
            return "swe_combined"
        return "control"

    uncached_per_category = {
        group: sum(1 for row in fresh_candidates if _row_group(row) == group)
        for group in ANALYSIS_GROUP_PRIORITY
    }
    category_targets = split_budget_by_category(llm_budget, uncached_per_category, llm_budget_split)
    rows_to_process = []
    for analysis_group in ANALYSIS_GROUP_PRIORITY:
        group_rows = [row for row in fresh_candidates if _row_group(row) == analysis_group]
        if not group_rows:
            continue
        selected_group_rows, _fresh_summary = select_fresh_call_tasks(
            group_rows,
            llm_budget=category_targets.get(analysis_group, 0),
            hash_key="classification_input_hash",
            groups=(analysis_group,),
        )
        rows_to_process.extend(selected_group_rows)
    category_allocation = {
        group: sum(1 for row in rows_to_process if _row_group(row) == group)
        for group in ANALYSIS_GROUP_PRIORITY
    }
    deferred = len(fresh_candidates) - len(rows_to_process)

    log_stage10_plan(
        log,
        summarize_stage10_routing(
            total_rows=total_rows,
            routed_rows=routed_row_count,
            selected_control_rows=selected_control_rows,
            reason_counts=reason_counts,
            unique_task_count=len(candidate_rows),
            duplicate_rows_collapsed=max(routed_row_count - len(candidate_rows), 0),
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
    log_stage10_selection_debug(
        log,
        candidate_rows=candidate_rows,
        supplemental_candidate_rows=supplemental_candidate_rows,
        cached_rows=cached_rows,
        supplemental_cached_rows=supplemental_cached_rows,
        fresh_candidates=fresh_candidates,
        rows_to_process=rows_to_process,
        deferred=deferred,
    )

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
                input_hash = str(row["classification_input_hash"])
                result = future.result()
                completed += 1
                if result is None:
                    continue
                log_stage10_sample_response(log, completed=completed, row=row, result=result)
                store_cached_row(
                    conn,
                    input_hash=input_hash,
                    task_name=CLASSIFICATION_TASK_NAME,
                    model=result["model"],
                    prompt_version=CLASSIFICATION_PROMPT_VERSION,
                    response_json=result["response_json"],
                    tokens_used=result["tokens_used"],
                )
                cached_rows[input_hash] = fetch_cached_row(
                    conn,
                    input_hash=input_hash,
                    task_name=CLASSIFICATION_TASK_NAME,
                    prompt_version=CLASSIFICATION_PROMPT_VERSION,
                )
                if completed in progress_checkpoints:
                    log.info(
                        "Stage 10 progress | completed=%s/%s fresh classification tasks (%.1f%%)",
                        f"{completed:,}",
                        f"{len(rows_to_process):,}",
                        100.0 * completed / len(rows_to_process),
                    )

    tmp_results_path = prepare_temp_output(results_path)
    tmp_integrated_path = prepare_temp_output(integrated_path)

    try:
        results_rows = [
            build_results_row(row, str(row["classification_input_hash"]), cached_rows.get(str(row["classification_input_hash"])))
            for row in candidate_rows
        ]
        results_rows.extend(
            build_results_row(
                row,
                str(row["classification_input_hash"]),
                supplemental_cached_rows.get(str(row["classification_input_hash"])),
            )
            for row in supplemental_candidate_rows
            if str(row["classification_input_hash"]) in supplemental_cached_rows
        )
        write_parquet_rows(results_rows, tmp_results_path)

        writer = None
        try:
            fresh_hashes = {str(row["classification_input_hash"]) for row in rows_to_process}
            for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
                chunk = pa.Table.from_batches([batch]).to_pandas()
                prepared = prepare_classification_rows(chunk)
                batch_hashes = (
                    prepared.loc[prepared["classification_input_hash"].notna(), "classification_input_hash"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                batch_cache = fetch_cached_rows(
                    conn,
                    batch_hashes,
                    CLASSIFICATION_TASK_NAME,
                    CLASSIFICATION_PROMPT_VERSION,
                )
                integrated = integrate_chunk(
                    chunk,
                    batch_cache,
                    fresh_hashes=fresh_hashes,
                )
                out_table = pa.Table.from_pandas(integrated, preserve_index=False)
                if writer is None:
                    integrated_schema = _promote_null_fields(out_table.schema)
                    writer = pq.ParquetWriter(tmp_integrated_path, integrated_schema)
                out_table = out_table.cast(integrated_schema)
                writer.write_table(out_table)
        finally:
            if writer is not None:
                writer.close()
    except Exception:
        cleanup_temp_file(tmp_results_path)
        cleanup_temp_file(tmp_integrated_path)
        raise
    finally:
        conn.close()

    promote_temp_file(tmp_results_path, results_path)
    promote_temp_file(tmp_integrated_path, integrated_path)
    if compat_output_path is not None:
        shutil.copy2(integrated_path, compat_output_path)

    elapsed = time.time() - t0
    log.info("Stage 10 complete in %.1fs", elapsed)
    log.info("Posting-level integrated artifact written to %s", integrated_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 10 LLM classification and integration")
    parser.add_argument(
        "--llm-budget",
        type=int,
        required=True,
        help=(
            "Max new LLM calls (REQUIRED, no default). "
            "Use 0 for cache-only. Counts unique classification tasks after "
            "Stage 9 frame inheritance and cache reuse."
        ),
    )
    parser.add_argument(
        "--llm-budget-split",
        type=str,
        default=DEFAULT_BUDGET_SPLIT,
        help=(
            "Fractional split of the budget across categories: "
            "swe_combined,control. Default '0.7,0.3'. Values are "
            "normalized to sum to 1.0."
        ),
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--integrated-output", type=Path, default=DEFAULT_INTEGRATED_PATH)
    parser.add_argument("--compat-output", type=Path, default=DEFAULT_COMPAT_OUTPUT_PATH)
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
    run_stage10(
        llm_budget=args.llm_budget,
        llm_budget_split=parse_budget_split(args.llm_budget_split),
        input_path=args.input,
        results_path=args.results_output,
        integrated_path=args.integrated_output,
        compat_output_path=args.compat_output,
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
