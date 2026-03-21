#!/usr/bin/env python3
"""
Stage 12: Three-way validation report.

Compares:
  - rule-based outputs
  - GPT-5.4 mini outputs from Stage 10 / Stage 11
  - GPT-5.4 full outputs on a stratified validation sample
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stage10_llm_classify import (
    CLASSIFICATION_PROMPT_VERSION,
    CLASSIFICATION_TASK_NAME,
    EXTRACTION_PROMPT_VERSION,
    EXTRACTION_TASK_NAME,
    fetch_cached_row,
    open_cache,
    render_classification_prompt,
    render_extraction_prompt,
    store_cached_row,
    try_provider,
    validate_classification_payload,
    validate_extraction_payload,
)
from stage11_llm_integrate import fuzzy_similarity, validate_extraction_payload as validate_extraction_selection


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "preprocessing" / "cache"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

DEFAULT_INPUT_PATH = DATA_DIR / "unified.parquet"
DEFAULT_OUTPUT_PATH = LOG_DIR / "validation_report_v2.md"
DEFAULT_FULL_CACHE_DB = CACHE_DIR / "llm_validation_full.db"
DEFAULT_ERROR_LOG = LOG_DIR / "stage12_validation_errors.jsonl"

RANDOM_SEED = 42


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "stage12_validation.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def map_rule_swe(row: pd.Series) -> str:
    if bool(row.get("is_swe")):
        return "SWE"
    if bool(row.get("is_swe_adjacent")):
        return "SWE_ADJACENT"
    return "NOT_SWE"


def map_rule_seniority(row: pd.Series) -> str:
    value = str(row.get("seniority_final") or "").strip().lower()
    mapping = {
        "entry": "entry",
        "entry level": "entry",
        "intern": "entry",
        "internship": "entry",
        "associate": "associate",
        "mid-senior": "mid-senior",
        "mid-senior level": "mid-senior",
        "director": "director",
        "executive": "director",
        "unknown": "unknown",
    }
    return mapping.get(value, "unknown")


def map_rule_ghost(row: pd.Series) -> str:
    risk = str(row.get("ghost_job_risk") or "").strip().lower()
    if risk == "low":
        return "realistic"
    if risk == "high":
        return "ghost_likely"
    if risk == "medium":
        return "inflated"
    return "realistic"


def disagreement_flag(row: pd.Series) -> bool:
    mini_swe = row.get("swe_classification_llm")
    mini_seniority = row.get("seniority_llm")
    mini_ghost = row.get("ghost_assessment_llm")
    mini_boilerplate = row.get("description_core_llm")
    if pd.isna(mini_swe) and pd.isna(mini_seniority) and pd.isna(mini_ghost) and pd.isna(mini_boilerplate):
        return False
    return any(
        [
            map_rule_swe(row) != mini_swe,
            map_rule_seniority(row) != mini_seniority,
            map_rule_ghost(row) != mini_ghost,
            fuzzy_similarity(row.get("description_core") or "", mini_boilerplate or "") < 0.99,
        ]
    )


def stratified_sample(df: pd.DataFrame, sample_size: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    rng = random.Random(seed)
    frame = df.copy()
    frame["_disagree"] = frame.apply(disagreement_flag, axis=1)
    frame["_source_group"] = (
        frame["source"].fillna("unknown").astype(str)
        + " | "
        + frame["source_platform"].fillna("unknown").astype(str)
    )

    samples = []
    sources = sorted(frame["_source_group"].dropna().unique().tolist())
    per_source = max(sample_size // max(len(sources), 1), 1)

    for source in sources:
        src_df = frame[frame["_source_group"] == source]
        disagree_df = src_df[src_df["_disagree"]]
        agree_df = src_df[~src_df["_disagree"]]

        n_disagree = min(len(disagree_df), max(per_source // 2, 1))
        n_total = min(len(src_df), per_source)
        n_agree = min(len(agree_df), max(n_total - n_disagree, 0))

        if n_disagree:
            disagree_idx = rng.sample(disagree_df.index.tolist(), n_disagree)
            samples.append(frame.loc[disagree_idx])
        if n_agree:
            agree_idx = rng.sample(agree_df.index.tolist(), n_agree)
            samples.append(frame.loc[agree_idx])

    sampled = pd.concat(samples) if samples else frame.head(0).copy()
    sampled = sampled.loc[~sampled.index.duplicated(keep="first")]
    if len(sampled) > sample_size:
        chosen = rng.sample(sampled.index.tolist(), sample_size)
        sampled = sampled.loc[chosen]
    elif len(sampled) < sample_size:
        remaining = frame.loc[~frame.index.isin(sampled.index)]
        need = min(sample_size - len(sampled), len(remaining))
        if need:
            extra_idx = rng.sample(remaining.index.tolist(), need)
            sampled = pd.concat([sampled, frame.loc[extra_idx]])
            sampled = sampled.loc[~sampled.index.duplicated(keep="first")]

    sampled = sampled.drop(columns=["_disagree", "_source_group"], errors="ignore")
    sampled = sampled.drop_duplicates(subset=["job_id"])
    if len(sampled) > sample_size:
        chosen = rng.sample(sampled.index.tolist(), sample_size)
        sampled = sampled.loc[chosen]
    return sampled.reset_index(drop=True)


def run_full_model(
    sample_df: pd.DataFrame,
    cache_db: Path,
    error_log_path: Path,
    log: logging.Logger,
    timeout_seconds: int,
    max_retries: int,
) -> pd.DataFrame:
    conn = open_cache(cache_db)
    results = []

    for row in sample_df.itertuples(index=False):
        description_hash = str(row.description_hash)

        classification_cached = fetch_cached_row(
            conn,
            description_hash,
            CLASSIFICATION_TASK_NAME,
            CLASSIFICATION_PROMPT_VERSION,
        )
        extraction_cached = fetch_cached_row(
            conn,
            description_hash,
            EXTRACTION_TASK_NAME,
            EXTRACTION_PROMPT_VERSION,
        )

        if classification_cached is None:
            classification_prompt = render_classification_prompt(
                row.title,
                row.company_name,
                row.description,
            )
            classification_result = try_provider(
                provider="codex",
                prompt=classification_prompt,
                model="gpt-5.4",
                task_name=CLASSIFICATION_TASK_NAME,
                description_hash=description_hash,
                error_log_path=error_log_path,
                log=log,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                payload_validator=validate_classification_payload,
            )
            if classification_result is None:
                append_jsonl(
                    error_log_path,
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "job_id": row.job_id,
                        "description_hash": description_hash,
                        "task_name": CLASSIFICATION_TASK_NAME,
                        "error_type": "validation_full_model_failed",
                    },
                )
                continue
            store_cached_row(
                conn,
                description_hash=description_hash,
                task_name=CLASSIFICATION_TASK_NAME,
                model=classification_result["model"],
                prompt_version=CLASSIFICATION_PROMPT_VERSION,
                response_json=classification_result["response_json"],
                tokens_used=classification_result["tokens_used"],
            )
            classification_cached = fetch_cached_row(
                conn,
                description_hash,
                CLASSIFICATION_TASK_NAME,
                CLASSIFICATION_PROMPT_VERSION,
            )

        if extraction_cached is None:
            extraction_prompt, _ = render_extraction_prompt(
                row.title,
                row.company_name,
                row.description,
            )
            extraction_result = try_provider(
                provider="codex",
                prompt=extraction_prompt,
                model="gpt-5.4",
                task_name=EXTRACTION_TASK_NAME,
                description_hash=description_hash,
                error_log_path=error_log_path,
                log=log,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                payload_validator=validate_extraction_payload,
            )
            if extraction_result is None:
                append_jsonl(
                    error_log_path,
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "job_id": row.job_id,
                        "description_hash": description_hash,
                        "task_name": EXTRACTION_TASK_NAME,
                        "error_type": "validation_full_model_failed",
                    },
                )
                continue
            store_cached_row(
                conn,
                description_hash=description_hash,
                task_name=EXTRACTION_TASK_NAME,
                model=extraction_result["model"],
                prompt_version=EXTRACTION_PROMPT_VERSION,
                response_json=extraction_result["response_json"],
                tokens_used=extraction_result["tokens_used"],
            )
            extraction_cached = fetch_cached_row(
                conn,
                description_hash,
                EXTRACTION_TASK_NAME,
                EXTRACTION_PROMPT_VERSION,
            )

        classification_payload = json.loads(classification_cached["response_json"])
        extraction_payload = json.loads(extraction_cached["response_json"])
        extraction_validation = validate_extraction_selection(
            "" if row.description is None else str(row.description),
            extraction_payload,
        )

        results.append(
            {
                "job_id": row.job_id,
                "full_model_classification": classification_cached["model"],
                "full_model_extraction": extraction_cached["model"],
                "swe_classification_full": classification_payload.get("swe_classification"),
                "seniority_full": classification_payload.get("seniority"),
                "ghost_assessment_full": classification_payload.get("ghost_assessment"),
                "description_core_full": (
                    extraction_validation["reconstructed_text"] if extraction_validation["passed"] else None
                ),
                "description_core_full_similarity": extraction_validation["similarity"],
                "description_core_full_validated": extraction_validation["passed"],
            }
        )

    conn.close()
    return pd.DataFrame(results)


def pairwise_labels(task: str, df: pd.DataFrame, left: str, right: str) -> tuple[list[str], list[str]]:
    left_vals = []
    right_vals = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        if task == "swe":
            mapping = {
                "rules": map_rule_swe(pd.Series(row_dict)),
                "mini": row.swe_classification_llm,
                "full": row.swe_classification_full,
            }
        elif task == "seniority":
            mapping = {
                "rules": map_rule_seniority(pd.Series(row_dict)),
                "mini": row.seniority_llm,
                "full": row.seniority_full,
            }
        elif task == "ghost":
            mapping = {
                "rules": map_rule_ghost(pd.Series(row_dict)),
                "mini": row.ghost_assessment_llm,
                "full": row.ghost_assessment_full,
            }
        else:
            raise ValueError(task)

        if mapping[left] is None or mapping[right] is None or pd.isna(mapping[left]) or pd.isna(mapping[right]):
            continue
        left_vals.append(mapping[left])
        right_vals.append(mapping[right])
    return left_vals, right_vals


def cohen_kappa(left_vals: list[str], right_vals: list[str]) -> float | None:
    n = len(left_vals)
    if n == 0:
        return None
    labels = sorted(set(left_vals) | set(right_vals))
    left_counts = Counter(left_vals)
    right_counts = Counter(right_vals)
    agree = sum(1 for left, right in zip(left_vals, right_vals) if left == right)
    po = agree / n
    pe = sum((left_counts[label] / n) * (right_counts[label] / n) for label in labels)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def kappa_label(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value > 0.80:
        return "strong"
    if value >= 0.60:
        return "moderate"
    return "weak"


def contingency_table(left_vals: list[str], right_vals: list[str]) -> pd.DataFrame:
    return pd.crosstab(
        pd.Series(left_vals, name="left"),
        pd.Series(right_vals, name="right"),
    )


def preview_text(value, limit: int = 250) -> str:
    if pd.isna(value):
        return ""
    return str(value)[:limit]


def boilerplate_summary(df: pd.DataFrame, left: str, right: str) -> dict:
    comparisons = 0
    exact = 0
    near = 0
    mean_similarity = []

    for row in df.itertuples(index=False):
        mapping = {
            "rules": row.description_core,
            "mini": row.description_core_llm,
            "full": row.description_core_full,
        }
        left_text = mapping[left]
        right_text = mapping[right]
        if left_text is None or right_text is None or pd.isna(left_text) or pd.isna(right_text):
            continue
        comparisons += 1
        similarity = fuzzy_similarity(str(left_text), str(right_text))
        mean_similarity.append(similarity)
        if similarity == 1.0:
            exact += 1
        if similarity >= 0.99:
            near += 1

    return {
        "comparisons": comparisons,
        "exact_match_rate": 0.0 if comparisons == 0 else exact / comparisons,
        "near_match_rate": 0.0 if comparisons == 0 else near / comparisons,
        "mean_similarity": None if not mean_similarity else sum(mean_similarity) / len(mean_similarity),
    }


def categorize_disagreement(task: str, row: pd.Series) -> str:
    if task == "swe":
        values = {map_rule_swe(row), row["swe_classification_llm"], row["swe_classification_full"]}
        if values <= {"SWE", "SWE_ADJACENT"}:
            return "definitional"
        if row["swe_classification_llm"] == row["swe_classification_full"] != map_rule_swe(row):
            return "misclassification"
        if map_rule_swe(row) == row["swe_classification_full"] != row["swe_classification_llm"]:
            return "misclassification"
        return "edge_case"

    if task == "seniority":
        values = {map_rule_seniority(row), row["seniority_llm"], row["seniority_full"]}
        if "unknown" in values:
            return "edge_case"
        if row["seniority_llm"] == row["seniority_full"] != map_rule_seniority(row):
            return "misclassification"
        if map_rule_seniority(row) == row["seniority_full"] != row["seniority_llm"]:
            return "misclassification"
        return "definitional"

    if task == "boilerplate":
        sim_rules_full = fuzzy_similarity(row["description_core"], row["description_core_full"])
        sim_mini_full = fuzzy_similarity(row["description_core_llm"], row["description_core_full"])
        if sim_mini_full >= 0.99 and sim_rules_full < 0.80:
            return "misclassification"
        if sim_rules_full >= 0.99 and sim_mini_full < 0.80:
            return "misclassification"
        if max(sim_rules_full, sim_mini_full) >= 0.80:
            return "definitional"
        return "edge_case"

    if task == "ghost":
        values = {map_rule_ghost(row), row["ghost_assessment_llm"], row["ghost_assessment_full"]}
        if values <= {"realistic", "inflated"}:
            return "definitional"
        if row["ghost_assessment_llm"] == row["ghost_assessment_full"] != map_rule_ghost(row):
            return "misclassification"
        if map_rule_ghost(row) == row["ghost_assessment_full"] != row["ghost_assessment_llm"]:
            return "misclassification"
        return "edge_case"

    raise ValueError(task)


def collect_examples(task: str, df: pd.DataFrame, limit_per_category: int = 5) -> dict[str, list[dict]]:
    examples = defaultdict(list)
    for _, row in df.iterrows():
        if task == "swe":
            disagrees = len({map_rule_swe(row), row["swe_classification_llm"], row["swe_classification_full"]}) > 1
        elif task == "seniority":
            disagrees = len({map_rule_seniority(row), row["seniority_llm"], row["seniority_full"]}) > 1
        elif task == "boilerplate":
            sims = [
                fuzzy_similarity(row["description_core"], row["description_core_llm"]),
                fuzzy_similarity(row["description_core"], row["description_core_full"]),
                fuzzy_similarity(row["description_core_llm"], row["description_core_full"]),
            ]
            disagrees = min(sims) < 0.99
        else:
            disagrees = len({map_rule_ghost(row), row["ghost_assessment_llm"], row["ghost_assessment_full"]}) > 1
        if not disagrees:
            continue

        category = categorize_disagreement(task, row)
        if len(examples[category]) >= limit_per_category:
            continue

        examples[category].append(
            {
                "job_id": row["job_id"],
                "source": row["source"],
                "source_platform": row["source_platform"],
                "title": row["title"],
                "company_name": row["company_name"],
                "description_preview": preview_text(row["description"], 500),
                "rules": {
                    "swe": map_rule_swe(row),
                    "seniority": map_rule_seniority(row),
                    "boilerplate": preview_text(row["description_core"], 250),
                    "ghost": map_rule_ghost(row),
                },
                "mini": {
                    "swe": row["swe_classification_llm"],
                    "seniority": row["seniority_llm"],
                    "boilerplate": preview_text(row["description_core_llm"], 250),
                    "ghost": row["ghost_assessment_llm"],
                },
                "full": {
                    "swe": row["swe_classification_full"],
                    "seniority": row["seniority_full"],
                    "boilerplate": preview_text(row["description_core_full"], 250),
                    "ghost": row["ghost_assessment_full"],
                },
            }
        )
    return examples


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No comparable rows._"
    headers = [""] + [str(col) for col in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(value) for value in row.tolist()]) + " |")
    return "\n".join(lines)


def build_report(df: pd.DataFrame, sample_size: int) -> str:
    lines = [
        "# Validation Report v2",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Sample size target: {sample_size}",
        f"Actual sampled rows: {len(df)}",
        "",
        "Sampling notes:",
        "- Stratified by `source | source_platform`.",
        "- Oversamples rows where rules and GPT-5.4 mini disagree on at least one task.",
        "- Seed = 42.",
        "",
    ]

    tasks = [
        ("swe", "SWE Classification"),
        ("seniority", "Seniority"),
        ("boilerplate", "Boilerplate"),
        ("ghost", "Ghost Job"),
    ]

    for task_key, task_title in tasks:
        lines.extend([f"## {task_title}", ""])
        if task_key != "boilerplate":
            for left, right in [("rules", "mini"), ("rules", "full"), ("mini", "full")]:
                left_vals, right_vals = pairwise_labels(task_key, df, left, right)
                ct = contingency_table(left_vals, right_vals)
                kappa = cohen_kappa(left_vals, right_vals)
                lines.append(f"### {left} vs {right}")
                lines.append("")
                lines.append(markdown_table(ct))
                lines.append("")
                lines.append(
                    f"Cohen's kappa: `{kappa:.3f}` ({kappa_label(kappa)})"
                    if kappa is not None
                    else "Cohen's kappa: `n/a`"
                )
                lines.append("")
        else:
            for left, right in [("rules", "mini"), ("rules", "full"), ("mini", "full")]:
                summary = boilerplate_summary(df, left, right)
                lines.append(f"### {left} vs {right}")
                lines.append("")
                lines.append(f"- Comparable rows: `{summary['comparisons']}`")
                lines.append(f"- Exact match rate: `{summary['exact_match_rate']:.3f}`")
                lines.append(f"- Fuzzy >= 0.99 match rate: `{summary['near_match_rate']:.3f}`")
                if summary["mean_similarity"] is None:
                    lines.append("- Mean similarity: `n/a`")
                else:
                    lines.append(f"- Mean similarity: `{summary['mean_similarity']:.3f}`")
                lines.append("")

        examples = collect_examples(task_key, df, limit_per_category=10)
        lines.append("### Disagreement examples")
        lines.append("")
        if not examples:
            lines.append("No disagreements in the sampled rows.")
            lines.append("")
            continue
        for category in ["definitional", "misclassification", "edge_case"]:
            lines.append(f"#### {category}")
            lines.append("")
            items = examples.get(category, [])
            if not items:
                lines.append("No examples sampled.")
                lines.append("")
                continue
            for item in items:
                lines.append(
                    f"- `{item['job_id']}` | {item['source']} | {item['source_platform']} | "
                    f"{item['title']} | {item['company_name']}"
                )
                lines.append(f"  Description preview: {item['description_preview']}")
                lines.append(
                    "  Rules / mini / full:"
                    f" SWE={item['rules']['swe']} / {item['mini']['swe']} / {item['full']['swe']};"
                    f" seniority={item['rules']['seniority']} / {item['mini']['seniority']} / {item['full']['seniority']};"
                    f" ghost={item['rules']['ghost']} / {item['mini']['ghost']} / {item['full']['ghost']}"
                )
                lines.append(
                    "  Boilerplate previews:"
                    f" rules=`{item['rules']['boilerplate']}`"
                    f" mini=`{item['mini']['boilerplate']}`"
                    f" full=`{item['full']['boilerplate']}`"
                )
            lines.append("")

    return "\n".join(lines)


def run_stage12(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    full_cache_db: Path = DEFAULT_FULL_CACHE_DB,
    error_log_path: Path = DEFAULT_ERROR_LOG,
    sample_size: int = 200,
    timeout_seconds: int = 240,
    max_retries: int = 2,
) -> None:
    log = configure_logging()
    t0 = time.time()

    log.info("=" * 70)
    log.info("Stage 12: Three-way validation")
    log.info("=" * 70)
    log.info("Input: %s", input_path)
    log.info("Output: %s", output_path)

    pf = pq.ParquetFile(input_path)
    frames = []
    cols = [
        "job_id",
        "source",
        "source_platform",
        "title",
        "company_name",
        "description",
        "description_hash",
        "description_core",
        "is_swe",
        "is_swe_adjacent",
        "seniority_final",
        "ghost_job_risk",
        "swe_classification_llm",
        "seniority_llm",
        "description_core_llm",
        "ghost_assessment_llm",
    ]
    for batch in pf.iter_batches(batch_size=100_000, columns=cols):
        frames.append(pa.Table.from_batches([batch]).to_pandas())
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    del frames
    gc.collect()

    candidate_df = df[
        df["swe_classification_llm"].notna()
        & df["seniority_llm"].notna()
        & df["ghost_assessment_llm"].notna()
    ].copy()
    sample_df = stratified_sample(candidate_df, sample_size=sample_size)
    log.info("Validation sample rows: %s", f"{len(sample_df):,}")

    full_df = run_full_model(
        sample_df=sample_df,
        cache_db=full_cache_db,
        error_log_path=error_log_path,
        log=log,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    merged = sample_df.merge(full_df, on="job_id", how="inner")

    report = build_report(merged, sample_size=sample_size)
    output_path.write_text(report, encoding="utf-8")

    elapsed_total = time.time() - t0
    log.info("Stage 12 complete in %.1fs", elapsed_total)
    log.info("Validation report written to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 12 validation")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--full-cache-db", type=Path, default=DEFAULT_FULL_CACHE_DB)
    parser.add_argument("--error-log", type=Path, default=DEFAULT_ERROR_LOG)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage12(
        input_path=args.input,
        output_path=args.output,
        full_cache_db=args.full_cache_db,
        error_log_path=args.error_log,
        sample_size=args.sample_size,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
