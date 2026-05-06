#!/usr/bin/env python3
"""
Stage 12: LLM multi-label classification on two axes.

Adds two new posting-level columns to the post-Stage-11 parquet:
  - skill_themes:  list[str] of length 0..8  from an 8-value enum
  - role_families: list[str] of length 0..17 from a 17-value enum

Runs on rows where Stage 10 said is_swe_combined_llm = TRUE. Control rows and
non-frame rows are passed through with NULL columns. Each eligible row is
classified by 3 independent reps on gpt-5.4-mini; the per-axis label is
positive if it appears in >= 2 of the successful reps. Per-rep results are
cached in SQLite keyed by (description_hash, prompt_version, model, rep_index).

Input:
  - preprocessing/intermediate/stage11_embeddings_integrated.parquet

Output:
  - preprocessing/intermediate/stage12_llm_classified.parquet

The output is row-preserving and adds two columns. The next stage
(stage_final_output.py) reads from this file.

Frozen prompt: stage12_classify_axes_prompt_v1.md (sibling of this file).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sqlite3
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from io_utils import (
    cleanup_temp_file,
    prepare_temp_output,
    promote_null_schema,
    promote_temp_file,
)
from llm_shared import (
    DEFAULT_QUOTA_WAIT_HOURS,
    OPENAI_RESPONSES_API_URL,
    build_openai_headers,
    parse_openai_response,
    summarize_openai_error,
)


# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "preprocessing" / "scripts"
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
CACHE_DIR = PROJECT_ROOT / "preprocessing" / "cache"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

DEFAULT_INPUT_PATH = INTERMEDIATE_DIR / "stage11_embeddings_integrated.parquet"
DEFAULT_OUTPUT_PATH = INTERMEDIATE_DIR / "stage12_llm_classified.parquet"
DEFAULT_CACHE_DB = CACHE_DIR / "llm_classify_axes.db"
DEFAULT_LOG_PATH = LOG_DIR / "stage12_classify_axes.log"
DEFAULT_ERROR_LOG_PATH = LOG_DIR / "stage12_classify_axes_errors.jsonl"

PROMPT_PATH = SCRIPTS_DIR / "stage12_classify_axes_prompt_v1.md"
PROMPT_VERSION_TAG = "combined_v1"  # bump suffix when the prompt file changes

# Pipeline conventions
CHUNK_SIZE = 50_000
TASK_NAME = "skill_role_classification"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REPS = 3
DEFAULT_MAX_WORKERS = 15
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_5XX_RETRIES = 5
# Hardcoded, not a CLI knob. Baked into prompt_version below so any change here
# automatically invalidates the cache.
REASONING_EFFORT = "medium"
MIN_DESCRIPTION_WORDS = 15  # matches stages 9/10

SKILL_THEMES_COL = "skill_themes"
ROLE_FAMILIES_COL = "role_families"


# Enum vocab — single source of truth for the runtime-side schema. The prompt
# file carries the same values; if you edit one, edit both and bump the
# prompt-version suffix so the cache invalidates.
SKILL_THEMES = [
    "people_management",
    "orchestration",
    "verification",
    "mentorship",
    "performance",
    "process_scaffolding",
    "legacy_stack",
    "context_infrastructure",
]

ROLE_FAMILIES = [
    "software_engineer_general",
    "frontend_web",
    "backend_api",
    "mobile",
    "embedded",
    "data_engineer",
    "ml_engineer",
    "ai_llm_engineer",
    "devops_sre_platform",
    "security",
    "qa_test",
    "solutions_field",
    "legacy_specialist",
    "data_analytics",
    "research",
    "infra_ops_admin",
    "people_manager",
]


def build_structured_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            SKILL_THEMES_COL: {
                "type": "array",
                "items": {"type": "string", "enum": list(SKILL_THEMES)},
            },
            ROLE_FAMILIES_COL: {
                "type": "array",
                "items": {"type": "string", "enum": list(ROLE_FAMILIES)},
            },
        },
        "required": [SKILL_THEMES_COL, ROLE_FAMILIES_COL],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


_SYSTEM_BLOCK_RE = re.compile(
    r"^## System message\s*\n+```(?:[a-zA-Z0-9_+-]*)?\n(?P<body>.*?)\n```",
    re.DOTALL | re.MULTILINE,
)


def load_system_prompt(prompt_path: Path = PROMPT_PATH) -> tuple[str, str]:
    """Return (system_prompt, prompt_version).

    The system prompt is the verbatim contents of the first fenced block under
    the "## System message" heading in the prompt markdown file. The prompt
    version embeds a 12-char hash of the file contents and the hardcoded
    REASONING_EFFORT, so editing either invalidates the cache automatically.
    """
    text = prompt_path.read_text(encoding="utf-8")
    match = _SYSTEM_BLOCK_RE.search(text)
    if not match:
        raise RuntimeError(
            f"Could not locate '## System message' fenced block in {prompt_path}"
        )
    system_prompt = match.group("body").rstrip("\n")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    prompt_version = f"{PROMPT_VERSION_TAG}:{digest}:effort_{REASONING_EFFORT}"
    return system_prompt, prompt_version


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def configure_logging(log_path: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def normalize_for_hash(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def compute_input_hash(title: object, description_core_llm: object) -> str:
    """Hash over (title, description_core_llm) — identifies the row's input.

    Cache key is (this_hash, prompt_version, model, rep_index).
    """
    h = hashlib.sha256()
    h.update(normalize_for_hash(title).encode("utf-8"))
    h.update(b"\n--\n")
    h.update(normalize_for_hash(description_core_llm).encode("utf-8"))
    return h.hexdigest()


def description_word_count(description_core_llm: object) -> int:
    if description_core_llm is None:
        return 0
    text = str(description_core_llm).strip()
    if not text:
        return 0
    return len(text.split())


def build_user_message(title: object, description_core_llm: object) -> str:
    title_str = (title or "").strip() if isinstance(title, str) else ""
    desc_str = (description_core_llm or "") if isinstance(description_core_llm, str) else ""
    return f"Title: {title_str}\nDescription: {desc_str}"


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class ClassificationCache:
    """Per-rep cache for Stage 12 LLM classifications.

    Schema: one row per (description_hash, prompt_version, model, rep_index).
    Failures are NOT cached, so reruns naturally retry failed reps.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS classifications (
                description_hash TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                model TEXT NOT NULL,
                rep_index INTEGER NOT NULL,
                skill_themes_json TEXT NOT NULL,
                role_families_json TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                latency_ms INTEGER,
                request_id TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (description_hash, prompt_version, model, rep_index)
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_classifications_lookup
            ON classifications (prompt_version, model, description_hash, rep_index)
            """
        )
        self._conn.commit()

    def fetch_for_hashes(
        self,
        hashes: Iterable[str],
        *,
        prompt_version: str,
        model: str,
    ) -> dict[tuple[str, int], tuple[list[str], list[str]]]:
        """Return {(hash, rep_index): (skill_themes, role_families)} for cached reps."""
        unique = sorted(set(hashes))
        if not unique:
            return {}
        out: dict[tuple[str, int], tuple[list[str], list[str]]] = {}
        # SQLite default param limit is 999 (or 32766 since 3.32). Stay safe at 800.
        with self._lock:
            for start in range(0, len(unique), 800):
                batch = unique[start : start + 800]
                placeholders = ",".join("?" for _ in batch)
                rows = self._conn.execute(
                    f"""
                    SELECT description_hash, rep_index, skill_themes_json, role_families_json
                    FROM classifications
                    WHERE prompt_version = ?
                      AND model = ?
                      AND description_hash IN ({placeholders})
                    """,
                    [prompt_version, model, *batch],
                ).fetchall()
                for desc_hash, rep_index, skills_json, roles_json in rows:
                    out[(desc_hash, int(rep_index))] = (
                        json.loads(skills_json),
                        json.loads(roles_json),
                    )
        return out

    def store(
        self,
        *,
        description_hash: str,
        prompt_version: str,
        model: str,
        rep_index: int,
        skill_themes: list[str],
        role_families: list[str],
        input_tokens: int | None,
        output_tokens: int | None,
        latency_ms: int | None,
        request_id: str | None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO classifications (
                    description_hash, prompt_version, model, rep_index,
                    skill_themes_json, role_families_json,
                    input_tokens, output_tokens, latency_ms, request_id,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    description_hash,
                    prompt_version,
                    model,
                    rep_index,
                    json.dumps(skill_themes, ensure_ascii=False),
                    json.dumps(role_families, ensure_ascii=False),
                    input_tokens,
                    output_tokens,
                    latency_ms,
                    request_id,
                    now,
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Quota pause + budget
# ---------------------------------------------------------------------------


class QuotaPause:
    """Process-wide pause shared across worker threads.

    On 429/quota-exhausted responses, workers call `pause(seconds)`. All workers
    block in `wait()` until the pause expires.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paused_until = 0.0

    def pause(self, seconds: float) -> None:
        if seconds <= 0:
            return
        with self._lock:
            self._paused_until = max(self._paused_until, time.time() + seconds)

    def wait(self) -> None:
        while True:
            with self._lock:
                remaining = self._paused_until - time.time()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 30.0))


class BudgetCounter:
    """Thread-safe counter capping fresh API calls. None = unlimited."""

    def __init__(self, limit: int | None) -> None:
        self._lock = threading.Lock()
        self._used = 0
        self._limit = limit

    def try_acquire(self) -> bool:
        with self._lock:
            if self._limit is not None and self._used >= self._limit:
                return False
            self._used += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._used = max(0, self._used - 1)

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def limit(self) -> int | None:
        return self._limit


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------


class RetryableHttpError(RuntimeError):
    def __init__(self, message: str, *, retry_delay: float | None = None) -> None:
        super().__init__(message)
        self.retry_delay = retry_delay


class FatalHttpError(RuntimeError):
    pass


class QuotaExhaustedError(RuntimeError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _is_quota_exhausted(response: httpx.Response) -> bool:
    try:
        error = (response.json() or {}).get("error", {}) or {}
    except ValueError:
        error = {}
    haystack = " ".join(str(error.get(k, "")) for k in ("code", "type", "message")).lower()
    return any(token in haystack for token in ("insufficient_quota", "quota", "billing"))


def _parse_seconds(value: str | None) -> float | None:
    if not value:
        return None
    raw = value.strip().lower()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    total = 0.0
    matched = False
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", raw):
        matched = True
        amount = float(amount)
        if unit == "ms":
            total += amount / 1000.0
        elif unit == "s":
            total += amount
        elif unit == "m":
            total += amount * 60.0
        elif unit == "h":
            total += amount * 3600.0
    return total if matched else None


def _retry_after_from_headers(headers: httpx.Headers) -> float | None:
    candidates = [
        _parse_seconds(headers.get("retry-after")),
        _parse_seconds(headers.get("x-ratelimit-reset-requests")),
        _parse_seconds(headers.get("x-ratelimit-reset-tokens")),
    ]
    candidates = [c for c in candidates if c is not None]
    return max(candidates) if candidates else None


def call_openai_once(
    *,
    system_prompt: str,
    user_message: str,
    model: str,
    schema: dict,
    timeout_seconds: int,
    input_hash: str,
) -> tuple[dict, dict]:
    """Single API call. Returns (parsed_payload, meta).

    Raises RetryableHttpError, FatalHttpError, or QuotaExhaustedError.
    """
    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_message}],
            },
        ],
        "reasoning": {"effort": REASONING_EFFORT},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "skill_role_classification",
                "strict": True,
                "schema": schema,
            }
        },
    }
    headers = build_openai_headers(input_hash=input_hash, task_name=TASK_NAME)

    t0 = time.time()
    try:
        response = httpx.post(
            OPENAI_RESPONSES_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
        raise RetryableHttpError(f"network error: {exc}") from exc
    latency_ms = int((time.time() - t0) * 1000)

    if response.status_code != 200:
        detail = summarize_openai_error(response)
        if response.status_code == 429:
            if _is_quota_exhausted(response):
                raise QuotaExhaustedError(detail)
            raise RetryableHttpError(
                f"rate-limited (429): {detail[:300]}",
                retry_delay=_retry_after_from_headers(response.headers),
            )
        if response.status_code in {408, 500, 502, 503, 504}:
            raise RetryableHttpError(
                f"transient {response.status_code}: {detail[:300]}",
                retry_delay=_retry_after_from_headers(response.headers),
            )
        raise FatalHttpError(f"HTTP {response.status_code}: {detail[:600]}")

    json_text, tokens_used, _cost, response_model, request_id = parse_openai_response(response)
    parsed = json.loads(json_text)

    skills = parsed.get(SKILL_THEMES_COL)
    roles = parsed.get(ROLE_FAMILIES_COL)
    if not isinstance(skills, list) or not isinstance(roles, list):
        raise FatalHttpError(f"response missing arrays: {json_text[:300]}")

    # Strict-schema enforcement above means the response MUST be in-vocab; this
    # check is a defense-in-depth assertion in case OpenAI drops strict-mode
    # for some reason.
    bad = [v for v in skills if v not in SKILL_THEMES] + [v for v in roles if v not in ROLE_FAMILIES]
    if bad:
        raise FatalHttpError(f"out-of-vocab labels: {bad}")

    meta = {
        "input_tokens": None,
        "output_tokens": None,
        "tokens_used": tokens_used,
        "latency_ms": latency_ms,
        "request_id": request_id,
        "response_model": response_model,
    }
    # Pull split token counts when available.
    try:
        usage = response.json().get("usage", {}) or {}
        meta["input_tokens"] = usage.get("input_tokens")
        meta["output_tokens"] = usage.get("output_tokens")
    except ValueError:
        pass

    return parsed, meta


def call_openai_with_retry(
    *,
    system_prompt: str,
    user_message: str,
    model: str,
    schema: dict,
    timeout_seconds: int,
    max_retries: int,
    quota_pause: QuotaPause,
    quota_wait_seconds: float,
    input_hash: str,
    log: logging.Logger,
) -> tuple[dict, dict]:
    """Wrap call_openai_once with exponential-backoff retry on 5xx/network/429-rate.

    Quota-exhaustion (429 + insufficient_quota) raises after activating a
    process-wide pause for `quota_wait_seconds`. The caller should then re-
    enqueue the call.
    """
    attempt = 0
    while True:
        quota_pause.wait()
        try:
            return call_openai_once(
                system_prompt=system_prompt,
                user_message=user_message,
                model=model,
                schema=schema,
                timeout_seconds=timeout_seconds,
                input_hash=input_hash,
            )
        except QuotaExhaustedError as exc:
            quota_pause.pause(quota_wait_seconds)
            log.warning(
                "Quota exhausted; pausing %.0fs before retrying. detail=%s",
                quota_wait_seconds,
                str(exc)[:200],
            )
            # Don't count this as a retry attempt — the call hasn't really run.
            continue
        except RetryableHttpError as exc:
            if attempt >= max_retries:
                raise
            delay = exc.retry_delay
            if delay is None:
                delay = min(60.0, 2.0 ** attempt)
            delay *= random.uniform(0.75, 1.25)
            time.sleep(delay)
            attempt += 1
            continue


# ---------------------------------------------------------------------------
# Majority voting
# ---------------------------------------------------------------------------


def majority_vote(rep_results: list[tuple[list[str], list[str]] | None]) -> tuple[list[str] | None, list[str] | None]:
    """Return (skills_majority, roles_majority) or (None, None) if too sparse.

    Rule: a label is positive iff it appears in MORE THAN HALF of successful
    reps. With 3 reps, positive = appeared >= 2 times. Requires >= 2 successful
    reps; otherwise returns (None, None).
    """
    successful = [r for r in rep_results if r is not None]
    if len(successful) < 2:
        return None, None
    threshold = (len(successful) // 2) + 1  # > 50%

    skill_counter: Counter[str] = Counter()
    role_counter: Counter[str] = Counter()
    for skills, roles in successful:
        skill_counter.update(set(skills))
        role_counter.update(set(roles))

    skills_out = sorted(label for label, n in skill_counter.items() if n >= threshold)
    roles_out = sorted(label for label, n in role_counter.items() if n >= threshold)
    return skills_out, roles_out


# ---------------------------------------------------------------------------
# Per-row classification orchestration
# ---------------------------------------------------------------------------


def classify_row_reps(
    *,
    description_hash: str,
    title: object,
    description_core_llm: object,
    cached_reps: dict[int, tuple[list[str], list[str]]],
    reps: int,
    cache: ClassificationCache,
    prompt_version: str,
    model: str,
    system_prompt: str,
    schema: dict,
    timeout_seconds: int,
    max_retries: int,
    quota_pause: QuotaPause,
    quota_wait_seconds: float,
    budget: BudgetCounter,
    error_log_path: Path,
    log: logging.Logger,
) -> tuple[list[tuple[list[str], list[str]] | None], int, int]:
    """Resolve all reps for one row. Returns (rep_results, fresh_calls, deferred_reps)."""
    rep_results: list[tuple[list[str], list[str]] | None] = [None] * reps
    fresh_calls = 0
    deferred = 0
    user_message = build_user_message(title, description_core_llm)

    for rep_index in range(reps):
        cached = cached_reps.get(rep_index)
        if cached is not None:
            rep_results[rep_index] = cached
            continue

        if not budget.try_acquire():
            deferred += 1
            continue

        try:
            parsed, meta = call_openai_with_retry(
                system_prompt=system_prompt,
                user_message=user_message,
                model=model,
                schema=schema,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                quota_pause=quota_pause,
                quota_wait_seconds=quota_wait_seconds,
                input_hash=f"{description_hash}:{rep_index}",
                log=log,
            )
        except (RetryableHttpError, FatalHttpError) as exc:
            budget.release()
            append_jsonl(
                error_log_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "description_hash": description_hash,
                    "rep_index": rep_index,
                    "model": model,
                    "prompt_version": prompt_version,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1500],
                },
            )
            log.warning(
                "Stage 12 rep failed | hash=%s rep=%d model=%s err=%s",
                description_hash[:12],
                rep_index,
                model,
                type(exc).__name__,
            )
            continue

        skills = list(parsed[SKILL_THEMES_COL])
        roles = list(parsed[ROLE_FAMILIES_COL])
        cache.store(
            description_hash=description_hash,
            prompt_version=prompt_version,
            model=model,
            rep_index=rep_index,
            skill_themes=skills,
            role_families=roles,
            input_tokens=meta.get("input_tokens"),
            output_tokens=meta.get("output_tokens"),
            latency_ms=meta.get("latency_ms"),
            request_id=meta.get("request_id"),
        )
        rep_results[rep_index] = (skills, roles)
        fresh_calls += 1

    return rep_results, fresh_calls, deferred


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


def _is_eligible_row(
    is_swe: object,
    description_core_llm: object,
) -> bool:
    if is_swe is not True:
        return False
    if description_word_count(description_core_llm) < MIN_DESCRIPTION_WORDS:
        return False
    return True


def run_stage12(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    cache_db: Path = DEFAULT_CACHE_DB,
    log_path: Path = DEFAULT_LOG_PATH,
    error_log_path: Path = DEFAULT_ERROR_LOG_PATH,
    model: str = DEFAULT_MODEL,
    reps: int = DEFAULT_REPS,
    max_workers: int = DEFAULT_MAX_WORKERS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_5XX_RETRIES,
    llm_budget: int | None = None,
    quota_wait_hours: float = DEFAULT_QUOTA_WAIT_HOURS,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    log = configure_logging(log_path)
    log.info("=" * 70)
    log.info("Stage 12: LLM multi-label classification (skill themes + role families)")
    log.info("=" * 70)
    log.info("Input:  %s", input_path)
    log.info("Output: %s", output_path)
    log.info("Cache:  %s", cache_db)
    log.info(
        "Config | model=%s reasoning_effort=%s reps=%d workers=%d budget=%s quota_wait=%.1fmin",
        model,
        REASONING_EFFORT,
        reps,
        max_workers,
        "unlimited" if llm_budget is None else f"{llm_budget:,}",
        quota_wait_hours * 60.0,
    )

    system_prompt, prompt_version = load_system_prompt()
    log.info("Prompt version: %s", prompt_version)
    schema = build_structured_schema()

    pf = pq.ParquetFile(input_path)
    input_columns = set(pf.schema.names)
    required = {"title", "description_core_llm", "is_swe_combined_llm"}
    missing = required - input_columns
    if missing:
        raise ValueError(f"Stage 12 input is missing required columns: {sorted(missing)}")

    cache = ClassificationCache(cache_db)
    quota_pause = QuotaPause()
    quota_wait_seconds = max(quota_wait_hours * 3600.0, 0.0)
    budget = BudgetCounter(llm_budget)

    tmp_output_path = prepare_temp_output(output_path)
    writer = None
    output_schema = None

    total_rows = 0
    eligible_rows = 0
    short_skipped = 0
    not_swe = 0
    classified_rows = 0
    null_rows = 0
    deferred_rows = 0
    fresh_calls_total = 0
    cache_hits = 0

    try:
        for batch_idx, record_batch in enumerate(pf.iter_batches(batch_size=chunk_size)):
            table = pa.Table.from_batches([record_batch])
            # Drop any pre-existing copies so re-runs cleanly overwrite.
            for col in (SKILL_THEMES_COL, ROLE_FAMILIES_COL):
                if col in table.column_names:
                    table = table.drop([col])

            titles = table.column("title").to_pylist()
            descriptions = table.column("description_core_llm").to_pylist()
            is_swe_flags = table.column("is_swe_combined_llm").to_pylist()

            n_rows = len(titles)
            row_eligible = [False] * n_rows
            row_hashes: list[str | None] = [None] * n_rows
            for i in range(n_rows):
                if is_swe_flags[i] is not True:
                    not_swe += 1
                    continue
                if description_word_count(descriptions[i]) < MIN_DESCRIPTION_WORDS:
                    short_skipped += 1
                    continue
                row_eligible[i] = True
                row_hashes[i] = compute_input_hash(titles[i], descriptions[i])

            eligible_indices = [i for i in range(n_rows) if row_eligible[i]]
            chunk_eligible = len(eligible_indices)
            eligible_rows += chunk_eligible

            # Bulk fetch any cached reps for this chunk.
            unique_hashes = {row_hashes[i] for i in eligible_indices if row_hashes[i] is not None}
            cached_map = cache.fetch_for_hashes(
                unique_hashes,
                prompt_version=prompt_version,
                model=model,
            )
            chunk_cache_hits = len(cached_map)
            cache_hits += chunk_cache_hits

            log.info(
                "Stage 12 chunk %d | rows=%s eligible=%s cached_reps=%s budget_used=%s",
                batch_idx,
                f"{n_rows:,}",
                f"{chunk_eligible:,}",
                f"{chunk_cache_hits:,}",
                f"{budget.used:,}",
            )

            # Resolve reps in parallel across rows in this chunk.
            row_outcomes: list[tuple[list[str] | None, list[str] | None]] = [(None, None)] * n_rows

            def _resolve(idx: int) -> tuple[int, int, int, list[tuple[list[str], list[str]] | None]]:
                desc_hash = row_hashes[idx]
                cached_for_row = {
                    rep_index: payload
                    for (h, rep_index), payload in cached_map.items()
                    if h == desc_hash
                }
                rep_results, fresh, deferred = classify_row_reps(
                    description_hash=desc_hash,
                    title=titles[idx],
                    description_core_llm=descriptions[idx],
                    cached_reps=cached_for_row,
                    reps=reps,
                    cache=cache,
                    prompt_version=prompt_version,
                    model=model,
                    system_prompt=system_prompt,
                    schema=schema,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    quota_pause=quota_pause,
                    quota_wait_seconds=quota_wait_seconds,
                    budget=budget,
                    error_log_path=error_log_path,
                    log=log,
                )
                return idx, fresh, deferred, rep_results

            if eligible_indices:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = [pool.submit(_resolve, idx) for idx in eligible_indices]
                    completed = 0
                    for fut in as_completed(futures):
                        idx, fresh, deferred_count, rep_results = fut.result()
                        fresh_calls_total += fresh
                        if deferred_count > 0:
                            deferred_rows += 1
                        skills, roles = majority_vote(rep_results)
                        if skills is None or roles is None:
                            null_rows += 1
                        else:
                            classified_rows += 1
                        row_outcomes[idx] = (skills, roles)
                        completed += 1
                        if completed % 1000 == 0 or completed == len(eligible_indices):
                            log.info(
                                "  chunk progress: %s/%s rows resolved | fresh_calls=%s budget_used=%s",
                                f"{completed:,}",
                                f"{len(eligible_indices):,}",
                                f"{fresh_calls_total:,}",
                                f"{budget.used:,}",
                            )

            skills_col = pa.array(
                [outcome[0] for outcome in row_outcomes],
                type=pa.list_(pa.string()),
            )
            roles_col = pa.array(
                [outcome[1] for outcome in row_outcomes],
                type=pa.list_(pa.string()),
            )
            out_table = (
                table.append_column(SKILL_THEMES_COL, skills_col)
                .append_column(ROLE_FAMILIES_COL, roles_col)
            )

            if writer is None:
                output_schema = promote_null_schema(out_table.schema)
                writer = pq.ParquetWriter(tmp_output_path, output_schema, compression="zstd")
            writer.write_table(out_table.cast(output_schema))
            total_rows += n_rows

        if writer is not None:
            writer.close()
            writer = None
        promote_temp_file(tmp_output_path, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        cleanup_temp_file(tmp_output_path)
        raise
    finally:
        cache.close()

    log.info("-" * 70)
    log.info(
        "Stage 12 complete | total_rows=%s | eligible=%s | classified=%s | null=%s | deferred_rows=%s",
        f"{total_rows:,}",
        f"{eligible_rows:,}",
        f"{classified_rows:,}",
        f"{null_rows:,}",
        f"{deferred_rows:,}",
    )
    log.info(
        "Filter breakdown   | not_swe=%s | short_description=%s",
        f"{not_swe:,}",
        f"{short_skipped:,}",
    )
    log.info(
        "API usage          | cache_hits=%s | fresh_calls=%s | budget_limit=%s",
        f"{cache_hits:,}",
        f"{fresh_calls_total:,}",
        "unlimited" if llm_budget is None else f"{llm_budget:,}",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 12 LLM multi-label classification")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--cache-db", type=Path, default=DEFAULT_CACHE_DB)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--error-log-path", type=Path, default=DEFAULT_ERROR_LOG_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_5XX_RETRIES,
                        help="Max retries on 5xx / network errors per call (excluding quota pauses)")
    parser.add_argument("--llm-budget", type=int, default=None,
                        help="Cap on fresh API calls. None = unlimited. Cached reps are free.")
    parser.add_argument("--quota-wait-hours", type=float, default=DEFAULT_QUOTA_WAIT_HOURS,
                        help="Pause duration on quota exhaustion (default 1 minute)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stage12(
        input_path=args.input,
        output_path=args.output,
        cache_db=args.cache_db,
        log_path=args.log_path,
        error_log_path=args.error_log_path,
        model=args.model,
        reps=args.reps,
        max_workers=args.max_workers,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        llm_budget=args.llm_budget,
        quota_wait_hours=args.quota_wait_hours,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
