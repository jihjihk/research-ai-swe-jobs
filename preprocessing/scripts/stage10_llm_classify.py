#!/usr/bin/env python3
"""
Stage 10: LLM extraction + classification.

This stage performs two separate cached tasks per unique description hash:
  1. Core-content extraction by numbered sentence-like units
  2. SWE / seniority / ghost classification

Architectural contract:
  - Stage 10 may deduplicate LLM calls by `description_hash` to reduce volume.
  - Stage 10 does NOT deduplicate analytical rows or postings.
  - The Stage 10 results parquet is one row per unique description hash.
  - Stage 11 is responsible for reattaching those cached outputs to every
    matching row in the full posting table.

Provider defaults:
  - Codex path: `gpt-5.4-mini`
  - Claude path: `haiku`

Operational notes:
  - Completed task responses are committed to SQLite immediately and reused on rerun.
  - Quota/rate-limit failures trigger a shared pause window (default: 5 hours)
    and then retries resume from the cache checkpoint.

Outputs:
  - preprocessing/cache/llm_responses.db
  - preprocessing/intermediate/stage10_llm_results.parquet
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import random
import re
import signal
import sqlite3
import statistics
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).parent.parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
CACHE_DIR = PROJECT_ROOT / "preprocessing" / "cache"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

DEFAULT_INPUT_PATH = INTERMEDIATE_DIR / "stage9_llm_candidates.parquet"
DEFAULT_RESULTS_PATH = INTERMEDIATE_DIR / "stage10_llm_results.parquet"
DEFAULT_CACHE_DB = CACHE_DIR / "llm_responses.db"
DEFAULT_ERROR_LOG = LOG_DIR / "llm_errors.jsonl"

CHUNK_SIZE = 1_000
SQLITE_IN_LIMIT = 900
PROFILE_RANDOM_SEED = 42

CLASSIFICATION_TASK_NAME = "classification"
EXTRACTION_TASK_NAME = "core_extraction_units"
TASK_NAMES = (CLASSIFICATION_TASK_NAME, EXTRACTION_TASK_NAME)
EXTRACTION_STATUS_OK = "ok"
EXTRACTION_STATUS_CANNOT_COMPLETE = "cannot_complete"
EXTRACTION_STATUS_ENUM = {EXTRACTION_STATUS_OK, EXTRACTION_STATUS_CANNOT_COMPLETE}
MAX_EXTRACTION_UNIT_CHARS = 700
MAX_EXTRACTION_UNIT_COUNT = 120
SINGLE_UNIT_WARNING_CHARS = 500

CLASSIFICATION_PROMPT_TEMPLATE = """You are a labor economics research assistant classifying job postings.
Perform the tasks below on this job posting. Return ONLY valid JSON.

TASK 1 - SWE CLASSIFICATION
Classify this role into exactly one category:
- "SWE": The role's primary function is writing, designing, or maintaining
  software. Includes software engineers, full-stack developers,
  frontend/backend engineers, mobile developers, ML engineers, data engineers
  who primarily write code, and DevOps engineers whose description emphasizes
  writing code for infrastructure.
- "SWE_ADJACENT": Technical roles that involve some code but where coding is
  not the primary function. Includes QA/SDET roles, technical program
  managers, solutions architects, many data-science roles, and product roles
  for developer tooling where coding is useful but not the primary output.
- "NOT_SWE": Roles where software development is not a meaningful part of the
  job. Includes non-technical roles, most IT support roles, hardware/product
  engineering roles focused on physical systems, and misleading "engineer"
  titles where the work is not software development.

TASK 2 - SENIORITY CLASSIFICATION
Use ONLY explicit seniority signals from the title or description.

Strong signals:
- "junior", "jr", "intern", "internship", "new grad", "graduate",
  "entry-level", "early career", "apprentice" -> "entry"
- "senior", "sr", "staff", "principal", "lead", "architect" -> "mid-senior"
- "director", "vp", "vice president", "head of", "chief" -> "director"

Weak company-specific signals:
- "associate", "analyst", "consultant", "fellow", "partner"
- numeric or Roman numeral levels such as I/II/III, 1/2/3, L3/L4/L5, E3/E4/E5

Rules for weak signals:
- Use them ONLY when the posting itself makes the mapping explicit.
- Do NOT assume company-specific numbering or title ladders from general knowledge.
- If the only seniority evidence is a weak company-specific signal, return "unknown".

IMPORTANT:
- Do NOT infer seniority from years of experience, responsibilities, tech stack,
  scope, or company reputation.
- When in doubt, classify as "unknown".

TASK 3 - GHOST JOB ASSESSMENT
Assess whether this posting's requirements are realistic for its stated level:
- "realistic": Requirements match the stated or apparent seniority level
- "inflated": Requirements are significantly higher than the stated level would
  normally demand
- "ghost_likely": Strong signs this is not a genuine open position

---

TITLE: {title}
COMPANY: {company}
DESCRIPTION:
{full_description}

---

Respond with this exact JSON structure:
{{
  "swe_classification": "SWE" | "SWE_ADJACENT" | "NOT_SWE",
  "seniority": "entry" | "associate" | "mid-senior" | "director" | "unknown",
  "ghost_assessment": "realistic" | "inflated" | "ghost_likely"
}}"""

EXTRACTION_PROMPT_TEMPLATE = """You are a labor economics research assistant removing boilerplate from job postings.
You will receive the description split into numbered extraction units. Return ONLY valid JSON.

Goal:
- Mark units that are clearly boilerplate or non-core.
- Keep units that contain core job content.

Core job content includes:
- role summary
- responsibilities
- requirements
- qualifications
- preferred qualifications
- tech stack
- day-to-day work
- role-specific logistics such as shift, department, travel, or contract length

Boilerplate or non-core text includes:
- company overview / mission / values
- generic benefits / compensation / perks
- EEO / diversity / legal text
- application instructions
- recruiter or staffing platform framing
- generic location / requisition metadata
- generic remote / hybrid policy text

Rules:
- Return IDs for units to DROP, not units to keep.
- If a unit mixes core and boilerplate, keep it.
- Prefer high precision on dropping. When uncertain, do not drop the unit.
- If the units are malformed or the task cannot be completed reliably, return
  "task_status": "cannot_complete" with empty ID lists.
- Keep `reason` short.

TITLE: {title}
COMPANY: {company}

NUMBERED UNITS:
{numbered_units}

Respond with this exact JSON structure:
{{
  "task_status": "ok" | "cannot_complete",
  "boilerplate_unit_ids": [1, 2],
  "uncertain_unit_ids": [3],
  "reason": "short phrase"
}}"""

CLASSIFICATION_PROMPT_VERSION = hashlib.sha256(
    CLASSIFICATION_PROMPT_TEMPLATE.encode("utf-8")
).hexdigest()
EXTRACTION_PROMPT_VERSION = hashlib.sha256(
    EXTRACTION_PROMPT_TEMPLATE.encode("utf-8")
).hexdigest()
PROMPT_BUNDLE_VERSION = hashlib.sha256(
    f"{CLASSIFICATION_PROMPT_VERSION}:{EXTRACTION_PROMPT_VERSION}".encode("utf-8")
).hexdigest()
PROMPT_VERSION = PROMPT_BUNDLE_VERSION

CLASSIFICATION_KEYS = {"swe_classification", "seniority", "ghost_assessment"}
EXTRACTION_KEYS = {"task_status", "boilerplate_unit_ids", "uncertain_unit_ids", "reason"}
SWE_ENUM = {"SWE", "SWE_ADJACENT", "NOT_SWE"}
SENIORITY_ENUM = {"entry", "associate", "mid-senior", "director", "unknown"}
GHOST_ENUM = {"realistic", "inflated", "ghost_likely"}
SUPPORTED_PROVIDERS = ("codex", "claude")
DEFAULT_QUOTA_WAIT_HOURS = 5.0
DEFAULT_CODEX_MODEL = "gpt-5.4-mini"
DEFAULT_CLAUDE_MODEL = "haiku"

STOP_REQUESTED = False
QUOTA_PAUSE_LOCK = threading.Lock()
QUOTA_PAUSED_UNTIL = 0.0


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


def install_signal_handlers(log: logging.Logger) -> None:
    def _handle_sigint(signum, frame):  # noqa: ARG001
        global STOP_REQUESTED
        STOP_REQUESTED = True
        log.warning("Interrupt requested. Finishing in-flight tasks before exiting.")

    signal.signal(signal.SIGINT, _handle_sigint)


def normalize_newlines(text) -> str:
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def segment_description_into_blocks(text) -> list[dict]:
    description = normalize_newlines(text)
    if not description.strip():
        return []

    raw_parts = re.split(r"\n\s*\n+", description)
    blocks = []
    cursor = 0
    for part in raw_parts:
        if not part.strip():
            continue
        start = description.find(part, cursor)
        if start < 0:
            start = cursor
        end = start + len(part)
        blocks.append(
            {
                "block_id": len(blocks) + 1,
                "start_char": start,
                "end_char": end,
                "text": part,
            }
        )
        cursor = end
    return blocks


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[-*•+]?\s*)?[A-Z0-9(])")
BULLET_LINE_RE = re.compile(r"^\s*(?:[-*•+]|[0-9]+[.)])\s+")
METADATA_LINE_RE = re.compile(
    r"^(?:"
    r"(?:req(?:uisition)?|job)\s*#?:"
    r"|location:"
    r"|salary:"
    r"|pay(?: range)?:"
    r"|job type:"
    r"|shift:"
    r"|schedule:"
    r"|department:"
    r"|category:"
    r"|status:"
    r"|benefits?:"
    r"|posted date:"
    r"|site name:"
    r")",
    flags=re.IGNORECASE,
)


def looks_like_heading(text: str) -> bool:
    cleaned = text.strip().strip("*#").strip()
    if not cleaned:
        return False
    if len(cleaned) > 80:
        return False
    if cleaned.endswith(":"):
        return True
    words = cleaned.split()
    if len(words) > 8:
        return False
    return cleaned == cleaned.title() or cleaned.isupper()


def split_long_text_into_units(text: str, max_chars: int = MAX_EXTRACTION_UNIT_CHARS) -> list[str]:
    candidate = text.strip()
    if not candidate:
        return []
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(candidate) if part.strip()]
    if len(parts) <= 1:
        return [candidate]
    for part in parts:
        if len(part) <= max_chars:
            continue
        return [candidate]
    units = parts
    return units


def split_line_into_units(line: str) -> list[str]:
    cleaned = line.strip()
    if not cleaned:
        return []
    if BULLET_LINE_RE.match(cleaned) or METADATA_LINE_RE.match(cleaned) or looks_like_heading(cleaned):
        return [cleaned]
    return split_long_text_into_units(cleaned)


def segment_description_into_units(text) -> list[dict]:
    description = normalize_newlines(text)
    if not description.strip():
        return []

    units: list[dict] = []
    block_texts = [block["text"] for block in segment_description_into_blocks(description)]
    raw_chunks = block_texts if block_texts else [description]

    for chunk in raw_chunks:
        lines = [line.strip() for line in chunk.split("\n") if line.strip()]
        if not lines:
            continue

        i = 0
        while i < len(lines):
            line = lines[i]
            if looks_like_heading(line) and i + 1 < len(lines) and not looks_like_heading(lines[i + 1]):
                merged = f"{line}\n{lines[i + 1].strip()}"
                if len(merged) <= MAX_EXTRACTION_UNIT_CHARS:
                    for piece in split_line_into_units(merged):
                        units.append(
                            {
                                "unit_id": len(units) + 1,
                                "unit_type": "heading_group",
                                "text": piece,
                            }
                        )
                    i += 2
                    continue

            for piece in split_line_into_units(line):
                units.append(
                    {
                        "unit_id": len(units) + 1,
                        "unit_type": "line",
                        "text": piece,
                    }
                )
            i += 1

    return units


def join_retained_units(units: list[dict], boilerplate_unit_ids: list[int]) -> str:
    dropped_ids = set(boilerplate_unit_ids)
    selected = [unit["text"] for unit in units if unit["unit_id"] not in dropped_ids]
    return "\n\n".join(selected)


def format_numbered_units(units: list[dict]) -> str:
    parts = []
    for unit in units:
        parts.append(f"[{unit['unit_id']}]\n{unit['text']}")
    return "\n\n".join(parts)


def render_classification_prompt(title, company, full_description) -> str:
    return CLASSIFICATION_PROMPT_TEMPLATE.format(
        title="" if title is None else str(title),
        company="" if company is None else str(company),
        full_description="" if full_description is None else str(full_description),
    )


def render_extraction_prompt(title, company, full_description) -> tuple[str, list[dict]]:
    units = segment_description_into_units(full_description)
    return (
        EXTRACTION_PROMPT_TEMPLATE.format(
            title="" if title is None else str(title),
            company="" if company is None else str(company),
            numbered_units=format_numbered_units(units),
        ),
        units,
    )


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def ensure_cache_schema(conn: sqlite3.Connection) -> None:
    existing = conn.execute("PRAGMA table_info(responses)").fetchall()
    expected = {
        "description_hash",
        "task_name",
        "prompt_version",
        "model",
        "response_json",
        "timestamp",
        "tokens_used",
    }

    if existing:
        existing_cols = {row[1] for row in existing}
        if expected - existing_cols:
            legacy_name = f"responses_legacy_{int(time.time())}"
            conn.execute(f"ALTER TABLE responses RENAME TO {legacy_name}")
            conn.commit()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS responses (
            description_hash TEXT NOT NULL,
            task_name TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            model TEXT NOT NULL,
            response_json TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            tokens_used INTEGER,
            PRIMARY KEY (description_hash, task_name, prompt_version)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_responses_lookup
        ON responses (task_name, prompt_version, description_hash)
        """
    )
    conn.commit()


def open_cache(cache_db: Path) -> sqlite3.Connection:
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cache_db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    ensure_cache_schema(conn)
    return conn


def fetch_cached_rows(
    conn: sqlite3.Connection,
    hashes: list[str],
    task_name: str,
    prompt_version: str,
) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not hashes:
        return rows
    for batch in chunked(hashes, SQLITE_IN_LIMIT):
        placeholders = ",".join("?" for _ in batch)
        query = (
            "SELECT description_hash, task_name, model, prompt_version, response_json, timestamp, tokens_used "
            f"FROM responses WHERE task_name = ? AND prompt_version = ? "
            f"AND description_hash IN ({placeholders})"
        )
        params = [task_name, prompt_version, *batch]
        for row in conn.execute(query, params):
            rows[row[0]] = {
                "description_hash": row[0],
                "task_name": row[1],
                "model": row[2],
                "prompt_version": row[3],
                "response_json": row[4],
                "timestamp": row[5],
                "tokens_used": row[6],
            }
    return rows


def fetch_cached_row(
    conn: sqlite3.Connection,
    description_hash: str,
    task_name: str,
    prompt_version: str,
) -> dict | None:
    row = conn.execute(
        """
        SELECT description_hash, task_name, model, prompt_version, response_json, timestamp, tokens_used
        FROM responses
        WHERE description_hash = ? AND task_name = ? AND prompt_version = ?
        """,
        (description_hash, task_name, prompt_version),
    ).fetchone()
    if row is None:
        return None
    return {
        "description_hash": row[0],
        "task_name": row[1],
        "model": row[2],
        "prompt_version": row[3],
        "response_json": row[4],
        "timestamp": row[5],
        "tokens_used": row[6],
    }


def store_cached_row(
    conn: sqlite3.Connection,
    description_hash: str,
    task_name: str,
    model: str,
    prompt_version: str,
    response_json: str,
    tokens_used: int | None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO responses
        (description_hash, task_name, prompt_version, model, response_json, timestamp, tokens_used)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            description_hash,
            task_name,
            prompt_version,
            model,
            response_json,
            datetime.now(timezone.utc).isoformat(),
            tokens_used,
        ),
    )
    conn.commit()


def normalize_json_candidate(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```json"):
        candidate = candidate.split("```json", 1)[1]
    if candidate.startswith("```"):
        candidate = candidate.split("```", 1)[1]
    if candidate.endswith("```"):
        candidate = candidate.rsplit("```", 1)[0]
    return candidate.strip()


def extract_first_json_object(text: str) -> str:
    candidate = normalize_json_candidate(text)
    decoder = json.JSONDecoder()
    for idx, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(candidate[idx:])
            return json.dumps(obj, ensure_ascii=False)
        except json.JSONDecodeError:
            continue
    raise ValueError("no_json_object_found")


def parse_codex_stdout(stdout: str) -> tuple[str, int | None, float | None]:
    tokens_used = None
    match = re.search(r"tokens used\s+([\d,]+)", stdout, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        tokens_used = int(match.group(1).replace(",", ""))
    json_text = extract_first_json_object(stdout)
    return json_text, tokens_used, None


def parse_claude_stdout(stdout: str) -> tuple[str, int | None, float | None]:
    outer = json.loads(stdout)
    response_text = outer.get("result", stdout)
    usage = outer.get("usage", {}) or {}
    tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    if tokens_used == 0:
        tokens_used = None
    cost = outer.get("total_cost_usd")
    json_text = extract_first_json_object(response_text)
    return json_text, tokens_used, cost


def validate_classification_payload(payload: dict) -> str | None:
    if not isinstance(payload, dict):
        return "response_not_object"
    if set(payload.keys()) != CLASSIFICATION_KEYS:
        return "invalid_keys"
    if payload["swe_classification"] not in SWE_ENUM:
        return "invalid_swe_classification"
    if payload["seniority"] not in SENIORITY_ENUM:
        return "invalid_seniority"
    if payload["ghost_assessment"] not in GHOST_ENUM:
        return "invalid_ghost_assessment"
    return None


def validate_extraction_payload(payload: dict) -> str | None:
    if not isinstance(payload, dict):
        return "response_not_object"
    if set(payload.keys()) != EXTRACTION_KEYS:
        return "invalid_keys"
    if payload["task_status"] not in EXTRACTION_STATUS_ENUM:
        return "invalid_task_status"
    for field_name in ("boilerplate_unit_ids", "uncertain_unit_ids"):
        keep_ids = payload[field_name]
        if not isinstance(keep_ids, list):
            return f"invalid_{field_name}"
        normalized_ids = []
        for value in keep_ids:
            if isinstance(value, bool) or not isinstance(value, int):
                return f"invalid_{field_name}"
            if value < 1:
                return f"invalid_{field_name}"
            normalized_ids.append(value)
        if len(normalized_ids) != len(set(normalized_ids)):
            return f"duplicate_{field_name}"
        if normalized_ids != sorted(normalized_ids):
            return f"unsorted_{field_name}"
    if not isinstance(payload["reason"], str):
        return "invalid_reason"
    if payload["task_status"] == EXTRACTION_STATUS_CANNOT_COMPLETE and (
        payload["boilerplate_unit_ids"] or payload["uncertain_unit_ids"]
    ):
        return "cannot_complete_with_ids"
    overlap = set(payload["boilerplate_unit_ids"]) & set(payload["uncertain_unit_ids"])
    if overlap:
        return "overlapping_extraction_ids"
    return None


def call_subprocess(command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def build_codex_command(prompt: str, model: str) -> list[str]:
    return [
        "codex",
        "exec",
        "--full-auto",
        "--config",
        f"model={model}",
        prompt,
        "--skip-git-repo-check",
    ]


def build_claude_command(prompt: str, model: str) -> list[str]:
    return [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "json",
    ]


def quota_retry_after_seconds(wait_hours: float) -> float:
    return max(wait_hours * 3600.0, 0.0)


def quota_pause_remaining_seconds() -> float:
    with QUOTA_PAUSE_LOCK:
        return max(QUOTA_PAUSED_UNTIL - time.time(), 0.0)


def wait_for_quota_pause() -> None:
    while True:
        remaining = quota_pause_remaining_seconds()
        if remaining <= 0 or STOP_REQUESTED:
            return
        time.sleep(min(remaining, 60.0))


def detect_quota_or_rate_limit(text: str) -> bool:
    candidate = (text or "").lower()
    if not candidate:
        return False
    patterns = (
        "insufficient_quota",
        "quota",
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
        "usage limit",
        "credit balance",
        "exceeded your current quota",
        "retry after",
        "request limit",
        "token limit reached",
    )
    return any(pattern in candidate for pattern in patterns)


def activate_quota_pause(
    provider: str,
    model: str,
    wait_seconds: float,
    log: logging.Logger,
    description_hash: str,
    task_name: str,
    detail: str,
) -> None:
    global QUOTA_PAUSED_UNTIL

    now = time.time()
    new_until = now + wait_seconds
    with QUOTA_PAUSE_LOCK:
        previous_until = QUOTA_PAUSED_UNTIL
        if new_until > QUOTA_PAUSED_UNTIL:
            QUOTA_PAUSED_UNTIL = new_until
        active_until = QUOTA_PAUSED_UNTIL

    if active_until > previous_until:
        resume_at = datetime.now(timezone.utc) + timedelta(seconds=max(active_until - now, 0.0))
        log.warning(
            "Quota/rate limit detected for %s/%s on %s (%s). Pausing all new provider calls until %s UTC.",
            provider,
            model,
            description_hash,
            task_name,
            resume_at.isoformat(timespec="seconds"),
        )
        if detail:
            log.warning("Quota detail: %s", detail[:500])


def try_provider(
    provider: str,
    prompt: str,
    model: str,
    task_name: str,
    description_hash: str,
    error_log_path: Path,
    log: logging.Logger,
    timeout_seconds: int,
    max_retries: int,
    payload_validator,
    quota_wait_hours: float,
) -> dict | None:
    backoff_seconds = [1, 2, 4]
    attempt = 1
    wait_seconds = quota_retry_after_seconds(quota_wait_hours)

    while attempt <= max_retries:
        wait_for_quota_pause()
        if STOP_REQUESTED:
            return None
        t0 = time.time()
        try:
            if provider == "codex":
                result = call_subprocess(build_codex_command(prompt, model), timeout_seconds)
            elif provider == "claude":
                result = call_subprocess(build_claude_command(prompt, model), timeout_seconds)
            else:
                raise ValueError(f"unknown provider: {provider}")

            latency = time.time() - t0
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            combined_output = "\n".join(part for part in (stderr, stdout) if part)

            if result.returncode != 0:
                is_quota_limited = detect_quota_or_rate_limit(combined_output)
                append_jsonl(
                    error_log_path,
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "description_hash": description_hash,
                        "task_name": task_name,
                        "provider": provider,
                        "model": model,
                        "attempt": attempt,
                        "error_type": "subprocess_nonzero_exit",
                        "returncode": result.returncode,
                        "stderr": stderr[:2000],
                        "raw_response": stdout[:4000],
                    },
                )
                if is_quota_limited:
                    activate_quota_pause(
                        provider=provider,
                        model=model,
                        wait_seconds=wait_seconds,
                        log=log,
                        description_hash=description_hash,
                        task_name=task_name,
                        detail=combined_output,
                    )
                    continue
                if attempt < max_retries:
                    time.sleep(backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)])
                    attempt += 1
                    continue
                return None

            if provider == "codex":
                response_json, tokens_used, cost_usd = parse_codex_stdout(stdout)
                response_model = model
            else:
                response_json, tokens_used, cost_usd = parse_claude_stdout(stdout)
                response_model = model

            payload = json.loads(response_json)
            validation_error = payload_validator(payload)
            if validation_error is not None:
                append_jsonl(
                    error_log_path,
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "description_hash": description_hash,
                        "task_name": task_name,
                        "provider": provider,
                        "model": response_model,
                        "attempt": attempt,
                        "error_type": validation_error,
                        "raw_response": stdout[:8000],
                    },
                )
                if attempt < max_retries:
                    time.sleep(backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)])
                    attempt += 1
                    continue
                return None

            return {
                "provider": provider,
                "model": response_model,
                "latency_seconds": latency,
                "response_json": json.dumps(payload, ensure_ascii=False),
                "payload": payload,
                "tokens_used": tokens_used,
                "cost_usd": cost_usd,
            }
        except subprocess.TimeoutExpired:
            append_jsonl(
                error_log_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "description_hash": description_hash,
                    "task_name": task_name,
                    "provider": provider,
                    "model": model,
                    "attempt": attempt,
                    "error_type": "timeout",
                    "raw_response": "",
                },
            )
            if attempt < max_retries:
                time.sleep(backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)])
                attempt += 1
                continue
        except (json.JSONDecodeError, ValueError) as exc:
            append_jsonl(
                error_log_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "description_hash": description_hash,
                    "task_name": task_name,
                    "provider": provider,
                    "model": model,
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "raw_response": stdout[:8000] if "stdout" in locals() else "",
                },
            )
            raw_text = stdout[:8000] if "stdout" in locals() else ""
            if detect_quota_or_rate_limit(raw_text):
                activate_quota_pause(
                    provider=provider,
                    model=model,
                    wait_seconds=wait_seconds,
                    log=log,
                    description_hash=description_hash,
                    task_name=task_name,
                    detail=raw_text,
                )
                continue
        except Exception as exc:  # noqa: BLE001
            log.exception("Provider %s failed for %s (%s)", provider, description_hash, task_name)
            append_jsonl(
                error_log_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "description_hash": description_hash,
                    "task_name": task_name,
                    "provider": provider,
                    "model": model,
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "raw_response": "",
                },
            )

        if attempt < max_retries:
            time.sleep(backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)])
            attempt += 1
            continue

        return None

    return None


def call_task_with_fallback(
    task_name: str,
    prompt: str,
    description_hash: str,
    error_log_path: Path,
    log: logging.Logger,
    codex_model: str,
    timeout_seconds: int,
    max_retries: int,
    payload_validator,
    provider_order: tuple[str, ...] = SUPPORTED_PROVIDERS,
    claude_model: str = DEFAULT_CLAUDE_MODEL,
    quota_wait_hours: float = DEFAULT_QUOTA_WAIT_HOURS,
) -> dict | None:
    for idx, provider in enumerate(provider_order):
        if provider == "codex":
            model = codex_model
        elif provider == "claude":
            model = claude_model
        else:  # pragma: no cover
            raise ValueError(f"unsupported provider: {provider}")

        result = try_provider(
            provider=provider,
            prompt=prompt,
            model=model,
            task_name=task_name,
            description_hash=description_hash,
            error_log_path=error_log_path,
            log=log,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            payload_validator=payload_validator,
            quota_wait_hours=quota_wait_hours,
        )
        if result is not None:
            return result
        if idx + 1 < len(provider_order):
            next_provider = provider_order[idx + 1]
            log.warning(
                "Falling back from %s to %s for %s (%s)",
                provider,
                next_provider,
                description_hash,
                task_name,
            )

    return None


def parse_provider_order(value: str) -> tuple[str, ...]:
    providers = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not providers:
        raise ValueError("provider order must include at least one provider")
    invalid = [provider for provider in providers if provider not in SUPPORTED_PROVIDERS]
    if invalid:
        raise ValueError(f"unsupported providers in order: {', '.join(invalid)}")
    if len(providers) != len(set(providers)):
        raise ValueError("provider order must not contain duplicates")
    return providers


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def profile_marker_path(bundle_version: str) -> Path:
    return CACHE_DIR / f"stage10_profile_{bundle_version}.json"


def write_profile_marker(bundle_version: str, payload: dict) -> None:
    profile_marker_path(bundle_version).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def has_profile_marker(bundle_version: str) -> bool:
    return profile_marker_path(bundle_version).exists()


def load_candidate_frame(input_path: Path) -> tuple[pd.DataFrame, int]:
    cols = [
        "job_id",
        "source",
        "source_platform",
        "title",
        "company_name",
        "description",
        "description_hash",
        "needs_llm_classification",
        "needs_llm_extraction",
        "llm_route_group",
    ]
    available_cols = [col for col in cols if col in pq.ParquetFile(input_path).schema.names]
    df = pq.read_table(input_path, columns=available_cols).to_pandas()
    raw_candidate_rows = len(df)
    df["description_hash"] = df["description_hash"].astype(str)

    if "needs_llm_classification" not in df.columns:
        df["needs_llm_classification"] = True
    if "needs_llm_extraction" not in df.columns:
        df["needs_llm_extraction"] = True
    if "llm_route_group" not in df.columns:
        df["llm_route_group"] = "classification_and_extraction"

    df["needs_llm_classification"] = df["needs_llm_classification"].fillna(False).astype(bool)
    df["needs_llm_extraction"] = df["needs_llm_extraction"].fillna(False).astype(bool)

    if df.duplicated(subset=["description_hash"]).any():
        grouped = (
            df.groupby("description_hash", as_index=False)
            .agg(
                {
                    "job_id": "first",
                    "source": "first",
                    "source_platform": "first",
                    "title": "first",
                    "company_name": "first",
                    "description": "first",
                    "needs_llm_classification": "max",
                    "needs_llm_extraction": "max",
                    "llm_route_group": "first",
                }
            )
        )
        df = grouped
    else:
        df = df.drop_duplicates(subset=["description_hash"], keep="first").reset_index(drop=True)

    return df, raw_candidate_rows


def select_profile_frame(df: pd.DataFrame, profile_limit: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    rng = random.Random(PROFILE_RANDOM_SEED)
    frame = df.copy()
    frame["_source_group"] = (
        frame["source"].fillna("unknown").astype(str)
        + " | "
        + frame["source_platform"].fillna("unknown").astype(str)
    )

    strata = sorted(frame["_source_group"].unique().tolist())
    per_stratum = max(profile_limit // max(len(strata), 1), 1)
    selected_by_group: dict[str, deque] = {}
    selected_indices: set[int] = set()

    for stratum in strata:
        stratum_indices = frame.index[frame["_source_group"] == stratum].tolist()
        sample_n = min(len(stratum_indices), per_stratum)
        chosen = rng.sample(stratum_indices, sample_n)
        selected_by_group[stratum] = deque(chosen)
        selected_indices.update(chosen)

    ordered_indices = []
    while any(selected_by_group.values()):
        for stratum in strata:
            if selected_by_group[stratum]:
                ordered_indices.append(selected_by_group[stratum].popleft())

    if len(ordered_indices) < profile_limit:
        remaining = [idx for idx in frame.index.tolist() if idx not in selected_indices]
        extra = rng.sample(remaining, min(profile_limit - len(ordered_indices), len(remaining)))
        ordered_indices.extend(extra)

    selected = frame.loc[ordered_indices].drop(columns=["_source_group"]).reset_index(drop=True)
    return selected


def has_uncached_rows(df: pd.DataFrame, conn: sqlite3.Connection) -> bool:
    if df.empty:
        return False

    records = df[["description_hash", "needs_llm_classification", "needs_llm_extraction"]].to_dict("records")
    for batch_records in chunked(records, CHUNK_SIZE):
        batch_hashes = [str(row["description_hash"]) for row in batch_records]
        cached_class = fetch_cached_rows(
            conn,
            batch_hashes,
            CLASSIFICATION_TASK_NAME,
            CLASSIFICATION_PROMPT_VERSION,
        )
        cached_extract = fetch_cached_rows(
            conn,
            batch_hashes,
            EXTRACTION_TASK_NAME,
            EXTRACTION_PROMPT_VERSION,
        )
        for row in batch_records:
            description_hash = str(row["description_hash"])
            if row["needs_llm_classification"] and description_hash not in cached_class:
                return True
            if row["needs_llm_extraction"] and description_hash not in cached_extract:
                return True
    return False


def process_row_tasks(
    row: dict,
    need_classification: bool,
    need_extraction: bool,
    codex_model: str,
    codex_delay: float,
    claude_delay: float,
    timeout_seconds: int,
    max_retries: int,
    error_log_path: Path,
    log: logging.Logger,
    provider_order: tuple[str, ...],
    claude_model: str,
    quota_wait_hours: float,
) -> dict:
    results = {
        "description_hash": row["description_hash"],
        "task_results": {},
        "tasks_attempted": 0,
        "task_errors": 0,
    }

    if need_extraction:
        extraction_prompt, units = render_extraction_prompt(
            row["title"],
            row["company_name"],
            row["description"],
        )
        results["tasks_attempted"] += 1
        if not units:
            payload = {
                "task_status": EXTRACTION_STATUS_CANNOT_COMPLETE,
                "boilerplate_unit_ids": [],
                "uncertain_unit_ids": [],
                "reason": "empty_description",
            }
            results["task_results"][EXTRACTION_TASK_NAME] = {
                "provider": "synthetic",
                "model": "synthetic-empty-description",
                "latency_seconds": 0.0,
                "response_json": json.dumps(payload, ensure_ascii=False),
                "payload": payload,
                "tokens_used": None,
                "cost_usd": None,
                "prompt_version": EXTRACTION_PROMPT_VERSION,
            }
        elif len(units) == 1 and len(units[0]["text"]) >= SINGLE_UNIT_WARNING_CHARS:
            payload = {
                "task_status": EXTRACTION_STATUS_CANNOT_COMPLETE,
                "boilerplate_unit_ids": [],
                "uncertain_unit_ids": [],
                "reason": "single_unit_description",
            }
            results["task_results"][EXTRACTION_TASK_NAME] = {
                "provider": "synthetic",
                "model": "synthetic-single-unit-fallback",
                "latency_seconds": 0.0,
                "response_json": json.dumps(payload, ensure_ascii=False),
                "payload": payload,
                "tokens_used": None,
                "cost_usd": None,
                "prompt_version": EXTRACTION_PROMPT_VERSION,
            }
        elif len(units) > MAX_EXTRACTION_UNIT_COUNT:
            payload = {
                "task_status": EXTRACTION_STATUS_CANNOT_COMPLETE,
                "boilerplate_unit_ids": [],
                "uncertain_unit_ids": [],
                "reason": "too_many_units",
            }
            results["task_results"][EXTRACTION_TASK_NAME] = {
                "provider": "synthetic",
                "model": "synthetic-too-many-units-fallback",
                "latency_seconds": 0.0,
                "response_json": json.dumps(payload, ensure_ascii=False),
                "payload": payload,
                "tokens_used": None,
                "cost_usd": None,
                "prompt_version": EXTRACTION_PROMPT_VERSION,
            }
        else:
            extraction_result = call_task_with_fallback(
                task_name=EXTRACTION_TASK_NAME,
                prompt=extraction_prompt,
                description_hash=row["description_hash"],
                error_log_path=error_log_path,
                log=log,
                codex_model=codex_model,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                payload_validator=validate_extraction_payload,
                provider_order=provider_order,
                claude_model=claude_model,
                quota_wait_hours=quota_wait_hours,
            )
            if extraction_result is None:
                results["task_errors"] += 1
            else:
                extraction_result["prompt_version"] = EXTRACTION_PROMPT_VERSION
                results["task_results"][EXTRACTION_TASK_NAME] = extraction_result
                time.sleep(codex_delay if extraction_result["provider"] == "codex" else claude_delay)

    if need_classification:
        classification_prompt = render_classification_prompt(
            row["title"],
            row["company_name"],
            row["description"],
        )
        results["tasks_attempted"] += 1
        classification_result = call_task_with_fallback(
            task_name=CLASSIFICATION_TASK_NAME,
            prompt=classification_prompt,
            description_hash=row["description_hash"],
            error_log_path=error_log_path,
            log=log,
            codex_model=codex_model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            payload_validator=validate_classification_payload,
            provider_order=provider_order,
            claude_model=claude_model,
            quota_wait_hours=quota_wait_hours,
        )
        if classification_result is None:
            results["task_errors"] += 1
        else:
            classification_result["prompt_version"] = CLASSIFICATION_PROMPT_VERSION
            results["task_results"][CLASSIFICATION_TASK_NAME] = classification_result
            time.sleep(codex_delay if classification_result["provider"] == "codex" else claude_delay)

    return results


def build_results_row(
    row_meta: dict,
    description_hash: str,
    classification_cached: dict | None,
    extraction_cached: dict | None,
) -> dict:
    if not bool(row_meta.get("needs_llm_classification", False)):
        classification_cached = None
    if not bool(row_meta.get("needs_llm_extraction", False)):
        extraction_cached = None

    classification_payload = (
        json.loads(classification_cached["response_json"]) if classification_cached is not None else {}
    )
    extraction_payload = json.loads(extraction_cached["response_json"]) if extraction_cached is not None else {}

    return {
        "description_hash": description_hash,
        "needs_llm_classification": bool(row_meta.get("needs_llm_classification", False)),
        "needs_llm_extraction": bool(row_meta.get("needs_llm_extraction", False)),
        "llm_route_group": row_meta.get("llm_route_group"),
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
        "swe_classification_llm": classification_payload.get("swe_classification"),
        "seniority_llm": classification_payload.get("seniority"),
        "ghost_assessment_llm": classification_payload.get("ghost_assessment"),
        "llm_model_extraction": None if extraction_cached is None else extraction_cached["model"],
        "llm_prompt_version_extraction": (
            None if extraction_cached is None else extraction_cached["prompt_version"]
        ),
        "extraction_response_json": None if extraction_cached is None else extraction_cached["response_json"],
        "extraction_tokens_used": (
            None
            if extraction_cached is None or extraction_cached["tokens_used"] is None
            else float(extraction_cached["tokens_used"])
        ),
        "extraction_task_status_llm": extraction_payload.get("task_status"),
        "extraction_boilerplate_unit_ids_llm": (
            None
            if extraction_cached is None
            else json.dumps(extraction_payload.get("boilerplate_unit_ids", []), ensure_ascii=False)
        ),
        "extraction_uncertain_unit_ids_llm": (
            None
            if extraction_cached is None
            else json.dumps(extraction_payload.get("uncertain_unit_ids", []), ensure_ascii=False)
        ),
        "extraction_reason_llm": extraction_payload.get("reason"),
    }


def write_results_rows(rows: list[dict], output_path: Path) -> None:
    writer = None
    try:
        for batch in chunked(rows, CHUNK_SIZE):
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def run_stage10(
    input_path: Path = DEFAULT_INPUT_PATH,
    results_path: Path = DEFAULT_RESULTS_PATH,
    cache_db: Path = DEFAULT_CACHE_DB,
    error_log_path: Path = DEFAULT_ERROR_LOG,
    profile: bool = False,
    profile_limit: int = 100,
    codex_model: str = DEFAULT_CODEX_MODEL,
    codex_delay: float = 0.5,
    claude_delay: float = 1.0,
    timeout_seconds: int = 180,
    max_retries: int = 3,
    max_workers: int = 12,
    provider_order: tuple[str, ...] = SUPPORTED_PROVIDERS,
    claude_model: str = DEFAULT_CLAUDE_MODEL,
    quota_wait_hours: float = DEFAULT_QUOTA_WAIT_HOURS,
) -> None:
    log = configure_logging()
    install_signal_handlers(log)
    t0 = time.time()

    log.info("=" * 70)
    log.info("Stage 10: LLM extraction + classification")
    log.info("=" * 70)
    log.info("Input: %s", input_path)
    log.info("Cache DB: %s", cache_db)
    log.info("Results output: %s", results_path)
    log.info("Prompt bundle version: %s", PROMPT_BUNDLE_VERSION)
    log.info("Profile mode: %s", profile)
    log.info("Max workers: %s", max_workers)
    log.info("Provider order: %s", ",".join(provider_order))
    log.info("Codex model: %s", codex_model)
    log.info("Claude model: %s", claude_model)
    log.info("Quota wait hours on rate-limit/quota errors: %.1f", quota_wait_hours)

    candidate_df, raw_candidate_rows = load_candidate_frame(input_path)
    total_unique_rows = len(candidate_df)
    target_df = select_profile_frame(candidate_df, profile_limit) if profile else candidate_df

    if not profile and results_path.exists():
        log.warning("Removing existing Stage 10 results file before rerun: %s", results_path)
        results_path.unlink()

    conn = open_cache(cache_db)
    if (
        not profile
        and has_uncached_rows(candidate_df, conn)
        and not has_profile_marker(PROMPT_BUNDLE_VERSION)
    ):
        raise RuntimeError(
            "Profile mode has not been run for the current prompt bundle version. "
            "Run stage10_llm_classify.py --profile first."
        )

    log.info("Candidate rows received from Stage 9: %s", f"{raw_candidate_rows:,}")
    log.info("Unique description hashes for LLM calling: %s", f"{total_unique_rows:,}")
    log.info(
        "Stage 10 deduplicates only LLM calls by description hash; analytical row preservation happens in Stage 11."
    )
    log.info(
        "Hashes requesting classification: %s",
        f"{int(target_df['needs_llm_classification'].sum()):,}",
    )
    log.info(
        "Hashes requesting extraction: %s",
        f"{int(target_df['needs_llm_extraction'].sum()):,}",
    )
    if profile:
        log.info("Profile sample size: %s", f"{len(target_df):,}")

    hashes = target_df["description_hash"].astype(str).tolist()
    cached_class = fetch_cached_rows(
        conn,
        hashes,
        CLASSIFICATION_TASK_NAME,
        CLASSIFICATION_PROMPT_VERSION,
    )
    cached_extract = fetch_cached_rows(
        conn,
        hashes,
        EXTRACTION_TASK_NAME,
        EXTRACTION_PROMPT_VERSION,
    )

    rows_to_process = []
    fully_cached_rows = 0
    total_tasks_needed = 0
    for row in target_df.to_dict("records"):
        description_hash = str(row["description_hash"])
        route_classification = bool(row.get("needs_llm_classification", False))
        route_extraction = bool(row.get("needs_llm_extraction", False))
        need_classification = route_classification and description_hash not in cached_class
        need_extraction = route_extraction and description_hash not in cached_extract
        if not need_classification and not need_extraction:
            fully_cached_rows += 1
            continue
        rows_to_process.append((row, need_classification, need_extraction))
        total_tasks_needed += int(need_classification) + int(need_extraction)

    log.info("Fully cached rows: %s", f"{fully_cached_rows:,}")
    log.info("Tasks requiring fresh LLM calls: %s", f"{total_tasks_needed:,}")

    tasks_completed = 0
    errors = 0
    latencies_by_task = {CLASSIFICATION_TASK_NAME: [], EXTRACTION_TASK_NAME: []}
    cost_values = []
    sample_responses = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for row, need_classification, need_extraction in rows_to_process:
            if STOP_REQUESTED:
                break
            future = executor.submit(
                process_row_tasks,
                row=row,
                need_classification=need_classification,
                need_extraction=need_extraction,
                codex_model=codex_model,
                codex_delay=codex_delay,
                claude_delay=claude_delay,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                error_log_path=error_log_path,
                log=log,
                provider_order=provider_order,
                claude_model=claude_model,
                quota_wait_hours=quota_wait_hours,
            )
            future_map[future] = row["description_hash"]

        for future in as_completed(future_map):
            result = future.result()
            description_hash = result["description_hash"]
            tasks_completed += result["tasks_attempted"]
            errors += result["task_errors"]

            for task_name, task_result in result["task_results"].items():
                store_cached_row(
                    conn,
                    description_hash=description_hash,
                    task_name=task_name,
                    model=task_result["model"],
                    prompt_version=task_result["prompt_version"],
                    response_json=task_result["response_json"],
                    tokens_used=task_result["tokens_used"],
                )
                if task_name == CLASSIFICATION_TASK_NAME:
                    cached_class[description_hash] = fetch_cached_row(
                        conn,
                        description_hash,
                        CLASSIFICATION_TASK_NAME,
                        CLASSIFICATION_PROMPT_VERSION,
                    )
                else:
                    cached_extract[description_hash] = fetch_cached_row(
                        conn,
                        description_hash,
                        EXTRACTION_TASK_NAME,
                        EXTRACTION_PROMPT_VERSION,
                    )
                latencies_by_task[task_name].append(task_result["latency_seconds"])
                if task_result["cost_usd"] is not None:
                    cost_values.append(float(task_result["cost_usd"]))
                if len(sample_responses) < 8:
                    sample_responses.append(
                        {
                            "description_hash": description_hash,
                            "task_name": task_name,
                            "model": task_result["model"],
                            "payload": task_result["payload"],
                        }
                    )

            if tasks_completed and tasks_completed % 25 == 0:
                elapsed = time.time() - t0
                avg_per_task = elapsed / max(tasks_completed, 1)
                remaining_tasks = max(total_tasks_needed - tasks_completed, 0)
                eta_seconds = remaining_tasks * avg_per_task
                log.info(
                    "Progress: tasks=%s/%s errors=%s elapsed=%.1fs eta=%.1fs",
                    f"{tasks_completed:,}",
                    f"{total_tasks_needed:,}",
                    f"{errors:,}",
                    elapsed,
                    eta_seconds,
                )

    if not profile:
        output_rows = []
        row_lookup = {str(row["description_hash"]): row for row in target_df.to_dict("records")}
        for batch in chunked(hashes, CHUNK_SIZE):
            cached_class_batch = fetch_cached_rows(
                conn,
                batch,
                CLASSIFICATION_TASK_NAME,
                CLASSIFICATION_PROMPT_VERSION,
            )
            cached_extract_batch = fetch_cached_rows(
                conn,
                batch,
                EXTRACTION_TASK_NAME,
                EXTRACTION_PROMPT_VERSION,
            )
            for description_hash in batch:
                output_rows.append(
                    build_results_row(
                        row_lookup[description_hash],
                        description_hash,
                        cached_class_batch.get(description_hash),
                        cached_extract_batch.get(description_hash),
                    )
                )
        write_results_rows(output_rows, results_path)

    conn.close()
    gc.collect()

    elapsed_total = time.time() - t0
    error_rate = errors / total_tasks_needed if total_tasks_needed else 0.0
    all_latencies = latencies_by_task[CLASSIFICATION_TASK_NAME] + latencies_by_task[EXTRACTION_TASK_NAME]
    mean_latency = statistics.mean(all_latencies) if all_latencies else None
    p95_latency = p95(all_latencies)

    log.info("=" * 70)
    log.info("Stage 10 complete in %.1fs", elapsed_total)
    log.info("=" * 70)
    log.info("  Candidate rows received from Stage 9: %s", f"{raw_candidate_rows:,}")
    log.info("  Unique description hashes for LLM calling: %s", f"{total_unique_rows:,}")
    log.info("  Target rows: %s", f"{len(target_df):,}")
    log.info("  Fully cached rows: %s", f"{fully_cached_rows:,}")
    log.info("  Fresh tasks attempted: %s", f"{total_tasks_needed:,}")
    log.info("  Output semantics: one Stage 10 row per unique description hash; no posting rows are dropped here.")
    log.info("  Errors: %s", f"{errors:,}")
    log.info("  Error rate: %.2f%%", error_rate * 100)
    if mean_latency is not None:
        log.info("  Mean latency per task: %.2fs", mean_latency)
    if p95_latency is not None:
        log.info("  P95 latency per task: %.2fs", p95_latency)
    for task_name in TASK_NAMES:
        task_latencies = latencies_by_task[task_name]
        if task_latencies:
            log.info(
                "  %s mean / P95 latency: %.2fs / %.2fs",
                task_name,
                statistics.mean(task_latencies),
                p95(task_latencies) or 0.0,
            )
    if not cost_values:
        log.info("  Estimated total cost: unknown (provider did not expose per-call cost)")
    else:
        mean_cost = statistics.mean(cost_values)
        estimated_total_cost = mean_cost * max(total_tasks_needed, 1)
        log.info("  Estimated total cost: $%.4f", estimated_total_cost)
    if sample_responses:
        log.info("  Sample responses:")
        for sample in sample_responses:
            log.info("    %s", json.dumps(sample, ensure_ascii=False)[:1200])

    profile_completed = profile and (not STOP_REQUESTED) and tasks_completed >= total_tasks_needed
    if profile_completed:
        marker_payload = {
            "prompt_bundle_version": PROMPT_BUNDLE_VERSION,
            "classification_prompt_version": CLASSIFICATION_PROMPT_VERSION,
            "extraction_prompt_version": EXTRACTION_PROMPT_VERSION,
            "profile_limit": profile_limit,
            "max_workers": max_workers,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "target_rows": len(target_df),
            "fresh_tasks_attempted": total_tasks_needed,
            "errors": errors,
            "mean_latency_seconds": mean_latency,
            "p95_latency_seconds": p95_latency,
        }
        write_profile_marker(PROMPT_BUNDLE_VERSION, marker_payload)
        log.info("Profile marker written to %s", profile_marker_path(PROMPT_BUNDLE_VERSION))
    elif profile:
        log.warning("Profile run did not complete; not writing profile marker.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 10 LLM extraction + classification")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--cache-db", type=Path, default=DEFAULT_CACHE_DB)
    parser.add_argument("--error-log", type=Path, default=DEFAULT_ERROR_LOG)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-limit", type=int, default=100)
    parser.add_argument("--codex-delay", type=float, default=0.5)
    parser.add_argument("--claude-delay", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=40)
    parser.add_argument(
        "--quota-wait-hours",
        type=float,
        default=DEFAULT_QUOTA_WAIT_HOURS,
        help="Shared pause window after quota/rate-limit failures before retrying the provider",
    )
    parser.add_argument(
        "--provider-order",
        default="codex,claude",
        help="Comma-separated provider order, e.g. 'codex,claude' or 'claude'",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    provider_order = parse_provider_order(args.provider_order)
    run_stage10(
        input_path=args.input,
        results_path=args.results_output,
        cache_db=args.cache_db,
        error_log_path=args.error_log,
        profile=args.profile,
        profile_limit=args.profile_limit,
        codex_delay=args.codex_delay,
        claude_delay=args.claude_delay,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        max_workers=args.max_workers,
        provider_order=provider_order,
        quota_wait_hours=args.quota_wait_hours,
    )
