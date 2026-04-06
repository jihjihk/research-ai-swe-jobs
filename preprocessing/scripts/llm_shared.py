from __future__ import annotations

import hashlib
import heapq
import json
import logging
import math
import random
import re
import shlex
import sqlite3
import statistics
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None

try:
    from io_utils import append_jsonl
except ImportError:  # pragma: no cover
    from .io_utils import append_jsonl


SQLITE_IN_LIMIT = 900

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

TASK 4 - YEARS OF EXPERIENCE EXTRACTION
Extract `yoe_min_years`.

Rules:
- Use explicit years-of-experience mentions only. Number words and digits both
  count.
- For a single qualification path, return the binding YOE floor: use the
  highest relevant years-of-experience mention on that path, including
  preferred figures.
- Tool/framework/domain-specific experience counts if it is the only YOE
  mention on that path, or if it is higher than the general-role YOE on that
  path.
- If the posting gives multiple acceptable qualification paths, return the
  lowest path-level YOE floor.
- If a range is given, return the lower bound.
- Ignore title/job-level numbers, dates, salaries, addresses, clearance levels,
  and other numbers not tied to YOE.
- If no relevant explicit YOE exists, return null.
- Do not use YOE to infer seniority.

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
  "ghost_assessment": "realistic" | "inflated" | "ghost_likely",
  "yoe_min_years": <integer|null>
}}"""

EXTRACTION_PROMPT_TEMPLATE = """You are preparing job-posting text for labor-market research.
You will receive numbered extraction units. Return ONLY valid JSON.

Goal:
Drop units that do not change what the worker does, must know, must have, must own, or must coordinate.

KEEP core job content:
- role summary
- responsibilities and day-to-day work
- requirements and qualifications
- preferred qualifications
- tech stack, tools, systems, methods
- domain expertise
- seniority- or scope-relevant expectations
- operational constraints that affect the work itself: travel frequency, shift coverage, on-call, clearance, contract length, or reporting line

DROP non-core text:
- company overview, mission, values, culture, and employer-branding
- all salary, pay, compensation, OTE, bonus, equity, and pay-range text
- all benefits, perks, insurance, PTO, leave, retirement, 401(k), wellness, tuition, and total-rewards text
- all EEO / equal-opportunity / anti-discrimination / accommodation / legal-policy boilerplate such as "equal opportunity employer", "all qualified applicants will receive consideration", or protected-class language
- all application instructions, recruiter/platform framing, and candidate-journey text
- generic metadata such as requisition IDs, posted dates, labels, and location headers
- all remote, hybrid, on-site, and work-model text, including in-office cadence, commute expectations, and flexibility language, unless it encodes a real work constraint like travel frequency, shift coverage, or clearance/facility access

Decision rule:
- Return IDs for units to DROP, not units to keep.
- Compensation is never core.
- Benefits are never core.
- Generic remote, hybrid, on-site, and work-model text is never core by itself.
- EEO/legal text is never core.
- EEO/legal boilerplate is never core, even when it mentions disability, veteran status, or other protected classes.
- Standalone headers inherit the type of their section. For example: Benefits, Benefits & Perks, Compensation, Salary Range, Work Model, Worker Category, EEO.
- If a unit mixes core content with salary, benefits, work-model, or EEO/legal text, put it in uncertain_unit_ids unless the whole unit is clearly non-core and should be dropped.
- Use only unit IDs that appear in the numbered units.
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

CLASSIFICATION_KEYS = {"swe_classification", "seniority", "ghost_assessment", "yoe_min_years"}
EXTRACTION_KEYS = {"task_status", "boilerplate_unit_ids", "uncertain_unit_ids", "reason"}
SWE_ENUM = {"SWE", "SWE_ADJACENT", "NOT_SWE"}
SENIORITY_ENUM = {"entry", "associate", "mid-senior", "director", "unknown"}
GHOST_ENUM = {"realistic", "inflated", "ghost_likely"}
SUPPORTED_PROVIDERS = ("codex", "claude")
ENGINE_TIER_FULL = "full"
ENGINE_TIER_NON_INTRUSIVE = "non_intrusive"
SUPPORTED_ENGINE_TIERS = (ENGINE_TIER_FULL, ENGINE_TIER_NON_INTRUSIVE)
DEFAULT_QUOTA_WAIT_HOURS = 5.0
DEFAULT_RETRY_SLEEP_SECONDS = 60.0
DEFAULT_CODEX_MODEL = "gpt-5.4-mini"
DEFAULT_CLAUDE_MODEL = "haiku"
DEFAULT_ENGINE_TIMEZONE = "America/Los_Angeles"
SAMPLED_RESPONSE_LOG_EVERY = 5_000
SAMPLED_RESPONSE_LOG_MAX_CHARS = 2_000
COMMAND_ERROR_LOG_MAX_CHARS = 4_000
REMOTE_SSH_KEY_PATH = "/home/jihgaboot/gabor/job-research/keys/scraper-key.pem"
REMOTE_SSH_HOST = "ec2-user@ec2-18-216-89-129.us-east-2.compute.amazonaws.com"
REMOTE_SSH_CONTROL_PATH = "~/.ssh/ssh-mux-%C"
REMOTE_SSH_CONTROL_PERSIST = "10m"

_REMOTE_EXECUTION = False

STOP_REQUESTED = False
QUOTA_PAUSE_LOCK = threading.Lock()
QUOTA_PAUSED_UNTIL = 0.0
REMOTE_SSH_MASTER_LOCK = threading.Lock()


def configure_remote_execution(enabled: bool) -> None:
    """Enable or disable SSH-based remote execution for LLM subprocess calls."""
    global _REMOTE_EXECUTION
    _REMOTE_EXECUTION = enabled


class LLMEngineConfig:
    def __init__(self, provider: str, model: str, tier: str = ENGINE_TIER_FULL) -> None:
        self.provider = provider
        self.model = model
        self.tier = tier


class LLMEngineState:
    def __init__(self) -> None:
        self.slot_key: str | None = None
        self.slot_calls_started = 0
        self.paused_until_ts = 0.0

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])(?:\s+|(?=[A-Z0-9(]))")
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
UI_NOISE_EXACT = {
    "show more",
    "show less",
}
KNOWN_SECTION_HEADERS = {
    "about team",
    "about us",
    "additional information",
    "additional responsibilities",
    "basic qualifications",
    "benefits",
    "citizenship requirement",
    "company description",
    "compensation",
    "core skills",
    "day to day",
    "day-to-day",
    "department",
    "description",
    "desired experience",
    "education",
    "education/experience",
    "essential duties and responsibilities",
    "experience range",
    "formal education",
    "job description",
    "job details",
    "job overview",
    "job responsibilities",
    "job summary",
    "job benefits",
    "long-term role",
    "maximum qualifications",
    "minimum qualifications",
    "must haves",
    "must-have",
    "must-haves",
    "nice to haves",
    "position profile",
    "position responsibilities",
    "position summary",
    "preferred qualification",
    "preferred qualifications",
    "preferred skills and education",
    "professional certifications",
    "qualifications",
    "required skills",
    "required skills / abilities",
    "required skills and experience",
    "requirement",
    "requirements",
    "responsibilities",
    "responsibilities include but are not limited to the following",
    "security clearance level",
    "security clearance requirements",
    "setting",
    "skills",
    "skills and qualifications",
    "technical skills",
    "the role",
    "top skills details",
    "type",
    "what you'll bring",
    "what you'll do",
    "what you bring",
    "what you’ll bring",
    "what you’ll do",
    "who we are",
    "who you are",
    "work/education experience",
    "years of professional experience",
}
INLINE_HEADER_PATTERN = re.compile(
    r"(?i)(?<=[.!?])(?=(?:Responsibilities|Qualifications|Requirements|Skills|Benefits|"
    r"Additional Responsibilities|Core Skills|What You'll Bring|What You'll Do|What you’ll bring|"
    r"What you’ll do|The Role|About us|About Us|Job Details|Minimum Qualifications|"
    r"Professional Certifications|Years of Professional Experience|Formal Education|"
    r"Citizenship Requirement|Security Clearance Requirements)(?:\s*:|\s*\.\.\.))"
)
ALL_CAPS_HEADER_RE = re.compile(r"^[A-Z0-9/&(),+ \-]{2,80}:?$")
LABEL_VALUE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9/&(),+ \-]{0,55}:?$")
LEADING_COLON_VALUE_RE = re.compile(r"^\s*:\s*.+$")
DOMAIN_RE = re.compile(r"\b(?:https?://\S+|[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+)\b")
ABBREVIATION_SENTINELS = {
    "U.S.": "U<S>S<PERIOD>",
    "u.s.": "u<s>s<period>",
    "e.g.": "e<PERIOD>g<PERIOD>",
    "i.e.": "i<PERIOD>e<PERIOD>",
    "Inc.": "Inc<PERIOD>",
    "inc.": "inc<PERIOD>",
    "LLC.": "LLC<PERIOD>",
    "llc.": "llc<PERIOD>",
    "Ltd.": "Ltd<PERIOD>",
    "ltd.": "ltd<PERIOD>",
    "Corp.": "Corp<PERIOD>",
    "corp.": "corp<PERIOD>",
    "Co.": "Co<PERIOD>",
    "co.": "co<PERIOD>",
    "St.": "St<PERIOD>",
    "st.": "st<PERIOD>",
    "Sr.": "Sr<PERIOD>",
    "sr.": "sr<PERIOD>",
    "Jr.": "Jr<PERIOD>",
    "jr.": "jr<PERIOD>",
    ".Net": "<DOT>Net",
    ".NET": "<DOT>NET",
}


def request_stop() -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_engine_list(value: str) -> tuple[str, ...]:
    providers = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not providers:
        raise ValueError("engines must include at least one provider")
    invalid = [provider for provider in providers if provider not in SUPPORTED_PROVIDERS]
    if invalid:
        raise ValueError(f"unsupported engines: {', '.join(invalid)}")
    if len(providers) != len(set(providers)):
        raise ValueError("engines must not contain duplicates")
    return providers


def parse_engine_tiers(
    value: str | None,
    enabled_providers: tuple[str, ...] | list[str] | None = None,
) -> dict[str, str]:
    providers = tuple(enabled_providers or SUPPORTED_PROVIDERS)
    tiers = {provider: ENGINE_TIER_FULL for provider in providers}
    if not value:
        return tiers

    for assignment in value.split(","):
        assignment = assignment.strip()
        if not assignment:
            continue
        provider, sep, tier = assignment.partition("=")
        provider = provider.strip().lower()
        tier = tier.strip().lower()
        if not sep:
            raise ValueError("engine tier assignments must use provider=tier")
        if provider not in tiers:
            raise ValueError(f"engine tier specified for disabled or unknown provider: {provider}")
        if tier not in SUPPORTED_ENGINE_TIERS:
            raise ValueError(f"unsupported engine tier for {provider}: {tier}")
        tiers[provider] = tier
    return tiers


def build_engine_configs(
    enabled_engines: tuple[str, ...],
    *,
    codex_model: str,
    claude_model: str,
    engine_tiers: dict[str, str] | None = None,
) -> tuple[LLMEngineConfig, ...]:
    resolved_tiers = parse_engine_tiers(None, enabled_engines) if engine_tiers is None else dict(engine_tiers)
    configs: list[LLMEngineConfig] = []
    for provider in enabled_engines:
        if provider == "codex":
            model = codex_model
        elif provider == "claude":
            model = claude_model
        else:  # pragma: no cover
            raise ValueError(f"unsupported provider: {provider}")
        tier = resolved_tiers.get(provider, ENGINE_TIER_FULL)
        if tier not in SUPPORTED_ENGINE_TIERS:
            raise ValueError(f"unsupported engine tier for {provider}: {tier}")
        configs.append(LLMEngineConfig(provider=provider, model=model, tier=tier))
    return tuple(configs)


def format_engine_labels(engines: tuple[LLMEngineConfig, ...] | list[LLMEngineConfig]) -> str:
    return ", ".join(f"{engine.provider}({engine.tier}:{engine.model})" for engine in engines)


def build_progress_checkpoints(total: int) -> tuple[int, ...]:
    if total <= 0:
        return ()
    fractions = (0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
    checkpoints = {1, total}
    checkpoints.update(max(1, math.ceil(total * fraction)) for fraction in fractions)
    return tuple(sorted(value for value in checkpoints if value <= total))


def should_log_sampled_response(completed: int, every_n: int = SAMPLED_RESPONSE_LOG_EVERY) -> bool:
    return completed == 1 or (every_n > 0 and completed % every_n == 0)


def _truncate_log_value(text: str, *, max_chars: int = SAMPLED_RESPONSE_LOG_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...<truncated>"


def log_sampled_llm_response(
    log: logging.Logger,
    *,
    stage_label: str,
    completed: int,
    input_hash: str | None,
    job_id: str | None,
    model: str | None,
    response_json: str | None,
    extra_fields: dict[str, object] | None = None,
    every_n: int = SAMPLED_RESPONSE_LOG_EVERY,
) -> None:
    if not should_log_sampled_response(completed, every_n):
        return

    payload_preview = "" if response_json is None else str(response_json)
    try:
        payload_preview = json.dumps(json.loads(payload_preview), ensure_ascii=False, sort_keys=True)
    except (TypeError, json.JSONDecodeError):
        pass

    meta_preview = "-"
    if extra_fields:
        meta_preview = json.dumps(extra_fields, ensure_ascii=False, sort_keys=True)

    log.info(
        "%s sample parsed response | completed=%s | job_id=%s | input_hash=%s | model=%s | meta=%s | payload=%s",
        stage_label,
        f"{completed:,}",
        job_id or "-",
        input_hash or "-",
        model or "-",
        _truncate_log_value(meta_preview),
        _truncate_log_value(payload_preview),
    )


def stderr_looks_like_error(text: str) -> bool:
    candidate = (text or "").lower()
    if not candidate:
        return False
    patterns = (
        "error",
        "failed",
        "exception",
        "traceback",
        "fatal",
        "no such file or directory",
        "syntax error",
        "permission denied",
    )
    return any(pattern in candidate for pattern in patterns)


def log_command_output(
    log: logging.Logger,
    *,
    level: int,
    summary: str,
    stdout: str,
    stderr: str,
    returncode: int | None = None,
) -> None:
    extra = ""
    if returncode is not None:
        extra = f" | returncode={returncode}"
    log.log(
        level,
        "%s%s | stderr=%s | stdout=%s",
        summary,
        extra,
        _truncate_log_value(stderr or "-", max_chars=COMMAND_ERROR_LOG_MAX_CHARS),
        _truncate_log_value(stdout or "-", max_chars=COMMAND_ERROR_LOG_MAX_CHARS),
    )


class LLMEngineRuntime:
    def __init__(
        self,
        engines: tuple[LLMEngineConfig, ...] | list[LLMEngineConfig],
        *,
        slot_timezone: str = DEFAULT_ENGINE_TIMEZONE,
        rng_seed: int | None = None,
    ) -> None:
        self.engines = tuple(engines)
        if not self.engines:
            raise ValueError("at least one engine config is required")
        providers = [engine.provider for engine in self.engines]
        if len(providers) != len(set(providers)):
            raise ValueError("engine configs must not repeat providers")
        self._engine_lookup = {engine.provider: engine for engine in self.engines}
        self._state = {engine.provider: LLMEngineState() for engine in self.engines}
        self._lock = threading.Lock()
        self._rng = random.Random(rng_seed)
        self._slot_timezone = ZoneInfo(slot_timezone)

    def _coerce_now(self, now: datetime | None = None) -> datetime:
        current = now or datetime.now(self._slot_timezone)
        if current.tzinfo is None:
            return current.replace(tzinfo=self._slot_timezone)
        return current.astimezone(self._slot_timezone)

    def _slot_window(self, now: datetime) -> tuple[str, float, int | None]:
        local_now = self._coerce_now(now)
        start_hour = min((local_now.hour // 5) * 5, 20)
        slot_start = local_now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        next_boundary = 24 if start_hour == 20 else start_hour + 5
        slot_end = slot_start.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=next_boundary)
        slot_index = {0: 0, 5: 1, 10: 2, 15: 3, 20: 4}[start_hour]
        if slot_index == 0:
            call_cap = None
        elif slot_index == 1:
            call_cap = 2_000
        else:
            call_cap = 1_000
        return slot_start.isoformat(timespec="seconds"), slot_end.timestamp(), call_cap

    def _refresh_state_for_slot_locked(
        self,
        engine: LLMEngineConfig,
        state: LLMEngineState,
        now: datetime,
    ) -> tuple[float | None, int | None]:
        if engine.tier != ENGINE_TIER_NON_INTRUSIVE:
            return None, None
        slot_key, slot_end_ts, call_cap = self._slot_window(now)
        if state.slot_key != slot_key:
            state.slot_key = slot_key
            state.slot_calls_started = 0
        return slot_end_ts, call_cap

    def _provider_wait_locked(self, provider: str, now: datetime) -> float:
        engine = self._engine_lookup[provider]
        state = self._state[provider]
        now_ts = self._coerce_now(now).timestamp()
        slot_end_ts, call_cap = self._refresh_state_for_slot_locked(engine, state, now)
        waits: list[float] = []
        if state.paused_until_ts > now_ts:
            waits.append(state.paused_until_ts - now_ts)
        if call_cap is not None and state.slot_calls_started >= call_cap and slot_end_ts is not None:
            waits.append(max(slot_end_ts - now_ts, 0.0))
        return max(min(waits), 0.0) if waits else 0.0

    def claim_next_engine(
        self,
        *,
        now: datetime | None = None,
        exclude_providers: set[str] | None = None,
    ) -> LLMEngineConfig | None:
        excluded = exclude_providers or set()
        current = self._coerce_now(now)
        with self._lock:
            available = [
                engine
                for engine in self.engines
                if engine.provider not in excluded and self._provider_wait_locked(engine.provider, current) <= 0
            ]
            if not available:
                return None
            chosen = self._rng.choice(available)
            if chosen.tier == ENGINE_TIER_NON_INTRUSIVE:
                self._state[chosen.provider].slot_calls_started += 1
            return chosen

    def next_available_delay(
        self,
        *,
        now: datetime | None = None,
        exclude_providers: set[str] | None = None,
    ) -> float | None:
        excluded = exclude_providers or set()
        current = self._coerce_now(now)
        waits: list[float] = []
        with self._lock:
            for engine in self.engines:
                if engine.provider in excluded:
                    continue
                wait_seconds = self._provider_wait_locked(engine.provider, current)
                if wait_seconds <= 0:
                    return 0.0
                waits.append(wait_seconds)
        return min(waits) if waits else None

    def provider_is_waiting(self, provider: str, *, now: datetime | None = None) -> bool:
        current = self._coerce_now(now)
        with self._lock:
            return self._provider_wait_locked(provider, current) > 0

    def claim_specific_engine(
        self,
        provider: str,
        *,
        now: datetime | None = None,
    ) -> LLMEngineConfig | None:
        current = self._coerce_now(now)
        with self._lock:
            if provider not in self._engine_lookup:
                raise ValueError(f"unknown engine provider: {provider}")
            if self._provider_wait_locked(provider, current) > 0:
                return None
            engine = self._engine_lookup[provider]
            if engine.tier == ENGINE_TIER_NON_INTRUSIVE:
                self._state[provider].slot_calls_started += 1
            return engine

    def provider_wait_delay(self, provider: str, *, now: datetime | None = None) -> float | None:
        current = self._coerce_now(now)
        with self._lock:
            if provider not in self._engine_lookup:
                raise ValueError(f"unknown engine provider: {provider}")
            wait_seconds = self._provider_wait_locked(provider, current)
        return wait_seconds

    def note_quota_hit(
        self,
        *,
        provider: str,
        quota_wait_hours: float,
        log: logging.Logger,
        input_hash: str,
        task_name: str,
        detail: str,
        now: datetime | None = None,
    ) -> None:
        current = self._coerce_now(now)
        now_ts = current.timestamp()
        with self._lock:
            engine = self._engine_lookup[provider]
            state = self._state[provider]
            slot_end_ts, _ = self._refresh_state_for_slot_locked(engine, state, current)
            if engine.tier == ENGINE_TIER_NON_INTRUSIVE and slot_end_ts is not None:
                paused_until_ts = slot_end_ts
            else:
                paused_until_ts = now_ts + quota_retry_after_seconds(quota_wait_hours)
            if paused_until_ts > state.paused_until_ts:
                state.paused_until_ts = paused_until_ts
            active_until_ts = state.paused_until_ts

        resume_at = datetime.fromtimestamp(active_until_ts, tz=timezone.utc)
        log.warning(
            "Quota/rate limit detected for %s/%s on %s (%s). Pausing %s until %s UTC.",
            provider,
            self._engine_lookup[provider].model,
            input_hash,
            task_name,
            provider,
            resume_at.isoformat(timespec="seconds"),
        )
        if detail:
            log.warning("Quota detail: %s", detail[:500])

def compute_description_hash(text) -> str:
    if pd.isna(text):
        text = ""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _normalize_hash_part(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def compute_input_hash(*parts) -> str:
    normalized = [_normalize_hash_part(part) for part in parts]
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_extraction_input_hash(title, company_name, raw_description) -> str:
    return compute_input_hash(title, company_name, raw_description)


def derive_classification_input(description_core_llm, description_core, description) -> str:
    for candidate in (description_core_llm, description_core, description):
        if candidate is None or pd.isna(candidate):
            continue
        text = str(candidate).strip()
        if text:
            return text
    return ""


def compute_classification_input_hash(title, company_name, classification_input) -> str:
    return compute_input_hash(title, company_name, classification_input)


def normalize_newlines(text) -> str:
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def canonical_heading(text: str) -> str:
    return re.sub(r"[:.]+$", "", text.strip().lower())


def is_ui_noise(line: str) -> bool:
    return line.strip().lower() in UI_NOISE_EXACT


def is_known_header(text: str) -> bool:
    return canonical_heading(text) in KNOWN_SECTION_HEADERS


def looks_like_heading(text: str) -> bool:
    cleaned = text.strip().strip("*#").strip()
    if not cleaned:
        return False
    if len(cleaned) > 70:
        return False
    if is_known_header(cleaned):
        return True
    if cleaned.endswith("..."):
        return True
    if cleaned.endswith(":") and len(cleaned.split()) <= 9:
        return True
    return bool(ALL_CAPS_HEADER_RE.fullmatch(cleaned) and len(cleaned.split()) <= 8)


def looks_like_label(line: str) -> bool:
    cleaned = line.strip()
    if not cleaned or len(cleaned) > 60:
        return False
    if looks_like_heading(cleaned):
        return True
    return bool(LABEL_VALUE_RE.fullmatch(cleaned) and len(cleaned.split()) <= 6)


def shield_abbreviations(text: str) -> str:
    protected = text
    for original, sentinel in ABBREVIATION_SENTINELS.items():
        protected = protected.replace(original, sentinel)
    protected = DOMAIN_RE.sub(lambda match: match.group(0).replace(".", "<DOT>"), protected)
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DECIMAL>", protected)
    return protected


def unshield_abbreviations(text: str) -> str:
    restored = text
    for original, sentinel in ABBREVIATION_SENTINELS.items():
        restored = restored.replace(sentinel, original)
    restored = restored.replace("<DOT>", ".")
    restored = restored.replace("<DECIMAL>", ".")
    return restored


def clean_description_for_unitizing(text) -> str:
    cleaned = normalize_newlines(text)
    cleaned = re.sub(r"(?im)^\s*show\s+(?:more|less)\s*$", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s*[•·]\s*", "\n- ", cleaned)
    cleaned = INLINE_HEADER_PATTERN.sub("\n", cleaned)
    cleaned = re.sub(
        r"(?<=[a-z0-9.)])(?=(?:Responsibilities|Qualifications|Requirements|Skills|Benefits|"
        r"About us|About Us|Job Details|Minimum Qualifications|Additional Responsibilities|"
        r"Core Skills)\s*:)",
        "\n",
        cleaned,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def merge_label_value_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or is_ui_noise(line):
            i += 1
            continue
        if i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if LEADING_COLON_VALUE_RE.match(nxt) and looks_like_label(line):
                merged.append(f"{line} {nxt}")
                i += 2
                continue
            if looks_like_label(line) and nxt and not looks_like_label(nxt) and len(nxt) <= 180:
                merged.append(f"{line} | {nxt}")
                i += 2
                continue
        merged.append(line)
        i += 1
    return merged


def classify_unitizer_shape(text: str) -> str:
    description = normalize_newlines(text)
    non_empty_lines = [line for line in description.split("\n") if line.strip()]
    bullet_count = description.count("·") + description.count("•")
    if len(non_empty_lines) <= 3 and len(description) >= 1200:
        return "compressed_blob"
    if bullet_count >= 4:
        return "inline_bullet_blob"
    label_like_count = sum(1 for line in non_empty_lines if looks_like_label(line))
    if label_like_count >= max(4, len(non_empty_lines) // 4):
        return "label_value"
    if len(non_empty_lines) >= 15:
        return "structured_lines"
    return "mixed"


def should_sentence_split(line: str, shape: str) -> bool:
    candidate = line.strip()
    if not candidate:
        return False
    sentence_marks = candidate.count(".") + candidate.count("?") + candidate.count("!")
    if shape in {"compressed_blob", "inline_bullet_blob"}:
        return sentence_marks >= 1 or len(candidate) > 180
    if len(candidate) > 420:
        return True
    if len(candidate) > 260 and sentence_marks >= 2:
        return True
    return False


def split_long_text_into_units(
    text: str,
    max_chars: int = MAX_EXTRACTION_UNIT_CHARS,
    *,
    shape: str = "mixed",
) -> list[str]:
    candidate = text.strip()
    if not candidate:
        return []
    protected = shield_abbreviations(candidate)
    parts = [unshield_abbreviations(part.strip()) for part in SENTENCE_SPLIT_RE.split(protected) if part.strip()]
    if len(parts) <= 1:
        if shape in {"compressed_blob", "inline_bullet_blob"} and len(candidate) > 240 and candidate.count(";") >= 3:
            return [part.strip() for part in candidate.split(";") if part.strip()]
        return [candidate]
    return parts


def chunk_oversize_units(parts: list[str], max_chars: int = MAX_EXTRACTION_UNIT_CHARS) -> list[str]:
    final_parts: list[str] = []
    for part in parts:
        if len(part) <= max_chars:
            final_parts.append(part)
            continue
        for subpart in re.split(r"(?<=[;:])\s+", part):
            subpart = subpart.strip()
            if not subpart:
                continue
            if len(subpart) <= max_chars:
                final_parts.append(subpart)
                continue
            words = subpart.split()
            buffer: list[str] = []
            current_len = 0
            for word in words:
                projected = current_len + len(word) + (1 if buffer else 0)
                if buffer and projected > max_chars:
                    final_parts.append(" ".join(buffer))
                    buffer = [word]
                    current_len = len(word)
                else:
                    buffer.append(word)
                    current_len = projected
            if buffer:
                final_parts.append(" ".join(buffer))
    return final_parts


def merge_micro_fragments(parts: list[str]) -> list[str]:
    merged: list[str] = []
    for part in parts:
        candidate = part.strip()
        if not candidate:
            continue
        if merged:
            prev = merged[-1]
            if len(candidate) < 28 and candidate[:1].islower() and len(prev) + len(candidate) + 1 <= 220:
                merged[-1] = f"{prev} {candidate}"
                continue
            if len(prev) < 28 and not looks_like_heading(prev) and len(prev) + len(candidate) + 1 <= 220:
                merged[-1] = f"{prev} {candidate}"
                continue
        merged.append(candidate)
    return merged


def expand_inline_delimiters(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped:
        return []
    if BULLET_LINE_RE.match(stripped):
        return [stripped]
    if stripped.count(" - ") >= 2 and len(stripped) > 180:
        return [part.strip() for part in stripped.split(" - ") if part.strip()]
    return [stripped]


def split_line_into_units(line: str, *, shape: str = "mixed") -> list[tuple[str, str]]:
    cleaned = line.strip()
    if not cleaned:
        return []
    if is_ui_noise(cleaned):
        return []
    if looks_like_heading(cleaned):
        return [("heading", cleaned)]
    if BULLET_LINE_RE.match(cleaned):
        return [("line", cleaned)]
    if METADATA_LINE_RE.match(cleaned) and len(cleaned) <= 180:
        return [("line", cleaned)]

    expanded_parts: list[str] = []
    for piece in expand_inline_delimiters(cleaned):
        if should_sentence_split(piece, shape):
            expanded_parts.extend(split_long_text_into_units(piece, shape=shape))
        else:
            expanded_parts.append(piece.strip())
    final_parts = merge_micro_fragments(chunk_oversize_units(expanded_parts))
    if len(final_parts) <= 1:
        return [("line", final_parts[0])] if final_parts else []
    return [("sentence", part) for part in final_parts]


def segment_description_into_units(text) -> list[dict]:
    description = clean_description_for_unitizing(text)
    if not description.strip():
        return []

    units: list[dict] = []
    shape = classify_unitizer_shape(text)
    raw_lines = [line.strip() for line in description.split("\n") if line.strip()]
    lines = merge_label_value_lines(raw_lines)

    for line in lines:
        if not line or is_ui_noise(line):
            continue
        for unit_type, piece in split_line_into_units(line, shape=shape):
            units.append(
                {
                    "unit_id": len(units) + 1,
                    "unit_type": unit_type,
                    "text": piece,
                }
            )

    return units


def join_retained_units(units: list[dict], boilerplate_unit_ids: list[int]) -> str:
    dropped_ids = set(boilerplate_unit_ids)
    selected = [unit["text"] for unit in units if unit["unit_id"] not in dropped_ids]
    return "\n".join(selected)


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


def ensure_cache_schema(conn: sqlite3.Connection) -> None:
    existing = conn.execute("PRAGMA table_info(responses)").fetchall()
    expected = {
        "input_hash",
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
            input_hash TEXT NOT NULL,
            task_name TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            model TEXT NOT NULL,
            response_json TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            tokens_used INTEGER,
            PRIMARY KEY (input_hash, task_name, prompt_version)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_responses_lookup
        ON responses (task_name, prompt_version, input_hash)
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Budget-constrained LLM selection
# ---------------------------------------------------------------------------
# Stages 9/10 accept a --llm-budget that caps NEW LLM calls across all data.
# The budget is split across SWE / SWE-adjacent / control categories, and
# within each category it is water-filled across scrape_date buckets so
# budget flows to thin days first.
BUDGET_CATEGORIES = ("swe", "swe_adjacent", "control")
DEFAULT_BUDGET_SPLIT = "0.4,0.3,0.3"


def _coerce_day(value: object) -> str:
    """Return `value` as a non-empty day string, or "unknown" for missing values.

    NaN is truthy in Python, so we check explicitly rather than using `or`.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    s = str(value).strip()
    return s or "unknown"


def stable_budget_score(scrape_date: str, input_hash: str) -> str:
    """Deterministic per-row score for budget-constrained within-day selection.

    Mirrors the stable_control_score pattern used for control cohort selection.
    Two runs with identical (scrape_date, input_hash) inputs produce identical
    scores, so selection is reproducible and incremental (higher budgets select
    supersets of lower budgets).
    """
    seed = f"budget-selection-v1|{scrape_date}|{input_hash}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def split_budget_by_category(
    budget: int,
    uncached_per_category: dict[str, int],
    shares: dict[str, float],
) -> dict[str, int]:
    """Split `budget` across categories using share-weighted proportional allocation.

    Each category gets `floor(budget * share / share_sum)`, capped by its
    uncached capacity. Floor remainders and any surplus from capped categories
    are distributed round-robin over the still-eligible categories in sorted
    order. Net effect: share ratios are preserved among eligible categories.

    Returns {category: number_allocated} with sum == budget (or less iff total
    capacity is exhausted).
    """
    allocation: dict[str, int] = {cat: 0 for cat in uncached_per_category}
    if budget <= 0:
        return allocation
    eligible = {cat: cap for cat, cap in uncached_per_category.items() if cap > 0}
    if not eligible:
        return allocation

    # Initial share-weighted distribution, capped by each category's capacity.
    share_sum = sum(shares.get(cat, 0.0) for cat in eligible) or 1.0
    for cat, cap in eligible.items():
        weight = shares.get(cat, 0.0) / share_sum
        allocation[cat] = min(int(math.floor(budget * weight)), cap)
    remaining = budget - sum(allocation.values())

    # Round-robin the remainder over still-eligible categories. Absorbs both
    # floor-rounding gaps and surplus cascaded from categories that hit their cap.
    sorted_cats = sorted(eligible)
    while remaining > 0:
        progressed = False
        for cat in sorted_cats:
            if allocation[cat] < eligible[cat]:
                allocation[cat] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    return allocation


def allocate_budget_across_days(
    uncached_per_day: dict[str, int],
    cached_per_day: dict[str, int],
    budget: int,
) -> dict[str, int]:
    """Water-fill `budget` across daily buckets, topping up the thinnest days first.

    Each day's "level" starts at its cached count. We pop the lowest-level day,
    add one unit, push it back at level+1, and repeat. Days with no uncached
    capacity are never pushed. Ties break by day name for deterministic output.

    Returns {day: number_allocated} with sum == budget, or sum < budget iff
    total capacity is exhausted.
    """
    if budget <= 0:
        return {day: 0 for day in uncached_per_day}

    allocation: dict[str, int] = {day: 0 for day in uncached_per_day}
    # Heap entries: (current_level, day). Only days with uncached capacity.
    heap = [
        (cached_per_day.get(day, 0), day)
        for day, capacity in uncached_per_day.items()
        if capacity > 0
    ]
    heapq.heapify(heap)

    remaining = budget
    while remaining > 0 and heap:
        level, day = heapq.heappop(heap)
        allocation[day] += 1
        remaining -= 1
        if allocation[day] < uncached_per_day[day]:
            heapq.heappush(heap, (level + 1, day))

    return allocation


def select_rows_by_budget(
    uncached_rows: list[dict],
    day_key: str,
    hash_key: str,
    allocation: dict[str, int],
) -> list[dict]:
    """Pick `allocation[day]` rows from each day's bucket, ordered by stable hash.

    Groups rows by `row[day_key]`, sorts each group by
    `stable_budget_score(day, row[hash_key])`, and takes the first N per day.
    Deterministic: same inputs → same output.
    """
    if not uncached_rows or not allocation:
        return []
    by_day: dict[object, list[dict]] = {}
    for row in uncached_rows:
        by_day.setdefault(row.get(day_key), []).append(row)
    selected: list[dict] = []
    for day, n in allocation.items():
        if n <= 0:
            continue
        day_rows = by_day.get(day, [])
        if not day_rows:
            continue
        day_str = str(day)
        scored = sorted(
            day_rows,
            key=lambda r: stable_budget_score(day_str, str(r.get(hash_key, ""))),
        )
        selected.extend(scored[:n])
    return selected


def parse_budget_split(raw: str) -> dict[str, float]:
    """Parse a comma-separated split like '0.4,0.3,0.3' into BUDGET_CATEGORIES.

    Values are normalized to sum to 1.0, so inputs like '40,30,30' also work.
    """
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != len(BUDGET_CATEGORIES):
        raise ValueError(
            f"budget split must have {len(BUDGET_CATEGORIES)} comma-separated "
            f"values (got {len(parts)}): {raw!r}"
        )
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"budget split values must be numeric: {raw!r}") from exc
    if any(v < 0 for v in values):
        raise ValueError(f"budget split values must be non-negative: {raw!r}")
    total = sum(values)
    if total <= 0:
        raise ValueError(f"budget split values must sum to > 0: {raw!r}")
    return {cat: v / total for cat, v in zip(BUDGET_CATEGORIES, values)}


def categorize_budget_candidate(row: dict) -> str | None:
    """Assign a candidate to one of BUDGET_CATEGORIES.

    Priority: SWE > SWE-adjacent > control. Returns None if the row matches
    none of the three flags (should not happen for routed candidates).
    """
    if bool(row.get("is_swe")):
        return "swe"
    if bool(row.get("is_swe_adjacent")):
        return "swe_adjacent"
    if bool(row.get("selected_for_control_cohort")):
        return "control"
    return None


def log_budget_plan(
    log: logging.Logger,
    *,
    budget: int,
    split: dict[str, float],
    uncached_per_category: dict[str, int],
    category_allocation: dict[str, int],
    deferred: int,
) -> None:
    """Emit the budget/allocation summary before LLM calls start."""
    log.info(
        "Budget plan | budget=%s | split=swe:%.2f/adj:%.2f/ctrl:%.2f",
        f"{budget:,}",
        split.get("swe", 0.0),
        split.get("swe_adjacent", 0.0),
        split.get("control", 0.0),
    )
    log.info(
        "Uncached | swe=%s | swe_adjacent=%s | control=%s | total=%s",
        f"{uncached_per_category.get('swe', 0):,}",
        f"{uncached_per_category.get('swe_adjacent', 0):,}",
        f"{uncached_per_category.get('control', 0):,}",
        f"{sum(uncached_per_category.values()):,}",
    )
    log.info(
        "Allocation | swe=%s | swe_adjacent=%s | control=%s | total=%s",
        f"{category_allocation.get('swe', 0):,}",
        f"{category_allocation.get('swe_adjacent', 0):,}",
        f"{category_allocation.get('control', 0):,}",
        f"{sum(category_allocation.values()):,}",
    )
    if deferred > 0:
        log.info("Deferred (budget-capped) | rows=%s", f"{deferred:,}")


def select_rows_with_budget(
    candidates: list[dict],
    cached_hashes: set[str],
    budget: int,
    split: dict[str, float],
    hash_key: str,
) -> tuple[list[dict], dict[str, int], dict[str, int]]:
    """Budget-constrained selection of candidate rows, two-level allocation.

    Level 1: split the budget across categories (40/30/30 by default), with
    surplus from capped categories cascading to others.
    Level 2: water-fill each category's allocation across scrape_date buckets.

    Each uncached candidate is normalized (scrape_date coerced to string) so
    downstream selection can group by it. Cached rows count toward the
    water-filling baseline but are never reprocessed.

    Returns: (selected_rows, category_allocation, uncached_per_category).
    """
    uncached_by_category: dict[str, list[dict]] = {cat: [] for cat in BUDGET_CATEGORIES}
    uncached_counts: dict[str, dict[str, int]] = {cat: {} for cat in BUDGET_CATEGORIES}
    cached_counts: dict[str, dict[str, int]] = {cat: {} for cat in BUDGET_CATEGORIES}

    for row in candidates:
        cat = categorize_budget_candidate(row)
        if cat is None:
            continue
        day = _coerce_day(row.get("scrape_date"))
        if str(row.get(hash_key)) in cached_hashes:
            cached_counts[cat][day] = cached_counts[cat].get(day, 0) + 1
        else:
            row_copy = dict(row)
            row_copy["scrape_date"] = day
            uncached_by_category[cat].append(row_copy)
            uncached_counts[cat][day] = uncached_counts[cat].get(day, 0) + 1

    uncached_per_category = {cat: len(rows) for cat, rows in uncached_by_category.items()}
    if budget <= 0:
        return [], {cat: 0 for cat in BUDGET_CATEGORIES}, uncached_per_category

    category_allocation = split_budget_by_category(
        budget=budget,
        uncached_per_category=uncached_per_category,
        shares=split,
    )

    selected: list[dict] = []
    for cat in BUDGET_CATEGORIES:
        cat_budget = category_allocation[cat]
        if cat_budget <= 0:
            continue
        day_alloc = allocate_budget_across_days(
            uncached_per_day=uncached_counts[cat],
            cached_per_day=cached_counts[cat],
            budget=cat_budget,
        )
        selected.extend(
            select_rows_by_budget(
                uncached_by_category[cat],
                day_key="scrape_date",
                hash_key=hash_key,
                allocation=day_alloc,
            )
        )
    return selected, category_allocation, uncached_per_category


# ---------------------------------------------------------------------------
# SQLite LLM response cache
# ---------------------------------------------------------------------------


def open_cache(cache_db: Path) -> sqlite3.Connection:
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cache_db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    ensure_cache_schema(conn)
    return conn


def fetch_cached_rows(
    conn: sqlite3.Connection,
    hashes: list[str] | None,
    task_name: str,
    prompt_version: str,
    *,
    input_hashes: list[str] | None = None,
    description_hashes: list[str] | None = None,
) -> dict[str, dict]:
    hashes = input_hashes or description_hashes or hashes or []
    rows: dict[str, dict] = {}
    if not hashes:
        return rows
    for batch in chunked(hashes, SQLITE_IN_LIMIT):
        placeholders = ",".join("?" for _ in batch)
        query = (
            "SELECT input_hash, task_name, model, prompt_version, response_json, timestamp, tokens_used "
            f"FROM responses WHERE task_name = ? AND prompt_version = ? "
            f"AND input_hash IN ({placeholders})"
        )
        params = [task_name, prompt_version, *batch]
        for row in conn.execute(query, params):
            rows[row[0]] = {
                "input_hash": row[0],
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
    hash_value: str | None = None,
    task_name: str | None = None,
    prompt_version: str | None = None,
    *,
    input_hash: str | None = None,
    description_hash: str | None = None,
) -> dict | None:
    resolved_hash = input_hash or description_hash or hash_value
    if resolved_hash is None or task_name is None or prompt_version is None:
        raise ValueError("input_hash is required")
    row = conn.execute(
        """
        SELECT input_hash, task_name, model, prompt_version, response_json, timestamp, tokens_used
        FROM responses
        WHERE input_hash = ? AND task_name = ? AND prompt_version = ?
        """,
        (resolved_hash, task_name, prompt_version),
    ).fetchone()
    if row is None:
        return None
    return {
        "input_hash": row[0],
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
    hash_value: str | None = None,
    task_name: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    response_json: str | None = None,
    tokens_used: int | None = None,
    *,
    input_hash: str | None = None,
    description_hash: str | None = None,
) -> None:
    resolved_hash = input_hash or description_hash or hash_value
    if (
        resolved_hash is None
        or task_name is None
        or model is None
        or prompt_version is None
        or response_json is None
    ):
        raise ValueError("input_hash is required")
    conn.execute(
        """
        INSERT OR REPLACE INTO responses
        (input_hash, task_name, prompt_version, model, response_json, timestamp, tokens_used)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            resolved_hash,
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
            repaired = repair_truncated_json_object(candidate[idx:])
            if repaired is None:
                continue
            obj = json.loads(repaired)
            return json.dumps(obj, ensure_ascii=False)
            continue
    raise ValueError("no_json_object_found")


def repair_truncated_json_object(text: str) -> str | None:
    lines = text.splitlines()
    for end in range(len(lines), 0, -1):
        prefix = "\n".join(lines[:end]).rstrip()
        repaired = repair_missing_closing_braces(prefix)
        if repaired is not None:
            return repaired
    return None


def repair_missing_closing_braces(text: str) -> str | None:
    candidate = text.rstrip()
    if not candidate.startswith("{") or candidate.endswith("}"):
        return None

    depth = 0
    in_string = False
    escape = False
    for char in candidate:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return None

    if in_string or depth <= 0:
        return None

    repaired = candidate + ("}" * depth)
    try:
        json.loads(repaired)
    except json.JSONDecodeError:
        return None
    return repaired


def looks_like_codex_json_event_stream(stdout: str) -> bool:
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return False
        return isinstance(event, dict) and isinstance(event.get("type"), str)
    return False


def parse_codex_usage_tokens(usage: object) -> int | None:
    if not isinstance(usage, dict):
        return None

    total = 0
    found = False
    for field_name in ("input_tokens", "output_tokens"):
        value = usage.get(field_name)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            total += value
            found = True

    if not found or total == 0:
        return None
    return total


def parse_codex_json_event_stream(stdout: str) -> tuple[str, int | None, float | None]:
    response_text = None
    tokens_used = None

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        event = json.loads(line)
        if not isinstance(event, dict):
            continue

        if event.get("type") == "item.completed":
            item = event.get("item")
            if (
                isinstance(item, dict)
                and item.get("type") == "agent_message"
                and isinstance(item.get("text"), str)
            ):
                response_text = item["text"]
        elif event.get("type") == "turn.completed":
            parsed_tokens = parse_codex_usage_tokens(event.get("usage"))
            if parsed_tokens is not None:
                tokens_used = parsed_tokens

    if response_text is None:
        raise ValueError("no_codex_agent_message")

    json_text = extract_first_json_object(response_text)
    return json_text, tokens_used, None


def parse_codex_stdout(stdout: str) -> tuple[str, int | None, float | None]:
    if looks_like_codex_json_event_stream(stdout):
        return parse_codex_json_event_stream(stdout)

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
    yoe_min_years = payload["yoe_min_years"]
    if yoe_min_years is not None:
        if isinstance(yoe_min_years, bool) or not isinstance(yoe_min_years, int):
            return "invalid_yoe_min_years"
        if yoe_min_years < 0 or yoe_min_years > 20:
            return "invalid_yoe_min_years"
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


def build_remote_ssh_master_check_command() -> list[str]:
    return [
        "ssh",
        "-i",
        REMOTE_SSH_KEY_PATH,
        "-o",
        "BatchMode=yes",
        "-o",
        f"ControlPath={REMOTE_SSH_CONTROL_PATH}",
        "-O",
        "check",
        REMOTE_SSH_HOST,
    ]


def build_remote_ssh_master_start_command() -> list[str]:
    return [
        "ssh",
        "-i",
        REMOTE_SSH_KEY_PATH,
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=yes",
        "-o",
        f"ControlPath={REMOTE_SSH_CONTROL_PATH}",
        "-o",
        f"ControlPersist={REMOTE_SSH_CONTROL_PERSIST}",
        "-N",
        "-f",
        REMOTE_SSH_HOST,
    ]


def ensure_remote_ssh_master(timeout_seconds: int) -> None:
    with REMOTE_SSH_MASTER_LOCK:
        check_result = subprocess.run(
            build_remote_ssh_master_check_command(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if check_result.returncode == 0:
            return

        start_result = subprocess.run(
            build_remote_ssh_master_start_command(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if start_result.returncode == 0:
            return

        raise subprocess.CalledProcessError(
            start_result.returncode,
            start_result.args,
            output=start_result.stdout,
            stderr=start_result.stderr,
        )


def build_remote_ssh_command(command: list[str]) -> list[str]:
    remote_cmd_string = shlex.join(command)
    return [
        "ssh",
        "-i",
        REMOTE_SSH_KEY_PATH,
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        f"ControlPath={REMOTE_SSH_CONTROL_PATH}",
        "-o",
        f"ControlPersist={REMOTE_SSH_CONTROL_PERSIST}",
        REMOTE_SSH_HOST,
        remote_cmd_string,
    ]


def call_subprocess(command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess:
    if _REMOTE_EXECUTION:
        ensure_remote_ssh_master(timeout_seconds)
        command = build_remote_ssh_command(command)

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
        "--disable",
        "shell_snapshot",
        "--full-auto",
        "--ephemeral",
        "--config",
        f"model={model}",
        "--config", "developer_instructions='You are a labor-market research tool. Return raw JSON.'",
        "--config", "model_reasoning_effort=medium",
        "--config", "model_verbosity=low",
        "--skip-git-repo-check",
        "--json",
        prompt,
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


def attempt_provider_call(
    provider: str,
    prompt: str,
    model: str,
    task_name: str,
    input_hash: str | None,
    error_log_path: Path,
    log: logging.Logger,
    timeout_seconds: int,
    max_retries: int,
    payload_validator: Callable[[dict], str | None],
    *,
    description_hash: str | None = None,
) -> tuple[dict | None, str | None, str]:
    backoff_seconds = [1, 2, 4]
    attempt = 1
    resolved_hash = input_hash or description_hash or ""
    stdout = ""
    stderr = ""
    combined_output = ""

    while attempt <= max_retries:
        try:
            t0 = time.time()
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
                log_command_output(
                    log,
                    level=logging.ERROR,
                    summary=(
                        f"Subprocess failed for {provider}/{model} on "
                        f"{resolved_hash or '-'} ({task_name})"
                    ),
                    stdout=stdout,
                    stderr=stderr,
                    returncode=result.returncode,
                )
                append_jsonl(
                    error_log_path,
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "input_hash": resolved_hash,
                        "description_hash": resolved_hash,
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
                    return None, "quota", combined_output
                if attempt < max_retries:
                    time.sleep(backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)])
                    attempt += 1
                    continue
                return None, "failed", combined_output

            if stderr_looks_like_error(stderr):
                log_command_output(
                    log,
                    level=logging.WARNING,
                    summary=(
                        f"Command stderr for {provider}/{model} on "
                        f"{resolved_hash or '-'} ({task_name})"
                    ),
                    stdout=stdout,
                    stderr=stderr,
                    returncode=result.returncode,
                )

            if provider == "codex":
                response_json, tokens_used, cost_usd = parse_codex_stdout(stdout)
                response_model = model
            else:
                response_json, tokens_used, cost_usd = parse_claude_stdout(stdout)
                response_model = model

            payload = json.loads(response_json)
            validation_error = payload_validator(payload)
            if validation_error is not None:
                log_command_output(
                    log,
                    level=logging.ERROR,
                    summary=(
                        f"Provider response validation failed for {provider}/{response_model} on "
                        f"{resolved_hash or '-'} ({task_name}): {validation_error}"
                    ),
                    stdout=stdout,
                    stderr=stderr,
                )
                append_jsonl(
                    error_log_path,
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "input_hash": resolved_hash,
                        "description_hash": resolved_hash,
                        "task_name": task_name,
                        "provider": provider,
                        "model": response_model,
                        "attempt": attempt,
                        "error_type": validation_error,
                        "stderr": stderr[:2000],
                        "raw_response": stdout[:8000],
                    },
                )
                if attempt < max_retries:
                    time.sleep(backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)])
                    attempt += 1
                    continue
                return None, "failed", stdout[:8000]

            return {
                "provider": provider,
                "model": response_model,
                "latency_seconds": latency,
                "response_json": json.dumps(payload, ensure_ascii=False),
                "payload": payload,
                "tokens_used": tokens_used,
                "cost_usd": cost_usd,
            }, None, ""
        except subprocess.TimeoutExpired:
            log.error(
                "Subprocess timed out for %s/%s on %s (%s) after %ss.",
                provider,
                model,
                resolved_hash or "-",
                task_name,
                timeout_seconds,
            )
            append_jsonl(
                error_log_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "input_hash": resolved_hash,
                    "description_hash": resolved_hash,
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
            return None, "failed", ""
        except (json.JSONDecodeError, ValueError) as exc:
            log_command_output(
                log,
                level=logging.ERROR,
                summary=(
                    f"Provider output parsing failed for {provider}/{model} on "
                    f"{resolved_hash or '-'} ({task_name}): {type(exc).__name__}"
                ),
                stdout=stdout,
                stderr=stderr,
            )
            append_jsonl(
                error_log_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "input_hash": resolved_hash,
                    "description_hash": resolved_hash,
                    "task_name": task_name,
                    "provider": provider,
                    "model": model,
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "stderr": stderr[:2000],
                    "raw_response": stdout[:8000] if "stdout" in locals() else "",
                },
            )
            raw_text = stdout[:8000] if "stdout" in locals() else ""
            if detect_quota_or_rate_limit(raw_text):
                return None, "quota", raw_text
            if attempt < max_retries:
                time.sleep(backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)])
                attempt += 1
                continue
            return None, "failed", raw_text
        except Exception:  # noqa: BLE001
            log.exception("Provider %s failed for %s (%s)", provider, resolved_hash, task_name)
            append_jsonl(
                error_log_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "input_hash": resolved_hash,
                    "description_hash": resolved_hash,
                    "task_name": task_name,
                    "provider": provider,
                    "model": model,
                    "attempt": attempt,
                    "error_type": "Exception",
                    "raw_response": "",
                },
            )
            if attempt < max_retries:
                time.sleep(backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)])
                attempt += 1
                continue
            return None, "failed", ""

    return None, "failed", ""


def execute_task_with_runtime(
    *,
    runtime: LLMEngineRuntime,
    task_name: str,
    prompt: str,
    input_hash: str,
    error_log_path: Path,
    log: logging.Logger,
    timeout_seconds: int,
    max_retries: int,
    payload_validator: Callable[[dict], str | None],
    retry_sleep_seconds: float = DEFAULT_RETRY_SLEEP_SECONDS,
    quota_wait_hours: float = DEFAULT_QUOTA_WAIT_HOURS,
) -> dict | None:
    while True:
        engine = runtime.claim_next_engine()
        if engine is None:
            wait_seconds = runtime.next_available_delay()
            if wait_seconds is None:
                return None
            if wait_seconds > 0:
                time.sleep(min(wait_seconds, retry_sleep_seconds))
                continue
            return None

        while True:
            result, failure_kind, detail = attempt_provider_call(
                provider=engine.provider,
                prompt=prompt,
                model=engine.model,
                task_name=task_name,
                input_hash=input_hash,
                error_log_path=error_log_path,
                log=log,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                payload_validator=payload_validator,
            )
            if result is not None:
                return result
            if STOP_REQUESTED:
                return None

            if failure_kind == "quota":
                runtime.note_quota_hit(
                    provider=engine.provider,
                    quota_wait_hours=quota_wait_hours,
                    log=log,
                    input_hash=input_hash,
                    task_name=task_name,
                    detail=detail,
                )
            else:
                log.warning(
                    "Call failed for %s on %s (%s). Waiting %.0fs before retrying the same engine.",
                    input_hash,
                    engine.provider,
                    task_name,
                    retry_sleep_seconds,
                )
                time.sleep(retry_sleep_seconds)

            while True:
                claimed_engine = runtime.claim_specific_engine(engine.provider)
                if claimed_engine is not None:
                    engine = claimed_engine
                    break
                wait_seconds = runtime.provider_wait_delay(engine.provider)
                if wait_seconds is None:
                    return None
                if wait_seconds > 0:
                    time.sleep(min(wait_seconds, retry_sleep_seconds))
                    continue
                time.sleep(retry_sleep_seconds)


def try_provider(
    provider: str,
    prompt: str,
    model: str,
    task_name: str,
    input_hash: str | None,
    error_log_path: Path,
    log: logging.Logger,
    timeout_seconds: int,
    max_retries: int,
    payload_validator: Callable[[dict], str | None],
    *,
    description_hash: str | None = None,
    quota_wait_hours: float = DEFAULT_QUOTA_WAIT_HOURS,
) -> dict | None:
    wait_seconds = quota_retry_after_seconds(quota_wait_hours)
    resolved_hash = input_hash or description_hash or ""

    while True:
        wait_for_quota_pause()
        if STOP_REQUESTED:
            return None
        result, failure_kind, detail = attempt_provider_call(
            provider=provider,
            prompt=prompt,
            model=model,
            task_name=task_name,
            input_hash=input_hash,
            error_log_path=error_log_path,
            log=log,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            payload_validator=payload_validator,
            description_hash=description_hash,
        )
        if result is not None:
            return result
        if failure_kind == "quota":
            activate_quota_pause(
                provider=provider,
                model=model,
                wait_seconds=wait_seconds,
                log=log,
                description_hash=resolved_hash,
                task_name=task_name,
                detail=detail,
            )
            continue
        return None


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


def validate_extraction_selection(
    original_description: str,
    payload: dict,
) -> dict:
    units = segment_description_into_units(original_description)
    task_status = payload.get("task_status")
    boilerplate_unit_ids = payload.get("boilerplate_unit_ids", []) or []
    uncertain_unit_ids = payload.get("uncertain_unit_ids", []) or []
    model_reason = payload.get("reason", "") or ""

    normalized_drop_ids = []
    invalid_drop_ids = []
    for value in boilerplate_unit_ids:
        if isinstance(value, bool):
            invalid_drop_ids.append(value)
            continue
        try:
            unit_id = int(value)
        except (TypeError, ValueError):
            invalid_drop_ids.append(value)
            continue
        if unit_id < 1 or unit_id > len(units):
            invalid_drop_ids.append(unit_id)
            continue
        normalized_drop_ids.append(unit_id)
    normalized_drop_ids = sorted(set(normalized_drop_ids))

    normalized_uncertain_ids = []
    invalid_uncertain_ids = []
    for value in uncertain_unit_ids:
        if isinstance(value, bool):
            invalid_uncertain_ids.append(value)
            continue
        try:
            unit_id = int(value)
        except (TypeError, ValueError):
            invalid_uncertain_ids.append(value)
            continue
        if unit_id < 1 or unit_id > len(units):
            invalid_uncertain_ids.append(unit_id)
            continue
        normalized_uncertain_ids.append(unit_id)
    normalized_uncertain_ids = sorted(set(normalized_uncertain_ids))

    overlap_ids = sorted(set(normalized_drop_ids) & set(normalized_uncertain_ids))
    reconstructed_text = (
        ""
        if task_status != EXTRACTION_STATUS_OK
        else join_retained_units(units, normalized_drop_ids)
    )
    total_chars = sum(len(unit["text"]) for unit in units)
    dropped_chars = sum(len(unit["text"]) for unit in units if unit["unit_id"] in set(normalized_drop_ids))
    drop_ratio_chars = None if total_chars == 0 else dropped_chars / total_chars
    kept_unit_ids = [unit["unit_id"] for unit in units if unit["unit_id"] not in set(normalized_drop_ids)]

    reason = "ok"
    passed = True
    if not units:
        passed = False
        reason = "no_units"
    elif invalid_drop_ids or invalid_uncertain_ids:
        passed = False
        reason = "invalid_unit_ids"
    elif overlap_ids:
        passed = False
        reason = "overlapping_unit_ids"
    elif task_status == EXTRACTION_STATUS_CANNOT_COMPLETE:
        passed = False
        reason = "cannot_complete"
    elif task_status != EXTRACTION_STATUS_OK:
        passed = False
        reason = "invalid_task_status"
    elif not reconstructed_text.strip():
        passed = False
        reason = "all_units_dropped"

    return {
        "passed": passed,
        "reason": reason,
        "task_status": task_status,
        "unit_count": len(units),
        "single_unit": len(units) == 1,
        "boilerplate_unit_ids": normalized_drop_ids,
        "uncertain_unit_ids": normalized_uncertain_ids,
        "invalid_drop_ids": invalid_drop_ids,
        "invalid_uncertain_ids": invalid_uncertain_ids,
        "overlap_ids": overlap_ids,
        "kept_unit_ids": kept_unit_ids,
        "drop_ratio_chars": drop_ratio_chars,
        "model_reason": model_reason,
        "reconstructed_text": reconstructed_text,
        "similarity": 1.0 if passed else 0.0,
    }


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[94]
