"""Single-posting LLM classifier — wraps OpenAI Responses API + structured output.

Reuses the preprocessing pipeline's auth convention (~/.config/job-research/openai.env).
Self-contained — does NOT import from preprocessing/ to keep this folder isolated until
it graduates to a pipeline stage.

Usage:
    from classifier import classify, Labels
    result = classify(title="...", description="...", model="gpt-5.4-nano", seed=42)
    # result = {"labels": [...], "raw_response": "...", "latency_s": 1.2, "request_id": "...",
    #          "cached_tokens": int, "input_tokens": int, "output_tokens": int}
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx

# ---- Constants ----
OPENAI_RESPONSES_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_ENV_FILE = "~/.config/job-research/openai.env"
OPENAI_ENV_FILE_OVERRIDE = "JOB_RESEARCH_OPENAI_ENV_FILE"
OPENAI_ENV_KEYS = ("OPENAI_API_KEY", "OPENAI_ORGANIZATION", "OPENAI_PROJECT")

LABELS: tuple[str, ...] = (
    "people_management",
    "orchestration",
    "verification",
    "mentorship",
    "performance",
    "process_scaffolding",
    "legacy_stack",
    "context_infrastructure",
)

# ---- Prompt (must mirror prompt_v1.md exactly) ----
PROMPT_VERSION = "v1"
SYSTEM_PROMPT = """You tag SWE job postings with which of 8 themes are explicitly present.

A theme applies only if the posting explicitly states it as a responsibility, requirement, or named skill — not as boilerplate, passing mention, or company description.

Themes:
- people_management: formal people-management authority (direct reports, performance reviews, hiring/firing, headcount, 1:1 cadence).
- orchestration: authoring specs, decomposing work, system architecture, OR orchestrating AI/agents (context engineering, multi-agent coordination, agent harnesses).
- verification: tests, CI/CD, code review for correctness, evals, observability as correctness gating, compliance/audit.
- mentorship: teaching, growing, or shepherding other engineers as a peer/IC (not as a manager).
- performance: low-level performance optimization (profiling, latency, throughput, kernel/network internals) OR explicit demands for deep technical depth.
- process_scaffolding: agile/scrum/sprints, requirements engineering, V&V, project coordination, SDLC governance.
- legacy_stack: required experience with legacy enterprise stacks (.NET Framework, COBOL, mainframe, Java EE, VMware/vSphere, Active Directory, etc.).
- context_infrastructure: documentation, runbooks, telemetry/observability stack hygiene, ADRs/RFCs, cross-functional coordination as substrate work.

Output a JSON array of slugs. Empty array if none. No commentary."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {"type": "string", "enum": list(LABELS)},
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}


log = logging.getLogger(__name__)


# ---------- Auth ----------

def _resolve_env_file() -> Path:
    raw = os.environ.get(OPENAI_ENV_FILE_OVERRIDE, DEFAULT_OPENAI_ENV_FILE)
    return Path(raw).expanduser()


def _parse_env_line(line: str) -> tuple[str, str] | None:
    candidate = line.strip()
    if not candidate or candidate.startswith("#"):
        return None
    if candidate.startswith("export "):
        candidate = candidate[len("export "):].strip()
    name, sep, raw = candidate.partition("=")
    if not sep:
        return None
    key = name.strip()
    if key not in OPENAI_ENV_KEYS:
        return None
    value = raw.strip()
    if not value:
        return key, ""
    parts = shlex.split(value, posix=True)
    if len(parts) != 1:
        raise RuntimeError(f"invalid value for {key} in {_resolve_env_file()}")
    return key, parts[0]


def _load_env_file() -> Path:
    env_file = _resolve_env_file()
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(line)
            if parsed and parsed[1]:
                os.environ.setdefault(parsed[0], parsed[1])
    return env_file


def _get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    env_file = _load_env_file()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError(f"OPENAI_API_KEY not set. Put it in {env_file} or export it.")


def _build_headers() -> dict[str, str]:
    api_key = _get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    _load_env_file()
    org = os.environ.get("OPENAI_ORGANIZATION", "").strip()
    project = os.environ.get("OPENAI_PROJECT", "").strip()
    if org:
        headers["OpenAI-Organization"] = org
    if project:
        headers["OpenAI-Project"] = project
    return headers


# ---------- Payload + parsing ----------

def _build_payload(
    *,
    system_prompt: str,
    user_text: str,
    model: str,
    seed: int | None,
) -> dict:
    # Note: OpenAI Responses API does not currently accept `seed`; we keep the
    # parameter on the public API for documentation/forward-compat but do not
    # forward it. Stability across reps comes from low-effort reasoning + the
    # deterministic structured-output schema rather than an explicit seed.
    return {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
        ],
        "reasoning": {"effort": "low"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "swe_topic_classification",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
    }


def _iter_json_text_candidates(value: object) -> Iterable[str]:
    """Mirrors llm_shared.py — Responses API returns nested output, dig out JSON text."""
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_json_text_candidates(item)
        return
    if not isinstance(value, dict):
        return
    for key in ("output_text", "text", "content", "value"):
        cand = value.get(key)
        if isinstance(cand, str):
            yield cand
    for cand in value.values():
        yield from _iter_json_text_candidates(cand)


def _extract_json(response_body: dict) -> tuple[dict | None, str | None]:
    for text in _iter_json_text_candidates(response_body):
        if not text or not text.strip():
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "labels" in parsed:
            return parsed, text
    return None, None


# ---------- Public API ----------

@dataclass
class ClassifyResult:
    labels: list[str]
    parsed_ok: bool
    raw_text: str | None
    latency_s: float
    request_id: str
    model: str
    seed: int | None
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "labels": self.labels,
            "parsed_ok": self.parsed_ok,
            "raw_text": self.raw_text,
            "latency_s": round(self.latency_s, 3),
            "request_id": self.request_id,
            "model": self.model,
            "seed": self.seed,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "error": self.error,
        }


def build_user_text(title: str, description: str, *, max_chars: int = 6000) -> str:
    title = (title or "").strip()
    desc = (description or "").strip()
    if len(desc) > max_chars:
        desc = desc[:max_chars].rstrip() + " ... [truncated]"
    return f"Title: {title}\nDescription: {desc}"


def classify(
    *,
    title: str,
    description: str,
    model: str,
    seed: int | None = None,
    timeout_s: float = 60.0,
    system_prompt_override: str | None = None,
    label_order_override: tuple[str, ...] | None = None,
) -> ClassifyResult:
    """Classify a single posting.

    `system_prompt_override` and `label_order_override` are for stability tests; leave None for production calls.
    """
    sys_prompt = system_prompt_override if system_prompt_override is not None else SYSTEM_PROMPT
    user_text = build_user_text(title, description)
    payload = _build_payload(system_prompt=sys_prompt, user_text=user_text, model=model, seed=seed)
    if label_order_override is not None:
        payload["text"]["format"]["schema"] = {
            **OUTPUT_SCHEMA,
            "properties": {
                "labels": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(label_order_override)},
                }
            },
        }

    headers = _build_headers()
    request_id = uuid.uuid4().hex
    headers.setdefault("X-Request-ID", request_id)

    t0 = time.time()
    error: str | None = None
    body: dict = {}
    try:
        resp = httpx.post(OPENAI_RESPONSES_API_URL, headers=headers, json=payload, timeout=timeout_s)
        latency = time.time() - t0
        if resp.status_code >= 400:
            error = f"HTTP {resp.status_code}: {resp.text[:500]}"
            return ClassifyResult(
                labels=[],
                parsed_ok=False,
                raw_text=resp.text[:2000],
                latency_s=latency,
                request_id=request_id,
                model=model,
                seed=seed,
                input_tokens=None,
                output_tokens=None,
                cached_tokens=None,
                error=error,
            )
        body = resp.json()
    except Exception as e:
        latency = time.time() - t0
        return ClassifyResult(
            labels=[],
            parsed_ok=False,
            raw_text=None,
            latency_s=latency,
            request_id=request_id,
            model=model,
            seed=seed,
            input_tokens=None,
            output_tokens=None,
            cached_tokens=None,
            error=f"{type(e).__name__}: {e}",
        )

    parsed, raw_text = _extract_json(body)
    usage = body.get("usage", {}) or {}
    in_tok = usage.get("input_tokens") or usage.get("prompt_tokens")
    out_tok = usage.get("output_tokens") or usage.get("completion_tokens")
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens")
    if cached is None:
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")

    if parsed is None:
        return ClassifyResult(
            labels=[],
            parsed_ok=False,
            raw_text=raw_text or json.dumps(body)[:2000],
            latency_s=latency,
            request_id=request_id,
            model=model,
            seed=seed,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_tokens=cached,
            error="parse_failed",
        )

    labels = parsed.get("labels", [])
    if not isinstance(labels, list):
        labels = []
    # filter to known labels only
    valid = [l for l in labels if l in LABELS]
    return ClassifyResult(
        labels=sorted(valid),
        parsed_ok=True,
        raw_text=raw_text,
        latency_s=latency,
        request_id=request_id,
        model=model,
        seed=seed,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cached_tokens=cached,
        error=None,
    )


if __name__ == "__main__":
    # Smoke test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--title", default="Senior Software Engineer, Platform")
    parser.add_argument("--description", default=(
        "We are looking for a senior engineer to lead the design of our distributed services. "
        "You will mentor a team of 5 engineers, run code reviews, write architecture decision records, "
        "and own latency/throughput optimization for our hot path. Required: 7+ years Go, deep "
        "understanding of distributed systems, Kafka, Kubernetes."
    ))
    args = parser.parse_args()
    result = classify(title=args.title, description=args.description, model=args.model, seed=42)
    print(json.dumps(result.to_dict(), indent=2))
