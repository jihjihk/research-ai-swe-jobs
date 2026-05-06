"""
Cluster naming via OpenAI Responses API (§5.1).

`propose_label(top_words, exemplars, *, model)` calls one OpenAI completion
per cluster and returns `{label, confidence, alternative}`. The prompt is
verbatim from §5.1 of `design.md`.

Auth follows the same `~/.config/job-research/openai.env` pattern as the
preprocessing pipeline.
"""

from __future__ import annotations

import json
import os
import shlex
import time
from typing import Any

import httpx

from figures.bertopic import config


_RESPONSES_URL = "https://api.openai.com/v1/responses"
_OPENAI_ENV_KEYS = {"OPENAI_API_KEY", "OPENAI_ORGANIZATION", "OPENAI_PROJECT"}


def _load_openai_env() -> None:
    if not config.OPENAI_ENV_FILE.exists():
        raise RuntimeError(
            f"OpenAI env file missing at {config.OPENAI_ENV_FILE}"
        )
    for raw in config.OPENAI_ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, _, raw_value = line.partition("=")
        key = key.strip()
        if key not in _OPENAI_ENV_KEYS or not raw_value.strip():
            continue
        parts = shlex.split(raw_value.strip(), posix=True)
        if len(parts) == 1:
            os.environ.setdefault(key, parts[0])


def _openai_headers(*, request_id: str) -> dict[str, str]:
    _load_openai_env()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    org = os.environ.get("OPENAI_ORGANIZATION", "").strip()
    project = os.environ.get("OPENAI_PROJECT", "").strip()
    if org:
        headers["OpenAI-Organization"] = org
    if project:
        headers["OpenAI-Project"] = project
    headers["X-Client-Request-Id"] = f"job-research:bertopic-naming:{request_id}"
    return headers


PROMPT_TEMPLATE = (
    "You are labeling a cluster of software-engineering job descriptions. "
    "Given these top words: {top_words}, and these representative posting "
    "excerpts (titles + first 200 chars of description_core_llm): "
    "{exemplars}, produce a 2-4 word noun-phrase label that names the role "
    "family or sub-archetype. Do not invent vocabulary not present in the "
    "words or excerpts. Output JSON: "
    '{{"label": str, "confidence": 0-1, "alternative": str}}.'
)


def _format_exemplars(exemplars: list[tuple[str, str]]) -> str:
    pieces = []
    for i, (title, snippet) in enumerate(exemplars, 1):
        clean_snippet = snippet.replace("\n", " ").strip()[:200]
        pieces.append(f"({i}) {title.strip()}: {clean_snippet}")
    return " | ".join(pieces)


def propose_label(
    *,
    top_words: list[str],
    exemplars: list[tuple[str, str]],
    model: str = config.LLM_MODEL_PRIMARY,
    request_id: str = "smoke",
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call the OpenAI Responses API once and return the parsed JSON label."""
    prompt = PROMPT_TEMPLATE.format(
        top_words=", ".join(top_words),
        exemplars=_format_exemplars(exemplars),
    )
    payload = {
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "text": {"format": {"type": "json_object"}},
    }

    headers = _openai_headers(request_id=request_id)
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = httpx.post(
                _RESPONSES_URL, headers=headers, json=payload, timeout=120.0,
            )
        except httpx.HTTPError as exc:
            last_error = exc
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 200:
            return _extract_json(response.json())
        if response.status_code in {429, 500, 502, 503, 504}:
            last_error = RuntimeError(
                f"transient {response.status_code}: {response.text[:300]}"
            )
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(
            f"naming call failed {response.status_code}: {response.text[:500]}"
        )

    raise RuntimeError(f"naming retries exhausted: {last_error}")


def _extract_json(body: dict[str, Any]) -> dict[str, Any]:
    """Pull the JSON object the model returned from a Responses-API payload."""
    output = body.get("output_text")
    if isinstance(output, str) and output.strip():
        return _parse_label_json(output)

    blocks = body.get("output") or []
    for block in blocks:
        for part in block.get("content", []) or []:
            text = part.get("text") if isinstance(part, dict) else None
            if isinstance(text, str) and text.strip():
                return _parse_label_json(text)

    choices = body.get("choices") or []
    for choice in choices:
        message = choice.get("message") if isinstance(choice, dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if isinstance(text, str) and text.strip():
            return _parse_label_json(text)

    raise RuntimeError(f"no text in naming response: {json.dumps(body)[:500]}")


def _parse_label_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # strip markdown fence
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"naming response was not JSON: {text[:300]}"
        ) from exc
    if "label" not in parsed:
        raise RuntimeError(f"naming JSON missing 'label': {parsed}")
    return parsed
