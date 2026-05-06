"""V3 classifier — supports three prompt variants for the v3 pilot.

Variants:
  - "skill"        : skill-only (8 themes), output {"skill_themes":[...]}
  - "role_family"  : role-family-only (17 families), output {"role_families":[...]}
  - "combined"     : both axes, output {"skill_themes":[...], "role_families":[...]}

Reuses auth + HTTP plumbing from classifier.py via direct import. Keeps
classifier.py (v1) callable for back-compat with the existing pilot data.

Public API:
    from classifier_v3 import classify_v3, VARIANT_PROMPTS
    result = classify_v3(variant="combined", title=..., description=..., model="gpt-5.4-mini")
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import httpx

# Reuse auth + low-level helpers from classifier.py
from classifier import (  # noqa: E402
    OPENAI_RESPONSES_API_URL,
    _build_headers,
    _iter_json_text_candidates,
    build_user_text,
)


# ---- Label sets ----
SKILL_LABELS: tuple[str, ...] = (
    "people_management",
    "orchestration",
    "verification",
    "mentorship",
    "performance",
    "process_scaffolding",
    "legacy_stack",
    "context_infrastructure",
)

ROLE_FAMILY_LABELS: tuple[str, ...] = (
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
)


# ---- System prompts (must mirror prompt_*.md exactly) ----

SYSTEM_PROMPT_SKILL = """You tag SWE job postings with which of 8 skill themes are explicitly named as responsibilities, requirements, or skills. Boilerplate, passing mentions, and company descriptions are not evidence.

Themes:
- people_management: direct reports, performance reviews, hiring/firing authority, headcount, 1:1 cadence. NOT bare "lead a team" / "tech lead" / "lead engineer".
- orchestration: authoring specs, ADRs, RFCs, or design docs; decomposing work for engineers or AI agents; multi-agent or context-engineering systems. NOT design patterns as required knowledge (MVC, MVVM); NOT "architecture experience" without authoring.
- verification: CI/CD pipelines, named test frameworks, code-review processes, evals, observability for regressions, compliance/audit, static analysis. NOT generic "writes tests".
- mentorship: mentoring, teaching, growing, or onboarding other engineers; pair programming; knowledge transfer. NOT managing a team (that's people_management).
- performance: profiling, latency, throughput, kernel/network internals, low-level optimization, "deep understanding of [a technical area]". NOT "high-performing team"; NOT "expert in [tool]" as recruiter fluff.
- process_scaffolding: agile, scrum, sprints, requirements engineering, V&V, project coordination, SDLC governance.
- legacy_stack: required experience with old-paradigm enterprise frameworks regardless of version — Java / Java EE, .NET / .NET Framework, ASP.NET, COBOL, mainframe, VMware / vSphere, Active Directory, BizTalk, ColdFusion, and similar.
- context_infrastructure: authoring runbooks, ADRs, RFCs, dashboards / SLOs; telemetry hygiene; schema documentation; technical writing. NOT generic "cross-functional collaboration" or "communication skills".

Reminder: tag only what the posting explicitly names. Multi-label allowed. Empty array means none. Output JSON with one array `skill_themes`, no commentary."""

SYSTEM_PROMPT_ROLE_FAMILY = """You tag SWE job postings with which of 17 engineering role families fit the posting. Multi-label allowed. Use only what the posting clearly indicates is the role's focus, not every adjacent technology mentioned.

Families:
- software_engineer_general: residual fallback for SWE postings without a clearer specialization. NEVER use alongside another family — only when no other fits.
- frontend_web: user-facing web/mobile UI/UX work. Full-stack roles default here.
- backend_api: server-side logic, APIs, databases, services.
- mobile: native or cross-platform mobile apps (iOS, Android, React Native, Flutter).
- embedded: firmware, IoT, hardware-near software, RTOS, FPGA.
- data_engineer: data pipelines, warehouses, ingestion infrastructure.
- ml_engineer: pre-LLM ML pipelines, model training, productionising ML systems, MLOps.
- ai_llm_engineer: LLM-based or agent systems — prompt engineering, RAG, foundation-model integration, multi-agent.
- devops_sre_platform: CI/CD, infrastructure-as-code, reliability, platform engineering, SRE.
- security: application or infrastructure security; AppSec, InfoSec, cyber.
- qa_test: software quality, testing, validation as the role's primary focus (QA Engineer, SDET, Test Automation).
- solutions_field: deploys / integrates with customers (pre- and post-sales) — Solutions Engineer, Field Applications, Customer Engineer, Forward-Deployed Engineer.
- legacy_specialist: maintains older or vendor-locked stacks — Mainframe, COBOL, classic .NET, Salesforce, ServiceNow, SAP/ABAP, Oracle Forms. Apply when the role's primary purpose is maintaining a legacy stack, not just one of many required skills.
- data_analytics: analytical, operational, or decision-support outputs — Data Scientist, Data Analyst, Analytics Engineer, BI. Not production ML pipelines.
- research: applied or basic research bridging academia and product — Research Scientist, Applied Scientist, Research Engineer.
- infra_ops_admin: operates / administrates databases, servers, networks rather than authoring software — DBA, sysadmin, network engineer, cloud admin.
- people_manager: engineering management focused on people, hiring, team operations. Engineering Manager, Software Engineering Manager, Development Manager. Excludes director/VP/C-level.

Reminder: title is dispositive when explicit; description-driven only when the title is generic. Output JSON with one array `role_families`. Empty array means none. No commentary."""

SYSTEM_PROMPT_COMBINED = """You tag SWE job postings on TWO axes: skill themes (8) and role families (17). Both are multi-label. Tag only what the posting explicitly names as a responsibility, requirement, or skill — not boilerplate, passing mentions, or company descriptions.

SKILL THEMES — what work the role does:
- people_management: direct reports, performance reviews, hiring/firing authority, headcount, 1:1 cadence. NOT bare "lead a team" / "tech lead" / "lead engineer".
- orchestration: authoring specs, ADRs, RFCs, or design docs; decomposing work for engineers or AI agents; multi-agent or context-engineering systems. NOT design patterns as required knowledge (MVC, MVVM); NOT "architecture experience" without authoring.
- verification: CI/CD pipelines, named test frameworks, code-review processes, evals, observability for regressions, compliance/audit, static analysis. NOT generic "writes tests".
- mentorship: mentoring, teaching, growing, or onboarding other engineers; pair programming; knowledge transfer. NOT managing a team (that's people_management).
- performance: profiling, latency, throughput, kernel/network internals, low-level optimization, "deep understanding of [a technical area]". NOT "high-performing team"; NOT "expert in [tool]" as recruiter fluff.
- process_scaffolding: agile, scrum, sprints, requirements engineering, V&V, project coordination, SDLC governance.
- legacy_stack: required experience with old-paradigm enterprise frameworks regardless of version — Java / Java EE, .NET / .NET Framework, ASP.NET, COBOL, mainframe, VMware / vSphere, Active Directory, BizTalk, ColdFusion, and similar.
- context_infrastructure: authoring runbooks, ADRs, RFCs, dashboards / SLOs; telemetry hygiene; schema documentation; technical writing. NOT generic "cross-functional collaboration" or "communication skills".

ROLE FAMILIES — the engineering archetype:
- software_engineer_general: residual fallback. NEVER use alongside another family — only when no other fits.
- frontend_web: user-facing web/mobile UI/UX work. Full-stack roles default here.
- backend_api: server-side logic, APIs, databases, services.
- mobile: native or cross-platform mobile apps (iOS, Android, React Native, Flutter).
- embedded: firmware, IoT, hardware-near software, RTOS, FPGA.
- data_engineer: data pipelines, warehouses, ingestion infrastructure.
- ml_engineer: pre-LLM ML pipelines, model training, productionising ML systems, MLOps.
- ai_llm_engineer: LLM-based or agent systems — prompt engineering, RAG, foundation-model integration, multi-agent.
- devops_sre_platform: CI/CD, infrastructure-as-code, reliability, platform engineering, SRE.
- security: application or infrastructure security.
- qa_test: software quality, testing, validation as the role's primary focus (QA Engineer, SDET, Test Automation).
- solutions_field: deploys / integrates with customers (pre- and post-sales) — Solutions Engineer, Field Applications, Forward-Deployed.
- legacy_specialist: role's primary purpose is maintaining a legacy stack (Mainframe, COBOL, classic .NET, Salesforce, ServiceNow, SAP/ABAP, Oracle Forms). Not just one of many required skills.
- data_analytics: analytical, operational, or decision-support outputs (Data Scientist, Data Analyst, Analytics Engineer). Not production ML pipelines.
- research: applied or basic research bridging academia and product (Research Scientist, Applied Scientist, Research Engineer).
- infra_ops_admin: operates/administrates databases, servers, networks rather than authoring software (DBA, sysadmin, network engineer, cloud admin).
- people_manager: engineering management focused on people, hiring, team operations (Engineering Manager). Excludes director/VP/C-level.

Reminder: tag only what the posting explicitly names. Multi-label allowed on both axes. Output JSON with two arrays — `skill_themes` and `role_families`. Empty arrays mean none. No commentary."""


VARIANT_PROMPTS = {
    "skill": SYSTEM_PROMPT_SKILL,
    "role_family": SYSTEM_PROMPT_ROLE_FAMILY,
    "combined": SYSTEM_PROMPT_COMBINED,
}


def _schema_for(variant: str,
                skill_order: tuple[str, ...] | None = None,
                role_order: tuple[str, ...] | None = None) -> dict:
    """Build the structured-output schema for a given variant.

    Optional skill_order / role_order let stability tests permute enum order.
    """
    skill_enum = list(skill_order or SKILL_LABELS)
    role_enum = list(role_order or ROLE_FAMILY_LABELS)
    if variant == "skill":
        return {
            "type": "object",
            "properties": {
                "skill_themes": {
                    "type": "array",
                    "items": {"type": "string", "enum": skill_enum},
                }
            },
            "required": ["skill_themes"],
            "additionalProperties": False,
        }
    if variant == "role_family":
        return {
            "type": "object",
            "properties": {
                "role_families": {
                    "type": "array",
                    "items": {"type": "string", "enum": role_enum},
                }
            },
            "required": ["role_families"],
            "additionalProperties": False,
        }
    if variant == "combined":
        return {
            "type": "object",
            "properties": {
                "skill_themes": {
                    "type": "array",
                    "items": {"type": "string", "enum": skill_enum},
                },
                "role_families": {
                    "type": "array",
                    "items": {"type": "string", "enum": role_enum},
                },
            },
            "required": ["skill_themes", "role_families"],
            "additionalProperties": False,
        }
    raise ValueError(f"unknown variant: {variant!r}")


def _build_payload(*, variant: str, system_prompt: str, user_text: str, model: str,
                   skill_order: tuple[str, ...] | None,
                   role_order: tuple[str, ...] | None) -> dict:
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
                "name": f"swe_classification_{variant}",
                "strict": True,
                "schema": _schema_for(variant, skill_order, role_order),
            }
        },
    }


def _extract_json(body: dict) -> tuple[dict | None, str | None]:
    """Walk Responses API output and return the first JSON object found."""
    for text in _iter_json_text_candidates(body):
        if not text or not text.strip():
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and ("skill_themes" in parsed or "role_families" in parsed):
            return parsed, text
    return None, None


@dataclass
class ClassifyV3Result:
    variant: str
    skill_themes: list[str] = field(default_factory=list)
    role_families: list[str] = field(default_factory=list)
    parsed_ok: bool = False
    raw_text: str | None = None
    latency_s: float = 0.0
    request_id: str = ""
    model: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "variant": self.variant,
            "skill_themes": self.skill_themes,
            "role_families": self.role_families,
            "parsed_ok": self.parsed_ok,
            "raw_text": self.raw_text,
            "latency_s": round(self.latency_s, 3),
            "request_id": self.request_id,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "error": self.error,
        }


def classify_v3(
    *,
    variant: str,
    title: str,
    description: str,
    model: str,
    timeout_s: float = 60.0,
    system_prompt_override: str | None = None,
    skill_order_override: tuple[str, ...] | None = None,
    role_order_override: tuple[str, ...] | None = None,
) -> ClassifyV3Result:
    """Classify a single posting under a variant."""
    if variant not in VARIANT_PROMPTS:
        raise ValueError(f"unknown variant: {variant!r}; choose from {list(VARIANT_PROMPTS)}")

    sys_prompt = system_prompt_override if system_prompt_override is not None else VARIANT_PROMPTS[variant]
    user_text = build_user_text(title, description)
    payload = _build_payload(
        variant=variant,
        system_prompt=sys_prompt,
        user_text=user_text,
        model=model,
        skill_order=skill_order_override,
        role_order=role_order_override,
    )

    headers = _build_headers()
    request_id = uuid.uuid4().hex
    headers.setdefault("X-Request-ID", request_id)

    t0 = time.time()
    try:
        resp = httpx.post(OPENAI_RESPONSES_API_URL, headers=headers, json=payload, timeout=timeout_s)
        latency = time.time() - t0
        if resp.status_code >= 400:
            return ClassifyV3Result(
                variant=variant,
                parsed_ok=False,
                raw_text=resp.text[:2000],
                latency_s=latency,
                request_id=request_id,
                model=model,
                error=f"HTTP {resp.status_code}: {resp.text[:500]}",
            )
        body = resp.json()
    except Exception as e:
        latency = time.time() - t0
        return ClassifyV3Result(
            variant=variant,
            parsed_ok=False,
            latency_s=latency,
            request_id=request_id,
            model=model,
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
        return ClassifyV3Result(
            variant=variant,
            parsed_ok=False,
            raw_text=raw_text or json.dumps(body)[:2000],
            latency_s=latency,
            request_id=request_id,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_tokens=cached,
            error="parse_failed",
        )

    skill_raw = parsed.get("skill_themes", []) if variant in ("skill", "combined") else []
    role_raw = parsed.get("role_families", []) if variant in ("role_family", "combined") else []
    skill_clean = sorted([l for l in (skill_raw or []) if l in SKILL_LABELS])
    role_clean = sorted([l for l in (role_raw or []) if l in ROLE_FAMILY_LABELS])
    return ClassifyV3Result(
        variant=variant,
        skill_themes=skill_clean,
        role_families=role_clean,
        parsed_ok=True,
        raw_text=raw_text,
        latency_s=latency,
        request_id=request_id,
        model=model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cached_tokens=cached,
        error=None,
    )


if __name__ == "__main__":
    # Smoke test: same posting through all 3 variants
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--title", default="Senior Software Engineer, ML Platform")
    parser.add_argument("--description", default=(
        "We are looking for a senior engineer to lead the design of our distributed services. "
        "You will mentor a team of 5 engineers, run code reviews, write architecture decision records, "
        "build training pipelines for our ML models, and own latency/throughput optimization for our hot path. "
        "Required: 7+ years Go, deep understanding of distributed systems, Kafka, Kubernetes, PyTorch."
    ))
    args = parser.parse_args()
    for variant in ("skill", "role_family", "combined"):
        result = classify_v3(
            variant=variant, title=args.title, description=args.description, model=args.model,
        )
        print(f"--- variant={variant} ---")
        print(json.dumps(result.to_dict(), indent=2))
        print()
