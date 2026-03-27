import json
import logging

import pandas as pd
import pytest

from tests.helpers.imports import load_module
from tests.helpers.llm_fakes import codex_stdout, completed_process


stage10 = load_module("stage10_llm_classify", "preprocessing/scripts/stage10_llm_classify.py")


@pytest.mark.unit
def test_segment_description_into_units_handles_headers_labels_and_abbrev():
    units = stage10.segment_description_into_units(
        """Job Description
Responsibilities:
Build APIs. Maintain services.
Benefits:
Health, dental, 401k.
Work Model:
Hybrid 3 days onsite.
U.S. office located in NYC."""
    )

    assert [unit["text"] for unit in units] == [
        "Job Description",
        "Responsibilities: | Build APIs. Maintain services.",
        "Benefits: | Health, dental, 401k.",
        "Work Model: | Hybrid 3 days onsite.",
        "U.S. office located in NYC.",
    ]


@pytest.mark.unit
def test_prepare_classification_rows_uses_cleaned_text_and_skip_logic():
    df = pd.DataFrame(
        [
            {
                "job_id": "c1",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Acme",
                "description": "A long enough raw description that should not be excluded by the Stage 9 short-description rule.",
                "description_core": "RULE CORE",
                "description_core_llm": "LLM CORE",
                "is_english": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "selected_for_control_cohort": False,
                "swe_classification_tier": "weak",
                "seniority_source": "content",
                "ghost_job_risk": "medium",
            },
            {
                "job_id": "c2",
                "source_platform": "linkedin",
                "title": "QA Engineer",
                "company_name": "Beta",
                "description": "tiny text",
                "description_core": "RULE CORE",
                "description_core_llm": "",
                "is_english": True,
                "is_swe": False,
                "is_swe_adjacent": True,
                "selected_for_control_cohort": False,
                "swe_classification_tier": "weak",
                "seniority_source": "content",
                "ghost_job_risk": "medium",
            },
            {
                "job_id": "c3",
                "source_platform": "linkedin",
                "title": "Backend Engineer",
                "company_name": "Gamma",
                "description": "A long enough raw description for routing that clearly exceeds the Stage 9 short-description threshold.",
                "description_core": "RULE CORE",
                "description_core_llm": "",
                "is_english": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "selected_for_control_cohort": False,
                "swe_classification_tier": "regex",
                "seniority_source": "title_native",
                "ghost_job_risk": "low",
            },
        ]
    )

    out = stage10.prepare_classification_rows(df).set_index("job_id")

    assert out.loc["c1", "classification_input"] == "LLM CORE"
    assert out.loc["c1", "needs_llm_classification"]
    assert out.loc["c1", "classification_input_hash"] == stage10.compute_classification_input_hash(
        "Software Engineer",
        "Acme",
        "LLM CORE",
    )
    assert out.loc["c2", "llm_classification_reason"] == "short_description_excluded_by_stage9"
    assert not out.loc["c2", "needs_llm_classification"]
    assert out.loc["c3", "llm_classification_reason"] == "high_confidence_technical_rules"
    assert not out.loc["c3", "needs_llm_classification"]


@pytest.mark.unit
def test_sqlite_cache_roundtrip_and_uncached_detection(tmp_path):
    cache_db = tmp_path / "llm_responses.db"
    conn = stage10.open_cache(cache_db)

    payload = {
        "swe_classification": "SWE",
        "seniority": "entry",
        "ghost_assessment": "realistic",
        "yoe_min_years": 1,
    }
    stage10.store_cached_row(
        conn,
        input_hash="hash-a",
        task_name=stage10.CLASSIFICATION_TASK_NAME,
        model="gpt-5.4-mini",
        prompt_version=stage10.CLASSIFICATION_PROMPT_VERSION,
        response_json=json.dumps(payload),
        tokens_used=11,
    )

    assert stage10.fetch_cached_row(
        conn,
        input_hash="hash-a",
        task_name=stage10.CLASSIFICATION_TASK_NAME,
        prompt_version=stage10.CLASSIFICATION_PROMPT_VERSION,
    )["tokens_used"] == 11

    candidate_df = pd.DataFrame(
        [
            {"classification_input_hash": "hash-a", "needs_llm_classification": True},
            {"classification_input_hash": "hash-b", "needs_llm_classification": True},
        ]
    )
    assert stage10.has_uncached_rows(candidate_df, conn)
    conn.close()


@pytest.mark.unit
def test_try_provider_and_fallback_use_stubbed_subprocess_and_order(monkeypatch, tmp_path):
    log = logging.getLogger("stage10-test")
    log.addHandler(logging.NullHandler())

    payload = {
        "swe_classification": "SWE",
        "seniority": "entry",
        "ghost_assessment": "realistic",
        "yoe_min_years": 1,
    }

    monkeypatch.setitem(
        stage10.try_provider.__globals__,
        "call_subprocess",
        lambda command, timeout_seconds: completed_process(stdout=codex_stdout(payload, tokens_used=123)),
    )
    result = stage10.try_provider(
        provider="codex",
        prompt="prompt",
        model="gpt-5.4-mini",
        task_name=stage10.CLASSIFICATION_TASK_NAME,
        input_hash="hash-a",
        error_log_path=tmp_path / "errors.jsonl",
        log=log,
        timeout_seconds=1,
        max_retries=1,
        payload_validator=stage10.validate_classification_payload,
        quota_wait_hours=0.01,
    )
    assert result["tokens_used"] == 123

    calls: list[str] = []

    def fake_try_provider(**kwargs):
        calls.append(kwargs["provider"])
        if kwargs["provider"] == "codex":
            return None
        return {
            "provider": kwargs["provider"],
            "model": kwargs["model"],
            "latency_seconds": 0.0,
            "response_json": json.dumps(payload),
            "payload": payload,
            "tokens_used": None,
            "cost_usd": None,
        }

    monkeypatch.setattr(stage10, "try_provider", fake_try_provider)
    result = stage10.call_task_with_fallback(
        task_name=stage10.CLASSIFICATION_TASK_NAME,
        prompt="prompt",
        input_hash="hash-b",
        error_log_path=tmp_path / "errors.jsonl",
        log=log,
        codex_model="gpt-5.4-mini",
        timeout_seconds=1,
        max_retries=1,
        payload_validator=stage10.validate_classification_payload,
        provider_order=("codex", "claude"),
        claude_model="haiku",
    )
    assert calls == ["codex", "claude"]
    assert result["provider"] == "claude"


@pytest.mark.unit
def test_build_results_row_maps_classification_payload():
    payload = {
        "swe_classification": "SWE",
        "seniority": "entry",
        "ghost_assessment": "realistic",
        "yoe_min_years": 1,
    }
    out = stage10.build_results_row(
        {
            "job_id": "job-1",
            "source": "scraped",
            "source_platform": "linkedin",
            "title": "Software Engineer",
            "company_name": "Acme",
            "classification_row_count": 2,
        },
        "hash-a",
        {
            "model": "gpt-5.4-mini",
            "prompt_version": stage10.CLASSIFICATION_PROMPT_VERSION,
            "response_json": json.dumps(payload),
            "tokens_used": 17,
        },
    )

    assert out["classification_input_hash"] == "hash-a"
    assert out["classification_row_count"] == 2
    assert out["swe_classification_llm"] == "SWE"
    assert out["seniority_llm"] == "entry"
    assert out["ghost_assessment_llm"] == "realistic"
