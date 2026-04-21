import json
import logging
from collections import Counter

import pandas as pd
import pytest

from tests.helpers.imports import load_module
from tests.helpers.llm_fakes import codex_stdout, completed_process


stage10 = load_module("stage10_llm_classify", "preprocessing/scripts/stage10_llm_classify.py")
llm_shared = load_module("llm_shared", "preprocessing/scripts/llm_shared.py")


def classification_payload(
    *,
    swe_classification: str = "SWE",
    seniority: str = "entry",
    ghost_assessment: str = "realistic",
    yoe_min_years: int | None = 1,
):
    return {
        "swe_classification": swe_classification,
        "seniority": seniority,
        "ghost_assessment": ghost_assessment,
        "yoe_min_years": yoe_min_years,
    }


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
def test_prepare_classification_rows_uses_cleaned_text_and_short_filter():
    df = pd.DataFrame(
        [
            {
                "job_id": "c1",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Acme",
                "description": "A long enough raw description that should not be excluded by the Stage 9 short-description rule.",
                "description_core_llm": "LLM CORE",
                "is_english": True,
                "selected_for_llm_frame": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "selected_for_control_cohort": False,
                "swe_classification_tier": "weak",
                "seniority_final": "unknown",
                "seniority_final_source": "unknown",
                "ghost_job_risk": "medium",
            },
            {
                "job_id": "c2",
                "source_platform": "linkedin",
                "title": "QA Engineer",
                "company_name": "Beta",
                "description": "tiny text",
                "description_core_llm": "",
                "is_english": True,
                "selected_for_llm_frame": True,
                "is_swe": False,
                "is_swe_adjacent": True,
                "selected_for_control_cohort": False,
                "swe_classification_tier": "weak",
                "seniority_final": "unknown",
                "seniority_final_source": "unknown",
                "ghost_job_risk": "medium",
            },
            {
                "job_id": "c3",
                "source_platform": "linkedin",
                "title": "Senior Backend Engineer",
                "company_name": "Gamma",
                "description": "A long enough raw description for routing that clearly exceeds the Stage 9 short-description threshold.",
                "description_core_llm": "",
                "is_english": True,
                "selected_for_llm_frame": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "selected_for_control_cohort": False,
                "swe_classification_tier": "regex",
                "seniority_final": "mid-senior",
                "seniority_final_source": "title_keyword",
                "ghost_job_risk": "low",
            },
            {
                "job_id": "c4",
                "source_platform": "linkedin",
                "title": "Backend Engineer",
                "company_name": "Delta",
                "description": "A long enough raw description for a row that should stay outside the inherited frame.",
                "description_core_llm": "",
                "is_english": True,
                "selected_for_llm_frame": False,
                "is_swe": True,
                "is_swe_adjacent": False,
                "selected_for_control_cohort": False,
                "swe_classification_tier": "weak",
                "seniority_final": "unknown",
                "seniority_final_source": "unknown",
                "ghost_job_risk": "medium",
            },
        ]
    )

    out = stage10.prepare_classification_rows(df).set_index("job_id")

    assert out.loc["c1", "classification_input"] == "LLM CORE"
    assert out.loc["c1", "needs_llm_classification"]
    assert out.loc["c1", "llm_classification_sample_tier"] == "core"
    assert out.loc["c1", "classification_input_hash"] == stage10.compute_classification_input_hash(
        "Software Engineer",
        "Acme",
        "LLM CORE",
    )
    assert out.loc["c2", "llm_classification_reason"] == "short_description_excluded_by_stage9"
    assert not out.loc["c2", "needs_llm_classification"]
    assert out.loc["c3", "llm_classification_reason"] == "routed"
    assert out.loc["c3", "needs_llm_classification"]
    assert out.loc["c4", "llm_classification_reason"] == "not_selected"
    assert out.loc["c4", "llm_classification_sample_tier"] == "none"
    assert not out.loc["c4", "needs_llm_classification"]


@pytest.mark.unit
def test_prepare_classification_rows_requires_inherited_stage9_frame():
    df = pd.DataFrame(
        [
            {
                "job_id": "missing-frame",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Acme",
                "description": "A long enough raw description that would otherwise be eligible for Stage 10 classification.",
                "description_core_llm": "LLM CORE",
                "is_english": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "selected_for_control_cohort": False,
                "swe_classification_tier": "weak",
                "seniority_final": "unknown",
                "seniority_final_source": "unknown",
                "ghost_job_risk": "medium",
            }
        ]
    )

    with pytest.raises(ValueError, match="selected_for_llm_frame"):
        stage10.prepare_classification_rows(df)


@pytest.mark.unit
def test_sqlite_cache_roundtrip_and_uncached_detection(tmp_path):
    cache_db = tmp_path / "llm_responses.db"
    conn = stage10.open_cache(cache_db)

    payload = classification_payload()
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
def test_try_provider_and_engine_runtime_use_stubbed_subprocess_and_config(monkeypatch, tmp_path):
    log = logging.getLogger("stage10-test")
    log.addHandler(logging.NullHandler())

    payload = classification_payload()

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

    captured = {}

    def fake_execute_task_with_runtime(**kwargs):
        captured["providers"] = [engine.provider for engine in kwargs["runtime"].engines]
        captured["tiers"] = [engine.tier for engine in kwargs["runtime"].engines]
        return {
            "provider": "claude",
            "model": "haiku",
            "latency_seconds": 0.0,
            "response_json": json.dumps(payload),
            "payload": payload,
            "tokens_used": None,
            "cost_usd": None,
        }

    monkeypatch.setattr(stage10, "execute_task_with_runtime", fake_execute_task_with_runtime)
    result = stage10.call_task_with_engine(
        task_name=stage10.CLASSIFICATION_TASK_NAME,
        prompt="prompt",
        input_hash="hash-b",
        error_log_path=tmp_path / "errors.jsonl",
        log=log,
        payload_validator=stage10.validate_classification_payload,
        enabled_engines=("codex", "claude"),
        engine_tiers={"codex": "full", "claude": "non_intrusive"},
    )
    assert captured["providers"] == ["codex", "claude"]
    assert captured["tiers"] == ["full", "non_intrusive"]
    assert result["provider"] == "claude"


@pytest.mark.unit
def test_parse_args_defaults_to_30_workers(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["stage10_llm_classify.py", "--llm-budget", "100"],
    )
    args = stage10.parse_args()
    assert args.max_workers == 30
    assert args.llm_budget == 100
    assert args.llm_budget_split == llm_shared.DEFAULT_BUDGET_SPLIT


@pytest.mark.unit
def test_parse_args_requires_llm_budget(monkeypatch):
    monkeypatch.setattr("sys.argv", ["stage10_llm_classify.py"])
    with pytest.raises(SystemExit):
        stage10.parse_args()


@pytest.mark.unit
def test_build_results_row_maps_classification_payload():
    payload = classification_payload()
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
    # Seniority is no longer surfaced as a separate seniority_llm column;
    # the LLM seniority value lives in the cached classification payload and
    # is integrated into seniority_final by integrate_chunk.
    assert "seniority_llm" not in out
    assert out["ghost_assessment_llm"] == "realistic"


@pytest.mark.unit
def test_integrate_chunk_maps_selected_rows_to_deferred_and_labeled_states():
    chunk = pd.DataFrame(
        [
            {
                "job_id": "rule-1",
                "source_platform": "linkedin",
                "title": "Senior Backend Engineer",
                "company_name": "Gamma",
                "description": "A long enough raw description for routing that clearly exceeds the Stage 9 short-description threshold.",
                "description_core_llm": "",
                "is_english": True,
                "selected_for_llm_frame": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "selected_for_control_cohort": False,
                "swe_classification_tier": "regex",
                "seniority_final": "mid-senior",
                "seniority_final_source": "title_keyword",
                "seniority_rule": "mid-senior",
                "seniority_rule_source": "title_keyword",
                "ghost_job_risk": "low",
            },
            {
                "job_id": "llm-1",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Acme",
                "description": "A long enough raw description that should not be excluded by the Stage 9 short-description rule.",
                "description_core_llm": "LLM CORE",
                "is_english": True,
                "selected_for_llm_frame": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "selected_for_control_cohort": False,
                "swe_classification_tier": "weak",
                "seniority_final": "unknown",
                "seniority_final_source": "unknown",
                "seniority_rule": "unknown",
                "seniority_rule_source": "unknown",
                "ghost_job_risk": "medium",
            },
            {
                "job_id": "out-1",
                "source_platform": "linkedin",
                "title": "Site Reliability Engineer",
                "company_name": "Delta",
                "description": "A long enough raw description for a row that should stay outside the inherited frame.",
                "description_core_llm": "",
                "is_english": True,
                "selected_for_llm_frame": False,
                "is_swe": True,
                "is_swe_adjacent": False,
                "selected_for_control_cohort": False,
                "swe_classification_tier": "weak",
                "seniority_final": "unknown",
                "seniority_final_source": "unknown",
                "seniority_rule": "unknown",
                "seniority_rule_source": "unknown",
                "ghost_job_risk": "medium",
            },
        ]
    )
    llm_hash = stage10.compute_classification_input_hash("Software Engineer", "Acme", "LLM CORE")
    cache = {
        llm_hash: {
            "response_json": json.dumps(
                {
                    "swe_classification": "SWE",
                    "seniority": "entry",
                    "ghost_assessment": "realistic",
                    "yoe_min_years": 1,
                }
            ),
            "model": "gpt-5.4-mini",
            "prompt_version": stage10.CLASSIFICATION_PROMPT_VERSION,
        }
    }

    out = stage10.integrate_chunk(chunk, cache, fresh_hashes=set()).set_index("job_id")

    # rule-1: routes now that the skip shortcut is gone. No cache entry → deferred.
    assert out.loc["rule-1", "llm_classification_sample_tier"] == "core"
    assert out.loc["rule-1", "llm_classification_coverage"] == "deferred"
    assert out.loc["rule-1", "llm_classification_resolution"] == "deferred"
    assert out.loc["rule-1", "seniority_final"] == "mid-senior"
    assert out.loc["rule-1", "seniority_final_source"] == "title_keyword"
    assert out.loc["rule-1", "seniority_rule"] == "mid-senior"
    assert out.loc["rule-1", "seniority_rule_source"] == "title_keyword"
    # llm-1: routed, LLM result overwrites seniority_final, source becomes "llm",
    # but seniority_rule preserves Stage 5's snapshot.
    assert out.loc["llm-1", "llm_classification_sample_tier"] == "core"
    assert out.loc["llm-1", "llm_classification_coverage"] == "labeled"
    assert out.loc["llm-1", "llm_classification_resolution"] == "cached_llm"
    assert out.loc["llm-1", "seniority_final"] == "entry"
    assert out.loc["llm-1", "seniority_final_source"] == "llm"
    assert out.loc["llm-1", "seniority_rule"] == "unknown"
    assert out.loc["llm-1", "seniority_rule_source"] == "unknown"
    # The legacy seniority_llm column no longer exists
    assert "seniority_llm" not in out.columns
    # out-1: outside the frame, seniority_final stays as Stage 5 wrote it
    assert out.loc["out-1", "llm_classification_sample_tier"] == "none"
    assert out.loc["out-1", "llm_classification_coverage"] == "not_selected"
    assert out.loc["out-1", "llm_classification_resolution"] == "not_selected"
    assert out.loc["out-1", "seniority_final"] == "unknown"


@pytest.mark.unit
def test_integrate_chunk_marks_fresh_hashes_as_fresh_llm():
    chunk = pd.DataFrame(
        [
            {
                "job_id": "fresh-1",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Acme",
                "description": "A long enough raw description that should not be excluded by the Stage 9 short-description rule.",
                "description_core_llm": "LLM CORE",
                "is_english": True,
                "selected_for_llm_frame": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "selected_for_control_cohort": False,
                "swe_classification_tier": "weak",
                "seniority_final": "unknown",
                "seniority_final_source": "unknown",
                "ghost_job_risk": "medium",
            },
        ]
    )
    fresh_hash = stage10.compute_classification_input_hash("Software Engineer", "Acme", "LLM CORE")
    cache = {
        fresh_hash: {
            "response_json": json.dumps(
                {
                    "swe_classification": "SWE",
                    "seniority": "entry",
                    "ghost_assessment": "realistic",
                    "yoe_min_years": 1,
                }
            ),
            "model": "gpt-5.4-mini",
            "prompt_version": stage10.CLASSIFICATION_PROMPT_VERSION,
        }
    }

    out = stage10.integrate_chunk(chunk, cache, fresh_hashes={fresh_hash}).set_index("job_id")

    assert out.loc["fresh-1", "llm_classification_sample_tier"] == "core"
    assert out.loc["fresh-1", "llm_classification_coverage"] == "labeled"
    assert out.loc["fresh-1", "llm_classification_resolution"] == "fresh_llm"


@pytest.mark.unit
def test_seniority_rule_preserved_when_llm_overrides():
    chunk = pd.DataFrame(
        [
            {
                "job_id": "override-1",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Acme",
                "description": "A long enough raw description that should not be excluded by the Stage 9 short-description rule.",
                "description_core_llm": "LLM CORE",
                "is_english": True,
                "selected_for_llm_frame": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "selected_for_control_cohort": False,
                "swe_classification_tier": "weak",
                "seniority_final": "junior",
                "seniority_final_source": "title_keyword",
                "seniority_rule": "junior",
                "seniority_rule_source": "title_keyword",
                "ghost_job_risk": "medium",
            },
        ]
    )
    cache_hash = stage10.compute_classification_input_hash("Software Engineer", "Acme", "LLM CORE")
    cache = {
        cache_hash: {
            "response_json": json.dumps(classification_payload(seniority="senior")),
            "model": "gpt-5.4-mini",
            "prompt_version": stage10.CLASSIFICATION_PROMPT_VERSION,
        }
    }

    out = stage10.integrate_chunk(chunk, cache, fresh_hashes=set()).set_index("job_id")

    assert out.loc["override-1", "seniority_final"] == "senior"
    assert out.loc["override-1", "seniority_final_source"] == "llm"
    assert out.loc["override-1", "seniority_rule"] == "junior"
    assert out.loc["override-1", "seniority_rule_source"] == "title_keyword"


@pytest.mark.unit
def test_prepare_rows_no_skip_for_strong_rule_based():
    df = pd.DataFrame(
        [
            {
                "job_id": "strong-1",
                "source_platform": "linkedin",
                "title": "Senior Backend Engineer",
                "company_name": "Gamma",
                "description": "A long enough raw description for routing that clearly exceeds the Stage 9 short-description threshold.",
                "description_core_llm": "LLM CORE",
                "is_english": True,
                "selected_for_llm_frame": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "selected_for_control_cohort": False,
                "swe_classification_tier": "regex",
                "seniority_final": "mid-senior",
                "seniority_final_source": "title_keyword",
                "ghost_job_risk": "low",
            }
        ]
    )

    out = stage10.prepare_classification_rows(df).set_index("job_id")

    assert out.loc["strong-1", "needs_llm_classification"]
    assert out.loc["strong-1", "llm_classification_reason"] == "routed"


@pytest.mark.unit
def test_integrate_chunk_falls_back_to_raw_description_when_stage9_llm_text_missing():
    raw_description = "A long enough raw description that acts as the fallback when Stage 9 LLM cleaned text is missing."
    chunk = pd.DataFrame(
        [
            {
                "job_id": "fallback-core-1",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Acme",
                "description": raw_description,
                "description_core_llm": "",
                "is_english": True,
                "selected_for_llm_frame": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "selected_for_control_cohort": False,
                "swe_classification_tier": "weak",
                "seniority_final": "unknown",
                "seniority_final_source": "unknown",
                "ghost_job_risk": "medium",
            }
        ]
    )
    cache_hash = stage10.compute_classification_input_hash("Software Engineer", "Acme", raw_description)
    cache = {
        cache_hash: {
            "response_json": json.dumps(classification_payload()),
            "model": "gpt-5.4-mini",
            "prompt_version": stage10.CLASSIFICATION_PROMPT_VERSION,
        }
    }

    out = stage10.integrate_chunk(chunk, cache, fresh_hashes=set()).set_index("job_id")

    assert out.loc["fallback-core-1", "classification_input_hash"] == cache_hash
    assert out.loc["fallback-core-1", "llm_classification_sample_tier"] == "core"
    assert out.loc["fallback-core-1", "llm_classification_coverage"] == "labeled"
    assert out.loc["fallback-core-1", "llm_classification_resolution"] == "cached_llm"
    # LLM result writes back to seniority_final with source='llm'
    assert out.loc["fallback-core-1", "seniority_final"] == "entry"
    assert out.loc["fallback-core-1", "seniority_final_source"] == "llm"


@pytest.mark.unit
def test_summarize_stage10_routing_reports_volume_drivers():
    summary = stage10.summarize_stage10_routing(
        total_rows=20,
        routed_rows=9,
        selected_control_rows=3,
        reason_counts=Counter(
            {
                "routed": 9,
                "short_description_excluded_by_stage9": 2,
                "not_selected": 3,
            }
        ),
        unique_task_count=7,
        duplicate_rows_collapsed=2,
        cached_task_count=4,
        fresh_task_count=3,
    )

    assert summary == {
        "total_rows": 20,
        "routed_rows": 9,
        "selected_control_rows": 3,
        "short_skip_rows": 2,
        "not_routed_rows": 3,
        "unique_tasks": 7,
        "duplicate_rows_collapsed": 2,
        "cached_tasks": 4,
        "fresh_tasks": 3,
    }


@pytest.mark.unit
def test_run_stage10_reuses_current_prompt_cache_for_supplemental_rows_outside_core(tmp_path):
    input_path = tmp_path / "stage9_llm_cleaned.parquet"
    raw_description = "A long enough raw description for a supplemental cached row that remains eligible for Stage 10 classification."
    row = {
        "job_id": "supp-1",
        "source": "scraped",
        "source_platform": "linkedin",
        "title": "Software Engineer",
        "company_name": "Acme",
        "description": raw_description,
        "description_core_llm": "",
        "is_english": True,
        "selected_for_llm_frame": False,
        "is_swe": True,
        "is_swe_adjacent": False,
        "is_control": False,
        "selected_for_control_cohort": False,
        "swe_classification_tier": "weak",
        "seniority_final": "unknown",
        "seniority_final_source": "unknown",
        "ghost_job_risk": "medium",
    }
    pd.DataFrame([row]).to_parquet(input_path, index=False)

    cache_db = tmp_path / "llm_responses.db"
    conn = stage10.open_cache(cache_db)
    supplemental_hash = stage10.compute_classification_input_hash("Software Engineer", "Acme", raw_description)
    stage10.store_cached_row(
        conn,
        input_hash=supplemental_hash,
        task_name=stage10.CLASSIFICATION_TASK_NAME,
        model="gpt-5.4-mini",
        prompt_version=stage10.CLASSIFICATION_PROMPT_VERSION,
        response_json=json.dumps(classification_payload()),
        tokens_used=17,
    )
    conn.close()

    results_path = tmp_path / "stage10_results.parquet"
    integrated_path = tmp_path / "stage10_integrated.parquet"
    stage10.run_stage10(
        llm_budget=0,
        llm_budget_split={"swe": 1.0, "swe_adjacent": 0.0, "control": 0.0},
        input_path=input_path,
        results_path=results_path,
        integrated_path=integrated_path,
        compat_output_path=None,
        cache_db=cache_db,
        error_log_path=tmp_path / "llm_errors.jsonl",
        max_workers=1,
        enabled_engines=("codex",),
    )

    integrated = pd.read_parquet(integrated_path).set_index("job_id")
    results = pd.read_parquet(results_path).set_index("job_id")

    assert not bool(integrated.loc["supp-1", "selected_for_llm_frame"])
    assert integrated.loc["supp-1", "llm_classification_sample_tier"] == "supplemental_cache"
    assert integrated.loc["supp-1", "llm_classification_coverage"] == "labeled"
    assert integrated.loc["supp-1", "llm_classification_resolution"] == "cached_llm"
    assert integrated.loc["supp-1", "seniority_final"] == "entry"
    assert integrated.loc["supp-1", "seniority_final_source"] == "llm"
    assert results.loc["supp-1", "classification_input_hash"] == supplemental_hash


@pytest.mark.unit
def test_run_stage10_gates_supplemental_cache_reuse_on_prompt_version(tmp_path):
    input_path = tmp_path / "stage9_llm_cleaned.parquet"
    row = {
        "job_id": "supp-old-prompt-1",
        "source": "scraped",
        "source_platform": "linkedin",
        "title": "Software Engineer",
        "company_name": "Acme",
        "description": "A long enough raw description for a supplemental row whose cached classification uses an old prompt version.",
        "description_core_llm": "",
        "is_english": True,
        "selected_for_llm_frame": False,
        "is_swe": True,
        "is_swe_adjacent": False,
        "is_control": False,
        "selected_for_control_cohort": False,
        "swe_classification_tier": "weak",
        "seniority_final": "unknown",
        "seniority_final_source": "unknown",
        "ghost_job_risk": "medium",
    }
    pd.DataFrame([row]).to_parquet(input_path, index=False)

    cache_db = tmp_path / "llm_responses.db"
    conn = stage10.open_cache(cache_db)
    supplemental_hash = stage10.compute_classification_input_hash("Software Engineer", "Acme", "RULE CORE ONLY")
    stage10.store_cached_row(
        conn,
        input_hash=supplemental_hash,
        task_name=stage10.CLASSIFICATION_TASK_NAME,
        model="gpt-5.4-mini",
        prompt_version="stale-prompt-version",
        response_json=json.dumps(classification_payload()),
        tokens_used=17,
    )
    conn.close()

    results_path = tmp_path / "stage10_results.parquet"
    integrated_path = tmp_path / "stage10_integrated.parquet"
    stage10.run_stage10(
        llm_budget=0,
        llm_budget_split={"swe": 1.0, "swe_adjacent": 0.0, "control": 0.0},
        input_path=input_path,
        results_path=results_path,
        integrated_path=integrated_path,
        compat_output_path=None,
        cache_db=cache_db,
        error_log_path=tmp_path / "llm_errors.jsonl",
        max_workers=1,
        enabled_engines=("codex",),
    )

    integrated = pd.read_parquet(integrated_path).set_index("job_id")
    results = pd.read_parquet(results_path)

    assert integrated.loc["supp-old-prompt-1", "llm_classification_sample_tier"] == "none"
    assert integrated.loc["supp-old-prompt-1", "llm_classification_coverage"] == "not_selected"
    assert integrated.loc["supp-old-prompt-1", "llm_classification_resolution"] == "not_selected"
    # Stale prompt → no LLM result, seniority_final stays as Stage 5 wrote it
    assert integrated.loc["supp-old-prompt-1", "seniority_final"] == "unknown"
    assert integrated.loc["supp-old-prompt-1", "seniority_final_source"] == "unknown"
    assert results.empty


@pytest.mark.unit
def test_run_stage10_applies_budget_split_before_fresh_task_selection(tmp_path, monkeypatch):
    input_path = tmp_path / "stage9_llm_cleaned.parquet"
    rows = []
    for idx in range(3):
        rows.append(
            {
                "job_id": f"swe-{idx}",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": f"Software Engineer {idx}",
                "company_name": f"Acme {idx}",
                "description": f"A long enough raw description for routed SWE row {idx} that exceeds the short-description threshold.",
                "description_core_llm": f"LLM CORE {idx}",
                "is_english": True,
                "selected_for_llm_frame": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "is_control": False,
                "swe_classification_tier": "weak",
                "seniority_final": "unknown",
                "seniority_final_source": "unknown",
                "ghost_job_risk": "medium",
                "date_posted": "2026-03-20",
                "scrape_date": "2026-03-20",
                "selection_date_bin": "2026-03-20",
            }
        )
    for idx in range(2):
        rows.append(
            {
                "job_id": f"adj-{idx}",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": f"QA Engineer {idx}",
                "company_name": f"Beta {idx}",
                "description": f"A long enough raw description for routed adjacent row {idx} that exceeds the short-description threshold.",
                "description_core_llm": f"LLM CORE ADJ {idx}",
                "is_english": True,
                "selected_for_llm_frame": True,
                "is_swe": False,
                "is_swe_adjacent": True,
                "is_control": False,
                "swe_classification_tier": "weak",
                "seniority_final": "unknown",
                "seniority_final_source": "unknown",
                "ghost_job_risk": "medium",
                "date_posted": "2026-03-20",
                "scrape_date": "2026-03-20",
                "selection_date_bin": "2026-03-20",
            }
        )
    rows.append(
        {
            "job_id": "ctrl-0",
            "source": "scraped",
            "source_platform": "linkedin",
            "title": "Financial Analyst",
            "company_name": "Gamma",
            "description": "A long enough raw description for a routed control row that exceeds the short-description threshold.",
            "description_core_llm": "LLM CORE",
            "is_english": True,
            "selected_for_llm_frame": True,
            "is_swe": False,
            "is_swe_adjacent": False,
            "is_control": True,
            "swe_classification_tier": "weak",
            "seniority_final": "unknown",
            "seniority_final_source": "unknown",
            "ghost_job_risk": "medium",
            "date_posted": "2026-03-20",
            "scrape_date": "2026-03-20",
            "selection_date_bin": "2026-03-20",
        }
    )
    pd.DataFrame(rows).to_parquet(input_path, index=False)

    calls = []

    def fake_select_fresh_call_tasks(unresolved_selected_rows, *, llm_budget, hash_key, groups):
        calls.append(
            {
                "groups": groups,
                "llm_budget": llm_budget,
                "count": len(unresolved_selected_rows),
            }
        )
        return [], {"selection_target": llm_budget, "selected_count": 0}

    monkeypatch.setattr(stage10, "select_fresh_call_tasks", fake_select_fresh_call_tasks)

    stage10.run_stage10(
        llm_budget=6,
        llm_budget_split={"swe": 0.5, "swe_adjacent": 0.3, "control": 0.2},
        input_path=input_path,
        results_path=tmp_path / "stage10_results.parquet",
        integrated_path=tmp_path / "stage10_integrated.parquet",
        compat_output_path=None,
        cache_db=tmp_path / "llm_responses.db",
        error_log_path=tmp_path / "llm_errors.jsonl",
        max_workers=1,
        enabled_engines=("codex",),
    )

    assert calls == [
        {"groups": ("swe",), "llm_budget": 3, "count": 3},
        {"groups": ("swe_adjacent",), "llm_budget": 2, "count": 2},
        {"groups": ("control",), "llm_budget": 1, "count": 1},
    ]
