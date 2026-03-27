from pathlib import Path
import json
import logging

import pandas as pd
import pytest

from tests.helpers.imports import load_module


stage12 = load_module("stage12_validation", "preprocessing/scripts/stage12_validation.py")

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.unit
def test_disagreement_helpers_and_report_generation():
    df = pd.read_parquet(FIXTURES / "synthetic/stage12/validation_sample.parquet")

    assert stage12.map_rule_swe(df.iloc[0]) == "SWE"
    assert stage12.map_rule_seniority(df.iloc[0]) == "entry"
    assert stage12.map_rule_ghost(df.iloc[2]) == "inflated"
    assert stage12.disagreement_flag(df.iloc[0]) is False
    assert stage12.disagreement_flag(df.iloc[1]) is True

    sampled = stage12.stratified_sample(df, sample_size=2, seed=42)
    assert set(sampled["job_id"]) == {"s12_2", "s12_4"}

    left_vals, right_vals = stage12.pairwise_labels("swe", sampled, "rules", "mini")
    assert set(left_vals) == {"SWE", "NOT_SWE"}
    assert set(right_vals) == {"NOT_SWE"}
    assert stage12.cohen_kappa(["A", "A", "B"], ["A", "B", "B"]) == pytest.approx(0.4)
    assert stage12.kappa_label(0.81) == "strong"
    assert stage12.kappa_label(0.60) == "moderate"
    assert stage12.kappa_label(None) == "n/a"

    report = stage12.build_report(sampled, sample_size=2)
    assert "# Validation Report v2" in report
    assert "## SWE Classification" in report
    assert "Cohen's kappa" in report
    assert "Disagreement examples" in report
    assert "s12_2" in report


@pytest.mark.unit
def test_run_full_model_uses_stubbed_provider_and_cache(tmp_path, monkeypatch):
    log = logging.getLogger("stage12-test")
    log.addHandler(logging.NullHandler())

    sample_df = pd.DataFrame(
        [
            {
                "job_id": "full-1",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Acme",
                "description": "RAW DESCRIPTION SHOULD ONLY FEED EXTRACTION.",
                "description_hash": "raw-lineage-hash-1",
                "description_core": "Rule-based fallback core.",
                "description_core_llm": "LLM CLEANED TEXT FOR CLASSIFICATION.",
            }
        ]
    )
    cache_db = tmp_path / "full_model.db"
    error_log_path = tmp_path / "errors.jsonl"

    calls: list[tuple[str, str, str, str]] = []

    def fake_try_provider(*, provider, prompt, model, task_name, input_hash, **kwargs):
        calls.append((provider, model, task_name, input_hash))
        if task_name == stage12.CLASSIFICATION_TASK_NAME:
            assert "LLM CLEANED TEXT FOR CLASSIFICATION." in prompt
            assert "RAW DESCRIPTION SHOULD ONLY FEED EXTRACTION." not in prompt
        else:
            assert "RAW DESCRIPTION SHOULD ONLY FEED EXTRACTION." in prompt
        if task_name == stage12.CLASSIFICATION_TASK_NAME:
            payload = {
                "swe_classification": "SWE",
                "seniority": "entry",
                "ghost_assessment": "realistic",
                "yoe_min_years": 1,
            }
        else:
            payload = {
                "task_status": "ok",
                "boilerplate_unit_ids": [],
                "uncertain_unit_ids": [],
                "reason": "clean",
            }
        return {
            "provider": provider,
            "model": model,
            "latency_seconds": 0.0,
            "response_json": json.dumps(payload),
            "payload": payload,
            "tokens_used": 7,
            "cost_usd": None,
        }

    monkeypatch.setattr(stage12, "try_provider", fake_try_provider)
    monkeypatch.setattr(stage12, "json", json, raising=False)
    out = stage12.run_full_model(
        sample_df=sample_df,
        cache_db=cache_db,
        error_log_path=error_log_path,
        log=log,
        timeout_seconds=1,
        max_retries=1,
    )

    assert calls == [
        (
            "codex",
            "gpt-5.4",
            stage12.CLASSIFICATION_TASK_NAME,
            stage12.compute_classification_input_hash(
                "Software Engineer",
                "Acme",
                "LLM CLEANED TEXT FOR CLASSIFICATION.",
            ),
        ),
        (
            "codex",
            "gpt-5.4",
            stage12.EXTRACTION_TASK_NAME,
            stage12.compute_extraction_input_hash(
                "Software Engineer",
                "Acme",
                "RAW DESCRIPTION SHOULD ONLY FEED EXTRACTION.",
            ),
        ),
    ]
    assert out.loc[0, "swe_classification_full"] == "SWE"
    assert out.loc[0, "description_core_full"] == "RAW DESCRIPTION SHOULD ONLY FEED EXTRACTION."
    assert out.loc[0, "description_core_full_validated"]

    conn = stage12.open_cache(cache_db)
    cached = stage12.fetch_cached_row(
        conn,
        stage12.compute_classification_input_hash(
            "Software Engineer",
            "Acme",
            "LLM CLEANED TEXT FOR CLASSIFICATION.",
        ),
        stage12.CLASSIFICATION_TASK_NAME,
        stage12.CLASSIFICATION_PROMPT_VERSION,
    )
    assert cached is not None
    cached_extract = stage12.fetch_cached_row(
        conn,
        stage12.compute_extraction_input_hash(
            "Software Engineer",
            "Acme",
            "RAW DESCRIPTION SHOULD ONLY FEED EXTRACTION.",
        ),
        stage12.EXTRACTION_TASK_NAME,
        stage12.EXTRACTION_PROMPT_VERSION,
    )
    assert cached_extract is not None
    conn.close()
