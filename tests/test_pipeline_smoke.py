import json
from pathlib import Path

import pandas as pd
import pytest

from tests.helpers.imports import load_module


run_pipeline = load_module("run_pipeline_smoke", "preprocessing/run_pipeline.py")
stage_final_output = load_module("stage_final_output_smoke", "preprocessing/scripts/stage_final_output.py")

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.unit
def test_validate_output_accepts_fixture_generated_stage_contracts(tmp_path, monkeypatch):
    stage9_candidates = tmp_path / "stage9_llm_extraction_candidates.parquet"
    pd.DataFrame(
        [
            {
                "extraction_input_hash": "extract-hash-1",
                "job_id": "job-1",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Acme",
                "description": "Build APIs and maintain services.",
                "description_hash": "raw-hash-1",
                "control_bucket": "scraped|2026-12",
                "selected_for_control_cohort": False,
                "llm_route_group": "technical_extraction",
                "source_row_count": 1,
            }
        ]
    ).to_parquet(stage9_candidates, index=False)

    stage9_results = tmp_path / "stage9_llm_extraction_results.parquet"
    pd.DataFrame(
        [
            {
                "extraction_input_hash": "extract-hash-1",
                "description_hash": "raw-hash-1",
                "job_id": "job-1",
                "source": "scraped",
                "source_platform": "linkedin",
                "llm_route_group": "technical_extraction",
                "source_row_count": 1,
                "selected_for_control_cohort": False,
                "llm_model_extraction": "gpt-5.4-mini",
                "llm_prompt_version_extraction": "prompt-v1",
                "llm_extraction_status": "ok",
                "llm_extraction_unit_ids": "[]",
                "llm_extraction_uncertain_unit_ids": "[]",
                "llm_extraction_reason": "clean",
                "llm_extraction_model_reason": "clean",
                "llm_extraction_units_count": 2,
                "llm_extraction_single_unit": False,
                "llm_extraction_drop_ratio": 0.0,
                "llm_extraction_validated": True,
                "llm_validator_reason": "ok",
                "description_core_llm": "Build APIs and maintain services.",
            }
        ]
    ).to_parquet(stage9_results, index=False)

    stage9_cleaned = tmp_path / "stage9_llm_cleaned.parquet"
    pd.DataFrame(
        [
            {
                "uid": "uid-1",
                "job_id": "job-1",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Acme",
                "description": "Build APIs and maintain services.",
                "description_core_llm": "Build APIs and maintain services.",
                "selected_for_control_cohort": False,
                "llm_text_skip_reason": None,
            }
        ]
    ).to_parquet(stage9_cleaned, index=False)

    stage9_control = tmp_path / "stage9_control_cohort.parquet"
    pd.DataFrame(
        [
            {
                "extraction_input_hash": "extract-hash-2",
                "control_bucket": "scraped|2026-12",
                "selected_for_control_cohort": True,
                "stable_score": "score-1",
                "job_id": "control-1",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Marketing Analyst",
                "company_name": "Beta",
                "description_hash": "raw-hash-2",
                "source_row_count": 1,
            }
        ]
    ).to_parquet(stage9_control, index=False)

    stage9_manifest = tmp_path / "stage9_core_frame_manifest.json"
    stage9_manifest.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "hash_key": "extraction_input_hash",
                "selected_hashes": ["extract-hash-1"],
            }
        ),
        encoding="utf-8",
    )

    stage10_results = tmp_path / "stage10_llm_classification_results.parquet"
    pd.DataFrame(
        [
            {
                "classification_input_hash": "class-hash-1",
                "title": "Software Engineer",
                "company_name": "Acme",
                "source": "scraped",
                "source_platform": "linkedin",
                "job_id": "job-1",
                "classification_row_count": 1,
                "llm_model_classification": "gpt-5.4-mini",
                "llm_prompt_version_classification": "prompt-v1",
                "classification_response_json": json.dumps(
                    {
                        "swe_classification": "SWE",
                        "seniority": "entry",
                        "ghost_assessment": "realistic",
                        "yoe_min_years": 1,
                    }
                ),
                "classification_tokens_used": 42.0,
                "swe_classification_llm": "SWE",
                "ghost_assessment_llm": "realistic",
                "yoe_min_years_llm": 1,
            }
        ]
    ).to_parquet(stage10_results, index=False)

    stage10_integrated = tmp_path / "stage10_llm_integrated.parquet"
    unified_rows = pd.read_parquet(FIXTURES / "sampled/stage11/integration_sample.parquet").head(1)
    unified_rows = unified_rows.assign(
        uid=["uid-1"],
        is_swe=True,
        is_control=False,
        is_swe_adjacent=False,
        analysis_group="swe_combined",
        seniority_final="entry",
        seniority_final_source="llm",
        is_aggregator=False,
        date_flag="ok",
        is_english=True,
        ghost_job_risk="low",
        ghost_assessment_llm="realistic",
    )
    unified_rows.to_parquet(stage10_integrated, index=False)

    obs_path = tmp_path / "stage1_observations.parquet"
    unified_obs_path = tmp_path / "unified_observations.parquet"
    quality_path = tmp_path / "quality_report.json"
    log_path = tmp_path / "preprocessing_log.txt"
    pd.DataFrame([{"uid": "uid-1", "scrape_date": "2026-03-20"}]).to_parquet(obs_path, index=False)
    monkeypatch.setattr(stage_final_output, "OBS_INPUT", obs_path)
    monkeypatch.setattr(stage_final_output, "STAGE_INPUTS", {})
    stage_final_output.build_unified_observations(stage10_integrated, unified_obs_path)
    report = stage_final_output.compute_quality_report(stage10_integrated, unified_obs_path)
    quality_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log_path.write_text(stage_final_output.build_log_text(report), encoding="utf-8")

    stage9_spec = {
        "outputs": [
            {"path": stage9_candidates, "kind": "parquet", "min_rows": 1, "check_col": "extraction_input_hash"},
            {"path": stage9_results, "kind": "parquet", "min_rows": 1, "check_col": "description_core_llm"},
            {"path": stage9_cleaned, "kind": "parquet", "min_rows": 1, "check_col": "description_core_llm"},
            {"path": stage9_control, "kind": "parquet", "min_rows": 1, "check_col": "selected_for_control_cohort"},
            {"path": stage9_manifest, "kind": "text"},
        ],
    }
    stage10_spec = {
        "outputs": [
            {"path": stage10_results, "kind": "parquet", "min_rows": 1, "check_col": "classification_input_hash"},
            {"path": stage10_integrated, "kind": "parquet", "min_rows": 1, "check_col": "description_core_llm"},
        ],
    }
    final_spec = {
        "outputs": [
            {"path": stage10_integrated, "kind": "parquet", "min_rows": 1},
            {"path": unified_obs_path, "kind": "parquet", "min_rows": 1},
            {"path": quality_path, "kind": "text"},
            {"path": log_path, "kind": "text"},
        ]
    }

    assert run_pipeline.validate_output(stage9_spec)
    assert run_pipeline.validate_output(stage10_spec)
    assert run_pipeline.validate_output(final_spec)
    assert report["funnel"]["final_unified"] == 1
