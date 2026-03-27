from pathlib import Path

import pandas as pd
import pytest

from tests.helpers.imports import load_module
from tests.helpers.stage_runner import write_parquet


run_pipeline = load_module("run_pipeline", "preprocessing/run_pipeline.py")
stage_final_output = load_module("stage_final_output", "preprocessing/scripts/stage_final_output.py")

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.unit
def test_runner_declares_expected_stage_contracts():
    stage_numbers = [stage["num"] for stage in run_pipeline.STAGES]
    assert stage_numbers == [1, 2, 3, 4, 5, "6-8", 9, 10, "final"]

    expected_checks = {
        2: "company_name_effective",
        3: "description_core",
        4: "company_name_canonical",
        5: "seniority_final",
        "6-8": "period",
        9: "extraction_input_hash",
        10: "classification_input_hash",
    }
    for stage in run_pipeline.STAGES:
        if stage["num"] in expected_checks:
            assert stage["outputs"][0]["check_col"] == expected_checks[stage["num"]]

    final_stage = run_pipeline.STAGES[-1]
    assert {output["kind"] for output in final_stage["outputs"]} == {"parquet", "text"}


@pytest.mark.unit
def test_final_helpers_join_observations_and_report_counts(tmp_path, monkeypatch):
    sample = pd.read_parquet(FIXTURES / "sampled/stage11/integration_sample.parquet").head(2)
    unified_rows = []
    obs_rows = []
    for idx, row in enumerate(sample.to_dict("records"), start=1):
        uid = f"uid-{idx}"
        unified_rows.append(
            {
                "uid": uid,
                "job_id": row["job_id"],
                "source": row["source"],
                "source_platform": row["source_platform"],
                "title": row["title"],
                "company_name": row["company_name"],
                "description": row["description"],
                "description_core": row["description_core_llm"],
                "description_core_llm": row["description_core_llm"],
                "description_core_full": row["description_core_llm"],
                "is_swe": idx == 1,
                "is_control": False,
                "is_swe_adjacent": idx == 2,
                "seniority_imputed": "entry" if idx == 1 else "unknown",
                "seniority_llm": row["seniority_llm"],
                "is_aggregator": False,
                "date_flag": "ok",
                "is_english": True,
                "ghost_job_risk": "low",
            }
        )
        obs_rows.append({"uid": uid, "scrape_date": "2026-03-20"})

    unified_path = tmp_path / "unified.parquet"
    obs_path = tmp_path / "stage1_observations.parquet"
    output_path = tmp_path / "unified_observations.parquet"
    write_parquet(unified_path, unified_rows)
    write_parquet(obs_path, obs_rows)
    monkeypatch.setattr(stage_final_output, "OBS_INPUT", obs_path)
    monkeypatch.setattr(stage_final_output, "STAGE_INPUTS", {})

    rows_written = stage_final_output.build_unified_observations(unified_path, output_path)
    report = stage_final_output.compute_quality_report(unified_path, output_path)

    assert rows_written == 2
    assert report["funnel"]["final_unified"] == 2
    assert report["funnel"]["final_observations"] == 2
    assert report["columns"]["unified_observations"] == pd.read_parquet(output_path).shape[1]
    assert output_path.exists()
    assert set(pd.read_parquet(output_path)["uid"]) == {"uid-1", "uid-2"}
    assert "final_observations" in report["funnel"]
