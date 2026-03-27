import json
from pathlib import Path

import duckdb
import pytest

from tests.helpers.imports import load_module
from tests.helpers.parquet_asserts import (
    assert_row_count_equal,
    assert_unique,
    parquet_columns,
)
from tests.helpers.stage_runner import patch_stage_dirs, write_parquet


stage_final = load_module(
    "stage_final_output",
    "preprocessing/scripts/stage_final_output.py",
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


def load_rows(*parts: str) -> list[dict]:
    return json.loads(FIXTURE_ROOT.joinpath(*parts).read_text())


def run_main_case(monkeypatch, temp_stage_dirs, fixture_name: str):
    unified_rows = load_rows(fixture_name, "final", "unified_rows.json")
    obs_rows = load_rows(fixture_name, "final", "observations_rows.json")

    unified_input = temp_stage_dirs["intermediate"] / f"{fixture_name}_unified.parquet"
    obs_input = temp_stage_dirs["intermediate"] / f"{fixture_name}_observations.parquet"
    write_parquet(unified_input, unified_rows)
    write_parquet(obs_input, obs_rows)

    patch_stage_dirs(monkeypatch, stage_final, temp_stage_dirs)
    monkeypatch.setattr(stage_final, "UNIFIED_INPUT", unified_input)
    monkeypatch.setattr(stage_final, "OBS_INPUT", obs_input)
    monkeypatch.setattr(stage_final, "UNIFIED_OUTPUT", temp_stage_dirs["data"] / "unified.parquet")
    monkeypatch.setattr(
        stage_final,
        "OBS_OUTPUT",
        temp_stage_dirs["data"] / "unified_observations.parquet",
    )
    monkeypatch.setattr(
        stage_final,
        "QUALITY_OUTPUT",
        temp_stage_dirs["data"] / "quality_report.json",
    )
    monkeypatch.setattr(
        stage_final,
        "LOG_OUTPUT",
        temp_stage_dirs["data"] / "preprocessing_log.txt",
    )
    monkeypatch.setattr(
        stage_final,
        "STAGE_INPUTS",
        {
            "stage1_unified": unified_input,
            "stage1_observations": obs_input,
        },
    )

    stage_final.main()

    unified_output = stage_final.UNIFIED_OUTPUT
    obs_output = stage_final.OBS_OUTPUT
    quality_output = stage_final.QUALITY_OUTPUT
    log_output = stage_final.LOG_OUTPUT
    report = json.loads(quality_output.read_text())
    computed_report = stage_final.compute_quality_report(unified_output, obs_output)
    log_text = log_output.read_text()

    return {
        "unified_input": unified_input,
        "obs_input": obs_input,
        "unified_output": unified_output,
        "obs_output": obs_output,
        "report": report,
        "computed_report": computed_report,
        "log_text": log_text,
        "unified_rows": unified_rows,
        "obs_rows": obs_rows,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fixture_name", "expected"),
    [
        (
            "synthetic",
            {
                "joined_rows": [
                    ("syn_adjacent", "2026-03-21"),
                    ("syn_control", "2019-12-31"),
                    ("syn_swe", "2026-03-20"),
                    ("syn_swe", "2026-03-21"),
                ],
                "funnel": {
                    "final_unified": 3,
                    "final_observations": 4,
                    "final_swe": 1,
                    "final_control": 1,
                    "final_adjacent": 1,
                },
                "quality_flags": {
                    "aggregators": 1,
                    "date_flagged": 1,
                    "non_english": 1,
                    "ghost_flagged": 1,
                },
                "coverage": 0.6667,
            },
        ),
        (
            "sampled",
            {
                "joined_rows": [
                    ("arshkon_1014822088", "2024-04-18"),
                    ("linkedin_li-3696371940", "2026-03-20"),
                    ("linkedin_li-3696371940", "2026-03-21"),
                    ("linkedin_li-3736674638", "2026-03-20"),
                    ("linkedin_li-3736674638", "2026-03-21"),
                ],
                "funnel": {
                    "final_unified": 3,
                    "final_observations": 5,
                    "final_swe": 1,
                    "final_control": 1,
                    "final_adjacent": 0,
                },
                "quality_flags": {
                    "aggregators": 0,
                    "date_flagged": 0,
                    "non_english": 0,
                    "ghost_flagged": 0,
                },
                "coverage": 0.6667,
            },
        ),
    ],
)
def test_main_writes_fixture_driven_outputs(monkeypatch, temp_stage_dirs, fixture_name, expected):
    result = run_main_case(monkeypatch, temp_stage_dirs, fixture_name)

    assert parquet_columns(result["unified_output"]) == parquet_columns(result["unified_input"])
    assert parquet_columns(result["obs_output"]) == parquet_columns(result["unified_input"])
    assert_row_count_equal(result["unified_output"], result["unified_input"])
    assert_row_count_equal(result["obs_output"], result["obs_input"])
    assert_unique(result["unified_output"], ["uid"])

    joined_rows = duckdb.execute(
        f"""
        SELECT uid, scrape_date
        FROM read_parquet('{result["obs_output"]}')
        ORDER BY uid, scrape_date
        """
    ).fetchall()
    assert joined_rows == expected["joined_rows"]

    assert result["report"] == result["computed_report"]
    assert result["log_text"] == stage_final.build_log_text(result["report"])
    assert result["report"]["columns"]["unified"] == len(parquet_columns(result["unified_input"]))
    assert result["report"]["columns"]["unified_observations"] == len(
        parquet_columns(result["obs_output"])
    )
    assert result["report"]["row_counts"] == {
        "stage1_unified": len(result["unified_rows"]),
        "stage1_observations": len(result["obs_rows"]),
    }
    assert result["report"]["funnel"] == expected["funnel"]
    assert result["report"]["quality_flags"] == expected["quality_flags"]
    assert result["report"]["classification_rates"]["description_core_llm_coverage"] == expected[
        "coverage"
    ]
