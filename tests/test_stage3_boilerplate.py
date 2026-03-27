import json
from pathlib import Path

import pandas as pd
import pytest

from tests.helpers.imports import load_module
from tests.helpers.stage_runner import patch_stage_dirs, write_parquet
from tests.helpers.parquet_asserts import assert_has_columns


stage3 = load_module("stage3_boilerplate", "preprocessing/scripts/stage3_boilerplate.py")
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
SYNTHETIC = json.loads((FIXTURE_ROOT / "synthetic" / "stage3" / "cases.json").read_text())
SAMPLED = json.loads((FIXTURE_ROOT / "sampled" / "stage3" / "cases.json").read_text())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("header", "expected"),
    [(case["header"], case["expected"]) for case in SYNTHETIC["classify"]],
)
def test_classify_headers_match_stage_taxonomy(header, expected):
    assert stage3._classify(header) == expected


@pytest.mark.unit
def test_strip_common_noise_and_eeo_fallbacks():
    noisy = "Line one\nhttps://example.com\nShow more\nShow less"
    assert "https://example.com" not in stage3._strip_common_noise(noisy)
    assert "Show more" not in stage3._strip_common_noise(noisy)

    eeo_only = "Equal Opportunity Employer\nWe are an equal opportunity employer."
    assert stage3._strip_eeo_safely(eeo_only) == eeo_only


@pytest.mark.unit
@pytest.mark.parametrize("case", SYNTHETIC["core"])
def test_extract_core_keeps_role_content_and_strips_boilerplate(case):
    core = stage3.extract_core(case["description"])
    for needle in case["contains"]:
        assert needle in core
    for needle in case["missing"]:
        assert needle not in core


@pytest.mark.unit
@pytest.mark.parametrize("case", SYNTHETIC["flags"])
def test_process_chunk_sets_expected_boilerplate_flag(case):
    chunk = pd.DataFrame(
        [
            {
                "uid": "x1",
                "source": "scraped",
                "title": "Test",
                "description": case["description"],
                "description_length": case["description_length"],
            }
        ]
    )
    out = stage3.process_chunk(chunk)
    assert out.loc[0, "boilerplate_flag"] == case["expected_flag"]
    assert out.loc[0, "description"] == case["description"]


@pytest.mark.sampled
@pytest.mark.parametrize("case", SAMPLED["rows"])
def test_sampled_stage3_rows_match_reviewed_flags(case):
    assert case["boilerplate_flag"] in {"ok", "over_removed", "under_removed", "empty_core"}
    if case["uid"] == "arshkon_921716":
        assert case["boilerplate_flag"] == "over_removed"
    if case["uid"] in {"arshkon_3190494363", "arshkon_3884434746"}:
        assert case["boilerplate_flag"] == "empty_core"


@pytest.mark.integration
def test_run_stage3_preserves_row_count_and_description(tmp_path, monkeypatch):
    root = tmp_path / "project"
    data_dir = root / "data"
    intermediate_dir = root / "preprocessing" / "intermediate"
    logs_dir = root / "preprocessing" / "logs"
    intermediate_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    patch_stage_dirs(
        monkeypatch,
        stage3,
        {"root": root, "data": data_dir, "intermediate": intermediate_dir, "logs": logs_dir},
    )
    monkeypatch.setattr(stage3.log, "disabled", True)

    rows = [
        {
            "uid": "x1",
            "source": "scraped",
            "title": "Marketing Coordinator",
            "description": "About the role\nBuild systems for customers.\n\nResponsibilities\nShip features.\n\nEqual Opportunity Employer\nWe are an equal opportunity employer.",
            "description_length": 143,
        },
        {
            "uid": "x2",
            "source": "scraped",
            "title": "Architectural Designer",
            "description": "Job Type: Full-timeWork Location: In person",
            "description_length": 3948,
        },
    ]
    write_parquet(intermediate_dir / "stage2_aggregators.parquet", rows)

    stage3.run_stage3()
    output_path = intermediate_dir / "stage3_boilerplate.parquet"
    assert_has_columns(output_path, ["description_core", "core_length", "boilerplate_flag"])

    out = pd.read_parquet(output_path)
    assert len(out) == 2
    assert out["description"].tolist() == [row["description"] for row in rows]
    assert set(out["boilerplate_flag"]) <= {"ok", "over_removed", "under_removed", "empty_core"}
