import json
from pathlib import Path

import pandas as pd
import pytest

from tests.helpers.imports import load_module
from tests.helpers.stage_runner import patch_stage_dirs, write_parquet
from tests.helpers.parquet_asserts import assert_has_columns


stage2 = load_module("stage2_aggregators", "preprocessing/scripts/stage2_aggregators.py")
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
SYNTHETIC = json.loads((FIXTURE_ROOT / "synthetic" / "stage2" / "cases.json").read_text())
SAMPLED = json.loads((FIXTURE_ROOT / "sampled" / "stage2" / "cases.json").read_text())


@pytest.mark.unit
@pytest.mark.parametrize("case", SYNTHETIC["aggregators"])
def test_is_aggregator_matches_precision_first_rules(case):
    assert stage2.is_aggregator(case["company_name"]) is case["expected"]


@pytest.mark.unit
@pytest.mark.parametrize("case", SYNTHETIC["real_employers"])
def test_extract_real_employer_rejects_fragments_and_finds_orgs(case):
    assert stage2.extract_real_employer(case["description"], case["company_name"]) == case["expected"]


@pytest.mark.sampled
@pytest.mark.parametrize("case", SAMPLED["rows"])
def test_sampled_real_employer_rows_match_reviewed_expectations(case):
    assert stage2.is_aggregator(case["company_name"]) is case["expected_aggregator"]
    assert stage2.extract_real_employer(case["description"], case["company_name"]) == case["expected_real_employer"]


@pytest.mark.integration
def test_run_stage2_preserves_rows_and_sets_company_effective(tmp_path, monkeypatch):
    root = tmp_path / "project"
    data_dir = root / "data"
    intermediate_dir = root / "preprocessing" / "intermediate"
    logs_dir = root / "preprocessing" / "logs"
    intermediate_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    patch_stage_dirs(
        monkeypatch,
        stage2,
        {"root": root, "data": data_dir, "intermediate": intermediate_dir, "logs": logs_dir},
    )
    monkeypatch.setattr(stage2.log, "disabled", True)

    rows = [
        {
            "uid": "keep-1",
            "source": "scraped",
            "source_platform": "linkedin",
            "site": "linkedin",
            "title": "Software Engineer",
            "company_name": "Acme Labs",
            "location": "Seattle WA",
            "description": "Build systems for customers.",
            "description_length": 28,
            "skills_raw": None,
            "seniority_native": None,
        },
        {
            "uid": "agg-pos",
            "source": "scraped",
            "source_platform": "linkedin",
            "site": "linkedin",
            "title": "Cobol Developer",
            "company_name": "Dice",
            "location": "Remote",
            "description": "Our client, Avenues International, Inc., is seeking the following. Apply via Dice today.",
            "description_length": 88,
            "skills_raw": None,
            "seniority_native": None,
        },
        {
            "uid": "agg-neg",
            "source": "scraped",
            "source_platform": "linkedin",
            "site": "linkedin",
            "title": "Benefits Coordinator",
            "company_name": "Capgemini",
            "location": "Remote",
            "description": "Job Description: the text mentions vendors and benefits administration but no real employer clause.",
            "description_length": 99,
            "skills_raw": None,
            "seniority_native": None,
        },
    ]
    write_parquet(intermediate_dir / "stage1_unified.parquet", rows)

    stage2.run_stage2()
    output_path = intermediate_dir / "stage2_aggregators.parquet"
    assert_has_columns(output_path, ["is_aggregator", "real_employer", "company_name_effective"])

    out = pd.read_parquet(output_path)
    assert len(out) == 3
    assert out.loc[out["uid"] == "keep-1", "company_name_effective"].item() == "Acme Labs"
    assert out.loc[out["uid"] == "agg-pos", "company_name_effective"].item() == "Avenues International"
    assert out.loc[out["uid"] == "agg-neg", "company_name_effective"].item() == "Capgemini"
    assert out.loc[out["uid"] == "agg-pos", "real_employer"].item() == "Avenues International"
    assert pd.isna(out.loc[out["uid"] == "agg-neg", "real_employer"].item())
