import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from tests.helpers.imports import load_module


stage1 = load_module("stage1_ingest", "preprocessing/scripts/stage1_ingest.py")
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
SYNTHETIC_STAGE1 = FIXTURE_ROOT / "synthetic" / "stage1"
SAMPLED_STAGE1 = FIXTURE_ROOT / "sampled" / "stage1"
SAMPLED_ROWS = json.loads((SAMPLED_STAGE1 / "cases.json").read_text())["rows"]


@pytest.mark.unit
def test_finalize_frame_writes_declared_output_columns_only():
    frame = stage1.finalize_frame(
        stage1.pd.DataFrame(
            [
                {
                    "uid": "x1",
                    "source": "scraped",
                    "source_platform": "linkedin",
                    "site": "linkedin",
                    "title": "Software Engineer",
                    "company_name": "Example Co",
                    "location": "Seattle, WA",
                    "date_posted": "2026-03-21",
                    "scrape_date": "2026-03-22",
                    "description": "Build systems",
                    "description_raw": "Build systems",
                    "seniority_native": "entry",
                    "company_industry": None,
                    "company_size": None,
                    "company_size_raw": None,
                    "company_size_category": None,
                    "search_query": "software engineer",
                    "query_tier": "tier1",
                    "search_metro_id": "1",
                    "search_metro_name": "Seattle Metro",
                    "search_metro_region": "west",
                    "search_location": "Seattle",
                    "work_type": "Full-time",
                    "is_remote": False,
                    "job_url": "https://example.com/job",
                    "skills_raw": None,
                    "asaniczka_skills": None,
                    "company_id_kaggle": None,
                    "unexpected": "drop me",
                }
            ]
        )
    )

    assert list(frame.columns) == stage1.OUTPUT_COLUMNS
    assert "unexpected" not in frame.columns
    assert "is_swe" not in frame.columns
    assert frame.loc[0, "job_id"] == "x1"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Entry level", "entry"),
        ("Mid senior", "mid-senior"),
        ("Not Applicable", None),
        ("mystery", None),
    ],
)
def test_map_seniority_is_schema_normalization_only(raw, expected):
    assert stage1.map_seniority(raw) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,001-5,000", 3000.5),
        ("10,001+", 10001.0),
        ("", None),
    ],
)
def test_parse_company_size_handles_ranges_and_garbage(raw, expected):
    value = stage1.parse_company_size(raw)
    if expected is None:
        assert pd.isna(value)
    else:
        assert value == expected


@pytest.mark.unit
def test_normalize_date_series_formats_and_coerces_bad_values():
    result = stage1.normalize_date_series(pd.Series(["2026-03-21", "not-a-date"]))
    assert result.iloc[0] == "2026-03-21"
    assert pd.isna(result.iloc[1])


@pytest.mark.unit
def test_list_scraped_files_filters_file_policy(tmp_path, monkeypatch):
    (tmp_path / "2026-03-21_swe_jobs.csv").write_text("ok")
    (tmp_path / "2026-03-21_non_swe_jobs.csv").write_text("ok")
    (tmp_path / "2026-03-21_yc_jobs.csv").write_text("ok")
    (tmp_path / "2026-03-21_jobs.csv").write_text("ok")
    (tmp_path / "2026-03-21_swe_jobs.txt").write_text("ok")
    monkeypatch.setattr(stage1, "SCRAPED_DIR", tmp_path)

    files = [path.name for path in stage1.list_scraped_files()]

    assert files == ["2026-03-21_non_swe_jobs.csv", "2026-03-21_swe_jobs.csv"]


@pytest.mark.integration
def test_load_scraped_keeps_canonical_and_observations_split(tmp_path, monkeypatch):
    monkeypatch.setattr(stage1, "SCRAPED_DIR", tmp_path)
    for filename, fixture_name in [
        ("2026-03-21_swe_jobs.csv", "valid_scraped_row.csv"),
        ("2026-03-22_non_swe_jobs.csv", "valid_scraped_row.csv"),
        ("2026-03-23_swe_jobs.csv", "invalid_scraped_row.csv"),
    ]:
        shutil.copyfile(SYNTHETIC_STAGE1 / fixture_name, tmp_path / filename)

    summary: dict[str, object] = {}
    canonical, observations = stage1.load_scraped(summary)

    assert summary["scraped_files_loaded"] == [
        "2026-03-21_swe_jobs.csv",
        "2026-03-22_non_swe_jobs.csv",
    ]
    assert summary["scraped_files_skipped"] == [
        {
            "file": "2026-03-23_swe_jobs.csv",
            "reason": "unexpected_column_count",
            "column_count": 40,
        }
    ]
    assert summary["scraped_observation_rows"] == 2
    assert summary["scraped_unique_ids"] == 1
    assert list(canonical.columns) == stage1.OUTPUT_COLUMNS
    assert list(observations.columns) == stage1.OUTPUT_COLUMNS
    assert len(canonical) == 1
    assert len(observations) == 2
    assert canonical.loc[0, "uid"] == "linkedin_ln-001"
    assert canonical.loc[0, "job_id"] == canonical.loc[0, "uid"]
    assert canonical.loc[0, "scrape_date"] == "2026-03-21"
    assert "is_swe" not in canonical.columns


@pytest.mark.sampled
@pytest.mark.parametrize("row", SAMPLED_ROWS)
def test_sampled_stage1_rows_match_reviewed_expectations(row):
    assert row["job_id"] == row["uid"]
    assert row["source"] in {"kaggle_asaniczka", "scraped"}
    assert row["seniority_native"] in {"mid-senior", None}
    assert row["search_metro_name"]
    if row["uid"] == "asaniczka_70c0f7742486e930":
        assert row["description_missing"] is True
        assert row["seniority_native"] == "mid-senior"
    if row["uid"] == "linkedin_li-3736674638":
        assert row["description_missing"] is False
        assert row["seniority_native"] is None
