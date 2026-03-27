import json
from pathlib import Path

import pandas as pd
import pytest

from tests.helpers.imports import load_module
from tests.helpers.stage_runner import patch_stage_dirs, write_parquet
from tests.helpers.parquet_asserts import assert_has_columns, assert_unique


stage4 = load_module("stage4_dedup", "preprocessing/scripts/stage4_dedup.py")
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
SYNTHETIC = json.loads((FIXTURE_ROOT / "synthetic" / "stage4" / "cases.json").read_text())
SAMPLED = json.loads((FIXTURE_ROOT / "sampled" / "stage4" / "cases.json").read_text())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  Example, Inc.  ", "example inc"),
        ("Seattle, WA, US", "seattle wa"),
        ("Lead Software Engineer - Remote", "lead software engineer"),
    ],
)
def test_normalize_helpers_keep_dedup_keys_stable(value, expected):
    if "US" in value:
        assert stage4._normalize_location_for_dedup(value) == expected
    elif "Remote" in value:
        assert stage4._normalize_title_for_dedup(value) == expected
    else:
        assert stage4._normalize_entity_text(value) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("data engineer", "senior data engineer", True),
        ("mechanical engineer", "electrical engineer", False),
        ("junior developer", "senior developer", False),
    ],
)
def test_titles_are_near_duplicates_only_for_safe_pairs(left, right, expected):
    assert stage4._titles_are_near_duplicates(left, right) is expected


@pytest.mark.unit
def test_hash_text_normalizes_whitespace_and_case():
    assert stage4._hash_text("  Hello   World  ") == stage4._hash_text("hello world")
    assert stage4._hash_text("") == ""


@pytest.mark.sampled
@pytest.mark.parametrize("case", SAMPLED["lookup_rows"])
def test_sampled_lookup_rows_cover_canonicalization_methods(case, tmp_path):
    input_path = tmp_path / "lookup_input.parquet"
    output_path = tmp_path / "lookup_output.parquet"
    write_parquet(
        input_path,
        [{"company_name_effective": row["company_name_effective"]} for row in SAMPLED["lookup_rows"]],
    )

    lookup_df, _, _, _ = stage4.build_stage4_company_lookup(input_path, output_path)
    row = lookup_df.loc[lookup_df["company_name_effective"] == case["company_name_effective"]].iloc[0]
    assert row["company_name_canonical"] == case["company_name_canonical"]
    assert row["company_name_canonical_method"] == case["company_name_canonical_method"]


@pytest.mark.sampled
@pytest.mark.parametrize("case", SAMPLED["multi_location_rows"])
def test_sampled_multi_location_rows_stay_flagged(case):
    assert case["is_multi_location"] is True
    assert case["company_name_canonical_method"] == "passthrough"


@pytest.mark.unit
def test_pass1_build_index_handles_exact_and_near_duplicate_groups(tmp_path):
    rows = []
    for group in SYNTHETIC["exact_job_id"] + SYNTHETIC["exact_opening"] + SYNTHETIC["near_duplicate"] + SYNTHETIC["meaningfully_different"] + SYNTHETIC["multi_location"]:
        rows.append(group)
    input_path = tmp_path / "stage4_input.parquet"
    write_parquet(input_path, rows)

    canonical_map = {row["company_name_effective"]: row["company_name_effective"] for row in rows}
    keep_indices, multi_loc_indices, funnel = stage4.pass1_build_index(input_path, "company_name_effective", canonical_map)

    assert len(rows) > len(keep_indices)
    assert any(f["after_near"] < f["raw"] for f in funnel.values())
    assert len(multi_loc_indices) == 2

    kept = pd.read_parquet(input_path).loc[sorted(keep_indices)]
    assert "dup-1" in kept["job_id"].tolist()
    assert "open-1" in kept["job_id"].tolist()
    assert "open-2" not in kept["job_id"].tolist()
    assert "near-2" in kept["job_id"].tolist()
    assert "near-1" not in kept["job_id"].tolist()
    assert {"diff-1", "diff-2"} <= set(kept["job_id"].tolist())


@pytest.mark.integration
def test_run_stage4_preserves_unique_uids_and_flags_multi_location(tmp_path, monkeypatch):
    root = tmp_path / "project"
    data_dir = root / "data"
    intermediate_dir = root / "preprocessing" / "intermediate"
    logs_dir = root / "preprocessing" / "logs"
    intermediate_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    patch_stage_dirs(
        monkeypatch,
        stage4,
        {"root": root, "data": data_dir, "intermediate": intermediate_dir, "logs": logs_dir},
    )
    monkeypatch.setattr(stage4.log, "disabled", True)

    rows = []
    for row in SYNTHETIC["exact_job_id"] + SYNTHETIC["exact_opening"] + SYNTHETIC["near_duplicate"] + SYNTHETIC["meaningfully_different"] + SYNTHETIC["multi_location"]:
        rows.append(row)
    write_parquet(intermediate_dir / "stage3_boilerplate.parquet", rows)

    stage4.run_stage4()
    output_path = intermediate_dir / "stage4_dedup.parquet"
    assert_has_columns(output_path, ["company_name_canonical", "company_name_canonical_method", "is_multi_location"])
    assert_unique(output_path, ["uid"])

    out = pd.read_parquet(output_path)
    assert len(out) < len(rows)
    assert out["is_multi_location"].any()
