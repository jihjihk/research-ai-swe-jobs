from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.helpers.imports import load_module
from tests.helpers.parquet_asserts import parquet_columns


build_core_mod = load_module(
    "build_unified_core",
    "preprocessing/scripts/build_unified_core.py",
)


def _typed_defaults() -> dict[str, object]:
    """Concrete values for every source column the projection depends on.
    Using non-null defaults avoids pyarrow inferring all-null columns as
    ``pa.null()`` which trips up the DuckDB filter. Fixture rows override
    whatever they care about.
    """
    str_cols = {
        "uid", "source", "period", "date_posted", "scrape_date",
        "title", "description", "description_core_llm",
        "company_name_effective", "company_name_canonical", "company_industry",
        "seniority_final", "seniority_final_source",
        "seniority_rule", "seniority_rule_source", "seniority_native",
        "location", "city_extracted", "state_normalized", "metro_area",
        "date_flag", "ghost_assessment_llm",
        "llm_extraction_coverage", "llm_classification_coverage",
    }
    bool_cols = {
        "is_aggregator", "is_swe_combined_llm", "is_control",
        "is_remote_inferred", "is_multi_location",
    }
    int_cols = {"yoe_extracted", "yoe_min_years_llm", "company_size"}
    out: dict[str, object] = {}
    for col in build_core_mod.SOURCE_COLUMNS_REQUIRED:
        if col in str_cols:
            out[col] = "x"
        elif col in bool_cols:
            out[col] = False
        elif col in int_cols:
            out[col] = 0
        else:
            raise AssertionError(f"Unclassified source column in test defaults: {col}")
    return out


def _make_row(uid: str, *, selected: bool, **overrides) -> dict:
    row = _typed_defaults()
    row["uid"] = uid
    row["source"] = "kaggle_arshkon"
    row["scrape_date"] = "2024-01-15"
    row["date_flag"] = "ok"
    row.update(overrides)
    row["selected_for_llm_frame"] = selected
    return row


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_unified_and_obs(tmp_path: Path) -> tuple[Path, Path]:
    rows = [
        _make_row("u1", selected=True, is_swe_combined_llm=True),
        _make_row("u2", selected=True, is_control=True),
        _make_row("u3", selected=False, is_swe_combined_llm=True),  # not selected
        _make_row("u4", selected=False, is_swe_combined_llm=True),  # not selected
        _make_row("u5", selected=True, is_swe_combined_llm=True),
        # In-frame but neither LLM-SWE nor rule-control (LLM rejected rule-SWE):
        # must be dropped from unified_core under the new analysis-frame filter.
        _make_row("u6", selected=True, is_swe_combined_llm=False, is_control=False),
    ]
    unified_path = tmp_path / "unified.parquet"
    _write_parquet(unified_path, rows)

    # Daily observations: a couple of dropped rows should not leak in.
    obs_rows = [
        {"uid": "u1", "scrape_date": "2024-01-15"},
        {"uid": "u1", "scrape_date": "2024-01-16"},
        {"uid": "u2", "scrape_date": "2024-01-15"},
        {"uid": "u3", "scrape_date": "2024-01-15"},  # not selected -> excluded
        {"uid": "u5", "scrape_date": "2024-01-17"},
    ]
    obs_path = tmp_path / "observations.parquet"
    _write_parquet(obs_path, obs_rows)
    return unified_path, obs_path


@pytest.mark.unit
def test_core_row_count_matches_filter(tmp_path):
    unified_path, obs_path = _write_unified_and_obs(tmp_path)
    core_path = tmp_path / "core.parquet"
    core_obs_path = tmp_path / "core_obs.parquet"

    summary = build_core_mod.build_core(unified_path, obs_path, core_path, core_obs_path)

    # Filter keeps selected_for_llm_frame rows that are LLM-SWE OR rule-control.
    expected = duckdb.execute(
        f"""
        SELECT count(*) FROM read_parquet('{unified_path}')
        WHERE selected_for_llm_frame = TRUE
          AND (is_swe_combined_llm = TRUE OR is_control = TRUE)
        """
    ).fetchone()[0]
    actual = duckdb.execute(
        f"SELECT count(*) FROM read_parquet('{core_path}')"
    ).fetchone()[0]
    # u1, u2, u5 pass; u6 (in-frame but LLM-rejected rule-SWE) is dropped.
    assert actual == expected == 3
    assert summary["core_rows"] == actual
    assert "selected_for_llm_frame = TRUE" in summary["row_filter"]
    assert "is_swe_combined_llm = TRUE OR is_control = TRUE" in summary["row_filter"]


@pytest.mark.unit
def test_core_columns_exactly_match_core_columns_constant(tmp_path):
    unified_path, obs_path = _write_unified_and_obs(tmp_path)
    core_path = tmp_path / "core.parquet"
    core_obs_path = tmp_path / "core_obs.parquet"

    build_core_mod.build_core(unified_path, obs_path, core_path, core_obs_path)

    assert parquet_columns(core_path) == list(build_core_mod.CORE_COLUMNS)
    # Observations share the same column set (scrape_date is just overridden).
    assert parquet_columns(core_obs_path) == list(build_core_mod.CORE_COLUMNS)


@pytest.mark.unit
def test_core_observations_uids_are_subset_of_core_uids(tmp_path):
    unified_path, obs_path = _write_unified_and_obs(tmp_path)
    core_path = tmp_path / "core.parquet"
    core_obs_path = tmp_path / "core_obs.parquet"

    build_core_mod.build_core(unified_path, obs_path, core_path, core_obs_path)

    core_uids = {
        r[0] for r in duckdb.execute(
            f"SELECT DISTINCT uid FROM read_parquet('{core_path}')"
        ).fetchall()
    }
    obs_uids = {
        r[0] for r in duckdb.execute(
            f"SELECT DISTINCT uid FROM read_parquet('{core_obs_path}')"
        ).fetchall()
    }
    assert obs_uids.issubset(core_uids)
    # u3 is in obs but not selected -> must be dropped from core_obs.
    assert "u3" not in obs_uids

    # scrape_date on core_obs comes from the observations side.
    n_obs_for_u1 = duckdb.execute(
        f"""
        SELECT count(*) FROM read_parquet('{core_obs_path}')
        WHERE uid = 'u1'
        """
    ).fetchone()[0]
    assert n_obs_for_u1 == 2


@pytest.mark.unit
def test_build_core_raises_when_filter_column_missing(tmp_path):
    row = _typed_defaults()
    row["uid"] = "u1"
    unified_path = tmp_path / "unified.parquet"
    _write_parquet(unified_path, [row])  # no selected_for_llm_frame

    obs_path = tmp_path / "obs.parquet"
    _write_parquet(obs_path, [{"uid": "u1", "scrape_date": "2024-01-15"}])

    with pytest.raises(ValueError, match="selected_for_llm_frame"):
        build_core_mod.build_core(
            unified_path, obs_path,
            tmp_path / "core.parquet", tmp_path / "core_obs.parquet",
        )


@pytest.mark.unit
def test_build_core_raises_when_required_column_missing(tmp_path):
    row = _make_row("u1", selected=True, is_swe_combined_llm=True)
    row.pop("description_core_llm")
    unified_path = tmp_path / "unified.parquet"
    _write_parquet(unified_path, [row])

    obs_path = tmp_path / "obs.parquet"
    _write_parquet(obs_path, [{"uid": "u1", "scrape_date": "2024-01-15"}])

    with pytest.raises(ValueError, match="description_core_llm"):
        build_core_mod.build_core(
            unified_path, obs_path,
            tmp_path / "core.parquet", tmp_path / "core_obs.parquet",
        )


@pytest.mark.unit
def test_has_core_filter_column(tmp_path):
    # With the column
    unified_with = tmp_path / "with.parquet"
    _write_parquet(unified_with, [_make_row("u1", selected=True, is_swe_combined_llm=True)])
    assert build_core_mod.has_core_filter_column(unified_with) is True

    # Without the column
    row = _typed_defaults()
    row["uid"] = "u1"
    unified_without = tmp_path / "without.parquet"
    _write_parquet(unified_without, [row])
    assert build_core_mod.has_core_filter_column(unified_without) is False
