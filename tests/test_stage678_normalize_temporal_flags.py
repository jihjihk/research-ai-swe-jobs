import json
from pathlib import Path

import pandas as pd
import pytest

from tests.helpers.imports import load_module


stage678 = load_module(
    "stage678_normalize_temporal_flags",
    "preprocessing/scripts/stage678_normalize_temporal_flags.py",
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
MISSING = object()


def load_rows(*parts: str) -> list[dict]:
    return json.loads(FIXTURE_ROOT.joinpath(*parts).read_text())


def load_df(*parts: str) -> pd.DataFrame:
    return pd.DataFrame(load_rows(*parts))


def assert_row_subset(row: pd.Series, expected: dict) -> None:
    for column, value in expected.items():
        actual = row[column]
        if value is MISSING or value is None:
            assert pd.isna(actual), f"{column} expected missing, got {actual!r}"
        else:
            assert actual == value, f"{column} expected {value!r}, got {actual!r}"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("scrape_date", "date_posted", "expected"),
    [
        ("2026-03-20", "2026-03-19", "ok"),
        ("2026-03-20", "2027-01-01", "ok"),
        ("2026-03-20", "2019-12-31", "date_posted_out_of_range"),
        ("2019-12-31", "2026-03-19", "scrape_date_out_of_range"),
        ("not-a-date", "2026-03-19", "scrape_date_invalid"),
    ],
)
def test_validate_dates_is_parseability_plus_2020_floor(scrape_date, date_posted, expected):
    assert stage678.validate_dates("scraped", "linkedin", scrape_date, date_posted) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("location", "expected"),
    [
        (
            "Seattle, WA",
            {
                "city_extracted": "Seattle",
                "state_normalized": "WA",
                "country_extracted": "US",
                "is_remote_location": False,
            },
        ),
        (
            "Austin, Texas",
            {
                "city_extracted": "Austin",
                "state_normalized": "TX",
                "country_extracted": "US",
                "is_remote_location": False,
            },
        ),
        (
            "New Jersey, United States",
            {
                "city_extracted": None,
                "state_normalized": "NJ",
                "country_extracted": "US",
                "is_remote_location": False,
            },
        ),
        (
            "Remote",
            {
                "city_extracted": None,
                "state_normalized": None,
                "country_extracted": None,
                "is_remote_location": True,
            },
        ),
    ],
)
def test_normalize_location_parses_common_location_shapes(location, expected):
    assert stage678.normalize_location(location) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "row_source",
        "row_location",
        "row_search_metro_name",
        "row_is_remote",
        "row_is_remote_inferred",
        "city_extracted",
        "state_normalized",
        "metro_aliases",
        "city_state_lookup",
        "city_state_reference_lookup",
        "expected",
    ),
    [
        (
            "scraped",
            "Fort Myers, FL",
            "Seattle Metro",
            False,
            False,
            "Fort Myers",
            "FL",
            {},
            {"fort myers|FL": ("Cape Coral-Fort Myers Metro", "high")},
            {"fort myers|FL": ("Cape Coral-Fort Myers Metro", "medium")},
            {
                "metro_area": "Seattle Metro",
                "metro_source": "search_metro",
                "metro_confidence": "high",
            },
        ),
        (
            "kaggle_arshkon",
            "Seattle Metro",
            None,
            False,
            False,
            None,
            None,
            {"seattle metro": "Seattle Metro"},
            {},
            {},
            {
                "metro_area": "Seattle Metro",
                "metro_source": "manual_alias",
                "metro_confidence": "high",
            },
        ),
        (
            "kaggle_arshkon",
            "Seattle, WA",
            None,
            False,
            False,
            "Seattle",
            "WA",
            {},
            {"seattle|WA": ("Seattle Metro", "high")},
            {},
            {
                "metro_area": "Seattle Metro",
                "metro_source": "city_state_lookup",
                "metro_confidence": "high",
            },
        ),
        (
            "kaggle_arshkon",
            "Seattle, WA",
            None,
            False,
            False,
            "Seattle",
            "WA",
            {},
            {},
            {"seattle|WA": ("Seattle Metro", "medium")},
            {
                "metro_area": "Seattle Metro",
                "metro_source": "city_state_reference",
                "metro_confidence": "medium",
            },
        ),
        (
            "scraped",
            "Seattle, WA",
            "Seattle Metro",
            True,
            False,
            "Seattle",
            "WA",
            {},
            {"seattle|WA": ("Seattle Metro", "high")},
            {"seattle|WA": ("Seattle Metro", "medium")},
            {
                "metro_area": None,
                "metro_source": "unresolved",
                "metro_confidence": "low",
            },
        ),
    ],
)
def test_infer_metro_respects_precedence_and_remote_guard(
    row_source,
    row_location,
    row_search_metro_name,
    row_is_remote,
    row_is_remote_inferred,
    city_extracted,
    state_normalized,
    metro_aliases,
    city_state_lookup,
    city_state_reference_lookup,
    expected,
):
    assert stage678.infer_metro(
        row_source,
        row_location,
        row_search_metro_name,
        row_is_remote,
        row_is_remote_inferred,
        city_extracted,
        state_normalized,
        metro_aliases,
        city_state_lookup,
        city_state_reference_lookup,
    ) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("We build reliable software systems and ship production code.", True),
        ("これは日本語です。", False),
        ("", MISSING),
        (None, MISSING),
    ],
)
def test_detect_language_handles_english_non_english_and_missing(text, expected):
    actual = stage678.detect_language(text)
    if expected is MISSING:
        assert pd.isna(actual)
    else:
        assert actual is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("description", "expected"),
    [
        (None, "empty"),
        ("", "empty"),
        ("short text", "too_short"),
        ("This description is long enough to pass the minimal quality screen.", "ok"),
    ],
)
def test_assess_description_quality_thresholds(description, expected):
    assert stage678.assess_description_quality(description) == expected


@pytest.mark.unit
def test_build_description_hash_is_stable_and_nullable():
    text = "Stable description text for hashing"
    assert stage678.build_description_hash(text) == stage678.build_description_hash(text)
    assert stage678.build_description_hash(text) != stage678.build_description_hash(text + "!")
    assert stage678.build_description_hash(" ") is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("seniority_final", "yoe_extracted", "yoe_contradiction", "expected"),
    [
        ("entry", 0, False, "low"),
        ("entry", 3, False, "medium"),
        ("entry", 5, False, "high"),
        ("entry", 3, True, "medium"),
        ("mid-senior", 7, True, "low"),
    ],
)
def test_detect_ghost_job_thresholds(
    seniority_final, yoe_extracted, yoe_contradiction, expected
):
    assert stage678.detect_ghost_job(seniority_final, yoe_extracted, yoe_contradiction) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("row_source", "scrape_date", "expected"),
    [
        ("scraped", "2026-03-20", "2026-03"),
        ("kaggle_arshkon", None, "2024-04"),
        ("kaggle_asaniczka", "2026-03-20", "2024-01"),
        ("unknown", "2026-03-20", "unknown"),
    ],
)
def test_derive_period_maps_scraped_and_historical_sources(
    row_source, scrape_date, expected
):
    assert stage678.derive_period(row_source, scrape_date) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fixture_name", "expected"),
    [
        (
            "synthetic",
            {
                "synthetic_search_metro": {
                    "city_extracted": "Seattle",
                    "state_normalized": "WA",
                    "country_extracted": "US",
                    "is_remote_inferred": False,
                    "metro_area": "Seattle Metro",
                    "metro_source": "search_metro",
                    "metro_confidence": "high",
                    "period": "2026-03",
                    "posting_age_days": pytest.approx(1.0),
                    "scrape_week": pytest.approx(12.0),
                    "date_flag": "ok",
                    "description_quality_flag": "ok",
                    "ghost_job_risk": "low",
                },
                "synthetic_remote_future": {
                    "is_remote_inferred": True,
                    "metro_area": None,
                    "metro_source": "unresolved",
                    "metro_confidence": "low",
                    "period": "2026-03",
                    "date_flag": "ok",
                    "description_quality_flag": "too_short",
                    "ghost_job_risk": "high",
                },
                "synthetic_reference": {
                    "city_extracted": "Fort Myers",
                    "state_normalized": "FL",
                    "country_extracted": "US",
                    "is_remote_inferred": False,
                    "metro_area": "Cape Coral-Fort Myers Metro",
                    "metro_source": "city_state_reference",
                    "metro_confidence": "medium",
                    "period": "2024-04",
                    "date_flag": "ok",
                    "description_quality_flag": "too_short",
                    "ghost_job_risk": "low",
                },
            },
        ),
        (
            "sampled",
            {
                "linkedin_li-3696371940": {
                    "metro_area": "New York City Metro",
                    "metro_source": "search_metro",
                    "metro_confidence": "high",
                    "period": "2026-03",
                    "posting_age_days": MISSING,
                    "date_flag": "ok",
                    "description_quality_flag": "ok",
                    "ghost_job_risk": "low",
                },
                "indeed_in-00092430f5fed366": {
                    "metro_area": "New York City Metro",
                    "metro_source": "search_metro",
                    "metro_confidence": "high",
                    "period": "2026-03",
                    "posting_age_days": pytest.approx(2.0),
                    "date_flag": "ok",
                    "description_quality_flag": "ok",
                    "ghost_job_risk": "low",
                },
                "arshkon_3190494363": {
                    "city_extracted": "Fort Myers",
                    "state_normalized": "FL",
                    "country_extracted": "US",
                    "is_remote_inferred": False,
                    "metro_area": "Cape Coral-Fort Myers Metro",
                    "metro_source": "city_state_reference",
                    "metro_confidence": "medium",
                    "period": "2024-04",
                    "date_flag": "ok",
                    "description_quality_flag": "too_short",
                    "ghost_job_risk": "low",
                },
                "arshkon_3884427957": {
                    "city_extracted": "Lansdale",
                    "state_normalized": "PA",
                    "country_extracted": "US",
                    "is_remote_inferred": False,
                    "metro_area": None,
                    "metro_source": "unresolved",
                    "metro_confidence": "low",
                    "period": "2024-04",
                    "date_flag": "ok",
                    "description_quality_flag": "ok",
                    "ghost_job_risk": "high",
                },
            },
        ),
    ],
)
def test_process_chunk_preserves_rows_and_enrichment(fixture_name, expected):
    df = load_df(fixture_name, "stage678", "rows.json")
    out = stage678.process_chunk(
        df.copy(),
        metro_aliases={},
        city_state_lookup={},
        city_state_reference_lookup={"fort myers|FL": ("Cape Coral-Fort Myers Metro", "medium")},
    )

    assert len(out) == len(df)
    assert out["uid"].tolist() == df["uid"].tolist()

    out_by_uid = out.set_index("uid")
    for uid, expected_subset in expected.items():
        assert_row_subset(out_by_uid.loc[uid], expected_subset)

    if fixture_name == "synthetic":
        assert out_by_uid.loc["synthetic_remote_future", "posting_age_days"] < 0
        assert bool(out_by_uid.loc["synthetic_search_metro", "is_english"]) is True
        assert out_by_uid.loc["synthetic_reference", "description_quality_flag"] == "too_short"
    else:
        assert out_by_uid.loc["indeed_in-00092430f5fed366", "posting_age_days"] == pytest.approx(2.0)
        assert pd.isna(out_by_uid.loc["linkedin_li-3696371940", "posting_age_days"])
        assert out_by_uid.loc["arshkon_3884427957", "ghost_job_risk"] == "high"
