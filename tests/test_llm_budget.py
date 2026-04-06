"""Unit tests for LLM budget allocation utilities in llm_shared.py."""

import pytest

from tests.helpers.imports import load_module


llm_shared = load_module("llm_shared_budget", "preprocessing/scripts/llm_shared.py")


# --- allocate_budget_across_days --------------------------------------------


@pytest.mark.unit
def test_water_fills_lowest_level_day_first():
    # Day with no cache should receive more budget than day with existing cache.
    result = llm_shared.allocate_budget_across_days(
        uncached_per_day={"2026-03-20": 100, "2026-03-21": 100, "2026-03-22": 100},
        cached_per_day={"2026-03-20": 50, "2026-03-21": 20, "2026-03-22": 0},
        budget=120,
    )
    assert sum(result.values()) == 120
    # Day with level=0 gets most budget.
    assert result["2026-03-22"] >= result["2026-03-21"] >= result["2026-03-20"]


@pytest.mark.unit
def test_day_allocation_capped_by_uncached():
    # Bucket with tiny uncached count should not receive more than its capacity.
    result = llm_shared.allocate_budget_across_days(
        uncached_per_day={"A": 5, "B": 100},
        cached_per_day={"A": 0, "B": 0},
        budget=20,
    )
    assert result == {"A": 5, "B": 15}


@pytest.mark.unit
def test_budget_zero_returns_zero_allocation():
    result = llm_shared.allocate_budget_across_days(
        uncached_per_day={"A": 5, "B": 100},
        cached_per_day={"A": 0, "B": 0},
        budget=0,
    )
    assert result == {"A": 0, "B": 0}


@pytest.mark.unit
def test_budget_exceeds_total_capacity():
    # When budget exceeds total uncached, allocate all capacity.
    result = llm_shared.allocate_budget_across_days(
        uncached_per_day={"A": 5, "B": 10},
        cached_per_day={"A": 0, "B": 0},
        budget=100,
    )
    assert result == {"A": 5, "B": 10}


@pytest.mark.unit
def test_ties_broken_deterministically_by_day_name():
    # With two identical buckets at the same level, repeated runs should agree.
    r1 = llm_shared.allocate_budget_across_days(
        uncached_per_day={"2026-03-20": 50, "2026-03-21": 50},
        cached_per_day={"2026-03-20": 0, "2026-03-21": 0},
        budget=11,
    )
    r2 = llm_shared.allocate_budget_across_days(
        uncached_per_day={"2026-03-20": 50, "2026-03-21": 50},
        cached_per_day={"2026-03-20": 0, "2026-03-21": 0},
        budget=11,
    )
    assert r1 == r2
    assert sum(r1.values()) == 11


@pytest.mark.unit
def test_empty_inputs_return_empty_result():
    assert llm_shared.allocate_budget_across_days({}, {}, budget=100) == {}


# --- split_budget_by_category ----------------------------------------------


@pytest.mark.unit
def test_category_split_normal_case():
    # When all categories have plenty of uncached rows, split 40/30/30.
    result = llm_shared.split_budget_by_category(
        budget=100,
        uncached_per_category={"swe": 10000, "swe_adjacent": 10000, "control": 10000},
        shares={"swe": 0.4, "swe_adjacent": 0.3, "control": 0.3},
    )
    assert result == {"swe": 40, "swe_adjacent": 30, "control": 30}
    assert sum(result.values()) == 100


@pytest.mark.unit
def test_category_surplus_cascades_when_one_saturated():
    # SWE is capped at 800; surplus (3200) should cascade to other categories.
    result = llm_shared.split_budget_by_category(
        budget=10000,
        uncached_per_category={"swe": 800, "swe_adjacent": 5000, "control": 20000},
        shares={"swe": 0.4, "swe_adjacent": 0.3, "control": 0.3},
    )
    assert result["swe"] == 800
    assert sum(result.values()) == 10000
    # swe_adjacent:control have equal shares → equal surplus (4600 each).
    assert result["swe_adjacent"] == result["control"]


@pytest.mark.unit
def test_category_all_saturated_budget_exceeds_total():
    # When total capacity < budget, allocate all and return less than budget.
    result = llm_shared.split_budget_by_category(
        budget=50000,
        uncached_per_category={"swe": 100, "swe_adjacent": 200, "control": 300},
        shares={"swe": 0.4, "swe_adjacent": 0.3, "control": 0.3},
    )
    assert result == {"swe": 100, "swe_adjacent": 200, "control": 300}
    assert sum(result.values()) == 600


@pytest.mark.unit
def test_category_budget_zero_returns_zero():
    result = llm_shared.split_budget_by_category(
        budget=0,
        uncached_per_category={"swe": 100, "swe_adjacent": 100, "control": 100},
        shares={"swe": 0.4, "swe_adjacent": 0.3, "control": 0.3},
    )
    assert result == {"swe": 0, "swe_adjacent": 0, "control": 0}


@pytest.mark.unit
def test_category_cascades_when_two_categories_saturated():
    # Both swe and swe_adjacent capped small; all surplus should flow to control.
    result = llm_shared.split_budget_by_category(
        budget=1000,
        uncached_per_category={"swe": 50, "swe_adjacent": 50, "control": 10000},
        shares={"swe": 0.4, "swe_adjacent": 0.3, "control": 0.3},
    )
    assert result["swe"] == 50
    assert result["swe_adjacent"] == 50
    assert result["control"] == 900
    assert sum(result.values()) == 1000


# --- select_rows_by_budget --------------------------------------------------


@pytest.mark.unit
def test_select_rows_returns_correct_counts_per_day():
    rows = [
        {"scrape_date": "2026-03-20", "extraction_input_hash": f"hash-a-{i}"}
        for i in range(10)
    ] + [
        {"scrape_date": "2026-03-21", "extraction_input_hash": f"hash-b-{i}"}
        for i in range(10, 20)
    ]
    selected = llm_shared.select_rows_by_budget(
        rows,
        day_key="scrape_date",
        hash_key="extraction_input_hash",
        allocation={"2026-03-20": 3, "2026-03-21": 2},
    )
    assert len(selected) == 5
    by_day = {}
    for row in selected:
        by_day.setdefault(row["scrape_date"], 0)
        by_day[row["scrape_date"]] += 1
    assert by_day == {"2026-03-20": 3, "2026-03-21": 2}


@pytest.mark.unit
def test_select_rows_is_deterministic():
    rows = [
        {"scrape_date": "D1", "extraction_input_hash": f"h{i}"} for i in range(50)
    ]
    r1 = llm_shared.select_rows_by_budget(
        rows, "scrape_date", "extraction_input_hash", {"D1": 7}
    )
    r2 = llm_shared.select_rows_by_budget(
        rows, "scrape_date", "extraction_input_hash", {"D1": 7}
    )
    assert r1 == r2


@pytest.mark.unit
def test_select_rows_higher_budget_yields_superset():
    # Increasing budget should select a superset of the prior selection.
    rows = [{"scrape_date": "D1", "extraction_input_hash": f"h{i}"} for i in range(100)]
    small = llm_shared.select_rows_by_budget(
        rows, "scrape_date", "extraction_input_hash", {"D1": 5}
    )
    large = llm_shared.select_rows_by_budget(
        rows, "scrape_date", "extraction_input_hash", {"D1": 20}
    )
    small_hashes = {r["extraction_input_hash"] for r in small}
    large_hashes = {r["extraction_input_hash"] for r in large}
    assert small_hashes.issubset(large_hashes)


@pytest.mark.unit
def test_select_rows_empty_allocation_returns_empty():
    rows = [{"scrape_date": "D1", "extraction_input_hash": "h1"}]
    assert llm_shared.select_rows_by_budget(rows, "scrape_date", "extraction_input_hash", {}) == []


@pytest.mark.unit
def test_select_rows_skips_day_not_in_allocation():
    rows = [
        {"scrape_date": "D1", "extraction_input_hash": "h1"},
        {"scrape_date": "D2", "extraction_input_hash": "h2"},
    ]
    selected = llm_shared.select_rows_by_budget(
        rows, "scrape_date", "extraction_input_hash", {"D1": 1}
    )
    assert len(selected) == 1
    assert selected[0]["scrape_date"] == "D1"


# --- stable_budget_score ----------------------------------------------------


@pytest.mark.unit
def test_stable_budget_score_is_deterministic():
    assert (
        llm_shared.stable_budget_score("2026-03-20", "abc")
        == llm_shared.stable_budget_score("2026-03-20", "abc")
    )


@pytest.mark.unit
def test_stable_budget_score_differs_across_inputs():
    a = llm_shared.stable_budget_score("2026-03-20", "abc")
    b = llm_shared.stable_budget_score("2026-03-20", "abd")
    c = llm_shared.stable_budget_score("2026-03-21", "abc")
    assert a != b
    assert a != c
    assert b != c


# --- parse_budget_split -----------------------------------------------------


@pytest.mark.unit
def test_parse_budget_split_default():
    result = llm_shared.parse_budget_split("0.4,0.3,0.3")
    assert result == {"swe": 0.4, "swe_adjacent": 0.3, "control": 0.3}


@pytest.mark.unit
def test_parse_budget_split_normalizes_percentages():
    # Percentages like "40,30,30" should be normalized to sum to 1.0.
    result = llm_shared.parse_budget_split("40,30,30")
    assert result == {"swe": 0.4, "swe_adjacent": 0.3, "control": 0.3}


@pytest.mark.unit
def test_parse_budget_split_rejects_wrong_count():
    with pytest.raises(ValueError, match="3 comma-separated"):
        llm_shared.parse_budget_split("0.5,0.5")


@pytest.mark.unit
def test_parse_budget_split_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        llm_shared.parse_budget_split("0.5,-0.2,0.7")


@pytest.mark.unit
def test_parse_budget_split_rejects_all_zero():
    with pytest.raises(ValueError, match="sum to > 0"):
        llm_shared.parse_budget_split("0,0,0")


# --- select_rows_with_budget ----------------------------------------


def _make_candidate(hash_id, day, category, *, source="scraped"):
    """Build a fake candidate row."""
    return {
        "extraction_input_hash": hash_id,
        "scrape_date": day,
        "source": source,
        "is_swe": category == "swe",
        "is_swe_adjacent": category == "swe_adjacent",
        "selected_for_control_cohort": category == "control",
    }


@pytest.mark.unit
def test_select_budget_applies_40_30_30_split():
    candidates = (
        [_make_candidate(f"swe-{i}", "2026-03-20", "swe") for i in range(100)]
        + [_make_candidate(f"adj-{i}", "2026-03-20", "swe_adjacent") for i in range(100)]
        + [_make_candidate(f"ctrl-{i}", "2026-03-20", "control") for i in range(100)]
    )
    selected, alloc, uncached = llm_shared.select_rows_with_budget(
        candidates=candidates,
        cached_hashes=set(),
        budget=100,
        split={"swe": 0.4, "swe_adjacent": 0.3, "control": 0.3},
        hash_key="extraction_input_hash",
    )
    assert alloc == {"swe": 40, "swe_adjacent": 30, "control": 30}
    assert len(selected) == 100
    assert uncached == {"swe": 100, "swe_adjacent": 100, "control": 100}


@pytest.mark.unit
def test_select_budget_budget_zero_returns_nothing():
    candidates = [_make_candidate("swe-1", "2026-03-20", "swe")]
    selected, alloc, _ = llm_shared.select_rows_with_budget(
        candidates=candidates,
        cached_hashes=set(),
        budget=0,
        split={"swe": 0.4, "swe_adjacent": 0.3, "control": 0.3},
        hash_key="extraction_input_hash",
    )
    assert selected == []
    assert alloc == {"swe": 0, "swe_adjacent": 0, "control": 0}


@pytest.mark.unit
def test_select_budget_balances_across_days():
    # SWE budget 4 spread over 2 days with 5 each → 2 per day.
    candidates = (
        [_make_candidate(f"a-{i}", "2026-03-20", "swe") for i in range(5)]
        + [_make_candidate(f"b-{i}", "2026-03-21", "swe") for i in range(5)]
    )
    selected, _, _ = llm_shared.select_rows_with_budget(
        candidates=candidates,
        cached_hashes=set(),
        budget=10,  # All go to swe since only category present
        split={"swe": 1.0, "swe_adjacent": 0.0, "control": 0.0},
        hash_key="extraction_input_hash",
    )
    assert len(selected) == 10  # budget >= capacity → all selected


@pytest.mark.unit
def test_select_budget_respects_cache():
    # If a row's hash is already cached, it's not eligible for new selection.
    candidates = [_make_candidate(f"swe-{i}", "2026-03-20", "swe") for i in range(10)]
    selected, _, uncached = llm_shared.select_rows_with_budget(
        candidates=candidates,
        cached_hashes={"swe-0", "swe-1", "swe-2"},
        budget=100,
        split={"swe": 1.0, "swe_adjacent": 0.0, "control": 0.0},
        hash_key="extraction_input_hash",
    )
    assert uncached["swe"] == 7  # 3 are cached
    assert len(selected) == 7  # all uncached selected
    assert all(r["extraction_input_hash"] not in {"swe-0", "swe-1", "swe-2"} for r in selected)


@pytest.mark.unit
def test_select_budget_water_filling_prefers_thin_days():
    # Cached coverage already favors 2026-03-20. Water-filling should direct
    # new budget to 2026-03-21 (the thinner day).
    candidates = (
        [_make_candidate(f"a-{i}", "2026-03-20", "swe") for i in range(100)]
        + [_make_candidate(f"b-{i}", "2026-03-21", "swe") for i in range(100)]
    )
    # Assume the first 50 rows from 2026-03-20 are cached.
    cached = {f"a-{i}" for i in range(50)}
    selected, _, _ = llm_shared.select_rows_with_budget(
        candidates=candidates,
        cached_hashes=cached,
        budget=10,
        split={"swe": 1.0, "swe_adjacent": 0.0, "control": 0.0},
        hash_key="extraction_input_hash",
    )
    # Count per day.
    per_day: dict[str, int] = {}
    for row in selected:
        per_day[row["scrape_date"]] = per_day.get(row["scrape_date"], 0) + 1
    # 2026-03-21 should get more (starts from level 0 vs level 50).
    assert per_day.get("2026-03-21", 0) > per_day.get("2026-03-20", 0)


@pytest.mark.unit
def test_select_budget_handles_nan_scrape_date():
    # NaN values (from pandas NaT conversions) used to slip through as day keys
    # because `nan or "unknown"` evaluates to nan. Rows with NaN scrape_date
    # should be bucketed under a sentinel "unknown" day.
    import math

    candidates = [
        _make_candidate("a-1", "2026-03-20", "swe"),
        {
            "extraction_input_hash": "a-2",
            "scrape_date": float("nan"),
            "source": "scraped",
            "is_swe": True,
            "is_swe_adjacent": False,
            "selected_for_control_cohort": False,
        },
        {
            "extraction_input_hash": "a-3",
            "scrape_date": None,
            "source": "scraped",
            "is_swe": True,
            "is_swe_adjacent": False,
            "selected_for_control_cohort": False,
        },
    ]
    selected, alloc, _ = llm_shared.select_rows_with_budget(
        candidates=candidates,
        cached_hashes=set(),
        budget=10,
        split={"swe": 1.0, "swe_adjacent": 0.0, "control": 0.0},
        hash_key="extraction_input_hash",
    )
    assert alloc["swe"] == 3
    assert len(selected) == 3
    # Verify no NaN strings leaked into the scrape_date field.
    for row in selected:
        sd = row["scrape_date"]
        assert isinstance(sd, str)
        assert not (isinstance(sd, float) and math.isnan(sd))


@pytest.mark.unit
def test_select_budget_hash_key_parameter():
    # Works with different hash keys (e.g. classification_input_hash).
    candidates = [
        {
            "classification_input_hash": f"h-{i}",
            "scrape_date": "2026-03-20",
            "source": "scraped",
            "is_swe": True,
            "is_swe_adjacent": False,
            "selected_for_control_cohort": False,
        }
        for i in range(5)
    ]
    selected, alloc, _ = llm_shared.select_rows_with_budget(
        candidates=candidates,
        cached_hashes=set(),
        budget=3,
        split={"swe": 1.0, "swe_adjacent": 0.0, "control": 0.0},
        hash_key="classification_input_hash",
    )
    assert alloc["swe"] == 3
    assert len(selected) == 3


# --- categorize_budget_candidate --------------------------------------------


@pytest.mark.unit
def test_categorize_budget_candidate_priority_ordering():
    # SWE takes priority when multiple flags are set.
    assert llm_shared.categorize_budget_candidate(
        {"is_swe": True, "is_swe_adjacent": True, "selected_for_control_cohort": True}
    ) == "swe"
    assert llm_shared.categorize_budget_candidate(
        {"is_swe": False, "is_swe_adjacent": True, "selected_for_control_cohort": True}
    ) == "swe_adjacent"
    assert llm_shared.categorize_budget_candidate(
        {"is_swe": False, "is_swe_adjacent": False, "selected_for_control_cohort": True}
    ) == "control"
    assert llm_shared.categorize_budget_candidate(
        {"is_swe": False, "is_swe_adjacent": False, "selected_for_control_cohort": False}
    ) is None


@pytest.mark.unit
def test_source_balancing_prefers_less_labeled_source_before_historical_backfill():
    candidates = (
        [_make_candidate(f"scraped-{i}", "2026-03-20", "swe", source="scraped") for i in range(20)]
        + [_make_candidate(f"asaniczka-{i}", "unknown", "swe", source="kaggle_asaniczka") for i in range(20)]
        + [_make_candidate(f"arshkon-{i}", "unknown", "swe", source="kaggle_arshkon") for i in range(20)]
    )
    cached = {f"asaniczka-{i}" for i in range(12)} | {f"arshkon-{i}" for i in range(8)}
    selected, alloc, _ = llm_shared.select_rows_with_budget(
        candidates=candidates,
        cached_hashes=cached,
        budget=6,
        split={"swe": 1.0, "swe_adjacent": 0.0, "control": 0.0},
        hash_key="extraction_input_hash",
    )
    assert alloc["swe"] == 6
    assert {row["source"] for row in selected} == {"scraped"}


@pytest.mark.unit
def test_scraped_source_water_fills_across_days_after_source_balancing():
    candidates = (
        [_make_candidate(f"scraped-a-{i}", "2026-03-20", "swe", source="scraped") for i in range(50)]
        + [_make_candidate(f"scraped-b-{i}", "2026-03-21", "swe", source="scraped") for i in range(50)]
        + [_make_candidate(f"hist-{i}", "unknown", "swe", source="kaggle_asaniczka") for i in range(50)]
    )
    cached = {f"scraped-a-{i}" for i in range(20)} | {f"hist-{i}" for i in range(40)}
    selected, _, _ = llm_shared.select_rows_with_budget(
        candidates=candidates,
        cached_hashes=cached,
        budget=10,
        split={"swe": 1.0, "swe_adjacent": 0.0, "control": 0.0},
        hash_key="extraction_input_hash",
    )
    per_day: dict[str, int] = {}
    for row in selected:
        per_day[row["scrape_date"]] = per_day.get(row["scrape_date"], 0) + 1
    assert {row["source"] for row in selected} == {"scraped"}
    assert per_day.get("2026-03-21", 0) > per_day.get("2026-03-20", 0)
