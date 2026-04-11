"""Unit tests for the LLM budget utilities used by Stages 9 and 10."""

import pytest

from tests.helpers.imports import load_module


llm_shared = load_module("llm_shared_budget", "preprocessing/scripts/llm_shared.py")


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
    # SWE is capped at 800; surplus should cascade to other categories.
    result = llm_shared.split_budget_by_category(
        budget=10000,
        uncached_per_category={"swe": 800, "swe_adjacent": 5000, "control": 20000},
        shares={"swe": 0.4, "swe_adjacent": 0.3, "control": 0.3},
    )
    assert result["swe"] == 800
    assert sum(result.values()) == 10000
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
    # Both swe and swe_adjacent are capped small; all surplus flows to control.
    result = llm_shared.split_budget_by_category(
        budget=1000,
        uncached_per_category={"swe": 50, "swe_adjacent": 50, "control": 10000},
        shares={"swe": 0.4, "swe_adjacent": 0.3, "control": 0.3},
    )
    assert result["swe"] == 50
    assert result["swe_adjacent"] == 50
    assert result["control"] == 900
    assert sum(result.values()) == 1000


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


# --- Stage 9/10 fresh-call budget flow -------------------------------------


def _make_frame_candidate(
    hash_id,
    date_bin,
    category,
    *,
    source="scraped",
    hash_key="extraction_input_hash",
):
    row = {
        hash_key: hash_id,
        "source": source,
        "is_swe": category == "swe",
        "is_swe_adjacent": category == "swe_adjacent",
        "is_control": category == "control",
        "selected_for_control_cohort": category == "control",
    }
    if source == "scraped":
        row["scrape_date"] = date_bin
        row["date_posted"] = None
    else:
        row["scrape_date"] = None
        row["date_posted"] = date_bin
    return row


def _select_fresh_calls_like_stages(
    candidates,
    *,
    cached_hashes=None,
    budget=100,
    split=None,
    hash_key="extraction_input_hash",
):
    cached_hashes = set() if cached_hashes is None else set(cached_hashes)
    split = {"swe": 0.4, "swe_adjacent": 0.3, "control": 0.3} if split is None else split
    fresh_candidates = [
        row for row in candidates if str(row.get(hash_key)) not in cached_hashes
    ]
    uncached_per_category = {
        analysis_group: sum(
            1
            for row in fresh_candidates
            if llm_shared.derive_analysis_group(row) == analysis_group
        )
        for analysis_group in llm_shared.ANALYSIS_GROUP_PRIORITY
    }
    category_targets = llm_shared.split_budget_by_category(
        budget, uncached_per_category, split
    )

    rows_to_process = []
    for analysis_group in llm_shared.ANALYSIS_GROUP_PRIORITY:
        group_rows = [
            row
            for row in fresh_candidates
            if llm_shared.derive_analysis_group(row) == analysis_group
        ]
        if not group_rows:
            continue
        selected_group_rows, _ = llm_shared.select_fresh_call_tasks(
            group_rows,
            llm_budget=category_targets.get(analysis_group, 0),
            hash_key=hash_key,
            groups=(analysis_group,),
        )
        rows_to_process.extend(selected_group_rows)
    return rows_to_process, category_targets, uncached_per_category


def _count_by(rows, key):
    counts = {}
    for row in rows:
        counts[row[key]] = counts.get(row[key], 0) + 1
    return counts


@pytest.mark.unit
def test_runtime_fresh_budget_applies_category_split_after_cache_filtering():
    candidates = (
        [_make_frame_candidate(f"swe-{i}", "d1", "swe") for i in range(100)]
        + [_make_frame_candidate(f"adj-{i}", "d1", "swe_adjacent") for i in range(100)]
        + [_make_frame_candidate(f"ctrl-{i}", "d1", "control") for i in range(100)]
    )
    selected, targets, uncached = _select_fresh_calls_like_stages(candidates, budget=100)

    assert targets == {"swe": 40, "swe_adjacent": 30, "control": 30}
    assert uncached == {"swe": 100, "swe_adjacent": 100, "control": 100}
    assert _count_by(selected, "analysis_group") == {
        "swe": 40,
        "swe_adjacent": 30,
        "control": 30,
    }


@pytest.mark.unit
def test_runtime_fresh_budget_counts_only_uncached_selected_rows():
    candidates = [
        _make_frame_candidate(f"swe-{i}", "d1", "swe") for i in range(10)
    ]
    selected, targets, uncached = _select_fresh_calls_like_stages(
        candidates,
        cached_hashes={"swe-0", "swe-1", "swe-2"},
        budget=100,
        split={"swe": 1.0, "swe_adjacent": 0.0, "control": 0.0},
    )

    assert targets == {"swe": 7, "swe_adjacent": 0, "control": 0}
    assert uncached == {"swe": 7, "swe_adjacent": 0, "control": 0}
    assert len(selected) == 7
    assert {row["extraction_input_hash"] for row in selected}.isdisjoint(
        {"swe-0", "swe-1", "swe-2"}
    )


@pytest.mark.unit
def test_runtime_fresh_selection_uses_frame_balancing_within_group():
    candidates = (
        [_make_frame_candidate(f"scraped-a-{i}", "d1", "swe") for i in range(6)]
        + [_make_frame_candidate(f"scraped-b-{i}", "d2", "swe") for i in range(6)]
        + [
            _make_frame_candidate(f"hist-{i}", "h1", "swe", source="kaggle_arshkon")
            for i in range(12)
        ]
    )
    selected, targets, _ = _select_fresh_calls_like_stages(
        candidates,
        budget=4,
        split={"swe": 1.0, "swe_adjacent": 0.0, "control": 0.0},
    )

    assert targets == {"swe": 4, "swe_adjacent": 0, "control": 0}
    assert _count_by(selected, "source") == {"kaggle_arshkon": 2, "scraped": 2}
    assert _count_by([row for row in selected if row["source"] == "scraped"], "date_bin") == {
        "d1": 1,
        "d2": 1,
    }


@pytest.mark.unit
def test_runtime_fresh_selection_accepts_classification_hash_key():
    candidates = [
        _make_frame_candidate(
            f"class-{i}",
            "d1",
            "swe",
            hash_key="classification_input_hash",
        )
        for i in range(5)
    ]
    selected, targets, _ = _select_fresh_calls_like_stages(
        candidates,
        budget=3,
        split={"swe": 1.0, "swe_adjacent": 0.0, "control": 0.0},
        hash_key="classification_input_hash",
    )

    assert targets == {"swe": 3, "swe_adjacent": 0, "control": 0}
    assert len(selected) == 3
    assert all("classification_input_hash" in row for row in selected)
