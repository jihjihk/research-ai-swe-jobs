import pytest

from tests.helpers.imports import load_module


llm_shared = load_module("llm_shared", "preprocessing/scripts/llm_shared.py")


def _candidate(
    uid: str,
    *,
    input_hash: str,
    source: str,
    date_posted: str | None = None,
    scrape_date: str | None = None,
    is_swe: bool = False,
    is_swe_adjacent: bool = False,
    is_control: bool = False,
    extra: dict | None = None,
):
    row = {
        "uid": uid,
        "source": source,
        "date_posted": date_posted,
        "scrape_date": scrape_date,
        "is_swe": is_swe,
        "is_swe_adjacent": is_swe_adjacent,
        "is_control": is_control,
        "task_hash": input_hash,
    }
    if extra:
        row.update(extra)
    return row


@pytest.mark.unit
def test_derive_analysis_group_and_date_bin_use_existing_flags_and_source_dates():
    assert llm_shared.derive_analysis_group({"is_swe": True}) == "swe_combined"
    assert llm_shared.derive_analysis_group({"is_swe_adjacent": True}) == "swe_combined"
    assert llm_shared.derive_analysis_group({"is_control": True}) == "control"
    assert llm_shared.derive_analysis_group({}) is None

    assert (
        llm_shared.derive_date_bin({"source": "scraped", "scrape_date": "2026-03-21", "date_posted": "2026-03-01"})
        == "2026-03-21"
    )
    assert (
        llm_shared.derive_date_bin(
            {"source": "kaggle_arshkon", "scrape_date": "2026-03-21", "date_posted": "2024-04-10"}
        )
        == "2024-04-10"
    )
    assert llm_shared.derive_date_bin({"source": "kaggle_asaniczka", "date_posted": None}) == "unknown"


@pytest.mark.unit
def test_select_task_frame_is_deterministic_and_larger_targets_are_supersets():
    candidates = [
        _candidate(
            f"row-{idx}",
            input_hash=f"hash-{idx}",
            source="scraped" if idx % 2 else "kaggle_arshkon",
            scrape_date=f"2026-03-{20 + (idx % 3):02d}",
            date_posted=f"2024-04-{5 + (idx % 3):02d}",
            is_swe=True,
        )
        for idx in range(12)
    ]

    selected_small, _ = llm_shared.select_task_frame(candidates, selection_target=4, hash_key="task_hash")
    rerun_small, _ = llm_shared.select_task_frame(candidates, selection_target=4, hash_key="task_hash")
    selected_large, _ = llm_shared.select_task_frame(candidates, selection_target=8, hash_key="task_hash")

    small_hashes = {row["task_hash"] for row in selected_small}
    rerun_hashes = {row["task_hash"] for row in rerun_small}
    large_hashes = {row["task_hash"] for row in selected_large}

    assert small_hashes == rerun_hashes
    assert small_hashes < large_hashes


@pytest.mark.unit
def test_select_task_frame_ignores_cache_like_fields_when_building_the_frame():
    base_candidates = [
        _candidate(
            f"base-{idx}",
            input_hash=f"hash-{idx}",
            source="scraped",
            scrape_date=f"2026-03-{20 + idx:02d}",
            is_swe=True,
        )
        for idx in range(6)
    ]
    decorated_candidates = [
        dict(row, cached_hit=(idx % 2 == 0), already_resolved=(idx % 3 == 0))
        for idx, row in enumerate(base_candidates)
    ]

    base_selected, _ = llm_shared.select_task_frame(base_candidates, selection_target=3, hash_key="task_hash")
    decorated_selected, _ = llm_shared.select_task_frame(decorated_candidates, selection_target=3, hash_key="task_hash")

    assert {row["task_hash"] for row in base_selected} == {row["task_hash"] for row in decorated_selected}


@pytest.mark.unit
def test_select_task_frame_per_group_targets_yield_exact_group_counts():
    """selection_targets={swe_combined: N1, control: N2} returns exactly N1+N2 rows
    with N1 from swe_combined and N2 from control, balanced across sources."""
    candidates = []
    # 200 swe_combined candidates across 2 sources / 2 dates
    for source in ("scraped", "kaggle_arshkon"):
        for day in ("2026-03-20", "2026-03-21"):
            for idx in range(50):
                candidates.append(
                    _candidate(
                        f"swe-{source}-{day}-{idx}",
                        input_hash=f"swe-{source}-{day}-{idx}",
                        source=source,
                        scrape_date=day if source == "scraped" else None,
                        date_posted=None if source == "scraped" else day,
                        is_swe=True,
                    )
                )
    # 200 control candidates similarly distributed
    for source in ("scraped", "kaggle_arshkon"):
        for day in ("2026-03-20", "2026-03-21"):
            for idx in range(50):
                candidates.append(
                    _candidate(
                        f"ctrl-{source}-{day}-{idx}",
                        input_hash=f"ctrl-{source}-{day}-{idx}",
                        source=source,
                        scrape_date=day if source == "scraped" else None,
                        date_posted=None if source == "scraped" else day,
                        is_control=True,
                    )
                )

    selected, summary = llm_shared.select_task_frame(
        candidates,
        selection_targets={"swe_combined": 60, "control": 20},
        hash_key="task_hash",
    )
    by_group: dict[str, int] = {}
    for row in selected:
        by_group[row["analysis_group"]] = by_group.get(row["analysis_group"], 0) + 1
    assert by_group == {"swe_combined": 60, "control": 20}
    assert summary["selected_count"] == 80
    assert summary["selection_target"] == 80


@pytest.mark.unit
def test_select_sticky_task_frame_per_group_targets_retain_existing_and_top_up_only_one_group(tmp_path):
    """Sticky retention with per-group targets adds new rows only on the group
    whose target exceeds its retained count."""
    manifest_path = tmp_path / "manifest.json"
    candidates = []
    # 50 swe_combined and 50 control candidates
    for idx in range(50):
        candidates.append(
            _candidate(f"swe-{idx}", input_hash=f"swe-{idx}", source="scraped",
                       scrape_date="2026-03-20", is_swe=True)
        )
        candidates.append(
            _candidate(f"ctrl-{idx}", input_hash=f"ctrl-{idx}", source="scraped",
                       scrape_date="2026-03-20", is_control=True)
        )

    # Step 1: build initial manifest with even split, target=20
    initial, _ = llm_shared.select_sticky_task_frame(
        candidates, selection_target=20, hash_key="task_hash",
        manifest_path=manifest_path,
    )
    initial_swe = [r for r in initial if r["analysis_group"] == "swe_combined"]
    initial_ctrl = [r for r in initial if r["analysis_group"] == "control"]
    assert len(initial_swe) == 10 and len(initial_ctrl) == 10
    llm_shared.write_core_frame_manifest(
        manifest_path,
        selected_hashes=[row["task_hash"] for row in initial],
        hash_key="task_hash",
    )

    # Step 2: bump swe_combined target to 30, keep control at 10.
    # Expect: 10 retained swe + 10 retained ctrl + 20 new swe top-ups = 40 total.
    next_selected, summary = llm_shared.select_sticky_task_frame(
        candidates,
        selection_targets={"swe_combined": 30, "control": 10},
        hash_key="task_hash",
        manifest_path=manifest_path,
    )
    by_group: dict[str, int] = {}
    for row in next_selected:
        by_group[row["analysis_group"]] = by_group.get(row["analysis_group"], 0) + 1
    assert by_group == {"swe_combined": 30, "control": 10}
    assert summary["retained_count"] == 20  # all initial rows kept
    assert summary["top_up_count"] == 20    # only swe topped up
    # The original 10 control rows are still there.
    initial_ctrl_hashes = {r["task_hash"] for r in initial_ctrl}
    assert initial_ctrl_hashes <= {r["task_hash"] for r in next_selected}


@pytest.mark.unit
def test_select_task_frame_balances_dates_within_source_group_and_spills_to_available_capacity():
    candidates = []
    for idx, day in enumerate(("2026-03-20", "2026-03-21", "2026-03-22")):
        for copy_idx in range(2):
            candidates.append(
                _candidate(
                    f"scraped-swe-{idx}-{copy_idx}",
                    input_hash=f"scraped-swe-{idx}-{copy_idx}",
                    source="scraped",
                    scrape_date=day,
                    is_swe=True,
                )
            )
    for idx in range(3):
        candidates.append(
            _candidate(
                f"arshkon-control-{idx}",
                input_hash=f"arshkon-control-{idx}",
                source="kaggle_arshkon",
                date_posted="2024-04-10",
                is_control=True,
            )
        )

    selected, summary = llm_shared.select_task_frame(candidates, selection_target=7, hash_key="task_hash")

    assert len(selected) == 7
    assert summary["selected_count"] == 7

    scraped_swe_counts = {}
    control_count = 0
    for row in selected:
        if row["source"] == "scraped" and row["analysis_group"] == "swe_combined":
            scraped_swe_counts[row["date_bin"]] = scraped_swe_counts.get(row["date_bin"], 0) + 1
        if row["analysis_group"] == "control":
            control_count += 1

    assert max(scraped_swe_counts.values()) - min(scraped_swe_counts.values()) <= 1
    assert control_count >= 1


@pytest.mark.unit
def test_select_fresh_call_tasks_reuses_same_selection_logic():
    unresolved_rows = [
        _candidate(
            f"row-{idx}",
            input_hash=f"hash-{idx}",
            source="scraped" if idx < 4 else "kaggle_asaniczka",
            scrape_date=f"2026-03-{20 + (idx % 2):02d}",
            date_posted=f"2024-01-{12 + (idx % 2):02d}",
            is_swe=(idx % 2 == 0),
            is_control=(idx % 2 == 1),
        )
        for idx in range(8)
    ]

    selected, summary = llm_shared.select_fresh_call_tasks(
        unresolved_rows,
        llm_budget=5,
        hash_key="task_hash",
    )

    assert len(selected) == 5
    assert summary["selection_target"] == 5
    assert summary["selected_count"] == 5


@pytest.mark.unit
def test_select_sticky_task_frame_carries_forward_manifest_and_tops_up(tmp_path):
    manifest_path = tmp_path / "stage9_core_frame_manifest.json"
    initial_candidates = [
        _candidate(
            f"row-{idx}",
            input_hash=f"hash-{idx}",
            source="scraped",
            scrape_date=f"2026-03-{20 + (idx % 2):02d}",
            is_swe=True,
        )
        for idx in range(5)
    ]

    initial_selected, _ = llm_shared.select_sticky_task_frame(
        initial_candidates,
        selection_target=3,
        hash_key="task_hash",
        manifest_path=manifest_path,
    )
    initial_hashes = [row["task_hash"] for row in initial_selected]
    llm_shared.write_core_frame_manifest(manifest_path, selected_hashes=initial_hashes, hash_key="task_hash")

    rerun_candidates = [
        row for row in initial_candidates
        if row["task_hash"] != initial_hashes[0]
    ] + [
        _candidate(
            "row-5",
            input_hash="hash-5",
            source="scraped",
            scrape_date="2026-03-22",
            is_swe=True,
        )
    ]
    rerun_selected, rerun_summary = llm_shared.select_sticky_task_frame(
        rerun_candidates,
        selection_target=3,
        hash_key="task_hash",
        manifest_path=manifest_path,
    )

    rerun_hashes = [row["task_hash"] for row in rerun_selected]
    assert rerun_summary["retained_count"] == 2
    assert rerun_summary["top_up_count"] == 1
    assert set(initial_hashes[1:]) <= set(rerun_hashes)
    assert llm_shared.load_core_frame_manifest(manifest_path)["selected_hashes"] == initial_hashes


@pytest.mark.unit
def test_select_sticky_task_frame_does_not_implicitly_shrink(tmp_path):
    manifest_path = tmp_path / "stage9_core_frame_manifest.json"
    candidates = [
        _candidate(
            f"row-{idx}",
            input_hash=f"hash-{idx}",
            source="scraped",
            scrape_date=f"2026-03-{20 + idx:02d}",
            is_swe=True,
        )
        for idx in range(6)
    ]

    selected_large, _ = llm_shared.select_sticky_task_frame(
        candidates,
        selection_target=5,
        hash_key="task_hash",
        manifest_path=manifest_path,
    )
    llm_shared.write_core_frame_manifest(
        manifest_path,
        selected_hashes=[row["task_hash"] for row in selected_large],
        hash_key="task_hash",
    )
    selected_small, summary = llm_shared.select_sticky_task_frame(
        candidates,
        selection_target=2,
        hash_key="task_hash",
        manifest_path=manifest_path,
    )

    assert len(selected_large) == 5
    assert [row["task_hash"] for row in selected_small] == [row["task_hash"] for row in selected_large]
    assert summary["selected_count"] == 5
    assert summary["retained_count"] == 5
    assert summary["top_up_count"] == 0


@pytest.mark.unit
def test_select_sticky_task_frame_reset_rebuilds_smaller_core(tmp_path):
    manifest_path = tmp_path / "stage9_core_frame_manifest.json"
    candidates = [
        _candidate(
            f"row-{idx}",
            input_hash=f"hash-{idx}",
            source="scraped",
            scrape_date=f"2026-03-{20 + idx:02d}",
            is_swe=True,
        )
        for idx in range(6)
    ]

    large_selected, _ = llm_shared.select_sticky_task_frame(
        candidates,
        selection_target=5,
        hash_key="task_hash",
        manifest_path=manifest_path,
    )
    llm_shared.write_core_frame_manifest(
        manifest_path,
        selected_hashes=[row["task_hash"] for row in large_selected],
        hash_key="task_hash",
    )
    reset_selected, reset_summary = llm_shared.select_sticky_task_frame(
        candidates,
        selection_target=2,
        hash_key="task_hash",
        manifest_path=manifest_path,
        reset=True,
    )
    stateless_selected, _ = llm_shared.select_task_frame(
        candidates,
        selection_target=2,
        hash_key="task_hash",
    )

    assert reset_summary["reset"] is True
    assert [row["task_hash"] for row in reset_selected] == [row["task_hash"] for row in stateless_selected]
    assert llm_shared.load_core_frame_manifest(manifest_path)["selected_hashes"] == [
        row["task_hash"] for row in large_selected
    ]
