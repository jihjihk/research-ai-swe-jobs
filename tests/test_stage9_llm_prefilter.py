import json
import logging

import pandas as pd
import pytest

from tests.helpers.imports import load_module


stage9 = load_module("stage9_llm_prefilter", "preprocessing/scripts/stage9_llm_prefilter.py")
llm_shared = load_module("llm_shared", "preprocessing/scripts/llm_shared.py")


def _long_description(core_sentence: str, boilerplate_sentence: str) -> str:
    return f"{core_sentence}\n{boilerplate_sentence}"


@pytest.mark.unit
def test_annotate_stage9_chunk_derives_frame_fields_and_short_skip():
    df = pd.DataFrame(
        [
            {
                "uid": "u1",
                "job_id": "s9_r1",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Acme",
                "description": "Build APIs and maintain services for customer-facing systems with Python, SQL, testing, deployment ownership, and operational support.",
                "scrape_date": "2026-03-20",
                "date_posted": None,
                "is_english": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "is_control": False,
            },
            {
                "uid": "u2",
                "job_id": "s9_r2",
                "source": "kaggle_arshkon",
                "source_platform": "linkedin",
                "title": "Business Analyst",
                "company_name": "Beta",
                "description": "Analyze sales trends, create monthly reporting decks, coordinate leadership reviews, and support finance planning across business units.",
                "scrape_date": None,
                "date_posted": "2024-04-18",
                "is_english": True,
                "is_swe": False,
                "is_swe_adjacent": False,
                "is_control": True,
            },
            {
                "uid": "u3",
                "job_id": "s9_r3",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "QA Analyst",
                "company_name": "Gamma",
                "description": "Test features quickly.",
                "scrape_date": "2026-03-22",
                "date_posted": None,
                "is_english": True,
                "is_swe": False,
                "is_swe_adjacent": True,
                "is_control": False,
            },
        ]
    )

    out = stage9.annotate_stage9_chunk(df).set_index("job_id")

    assert out.loc["s9_r1", "analysis_group"] == "swe_combined"
    assert out.loc["s9_r1", "selection_date_bin"] == "2026-03-20"
    assert out.loc["s9_r1", "eligible_for_extraction"]
    assert out.loc["s9_r1", "llm_text_skip_reason"] is None

    assert out.loc["s9_r2", "analysis_group"] == "control"
    assert out.loc["s9_r2", "selection_date_bin"] == "2024-04-18"
    assert out.loc["s9_r2", "eligible_for_extraction"]

    assert out.loc["s9_r3", "raw_description_word_count"] == 3
    assert out.loc["s9_r3", "analysis_group"] == "swe_combined"
    assert out.loc["s9_r3", "llm_text_skip_reason"] == "short_description_under_15_words"
    assert not out.loc["s9_r3", "eligible_for_extraction"]
    assert out.loc["s9_r3", "description_core_llm"] == ""


@pytest.mark.unit
def test_build_extraction_candidates_collapses_duplicates_inside_selected_frame():
    df = pd.DataFrame(
        [
            {
                "uid": "tech-1",
                "job_id": "tech-1",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Acme",
                "description": "Build APIs and maintain services with Python, SQL, testing, CI, deployment ownership, operational support, and cross-team collaboration across environments.",
                "scrape_date": "2026-03-20",
                "date_posted": None,
                "is_english": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "is_control": False,
            },
            {
                "uid": "tech-2",
                "job_id": "tech-2",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Acme",
                "description": "Build APIs and maintain services with Python, SQL, testing, CI, deployment ownership, operational support, and cross-team collaboration across environments.",
                "scrape_date": "2026-03-21",
                "date_posted": None,
                "is_english": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "is_control": False,
            },
            {
                "uid": "ctrl-1",
                "job_id": "ctrl-1",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Financial Analyst",
                "company_name": "Beta",
                "description": "Analyze budgets, forecast revenue, build dashboards, coordinate stakeholders, prepare monthly operating reviews, and support annual planning cycles.",
                "scrape_date": "2026-03-20",
                "date_posted": None,
                "is_english": True,
                "is_swe": False,
                "is_swe_adjacent": False,
                "is_control": True,
            },
            {
                "uid": "short-1",
                "job_id": "short-1",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "QA Analyst",
                "company_name": "Gamma",
                "description": "Test features quickly.",
                "scrape_date": "2026-03-20",
                "date_posted": None,
                "is_english": True,
                "is_swe": False,
                "is_swe_adjacent": True,
                "is_control": False,
            },
        ]
    )

    annotated = stage9.annotate_stage9_chunk(df)
    selected_hashes = set(annotated.loc[annotated["job_id"].isin(["tech-1", "tech-2", "ctrl-1"]), "extraction_input_hash"])
    candidates = stage9.build_extraction_candidates(annotated, selected_hashes).sort_values("analysis_group")

    assert candidates["source_row_count"].sum() == 3
    assert candidates["extraction_input_hash"].nunique() == 2

    control = candidates[candidates["analysis_group"] == "control"].iloc[0]
    assert bool(control["selected_for_control_cohort"])
    assert control["source_row_count"] == 1

    swe = candidates[candidates["analysis_group"] == "swe_combined"].iloc[0]
    assert bool(swe["selected_for_llm_frame"])
    assert not bool(swe["selected_for_control_cohort"])
    assert swe["source_row_count"] == 2


@pytest.mark.unit
def test_process_chunk_marks_selected_frame_and_control_flag():
    df = pd.DataFrame(
        [
            {
                "job_id": "swe-1",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Acme",
                "description": "Build APIs and maintain services with Python, SQL, testing, CI, deployment ownership, operational support, and cross-team collaboration across environments.",
                "scrape_date": "2026-03-20",
                "is_english": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "is_control": False,
            },
            {
                "job_id": "ctrl-1",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Financial Analyst",
                "company_name": "Beta",
                "description": "Analyze budgets, forecast revenue, build dashboards, coordinate stakeholders, prepare monthly operating reviews, and support annual planning cycles.",
                "scrape_date": "2026-03-20",
                "is_english": True,
                "is_swe": False,
                "is_swe_adjacent": False,
                "is_control": True,
            },
        ]
    )

    annotated = stage9.annotate_stage9_chunk(df)
    selected_hashes = {str(annotated.loc[annotated["job_id"] == "ctrl-1", "extraction_input_hash"].iloc[0])}
    processed = stage9.process_chunk(df, selected_frame_hashes=selected_hashes).set_index("job_id")

    assert not bool(processed.loc["swe-1", "selected_for_llm_frame"])
    assert processed.loc["swe-1", "llm_extraction_sample_tier"] == "none"
    assert processed.loc["swe-1", "llm_extraction_reason"] == "not_selected"

    assert bool(processed.loc["ctrl-1", "selected_for_llm_frame"])
    assert bool(processed.loc["ctrl-1", "selected_for_control_cohort"])
    assert processed.loc["ctrl-1", "llm_extraction_sample_tier"] == "core"
    assert bool(processed.loc["ctrl-1", "needs_llm_extraction"])
    assert processed.loc["ctrl-1", "llm_extraction_reason"] == "routed"


@pytest.mark.unit
def test_process_chunk_marks_supplemental_cache_without_expanding_core():
    df = pd.DataFrame(
        [
            {
                "job_id": "core-1",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Acme",
                "description": _long_description(
                    "Build APIs and maintain services with Python, testing, deployment ownership, and operational support.",
                    "Benefits include health insurance, PTO, commuter reimbursement, and wellness stipends for employees.",
                ),
                "scrape_date": "2026-03-20",
                "is_english": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "is_control": False,
            },
            {
                "job_id": "supp-1",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Beta",
                "description": _long_description(
                    "Design backend systems and maintain production services with Python, SQL, testing, and release coordination.",
                    "Benefits include equity, wellness programs, paid leave, and retirement matching for employees.",
                ),
                "scrape_date": "2026-03-20",
                "is_english": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "is_control": False,
            },
        ]
    )

    annotated = stage9.annotate_stage9_chunk(df)
    core_hash = str(annotated.loc[annotated["job_id"] == "core-1", "extraction_input_hash"].iloc[0])
    supplemental_hash = str(annotated.loc[annotated["job_id"] == "supp-1", "extraction_input_hash"].iloc[0])
    processed = stage9.process_chunk(
        df,
        selected_frame_hashes={core_hash},
        supplemental_cached_hashes={supplemental_hash},
    ).set_index("job_id")

    assert bool(processed.loc["core-1", "selected_for_llm_frame"])
    assert processed.loc["core-1", "llm_extraction_sample_tier"] == "core"
    assert bool(processed.loc["core-1", "needs_llm_extraction"])

    assert not bool(processed.loc["supp-1", "selected_for_llm_frame"])
    assert processed.loc["supp-1", "llm_extraction_sample_tier"] == "supplemental_cache"
    assert not bool(processed.loc["supp-1", "needs_llm_extraction"])
    assert processed.loc["supp-1", "llm_extraction_reason"] == "not_selected"


@pytest.mark.unit
def test_resolve_description_core_llm_returns_empty_string_when_ok_response_drops_all_units():
    row = pd.Series(
        {
            "description": "see our Careers page on our website for further information.\nShow more\nShow less",
            "short_description_skip": False,
            "analysis_group": "swe_combined",
            "selected_for_llm_frame": True,
            "extraction_input_hash": "hash-1",
        }
    )
    cached_rows = {
        "hash-1": {
            "response_json": (
                '{"task_status":"ok","boilerplate_unit_ids":[1],"uncertain_unit_ids":[],"reason":"application instruction"}'
            )
        }
    }

    assert stage9.resolve_description_core_llm(row, cached_rows) == ""


@pytest.mark.unit
def test_summarize_stage9_routing_reports_frame_metrics():
    prepared = pd.DataFrame(
        [
            {
                "is_linkedin": True,
                "is_english": True,
                "analysis_group": "swe_combined",
                "is_control": False,
                "llm_extraction_reason": "routed",
            },
            {
                "is_linkedin": True,
                "is_english": True,
                "analysis_group": "swe_combined",
                "is_control": False,
                "llm_extraction_reason": "short_description",
            },
            {
                "is_linkedin": True,
                "is_english": True,
                "analysis_group": "control",
                "is_control": True,
                "llm_extraction_reason": "not_selected",
            },
        ]
    )
    candidate_summary = pd.DataFrame(
        [
            {"analysis_group": "swe_combined", "source_row_count": 2},
            {"analysis_group": "control", "source_row_count": 1},
        ]
    )

    summary = stage9.summarize_stage9_routing(
        prepared,
        candidate_summary,
        cached_task_count=1,
        fresh_task_count=2,
    )

    assert summary == {
        "total_rows": 3,
        "linkedin_rows": 3,
        "english_rows": 3,
        "analysis_scope_rows": 3,
        "control_scope_rows": 1,
        "short_skip_rows": 1,
        "routed_rows": 1,
        "not_selected_rows": 1,
        "unique_tasks": 2,
        "swe_combined_tasks": 1,
        "control_tasks": 1,
        "duplicate_rows_collapsed": 1,
        "cached_tasks": 1,
        "fresh_tasks": 2,
    }


@pytest.mark.unit
def test_run_stage9_reuses_supplemental_cache_outside_core_without_expanding_core(tmp_path, monkeypatch):
    input_path = tmp_path / "stage8_final.parquet"
    candidates_path = tmp_path / "stage9_candidates.parquet"
    results_path = tmp_path / "stage9_results.parquet"
    cleaned_path = tmp_path / "stage9_cleaned.parquet"
    control_path = tmp_path / "stage9_control.parquet"
    manifest_path = tmp_path / "stage9_core_frame_manifest.json"
    cache_db = tmp_path / "llm_responses.db"

    rows = pd.DataFrame(
        [
            {
                "job_id": "row-a",
                "uid": "row-a",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Acme",
                "description": _long_description(
                    "Build APIs and maintain services across production systems with Python, testing, deployment ownership, and operational support.",
                    "Benefits include health insurance, PTO, commuter reimbursement, and wellness stipends for employees.",
                ),
                "scrape_date": "2026-03-20",
                "date_posted": None,
                "is_english": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "is_control": False,
            },
            {
                "job_id": "row-b",
                "uid": "row-b",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Beta",
                "description": _long_description(
                    "Design backend systems and maintain production services with Python, SQL, testing, and release coordination.",
                    "Benefits include equity, wellness programs, paid leave, and retirement matching for employees.",
                ),
                "scrape_date": "2026-03-21",
                "date_posted": None,
                "is_english": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "is_control": False,
            },
        ]
    )
    rows.to_parquet(input_path, index=False)
    monkeypatch.setattr(stage9, "configure_logging", lambda: logging.getLogger("stage9-test"))

    candidate_rows = stage9.build_candidate_records(input_path)
    selected_rows, _ = llm_shared.select_task_frame(
        candidate_rows,
        selection_target=1,
        hash_key="extraction_input_hash",
        groups=llm_shared.ANALYSIS_GROUP_PRIORITY,
    )
    core_hash = str(selected_rows[0]["extraction_input_hash"])
    supplemental_hash = next(
        str(row["extraction_input_hash"])
        for row in candidate_rows
        if str(row["extraction_input_hash"]) != core_hash
    )

    conn = llm_shared.open_cache(cache_db)
    llm_shared.store_cached_row(
        conn,
        input_hash=supplemental_hash,
        task_name=llm_shared.EXTRACTION_TASK_NAME,
        model="gpt-5.4-mini",
        prompt_version=llm_shared.EXTRACTION_PROMPT_VERSION,
        response_json=json.dumps(
            {
                "task_status": "ok",
                "boilerplate_unit_ids": [2],
                "uncertain_unit_ids": [],
                "reason": "drop benefits",
            }
        ),
        tokens_used=12,
    )
    conn.close()

    stage9.run_stage9(
        llm_budget=0,
        llm_budget_split={"swe_combined": 1.0, "control": 0.0},
        selection_target=1,
        input_path=input_path,
        candidates_path=candidates_path,
        results_path=results_path,
        cleaned_path=cleaned_path,
        control_cohort_path=control_path,
        core_frame_manifest_path=manifest_path,
        cache_db=cache_db,
        error_log_path=tmp_path / "llm_errors.jsonl",
        max_workers=1,
        enabled_engines=("codex",),
    )

    cleaned = pd.read_parquet(cleaned_path).set_index("job_id")
    candidates = pd.read_parquet(candidates_path).set_index("job_id")
    results = pd.read_parquet(results_path).set_index("job_id")
    manifest = llm_shared.load_core_frame_manifest(manifest_path)

    core_job_id = cleaned.index[cleaned["selected_for_llm_frame"]].tolist()
    assert len(core_job_id) == 1
    assert manifest["selected_hashes"] == [core_hash]

    supplemental_job_id = "row-b" if core_job_id[0] == "row-a" else "row-a"
    assert not bool(cleaned.loc[supplemental_job_id, "selected_for_llm_frame"])
    assert cleaned.loc[supplemental_job_id, "llm_extraction_sample_tier"] == "supplemental_cache"
    assert cleaned.loc[supplemental_job_id, "llm_extraction_coverage"] == "labeled"
    assert cleaned.loc[supplemental_job_id, "llm_extraction_resolution"] == "cached_llm"
    assert isinstance(cleaned.loc[supplemental_job_id, "description_core_llm"], str)
    assert "Benefits include" not in cleaned.loc[supplemental_job_id, "description_core_llm"]

    assert not bool(candidates.loc[supplemental_job_id, "selected_for_llm_frame"])
    assert candidates.loc[supplemental_job_id, "llm_extraction_sample_tier"] == "supplemental_cache"
    assert results.loc[supplemental_job_id, "llm_extraction_sample_tier"] == "supplemental_cache"


@pytest.mark.unit
def test_run_stage9_supplemental_cache_requires_current_prompt_version(tmp_path, monkeypatch):
    input_path = tmp_path / "stage8_final.parquet"
    cleaned_path = tmp_path / "stage9_cleaned.parquet"
    manifest_path = tmp_path / "stage9_core_frame_manifest.json"
    cache_db = tmp_path / "llm_responses.db"

    rows = pd.DataFrame(
        [
            {
                "job_id": "row-a",
                "uid": "row-a",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Acme",
                "description": _long_description(
                    "Build APIs and maintain services across production systems with Python, testing, deployment ownership, and operational support.",
                    "Benefits include health insurance, PTO, commuter reimbursement, and wellness stipends for employees.",
                ),
                "scrape_date": "2026-03-20",
                "date_posted": None,
                "is_english": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "is_control": False,
            },
            {
                "job_id": "row-b",
                "uid": "row-b",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Software Engineer",
                "company_name": "Beta",
                "description": _long_description(
                    "Design backend systems and maintain production services with Python, SQL, testing, and release coordination.",
                    "Benefits include equity, wellness programs, paid leave, and retirement matching for employees.",
                ),
                "scrape_date": "2026-03-21",
                "date_posted": None,
                "is_english": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "is_control": False,
            },
        ]
    )
    rows.to_parquet(input_path, index=False)
    monkeypatch.setattr(stage9, "configure_logging", lambda: logging.getLogger("stage9-test"))

    candidate_rows = stage9.build_candidate_records(input_path)
    selected_rows, _ = llm_shared.select_task_frame(
        candidate_rows,
        selection_target=1,
        hash_key="extraction_input_hash",
        groups=llm_shared.ANALYSIS_GROUP_PRIORITY,
    )
    core_hash = str(selected_rows[0]["extraction_input_hash"])
    stale_hash = next(
        str(row["extraction_input_hash"])
        for row in candidate_rows
        if str(row["extraction_input_hash"]) != core_hash
    )

    conn = llm_shared.open_cache(cache_db)
    llm_shared.store_cached_row(
        conn,
        input_hash=stale_hash,
        task_name=llm_shared.EXTRACTION_TASK_NAME,
        model="gpt-5.4-mini",
        prompt_version="stale-prompt-version",
        response_json=json.dumps(
            {
                "task_status": "ok",
                "boilerplate_unit_ids": [2],
                "uncertain_unit_ids": [],
                "reason": "drop benefits",
            }
        ),
        tokens_used=12,
    )
    conn.close()

    stage9.run_stage9(
        llm_budget=0,
        llm_budget_split={"swe_combined": 1.0, "control": 0.0},
        selection_target=1,
        input_path=input_path,
        candidates_path=tmp_path / "stage9_candidates.parquet",
        results_path=tmp_path / "stage9_results.parquet",
        cleaned_path=cleaned_path,
        control_cohort_path=tmp_path / "stage9_control.parquet",
        core_frame_manifest_path=manifest_path,
        cache_db=cache_db,
        error_log_path=tmp_path / "llm_errors.jsonl",
        max_workers=1,
        enabled_engines=("codex",),
    )

    cleaned = pd.read_parquet(cleaned_path).set_index("job_id")
    supplemental_job_id = "row-b" if cleaned.loc["row-a", "selected_for_llm_frame"] else "row-a"

    assert not bool(cleaned.loc[supplemental_job_id, "selected_for_llm_frame"])
    assert cleaned.loc[supplemental_job_id, "llm_extraction_sample_tier"] == "none"
    assert cleaned.loc[supplemental_job_id, "llm_extraction_coverage"] == "not_selected"
    assert cleaned.loc[supplemental_job_id, "llm_extraction_resolution"] == "not_selected"
    assert pd.isna(cleaned.loc[supplemental_job_id, "description_core_llm"])


@pytest.mark.unit
def test_extract_first_json_object_repairs_missing_final_brace():
    stdout = """codex
{
  "task_status": "ok",
  "boilerplate_unit_ids": [1, 2],
  "uncertain_unit_ids": [],
  "reason": "drop boilerplate"

tokens used
4,796
"""

    assert llm_shared.extract_first_json_object(stdout) == (
        '{"task_status": "ok", "boilerplate_unit_ids": [1, 2], '
        '"uncertain_unit_ids": [], "reason": "drop boilerplate"}'
    )


@pytest.mark.unit
def test_stage9_parse_args_defaults_to_30_workers_and_budget_split(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["stage9_llm_prefilter.py", "--llm-budget", "100"],
    )
    args = stage9.parse_args()
    assert args.max_workers == 30
    assert args.llm_budget == 100
    assert args.selection_target is None
    assert args.llm_budget_split == llm_shared.DEFAULT_BUDGET_SPLIT
    assert args.core_frame_manifest == stage9.DEFAULT_CORE_FRAME_MANIFEST_PATH
    assert args.reset_core_frame is False


@pytest.mark.unit
def test_stage9_parse_args_requires_llm_budget(monkeypatch):
    monkeypatch.setattr("sys.argv", ["stage9_llm_prefilter.py"])
    with pytest.raises(SystemExit):
        stage9.parse_args()


@pytest.mark.unit
def test_join_retained_units_uses_single_newlines():
    units = [
        {"unit_id": 1, "text": "First line"},
        {"unit_id": 2, "text": "Second line"},
        {"unit_id": 3, "text": "Third line"},
    ]

    assert llm_shared.join_retained_units(units, [2]) == "First line\nThird line"


@pytest.mark.unit
def test_should_cache_extraction_result_skips_transient_provider_failures():
    assert not stage9.should_cache_extraction_result({"model": "synthetic-provider-failed"})
    assert stage9.should_cache_extraction_result({"model": "synthetic-too-many-units-fallback"})
