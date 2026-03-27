import pandas as pd
import pytest

from tests.helpers.imports import load_module


stage9 = load_module("stage9_llm_prefilter", "preprocessing/scripts/stage9_llm_prefilter.py")


@pytest.mark.unit
def test_annotate_stage9_chunk_marks_short_descriptions_and_control_buckets():
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
                "period": "2026_current",
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
                "period": "2024_historical",
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
                "period": "2026_current",
                "is_english": True,
                "is_swe": False,
                "is_swe_adjacent": True,
                "is_control": False,
            },
        ]
    )

    out = stage9.annotate_stage9_chunk(df).set_index("job_id")

    assert out.loc["s9_r1", "raw_description_word_count"] >= 15
    assert out.loc["s9_r1", "control_bucket"] == "scraped|2026-12"
    assert out.loc["s9_r1", "eligible_for_extraction"]
    assert out.loc["s9_r1", "llm_text_skip_reason"] is None

    assert out.loc["s9_r2", "control_bucket"] == "kaggle_arshkon|2024_historical"
    assert out.loc["s9_r2", "eligible_control_unit"]
    assert not out.loc["s9_r2", "eligible_for_extraction"]

    assert out.loc["s9_r3", "raw_description_word_count"] == 3
    assert out.loc["s9_r3", "llm_text_skip_reason"] == "short_description_under_15_words"
    assert not out.loc["s9_r3", "eligible_for_extraction"]
    assert out.loc["s9_r3", "description_core_llm"] == ""


@pytest.mark.unit
def test_select_control_cohort_is_deterministic_and_redistributes_shortfall():
    df = pd.DataFrame(
        [
            {
                "uid": "swe-a1",
                "job_id": "swe-a1",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Software Engineer I",
                "company_name": "Acme",
                "description": "Build APIs and maintain backend services with Python, SQL, testing, CI, deployment ownership, incident response, and documentation.",
                "scrape_date": "2026-03-20",
                "period": "2026_current",
                "is_english": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "is_control": False,
            },
            {
                "uid": "swe-a2",
                "job_id": "swe-a2",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Software Engineer II",
                "company_name": "Acme",
                "description": "Design APIs and maintain backend services with Python, SQL, testing, CI, deployment ownership, incident response, and documentation.",
                "scrape_date": "2026-03-20",
                "period": "2026_current",
                "is_english": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "is_control": False,
            },
            {
                "uid": "ctrl-a1",
                "job_id": "ctrl-a1",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Financial Analyst",
                "company_name": "Beta",
                "description": "Analyze budgets, forecast revenue, build dashboards, coordinate stakeholders, prepare monthly operating reviews, and support annual planning cycles.",
                "scrape_date": "2026-03-20",
                "period": "2026_current",
                "is_english": True,
                "is_swe": False,
                "is_swe_adjacent": False,
                "is_control": True,
            },
            {
                "uid": "swe-b1",
                "job_id": "swe-b1",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Platform Engineer",
                "company_name": "Gamma",
                "description": "Build platform tooling, automate deployments, support observability, improve developer experience across teams, and maintain shared infrastructure standards.",
                "scrape_date": "2026-03-27",
                "period": "2026_current",
                "is_english": True,
                "is_swe": True,
                "is_swe_adjacent": False,
                "is_control": False,
            },
            {
                "uid": "ctrl-b1",
                "job_id": "ctrl-b1",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Operations Manager",
                "company_name": "Delta",
                "description": "Manage scheduling, vendor relationships, process metrics, stakeholder communication, weekly operations planning, and cross-functional process improvement work.",
                "scrape_date": "2026-03-27",
                "period": "2026_current",
                "is_english": True,
                "is_swe": False,
                "is_swe_adjacent": False,
                "is_control": True,
            },
            {
                "uid": "ctrl-b2",
                "job_id": "ctrl-b2",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Marketing Manager",
                "company_name": "Epsilon",
                "description": "Lead campaigns, manage agency execution, own reporting cadence, coordinate launches, synthesize market insights, and maintain stakeholder alignment.",
                "scrape_date": "2026-03-27",
                "period": "2026_current",
                "is_english": True,
                "is_swe": False,
                "is_swe_adjacent": False,
                "is_control": True,
            },
            {
                "uid": "ctrl-b3",
                "job_id": "ctrl-b3",
                "source": "scraped",
                "source_platform": "linkedin",
                "title": "Product Manager",
                "company_name": "Zeta",
                "description": "Own roadmap planning, define requirements, coordinate launches, manage stakeholders, track product outcomes, and support quarterly business reviews.",
                "scrape_date": "2026-03-27",
                "period": "2026_current",
                "is_english": True,
                "is_swe": False,
                "is_swe_adjacent": False,
                "is_control": True,
            },
        ]
    )

    annotated = stage9.annotate_stage9_chunk(df)
    selected = stage9.select_control_cohort(annotated)

    assert selected["selected_for_control_cohort"].sum() == 3
    counts = selected.groupby("control_bucket")["selected_for_control_cohort"].sum().to_dict()
    assert counts == {
        "scraped|2026-12": 1,
        "scraped|2026-13": 2,
    }

    rerun = stage9.select_control_cohort(annotated)
    pd.testing.assert_frame_equal(
        selected.sort_values("extraction_input_hash").reset_index(drop=True),
        rerun.sort_values("extraction_input_hash").reset_index(drop=True),
    )


@pytest.mark.unit
def test_build_extraction_candidates_collapses_duplicate_inputs_and_includes_selected_controls():
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
                "period": "2026_current",
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
                "period": "2026_current",
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
                "period": "2026_current",
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
                "period": "2026_current",
                "is_english": True,
                "is_swe": False,
                "is_swe_adjacent": True,
                "is_control": False,
            },
        ]
    )

    annotated = stage9.annotate_stage9_chunk(df)
    control_cohort = stage9.select_control_cohort(annotated)
    candidates = stage9.build_extraction_candidates(annotated, control_cohort).sort_values("llm_route_group")

    assert candidates["source_row_count"].sum() == 3
    assert candidates["extraction_input_hash"].nunique() == 2
    assert set(candidates["llm_route_group"]) == {"control_extraction", "technical_extraction"}

    technical = candidates[candidates["llm_route_group"] == "technical_extraction"].iloc[0]
    assert technical["source_row_count"] == 2
    assert technical["selected_for_control_cohort"] == False

    control = candidates[candidates["llm_route_group"] == "control_extraction"].iloc[0]
    assert control["source_row_count"] == 1
    assert control["selected_for_control_cohort"] == True
