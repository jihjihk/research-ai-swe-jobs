import math

import pytest

from tests.helpers.imports import load_module


stage5 = load_module("stage5_classification", "preprocessing/scripts/stage5_classification.py")


YOE_CASES = [
    (
        "minimum_five",
        "Qualifications: Minimum five years of recent experience with writing and supporting GitHub Actions pipelines.",
        5.0,
        "required_total_role",
    ),
    (
        "experience_label",
        "Experience: 10+ Years. Need to go for Face to Face interview.",
        10.0,
        "overall_total_role",
    ),
    (
        "experience_level",
        "Experience Level: 5+ years in software engineering.",
        5.0,
        "overall_total_role",
    ),
    (
        "degree_parallel",
        "Bachelor's degree with 6 years of experience or Master's degree with 4 years of experience.",
        4.0,
        "degree_ladder_parallel_min_path",
    ),
    (
        "cloud_range",
        "Required Skills: Minimum of 2-5+ years of experience with designing, delivering, maintaining, troubleshooting, and supporting various cloud environments.",
        2.0,
        "range_primary",
    ),
    (
        "working_in",
        "Minimum of 7 years working in data engineering and analytics.",
        7.0,
        "required_subskill_fallback",
    ),
    (
        "years_domain",
        "Required qualifications: 5 years of Mobile Development and 3 years of Kotlin Development.",
        5.0,
        "required_total_role",
    ),
    (
        "required_total_vs_aux",
        "Must have at least 10+ years' experience. Must have at least 1 year lead experience.",
        10.0,
        "required_total_role",
    ),
    (
        "founder_history",
        "About Our Founder: Gucci Westman has more than 20 years of professional experience as a major editorial makeup artist.",
        None,
        "no_valid_mentions",
    ),
    (
        "certification_deferral",
        "Board Certification requirement may be deferred up to 1 year if approved by Chief Nursing Officer.",
        None,
        "no_valid_mentions",
    ),
    (
        "company_history",
        "With over 40 years of experience, PB Built has built over 1 million square feet of commercial space.",
        None,
        "no_valid_mentions",
    ),
    (
        "zero_floor_policy",
        "Applicants may have 0-2 years of relevant experience.",
        None,
        "no_valid_mentions",
    ),
]


def _matches_expected(actual, expected) -> bool:
    if expected is None:
        return actual is None or (isinstance(actual, float) and math.isnan(actual))
    if actual is None or (isinstance(actual, float) and math.isnan(actual)):
        return False
    return float(actual) == float(expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("case_id", "text", "expected_yoe", "expected_rule"),
    YOE_CASES,
    ids=[case[0] for case in YOE_CASES],
)
def test_extract_yoe_features_matches_regression_cases(case_id, text, expected_yoe, expected_rule):
    del case_id
    result = stage5.extract_yoe_features(text)

    assert _matches_expected(result["yoe_extracted"], expected_yoe)
    assert result["yoe_resolution_rule"] == expected_rule


@pytest.mark.unit
@pytest.mark.parametrize(
    ("seniority", "yoe_extracted", "expected"),
    [
        ("entry", 3.0, True),
        ("entry", 2.0, False),
        ("associate", 5.0, True),
        ("associate", 4.0, False),
        ("mid-senior", 8.0, False),
    ],
)
def test_has_yoe_contradiction_uses_current_thresholds(seniority, yoe_extracted, expected):
    assert stage5.has_yoe_contradiction(seniority, yoe_extracted) is expected
