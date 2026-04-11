import json
import sys
import types
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.helpers.imports import load_module
from tests.test_stage5_yoe_extractor import YOE_CASES as YOE_REGRESSION_CASES


stage5 = load_module("stage5_classification", "preprocessing/scripts/stage5_classification.py")


TEST_FIXTURES = Path(__file__).resolve().parent / "fixtures"
SYNTHETIC_STAGE5 = TEST_FIXTURES / "synthetic" / "stage5"
SAMPLED_STAGE5 = TEST_FIXTURES / "sampled" / "stage5"


def _load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _payload_rows(rows: list[dict]) -> list[dict]:
    return [
        {k: v for k, v in row.items() if not k.startswith("expect_") and k != "review_note"}
        for row in rows
    ]


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_lookup(path: Path, rows: list[dict]) -> None:
    _write_parquet(path, rows)


def _read_output(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def _fake_sentence_transformer_module(candidate_vectors: dict[str, list[float]]):
    class FakeSentenceTransformer:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, texts, **_kwargs):
            texts = list(texts)
            if texts == stage5.SWE_REFS:
                return np.tile(np.array([1.0, 0.0, 0.0]), (len(texts), 1))
            vectors = [candidate_vectors.get(str(text).lower().strip(), [0.1, 0.0, 0.0]) for text in texts]
            return np.array(vectors, dtype=float)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    return module


@pytest.mark.unit
def test_normalize_strips_clearance_and_empty_parentheses():
    assert stage5.normalize_swe_title_key("Lead Network Engineer with Security Clearance") == "network engineer"
    assert stage5.normalize_swe_title_key("Oracle Application Developer (Senior)") == "oracle application developer"


@pytest.mark.unit
@pytest.mark.parametrize(
    "title",
    [
        "Oracle Application Developer (Senior)",
        "Junior data scientist/java programmer",
        "Senior Salesforce Developer",
        "ServiceNow Developer",
        "Mainframe Developer",
        "Senior .NET Developer",
        "Lead Python Developer",
        "Mobile iOS Developer",
        "Kotlin Engineer",
        "AI Engineer",
    ],
)
def test_primary_swe_patterns_cover_known_titles(title):
    assert stage5.is_primary_swe_title(stage5.normalize_swe_title_key(title))


@pytest.mark.unit
@pytest.mark.parametrize(
    "title",
    [
        "Senior DDI Network Engineer",
        "Sr SAP Basis Engineer",
        "Google Cloud Architect - Senior Manager",
        "Staff VMware Engineer",
        "Staff Data Scientist",
        "Security Engineer",
        "Systems Administrator",
        "Database Administrator",
        "Windows Engineer",
        "Salesforce Architect",
        "ServiceNow Architect",
        "Cloud Architect",
    ],
)
def test_adjacent_patterns_cover_missed_technical_families(title):
    assert stage5.is_adjacent_technical_title(stage5.normalize_swe_title_key(title))


@pytest.mark.unit
@pytest.mark.parametrize(
    "title",
    [
        "Civil Engineer",
        "Senior Electrical Engineer",
        "Mechanical Engineer",
        "Nurse Practitioner",
    ],
)
def test_control_titles_remain_non_swe(title):
    normalized = stage5.normalize_swe_title_key(title)
    assert not stage5.is_primary_swe_title(normalized)
    assert not stage5.is_adjacent_technical_title(normalized)


@pytest.mark.unit
@pytest.mark.parametrize(
    "rows, exc, message",
    [
        ([{"title": "Data Scientist"}], ValueError, "missing required columns"),
        (
            [
                {"title": "Data Scientist", "llm_swe_classification": "SWE"},
                {"title": "Junior Accountant", "llm_swe_classification": "maybe"},
            ],
            ValueError,
            "invalid labels",
        ),
        ([{"title": "", "llm_swe_classification": "SWE"}], ValueError, "no usable normalized title keys"),
    ],
)
def test_title_lookup_validation_errors(tmp_path, monkeypatch, rows, exc, message):
    lookup_path = tmp_path / "tier2_title_lookup.parquet"
    _write_lookup(lookup_path, rows)
    monkeypatch.setattr(stage5, "TIER2_LOOKUP_PATH", lookup_path)
    with pytest.raises(exc, match=message):
        stage5.load_tier2_title_lookup()


@pytest.mark.unit
def test_title_lookup_conflicting_keys_are_dropped(tmp_path, monkeypatch):
    lookup_path = tmp_path / "tier2_title_lookup.parquet"
    _write_lookup(
        lookup_path,
        [
            {"title": "Data Scientist", "llm_swe_classification": "SWE"},
            {"title": "Senior Data Scientist", "llm_swe_classification": "SWE_ADJACENT"},
            {"title": "Python Developer", "llm_swe_classification": "SWE"},
        ],
    )
    monkeypatch.setattr(stage5, "TIER2_LOOKUP_PATH", lookup_path)

    lookup = stage5.load_tier2_title_lookup()
    assert "data scientist" not in lookup
    assert lookup["python developer"] == "SWE"


@pytest.mark.unit
@pytest.mark.parametrize(
    "title, family, expected",
    [
        # Strong title-keyword rules → seniority_final populated
        ("Senior Software Engineer", "swe", ("mid-senior", "title_keyword")),
        ("Sr Backend Engineer", "swe", ("mid-senior", "title_keyword")),
        ("Staff Engineer", "swe", ("mid-senior", "title_keyword")),
        ("Principal Software Engineer", "swe", ("mid-senior", "title_keyword")),
        ("Lead Software Engineer", "swe", ("mid-senior", "title_keyword")),
        ("Junior Backend Developer", "swe", ("entry", "title_keyword")),
        ("Software Engineering Intern", "swe", ("entry", "title_keyword")),
        ("New Grad Software Engineer", "swe", ("entry", "title_keyword")),
        ("Director of Platform Engineering", "swe", ("director", "title_keyword")),
        ("VP of Engineering", "swe", ("director", "title_keyword")),
        # Title-manager rule (requires family role hint)
        ("Data Engineer Manager", "adjacent", ("mid-senior", "title_manager")),
        ("Accountant Manager", "control", ("mid-senior", "title_manager")),
        # Weak signals that USED to fire are now unknown
        ("Software Engineer I", "swe", ("unknown", "unknown")),
        ("Software Engineer II", "swe", ("unknown", "unknown")),
        ("Software Developer - Intermediate", "swe", ("unknown", "unknown")),
        ("Systems Engineer II", "adjacent", ("unknown", "unknown")),
        ("Accountant I", "control", ("unknown", "unknown")),
        # Plain titles with no seniority signal stay unknown
        ("Software Engineer", "swe", ("unknown", "unknown")),
        ("Backend Developer", "swe", ("unknown", "unknown")),
    ],
)
def test_classify_seniority_strong_rules_only(title, family, expected):
    """After 2026-04-10 simplification, classify_seniority returns only strong rules or unknown."""
    level, source = stage5.classify_seniority(title, family=family)
    assert (level, source) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "text, expected_yoe, expected_rule",
    [(text, expected_yoe, expected_rule) for _case_id, text, expected_yoe, expected_rule in YOE_REGRESSION_CASES],
)
def test_yoe_regression_cases(text, expected_yoe, expected_rule):
    features = stage5.extract_yoe_features(text)
    if expected_yoe is None:
        assert np.isnan(features["yoe_extracted"])
        assert features["yoe_match_count"] == 0
    else:
        assert features["yoe_extracted"] == expected_yoe
        assert features["yoe_min_extracted"] <= expected_yoe <= features["yoe_max_extracted"]
        assert features["yoe_match_count"] >= 1
    assert features["yoe_resolution_rule"] == expected_rule


@pytest.mark.unit
@pytest.mark.parametrize(
    "text, expected_reason",
    [
        ("About Our Founder: Gucci Westman has more than 20 years of professional experience as a major editorial makeup artist.", "company_history"),
        ("Board Certification requirement may be deferred up to 1 year if approved by Chief Nursing Officer.", "certification_deferral"),
        ("With over 40 years of experience, PB Built has built over 1 million square feet of commercial space.", "out_of_range"),
        ("Applicants may have 0-2 years of relevant experience.", "out_of_range"),
    ],
)
def test_yoe_candidate_rejection_reasons(text, expected_reason):
    mentions = stage5.collect_yoe_mentions(text)
    reasons = [m["reject_reason"] for m in mentions]
    assert expected_reason in reasons
    assert stage5.extract_yoe_features(text)["yoe_resolution_rule"] == "no_valid_mentions"


def _yoe_mention(**overrides):
    base = {
        "reject_reason": None,
        "required": False,
        "preferred": False,
        "total_role": False,
        "substitution": False,
        "certification_deferral": False,
        "degree_ladder": False,
        "parallel_alt": False,
        "subskill": False,
        "subordinate": False,
        "min_year": 4,
        "max_year": 4,
        "pattern": "forward_exp",
        "section_rank": 30,
        "start": 0,
    }
    base.update(overrides)
    return base


@pytest.mark.unit
@pytest.mark.parametrize(
    "mentions, expected_rule, expected_min",
    [
        (
            [
                _yoe_mention(required=True, total_role=True, min_year=5, max_year=5, section_rank=50),
                _yoe_mention(required=True, total_role=False, subskill=True, min_year=3, max_year=3, section_rank=40),
            ],
            "required_total_role",
            5,
        ),
        (
            [
                _yoe_mention(total_role=True, min_year=7, max_year=7, section_rank=30),
                _yoe_mention(total_role=True, min_year=9, max_year=9, section_rank=20),
            ],
            "overall_total_role",
            7,
        ),
        (
            [
                _yoe_mention(preferred=True, total_role=True, min_year=4, max_year=4, section_rank=20),
                _yoe_mention(preferred=True, total_role=True, min_year=6, max_year=6, section_rank=20),
            ],
            "preferred_total_role",
            6,
        ),
        (
            [
                _yoe_mention(required=True, total_role=False, min_year=3, max_year=3, section_rank=40),
                _yoe_mention(required=True, total_role=False, min_year=6, max_year=6, section_rank=40),
            ],
            "required_subskill_fallback",
            6,
        ),
        (
            [
                _yoe_mention(degree_ladder=True, parallel_alt=True, min_year=6, max_year=6, section_rank=40),
                _yoe_mention(degree_ladder=True, parallel_alt=True, min_year=4, max_year=4, section_rank=40),
            ],
            "degree_ladder_parallel_min_path",
            4,
        ),
        (
            [
                _yoe_mention(degree_ladder=True, parallel_alt=False, min_year=6, max_year=6, section_rank=40),
                _yoe_mention(degree_ladder=True, parallel_alt=False, min_year=4, max_year=4, section_rank=40),
            ],
            "degree_ladder_min_path",
            4,
        ),
        (
            [
                _yoe_mention(subskill=True, min_year=4, max_year=4),
                _yoe_mention(subskill=True, min_year=5, max_year=5),
            ],
            "subskill_only_abstain",
            None,
        ),
    ],
)
def test_choose_primary_yoe_branches(mentions, expected_rule, expected_min):
    primary, rule = stage5.choose_primary_yoe(mentions)
    assert rule == expected_rule
    if expected_min is None:
        assert primary is None
    else:
        assert primary["min_year"] == expected_min


@pytest.mark.fixture
@pytest.mark.integration
def test_tiny_stage5_parquet_integration_with_stubbed_embeddings(tmp_path, monkeypatch):
    rows = _payload_rows(_load_rows(SYNTHETIC_STAGE5 / "tiny_stage5_input.json"))
    input_path = tmp_path / "stage4_dedup.parquet"
    lookup_path = tmp_path / "tier2_title_lookup.parquet"
    output_path = tmp_path / "stage5_classification.parquet"

    _write_parquet(input_path, rows)
    _write_lookup(
        lookup_path,
        [{"title": "Data Scientist", "llm_swe_classification": "SWE_ADJACENT"}],
    )
    monkeypatch.setattr(stage5, "TIER2_LOOKUP_PATH", lookup_path)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        _fake_sentence_transformer_module(
            {
                "cloud reliability scientist": [0.95, 0.0, 0.0],
                "ai researcher": [0.78, 0.0, 0.0],
            }
        ),
    )

    swe_lookup, control_lookup = stage5.build_swe_lookup(input_path)
    seniority_title_lookup = stage5.build_seniority_title_lookup(input_path, swe_lookup, control_lookup)
    stats = stage5.streaming_write(
        input_path,
        output_path,
        swe_lookup,
        seniority_title_lookup,
        control_lookup,
    )

    output_rows = _read_output(output_path)
    assert stats["written"] == len(rows)
    assert len(output_rows) == len(rows)

    by_title = {row["title"]: row for row in output_rows}
    for row in _load_rows(SYNTHETIC_STAGE5 / "tiny_stage5_input.json"):
        out = by_title[row["title"]]
        assert bool(out["is_swe"]) == row["expect_is_swe"]
        assert bool(out["is_swe_adjacent"]) == row["expect_is_swe_adjacent"]
        assert bool(out["is_control"]) == row["expect_is_control"]
        assert out["swe_classification_tier"] == row["expect_swe_tier"]
        assert out["seniority_final"] == row["expect_seniority_final"]
        assert out["seniority_final_source"] == row["expect_seniority_final_source"]
        if row.get("expect_yoe") is not None:
            assert out["yoe_extracted"] == row["expect_yoe"]
            assert out["yoe_resolution_rule"] == row["expect_yoe_rule"]
            assert bool(out["yoe_seniority_contradiction"]) == row["expect_yoe_contradiction"]


@pytest.mark.sampled
@pytest.mark.integration
def test_reviewed_stage5_rows_cover_contract_paths(tmp_path, monkeypatch):
    rows = _payload_rows(_load_rows(SAMPLED_STAGE5 / "reviewed_stage5_rows.json"))
    input_path = tmp_path / "stage4_dedup.parquet"
    output_path = tmp_path / "stage5_classification.parquet"
    _write_parquet(input_path, rows)
    lookup_path = tmp_path / "tier2_title_lookup.parquet"
    _write_lookup(lookup_path, [{"title": "Data Scientist", "llm_swe_classification": "SWE"}])
    monkeypatch.setattr(stage5, "TIER2_LOOKUP_PATH", lookup_path)

    swe_lookup, control_lookup = stage5.build_swe_lookup(input_path)
    seniority_title_lookup = stage5.build_seniority_title_lookup(input_path, swe_lookup, control_lookup)
    stats = stage5.streaming_write(
        input_path,
        output_path,
        swe_lookup,
        seniority_title_lookup,
        control_lookup,
    )

    output_rows = _read_output(output_path)
    assert stats["written"] == len(rows)
    assert len(output_rows) == len(rows)

    by_uid = {row["uid"]: row for row in output_rows}
    for row in _load_rows(SAMPLED_STAGE5 / "reviewed_stage5_rows.json"):
        out = by_uid[row["uid"]]
        assert bool(out["is_swe"]) == row["expect_is_swe"]
        assert bool(out["is_swe_adjacent"]) == row["expect_is_swe_adjacent"]
        assert bool(out["is_control"]) == row["expect_is_control"]
        assert out["swe_classification_tier"] == row["expect_swe_tier"]
        assert out["seniority_final"] == row["expect_seniority_final"]
        assert out["seniority_final_source"] == row["expect_seniority_final_source"]
        if row.get("expect_yoe") is not None:
            assert out["yoe_extracted"] == row["expect_yoe"]
            assert out["yoe_resolution_rule"] == row["expect_yoe_rule"]
            assert bool(out["yoe_seniority_contradiction"]) == row["expect_yoe_contradiction"]
