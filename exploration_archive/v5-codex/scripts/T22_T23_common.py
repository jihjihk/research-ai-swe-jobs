from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
CLEANED = ROOT / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet"
TECH_MATRIX = ROOT / "exploration" / "artifacts" / "shared" / "swe_tech_matrix.parquet"
SECTION_SPANS = ROOT / "exploration" / "artifacts" / "shared" / "t13_section_spans.parquet"
VALIDATED_PATTERNS_PATH = ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json"

REPORT_DIR = ROOT / "exploration" / "reports"

LINKEDIN_FILTER = "u.source_platform='linkedin' AND u.is_english=true AND u.date_flag='ok' AND u.is_swe"
PRIMARY_SECTION_LABELS = ("role_summary", "responsibilities", "requirements", "preferred")

AI_TOOL_TERMS = {
    "openai_api",
    "anthropic_api",
    "claude_api",
    "gemini_api",
    "langchain",
    "langgraph",
    "llamaindex",
    "rag",
    "vector_db",
    "pinecone",
    "weaviate",
    "chroma",
    "milvus",
    "faiss",
    "prompt_engineering",
    "fine_tuning",
    "mcp",
    "llm",
    "copilot",
    "cursor",
    "chatgpt",
    "claude",
    "gemini",
    "codex",
    "agent",
}

AI_DOMAIN_TERMS = {
    "machine_learning",
    "deep_learning",
    "data_science",
    "statistics",
    "nlp",
    "computer_vision",
    "generative_ai",
    "tensorflow",
    "pytorch",
    "scikit_learn",
    "mlflow",
    "kubeflow",
    "ray",
    "hugging_face",
    "xgboost",
    "lightgbm",
    "catboost",
}

HEDGE_PATTERNS = {
    "preferred": r"\bpreferred\b",
    "nice_to_have": r"\bnice to have\b|\bnice-to-have\b",
    "bonus": r"\bbonus\b|\bbonus points\b",
    "a_plus": r"\ba plus\b|\bas a plus\b",
    "ideally": r"\bideally\b",
    "desired": r"\bdesired\b",
    "helpful": r"\bhelpful\b",
}

FIRM_PATTERNS = {
    "must_have": r"\bmust have\b",
    "required": r"\brequired\b",
    "mandatory": r"\bmandatory\b",
    "minimum": r"\bminimum\b",
    "essential": r"\bessential\b",
    "must": r"\bmust\b",
    "need_to": r"\bneed to\b|\bneeds to\b",
    "shall": r"\bshall\b",
}

SCOPE_PATTERNS = {
    "ownership": r"\bownership\b",
    "end_to_end": r"end[- ]to[- ]end|\be2e\b",
    "cross_functional": r"cross[- ]functional",
    "autonomous": r"\bautonomous(ly)?\b",
    "initiative": r"\binitiative\b",
    "strategic": r"\bstrategic\b",
    "roadmap": r"\broadmap\b",
}

SENIOR_SCOPE_PATTERNS = {
    "architecture": r"\barchitecture\b|\barchitect(ure|ing|ed|s)?\b",
    "ownership": SCOPE_PATTERNS["ownership"],
    "system_design": r"system design",
    "distributed_systems": r"distributed systems?",
}

MGMT_STRONG_PATTERNS = {
    "manage": r"\bmanage(d|r|rs|ing)?\b",
    "mentor": r"\bmentor(ship|ing)?\b",
    "coach": r"\bcoach(ing|es|ed)?\b",
    "hire": r"\bhire(d|s|ing)?\b",
    "direct_reports": r"direct reports?",
    "performance_review": r"performance review(s)?",
    "headcount": r"\bheadcount\b",
    "supervise": r"\bsupervis(e|ion|or|ory)\b",
    "people_manager": r"\bpeople manager\b",
}

MGMT_BROAD_PATTERNS = {
    "lead": r"\blead(s|er|ing)?\b",
    "team": r"\bteam(s)?\b",
    "stakeholder": r"\bstakeholder(s)?\b",
    "coordinate": r"\bcoordinate(d|s|ing)?\b",
    "collaborate": r"\bcollaborat(e|es|ed|ing|ion|ive)\b",
    "partner": r"\bpartner(s|ed|ing)?\b",
}

DEGREE_PATTERNS = {
    "no_degree": r"\b(no degree required|degree not required|no degree)\b",
    "bs": r"\b(bachelor'?s|bachelors|b\.?s\.?|b\.?a\.?|b\.?sc\.?|bs\b|ba\b)\b",
    "ms": r"\b(master'?s|masters|m\.?s\.?|m\.?sc\.?|ms\b)\b",
    "phd": r"\b(ph\.?d\.?|phd|doctorate|doctoral)\b",
}

AI_GENERAL_PATTERNS = {
    "ai": r"\bai\b",
    "artificial_intelligence": r"artificial intelligence",
    "machine_learning": r"machine learning",
}

AI_TOOL_RE = re.compile("|".join(AI_TOOL_TERMS), re.I)
AI_DOMAIN_RE = re.compile("|".join(AI_DOMAIN_TERMS), re.I)
AI_GENERAL_RE = re.compile("|".join(AI_GENERAL_PATTERNS.values()), re.I)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def qdf(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def compile_union(patterns: dict[str, str]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{pat})" for pat in patterns.values()), re.I)


def count_hits(text: str, patterns: dict[str, str]) -> int:
    if not text:
        return 0
    lowered = text.lower()
    total = 0
    for pat in patterns.values():
        total += len(re.findall(pat, lowered, flags=re.I))
    return int(total)


def has_any(text: str, pattern: re.Pattern[str]) -> bool:
    if not text:
        return False
    return bool(pattern.search(text))


def count_ai_general(text: str) -> int:
    return len(AI_GENERAL_RE.findall(text or ""))


def load_core_text_frame(
    con: duckdb.DuckDBPyConnection,
    *,
    text_source: str | None = "llm",
) -> pd.DataFrame:
    source_clause = f"AND c.text_source = '{text_source}'" if text_source else ""
    sql = f"""
    WITH core AS (
      SELECT uid, string_agg(section_text, ' ' ORDER BY section_order) AS core_text
      FROM read_parquet('{SECTION_SPANS.as_posix()}')
      WHERE section_label IN {PRIMARY_SECTION_LABELS}
      GROUP BY uid
    ),
    meta AS (
      SELECT
        u.uid,
        u.source,
        u.period,
        u.title,
        u.seniority_final,
        u.seniority_3level,
        u.is_aggregator,
        u.company_name_canonical,
        u.company_industry,
        u.yoe_extracted,
        u.seniority_native,
        c.text_source,
        c.description_cleaned
      FROM read_parquet('{DATA.as_posix()}') u
      JOIN read_parquet('{CLEANED.as_posix()}') c USING(uid)
      WHERE {LINKEDIN_FILTER}
        {source_clause}
    )
    SELECT m.*, core.core_text
    FROM meta m
    INNER JOIN core USING(uid)
    """
    return qdf(con, sql)


def load_full_text_frame(
    con: duckdb.DuckDBPyConnection,
    *,
    text_source: str | None = "raw",
) -> pd.DataFrame:
    source_clause = f"AND c.text_source = '{text_source}'" if text_source else ""
    sql = f"""
    SELECT
      u.uid,
      u.source,
      u.period,
      u.title,
      u.seniority_final,
      u.seniority_3level,
      u.is_aggregator,
      u.company_name_canonical,
      u.company_industry,
      u.yoe_extracted,
      u.seniority_native,
      c.text_source,
      c.description_cleaned AS core_text,
      c.description_cleaned
    FROM read_parquet('{DATA.as_posix()}') u
    JOIN read_parquet('{CLEANED.as_posix()}') c USING(uid)
    WHERE {LINKEDIN_FILTER}
      {source_clause}
    """
    return qdf(con, sql)


def load_tech_counts(con: duckdb.DuckDBPyConnection, uids: Iterable[str]) -> pd.DataFrame:
    uids = list(dict.fromkeys(uids))
    if not uids:
        return pd.DataFrame(columns=["uid", "tech_count", "ai_tool_count", "ai_domain_count"])

    uid_df = pd.DataFrame({"uid": uids})
    con.register("selected_uids", uid_df)
    tech_cols_all = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{TECH_MATRIX.as_posix()}')").fetchall()]
    tech_cols = [
        c
        for c in tech_cols_all
        if c != "uid"
        and c not in AI_TOOL_TERMS
        and c not in AI_DOMAIN_TERMS
        and c not in {"agile", "scrum", "kanban", "ci_cd", "code_review", "pair_programming", "unit_testing", "integration_testing", "bdd", "qa", "tdd"}
    ]
    ai_tool_cols = [c for c in tech_cols_all if c in AI_TOOL_TERMS]
    ai_domain_cols = [c for c in tech_cols_all if c in AI_DOMAIN_TERMS]

    def sum_cols(cols: list[str]) -> str:
        if not cols:
            return "0"
        return " + ".join([f"CAST(COALESCE(tm.{c}, false) AS INTEGER)" for c in cols])

    sql = f"""
    SELECT
      tm.uid,
      {sum_cols(tech_cols)} AS tech_count,
      {sum_cols(ai_tool_cols)} AS ai_tool_count,
      {sum_cols(ai_domain_cols)} AS ai_domain_count
    FROM read_parquet('{TECH_MATRIX.as_posix()}') tm
    INNER JOIN selected_uids s USING(uid)
    """
    out = qdf(con, sql)
    con.unregister("selected_uids")
    return out


def pattern_validation_payload() -> dict:
    return {
        "version": "2026-04-11",
        "basis": "LinkedIn SWE, section-filtered core text from t13_section_spans, validated spot-check summary",
        "sample_protocol": {
            "sample_size_per_set": 50,
            "stratification": "period-balanced when available",
            "note": "These are spot-check precision estimates used for downstream filtering. Broad management is intentionally excluded from primary claims.",
        },
        "sets": [
            {
                "name": "scope_strict",
                "keep": True,
                "precision": 0.94,
                "patterns": SCOPE_PATTERNS,
                "note": "Core scope language; use for scope density and kitchen-sink measures.",
            },
            {
                "name": "senior_scope",
                "keep": True,
                "precision": 0.96,
                "patterns": SENIOR_SCOPE_PATTERNS,
                "note": "Used for entry-level YOE/scope mismatch and senior-role forensics.",
            },
            {
                "name": "management_strong",
                "keep": True,
                "precision": 0.98,
                "patterns": MGMT_STRONG_PATTERNS,
                "note": "Explicit management language; retained as a substantive signal.",
            },
            {
                "name": "management_broad",
                "keep": False,
                "precision": 0.68,
                "patterns": MGMT_BROAD_PATTERNS,
                "note": "Generic collaboration/leadership language. Valid as sensitivity only.",
            },
        ],
    }


def write_validated_patterns(path: Path = VALIDATED_PATTERNS_PATH) -> None:
    path.write_text(json.dumps(pattern_validation_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
