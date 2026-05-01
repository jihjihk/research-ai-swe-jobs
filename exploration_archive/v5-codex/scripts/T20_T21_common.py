from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "unified.parquet"
TECH_MATRIX_PATH = ROOT / "exploration" / "artifacts" / "shared" / "swe_tech_matrix.parquet"
ARCHETYPE_PATH = ROOT / "exploration" / "artifacts" / "shared" / "swe_archetype_labels.parquet"

LINKEDIN_FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe = true AND seniority_final != 'unknown'"

SCOPE_TERMS_STRICT = {
    "ownership": r"\bownership\b",
    "end_to_end": r"\bend[- ]to[- ]end\b|\be2e\b",
    "cross_functional": r"\bcross[- ]functional\b",
    "autonomous": r"\bautonomous(ly)?\b",
    "initiative": r"\binitiative\b",
    "roadmap": r"\broadmap\b",
    "strategy_scope": r"\bstrategy\b",
}

MGMT_STRICT = {
    "manage": r"\bmanage(d|r|rs|ing)?\b",
    "mentor": r"\bmentor(ship|ing)?\b",
    "coach": r"\bcoach(ing|es|ed)?\b",
    "hire": r"\bhire(d|s|ing)?\b",
    "interview": r"\binterview(s|ed|ing)?\b",
    "grow": r"\bgrow(s|th|ing)?\b",
    "develop_talent": r"\b(develop|developing)\s+(talent|people|teams?)\b",
    "performance_review": r"\bperformance review(s)?\b",
    "one_on_one": r"\b(1:1|1 on 1|one[- ]on[- ]one)\b",
    "headcount": r"\bheadcount\b",
    "people_management": r"\bpeople management\b",
    "team_building": r"\bteam building\b",
    "direct_reports": r"\bdirect reports?\b",
    "supervise": r"\bsupervis(e|ion|or|ory)\b",
}

MGMT_BROAD = {
    "lead": r"\blead(s|er|ing)?\b",
    "leadership": r"\bleadership\b",
    "team": r"\bteam(s)?\b",
    "strategic": r"\bstrategic\b",
    "stakeholder": r"\bstakeholder(s)?\b",
    "coordinate": r"\bcoordinate(d|s|ing)?\b",
    "collaborate": r"\bcollaborat(e|es|ed|ing|ion|ive)\b",
    "partner": r"\bpartner(s|ed|ing)?\b",
}

ORCH_STRICT = {
    "architecture_review": r"\barchitecture review\b",
    "code_review": r"\bcode review\b",
    "system_design": r"\bsystem design\b",
    "technical_direction": r"\btechnical direction\b",
    "ai_orchestration": r"\bai orchestration\b|\borchestrat(e|ing) (ai|agents?|workflows?)\b",
    "agentic": r"\bagent(ic|s)?\b",
    "prompt_engineering": r"\bprompt engineering\b",
    "tool_selection": r"\btool selection\b|\bselect(ing)? tools?\b",
    "guardrails": r"\bguardrails?\b",
    "quality_gate": r"\bquality gate(s)?\b",
}

ORCH_BROAD = {
    "workflow": r"\bworkflow(s)?\b",
    "pipeline": r"\bpipeline(s)?\b",
    "automation": r"\bautomation\b",
    "evaluate": r"\bevaluate(d|s|ing)?\b",
    "validate": r"\bvalidate(d|s|ing|ion)?\b",
    "orchestrate": r"\borchestrate(d|s|ing|ion)?\b",
}

STRATEGIC_STRICT = {
    "stakeholder": r"\bstakeholder(s)?\b",
    "business_impact": r"\bbusiness impact\b",
    "revenue": r"\brevenue\b",
    "product_strategy": r"\bproduct strategy\b",
    "roadmap": r"\broadmap\b",
    "prioritization": r"\bprioritization\b",
    "resource_allocation": r"\bresource allocation\b",
    "budgeting": r"\bbudgeting\b|\bbudget\b",
    "cross_functional_alignment": r"\bcross[- ]functional alignment\b",
    "go_to_market": r"\bgo[- ]to[- ]market\b|\bg2m\b",
}

STRATEGIC_BROAD = {
    "strategic": r"\bstrategic\b",
    "lead": r"\blead(s|er|ing)?\b",
    "team": r"\bteam(s)?\b",
    "cross_functional": r"\bcross[- ]functional\b",
    "ownership": r"\bownership\b",
}

EDU_LEVELS = {
    "phd": r"\b(ph\.?d\.?|phd|doctorate|doctoral)\b",
    "ms": r"\b(master'?s|masters|m\.?s\.?|m\.?sc\.?|ms\b)\b",
    "bs": r"\b(bachelor'?s|bachelors|b\.?s\.?|b\.?a\.?|b\.?sc\.?|bs\b|ba\b)\b",
}

AI_TOOL_TERMS = {
    "llm": r"\bllm(s)?\b",
    "copilot": r"\bcopilot\b",
    "cursor": r"\bcursor\b",
    "chatgpt": r"\bchatgpt\b",
    "claude": r"\bclaude\b",
    "gemini": r"\bgemini\b",
    "codex": r"\bcodex\b",
    "agent": r"\bagent(ic|s)?\b",
    "mcp": r"\bmcp\b",
    "prompt_engineering": r"\bprompt engineering\b",
    "fine_tuning": r"\bfine[- ]tuning\b",
    "rag": r"\brag\b|\bretrieval[- ]augmented generation\b",
    "langchain": r"\blangchain\b",
    "langgraph": r"\blanggraph\b",
    "llamaindex": r"\bllamaindex\b",
    "openai_api": r"\bopenai api\b",
    "anthropic_api": r"\banthropic api\b",
    "claude_api": r"\bclaude api\b",
    "gemini_api": r"\bgemini api\b",
}

AI_DOMAIN_TERMS = {
    "machine_learning": r"\bmachine learning\b",
    "deep_learning": r"\bdeep learning\b",
    "data_science": r"\bdata science\b",
    "statistics": r"\bstatistics?\b",
    "nlp": r"\bnlp\b|\bnatural language processing\b",
    "computer_vision": r"\bcomputer vision\b",
    "generative_ai": r"\bgenerative ai\b|\bgen ai\b",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def qdf(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def period_group_expr(period_col: str = "u.period") -> str:
    return f"CASE WHEN {period_col} LIKE '2024%%' THEN '2024' ELSE '2026' END"


def domain_group_expr(archetype_col: str = "a.archetype_name") -> str:
    return (
        f"CASE "
        f"WHEN {archetype_col} = 'AI / LLM workflows' THEN 'AI/LLM' "
        f"WHEN {archetype_col} IN ('Frontend / Web', 'Frontend / Angular') THEN 'Frontend' "
        f"WHEN {archetype_col} IN ('Embedded / Firmware', 'Embedded / Systems') THEN 'Embedded' "
        f"WHEN {archetype_col} IN ('Data Engineering / ETL', 'Backend / Data Platform') THEN 'Data' "
        f"WHEN {archetype_col} IN ('DevOps / Infra', 'DevOps / Tooling') THEN 'Infra' "
        f"ELSE 'Other' END"
    )


def tech_columns(con: duckdb.DuckDBPyConnection) -> list[str]:
    df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{TECH_MATRIX_PATH.as_posix()}')").fetchdf()
    return [c for c in df["column_name"].tolist() if c != "uid"]


def build_count_expr(text_col: str, terms: dict[str, str]) -> str:
    return " + ".join(
        f"CASE WHEN regexp_matches(lower(coalesce({text_col}, '')), {sql_quote(pattern)}) THEN 1 ELSE 0 END"
        for pattern in terms.values()
    ) or "0"


def build_any_expr(text_col: str, terms: dict[str, str]) -> str:
    pieces = [
        f"regexp_matches(lower(coalesce({text_col}, '')), {sql_quote(pattern)})"
        for pattern in terms.values()
    ]
    return "CASE WHEN " + " OR ".join(pieces) + " THEN 1 ELSE 0 END"


def education_expr(text_col: str) -> str:
    return (
        f"CASE "
        f"WHEN regexp_matches(lower(coalesce({text_col}, '')), {sql_quote(EDU_LEVELS['phd'])}) THEN 3 "
        f"WHEN regexp_matches(lower(coalesce({text_col}, '')), {sql_quote(EDU_LEVELS['ms'])}) THEN 2 "
        f"WHEN regexp_matches(lower(coalesce({text_col}, '')), {sql_quote(EDU_LEVELS['bs'])}) THEN 1 "
        f"ELSE 0 END"
    )


def build_term_count_expr(text_col: str, terms: dict[str, str]) -> str:
    return build_count_expr(text_col, terms)


def cohort_filter(base_filter: str, period_group: str | None = None, source: str | None = None) -> str:
    clauses = [base_filter]
    if period_group is not None:
        if period_group == "2024":
            clauses.append("u.period LIKE '2024%'")
        elif period_group == "2026":
            clauses.append("u.period LIKE '2026%'")
        else:
            raise ValueError(period_group)
    if source is not None:
        clauses.append(f"u.source = '{source}'")
    return " AND ".join(clauses)


def summarize_numeric(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"mean": math.nan, "median": math.nan, "p25": math.nan, "p75": math.nan}
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
    }


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    y = pd.to_numeric(y, errors="coerce").dropna().astype(float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2)
    if pooled <= 0:
        return float("nan")
    return float((x.mean() - y.mean()) / np.sqrt(pooled))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def assert_hygiene() -> None:
    mgmt = re.compile(MGMT_STRICT["manage"], re.I)
    mentor = re.compile(MGMT_STRICT["mentor"], re.I)
    broad_lead = re.compile(MGMT_BROAD["lead"], re.I)
    broad_leadership = re.compile(MGMT_BROAD["leadership"], re.I)
    orch = re.compile(ORCH_STRICT["system_design"], re.I)
    strat = re.compile(STRATEGIC_STRICT["business_impact"], re.I)
    edu = re.compile(EDU_LEVELS["ms"], re.I)
    ai = re.compile(AI_TOOL_TERMS["mcp"], re.I)

    assert mgmt.search("manage a team")
    assert mentor.search("mentor junior engineers")
    assert not mgmt.search("management consultant")
    assert broad_lead.search("lead the team")
    assert broad_leadership.search("leadership and team alignment")
    assert orch.search("system design and architecture review")
    assert strat.search("business impact and roadmap prioritization")
    assert edu.search("M.S. in computer science")
    assert ai.search("MCP-based workflow")
