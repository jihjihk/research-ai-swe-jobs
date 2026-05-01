from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
TECH_MATRIX = ROOT / "exploration" / "artifacts" / "shared" / "swe_tech_matrix.parquet"

REPORT_DIR = ROOT / "exploration" / "reports"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def qdf(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return path


def save_fig(fig: plt.Figure, path: Path) -> Path:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


AI_TOOL_TERMS = {
    "llm": r"\bllm(s)?\b|\blarge language model(s)?\b",
    "openai_api": r"\bopenai api\b|\bopenai\b",
    "anthropic_api": r"\banthropic api\b|\banthropic\b",
    "claude_api": r"\bclaude api\b|\bclaude\b",
    "gemini_api": r"\bgemini api\b|\bgemini\b",
    "langchain": r"\blangchain\b",
    "langgraph": r"\blanggraph\b",
    "llamaindex": r"\bllamaindex\b",
    "rag": r"\brag\b|\bretrieval[- ]augmented generation\b",
    "vector_db": r"\bvector db\b|\bvector database(s)?\b",
    "pinecone": r"\bpinecone\b",
    "weaviate": r"\bweaviate\b",
    "chroma": r"\bchroma\b",
    "milvus": r"\bmilvus\b",
    "faiss": r"\bfaiss\b",
    "prompt_engineering": r"\bprompt engineering\b",
    "fine_tuning": r"\bfine[- ]tuning\b",
    "mcp": r"\bmcp\b",
    "copilot": r"\bcopilot\b",
    "cursor": r"\bcursor\b",
    "chatgpt": r"\bchatgpt\b",
    "claude": r"\bclaude\b",
    "gemini": r"\bgemini\b",
    "codex": r"\bcodex\b",
    "agent": r"\bagent(ic|s|ed|ing)?\b",
}

AI_ANY_TERMS = {
    **AI_TOOL_TERMS,
    "machine_learning": r"\bmachine learning\b",
    "deep_learning": r"\bdeep learning\b",
    "data_science": r"\bdata science\b",
    "statistics": r"\bstatistics\b",
    "nlp": r"\bnlp\b|\bnatural language processing\b",
    "computer_vision": r"\bcomputer vision\b",
    "generative_ai": r"\bgenerative ai\b|\bgen ai\b",
}

SCOPE_TERMS = {
    "ownership": r"\bownership\b",
    "end_to_end": r"end[- ]to[- ]end|\be2e\b",
    "cross_functional": r"cross[- ]functional",
    "autonomous": r"\bautonomous(ly)?\b",
    "initiative": r"\binitiative\b",
    "strategic": r"\bstrategic\b",
    "roadmap": r"\broadmap\b",
}

SOFT_SKILL_TERMS = {
    "communication": r"\bcommunication(s| skill(s)?)?\b",
    "collaboration": r"\bcollaboration|collaborative\b",
    "problem_solving": r"problem[- ]solving",
    "teamwork": r"\bteamwork\b",
    "presentation": r"\bpresentation(s)?\b",
    "interpersonal": r"\binterpersonal\b",
    "adaptability": r"\badaptability\b",
    "detail_oriented": r"detail[- ]oriented|attention to detail",
    "customer_focus": r"\bcustomer[- ]?facing|customer[- ]?focused\b",
}

MGMT_STRONG_TERMS = {
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

MGMT_BROAD_TERMS = {
    "lead": r"\blead(s|er|ing)?\b",
    "team": r"\bteam(s)?\b",
    "stakeholder": r"\bstakeholder(s)?\b",
    "coordinate": r"\bcoordinate(d|s|ing)?\b",
    "collaborate": r"\bcollaborat(e|es|ed|ing|ion|ive)\b",
    "partner": r"\bpartner(s|ed|ing)?\b",
}

EDU_LEVELS = {
    "phd": r"\b(ph\.?d\.?|phd|doctorate|doctoral)\b",
    "ms": r"\b(master'?s|masters|m\.?s\.?|m\.?sc\.?|ms\b)\b",
    "bs": r"\b(bachelor'?s|bachelors|b\.?s\.?|b\.?a\.?|b\.?sc\.?|bs\b|ba\b)\b",
}

TECH_SPECIAL_PATTERNS = {
    "c_plus_plus": r"(?:^|[^a-z0-9])(?:c\+\+|cpp|c plus plus)(?:$|[^a-z0-9])",
    "c_sharp": r"(?:^|[^a-z0-9])(?:c#|c sharp)(?:$|[^a-z0-9])",
    "dotnet": r"(?:^|[^a-z0-9])(?:\.net|asp\.net|aspnet|dotnet)(?:$|[^a-z0-9])",
    "nodejs": r"(?:^|[^a-z0-9])(?:node\.?js|nodejs)(?:$|[^a-z0-9])",
    "nextjs": r"(?:^|[^a-z0-9])(?:next\.?js|nextjs)(?:$|[^a-z0-9])",
    "django_rest_framework": r"\bdjango rest framework\b|\bdrf\b",
    "gitlab_ci": r"\bgitlab ci\b|\bgitlab-ci\b",
    "github_actions": r"\bgithub actions\b|\bgithub-actions\b",
    "ci_cd": r"\bci/cd\b|\bci cd\b",
    "machine_learning": r"\bmachine learning\b",
    "deep_learning": r"\bdeep learning\b",
    "data_science": r"\bdata science\b",
    "computer_vision": r"\bcomputer vision\b",
    "generative_ai": r"\bgenerative ai\b|\bgen ai\b",
    "openai_api": r"\bopenai api\b|\bopenai\b",
    "anthropic_api": r"\banthropic api\b|\banthropic\b",
    "claude_api": r"\bclaude api\b",
    "gemini_api": r"\bgemini api\b",
    "prompt_engineering": r"\bprompt engineering\b",
    "fine_tuning": r"\bfine[- ]tuning\b",
    "vector_db": r"\bvector db\b|\bvector database(s)?\b",
    "hugging_face": r"\bhugging face\b",
    "scikit_learn": r"\bscikit[- ]learn\b",
    "r_language": r"(?:^|[^a-z0-9])r(?:$|[^a-z0-9])",
    "rest_api": r"\brest api\b",
    "microservices": r"\bmicroservices?\b",
    "event_driven": r"\bevent[- ]driven\b",
    "cloudformation": r"\bcloudformation\b",
}


def regex_hygiene() -> None:
    assert re.search(AI_TOOL_TERMS["llm"], "need llm and large language models", re.I)
    assert re.search(AI_TOOL_TERMS["cursor"], "cursor and copilot", re.I)
    assert re.search(SCOPE_TERMS["cross_functional"], "cross-functional ownership", re.I)
    assert not re.search(MGMT_STRONG_TERMS["manage"], "management consultant", re.I)
    assert re.search(TECH_SPECIAL_PATTERNS["c_plus_plus"], "c++", re.I)
    assert re.search(TECH_SPECIAL_PATTERNS["c_sharp"], "c#", re.I)
    assert re.search(TECH_SPECIAL_PATTERNS["dotnet"], ".net", re.I)
    assert re.search(TECH_SPECIAL_PATTERNS["nodejs"], "node.js", re.I)
    assert re.search(TECH_SPECIAL_PATTERNS["nextjs"], "next.js", re.I)


def tech_term_list(con: duckdb.DuckDBPyConnection) -> list[str]:
    cols = qdf(con, f"DESCRIBE SELECT * FROM read_parquet('{TECH_MATRIX.as_posix()}')")
    tech_cols = [c for c in cols["column_name"].tolist() if c != "uid"]
    ai_exclude = set(AI_ANY_TERMS)
    method_exclude = {
        "agile",
        "scrum",
        "kanban",
        "ci_cd",
        "code_review",
        "pair_programming",
        "unit_testing",
        "integration_testing",
        "bdd",
        "qa",
        "tdd",
    }
    return [c for c in tech_cols if c not in ai_exclude and c not in method_exclude]


def pattern_for_term(term: str) -> str:
    if term in TECH_SPECIAL_PATTERNS:
        return TECH_SPECIAL_PATTERNS[term]
    if "_" in term:
        body = re.escape(term).replace("_", r"[-_ ]+")
        return rf"(?:^|[^a-z0-9]){body}(?:$|[^a-z0-9])"
    return rf"(?:^|[^a-z0-9]){re.escape(term)}(?:$|[^a-z0-9])"


def make_count_expr(text_col: str, terms: Iterable[str], alias_prefix: str | None = None) -> tuple[str, list[str]]:
    clauses = []
    aliases: list[str] = []
    for term in terms:
        pat = pattern_for_term(term)
        clauses.append(
            f"CASE WHEN regexp_matches(lower(coalesce({text_col}, '')), {sql_quote(pat)}) THEN 1 ELSE 0 END"
        )
        aliases.append(alias_prefix or term)
    expr = " + ".join(clauses) if clauses else "0"
    return expr, aliases


def make_binary_expr(text_col: str, patterns: dict[str, str]) -> str:
    pats = [f"regexp_matches(lower(coalesce({text_col}, '')), {sql_quote(pat)})" for pat in patterns.values()]
    return " OR ".join(pats) if pats else "false"


def make_sum_expr(text_col: str, patterns: dict[str, str]) -> str:
    parts = [
        f"CASE WHEN regexp_matches(lower(coalesce({text_col}, '')), {sql_quote(pattern)}) THEN 1 ELSE 0 END"
        for pattern in patterns.values()
    ]
    return " + ".join(parts) if parts else "0"


def within_group_summary(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in value_cols:
        if col not in out.columns:
            continue
    return out.groupby(group_cols, dropna=False)[value_cols].agg(["mean", "median"]).reset_index()
