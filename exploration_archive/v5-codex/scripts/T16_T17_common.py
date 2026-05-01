#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from T14_T15_common import ensure_dir, load_cleaned_text, load_tech_matrix, tech_columns


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
SHARED = ROOT / "exploration" / "artifacts" / "shared"
ARCHETYPE_PATH = SHARED / "swe_archetype_labels.parquet"

REPORT_DIR = ROOT / "exploration" / "reports"
FIG_DIR = ROOT / "exploration" / "figures"
TABLE_DIR = ROOT / "exploration" / "tables"

LINKEDIN_FILTER = "u.source_platform = 'linkedin' AND u.is_english = true AND u.date_flag = 'ok' AND u.is_swe = true"

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

SCOPE_TERMS_STRICT = {
    "ownership": r"\bownership\b",
    "end_to_end": r"end[- ]to[- ]end|\be2e\b",
    "cross_functional": r"cross[- ]functional",
    "autonomous": r"\bautonomous(ly)?\b",
    "roadmap": r"\broadmap\b",
}

EDU_LEVELS = {
    "phd": r"\b(ph\.?d\.?|phd|doctorate|doctoral)\b",
    "ms": r"\b(master'?s|masters|m\.?s\.?|m\.?sc\.?|ms\b)\b",
    "bs": r"\b(bachelor'?s|bachelors|b\.?s\.?|b\.?a\.?|b\.?sc\.?|bs\b|ba\b)\b",
}

AI_TOOL_COLUMNS = {
    "llm",
    "openai_api",
    "anthropic_api",
    "claude_api",
    "gemini_api",
    "prompt_engineering",
    "fine_tuning",
    "mcp",
    "agent",
    "copilot",
    "cursor",
    "chatgpt",
    "claude",
    "gemini",
    "codex",
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
}

DOMAIN_FAMILY_MAP = {
    "AI / LLM workflows": "AI / LLM",
    "Frontend / Web": "Frontend / Mobile",
    "Frontend / Angular": "Frontend / Mobile",
    "Mobile / iOS": "Frontend / Mobile",
    "Embedded / Firmware": "Embedded",
    "Embedded / Systems": "Embedded",
    "Backend / API": "Backend",
    "Backend / Data Platform": "Data / Platform",
    "Data Engineering / ETL": "Data / Platform",
    "DevOps / Infra": "DevOps",
    "DevOps / Tooling": "DevOps",
    "Requirements / Compliance": "Requirements / Workflow",
    "Requirements / Boilerplate": "Requirements / Workflow",
    "Generic Workflow / Admin": "Requirements / Workflow",
    "Cross-functional / Delivery": "Requirements / Workflow",
}


def ensure_dirs() -> None:
    for path in [REPORT_DIR, FIG_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def validate_regexes() -> None:
    ai = re.compile(r"\b(llm(s)?|openai api|claude api|gemini api|prompt engineering|fine[- ]tuning|mcp|agent(s|ic)?|copilot|cursor|chatgpt|claude|gemini|codex|langchain|langgraph|llamaindex|rag|vector (db|database|databases)|pinecone|weaviate|chroma|milvus|faiss)\b", re.I)
    scope = re.compile(SCOPE_TERMS_STRICT["cross_functional"], re.I)
    ownership = re.compile(SCOPE_TERMS_STRICT["ownership"], re.I)
    end_to_end = re.compile(SCOPE_TERMS_STRICT["end_to_end"], re.I)
    assert ai.search("LLM, Cursor, Claude, and fine-tuning")
    assert scope.search("cross-functional ownership")
    assert ownership.search("ownership of the roadmap")
    assert end_to_end.search("end-to-end platform work")
    assert not ai.search("general artificial intelligence strategy")


def load_base_frame() -> pd.DataFrame:
    tech_cols = tech_columns()
    meta_cols = [
        "uid",
        "source",
        "period",
        "company_name_canonical",
        "company_name_effective",
        "company_industry",
        "company_size",
        "is_aggregator",
        "is_remote_inferred",
        "is_multi_location",
        "metro_area",
        "seniority_final",
        "seniority_3level",
        "yoe_extracted",
        "description_length",
    ]
    con = duckdb.connect()
    meta = con.execute(
        f"""
        SELECT {", ".join(meta_cols)}
        FROM read_parquet('{DATA.as_posix()}') u
        WHERE {LINKEDIN_FILTER}
          AND (
            (u.source = 'kaggle_arshkon' AND u.period = '2024-04')
            OR (u.source = 'scraped' AND u.period IN ('2026-03', '2026-04'))
          )
        """
    ).fetchdf()
    text = load_cleaned_text(["uid", "description_cleaned", "text_source"])
    tech = load_tech_matrix(tech_cols)
    archetypes = con.execute(
        f"""
        SELECT uid, archetype_name
        FROM read_parquet('{ARCHETYPE_PATH.as_posix()}')
        """
    ).fetchdf()
    frame = meta.merge(text, on="uid", how="left").merge(tech, on="uid", how="left").merge(archetypes, on="uid", how="left")
    frame["company_key"] = (
        frame["company_name_canonical"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "unknown_company")
    )
    frame["year"] = np.where(frame["period"].astype(str).str.startswith("2024"), "2024", "2026")
    frame["domain_family"] = frame["archetype_name"].map(DOMAIN_FAMILY_MAP).fillna(frame["archetype_name"].fillna("Other"))
    return frame


def ai_cols_from_tech(tech_cols: list[str]) -> list[str]:
    return [c for c in tech_cols if c in AI_TOOL_COLUMNS]


def family_count(text: pd.Series, patterns: dict[str, str]) -> pd.Series:
    lower = text.fillna("").str.lower()
    total = pd.Series(0, index=text.index, dtype=int)
    for pat in patterns.values():
        total += lower.str.contains(pat, regex=True, na=False).astype(int)
    return total


def add_row_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tech_cols = tech_columns()
    ai_cols = ai_cols_from_tech(tech_cols)
    out["clean_len"] = out["description_cleaned"].fillna("").str.len().astype(float)
    out["tech_count"] = out[tech_cols].fillna(False).astype(bool).sum(axis=1).astype(int)
    out["tech_any"] = out["tech_count"] > 0
    out["ai_count"] = out[ai_cols].fillna(False).astype(bool).sum(axis=1).astype(int) if ai_cols else 0
    out["ai_any"] = out["ai_count"] > 0
    out["scope_count"] = family_count(out["description_cleaned"], SCOPE_TERMS_STRICT)
    out["scope_any"] = out["scope_count"] > 0
    out["soft_skill_count"] = family_count(out["description_cleaned"], SOFT_SKILL_TERMS)
    out["soft_skill_any"] = out["soft_skill_count"] > 0
    lower = out["description_cleaned"].fillna("").str.lower()
    out["education_level"] = np.select(
        [
            lower.str.contains(EDU_LEVELS["phd"], regex=True, na=False),
            lower.str.contains(EDU_LEVELS["ms"], regex=True, na=False),
            lower.str.contains(EDU_LEVELS["bs"], regex=True, na=False),
        ],
        ["phd", "ms", "bs"],
        default="none",
    )
    out["education_flag"] = (out["education_level"] != "none").astype(int)
    out["yoe_flag"] = out["yoe_extracted"].notna().astype(int)
    out["entry_final"] = out["seniority_final"].eq("entry")
    out["entry_yoe"] = out["yoe_extracted"].fillna(np.inf).le(2)
    out["ai_domain"] = out["domain_family"].eq("AI / LLM")
    out["requirement_breadth"] = (
        out["tech_count"]
        + out["soft_skill_count"]
        + out["scope_count"]
        + out["ai_count"]
        + out["education_flag"]
        + out["yoe_flag"]
    )
    out["stack_depth"] = (
        out["tech_any"].astype(int)
        + out["soft_skill_any"].astype(int)
        + out["scope_any"].astype(int)
        + out["ai_any"].astype(int)
        + out["education_flag"]
        + out["yoe_flag"]
    )
    return out


def company_period_summary(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["company_name_canonical", "year"]
    rows = []
    grouped = df.groupby(group_cols, dropna=False)
    for keys, sub in grouped:
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "n_posts": int(len(sub)),
                "llm_text_share": float((sub["text_source"] == "llm").mean()),
                "entry_final_share": float(sub["entry_final"].mean()),
                "entry_yoe_share": float(sub["entry_yoe"].mean()),
                "ai_any_share": float(sub["ai_any"].mean()),
                "ai_count_mean": float(sub["ai_count"].mean()),
                "scope_any_share": float(sub["scope_any"].mean()),
                "scope_count_mean": float(sub["scope_count"].mean()),
                "soft_skill_any_share": float(sub["soft_skill_any"].mean()),
                "soft_skill_count_mean": float(sub["soft_skill_count"].mean()),
                "clean_len_mean": float(sub["clean_len"].mean()),
                "tech_count_mean": float(sub["tech_count"].mean()),
                "requirement_breadth_mean": float(sub["requirement_breadth"].mean()),
                "stack_depth_mean": float(sub["stack_depth"].mean()),
                "ai_domain_share": float(sub["ai_domain"].mean()),
                "remote_share": float(sub["is_remote_inferred"].fillna(False).mean()),
            }
        )
        domain_counts = sub["domain_family"].value_counts(dropna=False)
        row["domain_entropy"] = normalized_entropy(domain_counts)
        row["top_domain_family"] = domain_counts.index[0] if len(domain_counts) else "unknown"
        row["top_domain_share"] = float(domain_counts.iloc[0] / len(sub)) if len(domain_counts) else 0.0
        row["company_industry_mode"] = (
            sub["company_industry"].dropna().astype(str).mode().iloc[0]
            if sub["company_industry"].notna().any()
            else "unknown"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def company_cap(df: pd.DataFrame, cap: int, group_cols: list[str]) -> pd.DataFrame:
    out = df.sort_values(group_cols + ["uid"]).copy()
    out["_rank"] = out.groupby(group_cols).cumcount()
    return out[out["_rank"] < cap].drop(columns=["_rank"])


def normalized_entropy(counts: pd.Series) -> float:
    probs = counts[counts > 0].astype(float)
    if probs.empty:
        return float("nan")
    probs = probs / probs.sum()
    entropy = float(-(probs * np.log(probs)).sum())
    if len(probs) <= 1:
        return 0.0
    return entropy / float(np.log(len(probs)))


def format_pct(x: float) -> float:
    return round(float(x) * 100.0, 2)
