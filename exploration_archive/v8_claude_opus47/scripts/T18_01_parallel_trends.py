"""T18 step 1 — Parallel-trends feature extraction for SWE / adjacent / control.

Operates on RAW description (not LLM-cleaned) so that the three occupation groups
are comparable (SWE has ~54% LLM coverage, adjacent ~99%, control ~24%).

Per-posting features:
  - ai_mention_strict  — V1-refined strict AI-tool pattern
  - ai_mention_broad   — V1-refined broad AI pattern (strict + generic ai/ml/llm)
  - desc_len_chars     — length(description)
  - desc_len_cleaned   — length(description_core_llm) when available else NaN
  - org_scope_count    — count of distinct org_scope patterns (V1 approved)
  - soft_skill_count
  - management_strict_count  — V1-refined strict mgmt pattern
  - tech_count         — T11 taxonomy applied to RAW description
  - ai_tech_count      — subset of tech_count that is AI-era
  - requirement_breadth = tech + scope + soft + mgmt_strict + ai_count_binary
  - education_level
  - has_* booleans

Writes feature parquet to exploration/artifacts/T18/T18_posting_features.parquet
and a compact metric-by-group-by-period CSV to exploration/tables/T18/T18_parallel_trends.csv.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNI = ROOT / "data" / "unified.parquet"
ART = ROOT / "exploration" / "artifacts" / "T18"
TAB = ROOT / "exploration" / "tables" / "T18"

ART.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# V1-refined AI patterns (from Gate 2 memo locked directives).
# ---------------------------------------------------------------------------
AI_STRICT_PAT = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|"
    r"llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|"
    r"vector database|pinecone|huggingface|hugging face)\b",
    re.IGNORECASE,
)
AI_BROAD_EXTRA_PAT = re.compile(
    r"\b(ai|artificial intelligence|ml|machine learning|llms?|large language models?|"
    r"generative ai|genai|anthropic)\b",
    re.IGNORECASE,
)

# V1-approved org scope pattern (from T11/V1)
ORG_SCOPE_PATS = {
    "ownership": re.compile(r"\bowner ?ship\b|\btakes? ownership\b", re.IGNORECASE),
    "end_to_end": re.compile(r"\bend[- ]to[- ]end\b", re.IGNORECASE),
    "cross_functional": re.compile(r"\bcross[- ]functional\b", re.IGNORECASE),
    "stakeholder": re.compile(r"\bstakeholder", re.IGNORECASE),
    "autonomous": re.compile(r"\bautonom(?:y|ous|ously)\b", re.IGNORECASE),
    "initiative": re.compile(r"\binitiative\b", re.IGNORECASE),
    "architect": re.compile(r"\barchitect(?:ure|ing)?\b", re.IGNORECASE),
    "system_design": re.compile(r"\bsystem design\b", re.IGNORECASE),
    "distributed_system": re.compile(r"\bdistributed system", re.IGNORECASE),
    "scalability": re.compile(r"\bscalab(?:le|ility)\b|\bat scale\b", re.IGNORECASE),
}

SOFT_PATS = {
    "collaboration": re.compile(r"\bcollaborat(?:e|ion|ing|ive)\b", re.IGNORECASE),
    "communication": re.compile(r"\bcommunicat(?:e|ion|ing|or|ors)\b", re.IGNORECASE),
    "problem_solving": re.compile(r"\bproblem[- ]solv(?:ing|er|ers)\b", re.IGNORECASE),
    "leadership": re.compile(r"\bleadership\b", re.IGNORECASE),
    "teamwork": re.compile(r"\bteamwork\b|\bteam[- ]player\b", re.IGNORECASE),
    "interpersonal": re.compile(r"\binterpersonal\b", re.IGNORECASE),
    "adaptable": re.compile(r"\badaptab(?:le|ility)\b|\badaptive\b|\bflexible\b", re.IGNORECASE),
    "self_motivated": re.compile(r"\bself[- ]motivat(?:e|ed|ion|ing)\b|\bself[- ]driven\b", re.IGNORECASE),
    "ownership": re.compile(r"\bowner ?ship\b|\btakes? ownership\b", re.IGNORECASE),
    "autonomous": re.compile(r"\bautonom(?:y|ous|ously)\b", re.IGNORECASE),
}

# V1-refined strict management (drop `manage`, `team_building` — <80% precision)
MGMT_STRICT_PATS = {
    "mentor": re.compile(r"\bmentor(?:s|ing|ship)?\b", re.IGNORECASE),
    "coach": re.compile(r"\bcoach(?:es|ing|ed)?\b", re.IGNORECASE),
    "hire": re.compile(r"\bhir(?:e|es|ing|ed)\b|\brecruit(?:ing|ment|ed)?\b", re.IGNORECASE),
    "direct_reports": re.compile(r"\bdirect reports?\b", re.IGNORECASE),
    "performance_review": re.compile(
        r"\bperformance reviews?\b|\bperformance appraisals?\b", re.IGNORECASE
    ),
    "headcount": re.compile(r"\bheadcount\b", re.IGNORECASE),
    "people_management": re.compile(r"\bpeople manage(?:ment|r)?\b", re.IGNORECASE),
}

EDU_PATS = {
    1: re.compile(r"\bb\.?s\.?\b|\bbachelor(?:'?s)?\b|\bb\.?sc\.?\b|\bundergraduate degree\b|\bba\b", re.IGNORECASE),
    2: re.compile(r"\bm\.?s\.?\b|\bmaster(?:'?s)?\b|\bm\.?sc\.?\b|\bgraduate degree\b", re.IGNORECASE),
    3: re.compile(r"\bph\.?d\b|\bdoctorate\b|\bdoctoral\b", re.IGNORECASE),
}

# ---------------------------------------------------------------------------
# Tech taxonomy (subset of T11 tech_matrix — shared with SWE but applied to raw
# descriptions of adjacent/control too). Matches description in lowercase.
# ---------------------------------------------------------------------------
TAXONOMY: dict[str, list[str]] = {
    "python": [r"\bpython\b"],
    "java": [r"\bjava\b(?!\s*script)"],
    "javascript": [r"\bjavascript\b", r"\bjs\b"],
    "typescript": [r"\btypescript\b"],
    "go": [r"\b(go|golang)\b"],
    "rust": [r"\brust\b"],
    "cpp": [r"\bc\+\+", r"\bcpp\b"],
    "csharp": [r"\bc#", r"\bcsharp\b"],
    "ruby": [r"\bruby\b"],
    "kotlin": [r"\bkotlin\b"],
    "swift": [r"\bswift\b"],
    "scala": [r"\bscala\b"],
    "php": [r"\bphp\b"],
    "sql": [r"\bsql\b"],
    "react": [r"\breact\b", r"\breact\.js\b", r"\breactjs\b"],
    "angular": [r"\bangular\b"],
    "vue": [r"\bvue\b", r"\bvue\.js\b", r"\bvuejs\b"],
    "nextjs": [r"\bnext\.js\b", r"\bnextjs\b"],
    "nodejs": [r"\bnode\b", r"\bnode\.js\b", r"\bnodejs\b"],
    "django": [r"\bdjango\b"],
    "flask": [r"\bflask\b"],
    "spring": [r"\bspring\b", r"\bspring\s*boot\b"],
    "dotnet": [r"\.net\b", r"\bdotnet\b"],
    "rails": [r"\brails\b", r"\bruby\s*on\s*rails\b"],
    "fastapi": [r"\bfastapi\b"],
    "aws": [r"\baws\b", r"\bamazon\s*web\s*services\b"],
    "azure": [r"\bazure\b"],
    "gcp": [r"\bgcp\b", r"\bgoogle\s*cloud\b"],
    "kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "docker": [r"\bdocker\b"],
    "terraform": [r"\bterraform\b"],
    "jenkins": [r"\bjenkins\b"],
    "github_actions": [r"\bgithub\s*actions?\b"],
    "cicd": [r"\bci\s*/\s*cd\b", r"\bci\\/cd\b", r"\bcicd\b", r"\bci-cd\b"],
    "postgresql": [r"\bpostgresql\b", r"\bpostgres\b"],
    "mysql": [r"\bmysql\b"],
    "mongodb": [r"\bmongodb\b", r"\bmongo\b"],
    "redis": [r"\bredis\b"],
    "kafka": [r"\bkafka\b"],
    "spark": [r"\bspark\b"],
    "snowflake": [r"\bsnowflake\b"],
    "databricks": [r"\bdatabricks\b"],
    "elasticsearch": [r"\belasticsearch\b", r"\belastic\s*search\b"],
    "bigquery": [r"\bbigquery\b", r"\bbig\s*query\b"],
    "tensorflow": [r"\btensorflow\b"],
    "pytorch": [r"\bpytorch\b"],
    "sklearn": [r"\bscikit\s*-?\s*learn\b", r"\bsklearn\b"],
    "pandas": [r"\bpandas\b"],
    "numpy": [r"\bnumpy\b"],
    "keras": [r"\bkeras\b"],
    "xgboost": [r"\bxgboost\b"],
    "langchain": [r"\blangchain\b"],
    "langgraph": [r"\blanggraph\b"],
    "rag_tech": [r"\brag\b", r"\bretrieval\s*augmented\b"],
    "vector_database": [r"\bvector\s*databases?\b", r"\bvector\s*dbs?\b", r"\bvector\s*stores?\b"],
    "pinecone": [r"\bpinecone\b"],
    "huggingface": [r"\bhuggingface\b", r"\bhugging\s*face\b"],
    "openai": [r"\bopenai\b"],
    "claude_tech": [r"\bclaude\b"],
    "gemini_tech": [r"\bgemini\b"],
    "llamaindex": [r"\bllamaindex\b", r"\bllama\s*index\b"],
    "anthropic": [r"\banthropic\b"],
    "ollama": [r"\bollama\b"],
    "copilot": [r"\bcopilot\b", r"\bgithub\s*copilot\b"],
    "cursor": [r"\bcursor\b"],
    "chatgpt": [r"\bchatgpt\b", r"\bchat\s*gpt\b"],
    "llm_token": [r"\bllm\b", r"\bllms\b"],
    "prompt_engineering": [r"\bprompt\s*engineering\b"],
    "fine_tuning": [r"\bfine[\s-]*tuning\b", r"\bfinetuning\b"],
    "agile": [r"\bagile\b"],
    "scrum": [r"\bscrum\b"],
    "devops": [r"\bdevops\b"],
    "microservices": [r"\bmicroservices?\b", r"\bmicro[\s-]*services?\b"],
}
AI_TECH_COLS = [
    "langchain", "langgraph", "rag_tech", "vector_database", "pinecone",
    "huggingface", "openai", "claude_tech", "gemini_tech",
    "llamaindex", "anthropic", "ollama",
    "copilot", "cursor", "chatgpt", "llm_token", "prompt_engineering",
    "fine_tuning",
    "pytorch", "tensorflow", "sklearn", "keras", "xgboost",
]

TECH_COMPILED = {
    col: [re.compile(p, re.IGNORECASE) for p in pats]
    for col, pats in TAXONOMY.items()
}


def count_tech_in_text(text: str) -> tuple[int, int]:
    """Return (total tech count, ai-tech count)."""
    if not text:
        return 0, 0
    total = 0
    ai = 0
    for col, pats in TECH_COMPILED.items():
        if any(p.search(text) for p in pats):
            total += 1
            if col in AI_TECH_COLS:
                ai += 1
    return total, ai


def count_distinct(text: str, pats: dict) -> int:
    if not text:
        return 0
    return sum(1 for p in pats.values() if p.search(text))


def highest_education(text: str) -> int:
    if not text:
        return 0
    level = 0
    for lvl, pat in EDU_PATS.items():
        if pat.search(text):
            level = max(level, lvl)
    return level


def has_match(text: str, pat: re.Pattern) -> bool:
    return bool(text) and pat.search(text) is not None


# ---------------------------------------------------------------------------
# Inline asserts
# ---------------------------------------------------------------------------
def _inline_tests():
    assert AI_STRICT_PAT.search("we use copilot and cursor")
    assert AI_STRICT_PAT.search("RAG pipelines")
    assert not AI_STRICT_PAT.search("just SQL and microservices")
    assert AI_BROAD_EXTRA_PAT.search("machine learning models")
    assert AI_BROAD_EXTRA_PAT.search("LLMs")
    t, a = count_tech_in_text("we use python, react, aws, copilot")
    assert t >= 3 and a >= 1, (t, a)
    assert count_distinct("end-to-end stakeholder collaboration", SOFT_PATS) >= 1
    assert count_distinct("end-to-end stakeholder systems", ORG_SCOPE_PATS) >= 2
    assert highest_education("PhD in CS") == 3
    print("  inline tests passed")


def main():
    _inline_tests()

    con = duckdb.connect()
    con.execute("SET memory_limit='22GB'")

    sql = """
    SELECT
      uid,
      source,
      period,
      scrape_date,
      date_posted,
      is_swe,
      is_swe_adjacent,
      is_control,
      swe_classification_tier,
      is_aggregator,
      title,
      title_normalized,
      company_name_canonical,
      seniority_final,
      seniority_3level,
      seniority_native,
      yoe_extracted,
      llm_extraction_coverage,
      posting_age_days,
      description,
      description_core_llm
    FROM read_parquet('{uni}')
    WHERE source_platform='linkedin'
      AND is_english=true
      AND date_flag='ok'
      AND (is_swe=true OR is_swe_adjacent=true OR is_control=true)
    """.format(uni=UNI)

    print("Loading rows…")
    df = con.execute(sql).df()
    print(f"  {len(df):,} rows loaded")

    # Sanity
    for col in ["description"]:
        df[col] = df[col].fillna("")
    df["description_core_llm"] = df["description_core_llm"].fillna("")

    # Group label
    def pick_group(row):
        if row["is_swe"]:
            return "SWE"
        if row["is_swe_adjacent"]:
            return "adjacent"
        if row["is_control"]:
            return "control"
        return None
    df["group"] = df.apply(pick_group, axis=1)
    print(df["group"].value_counts())

    # Lowercase once for pattern scans
    desc_lc = df["description"].str.lower()

    # Ai-mention
    print("Computing AI-mention…")
    df["ai_strict"] = desc_lc.str.contains(AI_STRICT_PAT.pattern, regex=True, na=False)
    df["ai_broad"] = df["ai_strict"] | desc_lc.str.contains(
        AI_BROAD_EXTRA_PAT.pattern, regex=True, na=False
    )

    # Length
    df["desc_len_chars"] = df["description"].str.len()
    df["desc_len_cleaned"] = df["description_core_llm"].str.len()

    # Counts — vectorized via str.contains on each sub-pattern
    print("Computing org-scope / soft / mgmt counts…")
    def matrix_count(patterns):
        out = np.zeros(len(df), dtype=np.int16)
        for p in patterns.values():
            out = out + desc_lc.str.contains(p.pattern, regex=True, na=False).astype(np.int16)
        return out
    df["org_scope_count"] = matrix_count(ORG_SCOPE_PATS)
    df["soft_skill_count"] = matrix_count(SOFT_PATS)
    df["mgmt_strict_count"] = matrix_count(MGMT_STRICT_PATS)

    # Education — apply as vector ops
    print("Computing education level…")
    df["edu_level"] = 0
    for lvl, pat in EDU_PATS.items():
        hit = desc_lc.str.contains(pat.pattern, regex=True, na=False)
        df.loc[hit, "edu_level"] = df.loc[hit, "edu_level"].clip(lower=lvl)

    # Tech count — vectorized per tech
    print("Computing tech counts (this is the heavy step)…")
    tech_total = np.zeros(len(df), dtype=np.int16)
    tech_ai = np.zeros(len(df), dtype=np.int16)
    for col, pats in TECH_COMPILED.items():
        combined = "|".join(f"(?:{p.pattern})" for p in pats)
        hit = desc_lc.str.contains(combined, regex=True, na=False)
        tech_total += hit.astype(np.int16)
        if col in AI_TECH_COLS:
            tech_ai += hit.astype(np.int16)
    df["tech_count"] = tech_total
    df["ai_tech_count"] = tech_ai

    # Binary collapses
    df["ai_strict_binary"] = df["ai_strict"].astype(int)
    df["ai_broad_binary"] = df["ai_broad"].astype(int)
    df["has_ai_tech"] = (df["ai_tech_count"] > 0).astype(int)

    # requirement_breadth — same definition as T11 (tech + scope + soft + mgmt_strict + ai_binary)
    df["requirement_breadth"] = (
        df["tech_count"].astype(int)
        + df["org_scope_count"].astype(int)
        + df["soft_skill_count"].astype(int)
        + df["mgmt_strict_count"].astype(int)
        + df["ai_strict_binary"].astype(int)
    )

    print(df[["group", "ai_strict", "ai_broad", "tech_count", "org_scope_count"]].groupby("group").mean())

    # Save feature table — only what we need downstream
    out_cols = [
        "uid", "source", "period", "group", "scrape_date", "date_posted",
        "title", "title_normalized", "company_name_canonical", "is_aggregator",
        "swe_classification_tier",
        "seniority_final", "seniority_3level", "seniority_native",
        "yoe_extracted", "llm_extraction_coverage", "posting_age_days",
        "ai_strict", "ai_broad",
        "ai_strict_binary", "ai_broad_binary", "has_ai_tech",
        "desc_len_chars", "desc_len_cleaned",
        "org_scope_count", "soft_skill_count", "mgmt_strict_count",
        "edu_level", "tech_count", "ai_tech_count", "requirement_breadth",
    ]
    feat = df[out_cols].copy()
    out_path = ART / "T18_posting_features.parquet"
    pq.write_table(pa.Table.from_pandas(feat), out_path, compression="snappy")
    print(f"Wrote features → {out_path}")


if __name__ == "__main__":
    main()
