#!/usr/bin/env python3
"""T18 cross-occupation boundary analysis.

Memory posture:
- scans only default LinkedIn rows in Arrow batches
- writes a skinny feature parquet under the assigned T18 table directory
- uses DuckDB with 4GB / one-thread limits for grouped outputs
- caps TF-IDF boundary samples at 200 SWE and 200 adjacent rows per period
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]
UNIFIED = ROOT / "data" / "unified.parquet"
SHARED = ROOT / "exploration" / "artifacts" / "shared"
TABLE_DIR = ROOT / "exploration" / "tables" / "T18"
FIG_DIR = ROOT / "exploration" / "figures" / "T18"
TMP_DIR = ROOT / "exploration" / "artifacts" / "tmp_duckdb"
FEATURES = TABLE_DIR / "cross_occupation_features.parquet"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=1")
    con.execute(f"PRAGMA temp_directory='{TMP_DIR.as_posix()}'")
    return con


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def stable_rank(value: object) -> int:
    return int(hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12], 16)


def compile_patterns(raw: dict[str, str]) -> dict[str, re.Pattern[str]]:
    return {name: re.compile(pattern, re.I) for name, pattern in raw.items()}


SOFT_SKILL_PATTERNS = compile_patterns(
    {
        "communication": r"\b(?:written and verbal communication|verbal and written communication|communication skills?|communicate (?:effectively|clearly)|strong communicator)\b",
        "collaboration": r"\b(?:collaborat(?:e|es|ing|ion)|partner(?:ing)? with|work(?:ing)? closely with)\b",
        "problem_solving": r"\b(?:problem[- ]solving|solve complex problems|analytical problem)\b",
        "teamwork": r"\b(?:teamwork|team player)\b",
        "interpersonal": r"\binterpersonal skills?\b",
        "presentation": r"\bpresentation skills?\b",
        "attention_detail": r"\battention to detail\b",
        "adaptability": r"\b(?:adaptability|adaptable|fast[- ]paced environment)\b",
    }
)

ORG_SCOPE_PATTERNS = compile_patterns(
    {
        "ownership": r"\b(?:take ownership|ownership (?:of|for)|own(?:s|ing)? (?:the |a |an )?(?:feature|features|service|services|system|systems|platform|roadmap|delivery|architecture|codebase|product))\b",
        "end_to_end": r"\bend[- ]to[- ]end\b",
        "cross_functional": r"\bcross[- ]functional\b",
        "stakeholder_scope": r"\bstakeholders?\b",
        "autonomy": r"\b(?:independently|autonomously|self[- ]starter|minimal supervision)\b",
        "initiative": r"\b(?:take initiative|drive initiatives?|proactively|proactive)\b",
        "roadmap_strategy": r"\b(?:roadmap|technical strategy|strategic initiatives?)\b",
        "business_impact": r"\b(?:business impact|customer impact|product impact)\b",
    }
)

MANAGEMENT_STRONG_PATTERNS = compile_patterns(
    {
        "manage_team": r"\b(?:(?:manage|managing|managed|management of)\s+(?:a\s+|the\s+)?(?:team|teams|engineers|developers|people|staff)|team management)\b",
        "mentor": r"\b(?:mentor|mentoring|mentorship)\b",
        "coach": r"\b(?:coach|coaching)\b",
        "hiring_interviewing": r"\b(?:hiring\s+(?:engineers|developers|team|talent|candidates|process|bar|manager)|interview(?:ing)? candidates|recruit(?:ing)?\s+(?:engineers|developers|talent|candidates)|talent acquisition)\b",
        "direct_reports": r"\bdirect reports?\b",
        "performance_review": r"\bperformance reviews?\b",
        "headcount": r"\bheadcount\b",
        "people_leadership": r"\bpeople (?:management|manager|leadership)\b",
    }
)

EDUCATION_PATTERNS = {
    "phd": re.compile(r"\b(?:ph\.?\s?d\.?|doctorate|doctoral)\b", re.I),
    "masters": re.compile(r"\b(?:master'?s|m\.?\s?s\.?|m\.?\s?sc\.?|ms degree)\b", re.I),
    "bachelors": re.compile(r"\b(?:bachelor'?s|b\.?\s?s\.?|b\.?\s?a\.?|bs/ms|ba/bs)\b", re.I),
    "degree_unspecified": re.compile(
        r"\b(?:degree in computer science|computer science degree|degree or equivalent|equivalent experience)\b",
        re.I,
    ),
}

SENIOR_TITLE_RE = re.compile(r"\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b", re.I)


def assert_patterns() -> None:
    assert ORG_SCOPE_PATTERNS["ownership"].search("Take ownership of the platform roadmap.")
    assert not ORG_SCOPE_PATTERNS["ownership"].search("employee owned company")
    assert MANAGEMENT_STRONG_PATTERNS["manage_team"].search("Manage a team of engineers.")
    assert not MANAGEMENT_STRONG_PATTERNS["manage_team"].search("manage memory allocation")
    assert EDUCATION_PATTERNS["bachelors"].search("Bachelor's degree")


NON_AI_TECH_PATTERNS = compile_patterns(
    {
        "python": r"\bpython\b",
        "java": r"\bjava\b",
        "javascript": r"\b(?:javascript|node\.?js|nodejs)\b",
        "typescript": r"\btypescript\b",
        "go": r"\bgolang\b",
        "rust": r"\brust\b",
        "cpp": r"\bc\+\+\b",
        "csharp_dotnet": r"\b(?:c#|c sharp|\.net|dotnet)\b",
        "sql": r"\bsql\b",
        "react": r"\breact\b",
        "angular": r"\bangular\b",
        "api": r"\b(?:api|apis|rest|graphql|grpc)\b",
        "microservices": r"\bmicroservices?\b",
        "aws": r"\b(?:aws|amazon web services)\b",
        "azure": r"\bazure\b",
        "gcp": r"\b(?:gcp|google cloud)\b",
        "kubernetes": r"\b(?:kubernetes|k8s)\b",
        "docker": r"\bdocker\b",
        "terraform": r"\bterraform\b",
        "ci_cd": r"\b(?:ci/cd|continuous integration|continuous delivery|continuous deployment)\b",
        "git": r"\b(?:git|github|gitlab)\b",
        "linux": r"\blinux\b",
        "kafka": r"\bkafka\b",
        "spark": r"\bspark\b",
        "snowflake": r"\bsnowflake\b",
        "databricks": r"\bdatabricks\b",
        "mongodb": r"\bmongodb\b",
        "postgresql": r"\b(?:postgres|postgresql)\b",
        "redis": r"\bredis\b",
        "airflow": r"\bairflow\b",
        "observability": r"\b(?:observability|prometheus|grafana|splunk|monitoring)\b",
        "tableau_powerbi": r"\b(?:tableau|power bi|powerbi)\b",
        "testing": r"\b(?:unit testing|integration testing|selenium|cypress|playwright|pytest|junit|jest)\b",
        "agile": r"\b(?:agile|scrum|kanban)\b",
        "devops_sre": r"\b(?:devops|sre|site reliability)\b",
        "security": r"\b(?:security|oauth|cybersecurity|infosec)\b",
    }
)

AI_TOOL_PATTERNS = compile_patterns(
    {
        "llm": r"\b(?:llms|large language models?|llm/text|llm[- ](?:based|powered|enabled|applications?|systems?|solutions?|products?|integration|integrations|evaluation|evals|models?)|(?:with|using|building|integrating|exposure to|familiarity with)\s+llm)\b",
        "generative_ai": r"\b(?:generative ai|genai|gen ai)\b",
        "coding_assistant": r"\b(?:github copilot|copilot|cursor(?: ai)?|codex)\b",
        "chat_model": r"\b(?:chatgpt|chat gpt|anthropic claude|claude (?:code|ai|api|sonnet|opus)|google gemini|gemini (?:model|api|ai))\b",
        "model_api": r"\b(?:openai api|openai apis|openai sdk|openai platform|anthropic api|anthropic apis|anthropic sdk|claude api)\b",
        "ai_agent": r"\b(?:ai agents?|coding agents?|agentic)\b",
        "prompt_engineering": r"\b(?:prompt engineering|prompt engineer|prompt design|prompt tuning)\b",
        "fine_tuning": r"\b(?:(?:model|llm|ai|ml|machine learning).{0,40}(?:fine[- ]tuning|finetuning|fine tune|fine-tune|fine tuned|fine-tuned)|(?:fine[- ]tuning|finetuning|fine tune|fine-tune|fine tuned|fine-tuned).{0,40}(?:model|llm|ai|ml|machine learning))\b",
        "evals": r"\b(?:evals|model evaluation|llm evaluation|ai evaluation)\b",
        "rag": r"\b(?:rag|retrieval augmented generation|retrieval-augmented generation)\b",
        "llm_framework": r"\b(?:langchain|llamaindex|llama index)\b",
        "vector_database": r"\b(?:vector databases?|vector dbs?|vector stores?|semantic search|pinecone|weaviate|chromadb|chroma vector)\b",
    }
)

AI_DOMAIN_PATTERNS = compile_patterns(
    {
        "machine_learning": r"\bmachine learning\b",
        "deep_learning": r"\bdeep learning\b",
        "nlp": r"\b(?:nlp|natural language processing)\b",
        "computer_vision": r"\bcomputer vision\b",
        "mlops": r"\bmlops\b",
        "pytorch_tensorflow": r"\b(?:pytorch|tensorflow)\b",
        "hugging_face": r"\b(?:hugging face|huggingface|transformers library)\b",
        "ai_ml": r"\b(?:ai/ml|artificial intelligence|ai model|ai models|ml model|ml models)\b",
    }
)

MCP_PATTERN = re.compile(r"\b(?:mcp|model context protocol)\b", re.I)
ALL_CONTEXT_PATTERNS = {**NON_AI_TECH_PATTERNS, **AI_TOOL_PATTERNS, **AI_DOMAIN_PATTERNS, "mcp": MCP_PATTERN}


def count_pattern_hits(text: str, patterns: dict[str, re.Pattern[str]]) -> tuple[int, dict[str, bool]]:
    flags = {name: bool(pattern.search(text)) for name, pattern in patterns.items()}
    return sum(flags.values()), flags


def education_any(text: str) -> bool:
    return any(pattern.search(text) for pattern in EDUCATION_PATTERNS.values())


def scan_tech(text: str) -> dict[str, object]:
    non_ai_matches = {name for name, pattern in NON_AI_TECH_PATTERNS.items() if pattern.search(text)}
    tool_matches = {name for name, pattern in AI_TOOL_PATTERNS.items() if pattern.search(text)}
    domain_matches = {name for name, pattern in AI_DOMAIN_PATTERNS.items() if pattern.search(text)}
    matched_cols = non_ai_matches | tool_matches | domain_matches
    first_tool: str | None = None
    first_domain: str | None = None
    for col in AI_TOOL_PATTERNS:
        if col in tool_matches:
            first_tool = col
            break
    for col in AI_DOMAIN_PATTERNS:
        if col in domain_matches:
            first_domain = col
            break
    ai_tool_count = len(tool_matches)
    ai_domain_count = len(domain_matches)
    ai_count = ai_tool_count + ai_domain_count
    return {
        "tech_count": len(matched_cols),
        "ai_tool_count": ai_tool_count,
        "ai_domain_count": ai_domain_count,
        "ai_count": ai_count,
        "tech_count_non_ai": len(matched_cols) - ai_count,
        "ai_tool_any": first_tool is not None,
        "ai_domain_any": first_domain is not None,
        "ai_broad_any": first_tool is not None or first_domain is not None,
        "mcp_any": bool(MCP_PATTERN.search(text)),
        "first_tool": first_tool,
        "first_domain": first_domain,
    }


def context_for_col(text: str, pattern: re.Pattern[str] | None) -> str:
    if not text or pattern is None:
        return ""
    match = pattern.search(text)
    if not match:
        return ""
    start = max(match.start() - 240, 0)
    end = min(match.end() + 240, len(text))
    return " ".join(text[start:end].split())[:650]


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def regex_int_expr(text_expr: str, pattern: re.Pattern[str]) -> str:
    return f"CASE WHEN regexp_matches({text_expr}, {sql_literal(pattern.pattern)}) THEN 1 ELSE 0 END"


def regex_sum_expr(text_expr: str, patterns: dict[str, re.Pattern[str]]) -> str:
    return " + ".join(regex_int_expr(text_expr, pattern) for pattern in patterns.values())


def regex_any_expr(text_expr: str, patterns: dict[str, re.Pattern[str]]) -> str:
    return " OR ".join(f"regexp_matches({text_expr}, {sql_literal(pattern.pattern)})" for pattern in patterns.values())


def regex_first_case(text_expr: str, patterns: dict[str, re.Pattern[str]]) -> str:
    pieces = [
        f"WHEN regexp_matches({text_expr}, {sql_literal(pattern.pattern)}) THEN {sql_literal(name)}"
        for name, pattern in patterns.items()
    ]
    return "CASE " + " ".join(pieces) + " ELSE NULL END"


def occ_group(is_swe: object, is_adjacent: object, is_control: object) -> str:
    if bool(is_swe):
        return "swe"
    if bool(is_adjacent):
        return "swe_adjacent"
    if bool(is_control):
        return "control"
    return "other"


def source_group(source: str | None) -> str:
    if source == "kaggle_arshkon":
        return "arshkon"
    if source == "kaggle_asaniczka":
        return "asaniczka"
    if source == "scraped":
        return "scraped_2026"
    return source or "unknown"


def period_group(source: str | None) -> str:
    return "2026" if source == "scraped" else "2024"


def classify_adjacent_role(title: str | None) -> str:
    text = (title or "").lower()
    if re.search(r"\b(data engineer|analytics engineer|etl engineer|data platform)\b", text):
        return "data_engineer"
    if re.search(r"\b(data scientist|machine learning engineer|ml engineer|ai engineer|research scientist)\b", text):
        return "data_scientist_ml"
    if re.search(r"\b(devops|site reliability|sre|cloud engineer|platform engineer|infrastructure engineer)\b", text):
        return "devops_cloud_platform"
    if re.search(r"\b(network engineer|network administrator|network architect)\b", text):
        return "network_engineer"
    if re.search(r"\b(security engineer|cyber|information security|infosec|soc analyst)\b", text):
        return "security"
    if re.search(r"\b(qa engineer|quality assurance|test engineer|automation engineer)\b", text):
        return "qa_test"
    if re.search(
        r"\b(product manager|program manager|project manager|business analyst|systems analyst|solutions? engineer|sales engineer|support engineer|implementation consultant)\b",
        text,
    ):
        return "product_solutions_support"
    if re.search(r"\b(database administrator|dba|systems administrator|system administrator)\b", text):
        return "systems_database_admin"
    return "other_adjacent"


FEATURE_SCHEMA = pa.schema(
    [
        ("uid", pa.string()),
        ("source", pa.string()),
        ("source_group", pa.string()),
        ("period", pa.string()),
        ("period_group", pa.string()),
        ("occ_group", pa.string()),
        ("adjacent_role_family", pa.string()),
        ("title", pa.string()),
        ("company_name_canonical", pa.string()),
        ("seniority_final", pa.string()),
        ("seniority_native", pa.string()),
        ("yoe_extracted", pa.float64()),
        ("yoe_known", pa.bool_()),
        ("J1", pa.bool_()),
        ("J2", pa.bool_()),
        ("J3", pa.bool_()),
        ("J4", pa.bool_()),
        ("S1", pa.bool_()),
        ("S2", pa.bool_()),
        ("S3", pa.bool_()),
        ("S4", pa.bool_()),
        ("is_aggregator", pa.bool_()),
        ("swe_classification_tier", pa.string()),
        ("llm_labeled", pa.bool_()),
        ("raw_char_len", pa.int64()),
        ("raw_tech_count", pa.int64()),
        ("raw_tech_count_non_ai", pa.int64()),
        ("raw_ai_count", pa.int64()),
        ("raw_ai_broad_any", pa.bool_()),
        ("raw_ai_tool_any", pa.bool_()),
        ("raw_ai_domain_any", pa.bool_()),
        ("raw_mcp_any", pa.bool_()),
        ("clean_char_len", pa.int64()),
        ("clean_tech_count", pa.int64()),
        ("clean_tech_count_non_ai", pa.int64()),
        ("clean_ai_count", pa.int64()),
        ("clean_ai_broad_any", pa.bool_()),
        ("org_scope_count", pa.int64()),
        ("org_scope_any", pa.bool_()),
        ("soft_skill_count", pa.int64()),
        ("management_strong_count", pa.int64()),
        ("education_any", pa.bool_()),
        ("yoe_requirement_any", pa.bool_()),
        ("requirement_breadth", pa.int64()),
    ]
)


def scan_features() -> pd.DataFrame:
    con = connect()
    raw_text = "raw_text_l"
    clean_text = "clean_text_l"
    raw_non_ai = regex_sum_expr(raw_text, NON_AI_TECH_PATTERNS)
    raw_tool = regex_sum_expr(raw_text, AI_TOOL_PATTERNS)
    raw_domain = regex_sum_expr(raw_text, AI_DOMAIN_PATTERNS)
    clean_non_ai = regex_sum_expr(clean_text, NON_AI_TECH_PATTERNS)
    clean_tool = regex_sum_expr(clean_text, AI_TOOL_PATTERNS)
    clean_domain = regex_sum_expr(clean_text, AI_DOMAIN_PATTERNS)
    clean_soft = regex_sum_expr(clean_text, SOFT_SKILL_PATTERNS)
    clean_scope = regex_sum_expr(clean_text, ORG_SCOPE_PATTERNS)
    clean_mgmt = regex_sum_expr(clean_text, MANAGEMENT_STRONG_PATTERNS)
    clean_edu_any = regex_any_expr(clean_text, EDUCATION_PATTERNS)
    raw_tool_any = regex_any_expr(raw_text, AI_TOOL_PATTERNS)
    raw_domain_any = regex_any_expr(raw_text, AI_DOMAIN_PATTERNS)
    clean_tool_any = regex_any_expr(clean_text, AI_TOOL_PATTERNS)
    clean_domain_any = regex_any_expr(clean_text, AI_DOMAIN_PATTERNS)

    feature_sql = f"""
        COPY (
            WITH base AS (
                SELECT
                    uid,
                    source,
                    CAST(period AS VARCHAR) AS period,
                    title,
                    company_name_canonical,
                    coalesce(seniority_final, 'unknown') AS seniority_final,
                    seniority_native,
                    yoe_extracted,
                    coalesce(is_aggregator, false) AS is_aggregator,
                    swe_classification_tier,
                    is_swe,
                    is_swe_adjacent,
                    is_control,
                    llm_extraction_coverage = 'labeled' AS llm_labeled,
                    coalesce(description_length, length(coalesce(description, ''))) AS raw_char_len,
                    lower(coalesce(description, '')) AS raw_text_l,
                    CASE
                        WHEN llm_extraction_coverage = 'labeled'
                        THEN lower(coalesce(description_core_llm, ''))
                        ELSE ''
                    END AS clean_text_l
                FROM read_parquet('{sql_path(UNIFIED)}')
                WHERE source_platform = 'linkedin'
                  AND is_english = true
                  AND date_flag = 'ok'
                  AND (is_swe = true OR is_swe_adjacent = true OR is_control = true)
            ),
            scored AS (
                SELECT
                    *,
                    {raw_non_ai} AS raw_non_ai_count,
                    {raw_tool} AS raw_tool_count,
                    {raw_domain} AS raw_domain_count,
                    {clean_non_ai} AS clean_non_ai_count,
                    {clean_tool} AS clean_tool_count,
                    {clean_domain} AS clean_domain_count,
                    {clean_soft} AS clean_soft_count,
                    {clean_scope} AS clean_scope_count,
                    {clean_mgmt} AS clean_mgmt_count,
                    {clean_edu_any} AS clean_edu_any,
                    {raw_tool_any} AS raw_tool_any,
                    {raw_domain_any} AS raw_domain_any,
                    regexp_matches({raw_text}, {sql_literal(MCP_PATTERN.pattern)}) AS raw_mcp_any,
                    {clean_tool_any} AS clean_tool_any,
                    {clean_domain_any} AS clean_domain_any
                FROM base
            )
            SELECT
                uid,
                source,
                CASE
                    WHEN source = 'kaggle_arshkon' THEN 'arshkon'
                    WHEN source = 'kaggle_asaniczka' THEN 'asaniczka'
                    WHEN source = 'scraped' THEN 'scraped_2026'
                    ELSE coalesce(source, 'unknown')
                END AS source_group,
                period,
                CASE WHEN source = 'scraped' THEN '2026' ELSE '2024' END AS period_group,
                CASE
                    WHEN is_swe THEN 'swe'
                    WHEN is_swe_adjacent THEN 'swe_adjacent'
                    WHEN is_control THEN 'control'
                    ELSE 'other'
                END AS occ_group,
                CASE
                    WHEN NOT is_swe_adjacent THEN NULL
                    WHEN regexp_matches(lower(coalesce(title,'')), '\\b(data engineer|analytics engineer|etl engineer|data platform)\\b') THEN 'data_engineer'
                    WHEN regexp_matches(lower(coalesce(title,'')), '\\b(data scientist|machine learning engineer|ml engineer|ai engineer|research scientist)\\b') THEN 'data_scientist_ml'
                    WHEN regexp_matches(lower(coalesce(title,'')), '\\b(devops|site reliability|sre|cloud engineer|platform engineer|infrastructure engineer)\\b') THEN 'devops_cloud_platform'
                    WHEN regexp_matches(lower(coalesce(title,'')), '\\b(network engineer|network administrator|network architect)\\b') THEN 'network_engineer'
                    WHEN regexp_matches(lower(coalesce(title,'')), '\\b(security engineer|cyber|information security|infosec|soc analyst)\\b') THEN 'security'
                    WHEN regexp_matches(lower(coalesce(title,'')), '\\b(qa engineer|quality assurance|test engineer|automation engineer)\\b') THEN 'qa_test'
                    WHEN regexp_matches(lower(coalesce(title,'')), '\\b(product manager|program manager|project manager|business analyst|systems analyst|solutions? engineer|sales engineer|support engineer|implementation consultant)\\b') THEN 'product_solutions_support'
                    WHEN regexp_matches(lower(coalesce(title,'')), '\\b(database administrator|dba|systems administrator|system administrator)\\b') THEN 'systems_database_admin'
                    ELSE 'other_adjacent'
                END AS adjacent_role_family,
                title,
                company_name_canonical,
                seniority_final,
                seniority_native,
                yoe_extracted,
                yoe_extracted IS NOT NULL AS yoe_known,
                seniority_final = 'entry' AS J1,
                seniority_final IN ('entry', 'associate') AS J2,
                yoe_extracted <= 2 AS J3,
                yoe_extracted <= 3 AS J4,
                seniority_final IN ('mid-senior', 'director') AS S1,
                seniority_final = 'director' AS S2,
                regexp_matches(lower(coalesce(title,'')), '\\b(senior|sr\\.?|staff|principal|lead|architect|distinguished)\\b') AS S3,
                yoe_extracted >= 5 AS S4,
                is_aggregator,
                swe_classification_tier,
                llm_labeled,
                raw_char_len,
                raw_non_ai_count + raw_tool_count + raw_domain_count AS raw_tech_count,
                raw_non_ai_count AS raw_tech_count_non_ai,
                raw_tool_count + raw_domain_count AS raw_ai_count,
                raw_tool_any OR raw_domain_any AS raw_ai_broad_any,
                raw_tool_any AS raw_ai_tool_any,
                raw_domain_any AS raw_ai_domain_any,
                raw_mcp_any,
                CASE WHEN llm_labeled THEN length(clean_text_l) ELSE NULL END AS clean_char_len,
                CASE WHEN llm_labeled THEN clean_non_ai_count + clean_tool_count + clean_domain_count ELSE NULL END AS clean_tech_count,
                CASE WHEN llm_labeled THEN clean_non_ai_count ELSE NULL END AS clean_tech_count_non_ai,
                CASE WHEN llm_labeled THEN clean_tool_count + clean_domain_count ELSE NULL END AS clean_ai_count,
                CASE WHEN llm_labeled THEN clean_tool_any OR clean_domain_any ELSE NULL END AS clean_ai_broad_any,
                CASE WHEN llm_labeled THEN clean_scope_count ELSE NULL END AS org_scope_count,
                CASE WHEN llm_labeled THEN clean_scope_count > 0 ELSE NULL END AS org_scope_any,
                CASE WHEN llm_labeled THEN clean_soft_count ELSE NULL END AS soft_skill_count,
                CASE WHEN llm_labeled THEN clean_mgmt_count ELSE NULL END AS management_strong_count,
                CASE WHEN llm_labeled THEN clean_edu_any ELSE NULL END AS education_any,
                CASE WHEN llm_labeled THEN yoe_extracted IS NOT NULL ELSE NULL END AS yoe_requirement_any,
                CASE
                    WHEN llm_labeled THEN
                        clean_non_ai_count + clean_tool_count + clean_domain_count
                        + clean_soft_count + clean_scope_count + clean_mgmt_count
                        + CASE WHEN clean_edu_any THEN 1 ELSE 0 END
                        + CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END
                    ELSE NULL
                END AS requirement_breadth
            FROM scored
        ) TO '{sql_path(FEATURES)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(feature_sql)
    con.close()
    return sample_validation_from_features()


def sample_validation_from_features() -> pd.DataFrame:
    con = connect()
    query = f"""
        WITH candidates AS (
            SELECT
                'ai_tool_primary_ex_mcp' AS family,
                period_group,
                occ_group,
                source,
                uid
            FROM read_parquet('{sql_path(FEATURES)}')
            WHERE raw_ai_tool_any
            QUALIFY row_number() OVER (
                PARTITION BY period_group
                ORDER BY md5(uid || 'tool')
            ) <= 25
            UNION ALL
            SELECT
                'ai_domain_primary_ex_generic_libs' AS family,
                period_group,
                occ_group,
                source,
                uid
            FROM read_parquet('{sql_path(FEATURES)}')
            WHERE raw_ai_domain_any
            QUALIFY row_number() OVER (
                PARTITION BY period_group
                ORDER BY md5(uid || 'domain')
            ) <= 25
        )
        SELECT c.*, u.description
        FROM candidates c
        JOIN read_parquet('{sql_path(UNIFIED)}') u USING (uid)
        ORDER BY family, period_group, uid
    """
    val = con.execute(query).fetchdf()
    con.close()
    if val.empty:
        return val
    matched = []
    contexts = []
    for row in val.to_dict("records"):
        text = row.get("description") or ""
        patterns = AI_TOOL_PATTERNS if row["family"] == "ai_tool_primary_ex_mcp" else AI_DOMAIN_PATTERNS
        indicator = None
        pattern = None
        for name, candidate in patterns.items():
            if candidate.search(text):
                indicator = name
                pattern = candidate
                break
        matched.append(indicator)
        contexts.append(context_for_col(text, pattern))
    val["matched_indicator"] = matched
    val["context"] = contexts
    val = val.drop(columns=["description"])
    val.to_csv(TABLE_DIR / "keyword_validation_samples.csv", index=False)
    return val


METRIC_COLUMNS = [
    ("broad_ai_share_raw", "Broad AI prevalence, raw description binary", "eligible rows"),
    ("ai_tool_share_raw", "AI-tool-specific prevalence, raw description binary", "eligible rows"),
    ("ai_domain_share_raw", "AI-domain prevalence, raw description binary", "eligible rows"),
    ("mcp_share_raw", "MCP prevalence, raw description binary, excluded from primary AI", "eligible rows"),
    ("raw_tech_count_mean", "Mean technology indicator count, raw description binary", "eligible rows"),
    ("raw_char_len_mean", "Mean raw description length", "eligible rows"),
    ("clean_char_len_mean_labeled", "Mean cleaned description length", "LLM-labeled rows"),
    ("org_scope_share_labeled", "Org-scope language prevalence", "LLM-labeled rows"),
    ("org_scope_count_mean_labeled", "Mean org-scope indicator count", "LLM-labeled rows"),
    ("requirement_breadth_mean_labeled", "Mean regex-derived requirement breadth", "LLM-labeled rows"),
    ("clean_tech_count_mean_labeled", "Mean technology indicator count, cleaned text", "LLM-labeled rows"),
    ("clean_ai_count_mean_labeled", "Mean AI indicator count, cleaned text", "LLM-labeled rows"),
]


def metric_select() -> str:
    return """
        count(*) AS eligible_n,
        sum(CASE WHEN llm_labeled THEN 1 ELSE 0 END) AS labeled_n,
        avg(CASE WHEN yoe_known THEN 1 ELSE 0 END) AS yoe_known_share,
        avg(CASE WHEN is_aggregator THEN 1 ELSE 0 END) AS aggregator_share,
        avg(CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END) AS seniority_entry_share_all,
        avg(CASE WHEN seniority_final = 'associate' THEN 1 ELSE 0 END) AS seniority_associate_share_all,
        avg(CASE WHEN seniority_final = 'mid-senior' THEN 1 ELSE 0 END) AS seniority_mid_senior_share_all,
        avg(CASE WHEN seniority_final = 'director' THEN 1 ELSE 0 END) AS seniority_director_share_all,
        avg(CASE WHEN seniority_final = 'unknown' THEN 1 ELSE 0 END) AS seniority_unknown_share_all,
        avg(CASE WHEN yoe_known THEN J3::INT ELSE NULL END) AS yoe_le2_share_known,
        avg(CASE WHEN yoe_known THEN J4::INT ELSE NULL END) AS yoe_le3_share_known,
        avg(CASE WHEN yoe_known THEN S4::INT ELSE NULL END) AS yoe_ge5_share_known,
        avg(raw_ai_broad_any::INT) AS broad_ai_share_raw,
        avg(raw_ai_tool_any::INT) AS ai_tool_share_raw,
        avg(raw_ai_domain_any::INT) AS ai_domain_share_raw,
        avg(raw_mcp_any::INT) AS mcp_share_raw,
        avg(raw_tech_count) AS raw_tech_count_mean,
        median(raw_tech_count) AS raw_tech_count_median,
        avg(raw_char_len) AS raw_char_len_mean,
        median(raw_char_len) AS raw_char_len_median,
        avg(CASE WHEN llm_labeled THEN clean_char_len ELSE NULL END) AS clean_char_len_mean_labeled,
        median(CASE WHEN llm_labeled THEN clean_char_len ELSE NULL END) AS clean_char_len_median_labeled,
        avg(CASE WHEN llm_labeled THEN org_scope_any::INT ELSE NULL END) AS org_scope_share_labeled,
        avg(CASE WHEN llm_labeled THEN org_scope_count ELSE NULL END) AS org_scope_count_mean_labeled,
        avg(CASE WHEN llm_labeled THEN requirement_breadth ELSE NULL END) AS requirement_breadth_mean_labeled,
        avg(CASE WHEN llm_labeled THEN clean_tech_count ELSE NULL END) AS clean_tech_count_mean_labeled,
        avg(CASE WHEN llm_labeled THEN clean_ai_count ELSE NULL END) AS clean_ai_count_mean_labeled
    """


def aggregate_period_metrics(con: duckdb.DuckDBPyConnection, spec: str, from_sql: str) -> pd.DataFrame:
    query = f"""
        SELECT
            '{spec}' AS spec,
            occ_group,
            period_group,
            {metric_select()}
        FROM {from_sql}
        GROUP BY occ_group, period_group
        ORDER BY spec, occ_group, period_group
    """
    return con.execute(query).fetchdf()


def aggregate_source_metrics(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    query = f"""
        SELECT
            occ_group,
            source_group,
            {metric_select()}
        FROM read_parquet('{sql_path(FEATURES)}')
        GROUP BY occ_group, source_group
        ORDER BY occ_group, source_group
    """
    return con.execute(query).fetchdf()


def compute_did(metrics: pd.DataFrame, spec: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    frame = metrics[metrics["spec"] == spec]
    for col, label, denominator in METRIC_COLUMNS:
        values: dict[str, dict[str, float]] = {}
        for occ in ["swe", "swe_adjacent", "control"]:
            part = frame[frame["occ_group"] == occ].set_index("period_group")
            if "2024" not in part.index or "2026" not in part.index:
                continue
            v2024 = float(part.loc["2024", col])
            v2026 = float(part.loc["2026", col])
            values[occ] = {"2024": v2024, "2026": v2026, "change": v2026 - v2024}
        if {"swe", "swe_adjacent", "control"} - set(values):
            continue
        swe_change = values["swe"]["change"]
        ctrl_change = values["control"]["change"]
        adj_change = values["swe_adjacent"]["change"]
        if np.sign(swe_change) == np.sign(ctrl_change) and abs(ctrl_change) >= 0.7 * abs(swe_change):
            specificity = "field_wide_or_control_parallel"
        elif np.sign(swe_change) == np.sign(ctrl_change) and abs(ctrl_change) > 0:
            specificity = "swe_amplified"
        else:
            specificity = "swe_specific_or_control_divergent"
        rows.append(
            {
                "spec": spec,
                "metric": col,
                "label": label,
                "denominator": denominator,
                "swe_2024": values["swe"]["2024"],
                "swe_2026": values["swe"]["2026"],
                "swe_change": swe_change,
                "adjacent_change": adj_change,
                "control_change": ctrl_change,
                "did_swe_minus_control": swe_change - ctrl_change,
                "did_adjacent_minus_control": adj_change - ctrl_change,
                "did_swe_minus_adjacent": swe_change - adj_change,
                "specificity_flag": specificity,
            }
        )
    return pd.DataFrame(rows)


DEFINITIONS = [
    ("J1", "junior", "seniority_final = entry", "eligible rows", "J1", "TRUE"),
    ("J2", "junior", "seniority_final in entry/associate", "eligible rows", "J2", "TRUE"),
    ("J3", "junior", "yoe_extracted <= 2", "YOE-known rows", "J3", "yoe_known"),
    ("J4", "junior", "yoe_extracted <= 3", "YOE-known rows", "J4", "yoe_known"),
    ("S1", "senior", "seniority_final in mid-senior/director", "eligible rows", "S1", "TRUE"),
    ("S2", "senior", "seniority_final = director", "eligible rows", "S2", "TRUE"),
    ("S3", "senior", "raw title senior regex", "eligible rows", "S3", "TRUE"),
    ("S4", "senior", "yoe_extracted >= 5", "YOE-known rows", "S4", "yoe_known"),
]


def seniority_panel(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for code, side, definition, denominator, flag, denom_cond in DEFINITIONS:
        query = f"""
            SELECT
                '{code}' AS definition_code,
                '{side}' AS side,
                '{definition}' AS definition,
                '{denominator}' AS denominator,
                occ_group,
                period_group,
                count(*) AS eligible_n,
                sum(CASE WHEN {denom_cond} THEN 1 ELSE 0 END) AS denominator_n,
                sum(CASE WHEN {denom_cond} AND {flag} THEN 1 ELSE 0 END) AS numerator_n,
                numerator_n::DOUBLE / NULLIF(denominator_n, 0) AS share
            FROM read_parquet('{sql_path(FEATURES)}')
            GROUP BY occ_group, period_group
        """
        frames.append(con.execute(query).fetchdf())
    panel = pd.concat(frames, ignore_index=True)
    rows = []
    for (occ, code), part in panel.groupby(["occ_group", "definition_code"]):
        p = part.set_index("period_group")
        if "2024" not in p.index or "2026" not in p.index:
            continue
        rows.append(
            {
                "occ_group": occ,
                "definition_code": code,
                "side": p.iloc[0]["side"],
                "definition": p.iloc[0]["definition"],
                "denominator": p.iloc[0]["denominator"],
                "share_2024": float(p.loc["2024", "share"]),
                "share_2026": float(p.loc["2026", "share"]),
                "change": float(p.loc["2026", "share"] - p.loc["2024", "share"]),
                "denominator_2024": int(p.loc["2024", "denominator_n"]),
                "denominator_2026": int(p.loc["2026", "denominator_n"]),
            }
        )
    changes = pd.DataFrame(rows)
    verdicts = []
    for occ in ["swe", "swe_adjacent", "control"]:
        for side, codes in [("junior", ["J1", "J2", "J3", "J4"]), ("senior", ["S1", "S2", "S3", "S4"])]:
            sub = changes[(changes["occ_group"] == occ) & (changes["definition_code"].isin(codes))]
            signs = np.sign(sub["change"].fillna(0).to_numpy())
            nonzero = signs[signs != 0]
            if len(nonzero) == 0:
                verdict = "flat_or_uninformative"
            elif np.all(nonzero > 0):
                verdict = "unanimous_up"
            elif np.all(nonzero < 0):
                verdict = "unanimous_down"
            else:
                verdict = "mixed"
            verdicts.append({"occ_group": occ, "side": side, "agreement_verdict": verdict})
    return panel.merge(pd.DataFrame(verdicts), on=["occ_group", "side"], how="left"), changes


def within_2024_calibration(source_metrics: pd.DataFrame, did: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label, denominator in METRIC_COLUMNS:
        vals: dict[str, dict[str, float]] = {}
        for occ in ["swe", "swe_adjacent", "control"]:
            part = source_metrics[source_metrics["occ_group"] == occ].set_index("source_group")
            if "arshkon" not in part.index or "asaniczka" not in part.index:
                continue
            vals[occ] = {
                "within_change": float(part.loc["asaniczka", col] - part.loc["arshkon", col]),
                "arshkon": float(part.loc["arshkon", col]),
                "asaniczka": float(part.loc["asaniczka", col]),
            }
        if {"swe", "swe_adjacent", "control"} - set(vals):
            continue
        cross = did[(did["spec"] == "base") & (did["metric"] == col)]
        cross_did = float(cross["did_swe_minus_control"].iloc[0]) if not cross.empty else np.nan
        within_did = vals["swe"]["within_change"] - vals["control"]["within_change"]
        ratio = np.nan if abs(within_did) < 1e-9 else cross_did / within_did
        rows.append(
            {
                "metric": col,
                "label": label,
                "denominator": denominator,
                "within_2024_swe_change_asaniczka_minus_arshkon": vals["swe"]["within_change"],
                "within_2024_adjacent_change": vals["swe_adjacent"]["within_change"],
                "within_2024_control_change": vals["control"]["within_change"],
                "within_2024_did_swe_minus_control": within_did,
                "cross_period_did_swe_minus_control": cross_did,
                "cross_to_within_did_ratio": ratio,
            }
        )
    return pd.DataFrame(rows)


def adjacent_role_outputs(con: duckdb.DuckDBPyConnection, period_metrics: pd.DataFrame) -> pd.DataFrame:
    query = f"""
        SELECT
            adjacent_role_family,
            period_group,
            {metric_select()}
        FROM read_parquet('{sql_path(FEATURES)}')
        WHERE occ_group = 'swe_adjacent'
        GROUP BY adjacent_role_family, period_group
        ORDER BY period_group, eligible_n DESC
    """
    roles = con.execute(query).fetchdf()
    swe_2026 = period_metrics[
        (period_metrics["spec"] == "base")
        & (period_metrics["occ_group"] == "swe")
        & (period_metrics["period_group"] == "2026")
    ].iloc[0]
    for col, _, _ in METRIC_COLUMNS:
        roles[f"{col}_gap_vs_swe_2026"] = roles[col] - float(swe_2026[col])
    return roles


def ai_gradient(period_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        ("broad_ai_share_raw", "broad_ai_primary_ex_mcp"),
        ("ai_tool_share_raw", "ai_tool_primary_ex_mcp"),
        ("ai_domain_share_raw", "ai_domain_primary_ex_generic_libs"),
    ]
    base = period_metrics[period_metrics["spec"] == "base"]
    for col, label in metrics:
        for period in ["2024", "2026"]:
            part = base[base["period_group"] == period].set_index("occ_group")
            rows.append(
                {
                    "metric": label,
                    "period_group": period,
                    "swe": float(part.loc["swe", col]),
                    "swe_adjacent": float(part.loc["swe_adjacent", col]),
                    "control": float(part.loc["control", col]),
                    "swe_minus_control": float(part.loc["swe", col] - part.loc["control", col]),
                    "adjacent_minus_control": float(part.loc["swe_adjacent", col] - part.loc["control", col]),
                    "swe_minus_adjacent": float(part.loc["swe", col] - part.loc["swe_adjacent", col]),
                }
            )
    out = pd.DataFrame(rows)
    changes = []
    for metric, part in out.groupby("metric"):
        p = part.set_index("period_group")
        changes.append(
            {
                "metric": metric,
                "gradient_change_swe_minus_control": float(
                    p.loc["2026", "swe_minus_control"] - p.loc["2024", "swe_minus_control"]
                ),
                "gradient_change_adjacent_minus_control": float(
                    p.loc["2026", "adjacent_minus_control"] - p.loc["2024", "adjacent_minus_control"]
                ),
                "gradient_change_swe_minus_adjacent": float(
                    p.loc["2026", "swe_minus_adjacent"] - p.loc["2024", "swe_minus_adjacent"]
                ),
            }
        )
    return out.merge(pd.DataFrame(changes), on="metric", how="left")


def tfidf_boundary(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query = f"""
        WITH base AS (
            SELECT
                uid,
                CASE WHEN is_swe THEN 'swe' ELSE 'swe_adjacent' END AS occ_group,
                CASE WHEN source = 'scraped' THEN '2026' ELSE '2024' END AS period_group,
                description_core_llm AS text
            FROM read_parquet('{sql_path(UNIFIED)}')
            WHERE source_platform = 'linkedin'
              AND is_english = true
              AND date_flag = 'ok'
              AND llm_extraction_coverage = 'labeled'
              AND (is_swe = true OR is_swe_adjacent = true)
              AND length(coalesce(description_core_llm, '')) >= 120
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY occ_group, period_group
                    ORDER BY md5(uid)
                ) AS rn
            FROM base
        )
        SELECT uid, occ_group, period_group, text
        FROM ranked
        WHERE rn <= 200
        ORDER BY period_group, occ_group, rn
    """
    sample = con.execute(query).fetchdf()
    sample.to_csv(TABLE_DIR / "boundary_tfidf_sample_index.csv", index=False)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=3,
        max_df=0.85,
        max_features=6000,
        ngram_range=(1, 2),
    )
    x = vectorizer.fit_transform(sample["text"].fillna("").tolist())
    sim_rows = []
    for period in ["2024", "2026"]:
        swe_mask = (sample["period_group"] == period) & (sample["occ_group"] == "swe")
        adj_mask = (sample["period_group"] == period) & (sample["occ_group"] == "swe_adjacent")
        xs = x[swe_mask.to_numpy()]
        xa = x[adj_mask.to_numpy()]
        swe_centroid = np.asarray(xs.mean(axis=0))
        adj_centroid = np.asarray(xa.mean(axis=0))
        sim_rows.append(
            {
                "period_group": period,
                "swe_n": int(swe_mask.sum()),
                "adjacent_n": int(adj_mask.sum()),
                "centroid_cosine_similarity": float(cosine_similarity(swe_centroid, adj_centroid)[0, 0]),
                "mean_pairwise_cosine_similarity": float(cosine_similarity(xs, xa).mean()),
            }
        )
    sim = pd.DataFrame(sim_rows)
    sim["centroid_similarity_change_2026_minus_2024"] = (
        float(sim.loc[sim["period_group"] == "2026", "centroid_cosine_similarity"].iloc[0])
        - float(sim.loc[sim["period_group"] == "2024", "centroid_cosine_similarity"].iloc[0])
    )
    sim["pairwise_similarity_change_2026_minus_2024"] = (
        float(sim.loc[sim["period_group"] == "2026", "mean_pairwise_cosine_similarity"].iloc[0])
        - float(sim.loc[sim["period_group"] == "2024", "mean_pairwise_cosine_similarity"].iloc[0])
    )
    terms = np.array(vectorizer.get_feature_names_out())
    means: dict[tuple[str, str], np.ndarray] = {}
    for period in ["2024", "2026"]:
        for occ in ["swe", "swe_adjacent"]:
            mask = ((sample["period_group"] == period) & (sample["occ_group"] == occ)).to_numpy()
            means[(period, occ)] = np.asarray(x[mask].mean(axis=0)).ravel()
    gap_2024 = means[("2024", "swe")] - means[("2024", "swe_adjacent")]
    gap_2026 = means[("2026", "swe")] - means[("2026", "swe_adjacent")]
    adj_delta = means[("2026", "swe_adjacent")] - means[("2024", "swe_adjacent")]
    swe_delta = means[("2026", "swe")] - means[("2024", "swe")]
    term_frame = pd.DataFrame(
        {
            "term": terms,
            "adjacent_delta_2026_minus_2024": adj_delta,
            "swe_delta_2026_minus_2024": swe_delta,
            "swe_minus_adjacent_gap_2024": gap_2024,
            "swe_minus_adjacent_gap_2026": gap_2026,
            "absolute_gap_narrowing": np.abs(gap_2024) - np.abs(gap_2026),
            "adjacent_2026_mean_tfidf": means[("2026", "swe_adjacent")],
            "swe_2026_mean_tfidf": means[("2026", "swe")],
        }
    )
    migrating = (
        term_frame[
            (term_frame["adjacent_delta_2026_minus_2024"] > 0)
            & (term_frame["absolute_gap_narrowing"] > 0)
        ]
        .assign(
            migration_score=lambda d: d["adjacent_delta_2026_minus_2024"] + d["absolute_gap_narrowing"]
        )
        .sort_values("migration_score", ascending=False)
        .head(80)
    )
    return sample.drop(columns=["text"]), sim, migrating


def write_figures(
    period_metrics: pd.DataFrame,
    did: pd.DataFrame,
    gradient: pd.DataFrame,
    boundary: pd.DataFrame,
    role_profiles: pd.DataFrame,
) -> None:
    base = period_metrics[period_metrics["spec"] == "base"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, label in [
        ("broad_ai_share_raw", "Broad AI"),
        ("ai_tool_share_raw", "AI tool"),
        ("ai_domain_share_raw", "AI domain"),
    ]:
        pivot = base.pivot(index="period_group", columns="occ_group", values=metric).loc[["2024", "2026"]]
        ax.plot(pivot.index, pivot["swe"], marker="o", label=f"SWE {label}")
        ax.plot(pivot.index, pivot["swe_adjacent"], marker="o", linestyle="--", label=f"Adjacent {label}")
        ax.plot(pivot.index, pivot["control"], marker="o", linestyle=":", label=f"Control {label}")
    ax.set_ylabel("Share of eligible rows")
    ax.set_title("AI Adoption Gradient By Occupation")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ai_gradient_by_occupation.png", dpi=180)
    plt.close(fig)

    core = did[
        (did["spec"] == "base")
        & did["metric"].isin(
            [
                "broad_ai_share_raw",
                "ai_tool_share_raw",
                "ai_domain_share_raw",
                "raw_tech_count_mean",
                "requirement_breadth_mean_labeled",
                "org_scope_share_labeled",
            ]
        )
    ].copy()
    core["short_metric"] = core["metric"].map(
        {
            "broad_ai_share_raw": "Broad AI",
            "ai_tool_share_raw": "AI tool",
            "ai_domain_share_raw": "AI domain",
            "raw_tech_count_mean": "Tech count",
            "requirement_breadth_mean_labeled": "Req breadth",
            "org_scope_share_labeled": "Org scope",
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(core["short_metric"], core["did_swe_minus_control"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("(SWE change) - (control change)")
    ax.set_title("Cross-Occupation Difference-In-Differences")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "did_core_metrics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(boundary["period_group"], boundary["centroid_cosine_similarity"], marker="o", label="Centroid")
    ax.plot(boundary["period_group"], boundary["mean_pairwise_cosine_similarity"], marker="o", label="Pairwise mean")
    ax.set_ylabel("TF-IDF cosine similarity")
    ax.set_title("SWE vs Adjacent Boundary Similarity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "boundary_tfidf_similarity.png", dpi=180)
    plt.close(fig)

    roles = role_profiles[role_profiles["period_group"] == "2026"].sort_values("eligible_n", ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(roles))
    ax.bar(x - 0.2, roles["ai_tool_share_raw"], width=0.4, label="AI tool share")
    ax.bar(x + 0.2, roles["ai_domain_share_raw"], width=0.4, label="AI domain share")
    ax.set_xticks(x)
    ax.set_xticklabels(roles["adjacent_role_family"], rotation=35, ha="right")
    ax.set_ylabel("Share of eligible rows")
    ax.set_title("2026 Adjacent Role AI Profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "adjacent_role_ai_profile_2026.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    assert_patterns()
    if FEATURES.exists():
        validation = sample_validation_from_features()
    else:
        validation = scan_features()
    validation_summary = (
        validation.groupby(["family", "period_group"], as_index=False)
        .agg(sampled_n=("uid", "count"), occupation_groups=("occ_group", lambda x: ",".join(sorted(set(x)))))
        if not validation.empty
        else pd.DataFrame()
    )
    validation_summary.to_csv(TABLE_DIR / "keyword_validation_sample_summary.csv", index=False)

    con = connect()
    base_from = f"read_parquet('{sql_path(FEATURES)}')"
    no_agg_from = f"(SELECT * FROM read_parquet('{sql_path(FEATURES)}') WHERE NOT is_aggregator)"
    no_title_lookup_from = (
        f"(SELECT * FROM read_parquet('{sql_path(FEATURES)}') "
        "WHERE occ_group <> 'swe' OR swe_classification_tier IS NULL OR swe_classification_tier <> 'title_lookup_llm')"
    )
    company_cap_from = f"""
        (
            SELECT * EXCLUDE(rn)
            FROM (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY occ_group, period_group, coalesce(company_name_canonical, '')
                        ORDER BY uid
                    ) AS rn
                FROM read_parquet('{sql_path(FEATURES)}')
            )
            WHERE rn <= 50
        )
    """
    metric_frames = [
        aggregate_period_metrics(con, "base", base_from),
        aggregate_period_metrics(con, "exclude_aggregators", no_agg_from),
        aggregate_period_metrics(con, "exclude_swe_title_lookup_llm", no_title_lookup_from),
        aggregate_period_metrics(con, "company_cap50", company_cap_from),
    ]
    period_metrics = pd.concat(metric_frames, ignore_index=True)
    period_metrics.to_csv(TABLE_DIR / "metric_by_occupation_period.csv", index=False)

    source_metrics = aggregate_source_metrics(con)
    source_metrics.to_csv(TABLE_DIR / "metric_by_occupation_source.csv", index=False)

    did = pd.concat([compute_did(period_metrics, spec) for spec in period_metrics["spec"].unique()], ignore_index=True)
    did.to_csv(TABLE_DIR / "difference_in_differences.csv", index=False)

    calibration = within_2024_calibration(source_metrics, did)
    calibration.to_csv(TABLE_DIR / "within_2024_calibration.csv", index=False)

    panel, panel_changes = seniority_panel(con)
    panel.to_csv(TABLE_DIR / "seniority_panel_by_occupation.csv", index=False)
    panel_changes.to_csv(TABLE_DIR / "seniority_panel_changes.csv", index=False)

    roles = adjacent_role_outputs(con, period_metrics)
    roles.to_csv(TABLE_DIR / "adjacent_role_profiles.csv", index=False)

    gradient = ai_gradient(period_metrics)
    gradient.to_csv(TABLE_DIR / "ai_adoption_gradient.csv", index=False)

    sample_idx, boundary, migrating = tfidf_boundary(con)
    sample_idx.to_csv(TABLE_DIR / "boundary_tfidf_sample_index_no_text.csv", index=False)
    boundary.to_csv(TABLE_DIR / "boundary_tfidf_similarity.csv", index=False)
    migrating.to_csv(TABLE_DIR / "boundary_migrating_terms.csv", index=False)

    write_figures(period_metrics, did, gradient, boundary, roles)

    run_summary = pd.DataFrame(
        [
            {
                "feature_rows": int(con.execute(f"SELECT count(*) FROM read_parquet('{sql_path(FEATURES)}')").fetchone()[0]),
                "validation_sample_rows": int(len(validation)),
                "tfidf_sample_rows": int(len(sample_idx)),
                "figure_count": 4,
            }
        ]
    )
    run_summary.to_csv(TABLE_DIR / "run_summary.csv", index=False)
    con.close()


if __name__ == "__main__":
    main()
