#!/usr/bin/env python3
"""Wave 1.5 shared preprocessing artifacts.

Builds shared exploration artifacts from the default LinkedIn SWE frame without
rerunning preprocessing or making LLM calls. Full-file access goes through
DuckDB with a 4GB memory limit and one thread; row-level artifacts are streamed
through Arrow batches.
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import re
import resource
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
OUT = ROOT / "exploration" / "artifacts" / "shared"

CLEANED_PATH = OUT / "swe_cleaned_text.parquet"
EMBEDDINGS_PATH = OUT / "swe_embeddings.npy"
EMBEDDING_INDEX_PATH = OUT / "swe_embedding_index.parquet"
EMBEDDING_CHECKPOINT_PATH = OUT / "swe_embedding_checkpoint.json"
TECH_MATRIX_PATH = OUT / "swe_tech_matrix.parquet"
COMPANY_STOPLIST_PATH = OUT / "company_stoplist.txt"
SKILLS_PATH = OUT / "asaniczka_structured_skills.parquet"
CALIBRATION_PATH = OUT / "calibration_table.csv"
TECH_SANITY_PATH = OUT / "tech_matrix_sanity.csv"
README_PATH = OUT / "README.md"
TAXONOMY_PATH = OUT / "tech_taxonomy.csv"
METADATA_PATH = OUT / "prep_build_metadata.json"

DEFAULT_WHERE = (
    "source_platform = 'linkedin' "
    "AND is_english = true "
    "AND date_flag = 'ok' "
    "AND is_swe = true"
)
SOURCE_ORDER = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]
SOURCE_LABEL = {
    "kaggle_arshkon": "arshkon",
    "kaggle_asaniczka": "asaniczka",
    "scraped": "scraped",
}
ARROW_BATCH_SIZE = 4096
EMBED_BATCH_SIZE = 64
EMBED_CHECKPOINT_EVERY = 512

# Keep technology tokens that collide with English stopwords or company names.
PROTECTED_TOKENS = {
    "ai",
    "api",
    "aws",
    "azure",
    "c",
    "c#",
    "c++",
    "ci/cd",
    "claude",
    "codex",
    "cursor",
    "dbt",
    "docker",
    "gcp",
    "gemini",
    "go",
    "java",
    "javascript",
    "js",
    "kubernetes",
    "llm",
    "mcp",
    ".net",
    "node.js",
    "openai",
    "python",
    "r",
    "rag",
    "react",
    "sql",
    "typescript",
}

COMPANY_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.&'/-]*")
TEXT_TOKEN_RE = re.compile(
    r"c\+\+|c#|\.net|[a-z0-9]+(?:[+#][a-z0-9]*)?(?:[./_-][a-z0-9+#]+)*",
    re.IGNORECASE,
)
MARKDOWN_ESCAPE_RE = re.compile(r"\\([+\-#.&_()\[\]\{\}!*\/])")
WHITESPACE_RE = re.compile(r"\s+")


TECH_PATTERNS: list[tuple[str, str, str, str]] = [
    ("python", "Python", r"(?<![a-z0-9+#.])python(?![a-z0-9+#.])", "language"),
    ("java", "Java", r"(?<![a-z0-9+#.])java(?![a-z0-9+#.])", "language"),
    ("javascript", "JavaScript", r"(?<![a-z0-9+#.])(?:javascript|java script|js)(?![a-z0-9+#.])", "language"),
    ("typescript", "TypeScript", r"(?<![a-z0-9+#.])(?:typescript|type script|ts)(?![a-z0-9+#.])", "language"),
    ("go", "Go", r"\b(?:golang|go lang|go programming|go language|go developer|go engineer|go services|go microservices|backend go)\b", "language"),
    ("rust", "Rust", r"(?<![a-z0-9+#.])rust(?![a-z0-9+#.])", "language"),
    ("c_language", "C", r"\b(?:c programming|programming in c|c/c\+\+|c/cpp|c and c\+\+|c, c\+\+)\b", "language"),
    ("cpp", "C++", r"(?<![a-z0-9])(?:c\+\+|cpp)(?![a-z0-9])", "language"),
    ("csharp", "C#", r"(?<![a-z0-9])(?:c#|c sharp|csharp)(?![a-z0-9])", "language"),
    ("ruby", "Ruby", r"(?<![a-z0-9+#.])ruby(?![a-z0-9+#.])", "language"),
    ("kotlin", "Kotlin", r"(?<![a-z0-9+#.])kotlin(?![a-z0-9+#.])", "language"),
    ("swift", "Swift", r"(?<![a-z0-9+#.])swift(?![a-z0-9+#.])", "language"),
    ("scala", "Scala", r"(?<![a-z0-9+#.])scala(?![a-z0-9+#.])", "language"),
    ("php", "PHP", r"(?<![a-z0-9+#.])php(?![a-z0-9+#.])", "language"),
    ("r_language", "R", r"\b(?:r language|r programming|programming in r|rstudio|r studio)\b", "language"),
    ("matlab", "MATLAB", r"(?<![a-z0-9+#.])matlab(?![a-z0-9+#.])", "language"),
    ("perl", "Perl", r"(?<![a-z0-9+#.])perl(?![a-z0-9+#.])", "language"),
    ("bash_shell", "Bash/Shell", r"\b(?:bash|shell scripting|unix shell|linux shell)\b", "language"),
    ("powershell", "PowerShell", r"(?<![a-z0-9+#.])powershell(?![a-z0-9+#.])", "language"),
    ("html", "HTML", r"(?<![a-z0-9+#.])html(?:5)?(?![a-z0-9+#.])", "language"),
    ("css", "CSS", r"(?<![a-z0-9+#.])css(?:3)?(?![a-z0-9+#.])", "language"),
    ("objective_c", "Objective-C", r"\bobjective[- ]c\b", "language"),
    ("dart", "Dart", r"(?<![a-z0-9+#.])dart(?![a-z0-9+#.])", "language"),
    ("elixir", "Elixir", r"(?<![a-z0-9+#.])elixir(?![a-z0-9+#.])", "language"),
    ("erlang", "Erlang", r"(?<![a-z0-9+#.])erlang(?![a-z0-9+#.])", "language"),
    ("react", "React", r"(?<![a-z0-9+#.])react(?:\.js|js)?(?![a-z0-9+#.])", "framework"),
    ("angular", "Angular", r"(?<![a-z0-9+#.])angular(?:js)?(?![a-z0-9+#.])", "framework"),
    ("vue", "Vue", r"(?<![a-z0-9+#.])vue(?:\.js|js)?(?![a-z0-9+#.])", "framework"),
    ("nextjs", "Next.js", r"(?<![a-z0-9+#.])next(?:\.js|js)?(?![a-z0-9+#.])", "framework"),
    ("nodejs", "Node.js", r"(?<![a-z0-9+#.])node(?:\.js|js)?(?![a-z0-9+#.])", "framework"),
    ("express", "Express", r"\bexpress(?:\.js|js)?\b", "framework"),
    ("django", "Django", r"(?<![a-z0-9+#.])django(?![a-z0-9+#.])", "framework"),
    ("flask", "Flask", r"(?<![a-z0-9+#.])flask(?![a-z0-9+#.])", "framework"),
    ("fastapi", "FastAPI", r"(?<![a-z0-9+#.])fastapi(?![a-z0-9+#.])", "framework"),
    ("spring", "Spring", r"\b(?:spring boot|spring framework|spring cloud|spring mvc)\b", "framework"),
    ("dotnet", ".NET", r"(?<![a-z0-9])(?:\.net|dotnet|dot net|asp\.net)(?![a-z0-9])", "framework"),
    ("rails", "Rails", r"\b(?:ruby on rails|rails)\b", "framework"),
    ("aspnet", "ASP.NET", r"\basp\.net\b", "framework"),
    ("graphql", "GraphQL", r"(?<![a-z0-9+#.])graphql(?![a-z0-9+#.])", "framework"),
    ("grpc", "gRPC", r"(?<![a-z0-9+#.])grpc(?![a-z0-9+#.])", "framework"),
    ("rest_api", "REST API", r"\b(?:restful api|rest api|rest apis|restful services?)\b", "framework"),
    ("microservices", "Microservices", r"\bmicro-?services?\b", "architecture"),
    ("serverless", "Serverless", r"(?<![a-z0-9+#.])serverless(?![a-z0-9+#.])", "architecture"),
    ("aws", "AWS", r"(?<![a-z0-9+#.])(?:aws|amazon web services)(?![a-z0-9+#.])", "cloud"),
    ("azure", "Azure", r"(?<![a-z0-9+#.])azure(?![a-z0-9+#.])", "cloud"),
    ("gcp", "GCP", r"(?<![a-z0-9+#.])(?:gcp|google cloud platform|google cloud)(?![a-z0-9+#.])", "cloud"),
    ("kubernetes", "Kubernetes", r"(?<![a-z0-9+#.])(?:kubernetes|k8s)(?![a-z0-9+#.])", "cloud"),
    ("docker", "Docker", r"(?<![a-z0-9+#.])docker(?![a-z0-9+#.])", "cloud"),
    ("terraform", "Terraform", r"(?<![a-z0-9+#.])terraform(?![a-z0-9+#.])", "cloud"),
    ("helm", "Helm", r"(?<![a-z0-9+#.])helm(?![a-z0-9+#.])", "cloud"),
    ("ansible", "Ansible", r"(?<![a-z0-9+#.])ansible(?![a-z0-9+#.])", "cloud"),
    ("ci_cd", "CI/CD", r"\b(?:ci\s*/\s*cd|cicd|continuous integration|continuous delivery|continuous deployment)\b", "devops"),
    ("jenkins", "Jenkins", r"(?<![a-z0-9+#.])jenkins(?![a-z0-9+#.])", "devops"),
    ("github_actions", "GitHub Actions", r"\bgithub actions?\b", "devops"),
    ("gitlab_ci", "GitLab CI", r"\bgitlab(?: ci| ci/cd| pipelines?)\b", "devops"),
    ("circleci", "CircleCI", r"(?<![a-z0-9+#.])circleci(?![a-z0-9+#.])", "devops"),
    ("buildkite", "Buildkite", r"(?<![a-z0-9+#.])buildkite(?![a-z0-9+#.])", "devops"),
    ("argo_cd", "Argo CD", r"\bargo(?: cd|cd)\b", "devops"),
    ("linux", "Linux", r"(?<![a-z0-9+#.])linux(?![a-z0-9+#.])", "ops"),
    ("unix", "Unix", r"(?<![a-z0-9+#.])unix(?![a-z0-9+#.])", "ops"),
    ("git", "Git", r"(?<![a-z0-9+#.])git(?![a-z0-9+#.])", "tooling"),
    ("github", "GitHub", r"(?<![a-z0-9+#.])github(?![a-z0-9+#.])", "tooling"),
    ("gitlab", "GitLab", r"(?<![a-z0-9+#.])gitlab(?![a-z0-9+#.])", "tooling"),
    ("bitbucket", "Bitbucket", r"(?<![a-z0-9+#.])bitbucket(?![a-z0-9+#.])", "tooling"),
    ("jira", "Jira", r"(?<![a-z0-9+#.])jira(?![a-z0-9+#.])", "tooling"),
    ("confluence", "Confluence", r"(?<![a-z0-9+#.])confluence(?![a-z0-9+#.])", "tooling"),
    ("sql", "SQL", r"(?<![a-z0-9+#.])sql(?![a-z0-9+#.])", "data"),
    ("postgresql", "PostgreSQL", r"\b(?:postgresql|postgres|postgre sql)\b", "data"),
    ("mysql", "MySQL", r"(?<![a-z0-9+#.])mysql(?![a-z0-9+#.])", "data"),
    ("mongodb", "MongoDB", r"\b(?:mongodb|mongo db)\b", "data"),
    ("redis", "Redis", r"(?<![a-z0-9+#.])redis(?![a-z0-9+#.])", "data"),
    ("kafka", "Kafka", r"(?<![a-z0-9+#.])kafka(?![a-z0-9+#.])", "data"),
    ("spark", "Spark", r"\b(?:apache spark|spark sql|pyspark)\b|(?<![a-z0-9+#.])spark(?![a-z0-9+#.])", "data"),
    ("snowflake", "Snowflake", r"(?<![a-z0-9+#.])snowflake(?![a-z0-9+#.])", "data"),
    ("databricks", "Databricks", r"(?<![a-z0-9+#.])databricks(?![a-z0-9+#.])", "data"),
    ("dbt", "dbt", r"(?<![a-z0-9+#.])dbt(?![a-z0-9+#.])", "data"),
    ("elasticsearch", "Elasticsearch", r"\b(?:elasticsearch|elastic search)\b", "data"),
    ("opensearch", "OpenSearch", r"(?<![a-z0-9+#.])opensearch(?![a-z0-9+#.])", "data"),
    ("oracle", "Oracle", r"(?<![a-z0-9+#.])oracle(?![a-z0-9+#.])", "data"),
    ("sql_server", "SQL Server", r"\b(?:sql server|mssql|ms sql)\b", "data"),
    ("cassandra", "Cassandra", r"(?<![a-z0-9+#.])cassandra(?![a-z0-9+#.])", "data"),
    ("dynamodb", "DynamoDB", r"\b(?:dynamodb|dynamo db)\b", "data"),
    ("bigquery", "BigQuery", r"(?<![a-z0-9+#.])bigquery(?![a-z0-9+#.])", "data"),
    ("redshift", "Redshift", r"(?<![a-z0-9+#.])redshift(?![a-z0-9+#.])", "data"),
    ("airflow", "Airflow", r"\b(?:apache airflow|airflow)\b", "data"),
    ("flink", "Flink", r"\b(?:apache flink|flink)\b", "data"),
    ("hadoop", "Hadoop", r"(?<![a-z0-9+#.])hadoop(?![a-z0-9+#.])", "data"),
    ("rabbitmq", "RabbitMQ", r"\b(?:rabbitmq|rabbit mq)\b", "data"),
    ("prometheus", "Prometheus", r"(?<![a-z0-9+#.])prometheus(?![a-z0-9+#.])", "ops"),
    ("grafana", "Grafana", r"(?<![a-z0-9+#.])grafana(?![a-z0-9+#.])", "ops"),
    ("splunk", "Splunk", r"(?<![a-z0-9+#.])splunk(?![a-z0-9+#.])", "ops"),
    ("tableau", "Tableau", r"(?<![a-z0-9+#.])tableau(?![a-z0-9+#.])", "data"),
    ("power_bi", "Power BI", r"\bpower bi\b", "data"),
    ("tensorflow", "TensorFlow", r"(?<![a-z0-9+#.])tensorflow(?![a-z0-9+#.])", "ai_ml"),
    ("pytorch", "PyTorch", r"(?<![a-z0-9+#.])pytorch(?![a-z0-9+#.])", "ai_ml"),
    ("scikit_learn", "scikit-learn", r"\b(?:scikit[- ]learn|sklearn)\b", "ai_ml"),
    ("pandas", "Pandas", r"(?<![a-z0-9+#.])pandas(?![a-z0-9+#.])", "ai_ml"),
    ("numpy", "NumPy", r"(?<![a-z0-9+#.])numpy(?![a-z0-9+#.])", "ai_ml"),
    ("scipy", "SciPy", r"(?<![a-z0-9+#.])scipy(?![a-z0-9+#.])", "ai_ml"),
    ("langchain", "LangChain", r"(?<![a-z0-9+#.])langchain(?![a-z0-9+#.])", "ai_ml"),
    ("llamaindex", "LlamaIndex", r"\b(?:llamaindex|llama index)\b", "ai_ml"),
    ("rag", "RAG", r"\b(?:rag|retrieval augmented generation|retrieval-augmented generation)\b", "ai_ml"),
    ("vector_databases", "Vector databases", r"\b(?:vector databases?|vector dbs?|vector stores?|semantic search)\b", "ai_ml"),
    ("pinecone", "Pinecone", r"(?<![a-z0-9+#.])pinecone(?![a-z0-9+#.])", "ai_ml"),
    ("weaviate", "Weaviate", r"(?<![a-z0-9+#.])weaviate(?![a-z0-9+#.])", "ai_ml"),
    ("chroma", "Chroma", r"\b(?:chroma db|chromadb|chroma vector)\b", "ai_ml"),
    ("hugging_face", "Hugging Face", r"\b(?:hugging face|huggingface|transformers library)\b", "ai_ml"),
    ("openai_api", "OpenAI API", r"\b(?:openai api|openai apis|openai sdk|openai platform)\b", "ai_tool"),
    ("claude_api", "Claude API", r"\b(?:claude api|claude apis|anthropic api|anthropic sdk)\b", "ai_tool"),
    ("anthropic_api", "Anthropic API", r"\b(?:anthropic api|anthropic apis|anthropic sdk)\b", "ai_tool"),
    ("prompt_engineering", "Prompt engineering", r"\b(?:prompt engineering|prompt engineer|prompt design|prompt tuning)\b", "ai_tool"),
    ("fine_tuning", "Fine-tuning", r"\b(?:fine[- ]tuning|finetuning|fine tune|fine-tune|fine tuned|fine-tuned)\b", "ai_tool"),
    ("mcp", "MCP", r"\b(?:mcp|model context protocol)\b", "ai_tool"),
    ("llm", "LLM", r"\b(?:llm|llms|large language model|large language models)\b", "ai_tool"),
    ("generative_ai", "Generative AI", r"\b(?:generative ai|genai|gen ai)\b", "ai_tool"),
    ("machine_learning", "Machine learning", r"\b(?:machine learning|ml model|ml models)\b", "ai_ml"),
    ("deep_learning", "Deep learning", r"\bdeep learning\b", "ai_ml"),
    ("nlp", "NLP", r"\b(?:nlp|natural language processing)\b", "ai_ml"),
    ("computer_vision", "Computer vision", r"\bcomputer vision\b", "ai_ml"),
    ("mlops", "MLOps", r"(?<![a-z0-9+#.])mlops(?![a-z0-9+#.])", "ai_ml"),
    ("copilot", "Copilot", r"\b(?:github copilot|copilot)\b", "ai_tool"),
    ("cursor", "Cursor", r"\bcursor(?: ai)?\b", "ai_tool"),
    ("chatgpt", "ChatGPT", r"\b(?:chatgpt|chat gpt)\b", "ai_tool"),
    ("claude", "Claude", r"(?<![a-z0-9+#.])claude(?![a-z0-9+#.])", "ai_tool"),
    ("gemini", "Gemini", r"(?<![a-z0-9+#.])gemini(?![a-z0-9+#.])", "ai_tool"),
    ("codex", "Codex", r"(?<![a-z0-9+#.])codex(?![a-z0-9+#.])", "ai_tool"),
    ("agents", "Agents", r"\b(?:ai agents?|coding agents?|agentic|agent-based)\b", "ai_tool"),
    ("evals", "Evals", r"\b(?:evals|model evaluation|llm evaluation|ai evaluation)\b", "ai_tool"),
    ("jest", "Jest", r"(?<![a-z0-9+#.])jest(?![a-z0-9+#.])", "testing"),
    ("pytest", "Pytest", r"(?<![a-z0-9+#.])pytest(?![a-z0-9+#.])", "testing"),
    ("junit", "JUnit", r"(?<![a-z0-9+#.])junit(?![a-z0-9+#.])", "testing"),
    ("selenium", "Selenium", r"(?<![a-z0-9+#.])selenium(?![a-z0-9+#.])", "testing"),
    ("cypress", "Cypress", r"(?<![a-z0-9+#.])cypress(?![a-z0-9+#.])", "testing"),
    ("playwright", "Playwright", r"(?<![a-z0-9+#.])playwright(?![a-z0-9+#.])", "testing"),
    ("mocha", "Mocha", r"(?<![a-z0-9+#.])mocha(?![a-z0-9+#.])", "testing"),
    ("unit_testing", "Unit testing", r"\bunit tests?|unit testing\b", "testing"),
    ("integration_testing", "Integration testing", r"\bintegration tests?|integration testing\b", "testing"),
    ("tdd", "TDD", r"\b(?:tdd|test driven development|test-driven development)\b", "practice"),
    ("bdd", "BDD", r"\b(?:bdd|behavior driven development|behaviour driven development)\b", "practice"),
    ("agile", "Agile", r"(?<![a-z0-9+#.])agile(?![a-z0-9+#.])", "practice"),
    ("scrum", "Scrum", r"(?<![a-z0-9+#.])scrum(?![a-z0-9+#.])", "practice"),
    ("kanban", "Kanban", r"(?<![a-z0-9+#.])kanban(?![a-z0-9+#.])", "practice"),
    ("devops", "DevOps", r"(?<![a-z0-9+#.])devops(?![a-z0-9+#.])", "practice"),
    ("sre", "SRE", r"\b(?:sre|site reliability engineering|site reliability engineer)\b", "practice"),
    ("observability", "Observability", r"\b(?:observability|monitoring and alerting|distributed tracing)\b", "practice"),
    ("security", "Security", r"\b(?:cybersecurity|application security|appsec|secure coding|security engineering)\b", "practice"),
    ("oauth", "OAuth", r"(?<![a-z0-9+#.])oauth(?:2)?(?![a-z0-9+#.])", "security"),
    ("api_design", "API design", r"\b(?:api design|api development|api architecture|apis)\b", "practice"),
]


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=1")
    return con


def q(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    s = html.unescape(str(text))
    s = MARKDOWN_ESCAPE_RE.sub(r"\1", s)
    s = s.replace("\u00a0", " ")
    s = WHITESPACE_RE.sub(" ", s)
    return s.strip().lower()


def company_tokens(name: object) -> list[str]:
    normalized = normalize_text(name)
    tokens = []
    for match in COMPANY_TOKEN_RE.finditer(normalized):
        token = match.group(0).strip("._-/#&+'")
        if len(token) >= 2:
            tokens.append(token)
    return tokens


def build_company_stoplist(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute(
        f"""
        SELECT DISTINCT company_name_canonical
        FROM read_parquet('{q(DATA)}')
        WHERE {DEFAULT_WHERE}
          AND company_name_canonical IS NOT NULL
          AND trim(company_name_canonical) <> ''
        """
    ).fetchall()
    stoplist: set[str] = set()
    for (name,) in rows:
        stoplist.update(company_tokens(name))
    COMPANY_STOPLIST_PATH.write_text("\n".join(sorted(stoplist)) + "\n")
    return stoplist


def clean_description(text: object, stoplist: set[str]) -> str:
    normalized = normalize_text(text)
    kept: list[str] = []
    for match in TEXT_TOKEN_RE.finditer(normalized):
        token = match.group(0).strip("_-")
        if not token:
            continue
        protected = token in PROTECTED_TOKENS
        if token.isdigit() and not protected:
            continue
        if len(token) < 2 and not protected:
            continue
        if token in ENGLISH_STOP_WORDS and not protected:
            continue
        if token in stoplist and not protected:
            continue
        kept.append(token)
    return " ".join(kept)


def compile_tech_patterns() -> list[tuple[str, str, re.Pattern[str], str]]:
    seen: set[str] = set()
    compiled = []
    for column, label, pattern, category in TECH_PATTERNS:
        if column in seen:
            raise ValueError(f"Duplicate technology column: {column}")
        seen.add(column)
        compiled.append((column, label, re.compile(pattern, re.IGNORECASE), category))
    return compiled


def assert_tech_regexes(patterns: list[tuple[str, str, re.Pattern[str], str]]) -> None:
    pat = {column: compiled for column, _label, compiled, _category in patterns}

    escaped = clean_description(r"C\+\+, C\#, ASP\.NET, and CI/CD", set())
    assert pat["cpp"].search(escaped), escaped
    assert pat["csharp"].search(escaped), escaped
    assert pat["dotnet"].search(escaped), escaped
    assert pat["ci_cd"].search(escaped), escaped

    java = clean_description("Java backend services", set())
    javascript = clean_description("JavaScript and TypeScript frontend", set())
    assert pat["java"].search(java), java
    assert not pat["java"].search(javascript), javascript
    assert pat["javascript"].search(javascript), javascript

    assert pat["dotnet"].search(clean_description(r"\.NET platform", set()))
    assert pat["dotnet"].search(clean_description("dotnet services", set()))
    assert pat["ci_cd"].search(clean_description(r"CI\/CD pipelines", set()))
    assert pat["cpp"].search(clean_description("C++ systems", set()))
    assert pat["csharp"].search(clean_description("C# services", set()))
    assert not pat["cpp"].search(clean_description("c plus plus not canonical", set()))


def write_tech_taxonomy(patterns: list[tuple[str, str, re.Pattern[str], str]]) -> None:
    with TAXONOMY_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["column", "label", "category", "regex"])
        writer.writeheader()
        for (column, label, pattern, category), (_c, _l, raw_pattern, _cat) in zip(patterns, TECH_PATTERNS):
            writer.writerow(
                {
                    "column": column,
                    "label": label,
                    "category": category,
                    "regex": raw_pattern,
                }
            )


def build_cleaned_text(con: duckdb.DuckDBPyConnection, stoplist: set[str]) -> dict[str, object]:
    if CLEANED_PATH.exists():
        CLEANED_PATH.unlink()

    select_cols = [
        "uid",
        "description",
        "description_core_llm",
        "llm_extraction_coverage",
        "source",
        "period",
        "seniority_final",
        "seniority_3level",
        "is_aggregator",
        "company_name_canonical",
        "metro_area",
        "yoe_extracted",
        "swe_classification_tier",
        "seniority_final_source",
    ]
    sql = f"""
        SELECT {", ".join(select_cols)}
        FROM read_parquet('{q(DATA)}')
        WHERE {DEFAULT_WHERE}
        ORDER BY source, uid
    """
    reader = con.execute(sql).fetch_record_batch(rows_per_batch=ARROW_BATCH_SIZE)
    writer: pq.ParquetWriter | None = None
    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    text_source_counts: Counter[str] = Counter()
    by_source_text: Counter[tuple[str, str]] = Counter()
    llm_missing_core = 0
    rows_written = 0

    schema = pa.schema(
        [
            ("uid", pa.string()),
            ("description_cleaned", pa.string()),
            ("text_source", pa.string()),
            ("source", pa.string()),
            ("period", pa.string()),
            ("seniority_final", pa.string()),
            ("seniority_3level", pa.string()),
            ("is_aggregator", pa.bool_()),
            ("company_name_canonical", pa.string()),
            ("metro_area", pa.string()),
            ("yoe_extracted", pa.float64()),
            ("swe_classification_tier", pa.string()),
            ("seniority_final_source", pa.string()),
        ]
    )

    try:
        for batch in reader:
            data = batch.to_pydict()
            n = len(data["uid"])
            out = {name: [] for name in schema.names}
            for i in range(n):
                coverage = data["llm_extraction_coverage"][i]
                core = data["description_core_llm"][i]
                raw = data["description"][i]
                if coverage == "labeled":
                    text_source = "llm"
                    text = core if core is not None else ""
                    if core is None:
                        llm_missing_core += 1
                else:
                    text_source = "raw"
                    text = raw
                cleaned = clean_description(text, stoplist)

                out["uid"].append(data["uid"][i])
                out["description_cleaned"].append(cleaned)
                out["text_source"].append(text_source)
                for col in [
                    "source",
                    "period",
                    "seniority_final",
                    "seniority_3level",
                    "is_aggregator",
                    "company_name_canonical",
                    "metro_area",
                    "yoe_extracted",
                    "swe_classification_tier",
                    "seniority_final_source",
                ]:
                    out[col].append(data[col][i])

                counts["rows"] += 1
                source = data["source"][i] or "NULL"
                source_counts[source] += 1
                text_source_counts[text_source] += 1
                by_source_text[(source, text_source)] += 1
                if not cleaned:
                    counts["empty_cleaned"] += 1
            table = pa.Table.from_pydict(out, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(CLEANED_PATH, schema=schema, compression="zstd")
            writer.write_table(table)
            rows_written += n
            print(f"[cleaned] wrote {rows_written:,} rows", flush=True)
    finally:
        if writer is not None:
            writer.close()

    return {
        "rows": int(counts["rows"]),
        "empty_cleaned": int(counts["empty_cleaned"]),
        "source_counts": dict(source_counts),
        "text_source_counts": dict(text_source_counts),
        "text_source_by_source": {f"{s}|{t}": n for (s, t), n in by_source_text.items()},
        "llm_missing_core_fallback_rows": int(llm_missing_core),
    }


def build_tech_matrix(patterns: list[tuple[str, str, re.Pattern[str], str]]) -> dict[str, object]:
    if TECH_MATRIX_PATH.exists():
        TECH_MATRIX_PATH.unlink()
    pf = pq.ParquetFile(CLEANED_PATH)
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    mention_counts = Counter()
    schema = pa.schema([("uid", pa.string()), *[(column, pa.bool_()) for column, _label, _pat, _cat in patterns]])

    try:
        for batch in pf.iter_batches(batch_size=ARROW_BATCH_SIZE, columns=["uid", "description_cleaned"]):
            data = batch.to_pydict()
            texts = [normalize_text(text) for text in data["description_cleaned"]]
            out: dict[str, pa.Array] = {"uid": pa.array(data["uid"], type=pa.string())}
            for column, _label, pattern, _category in patterns:
                values = [bool(pattern.search(text)) for text in texts]
                mention_counts[column] += int(sum(values))
                out[column] = pa.array(values, type=pa.bool_())
            table = pa.Table.from_pydict(out, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(TECH_MATRIX_PATH, schema=schema, compression="zstd")
            writer.write_table(table)
            rows_written += len(texts)
            print(f"[tech] wrote {rows_written:,} rows", flush=True)
    finally:
        if writer is not None:
            writer.close()

    return {
        "rows": rows_written,
        "technology_columns": len(patterns),
        "top_mentions": dict(mention_counts.most_common(20)),
    }


def detect_unified_columns(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{q(DATA)}')").fetchall()
    return {row[0] for row in rows}


def parse_skill_string(value: object) -> list[str]:
    if value is None:
        return []
    skills = []
    for part in str(value).split(","):
        skill = WHITESPACE_RE.sub(" ", part.strip())
        if skill:
            skills.append(skill)
    return skills


def clean_skill(value: str) -> str:
    return WHITESPACE_RE.sub(" ", normalize_text(value).replace("/", " / ")).strip()


def build_structured_skills(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    columns = detect_unified_columns(con)
    candidates = [name for name in ["skills_raw", "asaniczka_skills", "skills", "job_skills"] if name in columns]
    schema = pa.schema(
        [
            ("uid", pa.string()),
            ("skill_order", pa.int32()),
            ("skill_raw", pa.string()),
            ("skill_clean", pa.string()),
            ("skill_source", pa.string()),
        ]
    )
    if SKILLS_PATH.exists():
        SKILLS_PATH.unlink()
    if not candidates:
        pq.write_table(pa.Table.from_pydict({name: [] for name in schema.names}, schema=schema), SKILLS_PATH)
        return {"status": "limitation", "limitation": "No structured skills column found", "rows": 0, "source_field": None}

    coverage_exprs = [
        f"sum(CASE WHEN {field} IS NOT NULL AND trim({field}) <> '' THEN 1 ELSE 0 END) AS {field}"
        for field in candidates
    ]
    coverage = con.execute(
        f"""
        SELECT {", ".join(coverage_exprs)}
        FROM read_parquet('{q(DATA)}')
        WHERE {DEFAULT_WHERE}
          AND source = 'kaggle_asaniczka'
        """
    ).fetchone()
    counts = dict(zip(candidates, coverage, strict=True))
    source_field = max(candidates, key=lambda field: counts[field])
    if counts[source_field] == 0:
        pq.write_table(pa.Table.from_pydict({name: [] for name in schema.names}, schema=schema), SKILLS_PATH)
        return {
            "status": "limitation",
            "limitation": f"Structured skills columns present but empty for asaniczka SWE rows: {counts}",
            "rows": 0,
            "source_field": source_field,
        }

    sql = f"""
        SELECT uid, {source_field} AS skills_value
        FROM read_parquet('{q(DATA)}')
        WHERE {DEFAULT_WHERE}
          AND source = 'kaggle_asaniczka'
          AND {source_field} IS NOT NULL
          AND trim({source_field}) <> ''
        ORDER BY uid
    """
    reader = con.execute(sql).fetch_record_batch(rows_per_batch=ARROW_BATCH_SIZE)
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    posting_count = 0
    try:
        for batch in reader:
            data = batch.to_pydict()
            out = {name: [] for name in schema.names}
            for uid, skills_value in zip(data["uid"], data["skills_value"], strict=True):
                posting_count += 1
                for idx, skill in enumerate(parse_skill_string(skills_value), start=1):
                    out["uid"].append(uid)
                    out["skill_order"].append(idx)
                    out["skill_raw"].append(skill)
                    out["skill_clean"].append(clean_skill(skill))
                    out["skill_source"].append(source_field)
            table = pa.Table.from_pydict(out, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(SKILLS_PATH, schema=schema, compression="zstd")
            if table.num_rows:
                writer.write_table(table)
                rows_written += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    return {
        "status": "ok",
        "source_field": source_field,
        "field_coverage": counts,
        "postings_with_skills": posting_count,
        "rows": rows_written,
    }


def sql_bool_avg(expr: str) -> str:
    return f"avg(CASE WHEN {expr} THEN 1.0 ELSE 0.0 END)"


def metric_effect(values: dict[str, float | None]) -> tuple[float | None, float | None, float | None]:
    a = values.get("kaggle_arshkon")
    b = values.get("kaggle_asaniczka")
    s = values.get("scraped")
    within = None if a is None or b is None else b - a
    cross = None if a is None or s is None else s - a
    if within is None or cross is None or abs(within) < 1e-12:
        ratio = None
    else:
        ratio = abs(cross) / abs(within)
    return within, cross, ratio


def source_metric(con: duckdb.DuckDBPyConnection, sql: str) -> dict[str, tuple[float | None, int]]:
    rows = con.execute(sql).fetchall()
    out = {}
    for source, value, denominator in rows:
        out[source] = (None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value), int(denominator))
    return out


def build_calibration_table(patterns: list[tuple[str, str, re.Pattern[str], str]]) -> dict[str, object]:
    con = connect()
    tech_cols = [column for column, _label, _pat, _cat in patterns]
    tech_count_expr = " + ".join([f"CASE WHEN m.{col} THEN 1 ELSE 0 END" for col in tech_cols])
    ai_cols = [
        "llm",
        "generative_ai",
        "chatgpt",
        "copilot",
        "cursor",
        "claude",
        "gemini",
        "codex",
        "openai_api",
        "claude_api",
        "anthropic_api",
        "prompt_engineering",
        "rag",
        "mcp",
        "agents",
    ]
    cloud_cols = ["aws", "azure", "gcp", "kubernetes", "docker", "terraform", "helm", "ansible"]
    data_cols = ["sql", "postgresql", "mysql", "mongodb", "redis", "kafka", "spark", "snowflake", "databricks", "dbt"]
    frontend_cols = ["javascript", "typescript", "react", "angular", "vue", "nextjs", "nodejs", "html", "css"]
    testing_cols = ["jest", "pytest", "junit", "selenium", "cypress", "playwright", "unit_testing", "integration_testing", "tdd"]

    def or_cols(cols: list[str]) -> str:
        return " OR ".join([f"m.{col}" for col in cols])

    llm_labeled_avg = sql_bool_avg("llm_extraction_coverage = 'labeled'")
    raw_fallback_avg = sql_bool_avg("text_source = 'raw'")
    company_known_avg = sql_bool_avg("company_name_canonical IS NOT NULL AND trim(company_name_canonical) <> ''")
    metro_known_avg = sql_bool_avg("metro_area IS NOT NULL AND trim(metro_area) <> ''")
    seniority_unknown_avg = sql_bool_avg("seniority_final IS NULL OR seniority_final = 'unknown'")
    j1_avg = sql_bool_avg("seniority_final = 'entry'")
    j2_avg = sql_bool_avg("seniority_final IN ('entry','associate')")
    s1_avg = sql_bool_avg("seniority_final IN ('mid-senior','director')")
    s2_avg = sql_bool_avg("seniority_final = 'director'")

    metric_specs: list[dict[str, str]] = [
        {
            "metric": "description_length_mean",
            "metric_type": "mean",
            "definition": "Mean raw description_length in default LinkedIn SWE frame",
            "denominator": "all default LinkedIn SWE rows",
            "sql": f"SELECT source, avg(description_length)::DOUBLE AS value, count(*) AS denominator FROM read_parquet('{q(DATA)}') WHERE {DEFAULT_WHERE} GROUP BY source",
        },
        {
            "metric": "description_length_median",
            "metric_type": "median",
            "definition": "Median raw description_length in default LinkedIn SWE frame",
            "denominator": "all default LinkedIn SWE rows with non-null description_length",
            "sql": f"SELECT source, median(description_length)::DOUBLE AS value, count(description_length) AS denominator FROM read_parquet('{q(DATA)}') WHERE {DEFAULT_WHERE} GROUP BY source",
        },
        {
            "metric": "cleaned_text_length_mean",
            "metric_type": "mean",
            "definition": "Mean character length of description_cleaned",
            "denominator": "all rows in swe_cleaned_text.parquet",
            "sql": f"SELECT source, avg(length(description_cleaned))::DOUBLE AS value, count(*) AS denominator FROM read_parquet('{q(CLEANED_PATH)}') GROUP BY source",
        },
        {
            "metric": "cleaned_text_length_median",
            "metric_type": "median",
            "definition": "Median character length of description_cleaned",
            "denominator": "all rows in swe_cleaned_text.parquet",
            "sql": f"SELECT source, median(length(description_cleaned))::DOUBLE AS value, count(*) AS denominator FROM read_parquet('{q(CLEANED_PATH)}') GROUP BY source",
        },
        {
            "metric": "yoe_extracted_mean",
            "metric_type": "mean",
            "definition": "Mean yoe_extracted among rows with known YOE",
            "denominator": "YOE-known default LinkedIn SWE rows",
            "sql": f"SELECT source, avg(yoe_extracted)::DOUBLE AS value, count(yoe_extracted) AS denominator FROM read_parquet('{q(DATA)}') WHERE {DEFAULT_WHERE} GROUP BY source",
        },
        {
            "metric": "yoe_extracted_median",
            "metric_type": "median",
            "definition": "Median yoe_extracted among rows with known YOE",
            "denominator": "YOE-known default LinkedIn SWE rows",
            "sql": f"SELECT source, median(yoe_extracted)::DOUBLE AS value, count(yoe_extracted) AS denominator FROM read_parquet('{q(DATA)}') WHERE {DEFAULT_WHERE} GROUP BY source",
        },
        {
            "metric": "yoe_known_share",
            "metric_type": "proportion",
            "definition": "Share with non-null yoe_extracted",
            "denominator": "all default LinkedIn SWE rows",
            "sql": f"SELECT source, {sql_bool_avg('yoe_extracted IS NOT NULL')} AS value, count(*) AS denominator FROM read_parquet('{q(DATA)}') WHERE {DEFAULT_WHERE} GROUP BY source",
        },
        {
            "metric": "llm_text_labeled_share",
            "metric_type": "proportion",
            "definition": "Share with llm_extraction_coverage = 'labeled'",
            "denominator": "all default LinkedIn SWE rows",
            "sql": f"SELECT source, {llm_labeled_avg} AS value, count(*) AS denominator FROM read_parquet('{q(DATA)}') WHERE {DEFAULT_WHERE} GROUP BY source",
        },
        {
            "metric": "raw_text_fallback_share",
            "metric_type": "proportion",
            "definition": "Share of cleaned-text artifact using raw description fallback",
            "denominator": "all rows in swe_cleaned_text.parquet",
            "sql": f"SELECT source, {raw_fallback_avg} AS value, count(*) AS denominator FROM read_parquet('{q(CLEANED_PATH)}') GROUP BY source",
        },
        {
            "metric": "aggregator_share",
            "metric_type": "proportion",
            "definition": "Share flagged is_aggregator",
            "denominator": "all default LinkedIn SWE rows",
            "sql": f"SELECT source, {sql_bool_avg('coalesce(is_aggregator, false)')} AS value, count(*) AS denominator FROM read_parquet('{q(DATA)}') WHERE {DEFAULT_WHERE} GROUP BY source",
        },
        {
            "metric": "company_known_share",
            "metric_type": "proportion",
            "definition": "Share with non-empty company_name_canonical",
            "denominator": "all default LinkedIn SWE rows",
            "sql": f"SELECT source, {company_known_avg} AS value, count(*) AS denominator FROM read_parquet('{q(DATA)}') WHERE {DEFAULT_WHERE} GROUP BY source",
        },
        {
            "metric": "metro_known_share",
            "metric_type": "proportion",
            "definition": "Share with non-empty metro_area",
            "denominator": "all default LinkedIn SWE rows",
            "sql": f"SELECT source, {metro_known_avg} AS value, count(*) AS denominator FROM read_parquet('{q(DATA)}') WHERE {DEFAULT_WHERE} GROUP BY source",
        },
        {
            "metric": "seniority_unknown_share",
            "metric_type": "proportion",
            "definition": "Share with seniority_final null or unknown",
            "denominator": "all default LinkedIn SWE rows",
            "sql": f"SELECT source, {seniority_unknown_avg} AS value, count(*) AS denominator FROM read_parquet('{q(DATA)}') WHERE {DEFAULT_WHERE} GROUP BY source",
        },
        {
            "metric": "j1_entry_label_share",
            "metric_type": "proportion",
            "definition": "T30 J1: seniority_final = 'entry'",
            "denominator": "all default LinkedIn SWE rows",
            "sql": f"SELECT source, {j1_avg} AS value, count(*) AS denominator FROM read_parquet('{q(DATA)}') WHERE {DEFAULT_WHERE} GROUP BY source",
        },
        {
            "metric": "j2_entry_associate_share",
            "metric_type": "proportion",
            "definition": "T30 J2: seniority_final in entry/associate",
            "denominator": "all default LinkedIn SWE rows",
            "sql": f"SELECT source, {j2_avg} AS value, count(*) AS denominator FROM read_parquet('{q(DATA)}') WHERE {DEFAULT_WHERE} GROUP BY source",
        },
        {
            "metric": "j3_yoe_le_2_share",
            "metric_type": "proportion",
            "definition": "T30 J3: yoe_extracted <= 2",
            "denominator": "YOE-known default LinkedIn SWE rows",
            "sql": f"SELECT source, {sql_bool_avg('yoe_extracted <= 2')} AS value, count(*) AS denominator FROM read_parquet('{q(DATA)}') WHERE {DEFAULT_WHERE} AND yoe_extracted IS NOT NULL GROUP BY source",
        },
        {
            "metric": "j4_yoe_le_3_share",
            "metric_type": "proportion",
            "definition": "T30 J4: yoe_extracted <= 3",
            "denominator": "YOE-known default LinkedIn SWE rows",
            "sql": f"SELECT source, {sql_bool_avg('yoe_extracted <= 3')} AS value, count(*) AS denominator FROM read_parquet('{q(DATA)}') WHERE {DEFAULT_WHERE} AND yoe_extracted IS NOT NULL GROUP BY source",
        },
        {
            "metric": "s1_mid_senior_director_share",
            "metric_type": "proportion",
            "definition": "T30 S1: seniority_final in mid-senior/director",
            "denominator": "all default LinkedIn SWE rows",
            "sql": f"SELECT source, {s1_avg} AS value, count(*) AS denominator FROM read_parquet('{q(DATA)}') WHERE {DEFAULT_WHERE} GROUP BY source",
        },
        {
            "metric": "s2_director_share",
            "metric_type": "proportion",
            "definition": "T30 S2: seniority_final = 'director'",
            "denominator": "all default LinkedIn SWE rows",
            "sql": f"SELECT source, {s2_avg} AS value, count(*) AS denominator FROM read_parquet('{q(DATA)}') WHERE {DEFAULT_WHERE} GROUP BY source",
        },
        {
            "metric": "s4_yoe_ge_5_share",
            "metric_type": "proportion",
            "definition": "T30 S4: yoe_extracted >= 5",
            "denominator": "YOE-known default LinkedIn SWE rows",
            "sql": f"SELECT source, {sql_bool_avg('yoe_extracted >= 5')} AS value, count(*) AS denominator FROM read_parquet('{q(DATA)}') WHERE {DEFAULT_WHERE} AND yoe_extracted IS NOT NULL GROUP BY source",
        },
        {
            "metric": "s5_yoe_ge_8_share",
            "metric_type": "proportion",
            "definition": "T30 S5: yoe_extracted >= 8",
            "denominator": "YOE-known default LinkedIn SWE rows",
            "sql": f"SELECT source, {sql_bool_avg('yoe_extracted >= 8')} AS value, count(*) AS denominator FROM read_parquet('{q(DATA)}') WHERE {DEFAULT_WHERE} AND yoe_extracted IS NOT NULL GROUP BY source",
        },
        {
            "metric": "tech_count_mean",
            "metric_type": "mean",
            "definition": "Mean count of technology/tool/practice indicators true per posting",
            "denominator": "all rows in tech matrix",
            "sql": f"SELECT c.source, avg(({tech_count_expr})::DOUBLE) AS value, count(*) AS denominator FROM read_parquet('{q(TECH_MATRIX_PATH)}') m JOIN read_parquet('{q(CLEANED_PATH)}') c USING(uid) GROUP BY c.source",
        },
        {
            "metric": "ai_keyword_prevalence",
            "metric_type": "proportion",
            "definition": "Any AI tool/domain indicator in shared tech matrix",
            "denominator": "all rows in tech matrix",
            "sql": f"SELECT c.source, {sql_bool_avg(or_cols(ai_cols))} AS value, count(*) AS denominator FROM read_parquet('{q(TECH_MATRIX_PATH)}') m JOIN read_parquet('{q(CLEANED_PATH)}') c USING(uid) GROUP BY c.source",
        },
        {
            "metric": "cloud_container_prevalence",
            "metric_type": "proportion",
            "definition": "Any cloud/container/IaC indicator: AWS, Azure, GCP, Kubernetes, Docker, Terraform, Helm, Ansible",
            "denominator": "all rows in tech matrix",
            "sql": f"SELECT c.source, {sql_bool_avg(or_cols(cloud_cols))} AS value, count(*) AS denominator FROM read_parquet('{q(TECH_MATRIX_PATH)}') m JOIN read_parquet('{q(CLEANED_PATH)}') c USING(uid) GROUP BY c.source",
        },
        {
            "metric": "data_tool_prevalence",
            "metric_type": "proportion",
            "definition": "Any selected database/data-platform indicator",
            "denominator": "all rows in tech matrix",
            "sql": f"SELECT c.source, {sql_bool_avg(or_cols(data_cols))} AS value, count(*) AS denominator FROM read_parquet('{q(TECH_MATRIX_PATH)}') m JOIN read_parquet('{q(CLEANED_PATH)}') c USING(uid) GROUP BY c.source",
        },
        {
            "metric": "frontend_stack_prevalence",
            "metric_type": "proportion",
            "definition": "Any selected frontend/web stack indicator",
            "denominator": "all rows in tech matrix",
            "sql": f"SELECT c.source, {sql_bool_avg(or_cols(frontend_cols))} AS value, count(*) AS denominator FROM read_parquet('{q(TECH_MATRIX_PATH)}') m JOIN read_parquet('{q(CLEANED_PATH)}') c USING(uid) GROUP BY c.source",
        },
        {
            "metric": "testing_practice_prevalence",
            "metric_type": "proportion",
            "definition": "Any selected testing practice/tool indicator",
            "denominator": "all rows in tech matrix",
            "sql": f"SELECT c.source, {sql_bool_avg(or_cols(testing_cols))} AS value, count(*) AS denominator FROM read_parquet('{q(TECH_MATRIX_PATH)}') m JOIN read_parquet('{q(CLEANED_PATH)}') c USING(uid) GROUP BY c.source",
        },
    ]

    for col in [
        "python",
        "java",
        "javascript",
        "typescript",
        "cpp",
        "csharp",
        "dotnet",
        "aws",
        "azure",
        "kubernetes",
        "docker",
        "sql",
        "ci_cd",
        "llm",
        "copilot",
        "chatgpt",
        "prompt_engineering",
        "rag",
    ]:
        metric_specs.append(
            {
                "metric": f"{col}_prevalence",
                "metric_type": "proportion",
                "definition": f"Shared tech matrix indicator `{col}`",
                "denominator": "all rows in tech matrix",
                "sql": f"SELECT c.source, {sql_bool_avg(f'm.{col}')} AS value, count(*) AS denominator FROM read_parquet('{q(TECH_MATRIX_PATH)}') m JOIN read_parquet('{q(CLEANED_PATH)}') c USING(uid) GROUP BY c.source",
            }
        )

    text_keyword_specs = [
        (
            "management_indicator_rate",
            r"\b(?:manage|manager|management|mentor|mentorship|coach|coaching|hiring|leadership)\b",
            "Lightweight management/leadership keyword indicator; no semantic precision audit",
        ),
        (
            "scope_indicator_rate",
            r"\b(?:ownership|owner|end-to-end|end to end|cross-functional|stakeholder|architecture|architectural|scalable|systems design)\b",
            "Lightweight ownership/scope keyword indicator; no semantic precision audit",
        ),
        (
            "soft_skill_indicator_rate",
            r"\b(?:communication|collaboration|teamwork|problem solving|problem-solving|written communication|verbal communication)\b",
            "Lightweight soft-skill keyword indicator; no semantic precision audit",
        ),
    ]
    for metric, pattern, definition in text_keyword_specs:
        escaped_pattern = pattern.replace("'", "''")
        keyword_avg = sql_bool_avg(f"regexp_matches(description_cleaned, '{escaped_pattern}')")
        metric_specs.append(
            {
                "metric": metric,
                "metric_type": "proportion",
                "definition": definition,
                "denominator": "all rows in swe_cleaned_text.parquet",
                "sql": f"SELECT source, {keyword_avg} AS value, count(*) AS denominator FROM read_parquet('{q(CLEANED_PATH)}') GROUP BY source",
            }
        )

    rows = []
    for spec in metric_specs:
        try:
            result = source_metric(con, spec["sql"])
            values = {source: result.get(source, (None, 0))[0] for source in SOURCE_ORDER}
            denominators = {source: result.get(source, (None, 0))[1] for source in SOURCE_ORDER}
            within, cross, ratio = metric_effect(values)
            status = "ok"
            limitation = ""
            if any(values.get(source) is None for source in SOURCE_ORDER):
                status = "limitation"
                limitation = "Metric missing for at least one source."
            elif within is None or abs(within) < 1e-12:
                status = "ok_limited"
                limitation = "Within-2024 effect is zero or undefined, so calibration ratio is undefined."
            if "Lightweight" in spec["definition"]:
                status = "ok_limited" if status == "ok" else status
                limitation = (limitation + " " if limitation else "") + "Regex was not semantically precision-audited."
            rows.append(
                {
                    "metric": spec["metric"],
                    "metric_type": spec["metric_type"],
                    "definition": spec["definition"],
                    "denominator": spec["denominator"],
                    "arshkon_value": values.get("kaggle_arshkon"),
                    "asaniczka_value": values.get("kaggle_asaniczka"),
                    "scraped_value": values.get("scraped"),
                    "arshkon_denominator": denominators.get("kaggle_arshkon"),
                    "asaniczka_denominator": denominators.get("kaggle_asaniczka"),
                    "scraped_denominator": denominators.get("scraped"),
                    "within_2024_effect": within,
                    "cross_period_effect": cross,
                    "calibration_ratio_abs": ratio,
                    "status": status,
                    "limitation": limitation,
                }
            )
        except Exception as exc:  # keep table complete even if one metric breaks
            rows.append(
                {
                    "metric": spec["metric"],
                    "metric_type": spec["metric_type"],
                    "definition": spec["definition"],
                    "denominator": spec["denominator"],
                    "arshkon_value": None,
                    "asaniczka_value": None,
                    "scraped_value": None,
                    "arshkon_denominator": None,
                    "asaniczka_denominator": None,
                    "scraped_denominator": None,
                    "within_2024_effect": None,
                    "cross_period_effect": None,
                    "calibration_ratio_abs": None,
                    "status": "limitation",
                    "limitation": repr(exc),
                }
            )
    pd.DataFrame(rows).to_csv(CALIBRATION_PATH, index=False)
    con.close()
    return {"rows": len(rows), "ok_rows": sum(row["status"] == "ok" for row in rows)}


def build_tech_sanity(patterns: list[tuple[str, str, re.Pattern[str], str]]) -> dict[str, object]:
    con = connect()
    selects = []
    for column, label, _pattern, category in patterns:
        selects.append(
            f"""
            SELECT
              '{column}' AS technology,
              '{label.replace("'", "''")}' AS label,
              '{category}' AS category,
              c.source AS source,
              count(*) AS denominator,
              sum(CASE WHEN m.{column} THEN 1 ELSE 0 END) AS mentions,
              avg(CASE WHEN m.{column} THEN 1.0 ELSE 0.0 END) AS mention_rate
            FROM read_parquet('{q(TECH_MATRIX_PATH)}') m
            JOIN read_parquet('{q(CLEANED_PATH)}') c USING(uid)
            GROUP BY c.source
            """
        )
    long = con.execute(" UNION ALL ".join(selects)).fetchdf()
    con.close()

    pivot = long.pivot_table(
        index=["technology", "label", "category"],
        columns="source",
        values=["denominator", "mentions", "mention_rate"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{source}" for metric, source in pivot.columns]
    out = pivot.reset_index()
    for source in SOURCE_ORDER:
        for prefix in ["denominator", "mentions", "mention_rate"]:
            col = f"{prefix}_{source}"
            if col not in out:
                out[col] = np.nan
    a = out["mention_rate_kaggle_arshkon"].astype(float)
    s = out["mention_rate_scraped"].astype(float)
    ratio = np.where((s == 0) & (a == 0), np.nan, np.where(s == 0, np.inf, a / s))
    out["arshkon_to_scraped_ratio"] = ratio
    out["flag_ratio_gt_3_or_lt_0_33"] = (ratio > 3.0) | (ratio < 0.33)

    tricky = {"cpp", "csharp", "dotnet", "ci_cd", "java", "javascript"}
    notes = []
    for tech, flagged in zip(out["technology"], out["flag_ratio_gt_3_or_lt_0_33"], strict=True):
        if not flagged:
            notes.append("not flagged")
        elif tech in tricky:
            notes.append("flagged; regex edge-case asserts passed after markdown-escape normalization")
        else:
            notes.append("flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review")
    out["investigation_note"] = notes
    out = out.sort_values(["flag_ratio_gt_3_or_lt_0_33", "technology"], ascending=[False, True])
    out.to_csv(TECH_SANITY_PATH, index=False)
    return {
        "rows": len(out),
        "flagged_rows": int(out["flag_ratio_gt_3_or_lt_0_33"].sum()),
        "flagged_technologies": out.loc[out["flag_ratio_gt_3_or_lt_0_33"], "technology"].tolist(),
    }


def first_512_tokens(text: object) -> str:
    tokens = str(text or "").split()
    return " ".join(tokens[:512])


def write_embedding_checkpoint(payload: dict[str, object]) -> None:
    EMBEDDING_CHECKPOINT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_embeddings() -> dict[str, object]:
    from sentence_transformers import SentenceTransformer

    con = connect()
    target_rows = con.execute(
        f"SELECT count(*) FROM read_parquet('{q(CLEANED_PATH)}') WHERE text_source = 'llm'"
    ).fetchone()[0]
    con.close()

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name, device="cpu")
    dim = int(model.get_sentence_embedding_dimension())
    tmp_path = EMBEDDINGS_PATH.with_suffix(".partial.npy")
    if tmp_path.exists():
        tmp_path.unlink()
    embeddings = np.lib.format.open_memmap(tmp_path, mode="w+", dtype="float32", shape=(target_rows, dim))

    uid_rows: list[str] = []
    rows_written = 0
    batch_uids: list[str] = []
    batch_texts: list[str] = []
    error: str | None = None

    def flush() -> None:
        nonlocal rows_written, batch_texts, batch_uids
        if not batch_texts:
            return
        emb = model.encode(
            batch_texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
            device="cpu",
        ).astype("float32", copy=False)
        n = emb.shape[0]
        embeddings[rows_written : rows_written + n, :] = emb
        uid_rows.extend(batch_uids)
        rows_written += n
        batch_uids = []
        batch_texts = []
        if rows_written % EMBED_CHECKPOINT_EVERY == 0 or rows_written == target_rows:
            embeddings.flush()
            write_embedding_checkpoint(
                {
                    "model": model_name,
                    "target_rows": target_rows,
                    "rows_written": rows_written,
                    "complete": rows_written == target_rows,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "batch_size": EMBED_BATCH_SIZE,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                }
            )
            print(f"[embeddings] wrote {rows_written:,}/{target_rows:,}", flush=True)

    try:
        pf = pq.ParquetFile(CLEANED_PATH)
        for batch in pf.iter_batches(batch_size=ARROW_BATCH_SIZE, columns=["uid", "description_cleaned", "text_source"]):
            data = batch.to_pydict()
            for uid, text, text_source in zip(data["uid"], data["description_cleaned"], data["text_source"], strict=True):
                if text_source != "llm":
                    continue
                batch_uids.append(uid)
                batch_texts.append(first_512_tokens(text))
                if len(batch_texts) >= EMBED_BATCH_SIZE:
                    flush()
        flush()
    except Exception as exc:
        error = repr(exc)
    finally:
        embeddings.flush()
        del embeddings

    if rows_written == target_rows:
        tmp_path.replace(EMBEDDINGS_PATH)
        complete = True
    else:
        partial = np.load(tmp_path, mmap_mode="r")[:rows_written].copy()
        np.save(EMBEDDINGS_PATH, partial.astype("float32", copy=False))
        tmp_path.unlink(missing_ok=True)
        complete = False

    index_table = pa.Table.from_pydict(
        {
            "embedding_row": list(range(rows_written)),
            "uid": uid_rows[:rows_written],
            "model": [model_name] * rows_written,
            "text_source": ["llm"] * rows_written,
            "tokens_truncated_to": [512] * rows_written,
        },
        schema=pa.schema(
            [
                ("embedding_row", pa.int32()),
                ("uid", pa.string()),
                ("model", pa.string()),
                ("text_source", pa.string()),
                ("tokens_truncated_to", pa.int32()),
            ]
        ),
    )
    pq.write_table(index_table, EMBEDDING_INDEX_PATH, compression="zstd")
    checkpoint = {
        "model": model_name,
        "target_rows": target_rows,
        "rows_written": rows_written,
        "complete": complete,
        "error": error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "batch_size": EMBED_BATCH_SIZE,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "embedding_dim": dim,
    }
    write_embedding_checkpoint(checkpoint)
    return checkpoint


def scalar_counts() -> dict[str, object]:
    con = connect()
    out = {}
    out["cleaned_rows"] = con.execute(f"SELECT count(*) FROM read_parquet('{q(CLEANED_PATH)}')").fetchone()[0]
    out["tech_rows"] = con.execute(f"SELECT count(*) FROM read_parquet('{q(TECH_MATRIX_PATH)}')").fetchone()[0]
    out["embedding_index_rows"] = con.execute(f"SELECT count(*) FROM read_parquet('{q(EMBEDDING_INDEX_PATH)}')").fetchone()[0]
    out["skills_rows"] = con.execute(f"SELECT count(*) FROM read_parquet('{q(SKILLS_PATH)}')").fetchone()[0]
    out["text_source_distribution"] = con.execute(
        f"""
        SELECT source, text_source, count(*) AS rows
        FROM read_parquet('{q(CLEANED_PATH)}')
        GROUP BY source, text_source
        ORDER BY source, text_source
        """
    ).fetchdf().to_dict(orient="records")
    con.close()
    return out


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        display[col] = display[col].map(lambda value: "" if pd.isna(value) else str(value))
    headers = list(display.columns)
    rows = display.values.tolist()
    widths = [
        max(len(header), *(len(row[idx]) for row in rows)) if rows else len(header)
        for idx, header in enumerate(headers)
    ]
    header = "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def write_readme(metadata: dict[str, object]) -> None:
    sanity = pd.read_csv(TECH_SANITY_PATH)
    calibration = pd.read_csv(CALIBRATION_PATH)
    text_dist = pd.DataFrame(metadata["validation"]["text_source_distribution"])
    text_dist = text_dist[["source", "text_source", "rows"]]
    flagged = sanity[sanity["flag_ratio_gt_3_or_lt_0_33"] == True]  # noqa: E712
    top_flags = flagged[["technology", "label", "arshkon_to_scraped_ratio", "investigation_note"]].head(20)

    lines = [
        "# Wave 1.5 Shared Preprocessing Artifacts",
        "",
        f"Built: `{metadata['finished_at_utc']}`",
        "",
        "## Inventory",
        "",
        "| Artifact | Rows / columns | Notes |",
        "|---|---:|---|",
        f"| `swe_cleaned_text.parquet` | {metadata['validation']['cleaned_rows']:,} rows / 13 columns | Default LinkedIn, English, date-ok, SWE rows. Uses LLM cleaned text when labeled, otherwise raw fallback. |",
        f"| `swe_embeddings.npy` | {metadata['embedding']['rows_written']:,} rows x {metadata['embedding'].get('embedding_dim', 'unknown')} dims | all-MiniLM-L6-v2 embeddings for `text_source = 'llm'`, first 512 whitespace tokens, float32. |",
        f"| `swe_embedding_index.parquet` | {metadata['validation']['embedding_index_rows']:,} rows | Row index to `uid` mapping for embeddings. |",
        f"| `swe_tech_matrix.parquet` | {metadata['validation']['tech_rows']:,} rows / {metadata['tech_matrix']['technology_columns'] + 1} columns | `uid` plus boolean technology/tool/practice indicators. |",
        f"| `company_stoplist.txt` | {metadata['company_stoplist_tokens']:,} tokens | Lowercased tokens from `company_name_canonical` in the default SWE LinkedIn frame. |",
        f"| `asaniczka_structured_skills.parquet` | {metadata['validation']['skills_rows']:,} rows | Long-form parsed skills from `{metadata['structured_skills'].get('source_field')}`. |",
        f"| `calibration_table.csv` | {metadata['calibration']['rows']:,} metrics | Within-2024 and arshkon-to-scraped lightweight calibration metrics. |",
        f"| `tech_matrix_sanity.csv` | {metadata['tech_sanity']['rows']:,} technologies | Source-specific mention rates and arshkon/scraped ratio flags. |",
        "| `tech_taxonomy.csv` | "
        f"{metadata['tech_matrix']['technology_columns']:,} rows | Regex definitions used for the tech matrix. |",
        "| `prep_build_metadata.json` | 1 JSON document | Build parameters, counts, and memory posture. |",
        "",
        "## Row Counts And Text Sources",
        "",
        md_table(text_dist),
        "",
        "The scraped LinkedIn cleaned-text constraint from Gate 1 remains binding: roughly two-thirds of scraped SWE rows use raw text fallback in this shared artifact. Text-sensitive downstream tasks should filter to `text_source = 'llm'` and report coverage.",
        f"{metadata['cleaned_text']['llm_missing_core_fallback_rows']:,} rows had `llm_extraction_coverage = 'labeled'` but null `description_core_llm`; per the task definition they remain `text_source = 'llm'` with empty cleaned text. Total empty `description_cleaned` rows: {metadata['cleaned_text']['empty_cleaned']:,}.",
        "",
        "## Embedding Coverage",
        "",
        f"- Target rows (`text_source = 'llm'`): {metadata['embedding']['target_rows']:,}",
        f"- Rows embedded: {metadata['embedding']['rows_written']:,}",
        f"- Complete: `{metadata['embedding']['complete']}`",
        f"- Batch size: {metadata['embedding']['batch_size']}",
        f"- `CUDA_VISIBLE_DEVICES`: `{metadata['embedding']['cuda_visible_devices']}`",
        "",
        "## Tech Sanity",
        "",
        f"- Technology columns: {metadata['tech_matrix']['technology_columns']:,}",
        f"- Ratio-flagged technologies: {metadata['tech_sanity']['flagged_rows']:,}",
        "",
        md_table(top_flags) if not top_flags.empty else "_No arshkon-to-scraped ratio flags._",
        "",
        "Regex edge-case asserts ran before scanning and covered escaped markdown variants of `C++`, `C#`, `.NET`, `CI/CD`, plus Java vs JavaScript boundaries. Residual ratio flags are documented in `tech_matrix_sanity.csv`; most are expected to reflect real source/time composition or require semantic review rather than escaped-token undercounting.",
        "",
        "## Calibration Table",
        "",
        f"`calibration_table.csv` contains {metadata['calibration']['rows']:,} metrics. Rows marked `ok_limited` are computable but use lightweight regex indicators or have undefined calibration ratios because within-2024 variation is zero.",
        "",
        "## Memory-Safety Notes",
        "",
        "- DuckDB connections set `PRAGMA memory_limit='4GB'` and `PRAGMA threads=1`.",
        "- The script selects only the default LinkedIn SWE columns needed for artifacts.",
        f"- Arrow row batches: {ARROW_BATCH_SIZE:,}.",
        f"- Sentence-transformer batches: {EMBED_BATCH_SIZE:,}, CPU-only.",
        "- `data/unified.parquet` was not loaded wholesale into pandas.",
        "- Company-token and English-stopword stripping preserves protected technology tokens such as `Go`, `C++`, `.NET`, `OpenAI`, `R`, and `CI/CD` so the requested tech indicators are not erased before scanning.",
        f"- Peak process RSS reported by `resource.getrusage`: {metadata['peak_rss_mb']:.1f} MB.",
        "",
        "## Known Limitations From Gate 1",
        "",
        "- Scraped LinkedIn LLM cleaned-text coverage is low; raw fallback is present by design but is not valid for boilerplate-sensitive claims.",
        "- `seniority_final` is conservative and the unknown seniority pool is large and structured.",
        "- Asaniczka native `associate` is not a junior proxy; do not use native asaniczka seniority as an entry baseline.",
        "- Company composition is a first-order confound; corpus-level downstream tasks need company caps and aggregator sensitivity.",
        "- The tech matrix is regex-based and intended as a shared binary mention screen. Prevalence claims still need semantic validation under the task-reference rules.",
    ]
    README_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "default_filter": DEFAULT_WHERE,
        "duckdb_memory_limit": "4GB",
        "duckdb_threads": 1,
        "arrow_batch_size": ARROW_BATCH_SIZE,
        "embedding_batch_size": EMBED_BATCH_SIZE,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }

    con = connect()
    print("[prep] building company stoplist", flush=True)
    stoplist = build_company_stoplist(con)
    metadata["company_stoplist_tokens"] = len(stoplist)

    print("[prep] compiling and asserting tech regexes", flush=True)
    patterns = compile_tech_patterns()
    assert_tech_regexes(patterns)
    write_tech_taxonomy(patterns)

    print("[prep] building cleaned text", flush=True)
    metadata["cleaned_text"] = build_cleaned_text(con, stoplist)
    print("[prep] building structured skills", flush=True)
    metadata["structured_skills"] = build_structured_skills(con)
    con.close()

    print("[prep] building tech matrix", flush=True)
    metadata["tech_matrix"] = build_tech_matrix(patterns)

    print("[prep] building calibration table", flush=True)
    metadata["calibration"] = build_calibration_table(patterns)

    print("[prep] building tech sanity table", flush=True)
    metadata["tech_sanity"] = build_tech_sanity(patterns)

    print("[prep] building embeddings", flush=True)
    metadata["embedding"] = build_embeddings()

    metadata["validation"] = scalar_counts()
    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["build_seconds"] = round(time.time() - started, 2)
    metadata["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    write_readme(metadata)
    print(f"[prep] done in {metadata['build_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
