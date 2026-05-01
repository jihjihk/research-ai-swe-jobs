"""Wave 1.5 Agent Prep — Step 1: cleaned text artifact.

For all SWE LinkedIn rows passing the default filter:
- text_source='llm' if llm_extraction_coverage='labeled' (use description_core_llm)
- text_source='raw' otherwise (use raw description)
Tokenize, lowercase, strip company-name stoplist tokens + standard English stopwords.

Output: exploration/artifacts/shared/swe_cleaned_text.parquet
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET = "/home/jihgaboot/gabor/job-research/data/unified.parquet"
OUT = Path("/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_cleaned_text.parquet")
STOPLIST_PATH = Path(
    "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/company_stoplist.txt"
)

# Standard English stopwords via NLTK
import nltk

try:
    ENGLISH_STOPS = set(nltk.corpus.stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    ENGLISH_STOPS = set(nltk.corpus.stopwords.words("english"))


# Preserve + # . / for tech-stack detection (C++, C#, Node.js, CI/CD).
# Only strip punctuation that's clearly boilerplate separator.
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#./\-]*")


# Tech terms that must NEVER be stripped even if they appear in the company
# stoplist. Many companies are named after technologies (e.g. "Python Software
# Foundation", "OpenAI", "Cursor"), so the token-tokenized stoplist otherwise
# eats the technology vocabulary. This whitelist protects them.
TECH_PROTECT: frozenset[str] = frozenset(
    {
        # Languages
        "python", "java", "javascript", "typescript", "go", "golang", "rust",
        "c++", "c#", "ruby", "kotlin", "swift", "scala", "php", "perl",
        "bash", "shell", "sql", "matlab", "dart", "haskell", "elixir",
        "solidity", "r",
        # Frontend
        "react", "reactjs", "angular", "angularjs", "vue", "vuejs", "nextjs",
        "next.js", "svelte", "jquery", "html", "html5", "css", "css3",
        "tailwind", "webpack",
        # Backend
        "node", "nodejs", "node.js", "django", "flask", "spring", "springboot",
        ".net", "net", "dotnet", "rails", "fastapi", "express", "expressjs",
        "graphql", "rest", "restful", "grpc",
        # Cloud/devops
        "aws", "azure", "gcp", "kubernetes", "k8s", "docker", "terraform",
        "ansible", "ci/cd", "cicd", "jenkins", "argocd", "linux", "git",
        "helm", "serverless", "lambda",
        # Data
        "postgres", "postgresql", "mysql", "mongodb", "redis", "kafka",
        "spark", "hadoop", "snowflake", "databricks", "dbt", "elasticsearch",
        "airflow", "bigquery", "redshift", "cassandra", "dynamodb",
        # Traditional ML
        "tensorflow", "pytorch", "sklearn", "scikit-learn", "pandas", "numpy",
        "jupyter", "keras", "xgboost", "ml", "nlp",
        # LLM/AI
        "llm", "llms", "langchain", "langgraph", "rag", "pinecone", "chroma",
        "chromadb", "huggingface", "openai", "anthropic", "mcp", "agent",
        "agents", "agentic", "gpt", "gpt-4", "gpt-3", "transformer",
        "embedding", "embeddings",
        # AI tools
        "copilot", "cursor", "chatgpt", "claude", "gemini", "codex",
        # Testing
        "jest", "pytest", "selenium", "cypress", "junit", "playwright",
        # Practices
        "agile", "scrum", "tdd", "kanban", "microservice", "microservices",
        # Roles/pipeline common
        "backend", "frontend", "fullstack", "devops", "sre", "api",
    }
)


def load_stoplist() -> set[str]:
    tokens: set[str] = set()
    with STOPLIST_PATH.open() as f:
        for line in f:
            tok = line.strip().lower()
            if tok:
                tokens.add(tok)
    # Remove tech-protected terms from stoplist
    tokens -= TECH_PROTECT
    return tokens


def clean_text(text: str, stoplist: set[str], english_stops: set[str]) -> str:
    if not text:
        return ""
    # Lowercase tokenize
    tokens = TOKEN_RE.findall(text.lower())
    out = []
    for t in tokens:
        if t in stoplist:
            continue
        if t in english_stops:
            continue
        if len(t) < 2:
            continue
        out.append(t)
    return " ".join(out)


def main() -> None:
    t0 = time.time()
    stoplist = load_stoplist()
    print(f"Stoplist size: {len(stoplist):,}")
    print(f"English stopwords: {len(ENGLISH_STOPS):,}")

    # Assertions: tech stack tokens survive tokenization (input is pre-lowercased in clean_text).
    assert TOKEN_RE.findall("c++") == ["c++"], TOKEN_RE.findall("c++")
    assert TOKEN_RE.findall("node.js") == ["node.js"], TOKEN_RE.findall("node.js")
    assert "net" in TOKEN_RE.findall(".net"), TOKEN_RE.findall(".net")
    assert "google" in TOKEN_RE.findall("google llc")
    # Test cleaning strips google when stoplist contains google
    test_stop = {"google"}
    cleaned = clean_text("Google is hiring Python engineers for AWS", test_stop, ENGLISH_STOPS)
    assert "google" not in cleaned, cleaned
    assert "python" in cleaned, cleaned
    assert "aws" in cleaned, cleaned
    assert "is" not in cleaned.split(), cleaned  # stopword removed

    print("Inline assertions passed.")

    con = duckdb.connect()
    q = f"""
    SELECT
        uid,
        CASE WHEN llm_extraction_coverage = 'labeled' THEN description_core_llm
             ELSE description END AS text_in,
        CASE WHEN llm_extraction_coverage = 'labeled' THEN 'llm'
             ELSE 'raw' END AS text_source,
        source,
        period,
        seniority_final,
        seniority_3level,
        is_aggregator,
        company_name_canonical,
        metro_area,
        yoe_extracted,
        swe_classification_tier,
        seniority_final_source
    FROM '{PARQUET}'
    WHERE is_swe
      AND source_platform = 'linkedin'
      AND is_english = true
      AND date_flag = 'ok'
    """
    # Materialize to an arrow table, then clean in chunks
    arrow_tbl = con.execute(q).fetch_arrow_table()
    n = arrow_tbl.num_rows
    print(f"Rows to clean: {n:,}")

    cleaned_col = []
    text_in = arrow_tbl.column("text_in").to_pylist()
    for i, txt in enumerate(text_in):
        cleaned_col.append(clean_text(txt or "", stoplist, ENGLISH_STOPS))
        if (i + 1) % 20000 == 0:
            print(f"  cleaned {i + 1:,}/{n:,}")

    # Build output table
    out_cols = {
        "uid": arrow_tbl.column("uid"),
        "description_cleaned": pa.array(cleaned_col, type=pa.string()),
        "text_source": arrow_tbl.column("text_source"),
        "source": arrow_tbl.column("source"),
        "period": arrow_tbl.column("period"),
        "seniority_final": arrow_tbl.column("seniority_final"),
        "seniority_3level": arrow_tbl.column("seniority_3level"),
        "is_aggregator": arrow_tbl.column("is_aggregator"),
        "company_name_canonical": arrow_tbl.column("company_name_canonical"),
        "metro_area": arrow_tbl.column("metro_area"),
        "yoe_extracted": arrow_tbl.column("yoe_extracted"),
        "swe_classification_tier": arrow_tbl.column("swe_classification_tier"),
        "seniority_final_source": arrow_tbl.column("seniority_final_source"),
    }
    out_tbl = pa.table(out_cols)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_tbl, OUT, compression="zstd")

    # Report text_source × source × period distribution
    rep = (
        con.register("ct", out_tbl)
        .execute(
            "SELECT source, period, text_source, COUNT(*) n FROM ct "
            "GROUP BY source, period, text_source ORDER BY source, period, text_source"
        )
        .fetchall()
    )
    print("source | period | text_source | n")
    for r in rep:
        print(r)
    print(f"Done in {time.time() - t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
