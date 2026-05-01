from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
TEXT = ROOT / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet"
TECH = ROOT / "exploration" / "artifacts" / "shared" / "swe_tech_matrix.parquet"
LABELS = ROOT / "exploration" / "artifacts" / "shared" / "swe_archetype_labels.parquet"
EMBED = ROOT / "exploration" / "artifacts" / "shared" / "swe_embeddings.npy"
EMBED_INDEX = ROOT / "exploration" / "artifacts" / "shared" / "swe_embedding_index.parquet"
STOPLIST = ROOT / "exploration" / "artifacts" / "shared" / "company_stoplist.txt"

OUT_REPORT = ROOT / "exploration" / "reports" / "V1_verification.md"
OUT_TABLE = ROOT / "exploration" / "tables" / "V1"
OUT_FIG = ROOT / "exploration" / "figures" / "V1"

LLM_PERIODS = ["2024-04", "2024-01", "2026-03", "2026-04"]
PERIOD_BUCKETS = {"2024-04": "2024", "2024-01": "2024", "2026-03": "2026", "2026-04": "2026"}


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

SCOPE_TERMS = {
    "ownership": r"\bownership\b",
    "end_to_end": r"end[- ]to[- ]end|\be2e\b",
    "cross_functional": r"cross[- ]functional",
    "autonomous": r"\bautonomous(ly)?\b",
    "initiative": r"\binitiative\b",
    "strategic": r"\bstrategic\b",
    "roadmap": r"\broadmap\b",
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

TECH_FAMILY_MAP = {
    "frontend_web": [
        "html",
        "css",
        "sass",
        "less",
        "react",
        "react_native",
        "angular",
        "vue",
        "nextjs",
        "svelte",
        "redux",
        "webpack",
        "vite",
        "storybook",
        "tailwind",
        "bootstrap",
        "material_ui",
    ],
    "backend_api": [
        "nodejs",
        "express",
        "nestjs",
        "django",
        "django_rest_framework",
        "flask",
        "spring",
        "spring_boot",
        "dotnet",
        "aspnet",
        "rails",
        "laravel",
        "fastapi",
        "phoenix",
        "tornado",
        "bottle",
        "grpc",
        "graphql",
        "rest_api",
        "microservices",
        "event_driven",
        "oauth",
    ],
    "data_platform": [
        "sql",
        "postgresql",
        "mysql",
        "sqlite",
        "mongodb",
        "redis",
        "cassandra",
        "kafka",
        "spark",
        "hadoop",
        "hive",
        "presto",
        "trino",
        "snowflake",
        "databricks",
        "dbt",
        "elasticsearch",
        "airflow",
        "luigi",
        "airbyte",
        "delta_lake",
        "tableau",
        "powerbi",
        "looker",
        "superset",
        "metabase",
        "bigquery",
        "redshift",
    ],
    "cloud_devops": [
        "aws",
        "azure",
        "gcp",
        "kubernetes",
        "docker",
        "terraform",
        "ansible",
        "helm",
        "jenkins",
        "github_actions",
        "gitlab_ci",
        "circleci",
        "argo_cd",
        "openshift",
        "nomad",
        "prometheus",
        "grafana",
        "datadog",
        "new_relic",
        "splunk",
        "opentelemetry",
        "linux",
        "bash",
        "git",
        "serverless",
        "cloudformation",
        "pulumi",
    ],
    "ai_ml": [
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
        "pandas",
        "numpy",
        "jupyter",
        "xgboost",
        "lightgbm",
        "catboost",
        "mlflow",
        "kubeflow",
        "ray",
        "hugging_face",
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
    ],
    "testing_quality": [
        "junit",
        "pytest",
        "jest",
        "mocha",
        "chai",
        "selenium",
        "cypress",
        "playwright",
        "tdd",
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
    ],
    "language_general": [
        "python",
        "java",
        "javascript",
        "typescript",
        "go",
        "rust",
        "c_plus_plus",
        "c_sharp",
        "ruby",
        "kotlin",
        "swift",
        "scala",
        "php",
        "react",
        "angular",
        "vue",
        "nextjs",
        "nodejs",
        "django",
        "flask",
        "spring",
        "dotnet",
        "rails",
        "fastapi",
        "aws",
        "azure",
        "gcp",
        "kubernetes",
        "docker",
        "terraform",
        "postgresql",
        "mysql",
        "mongodb",
        "redis",
        "kafka",
        "spark",
        "snowflake",
        "databricks",
        "dbt",
        "elasticsearch",
        "tensorflow",
        "pytorch",
        "pandas",
        "numpy",
    ],
}

TECH_CATEGORIES = {
    "language": {
        "python",
        "java",
        "javascript",
        "typescript",
        "go",
        "rust",
        "c_plus_plus",
        "c_sharp",
        "ruby",
        "kotlin",
        "swift",
        "scala",
        "php",
    },
    "frontend": {
        "react",
        "react_native",
        "angular",
        "vue",
        "nextjs",
        "svelte",
        "redux",
        "webpack",
        "vite",
        "storybook",
        "tailwind",
        "bootstrap",
        "material_ui",
    },
    "backend": {
        "nodejs",
        "express",
        "nestjs",
        "django",
        "django_rest_framework",
        "flask",
        "spring",
        "spring_boot",
        "dotnet",
        "aspnet",
        "rails",
        "laravel",
        "fastapi",
        "phoenix",
        "tornado",
        "bottle",
        "grpc",
        "graphql",
        "rest_api",
        "microservices",
        "event_driven",
        "oauth",
    },
    "data": {
        "sql",
        "postgresql",
        "mysql",
        "sqlite",
        "mongodb",
        "redis",
        "cassandra",
        "kafka",
        "spark",
        "hadoop",
        "hive",
        "presto",
        "trino",
        "snowflake",
        "databricks",
        "dbt",
        "elasticsearch",
        "airflow",
        "luigi",
        "airbyte",
        "delta_lake",
        "tableau",
        "powerbi",
        "looker",
        "superset",
        "metabase",
        "bigquery",
        "redshift",
    },
    "cloud_devops": {
        "aws",
        "azure",
        "gcp",
        "kubernetes",
        "docker",
        "terraform",
        "ansible",
        "helm",
        "jenkins",
        "github_actions",
        "gitlab_ci",
        "circleci",
        "argo_cd",
        "openshift",
        "nomad",
        "prometheus",
        "grafana",
        "datadog",
        "new_relic",
        "splunk",
        "opentelemetry",
        "linux",
        "bash",
        "git",
        "serverless",
        "cloudformation",
        "pulumi",
    },
    "method_testing": {
        "junit",
        "pytest",
        "jest",
        "mocha",
        "chai",
        "selenium",
        "cypress",
        "playwright",
        "tdd",
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
    },
    "ai_domain": {
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
        "pandas",
        "numpy",
        "jupyter",
        "xgboost",
        "lightgbm",
        "catboost",
        "mlflow",
        "kubeflow",
        "ray",
        "hugging_face",
    },
    "ai_tool": {
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
    },
}


def ensure_dirs() -> None:
    for path in (OUT_REPORT.parent, OUT_TABLE, OUT_FIG):
        path.mkdir(parents=True, exist_ok=True)


def fetch_df(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def load_text_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(
        con,
        f"""
        SELECT uid, description_cleaned, text_source, source, period, seniority_final,
               seniority_3level, is_aggregator, company_name_canonical, yoe_extracted
        FROM read_parquet('{TEXT.as_posix()}')
        """,
    )


def load_tech_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(con, f"SELECT * FROM read_parquet('{TECH.as_posix()}')")


def load_labels_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(con, f"SELECT * FROM read_parquet('{LABELS.as_posix()}')")


def load_index_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(con, f"SELECT * FROM read_parquet('{EMBED_INDEX.as_posix()}')")


def load_raw_sample(con: duckdb.DuckDBPyConnection, uids: list[str]) -> pd.DataFrame:
    if not uids:
        return pd.DataFrame(columns=["uid", "title", "description"])
    values = ", ".join([sql_quote(u) for u in uids])
    return fetch_df(
        con,
        f"""
        SELECT uid, title, description
        FROM read_parquet('{DATA.as_posix()}')
        WHERE uid IN ({values})
        """,
    )


def make_section_regexes() -> list[tuple[str, re.Pattern[str]]]:
    section_phrases = {
        "role_summary": [
            "role summary",
            "summary",
            "about the role",
            "about this role",
            "job summary",
            "position summary",
            "overview",
            "the role",
        ],
        "responsibilities": [
            "responsibilities",
            "responsibility",
            "what you'll do",
            "what you will do",
            "what you'll be doing",
            "duties",
            "your responsibilities",
            "day to day",
            "what the job involves",
        ],
        "requirements": [
            "requirements",
            "requirement",
            "qualifications",
            "qualification",
            "what you'll need",
            "what you will need",
            "what you'll bring",
            "what you bring",
            "minimum qualifications",
            "basic qualifications",
            "required qualifications",
            "essential qualifications",
        ],
        "preferred": [
            "preferred qualifications",
            "preferred",
            "nice to have",
            "nice-to-have",
            "desired experience",
            "bonus qualifications",
            "ideal candidate",
            "desired qualifications",
        ],
        "benefits": [
            "benefits",
            "perks",
            "compensation",
            "salary",
            "pay",
            "equity",
            "bonus",
            "401k",
            "dental",
            "pto",
            "insurance",
            "health insurance",
        ],
        "about_company": [
            "about the company",
            "about us",
            "who we are",
            "company overview",
            "our company",
            "why us",
            "mission",
            "values",
            "employees",
        ],
        "legal": [
            "equal opportunity",
            "equal employment opportunity",
            "eeo",
            "eeoc",
            "accommodation",
            "sponsorship",
            "visa",
            "privacy",
            "disclaimer",
            "background check",
        ],
    }
    regexes: list[tuple[str, re.Pattern[str]]] = []
    for label, phrases in section_phrases.items():
        for phrase in sorted(phrases, key=lambda x: len(x), reverse=True):
            pat = re.compile(
                rf"(?is)^\s*[\*\-\|:>•]*\s*(?:{re.escape(phrase)})\b(?:\s*[:\-\|–—]\s*|\s+|$)(?P<rest>.*)$"
            )
            regexes.append((label, pat))
    return regexes


SECTION_REGEXES = make_section_regexes()
CORE_SECTION_LABELS = {"role_summary", "responsibilities", "requirements", "preferred"}
BOILERPLATE_SECTION_LABELS = {"benefits", "about_company", "legal"}


def extract_sections(text: str) -> list[dict]:
    if not text:
        return []
    parts = re.split(r"(\*\*[^*]{0,120}?\*\*|\|)", text)
    sections: list[dict] = []
    current_label = "unclassified"
    current_parts: list[str] = []
    order = 0

    def flush() -> None:
        nonlocal order, current_parts
        joined = " ".join(p.strip() for p in current_parts if p and p.strip())
        joined = re.sub(r"\s+", " ", joined).strip()
        if joined:
            sections.append(
                {
                    "segment_order": order,
                    "section_label": current_label,
                    "section_text": joined,
                    "section_chars": len(joined),
                }
            )
            order += 1
        current_parts = []

    for part in parts:
        if not part:
            continue
        stripped = part.strip()
        if not stripped or stripped == "|":
            continue
        if stripped.startswith("**") and stripped.endswith("**") and len(stripped) >= 4:
            inner = stripped[2:-2].strip()
            if not inner:
                continue
            header = None
            for label, regex in SECTION_REGEXES:
                m = regex.match(inner)
                if m:
                    header = (label, m.group("rest").strip())
                    break
            if header:
                label, rest = header
                flush()
                current_label = label
                if rest:
                    current_parts.append(rest)
                continue
            current_parts.append(inner)
            continue
        header = None
        for label, regex in SECTION_REGEXES:
            m = regex.match(stripped)
            if m:
                header = (label, m.group("rest").strip())
                break
        if header:
            label, rest = header
            flush()
            current_label = label
            if rest:
                current_parts.append(rest)
            continue
        current_parts.append(stripped)
    flush()
    return sections


def tokenize_normalized(text: str) -> list[str]:
    text = (text or "").lower().replace("’", "'").replace("`", "'")
    text = re.sub(r"(?i)\bc\+\+\b", "cplusplus", text)
    text = re.sub(r"(?i)\bc#\b", "csharp", text)
    text = re.sub(r"(?i)\b\.net\b", "dotnet", text)
    text = re.sub(r"(?i)\bnode\.?js\b", "nodejs", text)
    text = re.sub(r"(?i)\bnext\.?js\b", "nextjs", text)
    text = re.sub(r"(?i)\bci\s*/\s*cd\b", "cicd", text)
    text = re.sub(r"(?i)\bai\s*/\s*ml\b", "aiml", text)
    text = re.sub(r"(?i)\br&d\b", "rnd", text)
    return re.findall(r"[a-z0-9][a-z0-9+#.\-/]{1,}", text)


def normalize_stop_tokens(tokens: list[str]) -> set[str]:
    out = set()
    for tok in tokens:
        t = tok.strip().lower().replace("’", "'").replace("`", "'")
        t = re.sub(r"(?i)\bc\+\+\b", "cplusplus", t)
        t = re.sub(r"(?i)\bc#\b", "csharp", t)
        t = re.sub(r"(?i)\b\.net\b", "dotnet", t)
        t = re.sub(r"(?i)\bnode\.?js\b", "nodejs", t)
        t = re.sub(r"(?i)\bnext\.?js\b", "nextjs", t)
        t = re.sub(r"(?i)\bci\s*/\s*cd\b", "cicd", t)
        t = re.sub(r"(?i)\bai\s*/\s*ml\b", "aiml", t)
        t = re.sub(r"[^a-z0-9\s]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            out.add(t)
    return out


def load_stoplist() -> set[str]:
    raw = {line.strip() for line in STOPLIST.read_text().splitlines() if line.strip()}
    tokens = set()
    for value in raw:
        tokens.update(value.split())
    return normalize_stop_tokens(list(tokens))


def build_category_frame(text_df: pd.DataFrame, tech_df: pd.DataFrame) -> pd.DataFrame:
    df = text_df.merge(tech_df, on="uid", how="inner")
    tech_cols_all = [c for c in tech_df.columns if c != "uid"]
    ai_cols = [c for c in tech_cols_all if c in TECH_FAMILY_MAP["ai_ml"]]
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
    tech_cols = [c for c in tech_cols_all if c not in ai_cols and c not in method_exclude]

    for col in tech_cols_all:
        if col != "uid":
            df[col] = df[col].astype(bool)

    df["text_len"] = df["description_cleaned"].fillna("").str.len()
    df["tech_count"] = df[tech_cols].sum(axis=1).astype(int)
    df["ai_count"] = df[ai_cols].sum(axis=1).astype(int)

    def family_count(names: dict[str, str]) -> pd.Series:
        total = pd.Series(0, index=df.index, dtype=int)
        lower = df["description_cleaned"].fillna("").str.lower()
        for _, pat in names.items():
            total += lower.str.contains(pat, regex=True, na=False).astype(int)
        return total

    df["soft_skill_count"] = family_count(SOFT_SKILL_TERMS)
    df["scope_count"] = family_count(SCOPE_TERMS)
    df["management_strong_count"] = family_count(MGMT_STRONG_TERMS)
    df["management_broad_count"] = family_count(MGMT_BROAD_TERMS)
    lower = df["description_cleaned"].fillna("").str.lower()
    df["education_level"] = np.select(
        [
            lower.str.contains(EDU_LEVELS["phd"], regex=True, na=False),
            lower.str.contains(EDU_LEVELS["ms"], regex=True, na=False),
            lower.str.contains(EDU_LEVELS["bs"], regex=True, na=False),
        ],
        ["phd", "ms", "bs"],
        default="none",
    )
    df["education_flag"] = (df["education_level"] != "none").astype(int)
    df["yoe_flag"] = df["yoe_extracted"].notna().astype(int)
    df["requirement_breadth"] = (
        df["tech_count"]
        + df["soft_skill_count"]
        + df["scope_count"]
        + df["management_strong_count"]
        + df["ai_count"]
        + df["education_flag"]
        + df["yoe_flag"]
    )
    df["credential_stack_depth"] = (
        (df["tech_count"] > 0).astype(int)
        + (df["soft_skill_count"] > 0).astype(int)
        + (df["scope_count"] > 0).astype(int)
        + (df["management_strong_count"] > 0).astype(int)
        + (df["ai_count"] > 0).astype(int)
        + df["education_flag"]
        + df["yoe_flag"]
    )
    df["tech_density"] = df["tech_count"] / (df["text_len"].clip(lower=1) / 1000.0)
    df["scope_density"] = df["scope_count"] / (df["text_len"].clip(lower=1) / 1000.0)
    df["ai_any"] = df["ai_count"] > 0
    df["scope_any"] = df["scope_count"] > 0
    df["management_strong_any"] = df["management_strong_count"] > 0
    df["management_broad_any"] = df["management_broad_count"] > 0
    df["company_rank"] = (
        df.sort_values(["source", "period", "company_name_canonical", "uid"])
        .groupby(["source", "period", "company_name_canonical"])
        .cumcount()
        .reindex(df.index)
        .astype(int)
        + 1
    )
    return df


def regex_precision_sample(df: pd.DataFrame, category: str, term_map: dict[str, str], seed: int = 42) -> pd.DataFrame:
    if category == "ai":
        mask = df["ai_any"]
    elif category == "scope":
        mask = df["scope_any"]
    elif category == "management_strong":
        mask = df["management_strong_any"]
    elif category == "management_broad":
        mask = df["management_broad_any"]
    else:
        raise ValueError(category)

    sample = df.loc[mask, ["uid", "source", "period", "company_name_canonical", "description_cleaned"]].copy()
    sample["year_bucket"] = sample["period"].map(PERIOD_BUCKETS)

    frames = []
    for bucket, target_n in [("2024", 25), ("2026", 25)]:
        part = sample[sample["year_bucket"] == bucket].copy()
        if part.empty:
            continue
        take = min(target_n, len(part))
        frames.append(part.sample(n=take, random_state=seed))
    out = pd.concat(frames, ignore_index=True) if frames else sample.head(0).copy()

    def hits(text: str) -> str:
        txt = (text or "").lower()
        found = []
        for name, pat in term_map.items():
            if re.search(pat, txt, flags=re.I):
                found.append(name)
        return ", ".join(found)

    out["matched_terms"] = out["description_cleaned"].map(hits)
    out["excerpt"] = out["description_cleaned"].fillna("").str.replace(r"\s+", " ", regex=True).str.slice(0, 240)
    out["category"] = category
    return out[["category", "uid", "source", "period", "company_name_canonical", "matched_terms", "excerpt"]]


def summarize_metric(frame: pd.DataFrame, metric: str, group_cols: list[str]) -> pd.DataFrame:
    out = (
        frame.groupby(group_cols, dropna=False)[metric]
        .agg(n="size", mean="mean", median="median", p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75))
        .reset_index()
    )
    out.insert(len(group_cols), "metric", metric)
    return out


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    x = x.dropna().astype(float)
    y = y.dropna().astype(float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2)
    if pooled <= 0:
        return float("nan")
    return float((x.mean() - y.mean()) / math.sqrt(pooled))


def tech_domain_frame(text_df: pd.DataFrame, tech_df: pd.DataFrame) -> pd.DataFrame:
    joined = text_df.merge(tech_df, on="uid", how="inner")
    tech_cols = [c for c in tech_df.columns if c != "uid"]
    family_order = ["ai_ml", "data_platform", "frontend_web", "backend_api", "cloud_devops", "testing_quality", "language_general"]
    family_frame = pd.DataFrame({"uid": joined["uid"]})
    for family, cols in TECH_FAMILY_MAP.items():
        cols = [c for c in cols if c in tech_cols]
        if cols:
            family_frame[family] = joined[cols].fillna(False).astype(bool).sum(axis=1).astype(int)
        else:
            family_frame[family] = 0
    family_frame["family_max"] = family_frame[family_order].max(axis=1)
    domain = []
    for _, row in family_frame.iterrows():
        if row["family_max"] <= 0:
            domain.append("none")
            continue
        winners = [fam for fam in family_order if row[fam] == row["family_max"] and row[fam] > 0]
        domain.append(winners[0] if winners else "none")
    family_frame["tech_domain"] = domain
    return text_df[["uid", "source", "period", "text_source", "seniority_3level", "is_aggregator", "company_name_canonical"]].merge(
        family_frame[["uid", "tech_domain"]], on="uid", how="left"
    )


def t09_nmi_summary(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    joined = labels.merge(frame, on="uid", how="inner")
    rows = []
    y = {
        "tech_domain": joined["tech_domain"].astype(str),
        "seniority_3level": joined["seniority_3level"].astype(str),
        "year_period": joined["period"].str.slice(0, 4),
        "source": joined["source"].astype(str),
        "text_source": joined["text_source"].astype(str),
        "is_aggregator": joined["is_aggregator"].astype(str),
    }
    x = joined["archetype"].astype(str)
    for label, target in y.items():
        rows.append({"label": label, "nmi": normalized_mutual_info_score(x, target), "n": len(joined)})
    return pd.DataFrame(rows).sort_values("nmi", ascending=False)


def t09_archetype_shares(labels: pd.DataFrame, text_df: pd.DataFrame) -> pd.DataFrame:
    joined = labels.merge(text_df[["uid", "period"]], on="uid", how="inner")
    joined["year_period"] = joined["period"].str.slice(0, 4)
    out = (
        joined.groupby(["archetype", "archetype_name"], dropna=False)
        .agg(
            n=("uid", "size"),
            period_2024_share=("year_period", lambda s: (s == "2024").mean()),
            period_2026_share=("year_period", lambda s: (s == "2026").mean()),
        )
        .reset_index()
    )
    out["share"] = out["n"] / len(joined)
    return out.sort_values("n", ascending=False)


def t13_section_summary(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = fetch_df(
        con,
        f"""
        SELECT
          u.uid,
          u.source,
          u.period,
          u.company_name_canonical,
          u.description_length,
          u.description_core_llm AS cleaned_text
        FROM read_parquet('{DATA.as_posix()}') u
        WHERE u.source_platform='linkedin'
          AND u.is_english = true
          AND u.date_flag = 'ok'
          AND u.is_swe = true
          AND u.description_core_llm IS NOT NULL
        """,
    )
    df = df.sort_values(["source", "period", "company_name_canonical", "uid"]).reset_index(drop=True)
    df["company_rank"] = df.groupby(["source", "period", "company_name_canonical"]).cumcount() + 1
    df = df[df["company_rank"] <= 25].copy()
    parsed_rows = []
    for row in df[["uid", "source", "period", "cleaned_text", "description_length"]].itertuples(index=False):
        sections = extract_sections(row.cleaned_text)
        core_chars = sum(int(seg["section_chars"]) for seg in sections if seg["section_label"] in CORE_SECTION_LABELS)
        boiler_chars = sum(int(seg["section_chars"]) for seg in sections if seg["section_label"] in BOILERPLATE_SECTION_LABELS)
        total_chars = sum(int(seg["section_chars"]) for seg in sections)
        parsed_rows.append(
            {
                "uid": row.uid,
                "source": row.source,
                "period": row.period,
                "doc_len": int(row.description_length),
                "has_core_section": any(seg["section_label"] in CORE_SECTION_LABELS for seg in sections),
                "core_chars": core_chars,
                "boilerplate_chars": boiler_chars,
                "parsed_chars": total_chars,
                "core_share": core_chars / max(1, int(row.description_length)),
                "boilerplate_share": boiler_chars / max(1, int(row.description_length)),
            }
        )
    parsed = pd.DataFrame(parsed_rows)
    summary = (
        parsed.groupby(["source", "period"], dropna=False)
        .agg(
            n=("uid", "size"),
            mean_doc_len=("doc_len", "mean"),
            share_has_core_section=("has_core_section", "mean"),
            mean_core_share=("core_share", "mean"),
            mean_boilerplate_share=("boilerplate_share", "mean"),
        )
        .reset_index()
    )
    return parsed, summary


def t14_ai_share(text_df: pd.DataFrame, tech_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tech_cols = [c for c in tech_df.columns if c != "uid"]
    ai_cols = [
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
    ]
    ai_cols = [c for c in ai_cols if c in tech_cols]
    df = text_df.merge(tech_df, on="uid", how="inner").copy()
    df["ai_any"] = df[ai_cols].astype(bool).any(axis=1)
    df["company_rank"] = (
        df.sort_values(["source", "period", "company_name_canonical", "uid"])
        .groupby(["source", "period", "company_name_canonical"])
        .cumcount()
        .reindex(df.index)
        .astype(int)
        + 1
    )
    base = df.groupby("source", dropna=False).agg(n=("uid", "size"), ai_any_share=("ai_any", "mean")).reset_index()
    noagg = df.loc[~df["is_aggregator"]].groupby("source", dropna=False).agg(noagg_n=("uid", "size"), noagg_ai_any_share=("ai_any", "mean")).reset_index()
    out = base.merge(noagg, on="source", how="left")
    counts = df.groupby(["source", "period"], dropna=False).agg(n=("uid", "size"), ai_any_share=("ai_any", "mean"), aggregators=("is_aggregator", "sum")).reset_index()
    return out, counts


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return matrix / denom


def trimmed_centroid(x: np.ndarray, trim_frac: float = 0.10) -> np.ndarray:
    if x.shape[0] == 0:
        raise ValueError("empty group")
    if x.shape[0] == 1:
        return x[0]
    x = l2_normalize(np.asarray(x, dtype=np.float32))
    centroid = x.mean(axis=0)
    dists = 1.0 - (x @ centroid) / (np.linalg.norm(x, axis=1) * np.linalg.norm(centroid) + 1e-12)
    keep_n = max(1, int(math.ceil((1.0 - trim_frac) * x.shape[0])))
    keep_idx = np.argsort(dists)[:keep_n]
    return x[keep_idx].mean(axis=0)


def centroid_similarity(centroids: dict[str, np.ndarray]) -> pd.DataFrame:
    keys = sorted(centroids)
    rows = []
    for a in keys:
        row = {"group": a}
        for b in keys:
            denom = float(np.linalg.norm(centroids[a]) * np.linalg.norm(centroids[b]))
            row[b] = float(np.dot(centroids[a], centroids[b]) / (denom + 1e-12))
        rows.append(row)
    return pd.DataFrame(rows)


def source_convergence(frame: pd.DataFrame, vectors: np.ndarray, rep: str) -> pd.DataFrame:
    rows = []
    centroids = {}
    for source in ["kaggle_asaniczka", "kaggle_arshkon", "scraped"]:
        for sen in ["junior", "mid", "senior", "unknown"]:
            mask = (frame["source"] == source) & (frame["seniority_3level"] == sen)
            if mask.sum() == 0:
                continue
            centroids[(source, sen)] = trimmed_centroid(vectors[mask.to_numpy()], trim_frac=0.10)
    for source in ["kaggle_asaniczka", "kaggle_arshkon", "scraped"]:
        if (source, "junior") in centroids and (source, "senior") in centroids:
            a = centroids[(source, "junior")]
            b = centroids[(source, "senior")]
            rows.append(
                {
                    "representation": rep,
                    "source": source,
                    "metric": "junior_senior_similarity",
                    "value": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)),
                }
            )
        if (source, "junior") in centroids and (source, "mid") in centroids:
            a = centroids[(source, "junior")]
            b = centroids[(source, "mid")]
            rows.append(
                {
                    "representation": rep,
                    "source": source,
                    "metric": "junior_mid_similarity",
                    "value": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)),
                }
            )
        if (source, "mid") in centroids and (source, "senior") in centroids:
            a = centroids[(source, "mid")]
            b = centroids[(source, "senior")]
            rows.append(
                {
                    "representation": rep,
                    "source": source,
                    "metric": "mid_senior_similarity",
                    "value": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)),
                }
            )
    return pd.DataFrame(rows)


def group_similarity_matrix(frame: pd.DataFrame, vectors: np.ndarray, rep: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = frame["period"] + "|" + frame["seniority_3level"]
    centroids = {}
    for label in sorted(labels.unique()):
        mask = labels == label
        centroids[label] = trimmed_centroid(vectors[mask.to_numpy()], trim_frac=0.10)
    matrix = centroid_similarity(centroids)
    matrix.insert(1, "representation", rep)
    disp_rows = []
    for label in sorted(labels.unique()):
        mask = labels == label
        X = vectors[mask.to_numpy()]
        c = centroids[label]
        sims = (X @ c) / (np.linalg.norm(X, axis=1) * np.linalg.norm(c) + 1e-12)
        disp_rows.append(
            {
                "representation": rep,
                "group": label,
                "n": int(len(X)),
                "mean_cosine_distance": float((1 - sims).mean()),
                "p90_cosine_distance": float(np.quantile(1 - sims, 0.90)),
            }
        )
    return matrix, pd.DataFrame(disp_rows)


def tfidf_convergence(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stoplist = load_stoplist()

    def tokenizer(doc: str) -> list[str]:
        text = (doc or "").lower()
        text = re.sub(r"(?i)\bc\+\+\b", "cplusplus", text)
        text = re.sub(r"(?i)\bc#\b", "csharp", text)
        text = re.sub(r"(?i)\b\.net\b", "dotnet", text)
        text = re.sub(r"(?i)\bnode\.?js\b", "nodejs", text)
        text = re.sub(r"(?i)\bnext\.?js\b", "nextjs", text)
        text = re.sub(r"(?i)\bci\s*/\s*cd\b", "cicd", text)
        text = re.sub(r"(?i)\bai\s*/\s*ml\b", "aiml", text)
        text = re.sub(r"(?i)\br&d\b", "rnd", text)
        toks = re.findall(r"[a-z0-9][a-z0-9+#.\-/]{1,}", text)
        return [t for t in toks if t not in stoplist]

    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        min_df=5,
        max_features=25000,
        ngram_range=(1, 2),
        norm="l2",
    )
    tfidf = vectorizer.fit_transform(frame["description_cleaned"].fillna(""))
    tfidf_reduced = normalize(TruncatedSVD(n_components=100, random_state=42).fit_transform(tfidf))
    matrix, disp = group_similarity_matrix(frame, tfidf_reduced, "tfidf")
    src = source_convergence(frame, tfidf_reduced, "tfidf")
    return matrix, disp, src


def build_yoe_proxy_sims(frame: pd.DataFrame, vectors: np.ndarray, rep: str) -> pd.DataFrame:
    out = []
    mask_j = frame["yoe_extracted"].fillna(99).le(2)
    mask_s = frame["yoe_extracted"].fillna(-1).ge(3)
    for source in ["kaggle_asaniczka", "kaggle_arshkon", "scraped"]:
        mj = mask_j & (frame["source"] == source)
        ms = mask_s & (frame["source"] == source)
        if mj.sum() and ms.sum():
            a = trimmed_centroid(vectors[mj.to_numpy()], trim_frac=0.10)
            b = trimmed_centroid(vectors[ms.to_numpy()], trim_frac=0.10)
            out.append(
                {
                    "representation": rep,
                    "source": source,
                    "metric": "yoe_junior_senior_similarity",
                    "value": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)),
                }
            )
    return pd.DataFrame(out)


def main() -> None:
    ensure_dirs()
    con = duckdb.connect()

    text_df = load_text_frame(con)
    tech_df = load_tech_frame(con)
    labels_df = load_labels_frame(con)
    idx_df = load_index_frame(con)

    # Align embeddings and text for the llm-only semantic frame.
    emb = np.load(EMBED, mmap_mode="r")
    llm_text = text_df[text_df["text_source"] == "llm"].copy()
    llm_index = idx_df.merge(llm_text, on="uid", how="inner").sort_values("row_index").reset_index(drop=True)
    vectors = l2_normalize(np.asarray(emb, dtype=np.float32))[llm_index["row_index"].to_numpy()]

    # --- T09 ---
    tech_domain_df = tech_domain_frame(text_df, tech_df)
    t09_nmi = t09_nmi_summary(tech_domain_df, labels_df)
    t09_shares = t09_archetype_shares(labels_df, text_df)
    t09_ai = t09_shares.loc[t09_shares["archetype_name"] == "AI / LLM workflows"].copy()
    t09_ai["period_2026_share"] = t09_ai["period_2026_share"].astype(float)
    t09_ai["period_2024_share"] = t09_ai["period_2024_share"].astype(float)
    t09_nmi.to_csv(OUT_TABLE / "V1_t09_nmi_summary.csv", index=False)
    t09_shares.to_csv(OUT_TABLE / "V1_t09_archetype_shares.csv", index=False)

    # --- T11 ---
    cat_df = build_category_frame(text_df, tech_df)
    primary_llm = cat_df[(cat_df["text_source"] == "llm") & (cat_df["company_rank"] <= 25)].copy()
    primary_llm_noagg = primary_llm[~primary_llm["is_aggregator"]].copy()
    primary_raw = cat_df[(cat_df["text_source"] == "raw") & (cat_df["company_rank"] <= 25)].copy()

    metrics = [
        "tech_count",
        "ai_count",
        "soft_skill_count",
        "scope_count",
        "management_strong_count",
        "management_broad_count",
        "education_flag",
        "yoe_flag",
        "requirement_breadth",
        "credential_stack_depth",
        "text_len",
        "tech_density",
        "scope_density",
    ]
    summary_llm = pd.concat([summarize_metric(primary_llm, m, ["source", "period", "seniority_final"]) for m in metrics], ignore_index=True)
    summary_raw = pd.concat([summarize_metric(primary_raw, m, ["source", "period", "seniority_final"]) for m in metrics], ignore_index=True)
    summary_noagg = pd.concat([summarize_metric(primary_llm_noagg, m, ["source", "period", "seniority_final"]) for m in metrics], ignore_index=True)
    summary_llm.to_csv(OUT_TABLE / "V1_t11_summary_llm_cap25.csv", index=False)
    summary_raw.to_csv(OUT_TABLE / "V1_t11_summary_raw_cap25.csv", index=False)
    summary_noagg.to_csv(OUT_TABLE / "V1_t11_summary_noagg_llm_cap25.csv", index=False)

    entry_comp = (
        pd.concat(
            [
                primary_llm.assign(entry_definition="explicit_entry", period_group=primary_llm["source"].map({"kaggle_arshkon": "2024-arshkon", "scraped": "2026-scraped"})),
                primary_llm.assign(
                    entry_definition="yoe_proxy_le2",
                    period_group=np.where(primary_llm["source"] == "kaggle_arshkon", "2024-arshkon", "2026-scraped"),
                ),
            ]
        )
    )
    entry_rows = []
    for entry_def, subset in [
        ("explicit_entry", primary_llm[primary_llm["seniority_final"] == "entry"].copy()),
        ("yoe_proxy_le2", primary_llm[primary_llm["yoe_extracted"].fillna(99).le(2)].copy()),
    ]:
        for period_group, g in subset.groupby(np.where(subset["source"] == "kaggle_arshkon", "2024-arshkon", "2026-scraped")):
            entry_rows.append(
                {
                    "entry_definition": entry_def,
                    "period_group": period_group,
                    "n": len(g),
                    "mean_requirement_breadth": g["requirement_breadth"].mean(),
                    "mean_stack_depth": g["credential_stack_depth"].mean(),
                    "mean_tech_count": g["tech_count"].mean(),
                    "mean_tech_density": g["tech_density"].mean(),
                    "mean_scope_density": g["scope_density"].mean(),
                }
            )
    entry_comp_df = pd.DataFrame(entry_rows)
    entry_comp_df.to_csv(OUT_TABLE / "V1_t11_entry_comparison.csv", index=False)

    calib_rows = []
    baseline = primary_llm[(primary_llm["source"] == "kaggle_arshkon") & (primary_llm["period"] == "2024-04")]
    alt_2024 = primary_llm[(primary_llm["source"] == "kaggle_asaniczka") & (primary_llm["period"] == "2024-01")]
    current = primary_llm[primary_llm["source"] == "scraped"]
    for metric in metrics:
        b = baseline[metric]
        a = alt_2024[metric]
        c = current[metric]
        calib_rows.append(
            {
                "metric": metric,
                "baseline_mean": b.mean(),
                "alt_2024_mean": a.mean(),
                "current_mean": c.mean(),
                "within_2024_diff": a.mean() - b.mean(),
                "cross_period_diff": c.mean() - b.mean(),
                "within_2024_d": cohens_d(a, b),
                "cross_period_d": cohens_d(c, b),
                "signal_to_noise": abs(cohens_d(c, b)) / max(abs(cohens_d(a, b)), 1e-12),
            }
        )
    calib_df = pd.DataFrame(calib_rows)
    calib_df.to_csv(OUT_TABLE / "V1_t11_calibration.csv", index=False)

    sample_frames = []
    for cat, terms in [
        ("ai", {"llm": r"\bllm(s)?\b", "openai_api": r"\bopenai api\b", "anthropic_api": r"\banthropic api\b", "claude_api": r"\bclaude api\b", "gemini_api": r"\bgemini api\b", "prompt_engineering": r"prompt engineering", "fine_tuning": r"fine[- ]tuning", "mcp": r"\bmcp\b", "agent": r"\bagent(s|ic)?\b", "copilot": r"\bcopilot\b", "cursor": r"\bcursor\b", "chatgpt": r"\bchatgpt\b", "claude": r"\bclaude\b", "gemini": r"\bgemini\b", "codex": r"\bcodex\b", "langchain": r"\blangchain\b", "langgraph": r"\blanggraph\b", "llamaindex": r"\bllamaindex\b", "rag": r"\brag\b", "vector_db": r"vector (db|database|databases)", "pinecone": r"\bpinecone\b", "weaviate": r"\bweaviate\b", "chroma": r"\bchroma\b", "milvus": r"\bmilvus\b", "faiss": r"\bfaiss\b"}),
        ("scope", SCOPE_TERMS),
        ("management_strong", MGMT_STRONG_TERMS),
        ("management_broad", MGMT_BROAD_TERMS),
    ]:
        sample_frames.append(regex_precision_sample(primary_llm, cat, terms))
    precision_sample = pd.concat(sample_frames, ignore_index=True)
    precision_sample.to_csv(OUT_TABLE / "V1_keyword_precision_samples.csv", index=False)

    # --- T13 ---
    parsed_sections, section_summary = t13_section_summary(con)
    section_summary.to_csv(OUT_TABLE / "V1_t13_section_summary.csv", index=False)
    parsed_sections.to_csv(OUT_TABLE / "V1_t13_section_docs.csv", index=False)

    # --- T14 ---
    t14_share, t14_counts = t14_ai_share(text_df, tech_df)
    t14_share.to_csv(OUT_TABLE / "V1_t14_ai_share.csv", index=False)
    t14_counts.to_csv(OUT_TABLE / "V1_t14_ai_share_by_source_period.csv", index=False)

    # --- T15 ---
    llm_text = llm_index[["uid", "source", "period", "seniority_final", "seniority_3level", "is_aggregator", "company_name_canonical", "yoe_extracted", "description_cleaned", "row_index"]].copy()
    embed_matrix, embed_disp = group_similarity_matrix(llm_text, vectors, "embedding")
    embed_source = source_convergence(llm_text, vectors, "embedding")
    embed_yoe = build_yoe_proxy_sims(llm_text, vectors, "embedding")

    stoplist = load_stoplist()

    def tokenizer(doc: str) -> list[str]:
        text = (doc or "").lower()
        text = re.sub(r"(?i)\bc\+\+\b", "cplusplus", text)
        text = re.sub(r"(?i)\bc#\b", "csharp", text)
        text = re.sub(r"(?i)\b\.net\b", "dotnet", text)
        text = re.sub(r"(?i)\bnode\.?js\b", "nodejs", text)
        text = re.sub(r"(?i)\bnext\.?js\b", "nextjs", text)
        text = re.sub(r"(?i)\bci\s*/\s*cd\b", "cicd", text)
        text = re.sub(r"(?i)\bai\s*/\s*ml\b", "aiml", text)
        text = re.sub(r"(?i)\br&d\b", "rnd", text)
        toks = re.findall(r"[a-z0-9][a-z0-9+#.\-/]{1,}", text)
        return [t for t in toks if t not in stoplist]

    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        min_df=5,
        max_features=25000,
        ngram_range=(1, 2),
        norm="l2",
    )
    tfidf = vectorizer.fit_transform(llm_text["description_cleaned"].fillna(""))
    tfidf_reduced = normalize(TruncatedSVD(n_components=100, random_state=42).fit_transform(tfidf))
    tfidf_matrix, tfidf_disp = group_similarity_matrix(llm_text, tfidf_reduced, "tfidf")
    tfidf_source = source_convergence(llm_text, tfidf_reduced, "tfidf")
    tfidf_yoe = build_yoe_proxy_sims(llm_text, tfidf_reduced, "tfidf")

    embed_source.to_csv(OUT_TABLE / "V1_t15_source_convergence_embedding.csv", index=False)
    tfidf_source.to_csv(OUT_TABLE / "V1_t15_source_convergence_tfidf.csv", index=False)
    embed_yoe.to_csv(OUT_TABLE / "V1_t15_yoe_proxy_embedding.csv", index=False)
    tfidf_yoe.to_csv(OUT_TABLE / "V1_t15_yoe_proxy_tfidf.csv", index=False)
    embed_disp.to_csv(OUT_TABLE / "V1_t15_dispersion_embedding.csv", index=False)
    tfidf_disp.to_csv(OUT_TABLE / "V1_t15_dispersion_tfidf.csv", index=False)
    embed_matrix.to_csv(OUT_TABLE / "V1_t15_similarity_embedding.csv", index=False)
    tfidf_matrix.to_csv(OUT_TABLE / "V1_t15_similarity_tfidf.csv", index=False)

    # Sample raw snippets for precision review.
    uids = precision_sample["uid"].tolist()
    raw = load_raw_sample(con, uids)
    if not raw.empty:
        precision_sample = precision_sample.merge(raw, on="uid", how="left")
        precision_sample.to_csv(OUT_TABLE / "V1_keyword_precision_samples_with_titles.csv", index=False)

    summary = {
        "t09_nmi_tech_domain": float(t09_nmi.loc[t09_nmi["label"] == "tech_domain", "nmi"].iloc[0]),
        "t09_nmi_seniority": float(t09_nmi.loc[t09_nmi["label"] == "seniority_3level", "nmi"].iloc[0]),
        "t09_ai_2026_share": float(t09_ai["period_2026_share"].iloc[0]) if not t09_ai.empty else float("nan"),
        "t11_stack_depth_signal_to_noise": float(calib_df.loc[calib_df["metric"] == "credential_stack_depth", "signal_to_noise"].iloc[0]),
        "t11_scope_density_signal_to_noise": float(calib_df.loc[calib_df["metric"] == "scope_density", "signal_to_noise"].iloc[0]),
        "t13_arshkon_core_share": float(section_summary.loc[(section_summary["source"] == "kaggle_arshkon") & (section_summary["period"] == "2024-04"), "share_has_core_section"].iloc[0]),
        "t13_asaniczka_core_share": float(section_summary.loc[(section_summary["source"] == "kaggle_asaniczka") & (section_summary["period"] == "2024-01"), "share_has_core_section"].iloc[0]),
        "t13_scraped_2026_03_core_share": float(section_summary.loc[(section_summary["source"] == "scraped") & (section_summary["period"] == "2026-03"), "share_has_core_section"].iloc[0]),
        "t13_scraped_2026_04_core_share": float(section_summary.loc[(section_summary["source"] == "scraped") & (section_summary["period"] == "2026-04"), "share_has_core_section"].iloc[0]),
        "t14_ai_share_arshkon": float(t14_share.loc[t14_share["source"] == "kaggle_arshkon", "ai_any_share"].iloc[0]),
        "t14_ai_share_scraped": float(t14_share.loc[t14_share["source"] == "scraped", "ai_any_share"].iloc[0]),
        "t15_embed_arshkon": float(embed_source.loc[(embed_source["source"] == "kaggle_arshkon") & (embed_source["metric"] == "junior_senior_similarity"), "value"].iloc[0]),
        "t15_embed_scraped": float(embed_source.loc[(embed_source["source"] == "scraped") & (embed_source["metric"] == "junior_senior_similarity"), "value"].iloc[0]),
        "t15_tfidf_arshkon": float(tfidf_source.loc[(tfidf_source["source"] == "kaggle_arshkon") & (tfidf_source["metric"] == "junior_senior_similarity"), "value"].iloc[0]),
        "t15_tfidf_scraped": float(tfidf_source.loc[(tfidf_source["source"] == "scraped") & (tfidf_source["metric"] == "junior_senior_similarity"), "value"].iloc[0]),
    }
    pd.DataFrame([summary]).to_csv(OUT_TABLE / "V1_summary_metrics.csv", index=False)

    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
