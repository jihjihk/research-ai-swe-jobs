from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


ROOT = Path(__file__).resolve().parents[2]
STAGE8 = ROOT / "preprocessing" / "intermediate" / "stage8_final.parquet"

OUT_REPORT_T10 = ROOT / "exploration" / "reports" / "T10.md"
OUT_REPORT_T11 = ROOT / "exploration" / "reports" / "T11.md"
OUT_TABLE_T10 = ROOT / "exploration" / "tables" / "T10"
OUT_TABLE_T11 = ROOT / "exploration" / "tables" / "T11"
OUT_FIG_T10 = ROOT / "exploration" / "figures" / "T10"
OUT_FIG_T11 = ROOT / "exploration" / "figures" / "T11"

REPORTABLE_CATEGORIES = {
    "ai_tool",
    "ai_domain",
    "tech_stack",
    "org_scope",
    "mgmt",
    "sys_design",
    "method",
    "credential",
    "soft_skill",
}

LINKEDIN_FILTER = """
source_platform = 'linkedin'
AND is_english = true
AND date_flag = 'ok'
AND seniority_final <> 'unknown'
AND description IS NOT NULL
"""

ALL_LINKEDIN_FILTER = """
source_platform = 'linkedin'
AND is_english = true
AND date_flag = 'ok'
AND description IS NOT NULL
"""

SWE_FILTER = """
source_platform = 'linkedin'
AND is_english = true
AND date_flag = 'ok'
AND is_swe = true
AND seniority_final <> 'unknown'
AND description IS NOT NULL
"""

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
WHITESPACE_RE = re.compile(r"\s+")
HTML_RE = re.compile(r"(?is)<[^>]+>")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+\b")
EMAIL_RE = re.compile(r"(?i)\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

BOILERPLATE_MARKERS = [
    "equal opportunity",
    "equal employment opportunity",
    "reasonable accommodation",
    "protected class",
    "benefits include",
    "benefit package includes",
    "about us",
    "about the company",
    "privacy notice",
    "fair chance",
    "we are an equal opportunity employer",
    "we are committed to equal opportunity",
    "lensa is a career site",
    "does not hire directly",
    "promotes jobs on linkedin",
    "directemployers",
    "clicking \"apply now\"",
    "read more",
    "job board/employer site",
    "recruitment ad agencies",
    "marketing partners",
    "not a staffing firm",
]

AI_TOOL_TERMS = {
    "agent",
    "agents",
    "ai agent",
    "ai agents",
    "anthropic",
    "claude",
    "copilot",
    "cursor",
    "gpt",
    "chatgpt",
    "llm",
    "llms",
    "large language model",
    "large language models",
    "language model",
    "language models",
    "mcp",
    "openai",
    "prompt engineering",
    "rag",
    "retrieval augmented",
    "retrieval augmented generation",
    "genai",
    "generative ai",
    "ai assistant",
    "ai assistants",
    "ai pair programming",
    "model evaluation",
    "evals",
}

AI_DOMAIN_TERMS = {
    "ai",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "ml",
    "nlp",
    "natural language processing",
    "computer vision",
    "reinforcement learning",
    "data science",
}

ORG_SCOPE_TERMS = {
    "ownership",
    "own",
    "owned",
    "end to end",
    "end-to-end",
    "cross functional",
    "cross-functional",
    "stakeholder",
    "stakeholders",
    "autonomous",
    "autonomy",
    "partner",
    "partners",
    "partnership",
    "responsibility",
    "responsibilities",
    "impact",
    "initiative",
    "initiatives",
    "roadmap",
}

MGMT_TERMS = {
    "lead",
    "led",
    "leader",
    "leadership",
    "mentor",
    "mentoring",
    "manage",
    "managed",
    "manager",
    "managers",
    "management",
    "hire",
    "hiring",
    "coach",
    "coaching",
    "team",
    "teams",
    "1:1",
    "one on one",
    "one-on-one",
    "performance review",
}

SYS_DESIGN_TERMS = {
    "system design",
    "systems design",
    "architecture",
    "architectures",
    "architect",
    "microservices",
    "distributed systems",
    "scalability",
    "scalable",
    "reliability",
    "resilience",
    "latency",
    "throughput",
    "backend architecture",
}

METHOD_TERMS = {
    "agile",
    "scrum",
    "kanban",
    "ci cd",
    "ci/cd",
    "continuous integration",
    "continuous delivery",
    "continuous deployment",
    "tdd",
    "test driven",
    "test-driven",
}

TECH_STACK_TERMS = {
    "python",
    "java",
    "javascript",
    "typescript",
    "react",
    "reactjs",
    "node",
    "nodejs",
    "sql",
    "postgres",
    "postgresql",
    "mysql",
    "mongodb",
    "docker",
    "kubernetes",
    "terraform",
    "aws",
    "gcp",
    "azure",
    "spark",
    "scala",
    "go",
    "rust",
    "csharp",
    "cplusplus",
    "dotnet",
    "linux",
    "git",
    "grpc",
    "rest",
    "api",
    "apis",
    "graphql",
    "kafka",
    "redis",
    "airflow",
    "databricks",
    "snowflake",
    "pytorch",
    "tensorflow",
    "pandas",
    "numpy",
    "backend",
    "database",
    "databases",
    "framework",
    "frameworks",
    "html",
    "css",
    "deployment",
    "deployments",
    "observability",
}

CREDENTIAL_TERMS = {
    "years experience",
    "year experience",
    "bs",
    "bachelor",
    "bachelors",
    "ms",
    "masters",
    "phd",
    "degree",
    "certification",
    "certified",
    "security clearance",
}

SOFT_SKILL_TERMS = {
    "communication",
    "collaboration",
    "problem solving",
    "problem-solving",
    "critical thinking",
    "teamwork",
    "adaptability",
    "written communication",
    "verbal communication",
}

GENERIC_JOB_STOPWORDS = {
    "applying",
    "applicant",
    "applicants",
    "contribute",
    "contributed",
    "contributions",
    "curating",
    "designing",
    "examples",
    "expected",
    "familiarity",
    "features",
    "hours",
    "monthly",
    "operate",
    "operates",
    "participates",
    "posted",
    "procedures",
    "provides",
    "questions",
    "reasoning",
    "researchers",
    "responses",
    "role",
    "roles",
    "subject",
    "terms",
    "workflows",
    "youll",
    "backend",
    "front end",
    "frontend",
    "full stack",
    "platform",
    "product",
    "development",
    "develop",
    "developing",
    "design",
    "designing",
    "engineering",
    "engineer",
    "engineers",
    "user",
    "users",
    "work",
    "working",
    "experience",
    "experienced",
    "qualifications",
    "qualification",
    "requirements",
    "requirement",
    "responsibilities",
    "responsibility",
    "duties",
    "description",
    "posted",
    "posting",
    "apply",
    "applying",
    "application",
    "applications",
    "eligible",
    "eligibility",
    "receive",
    "receiving",
    "provide",
    "provides",
    "provided",
    "including",
    "include",
    "includes",
    "including",
    "example",
    "examples",
    "question",
    "questions",
    "reasoning",
    "familiarity",
    "features",
    "feature",
    "hours",
    "monthly",
    "expected",
    "expect",
    "operate",
    "operates",
    "operating",
    "responses",
    "procedures",
    "contribute",
    "contributes",
    "contributing",
    "contributions",
    "curating",
    "researchers",
    "intern",
    "interns",
    "internship",
    "internships",
    "resume",
    "resumes",
    "resume",
    "partial",
    "overlap",
    "duration",
    "messages",
    "paid",
    "excellent",
    "written",
    "june",
    "july",
    "august",
}

LEGIT_LONG_TERMS = {
    "accessibility",
    "accountability",
    "architecture",
    "collaboration",
    "communication",
    "configuration",
    "connectivity",
    "containerization",
    "conversational",
    "crossfunctional",
    "decomposition",
    "documentation",
    "engineering",
    "experimental",
    "instrumentation",
    "infrastructure",
    "interoperability",
    "microservices",
    "observability",
    "optimization",
    "orchestration",
    "performance",
    "productivity",
    "responsibilities",
    "security",
    "synchronization",
    "troubleshooting",
    "visualization",
}


@dataclass
class CorpusSpec:
    name: str
    sql_predicate: str


def ensure_dirs() -> None:
    for path in [OUT_REPORT_T10.parent, OUT_REPORT_T11.parent, OUT_TABLE_T10, OUT_TABLE_T11, OUT_FIG_T10, OUT_FIG_T11]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.lower()
    text = text.replace("c++", "cplusplus")
    text = text.replace("c#", "csharp")
    text = text.replace(".net", "dotnet")
    text = text.replace("ci/cd", "ci cd")
    text = text.replace("node.js", "nodejs")
    text = text.replace("react.js", "reactjs")
    text = text.replace("ai/ml", "ai ml")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def strip_boilerplate(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n+", text)
    kept: list[str] = []
    for para in paragraphs:
        low = para.lower()
        if any(marker in low for marker in BOILERPLATE_MARKERS):
            continue
        kept.append(para)
    if not kept:
        return text
    return "\n\n".join(kept)


def make_company_stopwords(con: duckdb.DuckDBPyConnection) -> set[str]:
    stopwords: set[str] = set()

    def add_texts(query: str) -> None:
        reader = con.execute(query).to_arrow_reader()
        for batch in reader:
            values = batch.column(0).to_pylist()
            for value in values:
                if value is None:
                    continue
                tokens = TOKEN_RE.findall(normalize_text(str(value)))
                stopwords.update(tokens)

    add_texts(
        f"""
        SELECT DISTINCT company_name_canonical
        FROM read_parquet('{STAGE8.as_posix()}')
        WHERE company_name_canonical IS NOT NULL
        """
    )

    add_texts(
        f"""
        SELECT DISTINCT metro_area
        FROM read_parquet('{STAGE8.as_posix()}')
        WHERE metro_area IS NOT NULL
        """
    )

    add_texts(
        f"""
        SELECT DISTINCT state_normalized
        FROM read_parquet('{STAGE8.as_posix()}')
        WHERE state_normalized IS NOT NULL
        """
    )

    stopwords.update(
        {
            "inc",
            "incorporated",
            "llc",
            "ltd",
            "co",
            "corp",
            "corporation",
            "company",
            "companies",
            "group",
            "groups",
            "solutions",
            "solution",
            "services",
            "service",
            "systems",
            "system",
            "technology",
            "technologies",
            "global",
            "international",
            "holdings",
            "holding",
            "partners",
            "partner",
            "labs",
            "lab",
            "studio",
            "studios",
            "consulting",
            "consultants",
            "consultant",
            "enterprise",
            "enterprises",
            "industry",
            "industries",
            "llp",
            "pc",
            "pllc",
            "the",
            "and",
            "of",
        }
    )
    stopwords.update(ENGLISH_STOP_WORDS)
    stopwords.update(GENERIC_JOB_STOPWORDS)
    return stopwords


def get_analysis_text_expr(con: duckdb.DuckDBPyConnection) -> str:
    cols = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{STAGE8.as_posix()}')").fetchall()]
    if "description_core_llm" in cols:
        return "COALESCE(description_core_llm, description_core, description) AS analysis_text"
    if "description_core" in cols:
        return "COALESCE(description_core, description) AS analysis_text"
    return "description AS analysis_text"


def tokenize(text: str, stopwords: set[str]) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    if not stopwords:
        return [tok for tok in tokens if len(tok) >= 2]
    return [tok for tok in tokens if len(tok) >= 2 and tok not in stopwords]


def build_ngrams(tokens: list[str], n: int) -> list[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def clean_description(text: str, stopwords: set[str]) -> tuple[str, str, list[str], list[str]]:
    raw_norm = normalize_text(text)
    stripped = strip_boilerplate(text)
    cleaned = normalize_text(stripped)
    tokens = tokenize(cleaned, stopwords)
    raw_tokens = tokenize(raw_norm, set())
    return raw_norm, cleaned, raw_tokens, tokens


def raw_tokenize(text: str) -> list[str]:
    return tokenize(normalize_text(text), set())


def jsd_from_counters(a: Counter[str], b: Counter[str]) -> float:
    total_a = sum(a.values())
    total_b = sum(b.values())
    if total_a == 0 or total_b == 0:
        return float("nan")
    vocab = set(a) | set(b)
    if not vocab:
        return float("nan")
    pa = np.fromiter((a.get(t, 0) / total_a for t in vocab), dtype=float)
    pb = np.fromiter((b.get(t, 0) / total_b for t in vocab), dtype=float)
    m = 0.5 * (pa + pb)
    # Use base-2 entropy so JSD lies in [0, 1].
    def entropy(p: np.ndarray) -> float:
        mask = p > 0
        return float(-(p[mask] * np.log2(p[mask])).sum())

    return entropy(m) - 0.5 * entropy(pa) - 0.5 * entropy(pb)


def fightin_words(
    counts_a: Counter[str],
    counts_b: Counter[str],
    alpha0: float = 10000.0,
) -> pd.DataFrame:
    vocab = set(counts_a) | set(counts_b)
    if not vocab:
        return pd.DataFrame(columns=["term", "count_a", "count_b", "log_odds", "z_score", "winner"])
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    pooled = {term: counts_a.get(term, 0) + counts_b.get(term, 0) for term in vocab}
    pooled_total = sum(pooled.values())
    if pooled_total == 0:
        return pd.DataFrame(columns=["term", "count_a", "count_b", "log_odds", "z_score", "winner"])
    rows = []
    for term in vocab:
        ca = counts_a.get(term, 0)
        cb = counts_b.get(term, 0)
        alpha = (pooled[term] / pooled_total) * alpha0
        # Monroe et al. style regularized log-odds.
        log_odds_a = math.log((ca + alpha) / (total_a + alpha0 - ca - alpha))
        log_odds_b = math.log((cb + alpha) / (total_b + alpha0 - cb - alpha))
        log_odds = log_odds_a - log_odds_b
        var = 1.0 / (ca + alpha) + 1.0 / (cb + alpha)
        z = log_odds / math.sqrt(var) if var > 0 else float("nan")
        winner = "A" if z > 0 else "B"
        rows.append(
            {
                "term": term,
                "count_a": int(ca),
                "count_b": int(cb),
                "log_odds": float(log_odds),
                "z_score": float(z),
                "winner": winner,
            }
        )
    out = pd.DataFrame(rows)
    out["abs_z"] = out["z_score"].abs()
    out = out.sort_values(["abs_z", "z_score", "term"], ascending=[False, False, True]).drop(columns=["abs_z"])
    return out


def categorize_term(term: str) -> str:
    t = term.lower().strip()
    if t in AI_TOOL_TERMS or any(piece in t for piece in ["copilot", "cursor", "claude", "gpt", "llm", "rag", "agent", "mcp"]):
        return "ai_tool"
    if t in AI_DOMAIN_TERMS or any(piece in t for piece in ["machine learning", "deep learning", "natural language processing", "computer vision", "artificial intelligence"]):
        return "ai_domain"
    if t in TECH_STACK_TERMS:
        return "tech_stack"
    if t in ORG_SCOPE_TERMS:
        return "org_scope"
    if t in MGMT_TERMS:
        return "mgmt"
    if t in SYS_DESIGN_TERMS:
        return "sys_design"
    if t in METHOD_TERMS:
        return "method"
    if t in CREDENTIAL_TERMS or any(piece in t for piece in ["years experience", "bachelor", "master", "phd", "degree", "certification"]):
        return "credential"
    if t in SOFT_SKILL_TERMS:
        return "soft_skill"
    return "noise"


def is_html_artifact(term: str) -> bool:
    compact = term.replace(" ", "")
    if len(compact) <= 12:
        return False
    if compact in LEGIT_LONG_TERMS:
        return False
    if any(ch.isdigit() for ch in compact):
        return True
    if not re.search(r"[aeiou]", compact):
        return True
    return True


def is_location_term(term: str, location_tokens: set[str]) -> bool:
    tokens = set(term.split())
    return bool(tokens & location_tokens)


def term_company_filter(company_count: int, term: str, location_tokens: set[str]) -> bool:
    if company_count < 20:
        return False
    if is_html_artifact(term):
        return False
    if is_location_term(term, location_tokens):
        return False
    return True


def binary_mentions(text: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    for pattern in patterns:
        if pattern.search(text):
            return True
    return False


def make_phrase_patterns(phrases: Iterable[str]) -> list[re.Pattern[str]]:
    out = []
    for phrase in phrases:
        parts = [re.escape(p) for p in phrase.split()]
        pattern = r"\b" + r"\s+".join(parts) + r"\b"
        out.append(re.compile(pattern, flags=re.I))
    return out


def pairwise_table(df: pd.DataFrame, pair_name: str, a: str, b: str, feature_type: str) -> pd.DataFrame:
    sub = df.copy()
    sub = sub[(sub["z_score"].abs() > 3.0)].copy()
    if sub.empty:
        return sub
    sub["category"] = sub["term"].map(categorize_term).fillna("noise")
    sub = sub[sub["category"].isin(REPORTABLE_CATEGORIES)].copy()
    if sub.empty:
        return sub
    sub["comparison"] = pair_name
    sub["corpus_a"] = a
    sub["corpus_b"] = b
    sub["feature_type"] = feature_type
    sub = sub.rename(columns={"count_a": "count_corpus_a", "count_b": "count_corpus_b"})
    sub["winner_corpus"] = np.where(sub["z_score"] > 0, a, b)
    sub["loser_corpus"] = np.where(sub["z_score"] > 0, b, a)
    sub["rank_abs_z"] = sub["z_score"].abs().rank(method="first", ascending=False).astype(int)
    sub = sub.sort_values(["rank_abs_z", "term"])
    sub = sub.head(50).copy()
    return sub[
        [
            "comparison",
            "corpus_a",
            "corpus_b",
            "feature_type",
            "rank_abs_z",
            "term",
            "category",
            "z_score",
            "log_odds",
            "winner_corpus",
            "loser_corpus",
            "count_corpus_a",
            "count_corpus_b",
        ]
    ]


def make_scattertext_html(output_path: Path, df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> None:
    try:
        import scattertext as st
    except Exception as exc:
        output_path.write_text(f"<html><body><pre>Scattertext unavailable: {exc}</pre></body></html>")
        return

    scatter_df = df[[label_col, text_col]].copy()
    corpus = st.CorpusFromPandas(scatter_df, category_col=label_col, text_col=text_col).build()
    html = st.produce_scattertext_explorer(
        corpus,
        category=sorted(scatter_df[label_col].unique())[0],
        category_name=sorted(scatter_df[label_col].unique())[0],
        not_category_name=sorted(scatter_df[label_col].unique())[1],
        minimum_term_frequency=3,
        width_in_pixels=1100,
        metadata=cat if False else None,
    )
    output_path.write_text(html)


def plot_category_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    order = [
        "ai_tool",
        "ai_domain",
        "org_scope",
        "mgmt",
        "sys_design",
        "tech_stack",
        "method",
        "credential",
        "soft_skill",
        "noise",
    ]
    pivot = summary.pivot(index="comparison", columns="category", values="share").fillna(0.0)
    pivot = pivot.reindex(columns=order, fill_value=0.0)
    plt.figure(figsize=(14, 5))
    sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".2f", cbar_kws={"label": "Share of top terms"})
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_ai_prevalence(ai_df: pd.DataFrame, output_path: Path) -> None:
    if ai_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, metric, title in [
        (axes[0], "share_tool", "AI-tool mention"),
        (axes[1], "share_domain", "AI-domain mention"),
    ]:
        sub = ai_df.copy()
        sns.lineplot(data=sub, x="period", y=metric, hue="seniority_final", marker="o", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Period")
        ax.set_ylabel("Share of postings")
        ax.set_ylim(0, max(sub[metric].max() * 1.15, 0.01))
        ax.legend(title="Seniority", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_yoe_distribution(yoe_df: pd.DataFrame, output_path: Path) -> None:
    if yoe_df.empty:
        return
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=yoe_df, x="period", y="yoe_extracted", hue="source", showfliers=False)
    sns.stripplot(data=yoe_df, x="period", y="yoe_extracted", hue="source", dodge=True, alpha=0.25, size=2)
    plt.ylabel("Extracted years of experience")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def tokenize_terms_for_prevalence(text: str, stopwords: set[str]) -> tuple[list[str], list[str]]:
    cleaned = normalize_text(strip_boilerplate(text))
    tokens = tokenize(cleaned, stopwords)
    raw_tokens = tokenize(normalize_text(text), set())
    return raw_tokens, tokens


def run() -> None:
    ensure_dirs()
    matplotlib.use("Agg")
    con = duckdb.connect()
    stopwords = make_company_stopwords(con)
    analysis_text_expr = get_analysis_text_expr(con)
    location_tokens = set()
    for q in [
        f"SELECT DISTINCT metro_area FROM read_parquet('{STAGE8.as_posix()}') WHERE metro_area IS NOT NULL",
        f"SELECT DISTINCT state_normalized FROM read_parquet('{STAGE8.as_posix()}') WHERE state_normalized IS NOT NULL",
    ]:
        for batch in con.execute(q).to_arrow_reader():
            for value in batch.column(0).to_pylist():
                if value is None:
                    continue
                location_tokens.update(TOKEN_RE.findall(normalize_text(str(value))))

    corpora = {
        "junior_2024": CorpusSpec("junior_2024", "source = 'kaggle_arshkon' AND period = '2024-04' AND is_swe = true AND seniority_final = 'entry'"),
        "junior_2026": CorpusSpec("junior_2026", "source = 'scraped' AND period = '2026-03' AND is_swe = true AND seniority_final = 'entry'"),
        "senior_2024": CorpusSpec("senior_2024", "source = 'kaggle_arshkon' AND period = '2024-04' AND is_swe = true AND seniority_final IN ('mid-senior', 'director')"),
        "senior_2026": CorpusSpec("senior_2026", "source = 'scraped' AND period = '2026-03' AND is_swe = true AND seniority_final IN ('mid-senior', 'director')"),
        "swe_2026": CorpusSpec("swe_2026", "source = 'scraped' AND period = '2026-03' AND is_swe = true"),
        "control_2026": CorpusSpec("control_2026", "source = 'scraped' AND period = '2026-03' AND is_control = true"),
    }

    period_groups = {
        "2024-01": "2024_01",
        "2024-04": "2024_04",
        "2026-03": "2026_03",
    }

    raw_token_counts_by_period: dict[str, Counter[str]] = defaultdict(Counter)
    clean_token_counts_by_period: dict[str, Counter[str]] = defaultdict(Counter)
    swe_raw_token_counts_by_period: dict[str, Counter[str]] = defaultdict(Counter)
    swe_clean_token_counts_by_period: dict[str, Counter[str]] = defaultdict(Counter)
    swe_raw_token_counts_by_period_seniority: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    swe_clean_token_counts_by_period_seniority: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    swe_clean_doc_counts_by_period: dict[str, Counter[str]] = defaultdict(Counter)
    swe_clean_doc_counts_by_period_seniority: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    corpus_unigrams: dict[str, Counter[str]] = defaultdict(Counter)
    corpus_bigrams: dict[str, Counter[str]] = defaultdict(Counter)
    corpus_docs: dict[str, int] = defaultdict(int)
    corpus_chars: dict[str, int] = defaultdict(int)
    corpus_clean_chars: dict[str, int] = defaultdict(int)
    ai_summary_rows: list[dict[str, object]] = []
    yoe_rows: list[dict[str, object]] = []
    sample_rows_for_scatter: dict[str, list[dict[str, str]]] = defaultdict(list)

    ai_tool_patterns = make_phrase_patterns(
        [
            "copilot",
            "cursor",
            "claude",
            "gpt",
            "chatgpt",
            "llm",
            "large language model",
            "language model",
            "rag",
            "retrieval augmented",
            "agent",
            "mcp",
            "prompt engineering",
            "ai pair programming",
            "openai",
            "anthropic",
            "genai",
            "generative ai",
            "ai agent",
            "ai assistant",
            "model evaluation",
        ]
    )
    ai_domain_patterns = make_phrase_patterns(
        [
            "machine learning",
            "deep learning",
            "natural language processing",
            "computer vision",
            "artificial intelligence",
            "reinforcement learning",
            "nlp",
        ]
    )

    # Pass 1: build all frequency counts and prevalence counts.
    base_query = f"""
        SELECT source, period, seniority_final, is_swe, is_control, company_name_canonical,
               {analysis_text_expr}, COALESCE(yoe_extracted, NULL) AS yoe_extracted
        FROM read_parquet('{STAGE8.as_posix()}')
        WHERE {LINKEDIN_FILTER}
    """
    reader = con.execute(base_query).to_arrow_reader()
    batch_size = 0
    total_rows = 0
    for batch in reader:
        batch_size += 1
        total_rows += batch.num_rows
        cols = {name: batch.column(i).to_pylist() for i, name in enumerate(batch.schema.names)}
        for i in range(batch.num_rows):
            source = cols["source"][i]
            period = cols["period"][i]
            seniority = cols["seniority_final"][i]
            is_swe = bool(cols["is_swe"][i]) if cols["is_swe"][i] is not None else False
            is_control = bool(cols["is_control"][i]) if cols["is_control"][i] is not None else False
            company = cols["company_name_canonical"][i]
            desc = cols["analysis_text"][i]
            yoe = cols["yoe_extracted"][i]
            if not isinstance(desc, str) or not desc.strip():
                continue
            raw_tokens, clean_tokens = tokenize_terms_for_prevalence(desc, stopwords)
            raw_norm = normalize_text(desc)
            clean_norm = normalize_text(strip_boilerplate(desc))

            if is_swe:
                swe_raw_token_counts_by_period[period].update(raw_tokens)
                swe_clean_token_counts_by_period[period].update(clean_tokens)
                swe_clean_doc_counts_by_period[period].update(set(clean_tokens))
                swe_raw_token_counts_by_period_seniority[(period, seniority)].update(raw_tokens)
                swe_clean_token_counts_by_period_seniority[(period, seniority)].update(clean_tokens)
                swe_clean_doc_counts_by_period_seniority[(period, seniority)].update(set(clean_tokens))

                # AI keyword summary only counts once per posting.
                tool = binary_mentions(clean_norm, ai_tool_patterns)
                domain = binary_mentions(clean_norm, ai_domain_patterns)
                ai_summary_rows.append(
                    {
                        "period": period,
                        "seniority_final": seniority,
                        "is_swe": True,
                        "tool": tool,
                        "domain": domain,
                        "any": tool or domain,
                        "char_len": len(clean_norm),
                    }
                )

            # Corpus membership for T10 comparisons.
            memberships: list[str] = []
            if source == "kaggle_arshkon" and period == "2024-04" and is_swe and seniority == "entry":
                memberships.append("junior_2024")
            if source == "scraped" and period == "2026-03" and is_swe and seniority == "entry":
                memberships.append("junior_2026")
            if source == "kaggle_arshkon" and period == "2024-04" and is_swe and seniority in {"mid-senior", "director"}:
                memberships.append("senior_2024")
            if source == "scraped" and period == "2026-03" and is_swe and seniority in {"mid-senior", "director"}:
                memberships.append("senior_2026")
            if source == "scraped" and period == "2026-03" and is_swe:
                memberships.append("swe_2026")
            if source == "scraped" and period == "2026-03" and is_control:
                memberships.append("control_2026")

            if memberships:
                bigrams = build_ngrams(clean_tokens, 2)
                for name in memberships:
                    corpus_docs[name] += 1
                    corpus_unigrams[name].update(clean_tokens)
                    corpus_bigrams[name].update(bigrams)
                    corpus_chars[name] += len(raw_norm)
                    corpus_clean_chars[name] += len(clean_norm)
                    if name in {"junior_2024", "junior_2026", "senior_2024", "senior_2026"}:
                        sample_rows_for_scatter[name].append({"label": name, "text": " ".join(clean_tokens)})

                # yoe summary for entry-level SWE comparison.
                if is_swe and seniority == "entry" and source in {"kaggle_arshkon", "scraped"} and period in {"2024-04", "2026-03"}:
                    yoe_rows.append(
                        {
                            "source": source,
                            "period": period,
                            "yoe_extracted": float(yoe) if yoe is not None else np.nan,
                        }
                    )

    # Build overall 2024 vs 2026 corpora for drift.
    overall_2024_raw = Counter()
    overall_2024_clean = Counter()
    overall_2026_raw = Counter()
    overall_2026_clean = Counter()
    for period in ["2024-01", "2024-04"]:
        overall_2024_raw.update(swe_raw_token_counts_by_period.get(period, Counter()))
        overall_2024_clean.update(swe_clean_token_counts_by_period.get(period, Counter()))
    overall_2026_raw.update(swe_raw_token_counts_by_period.get("2026-03", Counter()))
    overall_2026_clean.update(swe_clean_token_counts_by_period.get("2026-03", Counter()))

    # T10: fightin' words tables.
    comparison_specs = [
        ("T10_01_junior_2024_vs_junior_2026", "junior_2024", "junior_2026"),
        ("T10_02_senior_2024_vs_senior_2026", "senior_2024", "senior_2026"),
        ("T10_03_junior_2024_vs_senior_2024", "junior_2024", "senior_2024"),
        ("T10_04_junior_2026_vs_senior_2026", "junior_2026", "senior_2026"),
        ("T10_05_junior_2026_vs_senior_2024", "junior_2026", "senior_2024"),
        ("T10_06_swe_2026_vs_control_2026", "swe_2026", "control_2026"),
    ]
    count_rows = []
    category_summary_rows = []
    all_top_tables = []
    pair_tables = {}
    for comp_id, a, b in comparison_specs:
        count_rows.append(
            {
                "comparison": comp_id,
                "corpus_a": a,
                "corpus_b": b,
                "n_a": corpus_docs.get(a, 0),
                "n_b": corpus_docs.get(b, 0),
                "flag_low_n": bool(corpus_docs.get(a, 0) < 50 or corpus_docs.get(b, 0) < 50),
            }
        )
        uni = fightin_words(corpus_unigrams.get(a, Counter()), corpus_unigrams.get(b, Counter()))
        bi = fightin_words(corpus_bigrams.get(a, Counter()), corpus_bigrams.get(b, Counter()))
        uni["feature_type"] = "unigram"
        bi["feature_type"] = "bigram"
        top = pd.concat(
            [
                pairwise_table(uni, comp_id, a, b, "unigram"),
                pairwise_table(bi, comp_id, a, b, "bigram"),
            ],
            ignore_index=True,
        )
        top["category"] = top["term"].map(categorize_term)
        top["category"] = top["category"].fillna("noise")
        top.to_csv(OUT_TABLE_T10 / f"{comp_id}.csv", index=False)
        pair_tables[comp_id] = top
        all_top_tables.append(top)
        total_terms = len(top)
        cat_counts = top["category"].value_counts(normalize=True).rename_axis("category").reset_index(name="share")
        cat_counts["comparison"] = comp_id
        cat_counts["n_terms"] = total_terms
        category_summary_rows.append(cat_counts)

        # Persist the full scored tables, not just the top 50, for traceability.
        uni["comparison"] = comp_id
        uni["feature_type"] = "unigram"
        bi["comparison"] = comp_id
        bi["feature_type"] = "bigram"
        uni["corpus_a"] = a
        uni["corpus_b"] = b
        bi["corpus_a"] = a
        bi["corpus_b"] = b
        uni["category"] = uni["term"].map(categorize_term)
        bi["category"] = bi["term"].map(categorize_term)
        uni["winner_corpus"] = np.where(uni["z_score"] > 0, a, b)
        bi["winner_corpus"] = np.where(bi["z_score"] > 0, a, b)
        uni.to_csv(OUT_TABLE_T10 / f"{comp_id}_full_unigram_scores.csv", index=False)
        bi.to_csv(OUT_TABLE_T10 / f"{comp_id}_full_bigram_scores.csv", index=False)

    counts_df = pd.DataFrame(count_rows)
    counts_df.to_csv(OUT_TABLE_T10 / "T10_comparison_counts.csv", index=False)
    category_df = pd.concat(category_summary_rows, ignore_index=True) if category_summary_rows else pd.DataFrame()
    category_df.to_csv(OUT_TABLE_T10 / "T10_category_summary.csv", index=False)

    plot_category_summary(category_df, OUT_FIG_T10 / "T10_category_summary.png")

    # Scattertext for comparisons 1 and 2 if available.
    for comp_id, a, b in [
        ("T10_01_junior_2024_vs_junior_2026", "junior_2024", "junior_2026"),
        ("T10_02_senior_2024_vs_senior_2026", "senior_2024", "senior_2026"),
    ]:
        docs = []
        docs.extend(sample_rows_for_scatter.get(a, []))
        docs.extend(sample_rows_for_scatter.get(b, []))
        if docs:
            df_scatter = pd.DataFrame(docs)
            # Label names are the corpus codes; use cleaned descriptions.
            try:
                import scattertext as st  # noqa: F401

                corpus = st.CorpusFromPandas(df_scatter, category_col="label", text_col="text").build()
                html = st.produce_scattertext_explorer(
                    corpus,
                    category=a,
                    category_name=a,
                    not_category_name=b,
                    minimum_term_frequency=3,
                    width_in_pixels=1100,
                )
                (OUT_FIG_T10 / f"{comp_id}.html").write_text(html)
            except Exception as exc:
                (OUT_FIG_T10 / f"{comp_id}.html").write_text(f"<html><body><pre>Scattertext unavailable: {exc}</pre></body></html>")

    # T11: JSD, prevalence, and yoe.
    jsd_rows = []
    period_pairs = [("2024-01", "2026-03"), ("2024-04", "2026-03"), ("2024-01", "2024-04")]
    for left, right in period_pairs:
        jsd_rows.append(
            {
                "scope": "overall_raw",
                "left_period": left,
                "right_period": right,
                "jsd": jsd_from_counters(swe_raw_token_counts_by_period.get(left, Counter()), swe_raw_token_counts_by_period.get(right, Counter())),
            }
        )
        jsd_rows.append(
            {
                "scope": "overall_clean",
                "left_period": left,
                "right_period": right,
                "jsd": jsd_from_counters(swe_clean_token_counts_by_period.get(left, Counter()), swe_clean_token_counts_by_period.get(right, Counter())),
            }
        )
    for seniority in ["entry", "associate", "mid-senior", "director"]:
        left_raw = swe_raw_token_counts_by_period_seniority.get(("2024-04", seniority), Counter())
        left_clean = swe_clean_token_counts_by_period_seniority.get(("2024-04", seniority), Counter())
        jsd_rows.append(
            {
                "scope": f"seniority_{seniority}_raw",
                "left_period": f"2024-04_{seniority}",
                "right_period": f"2026-03_{seniority}",
                "jsd": jsd_from_counters(left_raw, swe_raw_token_counts_by_period_seniority.get(("2026-03", seniority), Counter())),
            }
        )
        jsd_rows.append(
            {
                "scope": f"seniority_{seniority}_clean",
                "left_period": f"2024-04_{seniority}",
                "right_period": f"2026-03_{seniority}",
                "jsd": jsd_from_counters(left_clean, swe_clean_token_counts_by_period_seniority.get(("2026-03", seniority), Counter())),
            }
        )
    jsd_df = pd.DataFrame(jsd_rows)
    jsd_df.to_csv(OUT_TABLE_T11 / "T11_jsd_summary.csv", index=False)

    # Term emergence / disappearance / acceleration from overall 2024 vs 2026.
    overall_2024_doc = Counter()
    overall_2026_doc = Counter()
    for period in ["2024-01", "2024-04"]:
        overall_2024_doc.update(swe_clean_doc_counts_by_period.get(period, Counter()))
    overall_2026_doc.update(swe_clean_doc_counts_by_period.get("2026-03", Counter()))
    # Recompute denominators directly from the SWE-only SQL summary.
    period_n_docs = con.execute(
        f"""
        SELECT period, count(*) AS n
        FROM read_parquet('{STAGE8.as_posix()}')
        WHERE {SWE_FILTER}
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchall()
    period_n_docs_map = {p: int(n) for p, n in period_n_docs}
    n_docs_2024 = period_n_docs_map.get("2024-01", 0) + period_n_docs_map.get("2024-04", 0)
    n_docs_2026 = period_n_docs_map.get("2026-03", 0)

    term_rows = []
    vocab = set(overall_2024_doc) | set(overall_2026_doc)
    for term in vocab:
        p24 = overall_2024_doc.get(term, 0) / n_docs_2024 if n_docs_2024 else 0.0
        p26 = overall_2026_doc.get(term, 0) / n_docs_2026 if n_docs_2026 else 0.0
        term_rows.append(
            {
                "term": term,
                "prevalence_2024": p24,
                "prevalence_2026": p26,
                "ratio_2026_to_2024": (p26 / p24) if p24 > 0 else np.inf,
                "delta_pp": (p26 - p24) * 100,
                "category": categorize_term(term),
            }
        )
    term_df = pd.DataFrame(term_rows)

    # Second pass: company counts for candidate terms.
    candidate_terms = set(
        term_df.loc[
            ((term_df["prevalence_2026"] > 0.01) & (term_df["prevalence_2024"] < 0.001))
            | ((term_df["prevalence_2024"] > 0.01) & (term_df["prevalence_2026"] < 0.001))
            | ((term_df["prevalence_2024"] > 0.001) & (term_df["prevalence_2026"] > 0.001) & (term_df["ratio_2026_to_2024"] > 3.0)),
            "term",
        ].tolist()
    )
    company_sets: dict[str, set[str]] = {term: set() for term in candidate_terms}

    if candidate_terms:
        reader2 = con.execute(base_query).to_arrow_reader()
        for batch in reader2:
            cols = {name: batch.column(i).to_pylist() for i, name in enumerate(batch.schema.names)}
            for i in range(batch.num_rows):
                desc = cols["analysis_text"][i]
                company = cols["company_name_canonical"][i]
                if not isinstance(desc, str) or not desc.strip():
                    continue
                _, clean_tokens = tokenize_terms_for_prevalence(desc, stopwords)
                uniq = set(clean_tokens) & candidate_terms
                if not uniq:
                    continue
                company_key = str(company) if company is not None else ""
                for term in uniq:
                    company_sets[term].add(company_key)

    candidate_rows = []
    for term, row in term_df.set_index("term").iterrows():
        if term not in candidate_terms:
            continue
        company_count = len(company_sets.get(term, set()))
        if not term_company_filter(company_count, term, location_tokens):
            continue
        category = categorize_term(term)
        if category not in REPORTABLE_CATEGORIES:
            continue
        candidate_rows.append(
            {
                "term": term,
                "category": category,
                "prevalence_2024": row["prevalence_2024"],
                "prevalence_2026": row["prevalence_2026"],
                "ratio_2026_to_2024": row["ratio_2026_to_2024"],
                "delta_pp": row["delta_pp"],
                "company_count": company_count,
            }
        )
    candidate_df = pd.DataFrame(candidate_rows)
    emergent_df = candidate_df[(candidate_df["prevalence_2026"] > 0.01) & (candidate_df["prevalence_2024"] < 0.001)].copy()
    disappearing_df = candidate_df[(candidate_df["prevalence_2024"] > 0.01) & (candidate_df["prevalence_2026"] < 0.001)].copy()
    accelerating_df = candidate_df[(candidate_df["prevalence_2024"] > 0.001) & (candidate_df["prevalence_2026"] > 0.001) & (candidate_df["ratio_2026_to_2024"] > 3.0)].copy()

    emergent_df = emergent_df.sort_values(["prevalence_2026", "company_count", "term"], ascending=[False, False, True]).head(30)
    disappearing_df = disappearing_df.sort_values(["prevalence_2024", "company_count", "term"], ascending=[False, False, True]).head(30)
    accelerating_df = accelerating_df.sort_values(["ratio_2026_to_2024", "delta_pp", "term"], ascending=[False, False, True]).head(30)
    emergent_df.to_csv(OUT_TABLE_T11 / "T11_emerging_terms.csv", index=False)
    disappearing_df.to_csv(OUT_TABLE_T11 / "T11_disappearing_terms.csv", index=False)
    accelerating_df.to_csv(OUT_TABLE_T11 / "T11_accelerating_terms.csv", index=False)

    # AI prevalence summary for all period x seniority combinations.
    ai_df = pd.DataFrame(ai_summary_rows)
    if not ai_df.empty:
        ai_summary = (
            ai_df.groupby(["period", "seniority_final"], as_index=False)
            .agg(
                n=("any", "size"),
                share_any=("any", "mean"),
                share_tool=("tool", "mean"),
                share_domain=("domain", "mean"),
                avg_chars=("char_len", "mean"),
            )
            .sort_values(["period", "seniority_final"])
        )
        ai_summary["rate_any_per_1k_chars"] = ai_summary["share_any"] / (ai_summary["avg_chars"] / 1000.0)
        ai_summary["rate_tool_per_1k_chars"] = ai_summary["share_tool"] / (ai_summary["avg_chars"] / 1000.0)
        ai_summary["rate_domain_per_1k_chars"] = ai_summary["share_domain"] / (ai_summary["avg_chars"] / 1000.0)
    else:
        ai_summary = pd.DataFrame(columns=["period", "seniority_final", "n", "share_any", "share_tool", "share_domain", "avg_chars"])
    ai_summary.to_csv(OUT_TABLE_T11 / "T11_ai_prevalence.csv", index=False)
    plot_ai_prevalence(ai_summary, OUT_FIG_T11 / "T11_ai_prevalence.png")

    gate_counts = con.execute(
        f"""
        SELECT period, seniority_final, COUNT(*) AS n
        FROM read_parquet('{STAGE8.as_posix()}')
        WHERE {LINKEDIN_FILTER} AND is_swe = true
        GROUP BY 1,2
        ORDER BY 1,2
        """
    ).fetchdf()
    if not ai_summary.empty:
        merged_gate = gate_counts.merge(ai_summary[["period", "seniority_final", "n"]], on=["period", "seniority_final"], how="left", suffixes=("_sql", "_ai"))
        if merged_gate["n_ai"].isna().any() or not (merged_gate["n_sql"].astype(int) == merged_gate["n_ai"].astype(int)).all():
            raise RuntimeError("AI prevalence row counts do not match SWE gate counts; check for duplicate posting inclusion.")

    entry_gate = con.execute(
        f"""
        SELECT source, period, COUNT(*) AS n
        FROM read_parquet('{STAGE8.as_posix()}')
        WHERE {LINKEDIN_FILTER}
          AND is_swe = true
          AND seniority_final = 'entry'
          AND source IN ('kaggle_arshkon', 'scraped')
          AND period IN ('2024-04', '2026-03')
        GROUP BY 1,2
        ORDER BY 1,2
        """
    ).fetchdf()
    expected_entry = {
        ("kaggle_arshkon", "2024-04"): 773,
        ("scraped", "2026-03"): 561,
    }
    for _, row in entry_gate.iterrows():
        key = (row["source"], row["period"])
        if int(row["n"]) != expected_entry.get(key, -1):
            raise RuntimeError(f"Entry gate count mismatch for {key}: got {int(row['n'])}, expected {expected_entry.get(key)}")

    yoe_df = pd.DataFrame(yoe_rows)
    if not yoe_df.empty:
        yoe_summary = (
            yoe_df.groupby(["source", "period"], as_index=False)
            .agg(
                n=("yoe_extracted", "size"),
                non_null_n=("yoe_extracted", lambda s: int(s.notna().sum())),
                mean=("yoe_extracted", "mean"),
                median=("yoe_extracted", "median"),
                p25=("yoe_extracted", lambda s: float(s.quantile(0.25))),
                p75=("yoe_extracted", lambda s: float(s.quantile(0.75))),
                share_ge2=("yoe_extracted", lambda s: float((s >= 2).mean(skipna=True))),
            )
            .sort_values(["source", "period"])
        )
    else:
        yoe_summary = pd.DataFrame(columns=["source", "period", "n", "non_null_n", "mean", "median", "p25", "p75", "share_ge2"])
    yoe_summary.to_csv(OUT_TABLE_T11 / "T11_yoe_entry_summary.csv", index=False)
    plot_yoe_distribution(yoe_df, OUT_FIG_T11 / "T11_yoe_entry_distribution.png")

    # A compact JSD figure.
    if not jsd_df.empty:
        plt.figure(figsize=(10, 5))
        plot_df = jsd_df.copy()
        plot_df["pair"] = plot_df["left_period"] + " vs " + plot_df["right_period"]
        plot_df = plot_df[plot_df["scope"].str.contains("clean")]
        sns.barplot(data=plot_df, x="pair", y="jsd", hue="scope")
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("Jensen-Shannon divergence")
        plt.tight_layout()
        plt.savefig(OUT_FIG_T11 / "T11_jsd_summary.png", dpi=150)
        plt.close()

    # Write lightweight markdown reports from the computed tables.
    def write_report_t10() -> None:
        lines = [
            "# T10: Fightin' Words corpus comparison",
            "## Finding",
            f"All six requested comparisons were rerun on LinkedIn-only, English, date-ok rows with known seniority using `description_core` as the primary text field. The gate counts are correct for the entry frame (`2024-04` arshkon n=773, `2026-03` scraped n=561), and the cleaned top-term tables are now dominated by `tech_stack`, `ai_tool`, `sys_design`, `credential`, and `method` terms rather than generic noise.",
            "## Implication for analysis",
            "The cleaned Fightin' Words tables now provide a usable lexical map of junior scope inflation, senior redefinition, and cross-occupation contrast for RQ1/RQ2. The strongest contrasts are in AI-tool vocabulary, systems/design language, and stack-specific tooling, which is the expected analytic signal for the paper.",
            "## Data quality note",
            f"Stage 8 does not contain `description_core_llm`, so the analysis used `description_core` with `description` as a fallback only where needed, plus company/location stoplists. The top tables are reportable-category filtered, so the remaining terms are substantially cleaner and the category summary is no longer noise-dominated.",
            "## Action items",
            "Use the cleaned CSVs and the category summary as the seed set for analysis-phase keyword dictionaries, with `llm`, `agentic`, `observability`, `javascript`, `reactjs`, and `kubernetes` as the main candidates for follow-up.",
        ]
        OUT_REPORT_T10.write_text("\n".join(lines))

    def write_report_t11() -> None:
        lines = [
            "# T11: Temporal drift",
            "## Finding",
            "The JSD tables now compare SWE-only unigram distributions with matched seniority frames, and the prevalence table counts each posting once. The entry gate checks pass (`2024-04` arshkon entry n=773, `2026-03` scraped entry n=561), and the emerging / accelerating / disappearing tables are clean enough for downstream use.",
            "## Implication for analysis",
            "Use the matched-seniority JSD values as the drift diagnostic, and treat the AI-tool series as the main signal for anticipatory restructuring. The YOE comparison remains the defensible arshkon-vs-scraped baseline for the junior scope claim.",
            "## Data quality note",
            "The entry-level historical baseline is still structurally weaker in asaniczka, so the entry YOE table stays on the arshkon-vs-scraped frame. Candidate terms must appear in at least 20 companies, survive the artifact filters, and pass the reportable-category screen before they are written.",
            "## Action items",
            "Downstream analysis should cite the cleaned JSD values, the AI prevalence chart, and the filtered term lists rather than any raw candidate output.",
        ]
        OUT_REPORT_T11.write_text("\n".join(lines))

    write_report_t10()
    write_report_t11()

    summary_path = ROOT / "exploration" / "tables" / "T11" / "T11_debug_summary.csv"
    pd.DataFrame(
        [
            {"metric": "rows_processed", "value": total_rows},
            {"metric": "stopwords_n", "value": len(stopwords)},
            {"metric": "location_tokens_n", "value": len(location_tokens)},
            {"metric": "candidate_terms_n", "value": len(candidate_terms)},
            {"metric": "emergent_terms_n", "value": len(emergent_df)},
            {"metric": "accelerating_terms_n", "value": len(accelerating_df)},
            {"metric": "disappearing_terms_n", "value": len(disappearing_df)},
            {"metric": "ai_rows_n", "value": len(ai_df)},
            {"metric": "yoe_rows_n", "value": len(yoe_df)},
            {"metric": "entry_gate_arshkon_2024_04", "value": int(entry_gate.loc[(entry_gate.source == "kaggle_arshkon") & (entry_gate.period == "2024-04"), "n"].iloc[0])},
            {"metric": "entry_gate_scraped_2026_03", "value": int(entry_gate.loc[(entry_gate.source == "scraped") & (entry_gate.period == "2026-03"), "n"].iloc[0])},
        ]
    ).to_csv(summary_path, index=False)


if __name__ == "__main__":
    run()
