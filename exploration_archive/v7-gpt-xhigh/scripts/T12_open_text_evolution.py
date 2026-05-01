#!/usr/bin/env python3
"""T12 open-ended text evolution.

Primary corpus comparison:
- LinkedIn SWE, LLM-cleaned text only.
- kaggle_arshkon 2024 vs scraped 2026.
- Aggregators excluded and company-capped at 50 postings per corpus/company.
- Company-name and location tokens stripped before tokenization.

The script uses T13's section artifact for requirements/responsibilities/preferred
section-filtered comparisons. It avoids BERTopic reruns and cross-validates against
T09 NMF archetype labels instead.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SHARED_TEXT = ROOT / "exploration/artifacts/shared/swe_cleaned_text.parquet"
STOPLIST_PATH = ROOT / "exploration/artifacts/shared/company_stoplist.txt"
T13_SECTIONS = ROOT / "exploration/tables/T13/section_text_by_uid.parquet"
T09_ARCHETYPES = ROOT / "exploration/artifacts/shared/swe_archetype_labels.parquet"
UNIFIED = ROOT / "data/unified.parquet"
TABLE_DIR = ROOT / "exploration/tables/T12"
FIG_DIR = ROOT / "exploration/figures/T12"

TOKEN_RE = re.compile(r"c\+\+|c#|\.net|ci/cd|node\.js|[a-z][a-z0-9+#]*(?:[./-][a-z0-9+#]+)*", re.I)
SENIOR_TITLE_RE = re.compile(r"\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b", re.I)

BASE_STOPWORDS = {
    "a",
    "about",
    "above",
    "across",
    "after",
    "again",
    "against",
    "all",
    "almost",
    "alone",
    "along",
    "already",
    "also",
    "although",
    "always",
    "am",
    "among",
    "an",
    "and",
    "another",
    "any",
    "are",
    "around",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "during",
    "each",
    "either",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "ll",
    "may",
    "me",
    "more",
    "most",
    "must",
    "my",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "within",
    "without",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

JOB_GENERIC_STOPWORDS = {
    "ability",
    "able",
    "applicant",
    "apply",
    "based",
    "candidate",
    "candidates",
    "career",
    "company",
    "description",
    "employee",
    "employees",
    "employer",
    "employment",
    "etc",
    "function",
    "functions",
    "including",
    "job",
    "jobs",
    "like",
    "located",
    "looking",
    "need",
    "new",
    "opportunity",
    "position",
    "preferred",
    "provide",
    "related",
    "required",
    "requirements",
    "responsibilities",
    "role",
    "skills",
    "team",
    "teams",
    "using",
    "e.g",
    "eg",
    "etc",
    "various",
    "location",
    "preferably",
    "assigned",
    "existing",
    "months",
    "updated",
    "include",
    "duties",
    "different",
    "effectively",
    "advancements",
    "practical",
    "similar",
    "directly",
    "work",
    "working",
}

PROTECTED_TOKENS = {"c", "r", "go", "c++", "c#", ".net", "ci/cd"}

STATE_AND_COUNTRY_TOKENS = {
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "hampshire",
    "jersey",
    "mexico",
    "york",
    "carolina",
    "dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode",
    "island",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "wisconsin",
    "wyoming",
    "united",
    "states",
    "usa",
    "us",
    "remote",
    "hybrid",
    "onsite",
}


@dataclass
class CorpusStats:
    label: str
    n_docs: int
    total_tokens: int
    companies: int
    counts: Counter
    doc_counts: Counter
    company_sets: Mapping[str, set]


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=1")
    return con


def stable_hash(value: str) -> int:
    return int(hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest(), 16)


def assert_regex_edges(stopwords: set[str]) -> None:
    text = "C++ C# .NET CI/CD Java JavaScript node.js Go R"
    toks = tokenize(text, stopwords)
    assert "c++" in toks
    assert "c#" in toks
    assert ".net" in toks
    assert "ci/cd" in toks
    assert "java" in toks and "javascript" in toks
    assert senior_s3({"title": "Senior Software Engineer"})
    assert senior_s3({"title": "Sr. Staff Engineer"})
    assert not senior_s3({"title": "Seniority Program Manager"})


def load_company_stoplist() -> set[str]:
    tokens = set()
    if STOPLIST_PATH.exists():
        for line in STOPLIST_PATH.read_text(encoding="utf-8").splitlines():
            tok = line.strip().lower()
            if tok:
                tokens.add(tok)
    # Preserve short language/tool tokens that are easy to erase accidentally.
    return tokens - PROTECTED_TOKENS


def load_location_stoplist() -> set[str]:
    con = connect()
    rows = con.execute(
        f"""
        SELECT DISTINCT metro_area
        FROM read_parquet('{SHARED_TEXT.as_posix()}')
        WHERE metro_area IS NOT NULL
        """
    ).fetchall()
    con.close()
    loc_tokens = set(STATE_AND_COUNTRY_TOKENS)
    for (metro,) in rows:
        for tok in TOKEN_RE.findall(str(metro).lower()):
            if len(tok) > 1:
                loc_tokens.add(tok)
    return loc_tokens


def normalize_token(token: str) -> str:
    token = token.lower()
    if token in {".net", "c++", "c#", "ci/cd"}:
        return token
    token = token.strip("'\".,;:()[]{}")
    token = token.replace("c\\+\\+", "c++").replace("c\\#", "c#")
    token = token.replace("nodejs", "node.js")
    token = token.replace("ci-cd", "ci/cd").replace("cicd", "ci/cd")
    return token


def looks_like_artifact(token: str) -> bool:
    if not token:
        return True
    if token in PROTECTED_TOKENS:
        return False
    if token.isdigit():
        return True
    if len(token) == 1:
        return True
    if len(token) > 28:
        return True
    if len(token) > 16 and not re.search(r"[aeiou]", token):
        return True
    if re.search(r"(.)\1{4,}", token):
        return True
    if token.startswith("http") or token.startswith("www"):
        return True
    return False


def tokenize(text: str, stopwords: set[str]) -> List[str]:
    text = (text or "").lower().replace("’", "'")
    toks: List[str] = []
    for raw in TOKEN_RE.findall(text):
        tok = normalize_token(raw)
        if tok in stopwords and tok not in PROTECTED_TOKENS:
            continue
        if looks_like_artifact(tok):
            continue
        toks.append(tok)
    return toks


def ngrams(tokens: Sequence[str], n: int) -> List[str]:
    if n == 1:
        return list(tokens)
    out = []
    for i in range(len(tokens) - n + 1):
        gram = tokens[i : i + n]
        if len(set(gram)) == 1:
            continue
        out.append(" ".join(gram))
    return out


def canonical_company(row: Mapping[str, object]) -> str:
    company = row.get("company_name_canonical")
    if company is None or str(company).strip() == "":
        return f"missing_company::{row.get('uid')}"
    return str(company)


def source_group(row: Mapping[str, object]) -> str:
    source = row.get("source")
    if source == "scraped":
        return "scraped_2026"
    if source == "kaggle_arshkon":
        return "arshkon_2024"
    if source == "kaggle_asaniczka":
        return "asaniczka_2024"
    return str(source)


def pooled_or_scraped(row: Mapping[str, object]) -> str | None:
    if row.get("source") in {"kaggle_arshkon", "kaggle_asaniczka"}:
        return "pooled_2024"
    if row.get("source") == "scraped":
        return "scraped_2026"
    return None


def arshkon_or_scraped(row: Mapping[str, object]) -> str | None:
    if row.get("source") == "kaggle_arshkon":
        return "arshkon_2024"
    if row.get("source") == "scraped":
        return "scraped_2026"
    return None


def arshkon_or_asaniczka(row: Mapping[str, object]) -> str | None:
    if row.get("source") == "kaggle_arshkon":
        return "arshkon_2024"
    if row.get("source") == "kaggle_asaniczka":
        return "asaniczka_2024"
    return None


def junior_j1(row: Mapping[str, object]) -> bool:
    return row.get("seniority_final") == "entry"


def junior_j2(row: Mapping[str, object]) -> bool:
    return row.get("seniority_final") in {"entry", "associate"}


def junior_j3(row: Mapping[str, object]) -> bool:
    value = row.get("yoe_extracted")
    return value is not None and not pd.isna(value) and float(value) <= 2


def junior_j4(row: Mapping[str, object]) -> bool:
    value = row.get("yoe_extracted")
    return value is not None and not pd.isna(value) and float(value) <= 3


def senior_s1(row: Mapping[str, object]) -> bool:
    return row.get("seniority_final") in {"mid-senior", "director"}


def senior_s2(row: Mapping[str, object]) -> bool:
    return row.get("seniority_final") == "director"


def senior_s3(row: Mapping[str, object]) -> bool:
    return bool(SENIOR_TITLE_RE.search(str(row.get("title") or "")))


def senior_s4(row: Mapping[str, object]) -> bool:
    value = row.get("yoe_extracted")
    return value is not None and not pd.isna(value) and float(value) >= 5


JUNIOR_FLAGS = {"J1": junior_j1, "J2": junior_j2, "J3": junior_j3, "J4": junior_j4}
SENIOR_FLAGS = {"S1": senior_s1, "S2": senior_s2, "S3": senior_s3, "S4": senior_s4}


def load_llm_full_rows() -> List[Dict[str, object]]:
    con = connect()
    query = f"""
    SELECT s.uid, s.description_cleaned AS text, s.text_source, s.source, s.period,
           s.seniority_final, s.seniority_3level, s.is_aggregator,
           s.company_name_canonical, s.metro_area, s.yoe_extracted,
           s.swe_classification_tier, s.seniority_final_source,
           u.title
    FROM read_parquet('{SHARED_TEXT.as_posix()}') s
    LEFT JOIN read_parquet('{UNIFIED.as_posix()}') u USING (uid)
    WHERE s.text_source='llm'
      AND s.description_cleaned IS NOT NULL
      AND length(s.description_cleaned) > 0
    """
    rows: List[Dict[str, object]] = []
    reader = con.execute(query).fetch_record_batch(rows_per_batch=4096)
    for batch in reader:
        rows.extend(batch.to_pylist())
    con.close()
    return rows


def load_section_rows() -> List[Dict[str, object]]:
    con = connect()
    query = f"""
    SELECT uid, requirements_responsibilities_text AS text, text_source, source, period,
           seniority_final, seniority_3level, is_aggregator, company_name_canonical,
           yoe_extracted, swe_classification_tier, seniority_final_source,
           CAST(NULL AS VARCHAR) AS title
    FROM read_parquet('{T13_SECTIONS.as_posix()}')
    WHERE text_source='llm'
      AND requirements_responsibilities_text IS NOT NULL
      AND length(requirements_responsibilities_text) >= 20
    """
    rows: List[Dict[str, object]] = []
    reader = con.execute(query).fetch_record_batch(rows_per_batch=4096)
    for batch in reader:
        rows.extend(batch.to_pylist())
    con.close()
    return rows


def load_raw_rows_for_arshkon_scraped() -> List[Dict[str, object]]:
    con = connect()
    query = f"""
    SELECT uid, description AS text, 'raw_description' AS text_source, source, period,
           seniority_final, seniority_3level, is_aggregator, company_name_canonical,
           metro_area, yoe_extracted, swe_classification_tier, seniority_final_source,
           title
    FROM read_parquet('{UNIFIED.as_posix()}')
    WHERE source_platform='linkedin'
      AND is_english=true
      AND date_flag='ok'
      AND is_swe=true
      AND source IN ('kaggle_arshkon', 'scraped')
      AND description IS NOT NULL
      AND length(description) > 0
    """
    rows: List[Dict[str, object]] = []
    reader = con.execute(query).fetch_record_batch(rows_per_batch=2048)
    for batch in reader:
        rows.extend(batch.to_pylist())
    con.close()
    return rows


def select_rows(
    rows: Sequence[Dict[str, object]],
    side_func: Callable[[Mapping[str, object]], str | None],
    *,
    exclude_aggregators: bool = True,
    company_cap: int | None = 50,
    exclude_title_lookup_llm: bool = False,
    extra_predicate: Callable[[Mapping[str, object]], bool] | None = None,
) -> List[Dict[str, object]]:
    ordered = sorted(rows, key=lambda row: stable_hash(str(row.get("uid"))))
    selected: List[Dict[str, object]] = []
    company_counts: MutableMapping[Tuple[str, str], int] = defaultdict(int)
    for row in ordered:
        side = side_func(row)
        if side is None:
            continue
        text = str(row.get("text") or "")
        if len(text.strip()) < 20:
            continue
        if exclude_aggregators and bool(row.get("is_aggregator")):
            continue
        if exclude_title_lookup_llm and row.get("swe_classification_tier") == "title_lookup_llm":
            continue
        if extra_predicate is not None and not extra_predicate(row):
            continue
        company = canonical_company(row)
        if company_cap is not None:
            key = (side, company)
            if company_counts[key] >= company_cap:
                continue
            company_counts[key] += 1
        out = dict(row)
        out["side"] = side
        out["company_key"] = company
        selected.append(out)
    return selected


def build_stats(
    rows: Sequence[Dict[str, object]],
    side_label: str,
    stopwords: set[str],
    *,
    ngram_n: int = 1,
) -> CorpusStats:
    counts: Counter = Counter()
    doc_counts: Counter = Counter()
    company_sets: MutableMapping[str, set] = defaultdict(set)
    companies = set()
    total_tokens = 0
    n_docs = 0
    for row in rows:
        if row["side"] != side_label:
            continue
        companies.add(row["company_key"])
        toks = tokenize(str(row.get("text") or ""), stopwords)
        terms = ngrams(toks, ngram_n)
        if not terms:
            continue
        n_docs += 1
        total_tokens += len(terms)
        counts.update(terms)
        seen = set(terms)
        for term in seen:
            doc_counts[term] += 1
            company_sets[term].add(row["company_key"])
    return CorpusStats(side_label, n_docs, total_tokens, len(companies), counts, doc_counts, company_sets)


def categorize_term(term: str) -> Tuple[str, str]:
    t = term.lower()
    padded = f" {t} "
    ai_tool = {
        "copilot",
        "cursor",
        "claude",
        "gpt",
        "chatgpt",
        "openai",
        "anthropic",
        "gemini",
        "llm",
        "llms",
        "rag",
        "agent",
        "agents",
        "prompt",
        "prompts",
        "mcp",
        "evals",
        "langgraph",
    }
    if any(re.search(rf"(^| ){re.escape(x)}($| )", t) for x in ai_tool):
        note = "mcp_ambiguous" if "mcp" in t else ""
        return "ai_tool", note
    if any(x in t for x in ["ai-assisted", "ai-enabled", "llm-based", "llm-powered", "multi-agent", "assistant"]):
        return "ai_tool", ""
    if any(x in padded for x in [" ai ", " ai/ml ", " ml ", " machine learning ", " deep learning ", " nlp "]):
        return "ai_domain", ""
    if any(x in t for x in ["computer vision", "generative", "neural", "model", "models", "data science", "embedding", "embeddings", "multimodal", "drift"]):
        return "ai_domain", ""
    if any(
        x in t
        for x in [
            "ownership",
            "end-to-end",
            "cross-functional",
            "stakeholder",
            "stakeholders",
            "roadmap",
            "autonomy",
            "deliver",
            "drive",
            "contribute",
            "product",
            "business",
            "customers",
            "governance",
            "operational",
            "operate",
            "use cases",
            "track record",
        ]
    ):
        return "org_scope", ""
    if any(x in t for x in ["mentor", "manage", "manager", "leadership", "hiring", "coach", "lead "]):
        return "mgmt", ""
    if any(
        x in t
        for x in [
            "architecture",
            "architect",
            "distributed",
            "scalable",
            "scalability",
            "system",
            "systems",
            "reliability",
            "performance",
            "optimization",
            "latency",
            "large-scale",
            "real-time",
            "high-performance",
            "maintainability",
            "evaluation",
        ]
    ):
        return "sys_design", ""
    if any(
        x in t
        for x in [
            "agile",
            "scrum",
            "ci/cd",
            "cicd",
            "devops",
            "tdd",
            "testing",
            "deployment",
            "pipelines",
            "pipeline",
            "workflow",
            "workflows",
            "observability",
            "continuous improvement",
            "continuous deployment",
        ]
    ):
        return "method", ""
    if any(
        x in t
        for x in [
            "python",
            "java",
            "javascript",
            "typescript",
            "react",
            "angular",
            "node",
            "aws",
            "azure",
            "gcp",
            "kubernetes",
            "docker",
            "terraform",
            "sql",
            "spark",
            "scala",
            "api",
            "apis",
            "microservices",
            "cloud",
            "cloud-native",
            "linux",
            "c++",
            "c#",
            ".net",
            "go",
            "rust",
            "ios",
            "android",
            "backend",
            "frontend",
            "full-stack",
            "framework",
            "frameworks",
            "tooling",
            "environment",
            "environments",
            "workloads",
            "database",
            "directory",
            "active directory",
            "html",
            "php",
            "jquery",
            "mvc",
            "unix",
            "j2ee",
            "rdbms",
            "nosql",
            "application",
            "applications",
            "drupal",
            "laravel",
            "visualforce",
        ]
    ):
        return "tech_stack", ""
    if any(
        x in t
        for x in [
            "year",
            "degree",
            "bachelor",
            "master",
            "phd",
            "certification",
            "clearance",
            "qualification",
            "experience",
            "familiarity",
            "hands-on",
            "proficiency",
            "demonstrated",
            "exposure",
            "minimum qualifications",
            "active clearance",
            "ts/sci",
            "polygraph",
        ]
    ):
        return "credential", ""
    if any(
        x in t
        for x in [
            "communication",
            "communicate",
            "collaboration",
            "problem-solving",
            "analytical",
            "teamwork",
            "interpersonal",
            "written",
            "oral",
            "verbal",
            "attention",
        ]
    ):
        return "soft_skill", ""
    if any(
        x in t
        for x in [
            "salary",
            "benefits",
            "compensation",
            "pay",
            "equity",
            "bonus",
            "dental",
            "401k",
            "pto",
            "culture",
            "mission",
            "values",
            "diversity",
            "inclusion",
            "sponsorship",
            "visa",
            "accommodation",
            "equal opportunity",
            "privacy",
            "people",
            "employees",
        ]
    ):
        return "boilerplate", ""
    return "noise", "uncategorized_or_context_needed"


def log_odds_metrics(
    a: CorpusStats,
    b: CorpusStats,
    *,
    comparison: str,
    text_scope: str,
    ngram_n: int,
    min_distinct_companies: int = 20,
    prior_strength: float = 5000.0,
) -> pd.DataFrame:
    vocab = set(a.counts) | set(b.counts)
    combined_total = sum(a.counts.values()) + sum(b.counts.values())
    if combined_total == 0:
        return pd.DataFrame()
    rows = []
    # Build prior over all retained vocabulary, then use the retained alpha0.
    retained = []
    for term in vocab:
        distinct_companies = len(a.company_sets.get(term, set()) | b.company_sets.get(term, set()))
        if distinct_companies < min_distinct_companies:
            continue
        if a.counts[term] + b.counts[term] < 5:
            continue
        retained.append(term)
    alpha = {
        term: 0.01 + prior_strength * ((a.counts[term] + b.counts[term]) / combined_total)
        for term in retained
    }
    alpha0 = sum(alpha.values())
    for term in retained:
        y_a = a.counts[term]
        y_b = b.counts[term]
        alpha_i = alpha[term]
        denom_a = max(a.total_tokens + alpha0 - y_a - alpha_i, 1e-9)
        denom_b = max(b.total_tokens + alpha0 - y_b - alpha_i, 1e-9)
        logodds_a = math.log((y_a + alpha_i) / denom_a)
        logodds_b = math.log((y_b + alpha_i) / denom_b)
        delta = logodds_b - logodds_a
        variance = 1.0 / (y_a + alpha_i) + 1.0 / (y_b + alpha_i)
        z = delta / math.sqrt(variance)
        category, note = categorize_term(term)
        doc_prev_a = a.doc_counts[term] / a.n_docs if a.n_docs else math.nan
        doc_prev_b = b.doc_counts[term] / b.n_docs if b.n_docs else math.nan
        rows.append(
            {
                "comparison": comparison,
                "text_scope": text_scope,
                "ngram_n": ngram_n,
                "term": term,
                "category": category,
                "category_note": note,
                "z_b_minus_a": z,
                "log_odds_delta_b_minus_a": delta,
                "direction": f"{b.label}_heavy" if z > 0 else f"{a.label}_heavy",
                "count_a": y_a,
                "count_b": y_b,
                "rate_a_per_10k_terms": y_a / a.total_tokens * 10000 if a.total_tokens else math.nan,
                "rate_b_per_10k_terms": y_b / b.total_tokens * 10000 if b.total_tokens else math.nan,
                "doc_count_a": a.doc_counts[term],
                "doc_count_b": b.doc_counts[term],
                "doc_prev_a": doc_prev_a,
                "doc_prev_b": doc_prev_b,
                "distinct_companies_total": len(a.company_sets.get(term, set()) | b.company_sets.get(term, set())),
                "distinct_companies_a": len(a.company_sets.get(term, set())),
                "distinct_companies_b": len(b.company_sets.get(term, set())),
                "label_a": a.label,
                "label_b": b.label,
                "n_docs_a": a.n_docs,
                "n_docs_b": b.n_docs,
                "tokens_a": a.total_tokens,
                "tokens_b": b.total_tokens,
                "companies_a": a.companies,
                "companies_b": b.companies,
            }
        )
    return pd.DataFrame(rows)


def top_terms(metrics: pd.DataFrame, top_k: int = 100) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    frames = []
    for direction, group in metrics.groupby("direction"):
        ascending = group["z_b_minus_a"].mean() < 0
        top = group.sort_values("z_b_minus_a", ascending=ascending).head(top_k).copy()
        top["rank_in_direction"] = range(1, len(top) + 1)
        frames.append(top)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_comparison(
    rows: Sequence[Dict[str, object]],
    *,
    comparison: str,
    text_scope: str,
    side_func: Callable[[Mapping[str, object]], str | None],
    label_a: str,
    label_b: str,
    stopwords: set[str],
    ngram_n: int = 1,
    exclude_aggregators: bool = True,
    company_cap: int | None = 50,
    exclude_title_lookup_llm: bool = False,
    extra_predicate: Callable[[Mapping[str, object]], bool] | None = None,
    min_distinct_companies: int = 20,
    top_k: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], List[Dict[str, object]]]:
    selected = select_rows(
        rows,
        side_func,
        exclude_aggregators=exclude_aggregators,
        company_cap=company_cap,
        exclude_title_lookup_llm=exclude_title_lookup_llm,
        extra_predicate=extra_predicate,
    )
    stats_a = build_stats(selected, label_a, stopwords, ngram_n=ngram_n)
    stats_b = build_stats(selected, label_b, stopwords, ngram_n=ngram_n)
    metrics = log_odds_metrics(
        stats_a,
        stats_b,
        comparison=comparison,
        text_scope=text_scope,
        ngram_n=ngram_n,
        min_distinct_companies=min_distinct_companies,
    )
    top = top_terms(metrics, top_k=top_k)
    counts = {
        "comparison": comparison,
        "text_scope": text_scope,
        "ngram_n": ngram_n,
        "label_a": label_a,
        "label_b": label_b,
        "n_docs_a": stats_a.n_docs,
        "n_docs_b": stats_b.n_docs,
        "tokens_a": stats_a.total_tokens,
        "tokens_b": stats_b.total_tokens,
        "companies_a": stats_a.companies,
        "companies_b": stats_b.companies,
        "exclude_aggregators": exclude_aggregators,
        "company_cap": company_cap if company_cap is not None else "none",
        "exclude_title_lookup_llm": exclude_title_lookup_llm,
        "min_distinct_companies": min_distinct_companies,
        "flag_n_lt_100": stats_a.n_docs < 100 or stats_b.n_docs < 100,
        "retained_terms": len(metrics),
    }
    return metrics, top, counts, selected


def write_category_summary(top_full: pd.DataFrame, top_section: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for label, df in [("full_text", top_full), ("section_filtered", top_section)]:
        if df.empty:
            continue
        tmp = df[df["rank_in_direction"] <= 100].copy()
        summary = (
            tmp.groupby(["text_scope", "direction", "category"])
            .size()
            .reset_index(name="top100_terms")
        )
        summary["share_top100"] = summary["top100_terms"] / summary.groupby(["text_scope", "direction"])[
            "top100_terms"
        ].transform("sum")
        frames.append(summary)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(TABLE_DIR / "category_summary_full_vs_section.csv", index=False)
    return out


def write_overlap(top_full: pd.DataFrame, top_section: pd.DataFrame) -> pd.DataFrame:
    full_terms = set(top_full["term"])
    section_terms = set(top_section["term"])
    rows = []
    for df, scope, other_set in [(top_full, "full_text", section_terms), (top_section, "section_filtered", full_terms)]:
        for _, row in df.iterrows():
            rows.append(
                {
                    "term": row["term"],
                    "scope": scope,
                    "direction": row["direction"],
                    "category": row["category"],
                    "rank_in_direction": row["rank_in_direction"],
                    "also_in_other_scope": row["term"] in other_set,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "full_vs_section_term_overlap.csv", index=False)
    return out


def emerging_accelerating_disappearing(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in metrics.iterrows():
        prev_a = row["doc_prev_a"]
        prev_b = row["doc_prev_b"]
        status = None
        ratio = math.inf if prev_a == 0 and prev_b > 0 else (prev_b / prev_a if prev_a else math.nan)
        if prev_b >= 0.01 and prev_a < 0.001:
            status = "emerging_2026"
        elif prev_a >= 0.001 and prev_b >= 0.01 and ratio > 3:
            status = "accelerating_2026"
        elif prev_a >= 0.01 and prev_b < 0.001:
            status = "disappearing_2026"
        if status:
            rows.append(
                {
                    "status": status,
                    "term": row["term"],
                    "category": row["category"],
                    "category_note": row["category_note"],
                    "doc_prev_2024": prev_a,
                    "doc_prev_2026": prev_b,
                    "prev_ratio_2026_to_2024": ratio,
                    "z_b_minus_a": row["z_b_minus_a"],
                    "count_2024": row["count_a"],
                    "count_2026": row["count_b"],
                    "distinct_companies_total": row["distinct_companies_total"],
                    "likely_boilerplate_or_noise": row["category"] in {"boilerplate", "noise"},
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["status", "doc_prev_2026"], ascending=[True, False])
    out.to_csv(TABLE_DIR / "emerging_accelerating_disappearing_terms.csv", index=False)
    return out


def sensitivity_rows(
    full_rows: Sequence[Dict[str, object]],
    raw_rows: Sequence[Dict[str, object]],
    stopwords: set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    specs = [
        {
            "comparison": "primary_arshkon_vs_scraped_llm_excl_agg_cap50",
            "rows": full_rows,
            "text_scope": "full_text",
            "side_func": arshkon_or_scraped,
            "label_a": "arshkon_2024",
            "label_b": "scraped_2026",
            "exclude_aggregators": True,
            "company_cap": 50,
            "exclude_title_lookup_llm": False,
        },
        {
            "comparison": "aggregators_included_cap50",
            "rows": full_rows,
            "text_scope": "full_text",
            "side_func": arshkon_or_scraped,
            "label_a": "arshkon_2024",
            "label_b": "scraped_2026",
            "exclude_aggregators": False,
            "company_cap": 50,
            "exclude_title_lookup_llm": False,
        },
        {
            "comparison": "no_company_cap_excl_agg",
            "rows": full_rows,
            "text_scope": "full_text",
            "side_func": arshkon_or_scraped,
            "label_a": "arshkon_2024",
            "label_b": "scraped_2026",
            "exclude_aggregators": True,
            "company_cap": None,
            "exclude_title_lookup_llm": False,
        },
        {
            "comparison": "raw_description_excl_agg_cap50",
            "rows": raw_rows,
            "text_scope": "raw_description",
            "side_func": arshkon_or_scraped,
            "label_a": "arshkon_2024",
            "label_b": "scraped_2026",
            "exclude_aggregators": True,
            "company_cap": 50,
            "exclude_title_lookup_llm": False,
        },
        {
            "comparison": "pooled_2024_vs_scraped_llm_excl_agg_cap50",
            "rows": full_rows,
            "text_scope": "full_text",
            "side_func": pooled_or_scraped,
            "label_a": "pooled_2024",
            "label_b": "scraped_2026",
            "exclude_aggregators": True,
            "company_cap": 50,
            "exclude_title_lookup_llm": False,
        },
        {
            "comparison": "within_2024_arshkon_vs_asaniczka_excl_agg_cap50",
            "rows": full_rows,
            "text_scope": "full_text",
            "side_func": arshkon_or_asaniczka,
            "label_a": "arshkon_2024",
            "label_b": "asaniczka_2024",
            "exclude_aggregators": True,
            "company_cap": 50,
            "exclude_title_lookup_llm": False,
        },
        {
            "comparison": "exclude_title_lookup_llm_excl_agg_cap50",
            "rows": full_rows,
            "text_scope": "full_text",
            "side_func": arshkon_or_scraped,
            "label_a": "arshkon_2024",
            "label_b": "scraped_2026",
            "exclude_aggregators": True,
            "company_cap": 50,
            "exclude_title_lookup_llm": True,
        },
    ]
    summary_rows = []
    term_frames = []
    primary_top_2026: set[str] | None = None
    for spec in specs:
        metrics, top, counts, _ = run_comparison(
            spec["rows"],
            comparison=spec["comparison"],
            text_scope=spec["text_scope"],
            side_func=spec["side_func"],
            label_a=spec["label_a"],
            label_b=spec["label_b"],
            stopwords=stopwords,
            ngram_n=1,
            exclude_aggregators=spec["exclude_aggregators"],
            company_cap=spec["company_cap"],
            exclude_title_lookup_llm=spec["exclude_title_lookup_llm"],
            min_distinct_companies=20,
            top_k=50,
        )
        top_2026 = set(top[top["direction"].str.contains(spec["label_b"])]["term"])
        if spec["comparison"].startswith("primary_"):
            primary_top_2026 = top_2026
        overlap = len(primary_top_2026 & top_2026) / len(primary_top_2026) if primary_top_2026 else math.nan
        category_counts = (
            top[top["direction"].str.contains(spec["label_b"])]["category"].value_counts().to_dict()
            if not top.empty
            else {}
        )
        counts.update(
            {
                "top_b_overlap_with_primary": overlap,
                "top_b_ai_tool_terms": category_counts.get("ai_tool", 0),
                "top_b_ai_domain_terms": category_counts.get("ai_domain", 0),
                "top_b_tech_stack_terms": category_counts.get("tech_stack", 0),
                "top_b_boilerplate_terms": category_counts.get("boilerplate", 0),
                "top_b_noise_terms": category_counts.get("noise", 0),
            }
        )
        summary_rows.append(counts)
        term_frames.append(top)
    summary = pd.DataFrame(summary_rows)
    terms = pd.concat(term_frames, ignore_index=True) if term_frames else pd.DataFrame()
    summary.to_csv(TABLE_DIR / "sensitivity_summary.csv", index=False)
    terms.to_csv(TABLE_DIR / "sensitivity_top_terms.csv", index=False)
    return summary, terms


def secondary_comparisons(full_rows: Sequence[Dict[str, object]], stopwords: set[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    specs = []
    for name, pred in JUNIOR_FLAGS.items():
        specs.append(
            {
                "comparison": f"entry_{name}_arshkon_vs_scraped",
                "side_func": arshkon_or_scraped,
                "label_a": "arshkon_2024",
                "label_b": "scraped_2026",
                "predicate": pred,
                "panel_definition": name,
                "panel_side": "junior",
            }
        )
    for name, pred in SENIOR_FLAGS.items():
        specs.append(
            {
                "comparison": f"senior_{name}_arshkon_vs_scraped",
                "side_func": arshkon_or_scraped,
                "label_a": "arshkon_2024",
                "label_b": "scraped_2026",
                "predicate": pred,
                "panel_definition": name,
                "panel_side": "senior",
            }
        )
    for j_name, j_pred in JUNIOR_FLAGS.items():
        for s_name in ["S1", "S4"]:
            s_pred = SENIOR_FLAGS[s_name]

            def side_func(row: Mapping[str, object], jp=j_pred, sp=s_pred) -> str | None:
                if row.get("source") == "scraped" and jp(row):
                    return "entry_2026"
                if row.get("source") == "kaggle_arshkon" and sp(row):
                    return "senior_2024"
                return None

            specs.append(
                {
                    "comparison": f"relabeling_{j_name}_2026_vs_{s_name}_2024",
                    "side_func": side_func,
                    "label_a": "senior_2024",
                    "label_b": "entry_2026",
                    "predicate": None,
                    "panel_definition": f"{j_name}_vs_{s_name}",
                    "panel_side": "diagnostic",
                }
            )
    for s_name in ["S1", "S4"]:
        s_pred = SENIOR_FLAGS[s_name]
        specs.append(
            {
                "comparison": f"within_2024_{s_name}_arshkon_vs_asaniczka",
                "side_func": arshkon_or_asaniczka,
                "label_a": "arshkon_2024",
                "label_b": "asaniczka_2024",
                "predicate": s_pred,
                "panel_definition": s_name,
                "panel_side": "within_2024_senior",
            }
        )

    count_rows = []
    top_frames = []
    for spec in specs:
        metrics, top, counts, _ = run_comparison(
            full_rows,
            comparison=spec["comparison"],
            text_scope="full_text",
            side_func=spec["side_func"],
            label_a=spec["label_a"],
            label_b=spec["label_b"],
            stopwords=stopwords,
            ngram_n=1,
            exclude_aggregators=True,
            company_cap=50,
            extra_predicate=spec["predicate"],
            min_distinct_companies=20,
            top_k=20,
        )
        counts["panel_definition"] = spec["panel_definition"]
        counts["panel_side"] = spec["panel_side"]
        count_rows.append(counts)
        if not top.empty:
            top["panel_definition"] = spec["panel_definition"]
            top["panel_side"] = spec["panel_side"]
            top_frames.append(top)
    counts_df = pd.DataFrame(count_rows)
    terms_df = pd.concat(top_frames, ignore_index=True) if top_frames else pd.DataFrame()
    counts_df.to_csv(TABLE_DIR / "secondary_comparison_counts.csv", index=False)
    terms_df.to_csv(TABLE_DIR / "secondary_comparison_top_terms.csv", index=False)
    return counts_df, terms_df


def bigram_analysis(full_rows: Sequence[Dict[str, object]], stopwords: set[str]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    metrics, top, counts, _ = run_comparison(
        full_rows,
        comparison="primary_arshkon_vs_scraped_bigram_llm_excl_agg_cap50",
        text_scope="full_text",
        side_func=arshkon_or_scraped,
        label_a="arshkon_2024",
        label_b="scraped_2026",
        stopwords=stopwords,
        ngram_n=2,
        exclude_aggregators=True,
        company_cap=50,
        min_distinct_companies=20,
        top_k=100,
    )
    top.to_csv(TABLE_DIR / "log_odds_bigram_top_terms.csv", index=False)
    return top, counts


def t09_cross_validation(primary_selected: Sequence[Dict[str, object]]) -> pd.DataFrame:
    con = connect()
    labels = con.execute(
        f"SELECT uid, archetype_name FROM read_parquet('{T09_ARCHETYPES.as_posix()}')"
    ).fetchdf()
    con.close()
    label_map = dict(zip(labels["uid"], labels["archetype_name"]))
    counts: MutableMapping[Tuple[str, str], int] = defaultdict(int)
    side_totals: Counter = Counter()
    for row in primary_selected:
        side = row["side"]
        archetype = label_map.get(row["uid"])
        if not archetype:
            continue
        counts[(side, archetype)] += 1
        side_totals[side] += 1
    out_rows = []
    for (side, archetype), n in counts.items():
        out_rows.append({"side": side, "archetype_name": archetype, "n_labeled": n, "share_labeled": n / side_totals[side]})
    out = pd.DataFrame(out_rows)
    if not out.empty:
        pivot = out.pivot(index="archetype_name", columns="side", values="share_labeled").fillna(0)
        if {"arshkon_2024", "scraped_2026"}.issubset(pivot.columns):
            pivot["change_scraped_minus_arshkon"] = pivot["scraped_2026"] - pivot["arshkon_2024"]
        out = out.merge(
            pivot[["change_scraped_minus_arshkon"]].reset_index()
            if "change_scraped_minus_arshkon" in pivot.columns
            else pd.DataFrame(columns=["archetype_name", "change_scraped_minus_arshkon"]),
            on="archetype_name",
            how="left",
        )
        out = out.sort_values(["change_scraped_minus_arshkon", "side"], ascending=[False, True])
    out.to_csv(TABLE_DIR / "t09_nmf_cross_validation.csv", index=False)
    return out


def context_samples(rows: Sequence[Dict[str, object]], terms: Sequence[str], stopwords: set[str]) -> pd.DataFrame:
    wanted = list(dict.fromkeys(terms))[:20]
    samples = []
    term_counts = Counter()
    for row in rows:
        if row.get("side") != "scraped_2026":
            continue
        text = str(row.get("text") or "")
        toks = set(tokenize(text, stopwords))
        lower_text = re.sub(r"\s+", " ", text.lower())
        for term in wanted:
            if term_counts[term] >= 3:
                continue
            if (" " in term and term in lower_text) or (term in toks):
                idx = lower_text.find(term)
                start = max(0, idx - 140) if idx >= 0 else 0
                end = min(len(lower_text), idx + len(term) + 180) if idx >= 0 else 280
                samples.append(
                    {
                        "term": term,
                        "uid": row["uid"],
                        "source": row["source"],
                        "side": row["side"],
                        "company_name_canonical": row.get("company_name_canonical"),
                        "context_preview": lower_text[start:end],
                        "manual_precision_claim": "not_claimed_context_sample_only",
                    }
                )
                term_counts[term] += 1
    out = pd.DataFrame(samples)
    out.to_csv(TABLE_DIR / "top_term_context_samples.csv", index=False)
    return out


def write_plots(category_summary: pd.DataFrame, top_full: pd.DataFrame, emerging: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    if not category_summary.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_df = category_summary.copy()
        plot_df["scope_direction"] = plot_df["text_scope"] + "\n" + plot_df["direction"]
        pivot = plot_df.pivot_table(index="scope_direction", columns="category", values="share_top100", fill_value=0)
        categories = [c for c in ["ai_tool", "ai_domain", "tech_stack", "org_scope", "sys_design", "method", "credential", "soft_skill", "mgmt", "boilerplate", "noise"] if c in pivot.columns]
        pivot[categories].plot(kind="bar", stacked=True, ax=ax, width=0.75)
        ax.set_ylabel("Share of top-100 distinguishing terms")
        ax.set_xlabel("")
        ax.set_title("Distinguishing-Term Categories")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "category_summary_full_vs_section.png", dpi=150)
        plt.close()

    if not top_full.empty:
        b_heavy = top_full[top_full["direction"].str.contains("scraped_2026")].sort_values("z_b_minus_a", ascending=False).head(25)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(b_heavy["term"][::-1], b_heavy["z_b_minus_a"][::-1], color="#4C78A8")
        ax.set_xlabel("Log-odds z-score")
        ax.set_title("Top 2026-Heavy Terms")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "top_2026_heavy_terms.png", dpi=150)
        plt.close()

    if not emerging.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot = emerging.groupby(["status", "category"]).size().reset_index(name="n")
        pivot = plot.pivot(index="status", columns="category", values="n").fillna(0)
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("Terms")
        ax.set_xlabel("")
        ax.set_title("Emerging / Accelerating / Disappearing Terms")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "term_status_categories.png", dpi=150)
        plt.close()

    if not sensitivity.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sens = sensitivity[~sensitivity["comparison"].str.startswith("primary_")]
        ax.barh(sens["comparison"], sens["top_b_overlap_with_primary"], color="#54A24B")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Overlap with primary top 2026-heavy terms")
        ax.set_title("Sensitivity Top-Term Overlap")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "sensitivity_top_term_overlap.png", dpi=150)
        plt.close()


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    company_stoplist = load_company_stoplist()
    location_stoplist = load_location_stoplist()
    stopwords = BASE_STOPWORDS | JOB_GENERIC_STOPWORDS | company_stoplist | location_stoplist
    assert_regex_edges(stopwords)

    full_rows = load_llm_full_rows()
    section_rows = load_section_rows()
    raw_rows = load_raw_rows_for_arshkon_scraped()

    coverage_con = connect()
    coverage = coverage_con.execute(
        f"""
        SELECT source, period, text_source, count(*) AS rows
        FROM read_parquet('{SHARED_TEXT.as_posix()}')
        GROUP BY 1,2,3
        ORDER BY 1,2,3
        """
    ).fetchdf()
    coverage_con.close()
    coverage.to_csv(TABLE_DIR / "text_source_coverage.csv", index=False)

    all_counts = []

    full_metrics, full_top, full_counts, primary_selected = run_comparison(
        full_rows,
        comparison="primary_arshkon_vs_scraped_llm_excl_agg_cap50",
        text_scope="full_text",
        side_func=arshkon_or_scraped,
        label_a="arshkon_2024",
        label_b="scraped_2026",
        stopwords=stopwords,
        ngram_n=1,
        exclude_aggregators=True,
        company_cap=50,
        min_distinct_companies=20,
        top_k=100,
    )
    full_top.to_csv(TABLE_DIR / "log_odds_full_top_terms.csv", index=False)
    full_metrics.to_csv(TABLE_DIR / "primary_full_all_term_metrics.csv", index=False)
    all_counts.append(full_counts)

    section_metrics, section_top, section_counts, section_selected = run_comparison(
        section_rows,
        comparison="section_filtered_arshkon_vs_scraped_llm_excl_agg_cap50",
        text_scope="section_filtered_req_resp_pref",
        side_func=arshkon_or_scraped,
        label_a="arshkon_2024",
        label_b="scraped_2026",
        stopwords=stopwords,
        ngram_n=1,
        exclude_aggregators=True,
        company_cap=50,
        min_distinct_companies=20,
        top_k=100,
    )
    section_top.to_csv(TABLE_DIR / "log_odds_section_filtered_top_terms.csv", index=False)
    section_metrics.to_csv(TABLE_DIR / "section_filtered_all_term_metrics.csv", index=False)
    all_counts.append(section_counts)

    bigram_top, bigram_counts = bigram_analysis(full_rows, stopwords)
    all_counts.append(bigram_counts)

    category_summary = write_category_summary(full_top, section_top)
    write_overlap(full_top, section_top)
    emerging = emerging_accelerating_disappearing(full_metrics)

    sensitivity_summary, sensitivity_terms = sensitivity_rows(full_rows, raw_rows, stopwords)
    secondary_counts, secondary_terms = secondary_comparisons(full_rows, stopwords)
    archetypes = t09_cross_validation(primary_selected)

    context_terms = list(full_top[full_top["direction"].str.contains("scraped_2026")]["term"].head(20))
    context_samples(primary_selected, context_terms, stopwords)

    counts_df = pd.concat([pd.DataFrame(all_counts), sensitivity_summary, secondary_counts], ignore_index=True, sort=False)
    counts_df.to_csv(TABLE_DIR / "corpus_counts_all_comparisons.csv", index=False)

    metadata = pd.DataFrame(
        [
            {
                "company_stoplist_tokens": len(company_stoplist),
                "location_stoplist_tokens": len(location_stoplist),
                "total_stopwords": len(stopwords),
                "bertopic_deviation": "not_rerun_due_T09_instability_and_OOM_guidance",
                "cross_validation_used": "T09_NMF_archetype_labels",
                "primary_company_cap": 50,
                "primary_aggregator_policy": "exclude_aggregators",
                "primary_text_source": "description_core_llm via shared description_cleaned where text_source='llm'",
                "section_filtered_source": "T13 requirements_responsibilities_text includes responsibilities, requirements, preferred",
            }
        ]
    )
    metadata.to_csv(TABLE_DIR / "analysis_metadata.csv", index=False)

    write_plots(category_summary, full_top, emerging, sensitivity_summary)
    print(f"Wrote T12 outputs to {TABLE_DIR} and {FIG_DIR}")


if __name__ == "__main__":
    main()
