#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
OUT_DIR = ROOT / "exploration" / "artifacts" / "shared"

DEFAULT_FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe"

TOKEN_PATTERN = re.compile(r"(?<!\w)[\w.+#/-]+(?!\w)")
SPLIT_PATTERN = re.compile(r"[^a-z0-9+#.]+")


def assert_regex_sanity() -> None:
    sample = "C++, C#, .NET, Go, Next.js, Node.js, Python, React, Claude, Cursor, Agile, TDD"
    lower = sample.lower()
    assert re.search(r"(?<!\w)c\+\+(?!\w)", lower)
    assert re.search(r"(?<!\w)c#(?!\w)", lower)
    assert re.search(r"(?<!\w)\.net(?!\w)", lower)
    assert re.search(r"(?<!\w)next\.js(?!\w)", lower)
    assert re.search(r"(?<!\w)node\.js(?!\w)", lower)
    assert not re.search(r"(?<!\w)c\+\+(?!\w)", "cpp")
    assert not re.search(r"(?<!\w)c#(?!\w)", "c sharp")
    assert not re.search(r"(?<!\w)\.net(?!\w)", "internet")


def q(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def qone(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchone()[0]


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def normalize_token(value: str) -> str:
    return value.strip().lower()


def build_company_stoplist(con: duckdb.DuckDBPyConnection) -> list[str]:
    values = con.execute(
        f"""
        SELECT DISTINCT company_name_canonical
        FROM read_parquet('{DATA.as_posix()}')
        WHERE company_name_canonical IS NOT NULL
          AND company_name_canonical <> ''
        """
    ).to_arrow_table().column(0).to_pylist()
    tokens: set[str] = set()
    for value in values:
        if not value:
            continue
        for token in SPLIT_PATTERN.split(value.lower()):
            token = token.strip()
            if token:
                tokens.add(token)
    return sorted(tokens)


def build_protected_tokens() -> set[str]:
    return {
        "go",
        "r",
        "c",
        "ai",
        "ml",
        "llm",
        "sql",
        "qa",
        "ux",
        "ui",
        "api",
        "db",
        "js",
        "ts",
        "aws",
        "gcp",
        "ios",
        "os",
        "net",
        "node",
        "next",
        "dotnet",
    }


def build_text_cleaner(company_stoplist: set[str]):
    stopwords = set(ENGLISH_STOP_WORDS) - {"go"}
    stopwords |= company_stoplist
    protected = build_protected_tokens()

    def clean(text: str | None) -> str:
        if text is None:
            return ""
        lowered = text.lower().replace("\r", " ").replace("\n", " ")

        def repl(match: re.Match[str]) -> str:
            token = normalize_token(match.group(0))
            if token in protected:
                return token
            if token in stopwords:
                return " "
            return token

        cleaned = TOKEN_PATTERN.sub(repl, lowered)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    return clean, stopwords, protected


@dataclass(frozen=True)
class TechFeature:
    name: str
    pattern: str


def build_tech_taxonomy() -> list[TechFeature]:
    tech: list[TechFeature] = []

    def add(name: str, pattern: str) -> None:
        tech.append(TechFeature(name=name, pattern=pattern))

    # Languages
    add("python", r"(?<!\w)python(?!\w)")
    add("java", r"(?<!\w)java(?!\w)")
    add("javascript", r"(?<!\w)javascript(?!\w)")
    add("typescript", r"(?<!\w)typescript(?!\w)")
    add("go", r"(?<!\w)(go|golang)(?!\w)")
    add("rust", r"(?<!\w)rust(?!\w)")
    add("c_plus_plus", r"(?<!\w)c\+\+(?!\w)")
    add("c_sharp", r"(?<!\w)c#(?!\w)")
    add("ruby", r"(?<!\w)ruby(?!\w)")
    add("kotlin", r"(?<!\w)kotlin(?!\w)")
    add("swift", r"(?<!\w)swift(?!\w)")
    add("scala", r"(?<!\w)scala(?!\w)")
    add("php", r"(?<!\w)php(?!\w)")
    add("perl", r"(?<!\w)perl(?!\w)")
    add("dart", r"(?<!\w)dart(?!\w)")
    add("lua", r"(?<!\w)lua(?!\w)")
    add("haskell", r"(?<!\w)haskell(?!\w)")
    add("clojure", r"(?<!\w)clojure(?!\w)")
    add("elixir", r"(?<!\w)elixir(?!\w)")
    add("erlang", r"(?<!\w)erlang(?!\w)")
    add("julia", r"(?<!\w)julia(?!\w)")
    add("matlab", r"(?<!\w)matlab(?!\w)")
    add("r_language", r"(?<!\w)r(?!\w)")

    # Frontend
    add("html", r"(?<!\w)html(?!\w)")
    add("css", r"(?<!\w)css(?!\w)")
    add("sass", r"(?<!\w)sass(?!\w)")
    add("less", r"(?<!\w)less(?!\w)")
    add("react", r"(?<!\w)react(?!\w)")
    add("react_native", r"(?<!\w)react native(?!\w)")
    add("angular", r"(?<!\w)angular(?!\w)")
    add("vue", r"(?<!\w)vue(?!\w)")
    add("nextjs", r"(?<!\w)next\.?js(?!\w)")
    add("svelte", r"(?<!\w)svelte(?!\w)")
    add("redux", r"(?<!\w)redux(?!\w)")
    add("webpack", r"(?<!\w)webpack(?!\w)")
    add("vite", r"(?<!\w)vite(?!\w)")
    add("storybook", r"(?<!\w)storybook(?!\w)")
    add("tailwind", r"(?<!\w)tailwind(?!\w)")
    add("bootstrap", r"(?<!\w)bootstrap(?!\w)")
    add("material_ui", r"(?<!\w)material ui(?!\w)")

    # Backend / web
    add("nodejs", r"(?<!\w)node\.?js(?!\w)")
    add("express", r"(?<!\w)express(?!\w)")
    add("nestjs", r"(?<!\w)nestjs(?!\w)")
    add("django", r"(?<!\w)django(?!\w)")
    add("django_rest_framework", r"(?<!\w)django rest framework(?!\w)")
    add("flask", r"(?<!\w)flask(?!\w)")
    add("spring", r"(?<!\w)spring(?!\w)")
    add("spring_boot", r"(?<!\w)spring boot(?!\w)")
    add("dotnet", r"(?<!\w)\.net(?!\w)")
    add("aspnet", r"(?<!\w)asp\.?net(?!\w)")
    add("rails", r"(?<!\w)rails(?!\w)")
    add("laravel", r"(?<!\w)laravel(?!\w)")
    add("fastapi", r"(?<!\w)fastapi(?!\w)")
    add("phoenix", r"(?<!\w)phoenix(?!\w)")
    add("tornado", r"(?<!\w)tornado(?!\w)")
    add("bottle", r"(?<!\w)bottle(?!\w)")
    add("grpc", r"(?<!\w)grpc(?!\w)")
    add("graphql", r"(?<!\w)graphql(?!\w)")
    add("rest_api", r"(?<!\w)(rest api|restful api)(?!\w)")
    add("microservices", r"(?<!\w)microservices(?!\w)")
    add("event_driven", r"(?<!\w)event driven(?!\w)")
    add("oauth", r"(?<!\w)oauth2?(?!\w)")

    # Data / analytics
    add("sql", r"(?<!\w)sql(?!\w)")
    add("postgresql", r"(?<!\w)(postgresql|postgres)(?!\w)")
    add("mysql", r"(?<!\w)mysql(?!\w)")
    add("sqlite", r"(?<!\w)sqlite(?!\w)")
    add("mongodb", r"(?<!\w)mongodb(?!\w)")
    add("redis", r"(?<!\w)redis(?!\w)")
    add("cassandra", r"(?<!\w)cassandra(?!\w)")
    add("kafka", r"(?<!\w)kafka(?!\w)")
    add("spark", r"(?<!\w)spark(?!\w)")
    add("hadoop", r"(?<!\w)hadoop(?!\w)")
    add("hive", r"(?<!\w)hive(?!\w)")
    add("presto", r"(?<!\w)presto(?!\w)")
    add("trino", r"(?<!\w)trino(?!\w)")
    add("snowflake", r"(?<!\w)snowflake(?!\w)")
    add("databricks", r"(?<!\w)databricks(?!\w)")
    add("dbt", r"(?<!\w)dbt(?!\w)")
    add("elasticsearch", r"(?<!\w)elasticsearch(?!\w)")
    add("airflow", r"(?<!\w)airflow(?!\w)")
    add("luigi", r"(?<!\w)luigi(?!\w)")
    add("airbyte", r"(?<!\w)airbyte(?!\w)")
    add("delta_lake", r"(?<!\w)delta lake(?!\w)")
    add("tableau", r"(?<!\w)tableau(?!\w)")
    add("powerbi", r"(?<!\w)power bi(?!\w)")
    add("looker", r"(?<!\w)looker(?!\w)")
    add("superset", r"(?<!\w)superset(?!\w)")
    add("metabase", r"(?<!\w)metabase(?!\w)")
    add("bigquery", r"(?<!\w)bigquery(?!\w)")
    add("redshift", r"(?<!\w)redshift(?!\w)")

    # Cloud / DevOps
    add("aws", r"(?<!\w)aws(?!\w)")
    add("azure", r"(?<!\w)azure(?!\w)")
    add("gcp", r"(?<!\w)(gcp|google cloud platform|google cloud)(?!\w)")
    add("kubernetes", r"(?<!\w)(kubernetes|k8s)(?!\w)")
    add("docker", r"(?<!\w)docker(?!\w)")
    add("terraform", r"(?<!\w)terraform(?!\w)")
    add("ansible", r"(?<!\w)ansible(?!\w)")
    add("helm", r"(?<!\w)helm(?!\w)")
    add("jenkins", r"(?<!\w)jenkins(?!\w)")
    add("github_actions", r"(?<!\w)github actions(?!\w)")
    add("gitlab_ci", r"(?<!\w)gitlab ci(?!\w)")
    add("circleci", r"(?<!\w)circleci(?!\w)")
    add("argo_cd", r"(?<!\w)argocd(?!\w)|(?<!\w)argo cd(?!\w)")
    add("openshift", r"(?<!\w)openshift(?!\w)")
    add("nomad", r"(?<!\w)nomad(?!\w)")
    add("prometheus", r"(?<!\w)prometheus(?!\w)")
    add("grafana", r"(?<!\w)grafana(?!\w)")
    add("datadog", r"(?<!\w)datadog(?!\w)")
    add("new_relic", r"(?<!\w)new relic(?!\w)")
    add("splunk", r"(?<!\w)splunk(?!\w)")
    add("opentelemetry", r"(?<!\w)opentelemetry(?!\w)")
    add("linux", r"(?<!\w)linux(?!\w)")
    add("bash", r"(?<!\w)bash(?!\w)")
    add("git", r"(?<!\w)git(?!\w)")
    add("serverless", r"(?<!\w)serverless(?!\w)")
    add("cloudformation", r"(?<!\w)cloudformation(?!\w)")
    add("pulumi", r"(?<!\w)pulumi(?!\w)")

    # Testing / practices
    add("junit", r"(?<!\w)junit(?!\w)")
    add("pytest", r"(?<!\w)pytest(?!\w)")
    add("jest", r"(?<!\w)jest(?!\w)")
    add("mocha", r"(?<!\w)mocha(?!\w)")
    add("chai", r"(?<!\w)chai(?!\w)")
    add("selenium", r"(?<!\w)selenium(?!\w)")
    add("cypress", r"(?<!\w)cypress(?!\w)")
    add("playwright", r"(?<!\w)playwright(?!\w)")
    add("tdd", r"(?<!\w)tdd(?!\w)")
    add("agile", r"(?<!\w)agile(?!\w)")
    add("scrum", r"(?<!\w)scrum(?!\w)")
    add("kanban", r"(?<!\w)kanban(?!\w)")
    add("ci_cd", r"(?<!\w)ci\s*/\s*cd(?!\w)")
    add("code_review", r"(?<!\w)code review(?!\w)")
    add("pair_programming", r"(?<!\w)pair programming(?!\w)")
    add("unit_testing", r"(?<!\w)unit testing(?!\w)")
    add("integration_testing", r"(?<!\w)integration testing(?!\w)")
    add("bdd", r"(?<!\w)bdd(?!\w)")
    add("qa", r"(?<!\w)qa(?!\w)|(?<!\w)quality assurance(?!\w)")

    # AI / ML
    add("machine_learning", r"(?<!\w)machine learning(?!\w)")
    add("deep_learning", r"(?<!\w)deep learning(?!\w)")
    add("data_science", r"(?<!\w)data science(?!\w)")
    add("statistics", r"(?<!\w)statistics(?!\w)")
    add("nlp", r"(?<!\w)nlp(?!\w)")
    add("computer_vision", r"(?<!\w)computer vision(?!\w)")
    add("generative_ai", r"(?<!\w)(generative ai|gen ai)(?!\w)")
    add("tensorflow", r"(?<!\w)tensorflow(?!\w)")
    add("pytorch", r"(?<!\w)pytorch(?!\w)")
    add("scikit_learn", r"(?<!\w)scikit[- ]learn(?!\w)")
    add("pandas", r"(?<!\w)pandas(?!\w)")
    add("numpy", r"(?<!\w)numpy(?!\w)")
    add("jupyter", r"(?<!\w)jupyter(?!\w)")
    add("xgboost", r"(?<!\w)xgboost(?!\w)")
    add("lightgbm", r"(?<!\w)lightgbm(?!\w)")
    add("catboost", r"(?<!\w)catboost(?!\w)")
    add("mlflow", r"(?<!\w)mlflow(?!\w)")
    add("kubeflow", r"(?<!\w)kubeflow(?!\w)")
    add("ray", r"(?<!\w)ray(?!\w)")
    add("hugging_face", r"(?<!\w)hugging face(?!\w)")
    add("openai_api", r"(?<!\w)openai api(?!\w)")
    add("anthropic_api", r"(?<!\w)anthropic api(?!\w)")
    add("claude_api", r"(?<!\w)claude api(?!\w)")
    add("gemini_api", r"(?<!\w)gemini api(?!\w)")
    add("langchain", r"(?<!\w)langchain(?!\w)")
    add("langgraph", r"(?<!\w)langgraph(?!\w)")
    add("llamaindex", r"(?<!\w)llamaindex(?!\w)")
    add("rag", r"(?<!\w)rag(?!\w)")
    add("vector_db", r"(?<!\w)(vector database|vector db)(?!\w)")
    add("pinecone", r"(?<!\w)pinecone(?!\w)")
    add("weaviate", r"(?<!\w)weaviate(?!\w)")
    add("chroma", r"(?<!\w)chroma(?!\w)")
    add("milvus", r"(?<!\w)milvus(?!\w)")
    add("faiss", r"(?<!\w)faiss(?!\w)")
    add("prompt_engineering", r"(?<!\w)prompt engineering(?!\w)")
    add("fine_tuning", r"(?<!\w)fine[- ]tuning(?!\w)")
    add("mcp", r"(?<!\w)mcp(?!\w)")
    add("llm", r"(?<!\w)llm(s)?(?!\w)")
    add("copilot", r"(?<!\w)copilot(?!\w)")
    add("cursor", r"(?<!\w)cursor(?!\w)")
    add("chatgpt", r"(?<!\w)chatgpt(?!\w)")
    add("claude", r"(?<!\w)claude(?!\w)")
    add("gemini", r"(?<!\w)gemini(?!\w)")
    add("codex", r"(?<!\w)codex(?!\w)")
    add("agent", r"(?<!\w)agent(s)?(?!\w)")

    return tech


def build_semantic_patterns() -> dict[str, str]:
    return {
        "ai_any": r"(machine learning|deep learning|nlp|computer vision|generative ai|gen ai|llm|large language model|rag|langchain|langgraph|llamaindex|prompt engineering|fine[- ]tuning|openai api|anthropic api|claude api|gemini api|copilot|cursor|chatgpt|claude|gemini|codex|mcp|agent)",
        "ai_tool": r"(llm|large language model|rag|langchain|langgraph|llamaindex|prompt engineering|fine[- ]tuning|openai api|anthropic api|claude api|gemini api|copilot|cursor|chatgpt|claude|gemini|codex|mcp|agent)",
        "ai_domain": r"(machine learning|deep learning|nlp|computer vision|data science|statistics|generative ai|gen ai)",
        "management_strong": r"(manage|managed|manager|mentor|coach|hire|hiring|direct reports|performance review|people manager|team lead)",
        "management_broad": r"(lead|leading|leadership|team|stakeholder|coordinate|collaborate|collaboration|partner|guide)",
        "scope_term": r"(ownership|end[- ]to[- ]end|cross[- ]functional|stakeholder|autonomous|initiative|drive|own the|owning|architecture|system design)",
        "soft_skill": r"(communication|collaboration|problem[- ]solving|teamwork|interpersonal|adaptability|presentation|written communication|verbal communication)",
        "credential": r"(bachelor|bs\b|ba\b|master|ms\b|phd\b|degree|certification|certified|license|equivalent experience)",
        "boilerplate": r"(salary|benefits|compensation|pay|equity|bonus|dental|401k|pto|culture|mission|values|diversity|inclusion|sponsorship|visa|employees|people)",
    }


def compile_patterns(patterns: dict[str, str]) -> dict[str, re.Pattern[str]]:
    return {name: re.compile(pattern, re.I) for name, pattern in patterns.items()}


def clean_text(text: str | None, stopwords: set[str], protected_tokens: set[str]) -> str:
    if text is None:
        return ""
    lowered = text.lower().replace("\r", " ").replace("\n", " ")

    def repl(match: re.Match[str]) -> str:
        token = normalize_token(match.group(0))
        if token in protected_tokens:
            return token
        if token in stopwords:
            return " "
        return token

    cleaned = TOKEN_PATTERN.sub(repl, lowered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def first_n_tokens(text: str, n: int = 512) -> str:
    if not text:
        return ""
    tokens = text.split()
    return " ".join(tokens[:n])


def build_tech_flags(text: str, tech_patterns: list[TechFeature]) -> dict[str, bool]:
    return {feature.name: bool(re.search(feature.pattern, text, re.I)) for feature in tech_patterns}


def count_binary_flags(flags: dict[str, bool]) -> int:
    return int(sum(1 for value in flags.values() if value))


def row_summary_stats(values: pd.Series, kind: str) -> dict[str, float]:
    if values.empty:
        return {"value": float("nan"), "n": 0}
    if kind == "mean":
        return {"value": float(values.mean()), "n": int(values.shape[0])}
    if kind == "median":
        return {"value": float(values.median()), "n": int(values.shape[0])}
    if kind == "share":
        return {"value": float(values.mean()), "n": int(values.shape[0])}
    raise ValueError(kind)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2)
    if pooled <= 0:
        return float("nan")
    return float((x.mean() - y.mean()) / math.sqrt(pooled))


def proportion_diff(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    return float(x.mean() - y.mean())


def effect_size(metric_type: str, x: np.ndarray, y: np.ndarray) -> float:
    if metric_type == "binary":
        return proportion_diff(x, y)
    return cohens_d(x, y)


def summarize_metric(
    name: str,
    metric_type: str,
    summary_kind: str,
    values_by_source: dict[str, np.ndarray],
) -> dict[str, object]:
    arshkon = values_by_source["kaggle_arshkon"]
    asaniczka = values_by_source["kaggle_asaniczka"]
    scraped = values_by_source["scraped"]

    arshkon_summary = row_summary_stats(pd.Series(arshkon), summary_kind)
    asaniczka_summary = row_summary_stats(pd.Series(asaniczka), summary_kind)
    scraped_summary = row_summary_stats(pd.Series(scraped), summary_kind)
    within = effect_size(metric_type, asaniczka, arshkon)
    cross = effect_size(metric_type, scraped, arshkon)
    ratio = abs(cross) / abs(within) if within not in (0, None) and np.isfinite(within) and within != 0 else float("nan")
    return {
        "metric": name,
        "metric_type": metric_type,
        "summary_kind": summary_kind,
        "arshkon_value": arshkon_summary["value"],
        "asaniczka_value": asaniczka_summary["value"],
        "scraped_value": scraped_summary["value"],
        "within_2024_effect": within,
        "cross_period_effect": cross,
        "calibration_ratio": ratio,
        "n_arshkon": arshkon_summary["n"],
        "n_asaniczka": asaniczka_summary["n"],
        "n_scraped": scraped_summary["n"],
    }


def write_parquet_from_rows(rows: list[dict], path: Path) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shared exploration artifacts.")
    parser.add_argument("--batch-size", type=int, default=2048, help="DuckDB Arrow reader batch size.")
    parser.add_argument("--embedding-batch-size", type=int, default=256, help="SentenceTransformer batch size.")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR, help="Shared artifact output directory.")
    args = parser.parse_args()

    assert_regex_sanity()

    start = time.perf_counter()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    tech_features = build_tech_taxonomy()
    semantic_patterns = compile_patterns(build_semantic_patterns())

    company_stoplist_list = build_company_stoplist(con)
    company_stoplist = set(company_stoplist_list)
    (out_dir / "company_stoplist.txt").write_text("\n".join(company_stoplist_list) + "\n", encoding="utf-8")

    clean_fn, _, _ = build_text_cleaner(company_stoplist)

    query = f"""
        SELECT
            uid,
            source,
            period,
            description_length,
            seniority_final,
            seniority_3level,
            is_aggregator,
            company_name_canonical,
            metro_area,
            yoe_extracted,
            swe_classification_tier,
            seniority_final_source,
            llm_extraction_coverage,
            description_core_llm,
            description
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {DEFAULT_FILTER}
        ORDER BY uid
    """

    reader = con.execute(query).to_arrow_reader()

    cleaned_rows: list[dict[str, object]] = []
    llm_rows: list[dict[str, object]] = []
    llm_texts: list[str] = []
    llm_uids: list[str] = []
    tech_row_data: list[dict[str, object]] = []

    metrics: dict[str, dict[str, list[float]]] = {
        "description_length": {k: [] for k in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]},
        "yoe": {k: [] for k in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]},
        "yoe_le_2": {k: [] for k in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]},
        "yoe_le_3": {k: [] for k in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]},
        "yoe_ge_5": {k: [] for k in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]},
        "entry_final": {k: [] for k in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]},
        "unknown_final": {k: [] for k in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]},
        "junior3": {k: [] for k in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]},
        "tech_count": {k: [] for k in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]},
        "tech_density": {k: [] for k in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]},
        "requirement_breadth": {k: [] for k in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]},
    }
    for name in [
        "ai_any",
        "ai_tool",
        "ai_domain",
        "management_strong",
        "management_broad",
        "scope_term",
        "soft_skill",
        "credential",
        "boilerplate",
    ]:
        metrics[name] = {k: [] for k in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]}
    for feature in tech_features:
        metrics[feature.name] = {k: [] for k in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]}

    llm_coverage_by_source = {k: {"labeled": 0, "raw": 0} for k in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]}
    source_counts = {k: 0 for k in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]}
    period_counts: dict[tuple[str, str], int] = {}
    source_period_counts: dict[tuple[str, str], int] = {}

    batch_count = 0
    total_rows = 0
    for batch in reader:
        batch_count += 1
        cols = batch.to_pydict()
        n = len(cols["uid"])
        for i in range(n):
            uid = cols["uid"][i]
            source = cols["source"][i]
            period = cols["period"][i]
            desc_len = float(cols["description_length"][i] or 0)
            seniority_final = cols["seniority_final"][i] or "unknown"
            seniority_3level = cols["seniority_3level"][i] or "unknown"
            is_aggregator = bool(cols["is_aggregator"][i]) if cols["is_aggregator"][i] is not None else False
            company_name_canonical = cols["company_name_canonical"][i] or ""
            metro_area = cols["metro_area"][i] or ""
            yoe = float(cols["yoe_extracted"][i]) if cols["yoe_extracted"][i] is not None else float("nan")
            swe_classification_tier = cols["swe_classification_tier"][i] or ""
            seniority_final_source = cols["seniority_final_source"][i] or ""
            llm_cov = cols["llm_extraction_coverage"][i] or "raw"
            raw_text = cols["description"][i] or ""
            llm_text = cols["description_core_llm"][i] or ""
            base_text = llm_text if llm_cov == "labeled" else raw_text
            cleaned = clean_fn(base_text)

            cleaned_rows.append(
                {
                    "uid": uid,
                    "description_cleaned": cleaned,
                    "text_source": "llm" if llm_cov == "labeled" else "raw",
                    "source": source,
                    "period": period,
                    "seniority_final": seniority_final,
                    "seniority_3level": seniority_3level,
                    "is_aggregator": is_aggregator,
                    "company_name_canonical": company_name_canonical,
                    "metro_area": metro_area,
                    "yoe_extracted": yoe,
                    "swe_classification_tier": swe_classification_tier,
                    "seniority_final_source": seniority_final_source,
                }
            )
            if llm_cov == "labeled":
                llm_rows.append(
                    {
                        "uid": uid,
                        "description_cleaned": cleaned,
                        "source": source,
                        "period": period,
                    }
                )
                llm_uids.append(uid)
                llm_texts.append(first_n_tokens(cleaned, 512))

            tech_flags = build_tech_flags(cleaned, tech_features)
            tech_row_data.append({"uid": uid, **tech_flags})
            tech_count = count_binary_flags(tech_flags)

            semantic_flags = {name: bool(pattern.search(cleaned)) for name, pattern in semantic_patterns.items()}
            req_breadth = sum(
                [
                    tech_count > 0,
                    semantic_flags["ai_any"],
                    semantic_flags["management_broad"] or semantic_flags["management_strong"],
                    semantic_flags["scope_term"],
                    semantic_flags["soft_skill"],
                    semantic_flags["credential"],
                ]
            )

            source_counts[source] += 1
            period_counts[(source, period)] = period_counts.get((source, period), 0) + 1
            source_period_counts[(source, period)] = source_period_counts.get((source, period), 0) + 1
            llm_coverage_by_source[source]["labeled" if llm_cov == "labeled" else "raw"] += 1

            metrics["description_length"][source].append(desc_len)
            metrics["yoe"][source].append(yoe)
            metrics["yoe_le_2"][source].append(1.0 if np.isfinite(yoe) and yoe <= 2 else 0.0)
            metrics["yoe_le_3"][source].append(1.0 if np.isfinite(yoe) and yoe <= 3 else 0.0)
            metrics["yoe_ge_5"][source].append(1.0 if np.isfinite(yoe) and yoe >= 5 else 0.0)
            metrics["entry_final"][source].append(1.0 if seniority_final == "entry" else 0.0)
            metrics["unknown_final"][source].append(1.0 if seniority_final == "unknown" else 0.0)
            metrics["junior3"][source].append(1.0 if seniority_3level == "junior" else 0.0)
            metrics["tech_count"][source].append(float(tech_count))
            metrics["tech_density"][source].append((float(tech_count) / desc_len * 1000.0) if desc_len > 0 else float("nan"))
            metrics["requirement_breadth"][source].append(float(req_breadth))
            for name, flag in semantic_flags.items():
                metrics[name][source].append(1.0 if flag else 0.0)
            for name, flag in tech_flags.items():
                metrics[name][source].append(1.0 if flag else 0.0)
            total_rows += 1

        if batch_count % 20 == 0:
            print(f"Processed {total_rows:,} rows...")

    cleaned_path = out_dir / "swe_cleaned_text.parquet"
    write_parquet_from_rows(cleaned_rows, cleaned_path)

    tech_path = out_dir / "swe_tech_matrix.parquet"
    write_parquet_from_rows(tech_row_data, tech_path)

    skills_query = f"""
        SELECT uid, skills_raw
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {DEFAULT_FILTER}
          AND source = 'kaggle_asaniczka'
          AND skills_raw IS NOT NULL
          AND skills_raw <> ''
        ORDER BY uid
    """
    skills_df = q(con, skills_query)
    skill_rows: list[dict[str, str]] = []
    for _, row in skills_df.iterrows():
        uid = row["uid"]
        raw = str(row["skills_raw"])
        seen: set[str] = set()
        for skill in re.split(r"[,;]", raw):
            skill_clean = re.sub(r"\s+", " ", skill.strip().lower())
            if not skill_clean:
                continue
            if skill_clean in seen:
                continue
            seen.add(skill_clean)
            skill_rows.append({"uid": uid, "skill": skill_clean})
    write_parquet_from_rows(skill_rows, out_dir / "asaniczka_structured_skills.parquet")

    # Embeddings: only rows with labeled LLM text.
    embed_status = "complete"
    embed_error = ""
    embedding_dim = 0
    embedding_covered_rows = 0
    covered_uids: list[str] = []
    if llm_texts:
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embedding_dim = model.get_sentence_embedding_dimension()
            embeddings_chunks: list[np.ndarray] = []
            for start in range(0, len(llm_texts), args.embedding_batch_size):
                end = min(start + args.embedding_batch_size, len(llm_texts))
                batch_texts = llm_texts[start:end]
                batch_embeddings = model.encode(
                    batch_texts,
                    batch_size=args.embedding_batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
                batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32)
                embeddings_chunks.append(batch_embeddings)
                covered_uids.extend(llm_uids[start:end])
            embeddings = np.concatenate(embeddings_chunks, axis=0) if embeddings_chunks else np.empty((0, embedding_dim), dtype=np.float32)
            np.save(out_dir / "swe_embeddings.npy", embeddings.astype(np.float32, copy=False))
            index_table = pa.table({"row_index": np.arange(len(covered_uids), dtype=np.int64), "uid": covered_uids})
            pq.write_table(index_table, out_dir / "swe_embedding_index.parquet")
            embedding_covered_rows = len(covered_uids)
        except Exception as exc:  # noqa: BLE001
            embed_status = "partial"
            embed_error = repr(exc)
            covered = 0
            if "embeddings_chunks" in locals():
                try:
                    if embeddings_chunks:
                        embeddings = np.concatenate(embeddings_chunks, axis=0).astype(np.float32, copy=False)
                        covered = embeddings.shape[0]
                    else:
                        embeddings = np.empty((0, embedding_dim or 384), dtype=np.float32)
                except Exception:  # noqa: BLE001
                    embeddings = np.empty((0, embedding_dim or 384), dtype=np.float32)
            else:
                embeddings = np.empty((0, embedding_dim or 384), dtype=np.float32)
            np.save(out_dir / "swe_embeddings.npy", embeddings)
            covered_uids = llm_uids[:covered]
            embedding_covered_rows = len(covered_uids)
            index_table = pa.table({"row_index": np.arange(len(covered_uids), dtype=np.int64), "uid": covered_uids})
            pq.write_table(index_table, out_dir / "swe_embedding_index.parquet")
            print(f"Embedding build failed after {covered:,} rows: {embed_error}")
    else:
        np.save(out_dir / "swe_embeddings.npy", np.empty((0, 384), dtype=np.float32))
        pq.write_table(pa.table({"row_index": pa.array([], type=pa.int64()), "uid": pa.array([], type=pa.string())}), out_dir / "swe_embedding_index.parquet")

    # Calibration table.
    grouped = {metric: {source: np.asarray(values[source], dtype=float) for source in values} for metric, values in metrics.items()}

    rows: list[dict[str, object]] = []
    metric_specs = [
        ("description_length_mean", "description_length", "continuous", "mean"),
        ("description_length_median", "description_length", "continuous", "median"),
        ("yoe_extracted_mean", "yoe", "continuous", "mean"),
        ("yoe_extracted_median", "yoe", "continuous", "median"),
        ("yoe_le_2_share", "yoe_le_2", "binary", "share"),
        ("yoe_le_3_share", "yoe_le_3", "binary", "share"),
        ("yoe_ge_5_share", "yoe_ge_5", "binary", "share"),
        ("seniority_final_entry_share", "entry_final", "binary", "share"),
        ("seniority_final_unknown_share", "unknown_final", "binary", "share"),
        ("seniority_3level_junior_share", "junior3", "binary", "share"),
        ("tech_count_mean", "tech_count", "continuous", "mean"),
        ("tech_count_median", "tech_count", "continuous", "median"),
        ("tech_density_mean", "tech_density", "continuous", "mean"),
        ("requirement_breadth_mean", "requirement_breadth", "continuous", "mean"),
        ("requirement_breadth_median", "requirement_breadth", "continuous", "median"),
        ("ai_any_share", "ai_any", "binary", "share"),
        ("ai_tool_share", "ai_tool", "binary", "share"),
        ("ai_domain_share", "ai_domain", "binary", "share"),
        ("management_strong_share", "management_strong", "binary", "share"),
        ("management_broad_share", "management_broad", "binary", "share"),
        ("scope_term_share", "scope_term", "binary", "share"),
        ("soft_skill_share", "soft_skill", "binary", "share"),
        ("credential_share", "credential", "binary", "share"),
        ("boilerplate_share", "boilerplate", "binary", "share"),
        ("python_share", "python", "binary", "share"),
        ("java_share", "java", "binary", "share"),
        ("javascript_share", "javascript", "binary", "share"),
        ("typescript_share", "typescript", "binary", "share"),
        ("go_share", "go", "binary", "share"),
        ("c_plus_plus_share", "c_plus_plus", "binary", "share"),
        ("c_sharp_share", "c_sharp", "binary", "share"),
        ("dotnet_share", "dotnet", "binary", "share"),
        ("sql_share", "sql", "binary", "share"),
        ("aws_share", "aws", "binary", "share"),
        ("kubernetes_share", "kubernetes", "binary", "share"),
        ("docker_share", "docker", "binary", "share"),
        ("react_share", "react", "binary", "share"),
        ("pytorch_share", "pytorch", "binary", "share"),
        ("langchain_share", "langchain", "binary", "share"),
        ("copilot_share", "copilot", "binary", "share"),
        ("prompt_engineering_share", "prompt_engineering", "binary", "share"),
        ("agile_share", "agile", "binary", "share"),
        ("tdd_share", "tdd", "binary", "share"),
    ]
    for metric_name, source_name, metric_type, summary_kind in metric_specs:
        rows.append(
            summarize_metric(
                metric_name,
                metric_type,
                summary_kind,
                grouped[source_name],
            )
        )

    calibration = pd.DataFrame(rows)
    calibration.sort_values("metric", inplace=True)
    calibration.to_csv(out_dir / "calibration_table.csv", index=False)

    # README.
    cleaned_df = pd.DataFrame(cleaned_rows)
    text_source_counts = cleaned_df.groupby(["source", "text_source"]).size().reset_index(name="n")
    overall_text_source = cleaned_df["text_source"].value_counts().to_dict()
    period_counts_df = (
        cleaned_df.groupby(["source", "period"]).size().reset_index(name="n").sort_values(["source", "period"])
    )
    llm_cov_df = pd.DataFrame(
        [
            {
                "source": source,
                "labeled": counts["labeled"],
                "raw": counts["raw"],
                "labeled_share": counts["labeled"] / max(1, counts["labeled"] + counts["raw"]),
            }
            for source, counts in llm_coverage_by_source.items()
        ]
    )

    embedding_rows = embedding_covered_rows
    embedding_coverage = embedding_rows / max(1, len(cleaned_rows))
    build_seconds = time.perf_counter() - start

    readme = f"""# Shared preprocessing artifacts

Build date: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}
Build time: {build_seconds:,.1f} seconds

## Contents

| Artifact | Path | Rows / shape | Notes |
|---|---|---:|---|
| Cleaned text | `swe_cleaned_text.parquet` | {len(cleaned_rows):,} rows | `description_cleaned`, `text_source`, and wave-friendly metadata for SWE LinkedIn rows |
| Embeddings | `swe_embeddings.npy` | {embedding_rows:,} rows × {embedding_dim or 384} dims | {embed_status}{f' (partial: {embed_error})' if embed_status == 'partial' else ''} |
| Embedding index | `swe_embedding_index.parquet` | {embedding_rows:,} rows | Maps embedding row index to `uid` |
| Technology matrix | `swe_tech_matrix.parquet` | {len(tech_row_data):,} rows × {len(tech_features) + 1} cols | Binary mention matrix from cleaned text |
| Company stoplist | `company_stoplist.txt` | {len(company_stoplist_list):,} tokens | One token per line |
| Structured skills | `asaniczka_structured_skills.parquet` | {len(skill_rows):,} rows | Parsed comma-separated `skills_raw` for asaniczka SWE |
| Calibration table | `calibration_table.csv` | {len(calibration):,} rows | Within-2024 vs arshkon-vs-scraped calibration metrics |

## SWE LinkedIn coverage

Total rows: {len(cleaned_rows):,}

Text source distribution:
{cleaned_df['text_source'].value_counts().to_string()}

By source:
{text_source_counts.to_string(index=False)}

LLM extraction coverage by source:
{llm_cov_df.to_string(index=False)}

Distinct periods by source:
{period_counts_df.to_string(index=False)}

## Notes

- `text_source = 'llm'` uses `description_core_llm`; `text_source = 'raw'` falls back to raw `description` after company-name and stopword stripping.
- Company stopwords are drawn from all `company_name_canonical` tokens across the unified dataset. A small protected token set preserves obvious technology tokens such as `go`, `r`, `c`, `.net`, `node.js`, and `next.js`.
- Embeddings are computed only for `text_source = 'llm'` rows, truncated to the first 512 tokens before encoding with `all-MiniLM-L6-v2`.
- The calibration table uses all SWE LinkedIn rows under the default filters and compares arshkon to asaniczka within 2024, then arshkon to scraped for the current cross-period signal.
- Rows with `llm_extraction_coverage = 'labeled'` are the only rows with LLM-cleaned text. Thin coverage remains a limitation for text-heavy downstream tasks.
- Embedding coverage: {embedding_coverage:.1%} of cleaned SWE LinkedIn rows.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Shared preprocessing complete in {build_seconds:,.1f}s")
    print(f"Cleaned rows: {len(cleaned_rows):,}")
    print(f"LLM text rows: {embedding_rows:,}")
    print(f"Embedding status: {embed_status}")


if __name__ == "__main__":
    main()
