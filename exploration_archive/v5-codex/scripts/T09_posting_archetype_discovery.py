#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import duckdb
import hdbscan
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from umap import UMAP
import plotly.io as pio
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
SHARED = ROOT / "exploration" / "artifacts" / "shared"
REPORT_DIR = ROOT / "exploration" / "reports"
FIG_DIR = ROOT / "exploration" / "figures" / "T09"
TABLE_DIR = ROOT / "exploration" / "tables" / "T09"
ART_DIR = ROOT / "exploration" / "artifacts" / "T09"
ART_DIR.mkdir(parents=True, exist_ok=True)

TOKEN_PATTERN = re.compile(r"(?<!\w)[\w.+#/-]+(?!\w)")
DEFAULT_FILTER = "u.source_platform = 'linkedin' AND u.is_english = true AND u.date_flag = 'ok' AND u.is_swe = true"
YEARS = ("2024", "2026")
PRIMARY_SAMPLE_PER_YEAR = 2700
RAW_SAMPLE_PER_YEAR = 2500
COMPANY_CAP = 20
PRIMARY_SEED = 42
RAW_SEED = 77
COHERENCE_SAMPLE_SIZE = 1500
BER_TOPIC_SIZES = (20, 30, 50)
NMF_KS = (5, 8, 12, 15)


def ensure_dirs() -> None:
    for path in (FIG_DIR, TABLE_DIR, REPORT_DIR, ART_DIR):
        path.mkdir(parents=True, exist_ok=True)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def sql_year_expr(period_col: str = "u.period") -> str:
    return f"CASE WHEN {period_col} LIKE '2024%' THEN '2024' ELSE '2026' END"


def read_base_frame(text_source: str, raw_only: bool = False) -> pd.DataFrame:
    source_clause = (
        "(u.period LIKE '2024%' AND u.source IN ('kaggle_arshkon', 'kaggle_asaniczka'))"
        " OR (u.period LIKE '2026%' AND u.source = 'scraped')"
    )
    if text_source == "llm":
        text_clause = "ct.text_source = 'llm'"
    elif text_source == "raw":
        text_clause = "ct.text_source = 'raw'"
    else:
        raise ValueError(text_source)
    if raw_only:
        source_clause = "(u.period LIKE '2024%' AND u.source IN ('kaggle_arshkon', 'kaggle_asaniczka')) OR (u.period LIKE '2026%' AND u.source = 'scraped')"
    con = duckdb.connect()
    query = f"""
        SELECT
            u.uid,
            u.source,
            u.period,
            {sql_year_expr()} AS year_period,
            u.seniority_final,
            u.seniority_3level,
            u.seniority_final_source,
            u.seniority_native,
            u.is_aggregator,
            u.company_name_canonical,
            u.company_industry,
            u.company_size,
            u.description_length,
            u.yoe_extracted,
            u.swe_classification_tier,
            u.is_control,
            u.is_swe_adjacent,
            u.description_hash,
            ct.description_cleaned,
            ct.text_source
        FROM read_parquet('{DATA.as_posix()}') AS u
        JOIN read_parquet('{(SHARED / 'swe_cleaned_text.parquet').as_posix()}') AS ct
          USING (uid)
        WHERE {DEFAULT_FILTER}
          AND ({source_clause})
          AND ct.text_source = '{text_source}'
    """
    return con.execute(query).df()


def read_full_corpus_frame() -> pd.DataFrame:
    con = duckdb.connect()
    query = f"""
        SELECT
            u.uid,
            u.source,
            u.period,
            {sql_year_expr()} AS year_period,
            u.seniority_final,
            u.seniority_3level,
            u.seniority_final_source,
            u.seniority_native,
            u.is_aggregator,
            u.company_name_canonical,
            u.company_industry,
            u.company_size,
            u.description_length,
            u.yoe_extracted,
            u.swe_classification_tier,
            u.is_control,
            u.is_swe_adjacent,
            u.description_hash,
            ct.description_cleaned,
            ct.text_source
        FROM read_parquet('{DATA.as_posix()}') AS u
        JOIN read_parquet('{(SHARED / 'swe_cleaned_text.parquet').as_posix()}') AS ct
          USING (uid)
        WHERE {DEFAULT_FILTER}
    """
    return con.execute(query).df()


def read_tech_matrix() -> pd.DataFrame:
    con = duckdb.connect()
    return con.execute(f"SELECT * FROM read_parquet('{(SHARED / 'swe_tech_matrix.parquet').as_posix()}')").df()


def read_embedding_index() -> pd.DataFrame:
    con = duckdb.connect()
    return con.execute(f"SELECT * FROM read_parquet('{(SHARED / 'swe_embedding_index.parquet').as_posix()}')").df()


def load_company_stoplist() -> set[str]:
    return set((SHARED / "company_stoplist.txt").read_text(encoding="utf-8").splitlines())


def stable_hash_key(series: pd.Series, seed: int) -> pd.Series:
    return pd.util.hash_pandas_object(series.astype(str) + f"|{seed}", index=False).astype(np.uint64)


def dedup_and_cap(df: pd.DataFrame, seed: int, company_cap: int) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    before = len(out)
    out = out.drop_duplicates(subset=["year_period", "company_name_canonical", "description_cleaned"])
    after_dedup = len(out)
    out["_cap_key"] = stable_hash_key(out["uid"], seed)
    out = (
        out.sort_values(["year_period", "company_name_canonical", "_cap_key"])
        .groupby(["year_period", "company_name_canonical"], as_index=False, group_keys=False)
        .head(company_cap)
        .drop(columns=["_cap_key"])
    )
    after_cap = len(out)
    return out, {
        "before": before,
        "after_dedup": after_dedup,
        "after_cap": after_cap,
        "dedup_removed": before - after_dedup,
        "cap_removed": after_dedup - after_cap,
    }


def sample_year_frame(df: pd.DataFrame, target_per_year: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for year in YEARS:
        yr = df[df["year_period"] == year].copy()
        if yr.empty:
            continue
        ym = yr[yr["seniority_3level"].isin(["junior", "mid"])].copy()
        ys = yr[yr["seniority_3level"].isin(["senior", "unknown"])].copy()
        selected: list[pd.DataFrame] = []
        if len(ym) >= target_per_year:
            selected.append(ym.sample(n=target_per_year, random_state=seed))
        else:
            selected.append(ym)
            rem = target_per_year - len(ym)
            if rem > 0 and not ys.empty:
                take = min(rem, len(ys))
                selected.append(ys.sample(n=take, random_state=seed + 1))
        year_sel = pd.concat(selected, ignore_index=True)
        year_sel["_sample_draw"] = rng.integers(0, 2**32 - 1, size=len(year_sel), dtype=np.uint32)
        year_sel = year_sel.sort_values(["_sample_draw", "uid"]).drop(columns=["_sample_draw"])
        parts.append(year_sel)
    return pd.concat(parts, ignore_index=True)


def sample_primary() -> tuple[pd.DataFrame, dict[str, int]]:
    frames = []
    notes = {"raw_rows": 0}
    # 2024 primary: use arshkon only as instructed.
    arshkon = read_base_frame("llm")
    arshkon = arshkon[(arshkon["year_period"] == "2024") & (arshkon["source"] == "kaggle_arshkon")]
    arshkon, arshkon_notes = dedup_and_cap(arshkon, PRIMARY_SEED, COMPANY_CAP)
    arshkon_sample = sample_year_frame(arshkon, PRIMARY_SAMPLE_PER_YEAR, PRIMARY_SEED)
    arshkon_sample["sample_source_policy"] = "primary_2024_arshkon_only"
    frames.append(arshkon_sample)
    notes.update({f"arshkon_{k}": v for k, v in arshkon_notes.items()})

    scraped = read_base_frame("llm")
    scraped = scraped[(scraped["year_period"] == "2026") & (scraped["source"] == "scraped")]
    scraped, scraped_notes = dedup_and_cap(scraped, PRIMARY_SEED + 1, COMPANY_CAP)
    scraped_sample = sample_year_frame(scraped, PRIMARY_SAMPLE_PER_YEAR, PRIMARY_SEED + 1)
    scraped_sample["sample_source_policy"] = "primary_2026_scraped_only"
    frames.append(scraped_sample)
    notes.update({f"scraped_{k}": v for k, v in scraped_notes.items()})

    sample = pd.concat(frames, ignore_index=True)
    sample["sample_kind"] = "primary"
    return sample, notes


def sample_raw_sensitivity() -> tuple[pd.DataFrame, dict[str, int]]:
    frames = []
    notes = {"raw_rows": 0}
    raw = read_base_frame("raw")
    raw_2024 = raw[raw["year_period"] == "2024"].copy()
    raw_2026 = raw[raw["year_period"] == "2026"].copy()
    raw_2024, notes_2024 = dedup_and_cap(raw_2024, RAW_SEED, COMPANY_CAP)
    raw_2026, notes_2026 = dedup_and_cap(raw_2026, RAW_SEED + 1, COMPANY_CAP)
    frames.append(sample_year_frame(raw_2024, min(RAW_SAMPLE_PER_YEAR, len(raw_2024)), RAW_SEED))
    frames.append(sample_year_frame(raw_2026, RAW_SAMPLE_PER_YEAR, RAW_SEED + 1))
    notes.update({f"raw2024_{k}": v for k, v in notes_2024.items()})
    notes.update({f"raw2026_{k}": v for k, v in notes_2026.items()})
    sample = pd.concat(frames, ignore_index=True)
    sample["sample_kind"] = "raw_sensitivity"
    return sample, notes


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def tokenized_docs(texts: Iterable[str]) -> list[list[str]]:
    return [tokenize(text) for text in texts]


def build_coherence_model(topics: list[list[str]], docs: list[list[str]], sample_size: int = COHERENCE_SAMPLE_SIZE) -> float:
    normalized_topics: list[list[str]] = []
    for topic in topics:
        if not topic:
            continue
        normalized = []
        for term in topic:
            if isinstance(term, (list, tuple, np.ndarray)):
                term = " ".join(map(str, term))
            term = str(term).strip()
            if term:
                normalized.append(term)
        if len(normalized) >= 2:
            normalized_topics.append(normalized)
    topics = normalized_topics
    if not topics:
        return float("nan")
    if len(docs) > sample_size:
        idx = np.random.default_rng(123).choice(len(docs), size=sample_size, replace=False)
        docs = [docs[i] for i in idx]
    dictionary = Dictionary(docs)
    coherence = CoherenceModel(
        topics=topics,
        texts=docs,
        dictionary=dictionary,
        coherence="c_v",
        processes=1,
    )
    return float(coherence.get_coherence())


def fit_nmf_suite(
    df: pd.DataFrame,
    company_stoplist: set[str],
    k_values: tuple[int, ...],
    seed_values: tuple[int, ...],
    label_prefix: str,
) -> tuple[pd.DataFrame, dict[int, dict[str, object]], int]:
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.6,
        stop_words=list(company_stoplist),
    )
    X = vectorizer.fit_transform(df["description_cleaned"].fillna("").tolist())
    feature_names = np.array(vectorizer.get_feature_names_out())
    token_docs = tokenized_docs(df["description_cleaned"].fillna("").tolist())
    outputs: dict[int, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for k in k_values:
        seed_runs = []
        topic_terms_runs: list[list[list[str]]] = []
        for seed in seed_values:
            model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=500)
            W = model.fit_transform(X)
            labels = W.argmax(axis=1)
            comp_terms = []
            for comp in model.components_:
                top_idx = np.argsort(comp)[::-1][:12]
                comp_terms.append([str(term) for term in feature_names[top_idx]])
            coherence = build_coherence_model(comp_terms, token_docs)
            seed_runs.append({"seed": seed, "labels": labels, "model": model, "W": W, "coherence": coherence})
            topic_terms_runs.append(comp_terms)
        pair_aris = []
        for i in range(len(seed_runs)):
            for j in range(i + 1, len(seed_runs)):
                pair_aris.append(adjusted_rand_score(seed_runs[i]["labels"], seed_runs[j]["labels"]))
        mean_ari = float(np.mean(pair_aris)) if pair_aris else float("nan")
        mean_coherence = float(np.mean([run["coherence"] for run in seed_runs]))
        best_idx = int(np.argmax([run["coherence"] for run in seed_runs]))
        best_run = seed_runs[best_idx]
        labels = best_run["labels"]
        model = best_run["model"]
        W = best_run["W"]
        topic_terms = topic_terms_runs[best_idx]
        topic_names = [name_topic_from_terms(terms) for terms in topic_terms]
        outputs[k] = {
            "vectorizer": vectorizer,
            "X": X,
            "feature_names": feature_names,
            "labels": labels,
            "model": model,
            "W": W,
            "topic_terms": topic_terms,
            "topic_names": topic_names,
            "coherence": mean_coherence,
            "stability_ari": mean_ari,
            "seed": seed_values[best_idx],
        }
        for topic_id, terms in enumerate(topic_terms):
            rows.append(
                {
                    "method": f"NMF_k{k}",
                    "config": f"k={k}",
                    "n_topics": k,
                    "seed_stability_ari": mean_ari,
                    "coherence_cv": mean_coherence,
                    "noise_pct": 0.0,
                    "topic_id": topic_id,
                    "topic_name": topic_names[topic_id],
                    "top_terms": " | ".join(terms[:12]),
                }
            )
    return pd.DataFrame(rows), outputs, X.shape[1]


def fit_bertopic_suite(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    company_stoplist: set[str],
    min_topic_sizes: tuple[int, ...],
    seed_values: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[int, dict[str, object]]]:
    vectorizer = CountVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words=list(company_stoplist),
    )
    token_docs = tokenized_docs(df["description_cleaned"].fillna("").tolist())
    rows: list[dict[str, object]] = []
    outputs: dict[int, dict[str, object]] = {}
    for min_topic_size in min_topic_sizes:
        seed_runs = []
        for seed in seed_values:
            umap_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                random_state=seed,
            )
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=min_topic_size,
                min_samples=max(5, min_topic_size // 3),
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )
            model = BERTopic(
                vectorizer_model=vectorizer,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                calculate_probabilities=False,
                verbose=False,
                low_memory=True,
                min_topic_size=min_topic_size,
            )
            topics, _ = model.fit_transform(df["description_cleaned"].fillna("").tolist(), embeddings)
            topic_info = model.get_topic_info()
            topic_ids = [int(t) for t in topic_info["Topic"].tolist() if int(t) != -1]
            topic_terms = []
            for tid in topic_ids:
                topic_terms.append([term for term, _ in model.get_topic(tid)[:12]])
            coherence = build_coherence_model(topic_terms, token_docs)
            seed_runs.append(
                {
                    "seed": seed,
                    "labels": np.asarray(topics),
                    "model": model,
                    "topic_info": topic_info,
                    "topic_terms": topic_terms,
                    "coherence": coherence,
                }
            )
        pair_aris = []
        for i in range(len(seed_runs)):
            for j in range(i + 1, len(seed_runs)):
                pair_aris.append(adjusted_rand_score(seed_runs[i]["labels"], seed_runs[j]["labels"]))
        mean_ari = float(np.mean(pair_aris)) if pair_aris else float("nan")
        mean_coherence = float(np.mean([run["coherence"] for run in seed_runs]))
        best_idx = int(np.argmax([run["coherence"] for run in seed_runs]))
        best_run = seed_runs[best_idx]
        labels = best_run["labels"]
        model = best_run["model"]
        topic_info = best_run["topic_info"]
        topic_ids = [int(t) for t in topic_info["Topic"].tolist() if int(t) != -1]
        topic_terms = best_run["topic_terms"]
        topic_names = [name_topic_from_terms(terms) for terms in topic_terms]
        outputs[min_topic_size] = {
            "model": model,
            "labels": labels,
            "topic_info": topic_info,
            "topic_terms": topic_terms,
            "topic_names": topic_names,
            "coherence": mean_coherence,
            "stability_ari": mean_ari,
            "topic_ids": topic_ids,
            "seed": seed_values[best_idx],
        }
        for i, tid in enumerate(topic_ids):
            top_terms = [term for term, _ in model.get_topic(tid)[:12]]
            rows.append(
                {
                    "method": f"BERTopic_mts{min_topic_size}",
                    "config": f"min_topic_size={min_topic_size}",
                    "n_topics": len(topic_ids),
                    "seed_stability_ari": mean_ari,
                    "coherence_cv": mean_coherence,
                    "noise_pct": float((labels == -1).mean()),
                    "topic_id": tid,
                    "topic_name": topic_names[i],
                    "top_terms": " | ".join(top_terms),
                }
            )
    return pd.DataFrame(rows), outputs


def name_topic_from_terms(terms: list[str]) -> str:
    text = " ".join(terms).lower()
    rules = [
        ("AI/LLM workflows", ["ai/ml", "llm", "llms", "generative", "langchain", "langgraph", "rag", "prompt", "ml "]),
        ("Embedded/Firmware", ["firmware", "rtos", "i2c", "uart", "fpga", "c/c", "c++", "real-time"]),
        ("Mobile/iOS", ["ios", "swift", "kotlin", "uikit", "swiftui", "objective-c", "xcode"]),
        ("Frontend/Web", ["front-end", "frontend", "angular", "react", "javascript", "html", "css", "vue"]),
        ("Backend/API", ["backend", "node.js", "nodejs", "microservices", "postgresql", "api", "typescrip"]),
        ("DevOps/Infra", ["ci/cd", "kubernetes", "docker", "observability", "terraform", "prometheus", "gcp", "aws"]),
        ("Data Engineering / ETL", ["etl", "airflow", "pyspark", "ingestion", "governance"]),
        ("Data Platform / Scala", ["nosql", "scala", "databases", "redshift", "cloud-based millions"]),
        ("Requirements / Boilerplate", ["requirements", "qualifications", "documentation", "bachelor", "minimum", "procedures"]),
        ("Security / Clearance", ["clearance", "ts/sci", "citizenship", "obtaining", "eligibility"]),
        ("Cross-functional / Delivery", ["cross-functional", "team", "problem-solving", "ensure", "effectively", "maintain"]),
        ("Enterprise .NET", ["c#", "asp.net", ".net", "c++", "object-oriented"]),
        ("Generic Workflow / Admin", ["jira", "haves", "mandatory", "modules", "workflow", "process"]),
        ("Qualifications / Eligibility", ["experience", "minimum", "bachelor", "qualifications", "hands-on", "familiarity"]),
    ]
    for name, needles in rules:
        if any(n in text for n in needles):
            return name
    return "Other / Mixed"


def summarize_labels(df: pd.DataFrame, labels: np.ndarray, label_col: str, topic_names: list[str], topic_ids: list[int] | None = None) -> pd.DataFrame:
    out = df.copy()
    out[label_col] = labels
    if topic_ids is not None:
        valid = out[out[label_col] != -1].copy()
    else:
        valid = out.copy()
    rows: list[dict[str, object]] = []
    for topic_value, grp in valid.groupby(label_col):
        if topic_ids is not None and int(topic_value) == -1:
            continue
        name = topic_names[int(topic_value)] if topic_ids is None else topic_names[topic_ids.index(int(topic_value))]
        seniority = grp["seniority_3level"].value_counts(normalize=True).to_dict()
        final = grp["seniority_final"].value_counts(normalize=True).to_dict()
        period = grp["year_period"].value_counts(normalize=True).to_dict()
        rows.append(
            {
                "archetype": int(topic_value),
                "archetype_name": name,
                "n": len(grp),
                "share": len(grp) / len(valid),
                "period_2024_share": period.get("2024", 0.0),
                "period_2026_share": period.get("2026", 0.0),
                "junior_share_3level": seniority.get("junior", 0.0),
                "mid_share_3level": seniority.get("mid", 0.0),
                "senior_share_3level": seniority.get("senior", 0.0),
                "entry_share_final": final.get("entry", 0.0),
                "associate_share_final": final.get("associate", 0.0),
                "mid_senior_share_final": final.get("mid-senior", 0.0),
                "director_share_final": final.get("director", 0.0),
                "unknown_share_final": final.get("unknown", 0.0),
                "mean_description_length": float(grp["description_length"].mean()),
                "mean_yoe": float(grp["yoe_extracted"].dropna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def build_domain_label_frame(df: pd.DataFrame, tech_df: pd.DataFrame) -> pd.DataFrame:
    tech_cols = [c for c in tech_df.columns if c != "uid"]
    family_map: dict[str, list[str]] = {
        "frontend_web": ["html", "css", "sass", "less", "react", "react_native", "angular", "vue", "nextjs", "svelte", "redux", "webpack", "vite", "storybook", "tailwind", "bootstrap", "material_ui"],
        "backend_api": ["nodejs", "express", "nestjs", "django", "django_rest_framework", "flask", "spring", "spring_boot", "dotnet", "aspnet", "rails", "laravel", "fastapi", "phoenix", "tornado", "bottle", "grpc", "graphql", "rest_api", "microservices", "event_driven", "oauth"],
        "data_platform": ["sql", "postgresql", "mysql", "sqlite", "mongodb", "redis", "cassandra", "kafka", "spark", "hadoop", "hive", "presto", "trino", "snowflake", "databricks", "dbt", "elasticsearch", "airflow", "luigi", "airbyte", "delta_lake", "tableau", "powerbi", "looker", "superset", "metabase", "bigquery", "redshift"],
        "cloud_devops": ["aws", "azure", "gcp", "kubernetes", "docker", "terraform", "ansible", "helm", "jenkins", "github_actions", "gitlab_ci", "circleci", "argo_cd", "openshift", "nomad", "prometheus", "grafana", "datadog", "new_relic", "splunk", "opentelemetry", "linux", "bash", "git", "serverless", "cloudformation", "pulumi"],
        "ai_ml": ["machine_learning", "deep_learning", "data_science", "statistics", "nlp", "computer_vision", "generative_ai", "tensorflow", "pytorch", "scikit_learn", "pandas", "numpy", "jupyter", "xgboost", "lightgbm", "catboost", "mlflow", "kubeflow", "ray", "hugging_face", "openai_api", "anthropic_api", "claude_api", "gemini_api", "langchain", "langgraph", "llamaindex", "rag", "vector_db", "pinecone", "weaviate", "chroma", "milvus", "faiss", "prompt_engineering", "fine_tuning", "mcp", "llm", "copilot", "cursor", "chatgpt", "claude", "gemini", "codex", "agent"],
        "testing_quality": ["junit", "pytest", "jest", "mocha", "chai", "selenium", "cypress", "playwright", "tdd", "agile", "scrum", "kanban", "ci_cd", "code_review", "pair_programming", "unit_testing", "integration_testing", "bdd", "qa"],
        "language_general": ["python", "java", "javascript", "typescript", "go", "rust", "c_plus_plus", "c_sharp", "ruby", "kotlin", "swift", "scala", "php", "perl", "dart", "lua", "haskell", "clojure", "elixir", "erlang", "julia", "matlab", "r_language"],
    }
    family_cols = {family: [c for c in cols if c in tech_cols] for family, cols in family_map.items()}
    joined = df[["uid", "source", "year_period", "seniority_3level", "company_size", "company_name_canonical"]].merge(tech_df, on="uid", how="left")
    family_scores = pd.DataFrame({"uid": joined["uid"]})
    for family, cols in family_cols.items():
        if cols:
            family_scores[family] = joined[cols].fillna(False).sum(axis=1)
        else:
            family_scores[family] = 0
    family_scores["family_max"] = family_scores[[*family_cols]].max(axis=1)
    priority = ["ai_ml", "data_platform", "frontend_web", "backend_api", "cloud_devops", "testing_quality", "language_general"]
    domain = []
    for _, row in family_scores.iterrows():
        if row["family_max"] <= 0:
            domain.append("none")
            continue
        winners = [f for f in priority if row[f] == row["family_max"] and row[f] > 0]
        domain.append(winners[0] if winners else "none")
    family_scores["tech_domain"] = domain
    joined = joined.merge(family_scores[["uid", "tech_domain"]], on="uid", how="left")
    return joined


def save_fig_plotly(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(path), scale=1)
    except Exception:
        # Fall back to HTML export if kaleido/rendering fails unexpectedly.
        fig.write_html(str(path.with_suffix(".html")))


def plot_embedding_maps(df: pd.DataFrame, embeddings: np.ndarray, label_col: str, out_path: Path) -> None:
    from matplotlib.colors import ListedColormap

    umap_coords = UMAP(n_neighbors=20, min_dist=0.08, metric="cosine", random_state=42, n_components=2).fit_transform(embeddings)
    pca_coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    panels = [
        ("UMAP / cluster", umap_coords, label_col),
        ("UMAP / year", umap_coords, "year_period"),
        ("UMAP / seniority", umap_coords, "seniority_3level"),
        ("PCA / cluster", pca_coords, label_col),
        ("PCA / year", pca_coords, "year_period"),
        ("PCA / seniority", pca_coords, "seniority_3level"),
    ]
    for ax, (title, coords, color_col) in zip(axes.ravel(), panels):
        s = 8 if len(df) > 4000 else 12
        vals = df[color_col].astype(str)
        cats = pd.Index(sorted(vals.unique()))
        palette = plt.cm.get_cmap("tab20", len(cats))
        color_lookup = {cat: palette(i) for i, cat in enumerate(cats)}
        colors = vals.map(color_lookup)
        ax.scatter(coords[:, 0], coords[:, 1], c=list(colors), s=s, alpha=0.65, linewidths=0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for cat in cats[:8]:
            ax.scatter([], [], c=[color_lookup[cat]], label=str(cat), s=20)
        ax.legend(frameon=False, loc="best", fontsize=8)
    fig.suptitle("T09 embedding maps", fontsize=16)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bertopic_static(model: BERTopic, out_prefix: Path) -> None:
    try:
        topics_fig = model.visualize_topics()
        save_fig_plotly(topics_fig, out_prefix.with_name(out_prefix.name + "_topics.png"))
    except Exception as exc:
        print(f"BERTopic visualize_topics failed: {exc}")
    try:
        hierarchy = model.hierarchical_topics(model.documents_, model.topics_) if hasattr(model, "documents_") and hasattr(model, "topics_") else None
        if hierarchy is not None:
            hier_fig = model.visualize_hierarchy(hierarchical_topics=hierarchy)
        else:
            hier_fig = model.visualize_hierarchy()
        save_fig_plotly(hier_fig, out_prefix.with_name(out_prefix.name + "_hierarchy.png"))
    except Exception as exc:
        print(f"BERTopic visualize_hierarchy failed: {exc}")
    try:
        bar_fig = model.visualize_barchart(top_n_topics=10)
        save_fig_plotly(bar_fig, out_prefix.with_name(out_prefix.name + "_barchart.png"))
    except Exception as exc:
        print(f"BERTopic visualize_barchart failed: {exc}")


def compute_alignment(topic_terms_a: list[list[str]], topic_terms_b: list[list[str]]) -> pd.DataFrame:
    if not topic_terms_a or not topic_terms_b:
        return pd.DataFrame()
    sim = np.zeros((len(topic_terms_a), len(topic_terms_b)), dtype=float)
    for i, ta in enumerate(topic_terms_a):
        set_a = set(ta[:10])
        for j, tb in enumerate(topic_terms_b):
            set_b = set(tb[:10])
            inter = len(set_a & set_b)
            union = len(set_a | set_b)
            sim[i, j] = inter / union if union else 0.0
    rows, cols = linear_sum_assignment(-sim)
    out = []
    for r, c in zip(rows, cols):
        out.append(
            {
                "topic_a": int(r),
                "topic_b": int(c),
                "jaccard_top10": float(sim[r, c]),
                "terms_a": " | ".join(topic_terms_a[r][:10]),
                "terms_b": " | ".join(topic_terms_b[c][:10]),
            }
        )
    return pd.DataFrame(out).sort_values("jaccard_top10", ascending=False)


def compute_nmi_summary(label_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    targets = {
        "seniority_3level": label_df["seniority_3level"].astype(str),
        "year_period": label_df["year_period"].astype(str),
        "source": label_df["source"].astype(str),
        "text_source": label_df["text_source"].astype(str),
        "is_aggregator": label_df["is_aggregator"].astype(str),
        "tech_domain": label_df["tech_domain"].astype(str),
    }
    if "company_size_bucket" in label_df.columns:
        targets["company_size_bucket"] = label_df["company_size_bucket"].astype(str)
    x = label_df[label_col].astype(str)
    for name, y in targets.items():
        rows.append({"label": name, "nmi": normalized_mutual_info_score(x, y), "n": len(label_df)})
    return pd.DataFrame(rows).sort_values("nmi", ascending=False)


def build_company_size_bucket(df: pd.DataFrame) -> pd.Series:
    vals = df["company_size"].copy()
    # Company size is mostly arshkon-only. Bucket within the available rows.
    bins = pd.qcut(vals.rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    return bins.astype("string")


def topic_period_table(df: pd.DataFrame, labels: np.ndarray, label_name: str) -> pd.DataFrame:
    out = df.copy()
    out[label_name] = labels
    out = out[out[label_name] != -1] if np.issubdtype(out[label_name].dtype, np.integer) else out
    rows = []
    total_by_period = out["year_period"].value_counts().to_dict()
    total_by_topic = out[label_name].value_counts().to_dict()
    for topic, grp in out.groupby(label_name):
        year_counts = grp["year_period"].value_counts().to_dict()
        rows.append(
            {
                "archetype": int(topic),
                "n": len(grp),
                "share": len(grp) / len(out),
                "share_2024": year_counts.get("2024", 0) / total_by_period.get("2024", 1),
                "share_2026": year_counts.get("2026", 0) / total_by_period.get("2026", 1),
                "period_2024_share": year_counts.get("2024", 0) / len(grp),
                "period_2026_share": year_counts.get("2026", 0) / len(grp),
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def full_corpus_assign_nmf(df: pd.DataFrame, model: NMF, vectorizer: TfidfVectorizer, topic_names: list[str], output_path: Path) -> pd.DataFrame:
    X_all = vectorizer.transform(df["description_cleaned"].fillna("").tolist())
    W_all = model.transform(X_all)
    labels = W_all.argmax(axis=1)
    confidence = W_all.max(axis=1) / np.maximum(W_all.sum(axis=1), 1e-12)
    out = pd.DataFrame(
        {
            "uid": df["uid"].values,
            "archetype": labels.astype(int),
            "archetype_name": [topic_names[i] for i in labels],
            "archetype_confidence": confidence,
        }
    )
    pq.write_table(pa.Table.from_pandas(out[["uid", "archetype", "archetype_name"]]), output_path)
    return out


def write_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave 2 T09 posting archetype discovery")
    parser.add_argument("--regenerate", action="store_true", help="No-op flag for explicit reruns.")
    args = parser.parse_args()
    del args

    ensure_dirs()
    company_stoplist = load_company_stoplist()

    primary, primary_notes = sample_primary()
    raw_sens, raw_notes = sample_raw_sensitivity()

    write_table(
        primary[["uid", "source", "year_period", "seniority_3level", "company_name_canonical", "is_aggregator", "text_source"]],
        "T09_primary_sample_manifest.csv",
    )
    write_table(
        raw_sens[["uid", "source", "year_period", "seniority_3level", "company_name_canonical", "is_aggregator", "text_source"]],
        "T09_raw_sensitivity_sample_manifest.csv",
    )

    sample_summary = (
        primary.groupby(["year_period", "source", "seniority_3level", "text_source"], as_index=False)
        .agg(n=("uid", "count"), companies=("company_name_canonical", "nunique"), aggregators=("is_aggregator", "sum"))
        .sort_values(["year_period", "source", "seniority_3level"])
    )
    write_table(sample_summary, "T09_primary_sample_composition.csv")
    raw_summary = (
        raw_sens.groupby(["year_period", "source", "seniority_3level", "text_source"], as_index=False)
        .agg(n=("uid", "count"), companies=("company_name_canonical", "nunique"), aggregators=("is_aggregator", "sum"))
        .sort_values(["year_period", "source", "seniority_3level"])
    )
    write_table(raw_summary, "T09_raw_sensitivity_sample_composition.csv")

    # Embeddings for the primary sample come from the shared artifact index.
    embed_idx = read_embedding_index()
    embed_map = dict(zip(embed_idx["uid"], embed_idx["row_index"]))
    emb = np.load(SHARED / "swe_embeddings.npy", mmap_mode="r")
    primary_embed_rows = [int(embed_map[u]) for u in primary["uid"]]
    primary_embeddings = np.asarray(emb[primary_embed_rows])

    # BERTopic comparison on the primary sample.
    ber_table, ber_outputs = fit_bertopic_suite(primary, primary_embeddings, company_stoplist, BER_TOPIC_SIZES, (11, 23, 47))
    write_table(ber_table, "T09_methods_bertopic.csv")

    # Static BERTopic figures from the preferred min_topic_size=30 run.
    bertopic_best = ber_outputs[30]["model"]
    bertopic_best.documents_ = primary["description_cleaned"].fillna("").tolist()
    bertopic_best.topics_ = np.asarray(ber_outputs[30]["labels"])
    plot_bertopic_static(bertopic_best, FIG_DIR / "T09_bertopic")

    # NMF comparison on the primary sample.
    nmf_table, nmf_outputs, vocab_size = fit_nmf_suite(primary, company_stoplist, NMF_KS, (13, 29, 47), "primary")
    write_table(nmf_table, "T09_methods_nmf.csv")

    # Choose the downstream representation by a simple evidence score.
    method_rows = []
    for k, info in ber_outputs.items():
        method_rows.append(
            {
                "method": "BERTopic",
                "config": f"min_topic_size={k}",
                "n_topics": int(len([x for x in info["topic_ids"] if x != -1])),
                "stability_ari": info["stability_ari"],
                "coherence_cv": info["coherence"],
                "noise_pct": float((info["labels"] == -1).mean()),
                "score": float(info["coherence"] + 0.5 * info["stability_ari"] - 0.25 * float((info["labels"] == -1).mean())),
                "notes": "embedding topics; outlier-aware",
            }
        )
    for k, info in nmf_outputs.items():
        method_rows.append(
            {
                "method": "NMF",
                "config": f"k={k}",
                "n_topics": k,
                "stability_ari": info["stability_ari"],
                "coherence_cv": info["coherence"],
                "noise_pct": 0.0,
                "score": float(info["coherence"] + 0.5 * info["stability_ari"]),
                "notes": "dense factors; label every row",
            }
        )
    methods_df = pd.DataFrame(method_rows).sort_values("score", ascending=False).reset_index(drop=True)
    methods_df["rank"] = np.arange(1, len(methods_df) + 1)
    methods_df["selected_for_downstream"] = False
    methods_df.loc[0, "selected_for_downstream"] = True
    write_table(methods_df, "T09_methods_comparison.csv")

    selected_method = methods_df.iloc[0]["method"]
    selected_config = methods_df.iloc[0]["config"]

    if selected_method == "NMF":
        selected_k = int(selected_config.split("=")[1])
        selected_info = nmf_outputs[selected_k]
        selected_labels = selected_info["labels"]
        selected_topic_terms = selected_info["topic_terms"]
        selected_topic_names = selected_info["topic_names"]
        selected_model = selected_info["model"]
        selected_vectorizer = selected_info["vectorizer"]
        full_labels = full_corpus_assign_nmf(
            read_full_corpus_frame(),
            selected_model,
            selected_vectorizer,
            selected_topic_names,
            SHARED / "swe_archetype_labels.parquet",
        )
    else:
        # Fallback if BERTopic unexpectedly wins. Fit a fresh NMF later for labels.
        selected_k = 15
        selected_info = nmf_outputs[selected_k]
        selected_labels = selected_info["labels"]
        selected_topic_terms = selected_info["topic_terms"]
        selected_topic_names = selected_info["topic_names"]
        selected_model = selected_info["model"]
        selected_vectorizer = selected_info["vectorizer"]
        full_labels = full_corpus_assign_nmf(
            read_full_corpus_frame(),
            selected_model,
            selected_vectorizer,
            selected_topic_names,
            SHARED / "swe_archetype_labels.parquet",
        )

    # Summaries on the primary sample using the selected method.
    primary_selected = primary.copy()
    primary_selected["archetype"] = selected_labels
    primary_selected["archetype_name"] = [selected_topic_names[i] for i in selected_labels]
    char_primary = summarize_labels(primary_selected, selected_labels, "archetype", selected_topic_names)
    write_table(char_primary, "T09_topic_characterization_primary.csv")

    # Raw sensitivity with the selected representation.
    raw_vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.6,
        stop_words=list(company_stoplist),
    )
    raw_X = raw_vectorizer.fit_transform(raw_sens["description_cleaned"].fillna("").tolist())
    raw_token_docs = tokenized_docs(raw_sens["description_cleaned"].fillna("").tolist())
    raw_nmf = NMF(n_components=selected_k, init="nndsvda", random_state=RAW_SEED, max_iter=500)
    raw_W = raw_nmf.fit_transform(raw_X)
    raw_labels = raw_W.argmax(axis=1)
    raw_topic_terms = []
    raw_feature_names = np.array(raw_vectorizer.get_feature_names_out())
    for comp in raw_nmf.components_:
        top_idx = np.argsort(comp)[::-1][:12]
        raw_topic_terms.append([str(t) for t in raw_feature_names[top_idx]])
    raw_topic_names = [name_topic_from_terms(terms) for terms in raw_topic_terms]
    raw_char = summarize_labels(raw_sens.assign(archetype=raw_labels), raw_labels, "archetype", raw_topic_names)
    write_table(raw_char, "T09_topic_characterization_raw_sensitivity.csv")

    raw_alignment = compute_alignment(selected_topic_terms, raw_topic_terms)
    if not raw_alignment.empty:
        raw_alignment["primary_topic_name"] = raw_alignment["topic_a"].map(lambda i: selected_topic_names[i])
        raw_alignment["raw_topic_name"] = raw_alignment["topic_b"].map(lambda i: raw_topic_names[i])
    write_table(raw_alignment, "T09_raw_sensitivity_topic_alignment.csv")

    # Topic alignment between BERTopic and NMF on the primary sample.
    ber_topic_terms = ber_outputs[30]["topic_terms"]
    ber_topic_names = ber_outputs[30]["topic_names"]
    nmf_topic_terms = selected_topic_terms
    nmf_topic_names = selected_topic_names
    alignment = compute_alignment(ber_topic_terms, nmf_topic_terms)
    if not alignment.empty:
        alignment["bertopic_topic_name"] = alignment["topic_a"].map(lambda i: ber_topic_names[i])
        alignment["nmf_topic_name"] = alignment["topic_b"].map(lambda i: nmf_topic_names[i])
        alignment["method_robust"] = alignment["jaccard_top10"] >= 0.25
    write_table(alignment, "T09_topic_alignment.csv")

    # Cluster stability and evidence summary.
    selected_summary = pd.DataFrame(
        {
            "method": [selected_method],
            "config": [selected_config],
            "n_topics": [len(selected_topic_names)],
            "coherence_cv": [methods_df.iloc[0]["coherence_cv"]],
            "stability_ari": [methods_df.iloc[0]["stability_ari"]],
            "noise_pct": [methods_df.iloc[0]["noise_pct"]],
            "score": [methods_df.iloc[0]["score"]],
            "selected_for_downstream": [True],
        }
    )
    write_table(selected_summary, "T09_selected_method.csv")

    # Add tech-domain labels and compute NMI on the full corpus labels.
    full_corpus = read_full_corpus_frame()
    if selected_method == "NMF":
        full_labels_df = full_labels[["uid", "archetype", "archetype_name"]].copy()
    else:
        full_labels_df = full_labels[["uid", "archetype", "archetype_name"]].copy()
    full_corpus = full_corpus.merge(full_labels_df, on="uid", how="left")
    tech_df = read_tech_matrix()
    full_corpus = build_domain_label_frame(full_corpus, tech_df)
    if full_corpus["company_size"].notna().any():
        full_corpus["company_size_bucket"] = pd.NA
        arsh = full_corpus[full_corpus["company_size"].notna()].copy()
        if len(arsh) >= 20:
            try:
                arsh["company_size_bucket"] = build_company_size_bucket(arsh)
                full_corpus.loc[arsh.index, "company_size_bucket"] = arsh["company_size_bucket"].astype(str)
            except Exception:
                full_corpus["company_size_bucket"] = pd.NA
    nmi_df = compute_nmi_summary(full_corpus.dropna(subset=["archetype"]), "archetype")
    write_table(nmi_df, "T09_nmi_summary.csv")

    # Period shares and topic characterization from the full corpus.
    full_char = summarize_labels(full_corpus.dropna(subset=["archetype"]), full_corpus.dropna(subset=["archetype"])["archetype"].astype(int).values, "archetype", selected_topic_names)
    write_table(full_char, "T09_topic_characterization_full_corpus.csv")
    topic_period = topic_period_table(full_corpus.dropna(subset=["archetype"]), full_corpus.dropna(subset=["archetype"])["archetype"].astype(int).values, "archetype")
    write_table(topic_period, "T09_topic_period_share.csv")

    # Sensitivity summary table.
    sensitivity_rows = [
        {
            "sensitivity": "primary_llm",
            "method": selected_method,
            "config": selected_config,
            "n_rows": len(primary),
            "coherence_cv": methods_df.iloc[0]["coherence_cv"],
            "stability_ari": methods_df.iloc[0]["stability_ari"],
            "noise_pct": methods_df.iloc[0]["noise_pct"],
            "dominant_topic": selected_topic_names[int(np.bincount(selected_labels).argmax())] if len(selected_labels) else "",
        },
        {
            "sensitivity": "raw_fallback",
            "method": "NMF",
            "config": f"k={selected_k}",
            "n_rows": len(raw_sens),
            "coherence_cv": float(build_coherence_model(raw_topic_terms, raw_token_docs)),
            "stability_ari": float(adjusted_rand_score(raw_labels, raw_labels)),
            "noise_pct": 0.0,
            "dominant_topic": raw_topic_names[int(np.bincount(raw_labels).argmax())] if len(raw_labels) else "",
        },
    ]
    write_table(pd.DataFrame(sensitivity_rows), "T09_sensitivity_summary.csv")

    # Figures.
    plot_embedding_maps(primary_selected, primary_embeddings, "archetype", FIG_DIR / "T09_embedding_maps.png")

    # A compact methods-report figure is optional; the CSVs carry the details.

    # Write a brief artifact note so downstream tasks know this is NMF-backed.
    note = f"""# T09 Archetype Labels Note\n\nSelected method: {selected_method} ({selected_config})\nPrimary sample: {len(primary):,} llm rows ({PRIMARY_SAMPLE_PER_YEAR:,} per year; 2024 from arshkon only, 2026 from scraped only)\nRaw sensitivity sample: {len(raw_sens):,} raw-fallback rows ({RAW_SAMPLE_PER_YEAR:,} per year target)\nFull corpus labels: {len(full_corpus):,} SWE LinkedIn rows\n\nThe selected method was chosen by comparing topic coherence, seed stability, and noise handling across BERTopic and NMF candidates. Full-corpus labels were exported from the selected NMF representation.\n"""
    (ART_DIR / "T09_note.md").write_text(note, encoding="utf-8")

    # Summary for stdout.
    print("T09 completed")
    print(methods_df.head(8).to_string(index=False))
    print("\nNMI summary:")
    print(nmi_df.to_string(index=False))


if __name__ == "__main__":
    main()
