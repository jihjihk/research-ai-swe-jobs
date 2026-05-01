#!/usr/bin/env python3
"""T09 posting archetype discovery.

Memory posture:
- DuckDB reads parquet with 4GB memory cap and 1 thread.
- Full parquet files are not loaded into pandas.
- Modeling uses one bounded sample; all-row label assignment is chunked.
"""

from __future__ import annotations

import gc
import itertools
import json
import os
import re
import resource
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "exploration" / "artifacts" / "shared"
TABLE_DIR = ROOT / "exploration" / "tables" / "T09"
FIG_DIR = ROOT / "exploration" / "figures" / "T09"
LABEL_PATH = SHARED / "swe_archetype_labels.parquet"
SUMMARY_PATH = TABLE_DIR / "run_summary.json"

CLEANED_PATH = SHARED / "swe_cleaned_text.parquet"
EMBED_INDEX_PATH = SHARED / "swe_embedding_index.parquet"
EMBED_PATH = SHARED / "swe_embeddings.npy"
TECH_PATH = SHARED / "swe_tech_matrix.parquet"
UNIFIED_PATH = ROOT / "data" / "unified.parquet"

RANDOM_SEED = 20260416
COMPANY_CAP = 50
MIN_CHARS = 100
SAMPLE_TARGETS = {
    "arshkon_2024": 2400,
    "asaniczka_2024": 1600,
    "scraped_2026_03": 2000,
    "scraped_2026_04": 2000,
}
NMF_KS = [5, 8, 12, 15]
NMF_SEEDS = [11, 23, 37]
BEST_NMF_K = 8
PRIMARY_BERTOPIC_MIN_TOPIC_SIZE = 30


TECH_DOMAIN_COLUMNS = {
    "ai_ml": [
        "tensorflow",
        "pytorch",
        "scikit_learn",
        "pandas",
        "numpy",
        "scipy",
        "langchain",
        "llamaindex",
        "rag",
        "vector_databases",
        "pinecone",
        "weaviate",
        "chroma",
        "hugging_face",
        "openai_api",
        "claude_api",
        "anthropic_api",
        "prompt_engineering",
        "fine_tuning",
        "mcp",
        "llm",
        "generative_ai",
        "machine_learning",
        "deep_learning",
        "nlp",
        "computer_vision",
        "mlops",
        "copilot",
        "cursor",
        "chatgpt",
        "claude",
        "gemini",
        "codex",
        "agents",
        "evals",
    ],
    "frontend_web": [
        "javascript",
        "typescript",
        "html",
        "css",
        "react",
        "angular",
        "vue",
        "nextjs",
        "jest",
        "cypress",
        "playwright",
        "mocha",
    ],
    "backend_api": [
        "java",
        "spring",
        "dotnet",
        "aspnet",
        "nodejs",
        "express",
        "django",
        "flask",
        "fastapi",
        "rails",
        "graphql",
        "grpc",
        "rest_api",
        "microservices",
        "serverless",
        "go",
        "ruby",
        "php",
        "csharp",
        "scala",
    ],
    "cloud_devops": [
        "aws",
        "azure",
        "gcp",
        "kubernetes",
        "docker",
        "terraform",
        "helm",
        "ansible",
        "ci_cd",
        "jenkins",
        "github_actions",
        "gitlab_ci",
        "circleci",
        "buildkite",
        "argo_cd",
        "devops",
        "sre",
        "observability",
        "prometheus",
        "grafana",
    ],
    "data_platform": [
        "sql",
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
        "opensearch",
        "oracle",
        "sql_server",
        "cassandra",
        "dynamodb",
        "bigquery",
        "redshift",
        "airflow",
        "flink",
        "hadoop",
        "rabbitmq",
        "tableau",
        "power_bi",
    ],
    "systems_embedded": [
        "c_language",
        "cpp",
        "rust",
        "linux",
        "unix",
        "bash_shell",
        "powershell",
        "objective_c",
        "swift",
        "kotlin",
        "dart",
    ],
    "security_quality": [
        "security",
        "oauth",
        "pytest",
        "junit",
        "selenium",
        "unit_testing",
        "integration_testing",
        "tdd",
        "bdd",
    ],
}

ARCHETYPE_NAMES_K8 = {
    0: "Product Backend Engineering",
    1: "Enterprise Application Support",
    2: "Java Dotnet Web Services",
    3: "Cloud DevOps Infrastructure",
    4: "Defense Clearance Systems",
    5: "AI LLM Platforms",
    6: "Embedded Firmware Systems",
    7: "Large Employer Data Platforms",
}


@dataclass
class ModelResult:
    method: str
    variant: str
    n_topics: int
    noise_share: float
    rough_coherence: float
    stability_ari: float | None
    notes: str


def assert_regex_examples() -> None:
    """Keep the small custom normalization helpers honest."""
    escaped = r"C\+\+ C\# \.NET CI\/CD"
    assert normalize_text_for_terms(escaped) == "C++ C# .NET CI/CD"
    assert year_group("2024-04") == "2024"
    assert year_group("2026-03") == "2026"
    assert quota_key("kaggle_arshkon", "2024-04") == "arshkon_2024"
    assert quota_key("scraped", "2026-04") == "scraped_2026_04"


def normalize_text_for_terms(text: str) -> str:
    return re.sub(r"\\([+\-#.&_()/\[\]{}!*])", r"\1", text)


def year_group(period: str) -> str:
    return str(period).split("-")[0]


def quota_key(source: str, period: str) -> str:
    if source == "kaggle_arshkon" and period == "2024-04":
        return "arshkon_2024"
    if source == "kaggle_asaniczka" and period == "2024-01":
        return "asaniczka_2024"
    if source == "scraped" and period == "2026-03":
        return "scraped_2026_03"
    if source == "scraped" and period == "2026-04":
        return "scraped_2026_04"
    return f"{source}_{period}"


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=1")
    return con


def write_df(df: pd.DataFrame, filename: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / filename, index=False)


def allocate_targets(counts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for qkey, target in SAMPLE_TARGETS.items():
        sub = counts[counts["quota_key"] == qkey].copy()
        available = int(sub["n"].sum())
        final_target = min(target, available)
        if final_target == 0:
            continue
        sub["raw_alloc"] = sub["n"] / available * final_target
        sub["alloc"] = np.floor(sub["raw_alloc"]).astype(int)
        remainder = final_target - int(sub["alloc"].sum())
        sub["frac"] = sub["raw_alloc"] - sub["alloc"]
        if remainder > 0:
            order = sub.sort_values(["frac", "n"], ascending=False).index.tolist()
            for idx in order[:remainder]:
                sub.loc[idx, "alloc"] += 1
        for rec in sub.to_dict("records"):
            rows.append(
                {
                    "quota_key": qkey,
                    "seniority_3level": rec["seniority_3level"],
                    "target_n": int(rec["alloc"]),
                    "available_n": int(rec["n"]),
                }
            )
    return pd.DataFrame(rows)


def sample_sql_base() -> str:
    return f"""
WITH joined AS (
  SELECT
    c.uid,
    c.description_cleaned,
    c.text_source,
    c.source,
    c.period,
    c.seniority_final,
    c.seniority_3level,
    c.is_aggregator,
    c.company_name_canonical,
    c.metro_area,
    c.yoe_extracted,
    c.swe_classification_tier,
    c.seniority_final_source,
    e.embedding_row,
    length(coalesce(c.description_cleaned, '')) AS char_len,
    array_length(regexp_split_to_array(trim(coalesce(c.description_cleaned, '')), '\\s+')) AS word_count,
    CASE
      WHEN c.source = 'kaggle_arshkon' AND c.period = '2024-04' THEN 'arshkon_2024'
      WHEN c.source = 'kaggle_asaniczka' AND c.period = '2024-01' THEN 'asaniczka_2024'
      WHEN c.source = 'scraped' AND c.period = '2026-03' THEN 'scraped_2026_03'
      WHEN c.source = 'scraped' AND c.period = '2026-04' THEN 'scraped_2026_04'
      ELSE c.source || '_' || c.period
    END AS quota_key,
    row_number() OVER (
      PARTITION BY coalesce(c.company_name_canonical, '__missing__')
      ORDER BY hash(c.uid, {RANDOM_SEED})
    ) AS company_rank
  FROM read_parquet('{CLEANED_PATH.as_posix()}') c
  JOIN read_parquet('{EMBED_INDEX_PATH.as_posix()}') e USING(uid)
  WHERE c.text_source = 'llm'
    AND length(coalesce(c.description_cleaned, '')) >= {MIN_CHARS}
)
SELECT * FROM joined
WHERE company_rank <= {COMPANY_CAP}
"""


def build_sample(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    base = sample_sql_base()
    counts = con.execute(
        f"""
        SELECT quota_key, seniority_3level, count(*) AS n
        FROM ({base})
        GROUP BY quota_key, seniority_3level
        ORDER BY quota_key, seniority_3level
        """
    ).fetchdf()
    alloc = allocate_targets(counts)
    write_df(counts, "candidate_counts_after_company_cap.csv")
    write_df(alloc, "sample_allocation_targets.csv")
    con.register("alloc", alloc)
    sample = con.execute(
        f"""
        WITH capped AS ({base}),
        ranked AS (
          SELECT capped.*,
                 row_number() OVER (
                   PARTITION BY capped.quota_key, capped.seniority_3level
                   ORDER BY hash(capped.uid, {RANDOM_SEED + 7})
                 ) AS stratum_rank
          FROM capped
          JOIN alloc USING(quota_key, seniority_3level)
        )
        SELECT
          uid, description_cleaned, text_source, source, period, seniority_final,
          seniority_3level, is_aggregator, company_name_canonical, metro_area,
          yoe_extracted, swe_classification_tier, seniority_final_source,
          embedding_row, char_len, word_count, quota_key
        FROM ranked
        JOIN alloc USING(quota_key, seniority_3level)
        WHERE stratum_rank <= target_n
        ORDER BY quota_key, seniority_3level, stratum_rank
        """
    ).fetchdf()
    sample["year"] = sample["period"].map(year_group)
    write_df(sample.drop(columns=["description_cleaned"]), "modeling_sample_index.csv")
    return sample


def load_sample_embeddings(sample: pd.DataFrame) -> np.ndarray:
    embeddings = np.load(EMBED_PATH, mmap_mode="r")
    rows = sample["embedding_row"].to_numpy(dtype=np.int64)
    arr = np.asarray(embeddings[rows], dtype=np.float32)
    return arr


def build_tfidf(texts: pd.Series) -> tuple[TfidfVectorizer, csr_matrix]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        preprocessor=normalize_text_for_terms,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=10,
        max_df=0.65,
        max_features=20000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_+\-#./]{1,}\b",
    )
    X = vectorizer.fit_transform(texts.fillna(""))
    return vectorizer, X


def topic_npmi(
    X: csr_matrix,
    feature_names: np.ndarray,
    topic_terms: list[list[str]],
    top_n: int = 10,
) -> float:
    if X.shape[0] == 0:
        return np.nan
    binary = (X > 0).astype(np.int8).tocsc()
    doc_count = X.shape[0]
    term_index = {term: i for i, term in enumerate(feature_names)}
    topic_scores: list[float] = []
    for terms in topic_terms:
        idx = [term_index[t] for t in terms[:top_n] if t in term_index]
        if len(idx) < 2:
            continue
        scores: list[float] = []
        for i, j in itertools.combinations(idx, 2):
            xi = binary[:, i]
            xj = binary[:, j]
            pi = xi.sum() / doc_count
            pj = xj.sum() / doc_count
            pij = xi.multiply(xj).sum() / doc_count
            if pij <= 0 or pi <= 0 or pj <= 0:
                continue
            scores.append(float(np.log(pij / (pi * pj)) / (-np.log(pij))))
        if scores:
            topic_scores.append(float(np.mean(scores)))
    if not topic_scores:
        return np.nan
    return float(np.mean(topic_scores))


def top_terms_from_components(
    components: np.ndarray, feature_names: np.ndarray, top_n: int = 20
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for component_id, weights in enumerate(components):
        top = np.argsort(weights)[::-1][:top_n]
        for rank, idx in enumerate(top, start=1):
            rows.append(
                {
                    "component": component_id,
                    "rank": rank,
                    "term": feature_names[idx],
                    "weight": float(weights[idx]),
                }
            )
    return pd.DataFrame(rows)


def fit_nmf_models(
    X: csr_matrix, feature_names: np.ndarray
) -> tuple[pd.DataFrame, dict[int, NMF], dict[int, np.ndarray]]:
    rows: list[dict[str, object]] = []
    fitted: dict[int, NMF] = {}
    assignments: dict[int, np.ndarray] = {}
    for k in NMF_KS:
        seed_assignments: list[np.ndarray] = []
        seed_coherence: list[float] = []
        seed_reconstruction: list[float] = []
        for seed in NMF_SEEDS:
            model = NMF(
                n_components=k,
                init="nndsvdar",
                random_state=seed,
                max_iter=350,
                solver="cd",
                beta_loss="frobenius",
            )
            W = model.fit_transform(X)
            labels = W.argmax(axis=1).astype(int)
            seed_assignments.append(labels)
            terms = top_terms_from_components(model.components_, feature_names, 20)
            term_lists = [
                terms.loc[terms["component"] == c, "term"].head(10).tolist()
                for c in range(k)
            ]
            seed_coherence.append(topic_npmi(X, feature_names, term_lists, top_n=10))
            seed_reconstruction.append(float(model.reconstruction_err_))
            if seed == NMF_SEEDS[0]:
                fitted[k] = model
                assignments[k] = labels
                terms.to_csv(TABLE_DIR / f"nmf_k{k}_top_terms.csv", index=False)
        aris = [
            adjusted_rand_score(a, b)
            for a, b in itertools.combinations(seed_assignments, 2)
        ]
        rows.append(
            {
                "method": "NMF",
                "variant": f"k={k}",
                "n_topics": k,
                "noise_share": 0.0,
                "rough_coherence_npmi": float(np.nanmean(seed_coherence)),
                "stability_ari_mean": float(np.mean(aris)),
                "stability_ari_min": float(np.min(aris)),
                "reconstruction_error_mean": float(np.mean(seed_reconstruction)),
                "notes": "Hard assignment by max component weight; no explicit noise class.",
            }
        )
    return pd.DataFrame(rows), fitted, assignments


def run_bertopic(
    docs: list[str], embeddings: np.ndarray, X: csr_matrix, feature_names: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame | None, np.ndarray | None]:
    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except Exception as exc:
        return (
            pd.DataFrame(
                [
                    {
                        "method": "BERTopic",
                        "variant": "import",
                        "n_topics": 0,
                        "noise_share": np.nan,
                        "rough_coherence_npmi": np.nan,
                        "stability_ari_mean": np.nan,
                        "stability_ari_min": np.nan,
                        "notes": f"Skipped: import failed: {exc!r}",
                    }
                ]
            ),
            None,
            None,
        )

    result_rows: list[dict[str, object]] = []
    primary_terms: pd.DataFrame | None = None
    primary_topics: np.ndarray | None = None
    seed_topics: list[np.ndarray] = []

    for seed_i, seed in enumerate([RANDOM_SEED, 101, 303]):
        if seed_i > 0 and rss_mb() > 7000:
            result_rows.append(
                {
                    "method": "BERTopic",
                    "variant": f"seed={seed}",
                    "n_topics": 0,
                    "noise_share": np.nan,
                    "rough_coherence_npmi": np.nan,
                    "stability_ari_mean": np.nan,
                    "stability_ari_min": np.nan,
                    "notes": "Skipped seed stability run because peak RSS exceeded 7GB.",
                }
            )
            continue
        try:
            vectorizer = CountVectorizer(
                lowercase=True,
                preprocessor=normalize_text_for_terms,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0,
                max_features=20000,
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_+\-#./]{1,}\b",
            )
            umap_model = UMAP(
                n_neighbors=30,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                random_state=seed,
                low_memory=True,
                n_jobs=1,
            )
            hdbscan_model = HDBSCAN(
                min_cluster_size=PRIMARY_BERTOPIC_MIN_TOPIC_SIZE,
                min_samples=10,
                metric="euclidean",
                prediction_data=True,
                core_dist_n_jobs=1,
            )
            model = BERTopic(
                embedding_model=None,
                vectorizer_model=vectorizer,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                min_topic_size=PRIMARY_BERTOPIC_MIN_TOPIC_SIZE,
                calculate_probabilities=False,
                low_memory=True,
                verbose=False,
            )
            topics, _ = model.fit_transform(docs, embeddings)
            topics_arr = np.asarray(topics)
            seed_topics.append(topics_arr)
            info = model.get_topic_info()
            n_topics = int((info["Topic"] >= 0).sum())
            noise_share = float((topics_arr == -1).mean())
            topic_terms: list[list[str]] = []
            term_rows: list[dict[str, object]] = []
            for topic_id in sorted(t for t in set(topics) if t != -1):
                words = model.get_topic(topic_id) or []
                clean_terms = [w for w, _ in words[:20]]
                topic_terms.append(clean_terms[:10])
                for rank, (term, weight) in enumerate(words[:20], start=1):
                    term_rows.append(
                        {
                            "topic": int(topic_id),
                            "rank": rank,
                            "term": term,
                            "weight": float(weight),
                        }
                    )
            coherence = topic_npmi(X, feature_names, topic_terms, top_n=10)
            if seed_i == 0:
                primary_terms = pd.DataFrame(term_rows)
                primary_topics = topics_arr
                if primary_terms is not None:
                    primary_terms.to_csv(TABLE_DIR / "bertopic_min30_top_terms.csv", index=False)
            result_rows.append(
                {
                    "method": "BERTopic",
                    "variant": f"min_topic_size=30 seed={seed}",
                    "n_topics": n_topics,
                    "noise_share": noise_share,
                    "rough_coherence_npmi": coherence,
                    "stability_ari_mean": np.nan,
                    "stability_ari_min": np.nan,
                    "notes": "UMAP + HDBSCAN; -1 is the HDBSCAN outlier class.",
                }
            )
            del model, umap_model, hdbscan_model, vectorizer
            gc.collect()
        except MemoryError:
            result_rows.append(
                {
                    "method": "BERTopic",
                    "variant": f"min_topic_size=30 seed={seed}",
                    "n_topics": 0,
                    "noise_share": np.nan,
                    "rough_coherence_npmi": np.nan,
                    "stability_ari_mean": np.nan,
                    "stability_ari_min": np.nan,
                    "notes": "Skipped/failed: MemoryError.",
                }
            )
            break
        except Exception as exc:
            result_rows.append(
                {
                    "method": "BERTopic",
                    "variant": f"min_topic_size=30 seed={seed}",
                    "n_topics": 0,
                    "noise_share": np.nan,
                    "rough_coherence_npmi": np.nan,
                    "stability_ari_mean": np.nan,
                    "stability_ari_min": np.nan,
                    "notes": f"Failed: {type(exc).__name__}: {exc}",
                }
            )
            break

    if len(seed_topics) >= 2:
        aris = [
            adjusted_rand_score(a, b) for a, b in itertools.combinations(seed_topics, 2)
        ]
        for row in result_rows:
            if row["method"] == "BERTopic" and str(row["variant"]).startswith("min_topic_size=30"):
                row["stability_ari_mean"] = float(np.mean(aris))
                row["stability_ari_min"] = float(np.min(aris))

    if primary_topics is not None and rss_mb() < 7000:
        for min_topic_size in [20, 50]:
            try:
                vectorizer = CountVectorizer(
                    lowercase=True,
                    preprocessor=normalize_text_for_terms,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=1.0,
                    max_features=20000,
                    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_+\-#./]{1,}\b",
                )
                umap_model = UMAP(
                    n_neighbors=30,
                    n_components=5,
                    min_dist=0.0,
                    metric="cosine",
                    random_state=RANDOM_SEED,
                    low_memory=True,
                    n_jobs=1,
                )
                hdbscan_model = HDBSCAN(
                    min_cluster_size=min_topic_size,
                    min_samples=10,
                    metric="euclidean",
                    prediction_data=False,
                    core_dist_n_jobs=1,
                )
                model = BERTopic(
                    embedding_model=None,
                    vectorizer_model=vectorizer,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    min_topic_size=min_topic_size,
                    calculate_probabilities=False,
                    low_memory=True,
                    verbose=False,
                )
                topics, _ = model.fit_transform(docs, embeddings)
                topics_arr = np.asarray(topics)
                info = model.get_topic_info()
                n_topics = int((info["Topic"] >= 0).sum())
                noise_share = float((topics_arr == -1).mean())
                topic_terms = []
                for topic_id in sorted(t for t in set(topics) if t != -1):
                    words = model.get_topic(topic_id) or []
                    topic_terms.append([w for w, _ in words[:10]])
                coherence = topic_npmi(X, feature_names, topic_terms, top_n=10)
                result_rows.append(
                    {
                        "method": "BERTopic",
                        "variant": f"min_topic_size={min_topic_size} seed={RANDOM_SEED}",
                        "n_topics": n_topics,
                        "noise_share": noise_share,
                        "rough_coherence_npmi": coherence,
                        "stability_ari_mean": np.nan,
                        "stability_ari_min": np.nan,
                        "notes": "Parameter sensitivity run; no seed stability.",
                    }
                )
                del model, umap_model, hdbscan_model, vectorizer
                gc.collect()
            except Exception as exc:
                result_rows.append(
                    {
                        "method": "BERTopic",
                        "variant": f"min_topic_size={min_topic_size} seed={RANDOM_SEED}",
                        "n_topics": 0,
                        "noise_share": np.nan,
                        "rough_coherence_npmi": np.nan,
                        "stability_ari_mean": np.nan,
                        "stability_ari_min": np.nan,
                        "notes": f"Failed: {type(exc).__name__}: {exc}",
                    }
                )
    return pd.DataFrame(result_rows), primary_terms, primary_topics


def align_topics(
    bertopic_terms: pd.DataFrame | None,
    nmf_terms: pd.DataFrame,
    nmf_k: int,
) -> pd.DataFrame:
    if bertopic_terms is None or bertopic_terms.empty:
        return pd.DataFrame(
            columns=[
                "bertopic_topic",
                "nmf_component",
                "jaccard_top20",
                "overlap_terms",
            ]
        )
    b_groups = {
        int(k): set(v["term"].head(20).astype(str))
        for k, v in bertopic_terms.groupby("topic")
    }
    n_groups = {
        int(k): set(v["term"].head(20).astype(str))
        for k, v in nmf_terms[nmf_terms["component"] < nmf_k].groupby("component")
    }
    rows = []
    for b, bterms in b_groups.items():
        best = None
        for n, nterms in n_groups.items():
            overlap = bterms & nterms
            union = bterms | nterms
            jac = len(overlap) / len(union) if union else 0.0
            cand = (jac, n, sorted(overlap))
            if best is None or cand[0] > best[0]:
                best = cand
        if best is not None:
            rows.append(
                {
                    "bertopic_topic": b,
                    "nmf_component": best[1],
                    "jaccard_top20": best[0],
                    "overlap_terms": "; ".join(best[2]),
                }
            )
    return pd.DataFrame(rows).sort_values("jaccard_top20", ascending=False)


def component_names(k: int) -> dict[int, str]:
    if k == 8:
        return ARCHETYPE_NAMES_K8
    return {i: f"NMF Component {i}" for i in range(k)}


def add_tech_domains(df: pd.DataFrame, tech: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(tech, on="uid", how="left")
    tech_cols = [c for c in tech.columns if c != "uid"]
    out[tech_cols] = out[tech_cols].fillna(False).astype(bool)
    out["tech_count"] = out[tech_cols].sum(axis=1).astype(int)
    domain_scores = {}
    for domain, cols in TECH_DOMAIN_COLUMNS.items():
        existing = [c for c in cols if c in out.columns]
        domain_scores[domain] = out[existing].sum(axis=1) if existing else 0
    scores = pd.DataFrame(domain_scores)
    max_score = scores.max(axis=1)
    out["tech_domain"] = scores.idxmax(axis=1)
    out.loc[max_score <= 0, "tech_domain"] = "no_tech_signal"
    return out


def fetch_sample_tech(con: duckdb.DuckDBPyConnection, sample: pd.DataFrame) -> pd.DataFrame:
    uid_df = sample[["uid"]].copy()
    con.register("sample_uids", uid_df)
    tech_cols = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{TECH_PATH.as_posix()}')"
    ).fetchdf()["column_name"].tolist()
    select_cols = ", ".join([f"t.{c}" for c in tech_cols])
    return con.execute(
        f"""
        SELECT {select_cols}
        FROM read_parquet('{TECH_PATH.as_posix()}') t
        JOIN sample_uids USING(uid)
        """
    ).fetchdf()


def sample_summaries(sample: pd.DataFrame, sample_with_tech: pd.DataFrame) -> None:
    write_df(
        sample.groupby(["source", "period", "seniority_3level", "text_source"], dropna=False)
        .agg(
            n=("uid", "count"),
            companies=("company_name_canonical", "nunique"),
            aggregator_rows=("is_aggregator", "sum"),
            mean_chars=("char_len", "mean"),
            mean_yoe=("yoe_extracted", "mean"),
        )
        .reset_index(),
        "sample_composition.csv",
    )
    top_companies = (
        sample.groupby(["company_name_canonical"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .head(30)
    )
    write_df(top_companies, "sample_top_companies.csv")
    write_df(
        sample_with_tech.groupby(["tech_domain"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False),
        "sample_tech_domain_distribution.csv",
    )


def assign_best_nmf_labels(
    vectorizer: TfidfVectorizer, model: NMF, con: duckdb.DuckDBPyConnection
) -> pd.DataFrame:
    names = component_names(BEST_NMF_K)
    query = f"""
      SELECT uid, description_cleaned
      FROM read_parquet('{CLEANED_PATH.as_posix()}')
      WHERE text_source = 'llm'
        AND length(coalesce(description_cleaned, '')) >= {MIN_CHARS}
      ORDER BY uid
    """
    reader = con.execute(query).fetch_record_batch(rows_per_batch=4096)
    out_frames: list[pd.DataFrame] = []
    for batch in reader:
        table = pa.Table.from_batches([batch])
        pdf = table.to_pandas()
        X_chunk = vectorizer.transform(pdf["description_cleaned"].fillna(""))
        W_chunk = model.transform(X_chunk)
        labels = W_chunk.argmax(axis=1).astype(int)
        out_frames.append(
            pd.DataFrame(
                {
                    "uid": pdf["uid"].to_numpy(),
                    "archetype": [f"nmf_{int(x)}" for x in labels],
                    "archetype_name": [names[int(x)] for x in labels],
                }
            )
        )
        del pdf, X_chunk, W_chunk, labels, table
        gc.collect()
    labels_df = pd.concat(out_frames, ignore_index=True)
    pq.write_table(pa.Table.from_pandas(labels_df, preserve_index=False), LABEL_PATH)
    return labels_df


def aggregate_label_outputs(con: duckdb.DuckDBPyConnection) -> None:
    label = pd.read_parquet(LABEL_PATH)
    con.register("labels", label)
    tech_cols = [
        c
        for c in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{TECH_PATH.as_posix()}')"
        ).fetchdf()["column_name"].tolist()
        if c != "uid"
    ]
    tech_sum_expr = " + ".join([f"CAST(t.{c} AS INTEGER)" for c in tech_cols])
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE label_meta AS
        SELECT
          l.uid,
          l.archetype,
          l.archetype_name,
          c.source,
          c.period,
          c.seniority_final,
          c.seniority_3level,
          c.is_aggregator,
          c.company_name_canonical,
          c.yoe_extracted,
          c.swe_classification_tier,
          length(coalesce(c.description_cleaned, '')) AS char_len,
          array_length(regexp_split_to_array(trim(coalesce(c.description_cleaned, '')), '\\s+')) AS word_count,
          ({tech_sum_expr})::INTEGER AS tech_count
        FROM labels l
        JOIN read_parquet('{CLEANED_PATH.as_posix()}') c USING(uid)
        LEFT JOIN read_parquet('{TECH_PATH.as_posix()}') t USING(uid)
        """
    )
    char_df = con.execute(
        """
        SELECT
          archetype,
          archetype_name,
          count(*) AS n,
          count(DISTINCT company_name_canonical) AS companies,
          avg(CASE WHEN is_aggregator THEN 1 ELSE 0 END) AS aggregator_share,
          avg(char_len) AS avg_chars,
          avg(word_count) AS avg_words,
          avg(yoe_extracted) AS avg_yoe_known,
          avg(tech_count) AS avg_tech_count,
          avg(CASE WHEN seniority_final = 'entry' THEN 1 WHEN seniority_final <> 'unknown' THEN 0 ELSE NULL END) AS j1_entry_share_known,
          avg(CASE WHEN seniority_final IN ('entry', 'associate') THEN 1 WHEN seniority_final <> 'unknown' THEN 0 ELSE NULL END) AS j2_entry_assoc_share_known,
          avg(CASE WHEN yoe_extracted <= 2 THEN 1 WHEN yoe_extracted IS NOT NULL THEN 0 ELSE NULL END) AS j3_yoe_le2_share_known,
          avg(CASE WHEN yoe_extracted <= 3 THEN 1 WHEN yoe_extracted IS NOT NULL THEN 0 ELSE NULL END) AS j4_yoe_le3_share_known
        FROM label_meta
        GROUP BY archetype, archetype_name
        ORDER BY archetype
        """
    ).fetchdf()
    write_df(char_df, "archetype_characterization.csv")

    period_df = con.execute(
        """
        SELECT
          archetype,
          archetype_name,
          period,
          count(*) AS n,
          count(*) * 1.0 / sum(count(*)) OVER (PARTITION BY period) AS share_of_period,
          count(*) * 1.0 / sum(count(*)) OVER (PARTITION BY archetype) AS share_of_archetype,
          avg(CASE WHEN is_aggregator THEN 1 ELSE 0 END) AS aggregator_share,
          avg(tech_count) AS avg_tech_count,
          avg(char_len) AS avg_chars
        FROM label_meta
        GROUP BY archetype, archetype_name, period
        ORDER BY archetype, period
        """
    ).fetchdf()
    write_df(period_df, "archetype_period_distribution.csv")

    entry_rows = []
    for definition, denom, condition in [
        ("J1", "known_seniority", "seniority_final = 'entry'"),
        ("J2", "known_seniority", "seniority_final IN ('entry','associate')"),
        ("J3", "yoe_known", "yoe_extracted <= 2"),
        ("J4", "yoe_known", "yoe_extracted <= 3"),
    ]:
        where_known = "seniority_final <> 'unknown'" if denom == "known_seniority" else "yoe_extracted IS NOT NULL"
        df = con.execute(
            f"""
            SELECT
              '{definition}' AS definition,
              '{denom}' AS denominator,
              archetype,
              archetype_name,
              period,
              sum(CASE WHEN {condition} THEN 1 ELSE 0 END) AS numerator,
              sum(CASE WHEN {where_known} THEN 1 ELSE 0 END) AS denominator_n,
              avg(CASE WHEN {condition} THEN 1 WHEN {where_known} THEN 0 ELSE NULL END) AS share
            FROM label_meta
            GROUP BY archetype, archetype_name, period
            ORDER BY definition, archetype, period
            """
        ).fetchdf()
        entry_rows.append(df)
    write_df(pd.concat(entry_rows, ignore_index=True), "archetype_entry_panel.csv")

    sens = con.execute(
        """
        WITH base AS (
          SELECT 'all_rows' AS spec, * FROM label_meta
          UNION ALL
          SELECT 'exclude_aggregators' AS spec, * FROM label_meta WHERE NOT is_aggregator
          UNION ALL
          SELECT 'exclude_title_lookup_llm' AS spec, * FROM label_meta
          WHERE coalesce(swe_classification_tier, '') <> 'title_lookup_llm'
        )
        SELECT
          spec,
          archetype,
          archetype_name,
          period,
          count(*) AS n,
          count(*) * 1.0 / sum(count(*)) OVER (PARTITION BY spec, period) AS share_of_period
        FROM base
        GROUP BY spec, archetype, archetype_name, period
        ORDER BY spec, archetype, period
        """
    ).fetchdf()
    write_df(sens, "archetype_temporal_sensitivities.csv")

    coverage = con.execute(
        f"""
        SELECT
          c.source,
          c.period,
          c.text_source,
          count(*) AS total_cleaned_rows,
          sum(CASE WHEN c.text_source='llm' THEN 1 ELSE 0 END) AS llm_rows,
          sum(CASE WHEN c.text_source='llm' AND length(coalesce(c.description_cleaned,'')) >= {MIN_CHARS} THEN 1 ELSE 0 END) AS llm_ge100_rows,
          sum(CASE WHEN l.uid IS NOT NULL THEN 1 ELSE 0 END) AS label_rows
        FROM read_parquet('{CLEANED_PATH.as_posix()}') c
        LEFT JOIN labels l USING(uid)
        GROUP BY c.source, c.period, c.text_source
        ORDER BY c.period, c.source, c.text_source
        """
    ).fetchdf()
    write_df(coverage, "label_coverage_by_source_period.csv")


def nmf_sample_outputs(
    sample: pd.DataFrame,
    sample_with_tech: pd.DataFrame,
    nmf_model: NMF,
    X: csr_matrix,
    feature_names: np.ndarray,
) -> pd.DataFrame:
    W = nmf_model.transform(X)
    labels = W.argmax(axis=1).astype(int)
    names = component_names(BEST_NMF_K)
    out = sample_with_tech.copy()
    out["archetype_id"] = labels
    out["archetype"] = [f"nmf_{x}" for x in labels]
    out["archetype_name"] = [names[int(x)] for x in labels]
    write_df(
        out[
            [
                "uid",
                "source",
                "period",
                "seniority_final",
                "seniority_3level",
                "is_aggregator",
                "company_name_canonical",
                "yoe_extracted",
                "swe_classification_tier",
                "char_len",
                "word_count",
                "tech_count",
                "tech_domain",
                "archetype",
                "archetype_name",
            ]
        ],
        "sample_best_method_assignments.csv",
    )
    return out


def dominant_structure(sample_labels: pd.DataFrame) -> pd.DataFrame:
    labels = sample_labels["archetype"].astype(str)
    top_companies = set(
        sample_labels["company_name_canonical"].value_counts(dropna=False).head(30).index
    )
    company_group = sample_labels["company_name_canonical"].where(
        sample_labels["company_name_canonical"].isin(top_companies), "other_company"
    )
    factors = {
        "tech_domain": sample_labels["tech_domain"].astype(str),
        "source": sample_labels["source"].astype(str),
        "period": sample_labels["period"].astype(str),
        "year": sample_labels["year"].astype(str),
        "seniority_3level": sample_labels["seniority_3level"].astype(str),
        "is_aggregator": sample_labels["is_aggregator"].astype(str),
        "top30_company": company_group.astype(str),
        "swe_classification_tier": sample_labels["swe_classification_tier"].astype(str),
    }
    rows = []
    for factor, vals in factors.items():
        rows.append(
            {
                "factor": factor,
                "nmi": float(normalized_mutual_info_score(labels, vals)),
                "levels": int(vals.nunique(dropna=False)),
            }
        )
    return pd.DataFrame(rows).sort_values("nmi", ascending=False)


def raw_text_sensitivity(
    con: duckdb.DuckDBPyConnection,
    sample_labels: pd.DataFrame,
    primary_terms: pd.DataFrame,
) -> pd.DataFrame:
    sample_uids = sample_labels[["uid", "archetype"]].copy()
    con.register("sample_label_uids", sample_uids)
    raw = con.execute(
        f"""
        SELECT s.uid, s.archetype, u.description
        FROM sample_label_uids s
        JOIN read_parquet('{UNIFIED_PATH.as_posix()}') u USING(uid)
        ORDER BY s.uid
        """
    ).fetchdf()
    vec = TfidfVectorizer(
        lowercase=True,
        preprocessor=normalize_text_for_terms,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=10,
        max_df=0.65,
        max_features=20000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_+\-#./]{1,}\b",
    )
    X_raw = vec.fit_transform(raw["description"].fillna(""))
    raw_model = NMF(
        n_components=BEST_NMF_K,
        init="nndsvdar",
        random_state=NMF_SEEDS[0],
        max_iter=350,
        solver="cd",
        beta_loss="frobenius",
    )
    W_raw = raw_model.fit_transform(X_raw)
    raw_labels = W_raw.argmax(axis=1).astype(int)
    nmi = normalized_mutual_info_score(raw["archetype"].astype(str), raw_labels.astype(str))
    raw_terms = top_terms_from_components(raw_model.components_, vec.get_feature_names_out(), 20)
    raw_terms.to_csv(TABLE_DIR / f"nmf_k{BEST_NMF_K}_raw_description_top_terms.csv", index=False)
    clean_sets = {
        int(k): set(v["term"].head(20).astype(str))
        for k, v in primary_terms.groupby("component")
    }
    raw_sets = {
        int(k): set(v["term"].head(20).astype(str)) for k, v in raw_terms.groupby("component")
    }
    sim = np.zeros((BEST_NMF_K, BEST_NMF_K))
    for i in range(BEST_NMF_K):
        for j in range(BEST_NMF_K):
            union = clean_sets.get(i, set()) | raw_sets.get(j, set())
            overlap = clean_sets.get(i, set()) & raw_sets.get(j, set())
            sim[i, j] = len(overlap) / len(union) if union else 0.0
    row_ind, col_ind = linear_sum_assignment(-sim)
    rows = [
        {
            "metric": "clean_vs_raw_assignment_nmi_same_sample",
            "value": float(nmi),
            "notes": "NMI between primary clean-text NMF labels and raw-description NMF labels; component IDs are not semantically aligned.",
        }
    ]
    for i, j in zip(row_ind, col_ind):
        rows.append(
            {
                "metric": f"top_term_jaccard_clean_component_{i}_raw_component_{j}",
                "value": float(sim[i, j]),
                "notes": "; ".join(sorted(clean_sets.get(i, set()) & raw_sets.get(j, set()))),
            }
        )
    return pd.DataFrame(rows)


def plot_embedding_panels(sample_labels: pd.DataFrame, embeddings: np.ndarray) -> None:
    from umap import UMAP

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    reducer = UMAP(
        n_neighbors=30,
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_SEED,
        low_memory=True,
        n_jobs=1,
    )
    coords = reducer.fit_transform(embeddings)
    pca = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(embeddings)
    for coord_name, arr, filename in [
        ("UMAP", coords, "embedding_umap_panels.png"),
        ("PCA", pca, "embedding_pca_panels.png"),
    ]:
        plot_df = sample_labels.copy()
        plot_df["x"] = arr[:, 0]
        plot_df["y"] = arr[:, 1]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
        panels = [
            ("archetype_name", "Archetype"),
            ("period", "Period"),
            ("seniority_3level", "Seniority"),
        ]
        for ax, (col, title) in zip(axes, panels):
            values = plot_df[col].astype(str)
            cats = values.drop_duplicates().tolist()
            cmap = plt.get_cmap("tab20")
            color_map = {cat: cmap(i % 20) for i, cat in enumerate(cats)}
            ax.scatter(
                plot_df["x"],
                plot_df["y"],
                s=4,
                alpha=0.55,
                linewidths=0,
                c=values.map(color_map),
            )
            ax.set_title(f"{coord_name} by {title}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", label=cat, markerfacecolor=color_map[cat], markersize=5)
                for cat in cats[:12]
            ]
            ax.legend(handles=handles, loc="best", fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    assert_regex_examples()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    con = connect()

    sample = build_sample(con)
    embeddings = load_sample_embeddings(sample)
    docs = sample["description_cleaned"].fillna("").tolist()
    vectorizer, X = build_tfidf(sample["description_cleaned"])
    feature_names = vectorizer.get_feature_names_out()

    sample_tech = fetch_sample_tech(con, sample)
    sample_with_tech = add_tech_domains(sample, sample_tech)
    sample_summaries(sample, sample_with_tech)

    nmf_summary, nmf_models, nmf_assignments = fit_nmf_models(X, feature_names)
    best_nmf = nmf_models[BEST_NMF_K]
    best_terms = pd.read_csv(TABLE_DIR / f"nmf_k{BEST_NMF_K}_top_terms.csv")

    bertopic_summary, bertopic_terms, bertopic_topics = run_bertopic(
        docs, embeddings, X, feature_names
    )
    method_summary = pd.concat([bertopic_summary, nmf_summary], ignore_index=True)
    method_summary.to_csv(TABLE_DIR / "methods_comparison.csv", index=False)

    alignment = align_topics(bertopic_terms, best_terms, BEST_NMF_K)
    write_df(alignment, "method_topic_alignment.csv")

    sample_labels = nmf_sample_outputs(sample, sample_with_tech, best_nmf, X, feature_names)
    nmi_df = dominant_structure(sample_labels)
    write_df(nmi_df, "dominant_structure_nmi.csv")

    raw_sens = raw_text_sensitivity(con, sample_labels, best_terms)
    write_df(raw_sens, "description_text_source_sensitivity.csv")

    labels_df = assign_best_nmf_labels(vectorizer, best_nmf, con)
    aggregate_label_outputs(con)

    plot_embedding_panels(sample_labels, embeddings)

    summary = {
        "sample_n": int(len(sample)),
        "company_cap": COMPANY_CAP,
        "min_chars": MIN_CHARS,
        "sample_targets": SAMPLE_TARGETS,
        "best_method": "NMF",
        "best_nmf_k": BEST_NMF_K,
        "label_rows": int(len(labels_df)),
        "label_path": str(LABEL_PATH.relative_to(ROOT)),
        "bertopic_completed": bool(
            (
                method_summary["method"].eq("BERTopic")
                & (method_summary["n_topics"].fillna(0) > 0)
            ).any()
        ),
        "peak_rss_mb": rss_mb(),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
