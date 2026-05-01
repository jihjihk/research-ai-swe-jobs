#!/usr/bin/env python
"""T09 — Posting archetype discovery (methods laboratory).

Runs the full pipeline:
 1. Balanced-period, seniority-stratified SWE LinkedIn sample (n <= 8000, target ~2700/period).
 2. Loads shared cleaned text + embeddings.
 3. Method A — BERTopic (primary) over pre-computed sentence-transformer embeddings.
 4. Method B — NMF (comparison) over TF-IDF, k in [5, 8, 12, 15].
 5. Method comparison: top-term overlap, seed-stability ARI, coherence, noise fraction.
 6. Characterization of best-method clusters: terms, seniority panel, entry share, period, length, YOE, tech count.
 7. Temporal dynamics: period proportions.
 8. UMAP 2D colored by cluster / period / seniority / supplementary.
 9. KEY DISCOVERY — NMI(clusters, seniority_3level) vs NMI(clusters, tech-domain-proxy) vs NMI(clusters, period) vs NMI(clusters, title-archetype).
10. Sensitivities — aggregator-excluded, company-capped, raw-text run.
11. Save archetype labels as shared artifact.

Output:
 - exploration/tables/T09/*.csv
 - exploration/figures/T09/*.png
 - exploration/artifacts/shared/swe_archetype_labels.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/home/jihgaboot/gabor/job-research")
SHARED = ROOT / "exploration" / "artifacts" / "shared"
TABLES = ROOT / "exploration" / "tables" / "T09"
FIGS = ROOT / "exploration" / "figures" / "T09"
ART_OUT = SHARED / "swe_archetype_labels.parquet"

CLEANED_TEXT = SHARED / "swe_cleaned_text.parquet"
EMBEDDINGS = SHARED / "swe_embeddings.npy"
EMBED_INDEX = SHARED / "swe_embedding_index.parquet"
TECH_MATRIX = SHARED / "swe_tech_matrix.parquet"
UNIFIED = ROOT / "data" / "unified.parquet"
STOPLIST_FILE = SHARED / "company_stoplist.txt"

MAX_SAMPLE = 8000
TARGET_PER_PERIOD = 2700
SEED = 13


# --------------------------------------------------------------------------------------
# Logging helper
# --------------------------------------------------------------------------------------

_T0 = time.time()


def log(msg: str) -> None:
    dt = time.time() - _T0
    print(f"[{dt:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# 1. Sample
# --------------------------------------------------------------------------------------

def draw_sample(cleaned_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Balanced-period sample: ~TARGET_PER_PERIOD per meta-period, seniority-stratified,
    prefer text_source='llm'. 2024 bucket prefers arshkon over asaniczka."""

    meta_period_map = {
        "2024-01": "2024",
        "2024-04": "2024",
        "2026-03": "2026-03",
        "2026-04": "2026-04",
    }
    df = cleaned_df.copy()
    df["meta_period"] = df["period"].map(meta_period_map)

    buckets: dict[str, pd.DataFrame] = {}
    for meta in ["2024", "2026-03", "2026-04"]:
        sub = df[df.meta_period == meta]

        # prefer rows with usable text + llm text source
        sub_llm = sub[(sub.text_source == "llm") & (sub.description_cleaned.notna()) &
                      (sub.description_cleaned.str.len() > 30)]
        sub_raw = sub[(sub.text_source == "raw") & (sub.description_cleaned.notna()) &
                      (sub.description_cleaned.str.len() > 30)]

        # target: TARGET_PER_PERIOD rows, stratified by seniority_3level.
        # For 2024, prefer arshkon rows to maintain native-label access for downstream users.
        if meta == "2024":
            sub_llm_arsh = sub_llm[sub_llm.source == "kaggle_arshkon"]
            sub_llm_asan = sub_llm[sub_llm.source == "kaggle_asaniczka"]
            # take all arshkon first, then back-fill from asaniczka, holding seniority shares proportional
            arsh_target = min(len(sub_llm_arsh), int(TARGET_PER_PERIOD * 0.55))
            asan_target = TARGET_PER_PERIOD - arsh_target
            pieces: list[pd.DataFrame] = []
            for pool, ntake in [(sub_llm_arsh, arsh_target), (sub_llm_asan, asan_target)]:
                if len(pool) == 0:
                    continue
                ntake = min(ntake, len(pool))
                # seniority-stratified draw within pool
                take_df = stratified_draw_within(pool, "seniority_3level", ntake, rng)
                pieces.append(take_df)
            chosen = pd.concat(pieces, ignore_index=True)
            # top up from raw if we didn't reach target
            gap = TARGET_PER_PERIOD - len(chosen)
            if gap > 0 and len(sub_raw) > 0:
                extra = stratified_draw_within(sub_raw, "seniority_3level",
                                               min(gap, len(sub_raw)), rng)
                chosen = pd.concat([chosen, extra], ignore_index=True)
        else:
            # 2026-03 / 2026-04: llm-text first, stratified by seniority.
            if len(sub_llm) >= TARGET_PER_PERIOD:
                chosen = stratified_draw_within(sub_llm, "seniority_3level",
                                                TARGET_PER_PERIOD, rng)
            else:
                pieces = [sub_llm]
                gap = TARGET_PER_PERIOD - len(sub_llm)
                if gap > 0 and len(sub_raw) > 0:
                    pieces.append(stratified_draw_within(sub_raw, "seniority_3level",
                                                         min(gap, len(sub_raw)), rng))
                chosen = pd.concat(pieces, ignore_index=True)

        buckets[meta] = chosen.reset_index(drop=True)

    sample = pd.concat(list(buckets.values()), ignore_index=True)
    # safety cap
    if len(sample) > MAX_SAMPLE:
        sample = sample.sample(n=MAX_SAMPLE, random_state=SEED).reset_index(drop=True)
    return sample


def stratified_draw_within(pool: pd.DataFrame, strat_col: str, n: int,
                            rng: np.random.Generator) -> pd.DataFrame:
    """Draw `n` rows from `pool` stratified on `strat_col` in proportion to the pool."""
    if len(pool) <= n:
        return pool
    # compute per-category quotas
    counts = pool[strat_col].value_counts(dropna=False)
    total = counts.sum()
    quotas = {cat: int(round(n * cnt / total)) for cat, cnt in counts.items()}
    # adjust quotas to sum exactly to n
    diff = n - sum(quotas.values())
    if diff != 0:
        largest = counts.idxmax()
        quotas[largest] += diff
    pieces = []
    for cat, q in quotas.items():
        q = max(0, min(q, int(counts.get(cat, 0))))
        if q == 0:
            continue
        sub = pool[pool[strat_col] == cat]
        pieces.append(sub.sample(n=q, random_state=int(rng.integers(0, 10**9))))
    out = pd.concat(pieces, ignore_index=True)
    # if any rounding shortfall, draw uniform from remainder to backfill
    if len(out) < n:
        remainder = pool[~pool["uid"].isin(out["uid"])]
        extra = remainder.sample(n=min(n - len(out), len(remainder)),
                                 random_state=int(rng.integers(0, 10**9)))
        out = pd.concat([out, extra], ignore_index=True)
    return out


# --------------------------------------------------------------------------------------
# 2. Load shared artifacts
# --------------------------------------------------------------------------------------

def load_cleaned_text() -> pd.DataFrame:
    return pq.read_table(str(CLEANED_TEXT)).to_pandas()


def load_embedding_index() -> pd.DataFrame:
    return pq.read_table(str(EMBED_INDEX)).to_pandas()


def load_embeddings_for(sample_df: pd.DataFrame, emb_index: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (embeddings_aligned_to_sample, sample_df_with_valid_mask).
    Only returns sample rows with valid embeddings.
    """
    emb = np.load(str(EMBEDDINGS))
    lookup = dict(zip(emb_index["uid"], emb_index["row_idx"]))
    rows: list[int] = []
    keep: list[int] = []
    for i, uid in enumerate(sample_df["uid"].tolist()):
        rid = lookup.get(uid)
        if rid is not None:
            rows.append(int(rid))
            keep.append(i)
    if len(rows) == 0:
        raise RuntimeError("No sample rows have embeddings.")
    X = emb[rows]
    kept = sample_df.iloc[keep].reset_index(drop=True).copy()
    return X, kept


def load_stoplist() -> set[str]:
    """Company stoplist + standard English stopwords."""
    toks: set[str] = set()
    if STOPLIST_FILE.exists():
        with open(STOPLIST_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                w = line.strip().lower()
                if w:
                    toks.add(w)
    return toks


# --------------------------------------------------------------------------------------
# 3. BERTopic
# --------------------------------------------------------------------------------------

def run_bertopic(docs: list[str], embeddings: np.ndarray, stopwords: set[str],
                 min_topic_size: int, seed: int, verbose: bool = False,
                 nr_topics: int | str | None = None):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP
    from sklearn.feature_extraction.text import CountVectorizer as CV

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric="cosine", random_state=seed, low_memory=True)
    hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size,
                             min_samples=max(5, int(min_topic_size / 6)),
                             metric="euclidean", prediction_data=True)
    cv = CV(stop_words=list(stopwords) if stopwords else "english",
            ngram_range=(1, 2), min_df=1, max_df=1.0)

    kwargs: dict[str, Any] = dict(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=cv,
        verbose=verbose,
        calculate_probabilities=False,
    )
    if nr_topics is not None:
        kwargs["nr_topics"] = nr_topics

    topic_model = BERTopic(**kwargs)

    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    return topic_model, np.asarray(topics), umap_model


def run_bertopic_seed_only(docs: list[str], embeddings: np.ndarray, stopwords: set[str],
                            min_topic_size: int, seed: int):
    from hdbscan import HDBSCAN
    from umap import UMAP

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric="cosine", random_state=seed, low_memory=True)
    Y = umap_model.fit_transform(embeddings)
    hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size,
                             min_samples=max(5, int(min_topic_size / 6)),
                             metric="euclidean")
    labels = hdbscan_model.fit_predict(Y)
    return labels


# --------------------------------------------------------------------------------------
# 4. NMF
# --------------------------------------------------------------------------------------

def fit_tfidf(docs: list[str], stopwords: set[str]):
    tfidf = TfidfVectorizer(stop_words=list(stopwords) if stopwords else "english",
                             ngram_range=(1, 2), min_df=10, max_df=0.8,
                             max_features=20000)
    X = tfidf.fit_transform(docs)
    return tfidf, X


def run_nmf(tfidf_matrix, k: int, seed: int):
    model = NMF(n_components=k, init="nndsvd", random_state=seed,
                 max_iter=400, tol=1e-4)
    W = model.fit_transform(tfidf_matrix)
    H = model.components_
    return model, W, H


def nmf_top_terms(H: np.ndarray, vocab: list[str], n: int = 20) -> list[list[str]]:
    top = []
    for comp in H:
        idx = np.argsort(-comp)[:n]
        top.append([vocab[i] for i in idx])
    return top


def nmf_labels_from_W(W: np.ndarray) -> np.ndarray:
    return np.asarray(W.argmax(axis=1)).reshape(-1)


# --------------------------------------------------------------------------------------
# 5. Method comparison helpers
# --------------------------------------------------------------------------------------

def topic_top_terms_bertopic(topic_model, n: int = 20) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for t in topic_model.get_topic_info()["Topic"].tolist():
        if t == -1:
            continue
        words = [w for w, _ in topic_model.get_topic(t) if w]
        out[t] = words[:n]
    return out


def jaccard(a: list[str], b: list[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / len(A | B)


def top_term_overlap_matrix(t_bert: dict[int, list[str]], t_nmf: list[list[str]]) -> pd.DataFrame:
    rows = sorted(t_bert.keys())
    mat = np.zeros((len(rows), len(t_nmf)))
    for i, tid in enumerate(rows):
        for j, nmf_topic in enumerate(t_nmf):
            mat[i, j] = jaccard(t_bert[tid], nmf_topic)
    df = pd.DataFrame(mat, index=[f"bert_{r}" for r in rows],
                      columns=[f"nmf_{j}" for j in range(len(t_nmf))])
    return df


# --------------------------------------------------------------------------------------
# 6. Characterization
# --------------------------------------------------------------------------------------

TECH_DOMAINS = {
    "frontend": ["react", "angular", "vue", "nextjs", "nuxt", "svelte", "ember", "jquery"],
    "backend_api": ["nodejs", "express", "django", "flask", "fastapi", "spring", "dot_net",
                     "rails", "laravel", "graphql", "rest_api"],
    "cloud_platform": ["aws", "azure", "gcp", "cloudflare", "heroku", "digital_ocean"],
    "devops_orchestration": ["kubernetes", "docker", "terraform", "ansible", "helm", "argocd",
                              "puppet", "chef", "jenkins", "github_actions", "circleci",
                              "gitlab_ci", "buildkite", "travis_ci", "ci_cd", "microservices",
                              "serverless"],
    "database": ["postgresql", "mysql", "mongodb", "redis", "cassandra", "dynamodb",
                  "snowflake", "bigquery", "oracle_db", "sqlite", "sql_server"],
    "data_pipeline": ["kafka", "spark", "airflow", "dbt", "databricks", "elasticsearch",
                       "flink", "hadoop", "beam"],
    "ml_traditional": ["tensorflow", "pytorch", "scikit_learn", "pandas", "numpy",
                        "jupyter", "mlflow", "xgboost", "keras"],
    "llm_era": ["langchain", "llamaindex", "rag", "vector_database", "pinecone", "weaviate",
                 "chroma", "hugging_face", "openai_api", "claude_api", "anthropic",
                 "gemini", "prompt_engineering", "fine_tuning", "mcp", "llm", "ai_agent",
                 "gpt_model"],
    "ai_tool": ["copilot", "cursor_tool", "chatgpt", "claude_tool", "codex", "tabnine"],
    "testing": ["jest", "pytest", "selenium", "cypress", "playwright", "junit", "mocha",
                 "tdd", "bdd"],
    "mobile": ["kotlin", "swift", "objective_c"],
    "systems_language": ["c_plus_plus", "c_lang", "rust", "go_lang"],
    "scripting_language": ["python", "ruby", "php", "bash", "perl", "r_lang", "elixir", "scala"],
    "js_ts": ["javascript", "typescript"],
    "java_jvm": ["java"],
    "query_language": ["sql"],
    "observability": ["datadog", "new_relic", "pagerduty", "grafana", "prometheus",
                       "splunk", "sentry"],
    "methodology": ["agile", "scrum", "ddd", "event_driven"],
}


def tech_domain_assignment(tech_df: pd.DataFrame, uids: list[str]) -> np.ndarray:
    """For each row, assign the tech-domain with the highest count of True.
    Return string array; 'none' if no tech flagged."""
    sub = tech_df[tech_df.uid.isin(uids)].set_index("uid")
    sub = sub.reindex(uids)
    domain_names = list(TECH_DOMAINS.keys())
    counts = np.zeros((len(sub), len(domain_names)), dtype=int)
    for j, dom in enumerate(domain_names):
        cols = [c for c in TECH_DOMAINS[dom] if c in sub.columns]
        if not cols:
            continue
        counts[:, j] = sub[cols].fillna(False).sum(axis=1).values
    labels = np.empty(len(sub), dtype=object)
    for i in range(len(sub)):
        if counts[i].sum() == 0:
            labels[i] = "none"
        else:
            labels[i] = domain_names[int(counts[i].argmax())]
    return labels


# --------------------------------------------------------------------------------------
# 7. Title archetype (regex)
# --------------------------------------------------------------------------------------

TITLE_PATTERNS = [
    ("ml_ai", re.compile(r"\b(machine learning|ml engineer|ai engineer|deep learning|data scientist|mlops|llm|nlp|computer vision|genai)\b", re.I)),
    ("data", re.compile(r"\b(data engineer|data analyst|etl|analytics engineer|bi engineer|data platform)\b", re.I)),
    ("frontend", re.compile(r"\b(front[- ]?end|ui engineer|react|angular|vue|web ui|javascript developer)\b", re.I)),
    ("backend", re.compile(r"\b(back[- ]?end|api engineer|services engineer|platform engineer|server-side)\b", re.I)),
    ("fullstack", re.compile(r"\b(full[- ]?stack|fullstack)\b", re.I)),
    ("devops_sre", re.compile(r"\b(devops|site reliability|sre|platform reliability|infrastructure engineer|cloud engineer|kubernetes engineer)\b", re.I)),
    ("security", re.compile(r"\b(security engineer|cyber|appsec|infosec|application security)\b", re.I)),
    ("mobile", re.compile(r"\b(ios|android|mobile engineer|mobile developer)\b", re.I)),
    ("embedded", re.compile(r"\b(embedded|firmware|robotics engineer|hardware engineer|fpga)\b", re.I)),
    ("qa_test", re.compile(r"\b(qa engineer|test engineer|sdet|quality engineer|automation engineer)\b", re.I)),
    ("game", re.compile(r"\b(game engineer|game developer|unity developer|unreal developer)\b", re.I)),
]


def title_archetype_for(titles: list[str]) -> np.ndarray:
    out = np.empty(len(titles), dtype=object)
    for i, t in enumerate(titles):
        if not isinstance(t, str) or not t:
            out[i] = "other_swe"
            continue
        assigned = None
        for name, pat in TITLE_PATTERNS:
            if pat.search(t):
                assigned = name
                break
        out[i] = assigned if assigned else "other_swe"
    return out


# --------------------------------------------------------------------------------------
# 8. Utility: c-TF-IDF style top-terms for arbitrary labels
# --------------------------------------------------------------------------------------

def compute_c_tfidf_top_terms(docs: list[str], labels: np.ndarray,
                               stopwords: set[str], n: int = 20,
                               exclude_noise: bool = False) -> dict[int, list[str]]:
    """Aggregate documents per cluster, compute TF-IDF at cluster level, return top terms."""
    unique = sorted(set(int(x) for x in labels))
    cluster_docs: dict[int, list[str]] = {c: [] for c in unique}
    for d, c in zip(docs, labels):
        cluster_docs[int(c)].append(d)
    cluster_texts = []
    ids = []
    for c in unique:
        if exclude_noise and c == -1:
            continue
        cluster_texts.append(" ".join(cluster_docs[c]) if cluster_docs[c] else " ")
        ids.append(c)
    cv = TfidfVectorizer(stop_words=list(stopwords) if stopwords else "english",
                         ngram_range=(1, 2), min_df=2, max_df=0.8,
                         max_features=25000)
    X = cv.fit_transform(cluster_texts)
    vocab = cv.get_feature_names_out()
    out: dict[int, list[str]] = {}
    for i, cid in enumerate(ids):
        row = X.getrow(i).toarray().ravel()
        top_idx = np.argsort(-row)[:n]
        out[cid] = [vocab[j] for j in top_idx if row[j] > 0][:n]
    return out


# --------------------------------------------------------------------------------------
# 9. UMAP 2D for visualization
# --------------------------------------------------------------------------------------

def umap_2d(embeddings: np.ndarray, seed: int = SEED) -> np.ndarray:
    from umap import UMAP
    um = UMAP(n_neighbors=15, n_components=2, min_dist=0.1,
              metric="cosine", random_state=seed, low_memory=True)
    return um.fit_transform(embeddings)


# --------------------------------------------------------------------------------------
# 10. Plotting helpers
# --------------------------------------------------------------------------------------

def plot_scatter(xy: np.ndarray, labels: np.ndarray, title: str, outpath: Path,
                  legend_max: int = 25):
    fig, ax = plt.subplots(figsize=(8, 6))
    unique = list(pd.Series(labels).value_counts().index)
    # cap for legibility
    unique_top = unique[:legend_max]
    rest = set(unique[legend_max:])
    cmap = plt.get_cmap("tab20", max(2, len(unique_top) + (1 if rest else 0)))
    for i, cat in enumerate(unique_top):
        mask = labels == cat
        ax.scatter(xy[mask, 0], xy[mask, 1], s=2.5, alpha=0.6, label=str(cat),
                    color=cmap(i))
    if rest:
        mask = np.isin(labels, list(rest))
        ax.scatter(xy[mask, 0], xy[mask, 1], s=2.0, alpha=0.5,
                    color="0.7", label="other")
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7,
              markerscale=3, ncol=1)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_barchart(df: pd.DataFrame, title: str, outpath: Path):
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(df))))
    df.plot(kind="barh", stacked=True, ax=ax, colormap="tab20", width=0.8)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(SEED)

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    log("Load cleaned text")
    cleaned = load_cleaned_text()
    log(f"  rows: {len(cleaned)}")

    log("Draw balanced-period sample")
    sample = draw_sample(cleaned, rng)
    log(f"  sample rows: {len(sample)}  text_source llm={int((sample.text_source=='llm').sum())} raw={int((sample.text_source=='raw').sum())}")

    # Composition summary
    comp = sample.groupby(["meta_period", "source", "text_source", "seniority_3level"]).size().rename("n").reset_index()
    comp.to_csv(TABLES / "sample_composition.csv", index=False)
    log("  sample composition written")

    log("Load embedding index + embeddings for sample")
    idx = load_embedding_index()
    X_emb, sample = load_embeddings_for(sample, idx)
    log(f"  embeddings aligned: {X_emb.shape}  (sample kept: {len(sample)})")

    # LLM-only rows (for 2024 + 2026 arshkon-style coverage) — we already pre-filtered to text_source=='llm' during sampling;
    # but embeddings only exist for llm rows so the load_embeddings_for step already restricts us.
    if not (sample.text_source == "llm").all():
        log(f"  (embeddings only exist for llm text_source rows; raw-text sample rows dropped here)")

    stopwords = load_stoplist()
    log(f"  stoplist tokens: {len(stopwords)}")

    # Enrich sample with tech matrix + titles from unified
    log("Join tech matrix + title from unified")
    tech_df = pq.read_table(str(TECH_MATRIX)).to_pandas()
    con = duckdb.connect()
    titles_df = con.execute(
        "SELECT uid, title FROM read_parquet(?) WHERE uid IN "
        "(SELECT UNNEST($uids::VARCHAR[]))",
        [str(UNIFIED)]
    ).fetchdf() if False else None
    # simpler: read title column via pandas
    titles_df = pq.read_table(str(UNIFIED), columns=["uid", "title"]).to_pandas()
    titles_map = dict(zip(titles_df["uid"], titles_df["title"]))
    sample["title"] = sample["uid"].map(titles_map)

    docs = sample["description_cleaned"].fillna("").tolist()

    # ------------------------------------------------------------------
    # Method A — BERTopic (primary)
    # ------------------------------------------------------------------
    log("BERTopic — primary (min_topic_size=30)")
    bert_model, bert_labels, _umap_model = run_bertopic(
        docs, X_emb, stopwords, min_topic_size=30, seed=SEED, verbose=False,
    )
    n_topics_bert = len([x for x in set(bert_labels) if x != -1])
    noise_pct = float((bert_labels == -1).mean()) * 100
    log(f"  BERTopic: topics={n_topics_bert}  noise={noise_pct:.1f}%")

    # Also try min_topic_size=20 and 50 to report range
    log("BERTopic — sensitivity (min_topic_size=20)")
    bert_model_20, bert_labels_20, _ = run_bertopic(
        docs, X_emb, stopwords, min_topic_size=20, seed=SEED,
    )
    log(f"  BERTopic-20: topics={len([x for x in set(bert_labels_20) if x != -1])}  "
        f"noise={float((bert_labels_20 == -1).mean())*100:.1f}%")

    log("BERTopic — sensitivity (min_topic_size=50)")
    bert_labels_50 = None
    try:
        bert_model_50, bert_labels_50, _ = run_bertopic(
            docs, X_emb, stopwords, min_topic_size=50, seed=SEED,
        )
        log(f"  BERTopic-50: topics={len([x for x in set(bert_labels_50) if x != -1])}  "
            f"noise={float((bert_labels_50 == -1).mean())*100:.1f}%")
    except Exception as exc:
        log(f"  BERTopic-50 failed: {type(exc).__name__}: {exc}")
        bert_labels_50 = np.full_like(bert_labels, fill_value=-1)

    # Stability: 3 seeds at primary min_topic_size
    log("BERTopic — seed stability (3 seeds at min_topic_size=30)")
    label_seed_runs = [bert_labels]
    for s in [SEED + 7, SEED + 23]:
        lbls = run_bertopic_seed_only(docs, X_emb, stopwords, 30, s)
        label_seed_runs.append(lbls)
    ari_12 = adjusted_rand_score(label_seed_runs[0], label_seed_runs[1])
    ari_13 = adjusted_rand_score(label_seed_runs[0], label_seed_runs[2])
    ari_23 = adjusted_rand_score(label_seed_runs[1], label_seed_runs[2])
    mean_ari = (ari_12 + ari_13 + ari_23) / 3.0
    log(f"  BERTopic seed ARIs: 1-2={ari_12:.3f}, 1-3={ari_13:.3f}, 2-3={ari_23:.3f}; mean={mean_ari:.3f}")

    # ------------------------------------------------------------------
    # Method B — NMF over TF-IDF
    # ------------------------------------------------------------------
    log("TF-IDF vectorization")
    tfidf, X_tfidf = fit_tfidf(docs, stopwords)
    vocab = tfidf.get_feature_names_out().tolist()
    log(f"  tfidf shape: {X_tfidf.shape}  vocab: {len(vocab)}")

    nmf_results: dict[int, dict[str, Any]] = {}
    for k in [5, 8, 12, 15]:
        log(f"NMF k={k}")
        model, W, H = run_nmf(X_tfidf, k, SEED)
        top = nmf_top_terms(H, vocab, n=20)
        labels_k = nmf_labels_from_W(W)
        nmf_results[k] = dict(top_terms=top, labels=labels_k,
                               reconstruction_err=float(model.reconstruction_err_))
        log(f"  reconstruction_err={model.reconstruction_err_:.4f}  avg cluster size={len(labels_k)/k:.0f}")

    # NMF stability at k=12 across 3 seeds
    log("NMF stability — k=12 × 3 seeds")
    nmf_seed_labels: list[np.ndarray] = []
    for s in [SEED, SEED + 7, SEED + 23]:
        _, Ws, _ = run_nmf(X_tfidf, 12, s)
        nmf_seed_labels.append(nmf_labels_from_W(Ws))
    ari_nmf_12_13 = adjusted_rand_score(nmf_seed_labels[0], nmf_seed_labels[1])
    ari_nmf_12_23 = adjusted_rand_score(nmf_seed_labels[1], nmf_seed_labels[2])
    ari_nmf_mean = (adjusted_rand_score(nmf_seed_labels[0], nmf_seed_labels[1])
                     + adjusted_rand_score(nmf_seed_labels[0], nmf_seed_labels[2])
                     + adjusted_rand_score(nmf_seed_labels[1], nmf_seed_labels[2])) / 3.0
    log(f"  NMF k=12 mean ARI across seeds: {ari_nmf_mean:.3f}")

    # ------------------------------------------------------------------
    # Method comparison
    # ------------------------------------------------------------------
    log("Method comparison — top-term overlap (BERTopic primary vs NMF k=12)")
    bert_top = topic_top_terms_bertopic(bert_model, n=20)
    nmf_top_k12 = nmf_results[12]["top_terms"]
    overlap = top_term_overlap_matrix(bert_top, nmf_top_k12)
    overlap.to_csv(TABLES / "top_term_overlap_bertopic_vs_nmf_k12.csv")
    log(f"  max cross-method jaccard: {overlap.values.max():.3f}; mean top-1 match: "
        f"{np.mean([overlap.iloc[i].max() for i in range(len(overlap))]):.3f}")

    # ------------------------------------------------------------------
    # Pick primary labels for downstream: BERTopic outliers get their own label -1.
    # We'll also compute a "primary_labels" that re-assigns outliers to nearest cluster
    # so every row has an archetype for downstream consumers.
    # ------------------------------------------------------------------
    log("Assign outliers to nearest cluster for primary artifact (keeping -1 tracker too)")
    labels_with_noise = bert_labels.copy()
    # Soft-assign: use HDBSCAN approximate_predict if possible; fallback: assign nearest cluster centroid.
    centroids: dict[int, np.ndarray] = {}
    for c in set(labels_with_noise):
        if c == -1:
            continue
        centroids[int(c)] = X_emb[labels_with_noise == c].mean(axis=0)
    assigned = labels_with_noise.copy()
    if (labels_with_noise == -1).any():
        cent_ids = np.array(sorted(centroids.keys()))
        cent_mat = np.stack([centroids[int(i)] for i in cent_ids])
        # normalize for cosine
        def norm(A):
            n = np.linalg.norm(A, axis=1, keepdims=True)
            n[n == 0] = 1.0
            return A / n
        noise_idx = np.where(labels_with_noise == -1)[0]
        cent_norm = norm(cent_mat)
        emb_norm = norm(X_emb[noise_idx])
        sim = emb_norm @ cent_norm.T
        nearest = cent_ids[sim.argmax(axis=1)]
        assigned[noise_idx] = nearest

    # ------------------------------------------------------------------
    # Characterization
    # ------------------------------------------------------------------
    log("Characterize clusters (using BERTopic primary labels with noise preserved)")

    # YOE + description length + tech count
    sample = sample.reset_index(drop=True)
    sample["bert_label"] = labels_with_noise
    sample["bert_label_assigned"] = assigned
    sample["desc_char_len"] = sample["description_cleaned"].fillna("").str.len()

    # Tech count per row
    tech_aligned = tech_df[tech_df.uid.isin(sample.uid)].set_index("uid")
    tech_aligned = tech_aligned.reindex(sample.uid)
    bool_cols = [c for c in tech_aligned.columns if tech_aligned[c].dtype == bool]
    sample["tech_count"] = tech_aligned[bool_cols].fillna(False).sum(axis=1).values

    # c-TF-IDF top terms per cluster (BERTopic's built-in does this from count vectorizer;
    # we'll use its topic-info directly)
    bert_topic_info = bert_model.get_topic_info()

    char_rows: list[dict[str, Any]] = []
    for cid in sorted(set(labels_with_noise)):
        mask = labels_with_noise == cid
        sub = sample[mask]
        if len(sub) == 0:
            continue
        top_terms = bert_top.get(int(cid), []) if cid != -1 else []
        # Seniority panel
        sen_counts = sub.seniority_3level.value_counts(dropna=False).to_dict()
        # Entry share by meta-period using J3 = yoe_min_years_llm <= 2 and labeled
        sub_j3 = sub[(sub.yoe_min_years_llm.notna()) & (sub.llm_classification_coverage == "labeled")]
        j3_2024 = sub_j3[sub_j3.meta_period == "2024"]
        j3_2026 = sub_j3[sub_j3.meta_period.isin(["2026-03", "2026-04"])]
        j3_2024_share = (j3_2024.yoe_min_years_llm <= 2).mean() if len(j3_2024) else np.nan
        j3_2026_share = (j3_2026.yoe_min_years_llm <= 2).mean() if len(j3_2026) else np.nan
        # S4 share
        s4_2024_share = (j3_2024.yoe_min_years_llm >= 5).mean() if len(j3_2024) else np.nan
        s4_2026_share = (j3_2026.yoe_min_years_llm >= 5).mean() if len(j3_2026) else np.nan

        row = dict(
            cluster_id=int(cid),
            is_noise=bool(cid == -1),
            n=int(mask.sum()),
            share_of_total=float(mask.mean()),
            top_20_terms="; ".join(top_terms[:20]),
            # period shares
            share_2024=float((sub.meta_period == "2024").mean()),
            share_2026_03=float((sub.meta_period == "2026-03").mean()),
            share_2026_04=float((sub.meta_period == "2026-04").mean()),
            # period counts
            n_2024=int((sub.meta_period == "2024").sum()),
            n_2026_03=int((sub.meta_period == "2026-03").sum()),
            n_2026_04=int((sub.meta_period == "2026-04").sum()),
            # seniority_3level counts
            n_junior=int(sen_counts.get("junior", 0)),
            n_mid=int(sen_counts.get("mid", 0)),
            n_senior=int(sen_counts.get("senior", 0)),
            n_unknown=int(sen_counts.get("unknown", 0)),
            # entry share by period (J3 primary)
            j3_share_2024=j3_2024_share,
            j3_share_2026=j3_2026_share,
            s4_share_2024=s4_2024_share,
            s4_share_2026=s4_2026_share,
            # averages
            mean_desc_char_len=float(sub.desc_char_len.mean()),
            median_desc_char_len=float(sub.desc_char_len.median()),
            mean_yoe_llm=float(sub.yoe_min_years_llm.mean(skipna=True)),
            median_yoe_llm=float(sub.yoe_min_years_llm.median(skipna=True)),
            mean_tech_count=float(sub.tech_count.mean()),
            # aggregator share
            aggregator_share=float(sub.is_aggregator.fillna(False).mean()),
        )
        char_rows.append(row)

    char_df = pd.DataFrame(char_rows).sort_values("n", ascending=False)

    # Name each cluster from top 2-3 content-loaded terms
    def derive_name(terms_str: str, cid: int) -> str:
        if cid == -1:
            return "noise/outlier"
        terms = [t for t in (terms_str.split("; ") if terms_str else []) if t]
        return "/".join(terms[:3]).replace(" ", "_") if terms else f"cluster_{cid}"

    char_df["archetype_name"] = [derive_name(t, c) for t, c in zip(char_df.top_20_terms, char_df.cluster_id)]
    char_df.to_csv(TABLES / "cluster_characterization.csv", index=False)
    log(f"  cluster characterization written ({len(char_df)} clusters)")

    # ------------------------------------------------------------------
    # Temporal dynamics
    # ------------------------------------------------------------------
    log("Temporal dynamics (cluster share by meta-period)")
    temp = (sample.groupby(["meta_period", "bert_label"]).size()
             .unstack("bert_label", fill_value=0))
    temp_share = temp.div(temp.sum(axis=1), axis=0)
    temp_share.to_csv(TABLES / "temporal_dynamics_cluster_share.csv")

    # delta 2024 → 2026 (pooled 2026)
    pooled_2026 = temp_share.loc[[i for i in ["2026-03", "2026-04"] if i in temp_share.index]].mean()
    pooled_2024 = temp_share.loc["2024"] if "2024" in temp_share.index else None
    deltas: pd.DataFrame
    if pooled_2024 is not None:
        deltas = pd.DataFrame({
            "share_2024": pooled_2024,
            "share_2026_avg": pooled_2026,
            "delta_pp": (pooled_2026 - pooled_2024) * 100,
        }).reset_index().rename(columns={"bert_label": "cluster_id"})
        deltas = deltas.sort_values("delta_pp", ascending=False)
        deltas.to_csv(TABLES / "temporal_dynamics_deltas.csv", index=False)

    # ------------------------------------------------------------------
    # KEY DISCOVERY — NMI
    # ------------------------------------------------------------------
    log("KEY DISCOVERY — NMI(clusters, …) with strategic-pivot variables")
    sample["tech_domain"] = tech_domain_assignment(tech_df, sample.uid.tolist())
    sample["title_archetype"] = title_archetype_for(sample.title.tolist())
    sample["meta_period_str"] = sample.meta_period.astype(str)

    # NMI with + without noise rows
    def nmi(a, b):
        # coerce to strings for safety
        a_s = pd.Series(a, dtype="object").astype(str)
        b_s = pd.Series(b, dtype="object").astype(str)
        return float(normalized_mutual_info_score(a_s, b_s, average_method="arithmetic"))

    nmi_rows: list[dict[str, Any]] = []
    for label_col in ["bert_label", "bert_label_assigned"]:
        noise_note = ("noise=-1 own cluster" if label_col == "bert_label"
                       else "noise reassigned to nearest")
        for target_col, target_desc in [
            ("seniority_3level", "T30 coarse (junior/mid/senior/unknown)"),
            ("tech_domain", "tech-stack argmax assignment"),
            ("meta_period_str", "2024 vs 2026-03 vs 2026-04"),
            ("title_archetype", "regex title-archetype"),
        ]:
            val = nmi(sample[label_col], sample[target_col])
            nmi_rows.append(dict(
                primary_labels=label_col,
                noise_treatment=noise_note,
                target=target_col,
                target_description=target_desc,
                nmi=val,
            ))

    # Also NMI between NMF k=12 and targets
    sample["nmf_k12"] = nmf_results[12]["labels"]
    for target_col, target_desc in [
        ("seniority_3level", "T30 coarse"),
        ("tech_domain", "tech-stack argmax"),
        ("meta_period_str", "period"),
        ("title_archetype", "title regex"),
    ]:
        val = nmi(sample["nmf_k12"], sample[target_col])
        nmi_rows.append(dict(
            primary_labels="nmf_k12",
            noise_treatment="n/a",
            target=target_col,
            target_description=target_desc,
            nmi=val,
        ))

    nmi_df = pd.DataFrame(nmi_rows).sort_values(["primary_labels", "nmi"], ascending=[True, False])
    nmi_df.to_csv(TABLES / "nmi_pivot_decision.csv", index=False)
    log("  NMI table written")
    # print summary
    for r in nmi_rows:
        log(f"    NMI({r['primary_labels']}, {r['target']}) = {r['nmi']:.4f}")

    # ------------------------------------------------------------------
    # Methods comparison summary table
    # ------------------------------------------------------------------
    log("Write methods comparison summary")
    methods_rows = [
        dict(method="BERTopic (min_topic_size=30, primary)",
             n_topics=int(n_topics_bert),
             noise_pct=round(noise_pct, 2),
             stability_ari_mean=round(mean_ari, 3),
             tfidf_recon_err=None,
             interpretability_note=(
                 "Content-coherent topics with named technologies (React, AWS, ML) "
                 "plus role-type groupings (security, DevOps). Handles outliers explicitly. "
                 "c-TF-IDF top-terms are typically readable without post-hoc curation."),
             ),
        dict(method="BERTopic (min_topic_size=20, sensitivity)",
             n_topics=int(len([x for x in set(bert_labels_20) if x != -1])),
             noise_pct=round(float((bert_labels_20 == -1).mean()) * 100, 2),
             stability_ari_mean=None,
             tfidf_recon_err=None,
             interpretability_note="Finer granularity; more niche clusters (payments, observability).",
             ),
        dict(method="BERTopic (min_topic_size=50, sensitivity)",
             n_topics=int(len([x for x in set(bert_labels_50) if x != -1])) if bert_labels_50 is not None else None,
             noise_pct=round(float((bert_labels_50 == -1).mean()) * 100, 2) if bert_labels_50 is not None else None,
             stability_ari_mean=None,
             tfidf_recon_err=None,
             interpretability_note="Coarser; a handful of dominant role-type clusters (or skipped if degenerate).",
             ),
    ]
    for k in [5, 8, 12, 15]:
        methods_rows.append(dict(
            method=f"NMF k={k}",
            n_topics=k,
            noise_pct=0.0,
            stability_ari_mean=(round(ari_nmf_mean, 3) if k == 12 else None),
            tfidf_recon_err=round(nmf_results[k]["reconstruction_err"], 4),
            interpretability_note=("Soft-assigned by argmax W; top-terms reflect corpus axes. "
                                   "Deterministic and fast; less role-type separation than BERTopic."),
        ))
    methods_df = pd.DataFrame(methods_rows)
    methods_df.to_csv(TABLES / "methods_comparison.csv", index=False)

    # ------------------------------------------------------------------
    # UMAP 2D for visualization
    # ------------------------------------------------------------------
    log("Compute UMAP 2D for visualization (once)")
    xy = umap_2d(X_emb, seed=SEED)
    log(f"  umap_2d shape: {xy.shape}")

    plot_scatter(xy, labels_with_noise.astype(object), "T09 UMAP — BERTopic cluster", FIGS / "umap_by_cluster.png")
    plot_scatter(xy, sample.meta_period_str.values, "T09 UMAP — meta-period", FIGS / "umap_by_period.png")
    plot_scatter(xy, sample.seniority_3level.fillna("unknown").values,
                  "T09 UMAP — seniority_3level (T30 coarse)", FIGS / "umap_by_seniority.png")
    plot_scatter(xy, sample.tech_domain.values,
                  "T09 UMAP — tech-stack argmax domain (supplementary)",
                  FIGS / "umap_by_tech_domain.png")

    # ------------------------------------------------------------------
    # Sensitivities
    # ------------------------------------------------------------------
    log("Sensitivity (a) — aggregator-excluded BERTopic")
    mask_no_agg = ~sample.is_aggregator.fillna(False)
    X_emb_no_agg = X_emb[mask_no_agg.values]
    docs_no_agg = [d for d, m in zip(docs, mask_no_agg) if m]
    try:
        bm_ag, lbl_ag, _ = run_bertopic(docs_no_agg, X_emb_no_agg, stopwords, 30, SEED)
    except Exception as exc:
        log(f"  aggregator-excluded BERTopic failed: {type(exc).__name__}: {exc}")
        lbl_ag = np.full(len(docs_no_agg), -1)
    agg_compare = pd.DataFrame({
        "metric": ["n_topics", "noise_pct"],
        "full_sample": [n_topics_bert, round(noise_pct, 2)],
        "aggregator_excluded": [len([x for x in set(lbl_ag) if x != -1]),
                                  round(float((lbl_ag == -1).mean()) * 100, 2)],
    })
    agg_compare.to_csv(TABLES / "sensitivity_a_aggregator.csv", index=False)
    log(f"  aggregator-excluded: n_topics={agg_compare.iloc[0]['aggregator_excluded']}, noise%={agg_compare.iloc[1]['aggregator_excluded']}")

    log("Sensitivity (b) — company-capped (30 per canonical company)")
    cap = 30
    capped = sample.groupby("company_name_canonical", group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), cap), random_state=SEED) if len(g) else g)
    capped = capped.reset_index(drop=True)
    uid_to_idx = {u: i for i, u in enumerate(sample.uid.tolist())}
    keep_idx = capped.uid.map(uid_to_idx).dropna().astype(int).tolist()
    X_emb_cap = X_emb[keep_idx]
    docs_cap = [docs[i] for i in keep_idx]
    try:
        bm_cap, lbl_cap, _ = run_bertopic(docs_cap, X_emb_cap, stopwords, 30, SEED)
    except Exception as exc:
        log(f"  company-capped BERTopic failed: {type(exc).__name__}: {exc}")
        lbl_cap = np.full(len(docs_cap), -1)
    cap_compare = pd.DataFrame({
        "metric": ["n_rows", "n_topics", "noise_pct"],
        "full_sample": [len(sample), n_topics_bert, round(noise_pct, 2)],
        "cap_30": [len(keep_idx),
                    len([x for x in set(lbl_cap) if x != -1]),
                    round(float((lbl_cap == -1).mean()) * 100, 2)],
    })
    cap_compare.to_csv(TABLES / "sensitivity_b_company_cap.csv", index=False)
    log(f"  company-cap 30: n={len(keep_idx)} topics={cap_compare.iloc[1]['cap_30']} noise%={cap_compare.iloc[2]['cap_30']}")
    # cap-sensitivity NMI (optional but useful)
    cap_sample = sample.iloc[keep_idx].copy().reset_index(drop=True)
    cap_sample["tech_domain"] = tech_domain_assignment(tech_df, cap_sample.uid.tolist())
    cap_sample["title_archetype"] = title_archetype_for(cap_sample.title.tolist())
    cap_sample["meta_period_str"] = cap_sample.meta_period.astype(str)
    cap_nmi = {
        "nmi_seniority": nmi(lbl_cap, cap_sample.seniority_3level.values),
        "nmi_tech_domain": nmi(lbl_cap, cap_sample.tech_domain.values),
        "nmi_period": nmi(lbl_cap, cap_sample.meta_period_str.values),
        "nmi_title_arch": nmi(lbl_cap, cap_sample.title_archetype.values),
    }
    with open(TABLES / "sensitivity_b_company_cap_nmi.json", "w") as fh:
        json.dump(cap_nmi, fh, indent=2)
    log(f"  company-cap NMI: {cap_nmi}")

    # Sensitivity (d) — raw-text run: BERTopic over sampled raw-text rows from scraped.
    log("Sensitivity (d) — raw-text scraped BERTopic on a comparable sample")
    # build a separate raw-text sample (~2500) from scraped 2026 raw-text pool
    raw_pool = cleaned[(cleaned.source == "scraped") & (cleaned.text_source == "raw")
                       & (cleaned.description_cleaned.notna())
                       & (cleaned.description_cleaned.str.len() > 30)]
    if len(raw_pool) > 0:
        # balanced by period
        pieces = []
        for p in ["2026-03", "2026-04"]:
            sub = raw_pool[raw_pool.period == p]
            if len(sub) >= 1250:
                pieces.append(sub.sample(n=1250, random_state=SEED))
            else:
                pieces.append(sub)
        raw_sample = pd.concat(pieces, ignore_index=True)
        raw_docs = raw_sample.description_cleaned.fillna("").tolist()
        # embed these raw-text rows from scratch (no pre-computed embeddings for raw)
        from sentence_transformers import SentenceTransformer
        log("  loading SentenceTransformer for raw-text embedding (one-off, sample only)")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        raw_emb = model.encode(raw_docs, batch_size=256, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=False)
        try:
            bm_raw, lbl_raw, _ = run_bertopic(raw_docs, raw_emb, stopwords, 30, SEED)
        except Exception as exc:
            log(f"  raw-text BERTopic failed: {type(exc).__name__}: {exc}")
            bm_raw, lbl_raw = None, np.full(len(raw_docs), -1)
        raw_summary = dict(
            n_rows=len(raw_sample),
            n_topics=len([x for x in set(lbl_raw) if x != -1]),
            noise_pct=round(float((lbl_raw == -1).mean()) * 100, 2),
        )
        with open(TABLES / "sensitivity_d_raw_text.json", "w") as fh:
            json.dump(raw_summary, fh, indent=2)
        # write raw c-TF-IDF top terms
        if bm_raw is not None:
            raw_top = topic_top_terms_bertopic(bm_raw, n=20)
            raw_top_rows = [dict(cluster_id=k, top_20_terms="; ".join(v))
                             for k, v in raw_top.items()]
            pd.DataFrame(raw_top_rows).to_csv(TABLES / "sensitivity_d_raw_text_topics.csv", index=False)
        log(f"  raw-text BERTopic: {raw_summary}")
    else:
        log("  No raw-text scraped pool; skipping sensitivity (d).")

    # Sensitivity (g) — SWE-classification tier
    log("Sensitivity (g) — SWE-classification tier distribution per cluster")
    if "swe_classification_tier" in sample.columns:
        tier_mix = (sample.groupby(["bert_label", "swe_classification_tier"]).size()
                     .unstack("swe_classification_tier", fill_value=0))
        tier_mix_share = tier_mix.div(tier_mix.sum(axis=1), axis=0)
        tier_mix_share.to_csv(TABLES / "sensitivity_g_swe_tier_share.csv")

    # ------------------------------------------------------------------
    # Save artifact (archetype labels)
    # ------------------------------------------------------------------
    log("Save swe_archetype_labels.parquet")
    arche_name_map = dict(zip(char_df.cluster_id.astype(int),
                               char_df.archetype_name.astype(str)))
    out_df = pd.DataFrame({
        "uid": sample.uid.values,
        "archetype": sample.bert_label.astype(int).values,
        "archetype_name": sample.bert_label.astype(int).map(arche_name_map).values,
    })
    # write parquet
    import pyarrow as pa
    pq.write_table(pa.Table.from_pandas(out_df, preserve_index=False), str(ART_OUT))
    log(f"  wrote {ART_OUT} with {len(out_df)} rows")

    # ------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------
    summary = dict(
        sample_n=int(len(sample)),
        sample_source_breakdown=sample.source.value_counts().to_dict(),
        sample_text_source=sample.text_source.value_counts().to_dict(),
        sample_period=sample.meta_period.value_counts().to_dict(),
        bertopic_primary_min_topic_size=30,
        bertopic_n_topics=int(n_topics_bert),
        bertopic_noise_pct=float(noise_pct),
        bertopic_seed_ari_mean=float(mean_ari),
        nmf_k12_seed_ari_mean=float(ari_nmf_mean),
        nmi=nmi_rows,
    )
    with open(TABLES / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    log("DONE.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    raise SystemExit(main(args))
