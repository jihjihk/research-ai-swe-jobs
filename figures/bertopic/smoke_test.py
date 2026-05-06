"""
Stage-0 smoke test (§13.1 S0.5).

Runs the full pipeline (UMAP → HDBSCAN → c-TF-IDF → one LLM naming call) on a
5 % stratified slice of Sample A. Catches BERTopic version drift, OpenAI auth
failures, slow steps, and obvious mega-cluster red flags before the
expensive Stage 1 sweeps begin.
"""

from __future__ import annotations

import sys

import duckdb
import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from umap import UMAP

from figures.bertopic import config
from figures.bertopic.embedding_cache import load_cache
from figures.bertopic.stage1.naming import propose_label


# Auxiliary embedding model used by KeyBERTInspired for top-word reranking.
# Clustering itself runs on pre-computed OpenAI 3072-d vectors passed to
# `fit_transform`; this small model only embeds candidate words / topics.
_KEYBERT_AUX_MODEL = "all-MiniLM-L6-v2"


def stratified_smoke_sample() -> tuple[list[str], list[str]]:
    """Return (uids, docs) for a 5 % stratified slice of Sample A."""
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")
    rows = con.execute(f"""
        WITH ranked AS (
            SELECT uid, period, description_core_llm,
                   row_number() OVER (
                       PARTITION BY period ORDER BY hash(uid)
                   ) AS rn,
                   count(*) OVER (PARTITION BY period) AS n_period
            FROM '{config.SAMPLE_A_PATH}'
        )
        SELECT uid, description_core_llm
        FROM ranked
        WHERE rn <= ceil(n_period * {config.SMOKE_TEST_FRACTION})
    """).fetchall()
    return [r[0] for r in rows], [r[1] for r in rows]


def build_count_vectorizer() -> CountVectorizer:
    stops = list(ENGLISH_STOP_WORDS) + list(config.CUSTOM_STOPWORDS)
    return CountVectorizer(
        ngram_range=config.NGRAM_RANGE,
        min_df=config.COUNT_MIN_DF,
        max_df=config.COUNT_MAX_DF,
        stop_words=stops,
        token_pattern=config.TOKEN_PATTERN,
    )


def build_topic_model(
    *, min_cluster_size: int, seed: int = config.SEED_PRIMARY,
) -> BERTopic:
    umap_model = UMAP(
        n_neighbors=config.UMAP_N_NEIGHBORS,
        n_components=config.UMAP_N_COMPONENTS,
        min_dist=config.UMAP_MIN_DIST,
        metric=config.UMAP_METRIC,
        random_state=seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(1, round(0.5 * min_cluster_size)),
        metric=config.HDBSCAN_METRIC,
        cluster_selection_method=config.HDBSCAN_CLUSTER_SELECTION_METHOD,
        prediction_data=config.HDBSCAN_PREDICTION_DATA,
    )
    representation_model = [
        KeyBERTInspired(top_n_words=config.KEYBERT_TOP_N_WORDS),
        MaximalMarginalRelevance(diversity=config.MMR_DIVERSITY),
    ]
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=build_count_vectorizer(),
        representation_model=representation_model,
        embedding_model=SentenceTransformer(_KEYBERT_AUX_MODEL),
        min_topic_size=min_cluster_size,
        calculate_probabilities=config.CALCULATE_PROBABILITIES,
        verbose=False,
    )


def _embeddings_for(uids: list[str]) -> np.ndarray:
    matrix, key_to_row = load_cache()
    rows = np.array([key_to_row[u] for u in uids], dtype=np.int64)
    return matrix[rows]


def main() -> None:
    print("Smoke test (§13.1 S0.5):")

    uids, docs = stratified_smoke_sample()
    print(f"  - sampled {len(uids)} rows ({config.SMOKE_TEST_FRACTION:.0%} of A)")

    embeddings = _embeddings_for(uids)
    print(f"  - retrieved embeddings of shape {embeddings.shape}")

    # Use a smaller min_cluster_size for the smoke slice; the goal is to verify
    # plumbing, not to characterize at the headline mcs.
    smoke_mcs = max(10, round(config.MCS_INITIAL * config.SMOKE_TEST_FRACTION * 4))
    print(f"  - fitting BERTopic with min_cluster_size={smoke_mcs}…")
    topic_model = build_topic_model(min_cluster_size=smoke_mcs)
    topics, _ = topic_model.fit_transform(docs, embeddings)
    n_topics = len({t for t in topics if t != -1})
    noise_rate = sum(1 for t in topics if t == -1) / len(topics)
    print(f"    found {n_topics} topics, noise rate {noise_rate:.2%}")
    if n_topics < 2:
        raise RuntimeError(
            "smoke test produced < 2 non-noise topics; pipeline likely broken"
        )

    info = topic_model.get_topic_info()
    info = info[info.Topic != -1].head(1)
    if info.empty:
        raise RuntimeError("smoke test produced no non-noise topics")
    sample_topic_id = int(info.iloc[0]["Topic"])
    top_words = [w for w, _ in topic_model.get_topic(sample_topic_id)[:15]]
    rep_docs = topic_model.get_representative_docs(sample_topic_id) or docs[:1]
    exemplars = [(f"posting {i}", rep_docs[i][:200]) for i in range(min(3, len(rep_docs)))]
    print(f"  - top words for topic {sample_topic_id}: {top_words[:5]}…")

    print("  - calling LLM naming once…")
    label = propose_label(
        top_words=top_words,
        exemplars=exemplars,
        request_id="smoke",
    )
    print(f"    proposed label: {label}")

    print("Smoke test OK.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — boundary, want loud failure
        print(f"Smoke test FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
