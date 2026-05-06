"""
Reusable BERTopic pipeline construction and metrics for Stage 1.

`build_topic_model(...)` constructs a BERTopic with pre-registered UMAP /
HDBSCAN / c-TF-IDF / KeyBERTInspired+MMR settings; the only knobs the caller
chooses are `min_cluster_size` and `seed`. `fit_topic_model(...)` runs
`fit_transform` over Sample A and returns the model, the per-row topic
assignment, and the embeddings array (so downstream metrics can reuse it
without a second cache load).

Cluster centroids in 3072-d posting space are computed once per fit — never
re-derived inside metric functions, since centroids are an expensive op on
~58 k × 3072 vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import duckdb
import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from umap import UMAP

from figures.bertopic import config
from figures.bertopic.embedding_cache import load_cache


_KEYBERT_AUX_MODEL = "all-MiniLM-L6-v2"
_AUX_BACKEND = None  # `bertopic.backend.SentenceTransformerBackend`


def _aux_embedding_backend():
    """Cached BERTopic-wrapped embedding backend.

    BERTopic's KeyBERTInspired representation calls `embed_documents` on
    `topic_model.embedding_model`, which is a backend wrapper, not a bare
    SentenceTransformer. Use `select_backend` to get the right type so that
    re-attaching after `BERTopic.load(save_embedding_model=False)` works.
    """
    global _AUX_BACKEND
    if _AUX_BACKEND is None:
        from bertopic.backend._utils import select_backend
        _AUX_BACKEND = select_backend(SentenceTransformer(_KEYBERT_AUX_MODEL))
    return _AUX_BACKEND


def load_topic_model(path) -> BERTopic:
    """`BERTopic.load` + re-attach the auxiliary embedding backend.

    We save with `save_embedding_model=False` to keep `.bertopic` files small.
    KeyBERTInspired needs a backend with `embed_documents` on the loaded
    instance during representation refinement (fires inside `reduce_topics`).
    """
    model = BERTopic.load(str(path))
    model.embedding_model = _aux_embedding_backend()
    return model


def make_permissive_vectorizer() -> CountVectorizer:
    """Vectorizer used after `reduce_topics` to small K.

    BERTopic's c-TF-IDF treats each topic as one document, so min_df=10
    (intended for the raw 58 k document corpus) requires a term to appear in
    ≥ 10 of K topics — impossible for K ≤ 10. We swap to a permissive
    `min_df=2, max_df=1.0` for any reduce_topics path. This is a documented
    deviation from §4.2 and is recorded in `prereg_log.md`.
    """
    stops = list(ENGLISH_STOP_WORDS) + list(config.CUSTOM_STOPWORDS)
    return CountVectorizer(
        ngram_range=config.NGRAM_RANGE,
        min_df=2,
        max_df=1.0,
        stop_words=stops,
        token_pattern=config.TOKEN_PATTERN,
    )


def load_topic_model_for_reduce(path) -> BERTopic:
    """`load_topic_model` plus a permissive vectorizer for reduce_topics."""
    model = load_topic_model(path)
    model.vectorizer_model = make_permissive_vectorizer()
    return model


@dataclass(frozen=True)
class FitResult:
    topic_model: BERTopic
    topics: np.ndarray  # length n_docs, -1 for noise
    embeddings: np.ndarray  # (n_docs, 3072)
    uids: tuple[str, ...]
    docs: tuple[str, ...]


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
        # `random_state` already forces UMAP to single-thread; setting
        # `n_jobs` explicitly silences the warning umap-learn emits.
        n_jobs=1,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(1, round(0.5 * min_cluster_size)),
        metric=config.HDBSCAN_METRIC,
        cluster_selection_method=config.HDBSCAN_CLUSTER_SELECTION_METHOD,
        prediction_data=config.HDBSCAN_PREDICTION_DATA,
        # Pin to single-thread; multi-thread HDBSCAN can produce non-byte-
        # identical labels on repeat fits (caught by S1.5 determinism check).
        core_dist_n_jobs=1,
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
        embedding_model=_aux_embedding_backend(),
        min_topic_size=min_cluster_size,
        calculate_probabilities=config.CALCULATE_PROBABILITIES,
        verbose=False,
    )


@lru_cache(maxsize=1)
def _load_cache_cached() -> tuple[np.ndarray, dict[str, int]]:
    return load_cache()


def load_sample(*, sample_path) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    """Return (uids, docs, embeddings) for the given sample parquet."""
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")
    rows = con.execute(
        f"SELECT uid, description_core_llm FROM '{sample_path}' ORDER BY uid"
    ).fetchall()
    uids = tuple(r[0] for r in rows)
    docs = tuple(r[1] for r in rows)

    matrix, key_to_row = _load_cache_cached()
    indices = np.array([key_to_row[u] for u in uids], dtype=np.int64)
    embeddings = matrix[indices]
    return uids, docs, embeddings


def fit_topic_model(
    *,
    min_cluster_size: int,
    seed: int = config.SEED_PRIMARY,
    sample_path=config.SAMPLE_A_PATH,
    uids: tuple[str, ...] | None = None,
    docs: tuple[str, ...] | None = None,
    embeddings: np.ndarray | None = None,
    permissive_vectorizer: bool = False,
) -> FitResult:
    """Fit BERTopic. Set `permissive_vectorizer=True` for small subsets where
    the §4.2 strict vectorizer (min_df=10, max_df=0.4) blows up because the
    cluster-level corpus has < 10 topics — which can happen on subsets of a
    few thousand rows.
    """
    if uids is None or docs is None or embeddings is None:
        uids, docs, embeddings = load_sample(sample_path=sample_path)
    topic_model = build_topic_model(min_cluster_size=min_cluster_size, seed=seed)
    if permissive_vectorizer:
        topic_model.vectorizer_model = make_permissive_vectorizer()
    topics, _ = topic_model.fit_transform(list(docs), embeddings)
    return FitResult(
        topic_model=topic_model,
        topics=np.asarray(topics, dtype=np.int64),
        embeddings=embeddings,
        uids=uids,
        docs=docs,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def noise_rate(topics: np.ndarray) -> float:
    return float(np.mean(topics == -1))


def n_clusters(topics: np.ndarray) -> int:
    return int(len({int(t) for t in topics if t != -1}))


def cluster_centroids(
    topics: np.ndarray, embeddings: np.ndarray
) -> dict[int, np.ndarray]:
    """Return {cluster_id: 3072-d L2-normalized centroid} (excludes -1)."""
    out: dict[int, np.ndarray] = {}
    for cid in {int(t) for t in topics if t != -1}:
        mask = topics == cid
        c = embeddings[mask].mean(axis=0)
        n = float(np.linalg.norm(c))
        if n > 0:
            c = c / n
        out[cid] = c.astype(np.float32)
    return out


def mean_inter_cluster_cosine(centroids: dict[int, np.ndarray]) -> float:
    if len(centroids) < 2:
        return float("nan")
    keys = list(centroids)
    matrix = np.stack([centroids[k] for k in keys])
    sims = matrix @ matrix.T
    np.fill_diagonal(sims, np.nan)
    return float(np.nanmean(sims))


def mean_intra_cluster_spread(
    topics: np.ndarray, embeddings: np.ndarray, centroids: dict[int, np.ndarray]
) -> float:
    spreads: list[float] = []
    for cid, centroid in centroids.items():
        members = embeddings[topics == cid]
        if len(members) < 2:
            continue
        sims = members @ centroid
        spreads.append(float(1.0 - sims.mean()))
    return float(np.mean(spreads)) if spreads else float("nan")


def largest_cluster_share(topics: np.ndarray) -> tuple[int, float]:
    """Return (largest_cluster_id, share) excluding noise."""
    valid = topics[topics != -1]
    if len(valid) == 0:
        return -1, 0.0
    ids, counts = np.unique(valid, return_counts=True)
    idx = int(np.argmax(counts))
    return int(ids[idx]), float(counts[idx] / len(topics))


def hungarian_centroid_alignment(
    centroids_a: dict[int, np.ndarray],
    centroids_b: dict[int, np.ndarray],
) -> float:
    """Mean cosine of the best 1-to-1 matching between the two centroid sets."""
    from scipy.optimize import linear_sum_assignment
    if not centroids_a or not centroids_b:
        return float("nan")
    keys_a = list(centroids_a)
    keys_b = list(centroids_b)
    mat_a = np.stack([centroids_a[k] for k in keys_a])
    mat_b = np.stack([centroids_b[k] for k in keys_b])
    sim = mat_a @ mat_b.T
    cost = -sim
    row_idx, col_idx = linear_sum_assignment(cost)
    return float(sim[row_idx, col_idx].mean())


def pairwise_metrics(topics_a: np.ndarray, topics_b: np.ndarray) -> dict[str, float]:
    """ARI / NMI between two topic assignments on the same documents."""
    return {
        "ari": float(adjusted_rand_score(topics_a, topics_b)),
        "nmi": float(normalized_mutual_info_score(topics_a, topics_b)),
    }


def reduce_to_k(fit: FitResult, k: int) -> np.ndarray:
    """Reduce topics to k and return the new per-row labels."""
    fit.topic_model.reduce_topics(list(fit.docs), nr_topics=k)
    new_topics = fit.topic_model.topics_
    return np.asarray(new_topics, dtype=np.int64)
