"""
Stage 2 sub-agent T-method — §7.4 NMF baseline + §7.5 MiniLM cross-embedding.

Compares the BERTopic OpenAI-3072 headline-K (= 10) clustering on Sample A
against:

  1. NMF (sklearn) on a TF-IDF matrix of `description_core_llm` using the
     same vectorizer settings as the c-TF-IDF stage of BERTopic
     (ngram (1,3), `min_df=10`, `max_df=0.4`, custom stopwords).
     Hard-assign by argmax over component loadings.
  2. BERTopic refit on Sample A using `all-MiniLM-L6-v2` 384-d embeddings
     (the v3 prior's embedding) at the same headline (mcs, seed=42),
     reduced to the headline K.

Outputs:
  - `data/bertopic/method_comparison.parquet` — one row per comparison
    (NMF, MiniLM-BERTopic) with ARI / NMI / n_overlap / notes.
  - Cluster-level alignment (Hungarian over centroid cosine, then top-words
    side-by-side for the 5 most-agreed and 5 most-disagreed clusters).

Hash-anchored to Stage 1 freeze; verifies the five expected hashes before
running, fails loud on mismatch. No re-fitting of the OpenAI BERTopic model.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfTransformer,
)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from figures.bertopic import config  # noqa: E402
from figures.bertopic.stage1 import pipeline  # noqa: E402

EXPECTED_HASHES = {
    "model_hash": "d51f15e613f62b221139503bc84e6d3757689aac5e07979beb6ed3dbce509415",
    "sample_hash": "6719a0250fbfcb630dad117b409d441697d493b209b219e1c9d08b09acfeb265",
    "embeddings_cache_hash": "29d77bf9e24e6250d7b303a17fb22b80b9575a09a46d88c9dbd5d75c3b479b27",
    "assignments_hash": "a03bc515094050996338094f28851126b8c1f07f7f3b26d2f678f6cb6808ab82",
    "config_hash": "bef20ab2916ad72bd87aaefb0d18ba13644f9989ddd8e9bad4eac2b01a07bce8",
}

OUTPUT_PARQUET = config.BERTOPIC_DATA_DIR / "method_comparison.parquet"
CLUSTER_COMPARISON_PARQUET = config.BERTOPIC_DATA_DIR / "method_cluster_alignment.parquet"
MINILM_TOPICS_NPY = config.INTERMEDIATE_DIR / "t_method_minilm_reducedK10.npy"
MINILM_RAW_TOPICS_NPY = config.INTERMEDIATE_DIR / "t_method_minilm_raw_topics.npy"
MINILM_EMB_NPY = config.INTERMEDIATE_DIR / "t_method_minilm_embeddings.npy"
NMF_TOPICS_NPY = config.INTERMEDIATE_DIR / "t_method_nmf_topics.npy"


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes() -> None:
    actual = {
        "model_hash": _file_hash(config.RAW_FIT_PATH),
        "sample_hash": _file_hash(config.SAMPLE_A_PATH),
        "embeddings_cache_hash": _file_hash(config.EMBEDDINGS_CACHE_PATH),
        "assignments_hash": _file_hash(config.ASSIGNMENTS_PATH),
        "config_hash": _file_hash(Path(config.__file__)),
    }
    bad = [k for k, v in EXPECTED_HASHES.items() if actual[k] != v]
    if bad:
        for k in bad:
            print(f"HASH MISMATCH: {k}\n  expected={EXPECTED_HASHES[k]}\n  actual  ={actual[k]}")
        raise SystemExit("Hash verification failed — aborting.")
    print("Hash verification: all five hashes match.")


def load_sample_a_with_assignments() -> tuple[
    list[str], list[str], np.ndarray, np.ndarray
]:
    """Return (uids, docs, openai_embeddings, openai_topic_ids).

    The four arrays are aligned by the same `ORDER BY uid` ordering used in
    `pipeline.load_sample`. We re-load the BERTopic assignments from the
    frozen `assignments.parquet` and reindex to that uid order.
    """
    uids, docs, embeddings = pipeline.load_sample(sample_path=config.SAMPLE_A_PATH)
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")
    rows = con.execute(
        f"SELECT uid, topic_id FROM '{config.ASSIGNMENTS_PATH}'"
    ).fetchall()
    label_by_uid = {u: int(t) for u, t in rows}
    topics = np.array([label_by_uid[u] for u in uids], dtype=np.int64)
    return list(uids), list(docs), embeddings, topics


# ---------------------------------------------------------------------------
# §7.4 NMF baseline
# ---------------------------------------------------------------------------

def run_nmf_baseline(
    docs: list[str], k: int, seed: int = 42
) -> tuple[np.ndarray, list[list[str]]]:
    """Fit TF-IDF + NMF(k) on the docs. Return (argmax labels, top-words)."""
    print(f"  NMF: building CountVectorizer ngram={config.NGRAM_RANGE}, "
          f"min_df={config.COUNT_MIN_DF}, max_df={config.COUNT_MAX_DF}…")
    stops = list(ENGLISH_STOP_WORDS) + list(config.CUSTOM_STOPWORDS)
    vectorizer = CountVectorizer(
        ngram_range=config.NGRAM_RANGE,
        min_df=config.COUNT_MIN_DF,
        max_df=config.COUNT_MAX_DF,
        stop_words=stops,
        token_pattern=config.TOKEN_PATTERN,
    )
    counts = vectorizer.fit_transform(docs)
    print(f"  NMF: vocab={counts.shape[1]:,} terms, nnz={counts.nnz:,}")
    tfidf = TfidfTransformer().fit_transform(counts)
    print(f"  NMF: fitting NMF(n_components={k}, random_state={seed})…")
    t0 = time.perf_counter()
    nmf = NMF(
        n_components=k,
        random_state=seed,
        init="nndsvd",
        max_iter=400,
        tol=1e-4,
    )
    W = nmf.fit_transform(tfidf)
    H = nmf.components_
    print(f"  NMF: fit_transform done in {time.perf_counter() - t0:.1f}s "
          f"(reconstruction err={nmf.reconstruction_err_:.3f})")
    labels = np.asarray(W.argmax(axis=1), dtype=np.int64)
    feature_names = vectorizer.get_feature_names_out()
    top_words: list[list[str]] = []
    n_top = 10
    for c in range(k):
        idx = np.argsort(H[c])[::-1][:n_top]
        top_words.append([feature_names[i] for i in idx])
    return labels, top_words


# ---------------------------------------------------------------------------
# §7.5 MiniLM cross-embedding rerun
# ---------------------------------------------------------------------------

def encode_minilm(docs: list[str]) -> np.ndarray:
    if MINILM_EMB_NPY.exists():
        emb = np.load(MINILM_EMB_NPY)
        if emb.shape == (len(docs), 384):
            print(f"  MiniLM: loaded cached embeddings {emb.shape}")
            return emb
        print(f"  MiniLM: cached emb shape {emb.shape} mismatch, re-encoding")
    print(f"  MiniLM: encoding {len(docs):,} docs with all-MiniLM-L6-v2…")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    t0 = time.perf_counter()
    emb = model.encode(
        docs,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    print(f"  MiniLM: encoded in {time.perf_counter() - t0:.1f}s, shape={emb.shape}")
    np.save(MINILM_EMB_NPY, emb)
    return emb


def fit_minilm_bertopic(
    uids: list[str], docs: list[str], emb_minilm: np.ndarray, k: int
) -> np.ndarray:
    """Fit BERTopic at headline (mcs, seed=42) on MiniLM embeddings,
    reduce to k. Return per-row labels in the same uid order as input."""
    if MINILM_TOPICS_NPY.exists():
        labels = np.load(MINILM_TOPICS_NPY)
        if len(labels) == len(uids):
            print(f"  MiniLM-BERTopic: loaded cached labels (len={len(labels)})")
            return labels
        print(f"  MiniLM-BERTopic: cached label len mismatch, refitting")

    headline_mcs = 70
    print(f"  MiniLM-BERTopic: building BERTopic(mcs={headline_mcs}, seed=42) on 384-d…")
    topic_model = pipeline.build_topic_model(
        min_cluster_size=headline_mcs, seed=config.SEED_PRIMARY,
    )
    t0 = time.perf_counter()
    topics_raw, _ = topic_model.fit_transform(docs, emb_minilm)
    topics_raw = np.asarray(topics_raw, dtype=np.int64)
    np.save(MINILM_RAW_TOPICS_NPY, topics_raw)
    n_raw = len({int(t) for t in topics_raw if t != -1})
    noise = float(np.mean(topics_raw == -1))
    print(f"  MiniLM-BERTopic: raw fit done in {time.perf_counter() - t0:.1f}s "
          f"(n_clusters={n_raw}, noise={noise:.1%})")

    # Reduce to k. Use permissive vectorizer for c-TF-IDF re-train (per pipeline).
    topic_model.vectorizer_model = pipeline.make_permissive_vectorizer()
    print(f"  MiniLM-BERTopic: reduce_topics(nr_topics={k})…")
    try:
        topic_model.reduce_topics(docs, nr_topics=k)
        labels = np.asarray(topic_model.topics_, dtype=np.int64)
    except Exception as exc:
        print(f"    reduce_topics failed: {exc}; using raw labels.")
        labels = topics_raw

    # Persist the model (small) so cluster top-words are recoverable in the
    # alignment step. We will read top-words back via the in-memory
    # `topic_model` object below; saving the npy is sufficient for caching.
    np.save(MINILM_TOPICS_NPY, labels)
    # Stash the model for later top-words extraction by writing a sidecar.
    minilm_model_path = config.INTERMEDIATE_DIR / "t_method_minilm_model.bertopic"
    topic_model.save(
        str(minilm_model_path),
        serialization="pickle",
        save_ctfidf=True,
        save_embedding_model=False,
    )
    print(f"  MiniLM-BERTopic: saved model -> {minilm_model_path}")
    return labels


def get_minilm_top_words(docs: list[str], k: int) -> dict[int, list[str]]:
    """Reload MiniLM-BERTopic model, return top-words per cluster id."""
    minilm_model_path = config.INTERMEDIATE_DIR / "t_method_minilm_model.bertopic"
    model = pipeline.load_topic_model_for_reduce(minilm_model_path)
    out: dict[int, list[str]] = {}
    info = model.get_topic_info()
    for tid in info["Topic"].tolist():
        if tid == -1:
            continue
        words = [w for w, _ in model.get_topic(int(tid))[:10]]
        out[int(tid)] = words
    return out


# ---------------------------------------------------------------------------
# Cluster-level Hungarian alignment + top-word side-by-side
# ---------------------------------------------------------------------------

def cluster_centroids_3072(
    topics: np.ndarray, embeddings: np.ndarray
) -> dict[int, np.ndarray]:
    """L2-normalized centroids in the OpenAI 3072-d space, excluding -1."""
    out: dict[int, np.ndarray] = {}
    for cid in sorted({int(t) for t in topics if t != -1}):
        mask = topics == cid
        c = embeddings[mask].mean(axis=0)
        n = float(np.linalg.norm(c))
        if n > 0:
            c = c / n
        out[cid] = c.astype(np.float32)
    return out


def hungarian_match(
    centroids_a: dict[int, np.ndarray],
    centroids_b: dict[int, np.ndarray],
) -> tuple[list[tuple[int, int, float]], np.ndarray, list[int], list[int]]:
    """Return list of (cid_a, cid_b, cosine), the full sim matrix, and key orders."""
    keys_a = list(centroids_a)
    keys_b = list(centroids_b)
    if not keys_a or not keys_b:
        return [], np.zeros((0, 0)), keys_a, keys_b
    mat_a = np.stack([centroids_a[k] for k in keys_a])
    mat_b = np.stack([centroids_b[k] for k in keys_b])
    sim = mat_a @ mat_b.T  # both are L2-normalized
    cost = -sim
    row_idx, col_idx = linear_sum_assignment(cost)
    pairs = [
        (keys_a[r], keys_b[c], float(sim[r, c]))
        for r, c in zip(row_idx, col_idx)
    ]
    return pairs, sim, keys_a, keys_b


def per_cluster_overlap_score(
    topics_open: np.ndarray, topics_other: np.ndarray, k_clusters: int = 9
) -> dict[int, float]:
    """For each OpenAI cluster, return the maximum (#shared/n_open) over any
    other-method cluster — i.e. the fraction of OpenAI cluster members captured
    by the best-matching other cluster. Excludes -1 from `topics_open`."""
    out: dict[int, float] = {}
    open_ids = sorted({int(t) for t in topics_open if t != -1})
    for oid in open_ids:
        mask = topics_open == oid
        n = int(mask.sum())
        if n == 0:
            out[oid] = float("nan")
            continue
        other = topics_other[mask]
        # Filter -1 in other-method labels.
        valid = other[other != -1] if (other == -1).any() else other
        if len(valid) == 0:
            out[oid] = 0.0
            continue
        ids, counts = np.unique(valid, return_counts=True)
        out[oid] = float(counts.max() / n)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== T-method ===")
    print("[1/6] Verifying Stage 1 hash bundle…")
    verify_hashes()

    print("[2/6] Loading Sample A + OpenAI assignments…")
    uids, docs, openai_emb, openai_topics = load_sample_a_with_assignments()
    print(f"  n_rows={len(uids):,}, openai_emb_dim={openai_emb.shape[1]}")
    print(f"  OpenAI: noise rate {(openai_topics == -1).mean():.1%}, "
          f"n_clusters={len({int(t) for t in openai_topics if t != -1})}")
    headline_k = 10  # spec: headline K = 10

    # -------------------------------------------------------------------
    # NMF baseline
    # -------------------------------------------------------------------
    print(f"\n[3/6] §7.4 NMF baseline (k={headline_k}, seed=42)…")
    if NMF_TOPICS_NPY.exists():
        nmf_labels = np.load(NMF_TOPICS_NPY)
        print(f"  NMF: loaded cached labels (len={len(nmf_labels)})")
        # Top words still need recompute for the alignment step; but if cache
        # exists, we recompute from the same vectorizer. To avoid duplicate
        # work, recompute fully here (cheap).
        nmf_labels, nmf_top_words = run_nmf_baseline(docs, k=headline_k, seed=42)
    else:
        nmf_labels, nmf_top_words = run_nmf_baseline(docs, k=headline_k, seed=42)
        np.save(NMF_TOPICS_NPY, nmf_labels)

    # ARI / NMI for NMF: compare against BERTopic labels including outliers
    # (the spec says "vs BERTopic headline-K assignments"; we report both
    # all-rows and excluding-OpenAI-noise variants for transparency).
    ari_nmf_all = float(adjusted_rand_score(openai_topics, nmf_labels))
    nmi_nmf_all = float(normalized_mutual_info_score(openai_topics, nmf_labels))
    mask_non_noise = openai_topics != -1
    ari_nmf_non_noise = float(adjusted_rand_score(
        openai_topics[mask_non_noise], nmf_labels[mask_non_noise]
    ))
    nmi_nmf_non_noise = float(normalized_mutual_info_score(
        openai_topics[mask_non_noise], nmf_labels[mask_non_noise]
    ))
    print(f"  NMF vs BERTopic (all rows incl. -1 noise as a class): "
          f"ARI={ari_nmf_all:.3f}, NMI={nmi_nmf_all:.3f}")
    print(f"  NMF vs BERTopic (excl. OpenAI noise rows, n="
          f"{int(mask_non_noise.sum()):,}): "
          f"ARI={ari_nmf_non_noise:.3f}, NMI={nmi_nmf_non_noise:.3f}")

    # -------------------------------------------------------------------
    # MiniLM cross-embedding
    # -------------------------------------------------------------------
    print(f"\n[4/6] §7.5 MiniLM cross-embedding (mcs=70, seed=42, "
          f"reduce K={headline_k})…")
    minilm_emb = encode_minilm(docs)
    minilm_topics = fit_minilm_bertopic(uids, docs, minilm_emb, k=headline_k)
    n_minilm_clusters = len({int(t) for t in minilm_topics if t != -1})
    minilm_noise = float(np.mean(minilm_topics == -1))
    print(f"  MiniLM-BERTopic: n_clusters={n_minilm_clusters}, "
          f"noise={minilm_noise:.1%}")

    ari_minilm_all = float(adjusted_rand_score(openai_topics, minilm_topics))
    nmi_minilm_all = float(normalized_mutual_info_score(openai_topics, minilm_topics))
    # Both-methods-non-noise mask for an apples-to-apples view.
    both_non_noise = (openai_topics != -1) & (minilm_topics != -1)
    ari_minilm_both = float(adjusted_rand_score(
        openai_topics[both_non_noise], minilm_topics[both_non_noise]
    ))
    nmi_minilm_both = float(normalized_mutual_info_score(
        openai_topics[both_non_noise], minilm_topics[both_non_noise]
    ))
    print(f"  MiniLM-BERTopic vs OpenAI (all rows, -1 as class): "
          f"ARI={ari_minilm_all:.3f}, NMI={nmi_minilm_all:.3f}")
    print(f"  MiniLM-BERTopic vs OpenAI (both-non-noise, n={int(both_non_noise.sum()):,}): "
          f"ARI={ari_minilm_both:.3f}, NMI={nmi_minilm_both:.3f}")

    # -------------------------------------------------------------------
    # Cluster-level Hungarian alignment + top-words side-by-side
    # -------------------------------------------------------------------
    print(f"\n[5/6] Cluster-level alignment (Hungarian on 3072-d centroids)…")

    # OpenAI top-words from frozen topic_info.parquet (label + top_words).
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")
    rows = con.execute(
        f"SELECT topic_id, label, top_words FROM '{config.TOPIC_INFO_PATH}' "
        "ORDER BY topic_id"
    ).fetchall()
    openai_top_words: dict[int, list[str]] = {int(t): list(w) for t, _, w in rows}
    openai_label: dict[int, str] = {int(t): l for t, l, _ in rows}

    openai_centroids = cluster_centroids_3072(openai_topics, openai_emb)

    # NMF: centroid in OpenAI 3072 space (using OpenAI embeddings as the
    # shared coordinate frame for Hungarian — methodological note: this
    # measures whether the NMF's argmax partition lies in similar regions
    # of the OpenAI space).
    nmf_centroids = cluster_centroids_3072(nmf_labels, openai_emb)
    nmf_pairs, _, _, _ = hungarian_match(openai_centroids, nmf_centroids)

    # MiniLM: centroid in OpenAI 3072 space, same rationale.
    minilm_centroids = cluster_centroids_3072(minilm_topics, openai_emb)
    minilm_pairs, _, _, _ = hungarian_match(openai_centroids, minilm_centroids)

    # Per-cluster overlap score (fraction of each OpenAI cluster captured by
    # its best-matching other-method cluster) — this drives the
    # "agree" / "disagree" picks.
    overlap_nmf = per_cluster_overlap_score(openai_topics, nmf_labels)
    overlap_minilm = per_cluster_overlap_score(openai_topics, minilm_topics)

    print("  per-cluster overlap (OpenAI cluster -> best NMF / best MiniLM):")
    for cid in sorted(overlap_nmf):
        print(f"    c{cid} ({openai_label.get(cid, '?')[:30]:30s}): "
              f"NMF={overlap_nmf[cid]:.2f}  MiniLM={overlap_minilm.get(cid, float('nan')):.2f}")

    minilm_top_words = get_minilm_top_words(docs, k=headline_k)

    # Build the cluster-comparison parquet.
    rows_out = []
    nmf_pair_dict = {a: (b, sim) for a, b, sim in nmf_pairs}
    mini_pair_dict = {a: (b, sim) for a, b, sim in minilm_pairs}
    for cid in sorted(openai_centroids):
        nmf_b, nmf_sim = nmf_pair_dict.get(cid, (None, float("nan")))
        mini_b, mini_sim = mini_pair_dict.get(cid, (None, float("nan")))
        rows_out.append({
            "openai_cluster_id": cid,
            "openai_label": openai_label.get(cid),
            "openai_top_words": openai_top_words.get(cid, [])[:10],
            "nmf_matched_cluster": int(nmf_b) if nmf_b is not None else -1,
            "nmf_centroid_cosine": nmf_sim,
            "nmf_overlap_share": overlap_nmf.get(cid, float("nan")),
            "nmf_top_words": (
                nmf_top_words[int(nmf_b)] if nmf_b is not None and 0 <= int(nmf_b) < len(nmf_top_words) else []
            ),
            "minilm_matched_cluster": int(mini_b) if mini_b is not None else -1,
            "minilm_centroid_cosine": mini_sim,
            "minilm_overlap_share": overlap_minilm.get(cid, float("nan")),
            "minilm_top_words": minilm_top_words.get(int(mini_b), []) if mini_b is not None else [],
        })
    cluster_df = pd.DataFrame(rows_out)
    pq.write_table(pa.Table.from_pandas(cluster_df), CLUSTER_COMPARISON_PARQUET, compression="zstd")
    print(f"  wrote {CLUSTER_COMPARISON_PARQUET}")

    # Pick top-5 most-agreed and top-5 most-disagreed (vs whichever method).
    # We use the MiniLM overlap_share for the "primary" agree/disagree call
    # because it is the cross-embedding test (the spec's headline). NMF
    # selections are reported alongside.
    cluster_df_sorted_min = cluster_df.sort_values("minilm_overlap_share", ascending=False)
    agree_min = cluster_df_sorted_min.head(5)["openai_cluster_id"].tolist()
    disagree_min = cluster_df_sorted_min.tail(5)["openai_cluster_id"].tolist()
    cluster_df_sorted_nmf = cluster_df.sort_values("nmf_overlap_share", ascending=False)
    agree_nmf = cluster_df_sorted_nmf.head(5)["openai_cluster_id"].tolist()
    disagree_nmf = cluster_df_sorted_nmf.tail(5)["openai_cluster_id"].tolist()
    print(f"  MiniLM agree-5 (highest overlap):    {agree_min}")
    print(f"  MiniLM disagree-5 (lowest overlap):  {disagree_min}")
    print(f"  NMF    agree-5 (highest overlap):    {agree_nmf}")
    print(f"  NMF    disagree-5 (lowest overlap):  {disagree_nmf}")

    # -------------------------------------------------------------------
    # Write method_comparison.parquet
    # -------------------------------------------------------------------
    print(f"\n[6/6] Writing {OUTPUT_PARQUET}…")
    summary_rows = [
        {
            "comparison": "nmf_vs_bertopic_openai_K10_all",
            "ari": ari_nmf_all,
            "nmi": nmi_nmf_all,
            "n_overlap": len(openai_topics),
            "notes": (
                "NMF(k=10, seed=42) on TF-IDF of description_core_llm; "
                f"vocab~{int(np.unique(np.concatenate([np.array([0])])).shape[0])}; "
                "OpenAI -1 noise rows retained as their own class."
            ),
        },
        {
            "comparison": "nmf_vs_bertopic_openai_K10_excl_openai_noise",
            "ari": ari_nmf_non_noise,
            "nmi": nmi_nmf_non_noise,
            "n_overlap": int(mask_non_noise.sum()),
            "notes": (
                "Same NMF fit; restricted to rows where OpenAI BERTopic "
                "assigned a non-noise cluster."
            ),
        },
        {
            "comparison": "minilm_bertopic_vs_bertopic_openai_K10_all",
            "ari": ari_minilm_all,
            "nmi": nmi_minilm_all,
            "n_overlap": len(openai_topics),
            "notes": (
                "BERTopic refit on all-MiniLM-L6-v2 384-d embeddings, "
                f"mcs=70, seed=42, reduced to K=10 (n_clusters_actual="
                f"{n_minilm_clusters}, noise={minilm_noise:.1%}); "
                "noise treated as own class on both sides."
            ),
        },
        {
            "comparison": "minilm_bertopic_vs_bertopic_openai_K10_both_non_noise",
            "ari": ari_minilm_both,
            "nmi": nmi_minilm_both,
            "n_overlap": int(both_non_noise.sum()),
            "notes": (
                "Same MiniLM-BERTopic fit; restricted to rows where both "
                "methods assigned a non-noise cluster."
            ),
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    pq.write_table(pa.Table.from_pandas(summary_df), OUTPUT_PARQUET, compression="zstd")
    print(f"  wrote {OUTPUT_PARQUET}")

    # -------------------------------------------------------------------
    # Memo-friendly summary dump
    # -------------------------------------------------------------------
    summary = {
        "headline_k": headline_k,
        "n_rows": len(uids),
        "openai_n_clusters": int(len({int(t) for t in openai_topics if t != -1})),
        "openai_noise_rate": float((openai_topics == -1).mean()),
        "minilm_n_clusters": n_minilm_clusters,
        "minilm_noise_rate": minilm_noise,
        "nmf": {
            "ari_all": ari_nmf_all,
            "nmi_all": nmi_nmf_all,
            "ari_excl_noise": ari_nmf_non_noise,
            "nmi_excl_noise": nmi_nmf_non_noise,
            "per_cluster_max_overlap": overlap_nmf,
            "hungarian_pairs": [
                {"openai": a, "nmf": b, "centroid_cosine": s}
                for a, b, s in nmf_pairs
            ],
        },
        "minilm": {
            "ari_all": ari_minilm_all,
            "nmi_all": nmi_minilm_all,
            "ari_both_non_noise": ari_minilm_both,
            "nmi_both_non_noise": nmi_minilm_both,
            "per_cluster_max_overlap": overlap_minilm,
            "hungarian_pairs": [
                {"openai": a, "minilm": b, "centroid_cosine": s}
                for a, b, s in minilm_pairs
            ],
        },
        "agree_disagree": {
            "minilm_agree_top5": agree_min,
            "minilm_disagree_bottom5": disagree_min,
            "nmf_agree_top5": agree_nmf,
            "nmf_disagree_bottom5": disagree_nmf,
        },
    }
    summary_path = config.INTERMEDIATE_DIR / "t_method_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"  wrote {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
