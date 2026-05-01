#!/usr/bin/env python3
"""
T09 step 2: Run BERTopic on the T09 sample using shared sentence-transformer
embeddings. Experiment with min_topic_size in {20, 30, 50}. Record:
  * number of topics
  * outlier / noise percentage
  * c-TF-IDF top terms per topic
  * topic coherence (UMass) on TF-IDF
  * stability (ARI) across 3 seeds at min_topic_size=30

Save topic assignments for the primary model (min_topic_size=30).
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.metrics import adjusted_rand_score

OUT_TABLES = "exploration/tables/T09"
OUT_FIG = "exploration/figures/T09"
os.makedirs(OUT_TABLES, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

# ---------------------------------------------------------------------------
# Load sample and embeddings
# ---------------------------------------------------------------------------
sample = pd.read_parquet("exploration/tables/T09/sample.parquet").reset_index(drop=True)
emb_index = pd.read_parquet("exploration/artifacts/shared/swe_embedding_index.parquet")
emb_all = np.load("exploration/artifacts/shared/swe_embeddings.npy")

# Align embeddings to sample order
uid_to_row = {u: i for i, u in enumerate(emb_index["uid"].tolist())}
idx = np.array([uid_to_row[u] for u in sample["uid"]], dtype=np.int64)
emb = emb_all[idx]
docs = sample["description_cleaned"].fillna("").tolist()
print(f"Docs: {len(docs):,}, embeddings: {emb.shape}")

# ---------------------------------------------------------------------------
# Vectorizer used for c-TF-IDF topic representation (shared across models)
# ---------------------------------------------------------------------------
def make_vectorizer() -> CountVectorizer:
    # NOTE: BERTopic fits the vectorizer on pseudo-docs (one per topic), so
    # min_df must be 1 to avoid failing when there are only a handful of topics.
    return CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1,
        max_df=1.0,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z+#\.\-]{1,30}\b",
    )


def make_coh_vectorizer() -> CountVectorizer:
    # Vectorizer used for coherence computation only, run over all docs.
    return CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=10,
        max_df=0.6,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z+#\.\-]{1,30}\b",
    )


def make_ctfidf() -> ClassTfidfTransformer:
    return ClassTfidfTransformer(reduce_frequent_words=True)


def make_umap(seed: int) -> UMAP:
    return UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
        low_memory=True,
    )


def make_hdbscan(min_topic_size: int) -> HDBSCAN:
    return HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )


def run_bertopic(min_topic_size: int, seed: int = 42) -> tuple[BERTopic, np.ndarray, np.ndarray]:
    model = BERTopic(
        umap_model=make_umap(seed),
        hdbscan_model=make_hdbscan(min_topic_size),
        vectorizer_model=make_vectorizer(),
        ctfidf_model=make_ctfidf(),
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(docs, embeddings=emb)
    return model, np.array(topics), None


# ---------------------------------------------------------------------------
# Experiment: min_topic_size in {20, 30, 50}
# ---------------------------------------------------------------------------
sweep_results = []
topics_by_min = {}
models_by_min = {}
for mts in [20, 30, 50]:
    print(f"\n--- BERTopic min_topic_size={mts} ---")
    model, topics, _ = run_bertopic(mts, seed=42)
    topics_by_min[mts] = topics
    models_by_min[mts] = model
    topic_info = model.get_topic_info()
    n_topics = int((topic_info["Topic"] >= 0).sum())
    noise = int((topics == -1).sum())
    pct_noise = 100.0 * noise / len(topics)
    print(f"  # topics (excl. -1): {n_topics}   noise: {noise} ({pct_noise:.1f}%)")
    sweep_results.append(
        {
            "min_topic_size": mts,
            "n_topics": n_topics,
            "n_docs": int(len(topics)),
            "n_noise": noise,
            "pct_noise": round(pct_noise, 2),
        }
    )

pd.DataFrame(sweep_results).to_csv(
    f"{OUT_TABLES}/bertopic_min_topic_size_sweep.csv", index=False
)
print("\nSweep results:")
print(pd.DataFrame(sweep_results).to_string(index=False))

# ---------------------------------------------------------------------------
# Primary model = min_topic_size=30
# ---------------------------------------------------------------------------
primary_model = models_by_min[30]
primary_topics = topics_by_min[30]

# Save topic info
topic_info = primary_model.get_topic_info()
topic_info.to_csv(f"{OUT_TABLES}/bertopic_topic_info.csv", index=False)

# Top-term table (top 20 per topic)
rows = []
for tid in topic_info["Topic"].tolist():
    if tid == -1:
        continue
    words = primary_model.get_topic(tid)
    for rank, (w, score) in enumerate(words[:20]):
        rows.append(
            {
                "topic": int(tid),
                "rank": int(rank),
                "term": w,
                "weight": float(score),
            }
        )
pd.DataFrame(rows).to_csv(f"{OUT_TABLES}/bertopic_topic_terms.csv", index=False)
print(f"Saved topic info + terms for {len(topic_info)-1} topics")

# Save primary topic assignments (uid -> topic)
pd.DataFrame({"uid": sample["uid"].values, "bertopic_topic_mts30": primary_topics}).to_parquet(
    f"{OUT_TABLES}/bertopic_topics.parquet", index=False
)

# ---------------------------------------------------------------------------
# Stability: 3 seeds at mts=30, ARI between runs (on non-noise common docs)
# ---------------------------------------------------------------------------
print("\n--- Stability: 3 seeds @ mts=30 ---")
seed_topics = {}
for seed in [42, 7, 2026]:
    print(f"  seed={seed}")
    model, topics, _ = run_bertopic(30, seed=seed)
    seed_topics[seed] = topics

pairs = [(42, 7), (42, 2026), (7, 2026)]
ari_rows = []
for a, b in pairs:
    ta = seed_topics[a]
    tb = seed_topics[b]
    # ARI on all docs (including noise as its own class)
    ari_all = adjusted_rand_score(ta, tb)
    mask = (ta != -1) & (tb != -1)
    ari_core = adjusted_rand_score(ta[mask], tb[mask])
    ari_rows.append(
        {
            "seed_a": a,
            "seed_b": b,
            "ari_all_docs": round(float(ari_all), 4),
            "ari_core_only": round(float(ari_core), 4),
            "n_core": int(mask.sum()),
        }
    )
pd.DataFrame(ari_rows).to_csv(f"{OUT_TABLES}/bertopic_stability_ari.csv", index=False)
print(pd.DataFrame(ari_rows).to_string(index=False))

# ---------------------------------------------------------------------------
# Topic coherence (simplified UMass) on the primary model.
# Uses the CountVectorizer matrix built from docs.
# ---------------------------------------------------------------------------
from scipy.sparse import csr_matrix

coh_vectorizer = make_coh_vectorizer()
X = coh_vectorizer.fit_transform(docs)
terms = coh_vectorizer.get_feature_names_out()
term_to_idx = {t: i for i, t in enumerate(terms)}
X_bin = (X > 0).astype(np.int8)
# Document frequency per term
df = np.asarray(X_bin.sum(axis=0)).ravel()
n_docs = X_bin.shape[0]
EPS = 1.0


def umass_coherence(topic_words: list[str], top_n: int = 10) -> float:
    w = [t for t in topic_words if t in term_to_idx][:top_n]
    if len(w) < 2:
        return float("nan")
    ids = [term_to_idx[t] for t in w]
    cols = X_bin[:, ids].toarray()  # (n_docs, k)
    total = 0.0
    pairs = 0
    for i in range(1, len(w)):
        for j in range(i):
            co = int((cols[:, i] & cols[:, j]).sum())
            d_j = int(cols[:, j].sum())
            total += np.log((co + EPS) / max(d_j, 1))
            pairs += 1
    return float(total / max(pairs, 1))


coh_rows = []
for tid in topic_info["Topic"].tolist():
    if tid == -1:
        continue
    words = [w for w, _ in primary_model.get_topic(tid)]
    coh = umass_coherence(words, top_n=10)
    coh_rows.append({"topic": int(tid), "umass_coherence": round(coh, 4)})
coh_df = pd.DataFrame(coh_rows)
coh_df.to_csv(f"{OUT_TABLES}/bertopic_topic_coherence.csv", index=False)
print(f"\nMean UMass coherence (primary): {coh_df['umass_coherence'].mean():.3f}")

# ---------------------------------------------------------------------------
# Save BERTopic visualizations as static images
# ---------------------------------------------------------------------------
try:
    fig = primary_model.visualize_topics()
    fig.write_image(f"{OUT_FIG}/bertopic_topics_scatter.png", scale=2)
    print("  Saved bertopic_topics_scatter.png")
except Exception as e:
    print(f"  visualize_topics failed: {e}")

try:
    fig = primary_model.visualize_barchart(top_n_topics=16, n_words=8)
    fig.write_image(f"{OUT_FIG}/bertopic_barchart.png", scale=2)
    print("  Saved bertopic_barchart.png")
except Exception as e:
    print(f"  visualize_barchart failed: {e}")

try:
    fig = primary_model.visualize_hierarchy(top_n_topics=30)
    fig.write_image(f"{OUT_FIG}/bertopic_hierarchy.png", scale=2)
    print("  Saved bertopic_hierarchy.png")
except Exception as e:
    print(f"  visualize_hierarchy failed: {e}")

# Save summary JSON
summary = {
    "sample_size": int(len(docs)),
    "primary_min_topic_size": 30,
    "primary_n_topics": int((topic_info["Topic"] >= 0).sum()),
    "primary_pct_noise": round(100.0 * (primary_topics == -1).mean(), 2),
    "mean_umass_coherence": round(float(coh_df["umass_coherence"].mean()), 4),
    "ari_seeds": ari_rows,
    "sweep": sweep_results,
}
with open(f"{OUT_TABLES}/bertopic_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nDONE BERTopic stage")
