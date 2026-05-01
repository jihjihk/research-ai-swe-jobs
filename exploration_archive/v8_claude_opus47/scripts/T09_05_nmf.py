"""T09 Step 5: NMF topic modeling on the same 8k sample.

Uses TF-IDF on description_cleaned (LLM-source only) with custom stopwords.
Experiments k in {5, 8, 12, 15, 22}. Picks k based on interpretability (and
a proxy: reconstruction error elbow + topic redundancy).

Writes:
  exploration/tables/T09/nmf_sweep.csv       (per k: reconstruction error, topic redundancy)
  exploration/tables/T09/nmf_topic_terms.csv (chosen k: topic, rank, term, weight)
  exploration/artifacts/T09/nmf_assignments.parquet  (uid, nmf_topic)
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF

from T09_03_bertopic import EXTRA_STOPWORDS

OUTDIR = "exploration/artifacts/T09"
TABLES = "exploration/tables/T09"
SEED = 20260417


def topic_redundancy(H, top_n=20):
    """Fraction of top-n pairs of topics whose top-20 terms overlap by >50%."""
    import itertools
    n_topics, n_words = H.shape
    top_idx = np.argsort(-H, axis=1)[:, :top_n]
    redundant = 0
    n_pairs = 0
    for i, j in itertools.combinations(range(n_topics), 2):
        n_pairs += 1
        overlap = len(set(top_idx[i]).intersection(top_idx[j]))
        if overlap > top_n // 2:
            redundant += 1
    return redundant / n_pairs if n_pairs > 0 else 0.0


def main():
    docs_df = pd.read_parquet(f"{OUTDIR}/sample_docs.parquet")
    docs = docs_df["description_cleaned"].tolist()
    print(f"Loaded {len(docs)} docs")

    stopword_list = list(ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS))
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=10,
        max_df=0.85,
        stop_words=stopword_list,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\+\#\.\-/]{1,}\b",
        max_features=20000,
    )
    X = tfidf.fit_transform(docs)
    print(f"TF-IDF: {X.shape}")
    feat = np.array(tfidf.get_feature_names_out())

    rows = []
    all_models = {}
    for k in [5, 8, 12, 15, 22]:
        print(f"k={k}")
        m = NMF(n_components=k, random_state=SEED, init="nndsvd", max_iter=300)
        W = m.fit_transform(X)
        H = m.components_
        err = m.reconstruction_err_
        red = topic_redundancy(H, top_n=20)
        rows.append({"k": k, "reconstruction_err": err, "redundancy": red})
        all_models[k] = (m, W, H)
        print(f"  err={err:.4f}, redundancy={red:.3f}")

    sweep = pd.DataFrame(rows)
    sweep.to_csv(f"{TABLES}/nmf_sweep.csv", index=False)

    # Pick k: minimize (redundancy + 0.0005 * err-normalized)
    # Simple heuristic: prefer lower redundancy, resolve ties with more topics
    # (more granularity) so long as redundancy stays low.
    # Based on the sweep we pick the largest k with redundancy < 0.08.
    good = sweep[sweep["redundancy"] < 0.08]
    if len(good) > 0:
        best_k = int(good["k"].max())
    else:
        best_k = int(sweep.loc[sweep["redundancy"].idxmin(), "k"])
    print(f"Picked k={best_k}")

    m, W, H = all_models[best_k]
    # Top-20 terms per topic
    top_terms_rows = []
    for t in range(best_k):
        top_idx = np.argsort(-H[t])[:20]
        for rank, idx in enumerate(top_idx):
            top_terms_rows.append({
                "topic": t, "rank": rank,
                "term": feat[idx], "weight": float(H[t, idx]),
            })
    pd.DataFrame(top_terms_rows).to_csv(f"{TABLES}/nmf_topic_terms.csv", index=False)

    # Hard assignments
    assignments = np.argmax(W, axis=1)
    out = docs_df[["uid"]].copy()
    out["nmf_topic"] = assignments
    out["run_id"] = f"nmf_k={best_k}_seed={SEED}"
    out.to_parquet(f"{OUTDIR}/nmf_assignments.parquet", index=False)

    # Save NMF sizes
    sizes = pd.Series(assignments).value_counts().sort_index()
    sizes_df = pd.DataFrame({"topic": sizes.index, "count": sizes.values})
    sizes_df.to_csv(f"{TABLES}/nmf_topic_sizes.csv", index=False)
    print(sizes_df.to_string(index=False))

    with open(f"{OUTDIR}/nmf_best_config.json", "w") as f:
        json.dump({"k": best_k, "reconstruction_err": float(m.reconstruction_err_),
                   "seed": SEED}, f, indent=2)


if __name__ == "__main__":
    main()
