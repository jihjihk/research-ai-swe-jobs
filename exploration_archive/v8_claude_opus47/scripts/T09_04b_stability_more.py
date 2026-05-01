"""T09 Step 4b: Additional stability seeds to separate collapse vs genuine
cluster structure. Runs 5 more seeds and records which seeds collapse.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from T09_03_bertopic import EXTRA_STOPWORDS, build_vectorizer, run_one
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

OUTDIR = "exploration/artifacts/T09"
TABLES = "exploration/tables/T09"

SEEDS = [11, 77, 99, 2026, 31337]


def main():
    with open(f"{OUTDIR}/bertopic_best_config.json") as f:
        cfg = json.load(f)
    best_mts = int(cfg["min_topic_size"])

    docs_df = pd.read_parquet(f"{OUTDIR}/sample_docs.parquet")
    embeddings = np.load(f"{OUTDIR}/sample_embeddings.npy")
    docs = docs_df["description_cleaned"].tolist()
    uids = docs_df["uid"].tolist()

    stopword_list = list(ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS))

    results = []
    for seed in SEEDS:
        print(f"Run seed={seed}")
        model, topics, probs, _ = run_one(docs, embeddings, uids, best_mts, seed, stopword_list)
        n_topics = int(len(np.unique(topics)) - (1 if -1 in topics else 0))
        outlier_frac = float((topics == -1).mean())
        results.append({
            "seed": seed, "n_topics": n_topics,
            "outlier_frac": outlier_frac,
            "topics": np.asarray(topics),
        })
        print(f"  n_topics={n_topics}, outlier_frac={outlier_frac:.3f}")

    # Summary table
    summary = pd.DataFrame([{
        "seed": r["seed"], "n_topics": r["n_topics"], "outlier_frac": r["outlier_frac"],
        "collapsed": r["n_topics"] < 8,
    } for r in results])
    summary.to_csv(f"{TABLES}/bertopic_stability_expanded.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
