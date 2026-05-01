"""T09 Step 16: Compute ARI across all 5 non-collapsed seeds for robust
stability estimate.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from T09_03_bertopic import EXTRA_STOPWORDS, run_one

OUTDIR = "exploration/artifacts/T09"
TABLES = "exploration/tables/T09"

# Only seeds that were confirmed non-collapsed
SEEDS = [20260417, 42, 77, 99, 2026]


def main():
    with open(f"{OUTDIR}/bertopic_best_config.json") as f:
        cfg = json.load(f)
    best_mts = int(cfg["min_topic_size"])
    docs_df = pd.read_parquet(f"{OUTDIR}/sample_docs.parquet")
    embeddings = np.load(f"{OUTDIR}/sample_embeddings.npy")
    docs = docs_df["description_cleaned"].tolist()
    uids = docs_df["uid"].tolist()
    stopword_list = list(ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS))
    runs = []
    for seed in SEEDS:
        print(f"Run seed={seed}")
        model, topics, probs, _ = run_one(docs, embeddings, uids, best_mts, seed, stopword_list)
        topics_reduced = np.asarray(model.reduce_outliers(docs, topics, strategy="c-tf-idf"))
        runs.append({"seed": seed, "topics_raw": np.asarray(topics),
                      "topics_reduced": topics_reduced})
        print(f"  n_topics={len(np.unique(topics)) - (1 if -1 in topics else 0)}")
    rows = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            ari_raw = adjusted_rand_score(runs[i]["topics_raw"], runs[j]["topics_raw"])
            ari_red = adjusted_rand_score(runs[i]["topics_reduced"], runs[j]["topics_reduced"])
            rows.append({"seed_a": runs[i]["seed"], "seed_b": runs[j]["seed"],
                          "ari_raw": ari_raw, "ari_reduced": ari_red})
            print(f"{runs[i]['seed']}×{runs[j]['seed']}: raw={ari_raw:.3f} red={ari_red:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{TABLES}/bertopic_stability_full.csv", index=False)
    print(f"\nMean ARI (raw) on non-collapsed seeds: {df.ari_raw.mean():.3f}")
    print(f"Mean ARI (reduced) on non-collapsed seeds: {df.ari_reduced.mean():.3f}")
    print(f"n pairs: {len(df)}")


if __name__ == "__main__":
    main()
