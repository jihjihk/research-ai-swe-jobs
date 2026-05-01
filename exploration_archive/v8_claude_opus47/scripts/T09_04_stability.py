"""T09 Step 4: Cluster stability via 3 BERTopic runs with different seeds.

Runs 3 BERTopic fits on the same 8k sample at best min_topic_size.
Computes Adjusted Rand Index between all pairs.

Writes: exploration/tables/T09/bertopic_stability_ari.csv
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

SEEDS = [20260417, 42, 1234]


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
        # Reduce outliers so every doc has a label -- same policy as step 3
        topics_reduced = model.reduce_outliers(docs, topics, strategy="c-tf-idf")
        topics_reduced = np.asarray(topics_reduced)
        runs.append({
            "seed": seed,
            "topics_raw": np.asarray(topics),
            "topics_reduced": topics_reduced,
            "n_topics": int(len(np.unique(topics)) - (1 if -1 in topics else 0)),
        })
        print(f"  n_topics={runs[-1]['n_topics']}, outlier_frac={(topics == -1).mean():.3f}")

    # ARI pairs (raw topics with outliers) and (reduced)
    rows = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a = runs[i]
            b = runs[j]
            # Raw topics: include outliers as their own class
            ari_raw = adjusted_rand_score(a["topics_raw"], b["topics_raw"])
            ari_red = adjusted_rand_score(a["topics_reduced"], b["topics_reduced"])
            rows.append({
                "seed_a": a["seed"], "seed_b": b["seed"],
                "n_topics_a": a["n_topics"], "n_topics_b": b["n_topics"],
                "ari_raw": ari_raw, "ari_reduced": ari_red,
            })
            print(f"Pair {a['seed']}×{b['seed']}: ARI_raw={ari_raw:.3f} ARI_red={ari_red:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{TABLES}/bertopic_stability_ari.csv", index=False)

    # Save per-seed assignments
    asg = docs_df[["uid"]].copy()
    for r in runs:
        asg[f"topic_seed{r['seed']}"] = r["topics_reduced"]
    asg.to_parquet(f"{OUTDIR}/bertopic_seed_assignments.parquet", index=False)
    print("Stability report saved.")


if __name__ == "__main__":
    main()
