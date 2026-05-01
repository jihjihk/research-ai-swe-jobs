"""T09 Step 6: Method comparison — BERTopic vs NMF.

Computes:
 1. Jaccard top-20 overlap matrix (BERTopic x NMF).
 2. Best 1:1 match per BERTopic topic.
 3. Global ARI between BERTopic (reduced) and NMF assignments.

Writes:
  exploration/tables/T09/method_jaccard_matrix.csv
  exploration/tables/T09/method_best_matches.csv
  exploration/tables/T09/method_ari.csv
  exploration/tables/T09/method_comparison_table.csv  (combined summary)
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score

OUTDIR = "exploration/artifacts/T09"
TABLES = "exploration/tables/T09"


def jaccard(a, b):
    s1, s2 = set(a), set(b)
    if not s1 and not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def main():
    bt_terms = pd.read_csv(f"{TABLES}/bertopic_topic_terms.csv")
    nmf_terms = pd.read_csv(f"{TABLES}/nmf_topic_terms.csv")

    # Top-20 per topic
    bt_topics = sorted(bt_terms["topic"].unique())
    nmf_topics = sorted(nmf_terms["topic"].unique())
    bt_top = {t: bt_terms[bt_terms.topic == t].sort_values("rank")["term"].head(20).tolist() for t in bt_topics}
    nmf_top = {t: nmf_terms[nmf_terms.topic == t].sort_values("rank")["term"].head(20).tolist() for t in nmf_topics}

    # Jaccard matrix
    rows = []
    mat = np.zeros((len(bt_topics), len(nmf_topics)))
    for i, bt in enumerate(bt_topics):
        for j, nt in enumerate(nmf_topics):
            j_score = jaccard(bt_top[bt], nmf_top[nt])
            mat[i, j] = j_score
            rows.append({"bertopic_topic": bt, "nmf_topic": nt, "jaccard": j_score})
    jmat = pd.DataFrame(mat, index=bt_topics, columns=nmf_topics)
    jmat.to_csv(f"{TABLES}/method_jaccard_matrix.csv", index_label="bertopic_topic")

    # Best 1:1 match per BERTopic topic (including -1)
    bm = []
    for i, bt in enumerate(bt_topics):
        jbest_idx = int(np.argmax(mat[i]))
        bm.append({
            "bertopic_topic": bt,
            "best_nmf_topic": nmf_topics[jbest_idx],
            "best_jaccard": float(mat[i, jbest_idx]),
            "bertopic_name": ",".join(bt_top[bt][:5]),
            "nmf_name": ",".join(nmf_top[nmf_topics[jbest_idx]][:5]),
        })
    bm_df = pd.DataFrame(bm)
    bm_df.to_csv(f"{TABLES}/method_best_matches.csv", index=False)
    print(bm_df.to_string(index=False))

    # Reverse: best 1:1 match per NMF topic
    rev = []
    for j, nt in enumerate(nmf_topics):
        ibest = int(np.argmax(mat[:, j]))
        rev.append({
            "nmf_topic": nt,
            "best_bertopic_topic": bt_topics[ibest],
            "best_jaccard": float(mat[ibest, j]),
            "nmf_name": ",".join(nmf_top[nt][:5]),
            "bertopic_name": ",".join(bt_top[bt_topics[ibest]][:5]),
        })
    rev_df = pd.DataFrame(rev)
    rev_df.to_csv(f"{TABLES}/method_best_matches_nmf_side.csv", index=False)

    # ARI between BERTopic reduced and NMF assignments
    bt_asg = pd.read_parquet(f"{OUTDIR}/bertopic_assignments.parquet")
    nmf_asg = pd.read_parquet(f"{OUTDIR}/nmf_assignments.parquet")
    merged = bt_asg.merge(nmf_asg, on="uid")
    ari_raw = adjusted_rand_score(merged["topic"], merged["nmf_topic"])
    ari_red = adjusted_rand_score(merged["topic_reduced"], merged["nmf_topic"])
    print(f"ARI raw (BT with outliers, NMF) = {ari_raw:.3f}")
    print(f"ARI reduced (BT reduced, NMF) = {ari_red:.3f}")
    pd.DataFrame([
        {"label": "ari_raw_bt_vs_nmf", "value": ari_raw},
        {"label": "ari_reduced_bt_vs_nmf", "value": ari_red},
    ]).to_csv(f"{TABLES}/method_ari.csv", index=False)

    # Summary of BERTopic→NMF matching quality
    bm_valid = bm_df[bm_df["bertopic_topic"] != -1]
    print(f"\nMedian best Jaccard (BT->NMF, excluding outlier topic): {bm_valid['best_jaccard'].median():.3f}")
    print(f"Min / Max: {bm_valid['best_jaccard'].min():.3f} / {bm_valid['best_jaccard'].max():.3f}")
    print(f"Topics with Jaccard >= 0.30 (well-matched):")
    print(bm_valid[bm_valid["best_jaccard"] >= 0.30][["bertopic_topic", "best_nmf_topic", "best_jaccard", "bertopic_name"]].to_string(index=False))


if __name__ == "__main__":
    main()
