#!/usr/bin/env python3
"""
T09 step 3: NMF on the T09 sample (TF-IDF) for comparison with BERTopic.
k = 5, 8, 12, 15. For each k, record top 20 terms per component and assign
each doc to its argmax component.
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

OUT_TABLES = "exploration/tables/T09"
os.makedirs(OUT_TABLES, exist_ok=True)

sample = pd.read_parquet("exploration/tables/T09/sample.parquet").reset_index(drop=True)
docs = sample["description_cleaned"].fillna("").tolist()
print(f"Docs: {len(docs):,}")

# ---------------------------------------------------------------------------
# TF-IDF with the same token pattern / stopwords as BERTopic
# ---------------------------------------------------------------------------
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    min_df=10,
    max_df=0.6,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z+#\.\-]{1,30}\b",
    sublinear_tf=True,
    norm="l2",
)
X = tfidf.fit_transform(docs)
terms = tfidf.get_feature_names_out()
print(f"TF-IDF shape: {X.shape}")

# ---------------------------------------------------------------------------
# NMF at several k
# ---------------------------------------------------------------------------
assign_rows = []  # long form: one row per (uid, k)
term_rows = []  # long form: one row per (k, component, rank, term, weight)
k_summary = []
nmf_assignments = {}
for k in [5, 8, 12, 15]:
    print(f"\n--- NMF k={k} ---")
    nmf = NMF(
        n_components=k,
        init="nndsvd",
        random_state=42,
        max_iter=400,
        beta_loss="frobenius",
    )
    W = nmf.fit_transform(X)
    H = nmf.components_
    recon_err = float(nmf.reconstruction_err_)
    assign = W.argmax(axis=1)
    # Top 20 terms per component
    for c in range(k):
        order = np.argsort(-H[c])[:20]
        for rank, idx in enumerate(order):
            term_rows.append(
                {
                    "k": k,
                    "component": c,
                    "rank": rank,
                    "term": terms[idx],
                    "weight": float(H[c, idx]),
                }
            )
    nmf_assignments[k] = assign
    k_summary.append(
        {
            "k": k,
            "reconstruction_error": round(recon_err, 4),
            "n_iter": int(nmf.n_iter_),
        }
    )
    print(f"  reconstruction_error={recon_err:.3f}, iters={nmf.n_iter_}")

pd.DataFrame(term_rows).to_csv(f"{OUT_TABLES}/nmf_topic_terms.csv", index=False)
pd.DataFrame(k_summary).to_csv(f"{OUT_TABLES}/nmf_summary.csv", index=False)

# Save assignments long form
rows = []
for k, assign in nmf_assignments.items():
    for uid, c in zip(sample["uid"].values, assign):
        rows.append({"uid": uid, "k": k, "nmf_component": int(c)})
pd.DataFrame(rows).to_parquet(f"{OUT_TABLES}/nmf_assignments.parquet", index=False)

# Also save wide form for k=12 as the primary comparison candidate
wide = pd.DataFrame(
    {
        "uid": sample["uid"].values,
        "nmf_k5": nmf_assignments[5],
        "nmf_k8": nmf_assignments[8],
        "nmf_k12": nmf_assignments[12],
        "nmf_k15": nmf_assignments[15],
    }
)
wide.to_parquet(f"{OUT_TABLES}/nmf_assignments_wide.parquet", index=False)
print("\nDONE NMF stage")
