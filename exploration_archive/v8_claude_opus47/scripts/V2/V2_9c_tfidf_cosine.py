"""V2.9c — T18 TF-IDF cosine SWE↔adjacent with 1,000 per cell.

Re-derive with 1,000 per cell using cleaned text to confirm 0.80→0.75 slight
divergence.
"""
import random
import pandas as pd
import duckdb
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO = Path("/home/jihgaboot/gabor/job-research")
OUT_DIR = REPO / "exploration/artifacts/V2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    con = duckdb.connect()
    q = """
    SELECT uid, is_swe, is_swe_adjacent,
           COALESCE(description_core_llm, description, '') AS text,
           CASE WHEN period LIKE '2024%' THEN '2024'
                WHEN period LIKE '2026%' THEN '2026' ELSE 'other' END AS period_bucket
    FROM 'data/unified.parquet'
    WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok'
      AND (is_swe OR is_swe_adjacent)
      AND description IS NOT NULL AND description != ''
    """
    df = con.execute(q).df()
    df = df[df["period_bucket"].isin(["2024", "2026"])]
    df = df[df["text"].str.len() > 100]  # filter very short
    print(f"[V2.9c] total rows: {len(df)}")

    random.seed(42)
    samples = {}
    N = 1000
    for per in ["2024", "2026"]:
        for grp in [True, False]:  # swe True, adjacent True when is_swe=False
            if grp:
                sub = df[(df["period_bucket"] == per) & (df["is_swe"])]
                key = ("swe", per)
            else:
                sub = df[(df["period_bucket"] == per) & (df["is_swe_adjacent"])]
                key = ("adj", per)
            s = sub.sample(min(N, len(sub)), random_state=42)
            samples[key] = s

    # TF-IDF fit on all samples, then compute mean-vector per cell
    all_texts = pd.concat([s["text"] for s in samples.values()])
    print("[V2.9c] fitting TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), stop_words="english",
        min_df=5, max_df=0.95,
    )
    X_all = tfidf.fit_transform(all_texts)

    # Mean-vector per cell
    idx = 0
    mean_vecs = {}
    for k, s in samples.items():
        n = len(s)
        X_k = X_all[idx:idx + n]
        mean_vecs[k] = np.asarray(X_k.mean(axis=0)).flatten()
        idx += n

    # Cosine SWE vs adjacent per period
    for per in ["2024", "2026"]:
        swe_v = mean_vecs[("swe", per)]
        adj_v = mean_vecs[("adj", per)]
        cos = cosine_similarity(swe_v.reshape(1, -1), adj_v.reshape(1, -1))[0, 0]
        n_swe = len(samples[("swe", per)])
        n_adj = len(samples[("adj", per)])
        print(f"[V2.9c] {per}: cos(SWE, adjacent) = {cos:.4f} (n_swe={n_swe}, n_adj={n_adj})")

    # Also: 2024-SWE vs 2026-SWE, 2024-adj vs 2026-adj
    for cell in ["swe", "adj"]:
        v0 = mean_vecs[(cell, "2024")]
        v1 = mean_vecs[(cell, "2026")]
        cos = cosine_similarity(v0.reshape(1, -1), v1.reshape(1, -1))[0, 0]
        print(f"[V2.9c] {cell}: cos(2024, 2026) = {cos:.4f}")

    print("\n  T18 claim: 2024-01 cos=0.800, 2024-04 cos=0.767, 2026-03 cos=0.751, 2026-04 cos=0.757")


if __name__ == "__main__":
    main()
