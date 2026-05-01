"""T15 Step 1: Stratified sample + TF-IDF representation.

Builds the analytical sample (up to 2000 per period x seniority_3level x source),
loads the matching embeddings, and computes TF-IDF SVD-100.

Outputs:
  artifacts/T15/sample_index.parquet         uids + group metadata + embedding row indices
  artifacts/T15/sample_embeddings.npy        (n_sample, 384) float32
  artifacts/T15/sample_tfidf_svd.npy         (n_sample, 100) float32
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import duckdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

ROOT = Path("/home/jihgaboot/gabor/job-research")
SHARED = ROOT / "exploration/artifacts/shared"
OUT = ROOT / "exploration/artifacts/T15"
OUT.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
PER_GROUP_CAP = 2000


def main():
    # Load metadata joined with combined best-available seniority
    con = duckdb.connect()
    meta = con.execute(f"""
        SELECT c.uid, c.source, c.period, c.seniority_3level, c.seniority_final,
               c.is_aggregator, c.yoe_extracted, c.text_source, c.description_cleaned,
               CASE
                 WHEN u.llm_classification_coverage='labeled' THEN u.seniority_llm
                 WHEN u.llm_classification_coverage='rule_sufficient' THEN u.seniority_final
                 ELSE NULL
               END AS seniority_best_available
        FROM read_parquet('{SHARED}/swe_cleaned_text.parquet') c
        LEFT JOIN read_parquet('{ROOT}/data/unified.parquet') u ON c.uid=u.uid
    """).df()
    meta["period2"] = meta["period"].map(
        {"2024-01": "2024", "2024-04": "2024", "2026-03": "2026", "2026-04": "2026"}
    )
    # Exclude aggregators (essential sensitivity (a))
    meta = meta[~meta["is_aggregator"].fillna(False)].reset_index(drop=True)
    print(f"  meta after aggregator exclusion: {len(meta)}")

    # Stratified sample: period2 x seniority_3level (skip unknown), up to 2000 per cell.
    sen_levels = ["junior", "mid", "senior"]
    rng = np.random.default_rng(RNG_SEED)
    sample_idx = []
    for p in ["2024", "2026"]:
        for s in sen_levels:
            m = meta[(meta["period2"] == p) & (meta["seniority_3level"] == s)]
            n = len(m)
            take = min(PER_GROUP_CAP, n)
            if take == 0:
                continue
            chosen = m.sample(take, random_state=rng.integers(1e9)).index.values
            sample_idx.extend(chosen.tolist())
            print(f"    {p} x {s}: {take}/{n}")
    sample_idx = np.array(sorted(set(sample_idx)))
    sample = meta.iloc[sample_idx].reset_index(drop=True).copy()
    print(f"  total sample: {len(sample)}")

    # Report text_source distribution
    print("  text_source distribution in sample:")
    print(sample.groupby(["period2", "seniority_3level", "text_source"]).size().unstack(fill_value=0))

    # Load full embeddings and index
    emb_idx = pq.read_table(SHARED / "swe_embedding_index.parquet").to_pandas()
    uid_to_row = {u: i for i, u in enumerate(emb_idx["uid"].tolist())}
    emb_full = np.load(SHARED / "swe_embeddings.npy", mmap_mode="r")
    sample_rows = np.array([uid_to_row[u] for u in sample["uid"].tolist()])
    sample_emb = np.asarray(emb_full[sample_rows]).astype(np.float32)
    print(f"  sample embeddings: {sample_emb.shape}")

    # Build TF-IDF on sample cleaned text, reduce to SVD 100
    texts = sample["description_cleaned"].fillna("").tolist()
    print("  fitting TF-IDF...")
    tfidf = TfidfVectorizer(
        min_df=10, max_df=0.85, ngram_range=(1, 2), max_features=30000,
        sublinear_tf=True, strip_accents="unicode",
    )
    X = tfidf.fit_transform(texts)
    print(f"  tfidf shape: {X.shape}")

    print("  SVD (100 comp)...")
    svd = TruncatedSVD(n_components=100, random_state=RNG_SEED)
    Z = svd.fit_transform(X)
    Z = normalize(Z)  # L2-normalize for cosine similarity via dot product
    print(f"  svd explained var: {svd.explained_variance_ratio_.sum():.3f}")

    # Save
    sample_out = sample[[
        "uid", "source", "period", "period2", "seniority_3level",
        "seniority_final", "seniority_best_available", "yoe_extracted", "text_source"
    ]].copy()
    sample_out["sample_row"] = np.arange(len(sample_out))
    sample_out.to_parquet(OUT / "sample_index.parquet", index=False)
    np.save(OUT / "sample_embeddings.npy", sample_emb)
    np.save(OUT / "sample_tfidf_svd.npy", Z.astype(np.float32))
    print(f"  wrote artifacts to {OUT}")
    print("Done T15 step 01.")


if __name__ == "__main__":
    main()
