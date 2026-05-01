"""T09 Step 2: Subset the shared MiniLM embeddings to the T09 sample.

Reads: exploration/artifacts/shared/swe_embeddings.npy (34,102 x 384)
       exploration/artifacts/shared/swe_embedding_index.parquet (row_idx, uid)
       exploration/artifacts/T09/sample_index.parquet
Writes: exploration/artifacts/T09/sample_embeddings.npy (N x 384)
        exploration/artifacts/T09/sample_docs.parquet (metadata + docs)
"""

import numpy as np
import pandas as pd


def main():
    emb = np.load("exploration/artifacts/shared/swe_embeddings.npy")
    idx = pd.read_parquet("exploration/artifacts/shared/swe_embedding_index.parquet")
    sample = pd.read_parquet("exploration/artifacts/T09/sample_index.parquet")
    print(f"emb shape: {emb.shape}")
    print(f"index rows: {len(idx)}")
    print(f"sample rows: {len(sample)}")

    # Join sample -> row_idx
    merged = sample.merge(idx, on="uid", how="left", validate="1:1")
    missing = merged.row_idx.isna().sum()
    print(f"sample uids with no embedding: {missing}")
    merged = merged.dropna(subset=["row_idx"]).reset_index(drop=True)
    merged["row_idx"] = merged.row_idx.astype(int)

    # Subset embeddings
    subset_emb = emb[merged.row_idx.values]
    print(f"subset embeddings: {subset_emb.shape}")

    # Order-preserving save
    np.save("exploration/artifacts/T09/sample_embeddings.npy", subset_emb)
    merged_cols = [
        "uid", "description_cleaned", "text_source", "source", "period",
        "seniority_final", "seniority_3level", "seniority_final_source",
        "is_aggregator", "company_name_canonical", "yoe_extracted",
        "swe_classification_tier", "description_cleaned_length",
        "_bucket_period", "row_idx",
    ]
    merged[merged_cols].to_parquet("exploration/artifacts/T09/sample_docs.parquet", index=False)
    print("Wrote sample_embeddings.npy and sample_docs.parquet")


if __name__ == "__main__":
    main()
