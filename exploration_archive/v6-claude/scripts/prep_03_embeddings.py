"""Wave 1.5 Agent Prep — Step 2: sentence-transformer embeddings.

Compute `all-MiniLM-L6-v2` embeddings on first 512 tokens of description_cleaned
for rows where text_source='llm' ONLY. Batch 256 at a time.

Outputs:
  exploration/artifacts/shared/swe_embeddings.npy       (float32, shape [n, 384])
  exploration/artifacts/shared/swe_embedding_index.parquet  (uid -> row idx)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

CLEANED = Path(
    "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_cleaned_text.parquet"
)
EMB_OUT = Path(
    "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_embeddings.npy"
)
IDX_OUT = Path(
    "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_embedding_index.parquet"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH = 256


def main() -> None:
    t0 = time.time()
    tbl = pq.read_table(CLEANED, columns=["uid", "description_cleaned", "text_source"])
    df = tbl.to_pandas()
    mask = df["text_source"] == "llm"
    sub = df.loc[mask].reset_index(drop=True)
    # Drop empty cleaned text
    sub = sub[sub["description_cleaned"].str.len() > 0].reset_index(drop=True)
    n = len(sub)
    print(f"Rows to embed (text_source='llm'): {n:,}")

    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    # Truncate to 512 tokens (model max is 256 by default; set explicitly)
    model.max_seq_length = 256  # all-MiniLM-L6-v2 hard cap

    texts = sub["description_cleaned"].tolist()
    uids = sub["uid"].tolist()

    # Pre-truncate by whitespace tokens to approx 512 words (prevents model tokenizer blowup)
    texts = [" ".join(t.split()[:512]) for t in texts]

    embs = np.zeros((n, 384), dtype=np.float32)
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        batch = texts[start:end]
        try:
            vecs = model.encode(
                batch,
                batch_size=BATCH,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embs[start:end] = vecs.astype(np.float32)
        except Exception as e:
            # Partial-save fallback
            print(f"OOM/err at batch {start}-{end}: {e}")
            # Save partial results
            partial_embs = embs[:start]
            partial_uids = uids[:start]
            np.save(EMB_OUT, partial_embs)
            idx_tbl = pa.table(
                {"row_idx": pa.array(list(range(len(partial_uids))), type=pa.int32()),
                 "uid": pa.array(partial_uids, type=pa.string())}
            )
            pq.write_table(idx_tbl, IDX_OUT, compression="zstd")
            print(f"Partial embeddings saved: {start:,} rows")
            raise
        if (end) % (BATCH * 8) == 0 or end == n:
            print(f"  {end:,}/{n:,} ({100 * end / n:.1f}%)")

    np.save(EMB_OUT, embs)
    idx_tbl = pa.table(
        {"row_idx": pa.array(list(range(n)), type=pa.int32()),
         "uid": pa.array(uids, type=pa.string())}
    )
    pq.write_table(idx_tbl, IDX_OUT, compression="zstd")
    print(f"Wrote {EMB_OUT} shape={embs.shape} and {IDX_OUT}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
