"""Wave 1.5 Agent Prep - Step 2: Sentence-transformer embeddings.

Compute all-MiniLM-L6-v2 embeddings on first 512 tokens of `description_cleaned`
for rows where text_source='llm'. Batch size 256, float32.

Output:
 - swe_embeddings.npy (NxD float32)
 - swe_embedding_index.parquet (row_idx -> uid map)
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb

OUT_DIR = Path("exploration/artifacts/shared")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_PATH = OUT_DIR / "swe_embeddings.npy"
IDX_PATH = OUT_DIR / "swe_embedding_index.parquet"
TEXT_PATH = OUT_DIR / "swe_cleaned_text.parquet"

BATCH_SIZE = 256


def main() -> None:
    t0 = time.time()

    print("[step2] importing sentence-transformers")
    from sentence_transformers import SentenceTransformer

    print("[step2] loading llm-text subset")
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT uid, description_cleaned
        FROM '{TEXT_PATH}'
        WHERE text_source = 'llm'
        """
    ).df()
    print(f"[step2] rows to embed: {len(df)}")

    # Keep uid order deterministic for the index file
    uids = df["uid"].tolist()
    texts = df["description_cleaned"].tolist()
    del df
    gc.collect()

    print("[step2] loading all-MiniLM-L6-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Enforce max_seq_length so we truncate at 512 tokens per spec (default is 256)
    model.max_seq_length = 512
    dim = model.get_sentence_embedding_dimension()
    print(f"[step2] model loaded. dim={dim}, max_seq_length={model.max_seq_length}")

    # Pre-allocate output
    n = len(texts)
    embeddings = np.zeros((n, dim), dtype=np.float32)

    # Batched encode
    import psutil
    process = psutil.Process()
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = texts[start:end]
        # Guard against None / empty — replace empty with a single space for stability
        safe_batch = [(t if isinstance(t, str) and t.strip() else " ") for t in batch]
        out = model.encode(
            safe_batch,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        embeddings[start:end] = out.astype(np.float32, copy=False)
        if (start // BATCH_SIZE) % 10 == 0:
            rss_mb = process.memory_info().rss / 1024 / 1024
            print(f"[step2]  batch {start//BATCH_SIZE+1}/{(n+BATCH_SIZE-1)//BATCH_SIZE} (rows {start}-{end}) RSS={rss_mb:.0f} MB")

    print("[step2] saving embeddings")
    np.save(EMB_PATH, embeddings)
    idx_tbl = pa.table({"row_idx": list(range(n)), "uid": uids})
    pq.write_table(idx_tbl, IDX_PATH)

    print(f"[step2] wrote {n} x {dim} embeddings -> {EMB_PATH}")
    print(f"[step2] wrote index -> {IDX_PATH}")
    elapsed = time.time() - t0
    print(f"[step2] elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()
