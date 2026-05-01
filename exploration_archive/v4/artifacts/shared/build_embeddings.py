#!/usr/bin/env python3
"""
Compute sentence-transformer embeddings for SWE cleaned text.
Reads swe_cleaned_text.parquet, outputs swe_embeddings.npy + swe_embedding_index.parquet.
"""
import sys
import time
import gc
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path

OUT = Path(__file__).resolve().parent
CLEANED = OUT / "swe_cleaned_text.parquet"

WALL_START = time.time()

def elapsed():
    return f"{time.time() - WALL_START:.1f}s"

# Load cleaned text
print(f"[{elapsed()}] Loading cleaned text ...", flush=True)
table = pq.read_table(CLEANED, columns=['uid', 'description_cleaned'])
uids = table.column('uid').to_pylist()
texts = table.column('description_cleaned').to_pylist()
del table
gc.collect()
print(f"[{elapsed()}] Loaded {len(uids)} rows", flush=True)

# Load model
print(f"[{elapsed()}] Loading sentence-transformer model ...", flush=True)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"[{elapsed()}] Model loaded", flush=True)

# Truncate texts to ~512 tokens worth of chars
MAX_CHARS = 2560
texts_truncated = [(t[:MAX_CHARS] if t else "") for t in texts]
del texts
gc.collect()

# Encode in batches
BATCH_SIZE = 512
n_rows = len(texts_truncated)
embedding_dim = 384
embeddings = np.zeros((n_rows, embedding_dim), dtype=np.float32)

n_batches = (n_rows + BATCH_SIZE - 1) // BATCH_SIZE
print(f"[{elapsed()}] Starting encoding: {n_rows} rows, {n_batches} batches of {BATCH_SIZE}", flush=True)

for i in range(0, n_rows, BATCH_SIZE):
    batch_idx = i // BATCH_SIZE
    end = min(i + BATCH_SIZE, n_rows)
    batch_texts = texts_truncated[i:end]

    t0 = time.time()
    batch_embeddings = model.encode(
        batch_texts,
        show_progress_bar=False,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True
    )
    dt = time.time() - t0
    embeddings[i:end] = batch_embeddings

    if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx == n_batches - 1:
        rate = len(batch_texts) / dt if dt > 0 else 0
        print(f"[{elapsed()}]   Batch {batch_idx + 1}/{n_batches}: {len(batch_texts)} rows in {dt:.1f}s ({rate:.0f} rows/s)", flush=True)

    del batch_texts, batch_embeddings
    gc.collect()

# Save
emb_path = OUT / "swe_embeddings.npy"
np.save(emb_path, embeddings)
print(f"[{elapsed()}] Embeddings saved: {embeddings.shape} ({embeddings.nbytes / 1e6:.1f} MB)", flush=True)

idx_table = pa.table({'uid': pa.array(uids, type=pa.string())})
idx_path = OUT / "swe_embedding_index.parquet"
pq.write_table(idx_table, idx_path)
print(f"[{elapsed()}] Index saved: {len(uids)} rows", flush=True)

del model, embeddings
gc.collect()
print(f"[{elapsed()}] Done.", flush=True)
