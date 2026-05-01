"""Agent Prep step 3: sentence-transformer embeddings on llm-cleaned text.

Compute 384-dim MiniLM embeddings for rows where text_source='llm'. Stream
into a disk-backed np.memmap in batches of 256 to stay under 31 GB RAM.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/home/jihgaboot/gabor/job-research")
IN_PATH = ROOT / "exploration/artifacts/shared/swe_cleaned_text.parquet"
OUT_EMB = ROOT / "exploration/artifacts/shared/swe_embeddings.npy"
OUT_IDX = ROOT / "exploration/artifacts/shared/swe_embedding_index.parquet"

MODEL_NAME = "all-MiniLM-L6-v2"
DIM = 384
BATCH = 256
# MiniLM context: 512 tokens, but we pass truncated text; the model tokenizer
# will truncate. We preemptively truncate at ~2500 characters (~500 tokens)
# to avoid upstream tokenization blow-up.
TRUNCATE_CHARS = 2500


def main() -> None:
    con = duckdb.connect()
    print("Loading LLM-labeled rows from cleaned-text artifact...")
    df = con.execute(
        f"""
        SELECT uid, description_cleaned
        FROM read_parquet('{IN_PATH}')
        WHERE text_source = 'llm'
          AND description_cleaned IS NOT NULL
          AND length(description_cleaned) > 0
        """
    ).df()
    n = len(df)
    print(f"LLM-labeled rows to embed: {n:,}")

    texts = [t[:TRUNCATE_CHARS] if len(t) > TRUNCATE_CHARS else t for t in df["description_cleaned"].values]

    # Memmap file so we don't accumulate 12k * 384 floats = ~18 MB (trivial,
    # but follow the spec's pattern so we're robust to larger datasets).
    OUT_EMB.parent.mkdir(parents=True, exist_ok=True)
    if OUT_EMB.exists():
        OUT_EMB.unlink()

    arr = np.memmap(OUT_EMB, dtype="float32", mode="w+", shape=(n, DIM))

    print(f"Loading {MODEL_NAME}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding in batches of {BATCH}...")
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        batch_texts = texts[start:end]
        emb = model.encode(batch_texts, batch_size=BATCH, show_progress_bar=False, convert_to_numpy=True)
        arr[start:end, :] = emb.astype("float32")
        if (start // BATCH) % 10 == 0:
            print(f"  encoded {end:,}/{n:,}")

    arr.flush()
    del arr

    # Re-save as regular .npy so consumers can np.load directly without knowing shape.
    # np.save automatically appends .npy to the filename if it's not already
    # there, so we save into a temp path and rename explicitly.
    final_arr = np.array(np.memmap(OUT_EMB, dtype="float32", mode="r", shape=(n, DIM)))
    tmp_path = OUT_EMB.parent / (OUT_EMB.name + ".tmp")
    np.save(str(tmp_path), final_arr)  # writes tmp_path.npy
    saved_tmp = tmp_path.parent / (tmp_path.name + ".npy")
    OUT_EMB.unlink()
    saved_tmp.rename(OUT_EMB)
    print(f"Saved {OUT_EMB} ({n:,} × {DIM})")

    # Write the index parquet
    idx_df = df[["uid"]].reset_index(drop=True)
    idx_df.insert(0, "row_idx", np.arange(n, dtype=np.int64))
    table = pa.Table.from_pandas(idx_df, preserve_index=False)
    pq.write_table(table, OUT_IDX, compression="snappy")
    print(f"Saved index {OUT_IDX}")

    size_mb = OUT_EMB.stat().st_size / (1024 * 1024)
    print(f"Embeddings size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
