"""
Build and load the embeddings cache for the BERTopic project.

The cache holds posting embeddings (read from `unified_core.parquet`,
restricted to the union of Sample A and Sample B uids) and anchor embeddings
(125 strings from `config.all_anchor_strings()`, freshly computed via OpenAI's
embeddings API) in a single float32 numpy array. The sidecar
`embeddings_cache.index.parquet` maps each anchor-id-or-uid string to a row
index in the array.

Anchors and postings sharing one space and one I/O path is intentional: every
downstream §6 analysis (axis projection, neighborhood diffusion, WEAT) needs
both at once, and a single source of truth eliminates a class of subtle bugs.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import duckdb
import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from figures.bertopic import config


_EMBEDDINGS_API_URL = "https://api.openai.com/v1/embeddings"
_OPENAI_ENV_KEYS = {"OPENAI_API_KEY", "OPENAI_ORGANIZATION", "OPENAI_PROJECT"}


# ---------------------------------------------------------------------------
# OpenAI auth (mirrors preprocessing/scripts/llm_shared.py for consistency)
# ---------------------------------------------------------------------------

def _load_openai_env() -> None:
    if not config.OPENAI_ENV_FILE.exists():
        raise RuntimeError(
            f"OpenAI env file missing at {config.OPENAI_ENV_FILE}"
        )
    for raw in config.OPENAI_ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, _, raw_value = line.partition("=")
        key = key.strip()
        if key not in _OPENAI_ENV_KEYS or not raw_value.strip():
            continue
        parts = shlex.split(raw_value.strip(), posix=True)
        if len(parts) == 1:
            os.environ.setdefault(key, parts[0])


def _openai_headers() -> dict[str, str]:
    _load_openai_env()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing — populate openai.env or env")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    org = os.environ.get("OPENAI_ORGANIZATION", "").strip()
    project = os.environ.get("OPENAI_PROJECT", "").strip()
    if org:
        headers["OpenAI-Organization"] = org
    if project:
        headers["OpenAI-Project"] = project
    headers["X-Client-Request-Id"] = "job-research:bertopic-anchor-embed"
    return headers


def embed_anchor_strings(anchor_strings: dict[str, str]) -> dict[str, np.ndarray]:
    """Embed all anchor strings in one batch and return id → vector."""
    if not anchor_strings:
        return {}
    keys = list(anchor_strings.keys())
    inputs = [anchor_strings[k] for k in keys]
    response = httpx.post(
        _EMBEDDINGS_API_URL,
        headers=_openai_headers(),
        json={
            "model": config.EMBEDDING_MODEL,
            "input": inputs,
            "dimensions": config.EMBEDDING_DIMS,
            "encoding_format": "float",
        },
        timeout=120.0,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"OpenAI embeddings failed: {response.status_code} "
            f"{response.text[:500]}"
        )
    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != len(keys):
        raise RuntimeError("OpenAI embeddings response length mismatch")

    out: dict[str, np.ndarray] = {}
    for item in data:
        idx = item["index"]
        vec = np.asarray(item["embedding"], dtype=np.float32)
        if vec.shape != (config.EMBEDDING_DIMS,):
            raise RuntimeError(
                f"Anchor embedding dim {vec.shape} != "
                f"({config.EMBEDDING_DIMS},)"
            )
        # Re-normalize defensively; text-embedding-3 returns L2-normalized but
        # the WEAT / axis math depends on it.
        norm = float(np.linalg.norm(vec))
        if not (0.99 <= norm <= 1.01):
            vec = vec / norm
        out[keys[idx]] = vec
    return out


# ---------------------------------------------------------------------------
# Posting embeddings — sourced from unified_core.parquet
# ---------------------------------------------------------------------------

def load_posting_embeddings() -> tuple[list[str], np.ndarray]:
    """Return (uids, matrix) for every uid that appears in Sample A or B.

    Reads the float32 vectors out of `unified_core.parquet` via DuckDB. Result
    is allocated once at the final shape; we do not concatenate per-row.
    """
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")

    sample_uids_query = (
        f"SELECT uid FROM '{config.SAMPLE_A_PATH}' "
        f"UNION SELECT uid FROM '{config.SAMPLE_B_PATH}'"
    )
    n_uids = con.execute(
        f"SELECT count(*) FROM ({sample_uids_query})"
    ).fetchone()[0]

    table = con.execute(f"""
        SELECT u.uid, c.job_description_embedding
        FROM '{config.UNIFIED_CORE_PATH}' c
        JOIN ({sample_uids_query}) u ON u.uid = c.uid
        ORDER BY u.uid
    """).fetch_arrow_table()

    uids = table.column("uid").to_pylist()
    if len(uids) != n_uids:
        raise RuntimeError(
            f"Expected {n_uids} embeddings; got {len(uids)} from unified_core"
        )

    matrix = np.empty((len(uids), config.EMBEDDING_DIMS), dtype=np.float32)
    embedding_col = table.column("job_description_embedding")
    # `arrow()` returns a ListArray of float32; `to_numpy(zero_copy_only=False)`
    # on the underlying values gives us a flat array, but the chunked-array
    # path is simpler to reason about row-wise.
    row_idx = 0
    for chunk in embedding_col.chunks:
        list_array = chunk
        values = np.asarray(list_array.values, dtype=np.float32)
        offsets = np.asarray(list_array.offsets, dtype=np.int64)
        for i in range(len(list_array)):
            start, end = offsets[i], offsets[i + 1]
            if end - start != config.EMBEDDING_DIMS:
                raise RuntimeError(
                    f"Posting {uids[row_idx]} has embedding dim "
                    f"{end - start} != {config.EMBEDDING_DIMS}"
                )
            matrix[row_idx] = values[start:end]
            row_idx += 1
    return uids, matrix


# ---------------------------------------------------------------------------
# Cache build + load
# ---------------------------------------------------------------------------

def build_cache() -> None:
    """Build embeddings_cache.npy and embeddings_cache.index.parquet."""
    config.BERTOPIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    posting_uids, posting_matrix = load_posting_embeddings()
    anchor_strings = config.all_anchor_strings()
    anchor_vectors = embed_anchor_strings(anchor_strings)

    n_postings = len(posting_uids)
    n_anchors = len(anchor_vectors)
    matrix = np.empty(
        (n_postings + n_anchors, config.EMBEDDING_DIMS),
        dtype=np.float32,
    )
    matrix[:n_postings] = posting_matrix

    anchor_keys: list[str] = []
    for i, key in enumerate(sorted(anchor_vectors)):
        matrix[n_postings + i] = anchor_vectors[key]
        anchor_keys.append(key)

    np.save(config.EMBEDDINGS_CACHE_PATH, matrix)

    keys = posting_uids + anchor_keys
    kinds = ["posting"] * n_postings + ["anchor"] * n_anchors
    row_indexes = list(range(len(keys)))
    index_table = pa.table({
        "key": keys,
        "kind": kinds,
        "row_index": row_indexes,
    })
    pq.write_table(
        index_table,
        config.EMBEDDINGS_INDEX_PATH,
        compression="zstd",
    )

    print(
        f"Wrote {n_postings} postings + {n_anchors} anchors → "
        f"{config.EMBEDDINGS_CACHE_PATH}"
    )


def load_cache() -> tuple[np.ndarray, dict[str, int]]:
    """Return (matrix, key→row_index) loaded from disk."""
    matrix = np.load(config.EMBEDDINGS_CACHE_PATH)
    index = pq.read_table(config.EMBEDDINGS_INDEX_PATH).to_pandas()
    return matrix, dict(zip(index["key"], index["row_index"]))


def main() -> None:
    build_cache()


if __name__ == "__main__":
    main()
