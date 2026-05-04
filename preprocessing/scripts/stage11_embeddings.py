#!/usr/bin/env python3
"""
Stage 11: OpenAI embeddings for title + LLM-cleaned description.

Input:
  - preprocessing/intermediate/stage10_llm_integrated.parquet

Output:
  - preprocessing/intermediate/stage11_embeddings_integrated.parquet

The output is row-preserving and adds one posting-level column:
  - job_description_embedding
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import logging
import random
import re
import sqlite3
import struct
import threading
import time
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from io_utils import cleanup_temp_file, prepare_temp_output, promote_temp_file, promote_null_schema
from llm_shared import build_openai_headers


PROJECT_ROOT = Path(__file__).parent.parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
CACHE_DIR = PROJECT_ROOT / "preprocessing" / "cache"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

DEFAULT_INPUT_PATH = INTERMEDIATE_DIR / "stage10_llm_integrated.parquet"
DEFAULT_OUTPUT_PATH = INTERMEDIATE_DIR / "stage11_embeddings_integrated.parquet"
DEFAULT_CACHE_DB = CACHE_DIR / "openai_embeddings.db"

EMBEDDINGS_API_URL = "https://api.openai.com/v1/embeddings"
EMBEDDING_TASK_NAME = "job_description_embedding"
EMBEDDING_INPUT_VERSION = "title_description_core_llm_v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 3072

# Keep chunks small because each row may carry a 3,072-float vector.
CHUNK_SIZE = 2_000
SQLITE_IN_LIMIT = 900
OUTPUT_COLUMN = "job_description_embedding"
CHARS_PER_TOKEN_SAFETY = 3


class EmbeddingTask:
    __slots__ = ("input_hash", "text", "token_estimate")

    def __init__(self, input_hash: str, text: str, token_estimate: int) -> None:
        self.input_hash = input_hash
        self.text = text
        self.token_estimate = token_estimate


class RetryableEmbeddingError(RuntimeError):
    def __init__(self, message: str, *, retry_delay: float | None = None) -> None:
        super().__init__(message)
        self.retry_delay = retry_delay


class FatalEmbeddingError(RuntimeError):
    pass


class RatePause:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paused_until = 0.0

    def pause(self, seconds: float) -> None:
        if seconds <= 0:
            return
        with self._lock:
            self._paused_until = max(self._paused_until, time.time() + seconds)

    def wait(self) -> None:
        while True:
            with self._lock:
                remaining = self._paused_until - time.time()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 30.0))


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "stage11_embeddings.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def chunked(seq: list, size: int):
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def build_embedding_text(
    title: object,
    description_core_llm: object,
    *,
    max_input_tokens: int,
) -> tuple[str | None, int]:
    description = normalize_text(description_core_llm)
    if not description:
        return None, 0

    clean_title = normalize_text(title)
    text = f"{clean_title}\n\n{description}" if clean_title else description
    max_chars = max_input_tokens * CHARS_PER_TOKEN_SAFETY
    if len(text) > max_chars:
        if clean_title:
            description_budget = max(0, max_chars - len(clean_title) - 2)
            text = f"{clean_title}\n\n{description[:description_budget].rstrip()}"
        else:
            text = description[:max_chars].rstrip()
    return text, estimate_tokens(text)


def compute_input_hash(text: str, *, model: str, dimensions: int) -> str:
    h = hashlib.sha256()
    h.update(EMBEDDING_INPUT_VERSION.encode("utf-8"))
    h.update(b"\0")
    h.update(model.encode("utf-8"))
    h.update(b"\0")
    h.update(str(dimensions).encode("ascii"))
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def open_embedding_cache(cache_db: Path) -> sqlite3.Connection:
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cache_db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            input_hash TEXT NOT NULL,
            model TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            input_version TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (input_hash, model, dimensions, input_version)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_embeddings_lookup
        ON embeddings (model, dimensions, input_version, input_hash)
        """
    )
    conn.commit()
    return conn


def pack_embedding(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *[float(value) for value in values])


def unpack_embedding(blob: bytes, dimensions: int) -> list[float]:
    return list(struct.unpack(f"<{dimensions}f", blob))


def fetch_cached_embeddings(
    conn: sqlite3.Connection,
    hashes: list[str],
    *,
    model: str,
    dimensions: int,
) -> dict[str, list[float]]:
    unique_hashes = sorted(set(hashes))
    if not unique_hashes:
        return {}

    out: dict[str, list[float]] = {}
    for batch in chunked(unique_hashes, SQLITE_IN_LIMIT):
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"""
            SELECT input_hash, embedding
            FROM embeddings
            WHERE model = ?
              AND dimensions = ?
              AND input_version = ?
              AND input_hash IN ({placeholders})
            """,
            [model, dimensions, EMBEDDING_INPUT_VERSION, *batch],
        ).fetchall()
        for input_hash, blob in rows:
            out[input_hash] = unpack_embedding(blob, dimensions)
    return out


def store_cached_embeddings(
    conn: sqlite3.Connection,
    embeddings: dict[str, list[float]],
    *,
    model: str,
    dimensions: int,
) -> None:
    if not embeddings:
        return
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT OR REPLACE INTO embeddings
        (input_hash, model, dimensions, input_version, embedding, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                input_hash,
                model,
                dimensions,
                EMBEDDING_INPUT_VERSION,
                pack_embedding(vector),
                now,
            )
            for input_hash, vector in embeddings.items()
        ],
    )
    conn.commit()


def parse_delay_seconds(value: str | None) -> float | None:
    if not value:
        return None
    raw = value.strip().lower()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass

    total = 0.0
    matched = False
    for number, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", raw):
        matched = True
        amount = float(number)
        if unit == "ms":
            total += amount / 1000.0
        elif unit == "s":
            total += amount
        elif unit == "m":
            total += amount * 60.0
        elif unit == "h":
            total += amount * 3600.0
    return total if matched else None


def retry_delay_from_headers(headers: httpx.Headers) -> float | None:
    delays = [
        parse_delay_seconds(headers.get("retry-after")),
        parse_delay_seconds(headers.get("x-ratelimit-reset-requests")),
        parse_delay_seconds(headers.get("x-ratelimit-reset-tokens")),
    ]
    delays = [delay for delay in delays if delay is not None]
    return max(delays) if delays else None


def is_quota_exhausted(response: httpx.Response) -> bool:
    try:
        error = (response.json() or {}).get("error", {}) or {}
    except ValueError:
        error = {}
    haystack = " ".join(
        str(error.get(key, "")) for key in ("code", "type", "message")
    ).lower()
    return "insufficient_quota" in haystack or "quota" in haystack or "billing" in haystack


def request_embedding_batch(
    tasks: list[EmbeddingTask],
    *,
    model: str,
    dimensions: int,
    timeout_seconds: int,
) -> dict[str, list[float]]:
    batch_id = hashlib.sha256("".join(task.input_hash for task in tasks).encode("ascii")).hexdigest()
    response = httpx.post(
        EMBEDDINGS_API_URL,
        headers=build_openai_headers(input_hash=batch_id, task_name=EMBEDDING_TASK_NAME),
        json={
            "model": model,
            "input": [task.text for task in tasks],
            "dimensions": dimensions,
            "encoding_format": "float",
        },
        timeout=timeout_seconds,
    )

    if response.status_code != 200:
        if response.status_code == 429 and not is_quota_exhausted(response):
            raise RetryableEmbeddingError(
                f"OpenAI embeddings rate limit: {response.text[:500]}",
                retry_delay=retry_delay_from_headers(response.headers),
            )
        if response.status_code in {408, 500, 502, 503, 504}:
            raise RetryableEmbeddingError(
                f"OpenAI embeddings transient error {response.status_code}: {response.text[:500]}",
                retry_delay=retry_delay_from_headers(response.headers),
            )
        raise FatalEmbeddingError(
            f"OpenAI embeddings request failed with status {response.status_code}: {response.text[:1000]}"
        )

    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != len(tasks):
        raise FatalEmbeddingError("OpenAI embeddings response length did not match request length")

    vectors_by_index: dict[int, list[float]] = {}
    for item in data:
        index = item.get("index")
        vector = item.get("embedding")
        if not isinstance(index, int) or not isinstance(vector, list):
            raise FatalEmbeddingError("OpenAI embeddings response had an invalid item")
        if len(vector) != dimensions:
            raise FatalEmbeddingError(
                f"OpenAI embeddings dimension mismatch: got {len(vector)}, expected {dimensions}"
            )
        vectors_by_index[index] = [float(value) for value in vector]

    if set(vectors_by_index) != set(range(len(tasks))):
        raise FatalEmbeddingError("OpenAI embeddings response indexes did not match request")

    return {
        task.input_hash: vectors_by_index[index]
        for index, task in enumerate(tasks)
    }


def embed_batch_with_retry(
    tasks: list[EmbeddingTask],
    *,
    model: str,
    dimensions: int,
    timeout_seconds: int,
    max_retries: int,
    rate_pause: RatePause,
    log: logging.Logger,
) -> dict[str, list[float]]:
    attempt = 0
    while True:
        rate_pause.wait()
        try:
            return request_embedding_batch(
                tasks,
                model=model,
                dimensions=dimensions,
                timeout_seconds=timeout_seconds,
            )
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            error = RetryableEmbeddingError(f"OpenAI embeddings network error: {exc}")
        except RetryableEmbeddingError as exc:
            error = exc

        if attempt >= max_retries:
            if len(tasks) > 1:
                mid = len(tasks) // 2
                log.warning(
                    "Retry exhausted for batch of %s; splitting into %s and %s",
                    len(tasks),
                    mid,
                    len(tasks) - mid,
                )
                left = embed_batch_with_retry(
                    tasks[:mid],
                    model=model,
                    dimensions=dimensions,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    rate_pause=rate_pause,
                    log=log,
                )
                right = embed_batch_with_retry(
                    tasks[mid:],
                    model=model,
                    dimensions=dimensions,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    rate_pause=rate_pause,
                    log=log,
                )
                return {**left, **right}
            raise error

        delay = error.retry_delay if isinstance(error, RetryableEmbeddingError) else None
        if delay is None:
            delay = min(120.0, 2.0 ** attempt)
        delay *= random.uniform(0.75, 1.25)
        if isinstance(error, RetryableEmbeddingError) and error.retry_delay is not None:
            rate_pause.pause(delay)
        time.sleep(delay)
        attempt += 1


def build_task_batches(
    tasks: list[EmbeddingTask],
    *,
    batch_size: int,
    max_batch_tokens: int,
) -> list[list[EmbeddingTask]]:
    batches: list[list[EmbeddingTask]] = []
    current: list[EmbeddingTask] = []
    current_tokens = 0

    for task in tasks:
        would_exceed_size = len(current) >= batch_size
        would_exceed_tokens = current and current_tokens + task.token_estimate > max_batch_tokens
        if would_exceed_size or would_exceed_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(task)
        current_tokens += task.token_estimate

    if current:
        batches.append(current)
    return batches


def embed_missing_tasks(
    tasks: list[EmbeddingTask],
    *,
    model: str,
    dimensions: int,
    batch_size: int,
    max_batch_tokens: int,
    max_workers: int,
    timeout_seconds: int,
    max_retries: int,
    log: logging.Logger,
) -> dict[str, list[float]]:
    if not tasks:
        return {}

    batches = build_task_batches(tasks, batch_size=batch_size, max_batch_tokens=max_batch_tokens)
    rate_pause = RatePause()
    log.info(
        "Embedding %s uncached texts in %s API batches",
        f"{len(tasks):,}",
        f"{len(batches):,}",
    )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    out: dict[str, list[float]] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                embed_batch_with_retry,
                batch,
                model=model,
                dimensions=dimensions,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                rate_pause=rate_pause,
                log=log,
            ): batch
            for batch in batches
        }
        for future in as_completed(future_map):
            result = future.result()
            out.update(result)
            completed += len(result)
            if completed == len(tasks) or completed % 5_000 == 0:
                log.info("Stage 11 embedding progress | completed=%s/%s", f"{completed:,}", f"{len(tasks):,}")
    return out


def run_stage11(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    cache_db: Path = DEFAULT_CACHE_DB,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    max_workers: int = 10,
    batch_size: int = 128,
    max_batch_tokens: int = 250_000,
    max_input_tokens: int = 8_192,
    timeout_seconds: int = 120,
    max_retries: int = 8,
) -> None:
    if embedding_dimensions <= 0:
        raise ValueError("embedding_dimensions must be > 0")
    if max_workers <= 0:
        raise ValueError("max_workers must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    log = configure_logging()
    log.info("=" * 70)
    log.info("Stage 11: OpenAI job description embeddings")
    log.info("=" * 70)
    log.info("Input: %s", input_path)
    log.info("Output: %s", output_path)
    log.info(
        "Embedding config | model=%s | dimensions=%s | workers=%s | batch_size=%s",
        embedding_model,
        embedding_dimensions,
        max_workers,
        batch_size,
    )

    pf = pq.ParquetFile(input_path)
    input_columns = set(pf.schema.names)
    missing = {"title", "description_core_llm"} - input_columns
    if missing:
        raise ValueError(f"Stage 11 input is missing required columns: {sorted(missing)}")

    conn = open_embedding_cache(cache_db)
    tmp_output_path = prepare_temp_output(output_path)
    writer = None
    output_schema = None
    total_rows = 0
    eligible_rows = 0
    cache_hits = 0
    fresh_rows = 0
    skipped_rows = 0

    try:
        for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
            table = pa.Table.from_batches([batch])
            if OUTPUT_COLUMN in table.column_names:
                table = table.drop([OUTPUT_COLUMN])

            titles = table.column("title").to_pylist()
            descriptions = table.column("description_core_llm").to_pylist()

            row_hashes: list[str | None] = []
            unique_tasks: dict[str, EmbeddingTask] = {}
            for title, description in zip(titles, descriptions):
                text, token_estimate = build_embedding_text(
                    title,
                    description,
                    max_input_tokens=max_input_tokens,
                )
                if text is None:
                    row_hashes.append(None)
                    skipped_rows += 1
                    continue
                input_hash = compute_input_hash(
                    text,
                    model=embedding_model,
                    dimensions=embedding_dimensions,
                )
                row_hashes.append(input_hash)
                unique_tasks.setdefault(input_hash, EmbeddingTask(input_hash, text, token_estimate))

            hashes = [input_hash for input_hash in row_hashes if input_hash is not None]
            eligible_rows += len(hashes)
            cached = fetch_cached_embeddings(
                conn,
                hashes,
                model=embedding_model,
                dimensions=embedding_dimensions,
            )
            cache_hits += sum(1 for input_hash in hashes if input_hash in cached)

            missing_tasks = [
                task for input_hash, task in unique_tasks.items()
                if input_hash not in cached
            ]
            fresh = embed_missing_tasks(
                missing_tasks,
                model=embedding_model,
                dimensions=embedding_dimensions,
                batch_size=batch_size,
                max_batch_tokens=max_batch_tokens,
                max_workers=max_workers,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                log=log,
            )
            store_cached_embeddings(
                conn,
                fresh,
                model=embedding_model,
                dimensions=embedding_dimensions,
            )
            cached.update(fresh)
            fresh_rows += sum(1 for input_hash in hashes if input_hash in fresh)

            embeddings = [
                None if input_hash is None else cached.get(input_hash)
                for input_hash in row_hashes
            ]
            unresolved = any(
                input_hash is not None and embedding is None
                for input_hash, embedding in zip(row_hashes, embeddings)
            )
            if unresolved:
                raise RuntimeError("Stage 11 failed to resolve all eligible embeddings")

            embedding_array = pa.array(embeddings, type=pa.list_(pa.float32()))
            out_table = table.append_column(OUTPUT_COLUMN, embedding_array)

            if writer is None:
                output_schema = promote_null_schema(out_table.schema)
                writer = pq.ParquetWriter(tmp_output_path, output_schema, compression="zstd")
            writer.write_table(out_table.cast(output_schema))
            total_rows += table.num_rows

        if writer is not None:
            writer.close()
            writer = None
        promote_temp_file(tmp_output_path, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        cleanup_temp_file(tmp_output_path)
        raise
    finally:
        conn.close()

    log.info(
        "Stage 11 complete | rows=%s | eligible=%s | skipped_empty=%s | cache_hits=%s | fresh_rows=%s",
        f"{total_rows:,}",
        f"{eligible_rows:,}",
        f"{skipped_rows:,}",
        f"{cache_hits:,}",
        f"{fresh_rows:,}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 11 OpenAI embeddings")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--cache-db", type=Path, default=DEFAULT_CACHE_DB)
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-dimensions", type=int, default=DEFAULT_EMBEDDING_DIMENSIONS)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batch-tokens", type=int, default=250_000)
    parser.add_argument("--max-input-tokens", type=int, default=8_192)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage11(
        input_path=args.input,
        output_path=args.output,
        cache_db=args.cache_db,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        max_input_tokens=args.max_input_tokens,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
