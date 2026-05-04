from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from tests.helpers.imports import load_module


stage11 = load_module("stage11_embeddings", "preprocessing/scripts/stage11_embeddings.py")


def _write_stage10_input(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "uid": "u1",
                "title": "Software Engineer",
                "description": "Boilerplate garbage. Build APIs.",
                "description_core_llm": "Build APIs.",
            },
            {
                "uid": "u2",
                "title": "Software Engineer",
                "description": "Different boilerplate. Build APIs.",
                "description_core_llm": "Build APIs.",
            },
            {
                "uid": "u3",
                "title": "Marketing Analyst",
                "description": "No cleaned description.",
                "description_core_llm": None,
            },
            {
                "uid": "u4",
                "title": "Data Engineer",
                "description": "Generic intro. Build pipelines.",
                "description_core_llm": "Build pipelines.",
            },
        ]
    ).to_parquet(path, index=False)


@pytest.mark.unit
def test_run_stage11_uses_description_core_and_cache(monkeypatch, tmp_path):
    input_path = tmp_path / "stage10_llm_integrated.parquet"
    output_path = tmp_path / "stage11_embeddings_integrated.parquet"
    second_output_path = tmp_path / "stage11_embeddings_integrated_second.parquet"
    cache_db = tmp_path / "openai_embeddings.db"
    _write_stage10_input(input_path)

    calls: list[list[str]] = []

    def fake_request(tasks, *, model, dimensions, timeout_seconds):
        calls.append([task.text for task in tasks])
        return {
            task.input_hash: [float(idx + 1)] * dimensions
            for idx, task in enumerate(tasks)
        }

    monkeypatch.setattr(stage11, "request_embedding_batch", fake_request)

    stage11.run_stage11(
        input_path=input_path,
        output_path=output_path,
        cache_db=cache_db,
        embedding_dimensions=3,
        max_workers=1,
        batch_size=2,
    )

    all_texts = [text for batch in calls for text in batch]
    assert all_texts == [
        "Software Engineer\n\nBuild APIs.",
        "Data Engineer\n\nBuild pipelines.",
    ]
    assert all("Boilerplate" not in text and "garbage" not in text for text in all_texts)

    embeddings = pq.read_table(output_path).column("job_description_embedding").to_pylist()
    assert embeddings[0] == embeddings[1]
    assert embeddings[2] is None
    assert embeddings[3] is not None

    def fail_if_called(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("cache hit should avoid API calls")

    monkeypatch.setattr(stage11, "request_embedding_batch", fail_if_called)
    stage11.run_stage11(
        input_path=input_path,
        output_path=second_output_path,
        cache_db=cache_db,
        embedding_dimensions=3,
        max_workers=1,
        batch_size=2,
    )
    assert pq.read_table(second_output_path).num_rows == 4


@pytest.mark.unit
def test_build_task_batches_respects_size_and_token_caps():
    tasks = [
        stage11.EmbeddingTask(f"h{i}", f"text {i}", token_estimate=tokens)
        for i, tokens in enumerate([10, 10, 90, 10])
    ]

    batches = stage11.build_task_batches(tasks, batch_size=3, max_batch_tokens=100)

    assert [[task.input_hash for task in batch] for batch in batches] == [
        ["h0", "h1"],
        ["h2", "h3"],
    ]


@pytest.mark.unit
def test_embed_batch_with_retry_retries_once(monkeypatch, tmp_path):
    attempts = {"count": 0}
    task = stage11.EmbeddingTask("h1", "Software Engineer\n\nBuild APIs.", token_estimate=8)

    def flaky_request(tasks, *, model, dimensions, timeout_seconds):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise stage11.RetryableEmbeddingError("temporary")
        return {"h1": [1.0, 2.0, 3.0]}

    monkeypatch.setattr(stage11, "request_embedding_batch", flaky_request)
    monkeypatch.setattr(stage11.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(stage11.random, "uniform", lambda _low, _high: 1.0)

    result = stage11.embed_batch_with_retry(
        [task],
        model="text-embedding-3-large",
        dimensions=3,
        timeout_seconds=1,
        max_retries=1,
        rate_pause=stage11.RatePause(),
        log=stage11.configure_logging(),
    )

    assert attempts["count"] == 2
    assert result == {"h1": [1.0, 2.0, 3.0]}
