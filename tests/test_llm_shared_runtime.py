import json
from datetime import datetime
import logging
import subprocess
from zoneinfo import ZoneInfo

import httpx
import pytest

from tests.helpers.imports import load_module
from tests.helpers.llm_fakes import codex_stdout, completed_process, openai_response_json


llm_shared = load_module("llm_shared_runtime", "preprocessing/scripts/llm_shared.py")


PACIFIC = ZoneInfo("America/Los_Angeles")


@pytest.mark.unit
def test_parse_engine_list_and_tiers_defaults_and_validates():
    assert llm_shared.parse_engine_list("codex,claude,openai") == ("codex", "claude", "openai")

    assert llm_shared.parse_engine_tiers(None, ("codex", "claude", "openai")) == {
        "codex": "full",
        "claude": "full",
        "openai": "full",
    }
    assert llm_shared.parse_engine_tiers(
        "codex=full,claude=non_intrusive,openai=full",
        ("codex", "claude", "openai"),
    ) == {
        "codex": "full",
        "claude": "non_intrusive",
        "openai": "full",
    }

    with pytest.raises(ValueError, match="unsupported engine tier"):
        llm_shared.parse_engine_tiers("codex=unknown", ("codex",))


@pytest.mark.unit
def test_format_engine_labels_and_progress_checkpoints_are_compact():
    engines = llm_shared.build_engine_configs(
        ("codex", "claude"),
        codex_model="gpt-5.4-mini",
        claude_model="haiku",
        openai_model="gpt-5.4-nano",
        engine_tiers={"codex": "full", "claude": "non_intrusive"},
    )
    assert llm_shared.format_engine_labels(engines) == (
        "codex(full:gpt-5.4-mini), claude(non_intrusive:haiku)"
    )
    assert llm_shared.build_progress_checkpoints(20) == (1, 2, 5, 10, 15, 18, 20)


@pytest.mark.unit
def test_build_codex_command_uses_json_event_output():
    command = llm_shared.build_codex_command("Return JSON", "gpt-5.4-mini")

    assert command == [
        "codex",
        "exec",
        "--disable",
        "shell_snapshot",
        "--full-auto",
        "--ephemeral",
        "--config",
        "model=gpt-5.4-mini",
        "--config",
        "developer_instructions='You are a labor-market research tool. Return raw JSON.'",
        "--config",
        "model_reasoning_effort=medium",
        "--config",
        "model_verbosity=low",
        "--skip-git-repo-check",
        "--json",
        "Return JSON",
    ]


@pytest.mark.unit
def test_build_remote_ssh_command_reuses_hashed_control_socket():
    command = llm_shared.build_remote_ssh_command(["codex", "exec", "Return JSON"])

    assert command == [
        "ssh",
        "-i",
        "/home/jihgaboot/gabor/job-research/keys/scraper-key.pem",
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=~/.ssh/ssh-mux-%C",
        "-o",
        "ControlPersist=10m",
        "ec2-user@ec2-18-216-89-129.us-east-2.compute.amazonaws.com",
        "codex exec 'Return JSON'",
    ]


@pytest.mark.unit
def test_ensure_remote_ssh_master_starts_master_when_missing(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
        calls.append(command)
        if "-O" in command:
            return subprocess.CompletedProcess(command, 255, stdout="", stderr="Master not running")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(llm_shared.subprocess, "run", fake_run)

    llm_shared.ensure_remote_ssh_master(timeout_seconds=15)

    assert len(calls) == 2
    assert calls[0] == [
        "ssh",
        "-i",
        "/home/jihgaboot/gabor/job-research/keys/scraper-key.pem",
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlPath=~/.ssh/ssh-mux-%C",
        "-O",
        "check",
        "ec2-user@ec2-18-216-89-129.us-east-2.compute.amazonaws.com",
    ]
    assert calls[1] == [
        "ssh",
        "-i",
        "/home/jihgaboot/gabor/job-research/keys/scraper-key.pem",
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=yes",
        "-o",
        "ControlPath=~/.ssh/ssh-mux-%C",
        "-o",
        "ControlPersist=10m",
        "-N",
        "-f",
        "ec2-user@ec2-18-216-89-129.us-east-2.compute.amazonaws.com",
    ]


@pytest.mark.unit
def test_ensure_remote_ssh_master_skips_start_when_master_exists(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="Master running", stderr="")

    monkeypatch.setattr(llm_shared.subprocess, "run", fake_run)

    llm_shared.ensure_remote_ssh_master(timeout_seconds=15)

    assert calls == [[
        "ssh",
        "-i",
        "/home/jihgaboot/gabor/job-research/keys/scraper-key.pem",
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlPath=~/.ssh/ssh-mux-%C",
        "-O",
        "check",
        "ec2-user@ec2-18-216-89-129.us-east-2.compute.amazonaws.com",
    ]]


@pytest.mark.unit
def test_parse_codex_stdout_reads_jsonl_events_and_usage():
    payload = {"swe_classification": "SWE", "seniority": "entry"}
    stdout = codex_stdout(payload, tokens_used=123)

    response_json, tokens_used, cost = llm_shared.parse_codex_stdout(stdout)

    assert json.loads(response_json) == payload
    assert tokens_used == 123
    assert cost is None


@pytest.mark.unit
def test_parse_codex_stdout_falls_back_to_legacy_transcript_format():
    stdout = """codex
{"ok": true}
tokens used
12,190
{"ok": true}
"""

    response_json, tokens_used, cost = llm_shared.parse_codex_stdout(stdout)

    assert json.loads(response_json) == {"ok": True}
    assert tokens_used == 12190
    assert cost is None


@pytest.mark.unit
def test_attempt_provider_call_logs_subprocess_failure_details(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(
        llm_shared,
        "call_subprocess",
        lambda command, timeout_seconds: completed_process(
            stdout="partial stdout",
            stderr="fatal provider error",
            returncode=1,
        ),
    )
    log = logging.getLogger("llm-runtime-test")

    with caplog.at_level(logging.ERROR):
        result, failure_kind, detail = llm_shared.attempt_provider_call(
            provider="codex",
            prompt="Return JSON",
            model="gpt-5.4-mini",
            task_name="classification",
            input_hash="hash-err",
            error_log_path=tmp_path / "llm_errors.jsonl",
            log=log,
            timeout_seconds=5,
            max_retries=1,
            payload_validator=lambda payload: None,
        )

    assert result is None
    assert failure_kind == "failed"
    assert detail == "fatal provider error\npartial stdout"
    assert "Subprocess failed for codex/gpt-5.4-mini on hash-err (classification)" in caplog.text
    assert "fatal provider error" in caplog.text
    assert "partial stdout" in caplog.text


@pytest.mark.unit
def test_attempt_provider_call_logs_error_like_stderr_on_success(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(
        llm_shared,
        "call_subprocess",
        lambda command, timeout_seconds: completed_process(
            stdout=codex_stdout({"ok": True}, tokens_used=12),
            stderr=(
                "2026-03-29T19:06:32Z ERROR codex_core::shell_snapshot: "
                "Shell snapshot validation failed"
            ),
            returncode=0,
        ),
    )
    log = logging.getLogger("llm-runtime-test")

    with caplog.at_level(logging.WARNING):
        result, failure_kind, detail = llm_shared.attempt_provider_call(
            provider="codex",
            prompt="Return JSON",
            model="gpt-5.4-mini",
            task_name="classification",
            input_hash="hash-stderr",
            error_log_path=tmp_path / "llm_errors.jsonl",
            log=log,
            timeout_seconds=5,
            max_retries=1,
            payload_validator=lambda payload: None,
        )

    assert failure_kind is None
    assert detail == ""
    assert result is not None
    assert result["payload"] == {"ok": True}
    assert "Command stderr for codex/gpt-5.4-mini on hash-stderr (classification)" in caplog.text
    assert "Shell snapshot validation failed" in caplog.text


@pytest.mark.unit
def test_openai_payload_uses_structured_outputs_for_classification():
    payload = llm_shared.build_openai_payload(
        prompt="Return JSON",
        task_name=llm_shared.CLASSIFICATION_TASK_NAME,
        model="gpt-5.4-nano",
    )

    assert payload["model"] == "gpt-5.4-nano"
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["name"] == "job_posting_classification"
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["schema"] == llm_shared.CLASSIFICATION_JSON_SCHEMA


@pytest.mark.unit
def test_build_openai_headers_loads_default_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / "openai.env"
    env_file.write_text(
        'export OPENAI_API_KEY="test-key-from-file"\n'
        'OPENAI_PROJECT=proj_123\n'
        "OPENAI_ORGANIZATION=org_456\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_PROJECT", raising=False)
    monkeypatch.delenv("OPENAI_ORGANIZATION", raising=False)
    monkeypatch.setenv("JOB_RESEARCH_OPENAI_ENV_FILE", str(env_file))

    headers = llm_shared.build_openai_headers(input_hash="hash-file", task_name="classification")

    assert headers["Authorization"] == "Bearer test-key-from-file"
    assert headers["OpenAI-Project"] == "proj_123"
    assert headers["OpenAI-Organization"] == "org_456"
    assert headers["X-Client-Request-Id"] == "job-research:classification:hash-file"


@pytest.mark.unit
def test_parse_openai_response_reads_structured_output_and_usage():
    response = httpx.Response(
        200,
        headers={"x-request-id": "req_123"},
        text=openai_response_json(
            {"swe_classification": "SWE", "seniority": "entry", "ghost_assessment": "realistic", "yoe_min_years": 1},
            input_tokens=11,
            output_tokens=7,
        ),
    )

    response_json, tokens_used, cost, model, request_id = llm_shared.parse_openai_response(response)

    assert json.loads(response_json) == {
        "swe_classification": "SWE",
        "seniority": "entry",
        "ghost_assessment": "realistic",
        "yoe_min_years": 1,
    }
    assert tokens_used == 18
    assert cost is None
    assert model == "gpt-5.4-nano"
    assert request_id == "req_123"


@pytest.mark.unit
def test_attempt_provider_call_openai_success(monkeypatch, tmp_path, caplog):
    payload = {
        "swe_classification": "SWE",
        "seniority": "entry",
        "ghost_assessment": "realistic",
        "yoe_min_years": 1,
    }
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        llm_shared.httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(
            200,
            headers={"x-request-id": "req_success"},
            text=openai_response_json(payload, input_tokens=12, output_tokens=5),
        ),
    )
    log = logging.getLogger("llm-runtime-test")

    with caplog.at_level(logging.INFO):
        result, failure_kind, detail = llm_shared.attempt_provider_call(
            provider="openai",
            prompt="Return JSON",
            model="gpt-5.4-nano",
            task_name=llm_shared.CLASSIFICATION_TASK_NAME,
            input_hash="hash-openai-ok",
            error_log_path=tmp_path / "llm_errors.jsonl",
            log=log,
            timeout_seconds=5,
            max_retries=3,
            payload_validator=llm_shared.validate_classification_payload,
        )

    assert failure_kind is None
    assert detail == ""
    assert result is not None
    assert result["tokens_used"] == 17
    assert result["payload"] == payload
    assert result["model"] == "gpt-5.4-nano"
    assert "request_id=req_success" in caplog.text


@pytest.mark.unit
def test_attempt_provider_call_openai_extraction_normalizes_unsorted_ids(monkeypatch, tmp_path):
    payload = {
        "task_status": "ok",
        "boilerplate_unit_ids": [5, 2, 4, 2],
        "uncertain_unit_ids": [9, 7, 7],
        "reason": "drop boilerplate",
    }
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        llm_shared.httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(
            200,
            headers={"x-request-id": "req_extract_ok"},
            text=openai_response_json(payload, input_tokens=12, output_tokens=5),
        ),
    )
    log = logging.getLogger("llm-runtime-test")

    result, failure_kind, detail = llm_shared.attempt_provider_call(
        provider="openai",
        prompt="Return JSON",
        model="gpt-5.4-nano",
        task_name=llm_shared.EXTRACTION_TASK_NAME,
        input_hash="hash-openai-extract-ok",
        error_log_path=tmp_path / "llm_errors.jsonl",
        log=log,
        timeout_seconds=5,
        max_retries=3,
        payload_validator=llm_shared.validate_extraction_payload,
    )

    assert failure_kind is None
    assert detail == ""
    assert result is not None
    assert result["payload"]["boilerplate_unit_ids"] == [2, 4, 5]
    assert result["payload"]["uncertain_unit_ids"] == [7, 9]


@pytest.mark.unit
def test_attempt_provider_call_openai_429_returns_quota_without_retry(monkeypatch, tmp_path):
    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        return httpx.Response(
            429,
            headers={"x-request-id": "req_quota", "x-ratelimit-reset-requests": "10s"},
            text=json.dumps({"error": {"message": "rate limit exceeded"}}),
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_shared.httpx, "post", fake_post)
    log = logging.getLogger("llm-runtime-test")

    result, failure_kind, detail = llm_shared.attempt_provider_call(
        provider="openai",
        prompt="Return JSON",
        model="gpt-5.4-nano",
        task_name=llm_shared.CLASSIFICATION_TASK_NAME,
        input_hash="hash-openai-429",
        error_log_path=tmp_path / "llm_errors.jsonl",
        log=log,
        timeout_seconds=5,
        max_retries=3,
        payload_validator=llm_shared.validate_classification_payload,
    )

    assert result is None
    assert failure_kind == "quota"
    assert "status=429" in detail
    assert calls["count"] == 1


@pytest.mark.unit
def test_attempt_provider_call_openai_missing_api_key_fails_cleanly(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("JOB_RESEARCH_OPENAI_ENV_FILE", str(tmp_path / "missing.env"))
    log = logging.getLogger("llm-runtime-test")

    result, failure_kind, detail = llm_shared.attempt_provider_call(
        provider="openai",
        prompt="Return JSON",
        model="gpt-5.4-nano",
        task_name=llm_shared.CLASSIFICATION_TASK_NAME,
        input_hash="hash-openai-missing-key",
        error_log_path=tmp_path / "llm_errors.jsonl",
        log=log,
        timeout_seconds=5,
        max_retries=3,
        payload_validator=llm_shared.validate_classification_payload,
    )

    assert result is None
    assert failure_kind == "failed"
    assert "OPENAI_API_KEY is not set." in detail
    assert "missing.env" in detail


@pytest.mark.unit
def test_execute_task_with_runtime_openai_failure_does_not_retry(monkeypatch, tmp_path):
    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        return httpx.Response(
            500,
            headers={"x-request-id": "req_fail"},
            text=json.dumps({"error": {"message": "server error"}}),
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_shared.httpx, "post", fake_post)

    runtime = llm_shared.LLMEngineRuntime(
        llm_shared.build_engine_configs(
            ("openai",),
            codex_model="gpt-5.4-mini",
            claude_model="haiku",
            openai_model="gpt-5.4-nano",
            engine_tiers={"openai": "full"},
        ),
        slot_timezone="America/Los_Angeles",
        rng_seed=1,
    )
    log = logging.getLogger("llm-runtime-test")
    log.addHandler(logging.NullHandler())

    result = llm_shared.execute_task_with_runtime(
        runtime=runtime,
        task_name=llm_shared.CLASSIFICATION_TASK_NAME,
        prompt="Return JSON",
        input_hash="hash-openai-fail",
        error_log_path=tmp_path / "llm_errors.jsonl",
        log=log,
        timeout_seconds=5,
        max_retries=3,
        payload_validator=llm_shared.validate_classification_payload,
        retry_sleep_seconds=0.01,
        quota_wait_hours=0.01,
    )

    assert result is None
    assert calls["count"] == 1


@pytest.mark.unit
def test_engine_runtime_keeps_quota_pauses_provider_scoped():
    runtime = llm_shared.LLMEngineRuntime(
        llm_shared.build_engine_configs(
            ("codex", "claude"),
            codex_model="gpt-5.4-mini",
            claude_model="haiku",
            openai_model="gpt-5.4-nano",
            engine_tiers={"codex": "full", "claude": "non_intrusive"},
        ),
        slot_timezone="America/Los_Angeles",
        rng_seed=7,
    )
    log = logging.getLogger("llm-runtime-test")
    log.addHandler(logging.NullHandler())
    now = datetime(2026, 3, 27, 8, 0, tzinfo=PACIFIC)

    runtime.note_quota_hit(
        provider="codex",
        quota_wait_hours=5,
        log=log,
        input_hash="hash-1",
        task_name="classification",
        detail="429 rate limit exceeded",
        now=now,
    )

    engine = runtime.claim_next_engine(now=now)
    assert engine is not None
    assert engine.provider == "claude"
    assert runtime.claim_next_engine(now=now, exclude_providers={"claude"}) is None


@pytest.mark.unit
def test_non_intrusive_runtime_caps_slot_budgets_and_resets_next_slot():
    runtime = llm_shared.LLMEngineRuntime(
        llm_shared.build_engine_configs(
            ("claude",),
            codex_model="gpt-5.4-mini",
            claude_model="haiku",
            openai_model="gpt-5.4-nano",
            engine_tiers={"claude": "non_intrusive"},
        ),
        slot_timezone="America/Los_Angeles",
        rng_seed=1,
    )

    five_am = datetime(2026, 3, 27, 5, 0, tzinfo=PACIFIC)
    for _ in range(2000):
        engine = runtime.claim_next_engine(now=five_am)
        assert engine is not None
        assert engine.provider == "claude"

    assert runtime.claim_next_engine(now=five_am) is None
    assert runtime.next_available_delay(now=five_am) == pytest.approx(5 * 3600)

    ten_am = datetime(2026, 3, 27, 10, 0, tzinfo=PACIFIC)
    engine = runtime.claim_next_engine(now=ten_am)
    assert engine is not None
    assert engine.provider == "claude"


@pytest.mark.unit
def test_non_intrusive_midnight_slot_runs_until_quota_and_pauses_to_slot_end():
    runtime = llm_shared.LLMEngineRuntime(
        llm_shared.build_engine_configs(
            ("claude",),
            codex_model="gpt-5.4-mini",
            claude_model="haiku",
            openai_model="gpt-5.4-nano",
            engine_tiers={"claude": "non_intrusive"},
        ),
        slot_timezone="America/Los_Angeles",
        rng_seed=1,
    )
    log = logging.getLogger("llm-runtime-test")
    log.addHandler(logging.NullHandler())
    one_am = datetime(2026, 3, 27, 1, 0, tzinfo=PACIFIC)

    for _ in range(2500):
        engine = runtime.claim_next_engine(now=one_am)
        assert engine is not None
        assert engine.provider == "claude"

    runtime.note_quota_hit(
        provider="claude",
        quota_wait_hours=5,
        log=log,
        input_hash="hash-2",
        task_name="classification",
        detail="429 rate limit exceeded",
        now=one_am,
    )

    assert runtime.claim_next_engine(now=one_am) is None
    assert runtime.next_available_delay(now=one_am) == pytest.approx(4 * 3600)


@pytest.mark.unit
def test_normalize_extraction_payload_sorts_and_deduplicates_ids():
    payload = {
        "task_status": "ok",
        "boilerplate_unit_ids": [5, 2, 5, 3],
        "uncertain_unit_ids": [9, 7, 7],
        "reason": "drop boilerplate",
    }

    normalized = llm_shared.normalize_extraction_payload(payload)

    assert normalized["boilerplate_unit_ids"] == [2, 3, 5]
    assert normalized["uncertain_unit_ids"] == [7, 9]


@pytest.mark.unit
def test_fetch_cached_rows_can_exclude_retryable_extraction_failures(tmp_path):
    conn = llm_shared.open_cache(tmp_path / "llm_responses.db")
    failed_payload = {
        "task_status": "cannot_complete",
        "boilerplate_unit_ids": [],
        "uncertain_unit_ids": [],
        "reason": "provider_failed",
    }
    ok_payload = {
        "task_status": "ok",
        "boilerplate_unit_ids": [1],
        "uncertain_unit_ids": [],
        "reason": "drop boilerplate",
    }
    llm_shared.store_cached_row(
        conn,
        input_hash="hash-failed",
        task_name=llm_shared.EXTRACTION_TASK_NAME,
        model="synthetic-provider-failed",
        prompt_version=llm_shared.EXTRACTION_PROMPT_VERSION,
        response_json=json.dumps(failed_payload),
        tokens_used=None,
    )
    llm_shared.store_cached_row(
        conn,
        input_hash="hash-ok",
        task_name=llm_shared.EXTRACTION_TASK_NAME,
        model="gpt-5.4-nano",
        prompt_version=llm_shared.EXTRACTION_PROMPT_VERSION,
        response_json=json.dumps(ok_payload),
        tokens_used=12,
    )

    rows = llm_shared.fetch_cached_rows(
        conn,
        ["hash-failed", "hash-ok"],
        llm_shared.EXTRACTION_TASK_NAME,
        llm_shared.EXTRACTION_PROMPT_VERSION,
        exclude_retryable_failures=True,
    )

    assert set(rows) == {"hash-ok"}
    assert (
        llm_shared.fetch_cached_row(
            conn,
            input_hash="hash-failed",
            task_name=llm_shared.EXTRACTION_TASK_NAME,
            prompt_version=llm_shared.EXTRACTION_PROMPT_VERSION,
            exclude_retryable_failures=True,
        )
        is None
    )
