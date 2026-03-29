import json
from datetime import datetime
import logging
from zoneinfo import ZoneInfo

import pytest

from tests.helpers.imports import load_module
from tests.helpers.llm_fakes import codex_stdout


llm_shared = load_module("llm_shared_runtime", "preprocessing/scripts/llm_shared.py")


PACIFIC = ZoneInfo("America/Los_Angeles")


@pytest.mark.unit
def test_parse_engine_list_and_tiers_defaults_and_validates():
    assert llm_shared.parse_engine_list("codex,claude") == ("codex", "claude")

    assert llm_shared.parse_engine_tiers(None, ("codex", "claude")) == {
        "codex": "full",
        "claude": "full",
    }
    assert llm_shared.parse_engine_tiers(
        "codex=full,claude=non_intrusive",
        ("codex", "claude"),
    ) == {
        "codex": "full",
        "claude": "non_intrusive",
    }

    with pytest.raises(ValueError, match="unsupported engine tier"):
        llm_shared.parse_engine_tiers("codex=unknown", ("codex",))


@pytest.mark.unit
def test_format_engine_labels_and_progress_checkpoints_are_compact():
    engines = llm_shared.build_engine_configs(
        ("codex", "claude"),
        codex_model="gpt-5.4-mini",
        claude_model="haiku",
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
        "--full-auto",
        "--config",
        "model=gpt-5.4-mini",
        "--skip-git-repo-check",
        "--json",
        "Return JSON",
    ]


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
def test_engine_runtime_keeps_quota_pauses_provider_scoped():
    runtime = llm_shared.LLMEngineRuntime(
        llm_shared.build_engine_configs(
            ("codex", "claude"),
            codex_model="gpt-5.4-mini",
            claude_model="haiku",
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
