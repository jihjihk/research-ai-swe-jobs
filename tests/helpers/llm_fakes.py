import json
import subprocess


def completed_process(*, stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["fake"], returncode=returncode, stdout=stdout, stderr=stderr)


def codex_stdout(payload: dict, *, tokens_used: int | None = None) -> str:
    usage = None
    if tokens_used is not None:
        usage = {
            "input_tokens": max(tokens_used - 1, 0),
            "cached_input_tokens": 0,
            "output_tokens": min(tokens_used, 1),
        }

    lines = [
        json.dumps({"type": "thread.started", "thread_id": "thread_123"}, ensure_ascii=False),
        json.dumps({"type": "turn.started"}, ensure_ascii=False),
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_0",
                    "type": "agent_message",
                    "text": json.dumps(payload, ensure_ascii=False),
                },
            },
            ensure_ascii=False,
        ),
    ]
    if usage is not None:
        lines.append(
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": usage,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def claude_stdout(payload: dict, *, input_tokens: int = 0, output_tokens: int = 0) -> str:
    return json.dumps(
        {
            "result": json.dumps(payload, ensure_ascii=False),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        },
        ensure_ascii=False,
    )


def quota_error_text() -> str:
    return "429 rate limit exceeded"
