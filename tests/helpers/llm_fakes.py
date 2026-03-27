import json
import subprocess


def completed_process(*, stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["fake"], returncode=returncode, stdout=stdout, stderr=stderr)


def codex_stdout(payload: dict, *, tokens_used: int | None = None) -> str:
    suffix = "" if tokens_used is None else f"\nTokens used {tokens_used}"
    return json.dumps(payload, ensure_ascii=False) + suffix


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
