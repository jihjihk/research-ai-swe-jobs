from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def temp_stage_dirs(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "workspace"
    dirs = {
        "root": root,
        "data": root / "data",
        "intermediate": root / "preprocessing" / "intermediate",
        "logs": root / "preprocessing" / "logs",
        "cache": root / "preprocessing" / "cache",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
