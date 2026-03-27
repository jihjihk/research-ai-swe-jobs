import importlib.util
import sys
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def load_module(module_name: str, relative_path: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    script_dir = str(path.parent)
    added_path = False
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        added_path = True
    try:
        spec.loader.exec_module(module)
    finally:
        if added_path:
            sys.path.remove(script_dir)
    return module
