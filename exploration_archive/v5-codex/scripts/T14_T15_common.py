from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "unified.parquet"
SHARED_DIR = ROOT / "exploration" / "artifacts" / "shared"
TECH_MATRIX_PATH = SHARED_DIR / "swe_tech_matrix.parquet"
TEXT_PATH = SHARED_DIR / "swe_cleaned_text.parquet"
EMBED_PATH = SHARED_DIR / "swe_embeddings.npy"
EMBED_INDEX_PATH = SHARED_DIR / "swe_embedding_index.parquet"
STRUCTURED_SKILLS_PATH = SHARED_DIR / "asaniczka_structured_skills.parquet"
STOPLIST_PATH = SHARED_DIR / "company_stoplist.txt"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def linkedin_swe_meta(columns: Iterable[str]) -> pd.DataFrame:
    cols = ", ".join(columns)
    q = f"""
    SELECT {cols}
    FROM read_parquet('{DATA_PATH}')
    WHERE source_platform='linkedin'
      AND is_english = true
      AND date_flag = 'ok'
      AND is_swe = true
    """
    return conn().execute(q).fetchdf()


def linkedin_swe_full_meta(columns: Iterable[str] | None = None) -> pd.DataFrame:
    if columns is None:
        columns = [
            "uid",
            "source",
            "period",
            "seniority_final",
            "seniority_3level",
            "yoe_extracted",
            "description_length",
            "is_aggregator",
            "company_name_canonical",
            "company_name_effective",
            "company_name",
        ]
    return linkedin_swe_meta(columns)


def tech_columns() -> list[str]:
    df = conn().execute(
        f"DESCRIBE SELECT * FROM read_parquet('{TECH_MATRIX_PATH}')"
    ).fetchdf()
    return [c for c in df["column_name"].tolist() if c != "uid"]


def load_tech_matrix(columns: Iterable[str] | None = None) -> pd.DataFrame:
    if columns is None:
        cols = tech_columns()
    else:
        cols = list(columns)
    select_cols = ", ".join([f'"{c}"' for c in cols])
    q = f"""
    SELECT uid, {select_cols}
    FROM read_parquet('{TECH_MATRIX_PATH}')
    """
    return conn().execute(q).fetchdf()


def load_cleaned_text(columns: Iterable[str] | None = None) -> pd.DataFrame:
    if columns is None:
        cols = [
            "uid",
            "description_cleaned",
            "text_source",
            "source",
            "period",
            "seniority_final",
            "seniority_3level",
            "is_aggregator",
            "company_name_canonical",
            "metro_area",
            "yoe_extracted",
            "swe_classification_tier",
            "seniority_final_source",
        ]
    else:
        cols = list(columns)
    return conn().execute(
        f"SELECT {', '.join(cols)} FROM read_parquet('{TEXT_PATH}')"
    ).fetchdf()


def load_embeddings() -> tuple[np.ndarray, pd.DataFrame]:
    emb = np.load(EMBED_PATH, mmap_mode="r")
    idx = conn().execute(
        f"SELECT row_index, uid FROM read_parquet('{EMBED_INDEX_PATH}') ORDER BY row_index"
    ).fetchdf()
    return emb, idx


def load_structured_skills() -> pd.DataFrame:
    return conn().execute(
        f"SELECT uid, skill FROM read_parquet('{STRUCTURED_SKILLS_PATH}')"
    ).fetchdf()


def load_stoplist() -> set[str]:
    with STOPLIST_PATH.open() as fh:
        return {line.strip() for line in fh if line.strip()}


_ALIASES = {
    "c_plus_plus": {"c++", "cpp", "c plus plus", "c_plus_plus"},
    "c_sharp": {"c#", "c sharp", "c_sharp"},
    "dotnet": {".net", "dotnet", "asp.net", "aspnet"},
    "nodejs": {"node.js", "node js", "nodejs"},
    "nextjs": {"next.js", "next js", "nextjs"},
    "django_rest_framework": {"django rest framework", "drf", "django_rest_framework"},
    "gitlab_ci": {"gitlab ci", "gitlab-ci", "gitlab_ci"},
    "github_actions": {"github actions", "github-actions", "github_actions"},
    "ci_cd": {"ci/cd", "ci cd", "ci_cd"},
    "machine_learning": {"machine learning", "machine_learning"},
    "deep_learning": {"deep learning", "deep_learning"},
    "data_science": {"data science", "data_science"},
    "computer_vision": {"computer vision", "computer_vision"},
    "generative_ai": {"generative ai", "generative_ai"},
    "openai_api": {"openai api", "openai_api"},
    "anthropic_api": {"anthropic api", "anthropic_api"},
    "claude_api": {"claude api", "claude_api"},
    "gemini_api": {"gemini api", "gemini_api"},
    "prompt_engineering": {"prompt engineering", "prompt_engineering"},
    "fine_tuning": {"fine tuning", "fine_tuning"},
    "vector_db": {"vector db", "vector database", "vector databases", "vector_db"},
    "hugging_face": {"hugging face", "hugging_face"},
    "scikit_learn": {"scikit learn", "scikit-learn", "scikit_learn"},
    "r_language": {"r", "r language", "r_language"},
    "rest_api": {"rest api", "rest_api"},
    "microservices": {"micro services", "microservices"},
    "event_driven": {"event driven", "event_driven"},
}


def canonicalize_skill(text: str) -> str:
    s = text.strip().lower()
    if s in _ALIASES:
        return s
    s = s.replace("&", " and ")
    s = s.replace("+", " plus ")
    s = s.replace("#", " sharp ")
    s = s.replace("/", " ")
    s = s.replace(".", " ")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def skill_to_tech_col(skill: str, tech_cols: Iterable[str] | None = None) -> str | None:
    canon = canonicalize_skill(skill)
    if tech_cols is None:
        tech_cols = tech_columns()
    tech_set = set(tech_cols)
    if canon in tech_set:
        return canon
    for tech, variants in _ALIASES.items():
        if canon == tech or skill.strip().lower() in variants or canon in variants:
            if tech in tech_set:
                return tech
    return None


def validate_skill_aliases() -> None:
    assert canonicalize_skill("C++") in {"c_plus_plus", "c_plus_plus"}
    assert canonicalize_skill("C#") in {"c_sharp", "c_sharp"}
    assert canonicalize_skill(".NET") == "net"
    assert skill_to_tech_col("C++", ["c_plus_plus"]) == "c_plus_plus"
    assert skill_to_tech_col("C#", ["c_sharp"]) == "c_sharp"
    assert skill_to_tech_col("Node.js", ["nodejs"]) == "nodejs"
    assert skill_to_tech_col("Next.js", ["nextjs"]) == "nextjs"
    assert skill_to_tech_col("CI/CD", ["ci_cd"]) == "ci_cd"


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return matrix / denom


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def trimmed_centroid(x: np.ndarray, trim_frac: float = 0.10) -> np.ndarray:
    if x.shape[0] == 0:
        raise ValueError("empty group")
    if x.shape[0] == 1:
        return x[0]
    x = l2_normalize(np.asarray(x, dtype=np.float32))
    centroid = x.mean(axis=0)
    dists = 1.0 - (x @ centroid) / (
        np.linalg.norm(x, axis=1) * np.linalg.norm(centroid) + 1e-12
    )
    keep_n = max(1, int(math.ceil((1.0 - trim_frac) * x.shape[0])))
    keep_idx = np.argsort(dists)[:keep_n]
    return x[keep_idx].mean(axis=0)


def weighted_mean(x: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        return x.mean(axis=0)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() == 0:
        return x.mean(axis=0)
    return np.average(x, axis=0, weights=weights)


def top_n_series(series: pd.Series, n: int = 20) -> pd.Series:
    return series.sort_values(ascending=False).head(n)

