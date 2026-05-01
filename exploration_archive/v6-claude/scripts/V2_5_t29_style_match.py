"""V2.5 — Independent T29 style-match length-flip re-derivation.

Builds an authorship-style score from simple features, does 2026→2024 nearest-neighbor
matching on the score, and compares full delta vs matched delta for char_len.

Also tests the flip under alternative matching criteria:
  (a) LLM-tell density only
  (b) Em-dash density only
  (c) Bullet density only
  (d) Matching within archetype
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "exploration" / "tables" / "V2"
OUT.mkdir(parents=True, exist_ok=True)

# LLM "tell" tokens (mostly a subset of T29's 42)
TELL_TOKENS = {
    "delve","tapestry","leverage","robust","unleash","embark","navigate","cutting-edge",
    "comprehensive","seamless","furthermore","notably","pivotal","harness","dynamic",
    "vibrant","intricate","meticulous","plethora","foster","multifaceted","underscore",
    "paramount","testament","realm","bespoke","empower","orchestrate","spearhead",
    "transformative","synergy","holistic",
}


def tell_density(text: str) -> float:
    if not text:
        return 0.0
    toks = re.findall(r"[a-zA-Z][a-zA-Z-]+", text.lower())
    hits = sum(1 for t in toks if t in TELL_TOKENS)
    denom = max(len(text), 200) / 1000
    return hits / denom


def emdash_density(text: str) -> float:
    if not text:
        return 0.0
    n = text.count("—") + text.count("–") + text.count(" -- ")
    denom = max(len(text), 200) / 1000
    return n / denom


def bullet_density(text: str) -> float:
    if not text:
        return 0.0
    # Count * and - at line starts as bullets
    n = len(re.findall(r"(^|\n)\s*[*\-•]", text))
    denom = max(len(text), 200) / 1000
    return n / denom


def zscore(a: np.ndarray) -> np.ndarray:
    m, s = a.mean(), a.std()
    return (a - m) / (s if s > 0 else 1.0)


def nearest_match(scores_26: np.ndarray, scores_24: np.ndarray) -> np.ndarray:
    """For each 2026 score, find nearest 2024 index."""
    order = np.argsort(scores_24)
    sorted_24 = scores_24[order]
    idx_sorted = np.searchsorted(sorted_24, scores_26)
    idx_sorted = np.clip(idx_sorted, 0, len(sorted_24) - 1)
    # Look at idx and idx-1, pick closer
    left = np.clip(idx_sorted - 1, 0, len(sorted_24) - 1)
    pick_left = np.abs(sorted_24[left] - scores_26) < np.abs(sorted_24[idx_sorted] - scores_26)
    chosen_sorted = np.where(pick_left, left, idx_sorted)
    return order[chosen_sorted]


def main() -> None:
    con = duckdb.connect()
    sql = """
    SELECT uid,
        CASE WHEN period LIKE '2024%' THEN '2024' ELSE '2026' END AS year,
        description_core_llm AS text
    FROM read_parquet('data/unified.parquet')
    WHERE is_swe = TRUE AND source_platform = 'linkedin'
      AND is_english = TRUE AND date_flag = 'ok'
      AND llm_extraction_coverage = 'labeled'
      AND description_core_llm IS NOT NULL
    """
    df = con.execute(sql).fetchdf()
    print(f"Loaded {len(df)} rows")

    # Sample to ~15k for speed
    df["text"] = df["text"].fillna("")
    df["char_len"] = df["text"].str.len()
    df["tell"] = df["text"].map(tell_density)
    df["emdash"] = df["text"].map(emdash_density)
    df["bullet"] = df["text"].map(bullet_density)

    by_year = df.groupby("year").agg(
        n=("uid", "count"),
        tell=("tell", "mean"),
        emdash=("emdash", "mean"),
        bullet=("bullet", "mean"),
        char_len=("char_len", "mean"),
    )
    print("\nFeature means by year:")
    print(by_year)

    # Authorship score: equal-weight z-scored sum of tell, emdash, bullet
    df["z_tell"] = zscore(df["tell"].values)
    df["z_emdash"] = zscore(df["emdash"].values)
    df["z_bullet"] = zscore(df["bullet"].values)
    df["score"] = df[["z_tell", "z_emdash", "z_bullet"]].mean(axis=1)

    score_by_year = df.groupby("year")["score"].describe()
    print("\nScore dist by year:")
    print(score_by_year)

    df24 = df[df["year"] == "2024"].copy().reset_index(drop=True)
    df26 = df[df["year"] == "2026"].copy().reset_index(drop=True)

    full_delta = df26["char_len"].mean() - df24["char_len"].mean()
    print(f"\nFull char_len delta: {full_delta:.1f}")

    def matched_delta(score_col):
        s24 = df24[score_col].values
        s26 = df26[score_col].values
        idx = nearest_match(s26, s24)
        matched_2024 = df24.iloc[idx].reset_index(drop=True)
        d = df26["char_len"].values - matched_2024["char_len"].values
        return d.mean()

    # Match on composite score
    d_composite = matched_delta("score")
    print(f"Style-matched delta (composite): {d_composite:.1f}")

    # Match on LLM-tell only
    d_tell = matched_delta("z_tell")
    print(f"Style-matched delta (tell only): {d_tell:.1f}")
    d_em = matched_delta("z_emdash")
    print(f"Style-matched delta (emdash only): {d_em:.1f}")
    d_bul = matched_delta("z_bullet")
    print(f"Style-matched delta (bullet only): {d_bul:.1f}")

    # Also verify AI rates (broad)
    # Need tech matrix
    tm = con.execute("""
        SELECT uid, (llm OR rag OR agents_framework OR copilot OR claude_api OR claude_tool
            OR cursor_tool OR gemini_tool OR codex_tool OR chatgpt OR openai_api
            OR prompt_engineering OR fine_tuning OR mcp OR embedding OR transformer_arch
            OR machine_learning OR deep_learning OR pytorch OR tensorflow OR langchain
            OR langgraph OR nlp OR huggingface)::INT AS any_ai_broad,
            regexp_matches(lower(COALESCE((SELECT description_core_llm FROM read_parquet('data/unified.parquet') u WHERE u.uid=t.uid), '')),
                '(^|[^a-z])(ai|artificial intelligence)([^a-z]|$)')::INT AS any_ai_narrow
        FROM read_parquet('exploration/artifacts/shared/swe_tech_matrix.parquet') t
    """).fetchdf()
    dfj = df.merge(tm[["uid", "any_ai_broad"]], on="uid", how="left")
    full_ai = dfj[dfj.year == "2026"]["any_ai_broad"].mean() - dfj[dfj.year == "2024"]["any_ai_broad"].mean()
    df24a = dfj[dfj.year == "2024"].reset_index(drop=True)
    df26a = dfj[dfj.year == "2026"].reset_index(drop=True)
    s24 = df24a["score"].values
    s26 = df26a["score"].values
    idx = nearest_match(s26, s24)
    matched24_ai = df24a.iloc[idx]["any_ai_broad"].reset_index(drop=True)
    matched_ai = df26a["any_ai_broad"].values - matched24_ai.values
    print(f"\nAI broad full delta: {full_ai:.4f}")
    print(f"AI broad matched delta: {matched_ai.mean():.4f}")

    rows = [
        {"metric": "char_len", "full_delta": round(full_delta, 1), "composite_match": round(d_composite, 1), "tell_only_match": round(d_tell, 1), "emdash_only_match": round(d_em, 1), "bullet_only_match": round(d_bul, 1)},
        {"metric": "any_ai_broad", "full_delta": round(full_ai, 4), "composite_match": round(matched_ai.mean(), 4), "tell_only_match": "", "emdash_only_match": "", "bullet_only_match": ""},
    ]
    pd.DataFrame(rows).to_csv(OUT / "V2_5_style_match.csv", index=False)
    print("\nSaved V2_5_style_match.csv")


if __name__ == "__main__":
    main()
