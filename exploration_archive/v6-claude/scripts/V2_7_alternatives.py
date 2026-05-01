"""V2.7 — Alternative explanation checks.

(1) Does the T21 director orchestration +156% rise hold when excluding top 10 AI-
    mentioning 2026 companies?
(2) Does the T16 within-company AI decomposition 92% finding hold on zero-2024 companies only?
    (already computed in V2.2)
(3) Does the T29 length flip depend on bullet density? (tested in V2.5)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "exploration" / "tables" / "V2"
OUT.mkdir(parents=True, exist_ok=True)

MGMT = ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json"


def main() -> None:
    patterns = json.loads(MGMT.read_text())
    orch_rx = re.compile(patterns["profiles"]["tech_orch"]["regex"], re.IGNORECASE)

    con = duckdb.connect()
    # Get director rows with text and company + ai-mention
    sql = """
    SELECT u.uid, u.company_name_canonical AS co,
        CASE WHEN u.period LIKE '2024%' THEN '2024' ELSE '2026' END AS year,
        u.description_core_llm AS text,
        (t.llm OR t.rag OR t.agents_framework OR t.copilot OR t.claude_api OR t.claude_tool
         OR t.cursor_tool OR t.chatgpt OR t.openai_api OR t.prompt_engineering OR t.fine_tuning
         OR t.machine_learning OR t.deep_learning OR t.pytorch OR t.tensorflow OR t.langchain)::INT AS any_ai
    FROM read_parquet('data/unified.parquet') u
    JOIN read_parquet('exploration/artifacts/shared/swe_tech_matrix.parquet') t USING (uid)
    WHERE u.is_swe AND u.source_platform='linkedin' AND u.is_english AND u.date_flag='ok'
      AND u.llm_extraction_coverage='labeled'
      AND u.seniority_final='director'
      AND u.description_core_llm IS NOT NULL
    """
    df = con.execute(sql).fetchdf()
    print(f"Director rows: {len(df)}")
    print(df.groupby("year").size())

    # Compute orch density
    def density(text):
        n = len(orch_rx.findall(text))
        return n / (max(len(text), 200) / 1000)

    df["orch_density"] = df["text"].fillna("").map(density)

    base = df.groupby("year")["orch_density"].mean()
    print(f"\nDirector orch density by year: {base.to_dict()}")
    # Top 10 AI-mentioning 2026 companies (by number of AI director rows)
    top_ai_cos = (
        df[(df["year"] == "2026") & (df["any_ai"] == 1)]
        .groupby("co").size().sort_values(ascending=False).head(10)
    )
    print(f"\nTop 10 AI-mentioning 2026 director companies:")
    print(top_ai_cos)

    excluded = df[~df["co"].isin(top_ai_cos.index)]
    base_excl = excluded.groupby("year")["orch_density"].mean()
    print(f"\nAfter excluding top 10 AI director companies:")
    print(base_excl)

    # Percent change
    full_pct = 100 * (base["2026"] - base["2024"]) / base["2024"]
    excl_pct = 100 * (base_excl["2026"] - base_excl["2024"]) / base_excl["2024"]
    print(f"\nFull director orch rise: +{full_pct:.0f}%")
    print(f"Excluding top 10 AI director cos: +{excl_pct:.0f}%")

    pd.DataFrame([
        {"label": "full", "dir_2024": round(base["2024"], 4), "dir_2026": round(base["2026"], 4), "pct_change": round(full_pct, 1)},
        {"label": "ex_top10_ai", "dir_2024": round(base_excl["2024"], 4), "dir_2026": round(base_excl["2026"], 4), "pct_change": round(excl_pct, 1)},
    ]).to_csv(OUT / "V2_7_director_orch_alt.csv", index=False)


if __name__ == "__main__":
    main()
