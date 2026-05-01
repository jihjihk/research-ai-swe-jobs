"""V2.3 — Independent T21 orchestration density re-derivation.

Loads validated patterns from validated_mgmt_patterns.json and re-computes
density per 1K chars for mid-senior and director in 2024 vs 2026 using
the shared cleaned text (text_source='llm').

Also runs the AI × senior interaction analysis and a 50-row precision audit.
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

UNIFIED = ROOT / "data" / "unified.parquet"
TECH = ROOT / "exploration" / "artifacts" / "shared" / "swe_tech_matrix.parquet"
MGMT = ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json"


def main() -> None:
    patterns = json.loads(MGMT.read_text())
    people_rx = re.compile(patterns["profiles"]["people_mgmt"]["regex"], re.IGNORECASE)
    orch_rx = re.compile(patterns["profiles"]["tech_orch"]["regex"], re.IGNORECASE)
    strat_rx = re.compile(patterns["profiles"]["strategic"]["regex"], re.IGNORECASE)

    con = duckdb.connect()
    # Pull senior rows from unified with description_core_llm
    sql = """
    SELECT uid, CASE WHEN period LIKE '2024%' THEN '2024' ELSE '2026' END AS year,
           seniority_final, description_core_llm
    FROM read_parquet('data/unified.parquet')
    WHERE is_swe = TRUE AND source_platform = 'linkedin'
      AND is_english = TRUE AND date_flag = 'ok'
      AND llm_extraction_coverage = 'labeled'
      AND seniority_final IN ('mid-senior', 'director', 'entry')
      AND description_core_llm IS NOT NULL
    """
    df = con.execute(sql).fetchdf()
    print(f"Loaded {len(df)} rows")
    print(df.groupby(["year", "seniority_final"]).size())

    # Compute per-posting densities
    def count_density(text, rx):
        n = len(rx.findall(text))
        denom = max(len(text), 200) / 1000
        return n / denom

    # For performance on 40k rows, use a loop with precompiled regex
    people_d = np.zeros(len(df))
    orch_d = np.zeros(len(df))
    strat_d = np.zeros(len(df))
    people_any = np.zeros(len(df), dtype=bool)
    orch_any = np.zeros(len(df), dtype=bool)
    strat_any = np.zeros(len(df), dtype=bool)

    for i, text in enumerate(df["description_core_llm"].fillna("").values):
        people_d[i] = count_density(text, people_rx)
        orch_d[i] = count_density(text, orch_rx)
        strat_d[i] = count_density(text, strat_rx)
        people_any[i] = people_rx.search(text) is not None
        orch_any[i] = orch_rx.search(text) is not None
        strat_any[i] = strat_rx.search(text) is not None

    df["people_density"] = people_d
    df["orch_density"] = orch_d
    df["strat_density"] = strat_d
    df["people_any"] = people_any
    df["orch_any"] = orch_any
    df["strat_any"] = strat_any

    agg = df.groupby(["year", "seniority_final"]).agg(
        n=("uid", "count"),
        people_density=("people_density", "mean"),
        orch_density=("orch_density", "mean"),
        strat_density=("strat_density", "mean"),
        people_any=("people_any", "mean"),
        orch_any=("orch_any", "mean"),
        strat_any=("strat_any", "mean"),
    ).round(4)
    print("\nDensity by period × seniority:")
    print(agg)
    agg.to_csv(OUT / "V2_3_density_by_period_seniority.csv")

    # T21 reported:
    # Mid-senior: people 0.186 → 0.232 (+25%); orch 0.168 → 0.332 (+98%)
    # Director: people 0.228 → 0.181 (−21%); orch 0.118 → 0.302 (+156%)
    # Report deltas and % change
    ms24 = agg.loc[("2024", "mid-senior")]
    ms26 = agg.loc[("2026", "mid-senior")]
    d24 = agg.loc[("2024", "director")]
    d26 = agg.loc[("2026", "director")]
    cmp_rows = [
        {
            "cell": "mid-senior people",
            "v2_2024": round(ms24["people_density"], 4),
            "v2_2026": round(ms26["people_density"], 4),
            "v2_delta_pct": round(100 * (ms26["people_density"] - ms24["people_density"]) / ms24["people_density"], 1),
            "t21_2024": 0.186, "t21_2026": 0.232, "t21_pct": 25,
        },
        {
            "cell": "mid-senior orch",
            "v2_2024": round(ms24["orch_density"], 4),
            "v2_2026": round(ms26["orch_density"], 4),
            "v2_delta_pct": round(100 * (ms26["orch_density"] - ms24["orch_density"]) / ms24["orch_density"], 1),
            "t21_2024": 0.168, "t21_2026": 0.332, "t21_pct": 98,
        },
        {
            "cell": "director people",
            "v2_2024": round(d24["people_density"], 4),
            "v2_2026": round(d26["people_density"], 4),
            "v2_delta_pct": round(100 * (d26["people_density"] - d24["people_density"]) / d24["people_density"], 1),
            "t21_2024": 0.228, "t21_2026": 0.181, "t21_pct": -21,
        },
        {
            "cell": "director orch",
            "v2_2024": round(d24["orch_density"], 4),
            "v2_2026": round(d26["orch_density"], 4),
            "v2_delta_pct": round(100 * (d26["orch_density"] - d24["orch_density"]) / d24["orch_density"], 1),
            "t21_2024": 0.118, "t21_2026": 0.302, "t21_pct": 156,
        },
    ]
    cmp_df = pd.DataFrame(cmp_rows)
    print("\nT21 comparison:")
    print(cmp_df)
    cmp_df.to_csv(OUT / "V2_3_t21_comparison.csv", index=False)

    # AI × senior interaction: need AI mention from tech matrix
    tsql = """
    SELECT uid, (llm OR rag OR agents_framework OR copilot OR claude_api OR claude_tool OR cursor_tool OR gemini_tool OR codex_tool OR chatgpt OR openai_api OR prompt_engineering OR fine_tuning OR mcp OR embedding OR transformer_arch OR machine_learning OR deep_learning OR pytorch OR tensorflow OR langchain OR langgraph OR nlp OR huggingface)::INT AS any_ai
    FROM read_parquet('exploration/artifacts/shared/swe_tech_matrix.parquet')
    """
    tdf = con.execute(tsql).fetchdf()
    df2 = df.merge(tdf, on="uid", how="left")
    senior = df2[df2["seniority_final"].isin(["mid-senior", "director"])]
    ai_inter = senior.groupby(["year", "any_ai"]).agg(
        n=("uid", "count"),
        orch_density=("orch_density", "mean"),
        people_density=("people_density", "mean"),
    ).round(4)
    print("\nAI × senior interaction (orch & people density):")
    print(ai_inter)
    ai_inter.to_csv(OUT / "V2_3_ai_senior_interaction.csv")

    # T21 reports: in 2026, AI-mentioning senior postings have orch density +76% vs non-AI
    # And people density identical.
    if (len(ai_inter) >= 4):
        n26_ai = ai_inter.loc[("2026", 1)]
        n26_no = ai_inter.loc[("2026", 0)]
        print(f"\n2026 AI vs no-AI senior:")
        print(f"  orch: {n26_no['orch_density']:.4f} (no-AI) vs {n26_ai['orch_density']:.4f} (AI) = {100*(n26_ai['orch_density']-n26_no['orch_density'])/n26_no['orch_density']:.1f}% uplift")
        print(f"  people: {n26_no['people_density']:.4f} (no-AI) vs {n26_ai['people_density']:.4f} (AI) = {100*(n26_ai['people_density']-n26_no['people_density'])/n26_no['people_density']:.1f}% uplift")

    # --- Precision audit: sample 50 postings with orch pattern in 2026 and 50 in 2024
    orch_2026 = df[(df["year"] == "2026") & (df["orch_any"])].sample(n=min(50, df[df["year"] == "2026"].orch_any.sum()), random_state=7)
    orch_2024 = df[(df["year"] == "2024") & (df["orch_any"])].sample(n=min(50, df[df["year"] == "2024"].orch_any.sum()), random_state=7)

    audit_rows = []
    for label, sub in [("2026", orch_2026), ("2024", orch_2024)]:
        for _, row in sub.iterrows():
            text = row["description_core_llm"] or ""
            matches = orch_rx.findall(text)
            # Grab surrounding context for each match
            contexts = []
            for m in orch_rx.finditer(text):
                s = max(0, m.start() - 40)
                e = min(len(text), m.end() + 40)
                contexts.append(text[s:e].replace("\n", " "))
            audit_rows.append({
                "year": label,
                "uid": row["uid"],
                "n_matches": len(matches),
                "contexts": " || ".join(contexts[:3]),
            })
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(OUT / "V2_3_orch_50row_audit.csv", index=False)
    print(f"\nWrote {len(audit_df)} audit rows to V2_3_orch_50row_audit.csv")
    print("Sample contexts (2026):")
    for _, r in orch_2026.head(5).iterrows():
        text = r["description_core_llm"] or ""
        m = orch_rx.search(text)
        if m:
            s = max(0, m.start() - 40)
            e = min(len(text), m.end() + 40)
            print(" ", text[s:e].replace("\n", " "))


if __name__ == "__main__":
    main()
