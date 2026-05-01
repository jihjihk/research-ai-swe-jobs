"""V2.6 — Independent T18 DiD replication.

Computes SWE vs control and SWE vs adjacent DiD on AI broad prevalence using
raw-description regex (the same methodology T18 used for cross-group analysis).
Also checks CIs via normal approximation.
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "exploration" / "tables" / "V2"
OUT.mkdir(parents=True, exist_ok=True)
UNIFIED = ROOT / "data" / "unified.parquet"

# T18 broad-AI regex (24-term union), applied to raw description via DuckDB regex.
# Mirror T18 patterns.
BROAD_REGEX = (
    r"\b(ai|artificial intelligence|machine learning|ml|deep learning|nlp|"
    r"llms?|generative ai|gen ai|rag|langchain|langgraph|copilot|claude|"
    r"anthropic|openai|gpt|chatgpt|gemini|agents?|agentic|vector db|"
    r"vector database|mcp|prompt engineering|fine-tuning)\b"
)

# Mutually exclusive groups
GROUPS = {
    "swe": "is_swe = TRUE",
    "adj": "is_swe_adjacent = TRUE",
    "ctrl": "is_control = TRUE",
}

DEFAULT_FILTER = "source_platform = 'linkedin' AND is_english = TRUE AND date_flag = 'ok'"


def main() -> None:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW u AS SELECT * FROM read_parquet('{UNIFIED}')")

    rows = []
    for g_name, g_filter in GROUPS.items():
        sql = f"""
        SELECT CASE WHEN period LIKE '2024%' THEN '2024' ELSE '2026' END AS year,
               COUNT(*) AS n,
               SUM(CASE WHEN regexp_matches(lower(description), '{BROAD_REGEX}') THEN 1 ELSE 0 END) AS hits
        FROM u
        WHERE {g_filter} AND {DEFAULT_FILTER}
        GROUP BY year
        ORDER BY year
        """
        df = con.execute(sql).fetchdf()
        df["group"] = g_name
        df["rate"] = df["hits"] / df["n"]
        rows.append(df)

    all_rows = pd.concat(rows, ignore_index=True)
    print("Group × period AI-broad rates:")
    print(all_rows)

    # Compute DiDs
    def get(group, year):
        r = all_rows[(all_rows["group"] == group) & (all_rows["year"] == year)].iloc[0]
        return float(r["rate"]), int(r["n"]), int(r["hits"])

    def var_p(n, hits):
        p = hits / n
        return p * (1 - p) / n

    # SWE-ctrl
    def did(g1, g2):
        p1_24, n1_24, h1_24 = get(g1, "2024")
        p1_26, n1_26, h1_26 = get(g1, "2026")
        p2_24, n2_24, h2_24 = get(g2, "2024")
        p2_26, n2_26, h2_26 = get(g2, "2026")
        did_val = (p1_26 - p1_24) - (p2_26 - p2_24)
        se = sqrt(
            var_p(n1_24, h1_24) + var_p(n1_26, h1_26)
            + var_p(n2_24, h2_24) + var_p(n2_26, h2_26)
        )
        ci_low = did_val - 1.96 * se
        ci_high = did_val + 1.96 * se
        return did_val, se, ci_low, ci_high

    results = []
    for label, (g1, g2) in [
        ("SWE_vs_ctrl", ("swe", "ctrl")),
        ("SWE_vs_adj", ("swe", "adj")),
        ("adj_vs_ctrl", ("adj", "ctrl")),
    ]:
        did_val, se, lo, hi = did(g1, g2)
        results.append(
            {
                "comparison": label,
                "did_pp": round(did_val * 100, 2),
                "se_pp": round(se * 100, 3),
                "ci95_low_pp": round(lo * 100, 2),
                "ci95_high_pp": round(hi * 100, 2),
                "ci_crosses_zero": lo < 0 < hi,
            }
        )
    r_df = pd.DataFrame(results)
    print("\nDiD results (broad AI):")
    print(r_df)
    all_rows.to_csv(OUT / "V2_6_group_period_rates.csv", index=False)
    r_df.to_csv(OUT / "V2_6_did.csv", index=False)


if __name__ == "__main__":
    main()
