"""V2.1 — Independent re-derivation of T23 lead-finding.

Computes posting-side AI rates (broad / narrow / tool / per-tool) for 2024 and 2026
SWE LinkedIn under default filters, then compares against the SO 2025 worker
benchmarks and runs the 50-85% sensitivity sweep.

Outputs: exploration/tables/V2/V2_1_*.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "exploration" / "tables" / "V2"
OUT.mkdir(parents=True, exist_ok=True)

UNIFIED = ROOT / "data" / "unified.parquet"
TECH = ROOT / "exploration" / "artifacts" / "shared" / "swe_tech_matrix.parquet"
CLEAN = ROOT / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet"

DEFAULT = (
    "is_swe = TRUE AND source_platform = 'linkedin' "
    "AND is_english = TRUE AND date_flag = 'ok'"
)

# T14/T23 broad 24-term set, using the available tech_matrix columns
BROAD_AI_COLS = [
    "llm",
    "rag",
    "agents_framework",
    "copilot",
    "claude_api",
    "claude_tool",
    "cursor_tool",
    "gemini_tool",
    "codex_tool",
    "chatgpt",
    "openai_api",
    "prompt_engineering",
    "fine_tuning",
    "mcp",
    "embedding",
    "transformer_arch",
    "machine_learning",
    "deep_learning",
    "pytorch",
    "tensorflow",
    "langchain",
    "langgraph",
    "nlp",
    "huggingface",
]

# AI-as-tool subset per T23
AI_TOOL_COLS = ["copilot", "cursor_tool", "claude_tool", "chatgpt", "prompt_engineering"]


def main() -> None:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW u AS SELECT * FROM read_parquet('{UNIFIED}')")
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{TECH}')")

    # -- Build the master join: uid + period + all tech columns of interest
    # The tech matrix is already SWE-LinkedIn default-filtered (n=63,701).
    # Confirm row count.
    n_total = con.execute(
        f"SELECT COUNT(*) FROM u WHERE {DEFAULT}"
    ).fetchone()[0]
    print(f"SWE LinkedIn default rows in unified: {n_total}")

    n_tech = con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    print(f"Tech matrix rows: {n_tech}")

    # Per-period posting AI rates
    broad_expr = " OR ".join([f"t.{c}" for c in BROAD_AI_COLS])
    tool_expr = " OR ".join([f"t.{c}" for c in AI_TOOL_COLS])

    # Narrow AI: LIKE '%\bai\b%' or artificial intelligence, on description text
    # Use description_core_llm when available, otherwise description; case-insensitive.
    narrow_sql = f"""
    SELECT CASE WHEN u.period LIKE '2024%' THEN '2024' ELSE '2026' END AS year,
        COUNT(*) AS n,
        SUM(CASE WHEN regexp_matches(lower(COALESCE(u.description_core_llm, u.description)),
            '(^|[^a-z])(ai|artificial intelligence)([^a-z]|$)') THEN 1 ELSE 0 END)::DOUBLE/COUNT(*) AS narrow_ai_rate
    FROM u
    WHERE {DEFAULT}
    GROUP BY year
    ORDER BY year
    """
    narrow_df = con.execute(narrow_sql).fetchdf()
    print("\nNarrow AI by period:")
    print(narrow_df)

    # Broad / tool / per-tool using tech matrix
    period_sql = f"""
    WITH j AS (
      SELECT u.uid,
        CASE WHEN u.period LIKE '2024%' THEN '2024' ELSE '2026' END AS year,
        (CASE WHEN {broad_expr} THEN 1 ELSE 0 END) AS broad_ai,
        (CASE WHEN {tool_expr} THEN 1 ELSE 0 END) AS ai_tool,
        t.copilot::INT AS copilot,
        t.cursor_tool::INT AS cursor_tool,
        t.claude_tool::INT AS claude_tool,
        t.chatgpt::INT AS chatgpt
      FROM u JOIN t USING (uid)
      WHERE {DEFAULT}
    )
    SELECT year, COUNT(*) AS n,
      AVG(broad_ai) AS broad_ai_rate,
      AVG(ai_tool) AS ai_tool_rate,
      AVG(copilot) AS copilot_rate,
      AVG(cursor_tool) AS cursor_tool_rate,
      AVG(claude_tool) AS claude_tool_rate,
      AVG(chatgpt) AS chatgpt_rate
    FROM j
    GROUP BY year ORDER BY year
    """
    broad_df = con.execute(period_sql).fetchdf()
    print("\nBroad/tool AI by period:")
    print(broad_df)

    merged = narrow_df.merge(broad_df, on=["year", "n"], how="outer")
    merged.to_csv(OUT / "V2_1_posting_ai_rates.csv", index=False)

    # T23 reported values for 2026
    t23_2026 = {
        "broad_ai_rate": 0.286,
        "narrow_ai_rate": 0.346,
        "ai_tool_rate": 0.0687,
        "copilot_rate": 0.0377,
        "chatgpt_rate": 0.0061,
        "claude_tool_rate": 0.0337,
        "cursor_tool_rate": 0.0191,
    }
    row2026 = merged[merged.year == "2026"].iloc[0]
    compare = []
    for k, v in t23_2026.items():
        mine = float(row2026[k])
        compare.append({
            "metric": k,
            "v2_rederived": round(mine, 4),
            "t23_reported": v,
            "abs_diff_pp": round(abs(mine - v) * 100, 3),
            "within_1pp": abs(mine - v) <= 0.01,
        })
    compare_df = pd.DataFrame(compare)
    print("\nT23 vs V2 re-derivation (2026 posting rates):")
    print(compare_df)
    compare_df.to_csv(OUT / "V2_1_posting_comparison_2026.csv", index=False)

    # Divergence computation
    workers = {
        "broad_ai": {"rate": 0.808, "label": "SO2025 any-use"},
        "narrow_ai": {"rate": 0.808, "label": "SO2025 any-use"},
        "ai_tool": {"rate": 0.808, "label": "SO2025 any-use"},
        "copilot": {"rate": 0.549, "label": "SO2025 copilot share × any-use"},
        "chatgpt": {"rate": 0.660, "label": "SO2025 chatgpt share × any-use"},
        "claude_tool": {"rate": 0.330, "label": "SO2025 claude share × any-use"},
    }
    rows = []
    for metric in ["broad_ai", "narrow_ai", "ai_tool", "copilot", "chatgpt", "claude_tool"]:
        col = metric + "_rate"
        mine = float(row2026[col])
        w = workers[metric]["rate"]
        rows.append(
            {
                "metric": metric,
                "posting_2026": round(mine, 4),
                "worker_rate": w,
                "worker_label": workers[metric]["label"],
                "gap_pp": round((mine - w) * 100, 2),
                "ratio_posting_over_worker": round(mine / w, 4) if w else None,
                "worker_over_posting": round(w / mine, 2) if mine > 0 else None,
            }
        )
    div_df = pd.DataFrame(rows)
    print("\nDivergence (V2 re-derived):")
    print(div_df)
    div_df.to_csv(OUT / "V2_1_divergence.csv", index=False)

    # Sensitivity sweep
    sens = []
    broad_rate = float(row2026["broad_ai_rate"])
    narrow_rate = float(row2026["narrow_ai_rate"])
    tool_rate = float(row2026["ai_tool_rate"])
    for w in [0.50, 0.65, 0.75, 0.808, 0.85]:
        sens.append(
            {
                "worker_assumption": w,
                "gap_broad": round((broad_rate - w) * 100, 2),
                "gap_narrow": round((narrow_rate - w) * 100, 2),
                "gap_tool": round((tool_rate - w) * 100, 2),
                "direction_holds_broad": broad_rate < w,
                "direction_holds_narrow": narrow_rate < w,
                "direction_holds_tool": tool_rate < w,
            }
        )
    sens_df = pd.DataFrame(sens)
    print("\nSensitivity sweep:")
    print(sens_df)
    sens_df.to_csv(OUT / "V2_1_sensitivity.csv", index=False)

    print("\nDone. Outputs under", OUT)


if __name__ == "__main__":
    main()
