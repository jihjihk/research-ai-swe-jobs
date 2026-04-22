"""
S27 v2 — Spot-checks for composite B threads.

Pulls 10 example postings per thread (Applied-AI title, Applied-AI cluster,
FDE, Legacy-substitution neighbors) to verify headline numbers are real.

Outputs eda/tables/S27_v2_spotcheck_*.csv
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE = str(PROJECT_ROOT / "data" / "unified_core.parquet")
TABLES = PROJECT_ROOT / "eda" / "tables"

AI_VOCAB_PATTERN = (
    r"(?i)\b(llm|gpt|chatgpt|claude|copilot|openai|anthropic|gemini|bard|mistral|"
    r"llama|large\ language\ model|generative\ ai|genai|gen\ ai|foundation\ model|"
    r"transformer\ model|ai\ agent|agentic|ai\-powered|ai\ tooling|ai\-assisted|"
    r"rag|retrieval\ augmented|vector\ database|vector\ store|embedding\ model|"
    r"prompt\ engineering|prompt\ engineer|ml\ ops|mlops|llmops|cursor\ ide|"
    r"windsurf\ ide|github\ copilot)\b"
)

APPLIED_AI_TITLE_REGEX = (
    r"(?i)\b(applied\s+ai|applied\s+ml|ai\s+engineer|ml\s+engineer|llm\s+engineer|"
    r"machine\s+learning\s+engineer|mlops\s+engineer|genai\s+engineer|"
    r"generative\s+ai\s+engineer|foundation\s+model\s+engineer|"
    r"agent(?:ic)?\s+engineer|ai/ml\s+engineer)\b"
)

FDE_TITLE_REGEX = r"(?i)forward[\s\-]?deployed"


def grab(con, sql, label):
    df = con.execute(sql).df()
    df["thread"] = label
    return df


def main():
    con = duckdb.connect()
    cols = ("uid", "title", "company_name_canonical", "period",
            "seniority_3level", "yoe_min_years_llm", "metro_area")

    select_cols = ", ".join(cols) + ", LEFT(description, 500) AS desc_excerpt"
    n = 10

    # Thread 1a — Applied-AI title regex (2026 senior)
    sql1a = f"""
      SELECT {select_cols}
      FROM '{CORE}'
      WHERE is_swe AND is_english AND date_flag='ok'
        AND period LIKE '2026%' AND seniority_3level='senior'
        AND regexp_matches(LOWER(COALESCE(title,'')), '{APPLIED_AI_TITLE_REGEX}')
      ORDER BY HASH(uid) LIMIT {n}
    """

    # Thread 1b — v9 T34 cluster 0 exemplars (description-only AI signal)
    sql1b = f"""
      WITH labels AS (
        SELECT * FROM read_parquet(
          'exploration-archive/v9_final_opus_47/artifacts/shared/swe_archetype_labels.parquet'
        )
      )
      SELECT {select_cols}
      FROM '{CORE}' u
      INNER JOIN labels l USING (uid)
      WHERE u.is_swe AND u.is_english AND u.date_flag='ok'
        AND u.period LIKE '2026%'
        AND l.archetype_name = 'models/systems/llm'
      ORDER BY HASH(u.uid) LIMIT {n}
    """

    # Thread 2 — FDE 2026
    sql2 = f"""
      SELECT {select_cols}
      FROM '{CORE}'
      WHERE is_swe AND is_english AND date_flag='ok'
        AND period LIKE '2026%'
        AND regexp_matches(COALESCE(title,''), '{FDE_TITLE_REGEX}')
      ORDER BY HASH(uid) LIMIT {n}
    """

    # Thread 3 — emerging cluster (systems/agent/workflows) 2026
    sql3 = f"""
      WITH labels AS (
        SELECT * FROM read_parquet(
          'exploration-archive/v9_final_opus_47/artifacts/shared/swe_archetype_labels.parquet'
        )
      )
      SELECT {select_cols}
      FROM '{CORE}' u
      INNER JOIN labels l USING (uid)
      WHERE u.is_swe AND u.is_english AND u.date_flag='ok'
        AND u.period LIKE '2026%'
        AND l.archetype_name = 'systems/agent/workflows'
      ORDER BY HASH(u.uid) LIMIT {n}
    """

    # Thread 4 — Legacy-neighbor postings (java developer, devops engineer, etc.)
    legacy_neighbors = ("java developer", "devops engineer", "web developer",
                        "full stack java developer", "database engineer",
                        "big data engineer")
    in_clause = ",".join("'" + s + "'" for s in legacy_neighbors)
    sql4 = f"""
      SELECT {select_cols},
             regexp_matches(description, '{AI_VOCAB_PATTERN}') AS ai_match
      FROM '{CORE}'
      WHERE is_swe AND is_english AND date_flag='ok'
        AND period LIKE '2026%'
        AND LOWER(TRIM(title)) IN ({in_clause})
      ORDER BY HASH(uid) LIMIT {n}
    """

    out = pd.concat([
        grab(con, sql1a, "1a_applied_ai_title_senior_2026"),
        grab(con, sql1b, "1b_v9_T34_cluster_models_systems_llm_2026"),
        grab(con, sql2, "2_fde_title_2026"),
        grab(con, sql3, "3_v9_systems_agent_workflows_2026"),
        grab(con, sql4, "4_legacy_neighbors_2026"),
    ], ignore_index=True)

    out.to_csv(TABLES / "S27_v2_spotchecks.csv", index=False)
    print(f"Wrote {TABLES / 'S27_v2_spotchecks.csv'}: {len(out)} rows")
    for thread, sub in out.groupby("thread"):
        print(f"\n=== {thread} (n={len(sub)}) ===")
        print(sub[["title", "company_name_canonical", "period",
                   "seniority_3level"]].to_string(index=False))


if __name__ == "__main__":
    main()
