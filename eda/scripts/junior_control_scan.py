"""
Junior-SWE vs junior-control scope scan.

Answers: have junior SWE job requirements changed differently from junior
control jobs (nurses, accountants, civil/mechanical/electrical engineers)
between 2024 and 2026? Also compares senior SWE vs senior control so the
two buckets can be read side-by-side.

Data:
- primary:  data/unified_core.parquet
- scope features (joined on uid):
  exploration/artifacts/shared/T11_posting_features.parquet
  (tech_count, requirement_breadth_resid, scope_density, credential_stack_depth)

Output:
- eda/tables/junior_scope_swe_vs_control.csv  (2x2x4 = 16 rows; 4 metric panels)
- eda/tables/junior_scope_features.csv         (scope-feature columns from join)

Run:
  ./.venv/bin/python eda/scripts/junior_control_scan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from scans import AI_VOCAB_PATTERN, text_col, text_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = PROJECT_ROOT / "data" / "unified_core.parquet"
T11_FEATURES_PATH = (
    PROJECT_ROOT / "exploration" / "artifacts" / "shared" / "T11_posting_features.parquet"
)
TABLES_DIR = PROJECT_ROOT / "eda" / "tables"

BASE_FILTER = (
    "is_english = true AND date_flag = 'ok' AND (is_swe = true OR is_control = true) "
    "AND seniority_3level IN ('junior','senior')"
)


def four_panel_scan(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """2x2x4 grid on the four primary metrics."""
    df = con.execute(f"""
      SELECT
        period,
        CASE WHEN is_swe THEN 'SWE' WHEN is_control THEN 'control' END AS occupation,
        seniority_3level AS seniority,
        COUNT(*) AS n,
        AVG(description_length) AS mean_desc_len,
        AVG(yoe_min_years_llm) AS mean_yoe_llm,
        COUNT(yoe_min_years_llm) AS n_with_yoe,
        SUM(CASE WHEN regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}') THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS ai_rate,
        SUM(CASE WHEN ghost_assessment_llm = 'inflated' THEN 1 ELSE 0 END)::DOUBLE
          / NULLIF(SUM(CASE WHEN ghost_assessment_llm IS NOT NULL THEN 1 ELSE 0 END), 0) AS inflated_rate,
        SUM(CASE WHEN ghost_assessment_llm IS NOT NULL THEN 1 ELSE 0 END) AS n_with_ghost
      FROM '{CORE_PATH}'
      WHERE {BASE_FILTER} AND {text_filter()}
      GROUP BY 1,2,3
      ORDER BY 1,2,3
    """).df()
    return df


def scope_features_scan(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Same 2x2x4 grid with scope features pulled from T11 features parquet."""
    if not T11_FEATURES_PATH.exists():
        print(f"  T11 features parquet not found — skipping scope-feature join.")
        return pd.DataFrame()
    df = con.execute(f"""
      WITH core AS (
        SELECT uid, period,
               CASE WHEN is_swe THEN 'SWE' WHEN is_control THEN 'control' END AS occupation,
               seniority_3level AS seniority
        FROM '{CORE_PATH}'
        WHERE {BASE_FILTER}
      ),
      feats AS (
        SELECT uid, tech_count, requirement_breadth_resid,
               scope_density, credential_stack_depth
        FROM '{T11_FEATURES_PATH}'
      )
      SELECT c.period, c.occupation, c.seniority,
             COUNT(*) AS n_joined,
             AVG(f.tech_count) AS mean_tech_count,
             AVG(f.requirement_breadth_resid) AS mean_breadth_resid,
             AVG(f.scope_density) AS mean_scope_density,
             AVG(f.credential_stack_depth) AS mean_credential_stack
      FROM core c
      INNER JOIN feats f USING (uid)
      GROUP BY 1,2,3
      ORDER BY 1,2,3
    """).df()
    return df


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    print("Scanning primary 4-metric grid…")
    grid = four_panel_scan(con)
    grid.to_csv(TABLES_DIR / "junior_scope_swe_vs_control.csv", index=False)
    print(f"  wrote {TABLES_DIR / 'junior_scope_swe_vs_control.csv'}  ({len(grid)} rows)")

    print("Scanning scope-feature join…")
    feats = scope_features_scan(con)
    if not feats.empty:
        feats.to_csv(TABLES_DIR / "junior_scope_features.csv", index=False)
        print(f"  wrote {TABLES_DIR / 'junior_scope_features.csv'}  ({len(feats)} rows)")

    print()
    print("Primary grid summary (2026-04 row per cell):")
    latest = grid[grid["period"] == "2026-04"][
        ["occupation", "seniority", "n", "mean_desc_len", "mean_yoe_llm", "ai_rate", "inflated_rate"]
    ]
    print(latest.to_string(index=False))


if __name__ == "__main__":
    main()
