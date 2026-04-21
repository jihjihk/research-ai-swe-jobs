"""
Robustness: does the balanced core (data/unified_core.parquet, 40/30/30
SWE/adjacent/control) produce the same within-group findings as the
natural-distribution file (data/unified.parquet)?

Concern: unified_core is artificially rebalanced. If Stage 9's sampling
is correlated with any measured dimension (AI-vocab, industry, seniority,
per-firm posting patterns), our findings could be sampling artifacts
rather than real data patterns.

Check: re-run the rate metrics behind each v2 finding on full unified.parquet
(restricted to source_platform='linkedin' AND is_english AND date_flag='ok'
to match the core's implicit filter), and compare to the core numbers.

Output:
  eda/tables/C_robustness_core_vs_full.csv

Run:
  ./.venv/bin/python eda/scripts/robustness_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from scans import AI_VOCAB_PATTERN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = PROJECT_ROOT / "data" / "unified_core.parquet"
FULL_PATH = PROJECT_ROOT / "data" / "unified.parquet"
TABLES_DIR = PROJECT_ROOT / "eda" / "tables"

TECH_INDUSTRIES_SQL = (
    "'Computer Software','Information Technology and Services',"
    "'Information Technology & Services','Internet','Computer Games',"
    "'Computer Hardware','Computer Networking','Computer & Network Security',"
    "'Semiconductors','Telecommunications','Software Development',"
    "'IT Services and IT Consulting','Technology, Information and Internet'"
)


def base_filter(path: Path) -> str:
    """Keep the filter parity between full and core. Core is LinkedIn by
    construction; on full we apply source_platform='linkedin' explicitly."""
    f = "is_english = true AND date_flag = 'ok'"
    if path == FULL_PATH:
        f += " AND source_platform = 'linkedin'"
    return f


def h1_h7_swe_vs_control(con: duckdb.DuckDBPyConnection, path: Path) -> pd.DataFrame:
    """AI-vocab rate within SWE vs within control by period."""
    df = con.execute(f"""
      SELECT period,
             SUM(CASE WHEN is_swe THEN 1 ELSE 0 END) AS n_swe,
             SUM(CASE WHEN is_control THEN 1 ELSE 0 END) AS n_ctrl,
             SUM(CASE WHEN is_swe
                       AND regexp_matches(description, '{AI_VOCAB_PATTERN}')
                      THEN 1 ELSE 0 END) AS n_ai_swe,
             SUM(CASE WHEN is_control
                       AND regexp_matches(description, '{AI_VOCAB_PATTERN}')
                      THEN 1 ELSE 0 END) AS n_ai_ctrl
      FROM '{path}'
      WHERE {base_filter(path)}
      GROUP BY 1 ORDER BY 1
    """).df()
    df["swe_ai_rate"] = df["n_ai_swe"] / df["n_swe"]
    df["ctrl_ai_rate"] = df["n_ai_ctrl"] / df["n_ctrl"]
    return df


def h4_nontech_share(con: duckdb.DuckDBPyConnection, path: Path) -> pd.DataFrame:
    """Non-tech industry share of SWE postings with labeled industry."""
    df = con.execute(f"""
      SELECT period,
             SUM(CASE WHEN company_industry IN ({TECH_INDUSTRIES_SQL})
                      THEN 1 ELSE 0 END) AS n_tech,
             SUM(CASE WHEN company_industry IS NOT NULL
                       AND company_industry NOT IN ({TECH_INDUSTRIES_SQL})
                      THEN 1 ELSE 0 END) AS n_nontech
      FROM '{path}'
      WHERE {base_filter(path)} AND is_swe = true
      GROUP BY 1 ORDER BY 1
    """).df()
    df["nontech_share"] = df["n_nontech"] / (df["n_tech"] + df["n_nontech"]).replace(0, pd.NA)
    return df


def h5a_ai_by_seniority(con: duckdb.DuckDBPyConnection, path: Path) -> pd.DataFrame:
    """AI-vocab rate by seniority within SWE, 2026-04 only (latest period)."""
    df = con.execute(f"""
      SELECT seniority_3level,
             COUNT(*) AS n,
             SUM(CASE WHEN regexp_matches(description, '{AI_VOCAB_PATTERN}')
                      THEN 1 ELSE 0 END) AS n_ai
      FROM '{path}'
      WHERE {base_filter(path)} AND is_swe = true AND period = '2026-04'
      GROUP BY 1 ORDER BY 1
    """).df()
    df["ai_rate"] = df["n_ai"] / df["n"]
    return df


def h13_within_firm_panel(con: duckdb.DuckDBPyConnection, path: Path) -> dict:
    """Within-firm AI-rewrite panel summary stats."""
    df = con.execute(f"""
      WITH bucketed AS (
        SELECT company_name_canonical,
               CASE WHEN source LIKE 'kaggle%' THEN '2024' ELSE '2026' END AS bucket,
               regexp_matches(description, '{AI_VOCAB_PATTERN}') AS ai
        FROM '{path}'
        WHERE {base_filter(path)} AND is_swe = true
          AND company_name_canonical IS NOT NULL
      ),
      co_panel AS (
        SELECT company_name_canonical,
               SUM(CASE WHEN bucket='2024' THEN 1 ELSE 0 END) AS n_2024,
               SUM(CASE WHEN bucket='2026' THEN 1 ELSE 0 END) AS n_2026,
               SUM(CASE WHEN bucket='2024' AND ai THEN 1 ELSE 0 END) AS ai_2024,
               SUM(CASE WHEN bucket='2026' AND ai THEN 1 ELSE 0 END) AS ai_2026
        FROM bucketed GROUP BY 1
      )
      SELECT * FROM co_panel WHERE n_2024 >= 5 AND n_2026 >= 5
    """).df()
    df["rate_2024"] = df["ai_2024"] / df["n_2024"]
    df["rate_2026"] = df["ai_2026"] / df["n_2026"]
    df["delta"] = df["rate_2026"] - df["rate_2024"]
    return {
        "n_firms": len(df),
        "mean_delta_pp": df["delta"].mean() * 100,
        "median_delta_pp": df["delta"].median() * 100,
        "pct_up": (df["delta"] > 0).mean() * 100,
        "pct_up_10pp": (df["delta"] > 0.10).mean() * 100,
        "pct_up_20pp": (df["delta"] > 0.20).mean() * 100,
    }


def build_comparison_rows(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """One row per (hypothesis, period or scope, metric) for both sources."""
    rows = []

    for src_label, path in [("core", CORE_PATH), ("full", FULL_PATH)]:
        # H1/H7 — SWE vs control by period
        df = h1_h7_swe_vs_control(con, path)
        for _, r in df.iterrows():
            rows.append({
                "source": src_label, "hypothesis": "H1/H7",
                "scope": r["period"], "metric": "swe_ai_rate",
                "value_pct": r["swe_ai_rate"] * 100,
                "n": int(r["n_swe"]),
            })
            rows.append({
                "source": src_label, "hypothesis": "H1/H7",
                "scope": r["period"], "metric": "ctrl_ai_rate",
                "value_pct": r["ctrl_ai_rate"] * 100,
                "n": int(r["n_ctrl"]),
            })

        # H4 — non-tech share by period
        df = h4_nontech_share(con, path)
        for _, r in df.iterrows():
            rows.append({
                "source": src_label, "hypothesis": "H4",
                "scope": r["period"], "metric": "nontech_share_of_swe",
                "value_pct": float(r["nontech_share"]) * 100 if pd.notna(r["nontech_share"]) else None,
                "n": int(r["n_tech"] + r["n_nontech"]),
            })

        # H5a — AI rate by seniority (2026-04)
        df = h5a_ai_by_seniority(con, path)
        for _, r in df.iterrows():
            rows.append({
                "source": src_label, "hypothesis": "H5a",
                "scope": f"2026-04 · {r['seniority_3level']}",
                "metric": "ai_rate",
                "value_pct": r["ai_rate"] * 100,
                "n": int(r["n"]),
            })

        # H13 — within-firm panel stats
        stats = h13_within_firm_panel(con, path)
        for metric, value in stats.items():
            rows.append({
                "source": src_label, "hypothesis": "H13",
                "scope": "2024↔2026 panel",
                "metric": metric,
                "value_pct": float(value) if "pp" in metric or "pct" in metric else None,
                "n": int(stats["n_firms"]),
            })

    return rows


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    rows = build_comparison_rows(con)
    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "C_robustness_core_vs_full.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows)")

    # Side-by-side comparison table (for the report)
    side = df.pivot_table(
        index=["hypothesis", "scope", "metric"],
        columns="source",
        values="value_pct",
        aggfunc="first",
    ).reset_index()
    side["abs_diff_pp"] = (side.get("core", 0) - side.get("full", 0)).abs()
    side_path = TABLES_DIR / "C_robustness_side_by_side.csv"
    side.to_csv(side_path, index=False)
    print(f"Wrote {side_path}")

    # Console summary
    print()
    print("Side-by-side summary (abs_diff_pp = |core − full|):")
    with pd.option_context("display.max_rows", 60, "display.max_colwidth", 40,
                            "display.float_format", lambda x: f"{x:.3f}"):
        print(side.to_string(index=False))


if __name__ == "__main__":
    main()
