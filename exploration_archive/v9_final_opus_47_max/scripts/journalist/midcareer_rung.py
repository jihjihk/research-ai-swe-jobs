"""YOE-bucket share investigation: where did the middle narrow?

Recent commentary (Simon Willison, Dec 2025; Hosseini & Lichtinger, SSRN 5425555)
argues the labor-market "narrowing middle" is mid-career (3-7 YOE), not junior.
Our earlier T30 panel used only the binary cut J3 (YOE<=2) and S4 (YOE>=5). This
script re-buckets the LLM YOE into a 5-rung scheme and computes 2024->2026 shares,
pp changes, Wilson 95% CIs, and the same on the 2,109 returning-company cohort.

Inputs:
  - data/unified_core.parquet
  - exploration/artifacts/shared/returning_companies_cohort.csv

Filters:
  - source_platform == 'linkedin'
  - is_swe == True
  - is_english == True
  - date_flag == 'ok'
  - llm_classification_coverage == 'labeled'  (YOE-based analysis requires labeled)

YOE buckets (on yoe_min_years_llm):
  A: 0-2    (<=2)
  B: 3-4
  C: 5-7
  D: 8-10
  E: 11+
  unknown: yoe_min_years_llm IS NULL   (reported separately, denominator split)

Periods:
  - pooled-2024 = kaggle_asaniczka (2024-01) + kaggle_arshkon (2024-04)
  - arshkon-only 2024 (robustness cut)
  - pooled-2026 = scraped (2026-03 + 2026-04)

Outputs:
  - exploration/tables/journalist/yoe_bucket_shares.csv
  - exploration/tables/journalist/yoe_bucket_returning.csv
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
UNIFIED_CORE = ROOT / "data" / "unified_core.parquet"
RETURNING_CSV = ROOT / "exploration" / "artifacts" / "shared" / "returning_companies_cohort.csv"
OUT_DIR = ROOT / "exploration" / "tables" / "journalist"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MAIN = OUT_DIR / "yoe_bucket_shares.csv"
OUT_RETURNING = OUT_DIR / "yoe_bucket_returning.csv"


BUCKET_ORDER = ["A_0-2", "B_3-4", "C_5-7", "D_8-10", "E_11plus", "unknown"]


def bucket_case_sql(col: str = "yoe_min_years_llm") -> str:
    """SQL CASE expression that assigns each row to a bucket label."""
    return (
        "CASE "
        f"WHEN {col} IS NULL THEN 'unknown' "
        f"WHEN {col} <= 2 THEN 'A_0-2' "
        f"WHEN {col} <= 4 THEN 'B_3-4' "
        f"WHEN {col} <= 7 THEN 'C_5-7' "
        f"WHEN {col} <= 10 THEN 'D_8-10' "
        "ELSE 'E_11plus' "
        "END"
    )


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion at 95%."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    lo = centre - half
    hi = centre + half
    return (max(0.0, lo), min(1.0, hi))


def shares_with_ci(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Given a long-form count frame with columns [bucket, n], produce share + CI.

    Denominator is the SUM of n across all buckets (including 'unknown'). We also
    compute a known-only share (denominator excludes 'unknown') so downstream
    pp-change comparisons on the 3-4 / 5-7 / 8-10 rungs can be read both ways.
    """
    total = int(df["n"].sum())
    known = int(df.loc[df["bucket"] != "unknown", "n"].sum())
    rows = []
    for bucket in BUCKET_ORDER:
        k = int(df.loc[df["bucket"] == bucket, "n"].sum())
        s_all_lo, s_all_hi = wilson_ci(k, total)
        if bucket == "unknown":
            s_known, s_known_lo, s_known_hi = (float("nan"),) * 3
        else:
            s_known_lo, s_known_hi = wilson_ci(k, known) if known else (float("nan"),) * 2
            s_known = k / known if known else float("nan")
        rows.append(
            {
                "period_label": label,
                "bucket": bucket,
                "n_bucket": k,
                "n_denominator_all": total,
                "n_denominator_known": known,
                "share_of_all": (k / total) if total else float("nan"),
                "share_of_all_lo": s_all_lo,
                "share_of_all_hi": s_all_hi,
                "share_of_known": s_known,
                "share_of_known_lo": s_known_lo,
                "share_of_known_hi": s_known_hi,
            }
        )
    return pd.DataFrame(rows)


def pp_change(tbl: pd.DataFrame, base: str, comp: str) -> pd.DataFrame:
    """Difference in share_of_all (and share_of_known) between two period_labels."""
    b = tbl[tbl["period_label"] == base].set_index("bucket")
    c = tbl[tbl["period_label"] == comp].set_index("bucket")
    rows = []
    for bucket in BUCKET_ORDER:
        if bucket not in b.index or bucket not in c.index:
            continue
        rows.append(
            {
                "period_label": f"delta_{comp}_minus_{base}",
                "bucket": bucket,
                "pp_change_share_of_all": c.loc[bucket, "share_of_all"] - b.loc[bucket, "share_of_all"],
                "pp_change_share_of_known": c.loc[bucket, "share_of_known"] - b.loc[bucket, "share_of_known"],
                "base_share_of_all": b.loc[bucket, "share_of_all"],
                "comp_share_of_all": c.loc[bucket, "share_of_all"],
                "base_share_of_known": b.loc[bucket, "share_of_known"],
                "comp_share_of_known": c.loc[bucket, "share_of_known"],
                "base_n_all": int(b.loc[bucket, "n_denominator_all"]),
                "comp_n_all": int(c.loc[bucket, "n_denominator_all"]),
            }
        )
    return pd.DataFrame(rows)


BASE_FILTER = (
    "source_platform = 'linkedin' "
    "AND is_swe "
    "AND is_english "
    "AND date_flag = 'ok' "
    "AND llm_classification_coverage = 'labeled'"
)


def bucket_counts(con: duckdb.DuckDBPyConnection, extra_where: str) -> pd.DataFrame:
    sql = f"""
    SELECT {bucket_case_sql()} AS bucket, COUNT(*) AS n
    FROM read_parquet('{UNIFIED_CORE.as_posix()}')
    WHERE {BASE_FILTER} {extra_where}
    GROUP BY 1
    """
    return con.sql(sql).df()


def build_main_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    # (1) pooled-2024: asaniczka 2024-01 + arshkon 2024-04
    frames.append(
        shares_with_ci(
            bucket_counts(
                con,
                "AND period IN ('2024-01','2024-04') AND source IN ('kaggle_asaniczka','kaggle_arshkon')",
            ),
            "pooled-2024",
        )
    )
    # (2) arshkon-only 2024 (robustness baseline)
    frames.append(
        shares_with_ci(
            bucket_counts(con, "AND source = 'kaggle_arshkon' AND period = '2024-04'"),
            "arshkon-2024",
        )
    )
    # (3) asaniczka-only 2024 (for completeness)
    frames.append(
        shares_with_ci(
            bucket_counts(con, "AND source = 'kaggle_asaniczka' AND period = '2024-01'"),
            "asaniczka-2024",
        )
    )
    # (4) pooled-2026: scraped 2026-03 + 2026-04
    frames.append(
        shares_with_ci(
            bucket_counts(con, "AND source = 'scraped' AND period IN ('2026-03','2026-04')"),
            "pooled-2026",
        )
    )

    shares = pd.concat(frames, ignore_index=True)

    # pp-change rows
    deltas = pd.concat(
        [
            pp_change(shares, "pooled-2024", "pooled-2026"),
            pp_change(shares, "arshkon-2024", "pooled-2026"),
            pp_change(shares, "asaniczka-2024", "pooled-2026"),
        ],
        ignore_index=True,
    )
    return shares, deltas


def build_returning_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Same table but restricted to the 2,109 returning-company cohort."""
    returning = pd.read_csv(RETURNING_CSV)
    companies = returning["company_name_canonical"].dropna().unique().tolist()
    # Register as DuckDB relation
    con.register("returning_cohort", pd.DataFrame({"company_name_canonical": companies}))

    extra = "AND company_name_canonical IN (SELECT company_name_canonical FROM returning_cohort)"

    frames = []
    frames.append(
        shares_with_ci(
            bucket_counts(
                con,
                f"AND period IN ('2024-01','2024-04') AND source IN ('kaggle_asaniczka','kaggle_arshkon') {extra}",
            ),
            "pooled-2024",
        )
    )
    frames.append(
        shares_with_ci(
            bucket_counts(con, f"AND source = 'kaggle_arshkon' AND period = '2024-04' {extra}"),
            "arshkon-2024",
        )
    )
    frames.append(
        shares_with_ci(
            bucket_counts(con, f"AND source = 'kaggle_asaniczka' AND period = '2024-01' {extra}"),
            "asaniczka-2024",
        )
    )
    frames.append(
        shares_with_ci(
            bucket_counts(con, f"AND source = 'scraped' AND period IN ('2026-03','2026-04') {extra}"),
            "pooled-2026",
        )
    )
    shares = pd.concat(frames, ignore_index=True)
    deltas = pd.concat(
        [
            pp_change(shares, "pooled-2024", "pooled-2026"),
            pp_change(shares, "arshkon-2024", "pooled-2026"),
        ],
        ignore_index=True,
    )
    return shares, deltas


def main() -> None:
    con = duckdb.connect()

    shares_full, deltas_full = build_main_table(con)
    out_full = pd.concat([shares_full, deltas_full], ignore_index=True)
    out_full.to_csv(OUT_MAIN, index=False)

    shares_ret, deltas_ret = build_returning_table(con)
    out_ret = pd.concat([shares_ret, deltas_ret], ignore_index=True)
    out_ret.to_csv(OUT_RETURNING, index=False)

    # Stdout summary for the agent's final report
    def _print_table(label, shares_df, deltas_df):
        print(f"\n=== {label} ===")
        piv = shares_df.pivot_table(
            index="bucket",
            columns="period_label",
            values="share_of_all",
        ).reindex(BUCKET_ORDER)
        n_piv = shares_df.pivot_table(
            index="bucket",
            columns="period_label",
            values="n_denominator_all",
            aggfunc="max",
        ).reindex(BUCKET_ORDER)
        print("share_of_all (all rows, including unknown):")
        print((piv * 100).round(2).to_string())
        print("\nn_denominator_all:")
        print(n_piv.astype("Int64").to_string())
        print("\npp changes (pooled-2026 minus pooled-2024):")
        d_pp = deltas_df[deltas_df["period_label"] == "delta_pooled-2026_minus_pooled-2024"].set_index("bucket")
        print((d_pp[["pp_change_share_of_all", "pp_change_share_of_known"]] * 100).round(2).to_string())
        print("\npp changes (pooled-2026 minus arshkon-2024):")
        d_pp2 = deltas_df[deltas_df["period_label"] == "delta_pooled-2026_minus_arshkon-2024"].set_index("bucket")
        print((d_pp2[["pp_change_share_of_all", "pp_change_share_of_known"]] * 100).round(2).to_string())
        print("\nWilson 95% CIs (share_of_all):")
        for pl in ["pooled-2024", "arshkon-2024", "pooled-2026"]:
            sub = shares_df[shares_df["period_label"] == pl].set_index("bucket")
            for b in BUCKET_ORDER:
                if b in sub.index:
                    row = sub.loc[b]
                    print(
                        f"  {pl:16s} {b:10s}  share={row['share_of_all']*100:6.2f}%  "
                        f"[{row['share_of_all_lo']*100:6.2f}%, {row['share_of_all_hi']*100:6.2f}%]  "
                        f"n={int(row['n_bucket'])}/{int(row['n_denominator_all'])}"
                    )

    _print_table("FULL SAMPLE", shares_full, deltas_full)
    _print_table("RETURNING-COMPANIES COHORT (n=2109 firms)", shares_ret, deltas_ret)

    print(f"\nWrote: {OUT_MAIN.relative_to(ROOT)}")
    print(f"Wrote: {OUT_RETURNING.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
