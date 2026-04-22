"""
Phase C triangulation — stress-test the strongest Phase B signals against
four independent slices.

Finalists (one per hypothesis + Sv):
  S1  (H1)  AI-vocab prevalence — SWE 2024 vs 2026
  S3  (H2)  new-AI-title share — 2024 vs 2026
  S10 (H6) Big Tech AI-vocab differential — vs rest, 2026
  S11 (H7) SWE vs control AI divergence — 2026 only
  Sv       description length — 2024 vs 2026

Stress-test slices (a metric survives if directional consistency holds on
at least 3 of 4 slices):
  slice_a  arshkon-only 2024 baseline (drops asaniczka)
  slice_b  metro-balanced (equal-weight across top-10 metros by SWE volume)
  slice_c  exclude aggregators (is_aggregator = false)
  slice_d  exclude is_multi_location (is_multi_location = false OR null)

Run:
  ./.venv/bin/python eda/scripts/triangulate.py
"""

from __future__ import annotations

import random
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Re-import the same pre-committed vocab lists + filter used in Phase B
from scans import (
    AI_VOCAB_PATTERN,
    BIG_TECH_CANONICAL,
    DEFAULT_FILTER,
    NEW_AI_TITLE_PATTERN,
    text_col,
    text_filter,
)

random.seed(0)
np.random.seed(0)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_PATH = PROJECT_ROOT / "data" / "unified.parquet"
TABLES_DIR = PROJECT_ROOT / "eda" / "tables"
FIGURES_DIR = PROJECT_ROOT / "eda" / "figures"


# ---------------------------------------------------------------------------
# Slice filter fragments
# ---------------------------------------------------------------------------

SLICES = {
    "baseline": "TRUE",
    "slice_a_arshkon_only_2024": "(period NOT IN ('2024-01') OR source = 'kaggle_arshkon')",
    # slice_b handled in-query by joining top-10 metros + equal weight
    "slice_b_metro_balanced": "metro_area IS NOT NULL",
    "slice_c_no_aggregator": "is_aggregator = false",
    "slice_d_no_multi_location": "(is_multi_location IS NULL OR is_multi_location = false)",
}


def rate_query(filter_clause, numerator_clause, swe_only=True, extra_where=None):
    """Return SQL computing (period, n, n_positive, rate) with the given filter."""
    where = DEFAULT_FILTER + f" AND ({filter_clause})"
    if swe_only:
        where += " AND is_swe = true"
    if extra_where:
        where += f" AND ({extra_where})"
    return f"""
      SELECT period,
             COUNT(*) AS n,
             SUM(CASE WHEN {numerator_clause} THEN 1 ELSE 0 END) AS n_positive
      FROM '{UNIFIED_PATH}'
      WHERE {where}
      GROUP BY 1 ORDER BY 1
    """


def metro_balanced_rate(con, numerator_clause, swe_only=True, extra_where=None):
    """Equal-weight average across top-10 metros by SWE volume."""
    where = DEFAULT_FILTER + " AND metro_area IS NOT NULL"
    if swe_only:
        where += " AND is_swe = true"
    if extra_where:
        where += f" AND ({extra_where})"
    # Find top-10 metros by total SWE volume across all periods
    top = con.execute(f"""
      SELECT metro_area FROM '{UNIFIED_PATH}'
      WHERE {where}
      GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 10
    """).df()["metro_area"].tolist()
    top_list = ", ".join(f"'{m}'" for m in top)
    df = con.execute(f"""
      SELECT period, metro_area,
             COUNT(*) AS n,
             SUM(CASE WHEN {numerator_clause} THEN 1 ELSE 0 END) AS n_positive
      FROM '{UNIFIED_PATH}'
      WHERE {where} AND metro_area IN ({top_list})
      GROUP BY 1,2 ORDER BY 1,2
    """).df()
    df["rate"] = df["n_positive"] / df["n"]
    # Equal-weight average per period across metros present
    out = df.groupby("period").agg(n=("n", "sum"),
                                    n_positive=("n_positive", "sum"),
                                    rate=("rate", "mean")).reset_index()
    return out


# ---------------------------------------------------------------------------
# Per-scan triangulation
# ---------------------------------------------------------------------------

def triangulate_s1(con):
    """AI-vocab rate on SWE by period under 5 slices."""
    numerator = f"regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}')"
    results = []
    for slice_name, filter_clause in SLICES.items():
        if slice_name == "slice_b_metro_balanced":
            df = metro_balanced_rate(con, numerator, extra_where=text_filter())
        else:
            df = con.execute(rate_query(filter_clause, numerator, extra_where=text_filter())).df()
            df["rate"] = df["n_positive"] / df["n"]
        df["slice"] = slice_name
        df["finalist"] = "S1_H1_ai_vocab"
        results.append(df)
    out = pd.concat(results, ignore_index=True)
    out.to_csv(TABLES_DIR / "C_triangulation_S1.csv", index=False)
    return out


def triangulate_s3(con):
    """New-AI-title share on SWE by period under 5 slices."""
    numerator = f"regexp_matches(title_normalized, '{NEW_AI_TITLE_PATTERN}')"
    results = []
    for slice_name, filter_clause in SLICES.items():
        if slice_name == "slice_b_metro_balanced":
            df = metro_balanced_rate(con, numerator)
        else:
            df = con.execute(rate_query(filter_clause, numerator)).df()
            df["rate"] = df["n_positive"] / df["n"]
        df["slice"] = slice_name
        df["finalist"] = "S3_H2_new_ai_title"
        results.append(df)
    out = pd.concat(results, ignore_index=True)
    out.to_csv(TABLES_DIR / "C_triangulation_S3.csv", index=False)
    return out


def triangulate_s10(con):
    """Big Tech vs rest AI-vocab differential under 5 slices."""
    bt_list = ", ".join(f"'{b}'" for b in BIG_TECH_CANONICAL)
    results = []
    for slice_name, filter_clause in SLICES.items():
        where = DEFAULT_FILTER + f" AND ({filter_clause}) AND is_swe = true AND {text_filter()}"
        if slice_name == "slice_b_metro_balanced":
            where += " AND metro_area IS NOT NULL"
        sql = f"""
          SELECT period,
                 CASE WHEN LOWER(company_name_canonical) IN ({bt_list})
                      THEN 'big_tech' ELSE 'rest' END AS tier,
                 COUNT(*) AS n,
                 SUM(CASE WHEN regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}') THEN 1 ELSE 0 END) AS n_ai
          FROM '{UNIFIED_PATH}'
          WHERE {where}
          GROUP BY 1,2 ORDER BY 1,2
        """
        df = con.execute(sql).df()
        df["ai_rate"] = df["n_ai"] / df["n"]
        df["slice"] = slice_name
        results.append(df)
    out = pd.concat(results, ignore_index=True)
    out.to_csv(TABLES_DIR / "C_triangulation_S10.csv", index=False)
    return out


def triangulate_s11(con):
    """SWE vs control AI-vocab divergence under 5 slices. Control only in scraped."""
    results = []
    for slice_name, filter_clause in SLICES.items():
        where = DEFAULT_FILTER + f" AND ({filter_clause}) AND {text_filter()}"
        if slice_name == "slice_b_metro_balanced":
            where += " AND metro_area IS NOT NULL"
        sql = f"""
          SELECT period,
                 CASE WHEN is_swe THEN 'swe'
                      WHEN is_control THEN 'control'
                      ELSE NULL END AS group_label,
                 COUNT(*) AS n,
                 SUM(CASE WHEN regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}') THEN 1 ELSE 0 END) AS n_ai
          FROM '{UNIFIED_PATH}'
          WHERE {where} AND (is_swe = true OR is_control = true)
          GROUP BY 1,2 ORDER BY 1,2
        """
        df = con.execute(sql).df()
        df["ai_rate"] = df["n_ai"] / df["n"]
        df["slice"] = slice_name
        results.append(df)
    out = pd.concat(results, ignore_index=True)
    out.to_csv(TABLES_DIR / "C_triangulation_S11.csv", index=False)
    return out


def triangulate_sv(con):
    """Mean description length on SWE by period under 5 slices."""
    results = []
    for slice_name, filter_clause in SLICES.items():
        where = DEFAULT_FILTER + f" AND ({filter_clause}) AND is_swe = true"
        if slice_name == "slice_b_metro_balanced":
            where += " AND metro_area IS NOT NULL"
        df = con.execute(f"""
          SELECT period,
                 COUNT(*) AS n,
                 AVG(description_length) AS mean_desc_len,
                 AVG(LENGTH(description_core_llm))
                   FILTER (WHERE llm_extraction_coverage = 'labeled') AS mean_core_len
          FROM '{UNIFIED_PATH}'
          WHERE {where}
          GROUP BY 1 ORDER BY 1
        """).df()
        df["slice"] = slice_name
        results.append(df)
    out = pd.concat(results, ignore_index=True)
    out.to_csv(TABLES_DIR / "C_triangulation_Sv.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Summary figure: per-finalist rate by slice × period
# ---------------------------------------------------------------------------

def summary_figure(s1, s3, s10, s11, sv):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # (0) S1 AI-vocab SWE — all slices
    plot_rate_by_slice(axes[0], s1,
                       title="S1 (H1) AI-vocab rate — SWE",
                       ylabel="AI-vocab rate")

    # (1) S3 new-AI-title share — all slices
    plot_rate_by_slice(axes[1], s3,
                       title="S3 (H2) new-AI-title share — SWE",
                       ylabel="share of SWE titles")

    # (2) S10 Big Tech vs rest AI-vocab differential (BT rate − rest rate)
    bt = s10[s10["tier"] == "big_tech"][["period", "slice", "ai_rate"]]
    rest = s10[s10["tier"] == "rest"][["period", "slice", "ai_rate"]]
    diff = bt.merge(rest, on=["period", "slice"], suffixes=("_bt", "_rest"))
    diff["differential"] = diff["ai_rate_bt"] - diff["ai_rate_rest"]
    diff = diff.rename(columns={"differential": "rate"})
    plot_rate_by_slice(axes[2], diff[["period", "slice", "rate"]],
                       title="S10 (H6) AI-vocab differential (BT − rest)",
                       ylabel="BT minus rest (pp)")

    # (3) S11 SWE rate − control rate
    sw = s11[s11["group_label"] == "swe"][["period", "slice", "ai_rate"]]
    ct = s11[s11["group_label"] == "control"][["period", "slice", "ai_rate"]]
    diff11 = sw.merge(ct, on=["period", "slice"], suffixes=("_swe", "_ctrl"))
    diff11["differential"] = diff11["ai_rate_swe"] - diff11["ai_rate_ctrl"]
    plot_rate_by_slice(axes[3], diff11[["period", "slice", "differential"]].rename(
        columns={"differential": "rate"}),
                       title="S11 (H7) AI-vocab differential (SWE − control)",
                       ylabel="SWE minus control (pp)")

    # (4) Sv mean description length
    sv2 = sv[["period", "slice", "mean_desc_len"]].rename(columns={"mean_desc_len": "rate"})
    plot_rate_by_slice(axes[4], sv2,
                       title="Sv — Mean raw description length — SWE",
                       ylabel="mean chars")

    # (5) Sv mean core length
    sv3 = sv[["period", "slice", "mean_core_len"]].rename(columns={"mean_core_len": "rate"})
    plot_rate_by_slice(axes[5], sv3,
                       title="Sv — Mean description_core_llm length — SWE",
                       ylabel="mean chars (labeled)")

    fig.suptitle("Phase C — Triangulation: finalists across 5 slices × period", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES_DIR / "C_triangulation_summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_rate_by_slice(ax, df, title, ylabel):
    periods = sorted(df["period"].unique())
    slices = [s for s in SLICES.keys() if s in df["slice"].unique()]
    x = np.arange(len(periods))
    width = 0.14
    colors = plt.cm.tab10(np.linspace(0, 1, len(slices)))
    for i, s in enumerate(slices):
        sub = df[df["slice"] == s].set_index("period").reindex(periods)
        ax.bar(x + (i - len(slices) / 2) * width, sub["rate"].fillna(0), width,
               label=s.replace("slice_", "").replace("_", " "), color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=25)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=6, loc="upper left")


def consistency_table(s1, s3, s10, s11, sv):
    """Build a directional-consistency summary: for each finalist, does the
    2024→2026 change have the same sign across all 5 slices?"""
    rows = []

    def earliest_latest(df, group_cols=None):
        if group_cols:
            tmp = df.groupby(["slice"] + group_cols).agg(
                min_period=("period", "min"), max_period=("period", "max")
            ).reset_index()
        else:
            tmp = df.groupby("slice").agg(
                min_period=("period", "min"), max_period=("period", "max")
            ).reset_index()
        return tmp

    def sign_check(df, rate_col, label, group_cols=None):
        """Return one row per group describing direction across slices."""
        per_slice = df.groupby(["slice"] + (group_cols or []))
        local_rows = []
        for keys, sub in per_slice:
            sub = sub.sort_values("period")
            if sub[rate_col].isna().all():
                continue
            first = sub.iloc[0][rate_col]
            last = sub.iloc[-1][rate_col]
            local_rows.append({
                "finalist": label,
                "slice_key": keys if isinstance(keys, tuple) else (keys,),
                "period_first": sub.iloc[0]["period"],
                "period_last": sub.iloc[-1]["period"],
                "rate_first": first,
                "rate_last": last,
                "delta": (last - first) if (pd.notna(first) and pd.notna(last)) else None,
                "direction": (
                    "up" if (pd.notna(first) and pd.notna(last) and last > first)
                    else ("down" if (pd.notna(first) and pd.notna(last) and last < first)
                          else "flat_or_null")
                ),
            })
        return local_rows

    rows += sign_check(s1, "rate", "S1_H1_ai_vocab")
    rows += sign_check(s3, "rate", "S3_H2_new_ai_title")
    rows += sign_check(s10.assign(rate=s10["ai_rate"]), "rate",
                       "S10_H6_ai_rate", group_cols=["tier"])
    rows += sign_check(s11.assign(rate=s11["ai_rate"]), "rate",
                       "S11_H7_ai_rate", group_cols=["group_label"])
    rows += sign_check(sv.rename(columns={"mean_desc_len": "rate"}),
                       "rate", "Sv_desc_length")

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "C_consistency.csv", index=False)
    return out


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    print("Triangulating S1 (H1) AI-vocab...")
    s1 = triangulate_s1(con)
    print("Triangulating S3 (H2) new-AI-title...")
    s3 = triangulate_s3(con)
    print("Triangulating S10 (H6) Big Tech vs rest...")
    s10 = triangulate_s10(con)
    print("Triangulating S11 (H7) SWE vs control...")
    s11 = triangulate_s11(con)
    print("Triangulating Sv description length...")
    sv = triangulate_sv(con)

    print("Building summary figure...")
    summary_figure(s1, s3, s10, s11, sv)

    print("Building consistency table...")
    consistency = consistency_table(s1, s3, s10, s11, sv)
    print(f"  {len(consistency)} consistency rows written")

    print("Done. Outputs: eda/tables/C_triangulation_*.csv, eda/figures/C_triangulation_summary.png")


if __name__ == "__main__":
    main()
