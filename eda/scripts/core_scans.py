"""
Phase B v2 — scans on `data/unified_core.parquet`.

`unified_core.parquet` is the Stage 9 balanced LLM-frame subset of
`unified.parquet`: 110k rows × 42 analysis-ready columns, LinkedIn-only,
with `is_control` populated across all four periods. Promoted to primary
analysis file for v2 of the EDA.

New scans (on top of S1–Sv re-run from scans.py):
  S12 H8   YOE trajectory by seniority_3level × period (LLM YOE)
  S13 H9   Vendor-specific mentions (Copilot / Cursor / Claude / …)
  S14 H10  AI-mention vs ghost/inflated rate
  S15 H11  Control AI-rise occupational drivers
  S16 H12  Posting survival (uses unified_core_observations.parquet)
  S17 H13  Within-firm overlap panel AI rewrite

Re-runs of earlier scans on core: S1, S3, S10, S11, Sv — with the same
pre-committed vocab and filter definitions.

Run:
  ./.venv/bin/python eda/scripts/core_scans.py
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scans import (
    AI_VOCAB_PATTERN,
    AI_VOCAB_PHRASES,
    BIG_TECH_CANONICAL,
    NEW_AI_TITLE_PATTERN,
    fig_caption,
    save_fig,
    period_order,
)

random.seed(0)
np.random.seed(0)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = PROJECT_ROOT / "data" / "unified_core.parquet"
CORE_OBS_PATH = PROJECT_ROOT / "data" / "unified_core_observations.parquet"
TABLES_DIR = PROJECT_ROOT / "eda" / "tables"
FIGURES_DIR = PROJECT_ROOT / "eda" / "figures"

# Core default filter — is_english AND date_flag='ok' (source_platform is linkedin by construction)
CORE_DEFAULT_FILTER = "is_english = true AND date_flag = 'ok'"

# Pre-committed vendor list (H9) — each pattern is a word-boundary regex. One
# vendor per row so we can track them independently.
VENDOR_PATTERNS: dict[str, str] = {
    "copilot":    r"(?i)\bcopilot\b",
    "cursor":     r"(?i)\bcursor\b",
    "claude":     r"(?i)\bclaude\b",
    "anthropic":  r"(?i)\banthropic\b",
    "chatgpt":    r"(?i)\bchatgpt\b",
    "openai":     r"(?i)\bopen ?ai\b",
    "gemini":     r"(?i)\bgemini\b",
    "llama":      r"(?i)\bllama\b",
    "mistral":    r"(?i)\bmistral\b",
    "gpt":        r"(?i)\bgpt(-?\d)?\b",
}


# ---------------------------------------------------------------------------
# Re-runs of S1 / S3 / S10 / S11 / Sv on core (writes to *_core.csv / .png)
# ---------------------------------------------------------------------------

def rerun_s1_core(con):
    df = con.execute(f"""
      SELECT period, seniority_3level,
             COUNT(*) AS n,
             SUM(CASE WHEN regexp_matches(description, '{AI_VOCAB_PATTERN}') THEN 1 ELSE 0 END) AS n_ai
      FROM '{CORE_PATH}'
      WHERE {CORE_DEFAULT_FILTER} AND is_swe
      GROUP BY 1,2 ORDER BY 1,2
    """).df()
    df["ai_rate"] = df["n_ai"] / df["n"]
    df.to_csv(TABLES_DIR / "S1_core_ai_vocab.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    levels = ["junior", "mid", "senior", "unknown"]
    colors = {"junior": "#d62728", "mid": "#ff7f0e", "senior": "#2ca02c", "unknown": "#7f7f7f"}
    periods = period_order(df)
    x = np.arange(len(periods))
    width = 0.2
    for i, lvl in enumerate(levels):
        sub = df[df["seniority_3level"] == lvl].set_index("period").reindex(periods)
        ax.bar(x + (i - 1.5) * width, sub["ai_rate"].fillna(0), width,
               label=lvl, color=colors.get(lvl, "#888"))
    ax.set_xticks(x); ax.set_xticklabels(periods, rotation=25)
    ax.set_ylabel("AI-vocab rate (SWE, core)")
    ax.set_title("S1 v2 (H1) — AI-vocab prevalence on unified_core")
    ax.legend(title="seniority_3level")
    fig_caption(ax, f"n = {int(df['n'].sum()):,} SWE in core; filter: is_english AND date_flag='ok'")
    save_fig(fig, "S1_core_ai_vocab")
    return df


def rerun_s11_core(con):
    df = con.execute(f"""
      SELECT period, analysis_group,
             COUNT(*) AS n,
             SUM(CASE WHEN regexp_matches(description, '{AI_VOCAB_PATTERN}') THEN 1 ELSE 0 END) AS n_ai
      FROM '{CORE_PATH}'
      WHERE {CORE_DEFAULT_FILTER} AND analysis_group IN ('swe', 'swe_adjacent', 'control')
      GROUP BY 1,2 ORDER BY 1,2
    """).df()
    df["ai_rate"] = df["n_ai"] / df["n"]
    df.to_csv(TABLES_DIR / "S11_core_swe_vs_control.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    periods = period_order(df)
    x = np.arange(len(periods))
    groups = ["swe", "swe_adjacent", "control"]
    colors = {"swe": "#d62728", "swe_adjacent": "#ff7f0e", "control": "#1f77b4"}
    for g in groups:
        sub = df[df["analysis_group"] == g].set_index("period").reindex(periods)
        ax.plot(periods, sub["ai_rate"].fillna(0), "-o", label=g, color=colors[g], linewidth=2)
    ax.set_ylabel("AI-vocab rate")
    ax.set_title("S11 v2 (H7) — SWE vs control vs adjacent on unified_core (true 2024 control baseline)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig_caption(ax, f"All rows LLM-frame; control baseline now exists in 2024")
    save_fig(fig, "S11_core_swe_vs_control")
    return df


def rerun_s10_core(con):
    bt = ", ".join(f"'{b}'" for b in BIG_TECH_CANONICAL)
    df = con.execute(f"""
      SELECT period,
             CASE WHEN LOWER(company_name_canonical) IN ({bt}) THEN 'big_tech' ELSE 'rest' END AS tier,
             COUNT(*) AS n,
             SUM(CASE WHEN regexp_matches(description, '{AI_VOCAB_PATTERN}') THEN 1 ELSE 0 END) AS n_ai
      FROM '{CORE_PATH}'
      WHERE {CORE_DEFAULT_FILTER} AND is_swe
      GROUP BY 1,2 ORDER BY 1,2
    """).df()
    df["ai_rate"] = df["n_ai"] / df["n"]
    period_totals = df.groupby("period")["n"].sum().rename("period_total").reset_index()
    df = df.merge(period_totals, on="period")
    df["volume_share"] = df["n"] / df["period_total"]
    df.to_csv(TABLES_DIR / "S10_core_bigtech_vs_rest.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bt_df = df[df["tier"] == "big_tech"]
    rest_df = df[df["tier"] == "rest"]
    periods = period_order(df)
    x = np.arange(len(periods))
    width = 0.35

    axes[0].bar(x, bt_df.set_index("period").reindex(periods)["volume_share"].fillna(0),
                color="#1f77b4")
    for i, p in enumerate(periods):
        row = bt_df[bt_df["period"] == p]
        if not row.empty:
            v = row["volume_share"].iloc[0]
            axes[0].text(i, v + 0.002, f"{v*100:.2f}%", ha="center", fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(periods, rotation=25)
    axes[0].set_ylabel("Big Tech share of SWE (core)")
    axes[0].set_title("(a) BT volume share")

    axes[1].bar(x - width/2, bt_df.set_index("period").reindex(periods)["ai_rate"].fillna(0),
                width, label="Big Tech", color="#1f77b4")
    axes[1].bar(x + width/2, rest_df.set_index("period").reindex(periods)["ai_rate"].fillna(0),
                width, label="rest", color="#ff7f0e")
    axes[1].set_xticks(x); axes[1].set_xticklabels(periods, rotation=25)
    axes[1].set_ylabel("AI-vocab rate")
    axes[1].set_title("(b) AI-vocab rate by tier")
    axes[1].legend()
    fig.suptitle("S10 v2 (H6) — Big Tech vs rest on unified_core")
    fig_caption(axes[0], f"n = {int(df['n'].sum()):,} SWE core rows; BT = {len(BIG_TECH_CANONICAL)} canonical names")
    fig.tight_layout()
    save_fig(fig, "S10_core_bigtech_vs_rest")
    return df


def rerun_s3_core(con):
    df = con.execute(f"""
      SELECT period, COUNT(*) AS n,
             SUM(CASE WHEN regexp_matches(LOWER(title), '{NEW_AI_TITLE_PATTERN}') THEN 1 ELSE 0 END) AS n_new_ai
      FROM '{CORE_PATH}'
      WHERE {CORE_DEFAULT_FILTER} AND is_swe
      GROUP BY 1 ORDER BY 1
    """).df()
    df["new_ai_share"] = df["n_new_ai"] / df["n"]
    df.to_csv(TABLES_DIR / "S3_core_new_ai_title.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["period"], df["new_ai_share"], color="#9467bd")
    for i, row in df.iterrows():
        ax.text(i, row["new_ai_share"] + 0.003,
                f"{row['new_ai_share']*100:.2f}%\n(n={int(row['n_new_ai']):,})",
                ha="center", fontsize=8)
    ax.set_ylabel("share of SWE titles (core)")
    ax.set_title("S3 v2 (H2) — new-AI-title share on unified_core")
    ax.tick_params(axis="x", rotation=25)
    fig_caption(ax, f"Patterns: ai engineer, ml engineer, llm engineer, agent engineer, …")
    save_fig(fig, "S3_core_new_ai_title")
    return df


def rerun_sv_core(con):
    df = con.execute(f"""
      SELECT period, seniority_3level, COUNT(*) AS n,
             AVG(description_length) AS mean_raw_len,
             AVG(LENGTH(description_core_llm)) AS mean_core_len
      FROM '{CORE_PATH}'
      WHERE {CORE_DEFAULT_FILTER} AND is_swe
      GROUP BY 1,2 ORDER BY 1,2
    """).df()
    df.to_csv(TABLES_DIR / "Sv_core_description_length.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pivot_raw = df.pivot_table(index="period", columns="seniority_3level",
                               values="mean_raw_len", fill_value=np.nan)
    pivot_raw.plot(kind="bar", ax=axes[0], colormap="tab10")
    axes[0].set_title("Mean raw description length (SWE, core)")
    axes[0].set_ylabel("chars")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(fontsize=7)

    pivot_core = df.pivot_table(index="period", columns="seniority_3level",
                                values="mean_core_len", fill_value=np.nan)
    pivot_core.plot(kind="bar", ax=axes[1], colormap="tab10")
    axes[1].set_title("Mean description_core_llm length (SWE, core)")
    axes[1].set_ylabel("chars")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(fontsize=7)

    fig.suptitle("Sv v2 — Description length on unified_core by seniority")
    fig.tight_layout()
    save_fig(fig, "Sv_core_description_length")
    return df


# ---------------------------------------------------------------------------
# S12 — H8 — YOE trajectory by seniority × period (LLM YOE)
# ---------------------------------------------------------------------------

def scan_s12(con):
    df = con.execute(f"""
      SELECT period, seniority_3level,
             COUNT(*) AS n,
             COUNT(yoe_min_years_llm) AS n_with_yoe,
             AVG(yoe_min_years_llm) AS mean_yoe,
             MEDIAN(yoe_min_years_llm) AS median_yoe,
             STDDEV(yoe_min_years_llm) AS sd_yoe,
             QUANTILE_CONT(yoe_min_years_llm, 0.9) AS p90_yoe
      FROM '{CORE_PATH}'
      WHERE {CORE_DEFAULT_FILTER} AND is_swe
      GROUP BY 1,2 ORDER BY 1,2
    """).df()
    df.to_csv(TABLES_DIR / "S12_yoe_trajectory.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    levels = ["junior", "mid", "senior"]
    colors = {"junior": "#d62728", "mid": "#ff7f0e", "senior": "#2ca02c"}
    periods = period_order(df)

    # Mean
    for lvl in levels:
        sub = df[df["seniority_3level"] == lvl].set_index("period").reindex(periods)
        axes[0].plot(periods, sub["mean_yoe"], "-o", color=colors[lvl], label=lvl, linewidth=2)
        for p, v in zip(periods, sub["mean_yoe"]):
            if pd.notna(v):
                axes[0].text(periods.index(p), v + 0.1, f"{v:.2f}", ha="center", fontsize=8,
                             color=colors[lvl])
    axes[0].set_ylabel("mean yoe_min_years_llm")
    axes[0].set_title("Mean LLM-YOE by seniority × period")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend()

    # Median
    for lvl in levels:
        sub = df[df["seniority_3level"] == lvl].set_index("period").reindex(periods)
        axes[1].plot(periods, sub["median_yoe"], "-s", color=colors[lvl], label=lvl, linewidth=2)
    axes[1].set_ylabel("median yoe_min_years_llm")
    axes[1].set_title("Median LLM-YOE by seniority × period")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend()

    fig.suptitle("S12 (H8) — YOE floor trajectory: counter-scope-inflation check")
    fig_caption(axes[0], f"n_SWE = {int(df['n'].sum()):,} core rows; n_with_LLM_YOE = {int(df['n_with_yoe'].sum()):,}")
    fig.tight_layout()
    save_fig(fig, "S12_yoe_trajectory")
    return df


# ---------------------------------------------------------------------------
# S13 — H9 — Vendor-specific mentions
# ---------------------------------------------------------------------------

def scan_s13(con):
    selects = ", ".join(
        f"SUM(CASE WHEN regexp_matches(description, '{p}') THEN 1 ELSE 0 END) AS n_{name}"
        for name, p in VENDOR_PATTERNS.items()
    )
    df = con.execute(f"""
      SELECT period, COUNT(*) AS n_total, {selects}
      FROM '{CORE_PATH}'
      WHERE {CORE_DEFAULT_FILTER} AND is_swe
      GROUP BY 1 ORDER BY 1
    """).df()
    rates = df[["period", "n_total"]].copy()
    for name in VENDOR_PATTERNS:
        rates[f"{name}_rate"] = df[f"n_{name}"] / df["n_total"]
    rates.to_csv(TABLES_DIR / "S13_vendor_mentions.csv", index=False)

    # Figure: small-multiples per-vendor rate over periods
    fig, ax = plt.subplots(figsize=(11, 6))
    periods = rates["period"].tolist()
    x = np.arange(len(periods))
    colors = plt.cm.tab20(np.linspace(0, 1, len(VENDOR_PATTERNS)))
    for (name, _), c in zip(VENDOR_PATTERNS.items(), colors):
        ax.plot(periods, rates[f"{name}_rate"], "-o", label=name, color=c, linewidth=1.5)
    ax.set_ylabel("mention rate in SWE descriptions")
    ax.set_title("S13 (H9) — Dev-tool vendor prevalence in SWE postings (core)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(ncol=2, fontsize=8)
    fig_caption(ax, f"n = {int(df['n_total'].sum()):,} SWE core rows across 4 periods")
    save_fig(fig, "S13_vendor_mentions")

    # Secondary: 2026-04 leaderboard
    last = rates.iloc[-1]
    leaderboard = pd.DataFrame({
        "vendor": list(VENDOR_PATTERNS.keys()),
        "rate_2026_04": [last[f"{v}_rate"] for v in VENDOR_PATTERNS],
    }).sort_values("rate_2026_04", ascending=False)
    leaderboard.to_csv(TABLES_DIR / "S13_vendor_leaderboard_2026_04.csv", index=False)

    return rates


# ---------------------------------------------------------------------------
# S14 — H10 — AI-mention × ghost rate
# ---------------------------------------------------------------------------

def scan_s14(con):
    df = con.execute(f"""
      SELECT period,
             CASE WHEN regexp_matches(description, '{AI_VOCAB_PATTERN}') THEN 'ai_mentioned' ELSE 'no_ai' END AS ai_tag,
             COUNT(*) AS n,
             SUM(CASE WHEN ghost_assessment_llm = 'inflated' THEN 1 ELSE 0 END) AS n_inflated,
             SUM(CASE WHEN ghost_assessment_llm = 'ghost_likely' THEN 1 ELSE 0 END) AS n_ghost,
             SUM(CASE WHEN ghost_assessment_llm = 'realistic' THEN 1 ELSE 0 END) AS n_realistic
      FROM '{CORE_PATH}'
      WHERE {CORE_DEFAULT_FILTER} AND is_swe AND ghost_assessment_llm IS NOT NULL
      GROUP BY 1,2 ORDER BY 1,2
    """).df()
    df["inflated_rate"] = df["n_inflated"] / df["n"]
    df["ghost_rate"] = df["n_ghost"] / df["n"]
    df["any_non_realistic"] = df["inflated_rate"] + df["ghost_rate"]
    df.to_csv(TABLES_DIR / "S14_ai_ghost_crosstab.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pivot_i = df.pivot_table(index="period", columns="ai_tag", values="inflated_rate", fill_value=0)
    pivot_i.plot(kind="bar", ax=axes[0], color=["#2ca02c", "#d62728"])
    axes[0].set_title("Inflated rate (SWE): AI-mentioned vs no_ai")
    axes[0].set_ylabel("share of SWE with ghost_assessment_llm='inflated'")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(title="ai_tag", fontsize=8)

    pivot_g = df.pivot_table(index="period", columns="ai_tag", values="ghost_rate", fill_value=0)
    pivot_g.plot(kind="bar", ax=axes[1], color=["#2ca02c", "#d62728"])
    axes[1].set_title("Ghost_likely rate (SWE): AI-mentioned vs no_ai")
    axes[1].set_ylabel("share with ghost_assessment_llm='ghost_likely'")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(title="ai_tag", fontsize=8)

    fig.suptitle("S14 (H10) — AI mention is not a ghost-job signal (core)")
    fig_caption(axes[0], f"n = {int(df['n'].sum()):,} SWE core rows with labeled ghost assessment")
    fig.tight_layout()
    save_fig(fig, "S14_ai_ghost_crosstab")
    return df


# ---------------------------------------------------------------------------
# S15 — H11 — Control AI-rise occupational drivers
# ---------------------------------------------------------------------------

def scan_s15(con):
    # Control-tier AI-mention rate by broad title category, 2026 only (where control is dense).
    # Bucket control titles into meaningful families using regex on title lowercase.
    df = con.execute(f"""
      SELECT period, title,
             COUNT(*) AS n,
             SUM(CASE WHEN regexp_matches(description, '{AI_VOCAB_PATTERN}') THEN 1 ELSE 0 END) AS n_ai
      FROM '{CORE_PATH}'
      WHERE {CORE_DEFAULT_FILTER} AND is_control
      GROUP BY 1,2
    """).df()

    def family(t):
        t = (t or "").lower()
        if re.search(r"\b(financial|finance|accountant|accounting|revenue|treasury|controller|bookkeep)\b", t):
            return "finance/accounting"
        if re.search(r"\b(electrical|substation|nuclear)\b", t) and "engineer" in t:
            return "electrical/nuclear engineer"
        if re.search(r"\bmechanical\b", t) and "engineer" in t:
            return "mechanical engineer"
        if re.search(r"\bcivil\b", t) and "engineer" in t:
            return "civil engineer"
        if re.search(r"\bchemical\b", t) and "engineer" in t:
            return "chemical engineer"
        if re.search(r"\b(nurse|rn|nursing|lpn)\b", t):
            return "nursing"
        if re.search(r"\b(marketing|brand)\b", t):
            return "marketing"
        if re.search(r"\b(hr|human resource|recruit|talent acq)\b", t):
            return "human resources"
        if re.search(r"\b(sales|account executive|account manager)\b", t):
            return "sales"
        return "other_control"

    df["family"] = df["title"].map(family)
    agg = df.groupby(["period", "family"]).agg(
        n=("n", "sum"), n_ai=("n_ai", "sum")
    ).reset_index()
    agg["ai_rate"] = agg["n_ai"] / agg["n"]
    agg.to_csv(TABLES_DIR / "S15_control_ai_drivers.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    families = sorted(agg["family"].unique())
    periods = period_order(agg)
    pivot = agg.pivot_table(index="family", columns="period", values="ai_rate", fill_value=0)
    # Put biggest rate-in-2026-04 at the top
    pivot["sort_by"] = pivot.get("2026-04", pivot.columns[-1])
    pivot = pivot.sort_values("sort_by").drop(columns="sort_by")
    pivot.plot(kind="barh", ax=ax, colormap="viridis", width=0.8)
    ax.set_xlabel("AI-vocab rate within control family")
    ax.set_title("S15 (H11) — Control AI-rise by occupational family (core)")
    ax.legend(title="period", fontsize=8)
    fig_caption(ax, f"n_control = {int(agg['n'].sum()):,}; family heuristic from title regex")
    save_fig(fig, "S15_control_ai_drivers")
    return agg


# ---------------------------------------------------------------------------
# S16 — H12 — Posting survival
# ---------------------------------------------------------------------------

def scan_s16(con):
    # Per-uid obs_count and first/last scrape_date in core observations.
    df = con.execute(f"""
      WITH uid_stats AS (
        SELECT uid,
               MIN(scrape_date) AS first_seen,
               MAX(scrape_date) AS last_seen,
               COUNT(*) AS obs_count,
               ANY_VALUE(period) AS period,
               ANY_VALUE(analysis_group) AS analysis_group,
               ANY_VALUE(ghost_assessment_llm) AS ghost,
               ANY_VALUE(description) AS description,
               ANY_VALUE(is_swe) AS is_swe
        FROM '{CORE_OBS_PATH}'
        WHERE is_english = true AND date_flag = 'ok'
        GROUP BY uid
      )
      SELECT uid, first_seen, last_seen, obs_count, period, analysis_group, ghost,
             regexp_matches(description, '{AI_VOCAB_PATTERN}') AS ai_mentioned,
             is_swe,
             DATE_DIFF('day', CAST(first_seen AS DATE), CAST(last_seen AS DATE)) AS days_spanned
      FROM uid_stats
    """).df()
    # Restrict to scraped 2026 periods where we have cadence (>1 obs possible)
    df_scraped = df[df["period"].isin(["2026-03", "2026-04"])].copy()

    # Survival summary: obs_count by ai_mentioned and analysis_group
    summary = df_scraped.groupby(["period", "analysis_group", "ai_mentioned"]).agg(
        n=("uid", "count"),
        mean_obs=("obs_count", "mean"),
        median_obs=("obs_count", "median"),
        mean_days=("days_spanned", "mean"),
        p90_days=("days_spanned", lambda s: np.nanquantile(s, 0.9)),
    ).reset_index()
    summary.to_csv(TABLES_DIR / "S16_posting_survival.csv", index=False)

    # Figure: mean obs_count by group × ai_tag × period
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sub = summary[summary["analysis_group"].isin(["swe", "control", "swe_adjacent"])]
    pivot_obs = sub.pivot_table(index=["analysis_group", "ai_mentioned"], columns="period",
                                values="mean_obs", fill_value=0)
    pivot_obs.plot(kind="bar", ax=axes[0], colormap="tab10", width=0.8)
    axes[0].set_title("Mean scrape-date observations per posting")
    axes[0].set_ylabel("mean obs per uid")
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].legend(title="period", fontsize=8)

    pivot_days = sub.pivot_table(index=["analysis_group", "ai_mentioned"], columns="period",
                                 values="mean_days", fill_value=0)
    pivot_days.plot(kind="bar", ax=axes[1], colormap="tab10", width=0.8)
    axes[1].set_title("Mean days spanned (last_seen − first_seen)")
    axes[1].set_ylabel("days")
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)
    axes[1].legend(title="period", fontsize=8)

    fig.suptitle("S16 (H12) — Posting survival: scraped 2026 only, by AI-tag × tier")
    fig_caption(axes[0], f"n_postings = {len(df_scraped):,} (2026-03 + 2026-04); Kaggle periods have single-obs postings")
    fig.tight_layout()
    save_fig(fig, "S16_posting_survival")
    return summary


# ---------------------------------------------------------------------------
# S17 — H13 — Within-firm overlap panel
# ---------------------------------------------------------------------------

def scan_s17(con):
    # Collapse core periods into two buckets: 2024 (Kaggle) and 2026 (scraped).
    # For each company_name_canonical with ≥5 SWE postings in BOTH buckets, compute
    # within-firm AI-vocab rate by bucket and the delta.
    df = con.execute(f"""
      WITH bucketed AS (
        SELECT company_name_canonical,
               CASE WHEN source LIKE 'kaggle%' THEN '2024' ELSE '2026' END AS bucket,
               regexp_matches(description, '{AI_VOCAB_PATTERN}') AS ai
        FROM '{CORE_PATH}'
        WHERE {CORE_DEFAULT_FILTER} AND is_swe
          AND company_name_canonical IS NOT NULL
      ),
      co_panel AS (
        SELECT company_name_canonical,
               SUM(CASE WHEN bucket='2024' THEN 1 ELSE 0 END) AS n_2024,
               SUM(CASE WHEN bucket='2026' THEN 1 ELSE 0 END) AS n_2026,
               SUM(CASE WHEN bucket='2024' AND ai THEN 1 ELSE 0 END) AS n_ai_2024,
               SUM(CASE WHEN bucket='2026' AND ai THEN 1 ELSE 0 END) AS n_ai_2026
        FROM bucketed
        GROUP BY company_name_canonical
      )
      SELECT * FROM co_panel
      WHERE n_2024 >= 5 AND n_2026 >= 5
      ORDER BY n_2024 + n_2026 DESC
    """).df()

    df["ai_rate_2024"] = df["n_ai_2024"] / df["n_2024"]
    df["ai_rate_2026"] = df["n_ai_2026"] / df["n_2026"]
    df["delta"] = df["ai_rate_2026"] - df["ai_rate_2024"]
    df.to_csv(TABLES_DIR / "S17_within_firm_panel.csv", index=False)

    # Summary stats
    n_cos = len(df)
    mean_delta = df["delta"].mean()
    pct_up = (df["delta"] > 0).mean()
    pct_up_10pp = (df["delta"] > 0.10).mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["delta"] * 100, bins=40, color="#2ca02c", edgecolor="white")
    axes[0].axvline(0, color="black", linestyle="--", alpha=0.5)
    axes[0].axvline(mean_delta * 100, color="#d62728", linestyle="-", linewidth=2,
                    label=f"mean = {mean_delta*100:.1f}pp")
    axes[0].set_xlabel("within-firm Δ AI-vocab rate (2026 − 2024), percentage points")
    axes[0].set_ylabel("# companies")
    axes[0].set_title(f"S17 (H13) — Within-firm AI-rewrite deltas (n={n_cos} cos, min 5+5 SWE postings)")
    axes[0].legend()

    axes[1].scatter(df["ai_rate_2024"] * 100, df["ai_rate_2026"] * 100, alpha=0.4, s=20)
    lim = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])
    axes[1].plot([0, lim], [0, lim], "k--", alpha=0.5, label="y = x (no change)")
    axes[1].set_xlabel("2024 AI-vocab rate")
    axes[1].set_ylabel("2026 AI-vocab rate")
    axes[1].set_title(f"{pct_up*100:.0f}% of companies rose, {pct_up_10pp*100:.0f}% rose >10pp")
    axes[1].legend()

    fig.suptitle("S17 (H13) — Within-firm AI rewrite is real on SWE overlap panel")
    fig_caption(axes[0], f"Panel = {n_cos} companies with ≥5 SWE postings in BOTH 2024 (Kaggle) and 2026 (scraped)")
    fig.tight_layout()
    save_fig(fig, "S17_within_firm_panel")

    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    steps = [
        ("Rerun S1  (H1 AI-vocab) on core",           rerun_s1_core),
        ("Rerun S3  (H2 new-AI-title) on core",       rerun_s3_core),
        ("Rerun S10 (H6 BT-vs-rest) on core",         rerun_s10_core),
        ("Rerun S11 (H7 SWE-vs-control) on core",     rerun_s11_core),
        ("Rerun Sv  (desc length) on core",           rerun_sv_core),
        ("S12 (H8)  YOE trajectory",                  scan_s12),
        ("S13 (H9)  Vendor mentions",                 scan_s13),
        ("S14 (H10) AI × ghost",                      scan_s14),
        ("S15 (H11) Control AI drivers",              scan_s15),
        ("S16 (H12) Posting survival",                scan_s16),
        ("S17 (H13) Within-firm overlap",             scan_s17),
    ]
    for name, fn in steps:
        print(f"Running {name} ...")
        try:
            fn(con)
            print("  OK")
        except Exception as e:
            print(f"  FAILED: {e}")
            raise


if __name__ == "__main__":
    main()
