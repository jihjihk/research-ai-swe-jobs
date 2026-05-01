"""T19 — JOLTS confound check.

Test whether the cooling labor market is biasing our cross-period
comparisons by computing the key within-scraped-window effects:
  - scraped early half vs late half (same instrument, different week)
  - arshkon early vs late half (within-2024 stability)
If these within-window effects are much smaller than the cross-period
effects for our headline metrics, then the cross-period estimate is not
meaningfully confounded by macro trends.

Also runs a Levene-style variance test per metric.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/home/jihgaboot/gabor/job-research")
PARQUET = ROOT / "data/unified.parquet"
TABLES = ROOT / "exploration/tables/T19"

DEFAULT_FILTER = (
    "source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true"
)

AI_BROAD_TERMS = [
    r"\bai\b", r"\bartificial intelligence\b", r"\bmachine learning\b", r"\bml\b",
    r"\bdeep learning\b", r"\bnlp\b", r"\bllms?\b", r"\bgenerative ai\b",
    r"\bgen\s?ai\b", r"\brag\b", r"\blangchain\b", r"\blanggraph\b",
    r"\bcopilot\b", r"\bclaude\b", r"\banthropic\b", r"\bopenai\b", r"\bgpt\b",
    r"\bchatgpt\b", r"\bgemini\b", r"\bagents?\b", r"\bagentic\b",
    r"\bvector (?:db|database)\b", r"\bmcp\b", r"\bprompt engineering\b",
]
AI_BROAD_SQL = " OR ".join(
    f"regexp_matches(lower(description), '{p}')" for p in AI_BROAD_TERMS
)
AI_NARROW_SQL = r"regexp_matches(lower(description), '\bai\b')"
COPILOT_SQL = r"regexp_matches(lower(description), '\bcopilot\b')"


def metric_block(where: str) -> str:
    return f"""
    SELECT
      COUNT(*) as n,
      AVG(CASE WHEN {AI_BROAD_SQL} THEN 1.0 ELSE 0.0 END) as ai_broad,
      AVG(CASE WHEN {AI_NARROW_SQL} THEN 1.0 ELSE 0.0 END) as ai_narrow,
      AVG(CASE WHEN {COPILOT_SQL} THEN 1.0 ELSE 0.0 END) as copilot,
      AVG(description_length) as desc_len,
      AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) as entry_share,
      AVG(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) as yoe_le2
    FROM read_parquet('{PARQUET}')
    WHERE {DEFAULT_FILTER} AND {where}
    """


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=6")

    # arshkon halves (by date_posted)
    dates = con.execute(
        f"""
        SELECT DISTINCT date_posted FROM read_parquet('{PARQUET}')
        WHERE {DEFAULT_FILTER} AND source='kaggle_arshkon'
        ORDER BY date_posted
        """
    ).fetch_df()
    mid = len(dates) // 2
    ars_early_dates = "'" + "','".join(dates.date_posted.iloc[:mid].tolist()) + "'"
    ars_late_dates = "'" + "','".join(dates.date_posted.iloc[mid:].tolist()) + "'"
    ars_early = con.execute(
        metric_block(
            f"source='kaggle_arshkon' AND date_posted IN ({ars_early_dates})"
        )
    ).fetchone()
    ars_late = con.execute(
        metric_block(
            f"source='kaggle_arshkon' AND date_posted IN ({ars_late_dates})"
        )
    ).fetchone()

    # scraped halves
    scrape_dates = con.execute(
        f"""
        SELECT DISTINCT scrape_date FROM read_parquet('{PARQUET}')
        WHERE {DEFAULT_FILTER} AND source='scraped'
        ORDER BY scrape_date
        """
    ).fetch_df()
    mid = len(scrape_dates) // 2
    scr_early = "'" + "','".join(scrape_dates.scrape_date.iloc[:mid].tolist()) + "'"
    scr_late = "'" + "','".join(scrape_dates.scrape_date.iloc[mid:].tolist()) + "'"
    scr_early_r = con.execute(
        metric_block(f"source='scraped' AND scrape_date IN ({scr_early})")
    ).fetchone()
    scr_late_r = con.execute(
        metric_block(f"source='scraped' AND scrape_date IN ({scr_late})")
    ).fetchone()

    cols = ["n", "ai_broad", "ai_narrow", "copilot", "desc_len", "entry_share", "yoe_le2"]
    rows = []
    for label, r in [
        ("arshkon_early", ars_early),
        ("arshkon_late", ars_late),
        ("scraped_early", scr_early_r),
        ("scraped_late", scr_late_r),
    ]:
        rows.append({"bin": label, **dict(zip(cols, r))})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "jolts_confound_halves.csv", index=False)
    print(df.to_string(index=False))

    # Within-window deltas and cross-window delta
    def delta(e, l, m):
        return l[m] - e[m]

    e_ars = dict(zip(cols, ars_early))
    l_ars = dict(zip(cols, ars_late))
    e_scr = dict(zip(cols, scr_early_r))
    l_scr = dict(zip(cols, scr_late_r))

    summary = []
    for m in ["ai_broad", "ai_narrow", "copilot", "desc_len", "entry_share", "yoe_le2"]:
        within_ars = l_ars[m] - e_ars[m]
        within_scr = l_scr[m] - e_scr[m]
        # Total cross-period using the two halves' overall means
        # Cross = ((scr_early + scr_late)/2) - ((ars_early + ars_late)/2) weighted by n
        mean_ars = (ars_early[0] * e_ars[m] + ars_late[0] * l_ars[m]) / (ars_early[0] + ars_late[0])
        mean_scr = (scr_early_r[0] * e_scr[m] + scr_late_r[0] * l_scr[m]) / (scr_early_r[0] + scr_late_r[0])
        cross = mean_scr - mean_ars
        ratio_ars = (abs(cross) / abs(within_ars)) if within_ars else float("inf")
        ratio_scr = (abs(cross) / abs(within_scr)) if within_scr else float("inf")
        summary.append(
            {
                "metric": m,
                "within_arshkon_delta": within_ars,
                "within_scraped_delta": within_scr,
                "cross_period_delta": cross,
                "cross_over_within_arshkon": ratio_ars,
                "cross_over_within_scraped": ratio_scr,
            }
        )
    sdf = pd.DataFrame(summary)
    sdf.to_csv(TABLES / "jolts_confound_summary.csv", index=False)
    print()
    print(sdf.to_string(index=False))

    # Also compute Levene's test for desc_length variance across scrape_date bins
    print("\nLevene's test — desc_length variance across scraped dates (10 bins)...")
    per_day = con.execute(
        f"""
        SELECT scrape_date, description_length
        FROM read_parquet('{PARQUET}')
        WHERE {DEFAULT_FILTER} AND source='scraped' AND description_length IS NOT NULL
        """
    ).fetch_df()
    groups = [g["description_length"].values for _, g in per_day.groupby("scrape_date")]
    if len(groups) > 1:
        stat, p = stats.levene(*groups, center="median")
        print(f"  Levene stat={stat:.3f}, p={p:.4f}")
        with open(TABLES / "levene_desc_length.txt", "w") as f:
            f.write(f"levene_stat\t{stat}\n")
            f.write(f"p_value\t{p}\n")
            f.write(f"n_groups\t{len(groups)}\n")


if __name__ == "__main__":
    main()
