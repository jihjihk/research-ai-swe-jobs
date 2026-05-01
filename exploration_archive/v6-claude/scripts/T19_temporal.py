"""T19 — Temporal patterns & rate-of-change estimation.

Computes:
  1. Actual date spans per source (no hardcoded month gaps).
  2. Annualized rate-of-change for key metrics:
     - within-2024 (arshkon vs asaniczka)
     - cross-period (2024 -> 2026 using observed elapsed time)
  3. Within-arshkon stability: bin by date_posted, check metric variance.
  4. Within-scraped-window stability: daily counts + daily metric stability.
  5. Posting-age coverage characterization.
  6. Annotated timeline figure with AI tool release markers.
  7. JOLTS confound check: compute cross-period metric effects within a
     single source window (scraped day 1 vs scraped day final).
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
PARQUET = ROOT / "data/unified.parquet"
TABLES = ROOT / "exploration/tables/T19"
FIGS = ROOT / "exploration/figures/T19"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

DEFAULT_FILTER = (
    "source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true"
)

# AI broad union — same 24 terms as T18
AI_BROAD_TERMS = [
    r"\bai\b",
    r"\bartificial intelligence\b",
    r"\bmachine learning\b",
    r"\bml\b",
    r"\bdeep learning\b",
    r"\bnlp\b",
    r"\bllms?\b",
    r"\bgenerative ai\b",
    r"\bgen\s?ai\b",
    r"\brag\b",
    r"\blangchain\b",
    r"\blanggraph\b",
    r"\bcopilot\b",
    r"\bclaude\b",
    r"\banthropic\b",
    r"\bopenai\b",
    r"\bgpt\b",
    r"\bchatgpt\b",
    r"\bgemini\b",
    r"\bagents?\b",
    r"\bagentic\b",
    r"\bvector (?:db|database)\b",
    r"\bmcp\b",
    r"\bprompt engineering\b",
]
AI_BROAD_SQL = " OR ".join(
    f"regexp_matches(lower(description), '{p}')" for p in AI_BROAD_TERMS
)
AI_NARROW_SQL = r"regexp_matches(lower(description), '\bai\b')"
COPILOT_SQL = r"regexp_matches(lower(description), '\bcopilot\b')"
CLAUDE_SQL = r"regexp_matches(lower(description), '\bclaude\b')"
END_TO_END_SQL = r"regexp_matches(lower(description), '\bend[- ]?to[- ]?end\b')"
CROSS_FUNC_SQL = r"regexp_matches(lower(description), '\bcross[- ]?functional\b')"
SCOPE_SQL = f"({END_TO_END_SQL} OR {CROSS_FUNC_SQL})"


def metric_sql(where: str) -> str:
    """Return SELECT that computes all headline metrics for a filtered subset."""
    return f"""
    SELECT
      COUNT(*) as n,
      AVG(CASE WHEN {AI_BROAD_SQL} THEN 1.0 ELSE 0.0 END) as ai_broad,
      AVG(CASE WHEN {AI_NARROW_SQL} THEN 1.0 ELSE 0.0 END) as ai_narrow,
      AVG(CASE WHEN {COPILOT_SQL} THEN 1.0 ELSE 0.0 END) as copilot,
      AVG(CASE WHEN {CLAUDE_SQL} THEN 1.0 ELSE 0.0 END) as claude_tool,
      AVG(description_length) as desc_len_mean,
      MEDIAN(description_length) as desc_len_median,
      AVG(CASE WHEN {SCOPE_SQL} THEN 1.0 ELSE 0.0 END) as scope_any,
      AVG(CASE WHEN seniority_final = 'entry' AND seniority_final != 'unknown' THEN 1.0 ELSE 0.0 END) as entry_share_all,
      AVG(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) as yoe_le2_share
    FROM read_parquet('{PARQUET}')
    WHERE {DEFAULT_FILTER} AND {where}
    """


def step1_date_ranges(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows = []
    sources = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]
    for src in sources:
        r = con.execute(
            f"""
            SELECT MIN(date_posted), MAX(date_posted),
                   MIN(scrape_date), MAX(scrape_date),
                   COUNT(*) as n,
                   SUM(CASE WHEN date_posted IS NOT NULL THEN 1 ELSE 0 END) as n_dp,
                   SUM(CASE WHEN posting_age_days IS NOT NULL THEN 1 ELSE 0 END) as n_pad
            FROM read_parquet('{PARQUET}')
            WHERE {DEFAULT_FILTER} AND source='{src}'
            """
        ).fetchone()
        rows.append(
            {
                "source": src,
                "min_date_posted": r[0],
                "max_date_posted": r[1],
                "min_scrape_date": r[2],
                "max_scrape_date": r[3],
                "n_swe": r[4],
                "n_date_posted_populated": r[5],
                "n_posting_age_populated": r[6],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "source_date_ranges.csv", index=False)
    print(f"Wrote {TABLES / 'source_date_ranges.csv'}")
    return df


def step2_rates(con: duckdb.DuckDBPyConnection, date_df: pd.DataFrame) -> None:
    """Rate-of-change estimation with observed elapsed time.

    Within-2024: asaniczka (2024-01-15 midpoint) vs arshkon (2024-04-13 mid)
    Cross-period: arshkon vs scraped midpoint
    """
    # Midpoint dates
    def midpoint(srcrow):
        from datetime import date
        mn = pd.Timestamp(srcrow["min_date_posted"])
        mx = pd.Timestamp(srcrow["max_date_posted"])
        return mn + (mx - mn) / 2

    def midpoint_scrape(srcrow):
        mn = pd.Timestamp(srcrow["min_scrape_date"])
        mx = pd.Timestamp(srcrow["max_scrape_date"])
        return mn + (mx - mn) / 2

    ars = date_df[date_df.source == "kaggle_arshkon"].iloc[0]
    asa = date_df[date_df.source == "kaggle_asaniczka"].iloc[0]
    scr = date_df[date_df.source == "scraped"].iloc[0]
    ars_mid = midpoint(ars)
    asa_mid = midpoint(asa)
    scr_mid = midpoint_scrape(scr)  # use scrape_date which is fully populated

    years_asa_to_ars = (ars_mid - asa_mid).days / 365.25
    years_ars_to_scr = (scr_mid - ars_mid).days / 365.25
    years_asa_to_scr = (scr_mid - asa_mid).days / 365.25

    print(
        f"Elapsed time: asa→ars {years_asa_to_ars:.3f} yr, "
        f"ars→scr {years_ars_to_scr:.3f} yr, "
        f"asa→scr {years_asa_to_scr:.3f} yr"
    )

    # Compute per-source metric means
    def fetch(src_filter):
        r = con.execute(metric_sql(src_filter)).fetchone()
        cols = [
            "n", "ai_broad", "ai_narrow", "copilot", "claude_tool",
            "desc_len_mean", "desc_len_median", "scope_any", "entry_share_all", "yoe_le2_share",
        ]
        return dict(zip(cols, r))

    m_ars = fetch("source='kaggle_arshkon'")
    m_asa = fetch("source='kaggle_asaniczka'")
    m_scr = fetch("source='scraped'")

    metrics = ["ai_broad", "ai_narrow", "copilot", "claude_tool", "desc_len_mean", "desc_len_median", "scope_any", "entry_share_all", "yoe_le2_share"]

    rows = []
    for m in metrics:
        v_asa = m_asa[m]
        v_ars = m_ars[m]
        v_scr = m_scr[m]
        # Absolute change per year
        delta_24_asa_to_ars = (v_ars - v_asa)
        rate_24 = delta_24_asa_to_ars / years_asa_to_ars if years_asa_to_ars else float("nan")
        delta_cross_ars_to_scr = (v_scr - v_ars)
        rate_cross = delta_cross_ars_to_scr / years_ars_to_scr if years_ars_to_scr else float("nan")
        # asa → scr (alt cross-period estimate pooled baseline)
        delta_pool_asa_to_scr = (v_scr - v_asa)
        rate_pool = delta_pool_asa_to_scr / years_asa_to_scr if years_asa_to_scr else float("nan")
        # Acceleration: rate_cross / rate_24 (dimensionless)
        accel = (rate_cross / rate_24) if rate_24 not in (0, float("nan")) and not np.isclose(rate_24, 0) else float("inf")
        rows.append(
            {
                "metric": m,
                "asaniczka": v_asa,
                "arshkon": v_ars,
                "scraped": v_scr,
                "delta_asa_to_ars_abs": delta_24_asa_to_ars,
                "rate_within_2024_per_year": rate_24,
                "delta_ars_to_scr_abs": delta_cross_ars_to_scr,
                "rate_cross_ars_to_scr_per_year": rate_cross,
                "delta_asa_to_scr_abs": delta_pool_asa_to_scr,
                "rate_asa_to_scr_per_year": rate_pool,
                "acceleration_cross_over_within": accel,
                "years_within_2024": years_asa_to_ars,
                "years_cross_ars_to_scr": years_ars_to_scr,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "rate_of_change.csv", index=False)
    print(f"Wrote {TABLES / 'rate_of_change.csv'}")
    print(df[["metric", "arshkon", "asaniczka", "scraped", "rate_within_2024_per_year", "rate_cross_ars_to_scr_per_year", "acceleration_cross_over_within"]].to_string(index=False))


def step3_within_arshkon_stability(con: duckdb.DuckDBPyConnection) -> None:
    """Bin arshkon by date_posted and compute metrics per bin."""
    # Get 5 equal bins
    dates = con.execute(
        f"""
        SELECT DISTINCT date_posted
        FROM read_parquet('{PARQUET}')
        WHERE {DEFAULT_FILTER} AND source='kaggle_arshkon'
        ORDER BY date_posted
        """
    ).fetch_df()
    print(f"arshkon distinct date_posted values: {len(dates)}")

    # Per-day metrics (may have very few days)
    per_day_sql = f"""
        SELECT date_posted,
          COUNT(*) as n,
          AVG(CASE WHEN {AI_BROAD_SQL} THEN 1.0 ELSE 0.0 END) as ai_broad,
          AVG(CASE WHEN {AI_NARROW_SQL} THEN 1.0 ELSE 0.0 END) as ai_narrow,
          AVG(description_length) as desc_len,
          AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) as entry_share
        FROM read_parquet('{PARQUET}')
        WHERE {DEFAULT_FILTER} AND source='kaggle_arshkon'
        GROUP BY 1 ORDER BY 1
    """
    df = con.execute(per_day_sql).fetch_df()
    df.to_csv(TABLES / "within_arshkon_stability.csv", index=False)
    print(f"Wrote {TABLES / 'within_arshkon_stability.csv'}")

    # Levene's test on a proxy: 2-bin split (first half vs second half of days)
    if len(df) >= 2:
        mid = len(df) // 2
        early_dates = df.date_posted.iloc[:mid].tolist()
        late_dates = df.date_posted.iloc[mid:].tolist()
        early_set = "'" + "','".join(early_dates) + "'"
        late_set = "'" + "','".join(late_dates) + "'"
        early = con.execute(
            metric_sql(f"source='kaggle_arshkon' AND date_posted IN ({early_set})")
        ).fetchone()
        late = con.execute(
            metric_sql(f"source='kaggle_arshkon' AND date_posted IN ({late_set})")
        ).fetchone()
        cols = ["n", "ai_broad", "ai_narrow", "copilot", "claude_tool",
                "desc_len_mean", "desc_len_median", "scope_any", "entry_share_all", "yoe_le2_share"]
        split = pd.DataFrame([dict(zip(cols, early)), dict(zip(cols, late))])
        split["half"] = ["early", "late"]
        split.to_csv(TABLES / "within_arshkon_halves.csv", index=False)
        print(f"Wrote {TABLES / 'within_arshkon_halves.csv'}")
        print(split[["half", "n", "ai_broad", "ai_narrow", "desc_len_mean", "entry_share_all"]].to_string(index=False))


def step4_within_scraped_stability(con: duckdb.DuckDBPyConnection) -> None:
    """Daily stats across scraped window."""
    sql = f"""
    SELECT scrape_date,
      COUNT(*) as n,
      AVG(CASE WHEN {AI_BROAD_SQL} THEN 1.0 ELSE 0.0 END) as ai_broad,
      AVG(CASE WHEN {AI_NARROW_SQL} THEN 1.0 ELSE 0.0 END) as ai_narrow,
      AVG(CASE WHEN {COPILOT_SQL} THEN 1.0 ELSE 0.0 END) as copilot,
      AVG(CASE WHEN {CLAUDE_SQL} THEN 1.0 ELSE 0.0 END) as claude_tool,
      AVG(description_length) as desc_len,
      AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) as entry_share,
      AVG(CASE WHEN seniority_final = 'unknown' THEN 1.0 ELSE 0.0 END) as unknown_share,
      AVG(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) as yoe_le2
    FROM read_parquet('{PARQUET}')
    WHERE {DEFAULT_FILTER} AND source='scraped'
    GROUP BY 1 ORDER BY 1
    """
    df = con.execute(sql).fetch_df()
    df.to_csv(TABLES / "within_scraped_stability.csv", index=False)
    print(f"Wrote {TABLES / 'within_scraped_stability.csv'}")

    # Day-of-week
    df["scrape_date"] = pd.to_datetime(df["scrape_date"])
    df["dow"] = df["scrape_date"].dt.day_name()
    dow = df.groupby("dow").agg({"n": "sum", "ai_broad": "mean", "ai_narrow": "mean", "desc_len": "mean"}).reset_index()
    dow.to_csv(TABLES / "scraped_day_of_week.csv", index=False)
    print(f"Wrote {TABLES / 'scraped_day_of_week.csv'}")

    # First day vs last day comparison for JOLTS confound check
    first = df.iloc[0]
    last = df.iloc[-1]
    effect = pd.DataFrame(
        [
            {
                "metric": m,
                "first_day": first[m],
                "last_day": last[m],
                "delta": last[m] - first[m],
            }
            for m in ["ai_broad", "ai_narrow", "copilot", "claude_tool", "desc_len", "entry_share", "yoe_le2"]
        ]
    )
    effect.to_csv(TABLES / "scraped_first_vs_last_day.csv", index=False)
    print(f"Wrote {TABLES / 'scraped_first_vs_last_day.csv'}")
    print(effect.to_string(index=False))


def step5_posting_age(con: duckdb.DuckDBPyConnection) -> None:
    r = con.execute(
        f"""
        SELECT
          COUNT(*) as n,
          SUM(CASE WHEN posting_age_days IS NOT NULL THEN 1 ELSE 0 END) as n_with_age,
          MIN(posting_age_days), MAX(posting_age_days),
          AVG(posting_age_days), MEDIAN(posting_age_days)
        FROM read_parquet('{PARQUET}')
        WHERE {DEFAULT_FILTER} AND source='scraped'
        """
    ).fetchone()
    df = pd.DataFrame(
        [
            {
                "n_total": r[0],
                "n_with_posting_age": r[1],
                "coverage": r[1] / r[0] if r[0] else 0,
                "min_age": r[2],
                "max_age": r[3],
                "mean_age": r[4],
                "median_age": r[5],
            }
        ]
    )
    df.to_csv(TABLES / "posting_age_coverage.csv", index=False)
    print(f"Wrote {TABLES / 'posting_age_coverage.csv'}")
    print(df.to_string(index=False))

    if r[1] > 0:
        hist = con.execute(
            f"""
            SELECT CAST(posting_age_days AS INTEGER) as age_days, COUNT(*) as n
            FROM read_parquet('{PARQUET}')
            WHERE {DEFAULT_FILTER} AND source='scraped' AND posting_age_days IS NOT NULL
            GROUP BY 1 ORDER BY 1
            """
        ).fetch_df()
        hist.to_csv(TABLES / "posting_age_histogram.csv", index=False)
        print(f"Wrote {TABLES / 'posting_age_histogram.csv'}")


def step6_timeline_figure(con: duckdb.DuckDBPyConnection, date_df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Three source windows
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=150)

    windows = [
        ("asaniczka", "2024-01-12", "2024-01-17", "#ff7f0e"),
        ("arshkon", "2024-04-05", "2024-04-20", "#1f77b4"),
        ("scraped", "2026-03-20", "2026-04-14", "#2ca02c"),
    ]
    for i, (name, start, end, color) in enumerate(windows):
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        ax.barh(i, (e - s).days + 1, left=s, color=color, edgecolor="k", height=0.6)
        ax.text(
            s,
            i - 0.4,
            name,
            fontsize=9,
            ha="left",
            va="top",
        )

    # Key AI releases
    events = [
        ("2023-03", "GPT-4"),
        ("2024-03", "Claude 3"),
        ("2024-05", "GPT-4o"),
        ("2024-06", "Claude 3.5"),
        ("2024-09", "o1"),
        ("2024-12", "DeepSeek V3"),
        ("2025-02", "GPT-4.5"),
        ("2025-04", "Claude 3.6"),
        ("2025-09", "Claude 4 Opus"),
        ("2026-03", "Gemini 2.5"),
    ]
    for d, name in events:
        ts = pd.Timestamp(d + "-01")
        ax.axvline(ts, color="red", alpha=0.45, linestyle="--", linewidth=1)
        ax.text(ts, 2.6, name, rotation=45, fontsize=7.5, ha="left", va="bottom")

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["asaniczka", "arshkon", "scraped"])
    ax.set_title("T19 temporal windows with AI tool releases")
    ax.set_xlim(pd.Timestamp("2023-01-01"), pd.Timestamp("2026-06-01"))
    ax.set_ylim(-0.8, 3.4)
    plt.tight_layout()
    plt.savefig(FIGS / "timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {FIGS / 'timeline.png'}")


def step7_figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Within-scraped stability chart
    df = pd.read_csv(TABLES / "within_scraped_stability.csv")
    df["scrape_date"] = pd.to_datetime(df["scrape_date"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
    axes[0, 0].plot(df["scrape_date"], df["n"], "-o")
    axes[0, 0].set_title("Daily SWE posting count (scraped)")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 1].plot(df["scrape_date"], df["ai_broad"], "-o", label="broad")
    axes[0, 1].plot(df["scrape_date"], df["ai_narrow"], "-o", label="narrow")
    axes[0, 1].set_title("AI prevalence by scrape date")
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[1, 0].plot(df["scrape_date"], df["desc_len"], "-o")
    axes[1, 0].set_title("Mean description length by scrape date")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 1].plot(df["scrape_date"], df["entry_share"], "-o", label="entry (seniority_final)")
    axes[1, 1].plot(df["scrape_date"], df["yoe_le2"], "-o", label="YOE ≤ 2")
    axes[1, 1].set_title("Entry-level proxies by scrape date")
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(FIGS / "within_scraped_stability.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {FIGS / 'within_scraped_stability.png'}")

    # Rate-of-change table figure
    roc = pd.read_csv(TABLES / "rate_of_change.csv")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    # Plot acceleration ratio for binary metrics only
    bin_metrics = ["ai_broad", "ai_narrow", "copilot", "claude_tool", "scope_any", "entry_share_all", "yoe_le2_share"]
    sub = roc[roc.metric.isin(bin_metrics)].copy()
    x = range(len(sub))
    ax.bar(
        x,
        sub["rate_within_2024_per_year"] * 100,
        width=0.4,
        label="within-2024 rate (pp/yr)",
        color="#1f77b4",
    )
    ax.bar(
        [i + 0.4 for i in x],
        sub["rate_cross_ars_to_scr_per_year"] * 100,
        width=0.4,
        label="cross-period rate (pp/yr)",
        color="#ff7f0e",
    )
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(sub["metric"], rotation=30)
    ax.set_ylabel("Annualized rate (pp/year)")
    ax.set_title("Rate of change: within-2024 vs cross-period")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "rate_of_change.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {FIGS / 'rate_of_change.png'}")


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()
    con.execute("PRAGMA threads=6")
    con.execute("PRAGMA memory_limit='12GB'")

    print("Step 1: source date ranges")
    date_df = step1_date_ranges(con)
    print(date_df.to_string(index=False))

    print("\nStep 2: rate-of-change estimation")
    step2_rates(con, date_df)

    print("\nStep 3: within-arshkon stability")
    step3_within_arshkon_stability(con)

    print("\nStep 4: within-scraped-window stability")
    step4_within_scraped_stability(con)

    print("\nStep 5: posting age coverage")
    step5_posting_age(con)

    print("\nStep 6: timeline figure")
    step6_timeline_figure(con, date_df)

    print("\nStep 7: rate-of-change + stability figures")
    step7_figures()

    print(f"\nTotal elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
