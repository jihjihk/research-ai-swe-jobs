"""T19 — Temporal patterns & rate-of-change estimation.

Honest characterization of our temporal structure (three snapshots, not a
time series). Outputs:

  Step 1: rate-of-change estimation across windows.
  Step 2: within-arshkon stability (internal date bins).
  Step 3: scraper yield characterization.
  Step 4: posting age analysis (posting_age_days).
  Step 5: within-scraped stability + day-of-week.
  Step 6: timeline figure annotated with AI releases.

Tables → exploration/tables/T19/
Figures → exploration/figures/T19/
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO = Path("/home/jihgaboot/gabor/job-research").resolve()
DATA = REPO / "data" / "unified.parquet"
OUT_TAB = REPO / "exploration" / "tables" / "T19"
OUT_FIG = REPO / "exploration" / "figures" / "T19"
OUT_TAB.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

SEED = 42

# V1-validated ai_strict (full, including fine-tuning)
AI_STRICT_RE = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|"
    r"llamaindex|langchain|prompt engineering|fine[- ]tun(?:e|ed|ing)|rag|"
    r"vector databas(?:e|es)|pinecone|huggingface|hugging face)\b",
    re.IGNORECASE,
)
AI_STRICT_CORE_RE = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|"
    r"llamaindex|langchain|prompt engineering|rag|vector databas(?:e|es)|"
    r"pinecone|huggingface|hugging face)\b",
    re.IGNORECASE,
)

SCOPE_RE = re.compile(
    r"\b(ownership|end[\s\-]to[\s\-]end|cross[\s\-]functional|initiative(?:s)?|stakeholder(?:s)?)\b",
    re.IGNORECASE,
)

_TECH_LIST = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go ", "rust",
    "kubernetes", "docker", "aws", "azure", "gcp", "sql", "react", "angular",
    "spring", "django", "fastapi", "flask", "pytorch", "tensorflow",
    "jenkins", "terraform", "git", "rest api", "graphql", "kafka",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "spark",
    "ci/cd", "linux", "scala", "ruby", "swift", "kotlin", "php",
    ".net", "node.js", "express", "airflow", "snowflake", "dbt",
    "tableau", "power bi", "hadoop", "databricks",
]
_TECH_RE = re.compile(
    r"(?<![a-z0-9])(?:" + "|".join(re.escape(t) for t in _TECH_LIST) + r")(?![a-z0-9])",
    re.IGNORECASE,
)


def count_techs(t: str) -> int:
    if not isinstance(t, str) or not t:
        return 0
    return len(set(m.group(0).lower() for m in _TECH_RE.finditer(t)))


def load_swe() -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    q = f"""
    SELECT uid, source, period, date_posted, scrape_date, scrape_week,
           posting_age_days,
           description, description_core_llm, llm_extraction_coverage,
           seniority_final, seniority_native,
           yoe_min_years_llm, yoe_extracted,
           is_swe, is_aggregator, swe_classification_tier
    FROM read_parquet('{DATA}')
    WHERE source_platform='linkedin' AND is_english AND date_flag='ok' AND is_swe
    """
    df = con.execute(q).fetchdf()
    df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")
    df["scrape_date"] = pd.to_datetime(df["scrape_date"], errors="coerce")
    return df


def compute_metrics_row(df: pd.DataFrame) -> dict:
    """Return a dict of key metric values on df (SWE LinkedIn slice)."""
    n = len(df)
    if n == 0:
        return {"n": 0}

    desc = df["description"].fillna("")
    ai = desc.str.contains(AI_STRICT_RE).mean()
    scope = desc.str.contains(SCOPE_RE).mean()
    length_median = float(desc.str.len().median())
    length_mean = float(desc.str.len().mean())

    # Tech count uses LLM-text fallback to raw.
    def pick(row):
        if row.get("llm_extraction_coverage") == "labeled" and isinstance(row.get("description_core_llm"), str):
            return row["description_core_llm"]
        return row.get("description") or ""
    texts = df.apply(pick, axis=1)
    tech_counts = texts.apply(count_techs)
    tech_median = float(tech_counts.median())
    tech_mean = float(tech_counts.mean())

    # Entry shares (primary: yoe_min_years_llm ≤ 2)
    yoe_llm = df["yoe_min_years_llm"]
    mask_yoe = yoe_llm.notna()
    yoe_llm_le2 = float(((yoe_llm <= 2) & mask_yoe).sum() / mask_yoe.sum()) if mask_yoe.sum() > 0 else float("nan")
    yoe_llm_ge5 = float(((yoe_llm >= 5) & mask_yoe).sum() / mask_yoe.sum()) if mask_yoe.sum() > 0 else float("nan")
    # seniority_final entry
    sen = df["seniority_final"].fillna("unknown")
    entry_final = float((sen == "entry").mean())
    # seniority_native entry (arshkon proxy for sanity)
    sen_native = df["seniority_native"].fillna("unknown")
    entry_native = float((sen_native == "entry").mean())
    # yoe_extracted
    yoe_rule = df["yoe_extracted"]
    mask_rule = yoe_rule.notna()
    yoe_rule_le2 = float(((yoe_rule <= 2) & mask_rule).sum() / mask_rule.sum()) if mask_rule.sum() > 0 else float("nan")

    return {
        "n": n,
        "ai_strict_prev": float(ai),
        "scope_prev": float(scope),
        "length_median": length_median,
        "length_mean": length_mean,
        "tech_count_median": tech_median,
        "tech_count_mean": tech_mean,
        "j3_yoe_llm_le2": yoe_llm_le2,
        "s4_yoe_llm_ge5": yoe_llm_ge5,
        "entry_seniority_final": entry_final,
        "entry_seniority_native": entry_native,
        "j3_yoe_extracted_le2": yoe_rule_le2,
        "n_yoe_llm_labeled": int(mask_yoe.sum()),
        "n_yoe_rule": int(mask_rule.sum()),
    }


def window_midpoint(df: pd.DataFrame) -> pd.Timestamp | None:
    """Return midpoint of date_posted for arshkon/asaniczka, else midpoint of scrape_date."""
    for col in ("date_posted", "scrape_date"):
        if col in df.columns and df[col].notna().any():
            vals = df[col].dropna()
            return vals.min() + (vals.max() - vals.min()) / 2
    return None


def step1_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    """Metric snapshots per source window, plus cross-period and within-2024 annualized rates."""
    windows = {
        "arshkon_2024": df[df["source"] == "kaggle_arshkon"],
        "asaniczka_2024": df[df["source"] == "kaggle_asaniczka"],
        "scraped_2026_03": df[(df["source"] == "scraped") & (df["period"] == "2026-03")],
        "scraped_2026_04": df[(df["source"] == "scraped") & (df["period"] == "2026-04")],
    }
    rows = []
    for name, sub in windows.items():
        m = compute_metrics_row(sub)
        min_date = None
        max_date = None
        if "arshkon" in name or "asaniczka" in name:
            col = "date_posted"
        else:
            col = "scrape_date"
        if sub[col].notna().any():
            min_date = sub[col].min().strftime("%Y-%m-%d")
            max_date = sub[col].max().strftime("%Y-%m-%d")
            mid = (sub[col].min() + (sub[col].max() - sub[col].min()) / 2).strftime("%Y-%m-%d")
        else:
            mid = None
        rows.append({"window": name, "date_min": min_date, "date_max": max_date, "date_mid": mid, **m})
    snap = pd.DataFrame(rows)
    snap.to_csv(OUT_TAB / "rate_snapshots.csv", index=False)

    # Compute rate changes across and within 2024.
    metrics = ["ai_strict_prev", "scope_prev", "length_median", "tech_count_median",
               "j3_yoe_llm_le2", "s4_yoe_llm_ge5", "entry_seniority_final",
               "entry_seniority_native", "j3_yoe_extracted_le2"]

    def days_between(a, b):
        return (pd.to_datetime(b) - pd.to_datetime(a)).days

    # Snapshots
    arsh_mid = snap.loc[snap["window"] == "arshkon_2024", "date_mid"].iloc[0]
    asan_mid = snap.loc[snap["window"] == "asaniczka_2024", "date_mid"].iloc[0]
    scr3_mid = snap.loc[snap["window"] == "scraped_2026_03", "date_mid"].iloc[0]
    scr4_mid = snap.loc[snap["window"] == "scraped_2026_04", "date_mid"].iloc[0]

    rate_rows = []
    for metric in metrics:
        arsh_v = float(snap.loc[snap["window"] == "arshkon_2024", metric].iloc[0])
        asan_v = float(snap.loc[snap["window"] == "asaniczka_2024", metric].iloc[0])
        scr3_v = float(snap.loc[snap["window"] == "scraped_2026_03", metric].iloc[0])
        scr4_v = float(snap.loc[snap["window"] == "scraped_2026_04", metric].iloc[0])

        # Within-2024: asaniczka → arshkon (different calendar dates within 2024)
        days_w24 = days_between(asan_mid, arsh_mid)
        w24_delta = arsh_v - asan_v
        w24_rate = (w24_delta / days_w24 * 365) if days_w24 != 0 else float("nan")

        # Within-scraped: 2026-03 → 2026-04
        days_w26 = days_between(scr3_mid, scr4_mid)
        w26_delta = scr4_v - scr3_v
        w26_rate = (w26_delta / days_w26 * 365) if days_w26 != 0 else float("nan")

        # Cross-period: pooled-2024 (mean of arshkon + asaniczka weighted by n)
        n_arsh = int(snap.loc[snap["window"] == "arshkon_2024", "n"].iloc[0])
        n_asan = int(snap.loc[snap["window"] == "asaniczka_2024", "n"].iloc[0])
        pooled_v = (arsh_v * n_arsh + asan_v * n_asan) / (n_arsh + n_asan) if (n_arsh + n_asan) else float("nan")
        # Pooled 2024 midpoint — weight by n
        pooled_mid = pd.to_datetime([arsh_mid] * n_arsh + [asan_mid] * n_asan).mean()
        # Scraped pooled
        n_scr3 = int(snap.loc[snap["window"] == "scraped_2026_03", "n"].iloc[0])
        n_scr4 = int(snap.loc[snap["window"] == "scraped_2026_04", "n"].iloc[0])
        scr_v = (scr3_v * n_scr3 + scr4_v * n_scr4) / (n_scr3 + n_scr4)
        scr_mid = pd.to_datetime([scr3_mid] * n_scr3 + [scr4_mid] * n_scr4).mean()

        days_cross = (scr_mid - pooled_mid).days
        cross_delta = scr_v - pooled_v
        cross_rate = (cross_delta / days_cross * 365) if days_cross != 0 else float("nan")

        accel_24 = abs(cross_rate / w24_rate) if (w24_rate and not np.isnan(w24_rate) and w24_rate != 0) else float("nan")

        rate_rows.append({
            "metric": metric,
            "arshkon_2024": arsh_v,
            "asaniczka_2024": asan_v,
            "pooled_2024": pooled_v,
            "scraped_2026_03": scr3_v,
            "scraped_2026_04": scr4_v,
            "scraped_pooled": scr_v,
            "within_2024_delta": w24_delta,
            "within_2024_days": days_w24,
            "within_2024_annualized_rate": w24_rate,
            "within_2026_delta": w26_delta,
            "within_2026_days": days_w26,
            "within_2026_annualized_rate": w26_rate,
            "cross_period_delta": cross_delta,
            "cross_period_days": days_cross,
            "cross_period_annualized_rate": cross_rate,
            "acceleration_ratio_cross_over_w24": accel_24,
        })
    rt = pd.DataFrame(rate_rows)
    rt.to_csv(OUT_TAB / "rate_of_change.csv", index=False)
    return snap, rt


def step2_within_arshkon(df: pd.DataFrame) -> pd.DataFrame:
    """Bin arshkon by internal date ranges and compute metrics."""
    arsh = df[df["source"] == "kaggle_arshkon"].copy()
    # Three bins by tercile of date_posted.
    arsh = arsh.sort_values("date_posted")
    arsh["bin"] = pd.qcut(arsh["date_posted"].astype("int64"), q=3,
                          labels=["bin1_early", "bin2_mid", "bin3_late"])
    rows = []
    for b, sub in arsh.groupby("bin", observed=True):
        m = compute_metrics_row(sub)
        rows.append({
            "bin": str(b),
            "n": len(sub),
            "date_min": sub["date_posted"].min().strftime("%Y-%m-%d"),
            "date_max": sub["date_posted"].max().strftime("%Y-%m-%d"),
            **m,
        })
    out = pd.DataFrame(rows)
    # Also at posting-date granularity
    day_rows = []
    for d, sub in arsh.groupby(arsh["date_posted"].dt.strftime("%Y-%m-%d")):
        if len(sub) < 30:
            continue
        m = compute_metrics_row(sub)
        day_rows.append({"date": d, **m})
    day_df = pd.DataFrame(day_rows)
    day_df.to_csv(OUT_TAB / "within_arshkon_by_day.csv", index=False)
    out.to_csv(OUT_TAB / "within_arshkon_by_tercile.csv", index=False)
    return out


def step3_scraper_yield(df: pd.DataFrame) -> pd.DataFrame:
    """Daily SWE posting counts across scrape dates; first day vs subsequent (stock vs flow)."""
    scr = df[df["source"] == "scraped"].copy()
    by_date = scr.groupby(scr["scrape_date"].dt.strftime("%Y-%m-%d")).agg(
        n=("uid", "size"),
    ).reset_index()
    by_date.columns = ["scrape_date", "n_swe"]

    # First scrape day vs rest: characterize stock vs flow
    first_day = by_date["scrape_date"].min()
    scr["is_first_day"] = (scr["scrape_date"].dt.strftime("%Y-%m-%d") == first_day).astype(int)

    rows = []
    for is_first in (1, 0):
        sub = scr[scr["is_first_day"] == is_first]
        m = compute_metrics_row(sub)
        rows.append({"window": "first_day" if is_first else "subsequent", **m})
    first_vs_rest = pd.DataFrame(rows)
    first_vs_rest.to_csv(OUT_TAB / "first_day_vs_subsequent.csv", index=False)

    # Metrics per scrape_date (for stability)
    metric_rows = []
    for d, sub in scr.groupby(scr["scrape_date"].dt.strftime("%Y-%m-%d")):
        if len(sub) < 100:
            continue
        m = compute_metrics_row(sub)
        metric_rows.append({"scrape_date": d, **m})
    metrics_by_day = pd.DataFrame(metric_rows)
    metrics_by_day.to_csv(OUT_TAB / "scraper_yield_by_day.csv", index=False)

    by_date.to_csv(OUT_TAB / "scraper_daily_counts.csv", index=False)
    return by_date, first_vs_rest, metrics_by_day


def step4_posting_age(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of posting_age_days where available."""
    scr = df[(df["source"] == "scraped") & df["posting_age_days"].notna()].copy()
    if len(scr) == 0:
        pd.DataFrame([{"n": 0, "note": "posting_age_days unavailable"}]).to_csv(
            OUT_TAB / "posting_age_distribution.csv", index=False)
        return pd.DataFrame()

    age = scr["posting_age_days"]
    stats = {
        "n": int(len(age)),
        "n_missing": int(df[df["source"] == "scraped"]["posting_age_days"].isna().sum()),
        "mean": float(age.mean()),
        "median": float(age.median()),
        "p10": float(age.quantile(0.10)),
        "p25": float(age.quantile(0.25)),
        "p75": float(age.quantile(0.75)),
        "p90": float(age.quantile(0.90)),
        "p99": float(age.quantile(0.99)),
        "min": float(age.min()),
        "max": float(age.max()),
    }
    pd.DataFrame([stats]).to_csv(OUT_TAB / "posting_age_distribution.csv", index=False)

    # Binned counts
    bins = [-0.5, 0.5, 1.5, 3.5, 7.5, 14.5, 30.5, 90.5, 365.5, 10000]
    labels = ["0d", "1d", "2-3d", "4-7d", "8-14d", "15-30d", "31-90d", "91-365d", ">365d"]
    age_binned = pd.cut(age, bins=bins, labels=labels)
    buckets = age_binned.value_counts().sort_index().reset_index()
    buckets.columns = ["bucket", "count"]
    buckets["share"] = buckets["count"] / buckets["count"].sum()
    buckets.to_csv(OUT_TAB / "posting_age_buckets.csv", index=False)
    return buckets


def step5_within_scraped_and_dow(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week analysis + within-scraped-window stability."""
    scr = df[df["source"] == "scraped"].copy()
    scr["dow"] = scr["scrape_date"].dt.day_name()
    scr["dow_num"] = scr["scrape_date"].dt.dayofweek

    rows = []
    for dow_num, sub in scr.groupby("dow_num"):
        m = compute_metrics_row(sub)
        rows.append({"dow_num": dow_num, "dow": sub["dow"].iloc[0], **m})
    dow_df = pd.DataFrame(rows).sort_values("dow_num")
    dow_df.to_csv(OUT_TAB / "dow_analysis.csv", index=False)

    # Also by ISO week
    rows = []
    for wk, sub in scr.groupby(scr["scrape_date"].dt.isocalendar().week):
        if len(sub) < 100:
            continue
        m = compute_metrics_row(sub)
        rows.append({"iso_week": int(wk), **m})
    wk_df = pd.DataFrame(rows)
    wk_df.to_csv(OUT_TAB / "iso_week_analysis.csv", index=False)
    return dow_df, wk_df


def step6_timeline_figure(snap: pd.DataFrame, by_date: pd.DataFrame,
                          within_arshkon: pd.DataFrame) -> None:
    """Render a timeline figure with AI releases and our three snapshots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        ai_releases = [
            ("GPT-4",        "2023-03-14"),
            ("Claude 3",     "2024-03-04"),
            ("GPT-4o",       "2024-05-13"),
            ("Claude 3.5 Sonnet", "2024-06-20"),
            ("o1",           "2024-09-12"),
            ("DeepSeek V3",  "2024-12-26"),
            ("GPT-4.5",      "2025-02-27"),
            ("Claude 3.6 Sonnet", "2025-04-01"),
            ("Claude 4 Opus", "2025-09-01"),
            ("Gemini 2.5 Pro", "2026-03-20"),
        ]

        fig, ax = plt.subplots(figsize=(12, 4.5))

        # Data windows as horizontal spans
        import datetime
        spans = [
            ("arshkon", "2024-04-05", "2024-04-20", "#1f77b4"),
            ("asaniczka", "2024-01-12", "2024-01-17", "#ff7f0e"),
            ("scraped", "2026-03-19", "2026-04-18", "#2ca02c"),
        ]
        y = 1.0
        for label, a, b, c in spans:
            ax.barh(y, width=(pd.to_datetime(b) - pd.to_datetime(a)).days + 1,
                    left=pd.to_datetime(a), height=0.3, color=c, alpha=0.7,
                    edgecolor="k", linewidth=0.5)
            ax.text(pd.to_datetime(a), y + 0.2, label, fontsize=8)
            y += 0.5

        # AI releases as vertical lines
        for name, d in ai_releases:
            dt = pd.to_datetime(d)
            ax.axvline(dt, color="#d62728", linestyle="--", alpha=0.6, lw=0.8)
            ax.text(dt, y + 0.1, name, rotation=60, fontsize=7, va="bottom", ha="left")

        ax.set_yticks([])
        ax.set_xlim(pd.to_datetime("2023-01-01"), pd.to_datetime("2026-06-01"))
        ax.set_ylim(0.5, y + 1.5)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
        ax.set_title("T19. Timeline of data windows and AI release events")
        fig.tight_layout()
        fig.savefig(OUT_FIG / "fig1_timeline.png", dpi=120)
        plt.close(fig)

        # Fig 2: daily scraper SWE yield
        if not by_date.empty:
            by_date2 = by_date.copy()
            by_date2["scrape_date"] = pd.to_datetime(by_date2["scrape_date"])
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(by_date2["scrape_date"], by_date2["n_swe"], color="#2ca02c")
            ax.set_title("T19. Daily SWE posting yield (scraped)")
            ax.set_ylabel("SWE postings")
            ax.set_xlabel("scrape_date")
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(OUT_FIG / "fig2_daily_yield.png", dpi=120)
            plt.close(fig)
    except Exception as e:
        print(f"[figures] {e}")


def main():
    print("Loading SWE data …")
    df = load_swe()
    print(f"  n={len(df):,} SWE LinkedIn rows")

    print("Step 1: rate of change …")
    snap, rt = step1_rate_of_change(df)
    print("  snapshots:")
    print(snap.to_string())
    print("  rates:")
    print(rt.to_string())

    print("Step 2: within-arshkon stability …")
    w_ark = step2_within_arshkon(df)
    print(w_ark.to_string())

    print("Step 3: scraper yield …")
    by_date, first_vs_rest, metrics_by_day = step3_scraper_yield(df)
    print("  first-day vs rest:")
    print(first_vs_rest.to_string())

    print("Step 4: posting age …")
    age = step4_posting_age(df)
    if not age.empty:
        print(age.to_string())

    print("Step 5: DoW + ISO week …")
    dow_df, wk_df = step5_within_scraped_and_dow(df)
    print(dow_df.to_string())

    print("Step 6: timeline figure …")
    step6_timeline_figure(snap, by_date, w_ark)

    print("Done.")


if __name__ == "__main__":
    main()
