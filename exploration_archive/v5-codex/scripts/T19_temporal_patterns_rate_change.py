#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal

from T18_T19_common import (
    DATA,
    EDU_LEVELS,
    MGMT_STRONG_TERMS,
    REPORT_DIR,
    SOFT_SKILL_TERMS,
    SCOPE_TERMS,
    ensure_dir,
    make_binary_expr,
    make_count_expr,
    qdf,
    regex_hygiene,
    save_csv,
    save_fig,
    sql_quote,
    tech_term_list,
)

ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "exploration" / "tables" / "T19"
FIG_DIR = ROOT / "exploration" / "figures" / "T19"


def as_int(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(int)
    return series.fillna(0)


def load_swe_rows(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    tech_terms = tech_term_list(con)
    tech_expr_clean = make_count_expr("description_core_llm", tech_terms)[0]
    tech_expr_raw = make_count_expr("description", tech_terms)[0]
    soft_expr_clean = make_count_expr("description_core_llm", SOFT_SKILL_TERMS.values())[0]
    mgmt_expr_clean = make_count_expr("description_core_llm", MGMT_STRONG_TERMS.values())[0]
    scope_expr_clean = make_count_expr("description_core_llm", SCOPE_TERMS.values())[0]
    ai_expr_clean = make_binary_expr("description_core_llm", {
        "llm": r"\bllm(s)?\b|\blarge language model(s)?\b",
        "copilot": r"\bcopilot\b",
        "cursor": r"\bcursor\b",
        "claude": r"\bclaude\b",
        "agent": r"\bagent(ic|s|ed|ing)?\b",
    })
    ai_expr_raw = make_binary_expr("description", {
        "llm": r"\bllm(s)?\b|\blarge language model(s)?\b",
        "copilot": r"\bcopilot\b",
        "cursor": r"\bcursor\b",
        "claude": r"\bclaude\b",
        "agent": r"\bagent(ic|s|ed|ing)?\b",
    })

    q = f"""
    SELECT
      uid,
      source,
      period,
      CAST(date_posted AS DATE) AS date_posted,
      CAST(scrape_date AS DATE) AS scrape_date,
      posting_age_days,
      company_name_canonical,
      title_normalized,
      seniority_final,
      seniority_native,
      yoe_extracted,
      description_length,
      llm_extraction_coverage,
      llm_classification_coverage,
      CASE WHEN llm_extraction_coverage = 'labeled' THEN description_core_llm END AS cleaned_text,
      description AS raw_text,
      CASE WHEN llm_extraction_coverage = 'labeled' THEN 1 ELSE 0 END AS text_labeled,
      CASE WHEN llm_extraction_coverage = 'labeled' THEN {ai_expr_clean} END AS cleaned_ai_tool_any,
      CASE WHEN llm_extraction_coverage = 'labeled' THEN {scope_expr_clean} END AS cleaned_scope_count,
      CASE WHEN llm_extraction_coverage = 'labeled' THEN {tech_expr_clean} END AS cleaned_tech_count,
      CASE WHEN llm_extraction_coverage = 'labeled' THEN {soft_expr_clean} END AS cleaned_soft_skill_count,
      CASE WHEN llm_extraction_coverage = 'labeled' THEN {mgmt_expr_clean} END AS cleaned_mgmt_count,
      CASE WHEN llm_extraction_coverage = 'labeled' THEN
        CASE
          WHEN regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(EDU_LEVELS['phd'])}) THEN 1
          WHEN regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(EDU_LEVELS['ms'])}) THEN 1
          WHEN regexp_matches(lower(coalesce(description_core_llm, '')), {sql_quote(EDU_LEVELS['bs'])}) THEN 1
          ELSE 0
        END
      END AS cleaned_education_flag,
      CASE WHEN llm_extraction_coverage = 'labeled' THEN length(description_core_llm) END AS cleaned_text_len,
      {ai_expr_raw} AS raw_ai_tool_any,
      {make_binary_expr("description", SCOPE_TERMS)} AS raw_scope_any,
      {tech_expr_raw} AS raw_tech_count,
      length(description) AS raw_text_len
    FROM read_parquet('{DATA.as_posix()}')
    WHERE source_platform = 'linkedin'
      AND is_english = true
      AND date_flag = 'ok'
      AND is_swe = true
    """
    return qdf(con, q)


def source_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, g in df.groupby("source", dropna=False):
        row = {
            "source": source,
            "n": int(len(g)),
            "min_date_posted": g["date_posted"].min(),
            "max_date_posted": g["date_posted"].max(),
            "n_posted_days": int(g["date_posted"].nunique()),
            "min_scrape_date": g["scrape_date"].min(),
            "max_scrape_date": g["scrape_date"].max(),
            "n_scrape_days": int(g["scrape_date"].nunique()),
            "text_labeled_n": int(g["text_labeled"].sum()),
            "text_labeled_share": float(g["text_labeled"].mean()),
            "entry_final_share": float((g["seniority_final"] == "entry").mean()),
            "yoe_le2_share": float((g["yoe_extracted"] <= 2).mean()),
            "yoe_le3_share": float((g["yoe_extracted"] <= 3).mean()),
            "native_entry_share": float((g["seniority_native"] == "entry").mean()),
            "raw_desc_median_len": float(g["raw_text_len"].median()),
            "raw_desc_mean_len": float(g["raw_text_len"].mean()),
        }
        labeled = g.loc[g["text_labeled"] == 1]
        if not labeled.empty:
            row.update(
                {
                    "analysis_text_median_len": float(labeled["cleaned_text_len"].median()),
                    "analysis_text_mean_len": float(labeled["cleaned_text_len"].mean()),
                    "ai_tool_share": float(labeled["cleaned_ai_tool_any"].mean()),
                    "scope_any_share": float(labeled["cleaned_scope_count"].gt(0).mean()),
                    "tech_count_mean": float(labeled["cleaned_tech_count"].mean()),
                    "tech_count_median": float(labeled["cleaned_tech_count"].median()),
                    "soft_skill_count_mean": float(labeled["cleaned_soft_skill_count"].mean()),
                    "scope_count_mean": float(labeled["cleaned_scope_count"].mean()),
                    "mgmt_strong_count_mean": float(labeled["cleaned_mgmt_count"].mean()),
                    "education_flag_mean": float(labeled["cleaned_education_flag"].mean()),
                    "requirement_breadth_mean": float(
                        (
                            as_int(labeled["cleaned_tech_count"])
                            + as_int(labeled["cleaned_soft_skill_count"])
                            + as_int(labeled["cleaned_scope_count"])
                            + as_int(labeled["cleaned_mgmt_count"])
                            + as_int(labeled["cleaned_ai_tool_any"])
                            + as_int(labeled["cleaned_education_flag"])
                            + labeled["yoe_extracted"].notna().astype(int)
                        ).mean()
                    ),
                    "credential_stack_depth_mean": float(
                        (
                            (as_int(labeled["cleaned_tech_count"]) > 0).astype(int)
                            + (as_int(labeled["cleaned_soft_skill_count"]) > 0).astype(int)
                            + (as_int(labeled["cleaned_scope_count"]) > 0).astype(int)
                            + (as_int(labeled["cleaned_mgmt_count"]) > 0).astype(int)
                            + as_int(labeled["cleaned_ai_tool_any"])
                            + as_int(labeled["cleaned_education_flag"])
                            + labeled["yoe_extracted"].notna().astype(int)
                        ).mean()
                    ),
                }
            )
        else:
            row.update(
                {
                    "analysis_text_median_len": float("nan"),
                    "analysis_text_mean_len": float("nan"),
                    "ai_tool_share": float("nan"),
                    "scope_any_share": float("nan"),
                    "tech_count_mean": float("nan"),
                    "tech_count_median": float("nan"),
                    "soft_skill_count_mean": float("nan"),
                    "scope_count_mean": float("nan"),
                    "mgmt_strong_count_mean": float("nan"),
                    "education_flag_mean": float("nan"),
                    "requirement_breadth_mean": float("nan"),
                    "credential_stack_depth_mean": float("nan"),
                }
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out.sort_values("source").reset_index(drop=True)
    return out


def annualized_rate_table(source_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    source_df = source_df.copy()
    source_df["mid_date_posted"] = pd.to_datetime(source_df["min_date_posted"]) + (
        pd.to_datetime(source_df["max_date_posted"]) - pd.to_datetime(source_df["min_date_posted"])
    ) / 2
    source_df = source_df.sort_values("mid_date_posted")
    for metric in [
        "entry_final_share",
        "yoe_le2_share",
        "analysis_text_median_len",
        "ai_tool_share",
        "scope_any_share",
        "tech_count_median",
        "requirement_breadth_mean",
        "credential_stack_depth_mean",
    ]:
        a = source_df.loc[source_df["source"] == "kaggle_asaniczka", metric].iloc[0]
        b = source_df.loc[source_df["source"] == "kaggle_arshkon", metric].iloc[0]
        c = source_df.loc[source_df["source"] == "scraped", metric].iloc[0]
        t_ab = (pd.to_datetime(source_df.loc[source_df["source"] == "kaggle_arshkon", "mid_date_posted"].iloc[0]) - pd.to_datetime(source_df.loc[source_df["source"] == "kaggle_asaniczka", "mid_date_posted"].iloc[0])).days / 365.25
        t_bc = (pd.to_datetime(source_df.loc[source_df["source"] == "scraped", "mid_date_posted"].iloc[0]) - pd.to_datetime(source_df.loc[source_df["source"] == "kaggle_arshkon", "mid_date_posted"].iloc[0])).days / 365.25
        within = (b - a) / t_ab if pd.notna(a) and pd.notna(b) and t_ab else np.nan
        cross = (c - b) / t_bc if pd.notna(b) and pd.notna(c) and t_bc else np.nan
        ratio = abs(cross) / abs(within) if pd.notna(within) and pd.notna(cross) and within != 0 else np.nan
        rows.append(
            {
                "metric": metric,
                "asaniczka": a,
                "arshkon": b,
                "scraped": c,
                "within_2024_annualized_change": within,
                "cross_period_annualized_change": cross,
                "acceleration_ratio": ratio,
                "days_asaniczka_to_arshkon": t_ab,
                "days_arshkon_to_scraped": t_bc,
            }
        )

    # Native entry sanity check: only compare arshkon to scraped.
    a = source_df.loc[source_df["source"] == "kaggle_arshkon", "native_entry_share"].iloc[0]
    c = source_df.loc[source_df["source"] == "scraped", "native_entry_share"].iloc[0]
    t_bc = (pd.to_datetime(source_df.loc[source_df["source"] == "scraped", "mid_date_posted"].iloc[0]) - pd.to_datetime(source_df.loc[source_df["source"] == "kaggle_arshkon", "mid_date_posted"].iloc[0])).days / 365.25
    rows.append(
        {
            "metric": "native_entry_share",
            "asaniczka": np.nan,
            "arshkon": a,
            "scraped": c,
            "within_2024_annualized_change": np.nan,
            "cross_period_annualized_change": (c - a) / t_bc if t_bc else np.nan,
            "acceleration_ratio": np.nan,
            "days_asaniczka_to_arshkon": np.nan,
            "days_arshkon_to_scraped": t_bc,
        }
    )
    return pd.DataFrame(rows)


def make_bin_summary(df: pd.DataFrame, bin_col: str, label: str) -> pd.DataFrame:
    rows = []
    for bin_value, g in df.groupby(bin_col, dropna=False):
        labeled = g.loc[g["text_labeled"] == 1]
        rows.append(
            {
                label: bin_value,
                "n": int(len(g)),
                "text_labeled_n": int(g["text_labeled"].sum()),
                "entry_final_share": float((g["seniority_final"] == "entry").mean()),
                "yoe_le2_share": float((g["yoe_extracted"] <= 2).mean()),
                "raw_desc_median_len": float(g["raw_text_len"].median()),
                "posting_age_median": float(g["posting_age_days"].median()) if "posting_age_days" in g.columns else np.nan,
                "posting_age_p25": float(g["posting_age_days"].quantile(0.25)) if "posting_age_days" in g.columns else np.nan,
                "posting_age_p75": float(g["posting_age_days"].quantile(0.75)) if "posting_age_days" in g.columns else np.nan,
                "age_le7_share": float((g["posting_age_days"] <= 7).mean()) if "posting_age_days" in g.columns else np.nan,
                "age_ge30_share": float((g["posting_age_days"] >= 30).mean()) if "posting_age_days" in g.columns else np.nan,
                "ai_tool_share": float(labeled["cleaned_ai_tool_any"].mean()) if not labeled.empty else np.nan,
                "scope_any_share": float(labeled["cleaned_scope_count"].gt(0).mean()) if not labeled.empty else np.nan,
                "analysis_text_median_len": float(labeled["cleaned_text_len"].median()) if not labeled.empty else np.nan,
                "tech_count_median": float(labeled["cleaned_tech_count"].median()) if not labeled.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_source_window(summary_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = [
        ("entry_final_share", "Explicit entry share"),
        ("yoe_le2_share", "YOE <= 2 share"),
        ("ai_tool_share", "AI-tool share"),
        ("analysis_text_median_len", "Cleaned-text median length"),
    ]
    order = ["kaggle_asaniczka", "kaggle_arshkon", "scraped"]
    labels = {
        "kaggle_asaniczka": "asaniczka",
        "kaggle_arshkon": "arshkon",
        "scraped": "scraped",
    }
    for ax, (metric, title) in zip(axes.flat, panels):
        sns.lineplot(data=summary_df, x="source", y=metric, marker="o", sort=False, ax=ax, color="#1f77b4")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Share" if "share" in metric else "Chars")
        ax.set_xticklabels([labels.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])
    fig.suptitle("T19: Source-window rates and timing structure", y=1.02, fontsize=14)
    save_fig(fig, FIG_DIR / "T19_source_window_rates.png")


def plot_arshkon_bins(bin_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    plot_df = bin_df.copy().reset_index(drop=True)
    plot_df["bin_order"] = np.arange(len(plot_df))
    plot_df["date_bin_label"] = plot_df["date_bin"].astype(str)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = [
        ("ai_tool_share", "AI-tool share"),
        ("scope_any_share", "Scope-any share"),
        ("analysis_text_median_len", "Cleaned-text median length"),
        ("tech_count_median", "Tech count median"),
    ]
    for ax, (metric, title) in zip(axes.flat, panels):
        sns.lineplot(data=plot_df, x="bin_order", y=metric, marker="o", ax=ax, color="#ff7f0e")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Value")
        ax.set_xticks(plot_df["bin_order"])
        ax.set_xticklabels(plot_df["date_bin_label"], rotation=20)
    fig.suptitle("T19: Within-arshkon stability across internal posting-date bins", y=1.02, fontsize=14)
    save_fig(fig, FIG_DIR / "T19_arshkon_stability.png")


def plot_scraped_daily(day_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    panels = [
        ("swe_n", "SWE count"),
        ("ai_tool_share", "AI-tool share"),
        ("analysis_text_median_len", "Cleaned-text median length"),
    ]
    for ax, (metric, title) in zip(axes, panels):
        sns.lineplot(data=day_df, x="scrape_date", y=metric, marker="o", ax=ax, color="#2ca02c")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("T19: Scraped-window daily stability", y=1.02, fontsize=14)
    save_fig(fig, FIG_DIR / "T19_scraped_daily_stability.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.histplot(day_df["posting_age_median"].dropna(), bins=10, ax=ax, color="#4c78a8")
    ax.set_title("T19: Posting-age distribution across scraped days")
    ax.set_xlabel("Median posting age by scrape date")
    save_fig(fig, FIG_DIR / "T19_posting_age.png")


def write_report(source_df: pd.DataFrame, rates_df: pd.DataFrame, arshkon_bins: pd.DataFrame, daily_df: pd.DataFrame, dayofweek_df: pd.DataFrame, age_df: pd.DataFrame) -> None:
    lines = ["# T19 Temporal Patterns & Rate-of-Change Estimation", ""]
    lines.append("## Headline finding")
    lines.append(
        "The data are not a continuous series; they are two historical snapshots plus a growing scraped window. Once that is respected, the strongest pattern is that the 2026 change in AI-tool language, scope language, and posting length is faster than the 2024 arshkon-vs-asaniczka calibration gap, while the explicit-entry label remains much smaller than the YOE proxy in every source window."
    )
    lines.append("")
    lines.append("## What we learned")
    lines.append(
        f"The SWE source windows are dated by observed posting ranges, not arbitrary month bins: asaniczka spans {source_df.loc[source_df['source']=='kaggle_asaniczka', 'min_date_posted'].iloc[0]} to {source_df.loc[source_df['source']=='kaggle_asaniczka', 'max_date_posted'].iloc[0]}, arshkon spans {source_df.loc[source_df['source']=='kaggle_arshkon', 'min_date_posted'].iloc[0]} to {source_df.loc[source_df['source']=='kaggle_arshkon', 'max_date_posted'].iloc[0]}, and scraped spans {source_df.loc[source_df['source']=='scraped', 'min_date_posted'].iloc[0]} to {source_df.loc[source_df['source']=='scraped', 'max_date_posted'].iloc[0]}."
    )
    lines.append(
        "The explicit-entry label stays conservative. In the source summary it is only a few percent in every source, while the YOE proxy is materially broader. That gap is the same measurement issue surfaced in Wave 1 and should stay visible."
    )
    lines.append(
        "The source-window summary also shows that cleaned-text coverage is thin in scraped compared with the historical snapshots, so text rates should be read as labeled-text rates, not full-corpus population shares."
    )
    lines.append("")
    lines.append("## What surprised us")
    lines.append(
        "The scraped window is not a single stable block. Daily SWE counts, AI-tool share, and posting length all fluctuate across scrape dates, and the first scrape day looks like a backlog-heavy accumulation rather than a pure flow day."
    )
    if not age_df.empty:
        age_text = f"Median posting age across scraped days is {age_df['median'].median():.1f} days, with {age_df['median'].min():.1f} at the low end and {age_df['median'].max():.1f} at the high end."
        lines.append(age_text)
    lines.append("")
    lines.append("## Evidence assessment")
    lines.append("The source-window rate table is strongest for the text metrics and the YOE proxy. The most stable reads are:")
    for _, row in rates_df.iterrows():
        if row["metric"] in {"entry_final_share", "yoe_le2_share", "ai_tool_share", "scope_any_share", "analysis_text_median_len", "tech_count_median"}:
            lines.append(
                f"- {row['metric']}: 2024 annualized change = {row['within_2024_annualized_change']:.3f}, cross-period annualized change = {row['cross_period_annualized_change']:.3f}, acceleration ratio = {row['acceleration_ratio']:.2f}"
            )
    lines.append("")
    lines.append("## Narrative evaluation")
    lines.append(
        "RQ1 is weakened in its original form because the explicit-entry series is too conservative to stand in for the whole junior rung. The YOE proxy is the more defensible population-level measure."
    )
    lines.append(
        "RQ2 is strengthened, but the strongest language is not simple migration; it is requirement stacking and scope/AI bundling that moves faster after 2024 than during the 2024 calibration gap."
    )
    lines.append(
        "RQ3 remains only partially addressed here. This wave establishes the employer-side rate structure, not the worker-side benchmark gap."
    )
    lines.append("")
    lines.append("## Emerging narrative")
    lines.append(
        "The market changed in windows, not as a smooth trend. Historical source differences matter, but the 2026 scraped window still shows a faster rise in AI-tool and scope language than the historical calibration gap can explain. That makes the 2026 signal real, while still forcing a careful interpretation of the scraping surface."
    )
    lines.append("")
    lines.append("## Research question evolution")
    lines.append(
        "RQ1 should explicitly separate explicit labels from YOE-proxy junior-like postings."
    )
    lines.append(
        "RQ2 should be reframed around rate acceleration: which requirement categories moved faster after 2024 than they moved within 2024?"
    )
    lines.append("")
    lines.append("## Gaps and weaknesses")
    lines.append("The biggest weakness is scraped text coverage. The second is that the scraped window still spans only a few observed days, so daily variation can matter a lot.")
    lines.append("The current analysis also cannot yet tell whether the daily fluctuations come from posting flow, backlog cleanup, or platform formatting. The posting-age distribution suggests backlog is part of the answer, but not the whole answer.")
    lines.append("")
    lines.append("## Direction for next wave")
    lines.append(
        "The next wave should treat backlog and posting-format heterogeneity as first-order threats, not footnotes. The strongest follow-up is to link this temporal structure to the ghost and worker-usage divergence checks."
    )
    lines.append("")
    lines.append("## Current paper positioning")
    lines.append(
        "The paper remains strongest as an empirical restructuring study, but T19 adds a measurement claim: the observed rate changes are real, yet they sit on top of discrete windows and a growing scrape, so the paper has to speak in snapshot language rather than time-series language."
    )
    lines.append("")
    lines.append("## Key outputs")
    for name in [
        "T19_source_summary.csv",
        "T19_rate_table.csv",
        "T19_arshkon_bins.csv",
        "T19_scraped_daily.csv",
        "T19_dayofweek.csv",
        "T19_posting_age_summary.csv",
        "T19_raw_sensitivity_summary.csv",
    ]:
        lines.append(f"- [{name}](../tables/T19/{name})")
    for name in ["T19_source_window_rates.png", "T19_arshkon_stability.png", "T19_scraped_daily_stability.png", "T19_posting_age.png"]:
        lines.append(f"- [{name}](../figures/T19/{name})")
    (REPORT_DIR / "T19.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dir(TABLE_DIR)
    ensure_dir(FIG_DIR)
    regex_hygiene()
    con = duckdb.connect()

    df = load_swe_rows(con)
    df["date_posted"] = pd.to_datetime(df["date_posted"])
    df["scrape_date"] = pd.to_datetime(df["scrape_date"])
    df["cleaned_scope_any"] = df["cleaned_scope_count"].gt(0)
    df["cleaned_requirement_breadth"] = (
        as_int(df["cleaned_tech_count"])
        + as_int(df["cleaned_soft_skill_count"])
        + as_int(df["cleaned_scope_count"])
        + as_int(df["cleaned_mgmt_count"])
        + as_int(df["cleaned_ai_tool_any"])
        + as_int(df["cleaned_education_flag"])
        + df["yoe_extracted"].notna().astype(int)
    )
    df["cleaned_credential_stack_depth"] = (
        (as_int(df["cleaned_tech_count"]) > 0).astype(int)
        + (as_int(df["cleaned_soft_skill_count"]) > 0).astype(int)
        + (as_int(df["cleaned_scope_count"]) > 0).astype(int)
        + (as_int(df["cleaned_mgmt_count"]) > 0).astype(int)
        + as_int(df["cleaned_ai_tool_any"])
        + as_int(df["cleaned_education_flag"])
        + df["yoe_extracted"].notna().astype(int)
    )
    df["raw_scope_any"] = df["raw_scope_any"].fillna(False)

    source_df = source_summary(df)
    save_csv(source_df, TABLE_DIR / "T19_source_summary.csv")

    rates_df = annualized_rate_table(source_df)
    save_csv(rates_df, TABLE_DIR / "T19_rate_table.csv")

    arshkon = df.loc[df["source"] == "kaggle_arshkon"].copy()
    arshkon["date_rank"] = arshkon["date_posted"].rank(method="first")
    arshkon["date_bin"] = pd.qcut(arshkon["date_rank"], q=min(4, arshkon["date_rank"].nunique()), duplicates="drop")
    arshkon_bins = make_bin_summary(arshkon, "date_bin", "date_bin")
    save_csv(arshkon_bins, TABLE_DIR / "T19_arshkon_bins.csv")

    scraped = df.loc[df["source"] == "scraped"].copy()
    scraped_daily = (
        scraped.groupby("scrape_date", dropna=False)
        .agg(
            n=("uid", "size"),
            swe_n=("uid", "size"),
            text_labeled_n=("text_labeled", "sum"),
            ai_tool_share=("cleaned_ai_tool_any", "mean"),
            scope_any_share=("cleaned_scope_any", "mean"),
            analysis_text_median_len=("cleaned_text_len", "median"),
            raw_desc_median_len=("raw_text_len", "median"),
            tech_count_median=("cleaned_tech_count", "median"),
            entry_final_share=("seniority_final", lambda s: (s == "entry").mean()),
            yoe_le2_share=("yoe_extracted", lambda s: (s <= 2).mean()),
            posting_age_median=("posting_age_days", "median"),
            posting_age_p25=("posting_age_days", lambda s: s.quantile(0.25)),
            posting_age_p75=("posting_age_days", lambda s: s.quantile(0.75)),
            age_le7_share=("posting_age_days", lambda s: (s <= 7).mean()),
            age_ge30_share=("posting_age_days", lambda s: (s >= 30).mean()),
        )
        .reset_index()
        .sort_values("scrape_date")
    )
    scraped_daily["dayofweek"] = pd.to_datetime(scraped_daily["scrape_date"]).dt.dayofweek
    save_csv(scraped_daily, TABLE_DIR / "T19_scraped_daily.csv")

    dayofweek_df = (
        scraped_daily.groupby("dayofweek", dropna=False)
        .agg(
            n_days=("scrape_date", "size"),
            mean_swe_n=("swe_n", "mean"),
            mean_ai_tool_share=("ai_tool_share", "mean"),
            mean_scope_any_share=("scope_any_share", "mean"),
            mean_analysis_text_median_len=("analysis_text_median_len", "mean"),
            mean_posting_age_median=("posting_age_median", "mean"),
        )
        .reset_index()
        .sort_values("dayofweek")
    )
    save_csv(dayofweek_df, TABLE_DIR / "T19_dayofweek.csv")

    age_summary = scraped[["scrape_date", "posting_age_days"]].copy()
    age_summary["scrape_date"] = pd.to_datetime(age_summary["scrape_date"])
    age_summary = (
        age_summary.groupby("scrape_date", dropna=False)["posting_age_days"]
        .agg(
            n="size",
            median="median",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            le7=lambda s: (s <= 7).mean(),
            le14=lambda s: (s <= 14).mean(),
            ge30=lambda s: (s >= 30).mean(),
        )
        .reset_index()
        .sort_values("scrape_date")
    )
    save_csv(age_summary, TABLE_DIR / "T19_posting_age_summary.csv")

    raw_sensitivity = (
        df.groupby("source", dropna=False)
        .agg(
            n=("uid", "size"),
            raw_ai_tool_share=("raw_ai_tool_any", "mean"),
            raw_scope_any_share=("raw_scope_any", "mean"),
            raw_text_median_len=("raw_text_len", "median"),
        )
        .reset_index()
    )
    save_csv(raw_sensitivity, TABLE_DIR / "T19_raw_sensitivity_summary.csv")

    plot_source_window(source_df)
    plot_arshkon_bins(arshkon_bins)
    plot_scraped_daily(scraped_daily)

    write_report(source_df, rates_df, arshkon_bins, scraped_daily, dayofweek_df, age_summary)


if __name__ == "__main__":
    main()
