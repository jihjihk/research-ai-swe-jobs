"""T08 — Distribution profiling & anomaly detection.

Computes baseline distributions for every meaningful numeric and categorical
variable by period and seniority, produces figures/tables, and ranks metrics
by 2024→2026 effect size (calibrated against within-2024 SNR).

Run:
    ./.venv/bin/python exploration/scripts/T08_distribution_profiling.py

All queries use the default filter:
    is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok'
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = str(ROOT / "data" / "unified.parquet")
FIG_DIR = ROOT / "exploration" / "figures" / "T08"
TAB_DIR = ROOT / "exploration" / "tables" / "T08"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# Default filter string
BASE = "is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok'"

# Calibration table for SNR lookups
CALIB = pd.read_csv(ROOT / "exploration" / "artifacts" / "shared" / "calibration_table.csv")
SENIORITY_PANEL = pd.read_csv(
    ROOT / "exploration" / "artifacts" / "shared" / "seniority_definition_panel.csv"
)
ENTRY_SPECIALIST = pd.read_csv(
    ROOT / "exploration" / "artifacts" / "shared" / "entry_specialist_employers.csv"
)


def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute(f"CREATE VIEW u AS SELECT * FROM '{UNIFIED}' WHERE {BASE}")
    return c


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TAB_DIR / f"{name}.csv", index=False)


def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%" if pd.notna(x) else "—"


# ---------------------------------------------------------------------------
# 1. Period × source counts and LLM coverage
# ---------------------------------------------------------------------------
def universe_counts() -> pd.DataFrame:
    c = con()
    df = c.execute(
        """
        SELECT source, period,
               COUNT(*) AS n,
               SUM(CASE WHEN llm_classification_coverage='labeled' THEN 1 ELSE 0 END) AS labeled,
               SUM(CASE WHEN llm_extraction_coverage='labeled' THEN 1 ELSE 0 END) AS extracted,
               SUM(CASE WHEN is_aggregator THEN 1 ELSE 0 END) AS aggregators
        FROM u GROUP BY 1,2 ORDER BY 1,2
        """
    ).df()
    df["labeled_pct"] = df["labeled"] / df["n"]
    df["aggregator_pct"] = df["aggregators"] / df["n"]
    save_table(df, "universe_counts")
    return df


# ---------------------------------------------------------------------------
# 2. Description length distributions
# ---------------------------------------------------------------------------
def length_profiles() -> dict:
    c = con()
    # Raw length by period × source
    stats = c.execute(
        """
        SELECT source, period,
               COUNT(*) AS n,
               AVG(description_length)::DOUBLE AS mean,
               STDDEV_SAMP(description_length)::DOUBLE AS sd,
               QUANTILE_CONT(description_length, 0.1)::DOUBLE AS p10,
               QUANTILE_CONT(description_length, 0.5)::DOUBLE AS median,
               QUANTILE_CONT(description_length, 0.9)::DOUBLE AS p90,
               QUANTILE_CONT(description_length, 0.99)::DOUBLE AS p99,
               MAX(description_length)::DOUBLE AS max
        FROM u GROUP BY 1,2 ORDER BY 1,2
        """
    ).df()
    save_table(stats, "desc_length_by_source_period")

    # By seniority_final × period (collapsed pooled-2024 / scraped)
    sen_stats = c.execute(
        """
        WITH tagged AS (
          SELECT CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period_coarse,
                 seniority_final,
                 description_length
          FROM u
        )
        SELECT period_coarse, seniority_final,
               COUNT(*) AS n,
               AVG(description_length)::DOUBLE AS mean,
               QUANTILE_CONT(description_length, 0.5)::DOUBLE AS median,
               STDDEV_SAMP(description_length)::DOUBLE AS sd
        FROM tagged
        GROUP BY 1,2
        ORDER BY 1,2
        """
    ).df()
    save_table(sen_stats, "desc_length_by_period_seniority")

    # Length percentiles of 2024 pooled vs 2026 for figure
    samples = c.execute(
        """
        SELECT CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period_coarse,
               description_length
        FROM u
        """
    ).df()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, xmax, title in zip(axes, [15000, 3000], ["Full range (0–15,000)", "Zoom 0–3,000"]):
        for label, sub in samples.groupby("period_coarse"):
            vals = sub["description_length"].clip(upper=xmax)
            ax.hist(vals, bins=50, alpha=0.55, label=f"{label} (n={len(sub):,})", density=True)
        ax.set_title(title)
        ax.set_xlabel("description_length (chars)")
        ax.set_ylabel("density")
        ax.legend()
    fig.suptitle("Description length by pooled period")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_desc_length.png", dpi=130)
    plt.close(fig)

    # Anomaly flag: bi-/tri-modality check. Compute simple decile table.
    deciles = samples.groupby("period_coarse")["description_length"].quantile(np.arange(0.1, 1.0, 0.1))
    deciles = deciles.unstack(level=0).reset_index().rename(columns={"level_1": "decile"})
    save_table(deciles, "desc_length_deciles")

    return {"stats": stats, "sen_stats": sen_stats}


# ---------------------------------------------------------------------------
# 3. YOE distribution
# ---------------------------------------------------------------------------
def yoe_profiles() -> dict:
    c = con()
    yoe_by_src = c.execute(
        """
        SELECT source, period,
               COUNT(*) AS n_total,
               COUNT(yoe_min_years_llm) AS n_labeled,
               AVG(yoe_min_years_llm)::DOUBLE AS mean_llm,
               QUANTILE_CONT(yoe_min_years_llm, 0.5)::DOUBLE AS median_llm,
               STDDEV_SAMP(yoe_min_years_llm)::DOUBLE AS sd_llm,
               SUM(CASE WHEN yoe_min_years_llm <= 2 THEN 1 ELSE 0 END) AS n_yoe_le2,
               SUM(CASE WHEN yoe_min_years_llm >= 5 THEN 1 ELSE 0 END) AS n_yoe_ge5
        FROM u
        GROUP BY 1,2 ORDER BY 1,2
        """
    ).df()
    yoe_by_src["j3_share_labeled"] = yoe_by_src["n_yoe_le2"] / yoe_by_src["n_labeled"]
    yoe_by_src["s4_share_labeled"] = yoe_by_src["n_yoe_ge5"] / yoe_by_src["n_labeled"]
    save_table(yoe_by_src, "yoe_llm_by_source_period")

    # YOE histogram
    yoe = c.execute(
        """
        SELECT CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period_coarse,
               source,
               yoe_min_years_llm
        FROM u WHERE yoe_min_years_llm IS NOT NULL
        """
    ).df()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # histograms capped at 15y
    bins = np.arange(0, 16, 1)
    for ax, restrict in zip(axes, [("kaggle_arshkon", "2024 arshkon"), ("kaggle_asaniczka", "2024 asaniczka")]):
        sub_src, sub_label = restrict
        sub = yoe[yoe["source"] == sub_src]["yoe_min_years_llm"].clip(upper=15)
        ax.hist(sub, bins=bins, alpha=0.6, label=sub_label + f" (n={len(sub):,})", color="steelblue")
        sub2 = yoe[yoe["period_coarse"] == "2026"]["yoe_min_years_llm"].clip(upper=15)
        ax.hist(sub2, bins=bins, alpha=0.4, label=f"2026 scraped (n={len(sub2):,})", color="darkorange")
        ax.set_xlabel("yoe_min_years_llm (capped 15)")
        ax.set_ylabel("count")
        ax.legend()
        ax.set_title(sub_label + " vs 2026")
    fig.suptitle("yoe_min_years_llm distribution (labeled rows)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_yoe_llm.png", dpi=130)
    plt.close(fig)
    return {"by_source": yoe_by_src}


# ---------------------------------------------------------------------------
# 4. seniority_final and seniority_3level distributions
# ---------------------------------------------------------------------------
def seniority_profiles() -> dict:
    c = con()
    final = c.execute(
        """
        SELECT source, period, seniority_final, COUNT(*) AS n
        FROM u
        GROUP BY 1,2,3 ORDER BY 1,2,3
        """
    ).df()
    totals = final.groupby(["source", "period"])["n"].sum().rename("total").reset_index()
    final = final.merge(totals, on=["source", "period"])
    final["share"] = final["n"] / final["total"]
    save_table(final, "seniority_final_by_source")

    native = c.execute(
        """
        SELECT source, period, seniority_native, COUNT(*) AS n
        FROM u
        GROUP BY 1,2,3 ORDER BY 1,2,3
        """
    ).df()
    totals = native.groupby(["source", "period"])["n"].sum().rename("total").reset_index()
    native = native.merge(totals, on=["source", "period"])
    native["share"] = native["n"] / native["total"]
    save_table(native, "seniority_native_by_source")

    sen_src = c.execute(
        """
        SELECT source, period, seniority_final_source, COUNT(*) AS n
        FROM u
        GROUP BY 1,2,3 ORDER BY 1,2,3
        """
    ).df()
    totals = sen_src.groupby(["source", "period"])["n"].sum().rename("total").reset_index()
    sen_src = sen_src.merge(totals, on=["source", "period"])
    sen_src["share"] = sen_src["n"] / sen_src["total"]
    save_table(sen_src, "seniority_final_source_by_source")

    # Figure: seniority_final shares pooled 2024 vs 2026
    pooled = c.execute(
        """
        WITH tagged AS (
          SELECT CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period_coarse,
                 seniority_final
          FROM u
        )
        SELECT period_coarse, seniority_final, COUNT(*) AS n
        FROM tagged GROUP BY 1,2 ORDER BY 1,2
        """
    ).df()
    totals = pooled.groupby("period_coarse")["n"].sum().rename("total").reset_index()
    pooled = pooled.merge(totals, on="period_coarse")
    pooled["share"] = pooled["n"] / pooled["total"]
    save_table(pooled, "seniority_final_pooled")

    order = ["entry", "associate", "mid-senior", "director", "unknown"]
    pooled["seniority_final"] = pd.Categorical(pooled["seniority_final"], order, ordered=True)
    pooled = pooled.sort_values(["period_coarse", "seniority_final"])
    pivot = pooled.pivot_table(index="seniority_final", columns="period_coarse", values="share", observed=True)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    pivot.plot.bar(ax=ax)
    ax.set_ylabel("Share of SWE postings")
    ax.set_title("seniority_final distribution — pooled 2024 vs 2026")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_seniority_final.png", dpi=130)
    plt.close(fig)

    return {"final_by_source": final, "pooled": pooled}


# ---------------------------------------------------------------------------
# 5. Arshkon-native entry diagnostic (YOE profile of seniority_native='entry')
# ---------------------------------------------------------------------------
def arshkon_native_entry_diagnostic() -> pd.DataFrame:
    c = con()
    df = c.execute(
        """
        SELECT seniority_native,
               COUNT(*) AS n,
               COUNT(yoe_min_years_llm) AS n_with_yoe,
               AVG(yoe_min_years_llm)::DOUBLE AS mean_yoe,
               QUANTILE_CONT(yoe_min_years_llm, 0.5)::DOUBLE AS median_yoe,
               SUM(CASE WHEN yoe_min_years_llm <= 2 THEN 1 ELSE 0 END) AS n_yoe_le2,
               SUM(CASE WHEN yoe_min_years_llm >= 5 THEN 1 ELSE 0 END) AS n_yoe_ge5
        FROM u
        WHERE source='kaggle_arshkon'
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    df["share_yoe_le2"] = df["n_yoe_le2"] / df["n_with_yoe"]
    df["share_yoe_ge5"] = df["n_yoe_ge5"] / df["n_with_yoe"]
    save_table(df, "arshkon_native_yoe_profile")
    return df


# ---------------------------------------------------------------------------
# 6. T30 panel — junior and senior 4-row tables (J1/J2/J3/J4 and S1/S2/S3/S4)
# ---------------------------------------------------------------------------
def t30_panel_table() -> pd.DataFrame:
    rows = []
    for defn in ["J1", "J2", "J3", "J4"]:
        pooled_2024 = SENIORITY_PANEL.query("definition==@defn and period=='pooled-2024'").iloc[0]
        pooled_2026 = SENIORITY_PANEL.query("definition==@defn and period=='pooled-2026'").iloc[0]
        arshkon = SENIORITY_PANEL.query("definition==@defn and source=='kaggle_arshkon'").iloc[0]
        asaniczka = SENIORITY_PANEL.query("definition==@defn and source=='kaggle_asaniczka'").iloc[0]
        rows.append({
            "side": "junior",
            "definition": defn,
            "n_pooled2024_denom": pooled_2024["n_of_denominator"],
            "n_pooled2026_denom": pooled_2026["n_of_denominator"],
            "share_2024_pooled_denom": pooled_2024["share_of_denominator"],
            "share_2026_pooled_denom": pooled_2026["share_of_denominator"],
            "share_arshkon_denom": arshkon["share_of_denominator"],
            "share_asaniczka_denom": asaniczka["share_of_denominator"],
            "cross_period_effect": pooled_2026["share_of_denominator"] - pooled_2024["share_of_denominator"],
            "arshkon_only_effect": pooled_2026["share_of_denominator"] - arshkon["share_of_denominator"],
            "within_2024_effect": asaniczka["share_of_denominator"] - arshkon["share_of_denominator"],
            "direction": pooled_2026["direction"],
        })
    for defn in ["S1", "S2", "S3", "S4"]:
        pooled_2024 = SENIORITY_PANEL.query("definition==@defn and period=='pooled-2024'").iloc[0]
        pooled_2026 = SENIORITY_PANEL.query("definition==@defn and period=='pooled-2026'").iloc[0]
        arshkon = SENIORITY_PANEL.query("definition==@defn and source=='kaggle_arshkon'").iloc[0]
        asaniczka = SENIORITY_PANEL.query("definition==@defn and source=='kaggle_asaniczka'").iloc[0]
        rows.append({
            "side": "senior",
            "definition": defn,
            "n_pooled2024_denom": pooled_2024["n_of_denominator"],
            "n_pooled2026_denom": pooled_2026["n_of_denominator"],
            "share_2024_pooled_denom": pooled_2024["share_of_denominator"],
            "share_2026_pooled_denom": pooled_2026["share_of_denominator"],
            "share_arshkon_denom": arshkon["share_of_denominator"],
            "share_asaniczka_denom": asaniczka["share_of_denominator"],
            "cross_period_effect": pooled_2026["share_of_denominator"] - pooled_2024["share_of_denominator"],
            "arshkon_only_effect": pooled_2026["share_of_denominator"] - arshkon["share_of_denominator"],
            "within_2024_effect": asaniczka["share_of_denominator"] - arshkon["share_of_denominator"],
            "direction": pooled_2026["direction"],
        })
    df = pd.DataFrame(rows)
    save_table(df, "t30_panel_juniorsenior")
    return df


# ---------------------------------------------------------------------------
# 7. Aggregator, metro, industry, state distributions
# ---------------------------------------------------------------------------
def categorical_profiles() -> None:
    c = con()
    # metro_area top 15 by 2026
    metro_top = c.execute(
        """
        WITH t AS (
          SELECT CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period_coarse,
                 metro_area
          FROM u WHERE metro_area IS NOT NULL
        )
        SELECT metro_area,
               SUM(CASE WHEN period_coarse='2024' THEN 1 ELSE 0 END) AS n_2024,
               SUM(CASE WHEN period_coarse='2026' THEN 1 ELSE 0 END) AS n_2026
        FROM t GROUP BY 1 ORDER BY n_2026 DESC LIMIT 20
        """
    ).df()
    totals = c.execute(
        """
        SELECT SUM(CASE WHEN source='scraped' THEN 0 ELSE 1 END) AS total_2024,
               SUM(CASE WHEN source='scraped' THEN 1 ELSE 0 END) AS total_2026
        FROM u WHERE metro_area IS NOT NULL
        """
    ).df().iloc[0]
    metro_top["share_2024"] = metro_top["n_2024"] / totals["total_2024"]
    metro_top["share_2026"] = metro_top["n_2026"] / totals["total_2026"]
    metro_top["delta_pp"] = (metro_top["share_2026"] - metro_top["share_2024"]) * 100
    save_table(metro_top, "top20_metro_area")

    # state top 15
    state_top = c.execute(
        """
        WITH t AS (
          SELECT CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period_coarse,
                 state_normalized
          FROM u WHERE state_normalized IS NOT NULL
        )
        SELECT state_normalized,
               SUM(CASE WHEN period_coarse='2024' THEN 1 ELSE 0 END) AS n_2024,
               SUM(CASE WHEN period_coarse='2026' THEN 1 ELSE 0 END) AS n_2026
        FROM t GROUP BY 1 ORDER BY n_2026 DESC LIMIT 20
        """
    ).df()
    totals_s = c.execute(
        """
        SELECT SUM(CASE WHEN source='scraped' THEN 0 ELSE 1 END) AS total_2024,
               SUM(CASE WHEN source='scraped' THEN 1 ELSE 0 END) AS total_2026
        FROM u WHERE state_normalized IS NOT NULL
        """
    ).df().iloc[0]
    state_top["share_2024"] = state_top["n_2024"] / totals_s["total_2024"]
    state_top["share_2026"] = state_top["n_2026"] / totals_s["total_2026"]
    state_top["delta_pp"] = (state_top["share_2026"] - state_top["share_2024"]) * 100
    save_table(state_top, "top20_state")

    # industry top 15 by each period (labels drift — so report each period separately)
    ind_top = c.execute(
        """
        WITH t AS (
          SELECT CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period_coarse,
                 company_industry
          FROM u WHERE company_industry IS NOT NULL
        )
        SELECT period_coarse, company_industry, COUNT(*) AS n
        FROM t
        GROUP BY 1,2
        """
    ).df()
    # top 15 per period
    out = []
    for pc, g in ind_top.groupby("period_coarse"):
        g = g.sort_values("n", ascending=False).head(15)
        g["share"] = g["n"] / g["n"].sum()
        out.append(g.assign(period_coarse=pc))
    save_table(pd.concat(out).reset_index(drop=True), "top15_industry_by_period")

    # aggregator share by source
    agg = c.execute(
        """
        SELECT source, period,
               COUNT(*) AS n,
               SUM(CASE WHEN is_aggregator THEN 1 ELSE 0 END) AS n_agg
        FROM u GROUP BY 1,2 ORDER BY 1,2
        """
    ).df()
    agg["share"] = agg["n_agg"] / agg["n"]
    save_table(agg, "aggregator_by_source")


# ---------------------------------------------------------------------------
# 8. Ghost risk, remote, multi-location
# ---------------------------------------------------------------------------
def quality_profiles() -> None:
    c = con()
    ghost_rule = c.execute(
        """
        SELECT source, period, ghost_job_risk, COUNT(*) AS n
        FROM u GROUP BY 1,2,3 ORDER BY 1,2,3
        """
    ).df()
    totals = ghost_rule.groupby(["source", "period"])["n"].sum().rename("total").reset_index()
    ghost_rule = ghost_rule.merge(totals, on=["source", "period"])
    ghost_rule["share"] = ghost_rule["n"] / ghost_rule["total"]
    save_table(ghost_rule, "ghost_job_risk_by_source")

    ghost_llm = c.execute(
        """
        SELECT source, period, ghost_assessment_llm, COUNT(*) AS n
        FROM u
        WHERE llm_classification_coverage='labeled'
        GROUP BY 1,2,3 ORDER BY 1,2,3
        """
    ).df()
    totals = ghost_llm.groupby(["source", "period"])["n"].sum().rename("total").reset_index()
    ghost_llm = ghost_llm.merge(totals, on=["source", "period"])
    ghost_llm["share"] = ghost_llm["n"] / ghost_llm["total"]
    save_table(ghost_llm, "ghost_assessment_llm_by_source")

    remote = c.execute(
        """
        SELECT source, period,
               COUNT(*) AS n,
               SUM(CASE WHEN is_remote_inferred THEN 1 ELSE 0 END) AS n_remote,
               SUM(CASE WHEN is_multi_location THEN 1 ELSE 0 END) AS n_multi
        FROM u GROUP BY 1,2 ORDER BY 1,2
        """
    ).df()
    remote["share_remote"] = remote["n_remote"] / remote["n"]
    remote["share_multi"] = remote["n_multi"] / remote["n"]
    save_table(remote, "remote_multi_location_by_source")


# ---------------------------------------------------------------------------
# 9. SWE classification tier distribution
# ---------------------------------------------------------------------------
def swe_tier_distribution() -> None:
    c = con()
    df = c.execute(
        """
        SELECT source, period, swe_classification_tier, COUNT(*) AS n
        FROM u GROUP BY 1,2,3 ORDER BY 1,2,3
        """
    ).df()
    totals = df.groupby(["source", "period"])["n"].sum().rename("total").reset_index()
    df = df.merge(totals, on=["source", "period"])
    df["share"] = df["n"] / df["total"]
    save_table(df, "swe_tier_by_source")


# ---------------------------------------------------------------------------
# 10. company_size stratification within arshkon
# ---------------------------------------------------------------------------
def company_size_strat() -> pd.DataFrame:
    c = con()
    # Quartiles of company_size on arshkon, then J3 share, AI strict share, tech count
    df = c.execute(
        """
        WITH ark AS (
          SELECT uid, description, description_length, company_size, yoe_min_years_llm,
                 CASE WHEN yoe_min_years_llm <= 2 THEN 1 ELSE 0 END AS j3_flag
          FROM u WHERE source='kaggle_arshkon' AND company_size IS NOT NULL
        ),
        binned AS (
          SELECT *, NTILE(4) OVER (ORDER BY company_size) AS size_q
          FROM ark
        )
        SELECT size_q,
               COUNT(*) AS n,
               AVG(company_size)::DOUBLE AS mean_size,
               MIN(company_size)::DOUBLE AS min_size,
               MAX(company_size)::DOUBLE AS max_size,
               COUNT(yoe_min_years_llm) AS n_with_yoe,
               SUM(CASE WHEN j3_flag=1 THEN 1 ELSE 0 END) AS n_j3,
               SUM(CASE WHEN regexp_matches(lower(description),
                                            '(copilot|cursor|claude|gpt-4|gpt4|codex|devin|windsurf|anthropic|chatgpt|openai|llm|rag\\b|prompt engineering|mcp\\b|langchain|llamaindex)') THEN 1 ELSE 0 END) AS n_ai,
               AVG(description_length)::DOUBLE AS mean_desc
        FROM binned
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    df["j3_share_labeled"] = df["n_j3"] / df["n_with_yoe"]
    df["ai_strict_share"] = df["n_ai"] / df["n"]
    save_table(df, "arshkon_company_size_quartiles")
    return df


# ---------------------------------------------------------------------------
# 11. Text-based metric computation with labeled filter
# ---------------------------------------------------------------------------
def text_rates(include_raw: bool = True) -> pd.DataFrame:
    """Compute binary keyword prevalence using raw description (all rows)."""
    c = con()
    ai_strict = r"(copilot|cursor|claude|gpt-4|gpt4|codex|devin|windsurf|anthropic|chatgpt|openai|llm\b|rag\b|prompt engineering|mcp\b|langchain|llamaindex)"
    ai_broad = r"(agent|\bml\b|\bai\b|llm|artificial intelligence|mcp)"
    mgmt_strict = r"(mentor|coach|hire|headcount|performance review|direct reports)"
    mgmt_broad = r"(lead|team|stakeholder|coordinate|manage)"
    scope = r"(ownership|end-to-end|cross-functional|autonomous|initiative|stakeholder)"
    soft = r"(collaborative|communication|teamwork|problem-solving|interpersonal|leadership)"
    phd = r"(\bphd\b|doctorate|ph\\.d)"
    ms = r"(\bmaster\b|\bm\\.s\b|\bms degree\b)"
    bs = r"(\bbachelor\b|\bb\\.s\b|\bbs degree\b)"
    df = c.execute(
        f"""
        WITH t AS (
          SELECT CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period_coarse,
                 source,
                 lower(description) AS d,
                 description_length
          FROM u
        )
        SELECT period_coarse, COUNT(*) AS n,
               SUM(CASE WHEN regexp_matches(d,'{ai_strict}') THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS ai_strict_share,
               SUM(CASE WHEN regexp_matches(d,'{ai_broad}') THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS ai_broad_share,
               SUM(CASE WHEN regexp_matches(d,'{mgmt_strict}') THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS mgmt_strict_share,
               SUM(CASE WHEN regexp_matches(d,'{mgmt_broad}') THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS mgmt_broad_share,
               SUM(CASE WHEN regexp_matches(d,'{scope}') THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS scope_share,
               SUM(CASE WHEN regexp_matches(d,'{soft}') THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS soft_share,
               SUM(CASE WHEN regexp_matches(d,'{phd}') THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS phd_share,
               SUM(CASE WHEN regexp_matches(d,'{ms}') THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS ms_share,
               SUM(CASE WHEN regexp_matches(d,'{bs}') THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS bs_share
        FROM t GROUP BY 1 ORDER BY 1
        """
    ).df()
    save_table(df, "text_binary_rates_pooled")

    # Per 1K char rates — count regex matches per doc via regexp_extract_all → len
    df_rate = c.execute(
        f"""
        WITH t AS (
          SELECT CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period_coarse,
                 lower(description) AS d,
                 description_length
          FROM u
        ), counts AS (
          SELECT period_coarse,
                 LENGTH(regexp_extract_all(d, '{ai_strict}'))::DOUBLE AS ai_strict_ct,
                 LENGTH(regexp_extract_all(d, '{mgmt_strict}'))::DOUBLE AS mgmt_strict_ct,
                 LENGTH(regexp_extract_all(d, '{scope}'))::DOUBLE AS scope_ct,
                 description_length
          FROM t
        )
        SELECT period_coarse,
               AVG(ai_strict_ct / (description_length / 1000.0))::DOUBLE AS ai_strict_per_1k,
               AVG(mgmt_strict_ct / (description_length / 1000.0))::DOUBLE AS mgmt_strict_per_1k,
               AVG(scope_ct / (description_length / 1000.0))::DOUBLE AS scope_per_1k
        FROM counts
        WHERE description_length >= 200
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    save_table(df_rate, "text_per_1k_rates_pooled")
    return df


# ---------------------------------------------------------------------------
# 12. Aggregator / company-capping / labeled-only sensitivities on J3
# ---------------------------------------------------------------------------
def junior_sensitivities() -> pd.DataFrame:
    c = con()
    # Register entry-specialist set as a DuckDB view
    es_df = ENTRY_SPECIALIST.dropna(subset=["company_name_canonical"]).copy()
    c.register("entry_specialists_df", es_df[["company_name_canonical"]])

    # Base: pooled 2024 vs 2026, J3
    def compute(where_extra: str = "", tag: str = "base") -> pd.DataFrame:
        q = f"""
        WITH t AS (
          SELECT CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period_coarse,
                 yoe_min_years_llm
          FROM u WHERE 1=1 {where_extra}
        )
        SELECT period_coarse,
               COUNT(*) AS n_total,
               COUNT(yoe_min_years_llm) AS n_labeled,
               SUM(CASE WHEN yoe_min_years_llm<=2 THEN 1 ELSE 0 END)::DOUBLE AS n_j3,
               SUM(CASE WHEN yoe_min_years_llm>=5 THEN 1 ELSE 0 END)::DOUBLE AS n_s4
        FROM t GROUP BY 1 ORDER BY 1
        """
        return c.execute(q).df().assign(tag=tag)

    variants = [
        ("base", ""),
        ("no_aggregator", " AND NOT is_aggregator"),
        ("no_entry_specialist",
            " AND company_name_canonical NOT IN (SELECT company_name_canonical FROM entry_specialists_df)"),
        ("tier_strict",
            " AND swe_classification_tier IN ('regex','embedding_high')"),
    ]
    dfs = []
    for tag, extra in variants:
        d = compute(extra, tag)
        d["j3_share_labeled"] = d["n_j3"] / d["n_labeled"]
        d["s4_share_labeled"] = d["n_s4"] / d["n_labeled"]
        dfs.append(d)
    df = pd.concat(dfs)
    save_table(df, "junior_senior_sensitivities")

    # Company-capped sensitivity (cap at 20 per canonical)
    q = """
    WITH t AS (
      SELECT CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period_coarse,
             company_name_canonical, uid, yoe_min_years_llm,
             ROW_NUMBER() OVER (PARTITION BY CASE WHEN source='scraped' THEN '2026' ELSE '2024' END,
                                              company_name_canonical ORDER BY uid) AS rn
      FROM u
    ), capped AS (SELECT * FROM t WHERE rn <= 20)
    SELECT period_coarse, COUNT(*) AS n, COUNT(yoe_min_years_llm) AS n_lab,
           SUM(CASE WHEN yoe_min_years_llm<=2 THEN 1 ELSE 0 END)::DOUBLE AS n_j3,
           SUM(CASE WHEN yoe_min_years_llm>=5 THEN 1 ELSE 0 END)::DOUBLE AS n_s4
    FROM capped GROUP BY 1 ORDER BY 1
    """
    cap = c.execute(q).df()
    cap["j3_share_labeled"] = cap["n_j3"] / cap["n_lab"]
    cap["s4_share_labeled"] = cap["n_s4"] / cap["n_lab"]
    cap["tag"] = "cap20"
    save_table(cap, "junior_senior_cap20")

    return df


# ---------------------------------------------------------------------------
# 13. Indeed cross-platform validation using rule yoe_extracted <= 2
# ---------------------------------------------------------------------------
def indeed_cross_platform() -> pd.DataFrame:
    c = con()
    # Switch to unified.parquet without default filter for Indeed
    df = c.execute(
        f"""
        SELECT source_platform,
               CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period_coarse,
               COUNT(*) AS n,
               COUNT(yoe_extracted) AS n_with_rule_yoe,
               SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END)::DOUBLE AS n_rule_j3,
               SUM(CASE WHEN yoe_extracted >= 5 THEN 1 ELSE 0 END)::DOUBLE AS n_rule_s4
        FROM '{UNIFIED}'
        WHERE is_swe AND is_english AND date_flag='ok'
        GROUP BY 1,2 ORDER BY 1,2
        """
    ).df()
    df["rule_j3_labeled_share"] = df["n_rule_j3"] / df["n_with_rule_yoe"]
    df["rule_s4_labeled_share"] = df["n_rule_s4"] / df["n_with_rule_yoe"]
    save_table(df, "indeed_cross_platform_rule_yoe")
    return df


# ---------------------------------------------------------------------------
# 14. Within-company vs between-company decomposition for J3 using returning co cohort
# ---------------------------------------------------------------------------
def within_between_j3() -> dict:
    c = con()
    ret = pd.read_csv(ROOT / "exploration" / "artifacts" / "shared" / "returning_companies_cohort.csv")

    # Overlap panel subset: use returning cohort
    # Aggregate: share of returning-cohort postings with J3 (yoe_llm<=2 labeled) by period
    q = f"""
    WITH ret AS (SELECT company_name_canonical FROM read_csv('{ROOT / "exploration/artifacts/shared/returning_companies_cohort.csv"}')),
         scope AS (
           SELECT u.*, CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS pc
           FROM u JOIN ret ON u.company_name_canonical = ret.company_name_canonical
         )
    SELECT pc,
           COUNT(*) AS n_total,
           COUNT(yoe_min_years_llm) AS n_labeled,
           SUM(CASE WHEN yoe_min_years_llm<=2 THEN 1 ELSE 0 END)::DOUBLE AS n_j3,
           SUM(CASE WHEN yoe_min_years_llm>=5 THEN 1 ELSE 0 END)::DOUBLE AS n_s4
    FROM scope GROUP BY 1 ORDER BY 1
    """
    agg = c.execute(q).df()
    agg["j3_share_labeled"] = agg["n_j3"] / agg["n_labeled"]
    agg["s4_share_labeled"] = agg["n_s4"] / agg["n_labeled"]
    save_table(agg, "returning_cohort_j3_s4")

    # Within-co (per returning co, Δ J3 share across periods), then Aggregate mean
    q = f"""
    WITH ret AS (SELECT company_name_canonical FROM read_csv('{ROOT / "exploration/artifacts/shared/returning_companies_cohort.csv"}')),
         scope AS (
           SELECT u.*, CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS pc
           FROM u JOIN ret ON u.company_name_canonical = ret.company_name_canonical
         ),
         perco AS (
           SELECT company_name_canonical, pc,
                  COUNT(yoe_min_years_llm) AS n_lab,
                  SUM(CASE WHEN yoe_min_years_llm<=2 THEN 1 ELSE 0 END)::DOUBLE AS n_j3
           FROM scope GROUP BY 1,2
         )
    SELECT pc,
           COUNT(*) AS n_co,
           AVG(CASE WHEN n_lab>=3 THEN n_j3/n_lab END)::DOUBLE AS mean_j3_within_min3lab,
           AVG(CASE WHEN n_lab>=1 THEN n_j3/n_lab END)::DOUBLE AS mean_j3_within_min1lab
    FROM perco GROUP BY 1 ORDER BY 1
    """
    within = c.execute(q).df()
    save_table(within, "returning_cohort_j3_within")
    return {"agg": agg, "within": within}


# ---------------------------------------------------------------------------
# 15. Effect-size ranking of metrics
# ---------------------------------------------------------------------------
def rank_metrics_by_effect() -> pd.DataFrame:
    """Use calibration_table.csv values + T30 extension, produce ranked table."""
    df = CALIB.copy()
    df["abs_cross"] = df["cross_period_effect"].abs()
    df["abs_within"] = df["within_2024_effect"].abs()
    df["snr"] = df["calibration_ratio"]
    df = df.sort_values(["abs_cross"], ascending=False)
    save_table(df[["metric", "metric_type", "arshkon_value", "asaniczka_value", "scraped_value",
                   "within_2024_effect", "cross_period_effect", "calibration_ratio", "snr_flag"]],
               "ranked_metrics")
    return df


# ---------------------------------------------------------------------------
# 16. Big-change composite figure (fig 4)
# ---------------------------------------------------------------------------
def composite_change_figure() -> None:
    df = CALIB[CALIB["metric_type"] == "proportion"].copy()
    df["cross_pp"] = df["cross_period_effect"] * 100
    df["within_pp"] = df["within_2024_effect"] * 100
    df = df.sort_values("cross_pp", key=lambda x: x.abs(), ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(df))
    ax.barh(y - 0.2, df["within_pp"].abs(), height=0.4, color="lightgray", label="|within-2024| (pp)")
    ax.barh(y + 0.2, df["cross_pp"].abs(), height=0.4, color="steelblue", label="|2024→2026| (pp)")
    ax.set_yticks(y)
    ax.set_yticklabels(df["metric"], fontsize=9)
    ax.set_xlabel("Absolute effect size (pp)")
    ax.set_title("Cross-period vs within-2024 absolute effect — proportion metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_effect_ranking.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Universe counts")
    print(universe_counts().to_string())

    print("\nDescription length profiles")
    length_profiles()

    print("\nYOE profiles")
    yoe_profiles()

    print("\nSeniority profiles")
    seniority_profiles()

    print("\nArshkon native entry diagnostic")
    print(arshkon_native_entry_diagnostic().to_string())

    print("\nT30 junior/senior panel")
    print(t30_panel_table().to_string())

    print("\nCategorical profiles")
    categorical_profiles()

    print("\nQuality profiles")
    quality_profiles()

    print("\nSWE tier distribution")
    swe_tier_distribution()

    print("\nCompany size stratification (arshkon)")
    print(company_size_strat().to_string())

    print("\nText binary + density rates")
    print(text_rates().to_string())

    print("\nJunior/senior sensitivities")
    print(junior_sensitivities().to_string())

    print("\nIndeed cross-platform")
    print(indeed_cross_platform().to_string())

    print("\nWithin-company vs between-company")
    print(within_between_j3())

    print("\nRanked metrics")
    print(rank_metrics_by_effect().to_string())

    print("\nComposite change figure")
    composite_change_figure()


if __name__ == "__main__":
    main()
