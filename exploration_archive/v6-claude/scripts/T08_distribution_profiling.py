"""
T08 — Distribution profiling & anomaly detection (Wave 2 Agent E)

Computes baseline distributions for SWE LinkedIn variables by period x seniority
and runs the essential sensitivity framework (a, b, c, e, f, g).

Primary inputs:
- data/unified.parquet (via DuckDB)
- exploration/artifacts/shared/swe_cleaned_text.parquet
- exploration/artifacts/shared/swe_tech_matrix.parquet
- exploration/artifacts/shared/calibration_table.csv

Outputs:
- exploration/figures/T08/*.png
- exploration/tables/T08/*.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/jihgaboot/gabor/job-research")
DATA = REPO / "data" / "unified.parquet"
SHARED = REPO / "exploration" / "artifacts" / "shared"
FIGS = REPO / "exploration" / "figures" / "T08"
TABS = REPO / "exploration" / "tables" / "T08"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

DEFAULT_FILTER = (
    "is_swe = true AND source_platform = 'linkedin' AND is_english = true "
    "AND date_flag = 'ok'"
)

TOP3_ENTRY_COMPANIES = ("walmart", "tiktok", "bytedance")


def _con():
    c = duckdb.connect()
    c.execute("PRAGMA threads=4")
    return c


def period_bucket_sql():
    # bucket rows into 2024 vs 2026 based on source
    return (
        "CASE WHEN source IN ('kaggle_arshkon','kaggle_asaniczka') THEN '2024' "
        "WHEN source = 'scraped' THEN '2026' END AS period_bucket"
    )


def load_core_frame() -> pd.DataFrame:
    """Load the SWE LinkedIn core frame with distribution-relevant columns."""
    con = _con()
    q = f"""
    SELECT
        uid,
        source,
        {period_bucket_sql()},
        seniority_final,
        seniority_final_source,
        seniority_3level,
        seniority_native,
        yoe_extracted,
        description_length,
        metro_area,
        company_industry,
        company_name_canonical,
        company_size,
        is_aggregator,
        swe_classification_tier,
        llm_extraction_coverage,
        llm_classification_coverage
    FROM '{DATA}'
    WHERE {DEFAULT_FILTER}
    """
    df = con.execute(q).df()
    con.close()
    return df


def load_tech_matrix() -> pd.DataFrame:
    return pq.read_table(str(SHARED / "swe_tech_matrix.parquet")).to_pandas()


def load_cleaned_text() -> pd.DataFrame:
    return pq.read_table(str(SHARED / "swe_cleaned_text.parquet")).to_pandas()


# ---------------------------------------------------------------------------
# STEP 1 — univariate profiling + figures
# ---------------------------------------------------------------------------

def _save_fig(name: str):
    plt.tight_layout()
    plt.savefig(FIGS / name, dpi=150, bbox_inches="tight")
    plt.close()


def step1_univariate(df: pd.DataFrame):
    print("[step 1] univariate profiling")

    # Period x seniority count table (seniority_final)
    cross = (
        df.groupby(["period_bucket", "seniority_final"])
        .size()
        .rename("n")
        .reset_index()
    )
    cross.to_csv(TABS / "period_by_seniority_final_counts.csv", index=False)

    # seniority_3level too
    cross3 = (
        df.groupby(["period_bucket", "seniority_3level"])
        .size()
        .rename("n")
        .reset_index()
    )
    cross3.to_csv(TABS / "period_by_seniority_3level_counts.csv", index=False)

    # --- figure: description_length histograms by period ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, period in zip(axes, ["2024", "2026"]):
        sub = df.loc[df.period_bucket == period, "description_length"].dropna()
        sub = sub[(sub > 0) & (sub < 20000)]
        ax.hist(sub, bins=60, color="#2b6cb0", alpha=0.85)
        ax.set_title(f"{period} description_length (n={len(sub):,})")
        ax.set_xlabel("chars")
        ax.set_ylabel("count")
        ax.axvline(sub.median(), color="red", lw=1, ls="--",
                   label=f"median={sub.median():.0f}")
        ax.legend()
    _save_fig("description_length_by_period.png")

    # --- figure: yoe_extracted histograms by period ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, period in zip(axes, ["2024", "2026"]):
        sub = df.loc[df.period_bucket == period, "yoe_extracted"].dropna()
        ax.hist(sub, bins=np.arange(0, 21) - 0.5, color="#38a169", alpha=0.85)
        ax.set_title(f"{period} yoe_extracted (n={len(sub):,})")
        ax.set_xlabel("years")
        ax.set_ylabel("count")
        ax.axvline(sub.median(), color="red", lw=1, ls="--",
                   label=f"median={sub.median():.1f}")
        ax.legend()
    _save_fig("yoe_extracted_by_period.png")

    # --- figure: seniority_final share by period ---
    cross_pct = (
        cross.assign(
            pct=cross.groupby("period_bucket")["n"].transform(lambda s: s / s.sum())
        )
    )
    piv = cross_pct.pivot(index="seniority_final", columns="period_bucket", values="pct").fillna(0)
    order = ["entry", "associate", "mid-senior", "director", "executive", "unknown"]
    piv = piv.reindex([x for x in order if x in piv.index])
    ax = piv.plot(kind="bar", figsize=(9, 4.5), color=["#3182ce", "#dd6b20"])
    ax.set_title("seniority_final share by period (of all)")
    ax.set_ylabel("share")
    ax.set_xlabel("seniority_final")
    _save_fig("seniority_final_share_by_period.png")

    # --- figure: seniority_3level share by period ---
    piv3 = (
        cross3.assign(
            pct=cross3.groupby("period_bucket")["n"].transform(lambda s: s / s.sum())
        )
        .pivot(index="seniority_3level", columns="period_bucket", values="pct")
        .fillna(0)
    )
    order3 = ["junior", "mid_senior", "manager", "unknown"]
    piv3 = piv3.reindex([x for x in order3 if x in piv3.index])
    ax = piv3.plot(kind="bar", figsize=(9, 4.5), color=["#3182ce", "#dd6b20"])
    ax.set_title("seniority_3level share by period (of all)")
    ax.set_ylabel("share")
    _save_fig("seniority_3level_share_by_period.png")

    # --- figure: is_aggregator share ---
    agg = (
        df.groupby(["period_bucket", "is_aggregator"])
        .size()
        .unstack(fill_value=0)
    )
    agg_pct = agg.div(agg.sum(axis=1), axis=0)
    ax = agg_pct.plot(kind="bar", stacked=True, figsize=(7, 4),
                      color=["#718096", "#e53e3e"])
    ax.set_title("is_aggregator share by period")
    ax.set_ylabel("share")
    _save_fig("is_aggregator_share_by_period.png")
    agg_pct.to_csv(TABS / "is_aggregator_by_period.csv")

    # --- figure: metro_area top 15 by period ---
    for period in ["2024", "2026"]:
        sub = df[df.period_bucket == period]
        top = sub["metro_area"].value_counts().dropna().head(15)
        if len(top) == 0:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        top.iloc[::-1].plot(kind="barh", ax=ax, color="#4a5568")
        ax.set_title(f"{period} top 15 metro_area (n={len(sub):,})")
        ax.set_xlabel("count")
        _save_fig(f"metro_top15_{period}.png")

    # --- figure: company_industry top 15 (arshkon + scraped) ---
    for src_label, src_filter in [
        ("arshkon_2024", df.source == "kaggle_arshkon"),
        ("scraped_2026", df.source == "scraped"),
    ]:
        sub = df[src_filter]
        top = sub["company_industry"].value_counts().dropna().head(15)
        if len(top) == 0:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        top.iloc[::-1].plot(kind="barh", ax=ax, color="#805ad5")
        ax.set_title(f"{src_label} top 15 company_industry (n={len(sub):,})")
        ax.set_xlabel("count")
        _save_fig(f"industry_top15_{src_label}.png")

    # write univariate summary table
    summary_rows = []
    for period in ["2024", "2026"]:
        sub = df[df.period_bucket == period]
        summary_rows.append({
            "period": period,
            "n": len(sub),
            "desc_len_mean": sub.description_length.mean(),
            "desc_len_median": sub.description_length.median(),
            "yoe_mean": sub.yoe_extracted.mean(),
            "yoe_median": sub.yoe_extracted.median(),
            "yoe_known_n": sub.yoe_extracted.notna().sum(),
            "seniority_known_n": (sub.seniority_final != "unknown").sum(),
            "entry_share_of_all": (sub.seniority_final == "entry").mean(),
            "entry_share_of_known": (
                (sub.seniority_final == "entry").sum()
                / max((sub.seniority_final != "unknown").sum(), 1)
            ),
            "is_aggregator_share": sub.is_aggregator.mean(),
        })
    pd.DataFrame(summary_rows).to_csv(TABS / "univariate_summary.csv", index=False)
    return cross_pct


# ---------------------------------------------------------------------------
# STEP 2 — anomaly detection
# ---------------------------------------------------------------------------

def step2_anomalies(df: pd.DataFrame):
    print("[step 2] anomaly detection")
    rows = []

    # bimodality check on description_length (use Hartigan-ish heuristic: ratio of
    # mode count to mean). We just report percentiles.
    for period in ["2024", "2026"]:
        sub = df.loc[df.period_bucket == period, "description_length"].dropna()
        qs = np.percentile(sub, [1, 5, 25, 50, 75, 95, 99])
        rows.append({
            "metric": f"description_length_{period}",
            "n": len(sub),
            "p01": qs[0], "p05": qs[1], "p25": qs[2], "p50": qs[3],
            "p75": qs[4], "p95": qs[5], "p99": qs[6],
            "mean": sub.mean(), "std": sub.std(),
            "pct_below_500": (sub < 500).mean(),
            "pct_above_10000": (sub > 10000).mean(),
        })

    # yoe anomalies
    for period in ["2024", "2026"]:
        sub = df.loc[df.period_bucket == period, "yoe_extracted"].dropna()
        qs = np.percentile(sub, [1, 5, 25, 50, 75, 95, 99])
        rows.append({
            "metric": f"yoe_extracted_{period}",
            "n": len(sub),
            "p01": qs[0], "p05": qs[1], "p25": qs[2], "p50": qs[3],
            "p75": qs[4], "p95": qs[5], "p99": qs[6],
            "mean": sub.mean(), "std": sub.std(),
            "pct_le2": (sub <= 2).mean(),
            "pct_ge10": (sub >= 10).mean(),
        })

    pd.DataFrame(rows).to_csv(TABS / "anomaly_distribution_flags.csv", index=False)

    # flag: companies with extreme entry share or extreme length
    # (per-company level)
    comp = (
        df.groupby(["period_bucket", "company_name_canonical"])
        .agg(
            n=("uid", "size"),
            entry_share=("seniority_final", lambda s: (s == "entry").mean()),
            yoe_le2_share=("yoe_extracted", lambda s: (s.dropna() <= 2).mean() if s.notna().any() else np.nan),
            mean_len=("description_length", "mean"),
        )
        .reset_index()
    )
    anom = comp[(comp.n >= 10) & ((comp.entry_share > 0.7) | (comp.yoe_le2_share > 0.8))]
    anom.sort_values("n", ascending=False).to_csv(TABS / "extreme_entry_companies.csv", index=False)

    # long descriptions outliers
    long_comp = comp[(comp.n >= 10) & (comp.mean_len > 8000)].sort_values("mean_len", ascending=False)
    long_comp.to_csv(TABS / "extreme_length_companies.csv", index=False)

    return comp


# ---------------------------------------------------------------------------
# STEP 3 — arshkon native='entry' diagnostic
# ---------------------------------------------------------------------------

def step3_native_entry_diagnostic(df: pd.DataFrame):
    print("[step 3] arshkon native='entry' YOE diagnostic")
    ark = df[df.source == "kaggle_arshkon"].copy()
    native_entry = ark[ark.seniority_native == "entry"].copy()
    yoe = native_entry.yoe_extracted.dropna()
    result = {
        "source": "kaggle_arshkon",
        "n_native_entry": len(native_entry),
        "n_with_yoe": len(yoe),
        "mean_yoe": float(yoe.mean()) if len(yoe) else np.nan,
        "median_yoe": float(yoe.median()) if len(yoe) else np.nan,
        "pct_yoe_ge5": float((yoe >= 5).mean()) if len(yoe) else np.nan,
        "pct_yoe_le2": float((yoe <= 2).mean()) if len(yoe) else np.nan,
        "pct_seniority_final_unknown": float((native_entry.seniority_final == "unknown").mean()),
    }
    # scraped comparison
    scr = df[df.source == "scraped"].copy()
    scr_ne = scr[scr.seniority_native == "entry"].copy()
    yoe_s = scr_ne.yoe_extracted.dropna()
    result_s = {
        "source": "scraped",
        "n_native_entry": len(scr_ne),
        "n_with_yoe": len(yoe_s),
        "mean_yoe": float(yoe_s.mean()) if len(yoe_s) else np.nan,
        "median_yoe": float(yoe_s.median()) if len(yoe_s) else np.nan,
        "pct_yoe_ge5": float((yoe_s >= 5).mean()) if len(yoe_s) else np.nan,
        "pct_yoe_le2": float((yoe_s <= 2).mean()) if len(yoe_s) else np.nan,
        "pct_seniority_final_unknown": float((scr_ne.seniority_final == "unknown").mean()),
    }
    pd.DataFrame([result, result_s]).to_csv(TABS / "native_entry_diagnostic.csv", index=False)
    return result, result_s


# ---------------------------------------------------------------------------
# STEP 5 — junior share trends under three operationalizations, both denominators
# ---------------------------------------------------------------------------

def step5_junior_share(df: pd.DataFrame):
    print("[step 5] junior share trends")
    rows = []

    def compute(frame, label):
        total = len(frame)
        # seniority_final
        known_sf = frame[frame.seniority_final != "unknown"]
        entry_of_all_sf = (frame.seniority_final == "entry").mean()
        entry_of_known_sf = (
            (known_sf.seniority_final == "entry").mean() if len(known_sf) else np.nan
        )

        # YOE proxy (<=2)
        yoe_known = frame[frame.yoe_extracted.notna()]
        yoe_le2_of_all = (frame.yoe_extracted <= 2).sum() / max(total, 1)
        yoe_le2_of_known = (
            (yoe_known.yoe_extracted <= 2).mean() if len(yoe_known) else np.nan
        )

        # YOE proxy <=3
        yoe_le3_of_known = (
            (yoe_known.yoe_extracted <= 3).mean() if len(yoe_known) else np.nan
        )

        rows.append({
            "slice": label,
            "n_total": total,
            "n_seniority_known": len(known_sf),
            "n_yoe_known": len(yoe_known),
            "sf_entry_of_all": entry_of_all_sf,
            "sf_entry_of_known": entry_of_known_sf,
            "yoe_le2_of_all": yoe_le2_of_all,
            "yoe_le2_of_known": yoe_le2_of_known,
            "yoe_le3_of_known": yoe_le3_of_known,
        })

    # primary: pooled-2024 vs scraped-2026
    compute(df[df.period_bucket == "2024"], "pooled_2024")
    compute(df[df.period_bucket == "2026"], "scraped_2026")

    # alt: arshkon only
    compute(df[df.source == "kaggle_arshkon"], "arshkon_only_2024")

    # exclude aggregators (sensitivity a)
    compute(df[(df.period_bucket == "2024") & (~df.is_aggregator)], "pooled_2024_noagg")
    compute(df[(df.period_bucket == "2026") & (~df.is_aggregator)], "scraped_2026_noagg")

    # exclude top-3 entry companies (sensitivity)
    top3_mask = df.company_name_canonical.fillna("").str.lower().str.contains(
        "|".join(TOP3_ENTRY_COMPANIES)
    )
    compute(df[(df.period_bucket == "2026") & (~top3_mask)], "scraped_2026_excl_top3")

    # exclude title_lookup_llm (sensitivity g)
    compute(
        df[(df.period_bucket == "2024") & (df.swe_classification_tier != "title_lookup_llm")],
        "pooled_2024_strict_swe",
    )
    compute(
        df[(df.period_bucket == "2026") & (df.swe_classification_tier != "title_lookup_llm")],
        "scraped_2026_strict_swe",
    )

    # arshkon-only vs scraped (primary source restriction — sensitivity e)
    compute(df[df.source == "kaggle_arshkon"], "arshkon_2024_primary")
    compute(df[df.source == "scraped"], "scraped_2026_primary")

    out = pd.DataFrame(rows)
    out.to_csv(TABS / "junior_share_operationalizations.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# STEP 6 — ranked change list (effect size + SNR)
# ---------------------------------------------------------------------------

def step6_ranked_changes(df: pd.DataFrame, tech: pd.DataFrame):
    print("[step 6] ranked change list (effect + SNR)")
    cal = pd.read_csv(SHARED / "calibration_table.csv")
    # rank by absolute cross_period_effect
    cal["abs_cross"] = cal["cross_period_effect"].abs()
    cal["survives_snr_ge2"] = cal["calibration_ratio"] >= 2
    ranked = cal.sort_values("abs_cross", ascending=False).reset_index(drop=True)
    ranked.to_csv(TABS / "ranked_changes_by_effect.csv", index=False)

    # also rank by calibration ratio (SNR)
    ranked_snr = cal.sort_values("calibration_ratio", ascending=False).reset_index(drop=True)
    ranked_snr.to_csv(TABS / "ranked_changes_by_snr.csv", index=False)

    # four-quadrant classification
    cal["effect_tier"] = pd.cut(cal["abs_cross"], bins=[-1, 0.05, 0.15, 1],
                                 labels=["small", "medium", "large"])
    cal["snr_tier"] = np.where(cal["calibration_ratio"] >= 2, "survives", "below")
    quad = cal.groupby(["effect_tier", "snr_tier"], observed=True).size().rename("n").reset_index()
    quad.to_csv(TABS / "effect_snr_quadrant.csv", index=False)
    return ranked


# ---------------------------------------------------------------------------
# STEP 8 — company size stratification (arshkon only)
# ---------------------------------------------------------------------------

def step8_company_size(df: pd.DataFrame, tech: pd.DataFrame):
    print("[step 8] company_size stratification (arshkon only)")
    ark = df[df.source == "kaggle_arshkon"].copy()
    # quartile by company_size (employees)
    ark = ark[ark.company_size.notna() & (ark.company_size > 0)]
    ark["size_quartile"] = pd.qcut(ark.company_size, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")

    # Join AI mention presence from tech matrix
    ai_cols = [c for c in tech.columns if c in ("llm", "copilot", "claude", "gpt", "openai", "rag", "ai_agent", "machine_learning", "deep_learning")]
    # Defensive — use any columns that exist
    ai_cols_present = [c for c in ai_cols if c in tech.columns]
    if ai_cols_present:
        tech_sub = tech[["uid"] + ai_cols_present].copy()
        tech_sub["ai_any"] = tech_sub[ai_cols_present].any(axis=1)
    else:
        tech_sub = tech[["uid"]].copy()
        tech_sub["ai_any"] = False
    # tech_count from full matrix
    non_uid = [c for c in tech.columns if c != "uid"]
    tech_count_series = tech[non_uid].sum(axis=1)
    tech_count = pd.DataFrame({"uid": tech["uid"].values, "tech_count": tech_count_series.values})

    ark = ark.merge(tech_sub[["uid", "ai_any"]], on="uid", how="left")
    ark = ark.merge(tech_count, on="uid", how="left")

    out = (
        ark.groupby("size_quartile", observed=True)
        .agg(
            n=("uid", "size"),
            entry_share_sf=("seniority_final", lambda s: (s == "entry").mean()),
            yoe_le2_share=("yoe_extracted", lambda s: (s.dropna() <= 2).mean() if s.notna().any() else np.nan),
            ai_any=("ai_any", "mean"),
            tech_count_mean=("tech_count", "mean"),
            desc_len_mean=("description_length", "mean"),
            mean_company_size=("company_size", "mean"),
        )
        .reset_index()
    )
    out.to_csv(TABS / "company_size_stratification_arshkon.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Sensitivity summary
# ---------------------------------------------------------------------------

def sensitivity_summary(df: pd.DataFrame):
    print("[sensitivity] summary table")
    cal = pd.read_csv(SHARED / "calibration_table.csv")
    # Build a small table for a handful of headline metrics under basic sensitivities
    headline_metrics = [
        "ai_keyword_prevalence_any",
        "ai_tool_specific_prevalence",
        "tech_copilot_prevalence",
        "tech_llm_prevalence",
        "tech_rag_prevalence",
        "org_scope_term_rate",
        "management_indicator_rate_broad",
        "management_indicator_rate_strict",
        "description_length",
        "seniority_final_entry_share_of_known",
        "yoe_le2_share",
    ]
    headlines = cal[cal.metric.isin(headline_metrics)].copy()
    headlines = headlines.sort_values("calibration_ratio", ascending=False)
    headlines.to_csv(TABS / "headline_metrics_with_snr.csv", index=False)
    return headlines


def main():
    df = load_core_frame()
    print(f"loaded {len(df):,} SWE LinkedIn rows")

    tech = load_tech_matrix()
    print(f"loaded tech matrix {len(tech):,}")

    step1_univariate(df)
    step2_anomalies(df)
    step3_native_entry_diagnostic(df)
    step5_junior_share(df)
    step6_ranked_changes(df, tech)
    step8_company_size(df, tech)
    sensitivity_summary(df)

    print("done")


if __name__ == "__main__":
    main()
