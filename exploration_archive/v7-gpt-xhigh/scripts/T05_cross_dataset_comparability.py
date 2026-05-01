#!/usr/bin/env python3
"""T05 cross-dataset comparability diagnostics.

Reads only the SWE LinkedIn subset from data/unified.parquet into memory after
DuckDB filtering. All full-file access is through DuckDB.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
TABLE_DIR = ROOT / "exploration" / "tables" / "T05"
FIG_DIR = ROOT / "exploration" / "figures" / "T05"
SUMMARY_PATH = TABLE_DIR / "summary.json"

SOURCES = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]
SOURCE_LABEL = {
    "kaggle_arshkon": "arshkon",
    "kaggle_asaniczka": "asaniczka",
    "scraped": "scraped_linkedin",
}


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def cramer_v_from_chi2(chi2: float, n: int, rows: int, cols: int) -> float:
    denom = n * max(1, min(rows - 1, cols - 1))
    return float(np.sqrt(chi2 / denom)) if denom else float("nan")


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def pairwise_chi_square(df: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for s1, s2 in combinations(SOURCES, 2):
        sub = df[df["source"].isin([s1, s2])]
        tab = pd.crosstab(sub["source"], sub[column])
        chi2, p, dof, _ = stats.chi2_contingency(tab.values)
        n = int(tab.values.sum())
        rows.append(
            {
                "metric": column,
                "source_a": SOURCE_LABEL[s1],
                "source_b": SOURCE_LABEL[s2],
                "n": n,
                "columns": tab.shape[1],
                "chi2": chi2,
                "dof": int(dof),
                "p_value": p,
                "cramers_v": cramer_v_from_chi2(chi2, n, tab.shape[0], tab.shape[1]),
            }
        )
    return pd.DataFrame(rows)


def load_linkedin_swe() -> pd.DataFrame:
    con = duckdb.connect()
    query = f"""
        SELECT
            uid,
            source,
            title_normalized,
            company_name_canonical,
            company_industry,
            description_length,
            is_multi_location,
            state_normalized,
            seniority_final,
            seniority_native,
            yoe_extracted
        FROM read_parquet('{DATA.as_posix()}')
        WHERE source_platform = 'linkedin'
          AND is_english = true
          AND date_flag = 'ok'
          AND is_swe = true
          AND source IN ('kaggle_arshkon', 'kaggle_asaniczka', 'scraped')
    """
    return con.execute(query).fetchdf()


def description_length_outputs(df: pd.DataFrame) -> pd.DataFrame:
    desc = df.dropna(subset=["description_length"]).copy()
    rows = []
    for s1, s2 in combinations(SOURCES, 2):
        x = desc.loc[desc["source"] == s1, "description_length"].to_numpy()
        y = desc.loc[desc["source"] == s2, "description_length"].to_numpy()
        ks = stats.ks_2samp(x, y)
        rows.append(
            {
                "source_a": SOURCE_LABEL[s1],
                "source_b": SOURCE_LABEL[s2],
                "n_a": len(x),
                "n_b": len(y),
                "mean_a": float(np.mean(x)),
                "mean_b": float(np.mean(y)),
                "median_a": float(np.median(x)),
                "median_b": float(np.median(y)),
                "ks_stat": float(ks.statistic),
                "p_value": float(ks.pvalue),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "description_length_ks.csv", index=False)

    quant = (
        desc.groupby("source")["description_length"]
        .quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        .unstack()
        .rename(columns=lambda c: f"q{int(c * 100):02d}")
        .reset_index()
    )
    quant["source"] = quant["source"].map(SOURCE_LABEL)
    quant.to_csv(TABLE_DIR / "description_length_quantiles.csv", index=False)

    upper = float(desc["description_length"].quantile(0.99))
    plt.figure(figsize=(9, 5))
    bins = np.linspace(0, upper, 60)
    for source in SOURCES:
        vals = desc.loc[desc["source"] == source, "description_length"].clip(upper=upper)
        plt.hist(vals, bins=bins, density=True, alpha=0.38, label=SOURCE_LABEL[source])
    plt.xlabel("Description length, characters (clipped at pooled 99th percentile)")
    plt.ylabel("Density")
    plt.title("T05: Description length distributions, SWE LinkedIn")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "description_length_hist.png", dpi=150)
    plt.close()
    return out


def company_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    company_df = df.dropna(subset=["company_name_canonical"]).copy()
    sets = {
        s: set(company_df.loc[company_df["source"] == s, "company_name_canonical"].astype(str))
        for s in SOURCES
    }
    rows = []
    for s1, s2 in combinations(SOURCES, 2):
        rows.append(
            {
                "source_a": SOURCE_LABEL[s1],
                "source_b": SOURCE_LABEL[s2],
                "n_companies_a": len(sets[s1]),
                "n_companies_b": len(sets[s2]),
                "intersection": len(sets[s1] & sets[s2]),
                "union": len(sets[s1] | sets[s2]),
                "jaccard": jaccard(sets[s1], sets[s2]),
            }
        )
    jacc = pd.DataFrame(rows)
    jacc.to_csv(TABLE_DIR / "company_jaccard.csv", index=False)

    top_counts = (
        company_df.groupby(["source", "company_name_canonical"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["source", "n"], ascending=[True, False])
    )
    top50 = top_counts.groupby("source").head(50)
    top50.to_csv(TABLE_DIR / "company_top50_by_source.csv", index=False)
    top_sets = {
        s: set(top50.loc[top50["source"] == s, "company_name_canonical"].astype(str))
        for s in SOURCES
    }
    top_rows = []
    for s1, s2 in combinations(SOURCES, 2):
        common = sorted(top_sets[s1] & top_sets[s2])
        top_rows.append(
            {
                "source_a": SOURCE_LABEL[s1],
                "source_b": SOURCE_LABEL[s2],
                "top50_overlap_n": len(common),
                "top50_jaccard": jaccard(top_sets[s1], top_sets[s2]),
                "common_top50_companies": "; ".join(common),
            }
        )
    top_overlap = pd.DataFrame(top_rows)
    top_overlap.to_csv(TABLE_DIR / "company_top50_overlap.csv", index=False)
    return jacc, top_overlap


def geography_outputs(df: pd.DataFrame) -> pd.DataFrame:
    exclusion = (
        df.assign(state_known=df["state_normalized"].notna())
        .groupby("source")
        .agg(
            total_swe=("uid", "size"),
            multi_location=("is_multi_location", "sum"),
            state_known=("state_known", "sum"),
        )
        .reset_index()
    )
    exclusion["state_excluded"] = exclusion["total_swe"] - exclusion["state_known"]
    exclusion["state_excluded_share"] = exclusion["state_excluded"] / exclusion["total_swe"]
    exclusion["source"] = exclusion["source"].map(SOURCE_LABEL)
    exclusion.to_csv(TABLE_DIR / "state_exclusions.csv", index=False)

    state_df = df.dropna(subset=["state_normalized"]).copy()
    state_counts = (
        state_df.groupby(["source", "state_normalized"])
        .size()
        .rename("n")
        .reset_index()
    )
    source_totals = state_counts.groupby("source")["n"].transform("sum")
    state_counts["share_of_state_known"] = state_counts["n"] / source_totals
    state_counts["source"] = state_counts["source"].map(SOURCE_LABEL)
    state_counts.to_csv(TABLE_DIR / "state_counts.csv", index=False)

    chi = pairwise_chi_square(state_df, "state_normalized")
    chi.to_csv(TABLE_DIR / "state_share_chi2.csv", index=False)

    top_states = (
        state_counts.groupby("state_normalized")["n"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .index
    )
    plot = state_counts[state_counts["state_normalized"].isin(top_states)]
    pivot = plot.pivot_table(
        index="state_normalized", columns="source", values="share_of_state_known", fill_value=0
    ).loc[top_states]
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Share of state-known SWE LinkedIn rows")
    plt.xlabel("State")
    plt.title("T05: Top state shares by source")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "state_share_top15.png", dpi=150)
    plt.close()
    return chi


def seniority_outputs(df: pd.DataFrame) -> pd.DataFrame:
    known = df[df["seniority_final"].notna() & (df["seniority_final"] != "unknown")].copy()
    dist = (
        known.groupby(["source", "seniority_final"])
        .size()
        .rename("n")
        .reset_index()
    )
    dist["known_seniority_denominator"] = dist.groupby("source")["n"].transform("sum")
    dist["share_of_known_seniority"] = dist["n"] / dist["known_seniority_denominator"]
    dist["source"] = dist["source"].map(SOURCE_LABEL)
    dist.to_csv(TABLE_DIR / "seniority_final_distribution_known.csv", index=False)
    unknown = (
        df.groupby("source")
        .agg(
            total_swe=("uid", "size"),
            unknown_seniority=("seniority_final", lambda s: int((s == "unknown").sum())),
        )
        .reset_index()
    )
    unknown["unknown_share"] = unknown["unknown_seniority"] / unknown["total_swe"]
    unknown["source"] = unknown["source"].map(SOURCE_LABEL)
    unknown.to_csv(TABLE_DIR / "seniority_unknown_rates.csv", index=False)

    chi = pairwise_chi_square(known, "seniority_final")
    chi.to_csv(TABLE_DIR / "seniority_final_chi2_known.csv", index=False)

    pivot = dist.pivot_table(
        index="seniority_final", columns="source", values="share_of_known_seniority", fill_value=0
    )
    pivot.plot(kind="bar", figsize=(8, 5))
    plt.ylabel("Share of known-seniority SWE LinkedIn rows")
    plt.xlabel("seniority_final")
    plt.title("T05: seniority_final distribution, unknown excluded")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "seniority_distribution_known.png", dpi=150)
    plt.close()
    return chi


def title_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    title_df = df.dropna(subset=["title_normalized"]).copy()
    sets = {
        s: set(title_df.loc[title_df["source"] == s, "title_normalized"].astype(str))
        for s in SOURCES
    }
    rows = []
    for s1, s2 in combinations(SOURCES, 2):
        rows.append(
            {
                "source_a": SOURCE_LABEL[s1],
                "source_b": SOURCE_LABEL[s2],
                "n_titles_a": len(sets[s1]),
                "n_titles_b": len(sets[s2]),
                "intersection": len(sets[s1] & sets[s2]),
                "union": len(sets[s1] | sets[s2]),
                "jaccard": jaccard(sets[s1], sets[s2]),
            }
        )
    title_j = pd.DataFrame(rows)
    title_j.to_csv(TABLE_DIR / "title_jaccard.csv", index=False)

    counts = (
        title_df.assign(period_group=np.where(title_df["source"] == "scraped", "2026_scraped", "2024_kaggle"))
        .groupby(["period_group", "title_normalized"])
        .size()
        .rename("n")
        .reset_index()
    )
    hist_titles = set(counts.loc[counts["period_group"] == "2024_kaggle", "title_normalized"])
    scraped_titles = set(counts.loc[counts["period_group"] == "2026_scraped", "title_normalized"])
    unique_rows = []
    for group, titles in [
        ("2024_only", hist_titles - scraped_titles),
        ("2026_only", scraped_titles - hist_titles),
    ]:
        sub = counts[counts["title_normalized"].isin(titles)].copy()
        sub["unique_period"] = group
        unique_rows.append(sub)
    unique = pd.concat(unique_rows, ignore_index=True)
    unique = unique.sort_values(["unique_period", "n"], ascending=[True, False])
    unique.groupby("unique_period").head(100).to_csv(TABLE_DIR / "titles_unique_to_period_top100.csv", index=False)
    return title_j, unique


def industry_outputs(df: pd.DataFrame) -> pd.DataFrame:
    industry_df = df[
        df["source"].isin(["kaggle_arshkon", "scraped"]) & df["company_industry"].notna()
    ].copy()
    counts = (
        industry_df.groupby(["source", "company_industry"])
        .size()
        .rename("n")
        .reset_index()
    )
    counts["denominator_industry_known"] = counts.groupby("source")["n"].transform("sum")
    counts["share_of_industry_known"] = counts["n"] / counts["denominator_industry_known"]
    counts["source"] = counts["source"].map(SOURCE_LABEL)
    counts.sort_values(["source", "n"], ascending=[True, False]).to_csv(
        TABLE_DIR / "industry_arshkon_vs_scraped.csv", index=False
    )
    tab = pd.crosstab(industry_df["source"], industry_df["company_industry"])
    chi2, p, dof, _ = stats.chi2_contingency(tab.values)
    n = int(tab.values.sum())
    out = pd.DataFrame(
        [
            {
                "source_a": "arshkon",
                "source_b": "scraped_linkedin",
                "n": n,
                "industry_categories": tab.shape[1],
                "chi2": chi2,
                "dof": int(dof),
                "p_value": p,
                "cramers_v": cramer_v_from_chi2(chi2, n, tab.shape[0], tab.shape[1]),
            }
        ]
    )
    out.to_csv(TABLE_DIR / "industry_chi2_arshkon_vs_scraped.csv", index=False)
    return out


def calibration_summary(
    desc_ks: pd.DataFrame,
    company_j: pd.DataFrame,
    state_chi: pd.DataFrame,
    senior_chi: pd.DataFrame,
    title_j: pd.DataFrame,
    industry_chi: pd.DataFrame,
) -> pd.DataFrame:
    def pair_lookup(frame: pd.DataFrame, col: str, a: str, b: str) -> float:
        row = frame[
            ((frame["source_a"] == a) & (frame["source_b"] == b))
            | ((frame["source_a"] == b) & (frame["source_b"] == a))
        ]
        return float(row.iloc[0][col]) if len(row) else float("nan")

    rows = []
    specs = [
        ("description_length_ks", desc_ks, "ks_stat", "larger means less comparable"),
        ("company_jaccard_distance", company_j, "jaccard", "larger distance means less overlap"),
        ("state_share_cramers_v", state_chi, "cramers_v", "larger means less comparable"),
        ("seniority_known_cramers_v", senior_chi, "cramers_v", "larger means less comparable"),
        ("title_jaccard_distance", title_j, "jaccard", "larger distance means less overlap"),
    ]
    for metric, frame, col, interpretation in specs:
        within = pair_lookup(frame, col, "arshkon", "asaniczka")
        ar_scr = pair_lookup(frame, col, "arshkon", "scraped_linkedin")
        as_scr = pair_lookup(frame, col, "asaniczka", "scraped_linkedin")
        if "jaccard_distance" in metric:
            within_effect = 1 - within
            ar_effect = 1 - ar_scr
            as_effect = 1 - as_scr
        else:
            within_effect = within
            ar_effect = ar_scr
            as_effect = as_scr
        rows.append(
            {
                "metric": metric,
                "within_2024_effect_arshkon_vs_asaniczka": within_effect,
                "arshkon_vs_scraped_effect": ar_effect,
                "asaniczka_vs_scraped_effect": as_effect,
                "arshkon_signal_to_within_2024_noise": abs(ar_effect) / abs(within_effect)
                if within_effect
                else float("nan"),
                "asaniczka_signal_to_within_2024_noise": abs(as_effect) / abs(within_effect)
                if within_effect
                else float("nan"),
                "interpretation": interpretation,
            }
        )
    rows.append(
        {
            "metric": "industry_cramers_v",
            "within_2024_effect_arshkon_vs_asaniczka": np.nan,
            "arshkon_vs_scraped_effect": float(industry_chi.iloc[0]["cramers_v"]),
            "asaniczka_vs_scraped_effect": np.nan,
            "arshkon_signal_to_within_2024_noise": np.nan,
            "asaniczka_signal_to_within_2024_noise": np.nan,
            "interpretation": "asaniczka has no industry data; no within-2024 calibration possible",
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "calibration_summary.csv", index=False)
    return out


def platform_labeling_stability() -> dict[str, object]:
    con = duckdb.connect()
    linkedin = con.execute(
        f"""
        SELECT
            uid, source, title_normalized, seniority_native, seniority_final, yoe_extracted
        FROM read_parquet('{DATA.as_posix()}')
        WHERE source_platform = 'linkedin'
          AND is_english = true
          AND date_flag = 'ok'
          AND is_swe = true
          AND source IN ('kaggle_arshkon', 'scraped')
          AND title_normalized IS NOT NULL
        """
    ).fetchdf()
    title_counts = (
        linkedin.groupby(["source", "title_normalized"]).size().rename("n").reset_index()
    )
    pivot = title_counts.pivot_table(
        index="title_normalized", columns="source", values="n", fill_value=0
    )
    common = pivot[(pivot.get("kaggle_arshkon", 0) > 0) & (pivot.get("scraped", 0) > 0)].copy()
    common["combined_n"] = common.get("kaggle_arshkon", 0) + common.get("scraped", 0)
    top_titles = list(common.sort_values("combined_n", ascending=False).head(20).index)
    pd.DataFrame({"title_normalized": top_titles}).to_csv(
        TABLE_DIR / "platform_top20_common_titles.csv", index=False
    )

    top = linkedin[linkedin["title_normalized"].isin(top_titles)].copy()
    top["seniority_native_filled"] = top["seniority_native"].fillna("null")
    native_dist = (
        top.groupby(["title_normalized", "source", "seniority_native_filled"])
        .size()
        .rename("n")
        .reset_index()
    )
    native_dist["denominator_title_source"] = native_dist.groupby(["title_normalized", "source"])[
        "n"
    ].transform("sum")
    native_dist["share"] = native_dist["n"] / native_dist["denominator_title_source"]
    native_dist["source"] = native_dist["source"].map(SOURCE_LABEL)
    native_dist.to_csv(TABLE_DIR / "platform_title_seniority_native_distribution.csv", index=False)

    title_chi_rows = []
    for title, sub in top.groupby("title_normalized"):
        tab = pd.crosstab(sub["source"], sub["seniority_native_filled"])
        if tab.shape[0] == 2 and tab.shape[1] > 1 and tab.values.sum() >= 10:
            chi2, p, dof, _ = stats.chi2_contingency(tab.values)
            n = int(tab.values.sum())
            title_chi_rows.append(
                {
                    "title_normalized": title,
                    "n": n,
                    "labels": tab.shape[1],
                    "chi2": chi2,
                    "dof": int(dof),
                    "p_value": p,
                    "cramers_v": cramer_v_from_chi2(chi2, n, tab.shape[0], tab.shape[1]),
                }
            )
    pd.DataFrame(title_chi_rows).sort_values("cramers_v", ascending=False).to_csv(
        TABLE_DIR / "platform_title_native_chi2.csv", index=False
    )

    yoe_rows = []
    for (title, label), sub in top.dropna(subset=["yoe_extracted"]).groupby(
        ["title_normalized", "seniority_native_filled"]
    ):
        if set(sub["source"]) != {"kaggle_arshkon", "scraped"}:
            continue
        x = sub.loc[sub["source"] == "kaggle_arshkon", "yoe_extracted"].to_numpy()
        y = sub.loc[sub["source"] == "scraped", "yoe_extracted"].to_numpy()
        if len(x) < 5 or len(y) < 5:
            continue
        ks = stats.ks_2samp(x, y)
        yoe_rows.append(
            {
                "title_normalized": title,
                "seniority_native": label,
                "arshkon_yoe_known_n": len(x),
                "scraped_yoe_known_n": len(y),
                "arshkon_yoe_mean": float(np.mean(x)),
                "scraped_yoe_mean": float(np.mean(y)),
                "arshkon_yoe_median": float(np.median(x)),
                "scraped_yoe_median": float(np.median(y)),
                "ks_stat": float(ks.statistic),
                "p_value": float(ks.pvalue),
            }
        )
    yoe_out = pd.DataFrame(yoe_rows).sort_values("ks_stat", ascending=False)
    yoe_out.to_csv(TABLE_DIR / "platform_title_seniority_yoe_ks.csv", index=False)

    indeed = con.execute(
        f"""
        SELECT
            title_normalized,
            seniority_final,
            yoe_extracted
        FROM read_parquet('{DATA.as_posix()}')
        WHERE source = 'scraped'
          AND source_platform = 'indeed'
          AND is_english = true
          AND date_flag = 'ok'
          AND is_swe = true
          AND title_normalized IS NOT NULL
        """
    ).fetchdf()
    indeed_top = indeed[indeed["title_normalized"].isin(top_titles)].copy()
    indeed_rows = []
    for title, sub in indeed_top.groupby("title_normalized"):
        known = sub[sub["seniority_final"].notna() & (sub["seniority_final"] != "unknown")]
        yoe_known = sub[sub["yoe_extracted"].notna()]
        indeed_rows.append(
            {
                "title_normalized": title,
                "indeed_n": len(sub),
                "seniority_known_n": len(known),
                "seniority_known_share": len(known) / len(sub) if len(sub) else np.nan,
                "entry_share_of_known_seniority": (known["seniority_final"] == "entry").mean()
                if len(known)
                else np.nan,
                "yoe_known_n": len(yoe_known),
                "yoe_known_share": len(yoe_known) / len(sub) if len(sub) else np.nan,
                "yoe_le2_share_of_yoe_known": (yoe_known["yoe_extracted"] <= 2).mean()
                if len(yoe_known)
                else np.nan,
            }
        )
    indeed_out = pd.DataFrame(indeed_rows).sort_values("indeed_n", ascending=False)
    indeed_out.to_csv(TABLE_DIR / "platform_indeed_cross_validation_top_titles.csv", index=False)

    overall_known = indeed[indeed["seniority_final"].notna() & (indeed["seniority_final"] != "unknown")]
    overall_yoe = indeed.dropna(subset=["yoe_extracted"])
    return {
        "top_common_titles": top_titles,
        "native_title_tests_n": len(title_chi_rows),
        "yoe_title_cell_tests_n": len(yoe_rows),
        "indeed_overall_n": int(len(indeed)),
        "indeed_seniority_known_n": int(len(overall_known)),
        "indeed_entry_share_known": float((overall_known["seniority_final"] == "entry").mean())
        if len(overall_known)
        else None,
        "indeed_yoe_known_n": int(len(overall_yoe)),
        "indeed_yoe_le2_share": float((overall_yoe["yoe_extracted"] <= 2).mean())
        if len(overall_yoe)
        else None,
    }


def main() -> None:
    ensure_dirs()
    df = load_linkedin_swe()
    df["source"] = pd.Categorical(df["source"], categories=SOURCES, ordered=True)

    sample = (
        df.groupby("source", observed=False)
        .agg(
            n_swe_linkedin=("uid", "size"),
            description_length_known=("description_length", lambda s: int(s.notna().sum())),
            company_known=("company_name_canonical", lambda s: int(s.notna().sum())),
            title_known=("title_normalized", lambda s: int(s.notna().sum())),
            seniority_known=("seniority_final", lambda s: int(((s.notna()) & (s != "unknown")).sum())),
            yoe_known=("yoe_extracted", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    sample["source"] = sample["source"].map(SOURCE_LABEL)
    sample.to_csv(TABLE_DIR / "analysis_sample_counts.csv", index=False)

    desc_ks = description_length_outputs(df)
    company_j, top_overlap = company_outputs(df)
    state_chi = geography_outputs(df)
    senior_chi = seniority_outputs(df)
    title_j, _unique_titles = title_outputs(df)
    industry_chi = industry_outputs(df)
    calibration = calibration_summary(desc_ks, company_j, state_chi, senior_chi, title_j, industry_chi)
    platform_summary = platform_labeling_stability()

    summary = {
        "sample_counts": sample.to_dict(orient="records"),
        "description_ks": desc_ks.to_dict(orient="records"),
        "company_jaccard": company_j.to_dict(orient="records"),
        "company_top50_overlap": top_overlap.to_dict(orient="records"),
        "state_chi2": state_chi.to_dict(orient="records"),
        "seniority_chi2": senior_chi.to_dict(orient="records"),
        "title_jaccard": title_j.to_dict(orient="records"),
        "industry_chi2": industry_chi.to_dict(orient="records"),
        "calibration_summary": calibration.to_dict(orient="records"),
        "platform_labeling_stability": platform_summary,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote T05 tables to {TABLE_DIR}")
    print(f"Wrote T05 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
