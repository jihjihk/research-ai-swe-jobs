"""T33 — Requirements-section change as hiring-bar signal (H_B).

Tests whether 2024→2026 change in requirements-section size correlates with
implicit hiring-bar shifts (YOE, credential stack depth, tech count, education).

Primary classifier: T13's section classifier (already computed in
`exploration/artifacts/shared/T13_readability_metrics.parquet`).
Sensitivity classifier: `T33_simple_regex_classifier.simple_classify`.

Outputs
-------
exploration/tables/T33/
    section_classifier_comparison.csv          # T13 vs simple for period means
    period_regression.csv                      # OLS coefficients
    hiring_bar_correlations.csv                # req_section_share × YOE/stack/tech/edu
    within_company_scatter.csv                 # Δ(req) × Δ(J3) per company
    narrative_sample_50.csv                    # 50 largest-contraction postings
    alt_explanation_deltalen.csv               # corr(Δreq, Δdesc_length)
    within_2024_calibration.csv
    verdict_summary.csv

exploration/figures/T33/
    hiring_bar_correlation_heatmap.png
    within_company_scatter.png
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent.parent
sys.path.insert(0, str(THIS_DIR))

from T13_section_classifier import classify_sections, SECTION_LABELS  # type: ignore  # noqa: E402
from T33_simple_regex_classifier import simple_classify, SIMPLE_LABELS  # type: ignore  # noqa: E402

UNIFIED = REPO / "data" / "unified.parquet"
T13_PARQ = REPO / "exploration" / "artifacts" / "shared" / "T13_readability_metrics.parquet"
T11_PARQ = REPO / "exploration" / "artifacts" / "shared" / "T11_posting_features.parquet"
T09_PARQ = REPO / "exploration" / "artifacts" / "shared" / "swe_archetype_labels.parquet"
T16_VECTORS = REPO / "exploration" / "tables" / "T16" / "company_change_vectors.csv"
OUT_TAB = REPO / "exploration" / "tables" / "T33"
OUT_FIG = REPO / "exploration" / "figures" / "T33"
OUT_TAB.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

BASE_FILTER = (
    "is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok'"
)

RNG = 33333333


def load_corpus() -> pd.DataFrame:
    """Load the SWE LinkedIn corpus with raw + LLM-cleaned text + posting features."""
    con = duckdb.connect()
    q = f"""
    SELECT
      u.uid,
      u.source,
      CASE WHEN u.source LIKE 'kaggle_%' THEN '2024' ELSE '2026' END AS period_year,
      u.period,
      u.company_name_canonical,
      u.is_aggregator,
      u.seniority_final,
      u.seniority_3level,
      u.yoe_min_years_llm,
      u.yoe_extracted,
      u.llm_classification_coverage,
      u.llm_extraction_coverage,
      u.description,
      u.description_core_llm,
      u.title
    FROM read_parquet('{UNIFIED.as_posix()}') u
    WHERE {BASE_FILTER}
      AND u.description IS NOT NULL
      AND length(u.description) >= 50
    """
    df = con.execute(q).df()
    return df


def load_t13_metrics() -> pd.DataFrame:
    con = duckdb.connect()
    q = f"""
    SELECT uid,
           sec_requirements_chars,
           sec_requirements_share,
           sec_responsibilities_chars,
           sec_responsibilities_share,
           sec_benefits_chars,
           sec_benefits_share,
           sec_summary_chars,
           sec_about_company_chars,
           sec_legal_chars,
           sec_preferred_chars,
           sec_unclassified_chars,
           raw_length,
           flesch_kincaid_grade,
           inclusive_density,
           imperative_density
    FROM read_parquet('{T13_PARQ.as_posix()}')
    """
    return con.execute(q).df()


def load_t11_features() -> pd.DataFrame:
    con = duckdb.connect()
    q = f"""
    SELECT uid,
           tech_count,
           requirement_breadth,
           credential_stack_depth,
           tech_density,
           scope_density,
           ai_binary,
           education_level,
           description_cleaned_length,
           text_source
    FROM read_parquet('{T11_PARQ.as_posix()}')
    """
    return con.execute(q).df()


def compute_simple_classifier(df: pd.DataFrame) -> pd.DataFrame:
    """Run T33 simple-regex classifier on each posting's raw description."""
    print(f"[simple-regex] classifying {len(df):,} postings on raw description…")
    req = np.zeros(len(df), dtype=np.int64)
    resp = np.zeros(len(df), dtype=np.int64)
    ben = np.zeros(len(df), dtype=np.int64)
    other = np.zeros(len(df), dtype=np.int64)
    total = np.zeros(len(df), dtype=np.int64)
    t0 = time.time()
    descs = df["description"].tolist()
    for i, desc in enumerate(descs):
        if desc is None:
            continue
        c = simple_classify(desc)
        req[i] = c["requirements"]
        resp[i] = c["responsibilities"]
        ben[i] = c["benefits"]
        other[i] = c["other"]
        total[i] = c["total"]
        if (i + 1) % 10000 == 0:
            print(f"  {i+1:,} / {len(df):,}  ({time.time()-t0:.1f}s)")
    print(f"  done in {time.time()-t0:.1f}s")
    out = pd.DataFrame({
        "uid": df["uid"].values,
        "simple_req_chars": req,
        "simple_resp_chars": resp,
        "simple_benefits_chars": ben,
        "simple_other_chars": other,
        "simple_total": total,
    })
    # Share = req / total, safeguard 0-division
    safe_total = np.where(out["simple_total"].values > 0, out["simple_total"].values, 1)
    out["simple_req_share"] = out["simple_req_chars"].values / safe_total
    return out


def period_section_compare(df: pd.DataFrame) -> pd.DataFrame:
    """For T13 + simple classifier: period × classifier means on req share + chars."""
    rows = []
    for period in ("2024", "2026"):
        sub = df[df.period_year == period]
        for classifier, share_col, chars_col in [
            ("T13", "sec_requirements_share", "sec_requirements_chars"),
            ("simple_regex", "simple_req_share", "simple_req_chars"),
        ]:
            rows.append({
                "period": period,
                "classifier": classifier,
                "n": len(sub),
                "mean_req_share": float(sub[share_col].mean()),
                "mean_req_chars": float(sub[chars_col].mean()),
                "median_req_chars": float(sub[chars_col].median()),
            })
    out = pd.DataFrame(rows)
    # Delta rows
    delta_rows = []
    for classifier, share_col, chars_col in [
        ("T13", "sec_requirements_share", "sec_requirements_chars"),
        ("simple_regex", "simple_req_share", "simple_req_chars"),
    ]:
        share_2024 = df.loc[df.period_year == "2024", share_col].mean()
        share_2026 = df.loc[df.period_year == "2026", share_col].mean()
        chars_2024 = df.loc[df.period_year == "2024", chars_col].mean()
        chars_2026 = df.loc[df.period_year == "2026", chars_col].mean()
        delta_rows.append({
            "period": "Δ(2026-2024)",
            "classifier": classifier,
            "n": len(df),
            "mean_req_share": float(share_2026 - share_2024),
            "mean_req_chars": float(chars_2026 - chars_2024),
            "median_req_chars": float("nan"),
        })
    out = pd.concat([out, pd.DataFrame(delta_rows)], ignore_index=True)
    return out


def within_2024_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Compute within-2024 arshkon vs asaniczka gap for req_section_share + chars."""
    rows = []
    for classifier, share_col, chars_col in [
        ("T13", "sec_requirements_share", "sec_requirements_chars"),
        ("simple_regex", "simple_req_share", "simple_req_chars"),
    ]:
        arsh = df[(df.period_year == "2024") & (df.source == "kaggle_arshkon")]
        asan = df[(df.period_year == "2024") & (df.source == "kaggle_asaniczka")]
        scraped = df[df.period_year == "2026"]
        pooled24 = df[df.period_year == "2024"]
        w24_share_gap = abs(arsh[share_col].mean() - asan[share_col].mean())
        w24_chars_gap = abs(arsh[chars_col].mean() - asan[chars_col].mean())
        cross_share = abs(scraped[share_col].mean() - pooled24[share_col].mean())
        cross_chars = abs(scraped[chars_col].mean() - pooled24[chars_col].mean())
        rows.append({
            "classifier": classifier,
            "within_2024_share_gap": w24_share_gap,
            "cross_period_share_gap": cross_share,
            "SNR_share": cross_share / w24_share_gap if w24_share_gap > 0 else float("nan"),
            "within_2024_chars_gap": w24_chars_gap,
            "cross_period_chars_gap": cross_chars,
            "SNR_chars": cross_chars / w24_chars_gap if w24_chars_gap > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def period_regression(df: pd.DataFrame, share_col: str, label: str) -> pd.DataFrame:
    """Fit: req_section_share ~ period + seniority + archetype + is_aggregator +
    log(desc_length) + period×seniority + period×archetype.

    Period encoded as 0/1 (2024=0, 2026=1).
    """
    d = df.copy()
    d = d[d[share_col].notna()]
    d["period_2026"] = (d.period_year == "2026").astype(int)
    d["log_desc_len"] = np.log1p(d["raw_length"].astype(float))
    # Use seniority_final; drop rare associate (n=175)
    d = d[d.seniority_final.isin(["entry", "mid-senior", "director", "unknown"])]
    # Archetype — fill missing with "unclassified"
    if "archetype_label" in d.columns:
        d["archetype"] = d["archetype_label"].fillna("unclassified")
    else:
        d["archetype"] = "unclassified"
    # Collapse rare archetypes for regression efficiency
    counts = d["archetype"].value_counts()
    keep = counts[counts >= 500].index
    d["archetype"] = d["archetype"].where(d["archetype"].isin(keep), "other")
    d["is_aggregator_int"] = d["is_aggregator"].astype(int)
    # Baseline: period=2024, seniority=mid-senior, archetype=most common
    try:
        most_common_arche = counts.index[0]
    except Exception:
        most_common_arche = "unclassified"
    d["archetype"] = pd.Categorical(
        d["archetype"], categories=list(d["archetype"].unique()), ordered=False
    )
    d["seniority_final"] = pd.Categorical(
        d["seniority_final"], categories=["mid-senior", "entry", "director", "unknown"]
    )
    formula = (
        f"{share_col} ~ period_2026 + C(seniority_final) + C(archetype) "
        f"+ is_aggregator_int + log_desc_len "
        f"+ period_2026:C(seniority_final) + period_2026:C(archetype)"
    )
    print(f"[regression:{label}] fitting on n={len(d):,}")
    # Use HC3 robust SEs
    try:
        model = smf.ols(formula, data=d).fit(cov_type="HC3")
    except Exception as exc:
        print(f"  OLS failed: {exc}")
        return pd.DataFrame()
    tbl = pd.DataFrame({
        "coef": model.params,
        "se": model.bse,
        "t": model.tvalues,
        "p": model.pvalues,
        "ci_low": model.conf_int()[0],
        "ci_high": model.conf_int()[1],
    })
    tbl.index.name = "term"
    tbl = tbl.reset_index()
    tbl["classifier"] = label
    tbl["n"] = len(d)
    tbl["r2"] = float(model.rsquared)
    return tbl


def hiring_bar_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Within 2026 scraped SWE, compute corr(req_section_share, hiring-bar proxies)
    within J3 and S4 panel primaries + J2 / S1 sensitivities.
    """
    scraped = df[(df.period_year == "2026") & (df.llm_classification_coverage == "labeled")].copy()
    scraped["is_J3"] = scraped["yoe_min_years_llm"] <= 2
    scraped["is_S4"] = scraped["yoe_min_years_llm"] >= 5
    scraped["is_J2"] = scraped["yoe_min_years_llm"] <= 1
    scraped["is_S1"] = scraped["seniority_final"].isin(["mid-senior", "director"])
    # Pooled
    rows = []
    proxies = [
        ("yoe_min_years_llm", "YOE (LLM)"),
        ("credential_stack_depth", "credential stack depth"),
        ("tech_count", "tech count"),
        ("education_level", "education level"),
    ]
    subsets = {
        "all scraped 2026": scraped,
        "J3 (yoe≤2)": scraped[scraped.is_J3],
        "S4 (yoe≥5)": scraped[scraped.is_S4],
        "J2 (yoe≤1)": scraped[scraped.is_J2],
        "S1 (mid-sen+dir)": scraped[scraped.is_S1],
    }
    classifiers = [
        ("T13", "sec_requirements_share"),
        ("T13", "sec_requirements_chars"),
        ("simple_regex", "simple_req_share"),
        ("simple_regex", "simple_req_chars"),
    ]
    for subset_name, sub in subsets.items():
        for classifier, req_col in classifiers:
            for proxy_col, proxy_lbl in proxies:
                both = sub[[req_col, proxy_col]].dropna()
                if len(both) < 50:
                    continue
                # Spearman
                spear = both[req_col].corr(both[proxy_col], method="spearman")
                pear = both[req_col].corr(both[proxy_col], method="pearson")
                rows.append({
                    "subset": subset_name,
                    "classifier": classifier,
                    "req_variable": req_col,
                    "proxy": proxy_lbl,
                    "n": len(both),
                    "spearman_r": float(spear),
                    "pearson_r": float(pear),
                })
    return pd.DataFrame(rows)


def within_company_scatter(df: pd.DataFrame) -> pd.DataFrame:
    """Per-company Δ(req_section_share) vs Δ(J3_share) from 2024 → 2026.

    Uses arshkon∩scraped overlap panel — companies with ≥5 SWE postings in BOTH
    periods (same frame as T16 company_change_vectors.csv).
    """
    # Compute company means
    d = df.copy()
    d["is_J3"] = (d["yoe_min_years_llm"] <= 2).astype(float)
    d["is_J3"] = d["is_J3"].where(d["yoe_min_years_llm"].notna(), np.nan)
    # For req share use T13 primary + simple_regex sensitivity
    agg_2024 = (
        d[d.period_year == "2024"]
        .groupby("company_name_canonical", as_index=False)
        .agg(
            n_2024=("uid", "count"),
            req_share_t13_2024=("sec_requirements_share", "mean"),
            req_share_simple_2024=("simple_req_share", "mean"),
            req_chars_t13_2024=("sec_requirements_chars", "mean"),
            req_chars_simple_2024=("simple_req_chars", "mean"),
            j3_share_2024=("is_J3", "mean"),
            desc_len_2024=("raw_length", "mean"),
        )
    )
    agg_2026 = (
        d[d.period_year == "2026"]
        .groupby("company_name_canonical", as_index=False)
        .agg(
            n_2026=("uid", "count"),
            req_share_t13_2026=("sec_requirements_share", "mean"),
            req_share_simple_2026=("simple_req_share", "mean"),
            req_chars_t13_2026=("sec_requirements_chars", "mean"),
            req_chars_simple_2026=("simple_req_chars", "mean"),
            j3_share_2026=("is_J3", "mean"),
            desc_len_2026=("raw_length", "mean"),
        )
    )
    panel = agg_2024.merge(agg_2026, on="company_name_canonical", how="inner")
    panel = panel[(panel.n_2024 >= 5) & (panel.n_2026 >= 5)].copy()
    panel["d_req_share_t13"] = panel["req_share_t13_2026"] - panel["req_share_t13_2024"]
    panel["d_req_share_simple"] = panel["req_share_simple_2026"] - panel["req_share_simple_2024"]
    panel["d_req_chars_t13"] = panel["req_chars_t13_2026"] - panel["req_chars_t13_2024"]
    panel["d_req_chars_simple"] = panel["req_chars_simple_2026"] - panel["req_chars_simple_2024"]
    panel["d_j3_share"] = panel["j3_share_2026"] - panel["j3_share_2024"]
    panel["d_desc_len"] = panel["desc_len_2026"] - panel["desc_len_2024"]
    return panel


def narrative_sample(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Sample 50 2026 scraped SWE postings with LARGEST req-section contraction
    relative to (company, title) 2024 mean.
    """
    d = df.copy()
    scraped = d[d.period_year == "2026"].copy()
    k24 = d[d.period_year == "2024"].copy()
    # Company+title 2024 mean req share (T13)
    co_title_mean = (
        k24.groupby(["company_name_canonical", "title"], as_index=False)
        ["sec_requirements_share"].mean()
        .rename(columns={"sec_requirements_share": "req_share_2024_cotitle"})
    )
    scraped = scraped.merge(co_title_mean, on=["company_name_canonical", "title"], how="left")
    # Fallback: company-level 2024 mean if (co, title) missing
    co_mean = (
        k24.groupby("company_name_canonical", as_index=False)["sec_requirements_share"].mean()
        .rename(columns={"sec_requirements_share": "req_share_2024_co"})
    )
    scraped = scraped.merge(co_mean, on="company_name_canonical", how="left")
    scraped["req_share_2024_ref"] = scraped["req_share_2024_cotitle"].fillna(scraped["req_share_2024_co"])
    # Delta = 2026 share - reference
    scraped["delta_req_share"] = scraped["sec_requirements_share"] - scraped["req_share_2024_ref"]
    scraped = scraped[scraped.req_share_2024_ref.notna()]
    # Keep postings with longest raw_length above 1000 to avoid classifier noise from short
    scraped = scraped[scraped.raw_length >= 1000].copy()
    # Pick the 50 with most negative delta_req_share
    scraped = scraped.sort_values("delta_req_share", ascending=True).head(n).copy()
    cols_out = [
        "uid", "company_name_canonical", "title", "seniority_final",
        "yoe_min_years_llm", "raw_length", "sec_requirements_share",
        "req_share_2024_ref", "delta_req_share",
        "sec_responsibilities_share", "sec_benefits_share", "sec_summary_chars",
        "sec_about_company_chars", "sec_legal_chars",
    ]
    out = scraped[cols_out + ["description", "description_core_llm"]].copy()
    # Snapshot 1st 600 chars of raw description to CSV-friendly form
    def excerpt(t: str | None, n: int = 600) -> str:
        if t is None:
            return ""
        return t[:n].replace("\r", " ").replace("\n", " ")
    out["desc_excerpt_600"] = out["description"].apply(excerpt)
    return out


def narrative_classify(sample: pd.DataFrame) -> pd.DataFrame:
    """Programmatic classification of each narrative sample into:
       (a) Genuine technical-requirement migration into responsibilities
       (b) Pure culture/benefits expansion with no added requirements
       (c) Substantive requirement loosening ("no degree required", etc.)
       (d) Other
    """
    import re
    tech_req_kw = re.compile(
        r"\b(python|java\b|javascript|typescript|react|aws|azure|gcp|kubernetes|k8s|docker|terraform|sql|nosql|postgres|mysql|spark|hadoop|kafka|airflow|mongodb|redis|graphql|rest\s+api|microservices|system\s+design|distributed\s+systems|ci/cd|devops|machine\s+learning|ml\b|llm|ai\b|tensorflow|pytorch|scala|rust|go\b|golang|c\+\+|c#|ruby|rails|django|flask|node\.?js|spring|hibernate|oauth|saml|rsa|encryption|kubernetes|helm|jenkins|gitlab|github\s+actions|ansible|puppet|chef|grafana|prometheus|elk|elasticsearch|kibana|logstash|jira|agile|scrum|tdd|bdd|unit\s+test|integration\s+test)\b",
        re.IGNORECASE,
    )
    loose_kw = re.compile(
        r"\b(no\s+degree\s+(required|necessary)|self[\s-]taught|without\s+(a\s+)?degree|bootcamp(?:\s+ok)?|alternative\s+(education|credential)|experience\s+in\s+lieu|transferable\s+skills|we\s+will\s+train|training\s+provided|willing(ness)?\s+to\s+learn|open\s+to\s+all\s+backgrounds|0\+\s*years|entry[\s-]?level\s+welcome|new\s+grad(?:uate)?s?\s+welcome)\b",
        re.IGNORECASE,
    )
    benefits_kw = re.compile(
        r"\b(401\(?k\)?|medical\s+insurance|health\s+insurance|dental|vision|flexible\s+(hours|schedule|time)|unlimited\s+pto|paid\s+time\s+off|parental\s+leave|wellness|mental\s+health|remote[\s-]first|work[\s-]from[\s-]home|stock\s+options?|rsus|espp|diversity|inclusion|belonging|company\s+culture|team\s+events|offsite|retreat|free\s+lunch|snack|yoga|gym|fitness\s+reimbursement|home\s+office\s+stipend|ergonomic|pet[\s-]friendly)\b",
        re.IGNORECASE,
    )
    # For each sample posting, take the *non-requirements* portion
    # (responsibilities + summary + about_company) and count matches.
    # Here we just use the first 2000 chars of description as a proxy.
    rows = []
    for _, row in sample.iterrows():
        text = (row.get("description") or "")
        # We want to classify what *replaced* requirements. Focus on non-req sections:
        # Use heuristic on full text, then weight by section share movements.
        tech_hits = len(tech_req_kw.findall(text))
        loose_hits = len(loose_kw.findall(text))
        benefit_hits = len(benefits_kw.findall(text))
        # Tally shares of non-req expansion buckets
        classification = "d_other"
        # Simple decision tree:
        if loose_hits >= 1:
            classification = "c_requirement_loosening"
        elif benefit_hits >= 4 and tech_hits <= 3:
            classification = "b_culture_benefits_expansion"
        elif tech_hits >= 5:
            classification = "a_tech_req_migration_to_resp"
        rows.append({
            "uid": row["uid"],
            "company": row["company_name_canonical"],
            "title": row["title"],
            "seniority_final": row["seniority_final"],
            "yoe_min_years_llm": row["yoe_min_years_llm"],
            "delta_req_share": row["delta_req_share"],
            "req_share_2024_ref": row["req_share_2024_ref"],
            "req_share_2026": row["sec_requirements_share"],
            "tech_kw_hits": tech_hits,
            "loose_kw_hits": loose_hits,
            "benefit_kw_hits": benefit_hits,
            "classification": classification,
            "desc_excerpt": row["desc_excerpt_600"],
        })
    return pd.DataFrame(rows)


def alt_explanation(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation of Δ(req_section_share) with Δ(description_length) at
    posting level. Strongly positive → shrinkage is proportional/relative
    (narrative expansion dominates), not absolute.

    Can only compute at posting level indirectly; instead compute cross-period
    correlation at company level (from within_company panel). Primary here:
    company-level Δ × Δ.
    """
    panel = within_company_scatter(df)
    rows = []
    for classifier, d_share_col, d_chars_col in [
        ("T13", "d_req_share_t13", "d_req_chars_t13"),
        ("simple_regex", "d_req_share_simple", "d_req_chars_simple"),
    ]:
        both = panel[[d_share_col, "d_desc_len"]].dropna()
        both_chars = panel[[d_chars_col, "d_desc_len"]].dropna()
        rows.append({
            "classifier": classifier,
            "n_companies": len(both),
            "corr_d_reqshare_d_desclen": float(both[d_share_col].corr(both["d_desc_len"])),
            "corr_d_reqchars_d_desclen": float(both_chars[d_chars_col].corr(both_chars["d_desc_len"])),
        })
    return pd.DataFrame(rows)


def text_source_sensitivity_regression(df: pd.DataFrame) -> pd.DataFrame:
    """Re-fit period regression on full corpus raw vs labeled-only (LLM frame)."""
    out_frames = []
    # Full corpus
    tbl_full = period_regression(df, "sec_requirements_share", "T13_full")
    out_frames.append(tbl_full.assign(subset="full_corpus"))
    tbl_full2 = period_regression(df, "simple_req_share", "simple_full")
    out_frames.append(tbl_full2.assign(subset="full_corpus"))
    # LLM frame only
    llm_df = df[df.llm_extraction_coverage == "labeled"].copy()
    tbl_llm = period_regression(llm_df, "sec_requirements_share", "T13_llm_frame")
    out_frames.append(tbl_llm.assign(subset="llm_frame_only"))
    tbl_llm2 = period_regression(llm_df, "simple_req_share", "simple_llm_frame")
    out_frames.append(tbl_llm2.assign(subset="llm_frame_only"))
    return pd.concat(out_frames, ignore_index=True)


def main():
    print("=" * 70)
    print("T33 — Requirements-section change as hiring-bar signal (H_B)")
    print("=" * 70)

    # 1. Load data
    print("[load] unified SWE LinkedIn corpus")
    df = load_corpus()
    print(f"  n rows = {len(df):,}")

    # 2. Merge in T13 section metrics
    print("[load] T13 section metrics")
    t13 = load_t13_metrics()
    df = df.merge(t13, on="uid", how="left")
    missing_t13 = df["sec_requirements_share"].isna().sum()
    print(f"  merged; {missing_t13:,} rows missing T13 metrics")
    # drop any rows without T13 metrics (should be 0 given T13 covers 68137)
    df = df[df["sec_requirements_share"].notna()].copy()
    print(f"  n after T13 drop = {len(df):,}")

    # 3. Merge T11 features (tech_count, credential_stack_depth, education_level, ai_binary)
    print("[load] T11 features")
    t11 = load_t11_features()
    df = df.merge(t11, on="uid", how="left")

    # 4. Merge T09 archetype labels (coverage: 8,000 of 68,137; rest = unclassified)
    print("[load] T09 archetype labels")
    con = duckdb.connect()
    try:
        t09 = con.execute(
            f"SELECT uid, archetype_name AS archetype_label FROM read_parquet('{T09_PARQ.as_posix()}')"
        ).df()
        df = df.merge(t09, on="uid", how="left")
        covered = df["archetype_label"].notna().sum()
        print(f"  archetype_label covered {covered:,} of {len(df):,} rows")
    except Exception as exc:
        print(f"  T09 load failed: {exc}")
        df["archetype_label"] = pd.NA

    # 5. Run simple-regex classifier on raw text
    print("[classify] simple-regex sensitivity classifier")
    simple = compute_simple_classifier(df)
    df = df.merge(simple, on="uid", how="left")

    # 6. Section classifier comparison (period means)
    print("[analyze] section classifier comparison (T13 vs simple)")
    compare = period_section_compare(df)
    compare.to_csv(OUT_TAB / "section_classifier_comparison.csv", index=False)
    print(compare.to_string(index=False))

    # 7. Within-2024 calibration
    print("[analyze] within-2024 calibration")
    cal = within_2024_calibration(df)
    cal.to_csv(OUT_TAB / "within_2024_calibration.csv", index=False)
    print(cal.to_string(index=False))

    # 8. Period-effect regression (text-source sensitivity)
    print("[analyze] period regression + text-source sensitivity")
    reg = text_source_sensitivity_regression(df)
    reg.to_csv(OUT_TAB / "period_regression.csv", index=False)
    # Print period_2026 rows only for quick read
    preview = reg[reg["term"].astype(str).str.startswith("period_2026") & ~reg["term"].astype(str).str.contains(":")]
    print(preview.to_string(index=False))

    # 9. Hiring-bar correlations in 2026 scraped
    print("[analyze] hiring-bar correlations in 2026 scraped")
    corr = hiring_bar_correlations(df)
    corr.to_csv(OUT_TAB / "hiring_bar_correlations.csv", index=False)
    # Print primary rows
    primary = corr[
        (corr.subset.isin(["all scraped 2026", "J3 (yoe≤2)", "S4 (yoe≥5)"]))
        & (corr.req_variable.isin(["sec_requirements_share", "simple_req_share"]))
    ]
    print(primary.to_string(index=False))

    # 10. Within-company scatter (arshkon∩scraped overlap panel)
    print("[analyze] within-company scatter (overlap panel)")
    panel = within_company_scatter(df)
    panel.to_csv(OUT_TAB / "within_company_scatter.csv", index=False)
    print(f"  n companies with ≥5 in both periods = {len(panel)}")
    if len(panel) >= 10:
        corr_t13 = panel[["d_req_share_t13", "d_j3_share"]].corr().iloc[0, 1]
        corr_simple = panel[["d_req_share_simple", "d_j3_share"]].corr().iloc[0, 1]
        print(f"  corr(Δ req_share_t13, Δ J3_share) = {corr_t13:+.3f}")
        print(f"  corr(Δ req_share_simple, Δ J3_share) = {corr_simple:+.3f}")

    # 11. Narrative 50-sample classification
    print("[analyze] narrative 50-sample classification")
    sample = narrative_sample(df, n=50)
    narr = narrative_classify(sample)
    narr.to_csv(OUT_TAB / "narrative_sample_50.csv", index=False)
    # Summary
    summary = narr["classification"].value_counts()
    print("Classification breakdown (50 samples with largest req-share contraction):")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # 12. Alt-explanation check — Δ(req) vs Δ(desc_length)
    print("[analyze] alt-explanation: Δ(req) vs Δ(desc_length)")
    alt = alt_explanation(df)
    alt.to_csv(OUT_TAB / "alt_explanation_deltalen.csv", index=False)
    print(alt.to_string(index=False))

    # 13. Figures
    print("[figure] hiring-bar correlation heatmap")
    _figure_corr_heatmap(corr)
    print("[figure] within-company scatter")
    _figure_within_co_scatter(panel)

    # 14. Verdict summary table
    print("[verdict] summary")
    verdict = _build_verdict(compare, cal, reg, corr, panel, alt, narr)
    verdict.to_csv(OUT_TAB / "verdict_summary.csv", index=False)
    print(verdict.to_string(index=False))

    print("\n[done] outputs under exploration/tables/T33/")


def _figure_corr_heatmap(corr: pd.DataFrame) -> None:
    # Build matrix: rows = (classifier, req_var, subset), cols = proxy
    sub = corr[corr.subset.isin(["all scraped 2026", "J3 (yoe≤2)", "S4 (yoe≥5)"]) &
               corr.req_variable.isin(["sec_requirements_share", "simple_req_share"])].copy()
    sub["row_label"] = sub["classifier"] + " | " + sub["req_variable"] + " | " + sub["subset"]
    pivot = sub.pivot(index="row_label", columns="proxy", values="spearman_r")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="black" if abs(v) < 0.2 else "white", fontsize=7)
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("T33 — req section vs hiring-bar proxies (2026 scraped)")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "hiring_bar_correlation_heatmap.png", dpi=110)
    plt.close(fig)


def _figure_within_co_scatter(panel: pd.DataFrame) -> None:
    if len(panel) < 10:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (col, label) in zip(axes, [
        ("d_req_share_t13", "T13 classifier"),
        ("d_req_share_simple", "Simple-regex classifier"),
    ]):
        x = panel[col].values
        y = panel["d_j3_share"].values
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], alpha=0.5, s=12)
        if mask.sum() > 2:
            r = np.corrcoef(x[mask], y[mask])[0, 1]
            coef = np.polyfit(x[mask], y[mask], 1)
            xx = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(xx, np.polyval(coef, xx), color="red", linewidth=1.2,
                    label=f"r={r:+.3f}")
            ax.legend(loc="best")
        ax.axhline(0, color="grey", alpha=0.3, linewidth=0.5)
        ax.axvline(0, color="grey", alpha=0.3, linewidth=0.5)
        ax.set_xlabel(f"Δ req_section_share ({label})")
        ax.set_ylabel("Δ J3 (YOE≤2) share")
        ax.set_title(f"{label}: n={mask.sum()} companies")
    fig.suptitle("T33 — Within-company: Δ req-share vs Δ J3-share")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "within_company_scatter.png", dpi=110)
    plt.close(fig)


def _build_verdict(compare, cal, reg, corr, panel, alt, narr) -> pd.DataFrame:
    rows = []
    # period effect T13 / simple
    reg_p13 = reg[(reg["classifier"] == "T13_full") & (reg["term"] == "period_2026")]
    reg_ps = reg[(reg["classifier"] == "simple_full") & (reg["term"] == "period_2026")]
    rows.append({
        "key": "period effect T13 (full corpus req_share)",
        "value": f"{reg_p13['coef'].iloc[0]:+.4f}" if len(reg_p13) else "NA",
        "ci_95": (f"[{reg_p13['ci_low'].iloc[0]:+.4f}, {reg_p13['ci_high'].iloc[0]:+.4f}]"
                   if len(reg_p13) else "NA"),
    })
    rows.append({
        "key": "period effect simple_regex (full corpus req_share)",
        "value": f"{reg_ps['coef'].iloc[0]:+.4f}" if len(reg_ps) else "NA",
        "ci_95": (f"[{reg_ps['ci_low'].iloc[0]:+.4f}, {reg_ps['ci_high'].iloc[0]:+.4f}]"
                   if len(reg_ps) else "NA"),
    })
    # SNR
    cal_t13 = cal[cal.classifier == "T13"].iloc[0]
    cal_s = cal[cal.classifier == "simple_regex"].iloc[0]
    rows.append({
        "key": "T13 within-2024 share SNR",
        "value": f"{cal_t13['SNR_share']:.2f}",
        "ci_95": "NA",
    })
    rows.append({
        "key": "simple within-2024 share SNR",
        "value": f"{cal_s['SNR_share']:.2f}",
        "ci_95": "NA",
    })
    # Hiring-bar proxies (J3)
    for clf, req_var in [("T13", "sec_requirements_share"), ("simple_regex", "simple_req_share")]:
        sub = corr[(corr.classifier == clf)
                   & (corr.req_variable == req_var)
                   & (corr.subset == "all scraped 2026")]
        for _, r in sub.iterrows():
            rows.append({
                "key": f"[{clf}] corr({req_var}, {r['proxy']})  (all 2026 scraped)",
                "value": f"ρ={r['spearman_r']:+.3f}",
                "ci_95": f"n={r['n']}",
            })
    # Within-company correlation
    if len(panel) >= 10:
        r1 = panel[["d_req_share_t13", "d_j3_share"]].corr().iloc[0, 1]
        r2 = panel[["d_req_share_simple", "d_j3_share"]].corr().iloc[0, 1]
        rows.append({
            "key": "within-co corr(Δreq_share_t13, Δj3_share)",
            "value": f"{r1:+.3f}",
            "ci_95": f"n={len(panel)}",
        })
        rows.append({
            "key": "within-co corr(Δreq_share_simple, Δj3_share)",
            "value": f"{r2:+.3f}",
            "ci_95": f"n={len(panel)}",
        })
    # Alt-explanation
    for _, r in alt.iterrows():
        rows.append({
            "key": f"[{r['classifier']}] corr(Δreq_share, Δdesc_len) (company-level)",
            "value": f"{r['corr_d_reqshare_d_desclen']:+.3f}",
            "ci_95": f"n={r['n_companies']}",
        })
    # Narrative
    vc = narr["classification"].value_counts(normalize=True)
    for k, v in vc.items():
        rows.append({
            "key": f"narrative 50 — {k} share",
            "value": f"{v:.1%}",
            "ci_95": "n=50",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
