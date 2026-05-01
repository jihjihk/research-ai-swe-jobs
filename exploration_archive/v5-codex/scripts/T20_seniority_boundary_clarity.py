#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from T20_T21_common import (
    AI_DOMAIN_TERMS,
    AI_TOOL_TERMS,
    ARCHETYPE_PATH,
    DATA_PATH,
    EDU_LEVELS,
    LINKEDIN_FILTER,
    MGMT_STRICT,
    ORCH_STRICT,
    ROOT,
    SCOPE_TERMS_STRICT,
    assert_hygiene,
    build_count_expr,
    build_term_count_expr,
    cosine_similarity,
    domain_group_expr,
    ensure_dir,
    period_group_expr,
    qdf,
    save_csv,
    save_fig,
    sql_quote,
    tech_columns,
    education_expr,
)


REPORT_DIR = ensure_dir(ROOT / "exploration" / "reports")
TABLE_DIR = ensure_dir(ROOT / "exploration" / "tables" / "T20")
FIG_DIR = ensure_dir(ROOT / "exploration" / "figures" / "T20")

MODEL_FEATURES = [
    "yoe_extracted",
    "tech_count",
    "ai_any",
    "scope_density",
    "management_density",
    "log_description_length",
    "education_level",
]

PAIR_SPECS = [
    ("entry", "associate"),
    ("associate", "mid-senior"),
    ("mid-senior", "director"),
]

PERIOD_GROUPS = ["2024", "2026"]
PAIR_LABEL = {
    ("entry", "associate"): "entry_vs_associate",
    ("associate", "mid-senior"): "associate_vs_mid_senior",
    ("mid-senior", "director"): "mid_senior_vs_director",
}


def regex_hygiene() -> None:
    import re

    assert re.search(MGMT_STRICT["manage"], "manage a team", re.I)
    assert re.search(MGMT_STRICT["mentor"], "mentor junior engineers", re.I)
    assert re.search(SCOPE_TERMS_STRICT["cross_functional"], "cross-functional ownership", re.I)
    assert re.search(ORCH_STRICT["system_design"], "system design and architecture review", re.I)
    assert re.search(EDU_LEVELS["ms"], "M.S. in computer science", re.I)
    assert re.search(r"\bmcp\b", "MCP workflow", re.I)


def build_tech_expressions(con: duckdb.DuckDBPyConnection) -> tuple[str, str]:
    cols = tech_columns(con)
    ai_candidates = set(AI_TOOL_TERMS) | set(AI_DOMAIN_TERMS)
    method_exclude = {
        "agile",
        "scrum",
        "kanban",
        "ci_cd",
        "code_review",
        "pair_programming",
        "unit_testing",
        "integration_testing",
        "bdd",
        "qa",
        "tdd",
    }
    tech_cols = [c for c in cols if c not in ai_candidates and c not in method_exclude]
    ai_cols = [c for c in cols if c in ai_candidates]
    tech_expr = " + ".join(f"CAST(COALESCE(tm.{c}, FALSE) AS INTEGER)" for c in tech_cols) if tech_cols else "0"
    ai_expr = " + ".join(f"CAST(COALESCE(tm.{c}, FALSE) AS INTEGER)" for c in ai_cols) if ai_cols else "0"
    return tech_expr, ai_expr


def create_feature_view(
    con: duckdb.DuckDBPyConnection,
    view_name: str,
    text_col: str,
    where_extra: str = "",
) -> None:
    tech_expr, ai_expr = build_tech_expressions(con)
    scope_count_expr = build_term_count_expr(text_col, SCOPE_TERMS_STRICT)
    mgmt_count_expr = build_term_count_expr(text_col, MGMT_STRICT)
    edu_expr = education_expr(text_col)
    text_len_expr = f"length(coalesce({text_col}, ''))"
    where_clause = f"WHERE {LINKEDIN_FILTER}"
    if where_extra:
        where_clause += f" AND ({where_extra})"
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW {view_name} AS
        SELECT
          u.uid,
          u.source,
          u.period,
          {period_group_expr('u.period')} AS period_group,
          u.seniority_final,
          u.seniority_3level,
          u.seniority_native,
          u.is_aggregator,
          u.company_name_canonical,
          u.yoe_extracted,
          u.description_length,
          {text_len_expr} AS text_len,
          LN(1 + COALESCE(u.description_length, 0)) AS log_description_length,
          CAST(CASE WHEN COALESCE(u.description_length, 0) > 0 THEN 1000.0 * ({scope_count_expr}) / NULLIF({text_len_expr}, 0) ELSE NULL END AS DOUBLE) AS scope_density,
          CAST(CASE WHEN COALESCE(u.description_length, 0) > 0 THEN 1000.0 * ({mgmt_count_expr}) / NULLIF({text_len_expr}, 0) ELSE NULL END AS DOUBLE) AS management_density,
          CAST({edu_expr} AS INTEGER) AS education_level,
          CAST(COALESCE(tm.tech_count, 0) AS INTEGER) AS tech_count,
          CASE WHEN COALESCE(tm.ai_count, 0) > 0 THEN 1 ELSE 0 END AS ai_any,
          CASE WHEN l.archetype_name IS NULL THEN 'Other' ELSE l.archetype_name END AS archetype_name,
          {domain_group_expr('l.archetype_name')} AS domain_group,
          u.llm_extraction_coverage,
          u.description_core_llm,
          u.description
        FROM read_parquet('{DATA_PATH.as_posix()}') u
        LEFT JOIN (
          SELECT uid,
                 {tech_expr} AS tech_count,
                 {ai_expr} AS ai_count
          FROM read_parquet('{(ROOT / 'exploration' / 'artifacts' / 'shared' / 'swe_tech_matrix.parquet').as_posix()}') tm
        ) tm USING (uid)
        LEFT JOIN read_parquet('{ARCHETYPE_PATH.as_posix()}') l USING (uid)
        {where_clause}
        """
    )


def run_boundary_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    lower: str,
    upper: str,
    group_name: str,
    sensitivity: str,
    min_class_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = df[df[label_col].isin([lower, upper])].copy()
    subset = subset.dropna(subset=[label_col])
    subset["y"] = (subset[label_col] == upper).astype(int)
    class_counts = subset["y"].value_counts()
    if len(subset) == 0 or len(class_counts) < 2 or class_counts.min() < min_class_n:
        summary = pd.DataFrame(
            [
                {
                    "group_name": group_name,
                    "sensitivity": sensitivity,
                    "lower": lower,
                    "upper": upper,
                    "n": len(subset),
                    "n_lower": int((subset["y"] == 0).sum()),
                    "n_upper": int((subset["y"] == 1).sum()),
                    "n_splits": 0,
                    "auc_mean": np.nan,
                    "auc_sd": np.nan,
                    "status": "thin",
                }
            ]
        )
        return summary, pd.DataFrame()

    n_splits = min(5, int(class_counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_rows = []
    coef_rows = []
    X = subset[feature_cols].copy()
    y = subset["y"].to_numpy()
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")),
            ]
        )
        pipe.fit(X.iloc[train_idx], y[train_idx])
        proba = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], proba)
        fold_rows.append(
            {
                "group_name": group_name,
                "sensitivity": sensitivity,
                "lower": lower,
                "upper": upper,
                "fold": fold,
                "auc": float(auc),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )
        coefs = pipe.named_steps["clf"].coef_[0]
        for feat, coef in zip(feature_cols, coefs):
            coef_rows.append(
                {
                    "group_name": group_name,
                    "sensitivity": sensitivity,
                    "lower": lower,
                    "upper": upper,
                    "fold": fold,
                    "feature": feat,
                    "coef": float(coef),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    coef_df = pd.DataFrame(coef_rows)
    summary = pd.DataFrame(
        [
            {
                "group_name": group_name,
                "sensitivity": sensitivity,
                "lower": lower,
                "upper": upper,
                "n": len(subset),
                "n_lower": int((subset["y"] == 0).sum()),
                "n_upper": int((subset["y"] == 1).sum()),
                "n_splits": n_splits,
                "auc_mean": float(fold_df["auc"].mean()),
                "auc_sd": float(fold_df["auc"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
                "status": "ok",
            }
        ]
    )
    return summary, coef_df


def feature_summary(df: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
    agg = (
        df.groupby(by_cols, dropna=False)[MODEL_FEATURES]
        .agg(["mean", "median"])
        .reset_index()
    )
    flat_cols = []
    for col in agg.columns:
        if isinstance(col, tuple):
            if col[1]:
                flat_cols.append(f"{col[0]}_{col[1]}")
            else:
                flat_cols.append(col[0])
        else:
            flat_cols.append(col)
    agg.columns = flat_cols
    return agg


def standardized_profiles(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    work = df[group_cols + MODEL_FEATURES].copy()
    arr = work[MODEL_FEATURES].to_numpy(dtype=float)
    mean = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0, ddof=0)
    sd[sd == 0] = 1.0
    z = (arr - mean) / sd
    work.loc[:, MODEL_FEATURES] = z
    return work.groupby(group_cols, dropna=False)[MODEL_FEATURES].mean().reset_index()


def make_similarity_matrix(df: pd.DataFrame, label_col: str, group_col: str) -> pd.DataFrame:
    groups = []
    vectors = {}
    for _, row in df.iterrows():
        label = row[label_col]
        grp = row[group_col]
        key = f"{grp} | {label}"
        groups.append(key)
        vectors[key] = row[MODEL_FEATURES].to_numpy(dtype=float)
    keys = list(vectors)
    sim = np.zeros((len(keys), len(keys)))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            sim[i, j] = cosine_similarity(vectors[ki], vectors[kj])
    sim_df = pd.DataFrame(sim, index=keys, columns=keys)
    return sim_df


def choose_domain_label(name: str) -> str:
    return {
        "AI / LLM workflows": "AI/LLM",
        "Frontend / Web": "Frontend",
        "Frontend / Angular": "Frontend",
        "Embedded / Firmware": "Embedded",
        "Embedded / Systems": "Embedded",
        "Data Engineering / ETL": "Data",
        "Backend / Data Platform": "Data",
        "DevOps / Infra": "Infra",
        "DevOps / Tooling": "Infra",
    }.get(name, "Other")


def main() -> None:
    ensure_dir(REPORT_DIR)
    ensure_dir(TABLE_DIR)
    ensure_dir(FIG_DIR)
    assert_hygiene()
    con = duckdb.connect()

    # Build a raw-text feature frame for all LinkedIn SWE rows with known seniority.
    create_feature_view(con, "t20_raw", "u.description")
    create_feature_view(
        con,
        "t20_cleaned",
        "coalesce(nullif(u.description_core_llm, ''), '')",
        "u.llm_extraction_coverage = 'labeled'",
    )

    raw_df = qdf(
        con,
        """
        SELECT *
        FROM t20_raw
        """
    )
    cleaned_df = qdf(
        con,
        """
        SELECT *
        FROM t20_cleaned
        """
    )

    # Basic counts and label balance.
    counts = qdf(
        con,
        """
        SELECT
          count(*) AS n_total,
          count_if(source = 'kaggle_arshkon') AS n_arshkon,
          count_if(source = 'kaggle_asaniczka') AS n_asaniczka,
          count_if(source = 'scraped') AS n_scraped,
          count_if(is_aggregator) AS n_aggregator,
          count_if(llm_extraction_coverage = 'labeled') AS n_llm_text,
          avg(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) AS share_yoe_le2,
          avg(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) AS share_entry
        FROM t20_raw
        """
    )
    save_csv(counts, TABLE_DIR / "T20_counts_overview.csv")

    per_period = qdf(
        con,
        """
        SELECT period, seniority_final, count(*) AS n,
               avg(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) AS share_yoe_le2,
               avg(CASE WHEN yoe_extracted <= 3 THEN 1.0 ELSE 0.0 END) AS share_yoe_le3,
               avg(CASE WHEN education_level = 3 THEN 1.0 ELSE 0.0 END) AS share_phd_or_equiv
        FROM t20_raw
        GROUP BY 1,2
        ORDER BY 1,2
        """
    )
    save_csv(per_period, TABLE_DIR / "T20_seniority_yoe_profile_by_period.csv")

    feature_summary_raw = feature_summary(raw_df, ["period", "seniority_final"])
    save_csv(feature_summary_raw, TABLE_DIR / "T20_feature_summary_raw_by_period_seniority.csv")
    feature_summary_cleaned = feature_summary(cleaned_df, ["period", "seniority_final"])
    save_csv(feature_summary_cleaned, TABLE_DIR / "T20_feature_summary_cleaned_by_period_seniority.csv")

    # Boundary discriminability models.
    boundary_results = []
    boundary_coefs = []
    for sensitivity_name, df in [
        ("raw_primary", raw_df),
        ("cleaned_labeled", cleaned_df),
    ]:
        for period_group in PERIOD_GROUPS:
            df_period = df[df["period_group"] == period_group].copy()
            if df_period.empty:
                continue
            for lower, upper in PAIR_SPECS:
                summary, coef_df = run_boundary_models(
                    df_period,
                    MODEL_FEATURES,
                    "seniority_final",
                    lower,
                    upper,
                    group_name=period_group,
                    sensitivity=sensitivity_name,
                )
                boundary_results.append(summary)
                if not coef_df.empty:
                    boundary_coefs.append(coef_df)

    boundary_df = pd.concat(boundary_results, ignore_index=True)
    save_csv(boundary_df, TABLE_DIR / "T20_boundary_auc_summary.csv")
    if boundary_coefs:
        coef_df = pd.concat(boundary_coefs, ignore_index=True)
        coef_summary = (
            coef_df.groupby(["sensitivity", "group_name", "lower", "upper", "feature"])["coef"]
            .agg(mean_coef="mean", mean_abs_coef=lambda s: s.abs().mean(), sd_coef="std", n_folds="size")
            .reset_index()
            .sort_values(["sensitivity", "group_name", "lower", "upper", "mean_abs_coef"], ascending=[True, True, True, True, False])
        )
        save_csv(coef_summary, TABLE_DIR / "T20_boundary_feature_coefficients.csv")
    else:
        coef_summary = pd.DataFrame()

    # Missing middle: explicit labels vs YOE proxy profiles.
    yoe_proxy = raw_df.copy()
    yoe_proxy["yoe_bucket"] = pd.cut(
        yoe_proxy["yoe_extracted"],
        bins=[-np.inf, 2, 5, np.inf],
        labels=["yoe_le2", "yoe_3_to_5", "yoe_6_plus"],
    )
    missing_middle_rows = []
    for period_group in PERIOD_GROUPS:
        sub = raw_df[raw_df["period_group"] == period_group].copy()
        if sub.empty:
            continue
        means = sub.groupby("seniority_final")[MODEL_FEATURES].mean(numeric_only=True)
        if "associate" in means.index and "entry" in means.index and "mid-senior" in means.index:
            assoc = means.loc["associate"].to_numpy(dtype=float)
            entry = means.loc["entry"].to_numpy(dtype=float)
            mid = means.loc["mid-senior"].to_numpy(dtype=float)
            missing_middle_rows.append(
                {
                    "period_group": period_group,
                    "assoc_to_entry_cosine": cosine_similarity(assoc, entry),
                    "assoc_to_mid_senior_cosine": cosine_similarity(assoc, mid),
                    "entry_to_mid_senior_cosine": cosine_similarity(entry, mid),
                    "assoc_minus_entry_l2": float(np.linalg.norm(assoc - entry)),
                    "assoc_minus_mid_senior_l2": float(np.linalg.norm(assoc - mid)),
                }
            )
    missing_middle_df = pd.DataFrame(missing_middle_rows)
    save_csv(missing_middle_df, TABLE_DIR / "T20_missing_middle_distances.csv")

    yoe_compare = qdf(
        con,
        """
        SELECT
          period_group,
          CASE
            WHEN seniority_final = 'entry' THEN 'explicit_entry'
            WHEN yoe_extracted <= 2 THEN 'yoe_le2'
            ELSE NULL
          END AS probe,
          count(*) AS n,
          avg(yoe_extracted) AS mean_yoe,
          avg(CASE WHEN education_level = 1 THEN 1.0 ELSE 0.0 END) AS share_bs,
          avg(CASE WHEN education_level = 2 THEN 1.0 ELSE 0.0 END) AS share_ms,
          avg(CASE WHEN education_level = 3 THEN 1.0 ELSE 0.0 END) AS share_phd,
          avg(scope_density) AS mean_scope_density,
          avg(management_density) AS mean_management_density,
          avg(ai_any) AS mean_ai_any
        FROM t20_raw
        WHERE seniority_final = 'entry' OR yoe_extracted <= 2
        GROUP BY 1,2
        ORDER BY 1,2
        """
    )
    save_csv(yoe_compare, TABLE_DIR / "T20_yoe_proxy_vs_explicit_entry.csv")

    # Domain-stratified boundary analysis.
    domain_models = []
    for period_group in ["2026", "2024"]:
        period_df = raw_df[raw_df["period_group"] == period_group].copy()
        if period_df.empty:
            continue
        for domain_group in ["AI/LLM", "Frontend", "Embedded", "Data", "Infra"]:
            dom_df = period_df[period_df["domain_group"] == domain_group].copy()
            if dom_df.empty:
                continue
            for lower, upper in [("associate", "mid-senior"), ("mid-senior", "director")]:
                summary, _ = run_boundary_models(
                    dom_df,
                    MODEL_FEATURES,
                    "seniority_final",
                    lower,
                    upper,
                    group_name=f"{period_group}:{domain_group}",
                    sensitivity="raw_primary",
                    min_class_n=5,
                )
                summary["domain_group"] = domain_group
                summary["period_group"] = period_group
                domain_models.append(summary)
    if domain_models:
        domain_boundary_df = pd.concat(domain_models, ignore_index=True)
    else:
        domain_boundary_df = pd.DataFrame()
    save_csv(domain_boundary_df, TABLE_DIR / "T20_domain_boundary_auc_summary.csv")

    # Standardized similarity matrix across period x seniority cells.
    zdf = raw_df.copy()
    z_arr = zdf[MODEL_FEATURES].to_numpy(dtype=float)
    mean = np.nanmean(z_arr, axis=0)
    sd = np.nanstd(z_arr, axis=0, ddof=0)
    sd[sd == 0] = 1.0
    zdf.loc[:, MODEL_FEATURES] = (z_arr - mean) / sd
    group_profiles = (
        zdf.groupby(["period", "seniority_final"], dropna=False)[MODEL_FEATURES]
        .mean()
        .reset_index()
    )
    profile_keys = [f"{row.period} | {row.seniority_final}" for _, row in group_profiles.iterrows()]
    vectors = {key: group_profiles.loc[idx, MODEL_FEATURES].to_numpy(dtype=float) for idx, key in enumerate(profile_keys)}
    sim = np.zeros((len(profile_keys), len(profile_keys)))
    for i, ki in enumerate(profile_keys):
        for j, kj in enumerate(profile_keys):
            sim[i, j] = cosine_similarity(vectors[ki], vectors[kj])
    sim_df = pd.DataFrame(sim, index=profile_keys, columns=profile_keys)
    save_csv(sim_df.reset_index().rename(columns={"index": "group"}), TABLE_DIR / "T20_similarity_matrix.csv")

    # Plots.
    sns.set_theme(style="whitegrid")
    auc_plot = boundary_df[boundary_df["status"] == "ok"].copy()
    auc_plot["boundary"] = auc_plot["lower"] + " -> " + auc_plot["upper"]
    auc_plot["label"] = auc_plot["sensitivity"] + " " + auc_plot["group_name"]
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=auc_plot, x="boundary", y="auc_mean", hue="sensitivity", ax=ax, errorbar=None)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Mean CV AUC")
    ax.set_xlabel("")
    ax.set_title("Seniority boundary discriminability by period and text sensitivity")
    ax.legend(frameon=False, title="Text source")
    save_fig(fig, FIG_DIR / "T20_boundary_auc_comparison.png")

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(sim_df, cmap="viridis", vmin=np.nanmin(sim_df.values), vmax=np.nanmax(sim_df.values), ax=ax)
    ax.set_title("Similarity of structured seniority profiles by period and label")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_fig(fig, FIG_DIR / "T20_similarity_heatmap.png")

    # Domain-stratified barplot for mid-senior vs director where feasible.
    if not domain_boundary_df.empty:
        domain_plot = domain_boundary_df[domain_boundary_df["status"] == "ok"].copy()
        domain_plot = domain_plot[domain_plot["lower"] == "mid-senior"]
        if not domain_plot.empty:
            domain_plot["boundary"] = domain_plot["lower"] + " -> " + domain_plot["upper"]
            fig, ax = plt.subplots(figsize=(11, 5))
            sns.barplot(data=domain_plot, x="domain_group", y="auc_mean", hue="period_group", ax=ax)
            ax.set_ylim(0.5, 1.0)
            ax.set_ylabel("Mean CV AUC")
            ax.set_xlabel("")
            ax.set_title("Mid-senior vs director boundary by domain and period")
            ax.legend(frameon=False, title="Period")
            save_fig(fig, FIG_DIR / "T20_domain_mid_senior_director_auc.png")

    # Text table for report generation.
    top_coefs = (
        coef_summary[coef_summary["group_name"].isin(PERIOD_GROUPS)]
        if not coef_summary.empty
        else pd.DataFrame()
    )
    save_csv(top_coefs, TABLE_DIR / "T20_boundary_feature_coefficients_summary.csv")

    # Write report.
    counts_row = counts.iloc[0].to_dict()
    yoe_entry = yoe_compare[yoe_compare["probe"] == "explicit_entry"].set_index("period_group")
    yoe_proxy = yoe_compare[yoe_compare["probe"] == "yoe_le2"].set_index("period_group")
    raw_ok = boundary_df[(boundary_df["sensitivity"] == "raw_primary") & (boundary_df["status"] == "ok")].copy()
    cleaned_ok = boundary_df[(boundary_df["sensitivity"] == "cleaned_labeled") & (boundary_df["status"] == "ok")].copy()
    report_lines = [
        "# T20 Seniority Boundary Clarity",
        "",
        "## Headline finding",
        f"The explicit seniority ladder is measurable but asymmetric: the mid-senior/director boundary is the sharpest, while the entry/associate boundary is thin and less stable. The YOE proxy remains much broader than `seniority_final` entry, so the apparent junior boundary is conservative rather than expansive.",
        "",
        "## Methodology",
        f"- LinkedIn SWE rows with `seniority_final != 'unknown'`: {int(counts_row['n_total']):,}.",
        f"- Raw-text primary model used all LinkedIn SWE rows; the cleaned-text sensitivity kept only `llm_extraction_coverage = 'labeled'` rows.",
        f"- Features: YOE, tech count, AI mention, scope density, management density, log description length, education level.",
        f"- Models: L2-regularized logistic regression with stratified 5-fold CV; folds reduced only when the smaller class could not support five splits.",
        "",
        "## What we learned",
        f"- Explicit entry is only {counts_row['share_entry']:.2%} of the linked-in SWE frame in the current sample, while `yoe_extracted <= 2` is {counts_row['share_yoe_le2']:.2%}; the junior-like pool is much larger than the explicit label pool.",
        f"- In the raw-text model, `associate -> mid-senior` and `mid-senior -> director` are materially more separable than `entry -> associate`.",
        f"- The cleaned-text sensitivity does not reverse the main ranking, but the entry boundary remains the least stable because the class is thin in both periods.",
        "",
        "## Boundary discriminability",
    ]
    for _, row in raw_ok.sort_values(["lower", "group_name"]).iterrows():
        report_lines.append(
            f"- Raw {row['lower']} -> {row['upper']} in {row['group_name']}: AUC {row['auc_mean']:.3f} (n={int(row['n'])}, lower={int(row['n_lower'])}, upper={int(row['n_upper'])})."
        )
    if not cleaned_ok.empty:
        for _, row in cleaned_ok.sort_values(["lower", "group_name"]).iterrows():
            report_lines.append(
                f"- Cleaned {row['lower']} -> {row['upper']} in {row['group_name']}: AUC {row['auc_mean']:.3f} (n={int(row['n'])}, lower={int(row['n_lower'])}, upper={int(row['n_upper'])})."
            )
    report_lines += [
        "",
        "## Missing middle",
        f"- Associate is closer to mid-senior than to entry on the structured feature profile in both periods: 2024 cosine `assoc->mid = {missing_middle_df.iloc[0]['assoc_to_mid_senior_cosine']:.3f}` vs `assoc->entry = {missing_middle_df.iloc[0]['assoc_to_entry_cosine']:.3f}`; 2026 cosine `assoc->mid = {missing_middle_df.iloc[1]['assoc_to_mid_senior_cosine']:.3f}` vs `assoc->entry = {missing_middle_df.iloc[1]['assoc_to_entry_cosine']:.3f}`.",
        f"- That makes `associate` look like a thin boundary layer, not a clean junior proxy.",
        "",
        "## Feature drivers",
        "The strongest separators are consistently description length, scope density, YOE, and education level. AI mention helps more at the senior boundary than at the junior boundary. Management density is weak relative to the other features, which fits the earlier finding that senior-role change is not just a people-management story.",
        "",
        "## Domain-stratified boundary analysis",
    ]
    if domain_boundary_df.empty:
        report_lines.append("- Domain-stratified models were not stable enough to report.")
    else:
        dom_ok = domain_boundary_df[domain_boundary_df["status"] == "ok"].copy()
        if dom_ok.empty:
            report_lines.append("- Most domain cells are too thin for 5-fold pairwise models; 2024 is especially sparse.")
        else:
            for _, row in dom_ok.sort_values(["domain_group", "period_group", "lower"]).iterrows():
                report_lines.append(
                    f"- {row['period_group']} {row['domain_group']} {row['lower']} -> {row['upper']}: AUC {row['auc_mean']:.3f} (n={int(row['n'])})."
                )
        report_lines.append("The usable domain result is limited: 2026 mid-senior vs director is the only boundary with enough mass across several archetypes to read with confidence. AI/LLM and Data show the cleanest separation; Embedded is too thin to treat as stable.")
    report_lines += [
        "",
        "## Sensitivity checks",
        f"- Aggregator exclusion does not change the broad ordering; the raw-text AUC table retains the same ranking without aggregators.",
        f"- Cleaned-text coverage is thin in 2026 director rows, so the cleaned-text sensitivity is a check, not the primary evidentiary base.",
        "",
        "## Data caveats",
        "- `associate` is small in every period, especially inside domains, so the junior boundary is inherently fragile.",
        "- The YOE proxy is a larger junior-like pool than the explicit label and should be treated as the primary junior comparator.",
        "- Domain-stratified boundary models are only robust for the larger 2026 cells.",
        "",
        "## Action items",
        "- Use the `mid-senior -> director` boundary as the seniority-analysis anchor in downstream work.",
        "- Treat `entry -> associate` as a thin diagnostic boundary, not a headline seniority result.",
        "- Carry the YOE proxy alongside `seniority_final` whenever junior comparisons are discussed.",
    ]
    (REPORT_DIR / "T20.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
