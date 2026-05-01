"""T20 — Seniority boundary clarity (supervised-feature discriminability).

Complement to T15 (unsupervised text-centroid cosine): uses logistic regression
on T11 structured features to test whether boundaries between adjacent seniority
levels sharpened 2024→2026.

Filters:
- SWE LinkedIn default filter (source_platform='linkedin', is_english=true,
  date_flag='ok', is_swe=true).
- seniority_final != 'unknown'.

Outputs:
- exploration/tables/T20/auc_by_boundary_period.csv (AUC 5-fold CV by boundary × period)
- exploration/tables/T20/feature_importance_by_boundary_period.csv (top-5 coefficients)
- exploration/tables/T20/gap_evolution.csv  (Δgap + attribution table per feature)
- exploration/tables/T20/yoe_period_interaction.csv  (coefs, CI per feature)
- exploration/tables/T20/missing_middle.csv
- exploration/tables/T20/domain_stratified_auc.csv
- exploration/tables/T20/feature_heatmap_long.csv
- exploration/figures/T20/*.png

Sensitivities:
- (a) aggregator: run with is_aggregator=false on separate rows of AUC table.
- (c) T30 panel: gap evolution reported under S4/J3 primary (seniority_final senior,
  seniority_final entry) AND S1/S2 and J1/J2 YOE-based variants.
- (g) SWE tier optional — reported as simple descriptor.

Gate 2 pre-commits:
- Primary J3/S4, pooled-2024 baseline; T30 4-row panel for gap-evolution reporting.
- Within-2024 SNR (arshkon vs asaniczka) for sensitivity check.
"""

from __future__ import annotations

import os
import sys
import json
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import statsmodels.api as sm

ROOT = Path("/home/jihgaboot/gabor/job-research")
OUT = ROOT / "exploration" / "tables" / "T20"
FIG = ROOT / "exploration" / "figures" / "T20"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RNG = 20202020

# -------- Features

# Structured features available in T11_posting_features.parquet
FEATURES = [
    "yoe_min_years_llm",
    "tech_count",
    "ai_binary",
    "scope_density",
    "mgmt_strong_density",
    "description_cleaned_length",
    "education_level",
    "requirement_breadth",
    "credential_stack_depth",
]

# Adjacent-level pairs for boundary AUC
BOUNDARY_PAIRS = [
    ("entry", "associate"),
    ("associate", "mid-senior"),
    ("mid-senior", "director"),
]


def load_merged() -> pd.DataFrame:
    """Load T11 features joined with seniority_final, seniority_3level, analysis metadata."""
    con = duckdb.connect()
    q = """
        SELECT
            t.uid,
            t.tech_count,
            t.ai_binary,
            t.scope_density,
            t.mgmt_strong_density,
            t.mgmt_broad_density,
            t.yoe_min_years_llm,
            t.description_cleaned_length,
            t.education_level,
            t.requirement_breadth,
            t.credential_stack_depth,
            t.period,
            t.source,
            u.seniority_final,
            u.seniority_3level,
            u.company_name_canonical,
            u.is_aggregator,
            u.title,
            u.yoe_extracted,
            u.analysis_group
        FROM read_parquet(?) t
        LEFT JOIN read_parquet(?) u ON t.uid = u.uid
        WHERE u.source_platform='linkedin'
          AND u.is_english=TRUE
          AND u.date_flag='ok'
          AND u.is_swe=TRUE
    """
    df = con.execute(
        q,
        [
            str(ROOT / "exploration/artifacts/shared/T11_posting_features.parquet"),
            str(ROOT / "data/unified.parquet"),
        ],
    ).fetchdf()

    # period2: 2024 (arshkon + asaniczka) vs 2026 (scraped)
    def mk(p):
        if isinstance(p, str) and p.startswith("2024"):
            return "2024"
        if isinstance(p, str) and p.startswith("2026"):
            return "2026"
        return None

    df["period2"] = df["period"].astype(str).apply(mk)
    df = df[df["period2"].isin(["2024", "2026"])].copy()

    # YOE fallback: use yoe_extracted if yoe_min_years_llm null
    df["yoe_eff"] = df["yoe_min_years_llm"].where(
        df["yoe_min_years_llm"].notna(), df["yoe_extracted"]
    )
    # Median impute for yoe_eff where both null (only for AUC feature matrix)
    return df


def build_feature_matrix(dfc: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Fill yoe from yoe_eff, median-impute remaining nulls."""
    X = dfc.copy()
    # Swap yoe_min_years_llm -> yoe_eff for modeling purposes (fall back)
    X["yoe_min_years_llm_mod"] = X["yoe_eff"]
    feats_mod = []
    for f in features:
        f2 = "yoe_min_years_llm_mod" if f == "yoe_min_years_llm" else f
        feats_mod.append(f2)
    Xf = X[feats_mod].copy()
    # Median impute
    imp = SimpleImputer(strategy="median")
    Xf_imp = pd.DataFrame(imp.fit_transform(Xf), columns=feats_mod, index=Xf.index)
    return Xf_imp, feats_mod


def auc_cv(Xmat: np.ndarray, y: np.ndarray, random_state: int = RNG) -> tuple[float, list[float]]:
    """5-fold stratified CV AUC (L2 logistic). Returns (mean, per-fold)."""
    if len(np.unique(y)) < 2:
        return float("nan"), []
    if (y.sum() < 5) or ((1 - y).sum() < 5):
        return float("nan"), []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    aucs = []
    for tr, te in skf.split(Xmat, y):
        sc = StandardScaler().fit(Xmat[tr])
        Xtr = sc.transform(Xmat[tr])
        Xte = sc.transform(Xmat[te])
        clf = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=random_state
        )
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        try:
            a = roc_auc_score(y[te], p)
        except Exception:
            a = float("nan")
        aucs.append(a)
    return float(np.nanmean(aucs)), aucs


def fit_full_coeffs(Xmat: np.ndarray, y: np.ndarray, feat_names: list[str]) -> pd.DataFrame:
    """Fit L2 logistic on full data, return per-feature standardized coefficients."""
    sc = StandardScaler().fit(Xmat)
    Xs = sc.transform(Xmat)
    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=RNG
    )
    clf.fit(Xs, y)
    return pd.DataFrame({"feature": feat_names, "coef_std": clf.coef_[0]})


def run_boundary_auc(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each (period, boundary_pair) compute AUC CV + full-data standardized coefficients."""
    rows = []
    imp_rows = []
    for period in ["2024", "2026"]:
        dfp = df[df["period2"] == period].copy()
        for low, high in BOUNDARY_PAIRS:
            sub = dfp[dfp["seniority_final"].isin([low, high])].copy()
            n_low = (sub["seniority_final"] == low).sum()
            n_high = (sub["seniority_final"] == high).sum()
            if n_low < 10 or n_high < 10:
                rows.append(
                    {
                        "period": period,
                        "boundary": f"{low}↔{high}",
                        "n_low": int(n_low),
                        "n_high": int(n_high),
                        "auc_mean": float("nan"),
                        "auc_folds": "",
                        "note": "underpowered (<10 per side)",
                    }
                )
                continue
            Xmat, feat_names = build_feature_matrix(sub, features)
            y = (sub["seniority_final"].values == high).astype(int)
            mean_auc, folds = auc_cv(Xmat.values, y)
            rows.append(
                {
                    "period": period,
                    "boundary": f"{low}↔{high}",
                    "n_low": int(n_low),
                    "n_high": int(n_high),
                    "auc_mean": mean_auc,
                    "auc_folds": ",".join(f"{a:.4f}" for a in folds),
                    "note": "",
                }
            )
            coefs = fit_full_coeffs(Xmat.values, y, feat_names)
            coefs["period"] = period
            coefs["boundary"] = f"{low}↔{high}"
            imp_rows.append(coefs)

    # Also an overall all-adjacent senior vs junior (aggregated robust)
    for period in ["2024", "2026"]:
        dfp = df[df["period2"] == period].copy()
        # seniority_3level junior vs senior
        sub = dfp[dfp["seniority_3level"].isin(["junior", "senior"])].copy()
        n_j = (sub["seniority_3level"] == "junior").sum()
        n_s = (sub["seniority_3level"] == "senior").sum()
        if n_j < 10 or n_s < 10:
            rows.append(
                {
                    "period": period,
                    "boundary": "junior↔senior (3level)",
                    "n_low": int(n_j),
                    "n_high": int(n_s),
                    "auc_mean": float("nan"),
                    "auc_folds": "",
                    "note": "underpowered",
                }
            )
            continue
        Xmat, feat_names = build_feature_matrix(sub, features)
        y = (sub["seniority_3level"].values == "senior").astype(int)
        mean_auc, folds = auc_cv(Xmat.values, y)
        rows.append(
            {
                "period": period,
                "boundary": "junior↔senior (3level)",
                "n_low": int(n_j),
                "n_high": int(n_s),
                "auc_mean": mean_auc,
                "auc_folds": ",".join(f"{a:.4f}" for a in folds),
                "note": "3level aggregate",
            }
        )
        coefs = fit_full_coeffs(Xmat.values, y, feat_names)
        coefs["period"] = period
        coefs["boundary"] = "junior↔senior (3level)"
        imp_rows.append(coefs)

    imp_df = pd.concat(imp_rows, ignore_index=True) if imp_rows else pd.DataFrame()
    return pd.DataFrame(rows), imp_df


def run_auc_aggregator_sensitivity(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Sensitivity (a): re-run AUC with is_aggregator=false."""
    sub = df[df["is_aggregator"] != True].copy()  # noqa
    auc_df, _ = run_boundary_auc(sub, features)
    auc_df["sensitivity"] = "aggregator_excluded"
    return auc_df


def run_auc_no_yoe(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Sensitivity: drop YOE from feature set (so we know how much AUC is YOE-driven)."""
    feats_no_yoe = [f for f in features if f != "yoe_min_years_llm"]
    auc_df, _ = run_boundary_auc(df, feats_no_yoe)
    auc_df["sensitivity"] = "no_yoe"
    return auc_df


# -------------------- Gap evolution + attribution


def build_panel_groups(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Define T30 4-row panel groups for junior and senior.

    Primary: J3 (yoe<=2) vs S4 (yoe>=5) using yoe_min_years_llm.
    Sensitivities: J1 (seniority_final entry), J2 (seniority_3level junior);
                   S1 (seniority_final mid-senior+director), S2 (seniority_final director only).

    Returns dict panel_name -> df subset.
    """
    panels = {}

    # J3 / S4 primaries (YOE-based, label-independent)
    yoe_base = df[df["yoe_min_years_llm"].notna()].copy()
    panels["J3"] = yoe_base[yoe_base["yoe_min_years_llm"] <= 2].copy()
    panels["S4"] = yoe_base[yoe_base["yoe_min_years_llm"] >= 5].copy()

    # J1 / S1 (label-based sensitivities)
    panels["J1_entry"] = df[df["seniority_final"] == "entry"].copy()
    panels["S1_midsenior"] = df[df["seniority_final"] == "mid-senior"].copy()

    # J2 (seniority_3level junior); S2 director-only
    panels["J2_junior3"] = df[df["seniority_3level"] == "junior"].copy()
    panels["S2_director"] = df[df["seniority_final"] == "director"].copy()

    return panels


def gap_evolution_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """For each feature, compute gap_2024/gap_2026/Δgap/ΔM_senior/ΔM_junior + attribution.

    Reports per panel pair (J3/S4 primary; J1/S1 and J2/S2 sensitivities).
    """
    # Use yoe_eff where yoe_min_years_llm is the modeling feature
    dfw = df.copy()
    dfw["yoe_min_years_llm"] = dfw["yoe_eff"].where(dfw["yoe_eff"].notna(), dfw["yoe_min_years_llm"])

    panels = build_panel_groups(dfw)

    panel_pairs = [
        ("J3", "S4", "primary (yoe)"),
        ("J1_entry", "S1_midsenior", "sens label-based"),
        ("J2_junior3", "S2_director", "sens 3level+director"),
    ]

    rows = []
    for junior_name, senior_name, pair_note in panel_pairs:
        dfJ = panels[junior_name]
        dfS = panels[senior_name]
        for feat in features:
            mean_J_2024 = dfJ.loc[dfJ["period2"] == "2024", feat].dropna().mean()
            mean_J_2026 = dfJ.loc[dfJ["period2"] == "2026", feat].dropna().mean()
            mean_S_2024 = dfS.loc[dfS["period2"] == "2024", feat].dropna().mean()
            mean_S_2026 = dfS.loc[dfS["period2"] == "2026", feat].dropna().mean()
            gap_2024 = mean_S_2024 - mean_J_2024
            gap_2026 = mean_S_2026 - mean_J_2026
            dgap = gap_2026 - gap_2024
            dM_senior = mean_S_2026 - mean_S_2024
            dM_junior = mean_J_2026 - mean_J_2024
            denom = abs(dM_senior) + abs(dM_junior)
            attribution_senior = (abs(dM_senior) / denom) if denom > 1e-12 else float("nan")

            rows.append(
                {
                    "panel_pair": f"{junior_name} vs {senior_name}",
                    "note": pair_note,
                    "feature": feat,
                    "n_J_2024": int(dfJ.loc[dfJ["period2"] == "2024", feat].dropna().size),
                    "n_J_2026": int(dfJ.loc[dfJ["period2"] == "2026", feat].dropna().size),
                    "n_S_2024": int(dfS.loc[dfS["period2"] == "2024", feat].dropna().size),
                    "n_S_2026": int(dfS.loc[dfS["period2"] == "2026", feat].dropna().size),
                    "mean_J_2024": mean_J_2024,
                    "mean_J_2026": mean_J_2026,
                    "mean_S_2024": mean_S_2024,
                    "mean_S_2026": mean_S_2026,
                    "gap_2024": gap_2024,
                    "gap_2026": gap_2026,
                    "delta_gap": dgap,
                    "delta_M_senior": dM_senior,
                    "delta_M_junior": dM_junior,
                    "attribution_senior": attribution_senior,
                }
            )

    return pd.DataFrame(rows)


def within_2024_snr(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Within-2024 calibration: arshkon vs asaniczka absolute feature gap for SNR context."""
    df24 = df[df["period2"] == "2024"].copy()
    rows = []
    for feat in features:
        s_ars = df24[df24["source"] == "kaggle_arshkon"][feat].dropna()
        s_asa = df24[df24["source"] == "kaggle_asaniczka"][feat].dropna()
        if len(s_ars) < 10 or len(s_asa) < 10:
            continue
        within_gap = abs(s_ars.mean() - s_asa.mean())
        rows.append(
            {
                "feature": feat,
                "mean_arshkon": s_ars.mean(),
                "mean_asaniczka": s_asa.mean(),
                "within_2024_abs_gap": within_gap,
            }
        )
    return pd.DataFrame(rows)


# -------------------- YOE × period interaction


def yoe_period_interaction(df: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    """OLS: M ~ yoe + period + yoe*period + log(desc_length)  on LLM-frame rows."""
    dfw = df[df["yoe_min_years_llm"].notna()].copy()
    dfw["log_len"] = np.log1p(dfw["description_cleaned_length"].astype(float).clip(lower=1))
    dfw["period_2026"] = (dfw["period2"] == "2026").astype(float)
    dfw["yoe_c"] = dfw["yoe_min_years_llm"].astype(float).clip(lower=0, upper=20)
    dfw["yoe_x_p"] = dfw["yoe_c"] * dfw["period_2026"]

    rows = []
    for out in outcomes:
        d = dfw[[out, "yoe_c", "period_2026", "yoe_x_p", "log_len"]].dropna().copy()
        if len(d) < 100:
            continue
        # Force all numeric
        d[out] = d[out].astype(float)
        d["yoe_c"] = d["yoe_c"].astype(float)
        d["period_2026"] = d["period_2026"].astype(float)
        d["yoe_x_p"] = d["yoe_x_p"].astype(float)
        d["log_len"] = d["log_len"].astype(float)
        X = sm.add_constant(d[["yoe_c", "period_2026", "yoe_x_p", "log_len"]]).astype(float)
        y = d[out].astype(float)
        try:
            model = sm.OLS(y.values, X.values).fit(cov_type="HC1")
        except Exception as e:
            print(f"[yoe_period_interaction] OLS failed for {out}: {e}", flush=True)
            continue
        # X columns are: const, yoe_c, period_2026, yoe_x_p, log_len
        p = model.params
        se = model.bse
        pv = model.pvalues
        row = {
            "outcome": out,
            "n": len(d),
            "intercept": float(p[0]),
            "yoe_coef": float(p[1]),
            "yoe_se": float(se[1]),
            "period_2026_coef": float(p[2]),
            "period_2026_se": float(se[2]),
            "yoe_x_period_coef": float(p[3]),
            "yoe_x_period_se": float(se[3]),
            "yoe_x_period_p": float(pv[3]),
            "yoe_x_period_ci_low": float(p[3] - 1.96 * se[3]),
            "yoe_x_period_ci_high": float(p[3] + 1.96 * se[3]),
            "log_len_coef": float(p[4]),
            "r2": float(model.rsquared),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# -------------------- Missing middle


def missing_middle(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Distances (standardized) from associate centroid to entry, mid-senior, director centroids."""
    rows = []
    dfw = df.copy()
    dfw["yoe_min_years_llm"] = dfw["yoe_eff"].where(
        dfw["yoe_eff"].notna(), dfw["yoe_min_years_llm"]
    )
    # Standardize features across the full SWE pool (both periods combined)
    base = dfw[features].apply(pd.to_numeric, errors="coerce")
    means = base.mean()
    stds = base.std().replace(0, 1)
    std_base = (base - means) / stds

    dfw_std = dfw.copy()
    for f in features:
        dfw_std[f + "_z"] = std_base[f]

    levels = ["entry", "associate", "mid-senior", "director"]
    for period in ["2024", "2026"]:
        centroids = {}
        ns = {}
        for lev in levels:
            sub = dfw_std[(dfw_std["period2"] == period) & (dfw_std["seniority_final"] == lev)]
            ns[lev] = len(sub)
            if len(sub) >= 10:
                centroids[lev] = sub[[f + "_z" for f in features]].mean().values
        if "associate" not in centroids:
            continue
        for lev in levels:
            if lev == "associate" or lev not in centroids:
                continue
            dist = float(np.linalg.norm(centroids["associate"] - centroids[lev]))
            rows.append(
                {
                    "period": period,
                    "from": "associate",
                    "to": lev,
                    "n_associate": ns.get("associate", 0),
                    f"n_{lev}": ns.get(lev, 0),
                    "euclidean_distance_z": dist,
                }
            )
    return pd.DataFrame(rows)


# -------------------- Domain-stratified


def domain_stratified_auc(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """AUC within each domain archetype (from swe_archetype_labels.parquet)."""
    arch = pd.read_parquet(
        str(ROOT / "exploration/artifacts/shared/swe_archetype_labels.parquet")
    )
    dfm = df.merge(arch[["uid", "archetype_name"]], on="uid", how="inner")
    rows = []
    for period in ["2024", "2026"]:
        dfp = dfm[dfm["period2"] == period].copy()
        for arch_name in dfp["archetype_name"].value_counts().head(15).index:
            sub = dfp[dfp["archetype_name"] == arch_name].copy()
            sub = sub[sub["seniority_3level"].isin(["junior", "senior"])]
            n_j = (sub["seniority_3level"] == "junior").sum()
            n_s = (sub["seniority_3level"] == "senior").sum()
            if n_j < 10 or n_s < 10:
                rows.append(
                    {
                        "period": period,
                        "archetype_name": arch_name,
                        "n_junior": int(n_j),
                        "n_senior": int(n_s),
                        "auc_mean": float("nan"),
                        "note": "underpowered",
                    }
                )
                continue
            Xmat, feat_names = build_feature_matrix(sub, features)
            y = (sub["seniority_3level"].values == "senior").astype(int)
            mean_auc, _ = auc_cv(Xmat.values, y)
            rows.append(
                {
                    "period": period,
                    "archetype_name": arch_name,
                    "n_junior": int(n_j),
                    "n_senior": int(n_s),
                    "auc_mean": mean_auc,
                    "note": "",
                }
            )
    return pd.DataFrame(rows)


# -------------------- Feature heatmap


def feature_heatmap(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Mean feature profile per seniority × period (long format)."""
    dfw = df.copy()
    dfw["yoe_min_years_llm"] = dfw["yoe_eff"].where(
        dfw["yoe_eff"].notna(), dfw["yoe_min_years_llm"]
    )
    rows = []
    for period in ["2024", "2026"]:
        for lev in ["entry", "associate", "mid-senior", "director"]:
            sub = dfw[(dfw["period2"] == period) & (dfw["seniority_final"] == lev)]
            if len(sub) < 10:
                continue
            for f in features:
                rows.append(
                    {
                        "period": period,
                        "seniority_final": lev,
                        "feature": f,
                        "n": int(sub[f].notna().sum()),
                        "mean": float(sub[f].dropna().mean()) if sub[f].notna().any() else float("nan"),
                        "std": float(sub[f].dropna().std()) if sub[f].notna().any() else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def centroid_similarity_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Full pairwise cosine similarity of seniority × period centroids (structured features)."""
    dfw = df.copy()
    dfw["yoe_min_years_llm"] = dfw["yoe_eff"].where(
        dfw["yoe_eff"].notna(), dfw["yoe_min_years_llm"]
    )
    # z-standardize across pool
    base = dfw[features].apply(pd.to_numeric, errors="coerce")
    means = base.mean()
    stds = base.std().replace(0, 1)
    std = (base - means) / stds
    dfw = dfw.copy()
    for f in features:
        dfw[f + "_z"] = std[f]

    groups = []
    for period in ["2024", "2026"]:
        for lev in ["entry", "associate", "mid-senior", "director"]:
            sub = dfw[(dfw["period2"] == period) & (dfw["seniority_final"] == lev)]
            if len(sub) >= 10:
                c = sub[[f + "_z" for f in features]].mean().values
                groups.append((f"{lev}_{period}", c, len(sub)))

    # cosine
    rows = []
    for i, (name_i, c_i, n_i) in enumerate(groups):
        ni = np.linalg.norm(c_i) + 1e-12
        for j, (name_j, c_j, n_j) in enumerate(groups):
            nj = np.linalg.norm(c_j) + 1e-12
            cos = float(np.dot(c_i, c_j) / (ni * nj))
            rows.append(
                {"group_i": name_i, "group_j": name_j, "n_i": n_i, "n_j": n_j, "cosine": cos}
            )
    return pd.DataFrame(rows)


# -------------------- Main


def main() -> None:
    print("[T20] Loading merged dataframe ...", flush=True)
    df = load_merged()
    print(f"[T20] total rows: {len(df)}", flush=True)
    print(df["seniority_final"].value_counts(dropna=False).to_string(), flush=True)

    # Restrict to seniority_final != 'unknown'
    df_known = df[df["seniority_final"] != "unknown"].copy()
    print(f"[T20] seniority_final known rows: {len(df_known)}", flush=True)

    # 1) AUC boundary analysis
    print("\n[T20] Step 1 — Boundary AUC + feature importance ...", flush=True)
    auc_df, imp_df = run_boundary_auc(df_known, FEATURES)
    auc_df.to_csv(OUT / "auc_by_boundary_period.csv", index=False)
    imp_df.to_csv(OUT / "feature_importance_full.csv", index=False)
    print(auc_df.to_string(index=False), flush=True)

    # Top-5 discriminating features per boundary × period
    top5 = (
        imp_df.assign(abscoef=imp_df["coef_std"].abs())
        .sort_values(["period", "boundary", "abscoef"], ascending=[True, True, False])
        .groupby(["period", "boundary"])
        .head(5)
        .reset_index(drop=True)
    )
    top5.to_csv(OUT / "feature_importance_top5_by_boundary_period.csv", index=False)
    print("\n[T20] Top-5 features per boundary × period (by |coef_std|):", flush=True)
    print(top5.to_string(index=False), flush=True)

    # Sensitivity: aggregator excluded
    print("\n[T20] Sensitivity — aggregator excluded ...", flush=True)
    auc_agg = run_auc_aggregator_sensitivity(df_known, FEATURES)
    auc_agg.to_csv(OUT / "auc_sensitivity_aggregator.csv", index=False)
    print(auc_agg.to_string(index=False), flush=True)

    # Sensitivity: YOE dropped
    print("\n[T20] Sensitivity — YOE dropped ...", flush=True)
    auc_noyoe = run_auc_no_yoe(df_known, FEATURES)
    auc_noyoe.to_csv(OUT / "auc_sensitivity_no_yoe.csv", index=False)
    print(auc_noyoe.to_string(index=False), flush=True)

    # 2) Δgap + attribution
    print("\n[T20] Step 2 — Gap evolution + attribution ...", flush=True)
    gap_df = gap_evolution_table(df, FEATURES)
    gap_df.to_csv(OUT / "gap_evolution.csv", index=False)
    # Display primary panel
    print("\n[T20] Gap evolution — primary (J3 vs S4):", flush=True)
    print(gap_df[gap_df["panel_pair"] == "J3 vs S4"][
        ["feature", "gap_2024", "gap_2026", "delta_gap",
         "delta_M_senior", "delta_M_junior", "attribution_senior"]
    ].to_string(index=False), flush=True)

    # Within-2024 SNR
    print("\n[T20] Within-2024 SNR (arshkon vs asaniczka) for features ...", flush=True)
    snr_df = within_2024_snr(df_known, FEATURES)
    snr_df.to_csv(OUT / "within_2024_snr.csv", index=False)
    print(snr_df.to_string(index=False), flush=True)

    # 3) YOE × period interaction
    print("\n[T20] Step 3 — YOE × period interaction OLS ...", flush=True)
    outcomes = [
        "ai_binary",
        "requirement_breadth",
        "scope_density",
        "mgmt_strong_density",
        "tech_count",
        "credential_stack_depth",
    ]
    ypi = yoe_period_interaction(df, outcomes)
    ypi.to_csv(OUT / "yoe_period_interaction.csv", index=False)
    print(ypi[
        ["outcome", "n", "yoe_coef", "period_2026_coef",
         "yoe_x_period_coef", "yoe_x_period_ci_low", "yoe_x_period_ci_high", "yoe_x_period_p", "r2"]
    ].to_string(index=False), flush=True)

    # 4) Missing middle
    print("\n[T20] Step 4 — Missing middle (associate ↔ neighbors) ...", flush=True)
    mm = missing_middle(df_known, FEATURES)
    mm.to_csv(OUT / "missing_middle.csv", index=False)
    print(mm.to_string(index=False), flush=True)

    # 5) Domain-stratified AUC
    print("\n[T20] Step 5 — Domain-stratified AUC ...", flush=True)
    domain_df = domain_stratified_auc(df_known, FEATURES)
    domain_df.to_csv(OUT / "domain_stratified_auc.csv", index=False)
    print(domain_df.to_string(index=False), flush=True)

    # 6) Feature heatmap + centroid similarity
    print("\n[T20] Step 6 — Feature heatmap + centroid cosine ...", flush=True)
    fh = feature_heatmap(df_known, FEATURES)
    fh.to_csv(OUT / "feature_heatmap_long.csv", index=False)
    cs = centroid_similarity_matrix(df_known, FEATURES)
    cs.to_csv(OUT / "centroid_similarity_matrix.csv", index=False)
    print("\nCentroid cosine matrix (top 5 rows):", flush=True)
    print(cs.head(20).to_string(index=False), flush=True)

    # Write metadata
    meta = {
        "features": FEATURES,
        "boundary_pairs": BOUNDARY_PAIRS,
        "n_rows_total": int(len(df)),
        "n_rows_known_seniority": int(len(df_known)),
        "random_state": RNG,
        "cv_method": "StratifiedKFold(n_splits=5, shuffle=True)",
        "model": "LogisticRegression(L2, C=1.0, solver=lbfgs)",
    }
    with open(OUT / "metadata.json", "w") as fp:
        json.dump(meta, fp, indent=2, default=str)

    print("\n[T20] DONE. Outputs in", OUT, flush=True)


if __name__ == "__main__":
    main()
