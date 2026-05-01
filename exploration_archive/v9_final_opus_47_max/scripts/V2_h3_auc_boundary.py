"""V2 Phase A — H_w3_3: Independent re-derivation of T20 boundary AUC sharpening.

Claim (T20): AUC junior↔associate +0.093; associate↔mid-senior +0.150; mid-senior↔director −0.022
V2 independent approach: load T11_posting_features.parquet, train logistic regression with 5-fold CV
using the same feature set, compute AUC per boundary × period.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = str(ROOT / "data" / "unified.parquet")
T11 = str(ROOT / "exploration" / "artifacts" / "shared" / "T11_posting_features.parquet")
OUT = ROOT / "exploration" / "tables" / "V2"

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


def load():
    # T11 features joined with seniority_final from unified
    q = f"""
    SELECT t.uid, t.period, t.tech_count, t.ai_binary, t.scope_density, t.mgmt_strong_density,
           t.description_cleaned_length, t.education_level, t.requirement_breadth,
           t.credential_stack_depth, t.yoe_min_years_llm,
           u.seniority_final, u.source
    FROM read_parquet('{T11}') t
    JOIN read_parquet('{UNIFIED}') u ON t.uid = u.uid
    WHERE u.source_platform='linkedin' AND u.is_english=TRUE AND u.date_flag='ok' AND u.is_swe=TRUE
    """
    con = duckdb.connect()
    df = con.execute(q).df()
    con.close()
    # Period label
    df["period_era"] = np.where(df["source"] == "scraped", "2026", "2024")
    return df


def boundary_auc(df, lo_label, hi_label, era):
    sub = df[(df["seniority_final"].isin([lo_label, hi_label])) & (df["period_era"] == era)].copy()
    sub = sub.dropna(subset=FEATURES + ["seniority_final"])
    if len(sub) < 20:
        return None, len(sub)
    sub["y"] = (sub["seniority_final"] == hi_label).astype(int)
    n_lo = (sub["y"] == 0).sum()
    n_hi = (sub["y"] == 1).sum()
    if min(n_lo, n_hi) < 5:
        return None, len(sub)

    X = sub[FEATURES].to_numpy(dtype=float)
    # Median imputation per col before standardization
    med = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = med[j]

    y = sub["y"].to_numpy()
    k = min(5, min(n_lo, n_hi))
    if k < 2:
        return None, len(sub)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=20202020)
    aucs = []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[tr])
        Xv = scaler.transform(X[te])
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
        clf.fit(Xt, y[tr])
        s = clf.predict_proba(Xv)[:, 1]
        aucs.append(roc_auc_score(y[te], s))
    return float(np.mean(aucs)), len(sub)


def main():
    df = load()

    pairs = [
        ("entry", "associate", "entry_assoc"),
        ("associate", "mid-senior", "assoc_midsen"),
        ("mid-senior", "director", "midsen_dir"),
    ]

    out_rows = []
    for lo, hi, name in pairs:
        for era in ["2024", "2026"]:
            auc, n = boundary_auc(df, lo, hi, era)
            out_rows.append({"boundary": name, "era": era, "auc": auc, "n": n})

    df_out = pd.DataFrame(out_rows)
    pivot = df_out.pivot(index="boundary", columns="era", values="auc")
    pivot["delta"] = pivot["2026"] - pivot["2024"]
    print("\nH_w3_3 T20 AUC re-derivation (V2 independent):")
    print(df_out.to_string(index=False))
    print("\nPivot with delta:")
    print(pivot.to_string())

    pivot.to_csv(OUT / "H_w3_3_auc.csv")

    # Phase E (robustness) — panel variants J1/J2/J3/J4 sharpening using YOE-based splits as labels
    # Since T20 is based on seniority_final, the T30 panel check here would be alternate label definitions.
    # Approach: use J3 (yoe≤2) vs S4 (yoe≥5) as a 2-class boundary
    print("\nAlternative J3/S4 boundary AUC under YOE-based label (V2 robustness):")
    for era in ["2024", "2026"]:
        sub = df[(df["period_era"] == era) & df["yoe_min_years_llm"].notna()].copy()
        sub = sub[(sub["yoe_min_years_llm"] <= 2) | (sub["yoe_min_years_llm"] >= 5)]
        sub = sub.dropna(subset=[f for f in FEATURES if f != "yoe_min_years_llm"])
        sub["y"] = (sub["yoe_min_years_llm"] >= 5).astype(int)
        n_lo = (sub["y"] == 0).sum()
        n_hi = (sub["y"] == 1).sum()
        feats_noyoe = [f for f in FEATURES if f != "yoe_min_years_llm"]
        X = sub[feats_noyoe].to_numpy(dtype=float)
        med = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            X[np.isnan(X[:, j]), j] = med[j]
        y = sub["y"].to_numpy()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20202020)
        aucs = []
        for tr, te in skf.split(X, y):
            scaler = StandardScaler()
            Xt = scaler.fit_transform(X[tr])
            Xv = scaler.transform(X[te])
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
            clf.fit(Xt, y[tr])
            s = clf.predict_proba(Xv)[:, 1]
            aucs.append(roc_auc_score(y[te], s))
        print(f"  J3↔S4 era {era}: n={len(sub)}, n_J3={n_lo}, n_S4={n_hi}, AUC={np.mean(aucs):.3f}")


if __name__ == "__main__":
    main()
