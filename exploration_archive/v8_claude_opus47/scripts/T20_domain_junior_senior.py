"""
T20 Step 6 (powered version): Junior vs Senior boundary per archetype.

Since the adjacent-level boundaries have thin cells within per-archetype
subsets (the archetype sample is 8,000 rows), we run a binary J2 vs S1
boundary per archetype x period. This still answers the spec's core
question: does domain affect boundary sharpness and its trajectory?

J2 = entry + associate  vs  S1 = mid-senior + director

Inputs:
  - exploration/artifacts/T20/T20_features.parquet
  - exploration/artifacts/shared/swe_archetype_labels.parquet
Outputs:
  - exploration/tables/T20/T20_domain_js_boundary_auc.csv
  - exploration/figures/T20/T20_domain_js_boundary_auc.png
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/home/jihgaboot/gabor/job-research")
FEATURES = ROOT / "exploration/artifacts/T20/T20_features.parquet"
ARCHETYPES = ROOT / "exploration/artifacts/shared/swe_archetype_labels.parquet"
TBL = ROOT / "exploration/tables/T20"
FIG = ROOT / "exploration/figures/T20"

df = pq.read_table(FEATURES).to_pandas()
arch = pq.read_table(ARCHETYPES).to_pandas()
df = df.merge(arch[["uid", "archetype_name"]], on="uid", how="left")

# J2 / S1 bucket
def js_bucket(x: str) -> str:
    if x in ("entry", "associate"):
        return "junior"
    if x in ("mid-senior", "director"):
        return "senior"
    return "other"


df["js"] = df["seniority_final"].apply(js_bucket)

TOP_ARCHETYPES = [
    "generic_software_engineer",
    "ai_ml_engineering",
    "frontend_react",
    "java_spring_backend",
    "systems_engineering",
    "data_engineering",
    "cloud_devops",
]

FEATURE_COLS = [
    "yoe_numeric",
    "tech_count",
    "ai_mention",
    "org_scope_density",
    "management_density",
    "description_length_cleaned",
    "education_level",
]

MIN_N = 15
rows = []
coef_rows = []
for arch_name in TOP_ARCHETYPES:
    for period in ["2024", "2026"]:
        sub = df[
            (df["archetype_name"] == arch_name)
            & (df["period_bucket"] == period)
            & (df["js"].isin(["junior", "senior"]))
        ]
        y = (sub["js"] == "senior").astype(int).values
        n_pos = int(y.sum())
        n_neg = int(len(y) - y.sum())
        if n_pos < MIN_N or n_neg < MIN_N:
            rows.append(
                {
                    "archetype": arch_name,
                    "period": period,
                    "n_pos_senior": n_pos,
                    "n_neg_junior": n_neg,
                    "auc_mean": None,
                    "auc_std": None,
                    "status": "insufficient_n",
                }
            )
            continue
        X = sub[FEATURE_COLS].values.astype(float)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        n_splits = min(5, n_pos, n_neg)
        if n_splits < 2:
            rows.append(
                {
                    "archetype": arch_name,
                    "period": period,
                    "n_pos_senior": n_pos,
                    "n_neg_junior": n_neg,
                    "auc_mean": None,
                    "auc_std": None,
                    "status": "insufficient_n",
                }
            )
            continue
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        aucs = []
        for tr, te in skf.split(Xs, y):
            clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=500)
            clf.fit(Xs[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(Xs[te])[:, 1]))
        # Full-fit coef for top-feature interpretation
        clf_full = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=500)
        clf_full.fit(Xs, y)
        for f, c in zip(FEATURE_COLS, clf_full.coef_[0]):
            coef_rows.append(
                {
                    "archetype": arch_name,
                    "period": period,
                    "feature": f,
                    "coef_z": float(c),
                    "abs_coef": abs(float(c)),
                }
            )
        rows.append(
            {
                "archetype": arch_name,
                "period": period,
                "n_pos_senior": n_pos,
                "n_neg_junior": n_neg,
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
                "status": "ok",
            }
        )

out = pd.DataFrame(rows)
out.to_csv(TBL / "T20_domain_js_boundary_auc.csv", index=False)
print(f"wrote {TBL / 'T20_domain_js_boundary_auc.csv'}")
print(out.to_string())

coef_out = pd.DataFrame(coef_rows)
coef_out.to_csv(TBL / "T20_domain_js_boundary_coefficients.csv", index=False)
print(f"wrote {TBL / 'T20_domain_js_boundary_coefficients.csv'}")

# Plot
fig, ax = plt.subplots(figsize=(11, 6))
pvt = out.pivot(index="archetype", columns="period", values="auc_mean")
pvt = pvt.reindex([a for a in TOP_ARCHETYPES if a in pvt.index])
x = np.arange(len(pvt))
w = 0.35
for i, per in enumerate(["2024", "2026"]):
    offset = -w / 2 if per == "2024" else w / 2
    if per not in pvt.columns:
        continue
    values = pvt[per].values
    ax.bar(x + offset, values, w, label=per)
    for xi, v in zip(x, values):
        if pd.notna(v):
            ax.text(xi + offset, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
ax.axhline(0.5, color="grey", lw=0.8)
ax.set_ylim(0.5, 1.0)
ax.set_xticks(x)
ax.set_xticklabels([a[:20] for a in pvt.index], rotation=25, ha="right")
ax.set_ylabel("CV AUC (junior vs senior)")
ax.set_title("T20: Domain-stratified J2 vs S1 boundary AUC")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "T20_domain_js_boundary_auc.png", dpi=150)
plt.close()
print(f"wrote {FIG / 'T20_domain_js_boundary_auc.png'}")
