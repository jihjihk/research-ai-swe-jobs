"""
T20 Step 6: Domain-stratified boundary analysis.

For each of the top 5 archetypes (by volume), run boundary discriminability
and compare 2024 vs 2026. Small archetypes may be under-powered -> skip any
where n_pos or n_neg < 20 per (period, boundary).

Inputs:
  - exploration/artifacts/T20/T20_features.parquet
  - exploration/artifacts/shared/swe_archetype_labels.parquet
Outputs:
  - exploration/tables/T20/T20_domain_boundary_auc.csv
  - exploration/figures/T20/T20_domain_boundary_auc.png
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
print(f"features with archetype: {df['archetype_name'].notna().sum():,}")

# Top 5 by volume
TOP5 = [
    "generic_software_engineer",
    "ai_ml_engineering",
    "frontend_react",
    "java_spring_backend",
    "systems_engineering",
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

BOUNDARIES = [
    ("entry", "associate"),
    ("associate", "mid-senior"),
    ("mid-senior", "director"),
]

MIN_N_PER_CLASS = 20
rows = []
for arch_name in TOP5:
    sub_arch = df[df["archetype_name"] == arch_name]
    print(f"\n{arch_name}: n={len(sub_arch):,}")
    for period in ["2024", "2026"]:
        for pos, neg in BOUNDARIES:
            sub = sub_arch[
                (sub_arch["period_bucket"] == period)
                & (sub_arch["seniority_final"].isin([pos, neg]))
            ]
            y = (sub["seniority_final"] == pos).astype(int).values
            n_pos = int(y.sum())
            n_neg = int(len(y) - y.sum())
            if n_pos < MIN_N_PER_CLASS or n_neg < MIN_N_PER_CLASS:
                rows.append(
                    {
                        "archetype": arch_name,
                        "period": period,
                        "boundary": f"{neg}_vs_{pos}",
                        "n_pos": n_pos,
                        "n_neg": n_neg,
                        "auc_mean": None,
                        "auc_std": None,
                        "status": "insufficient_n",
                    }
                )
                continue
            X = sub[FEATURE_COLS].values.astype(float)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            skf = StratifiedKFold(n_splits=min(5, n_pos, n_neg), shuffle=True, random_state=42)
            aucs = []
            for tr, te in skf.split(Xs, y):
                clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=500)
                clf.fit(Xs[tr], y[tr])
                aucs.append(roc_auc_score(y[te], clf.predict_proba(Xs[te])[:, 1]))
            rows.append(
                {
                    "archetype": arch_name,
                    "period": period,
                    "boundary": f"{neg}_vs_{pos}",
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "auc_mean": float(np.mean(aucs)),
                    "auc_std": float(np.std(aucs)),
                    "status": "ok",
                }
            )

out = pd.DataFrame(rows)
out.to_csv(TBL / "T20_domain_boundary_auc.csv", index=False)
print(f"\nwrote {TBL / 'T20_domain_boundary_auc.csv'}")
print(out.to_string())

# Visual: grid of boundaries × archetype, AUC by period
fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
for i, (pos, neg) in enumerate(BOUNDARIES):
    b = f"{neg}_vs_{pos}"
    sub = out[out["boundary"] == b].copy()
    pvt = sub.pivot(index="archetype", columns="period", values="auc_mean")
    pvt = pvt.reindex(TOP5)
    x = np.arange(len(pvt))
    w = 0.35
    axes[i].bar(x - w / 2, pvt.get("2024"), w, label="2024")
    axes[i].bar(x + w / 2, pvt.get("2026"), w, label="2026")
    axes[i].set_xticks(x)
    axes[i].set_xticklabels([a[:16] for a in pvt.index], rotation=25, ha="right")
    axes[i].axhline(0.5, color="grey", lw=0.8)
    axes[i].set_ylim(0.4, 1.0)
    axes[i].set_title(f"Boundary: {b}")
    if i == 0:
        axes[i].set_ylabel("CV AUC")
        axes[i].legend()
plt.tight_layout()
plt.savefig(FIG / "T20_domain_boundary_auc.png", dpi=150)
plt.close()
print(f"wrote {FIG / 'T20_domain_boundary_auc.png'}")
