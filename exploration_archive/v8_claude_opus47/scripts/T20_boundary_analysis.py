"""
T20 Step 2-5: Boundary discriminability analysis.

For each adjacent seniority boundary, train L2 logistic regression and
compute AUC (5-fold stratified CV). Repeat for 2024 and 2026 separately.

Step 2: train per boundary x period
Step 3: compare AUCs
Step 4: top 5 feature coefficients (standardized)
Step 5: "missing middle" — feature centroid distances associate vs entry/mid-senior
Step 7: full heatmap of mean features per seniority x period

Inputs:
  - exploration/artifacts/T20/T20_features.parquet
Outputs:
  - exploration/tables/T20/T20_boundary_auc.csv
  - exploration/tables/T20/T20_feature_coefficients.csv
  - exploration/tables/T20/T20_missing_middle.csv
  - exploration/tables/T20/T20_feature_centroids.csv
  - exploration/figures/T20/T20_boundary_auc.png
  - exploration/figures/T20/T20_feature_heatmap.png
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path("/home/jihgaboot/gabor/job-research")
IN = ROOT / "exploration/artifacts/T20/T20_features.parquet"
TBL = ROOT / "exploration/tables/T20"
FIG = ROOT / "exploration/figures/T20"
TBL.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# Load
print(f"Loading {IN} ...")
df = pq.read_table(IN).to_pandas()

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

LEVELS_ORDER = ["entry", "associate", "mid-senior", "director"]


def fit_boundary(sub: pd.DataFrame, pos_label: str, neg_label: str) -> dict:
    """Train LR, return cv AUC mean/std and standardized coefficients."""
    y = (sub["seniority_final"] == pos_label).astype(int).values
    X = sub[FEATURE_COLS].values.astype(float)
    n_pos = int(y.sum())
    n_neg = int(len(y) - y.sum())
    if n_pos < 10 or n_neg < 10:
        return {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auc_mean": None,
            "auc_std": None,
            "coef_per_feature": {f: None for f in FEATURE_COLS},
            "status": "insufficient_n",
        }

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    from sklearn.metrics import roc_auc_score
    for tr, te in skf.split(Xs, y):
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=500)
        clf.fit(Xs[tr], y[tr])
        probs = clf.predict_proba(Xs[te])[:, 1]
        aucs.append(roc_auc_score(y[te], probs))

    # Coefficients from full fit
    clf_full = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=500)
    clf_full.fit(Xs, y)
    coefs = dict(zip(FEATURE_COLS, clf_full.coef_[0].tolist()))

    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "coef_per_feature": coefs,
        "status": "ok",
    }


def run_boundary_analysis(
    df_in: pd.DataFrame, label: str, agg_excl: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run boundary x period; return AUC and coefficient frames."""
    rows_auc = []
    rows_coef = []
    for period in ["2024", "2026"]:
        for pos, neg in BOUNDARIES:
            sub = df_in[
                (df_in["period_bucket"] == period)
                & (df_in["seniority_final"].isin([pos, neg]))
            ].copy()
            if agg_excl:
                sub = sub[~sub["is_aggregator"]]
            boundary_label = f"{neg}_vs_{pos}"
            res = fit_boundary(sub, pos, neg)
            rows_auc.append(
                {
                    "label": label,
                    "period": period,
                    "boundary": boundary_label,
                    "n_neg_" + neg: res["n_neg"],
                    "n_pos_" + pos: res["n_pos"],
                    "n_total": res["n_neg"] + res["n_pos"],
                    "auc_mean": res["auc_mean"],
                    "auc_std": res["auc_std"],
                    "status": res["status"],
                }
            )
            if res["status"] == "ok":
                for f, c in res["coef_per_feature"].items():
                    rows_coef.append(
                        {
                            "label": label,
                            "period": period,
                            "boundary": boundary_label,
                            "feature": f,
                            "coef_standardized": c,
                            "abs_coef": abs(c),
                        }
                    )
    auc_df = pd.DataFrame(rows_auc)
    coef_df = pd.DataFrame(rows_coef)
    return auc_df, coef_df


# Run primary (all) and aggregator-excluded
auc_primary, coef_primary = run_boundary_analysis(df, "primary", agg_excl=False)
auc_noagg, coef_noagg = run_boundary_analysis(df, "no_aggregator", agg_excl=True)

# T30-panel sensitivity: J3 (yoe<=2) vs S4 (yoe>=5) boundary
df_panel = df.copy()
df_panel["seniority_panel"] = "unknown"
mask_j3 = df_panel["yoe_numeric"] <= 2
mask_s4 = df_panel["yoe_numeric"] >= 5
df_panel.loc[mask_j3, "seniority_panel"] = "j3_junior"
df_panel.loc[mask_s4, "seniority_panel"] = "s4_senior"
# Run simple J3 vs S4 discriminability
rows_panel = []
from sklearn.metrics import roc_auc_score
for period in ["2024", "2026"]:
    sub = df_panel[
        (df_panel["period_bucket"] == period)
        & (df_panel["seniority_panel"].isin(["j3_junior", "s4_senior"]))
    ].copy()
    # drop yoe_numeric from features since we defined the split on it
    feat_panel = [f for f in FEATURE_COLS if f != "yoe_numeric"]
    y = (sub["seniority_panel"] == "s4_senior").astype(int).values
    X = sub[feat_panel].values.astype(float)
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        continue
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(Xs, y):
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=500)
        clf.fit(Xs[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xs[te])[:, 1]))
    rows_panel.append(
        {
            "label": "J3_vs_S4_yoe",
            "period": period,
            "boundary": "j3_junior_vs_s4_senior_yoe_panel",
            "n_pos_s4": int(y.sum()),
            "n_neg_j3": int(len(y) - y.sum()),
            "n_total": int(len(y)),
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "status": "ok",
        }
    )

auc_panel = pd.DataFrame(rows_panel)

# Concatenate and save
auc_all = pd.concat([auc_primary, auc_noagg, auc_panel], ignore_index=True)
auc_all.to_csv(TBL / "T20_boundary_auc.csv", index=False)
print(f"wrote {TBL / 'T20_boundary_auc.csv'}")

coef_all = pd.concat([coef_primary, coef_noagg], ignore_index=True)
# Rank top 5 per (label, period, boundary)
coef_all["rank"] = (
    coef_all.groupby(["label", "period", "boundary"])["abs_coef"]
    .rank(method="dense", ascending=False)
    .astype(int)
)
coef_all.to_csv(TBL / "T20_feature_coefficients.csv", index=False)
print(f"wrote {TBL / 'T20_feature_coefficients.csv'}")

# ---------------------------------------------------------------------------
# Step 5: "Missing middle" analysis - centroid distances in feature space
# ---------------------------------------------------------------------------
# Compute per-seniority-level centroid (standardized features) per period
print("Computing feature centroids ...")
scaler = StandardScaler()
Xs = scaler.fit_transform(df[FEATURE_COLS].values.astype(float))
df_scaled = df.copy()
for i, f in enumerate(FEATURE_COLS):
    df_scaled[f"{f}_z"] = Xs[:, i]

centroids = (
    df_scaled.groupby(["period_bucket", "seniority_final"])[
        [f"{f}_z" for f in FEATURE_COLS]
    ]
    .mean()
    .reset_index()
)
centroids.to_csv(TBL / "T20_feature_centroids.csv", index=False)
print(f"wrote {TBL / 'T20_feature_centroids.csv'}")


def centroid_vec(cent_df: pd.DataFrame, period: str, sen: str) -> np.ndarray | None:
    row = cent_df[
        (cent_df["period_bucket"] == period) & (cent_df["seniority_final"] == sen)
    ]
    if len(row) == 0:
        return None
    return row[[f"{f}_z" for f in FEATURE_COLS]].values[0]


# Missing-middle: associate's distance to entry vs mid-senior per period
rows_mm = []
for period in ["2024", "2026"]:
    c_entry = centroid_vec(centroids, period, "entry")
    c_assoc = centroid_vec(centroids, period, "associate")
    c_midsr = centroid_vec(centroids, period, "mid-senior")
    c_dir = centroid_vec(centroids, period, "director")
    if any(v is None for v in (c_entry, c_assoc, c_midsr)):
        continue
    d_assoc_entry = float(np.linalg.norm(c_assoc - c_entry))
    d_assoc_midsr = float(np.linalg.norm(c_assoc - c_midsr))
    d_entry_midsr = float(np.linalg.norm(c_entry - c_midsr))
    d_midsr_dir = float(np.linalg.norm(c_midsr - c_dir)) if c_dir is not None else None
    ratio = d_assoc_entry / d_assoc_midsr if d_assoc_midsr > 0 else None
    rows_mm.append(
        {
            "period": period,
            "d_associate_to_entry": d_assoc_entry,
            "d_associate_to_midsenior": d_assoc_midsr,
            "ratio_assoc_to_entry_over_midsenior": ratio,
            "d_entry_to_midsenior": d_entry_midsr,
            "d_midsenior_to_director": d_midsr_dir,
        }
    )
mm = pd.DataFrame(rows_mm)
mm.to_csv(TBL / "T20_missing_middle.csv", index=False)
print(f"wrote {TBL / 'T20_missing_middle.csv'}")
print(mm.to_string())

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# Fig 1: AUC comparison bar chart across boundaries x periods (primary)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
plot_df = auc_primary.pivot(index="boundary", columns="period", values="auc_mean")
plot_df = plot_df.loc[[f"{n}_vs_{p}" for p, n in BOUNDARIES]]  # preserve order
x = np.arange(len(plot_df))
w = 0.35
ax.bar(x - w / 2, plot_df["2024"].values, w, label="2024")
ax.bar(x + w / 2, plot_df["2026"].values, w, label="2026")
ax.set_xticks(x)
ax.set_xticklabels(plot_df.index, rotation=20)
ax.set_ylim(0.5, 1.0)
ax.axhline(0.5, color="grey", lw=0.8)
ax.set_ylabel("5-fold CV AUC")
ax.set_title("T20: Seniority boundary discriminability (primary spec, all rows)")
for i, b in enumerate(plot_df.index):
    for off, per in [(-w / 2, "2024"), (w / 2, "2026")]:
        v = plot_df.loc[b, per]
        if pd.notna(v):
            ax.text(i + off, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "T20_boundary_auc.png", dpi=150)
plt.close()
print(f"wrote {FIG / 'T20_boundary_auc.png'}")

# Fig 2: Feature heatmap (raw means) per seniority x period
feature_means = (
    df.groupby(["period_bucket", "seniority_final"])[FEATURE_COLS]
    .mean()
    .reset_index()
)
hm = feature_means.set_index(["seniority_final", "period_bucket"]).unstack("period_bucket")
# Normalize per feature (z-score across all cells) for heatmap legibility
hm_z = hm.copy()
for f in FEATURE_COLS:
    sub = hm_z[f]
    mu = sub.values.mean()
    sd = sub.values.std() if sub.values.std() > 0 else 1.0
    hm_z[f] = (sub - mu) / sd
# Collapse into a 2D plot: rows = seniority_period, cols = feature
hm_stack = feature_means.set_index(["seniority_final", "period_bucket"]).reindex(
    [
        ("entry", "2024"),
        ("entry", "2026"),
        ("associate", "2024"),
        ("associate", "2026"),
        ("mid-senior", "2024"),
        ("mid-senior", "2026"),
        ("director", "2024"),
        ("director", "2026"),
    ]
)
# Z-score per feature across all cells for comparable color scale
hm_stack_z = hm_stack.copy()
for f in FEATURE_COLS:
    col = hm_stack[f]
    mu = col.mean()
    sd = col.std() if col.std() > 0 else 1.0
    hm_stack_z[f] = (col - mu) / sd

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    hm_stack_z,
    annot=hm_stack.round(2),
    fmt="",
    cmap="RdBu_r",
    center=0,
    cbar_kws={"label": "z-score (color) | raw mean (annot)"},
    ax=ax,
)
ax.set_title("T20: Feature profile per seniority × period")
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(FIG / "T20_feature_heatmap.png", dpi=150)
plt.close()
print(f"wrote {FIG / 'T20_feature_heatmap.png'}")

# Fig 3: Feature coefficient horizontal bar per boundary
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
boundaries_ordered = [f"{n}_vs_{p}" for p, n in BOUNDARIES]
for i, b in enumerate(boundaries_ordered):
    sub = coef_primary[coef_primary["boundary"] == b].copy()
    if len(sub) == 0:
        axes[i].set_title(f"{b}\n(insufficient n)")
        continue
    pvt = sub.pivot(index="feature", columns="period", values="coef_standardized")
    pvt = pvt.reindex(FEATURE_COLS)
    y = np.arange(len(pvt))
    w = 0.35
    axes[i].barh(y - w / 2, pvt.get("2024"), w, label="2024")
    axes[i].barh(y + w / 2, pvt.get("2026"), w, label="2026")
    axes[i].set_yticks(y)
    axes[i].set_yticklabels(pvt.index)
    axes[i].axvline(0, color="k", lw=0.6)
    axes[i].set_title(f"Boundary: {b}")
    axes[i].set_xlabel("Standardized coefficient (higher → pos class)")
    if i == 0:
        axes[i].legend()
plt.tight_layout()
plt.savefig(FIG / "T20_boundary_coefficients.png", dpi=150)
plt.close()
print(f"wrote {FIG / 'T20_boundary_coefficients.png'}")

# Print key tables
print("\nPrimary AUC:")
print(auc_primary.to_string())
print("\nNo-aggregator AUC:")
print(auc_noagg.to_string())
print("\nPanel J3 vs S4 AUC:")
print(auc_panel.to_string())
