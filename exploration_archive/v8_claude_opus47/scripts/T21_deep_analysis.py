"""
T21 Steps 2-9: Scatter, sub-archetypes, AI interaction, director content,
staff-title profiling, cross-seniority comparison, domain stratification.

Uses existing T21_densities.parquet. Adds title_normalized from unified.parquet.
Uses swe_archetype_labels.parquet for domain stratification.

Inputs:
  - exploration/artifacts/T21/T21_densities.parquet
  - exploration/artifacts/shared/swe_archetype_labels.parquet
  - exploration/artifacts/shared/swe_cleaned_text.parquet
  - exploration/artifacts/shared/swe_embeddings.npy + swe_embedding_index.parquet
  - data/unified.parquet (for title_normalized)

Outputs (in exploration/tables/T21/ and exploration/figures/T21/):
  - T21_cross_seniority_mentor.csv
  - T21_staff_title_profile.csv
  - T21_ai_interaction.csv
  - T21_director_content.csv
  - T21_subarchetypes.csv + T21_subarchetype_profiles.csv
  - T21_domain_stratified.csv
  - Figures: scatter densities, sub-archetypes, AI interaction, cross-seniority
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/home/jihgaboot/gabor/job-research")
DENS = ROOT / "exploration/artifacts/T21/T21_densities.parquet"
ARCH = ROOT / "exploration/artifacts/shared/swe_archetype_labels.parquet"
UNIFIED = ROOT / "data/unified.parquet"
CLEANED = ROOT / "exploration/artifacts/shared/swe_cleaned_text.parquet"
TBL = ROOT / "exploration/tables/T21"
FIG = ROOT / "exploration/figures/T21"
TBL.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load densities
# ---------------------------------------------------------------------------
print("Loading T21_densities.parquet ...")
d = pq.read_table(DENS).to_pandas()
print(f"  {len(d):,} rows")

# AI mention pattern (V1-refined strict)
AI_PATTERN = re.compile(
    r"\b("
    r"copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|"
    r"llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|"
    r"vector database|pinecone|huggingface|hugging face"
    r")\b",
    re.IGNORECASE,
)

# We need description_cleaned for AI detection — load and merge
print("Loading description_cleaned for AI detection ...")
clean = pq.read_table(
    CLEANED, columns=["uid", "description_cleaned"]
).to_pandas()
d = d.merge(clean, on="uid", how="left")

d["ai_mention"] = d["description_cleaned"].apply(
    lambda t: 1 if (t and AI_PATTERN.search(t)) else 0
).astype("int64")
print(f"  AI-mention share overall: {d['ai_mention'].mean():.3f}")

# ---------------------------------------------------------------------------
# Load title_normalized for staff-title profiling (Step 6)
# ---------------------------------------------------------------------------
print("Loading unified.parquet titles ...")
titles = pq.read_table(UNIFIED, columns=["uid", "title_normalized", "title"]).to_pandas()
d = d.merge(titles, on="uid", how="left")
print(f"  merged titles: {d['title_normalized'].notna().sum():,}")

# ---------------------------------------------------------------------------
# Load archetypes for Step 8
# ---------------------------------------------------------------------------
print("Loading archetype labels ...")
arch = pq.read_table(ARCH).to_pandas()
d = d.merge(arch[["uid", "archetype_name"]], on="uid", how="left")
print(f"  archetype non-null: {d['archetype_name'].notna().sum():,}")

# Define seniority subset helpers
SENIOR = d[d["seniority_final"].isin(["mid-senior", "director"])].copy()
ARSH_2024 = (d["source"] == "kaggle_arshkon") & (d["period_bucket"] == "2024")
SCR_2026 = (d["source"] == "scraped") & (d["period_bucket"] == "2026")

# ---------------------------------------------------------------------------
# STEP 7: Cross-seniority mentor comparison
# ---------------------------------------------------------------------------
# V1-refined mentor rate at J2 (junior = entry+associate) vs S1 (senior = mid-senior+director)
# at entry, associate, mid-senior, director separately per period.
# If mentor rose at every level, weakens "senior archetype shift toward mentoring" to corpus-wide.
print("\n[Step 7] Cross-seniority mentor comparison ...")

MENTOR_PAT = re.compile(r"\bmentor\w*", re.IGNORECASE)
d["mentor_count"] = d["description_cleaned"].apply(
    lambda t: len(MENTOR_PAT.findall(t)) if t else 0
)
d["mentor_density"] = (d["mentor_count"] / d["desc_len"].clip(lower=1) * 1000).astype(float)
d["mentor_binary"] = (d["mentor_count"] > 0).astype(int)

cross_rows = []
for period in ["2024", "2026"]:
    for sen in ["entry", "associate", "mid-senior", "director"]:
        sub = d[(d["period_bucket"] == period) & (d["seniority_final"] == sen)]
        if len(sub) == 0:
            continue
        cross_rows.append(
            {
                "period": period,
                "seniority": sen,
                "n": len(sub),
                "mentor_binary_share": float(sub["mentor_binary"].mean()),
                "mentor_density_mean": float(sub["mentor_density"].mean()),
                "mgmt_binary_share": float(sub["mgmt_binary"].mean()),
                "mgmt_density_mean": float(sub["mgmt_density"].mean()),
                "subset": "all",
            }
        )
# Also arshkon-only 2024 baseline
for sen in ["entry", "associate", "mid-senior", "director"]:
    sub = d[ARSH_2024 & (d["seniority_final"] == sen)]
    if len(sub) == 0:
        continue
    cross_rows.append(
        {
            "period": "2024",
            "seniority": sen,
            "n": len(sub),
            "mentor_binary_share": float(sub["mentor_binary"].mean()),
            "mentor_density_mean": float(sub["mentor_density"].mean()),
            "mgmt_binary_share": float(sub["mgmt_binary"].mean()),
            "mgmt_density_mean": float(sub["mgmt_density"].mean()),
            "subset": "arshkon_only_2024",
        }
    )
# 2026 scraped only (same as all since 2026 is scraped-only)
cross_df = pd.DataFrame(cross_rows)
cross_df.to_csv(TBL / "T21_cross_seniority_mentor.csv", index=False)
print(cross_df.to_string())

# Compute rate-ratios: 2026/2024 mentor rate per seniority
all_subset = cross_df[cross_df["subset"] == "all"]
pvt = all_subset.pivot(index="seniority", columns="period", values="mentor_binary_share")
pvt["ratio_2026_over_2024"] = pvt["2026"] / pvt["2024"].clip(lower=1e-6)
pvt = pvt.reindex(["entry", "associate", "mid-senior", "director"])
pvt.to_csv(TBL / "T21_cross_seniority_mentor_ratio.csv")
print("\nMentor rate ratios 2026/2024:")
print(pvt.to_string())

# ---------------------------------------------------------------------------
# STEP 6: Staff-title profiling
# ---------------------------------------------------------------------------
# "staff" title doubled 2.6% to 6.3% (T10). What language profile do 2026
# staff-titled postings have?
print("\n[Step 6] Staff-title profiling ...")

# detect staff-title (staff engineer, staff software engineer, etc.)
STAFF_TITLE_PAT = re.compile(r"\bstaff\b", re.IGNORECASE)

def has_staff_title(t):
    if not isinstance(t, str):
        return False
    return bool(STAFF_TITLE_PAT.search(t))


d["staff_title"] = d["title_normalized"].apply(has_staff_title) | d["title"].apply(has_staff_title)
staff_rows = []

for period in ["2024", "2026"]:
    for senior_group in ["all_senior", "mid-senior_only"]:
        if senior_group == "all_senior":
            base = d[(d["period_bucket"] == period) & (d["seniority_final"].isin(["mid-senior", "director"]))]
        else:
            base = d[(d["period_bucket"] == period) & (d["seniority_final"] == "mid-senior")]
        staff_sub = base[base["staff_title"]]
        other_sub = base[~base["staff_title"]]
        if len(staff_sub) < 5:
            continue
        staff_rows.append(
            {
                "period": period,
                "senior_group": senior_group,
                "n_staff": len(staff_sub),
                "n_other": len(other_sub),
                "mgmt_binary_staff": float(staff_sub["mgmt_binary"].mean()),
                "mgmt_binary_other": float(other_sub["mgmt_binary"].mean()),
                "orch_strict_binary_staff": float(staff_sub["orch_strict_binary"].mean()),
                "orch_strict_binary_other": float(other_sub["orch_strict_binary"].mean()),
                "strat_strict_binary_staff": float(staff_sub["strat_strict_binary"].mean()),
                "strat_strict_binary_other": float(other_sub["strat_strict_binary"].mean()),
                "strat_broad_binary_staff": float(staff_sub["strat_broad_binary"].mean()),
                "strat_broad_binary_other": float(other_sub["strat_broad_binary"].mean()),
                "ai_mention_staff": float(staff_sub["ai_mention"].mean()),
                "ai_mention_other": float(other_sub["ai_mention"].mean()),
                "mentor_binary_staff": float(staff_sub["mentor_binary"].mean()),
                "mentor_binary_other": float(other_sub["mentor_binary"].mean()),
                "desc_len_staff": float(staff_sub["desc_len"].mean()),
                "desc_len_other": float(other_sub["desc_len"].mean()),
            }
        )

staff_df = pd.DataFrame(staff_rows)
staff_df.to_csv(TBL / "T21_staff_title_profile.csv", index=False)
print(staff_df.to_string())

# ---------------------------------------------------------------------------
# STEP 5: Director-specific content
# ---------------------------------------------------------------------------
print("\n[Step 5] Director content profiling ...")

# Compare director vs mid-senior per period for each profile density
director_rows = []
for period in ["2024", "2026"]:
    midsr = d[(d["period_bucket"] == period) & (d["seniority_final"] == "mid-senior")]
    dirc = d[(d["period_bucket"] == period) & (d["seniority_final"] == "director")]
    if len(dirc) == 0:
        continue
    director_rows.append(
        {
            "period": period,
            "n_director": len(dirc),
            "n_midsenior": len(midsr),
            "mgmt_binary_dir": float(dirc["mgmt_binary"].mean()),
            "mgmt_binary_midsr": float(midsr["mgmt_binary"].mean()),
            "mgmt_ratio_dir_over_midsr": (
                float(dirc["mgmt_binary"].mean()) / float(midsr["mgmt_binary"].mean())
                if midsr["mgmt_binary"].mean() > 0 else None
            ),
            "orch_strict_binary_dir": float(dirc["orch_strict_binary"].mean()),
            "orch_strict_binary_midsr": float(midsr["orch_strict_binary"].mean()),
            "strat_strict_binary_dir": float(dirc["strat_strict_binary"].mean()),
            "strat_strict_binary_midsr": float(midsr["strat_strict_binary"].mean()),
            "strat_broad_binary_dir": float(dirc["strat_broad_binary"].mean()),
            "strat_broad_binary_midsr": float(midsr["strat_broad_binary"].mean()),
            "ai_mention_dir": float(dirc["ai_mention"].mean()),
            "ai_mention_midsr": float(midsr["ai_mention"].mean()),
            "mentor_binary_dir": float(dirc["mentor_binary"].mean()),
            "mentor_binary_midsr": float(midsr["mentor_binary"].mean()),
            "desc_len_dir": float(dirc["desc_len"].mean()),
            "desc_len_midsr": float(midsr["desc_len"].mean()),
        }
    )
director_df = pd.DataFrame(director_rows)
director_df.to_csv(TBL / "T21_director_content.csv", index=False)
print(director_df.to_string())

# ---------------------------------------------------------------------------
# STEP 4: AI interaction — senior + AI-mentioning vs non-AI
# ---------------------------------------------------------------------------
print("\n[Step 4] AI interaction in senior postings ...")
ai_rows = []
for period in ["2024", "2026"]:
    base = d[
        (d["period_bucket"] == period) & (d["seniority_final"].isin(["mid-senior", "director"]))
    ]
    ai_sub = base[base["ai_mention"] == 1]
    noai_sub = base[base["ai_mention"] == 0]
    ai_rows.append(
        {
            "period": period,
            "n_ai": len(ai_sub),
            "n_noai": len(noai_sub),
            "share_ai": len(ai_sub) / max(1, len(base)),
            "mgmt_density_ai": float(ai_sub["mgmt_density"].mean()) if len(ai_sub) else None,
            "mgmt_density_noai": float(noai_sub["mgmt_density"].mean()) if len(noai_sub) else None,
            "mgmt_binary_ai": float(ai_sub["mgmt_binary"].mean()) if len(ai_sub) else None,
            "mgmt_binary_noai": float(noai_sub["mgmt_binary"].mean()) if len(noai_sub) else None,
            "orch_strict_density_ai": float(ai_sub["orch_strict_density"].mean()) if len(ai_sub) else None,
            "orch_strict_density_noai": float(noai_sub["orch_strict_density"].mean()) if len(noai_sub) else None,
            "strat_strict_density_ai": float(ai_sub["strat_strict_density"].mean()) if len(ai_sub) else None,
            "strat_strict_density_noai": float(noai_sub["strat_strict_density"].mean()) if len(noai_sub) else None,
            "strat_broad_density_ai": float(ai_sub["strat_broad_density"].mean()) if len(ai_sub) else None,
            "strat_broad_density_noai": float(noai_sub["strat_broad_density"].mean()) if len(noai_sub) else None,
        }
    )
ai_df = pd.DataFrame(ai_rows)
ai_df.to_csv(TBL / "T21_ai_interaction.csv", index=False)
print(ai_df.to_string())

# ---------------------------------------------------------------------------
# STEP 2: 2D scatter of densities per period (senior only, arshkon 2024 + scraped 2026)
# ---------------------------------------------------------------------------
print("\n[Step 2] Density scatter visualization ...")
# Use mgmt_density vs orch_strict_density — color by strat_strict_density
# Senior subset per period, with source anchoring
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
for i, period in enumerate(["2024", "2026"]):
    if period == "2024":
        sub = d[ARSH_2024 & d["seniority_final"].isin(["mid-senior", "director"])]
    else:
        sub = d[SCR_2026 & d["seniority_final"].isin(["mid-senior", "director"])]
    if len(sub) > 2000:
        sub = sub.sample(2000, random_state=42)
    sc = axes[i].scatter(
        sub["mgmt_density"],
        sub["orch_strict_density"],
        c=sub["strat_strict_density"].clip(upper=np.quantile(d["strat_strict_density"], 0.99)),
        cmap="viridis",
        s=10,
        alpha=0.5,
    )
    axes[i].set_title(f"Senior {period} (n={len(sub)})")
    axes[i].set_xlabel("Management density (per 1K chars)")
    if i == 0:
        axes[i].set_ylabel("Orchestration-strict density")
    axes[i].set_xlim(0, min(10, np.quantile(d["mgmt_density"], 0.99)))
    axes[i].set_ylim(0, min(30, np.quantile(d["orch_strict_density"], 0.99)))
    plt.colorbar(sc, ax=axes[i], label="strat-strict density")
plt.tight_layout()
plt.savefig(FIG / "T21_density_scatter.png", dpi=150)
plt.close()
print(f"wrote {FIG / 'T21_density_scatter.png'}")

# ---------------------------------------------------------------------------
# STEP 3: Senior sub-archetypes (k-means, k=4)
# ---------------------------------------------------------------------------
print("\n[Step 3] Senior sub-archetypes via k-means ...")

# Restrict to senior (mid-senior+director) arshkon 2024 + scraped 2026 for fair comparison
# Use the three density features + AI + desc_len
SENIOR_SUB = d[
    d["seniority_final"].isin(["mid-senior", "director"])
    & ((ARSH_2024) | (SCR_2026))
].copy()

feature_cols = [
    "mgmt_density",
    "orch_strict_density",
    "strat_broad_density",
    "ai_mention",
]
X = SENIOR_SUB[feature_cols].fillna(0).values.astype(float)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

km = KMeans(n_clusters=4, random_state=42, n_init=10)
SENIOR_SUB["cluster"] = km.fit_predict(Xs)

# Profile each cluster
profile = (
    SENIOR_SUB.groupby("cluster")
    .agg(
        n=("uid", "count"),
        mgmt_density_mean=("mgmt_density", "mean"),
        orch_strict_density_mean=("orch_strict_density", "mean"),
        strat_broad_density_mean=("strat_broad_density", "mean"),
        ai_mention_share=("ai_mention", "mean"),
        mentor_binary_share=("mentor_binary", "mean"),
        share_director=("seniority_final", lambda s: (s == "director").mean()),
    )
    .reset_index()
)

# Period share per cluster
period_share = (
    SENIOR_SUB.groupby(["cluster", "period_bucket"])
    .size()
    .unstack(fill_value=0)
)
period_share["share_2026"] = period_share["2026"] / (period_share["2024"] + period_share["2026"])
period_share = period_share.reset_index()
profile = profile.merge(period_share, on="cluster")

# Name clusters by dominant features (rule-based)
def name_cluster(row):
    m = row["mgmt_density_mean"]
    o = row["orch_strict_density_mean"]
    s = row["strat_broad_density_mean"]
    ai = row["ai_mention_share"]
    # High on all → strategic_leader; high orch only → orchestrator; high mgmt only → manager
    parts = []
    if m > profile["mgmt_density_mean"].median():
        parts.append("mgmt")
    if o > profile["orch_strict_density_mean"].median():
        parts.append("orch")
    if s > profile["strat_broad_density_mean"].median():
        parts.append("strat")
    if ai > profile["ai_mention_share"].median():
        parts.append("ai")
    return "_".join(parts) if parts else "baseline_ic"


profile["cluster_name"] = profile.apply(name_cluster, axis=1)
profile = profile.sort_values("n", ascending=False)
profile.to_csv(TBL / "T21_subarchetype_profiles.csv", index=False)
print(profile.to_string())

# Save cluster assignments
SENIOR_SUB[["uid", "cluster", "period_bucket", "seniority_final", "source"]].to_csv(
    TBL / "T21_subarchetypes.csv", index=False
)

# Plot cluster centroids (3D-like 2D projection)
fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.tab10(np.arange(len(profile)))
for idx, row in profile.iterrows():
    size = 50 + row["n"] / 5
    ai_alpha = 0.4 + 0.6 * row["ai_mention_share"]
    ax.scatter(
        row["mgmt_density_mean"],
        row["orch_strict_density_mean"],
        s=size,
        c=[colors[idx]],
        label=f"{row['cluster_name']} (n={int(row['n'])}, 2026:{row['share_2026']:.1%})",
        alpha=ai_alpha,
        edgecolor="k",
    )
ax.set_xlabel("Mean management density (per 1K chars)")
ax.set_ylabel("Mean orchestration-strict density (per 1K chars)")
ax.set_title("T21: Senior sub-archetypes (k=4). Size=n, alpha=AI share")
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig(FIG / "T21_subarchetypes.png", dpi=150)
plt.close()
print(f"wrote {FIG / 'T21_subarchetypes.png'}")

# ---------------------------------------------------------------------------
# STEP 8: Domain stratification — is the shift uniform or ML/AI-concentrated?
# ---------------------------------------------------------------------------
print("\n[Step 8] Domain stratification of density shifts ...")

TOP_ARCHS = [
    "generic_software_engineer",
    "ai_ml_engineering",
    "frontend_react",
    "java_spring_backend",
    "systems_engineering",
    "data_engineering",
    "cloud_devops",
]

dom_rows = []
for archn in TOP_ARCHS:
    for period in ["2024", "2026"]:
        sub = d[
            (d["archetype_name"] == archn)
            & (d["period_bucket"] == period)
            & (d["seniority_final"].isin(["mid-senior", "director"]))
        ]
        if len(sub) < 20:
            continue
        dom_rows.append(
            {
                "archetype": archn,
                "period": period,
                "n": len(sub),
                "mgmt_binary_share": float(sub["mgmt_binary"].mean()),
                "mgmt_density_mean": float(sub["mgmt_density"].mean()),
                "orch_strict_binary_share": float(sub["orch_strict_binary"].mean()),
                "orch_strict_density_mean": float(sub["orch_strict_density"].mean()),
                "strat_broad_binary_share": float(sub["strat_broad_binary"].mean()),
                "strat_broad_density_mean": float(sub["strat_broad_density"].mean()),
                "ai_mention_share": float(sub["ai_mention"].mean()),
                "mentor_binary_share": float(sub["mentor_binary"].mean()),
            }
        )
dom_df = pd.DataFrame(dom_rows)
dom_df.to_csv(TBL / "T21_domain_stratified.csv", index=False)
print(dom_df.to_string())

# Compute delta 2026 - 2024 per archetype
delta_rows = []
for archn in TOP_ARCHS:
    sub = dom_df[dom_df["archetype"] == archn]
    if len(sub) < 2:
        continue
    r24 = sub[sub["period"] == "2024"]
    r26 = sub[sub["period"] == "2026"]
    if len(r24) == 0 or len(r26) == 0:
        continue
    delta_rows.append(
        {
            "archetype": archn,
            "n_2024": int(r24["n"].values[0]),
            "n_2026": int(r26["n"].values[0]),
            "delta_mgmt_binary": float(r26["mgmt_binary_share"].values[0] - r24["mgmt_binary_share"].values[0]),
            "delta_orch_strict_binary": float(r26["orch_strict_binary_share"].values[0] - r24["orch_strict_binary_share"].values[0]),
            "delta_strat_broad_binary": float(r26["strat_broad_binary_share"].values[0] - r24["strat_broad_binary_share"].values[0]),
            "delta_ai_mention": float(r26["ai_mention_share"].values[0] - r24["ai_mention_share"].values[0]),
            "delta_mentor_binary": float(r26["mentor_binary_share"].values[0] - r24["mentor_binary_share"].values[0]),
        }
    )
delta_df = pd.DataFrame(delta_rows)
delta_df.to_csv(TBL / "T21_domain_deltas.csv", index=False)
print("\nDomain deltas (2026-2024):")
print(delta_df.to_string())

print("\nDONE.")
