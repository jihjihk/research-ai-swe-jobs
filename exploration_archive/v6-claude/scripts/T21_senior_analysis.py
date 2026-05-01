"""T21 — Senior role evolution deep dive.

Requires `validated_mgmt_patterns.json` from `T21_validate_patterns.py`.

Computes per-posting density (per 1K chars) of the three validated profiles
(people_mgmt, tech_orch, strategic) for senior postings (mid-senior + director).

Analyses:
  1. Density by profile × period × seniority (primary result).
  2. 2D scatter (mgmt vs orch, strategic vs orch) colored by period.
  3. Senior sub-archetype clustering (k=4 k-means on z-scored densities).
  4. AI × senior interaction.
  5. Director-specific deep dive.
  6. Senior title compression analysis (source of "senior" drop).
  7. Cross-seniority management comparison (entry vs mid-senior vs director).
  8. Sensitivity (a): aggregator exclusion.

Inputs:
  - exploration/artifacts/shared/validated_mgmt_patterns.json
  - data/unified.parquet (raw `description_core_llm` for density computation)
  - exploration/artifacts/shared/swe_cleaned_text.parquet
  - exploration/artifacts/shared/swe_tech_matrix.parquet
  - exploration/artifacts/shared/swe_archetype_labels.parquet

Outputs:
  - tables/T21/density_by_profile_period_seniority.csv
  - tables/T21/senior_subarchetypes.csv
  - tables/T21/ai_senior_interaction.csv
  - tables/T21/director_deep_dive.csv
  - tables/T21/title_compression.csv
  - tables/T21/cross_seniority_mgmt.csv
  - tables/T21/sens_a_density_no_aggregator.csv
  - figures/T21/density_profile_shift.png
  - figures/T21/subarchetype_distribution.png
  - figures/T21/title_compression.png
  - figures/T21/ai_senior_interaction.png
  - figures/T21/cross_seniority_mgmt.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/jihgaboot/gabor/job-research")
SHARED = ROOT / "exploration/artifacts/shared"
T21_TABLES = ROOT / "exploration/tables/T21"
T21_FIGS = ROOT / "exploration/figures/T21"
T21_FIGS.mkdir(parents=True, exist_ok=True)

# --- Load validated patterns ----------------------------------------------

with (SHARED / "validated_mgmt_patterns.json").open() as f:
    PAT = json.load(f)

RE_PEOPLE = re.compile(PAT["profiles"]["people_mgmt"]["regex"], re.IGNORECASE)
RE_ORCH = re.compile(PAT["profiles"]["tech_orch"]["regex"], re.IGNORECASE)
RE_STRAT = re.compile(PAT["profiles"]["strategic"]["regex"], re.IGNORECASE)
print(f"[T21] Loaded validated patterns: people={len(PAT['profiles']['people_mgmt']['kept_patterns'])}, orch={len(PAT['profiles']['tech_orch']['kept_patterns'])}, strat={len(PAT['profiles']['strategic']['kept_patterns'])}")

# --- Load senior + entry postings ------------------------------------------

print("[T21] Loading postings from unified.parquet")
con = duckdb.connect()
q = """
SELECT u.uid,
       u.description_core_llm AS text,
       CASE WHEN substr(u.period, 1, 4) = '2024' THEN '2024'
            WHEN substr(u.period, 1, 4) = '2026' THEN '2026'
            ELSE substr(u.period, 1, 4) END AS period2,
       u.seniority_final,
       u.seniority_final_source,
       u.is_aggregator,
       u.title,
       u.yoe_extracted,
       u.swe_classification_tier
FROM 'data/unified.parquet' AS u
WHERE u.is_swe = true
  AND u.source_platform = 'linkedin'
  AND u.is_english = true
  AND u.date_flag = 'ok'
  AND u.llm_extraction_coverage = 'labeled'
  AND u.description_core_llm IS NOT NULL
  AND u.seniority_final IN ('entry', 'associate', 'mid-senior', 'director')
"""
df = con.execute(q).fetchdf()
df = df[df["period2"].isin(["2024", "2026"])].reset_index(drop=True)
print(f"[T21] Loaded {len(df):,} labeled LLM-text postings across 4 seniority levels")

# Merge AI mention from tech matrix
tm = pd.read_parquet(SHARED / "swe_tech_matrix.parquet")
AI_COLS = [
    "llm", "langchain", "langgraph", "rag", "vector_db", "pinecone", "chromadb",
    "huggingface", "openai_api", "claude_api", "prompt_engineering", "fine_tuning",
    "mcp", "agents_framework", "gpt", "transformer_arch", "embedding", "copilot",
    "cursor_tool", "chatgpt", "claude_tool", "gemini_tool", "codex_tool",
    "machine_learning", "deep_learning", "nlp", "computer_vision",
]
ai_cols_present = [c for c in AI_COLS if c in tm.columns]
tm_ai = pd.DataFrame({"uid": tm["uid"], "ai_mention": tm[ai_cols_present].any(axis=1).astype(int)})
df = df.merge(tm_ai, on="uid", how="left")
df["ai_mention"] = df["ai_mention"].fillna(0).astype(int)

# Merge archetype
arch = pd.read_parquet(SHARED / "swe_archetype_labels.parquet")
df = df.merge(arch, on="uid", how="left")

# --- Density computation ---------------------------------------------------

texts = df["text"].fillna("").tolist()
desc_len = np.array([len(t) for t in texts], dtype=float)
desc_len_k = np.maximum(desc_len, 200.0) / 1000.0

people_counts = np.array([len(RE_PEOPLE.findall(t)) for t in texts], dtype=float)
orch_counts = np.array([len(RE_ORCH.findall(t)) for t in texts], dtype=float)
strat_counts = np.array([len(RE_STRAT.findall(t)) for t in texts], dtype=float)

df["desc_len_chars"] = desc_len
df["people_density"] = people_counts / desc_len_k
df["orch_density"] = orch_counts / desc_len_k
df["strat_density"] = strat_counts / desc_len_k
df["people_any"] = (people_counts > 0).astype(int)
df["orch_any"] = (orch_counts > 0).astype(int)
df["strat_any"] = (strat_counts > 0).astype(int)

print("[T21] Densities computed")

# --- 1. Density by profile × period × seniority ---------------------------

sen_frame = df[df["seniority_final"].isin(["mid-senior", "director"])].copy()
agg = (
    sen_frame.groupby(["seniority_final", "period2"])
    .agg(
        n=("uid", "count"),
        people_density_mean=("people_density", "mean"),
        orch_density_mean=("orch_density", "mean"),
        strat_density_mean=("strat_density", "mean"),
        people_any_share=("people_any", "mean"),
        orch_any_share=("orch_any", "mean"),
        strat_any_share=("strat_any", "mean"),
        desc_len_mean=("desc_len_chars", "mean"),
    )
    .reset_index()
)
agg.to_csv(T21_TABLES / "density_by_profile_period_seniority.csv", index=False)
print(agg.to_string())

# --- 2. 2D scatter plot ----------------------------------------------------

rng = np.random.default_rng(7)
plot_sample = sen_frame[sen_frame["seniority_final"] == "mid-senior"].copy()
# subsample for visual clarity
idx24 = plot_sample.index[plot_sample["period2"] == "2024"]
idx26 = plot_sample.index[plot_sample["period2"] == "2026"]
take = min(2000, len(idx24), len(idx26))
sub_idx = np.concatenate([
    rng.choice(idx24, size=take, replace=False),
    rng.choice(idx26, size=take, replace=False),
])
plot_sub = plot_sample.loc[sub_idx]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (x, y, xl, yl) in zip(
    axes,
    [
        ("people_density", "orch_density", "People mgmt density (per 1K)", "Tech orch density (per 1K)"),
        ("strat_density", "orch_density", "Strategic density (per 1K)", "Tech orch density (per 1K)"),
    ],
):
    for period, color in [("2024", "#4C72B0"), ("2026", "#DD8452")]:
        sub = plot_sub[plot_sub["period2"] == period]
        ax.scatter(sub[x], sub[y], s=4, alpha=0.3, color=color, label=period)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.legend()
axes[0].set_title("T21 senior language profiles: people vs orchestration")
axes[1].set_title("T21 senior language profiles: strategic vs orchestration")
fig.tight_layout()
fig.savefig(T21_FIGS / "density_profile_shift.png", dpi=150)
plt.close(fig)

# --- 3. Senior sub-archetype clustering -----------------------------------

X_sen = sen_frame[["people_density", "orch_density", "strat_density"]].to_numpy()
scaler = StandardScaler().fit(X_sen)
X_sen_s = scaler.transform(X_sen)
km = KMeans(n_clusters=4, random_state=42, n_init=10)
sen_frame["sub_archetype"] = km.fit_predict(X_sen_s)

# Label clusters by which density is highest
center_raw = scaler.inverse_transform(km.cluster_centers_)
sub_labels = {}
for i, (p, o, s) in enumerate(center_raw):
    total = p + o + s
    if total < 0.3:
        sub_labels[i] = "generic (low all)"
    else:
        dom = max([(p, "people-manager"), (o, "tech-lead"), (s, "strategist")])[1]
        sub_labels[i] = dom
sen_frame["sub_archetype_label"] = sen_frame["sub_archetype"].map(sub_labels)

sub_agg = (
    sen_frame.groupby(["period2", "sub_archetype_label"])
    .agg(n=("uid", "count"))
    .reset_index()
)
sub_agg["share"] = sub_agg.groupby("period2")["n"].transform(lambda x: x / x.sum())

# Cluster centers
center_df = pd.DataFrame(center_raw, columns=["people_density", "orch_density", "strat_density"])
center_df["label"] = [sub_labels[i] for i in range(len(center_df))]
center_df.to_csv(T21_TABLES / "senior_subarchetype_centers.csv", index=False)

sub_agg.to_csv(T21_TABLES / "senior_subarchetypes.csv", index=False)
print("\n[Sub-archetypes]")
print(sub_agg.to_string())

# --- 4. AI × senior interaction -------------------------------------------

ai_int = (
    sen_frame.groupby(["period2", "ai_mention"])
    .agg(
        n=("uid", "count"),
        people_density=("people_density", "mean"),
        orch_density=("orch_density", "mean"),
        strat_density=("strat_density", "mean"),
    )
    .reset_index()
)
ai_int["ai_mention"] = ai_int["ai_mention"].map({0: "no_ai", 1: "ai"})
ai_int.to_csv(T21_TABLES / "ai_senior_interaction.csv", index=False)
print("\n[AI × senior]")
print(ai_int.to_string())

# --- 5. Director-specific deep dive ---------------------------------------

director = sen_frame[sen_frame["seniority_final"] == "director"]
midsen = sen_frame[sen_frame["seniority_final"] == "mid-senior"]
dir_rows = []
for period in ("2024", "2026"):
    d = director[director["period2"] == period]
    m = midsen[midsen["period2"] == period]
    dir_rows.append({
        "period": period, "n_director": len(d), "n_midsen": len(m),
        "dir_people": d["people_density"].mean(),
        "mid_people": m["people_density"].mean(),
        "dir_orch": d["orch_density"].mean(),
        "mid_orch": m["orch_density"].mean(),
        "dir_strat": d["strat_density"].mean(),
        "mid_strat": m["strat_density"].mean(),
    })
pd.DataFrame(dir_rows).to_csv(T21_TABLES / "director_deep_dive.csv", index=False)

# --- 6. Title compression analysis ----------------------------------------

print("\n[T21] Loading all SWE rows for title analysis")
q_titles = """
SELECT u.uid, u.title, u.seniority_final, u.seniority_final_source,
       CASE WHEN substr(u.period, 1, 4) = '2024' THEN '2024'
            WHEN substr(u.period, 1, 4) = '2026' THEN '2026'
            ELSE substr(u.period, 1, 4) END AS period2
FROM 'data/unified.parquet' AS u
WHERE u.is_swe = true
  AND u.source_platform = 'linkedin'
  AND u.is_english = true
  AND u.date_flag = 'ok'
"""
t_all = con.execute(q_titles).fetchdf()
t_all = t_all[t_all["period2"].isin(["2024", "2026"])].reset_index(drop=True)
t_all["title_l"] = t_all["title"].fillna("").str.lower()
t_all["has_senior"] = t_all["title_l"].str.contains(r"\bsenior\b|\bsr\.?\b", regex=True)
t_all["has_staff"] = t_all["title_l"].str.contains(r"\bstaff\b", regex=True)
t_all["has_principal"] = t_all["title_l"].str.contains(r"\bprincipal\b", regex=True)
t_all["has_lead"] = t_all["title_l"].str.contains(r"\blead\b", regex=True)

tc = (
    t_all.groupby("period2")
    .agg(
        n=("uid", "count"),
        senior_share=("has_senior", "mean"),
        staff_share=("has_staff", "mean"),
        principal_share=("has_principal", "mean"),
        lead_share=("has_lead", "mean"),
    )
    .reset_index()
)
tc.to_csv(T21_TABLES / "title_compression.csv", index=False)
print(tc.to_string())

# seniority_final_source distribution for senior-title vs staff-title rows
sf_senior = t_all[t_all["has_senior"]].groupby(["period2", "seniority_final_source"]).size().reset_index(name="n")
sf_staff = t_all[t_all["has_staff"]].groupby(["period2", "seniority_final_source"]).size().reset_index(name="n")
sf_senior["title_group"] = "senior"
sf_staff["title_group"] = "staff"
sf = pd.concat([sf_senior, sf_staff], ignore_index=True)
sf.to_csv(T21_TABLES / "title_compression_by_source.csv", index=False)

# Seniority_final breakdown for rows with 'senior' vs 'staff' in title
sf2_senior = t_all[t_all["has_senior"]].groupby(["period2", "seniority_final"]).size().reset_index(name="n")
sf2_senior["title_group"] = "senior"
sf2_staff = t_all[t_all["has_staff"]].groupby(["period2", "seniority_final"]).size().reset_index(name="n")
sf2_staff["title_group"] = "staff"
sf2 = pd.concat([sf2_senior, sf2_staff], ignore_index=True)
sf2.to_csv(T21_TABLES / "title_compression_vs_final.csv", index=False)

# Within-company: do SAME companies that posted "senior" now post "staff"?
# Join titles with company_name_canonical via the cleaned text frame
ct_frame = pd.read_parquet(SHARED / "swe_cleaned_text.parquet", columns=["uid", "company_name_canonical"])
t_all_co = t_all.merge(ct_frame, on="uid", how="left")
co_stats = (
    t_all_co.groupby(["company_name_canonical", "period2"])
    .agg(n_total=("uid", "count"),
         n_senior=("has_senior", "sum"),
         n_staff=("has_staff", "sum"))
    .reset_index()
)
co_stats["senior_share"] = co_stats["n_senior"] / co_stats["n_total"]
co_stats["staff_share"] = co_stats["n_staff"] / co_stats["n_total"]
# Pivot to find companies that appear in both periods with n>=10
co_pivot = co_stats.pivot(index="company_name_canonical", columns="period2", values=["senior_share", "staff_share", "n_total"])
co_pivot.columns = [f"{a}_{b}" for a, b in co_pivot.columns]
co_pivot = co_pivot.dropna(subset=["n_total_2024", "n_total_2026"])
co_pivot = co_pivot[(co_pivot["n_total_2024"] >= 10) & (co_pivot["n_total_2026"] >= 10)]
co_pivot["senior_delta"] = co_pivot["senior_share_2026"] - co_pivot["senior_share_2024"]
co_pivot["staff_delta"] = co_pivot["staff_share_2026"] - co_pivot["staff_share_2024"]
co_pivot.sort_values("senior_delta").to_csv(T21_TABLES / "overlap_company_senior_staff.csv")
print(f"\n[Overlap panel] companies with n≥10 in both periods: {len(co_pivot)}")
print(f"  Mean senior_delta: {co_pivot['senior_delta'].mean():+.3f}")
print(f"  Mean staff_delta:  {co_pivot['staff_delta'].mean():+.3f}")
print(f"  Companies where senior dropped: {(co_pivot['senior_delta'] < 0).sum()} / {len(co_pivot)}")
print(f"  Companies where staff rose:     {(co_pivot['staff_delta'] > 0).sum()} / {len(co_pivot)}")

# --- 7. Cross-seniority management comparison -----------------------------

cs_rows = []
for sen_level in ("entry", "mid-senior", "director"):
    for period in ("2024", "2026"):
        sub = df[(df["seniority_final"] == sen_level) & (df["period2"] == period)]
        if len(sub) < 30:
            continue
        cs_rows.append({
            "seniority": sen_level,
            "period": period,
            "n": len(sub),
            "people_density": sub["people_density"].mean(),
            "orch_density": sub["orch_density"].mean(),
            "strat_density": sub["strat_density"].mean(),
            "people_any": sub["people_any"].mean(),
        })
cs_df = pd.DataFrame(cs_rows)
cs_df.to_csv(T21_TABLES / "cross_seniority_mgmt.csv", index=False)
print("\n[Cross-seniority mgmt]")
print(cs_df.to_string())

# --- 8. Sensitivity (a): aggregator exclusion -----------------------------

noagg = sen_frame[~sen_frame["is_aggregator"].astype(bool)]
noagg_agg = (
    noagg.groupby(["seniority_final", "period2"])
    .agg(
        n=("uid", "count"),
        people_density=("people_density", "mean"),
        orch_density=("orch_density", "mean"),
        strat_density=("strat_density", "mean"),
    )
    .reset_index()
)
noagg_agg.to_csv(T21_TABLES / "sens_a_density_no_aggregator.csv", index=False)

# --- Figures --------------------------------------------------------------

# Density profile shift (senior only)
fig, ax = plt.subplots(figsize=(9, 5))
mid = agg[agg["seniority_final"] == "mid-senior"].set_index("period2")
profiles_plot = ["people_density_mean", "orch_density_mean", "strat_density_mean"]
labels = ["People mgmt", "Tech orch", "Strategic"]
x = np.arange(len(labels))
width = 0.35
ax.bar(x - width/2, mid.loc["2024", profiles_plot], width, label="2024", color="#4C72B0")
ax.bar(x + width/2, mid.loc["2026", profiles_plot], width, label="2026", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Density (per 1K chars, validated patterns)")
ax.set_title("T21 mid-senior language-profile density, 2024 vs 2026")
ax.legend()
fig.tight_layout()
fig.savefig(T21_FIGS / "density_profile_mean.png", dpi=150)
plt.close(fig)

# Sub-archetype distribution
fig, ax = plt.subplots(figsize=(8, 5))
sap = sub_agg.pivot(index="sub_archetype_label", columns="period2", values="share")
sap.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
ax.set_ylabel("Share of senior postings")
ax.set_title("T21 senior sub-archetype share by period")
plt.xticks(rotation=15)
ax.legend(title="Period")
fig.tight_layout()
fig.savefig(T21_FIGS / "subarchetype_distribution.png", dpi=150)
plt.close(fig)

# Title compression
fig, ax = plt.subplots(figsize=(9, 5))
tc_plot = tc.set_index("period2")[["senior_share", "staff_share", "principal_share", "lead_share"]]
tc_plot.T.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
ax.set_ylabel("Share of titles")
ax.set_title("T21 senior title compression")
plt.xticks(rotation=15)
fig.tight_layout()
fig.savefig(T21_FIGS / "title_compression.png", dpi=150)
plt.close(fig)

# AI × senior interaction
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, prof, lbl in zip(axes, ["people_density", "orch_density", "strat_density"], ["People mgmt", "Tech orch", "Strategic"]):
    ai_pivot = ai_int.pivot(index="period2", columns="ai_mention", values=prof)
    ai_pivot[["no_ai", "ai"]].plot(kind="bar", ax=ax, color=["#C0C0C0", "#D54B4B"])
    ax.set_title(lbl)
    ax.set_ylabel("density per 1K")
    plt.setp(ax.get_xticklabels(), rotation=0)
fig.suptitle("T21 AI × senior interaction: profile density")
fig.tight_layout()
fig.savefig(T21_FIGS / "ai_senior_interaction.png", dpi=150)
plt.close(fig)

# Cross-seniority management
fig, ax = plt.subplots(figsize=(9, 5))
for period, color in [("2024", "#4C72B0"), ("2026", "#DD8452")]:
    sub = cs_df[cs_df["period"] == period]
    ax.plot(sub["seniority"], sub["people_density"], marker="o", label=period, color=color)
ax.set_ylabel("People-management density (per 1K chars)")
ax.set_xlabel("Seniority")
ax.set_title("T21 people-management density by seniority and period")
ax.legend()
fig.tight_layout()
fig.savefig(T21_FIGS / "cross_seniority_mgmt.png", dpi=150)
plt.close(fig)

print("\n[T21] Done")
