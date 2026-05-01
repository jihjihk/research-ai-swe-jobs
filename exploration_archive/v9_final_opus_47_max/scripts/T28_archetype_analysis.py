"""T28 — Domain-stratified scope changes by archetype.

Steps:
1. Load T09 archetype labels; project missing labels via nearest-centroid embeddings.
2. Entry-share decomposition (within/between/interaction) by archetype × period.
3. Domain-stratified scope inflation (breadth_resid, tech_count, scope_density, AI, stack depth).
4. Junior vs senior content within each archetype.
5. Senior archetype shift by domain.
6. AI/ML archetype cross-validation.

Primary seniority:
- J3 = yoe_min_years_llm <= 2 (labeled rows only for YOE)
- S4 = yoe_min_years_llm >= 5
- J1/J2 = seniority_final == 'entry' / seniority_3level == 'junior' (label sensitivities)

Usage:
    ./.venv/bin/python exploration/scripts/T28_archetype_analysis.py
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
import duckdb
import pyarrow.parquet as pq

ROOT = "/home/jihgaboot/gabor/job-research"
UNI = f"{ROOT}/data/unified.parquet"
SHARED = f"{ROOT}/exploration/artifacts/shared"
OUT = f"{ROOT}/exploration/tables/T28"
os.makedirs(OUT, exist_ok=True)

SCOPE_SQL = (
    "is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok'"
)

# --------------------------------------------------------------------------
# Step 0: Load SWE corpus with all needed columns
# --------------------------------------------------------------------------
print("=" * 72)
print("Step 0: Load corpus")
print("=" * 72)

con = duckdb.connect()
corpus = con.execute(f"""
    SELECT uid, source, period, company_name_canonical, is_aggregator,
           seniority_final, seniority_3level, yoe_min_years_llm,
           llm_classification_coverage
    FROM read_parquet('{UNI}')
    WHERE {SCOPE_SQL}
""").fetchdf()

# Period flag: 2024 (arshkon/asaniczka) vs 2026 (scraped)
corpus["period_year"] = np.where(corpus["source"] == "scraped", "2026", "2024")
corpus["is_J3"] = (corpus["yoe_min_years_llm"] <= 2) & corpus["yoe_min_years_llm"].notna()
corpus["is_S4"] = (corpus["yoe_min_years_llm"] >= 5) & corpus["yoe_min_years_llm"].notna()
corpus["is_J1"] = corpus["seniority_final"] == "entry"
corpus["is_J2"] = corpus["seniority_3level"] == "junior"

print(f"Corpus rows: {len(corpus):,}")
print("Period x source:")
print(corpus.groupby(["period_year", "source"]).size())

# --------------------------------------------------------------------------
# Step 1: Load T09 archetype labels; project missing via nearest-centroid
# --------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Step 1: Archetype labels + nearest-centroid projection")
print("=" * 72)

arch = pq.read_table(f"{SHARED}/swe_archetype_labels.parquet").to_pandas()
print(f"T09 labeled: {len(arch):,}")

# Merge into corpus
corpus = corpus.merge(arch, on="uid", how="left")

# Embeddings for projection
emb = np.load(f"{SHARED}/swe_embeddings.npy")
emb_idx = pq.read_table(f"{SHARED}/swe_embedding_index.parquet").to_pandas()
emb_idx = emb_idx.rename(columns={"row_idx": "emb_row"})
print(f"Embeddings shape: {emb.shape}")

# Build uid -> embedding row
uid_to_row = dict(zip(emb_idx["uid"], emb_idx["emb_row"]))

# Compute centroid per archetype (only from T09 labeled rows with embedding)
arch_labeled = corpus[corpus["archetype"].notna()].copy()
arch_labeled["emb_row"] = arch_labeled["uid"].map(uid_to_row)
arch_labeled_with_emb = arch_labeled.dropna(subset=["emb_row"])
arch_labeled_with_emb["emb_row"] = arch_labeled_with_emb["emb_row"].astype(int)

print(f"T09 labels with embedding: {len(arch_labeled_with_emb):,}")

# Archetype centroids — for each archetype including -1 noise
centroids = {}
name_map = {}
for arch_id, grp in arch_labeled_with_emb.groupby("archetype"):
    rows = grp["emb_row"].values
    if len(rows) == 0:
        continue
    centroids[int(arch_id)] = emb[rows].mean(axis=0)
    name_map[int(arch_id)] = grp["archetype_name"].iloc[0]

arch_ids = sorted(centroids.keys())
print(f"Archetypes: {len(arch_ids)} (incl. noise -1)")

# L2-normalize centroids
C = np.vstack([centroids[a] for a in arch_ids])
C_norm = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)

# Project missing labels for rows that have embedding
need_proj_mask = corpus["archetype"].isna() & corpus["uid"].isin(uid_to_row)
n_need = need_proj_mask.sum()
print(f"Rows needing projection: {n_need:,}")

if n_need > 0:
    proj_rows = corpus.loc[need_proj_mask].copy()
    proj_rows["emb_row"] = proj_rows["uid"].map(uid_to_row).astype(int)
    proj_emb = emb[proj_rows["emb_row"].values]
    proj_emb_norm = proj_emb / (np.linalg.norm(proj_emb, axis=1, keepdims=True) + 1e-12)
    # Cosine sim: (N x 384) @ (384 x K) -> N x K
    sims = proj_emb_norm @ C_norm.T
    # Best archetype excluding -1 noise -> but we want the nearest incl noise
    # Gate 2 spec allows noise; we'll assign to closest non-noise too
    best = np.argmax(sims, axis=1)
    proj_ids = np.array([arch_ids[i] for i in best])
    proj_names = np.array([name_map[i] for i in proj_ids])
    # Also compute "primary" = nearest non-noise, used where noise-dominated projection is unhelpful
    nonneg_mask = np.array([a >= 0 for a in arch_ids], dtype=bool)
    if nonneg_mask.any():
        sims_nonnoise = sims[:, nonneg_mask]
        nonneg_ids = [arch_ids[i] for i, m in enumerate(nonneg_mask) if m]
        best_nn = np.argmax(sims_nonnoise, axis=1)
        proj_ids_primary = np.array([nonneg_ids[i] for i in best_nn])
        proj_names_primary = np.array([name_map[i] for i in proj_ids_primary])
    else:
        proj_ids_primary = proj_ids
        proj_names_primary = proj_names
    corpus.loc[need_proj_mask, "archetype"] = proj_ids
    corpus.loc[need_proj_mask, "archetype_name"] = proj_names
    corpus.loc[need_proj_mask, "archetype_primary"] = proj_ids_primary
    corpus.loc[need_proj_mask, "archetype_primary_name"] = proj_names_primary
    corpus.loc[need_proj_mask, "archetype_source"] = "projected"

# For T09-labeled rows, archetype_primary = archetype if !=-1 else nearest non-noise
labeled_mask = corpus["archetype_source"].isna() & corpus["archetype"].notna()
corpus.loc[labeled_mask, "archetype_source"] = "t09"
# For T09-labeled noise rows, additionally compute archetype_primary as nearest non-noise centroid
t09_noise = (corpus["archetype_source"] == "t09") & (corpus["archetype"] == -1)
t09_noise_with_emb = corpus[t09_noise & corpus["uid"].isin(uid_to_row)]
if len(t09_noise_with_emb) > 0:
    t09_noise_rows = t09_noise_with_emb["uid"].map(uid_to_row).astype(int).values
    t09_noise_emb = emb[t09_noise_rows]
    t09_noise_emb_norm = t09_noise_emb / (np.linalg.norm(t09_noise_emb, axis=1, keepdims=True) + 1e-12)
    nonneg_mask = np.array([a >= 0 for a in arch_ids], dtype=bool)
    nonneg_ids = [arch_ids[i] for i, m in enumerate(nonneg_mask) if m]
    sims_nn = t09_noise_emb_norm @ C_norm[nonneg_mask].T
    best_nn = np.argmax(sims_nn, axis=1)
    prim_ids = np.array([nonneg_ids[i] for i in best_nn])
    prim_names = np.array([name_map[i] for i in prim_ids])
    corpus.loc[t09_noise_with_emb.index, "archetype_primary"] = prim_ids
    corpus.loc[t09_noise_with_emb.index, "archetype_primary_name"] = prim_names

# For T09-labeled non-noise rows: archetype_primary = archetype
t09_clean = (corpus["archetype_source"] == "t09") & (corpus["archetype"] != -1)
corpus.loc[t09_clean, "archetype_primary"] = corpus.loc[t09_clean, "archetype"]
corpus.loc[t09_clean, "archetype_primary_name"] = corpus.loc[t09_clean, "archetype_name"]

# Rows without any embedding (raw-text-only scraped) — these have no archetype.
unassigned = corpus["archetype"].isna()
print(f"Rows without archetype (no embedding): {unassigned.sum():,}")
corpus.loc[unassigned, "archetype_source"] = "unassigned"
corpus["archetype"] = corpus["archetype"].astype("Int64")
corpus["archetype_primary"] = corpus["archetype_primary"].astype("Int64")

# Coverage by period
print("\nArchetype coverage by source x period:")
print(corpus.groupby(["source", "period", "archetype_source"]).size().unstack(fill_value=0))

# --------------------------------------------------------------------------
# Bucket archetypes into Gate 2 domain groups (reduce from 36+ to ~10)
# --------------------------------------------------------------------------
print("\n" + "-" * 60)
print("Archetype grouping into domains")
print("-" * 60)

def map_domain(name: str) -> str:
    if name is None or pd.isna(name):
        return "unknown"
    n = str(name).lower()
    # AI/ML cluster
    if any(k in n for k in ["model", "llm", "agent", "workflow", "perception", "driving"]):
        if "ml" in n or "models" in n or "llm" in n or "agent" in n or "perception" in n:
            return "ai_ml"
    if "agent" in n or "llm" in n or "models" in n:
        return "ai_ml"
    # Cloud / DevOps / SRE
    if any(k in n for k in ["kubernetes", "terraform", "cicd", "azure", "devops"]):
        return "cloud_devops"
    # Java / Backend
    if any(k in n for k in ["java/boot", "java/backend", "spring", "microservices", "rails", "backend/systems", "backend/javascript"]):
        return "backend"
    # Data engineering / pipelines
    if any(k in n for k in ["pipelines", "etl", "sql/etl"]):
        return "data_eng"
    # SQL/aspnet/MVC — legacy .NET stack
    if any(k in n for k in ["aspnet", "mvc", "sql/aspnet"]):
        return "dotnet_legacy"
    # Mobile
    if any(k in n for k in ["android", "kotlin", "swiftui", "ios"]):
        return "mobile"
    # QA/test
    if any(k in n for k in ["qa", "selenium", "manual"]):
        return "qa_test"
    # Clearance/defense
    if "clearance" in n or "export" in n or "routing" in n:
        return "clearance_defense"
    # Firmware / embedded / Linux systems
    if any(k in n for k in ["firmware", "ethernet", "linux", "embedded"]):
        return "embedded_firmware"
    # Intern
    if "intern" in n:
        return "intern_early"
    # Academic/research/PhD
    if "phd" in n or "degree_phd" in n:
        return "research_academic"
    # Generic software/requirements
    if "software" in n and ("requirements" in n or "systems" in n or "solutions" in n):
        return "generic_swe"
    # Customer/app
    if "customer" in n:
        return "customer_apps"
    # Scrum
    if "scrum" in n or "sprint" in n:
        return "process_scrum"
    # Frontend
    if "angular" in n or "javascript" in n or "typescript" in n or "features/backend/rails" in n:
        return "frontend_full"
    # Noise
    if "noise" in n or "outlier" in n:
        return "noise"
    return "other"

corpus["domain"] = corpus["archetype_primary_name"].apply(map_domain)
# Coarse domain: rely on primary (noise-resolved)
print("Domain distribution (primary, noise-resolved):")
print(corpus.groupby("domain").size().sort_values(ascending=False))

# Domain x period distribution
dom_period = corpus.groupby(["domain", "period_year"]).size().unstack(fill_value=0)
dom_period["pct_2024"] = dom_period["2024"] / dom_period["2024"].sum() * 100
dom_period["pct_2026"] = dom_period["2026"] / dom_period["2026"].sum() * 100
dom_period["delta_pp"] = dom_period["pct_2026"] - dom_period["pct_2024"]
dom_period = dom_period.sort_values("pct_2026", ascending=False)
print("\nDomain distribution by period (rows, % shares, delta pp):")
print(dom_period.round(2))
dom_period.to_csv(f"{OUT}/step1_domain_distribution.csv")

# Save enriched corpus for downstream T28 steps
corpus_cols = ["uid", "source", "period", "period_year", "company_name_canonical",
               "is_aggregator", "seniority_final", "seniority_3level",
               "yoe_min_years_llm", "is_J3", "is_S4", "is_J1", "is_J2",
               "archetype", "archetype_name", "archetype_primary",
               "archetype_primary_name", "archetype_source", "domain"]
corpus[corpus_cols].to_parquet(f"{OUT}/T28_corpus_with_archetype.parquet", index=False)
print(f"\nSaved enriched corpus: {len(corpus):,} rows")

# --------------------------------------------------------------------------
# Step 2: Entry-share decomposition (within/between/interaction) by archetype
# --------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Step 2: Entry-share decomposition by archetype x period")
print("=" * 72)


def decompose(df: pd.DataFrame, group: str, metric: str, period_col: str = "period_year") -> dict:
    """Shift-share: aggregate change = within + between + interaction.

    agg = sum over g of w_g * m_g
    Δagg = sum_g w_g^0 * Δm_g    [within]
         + sum_g Δw_g * m_g^0    [between]
         + sum_g Δw_g * Δm_g      [interaction]
    """
    # Drop rows where metric is NA
    df = df.dropna(subset=[metric])
    periods = sorted(df[period_col].dropna().unique())
    if len(periods) < 2:
        return {}
    p0, p1 = periods[0], periods[1]
    d0 = df[df[period_col] == p0]
    d1 = df[df[period_col] == p1]
    if len(d0) == 0 or len(d1) == 0:
        return {}
    # Shares (weights)
    w0 = d0.groupby(group).size() / len(d0)
    w1 = d1.groupby(group).size() / len(d1)
    # Means (metric per group)
    m0 = d0.groupby(group)[metric].mean()
    m1 = d1.groupby(group)[metric].mean()
    # Align
    groups = sorted(set(w0.index) | set(w1.index))
    w0 = w0.reindex(groups, fill_value=0)
    w1 = w1.reindex(groups, fill_value=0)
    m0 = m0.reindex(groups, fill_value=0)
    m1 = m1.reindex(groups, fill_value=0)
    agg0 = (w0 * m0).sum()
    agg1 = (w1 * m1).sum()
    delta = agg1 - agg0
    within = (w0 * (m1 - m0)).sum()
    between = ((w1 - w0) * m0).sum()
    interaction = ((w1 - w0) * (m1 - m0)).sum()
    # Per-group detail
    per_group = pd.DataFrame({
        "w0": w0, "w1": w1, "m0": m0, "m1": m1,
        "delta_w": w1 - w0, "delta_m": m1 - m0,
        "within_contrib": w0 * (m1 - m0),
        "between_contrib": (w1 - w0) * m0,
        "interaction_contrib": (w1 - w0) * (m1 - m0),
    })
    per_group = per_group.sort_values("within_contrib", ascending=False)
    return {
        "agg_2024": agg0,
        "agg_2026": agg1,
        "delta": delta,
        "within": within,
        "between": between,
        "interaction": interaction,
        "per_group": per_group,
    }


# J3 entry share by domain (exclude unassigned rows, rows without YOE)
j3_corpus = corpus[corpus["domain"] != "unknown"].dropna(subset=["yoe_min_years_llm"])
j3_corpus["is_J3_numeric"] = j3_corpus["is_J3"].astype(float)
print(f"J3 frame (rows with YOE + archetype): {len(j3_corpus):,}")

decomp_j3 = decompose(j3_corpus, "domain", "is_J3_numeric")
print(f"\nJ3 entry share 2024 → 2026: {decomp_j3['agg_2024']:.4f} → {decomp_j3['agg_2026']:.4f}")
print(f"  Total delta: {decomp_j3['delta']*100:.2f} pp")
print(f"  Within:      {decomp_j3['within']*100:.2f} pp")
print(f"  Between:     {decomp_j3['between']*100:.2f} pp")
print(f"  Interaction: {decomp_j3['interaction']*100:.2f} pp")

# Save decomposition table
dec_summary_rows = []
for senior_label, col, label_name in [
    ("J3 (YOE<=2, primary)", "is_J3_numeric", "is_J3_numeric"),
    ("J1 (seniority_final=entry)", "is_J1_numeric", "is_J1_numeric"),
    ("J2 (seniority_3level=junior)", "is_J2_numeric", "is_J2_numeric"),
    ("S4 (YOE>=5)", "is_S4_numeric", "is_S4_numeric"),
]:
    c = corpus[corpus["domain"] != "unknown"].copy()
    if senior_label.startswith("J3") or senior_label.startswith("S4"):
        c = c.dropna(subset=["yoe_min_years_llm"])
    c[col] = {"is_J3_numeric": c["is_J3"], "is_J1_numeric": c["is_J1"],
              "is_J2_numeric": c["is_J2"], "is_S4_numeric": c["is_S4"]}[col].astype(float)
    dec = decompose(c, "domain", col)
    if not dec:
        continue
    dec_summary_rows.append({
        "metric": senior_label,
        "value_2024": round(dec["agg_2024"], 4),
        "value_2026": round(dec["agg_2026"], 4),
        "delta_pp": round(dec["delta"] * 100, 2),
        "within_pp": round(dec["within"] * 100, 2),
        "between_pp": round(dec["between"] * 100, 2),
        "interaction_pp": round(dec["interaction"] * 100, 2),
        "n_2024": int((c["period_year"] == "2024").sum()),
        "n_2026": int((c["period_year"] == "2026").sum()),
    })
    # Per-group detail
    pg = dec["per_group"].copy()
    pg["metric"] = senior_label
    pg.to_csv(f"{OUT}/step2_decomp_{col}_by_domain.csv", index=True)

dec_summary = pd.DataFrame(dec_summary_rows)
dec_summary.to_csv(f"{OUT}/step2_entry_share_decomposition.csv", index=False)
print("\n=== Entry/seniority share decomposition ===")
print(dec_summary.to_string(index=False))

# Sensitivity: aggregator exclusion
print("\n-- Sensitivity: exclude aggregators --")
c_nonagg = corpus[(corpus["domain"] != "unknown") & (~corpus["is_aggregator"].fillna(False))].copy()
c_nonagg = c_nonagg.dropna(subset=["yoe_min_years_llm"])
c_nonagg["is_J3_numeric"] = c_nonagg["is_J3"].astype(float)
dec_nonagg = decompose(c_nonagg, "domain", "is_J3_numeric")
print(f"J3 delta pp: {dec_nonagg['delta']*100:.2f} | within: {dec_nonagg['within']*100:.2f} | "
      f"between: {dec_nonagg['between']*100:.2f} | interaction: {dec_nonagg['interaction']*100:.2f}")

# --------------------------------------------------------------------------
# Step 3: Domain-stratified scope inflation
# --------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Step 3: Domain-stratified scope inflation")
print("=" * 72)

t11 = pq.read_table(f"{SHARED}/T11_posting_features.parquet").to_pandas()
t11_cols = ["uid", "tech_count", "requirement_breadth", "requirement_breadth_resid",
            "credential_stack_depth", "credential_stack_depth_resid",
            "scope_density", "scope_density_resid", "ai_binary",
            "description_cleaned_length"]
t11 = t11[t11_cols]

# Merge on uid
work = corpus.merge(t11, on="uid", how="inner")
print(f"Merged with T11: {len(work):,} rows")

# Filter to domain in-scope (exclude unknown + unassigned)
work = work[(work["domain"] != "unknown") & work["archetype_primary_name"].notna()]

# Per-archetype scope changes
scope_rows = []
for dom, grp in work.groupby("domain"):
    for period_year, g in grp.groupby("period_year"):
        if len(g) < 10:
            continue
        scope_rows.append({
            "domain": dom,
            "period_year": period_year,
            "n": len(g),
            "breadth_resid_mean": g["requirement_breadth_resid"].mean(),
            "tech_count_mean": g["tech_count"].mean(),
            "scope_density_mean": g["scope_density"].mean(),
            "ai_binary_rate": g["ai_binary"].astype(float).mean(),
            "credential_stack_depth_mean": g["credential_stack_depth"].mean(),
            "desc_length_mean": g["description_cleaned_length"].mean(),
        })
scope_df = pd.DataFrame(scope_rows)

# Pivot for easier change computation
def piv(col):
    p = scope_df.pivot(index="domain", columns="period_year", values=col)
    p.columns = [f"{col}_{c}" for c in p.columns]
    return p

pivots = [piv(c) for c in ["breadth_resid_mean", "tech_count_mean", "scope_density_mean",
                             "ai_binary_rate", "credential_stack_depth_mean", "desc_length_mean", "n"]]
scope_wide = pd.concat(pivots, axis=1)
# Deltas
for base in ["breadth_resid_mean", "tech_count_mean", "scope_density_mean",
             "ai_binary_rate", "credential_stack_depth_mean", "desc_length_mean"]:
    scope_wide[f"{base}_delta"] = scope_wide[f"{base}_2026"] - scope_wide[f"{base}_2024"]

# Sort by tech_count delta descending
scope_wide = scope_wide.sort_values("tech_count_mean_delta", ascending=False)
scope_wide.to_csv(f"{OUT}/step3_scope_by_domain.csv")
print("\n=== Per-domain scope changes (sorted by tech_count delta) ===")
cols = [c for c in scope_wide.columns if c.endswith("_delta") or c.startswith("n_")]
print(scope_wide[cols].round(3).to_string())

# Decomposition of scope metrics
scope_metric_decomps = []
for metric in ["requirement_breadth_resid", "tech_count", "scope_density",
               "credential_stack_depth", "ai_binary"]:
    c = work.copy()
    if metric == "ai_binary":
        c["_metric"] = c["ai_binary"].astype(float)
    else:
        c["_metric"] = c[metric].astype(float)
    dec = decompose(c, "domain", "_metric")
    if not dec:
        continue
    scope_metric_decomps.append({
        "metric": metric,
        "value_2024": round(dec["agg_2024"], 4),
        "value_2026": round(dec["agg_2026"], 4),
        "delta": round(dec["delta"], 4),
        "within": round(dec["within"], 4),
        "between": round(dec["between"], 4),
        "interaction": round(dec["interaction"], 4),
    })
    pd.DataFrame([dec["per_group"]]) if False else None
    dec["per_group"].to_csv(f"{OUT}/step3_decomp_{metric}_by_domain.csv")

scope_dec_df = pd.DataFrame(scope_metric_decomps)
scope_dec_df.to_csv(f"{OUT}/step3_scope_decomposition.csv", index=False)
print("\n=== Scope metric decomposition across domains ===")
print(scope_dec_df.to_string(index=False))

# --------------------------------------------------------------------------
# Step 4: Junior vs Senior within each archetype
# --------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Step 4: Junior vs Senior within each archetype")
print("=" * 72)

# Build per-domain per-period per-tier (J3/S4) summary for breadth, AI, scope, mgmt(broad resid)
# Merge in mgmt_broad from T11
t11_mgmt = pq.read_table(f"{SHARED}/T11_posting_features.parquet").to_pandas()[
    ["uid", "mgmt_broad_density_resid"]]
work_js = work.merge(t11_mgmt, on="uid", how="left")

js_rows = []
for dom, grp in work_js.groupby("domain"):
    for period_year, g in grp.groupby("period_year"):
        for tier_name, mask in [("J3", g["is_J3"].fillna(False)),
                                  ("S4", g["is_S4"].fillna(False))]:
            sub = g[mask]
            if len(sub) < 20:
                continue
            js_rows.append({
                "domain": dom,
                "period_year": period_year,
                "tier": tier_name,
                "n": len(sub),
                "breadth_resid": sub["requirement_breadth_resid"].mean(),
                "ai_binary_rate": sub["ai_binary"].astype(float).mean(),
                "scope_density": sub["scope_density"].mean(),
                "mgmt_broad_density_resid": sub["mgmt_broad_density_resid"].mean(),
                "tech_count": sub["tech_count"].mean(),
                "credential_stack_depth": sub["credential_stack_depth"].mean(),
            })
js_df = pd.DataFrame(js_rows)
js_df.to_csv(f"{OUT}/step4_junior_senior_by_domain.csv", index=False)

# Build gap table: S4 - J3 by domain x period
gap_rows = []
for dom in js_df["domain"].unique():
    for period_year in ["2024", "2026"]:
        sub = js_df[(js_df["domain"] == dom) & (js_df["period_year"] == period_year)]
        j3 = sub[sub["tier"] == "J3"]
        s4 = sub[sub["tier"] == "S4"]
        if len(j3) == 0 or len(s4) == 0:
            continue
        gap_rows.append({
            "domain": dom,
            "period_year": period_year,
            "breadth_gap": s4["breadth_resid"].iloc[0] - j3["breadth_resid"].iloc[0],
            "ai_gap": s4["ai_binary_rate"].iloc[0] - j3["ai_binary_rate"].iloc[0],
            "scope_gap": s4["scope_density"].iloc[0] - j3["scope_density"].iloc[0],
            "tech_count_gap": s4["tech_count"].iloc[0] - j3["tech_count"].iloc[0],
            "credential_gap": s4["credential_stack_depth"].iloc[0] - j3["credential_stack_depth"].iloc[0],
            "n_j3": int(j3["n"].iloc[0]),
            "n_s4": int(s4["n"].iloc[0]),
        })
gap_df = pd.DataFrame(gap_rows)
# Gap change 2024->2026
if len(gap_df) > 0:
    gap_wide = gap_df.pivot(index="domain", columns="period_year",
                               values=["breadth_gap", "ai_gap", "scope_gap", "tech_count_gap", "credential_gap", "n_j3", "n_s4"])
    # Flatten
    gap_wide.columns = [f"{c[0]}_{c[1]}" for c in gap_wide.columns]
    for col in ["breadth_gap", "ai_gap", "scope_gap", "tech_count_gap", "credential_gap"]:
        if f"{col}_2024" in gap_wide.columns and f"{col}_2026" in gap_wide.columns:
            gap_wide[f"{col}_change"] = gap_wide[f"{col}_2026"] - gap_wide[f"{col}_2024"]
    gap_wide = gap_wide.dropna(subset=[c for c in gap_wide.columns if c.endswith("_change")])
    gap_wide = gap_wide.sort_values("breadth_gap_change", ascending=False)
    gap_wide.to_csv(f"{OUT}/step4_jr_sr_gap_change_by_domain.csv")
    print("=== S4 vs J3 gap change by domain (positive = gap widened) ===")
    show_cols = [c for c in gap_wide.columns if c.endswith("_change") or c.startswith("n_")]
    print(gap_wide[show_cols].round(3).to_string())

# --------------------------------------------------------------------------
# Step 5: Senior archetype shift by domain
# --------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Step 5: Senior scope change by domain (is it universal across domains?)")
print("=" * 72)

# For each domain, compute S4 breadth_resid delta and J3 breadth_resid delta
s4_change_rows = []
for dom, grp in work.groupby("domain"):
    for tier_name, mask_col in [("J3", "is_J3"), ("S4", "is_S4")]:
        g = grp[grp[mask_col].fillna(False)]
        if len(g) < 30:
            continue
        d0 = g[g["period_year"] == "2024"]
        d1 = g[g["period_year"] == "2026"]
        if len(d0) < 10 or len(d1) < 10:
            continue
        s4_change_rows.append({
            "domain": dom,
            "tier": tier_name,
            "breadth_resid_2024": d0["requirement_breadth_resid"].mean(),
            "breadth_resid_2026": d1["requirement_breadth_resid"].mean(),
            "breadth_resid_delta": d1["requirement_breadth_resid"].mean() - d0["requirement_breadth_resid"].mean(),
            "tech_count_2024": d0["tech_count"].mean(),
            "tech_count_2026": d1["tech_count"].mean(),
            "tech_count_delta": d1["tech_count"].mean() - d0["tech_count"].mean(),
            "ai_2024": d0["ai_binary"].astype(float).mean(),
            "ai_2026": d1["ai_binary"].astype(float).mean(),
            "ai_delta": d1["ai_binary"].astype(float).mean() - d0["ai_binary"].astype(float).mean(),
            "n_2024": len(d0),
            "n_2026": len(d1),
        })
s4_change_df = pd.DataFrame(s4_change_rows)
s4_change_df.to_csv(f"{OUT}/step5_senior_junior_change_by_domain.csv", index=False)
# Compare S4 vs J3 delta per domain
s4_vs_j3 = s4_change_df.pivot(index="domain", columns="tier",
                                  values=["breadth_resid_delta", "tech_count_delta", "ai_delta"])
s4_vs_j3.columns = [f"{c[0]}_{c[1]}" for c in s4_vs_j3.columns]
for metric in ["breadth_resid_delta", "tech_count_delta", "ai_delta"]:
    s4_col = f"{metric}_S4"
    j3_col = f"{metric}_J3"
    if s4_col in s4_vs_j3.columns and j3_col in s4_vs_j3.columns:
        s4_vs_j3[f"{metric}_S4_minus_J3"] = s4_vs_j3[s4_col] - s4_vs_j3[j3_col]
s4_vs_j3.to_csv(f"{OUT}/step5_S4_minus_J3_by_domain.csv")
print("=== S4 vs J3 scope delta by domain (S4 - J3 shows where senior inflates more) ===")
print(s4_vs_j3.round(3).to_string())

# --------------------------------------------------------------------------
# Step 6: AI/ML archetype cross-validation
# --------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Step 6: AI/ML archetype deep dive")
print("=" * 72)

aiml = corpus[corpus["domain"] == "ai_ml"].copy()
print(f"AI/ML rows: {len(aiml):,}")
print(f"By period: {aiml.groupby('period_year').size().to_dict()}")

# Top employers
top_emp_2024 = aiml[aiml["period_year"] == "2024"]["company_name_canonical"].value_counts().head(15)
top_emp_2026 = aiml[aiml["period_year"] == "2026"]["company_name_canonical"].value_counts().head(15)
print("\nTop AI/ML employers 2024:")
print(top_emp_2024.to_string())
print("\nTop AI/ML employers 2026:")
print(top_emp_2026.to_string())
# Save
pd.concat([
    top_emp_2024.rename("n_2024").to_frame().assign(period="2024"),
    top_emp_2026.rename("n_2026").to_frame().assign(period="2026"),
]).to_csv(f"{OUT}/step6_aiml_top_employers.csv")

# J3 vs S4 mix
aiml_with_yoe = aiml.dropna(subset=["yoe_min_years_llm"])
mix = aiml_with_yoe.groupby("period_year").agg(
    n=("uid", "count"),
    j3_rate=("is_J3", lambda s: s.astype(float).mean()),
    s4_rate=("is_S4", lambda s: s.astype(float).mean()),
    yoe_median=("yoe_min_years_llm", "median"),
    yoe_mean=("yoe_min_years_llm", "mean"),
)
print("\nJ3/S4 mix in AI/ML:")
print(mix.round(3))
mix.to_csv(f"{OUT}/step6_aiml_yoe_mix.csv")

# Tech stack — pull tech matrix for AI/ML
tech = pq.read_table(f"{SHARED}/swe_tech_matrix.parquet").to_pandas()
tech_cols = [c for c in tech.columns if c != "uid"]
aiml_tech = aiml.merge(tech, on="uid", how="left")
tech_rates = {}
for period_year in ["2024", "2026"]:
    sub = aiml_tech[aiml_tech["period_year"] == period_year]
    if len(sub) == 0:
        continue
    tech_rates[period_year] = sub[tech_cols].astype(float).mean()
tech_rate_df = pd.DataFrame(tech_rates)
tech_rate_df["delta"] = tech_rate_df["2026"] - tech_rate_df["2024"]
tech_rate_df = tech_rate_df.sort_values("delta", ascending=False)
# Save top 30 risers + fallers
aiml_top = pd.concat([tech_rate_df.head(25), tech_rate_df.tail(10)])
aiml_top.to_csv(f"{OUT}/step6_aiml_tech_delta.csv")
print("\n=== AI/ML tech stack: top risers 2024 -> 2026 ===")
print(tech_rate_df.head(25).round(3).to_string())

# New entrants vs existing employers
emp_2024 = set(corpus[corpus["period_year"] == "2024"]["company_name_canonical"].dropna())
emp_2026 = set(corpus[corpus["period_year"] == "2026"]["company_name_canonical"].dropna())
aiml_2026 = aiml[aiml["period_year"] == "2026"]
new_emp = aiml_2026[~aiml_2026["company_name_canonical"].isin(emp_2024)]
returning_emp = aiml_2026[aiml_2026["company_name_canonical"].isin(emp_2024)]
print(f"\nAI/ML 2026 postings from NEW employers (not in 2024 corpus): {len(new_emp):,} ({len(new_emp)/len(aiml_2026)*100:.1f}%)")
print(f"AI/ML 2026 postings from RETURNING employers: {len(returning_emp):,} ({len(returning_emp)/len(aiml_2026)*100:.1f}%)")

# Description length and credential stack in AI/ML
aiml_scope = aiml.merge(t11[["uid", "tech_count", "requirement_breadth", "credential_stack_depth",
                                  "description_cleaned_length", "ai_binary"]], on="uid", how="left")
aiml_profile = aiml_scope.groupby("period_year").agg(
    n=("uid", "count"),
    desc_length=("description_cleaned_length", "mean"),
    breadth=("requirement_breadth", "mean"),
    tech_count=("tech_count", "mean"),
    credential=("credential_stack_depth", "mean"),
    ai_rate=("ai_binary", lambda s: s.astype(float).mean()),
)
print("\n=== AI/ML archetype profile by period ===")
print(aiml_profile.round(2))
aiml_profile.to_csv(f"{OUT}/step6_aiml_profile.csv")

# Verdict by source (ML engineer cross-source caveat)
aiml_by_src = aiml.groupby(["period_year", "source"]).size().unstack(fill_value=0)
print("\nAI/ML by period x source:")
print(aiml_by_src)
aiml_by_src.to_csv(f"{OUT}/step6_aiml_by_source.csv")

# T30 panel sensitivity on J3/S4 decomp
print("\n" + "=" * 72)
print("Step 7: T30 panel sensitivity (arshkon-only)")
print("=" * 72)

# Arshkon-only baseline for senior decomp
c_ark = corpus[(corpus["domain"] != "unknown") & (corpus["source"] != "kaggle_asaniczka")].copy()
c_ark = c_ark.dropna(subset=["yoe_min_years_llm"])
c_ark["is_J3_numeric"] = c_ark["is_J3"].astype(float)
c_ark["is_S4_numeric"] = c_ark["is_S4"].astype(float)

dec_ark_j3 = decompose(c_ark, "domain", "is_J3_numeric")
dec_ark_s4 = decompose(c_ark, "domain", "is_S4_numeric")
print(f"Arshkon-only J3 decomp: delta={dec_ark_j3['delta']*100:.2f} pp, within={dec_ark_j3['within']*100:.2f} pp, between={dec_ark_j3['between']*100:.2f} pp")
print(f"Arshkon-only S4 decomp: delta={dec_ark_s4['delta']*100:.2f} pp, within={dec_ark_s4['within']*100:.2f} pp, between={dec_ark_s4['between']*100:.2f} pp")

# Save full T30 sensitivity
t30_rows = []
for baseline_name, df_baseline in [
    ("pooled-2024 (primary)", corpus[corpus["domain"] != "unknown"]),
    ("arshkon-only", corpus[(corpus["domain"] != "unknown") & (corpus["source"] != "kaggle_asaniczka")]),
    ("non-aggregator", corpus[(corpus["domain"] != "unknown") & (~corpus["is_aggregator"].fillna(False))]),
]:
    for def_name, col in [("J3", "is_J3"), ("S4", "is_S4"), ("J1", "is_J1"), ("J2", "is_J2")]:
        c = df_baseline.copy()
        if def_name in ("J3", "S4"):
            c = c.dropna(subset=["yoe_min_years_llm"])
        c["_m"] = c[col].astype(float)
        dec = decompose(c, "domain", "_m")
        if not dec:
            continue
        t30_rows.append({
            "baseline": baseline_name,
            "definition": def_name,
            "delta_pp": round(dec["delta"] * 100, 2),
            "within_pp": round(dec["within"] * 100, 2),
            "between_pp": round(dec["between"] * 100, 2),
            "interaction_pp": round(dec["interaction"] * 100, 2),
            "n_2024": int((c["period_year"] == "2024").sum()),
            "n_2026": int((c["period_year"] == "2026").sum()),
        })
t30_df = pd.DataFrame(t30_rows)
t30_df.to_csv(f"{OUT}/step7_t30_panel_decomposition.csv", index=False)
print("\n=== T30 panel decomposition (baseline x definition) ===")
print(t30_df.to_string(index=False))

print("\nDone. Outputs in", OUT)
