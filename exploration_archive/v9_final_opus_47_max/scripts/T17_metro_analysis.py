"""
T17 — Geographic market structure.

For SWE LinkedIn, compute metro-level 2024 vs 2026 metrics:
- Entry share (J3 primary, J1/J2 sens)
- AI prevalence (ai_strict primary; ai_broad_no_mcp sensitivity)
- Org scope term count (V1 scope)
- Median description length
- Tech diversity (median distinct tech count)

Remote share (2026): use is_remote.
Domain archetype distribution: cross-tabulate T09 archetype × metro.

Multi-location rows: excluded naturally by metro_area IS NULL (collapse).
Excluded count reported.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

ROOT = Path(__file__).resolve().parents[2]
UNIFIED = str(ROOT / "data" / "unified.parquet")
CLEAN_TEXT = str(ROOT / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet")
TECH_MATRIX = str(ROOT / "exploration" / "artifacts" / "shared" / "swe_tech_matrix.parquet")
T11_FEATS = str(ROOT / "exploration" / "artifacts" / "shared" / "T11_posting_features.parquet")
ARCHETYPE = str(ROOT / "exploration" / "artifacts" / "shared" / "swe_archetype_labels.parquet")
PATTERNS_PATH = str(ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json")
OUT_DIR = ROOT / "exploration" / "tables" / "T17"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATTERNS = json.loads(Path(PATTERNS_PATH).read_text())
AI_STRICT = PATTERNS["ai_strict"]["pattern"]
SCOPE = PATTERNS["scope"]["pattern"]
AI_BROAD_NO_MCP = PATTERNS["ai_broad"]["pattern"].replace("|mcp)", ")")

DEFAULT_FILTER = (
    "source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true"
)

con = duckdb.connect()

# flags
con.execute(f"""
CREATE OR REPLACE TABLE tf AS
SELECT
  uid,
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS ai_strict,
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS ai_broad_nomcp,
  CAST(list_reduce(
    list_transform(regexp_extract_all(lower(description_cleaned), ?), x -> 1),
    (acc, v) -> acc + v, 0
  ) AS BIGINT) AS scope_term_count
FROM '{CLEAN_TEXT}'
""", [AI_STRICT, AI_BROAD_NO_MCP, SCOPE])

con.execute(f"""
CREATE OR REPLACE TABLE base AS
SELECT
  u.uid,
  u.metro_area,
  u.state_normalized,
  u.is_multi_location,
  u.is_remote,
  u.is_aggregator,
  u.company_name_canonical,
  u.source,
  u.period,
  CASE WHEN u.period IN ('2024-01','2024-04') THEN '2024' ELSE '2026' END AS era,
  u.yoe_min_years_llm,
  u.llm_classification_coverage,
  u.seniority_final,
  u.description_length,
  COALESCE(tf.ai_strict,0) AS ai_strict,
  COALESCE(tf.ai_broad_nomcp,0) AS ai_broad_nomcp,
  COALESCE(tf.scope_term_count,0) AS scope_term_count,
  COALESCE(tc.tech_count,0) AS tech_count,
  tc.requirement_breadth_resid,
  CASE WHEN u.seniority_final='entry' THEN 1 ELSE 0 END AS j1,
  CASE WHEN u.seniority_final='associate' THEN 1 ELSE 0 END AS j2,
  CASE WHEN u.llm_classification_coverage='labeled' AND u.yoe_min_years_llm IS NOT NULL
         AND u.yoe_min_years_llm <= 2 THEN 1 ELSE 0 END AS j3,
  CASE WHEN u.llm_classification_coverage='labeled' AND u.yoe_min_years_llm IS NOT NULL
       THEN 1 ELSE 0 END AS labeled,
  arc.archetype_name
FROM '{UNIFIED}' u
LEFT JOIN tf USING (uid)
LEFT JOIN '{T11_FEATS}' tc USING (uid)
LEFT JOIN '{ARCHETYPE}' arc USING (uid)
WHERE {DEFAULT_FILTER}
""")

n = con.execute("SELECT COUNT(*) FROM base").fetchone()[0]
print(f"[base] {n} SWE LinkedIn rows")

# Excluded multi-location + null-metro
excl = con.execute("""
SELECT
  SUM(CASE WHEN is_multi_location THEN 1 ELSE 0 END) AS n_multi,
  SUM(CASE WHEN metro_area IS NULL AND NOT is_multi_location THEN 1 ELSE 0 END) AS n_null_no_multi,
  SUM(CASE WHEN metro_area IS NULL THEN 1 ELSE 0 END) AS n_null
FROM base
""").fetchone()
print(f"[excluded] multi={excl[0]} null-metro (non-multi)={excl[1]} total-null-metro={excl[2]}")

# --------------------------------------------------------------
# Step 1: Metro-level metrics with ≥50 SWE per era
# --------------------------------------------------------------
metros_cnt = con.execute("""
SELECT metro_area,
  SUM(CASE WHEN era='2024' THEN 1 ELSE 0 END) AS n_2024,
  SUM(CASE WHEN era='2026' THEN 1 ELSE 0 END) AS n_2026
FROM base
WHERE metro_area IS NOT NULL
GROUP BY 1
""").df()
keep_metros = metros_cnt.loc[(metros_cnt.n_2024 >= 50) & (metros_cnt.n_2026 >= 50), "metro_area"].tolist()
print(f"[metros] with ≥50 in both eras: {len(keep_metros)}")

df = con.execute("SELECT * FROM base WHERE metro_area IS NOT NULL").df()
df = df[df.metro_area.isin(keep_metros)]
print(f"[df] {len(df)} rows in qualifying metros")


# per metro × era metrics
def metro_era(df):
    g = df.groupby(["metro_area", "era"]).agg(
        n=("uid", "size"),
        j1_share=("j1", "mean"),
        j2_share=("j2", "mean"),
        j3_num=("j3", "sum"),
        j3_den=("labeled", "sum"),
        ai_strict=("ai_strict", "mean"),
        ai_broad_nomcp=("ai_broad_nomcp", "mean"),
        scope_term_count=("scope_term_count", "mean"),
        desc_length_median=("description_length", "median"),
        tech_count_median=("tech_count", "median"),
        remote_share=("is_remote", "mean"),
    ).reset_index()
    g["j3_share_labeled"] = g["j3_num"] / g["j3_den"].replace(0, np.nan)
    return g


metro_era_df = metro_era(df)
print(f"[metro_era] {len(metro_era_df)} rows")

# Wide form
piv = metro_era_df.pivot(index="metro_area", columns="era").copy()
piv.columns = [f"{a}_{b}" for a, b in piv.columns]
piv = piv.reset_index()

# Deltas
for col in ["j1_share", "j2_share", "j3_share_labeled", "ai_strict", "ai_broad_nomcp",
            "scope_term_count", "desc_length_median", "tech_count_median"]:
    piv[f"{col}_delta"] = piv[f"{col}_2026"] - piv[f"{col}_2024"]

piv["n_2024"] = piv["n_2024"]
piv["n_2026"] = piv["n_2026"]
piv["remote_share_2026"] = piv["remote_share_2026"]
piv = piv.sort_values("n_2026", ascending=False)

# Save metro-level metrics
piv.to_csv(OUT_DIR / "metro_metrics.csv", index=False)
print("[save] metro_metrics.csv")

# Print key metro table
cols = ["metro_area", "n_2024", "n_2026",
        "j3_share_labeled_2024", "j3_share_labeled_2026", "j3_share_labeled_delta",
        "ai_strict_2024", "ai_strict_2026", "ai_strict_delta",
        "scope_term_count_delta", "desc_length_median_delta", "tech_count_median_delta",
        "remote_share_2026"]
print(piv[cols].round(3).to_string(index=False))

# --------------------------------------------------------------
# Step 2: Metro rankings
# --------------------------------------------------------------
rankings = {}
for metric in ["j3_share_labeled_delta", "ai_strict_delta", "scope_term_count_delta",
               "desc_length_median_delta", "tech_count_median_delta"]:
    rk = piv.sort_values(metric, ascending=False)[["metro_area", metric]].reset_index(drop=True)
    rk["rank"] = rk.index + 1
    rankings[metric] = rk
    print(f"\n[rank] top/bottom 5 by {metric}:")
    print("top:", rk.head(5).to_dict(orient="records"))
    print("bot:", rk.tail(5).to_dict(orient="records"))

ranks_out = pd.concat(
    [rankings[m].assign(metric=m) for m in rankings],
    ignore_index=True,
)
ranks_out.to_csv(OUT_DIR / "metro_rankings.csv", index=False)

# --------------------------------------------------------------
# Step 3: Tech hub vs non-tech-hub comparison
# --------------------------------------------------------------
TECH_HUBS = {"San Francisco Bay Area", "New York City Metro", "Seattle Metro",
             "Austin Metro", "Boston Metro"}
piv["is_tech_hub"] = piv.metro_area.isin(TECH_HUBS)
hub_summary = piv.groupby("is_tech_hub").agg(
    n_metros=("metro_area", "size"),
    j3_delta_mean=("j3_share_labeled_delta", "mean"),
    ai_strict_delta_mean=("ai_strict_delta", "mean"),
    scope_delta_mean=("scope_term_count_delta", "mean"),
    desc_length_delta_mean=("desc_length_median_delta", "mean"),
    tech_count_delta_mean=("tech_count_median_delta", "mean"),
).reset_index()
hub_summary.to_csv(OUT_DIR / "tech_hub_vs_non.csv", index=False)
print("\n[save] tech_hub_vs_non.csv")
print(hub_summary.to_string(index=False))

# --------------------------------------------------------------
# Step 4: Metro-level correlation (AI delta × entry delta, AI delta × scope delta)
# --------------------------------------------------------------
corrs = {}
for a, b in [("ai_strict_delta", "j3_share_labeled_delta"),
             ("ai_strict_delta", "scope_term_count_delta"),
             ("ai_strict_delta", "desc_length_median_delta"),
             ("j3_share_labeled_delta", "scope_term_count_delta"),
             ("ai_strict_delta", "tech_count_median_delta")]:
    sub = piv.dropna(subset=[a, b])
    r = sub[a].corr(sub[b])
    rs = sub[a].corr(sub[b], method="spearman")
    corrs[(a, b)] = {"pearson": r, "spearman": rs, "n": len(sub)}
    print(f"[corr] {a} × {b}: pearson={r:.3f} spearman={rs:.3f} n={len(sub)}")

corr_df = pd.DataFrame([
    {"x": k[0], "y": k[1], "pearson": v["pearson"], "spearman": v["spearman"], "n": v["n"]}
    for k, v in corrs.items()
])
corr_df.to_csv(OUT_DIR / "metro_correlations.csv", index=False)

# --------------------------------------------------------------
# Step 5: Remote share by metro (scraped 2026 only)
# --------------------------------------------------------------
remote_by = df[df.era == "2026"].groupby("metro_area").agg(
    n=("uid", "size"),
    remote_share=("is_remote", "mean"),
).reset_index().sort_values("remote_share", ascending=False)
remote_by.to_csv(OUT_DIR / "remote_share_by_metro.csv", index=False)
print("\n[save] remote_share_by_metro.csv")
print(remote_by.head(20).to_string(index=False))

# --------------------------------------------------------------
# Step 6: Domain archetype distribution by metro
# --------------------------------------------------------------
arc_df = df.dropna(subset=["archetype_name"]).copy()
print(f"\n[archetype] rows with label: {len(arc_df)}")
# Share of each archetype per metro × era
arc_metro = arc_df.groupby(["metro_area", "era", "archetype_name"]).size().reset_index(name="n")
# Normalize within metro×era
arc_totals = arc_metro.groupby(["metro_area", "era"])["n"].sum().reset_index(name="total")
arc_metro = arc_metro.merge(arc_totals, on=["metro_area", "era"])
arc_metro["share"] = arc_metro["n"] / arc_metro["total"]

# Save long form
arc_metro.to_csv(OUT_DIR / "archetype_by_metro_long.csv", index=False)

# Wide form for the ML/LLM, frontend, embedded archetypes
key_arcs = ["models/systems/llm", "backend/javascript/typescript",
            "android/kotlin/jetpack", "systems/clearance/requirements",
            "qa/selenium/manual", "kubernetes/terraform/cicd",
            "pipelines/sql/etl", "java/boot/microservices"]

piv_arc = arc_metro[arc_metro.archetype_name.isin(key_arcs)].pivot_table(
    index="metro_area", columns=["archetype_name", "era"], values="share", aggfunc="first"
)
piv_arc.columns = [f"{a}_{b}" for a, b in piv_arc.columns]
piv_arc = piv_arc.reset_index()
piv_arc.to_csv(OUT_DIR / "archetype_by_metro_wide.csv", index=False)

# Top metros for ML/LLM
ml_by_metro = arc_metro[arc_metro.archetype_name == "models/systems/llm"].copy()
ml_by_metro = ml_by_metro.pivot(index="metro_area", columns="era", values="share").fillna(0)
ml_by_metro["delta"] = ml_by_metro.get("2026", 0) - ml_by_metro.get("2024", 0)
ml_by_metro = ml_by_metro.reset_index().sort_values("delta", ascending=False)
ml_by_metro.to_csv(OUT_DIR / "ml_llm_archetype_by_metro.csv", index=False)
print(f"[save] ml_llm_archetype_by_metro.csv — top 10 by delta:")
print(ml_by_metro.head(10).to_string(index=False))

# Archetype concentration change — HHI per metro
def hhi(df_metric):
    shares = df_metric.values
    return float((shares ** 2).sum())


hhi_rows = []
for (metro, era), sub in arc_metro.groupby(["metro_area", "era"]):
    hhi_rows.append({"metro_area": metro, "era": era, "hhi": hhi(sub["share"]), "n_arcs": len(sub)})
hhi_df = pd.DataFrame(hhi_rows)
hhi_wide = hhi_df.pivot(index="metro_area", columns="era", values="hhi").reset_index()
hhi_wide.columns = ["metro_area", "hhi_2024", "hhi_2026"]
hhi_wide["hhi_delta"] = hhi_wide["hhi_2026"] - hhi_wide["hhi_2024"]
hhi_wide.to_csv(OUT_DIR / "archetype_hhi_by_metro.csv", index=False)
print(f"\n[save] archetype_hhi_by_metro.csv")
print(hhi_wide.sort_values("hhi_delta", ascending=False).head(10).to_string(index=False))

# --------------------------------------------------------------
# Step 7: Metro heatmap data (save CSV for plotting, don't require matplotlib)
# --------------------------------------------------------------
heat = piv[["metro_area", "j3_share_labeled_delta", "ai_strict_delta",
            "scope_term_count_delta", "desc_length_median_delta", "tech_count_median_delta",
            "n_2026"]].copy()
heat = heat.sort_values("n_2026", ascending=False)
heat.to_csv(OUT_DIR / "metro_heatmap_data.csv", index=False)

# Try rendering an actual heatmap PNG
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(heat))))
    metrics_cols = ["j3_share_labeled_delta", "ai_strict_delta",
                    "scope_term_count_delta", "desc_length_median_delta",
                    "tech_count_median_delta"]
    labels = ["J3 share Δ", "AI-strict Δ", "Scope-terms Δ", "Desc-len(med) Δ", "Tech-count(med) Δ"]
    # standardize each column for visual comparability
    M = heat[metrics_cols].copy()
    for c in metrics_cols:
        mu = M[c].mean()
        sd = M[c].std() or 1.0
        M[c] = (M[c] - mu) / sd
    im = ax.imshow(M.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_yticks(range(len(heat)))
    ax.set_yticklabels(heat["metro_area"], fontsize=8)
    ax.set_xticks(range(len(metrics_cols)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Z-score vs cross-metro mean", fontsize=8)
    ax.set_title("Metro × metric: 2024→2026 change (z-scored across metros)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "metro_heatmap.png", dpi=140)
    plt.close()
    print("[save] metro_heatmap.png")
except Exception as e:
    print(f"[heatmap:warn] {e}")


# --------------------------------------------------------------
# Step 8: Sensitivity — aggregator exclusion, company capping
# --------------------------------------------------------------
# Aggregator exclusion per metro
nonagg = df[~df.is_aggregator].copy()
nonagg_m = metro_era(nonagg)
nonagg_piv = nonagg_m.pivot(index="metro_area", columns="era").copy()
nonagg_piv.columns = [f"{a}_{b}" for a, b in nonagg_piv.columns]
nonagg_piv = nonagg_piv.reset_index()
for col in ["j3_share_labeled", "ai_strict", "scope_term_count"]:
    nonagg_piv[f"{col}_delta"] = nonagg_piv[f"{col}_2026"] - nonagg_piv[f"{col}_2024"]

# Compare vs all-inclusive
comp_cols = ["metro_area", "j3_share_labeled_delta", "ai_strict_delta", "scope_term_count_delta"]
all_m = piv[comp_cols].rename(columns={
    "j3_share_labeled_delta": "j3_delta_all",
    "ai_strict_delta": "ai_delta_all",
    "scope_term_count_delta": "scope_delta_all",
})
nonagg_cmp = nonagg_piv[comp_cols].rename(columns={
    "j3_share_labeled_delta": "j3_delta_no_agg",
    "ai_strict_delta": "ai_delta_no_agg",
    "scope_term_count_delta": "scope_delta_no_agg",
})
sens_metro = all_m.merge(nonagg_cmp, on="metro_area", how="outer")
sens_metro.to_csv(OUT_DIR / "sensitivity_aggregator_by_metro.csv", index=False)
print(f"[save] sensitivity_aggregator_by_metro.csv")

# Company-cap sensitivity: cap any single company at max 20 postings per metro × era
def capped_share(df, cap=20, metric="ai_strict"):
    g = df.groupby(["metro_area", "era", "company_name_canonical"]).size().reset_index(name="n")
    g["capped_n"] = g["n"].clip(upper=cap)
    # Build weighted metric at the capped level
    df2 = df.merge(g[["metro_area", "era", "company_name_canonical", "n"]],
                   on=["metro_area", "era", "company_name_canonical"])
    # weight = min(cap, n) / n per posting
    df2["weight"] = df2["n"].clip(upper=cap) / df2["n"]
    out = df2.groupby(["metro_area", "era"]).apply(
        lambda x: np.average(x[metric].values, weights=x["weight"].values)
    ).reset_index(name=f"{metric}_cap{cap}")
    return out


# apply for AI-strict and J3
cap_ai = capped_share(df, cap=20, metric="ai_strict")
cap_j3 = capped_share(df[df.labeled == 1], cap=20, metric="j3")
cap_ai_w = cap_ai.pivot(index="metro_area", columns="era", values="ai_strict_cap20").reset_index()
cap_ai_w.columns = ["metro_area", "ai_strict_cap20_2024", "ai_strict_cap20_2026"]
cap_ai_w["ai_strict_cap20_delta"] = cap_ai_w["ai_strict_cap20_2026"] - cap_ai_w["ai_strict_cap20_2024"]
cap_j3_w = cap_j3.pivot(index="metro_area", columns="era", values="j3_cap20").reset_index()
cap_j3_w.columns = ["metro_area", "j3_cap20_2024", "j3_cap20_2026"]
cap_j3_w["j3_cap20_delta"] = cap_j3_w["j3_cap20_2026"] - cap_j3_w["j3_cap20_2024"]
cap_out = cap_ai_w.merge(cap_j3_w, on="metro_area", how="outer")
cap_out = cap_out.merge(piv[["metro_area", "ai_strict_delta", "j3_share_labeled_delta"]], on="metro_area")
cap_out.to_csv(OUT_DIR / "sensitivity_company_cap20_by_metro.csv", index=False)
print(f"[save] sensitivity_company_cap20_by_metro.csv")
print(cap_out[["metro_area", "ai_strict_delta", "ai_strict_cap20_delta",
               "j3_share_labeled_delta", "j3_cap20_delta"]].head(10).to_string(index=False))
