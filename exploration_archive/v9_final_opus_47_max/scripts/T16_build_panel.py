"""
T16 — Company hiring strategy typology.

Builds:
- overlap_panel.csv: arshkon∩scraped ≥3, arshkon∩scraped ≥5, pooled∩scraped ≥5
- company_change_vectors.csv: per-company 2024→2026 delta vectors under multiple panels
- within_between_decomposition.csv: within-co vs between-co decomposition per metric × panel
- cluster_summary.csv: k-means strategy clusters
- scope_same_co_senior_vs_junior.csv: V1 scope claim on same-co restriction
- new_entrants_vs_returning.csv: new-market entrants profile vs returning cohort
- aggregator_vs_direct.csv: aggregator comparison

All SWE, LinkedIn-only, default filter.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
UNIFIED = str(ROOT / "data" / "unified.parquet")
CLEAN_TEXT = str(ROOT / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet")
TECH_MATRIX = str(ROOT / "exploration" / "artifacts" / "shared" / "swe_tech_matrix.parquet")
T11_FEATS = str(ROOT / "exploration" / "artifacts" / "shared" / "T11_posting_features.parquet")
PATTERNS_PATH = str(ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json")
OUT_DIR = ROOT / "exploration" / "tables" / "T16"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENTRY_SPEC = str(ROOT / "exploration" / "artifacts" / "shared" / "entry_specialist_employers.csv")

DEFAULT_FILTER = (
    "source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true"
)

PATTERNS = json.loads(Path(PATTERNS_PATH).read_text())

# V1-validated patterns (AI-strict primary; drop fine-tuning/MCP contamination at baseline handled by using ai_strict)
AI_STRICT = PATTERNS["ai_strict"]["pattern"]
MGMT_REBUILT = PATTERNS["v1_rebuilt_patterns"]["mgmt_strict_v1_rebuilt"]["pattern"]
SCOPE = PATTERNS["scope"]["pattern"]
# also ai_broad with MCP-drop for optional comparison
AI_BROAD_NO_MCP = PATTERNS["ai_broad"]["pattern"].replace("|mcp)", ")")

print("[setup] patterns loaded")
print(f"  ai_strict head: {AI_STRICT[:80]}")
print(f"  scope head: {SCOPE[:80]}")
print(f"  mgmt_rebuilt head: {MGMT_REBUILT[:80]}")

con = duckdb.connect()

# ------------------------------------------------------------------
# Step 0: Register view with pattern-matched flags on cleaned text.
# ------------------------------------------------------------------
con.execute(f"""
CREATE OR REPLACE TABLE text_flags AS
SELECT
  uid,
  -- AI strict (V1 validated)
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS ai_strict,
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS ai_broad_nomcp,
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS scope_bin,
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS mgmt_rebuilt,
  -- scope term COUNT (for org_scope term count metric)
  length(description_cleaned) - length(regexp_replace(lower(description_cleaned), ?, '', 'g')) AS scope_match_span,
  description_cleaned,
  length(description_cleaned) AS desc_len
FROM '{CLEAN_TEXT}'
WHERE description_cleaned IS NOT NULL
""", [AI_STRICT, AI_BROAD_NO_MCP, SCOPE, MGMT_REBUILT, SCOPE])

# count org_scope tokens via regex split
# simpler approach: use regexp_extract_all
con.execute(f"""
CREATE OR REPLACE TABLE scope_counts AS
SELECT
  uid,
  CAST(list_reduce(
    list_transform(
      regexp_extract_all(lower(description_cleaned), ?),
      x -> 1
    ),
    (acc, v) -> acc + v,
    0
  ) AS BIGINT) AS scope_term_count
FROM '{CLEAN_TEXT}'
WHERE description_cleaned IS NOT NULL
""", [SCOPE])

# ------------------------------------------------------------------
# Step 1: Build base SWE frame (cleaned_text is pre-filtered SWE LinkedIn)
# ------------------------------------------------------------------
con.execute(f"""
CREATE OR REPLACE TABLE base AS
SELECT
  u.uid,
  u.company_name_canonical,
  u.is_aggregator,
  u.source,
  u.period,
  CASE WHEN u.period IN ('2024-01','2024-04') THEN '2024' ELSE '2026' END AS era,
  u.yoe_min_years_llm,
  u.llm_classification_coverage,
  u.seniority_final,
  u.description_length,
  tf.ai_strict,
  tf.ai_broad_nomcp,
  tf.scope_bin,
  tf.mgmt_rebuilt,
  tf.desc_len AS cleaned_len,
  sc.scope_term_count,
  tc.tech_count,
  tc.requirement_breadth_resid,
  tc.credential_stack_depth
FROM 'data/unified.parquet' u
LEFT JOIN text_flags tf USING (uid)
LEFT JOIN scope_counts sc USING (uid)
LEFT JOIN '{T11_FEATS}' tc USING (uid)
WHERE {DEFAULT_FILTER}
  AND u.company_name_canonical IS NOT NULL
""")

n_base = con.execute("SELECT COUNT(*) FROM base").fetchone()[0]
print(f"[base] {n_base} SWE LinkedIn rows")

# ------------------------------------------------------------------
# Step 2: Panel membership — arshkon_min3 / arshkon_min5 / pooled_min5
# ------------------------------------------------------------------
panel_q = """
WITH cmp AS (
  SELECT company_name_canonical,
    SUM(CASE WHEN source='kaggle_arshkon' THEN 1 ELSE 0 END) AS n_2024_arshkon,
    SUM(CASE WHEN source='kaggle_asaniczka' THEN 1 ELSE 0 END) AS n_2024_asaniczka,
    SUM(CASE WHEN source IN ('kaggle_arshkon','kaggle_asaniczka') THEN 1 ELSE 0 END) AS n_2024_pooled,
    SUM(CASE WHEN source='scraped' THEN 1 ELSE 0 END) AS n_2026,
    MAX(CASE WHEN is_aggregator THEN 1 ELSE 0 END)::BOOLEAN AS is_aggregator
  FROM base
  GROUP BY 1
)
SELECT * FROM cmp
"""
cmp = con.execute(panel_q).df()
print(f"[panel] {len(cmp)} distinct companies with at least one SWE posting")

# Join entry-specialist flag
ent = pd.read_csv(ENTRY_SPEC)
cmp["is_entry_specialist"] = cmp["company_name_canonical"].isin(ent["company_name_canonical"]).astype(bool)

# three panels (long-form)
frames = []
frames.append(cmp.loc[(cmp.n_2024_arshkon >= 3) & (cmp.n_2026 >= 3)].assign(panel_type="arshkon_min3"))
frames.append(cmp.loc[(cmp.n_2024_arshkon >= 5) & (cmp.n_2026 >= 5)].assign(panel_type="arshkon_min5"))
frames.append(cmp.loc[(cmp.n_2024_pooled >= 5) & (cmp.n_2026 >= 5)].assign(panel_type="pooled_min5"))
overlap = pd.concat(frames, ignore_index=True)
print(
    f"[panels] arshkon_min3: {(overlap.panel_type=='arshkon_min3').sum()} | "
    f"arshkon_min5: {(overlap.panel_type=='arshkon_min5').sum()} | "
    f"pooled_min5: {(overlap.panel_type=='pooled_min5').sum()}"
)

# ------------------------------------------------------------------
# Step 3: Per-company × era metrics (need YOE-labeled denominator for J3/J1/J2)
# ------------------------------------------------------------------
# YOE-labeled subset: llm_classification_coverage='labeled' for J3
# For J1 (seniority_final=='entry') and J2 (seniority_final=='associate') we use seniority_final directly (Gate 1 exception)
# NOTE: asaniczka has ZERO seniority_native=entry but seniority_final is allowed.
# Register base with era
con.execute("""
CREATE OR REPLACE TABLE base_era AS
SELECT
  b.*,
  CASE WHEN b.seniority_final='entry' THEN 1 ELSE 0 END AS j1_entry,
  CASE WHEN b.seniority_final='associate' THEN 1 ELSE 0 END AS j2_assoc,
  CASE
    WHEN b.llm_classification_coverage='labeled' AND b.yoe_min_years_llm IS NOT NULL
         AND b.yoe_min_years_llm <= 2 THEN 1 ELSE 0 END AS j3_flag,
  CASE
    WHEN b.llm_classification_coverage='labeled' AND b.yoe_min_years_llm IS NOT NULL
    THEN 1 ELSE 0 END AS j3_denom
FROM base b
""")

# per-era, per-company aggregate with J3 using labeled denominator
co_era_q = """
SELECT
  company_name_canonical,
  era,
  COUNT(*) AS n,
  AVG(j1_entry)::DOUBLE AS entry_share_j1,
  AVG(j2_assoc)::DOUBLE AS entry_share_j2,
  -- J3 share: over labeled denominator (yoe_min_years_llm notnull)
  CAST(SUM(j3_flag) AS DOUBLE) / NULLIF(SUM(j3_denom),0) AS entry_share_j3_labeled,
  -- J3 as share of ALL postings for this company (fall-back)
  AVG(j3_flag)::DOUBLE AS entry_share_j3_all,
  SUM(j3_denom) AS n_labeled,
  AVG(ai_strict)::DOUBLE AS ai_prev_strict,
  AVG(ai_broad_nomcp)::DOUBLE AS ai_prev_broad,
  AVG(description_length)::DOUBLE AS desc_len_mean,
  AVG(tech_count)::DOUBLE AS tech_count_mean,
  AVG(scope_term_count)::DOUBLE AS scope_mean,
  AVG(requirement_breadth_resid)::DOUBLE AS breadth_resid_mean,
  AVG(credential_stack_depth)::DOUBLE AS credstack_mean
FROM base_era
GROUP BY 1,2
"""
co_era = con.execute(co_era_q).df()
print(f"[co_era] {len(co_era)} company×era rows")

# wide
wide = co_era.pivot(index="company_name_canonical", columns="era").copy()
# flatten
wide.columns = [f"{a}_{b}" for a, b in wide.columns]
wide = wide.reset_index()

# Compute deltas
def delta(col):
    return wide[f"{col}_2026"] - wide[f"{col}_2024"]

wide["entry_share_delta_j1"] = delta("entry_share_j1")
wide["entry_share_delta_j2"] = delta("entry_share_j2")
wide["entry_share_delta_j3"] = delta("entry_share_j3_labeled")
wide["ai_prevalence_delta_strict"] = delta("ai_prev_strict")
wide["ai_prevalence_delta_broad"] = delta("ai_prev_broad")
wide["desc_length_delta"] = delta("desc_len_mean")
wide["tech_count_delta"] = delta("tech_count_mean")
wide["org_scope_delta"] = delta("scope_mean")
wide["breadth_resid_delta"] = delta("breadth_resid_mean")
wide["credential_stack_depth_delta"] = delta("credstack_mean")

# Merge with panel assignments (cross-join long-form per panel_type)
panel_assign = overlap[["company_name_canonical", "panel_type",
                        "n_2024_arshkon", "n_2024_asaniczka", "n_2024_pooled",
                        "n_2026", "is_aggregator", "is_entry_specialist"]]
vecs = panel_assign.merge(wide, on="company_name_canonical", how="left")

print(f"[vecs] pre-cluster rows: {len(vecs)}")

# ------------------------------------------------------------------
# Step 4: K-means clustering on change vectors (pooled_min5 frame)
# ------------------------------------------------------------------
cluster_cols = [
    "entry_share_delta_j3",
    "ai_prevalence_delta_strict",
    "desc_length_delta",
    "tech_count_delta",
    "org_scope_delta",
    "breadth_resid_delta",
    "credential_stack_depth_delta",
]

def cluster_frame(frame: pd.DataFrame, k: int = 5, seed: int = 42):
    X = frame[cluster_cols].copy()
    # impute column medians for NA (J3 may be missing for companies with no labeled rows)
    X = X.fillna(X.median())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    lab = km.fit_predict(Xs)
    # name clusters by dominant direction: look at (entry_j3, ai_strict, desc_len, breadth_resid, scope)
    centers_orig = scaler.inverse_transform(km.cluster_centers_)
    names = []
    for c in centers_orig:
        d = dict(zip(cluster_cols, c))
        # heuristic naming
        if d["ai_prevalence_delta_strict"] > 0.25 and d["breadth_resid_delta"] > 1.0:
            names.append("ai_forward_scope_inflator")
        elif d["ai_prevalence_delta_strict"] > 0.15 and d["desc_length_delta"] > 1500:
            names.append("ai_forward")
        elif d["entry_share_delta_j3"] > 0.05 and d["ai_prevalence_delta_strict"] > 0.05:
            names.append("entry_expander")
        elif d["breadth_resid_delta"] > 0.75 and d["credential_stack_depth_delta"] > 0.25:
            names.append("scope_inflator")
        elif abs(d["ai_prevalence_delta_strict"]) < 0.1 and abs(d["entry_share_delta_j3"]) < 0.03:
            names.append("traditional_hold")
        elif d["entry_share_delta_j3"] < -0.03:
            names.append("senior_tilt")
        else:
            names.append("mixed_change")
    # de-dup name collisions with index
    seen = {}
    uniq = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        uniq.append(f"{n}_{seen[n]}" if names.count(n) > 1 else n)
    centers_named = dict(zip(range(k), uniq))
    # map cluster id -> label
    frame = frame.copy()
    frame["strategy_cluster_id"] = lab
    frame["strategy_cluster"] = [centers_named[c] for c in lab]
    return frame, centers_orig, centers_named


# cluster on pooled_min5 frame only (it's the V1-recommended primary + n=356 is big enough)
pool = vecs[vecs.panel_type == "pooled_min5"].copy()
arsh5 = vecs[vecs.panel_type == "arshkon_min5"].copy()
arsh3 = vecs[vecs.panel_type == "arshkon_min3"].copy()

pool_clustered, pool_centers, pool_names = cluster_frame(pool, k=5)
arsh5_clustered, _, _ = cluster_frame(arsh5, k=5)
arsh3_clustered, _, _ = cluster_frame(arsh3, k=5)

# recombine
clustered = pd.concat([pool_clustered, arsh5_clustered, arsh3_clustered], ignore_index=True)
# arshkon-only min3 companies not in others still get cluster via their arshkon_min3 frame
print(f"[clusters] pool clusters:\n{pool_clustered['strategy_cluster'].value_counts()}")

# Save overlap panel
overlap_out = overlap.merge(
    clustered[["company_name_canonical", "panel_type", "strategy_cluster"]],
    on=["company_name_canonical", "panel_type"],
    how="left",
)
overlap_out = overlap_out[[
    "company_name_canonical",
    "n_2024_arshkon", "n_2024_asaniczka", "n_2024_pooled", "n_2026",
    "is_aggregator", "is_entry_specialist", "strategy_cluster", "panel_type",
]]
overlap_out.to_csv(OUT_DIR / "overlap_panel.csv", index=False)
print(f"[save] overlap_panel.csv n={len(overlap_out)}")

# Save company change vectors (as specified in dispatch)
vec_out = clustered[[
    "company_name_canonical",
    "entry_share_delta_j3", "entry_share_delta_j1", "entry_share_delta_j2",
    "ai_prevalence_delta_strict", "ai_prevalence_delta_broad",
    "desc_length_delta", "tech_count_delta", "org_scope_delta",
    "breadth_resid_delta", "credential_stack_depth_delta",
    "panel_type",
]].copy()
vec_out.to_csv(OUT_DIR / "company_change_vectors.csv", index=False)
print(f"[save] company_change_vectors.csv n={len(vec_out)}")

# Save cluster centers for report
center_df = pd.DataFrame(pool_centers, columns=cluster_cols)
center_df["cluster_name"] = [pool_names[i] for i in range(len(center_df))]
center_df["n_companies"] = pool_clustered.groupby("strategy_cluster_id").size().reindex(range(len(center_df))).fillna(0).astype(int).values
center_df.to_csv(OUT_DIR / "cluster_summary.csv", index=False)
print("[save] cluster_summary.csv")

# Print centers
print("[pool cluster centers]")
print(center_df.to_string(index=False))

# Save wide state for downstream calls
wide.to_csv(OUT_DIR / "_wide_company_era.csv", index=False)
print("[save] _wide_company_era.csv")
