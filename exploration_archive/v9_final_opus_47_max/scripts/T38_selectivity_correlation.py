"""
T38 — Hiring-selectivity × scope-broadening correlation (H_N).

Test whether 2024→2026 scope-broadening is partly a selectivity response to hiring
slowdown (JOLTS 2026 Info openings 0.71× of 2023). If companies with the largest
posting-volume contraction are also writing broadest JDs, "scope broadening" is
partly "filter-raising under hiring constraint."

Steps:
1. Build per-company posting_volume_log_ratio (2026 daily rate / 2024 daily rate, log).
2. Content deltas per company (already in T16 company_change_vectors + some new).
3. Pearson + Spearman of volume-log-ratio with each content Δ.
4. Stratify by company size (large/mid/small).
5. Stratify by archetype (T09 projected nearest-centroid).
6. Robustness: exclude tech giants; exclude aggregators.

Outputs:
- exploration/tables/T38/company_volume_content.csv
- exploration/tables/T38/correlation_matrix_full.csv
- exploration/tables/T38/correlation_matrix_by_size.csv
- exploration/tables/T38/correlation_matrix_by_archetype.csv
- exploration/tables/T38/correlation_robustness.csv
- exploration/tables/T38/within_2024_baseline.csv
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
UNIFIED = str(ROOT / "data" / "unified.parquet")
CLEAN_TEXT = str(ROOT / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet")
T11_FEATS = str(ROOT / "exploration" / "artifacts" / "shared" / "T11_posting_features.parquet")
T28_CORPUS = str(ROOT / "exploration" / "tables" / "T28" / "T28_corpus_with_archetype.parquet")
OVERLAP_PANEL = str(ROOT / "exploration" / "tables" / "T16" / "overlap_panel.csv")
CHANGE_VECTORS = str(ROOT / "exploration" / "tables" / "T16" / "company_change_vectors.csv")
PATTERNS_PATH = str(ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json")

OUT = ROOT / "exploration" / "tables" / "T38"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

DEFAULT_FILTER = (
    "source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true"
)

PATTERNS = json.loads(Path(PATTERNS_PATH).read_text())
AI_STRICT = PATTERNS["ai_strict"]["pattern"]
MGMT_REBUILT = PATTERNS["v1_rebuilt_patterns"]["mgmt_strict_v1_rebuilt"]["pattern"]
SCOPE = PATTERNS["scope"]["pattern"]

# Scrape windows
W_ARSHKON_DAYS = 16  # 2024-04-05 to 2024-04-20
W_ASANICZKA_DAYS = 6  # 2024-01-12 to 2024-01-17
W_POOLED_2024_DAYS = W_ARSHKON_DAYS + W_ASANICZKA_DAYS  # 22 (different sources together)
W_SCRAPED_DAYS = 30  # 2026-03-20 to 2026-04-18
# These are duration approximations; we use them as divisors.

print("[T38] Loading panels ...")
con = duckdb.connect()

con.execute(f"""
CREATE OR REPLACE TABLE text_flags AS
SELECT
  uid,
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS ai_strict,
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS mgmt_rebuilt,
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS scope_bin,
  length(description_cleaned) AS cleaned_len
FROM '{CLEAN_TEXT}'
WHERE description_cleaned IS NOT NULL
""", [AI_STRICT, MGMT_REBUILT, SCOPE])

# archetype labels via T28 (has projected + unassigned)
con.execute(f"""
CREATE OR REPLACE VIEW arch AS
SELECT uid, archetype_primary, archetype_primary_name, archetype_source, domain
FROM '{T28_CORPUS}'
""")

# base frame (match T16 style)
con.execute(f"""
CREATE OR REPLACE TABLE base AS
SELECT
  u.uid,
  u.company_name_canonical,
  u.is_aggregator,
  u.source,
  u.period,
  u.company_size,
  u.company_industry,
  u.scrape_date,
  u.date_posted,
  CASE WHEN u.period IN ('2024-01','2024-04') THEN '2024' ELSE '2026' END AS era,
  u.yoe_min_years_llm,
  u.yoe_extracted,
  u.llm_classification_coverage,
  u.seniority_final,
  u.description_length,
  tf.ai_strict,
  tf.mgmt_rebuilt,
  tf.scope_bin,
  tf.cleaned_len,
  tc.tech_count,
  tc.requirement_breadth,
  tc.requirement_breadth_resid,
  tc.credential_stack_depth,
  a.archetype_primary_name AS archetype_name,
  a.archetype_source AS archetype_source,
  a.domain
FROM 'data/unified.parquet' u
LEFT JOIN text_flags tf USING(uid)
LEFT JOIN '{T11_FEATS}' tc USING(uid)
LEFT JOIN arch a USING(uid)
WHERE {DEFAULT_FILTER}
  AND u.company_name_canonical IS NOT NULL
""")

# Load overlap panel + change vectors
ov = pd.read_csv(OVERLAP_PANEL)
cv = pd.read_csv(CHANGE_VECTORS)
print(f"[T38] overlap_panel rows: {len(ov)}, change_vectors rows: {len(cv)}")

# Focus on arshkon-scraped overlap panel per task spec, but include pooled as sensitivity.
# Task: "Start from arshkon∩scraped overlap panel" — use panel_type == 'arshkon_min3' (243 co) or arshkon_min5 (125 co).
# Primary: arshkon_min3 per better power; co-primary pooled_min5.
panels_to_run = {
    "arshkon_min3": ov[ov.panel_type == "arshkon_min3"]["company_name_canonical"].unique(),
    "arshkon_min5": ov[ov.panel_type == "arshkon_min5"]["company_name_canonical"].unique(),
    "pooled_min5":  ov[ov.panel_type == "pooled_min5"]["company_name_canonical"].unique(),
}
for k, v in panels_to_run.items():
    print(f"  panel {k}: n_co={len(v)}")

# Per-company posting_volume: compute 2024 and 2026 posting volumes
# For each panel, use appropriate window divisors
# For arshkon_min3/min5: 2024 volume uses ARSHKON rows only (/ W_ARSHKON_DAYS)
# For pooled_min5: 2024 volume uses BOTH arshkon + asaniczka rows (summed), divided by pooled window
# 2026 always uses scraped / W_SCRAPED_DAYS
def compute_volumes(companies: list, panel: str) -> pd.DataFrame:
    q = f"""
    SELECT
      company_name_canonical,
      SUM(CASE WHEN source='kaggle_arshkon' THEN 1 ELSE 0 END) AS n_2024_arshkon,
      SUM(CASE WHEN source='kaggle_asaniczka' THEN 1 ELSE 0 END) AS n_2024_asaniczka,
      SUM(CASE WHEN source='scraped' THEN 1 ELSE 0 END) AS n_2026_scraped,
      MAX(CASE WHEN is_aggregator THEN 1 ELSE 0 END)::BOOLEAN AS is_aggregator
    FROM base
    WHERE company_name_canonical IN ({','.join([f"'{re.escape(c).replace(chr(39), chr(39)*2)}'" for c in []])})
    GROUP BY 1
    """
    # Use DuckDB registration instead for long lists
    ser = pd.DataFrame({"company_name_canonical": list(companies)})
    con.register(f"cos_{panel}", ser)
    q = f"""
    SELECT
      b.company_name_canonical,
      SUM(CASE WHEN b.source='kaggle_arshkon' THEN 1 ELSE 0 END) AS n_2024_arshkon,
      SUM(CASE WHEN b.source='kaggle_asaniczka' THEN 1 ELSE 0 END) AS n_2024_asaniczka,
      SUM(CASE WHEN b.source='scraped' THEN 1 ELSE 0 END) AS n_2026_scraped,
      MAX(CASE WHEN b.is_aggregator THEN 1 ELSE 0 END)::BOOLEAN AS is_aggregator
    FROM base b
    WHERE b.company_name_canonical IN (SELECT company_name_canonical FROM cos_{panel})
    GROUP BY 1
    """
    return con.execute(q).df()


# Build content deltas per company: we use T16 precomputed vectors for most, but also
# compute yoe_min_years_llm median delta and mentor rate delta on S1 subset (per task spec).
def compute_content_deltas(companies: list, panel: str) -> pd.DataFrame:
    ser = pd.DataFrame({"company_name_canonical": list(companies)})
    con.register(f"cos_ct_{panel}", ser)
    # S1 flag = labeled AND yoe>=3 (for mentor rate restriction per spec)
    q = f"""
    WITH enriched AS (
      SELECT b.*,
        CASE WHEN b.llm_classification_coverage='labeled' AND b.yoe_min_years_llm IS NOT NULL
             AND b.yoe_min_years_llm >= 3 THEN 1 ELSE 0 END AS is_s1
      FROM base b
      WHERE b.company_name_canonical IN (SELECT company_name_canonical FROM cos_ct_{panel})
    )
    SELECT
      company_name_canonical,
      era,
      AVG(requirement_breadth_resid)::DOUBLE AS breadth_resid_mean,
      AVG(ai_strict)::DOUBLE AS ai_strict_mean,
      AVG(mgmt_rebuilt)::DOUBLE AS mgmt_mean,
      SUM(is_s1 * mgmt_rebuilt)::DOUBLE / NULLIF(SUM(is_s1),0) AS mentor_on_s1_rate,
      MEDIAN(description_length)::DOUBLE AS desc_len_median,
      MEDIAN(yoe_min_years_llm)::DOUBLE AS yoe_llm_median,
      MEDIAN(yoe_extracted)::DOUBLE AS yoe_rule_median,
      AVG(tech_count)::DOUBLE AS tech_count_mean,
      AVG(scope_bin)::DOUBLE AS scope_mean,
      AVG(credential_stack_depth)::DOUBLE AS credstack_mean,
      SUM(CASE WHEN era='2024' AND source='kaggle_arshkon' THEN 1 ELSE 0 END) AS n_arshkon,
      SUM(CASE WHEN era='2024' AND source='kaggle_asaniczka' THEN 1 ELSE 0 END) AS n_asaniczka,
      SUM(CASE WHEN era='2026' THEN 1 ELSE 0 END) AS n_scraped,
      MAX(CASE WHEN is_aggregator THEN 1 ELSE 0 END)::BOOLEAN AS is_aggregator
    FROM enriched
    GROUP BY 1, 2
    """
    df = con.execute(q).df()
    # pivot to wide
    cols_num = ["breadth_resid_mean", "ai_strict_mean", "mgmt_mean", "mentor_on_s1_rate",
                "desc_len_median", "yoe_llm_median", "yoe_rule_median", "tech_count_mean",
                "scope_mean", "credstack_mean"]
    pv = df.pivot(index="company_name_canonical", columns="era", values=cols_num).fillna(np.nan)
    pv.columns = [f"{a}_{b}" for a, b in pv.columns]
    pv = pv.reset_index()
    for col in cols_num:
        pv[f"{col}_delta"] = pv[f"{col}_2026"] - pv[f"{col}_2024"]
    # attach aggregator
    meta = df.groupby("company_name_canonical").agg(is_aggregator=("is_aggregator","max")).reset_index()
    pv = pv.merge(meta, on="company_name_canonical", how="left")
    return pv


def log_safe(x):
    return np.log(np.where(x > 0, x, np.nan))


def corr_row(x, y, name):
    mask = (~pd.isna(x)) & (~pd.isna(y))
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    n = len(x)
    if n < 5:
        return {"metric": name, "n": n, "pearson_r": np.nan, "pearson_p": np.nan,
                "pearson_lo": np.nan, "pearson_hi": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan}
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    # Fisher z-CI for pearson
    z = np.arctanh(pr)
    se = 1.0 / np.sqrt(max(n - 3, 1))
    lo = np.tanh(z - 1.96 * se)
    hi = np.tanh(z + 1.96 * se)
    return {"metric": name, "n": n, "pearson_r": pr, "pearson_p": pp,
            "pearson_lo": lo, "pearson_hi": hi, "spearman_r": sr, "spearman_p": sp}


def build_panel_df(panel_name: str) -> pd.DataFrame:
    cos = panels_to_run[panel_name]
    vol = compute_volumes(cos, panel_name)
    cd = compute_content_deltas(cos, panel_name)
    d = vol.merge(cd.drop(columns=[c for c in ["is_aggregator"] if c in cd.columns]), on="company_name_canonical", how="inner")
    # volume denominators
    if panel_name.startswith("arshkon"):
        d["vol_2024"] = d["n_2024_arshkon"] / W_ARSHKON_DAYS
    else:
        # pooled: arshkon/asaniczka separate windows — use max of two daily rates for 2024 (conservative)
        d["vol_2024_arshkon_daily"] = d["n_2024_arshkon"] / W_ARSHKON_DAYS
        d["vol_2024_asaniczka_daily"] = d["n_2024_asaniczka"] / W_ASANICZKA_DAYS
        # Use POOLED DAILY RATE: average of the two rates where both present; else non-zero
        d["vol_2024"] = d[["vol_2024_arshkon_daily", "vol_2024_asaniczka_daily"]].mean(axis=1, skipna=True)
    d["vol_2026"] = d["n_2026_scraped"] / W_SCRAPED_DAYS
    # log ratio with +0.5 Laplace smoothing for zero-cells
    d["vol_log_ratio"] = np.log((d["vol_2026"] + 0.5/W_SCRAPED_DAYS) / (d["vol_2024"] + 0.5/W_ARSHKON_DAYS))
    # company_size snapshot (from 2024 or 2026 if available): use max across company
    sz_q = f"""
    SELECT company_name_canonical, MAX(company_size) AS cmax,
           AVG(company_size) AS cavg
    FROM base
    WHERE company_name_canonical IN (SELECT company_name_canonical FROM cos_{panel_name})
      AND company_size IS NOT NULL
    GROUP BY 1
    """
    sz = con.execute(sz_q).df()
    d = d.merge(sz, on="company_name_canonical", how="left")
    d["size_class"] = np.select(
        [d["cmax"] >= 10000, (d["cmax"] >= 1000) & (d["cmax"] < 10000), d["cmax"] < 1000],
        ["large", "mid", "small"], default="unknown"
    )
    # archetype: majority archetype across company postings (mode; source-aware only)
    arch_q = f"""
    SELECT company_name_canonical, archetype_name,
           COUNT(*) AS n
    FROM base
    WHERE company_name_canonical IN (SELECT company_name_canonical FROM cos_{panel_name})
      AND archetype_name IS NOT NULL AND archetype_name != 'noise/outlier'
    GROUP BY 1, 2
    """
    arch = con.execute(arch_q).df()
    arch["rk"] = arch.groupby("company_name_canonical")["n"].rank(ascending=False, method="first")
    arch_top = arch[arch.rk == 1][["company_name_canonical", "archetype_name"]]
    d = d.merge(arch_top, on="company_name_canonical", how="left")
    return d


# ---- Tech-giant regex ----
TECH_GIANTS_RE = re.compile(r"^(google|amazon|amazon web services|aws|microsoft|meta|apple|apple inc|facebook|alphabet|waymo|deepmind|anthropic|openai)(\s|,|$)", re.I)

def is_tech_giant(name):
    if pd.isna(name): return False
    return bool(TECH_GIANTS_RE.match(name))


# Build per-panel dataframes
print("\n[T38] Build per-panel company-level dataframes ...")
panel_dfs = {}
for pn in ["arshkon_min3", "arshkon_min5", "pooled_min5"]:
    print(f"  {pn} ...")
    dfp = build_panel_df(pn)
    dfp["is_tech_giant"] = dfp["company_name_canonical"].apply(is_tech_giant)
    panel_dfs[pn] = dfp
    print(f"    n={len(dfp)}, tech_giants={dfp.is_tech_giant.sum()}, aggregators={dfp.is_aggregator.sum()}")

# Save the primary panel
panel_dfs["arshkon_min3"].to_csv(OUT / "company_volume_content_arshkon_min3.csv", index=False)
panel_dfs["arshkon_min5"].to_csv(OUT / "company_volume_content_arshkon_min5.csv", index=False)
panel_dfs["pooled_min5"].to_csv(OUT / "company_volume_content_pooled_min5.csv", index=False)

# ---- Correlation matrix primary (arshkon_min3) ----
print("\n[T38] Correlation matrix primary (arshkon_min3) ...")
d_prim = panel_dfs["arshkon_min3"].copy()
content_metrics = [
    ("breadth_resid_mean_delta", "breadth_resid_delta"),
    ("ai_strict_mean_delta", "ai_strict_delta"),
    ("mentor_on_s1_rate_delta", "mentor_on_S1_delta"),
    ("desc_len_median_delta", "desc_len_median_delta"),
    ("yoe_llm_median_delta", "yoe_min_years_llm_delta"),
    ("yoe_rule_median_delta", "yoe_extracted_delta"),
    ("tech_count_mean_delta", "tech_count_delta"),
    ("scope_mean_delta", "scope_v1_delta"),
    ("credstack_mean_delta", "credential_stack_depth_delta"),
    ("mgmt_mean_delta", "mgmt_rebuilt_delta"),
]
def compute_corr_table(df, content_metrics, panel_label):
    rows = []
    for col, name in content_metrics:
        row = corr_row(df["vol_log_ratio"], df[col], name)
        row["panel"] = panel_label
        rows.append(row)
    return pd.DataFrame(rows)

full_prim = compute_corr_table(d_prim, content_metrics, "arshkon_min3")
full_prim.to_csv(OUT / "correlation_matrix_full_arshkon_min3.csv", index=False)
print(full_prim.to_string(index=False))

# Also pooled_min5 and arshkon_min5
full_sec = compute_corr_table(panel_dfs["pooled_min5"], content_metrics, "pooled_min5")
full_sec.to_csv(OUT / "correlation_matrix_full_pooled_min5.csv", index=False)
print("\npooled_min5:")
print(full_sec.to_string(index=False))
full_arsh5 = compute_corr_table(panel_dfs["arshkon_min5"], content_metrics, "arshkon_min5")
full_arsh5.to_csv(OUT / "correlation_matrix_full_arshkon_min5.csv", index=False)
print("\narshkon_min5:")
print(full_arsh5.to_string(index=False))

# ---- Stratify by size ----
print("\n[T38] By company size stratification (arshkon_min3 primary)")
size_rows = []
for size in ["large", "mid", "small"]:
    sub = d_prim[d_prim.size_class == size]
    if len(sub) < 5:
        continue
    cm = compute_corr_table(sub, content_metrics, f"arshkon_min3_{size}")
    cm["size_class"] = size
    cm["n_cos"] = len(sub)
    size_rows.append(cm)
size_df = pd.concat(size_rows, ignore_index=True)
size_df.to_csv(OUT / "correlation_matrix_by_size.csv", index=False)
print(size_df.to_string(index=False))

# ---- Stratify by archetype ----
print("\n[T38] By archetype stratification (arshkon_min3 primary)")
arch_counts = d_prim["archetype_name"].value_counts()
print(arch_counts.head(15))
arch_rows = []
for arch in d_prim["archetype_name"].dropna().unique():
    sub = d_prim[d_prim.archetype_name == arch]
    if len(sub) < 8:
        continue
    cm = compute_corr_table(sub, content_metrics, f"arshkon_min3_{arch}")
    cm["archetype"] = arch
    cm["n_cos"] = len(sub)
    arch_rows.append(cm)
arch_df = pd.concat(arch_rows, ignore_index=True) if arch_rows else pd.DataFrame()
arch_df.to_csv(OUT / "correlation_matrix_by_archetype.csv", index=False)
print(arch_df.head(40).to_string(index=False) if len(arch_df) > 0 else "(no archetypes with n>=8)")

# ---- Robustness: exclude tech giants ----
print("\n[T38] Robustness: exclude tech giants (arshkon_min3)")
d_no_giants = d_prim[~d_prim.is_tech_giant]
print(f"  giants removed: {d_prim.is_tech_giant.sum()}; n remaining: {len(d_no_giants)}")
giants_df = compute_corr_table(d_no_giants, content_metrics, "arshkon_min3_no_giants")
giants_df["robustness"] = "no_tech_giants"
print(giants_df.to_string(index=False))

print("\n[T38] Robustness: exclude aggregators (arshkon_min3)")
d_no_agg = d_prim[~d_prim.is_aggregator]
print(f"  aggregators removed: {d_prim.is_aggregator.sum()}; n remaining: {len(d_no_agg)}")
agg_df = compute_corr_table(d_no_agg, content_metrics, "arshkon_min3_no_aggregators")
agg_df["robustness"] = "no_aggregators"
print(agg_df.to_string(index=False))

# combined robustness
print("\n[T38] Robustness: exclude BOTH tech giants and aggregators (arshkon_min3)")
d_both = d_prim[(~d_prim.is_tech_giant) & (~d_prim.is_aggregator)]
print(f"  combined removed; n remaining: {len(d_both)}")
both_df = compute_corr_table(d_both, content_metrics, "arshkon_min3_no_giants_no_aggregators")
both_df["robustness"] = "no_giants_no_aggregators"
print(both_df.to_string(index=False))

robust_all = pd.concat([full_prim.assign(robustness="full"), giants_df, agg_df, both_df], ignore_index=True)
robust_all.to_csv(OUT / "correlation_robustness.csv", index=False)

# ---- Within-2024 baseline (per task essential sensitivity (f)) ----
# Split 2024 into arshkon-only vs asaniczka-only per-company content; compute a pseudo-volume-ratio
# (arshkon daily rate vs asaniczka daily rate) and correlate with pseudo-content-delta.
# This reproduces the T38 correlation but in a within-2024 context (no period confound).
print("\n[T38] Within-2024 baseline: arshkon vs asaniczka within-2024 calibration")
q = f"""
WITH enr AS (
  SELECT b.*
  FROM base b
  WHERE era = '2024'
)
SELECT
  company_name_canonical,
  source,
  COUNT(*) AS n,
  AVG(requirement_breadth_resid)::DOUBLE AS breadth_resid_mean,
  AVG(ai_strict)::DOUBLE AS ai_strict_mean,
  AVG(mgmt_rebuilt)::DOUBLE AS mgmt_mean,
  MEDIAN(description_length)::DOUBLE AS desc_len_median,
  MEDIAN(yoe_min_years_llm)::DOUBLE AS yoe_llm_median,
  AVG(tech_count)::DOUBLE AS tech_count_mean,
  AVG(scope_bin)::DOUBLE AS scope_mean,
  MAX(is_aggregator)::BOOLEAN AS is_aggregator
FROM enr
GROUP BY 1, 2
"""
w24 = con.execute(q).df()
# pivot
cols_24 = ["breadth_resid_mean", "ai_strict_mean", "mgmt_mean", "desc_len_median",
           "yoe_llm_median", "tech_count_mean", "scope_mean", "n"]
pv = w24.pivot(index="company_name_canonical", columns="source", values=cols_24)
pv.columns = [f"{a}_{b}" for a, b in pv.columns]
pv = pv.reset_index()
# Require both arshkon and asaniczka rows (n >= 3)
req = (pv.get("n_kaggle_arshkon", 0) >= 3) & (pv.get("n_kaggle_asaniczka", 0) >= 3)
pv = pv[req].copy()
# fake volume ratio: log(arshkon daily rate / asaniczka daily rate) as pseudo
pv["pseudo_vol_log_ratio"] = np.log((pv["n_kaggle_arshkon"] / W_ARSHKON_DAYS + 0.01) /
                                     (pv["n_kaggle_asaniczka"] / W_ASANICZKA_DAYS + 0.01))
# pseudo content deltas
for m in ["breadth_resid_mean","ai_strict_mean","mgmt_mean","desc_len_median",
          "yoe_llm_median","tech_count_mean","scope_mean"]:
    pv[f"{m}_delta_within_2024"] = pv[f"{m}_kaggle_arshkon"] - pv[f"{m}_kaggle_asaniczka"]
# Compute correlation
w24_rows = []
for m in ["breadth_resid_mean","ai_strict_mean","mgmt_mean","desc_len_median",
          "yoe_llm_median","tech_count_mean","scope_mean"]:
    row = corr_row(pv["pseudo_vol_log_ratio"], pv[f"{m}_delta_within_2024"], m)
    w24_rows.append(row)
w24_df = pd.DataFrame(w24_rows)
w24_df["cohort"] = "within_2024_arshkon_vs_asaniczka"
w24_df.to_csv(OUT / "within_2024_baseline.csv", index=False)
print(w24_df.to_string(index=False))

# ---- T30 panel seniority split on primary correlation ----
print("\n[T38] Seniority-stratified sub-analysis (arshkon_min3, breadth_resid within J3 and S4)")
q_strat = f"""
WITH enr AS (
  SELECT b.*,
    CASE WHEN b.llm_classification_coverage='labeled' AND b.yoe_min_years_llm IS NOT NULL
         AND b.yoe_min_years_llm <= 2 THEN 1 ELSE 0 END AS is_j3,
    CASE WHEN b.llm_classification_coverage='labeled' AND b.yoe_min_years_llm IS NOT NULL
         AND b.yoe_min_years_llm >= 5 THEN 1 ELSE 0 END AS is_s4
  FROM base b
  WHERE b.company_name_canonical IN (SELECT company_name_canonical FROM cos_ct_arshkon_min3)
)
SELECT company_name_canonical, era,
       AVG(CASE WHEN is_j3=1 THEN requirement_breadth_resid END) AS breadth_resid_J3,
       AVG(CASE WHEN is_s4=1 THEN requirement_breadth_resid END) AS breadth_resid_S4,
       AVG(CASE WHEN is_j3=1 THEN ai_strict END) AS ai_strict_J3,
       AVG(CASE WHEN is_s4=1 THEN ai_strict END) AS ai_strict_S4
FROM enr GROUP BY 1,2
"""
strat = con.execute(q_strat).df()
pvs = strat.pivot(index="company_name_canonical", columns="era")
pvs.columns = [f"{a}_{b}" for a, b in pvs.columns]
pvs = pvs.reset_index()
for m in ["breadth_resid_J3", "breadth_resid_S4", "ai_strict_J3", "ai_strict_S4"]:
    pvs[f"{m}_delta"] = pvs[f"{m}_2026"] - pvs[f"{m}_2024"]
merged = d_prim[["company_name_canonical","vol_log_ratio"]].merge(pvs, on="company_name_canonical", how="inner")
strat_rows = []
for m in ["breadth_resid_J3_delta", "breadth_resid_S4_delta", "ai_strict_J3_delta", "ai_strict_S4_delta"]:
    strat_rows.append(corr_row(merged["vol_log_ratio"], merged[m], m))
strat_df = pd.DataFrame(strat_rows)
strat_df.to_csv(OUT / "correlation_by_seniority.csv", index=False)
print(strat_df.to_string(index=False))

# ---- Directionality summary ----
def summarize(df, panel_name):
    # For each metric, report sign and significance
    out = df.copy()
    out["dir"] = np.select([out.pearson_r < -0.1, out.pearson_r > 0.1],
                            ["negative (selectivity-consistent)", "positive (reverse)"],
                            default="null")
    out["sig"] = out.pearson_p < 0.05
    return out[["metric","n","pearson_r","pearson_lo","pearson_hi","pearson_p","spearman_r","spearman_p","dir","sig"]]

print("\n=== DIRECTIONALITY: arshkon_min3 primary ===")
print(summarize(full_prim, "arshkon_min3").to_string(index=False))

print("\n=== DIRECTIONALITY: arshkon_min3 no_aggregators ===")
print(summarize(agg_df, "arshkon_min3_no_aggregators").to_string(index=False))

print("\n=== DIRECTIONALITY: arshkon_min3 no_giants_no_aggregators ===")
print(summarize(both_df, "arshkon_min3_no_giants_no_aggregators").to_string(index=False))

# Save key figure data (arshkon_min3 no_giants_no_aggregators final primary)
pd.concat([
    full_prim.assign(panel="full"),
    giants_df.assign(panel="no_giants"),
    agg_df.assign(panel="no_aggregators"),
    both_df.assign(panel="no_giants_no_aggregators"),
    full_sec.assign(panel="pooled_min5"),
    full_arsh5.assign(panel="arshkon_min5"),
    w24_df.rename(columns={"cohort":"panel"}),
], ignore_index=True).to_csv(OUT / "correlation_all.csv", index=False)

print("\n[T38] ALL DONE.")
