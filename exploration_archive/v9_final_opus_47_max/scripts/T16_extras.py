"""
T16 extras — within-company scope J3 vs S4, new market entrants, aggregator split, sensitivities.
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
T11_FEATS = str(ROOT / "exploration" / "artifacts" / "shared" / "T11_posting_features.parquet")
ARCHETYPE = str(ROOT / "exploration" / "artifacts" / "shared" / "swe_archetype_labels.parquet")
PATTERNS_PATH = str(ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json")
ENTRY_SPEC = str(ROOT / "exploration" / "artifacts" / "shared" / "entry_specialist_employers.csv")
OUT_DIR = ROOT / "exploration" / "tables" / "T16"

PATTERNS = json.loads(Path(PATTERNS_PATH).read_text())
AI_STRICT = PATTERNS["ai_strict"]["pattern"]
SCOPE = PATTERNS["scope"]["pattern"]
MGMT_REBUILT = PATTERNS["v1_rebuilt_patterns"]["mgmt_strict_v1_rebuilt"]["pattern"]

DEFAULT_FILTER = (
    "source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true"
)

con = duckdb.connect()

con.execute(f"""
CREATE OR REPLACE TABLE tf AS
SELECT
  uid,
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS ai_strict,
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS mgmt_rebuilt,
  CAST(list_reduce(
    list_transform(regexp_extract_all(lower(description_cleaned), ?), x -> 1),
    (acc, v) -> acc + v, 0
  ) AS BIGINT) AS scope_term_count
FROM '{CLEAN_TEXT}'
""", [AI_STRICT, MGMT_REBUILT, SCOPE])

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
  COALESCE(tf.ai_strict,0) AS ai_strict,
  COALESCE(tf.mgmt_rebuilt,0) AS mgmt_rebuilt,
  COALESCE(tf.scope_term_count,0) AS scope_term_count,
  COALESCE(tc.tech_count,0) AS tech_count,
  tc.requirement_breadth_resid,
  tc.credential_stack_depth,
  CASE WHEN u.llm_classification_coverage='labeled' AND u.yoe_min_years_llm IS NOT NULL
         AND u.yoe_min_years_llm <= 2 THEN 1 ELSE 0 END AS j3_flag,
  CASE WHEN u.llm_classification_coverage='labeled' AND u.yoe_min_years_llm IS NOT NULL
         AND u.yoe_min_years_llm >= 5 THEN 1 ELSE 0 END AS s4_flag,
  CASE WHEN u.llm_classification_coverage='labeled' AND u.yoe_min_years_llm IS NOT NULL
       THEN 1 ELSE 0 END AS labeled
FROM '{UNIFIED}' u
LEFT JOIN tf USING (uid)
LEFT JOIN '{T11_FEATS}' tc USING (uid)
WHERE {DEFAULT_FILTER}
  AND u.company_name_canonical IS NOT NULL
""")

# panels
cmp = con.execute("""
SELECT company_name_canonical,
  SUM(CASE WHEN source='kaggle_arshkon' THEN 1 ELSE 0 END) AS n_arshkon,
  SUM(CASE WHEN source IN ('kaggle_arshkon','kaggle_asaniczka') THEN 1 ELSE 0 END) AS n_pooled,
  SUM(CASE WHEN source='scraped' THEN 1 ELSE 0 END) AS n_scraped
FROM base GROUP BY 1
""").df()

arshkon_min5 = set(cmp.loc[(cmp.n_arshkon >= 5) & (cmp.n_scraped >= 5), "company_name_canonical"])
pooled_min5 = set(cmp.loc[(cmp.n_pooled >= 5) & (cmp.n_scraped >= 5), "company_name_canonical"])

# ----------------------------------------------------------------------------
# Step 5: Within-company scope inflation — J3 vs S4 on same-co restriction
# ----------------------------------------------------------------------------
def within_co_change(df, metric_col, group_col):
    """Per group_col (seniority), compute same-company within-co change using 2024/2026 means."""
    g = df.groupby(["company_name_canonical", group_col, "era"]).agg(
        n=("uid", "size"),
        m=(metric_col, "mean"),
    ).reset_index()
    w = g.pivot(index=["company_name_canonical", group_col], columns="era", values=["n", "m"]).fillna(0)
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    if "n_2024" not in w or "n_2026" not in w:
        return None
    both = w[(w["n_2024"] > 0) & (w["n_2026"] > 0)].copy()
    if both.empty:
        return None
    # weight by symmetric volume
    nbar = 0.5 * (both["n_2024"] + both["n_2026"])
    within = float((nbar * (both["m_2026"] - both["m_2024"])).sum() / nbar.sum())
    return within


# per panel × per seniority
rows = []
for panel_name, comps in [("arshkon_min5", arshkon_min5), ("pooled_min5", pooled_min5)]:
    src_filter = "arshkon" if panel_name.startswith("arshkon") else None
    df = con.execute("""
    SELECT uid, company_name_canonical, source, era,
           j3_flag, s4_flag, scope_term_count, mgmt_rebuilt, requirement_breadth_resid
    FROM base
    """).df()
    df = df[df.company_name_canonical.isin(comps)]
    if src_filter == "arshkon":
        df = df[(df.era == "2026") | (df.source == "kaggle_arshkon")]
    # J3 within
    d_j3 = df[df.j3_flag == 1].copy()
    d_s4 = df[df.s4_flag == 1].copy()
    for metric in ["scope_term_count", "mgmt_rebuilt", "requirement_breadth_resid"]:
        # per-company mean per era
        for level, dd in [("J3", d_j3), ("S4", d_s4)]:
            if dd.empty:
                continue
            g = dd.groupby(["company_name_canonical", "era"]).agg(n=("uid", "size"), m=(metric, "mean")).reset_index()
            w = g.pivot(index="company_name_canonical", columns="era", values=["n", "m"]).fillna(0)
            w.columns = [f"{a}_{b}" for a, b in w.columns]
            if "n_2024" not in w or "n_2026" not in w:
                continue
            both = w[(w["n_2024"] > 0) & (w["n_2026"] > 0)].copy()
            if both.empty:
                continue
            nbar = 0.5 * (both["n_2024"] + both["n_2026"])
            within = float((nbar * (both["m_2026"] - both["m_2024"])).sum() / nbar.sum())
            mean24 = float((both["n_2024"] * both["m_2024"]).sum() / both["n_2024"].sum())
            mean26 = float((both["n_2026"] * both["m_2026"]).sum() / both["n_2026"].sum())
            rows.append({
                "panel": panel_name,
                "seniority": level,
                "metric": metric,
                "within_co_delta": within,
                "n_both_eras_cos": int(len(both)),
                "mean_2024": mean24,
                "mean_2026": mean26,
                "total_delta": mean26 - mean24,
            })

same_co_sen = pd.DataFrame(rows)
same_co_sen.to_csv(OUT_DIR / "scope_same_co_senior_vs_junior.csv", index=False)
print(f"[save] scope_same_co_senior_vs_junior.csv n={len(same_co_sen)}")
print(same_co_sen.to_string(index=False))
print()

# ----------------------------------------------------------------------------
# Step 6: New market entrants (companies in 2026 with no 2024 match)
# ----------------------------------------------------------------------------
df_all = con.execute("""
SELECT uid, company_name_canonical, is_aggregator, era, source, description_length,
       ai_strict, mgmt_rebuilt, scope_term_count, tech_count,
       j3_flag, s4_flag, labeled, requirement_breadth_resid, credential_stack_depth
FROM base
""").df()

cmp_eras = df_all.groupby("company_name_canonical").agg(
    has_2024=("era", lambda s: (s == "2024").any()),
    has_2026=("era", lambda s: (s == "2026").any()),
    n_total=("uid", "size"),
    is_agg=("is_aggregator", "any"),
).reset_index()

new_cos = set(cmp_eras.loc[(~cmp_eras.has_2024) & (cmp_eras.has_2026), "company_name_canonical"])
returning = set(cmp_eras.loc[cmp_eras.has_2024 & cmp_eras.has_2026, "company_name_canonical"])
only24 = set(cmp_eras.loc[cmp_eras.has_2024 & (~cmp_eras.has_2026), "company_name_canonical"])
print(f"[entrants] new_cos={len(new_cos)} returning={len(returning)} only24={len(only24)}")

# 2026 rows only
d26 = df_all[df_all.era == "2026"].copy()
d26["cohort"] = np.where(d26.company_name_canonical.isin(returning), "returning",
                    np.where(d26.company_name_canonical.isin(new_cos), "new_entrant", "other"))

# J3 share for new vs returning
summary = []
for cohort in ["returning", "new_entrant"]:
    sub = d26[d26.cohort == cohort]
    lab = sub[sub.labeled == 1]
    n = len(sub)
    n_co = sub.company_name_canonical.nunique()
    j3_share = lab.j3_flag.mean() if not lab.empty else np.nan
    s4_share = lab.s4_flag.mean() if not lab.empty else np.nan
    summary.append({
        "cohort": cohort,
        "n_postings_2026": n,
        "n_companies": n_co,
        "j3_share_labeled": j3_share,
        "s4_share_labeled": s4_share,
        "ai_strict_prev": sub.ai_strict.mean(),
        "mgmt_rebuilt_prev": sub.mgmt_rebuilt.mean(),
        "scope_term_count_mean": sub.scope_term_count.mean(),
        "desc_length_mean": sub.description_length.mean(),
        "tech_count_mean": sub.tech_count.mean(),
        "breadth_resid_mean": sub.requirement_breadth_resid.mean(),
        "credstack_mean": sub.credential_stack_depth.mean(),
        "aggregator_share": sub.is_aggregator.mean(),
    })

new_vs_ret = pd.DataFrame(summary)
new_vs_ret.to_csv(OUT_DIR / "new_entrants_vs_returning.csv", index=False)
print("[save] new_entrants_vs_returning.csv")
print(new_vs_ret.to_string(index=False))
print()

# Industry profile of new entrants (if we pull company_industry)
ind_q = """
SELECT company_name_canonical,
       MAX(company_industry) AS company_industry
FROM 'data/unified.parquet'
GROUP BY 1
"""
ind = con.execute(ind_q).df()
ind["cohort"] = np.where(ind.company_name_canonical.isin(returning), "returning",
                    np.where(ind.company_name_canonical.isin(new_cos), "new_entrant", "other"))
top_ind = ind[ind.cohort == "new_entrant"].company_industry.value_counts().head(15).reset_index()
top_ind.columns = ["industry", "n_new_entrant_companies"]
top_ind.to_csv(OUT_DIR / "new_entrant_industries.csv", index=False)
print("[save] new_entrant_industries.csv")

# ----------------------------------------------------------------------------
# Step 7: Aggregator vs direct employer comparison (returning cohort)
# ----------------------------------------------------------------------------
# Use pooled_min5 panel
pool_df = df_all[df_all.company_name_canonical.isin(pooled_min5)].copy()
agg_by = []
for agg_flag, agg_name in [(True, "aggregator"), (False, "direct")]:
    sub = pool_df[pool_df.is_aggregator == agg_flag]
    for era_v in ["2024", "2026"]:
        s = sub[sub.era == era_v]
        lab = s[s.labeled == 1]
        agg_by.append({
            "cohort": agg_name,
            "era": era_v,
            "n_postings": len(s),
            "n_companies": s.company_name_canonical.nunique(),
            "j3_share_labeled": lab.j3_flag.mean() if not lab.empty else np.nan,
            "ai_strict_prev": s.ai_strict.mean(),
            "scope_term_count_mean": s.scope_term_count.mean(),
            "desc_length_mean": s.description_length.mean(),
            "tech_count_mean": s.tech_count.mean(),
            "breadth_resid_mean": s.requirement_breadth_resid.mean(),
        })

agg_df = pd.DataFrame(agg_by)
agg_df.to_csv(OUT_DIR / "aggregator_vs_direct.csv", index=False)
print("[save] aggregator_vs_direct.csv")
print(agg_df.to_string(index=False))

# aggregator-exclusion sensitivity: pooled_min5 J3 decomposition dropping aggregators
nonagg = pool_df[~pool_df.is_aggregator].copy()
nonagg_cos = nonagg.company_name_canonical.unique()
print(f"\n[sensitivity] pooled_min5 non-aggregator companies: {nonagg.company_name_canonical.nunique()}")

# decompose on non-aggregator subset for J3, ai, scope, breadth_resid
def simple_decompose(df, metric, denom=None):
    if denom is not None:
        g = df.groupby(["company_name_canonical", "era"]).agg(
            n=(denom, "sum"), m=(metric, "sum")
        ).reset_index()
    else:
        g = df.groupby(["company_name_canonical", "era"]).agg(
            n=(metric, "size"), m=(metric, "sum")
        ).reset_index()
    g["mean"] = g["m"] / g["n"].replace(0, np.nan)
    w = g.pivot(index="company_name_canonical", columns="era", values=["n", "mean"]).fillna(0)
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    den24 = w["n_2024"].sum()
    den26 = w["n_2026"].sum()
    agg24 = (w["n_2024"] * w["mean_2024"].fillna(0)).sum() / den24 if den24 else np.nan
    agg26 = (w["n_2026"] * w["mean_2026"].fillna(0)).sum() / den26 if den26 else np.nan
    total = agg26 - agg24
    s24 = w["n_2024"] / den24
    s26 = w["n_2026"] / den26
    sbar = 0.5 * (s24 + s26)
    both = w[(w["n_2024"] > 0) & (w["n_2026"] > 0)]
    within = float((sbar.loc[both.index] * (w.loc[both.index, "mean_2026"] - w.loc[both.index, "mean_2024"])).sum())
    mbar = 0.5 * (w["mean_2024"].fillna(0) + w["mean_2026"].fillna(0))
    only26 = w.index[(w["n_2024"] == 0) & (w["n_2026"] > 0)]
    only24 = w.index[(w["n_2024"] > 0) & (w["n_2026"] == 0)]
    mbar.loc[only26] = w.loc[only26, "mean_2026"]
    mbar.loc[only24] = w.loc[only24, "mean_2024"]
    between = float(((s26 - s24) * mbar).sum())
    return dict(agg_2024=agg24, agg_2026=agg26, total=total, within=within, between=between,
                n_both=int(len(both)))

sens_rows = []
for frame_name, frame in [("pooled_min5", pool_df), ("pooled_min5_no_agg", nonagg)]:
    for label, metric, denom in [
        ("entry_share_j3", "j3_flag", "labeled"),
        ("ai_strict", "ai_strict", None),
        ("scope_term_count", "scope_term_count", None),
        ("breadth_resid", "requirement_breadth_resid", None),
        ("desc_length", "description_length", None),
    ]:
        d = frame.dropna(subset=[metric])
        res = simple_decompose(d, metric, denom)
        sens_rows.append({"panel": frame_name, "metric": label, **res})

sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(OUT_DIR / "sensitivity_aggregator_exclusion.csv", index=False)
print("[save] sensitivity_aggregator_exclusion.csv")
print(sens_df.to_string(index=False))
