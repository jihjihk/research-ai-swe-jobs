"""
T16 step 4 — Within-company vs between-company decomposition.

For each metric and panel, compute:
  aggregate Δ  = Σ_i (p_i2026*share_i2026) - Σ_i (p_i2024*share_i2024)
  within-co Δ  = Σ_i p_i2026_share * (metric_i2026 - metric_i2024)        [composition held at 2026 weights]
  between-co Δ = aggregate - within

  Also report within via 2024 weights, then report symmetric (mean).

For entry_share we use J3 (labeled denominator) + J1/J2 (seniority_final).
For ai_strict / desc_length / breadth_resid, standard.

Arshkon∩scraped and pooled∩scraped separately.
If T09 archetype labels available, add within-domain / between-domain layer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UNIFIED = str(ROOT / "data" / "unified.parquet")
CLEAN_TEXT = str(ROOT / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet")
T11_FEATS = str(ROOT / "exploration" / "artifacts" / "shared" / "T11_posting_features.parquet")
ARCHETYPE = str(ROOT / "exploration" / "artifacts" / "shared" / "swe_archetype_labels.parquet")
PATTERNS_PATH = str(ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json")
OUT_DIR = ROOT / "exploration" / "tables" / "T16"

PATTERNS = json.loads(Path(PATTERNS_PATH).read_text())
AI_STRICT = PATTERNS["ai_strict"]["pattern"]
MGMT_REBUILT = PATTERNS["v1_rebuilt_patterns"]["mgmt_strict_v1_rebuilt"]["pattern"]
SCOPE = PATTERNS["scope"]["pattern"]

DEFAULT_FILTER = (
    "source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true"
)

con = duckdb.connect()

# Recreate base posting-level frame with all metrics and company_name_canonical.
con.execute(f"""
CREATE OR REPLACE TABLE tf AS
SELECT
  uid,
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS ai_strict,
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS mgmt_rebuilt,
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS scope_bin,
  CAST(list_reduce(
    list_transform(regexp_extract_all(lower(description_cleaned), ?), x -> 1),
    (acc, v) -> acc + v, 0
  ) AS BIGINT) AS scope_term_count,
  length(description_cleaned) AS cleaned_len
FROM '{CLEAN_TEXT}'
""", [AI_STRICT, MGMT_REBUILT, SCOPE, SCOPE])

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
  COALESCE(tf.scope_bin,0) AS scope_bin,
  COALESCE(tf.scope_term_count,0) AS scope_term_count,
  tf.cleaned_len,
  COALESCE(tc.tech_count,0) AS tech_count,
  tc.requirement_breadth_resid,
  tc.credential_stack_depth,
  -- entry flags
  CASE WHEN u.seniority_final='entry' THEN 1 ELSE 0 END AS j1_entry,
  CASE WHEN u.seniority_final='associate' THEN 1 ELSE 0 END AS j2_assoc,
  CASE
    WHEN u.llm_classification_coverage='labeled' AND u.yoe_min_years_llm IS NOT NULL
         AND u.yoe_min_years_llm <= 2 THEN 1 ELSE 0 END AS j3_flag,
  CASE WHEN u.llm_classification_coverage='labeled' AND u.yoe_min_years_llm IS NOT NULL
       THEN 1 ELSE 0 END AS j3_denom,
  CASE WHEN u.llm_classification_coverage='labeled' AND u.yoe_min_years_llm IS NOT NULL
         AND u.yoe_min_years_llm >= 5 THEN 1 ELSE 0 END AS s4_flag,
  -- archetype join
  arc.archetype_name AS archetype_name
FROM '{UNIFIED}' u
LEFT JOIN tf USING (uid)
LEFT JOIN '{T11_FEATS}' tc USING (uid)
LEFT JOIN '{ARCHETYPE}' arc USING (uid)
WHERE {DEFAULT_FILTER}
  AND u.company_name_canonical IS NOT NULL
""")

n_base = con.execute("SELECT COUNT(*) FROM base").fetchone()[0]
print(f"[base] {n_base} rows")


# -----------------------------------------------------------------------
# Helper: within-between decomposition using symmetric (Oaxaca-like) weights
# -----------------------------------------------------------------------
def decompose(df_post: pd.DataFrame, metric: str, denom_col: str | None = None):
    """df_post has per-posting rows with columns [company, era, metric, (denom)].
    Uses symmetric weights: within = 0.5*(w2026 (m26-m24) + w2024 (m26-m24)) summed across companies,
                              between = 0.5*( (w2026-w2024)*(m26+m24) ) summed.
    If denom_col present, metric is a share: numerator=metric, denominator=denom_col.
    Only companies present in BOTH eras contribute to within; between uses all.
    """
    if denom_col is None:
        g = df_post.groupby(["company_name_canonical", "era"]).agg(
            n=("metric", "size"), sum_m=("metric", "sum")
        ).reset_index()
        g["mean_m"] = g["sum_m"] / g["n"]
    else:
        g = df_post.groupby(["company_name_canonical", "era"]).agg(
            n=(denom_col, "sum"), sum_m=("metric", "sum")
        ).reset_index()
        g["mean_m"] = g["sum_m"] / g["n"].replace(0, np.nan)

    # pivot wide
    w = g.pivot(index="company_name_canonical", columns="era", values=["n", "mean_m"]).fillna(0)
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w["N2024"] = w.get("n_2024", 0.0)
    w["N2026"] = w.get("n_2026", 0.0)
    w["m2024"] = w.get("mean_m_2024", np.nan)
    w["m2026"] = w.get("mean_m_2026", np.nan)

    # aggregate
    num24 = (w["N2024"] * w["m2024"].fillna(0)).sum()
    den24 = w["N2024"].sum()
    num26 = (w["N2026"] * w["m2026"].fillna(0)).sum()
    den26 = w["N2026"].sum()
    agg24 = num24 / den24 if den24 else np.nan
    agg26 = num26 / den26 if den26 else np.nan
    total = agg26 - agg24

    # restrict to overlap for within-co estimation but use aggregate denominators
    both = w[(w["N2024"] > 0) & (w["N2026"] > 0)].copy()

    # shares
    s24 = w["N2024"] / den24
    s26 = w["N2026"] / den26
    # within: symmetric average weights, only on both-era companies
    sbar = 0.5 * (s24 + s26)
    both_idx = both.index
    within = float((sbar.loc[both_idx] * (w.loc[both_idx, "m2026"] - w.loc[both_idx, "m2024"])).sum())

    # between: (s26 - s24) * m_bar, where m_bar uses within-co avg where available, else single-era metric
    mbar = 0.5 * (w["m2024"].fillna(0) + w["m2026"].fillna(0))
    # for companies appearing only in one era, use that metric alone
    only26 = w.index[(w["N2024"] == 0) & (w["N2026"] > 0)]
    only24 = w.index[(w["N2024"] > 0) & (w["N2026"] == 0)]
    mbar.loc[only26] = w.loc[only26, "m2026"]
    mbar.loc[only24] = w.loc[only24, "m2024"]
    between = float(((s26 - s24) * mbar).sum())

    return {
        "agg_2024": agg24,
        "agg_2026": agg26,
        "total_delta": total,
        "within_delta": within,
        "between_delta": between,
        "n_both_eras": int(len(both)),
        "n_2024": int((w["N2024"] > 0).sum()),
        "n_2026": int((w["N2026"] > 0).sum()),
        "volume_2024": int(den24),
        "volume_2026": int(den26),
    }


# -----------------------------------------------------------------------
# Build panels (company filters)
# -----------------------------------------------------------------------
cmp = con.execute("""
SELECT company_name_canonical,
  SUM(CASE WHEN source='kaggle_arshkon' THEN 1 ELSE 0 END) AS n_arshkon,
  SUM(CASE WHEN source='kaggle_asaniczka' THEN 1 ELSE 0 END) AS n_asan,
  SUM(CASE WHEN source IN ('kaggle_arshkon','kaggle_asaniczka') THEN 1 ELSE 0 END) AS n_pooled,
  SUM(CASE WHEN source='scraped' THEN 1 ELSE 0 END) AS n_scraped
FROM base GROUP BY 1
""").df()

panels = {
    "arshkon_min3": cmp.loc[(cmp.n_arshkon >= 3) & (cmp.n_scraped >= 3), "company_name_canonical"].tolist(),
    "arshkon_min5": cmp.loc[(cmp.n_arshkon >= 5) & (cmp.n_scraped >= 5), "company_name_canonical"].tolist(),
    "pooled_min5": cmp.loc[(cmp.n_pooled >= 5) & (cmp.n_scraped >= 5), "company_name_canonical"].tolist(),
}
for k, v in panels.items():
    print(f"[panel:{k}] n={len(v)}")

# -----------------------------------------------------------------------
# Get posting-level records for each metric
# -----------------------------------------------------------------------
def frame_for_metric(metric_col: str, denom_col: str | None = None, source_filter: str = "all"):
    """Return posting-level df with company, era, metric (and denom if share).
    source_filter: 'all' | 'arshkon' | 'pooled_2024_and_scraped'.
    For arshkon panels, restrict 2024 to arshkon; 2026 is always scraped.
    """
    base_sel = f"""
    SELECT company_name_canonical, era, source, {metric_col} AS metric
           {', '+denom_col+' AS denom' if denom_col else ''}
    FROM base
    """
    df = con.execute(base_sel).df()
    if source_filter == "arshkon":
        df = df[(df.era == "2026") | (df.source == "kaggle_arshkon")]
    return df


metrics_spec = [
    # (label, metric_col, denom_col)
    ("entry_share_j1", "j1_entry", None),
    ("entry_share_j2", "j2_assoc", None),
    ("entry_share_j3", "j3_flag", "j3_denom"),
    ("ai_prevalence_strict", "ai_strict", None),
    ("desc_length", "description_length", None),
    ("cleaned_length", "cleaned_len", None),
    ("breadth_resid", "requirement_breadth_resid", None),
    ("credential_stack_depth", "credential_stack_depth", None),
    ("scope_term_count", "scope_term_count", None),
    ("tech_count", "tech_count", None),
    ("mgmt_rebuilt", "mgmt_rebuilt", None),
]


rows = []
for panel_name, comps in panels.items():
    src_filter = "arshkon" if panel_name.startswith("arshkon") else "all"
    for label, metric, denom in metrics_spec:
        df = frame_for_metric(metric, denom, src_filter)
        df = df[df.company_name_canonical.isin(comps)]
        if df.empty:
            continue
        if denom is not None:
            # compute per-posting rate: numerator/denominator for shares where denom is 0/1 flag
            d = df.dropna(subset=["metric", "denom"]).copy()
            if d.empty:
                continue
            res = decompose(d, "metric", "denom")
        else:
            d = df.dropna(subset=["metric"]).copy()
            if d.empty:
                continue
            res = decompose(d, "metric")
        rows.append({
            "panel": panel_name,
            "metric": label,
            **res,
        })

dec = pd.DataFrame(rows)
dec.to_csv(OUT_DIR / "within_between_decomposition.csv", index=False)
print(f"[save] within_between_decomposition.csv n={len(dec)}")
print(dec.to_string(index=False))

# -----------------------------------------------------------------------
# Domain-layer decomposition (pooled_min5): within-domain vs between-domain vs between-company
# Using T09 archetype labels where available.
# -----------------------------------------------------------------------
# Archetype coverage is 8,000 of 48,223 rows. Still worth checking where labeled.
dom_rows = []
comps = panels["pooled_min5"]
base_df = con.execute("""
SELECT company_name_canonical, era, archetype_name,
       j1_entry, j2_assoc, j3_flag, j3_denom,
       ai_strict, mgmt_rebuilt, scope_term_count,
       description_length, cleaned_len, tech_count,
       requirement_breadth_resid, credential_stack_depth
FROM base
WHERE archetype_name IS NOT NULL
""").df()
base_df = base_df[base_df.company_name_canonical.isin(comps)]
base_df["archetype_name"] = base_df["archetype_name"].fillna("unknown")
print(f"[domain] rows with archetype in pool panel: {len(base_df)} across {base_df.archetype_name.nunique()} archetypes")


def decompose_domain(df: pd.DataFrame, metric: str, denom: str | None = None):
    # shares per archetype × era
    if denom:
        g = df.groupby(["archetype_name", "era"]).agg(
            N=(denom, "sum"), M=(metric, "sum")
        ).reset_index()
    else:
        g = df.groupby(["archetype_name", "era"]).agg(
            N=(metric, "size"), M=(metric, "sum")
        ).reset_index()
    g["mean"] = g["M"] / g["N"].replace(0, np.nan)
    w = g.pivot(index="archetype_name", columns="era", values=["N", "mean"]).fillna(0)
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    if "N_2024" not in w or "N_2026" not in w:
        return None
    s24 = w["N_2024"] / w["N_2024"].sum()
    s26 = w["N_2026"] / w["N_2026"].sum()
    mbar = 0.5 * (w["mean_2024"].fillna(0) + w["mean_2026"].fillna(0))
    within = float((0.5 * (s24 + s26) * (w["mean_2026"] - w["mean_2024"].fillna(0))).sum())
    between = float(((s26 - s24) * mbar).sum())
    total = within + between
    return {"within_domain_delta": within, "between_domain_delta": between, "domain_total": total}


for label, metric, denom in metrics_spec:
    try:
        res = decompose_domain(base_df, metric, denom)
        if res is None:
            continue
        dom_rows.append({"metric": label, **res})
    except Exception as e:
        print(f"[domain:warn] {label}: {e}")

dom = pd.DataFrame(dom_rows)
dom.to_csv(OUT_DIR / "within_between_domain.csv", index=False)
print(f"[save] within_between_domain.csv n={len(dom)}")
print(dom.to_string(index=False))
