"""
T37 — Sampling-frame returning-companies sensitivity (H_H).

Quantify how much of each anticipated Gate-3 headline is a sampling-frame artifact
vs genuine longitudinal signal. Restrict analysis to returning-companies cohort
(firms with SWE presence in both 2024 and scraped 2026) and re-run each headline
on that restricted sample.

Outputs:
- exploration/tables/T37/headline_sensitivity.csv  (per-headline full-corpus vs returning-cohort)
- exploration/tables/T37/t30_panel_returning.csv   (J1/J2/J3/J4, S1/S2/S3/S4 on returning)
- exploration/tables/T37/ai_strict_within_comparison.csv (T16 vs T37 within-company reconciliation)
- exploration/tables/T37/per_seniority_credential_stack.csv
- exploration/tables/T37/ci_bootstrap.csv
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
T13_METRICS = str(ROOT / "exploration" / "artifacts" / "shared" / "T13_readability_metrics.parquet")
T28_CORPUS = str(ROOT / "exploration" / "tables" / "T28" / "T28_corpus_with_archetype.parquet")
T21_CLUSTERS = str(ROOT / "exploration" / "tables" / "T21" / "cluster_assignments.csv")
RETURNING = str(ROOT / "exploration" / "artifacts" / "shared" / "returning_companies_cohort.csv")
PATTERNS_PATH = str(ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json")
T29_AUTHOR = str(ROOT / "exploration" / "tables" / "T29" / "authorship_scores.csv")

OUT = ROOT / "exploration" / "tables" / "T37"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

DEFAULT_FILTER = (
    "source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true"
)

PATTERNS = json.loads(Path(PATTERNS_PATH).read_text())
AI_STRICT = PATTERNS["ai_strict"]["pattern"]
MGMT_REBUILT = PATTERNS["v1_rebuilt_patterns"]["mgmt_strict_v1_rebuilt"]["pattern"]
SCOPE = PATTERNS["scope"]["pattern"]
SCOPE_KITCHEN_SINK = PATTERNS["t22_patterns"]["scope_kitchen_sink"]["pattern"]
HEDGING = PATTERNS["t22_patterns"]["aspiration_hedging"]["pattern"]

print("[T37] Loading base ...")
con = duckdb.connect()
con.execute(f"""
CREATE OR REPLACE TABLE text_flags AS
SELECT
  uid,
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS ai_strict,
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS mgmt_rebuilt,
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS scope_bin,
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS scope_kitchen_sink_bin,
  length(description_cleaned) - length(regexp_replace(lower(description_cleaned), ?, '', 'g')) AS scope_span,
  length(description_cleaned) AS cleaned_len
FROM '{CLEAN_TEXT}'
WHERE description_cleaned IS NOT NULL
""", [AI_STRICT, MGMT_REBUILT, SCOPE, SCOPE_KITCHEN_SINK, SCOPE])

con.execute(f"""
CREATE OR REPLACE TABLE scope_counts AS
SELECT
  uid,
  CAST(list_reduce(
    list_transform(
      regexp_extract_all(lower(description_cleaned), ?),
      x -> 1
    ), (a,v) -> a + v, 0
  ) AS BIGINT) AS scope_term_count,
  CAST(list_reduce(
    list_transform(
      regexp_extract_all(lower(description_cleaned), ?),
      x -> 1
    ), (a,v) -> a + v, 0
  ) AS BIGINT) AS scope_kitchen_sink_count
FROM '{CLEAN_TEXT}'
WHERE description_cleaned IS NOT NULL
""", [SCOPE, SCOPE_KITCHEN_SINK])

# Load T28 for archetype/domain
con.execute(f"""
CREATE OR REPLACE VIEW t28 AS
SELECT uid, archetype, archetype_primary, archetype_primary_name, domain FROM '{T28_CORPUS}'
""")

# T21 senior cluster assignments (junk for the AI-oriented senior cluster share)
# T21 cluster_assignments columns: uid, cluster_id, cluster_name, mgmt_density_v1_rebuilt, ...
con.execute(f"""
CREATE OR REPLACE VIEW t21_clusters AS
SELECT uid, cluster_id, cluster_name FROM read_csv_auto('{T21_CLUSTERS}')
""")

# Tech matrix for CI/CD prevalence (find tech key name)
TECH_MATRIX = str(ROOT / "exploration" / "artifacts" / "shared" / "swe_tech_matrix.parquet")
con.execute(f"CREATE OR REPLACE VIEW tech AS SELECT * FROM '{TECH_MATRIX}'")
cols_tech = con.execute("SELECT * FROM tech LIMIT 0").df().columns.tolist()
cicd_col = None
for c in cols_tech:
    if c.lower().strip() == "ci/cd":
        cicd_col = c
        break
if cicd_col is None:
    for c in cols_tech:
        if "cicd" in c.lower() or "ci_cd" in c.lower():
            cicd_col = c
            break
print(f"[T37] CI/CD column: {cicd_col!r}")

# Base
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
  tf.mgmt_rebuilt,
  tf.scope_bin,
  tf.scope_kitchen_sink_bin,
  sc.scope_term_count,
  sc.scope_kitchen_sink_count,
  tf.cleaned_len,
  tc.tech_count,
  tc.requirement_breadth,
  tc.requirement_breadth_resid,
  tc.credential_stack_depth,
  tc.description_cleaned_length,
  tc.ai_binary AS t11_ai_binary
FROM 'data/unified.parquet' u
LEFT JOIN text_flags tf USING (uid)
LEFT JOIN scope_counts sc USING (uid)
LEFT JOIN '{T11_FEATS}' tc USING (uid)
WHERE {DEFAULT_FILTER}
  AND u.company_name_canonical IS NOT NULL
""")

# Flags
con.execute("""
CREATE OR REPLACE TABLE base_era AS
SELECT b.*,
  CASE WHEN b.llm_classification_coverage='labeled' AND b.yoe_min_years_llm IS NOT NULL
       AND b.yoe_min_years_llm <= 2 THEN 1 ELSE 0 END AS j3_flag,
  CASE WHEN b.llm_classification_coverage='labeled' AND b.yoe_min_years_llm IS NOT NULL
       AND b.yoe_min_years_llm >= 5 THEN 1 ELSE 0 END AS s4_flag,
  CASE WHEN b.llm_classification_coverage='labeled' AND b.yoe_min_years_llm IS NOT NULL
       THEN 1 ELSE 0 END AS labeled_flag,
  CASE WHEN b.seniority_final='entry' THEN 1 ELSE 0 END AS j1_flag,
  CASE WHEN b.seniority_final IN ('entry','associate') THEN 1 ELSE 0 END AS j2_flag,
  CASE WHEN b.llm_classification_coverage='labeled' AND b.yoe_min_years_llm IS NOT NULL
       AND b.yoe_min_years_llm >= 3 THEN 1 ELSE 0 END AS s1_flag,
  CASE WHEN b.seniority_final='director' THEN 1 ELSE 0 END AS s2_flag,
  CASE WHEN b.seniority_final='mid-senior' THEN 1 ELSE 0 END AS s3_flag
FROM base b
""")

n_base_2024 = con.execute("SELECT COUNT(*) FROM base_era WHERE era='2024'").fetchone()[0]
n_base_2026 = con.execute("SELECT COUNT(*) FROM base_era WHERE era='2026'").fetchone()[0]
print(f"[T37] base era: 2024 n={n_base_2024}, 2026 n={n_base_2026}")

# Load returning
ret_df = pd.read_csv(RETURNING)
print(f"[T37] returning cohort: {len(ret_df)} companies")
# Register
con.register("ret_cohort", ret_df[["company_name_canonical"]])

# Check coverage
cov = con.execute("""
SELECT era, COUNT(*) AS n,
       SUM(CASE WHEN b.company_name_canonical IN (SELECT company_name_canonical FROM ret_cohort) THEN 1 ELSE 0 END) AS n_returning,
       ROUND(100.0*SUM(CASE WHEN b.company_name_canonical IN (SELECT company_name_canonical FROM ret_cohort) THEN 1 ELSE 0 END)/COUNT(*),2) AS pct
FROM base_era b GROUP BY era ORDER BY era
""").df()
print(cov.to_string(index=False))

# ----------------------------------------------------------------------
# Helper: compute share of flag in given era × cohort (returning or all)
# ----------------------------------------------------------------------
def share(flag_col: str, era: str, cohort: str = "all", denom: str = "all", **kwargs) -> tuple[float, int, int]:
    """Returns (share, n_num, n_denom). denom='labeled' restricts to labeled_flag=1."""
    cohort_clause = "" if cohort == "all" else "AND b.company_name_canonical IN (SELECT company_name_canonical FROM ret_cohort)"
    if denom == "labeled":
        q = f"""
        SELECT SUM({flag_col})::DOUBLE AS num, SUM(labeled_flag)::DOUBLE AS denom
        FROM base_era b
        WHERE era='{era}' {cohort_clause}
        """
    elif denom == "all":
        q = f"""
        SELECT SUM({flag_col})::DOUBLE AS num, COUNT(*)::DOUBLE AS denom
        FROM base_era b
        WHERE era='{era}' {cohort_clause}
        """
    else:
        raise ValueError(denom)
    r = con.execute(q).fetchone()
    num, den = r[0] or 0.0, r[1] or 0.0
    return (num / den if den > 0 else np.nan, int(num), int(den))


def share_stratified(flag_col: str, stratum_flag: str, era: str, cohort: str = "all") -> tuple[float, int, int]:
    """Fraction of stratum_flag=1 rows with flag_col=1 in era × cohort."""
    cohort_clause = "" if cohort == "all" else "AND b.company_name_canonical IN (SELECT company_name_canonical FROM ret_cohort)"
    q = f"""
    SELECT SUM({flag_col})::DOUBLE AS num, SUM({stratum_flag})::DOUBLE AS denom
    FROM base_era b
    WHERE era='{era}' {cohort_clause}
    """
    r = con.execute(q).fetchone()
    num, den = r[0] or 0.0, r[1] or 0.0
    return (num / den if den > 0 else np.nan, int(num), int(den))


# ----------------------------------------------------------------------
# Bootstrap helper for shares (produces 95% CI at cohort×era)
# ----------------------------------------------------------------------
def bootstrap_share_delta(flag_col: str, cohort: str = "returning", denom: str = "all", B: int = 500) -> dict:
    """Cluster bootstrap at company level on the RETURNING cohort. Returns dict with delta, lo, hi."""
    cohort_clause = ("AND b.company_name_canonical IN (SELECT company_name_canonical FROM ret_cohort)"
                     if cohort == "returning" else "")
    denom_clause = "AND labeled_flag=1" if denom == "labeled" else ""
    q = f"""
    SELECT company_name_canonical, era, {flag_col} AS f
    FROM base_era b
    WHERE 1=1 {cohort_clause} {denom_clause}
    """
    df = con.execute(q).df()
    companies = df["company_name_canonical"].dropna().unique()
    # Observed
    o_24 = df.loc[df.era == "2024", "f"].mean()
    o_26 = df.loc[df.era == "2026", "f"].mean()
    observed = o_26 - o_24
    # bootstrap
    deltas = []
    for _ in range(B):
        samp_cos = RNG.choice(companies, size=len(companies), replace=True)
        # pd.concat on a list of frames is slow for large B; accept cost
        sub = df.loc[df["company_name_canonical"].isin(samp_cos)]
        # but this treats duplicate-sampled companies as single inclusions, which is WRONG for cluster bootstrap
        # redo properly: build per-company groups once, then index
        # For efficiency we'll approximate with:
        # stratify then draw companies with replacement and concat.
        groups = df.groupby("company_name_canonical")
        pass
    # simpler: perform proper cluster bootstrap manually
    del deltas
    by_co = {c: df.loc[df["company_name_canonical"] == c] for c in companies}
    deltas = np.empty(B)
    for i in range(B):
        samp_cos = RNG.choice(companies, size=len(companies), replace=True)
        parts = [by_co[c] for c in samp_cos]
        if not parts:
            deltas[i] = np.nan
            continue
        sub = pd.concat(parts)
        s24 = sub.loc[sub.era == "2024", "f"]
        s26 = sub.loc[sub.era == "2026", "f"]
        if len(s24) == 0 or len(s26) == 0:
            deltas[i] = np.nan
            continue
        deltas[i] = s26.mean() - s24.mean()
    deltas = deltas[~np.isnan(deltas)]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": observed, "lo": lo, "hi": hi, "B_used": len(deltas)}


# ----------------------------------------------------------------------
# Helper: mean of continuous column in era × cohort
# ----------------------------------------------------------------------
def mean_cont(col: str, era: str, cohort: str = "all", stratum_flag: str | None = None) -> tuple[float, int]:
    cohort_clause = "" if cohort == "all" else "AND b.company_name_canonical IN (SELECT company_name_canonical FROM ret_cohort)"
    stratum_clause = f"AND {stratum_flag}=1" if stratum_flag else ""
    q = f"""
    SELECT AVG({col})::DOUBLE AS m, COUNT(*)::BIGINT AS n
    FROM base_era b
    WHERE era='{era}' AND {col} IS NOT NULL {cohort_clause} {stratum_clause}
    """
    r = con.execute(q).fetchone()
    return (r[0], r[1] or 0)


def median_cont(col: str, era: str, cohort: str = "all") -> tuple[float, int]:
    cohort_clause = "" if cohort == "all" else "AND b.company_name_canonical IN (SELECT company_name_canonical FROM ret_cohort)"
    q = f"""
    SELECT median({col})::DOUBLE AS m, COUNT(*)::BIGINT AS n
    FROM base_era b
    WHERE era='{era}' AND {col} IS NOT NULL {cohort_clause}
    """
    r = con.execute(q).fetchone()
    return (r[0], r[1] or 0)


# ----------------------------------------------------------------------
# H_a: AI-strict prevalence delta — ALL SWE, binary share
# ----------------------------------------------------------------------
print("\n[H_a] AI-strict prevalence Δ (V1-validated ai_strict, all-SWE, raw share)")
fa_24 = share("ai_strict", "2024", "all", "all")
fa_26 = share("ai_strict", "2026", "all", "all")
ra_24 = share("ai_strict", "2024", "returning", "all")
ra_26 = share("ai_strict", "2026", "returning", "all")
print(f"  full 2024={fa_24[0]:.4f} n={fa_24[2]}, 2026={fa_26[0]:.4f} n={fa_26[2]}, Δ={fa_26[0]-fa_24[0]:.4f}")
print(f"  retn 2024={ra_24[0]:.4f} n={ra_24[2]}, 2026={ra_26[0]:.4f} n={ra_26[2]}, Δ={ra_26[0]-ra_24[0]:.4f}")

# ----------------------------------------------------------------------
# H_b: J3 entry share Δ (labeled denominator)
# ----------------------------------------------------------------------
print("\n[H_b] J3 entry share Δ (yoe_min_years_llm <= 2 on labeled)")
fb_24 = share("j3_flag", "2024", "all", "labeled")
fb_26 = share("j3_flag", "2026", "all", "labeled")
rb_24 = share("j3_flag", "2024", "returning", "labeled")
rb_26 = share("j3_flag", "2026", "returning", "labeled")
print(f"  full 2024={fb_24[0]:.4f} denom={fb_24[2]}, 2026={fb_26[0]:.4f} denom={fb_26[2]}, Δ={fb_26[0]-fb_24[0]:.4f}")
print(f"  retn 2024={rb_24[0]:.4f} denom={rb_24[2]}, 2026={rb_26[0]:.4f} denom={rb_26[2]}, Δ={rb_26[0]-rb_24[0]:.4f}")
# arshkon-only on full
arshkon_24_j3 = con.execute("""
SELECT SUM(j3_flag)::DOUBLE / NULLIF(SUM(labeled_flag),0) AS p, SUM(labeled_flag)::BIGINT AS n
FROM base_era WHERE era='2024' AND source='kaggle_arshkon'
""").fetchone()
scraped_26_j3 = con.execute("""
SELECT SUM(j3_flag)::DOUBLE / NULLIF(SUM(labeled_flag),0) AS p, SUM(labeled_flag)::BIGINT AS n
FROM base_era WHERE era='2026'
""").fetchone()
print(f"  arshkon-only 2024={arshkon_24_j3[0]:.4f} n={arshkon_24_j3[1]}, scraped 2026={scraped_26_j3[0]:.4f} n={scraped_26_j3[1]}, Δ={scraped_26_j3[0]-arshkon_24_j3[0]:.4f}")

# arshkon-only returning
arshkon_ret_24_j3 = con.execute("""
SELECT SUM(j3_flag)::DOUBLE / NULLIF(SUM(labeled_flag),0) AS p, SUM(labeled_flag)::BIGINT AS n
FROM base_era b WHERE era='2024' AND source='kaggle_arshkon' AND
  b.company_name_canonical IN (SELECT company_name_canonical FROM ret_cohort)
""").fetchone()
scraped_ret_26_j3 = con.execute("""
SELECT SUM(j3_flag)::DOUBLE / NULLIF(SUM(labeled_flag),0) AS p, SUM(labeled_flag)::BIGINT AS n
FROM base_era b WHERE era='2026' AND
  b.company_name_canonical IN (SELECT company_name_canonical FROM ret_cohort)
""").fetchone()
print(f"  arshkon-only-returning 2024={arshkon_ret_24_j3[0]:.4f} n={arshkon_ret_24_j3[1]}, scraped-returning 2026={scraped_ret_26_j3[0]:.4f} n={scraped_ret_26_j3[1]}, Δ={scraped_ret_26_j3[0]-arshkon_ret_24_j3[0]:.4f}")

# ----------------------------------------------------------------------
# H_c: S4 senior share Δ
# ----------------------------------------------------------------------
print("\n[H_c] S4 senior share Δ (yoe_min_years_llm >= 5 on labeled)")
fc_24 = share("s4_flag", "2024", "all", "labeled")
fc_26 = share("s4_flag", "2026", "all", "labeled")
rc_24 = share("s4_flag", "2024", "returning", "labeled")
rc_26 = share("s4_flag", "2026", "returning", "labeled")
print(f"  full pool 2024={fc_24[0]:.4f} denom={fc_24[2]}, 2026={fc_26[0]:.4f} denom={fc_26[2]}, Δ={fc_26[0]-fc_24[0]:.4f}")
print(f"  retn 2024={rc_24[0]:.4f} denom={rc_24[2]}, 2026={rc_26[0]:.4f} denom={rc_26[2]}, Δ={rc_26[0]-rc_24[0]:.4f}")
# arshkon-only
arsh_full_24_s4 = con.execute("""
SELECT SUM(s4_flag)::DOUBLE / NULLIF(SUM(labeled_flag),0) AS p FROM base_era WHERE era='2024' AND source='kaggle_arshkon'
""").fetchone()
arsh_ret_24_s4 = con.execute("""
SELECT SUM(s4_flag)::DOUBLE / NULLIF(SUM(labeled_flag),0) AS p FROM base_era b WHERE era='2024' AND source='kaggle_arshkon'
AND b.company_name_canonical IN (SELECT company_name_canonical FROM ret_cohort)
""").fetchone()
print(f"  arshkon-only full 2024 S4={arsh_full_24_s4[0]:.4f}, arshkon-only returning 2024 S4={arsh_ret_24_s4[0]:.4f}")
print(f"  arshkon-only full Δ to scraped 2026: {fc_26[0]-arsh_full_24_s4[0]:.4f}, arshkon-only-returning Δ: {rc_26[0]-arsh_ret_24_s4[0]:.4f}")

# ----------------------------------------------------------------------
# H_d: Requirement breadth residualized S4 / J3
# ----------------------------------------------------------------------
print("\n[H_d] Requirement breadth residualized (T11)")
fd_j3_24 = mean_cont("requirement_breadth_resid", "2024", "all", "j3_flag")
fd_j3_26 = mean_cont("requirement_breadth_resid", "2026", "all", "j3_flag")
rd_j3_24 = mean_cont("requirement_breadth_resid", "2024", "returning", "j3_flag")
rd_j3_26 = mean_cont("requirement_breadth_resid", "2026", "returning", "j3_flag")
print(f"  J3 full 2024={fd_j3_24[0]:.3f} n={fd_j3_24[1]}, 2026={fd_j3_26[0]:.3f} n={fd_j3_26[1]}, Δ={fd_j3_26[0]-fd_j3_24[0]:.3f}")
print(f"  J3 retn 2024={rd_j3_24[0]:.3f} n={rd_j3_24[1]}, 2026={rd_j3_26[0]:.3f} n={rd_j3_26[1]}, Δ={rd_j3_26[0]-rd_j3_24[0]:.3f}")
fd_s4_24 = mean_cont("requirement_breadth_resid", "2024", "all", "s4_flag")
fd_s4_26 = mean_cont("requirement_breadth_resid", "2026", "all", "s4_flag")
rd_s4_24 = mean_cont("requirement_breadth_resid", "2024", "returning", "s4_flag")
rd_s4_26 = mean_cont("requirement_breadth_resid", "2026", "returning", "s4_flag")
print(f"  S4 full 2024={fd_s4_24[0]:.3f} n={fd_s4_24[1]}, 2026={fd_s4_26[0]:.3f} n={fd_s4_26[1]}, Δ={fd_s4_26[0]-fd_s4_24[0]:.3f}")
print(f"  S4 retn 2024={rd_s4_24[0]:.3f} n={rd_s4_24[1]}, 2026={rd_s4_26[0]:.3f} n={rd_s4_26[1]}, Δ={rd_s4_26[0]-rd_s4_24[0]:.3f}")

# ----------------------------------------------------------------------
# H_e: Credential stack depth >= 5 share (J3 and S4)
# ----------------------------------------------------------------------
print("\n[H_e] Credential stack depth >= 5 categories (J3/S4)")
# We need share of credential_stack_depth >= 5
# Compute directly via SQL.
def cred_ge5_share(era: str, cohort: str, stratum: str) -> tuple[float, int, int]:
    cohort_clause = "" if cohort == "all" else "AND b.company_name_canonical IN (SELECT company_name_canonical FROM ret_cohort)"
    q = f"""
    SELECT SUM(CASE WHEN credential_stack_depth >= 5 THEN 1 ELSE 0 END)::DOUBLE AS num,
           COUNT(*)::DOUBLE AS denom
    FROM base_era b
    WHERE era='{era}' AND credential_stack_depth IS NOT NULL AND {stratum}=1 {cohort_clause}
    """
    r = con.execute(q).fetchone()
    num, den = r[0] or 0.0, r[1] or 0.0
    return (num / den if den > 0 else np.nan, int(num), int(den))

fe_j3_24 = cred_ge5_share("2024", "all", "j3_flag")
fe_j3_26 = cred_ge5_share("2026", "all", "j3_flag")
re_j3_24 = cred_ge5_share("2024", "returning", "j3_flag")
re_j3_26 = cred_ge5_share("2026", "returning", "j3_flag")
print(f"  J3 full 2024={fe_j3_24[0]:.4f} n={fe_j3_24[2]}, 2026={fe_j3_26[0]:.4f} n={fe_j3_26[2]}, Δ={fe_j3_26[0]-fe_j3_24[0]:.4f}")
print(f"  J3 retn 2024={re_j3_24[0]:.4f} n={re_j3_24[2]}, 2026={re_j3_26[0]:.4f} n={re_j3_26[2]}, Δ={re_j3_26[0]-re_j3_24[0]:.4f}")
fe_s4_24 = cred_ge5_share("2024", "all", "s4_flag")
fe_s4_26 = cred_ge5_share("2026", "all", "s4_flag")
re_s4_24 = cred_ge5_share("2024", "returning", "s4_flag")
re_s4_26 = cred_ge5_share("2026", "returning", "s4_flag")
print(f"  S4 full 2024={fe_s4_24[0]:.4f} n={fe_s4_24[2]}, 2026={fe_s4_26[0]:.4f} n={fe_s4_26[2]}, Δ={fe_s4_26[0]-fe_s4_24[0]:.4f}")
print(f"  S4 retn 2024={re_s4_24[0]:.4f} n={re_s4_24[2]}, 2026={re_s4_26[0]:.4f} n={re_s4_26[2]}, Δ={re_s4_26[0]-re_s4_24[0]:.4f}")

# ----------------------------------------------------------------------
# H_f: Requirements section share (T13 classifier) — DEMOTED to directional (below noise)
# We use T13's classifier output from shared T13_readability_metrics if available
# Check columns
# ----------------------------------------------------------------------
print("\n[H_f] Requirements-section share (T13)")
try:
    t13_cols = con.execute(f"SELECT * FROM '{T13_METRICS}' LIMIT 0").df().columns.tolist()
    print(f"  T13 cols: {t13_cols[:20]}")
    # Look for section share columns
    share_cols = [c for c in t13_cols if "share" in c.lower() and "req" in c.lower()]
    char_cols = [c for c in t13_cols if "char" in c.lower() and "req" in c.lower()]
    print(f"  req-section share cols: {share_cols}")
    print(f"  req-section char cols: {char_cols}")
    # If present, compute delta
    if share_cols:
        rc = share_cols[0]
        q = f"""
        WITH src AS (
          SELECT CASE WHEN u.period IN ('2024-01','2024-04') THEN '2024' ELSE '2026' END AS era,
                 t.{rc} AS v
          FROM 'data/unified.parquet' u
          JOIN '{T13_METRICS}' t USING(uid)
          WHERE {DEFAULT_FILTER} AND t.{rc} IS NOT NULL
        )
        SELECT era, AVG(v) AS m, COUNT(*) AS n FROM src GROUP BY era ORDER BY era
        """
        ff_t13 = con.execute(q).df()
        print(f"  full {rc}:\n{ff_t13.to_string(index=False)}")
        q = f"""
        WITH src AS (
          SELECT CASE WHEN u.period IN ('2024-01','2024-04') THEN '2024' ELSE '2026' END AS era,
                 t.{rc} AS v
          FROM 'data/unified.parquet' u
          JOIN '{T13_METRICS}' t USING(uid)
          WHERE {DEFAULT_FILTER} AND t.{rc} IS NOT NULL
            AND u.company_name_canonical IN (SELECT company_name_canonical FROM ret_cohort)
        )
        SELECT era, AVG(v) AS m, COUNT(*) AS n FROM src GROUP BY era ORDER BY era
        """
        rr_t13 = con.execute(q).df()
        print(f"  retn {rc}:\n{rr_t13.to_string(index=False)}")
    else:
        print("  (no shared req-share col)")
        ff_t13 = pd.DataFrame(); rr_t13 = pd.DataFrame()
except Exception as e:
    print(f"  T13 metrics unavailable: {e}")

# ----------------------------------------------------------------------
# H_g: Length median Δ (raw description)
# ----------------------------------------------------------------------
print("\n[H_g] Description length (raw chars, median)")
lg_24 = median_cont("description_length", "2024", "all")
lg_26 = median_cont("description_length", "2026", "all")
rg_24 = median_cont("description_length", "2024", "returning")
rg_26 = median_cont("description_length", "2026", "returning")
print(f"  full median 2024={lg_24[0]:.0f}, 2026={lg_26[0]:.0f}, Δ={lg_26[0]-lg_24[0]:.0f}")
print(f"  retn median 2024={rg_24[0]:.0f}, 2026={rg_26[0]:.0f}, Δ={rg_26[0]-rg_24[0]:.0f}")

# ----------------------------------------------------------------------
# H_h: Scope term rate Δ (T28 reports +21.0 pp per calibration)
# Use scope_bin (V1 validated scope pattern) as the prevalence metric
# ----------------------------------------------------------------------
print("\n[H_h] Scope term rate Δ (V1 scope pattern, share >=1 match)")
fh_24 = share("scope_bin", "2024", "all", "all")
fh_26 = share("scope_bin", "2026", "all", "all")
rh_24 = share("scope_bin", "2024", "returning", "all")
rh_26 = share("scope_bin", "2026", "returning", "all")
print(f"  full 2024={fh_24[0]:.4f}, 2026={fh_26[0]:.4f}, Δ={fh_26[0]-fh_24[0]:.4f}")
print(f"  retn 2024={rh_24[0]:.4f}, 2026={rh_26[0]:.4f}, Δ={rh_26[0]-rh_24[0]:.4f}")
# Alt: scope_kitchen_sink rate
print("  (alt scope_kitchen_sink binary)")
fk_24 = share("scope_kitchen_sink_bin", "2024", "all", "all")
fk_26 = share("scope_kitchen_sink_bin", "2026", "all", "all")
rk_24 = share("scope_kitchen_sink_bin", "2024", "returning", "all")
rk_26 = share("scope_kitchen_sink_bin", "2026", "returning", "all")
print(f"  kitchen-sink full 2024={fk_24[0]:.4f}, 2026={fk_26[0]:.4f}, Δ={fk_26[0]-fk_24[0]:.4f}")
print(f"  kitchen-sink retn 2024={rk_24[0]:.4f}, 2026={rk_26[0]:.4f}, Δ={rk_26[0]-rk_24[0]:.4f}")

# ----------------------------------------------------------------------
# H_i: CI/CD prevalence at S4 — +20.6 pp
# ----------------------------------------------------------------------
print("\n[H_i] CI/CD prevalence at S4")
if cicd_col:
    q = f"""
    WITH src AS (
      SELECT
        CASE WHEN u.period IN ('2024-01','2024-04') THEN '2024' ELSE '2026' END AS era,
        t."{cicd_col}"::DOUBLE AS cicd
      FROM 'data/unified.parquet' u
      JOIN tech t USING(uid)
      JOIN base_era b USING(uid)
      WHERE {DEFAULT_FILTER} AND b.s4_flag=1
    )
    SELECT era, AVG(cicd) AS p, COUNT(*) AS n FROM src GROUP BY era ORDER BY era
    """
    ff_ci = con.execute(q).df()
    print(f"  full S4:\n{ff_ci.to_string(index=False)}")
    q_ret = f"""
    WITH src AS (
      SELECT
        CASE WHEN u.period IN ('2024-01','2024-04') THEN '2024' ELSE '2026' END AS era,
        t."{cicd_col}"::DOUBLE AS cicd
      FROM 'data/unified.parquet' u
      JOIN tech t USING(uid)
      JOIN base_era b USING(uid)
      WHERE {DEFAULT_FILTER} AND b.s4_flag=1
        AND u.company_name_canonical IN (SELECT company_name_canonical FROM ret_cohort)
    )
    SELECT era, AVG(cicd) AS p, COUNT(*) AS n FROM src GROUP BY era ORDER BY era
    """
    rr_ci = con.execute(q_ret).df()
    print(f"  retn S4:\n{rr_ci.to_string(index=False)}")
else:
    ff_ci = pd.DataFrame()
    rr_ci = pd.DataFrame()

# ----------------------------------------------------------------------
# H_j: AI-oriented senior cluster share Δ (T21 cluster = 'AI-oriented' among seniors)
# ----------------------------------------------------------------------
print("\n[H_j] AI-oriented senior cluster share Δ (T21)")
try:
    tclust = pd.read_csv(T21_CLUSTERS)
    print(f"  T21 clusters rows: {len(tclust)}; unique clusters: {tclust['cluster_name'].unique()}")
    # cluster denotes senior-only subset; match 'AI-oriented'
    ai_cluster = [c for c in tclust["cluster_name"].unique() if "AI-oriented" in c or "AI-" in c]
    print(f"  AI cluster label: {ai_cluster}")
    ai_name = ai_cluster[0] if ai_cluster else None
    if ai_name:
        # Compute share per era (denom = all senior postings in cluster assignments)
        per = tclust.groupby("period").agg(
            n=("uid", "count"),
            n_ai=("cluster_name", lambda s: (s == ai_name).sum()),
        )
        per["share_ai"] = per["n_ai"] / per["n"]
        print(f"  full senior cluster breakdown:\n{per}")
        # returning
        ret_cs = set(ret_df["company_name_canonical"].tolist())
        # Need company for each uid; join via base
        uids = tclust["uid"].tolist()
        q_comps = f"""
        SELECT uid, company_name_canonical FROM base WHERE uid IN ({','.join([f"'{u}'" for u in uids[:50000]])})
        """
        # duckdb may error on very long IN — use register instead
        con.register("clust_uids", tclust[["uid"]])
        q_comps = """
        SELECT c.uid, b.company_name_canonical FROM clust_uids c LEFT JOIN base b USING(uid)
        """
        uid_co = con.execute(q_comps).df()
        tclust_merge = tclust.merge(uid_co, on="uid", how="left")
        tclust_ret = tclust_merge[tclust_merge["company_name_canonical"].isin(ret_cs)]
        per_ret = tclust_ret.groupby("period").agg(
            n=("uid", "count"),
            n_ai=("cluster_name", lambda s: (s == ai_name).sum()),
        )
        per_ret["share_ai"] = per_ret["n_ai"] / per_ret["n"]
        print(f"  retn senior cluster breakdown:\n{per_ret}")
    else:
        per = pd.DataFrame(); per_ret = pd.DataFrame()
except Exception as e:
    print(f"  T21 clusters unavailable: {e}")
    per = pd.DataFrame(); per_ret = pd.DataFrame()

# ----------------------------------------------------------------------
# Assembly: write per-headline sensitivity table
# ----------------------------------------------------------------------
print("\n[T37] Building sensitivity table ...")
rows = []

def add(metric: str, full_delta: float, ret_delta: float, full_24: float, full_26: float, ret_24: float, ret_26: float, unit: str, note: str = ""):
    ratio = ret_delta / full_delta if abs(full_delta) >= 1e-6 else np.nan
    if unit == "pp":
        mat_threshold = 0.01  # 1 pp on share
    elif unit == "chars":
        mat_threshold = 100  # trivially material for descriptive
    else:
        mat_threshold = 0.2  # continuous SD threshold
    material = abs(full_delta) >= mat_threshold
    if not material:
        ratio_disp = "undefined"
        verdict = "undefined — full-corpus Δ too small"
    else:
        if ratio >= 0.80:
            verdict = "robust"
        elif ratio >= 0.50:
            verdict = "partially_robust"
        else:
            verdict = "sampling_frame_driven"
        ratio_disp = f"{ratio:.2f}"
    rows.append({
        "metric": metric,
        "full_corpus_2024": full_24,
        "full_corpus_2026": full_26,
        "full_corpus_delta": full_delta,
        "returning_cohort_2024": ret_24,
        "returning_cohort_2026": ret_26,
        "returning_cohort_delta": ret_delta,
        "retention_ratio": ratio_disp,
        "unit": unit,
        "verdict": verdict,
        "note": note,
    })

add("H_a: AI-strict prevalence",
    fa_26[0]-fa_24[0], ra_26[0]-ra_24[0],
    fa_24[0], fa_26[0], ra_24[0], ra_26[0], "pp",
    "V1 ai_strict pattern, all-SWE raw share")

add("H_b: J3 entry share (pooled baseline)",
    fb_26[0]-fb_24[0], rb_26[0]-rb_24[0],
    fb_24[0], fb_26[0], rb_24[0], rb_26[0], "pp",
    "yoe<=2 labeled; full uses pooled-2024")

add("H_b-alt: J3 entry share (arshkon-only baseline)",
    scraped_26_j3[0]-arshkon_24_j3[0], scraped_ret_26_j3[0]-arshkon_ret_24_j3[0],
    arshkon_24_j3[0], scraped_26_j3[0], arshkon_ret_24_j3[0], scraped_ret_26_j3[0], "pp",
    "arshkon-only 2024 baseline (Gate 1 senior primary frame)")

add("H_c: S4 senior share (pooled baseline)",
    fc_26[0]-fc_24[0], rc_26[0]-rc_24[0],
    fc_24[0], fc_26[0], rc_24[0], rc_26[0], "pp",
    "yoe>=5 labeled; pooled-2024 baseline")

add("H_c-alt: S4 senior share (arshkon-only baseline)",
    fc_26[0]-arsh_full_24_s4[0], rc_26[0]-arsh_ret_24_s4[0],
    arsh_full_24_s4[0], fc_26[0], arsh_ret_24_s4[0], rc_26[0], "pp",
    "arshkon-only 2024; senior co-primary frame")

add("H_d-J3: Breadth residualized (J3)",
    fd_j3_26[0]-fd_j3_24[0], rd_j3_26[0]-rd_j3_24[0],
    fd_j3_24[0], fd_j3_26[0], rd_j3_24[0], rd_j3_26[0], "items",
    "T11 length-residualized breadth, J3")

add("H_d-S4: Breadth residualized (S4)",
    fd_s4_26[0]-fd_s4_24[0], rd_s4_26[0]-rd_s4_24[0],
    fd_s4_24[0], fd_s4_26[0], rd_s4_24[0], rd_s4_26[0], "items",
    "T11 length-residualized breadth, S4")

add("H_e-J3: Credential stack >=5 (J3)",
    fe_j3_26[0]-fe_j3_24[0], re_j3_26[0]-re_j3_24[0],
    fe_j3_24[0], fe_j3_26[0], re_j3_24[0], re_j3_26[0], "pp",
    "T11 credential_stack_depth >=5 share, J3")

add("H_e-S4: Credential stack >=5 (S4)",
    fe_s4_26[0]-fe_s4_24[0], re_s4_26[0]-re_s4_24[0],
    fe_s4_24[0], fe_s4_26[0], re_s4_24[0], re_s4_26[0], "pp",
    "T11 credential_stack_depth >=5 share, S4")

# H_f: T13 requirements-section share
try:
    if len(ff_t13) >= 2 and len(rr_t13) >= 2:
        ff_t13 = ff_t13.set_index("era")
        rr_t13 = rr_t13.set_index("era")
        add("H_f: T13 requirements-section share",
            ff_t13.loc["2026","m"]-ff_t13.loc["2024","m"],
            rr_t13.loc["2026","m"]-rr_t13.loc["2024","m"],
            ff_t13.loc["2024","m"], ff_t13.loc["2026","m"],
            rr_t13.loc["2024","m"], rr_t13.loc["2026","m"], "pp",
            "T13 classifier requirements-section share (below-noise aggregate)")
except Exception as e:
    print(f"[warn] H_f skipped: {e}")

add("H_g: Description length median (raw chars)",
    lg_26[0]-lg_24[0], rg_26[0]-rg_24[0],
    lg_24[0], lg_26[0], rg_24[0], rg_26[0], "chars",
    "raw description_length median")

add("H_h: Scope term prevalence (V1 binary)",
    fh_26[0]-fh_24[0], rh_26[0]-rh_24[0],
    fh_24[0], fh_26[0], rh_24[0], rh_26[0], "pp",
    "V1 scope pattern binary share")

add("H_h-alt: Scope kitchen-sink prevalence",
    fk_26[0]-fk_24[0], rk_26[0]-rk_24[0],
    fk_24[0], fk_26[0], rk_24[0], rk_26[0], "pp",
    "T22 scope_kitchen_sink binary share")

# CI/CD S4
if cicd_col and len(ff_ci) > 0 and len(rr_ci) > 0:
    try:
        ffv = ff_ci.set_index("era")["p"]
        rrv = rr_ci.set_index("era")["p"]
        add("H_i: CI/CD tech at S4",
            ffv["2026"] - ffv["2024"], rrv["2026"] - rrv["2024"],
            ffv["2024"], ffv["2026"], rrv["2024"], rrv["2026"], "pp",
            "CI/CD tech matrix column at S4 yoe>=5")
    except Exception as e:
        print(f"[warn] H_i skipped: {e}")

# AI-oriented senior cluster share
if len(per) > 0 and len(per_ret) > 0:
    try:
        f24 = per.loc[per.index == 2024, "share_ai"].values[0]
        f26 = per.loc[per.index == 2026, "share_ai"].values[0]
        r24 = per_ret.loc[per_ret.index == 2024, "share_ai"].values[0]
        r26 = per_ret.loc[per_ret.index == 2026, "share_ai"].values[0]
        add("H_j: AI-oriented senior cluster share (T21)",
            f26 - f24, r26 - r24, f24, f26, r24, r26, "pp",
            "T21 AI-oriented cluster share among senior postings")
    except Exception as e:
        print(f"[warn] H_j skipped: {e}")

headline_df = pd.DataFrame(rows)
headline_df.to_csv(OUT / "headline_sensitivity.csv", index=False)
print(f"[save] headline_sensitivity.csv ({len(headline_df)} rows)")
print(headline_df[["metric","full_corpus_delta","returning_cohort_delta","retention_ratio","verdict"]].to_string(index=False))

# ----------------------------------------------------------------------
# 2. T30 panel J1/J2/J3/J4, S1/S2/S3/S4 on returning cohort
# ----------------------------------------------------------------------
print("\n[T37] T30 panel on returning cohort")
panel_rows = []
for flag, side, denom in [
    ("j1_flag", "J1", "all"),
    ("j2_flag", "J2", "all"),
    ("j3_flag", "J3", "labeled"),
    # J4 = yoe<=3
    ]:
    for cohort in ["all", "returning"]:
        s24 = share(flag, "2024", cohort, denom)
        s26 = share(flag, "2026", cohort, denom)
        panel_rows.append({
            "definition": side, "cohort": cohort,
            "share_2024": s24[0], "share_2026": s26[0],
            "delta": s26[0]-s24[0], "n_denom_2024": s24[2], "n_denom_2026": s26[2]
        })

# j4 = yoe<=3
con.execute("""
ALTER TABLE base_era ADD COLUMN IF NOT EXISTS j4_flag INTEGER;
UPDATE base_era SET j4_flag = CASE WHEN labeled_flag=1 AND yoe_min_years_llm <= 3 THEN 1 ELSE 0 END
""")
for cohort in ["all", "returning"]:
    s24 = share("j4_flag", "2024", cohort, "labeled")
    s26 = share("j4_flag", "2026", cohort, "labeled")
    panel_rows.append({
        "definition": "J4", "cohort": cohort,
        "share_2024": s24[0], "share_2026": s26[0],
        "delta": s26[0]-s24[0], "n_denom_2024": s24[2], "n_denom_2026": s26[2]
    })

# S1..S4
# S1 = yoe>=3; S2 = director label; S3 = mid-senior; S4 = yoe>=5
con.execute("""
ALTER TABLE base_era ADD COLUMN IF NOT EXISTS s1b_flag INTEGER;
UPDATE base_era SET s1b_flag = CASE WHEN labeled_flag=1 AND yoe_min_years_llm >= 3 THEN 1 ELSE 0 END
""")
for flag, side, denom in [
    ("s1b_flag", "S1", "labeled"),
    ("s2_flag", "S2", "all"),
    ("s3_flag", "S3", "all"),
    ("s4_flag", "S4", "labeled"),
]:
    for cohort in ["all", "returning"]:
        s24 = share(flag, "2024", cohort, denom)
        s26 = share(flag, "2026", cohort, denom)
        panel_rows.append({
            "definition": side, "cohort": cohort,
            "share_2024": s24[0], "share_2026": s26[0],
            "delta": s26[0]-s24[0], "n_denom_2024": s24[2], "n_denom_2026": s26[2]
        })

panel_df = pd.DataFrame(panel_rows)
panel_df.to_csv(OUT / "t30_panel_returning.csv", index=False)
print(panel_df.to_string(index=False))

# ----------------------------------------------------------------------
# 3. Cross-check: T37 vs T16 within-company AI-strict on returning cohort
# ----------------------------------------------------------------------
print("\n[T37] Within-company AI-strict on returning cohort (compare to T16 panels)")
# T16 reports on overlap_panel arshkon_min5 / pooled_min5 (different panels)
# Returning cohort = ≥1 post in 2024 AND ≥1 post in 2026 (pooled 2024 = arshkon ∪ asaniczka)
# so a superset of pooled_min5 (which required ≥5 posts in each era)
# Within-company decomposition:
#   aggregate Δ = sum_c w_c^2026 * p_c^2026 - sum_c w_c^2024 * p_c^2024
#   within = sum_c w_sym_c * (p_c^2026 - p_c^2024)
#   between = aggregate - within (residual)
# Using symmetric weight w_sym_c = (w_c^2024 + w_c^2026) / 2
def decompose_within(flag: str, denom: str, cohort: str) -> dict:
    cohort_clause = ("AND b.company_name_canonical IN (SELECT company_name_canonical FROM ret_cohort)"
                     if cohort == "returning" else "")
    denom_clause = "AND labeled_flag=1" if denom == "labeled" else ""
    q = f"""
    WITH co AS (
      SELECT era, company_name_canonical,
             SUM({flag})::DOUBLE AS num, COUNT(*)::DOUBLE AS denom
      FROM base_era b
      WHERE 1=1 {cohort_clause} {denom_clause}
      GROUP BY era, company_name_canonical
    ),
    wide AS (
      SELECT company_name_canonical,
             MAX(CASE WHEN era='2024' THEN num END) AS n24,
             MAX(CASE WHEN era='2024' THEN denom END) AS d24,
             MAX(CASE WHEN era='2026' THEN num END) AS n26,
             MAX(CASE WHEN era='2026' THEN denom END) AS d26
      FROM co GROUP BY company_name_canonical
    ),
    tot AS (
      SELECT SUM(COALESCE(d24,0)) AS T24, SUM(COALESCE(d26,0)) AS T26 FROM wide
    ),
    weighted AS (
      SELECT w.*,
             COALESCE(d24,0)/NULLIF((SELECT T24 FROM tot),0) AS w24,
             COALESCE(d26,0)/NULLIF((SELECT T26 FROM tot),0) AS w26,
             COALESCE(n24,0)/NULLIF(COALESCE(d24,0),0) AS p24,
             COALESCE(n26,0)/NULLIF(COALESCE(d26,0),0) AS p26
      FROM wide w
    )
    SELECT
      SUM(COALESCE(p26,0)*w26) - SUM(COALESCE(p24,0)*w24) AS aggregate_delta,
      SUM((COALESCE(p26,0)-COALESCE(p24,0)) * 0.5*(COALESCE(w24,0)+COALESCE(w26,0))) AS within_sym,
      COUNT(*) AS n_cos,
      SUM(CASE WHEN d24 IS NOT NULL AND d24>0 AND d26 IS NOT NULL AND d26>0 THEN 1 ELSE 0 END) AS n_cos_both
    FROM weighted
    """
    r = con.execute(q).fetchone()
    agg, within, ncos, ncos_both = r
    between = (agg or 0) - (within or 0)
    return {
        "aggregate_delta": agg, "within_sym": within, "between_residual": between,
        "n_cos": ncos, "n_cos_both_eras": ncos_both
    }

ai_ret = decompose_within("ai_strict", "all", "returning")
ai_all = decompose_within("ai_strict", "all", "all")
print(f"  AI-strict all cohort: {ai_all}")
print(f"  AI-strict returning: {ai_ret}")

j3_ret = decompose_within("j3_flag", "labeled", "returning")
j3_all = decompose_within("j3_flag", "labeled", "all")
print(f"  J3 all cohort: {j3_all}")
print(f"  J3 returning: {j3_ret}")

s4_ret = decompose_within("s4_flag", "labeled", "returning")
s4_all = decompose_within("s4_flag", "labeled", "all")
print(f"  S4 all cohort: {s4_all}")
print(f"  S4 returning: {s4_ret}")

scope_ret = decompose_within("scope_bin", "all", "returning")
scope_all = decompose_within("scope_bin", "all", "all")
print(f"  Scope all cohort: {scope_all}")
print(f"  Scope returning: {scope_ret}")

within_rows = [
    {"metric": "AI-strict prevalence", "cohort": "all",
     **ai_all},
    {"metric": "AI-strict prevalence", "cohort": "returning",
     **ai_ret},
    {"metric": "J3 entry share (labeled)", "cohort": "all",
     **j3_all},
    {"metric": "J3 entry share (labeled)", "cohort": "returning",
     **j3_ret},
    {"metric": "S4 senior share (labeled)", "cohort": "all",
     **s4_all},
    {"metric": "S4 senior share (labeled)", "cohort": "returning",
     **s4_ret},
    {"metric": "Scope prevalence (V1)", "cohort": "all",
     **scope_all},
    {"metric": "Scope prevalence (V1)", "cohort": "returning",
     **scope_ret},
]
within_df = pd.DataFrame(within_rows)
within_df.to_csv(OUT / "within_between_returning_vs_all.csv", index=False)
print(within_df.to_string(index=False))

# T16 arshkon_min5 AI-strict within-co was +8.34 pp (+0.0834)
# T16 pooled_min5 within-co was +7.65 pp (+0.0765)
# Compare to returning here:
with_cmp = [
    {"panel": "T16 arshkon_min5", "n_cos": 125, "ai_strict_within_co": 0.0834, "note": "V1-validated reference"},
    {"panel": "T16 pooled_min5", "n_cos": 356, "ai_strict_within_co": 0.0765, "note": "V1-validated reference"},
    {"panel": "T37 returning cohort (all ≥1×≥1)", "n_cos": ai_ret["n_cos"], "ai_strict_within_co": ai_ret["within_sym"], "note": "within-firm symmetric weighted"},
]
pd.DataFrame(with_cmp).to_csv(OUT / "ai_strict_within_comparison.csv", index=False)
print(pd.DataFrame(with_cmp).to_string(index=False))

# ----------------------------------------------------------------------
# 4. Sensitivity: per-seniority breadth (J3+S4 panel) and cred-stack panel on returning
# ----------------------------------------------------------------------
print("\n[T37] Per-seniority panel on returning cohort (breadth & credstack)")
seniority_rows = []
for flag, side, denom in [("j3_flag", "J3", "labeled"), ("s4_flag", "S4", "labeled"),
                           ("j1_flag", "J1", "all"), ("j2_flag", "J2", "all"),
                           ("j4_flag", "J4", "labeled"), ("s1b_flag", "S1", "labeled"),
                           ("s3_flag", "S3", "all")]:
    for cohort in ["all", "returning"]:
        m_b24 = mean_cont("requirement_breadth_resid", "2024", cohort, stratum_flag=flag)
        m_b26 = mean_cont("requirement_breadth_resid", "2026", cohort, stratum_flag=flag)
        m_c24 = cred_ge5_share("2024", cohort, flag)
        m_c26 = cred_ge5_share("2026", cohort, flag)
        seniority_rows.append({
            "seniority": side, "cohort": cohort,
            "breadth_resid_2024": m_b24[0], "breadth_resid_2026": m_b26[0],
            "breadth_resid_delta": (m_b26[0] or 0) - (m_b24[0] or 0),
            "credstack_ge5_2024": m_c24[0], "credstack_ge5_2026": m_c26[0],
            "credstack_ge5_delta": (m_c26[0] or 0) - (m_c24[0] or 0),
            "n_2024": m_b24[1], "n_2026": m_b26[1],
        })

pd.DataFrame(seniority_rows).to_csv(OUT / "per_seniority_credential_stack.csv", index=False)
print(pd.DataFrame(seniority_rows).to_string(index=False))

# ----------------------------------------------------------------------
# 5. Bootstrap 95% CI on key returning-cohort deltas
# ----------------------------------------------------------------------
print("\n[T37] Cluster bootstrap 95% CI on returning-cohort deltas ...")
ci_rows = []
# AI strict
bci = bootstrap_share_delta("ai_strict", "returning", "all", B=300)
ci_rows.append({"metric": "AI-strict (all-SWE)", **bci})
# J3
bci = bootstrap_share_delta("j3_flag", "returning", "labeled", B=300)
ci_rows.append({"metric": "J3 entry (labeled)", **bci})
# S4
bci = bootstrap_share_delta("s4_flag", "returning", "labeled", B=300)
ci_rows.append({"metric": "S4 senior (labeled)", **bci})
# Scope
bci = bootstrap_share_delta("scope_bin", "returning", "all", B=300)
ci_rows.append({"metric": "Scope (V1 binary)", **bci})
ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv(OUT / "ci_bootstrap.csv", index=False)
print(ci_df.to_string(index=False))

print("\n[T37] ALL DONE.")
