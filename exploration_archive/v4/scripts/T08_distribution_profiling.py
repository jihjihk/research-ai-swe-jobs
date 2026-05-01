#!/usr/bin/env python3
"""T08: Distribution profiling & anomaly detection.

Baselines distributions and surfaces anomalies across ALL meaningful variables.
Heavy sensitivity framework: full seniority ablation including combined
best-available column + label-independent YOE proxy.

Outputs:
  - exploration/figures/T08/ (<= 4 PNGs, 150dpi)
  - exploration/tables/T08/  (CSVs)
  - exploration/reports/T08.md (written separately)
"""

import os
import json
import warnings
from collections import Counter

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

PARQUET = 'data/unified.parquet'
FIG_DIR = 'exploration/figures/T08'
TAB_DIR = 'exploration/tables/T08'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

BASE_FILTER = """
    source_platform = 'linkedin'
    AND is_english = true
    AND date_flag = 'ok'
    AND is_swe = true
"""

SENIORITY_CTE = """
    CASE
      WHEN llm_classification_coverage = 'labeled'         THEN seniority_llm
      WHEN llm_classification_coverage = 'rule_sufficient' THEN seniority_final
      ELSE NULL
    END AS seniority_best_available
"""

# We analyse periods at three levels:
# - period_raw: 2024-01, 2024-04, 2026-03, 2026-04
# - period_coarse: 2024 vs 2026 (pooled)
# - source: for calibration / quality checks
PERIOD_COARSE_CASE = """
    CASE
      WHEN period IN ('2024-01','2024-04') THEN '2024'
      WHEN period IN ('2026-03','2026-04') THEN '2026'
      ELSE NULL
    END AS period_coarse
"""

con = duckdb.connect()

# ---------------------------------------------------------------
# Load the SWE frame into a DuckDB view with derived columns.
# ---------------------------------------------------------------
con.execute(f"""
CREATE OR REPLACE VIEW swe AS
SELECT *,
  {SENIORITY_CTE},
  {PERIOD_COARSE_CASE}
FROM '{PARQUET}'
WHERE {BASE_FILTER}
""")

n_total = con.execute("SELECT COUNT(*) FROM swe").fetchone()[0]
print(f"SWE frame (LinkedIn, English, date_flag=ok): {n_total:,} rows")

counts_by_period = con.execute(
    "SELECT period, source, COUNT(*) FROM swe GROUP BY 1,2 ORDER BY 1,2"
).fetchall()
print("\nRow counts by period/source:")
for r in counts_by_period:
    print(" ", r)

# ===================================================================
# 1. UNIVARIATE PROFILING
# ===================================================================
print("\n=== 1. Univariate profiling ===")

# ---------- 1a. Numeric summary by period ----------
numeric_cols = [
    'description_length', 'core_length', 'yoe_extracted',
    'swe_confidence', 'seniority_final_confidence'
]
numeric_rows = []
for col in numeric_cols:
    q = f"""
    SELECT period, COUNT(*) n,
           AVG({col}) mean, MEDIAN({col}) median,
           STDDEV({col}) sd,
           QUANTILE_CONT({col}, 0.10) p10,
           QUANTILE_CONT({col}, 0.25) p25,
           QUANTILE_CONT({col}, 0.75) p75,
           QUANTILE_CONT({col}, 0.90) p90,
           MIN({col}) min, MAX({col}) max,
           SUM(CASE WHEN {col} IS NOT NULL THEN 1 ELSE 0 END)*1.0/COUNT(*) coverage
    FROM swe
    GROUP BY 1 ORDER BY 1
    """
    df = con.execute(q).fetchdf()
    df['variable'] = col
    numeric_rows.append(df)
numeric_by_period = pd.concat(numeric_rows, ignore_index=True)
numeric_by_period.to_csv(f'{TAB_DIR}/01_numeric_summary_by_period.csv', index=False)
print(f"  -> 01_numeric_summary_by_period.csv ({len(numeric_by_period)} rows)")

# ---------- 1b. Numeric summary by period × seniority (combined) ----------
rows = []
for col in ['description_length', 'core_length', 'yoe_extracted']:
    q = f"""
    SELECT period_coarse AS period, COALESCE(seniority_best_available,'unknown') seniority,
           COUNT(*) n, AVG({col}) mean, MEDIAN({col}) median
    FROM swe
    WHERE period_coarse IS NOT NULL
    GROUP BY 1,2 ORDER BY 1,2
    """
    df = con.execute(q).fetchdf()
    df['variable'] = col
    rows.append(df)
numeric_sen = pd.concat(rows, ignore_index=True)
numeric_sen.to_csv(f'{TAB_DIR}/02_numeric_by_period_seniority.csv', index=False)
print(f"  -> 02_numeric_by_period_seniority.csv")

# ---------- 1c. Categorical distributions ----------
# Seniority distributions under full ablation
SENIORITY_VARIANTS = {
    'seniority_best_available': "COALESCE(seniority_best_available,'unknown')",
    'seniority_native': "COALESCE(seniority_native,'unknown')",
    'seniority_final': "COALESCE(seniority_final,'unknown')",
    'seniority_imputed': "COALESCE(seniority_imputed,'unknown')",
    'seniority_3level': "COALESCE(seniority_3level,'unknown')",
}
sen_rows = []
for variant, expr in SENIORITY_VARIANTS.items():
    q = f"""
    SELECT period, source, {expr} AS level, COUNT(*) n
    FROM swe
    GROUP BY 1,2,3 ORDER BY 1,2,3
    """
    df = con.execute(q).fetchdf()
    df['variant'] = variant
    sen_rows.append(df)
sen_dist = pd.concat(sen_rows, ignore_index=True)
sen_dist.to_csv(f'{TAB_DIR}/03_seniority_distribution_by_variant.csv', index=False)
print(f"  -> 03_seniority_distribution_by_variant.csv")

# is_aggregator by period
agg_df = con.execute("""
SELECT period, source, COUNT(*) n,
       AVG(CASE WHEN is_aggregator THEN 1.0 ELSE 0.0 END) aggregator_share
FROM swe GROUP BY 1,2 ORDER BY 1,2
""").fetchdf()
agg_df.to_csv(f'{TAB_DIR}/04_aggregator_share_by_period.csv', index=False)

# metro_area top 15 by period
metro_df = con.execute("""
SELECT period, metro_area, COUNT(*) n
FROM swe
WHERE metro_area IS NOT NULL
GROUP BY 1,2 ORDER BY 1, n DESC
""").fetchdf()
# Keep top 15 per period
metro_top = (
    metro_df.groupby('period', group_keys=False)
    .apply(lambda g: g.nlargest(15, 'n'))
    .reset_index(drop=True)
)
metro_top.to_csv(f'{TAB_DIR}/05_metro_top15_by_period.csv', index=False)

# company_industry top 15 by period (where available)
ind_df = con.execute("""
SELECT period, company_industry, COUNT(*) n
FROM swe
WHERE company_industry IS NOT NULL
GROUP BY 1,2 ORDER BY 1, n DESC
""").fetchdf()
ind_top = (
    ind_df.groupby('period', group_keys=False)
    .apply(lambda g: g.nlargest(15, 'n'))
    .reset_index(drop=True)
)
ind_top.to_csv(f'{TAB_DIR}/06_industry_top15_by_period.csv', index=False)

# swe_classification_tier
tier_df = con.execute("""
SELECT period, swe_classification_tier, COUNT(*) n
FROM swe GROUP BY 1,2 ORDER BY 1,2
""").fetchdf()
tier_df.to_csv(f'{TAB_DIR}/07_swe_tier_by_period.csv', index=False)

# is_remote_inferred and ghost_job_risk and ghost_assessment_llm
misc_df = con.execute("""
SELECT period,
       AVG(CASE WHEN is_remote_inferred THEN 1.0 ELSE 0.0 END) remote_share,
       AVG(CASE WHEN ghost_job_risk IS NOT NULL AND ghost_job_risk != 'low' THEN 1.0 ELSE 0.0 END) ghost_nonlow_share,
       AVG(CASE WHEN ghost_assessment_llm IS NOT NULL AND ghost_assessment_llm != 'realistic' THEN 1.0 ELSE 0.0 END) ghost_llm_nonreal_share,
       SUM(CASE WHEN ghost_assessment_llm IS NOT NULL THEN 1 ELSE 0 END)*1.0/COUNT(*) ghost_llm_coverage,
       AVG(CASE WHEN yoe_extracted IS NOT NULL THEN 1.0 ELSE 0.0 END) yoe_coverage
FROM swe GROUP BY 1 ORDER BY 1
""").fetchdf()
misc_df.to_csv(f'{TAB_DIR}/08_misc_flags_by_period.csv', index=False)

# company_size quartiles (arshkon only)
arshkon_size = con.execute("""
SELECT
  NTILE(4) OVER (ORDER BY company_size) AS size_q,
  *
FROM swe
WHERE source = 'kaggle_arshkon' AND company_size IS NOT NULL
""").fetchdf()
print(f"  arshkon SWE with company_size: {len(arshkon_size):,}")

# ===================================================================
# 2. ANOMALY DETECTION
# ===================================================================
print("\n=== 2. Anomaly detection ===")

anomaly_notes = []

# Bimodality check on description_length per period via Hartigan's dip-proxy:
# we'll use a simple kurtosis + gap heuristic and report distributions visually.
for col in ['description_length', 'core_length', 'yoe_extracted']:
    for period in ['2024-01','2024-04','2026-03','2026-04']:
        arr = con.execute(
            f"SELECT {col} FROM swe WHERE period = '{period}' AND {col} IS NOT NULL"
        ).fetchdf()[col].values
        if len(arr) < 100:
            continue
        sk = stats.skew(arr)
        kt = stats.kurtosis(arr)
        # Heuristic: heavy skew or heavy tails
        if abs(sk) > 2.0 or kt > 10:
            anomaly_notes.append({
                'variable': col, 'period': period,
                'n': len(arr), 'skew': round(float(sk),3),
                'kurtosis': round(float(kt),3),
                'flag': 'heavy_skew_or_tail'
            })

# yoe_seniority_contradiction by period
contra = con.execute("""
SELECT period, AVG(CASE WHEN yoe_seniority_contradiction THEN 1.0 ELSE 0.0 END) rate,
       SUM(CASE WHEN yoe_seniority_contradiction THEN 1 ELSE 0 END) n_contra,
       COUNT(*) total
FROM swe GROUP BY 1 ORDER BY 1
""").fetchdf()
contra.to_csv(f'{TAB_DIR}/09_yoe_contradiction_by_period.csv', index=False)
for _, row in contra.iterrows():
    if row['rate'] > 0.10:
        anomaly_notes.append({
            'variable': 'yoe_seniority_contradiction',
            'period': row['period'], 'n': int(row['total']),
            'skew': None, 'kurtosis': None,
            'flag': f"contradiction_rate={row['rate']:.3f}"
        })

pd.DataFrame(anomaly_notes).to_csv(f'{TAB_DIR}/10_anomaly_notes.csv', index=False)
print(f"  flagged anomalies: {len(anomaly_notes)}")

# ===================================================================
# 3. NATIVE-LABEL QUALITY DIAGNOSTIC
# ===================================================================
print("\n=== 3. Native-label quality diagnostic ===")

native_diag = con.execute("""
SELECT source, period, seniority_native AS level,
       COUNT(*) n,
       AVG(yoe_extracted) yoe_mean,
       MEDIAN(yoe_extracted) yoe_median,
       AVG(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) share_yoe_le2,
       AVG(CASE WHEN yoe_extracted >= 5 THEN 1.0 ELSE 0.0 END) share_yoe_ge5,
       SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) n_with_yoe
FROM swe
WHERE seniority_native IS NOT NULL
GROUP BY 1,2,3 ORDER BY 1,2,3
""").fetchdf()
native_diag.to_csv(f'{TAB_DIR}/11_native_label_yoe_profile.csv', index=False)
print(native_diag.to_string())

# ===================================================================
# 4. WITHIN-2024 CALIBRATION (arshkon vs asaniczka, mid-senior SWE)
# ===================================================================
print("\n=== 4. Within-2024 calibration (mid-senior SWE) ===")

# Use seniority_best_available = 'mid-senior' for both.
# (Asaniczka has no native entry, but has mid-senior fine.)
cal_q = """
SELECT source,
  COUNT(*) n,
  AVG(description_length) desc_len_mean,
  MEDIAN(description_length) desc_len_median,
  AVG(core_length) core_len_mean,
  AVG(yoe_extracted) yoe_mean,
  AVG(CASE WHEN is_aggregator THEN 1.0 ELSE 0.0 END) agg_share
FROM swe
WHERE period IN ('2024-01','2024-04')
  AND COALESCE(seniority_best_available,'unknown') = 'mid-senior'
GROUP BY 1
"""
cal_mid = con.execute(cal_q).fetchdf()
cal_mid.to_csv(f'{TAB_DIR}/12_within2024_midsenior_means.csv', index=False)
print(cal_mid.to_string())

# Cohen's d on description_length within mid-senior across arshkon vs asaniczka
def cohens_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a)-1)*sa + (len(b)-1)*sb) / (len(a)+len(b)-2))
    if pooled == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / pooled

# Extract arrays
arr_arsh = con.execute("""
SELECT description_length, core_length, yoe_extracted
FROM swe WHERE source='kaggle_arshkon'
  AND COALESCE(seniority_best_available,'unknown')='mid-senior'
""").fetchdf()
arr_asa = con.execute("""
SELECT description_length, core_length, yoe_extracted
FROM swe WHERE source='kaggle_asaniczka'
  AND COALESCE(seniority_best_available,'unknown')='mid-senior'
""").fetchdf()
arr_scr = con.execute("""
SELECT description_length, core_length, yoe_extracted
FROM swe WHERE source='scraped'
  AND COALESCE(seniority_best_available,'unknown')='mid-senior'
""").fetchdf()

calib_rows = []
for metric in ['description_length', 'core_length', 'yoe_extracted']:
    d_within = cohens_d(arr_arsh[metric], arr_asa[metric])
    d_arsh_scr = cohens_d(arr_arsh[metric], arr_scr[metric])
    d_asa_scr = cohens_d(arr_asa[metric], arr_scr[metric])
    calib_rows.append({
        'metric': metric,
        'd_within2024_arsh_vs_asa': round(float(d_within), 4) if not np.isnan(d_within) else None,
        'd_arsh_vs_scraped': round(float(d_arsh_scr), 4) if not np.isnan(d_arsh_scr) else None,
        'd_asa_vs_scraped': round(float(d_asa_scr), 4) if not np.isnan(d_asa_scr) else None,
        'ratio_crossperiod_over_within': (
            round(abs(d_asa_scr)/abs(d_within), 2) if d_within and not np.isnan(d_within) and d_within != 0 else None
        ),
        'n_arsh': int(len(arr_arsh)),
        'n_asa': int(len(arr_asa)),
        'n_scr': int(len(arr_scr)),
    })
calib_df = pd.DataFrame(calib_rows)
calib_df.to_csv(f'{TAB_DIR}/13_calibration_midsenior_cohensd.csv', index=False)
print(calib_df.to_string())

# ===================================================================
# 5. ENTRY SHARE TRENDS - FULL ABLATION
# ===================================================================
print("\n=== 5. Entry share under full ablation ===")

# Helper: entry share for a given expression, filter, period grouping
def entry_share(expr, extra_where=""):
    q = f"""
    SELECT period_coarse AS period,
           COUNT(*) n_all,
           SUM(CASE WHEN {expr} IS NOT NULL AND {expr} != 'unknown' THEN 1 ELSE 0 END) n_known,
           SUM(CASE WHEN {expr} = 'entry' THEN 1 ELSE 0 END) n_entry,
           SUM(CASE WHEN {expr} = 'entry' THEN 1 ELSE 0 END)*1.0/COUNT(*) share_of_all,
           SUM(CASE WHEN {expr} = 'entry' THEN 1 ELSE 0 END)*1.0/
             NULLIF(SUM(CASE WHEN {expr} IS NOT NULL AND {expr} != 'unknown' THEN 1 ELSE 0 END),0) share_of_known
    FROM swe
    WHERE period_coarse IS NOT NULL {extra_where}
    GROUP BY 1 ORDER BY 1
    """
    return con.execute(q).fetchdf()

# Combined best-available (asaniczka included — allowed per rule)
best_df = entry_share("seniority_best_available")
best_df['variant'] = 'seniority_best_available (2024 pooled)'

# seniority_native (arshkon-only 2024 baseline)
nat_df = entry_share(
    "seniority_native",
    extra_where=" AND (period_coarse = '2026' OR source = 'kaggle_arshkon')"
)
nat_df['variant'] = 'seniority_native (arshkon-only 2024)'

# seniority_final (arshkon-only 2024 baseline)
fin_df = entry_share(
    "seniority_final",
    extra_where=" AND (period_coarse = '2026' OR source = 'kaggle_arshkon')"
)
fin_df['variant'] = 'seniority_final (arshkon-only 2024)'

# seniority_imputed where != unknown (pooled 2024 ok)
imp_df = entry_share("seniority_imputed")
imp_df['variant'] = 'seniority_imputed (2024 pooled)'

# YOE proxy: yoe_extracted <= 2 and <= 3
yoe2_q = """
SELECT period_coarse AS period,
       COUNT(*) n_all,
       SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) n_with_yoe,
       SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) n_le2,
       SUM(CASE WHEN yoe_extracted <= 3 THEN 1 ELSE 0 END) n_le3,
       SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END)*1.0/COUNT(*) share_le2_of_all,
       SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END)*1.0/
         NULLIF(SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END),0) share_le2_of_yoe,
       SUM(CASE WHEN yoe_extracted <= 3 THEN 1 ELSE 0 END)*1.0/
         NULLIF(SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END),0) share_le3_of_yoe
FROM swe
WHERE period_coarse IS NOT NULL
GROUP BY 1 ORDER BY 1
"""
yoe_df = con.execute(yoe2_q).fetchdf()
yoe_df.to_csv(f'{TAB_DIR}/15_yoe_proxy_entry_share.csv', index=False)
print("YOE proxy entry share:")
print(yoe_df.to_string())

entry_ablation = pd.concat([best_df, nat_df, fin_df, imp_df], ignore_index=True)
entry_ablation.to_csv(f'{TAB_DIR}/14_entry_share_ablation.csv', index=False)
print("\nEntry share ablation (label-based):")
print(entry_ablation.to_string())

# ===================================================================
# 6. RANKED CHANGES BETWEEN PERIODS (2024 pooled -> 2026 pooled)
# ===================================================================
print("\n=== 6. Ranked change list 2024 -> 2026 ===")

# Build a list of metrics (binary / continuous means)
# For continuous use Cohen's d; for binary use absolute difference in proportion.
binary_metrics_sql = {
    'is_aggregator': "AVG(CASE WHEN is_aggregator THEN 1.0 ELSE 0.0 END)",
    'is_remote_inferred': "AVG(CASE WHEN is_remote_inferred THEN 1.0 ELSE 0.0 END)",
    'is_multi_location': "AVG(CASE WHEN is_multi_location THEN 1.0 ELSE 0.0 END)",
    'has_yoe': "AVG(CASE WHEN yoe_extracted IS NOT NULL THEN 1.0 ELSE 0.0 END)",
    'yoe_contradiction': "AVG(CASE WHEN yoe_seniority_contradiction THEN 1.0 ELSE 0.0 END)",
    'senrate_entry_best': "AVG(CASE WHEN seniority_best_available='entry' THEN 1.0 ELSE 0.0 END)",
    'senrate_entry_imputed': "AVG(CASE WHEN seniority_imputed='entry' THEN 1.0 ELSE 0.0 END)",
    'senrate_entry_final': "AVG(CASE WHEN seniority_final='entry' THEN 1.0 ELSE 0.0 END)",
    'senrate_midsenior_best': "AVG(CASE WHEN seniority_best_available='mid-senior' THEN 1.0 ELSE 0.0 END)",
    'senrate_director_best': "AVG(CASE WHEN seniority_best_available='director' THEN 1.0 ELSE 0.0 END)",
    'yoe_le2_share': "AVG(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END)",
    'yoe_le3_share': "AVG(CASE WHEN yoe_extracted <= 3 THEN 1.0 ELSE 0.0 END)",
}
# Binary keyword density in description (boilerplate-inclusive, quick signal)
kw_pats = {
    'desc_contains_ai': "LOWER(description) LIKE '%artificial intelligence%' OR LOWER(description) LIKE '% ai %' OR LOWER(description) LIKE '% genai %' OR LOWER(description) LIKE '%generative ai%'",
    'desc_contains_llm': "LOWER(description) LIKE '% llm %' OR LOWER(description) LIKE '% llms %' OR LOWER(description) LIKE '%large language model%'",
    'desc_contains_copilot': "LOWER(description) LIKE '%copilot%'",
    'desc_contains_cursor': "LOWER(description) LIKE '% cursor ide%' OR LOWER(description) LIKE '%cursor editor%' OR LOWER(description) LIKE '%using cursor%'",
    'desc_contains_agent': "LOWER(description) LIKE '% agent %' OR LOWER(description) LIKE '% agents %' OR LOWER(description) LIKE '% agentic%'",
    'desc_contains_rag': "LOWER(description) LIKE '% rag %' OR LOWER(description) LIKE '%retrieval augmented%' OR LOWER(description) LIKE '%retrieval-augmented%'",
    'desc_contains_prompt': "LOWER(description) LIKE '%prompt engineer%'",
    'desc_contains_python': "LOWER(description) LIKE '%python%'",
    'desc_contains_typescript': "LOWER(description) LIKE '%typescript%'",
    'desc_contains_react': "LOWER(description) LIKE '%react%'",
    'desc_contains_kubernetes': "LOWER(description) LIKE '%kubernetes%' OR LOWER(description) LIKE '%k8s%'",
    'desc_contains_terraform': "LOWER(description) LIKE '%terraform%'",
    'desc_contains_ownership': "LOWER(description) LIKE '%ownership%' OR LOWER(description) LIKE '% own the %'",
    'desc_contains_lead': "LOWER(description) LIKE '% lead %' OR LOWER(description) LIKE '%leading%'",
    'desc_contains_scope': "LOWER(description) LIKE '% scope %' OR LOWER(description) LIKE '%cross-functional%'",
}
for name, pat in kw_pats.items():
    binary_metrics_sql[name] = f"AVG(CASE WHEN {pat} THEN 1.0 ELSE 0.0 END)"

continuous_metrics_sql = {
    'description_length': 'description_length',
    'core_length': 'core_length',
    'yoe_extracted': 'yoe_extracted',
}

def metric_by_period_coarse(sql_expr, metric_name, kind):
    q = f"""
    SELECT period_coarse AS period, COUNT(*) n, {sql_expr} val
    FROM swe
    WHERE period_coarse IS NOT NULL
    GROUP BY 1 ORDER BY 1
    """
    return con.execute(q).fetchdf()

ranked = []
for name, expr in binary_metrics_sql.items():
    df = metric_by_period_coarse(expr, name, 'binary')
    try:
        v24 = float(df.loc[df['period']=='2024','val'].values[0])
        v26 = float(df.loc[df['period']=='2026','val'].values[0])
    except (IndexError, TypeError):
        continue
    abs_diff = v26 - v24
    rel_diff = (v26 - v24) / v24 if v24 else np.nan
    ranked.append({
        'metric': name, 'kind': 'binary',
        'val_2024': round(v24,5), 'val_2026': round(v26,5),
        'abs_diff': round(abs_diff,5),
        'rel_diff_pct': round(rel_diff*100,2) if not np.isnan(rel_diff) else None,
        'effect_size': round(abs_diff,5),
    })

# Continuous: compute Cohen's d between pooled 2024 and pooled 2026 arrays
for name, col in continuous_metrics_sql.items():
    df = con.execute(f"""
    SELECT period_coarse AS period, {col} AS val
    FROM swe
    WHERE period_coarse IS NOT NULL AND {col} IS NOT NULL
    """).fetchdf()
    a = df.loc[df['period']=='2024','val'].values
    b = df.loc[df['period']=='2026','val'].values
    if len(a) < 2 or len(b) < 2:
        continue
    d = cohens_d(a, b)  # 2024 - 2026 direction
    ranked.append({
        'metric': name, 'kind': 'continuous',
        'val_2024': round(float(np.mean(a)),3),
        'val_2026': round(float(np.mean(b)),3),
        'abs_diff': round(float(np.mean(b)-np.mean(a)),3),
        'rel_diff_pct': round((np.mean(b)-np.mean(a))/np.mean(a)*100,2) if np.mean(a) else None,
        'effect_size': round(float(d),4) if not np.isnan(d) else None,
    })

ranked_df = pd.DataFrame(ranked).sort_values(
    by='effect_size', key=lambda s: s.abs(), ascending=False
).reset_index(drop=True)
ranked_df.to_csv(f'{TAB_DIR}/16_ranked_changes_2024_to_2026.csv', index=False)
print(ranked_df.head(20).to_string())

# ===================================================================
# 7. DOMAIN × SENIORITY — skip (T09 labels not available yet)
# ===================================================================
print("\n=== 7. Domain × seniority: T09 labels not available, skipping ===")

# ===================================================================
# 8. COMPANY SIZE STRATIFICATION (arshkon only)
# ===================================================================
print("\n=== 8. Company size stratification (arshkon only) ===")

# Quartile entry share, AI keyword, tech count proxies
size_q = """
WITH arsh AS (
  SELECT *,
    NTILE(4) OVER (ORDER BY company_size) AS size_q
  FROM swe
  WHERE source='kaggle_arshkon' AND company_size IS NOT NULL
)
SELECT size_q,
  MIN(company_size) min_size,
  MAX(company_size) max_size,
  COUNT(*) n,
  AVG(CASE WHEN seniority_best_available='entry' THEN 1.0 ELSE 0.0 END) entry_best_share,
  AVG(CASE WHEN seniority_native='entry' THEN 1.0 ELSE 0.0 END) entry_native_share,
  AVG(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) yoe_le2_share,
  AVG(description_length) desc_len_mean,
  AVG(core_length) core_len_mean,
  AVG(CASE WHEN LOWER(description) LIKE '%copilot%' OR LOWER(description) LIKE '%generative ai%' THEN 1.0 ELSE 0.0 END) ai_kw_share
FROM arsh
GROUP BY 1 ORDER BY 1
"""
size_df = con.execute(size_q).fetchdf()
size_df.to_csv(f'{TAB_DIR}/17_arshkon_company_size_quartiles.csv', index=False)
print(size_df.to_string())

# Cross-period: posting volume per company as proxy
pv_q = """
WITH pcnt AS (
  SELECT period_coarse AS period, company_name_canonical,
         COUNT(*) n_posts
  FROM swe
  WHERE company_name_canonical IS NOT NULL AND period_coarse IS NOT NULL
  GROUP BY 1,2
)
SELECT period,
  AVG(n_posts) mean_posts_per_company,
  MEDIAN(n_posts) median_posts_per_company,
  QUANTILE_CONT(n_posts, 0.90) p90_posts_per_company,
  MAX(n_posts) max_posts,
  COUNT(*) n_companies
FROM pcnt GROUP BY 1 ORDER BY 1
"""
pv_df = con.execute(pv_q).fetchdf()
pv_df.to_csv(f'{TAB_DIR}/18_posting_volume_per_company.csv', index=False)
print(pv_df.to_string())

# ===================================================================
# FIGURES (max 4)
# ===================================================================
print("\n=== Generating figures ===")

sns.set_style('whitegrid')

# Figure 1: Numeric distributions by period (4 subpanels)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col, clip in zip(
    axes.flat,
    ['description_length', 'core_length', 'yoe_extracted', 'seniority_final_confidence'],
    [20000, 20000, 20, 1.0],
):
    for period, colour in zip(['2024-01','2024-04','2026-03','2026-04'],
                              ['#4c72b0','#dd8452','#55a868','#c44e52']):
        arr = con.execute(
            f"SELECT {col} FROM swe WHERE period='{period}' AND {col} IS NOT NULL"
        ).fetchdf()[col].values
        if len(arr) == 0:
            continue
        arr = arr[arr <= clip]
        ax.hist(arr, bins=60, alpha=0.45, label=period, color=colour, density=True)
    ax.set_title(col)
    ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig1_numeric_distributions.png', dpi=150)
plt.close()
print("  fig1_numeric_distributions.png")

# Figure 2: Entry share ablation
fig, ax = plt.subplots(figsize=(10, 6))
variants_for_plot = {
    'seniority_best_available (2024 pooled)': ('#1f77b4', 'share_of_all'),
    'seniority_native (arshkon-only 2024)': ('#ff7f0e', 'share_of_all'),
    'seniority_final (arshkon-only 2024)': ('#2ca02c', 'share_of_all'),
    'seniority_imputed (2024 pooled)': ('#d62728', 'share_of_all'),
}
x_lab = ['2024', '2026']
for variant, (c, col) in variants_for_plot.items():
    sub = entry_ablation[entry_ablation['variant'] == variant]
    if len(sub) == 0:
        continue
    y = [sub.loc[sub['period']==p, col].values[0] if len(sub.loc[sub['period']==p])>0 else np.nan for p in x_lab]
    ax.plot(x_lab, y, marker='o', label=variant, color=c, linewidth=2)
# YOE proxy
y_yoe2 = [yoe_df.loc[yoe_df['period']==p,'share_le2_of_yoe'].values[0] if len(yoe_df.loc[yoe_df['period']==p])>0 else np.nan for p in x_lab]
ax.plot(x_lab, y_yoe2, marker='s', label='YOE proxy (yoe<=2 of yoe-known)',
        color='black', linewidth=2, linestyle='--')
ax.set_ylabel('Entry share')
ax.set_title('Entry share by period — full seniority ablation')
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(0, max(0.35, ax.get_ylim()[1]))
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig2_entry_share_ablation.png', dpi=150)
plt.close()
print("  fig2_entry_share_ablation.png")

# Figure 3: Native-entry YOE profile (stacked by source)
fig, ax = plt.subplots(figsize=(10, 6))
ne = native_diag[native_diag['level']=='entry'].copy()
ne['label'] = ne['source'] + ' ' + ne['period']
x = np.arange(len(ne))
w = 0.35
ax.bar(x - w/2, ne['share_yoe_le2'], w, label='share YOE<=2', color='#2ca02c')
ax.bar(x + w/2, ne['share_yoe_ge5'], w, label='share YOE>=5', color='#d62728')
ax.set_xticks(x)
ax.set_xticklabels(ne['label'], rotation=30, ha='right')
ax.set_ylabel('Share of native-entry rows')
ax.set_title('Native-entry YOE profile by source (label quality diagnostic)')
for i, (le2, ge5, n) in enumerate(zip(ne['share_yoe_le2'], ne['share_yoe_ge5'], ne['n'])):
    ax.text(i, max(le2, ge5)+0.02, f"n={n}", ha='center', fontsize=8)
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig3_native_entry_yoe_profile.png', dpi=150)
plt.close()
print("  fig3_native_entry_yoe_profile.png")

# Figure 4: Ranked changes top 15
fig, ax = plt.subplots(figsize=(10, 8))
top = ranked_df.dropna(subset=['effect_size']).head(15).iloc[::-1]
colors = ['#2ca02c' if v >= 0 else '#d62728' for v in top['effect_size']]
ax.barh(top['metric'], top['effect_size'], color=colors)
ax.set_xlabel('Effect size (Cohen d for continuous, abs diff for binary)')
ax.set_title('Top 15 metrics by absolute change: 2024 -> 2026')
ax.axvline(0, color='k', linewidth=0.6)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig4_ranked_changes.png', dpi=150)
plt.close()
print("  fig4_ranked_changes.png")

# ===================================================================
# SENSITIVITY: aggregator exclusion + company capping
# ===================================================================
print("\n=== Sensitivities ===")

# Aggregator-excluded entry share (best-available)
aex = con.execute("""
SELECT period_coarse AS period,
  SUM(CASE WHEN seniority_best_available='entry' THEN 1 ELSE 0 END)*1.0/COUNT(*) share_all,
  COUNT(*) n
FROM swe
WHERE period_coarse IS NOT NULL AND is_aggregator = false
GROUP BY 1 ORDER BY 1
""").fetchdf()
aex['variant'] = 'aggregator_excluded'

# Company capped: cap each company to max 20 postings per period
cap_q = """
WITH ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY period_coarse, company_name_canonical ORDER BY uid) AS rn
  FROM swe WHERE period_coarse IS NOT NULL
)
SELECT period_coarse AS period,
  SUM(CASE WHEN seniority_best_available='entry' THEN 1 ELSE 0 END)*1.0/COUNT(*) share_all,
  COUNT(*) n
FROM ranked
WHERE rn <= 20
GROUP BY 1 ORDER BY 1
"""
cap_df = con.execute(cap_q).fetchdf()
cap_df['variant'] = 'company_capped_20'

sens_df = pd.concat([aex, cap_df], ignore_index=True)
sens_df.to_csv(f'{TAB_DIR}/19_sensitivity_entry_best.csv', index=False)
print(sens_df.to_string())

# Source-excluded: arshkon-only 2024 combined with scraped 2026 (best-available)
src_q = """
SELECT period_coarse AS period, source,
  SUM(CASE WHEN seniority_best_available='entry' THEN 1 ELSE 0 END)*1.0/COUNT(*) share_all,
  SUM(CASE WHEN seniority_best_available='entry' THEN 1 ELSE 0 END)*1.0/
    NULLIF(SUM(CASE WHEN seniority_best_available IS NOT NULL AND seniority_best_available != 'unknown' THEN 1 ELSE 0 END), 0) share_known,
  COUNT(*) n
FROM swe
WHERE period_coarse IS NOT NULL
GROUP BY 1,2 ORDER BY 1,2
"""
src_df = con.execute(src_q).fetchdf()
src_df.to_csv(f'{TAB_DIR}/20_source_split_entry_best.csv', index=False)
print(src_df.to_string())

print("\nT08 done.")
