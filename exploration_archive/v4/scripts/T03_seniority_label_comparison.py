#!/usr/bin/env python3
"""T03: Seniority label comparison across all variants.

Cross-tabulates seniority columns, computes agreement/kappa,
assesses per-class accuracy of rule-based vs native,
and computes RQ1 junior share under multiple operationalizations.
"""
import duckdb
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
import json

BASE = "/home/jihgaboot/gabor/job-research"
FIG_DIR = f"{BASE}/exploration/figures/T03"
TBL_DIR = f"{BASE}/exploration/tables/T03"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

con = duckdb.connect()
PARQUET = f"'{BASE}/data/unified.parquet'"
BASE_FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"
SWE_FILTER = f"{BASE_FILTER} AND is_swe = true"

# ============================================================
# 1. Cross-tabulations of seniority variants (SWE rows)
# ============================================================
print("=" * 60)
print("STEP 1: Cross-tabulations of seniority variants")
print("=" * 60)

# 1a. seniority_native vs seniority_final (where native is non-null)
print("\n--- seniority_native vs seniority_final (both non-null) ---")
ct_nf = con.execute(f"""
    SELECT seniority_native, seniority_final, count(*) as n
    FROM {PARQUET}
    WHERE {SWE_FILTER}
      AND seniority_native IS NOT NULL
    GROUP BY seniority_native, seniority_final
    ORDER BY seniority_native, seniority_final
""").fetchdf()
ct_nf_pivot = ct_nf.pivot_table(index='seniority_native', columns='seniority_final', values='n', fill_value=0, aggfunc='sum')
print(ct_nf_pivot)
ct_nf_pivot.to_csv(f"{TBL_DIR}/crosstab_native_vs_final.csv")

# 1b. seniority_imputed vs seniority_final (where imputed != unknown)
print("\n--- seniority_imputed vs seniority_final (imputed != unknown) ---")
ct_if = con.execute(f"""
    SELECT seniority_imputed, seniority_final, count(*) as n
    FROM {PARQUET}
    WHERE {SWE_FILTER}
      AND seniority_imputed != 'unknown'
    GROUP BY seniority_imputed, seniority_final
    ORDER BY seniority_imputed, seniority_final
""").fetchdf()
ct_if_pivot = ct_if.pivot_table(index='seniority_imputed', columns='seniority_final', values='n', fill_value=0, aggfunc='sum')
print(ct_if_pivot)
ct_if_pivot.to_csv(f"{TBL_DIR}/crosstab_imputed_vs_final.csv")

# 1c. seniority_native vs seniority_imputed (where both non-null/unknown)
print("\n--- seniority_native vs seniority_imputed (both non-null, imputed != unknown) ---")
ct_ni = con.execute(f"""
    SELECT seniority_native, seniority_imputed, count(*) as n
    FROM {PARQUET}
    WHERE {SWE_FILTER}
      AND seniority_native IS NOT NULL
      AND seniority_imputed != 'unknown'
    GROUP BY seniority_native, seniority_imputed
    ORDER BY seniority_native, seniority_imputed
""").fetchdf()
ct_ni_pivot = ct_ni.pivot_table(index='seniority_native', columns='seniority_imputed', values='n', fill_value=0, aggfunc='sum')
print(ct_ni_pivot)
ct_ni_pivot.to_csv(f"{TBL_DIR}/crosstab_native_vs_imputed.csv")

# 1d. seniority_llm vs seniority_final (where llm labeled, llm != unknown)
print("\n--- seniority_llm vs seniority_final (llm labeled, llm != unknown) ---")
ct_lf = con.execute(f"""
    SELECT seniority_llm, seniority_final, count(*) as n
    FROM {PARQUET}
    WHERE {SWE_FILTER}
      AND llm_classification_coverage = 'labeled'
      AND seniority_llm != 'unknown'
    GROUP BY seniority_llm, seniority_final
    ORDER BY seniority_llm, seniority_final
""").fetchdf()
if len(ct_lf) > 0:
    ct_lf_pivot = ct_lf.pivot_table(index='seniority_llm', columns='seniority_final', values='n', fill_value=0, aggfunc='sum')
    print(ct_lf_pivot)
    ct_lf_pivot.to_csv(f"{TBL_DIR}/crosstab_llm_vs_final.csv")
else:
    print("No rows with seniority_llm labeled and non-unknown")

# 1e. seniority_llm vs seniority_native (where both available)
print("\n--- seniority_llm vs seniority_native (llm labeled, both non-null/unknown) ---")
ct_ln = con.execute(f"""
    SELECT seniority_llm, seniority_native, count(*) as n
    FROM {PARQUET}
    WHERE {SWE_FILTER}
      AND llm_classification_coverage = 'labeled'
      AND seniority_llm != 'unknown'
      AND seniority_native IS NOT NULL
    GROUP BY seniority_llm, seniority_native
    ORDER BY seniority_llm, seniority_native
""").fetchdf()
if len(ct_ln) > 0:
    ct_ln_pivot = ct_ln.pivot_table(index='seniority_llm', columns='seniority_native', values='n', fill_value=0, aggfunc='sum')
    print(ct_ln_pivot)
    ct_ln_pivot.to_csv(f"{TBL_DIR}/crosstab_llm_vs_native.csv")

# ============================================================
# 2. Agreement rate and Cohen's kappa for each pair
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Agreement rates and Cohen's kappa")
print("=" * 60)

def compute_agreement(df, col_a, col_b):
    """Compute agreement rate and Cohen's kappa from a crosstab df."""
    total = df['n'].sum()
    agree = df[df[col_a] == df[col_b]]['n'].sum()
    agreement_rate = agree / total if total > 0 else 0

    # Expand for kappa
    labels_a = []
    labels_b = []
    for _, row in df.iterrows():
        labels_a.extend([row[col_a]] * int(row['n']))
        labels_b.extend([row[col_b]] * int(row['n']))

    if len(set(labels_a)) < 2 and len(set(labels_b)) < 2:
        kappa = float('nan')
    else:
        kappa = cohen_kappa_score(labels_a, labels_b)

    return agreement_rate, kappa, total

pairs = [
    ("seniority_native", "seniority_final", ct_nf),
    ("seniority_imputed", "seniority_final", ct_if),
    ("seniority_native", "seniority_imputed", ct_ni),
]

# Add LLM pairs if available
if len(ct_lf) > 0:
    pairs.append(("seniority_llm", "seniority_final", ct_lf))
if len(ct_ln) > 0:
    pairs.append(("seniority_llm", "seniority_native", ct_ln))

agreement_results = []
for col_a, col_b, df in pairs:
    rate, kappa, n = compute_agreement(df, col_a, col_b)
    print(f"  {col_a} vs {col_b}: agreement={rate:.3f}, kappa={kappa:.3f}, n={n}")
    agreement_results.append({
        'col_a': col_a, 'col_b': col_b,
        'agreement_rate': round(rate, 4), 'kappa': round(kappa, 4), 'n': n
    })

pd.DataFrame(agreement_results).to_csv(f"{TBL_DIR}/agreement_kappa.csv", index=False)

# ============================================================
# 3. Per-class accuracy: rule-based (imputed) vs native as ground truth
#    For Arshkon SWE
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Per-class accuracy of rule-based vs native (Arshkon SWE)")
print("=" * 60)

arshkon_eval = con.execute(f"""
    SELECT seniority_native as true_label, seniority_imputed as pred_label, count(*) as n
    FROM {PARQUET}
    WHERE {SWE_FILTER}
      AND source = 'kaggle_arshkon'
      AND seniority_native IS NOT NULL
      AND seniority_imputed != 'unknown'
    GROUP BY seniority_native, seniority_imputed
""").fetchdf()

print(f"Arshkon SWE with native + non-unknown imputed: {arshkon_eval['n'].sum()} rows")

if len(arshkon_eval) > 0:
    true_a = []
    pred_a = []
    for _, row in arshkon_eval.iterrows():
        true_a.extend([row['true_label']] * int(row['n']))
        pred_a.extend([row['pred_label']] * int(row['n']))

    labels_order = ['entry', 'associate', 'mid-senior', 'director']
    present_labels = sorted(set(true_a + pred_a), key=lambda x: labels_order.index(x) if x in labels_order else 99)

    print("\nClassification report (arshkon, imputed vs native):")
    print(classification_report(true_a, pred_a, labels=present_labels, zero_division=0))

    # Save confusion matrix
    cm = confusion_matrix(true_a, pred_a, labels=present_labels)
    cm_df = pd.DataFrame(cm, index=present_labels, columns=present_labels)
    cm_df.index.name = 'native (true)'
    cm_df.columns.name = 'imputed (pred)'
    print(cm_df)
    cm_df.to_csv(f"{TBL_DIR}/confusion_matrix_arshkon_imputed_vs_native.csv")

# Also get arshkon coverage: how many have native, how many have non-unknown imputed
arshkon_coverage = con.execute(f"""
    SELECT
      count(*) as total_swe,
      count(seniority_native) as has_native,
      sum(CASE WHEN seniority_imputed != 'unknown' THEN 1 ELSE 0 END) as has_imputed,
      sum(CASE WHEN seniority_native IS NOT NULL AND seniority_imputed != 'unknown' THEN 1 ELSE 0 END) as has_both
    FROM {PARQUET}
    WHERE {SWE_FILTER} AND source = 'kaggle_arshkon'
""").fetchdf()
print("\nArshkon SWE coverage:")
print(arshkon_coverage.to_string())

# ============================================================
# 4. Same for scraped LinkedIn SWE (temporal instability check)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Per-class accuracy of rule-based vs native (Scraped SWE)")
print("=" * 60)

scraped_eval = con.execute(f"""
    SELECT seniority_native as true_label, seniority_imputed as pred_label, count(*) as n
    FROM {PARQUET}
    WHERE {SWE_FILTER}
      AND source = 'scraped'
      AND seniority_native IS NOT NULL
      AND seniority_imputed != 'unknown'
    GROUP BY seniority_native, seniority_imputed
""").fetchdf()

print(f"Scraped SWE with native + non-unknown imputed: {scraped_eval['n'].sum()} rows")

if len(scraped_eval) > 0:
    true_s = []
    pred_s = []
    for _, row in scraped_eval.iterrows():
        true_s.extend([row['true_label']] * int(row['n']))
        pred_s.extend([row['pred_label']] * int(row['n']))

    present_labels_s = sorted(set(true_s + pred_s), key=lambda x: labels_order.index(x) if x in labels_order else 99)

    print("\nClassification report (scraped, imputed vs native):")
    print(classification_report(true_s, pred_s, labels=present_labels_s, zero_division=0))

    cm_s = confusion_matrix(true_s, pred_s, labels=present_labels_s)
    cm_s_df = pd.DataFrame(cm_s, index=present_labels_s, columns=present_labels_s)
    cm_s_df.index.name = 'native (true)'
    cm_s_df.columns.name = 'imputed (pred)'
    print(cm_s_df)
    cm_s_df.to_csv(f"{TBL_DIR}/confusion_matrix_scraped_imputed_vs_native.csv")

# Scraped coverage
scraped_coverage = con.execute(f"""
    SELECT
      count(*) as total_swe,
      count(seniority_native) as has_native,
      sum(CASE WHEN seniority_imputed != 'unknown' THEN 1 ELSE 0 END) as has_imputed,
      sum(CASE WHEN seniority_native IS NOT NULL AND seniority_imputed != 'unknown' THEN 1 ELSE 0 END) as has_both
    FROM {PARQUET}
    WHERE {SWE_FILTER} AND source = 'scraped'
""").fetchdf()
print("\nScraped SWE coverage:")
print(scraped_coverage.to_string())

# ============================================================
# 4b. Asaniczka evaluation (to understand the entry-level gap)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4b: Asaniczka seniority profile")
print("=" * 60)

asaniczka_native = con.execute(f"""
    SELECT seniority_native, count(*) as n
    FROM {PARQUET}
    WHERE {SWE_FILTER} AND source = 'kaggle_asaniczka'
    GROUP BY seniority_native ORDER BY n DESC
""").fetchdf()
print("Asaniczka SWE seniority_native:")
print(asaniczka_native.to_string())

asaniczka_final = con.execute(f"""
    SELECT seniority_final, count(*) as n
    FROM {PARQUET}
    WHERE {SWE_FILTER} AND source = 'kaggle_asaniczka'
    GROUP BY seniority_final ORDER BY n DESC
""").fetchdf()
print("\nAsaniczka SWE seniority_final:")
print(asaniczka_final.to_string())

asaniczka_llm = con.execute(f"""
    SELECT seniority_llm, count(*) as n
    FROM {PARQUET}
    WHERE {SWE_FILTER} AND source = 'kaggle_asaniczka'
      AND llm_classification_coverage = 'labeled'
    GROUP BY seniority_llm ORDER BY n DESC
""").fetchdf()
print("\nAsaniczka SWE seniority_llm (labeled only):")
print(asaniczka_llm.to_string())

# ============================================================
# 5. RQ1 Junior share under 4+ operationalizations
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: RQ1 Junior share comparison across operationalizations")
print("=" * 60)

# 5a. seniority_final (all non-unknown)
print("\n--- 5a: seniority_final (all non-unknown) ---")
q5a = con.execute(f"""
    SELECT source, period, seniority_final, count(*) as n
    FROM {PARQUET}
    WHERE {SWE_FILTER} AND seniority_final != 'unknown'
    GROUP BY source, period, seniority_final
    ORDER BY source, period, seniority_final
""").fetchdf()
print(q5a.to_string())
q5a.to_csv(f"{TBL_DIR}/rq1_seniority_final_by_source_period.csv", index=False)

# 5b. seniority_native only (non-null)
print("\n--- 5b: seniority_native only (non-null) ---")
q5b = con.execute(f"""
    SELECT source, period, seniority_native, count(*) as n
    FROM {PARQUET}
    WHERE {SWE_FILTER} AND seniority_native IS NOT NULL
    GROUP BY source, period, seniority_native
    ORDER BY source, period, seniority_native
""").fetchdf()
print(q5b.to_string())
q5b.to_csv(f"{TBL_DIR}/rq1_seniority_native_by_source_period.csv", index=False)

# 5c. High-confidence: seniority_final_source IN ('title_keyword', 'native_backfill')
print("\n--- 5c: seniority_final high-confidence (title_keyword OR native_backfill) ---")
q5c = con.execute(f"""
    SELECT source, period, seniority_final, count(*) as n
    FROM {PARQUET}
    WHERE {SWE_FILTER}
      AND seniority_final != 'unknown'
      AND seniority_final_source IN ('title_keyword', 'native_backfill')
    GROUP BY source, period, seniority_final
    ORDER BY source, period, seniority_final
""").fetchdf()
print(q5c.to_string())
q5c.to_csv(f"{TBL_DIR}/rq1_seniority_final_highconf_by_source_period.csv", index=False)

# 5d. Including weak signals (all seniority_final, even weak title ones)
print("\n--- 5d: seniority_final including weak signals ---")
q5d = con.execute(f"""
    SELECT source, period, seniority_final, seniority_final_source, count(*) as n
    FROM {PARQUET}
    WHERE {SWE_FILTER} AND seniority_final != 'unknown'
    GROUP BY source, period, seniority_final, seniority_final_source
    ORDER BY source, period, seniority_final, seniority_final_source
""").fetchdf()
print(q5d.to_string())
q5d.to_csv(f"{TBL_DIR}/rq1_seniority_final_with_sources.csv", index=False)

# 5e. seniority_llm (labeled only)
print("\n--- 5e: seniority_llm (labeled only, non-unknown) ---")
q5e = con.execute(f"""
    SELECT source, period, seniority_llm, count(*) as n
    FROM {PARQUET}
    WHERE {SWE_FILTER}
      AND llm_classification_coverage = 'labeled'
      AND seniority_llm != 'unknown'
    GROUP BY source, period, seniority_llm
    ORDER BY source, period, seniority_llm
""").fetchdf()
print(q5e.to_string())
q5e.to_csv(f"{TBL_DIR}/rq1_seniority_llm_by_source_period.csv", index=False)

# 5f. seniority_llm including rule_sufficient
print("\n--- 5f: seniority_llm + rule_sufficient (best-available) ---")
q5f = con.execute(f"""
    SELECT source, period,
      CASE WHEN llm_classification_coverage = 'labeled' THEN seniority_llm
           WHEN llm_classification_coverage = 'rule_sufficient' THEN seniority_final
           ELSE NULL END as seniority_best,
      llm_classification_coverage as coverage_type,
      count(*) as n
    FROM {PARQUET}
    WHERE {SWE_FILTER}
      AND llm_classification_coverage IN ('labeled', 'rule_sufficient')
    GROUP BY source, period, seniority_best, coverage_type
    ORDER BY source, period, seniority_best
""").fetchdf()
print(q5f.to_string())
q5f.to_csv(f"{TBL_DIR}/rq1_seniority_best_available_by_source_period.csv", index=False)

# ============================================================
# 6. Compute junior shares and check directional agreement
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Junior share summary across operationalizations")
print("=" * 60)

def compute_junior_share(df, seniority_col, source_filter=None, period_filter=None):
    """Compute entry-level share from a distribution df."""
    mask = pd.Series([True] * len(df))
    if source_filter:
        mask = mask & (df['source'] == source_filter)
    if period_filter:
        mask = mask & (df['period'] == period_filter)
    subset = df[mask]
    total = subset['n'].sum()
    entry = subset[subset[seniority_col] == 'entry']['n'].sum()
    return entry / total if total > 0 else 0, entry, total

# Summary table for junior shares
junior_shares = []

# Using seniority_final
for source in ['kaggle_arshkon', 'kaggle_asaniczka', 'scraped']:
    for period in sorted(q5a['period'].unique()):
        share, entry, total = compute_junior_share(q5a, 'seniority_final', source, period)
        if total > 0:
            junior_shares.append({
                'operationalization': 'seniority_final',
                'source': source, 'period': period,
                'entry_count': entry, 'total': total,
                'entry_share': round(share, 4)
            })

# Using seniority_native
for source in ['kaggle_arshkon', 'kaggle_asaniczka', 'scraped']:
    for period in sorted(q5b['period'].unique()):
        share, entry, total = compute_junior_share(q5b, 'seniority_native', source, period)
        if total > 0:
            junior_shares.append({
                'operationalization': 'seniority_native',
                'source': source, 'period': period,
                'entry_count': entry, 'total': total,
                'entry_share': round(share, 4)
            })

# Using high-confidence
for source in ['kaggle_arshkon', 'kaggle_asaniczka', 'scraped']:
    for period in sorted(q5c['period'].unique()):
        share, entry, total = compute_junior_share(q5c, 'seniority_final', source, period)
        if total > 0:
            junior_shares.append({
                'operationalization': 'seniority_final_highconf',
                'source': source, 'period': period,
                'entry_count': entry, 'total': total,
                'entry_share': round(share, 4)
            })

# Using seniority_llm (labeled only)
for source in q5e['source'].unique():
    for period in sorted(q5e['period'].unique()):
        share, entry, total = compute_junior_share(q5e, 'seniority_llm', source, period)
        if total > 0:
            junior_shares.append({
                'operationalization': 'seniority_llm',
                'source': source, 'period': period,
                'entry_count': entry, 'total': total,
                'entry_share': round(share, 4)
            })

js_df = pd.DataFrame(junior_shares)
print(js_df.to_string())
js_df.to_csv(f"{TBL_DIR}/junior_share_comparison.csv", index=False)

# ============================================================
# 6b. Pooled 2024 vs 2026 junior shares
# ============================================================
print("\n--- Pooled period comparison ---")

# For each operationalization, compute 2024 arshkon-only vs scraped 2026
pooled_results = []

# seniority_final: arshkon only for 2024
for op, df, col in [
    ('seniority_final', q5a, 'seniority_final'),
    ('seniority_native', q5b, 'seniority_native'),
    ('seniority_final_highconf', q5c, 'seniority_final'),
]:
    # 2024 arshkon-only
    share_24, entry_24, total_24 = compute_junior_share(df, col, 'kaggle_arshkon')
    # 2026 scraped
    share_26, entry_26, total_26 = compute_junior_share(df, col, 'scraped')

    if total_24 > 0 and total_26 > 0:
        pooled_results.append({
            'operationalization': op,
            'period_2024_source': 'arshkon_only',
            'entry_share_2024': round(share_24, 4),
            'entry_count_2024': entry_24,
            'total_2024': total_24,
            'entry_share_2026': round(share_26, 4),
            'entry_count_2026': entry_26,
            'total_2026': total_26,
            'direction': 'increase' if share_26 > share_24 else 'decrease' if share_26 < share_24 else 'flat'
        })

# seniority_final pooled 2024 (arshkon + asaniczka) - the dangerous one
share_24_pool, entry_24_pool, total_24_pool = 0, 0, 0
for s in ['kaggle_arshkon', 'kaggle_asaniczka']:
    sh, e, t = compute_junior_share(q5a, 'seniority_final', s)
    entry_24_pool += e
    total_24_pool += t
share_24_pool = entry_24_pool / total_24_pool if total_24_pool > 0 else 0
share_26_f, entry_26_f, total_26_f = compute_junior_share(q5a, 'seniority_final', 'scraped')

pooled_results.append({
    'operationalization': 'seniority_final_POOLED_2024',
    'period_2024_source': 'arshkon+asaniczka',
    'entry_share_2024': round(share_24_pool, 4),
    'entry_count_2024': entry_24_pool,
    'total_2024': total_24_pool,
    'entry_share_2026': round(share_26_f, 4),
    'entry_count_2026': entry_26_f,
    'total_2026': total_26_f,
    'direction': 'increase' if share_26_f > share_24_pool else 'decrease' if share_26_f < share_24_pool else 'flat'
})

# seniority_llm pooled
if len(q5e) > 0:
    for src_2024 in ['kaggle_arshkon', 'kaggle_asaniczka']:
        share_24_l, entry_24_l, total_24_l = compute_junior_share(q5e, 'seniority_llm', src_2024)
        share_26_l, entry_26_l, total_26_l = compute_junior_share(q5e, 'seniority_llm', 'scraped')
        if total_24_l > 0 and total_26_l > 0:
            pooled_results.append({
                'operationalization': f'seniority_llm ({src_2024})',
                'period_2024_source': src_2024,
                'entry_share_2024': round(share_24_l, 4),
                'entry_count_2024': entry_24_l,
                'total_2024': total_24_l,
                'entry_share_2026': round(share_26_l, 4),
                'entry_count_2026': entry_26_l,
                'total_2026': total_26_l,
                'direction': 'increase' if share_26_l > share_24_l else 'decrease' if share_26_l < share_24_l else 'flat'
            })

pr_df = pd.DataFrame(pooled_results)
print(pr_df.to_string())
pr_df.to_csv(f"{TBL_DIR}/junior_share_period_comparison.csv", index=False)

# ============================================================
# 7. Source-specific seniority distributions
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Full seniority distributions by source & operationalization")
print("=" * 60)

# seniority_final distribution by source
q7a = con.execute(f"""
    SELECT source, seniority_final, count(*) as n,
      round(count(*) * 100.0 / sum(count(*)) OVER (PARTITION BY source), 2) as pct
    FROM {PARQUET}
    WHERE {SWE_FILTER}
    GROUP BY source, seniority_final
    ORDER BY source, seniority_final
""").fetchdf()
print("seniority_final distribution by source:")
print(q7a.to_string())
q7a.to_csv(f"{TBL_DIR}/seniority_final_distribution_by_source.csv", index=False)

# seniority_llm distribution by source (labeled only)
q7b = con.execute(f"""
    SELECT source, seniority_llm, count(*) as n,
      round(count(*) * 100.0 / sum(count(*)) OVER (PARTITION BY source), 2) as pct
    FROM {PARQUET}
    WHERE {SWE_FILTER} AND llm_classification_coverage = 'labeled'
    GROUP BY source, seniority_llm
    ORDER BY source, seniority_llm
""").fetchdf()
print("\nseniority_llm distribution by source (labeled only):")
print(q7b.to_string())
q7b.to_csv(f"{TBL_DIR}/seniority_llm_distribution_by_source.csv", index=False)

# ============================================================
# 8. Seniority_final_source breakdown
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: seniority_final_source breakdown by source")
print("=" * 60)

q8 = con.execute(f"""
    SELECT source, seniority_final_source, count(*) as n,
      round(count(*) * 100.0 / sum(count(*)) OVER (PARTITION BY source), 2) as pct
    FROM {PARQUET}
    WHERE {SWE_FILTER}
    GROUP BY source, seniority_final_source
    ORDER BY source, seniority_final_source
""").fetchdf()
print(q8.to_string())
q8.to_csv(f"{TBL_DIR}/seniority_final_source_breakdown.csv", index=False)

# ============================================================
# FIGURES
# ============================================================
print("\n" + "=" * 60)
print("Creating figures...")
print("=" * 60)

# Figure 1: Junior share comparison across operationalizations
fig, ax = plt.subplots(figsize=(10, 6))
# Filter to arshkon 2024 vs scraped 2026 only
plot_data = pr_df[pr_df['period_2024_source'].isin(['arshkon_only', 'arshkon+asaniczka'])].copy()
if len(plot_data) > 0:
    x = np.arange(len(plot_data))
    width = 0.35
    bars1 = ax.bar(x - width/2, plot_data['entry_share_2024'], width, label='2024', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, plot_data['entry_share_2026'], width, label='2026', color='coral', alpha=0.8)
    ax.set_ylabel('Entry-level share')
    ax.set_title('RQ1 Junior Share by Seniority Operationalization')
    ax.set_xticks(x)
    labels = plot_data['operationalization'].values
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim(0, max(plot_data['entry_share_2024'].max(), plot_data['entry_share_2026'].max()) * 1.3)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.002, f'{h:.3f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.002, f'{h:.3f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/junior_share_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# Figure 2: Heatmap of seniority_native vs seniority_final crosstab
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

if len(ct_nf_pivot) > 0:
    sns.heatmap(ct_nf_pivot, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('seniority_native vs seniority_final\n(SWE, where native non-null)')
    axes[0].set_ylabel('seniority_native')
    axes[0].set_xlabel('seniority_final')

if len(ct_ni_pivot) > 0:
    sns.heatmap(ct_ni_pivot, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
    axes[1].set_title('seniority_native vs seniority_imputed\n(SWE, both non-null/unknown)')
    axes[1].set_ylabel('seniority_native')
    axes[1].set_xlabel('seniority_imputed')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/seniority_crosstabs_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

# Figure 3: Seniority distribution by source (seniority_final)
fig, ax = plt.subplots(figsize=(10, 6))
pivot_7a = q7a.pivot_table(index='source', columns='seniority_final', values='pct', fill_value=0)
cols_order = ['entry', 'associate', 'mid-senior', 'director', 'unknown']
cols_present = [c for c in cols_order if c in pivot_7a.columns]
pivot_7a[cols_present].plot(kind='bar', ax=ax, colormap='tab10')
ax.set_ylabel('Percentage')
ax.set_title('Seniority Distribution by Source (seniority_final, SWE)')
ax.set_xlabel('Source')
ax.legend(title='Seniority', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/seniority_distribution_by_source.png", dpi=150, bbox_inches='tight')
plt.close()

# Figure 4: LLM seniority distribution comparison (if data available)
if len(q7b) > 0 and q7b[q7b['seniority_llm'] != 'unknown'].shape[0] > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    q7b_nonunk = q7b[q7b['seniority_llm'] != 'unknown'].copy()
    # Recompute pct excluding unknown
    q7b_totals = q7b_nonunk.groupby('source')['n'].sum().reset_index().rename(columns={'n': 'total'})
    q7b_nonunk = q7b_nonunk.merge(q7b_totals, on='source')
    q7b_nonunk['pct_adj'] = (q7b_nonunk['n'] / q7b_nonunk['total'] * 100).round(2)

    pivot_7b = q7b_nonunk.pivot_table(index='source', columns='seniority_llm', values='pct_adj', fill_value=0)
    cols_present_b = [c for c in cols_order if c in pivot_7b.columns]
    pivot_7b[cols_present_b].plot(kind='bar', ax=ax, colormap='tab10')
    ax.set_ylabel('Percentage (excluding unknown)')
    ax.set_title('LLM Seniority Distribution by Source (labeled rows, excluding unknown)')
    ax.set_xlabel('Source')
    ax.legend(title='Seniority LLM', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/seniority_llm_distribution_by_source.png", dpi=150, bbox_inches='tight')
    plt.close()

print("\nAll T03 outputs saved.")
print(f"  Figures: {FIG_DIR}/")
print(f"  Tables:  {TBL_DIR}/")
