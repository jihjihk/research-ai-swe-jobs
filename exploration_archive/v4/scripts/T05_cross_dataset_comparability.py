#!/usr/bin/env python3
"""T05: Cross-dataset comparability analysis.

Tests whether dataset differences reflect real labor market changes vs artifacts.
Compares arshkon (2024-04), asaniczka (2024-01), and scraped (2026-03+) on:
  1. Description length (KS tests + histograms)
  2. Company overlap (Jaccard + top-50)
  3. Geographic distributions (chi-squared)
  4. Seniority distributions (chi-squared, ablation framework)
  5. Title vocabulary (Jaccard + unique titles)
  6. Industry comparison (arshkon vs scraped)
  7. Artifact diagnostics
  8. Within-2024 calibration (arshkon vs asaniczka)
  9. Platform labeling stability test
"""

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings('ignore')

PARQUET = 'data/unified.parquet'
FIG_DIR = 'exploration/figures/T05'
TAB_DIR = 'exploration/tables/T05'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

BASE_FILTER = """
    source_platform = 'linkedin'
    AND is_english = true
    AND date_flag = 'ok'
    AND is_swe = true
"""

con = duckdb.connect()

results = {}

# ============================================================
# 1. Description length: KS test + overlapping histograms
# ============================================================
print("=== 1. Description Length Analysis ===")

desc_data = con.execute(f"""
    SELECT source, description_length, core_length
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
      AND description_length IS NOT NULL
""").fetchdf()

sources = ['kaggle_arshkon', 'kaggle_asaniczka', 'scraped']
source_labels = {'kaggle_arshkon': 'Arshkon (2024-04)', 'kaggle_asaniczka': 'Asaniczka (2024-01)', 'scraped': 'Scraped (2026)'}

# KS tests pairwise
print("\nDescription length KS tests (pairwise):")
ks_results = []
for i, s1 in enumerate(sources):
    for s2 in sources[i+1:]:
        d1 = desc_data[desc_data.source == s1]['description_length'].values
        d2 = desc_data[desc_data.source == s2]['description_length'].values
        stat, pval = stats.ks_2samp(d1, d2)
        print(f"  {source_labels[s1]} vs {source_labels[s2]}: KS={stat:.4f}, p={pval:.2e}")
        ks_results.append({'comparison': f'{s1} vs {s2}', 'metric': 'description_length',
                           'ks_stat': stat, 'p_value': pval, 'n1': len(d1), 'n2': len(d2),
                           'median1': np.median(d1), 'median2': np.median(d2)})

print("\nCore length KS tests (pairwise):")
for i, s1 in enumerate(sources):
    for s2 in sources[i+1:]:
        d1 = desc_data[desc_data.source == s1]['core_length'].dropna().values
        d2 = desc_data[desc_data.source == s2]['core_length'].dropna().values
        stat, pval = stats.ks_2samp(d1, d2)
        print(f"  {source_labels[s1]} vs {source_labels[s2]}: KS={stat:.4f}, p={pval:.2e}")
        ks_results.append({'comparison': f'{s1} vs {s2}', 'metric': 'core_length',
                           'ks_stat': stat, 'p_value': pval, 'n1': len(d1), 'n2': len(d2),
                           'median1': np.median(d1), 'median2': np.median(d2)})

pd.DataFrame(ks_results).to_csv(f'{TAB_DIR}/ks_test_results.csv', index=False)
results['ks_tests'] = ks_results

# Summary statistics
print("\nDescription length summary:")
for s in sources:
    d = desc_data[desc_data.source == s]['description_length'].values
    print(f"  {source_labels[s]:30s} n={len(d):>6,d}  median={np.median(d):>7.0f}  mean={np.mean(d):>7.0f}  std={np.std(d):>7.0f}  p25={np.percentile(d,25):>7.0f}  p75={np.percentile(d,75):>7.0f}")

# Histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#2196F3', '#FF9800', '#4CAF50']

for ax, col, title in [(axes[0], 'description_length', 'Full Description Length'),
                        (axes[1], 'core_length', 'Core Length (Rule-based)')]:
    for s, c in zip(sources, colors):
        d = desc_data[desc_data.source == s][col].dropna().values
        ax.hist(d, bins=80, range=(0, 15000), alpha=0.45, label=source_labels[s], color=c, density=True)
    ax.set_xlabel('Characters')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 15000)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/description_length_histograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: description_length_histograms.png")

# ============================================================
# 2. Company overlap: Jaccard + top-50
# ============================================================
print("\n=== 2. Company Overlap Analysis ===")

company_data = con.execute(f"""
    SELECT source, company_name_canonical, COUNT(*) as cnt
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
      AND company_name_canonical IS NOT NULL
    GROUP BY source, company_name_canonical
""").fetchdf()

company_sets = {}
company_top50 = {}
for s in sources:
    sub = company_data[company_data.source == s].sort_values('cnt', ascending=False)
    company_sets[s] = set(sub['company_name_canonical'].values)
    company_top50[s] = set(sub.head(50)['company_name_canonical'].values)

print("\nCompany Jaccard similarity (all companies):")
jaccard_results = []
for i, s1 in enumerate(sources):
    for s2 in sources[i+1:]:
        intersection = len(company_sets[s1] & company_sets[s2])
        union = len(company_sets[s1] | company_sets[s2])
        jaccard = intersection / union if union > 0 else 0
        print(f"  {source_labels[s1]} vs {source_labels[s2]}: Jaccard={jaccard:.4f} ({intersection} shared / {union} total)")
        jaccard_results.append({'comparison': f'{s1} vs {s2}', 'scope': 'all',
                                'jaccard': jaccard, 'intersection': intersection, 'union': union,
                                'set1_size': len(company_sets[s1]), 'set2_size': len(company_sets[s2])})

print("\nCompany Jaccard similarity (top 50):")
for i, s1 in enumerate(sources):
    for s2 in sources[i+1:]:
        intersection = len(company_top50[s1] & company_top50[s2])
        union = len(company_top50[s1] | company_top50[s2])
        jaccard = intersection / union if union > 0 else 0
        print(f"  {source_labels[s1]} vs {source_labels[s2]}: Jaccard={jaccard:.4f} ({intersection} shared / {union} total)")
        jaccard_results.append({'comparison': f'{s1} vs {s2}', 'scope': 'top50',
                                'jaccard': jaccard, 'intersection': intersection, 'union': union,
                                'set1_size': len(company_top50[s1]), 'set2_size': len(company_top50[s2])})

pd.DataFrame(jaccard_results).to_csv(f'{TAB_DIR}/company_jaccard.csv', index=False)
results['company_jaccard'] = jaccard_results

# Top 50 company overlap detail
print("\nTop 50 companies by source:")
for s in sources:
    sub = company_data[company_data.source == s].sort_values('cnt', ascending=False).head(10)
    print(f"\n  {source_labels[s]} top 10:")
    for _, row in sub.iterrows():
        print(f"    {row['company_name_canonical']:40s} {int(row['cnt']):>5d}")

# ============================================================
# 3. Geographic: state-level chi-squared
# ============================================================
print("\n=== 3. Geographic Distribution Analysis ===")

geo_data = con.execute(f"""
    SELECT source, state_normalized, COUNT(*) as cnt
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
      AND state_normalized IS NOT NULL
      AND state_normalized != ''
    GROUP BY source, state_normalized
    ORDER BY source, cnt DESC
""").fetchdf()

# Pivot to source x state matrix
geo_pivot = geo_data.pivot_table(index='state_normalized', columns='source', values='cnt', fill_value=0)

# Top 15 states
top_states = geo_data.groupby('state_normalized')['cnt'].sum().nlargest(15).index.tolist()
geo_top = geo_pivot.loc[geo_pivot.index.isin(top_states)]

# Chi-squared pairwise
print("\nState distribution chi-squared tests (pairwise, top 15 states):")
chi2_geo_results = []
for i, s1 in enumerate(sources):
    for s2 in sources[i+1:]:
        if s1 in geo_top.columns and s2 in geo_top.columns:
            obs = geo_top[[s1, s2]].values
            chi2, p, dof, expected = stats.chi2_contingency(obs)
            cramers_v = np.sqrt(chi2 / (obs.sum() * (min(obs.shape) - 1)))
            print(f"  {source_labels[s1]} vs {source_labels[s2]}: chi2={chi2:.1f}, p={p:.2e}, Cramer's V={cramers_v:.4f}")
            chi2_geo_results.append({'comparison': f'{s1} vs {s2}', 'chi2': chi2, 'p_value': p,
                                     'dof': dof, 'cramers_v': cramers_v})

# State share comparison table
geo_shares = geo_top.div(geo_top.sum(axis=0), axis=1) * 100
geo_shares.columns = [source_labels.get(c, c) for c in geo_shares.columns]
geo_shares = geo_shares.sort_values(geo_shares.columns[0], ascending=False)
geo_shares.to_csv(f'{TAB_DIR}/state_shares_top15.csv')
print("\n  State shares (top 15):")
print(geo_shares.round(1).to_string())

pd.DataFrame(chi2_geo_results).to_csv(f'{TAB_DIR}/chi2_geographic.csv', index=False)

# ============================================================
# 4. Seniority distributions (ablation framework)
# ============================================================
print("\n=== 4. Seniority Distribution Analysis ===")

# 4a. seniority_final (exclude unknown)
sen_data_final = con.execute(f"""
    SELECT source, seniority_final, COUNT(*) as cnt
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
      AND seniority_final != 'unknown'
    GROUP BY source, seniority_final
    ORDER BY source, seniority_final
""").fetchdf()

print("\nseniority_final distribution (excluding unknown):")
sen_pivot_final = sen_data_final.pivot_table(index='seniority_final', columns='source', values='cnt', fill_value=0)
sen_shares_final = sen_pivot_final.div(sen_pivot_final.sum(axis=0), axis=1) * 100
print(sen_shares_final.round(1).to_string())

# Chi-squared pairwise for seniority_final
print("\nSeniority_final chi-squared tests (pairwise):")
chi2_sen_results = []
for i, s1 in enumerate(sources):
    for s2 in sources[i+1:]:
        if s1 in sen_pivot_final.columns and s2 in sen_pivot_final.columns:
            obs = sen_pivot_final[[s1, s2]].values
            chi2, p, dof, expected = stats.chi2_contingency(obs)
            cramers_v = np.sqrt(chi2 / (obs.sum() * (min(obs.shape) - 1)))
            print(f"  {source_labels[s1]} vs {source_labels[s2]}: chi2={chi2:.1f}, p={p:.2e}, Cramer's V={cramers_v:.4f}")
            chi2_sen_results.append({'variable': 'seniority_final', 'comparison': f'{s1} vs {s2}',
                                     'chi2': chi2, 'p_value': p, 'dof': dof, 'cramers_v': cramers_v})

# 4b. seniority_llm (primary - labeled rows only)
sen_data_llm = con.execute(f"""
    SELECT source, seniority_llm, COUNT(*) as cnt
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
      AND llm_classification_coverage = 'labeled'
      AND seniority_llm IS NOT NULL
      AND seniority_llm != 'unknown'
    GROUP BY source, seniority_llm
    ORDER BY source, seniority_llm
""").fetchdf()

print("\nseniority_llm distribution (labeled, excluding unknown):")
sen_pivot_llm = sen_data_llm.pivot_table(index='seniority_llm', columns='source', values='cnt', fill_value=0)
sen_shares_llm = sen_pivot_llm.div(sen_pivot_llm.sum(axis=0), axis=1) * 100
print(sen_shares_llm.round(1).to_string())

# Chi-squared for seniority_llm
print("\nSeniority_llm chi-squared tests (pairwise, labeled rows):")
for i, s1 in enumerate(sources):
    for s2 in sources[i+1:]:
        if s1 in sen_pivot_llm.columns and s2 in sen_pivot_llm.columns:
            obs = sen_pivot_llm[[s1, s2]].values
            if obs.min() > 0:
                chi2, p, dof, expected = stats.chi2_contingency(obs)
                cramers_v = np.sqrt(chi2 / (obs.sum() * (min(obs.shape) - 1)))
                print(f"  {source_labels[s1]} vs {source_labels[s2]}: chi2={chi2:.1f}, p={p:.2e}, Cramer's V={cramers_v:.4f}")
                chi2_sen_results.append({'variable': 'seniority_llm', 'comparison': f'{s1} vs {s2}',
                                         'chi2': chi2, 'p_value': p, 'dof': dof, 'cramers_v': cramers_v})

# 4c. seniority_native (arshkon vs scraped only, since asaniczka has no entry)
sen_data_native = con.execute(f"""
    SELECT source, seniority_native, COUNT(*) as cnt
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
      AND seniority_native IS NOT NULL
      AND seniority_native != ''
    GROUP BY source, seniority_native
    ORDER BY source, seniority_native
""").fetchdf()

print("\nseniority_native distribution:")
sen_pivot_native = sen_data_native.pivot_table(index='seniority_native', columns='source', values='cnt', fill_value=0)
sen_shares_native = sen_pivot_native.div(sen_pivot_native.sum(axis=0), axis=1) * 100
print(sen_shares_native.round(1).to_string())

# 4d. seniority_imputed (where != unknown)
sen_data_imputed = con.execute(f"""
    SELECT source, seniority_imputed, COUNT(*) as cnt
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
      AND seniority_imputed IS NOT NULL
      AND seniority_imputed != 'unknown'
    GROUP BY source, seniority_imputed
    ORDER BY source, seniority_imputed
""").fetchdf()

print("\nseniority_imputed distribution (excluding unknown):")
sen_pivot_imputed = sen_data_imputed.pivot_table(index='seniority_imputed', columns='source', values='cnt', fill_value=0)
sen_shares_imputed = sen_pivot_imputed.div(sen_pivot_imputed.sum(axis=0), axis=1) * 100
print(sen_shares_imputed.round(1).to_string())

# Save all seniority tables
all_sen_shares = []
for var_name, pivot_df in [('seniority_final', sen_shares_final),
                           ('seniority_llm', sen_shares_llm),
                           ('seniority_native', sen_shares_native),
                           ('seniority_imputed', sen_shares_imputed)]:
    df = pivot_df.copy()
    df.columns = [source_labels.get(c, c) for c in df.columns]
    df['variable'] = var_name
    all_sen_shares.append(df)

pd.concat(all_sen_shares).to_csv(f'{TAB_DIR}/seniority_distributions_ablation.csv')
pd.DataFrame(chi2_sen_results).to_csv(f'{TAB_DIR}/chi2_seniority.csv', index=False)

# Entry share summary across all operationalizations
print("\n=== ENTRY-LEVEL SHARE ABLATION SUMMARY ===")
print("(Entry share among known-seniority rows)")
for var_name, pivot_df in [('seniority_llm', sen_pivot_llm),
                           ('seniority_native', sen_pivot_native),
                           ('seniority_final', sen_pivot_final),
                           ('seniority_imputed', sen_pivot_imputed)]:
    print(f"\n  {var_name}:")
    for s in sources:
        if s in pivot_df.columns and 'entry' in pivot_df.index:
            total = pivot_df[s].sum()
            entry = pivot_df.loc['entry', s] if 'entry' in pivot_df.index else 0
            pct = entry / total * 100 if total > 0 else 0
            print(f"    {source_labels[s]:30s} entry={int(entry):>5d}/{int(total):>6d} = {pct:>5.1f}%")

# ============================================================
# 5. Title vocabulary
# ============================================================
print("\n=== 5. Title Vocabulary Analysis ===")

title_data = con.execute(f"""
    SELECT source, title_normalized, COUNT(*) as cnt
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
      AND title_normalized IS NOT NULL
    GROUP BY source, title_normalized
""").fetchdf()

title_sets = {}
title_top50 = {}
for s in sources:
    sub = title_data[title_data.source == s].sort_values('cnt', ascending=False)
    title_sets[s] = set(sub['title_normalized'].values)
    title_top50[s] = set(sub.head(50)['title_normalized'].values)

print("\nTitle Jaccard similarity (all titles):")
title_jaccard = []
for i, s1 in enumerate(sources):
    for s2 in sources[i+1:]:
        intersection = len(title_sets[s1] & title_sets[s2])
        union = len(title_sets[s1] | title_sets[s2])
        jaccard = intersection / union if union > 0 else 0
        print(f"  {source_labels[s1]} vs {source_labels[s2]}: Jaccard={jaccard:.4f} ({intersection}/{union})")
        title_jaccard.append({'comparison': f'{s1} vs {s2}', 'scope': 'all',
                              'jaccard': jaccard, 'intersection': intersection, 'union': union})

print("\nTitle Jaccard similarity (top 50):")
for i, s1 in enumerate(sources):
    for s2 in sources[i+1:]:
        intersection = len(title_top50[s1] & title_top50[s2])
        union = len(title_top50[s1] | title_top50[s2])
        jaccard = intersection / union if union > 0 else 0
        print(f"  {source_labels[s1]} vs {source_labels[s2]}: Jaccard={jaccard:.4f} ({intersection}/{union})")
        title_jaccard.append({'comparison': f'{s1} vs {s2}', 'scope': 'top50',
                              'jaccard': jaccard, 'intersection': intersection, 'union': union})

pd.DataFrame(title_jaccard).to_csv(f'{TAB_DIR}/title_jaccard.csv', index=False)

# Titles unique to one period
print("\nTitles unique to scraped (not in any 2024 source), by frequency:")
titles_2024 = title_sets['kaggle_arshkon'] | title_sets['kaggle_asaniczka']
unique_scraped = title_sets['scraped'] - titles_2024
unique_scraped_df = title_data[(title_data.source == 'scraped') &
                                (title_data.title_normalized.isin(unique_scraped))].sort_values('cnt', ascending=False)
print(f"  {len(unique_scraped)} titles unique to scraped. Top 20:")
for _, row in unique_scraped_df.head(20).iterrows():
    print(f"    {row['title_normalized']:50s} {int(row['cnt']):>5d}")

unique_2024_only = titles_2024 - title_sets['scraped']
print(f"\n  {len(unique_2024_only)} titles in 2024 but not scraped.")

# Save unique titles
unique_scraped_df.to_csv(f'{TAB_DIR}/titles_unique_to_scraped.csv', index=False)

# ============================================================
# 6. Industry comparison (arshkon vs scraped)
# ============================================================
print("\n=== 6. Industry Comparison (arshkon vs scraped) ===")

industry_data = con.execute(f"""
    SELECT source, company_industry, COUNT(*) as cnt
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
      AND company_industry IS NOT NULL
      AND company_industry != ''
      AND source IN ('kaggle_arshkon', 'scraped')
    GROUP BY source, company_industry
    ORDER BY source, cnt DESC
""").fetchdf()

industry_pivot = industry_data.pivot_table(index='company_industry', columns='source', values='cnt', fill_value=0)
industry_shares = industry_pivot.div(industry_pivot.sum(axis=0), axis=1) * 100

# Top 15 industries
top_industries = industry_data.groupby('company_industry')['cnt'].sum().nlargest(15).index.tolist()
ind_top = industry_pivot.loc[industry_pivot.index.isin(top_industries)]

print("\nIndustry shares (top 15):")
ind_shares_top = ind_top.div(ind_top.sum(axis=0), axis=1) * 100
ind_shares_top.columns = [source_labels.get(c, c) for c in ind_shares_top.columns]
print(ind_shares_top.round(1).to_string())

# Chi-squared for industry
if ind_top.shape[0] > 1 and ind_top.shape[1] == 2:
    chi2, p, dof, expected = stats.chi2_contingency(ind_top.values)
    cramers_v = np.sqrt(chi2 / (ind_top.values.sum() * (min(ind_top.shape) - 1)))
    print(f"\n  Industry chi-squared: chi2={chi2:.1f}, p={p:.2e}, Cramer's V={cramers_v:.4f}")

ind_shares_top.to_csv(f'{TAB_DIR}/industry_shares.csv')

# ============================================================
# 7. Artifact diagnostics
# ============================================================
print("\n=== 7. Artifact Diagnostics ===")

# 7a. Description length by source - median difference + effect size
print("\nDescription length Cohen's d (pairwise):")
for i, s1 in enumerate(sources):
    for s2 in sources[i+1:]:
        d1 = desc_data[desc_data.source == s1]['description_length'].values
        d2 = desc_data[desc_data.source == s2]['description_length'].values
        pooled_std = np.sqrt((np.var(d1) + np.var(d2)) / 2)
        cohens_d = (np.mean(d1) - np.mean(d2)) / pooled_std if pooled_std > 0 else 0
        print(f"  {source_labels[s1]} vs {source_labels[s2]}: Cohen's d={cohens_d:.3f}")

# 7b. Check if is_aggregator rates differ
print("\nAggregator rates by source:")
agg_data = con.execute(f"""
    SELECT source,
           COUNT(*) as total,
           SUM(CASE WHEN is_aggregator THEN 1 ELSE 0 END) as agg_count,
           ROUND(100.0 * SUM(CASE WHEN is_aggregator THEN 1 ELSE 0 END) / COUNT(*), 1) as agg_pct
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
    GROUP BY source
""").fetchdf()
print(agg_data.to_string(index=False))

# 7c. Description quality flags
print("\nDescription quality flags by source:")
qual_data = con.execute(f"""
    SELECT source, description_quality_flag, COUNT(*) as cnt
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
    GROUP BY source, description_quality_flag
    ORDER BY source, description_quality_flag
""").fetchdf()
print(qual_data.to_string(index=False))

# 7d. Remote flag distribution (known artifact)
print("\nRemote inferred rates by source:")
remote_data = con.execute(f"""
    SELECT source,
           COUNT(*) as total,
           SUM(CASE WHEN is_remote_inferred THEN 1 ELSE 0 END) as remote_count,
           ROUND(100.0 * SUM(CASE WHEN is_remote_inferred THEN 1 ELSE 0 END) / COUNT(*), 1) as remote_pct
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
    GROUP BY source
""").fetchdf()
print(remote_data.to_string(index=False))

# ============================================================
# 8. Within-2024 calibration
# ============================================================
print("\n=== 8. Within-2024 Calibration (Arshkon vs Asaniczka) ===")
print("This establishes baseline cross-source variability for same-period data.\n")

# Description length
d_ark = desc_data[desc_data.source == 'kaggle_arshkon']['description_length'].values
d_asa = desc_data[desc_data.source == 'kaggle_asaniczka']['description_length'].values
d_scr = desc_data[desc_data.source == 'scraped']['description_length'].values

ks_within24, p_within24 = stats.ks_2samp(d_ark, d_asa)
ks_ark_scr, p_ark_scr = stats.ks_2samp(d_ark, d_scr)
ks_asa_scr, p_asa_scr = stats.ks_2samp(d_asa, d_scr)

print("Description length KS (calibration):")
print(f"  Within-2024 (arshkon vs asaniczka):  KS={ks_within24:.4f}, p={p_within24:.2e}")
print(f"  Across-time (arshkon vs scraped):    KS={ks_ark_scr:.4f}, p={p_ark_scr:.2e}")
print(f"  Across-time (asaniczka vs scraped):  KS={ks_asa_scr:.4f}, p={p_asa_scr:.2e}")
print(f"  Ratio (cross-time / within-2024): arshkon-scraped / within = {ks_ark_scr/ks_within24:.2f}x")

# Company overlap calibration
jac_within24_all = len(company_sets['kaggle_arshkon'] & company_sets['kaggle_asaniczka']) / len(company_sets['kaggle_arshkon'] | company_sets['kaggle_asaniczka'])
jac_ark_scr_all = len(company_sets['kaggle_arshkon'] & company_sets['scraped']) / len(company_sets['kaggle_arshkon'] | company_sets['scraped'])
jac_asa_scr_all = len(company_sets['kaggle_asaniczka'] & company_sets['scraped']) / len(company_sets['kaggle_asaniczka'] | company_sets['scraped'])

print(f"\nCompany Jaccard (calibration):")
print(f"  Within-2024:             {jac_within24_all:.4f}")
print(f"  Arshkon vs scraped:      {jac_ark_scr_all:.4f}")
print(f"  Asaniczka vs scraped:    {jac_asa_scr_all:.4f}")

# Seniority calibration (native: arshkon vs asaniczka)
# But note asaniczka has no entry
print("\nSeniority native calibration (arshkon vs asaniczka):")
print("  NOTE: Asaniczka has no entry-level native labels - comparison is limited to mid-senior/associate/director")
for s in ['kaggle_arshkon', 'kaggle_asaniczka']:
    sub = sen_data_native[sen_data_native.source == s]
    total = sub['cnt'].sum()
    print(f"  {source_labels[s]}:")
    for _, row in sub.iterrows():
        print(f"    {row['seniority_native']:15s} {int(row['cnt']):>6d} ({row['cnt']/total*100:.1f}%)")

# Geographic calibration
if 'kaggle_arshkon' in geo_top.columns and 'kaggle_asaniczka' in geo_top.columns:
    obs_within = geo_top[['kaggle_arshkon', 'kaggle_asaniczka']].values
    chi2_w, p_w, dof_w, _ = stats.chi2_contingency(obs_within)
    v_w = np.sqrt(chi2_w / (obs_within.sum() * (min(obs_within.shape) - 1)))
    print(f"\nGeographic chi-squared (within-2024): chi2={chi2_w:.1f}, p={p_w:.2e}, Cramer's V={v_w:.4f}")

# Calibration summary table
cal_data = []
cal_data.append({'metric': 'desc_length_KS', 'within_2024_arshkon_vs_asaniczka': ks_within24,
                 'arshkon_vs_scraped': ks_ark_scr, 'asaniczka_vs_scraped': ks_asa_scr})
cal_data.append({'metric': 'company_jaccard', 'within_2024_arshkon_vs_asaniczka': jac_within24_all,
                 'arshkon_vs_scraped': jac_ark_scr_all, 'asaniczka_vs_scraped': jac_asa_scr_all})
pd.DataFrame(cal_data).to_csv(f'{TAB_DIR}/within_2024_calibration.csv', index=False)

# ============================================================
# 9. Platform labeling stability test
# ============================================================
print("\n=== 9. Platform Labeling Stability Test ===")

# 9a. Top 20 SWE titles in both arshkon and scraped
shared_titles_data = con.execute(f"""
    WITH title_counts AS (
        SELECT source, title_normalized, COUNT(*) as cnt
        FROM '{PARQUET}'
        WHERE {BASE_FILTER}
        GROUP BY source, title_normalized
    ),
    both_periods AS (
        SELECT a.title_normalized, a.cnt as arshkon_cnt, s.cnt as scraped_cnt
        FROM (SELECT * FROM title_counts WHERE source = 'kaggle_arshkon') a
        JOIN (SELECT * FROM title_counts WHERE source = 'scraped') s
        ON a.title_normalized = s.title_normalized
    )
    SELECT *, arshkon_cnt + scraped_cnt as total
    FROM both_periods
    ORDER BY total DESC
    LIMIT 20
""").fetchdf()

top20_titles = shared_titles_data['title_normalized'].tolist()
print(f"\nTop 20 shared titles (arshkon & scraped):")
for _, row in shared_titles_data.iterrows():
    print(f"  {row['title_normalized']:50s} arshkon={int(row['arshkon_cnt']):>4d}  scraped={int(row['scraped_cnt']):>4d}")

# 9b. Native seniority label per title, arshkon vs scraped
if len(top20_titles) > 0:
    title_list_sql = ", ".join([f"'{t}'" for t in top20_titles])
    title_sen_data = con.execute(f"""
        SELECT source, title_normalized, seniority_native, COUNT(*) as cnt
        FROM '{PARQUET}'
        WHERE {BASE_FILTER}
          AND title_normalized IN ({title_list_sql})
          AND seniority_native IS NOT NULL
          AND source IN ('kaggle_arshkon', 'scraped')
        GROUP BY source, title_normalized, seniority_native
        ORDER BY title_normalized, source, seniority_native
    """).fetchdf()

    print("\nNative seniority labels for top titles (arshkon vs scraped):")
    stability_results = []
    for title in top20_titles[:10]:
        print(f"\n  '{title}':")
        for s in ['kaggle_arshkon', 'scraped']:
            sub = title_sen_data[(title_sen_data.title_normalized == title) & (title_sen_data.source == s)]
            total = sub['cnt'].sum()
            for _, row in sub.iterrows():
                pct = row['cnt'] / total * 100 if total > 0 else 0
                print(f"    {source_labels[s]:30s} {row['seniority_native']:15s} {int(row['cnt']):>4d} ({pct:.1f}%)")
                stability_results.append({'title': title, 'source': s,
                                          'seniority': row['seniority_native'], 'count': int(row['cnt']), 'pct': pct})

    pd.DataFrame(stability_results).to_csv(f'{TAB_DIR}/platform_labeling_stability.csv', index=False)

# 9c. YOE distributions for shared title x seniority cells
print("\nYOE distributions for shared title x seniority cells:")
if len(top20_titles) > 0:
    yoe_data = con.execute(f"""
        SELECT source, title_normalized, seniority_native,
               MEDIAN(yoe_extracted) as median_yoe,
               AVG(yoe_extracted) as mean_yoe,
               COUNT(yoe_extracted) as yoe_n
        FROM '{PARQUET}'
        WHERE {BASE_FILTER}
          AND title_normalized IN ({title_list_sql})
          AND seniority_native IS NOT NULL
          AND yoe_extracted IS NOT NULL
          AND source IN ('kaggle_arshkon', 'scraped')
        GROUP BY source, title_normalized, seniority_native
        HAVING COUNT(yoe_extracted) >= 3
        ORDER BY title_normalized, seniority_native, source
    """).fetchdf()

    print("  (title, seniority) cells with >=3 YOE observations in both periods:")
    for _, row in yoe_data.iterrows():
        print(f"    {row['title_normalized']:45s} {row['seniority_native']:15s} {row['source']:20s} n={int(row['yoe_n']):>3d} median_yoe={row['median_yoe']:.1f} mean_yoe={row['mean_yoe']:.1f}")

    yoe_data.to_csv(f'{TAB_DIR}/yoe_by_title_seniority.csv', index=False)

# 9d. Indeed cross-validation
print("\n\n=== 9d. Indeed Cross-Validation ===")
indeed_data = con.execute(f"""
    SELECT seniority_imputed, COUNT(*) as cnt
    FROM '{PARQUET}'
    WHERE source_platform = 'indeed'
      AND is_english = true
      AND date_flag = 'ok'
      AND is_swe = true
      AND seniority_imputed IS NOT NULL
      AND seniority_imputed != 'unknown'
    GROUP BY seniority_imputed
    ORDER BY seniority_imputed
""").fetchdf()

print("Indeed scraped SWE seniority_imputed distribution:")
total_indeed = indeed_data['cnt'].sum()
for _, row in indeed_data.iterrows():
    print(f"  {row['seniority_imputed']:15s} {int(row['cnt']):>5d} ({row['cnt']/total_indeed*100:.1f}%)")

# Compare to LinkedIn scraped seniority_imputed
linkedin_imputed = con.execute(f"""
    SELECT seniority_imputed, COUNT(*) as cnt
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
      AND source = 'scraped'
      AND seniority_imputed IS NOT NULL
      AND seniority_imputed != 'unknown'
    GROUP BY seniority_imputed
    ORDER BY seniority_imputed
""").fetchdf()

print("\nLinkedIn scraped SWE seniority_imputed distribution:")
total_li = linkedin_imputed['cnt'].sum()
for _, row in linkedin_imputed.iterrows():
    print(f"  {row['seniority_imputed']:15s} {int(row['cnt']):>5d} ({row['cnt']/total_li*100:.1f}%)")

# ============================================================
# Composite figure: seniority ablation
# ============================================================
print("\n=== Creating seniority ablation figure ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
seniority_order = ['entry', 'associate', 'mid-senior', 'director']

for ax, (var_name, pivot_df) in zip(axes.flat,
    [('seniority_llm (LLM, primary)', sen_pivot_llm),
     ('seniority_native (platform)', sen_pivot_native),
     ('seniority_final (rule+native)', sen_pivot_final),
     ('seniority_imputed (rule-only)', sen_pivot_imputed)]):

    shares = pivot_df.div(pivot_df.sum(axis=0), axis=1) * 100
    # Reorder
    ordered = [s for s in seniority_order if s in shares.index]
    shares = shares.loc[ordered]

    x = np.arange(len(ordered))
    width = 0.25
    for i, s in enumerate(sources):
        if s in shares.columns:
            vals = shares[s].values
            ax.bar(x + i*width, vals, width, label=source_labels[s], color=colors[i], alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(ordered, rotation=0)
    ax.set_ylabel('Share (%)')
    ax.set_title(var_name)
    ax.legend(fontsize=8)

plt.suptitle('SWE Seniority Distribution by Source (LinkedIn)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/seniority_ablation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: seniority_ablation.png")

# ============================================================
# Summary figure: geographic comparison
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
top10_states = geo_data.groupby('state_normalized')['cnt'].sum().nlargest(10).index.tolist()
geo_top10 = geo_pivot.loc[geo_pivot.index.isin(top10_states)]
geo_shares_10 = geo_top10.div(geo_top10.sum(axis=0), axis=1) * 100
geo_shares_10 = geo_shares_10.sort_values(geo_shares_10.columns[0], ascending=False)

x = np.arange(len(geo_shares_10))
width = 0.25
for i, s in enumerate(sources):
    if s in geo_shares_10.columns:
        ax.bar(x + i*width, geo_shares_10[s].values, width,
               label=source_labels[s], color=colors[i], alpha=0.8)

ax.set_xticks(x + width)
ax.set_xticklabels(geo_shares_10.index, rotation=45, ha='right')
ax.set_ylabel('Share of SWE postings (%)')
ax.set_title('SWE Postings by State (Top 10)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/geographic_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: geographic_comparison.png")

con.close()
print("\n=== T05 Complete ===")
