#!/usr/bin/env python3
"""T05: Cross-dataset comparability analysis.

Tests whether the three datasets (arshkon 2024-04, asaniczka 2024-01, scraped 2026-03)
are measuring the same thing, or if differences are artifacts of data collection.
"""

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

BASE = Path('/home/jihgaboot/gabor/job-research')
PARQUET = str(BASE / 'preprocessing/intermediate/stage8_final.parquet')
FIG_DIR = BASE / 'exploration/figures/T05'
TAB_DIR = BASE / 'exploration/tables/T05'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")

# Default filters
BASE_FILTER = """
    source_platform = 'linkedin'
    AND is_english = true
    AND date_flag = 'ok'
    AND is_swe = true
"""

results = {}  # store all results for the report

# ============================================================
# 1. DESCRIPTION LENGTH: KS test + overlapping histograms
# ============================================================
print("=== 1. Description Length Analysis ===")

desc_data = con.execute(f"""
    SELECT source, description_length, core_length
    FROM parquet_scan('{PARQUET}')
    WHERE {BASE_FILTER}
""").fetchdf()

# Summary stats
desc_stats = desc_data.groupby('source').agg({
    'description_length': ['count', 'mean', 'median', 'std', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
    'core_length': ['mean', 'median', 'std', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
}).round(1)
desc_stats.columns = [
    'desc_n', 'desc_mean', 'desc_median', 'desc_std', 'desc_q25', 'desc_q75',
    'core_mean', 'core_median', 'core_std', 'core_q25', 'core_q75'
]
print(desc_stats)
desc_stats.to_csv(TAB_DIR / 'description_length_stats.csv')
results['desc_stats'] = desc_stats

# KS tests pairwise
sources = sorted(desc_data['source'].unique())
ks_results = []
for i in range(len(sources)):
    for j in range(i+1, len(sources)):
        s1, s2 = sources[i], sources[j]
        for col in ['description_length', 'core_length']:
            d1 = desc_data[desc_data['source'] == s1][col].dropna()
            d2 = desc_data[desc_data['source'] == s2][col].dropna()
            stat, pval = stats.ks_2samp(d1, d2)
            ks_results.append({
                'source_1': s1, 'source_2': s2,
                'metric': col,
                'ks_statistic': round(stat, 4),
                'p_value': f'{pval:.2e}',
                'n1': len(d1), 'n2': len(d2),
                'median_1': round(d1.median(), 1),
                'median_2': round(d2.median(), 1),
                'median_ratio': round(d2.median() / d1.median(), 3) if d1.median() > 0 else None
            })

ks_df = pd.DataFrame(ks_results)
print("\nKS Test Results:")
print(ks_df.to_string(index=False))
ks_df.to_csv(TAB_DIR / 'ks_tests_description_length.csv', index=False)
results['ks_tests'] = ks_df

# Overlapping histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, title in zip(axes, ['description_length', 'core_length'],
                           ['Full Description Length', 'Core Description Length']):
    for src in sources:
        vals = desc_data[desc_data['source'] == src][col].dropna()
        label = f"{src.replace('kaggle_', '')} (n={len(vals):,}, med={vals.median():.0f})"
        ax.hist(vals, bins=100, alpha=0.4, label=label, density=True,
                range=(0, min(20000, vals.quantile(0.99))))
    ax.set_xlabel('Characters')
    ax.set_ylabel('Density')
    ax.set_title(f'{title} Distribution (SWE, LinkedIn)')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / 'description_length_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved description_length_distributions.png")

# ============================================================
# 2. COMPANY OVERLAP: Jaccard similarity pairwise
# ============================================================
print("\n=== 2. Company Overlap ===")

company_sets = {}
for src in sources:
    r = con.execute(f"""
        SELECT DISTINCT company_name_canonical
        FROM parquet_scan('{PARQUET}')
        WHERE {BASE_FILTER}
          AND source = '{src}'
          AND company_name_canonical IS NOT NULL
          AND company_name_canonical != ''
    """).fetchdf()
    company_sets[src] = set(r['company_name_canonical'])
    print(f"  {src}: {len(company_sets[src]):,} unique companies")

overlap_results = []
for i in range(len(sources)):
    for j in range(i+1, len(sources)):
        s1, s2 = sources[i], sources[j]
        inter = company_sets[s1] & company_sets[s2]
        union = company_sets[s1] | company_sets[s2]
        jaccard = len(inter) / len(union) if len(union) > 0 else 0
        overlap_results.append({
            'source_1': s1, 'source_2': s2,
            'companies_1': len(company_sets[s1]),
            'companies_2': len(company_sets[s2]),
            'intersection': len(inter),
            'union': len(union),
            'jaccard': round(jaccard, 4),
            'overlap_pct_of_smaller': round(len(inter) / min(len(company_sets[s1]), len(company_sets[s2])) * 100, 1)
        })

overlap_df = pd.DataFrame(overlap_results)
print(overlap_df.to_string(index=False))
overlap_df.to_csv(TAB_DIR / 'company_overlap_jaccard.csv', index=False)
results['company_overlap'] = overlap_df

# Top-50 overlap
for i in range(len(sources)):
    for j in range(i+1, len(sources)):
        s1, s2 = sources[i], sources[j]
        # Get top-50 by posting count in each source
        for src in [s1, s2]:
            r = con.execute(f"""
                SELECT company_name_canonical, COUNT(*) as n
                FROM parquet_scan('{PARQUET}')
                WHERE {BASE_FILTER}
                  AND source = '{src}'
                  AND company_name_canonical IS NOT NULL
                GROUP BY company_name_canonical
                ORDER BY n DESC
                LIMIT 50
            """).fetchdf()
            company_sets[f'top50_{src}'] = set(r['company_name_canonical'])

top50_results = []
for i in range(len(sources)):
    for j in range(i+1, len(sources)):
        s1, s2 = sources[i], sources[j]
        t1 = company_sets[f'top50_{s1}']
        t2 = company_sets[f'top50_{s2}']
        inter = t1 & t2
        top50_results.append({
            'source_1': s1, 'source_2': s2,
            'top50_overlap': len(inter),
            'overlap_names': ', '.join(sorted(list(inter))[:20]) + ('...' if len(inter) > 20 else '')
        })

print("\nTop-50 company overlap:")
for r in top50_results:
    print(f"  {r['source_1']} vs {r['source_2']}: {r['top50_overlap']}/50 overlap")
    print(f"    Examples: {r['overlap_names'][:200]}")

top50_df = pd.DataFrame(top50_results)
top50_df.to_csv(TAB_DIR / 'company_overlap_top50.csv', index=False)
results['top50_overlap'] = top50_results

# ============================================================
# 3. GEOGRAPHIC: state-level SWE counts, chi-squared
# ============================================================
print("\n=== 3. Geographic Distribution ===")

geo_data = con.execute(f"""
    SELECT source, state_normalized, COUNT(*) as n
    FROM parquet_scan('{PARQUET}')
    WHERE {BASE_FILTER}
      AND state_normalized IS NOT NULL
      AND state_normalized != ''
    GROUP BY source, state_normalized
""").fetchdf()

# Pivot to get state x source matrix
geo_pivot = geo_data.pivot_table(index='state_normalized', columns='source', values='n', fill_value=0)
geo_pivot = geo_pivot.loc[geo_pivot.sum(axis=1) >= 10]  # states with at least 10 postings total

# State shares
geo_shares = geo_pivot.div(geo_pivot.sum(axis=0), axis=1) * 100
geo_shares = geo_shares.round(2)

# Top-15 states by total
top_states = geo_pivot.sum(axis=1).nlargest(15).index
geo_top = geo_pivot.loc[top_states]
geo_shares_top = geo_shares.loc[top_states]

print("Top-15 states (SWE postings):")
print(geo_top)
print("\nShares (%):")
print(geo_shares_top)

geo_top.to_csv(TAB_DIR / 'geographic_top15_counts.csv')
geo_shares_top.to_csv(TAB_DIR / 'geographic_top15_shares.csv')

# Chi-squared tests pairwise
geo_chi2 = []
for i in range(len(sources)):
    for j in range(i+1, len(sources)):
        s1, s2 = sources[i], sources[j]
        if s1 in geo_pivot.columns and s2 in geo_pivot.columns:
            # Use states present in both
            both = geo_pivot[[s1, s2]].copy()
            both = both[(both[s1] > 0) | (both[s2] > 0)]
            chi2, pval, dof, expected = stats.chi2_contingency(both.T)
            # Cramér's V
            n_total = both.sum().sum()
            k = min(both.shape) - 1
            cramers_v = np.sqrt(chi2 / (n_total * k)) if k > 0 else 0
            geo_chi2.append({
                'source_1': s1, 'source_2': s2,
                'chi2': round(chi2, 1),
                'p_value': f'{pval:.2e}',
                'dof': dof,
                'cramers_v': round(cramers_v, 4),
                'n_states': len(both)
            })

geo_chi2_df = pd.DataFrame(geo_chi2)
print("\nChi-squared tests (geographic distribution):")
print(geo_chi2_df.to_string(index=False))
geo_chi2_df.to_csv(TAB_DIR / 'geographic_chi2.csv', index=False)
results['geo_chi2'] = geo_chi2_df

# ============================================================
# 4. SENIORITY: distributions (exclude unknown), chi-squared
# ============================================================
print("\n=== 4. Seniority Distribution ===")

sen_data = con.execute(f"""
    SELECT source, seniority_final, seniority_3level, COUNT(*) as n
    FROM parquet_scan('{PARQUET}')
    WHERE {BASE_FILTER}
      AND seniority_final != 'unknown'
    GROUP BY source, seniority_final, seniority_3level
    ORDER BY source, seniority_final
""").fetchdf()

# Pivot
sen_pivot = sen_data.pivot_table(index='seniority_final', columns='source', values='n', fill_value=0)
sen_shares = sen_pivot.div(sen_pivot.sum(axis=0), axis=1) * 100

print("Seniority counts (excl. unknown):")
print(sen_pivot)
print("\nSeniority shares (%):")
print(sen_shares.round(2))

sen_pivot.to_csv(TAB_DIR / 'seniority_counts.csv')
sen_shares.round(2).to_csv(TAB_DIR / 'seniority_shares.csv')

# Also show unknown rates
unknown_data = con.execute(f"""
    SELECT source,
           COUNT(*) as total,
           SUM(CASE WHEN seniority_final = 'unknown' THEN 1 ELSE 0 END) as unknown_n,
           ROUND(100.0 * SUM(CASE WHEN seniority_final = 'unknown' THEN 1 ELSE 0 END) / COUNT(*), 1) as unknown_pct
    FROM parquet_scan('{PARQUET}')
    WHERE {BASE_FILTER}
    GROUP BY source
""").fetchdf()
print("\nUnknown seniority rates:")
print(unknown_data.to_string(index=False))
results['seniority_unknown'] = unknown_data

# Chi-squared pairwise on seniority
sen_chi2 = []
for i in range(len(sources)):
    for j in range(i+1, len(sources)):
        s1, s2 = sources[i], sources[j]
        if s1 in sen_pivot.columns and s2 in sen_pivot.columns:
            table = sen_pivot[[s1, s2]]
            table = table[(table[s1] > 0) | (table[s2] > 0)]
            chi2, pval, dof, expected = stats.chi2_contingency(table.T)
            n_total = table.sum().sum()
            k = min(table.shape) - 1
            cramers_v = np.sqrt(chi2 / (n_total * k)) if k > 0 else 0
            sen_chi2.append({
                'source_1': s1, 'source_2': s2,
                'chi2': round(chi2, 1),
                'p_value': f'{pval:.2e}',
                'dof': dof,
                'cramers_v': round(cramers_v, 4)
            })

sen_chi2_df = pd.DataFrame(sen_chi2)
print("\nChi-squared tests (seniority):")
print(sen_chi2_df.to_string(index=False))
sen_chi2_df.to_csv(TAB_DIR / 'seniority_chi2.csv', index=False)
results['sen_chi2'] = sen_chi2_df

# 3-level seniority view
sen3_data = con.execute(f"""
    SELECT source, seniority_3level, COUNT(*) as n
    FROM parquet_scan('{PARQUET}')
    WHERE {BASE_FILTER}
      AND seniority_3level != 'unknown'
    GROUP BY source, seniority_3level
    ORDER BY source, seniority_3level
""").fetchdf()

sen3_pivot = sen3_data.pivot_table(index='seniority_3level', columns='source', values='n', fill_value=0)
sen3_shares = sen3_pivot.div(sen3_pivot.sum(axis=0), axis=1) * 100
print("\n3-level seniority shares (%):")
print(sen3_shares.round(2))
sen3_shares.round(2).to_csv(TAB_DIR / 'seniority_3level_shares.csv')
results['sen3_shares'] = sen3_shares

# ============================================================
# 5. TITLE VOCABULARY: Jaccard of title_normalized sets
# ============================================================
print("\n=== 5. Title Vocabulary ===")

title_sets = {}
for src in sources:
    r = con.execute(f"""
        SELECT DISTINCT title_normalized
        FROM parquet_scan('{PARQUET}')
        WHERE {BASE_FILTER}
          AND source = '{src}'
          AND title_normalized IS NOT NULL
    """).fetchdf()
    title_sets[src] = set(r['title_normalized'])
    print(f"  {src}: {len(title_sets[src]):,} unique titles")

title_overlap = []
for i in range(len(sources)):
    for j in range(i+1, len(sources)):
        s1, s2 = sources[i], sources[j]
        inter = title_sets[s1] & title_sets[s2]
        union = title_sets[s1] | title_sets[s2]
        jaccard = len(inter) / len(union) if len(union) > 0 else 0
        # Titles unique to each
        only_s1 = title_sets[s1] - title_sets[s2]
        only_s2 = title_sets[s2] - title_sets[s1]
        title_overlap.append({
            'source_1': s1, 'source_2': s2,
            'titles_1': len(title_sets[s1]),
            'titles_2': len(title_sets[s2]),
            'intersection': len(inter),
            'union': len(union),
            'jaccard': round(jaccard, 4),
            'unique_to_s1': len(only_s1),
            'unique_to_s2': len(only_s2)
        })

title_overlap_df = pd.DataFrame(title_overlap)
print(title_overlap_df.to_string(index=False))
title_overlap_df.to_csv(TAB_DIR / 'title_vocabulary_jaccard.csv', index=False)

# Titles unique to 2026 (scraped) vs any 2024 source
titles_2024 = title_sets.get('kaggle_arshkon', set()) | title_sets.get('kaggle_asaniczka', set())
titles_2026 = title_sets.get('scraped', set())
only_2026 = titles_2026 - titles_2024
only_2024 = titles_2024 - titles_2026
print(f"\nTitles unique to 2026: {len(only_2026)}")
print(f"Titles unique to 2024: {len(only_2024)}")
print(f"Shared: {len(titles_2024 & titles_2026)}")

# Get counts for unique-to-2026 titles
if only_2026:
    # Most common unique-to-2026 titles
    unique_2026_list = "','".join([t.replace("'", "''") for t in list(only_2026)[:500]])
    r = con.execute(f"""
        SELECT title_normalized, COUNT(*) as n
        FROM parquet_scan('{PARQUET}')
        WHERE {BASE_FILTER}
          AND source = 'scraped'
          AND title_normalized IN ('{unique_2026_list}')
        GROUP BY title_normalized
        ORDER BY n DESC
        LIMIT 30
    """).fetchdf()
    print("\nTop unique-to-2026 titles:")
    print(r.to_string(index=False))
    r.to_csv(TAB_DIR / 'titles_unique_to_2026.csv', index=False)

# ============================================================
# 6. INDUSTRY: arshkon vs scraped
# ============================================================
print("\n=== 6. Industry Distribution ===")

industry_data = con.execute(f"""
    SELECT source, company_industry, COUNT(*) as n
    FROM parquet_scan('{PARQUET}')
    WHERE {BASE_FILTER}
      AND source IN ('kaggle_arshkon', 'scraped')
      AND company_industry IS NOT NULL
      AND company_industry != ''
    GROUP BY source, company_industry
    ORDER BY source, n DESC
""").fetchdf()

ind_pivot = industry_data.pivot_table(index='company_industry', columns='source', values='n', fill_value=0)
ind_pivot['total'] = ind_pivot.sum(axis=1)
ind_pivot = ind_pivot.sort_values('total', ascending=False)

# Top-20 industries
ind_top20 = ind_pivot.head(20).copy()
for col in ['kaggle_arshkon', 'scraped']:
    if col in ind_top20.columns:
        col_total = ind_top20[col].sum()
        ind_top20[f'{col}_pct'] = (ind_top20[col] / col_total * 100).round(1)

print("Top-20 industries (arshkon vs scraped):")
print(ind_top20.to_string())
ind_top20.to_csv(TAB_DIR / 'industry_top20.csv')
results['industry'] = ind_top20

# Chi-squared on industry
if 'kaggle_arshkon' in ind_pivot.columns and 'scraped' in ind_pivot.columns:
    ind_table = ind_pivot[['kaggle_arshkon', 'scraped']]
    ind_table = ind_table[(ind_table['kaggle_arshkon'] >= 5) | (ind_table['scraped'] >= 5)]
    chi2, pval, dof, expected = stats.chi2_contingency(ind_table.T)
    n_total = ind_table.sum().sum()
    k = min(ind_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n_total * k)) if k > 0 else 0
    print(f"\nIndustry chi2: {chi2:.1f}, p={pval:.2e}, V={cramers_v:.4f}, dof={dof}")
    results['industry_chi2'] = {'chi2': chi2, 'p': pval, 'cramers_v': cramers_v, 'dof': dof}

# ============================================================
# 7. ARTIFACT DIAGNOSTIC
# ============================================================
print("\n=== 7. Artifact Diagnostic ===")

# Core-to-description ratio by source (boilerplate sensitivity)
ratio_data = con.execute(f"""
    SELECT source,
           AVG(core_length / NULLIF(description_length, 0)) as mean_core_ratio,
           MEDIAN(core_length / NULLIF(description_length, 0)) as median_core_ratio,
           STDDEV(core_length / NULLIF(description_length, 0)) as std_core_ratio
    FROM parquet_scan('{PARQUET}')
    WHERE {BASE_FILTER}
      AND core_length IS NOT NULL
      AND description_length > 0
    GROUP BY source
""").fetchdf()
print("Core/description length ratio by source:")
print(ratio_data.to_string(index=False))
ratio_data.to_csv(TAB_DIR / 'core_to_desc_ratio.csv', index=False)
results['core_ratio'] = ratio_data

# Description quality flags by source
quality_data = con.execute(f"""
    SELECT source, description_quality_flag, COUNT(*) as n
    FROM parquet_scan('{PARQUET}')
    WHERE {BASE_FILTER}
    GROUP BY source, description_quality_flag
    ORDER BY source, description_quality_flag
""").fetchdf()
print("\nDescription quality flags:")
print(quality_data.to_string(index=False))

# SWE classification tier by source
tier_data = con.execute(f"""
    SELECT source, swe_classification_tier, COUNT(*) as n
    FROM parquet_scan('{PARQUET}')
    WHERE {BASE_FILTER}
    GROUP BY source, swe_classification_tier
    ORDER BY source, swe_classification_tier
""").fetchdf()
print("\nSWE classification tier by source:")
print(tier_data.to_string(index=False))
tier_data.to_csv(TAB_DIR / 'swe_classification_tier.csv', index=False)
results['swe_tier'] = tier_data

# Seniority resolution source by dataset
sen_source_data = con.execute(f"""
    SELECT source, seniority_final_source, COUNT(*) as n
    FROM parquet_scan('{PARQUET}')
    WHERE {BASE_FILTER}
      AND seniority_final != 'unknown'
    GROUP BY source, seniority_final_source
    ORDER BY source, n DESC
""").fetchdf()
print("\nSeniority label provenance by source:")
print(sen_source_data.to_string(index=False))
sen_source_data.to_csv(TAB_DIR / 'seniority_label_provenance.csv', index=False)

# ============================================================
# 8. WITHIN-2024 CALIBRATION
# ============================================================
print("\n=== 8. Within-2024 Calibration ===")
print("Comparing arshkon (2024-04) vs asaniczka (2024-01) as baseline variability")

# Description length
for col in ['description_length', 'core_length']:
    d_arsh = desc_data[desc_data['source'] == 'kaggle_arshkon'][col].dropna()
    d_asan = desc_data[desc_data['source'] == 'kaggle_asaniczka'][col].dropna()
    stat, pval = stats.ks_2samp(d_arsh, d_asan)
    print(f"  {col}: KS={stat:.4f}, p={pval:.2e}")
    print(f"    arshkon median={d_arsh.median():.0f}, asaniczka median={d_asan.median():.0f}")

# Seniority within-2024
print("\n  Seniority distribution within-2024:")
for s in ['kaggle_arshkon', 'kaggle_asaniczka']:
    if s in sen_pivot.columns:
        shares = (sen_pivot[s] / sen_pivot[s].sum() * 100).round(1)
        print(f"    {s}: {shares.to_dict()}")

# Company overlap within-2024
s1, s2 = 'kaggle_arshkon', 'kaggle_asaniczka'
inter = company_sets[s1] & company_sets[s2]
print(f"\n  Company overlap (arshkon vs asaniczka): {len(inter)} / min({len(company_sets[s1])}, {len(company_sets[s2])}) = {len(inter)/min(len(company_sets[s1]), len(company_sets[s2]))*100:.1f}%")
print(f"  Jaccard: {len(inter) / len(company_sets[s1] | company_sets[s2]):.4f}")

# Geographic within-2024
if 'kaggle_arshkon' in geo_pivot.columns and 'kaggle_asaniczka' in geo_pivot.columns:
    cal_table = geo_pivot[['kaggle_arshkon', 'kaggle_asaniczka']]
    cal_table = cal_table[(cal_table['kaggle_arshkon'] > 0) | (cal_table['kaggle_asaniczka'] > 0)]
    chi2, pval, dof, exp = stats.chi2_contingency(cal_table.T)
    n_total = cal_table.sum().sum()
    k = min(cal_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n_total * k)) if k > 0 else 0
    print(f"  Geographic chi2 (within-2024): chi2={chi2:.1f}, p={pval:.2e}, V={cramers_v:.4f}")

# Title vocabulary within-2024
inter_titles = title_sets['kaggle_arshkon'] & title_sets['kaggle_asaniczka']
union_titles = title_sets['kaggle_arshkon'] | title_sets['kaggle_asaniczka']
print(f"  Title Jaccard (within-2024): {len(inter_titles)/len(union_titles):.4f}")

print("\n=== Analysis complete ===")

# ============================================================
# FIGURE 2: Seniority distribution bar chart
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 5-level
sen_shares_plot = sen_shares.reindex(['entry', 'associate', 'mid-senior', 'director'])
sen_shares_plot.plot(kind='bar', ax=axes[0], rot=0)
axes[0].set_title('Seniority Distribution (5-level, excl. unknown)')
axes[0].set_ylabel('Share (%)')
axes[0].legend(title='Source', fontsize=8)

# 3-level
sen3_shares_plot = results['sen3_shares'].reindex(['junior', 'mid', 'senior'])
sen3_shares_plot.plot(kind='bar', ax=axes[1], rot=0)
axes[1].set_title('Seniority Distribution (3-level, excl. unknown)')
axes[1].set_ylabel('Share (%)')
axes[1].legend(title='Source', fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / 'seniority_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved seniority_distributions.png")

# ============================================================
# FIGURE 3: Geographic heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))
geo_shares_top_sorted = geo_shares.loc[top_states].sort_values(
    geo_shares.columns[0], ascending=False)

# Rename columns for display
rename = {c: c.replace('kaggle_', '') for c in geo_shares_top_sorted.columns}
sns.heatmap(geo_shares_top_sorted.rename(columns=rename),
            annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
ax.set_title('SWE Posting Share by State (%, top 15 states)')
ax.set_ylabel('State')

plt.tight_layout()
plt.savefig(FIG_DIR / 'geographic_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved geographic_heatmap.png")

# ============================================================
# CALIBRATION TABLE
# ============================================================
cal_rows = []
# For each metric, compute within-2024 vs cross-period
metrics = [
    ('Description length (KS)', 'description_length'),
    ('Core length (KS)', 'core_length'),
]

for label, col in metrics:
    d_arsh = desc_data[desc_data['source'] == 'kaggle_arshkon'][col].dropna()
    d_asan = desc_data[desc_data['source'] == 'kaggle_asaniczka'][col].dropna()
    d_scra = desc_data[desc_data['source'] == 'scraped'][col].dropna()

    ks_24, p_24 = stats.ks_2samp(d_arsh, d_asan)
    ks_cross_a, p_cross_a = stats.ks_2samp(d_arsh, d_scra)
    ks_cross_b, p_cross_b = stats.ks_2samp(d_asan, d_scra)

    cal_rows.append({
        'metric': label,
        'within_2024_KS': round(ks_24, 4),
        'arshkon_vs_scraped_KS': round(ks_cross_a, 4),
        'asaniczka_vs_scraped_KS': round(ks_cross_b, 4),
        'amplification_factor': round(max(ks_cross_a, ks_cross_b) / max(ks_24, 0.001), 2)
    })

cal_df = pd.DataFrame(cal_rows)
# Add seniority and geographic comparisons
# Seniority Cramér's V
sen_cramers = {}
for row in results['sen_chi2'].to_dict('records'):
    key = f"{row['source_1']}_vs_{row['source_2']}"
    sen_cramers[key] = row['cramers_v']

geo_cramers = {}
for row in results['geo_chi2'].to_dict('records'):
    key = f"{row['source_1']}_vs_{row['source_2']}"
    geo_cramers[key] = row['cramers_v']

cal_rows.append({
    'metric': 'Seniority (Cramér V)',
    'within_2024_KS': sen_cramers.get('kaggle_arshkon_vs_kaggle_asaniczka', 'N/A'),
    'arshkon_vs_scraped_KS': sen_cramers.get('kaggle_arshkon_vs_scraped', 'N/A'),
    'asaniczka_vs_scraped_KS': sen_cramers.get('kaggle_asaniczka_vs_scraped', 'N/A'),
    'amplification_factor': ''
})

cal_rows.append({
    'metric': 'Geography (Cramér V)',
    'within_2024_KS': geo_cramers.get('kaggle_arshkon_vs_kaggle_asaniczka', 'N/A'),
    'arshkon_vs_scraped_KS': geo_cramers.get('kaggle_arshkon_vs_scraped', 'N/A'),
    'asaniczka_vs_scraped_KS': geo_cramers.get('kaggle_asaniczka_vs_scraped', 'N/A'),
    'amplification_factor': ''
})

cal_df = pd.DataFrame(cal_rows)
print("\nCalibration table:")
print(cal_df.to_string(index=False))
cal_df.to_csv(TAB_DIR / 'calibration_table.csv', index=False)

print("\n=== T05 COMPLETE ===")
