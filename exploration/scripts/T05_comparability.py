#!/usr/bin/env python3
"""T05: Cross-dataset comparability analysis.
Generates figures, tables, and data for the T05 report.
"""
import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

PARQUET = 'data/unified.parquet'
FIG_DIR = 'exploration/figures/T05'
TBL_DIR = 'exploration/tables/T05'

BASE_FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe = true"

con = duckdb.connect()

# ── Helper ──
def source_label(s):
    return {'kaggle_arshkon': 'Arshkon (Apr 2024)',
            'kaggle_asaniczka': 'Asaniczka (Jan 2024)',
            'scraped': 'Scraped (Mar 2026)'}[s]

SOURCES = ['kaggle_arshkon', 'kaggle_asaniczka', 'scraped']
COLORS = {'kaggle_arshkon': '#1f77b4', 'kaggle_asaniczka': '#ff7f0e', 'scraped': '#2ca02c'}

# ============================================================
# 1. DESCRIPTION LENGTH: KS test + overlapping histograms
# ============================================================
print("Step 1: Description length analysis...")

dl_data = {}
cl_data = {}
for src in SOURCES:
    q = f"SELECT description_length, core_length FROM '{PARQUET}' WHERE {BASE_FILTER} AND source = '{src}'"
    df = con.execute(q).fetchdf()
    dl_data[src] = df['description_length'].dropna().values
    cl_data[src] = df['core_length'].dropna().values

# KS tests
ks_results = []
for (a, b) in combinations(SOURCES, 2):
    ks_dl = stats.ks_2samp(dl_data[a], dl_data[b])
    ks_cl = stats.ks_2samp(cl_data[a], cl_data[b])
    ks_results.append({
        'pair': f"{a} vs {b}",
        'ks_stat_description_length': round(ks_dl.statistic, 4),
        'p_description_length': f"{ks_dl.pvalue:.2e}",
        'ks_stat_core_length': round(ks_cl.statistic, 4),
        'p_core_length': f"{ks_cl.pvalue:.2e}"
    })
ks_df = pd.DataFrame(ks_results)
ks_df.to_csv(f'{TBL_DIR}/ks_tests_description_length.csv', index=False)
print(ks_df.to_string(index=False))

# Histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for src in SOURCES:
    axes[0].hist(dl_data[src], bins=80, range=(0, 12000), alpha=0.45,
                 label=source_label(src), color=COLORS[src], density=True)
    axes[1].hist(cl_data[src], bins=80, range=(0, 10000), alpha=0.45,
                 label=source_label(src), color=COLORS[src], density=True)
axes[0].set_title('Description Length Distribution')
axes[0].set_xlabel('Characters')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[1].set_title('Core Length Distribution')
axes[1].set_xlabel('Characters')
axes[1].set_ylabel('Density')
axes[1].legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/description_length_histograms.png', dpi=150)
plt.close()

# Summary stats table
summary_rows = []
for src in SOURCES:
    for col_name, arr in [('description_length', dl_data[src]), ('core_length', cl_data[src])]:
        summary_rows.append({
            'source': src,
            'column': col_name,
            'n': len(arr),
            'mean': round(np.mean(arr), 0),
            'median': round(np.median(arr), 0),
            'std': round(np.std(arr), 0),
            'p25': round(np.percentile(arr, 25), 0),
            'p75': round(np.percentile(arr, 75), 0),
        })
pd.DataFrame(summary_rows).to_csv(f'{TBL_DIR}/description_length_summary.csv', index=False)

# ============================================================
# 2. COMPANY OVERLAP: Jaccard similarity of company_name_canonical
# ============================================================
print("\nStep 2: Company overlap...")

company_sets = {}
for src in SOURCES:
    q = f"SELECT DISTINCT company_name_canonical FROM '{PARQUET}' WHERE {BASE_FILTER} AND source = '{src}' AND company_name_canonical IS NOT NULL"
    company_sets[src] = set(con.execute(q).fetchdf()['company_name_canonical'])

jaccard_results = []
for (a, b) in combinations(SOURCES, 2):
    intersection = company_sets[a] & company_sets[b]
    union = company_sets[a] | company_sets[b]
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0
    jaccard_results.append({
        'pair': f"{a} vs {b}",
        'set_a_size': len(company_sets[a]),
        'set_b_size': len(company_sets[b]),
        'intersection': len(intersection),
        'union': len(union),
        'jaccard': round(jaccard, 4)
    })
    print(f"  {a} vs {b}: |A|={len(company_sets[a])}, |B|={len(company_sets[b])}, |A&B|={len(intersection)}, Jaccard={jaccard:.4f}")
pd.DataFrame(jaccard_results).to_csv(f'{TBL_DIR}/company_jaccard.csv', index=False)

# Top-50 overlap
for (a, b) in combinations(SOURCES, 2):
    q_a = f"""
    SELECT company_name_canonical, COUNT(*) as n FROM '{PARQUET}'
    WHERE {BASE_FILTER} AND source = '{a}' AND company_name_canonical IS NOT NULL
    GROUP BY company_name_canonical ORDER BY n DESC LIMIT 50
    """
    q_b = f"""
    SELECT company_name_canonical, COUNT(*) as n FROM '{PARQUET}'
    WHERE {BASE_FILTER} AND source = '{b}' AND company_name_canonical IS NOT NULL
    GROUP BY company_name_canonical ORDER BY n DESC LIMIT 50
    """
    top_a = set(con.execute(q_a).fetchdf()['company_name_canonical'])
    top_b = set(con.execute(q_b).fetchdf()['company_name_canonical'])
    overlap = top_a & top_b
    print(f"  Top-50 overlap {a} vs {b}: {len(overlap)}/50")

# ============================================================
# 3. GEOGRAPHIC: state-level SWE counts, chi-squared
# ============================================================
print("\nStep 3: Geographic analysis...")

state_data = {}
for src in SOURCES:
    q = f"""
    SELECT state_normalized, COUNT(*) as n FROM '{PARQUET}'
    WHERE {BASE_FILTER} AND source = '{src}' AND state_normalized IS NOT NULL AND state_normalized != ''
    GROUP BY state_normalized ORDER BY n DESC
    """
    state_data[src] = con.execute(q).fetchdf().set_index('state_normalized')['n']

# Merge into a single df for chi-squared
state_df = pd.DataFrame(state_data).fillna(0).astype(int)
state_df.to_csv(f'{TBL_DIR}/state_counts.csv')

# Chi-squared pairwise on top-20 states
top_states = state_df.sum(axis=1).nlargest(20).index
chi2_results = []
for (a, b) in combinations(SOURCES, 2):
    sub = state_df.loc[top_states, [a, b]].copy()
    sub = sub[(sub[a] > 0) | (sub[b] > 0)]
    # Convert to proportions for chi-squared
    chi2, p, dof, expected = stats.chi2_contingency([sub[a].values, sub[b].values])
    chi2_results.append({
        'pair': f"{a} vs {b}",
        'chi2': round(chi2, 2),
        'p_value': f"{p:.2e}",
        'dof': dof,
        'n_states': len(sub)
    })
    print(f"  Chi-squared {a} vs {b}: chi2={chi2:.2f}, p={p:.2e}")
pd.DataFrame(chi2_results).to_csv(f'{TBL_DIR}/geographic_chi2.csv', index=False)

# State share comparison figure
fig, ax = plt.subplots(figsize=(14, 6))
top10 = state_df.sum(axis=1).nlargest(10).index
bar_data = state_df.loc[top10].copy()
for src in SOURCES:
    bar_data[f'{src}_share'] = bar_data[src] / bar_data[src].sum() * 100
x = np.arange(len(top10))
width = 0.25
for i, src in enumerate(SOURCES):
    ax.bar(x + i*width, bar_data[f'{src}_share'], width, label=source_label(src), color=COLORS[src])
ax.set_xticks(x + width)
ax.set_xticklabels(top10, rotation=45, ha='right')
ax.set_ylabel('Share of SWE postings (%)')
ax.set_title('Top-10 State Shares by Source')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/state_shares.png', dpi=150)
plt.close()

# ============================================================
# 4. SENIORITY: distributions (exclude unknown), chi-squared pairwise
# ============================================================
print("\nStep 4: Seniority distributions...")

sen_data = {}
for src in SOURCES:
    q = f"""
    SELECT seniority_final, COUNT(*) as n FROM '{PARQUET}'
    WHERE {BASE_FILTER} AND source = '{src}' AND seniority_final != 'unknown'
    GROUP BY seniority_final ORDER BY seniority_final
    """
    sen_data[src] = con.execute(q).fetchdf().set_index('seniority_final')['n']

sen_df = pd.DataFrame(sen_data).fillna(0).astype(int)
sen_df['total'] = sen_df.sum(axis=1)
for src in SOURCES:
    sen_df[f'{src}_pct'] = (sen_df[src] / sen_df[src].sum() * 100).round(2)
sen_df.to_csv(f'{TBL_DIR}/seniority_distribution.csv')
print(sen_df.to_string())

# Chi-squared pairwise
sen_chi2 = []
for (a, b) in combinations(SOURCES, 2):
    sub = sen_df[[a, b]].copy()
    sub = sub[(sub[a] > 0) | (sub[b] > 0)]
    chi2, p, dof, expected = stats.chi2_contingency([sub[a].values, sub[b].values])
    sen_chi2.append({
        'pair': f"{a} vs {b}",
        'chi2': round(chi2, 2),
        'p_value': f"{p:.2e}",
        'dof': dof
    })
    print(f"  Chi-squared {a} vs {b}: chi2={chi2:.2f}, p={p:.2e}")
pd.DataFrame(sen_chi2).to_csv(f'{TBL_DIR}/seniority_chi2.csv', index=False)

# Seniority bar chart
fig, ax = plt.subplots(figsize=(10, 5))
levels = ['entry', 'associate', 'mid-senior', 'director']
x = np.arange(len(levels))
width = 0.25
for i, src in enumerate(SOURCES):
    vals = [sen_df.loc[l, f'{src}_pct'] if l in sen_df.index else 0 for l in levels]
    ax.bar(x + i*width, vals, width, label=source_label(src), color=COLORS[src])
ax.set_xticks(x + width)
ax.set_xticklabels(levels)
ax.set_ylabel('Share (%)')
ax.set_title('Seniority Distribution (excluding unknown)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/seniority_distribution.png', dpi=150)
plt.close()

# ============================================================
# 5. TITLE VOCABULARY: Jaccard of title_normalized sets
# ============================================================
print("\nStep 5: Title vocabulary...")

title_sets = {}
for src in SOURCES:
    q = f"SELECT DISTINCT title_normalized FROM '{PARQUET}' WHERE {BASE_FILTER} AND source = '{src}' AND title_normalized IS NOT NULL"
    title_sets[src] = set(con.execute(q).fetchdf()['title_normalized'])

title_jaccard = []
for (a, b) in combinations(SOURCES, 2):
    intersection = title_sets[a] & title_sets[b]
    union = title_sets[a] | title_sets[b]
    jaccard = len(intersection) / len(union) if union else 0
    only_a = title_sets[a] - title_sets[b]
    only_b = title_sets[b] - title_sets[a]
    title_jaccard.append({
        'pair': f"{a} vs {b}",
        'set_a_size': len(title_sets[a]),
        'set_b_size': len(title_sets[b]),
        'intersection': len(intersection),
        'union': len(union),
        'jaccard': round(jaccard, 4),
        'unique_to_a': len(only_a),
        'unique_to_b': len(only_b)
    })
    print(f"  {a} vs {b}: Jaccard={jaccard:.4f}, unique_a={len(only_a)}, unique_b={len(only_b)}")
pd.DataFrame(title_jaccard).to_csv(f'{TBL_DIR}/title_jaccard.csv', index=False)

# Titles unique to one period — top examples by posting count
for src in SOURCES:
    others = set()
    for s2 in SOURCES:
        if s2 != src:
            others |= title_sets[s2]
    unique = title_sets[src] - others
    # Get counts for unique titles
    if unique:
        unique_list = "', '".join(t.replace("'", "''") for t in list(unique)[:5000])
        q = f"""
        SELECT title_normalized, COUNT(*) as n FROM '{PARQUET}'
        WHERE {BASE_FILTER} AND source = '{src}' AND title_normalized IN ('{unique_list}')
        GROUP BY title_normalized ORDER BY n DESC LIMIT 25
        """
        try:
            udf = con.execute(q).fetchdf()
            udf.to_csv(f'{TBL_DIR}/titles_unique_to_{src.replace("kaggle_", "")}.csv', index=False)
        except:
            pass

# ============================================================
# 6. INDUSTRY: arshkon vs scraped (asaniczka has no industry)
# ============================================================
print("\nStep 6: Industry analysis (arshkon vs scraped)...")

ind_data = {}
for src in ['kaggle_arshkon', 'scraped']:
    q = f"""
    SELECT company_industry, COUNT(*) as n FROM '{PARQUET}'
    WHERE {BASE_FILTER} AND source = '{src}' AND company_industry IS NOT NULL AND company_industry != ''
    GROUP BY company_industry ORDER BY n DESC
    """
    ind_data[src] = con.execute(q).fetchdf().set_index('company_industry')['n']

ind_df = pd.DataFrame(ind_data).fillna(0).astype(int)
ind_df.columns = ['arshkon', 'scraped']
for c in ['arshkon', 'scraped']:
    ind_df[f'{c}_pct'] = (ind_df[c] / ind_df[c].sum() * 100).round(2)
ind_df = ind_df.sort_values('arshkon', ascending=False)
ind_df.to_csv(f'{TBL_DIR}/industry_distribution.csv')

# Top-15 industries comparison
fig, ax = plt.subplots(figsize=(14, 6))
top15 = ind_df.head(15)
x = np.arange(len(top15))
width = 0.35
ax.bar(x - width/2, top15['arshkon_pct'], width, label='Arshkon (Apr 2024)', color=COLORS['kaggle_arshkon'])
ax.bar(x + width/2, top15['scraped_pct'], width, label='Scraped (Mar 2026)', color=COLORS['scraped'])
ax.set_xticks(x)
ax.set_xticklabels(top15.index, rotation=60, ha='right', fontsize=8)
ax.set_ylabel('Share (%)')
ax.set_title('Industry Distribution: Arshkon vs Scraped')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/industry_distribution.png', dpi=150)
plt.close()

# Chi-squared on top-15 industries
top15_counts = ind_df.head(15)[['arshkon', 'scraped']]
chi2, p, dof, expected = stats.chi2_contingency([top15_counts['arshkon'].values, top15_counts['scraped'].values])
print(f"  Industry chi-squared (top 15): chi2={chi2:.2f}, p={p:.2e}")

# ============================================================
# 7. ARTIFACT DIAGNOSTIC
# ============================================================
print("\nStep 7: Artifact diagnostic...")

# Check if asaniczka's lack of entry-level is a labeling artifact
print("  Asaniczka seniority_native value distribution:")
q = f"""
SELECT seniority_native, COUNT(*) as n FROM '{PARQUET}'
WHERE {BASE_FILTER} AND source = 'kaggle_asaniczka' AND is_swe = true
GROUP BY seniority_native ORDER BY n DESC
"""
for row in con.execute(q).fetchall():
    print(f"    {row[0]:20s}  n={row[1]}")

# Description length by source — test whether scraper captures more text
print("\n  Description length percentiles:")
for src in SOURCES:
    q = f"""
    SELECT
        percentile_cont(0.10) WITHIN GROUP (ORDER BY description_length) as p10,
        percentile_cont(0.50) WITHIN GROUP (ORDER BY description_length) as p50,
        percentile_cont(0.90) WITHIN GROUP (ORDER BY description_length) as p90
    FROM '{PARQUET}' WHERE {BASE_FILTER} AND source = '{src}'
    """
    row = con.execute(q).fetchone()
    print(f"    {src:25s}  p10={row[0]:.0f}  p50={row[1]:.0f}  p90={row[2]:.0f}")

# Check remote rate differences
print("\n  Remote rate by source:")
for src in SOURCES:
    q = f"""
    SELECT
        COUNT(*) FILTER (WHERE is_remote_inferred = true) * 100.0 / COUNT(*) as remote_pct,
        COUNT(*) as n
    FROM '{PARQUET}' WHERE {BASE_FILTER} AND source = '{src}'
    """
    row = con.execute(q).fetchone()
    print(f"    {src:25s}  remote={row[0]:.1f}%  n={row[1]}")

# ============================================================
# 8. WITHIN-2024 CALIBRATION: arshkon vs asaniczka
# ============================================================
print("\nStep 8: Within-2024 calibration (arshkon vs asaniczka)...")

calibration_rows = []

# 8a. Description length KS
ks_cal = stats.ks_2samp(dl_data['kaggle_arshkon'], dl_data['kaggle_asaniczka'])
calibration_rows.append({
    'metric': 'description_length KS',
    'statistic': round(ks_cal.statistic, 4),
    'p_value': f"{ks_cal.pvalue:.2e}",
    'interpretation': 'Baseline cross-source variability'
})
ks_cal_cl = stats.ks_2samp(cl_data['kaggle_arshkon'], cl_data['kaggle_asaniczka'])
calibration_rows.append({
    'metric': 'core_length KS',
    'statistic': round(ks_cal_cl.statistic, 4),
    'p_value': f"{ks_cal_cl.pvalue:.2e}",
    'interpretation': 'Baseline cross-source variability'
})

# 8b. Company overlap
cal_jaccard = len(company_sets['kaggle_arshkon'] & company_sets['kaggle_asaniczka']) / len(company_sets['kaggle_arshkon'] | company_sets['kaggle_asaniczka'])
calibration_rows.append({
    'metric': 'Company Jaccard',
    'statistic': round(cal_jaccard, 4),
    'p_value': 'N/A',
    'interpretation': 'Baseline company overlap'
})

# 8c. Geographic chi-squared
sub_geo = state_df.loc[top_states, ['kaggle_arshkon', 'kaggle_asaniczka']].copy()
sub_geo = sub_geo[(sub_geo['kaggle_arshkon'] > 0) | (sub_geo['kaggle_asaniczka'] > 0)]
chi2_geo, p_geo, _, _ = stats.chi2_contingency([sub_geo['kaggle_arshkon'].values, sub_geo['kaggle_asaniczka'].values])
calibration_rows.append({
    'metric': 'Geographic chi2 (top 20 states)',
    'statistic': round(chi2_geo, 2),
    'p_value': f"{p_geo:.2e}",
    'interpretation': 'Baseline geographic variability'
})

# 8d. Seniority chi-squared
# Arshkon vs asaniczka seniority (exclude unknown)
sen_sub = sen_df[['kaggle_arshkon', 'kaggle_asaniczka']].copy()
sen_sub = sen_sub[(sen_sub['kaggle_arshkon'] > 0) | (sen_sub['kaggle_asaniczka'] > 0)]
chi2_sen, p_sen, _, _ = stats.chi2_contingency([sen_sub['kaggle_arshkon'].values, sen_sub['kaggle_asaniczka'].values])
calibration_rows.append({
    'metric': 'Seniority chi2',
    'statistic': round(chi2_sen, 2),
    'p_value': f"{p_sen:.2e}",
    'interpretation': 'Baseline seniority variability (includes asaniczka label gaps)'
})

# 8e. Title Jaccard
cal_title = len(title_sets['kaggle_arshkon'] & title_sets['kaggle_asaniczka']) / len(title_sets['kaggle_arshkon'] | title_sets['kaggle_asaniczka'])
calibration_rows.append({
    'metric': 'Title Jaccard',
    'statistic': round(cal_title, 4),
    'p_value': 'N/A',
    'interpretation': 'Baseline title overlap'
})

# Now add 2024 vs 2026 comparisons for calibration context
# arshkon vs scraped
ks_2426 = stats.ks_2samp(dl_data['kaggle_arshkon'], dl_data['scraped'])
calibration_rows.append({
    'metric': 'description_length KS (arshkon vs scraped)',
    'statistic': round(ks_2426.statistic, 4),
    'p_value': f"{ks_2426.pvalue:.2e}",
    'interpretation': 'Cross-period comparison'
})
j_2426 = len(company_sets['kaggle_arshkon'] & company_sets['scraped']) / len(company_sets['kaggle_arshkon'] | company_sets['scraped'])
calibration_rows.append({
    'metric': 'Company Jaccard (arshkon vs scraped)',
    'statistic': round(j_2426, 4),
    'p_value': 'N/A',
    'interpretation': 'Cross-period comparison'
})

cal_df = pd.DataFrame(calibration_rows)
cal_df.to_csv(f'{TBL_DIR}/within_2024_calibration.csv', index=False)
print(cal_df.to_string(index=False))

print("\nT05 analysis complete. Outputs in:", FIG_DIR, TBL_DIR)
