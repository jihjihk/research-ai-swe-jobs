#!/usr/bin/env python3
"""T06: Company concentration analysis.
Generates figures, tables, and data for the T06 report.
"""
import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PARQUET = 'data/unified.parquet'
FIG_DIR = 'exploration/figures/T06'
TBL_DIR = 'exploration/tables/T06'

BASE_FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe = true"

con = duckdb.connect()

SOURCES = ['kaggle_arshkon', 'kaggle_asaniczka', 'scraped']
COLORS = {'kaggle_arshkon': '#1f77b4', 'kaggle_asaniczka': '#ff7f0e', 'scraped': '#2ca02c'}

def source_label(s):
    return {'kaggle_arshkon': 'Arshkon (Apr 2024)',
            'kaggle_asaniczka': 'Asaniczka (Jan 2024)',
            'scraped': 'Scraped (Mar 2026)'}[s]

# ============================================================
# 1. Per dataset: HHI, top-1/5/10/20 share, Gini
# ============================================================
print("Step 1: Concentration metrics...")

def compute_concentration(counts):
    """Compute HHI, top-k shares, and Gini from a series of posting counts."""
    total = counts.sum()
    shares = (counts / total).values
    shares_sorted = np.sort(shares)[::-1]

    # HHI
    hhi = (shares**2).sum()

    # Top-k shares
    top1 = shares_sorted[0] * 100 if len(shares_sorted) >= 1 else 0
    top5 = shares_sorted[:5].sum() * 100 if len(shares_sorted) >= 5 else shares_sorted.sum() * 100
    top10 = shares_sorted[:10].sum() * 100 if len(shares_sorted) >= 10 else shares_sorted.sum() * 100
    top20 = shares_sorted[:20].sum() * 100 if len(shares_sorted) >= 20 else shares_sorted.sum() * 100

    # Gini coefficient
    n = len(shares_sorted)
    if n == 0:
        gini = 0
    else:
        sorted_asc = np.sort(counts.values).astype(float)
        cumulative = np.cumsum(sorted_asc)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_asc)) / (n * np.sum(sorted_asc))) - (n + 1) / n

    return {
        'hhi': round(hhi, 6),
        'hhi_times_10000': round(hhi * 10000, 2),
        'top1_pct': round(top1, 2),
        'top5_pct': round(top5, 2),
        'top10_pct': round(top10, 2),
        'top20_pct': round(top20, 2),
        'gini': round(gini, 4),
        'n_companies': n,
        'total_postings': total
    }

concentration_rows = []
company_counts = {}
for src in SOURCES:
    q = f"""
    SELECT company_name_canonical, COUNT(*) as n FROM '{PARQUET}'
    WHERE {BASE_FILTER} AND source = '{src}' AND company_name_canonical IS NOT NULL
    GROUP BY company_name_canonical ORDER BY n DESC
    """
    df = con.execute(q).fetchdf()
    company_counts[src] = df.set_index('company_name_canonical')['n']
    metrics = compute_concentration(company_counts[src])
    metrics['source'] = src
    concentration_rows.append(metrics)
    print(f"  {src}: HHI={metrics['hhi_times_10000']}, Top-5={metrics['top5_pct']}%, "
          f"Top-20={metrics['top20_pct']}%, Gini={metrics['gini']}, n_companies={metrics['n_companies']}")

conc_df = pd.DataFrame(concentration_rows)
conc_df = conc_df[['source', 'n_companies', 'total_postings', 'hhi', 'hhi_times_10000',
                    'top1_pct', 'top5_pct', 'top10_pct', 'top20_pct', 'gini']]
conc_df.to_csv(f'{TBL_DIR}/concentration_metrics.csv', index=False)

# Lorenz curves
fig, ax = plt.subplots(figsize=(8, 6))
for src in SOURCES:
    counts = np.sort(company_counts[src].values).astype(float)
    cumshare = np.cumsum(counts) / counts.sum()
    x = np.arange(1, len(counts)+1) / len(counts)
    ax.plot(x, cumshare, label=source_label(src), color=COLORS[src], linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect equality')
ax.set_xlabel('Cumulative share of companies')
ax.set_ylabel('Cumulative share of postings')
ax.set_title('Lorenz Curves: Company Posting Concentration')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/lorenz_curves.png', dpi=150)
plt.close()

# ============================================================
# 2. Companies with >3% SWE postings in any dataset
# ============================================================
print("\nStep 2: Companies with >3% share...")

dominant_companies = []
for src in SOURCES:
    total = company_counts[src].sum()
    threshold = total * 0.03
    big = company_counts[src][company_counts[src] >= threshold]
    for company, count in big.items():
        dominant_companies.append({
            'source': src,
            'company': company,
            'postings': count,
            'share_pct': round(count / total * 100, 2)
        })

dom_df = pd.DataFrame(dominant_companies).sort_values(['source', 'share_pct'], ascending=[True, False])
dom_df.to_csv(f'{TBL_DIR}/dominant_companies.csv', index=False)
print(dom_df.to_string(index=False))

# ============================================================
# 3. Overlap set: within-company seniority across periods
# ============================================================
print("\nStep 3: Within-company seniority comparison (arshkon vs scraped overlap)...")

overlap_companies = set(company_counts['kaggle_arshkon'].index) & set(company_counts['scraped'].index)
# Filter to companies with at least 5 postings in each
good_overlap = []
for c in overlap_companies:
    if company_counts['kaggle_arshkon'].get(c, 0) >= 5 and company_counts['scraped'].get(c, 0) >= 5:
        good_overlap.append(c)
print(f"  Overlap companies (>=5 postings each): {len(good_overlap)}")

# Get seniority distributions for overlap companies
within_rows = []
for src in ['kaggle_arshkon', 'scraped']:
    if good_overlap:
        companies_str = "', '".join(c.replace("'", "''") for c in good_overlap)
        q = f"""
        SELECT company_name_canonical, seniority_final, COUNT(*) as n FROM '{PARQUET}'
        WHERE {BASE_FILTER} AND source = '{src}'
          AND company_name_canonical IN ('{companies_str}')
          AND seniority_final != 'unknown'
        GROUP BY company_name_canonical, seniority_final
        """
        df = con.execute(q).fetchdf()
        for _, row in df.iterrows():
            within_rows.append({
                'source': src,
                'company': row['company_name_canonical'],
                'seniority': row['seniority_final'],
                'count': row['n']
            })

within_df = pd.DataFrame(within_rows)
if not within_df.empty:
    within_pivot = within_df.pivot_table(
        index=['company', 'seniority'], columns='source', values='count', fill_value=0
    )
    within_pivot.to_csv(f'{TBL_DIR}/within_company_seniority.csv')

    # Aggregate: entry share in overlap companies
    agg = within_df.groupby(['source', 'seniority'])['count'].sum().unstack(fill_value=0)
    for src in ['kaggle_arshkon', 'scraped']:
        if src in agg.index:
            total = agg.loc[src].sum()
            print(f"  {src} overlap companies seniority:")
            for col in agg.columns:
                print(f"    {col}: {agg.loc[src, col]} ({agg.loc[src, col]/total*100:.1f}%)")

# ============================================================
# 4. Company-capped sensitivity: junior share after capping
# ============================================================
print("\nStep 4: Company-capped sensitivity (cap=10 postings/company)...")

cap_results = []
for src in SOURCES:
    q = f"""
    WITH company_postings AS (
        SELECT company_name_canonical, seniority_final,
               ROW_NUMBER() OVER (PARTITION BY company_name_canonical ORDER BY uid) as rn
        FROM '{PARQUET}'
        WHERE {BASE_FILTER} AND source = '{src}'
          AND company_name_canonical IS NOT NULL
          AND seniority_final != 'unknown'
    )
    SELECT seniority_final, COUNT(*) as n FROM company_postings
    WHERE rn <= 10
    GROUP BY seniority_final
    ORDER BY seniority_final
    """
    df = con.execute(q).fetchdf()
    total = df['n'].sum()
    sen_dict = dict(zip(df['seniority_final'], df['n']))

    # Uncapped
    q2 = f"""
    SELECT seniority_final, COUNT(*) as n FROM '{PARQUET}'
    WHERE {BASE_FILTER} AND source = '{src}'
      AND company_name_canonical IS NOT NULL AND seniority_final != 'unknown'
    GROUP BY seniority_final
    """
    df2 = con.execute(q2).fetchdf()
    total2 = df2['n'].sum()
    sen_dict2 = dict(zip(df2['seniority_final'], df2['n']))

    cap_results.append({
        'source': src,
        'method': 'uncapped',
        'entry_pct': round(sen_dict2.get('entry', 0) / total2 * 100, 2) if total2 > 0 else 0,
        'associate_pct': round(sen_dict2.get('associate', 0) / total2 * 100, 2) if total2 > 0 else 0,
        'mid_senior_pct': round(sen_dict2.get('mid-senior', 0) / total2 * 100, 2) if total2 > 0 else 0,
        'director_pct': round(sen_dict2.get('director', 0) / total2 * 100, 2) if total2 > 0 else 0,
        'total': total2
    })
    cap_results.append({
        'source': src,
        'method': 'capped_10',
        'entry_pct': round(sen_dict.get('entry', 0) / total * 100, 2) if total > 0 else 0,
        'associate_pct': round(sen_dict.get('associate', 0) / total * 100, 2) if total > 0 else 0,
        'mid_senior_pct': round(sen_dict.get('mid-senior', 0) / total * 100, 2) if total > 0 else 0,
        'director_pct': round(sen_dict.get('director', 0) / total * 100, 2) if total > 0 else 0,
        'total': total
    })

cap_df = pd.DataFrame(cap_results)
cap_df.to_csv(f'{TBL_DIR}/capped_sensitivity.csv', index=False)
print(cap_df.to_string(index=False))

# ============================================================
# 5. AGGREGATOR ANALYSIS
# ============================================================
print("\nStep 5: Aggregator analysis...")

agg_rows = []
for src in SOURCES:
    for is_agg in [True, False]:
        q = f"""
        SELECT
            COUNT(*) as n,
            AVG(description_length) as mean_dl,
            COUNT(*) FILTER (WHERE seniority_final = 'entry') as n_entry,
            COUNT(*) FILTER (WHERE seniority_final != 'unknown') as n_known_sen
        FROM '{PARQUET}'
        WHERE {BASE_FILTER} AND source = '{src}' AND is_aggregator = {str(is_agg).lower()}
        """
        row = con.execute(q).fetchone()
        entry_pct = (row[2] / row[3] * 100) if row[3] > 0 else 0
        agg_rows.append({
            'source': src,
            'is_aggregator': is_agg,
            'n': row[0],
            'pct_of_source': round(row[0] / (company_counts[src].sum() + (row[0] if is_agg else 0)) * 100, 2),
            'mean_description_length': round(row[1], 0) if row[1] else 0,
            'entry_pct_of_known': round(entry_pct, 2),
            'n_known_seniority': row[3]
        })

agg_df = pd.DataFrame(agg_rows)
agg_df.to_csv(f'{TBL_DIR}/aggregator_profile.csv', index=False)
print(agg_df.to_string(index=False))

# Aggregator fraction by source
print("\n  Aggregator fraction of SWE postings:")
for src in SOURCES:
    q = f"""
    SELECT
        COUNT(*) FILTER (WHERE is_aggregator = true) as n_agg,
        COUNT(*) as total
    FROM '{PARQUET}' WHERE {BASE_FILTER} AND source = '{src}'
    """
    row = con.execute(q).fetchone()
    print(f"    {src:25s}  agg={row[0]:5d}/{row[1]:5d} = {row[0]/row[1]*100:.1f}%")

# Top aggregators
print("\n  Top aggregators by source:")
for src in SOURCES:
    q = f"""
    SELECT company_name, COUNT(*) as n FROM '{PARQUET}'
    WHERE {BASE_FILTER} AND source = '{src}' AND is_aggregator = true
    GROUP BY company_name ORDER BY n DESC LIMIT 5
    """
    print(f"  {src}:")
    for row in con.execute(q).fetchall():
        print(f"    {row[0]:40s}  n={row[1]}")

# ============================================================
# 6. NEW ENTRANTS: Companies in 2026 with no match in 2024
# ============================================================
print("\nStep 6: New entrants in scraped vs arshkon...")

arshkon_companies = set(company_counts['kaggle_arshkon'].index)
scraped_companies = set(company_counts['scraped'].index)
asaniczka_companies = set(company_counts['kaggle_asaniczka'].index)
all_2024 = arshkon_companies | asaniczka_companies

new_vs_arshkon = scraped_companies - arshkon_companies
new_vs_all_2024 = scraped_companies - all_2024

print(f"  Scraped companies: {len(scraped_companies)}")
print(f"  New vs arshkon only: {len(new_vs_arshkon)} ({len(new_vs_arshkon)/len(scraped_companies)*100:.1f}%)")
print(f"  New vs all 2024 (arshkon+asaniczka): {len(new_vs_all_2024)} ({len(new_vs_all_2024)/len(scraped_companies)*100:.1f}%)")

# How many postings do new entrants account for?
if new_vs_arshkon:
    new_companies_str = "', '".join(c.replace("'", "''") for c in list(new_vs_arshkon)[:10000])
    q = f"""
    SELECT COUNT(*) as n FROM '{PARQUET}'
    WHERE {BASE_FILTER} AND source = 'scraped'
      AND company_name_canonical IN ('{new_companies_str}')
    """
    n_new = con.execute(q).fetchone()[0]
    total_scraped = company_counts['scraped'].sum()
    print(f"  New entrant postings (vs arshkon): {n_new} ({n_new/total_scraped*100:.1f}% of scraped)")

if new_vs_all_2024:
    new_companies_str2 = "', '".join(c.replace("'", "''") for c in list(new_vs_all_2024)[:10000])
    q2 = f"""
    SELECT COUNT(*) as n FROM '{PARQUET}'
    WHERE {BASE_FILTER} AND source = 'scraped'
      AND company_name_canonical IN ('{new_companies_str2}')
    """
    n_new2 = con.execute(q2).fetchone()[0]
    print(f"  New entrant postings (vs all 2024): {n_new2} ({n_new2/total_scraped*100:.1f}% of scraped)")

# New entrants - seniority profile
if new_vs_arshkon:
    q = f"""
    SELECT seniority_final, COUNT(*) as n FROM '{PARQUET}'
    WHERE {BASE_FILTER} AND source = 'scraped'
      AND company_name_canonical IN ('{new_companies_str}')
      AND seniority_final != 'unknown'
    GROUP BY seniority_final ORDER BY n DESC
    """
    new_sen = con.execute(q).fetchdf()
    print("\n  New entrant seniority distribution:")
    total_ns = new_sen['n'].sum()
    for _, row in new_sen.iterrows():
        print(f"    {row['seniority_final']:15s}  {row['n']:5d}  ({row['n']/total_ns*100:.1f}%)")

# Save new entrant summary
new_entrant_data = {
    'metric': [
        'scraped_total_companies',
        'new_vs_arshkon',
        'new_vs_all_2024',
        'new_vs_arshkon_pct',
        'new_vs_all_2024_pct',
    ],
    'value': [
        len(scraped_companies),
        len(new_vs_arshkon),
        len(new_vs_all_2024),
        round(len(new_vs_arshkon)/len(scraped_companies)*100, 1),
        round(len(new_vs_all_2024)/len(scraped_companies)*100, 1),
    ]
}
pd.DataFrame(new_entrant_data).to_csv(f'{TBL_DIR}/new_entrants.csv', index=False)

# ============================================================
# Concentration bar chart
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Top-k shares
metrics_to_plot = ['top1_pct', 'top5_pct', 'top10_pct', 'top20_pct']
labels = ['Top 1', 'Top 5', 'Top 10', 'Top 20']
x = np.arange(len(labels))
width = 0.25
for i, src in enumerate(SOURCES):
    row = conc_df[conc_df['source'] == src].iloc[0]
    vals = [row[m] for m in metrics_to_plot]
    axes[0].bar(x + i*width, vals, width, label=source_label(src), color=COLORS[src])
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(labels)
axes[0].set_ylabel('Share of SWE postings (%)')
axes[0].set_title('Company Concentration: Top-k Shares')
axes[0].legend()

# Gini + HHI
metrics2 = ['gini']
for i, src in enumerate(SOURCES):
    row = conc_df[conc_df['source'] == src].iloc[0]
    axes[1].bar(i, row['gini'], color=COLORS[src], label=source_label(src))
axes[1].set_xticks(range(len(SOURCES)))
axes[1].set_xticklabels([source_label(s) for s in SOURCES], rotation=15, ha='right')
axes[1].set_ylabel('Gini coefficient')
axes[1].set_title('Posting Inequality (Gini)')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/concentration_metrics.png', dpi=150)
plt.close()

print(f"\nT06 analysis complete. Outputs in: {FIG_DIR} {TBL_DIR}")
