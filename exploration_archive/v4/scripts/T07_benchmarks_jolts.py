#!/usr/bin/env python3
"""
T07 Part B: External benchmarks.
- JOLTS Information sector job openings (from BLS API)
- BLS OES state-level employment comparison (using published May 2023 data)
- Industry distribution comparison
- Sample representativeness framing
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy import stats
import duckdb

OUT_TABLES = "exploration/tables/T07"
OUT_FIGURES = "exploration/figures/T07"
os.makedirs(OUT_TABLES, exist_ok=True)
os.makedirs(OUT_FIGURES, exist_ok=True)

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")

BASE_FILTER = """
    source_platform = 'linkedin'
    AND is_english = true
    AND date_flag = 'ok'
    AND is_swe = true
"""

# =============================================================================
# 1. JOLTS Information Sector Data
# =============================================================================

# Seasonally adjusted (JTS) JOLTS Information sector job openings
# Values in thousands, from BLS API (JTS510000000000000JOL)
jolts_sa = {
    '2020-01': 130, '2020-02': 138, '2020-03': 121, '2020-04': 118,
    '2020-05': 93, '2020-06': 106, '2020-07': 79, '2020-08': 86,
    '2020-09': 110, '2020-10': 126, '2020-11': 109, '2020-12': 123,
    '2021-01': 170, '2021-02': 109, '2021-03': 107, '2021-04': 128,
    '2021-05': 151, '2021-06': 165, '2021-07': 216, '2021-08': 216,
    '2021-09': 201, '2021-10': 208, '2021-11': 216, '2021-12': 249,
    '2022-01': 238, '2022-02': 242, '2022-03': 262, '2022-04': 274,
    '2022-05': 249, '2022-06': 254, '2022-07': 231, '2022-08': 207,
    '2022-09': 216, '2022-10': 215, '2022-11': 223, '2022-12': 80,
    '2023-01': 101, '2023-02': 144, '2023-03': 145, '2023-04': 146,
    '2023-05': 167, '2023-06': 156, '2023-07': 148, '2023-08': 155,
    '2023-09': 97, '2023-10': 101, '2023-11': 150, '2023-12': 141,
    '2024-01': 160, '2024-02': 129, '2024-03': 144, '2024-04': 87,
    '2024-05': 120, '2024-06': 108, '2024-07': 105, '2024-08': 105,
    '2024-09': 114, '2024-10': 178, '2024-11': 119, '2024-12': 105,
    '2025-01': 101, '2025-02': 125, '2025-03': 115, '2025-04': 134,
    '2025-05': 109, '2025-06': 123, '2025-07': 157, '2025-08': 141,
    '2025-09': 119, '2025-10': 117, '2025-11': 88, '2025-12': 112,
}

jolts_df = pd.DataFrame([
    {'date': datetime.strptime(k, '%Y-%m'), 'openings_thousands': v}
    for k, v in jolts_sa.items()
]).sort_values('date')

jolts_df.to_csv(f"{OUT_TABLES}/jolts_information_sa.csv", index=False)

# --- Plot JOLTS with our data periods annotated ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(jolts_df['date'], jolts_df['openings_thousands'], 'b-', linewidth=1.5, label='JOLTS Info Sector (SA, 000s)')

# Annotate our data periods
data_periods = {
    'Asaniczka\n(Jan 2024)': datetime(2024, 1, 15),
    'Arshkon\n(Apr 2024)': datetime(2024, 4, 15),
    'Scraped\n(Mar 2026)': datetime(2026, 3, 15),
}

# Shaded regions for our data periods
ax.axvspan(datetime(2024, 1, 1), datetime(2024, 1, 31), alpha=0.15, color='green', label='Asaniczka period')
ax.axvspan(datetime(2024, 4, 1), datetime(2024, 4, 30), alpha=0.15, color='orange', label='Arshkon period')
# Scraped period is beyond JOLTS data range, annotate with arrow
ax.annotate('Scraped period\n(Mar 2026)', xy=(datetime(2025, 12, 1), 112),
            xytext=(datetime(2025, 6, 1), 200),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=9, color='red', ha='center')

# AI release windows
ai_releases = [
    (datetime(2024, 5, 13), 'GPT-4o'),
    (datetime(2024, 6, 20), 'Claude 3.5\nSonnet'),
    (datetime(2025, 5, 22), 'Claude 4'),
]
for date, label in ai_releases:
    ax.axvline(date, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(date, ax.get_ylim()[1] * 0.95, label, fontsize=7, ha='center', va='top', color='gray')

ax.set_xlabel('Date')
ax.set_ylabel('Job Openings (thousands)')
ax.set_title('JOLTS Information Sector Job Openings with Study Data Periods')
ax.legend(loc='upper left', fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/jolts_context.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved JOLTS context figure")


# Compute summary stats for our data periods
jolts_jan24 = jolts_sa.get('2024-01', None)
jolts_apr24 = jolts_sa.get('2024-04', None)

# Average JOLTS for 2024 (full year)
jolts_2024_vals = [v for k, v in jolts_sa.items() if k.startswith('2024')]
jolts_2024_mean = np.mean(jolts_2024_vals)

# Average for 2022 peak
jolts_2022_vals = [v for k, v in jolts_sa.items() if k.startswith('2022')]
jolts_2022_mean = np.mean(jolts_2022_vals)

# Average for late 2025
jolts_late2025 = [v for k, v in jolts_sa.items() if k.startswith('2025') and int(k.split('-')[1]) >= 9]
jolts_late2025_mean = np.mean(jolts_late2025) if jolts_late2025 else None

print(f"\nJOLTS Context:")
print(f"  2022 peak average: {jolts_2022_mean:.0f}K")
print(f"  2024 average: {jolts_2024_mean:.0f}K")
print(f"  Jan 2024 (asaniczka): {jolts_jan24}K")
print(f"  Apr 2024 (arshkon): {jolts_apr24}K")
print(f"  Late 2025 avg (Sep-Dec): {jolts_late2025_mean:.0f}K")
print(f"  2024 vs 2022 peak: {(jolts_2024_mean/jolts_2022_mean - 1)*100:.1f}%")


# =============================================================================
# 2. BLS OES State-Level Comparison
# =============================================================================
# BLS OES May 2023 state employment for SOC 15-1252 (Software Developers)
# Source: https://www.bls.gov/oes/2023/may/oes151252.htm
# These are the official published numbers (employment in occupations)

# May 2023 BLS OES state employment for Software Developers (15-1252)
# Top states - from BLS published data
bls_oes_15_1252 = {
    'CA': 228350, 'TX': 143930, 'WA': 94240, 'NY': 86120, 'VA': 79220,
    'FL': 64710, 'IL': 53660, 'GA': 52550, 'PA': 50960, 'MA': 50760,
    'NJ': 49350, 'NC': 48870, 'OH': 43770, 'MD': 40400, 'CO': 40140,
    'MN': 34150, 'MI': 31690, 'AZ': 29610, 'MO': 23810, 'CT': 20740,
    'IN': 20460, 'WI': 19930, 'OR': 19760, 'UT': 18990, 'TN': 18620,
    'AL': 12980, 'SC': 12590, 'IA': 11800, 'KS': 10910, 'NE': 10650,
    'DE': 8420, 'NH': 8190, 'KY': 7870, 'NV': 7360, 'LA': 7290,
    'OK': 7000, 'AR': 5780, 'NM': 5730, 'RI': 5240, 'ID': 4900,
    'DC': 15670, 'ME': 3370, 'MS': 3710, 'MT': 2240, 'WV': 2870,
    'ND': 2510, 'SD': 2530, 'HI': 2930, 'AK': 1050, 'WY': 670,
    'VT': 2440
}

# Also include 15-1256 (Software Quality Assurance Analysts and Testers) for combined SOC
bls_oes_15_1256 = {
    'CA': 32120, 'TX': 23200, 'WA': 11310, 'NY': 14430, 'VA': 10050,
    'FL': 10510, 'IL': 9770, 'GA': 7840, 'PA': 7980, 'MA': 7210,
    'NJ': 10220, 'NC': 7450, 'OH': 7710, 'MD': 5520, 'CO': 5840,
    'MN': 6390, 'MI': 5350, 'AZ': 4700, 'MO': 4330, 'CT': 3760,
    'IN': 3500, 'WI': 3340, 'OR': 2680, 'UT': 3040, 'TN': 3110,
}

# Combine 15-1252 + 15-1256 for states where both available
bls_combined = {}
for state in bls_oes_15_1252:
    bls_combined[state] = bls_oes_15_1252[state] + bls_oes_15_1256.get(state, 0)

# Get our state counts
our_state = con.execute(f"""
    SELECT state_normalized, source,
           COUNT(*) as n_swe
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER}
      AND state_normalized IS NOT NULL AND state_normalized != ''
    GROUP BY state_normalized, source
""").fetchdf()

our_state_total = our_state.groupby('state_normalized')['n_swe'].sum().reset_index()
our_state_total.columns = ['state', 'our_swe']

# Merge
comparison = our_state_total.copy()
comparison['bls_15_1252'] = comparison['state'].map(bls_oes_15_1252)
comparison['bls_combined'] = comparison['state'].map(bls_combined)
comparison = comparison.dropna(subset=['bls_15_1252'])

# Compute correlations
r_1252, p_1252 = stats.pearsonr(comparison['our_swe'], comparison['bls_15_1252'])
r_combined, p_combined = stats.pearsonr(comparison['our_swe'], comparison['bls_combined'])

# Log-log correlation (more appropriate for count data spanning orders of magnitude)
log_our = np.log10(comparison['our_swe'].values + 1)
log_bls = np.log10(comparison['bls_15_1252'].values + 1)
r_log, p_log = stats.pearsonr(log_our, log_bls)

# Spearman rank correlation
rho_rank, p_rank = stats.spearmanr(comparison['our_swe'], comparison['bls_15_1252'])

print(f"\nState-level correlation with BLS OES (May 2023):")
print(f"  N states: {len(comparison)}")
print(f"  Pearson r (15-1252): {r_1252:.4f} (p={p_1252:.2e})")
print(f"  Pearson r (combined): {r_combined:.4f} (p={p_combined:.2e})")
print(f"  Pearson r (log-log): {r_log:.4f} (p={p_log:.2e})")
print(f"  Spearman rho: {rho_rank:.4f} (p={p_rank:.2e})")

comparison.to_csv(f"{OUT_TABLES}/state_bls_comparison.csv", index=False)

# --- Plot state-level correlation ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Raw
ax = axes[0]
ax.scatter(comparison['bls_15_1252'] / 1000, comparison['our_swe'], alpha=0.6, s=30)
for _, row in comparison.iterrows():
    if row['our_swe'] > 3000 or row['bls_15_1252'] > 80000:
        ax.annotate(row['state'], (row['bls_15_1252']/1000, row['our_swe']),
                   fontsize=7, alpha=0.7)
ax.set_xlabel('BLS OES Employment (15-1252, thousands)')
ax.set_ylabel('Our LinkedIn SWE Postings')
ax.set_title(f'State-Level: Raw Counts\nr = {r_1252:.3f}')

# Log-log
ax = axes[1]
ax.scatter(log_bls, log_our, alpha=0.6, s=30)
# Regression line
slope, intercept = np.polyfit(log_bls, log_our, 1)
x_line = np.linspace(log_bls.min(), log_bls.max(), 100)
ax.plot(x_line, slope * x_line + intercept, 'r--', linewidth=1, alpha=0.7)
for _, row in comparison.iterrows():
    if row['our_swe'] > 3000 or row['bls_15_1252'] > 80000:
        ax.annotate(row['state'],
                   (np.log10(row['bls_15_1252']+1), np.log10(row['our_swe']+1)),
                   fontsize=7, alpha=0.7)
ax.set_xlabel('log10(BLS OES Employment)')
ax.set_ylabel('log10(Our LinkedIn SWE Postings)')
ax.set_title(f'State-Level: Log-Log\nr = {r_log:.3f}, rho = {rho_rank:.3f}')

plt.suptitle('Geographic Representativeness: Our Sample vs BLS OES May 2023', fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/state_bls_correlation.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved state-level BLS correlation figure")


# =============================================================================
# 3. Industry distribution comparison (our data only - BLS uses NAICS)
# =============================================================================
# We can't directly match our company_industry to NAICS, but we can show our distribution

industry_arshkon = con.execute(f"""
    SELECT company_industry, COUNT(*) as n,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER} AND source = 'kaggle_arshkon'
      AND company_industry IS NOT NULL AND company_industry != ''
    GROUP BY company_industry ORDER BY n DESC LIMIT 20
""").fetchdf()

industry_scraped = con.execute(f"""
    SELECT company_industry, COUNT(*) as n,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER} AND source = 'scraped'
      AND company_industry IS NOT NULL AND company_industry != ''
    GROUP BY company_industry ORDER BY n DESC LIMIT 20
""").fetchdf()

# Industry availability by source
industry_avail = con.execute(f"""
    SELECT source,
           SUM(CASE WHEN company_industry IS NOT NULL AND company_industry != '' THEN 1 ELSE 0 END) as n_with_industry,
           COUNT(*) as n_total,
           ROUND(SUM(CASE WHEN company_industry IS NOT NULL AND company_industry != '' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER}
    GROUP BY source ORDER BY source
""").fetchdf()

print("\nIndustry label availability:")
print(industry_avail.to_string(index=False))
print(f"\nTop industries (arshkon):")
print(industry_arshkon.to_string(index=False))
print(f"\nTop industries (scraped):")
print(industry_scraped.to_string(index=False))


# =============================================================================
# 4. Sample representativeness framing
# =============================================================================

# Our scraped data covers 26 metros. How much of US SWE employment do these cover?
# Using BLS data for the top metro areas (approximate from BLS OES metro data)
# Top 26 metros account for roughly 60-70% of US SWE employment

# Total BLS employment in our represented states
states_in_sample = our_state_total['state'].unique()
bls_in_sample = sum(bls_oes_15_1252.get(s, 0) for s in states_in_sample)
bls_total = sum(bls_oes_15_1252.values())

print(f"\nSample representativeness:")
print(f"  BLS total 15-1252 employment: {bls_total:,}")
print(f"  States in our sample: {len(states_in_sample)}")
print(f"  BLS employment in our states: {bls_in_sample:,} ({bls_in_sample/bls_total*100:.1f}%)")

# But we don't have uniform coverage -- our scraped data only covers 26 metros
n_scraped_metro = con.execute(f"""
    SELECT COUNT(*) FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER} AND source = 'scraped' AND metro_area IS NOT NULL
""").fetchone()[0]
n_scraped_total = con.execute(f"""
    SELECT COUNT(*) FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER} AND source = 'scraped'
""").fetchone()[0]
print(f"  Scraped SWE with metro assignment: {n_scraped_metro:,} / {n_scraped_total:,} ({n_scraped_metro/n_scraped_total*100:.1f}%)")

# Source-by-source counts
by_source = con.execute(f"""
    SELECT source, period,
           COUNT(*) as n_swe,
           SUM(CASE WHEN metro_area IS NOT NULL THEN 1 ELSE 0 END) as n_with_metro,
           SUM(CASE WHEN state_normalized IS NOT NULL AND state_normalized != '' THEN 1 ELSE 0 END) as n_with_state
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER}
    GROUP BY source, period ORDER BY source
""").fetchdf()
print("\nSource coverage:")
print(by_source.to_string(index=False))


# =============================================================================
# 5. Create summary representativeness figure
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Top 10 states comparison
top_states = comparison.nlargest(10, 'bls_15_1252')
x = np.arange(len(top_states))
width = 0.35

ax = axes[0]
# Normalize both to percentages of total
bls_pcts = top_states['bls_15_1252'] / top_states['bls_15_1252'].sum() * 100
our_pcts = top_states['our_swe'] / top_states['our_swe'].sum() * 100
ax.barh(x - width/2, bls_pcts, width, label='BLS OES 15-1252', color='steelblue', alpha=0.7)
ax.barh(x + width/2, our_pcts, width, label='Our SWE Postings', color='coral', alpha=0.7)
ax.set_yticks(x)
ax.set_yticklabels(top_states['state'])
ax.set_xlabel('Share of Top-10 State Total (%)')
ax.set_title('Top 10 States: BLS vs Our Sample')
ax.legend(fontsize=8)
ax.invert_yaxis()

# Panel 2: JOLTS trend context
ax = axes[1]
# Quarterly averages for cleaner visualization
jolts_df_copy = jolts_df.copy()
jolts_df_copy['quarter'] = jolts_df_copy['date'].dt.to_period('Q')
quarterly = jolts_df_copy.groupby('quarter')['openings_thousands'].mean().reset_index()
quarterly['date'] = quarterly['quarter'].dt.to_timestamp()

ax.bar(quarterly['date'], quarterly['openings_thousands'], width=80, alpha=0.6, color='steelblue')

# Annotate periods
for label, date in [('Asaniczka\n(Jan 24)', datetime(2024, 1, 15)),
                     ('Arshkon\n(Apr 24)', datetime(2024, 4, 15))]:
    ax.axvline(date, color='green', linestyle='--', alpha=0.6)
    ylim = ax.get_ylim()
    ax.text(date, ylim[1] * 0.92, label, fontsize=7, ha='center', color='green')

ax.set_xlabel('Quarter')
ax.set_ylabel('Avg Monthly Openings (000s)')
ax.set_title('JOLTS Info Sector: Quarterly Avg')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-Q%q'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.suptitle('External Benchmarks: Geographic & Hiring Cycle Context', fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/external_benchmarks.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved external benchmarks figure")

print("\nDone with Part B benchmarks.")
