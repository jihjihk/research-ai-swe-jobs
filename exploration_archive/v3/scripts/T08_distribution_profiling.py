#!/usr/bin/env python3
"""
T08: Distribution profiling & anomaly detection

Produces comprehensive baseline distributions for all available variables
by period and seniority, with deep investigation of the YOE paradox and
within-2024 calibration.

Outputs:
  exploration/figures/T08/  — PNG figures (150dpi)
  exploration/tables/T08/   — CSV tables
  exploration/reports/T08.md — analysis report
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import duckdb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE_DIR, 'data', 'unified.parquet')
FIG_DIR = os.path.join(BASE_DIR, 'exploration', 'figures', 'T08')
TBL_DIR = os.path.join(BASE_DIR, 'exploration', 'tables', 'T08')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

FILT = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe = true"

con = duckdb.connect()

# ── Colour palette ─────────────────────────────────────────────────────
PERIOD_COLORS = {'2024-01': '#4C72B0', '2024-04': '#DD8452', '2026-03': '#55A868'}
PERIOD_LABELS = {'2024-01': 'Jan 2024 (asaniczka)', '2024-04': 'Apr 2024 (arshkon)', '2026-03': 'Mar 2026 (scraped)'}
SEN_COLORS = {'junior': '#E24A33', 'mid': '#FBC15E', 'senior': '#348ABD', 'unknown': '#999999'}

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
})


def save_fig(fig, name, dpi=150):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {path}")


def save_table(df, name):
    path = os.path.join(TBL_DIR, name)
    df.to_csv(path, index=False)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Univariate profiling
# ═══════════════════════════════════════════════════════════════════════
print("\n=== STEP 1: Univariate profiling ===")

# 1a: Seniority distributions by period
print("  1a: Seniority distributions")
sen_dist = con.execute(f"""
SELECT period, seniority_final,
       COUNT(*) as n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY period), 2) as pct
FROM '{DATA}'
WHERE {FILT}
GROUP BY period, seniority_final
ORDER BY period, seniority_final
""").fetchdf()
save_table(sen_dist, 'seniority_final_by_period.csv')

sen3_dist = con.execute(f"""
SELECT period, seniority_3level,
       COUNT(*) as n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY period), 2) as pct
FROM '{DATA}'
WHERE {FILT}
GROUP BY period, seniority_3level
ORDER BY period, seniority_3level
""").fetchdf()
save_table(sen3_dist, 'seniority_3level_by_period.csv')

# Known-seniority only share
sen3_known = con.execute(f"""
SELECT period, seniority_3level,
       COUNT(*) as n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY period), 2) as pct
FROM '{DATA}'
WHERE {FILT} AND seniority_3level != 'unknown'
GROUP BY period, seniority_3level
ORDER BY period, seniority_3level
""").fetchdf()
save_table(sen3_known, 'seniority_3level_known_only.csv')


# 1b: Description length histograms by period
print("  1b: Description length distributions")
desc_data = con.execute(f"""
SELECT period, description_length, core_length
FROM '{DATA}'
WHERE {FILT}
""").fetchdf()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for period in ['2024-01', '2024-04', '2026-03']:
    sub = desc_data[desc_data['period'] == period]
    axes[0].hist(sub['description_length'].clip(upper=15000), bins=80, alpha=0.5,
                 color=PERIOD_COLORS[period], label=PERIOD_LABELS[period], density=True)
    axes[1].hist(sub['core_length'].dropna().clip(upper=12000), bins=80, alpha=0.5,
                 color=PERIOD_COLORS[period], label=PERIOD_LABELS[period], density=True)
axes[0].set_title('Description Length (chars)')
axes[0].set_xlabel('Characters')
axes[0].legend()
axes[1].set_title('Core Length (chars, after boilerplate removal)')
axes[1].set_xlabel('Characters')
axes[1].legend()
fig.suptitle('Description Length Distributions by Period (SWE, LinkedIn)', fontsize=13)
fig.tight_layout()
save_fig(fig, 'fig1_description_length_by_period.png')

# Description length summary stats
desc_stats = con.execute(f"""
SELECT period, seniority_3level,
       COUNT(*) as n,
       ROUND(AVG(description_length), 0) as mean_desc_len,
       ROUND(MEDIAN(description_length), 0) as med_desc_len,
       ROUND(STDDEV(description_length), 0) as sd_desc_len,
       ROUND(AVG(core_length), 0) as mean_core_len,
       ROUND(MEDIAN(core_length), 0) as med_core_len,
       ROUND(STDDEV(core_length), 0) as sd_core_len
FROM '{DATA}'
WHERE {FILT}
GROUP BY period, seniority_3level
ORDER BY period, seniority_3level
""").fetchdf()
save_table(desc_stats, 'description_length_by_period_seniority.csv')


# 1c: YOE distributions
print("  1c: YOE distributions")
yoe_data = con.execute(f"""
SELECT period, seniority_3level, yoe_extracted
FROM '{DATA}'
WHERE {FILT} AND yoe_extracted IS NOT NULL
""").fetchdf()

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
for i, period in enumerate(['2024-01', '2024-04', '2026-03']):
    sub = yoe_data[yoe_data['period'] == period]
    for sen in ['junior', 'mid', 'senior']:
        ssub = sub[sub['seniority_3level'] == sen]
        if len(ssub) > 10:
            axes[i].hist(ssub['yoe_extracted'].clip(upper=15), bins=range(0, 17), alpha=0.6,
                         color=SEN_COLORS[sen], label=sen, density=True)
    axes[i].set_title(PERIOD_LABELS[period])
    axes[i].set_xlabel('Years of Experience')
    axes[i].legend()
axes[0].set_ylabel('Density')
fig.suptitle('YOE Distribution by Period and Seniority (SWE, LinkedIn)', fontsize=13)
fig.tight_layout()
save_fig(fig, 'fig2_yoe_by_period_seniority.png')

# YOE summary stats
yoe_stats = con.execute(f"""
SELECT period, seniority_3level,
       COUNT(*) as n_total,
       SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) as n_yoe,
       ROUND(100.0 * SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_yoe,
       ROUND(MEDIAN(yoe_extracted), 1) as median_yoe,
       ROUND(AVG(yoe_extracted), 2) as mean_yoe,
       ROUND(STDDEV(yoe_extracted), 2) as sd_yoe,
       ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY yoe_extracted), 1) as q25,
       ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY yoe_extracted), 1) as q75
FROM '{DATA}'
WHERE {FILT}
GROUP BY period, seniority_3level
ORDER BY period, seniority_3level
""").fetchdf()
save_table(yoe_stats, 'yoe_by_period_seniority.csv')


# 1d: Aggregator distribution
print("  1d: Aggregator distribution")
agg_dist = con.execute(f"""
SELECT period, seniority_3level, is_aggregator,
       COUNT(*) as n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY period, seniority_3level), 2) as pct
FROM '{DATA}'
WHERE {FILT}
GROUP BY period, seniority_3level, is_aggregator
ORDER BY period, seniority_3level, is_aggregator
""").fetchdf()
save_table(agg_dist, 'aggregator_by_period_seniority.csv')


# 1e: Metro area distributions
print("  1e: Metro area distributions")
metro_dist = con.execute(f"""
WITH ranked AS (
  SELECT period, metro_area, COUNT(*) as n,
         ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY period), 2) as pct,
         ROW_NUMBER() OVER (PARTITION BY period ORDER BY COUNT(*) DESC) as rn
  FROM '{DATA}'
  WHERE {FILT} AND metro_area IS NOT NULL
  GROUP BY period, metro_area
)
SELECT period, metro_area, n, pct, rn
FROM ranked WHERE rn <= 15
ORDER BY period, rn
""").fetchdf()
save_table(metro_dist, 'metro_top15_by_period.csv')


# 1f: Company industry distributions (arshkon + scraped only)
print("  1f: Company industry distributions")
industry_dist = con.execute(f"""
WITH ranked AS (
  SELECT period, company_industry, COUNT(*) as n,
         ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY period), 2) as pct,
         ROW_NUMBER() OVER (PARTITION BY period ORDER BY COUNT(*) DESC) as rn
  FROM '{DATA}'
  WHERE {FILT} AND company_industry IS NOT NULL
  GROUP BY period, company_industry
)
SELECT period, company_industry, n, pct, rn
FROM ranked WHERE rn <= 15
ORDER BY period, rn
""").fetchdf()
save_table(industry_dist, 'industry_top15_by_period.csv')


# 1g: SWE classification tier
print("  1g: SWE classification tier")
tier_dist = con.execute(f"""
SELECT period, swe_classification_tier,
       COUNT(*) as n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY period), 2) as pct
FROM '{DATA}'
WHERE {FILT}
GROUP BY period, swe_classification_tier
ORDER BY period, swe_classification_tier
""").fetchdf()
save_table(tier_dist, 'swe_tier_by_period.csv')


# 1h: Seniority label provenance
print("  1h: Seniority label provenance")
prov_dist = con.execute(f"""
SELECT period, seniority_final_source,
       COUNT(*) as n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY period), 2) as pct
FROM '{DATA}'
WHERE {FILT}
GROUP BY period, seniority_final_source
ORDER BY period, seniority_final_source
""").fetchdf()
save_table(prov_dist, 'seniority_provenance_by_period.csv')


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Anomaly detection — bimodality, skew, subpopulations
# ═══════════════════════════════════════════════════════════════════════
print("\n=== STEP 2: Anomaly detection ===")

# 2a: Description length bimodality check by period
print("  2a: Description length bimodality / skew")
anomaly_records = []

for period in ['2024-01', '2024-04', '2026-03']:
    for col in ['description_length', 'core_length']:
        vals = con.execute(f"""
        SELECT {col} FROM '{DATA}'
        WHERE {FILT} AND period = '{period}' AND {col} IS NOT NULL
        """).fetchdf()[col].values
        skewness = float(stats.skew(vals))
        kurtosis = float(stats.kurtosis(vals))
        # Hartigan's dip test approximation via coefficient of variation
        cv = float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0
        anomaly_records.append({
            'period': period, 'variable': col,
            'n': len(vals), 'mean': round(float(np.mean(vals)), 0),
            'median': round(float(np.median(vals)), 0),
            'skewness': round(skewness, 3), 'kurtosis': round(kurtosis, 3),
            'cv': round(cv, 3),
            'flag': 'HIGH_SKEW' if abs(skewness) > 2 else ('MODERATE_SKEW' if abs(skewness) > 1 else 'ok')
        })

for period in ['2024-04', '2026-03']:
    vals = con.execute(f"""
    SELECT yoe_extracted FROM '{DATA}'
    WHERE {FILT} AND period = '{period}' AND yoe_extracted IS NOT NULL
    """).fetchdf()['yoe_extracted'].values
    skewness = float(stats.skew(vals))
    kurtosis = float(stats.kurtosis(vals))
    anomaly_records.append({
        'period': period, 'variable': 'yoe_extracted',
        'n': len(vals), 'mean': round(float(np.mean(vals)), 2),
        'median': round(float(np.median(vals)), 1),
        'skewness': round(skewness, 3), 'kurtosis': round(kurtosis, 3),
        'cv': round(float(np.std(vals) / np.mean(vals)), 3),
        'flag': 'HIGH_SKEW' if abs(skewness) > 2 else ('MODERATE_SKEW' if abs(skewness) > 1 else 'ok')
    })

import pandas as pd
anomaly_df = pd.DataFrame(anomaly_records)
save_table(anomaly_df, 'anomaly_flags.csv')

# 2b: Entry-level YOE bimodality check (arshkon 2024-04)
print("  2b: Entry-level YOE bimodality (arshkon)")
yoe_entry_arshkon = con.execute(f"""
SELECT yoe_extracted FROM '{DATA}'
WHERE {FILT} AND seniority_final = 'entry' AND period = '2024-04' AND yoe_extracted IS NOT NULL
""").fetchdf()['yoe_extracted'].values

print(f"    arshkon entry YOE: n={len(yoe_entry_arshkon)}, mean={np.mean(yoe_entry_arshkon):.1f}, "
      f"median={np.median(yoe_entry_arshkon):.1f}, skew={stats.skew(yoe_entry_arshkon):.2f}")

yoe_entry_scraped = con.execute(f"""
SELECT yoe_extracted FROM '{DATA}'
WHERE {FILT} AND seniority_final = 'entry' AND period = '2026-03' AND yoe_extracted IS NOT NULL
""").fetchdf()['yoe_extracted'].values

print(f"    scraped entry YOE: n={len(yoe_entry_scraped)}, mean={np.mean(yoe_entry_scraped):.1f}, "
      f"median={np.median(yoe_entry_scraped):.1f}, skew={stats.skew(yoe_entry_scraped):.2f}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: YOE Paradox Deep Dive (HIGHEST PRIORITY)
# ═══════════════════════════════════════════════════════════════════════
print("\n=== STEP 3: YOE Paradox Deep Dive ===")

# 3a: Entry YOE by seniority_final_source (provenance)
print("  3a: Entry YOE by provenance")
yoe_prov = con.execute(f"""
SELECT period, seniority_final_source,
       COUNT(*) as n,
       SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) as has_yoe,
       ROUND(100.0 * SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_yoe,
       ROUND(MEDIAN(yoe_extracted), 1) as median_yoe,
       ROUND(AVG(yoe_extracted), 2) as mean_yoe,
       ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY yoe_extracted), 1) as q25,
       ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY yoe_extracted), 1) as q75
FROM '{DATA}'
WHERE {FILT} AND seniority_final = 'entry'
GROUP BY period, seniority_final_source
ORDER BY period, seniority_final_source
""").fetchdf()
save_table(yoe_prov, 'yoe_entry_by_provenance.csv')

# 3b: Entry YOE by aggregator status
print("  3b: Entry YOE by aggregator")
yoe_agg = con.execute(f"""
SELECT period, is_aggregator,
       COUNT(*) as n,
       SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) as has_yoe,
       ROUND(MEDIAN(yoe_extracted), 1) as median_yoe,
       ROUND(AVG(yoe_extracted), 2) as mean_yoe,
       ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY yoe_extracted), 1) as q25,
       ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY yoe_extracted), 1) as q75
FROM '{DATA}'
WHERE {FILT} AND seniority_final = 'entry'
GROUP BY period, is_aggregator
ORDER BY period, is_aggregator
""").fetchdf()
save_table(yoe_agg, 'yoe_entry_by_aggregator.csv')

# 3c: Entry YOE by top titles (matching titles across periods)
print("  3c: Entry YOE by matching titles")
yoe_titles = con.execute(f"""
WITH t1 AS (
  SELECT title_normalized, COUNT(*) as n24,
         ROUND(MEDIAN(yoe_extracted), 1) as med_yoe_24,
         ROUND(AVG(yoe_extracted), 2) as mean_yoe_24,
         SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) as has_yoe_24
  FROM '{DATA}'
  WHERE {FILT} AND seniority_final = 'entry' AND period = '2024-04'
  GROUP BY title_normalized
  HAVING COUNT(*) >= 5
),
t2 AS (
  SELECT title_normalized, COUNT(*) as n26,
         ROUND(MEDIAN(yoe_extracted), 1) as med_yoe_26,
         ROUND(AVG(yoe_extracted), 2) as mean_yoe_26,
         SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) as has_yoe_26
  FROM '{DATA}'
  WHERE {FILT} AND seniority_final = 'entry' AND period = '2026-03'
  GROUP BY title_normalized
  HAVING COUNT(*) >= 5
)
SELECT t1.title_normalized, n24, n26, has_yoe_24, has_yoe_26,
       med_yoe_24, med_yoe_26, mean_yoe_24, mean_yoe_26,
       med_yoe_26 - med_yoe_24 as med_change
FROM t1 JOIN t2 USING (title_normalized)
ORDER BY n24 + n26 DESC
""").fetchdf()
save_table(yoe_titles, 'yoe_entry_matched_titles.csv')

# 3d: Company capping sensitivity
print("  3d: Company capping sensitivity")
cap_results = []
for cap in [3, 5, 10, 50, 9999]:
    r = con.execute(f"""
    WITH capped AS (
      SELECT *, ROW_NUMBER() OVER (PARTITION BY period, company_name_canonical ORDER BY uid) as rn
      FROM '{DATA}'
      WHERE {FILT} AND seniority_final = 'entry'
    )
    SELECT period,
           COUNT(*) as n,
           SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) as has_yoe,
           ROUND(MEDIAN(yoe_extracted), 2) as median_yoe,
           ROUND(AVG(yoe_extracted), 3) as mean_yoe
    FROM capped WHERE rn <= {cap}
    GROUP BY period
    ORDER BY period
    """).fetchdf()
    r['company_cap'] = cap
    cap_results.append(r)

cap_df = pd.concat(cap_results, ignore_index=True)
save_table(cap_df, 'yoe_entry_company_capping.csv')

# 3e: YOE binned distribution (slot elimination)
print("  3e: Slot elimination analysis")
yoe_bins = con.execute(f"""
SELECT period,
       CASE WHEN yoe_extracted IS NULL THEN 'missing'
            WHEN yoe_extracted = 0 THEN '0'
            WHEN yoe_extracted BETWEEN 1 AND 2 THEN '1-2'
            WHEN yoe_extracted BETWEEN 3 AND 4 THEN '3-4'
            WHEN yoe_extracted >= 5 THEN '5+'
       END as yoe_bin,
       COUNT(*) as n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY period), 2) as pct
FROM '{DATA}'
WHERE {FILT} AND seniority_final = 'entry'
GROUP BY period, yoe_bin
ORDER BY period, yoe_bin
""").fetchdf()
save_table(yoe_bins, 'yoe_entry_binned.csv')

# Slot elimination by aggregator
yoe_bins_agg = con.execute(f"""
SELECT period, is_aggregator,
       CASE WHEN yoe_extracted IS NULL THEN 'missing'
            WHEN yoe_extracted = 0 THEN '0'
            WHEN yoe_extracted BETWEEN 1 AND 2 THEN '1-2'
            WHEN yoe_extracted BETWEEN 3 AND 4 THEN '3-4'
            WHEN yoe_extracted >= 5 THEN '5+'
       END as yoe_bin,
       COUNT(*) as n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY period, is_aggregator), 2) as pct
FROM '{DATA}'
WHERE {FILT} AND seniority_final = 'entry'
GROUP BY period, is_aggregator, yoe_bin
ORDER BY period, is_aggregator, yoe_bin
""").fetchdf()
save_table(yoe_bins_agg, 'yoe_entry_binned_by_aggregator.csv')

# 3f: Top entry-level companies comparison
print("  3f: Top entry-level companies")
for period_code, period_label in [('2024-04', 'arshkon'), ('2026-03', 'scraped')]:
    top_co = con.execute(f"""
    SELECT company_name_canonical, is_aggregator,
           COUNT(*) as n,
           SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) as has_yoe,
           ROUND(MEDIAN(yoe_extracted), 1) as med_yoe,
           ROUND(AVG(yoe_extracted), 2) as mean_yoe
    FROM '{DATA}'
    WHERE {FILT} AND seniority_final = 'entry' AND period = '{period_code}'
    GROUP BY company_name_canonical, is_aggregator
    ORDER BY n DESC
    LIMIT 25
    """).fetchdf()
    save_table(top_co, f'entry_top_companies_{period_label}.csv')

# 3g: YOE extraction coverage analysis
print("  3g: YOE extraction coverage")
yoe_coverage = con.execute(f"""
SELECT period, seniority_3level,
       COUNT(*) as n_total,
       SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) as has_yoe,
       ROUND(100.0 * SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_has_yoe,
       -- check if short descriptions explain missing YOE
       SUM(CASE WHEN yoe_extracted IS NULL AND description_length < 500 THEN 1 ELSE 0 END) as no_yoe_short_desc,
       SUM(CASE WHEN yoe_extracted IS NULL AND description_length >= 500 THEN 1 ELSE 0 END) as no_yoe_long_desc
FROM '{DATA}'
WHERE {FILT}
GROUP BY period, seniority_3level
ORDER BY period, seniority_3level
""").fetchdf()
save_table(yoe_coverage, 'yoe_extraction_coverage.csv')


# ── YOE Paradox figure ──
print("  3h: YOE paradox figure")

# Figure 3: YOE paradox -- entry-level YOE histogram by period
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Panel 1: Entry-level YOE distribution comparison
for period, color in [('2024-04', PERIOD_COLORS['2024-04']), ('2026-03', PERIOD_COLORS['2026-03'])]:
    vals = con.execute(f"""
    SELECT yoe_extracted FROM '{DATA}'
    WHERE {FILT} AND seniority_final = 'entry' AND period = '{period}' AND yoe_extracted IS NOT NULL
    """).fetchdf()['yoe_extracted'].values
    axes[0].hist(vals.clip(max=12), bins=range(0, 14), alpha=0.6, color=color,
                 label=PERIOD_LABELS[period], density=True, edgecolor='white')
axes[0].set_title('Entry-Level YOE Distribution')
axes[0].set_xlabel('Years of Experience')
axes[0].set_ylabel('Density')
axes[0].legend()

# Panel 2: YOE bin shares (slot elimination)
bin_order = ['0', '1-2', '3-4', '5+']
for i, period in enumerate(['2024-04', '2026-03']):
    vals = []
    for b in bin_order:
        r = yoe_bins[(yoe_bins['period'] == period) & (yoe_bins['yoe_bin'] == b)]
        vals.append(float(r['pct'].iloc[0]) if len(r) > 0 else 0)
    x = np.arange(len(bin_order))
    w = 0.35
    axes[1].bar(x + i*w, vals, w, color=PERIOD_COLORS[period],
                label=PERIOD_LABELS[period], edgecolor='white')
axes[1].set_xticks(np.arange(len(bin_order)) + 0.175)
axes[1].set_xticklabels(bin_order)
axes[1].set_title('Entry-Level YOE Bins (% of all entry)')
axes[1].set_ylabel('% of Entry Postings')
axes[1].legend()

# Panel 3: native_backfill entry YOE histogram
for period, color in [('2024-04', PERIOD_COLORS['2024-04']), ('2026-03', PERIOD_COLORS['2026-03'])]:
    vals = con.execute(f"""
    SELECT yoe_extracted FROM '{DATA}'
    WHERE {FILT} AND seniority_final = 'entry' AND period = '{period}'
      AND seniority_final_source = 'native_backfill' AND yoe_extracted IS NOT NULL
    """).fetchdf()['yoe_extracted'].values
    axes[2].hist(vals.clip(max=12), bins=range(0, 14), alpha=0.6, color=color,
                 label=f'{PERIOD_LABELS[period]} (n={len(vals)})', density=True, edgecolor='white')
axes[2].set_title('Native-Backfill Entry YOE Only')
axes[2].set_xlabel('Years of Experience')
axes[2].set_ylabel('Density')
axes[2].legend()

fig.suptitle('YOE Paradox: Entry-Level YOE Decreased 2024→2026', fontsize=13, fontweight='bold')
fig.tight_layout()
save_fig(fig, 'fig3_yoe_paradox.png')


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Within-2024 calibration (arshkon vs asaniczka, mid-senior SWE)
# ═══════════════════════════════════════════════════════════════════════
print("\n=== STEP 4: Within-2024 calibration ===")

# Load tech matrix for calibration
tech_matrix = con.execute(f"""
SELECT t.*, s.period, s.source, s.seniority_final, s.is_aggregator
FROM 'exploration/artifacts/shared/swe_tech_matrix.parquet' t
JOIN (
  SELECT uid, period, source, seniority_final, is_aggregator
  FROM '{DATA}'
  WHERE {FILT}
) s ON t.uid = s.uid
""").fetchdf()

tech_cols = [c for c in tech_matrix.columns if c not in ['uid', 'period', 'source', 'seniority_final', 'is_aggregator']]

# Mid-senior only for calibration
ms_arshkon = tech_matrix[(tech_matrix['source'] == 'kaggle_arshkon') & (tech_matrix['seniority_final'] == 'mid-senior')]
ms_asaniczka = tech_matrix[(tech_matrix['source'] == 'kaggle_asaniczka') & (tech_matrix['seniority_final'] == 'mid-senior')]
ms_scraped = tech_matrix[(tech_matrix['source'] == 'scraped') & (tech_matrix['seniority_final'] == 'mid-senior')]

calibration_records = []
for col in tech_cols:
    r1 = ms_asaniczka[col].mean()
    r2 = ms_arshkon[col].mean()
    r3 = ms_scraped[col].mean()

    # Within-2024 difference (arshkon - asaniczka)
    within_2024 = r2 - r1
    # 2024-to-2026 difference (scraped - arshkon)
    to_2026 = r3 - r2

    calibration_records.append({
        'technology': col,
        'asaniczka_rate': round(r1, 4),
        'arshkon_rate': round(r2, 4),
        'scraped_rate': round(r3, 4),
        'within_2024_diff': round(within_2024, 4),
        '2024_2026_diff': round(to_2026, 4),
        'ratio': round(abs(to_2026) / max(abs(within_2024), 0.001), 2) if abs(within_2024) > 0.005 else np.nan
    })

calib_df = pd.DataFrame(calibration_records)
calib_df = calib_df.sort_values('2024_2026_diff', key=abs, ascending=False)
save_table(calib_df, 'within_2024_calibration_tech.csv')

# Description length calibration
desc_calib = con.execute(f"""
SELECT source, period,
       COUNT(*) as n,
       ROUND(AVG(description_length), 1) as mean_desc_len,
       ROUND(STDDEV(description_length), 1) as sd_desc_len,
       ROUND(AVG(core_length), 1) as mean_core_len,
       ROUND(STDDEV(core_length), 1) as sd_core_len,
       ROUND(AVG(yoe_extracted), 2) as mean_yoe,
       ROUND(STDDEV(yoe_extracted), 2) as sd_yoe
FROM '{DATA}'
WHERE {FILT} AND seniority_final = 'mid-senior'
GROUP BY source, period
ORDER BY period, source
""").fetchdf()
save_table(desc_calib, 'calibration_midsenior_desc.csv')

# Cohen's d for description length
def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled_std = np.sqrt(((n1-1)*np.std(g1, ddof=1)**2 + (n2-1)*np.std(g2, ddof=1)**2) / (n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0

dl_asa = con.execute(f"SELECT description_length FROM '{DATA}' WHERE {FILT} AND seniority_final='mid-senior' AND source='kaggle_asaniczka'").fetchdf()['description_length'].values
dl_arsh = con.execute(f"SELECT description_length FROM '{DATA}' WHERE {FILT} AND seniority_final='mid-senior' AND source='kaggle_arshkon'").fetchdf()['description_length'].values
dl_scr = con.execute(f"SELECT description_length FROM '{DATA}' WHERE {FILT} AND seniority_final='mid-senior' AND source='scraped'").fetchdf()['description_length'].values

cl_asa = con.execute(f"SELECT core_length FROM '{DATA}' WHERE {FILT} AND seniority_final='mid-senior' AND source='kaggle_asaniczka' AND core_length IS NOT NULL").fetchdf()['core_length'].values
cl_arsh = con.execute(f"SELECT core_length FROM '{DATA}' WHERE {FILT} AND seniority_final='mid-senior' AND source='kaggle_arshkon' AND core_length IS NOT NULL").fetchdf()['core_length'].values
cl_scr = con.execute(f"SELECT core_length FROM '{DATA}' WHERE {FILT} AND seniority_final='mid-senior' AND source='scraped' AND core_length IS NOT NULL").fetchdf()['core_length'].values

calib_effect_records = []
metrics = [
    ('description_length', dl_asa, dl_arsh, dl_scr),
    ('core_length', cl_asa, cl_arsh, cl_scr),
]

for name, v_asa, v_arsh, v_scr in metrics:
    d_within = cohens_d(v_arsh, v_asa)
    d_trend = cohens_d(v_scr, v_arsh)
    calib_effect_records.append({
        'metric': name,
        'within_2024_d': round(d_within, 3),
        '2024_2026_d': round(d_trend, 3),
        'ratio_trend_within': round(abs(d_trend) / max(abs(d_within), 0.001), 2),
        'signal_exceeds_noise': abs(d_trend) > abs(d_within)
    })

calib_effect_df = pd.DataFrame(calib_effect_records)
save_table(calib_effect_df, 'calibration_effect_sizes.csv')


# ═══════════════════════════════════════════════════════════════════════
# STEP 5: Junior share trends
# ═══════════════════════════════════════════════════════════════════════
print("\n=== STEP 5: Junior share trends ===")

# Using seniority_final (excluding asaniczka)
jr_share = con.execute(f"""
SELECT period, seniority_final,
       COUNT(*) as n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY period), 2) as pct_all,
       ROUND(100.0 * COUNT(*) /
             SUM(CASE WHEN seniority_final != 'unknown' THEN COUNT(*) ELSE 0 END) OVER (PARTITION BY period), 2) as pct_known
FROM '{DATA}'
WHERE {FILT} AND source != 'kaggle_asaniczka'
GROUP BY period, seniority_final
ORDER BY period, seniority_final
""").fetchdf()
save_table(jr_share, 'junior_share_by_period.csv')

# Sensitivity: aggregator exclusion
jr_share_noagg = con.execute(f"""
SELECT period, seniority_final,
       COUNT(*) as n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY period), 2) as pct_all
FROM '{DATA}'
WHERE {FILT} AND source != 'kaggle_asaniczka' AND is_aggregator = false
GROUP BY period, seniority_final
ORDER BY period, seniority_final
""").fetchdf()
save_table(jr_share_noagg, 'junior_share_noagg.csv')

# Sensitivity: seniority operationalization
jr_share_native = con.execute(f"""
SELECT period, seniority_native,
       COUNT(*) as n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY period), 2) as pct_all
FROM '{DATA}'
WHERE {FILT} AND source != 'kaggle_asaniczka' AND seniority_native IS NOT NULL
GROUP BY period, seniority_native
ORDER BY period, seniority_native
""").fetchdf()
save_table(jr_share_native, 'junior_share_native.csv')


# ═══════════════════════════════════════════════════════════════════════
# STEP 6: Ranked change list — largest effect sizes between periods
# ═══════════════════════════════════════════════════════════════════════
print("\n=== STEP 6: Ranked change list ===")

change_records = []

# 6a: Tech prevalence changes (arshkon -> scraped, mid-senior)
for col in tech_cols:
    r_arsh = ms_arshkon[col].mean()
    r_scr = ms_scraped[col].mean()
    n1 = len(ms_arshkon)
    n2 = len(ms_scraped)

    if r_arsh > 0.01 or r_scr > 0.01:  # at least 1% in one period
        diff = r_scr - r_arsh
        # Cohen's h for proportions
        h = 2 * (np.arcsin(np.sqrt(r_scr)) - np.arcsin(np.sqrt(r_arsh)))
        change_records.append({
            'variable': col, 'category': 'technology',
            'arshkon_2024': round(r_arsh, 4),
            'scraped_2026': round(r_scr, 4),
            'diff': round(diff, 4),
            'cohens_h': round(h, 3),
            'abs_h': round(abs(h), 3)
        })

# 6b: Structural changes
# Aggregator share
for sen in ['junior', 'mid', 'senior']:
    r1 = con.execute(f"""
    SELECT ROUND(AVG(CASE WHEN is_aggregator THEN 1.0 ELSE 0.0 END), 4) as rate
    FROM '{DATA}' WHERE {FILT} AND period='2024-04' AND seniority_3level='{sen}'
    """).fetchdf()['rate'].iloc[0]
    r2 = con.execute(f"""
    SELECT ROUND(AVG(CASE WHEN is_aggregator THEN 1.0 ELSE 0.0 END), 4) as rate
    FROM '{DATA}' WHERE {FILT} AND period='2026-03' AND seniority_3level='{sen}'
    """).fetchdf()['rate'].iloc[0]
    r1, r2 = float(r1), float(r2)
    h = 2 * (np.arcsin(np.sqrt(r2)) - np.arcsin(np.sqrt(r1))) if r1 > 0 and r2 > 0 else 0
    change_records.append({
        'variable': f'aggregator_share_{sen}', 'category': 'structural',
        'arshkon_2024': round(r1, 4), 'scraped_2026': round(r2, 4),
        'diff': round(r2-r1, 4), 'cohens_h': round(h, 3), 'abs_h': round(abs(h), 3)
    })

# Description length effect size
d_desc = cohens_d(dl_scr, dl_arsh)
change_records.append({
    'variable': 'description_length_midsenior', 'category': 'structural',
    'arshkon_2024': round(float(np.mean(dl_arsh)), 0),
    'scraped_2026': round(float(np.mean(dl_scr)), 0),
    'diff': round(float(np.mean(dl_scr) - np.mean(dl_arsh)), 0),
    'cohens_h': round(d_desc, 3), 'abs_h': round(abs(d_desc), 3)
})

d_core = cohens_d(cl_scr, cl_arsh)
change_records.append({
    'variable': 'core_length_midsenior', 'category': 'structural',
    'arshkon_2024': round(float(np.mean(cl_arsh)), 0),
    'scraped_2026': round(float(np.mean(cl_scr)), 0),
    'diff': round(float(np.mean(cl_scr) - np.mean(cl_arsh)), 0),
    'cohens_h': round(d_core, 3), 'abs_h': round(abs(d_core), 3)
})

change_df = pd.DataFrame(change_records)
change_df = change_df.sort_values('abs_h', ascending=False)
save_table(change_df, 'ranked_changes.csv')


# ═══════════════════════════════════════════════════════════════════════
# STEP 7: Additional sensitivity analyses
# ═══════════════════════════════════════════════════════════════════════
print("\n=== STEP 7: Sensitivity analyses ===")

# 7a: SWE classification tier sensitivity for junior share
print("  7a: SWE tier sensitivity")
jr_by_tier = con.execute(f"""
SELECT period, swe_classification_tier, seniority_3level,
       COUNT(*) as n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY period, swe_classification_tier), 2) as pct
FROM '{DATA}'
WHERE {FILT} AND source != 'kaggle_asaniczka' AND seniority_3level != 'unknown'
GROUP BY period, swe_classification_tier, seniority_3level
ORDER BY period, swe_classification_tier, seniority_3level
""").fetchdf()
save_table(jr_by_tier, 'junior_share_by_swe_tier.csv')

# 7b: Aggregator exclusion sensitivity for distributions
print("  7b: Aggregator exclusion")
yoe_noagg = con.execute(f"""
SELECT period, seniority_3level,
       COUNT(*) as n,
       ROUND(MEDIAN(yoe_extracted), 1) as median_yoe,
       ROUND(AVG(yoe_extracted), 2) as mean_yoe,
       ROUND(MEDIAN(description_length), 0) as med_desc_len
FROM '{DATA}'
WHERE {FILT} AND is_aggregator = false
GROUP BY period, seniority_3level
ORDER BY period, seniority_3level
""").fetchdf()
save_table(yoe_noagg, 'distributions_noagg.csv')


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Comprehensive summary — junior share + calibration
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Generating summary figure ===")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Panel A: Junior share by period (excl asaniczka, known seniority only)
jr_data = con.execute(f"""
SELECT period, seniority_3level,
       COUNT(*) as n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY period), 2) as pct
FROM '{DATA}'
WHERE {FILT} AND source != 'kaggle_asaniczka' AND seniority_3level != 'unknown'
GROUP BY period, seniority_3level
ORDER BY period, seniority_3level
""").fetchdf()

for period in ['2024-04', '2026-03']:
    sub = jr_data[jr_data['period'] == period]
    levels = sub['seniority_3level'].tolist()
    pcts = sub['pct'].tolist()
    x = np.arange(len(levels))
    w = 0.35
    offset = 0 if period == '2024-04' else 0.35
    axes[0, 0].bar(x + offset, pcts, w, color=PERIOD_COLORS[period],
                    label=PERIOD_LABELS[period], edgecolor='white')
axes[0, 0].set_xticks(np.arange(len(levels)) + 0.175)
axes[0, 0].set_xticklabels(levels)
axes[0, 0].set_title('A. Seniority Share (known seniority, excl asaniczka)')
axes[0, 0].set_ylabel('% of Known-Seniority SWE')
axes[0, 0].legend()

# Panel B: Entry-level YOE box comparison
entry_yoe_24 = con.execute(f"""
SELECT yoe_extracted FROM '{DATA}'
WHERE {FILT} AND seniority_final='entry' AND period='2024-04' AND yoe_extracted IS NOT NULL
""").fetchdf()['yoe_extracted'].values
entry_yoe_26 = con.execute(f"""
SELECT yoe_extracted FROM '{DATA}'
WHERE {FILT} AND seniority_final='entry' AND period='2026-03' AND yoe_extracted IS NOT NULL
""").fetchdf()['yoe_extracted'].values

bp = axes[0, 1].boxplot([entry_yoe_24, entry_yoe_26],
                         labels=['Apr 2024\n(arshkon)', 'Mar 2026\n(scraped)'],
                         patch_artist=True, showfliers=True,
                         flierprops={'markersize': 2, 'alpha': 0.3})
bp['boxes'][0].set_facecolor(PERIOD_COLORS['2024-04'])
bp['boxes'][1].set_facecolor(PERIOD_COLORS['2026-03'])
for box in bp['boxes']:
    box.set_alpha(0.7)
axes[0, 1].set_title('B. Entry-Level YOE Distribution')
axes[0, 1].set_ylabel('Years of Experience')
axes[0, 1].annotate(f'Median: {np.median(entry_yoe_24):.0f} → {np.median(entry_yoe_26):.0f}',
                     xy=(0.5, 0.95), xycoords='axes fraction', ha='center',
                     fontsize=10, fontweight='bold', color='red')

# Panel C: Top tech changes (ranked by absolute Cohen's h)
top_tech = change_df[change_df['category'] == 'technology'].head(20)
y_pos = np.arange(len(top_tech))
colors = ['#55A868' if d > 0 else '#C44E52' for d in top_tech['cohens_h']]
axes[1, 0].barh(y_pos, top_tech['cohens_h'], color=colors, edgecolor='white')
axes[1, 0].set_yticks(y_pos)
axes[1, 0].set_yticklabels(top_tech['variable'], fontsize=7)
axes[1, 0].set_title('C. Top 20 Tech Changes (Cohen\'s h, 2024→2026)')
axes[1, 0].set_xlabel("Cohen's h (positive = increased)")
axes[1, 0].axvline(0, color='black', linewidth=0.5)
axes[1, 0].invert_yaxis()

# Panel D: Calibration scatter (within-2024 vs 2024-2026 for tech)
calib_tech = calib_df.dropna(subset=['ratio'])
within = calib_tech['within_2024_diff'].abs()
trend = calib_tech['2024_2026_diff'].abs()
axes[1, 1].scatter(within, trend, alpha=0.4, s=20, color='#4C72B0')
max_val = max(within.max(), trend.max()) * 1.1
axes[1, 1].plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)
axes[1, 1].set_title('D. Calibration: Within-2024 vs 2024→2026 Changes')
axes[1, 1].set_xlabel('|Within-2024 difference| (instrument noise)')
axes[1, 1].set_ylabel('|2024→2026 difference| (signal + noise)')
# Label outliers
for _, row in calib_tech.iterrows():
    if abs(row['2024_2026_diff']) > 0.08 and row['ratio'] > 3:
        axes[1, 1].annotate(row['technology'], (abs(row['within_2024_diff']), abs(row['2024_2026_diff'])),
                            fontsize=6, alpha=0.7)

fig.suptitle('T08: Distribution Profiling & Anomaly Detection — Key Results', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
save_fig(fig, 'fig4_summary.png')


print("\n=== All outputs complete ===")
print(f"Figures: {FIG_DIR}")
print(f"Tables: {TBL_DIR}")
