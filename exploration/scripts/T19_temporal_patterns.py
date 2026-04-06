#!/usr/bin/env python3
"""T19: Temporal patterns & rate-of-change estimation.

Characterizes temporal structure, estimates rates of change across snapshots,
and checks within-period stability.
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DATA = 'data/unified.parquet'
FIG_DIR = Path('exploration/figures/T19')
TBL_DIR = Path('exploration/tables/T19')
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

BASE_FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"

con = duckdb.connect()
con.execute("PRAGMA enable_progress_bar=false")

# AI keyword patterns (same as T18)
AI_BROAD = """(
    LOWER(description) LIKE '%artificial intelligence%'
    OR LOWER(description) LIKE '%machine learning%'
    OR LOWER(description) LIKE '%deep learning%'
    OR LOWER(description) LIKE '%neural network%'
    OR LOWER(description) LIKE '% llm%'
    OR LOWER(description) LIKE '%large language model%'
    OR LOWER(description) LIKE '% gpt%'
    OR LOWER(description) LIKE '%openai%'
    OR LOWER(description) LIKE '%chatgpt%'
    OR LOWER(description) LIKE '%generative ai%'
    OR LOWER(description) LIKE '%copilot%'
    OR LOWER(description) LIKE '%prompt engineer%'
    OR LOWER(description) LIKE '%langchain%'
    OR LOWER(description) LIKE '%retrieval augmented%'
    OR LOWER(description) LIKE '%ai/ml%'
    OR LOWER(description) LIKE '%ml/ai%'
    OR LOWER(description) LIKE '%computer vision%'
    OR LOWER(description) LIKE '%natural language processing%'
    OR LOWER(description) LIKE '% nlp %'
    OR LOWER(description) LIKE '%tensorflow%'
    OR LOWER(description) LIKE '%pytorch%'
)"""

GENAI = """(
    LOWER(description) LIKE '% llm%'
    OR LOWER(description) LIKE '%large language model%'
    OR LOWER(description) LIKE '% gpt%'
    OR LOWER(description) LIKE '%openai%'
    OR LOWER(description) LIKE '%chatgpt%'
    OR LOWER(description) LIKE '%generative ai%'
    OR LOWER(description) LIKE '%copilot%'
    OR LOWER(description) LIKE '%prompt engineer%'
    OR LOWER(description) LIKE '%langchain%'
    OR LOWER(description) LIKE '%retrieval augmented%'
    OR LOWER(description) LIKE '%ai agent%'
    OR LOWER(description) LIKE '%agentic%'
)"""

SCOPE_COLLAB = """(
    LOWER(description) LIKE '%cross-functional%'
    OR LOWER(description) LIKE '%cross functional%'
    OR LOWER(description) LIKE '%stakeholder%'
)"""

TECH_COUNT_EXPR = """(
    (CASE WHEN LOWER(description) LIKE '%python%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%java %' OR LOWER(description) LIKE '%java,%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%javascript%' OR LOWER(description) LIKE '%typescript%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%react%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%angular%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '% c++%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '% c#%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%golang%' OR LOWER(description) LIKE '% go %' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%kubernetes%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%docker%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%terraform%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%aws%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%azure%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%gcp%' OR LOWER(description) LIKE '%google cloud%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%sql%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%node.js%' OR LOWER(description) LIKE '%nodejs%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%ruby%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%scala%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%spark%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%kafka%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%redis%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%mongodb%' OR LOWER(description) LIKE '%postgres%' OR LOWER(description) LIKE '%mysql%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%ci/cd%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '% git %' OR LOWER(description) LIKE '%github%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%linux%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%agile%' OR LOWER(description) LIKE '%scrum%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%rest api%' OR LOWER(description) LIKE '%restful%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%microservice%' THEN 1 ELSE 0 END) +
    (CASE WHEN LOWER(description) LIKE '%swift%' OR LOWER(description) LIKE '%kotlin%' THEN 1 ELSE 0 END)
)"""


print("=" * 60)
print("T19: Temporal Patterns & Rate-of-Change Estimation")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 1: Rate-of-change estimation
# ─────────────────────────────────────────────

print("\n--- STEP 1: Rate-of-change estimation ---\n")

# Key metrics at each snapshot (SWE only)
metrics_df = con.execute(f"""
SELECT
    period,
    COUNT(*) as n,
    -- Entry share (among non-unknown seniority)
    SUM(CASE WHEN seniority_3level = 'junior' THEN 1 ELSE 0 END)*1.0 /
        NULLIF(SUM(CASE WHEN seniority_3level != 'unknown' THEN 1 ELSE 0 END), 0) as entry_share,
    -- AI prevalence
    SUM(CASE WHEN {AI_BROAD} THEN 1 ELSE 0 END)*1.0/COUNT(*) as ai_prevalence,
    -- GenAI prevalence
    SUM(CASE WHEN {GENAI} THEN 1 ELSE 0 END)*1.0/COUNT(*) as genai_prevalence,
    -- Description length (median core)
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY core_length) as desc_length_median,
    -- Tech count
    AVG({TECH_COUNT_EXPR}) as tech_count_avg,
    -- Scope density (cross-functional/stakeholder)
    SUM(CASE WHEN {SCOPE_COLLAB} THEN 1 ELSE 0 END)*1.0/COUNT(*) as scope_density
FROM '{DATA}'
WHERE {BASE_FILTER} AND is_swe = true
GROUP BY period ORDER BY period
""").fetchdf()

print("Snapshot values (SWE):")
print(metrics_df.to_string(index=False))
print()

# Compute annualized rates
# Timestamps: asaniczka=Jan 2024 (2024.0), arshkon=Apr 2024 (2024.25), scraped=Mar 2026 (2026.25)
time_map = {'2024-01': 2024.0, '2024-04': 2024.25, '2026-03': 2026.25}

rate_rows = []
metric_cols = ['entry_share', 'ai_prevalence', 'genai_prevalence',
               'desc_length_median', 'tech_count_avg', 'scope_density']

for col in metric_cols:
    vals = {}
    for _, r in metrics_df.iterrows():
        vals[r['period']] = r[col]

    # Within-2024 rate (Jan -> Apr, 0.25 years)
    if '2024-01' in vals and '2024-04' in vals:
        w24_change = vals['2024-04'] - vals['2024-01']
        w24_annualized = w24_change / 0.25
    else:
        w24_change = np.nan
        w24_annualized = np.nan

    # Cross-period rate (Apr 2024 -> Mar 2026, 2.0 years)
    if '2024-04' in vals and '2026-03' in vals:
        cross_change = vals['2026-03'] - vals['2024-04']
        cross_annualized = cross_change / 2.0
    else:
        cross_change = np.nan
        cross_annualized = np.nan

    # Full-span rate (Jan 2024 -> Mar 2026, 2.25 years)
    if '2024-01' in vals and '2026-03' in vals:
        full_change = vals['2026-03'] - vals['2024-01']
        full_annualized = full_change / 2.25
    else:
        full_change = np.nan
        full_annualized = np.nan

    # Acceleration ratio
    if w24_annualized and w24_annualized != 0 and not np.isnan(w24_annualized):
        accel_ratio = cross_annualized / w24_annualized
    else:
        accel_ratio = np.nan

    rate_rows.append({
        'metric': col,
        'val_2024_01': vals.get('2024-01'),
        'val_2024_04': vals.get('2024-04'),
        'val_2026_03': vals.get('2026-03'),
        'within_2024_change': w24_change,
        'within_2024_annualized': w24_annualized,
        'cross_period_change': cross_change,
        'cross_period_annualized': cross_annualized,
        'full_span_annualized': full_annualized,
        'acceleration_ratio': accel_ratio,
    })

rate_df = pd.DataFrame(rate_rows)
rate_df.to_csv(TBL_DIR / 'rate_of_change.csv', index=False)

print("Rate-of-change table:")
for _, r in rate_df.iterrows():
    print(f"\n  {r['metric']}:")
    print(f"    Jan 2024: {r['val_2024_01']:.4f}  Apr 2024: {r['val_2024_04']:.4f}  Mar 2026: {r['val_2026_03']:.4f}")
    print(f"    Within-2024 annualized: {r['within_2024_annualized']:+.4f}")
    print(f"    Cross-period annualized: {r['cross_period_annualized']:+.4f}")
    accel = r['acceleration_ratio']
    accel_str = f"{accel:.2f}" if not np.isnan(accel) else "N/A (zero denominator)"
    print(f"    Acceleration ratio: {accel_str}")


# ─────────────────────────────────────────────
# STEP 2: Within-arshkon stability
# ─────────────────────────────────────────────

print("\n\n--- STEP 2: Within-arshkon stability ---\n")

# Arshkon has date_posted for most rows
arshkon_weekly = con.execute(f"""
SELECT
    DATE_TRUNC('week', TRY_CAST(date_posted AS DATE)) as post_week,
    COUNT(*) as n,
    SUM(CASE WHEN {AI_BROAD} THEN 1 ELSE 0 END)*1.0/COUNT(*) as ai_share,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY core_length) as med_core_len,
    AVG({TECH_COUNT_EXPR}) as avg_tech,
    SUM(CASE WHEN seniority_3level = 'junior' THEN 1 ELSE 0 END)*1.0 /
        NULLIF(SUM(CASE WHEN seniority_3level != 'unknown' THEN 1 ELSE 0 END), 0) as entry_share
FROM '{DATA}'
WHERE {BASE_FILTER} AND is_swe = true AND source = 'kaggle_arshkon'
  AND date_posted IS NOT NULL AND TRY_CAST(date_posted AS DATE) IS NOT NULL
GROUP BY post_week
HAVING COUNT(*) >= 20
ORDER BY post_week
""").fetchdf()

print(f"Arshkon weekly variation (SWE, {len(arshkon_weekly)} weeks with n>=20):")
print(arshkon_weekly.to_string(index=False))

if len(arshkon_weekly) > 1:
    for col in ['ai_share', 'med_core_len', 'avg_tech', 'entry_share']:
        vals = arshkon_weekly[col].dropna()
        if len(vals) > 1:
            cv = vals.std() / vals.mean() if vals.mean() != 0 else np.nan
            print(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, CV={cv:.3f}")

arshkon_weekly.to_csv(TBL_DIR / 'arshkon_weekly_stability.csv', index=False)


# ─────────────────────────────────────────────
# STEP 3: Scraper yield characterization
# ─────────────────────────────────────────────

print("\n\n--- STEP 3: Scraper yield characterization ---\n")

scraper_daily = con.execute(f"""
SELECT
    scrape_date,
    COUNT(*) as all_rows,
    SUM(CASE WHEN is_swe THEN 1 ELSE 0 END) as swe_rows,
    SUM(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) as adj_rows,
    SUM(CASE WHEN is_control THEN 1 ELSE 0 END) as ctrl_rows
FROM '{DATA}'
WHERE {BASE_FILTER} AND source = 'scraped' AND scrape_date IS NOT NULL
GROUP BY scrape_date
ORDER BY scrape_date
""").fetchdf()

print(f"Daily scrape yield ({len(scraper_daily)} days):")
print(scraper_daily.to_string(index=False))

if len(scraper_daily) > 3:
    # First day vs rest
    first_day = scraper_daily.iloc[0]
    rest = scraper_daily.iloc[1:]
    print(f"\n  First day ({first_day['scrape_date']}): {first_day['swe_rows']:,} SWE, {first_day['all_rows']:,} total")
    print(f"  Steady-state mean: {rest['swe_rows'].mean():,.0f} SWE/day, {rest['all_rows'].mean():,.0f} total/day")
    print(f"  First day / steady-state ratio (SWE): {first_day['swe_rows']/rest['swe_rows'].mean():.1f}x")

scraper_daily.to_csv(TBL_DIR / 'scraper_daily_yield.csv', index=False)


# ─────────────────────────────────────────────
# STEP 4: Within-March stability
# ─────────────────────────────────────────────

print("\n\n--- STEP 4: Within-March stability ---\n")

march_daily = con.execute(f"""
SELECT
    scrape_date,
    COUNT(*) as n,
    SUM(CASE WHEN {AI_BROAD} THEN 1 ELSE 0 END)*1.0/COUNT(*) as ai_share,
    SUM(CASE WHEN {GENAI} THEN 1 ELSE 0 END)*1.0/COUNT(*) as genai_share,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY core_length) as med_core_len,
    AVG({TECH_COUNT_EXPR}) as avg_tech,
    SUM(CASE WHEN seniority_3level = 'junior' THEN 1 ELSE 0 END)*1.0 /
        NULLIF(SUM(CASE WHEN seniority_3level != 'unknown' THEN 1 ELSE 0 END), 0) as entry_share,
    SUM(CASE WHEN {SCOPE_COLLAB} THEN 1 ELSE 0 END)*1.0/COUNT(*) as collab_share
FROM '{DATA}'
WHERE {BASE_FILTER} AND is_swe = true AND source = 'scraped' AND scrape_date IS NOT NULL
GROUP BY scrape_date
HAVING COUNT(*) >= 50
ORDER BY scrape_date
""").fetchdf()

print(f"Within-March daily stability (SWE, {len(march_daily)} days):")
print(march_daily.to_string(index=False))

if len(march_daily) > 1:
    print("\nCoefficient of Variation:")
    for col in ['ai_share', 'genai_share', 'med_core_len', 'avg_tech', 'entry_share', 'collab_share']:
        vals = march_daily[col].dropna()
        if len(vals) > 1 and vals.mean() != 0:
            cv = vals.std() / vals.mean()
            print(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, CV={cv:.3f}")

    # Day-of-week effects
    march_daily['scrape_date'] = pd.to_datetime(march_daily['scrape_date'])
    march_daily['dow'] = march_daily['scrape_date'].dt.day_name()

    print("\nDay-of-week effects:")
    dow_stats = march_daily.groupby('dow')[['n', 'ai_share', 'med_core_len']].mean()
    for dow in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        if dow in dow_stats.index:
            r = dow_stats.loc[dow]
            print(f"  {dow:12s}  n={r['n']:,.0f}  AI={r['ai_share']:.3f}  core={r['med_core_len']:,.0f}")

march_daily.to_csv(TBL_DIR / 'march_daily_stability.csv', index=False)


# ─────────────────────────────────────────────
# STEP 5: Timeline contextualization
# ─────────────────────────────────────────────

print("\n\n--- STEP 5: Timeline contextualization ---\n")

# Key AI releases
ai_releases = [
    ('2024-01', 'Data snapshot: Asaniczka (Jan 2024)\nContext: GPT-4 (Mar 2023), Claude 2 (Jul 2023)'),
    ('2024-04', 'Data snapshot: Arshkon (Apr 2024)\nContext: GPT-4o (May 2024 imminent), Gemini 1.5 Pro (Feb 2024)'),
    ('2024-05', 'GPT-4o released'),
    ('2024-06', 'Claude 3.5 Sonnet released'),
    ('2024-09', 'o1 released'),
    ('2024-12', 'DeepSeek V3 released'),
    ('2025-02', 'Claude 3.5 Sonnet refresh, GPT-o3-mini'),
    ('2025-05', 'Claude 4 Opus released'),
    ('2025-09', 'Claude 4 Sonnet released'),
    ('2025-12', 'Gemini 2.0 Flash'),
    ('2026-01', 'Claude Opus 4.5 released'),
    ('2026-03', 'Data snapshot: Scraped (Mar 2026)\nContext: Claude Opus 4.5, Gemini 2.5 Pro, DeepSeek R2'),
]

print("Timeline of AI releases between snapshots:")
for date, event in ai_releases:
    print(f"  {date}: {event}")


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

print("\n--- Generating figures ---\n")

# FIGURE 1: Rate-of-change bar chart
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('T19: Annualized Rate of Change (SWE Postings)', fontsize=14, fontweight='bold')

# Normalize to percentage of baseline for comparability
rate_pct = rate_df.copy()
for _, r in rate_pct.iterrows():
    base = r['val_2024_01'] if r['val_2024_01'] != 0 else r['val_2024_04']
    if base and base != 0:
        rate_pct.loc[rate_pct['metric'] == r['metric'], 'w24_pct'] = r['within_2024_annualized'] / base * 100
        rate_pct.loc[rate_pct['metric'] == r['metric'], 'cross_pct'] = r['cross_period_annualized'] / base * 100

metrics_labels = {
    'entry_share': 'Entry share',
    'ai_prevalence': 'AI prevalence',
    'genai_prevalence': 'GenAI prevalence',
    'desc_length_median': 'Desc length',
    'tech_count_avg': 'Tech count',
    'scope_density': 'Scope density'
}

x = np.arange(len(rate_pct))
width = 0.35
bars1 = ax.bar(x - width/2, rate_pct['w24_pct'], width, label='Within-2024 (annualized)', color='#2196F3', alpha=0.8)
bars2 = ax.bar(x + width/2, rate_pct['cross_pct'], width, label='Cross-period (annualized)', color='#FF5722', alpha=0.8)

ax.set_ylabel('Annualized change (% of baseline)')
ax.set_xticks(x)
ax.set_xticklabels([metrics_labels.get(m, m) for m in rate_pct['metric']], rotation=20, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(FIG_DIR / 'rate_of_change.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved rate_of_change.png")

# FIGURE 2: Scraper yield
if len(scraper_daily) > 1:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('T19: Scraper Daily Yield', fontsize=14, fontweight='bold')

    dates = pd.to_datetime(scraper_daily['scrape_date'])
    axes[0].bar(dates, scraper_daily['swe_rows'], color='#2196F3', alpha=0.8, label='SWE')
    axes[0].set_ylabel('SWE postings/day')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(dates, scraper_daily['all_rows'], color='#4CAF50', alpha=0.8, label='All')
    axes[1].set_ylabel('All postings/day')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'scraper_daily_yield.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved scraper_daily_yield.png")

# FIGURE 3: Within-March stability
if len(march_daily) > 1:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('T19: Within-March Stability (SWE Postings)', fontsize=14, fontweight='bold')

    dates = pd.to_datetime(march_daily['scrape_date'])

    ax = axes[0, 0]
    ax.plot(dates, march_daily['ai_share'], 'o-', color='#2196F3', markersize=4)
    ax.set_title('AI Prevalence')
    ax.set_ylabel('Share')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(dates, march_daily['med_core_len'], 'o-', color='#FF9800', markersize=4)
    ax.set_title('Median Description Length')
    ax.set_ylabel('Characters')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(dates, march_daily['avg_tech'], 'o-', color='#4CAF50', markersize=4)
    ax.set_title('Avg Tech Mentions')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(dates, march_daily['entry_share'], 'o-', color='#9C27B0', markersize=4)
    ax.set_title('Entry Share (known seniority)')
    ax.set_ylabel('Share')
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'within_march_stability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved within_march_stability.png")

# FIGURE 4: Timeline with AI releases
fig, ax = plt.subplots(figsize=(14, 4))
fig.suptitle('T19: Data Snapshots and AI Release Timeline', fontsize=13, fontweight='bold')

# Data snapshots
snapshot_dates = ['2024-01-15', '2024-04-15', '2026-03-15']
snapshot_labels = ['Asaniczka\n(Jan 2024)', 'Arshkon\n(Apr 2024)', 'Scraped\n(Mar 2026)']
snapshot_x = [pd.Timestamp(d) for d in snapshot_dates]
ax.scatter(snapshot_x, [0]*3, s=200, c='#2196F3', zorder=5, marker='D')
for x, label in zip(snapshot_x, snapshot_labels):
    ax.annotate(label, (x, 0), textcoords="offset points", xytext=(0, 30),
                ha='center', fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2196F3'))

# AI releases
ai_events = [
    ('2024-05-01', 'GPT-4o'),
    ('2024-06-01', 'Claude 3.5\nSonnet'),
    ('2024-09-01', 'o1'),
    ('2024-12-01', 'DeepSeek V3'),
    ('2025-05-01', 'Claude 4\nOpus'),
    ('2025-09-01', 'Claude 4\nSonnet'),
    ('2026-01-01', 'Claude\nOpus 4.5'),
    ('2026-03-01', 'Gemini\n2.5 Pro'),
]

for i, (date, label) in enumerate(ai_events):
    x = pd.Timestamp(date)
    y_offset = -30 if i % 2 == 0 else -50
    ax.scatter([x], [0], s=60, c='#FF5722', zorder=4, marker='|')
    ax.annotate(label, (x, 0), textcoords="offset points", xytext=(0, y_offset),
                ha='center', fontsize=7, color='#FF5722',
                arrowprops=dict(arrowstyle='->', color='#FF5722', alpha=0.5))

ax.axhline(y=0, color='gray', linewidth=0.5)
ax.set_xlim(pd.Timestamp('2023-10-01'), pd.Timestamp('2026-06-01'))
ax.set_ylim(-0.15, 0.1)
ax.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig(FIG_DIR / 'timeline_contextualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved timeline_contextualization.png")

print("\nT19 analysis complete.")
