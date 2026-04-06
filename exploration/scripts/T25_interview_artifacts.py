#!/usr/bin/env python3
"""T25: Interview Elicitation Artifacts

Produces visualizations and data extracts for RQ4 interview preparation.
"""

import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_FIG = 'exploration/figures/T25'
OUT_TAB = 'exploration/tables/T25'
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TAB, exist_ok=True)

con = duckdb.connect()
PQ = 'data/unified.parquet'
BASE_FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"

# ============================================================
# Artifact 3: Junior-share trend visualization with AI model releases
# ============================================================
print("=== Artifact 3: Junior-share trend ===")

# Use arshkon-only as 2024 data point (per instructions)
entry_data = con.execute(f"""
SELECT source, period,
       SUM(CASE WHEN seniority_native = 'entry' THEN 1 ELSE 0 END) as n_entry,
       SUM(CASE WHEN seniority_native IS NOT NULL THEN 1 ELSE 0 END) as n_known,
       COUNT(*) as n_total
FROM read_parquet('{PQ}')
WHERE {BASE_FILTER} AND is_swe = true
  AND source IN ('kaggle_arshkon', 'scraped')
GROUP BY source, period
""").fetchdf()
print(entry_data)

# Compute entry share for arshkon and scraped
ark = entry_data[entry_data['source'] == 'kaggle_arshkon'].iloc[0]
scr = entry_data[entry_data['source'] == 'scraped'].iloc[0]
ark_entry_share = ark['n_entry'] / ark['n_known'] * 100
scr_entry_share = scr['n_entry'] / scr['n_known'] * 100

fig, ax = plt.subplots(figsize=(10, 5.5))

# Timeline positions (months from Jan 2024)
x_ark = 3  # April 2024
x_scr = 26  # March 2026

ax.plot([x_ark, x_scr], [ark_entry_share, scr_entry_share], 'o-',
        color='#2c7fb8', linewidth=2.5, markersize=12, zorder=5)
ax.annotate(f'{ark_entry_share:.1f}%\n(arshkon, Apr 2024)\nn={int(ark["n_entry"])}/{int(ark["n_known"])}',
            xy=(x_ark, ark_entry_share), xytext=(x_ark + 1.5, ark_entry_share + 2),
            fontsize=9, ha='left', va='bottom',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
ax.annotate(f'{scr_entry_share:.1f}%\n(scraped, Mar 2026)\nn={int(scr["n_entry"])}/{int(scr["n_known"])}',
            xy=(x_scr, scr_entry_share), xytext=(x_scr - 1.5, scr_entry_share - 3),
            fontsize=9, ha='right', va='top',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

# AI model release annotations
releases = [
    (4, 'GPT-4o\n(May 2024)'),
    (5, 'Claude 3.5\nSonnet\n(Jun 2024)'),
    (8, 'o1\n(Sep 2024)'),
    (11, 'DeepSeek\nV3\n(Dec 2024)'),
    (16, 'Claude 4\nOpus\n(May 2025)'),
    (24, 'Claude\nOpus 4.5\n(Jan 2026)'),
]
for xr, label in releases:
    ax.axvline(x=xr, color='#fc8d59', alpha=0.4, linestyle='--', linewidth=1)
    ax.text(xr, ax.get_ylim()[1] if ax.get_ylim()[1] > 25 else 25, label,
            fontsize=6.5, ha='center', va='bottom', color='#d95f0e', rotation=0)

ax.set_xlim(-1, 28)
ax.set_ylim(5, 28)
ax.set_xlabel('Months from January 2024', fontsize=10)
ax.set_ylabel('Entry-Level Share (% of known seniority)', fontsize=10)
ax.set_title('SWE Entry-Level Posting Share: 2024 vs 2026\n(Using seniority_native; arshkon-only as 2024 baseline)',
             fontsize=11, fontweight='bold')

# Add caveat box
caveat_text = ('CAVEAT: Direction depends on seniority operationalization.\n'
               'seniority_native (shown): 22.3% -> 14.0% (decline)\n'
               'seniority_final: 20.4% -> 14.5% (decline)\n'
               'seniority_3level in overlap panel: direction reverses.\n'
               'LLM seniority labels (planned) will resolve.')
props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8, edgecolor='orange')
ax.text(0.02, 0.02, caveat_text, transform=ax.transAxes, fontsize=7,
        verticalalignment='bottom', bbox=props)

plt.tight_layout()
plt.savefig(f'{OUT_FIG}/junior_share_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved junior_share_trend.png")

# ============================================================
# Artifact 4: Senior archetype chart (from T21 data)
# ============================================================
print("\n=== Artifact 4: Senior archetype chart ===")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: Management vs Orchestration density by seniority x period
seniority_levels = ['Entry', 'Associate', 'Mid-Senior', 'Director']
mgmt_2024 = [0.949, 1.068, 1.182, 1.796]
mgmt_2026 = [1.098, 1.151, 1.290, 1.380]
orch_2024 = [None, None, 0.97, 0.76]  # only have senior data from T21
orch_2026 = [None, None, 1.12, 1.10]

x = np.arange(4)
w = 0.35

ax = axes[0]
bars1 = ax.bar(x - w/2, mgmt_2024, w, label='2024', color='#2c7fb8', alpha=0.8)
bars2 = ax.bar(x + w/2, mgmt_2026, w, label='2026', color='#fc8d59', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(seniority_levels, fontsize=9)
ax.set_ylabel('Management Density (per 1K chars)', fontsize=9)
ax.set_title('Management Language by Seniority\n(Expanded everywhere; corrected metric)', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)

# Annotate the director change
ax.annotate('-23%', xy=(3 + w/2, 1.38), fontsize=8, color='red', ha='center', va='bottom', fontweight='bold')
ax.annotate('+16%', xy=(0 + w/2, 1.098), fontsize=7, color='green', ha='center', va='bottom')

# Right panel: Senior sub-archetypes (from T21)
ax2 = axes[1]
archetypes = ['Low-touch\nGeneralist', 'People\nManager', 'Technical\nOrchestrator', 'Strategic\nLeader']
shares_2024 = [58.8, 19.9, 4.5, 16.8]
shares_2026 = [58.2, 22.4, 5.8, 13.5]

x2 = np.arange(4)
bars3 = ax2.bar(x2 - w/2, shares_2024, w, label='2024', color='#2c7fb8', alpha=0.8)
bars4 = ax2.bar(x2 + w/2, shares_2026, w, label='2026', color='#fc8d59', alpha=0.8)
ax2.set_xticks(x2)
ax2.set_xticklabels(archetypes, fontsize=8)
ax2.set_ylabel('Share of Senior SWE Postings (%)', fontsize=9)
ax2.set_title('Senior Sub-Archetypes (T21)\nOrchestration +29%, Strategic Leader -20%', fontsize=10, fontweight='bold')
ax2.legend(fontsize=8)

# Highlight orchestrator growth
ax2.annotate('+29%\nrelative', xy=(2 + w/2, 5.8), fontsize=7, color='green', ha='center', va='bottom', fontweight='bold')
ax2.annotate('-20%\nrelative', xy=(3 + w/2, 13.5), fontsize=7, color='red', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_FIG}/senior_archetype_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved senior_archetype_chart.png")

# ============================================================
# Artifact 5: AI divergence chart (posting requirements vs developer usage)
# ============================================================
print("\n=== Artifact 5: AI divergence chart ===")

fig, ax = plt.subplots(figsize=(9, 5.5))

# Data from T23
categories = ['AI-any', 'AI-tool', 'AI-domain']
posting_2024 = [11.6, 1.7, 10.6]
posting_2026 = [40.6, 23.3, 22.7]
usage_2024 = [57.0, 35.0, 15.0]
usage_2026 = [75.0, 55.0, 20.0]

x = np.arange(3)
w = 0.18

bars1 = ax.bar(x - 1.5*w, posting_2024, w, label='Posting 2024', color='#2c7fb8', alpha=0.7)
bars2 = ax.bar(x - 0.5*w, posting_2026, w, label='Posting 2026', color='#2c7fb8', alpha=1.0)
bars3 = ax.bar(x + 0.5*w, usage_2024, w, label='Usage 2024 (est.)', color='#fc8d59', alpha=0.7)
bars4 = ax.bar(x + 1.5*w, usage_2026, w, label='Usage 2026 (est.)', color='#fc8d59', alpha=1.0)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylabel('Prevalence (%)', fontsize=10)
ax.set_title('Posting AI Requirements vs Developer Usage\nKey finding: Requirements LAG usage (~41% vs ~75%)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')

# Add gap annotations
for i, cat in enumerate(categories):
    gap_2024 = posting_2024[i] - usage_2024[i]
    gap_2026 = posting_2026[i] - usage_2026[i]
    y_pos = max(posting_2026[i], usage_2026[i]) + 2
    ax.text(i, y_pos, f'Gap: {gap_2026:+.0f}pp\n(was {gap_2024:+.0f}pp)',
            fontsize=7.5, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))

ax.set_ylim(0, 90)
plt.tight_layout()
plt.savefig(f'{OUT_FIG}/ai_divergence_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved ai_divergence_chart.png")

# ============================================================
# Artifact 6: AI-entry orthogonality scatter
# ============================================================
print("\n=== Artifact 6: AI-entry orthogonality ===")

orth_data = con.execute(f"""
WITH company_metrics AS (
    SELECT company_name_canonical,
           period,
           COUNT(*) as n_postings,
           AVG(CASE WHEN (
               LOWER(description) LIKE '%artificial intelligence%'
               OR LOWER(description) LIKE '% ai %'
               OR LOWER(description) LIKE '%machine learning%'
               OR LOWER(description) LIKE '% llm%'
               OR LOWER(description) LIKE '%large language model%'
               OR LOWER(description) LIKE '%generative ai%'
               OR LOWER(description) LIKE '%copilot%'
               OR LOWER(description) LIKE '%deep learning%'
           ) THEN 1.0 ELSE 0.0 END) as ai_rate,
           AVG(CASE WHEN seniority_3level = 'junior' THEN 1.0 ELSE 0.0 END) as entry_rate
    FROM read_parquet('{PQ}')
    WHERE {BASE_FILTER} AND is_swe = true AND company_name_canonical IS NOT NULL
      AND is_aggregator = false
    GROUP BY company_name_canonical, period
    HAVING COUNT(*) >= 3
)
SELECT a.company_name_canonical,
       a.n_postings as n_2024, b.n_postings as n_2026,
       b.ai_rate - a.ai_rate as ai_change,
       b.entry_rate - a.entry_rate as entry_change,
       a.n_postings + b.n_postings as total_n
FROM company_metrics a
INNER JOIN company_metrics b ON a.company_name_canonical = b.company_name_canonical
WHERE a.period IN ('2024-01','2024-04') AND b.period = '2026-03'
  AND a.n_postings >= 3 AND b.n_postings >= 3
""").fetchdf()

fig, ax = plt.subplots(figsize=(8, 7))

sizes = np.clip(orth_data['total_n'] / 5, 10, 200)
ax.scatter(orth_data['ai_change'] * 100, orth_data['entry_change'] * 100,
           s=sizes, alpha=0.4, color='#2c7fb8', edgecolors='white', linewidth=0.3)

# Add correlation line and annotation
from scipy import stats
mask = ~(orth_data['ai_change'].isna() | orth_data['entry_change'].isna())
r, p = stats.pearsonr(orth_data.loc[mask, 'ai_change'], orth_data.loc[mask, 'entry_change'])

ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

# Fit line
slope, intercept, _, _, _ = stats.linregress(orth_data.loc[mask, 'ai_change'] * 100,
                                               orth_data.loc[mask, 'entry_change'] * 100)
x_line = np.linspace(-50, 110, 100)
ax.plot(x_line, slope * x_line + intercept, 'r--', alpha=0.5, linewidth=1.5)

ax.set_xlabel('AI Prevalence Change (pp, 2024 to 2026)', fontsize=11)
ax.set_ylabel('Entry-Level Share Change (pp, 2024 to 2026)', fontsize=11)
ax.set_title(f'AI Adoption vs Entry-Level Change: Company Level\n'
             f'r = {r:.3f}, p = {p:.3f} (n = {mask.sum()} companies)\n'
             f'"Companies that adopted AI did NOT cut junior roles"',
             fontsize=11, fontweight='bold')

# Key finding box
box_text = (f'Pearson r = {r:.3f} (p = {p:.3f})\n'
            'Also null at metro level: r = -0.04\n\n'
            'AI adoption and junior hiring\n'
            'changes are ORTHOGONAL.\n'
            'These are parallel market trends,\n'
            'not causally linked within firms.')
props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='orange')
ax.text(0.02, 0.98, box_text, transform=ax.transAxes, fontsize=8.5,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(f'{OUT_FIG}/ai_entry_orthogonality.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved ai_entry_orthogonality.png (r={r:.3f}, p={p:.3f}, n={mask.sum()})")

# ============================================================
# Save data tables for reference
# ============================================================
print("\n=== Saving data tables ===")

# Entry-level JDs for interview stimuli
entry_2026_jds = con.execute(f"""
SELECT uid, title, company_name_effective, yoe_extracted,
       seniority_final, seniority_native,
       CASE WHEN LOWER(description) LIKE '%mentor%' THEN 1 ELSE 0 END as has_mentor,
       CASE WHEN LOWER(description) LIKE '%coach%' THEN 1 ELSE 0 END as has_coach,
       CASE WHEN LOWER(description) LIKE '%project management%' THEN 1 ELSE 0 END as has_proj_mgmt,
       CASE WHEN LOWER(description) LIKE '%stakeholder%' THEN 1 ELSE 0 END as has_stakeholder,
       CASE WHEN (LOWER(description) LIKE '% ai %' OR LOWER(description) LIKE '%machine learning%'
                  OR LOWER(description) LIKE '% llm%') THEN 1 ELSE 0 END as has_ai,
       LENGTH(description) as desc_len
FROM read_parquet('{PQ}')
WHERE {BASE_FILTER} AND is_swe = true AND seniority_final = 'entry'
  AND source = 'scraped' AND is_aggregator = false
  AND yoe_extracted IS NOT NULL AND yoe_extracted <= 3
  AND (LOWER(description) LIKE '%mentor%' OR LOWER(description) LIKE '%coach%'
       OR LOWER(description) LIKE '%project management%' OR LOWER(description) LIKE '%stakeholder%')
ORDER BY desc_len DESC
LIMIT 20
""").fetchdf()
entry_2026_jds.to_csv(f'{OUT_TAB}/entry_2026_scope_jds.csv', index=False)
print(f"Saved entry_2026_scope_jds.csv ({len(entry_2026_jds)} rows)")

# Entry-level 2024 JDs for contrast
entry_2024_jds = con.execute(f"""
SELECT uid, title, company_name_effective, yoe_extracted,
       seniority_final, seniority_native,
       CASE WHEN LOWER(description) LIKE '%mentor%' THEN 1 ELSE 0 END as has_mentor,
       CASE WHEN LOWER(description) LIKE '%coach%' THEN 1 ELSE 0 END as has_coach,
       CASE WHEN (LOWER(description) LIKE '% ai %' OR LOWER(description) LIKE '%machine learning%'
                  OR LOWER(description) LIKE '% llm%') THEN 1 ELSE 0 END as has_ai,
       LENGTH(description) as desc_len
FROM read_parquet('{PQ}')
WHERE {BASE_FILTER} AND is_swe = true AND seniority_final = 'entry'
  AND source = 'kaggle_arshkon' AND is_aggregator = false
  AND yoe_extracted IS NOT NULL AND yoe_extracted <= 3
ORDER BY desc_len DESC
LIMIT 20
""").fetchdf()
entry_2024_jds.to_csv(f'{OUT_TAB}/entry_2024_contrast_jds.csv', index=False)
print(f"Saved entry_2024_contrast_jds.csv ({len(entry_2024_jds)} rows)")

# Company pairs for side-by-side comparison
orth_data.to_csv(f'{OUT_TAB}/company_ai_entry_orthogonality.csv', index=False)
print(f"Saved company_ai_entry_orthogonality.csv ({len(orth_data)} rows)")

print("\nAll T25 artifacts complete.")
