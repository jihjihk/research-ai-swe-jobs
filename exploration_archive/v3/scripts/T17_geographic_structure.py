#!/usr/bin/env python3
"""T17: Geographic market structure — metro-level analysis of SWE market changes."""

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os, warnings
warnings.filterwarnings('ignore')

OUT_FIG = 'exploration/figures/T17'
OUT_TBL = 'exploration/tables/T17'
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TBL, exist_ok=True)

con = duckdb.connect()
PARQUET = 'data/unified.parquet'

# ========== Load data ==========
print("=== Loading SWE data ===")
base = con.execute(f"""
SELECT uid, source, period, metro_area, company_name_canonical, is_aggregator,
       seniority_final, seniority_3level,
       description_core, description_length, core_length,
       is_remote_inferred, is_remote
FROM read_parquet('{PARQUET}')
WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true
""").df()
print(f"SWE base: {len(base)} rows")

base['era'] = base['period'].map({'2024-01': '2024', '2024-04': '2024', '2026-03': '2026'})

# Tech matrix
tech = con.execute("""SELECT * FROM read_parquet('exploration/artifacts/shared/swe_tech_matrix.parquet')""").df()
tech_cols = [c for c in tech.columns if c != 'uid']
ai_cols = [c for c in tech_cols if c.startswith('ai_')]

base = base.merge(tech[['uid'] + tech_cols], on='uid', how='left')
base['has_ai'] = base[ai_cols].any(axis=1).astype(int)
base['is_entry'] = (base['seniority_3level'] == 'junior').astype(int)
base['desc_len'] = base['core_length'].fillna(base['description_length'])

# Org scope terms
org_scope_pattern = r'(?i)\b(manage|management|lead|leadership|stakeholder|cross-functional|strategic|mentor|coordinate|oversee|collaborate)\b'
base['org_scope_count'] = base['description_core'].fillna('').str.count(org_scope_pattern)

# Tech diversity = number of unique tech columns true
base['tech_count'] = base[tech_cols].sum(axis=1)

print(f"Metro area coverage: {base['metro_area'].notna().mean():.1%}")
print(f"Unique metros: {base['metro_area'].nunique()}")

# ========== STEP 1: Metro-level metrics ==========
print("\n=== Step 1: Metro-level metrics ===")

# Filter to known metros
metro_data = base[base['metro_area'].notna()].copy()

# Count by metro and era
metro_era_n = metro_data.groupby(['metro_area', 'era']).size().unstack(fill_value=0)
metro_era_n.columns = ['n_2024', 'n_2026']

# Filter to metros with >= 50 SWE per era
qual_metros = metro_era_n[(metro_era_n['n_2024'] >= 50) & (metro_era_n['n_2026'] >= 50)]
print(f"Metros with >=50 SWE per era: {len(qual_metros)}")
print(f"Total postings in qualifying metros: {qual_metros.sum().sum()}")

# Calculate per-metro-era metrics
def metro_metrics(group):
    return pd.Series({
        'n': len(group),
        'entry_share': group['is_entry'].mean(),
        'ai_prevalence': group['has_ai'].mean(),
        'org_scope_composite': group['org_scope_count'].mean(),
        'median_desc_len': group['desc_len'].median(),
        'mean_desc_len': group['desc_len'].mean(),
        'tech_diversity': group['tech_count'].mean(),
        'remote_share': group['is_remote_inferred'].mean() if group['is_remote_inferred'].notna().any() else np.nan,
    })

qual_metro_names = set(qual_metros.index)
metro_filtered = metro_data[metro_data['metro_area'].isin(qual_metro_names)]

metro_metrics_df = metro_filtered.groupby(['metro_area', 'era']).apply(metro_metrics).unstack('era')
metro_metrics_df.columns = [f'{m}_{e}' for m, e in metro_metrics_df.columns]

# Save raw metro metrics
metro_metrics_df.to_csv(f'{OUT_TBL}/metro_metrics_by_era.csv')

# Compute changes
metrics_list = ['entry_share', 'ai_prevalence', 'org_scope_composite', 'median_desc_len', 'tech_diversity']
metro_change = pd.DataFrame(index=metro_metrics_df.index)
for m in metrics_list:
    metro_change[f'd_{m}'] = metro_metrics_df[f'{m}_2026'] - metro_metrics_df[f'{m}_2024']
    metro_change[f'{m}_2024'] = metro_metrics_df[f'{m}_2024']
    metro_change[f'{m}_2026'] = metro_metrics_df[f'{m}_2026']

metro_change['n_2024'] = metro_metrics_df['n_2024']
metro_change['n_2026'] = metro_metrics_df['n_2026']
metro_change['remote_share_2026'] = metro_metrics_df.get('remote_share_2026', np.nan)

print("\nMetro change summary:")
for c in [cc for cc in metro_change.columns if cc.startswith('d_')]:
    print(f"  {c}: mean={metro_change[c].mean():.4f}, std={metro_change[c].std():.4f}, "
          f"min={metro_change[c].min():.4f}, max={metro_change[c].max():.4f}")

# ========== STEP 2: Rank metros by magnitude of change ==========
print("\n=== Step 2: Metro rankings ===")
for metric in metrics_list:
    col = f'd_{metric}'
    print(f"\nTop 5 metros by {metric} increase:")
    top = metro_change[col].sort_values(ascending=False).head(5)
    for m, v in top.items():
        print(f"  {m}: {v:.4f} ({metro_change.loc[m, f'{metric}_2024']:.4f} -> {metro_change.loc[m, f'{metric}_2026']:.4f})")

    print(f"Bottom 5 metros by {metric} change:")
    bot = metro_change[col].sort_values().head(5)
    for m, v in bot.items():
        print(f"  {m}: {v:.4f} ({metro_change.loc[m, f'{metric}_2024']:.4f} -> {metro_change.loc[m, f'{metric}_2026']:.4f})")

metro_change.to_csv(f'{OUT_TBL}/metro_change_ranked.csv')

# ========== STEP 3: Geographic patterns — tech hub vs non-hub ==========
print("\n=== Step 3: Geographic patterns ===")
tech_hubs = ['San Francisco Bay Area', 'Seattle Metro', 'San Jose Metro', 'Austin Metro',
             'Boston Metro', 'New York City Metro']
# Check which are present
available_hubs = [h for h in tech_hubs if h in metro_change.index]
non_hubs = [m for m in metro_change.index if m not in tech_hubs]

print(f"Tech hubs in data: {available_hubs}")
print(f"Non-hub metros: {len(non_hubs)}")

# Compare hub vs non-hub changes
hub_changes = metro_change.loc[metro_change.index.isin(tech_hubs)]
nonhub_changes = metro_change.loc[~metro_change.index.isin(tech_hubs)]

print("\nHub vs non-hub change comparison:")
for metric in metrics_list:
    col = f'd_{metric}'
    hub_mean = hub_changes[col].mean()
    nonhub_mean = nonhub_changes[col].mean()
    t_stat, p_val = stats.ttest_ind(hub_changes[col].dropna(), nonhub_changes[col].dropna())
    print(f"  {metric}: hubs={hub_mean:.4f}, non-hubs={nonhub_mean:.4f}, diff={hub_mean-nonhub_mean:.4f}, t={t_stat:.2f}, p={p_val:.3f}")

# Is change concentrated or uniform?
print("\nEntry decline concentration (coefficient of variation):")
for metric in metrics_list:
    col = f'd_{metric}'
    vals = metro_change[col].dropna()
    cv = vals.std() / abs(vals.mean()) if vals.mean() != 0 else np.nan
    print(f"  {metric}: CV = {cv:.2f}")

# ========== STEP 4: Correlation: AI surge vs entry decline ==========
print("\n=== Step 4: Metro correlations ===")
corr_pairs = [
    ('d_ai_prevalence', 'd_entry_share', 'AI surge vs entry change'),
    ('d_ai_prevalence', 'd_org_scope_composite', 'AI surge vs scope inflation'),
    ('d_ai_prevalence', 'd_tech_diversity', 'AI surge vs tech diversity'),
    ('d_entry_share', 'd_org_scope_composite', 'Entry change vs scope inflation'),
    ('d_ai_prevalence', 'd_median_desc_len', 'AI surge vs description length'),
]

corr_results = []
for x, y, label in corr_pairs:
    valid = metro_change[[x, y]].dropna()
    r, p = stats.pearsonr(valid[x], valid[y])
    rho, p_rho = stats.spearmanr(valid[x], valid[y])
    print(f"  {label}: Pearson r={r:.3f} (p={p:.3f}), Spearman rho={rho:.3f} (p={p_rho:.3f})")
    corr_results.append({'x': x, 'y': y, 'label': label, 'pearson_r': r, 'pearson_p': p,
                          'spearman_rho': rho, 'spearman_p': p_rho, 'n_metros': len(valid)})

pd.DataFrame(corr_results).to_csv(f'{OUT_TBL}/metro_correlations.csv', index=False)

# ========== STEP 5: Remote work by metro (2026 only) ==========
print("\n=== Step 5: Remote work by metro (2026) ===")
remote_2026 = metro_filtered[metro_filtered['era'] == '2026'].groupby('metro_area').agg(
    n=('uid', 'count'),
    remote_share=('is_remote_inferred', 'mean'),
    remote_inferred_n=('is_remote_inferred', 'sum')
).sort_values('remote_share', ascending=False)

print("Remote share by metro (2026):")
for m, row in remote_2026.iterrows():
    print(f"  {m}: {row['remote_share']:.1%} ({int(row['remote_inferred_n'])}/{int(row['n'])})")

remote_2026.to_csv(f'{OUT_TBL}/remote_by_metro_2026.csv')

# ========== STEP 6: Sensitivities ==========
print("\n=== Step 6: Sensitivity — Aggregator exclusion ===")
metro_no_agg = metro_filtered[~metro_filtered['is_aggregator']]
metro_no_agg_metrics = metro_no_agg.groupby(['metro_area', 'era']).apply(metro_metrics).unstack('era')
metro_no_agg_metrics.columns = [f'{m}_{e}' for m, e in metro_no_agg_metrics.columns]

metro_change_na = pd.DataFrame(index=metro_no_agg_metrics.index)
for m in metrics_list:
    metro_change_na[f'd_{m}'] = metro_no_agg_metrics[f'{m}_2026'] - metro_no_agg_metrics[f'{m}_2024']

# Compare sensitivities
print("Sensitivity comparison (aggregator exclusion):")
for metric in metrics_list:
    col = f'd_{metric}'
    base_mean = metro_change[col].mean()
    na_mean = metro_change_na[col].mean()
    pct_change = abs(na_mean - base_mean) / abs(base_mean) * 100 if base_mean != 0 else np.nan
    print(f"  {metric}: base={base_mean:.4f}, no_agg={na_mean:.4f}, pct_change={pct_change:.1f}%")

print("\n=== Sensitivity — Company capping (max 10 per company per metro per era) ===")
# Cap at 10 postings per company per metro per era to avoid single-company dominance
metro_filtered_capped = metro_filtered.copy()
metro_filtered_capped['rn'] = metro_filtered_capped.groupby(
    ['company_name_canonical', 'metro_area', 'era']).cumcount()
metro_capped = metro_filtered_capped[metro_filtered_capped['rn'] < 10].copy()
print(f"Rows before cap: {len(metro_filtered)}, after cap: {len(metro_capped)}")

metro_capped_metrics = metro_capped.groupby(['metro_area', 'era']).apply(metro_metrics).unstack('era')
metro_capped_metrics.columns = [f'{m}_{e}' for m, e in metro_capped_metrics.columns]

metro_change_cap = pd.DataFrame(index=metro_capped_metrics.index)
for m in metrics_list:
    metro_change_cap[f'd_{m}'] = metro_capped_metrics[f'{m}_2026'] - metro_capped_metrics[f'{m}_2024']

print("Sensitivity comparison (company cap=10):")
for metric in metrics_list:
    col = f'd_{metric}'
    base_mean = metro_change[col].mean()
    cap_mean = metro_change_cap[col].mean() if col in metro_change_cap.columns else np.nan
    pct_change = abs(cap_mean - base_mean) / abs(base_mean) * 100 if base_mean != 0 else np.nan
    print(f"  {metric}: base={base_mean:.4f}, capped={cap_mean:.4f}, pct_change={pct_change:.1f}%")

# Rank stability
for metric in metrics_list:
    col = f'd_{metric}'
    if col in metro_change.columns and col in metro_change_cap.columns:
        common_idx = metro_change.index.intersection(metro_change_cap.index)
        base_rank = metro_change.loc[common_idx, col].rank()
        cap_rank = metro_change_cap.loc[common_idx, col].rank()
        rho, p = stats.spearmanr(base_rank, cap_rank)
        print(f"  Rank stability ({metric}): rho={rho:.3f}, p={p:.4f}")

# Save sensitivity comparison
sens_df = pd.DataFrame({
    'metric': metrics_list,
    'base_mean': [metro_change[f'd_{m}'].mean() for m in metrics_list],
    'no_agg_mean': [metro_change_na[f'd_{m}'].mean() for m in metrics_list],
    'capped_mean': [metro_change_cap[f'd_{m}'].mean() if f'd_{m}' in metro_change_cap.columns else np.nan for m in metrics_list],
})
sens_df['no_agg_pct_change'] = abs(sens_df['no_agg_mean'] - sens_df['base_mean']) / abs(sens_df['base_mean']) * 100
sens_df['capped_pct_change'] = abs(sens_df['capped_mean'] - sens_df['base_mean']) / abs(sens_df['base_mean']) * 100
sens_df.to_csv(f'{OUT_TBL}/sensitivity_comparison.csv', index=False)

# ========== FIGURES ==========
print("\n=== Generating figures ===")

# Figure 1: Metro heatmap (metros x metrics, colored by change magnitude)
change_display = metro_change[[f'd_{m}' for m in metrics_list]].copy()
change_display.columns = ['Entry share', 'AI prevalence', 'Org scope', 'Median desc len', 'Tech diversity']

# Standardize for heatmap display
change_z = (change_display - change_display.mean()) / change_display.std()

# Sort by AI prevalence change for visual coherence
change_z = change_z.sort_values('AI prevalence', ascending=False)

fig, ax = plt.subplots(figsize=(10, max(8, len(change_z) * 0.35)))
sns.heatmap(change_z, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Z-score of change'})
ax.set_title('Metro-Level SWE Market Changes (2024 to 2026)\nStandardized Change Magnitude', fontsize=12)
ax.set_xlabel('Metric')
ax.set_ylabel('Metro Area')
plt.tight_layout()
plt.savefig(f'{OUT_FIG}/metro_heatmap.png', dpi=150)
plt.close()

# Figure 2: AI surge vs entry decline scatter
fig, ax = plt.subplots(figsize=(9, 7))
x = metro_change['d_ai_prevalence']
y = metro_change['d_entry_share']
sizes = metro_change['n_2026'] / metro_change['n_2026'].max() * 200 + 20

# Color by hub status
colors = ['#e74c3c' if m in tech_hubs else '#3498db' for m in metro_change.index]
scatter = ax.scatter(x, y, s=sizes, c=colors, alpha=0.7, edgecolors='white', linewidth=0.5)

# Add metro labels
for m in metro_change.index:
    ax.annotate(m.replace(' Metro', '').replace(' Area', ''),
                (metro_change.loc[m, 'd_ai_prevalence'], metro_change.loc[m, 'd_entry_share']),
                fontsize=7, alpha=0.8, ha='center', va='bottom')

ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

# Fit line
valid = metro_change[['d_ai_prevalence', 'd_entry_share']].dropna()
slope, intercept, r, p, se = stats.linregress(valid['d_ai_prevalence'], valid['d_entry_share'])
x_fit = np.linspace(x.min(), x.max(), 100)
ax.plot(x_fit, slope * x_fit + intercept, 'k--', alpha=0.5, linewidth=1)
ax.text(0.02, 0.98, f'r = {r:.3f}, p = {p:.3f}', transform=ax.transAxes,
        fontsize=10, va='top', fontweight='bold')

# Legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='Tech hub'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, label='Non-hub')]
ax.legend(handles=legend_elements, loc='lower right')

ax.set_xlabel('Change in AI Prevalence (pp)')
ax.set_ylabel('Change in Entry Share (pp)')
ax.set_title('Metro-Level: AI Adoption Surge vs Entry Role Change')
plt.tight_layout()
plt.savefig(f'{OUT_FIG}/ai_vs_entry_metro.png', dpi=150)
plt.close()

# Figure 3: Top/bottom metros bar chart for entry share change
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Entry share
sorted_entry = metro_change['d_entry_share'].sort_values()
colors_entry = ['#e74c3c' if v < 0 else '#27ae60' for v in sorted_entry]
axes[0].barh(range(len(sorted_entry)), sorted_entry.values, color=colors_entry, alpha=0.8)
axes[0].set_yticks(range(len(sorted_entry)))
axes[0].set_yticklabels([m.replace(' Metro', '').replace(' Area', '') for m in sorted_entry.index], fontsize=7)
axes[0].set_xlabel('Change in Entry Share (pp)')
axes[0].set_title('Entry Share Change by Metro')
axes[0].axvline(0, color='black', linewidth=0.5)

# AI prevalence
sorted_ai = metro_change['d_ai_prevalence'].sort_values()
colors_ai = ['#3498db' if v > 0 else '#95a5a6' for v in sorted_ai]
axes[1].barh(range(len(sorted_ai)), sorted_ai.values, color=colors_ai, alpha=0.8)
axes[1].set_yticks(range(len(sorted_ai)))
axes[1].set_yticklabels([m.replace(' Metro', '').replace(' Area', '') for m in sorted_ai.index], fontsize=7)
axes[1].set_xlabel('Change in AI Prevalence (pp)')
axes[1].set_title('AI Prevalence Change by Metro')
axes[1].axvline(0, color='black', linewidth=0.5)

fig.suptitle('Geographic Heterogeneity in SWE Market Changes', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_FIG}/metro_bar_charts.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 4: Remote share by metro (2026)
fig, ax = plt.subplots(figsize=(10, 6))
remote_sorted = remote_2026['remote_share'].sort_values(ascending=True)
bars = ax.barh(range(len(remote_sorted)), remote_sorted.values, color='#9b59b6', alpha=0.7)
ax.set_yticks(range(len(remote_sorted)))
ax.set_yticklabels([m.replace(' Metro', '').replace(' Area', '') for m in remote_sorted.index], fontsize=7)
ax.set_xlabel('Remote Work Share (2026)')
ax.set_title('Remote Work Share by Metro Area (2026 SWE Postings)')
# Add value labels
for bar, val in zip(bars, remote_sorted.values):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.0%}', ha='left', va='center', fontsize=7)
plt.tight_layout()
plt.savefig(f'{OUT_FIG}/remote_by_metro.png', dpi=150)
plt.close()

print(f"\nT17 analysis complete. Outputs in {OUT_TBL}/ and {OUT_FIG}/")
print("Tables:", os.listdir(OUT_TBL))
print("Figures:", os.listdir(OUT_FIG))
