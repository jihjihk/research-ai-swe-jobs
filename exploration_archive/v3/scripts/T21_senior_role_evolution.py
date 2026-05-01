#!/usr/bin/env python3
"""T21: Senior Role Evolution Deep Dive

Analyzes how senior SWE roles evolved across three language dimensions:
people management, technical orchestration, and strategic scope.
Compares to entry-level management surge from T11.
"""

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os, re, json

PARQUET = 'data/unified.parquet'
CLEANED_TEXT = 'exploration/artifacts/shared/swe_cleaned_text.parquet'
TECH_MATRIX = 'exploration/artifacts/shared/swe_tech_matrix.parquet'
FIG_DIR = 'exploration/figures/T21'
TBL_DIR = 'exploration/tables/T21'

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

# ── Language profile keyword lists ──

PEOPLE_MANAGEMENT = [
    r'\bmanag(?:e|es|ed|ing|ement|er)\b',
    r'\bmentor\w*', r'\bcoach\w*',
    r'\bhir(?:e|es|ed|ing)\b', r'\binterview\w*',
    r'\bgrow(?:th|ing)?\b',
    r'\bdevelop\s+talent', r'\btalent\s+develop',
    r'\bperformance\s+review', r'\bcareer\s+develop\w*',
    r'\b1[\:\-]1(?:s)?\b', r'\bone[\:\-]on[\:\-]one',
    r'\bheadcount\b', r'\bdirect\s+report\w*',
    r'\bpeople\s+manage\w*', r'\bstaff(?:ing)?\b',
    r'\bteam\s+lead\w*', r'\blead\s+(?:a\s+)?team',
    r'\bsupervis\w*', r'\brecruit\w*'
]

TECHNICAL_ORCHESTRATION = [
    r'\barchitecture\s+review\w*', r'\bcode\s+review\w*',
    r'\bsystem\s+design\b', r'\btechnical\s+direction\b',
    r'\btechnical\s+lead\w*', r'\btech\s+lead\w*',
    r'\bai\s+orchestrat\w*', r'\bagent\b',
    r'\bworkflow\b', r'\bpipeline\b',
    r'\bautomation\b', r'\bautomat\w+',
    r'\bevaluat\w*', r'\bvalidat\w*',
    r'\bquality\s+gate\w*', r'\bguardrail\w*',
    r'\bprompt\s+engineer\w*',
    r'\bdesign\s+pattern\w*', r'\btechnical\s+strategy',
    r'\btechnical\s+debt\b', r'\bplatform\s+engineer\w*',
    r'\binfrastructure\s+(?:design|architect)\w*'
]

STRATEGIC_SCOPE = [
    r'\bstakeholder\w*',
    r'\bbusiness\s+impact\b', r'\brevenue\b',
    r'\bproduct\s+strateg\w*', r'\broadmap\w*',
    r'\bprioritiz\w*', r'\bresource\s+alloc\w*',
    r'\bbudget\w*',
    r'\bcross[\-\s]functional\s+align\w*',
    r'\bcross[\-\s]functional\b',
    r'\bexecutive\b', r'\bc[\-\s]suite\b',
    r'\bkpi\b', r'\bmetric\w*\s+(?:driv|own)',
    r'\borgani[sz]ational\b',
    r'\bstrategic\s+(?:plan|init|direction|vision|partner)\w*',
    r'\bbusiness\s+(?:case|objective|requirement|goal)\w*',
    r'\bmarket\s+(?:analysis|research|opportunit)\w*'
]


def count_matches(text, patterns):
    if not text:
        return 0
    text_lower = text.lower()
    count = 0
    for p in patterns:
        count += len(re.findall(p, text_lower))
    return count


def density_per_1k(count, length):
    if length == 0:
        return 0.0
    return count / length * 1000


def has_any_match(text, patterns):
    if not text:
        return False
    text_lower = text.lower()
    for p in patterns:
        if re.search(p, text_lower):
            return True
    return False


# ── Load data ──
print("Loading data...")
con = duckdb.connect()
con.execute("SET memory_limit='4GB'")

# Load ALL SWE with known seniority (for cross-seniority comparison)
df_all = con.execute("""
SELECT
    u.uid,
    u.period,
    u.source,
    u.seniority_final,
    u.seniority_3level,
    u.is_aggregator,
    u.company_name_canonical,
    COALESCE(c.description_cleaned, u.description_core, u.description) as text_for_analysis,
    LENGTH(COALESCE(c.description_cleaned, u.description_core, u.description)) as text_length,
    u.yoe_extracted
FROM read_parquet('{parquet}') u
LEFT JOIN read_parquet('{cleaned}') c ON u.uid = c.uid
WHERE u.source_platform = 'linkedin'
  AND u.is_english = true
  AND u.date_flag = 'ok'
  AND u.is_swe = true
  AND u.seniority_final != 'unknown'
""".format(parquet=PARQUET, cleaned=CLEANED_TEXT)).df()

# Tech matrix for AI mentions
tech_df = con.execute("""
SELECT * FROM read_parquet('{tech}')
""".format(tech=TECH_MATRIX)).df()

ai_cols = [c for c in tech_df.columns if c.startswith('ai_')]
tech_df['ai_mention'] = tech_df[ai_cols].any(axis=1).astype(int)
tech_df_sub = tech_df[['uid', 'ai_mention']]

df_all = df_all.merge(tech_df_sub, on='uid', how='left')
df_all['ai_mention'] = df_all['ai_mention'].fillna(0)

print(f"Total SWE with known seniority: {len(df_all)}")
print(f"Senior (mid-senior + director): {len(df_all[df_all['seniority_final'].isin(['mid-senior','director'])])}")

# Period grouping
df_all['period_group'] = df_all['period'].map({
    '2024-01': '2024', '2024-04': '2024', '2026-03': '2026'
})

# ── Compute language profiles ──
print("\nComputing language profiles...")

df_all['mgmt_count'] = df_all['text_for_analysis'].apply(lambda x: count_matches(x, PEOPLE_MANAGEMENT))
df_all['orch_count'] = df_all['text_for_analysis'].apply(lambda x: count_matches(x, TECHNICAL_ORCHESTRATION))
df_all['strat_count'] = df_all['text_for_analysis'].apply(lambda x: count_matches(x, STRATEGIC_SCOPE))

df_all['mgmt_density'] = df_all.apply(lambda r: density_per_1k(r['mgmt_count'], r['text_length']), axis=1)
df_all['orch_density'] = df_all.apply(lambda r: density_per_1k(r['orch_count'], r['text_length']), axis=1)
df_all['strat_density'] = df_all.apply(lambda r: density_per_1k(r['strat_count'], r['text_length']), axis=1)

df_all['has_mgmt'] = df_all['text_for_analysis'].apply(lambda x: has_any_match(x, PEOPLE_MANAGEMENT))
df_all['has_orch'] = df_all['text_for_analysis'].apply(lambda x: has_any_match(x, TECHNICAL_ORCHESTRATION))
df_all['has_strat'] = df_all['text_for_analysis'].apply(lambda x: has_any_match(x, STRATEGIC_SCOPE))

print("  Done.")

# ── Focus on senior roles (mid-senior + director) ──
df_senior = df_all[df_all['seniority_final'].isin(['mid-senior', 'director'])].copy()
print(f"\nSenior subset: {len(df_senior)}")

# ── Step 1: Language profile summary by period ──
print("\n=== Language Profiles by Period (Senior Roles) ===")

profile_rows = []
for pg in ['2024', '2026']:
    for sen in ['mid-senior', 'director']:
        subset = df_senior[(df_senior['period_group'] == pg) & (df_senior['seniority_final'] == sen)]
        if len(subset) < 10:
            continue
        row = {
            'period': pg, 'seniority': sen, 'n': len(subset),
            'mgmt_density_mean': subset['mgmt_density'].mean(),
            'mgmt_density_median': subset['mgmt_density'].median(),
            'orch_density_mean': subset['orch_density'].mean(),
            'orch_density_median': subset['orch_density'].median(),
            'strat_density_mean': subset['strat_density'].mean(),
            'strat_density_median': subset['strat_density'].median(),
            'pct_has_mgmt': subset['has_mgmt'].mean() * 100,
            'pct_has_orch': subset['has_orch'].mean() * 100,
            'pct_has_strat': subset['has_strat'].mean() * 100,
            'pct_ai_mention': subset['ai_mention'].mean() * 100,
            'mean_text_length': subset['text_length'].mean(),
        }
        profile_rows.append(row)
        print(f"  {pg} {sen} (n={row['n']}): mgmt={row['mgmt_density_mean']:.2f}, "
              f"orch={row['orch_density_mean']:.2f}, strat={row['strat_density_mean']:.2f}")

profile_df = pd.DataFrame(profile_rows)
profile_df.to_csv(f'{TBL_DIR}/senior_language_profiles.csv', index=False)

# ── Step 2: 2D scatter - Management vs Orchestration ──
print("\n=== Generating Scatter Plots ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, sen_level in enumerate(['mid-senior', 'director']):
    ax = axes[idx]
    for pg, color, marker in [('2024', '#4ECDC4', 'o'), ('2026', '#FF6B6B', 's')]:
        subset = df_senior[(df_senior['period_group'] == pg) & (df_senior['seniority_final'] == sen_level)]
        if len(subset) == 0:
            continue
        # Subsample for readability (max 2000 per group)
        if len(subset) > 2000:
            sample = subset.sample(2000, random_state=42)
        else:
            sample = subset
        ax.scatter(sample['mgmt_density'], sample['orch_density'],
                   c=color, marker=marker, alpha=0.15, s=15, label=f'{pg} (n={len(subset)})')
        # Add mean point
        ax.scatter(subset['mgmt_density'].mean(), subset['orch_density'].mean(),
                   c=color, marker='X', s=200, edgecolors='black', linewidth=1.5, zorder=5)

    ax.set_xlabel('People Management Density (per 1K chars)', fontsize=10)
    ax.set_ylabel('Technical Orchestration Density (per 1K chars)', fontsize=10)
    ax.set_title(f'{sen_level.title()} Roles', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, max(6, ax.get_xlim()[1]))
    ax.set_ylim(-0.5, max(6, ax.get_ylim()[1]))

plt.suptitle('Senior SWE: Management vs Orchestration Language',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/mgmt_vs_orch_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved mgmt_vs_orch_scatter.png")

# ── Step 3: 3D view as 2D triangle (management vs orchestration vs strategic) ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (x_dim, y_dim, x_label, y_label) in enumerate([
    ('mgmt_density', 'strat_density', 'People Management', 'Strategic Scope'),
    ('orch_density', 'strat_density', 'Technical Orchestration', 'Strategic Scope')
]):
    ax = axes[idx]
    for pg, color, marker in [('2024', '#4ECDC4', 'o'), ('2026', '#FF6B6B', 's')]:
        subset = df_senior[df_senior['period_group'] == pg]
        if len(subset) == 0:
            continue
        if len(subset) > 2000:
            sample = subset.sample(2000, random_state=42)
        else:
            sample = subset
        ax.scatter(sample[x_dim], sample[y_dim],
                   c=color, marker=marker, alpha=0.15, s=15, label=f'{pg} (n={len(subset)})')
        ax.scatter(subset[x_dim].mean(), subset[y_dim].mean(),
                   c=color, marker='X', s=200, edgecolors='black', linewidth=1.5, zorder=5)

    ax.set_xlabel(f'{x_label} Density (per 1K chars)', fontsize=10)
    ax.set_ylabel(f'{y_label} Density (per 1K chars)', fontsize=10)
    ax.legend(fontsize=9)

plt.suptitle('Senior SWE: Three-Dimensional Language Profile',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/three_dimension_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved three_dimension_scatter.png")

# ── Step 4: Senior sub-archetypes via clustering ──
print("\n=== Senior Sub-Archetype Clustering ===")

cluster_features = ['mgmt_density', 'orch_density', 'strat_density']
cluster_data = df_senior[cluster_features].values

scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_data)

# Try 3-5 clusters, pick best silhouette
from sklearn.metrics import silhouette_score

best_k = 3
best_sil = -1
for k in [3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(cluster_scaled)
    sil = silhouette_score(cluster_scaled, labels, sample_size=min(5000, len(cluster_scaled)))
    print(f"  k={k}: silhouette={sil:.3f}")
    if sil > best_sil:
        best_sil = sil
        best_k = k

print(f"  Best k={best_k} (silhouette={best_sil:.3f})")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
df_senior = df_senior.copy()
df_senior['cluster'] = km_final.fit_predict(cluster_scaled)

# Characterize clusters
cluster_rows = []
for c in range(best_k):
    cdata = df_senior[df_senior['cluster'] == c]
    row = {
        'cluster': c,
        'n': len(cdata),
        'pct': len(cdata) / len(df_senior) * 100,
        'mgmt_density': cdata['mgmt_density'].mean(),
        'orch_density': cdata['orch_density'].mean(),
        'strat_density': cdata['strat_density'].mean(),
        'pct_ai': cdata['ai_mention'].mean() * 100,
        'pct_director': (cdata['seniority_final'] == 'director').mean() * 100,
    }
    # Period composition
    for pg in ['2024', '2026']:
        pg_n = len(cdata[cdata['period_group'] == pg])
        row[f'n_{pg}'] = pg_n
        row[f'pct_{pg}'] = pg_n / max(1, len(df_senior[df_senior['period_group'] == pg])) * 100
    cluster_rows.append(row)

cluster_df = pd.DataFrame(cluster_rows)
print("\nCluster profiles:")
print(cluster_df.to_string(index=False))
cluster_df.to_csv(f'{TBL_DIR}/senior_archetypes.csv', index=False)

# Name clusters based on dominant dimension
for _, row in cluster_df.iterrows():
    dims = {'mgmt': row['mgmt_density'], 'orch': row['orch_density'], 'strat': row['strat_density']}
    dominant = max(dims, key=dims.get)
    print(f"  Cluster {row['cluster']}: Dominant={dominant}, "
          f"2024={row['pct_2024']:.1f}% of 2024, 2026={row['pct_2026']:.1f}% of 2026")

# ── Step 5: AI interaction ──
print("\n=== AI Interaction in Senior Roles ===")

ai_comparison_rows = []
for pg in ['2024', '2026']:
    for ai_flag in [0, 1]:
        subset = df_senior[(df_senior['period_group'] == pg) & (df_senior['ai_mention'] == ai_flag)]
        if len(subset) < 10:
            continue
        row = {
            'period': pg,
            'ai_mentioning': bool(ai_flag),
            'n': len(subset),
            'mgmt_density': subset['mgmt_density'].mean(),
            'orch_density': subset['orch_density'].mean(),
            'strat_density': subset['strat_density'].mean(),
            'mean_text_length': subset['text_length'].mean()
        }
        ai_comparison_rows.append(row)
        label = 'AI-mentioning' if ai_flag else 'Non-AI'
        print(f"  {pg} {label} (n={row['n']}): mgmt={row['mgmt_density']:.2f}, "
              f"orch={row['orch_density']:.2f}, strat={row['strat_density']:.2f}")

ai_df = pd.DataFrame(ai_comparison_rows)
ai_df.to_csv(f'{TBL_DIR}/senior_ai_interaction.csv', index=False)

# ── Step 6: Director deep dive ──
print("\n=== Director Deep Dive ===")
director = df_all[df_all['seniority_final'] == 'director'].copy()
midsenior = df_all[df_all['seniority_final'] == 'mid-senior'].copy()

director_rows = []
for pg in ['2024', '2026']:
    for level, data in [('director', director), ('mid-senior', midsenior)]:
        subset = data[data['period_group'] == pg]
        if len(subset) < 10:
            continue
        row = {
            'period': pg, 'level': level, 'n': len(subset),
            'mgmt_density': subset['mgmt_density'].mean(),
            'orch_density': subset['orch_density'].mean(),
            'strat_density': subset['strat_density'].mean(),
            'pct_ai': subset['ai_mention'].mean() * 100,
            'pct_has_mgmt': subset['has_mgmt'].mean() * 100,
            'pct_has_orch': subset['has_orch'].mean() * 100,
            'pct_has_strat': subset['has_strat'].mean() * 100,
        }
        director_rows.append(row)
        print(f"  {pg} {level} (n={row['n']}): mgmt={row['mgmt_density']:.2f}, "
              f"orch={row['orch_density']:.2f}, strat={row['strat_density']:.2f}, "
              f"ai={row['pct_ai']:.1f}%")

director_df = pd.DataFrame(director_rows)
director_df.to_csv(f'{TBL_DIR}/director_comparison.csv', index=False)

# ── Step 7: The "new senior" archetype question ──
print("\n=== New Senior Archetype Detection ===")

# Identify cluster compositions that shifted dramatically
for c in range(best_k):
    c2024 = df_senior[(df_senior['cluster'] == c) & (df_senior['period_group'] == '2024')]
    c2026 = df_senior[(df_senior['cluster'] == c) & (df_senior['period_group'] == '2026')]
    total_2024 = len(df_senior[df_senior['period_group'] == '2024'])
    total_2026 = len(df_senior[df_senior['period_group'] == '2026'])
    share_2024 = len(c2024) / total_2024 * 100 if total_2024 > 0 else 0
    share_2026 = len(c2026) / total_2026 * 100 if total_2026 > 0 else 0
    shift = share_2026 - share_2024
    print(f"  Cluster {c}: 2024 share={share_2024:.1f}%, 2026 share={share_2026:.1f}%, shift={shift:+.1f}pp")

    # AI content in 2026-only postings within emergent clusters
    if shift > 5:
        ai_rate = c2026['ai_mention'].mean() * 100
        orch_rate = c2026['orch_density'].mean()
        print(f"    -> GROWING cluster. 2026 AI rate: {ai_rate:.1f}%, orch density: {orch_rate:.2f}")

# ── Step 8: Entry vs Senior management comparison (THE KEY TEST) ──
print("\n=== Entry vs Senior Management Language Comparison ===")
print("(Tests whether management language migrated downward or expanded everywhere)")

comparison_rows = []
for pg in ['2024', '2026']:
    for sen in ['entry', 'associate', 'mid-senior', 'director']:
        subset = df_all[(df_all['period_group'] == pg) & (df_all['seniority_final'] == sen)]
        if len(subset) < 10:
            continue
        row = {
            'period': pg, 'seniority': sen, 'n': len(subset),
            'mgmt_density_mean': subset['mgmt_density'].mean(),
            'mgmt_density_median': subset['mgmt_density'].median(),
            'orch_density_mean': subset['orch_density'].mean(),
            'strat_density_mean': subset['strat_density'].mean(),
            'pct_has_mgmt': subset['has_mgmt'].mean() * 100,
            'pct_has_orch': subset['has_orch'].mean() * 100,
            'pct_has_strat': subset['has_strat'].mean() * 100,
            'mgmt_count_mean': subset['mgmt_count'].mean(),
            'orch_count_mean': subset['orch_count'].mean(),
            'strat_count_mean': subset['strat_count'].mean(),
            'mean_text_length': subset['text_length'].mean()
        }
        comparison_rows.append(row)
        print(f"  {pg} {sen} (n={row['n']}): mgmt_density={row['mgmt_density_mean']:.3f}, "
              f"has_mgmt={row['pct_has_mgmt']:.1f}%, "
              f"has_orch={row['pct_has_orch']:.1f}%")

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(f'{TBL_DIR}/entry_vs_senior_mgmt.csv', index=False)

# Compute the specific comparison: management language change at each level
print("\n  Management language change by seniority:")
for sen in ['entry', 'associate', 'mid-senior', 'director']:
    r2024 = comparison_df[(comparison_df['period'] == '2024') & (comparison_df['seniority'] == sen)]
    r2026 = comparison_df[(comparison_df['period'] == '2026') & (comparison_df['seniority'] == sen)]
    if len(r2024) > 0 and len(r2026) > 0:
        mgmt_pct_2024 = r2024.iloc[0]['pct_has_mgmt']
        mgmt_pct_2026 = r2026.iloc[0]['pct_has_mgmt']
        mgmt_dens_2024 = r2024.iloc[0]['mgmt_density_mean']
        mgmt_dens_2026 = r2026.iloc[0]['mgmt_density_mean']
        print(f"    {sen}: has_mgmt {mgmt_pct_2024:.1f}% -> {mgmt_pct_2026:.1f}% "
              f"(change: {mgmt_pct_2026 - mgmt_pct_2024:+.1f}pp), "
              f"density {mgmt_dens_2024:.3f} -> {mgmt_dens_2026:.3f} "
              f"({(mgmt_dens_2026 - mgmt_dens_2024)/max(mgmt_dens_2024, 0.001)*100:+.1f}%)")

# Statistical tests: Mann-Whitney for management density changes
print("\n  Statistical tests (Mann-Whitney U, management density):")
for sen in ['entry', 'associate', 'mid-senior', 'director']:
    d2024 = df_all[(df_all['period_group'] == '2024') & (df_all['seniority_final'] == sen)]['mgmt_density']
    d2026 = df_all[(df_all['period_group'] == '2026') & (df_all['seniority_final'] == sen)]['mgmt_density']
    if len(d2024) >= 10 and len(d2026) >= 10:
        u_stat, p_val = stats.mannwhitneyu(d2024, d2026, alternative='two-sided')
        r_effect = 1 - (2 * u_stat) / (len(d2024) * len(d2026))
        direction = 'INCREASED' if d2026.mean() > d2024.mean() else 'DECREASED'
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"    {sen}: U={u_stat:.0f}, p={p_val:.4g}, r={r_effect:.3f}, {direction} {sig}")

# ── Sensitivity (a): Aggregator exclusion ──
print("\n=== Sensitivity: Aggregator Exclusion ===")
df_all_noagg = df_all[~df_all['is_aggregator']].copy()

sens_rows = []
for pg in ['2024', '2026']:
    for sen in ['entry', 'mid-senior', 'director']:
        for label, data in [('all', df_all), ('no_agg', df_all_noagg)]:
            subset = data[(data['period_group'] == pg) & (data['seniority_final'] == sen)]
            if len(subset) < 10:
                continue
            sens_rows.append({
                'period': pg, 'seniority': sen, 'sample': label, 'n': len(subset),
                'mgmt_density': subset['mgmt_density'].mean(),
                'orch_density': subset['orch_density'].mean(),
                'strat_density': subset['strat_density'].mean(),
                'pct_has_mgmt': subset['has_mgmt'].mean() * 100,
            })

sens_agg_df = pd.DataFrame(sens_rows)
sens_agg_df.to_csv(f'{TBL_DIR}/sensitivity_aggregator.csv', index=False)
print(sens_agg_df.to_string(index=False))

# ── Generate entry vs senior comparison chart ──
print("\n=== Generating Entry vs Senior Comparison Chart ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
seniority_order = ['entry', 'associate', 'mid-senior', 'director']

for idx, (metric, title) in enumerate([
    ('mgmt_density_mean', 'People Management Density'),
    ('orch_density_mean', 'Technical Orchestration Density'),
    ('strat_density_mean', 'Strategic Scope Density')
]):
    ax = axes[idx]
    x = np.arange(len(seniority_order))
    width = 0.35

    vals_2024 = []
    vals_2026 = []
    for sen in seniority_order:
        r2024 = comparison_df[(comparison_df['period'] == '2024') & (comparison_df['seniority'] == sen)]
        r2026 = comparison_df[(comparison_df['period'] == '2026') & (comparison_df['seniority'] == sen)]
        vals_2024.append(r2024.iloc[0][metric] if len(r2024) > 0 else 0)
        vals_2026.append(r2026.iloc[0][metric] if len(r2026) > 0 else 0)

    bars1 = ax.bar(x - width/2, vals_2024, width, label='2024', color='#4ECDC4', alpha=0.85)
    bars2 = ax.bar(x + width/2, vals_2026, width, label='2026', color='#FF6B6B', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(seniority_order, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Density (per 1K chars)')
    ax.legend(fontsize=10)

    # Add value labels
    for bar, val in zip(bars1, vals_2024):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, vals_2026):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Language Profile Evolution Across All Seniority Levels',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/entry_vs_senior_language.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved entry_vs_senior_language.png")

# ── Binary indicators chart ──
fig, ax = plt.subplots(figsize=(12, 7))

categories = []
vals_2024_pct = []
vals_2026_pct = []
for sen in seniority_order:
    for metric, label in [('pct_has_mgmt', 'Management'), ('pct_has_orch', 'Orchestration'), ('pct_has_strat', 'Strategic')]:
        r2024 = comparison_df[(comparison_df['period'] == '2024') & (comparison_df['seniority'] == sen)]
        r2026 = comparison_df[(comparison_df['period'] == '2026') & (comparison_df['seniority'] == sen)]
        if len(r2024) > 0 and len(r2026) > 0:
            categories.append(f"{sen}\n{label}")
            vals_2024_pct.append(r2024.iloc[0][metric])
            vals_2026_pct.append(r2026.iloc[0][metric])

x = np.arange(len(categories))
width = 0.35
bars1 = ax.bar(x - width/2, vals_2024_pct, width, label='2024', color='#4ECDC4', alpha=0.85)
bars2 = ax.bar(x + width/2, vals_2026_pct, width, label='2026', color='#FF6B6B', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=8, rotation=45, ha='right')
ax.set_ylabel('% of Postings', fontsize=11)
ax.set_title('Binary Indicator Presence by Seniority Level', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/binary_indicators_by_seniority.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved binary_indicators_by_seniority.png")

print("\n=== DONE ===")
print(f"Figures: {FIG_DIR}/")
print(f"Tables: {TBL_DIR}/")
