#!/usr/bin/env python3
"""T16: Company hiring strategy typology — overlap panel, clustering, decomposition."""

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
import os, warnings
warnings.filterwarnings('ignore')

OUT_FIG = 'exploration/figures/T16'
OUT_TBL = 'exploration/tables/T16'
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TBL, exist_ok=True)

con = duckdb.connect()
PARQUET = 'data/unified.parquet'

# ========== STEP 1: Base SWE panel ==========
print("=== Step 1: Build SWE panel ===")
base = con.execute(f"""
SELECT uid, source, period, company_name_canonical, is_aggregator,
       seniority_final, seniority_3level,
       description_core, description_core_llm, llm_extraction_coverage,
       description_length, core_length,
       company_industry, metro_area, is_remote_inferred
FROM read_parquet('{PARQUET}')
WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true
""").df()
print(f"SWE base: {len(base)} rows")

# Map to two periods: 2024 (arshkon only for entry labels) and 2026
# For company overlap, we pool arshkon + asaniczka as "2024"
base['era'] = base['period'].map({'2024-01': '2024', '2024-04': '2024', '2026-03': '2026'})
print(f"Era counts: {base['era'].value_counts().to_dict()}")

# Tech matrix
tech = con.execute("""
SELECT * FROM read_parquet('exploration/artifacts/shared/swe_tech_matrix.parquet')
""").df()
tech_cols = [c for c in tech.columns if c != 'uid']
ai_cols = [c for c in tech_cols if c.startswith('ai_')]
print(f"Tech columns: {len(tech_cols)}, AI columns: {len(ai_cols)}")

base = base.merge(tech[['uid'] + ai_cols + ['lang_python']], on='uid', how='left')

# Compute metrics per posting
base['has_ai'] = base[ai_cols].any(axis=1).astype(int)
base['ai_count'] = base[ai_cols].sum(axis=1)

# Org scope terms (management/leadership/stakeholder/cross-functional/strategic)
org_scope_pattern = r'(?i)\b(manage|management|lead|leadership|stakeholder|cross-functional|strategic|mentor|coordinate|oversee|collaborate)\b'
# Use description (full text) for keyword detection
base['text_for_kw'] = base['description_core'].fillna(base.get('description_core_llm', ''))
base['org_scope_count'] = base['text_for_kw'].fillna('').str.count(org_scope_pattern)

# Entry flag
base['is_entry'] = (base['seniority_3level'] == 'junior').astype(int)

# Description length (use core_length for consistency)
base['desc_len'] = base['core_length'].fillna(base['description_length'])

# Tech count from full tech matrix
tech_count = tech.set_index('uid')[tech_cols].sum(axis=1).rename('tech_count')
base = base.merge(tech_count, on='uid', how='left')

# ========== STEP 2: Identify overlap companies (>=3 SWE in BOTH eras) ==========
print("\n=== Step 2: Overlap companies ===")
co_era = base.groupby(['company_name_canonical', 'era']).size().unstack(fill_value=0)
co_era.columns = ['n_2024', 'n_2026']
overlap = co_era[(co_era['n_2024'] >= 3) & (co_era['n_2026'] >= 3)]
print(f"Companies with >=3 SWE in both eras: {len(overlap)}")
print(f"Median postings 2024: {overlap['n_2024'].median():.0f}, 2026: {overlap['n_2026'].median():.0f}")

# ========== STEP 3: Per-company change metrics ==========
print("\n=== Step 3: Per-company metrics ===")
overlap_names = set(overlap.index)
panel = base[base['company_name_canonical'].isin(overlap_names)].copy()
print(f"Panel rows: {len(panel)}")

def company_metrics(df):
    """Compute metrics for a group."""
    n = len(df)
    return pd.Series({
        'n_postings': n,
        'entry_share': df['is_entry'].mean(),
        'ai_prevalence': df['has_ai'].mean(),
        'mean_desc_len': df['desc_len'].mean(),
        'mean_tech_count': df['tech_count'].mean(),
        'mean_org_scope': df['org_scope_count'].mean(),
        'mean_ai_count': df['ai_count'].mean(),
    })

metrics_by_era = panel.groupby(['company_name_canonical', 'era']).apply(company_metrics).unstack('era')
# Flatten multi-index columns
metrics_by_era.columns = [f"{m}_{e}" for m, e in metrics_by_era.columns]

# Change metrics (2026 - 2024)
change = pd.DataFrame(index=metrics_by_era.index)
for m in ['entry_share', 'ai_prevalence', 'mean_desc_len', 'mean_tech_count', 'mean_org_scope', 'mean_ai_count']:
    change[f'd_{m}'] = metrics_by_era[f'{m}_2026'] - metrics_by_era[f'{m}_2024']
    change[f'{m}_2024'] = metrics_by_era[f'{m}_2024']
    change[f'{m}_2026'] = metrics_by_era[f'{m}_2026']

change['n_2024'] = metrics_by_era['n_postings_2024']
change['n_2026'] = metrics_by_era['n_postings_2026']
change['d_log_n'] = np.log(change['n_2026'] / change['n_2024'])

change = change.dropna(subset=[c for c in change.columns if c.startswith('d_')])
print(f"Companies with complete change data: {len(change)}")

# Summary stats
print("\nChange metric summary:")
for c in [cc for cc in change.columns if cc.startswith('d_')]:
    print(f"  {c}: mean={change[c].mean():.4f}, median={change[c].median():.4f}, std={change[c].std():.4f}")

# ========== STEP 4: Cluster companies by change profile ==========
print("\n=== Step 4: Clustering ===")
cluster_features = ['d_entry_share', 'd_ai_prevalence', 'd_mean_desc_len', 'd_mean_tech_count', 'd_mean_org_scope']
X = change[cluster_features].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal k using silhouette
from sklearn.metrics import silhouette_score
sil_scores = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, labels)
    print(f"  k={k}: silhouette={sil_scores[k]:.3f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"Best k by silhouette: {best_k}")

# Also run with k=4 to get interpretable strategies
for k_use in [best_k, 4]:
    km = KMeans(n_clusters=k_use, n_init=50, random_state=42)
    change[f'cluster_k{k_use}'] = km.fit_predict(X_scaled)

# Profile clusters with k=4
k_final = 4
print(f"\n=== Cluster profiles (k={k_final}) ===")
for cl in range(k_final):
    subset = change[change[f'cluster_k{k_final}'] == cl]
    print(f"\nCluster {cl} (n={len(subset)}):")
    for feat in cluster_features + ['d_log_n']:
        print(f"  {feat}: mean={subset[feat].mean():.4f}, median={subset[feat].median():.4f}")

# Name clusters based on profiles
cluster_profiles = change.groupby(f'cluster_k{k_final}')[cluster_features + ['d_log_n', 'n_2024', 'n_2026']].mean()
print("\nCluster means for naming:")
print(cluster_profiles.to_string())

# Save cluster assignment
cluster_out = change[[f'cluster_k{k_final}', 'n_2024', 'n_2026', 'd_log_n'] + cluster_features].copy()
cluster_out.to_csv(f'{OUT_TBL}/company_clusters.csv')

# ========== STEP 5: Within-company vs between-company decomposition ==========
print("\n=== Step 5: Decomposition ===")

# Methodology: Oaxaca-style shift-share decomposition
# Total change = Within-company + Between-company (compositional)
# Within = sum_j [share_j_bar * delta_x_j]  (same companies, different metrics)
# Between = sum_j [delta_share_j * x_j_bar]  (different companies, same metrics)
# Cross = sum_j [delta_share_j * delta_x_j]

# All companies (not just overlap) contribute to the aggregate
all_2024 = base[base['era'] == '2024'].copy()
all_2026 = base[base['era'] == '2026'].copy()
N_2024 = len(all_2024)
N_2026 = len(all_2026)

# Aggregate metrics
agg_metrics = ['is_entry', 'has_ai', 'desc_len']
agg_labels = ['Entry share', 'AI prevalence', 'Description length']

decomp_results = []

for metric, label in zip(agg_metrics, agg_labels):
    # Aggregate change
    agg_2024 = all_2024[metric].mean()
    agg_2026 = all_2026[metric].mean()
    total_change = agg_2026 - agg_2024

    # Company-level for overlap set
    overlap_2024 = all_2024[all_2024['company_name_canonical'].isin(overlap_names)]
    overlap_2026 = all_2026[all_2026['company_name_canonical'].isin(overlap_names)]

    # Non-overlap (new entrants / exits)
    exit_only = all_2024[~all_2024['company_name_canonical'].isin(overlap_names)]
    enter_only = all_2026[~all_2026['company_name_canonical'].isin(overlap_names)]

    # Within-company (overlap companies only)
    co_mean_2024 = overlap_2024.groupby('company_name_canonical')[metric].mean()
    co_mean_2026 = overlap_2026.groupby('company_name_canonical')[metric].mean()
    co_n_2024 = overlap_2024.groupby('company_name_canonical').size()
    co_n_2026 = overlap_2026.groupby('company_name_canonical').size()

    common = co_mean_2024.index.intersection(co_mean_2026.index)

    # Overlap share in each era
    overlap_share_2024 = co_n_2024[common].sum() / N_2024
    overlap_share_2026 = co_n_2026[common].sum() / N_2026
    overlap_share_avg = (overlap_share_2024 + overlap_share_2026) / 2

    # Within-company: change in metric among overlap companies (weighted by avg share)
    within_mean_2024 = overlap_2024[metric].mean()
    within_mean_2026 = overlap_2026[metric].mean()
    within_change = within_mean_2026 - within_mean_2024
    within_weighted = within_change * overlap_share_avg

    # Composition: metric difference between entrants vs exits
    exit_mean = exit_only[metric].mean() if len(exit_only) > 0 else 0
    enter_mean = enter_only[metric].mean() if len(enter_only) > 0 else 0

    # Between: composition shift (new companies replacing old ones)
    exit_share = len(exit_only) / N_2024
    enter_share = len(enter_only) / N_2026

    # Shares in era pools
    overlap_mean_2024 = within_mean_2024
    overlap_mean_2026 = within_mean_2026

    # More precise decomposition using group means
    # agg_2024 = overlap_share_2024 * overlap_mean_2024 + (1 - overlap_share_2024) * exit_mean
    # agg_2026 = overlap_share_2026 * overlap_mean_2026 + (1 - overlap_share_2026) * enter_mean
    # delta = (overlap_share_avg)(overlap_mean_2026 - overlap_mean_2024) [WITHIN]
    #       + (overlap_share_2026 - overlap_share_2024)(overlap_mean_avg - non_overlap_mean_avg) [BETWEEN reallocation]
    #       + enter_share*enter_mean - exit_share*exit_mean adjusted terms [ENTRY/EXIT]

    # Simple 3-component decomp:
    # Total = within_overlap + composition_from_entry_exit + interaction

    decomp_results.append({
        'metric': label,
        'agg_2024': agg_2024,
        'agg_2026': agg_2026,
        'total_change': total_change,
        'overlap_within_change': within_change,
        'overlap_within_weighted': within_weighted,
        'overlap_share_2024': overlap_share_2024,
        'overlap_share_2026': overlap_share_2026,
        'exit_mean': exit_mean,
        'enter_mean': enter_mean,
        'n_overlap_2024': len(overlap_2024),
        'n_overlap_2026': len(overlap_2026),
        'n_exit': len(exit_only),
        'n_enter': len(enter_only),
    })

decomp_df = pd.DataFrame(decomp_results)
print("\nDecomposition results:")
print(decomp_df.to_string())
decomp_df.to_csv(f'{OUT_TBL}/decomposition_basic.csv', index=False)

# ========== STEP 5b: Formal shift-share decomposition ==========
print("\n=== Step 5b: Formal shift-share decomposition ===")
# For each metric, we decompose into:
# 1. Within-overlap: how much changed among companies present in both periods
# 2. Entry effect: how much new entrants shift the aggregate
# 3. Exit effect: how much exiting companies shift the aggregate
# Using posting-weighted decomposition

formal_decomp = []

for metric, label in zip(agg_metrics, agg_labels):
    # Partition into 3 groups: overlap, exit-only (2024), enter-only (2026)
    ov24 = all_2024[all_2024['company_name_canonical'].isin(overlap_names)]
    ov26 = all_2026[all_2026['company_name_canonical'].isin(overlap_names)]
    ex = all_2024[~all_2024['company_name_canonical'].isin(overlap_names)]
    en = all_2026[~all_2026['company_name_canonical'].isin(overlap_names)]

    # Shares of total
    w_ov24 = len(ov24) / N_2024
    w_ex = len(ex) / N_2024
    w_ov26 = len(ov26) / N_2026
    w_en = len(en) / N_2026

    # Group means
    m_ov24 = ov24[metric].mean()
    m_ov26 = ov26[metric].mean()
    m_ex = ex[metric].mean() if len(ex) > 0 else 0
    m_en = en[metric].mean() if len(en) > 0 else 0

    # Aggregate in each period
    A24 = w_ov24 * m_ov24 + w_ex * m_ex  # should equal all_2024[metric].mean()
    A26 = w_ov26 * m_ov26 + w_en * m_en  # should equal all_2026[metric].mean()
    total = A26 - A24

    # Within-overlap component: w_ov_avg * (m_ov26 - m_ov24)
    w_ov_avg = (w_ov24 + w_ov26) / 2
    within = w_ov_avg * (m_ov26 - m_ov24)

    # Reallocation: (w_ov26 - w_ov24) * m_ov_avg + entry/exit
    # Simpler: residual
    composition = total - within

    # Within-company (weighted by company posting count)
    # Per-company within change
    co_m24 = ov24.groupby('company_name_canonical')[metric].agg(['mean', 'count']).rename(columns={'mean': 'm24', 'count': 'n24'})
    co_m26 = ov26.groupby('company_name_canonical')[metric].agg(['mean', 'count']).rename(columns={'mean': 'm26', 'count': 'n26'})
    co_both = co_m24.join(co_m26, how='inner')
    co_both['w_avg'] = (co_both['n24'] / co_both['n24'].sum() + co_both['n26'] / co_both['n26'].sum()) / 2
    co_both['within_i'] = co_both['w_avg'] * (co_both['m26'] - co_both['m24'])

    # Sum of within-company changes
    pure_within = co_both['within_i'].sum()
    # Reallocation among overlap companies
    intra_realloc = within - pure_within * w_ov_avg  # interaction

    within_pct = (within / abs(total) * 100) if total != 0 else np.nan
    comp_pct = (composition / abs(total) * 100) if total != 0 else np.nan

    formal_decomp.append({
        'metric': label,
        'total_change': total,
        'within_overlap': within,
        'within_pct': within_pct,
        'composition': composition,
        'composition_pct': comp_pct,
        'overlap_share_2024': w_ov24,
        'overlap_share_2026': w_ov26,
        'overlap_mean_2024': m_ov24,
        'overlap_mean_2026': m_ov26,
        'entrant_mean': m_en,
        'exit_mean': m_ex,
        'n_overlap_companies': len(co_both),
    })

    print(f"\n{label}:")
    print(f"  Total change: {total:.4f}")
    print(f"  Within-overlap: {within:.4f} ({within_pct:.1f}%)")
    print(f"  Composition: {composition:.4f} ({comp_pct:.1f}%)")
    print(f"  Overlap share 2024: {w_ov24:.3f}, 2026: {w_ov26:.3f}")
    print(f"  Overlap mean: 2024={m_ov24:.4f}, 2026={m_ov26:.4f}")
    print(f"  Entrant mean: {m_en:.4f}, Exit mean: {m_ex:.4f}")

formal_decomp_df = pd.DataFrame(formal_decomp)
formal_decomp_df.to_csv(f'{OUT_TBL}/decomposition_formal.csv', index=False)

# ========== STEP 6: New market entrants ==========
print("\n=== Step 6: New market entrants ===")
companies_2024 = set(all_2024['company_name_canonical'].dropna().unique())
companies_2026 = set(all_2026['company_name_canonical'].dropna().unique())
new_entrants = companies_2026 - companies_2024
exited = companies_2024 - companies_2026
print(f"Companies in 2024 only: {len(exited)}")
print(f"Companies in 2026 only (new entrants): {len(new_entrants)}")
print(f"Companies in both: {len(companies_2024 & companies_2026)}")

# Profile new entrants vs continuers
entrant_postings = all_2026[all_2026['company_name_canonical'].isin(new_entrants)]
continuer_postings = all_2026[all_2026['company_name_canonical'].isin(companies_2024 & companies_2026)]

print(f"\nNew entrant postings: {len(entrant_postings)}")
print(f"Continuer postings: {len(continuer_postings)}")

# Compare metrics
for metric, label in zip(agg_metrics + ['tech_count', 'org_scope_count'],
                          agg_labels + ['Tech count', 'Org scope']):
    ent_m = entrant_postings[metric].mean()
    cont_m = continuer_postings[metric].mean()
    print(f"  {label}: entrants={ent_m:.4f}, continuers={cont_m:.4f}, diff={ent_m-cont_m:.4f}")

# Industry of new entrants (where available)
entrant_industry = entrant_postings.dropna(subset=['company_industry'])
if len(entrant_industry) > 0:
    top_industries = entrant_industry['company_industry'].value_counts().head(15)
    print(f"\nTop industries of new entrants (n={len(entrant_industry)} with industry data):")
    for ind, cnt in top_industries.items():
        print(f"  {ind}: {cnt}")

entrant_summary = pd.DataFrame({
    'metric': agg_labels + ['Tech count', 'Org scope'],
    'entrant_mean': [entrant_postings[m].mean() for m in agg_metrics + ['tech_count', 'org_scope_count']],
    'continuer_mean': [continuer_postings[m].mean() for m in agg_metrics + ['tech_count', 'org_scope_count']],
})
entrant_summary['diff'] = entrant_summary['entrant_mean'] - entrant_summary['continuer_mean']
entrant_summary.to_csv(f'{OUT_TBL}/new_entrant_comparison.csv', index=False)

# ========== STEP 7: Aggregator vs direct employer patterns ==========
print("\n=== Step 7: Aggregator vs direct ===")
for era in ['2024', '2026']:
    subset = base[base['era'] == era]
    for agg_val, agg_label in [(True, 'Aggregator'), (False, 'Direct')]:
        ss = subset[subset['is_aggregator'] == agg_val]
        print(f"  {era} {agg_label}: n={len(ss)}, entry={ss['is_entry'].mean():.4f}, "
              f"AI={ss['has_ai'].mean():.4f}, desc_len={ss['desc_len'].mean():.0f}, "
              f"tech_count={ss['tech_count'].mean():.1f}")

# Aggregator change patterns for overlap set
change['is_aggregator'] = False  # need to fill from data
for co in change.index:
    co_data = panel[panel['company_name_canonical'] == co]
    change.loc[co, 'is_aggregator'] = co_data['is_aggregator'].any()

agg_change = change.groupby('is_aggregator')[cluster_features + ['d_log_n']].mean()
print("\nChange by aggregator status:")
print(agg_change.to_string())

# ========== SENSITIVITY: Aggregator exclusion ==========
print("\n=== Sensitivity: Aggregator exclusion ===")
base_no_agg = base[~base['is_aggregator']].copy()
all_2024_na = base_no_agg[base_no_agg['era'] == '2024']
all_2026_na = base_no_agg[base_no_agg['era'] == '2026']

# Recompute overlap without aggregators
co_era_na = base_no_agg.groupby(['company_name_canonical', 'era']).size().unstack(fill_value=0)
co_era_na.columns = ['n_2024', 'n_2026']
overlap_na = co_era_na[(co_era_na['n_2024'] >= 3) & (co_era_na['n_2026'] >= 3)]
print(f"Overlap companies (no aggregators): {len(overlap_na)} (was {len(overlap)} with aggregators)")

# Recompute decomposition without aggregators
overlap_names_na = set(overlap_na.index)
sens_decomp = []
for metric, label in zip(agg_metrics, agg_labels):
    ov24 = all_2024_na[all_2024_na['company_name_canonical'].isin(overlap_names_na)]
    ov26 = all_2026_na[all_2026_na['company_name_canonical'].isin(overlap_names_na)]

    N24 = len(all_2024_na)
    N26 = len(all_2026_na)

    w_ov24 = len(ov24) / N24 if N24 > 0 else 0
    w_ov26 = len(ov26) / N26 if N26 > 0 else 0
    w_ov_avg = (w_ov24 + w_ov26) / 2

    m_ov24 = ov24[metric].mean() if len(ov24) > 0 else 0
    m_ov26 = ov26[metric].mean() if len(ov26) > 0 else 0

    total = all_2026_na[metric].mean() - all_2024_na[metric].mean()
    within = w_ov_avg * (m_ov26 - m_ov24)
    composition = total - within
    within_pct = (within / abs(total) * 100) if total != 0 else np.nan

    sens_decomp.append({
        'metric': label,
        'total_change_no_agg': total,
        'within_no_agg': within,
        'within_pct_no_agg': within_pct,
        'composition_no_agg': composition,
    })
    print(f"  {label} (no agg): total={total:.4f}, within={within:.4f} ({within_pct:.1f}%), comp={composition:.4f}")

sens_decomp_df = pd.DataFrame(sens_decomp)
# Merge with main decomposition for comparison
sens_compare = formal_decomp_df[['metric', 'total_change', 'within_overlap', 'within_pct', 'composition']].merge(
    sens_decomp_df, on='metric')
sens_compare.to_csv(f'{OUT_TBL}/sensitivity_aggregator_decomposition.csv', index=False)

# ========== FIGURES ==========
print("\n=== Generating figures ===")

# Figure 1: Cluster profiles radar/heatmap
fig, ax = plt.subplots(figsize=(10, 6))
cluster_means = change.groupby(f'cluster_k{k_final}')[cluster_features].mean()
# Standardize for display
cluster_z = (cluster_means - cluster_means.mean()) / cluster_means.std()
feature_labels = ['Entry share', 'AI prevalence', 'Desc length', 'Tech count', 'Org scope']
sns.heatmap(cluster_z.values, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=feature_labels,
            yticklabels=[f'C{i} (n={int((change[f"cluster_k{k_final}"]==i).sum())})' for i in range(k_final)],
            ax=ax, vmin=-2, vmax=2)
ax.set_title(f'Company Strategy Clusters (k={k_final}): Standardized Change Profiles')
ax.set_xlabel('Change Metric')
ax.set_ylabel('Cluster')
plt.tight_layout()
plt.savefig(f'{OUT_FIG}/cluster_heatmap.png', dpi=150)
plt.close()

# Figure 2: Decomposition bar chart
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
for i, (metric, label) in enumerate(zip(agg_metrics, agg_labels)):
    row = formal_decomp_df[formal_decomp_df['metric'] == label].iloc[0]
    total = row['total_change']
    within = row['within_overlap']
    comp = row['composition']

    bars = axes[i].bar(['Total', 'Within\noverlap', 'Composition'], [total, within, comp],
                        color=['#2c3e50', '#2980b9', '#e74c3c'], alpha=0.8)
    axes[i].set_title(label)
    axes[i].axhline(0, color='gray', linewidth=0.5)
    axes[i].set_ylabel('Change' if i == 0 else '')

    # Add value labels
    for bar, val in zip(bars, [total, within, comp]):
        axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001 * np.sign(bar.get_height()),
                     f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)

fig.suptitle('Within-Company vs Compositional Decomposition', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_FIG}/decomposition_bars.png', dpi=150)
plt.close()

# Figure 3: Company scatter - AI change vs entry share change
fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.Set2(np.linspace(0, 1, k_final))
for cl in range(k_final):
    mask = change[f'cluster_k{k_final}'] == cl
    ax.scatter(change.loc[mask, 'd_ai_prevalence'], change.loc[mask, 'd_entry_share'],
               alpha=0.5, s=30, color=colors[cl], label=f'Cluster {cl} (n={mask.sum()})')
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('Change in AI Prevalence (2024 to 2026)')
ax.set_ylabel('Change in Entry Share (2024 to 2026)')
ax.set_title('Company-Level AI Adoption vs Entry Role Changes')
ax.legend(fontsize=8)
# Add correlation
r, p = stats.pearsonr(change['d_ai_prevalence'].dropna(), change['d_entry_share'].dropna())
ax.text(0.02, 0.98, f'r = {r:.3f}, p = {p:.3f}', transform=ax.transAxes,
        fontsize=9, va='top', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_FIG}/ai_vs_entry_scatter.png', dpi=150)
plt.close()

# Figure 4: Volume change by cluster
fig, ax = plt.subplots(figsize=(8, 5))
for cl in range(k_final):
    subset = change[change[f'cluster_k{k_final}'] == cl]
    ax.hist(subset['d_log_n'], bins=20, alpha=0.5, label=f'Cluster {cl} (n={len(subset)})')
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('Log change in posting volume')
ax.set_ylabel('Number of companies')
ax.set_title('Posting Volume Change by Strategy Cluster')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT_FIG}/volume_change_by_cluster.png', dpi=150)
plt.close()

print(f"\nT16 analysis complete. Outputs in {OUT_TBL}/ and {OUT_FIG}/")
print("Tables:", os.listdir(OUT_TBL))
print("Figures:", os.listdir(OUT_FIG))
