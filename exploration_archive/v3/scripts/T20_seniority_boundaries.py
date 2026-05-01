#!/usr/bin/env python3
"""T20: Seniority Boundary Clarity Analysis

Measures how sharp the boundaries between seniority levels are,
whether they blurred between periods, and what drives separation.
"""

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import os, re, json

PARQUET = 'data/unified.parquet'
TECH_MATRIX = 'exploration/artifacts/shared/swe_tech_matrix.parquet'
CLEANED_TEXT = 'exploration/artifacts/shared/swe_cleaned_text.parquet'
FIG_DIR = 'exploration/figures/T20'
TBL_DIR = 'exploration/tables/T20'

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

# ── Management, scope, and AI keyword lists (reuse from T11 patterns) ──

MANAGEMENT_TERMS = [
    r'\bmanag\w*', r'\bmentor\w*', r'\bcoach\w*', r'\bhir(?:e|ing)\b',
    r'\binterview\w*', r'\bgrow(?:th)?\b', r'\bdevelop\s+talent',
    r'\bperformance\s+review', r'\bcareer\s+develop', r'\b1[:\-]1\b',
    r'\bheadcount\b', r'\bdirect\s+report', r'\blead\s+(?:a\s+)?team',
    r'\bteam\s+lead', r'\bpeople\s+manage', r'\bstaff(?:ing)?\b'
]

ORG_SCOPE_TERMS = [
    r'\bstakeholder\w*', r'\bcross[\-\s]functional', r'\bownership\b',
    r'\bown\s+(?:the|end)', r'\bdrive\s+(?:the|key|business)',
    r'\binfluence\b', r'\bescalat\w*', r'\bstrategic\b', r'\broadmap\b',
    r'\bprioritiz\w*', r'\bresource\s+alloc', r'\bbudget\w*',
    r'\bbusiness\s+impact', r'\brevenue\b', r'\bproduct\s+strateg'
]

AI_TERMS = [
    r'\bmachine\s+learning\b', r'\bml\b', r'\bdeep\s+learning\b',
    r'\bneural\s+net\w*', r'\bllm\b', r'\blarge\s+language\b',
    r'\bgenerative\s+ai\b', r'\bgen\s*ai\b', r'\bai\s+agent',
    r'\bprompt\s+engineer', r'\brag\b', r'\bretrieval[\-\s]augment',
    r'\bfine[\-\s]tun', r'\btransformer\w*', r'\bgpt\b',
    r'\bchatbot\b', r'\bcopilot\b', r'\bai[\-\s](?:powered|driven|based)'
]

EDUCATION_PATTERNS = {
    'phd': r'\b(?:ph\.?d|doctorate|doctoral)\b',
    'ms': r'\b(?:master\'?s?|m\.?s\.?(?:\s|$)|msc)\b',
    'bs': r'\b(?:bachelor\'?s?|b\.?s\.?(?:\s|$)|bsc|undergraduate)\b'
}


def count_pattern_matches(text, patterns):
    """Count total regex matches from a list of patterns."""
    if not text:
        return 0
    text_lower = text.lower()
    count = 0
    for p in patterns:
        count += len(re.findall(p, text_lower))
    return count


def extract_education_ordinal(text):
    """Extract highest education level as ordinal: 0=none, 1=bs, 2=ms, 3=phd."""
    if not text:
        return 0
    text_lower = text.lower()
    if re.search(EDUCATION_PATTERNS['phd'], text_lower):
        return 3
    if re.search(EDUCATION_PATTERNS['ms'], text_lower):
        return 2
    if re.search(EDUCATION_PATTERNS['bs'], text_lower):
        return 1
    return 0


def density_per_1k(count, text_len):
    """Convert count to density per 1K characters."""
    if text_len == 0:
        return 0.0
    return count / text_len * 1000


# ── Step 1: Load data ──
print("Loading data...")
con = duckdb.connect()
con.execute("SET memory_limit='4GB'")

# Main table: SWE, LinkedIn, known seniority
df = con.execute("""
SELECT
    u.uid,
    u.period,
    u.source,
    u.seniority_final,
    u.seniority_3level,
    u.seniority_final_source,
    u.yoe_extracted,
    u.description_length,
    u.is_aggregator,
    COALESCE(c.description_cleaned, u.description_core, u.description) as text_for_analysis,
    LENGTH(COALESCE(c.description_cleaned, u.description_core, u.description)) as text_length
FROM read_parquet('{parquet}') u
LEFT JOIN read_parquet('{cleaned}') c ON u.uid = c.uid
WHERE u.source_platform = 'linkedin'
  AND u.is_english = true
  AND u.date_flag = 'ok'
  AND u.is_swe = true
  AND u.seniority_final != 'unknown'
""".format(parquet=PARQUET, cleaned=CLEANED_TEXT)).df()

# Tech matrix
tech_df = con.execute("""
SELECT * FROM read_parquet('{tech}')
""".format(tech=TECH_MATRIX)).df()

print(f"Loaded {len(df)} SWE rows with known seniority")
print(f"Tech matrix: {len(tech_df)} rows x {len(tech_df.columns)-1} tech cols")

# Merge tech info
ai_cols = [c for c in tech_df.columns if c.startswith('ai_')]
tech_df['tech_count'] = tech_df[[c for c in tech_df.columns if c != 'uid']].sum(axis=1)
tech_df['ai_mention'] = tech_df[ai_cols].any(axis=1).astype(int)
tech_df_subset = tech_df[['uid', 'tech_count', 'ai_mention']]

df = df.merge(tech_df_subset, on='uid', how='left')
df['tech_count'] = df['tech_count'].fillna(0)
df['ai_mention'] = df['ai_mention'].fillna(0)

# ── Step 2: Feature extraction ──
print("Extracting features...")

# Management language density
df['mgmt_count'] = df['text_for_analysis'].apply(lambda x: count_pattern_matches(x, MANAGEMENT_TERMS))
df['mgmt_density'] = df.apply(lambda r: density_per_1k(r['mgmt_count'], r['text_length']) if r['text_length'] else 0, axis=1)

# Org scope density
df['scope_count'] = df['text_for_analysis'].apply(lambda x: count_pattern_matches(x, ORG_SCOPE_TERMS))
df['scope_density'] = df.apply(lambda r: density_per_1k(r['scope_count'], r['text_length']) if r['text_length'] else 0, axis=1)

# Education ordinal
df['education_ordinal'] = df['text_for_analysis'].apply(extract_education_ordinal)

# YOE imputed (median where null)
yoe_median = df['yoe_extracted'].median()
df['yoe_imputed'] = df['yoe_extracted'].fillna(yoe_median)

# Description length normalized (log)
df['desc_length_log'] = np.log1p(df['text_length'].fillna(0))

print("Feature extraction complete.")
print(f"  YOE median (for imputation): {yoe_median}")
print(f"  Management density mean: {df['mgmt_density'].mean():.3f}")
print(f"  Scope density mean: {df['scope_density'].mean():.3f}")

# Define period groups
# Combine 2024-01 and 2024-04 as "2024", keep 2026-03 as "2026"
df['period_group'] = df['period'].map({
    '2024-01': '2024', '2024-04': '2024', '2026-03': '2026'
})

# Feature names for logistic regression
FEATURE_NAMES = [
    'yoe_imputed', 'tech_count', 'ai_mention',
    'scope_density', 'mgmt_density', 'desc_length_log',
    'education_ordinal'
]

# ── Step 3: Boundary discriminability ──
print("\n=== Boundary Discriminability Analysis ===")

SENIORITY_ORDER = ['entry', 'associate', 'mid-senior', 'director']
ADJACENT_PAIRS = [
    ('entry', 'associate'),
    ('associate', 'mid-senior'),
    ('mid-senior', 'director')
]

def run_boundary_analysis(data, pair, label):
    """Train L2 logistic regression, 5-fold stratified CV, return AUC and feature importances."""
    s1, s2 = pair
    subset = data[data['seniority_final'].isin([s1, s2])].copy()

    if len(subset) < 20:
        return None

    # Check class balance
    counts = subset['seniority_final'].value_counts()
    if counts.min() < 10:
        return None

    X = subset[FEATURE_NAMES].values
    y = (subset['seniority_final'] == s2).astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)

    # 5-fold stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    try:
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
    except Exception as e:
        print(f"  Warning: CV failed for {label}: {e}")
        return None

    # Fit on full data for feature importances
    model.fit(X_scaled, y)
    coefs = model.coef_[0]

    # Feature importances (absolute coefficient)
    importance = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'coefficient': coefs,
        'abs_coef': np.abs(coefs)
    }).sort_values('abs_coef', ascending=False)

    return {
        'pair': f"{s1} vs {s2}",
        'label': label,
        'n': len(subset),
        'n_class0': int(counts.get(s1, 0)),
        'n_class1': int(counts.get(s2, 0)),
        'auc_mean': scores.mean(),
        'auc_std': scores.std(),
        'auc_folds': scores.tolist(),
        'feature_importance': importance,
        'top5_features': importance.head(5)
    }


results = []
for period_group in ['2024', '2026']:
    pdata = df[df['period_group'] == period_group]
    for pair in ADJACENT_PAIRS:
        label = f"{pair[0]}_vs_{pair[1]}_{period_group}"
        result = run_boundary_analysis(pdata, pair, label)
        if result:
            result['period_group'] = period_group
            results.append(result)
            print(f"  {label}: AUC = {result['auc_mean']:.3f} +/- {result['auc_std']:.3f} (n={result['n']})")
        else:
            print(f"  {label}: SKIPPED (insufficient data)")

# ── Step 4: Compare AUCs between periods ──
print("\n=== AUC Comparison ===")
auc_comparison = []
for pair in ADJACENT_PAIRS:
    r2024 = next((r for r in results if r['pair'] == f"{pair[0]} vs {pair[1]}" and r['period_group'] == '2024'), None)
    r2026 = next((r for r in results if r['pair'] == f"{pair[0]} vs {pair[1]}" and r['period_group'] == '2026'), None)

    row = {
        'boundary': f"{pair[0]} vs {pair[1]}",
        'auc_2024': r2024['auc_mean'] if r2024 else None,
        'auc_2024_std': r2024['auc_std'] if r2024 else None,
        'n_2024': r2024['n'] if r2024 else 0,
        'auc_2026': r2026['auc_mean'] if r2026 else None,
        'auc_2026_std': r2026['auc_std'] if r2026 else None,
        'n_2026': r2026['n'] if r2026 else 0,
    }
    if row['auc_2024'] and row['auc_2026']:
        row['auc_change'] = row['auc_2026'] - row['auc_2024']
        row['interpretation'] = 'BLURRED' if row['auc_change'] < -0.02 else ('SHARPENED' if row['auc_change'] > 0.02 else 'STABLE')
    else:
        row['auc_change'] = None
        row['interpretation'] = 'N/A'

    auc_comparison.append(row)
    print(f"  {row['boundary']}: 2024 AUC={row['auc_2024']}, 2026 AUC={row['auc_2026']}, Change={row['auc_change']}, {row['interpretation']}")

auc_df = pd.DataFrame(auc_comparison)
auc_df.to_csv(f'{TBL_DIR}/auc_comparison.csv', index=False)

# ── Step 5: Feature importance comparison ──
print("\n=== Feature Importance Changes ===")
fi_rows = []
for r in results:
    for _, frow in r['feature_importance'].iterrows():
        fi_rows.append({
            'pair': r['pair'],
            'period_group': r['period_group'],
            'feature': frow['feature'],
            'coefficient': frow['coefficient'],
            'abs_coef': frow['abs_coef'],
            'rank': list(r['feature_importance']['feature']).index(frow['feature']) + 1
        })

fi_df = pd.DataFrame(fi_rows)
fi_df.to_csv(f'{TBL_DIR}/feature_importance.csv', index=False)

# Print top features per boundary per period
for pair in ADJACENT_PAIRS:
    pair_label = f"{pair[0]} vs {pair[1]}"
    print(f"\n  {pair_label}:")
    for pg in ['2024', '2026']:
        r = next((r for r in results if r['pair'] == pair_label and r['period_group'] == pg), None)
        if r:
            print(f"    {pg} top 5: {', '.join(r['top5_features']['feature'].tolist())}")

# ── Step 6: Sensitivity (a) - Aggregator exclusion ──
print("\n=== Sensitivity: Aggregator Exclusion ===")
df_noagg = df[~df['is_aggregator']].copy()

results_noagg = []
for period_group in ['2024', '2026']:
    pdata = df_noagg[df_noagg['period_group'] == period_group]
    for pair in ADJACENT_PAIRS:
        label = f"{pair[0]}_vs_{pair[1]}_{period_group}_noagg"
        result = run_boundary_analysis(pdata, pair, label)
        if result:
            result['period_group'] = period_group
            results_noagg.append(result)
            print(f"  {label}: AUC = {result['auc_mean']:.3f} +/- {result['auc_std']:.3f} (n={result['n']})")

# Build sensitivity comparison
sens_rows = []
for pair in ADJACENT_PAIRS:
    pair_label = f"{pair[0]} vs {pair[1]}"
    for pg in ['2024', '2026']:
        r_all = next((r for r in results if r['pair'] == pair_label and r['period_group'] == pg), None)
        r_noagg = next((r for r in results_noagg if r['pair'] == pair_label and r['period_group'] == pg), None)
        sens_rows.append({
            'boundary': pair_label,
            'period': pg,
            'auc_all': r_all['auc_mean'] if r_all else None,
            'auc_noagg': r_noagg['auc_mean'] if r_noagg else None,
            'diff': (r_noagg['auc_mean'] - r_all['auc_mean']) if (r_all and r_noagg) else None
        })

sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(f'{TBL_DIR}/sensitivity_aggregator.csv', index=False)
print(sens_df.to_string())

# ── Step 7: Sensitivity (c) - Seniority operationalization ──
print("\n=== Sensitivity: Seniority Operationalization ===")
# Use seniority_3level instead of seniority_final
PAIRS_3LEVEL = [
    ('junior', 'mid'),
    ('mid', 'senior')
]

results_3level = []
for period_group in ['2024', '2026']:
    pdata = df[df['period_group'] == period_group].copy()
    for pair in PAIRS_3LEVEL:
        s1, s2 = pair
        subset = pdata[pdata['seniority_3level'].isin([s1, s2])].copy()

        if len(subset) < 20:
            print(f"  {s1}_vs_{s2}_{period_group}_3level: SKIPPED")
            continue

        counts = subset['seniority_3level'].value_counts()
        if counts.min() < 10:
            print(f"  {s1}_vs_{s2}_{period_group}_3level: SKIPPED (min count {counts.min()})")
            continue

        X = subset[FEATURE_NAMES].values
        y = (subset['seniority_3level'] == s2).astype(int).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        try:
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
            results_3level.append({
                'pair': f"{s1} vs {s2}",
                'period_group': period_group,
                'auc_mean': scores.mean(),
                'auc_std': scores.std(),
                'n': len(subset)
            })
            print(f"  {s1}_vs_{s2}_{period_group}_3level: AUC = {scores.mean():.3f} +/- {scores.std():.3f} (n={len(subset)})")
        except Exception as e:
            print(f"  Warning: {e}")

r3_df = pd.DataFrame(results_3level)
r3_df.to_csv(f'{TBL_DIR}/sensitivity_3level.csv', index=False)

# ── Step 8: "Missing middle" analysis ──
print("\n=== Missing Middle Analysis ===")
# Compare associate feature profiles to entry and mid-senior
profile_rows = []
for pg in ['2024', '2026']:
    for sen in SENIORITY_ORDER:
        subset = df[(df['period_group'] == pg) & (df['seniority_final'] == sen)]
        if len(subset) < 10:
            continue
        row = {'period': pg, 'seniority': sen, 'n': len(subset)}
        for feat in FEATURE_NAMES:
            row[f'{feat}_mean'] = subset[feat].mean()
            row[f'{feat}_std'] = subset[feat].std()
        profile_rows.append(row)

profile_df = pd.DataFrame(profile_rows)
profile_df.to_csv(f'{TBL_DIR}/seniority_profiles.csv', index=False)

# Compute distances: associate-to-entry vs associate-to-midsenior
print("\nAssociate positioning (Euclidean distance in feature space):")
for pg in ['2024', '2026']:
    profiles = {}
    for sen in ['entry', 'associate', 'mid-senior']:
        row = profile_df[(profile_df['period'] == pg) & (profile_df['seniority'] == sen)]
        if len(row) == 0:
            continue
        profiles[sen] = np.array([row.iloc[0][f'{f}_mean'] for f in FEATURE_NAMES])

    if 'associate' in profiles:
        if 'entry' in profiles:
            d_entry = np.linalg.norm(profiles['associate'] - profiles['entry'])
        else:
            d_entry = float('nan')
        if 'mid-senior' in profiles:
            d_mid = np.linalg.norm(profiles['associate'] - profiles['mid-senior'])
        else:
            d_mid = float('nan')
        print(f"  {pg}: Associate-to-Entry = {d_entry:.3f}, Associate-to-MidSenior = {d_mid:.3f}")
        if not np.isnan(d_entry) and not np.isnan(d_mid):
            ratio = d_entry / (d_entry + d_mid)
            print(f"    Relative position (0=entry, 1=mid-senior): {ratio:.3f}")

# ── Step 9: Full similarity heatmap ──
print("\n=== Feature Profile Heatmap ===")

# Prepare heatmap data
heatmap_labels = []
heatmap_data = []
for pg in ['2024', '2026']:
    for sen in SENIORITY_ORDER:
        row = profile_df[(profile_df['period'] == pg) & (profile_df['seniority'] == sen)]
        if len(row) == 0:
            continue
        vals = [row.iloc[0][f'{f}_mean'] for f in FEATURE_NAMES]
        heatmap_data.append(vals)
        heatmap_labels.append(f"{sen}\n{pg}")

if heatmap_data:
    heatmap_arr = np.array(heatmap_data)
    # Standardize columns for heatmap
    col_means = heatmap_arr.mean(axis=0)
    col_stds = heatmap_arr.std(axis=0)
    col_stds[col_stds == 0] = 1
    heatmap_norm = (heatmap_arr - col_means) / col_stds

    feature_labels = ['YOE', 'Tech Count', 'AI Mention', 'Scope Density',
                      'Mgmt Density', 'Desc Length (log)', 'Education']

    fig, ax = plt.subplots(figsize=(10, 7))
    im = sns.heatmap(heatmap_norm, xticklabels=feature_labels, yticklabels=heatmap_labels,
                     cmap='RdYlBu_r', center=0, annot=True, fmt='.2f',
                     linewidths=0.5, ax=ax)
    ax.set_title('Seniority x Period Feature Profiles (z-scored)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Seniority Level / Period')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/feature_profile_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved feature_profile_heatmap.png")

# ── Step 10: AUC comparison chart ──
fig, ax = plt.subplots(figsize=(10, 6))
boundaries = [f"{p[0]} vs\n{p[1]}" for p in ADJACENT_PAIRS]
x = np.arange(len(boundaries))
width = 0.35

aucs_2024 = []
aucs_2026 = []
errs_2024 = []
errs_2026 = []

for pair in ADJACENT_PAIRS:
    pair_label = f"{pair[0]} vs {pair[1]}"
    r2024 = next((r for r in results if r['pair'] == pair_label and r['period_group'] == '2024'), None)
    r2026 = next((r for r in results if r['pair'] == pair_label and r['period_group'] == '2026'), None)
    aucs_2024.append(r2024['auc_mean'] if r2024 else 0)
    aucs_2026.append(r2026['auc_mean'] if r2026 else 0)
    errs_2024.append(r2024['auc_std'] if r2024 else 0)
    errs_2026.append(r2026['auc_std'] if r2026 else 0)

bars1 = ax.bar(x - width/2, aucs_2024, width, yerr=errs_2024, label='2024',
               color='#4ECDC4', alpha=0.85, capsize=4)
bars2 = ax.bar(x + width/2, aucs_2026, width, yerr=errs_2026, label='2026',
               color='#FF6B6B', alpha=0.85, capsize=4)

ax.set_ylabel('AUC (5-fold CV)', fontsize=12)
ax.set_title('Seniority Boundary Discriminability by Period', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(boundaries, fontsize=11)
ax.legend(fontsize=12)
ax.set_ylim(0.5, 1.0)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')

# Add AUC values on bars
for bar, val in zip(bars1, aucs_2024):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, aucs_2026):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/auc_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved auc_comparison.png")

# ── Step 11: Feature importance comparison chart ──
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
feature_display = {
    'yoe_imputed': 'YOE',
    'tech_count': 'Tech Count',
    'ai_mention': 'AI Mention',
    'scope_density': 'Scope Density',
    'mgmt_density': 'Mgmt Density',
    'desc_length_log': 'Desc Length',
    'education_ordinal': 'Education'
}

for idx, pair in enumerate(ADJACENT_PAIRS):
    ax = axes[idx]
    pair_label = f"{pair[0]} vs {pair[1]}"

    for pg, color, offset in [('2024', '#4ECDC4', -0.15), ('2026', '#FF6B6B', 0.15)]:
        r = next((r for r in results if r['pair'] == pair_label and r['period_group'] == pg), None)
        if not r:
            continue
        fi = r['feature_importance']
        y_pos = np.arange(len(FEATURE_NAMES))
        vals = [fi[fi['feature'] == f]['coefficient'].values[0] if f in fi['feature'].values else 0
                for f in FEATURE_NAMES]
        ax.barh(y_pos + offset, vals, 0.28, label=pg, color=color, alpha=0.85)

    ax.set_yticks(np.arange(len(FEATURE_NAMES)))
    ax.set_yticklabels([feature_display.get(f, f) for f in FEATURE_NAMES])
    ax.set_title(f'{pair_label}', fontsize=11, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Log-Reg Coefficient')
    if idx == 0:
        ax.legend(fontsize=10)

plt.suptitle('Feature Importance for Seniority Boundaries (L2 Logistic Regression)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/feature_importance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved feature_importance_comparison.png")

# ── Step 12: Missing middle visualization ──
fig, axes = plt.subplots(1, len(FEATURE_NAMES), figsize=(20, 5))

for i, feat in enumerate(FEATURE_NAMES):
    ax = axes[i]
    for pg, marker, color in [('2024', 'o', '#4ECDC4'), ('2026', 's', '#FF6B6B')]:
        vals = []
        labels = []
        for sen in SENIORITY_ORDER:
            row = profile_df[(profile_df['period'] == pg) & (profile_df['seniority'] == sen)]
            if len(row) > 0:
                vals.append(row.iloc[0][f'{feat}_mean'])
                labels.append(sen)
        if vals:
            ax.plot(range(len(vals)), vals, marker=marker, color=color, label=pg, linewidth=2, markersize=8)

    ax.set_xticks(range(len(SENIORITY_ORDER)))
    ax.set_xticklabels([s[:3] for s in SENIORITY_ORDER], fontsize=8)
    ax.set_title(feature_display.get(feat, feat), fontsize=9, fontweight='bold')
    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle('Feature Values Across Seniority Levels: "Missing Middle" Analysis',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/missing_middle.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved missing_middle.png")

# ── Summary stats for report ──
print("\n\n=== SUMMARY FOR REPORT ===")
print(f"\nSample sizes by period x seniority:")
for pg in ['2024', '2026']:
    for sen in SENIORITY_ORDER:
        n = len(df[(df['period_group'] == pg) & (df['seniority_final'] == sen)])
        if n > 0:
            print(f"  {pg} {sen}: {n}")

print(f"\nAUC comparison table:")
print(auc_df.to_string(index=False))

print(f"\nFeature profiles:")
print(profile_df.to_string(index=False))

print("\nDone! All outputs saved.")
