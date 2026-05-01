"""T17: Geographic market structure.

Metro-level metrics, change rankings, hub vs non-hub patterns, AI↔entry
correlations, remote share, archetype geographic distribution.
"""
import os
import re
import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('exploration/figures/T17', exist_ok=True)
os.makedirs('exploration/tables/T17', exist_ok=True)

con = duckdb.connect()

BASE_FILTER = (
    "source_platform = 'linkedin' AND is_english = true "
    "AND date_flag = 'ok' AND is_swe = true"
)

AI_PAT = re.compile(
    r'\b(agentic|\bllm\b|\brag\b|copilot|claude|cursor|langchain|langgraph|'
    r'mcp|openai|chatgpt|gpt-?\d|ai[- ]agent|multi[- ]agent|gen[- ]?ai|'
    r'generative ai|large language model|prompt engineering)\b',
    re.IGNORECASE,
)
SCOPE_PAT = re.compile(r'\b(end[- ]to[- ]end|ownership|own the|cross[- ]functional|drive initiatives?|technical leader(?:ship)?|lead projects?|architect solutions?)\b', re.I)

assert AI_PAT.search("Build agentic llm pipelines"), "ai broken"
assert SCOPE_PAT.search("end-to-end ownership"), "scope broken"

# -------------------- Load data --------------------
print("[1/7] Loading data...")
q = f"""
SELECT
  uid,
  source,
  company_name_canonical,
  company_name_effective,
  is_aggregator,
  metro_area,
  description,
  description_core_llm,
  description_hash,
  is_remote,
  seniority_native,
  seniority_final,
  seniority_llm,
  llm_classification_coverage,
  yoe_extracted,
  CASE
    WHEN llm_classification_coverage = 'labeled'         THEN seniority_llm
    WHEN llm_classification_coverage = 'rule_sufficient' THEN seniority_final
    ELSE NULL
  END AS seniority_best
FROM 'data/unified.parquet'
WHERE {BASE_FILTER}
"""
df = con.execute(q).fetchdf()
print(f"  rows: {len(df):,}")

df['company'] = df['company_name_canonical'].fillna(df['company_name_effective'])
df = df[df['metro_area'].notna() & (df['metro_area'] != '')]

# Dedup within company by description_hash
df_sorted = df.sort_values('uid')
null_hash = df_sorted['description_hash'].isna()
mask = (~df_sorted.duplicated(subset=['company', 'description_hash'], keep='first')) | null_hash
df = df_sorted[mask].copy()
print(f"  after dedup: {len(df):,}")

# Pattern matching
df['text'] = df['description_core_llm'].fillna(df['description']).fillna('')
df['has_ai'] = df['text'].str.contains(AI_PAT, na=False)
df['has_scope'] = df['text'].str.contains(SCOPE_PAT, na=False)
df['desc_len'] = df['description'].fillna('').str.len()
df['is_entry_best'] = df['seniority_best'] == 'entry'
df['yoe_le2'] = (df['yoe_extracted'] <= 2) & df['yoe_extracted'].notna()

# Period flag
df['period_2024'] = df['source'].isin(['kaggle_arshkon', 'kaggle_asaniczka'])
df['period_2026'] = df['source'] == 'scraped'
df['period'] = np.where(df['period_2026'], '2026', '2024')

# Tech count
tech = con.execute(
    "SELECT * FROM 'exploration/artifacts/shared/swe_tech_matrix.parquet'"
).fetchdf()
tech_cols = [c for c in tech.columns if c not in ('uid', 'c_cpp', 'csharp')]
tech['tech_count'] = tech[tech_cols[1:]].sum(axis=1) if len(tech_cols) > 1 else 0
df = df.merge(tech[['uid', 'tech_count']], on='uid', how='left')
df['tech_count'] = df['tech_count'].fillna(0)

# -------------------- Metro-level metrics --------------------
print("[2/7] Metro aggregates...")

def agg_metro(sub):
    n = len(sub)
    n_best = sub['seniority_best'].notna().sum()
    n_yoe = sub['yoe_extracted'].notna().sum()
    return pd.Series({
        'n': n,
        'entry_share_best': sub['is_entry_best'].sum() / n_best if n_best else np.nan,
        'entry_share_yoe': sub['yoe_le2'].sum() / n_yoe if n_yoe else np.nan,
        'ai_rate': sub['has_ai'].mean(),
        'scope_rate': sub['has_scope'].mean(),
        'median_desc_len': sub['desc_len'].median(),
        'median_tech': sub['tech_count'].median(),
        'remote_share': sub['is_remote'].mean() if 'is_remote' in sub else np.nan,
    })

# Pooled 2024 & scraped 2026
metro_2024 = df[df['period'] == '2024'].groupby('metro_area').apply(agg_metro, include_groups=False)
metro_2026 = df[df['period'] == '2026'].groupby('metro_area').apply(agg_metro, include_groups=False)

# Also arshkon-only for native-entry ablation
metro_2024_arsh = df[df['source'] == 'kaggle_arshkon'].groupby('metro_area').apply(
    lambda s: pd.Series({
        'n': len(s),
        'entry_share_native': (s['seniority_native'] == 'entry').sum() / len(s),
    }),
    include_groups=False,
)

# Filter: >=50 SWE postings in both periods
metros_keep = metro_2024.index.intersection(metro_2026.index)
metro_2024_k = metro_2024.loc[metros_keep]
metro_2026_k = metro_2026.loc[metros_keep]
keep_mask = (metro_2024_k['n'] >= 50) & (metro_2026_k['n'] >= 50)
metro_2024_k = metro_2024_k[keep_mask]
metro_2026_k = metro_2026_k[keep_mask]

print(f"  metros with >=50 in both periods: {len(metro_2024_k)}")

metro_delta = metro_2026_k - metro_2024_k
metro_delta = metro_delta.drop(columns=['n'])
metro_delta['n_2024'] = metro_2024_k['n']
metro_delta['n_2026'] = metro_2026_k['n']
metro_delta.to_csv('exploration/tables/T17/metro_changes.csv')

metro_both = metro_2024_k.add_suffix('_2024').join(metro_2026_k.add_suffix('_2026'))
metro_both.to_csv('exploration/tables/T17/metro_metrics_both_periods.csv')
print(f"  ΔAI median: {metro_delta['ai_rate'].median():.4f}")
print(f"  ΔYOE entry median: {metro_delta['entry_share_yoe'].median():.4f}")

# -------------------- Rank metros --------------------
print("[3/7] Ranking metros...")
rank_cols = ['entry_share_yoe', 'entry_share_best', 'ai_rate', 'scope_rate', 'median_desc_len', 'median_tech']
for col in rank_cols:
    top5 = metro_delta[col].sort_values(ascending=False).head(5)
    bot5 = metro_delta[col].sort_values(ascending=True).head(5)
    print(f"  Δ{col}:")
    print(f"    top: {top5.to_dict()}")
    print(f"    bot: {bot5.to_dict()}")

# -------------------- Hub vs non-hub --------------------
print("[4/7] Hub vs non-hub...")
hubs = ['San Francisco-Oakland-Berkeley, CA', 'New York-Newark-Jersey City, NY-NJ-PA',
        'Seattle-Tacoma-Bellevue, WA', 'Austin-Round Rock-Georgetown, TX',
        'San Jose-Sunnyvale-Santa Clara, CA']
# approximate match
def is_hub(m):
    return any(h.split(',')[0].split('-')[0].lower() in m.lower() for h in hubs)

metro_delta['is_hub'] = metro_delta.index.map(is_hub)
hub_cmp = metro_delta.groupby('is_hub')[rank_cols].mean()
hub_cmp.to_csv('exploration/tables/T17/hub_vs_nonhub.csv')
print(hub_cmp)

# -------------------- Metro-level correlation --------------------
print("[5/7] Metro-level correlations...")
from scipy.stats import pearsonr, spearmanr

corr_pairs = [
    ('ai_rate', 'entry_share_yoe'),
    ('ai_rate', 'entry_share_best'),
    ('ai_rate', 'scope_rate'),
    ('ai_rate', 'median_desc_len'),
    ('entry_share_yoe', 'scope_rate'),
]
corr_rows = []
for a, b in corr_pairs:
    d = metro_delta[[a, b]].dropna()
    if len(d) >= 5:
        pr = pearsonr(d[a], d[b])
        sr = spearmanr(d[a], d[b])
        corr_rows.append({'x': a, 'y': b, 'pearson_r': pr.statistic, 'pearson_p': pr.pvalue,
                          'spearman_r': sr.correlation, 'spearman_p': sr.pvalue, 'n': len(d)})
corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv('exploration/tables/T17/metro_correlations.csv', index=False)
print(corr_df.to_string(index=False))

# Scatter: AI change vs entry change
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(metro_delta['ai_rate'], metro_delta['entry_share_yoe'])
for m, row in metro_delta.iterrows():
    short = m.split(',')[0][:20]
    axes[0].annotate(short, (row['ai_rate'], row['entry_share_yoe']), fontsize=6)
axes[0].set_xlabel('Δ AI rate')
axes[0].set_ylabel('Δ entry share (YOE<=2)')
axes[0].set_title('Metro: AI vs entry (YOE)')
axes[0].axhline(0, color='red', lw=0.5)
axes[0].axvline(0, color='red', lw=0.5)

axes[1].scatter(metro_delta['ai_rate'], metro_delta['scope_rate'])
for m, row in metro_delta.iterrows():
    short = m.split(',')[0][:20]
    axes[1].annotate(short, (row['ai_rate'], row['scope_rate']), fontsize=6)
axes[1].set_xlabel('Δ AI rate')
axes[1].set_ylabel('Δ scope rate')
axes[1].set_title('Metro: AI vs scope')
axes[1].axhline(0, color='red', lw=0.5)
axes[1].axvline(0, color='red', lw=0.5)
plt.tight_layout()
plt.savefig('exploration/figures/T17/metro_correlations.png', dpi=120)
plt.close()

# -------------------- Remote share --------------------
print("[6/7] Remote share...")
# DATA CAVEAT: scraped remote rows have metro_area=NULL (geographic routing drops
# them). Compute corpus-level remote share & AI/scope breakdown outside the metro
# panel; report per-metro remote share will always be ~0 because metro-assigned
# rows exclude remote.
# Reload unfiltered-by-metro slice
q_rem = f"""
SELECT source, is_remote, is_aggregator,
  CASE WHEN description_core_llm IS NOT NULL THEN description_core_llm ELSE description END AS txt
FROM 'data/unified.parquet'
WHERE {BASE_FILTER}
"""
rem_df = con.execute(q_rem).fetchdf()
rem_df['period'] = np.where(rem_df['source'] == 'scraped', '2026', '2024')
rem_df['has_ai'] = rem_df['txt'].fillna('').str.contains(AI_PAT, na=False)
rem_df['has_scope'] = rem_df['txt'].fillna('').str.contains(SCOPE_PAT, na=False)
rem_summary = rem_df.groupby(['period', 'is_remote']).agg(
    n=('source', 'count'),
    ai_rate=('has_ai', 'mean'),
    scope_rate=('has_scope', 'mean'),
)
print(rem_summary)
rem_summary.to_csv('exploration/tables/T17/remote_vs_onsite.csv')
# Share remote by period
share_rem = rem_df.groupby('period')['is_remote'].mean()
print(f"  remote share by period:\n{share_rem}")
share_rem.to_csv('exploration/tables/T17/remote_share_by_period.csv')

# -------------------- Archetype geographic distribution --------------------
print("[7/7] Archetype geographic distribution...")
arch = con.execute(
    "SELECT uid, archetype_name FROM 'exploration/artifacts/shared/swe_archetype_labels.parquet'"
).fetchdf()
df_arch = df.merge(arch, on='uid', how='inner')
print(f"  archetype-labeled rows in metro panel: {len(df_arch):,}")

# Archetype by metro, period
arch_metro_2024 = df_arch[df_arch['period'] == '2024'].groupby(['metro_area', 'archetype_name']).size().unstack(fill_value=0)
arch_metro_2026 = df_arch[df_arch['period'] == '2026'].groupby(['metro_area', 'archetype_name']).size().unstack(fill_value=0)

arch_metro_2024_pct = arch_metro_2024.div(arch_metro_2024.sum(axis=1), axis=0)
arch_metro_2026_pct = arch_metro_2026.div(arch_metro_2026.sum(axis=1), axis=0)

# Keep metros with >=30 archetype-labeled rows in each period
keep_arch = arch_metro_2024.sum(axis=1).ge(30) & arch_metro_2026.sum(axis=1).ge(30)
common = arch_metro_2024_pct.index.intersection(arch_metro_2026_pct.index)
keep_idx = [m for m in common if keep_arch.get(m, False) and arch_metro_2026.sum(axis=1).get(m, 0) >= 30]
print(f"  metros with >=30 archetype labels both periods: {len(keep_idx)}")

arch_metro_2026_pct_k = arch_metro_2026_pct.loc[keep_idx]
arch_metro_2024_pct_k = arch_metro_2024_pct.loc[keep_idx]

arch_metro_2026_pct_k.to_csv('exploration/tables/T17/archetype_by_metro_2026.csv')
arch_metro_2024_pct_k.to_csv('exploration/tables/T17/archetype_by_metro_2024.csv')

# Which metro has the highest AI/ML share?
if 'ai_ml_genai' in arch_metro_2026_pct_k.columns or any('ai' in c.lower() or 'ml' in c.lower() for c in arch_metro_2026_pct_k.columns):
    ai_col_candidates = [c for c in arch_metro_2026_pct_k.columns if 'ai' in c.lower() or 'ml' in c.lower() or 'genai' in c.lower()]
    print(f"  AI-related archetypes: {ai_col_candidates}")
    if ai_col_candidates:
        ai_col = ai_col_candidates[0]
        top_ai = arch_metro_2026_pct_k[ai_col].sort_values(ascending=False).head(10)
        print(f"  top metros for {ai_col}:\n{top_ai}")

# Change in archetype mix by metro
common_archs = arch_metro_2024_pct_k.columns.intersection(arch_metro_2026_pct_k.columns)
arch_metro_delta = arch_metro_2026_pct_k[common_archs] - arch_metro_2024_pct_k[common_archs]
arch_metro_delta.to_csv('exploration/tables/T17/archetype_delta_by_metro.csv')

# -------------------- Heatmap --------------------
print("  heatmap...")
heat_cols = ['entry_share_yoe', 'entry_share_best', 'ai_rate', 'scope_rate', 'median_desc_len', 'median_tech']
heat_data = metro_delta[heat_cols].copy()
# Short metro names
heat_data.index = [m.split(',')[0][:25] for m in heat_data.index]

# Normalize each column to [-1,1] for visual comparability
norm = heat_data.copy()
for c in heat_cols:
    vmax = max(abs(norm[c].min()), abs(norm[c].max()))
    if vmax > 0:
        norm[c] = norm[c] / vmax

fig, ax = plt.subplots(figsize=(10, max(6, len(norm) * 0.25)))
sns.heatmap(norm, cmap='RdBu_r', center=0, annot=heat_data.round(3), fmt='', cbar_kws={'label': 'normalized Δ'}, ax=ax)
ax.set_title('Metro changes 2024→2026 (normalized; raw values annotated)')
plt.tight_layout()
plt.savefig('exploration/figures/T17/metro_heatmap.png', dpi=120)
plt.close()

# -------------------- Sensitivity: aggregator exclusion --------------------
print("  sensitivity: no aggregators...")
df_noagg = df[df['is_aggregator'] != True]
metro_2024_na = df_noagg[df_noagg['period'] == '2024'].groupby('metro_area').apply(agg_metro, include_groups=False)
metro_2026_na = df_noagg[df_noagg['period'] == '2026'].groupby('metro_area').apply(agg_metro, include_groups=False)
common = metro_2024_na.index.intersection(metro_2026_na.index)
keep = (metro_2024_na.loc[common]['n'] >= 50) & (metro_2026_na.loc[common]['n'] >= 50)
common_k = common[keep]
delta_na = metro_2026_na.loc[common_k] - metro_2024_na.loc[common_k]
print(f"  no-agg metros: {len(common_k)}")
print(f"  ΔAI median (no-agg): {delta_na['ai_rate'].median():.4f}")
print(f"  ΔYOE entry median (no-agg): {delta_na['entry_share_yoe'].median():.4f}")
delta_na.drop(columns=['n']).to_csv('exploration/tables/T17/metro_changes_no_aggregator.csv')

# NYC caveat check
print("  NYC coverage check:")
nyc_mask = df['metro_area'].str.contains('New York', na=False)
nyc_counts = df[nyc_mask].groupby('source').size()
print(f"    NYC by source:\n{nyc_counts}")

print("\nT17 done.")
