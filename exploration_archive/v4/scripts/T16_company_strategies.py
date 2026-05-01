"""T16: Company hiring strategy typology.

Focus: entry-poster concentration, per-company change metrics on overlap panel,
k-means strategy clustering, within/between decomposition, new entrants.

SWE + LinkedIn + is_english + date_flag=ok. Primary period=scraped (2026) vs
arshkon baseline (2024, since asaniczka has no native entry labels). Pooled
2024 used only for combined-column/YOE metrics where asaniczka is admissible.
"""
import os
import re
import numpy as np
import pandas as pd
import duckdb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

os.makedirs('exploration/figures/T16', exist_ok=True)
os.makedirs('exploration/tables/T16', exist_ok=True)

con = duckdb.connect()

BASE_FILTER = (
    "source_platform = 'linkedin' AND is_english = true "
    "AND date_flag = 'ok' AND is_swe = true"
)

# -------------------- Assertions for regex --------------------
# Validated AI keywords per V1: agentic, rag, llm, copilot, claude, cursor,
# 'ai agent', 'multi-agent' are clean; bare 'agent' is contaminated.
AI_PAT = re.compile(
    r'\b(agentic|\bllm\b|\brag\b|copilot|claude|cursor|langchain|langgraph|'
    r'mcp|openai|chatgpt|gpt-?\d|ai[- ]agent|multi[- ]agent|gen[- ]?ai|'
    r'generative ai|large language model|prompt engineering)\b',
    re.IGNORECASE,
)

# Mentoring (strict): "mentor <target>"
MENTOR_PAT = re.compile(r'\bmentor(?:ing|s|ed)?\s+(?:junior|engineers?|team|others|peers|interns?|new hires?)\b', re.I)

# Scope/ownership
SCOPE_PAT = re.compile(r'\b(end[- ]to[- ]end|ownership|own the|cross[- ]functional|drive initiatives?|technical leader(?:ship)?|lead projects?|architect solutions?)\b', re.I)

# Smoke tests
assert AI_PAT.search("We use LLMs and RAG pipelines."), "AI pattern broken"
assert AI_PAT.search("Build agentic workflows with Claude."), "AI agentic broken"
assert not AI_PAT.search("Insurance agent for the firm"), "Should not match bare agent"
assert MENTOR_PAT.search("mentoring junior engineers"), "mentor pattern broken"
assert not MENTOR_PAT.search("mentoring culture"), "should not match mentoring alone"
assert SCOPE_PAT.search("own the end-to-end workflow"), "scope pattern broken"
assert SCOPE_PAT.search("Lead projects across teams"), "scope lead broken"

# -------------------- Load base data --------------------
print("[1/9] Loading base data...")

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
  seniority_native,
  seniority_final,
  seniority_llm,
  seniority_imputed,
  llm_classification_coverage,
  yoe_extracted,
  company_industry,
  CASE
    WHEN llm_classification_coverage = 'labeled'         THEN seniority_llm
    WHEN llm_classification_coverage = 'rule_sufficient' THEN seniority_final
    ELSE NULL
  END AS seniority_best
FROM 'data/unified.parquet'
WHERE {BASE_FILTER}
"""
df = con.execute(q).fetchdf()
print(f"  total SWE rows: {len(df):,}")

# Canonical company fallback
df['company'] = df['company_name_canonical'].fillna(df['company_name_effective'])
df = df[df['company'].notna() & (df['company'].str.strip() != '')]

# Source → period mapping
df['period'] = np.where(df['source'] == 'scraped', '2026', '2024')
df['source_is_arshkon'] = df['source'] == 'kaggle_arshkon'

# Text for pattern matching
df['text'] = df['description_core_llm'].fillna(df['description']).fillna('')

# -------------------- Step 1: dedup within company --------------------
print("[2/9] Dedup by description_hash within company...")
raw_n = len(df)
# Keep first occurrence per (company, description_hash) where hash is non-null
df_sorted = df.sort_values('uid')
# Rows with null hash treated as unique (keep all)
null_hash = df_sorted['description_hash'].isna()
dedup_keep = (~df_sorted.duplicated(subset=['company', 'description_hash'], keep='first')) | null_hash
df_dedup = df_sorted[dedup_keep].copy()
print(f"  raw={raw_n:,}  dedup={len(df_dedup):,}  collapsed={raw_n - len(df_dedup):,}")

# Dedup impact by company
dup_counts = (
    df.assign(dup=~dedup_keep.reindex(df.index, fill_value=False).values)
    .groupby('company', as_index=False)
    .agg(raw=('uid', 'count'), dup_collapsed=('dup', 'sum'))
)
dup_counts['collapse_rate'] = dup_counts['dup_collapsed'] / dup_counts['raw']
dup_counts = dup_counts.sort_values('dup_collapsed', ascending=False)
dup_counts.head(30).to_csv('exploration/tables/T16/dedup_impact_top30.csv', index=False)
print(f"  top dedup victims:\n{dup_counts.head(10).to_string(index=False)}")

# Use dedup panel from here forward
df = df_dedup

# -------------------- Step 2: overlap panel --------------------
print("[3/9] Building overlap panel (>=3 SWE both periods)...")
# "Both periods" means arshkon 2024 AND scraped 2026 (arshkon is the 2024 baseline
# for native-entry analyses). Also separately compute pooled-2024 panel.

arshkon_counts = (
    df[df['source'] == 'kaggle_arshkon']
    .groupby('company').size().rename('n_arshkon')
)
scraped_counts = (
    df[df['source'] == 'scraped']
    .groupby('company').size().rename('n_scraped')
)
asan_counts = (
    df[df['source'] == 'kaggle_asaniczka']
    .groupby('company').size().rename('n_asaniczka')
)

panel = pd.concat([arshkon_counts, scraped_counts, asan_counts], axis=1).fillna(0).astype(int)
panel['n_2024_pooled'] = panel['n_arshkon'] + panel['n_asaniczka']
panel['n_2026'] = panel['n_scraped']

overlap_arsh = panel[(panel['n_arshkon'] >= 3) & (panel['n_scraped'] >= 3)].copy()
overlap_pool = panel[(panel['n_2024_pooled'] >= 3) & (panel['n_2026'] >= 3)].copy()
print(f"  arshkon∩scraped overlap: {len(overlap_arsh)} companies")
print(f"  pooled2024∩scraped overlap: {len(overlap_pool)} companies")

# -------------------- Step 3: entry-poster concentration --------------------
print("[4/9] Entry-poster concentration (scraped, >=5 SWE dedup)...")

scraped = df[df['source'] == 'scraped'].copy()
scraped['yoe_le2'] = (scraped['yoe_extracted'] <= 2) & scraped['yoe_extracted'].notna()
scraped['is_entry_best'] = scraped['seniority_best'] == 'entry'

co_totals = scraped.groupby('company').agg(
    n_swe=('uid', 'count'),
    n_entry_best=('is_entry_best', 'sum'),
    n_entry_yoe=('yoe_le2', 'sum'),
    n_yoe_known=('yoe_extracted', lambda s: s.notna().sum()),
    n_best_known=('seniority_best', lambda s: s.notna().sum()),
    industry=('company_industry', lambda s: s.mode().iloc[0] if len(s.mode()) else None),
    mean_yoe=('yoe_extracted', 'mean'),
)
substantial = co_totals[co_totals['n_swe'] >= 5].copy()
print(f"  scraped companies with >=5 SWE (dedup): {len(substantial)}")

sub_best = substantial[substantial['n_best_known'] > 0]
zero_entry_best = (substantial['n_entry_best'] == 0).sum()
zero_entry_yoe = (substantial[substantial['n_yoe_known'] > 0]['n_entry_yoe'] == 0).sum()
print(f"  # with any entry rows under combined column: {len(substantial) - zero_entry_best}")
print(f"  # with ZERO entry rows (combined col): {zero_entry_best}  ({zero_entry_best/len(substantial):.1%})")
print(f"  # with ZERO entry rows (YOE<=2): {zero_entry_yoe}  ({zero_entry_yoe/len(substantial):.1%})")

# Top 20 entry posters under YOE proxy (cleaner)
top_entry = substantial.sort_values('n_entry_yoe', ascending=False).head(20).copy()
top_entry['entry_share_yoe'] = top_entry['n_entry_yoe'] / top_entry['n_yoe_known']
top_entry['entry_share_best'] = top_entry['n_entry_best'] / top_entry['n_best_known'].replace(0, np.nan)
top_entry.to_csv('exploration/tables/T16/top20_entry_posters_scraped.csv')
print("  top 20 entry posters (YOE proxy):")
print(top_entry[['n_swe', 'n_entry_yoe', 'entry_share_yoe', 'entry_share_best', 'mean_yoe', 'industry']].head(20).to_string())

# Compare entry-poster vs non-entry-poster
substantial['is_entry_poster'] = substantial['n_entry_yoe'] >= 1
poster_profile = substantial.groupby('is_entry_poster').agg(
    n_companies=('n_swe', 'count'),
    mean_postings=('n_swe', 'mean'),
    mean_yoe=('mean_yoe', 'mean'),
)
print(f"  profile:\n{poster_profile}")

# Industry breakdown of entry-posters
entry_poster_ind = substantial[substantial['is_entry_poster']].groupby('industry').size().sort_values(ascending=False).head(15)
entry_poster_ind.to_csv('exploration/tables/T16/entry_poster_industries.csv')

# -------------------- Step 4: per-company change metrics --------------------
print("[5/9] Per-company change metrics (overlap = arshkon∩scraped)...")

# Pre-compute binaries on full df
df['has_ai'] = df['text'].str.contains(AI_PAT, na=False)
df['has_scope'] = df['text'].str.contains(SCOPE_PAT, na=False)
df['has_mentor_strict'] = df['text'].str.contains(MENTOR_PAT, na=False)
df['desc_len'] = df['description'].fillna('').str.len()
df['desc_len_llm'] = df['description_core_llm'].fillna('').str.len()
df['has_llm_text'] = df['description_core_llm'].notna()
df['yoe_le2'] = (df['yoe_extracted'] <= 2) & df['yoe_extracted'].notna()
df['is_entry_best'] = df['seniority_best'] == 'entry'
df['is_entry_native'] = df['seniority_native'] == 'entry'
df['is_entry_final'] = df['seniority_final'] == 'entry'

# Load tech matrix for tech counts (skip c_cpp/csharp per bug)
tech = con.execute(
    "SELECT * FROM 'exploration/artifacts/shared/swe_tech_matrix.parquet'"
).fetchdf()
tech_cols = [c for c in tech.columns if c not in ('uid', 'c_cpp', 'csharp')]
tech['tech_count'] = tech[tech_cols[1:]].sum(axis=1) if len(tech_cols) > 1 else 0
tech_small = tech[['uid', 'tech_count']]
df = df.merge(tech_small, on='uid', how='left')
df['tech_count'] = df['tech_count'].fillna(0)

# Build per-company, per-period aggregates (overlap companies only)
overlap_companies = set(overlap_arsh.index)

def agg_period(sub):
    n = len(sub)
    n_best = sub['seniority_best'].notna().sum()
    n_yoe = sub['yoe_extracted'].notna().sum()
    n_llm = sub['has_llm_text'].sum()
    return pd.Series({
        'n': n,
        'entry_share_best': sub['is_entry_best'].sum() / n_best if n_best else np.nan,
        'entry_share_yoe': sub['yoe_le2'].sum() / n_yoe if n_yoe else np.nan,
        'ai_rate': sub['has_ai'].mean(),
        'mean_desc_len': sub['desc_len'].mean(),
        'mean_desc_len_llm_only': sub.loc[sub['has_llm_text'], 'desc_len_llm'].mean() if n_llm else np.nan,
        'mean_tech_count': sub['tech_count'].mean(),
        'mean_scope': sub['has_scope'].mean(),
        'mean_mentor': sub['has_mentor_strict'].mean(),
    })

overlap_df = df[df['company'].isin(overlap_companies)].copy()
arsh_agg = (
    overlap_df[overlap_df['source'] == 'kaggle_arshkon']
    .groupby('company').apply(agg_period).add_suffix('_2024')
)
scr_agg = (
    overlap_df[overlap_df['source'] == 'scraped']
    .groupby('company').apply(agg_period).add_suffix('_2026')
)
co_change = arsh_agg.join(scr_agg, how='inner')

for m in ['entry_share_best', 'entry_share_yoe', 'ai_rate', 'mean_desc_len', 'mean_desc_len_llm_only', 'mean_tech_count', 'mean_scope', 'mean_mentor']:
    co_change[f'delta_{m}'] = co_change[f'{m}_2026'] - co_change[f'{m}_2024']

co_change.to_csv('exploration/tables/T16/overlap_per_company_changes.csv')
print(f"  overlap panel (arsh∩scr, >=3 both): {len(co_change)} companies")
print(f"  mean Δ ai_rate: {co_change['delta_ai_rate'].mean():.4f}")
print(f"  mean Δ mean_desc_len: {co_change['delta_mean_desc_len'].mean():.1f}")
print(f"  mean Δ mean_desc_len_llm_only: {co_change['delta_mean_desc_len_llm_only'].mean():.1f}")
print(f"  mean Δ mean_tech_count: {co_change['delta_mean_tech_count'].mean():.2f}")
print(f"  mean Δ entry_share_yoe: {co_change['delta_entry_share_yoe'].mean():.4f}")
print(f"  mean Δ entry_share_best: {co_change['delta_entry_share_best'].mean():.4f}")

# Disagreement between combined column and YOE at company level
disagree = co_change.dropna(subset=['delta_entry_share_best', 'delta_entry_share_yoe']).copy()
if len(disagree) > 0:
    disagree['sign_diff'] = np.sign(disagree['delta_entry_share_best']) != np.sign(disagree['delta_entry_share_yoe'])
    n_disagree = disagree['sign_diff'].sum()
    print(f"  companies with direction disagreement best vs YOE: {n_disagree}/{len(disagree)}")

# -------------------- Step 5: k-means clustering --------------------
print("[6/9] k-means on change vectors...")

feat_cols = ['delta_entry_share_yoe', 'delta_ai_rate', 'delta_mean_desc_len_llm_only', 'delta_mean_tech_count', 'delta_mean_scope']
cluster_df = co_change[feat_cols].copy()
# Fill NaN with column median (for LLM-only length which may be missing)
cluster_df = cluster_df.fillna(cluster_df.median())

scaler = StandardScaler()
X = scaler.fit_transform(cluster_df.values)

# Elbow via inertia
inertias = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertias.append(km.inertia_)
# pick k=5 (pragmatic; also check elbow)
K = 5
km = KMeans(n_clusters=K, n_init=20, random_state=42).fit(X)
cluster_df['cluster'] = km.labels_
co_change['cluster'] = km.labels_

# Cluster profile
clust_summary = cluster_df.groupby('cluster').agg(['mean', 'count'])
print("  cluster centers (original units):")
center_orig = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=feat_cols)
center_orig['n_companies'] = [(km.labels_ == c).sum() for c in range(K)]
print(center_orig.to_string())
center_orig.to_csv('exploration/tables/T16/cluster_centers.csv')

# Name clusters by dominant feature
def name_cluster(row):
    # z-scored
    z = {c: (row[c] - center_orig[c].mean()) / center_orig[c].std() for c in feat_cols}
    top = max(z.items(), key=lambda kv: abs(kv[1]))
    direction = 'high' if top[1] > 0 else 'low'
    labels = {
        'delta_entry_share_yoe': 'entry',
        'delta_ai_rate': 'AI',
        'delta_mean_desc_len_llm_only': 'length',
        'delta_mean_tech_count': 'tech',
        'delta_mean_scope': 'scope',
    }
    return f"{direction}-{labels[top[0]]}"

center_orig['name'] = center_orig.apply(name_cluster, axis=1)
print(center_orig[['name', 'n_companies']])

# Exemplars: nearest to center
from sklearn.metrics.pairwise import euclidean_distances
exemplars = {}
for c in range(K):
    mask = km.labels_ == c
    idx = np.where(mask)[0]
    dists = euclidean_distances(X[idx], km.cluster_centers_[c:c+1]).flatten()
    top5 = idx[np.argsort(dists)[:5]]
    exemplars[c] = co_change.index[top5].tolist()
    print(f"  cluster {c} ({center_orig.loc[c, 'name']}): {exemplars[c]}")

# Save with cluster labels
co_change.to_csv('exploration/tables/T16/overlap_per_company_changes_with_clusters.csv')

# Plot cluster deltas
fig, axes = plt.subplots(1, len(feat_cols), figsize=(18, 4))
for i, col in enumerate(feat_cols):
    data = [co_change.loc[co_change['cluster'] == c, col].dropna().values for c in range(K)]
    axes[i].boxplot(data, labels=[center_orig.loc[c, 'name'] for c in range(K)])
    axes[i].set_title(col, fontsize=9)
    axes[i].axhline(0, color='red', lw=0.5, ls='--')
    axes[i].tick_params(axis='x', rotation=30, labelsize=7)
plt.tight_layout()
plt.savefig('exploration/figures/T16/cluster_boxplots.png', dpi=120)
plt.close()

# -------------------- Step 6: within/between decomposition --------------------
print("[7/9] Within/between company decomposition...")

def decompose(df_period, period_label, value_col, weight_col='n'):
    # Ignore NaN
    d = df_period.dropna(subset=[value_col])
    if len(d) == 0:
        return np.nan
    w = d[weight_col] if weight_col in d.columns else np.ones(len(d))
    return (d[value_col] * w).sum() / w.sum()

# For overlap companies only (fixed panel), decompose period→period change in
#   (1) entry_share under multiple operationalizations
#   (2) AI rate
#   (3) description length (raw & LLM-only)

def decomp_on_metric(co_change, metric_prefix, label):
    """Shift-share: aggregate change = within (same weights, Δmetric) + between (same metric, Δweights).
    Use 2024 row weight = n_2024, 2026 row weight = n_2026.
    """
    d = co_change.dropna(subset=[f'{metric_prefix}_2024', f'{metric_prefix}_2026']).copy()
    if len(d) == 0:
        return None
    w24 = d['n_2024']
    w26 = d['n_2026']
    m24 = d[f'{metric_prefix}_2024']
    m26 = d[f'{metric_prefix}_2026']
    s24 = w24 / w24.sum()
    s26 = w26 / w26.sum()
    agg_24 = (s24 * m24).sum()
    agg_26 = (s26 * m26).sum()
    # within (using average share)
    s_avg = (s24 + s26) / 2
    m_avg = (m24 + m26) / 2
    within = (s_avg * (m26 - m24)).sum()
    between = ((s26 - s24) * m_avg).sum()
    total = agg_26 - agg_24
    return {
        'metric': label,
        'agg_2024': agg_24,
        'agg_2026': agg_26,
        'total_change': total,
        'within_company': within,
        'between_company': between,
        'within_pct': within / total * 100 if total != 0 else np.nan,
        'between_pct': between / total * 100 if total != 0 else np.nan,
    }

decomp_rows = []
# n totals per company per period (adding to co_change)
co_change['n_2024'] = co_change['n_2024']  # already exists from agg_period
co_change['n_2026'] = co_change['n_2026']

for prefix, label in [
    ('entry_share_best', 'entry_share (combined col)'),
    ('entry_share_yoe', 'entry_share (YOE<=2)'),
    ('ai_rate', 'AI rate'),
    ('mean_desc_len', 'desc length (raw)'),
    ('mean_desc_len_llm_only', 'desc length (LLM-only)'),
    ('mean_tech_count', 'tech count'),
    ('mean_scope', 'scope mentions'),
]:
    r = decomp_on_metric(co_change, prefix, label)
    if r:
        decomp_rows.append(r)

# Also add native/final entry decompositions for ablation
for prefix_col, out_prefix, label in [
    ('is_entry_native', 'ens_native', 'entry_share (native, arshkon-only)'),
    ('is_entry_final', 'ens_final', 'entry_share (seniority_final)'),
]:
    # Recompute per-company entry shares using native/final
    tmp24 = overlap_df[overlap_df['source'] == 'kaggle_arshkon'].groupby('company').agg(
        n=('uid', 'count'),
        entry=(prefix_col, 'sum')
    )
    tmp24['share'] = tmp24['entry'] / tmp24['n']
    tmp26 = overlap_df[overlap_df['source'] == 'scraped'].groupby('company').agg(
        n=('uid', 'count'),
        entry=(prefix_col, 'sum')
    )
    tmp26['share'] = tmp26['entry'] / tmp26['n']
    joined = tmp24[['n', 'share']].add_suffix('_24').join(tmp26[['n', 'share']].add_suffix('_26'), how='inner')
    if len(joined) == 0:
        continue
    s24 = joined['n_24'] / joined['n_24'].sum()
    s26 = joined['n_26'] / joined['n_26'].sum()
    m24 = joined['share_24']
    m26 = joined['share_26']
    s_avg = (s24 + s26) / 2
    m_avg = (m24 + m26) / 2
    within = (s_avg * (m26 - m24)).sum()
    between = ((s26 - s24) * m_avg).sum()
    total = (s26 * m26).sum() - (s24 * m24).sum()
    decomp_rows.append({
        'metric': label,
        'agg_2024': (s24 * m24).sum(),
        'agg_2026': (s26 * m26).sum(),
        'total_change': total,
        'within_company': within,
        'between_company': between,
        'within_pct': within / total * 100 if total != 0 else np.nan,
        'between_pct': between / total * 100 if total != 0 else np.nan,
    })

decomp_df = pd.DataFrame(decomp_rows)
decomp_df.to_csv('exploration/tables/T16/decomposition.csv', index=False)
print(decomp_df.to_string(index=False))

# -------------------- Step 6b: archetype-stratified decomposition --------------------
print("  archetype-stratified decomposition...")
arch = con.execute("SELECT uid, archetype_name FROM 'exploration/artifacts/shared/swe_archetype_labels.parquet'").fetchdf()
df_arch = df.merge(arch, on='uid', how='inner')
overlap_arch = df_arch[df_arch['company'].isin(overlap_companies)].copy()

# Compute within-domain, between-domain, between-company decomposition for YOE entry share
# Create company×domain cells
def cell_agg(sub, col):
    return pd.Series({
        'n': len(sub),
        'val': sub[col].sum() / sub[col].notna().sum() if sub[col].notna().sum() else np.nan,
    })

arch_rows = []
for col, label in [('yoe_le2', 'entry (YOE<=2)'), ('has_ai', 'AI rate')]:
    c24 = overlap_arch[overlap_arch['source'] == 'kaggle_arshkon'].groupby(['company', 'archetype_name']).apply(lambda s: cell_agg(s, col))
    c26 = overlap_arch[overlap_arch['source'] == 'scraped'].groupby(['company', 'archetype_name']).apply(lambda s: cell_agg(s, col))
    if c24.empty or c26.empty:
        continue
    joined = c24.add_suffix('_24').join(c26.add_suffix('_26'), how='inner').dropna()
    if len(joined) == 0:
        continue
    joined['n_24'] = joined['n_24'].astype(float)
    joined['n_26'] = joined['n_26'].astype(float)
    # Overall decomposition into within (same cell), between-archetype (different archetype share), between-company within archetype
    # Simplified: within-cell Δ weighted by avg share
    s24 = joined['n_24'] / joined['n_24'].sum()
    s26 = joined['n_26'] / joined['n_26'].sum()
    s_avg = (s24 + s26) / 2
    within = (s_avg * (joined['val_26'] - joined['val_24'])).sum()
    between = ((s26 - s24) * (joined['val_24'] + joined['val_26']) / 2).sum()
    total = within + between
    arch_rows.append({'metric': label, 'within_cell': within, 'between_cell': between, 'total': total})

if arch_rows:
    pd.DataFrame(arch_rows).to_csv('exploration/tables/T16/decomposition_archetype.csv', index=False)
    print(pd.DataFrame(arch_rows).to_string(index=False))

# -------------------- Step 7: scope inflation within company --------------------
print("[8/9] Within-company entry scope inflation test...")
# Compare entry-level scope mentions 2024 vs 2026 within same companies
entry_by_period = (
    overlap_df[overlap_df['yoe_le2']]
    .groupby(['company', 'source'])
    .agg(n=('uid', 'count'), scope=('has_scope', 'mean'), mentor=('has_mentor_strict', 'mean'), ai=('has_ai', 'mean'))
)
entry_pivot = entry_by_period.unstack('source')
# Need both periods
if ('n', 'kaggle_arshkon') in entry_pivot.columns and ('n', 'scraped') in entry_pivot.columns:
    mask = entry_pivot[('n', 'kaggle_arshkon')].fillna(0) >= 2
    mask &= entry_pivot[('n', 'scraped')].fillna(0) >= 2
    sub = entry_pivot[mask]
    scope_24 = sub[('scope', 'kaggle_arshkon')]
    scope_26 = sub[('scope', 'scraped')]
    mentor_24 = sub[('mentor', 'kaggle_arshkon')]
    mentor_26 = sub[('mentor', 'scraped')]
    print(f"  companies with entry rows in both: {len(sub)}")
    if len(sub):
        print(f"  within-company Δ scope (entry-only): {(scope_26 - scope_24).mean():.4f}")
        print(f"  within-company Δ strict mentor (entry-only): {(mentor_26 - mentor_24).mean():.4f}")
        sub.to_csv('exploration/tables/T16/within_company_entry_scope.csv')

# -------------------- Step 8: new entrants --------------------
print("[9/9] New market entrants (2026 only)...")
arsh_set = set(arshkon_counts.index)
pooled_2024_set = set(arshkon_counts.index) | set(asan_counts.index)

scraped_co_totals = (
    df[df['source'] == 'scraped']
    .groupby('company').agg(
        n=('uid', 'count'),
        ai=('has_ai', 'mean'),
        desc_len=('desc_len', 'mean'),
        yoe_entry_rate=('yoe_le2', 'mean'),
        scope=('has_scope', 'mean'),
        industry=('company_industry', lambda s: s.mode().iloc[0] if len(s.mode()) else None),
    )
)
# Companies in scraped but not in 2024 (either source)
new_entrants = scraped_co_totals[~scraped_co_totals.index.isin(pooled_2024_set)]
new_entrants_sub = new_entrants[new_entrants['n'] >= 5]
returners = scraped_co_totals[scraped_co_totals.index.isin(pooled_2024_set)]
returners_sub = returners[returners['n'] >= 5]

print(f"  new entrants (scraped only, >=5): {len(new_entrants_sub)}")
print(f"  returners (in 2024 too, >=5): {len(returners_sub)}")
print("  comparison (means):")
comparison = pd.DataFrame({
    'new_entrants': new_entrants_sub[['ai', 'desc_len', 'yoe_entry_rate', 'scope']].mean(),
    'returners': returners_sub[['ai', 'desc_len', 'yoe_entry_rate', 'scope']].mean(),
})
comparison['diff'] = comparison['new_entrants'] - comparison['returners']
print(comparison)
comparison.to_csv('exploration/tables/T16/new_entrants_vs_returners.csv')

# Top new entrant industries
new_ent_ind = new_entrants_sub.groupby('industry').size().sort_values(ascending=False).head(15)
new_ent_ind.to_csv('exploration/tables/T16/new_entrants_top_industries.csv')
print(f"  top new entrant industries:\n{new_ent_ind}")

# -------------------- Aggregator vs direct --------------------
print("  aggregator vs direct comparison...")
agg_change = df.groupby(['is_aggregator', 'source']).agg(
    n=('uid', 'count'),
    ai=('has_ai', 'mean'),
    yoe_entry=('yoe_le2', 'mean'),
    scope=('has_scope', 'mean'),
)
print(agg_change)
agg_change.to_csv('exploration/tables/T16/aggregator_vs_direct.csv')

# -------------------- Sensitivity: aggregator exclusion --------------------
print("  sensitivity: drop aggregators, re-decompose YOE entry share...")
df_noagg = df[df['is_aggregator'] != True]
overlap_noagg = df_noagg[df_noagg['company'].isin(overlap_companies)].copy()
c24 = overlap_noagg[overlap_noagg['source'] == 'kaggle_arshkon'].groupby('company').agg(n=('uid', 'count'), yoe_entry=('yoe_le2', 'mean'))
c26 = overlap_noagg[overlap_noagg['source'] == 'scraped'].groupby('company').agg(n=('uid', 'count'), yoe_entry=('yoe_le2', 'mean'))
jj = c24.add_suffix('_24').join(c26.add_suffix('_26'), how='inner').dropna()
if len(jj) > 0:
    s24 = jj['n_24'] / jj['n_24'].sum()
    s26 = jj['n_26'] / jj['n_26'].sum()
    s_avg = (s24 + s26) / 2
    w = (s_avg * (jj['yoe_entry_26'] - jj['yoe_entry_24'])).sum()
    b = ((s26 - s24) * (jj['yoe_entry_24'] + jj['yoe_entry_26']) / 2).sum()
    print(f"    no-aggregator overlap panel ({len(jj)} cos): within={w:.4f} between={b:.4f}")

print("\nT16 done.")
