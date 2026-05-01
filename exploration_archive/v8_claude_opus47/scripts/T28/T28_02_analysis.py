"""T28 steps 2-7: domain-stratified scope changes by archetype.

Joins:
- projected_archetypes.parquet (T28_01)
- swe_cleaned_text.parquet (source, period, seniority, text_source, company, uid)
- T11/T11_posting_features.parquet (requirement_breadth, tech_count, org_scope_count, etc.)
- swe_tech_matrix.parquet (not needed right now; T11 features have tech_count already)

Produces:
- T28_archetype_period_share.csv  archetype x period row counts & shares
- T28_entry_decomposition.csv  within vs between-archetype entry-share decomposition (J2, J3)
- T28_ai_mention_by_archetype.csv  AI-mention strict share 2024 -> 2026 per archetype
- T28_scope_by_archetype.csv  breadth, tech_count, org_scope, AI, credential stack changes per archetype
- T28_junior_senior_by_archetype.csv  junior vs senior content per archetype
- T28_mentor_by_archetype.csv  V1-refined mentor pattern delta per archetype
- T28_ai_ml_deep_dive.csv  employer / entry mix / tech stack profile within ai_ml_engineering
"""
from __future__ import annotations
import re
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import duckdb
from pathlib import Path

SHARED = Path('/home/jihgaboot/gabor/job-research/exploration/artifacts/shared')
T11 = Path('/home/jihgaboot/gabor/job-research/exploration/artifacts/T11')
T28 = Path('/home/jihgaboot/gabor/job-research/exploration/artifacts/T28')
OUT_TAB = Path('/home/jihgaboot/gabor/job-research/exploration/tables/T28')
OUT_TAB.mkdir(parents=True, exist_ok=True)

# ---------- Load ----------
print('[load] projected archetypes')
proj = pq.read_table(T28 / 'projected_archetypes.parquet').to_pandas()
# projected_archetype only populated for the 34k LLM rows

print('[load] cleaned text metadata (all 63,701; we will filter)')
cleaned = pq.read_table(
    SHARED / 'swe_cleaned_text.parquet',
    columns=['uid','source','period','seniority_final','seniority_3level','is_aggregator','company_name_canonical','yoe_extracted','text_source','description_cleaned'],
).to_pandas()
print(f'  cleaned rows {len(cleaned)}')
# LLM-labeled only for archetype-joined analyses
cleaned_llm = cleaned[cleaned['text_source'] == 'llm'].copy()
print(f'  llm rows {len(cleaned_llm)}')

print('[load] T11 posting features')
t11 = pq.read_table(T11 / 'T11_posting_features.parquet').to_pandas()
print(f'  t11 rows {len(t11)}')

# Merge meta + features + archetype (LLM rows only)
df = cleaned_llm.merge(proj[['uid','projected_archetype']], on='uid', how='left')
df = df.merge(
    t11[['uid','tech_count','ai_count_tech','desc_len_chars','soft_skill_count','org_scope_count','management_STRICT_count','management_STRICT_binary','management_BROAD_count','management_BROAD_binary','credential_stack_depth','requirement_breadth','ai_count','yoe_numeric','is_specialist_company']],
    on='uid', how='left'
)
print(f'  merged rows {len(df)} (expected 34,102 LLM rows)')

# period_bucket
df['period_bucket'] = df['period'].str.startswith('2024').map({True:'2024', False:'2026'})

# ---- V1-refined patterns ----
AI_STRICT_RX = re.compile(r'\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|huggingface|hugging face)\b', re.I)
AI_BROAD_EXTRA = re.compile(r'\b(ai|artificial intelligence|ml|machine learning|llm|large language model|generative ai|genai|anthropic)\b', re.I)
# V1-refined mentor pattern (dropped manage + team_building which had <50% precision)
MENTOR_RX = re.compile(r'\b(mentor|mentoring|mentored|mentorship|coach|coaching|coached|hiring|headcount|performance review|performance reviews)\b', re.I)

def ai_strict(x):
    return 1 if isinstance(x, str) and AI_STRICT_RX.search(x) else 0
def ai_broad(x):
    if not isinstance(x, str):
        return 0
    return 1 if (AI_STRICT_RX.search(x) or AI_BROAD_EXTRA.search(x)) else 0
def mentor_match(x):
    return 1 if isinstance(x, str) and MENTOR_RX.search(x) else 0

print('[compute] AI binaries + mentor on cleaned text (LLM rows)')
df['ai_strict_bin'] = df['description_cleaned'].apply(ai_strict).astype(np.int8)
df['ai_broad_bin'] = df['description_cleaned'].apply(ai_broad).astype(np.int8)
df['mentor_bin'] = df['description_cleaned'].apply(mentor_match).astype(np.int8)

# ---- Length-residualized requirement_breadth ----
# Per V1 / T11 approach: regress breadth on log(desc_len). Residuals are the
# length-free breadth. We fit on 2024 baseline so residual captures content.
import statsmodels.api as sm
mask24 = df['period_bucket'] == '2024'
X = np.log(np.clip(df.loc[mask24, 'desc_len_chars'].values.astype(float), 1, None))
y = df.loc[mask24, 'requirement_breadth'].values.astype(float)
X1 = sm.add_constant(X)
model = sm.OLS(y, X1).fit()
beta0, beta1 = model.params
print(f'  breadth ~ a + b*log(len): beta0={beta0:.3f} beta1={beta1:.3f} R2={model.rsquared:.3f}')
logL = np.log(np.clip(df['desc_len_chars'].values.astype(float), 1, None))
df['breadth_resid'] = df['requirement_breadth'].astype(float) - (beta0 + beta1 * logL)

# top 10 archetypes (by total volume in LLM corpus after filtering specialists for robustness later)
TOP10 = [
    'generic_software_engineer','cloud_devops','data_engineering','systems_engineering',
    'ai_ml_engineering','frontend_react','java_spring_backend','customer_project_engineer',
    'backend_platform','network_security_linux',
]

# ---------- Step 1: archetype x period distribution ----------
print('[step 1] archetype x period')
tbl = df.groupby(['projected_archetype','period_bucket']).size().unstack(fill_value=0)
tbl['total'] = tbl.sum(axis=1)
tbl['share_2024'] = tbl['2024'] / tbl['2024'].sum()
tbl['share_2026'] = tbl['2026'] / tbl['2026'].sum()
tbl['delta_share_pp'] = (tbl['share_2026'] - tbl['share_2024']) * 100
tbl = tbl.sort_values('total', ascending=False)
tbl.to_csv(OUT_TAB / 'T28_archetype_period_share.csv')
print(tbl.head(22))

# ---------- Step 2: entry decomposition ----------
# Define entry under 2 operationalizations
# J2: seniority_final in {entry, associate}
# J3: yoe_numeric <= 2 (T11 imputes; but we only use rows where yoe_numeric > 0 flag is True)
df['is_entry_J2'] = df['seniority_final'].isin(['entry','associate']).astype(int)
# For J3: use yoe_numeric if not imputed - T11 gives yoe_numeric always populated; condition on yoe_extracted presence
has_yoe = df['yoe_extracted'].notna()
df['is_entry_J3'] = np.where(has_yoe, (df['yoe_numeric'] <= 2).astype(int), np.nan)

def decomp(df_in, entry_col, weight_col='w'):
    """Shift-share decomposition: delta in aggregate entry share from 2024 to 2026.

    delta = sum_k [pi_k(2026) * p_k(2026) - pi_k(2024) * p_k(2024)]
          = within  + between + interaction
    using
      within_k = pi_k(2024) * (p_k(2026) - p_k(2024))
      between_k = p_k(2024) * (pi_k(2026) - pi_k(2024))
      interaction_k = (pi_k(2026) - pi_k(2024)) * (p_k(2026) - p_k(2024))
    where p_k = share of archetype k, pi_k = entry share within archetype k.
    """
    d = df_in[df_in[entry_col].notna()].copy()
    agg24 = d.loc[d['period_bucket']=='2024', entry_col].mean() if (d['period_bucket']=='2024').any() else np.nan
    agg26 = d.loc[d['period_bucket']=='2026', entry_col].mean() if (d['period_bucket']=='2026').any() else np.nan
    delta = agg26 - agg24
    rows = []
    # counts per archetype/period, and entry shares
    grp = d.groupby(['projected_archetype','period_bucket']).agg(n=(entry_col,'size'), pi=(entry_col,'mean')).unstack('period_bucket')
    n24 = grp[('n','2024')].fillna(0); n26 = grp[('n','2026')].fillna(0)
    pi24 = grp[('pi','2024')]; pi26 = grp[('pi','2026')]
    tot24 = n24.sum(); tot26 = n26.sum()
    p24 = n24 / tot24; p26 = n26 / tot26
    within = (p24 * (pi26.fillna(0) - pi24.fillna(0))).fillna(0)
    between = (pi24.fillna(0) * (p26 - p24)).fillna(0)
    interaction = ((pi26.fillna(0) - pi24.fillna(0)) * (p26 - p24)).fillna(0)
    out = pd.DataFrame({
        'n_2024':n24,'n_2026':n26,'p_2024':p24,'p_2026':p26,
        'pi_2024':pi24,'pi_2026':pi26,'within':within,'between':between,'interaction':interaction,
    })
    out.loc['_TOTAL'] = {
        'n_2024': n24.sum(), 'n_2026': n26.sum(),
        'p_2024': p24.sum(), 'p_2026': p26.sum(),
        'pi_2024': agg24, 'pi_2026': agg26,
        'within': within.sum(), 'between': between.sum(), 'interaction': interaction.sum(),
    }
    out['delta_pi'] = out['pi_2026'] - out['pi_2024']
    return out, delta

print('[step 2] entry decomposition J2')
j2_out, j2_delta = decomp(df, 'is_entry_J2')
print(f'  aggregate J2 delta: {j2_delta*100:.2f}pp')
j2_out.to_csv(OUT_TAB / 'T28_entry_decomposition_J2.csv')

print('[step 2] entry decomposition J3 (yoe known only)')
df_j3 = df[df['is_entry_J3'].notna()].copy()
j3_out, j3_delta = decomp(df_j3, 'is_entry_J3')
print(f'  aggregate J3 delta (yoe-known subset): {j3_delta*100:.2f}pp')
j3_out.to_csv(OUT_TAB / 'T28_entry_decomposition_J3.csv')

# Within vs between totals
tot_within_j2 = j2_out.loc['_TOTAL','within']
tot_between_j2 = j2_out.loc['_TOTAL','between']
tot_interaction_j2 = j2_out.loc['_TOTAL','interaction']
print(f'[J2 decomp] within {tot_within_j2*100:.2f}pp between {tot_between_j2*100:.2f}pp interaction {tot_interaction_j2*100:.3f}pp sum {(tot_within_j2+tot_between_j2+tot_interaction_j2)*100:.2f}pp')

# ---------- Step 3: AI-mention by archetype (THE critical test) ----------
print('[step 3] AI mention by archetype')
def per_group_rates(d, flag_col):
    g = d.groupby(['projected_archetype','period_bucket'])[flag_col].agg(['mean','size']).unstack('period_bucket')
    out = pd.DataFrame({
        'n_2024': g[('size','2024')].fillna(0).astype(int),
        'n_2026': g[('size','2026')].fillna(0).astype(int),
        'share_2024': g[('mean','2024')],
        'share_2026': g[('mean','2026')],
    })
    out['delta_pp'] = (out['share_2026'] - out['share_2024']) * 100
    # SE under indep Bernoulli
    var24 = (out['share_2024'] * (1-out['share_2024'])) / np.clip(out['n_2024'],1,None)
    var26 = (out['share_2026'] * (1-out['share_2026'])) / np.clip(out['n_2026'],1,None)
    out['se_pp'] = np.sqrt(var24 + var26) * 100
    out['snr'] = np.where(out['se_pp']>0, np.abs(out['delta_pp'])/out['se_pp'], np.nan)
    return out.sort_values('n_2024', ascending=False)

ai_strict_by = per_group_rates(df, 'ai_strict_bin')
ai_broad_by = per_group_rates(df, 'ai_broad_bin')
ai_strict_by.to_csv(OUT_TAB / 'T28_ai_strict_by_archetype.csv')
ai_broad_by.to_csv(OUT_TAB / 'T28_ai_broad_by_archetype.csv')
print('AI strict by archetype:')
print(ai_strict_by)

# ---------- Step 4: scope inflation per archetype ----------
print('[step 4] scope metrics by archetype')
def per_group_means(d, col):
    g = d.groupby(['projected_archetype','period_bucket'])[col].agg(['mean','std','size']).unstack('period_bucket')
    out = pd.DataFrame({
        'mean_2024': g[('mean','2024')],
        'mean_2026': g[('mean','2026')],
        'sd_2024': g[('std','2024')],
        'sd_2026': g[('std','2026')],
        'n_2024': g[('size','2024')].fillna(0).astype(int),
        'n_2026': g[('size','2026')].fillna(0).astype(int),
    })
    out['delta'] = out['mean_2026'] - out['mean_2024']
    # pooled SE assuming indep
    var24 = out['sd_2024']**2 / np.clip(out['n_2024'],1,None)
    var26 = out['sd_2026']**2 / np.clip(out['n_2026'],1,None)
    out['se'] = np.sqrt(var24 + var26)
    out['snr'] = np.where(out['se']>0, np.abs(out['delta'])/out['se'], np.nan)
    return out.sort_values('n_2024', ascending=False)

scope_cols = ['requirement_breadth','breadth_resid','tech_count','org_scope_count','credential_stack_depth','desc_len_chars']
scope_rows = []
for c in scope_cols:
    t = per_group_means(df, c)
    t['metric'] = c
    scope_rows.append(t.reset_index())
scope = pd.concat(scope_rows, axis=0, ignore_index=True)
scope.to_csv(OUT_TAB / 'T28_scope_by_archetype.csv', index=False)

# AI-mention share rows (for same table)
ai_rows = []
for label, series in [('ai_strict_share', ai_strict_by), ('ai_broad_share', ai_broad_by)]:
    d = series.reset_index()
    d['metric'] = label
    d = d.rename(columns={'share_2024':'mean_2024','share_2026':'mean_2026','se_pp':'se','delta_pp':'delta'})
    d['sd_2024']=np.nan; d['sd_2026']=np.nan
    ai_rows.append(d[['projected_archetype','mean_2024','mean_2026','sd_2024','sd_2026','n_2024','n_2026','delta','se','snr','metric']])
scope2 = pd.concat(scope_rows + ai_rows, axis=0, ignore_index=True)
scope2.to_csv(OUT_TAB / 'T28_scope_plus_ai_by_archetype.csv', index=False)

# ---------- Step 5: junior vs senior content within each archetype ----------
print('[step 5] junior vs senior content by archetype')
def j_vs_s(d, metric_col, is_bin=False):
    """For each archetype, compute (J2 2024, J2 2026, S1 2024, S1 2026, gap2024, gap2026, gap_change)."""
    d = d.copy()
    d['seniority_s1'] = d['seniority_final'].isin(['mid-senior','director']).astype(int)
    d['seniority_j2'] = d['seniority_final'].isin(['entry','associate']).astype(int)
    rows = []
    for arch, sub in d.groupby('projected_archetype'):
        def mean_of(mask):
            s = sub.loc[mask, metric_col]
            return s.mean() if len(s) else np.nan, int(len(s))
        j24, n_j24 = mean_of((sub['seniority_j2']==1) & (sub['period_bucket']=='2024'))
        j26, n_j26 = mean_of((sub['seniority_j2']==1) & (sub['period_bucket']=='2026'))
        s24, n_s24 = mean_of((sub['seniority_s1']==1) & (sub['period_bucket']=='2024'))
        s26, n_s26 = mean_of((sub['seniority_s1']==1) & (sub['period_bucket']=='2026'))
        rows.append({
            'archetype': arch, 'metric': metric_col,
            'j2_mean_2024': j24, 'j2_mean_2026': j26, 'n_j2_2024': n_j24, 'n_j2_2026': n_j26,
            's1_mean_2024': s24, 's1_mean_2026': s26, 'n_s1_2024': n_s24, 'n_s1_2026': n_s26,
            'gap_2024': (s24-j24) if (pd.notna(s24) and pd.notna(j24)) else np.nan,
            'gap_2026': (s26-j26) if (pd.notna(s26) and pd.notna(j26)) else np.nan,
        })
    out = pd.DataFrame(rows)
    out['gap_change'] = out['gap_2026'] - out['gap_2024']
    return out.sort_values(['metric','archetype'])

js_rows = []
for c in ['requirement_breadth','breadth_resid','ai_strict_bin','ai_broad_bin','mentor_bin','org_scope_count']:
    js_rows.append(j_vs_s(df, c))
js = pd.concat(js_rows, axis=0, ignore_index=True)
js.to_csv(OUT_TAB / 'T28_junior_senior_by_archetype.csv', index=False)

# ---------- Step 6: mentor rate by archetype ----------
print('[step 6] mentor rate by archetype (senior-only, S1)')
senior = df[df['seniority_final'].isin(['mid-senior','director'])]
mentor_by = per_group_rates(senior, 'mentor_bin')
mentor_by.to_csv(OUT_TAB / 'T28_mentor_senior_by_archetype.csv')

# ---------- Step 7: AI/ML archetype deep dive ----------
print('[step 7] AI/ML archetype deep dive')
aiml = df[df['projected_archetype']=='ai_ml_engineering'].copy()
aiml_period = aiml.groupby('period_bucket').agg(
    n=('uid','size'),
    avg_desc_len=('desc_len_chars','mean'),
    avg_breadth=('requirement_breadth','mean'),
    avg_breadth_resid=('breadth_resid','mean'),
    avg_tech=('tech_count','mean'),
    avg_org_scope=('org_scope_count','mean'),
    avg_credential=('credential_stack_depth','mean'),
    ai_strict_share=('ai_strict_bin','mean'),
    ai_broad_share=('ai_broad_bin','mean'),
    mentor_share=('mentor_bin','mean'),
    entry_share_J2=('is_entry_J2','mean'),
)
aiml_period.to_csv(OUT_TAB / 'T28_aiml_period_profile.csv')

# Employer-level within AI/ML
aiml_emp = aiml.groupby(['company_name_canonical','period_bucket']).size().unstack(fill_value=0)
aiml_emp.columns = [f'n_{c}' for c in aiml_emp.columns]
for c in ['n_2024','n_2026']:
    if c not in aiml_emp.columns:
        aiml_emp[c] = 0
aiml_emp['total'] = aiml_emp['n_2024'] + aiml_emp['n_2026']
aiml_emp['present_both'] = ((aiml_emp['n_2024']>0) & (aiml_emp['n_2026']>0))
aiml_emp['only_2026'] = ((aiml_emp['n_2024']==0) & (aiml_emp['n_2026']>0))
aiml_emp['only_2024'] = ((aiml_emp['n_2024']>0) & (aiml_emp['n_2026']==0))

n_both = aiml_emp['present_both'].sum()
n_only26 = aiml_emp['only_2026'].sum()
n_only24 = aiml_emp['only_2024'].sum()
vol_both_24 = aiml_emp.loc[aiml_emp['present_both'],'n_2024'].sum()
vol_both_26 = aiml_emp.loc[aiml_emp['present_both'],'n_2026'].sum()
vol_only24 = aiml_emp.loc[aiml_emp['only_2024'],'n_2024'].sum()
vol_only26 = aiml_emp.loc[aiml_emp['only_2026'],'n_2026'].sum()
print(f'  AI/ML employers present-both {n_both}, only 2024 {n_only24}, only 2026 {n_only26}')
print(f'  AI/ML volume   both-24 {vol_both_24}, both-26 {vol_both_26}, only-24 {vol_only24}, only-26 {vol_only26}')

aiml_summary = {
    'employers_both': int(n_both),
    'employers_only_2024': int(n_only24),
    'employers_only_2026': int(n_only26),
    'volume_2024_from_both': int(vol_both_24),
    'volume_2026_from_both': int(vol_both_26),
    'volume_2024_only24': int(vol_only24),
    'volume_2026_only26': int(vol_only26),
}
with (T28 / 'aiml_employer_entry_mix.json').open('w') as f:
    json.dump(aiml_summary, f, indent=2)

aiml_emp.sort_values('total', ascending=False).head(50).to_csv(OUT_TAB / 'T28_aiml_top_employers.csv')

# AI/ML entry-vs-senior mix
aiml_senior = aiml.groupby(['period_bucket','seniority_final']).size().unstack(fill_value=0)
aiml_senior.to_csv(OUT_TAB / 'T28_aiml_seniority_mix.csv')

# ---------- Global summary file ----------
summary = {
    'n_llm_corpus': int(len(df)),
    'n_2024': int((df['period_bucket']=='2024').sum()),
    'n_2026': int((df['period_bucket']=='2026').sum()),
    'ai_strict_agg_2024': float(df.loc[df['period_bucket']=='2024','ai_strict_bin'].mean()),
    'ai_strict_agg_2026': float(df.loc[df['period_bucket']=='2026','ai_strict_bin'].mean()),
    'ai_broad_agg_2024': float(df.loc[df['period_bucket']=='2024','ai_broad_bin'].mean()),
    'ai_broad_agg_2026': float(df.loc[df['period_bucket']=='2026','ai_broad_bin'].mean()),
    'breadth_agg_2024': float(df.loc[df['period_bucket']=='2024','requirement_breadth'].mean()),
    'breadth_agg_2026': float(df.loc[df['period_bucket']=='2026','requirement_breadth'].mean()),
    'breadth_resid_agg_2024': float(df.loc[df['period_bucket']=='2024','breadth_resid'].mean()),
    'breadth_resid_agg_2026': float(df.loc[df['period_bucket']=='2026','breadth_resid'].mean()),
    'J2_agg_2024': float(df.loc[df['period_bucket']=='2024','is_entry_J2'].mean()),
    'J2_agg_2026': float(df.loc[df['period_bucket']=='2026','is_entry_J2'].mean()),
    'J2_within_pp': float(tot_within_j2*100),
    'J2_between_pp': float(tot_between_j2*100),
    'J2_interaction_pp': float(tot_interaction_j2*100),
    'aiml_summary': aiml_summary,
    'breadth_regress_beta0': float(beta0),
    'breadth_regress_beta1': float(beta1),
    'breadth_regress_R2': float(model.rsquared),
}
with (T28 / 'T28_summary.json').open('w') as f:
    json.dump(summary, f, indent=2)
print('[done] wrote', T28 / 'T28_summary.json')
