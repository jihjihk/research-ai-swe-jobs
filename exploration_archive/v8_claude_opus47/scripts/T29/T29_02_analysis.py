"""T29 steps 4-8: distribution / per-company / correlations / low-LLM subset test.

Reads:
- exploration/artifacts/T29/authorship_scores.parquet (n=63,701; period, score, features)
- exploration/artifacts/shared/swe_cleaned_text.parquet (text, seniority, source, company)
- exploration/artifacts/T11/T11_posting_features.parquet (breadth, tech_count, etc.)
- exploration/artifacts/T28/projected_archetypes.parquet (for optional archetype cuts)

Writes:
- T29_score_distribution_period.csv
- T29_score_quantiles.csv
- T29_most_llm_companies.csv
- T29_least_llm_companies.csv
- T29_correlations_posting.csv
- T29_company_panel_corr.csv
- T29_low_llm_subset_headlines.csv
- T29_low_llm_vs_full_detailed.csv
- T29_summary.json

Text patterns (AI strict/broad, mentor, requirements-section) are applied to
cleaned text from the shared artifact. Length-residualized requirement_breadth
is computed from an OLS fit on the 2024 LLM rows (same recipe as T28).
"""
from __future__ import annotations
import re
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
from pathlib import Path

SHARED = Path('/home/jihgaboot/gabor/job-research/exploration/artifacts/shared')
T11 = Path('/home/jihgaboot/gabor/job-research/exploration/artifacts/T11')
T28 = Path('/home/jihgaboot/gabor/job-research/exploration/artifacts/T28')
T29 = Path('/home/jihgaboot/gabor/job-research/exploration/artifacts/T29')
OUT_TAB = Path('/home/jihgaboot/gabor/job-research/exploration/tables/T29')

print('[load] authorship scores parquet')
au = pq.read_table(T29 / 'authorship_scores.parquet').to_pandas()
print(f'  n={len(au)}')

print('[load] cleaned text (for AI/mentor recomputation)')
cl = pq.read_table(SHARED / 'swe_cleaned_text.parquet',
    columns=['uid','source','period','text_source','is_aggregator','company_name_canonical','seniority_final','description_cleaned']
).to_pandas()
print(f'  n={len(cl)}')

print('[load] T11 features')
t11 = pq.read_table(T11 / 'T11_posting_features.parquet').to_pandas()
print(f'  n={len(t11)}')

print('[load] projected archetypes (LLM only)')
proj = pq.read_table(T28 / 'projected_archetypes.parquet').to_pandas()

# ---- merge ----
df = au.merge(cl[['uid','description_cleaned']], on='uid', how='left')
df = df.merge(t11[['uid','tech_count','ai_count_tech','desc_len_chars','org_scope_count','credential_stack_depth','requirement_breadth']], on='uid', how='left')
df = df.merge(proj[['uid','projected_archetype']], on='uid', how='left')
df['description_cleaned'] = df['description_cleaned'].fillna('')

# ---- recompute AI / mentor flags on ALL rows (including text_source='raw') ----
AI_STRICT_RX = re.compile(r'\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|huggingface|hugging face)\b', re.I)
AI_BROAD_EXTRA = re.compile(r'\b(ai|artificial intelligence|ml|machine learning|llm|large language model|generative ai|genai|anthropic)\b', re.I)
MENTOR_RX = re.compile(r'\b(mentor|mentoring|mentored|mentorship|coach|coaching|coached|hiring|headcount|performance review|performance reviews)\b', re.I)
# Requirements section marker — per T13 approach we approximate with header patterns + bullet density in the "requirements" portion.
# Here we use a simpler proxy: presence of a 'requirements' / 'qualifications' heading, and whether a posting has a distinct requirements
# section. Since we don't have T13 classifier artifacts we emit a binary indicator instead of characters.
REQ_HEADER_RX = re.compile(r'(?:\n|\r|^)\s*(?:requirements|qualifications|what you(?:\'| ha)ll bring|what we\'re looking for|what we look for|what you need|minimum qualifications|preferred qualifications)\b', re.I)

print('[compute] text flags on full 63,701')
t = df['description_cleaned'].values
ai_strict = np.zeros(len(df), dtype=np.int8)
ai_broad = np.zeros(len(df), dtype=np.int8)
mentor = np.zeros(len(df), dtype=np.int8)
req_header = np.zeros(len(df), dtype=np.int8)
for i, s in enumerate(t):
    if not isinstance(s, str):
        continue
    if AI_STRICT_RX.search(s):
        ai_strict[i] = 1
        ai_broad[i] = 1
    elif AI_BROAD_EXTRA.search(s):
        ai_broad[i] = 1
    if MENTOR_RX.search(s):
        mentor[i] = 1
    if REQ_HEADER_RX.search(s):
        req_header[i] = 1
df['ai_strict_bin'] = ai_strict
df['ai_broad_bin'] = ai_broad
df['mentor_bin'] = mentor
df['req_header_bin'] = req_header

# ---- length-residualize requirement_breadth (baseline: 2024 all rows with breadth) ----
mask_base = (df['period_bucket']=='2024') & df['requirement_breadth'].notna() & df['desc_len_chars'].notna() & (df['desc_len_chars']>0)
X = np.log(df.loc[mask_base,'desc_len_chars'].astype(float).clip(lower=1).values)
y = df.loc[mask_base,'requirement_breadth'].astype(float).values
X1 = sm.add_constant(X)
ols = sm.OLS(y, X1).fit()
beta0, beta1 = ols.params
print(f'  breadth ~ a + b*log(len): beta0={beta0:.3f} beta1={beta1:.3f} R2={ols.rsquared:.3f}')
logL = np.log(df['desc_len_chars'].astype(float).clip(lower=1).values)
df['breadth_resid'] = df['requirement_breadth'].astype(float) - (beta0 + beta1 * logL)

# ---- Senior mentor slice mask ----
df['is_senior'] = df['seniority_final'].isin(['mid-senior','director']).astype(int)

# ---- 4. Distribution by period ----
print('[step 4] score distribution')
quantiles = [0.05,0.1,0.25,0.5,0.75,0.9,0.95]
q_rows = []
for pb, sub in df.groupby('period_bucket'):
    q = sub['authorship_score'].quantile(quantiles).to_dict()
    q_rows.append({'period_bucket': pb, 'n': int(len(sub)), **{f'q{int(p*100):02d}': float(q[p]) for p in quantiles}})
q_out = pd.DataFrame(q_rows)
q_out.to_csv(OUT_TAB / 'T29_score_quantiles.csv', index=False)

# by text_source × period
dist_rows = []
for (pb, ts), sub in df.groupby(['period_bucket','text_source']):
    dist_rows.append({
        'period_bucket': pb, 'text_source': ts, 'n': int(len(sub)),
        'score_mean': float(sub['authorship_score'].mean()),
        'score_median': float(sub['authorship_score'].median()),
        'score_sd': float(sub['authorship_score'].std()),
        'frac_score_gt_1': float((sub['authorship_score']>1).mean()),
        'frac_score_gt_2': float((sub['authorship_score']>2).mean()),
    })
pd.DataFrame(dist_rows).to_csv(OUT_TAB / 'T29_score_distribution_period.csv', index=False)

# ---- 5. Per-company profile ----
print('[step 5] per-company profile')
comp = df[df['company_name_canonical'].notna()].copy()
# min 10 postings across both periods to be reliable
comp_stats = comp.groupby('company_name_canonical').agg(
    n=('uid','size'),
    n_2024=('period_bucket', lambda x: int((x=='2024').sum())),
    n_2026=('period_bucket', lambda x: int((x=='2026').sum())),
    score_mean=('authorship_score','mean'),
    score_median=('authorship_score','median'),
    score_sd=('authorship_score','std'),
    vocab_mean=('llm_vocab_density_per_1k','mean'),
    em_dash_mean=('em_dash_per_1k','mean'),
    bullet_mean=('bullet_per_1k','mean'),
).reset_index()
# change panel: companies with >=5 postings in BOTH periods
panel = comp[comp.groupby('company_name_canonical')['period_bucket'].transform(lambda x: (x=='2024').sum() >= 5 and (x=='2026').sum() >= 5)]
print(f'  panel companies: {panel["company_name_canonical"].nunique()}')
panel_cmp = panel.groupby(['company_name_canonical','period_bucket']).agg(
    n=('uid','size'),
    score_mean=('authorship_score','mean'),
    vocab_mean=('llm_vocab_density_per_1k','mean'),
    em_dash_mean=('em_dash_per_1k','mean'),
    bullet_mean=('bullet_per_1k','mean'),
    ai_strict=('ai_strict_bin','mean'),
    ai_broad=('ai_broad_bin','mean'),
    breadth_resid=('breadth_resid','mean'),
    desc_len=('desc_len_chars','mean'),
).unstack('period_bucket')
panel_cmp.columns = ['_'.join(map(str,c)) for c in panel_cmp.columns]
panel_cmp = panel_cmp.reset_index()
panel_cmp['score_delta'] = panel_cmp['score_mean_2026'] - panel_cmp['score_mean_2024']
panel_cmp['ai_strict_delta'] = panel_cmp['ai_strict_2026'] - panel_cmp['ai_strict_2024']
panel_cmp['ai_broad_delta'] = panel_cmp['ai_broad_2026'] - panel_cmp['ai_broad_2024']
panel_cmp['breadth_resid_delta'] = panel_cmp['breadth_resid_2026'] - panel_cmp['breadth_resid_2024']
panel_cmp['desc_len_delta'] = panel_cmp['desc_len_2026'] - panel_cmp['desc_len_2024']
panel_cmp.to_csv(OUT_TAB / 'T29_company_panel.csv', index=False)
print(f'  panel rows: {len(panel_cmp)}')

# top and bottom 20 per-company (by 2026 mean score, min 10 postings 2026)
comp_26 = comp[comp['period_bucket']=='2026'].groupby('company_name_canonical').agg(
    n_2026=('uid','size'),
    score_mean_2026=('authorship_score','mean'),
    vocab_mean_2026=('llm_vocab_density_per_1k','mean'),
    em_dash_mean_2026=('em_dash_per_1k','mean'),
    bullet_mean_2026=('bullet_per_1k','mean'),
).reset_index()
comp_26 = comp_26[comp_26['n_2026']>=10]
comp_26_top = comp_26.sort_values('score_mean_2026', ascending=False).head(20)
comp_26_bot = comp_26.sort_values('score_mean_2026', ascending=True).head(20)
comp_26_top.to_csv(OUT_TAB / 'T29_most_llm_companies.csv', index=False)
comp_26_bot.to_csv(OUT_TAB / 'T29_least_llm_companies.csv', index=False)

# ---- 6. Correlation with Wave 2 findings ----
print('[step 6] correlations posting and company-panel')
# Posting-level correlations (each period separately, and pooled)
def corrs_block(sub, label):
    out = {}
    for col in ['desc_len_chars','tech_count','ai_strict_bin','ai_broad_bin','breadth_resid','requirement_breadth','mentor_bin','req_header_bin']:
        v = sub[['authorship_score', col]].dropna()
        if len(v) < 30:
            out[col] = np.nan
        else:
            out[col] = float(v['authorship_score'].corr(v[col]))
    out['slice'] = label
    out['n'] = int(len(sub))
    return out

post_rows = [corrs_block(df, 'all'),
             corrs_block(df[df['period_bucket']=='2024'], '2024'),
             corrs_block(df[df['period_bucket']=='2026'], '2026')]
pd.DataFrame(post_rows).to_csv(OUT_TAB / 'T29_correlations_posting.csv', index=False)
print(post_rows)

# Company-panel correlation: change in score vs change in AI-mention / breadth / desc-length
pc_rows = []
for col_delta in ['ai_strict_delta','ai_broad_delta','breadth_resid_delta','desc_len_delta']:
    v = panel_cmp[['score_delta', col_delta]].dropna()
    if len(v) < 30:
        pc_rows.append({'metric_delta': col_delta, 'n': int(len(v)), 'rho': np.nan})
    else:
        pc_rows.append({'metric_delta': col_delta, 'n': int(len(v)), 'rho': float(v['score_delta'].corr(v[col_delta]))})
pd.DataFrame(pc_rows).to_csv(OUT_TAB / 'T29_company_panel_corr.csv', index=False)

# ---- 7. UNIFYING-MECHANISM TEST: bottom 40% by authorship score ----
# Critical: we need to define the LOW subset CONSISTENTLY across 2024 and 2026.
# If we use a global quantile and most LLM-like postings are 2026, we'll
# drop almost all 2026 postings. Instead we use WITHIN-PERIOD quantile:
# keep bottom 40% by score in each period. This preserves period comparison
# while subsetting to low-LLM-authored rows.
print('[step 7] unifying-mechanism test: bottom-40 by score (within period)')
df['within_period_qtile'] = df.groupby('period_bucket')['authorship_score'].rank(pct=True)
low = df[df['within_period_qtile'] <= 0.4].copy()
print(f'  low subset n={len(low)}  (2024={int((low["period_bucket"]=="2024").sum())} 2026={int((low["period_bucket"]=="2026").sum())})')

# Also compute GLOBAL threshold subset for sensitivity
global_thr = df['authorship_score'].quantile(0.40)
low_global = df[df['authorship_score'] <= global_thr].copy()
print(f'  low global n={len(low_global)} threshold={global_thr:.3f}')

# Full-corpus and low-subset Δ on headline metrics
def period_delta(d, col, binary=True):
    d24 = d[d['period_bucket']=='2024']
    d26 = d[d['period_bucket']=='2026']
    if binary:
        m24 = d24[col].mean(); m26 = d26[col].mean()
        se24 = np.sqrt(m24*(1-m24)/max(len(d24),1))
        se26 = np.sqrt(m26*(1-m26)/max(len(d26),1))
    else:
        m24 = d24[col].mean(); m26 = d26[col].mean()
        se24 = d24[col].std()/np.sqrt(max(len(d24),1))
        se26 = d26[col].std()/np.sqrt(max(len(d26),1))
    delta = m26 - m24
    se = np.sqrt(se24**2 + se26**2)
    return m24, m26, delta, se, int(len(d24)), int(len(d26))

metrics = [
    ('ai_strict_bin', True, 'AI-mention strict'),
    ('ai_broad_bin', True, 'AI-mention broad'),
    ('mentor_bin', True, 'Mentor (senior-only is applied separately)'),
    ('req_header_bin', True, 'Requirements-header presence'),
    ('breadth_resid', False, 'requirement_breadth length-residualized'),
    ('requirement_breadth', False, 'requirement_breadth raw'),
    ('desc_len_chars', False, 'Description length'),
]

rows = []
for subset_name, subset in [('full', df), ('low40_within_period', low), ('low40_global', low_global)]:
    for col, binary, lbl in metrics:
        m24, m26, delta, se, n24, n26 = period_delta(subset, col, binary=binary)
        rows.append({
            'subset': subset_name, 'metric': col, 'label': lbl,
            'mean_2024': m24, 'mean_2026': m26, 'delta': delta, 'se': se,
            'snr': abs(delta)/se if se>0 else np.nan,
            'n_2024': n24, 'n_2026': n26,
        })
    # senior-only mentor (S1)
    sen_sub = subset[subset['is_senior']==1]
    m24, m26, delta, se, n24, n26 = period_delta(sen_sub, 'mentor_bin', binary=True)
    rows.append({
        'subset': subset_name, 'metric': 'mentor_senior_only', 'label': 'Mentor senior-only (S1)',
        'mean_2024': m24, 'mean_2026': m26, 'delta': delta, 'se': se,
        'snr': abs(delta)/se if se>0 else np.nan,
        'n_2024': n24, 'n_2026': n26,
    })

headlines = pd.DataFrame(rows)
headlines.to_csv(OUT_TAB / 'T29_low_llm_subset_headlines.csv', index=False)

# Attenuation factor relative to full-corpus delta
piv = headlines.pivot_table(index='metric', columns='subset', values='delta')
piv['att_within'] = piv['low40_within_period'] / piv['full'].replace({0: np.nan})
piv['att_global'] = piv['low40_global'] / piv['full'].replace({0: np.nan})
piv.to_csv(OUT_TAB / 'T29_low_llm_vs_full_detailed.csv')

# ---- 8. Archetype cut (bonus) ----
if 'projected_archetype' in df.columns:
    # only where archetype known (LLM rows)
    dff = df[df['projected_archetype'].notna()].copy()
    arch_rows = []
    for arch, sub in dff.groupby('projected_archetype'):
        if len(sub) < 50:
            continue
        q = sub.groupby('period_bucket')['authorship_score'].median().to_dict()
        arch_rows.append({
            'archetype': arch,
            'n_2024': int((sub['period_bucket']=='2024').sum()),
            'n_2026': int((sub['period_bucket']=='2026').sum()),
            'score_median_2024': float(q.get('2024', np.nan)),
            'score_median_2026': float(q.get('2026', np.nan)),
            'score_delta_median': float(q.get('2026', np.nan) - q.get('2024', np.nan)) if ('2024' in q and '2026' in q) else np.nan,
        })
    pd.DataFrame(arch_rows).sort_values('n_2026', ascending=False).to_csv(OUT_TAB / 'T29_score_by_archetype.csv', index=False)

# ---- Save summary JSON ----
summ = {
    'n_total': int(len(df)),
    'n_2024': int((df['period_bucket']=='2024').sum()),
    'n_2026': int((df['period_bucket']=='2026').sum()),
    'score_median_2024': float(df[df['period_bucket']=='2024']['authorship_score'].median()),
    'score_median_2026': float(df[df['period_bucket']=='2026']['authorship_score'].median()),
    'score_mean_delta': float(df[df['period_bucket']=='2026']['authorship_score'].mean() - df[df['period_bucket']=='2024']['authorship_score'].mean()),
    'breadth_regress': {'beta0': float(beta0), 'beta1': float(beta1), 'R2': float(ols.rsquared)},
    'low40_within_period_n': int(len(low)),
    'low40_global_n': int(len(low_global)),
    'panel_n_companies': int(panel['company_name_canonical'].nunique()),
    'attenuations_within_period': {m: float(v) if v==v else None for m, v in piv['att_within'].items()},
    'attenuations_global': {m: float(v) if v==v else None for m, v in piv['att_global'].items()},
}
with (T29 / 'T29_analysis_summary.json').open('w') as f:
    json.dump(summ, f, indent=2)
print('[done]')
print(json.dumps(summ, indent=2))
