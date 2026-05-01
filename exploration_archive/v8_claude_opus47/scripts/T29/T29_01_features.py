"""T29 step 1+3: compute per-posting LLM-authorship features.

Features (per posting):
- llm_vocab_density_per_1k  : density of signature LLM vocab per 1K chars
- em_dash_per_1k            : density of em-dash (— or --) per 1K chars of RAW text
- sentence_mean_len         : mean sentence length (chars)
- sentence_sd_len           : SD of sentence length
- ttr_first1k               : type-token ratio on first 1K chars (cleaned)
- bullet_rate_per_1k        : bullet markers per 1K chars
- para_mean_len / para_sd_len : paragraph length
- desc_len_cleaned          : cleaned description length
- desc_len_raw              : raw description length

We compute features on BOTH cleaned and raw text where appropriate:
- em_dash MUST come from raw description (cleaning normalizes whitespace; em-dash could survive but markdown and hyphens would be changed)
- llm_vocab on cleaned text (English words lowercased)
- bullet rate on raw (bullet markers like '\n- ', '\n* ', ' • ')

Composite authorship_score = z-scored linear combo. We calibrate on the
2024 baseline so 2026 postings are scored relative to the pre-LLM
distribution.

Runs in a streaming manner over the full 63,701-row corpus. Reads raw via
duckdb from unified.parquet and joins cleaned text from swe_cleaned_text.
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
T29 = Path('/home/jihgaboot/gabor/job-research/exploration/artifacts/T29')
OUT_TAB = Path('/home/jihgaboot/gabor/job-research/exploration/tables/T29')
T29.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

UNIFIED = '/home/jihgaboot/gabor/job-research/data/unified.parquet'

print('[step 1] load cleaned text for SWE LinkedIn')
cleaned = pq.read_table(SHARED / 'swe_cleaned_text.parquet',
    columns=['uid','source','period','text_source','is_aggregator','company_name_canonical','seniority_final','description_cleaned'],
).to_pandas()
print(f'  cleaned rows {len(cleaned)}')

print('[step 2] pull raw description via duckdb')
con = duckdb.connect()
raw = con.execute(f"""
  SELECT uid, description AS raw_desc
  FROM parquet_scan('{UNIFIED}')
  WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE
""").fetchdf()
print(f'  raw rows {len(raw)}')
df = cleaned.merge(raw, on='uid', how='left')
print(f'  merged rows {len(df)}')

# missing raw?
miss = df['raw_desc'].isna().sum()
print(f'  missing raw: {miss}')
if miss:
    df['raw_desc'] = df['raw_desc'].fillna('')

# Prefill cleaned desc with empty for safety
df['description_cleaned'] = df['description_cleaned'].fillna('')

# ----- feature regex -----
LLM_VOCAB = re.compile(
    r'\b('
    r'delve|tapestry|leverage|leverages|leveraging|leveraged|robust|unleash|embark on|navigate|navigates|navigating|'
    r'cutting[- ]edge|in the realm of|comprehensive|seamless|seamlessly|furthermore|moreover|'
    r"it'?s worth noting|notably|align with|aligns with|aligning with|at the forefront|pivotal|"
    r'harness|harnesses|harnessing|dynamic|vibrant|mission[- ]critical|thrives|thriving|'
    r'ambiguous|ambiguity|trade[- ]offs?|stakeholders?|'
    # less common but shared-LLM:
    r'deep dive|deep dives|ever[- ]evolving|ever[- ]changing|in today'+"'"+r's (?:fast[- ]paced |rapidly (?:evolving|changing) )?(?:world|landscape)'
    r')\b',
    re.I,
)
EM_DASH_RX = re.compile(r'(—| -- | --|— )')  # em-dash OR double-hyphen-as-dash (with surrounding space)
EM_DASH_UNICODE_RX = re.compile(r'—')
EM_DASH_ASCII_RX = re.compile(r'(?:^| )--(?= |$)', re.M)
# Sentence splitter: split on . ? ! followed by whitespace+upper or end
SENT_SPLIT_RX = re.compile(r'(?<=[.!?])\s+(?=[A-Z(])')
BULLET_RX = re.compile(r'(^|\n)\s*(?:[-*•·]|\d+\.)\s+', re.M)
WORD_RX = re.compile(r"[a-zA-Z']+")


def feat_row(raw: str, cleaned: str) -> dict:
    L_raw = max(len(raw), 1)
    L_cl = max(len(cleaned), 1)
    # vocab density on cleaned
    vocab_hits = LLM_VOCAB.findall(cleaned)
    vocab_density_k = 1000.0 * len(vocab_hits) / L_cl
    # em-dash on raw
    ed_u = len(EM_DASH_UNICODE_RX.findall(raw))
    ed_a = len(EM_DASH_ASCII_RX.findall(raw))
    em_dash_k = 1000.0 * (ed_u + ed_a) / L_raw
    # sentences — derived from cleaned (sentence splits preserved across cleaning)
    if L_cl > 10:
        sents = [s for s in SENT_SPLIT_RX.split(cleaned) if len(s.strip()) > 0]
    else:
        sents = []
    if sents:
        lens = np.array([len(s) for s in sents], dtype=np.float32)
        sent_mean = float(lens.mean())
        sent_sd = float(lens.std())
        n_sent = len(sents)
    else:
        sent_mean, sent_sd, n_sent = np.nan, np.nan, 0
    # TTR on first 1000 chars cleaned
    head = cleaned[:1000]
    tokens = WORD_RX.findall(head.lower())
    if tokens:
        ttr = len(set(tokens)) / len(tokens)
    else:
        ttr = np.nan
    # bullets on raw
    n_bul = len(BULLET_RX.findall(raw))
    bullet_k = 1000.0 * n_bul / L_raw
    # paragraphs on raw (double-newline separators)
    paras = [p for p in re.split(r'\n{2,}', raw) if len(p.strip()) > 0]
    if paras:
        plens = np.array([len(p) for p in paras], dtype=np.float32)
        para_mean = float(plens.mean())
        para_sd = float(plens.std())
    else:
        para_mean, para_sd = np.nan, np.nan
    return {
        'llm_vocab_density_per_1k': vocab_density_k,
        'em_dash_per_1k': em_dash_k,
        'sentence_mean_len': sent_mean,
        'sentence_sd_len': sent_sd,
        'n_sentences': n_sent,
        'ttr_first1k': ttr,
        'bullet_per_1k': bullet_k,
        'para_mean_len': para_mean,
        'para_sd_len': para_sd,
        'desc_len_cleaned': L_cl,
        'desc_len_raw': L_raw,
    }

print('[step 3] compute features row-by-row')
feat_rows = []
for i, (raw, cl) in enumerate(zip(df['raw_desc'].values, df['description_cleaned'].values)):
    feat_rows.append(feat_row(raw if isinstance(raw, str) else '', cl if isinstance(cl, str) else ''))
    if (i+1) % 10000 == 0:
        print(f'  {i+1}/{len(df)}')
feat = pd.DataFrame(feat_rows)
df = pd.concat([df.reset_index(drop=True), feat], axis=1)

# period bucket
df['period_bucket'] = df['period'].str.startswith('2024').map({True:'2024', False:'2026'})

# ----- Composite score -----
# z-score each feature using 2024 baseline (so score is relative to pre-LLM).
# Then weighted sum. Weights are equal across the 4 headline signals.
head_feats = ['llm_vocab_density_per_1k','em_dash_per_1k','sentence_mean_len','bullet_per_1k']
# NOTE: sentence_sd and para variance are secondary and can be confounded,
# so we keep them out of the primary composite. TTR is (1 - ttr) sign — low
# TTR is more LLM-like. We flip it.
def zscore_by_baseline(col, baseline_mask):
    mu = df.loc[baseline_mask, col].mean()
    sd = df.loc[baseline_mask, col].std()
    if sd == 0 or not np.isfinite(sd):
        return np.zeros(len(df))
    return (df[col].values - mu) / sd

base = df['period_bucket'] == '2024'
z_parts = []
for c in head_feats:
    z_parts.append(zscore_by_baseline(c, base))
# flip ttr (low diversity -> LLM)
ttr_z = -zscore_by_baseline('ttr_first1k', base)
z_parts.append(ttr_z)
score = np.nanmean(np.vstack(z_parts), axis=0)
df['authorship_score'] = score

# ----- save -----
print('[step 4] save authorship scores')
keep = ['uid','source','period','period_bucket','text_source','is_aggregator','company_name_canonical','seniority_final'] + \
       ['llm_vocab_density_per_1k','em_dash_per_1k','sentence_mean_len','sentence_sd_len','n_sentences','ttr_first1k','bullet_per_1k','para_mean_len','para_sd_len','desc_len_cleaned','desc_len_raw','authorship_score']
df[keep].to_csv(OUT_TAB / 'authorship_scores.csv', index=False)
df[keep].to_parquet(T29 / 'authorship_scores.parquet', index=False)
print(f'  saved {OUT_TAB / "authorship_scores.csv"}')

# summary by period + text_source
summary = df.groupby(['period_bucket','text_source']).agg(
    n=('uid','size'),
    vocab_mean=('llm_vocab_density_per_1k','mean'),
    em_dash_mean=('em_dash_per_1k','mean'),
    sent_mean=('sentence_mean_len','mean'),
    ttr_mean=('ttr_first1k','mean'),
    bullet_mean=('bullet_per_1k','mean'),
    desc_len_mean=('desc_len_cleaned','mean'),
    score_mean=('authorship_score','mean'),
    score_median=('authorship_score','median'),
    score_sd=('authorship_score','std'),
)
print(summary)
summary.to_csv(OUT_TAB / 'authorship_feature_means.csv')

# period-only summary
pd_sum = df.groupby('period_bucket').agg(
    n=('uid','size'),
    score_mean=('authorship_score','mean'),
    score_median=('authorship_score','median'),
    score_sd=('authorship_score','std'),
    vocab_mean=('llm_vocab_density_per_1k','mean'),
    em_dash_mean=('em_dash_per_1k','mean'),
    bullet_mean=('bullet_per_1k','mean'),
    ttr_mean=('ttr_first1k','mean'),
    sent_mean=('sentence_mean_len','mean'),
)
print(pd_sum)
pd_sum.to_csv(OUT_TAB / 'authorship_score_by_period.csv')

json_summary = {
    'n': int(len(df)),
    'n_2024': int((df['period_bucket']=='2024').sum()),
    'n_2026': int((df['period_bucket']=='2026').sum()),
    'score_median_2024': float(df.loc[df['period_bucket']=='2024','authorship_score'].median()),
    'score_median_2026': float(df.loc[df['period_bucket']=='2026','authorship_score'].median()),
    'score_mean_2024': float(df.loc[df['period_bucket']=='2024','authorship_score'].mean()),
    'score_mean_2026': float(df.loc[df['period_bucket']=='2026','authorship_score'].mean()),
    'score_sd_2024': float(df.loc[df['period_bucket']=='2024','authorship_score'].std()),
    'score_sd_2026': float(df.loc[df['period_bucket']=='2026','authorship_score'].std()),
    'feature_weights': 'equal z-score across [llm_vocab_k, em_dash_k, sent_mean, bullet_k, -ttr_first1k]',
    'baseline_period': '2024',
}
with (T29 / 'T29_feature_summary.json').open('w') as f:
    json.dump(json_summary, f, indent=2)
