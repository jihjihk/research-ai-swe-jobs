#!/usr/bin/env python3
"""
T12: Open-ended text evolution — Fightin' Words, emerging/disappearing terms,
bigram analysis, BERTopic cross-validation.

Primary comparison: arshkon (2024-04) vs scraped (2026-03), all SWE.
Secondary: seniority-stratified, within-2024 calibration.
"""

import sys, os, json, warnings, re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import polygamma
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
                      'figure.figsize': (12, 8), 'savefig.bbox': 'tight'})

BASE = Path('/home/jihgaboot/gabor/job-research')
FIG_DIR = BASE / 'exploration/figures/T12'
TAB_DIR = BASE / 'exploration/tables/T12'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

SHARED = BASE / 'exploration/artifacts/shared'

# ──────────────────────────────────────────────────────────────────────
# Semantic category mapping
# ──────────────────────────────────────────────────────────────────────
CATEGORY_PATTERNS = {
    'ai_tool': r'\b(copilot|chatgpt|github copilot|cursor|codewhisperer|tabnine|codex|anthropic|openai|gemini|claude|gpt|llm|llms|large language model|generative ai|gen ai)\b',
    'ai_domain': r'\b(machine learning|deep learning|nlp|natural language|computer vision|neural net|reinforcement learning|transformer|diffusion|embedding|vector database|rag|retrieval augmented|fine.?tun|prompt engineer|ai agent|agentic|multimodal|ai safety|responsible ai|alignment)\b',
    'tech_stack': r'\b(python|java|javascript|typescript|react|angular|vue|node|docker|kubernetes|aws|azure|gcp|terraform|sql|nosql|postgres|mongodb|redis|kafka|spark|hadoop|airflow|ci.?cd|git|linux|rest|graphql|microservice|serverless|cloud|devops|golang|rust|swift|kotlin|scala|ruby|rails|django|flask|spring|fastapi|nextjs|webpack|html|css)\b',
    'org_scope': r'\b(cross.?functional|stakeholder|collaborate|partnership|alignment|roadmap|strategy|initiative|prioriti|backlog|sprint|scrum|kanban|agile|waterfall|jira|confluence|product owner|program manager)\b',
    'mgmt': r'\b(lead|mentor|coach|manag|supervis|hire|recruit|headcount|team.?size|direct report|performance review|career development|one.?on.?one|skip.?level|staff engineer|principal|architect|tech lead|engineering manager|vp of engineering|cto|director)\b',
    'sys_design': r'\b(system design|architecture|scalab|distributed|high.?availability|fault.?tolerant|load.?balanc|caching|sharding|replication|latency|throughput|sla|reliability|observability|monitoring|alerting|incident|on.?call|sre|site reliability|infrastructure)\b',
    'method': r'\b(test.?driven|tdd|bdd|code review|pull request|pair program|mob program|trunk.?based|feature flag|a.?b test|canary|blue.?green|continuous|deploy|release|version control|documentation|technical writing|rfc|design doc|specification)\b',
    'credential': r'\b(bachelor|master|phd|degree|certification|certified|cka|aws certified|pmp|csm|comptia|cissp|ccna|bootcamp|self.?taught|cs degree|computer science degree|accredited|gpa)\b',
    'soft_skill': r'\b(communicat|teamwork|collaborat|problem.?solv|critical think|adaptab|creative|innovati|curios|passion|self.?motivated|detail.?oriented|time management|multitask|ownership|accountability|empathy|emotional intelligence|growth mindset|resilient|flexible)\b',
    'noise': r'\b(equal opportunity|eeo|disability|veteran|affirmative action|we are an|reasonable accommodation|drug.?free|background check|e.?verify|401k|dental|vision|pto|paid time off|health insurance|benefits|perks|salary range|compensation|bonus|equity|stock option|rsu|relocation|visa|sponsorship|h1b|us citizen)\b',
}

def categorize_term(term):
    """Assign semantic category to a term/phrase."""
    t = term.lower()
    for cat, pat in CATEGORY_PATTERNS.items():
        if re.search(pat, t):
            return cat
    return 'uncategorized'


# ──────────────────────────────────────────────────────────────────────
# Fightin' Words (Monroe et al. 2008) — log-odds ratio with informative Dirichlet prior
# ──────────────────────────────────────────────────────────────────────
def fighting_words(counts_a, counts_b, vocab, alpha_prior=0.01):
    """
    Compute log-odds ratio with informative Dirichlet prior.
    Returns DataFrame with z-scores (positive = more in corpus_b).
    """
    n_a = counts_a.sum()
    n_b = counts_b.sum()
    alpha_0 = alpha_prior * len(vocab)

    # Posterior expected log-odds
    delta = (np.log((counts_b + alpha_prior) / (n_b + alpha_0 - counts_b - alpha_prior))
             - np.log((counts_a + alpha_prior) / (n_a + alpha_0 - counts_a - alpha_prior)))

    # Variance from approximation
    var = (1.0 / (counts_b + alpha_prior) + 1.0 / (counts_a + alpha_prior))

    z_scores = delta / np.sqrt(var)

    df = pd.DataFrame({
        'term': vocab,
        'count_a': counts_a,
        'count_b': counts_b,
        'rate_a': counts_a / n_a * 1000,
        'rate_b': counts_b / n_b * 1000,
        'log_odds_ratio': delta,
        'z_score': z_scores,
    })
    df['category'] = df['term'].apply(categorize_term)
    return df.sort_values('z_score', ascending=False)


def run_comparison(texts_a, texts_b, label_a, label_b, min_df=5, ngram_range=(1,1)):
    """Run Fightin' Words on two text corpora. Returns full results DataFrame."""
    vec = CountVectorizer(min_df=min_df, max_df=0.95, ngram_range=ngram_range,
                          token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z+#.]{1,}\b')

    all_texts = list(texts_a) + list(texts_b)
    dtm = vec.fit_transform(all_texts)
    vocab = vec.get_feature_names_out()

    n_a = len(texts_a)
    counts_a = np.array(dtm[:n_a].sum(axis=0)).flatten()
    counts_b = np.array(dtm[n_a:].sum(axis=0)).flatten()

    fw = fighting_words(counts_a, counts_b, vocab)
    fw['corpus_a'] = label_a
    fw['corpus_b'] = label_b
    fw['n_a'] = n_a
    fw['n_b'] = len(texts_b)
    return fw


def get_emerging_disappearing_accelerating(fw_df, rate_threshold_high=10.0, rate_threshold_low=1.0, accel_factor=3):
    """Identify emerging, disappearing, and accelerating terms from FW results."""
    # Emerging: >1 per 1K in B, <0.1 per 1K in A
    emerging = fw_df[(fw_df['rate_b'] > rate_threshold_high) & (fw_df['rate_a'] < rate_threshold_low)].copy()
    emerging = emerging.sort_values('z_score', ascending=False)

    # Disappearing: >1 per 1K in A, <0.1 per 1K in B
    disappearing = fw_df[(fw_df['rate_a'] > rate_threshold_high) & (fw_df['rate_b'] < rate_threshold_low)].copy()
    disappearing = disappearing.sort_values('z_score', ascending=True)

    # Accelerating: existed in both, grew >accel_factor
    mask_accel = (fw_df['rate_a'] > 0.5) & (fw_df['rate_b'] > 0.5) & (fw_df['rate_b'] / fw_df['rate_a'].clip(0.001) > accel_factor)
    accelerating = fw_df[mask_accel].copy()
    accelerating['growth_ratio'] = accelerating['rate_b'] / accelerating['rate_a'].clip(0.001)
    accelerating = accelerating.sort_values('growth_ratio', ascending=False)

    return emerging, disappearing, accelerating


# ──────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────
print("Loading shared SWE cleaned text...")
con = duckdb.connect()
df = con.execute(f"""
    SELECT uid, description_cleaned, text_source, source, period,
           seniority_final, seniority_3level, is_aggregator,
           company_name_canonical
    FROM '{SHARED}/swe_cleaned_text.parquet'
""").fetchdf()

print(f"Total rows: {len(df)}")
print(f"Source distribution:\n{df.groupby(['source','period']).size()}")

# Filter out very short texts
df = df[df['description_cleaned'].str.len() > 50].copy()
print(f"After length filter: {len(df)}")

# ──────────────────────────────────────────────────────────────────────
# Company stoplist (for reference, but text is already company-cleaned)
# ──────────────────────────────────────────────────────────────────────
stoplist = set()
with open(SHARED / 'company_stoplist.txt') as f:
    for line in f:
        stoplist.add(line.strip().lower())
print(f"Company stoplist: {len(stoplist)} tokens")

# ──────────────────────────────────────────────────────────────────────
# Define corpora
# ──────────────────────────────────────────────────────────────────────
arshkon = df[df['source'] == 'kaggle_arshkon']
asaniczka = df[df['source'] == 'kaggle_asaniczka']
scraped = df[df['source'] == 'scraped']

print(f"\nCorpus sizes:")
print(f"  arshkon (2024-04): {len(arshkon)}")
print(f"  asaniczka (2024-01): {len(asaniczka)}")
print(f"  scraped (2026-03): {len(scraped)}")

# ──────────────────────────────────────────────────────────────────────
# PRIMARY COMPARISON: arshkon 2024 vs scraped 2026 (unigrams)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PRIMARY: arshkon (2024-04) vs scraped (2026-03) — unigrams")
print("="*70)

fw_primary = run_comparison(
    arshkon['description_cleaned'].values,
    scraped['description_cleaned'].values,
    'arshkon_2024', 'scraped_2026',
    min_df=5, ngram_range=(1,1)
)

# Top 100 each direction
top_2026 = fw_primary.head(100)
top_2024 = fw_primary.tail(100).sort_values('z_score')

print(f"\nTop 20 terms distinguishing 2026 (positive z):")
for _, row in top_2026.head(20).iterrows():
    print(f"  {row['term']:25s}  z={row['z_score']:7.1f}  rate_2024={row['rate_a']:6.1f}  rate_2026={row['rate_b']:6.1f}  [{row['category']}]")

print(f"\nTop 20 terms distinguishing 2024 (negative z):")
for _, row in top_2024.head(20).iterrows():
    print(f"  {row['term']:25s}  z={row['z_score']:7.1f}  rate_2024={row['rate_a']:6.1f}  rate_2026={row['rate_b']:6.1f}  [{row['category']}]")

# Category summary
cat_summary_2026 = top_2026.groupby('category').agg(
    n_terms=('term', 'count'),
    mean_z=('z_score', 'mean'),
    top_terms=('term', lambda x: ', '.join(x.head(5)))
).sort_values('n_terms', ascending=False)

cat_summary_2024 = top_2024.groupby('category').agg(
    n_terms=('term', 'count'),
    mean_z=('z_score', 'mean'),
    top_terms=('term', lambda x: ', '.join(x.head(5)))
).sort_values('n_terms', ascending=False)

print(f"\n--- Category summary (top 100 terms distinguishing 2026) ---")
print(cat_summary_2026.to_string())
print(f"\n--- Category summary (top 100 terms distinguishing 2024) ---")
print(cat_summary_2024.to_string())

# Emerging / disappearing / accelerating
emerging, disappearing, accelerating = get_emerging_disappearing_accelerating(fw_primary)
print(f"\nEmerging terms (>10/1K in 2026, <1/1K in 2024): {len(emerging)}")
if len(emerging) > 0:
    for _, row in emerging.head(20).iterrows():
        print(f"  {row['term']:25s}  rate_2024={row['rate_a']:6.2f}  rate_2026={row['rate_b']:6.2f}  [{row['category']}]")

print(f"\nDisappearing terms (>10/1K in 2024, <1/1K in 2026): {len(disappearing)}")
if len(disappearing) > 0:
    for _, row in disappearing.head(20).iterrows():
        print(f"  {row['term']:25s}  rate_2024={row['rate_a']:6.2f}  rate_2026={row['rate_b']:6.2f}  [{row['category']}]")

print(f"\nAccelerating terms (>0.5/1K both periods, >3x growth): {len(accelerating)}")
if len(accelerating) > 0:
    for _, row in accelerating.head(20).iterrows():
        print(f"  {row['term']:25s}  rate_2024={row['rate_a']:6.2f}  rate_2026={row['rate_b']:6.2f}  x{row['growth_ratio']:.1f}  [{row['category']}]")

# Save primary results
fw_primary.to_csv(TAB_DIR / 'fw_primary_unigrams.csv', index=False)
top_2026.to_csv(TAB_DIR / 'top100_2026_unigrams.csv', index=False)
top_2024.to_csv(TAB_DIR / 'top100_2024_unigrams.csv', index=False)
emerging.to_csv(TAB_DIR / 'emerging_terms.csv', index=False)
disappearing.to_csv(TAB_DIR / 'disappearing_terms.csv', index=False)
accelerating.to_csv(TAB_DIR / 'accelerating_terms.csv', index=False)

# ──────────────────────────────────────────────────────────────────────
# BIGRAM ANALYSIS (step 7)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("BIGRAM ANALYSIS: arshkon (2024-04) vs scraped (2026-03)")
print("="*70)

fw_bigrams = run_comparison(
    arshkon['description_cleaned'].values,
    scraped['description_cleaned'].values,
    'arshkon_2024', 'scraped_2026',
    min_df=5, ngram_range=(2,2)
)

top_bi_2026 = fw_bigrams.head(100)
top_bi_2024 = fw_bigrams.tail(100).sort_values('z_score')

print(f"\nTop 30 bigrams distinguishing 2026:")
for _, row in top_bi_2026.head(30).iterrows():
    print(f"  {row['term']:40s}  z={row['z_score']:7.1f}  rate_2024={row['rate_a']:6.1f}  rate_2026={row['rate_b']:6.1f}  [{row['category']}]")

print(f"\nTop 30 bigrams distinguishing 2024:")
for _, row in top_bi_2024.head(30).iterrows():
    print(f"  {row['term']:40s}  z={row['z_score']:7.1f}  rate_2024={row['rate_a']:6.1f}  rate_2026={row['rate_b']:6.1f}  [{row['category']}]")

# Emerging/disappearing bigrams
bi_emerging, bi_disappearing, bi_accelerating = get_emerging_disappearing_accelerating(fw_bigrams)
print(f"\nEmerging bigrams: {len(bi_emerging)}")
for _, row in bi_emerging.head(15).iterrows():
    print(f"  {row['term']:40s}  rate_2024={row['rate_a']:6.2f}  rate_2026={row['rate_b']:6.2f}  [{row['category']}]")

fw_bigrams.to_csv(TAB_DIR / 'fw_primary_bigrams.csv', index=False)
top_bi_2026.to_csv(TAB_DIR / 'top100_2026_bigrams.csv', index=False)
top_bi_2024.to_csv(TAB_DIR / 'top100_2024_bigrams.csv', index=False)
bi_emerging.to_csv(TAB_DIR / 'emerging_bigrams.csv', index=False)

# ──────────────────────────────────────────────────────────────────────
# SECONDARY COMPARISONS (step 6)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECONDARY COMPARISONS — seniority-stratified")
print("="*70)

secondary_results = {}

# Entry 2024 vs Entry 2026
entry_2024 = arshkon[arshkon['seniority_3level'] == 'junior']
entry_2026 = scraped[scraped['seniority_3level'] == 'junior']
print(f"\nEntry 2024 (arshkon): n={len(entry_2024)}")
print(f"Entry 2026 (scraped): n={len(entry_2026)}")

if len(entry_2024) >= 100 and len(entry_2026) >= 100:
    fw_entry = run_comparison(entry_2024['description_cleaned'].values,
                               entry_2026['description_cleaned'].values,
                               'entry_2024', 'entry_2026', min_df=3)
    secondary_results['entry_evolution'] = fw_entry
    fw_entry.to_csv(TAB_DIR / 'fw_entry_evolution.csv', index=False)
    print("  Top 10 entry 2026 terms:")
    for _, row in fw_entry.head(10).iterrows():
        print(f"    {row['term']:25s}  z={row['z_score']:7.1f}  [{row['category']}]")
else:
    flag = "WARNING: n < 100 for entry comparison"
    print(f"  {flag}")
    secondary_results['entry_evolution'] = flag

# Mid-senior 2024 vs Mid-senior 2026
senior_2024 = arshkon[arshkon['seniority_3level'] == 'senior']
senior_2026 = scraped[scraped['seniority_3level'] == 'senior']
print(f"\nMid-senior 2024 (arshkon): n={len(senior_2024)}")
print(f"Mid-senior 2026 (scraped): n={len(senior_2026)}")

if len(senior_2024) >= 100 and len(senior_2026) >= 100:
    fw_senior = run_comparison(senior_2024['description_cleaned'].values,
                                senior_2026['description_cleaned'].values,
                                'senior_2024', 'senior_2026', min_df=5)
    secondary_results['senior_evolution'] = fw_senior
    fw_senior.to_csv(TAB_DIR / 'fw_senior_evolution.csv', index=False)
    print("  Top 10 senior 2026 terms:")
    for _, row in fw_senior.head(10).iterrows():
        print(f"    {row['term']:25s}  z={row['z_score']:7.1f}  [{row['category']}]")
else:
    flag = "WARNING: n < 100"
    print(f"  {flag}")

# Entry 2026 vs Mid-senior 2024 (relabeling diagnostic)
print(f"\nRelabeling diagnostic: Entry 2026 vs Mid-senior 2024")
print(f"  Entry 2026: n={len(entry_2026)}, Mid-senior 2024: n={len(senior_2024)}")

if len(entry_2026) >= 100 and len(senior_2024) >= 100:
    fw_relabel = run_comparison(senior_2024['description_cleaned'].values,
                                 entry_2026['description_cleaned'].values,
                                 'senior_2024', 'entry_2026', min_df=5)
    secondary_results['relabeling'] = fw_relabel
    fw_relabel.to_csv(TAB_DIR / 'fw_relabeling_diagnostic.csv', index=False)
    print("  Top 10 entry-2026-distinctive terms (vs senior-2024):")
    for _, row in fw_relabel.head(10).iterrows():
        print(f"    {row['term']:25s}  z={row['z_score']:7.1f}  [{row['category']}]")

# Within-2024 calibration: arshkon vs asaniczka mid-senior
senior_arshkon = arshkon[arshkon['seniority_3level'] == 'senior']
senior_asaniczka = asaniczka[asaniczka['seniority_3level'] == 'senior']
print(f"\nWithin-2024 calibration: arshkon senior vs asaniczka senior")
print(f"  arshkon senior: n={len(senior_arshkon)}, asaniczka senior: n={len(senior_asaniczka)}")

if len(senior_arshkon) >= 100 and len(senior_asaniczka) >= 100:
    fw_calib = run_comparison(senior_arshkon['description_cleaned'].values,
                               senior_asaniczka['description_cleaned'].values,
                               'arshkon_senior', 'asaniczka_senior', min_df=5)
    secondary_results['calibration'] = fw_calib
    fw_calib.to_csv(TAB_DIR / 'fw_within2024_calibration.csv', index=False)

    print("  Top 10 instrument-difference terms:")
    top_cal = pd.concat([fw_calib.head(5), fw_calib.tail(5)])
    for _, row in top_cal.iterrows():
        print(f"    {row['term']:25s}  z={row['z_score']:7.1f}  rate_arsh={row['rate_a']:6.1f}  rate_asan={row['rate_b']:6.1f}  [{row['category']}]")

    # Compare calibration z-scores to primary to flag artifacts
    if 'calibration' in secondary_results and isinstance(secondary_results['calibration'], pd.DataFrame):
        calib_df = secondary_results['calibration']
        calib_high = set(calib_df[calib_df['z_score'].abs() > 5]['term'].values)
        primary_high = set(fw_primary[fw_primary['z_score'].abs() > 5]['term'].values)
        overlap = calib_high & primary_high
        print(f"\n  Instrument artifacts: {len(overlap)} terms have |z|>5 in BOTH calibration and primary")
        if overlap:
            print(f"  Flagged: {sorted(list(overlap))[:20]}")

# ──────────────────────────────────────────────────────────────────────
# SENSITIVITY: Aggregator exclusion
# ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SENSITIVITY: Aggregator exclusion")
print("="*70)

arshkon_noagg = arshkon[~arshkon['is_aggregator']]
scraped_noagg = scraped[~scraped['is_aggregator']]
print(f"  arshkon non-agg: {len(arshkon_noagg)} (removed {len(arshkon) - len(arshkon_noagg)})")
print(f"  scraped non-agg: {len(scraped_noagg)} (removed {len(scraped) - len(scraped_noagg)})")

fw_noagg = run_comparison(
    arshkon_noagg['description_cleaned'].values,
    scraped_noagg['description_cleaned'].values,
    'arshkon_noagg', 'scraped_noagg', min_df=5
)

# Compare top terms with primary
primary_top50_2026 = set(fw_primary.head(50)['term'].values)
noagg_top50_2026 = set(fw_noagg.head(50)['term'].values)
overlap = len(primary_top50_2026 & noagg_top50_2026)
print(f"  Top-50 overlap with primary: {overlap}/50 ({overlap/50*100:.0f}%)")
fw_noagg.to_csv(TAB_DIR / 'fw_sensitivity_noagg.csv', index=False)

# ──────────────────────────────────────────────────────────────────────
# SENSITIVITY: Company capping (max 20 postings per company per source)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SENSITIVITY: Company capping (max 20 per company per source)")
print("="*70)

def cap_companies(df_in, max_per=20):
    return df_in.groupby('company_name_canonical').apply(
        lambda g: g.sample(min(len(g), max_per), random_state=42)
    ).reset_index(drop=True)

arshkon_capped = cap_companies(arshkon)
scraped_capped = cap_companies(scraped)
print(f"  arshkon capped: {len(arshkon_capped)} (from {len(arshkon)})")
print(f"  scraped capped: {len(scraped_capped)} (from {len(scraped)})")

fw_capped = run_comparison(
    arshkon_capped['description_cleaned'].values,
    scraped_capped['description_cleaned'].values,
    'arshkon_capped', 'scraped_capped', min_df=5
)

capped_top50_2026 = set(fw_capped.head(50)['term'].values)
overlap_cap = len(primary_top50_2026 & capped_top50_2026)
print(f"  Top-50 overlap with primary: {overlap_cap}/50 ({overlap_cap/50*100:.0f}%)")
fw_capped.to_csv(TAB_DIR / 'fw_sensitivity_capped.csv', index=False)

# ──────────────────────────────────────────────────────────────────────
# SENSITIVITY: Text source (LLM-cleaned only)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SENSITIVITY: LLM-cleaned text only (arshkon only; scraped has no LLM text)")
print("="*70)

arshkon_llm = arshkon[arshkon['text_source'] == 'llm']
print(f"  arshkon LLM-cleaned: {len(arshkon_llm)}")
print(f"  scraped (all rule-based): {len(scraped)}")

if len(arshkon_llm) >= 100:
    fw_llm = run_comparison(
        arshkon_llm['description_cleaned'].values,
        scraped['description_cleaned'].values,
        'arshkon_llm', 'scraped_rule', min_df=5
    )
    llm_top50 = set(fw_llm.head(50)['term'].values)
    overlap_llm = len(primary_top50_2026 & llm_top50)
    print(f"  Top-50 overlap with primary: {overlap_llm}/50 ({overlap_llm/50*100:.0f}%)")
    fw_llm.to_csv(TAB_DIR / 'fw_sensitivity_llm_text.csv', index=False)

# ──────────────────────────────────────────────────────────────────────
# FIGURES
# ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("Generating figures...")
print("="*70)

# Figure 1: Top terms volcano plot
fig, ax = plt.subplots(figsize=(14, 9))

# Use primary results
plot_df = fw_primary.copy()
plot_df['max_rate'] = plot_df[['rate_a', 'rate_b']].max(axis=1)
plot_df = plot_df[plot_df['max_rate'] > 0.5]  # Filter rare terms

colors = {
    'ai_tool': '#e41a1c', 'ai_domain': '#ff7f00', 'tech_stack': '#377eb8',
    'org_scope': '#4daf4a', 'mgmt': '#984ea3', 'sys_design': '#a65628',
    'method': '#f781bf', 'credential': '#999999', 'soft_skill': '#66c2a5',
    'noise': '#cccccc', 'uncategorized': '#dddddd'
}

for cat, color in colors.items():
    mask = plot_df['category'] == cat
    sub = plot_df[mask]
    if len(sub) > 0:
        ax.scatter(sub['z_score'], sub['max_rate'],
                   c=color, label=f'{cat} ({len(sub)})', alpha=0.5, s=20, edgecolors='none')

# Label top terms
for direction_df, ha, offset in [(top_2026.head(15), 'left', 0.3),
                                   (top_2024.head(15), 'right', -0.3)]:
    for _, row in direction_df.iterrows():
        rate = max(row['rate_a'], row['rate_b'])
        if rate > 0.5:
            ax.annotate(row['term'], (row['z_score'], rate), fontsize=7,
                       ha=ha, alpha=0.8,
                       xytext=(offset*30, 2), textcoords='offset points',
                       arrowprops=dict(arrowstyle='-', alpha=0.3, lw=0.5))

ax.axvline(0, color='gray', lw=0.5, ls='--')
ax.set_xlabel('Z-score (positive = more in 2026, negative = more in 2024)')
ax.set_ylabel('Max rate per 1K chars')
ax.set_title(f'Fightin\' Words: arshkon 2024 (n={len(arshkon)}) vs scraped 2026 (n={len(scraped)}) — unigrams')
ax.legend(loc='upper left', fontsize=7, ncol=2, framealpha=0.9)
plt.savefig(FIG_DIR / 'fw_volcano_primary.png')
plt.close()
print("  Saved fw_volcano_primary.png")

# Figure 2: Category summary bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for ax, summary, title, direction in [
    (axes[0], cat_summary_2026, 'More in 2026', 'rising'),
    (axes[1], cat_summary_2024, 'More in 2024', 'declining')]:

    summary_plot = summary.reset_index()
    summary_plot = summary_plot[summary_plot['category'] != 'uncategorized'].head(10)

    bar_colors = [colors.get(c, '#999999') for c in summary_plot['category']]
    ax.barh(summary_plot['category'], summary_plot['n_terms'], color=bar_colors, alpha=0.8)
    ax.set_xlabel('Number of terms in top 100')
    ax.set_title(f'{title}')
    ax.invert_yaxis()

plt.suptitle('Semantic category distribution of top 100 distinguishing terms', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'category_summary.png')
plt.close()
print("  Saved category_summary.png")

# Figure 3: Sensitivity comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

sensitivities = [
    ('Primary', fw_primary, 'fw_primary'),
    ('No aggregators', fw_noagg, 'fw_noagg'),
    ('Company-capped', fw_capped, 'fw_capped'),
]
if len(arshkon_llm) >= 100:
    sensitivities.append(('LLM text only', fw_llm, 'fw_llm'))

for idx, (title, fw, _) in enumerate(sensitivities):
    ax = axes[idx // 2][idx % 2]
    top15 = fw.head(15)
    ax.barh(range(len(top15)), top15['z_score'].values, color='#e41a1c', alpha=0.7)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15['term'].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Z-score')
    ax.set_title(f'{title}')

plt.suptitle('Sensitivity: Top 15 "more in 2026" terms across specifications', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'sensitivity_comparison.png')
plt.close()
print("  Saved sensitivity_comparison.png")

# Figure 4: Bigram top terms
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

for ax, df_plot, title, color in [
    (axes[0], top_bi_2026.head(25), 'Bigrams more in 2026', '#e41a1c'),
    (axes[1], top_bi_2024.head(25), 'Bigrams more in 2024', '#377eb8')]:

    ax.barh(range(len(df_plot)), df_plot['z_score'].abs().values, color=color, alpha=0.7)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot['term'].values, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('|Z-score|')
    ax.set_title(title)

plt.suptitle(f'Bigram Fightin\' Words: arshkon 2024 vs scraped 2026', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'bigram_top_terms.png')
plt.close()
print("  Saved bigram_top_terms.png")

# ──────────────────────────────────────────────────────────────────────
# CATEGORY-LEVEL AGGREGATION TABLE
# ──────────────────────────────────────────────────────────────────────
print("\nBuilding category-level summary table...")

# For ALL terms with |z| > 3
sig_terms = fw_primary[fw_primary['z_score'].abs() > 3].copy()
sig_terms['direction'] = np.where(sig_terms['z_score'] > 0, 'rising_2026', 'declining_2026')

cat_pivot = sig_terms.groupby(['category', 'direction']).agg(
    n_terms=('term', 'count'),
    mean_abs_z=('z_score', lambda x: x.abs().mean()),
    top_terms=('term', lambda x: ', '.join(x.head(10)))
).reset_index()

cat_pivot.to_csv(TAB_DIR / 'category_level_summary.csv', index=False)
print(cat_pivot.to_string())

print("\n" + "="*70)
print("T12 COMPLETE — Tables and figures saved.")
print("="*70)
