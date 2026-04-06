#!/usr/bin/env python3
"""T22: Ghost & aspirational requirements forensics.

Stress-tests the paper's headline findings (entry-level management indicators
jumped from 9.4% to 40.8%) by checking for ghost-like and aspirational
requirement patterns.
"""

import duckdb
import pandas as pd
import numpy as np
import re
import json
import os
from collections import Counter
from pathlib import Path

# Output paths
BASE = Path("/home/jihgaboot/gabor/job-research")
FIG_DIR = BASE / "exploration/figures/T22"
TBL_DIR = BASE / "exploration/tables/T22"
RPT_DIR = BASE / "exploration/reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)
RPT_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.1)

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")

DATA = str(BASE / "data/unified.parquet")
BASE_FILTER = "source_platform='linkedin' AND is_english=true AND date_flag='ok'"
SWE_FILTER = f"{BASE_FILTER} AND is_swe=true"

# =============================================================================
# PART 1: Load SWE data with needed columns
# =============================================================================
print("Loading SWE data...")
df = con.execute(f"""
SELECT
  uid, title, title_normalized, period, source,
  seniority_3level, seniority_final,
  is_aggregator,
  company_name_effective, company_name_canonical,
  company_industry,
  yoe_extracted, yoe_seniority_contradiction,
  ghost_job_risk,
  description_core, description_core_llm, description,
  llm_extraction_coverage, core_length
FROM '{DATA}'
WHERE {SWE_FILTER}
""").fetchdf()
print(f"  Loaded {len(df)} SWE rows")

# Choose best text: prefer description_core_llm, fallback to description_core, then description
df['text'] = df['description_core_llm'].where(
    (df['llm_extraction_coverage'] == 'labeled') &
    (df['description_core_llm'].notna()) &
    (df['description_core_llm'].str.len() > 50),
    df['description_core'].where(
        df['description_core'].notna() & (df['description_core'].str.len() > 50),
        df['description']
    )
)

# =============================================================================
# PART 2: Compute ghost indicators
# =============================================================================
print("Computing ghost indicators...")

# --- 2a. Tech count ---
TECH_PATTERNS = [
    r'\bpython\b', r'\bjava\b', r'\bjavascript\b', r'\btypescript\b',
    r'\bc\+\+\b', r'\bc#\b', r'\bruby\b', r'\bgo\b(?:lang)?', r'\brust\b',
    r'\bswift\b', r'\bkotlin\b', r'\bscala\b', r'\bphp\b', r'\bperl\b',
    r'\breact\b', r'\bangular\b', r'\bvue\b', r'\bnode\.?js\b', r'\bnext\.?js\b',
    r'\bdjango\b', r'\bflask\b', r'\bspring\b', r'\b\.net\b', r'\brails\b',
    r'\baws\b', r'\bazure\b', r'\bgcp\b', r'\bkubernetes\b', r'\bdocker\b',
    r'\bterraform\b', r'\bjenkins\b', r'\bci/cd\b', r'\bgit\b',
    r'\bsql\b', r'\bpostgres\b', r'\bmongo\b', r'\bredis\b', r'\belasticsearch\b',
    r'\bkafka\b', r'\bspark\b', r'\bhadoop\b', r'\bairflow\b',
    r'\bgraphql\b', r'\brest\b', r'\bgrpc\b', r'\bmicroservices?\b',
    r'\bhtml\b', r'\bcss\b', r'\bsass\b',
    r'\btensorflow\b', r'\bpytorch\b', r'\bscikit\b', r'\bpandas\b', r'\bnumpy\b',
    r'\blinux\b', r'\bagile\b', r'\bscrum\b', r'\bjira\b',
]

def count_techs(text):
    if not isinstance(text, str) or len(text) < 10:
        return 0
    text_lower = text.lower()
    return sum(1 for pat in TECH_PATTERNS if re.search(pat, text_lower))

# --- 2b. Org scope terms ---
ORG_SCOPE_TERMS = [
    r'\bcross[- ]?functional\b', r'\bstakeholder\b', r'\bbudget\b',
    r'\bstrategic\b', r'\broadmap\b', r'\bkpi\b', r'\bokr\b',
    r'\bpeople management\b', r'\bheadcount\b', r'\bp&l\b',
    r'\bexecutive\b', r'\bboard\b', r'\bc-suite\b', r'\bleadership\b',
    r'\bmanage\s+a\s+team\b', r'\bmanaging\s+a\s+team\b',
    r'\bdirect\s+reports?\b', r'\bteam\s+of\s+\d+\b',
    r'\bmentoring\b', r'\bmentor\b', r'\bcoaching\b',
]

def count_org_scope(text):
    if not isinstance(text, str) or len(text) < 10:
        return 0
    text_lower = text.lower()
    return sum(1 for pat in ORG_SCOPE_TERMS if re.search(pat, text_lower))

# --- 2c. Aspiration ratio ---
HEDGE_PATTERNS = [
    r'\bideally\b', r'\bnice\s+to\s+have\b', r'\bpreferred\b',
    r'\bbonus\b', r'\ba\s+plus\b', r'\bdesirable\b', r'\badvantage\b',
    r'\bnot\s+required\b', r'\bhelpful\b', r'\bprefer(?:ably|red)?\b',
    r'\boptional\b', r'\bwould\s+be\s+great\b',
]
FIRM_PATTERNS = [
    r'\bmust\s+have\b', r'\brequired\b', r'\bminimum\b', r'\bmandatory\b',
    r'\bessential\b', r'\bnecessary\b', r'\byou\s+will\b', r'\byou\s+must\b',
    r'\brequire[sd]?\b', r'\bmust\b',
]

def aspiration_ratio(text):
    if not isinstance(text, str) or len(text) < 10:
        return np.nan
    text_lower = text.lower()
    hedge = sum(1 for pat in HEDGE_PATTERNS if re.search(pat, text_lower))
    firm = sum(1 for pat in FIRM_PATTERNS if re.search(pat, text_lower))
    if firm == 0:
        return hedge if hedge > 0 else np.nan  # pure hedge or no signal
    return hedge / firm

# --- 2d. Management indicator patterns (same as pipeline) ---
MGMT_PATTERNS = [
    r'\bleadership\b', r'\bleading\b', r'\blead\s+(?:a\s+)?team\b',
    r'\bpeople\s+management\b', r'\bmanage\s+(?:a\s+)?team\b',
    r'\bmanaging\s+(?:a\s+)?team\b', r'\bdirect\s+reports?\b',
    r'\bteam\s+of\s+\d+\b', r'\bmentor(?:ing)?\b', r'\bcoach(?:ing)?\b',
    r'\bcross[- ]?functional\b', r'\bstakeholder\b',
    r'\bproject\s+management\b', r'\bprogram\s+management\b',
    r'\bbudget\b', r'\bstrategic\b', r'\broadmap\b',
]

def has_mgmt_indicator(text):
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in MGMT_PATTERNS)

# --- 2e. AI requirement patterns ---
AI_TOOL_PATTERNS = [
    r'\bcopilot\b', r'\bcursor\b', r'\bclaude\b', r'\bchatgpt\b',
    r'\bgpt[-\s]?\d\b', r'\bllm\b', r'\blarge\s+language\s+model\b',
    r'\bprompt\s+engineering\b', r'\bai\s+coding\b', r'\bai\s+assist\b',
    r'\bcode\s+generation\b', r'\bgithub\s+copilot\b',
    r'\bai\s+tool\b', r'\bai[-\s]?powered\b',
]
AI_DOMAIN_PATTERNS = [
    r'\bmachine\s+learning\b', r'\bdeep\s+learning\b', r'\bnlp\b',
    r'\bnatural\s+language\s+processing\b', r'\bcomputer\s+vision\b',
    r'\bmodel\s+training\b', r'\bneural\s+network\b', r'\btransformer\b',
    r'\bbert\b', r'\breinforcement\s+learning\b', r'\bml\s+engineer\b',
    r'\bml\s+pipeline\b', r'\bfeature\s+engineering\b',
]
AI_GENERAL_PATTERNS = [
    r'\bartificial\s+intelligence\b', r'\b(?:ai|a\.i\.)\b',
    r'\bmachine\s+learning\b',  # overlap with domain, counted separately
]

def ai_category(text):
    """Returns (has_ai_tool, has_ai_domain, has_ai_general, has_any_ai)"""
    if not isinstance(text, str):
        return (False, False, False, False)
    text_lower = text.lower()
    tool = any(re.search(pat, text_lower) for pat in AI_TOOL_PATTERNS)
    domain = any(re.search(pat, text_lower) for pat in AI_DOMAIN_PATTERNS)
    general = any(re.search(pat, text_lower) for pat in AI_GENERAL_PATTERNS)
    return (tool, domain, general, tool or domain or general)

# Apply indicators
print("  Computing tech counts...")
df['tech_count'] = df['text'].apply(count_techs)
print("  Computing org scope...")
df['org_scope_count'] = df['text'].apply(count_org_scope)
print("  Computing kitchen-sink score...")
df['kitchen_sink'] = df['tech_count'] * df['org_scope_count']
print("  Computing aspiration ratio...")
df['aspiration_ratio'] = df['text'].apply(aspiration_ratio)
print("  Computing management indicators...")
df['has_mgmt'] = df['text'].apply(has_mgmt_indicator)
print("  Computing AI categories...")
ai_cats = df['text'].apply(ai_category)
df['has_ai_tool'] = ai_cats.apply(lambda x: x[0])
df['has_ai_domain'] = ai_cats.apply(lambda x: x[1])
df['has_ai_general'] = ai_cats.apply(lambda x: x[2])
df['has_any_ai'] = ai_cats.apply(lambda x: x[3])

# --- 2f. YOE-scope mismatch ---
df['yoe_scope_mismatch'] = (
    (df['seniority_3level'] == 'junior') &
    ((df['yoe_extracted'] >= 5) | (df['org_scope_count'] >= 3))
)

# =============================================================================
# PART 3: Ghost prevalence by period x seniority
# =============================================================================
print("Computing prevalence tables...")

def prevalence_table(df, metric, label):
    """Compute mean of metric by period x seniority."""
    grouped = df.groupby(['period', 'seniority_3level']).agg(
        n=('uid', 'count'),
        mean_val=(metric, 'mean'),
    ).reset_index()
    grouped.columns = ['period', 'seniority', 'n', f'mean_{label}']
    return grouped

# Prevalence tables
tbl_kitchen = prevalence_table(df, 'kitchen_sink', 'kitchen_sink')
tbl_aspiration = df.dropna(subset=['aspiration_ratio']).pipe(
    lambda d: prevalence_table(d, 'aspiration_ratio', 'aspiration_ratio')
)
tbl_mgmt = prevalence_table(df, 'has_mgmt', 'mgmt_rate')
tbl_mismatch = df[df['seniority_3level'] == 'junior'].groupby('period').agg(
    n=('uid', 'count'),
    mismatch_rate=('yoe_scope_mismatch', 'mean'),
).reset_index()
tbl_ai = prevalence_table(df, 'has_any_ai', 'ai_rate')

# Save tables
tbl_kitchen.to_csv(TBL_DIR / "ghost_kitchen_sink_by_period_seniority.csv", index=False)
tbl_aspiration.to_csv(TBL_DIR / "ghost_aspiration_ratio_by_period_seniority.csv", index=False)
tbl_mgmt.to_csv(TBL_DIR / "ghost_mgmt_rate_by_period_seniority.csv", index=False)
tbl_mismatch.to_csv(TBL_DIR / "ghost_yoe_mismatch_junior.csv", index=False)
tbl_ai.to_csv(TBL_DIR / "ghost_ai_rate_by_period_seniority.csv", index=False)

print("  Management indicator rates:")
print(tbl_mgmt[tbl_mgmt['seniority'].isin(['junior', 'senior'])].to_string())

# =============================================================================
# PART 4: AI ghostiness test (CRITICAL)
# =============================================================================
print("\nAI ghostiness test...")

def ai_aspiration_analysis(text):
    """Check if AI terms appear in hedged vs firm contexts."""
    if not isinstance(text, str) or len(text) < 50:
        return {}
    text_lower = text.lower()

    # Find sentences/phrases with AI terms
    sentences = re.split(r'[.;!\n]', text_lower)

    ai_in_hedge = 0
    ai_in_firm = 0
    nonai_in_hedge = 0
    nonai_in_firm = 0

    for sent in sentences:
        has_ai = any(re.search(pat, sent) for pat in AI_TOOL_PATTERNS + AI_DOMAIN_PATTERNS + AI_GENERAL_PATTERNS)
        has_hedge = any(re.search(pat, sent) for pat in HEDGE_PATTERNS)
        has_firm = any(re.search(pat, sent) for pat in FIRM_PATTERNS)

        if has_ai:
            if has_hedge:
                ai_in_hedge += 1
            if has_firm:
                ai_in_firm += 1
        else:
            if has_hedge:
                nonai_in_hedge += 1
            if has_firm:
                nonai_in_firm += 1

    return {
        'ai_hedge': ai_in_hedge, 'ai_firm': ai_in_firm,
        'nonai_hedge': nonai_in_hedge, 'nonai_firm': nonai_in_firm,
    }

# Run on AI-containing postings
ai_postings = df[df['has_any_ai']].copy()
print(f"  Analyzing {len(ai_postings)} AI-containing postings...")

ai_asp_results = ai_postings['text'].apply(ai_aspiration_analysis)
ai_postings['ai_hedge'] = ai_asp_results.apply(lambda x: x.get('ai_hedge', 0))
ai_postings['ai_firm'] = ai_asp_results.apply(lambda x: x.get('ai_firm', 0))
ai_postings['nonai_hedge'] = ai_asp_results.apply(lambda x: x.get('nonai_hedge', 0))
ai_postings['nonai_firm'] = ai_asp_results.apply(lambda x: x.get('nonai_firm', 0))

# Summary
ai_totals = ai_postings[['ai_hedge', 'ai_firm', 'nonai_hedge', 'nonai_firm']].sum()
print(f"  AI sentences in hedged context: {ai_totals['ai_hedge']}")
print(f"  AI sentences in firm context: {ai_totals['ai_firm']}")
print(f"  Non-AI sentences in hedged context: {ai_totals['nonai_hedge']}")
print(f"  Non-AI sentences in firm context: {ai_totals['nonai_firm']}")

ai_hedge_rate = ai_totals['ai_hedge'] / max(ai_totals['ai_hedge'] + ai_totals['ai_firm'], 1)
nonai_hedge_rate = ai_totals['nonai_hedge'] / max(ai_totals['nonai_hedge'] + ai_totals['nonai_firm'], 1)
print(f"  AI hedge fraction: {ai_hedge_rate:.3f}")
print(f"  Non-AI hedge fraction: {nonai_hedge_rate:.3f}")

# By period
ai_asp_by_period = ai_postings.groupby('period').agg(
    ai_hedge=('ai_hedge', 'sum'),
    ai_firm=('ai_firm', 'sum'),
    nonai_hedge=('nonai_hedge', 'sum'),
    nonai_firm=('nonai_firm', 'sum'),
    n=('uid', 'count'),
).reset_index()
ai_asp_by_period['ai_hedge_frac'] = ai_asp_by_period['ai_hedge'] / (ai_asp_by_period['ai_hedge'] + ai_asp_by_period['ai_firm']).clip(lower=1)
ai_asp_by_period['nonai_hedge_frac'] = ai_asp_by_period['nonai_hedge'] / (ai_asp_by_period['nonai_hedge'] + ai_asp_by_period['nonai_firm']).clip(lower=1)
ai_asp_by_period.to_csv(TBL_DIR / "ai_aspiration_by_period.csv", index=False)
print(ai_asp_by_period.to_string())

# =============================================================================
# PART 5: Management indicator validity test (CRITICAL)
# =============================================================================
print("\nManagement indicator validity test...")

# 5a: Where do management terms appear? In requirements or boilerplate?
# We check: in description_core (boilerplate removed) vs description (full)
entry_2026 = df[(df['seniority_3level'] == 'junior') & (df['period'] == '2026-03')].copy()
entry_2026_mgmt = entry_2026[entry_2026['has_mgmt']].copy()
print(f"  2026 entry-level SWE postings: {len(entry_2026)}")
print(f"  2026 entry-level with mgmt indicators: {len(entry_2026_mgmt)} ({len(entry_2026_mgmt)/len(entry_2026)*100:.1f}%)")

# Check if mgmt terms appear in core text vs full text
def mgmt_in_core_vs_full(row):
    """Check where management terms appear."""
    core = row.get('description_core', '') or ''
    full = row.get('description', '') or ''

    core_lower = core.lower()
    full_lower = full.lower()

    in_core = any(re.search(pat, core_lower) for pat in MGMT_PATTERNS)
    in_full = any(re.search(pat, full_lower) for pat in MGMT_PATTERNS)

    return pd.Series({'mgmt_in_core': in_core, 'mgmt_in_full': in_full})

mgmt_location = entry_2026_mgmt.apply(mgmt_in_core_vs_full, axis=1)
entry_2026_mgmt = pd.concat([entry_2026_mgmt, mgmt_location], axis=1)

both = (entry_2026_mgmt['mgmt_in_core'] & entry_2026_mgmt['mgmt_in_full']).sum()
core_only = (entry_2026_mgmt['mgmt_in_core'] & ~entry_2026_mgmt['mgmt_in_full']).sum()
full_only = (~entry_2026_mgmt['mgmt_in_core'] & entry_2026_mgmt['mgmt_in_full']).sum()
neither = (~entry_2026_mgmt['mgmt_in_core'] & ~entry_2026_mgmt['mgmt_in_full']).sum()

print(f"  Management terms in core AND full: {both}")
print(f"  Management terms in core only: {core_only}")
print(f"  Management terms in full only (boilerplate): {full_only}")
print(f"  Management terms in neither (text column issue): {neither}")

# 5b: Hedging analysis for management terms in entry-level
def mgmt_hedging_analysis(text):
    """Check if management terms are hedged or firm."""
    if not isinstance(text, str) or len(text) < 30:
        return {'mgmt_hedged': 0, 'mgmt_firm': 0, 'mgmt_neutral': 0, 'mgmt_terms': []}

    text_lower = text.lower()
    sentences = re.split(r'[.;!\n]', text_lower)

    mgmt_hedged = 0
    mgmt_firm = 0
    mgmt_neutral = 0
    terms_found = []

    for sent in sentences:
        mgmt_matches = [pat for pat in MGMT_PATTERNS if re.search(pat, sent)]
        if not mgmt_matches:
            continue

        has_hedge = any(re.search(pat, sent) for pat in HEDGE_PATTERNS)
        has_firm = any(re.search(pat, sent) for pat in FIRM_PATTERNS)

        for m in mgmt_matches:
            match = re.search(m, sent)
            if match:
                terms_found.append(match.group())

        if has_hedge and not has_firm:
            mgmt_hedged += 1
        elif has_firm:
            mgmt_firm += 1
        else:
            mgmt_neutral += 1

    return {
        'mgmt_hedged': mgmt_hedged,
        'mgmt_firm': mgmt_firm,
        'mgmt_neutral': mgmt_neutral,
        'mgmt_terms': terms_found,
    }

mgmt_hedge = entry_2026_mgmt['text'].apply(mgmt_hedging_analysis)
entry_2026_mgmt['mgmt_hedged'] = mgmt_hedge.apply(lambda x: x['mgmt_hedged'])
entry_2026_mgmt['mgmt_firm'] = mgmt_hedge.apply(lambda x: x['mgmt_firm'])
entry_2026_mgmt['mgmt_neutral'] = mgmt_hedge.apply(lambda x: x['mgmt_neutral'])
entry_2026_mgmt['mgmt_terms'] = mgmt_hedge.apply(lambda x: x['mgmt_terms'])

total_hedged = entry_2026_mgmt['mgmt_hedged'].sum()
total_firm = entry_2026_mgmt['mgmt_firm'].sum()
total_neutral = entry_2026_mgmt['mgmt_neutral'].sum()
total_mgmt_sents = total_hedged + total_firm + total_neutral
print(f"\n  Management term context in 2026 entry-level:")
print(f"    Hedged: {total_hedged} ({total_hedged/total_mgmt_sents*100:.1f}%)")
print(f"    Firm/required: {total_firm} ({total_firm/total_mgmt_sents*100:.1f}%)")
print(f"    Neutral (no hedge/firm marker): {total_neutral} ({total_neutral/total_mgmt_sents*100:.1f}%)")

# 5c: Most common management terms at entry level
all_mgmt_terms = []
for terms in entry_2026_mgmt['mgmt_terms']:
    all_mgmt_terms.extend(terms)
term_counts = Counter(all_mgmt_terms)
print(f"\n  Top management terms in 2026 entry-level postings:")
for term, count in term_counts.most_common(20):
    print(f"    '{term}': {count}")

# Same analysis for 2024 entry-level
entry_2024 = df[(df['seniority_3level'] == 'junior') & (df['period'].isin(['2024-01', '2024-04']))].copy()
entry_2024_mgmt = entry_2024[entry_2024['has_mgmt']].copy()
print(f"\n  2024 entry-level SWE postings: {len(entry_2024)}")
print(f"  2024 entry-level with mgmt indicators: {len(entry_2024_mgmt)} ({len(entry_2024_mgmt)/len(entry_2024)*100:.1f}% if entry>0)")

if len(entry_2024_mgmt) > 0:
    mgmt_hedge_2024 = entry_2024_mgmt['text'].apply(mgmt_hedging_analysis)
    entry_2024_mgmt['mgmt_hedged'] = mgmt_hedge_2024.apply(lambda x: x['mgmt_hedged'])
    entry_2024_mgmt['mgmt_firm'] = mgmt_hedge_2024.apply(lambda x: x['mgmt_firm'])
    entry_2024_mgmt['mgmt_neutral'] = mgmt_hedge_2024.apply(lambda x: x['mgmt_neutral'])
    entry_2024_mgmt['mgmt_terms'] = mgmt_hedge_2024.apply(lambda x: x['mgmt_terms'])

    t_hedged_24 = entry_2024_mgmt['mgmt_hedged'].sum()
    t_firm_24 = entry_2024_mgmt['mgmt_firm'].sum()
    t_neutral_24 = entry_2024_mgmt['mgmt_neutral'].sum()
    t_total_24 = t_hedged_24 + t_firm_24 + t_neutral_24
    if t_total_24 > 0:
        print(f"  Management term context in 2024 entry-level:")
        print(f"    Hedged: {t_hedged_24} ({t_hedged_24/t_total_24*100:.1f}%)")
        print(f"    Firm/required: {t_firm_24} ({t_firm_24/t_total_24*100:.1f}%)")
        print(f"    Neutral: {t_neutral_24} ({t_neutral_24/t_total_24*100:.1f}%)")

# =============================================================================
# PART 5d: Sample 20 entry-level postings with management terms
# =============================================================================
print("\nSampling 20 entry-level postings with management terms...")

sample = entry_2026_mgmt.sample(min(20, len(entry_2026_mgmt)), random_state=42)
samples_out = []
for _, row in sample.iterrows():
    text = str(row['text'])[:1500]
    samples_out.append({
        'uid': row['uid'],
        'title': row['title'],
        'company': row['company_name_effective'],
        'seniority': row['seniority_final'],
        'yoe': row['yoe_extracted'],
        'mgmt_terms': row['mgmt_terms'],
        'is_aggregator': row['is_aggregator'],
        'text_snippet': text,
    })

# Save sample
pd.DataFrame(samples_out).to_csv(TBL_DIR / "mgmt_entry_level_sample_20.csv", index=False)

# =============================================================================
# PART 6: Top 20 most ghost-like entry-level postings
# =============================================================================
print("Finding top 20 ghost-like entry-level postings...")

entry_all = df[df['seniority_3level'] == 'junior'].copy()
# Composite ghost score: kitchen_sink (normalized) + aspiration_ratio (normalized) + yoe_mismatch
ks_std = max(entry_all['kitchen_sink'].std(), 0.01)
entry_all['ks_norm'] = (entry_all['kitchen_sink'] - entry_all['kitchen_sink'].mean()) / ks_std
ar_filled = entry_all['aspiration_ratio'].fillna(0)
ar_std = max(ar_filled.std(), 0.01)
entry_all['ar_norm'] = (ar_filled - ar_filled.mean()) / ar_std
entry_all['ghost_composite'] = entry_all['ks_norm'] + entry_all['ar_norm'] + entry_all['yoe_scope_mismatch'].astype(float) * 2

top_ghost = entry_all.nlargest(20, 'ghost_composite')
ghost_out = []
for _, row in top_ghost.iterrows():
    text = str(row['text'])[:1500]
    ghost_out.append({
        'uid': row['uid'],
        'title': row['title'],
        'company': row['company_name_effective'],
        'period': row['period'],
        'seniority': row['seniority_final'],
        'yoe': row['yoe_extracted'],
        'tech_count': row['tech_count'],
        'org_scope_count': row['org_scope_count'],
        'kitchen_sink': row['kitchen_sink'],
        'aspiration_ratio': row['aspiration_ratio'],
        'ghost_composite': row['ghost_composite'],
        'is_aggregator': row['is_aggregator'],
        'text_snippet': text,
    })

pd.DataFrame(ghost_out).to_csv(TBL_DIR / "top20_ghost_entry_level.csv", index=False)

# =============================================================================
# PART 7: Aggregator vs direct comparison
# =============================================================================
print("Aggregator vs direct comparison...")

agg_comparison = df.groupby(['is_aggregator', 'period', 'seniority_3level']).agg(
    n=('uid', 'count'),
    mean_kitchen_sink=('kitchen_sink', 'mean'),
    mean_tech_count=('tech_count', 'mean'),
    mean_org_scope=('org_scope_count', 'mean'),
    mean_aspiration=('aspiration_ratio', 'mean'),
    mgmt_rate=('has_mgmt', 'mean'),
    ai_rate=('has_any_ai', 'mean'),
).reset_index()
agg_comparison.to_csv(TBL_DIR / "aggregator_vs_direct_ghost.csv", index=False)

# Focused: junior only
agg_junior = agg_comparison[agg_comparison['seniority_3level'] == 'junior']
print("\n  Junior-level aggregator vs direct:")
print(agg_junior.to_string())

# =============================================================================
# PART 8: Industry patterns
# =============================================================================
print("\nIndustry patterns...")

# Only scraped + arshkon have company_industry
industry_df = df[df['company_industry'].notna() & (df['company_industry'] != '')].copy()
print(f"  Rows with industry info: {len(industry_df)}")

if len(industry_df) > 100:
    # Top industries by ghost indicators
    industry_ghost = industry_df.groupby('company_industry').agg(
        n=('uid', 'count'),
        mean_kitchen_sink=('kitchen_sink', 'mean'),
        mean_aspiration=('aspiration_ratio', 'mean'),
        mgmt_rate=('has_mgmt', 'mean'),
        ai_rate=('has_any_ai', 'mean'),
    ).reset_index()
    industry_ghost = industry_ghost[industry_ghost['n'] >= 20].sort_values('mean_kitchen_sink', ascending=False)
    industry_ghost.to_csv(TBL_DIR / "industry_ghost_indicators.csv", index=False)
    print(f"  Industries with >=20 SWE postings: {len(industry_ghost)}")
    print(industry_ghost.head(15).to_string())

# =============================================================================
# PART 9: Template saturation (company copy-paste detection)
# =============================================================================
print("\nTemplate saturation analysis...")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Companies with >= 5 SWE postings in any period
company_counts = df.groupby(['company_name_canonical', 'period']).size().reset_index(name='n')
prolific = company_counts[company_counts['n'] >= 5][['company_name_canonical', 'period']].copy()
print(f"  Company-period pairs with >=5 postings: {len(prolific)}")

# For each, compute mean pairwise cosine similarity
template_results = []
for _, row in prolific.iterrows():
    co = row['company_name_canonical']
    per = row['period']
    subset = df[(df['company_name_canonical'] == co) & (df['period'] == per)]
    texts = subset['text'].dropna().tolist()
    if len(texts) < 2:
        continue

    try:
        # Limit texts for memory
        texts = texts[:50]
        tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
        mat = tfidf.fit_transform(texts)
        sim = cosine_similarity(mat)
        # Mean pairwise (exclude diagonal)
        n = sim.shape[0]
        if n < 2:
            continue
        mask = ~np.eye(n, dtype=bool)
        mean_sim = sim[mask].mean()
        template_results.append({
            'company': co,
            'period': per,
            'n_postings': len(texts),
            'mean_pairwise_sim': mean_sim,
        })
    except Exception:
        continue

template_df = pd.DataFrame(template_results)
if len(template_df) > 0:
    template_df = template_df.sort_values('mean_pairwise_sim', ascending=False)
    template_df.to_csv(TBL_DIR / "template_saturation.csv", index=False)

    # Flag companies with mean sim > 0.8
    high_sim = template_df[template_df['mean_pairwise_sim'] > 0.8]
    print(f"  Companies with mean sim > 0.8: {len(high_sim)}")
    print(f"  Companies with mean sim > 0.6: {len(template_df[template_df['mean_pairwise_sim'] > 0.6])}")
    print(f"  Overall distribution:")
    print(f"    Mean: {template_df['mean_pairwise_sim'].mean():.3f}")
    print(f"    Median: {template_df['mean_pairwise_sim'].median():.3f}")
    print(f"    P75: {template_df['mean_pairwise_sim'].quantile(0.75):.3f}")
    print(f"    P90: {template_df['mean_pairwise_sim'].quantile(0.90):.3f}")

# =============================================================================
# PART 10: FIGURES
# =============================================================================
print("\nGenerating figures...")

# --- Figure 1: Ghost indicators by period for junior postings ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1a: Kitchen-sink score
junior_ks = df[df['seniority_3level'] == 'junior'].groupby('period')['kitchen_sink'].mean()
senior_ks = df[df['seniority_3level'] == 'senior'].groupby('period')['kitchen_sink'].mean()
ax = axes[0, 0]
x = range(len(junior_ks))
labels = junior_ks.index.tolist()
ax.bar([i-0.15 for i in x], junior_ks.values, 0.3, label='Junior', color='#e74c3c')
ax.bar([i+0.15 for i in x], senior_ks.values, 0.3, label='Senior', color='#3498db')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Mean kitchen-sink score')
ax.set_title('Kitchen-Sink Score by Period')
ax.legend()

# 1b: Aspiration ratio
jr_asp = df[df['seniority_3level'] == 'junior'].groupby('period')['aspiration_ratio'].mean()
sr_asp = df[df['seniority_3level'] == 'senior'].groupby('period')['aspiration_ratio'].mean()
ax = axes[0, 1]
ax.bar([i-0.15 for i in x], jr_asp.values, 0.3, label='Junior', color='#e74c3c')
ax.bar([i+0.15 for i in x], sr_asp.values, 0.3, label='Senior', color='#3498db')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Mean aspiration ratio')
ax.set_title('Aspiration Ratio by Period')
ax.legend()

# 1c: Management indicator rate
jr_mgmt = df[df['seniority_3level'] == 'junior'].groupby('period')['has_mgmt'].mean()
sr_mgmt = df[df['seniority_3level'] == 'senior'].groupby('period')['has_mgmt'].mean()
ax = axes[1, 0]
ax.bar([i-0.15 for i in x], jr_mgmt.values * 100, 0.3, label='Junior', color='#e74c3c')
ax.bar([i+0.15 for i in x], sr_mgmt.values * 100, 0.3, label='Senior', color='#3498db')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Management indicator rate (%)')
ax.set_title('Management Indicators by Period')
ax.legend()

# 1d: AI rate
jr_ai = df[df['seniority_3level'] == 'junior'].groupby('period')['has_any_ai'].mean()
sr_ai = df[df['seniority_3level'] == 'senior'].groupby('period')['has_any_ai'].mean()
ax = axes[1, 1]
ax.bar([i-0.15 for i in x], jr_ai.values * 100, 0.3, label='Junior', color='#e74c3c')
ax.bar([i+0.15 for i in x], sr_ai.values * 100, 0.3, label='Senior', color='#3498db')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('AI requirement rate (%)')
ax.set_title('AI Requirements by Period')
ax.legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "ghost_indicators_overview.png", dpi=150)
plt.close()

# --- Figure 2: Aggregator vs Direct for junior ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (metric, label) in enumerate([
    ('mean_kitchen_sink', 'Kitchen-Sink Score'),
    ('mgmt_rate', 'Management Indicator Rate'),
    ('ai_rate', 'AI Requirement Rate'),
]):
    ax = axes[idx]
    agg_j = agg_comparison[(agg_comparison['seniority_3level'] == 'junior')]
    for agg_val, color, agg_label in [(True, '#e67e22', 'Aggregator'), (False, '#2ecc71', 'Direct')]:
        sub = agg_j[agg_j['is_aggregator'] == agg_val].sort_values('period')
        if len(sub) > 0:
            ax.plot(sub['period'], sub[metric], 'o-', color=color, label=agg_label, markersize=8)
    ax.set_title(f'Junior: {label}')
    ax.set_ylabel(label)
    ax.legend()
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(FIG_DIR / "aggregator_vs_direct_junior.png", dpi=150)
plt.close()

# --- Figure 3: AI aspiration analysis ---
fig, ax = plt.subplots(figsize=(8, 5))
if len(ai_asp_by_period) > 0:
    periods = ai_asp_by_period['period'].tolist()
    x = range(len(periods))
    ax.bar([i-0.15 for i in x], ai_asp_by_period['ai_hedge_frac'].values * 100, 0.3,
           label='AI terms hedge fraction', color='#9b59b6')
    ax.bar([i+0.15 for i in x], ai_asp_by_period['nonai_hedge_frac'].values * 100, 0.3,
           label='Non-AI terms hedge fraction', color='#34495e')
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.set_ylabel('Hedge fraction (%)')
    ax.set_title('AI vs Non-AI Terms: Fraction in Hedged Context')
    ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "ai_aspiration_comparison.png", dpi=150)
plt.close()

# --- Figure 4: Management term breakdown for entry-level ---
fig, ax = plt.subplots(figsize=(10, 6))
if len(term_counts) > 0:
    top_terms = term_counts.most_common(15)
    terms_list = [t[0] for t in top_terms]
    counts_list = [t[1] for t in top_terms]
    y_pos = range(len(terms_list))
    ax.barh(y_pos, counts_list, color='#e74c3c')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms_list)
    ax.invert_yaxis()
    ax.set_xlabel('Count')
    ax.set_title('Top Management Terms in 2026 Entry-Level SWE Postings')
plt.tight_layout()
plt.savefig(FIG_DIR / "mgmt_terms_entry_level.png", dpi=150)
plt.close()

print("\nAll figures saved.")

# =============================================================================
# PART 11: Deeper management validity — section-level analysis
# =============================================================================
print("\nSection-level management analysis...")

# Attempt to identify sections in the text and check where management terms fall
def section_analysis(text):
    """Try to identify which section management terms appear in."""
    if not isinstance(text, str) or len(text) < 100:
        return {'in_requirements': False, 'in_responsibilities': False, 'in_qualifications': False,
                'in_about': False, 'in_benefits': False, 'section_unknown': True}

    text_lower = text.lower()

    # Common section headers
    req_headers = [r'requirements?', r'qualifications?', r'what\s+you\s+need', r'what\s+we\s+(?:re\s+)?looking\s+for',
                   r'must\s+have', r'minimum\s+qualifications?', r'you\s+have', r'your\s+skills']
    resp_headers = [r'responsibilities?', r'what\s+you.?ll\s+do', r'the\s+role', r'about\s+the\s+role',
                    r'key\s+duties', r'you\s+will', r'in\s+this\s+role']
    about_headers = [r'about\s+(?:us|the\s+company|our)', r'who\s+we\s+are', r'company\s+description',
                     r'our\s+mission', r'our\s+values']
    benefits_headers = [r'benefits?', r'perks?', r'what\s+we\s+offer', r'compensation']

    def find_section(headers):
        positions = []
        for h in headers:
            for m in re.finditer(h, text_lower):
                positions.append(m.start())
        return sorted(positions)

    req_pos = find_section(req_headers)
    resp_pos = find_section(resp_headers)
    about_pos = find_section(about_headers)
    ben_pos = find_section(benefits_headers)

    # Find all mgmt term positions
    mgmt_positions = []
    for pat in MGMT_PATTERNS:
        for m in re.finditer(pat, text_lower):
            mgmt_positions.append(m.start())

    if not mgmt_positions:
        return {'in_requirements': False, 'in_responsibilities': False, 'in_qualifications': False,
                'in_about': False, 'in_benefits': False, 'section_unknown': True}

    # Assign each mgmt term to nearest preceding section header
    all_sections = (
        [(p, 'requirements') for p in req_pos] +
        [(p, 'responsibilities') for p in resp_pos] +
        [(p, 'about') for p in about_pos] +
        [(p, 'benefits') for p in ben_pos]
    )
    all_sections.sort(key=lambda x: x[0])

    results = {'in_requirements': False, 'in_responsibilities': False,
               'in_about': False, 'in_benefits': False, 'section_unknown': False}

    for mpos in mgmt_positions:
        # Find nearest preceding section
        best_section = 'unknown'
        for spos, stype in reversed(all_sections):
            if spos < mpos:
                best_section = stype
                break

        if best_section == 'unknown':
            results['section_unknown'] = True
        else:
            results[f'in_{best_section}'] = True

    return results

# Run on 2026 entry-level mgmt postings
print(f"  Analyzing section placement for {len(entry_2026_mgmt)} postings...")
section_results = entry_2026_mgmt['text'].apply(section_analysis)
section_df = pd.DataFrame(section_results.tolist())

print(f"\n  Management terms section placement (2026 entry-level):")
for col in ['in_requirements', 'in_responsibilities', 'in_about', 'in_benefits', 'section_unknown']:
    rate = section_df[col].mean() * 100
    print(f"    {col}: {rate:.1f}%")

# Same for 2024 entry-level
if len(entry_2024_mgmt) > 0:
    section_results_24 = entry_2024_mgmt['text'].apply(section_analysis)
    section_df_24 = pd.DataFrame(section_results_24.tolist())
    print(f"\n  Management terms section placement (2024 entry-level):")
    for col in ['in_requirements', 'in_responsibilities', 'in_about', 'in_benefits', 'section_unknown']:
        rate = section_df_24[col].mean() * 100
        print(f"    {col}: {rate:.1f}%")

# Section comparison table
section_summary = pd.DataFrame({
    'section': ['requirements', 'responsibilities', 'about_company', 'benefits', 'unknown'],
    'pct_2026_entry': [section_df['in_requirements'].mean()*100, section_df['in_responsibilities'].mean()*100,
                       section_df['in_about'].mean()*100, section_df['in_benefits'].mean()*100,
                       section_df['section_unknown'].mean()*100],
})
if len(entry_2024_mgmt) > 0:
    section_summary['pct_2024_entry'] = [
        section_df_24['in_requirements'].mean()*100, section_df_24['in_responsibilities'].mean()*100,
        section_df_24['in_about'].mean()*100, section_df_24['in_benefits'].mean()*100,
        section_df_24['section_unknown'].mean()*100,
    ]
section_summary.to_csv(TBL_DIR / "mgmt_section_placement.csv", index=False)

# =============================================================================
# PART 12: Broader management context — what do mgmt terms look like in context?
# =============================================================================
print("\nExtracting management term contexts...")

def extract_mgmt_contexts(text, n_chars=150):
    """Extract surrounding context for each management term match."""
    if not isinstance(text, str):
        return []
    text_lower = text.lower()
    contexts = []
    for pat in MGMT_PATTERNS:
        for m in re.finditer(pat, text_lower):
            start = max(0, m.start() - n_chars)
            end = min(len(text), m.end() + n_chars)
            # Use original case text
            snippet = text[start:end].strip()
            contexts.append({
                'term': m.group(),
                'context': snippet,
            })
    return contexts

# Get contexts from 2026 entry-level
all_contexts_2026 = []
for _, row in entry_2026_mgmt.iterrows():
    ctxs = extract_mgmt_contexts(row['text'])
    for ctx in ctxs[:3]:  # limit per posting
        ctx['uid'] = row['uid']
        ctx['title'] = row['title']
        ctx['company'] = row['company_name_effective']
        all_contexts_2026.append(ctx)

contexts_df = pd.DataFrame(all_contexts_2026)
if len(contexts_df) > 0:
    # Save sample of 50 contexts
    sample_ctx = contexts_df.sample(min(50, len(contexts_df)), random_state=42)
    sample_ctx.to_csv(TBL_DIR / "mgmt_term_contexts_sample_50.csv", index=False)
    print(f"  Saved {len(sample_ctx)} context samples")

# =============================================================================
# PART 13: Management rate controlling for text source
# =============================================================================
print("\nSensitivity: management rate using different text columns...")

# Recompute mgmt rate using description (full, with boilerplate)
# vs description_core (rule-based cleaned) vs text (best available)
for col_name, col in [('description', 'description'), ('description_core', 'description_core'), ('text', 'text')]:
    rates = df.copy()
    rates['mgmt_check'] = rates[col].apply(has_mgmt_indicator)
    rate_tbl = rates.groupby(['period', 'seniority_3level']).agg(
        n=('uid', 'count'),
        mgmt_rate=('mgmt_check', 'mean'),
    ).reset_index()
    junior_rates = rate_tbl[rate_tbl['seniority_3level'] == 'junior']
    print(f"\n  Mgmt rate (junior) using '{col_name}':")
    for _, r in junior_rates.iterrows():
        print(f"    {r['period']}: {r['mgmt_rate']*100:.1f}% (n={r['n']})")

# =============================================================================
# COMPLETE — print summary
# =============================================================================
print("\n" + "="*70)
print("T22 ANALYSIS COMPLETE")
print("="*70)
print(f"Tables saved to: {TBL_DIR}")
print(f"Figures saved to: {FIG_DIR}")
