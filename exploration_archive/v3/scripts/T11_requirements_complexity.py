#!/usr/bin/env python3
"""T11: Requirements complexity & credential stacking — the scope inflation test.

CRITICAL TASK: Determines whether "scope inflation" exists beyond junior share decline.
If entry-level requirement complexity INCREASED despite lower YOE, that's scope inflation
in a different dimension. If it didn't, the paper leads with slot elimination, not scope inflation.
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import json
from scipy import stats

BASE = Path("/home/jihgaboot/gabor/job-research")
DATA = BASE / "data/unified.parquet"
TECH_MATRIX = BASE / "exploration/artifacts/shared/swe_tech_matrix.parquet"
CLEANED = BASE / "exploration/artifacts/shared/swe_cleaned_text.parquet"
FIG_DIR = BASE / "exploration/figures/T11"
TAB_DIR = BASE / "exploration/tables/T11"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

FILTERS = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe = true"
con = duckdb.connect()

# ============================================================
# 0. Load and join all data
# ============================================================
print("Loading data...")

# Base metadata
meta = con.sql(f"""
    SELECT uid, title, title_normalized, period, source,
           seniority_final, seniority_3level, seniority_final_source,
           is_aggregator, company_name_canonical,
           yoe_extracted, yoe_min_extracted, yoe_max_extracted,
           description_length, core_length,
           swe_classification_tier
    FROM '{DATA}'
    WHERE {FILTERS}
""").fetchdf()
print(f"Total SWE rows: {len(meta)}")

# Tech matrix — binary tech counts
tech = con.sql(f"SELECT * FROM '{TECH_MATRIX}'").fetchdf()
tech_cols = [c for c in tech.columns if c != 'uid']
print(f"Tech matrix: {len(tech)} rows, {len(tech_cols)} technologies")

# Cleaned text for text-based requirement extraction
cleaned = con.sql(f"""
    SELECT uid, description_cleaned, text_source
    FROM '{CLEANED}'
""").fetchdf()
print(f"Cleaned text: {len(cleaned)} rows")

# Merge everything
df = meta.merge(tech, on='uid', how='left').merge(cleaned, on='uid', how='left')
print(f"Merged: {len(df)} rows")

# ============================================================
# 1. Build requirement feature extractor
# ============================================================
print("\n=== 1. Building requirement features ===")

# 1a. Tech count from tech matrix
df['tech_count'] = df[tech_cols].sum(axis=1).astype(int)

# Categorize tech columns
lang_cols = [c for c in tech_cols if c.startswith('lang_')]
fe_cols = [c for c in tech_cols if c.startswith('fe_')]
be_cols = [c for c in tech_cols if c.startswith('be_')]
cloud_cols = [c for c in tech_cols if c.startswith('cloud_')]
devops_cols = [c for c in tech_cols if c.startswith('devops_')]
data_cols = [c for c in tech_cols if c.startswith('data_')]
ml_cols = [c for c in tech_cols if c.startswith('ml_')]
ai_cols = [c for c in tech_cols if c.startswith('ai_')]
tool_cols = [c for c in tech_cols if c.startswith('tool_')]
test_cols = [c for c in tech_cols if c.startswith('test_')]
practice_cols = [c for c in tech_cols if c.startswith('practice_')]
mobile_cols = [c for c in tech_cols if c.startswith('mobile_')]
security_cols = [c for c in tech_cols if c.startswith('security_')]

df['lang_count'] = df[lang_cols].sum(axis=1).astype(int)
df['framework_count'] = df[fe_cols + be_cols].sum(axis=1).astype(int)
df['infra_count'] = df[cloud_cols + devops_cols].sum(axis=1).astype(int)
df['data_count'] = df[data_cols].sum(axis=1).astype(int)
df['ml_ai_count'] = df[ml_cols + ai_cols + tool_cols].sum(axis=1).astype(int)
df['testing_count'] = df[test_cols].sum(axis=1).astype(int)

# 1b. Text-based requirement extraction using regex on cleaned text
def count_pattern_matches(text, patterns):
    """Count distinct pattern matches in text."""
    if pd.isna(text) or not text:
        return 0
    text_lower = text.lower()
    count = 0
    for pat in patterns:
        if re.search(pat, text_lower):
            count += 1
    return count

def has_pattern(text, patterns):
    """Binary: any pattern match."""
    if pd.isna(text) or not text:
        return False
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in patterns)

# Soft skills
soft_skill_patterns = [
    r'\bcollaborat(?:e|ion|ive)\b', r'\bcommunicat(?:e|ion)\b', r'\bteamwork\b',
    r'\bproblem[\s-]solv(?:e|ing)\b', r'\bcritical thinking\b', r'\badaptab(?:le|ility)\b',
    r'\bself[\s-]motivat(?:ed|ion)\b', r'\btime management\b', r'\binterpersonal\b',
    r'\bempathy\b', r'\bcreativi?ty\b', r'\binnovati(?:ve|on)\b', r'\bcurious|curiosity\b',
    r'\bpresentation skills\b', r'\bwritten\s+(?:and\s+)?(?:oral|verbal)\b',
    r'\bverbal\s+(?:and\s+)?written\b', r'\bmentor(?:ing|ship)\b',
    r'\bcoach(?:ing)?\b', r'\bnegotiat(?:e|ion)\b', r'\bconflict resolution\b',
    r'\binfluenc(?:e|ing)\b', r'\bempowe?r(?:ment|ing)?\b',
    r'\bactive listening\b', r'\bwork ethic\b', r'\battention to detail\b',
    r'\banalytical\s+(?:skills?|thinking|ability)\b', r'\bstrategic thinking\b',
]

# Organizational scope / ownership terms
scope_patterns = [
    r'\bownership\b', r'\bend[\s-]to[\s-]end\b', r'\bcross[\s-]functional\b',
    r'\bstakeholder\b', r'\bautonomous(?:ly)?\b', r'\bindependent(?:ly)?\b',
    r'\bself[\s-]directed\b', r'\bfull[\s-]?stack\b.*\bownership\b',
    r'\btechnical\s+(?:leadership|direction)\b', r'\btechnical\s+vision\b',
    r'\barchitectur(?:e|al)\s+(?:decisions?|direction)\b',
    r'\bdesign\s+(?:decisions?|patterns?)\b', r'\bdriv(?:e|ing)\s+(?:technical|engineering)\b',
    r'\bown(?:s|ing)?\s+(?:the|a|entire|full)\b', r'\btake\s+ownership\b',
    r'\bscope\s+(?:and|of)\b.*\bdefin(?:e|ition)\b', r'\broadmap\b',
    r'\bproduct\s+(?:strategy|vision|direction)\b', r'\bstrategic\s+(?:planning|initiatives?)\b',
    r'\binfluence\s+(?:technical|engineering|product)\b',
    r'\bset\s+(?:technical|engineering)\s+(?:direction|standards?)\b',
]

# Management / leadership indicators
mgmt_patterns = [
    r'\bmanag(?:e|ing|ement)\s+(?:a\s+)?team\b', r'\blead(?:ing)?\s+(?:a\s+)?team\b',
    r'\bdirect\s+reports?\b', r'\bhir(?:e|ing)\b', r'\bperformance\s+reviews?\b',
    r'\bpeople\s+management\b', r'\bteam\s+(?:lead|management|building)\b',
    r'\bmentor(?:ing)?\s+(?:junior|engineers?|developers?)\b',
    r'\bcoach(?:ing)?\s+(?:team|engineers?)\b', r'\brecruit(?:ing|ment)?\b',
    r'\bstaff(?:ing)?\s+(?:decisions?|planning)\b',
    r'\b(?:1|2|3|4|5)\+?\s+(?:direct\s+)?reports?\b',
    r'\bbudget\s+(?:management|responsibility|planning)\b',
]

# AI-specific requirements
ai_req_patterns = [
    r'\b(?:ai|artificial intelligence)\s+(?:tools?|models?|systems?)\b',
    r'\bllm(?:s|\'s)?\b', r'\blarge\s+language\s+model\b',
    r'\bgenerative\s+ai\b', r'\bgen[\s-]?ai\b', r'\bprompt\s+engineering\b',
    r'\bai[\s-](?:native|first|powered|driven|enabled)\b',
    r'\bcopilot|github\s+copilot\b', r'\bcursor\b.*\b(?:ide|editor|ai)\b',
    r'\bchatgpt\b', r'\bclaud(?:e|e\s+ai)\b', r'\bgpt[\s-]?\d\b',
    r'\brag\b.*\b(?:retrieval|pipeline|system)\b', r'\bvector\s+(?:database|store|search|db)\b',
    r'\bagent(?:ic|s)?\s+(?:framework|system|pipeline)\b',
    r'\bfine[\s-]?tun(?:e|ing)\b', r'\btransformer(?:s)?\s+(?:model|architecture)\b',
    r'\bembedding(?:s)?\b.*\b(?:model|vector|search)\b',
    r'\bai\s+(?:coding|programming|development)\s+(?:tools?|assistants?)\b',
    r'\bmcp\b.*\b(?:protocol|server|tool)\b',
]

# Education level extraction
def extract_education(text):
    """Extract highest education level mentioned: phd > ms > bs > none."""
    if pd.isna(text) or not text:
        return 'none'
    text_lower = text.lower()
    if re.search(r'\b(?:ph\.?d|doctorate|doctoral)\b', text_lower):
        return 'phd'
    if re.search(r'\b(?:m\.?s\.?|master\'?s?|msc|m\.?eng)\b', text_lower):
        return 'ms'
    if re.search(r'\b(?:b\.?s\.?|bachelor\'?s?|bsc|b\.?eng|b\.?a\.?|undergraduate|degree)\b', text_lower):
        return 'bs'
    return 'none'

print("Extracting text-based features...")
texts = df['description_cleaned'].fillna('')

# Vectorized operations for speed
df['soft_skill_count'] = texts.apply(lambda t: count_pattern_matches(t, soft_skill_patterns))
print("  soft skills done")
df['scope_count'] = texts.apply(lambda t: count_pattern_matches(t, scope_patterns))
print("  scope done")
df['mgmt_count'] = texts.apply(lambda t: count_pattern_matches(t, mgmt_patterns))
print("  management done")
df['ai_req_count'] = texts.apply(lambda t: count_pattern_matches(t, ai_req_patterns))
print("  AI requirements done")
df['education_level'] = texts.apply(extract_education)
print("  education done")

# Binary indicators
df['has_soft_skills'] = (df['soft_skill_count'] > 0).astype(int)
df['has_scope_terms'] = (df['scope_count'] > 0).astype(int)
df['has_mgmt_terms'] = (df['mgmt_count'] > 0).astype(int)
df['has_ai_req'] = (df['ai_req_count'] > 0).astype(int)
df['has_education'] = (df['education_level'] != 'none').astype(int)
df['has_yoe'] = (~df['yoe_extracted'].isna()).astype(int)

# Education level numeric
edu_map = {'none': 0, 'bs': 1, 'ms': 2, 'phd': 3}
df['education_numeric'] = df['education_level'].map(edu_map)

# ============================================================
# 2. Complexity metrics
# ============================================================
print("\n=== 2. Computing complexity metrics ===")

# Requirement categories present (max 7: tech, soft_skills, scope, mgmt, ai, education, yoe)
df['requirement_breadth'] = (
    (df['tech_count'] > 0).astype(int) +
    df['has_soft_skills'] +
    df['has_scope_terms'] +
    df['has_mgmt_terms'] +
    df['has_ai_req'] +
    df['has_education'] +
    df['has_yoe']
)

# Credential stack depth — more fine-grained (max=categories with >=1 mention)
df['credential_stack_depth'] = df['requirement_breadth']  # same concept, different name for now

# Density metrics (per 1K chars)
desc_len = df['description_length'].replace(0, np.nan)
df['tech_density'] = df['tech_count'] / (desc_len / 1000)
df['scope_density'] = df['scope_count'] / (desc_len / 1000)
df['soft_skill_density'] = df['soft_skill_count'] / (desc_len / 1000)
df['ai_density'] = df['ai_req_count'] / (desc_len / 1000)

# Summary stats
print("\nOverall complexity metrics:")
metrics = ['tech_count', 'lang_count', 'framework_count', 'infra_count',
           'soft_skill_count', 'scope_count', 'mgmt_count', 'ai_req_count',
           'requirement_breadth', 'tech_density', 'scope_density']
for m in metrics:
    print(f"  {m}: mean={df[m].mean():.2f}, median={df[m].median():.1f}, "
          f"p25={df[m].quantile(0.25):.1f}, p75={df[m].quantile(0.75):.1f}")

# ============================================================
# 3. Compare distributions by period x seniority
# ============================================================
print("\n=== 3. Period x Seniority comparison ===")

# Exclude asaniczka from seniority-stratified analyses (no entry-level labels)
# Use arshkon for 2024 baseline
df_for_seniority = df[df['source'] != 'kaggle_asaniczka']

key_metrics = ['tech_count', 'lang_count', 'soft_skill_count', 'scope_count',
               'mgmt_count', 'ai_req_count', 'requirement_breadth',
               'tech_density', 'scope_density', 'education_numeric']

comparison_rows = []
for period in ['2024-04', '2026-03']:
    for seniority in ['junior', 'mid', 'senior', 'unknown']:
        sub = df_for_seniority[(df_for_seniority['period'] == period) &
                                (df_for_seniority['seniority_3level'] == seniority)]
        if len(sub) < 5:
            continue
        row = {'period': period, 'seniority': seniority, 'n': len(sub)}
        for m in key_metrics:
            row[f'{m}_mean'] = sub[m].mean()
            row[f'{m}_median'] = sub[m].median()
        row['pct_has_ai'] = sub['has_ai_req'].mean()
        row['pct_has_education'] = sub['has_education'].mean()
        row['pct_has_yoe'] = sub['has_yoe'].mean()
        row['pct_has_mgmt'] = sub['has_mgmt_terms'].mean()
        row['pct_has_scope'] = sub['has_scope_terms'].mean()
        row['yoe_mean'] = sub['yoe_extracted'].dropna().mean()
        comparison_rows.append(row)

# Also: all-seniority comparison by period (using all three sources for volume)
for period in ['2024-01', '2024-04', '2026-03']:
    sub = df[df['period'] == period]
    row = {'period': period, 'seniority': 'ALL', 'n': len(sub)}
    for m in key_metrics:
        row[f'{m}_mean'] = sub[m].mean()
        row[f'{m}_median'] = sub[m].median()
    row['pct_has_ai'] = sub['has_ai_req'].mean()
    row['pct_has_education'] = sub['has_education'].mean()
    row['pct_has_yoe'] = sub['has_yoe'].mean()
    row['pct_has_mgmt'] = sub['has_mgmt_terms'].mean()
    row['pct_has_scope'] = sub['has_scope_terms'].mean()
    row['yoe_mean'] = sub['yoe_extracted'].dropna().mean()
    comparison_rows.append(row)

comp_df = pd.DataFrame(comparison_rows)
print(comp_df.to_string())
comp_df.to_csv(TAB_DIR / "complexity_by_period_seniority.csv", index=False)

# ============================================================
# 4. Credential stacking — are 2026 postings asking for MORE types simultaneously?
# ============================================================
print("\n=== 4. Credential stacking ===")

stacking_results = []
for period in ['2024-01', '2024-04', '2026-03']:
    sub = df[df['period'] == period]
    dist = sub['requirement_breadth'].value_counts(normalize=True).sort_index()
    row = {'period': period, 'n': len(sub),
           'mean_breadth': sub['requirement_breadth'].mean(),
           'median_breadth': sub['requirement_breadth'].median(),
           'pct_breadth_ge4': (sub['requirement_breadth'] >= 4).mean(),
           'pct_breadth_ge5': (sub['requirement_breadth'] >= 5).mean(),
           'pct_breadth_ge6': (sub['requirement_breadth'] >= 6).mean(),
           }
    stacking_results.append(row)
    print(f"\n{period} (n={len(sub)}):")
    print(f"  Mean breadth: {sub['requirement_breadth'].mean():.2f}")
    print(f"  Median breadth: {sub['requirement_breadth'].median():.0f}")
    print(f"  % >= 4 categories: {(sub['requirement_breadth'] >= 4).mean():.1%}")
    print(f"  % >= 5 categories: {(sub['requirement_breadth'] >= 5).mean():.1%}")
    print(f"  % >= 6 categories: {(sub['requirement_breadth'] >= 6).mean():.1%}")
    print(f"  Distribution: {dist.round(3).to_dict()}")

# By seniority (arshkon vs scraped)
print("\nCredential stacking by seniority:")
for seniority in ['junior', 'mid', 'senior']:
    for period in ['2024-04', '2026-03']:
        sub = df_for_seniority[(df_for_seniority['period'] == period) &
                                (df_for_seniority['seniority_3level'] == seniority)]
        if len(sub) < 5:
            continue
        print(f"  {period} {seniority} (n={len(sub)}): mean breadth={sub['requirement_breadth'].mean():.2f}, "
              f">=5: {(sub['requirement_breadth'] >= 5).mean():.1%}")
        stacking_results.append({
            'period': period, 'seniority': seniority, 'n': len(sub),
            'mean_breadth': sub['requirement_breadth'].mean(),
            'median_breadth': sub['requirement_breadth'].median(),
            'pct_breadth_ge4': (sub['requirement_breadth'] >= 4).mean(),
            'pct_breadth_ge5': (sub['requirement_breadth'] >= 5).mean(),
            'pct_breadth_ge6': (sub['requirement_breadth'] >= 6).mean(),
        })

pd.DataFrame(stacking_results).to_csv(TAB_DIR / "credential_stacking.csv", index=False)

# ============================================================
# 5. Entry-level complexity: THE DIRECT SCOPE INFLATION TEST
# ============================================================
print("\n=== 5. ENTRY-LEVEL SCOPE INFLATION TEST ===")

entry_2024 = df_for_seniority[(df_for_seniority['period'] == '2024-04') &
                               (df_for_seniority['seniority_3level'] == 'junior')]
entry_2026 = df_for_seniority[(df_for_seniority['period'] == '2026-03') &
                               (df_for_seniority['seniority_3level'] == 'junior')]

print(f"Entry-level 2024 (arshkon): n={len(entry_2024)}")
print(f"Entry-level 2026 (scraped): n={len(entry_2026)}")

all_metrics = ['tech_count', 'lang_count', 'framework_count', 'infra_count',
               'data_count', 'ml_ai_count', 'testing_count',
               'soft_skill_count', 'scope_count', 'mgmt_count', 'ai_req_count',
               'requirement_breadth', 'tech_density', 'scope_density',
               'soft_skill_density', 'ai_density', 'education_numeric']

scope_test_results = []
for m in all_metrics:
    v2024 = entry_2024[m].dropna()
    v2026 = entry_2026[m].dropna()
    if len(v2024) < 5 or len(v2026) < 5:
        continue
    # Mann-Whitney U test (non-parametric)
    stat, pval = stats.mannwhitneyu(v2024, v2026, alternative='two-sided')
    # Effect size (rank-biserial)
    n1, n2 = len(v2024), len(v2026)
    r_effect = 1 - (2 * stat) / (n1 * n2)

    result = {
        'metric': m,
        'entry_2024_mean': v2024.mean(),
        'entry_2024_median': v2024.median(),
        'entry_2026_mean': v2026.mean(),
        'entry_2026_median': v2026.median(),
        'change_mean': v2026.mean() - v2024.mean(),
        'change_pct': ((v2026.mean() - v2024.mean()) / v2024.mean() * 100) if v2024.mean() != 0 else np.nan,
        'mann_whitney_p': pval,
        'rank_biserial_r': r_effect,
        'n_2024': len(v2024),
        'n_2026': len(v2026),
    }
    scope_test_results.append(result)
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    print(f"  {m}: 2024={v2024.mean():.2f} -> 2026={v2026.mean():.2f} "
          f"({v2026.mean() - v2024.mean():+.2f}, {sig}, r={r_effect:.3f})")

scope_df = pd.DataFrame(scope_test_results)
scope_df.to_csv(TAB_DIR / "entry_level_scope_inflation_test.csv", index=False)

# Binary indicator comparison for entry-level
print("\nEntry-level binary indicators:")
binary_metrics = ['has_soft_skills', 'has_scope_terms', 'has_mgmt_terms', 'has_ai_req',
                  'has_education', 'has_yoe']
binary_results = []
for m in binary_metrics:
    p2024 = entry_2024[m].mean()
    p2026 = entry_2026[m].mean()
    # Two-proportion z-test
    n1, n2 = len(entry_2024), len(entry_2026)
    p_pool = (entry_2024[m].sum() + entry_2026[m].sum()) / (n1 + n2)
    if p_pool > 0 and p_pool < 1:
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        z = (p2026 - p2024) / se
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        z, pval = np.nan, np.nan
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    print(f"  {m}: 2024={p2024:.1%} -> 2026={p2026:.1%} ({p2026 - p2024:+.1%}, {sig})")
    binary_results.append({
        'metric': m, 'entry_2024': p2024, 'entry_2026': p2026,
        'pp_change': p2026 - p2024, 'z_stat': z, 'p_value': pval
    })

pd.DataFrame(binary_results).to_csv(TAB_DIR / "entry_level_binary_indicators.csv", index=False)

# Education level distribution
print("\nEntry-level education distribution:")
for period, sub in [('2024-04', entry_2024), ('2026-03', entry_2026)]:
    dist = sub['education_level'].value_counts(normalize=True).sort_index()
    print(f"  {period}: {dist.to_dict()}")

# YOE for entry-level
print("\nEntry-level YOE:")
for period, sub in [('2024-04', entry_2024), ('2026-03', entry_2026)]:
    yoe = sub['yoe_extracted'].dropna()
    print(f"  {period}: n={len(yoe)}, mean={yoe.mean():.1f}, median={yoe.median():.1f}")

# ============================================================
# 5b. Same analysis for MID and SENIOR to see if it's universal
# ============================================================
print("\n=== 5b. Complexity change by ALL seniority levels ===")

for seniority in ['junior', 'mid', 'senior']:
    s2024 = df_for_seniority[(df_for_seniority['period'] == '2024-04') &
                              (df_for_seniority['seniority_3level'] == seniority)]
    s2026 = df_for_seniority[(df_for_seniority['period'] == '2026-03') &
                              (df_for_seniority['seniority_3level'] == seniority)]
    if len(s2024) < 5 or len(s2026) < 5:
        continue
    print(f"\n  {seniority.upper()} (n={len(s2024)} -> {len(s2026)}):")
    for m in ['tech_count', 'requirement_breadth', 'scope_count', 'ai_req_count',
              'soft_skill_count', 'mgmt_count']:
        v24 = s2024[m].mean()
        v26 = s2026[m].mean()
        stat, pval = stats.mannwhitneyu(s2024[m].dropna(), s2026[m].dropna(), alternative='two-sided')
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"    {m}: {v24:.2f} -> {v26:.2f} ({v26-v24:+.2f}, {sig})")

# ============================================================
# SENSITIVITY CHECKS
# ============================================================
print("\n=== SENSITIVITY CHECKS ===")

# (a) Aggregator exclusion
print("\n--- (a) Aggregator exclusion ---")
for seniority in ['junior', 'mid', 'senior']:
    for agg_filter in [False, True]:
        sub = df_for_seniority.copy()
        if agg_filter:
            sub = sub[~sub['is_aggregator']]
        for period in ['2024-04', '2026-03']:
            s = sub[(sub['period'] == period) & (sub['seniority_3level'] == seniority)]
            if len(s) < 5:
                continue
            tag = "no_agg" if agg_filter else "all"
            print(f"  {seniority} {period} {tag} (n={len(s)}): tech={s['tech_count'].mean():.1f}, "
                  f"breadth={s['requirement_breadth'].mean():.2f}, ai={s['has_ai_req'].mean():.1%}")

# (b) Company capping (max 20 postings per company per period x seniority)
print("\n--- (b) Company capping (max 20) ---")
df_capped = df_for_seniority.copy()
df_capped['cap_key'] = df_capped['period'] + '_' + df_capped['seniority_3level'] + '_' + df_capped['company_name_canonical'].fillna('UNKNOWN')
df_capped = df_capped.groupby('cap_key').apply(
    lambda x: x.sample(n=min(20, len(x)), random_state=42),
    include_groups=False
).reset_index(drop=True)

for seniority in ['junior', 'mid', 'senior']:
    for period in ['2024-04', '2026-03']:
        s = df_capped[(df_capped['period'] == period) & (df_capped['seniority_3level'] == seniority)]
        if len(s) < 5:
            continue
        print(f"  {seniority} {period} capped (n={len(s)}): tech={s['tech_count'].mean():.1f}, "
              f"breadth={s['requirement_breadth'].mean():.2f}, ai={s['has_ai_req'].mean():.1%}")

# (c) Seniority operationalization: seniority_final vs seniority_3level
print("\n--- (c) Seniority operationalization ---")
# Compare using seniority_final 'entry' vs 'associate' vs seniority_3level 'junior'
for period in ['2024-04', '2026-03']:
    for sen_val in ['entry', 'associate']:
        s = df_for_seniority[(df_for_seniority['period'] == period) &
                              (df_for_seniority['seniority_final'] == sen_val)]
        if len(s) < 5:
            continue
        print(f"  {period} seniority_final={sen_val} (n={len(s)}): tech={s['tech_count'].mean():.1f}, "
              f"breadth={s['requirement_breadth'].mean():.2f}")

# (f) Within-2024 calibration: arshkon vs asaniczka (ALL seniority)
print("\n--- (f) Within-2024 calibration ---")
for source_name, source_val in [('arshkon', 'kaggle_arshkon'), ('asaniczka', 'kaggle_asaniczka'), ('scraped', 'scraped')]:
    sub = df[df['source'] == source_val]
    print(f"  {source_name} (n={len(sub)}): tech={sub['tech_count'].mean():.1f}, "
          f"breadth={sub['requirement_breadth'].mean():.2f}, "
          f"scope={sub['scope_count'].mean():.2f}, ai={sub['has_ai_req'].mean():.1%}, "
          f"soft={sub['soft_skill_count'].mean():.2f}")

# ============================================================
# 6. Outlier analysis: top 1% by requirement_breadth
# ============================================================
print("\n=== 6. Outlier analysis ===")

p99 = df['requirement_breadth'].quantile(0.99)
outliers = df[df['requirement_breadth'] >= p99]
print(f"Top 1% threshold: breadth >= {p99}")
print(f"Outlier count: {len(outliers)}")

# Are outliers template-bloated?
outlier_stats = []
for period in ['2024-01', '2024-04', '2026-03']:
    sub = outliers[outliers['period'] == period]
    if len(sub) == 0:
        continue
    company_conc = sub['company_name_canonical'].value_counts()
    top_company = company_conc.index[0] if len(company_conc) > 0 else 'N/A'
    top_share = company_conc.iloc[0] / len(sub) if len(sub) > 0 else 0
    unique_companies = sub['company_name_canonical'].nunique()
    print(f"  {period}: {len(sub)} outliers, {unique_companies} companies, "
          f"top company: {top_company} ({top_share:.1%}), "
          f"mean desc_len={sub['description_length'].mean():.0f}, "
          f"mean tech={sub['tech_count'].mean():.1f}")
    outlier_stats.append({
        'period': period, 'n': len(sub), 'unique_companies': unique_companies,
        'top_company': top_company, 'top_company_share': top_share,
        'mean_desc_length': sub['description_length'].mean(),
        'mean_tech_count': sub['tech_count'].mean(),
        'is_aggregator_pct': sub['is_aggregator'].mean()
    })

pd.DataFrame(outlier_stats).to_csv(TAB_DIR / "outlier_analysis.csv", index=False)

# ============================================================
# FIGURES
# ============================================================
print("\n=== Generating figures ===")

# Fig 1: Complexity distributions by period (violin plots)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics_to_plot = [
    ('tech_count', 'Distinct Technology Mentions'),
    ('requirement_breadth', 'Requirement Breadth (max 7)'),
    ('scope_count', 'Organizational Scope Terms'),
    ('ai_req_count', 'AI-Specific Requirements'),
]

for ax, (metric, label) in zip(axes.flat, metrics_to_plot):
    plot_df = df[df['period'].isin(['2024-01', '2024-04', '2026-03'])][['period', metric]].dropna()
    # Cap outliers for visualization
    cap = plot_df[metric].quantile(0.99)
    plot_df[metric] = plot_df[metric].clip(upper=cap)

    parts = ax.violinplot(
        [plot_df[plot_df['period'] == p][metric].values for p in ['2024-01', '2024-04', '2026-03']],
        positions=[1, 2, 3], showmedians=True, showextrema=False
    )
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['2024-01\n(asaniczka)', '2024-04\n(arshkon)', '2026-03\n(scraped)'])
    ax.set_ylabel(label)
    ax.set_title(label)
    # Add means
    for i, p in enumerate(['2024-01', '2024-04', '2026-03'], 1):
        mean_val = plot_df[plot_df['period'] == p][metric].mean()
        ax.scatter([i], [mean_val], color='red', zorder=3, s=50, marker='D', label='mean' if i == 1 else '')
    if ax == axes.flat[0]:
        ax.legend(fontsize=8)

plt.suptitle('Requirements Complexity Distributions by Period (SWE, LinkedIn)', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "complexity_distributions.png", dpi=150, bbox_inches='tight')
plt.close()

# Fig 2: Entry-level scope inflation — the money chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Key metrics comparison (entry-level only)
entry_metrics = ['tech_count', 'soft_skill_count', 'scope_count', 'ai_req_count',
                 'requirement_breadth', 'education_numeric']
entry_labels = ['Tech\nCount', 'Soft\nSkills', 'Scope\nTerms', 'AI\nReqs',
                'Req.\nBreadth', 'Education\nLevel']

means_24 = [entry_2024[m].mean() for m in entry_metrics]
means_26 = [entry_2026[m].mean() for m in entry_metrics]

x = np.arange(len(entry_metrics))
width = 0.35

ax = axes[0]
bars1 = ax.bar(x - width/2, means_24, width, label=f'Entry 2024 (n={len(entry_2024)})', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, means_26, width, label=f'Entry 2026 (n={len(entry_2026)})', color='coral', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(entry_labels, fontsize=9)
ax.set_ylabel('Mean count / level')
ax.set_title('A. Entry-Level Requirements: 2024 vs 2026')
ax.legend()
ax.grid(True, alpha=0.2, axis='y')

# Panel B: Requirement breadth distributions
ax = axes[1]
bins = np.arange(0, 8) - 0.5
ax.hist(entry_2024['requirement_breadth'], bins=bins, alpha=0.6, density=True,
        label=f'Entry 2024 (n={len(entry_2024)})', color='steelblue')
ax.hist(entry_2026['requirement_breadth'], bins=bins, alpha=0.6, density=True,
        label=f'Entry 2026 (n={len(entry_2026)})', color='coral')
ax.set_xlabel('Requirement Breadth (categories with >=1 mention)')
ax.set_ylabel('Density')
ax.set_title('B. Credential Stacking: Entry-Level')
ax.legend()
ax.set_xticks(range(8))

plt.tight_layout()
plt.savefig(FIG_DIR / "entry_level_scope_inflation.png", dpi=150, bbox_inches='tight')
plt.close()

# Fig 3: Heatmap of complexity change by seniority
fig, ax = plt.subplots(figsize=(12, 5))

heatmap_metrics = ['tech_count', 'requirement_breadth', 'scope_count',
                   'ai_req_count', 'soft_skill_count', 'mgmt_count']
heatmap_labels = ['Tech Count', 'Req. Breadth', 'Scope Terms',
                  'AI Requirements', 'Soft Skills', 'Management']

change_data = []
for seniority in ['junior', 'mid', 'senior']:
    s24 = df_for_seniority[(df_for_seniority['period'] == '2024-04') &
                            (df_for_seniority['seniority_3level'] == seniority)]
    s26 = df_for_seniority[(df_for_seniority['period'] == '2026-03') &
                            (df_for_seniority['seniority_3level'] == seniority)]
    if len(s24) < 5 or len(s26) < 5:
        change_data.append([0] * len(heatmap_metrics))
        continue
    changes = []
    for m in heatmap_metrics:
        m24 = s24[m].mean()
        m26 = s26[m].mean()
        # Percent change
        pct = ((m26 - m24) / m24 * 100) if m24 != 0 else (100 if m26 > 0 else 0)
        changes.append(pct)
    change_data.append(changes)

change_matrix = np.array(change_data)
im = ax.imshow(change_matrix, cmap='RdBu_r', aspect='auto', vmin=-100, vmax=200)
ax.set_xticks(range(len(heatmap_labels)))
ax.set_xticklabels(heatmap_labels, rotation=30, ha='right')
ax.set_yticks(range(3))
ax.set_yticklabels(['Junior', 'Mid', 'Senior'])
ax.set_title('Percent Change in Requirements Complexity (2024-04 to 2026-03)')
plt.colorbar(im, ax=ax, label='% Change')

# Add text annotations
for i in range(3):
    for j in range(len(heatmap_metrics)):
        text = ax.text(j, i, f'{change_matrix[i, j]:.0f}%',
                       ha='center', va='center', fontsize=10,
                       color='white' if abs(change_matrix[i, j]) > 80 else 'black')

plt.tight_layout()
plt.savefig(FIG_DIR / "complexity_change_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

# Fig 4: Credential stacking by period (stacked bar of categories present)
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['has_tech', 'has_soft_skills', 'has_scope_terms', 'has_mgmt_terms',
              'has_ai_req', 'has_education', 'has_yoe']
cat_labels = ['Technology', 'Soft Skills', 'Scope/Ownership', 'Management',
              'AI Requirements', 'Education', 'YOE Specified']

df['has_tech'] = (df['tech_count'] > 0).astype(int)

stacking_data = {}
for period in ['2024-01', '2024-04', '2026-03']:
    sub = df[df['period'] == period]
    stacking_data[period] = [sub[c].mean() for c in categories]

x = np.arange(3)
width = 0.5
bottom = np.zeros(3)
colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))

for i, (cat, label) in enumerate(zip(categories, cat_labels)):
    vals = [stacking_data[p][i] for p in ['2024-01', '2024-04', '2026-03']]
    ax.bar(x, vals, width, bottom=bottom, label=label, color=colors[i], alpha=0.85)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(['2024-01\n(asaniczka)', '2024-04\n(arshkon)', '2026-03\n(scraped)'])
ax.set_ylabel('Cumulative Category Presence Rate')
ax.set_title('Credential Stacking: Category Presence by Period')
ax.legend(loc='upper right', fontsize=8)
ax.set_ylim(0, max(bottom) * 1.05)
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(FIG_DIR / "credential_stacking.png", dpi=150, bbox_inches='tight')
plt.close()

print("\n=== T11 complete ===")
print(f"Figures: {list(FIG_DIR.glob('*.png'))}")
print(f"Tables: {list(TAB_DIR.glob('*.csv'))}")
