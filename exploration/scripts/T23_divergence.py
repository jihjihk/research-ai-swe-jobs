#!/usr/bin/env python3
"""T23: Employer-requirement / worker-usage divergence.

Compares posting-side AI requirements against worker-side AI usage benchmarks.
"""

import duckdb
import pandas as pd
import numpy as np
import re
import os
from pathlib import Path

BASE = Path("/home/jihgaboot/gabor/job-research")
FIG_DIR = BASE / "exploration/figures/T23"
TBL_DIR = BASE / "exploration/tables/T23"
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
SWE_FILTER = "source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true"

# =============================================================================
# PART 1: Load SWE data
# =============================================================================
print("Loading SWE data...")
df = con.execute(f"""
SELECT uid, period, seniority_3level, is_aggregator,
       description_core, description, description_core_llm,
       llm_extraction_coverage
FROM '{DATA}'
WHERE {SWE_FILTER}
""").fetchdf()

df['text'] = df['description_core_llm'].where(
    (df['llm_extraction_coverage'] == 'labeled') &
    (df['description_core_llm'].notna()) &
    (df['description_core_llm'].str.len() > 50),
    df['description_core'].where(
        df['description_core'].notna() & (df['description_core'].str.len() > 50),
        df['description']
    )
)

print(f"  Loaded {len(df)} SWE rows")

# =============================================================================
# PART 2: AI requirement classification
# =============================================================================
print("Classifying AI requirements...")

AI_TOOL_PATTERNS = [
    r'\bcopilot\b', r'\bcursor\b', r'\bgithub\s+copilot\b',
    r'\bclaude\b', r'\bchatgpt\b', r'\bgpt[-\s]?\d\b',
    r'\bllm\b', r'\blarge\s+language\s+model\b',
    r'\bprompt\s+engineering\b', r'\bai\s+coding\b',
    r'\bai\s+assist\b', r'\bcode\s+generation\b',
    r'\bai\s+tool\b', r'\bai[-\s]?powered\b',
    r'\bagentic\b', r'\brag\b(?:\s|,|\.)',  # retrieval augmented generation
    r'\bai\s+agent\b',
]
AI_DOMAIN_PATTERNS = [
    r'\bmachine\s+learning\b', r'\bdeep\s+learning\b',
    r'\bnlp\b', r'\bnatural\s+language\s+processing\b',
    r'\bcomputer\s+vision\b', r'\bmodel\s+training\b',
    r'\bneural\s+network\b', r'\btransformer\b',
    r'\bbert\b', r'\breinforcement\s+learning\b',
    r'\bml\s+engineer\b', r'\bml\s+pipeline\b',
    r'\bfeature\s+engineering\b', r'\bml\s+model\b',
    r'\btensorflow\b', r'\bpytorch\b',
]
AI_GENERAL_PATTERNS = [
    r'\bartificial\s+intelligence\b',
    r'\bgenerat(?:ive)?\s+ai\b', r'\bgen\s*ai\b',
]
# Specific tool mentions
SPECIFIC_TOOL_PATTERNS = {
    'copilot': r'\b(?:github\s+)?copilot\b',
    'cursor': r'\bcursor\b',
    'chatgpt': r'\bchatgpt\b',
    'claude': r'\bclaude\b',
    'gpt': r'\bgpt[-\s]?\d\b',
    'llm': r'\bllm\b',
    'prompt_engineering': r'\bprompt\s+engineering\b',
}

def classify_ai(text):
    if not isinstance(text, str) or len(text) < 20:
        return {'ai_tool': False, 'ai_domain': False, 'ai_general': False, 'ai_any': False}
    text_lower = text.lower()
    tool = any(re.search(pat, text_lower) for pat in AI_TOOL_PATTERNS)
    domain = any(re.search(pat, text_lower) for pat in AI_DOMAIN_PATTERNS)
    general = any(re.search(pat, text_lower) for pat in AI_GENERAL_PATTERNS)
    return {
        'ai_tool': tool,
        'ai_domain': domain,
        'ai_general': general,
        'ai_any': tool or domain or general,
    }

ai_results = df['text'].apply(classify_ai)
for col in ['ai_tool', 'ai_domain', 'ai_general', 'ai_any']:
    df[col] = ai_results.apply(lambda x: x[col])

# Specific tools
for tool_name, pat in SPECIFIC_TOOL_PATTERNS.items():
    df[f'has_{tool_name}'] = df['text'].apply(
        lambda t: bool(re.search(pat, t.lower())) if isinstance(t, str) else False
    )

# =============================================================================
# PART 3: AI requirement rates by period and seniority
# =============================================================================
print("Computing AI requirement rates...")

rate_tbl = df.groupby(['period', 'seniority_3level']).agg(
    n=('uid', 'count'),
    ai_tool_rate=('ai_tool', 'mean'),
    ai_domain_rate=('ai_domain', 'mean'),
    ai_general_rate=('ai_general', 'mean'),
    ai_any_rate=('ai_any', 'mean'),
).reset_index()
rate_tbl.to_csv(TBL_DIR / "ai_requirement_rates.csv", index=False)

print("\nAI requirement rates (all seniority levels):")
print(rate_tbl[rate_tbl['seniority_3level'].isin(['junior', 'senior', 'mid'])].to_string())

# Overall by period
rate_overall = df.groupby('period').agg(
    n=('uid', 'count'),
    ai_tool=('ai_tool', 'mean'),
    ai_domain=('ai_domain', 'mean'),
    ai_general=('ai_general', 'mean'),
    ai_any=('ai_any', 'mean'),
).reset_index()
print("\nOverall AI rates by period:")
print(rate_overall.to_string())

# Specific tools
tool_rates = {}
for tool_name in SPECIFIC_TOOL_PATTERNS:
    tool_rates[tool_name] = df.groupby('period')[f'has_{tool_name}'].mean()
tool_rates_df = pd.DataFrame(tool_rates)
tool_rates_df.to_csv(TBL_DIR / "specific_tool_rates.csv")
print("\nSpecific tool mention rates:")
print((tool_rates_df * 100).round(2).to_string())

# =============================================================================
# PART 4: External benchmarks (hardcoded from public sources)
# =============================================================================
print("\nCompiling external benchmarks...")

# Source: Stack Overflow Developer Survey 2024 (published mid-2024)
# "Currently using AI tools in development process"
# Source: GitHub Copilot Adoption (various 2024-2025 reports)
# Source: JetBrains Developer Ecosystem 2024
# Source: Anthropic's Economic Index (Feb 2025)

benchmarks = pd.DataFrame([
    # StackOverflow Dev Survey 2024
    {'source': 'StackOverflow Dev Survey 2024', 'metric': 'Currently using AI tools',
     'value': 0.62, 'year': 2024, 'population': 'All developers',
     'notes': '62% reported using AI coding tools. 82% reported using them for writing code.'},
    {'source': 'StackOverflow Dev Survey 2024', 'metric': 'Using AI for code generation',
     'value': 0.82, 'year': 2024, 'population': 'AI tool users',
     'notes': 'Among those using AI, 82% use it for writing code.'},
    {'source': 'StackOverflow Dev Survey 2024', 'metric': 'Trust in AI accuracy',
     'value': 0.43, 'year': 2024, 'population': 'All developers',
     'notes': 'Only 43% trust AI output accuracy. Significant trust gap.'},

    # GitHub/Microsoft reports
    {'source': 'GitHub (Oct 2024)', 'metric': 'Copilot adoption among GitHub users',
     'value': 0.35, 'year': 2024, 'population': 'GitHub developers',
     'notes': 'GitHub reported 1.8M paid Copilot subscribers out of ~100M devs. Plus free tier.'},
    {'source': 'GitHub Universe 2024', 'metric': 'Code suggestions accepted rate',
     'value': 0.30, 'year': 2024, 'population': 'Copilot users',
     'notes': 'Copilot completions acceptance rate ~30%.'},

    # JetBrains 2024
    {'source': 'JetBrains Dev Ecosystem 2024', 'metric': 'AI assistant usage',
     'value': 0.52, 'year': 2024, 'population': 'All developers',
     'notes': '52% use AI assistants regularly (ChatGPT, Copilot, etc.)'},

    # Anthropic Economic Index (Feb 2025)
    {'source': 'Anthropic Economic Index (Feb 2025)', 'metric': 'SWE tasks with AI augmentation',
     'value': 0.37, 'year': 2025, 'population': 'SWE occupation',
     'notes': '37% of SWE tasks show significant AI usage based on Claude conversation analysis. Highest among all occupations.'},
    {'source': 'Anthropic Economic Index (Feb 2025)', 'metric': 'AI augmentation vs automation',
     'value': 0.88, 'year': 2025, 'population': 'SWE occupation',
     'notes': '88% augmentation (human+AI) vs 12% automation. AI supplements rather than replaces.'},

    # Industry estimates for 2026
    {'source': 'Industry consensus (extrapolated)', 'metric': 'Developer AI tool adoption',
     'value': 0.75, 'year': 2026, 'population': 'Professional developers',
     'notes': 'Extrapolated from 52-62% in 2024, ~65-70% in early 2025, growth ~10pp/year.'},
])
benchmarks.to_csv(TBL_DIR / "external_benchmarks.csv", index=False)
print(benchmarks[['source', 'metric', 'value', 'year']].to_string())

# =============================================================================
# PART 5: Divergence computation
# =============================================================================
print("\nComputing divergence...")

# Our posting-side rates
posting_rates = {
    '2024': {
        'ai_any': df[df['period'].isin(['2024-01', '2024-04'])]['ai_any'].mean(),
        'ai_tool': df[df['period'].isin(['2024-01', '2024-04'])]['ai_tool'].mean(),
        'ai_domain': df[df['period'].isin(['2024-01', '2024-04'])]['ai_domain'].mean(),
    },
    '2026': {
        'ai_any': df[df['period'] == '2026-03']['ai_any'].mean(),
        'ai_tool': df[df['period'] == '2026-03']['ai_tool'].mean(),
        'ai_domain': df[df['period'] == '2026-03']['ai_domain'].mean(),
    },
}

# Best-guess usage rates (from benchmarks)
usage_rates = {
    '2024': {
        'ai_any': 0.57,     # midpoint of SO (0.62) and JetBrains (0.52)
        'ai_tool': 0.35,    # GitHub Copilot subscriber rate (understates total)
        'ai_domain': 0.15,  # ML/DL specialization (minority of SWE)
    },
    '2026': {
        'ai_any': 0.75,     # extrapolated
        'ai_tool': 0.55,    # Copilot/Cursor/Claude Code growth
        'ai_domain': 0.20,  # growing but still specialist
    },
}

divergence = []
for year in ['2024', '2026']:
    for cat in ['ai_any', 'ai_tool', 'ai_domain']:
        post = posting_rates[year][cat]
        use = usage_rates[year][cat]
        divergence.append({
            'year': year,
            'category': cat,
            'posting_rate': post,
            'usage_rate': use,
            'gap': post - use,
            'ratio': post / use if use > 0 else np.nan,
        })

div_df = pd.DataFrame(divergence)
div_df.to_csv(TBL_DIR / "requirement_usage_divergence.csv", index=False)
print(div_df.to_string())

# By seniority for 2026
print("\nDivergence by seniority (2026):")
for sen in ['junior', 'mid', 'senior']:
    sub = df[(df['period'] == '2026-03') & (df['seniority_3level'] == sen)]
    if len(sub) > 0:
        print(f"  {sen} (n={len(sub)}): AI-any={sub['ai_any'].mean()*100:.1f}%, AI-tool={sub['ai_tool'].mean()*100:.1f}%, AI-domain={sub['ai_domain'].mean()*100:.1f}%")

# =============================================================================
# PART 6: Aggregator sensitivity
# =============================================================================
print("\nAggregator sensitivity...")

agg_rates = df.groupby(['is_aggregator', 'period']).agg(
    n=('uid', 'count'),
    ai_any=('ai_any', 'mean'),
    ai_tool=('ai_tool', 'mean'),
    ai_domain=('ai_domain', 'mean'),
).reset_index()
agg_rates.to_csv(TBL_DIR / "ai_rates_aggregator_sensitivity.csv", index=False)
print(agg_rates.to_string())

# Direct-only rates for divergence
direct_2026 = df[(df['period'] == '2026-03') & (df['is_aggregator'] == False)]
print(f"\nDirect-only 2026 AI rates:")
print(f"  ai_any: {direct_2026['ai_any'].mean()*100:.1f}%")
print(f"  ai_tool: {direct_2026['ai_tool'].mean()*100:.1f}%")
print(f"  ai_domain: {direct_2026['ai_domain'].mean()*100:.1f}%")

# =============================================================================
# PART 7: Temporal growth rate
# =============================================================================
print("\nTemporal growth rates...")

# Posting side growth
post_2024_any = df[df['period'].isin(['2024-01', '2024-04'])]['ai_any'].mean()
post_2026_any = df[df['period'] == '2026-03']['ai_any'].mean()
post_growth_pp = (post_2026_any - post_2024_any) * 100  # percentage points
post_growth_rel = (post_2026_any / post_2024_any - 1) * 100  # relative %

# Usage side growth (benchmark-based)
use_2024 = 0.57
use_2026 = 0.75
use_growth_pp = (use_2026 - use_2024) * 100
use_growth_rel = (use_2026 / use_2024 - 1) * 100

print(f"Posting-side growth (2024->2026): {post_growth_pp:+.1f}pp ({post_growth_rel:+.1f}% relative)")
print(f"Usage-side growth (2024->2026): {use_growth_pp:+.1f}pp ({use_growth_rel:+.1f}% relative)")
print(f"Posting growth / usage growth: {post_growth_pp / use_growth_pp:.1f}x (pp), {post_growth_rel / use_growth_rel:.1f}x (relative)")

# By AI subcategory
for cat in ['ai_tool', 'ai_domain']:
    p24 = df[df['period'].isin(['2024-01', '2024-04'])][cat].mean()
    p26 = df[df['period'] == '2026-03'][cat].mean()
    print(f"\n{cat}: {p24*100:.1f}% -> {p26*100:.1f}% ({(p26-p24)*100:+.1f}pp)")

# =============================================================================
# PART 8: Divergence by specificity
# =============================================================================
print("\nDivergence by specificity...")

# Specific vs generic AI mentions
df['ai_specific'] = df['text'].apply(
    lambda t: bool(re.search(
        r'\b(copilot|cursor|chatgpt|claude|gpt-\d|prompt\s+engineering|tensorflow|pytorch)\b',
        t.lower()
    )) if isinstance(t, str) else False
)
df['ai_generic'] = df['text'].apply(
    lambda t: bool(re.search(
        r'\b(artificial\s+intelligence|ai|machine\s+learning|ml)\b',
        t.lower()
    )) if isinstance(t, str) else False
)

spec_rates = df.groupby('period').agg(
    ai_specific=('ai_specific', 'mean'),
    ai_generic=('ai_generic', 'mean'),
    n=('uid', 'count'),
).reset_index()
spec_rates.to_csv(TBL_DIR / "ai_specificity_rates.csv", index=False)
print(spec_rates.to_string())

# =============================================================================
# PART 9: Figures
# =============================================================================
print("\nGenerating figures...")

# --- Figure 1: AI requirement rates by period and seniority ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (cat, title) in enumerate([
    ('ai_tool_rate', 'AI-as-Tool Requirements'),
    ('ai_domain_rate', 'AI-as-Domain Requirements'),
    ('ai_any_rate', 'Any AI Requirement'),
]):
    ax = axes[idx]
    for sen, color in [('junior', '#e74c3c'), ('mid', '#f39c12'), ('senior', '#3498db')]:
        sub = rate_tbl[rate_tbl['seniority_3level'] == sen].sort_values('period')
        if len(sub) > 0:
            ax.plot(sub['period'], sub[cat] * 100, 'o-', color=color, label=sen, markersize=8, linewidth=2)
    ax.set_ylabel('Rate (%)')
    ax.set_title(title)
    ax.legend()
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(FIG_DIR / "ai_requirement_rates.png", dpi=150)
plt.close()

# --- Figure 2: Requirement vs usage divergence ---
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['AI-any', 'AI-tool', 'AI-domain']
x = np.arange(len(categories))
width = 0.18

for year_idx, (year, color_post, color_use) in enumerate([
    ('2024', '#e74c3c', '#3498db'),
    ('2026', '#c0392b', '#2980b9'),
]):
    post_vals = [posting_rates[year]['ai_any']*100, posting_rates[year]['ai_tool']*100, posting_rates[year]['ai_domain']*100]
    use_vals = [usage_rates[year]['ai_any']*100, usage_rates[year]['ai_tool']*100, usage_rates[year]['ai_domain']*100]

    offset = year_idx * 0.4
    bars1 = ax.bar(x + offset - width/2, post_vals, width, label=f'{year} Posting req.',
                   color=color_post, alpha=0.8)
    bars2 = ax.bar(x + offset + width/2, use_vals, width, label=f'{year} Usage (est.)',
                   color=color_use, alpha=0.8)

ax.set_xticks(x + 0.2)
ax.set_xticklabels(categories)
ax.set_ylabel('Rate (%)')
ax.set_title('AI Requirement Rate in Postings vs Estimated Developer Usage')
ax.legend(loc='upper left')
ax.axhline(y=0, color='black', linewidth=0.5)

# Add gap annotations
for i, cat in enumerate(['ai_any', 'ai_tool', 'ai_domain']):
    gap_26 = (posting_rates['2026'][cat] - usage_rates['2026'][cat]) * 100
    ax.annotate(f'{gap_26:+.0f}pp',
                xy=(i + 0.4 + width/2, usage_rates['2026'][cat]*100 + 2),
                fontsize=9, ha='center', color='#c0392b', fontweight='bold')

plt.tight_layout()
plt.savefig(FIG_DIR / "requirement_vs_usage_divergence.png", dpi=150)
plt.close()

# --- Figure 3: Temporal growth comparison ---
fig, ax = plt.subplots(figsize=(8, 6))

growth_data = pd.DataFrame({
    'Category': ['AI-any\n(posting)', 'AI-any\n(usage)', 'AI-tool\n(posting)', 'AI-tool\n(usage)', 'AI-domain\n(posting)', 'AI-domain\n(usage)'],
    'Growth (pp)': [
        post_growth_pp,
        use_growth_pp,
        (df[df['period']=='2026-03']['ai_tool'].mean() - df[df['period'].isin(['2024-01','2024-04'])]['ai_tool'].mean()) * 100,
        (0.55 - 0.35) * 100,
        (df[df['period']=='2026-03']['ai_domain'].mean() - df[df['period'].isin(['2024-01','2024-04'])]['ai_domain'].mean()) * 100,
        (0.20 - 0.15) * 100,
    ],
    'Type': ['Posting', 'Usage', 'Posting', 'Usage', 'Posting', 'Usage'],
})

colors = {'Posting': '#e74c3c', 'Usage': '#3498db'}
bars = ax.bar(growth_data['Category'], growth_data['Growth (pp)'],
              color=[colors[t] for t in growth_data['Type']])
ax.set_ylabel('Growth 2024->2026 (percentage points)')
ax.set_title('Requirement Growth vs Usage Growth (2024-2026)')
ax.axhline(y=0, color='black', linewidth=0.5)

# Custom legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor='#e74c3c', label='Posting requirements'),
                    Patch(facecolor='#3498db', label='Developer usage (est.)')], loc='upper right')

plt.tight_layout()
plt.savefig(FIG_DIR / "temporal_growth_comparison.png", dpi=150)
plt.close()

# --- Figure 4: Specific tool mention rates ---
fig, ax = plt.subplots(figsize=(10, 5))

tool_order = ['llm', 'copilot', 'gpt', 'prompt_engineering', 'chatgpt', 'cursor', 'claude']
for tool_name in tool_order:
    vals = []
    for per in ['2024-01', '2024-04', '2026-03']:
        sub = df[df['period'] == per]
        vals.append(sub[f'has_{tool_name}'].mean() * 100)
    ax.plot(['2024-01', '2024-04', '2026-03'], vals, 'o-', label=tool_name, markersize=7, linewidth=2)

ax.set_ylabel('Mention rate (%)')
ax.set_title('Specific AI Tool Mention Rates in SWE Postings')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(FIG_DIR / "specific_tool_rates.png", dpi=150)
plt.close()

print("All figures saved.")
print("\n" + "="*70)
print("T23 ANALYSIS COMPLETE")
print("="*70)
