#!/usr/bin/env python3
"""
T13: Linguistic & structural evolution of job postings.

CRITICAL OUTPUT: Section anatomy analysis — what drove the 57-67% description
length growth? Requirements sections (real signal) or boilerplate/benefits?

Steps:
1. Readability metrics (textstat) on both raw description and LLM-cleaned text
2. Section anatomy via regex classifier on raw description
3. Stacked bar: section composition by period (what grew?)
4. Tone markers per 1K chars
5. Entry-level vs mid-senior comparison
"""

import sys, os, warnings, re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import duckdb
import textstat

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
    'figure.figsize': (12, 8), 'savefig.bbox': 'tight'
})

BASE = Path('/home/jihgaboot/gabor/job-research')
FIG_DIR = BASE / 'exploration/figures/T13'
TAB_DIR = BASE / 'exploration/tables/T13'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
SHARED = BASE / 'exploration/artifacts/shared'

# ──────────────────────────────────────────────────────────────────────
# Section classifier patterns
# ──────────────────────────────────────────────────────────────────────
SECTION_PATTERNS = {
    'role_summary': [
        r'(?i)(?:^|\n)\s*(?:about\s+(?:the\s+)?(?:role|position|job|opportunity)|role\s+(?:summary|overview|description)|job\s+(?:summary|overview|description)|position\s+(?:summary|overview|description)|overview\s*[:;]|the\s+opportunity|who\s+we\s+are\s+looking\s+for)',
    ],
    'responsibilities': [
        r'(?i)(?:^|\n)\s*(?:responsibilities|what\s+you\'?ll?\s+(?:do|work\s+on|be\s+doing)|key\s+responsibilities|primary\s+responsibilities|your\s+(?:role|responsibilities|day\s+to\s+day)|in\s+this\s+role|the\s+role|day\s+to\s+day|what\s+you\s+will\s+do|duties)',
    ],
    'requirements': [
        r'(?i)(?:^|\n)\s*(?:requirements|qualifications|what\s+(?:you\'?ll?\s+)?(?:need|bring)|minimum\s+qualifications|basic\s+qualifications|required\s+(?:qualifications|skills|experience)|must\s+have|who\s+you\s+are|skills\s+(?:and|&)\s+experience|you\s+have)',
    ],
    'preferred': [
        r'(?i)(?:^|\n)\s*(?:preferred|nice\s+to\s+have|bonus|desirable|plus|additional\s+(?:qualifications|skills)|preferred\s+qualifications|it\'?s?\s+a\s+(?:plus|bonus)|extra\s+credit|preferred\s+skills|would\s+be\s+nice)',
    ],
    'benefits': [
        r'(?i)(?:^|\n)\s*(?:benefits|perks|compensation|what\s+we\s+offer|we\s+offer|our\s+benefits|total\s+rewards|pay\s+(?:range|transparency)|salary\s+(?:range|information)|the\s+(?:salary|pay|compensation)|(?:base\s+)?(?:salary|pay)\s+(?:for|range)|why\s+(?:you\'?ll?\s+love|join)|employee\s+benefits)',
    ],
    'about_company': [
        r'(?i)(?:^|\n)\s*(?:about\s+(?:us|the\s+company|our\s+(?:company|team))|who\s+we\s+are|our\s+(?:mission|story|culture|values)|company\s+(?:overview|description)|join\s+(?:us|our\s+team)|life\s+at)',
    ],
    'legal_eeo': [
        r'(?i)(?:^|\n)\s*(?:equal\s+(?:opportunity|employment)|eeo|diversity|inclusion|non.?discrimination|we\s+are\s+an?\s+equal|accommodation|americans?\s+with\s+disabilit|e.?verify|background\s+check|drug.?(?:free|test)|employment\s+eligibility)',
    ],
}


def classify_sections(text):
    """
    Classify a job description into sections by finding header patterns
    and computing character counts between them.
    Returns dict of {section_name: char_count}.
    """
    if not text or len(text) < 10:
        return {'unclassified': 0}

    # Find all section boundaries
    boundaries = []
    for section_name, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            for match in re.finditer(pat, text):
                boundaries.append((match.start(), section_name))

    if not boundaries:
        return {'unclassified': len(text)}

    # Sort by position
    boundaries.sort(key=lambda x: x[0])

    # Compute section sizes
    result = {}
    # Text before first header = unclassified intro
    if boundaries[0][0] > 0:
        result['unclassified'] = boundaries[0][0]

    for i, (pos, name) in enumerate(boundaries):
        if i + 1 < len(boundaries):
            end = boundaries[i + 1][0]
        else:
            end = len(text)
        chars = end - pos
        if name in result:
            result[name] += chars
        else:
            result[name] = chars

    return result


def compute_readability(text):
    """Compute readability metrics for a text. Returns dict."""
    if not text or len(text) < 100:
        return None
    try:
        # textstat needs enough text to be reliable
        return {
            'flesch_kincaid': textstat.flesch_kincaid_grade(text),
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'gunning_fog': textstat.gunning_fog(text),
            'avg_sentence_length': textstat.avg_sentence_length(text),
            'syllable_count': textstat.syllable_count(text),
            'lexicon_count': textstat.lexicon_count(text),
            'sentence_count': textstat.sentence_count(text),
        }
    except Exception:
        return None


def compute_tone_markers(text):
    """Compute tone markers per 1K chars. Returns dict."""
    if not text or len(text) < 100:
        return None

    n_chars = len(text)
    scale = 1000.0 / n_chars

    # Imperative density: verbs at start of bullet/sentence
    imperatives = len(re.findall(
        r'(?:^|\n)\s*[-•*]?\s*(?:build|create|design|develop|implement|lead|manage|write|ensure|maintain|drive|collaborate|work|deliver|optimize|define|monitor|establish|support|review|architect|deploy|test|debug|analyze|automate|evaluate|contribute|partner|mentor|own)',
        text, re.IGNORECASE
    ))

    # Inclusive language
    inclusive = len(re.findall(
        r'\b(?:diverse|diversity|inclusion|inclusive|belonging|equit|accessible|welcoming|regardless|all backgrounds|underrepresented)\b',
        text, re.IGNORECASE
    ))

    # Marketing language
    marketing = len(re.findall(
        r'\b(?:world.?class|cutting.?edge|game.?chang|disrupt|revolutioniz|transform|innovat|exciting|amazing|incredible|passionate|thriving|dynamic|fast.?paced|high.?impact|mission.?driven|best.?in.?class|industry.?leading|state.?of.?the.?art|next.?generation)\b',
        text, re.IGNORECASE
    ))

    # Formal markers
    formal = len(re.findall(
        r'\b(?:shall|pursuant|herein|thereof|hereby|notwithstanding|aforementioned|commensurate|requisite|proficienc|competenc|demonstrat\w+\s+ability|proven\s+track\s+record|aptitude)\b',
        text, re.IGNORECASE
    ))

    # Informal markers
    informal = len(re.findall(
        r'\b(?:you\'ll|we\'re|you\'re|we\'ll|awesome|cool|rockstar|ninja|guru|wizard|superstar|kick.?ass|hustle|swag|dope|crush it|killing it)\b',
        text, re.IGNORECASE
    ))

    # "AI-forward" language
    ai_forward = len(re.findall(
        r'\b(?:ai.?first|ai.?native|ai.?powered|ai.?driven|ai.?enabled|leverage\s+ai|leveraging\s+ai|ai\s+tool|copilot|chatgpt|github\s+copilot|cursor|generative\s+ai|gen\s+ai|large\s+language|llm|prompt\s+engineer|ai\s+agent|agentic)\b',
        text, re.IGNORECASE
    ))

    return {
        'imperative_per_1k': imperatives * scale,
        'inclusive_per_1k': inclusive * scale,
        'marketing_per_1k': marketing * scale,
        'formal_per_1k': formal * scale,
        'informal_per_1k': informal * scale,
        'ai_forward_per_1k': ai_forward * scale,
    }


# ──────────────────────────────────────────────────────────────────────
# Load data — we need both raw description and cleaned text
# ──────────────────────────────────────────────────────────────────────
print("Loading data from unified.parquet...")
con = duckdb.connect()

# Get SWE LinkedIn rows with both raw and cleaned descriptions
df = con.execute(f"""
    SELECT
        u.uid,
        u.description,
        u.description_core_llm,
        u.description_core,
        u.description_length,
        u.core_length,
        u.source,
        u.period,
        u.seniority_final,
        u.seniority_3level,
        u.is_aggregator,
        u.company_name_canonical,
        u.llm_extraction_coverage,
        c.text_source,
        c.description_cleaned
    FROM '{BASE}/data/unified.parquet' u
    LEFT JOIN '{SHARED}/swe_cleaned_text.parquet' c ON u.uid = c.uid
    WHERE u.source_platform = 'linkedin'
      AND u.is_english = true
      AND u.date_flag = 'ok'
      AND u.is_swe = true
""").fetchdf()

print(f"Loaded {len(df)} SWE LinkedIn rows")
print(f"Sources: {df.groupby('source').size().to_dict()}")

# ──────────────────────────────────────────────────────────────────────
# Step 1: Readability metrics
# ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 1: Readability metrics")
print("="*70)

# Sample for readability (textstat is slow on large corpora)
READABILITY_N = 2000
rng = np.random.RandomState(42)

readability_results = []

for source_name, source_df in df.groupby('source'):
    for sen_level in ['junior', 'mid', 'senior', 'unknown']:
        sub = source_df[source_df['seniority_3level'] == sen_level]
        if len(sub) == 0:
            continue
        sample = sub.sample(min(READABILITY_N, len(sub)), random_state=42)

        for _, row in sample.iterrows():
            # Run on raw description
            raw_r = compute_readability(row['description'])
            if raw_r:
                raw_r.update({
                    'uid': row['uid'],
                    'source': source_name,
                    'period': row['period'],
                    'seniority_3level': sen_level,
                    'text_type': 'raw_description',
                    'text_length': len(row['description']) if row['description'] else 0,
                })
                readability_results.append(raw_r)

            # Run on cleaned text (LLM-cleaned where available, else description_core)
            cleaned = row.get('description_cleaned')
            if cleaned and len(str(cleaned)) > 100:
                cl_r = compute_readability(cleaned)
                if cl_r:
                    cl_r.update({
                        'uid': row['uid'],
                        'source': source_name,
                        'period': row['period'],
                        'seniority_3level': sen_level,
                        'text_type': 'cleaned',
                        'text_length': len(cleaned),
                    })
                    readability_results.append(cl_r)

    print(f"  Processed {source_name}")

read_df = pd.DataFrame(readability_results)
print(f"Readability records: {len(read_df)}")

# Summary table
readability_summary = read_df.groupby(['period', 'seniority_3level', 'text_type']).agg({
    'flesch_kincaid': ['mean', 'median', 'std'],
    'flesch_reading_ease': ['mean', 'median'],
    'gunning_fog': ['mean', 'median'],
    'avg_sentence_length': ['mean', 'median'],
    'text_length': ['mean', 'median', 'count'],
}).round(2)

print("\nReadability by period x seniority (raw description):")
raw_summary = read_df[read_df['text_type'] == 'raw_description'].groupby(['period', 'seniority_3level']).agg({
    'flesch_kincaid': 'median',
    'flesch_reading_ease': 'median',
    'gunning_fog': 'median',
    'avg_sentence_length': 'median',
    'text_length': ['median', 'count'],
}).round(1)
print(raw_summary.to_string())

print("\nReadability by period x seniority (cleaned text):")
clean_summary = read_df[read_df['text_type'] == 'cleaned'].groupby(['period', 'seniority_3level']).agg({
    'flesch_kincaid': 'median',
    'flesch_reading_ease': 'median',
    'gunning_fog': 'median',
    'avg_sentence_length': 'median',
    'text_length': ['median', 'count'],
}).round(1)
print(clean_summary.to_string())

read_df.to_csv(TAB_DIR / 'readability_detailed.csv', index=False)
readability_summary.to_csv(TAB_DIR / 'readability_summary.csv')

# ──────────────────────────────────────────────────────────────────────
# Step 2: Section anatomy (run on RAW description)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 2: Section anatomy classification")
print("="*70)

section_results = []
for idx, row in df.iterrows():
    desc = row['description']
    if not desc or len(str(desc)) < 50:
        continue
    sections = classify_sections(str(desc))
    total = sum(sections.values())
    if total == 0:
        continue

    for sec_name, chars in sections.items():
        section_results.append({
            'uid': row['uid'],
            'source': row['source'],
            'period': row['period'],
            'seniority_3level': row['seniority_3level'],
            'is_aggregator': row['is_aggregator'],
            'section': sec_name,
            'chars': chars,
            'pct': chars / total,
            'total_length': total,
        })

    if idx % 10000 == 0:
        print(f"  Processed {idx}/{len(df)} rows...")

sec_df = pd.DataFrame(section_results)
print(f"Section records: {len(sec_df)}")

# Pivot: median pct by section x period x seniority
section_pivot = sec_df.groupby(['period', 'seniority_3level', 'section']).agg(
    median_pct=('pct', 'median'),
    mean_pct=('pct', 'mean'),
    median_chars=('chars', 'median'),
    mean_chars=('chars', 'mean'),
    n_postings=('uid', 'nunique'),
).reset_index()

# Also compute section prevalence (what fraction of postings have each section)
n_postings_total = sec_df.groupby(['period', 'seniority_3level'])['uid'].nunique().reset_index()
n_postings_total.columns = ['period', 'seniority_3level', 'total_postings']

section_prevalence = sec_df[sec_df['chars'] > 0].groupby(['period', 'seniority_3level', 'section'])['uid'].nunique().reset_index()
section_prevalence.columns = ['period', 'seniority_3level', 'section', 'n_with_section']
section_prevalence = section_prevalence.merge(n_postings_total, on=['period', 'seniority_3level'])
section_prevalence['prevalence'] = section_prevalence['n_with_section'] / section_prevalence['total_postings']

print("\n--- Section prevalence by period (all seniority) ---")
prev_all = sec_df[sec_df['chars'] > 0].groupby(['period', 'section'])['uid'].nunique().reset_index()
prev_all.columns = ['period', 'section', 'n_with']
n_all = sec_df.groupby('period')['uid'].nunique().reset_index()
n_all.columns = ['period', 'total']
prev_all = prev_all.merge(n_all, on='period')
prev_all['prevalence'] = (prev_all['n_with'] / prev_all['total'] * 100).round(1)
prev_pivot = prev_all.pivot(index='section', columns='period', values='prevalence').fillna(0)
print(prev_pivot.to_string())

# ──────────────────────────────────────────────────────────────────────
# Step 3: What's driving the length growth?
# ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 3: What's driving the length growth?")
print("="*70)

# Median section chars by period (all SWE, all seniority)
section_by_period = sec_df.groupby(['period', 'section']).agg(
    median_chars=('chars', 'median'),
    mean_chars=('chars', 'mean'),
    total_chars=('chars', 'sum'),
    n=('uid', 'nunique')
).reset_index()

# Also compute per-posting mean section chars (sum chars per section / n postings)
agg_by_period = sec_df.groupby(['period', 'section']).agg(
    total_chars=('chars', 'sum'),
).reset_index()
n_postings_per_period = sec_df.groupby('period')['uid'].nunique().reset_index()
n_postings_per_period.columns = ['period', 'n_postings']
agg_by_period = agg_by_period.merge(n_postings_per_period, on='period')
agg_by_period['avg_chars_per_posting'] = agg_by_period['total_chars'] / agg_by_period['n_postings']

print("\nAvg chars per posting by section and period:")
chars_pivot = agg_by_period.pivot(index='section', columns='period', values='avg_chars_per_posting').fillna(0).round(0)
if '2024-01' in chars_pivot.columns and '2026-03' in chars_pivot.columns:
    chars_pivot['change_01_to_26'] = chars_pivot['2026-03'] - chars_pivot['2024-01']
    chars_pivot['pct_change_01_to_26'] = ((chars_pivot['2026-03'] / chars_pivot['2024-01'].clip(1) - 1) * 100).round(1)
if '2024-04' in chars_pivot.columns and '2026-03' in chars_pivot.columns:
    chars_pivot['change_04_to_26'] = chars_pivot['2026-03'] - chars_pivot['2024-04']
    chars_pivot['pct_change_04_to_26'] = ((chars_pivot['2026-03'] / chars_pivot['2024-04'].clip(1) - 1) * 100).round(1)
print(chars_pivot.to_string())

# Total length growth attribution
if '2024-04' in chars_pivot.columns and '2026-03' in chars_pivot.columns:
    total_growth = chars_pivot['change_04_to_26'].sum()
    print(f"\nTotal avg length growth (arshkon→scraped): {total_growth:.0f} chars")
    print("Growth attribution by section:")
    for sec in chars_pivot.index:
        contrib = chars_pivot.loc[sec, 'change_04_to_26']
        pct_of_total = contrib / total_growth * 100 if total_growth != 0 else 0
        print(f"  {sec:20s}: {contrib:+7.0f} chars ({pct_of_total:+5.1f}% of total growth)")

# Save section results
section_by_period.to_csv(TAB_DIR / 'section_anatomy_by_period.csv', index=False)
section_pivot.to_csv(TAB_DIR / 'section_anatomy_pivot.csv')
section_prevalence.to_csv(TAB_DIR / 'section_prevalence.csv', index=False)

# ──────────────────────────────────────────────────────────────────────
# Step 4: Tone markers
# ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 4: Tone markers")
print("="*70)

tone_results = []
for _, row in df.iterrows():
    desc = row['description']
    if not desc or len(str(desc)) < 100:
        continue
    tone = compute_tone_markers(str(desc))
    if tone:
        tone.update({
            'uid': row['uid'],
            'source': row['source'],
            'period': row['period'],
            'seniority_3level': row['seniority_3level'],
            'is_aggregator': row['is_aggregator'],
            'description_length': len(str(desc)),
        })
        tone_results.append(tone)

tone_df = pd.DataFrame(tone_results)
print(f"Tone records: {len(tone_df)}")

tone_summary = tone_df.groupby(['period', 'seniority_3level']).agg({
    'imperative_per_1k': ['mean', 'median'],
    'inclusive_per_1k': ['mean', 'median'],
    'marketing_per_1k': ['mean', 'median'],
    'formal_per_1k': ['mean', 'median'],
    'informal_per_1k': ['mean', 'median'],
    'ai_forward_per_1k': ['mean', 'median'],
    'uid': 'count',
}).round(3)

print("\nTone markers by period x seniority (median per 1K chars):")
tone_med = tone_df.groupby(['period', 'seniority_3level']).agg({
    'imperative_per_1k': 'median',
    'inclusive_per_1k': 'median',
    'marketing_per_1k': 'median',
    'formal_per_1k': 'median',
    'informal_per_1k': 'median',
    'ai_forward_per_1k': 'median',
}).round(3)
print(tone_med.to_string())

tone_df.to_csv(TAB_DIR / 'tone_markers_detailed.csv', index=False)
tone_summary.to_csv(TAB_DIR / 'tone_markers_summary.csv')

# ──────────────────────────────────────────────────────────────────────
# Step 5: Entry-level specifically
# ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 5: Entry-level specific analysis")
print("="*70)

for metric_name, metric_df, agg_col in [
    ('readability', read_df[read_df['text_type'] == 'raw_description'], 'flesch_kincaid'),
    ('tone_imperative', tone_df, 'imperative_per_1k'),
    ('tone_ai_forward', tone_df, 'ai_forward_per_1k'),
    ('tone_marketing', tone_df, 'marketing_per_1k'),
    ('tone_inclusive', tone_df, 'inclusive_per_1k'),
]:
    entry = metric_df[metric_df['seniority_3level'] == 'junior']
    senior = metric_df[metric_df['seniority_3level'] == 'senior']

    print(f"\n{metric_name} ({agg_col}):")
    for level_name, level_df in [('entry', entry), ('senior', senior)]:
        for period in ['2024-01', '2024-04', '2026-03']:
            sub = level_df[level_df['period'] == period]
            if len(sub) > 0:
                print(f"  {level_name:8s} {period}: median={sub[agg_col].median():.2f}, n={len(sub)}")

# ──────────────────────────────────────────────────────────────────────
# SENSITIVITY: Aggregator exclusion
# ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SENSITIVITY: Aggregator exclusion — section anatomy")
print("="*70)

sec_noagg = sec_df[~sec_df['is_aggregator']]
agg_noagg = sec_noagg.groupby(['period', 'section']).agg(
    total_chars=('chars', 'sum'),
).reset_index()
n_noagg = sec_noagg.groupby('period')['uid'].nunique().reset_index()
n_noagg.columns = ['period', 'n_postings']
agg_noagg = agg_noagg.merge(n_noagg, on='period')
agg_noagg['avg_chars_per_posting'] = agg_noagg['total_chars'] / agg_noagg['n_postings']

chars_noagg = agg_noagg.pivot(index='section', columns='period', values='avg_chars_per_posting').fillna(0).round(0)
if '2024-04' in chars_noagg.columns and '2026-03' in chars_noagg.columns:
    chars_noagg['change_04_to_26'] = chars_noagg['2026-03'] - chars_noagg['2024-04']
    total_noagg = chars_noagg['change_04_to_26'].sum()
    print(f"Total growth (no agg): {total_noagg:.0f}")
    print("Attribution:")
    for sec in chars_noagg.index:
        contrib = chars_noagg.loc[sec, 'change_04_to_26']
        pct = contrib / total_noagg * 100 if total_noagg != 0 else 0
        print(f"  {sec:20s}: {contrib:+7.0f} chars ({pct:+5.1f}% of total growth)")

chars_noagg.to_csv(TAB_DIR / 'section_anatomy_noagg.csv')

# ──────────────────────────────────────────────────────────────────────
# FIGURES
# ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("Generating figures...")
print("="*70)

# Figure 1: Stacked bar — section composition by period (the critical figure)
all_sections = ['role_summary', 'responsibilities', 'requirements', 'preferred',
                'benefits', 'about_company', 'legal_eeo', 'unclassified']
section_colors = {
    'role_summary': '#2196F3',
    'responsibilities': '#4CAF50',
    'requirements': '#FF9800',
    'preferred': '#FFC107',
    'benefits': '#E91E63',
    'about_company': '#9C27B0',
    'legal_eeo': '#795548',
    'unclassified': '#BDBDBD',
}

fig, axes = plt.subplots(1, 3, figsize=(16, 7))

for ax_idx, (period, period_label) in enumerate([
    ('2024-01', 'asaniczka\n(Jan 2024)'),
    ('2024-04', 'arshkon\n(Apr 2024)'),
    ('2026-03', 'scraped\n(Mar 2026)')
]):
    ax = axes[ax_idx]
    period_data = agg_by_period[agg_by_period['period'] == period]

    total_avg = period_data['avg_chars_per_posting'].sum()
    bottom = 0

    for sec in all_sections:
        sec_data = period_data[period_data['section'] == sec]
        if len(sec_data) > 0:
            val = sec_data['avg_chars_per_posting'].values[0]
        else:
            val = 0
        ax.bar(0, val, bottom=bottom, color=section_colors.get(sec, '#999'),
               label=sec if ax_idx == 0 else None, width=0.6, edgecolor='white', linewidth=0.5)
        if val > total_avg * 0.04:  # label if >4%
            ax.text(0, bottom + val/2, f'{val:.0f}', ha='center', va='center', fontsize=8)
        bottom += val

    ax.set_title(f'{period_label}\n(avg {total_avg:.0f} chars)', fontsize=11)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    if ax_idx == 0:
        ax.set_ylabel('Average chars per posting')

fig.legend(all_sections, loc='lower center', ncol=4, fontsize=9, framealpha=0.9)
plt.suptitle('Job Description Section Anatomy by Period\n(SWE, LinkedIn, all seniority)', fontsize=13, y=1.02)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(FIG_DIR / 'section_anatomy_stacked.png')
plt.close()
print("  Saved section_anatomy_stacked.png")

# Figure 2: Section growth waterfall (arshkon -> scraped)
if '2024-04' in chars_pivot.columns and '2026-03' in chars_pivot.columns:
    fig, ax = plt.subplots(figsize=(12, 7))

    growth_data = chars_pivot[['change_04_to_26']].copy()
    growth_data = growth_data.sort_values('change_04_to_26', ascending=False)

    colors_bar = []
    for sec in growth_data.index:
        if sec in ['requirements', 'responsibilities', 'preferred', 'role_summary']:
            colors_bar.append('#4CAF50')  # green = requirements-like
        elif sec in ['benefits', 'about_company', 'legal_eeo']:
            colors_bar.append('#E91E63')  # red = boilerplate
        else:
            colors_bar.append('#BDBDBD')  # gray

    bars = ax.barh(range(len(growth_data)), growth_data['change_04_to_26'].values,
                   color=colors_bar, alpha=0.8, edgecolor='white')

    ax.set_yticks(range(len(growth_data)))
    ax.set_yticklabels(growth_data.index, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('Change in avg chars per posting (arshkon 2024 -> scraped 2026)')
    ax.set_title('What Drove the Description Length Growth?\n(Green = requirements/role content, Red = boilerplate/benefits)', fontsize=12)

    for i, v in enumerate(growth_data['change_04_to_26'].values):
        ax.text(v + (10 if v >= 0 else -10), i, f'{v:+.0f}',
                ha='left' if v >= 0 else 'right', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'section_growth_waterfall.png')
    plt.close()
    print("  Saved section_growth_waterfall.png")

# Figure 3: Readability by period x seniority
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = [('flesch_kincaid', 'Flesch-Kincaid Grade Level'),
           ('flesch_reading_ease', 'Flesch Reading Ease'),
           ('gunning_fog', 'Gunning Fog Index'),
           ('avg_sentence_length', 'Avg Sentence Length')]

raw_read = read_df[read_df['text_type'] == 'raw_description'].copy()

for ax, (metric, title) in zip(axes.flat, metrics):
    for sen, color, marker in [('junior', '#e41a1c', 'o'), ('senior', '#377eb8', 's')]:
        sub = raw_read[raw_read['seniority_3level'] == sen]
        if len(sub) == 0:
            continue
        medians = sub.groupby('period')[metric].median()
        counts = sub.groupby('period')[metric].count()
        periods_ordered = ['2024-01', '2024-04', '2026-03']
        x_vals = [i for i, p in enumerate(periods_ordered) if p in medians.index]
        y_vals = [medians[p] for p in periods_ordered if p in medians.index]
        labels = [f'n={counts[p]}' for p in periods_ordered if p in medians.index]
        ax.plot(x_vals, y_vals, marker=marker, color=color, label=sen, linewidth=2, markersize=8)
        for x, y, lab in zip(x_vals, y_vals, labels):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7, color=color)

    ax.set_xticks(range(3))
    ax.set_xticklabels(['Jan 2024', 'Apr 2024', 'Mar 2026'], fontsize=9)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('Readability Metrics by Period and Seniority\n(SWE, LinkedIn, raw description)', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'readability_trends.png')
plt.close()
print("  Saved readability_trends.png")

# Figure 4: Tone markers by period
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
tone_metrics = [
    ('imperative_per_1k', 'Imperative Density'),
    ('inclusive_per_1k', 'Inclusive Language'),
    ('marketing_per_1k', 'Marketing Language'),
    ('formal_per_1k', 'Formal Language'),
    ('informal_per_1k', 'Informal Language'),
    ('ai_forward_per_1k', 'AI-Forward Language'),
]

for ax, (metric, title) in zip(axes.flat, tone_metrics):
    for sen, color in [('junior', '#e41a1c'), ('senior', '#377eb8')]:
        sub = tone_df[tone_df['seniority_3level'] == sen]
        medians = sub.groupby('period')[metric].median()
        periods_ordered = ['2024-01', '2024-04', '2026-03']
        x_vals = [i for i, p in enumerate(periods_ordered) if p in medians.index]
        y_vals = [medians[p] for p in periods_ordered if p in medians.index]
        ax.plot(x_vals, y_vals, marker='o', color=color, label=sen, linewidth=2, markersize=8)

    ax.set_xticks(range(3))
    ax.set_xticklabels(['Jan 24', 'Apr 24', 'Mar 26'], fontsize=9)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Tone Marker Trends by Period and Seniority\n(per 1K chars, SWE, LinkedIn)', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'tone_markers_trends.png')
plt.close()
print("  Saved tone_markers_trends.png")

print("\n" + "="*70)
print("T13 COMPLETE — Tables and figures saved.")
print("="*70)
