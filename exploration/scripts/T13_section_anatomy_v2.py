#!/usr/bin/env python3
"""
T13 Section Anatomy V2: Improved section classifier that handles both
markdown-formatted and plain-text section headers.

This is the critical analysis showing what drove the 57-67% length growth.
"""

import sys, os, warnings, re
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('/home/jihgaboot/gabor/job-research')
FIG_DIR = BASE / 'exploration/figures/T13'
TAB_DIR = BASE / 'exploration/tables/T13'

# ──────────────────────────────────────────────────────────────────────
# Improved section classifier — handles markdown **headers** and plain text
# ──────────────────────────────────────────────────────────────────────
# Map header text -> section category
HEADER_MAP = {
    'role_summary': [
        'about the role', 'about this role', 'the role', 'role overview',
        'role summary', 'position overview', 'position summary', 'the opportunity',
        'about the position', 'about the job', 'about this position',
        'job summary', 'opportunity', 'the position', 'role description',
        'about this opportunity', 'job overview', 'summary', 'overview',
        'who we need', 'what is the opportunity', 'your opportunity',
    ],
    'responsibilities': [
        'responsibilities', 'key responsibilities', 'job responsibilities',
        'what you\'ll do', 'what you will do', 'what you\'ll be doing',
        'your responsibilities', 'primary responsibilities', 'duties',
        'in this role', 'day to day', 'the impact you will make',
        'what you\'ll work on', 'how you\'ll make an impact',
        'core responsibilities', 'essential functions', 'job duties',
        'what you will be doing', 'as a .* you will', 'your day to day',
        'what you\'ll do:', 'what you\'ll do',
    ],
    'requirements': [
        'qualifications', 'requirements', 'required qualifications',
        'basic qualifications', 'minimum qualifications', 'what you\'ll need',
        'what we\'re looking for', 'what you need', 'who you are',
        'what you\'ll bring', 'skills and experience', 'what we look for',
        'required skills', 'required experience', 'must have',
        'what you bring', 'your background', 'we\'re looking for',
        'skills & experience', 'education and experience',
        'minimum qualifications:', 'basic qualifications:',
        'required qualifications:', 'qualifications:', 'requirements:',
        'you have', 'what you have',
    ],
    'preferred': [
        'preferred qualifications', 'nice to have', 'bonus', 'preferred',
        'preferred qualifications:', 'nice to haves', 'additional qualifications',
        'desired qualifications', 'it would be a plus', 'preferred skills',
        'bonus qualifications', 'pluses', 'a plus',
    ],
    'benefits': [
        'benefits', 'what we offer', 'perks', 'compensation',
        'total rewards', 'pay range', 'salary range', 'benefits and perks',
        'our benefits', 'compensation and benefits', 'why join us',
        'why you\'ll love working here', 'we offer', 'benefits:',
        'employee benefits', 'benefits package', 'pay transparency',
        'compensation:', 'more than just important work',
    ],
    'about_company': [
        'about us', 'about the company', 'who we are', 'our mission',
        'about the team', 'about', 'company', 'our story', 'our culture',
        'about .*', 'the company', 'our values', 'meet the team',
        'who are we', 'life at', 'join us', 'about our team',
        'company overview', 'about the organization',
    ],
    'legal_eeo': [
        'equal opportunity employer', 'eeo', 'disclaimer', 'legal',
        'eeo statement', 'equal opportunity', 'diversity', 'accessibility',
        'additional information', 'notice', 'privacy', 'non-discrimination',
        'accommodation', 'e-verify',
    ],
    'description': [
        'description', 'job description', 'job description:',
    ],
}

# Flatten for lookup
HEADER_LOOKUP = {}
for section, headers in HEADER_MAP.items():
    for h in headers:
        HEADER_LOOKUP[h.lower().strip().rstrip(':')] = section


def match_header(text):
    """Match a header text to a section category."""
    t = text.lower().strip().rstrip(':').strip()
    # Direct lookup
    if t in HEADER_LOOKUP:
        return HEADER_LOOKUP[t]
    # Fuzzy match for "about [company name]" -> about_company
    if t.startswith('about ') and len(t) > 10:
        # If it's "about the role/position/job" -> already caught
        # If it's "about [something else]" -> about_company
        rest = t[6:]
        if rest not in ['the role', 'this role', 'the position', 'this position',
                        'the job', 'this job', 'the opportunity', 'this opportunity']:
            return 'about_company'
    return None


def classify_sections_v2(text):
    """
    Improved section classifier that handles both markdown and plain text headers.
    Returns dict of {section_name: char_count}.
    """
    if not text or len(text) < 10:
        return {'unclassified': 0}

    # Find all section boundaries from markdown headers: **Header** or **Header:**
    boundaries = []

    # Markdown headers: **text** at start of line or after whitespace
    for m in re.finditer(r'\*\*([^*]+)\*\*', text):
        header_text = m.group(1).strip()
        if len(header_text) < 80:  # Skip long "bold" text that's not a header
            section = match_header(header_text)
            if section:
                boundaries.append((m.start(), section, header_text))

    # Plain text headers: "Header:" or "Header" at start of line, Title Case, followed by newline
    for m in re.finditer(r'(?:^|\n)\s*([A-Z][A-Za-z\s/&\']+?)(?::?\s*(?:\n|$))', text):
        header_text = m.group(1).strip()
        if len(header_text) < 60:
            section = match_header(header_text)
            if section:
                # Don't add if we already have a markdown header at a nearby position
                pos = m.start()
                if not any(abs(pos - bp) < 10 for bp, _, _ in boundaries):
                    boundaries.append((pos, section, header_text))

    if not boundaries:
        return {'unclassified': len(text)}

    # Sort by position
    boundaries.sort(key=lambda x: x[0])

    # Compute section sizes
    result = {}

    # Text before first header = unclassified intro (often role summary without header)
    if boundaries[0][0] > 20:
        # Heuristic: if the intro is short (<200 chars), it might be metadata
        intro_len = boundaries[0][0]
        if intro_len > 200:
            # Check if it reads like a role summary (has sentences, not just metadata)
            intro_text = text[:boundaries[0][0]]
            if re.search(r'[.!?]\s', intro_text):
                result['role_summary'] = intro_len
            else:
                result['unclassified'] = intro_len
        else:
            result['unclassified'] = intro_len

    for i, (pos, name, _) in enumerate(boundaries):
        if i + 1 < len(boundaries):
            end = boundaries[i + 1][0]
        else:
            end = len(text)
        chars = end - pos

        # The 'description' section header is often used as a container
        # that contains the actual role summary/responsibilities
        if name == 'description':
            name = 'role_summary'

        if name in result:
            result[name] += chars
        else:
            result[name] = chars

    return result


# ──────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────
print("Loading data...")
con = duckdb.connect()
df = con.execute(f"""
    SELECT uid, description, source, period, seniority_3level, is_aggregator,
           description_length
    FROM '{BASE}/data/unified.parquet'
    WHERE source_platform = 'linkedin'
      AND is_english = true
      AND date_flag = 'ok'
      AND is_swe = true
""").fetchdf()

print(f"Loaded {len(df)} SWE LinkedIn rows")

# ──────────────────────────────────────────────────────────────────────
# Run improved section classifier
# ──────────────────────────────────────────────────────────────────────
print("Classifying sections with V2 classifier...")

section_results = []
for idx, row in df.iterrows():
    desc = row['description']
    if not desc or len(str(desc)) < 50:
        continue
    sections = classify_sections_v2(str(desc))
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

# ──────────────────────────────────────────────────────────────────────
# Section prevalence
# ──────────────────────────────────────────────────────────────────────
print("\n--- Section prevalence by period (V2 classifier) ---")
prev_all = sec_df[sec_df['chars'] > 0].groupby(['period', 'section'])['uid'].nunique().reset_index()
prev_all.columns = ['period', 'section', 'n_with']
n_all = sec_df.groupby('period')['uid'].nunique().reset_index()
n_all.columns = ['period', 'total']
prev_all = prev_all.merge(n_all, on='period')
prev_all['prevalence'] = (prev_all['n_with'] / prev_all['total'] * 100).round(1)
prev_pivot = prev_all.pivot(index='section', columns='period', values='prevalence').fillna(0)
print(prev_pivot.to_string())

# ──────────────────────────────────────────────────────────────────────
# Avg chars per posting by section and period
# ──────────────────────────────────────────────────────────────────────
print("\n--- Avg chars per posting by section and period ---")
agg_by_period = sec_df.groupby(['period', 'section']).agg(
    total_chars=('chars', 'sum'),
).reset_index()
n_postings_per_period = sec_df.groupby('period')['uid'].nunique().reset_index()
n_postings_per_period.columns = ['period', 'n_postings']
agg_by_period = agg_by_period.merge(n_postings_per_period, on='period')
agg_by_period['avg_chars_per_posting'] = agg_by_period['total_chars'] / agg_by_period['n_postings']

chars_pivot = agg_by_period.pivot(index='section', columns='period', values='avg_chars_per_posting').fillna(0).round(0)

for base_period, label in [('2024-01', '01_to_26'), ('2024-04', '04_to_26')]:
    if base_period in chars_pivot.columns and '2026-03' in chars_pivot.columns:
        chars_pivot[f'change_{label}'] = chars_pivot['2026-03'] - chars_pivot[base_period]
        chars_pivot[f'pct_change_{label}'] = ((chars_pivot['2026-03'] / chars_pivot[base_period].clip(1) - 1) * 100).round(1)

print(chars_pivot.to_string())

# Growth attribution
if '2024-04' in chars_pivot.columns and '2026-03' in chars_pivot.columns:
    total_growth = chars_pivot['change_04_to_26'].sum()
    print(f"\nTotal avg length growth (arshkon→scraped): {total_growth:.0f} chars")
    print("Growth attribution by section:")
    for sec in chars_pivot.sort_values('change_04_to_26', ascending=False).index:
        contrib = chars_pivot.loc[sec, 'change_04_to_26']
        pct_of_total = contrib / total_growth * 100 if total_growth != 0 else 0
        print(f"  {sec:20s}: {contrib:+7.0f} chars ({pct_of_total:+5.1f}% of total growth)")

# ──────────────────────────────────────────────────────────────────────
# By seniority
# ──────────────────────────────────────────────────────────────────────
print("\n--- Section anatomy by period x seniority ---")
for seniority in ['junior', 'senior']:
    sec_sub = sec_df[sec_df['seniority_3level'] == seniority]
    agg_sub = sec_sub.groupby(['period', 'section']).agg(total_chars=('chars', 'sum')).reset_index()
    n_sub = sec_sub.groupby('period')['uid'].nunique().reset_index()
    n_sub.columns = ['period', 'n_postings']
    agg_sub = agg_sub.merge(n_sub, on='period')
    agg_sub['avg_chars'] = agg_sub['total_chars'] / agg_sub['n_postings']
    pivot_sub = agg_sub.pivot(index='section', columns='period', values='avg_chars').fillna(0).round(0)
    if '2024-04' in pivot_sub.columns and '2026-03' in pivot_sub.columns:
        pivot_sub['change'] = pivot_sub['2026-03'] - pivot_sub['2024-04']
    print(f"\n  {seniority.upper()}:")
    print(pivot_sub.to_string())

# ──────────────────────────────────────────────────────────────────────
# Sensitivity: No aggregators
# ──────────────────────────────────────────────────────────────────────
print("\n--- SENSITIVITY: No aggregators ---")
sec_noagg = sec_df[~sec_df['is_aggregator']]
agg_noagg = sec_noagg.groupby(['period', 'section']).agg(total_chars=('chars', 'sum')).reset_index()
n_noagg = sec_noagg.groupby('period')['uid'].nunique().reset_index()
n_noagg.columns = ['period', 'n_postings']
agg_noagg = agg_noagg.merge(n_noagg, on='period')
agg_noagg['avg_chars'] = agg_noagg['total_chars'] / agg_noagg['n_postings']
chars_noagg = agg_noagg.pivot(index='section', columns='period', values='avg_chars').fillna(0).round(0)
if '2024-04' in chars_noagg.columns and '2026-03' in chars_noagg.columns:
    chars_noagg['change'] = chars_noagg['2026-03'] - chars_noagg['2024-04']
    total_noagg = chars_noagg['change'].sum()
    print(f"Total growth (no agg): {total_noagg:.0f}")
    for sec in chars_noagg.sort_values('change', ascending=False).index:
        contrib = chars_noagg.loc[sec, 'change']
        pct = contrib / total_noagg * 100 if total_noagg != 0 else 0
        print(f"  {sec:20s}: {contrib:+7.0f} chars ({pct:+5.1f}%)")

# ──────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────
sec_df.to_csv(TAB_DIR / 'section_anatomy_v2_detailed.csv', index=False)
chars_pivot.to_csv(TAB_DIR / 'section_anatomy_v2_pivot.csv')
prev_pivot.to_csv(TAB_DIR / 'section_prevalence_v2.csv')

# ──────────────────────────────────────────────────────────────────────
# FIGURES
# ──────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")

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

section_labels = {
    'role_summary': 'Role Summary / About Role',
    'responsibilities': 'Responsibilities',
    'requirements': 'Requirements / Qualifications',
    'preferred': 'Preferred / Nice-to-have',
    'benefits': 'Benefits / Compensation',
    'about_company': 'About the Company',
    'legal_eeo': 'Legal / EEO',
    'unclassified': 'Unclassified',
}

# Figure 1: Stacked bar — section composition by period
fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)

period_info = [
    ('2024-01', 'asaniczka\n(Jan 2024)'),
    ('2024-04', 'arshkon\n(Apr 2024)'),
    ('2026-03', 'scraped\n(Mar 2026)')
]

totals = {}
for ax_idx, (period, period_label) in enumerate(period_info):
    ax = axes[ax_idx]
    period_data = agg_by_period[agg_by_period['period'] == period]
    n_post = n_postings_per_period[n_postings_per_period['period'] == period]['n_postings'].values[0]

    total_avg = period_data['avg_chars_per_posting'].sum()
    totals[period] = total_avg
    bottom = 0

    for sec in all_sections:
        sec_data = period_data[period_data['section'] == sec]
        if len(sec_data) > 0:
            val = sec_data['avg_chars_per_posting'].values[0]
        else:
            val = 0
        ax.bar(0, val, bottom=bottom, color=section_colors.get(sec, '#999'),
               label=section_labels.get(sec, sec) if ax_idx == 0 else None,
               width=0.6, edgecolor='white', linewidth=0.5)
        if val > total_avg * 0.03:
            ax.text(0, bottom + val/2, f'{val:.0f}\n({val/total_avg*100:.0f}%)',
                    ha='center', va='center', fontsize=7, fontweight='bold')
        bottom += val

    ax.set_title(f'{period_label}\nn={n_post}, avg={total_avg:.0f} chars', fontsize=11)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])

axes[0].set_ylabel('Average chars per posting')

fig.legend(loc='lower center', ncol=4, fontsize=9, framealpha=0.9)
plt.suptitle('Job Description Section Anatomy by Period (V2 Classifier)\nSWE, LinkedIn, all seniority', fontsize=13, y=1.02)
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig(FIG_DIR / 'section_anatomy_v2_stacked.png')
plt.close()
print("  Saved section_anatomy_v2_stacked.png")

# Figure 2: Growth waterfall
if '2024-04' in chars_pivot.columns and '2026-03' in chars_pivot.columns:
    fig, ax = plt.subplots(figsize=(12, 7))

    growth = chars_pivot[['change_04_to_26']].dropna().copy()
    growth = growth.sort_values('change_04_to_26', ascending=False)

    colors_bar = []
    for sec in growth.index:
        if sec in ['requirements', 'responsibilities', 'preferred', 'role_summary']:
            colors_bar.append('#4CAF50')
        elif sec in ['benefits', 'about_company', 'legal_eeo']:
            colors_bar.append('#E91E63')
        else:
            colors_bar.append('#BDBDBD')

    bars = ax.barh(range(len(growth)), growth['change_04_to_26'].values,
                   color=colors_bar, alpha=0.8, edgecolor='white')

    # Labels
    for i, (sec, row_val) in enumerate(growth.iterrows()):
        v = row_val['change_04_to_26']
        total_g = chars_pivot['change_04_to_26'].sum()
        pct = v / total_g * 100 if total_g else 0
        label = f'{v:+.0f} ({pct:+.0f}%)'
        ax.text(v + (30 if v >= 0 else -30), i, label,
                ha='left' if v >= 0 else 'right', va='center', fontsize=10)

    ax.set_yticks(range(len(growth)))
    ax.set_yticklabels([section_labels.get(s, s) for s in growth.index], fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('Change in avg chars per posting (arshkon 2024 -> scraped 2026)')
    ax.set_title('What Drove the Description Length Growth? (V2 Classifier)\nGreen = requirements/role content | Red = boilerplate/benefits | Gray = unclassified',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'section_growth_waterfall_v2.png')
    plt.close()
    print("  Saved section_growth_waterfall_v2.png")

# Figure 3: Section anatomy by seniority x period (2x2)
fig, axes = plt.subplots(2, 3, figsize=(16, 12), sharey='row')

for row_idx, seniority in enumerate(['junior', 'senior']):
    sec_sub = sec_df[sec_df['seniority_3level'] == seniority]
    agg_sub = sec_sub.groupby(['period', 'section']).agg(total_chars=('chars', 'sum')).reset_index()
    n_sub = sec_sub.groupby('period')['uid'].nunique().reset_index()
    n_sub.columns = ['period', 'n_postings']
    agg_sub = agg_sub.merge(n_sub, on='period')
    agg_sub['avg_chars'] = agg_sub['total_chars'] / agg_sub['n_postings']

    for col_idx, (period, plabel) in enumerate(period_info):
        ax = axes[row_idx][col_idx]
        pdata = agg_sub[agg_sub['period'] == period]
        n_p = n_sub[n_sub['period'] == period]['n_postings'].values
        n_p = n_p[0] if len(n_p) > 0 else 0
        total = pdata['avg_chars'].sum() if len(pdata) > 0 else 0

        bottom = 0
        for sec in all_sections:
            sd = pdata[pdata['section'] == sec]
            val = sd['avg_chars'].values[0] if len(sd) > 0 else 0
            ax.bar(0, val, bottom=bottom, color=section_colors.get(sec, '#999'),
                   width=0.6, edgecolor='white', linewidth=0.5)
            if val > total * 0.04 and total > 0:
                ax.text(0, bottom + val/2, f'{val:.0f}', ha='center', va='center', fontsize=7)
            bottom += val

        title = f'{seniority.upper()} | {plabel.split(chr(10))[0]}\nn={n_p}'
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])

    axes[row_idx][0].set_ylabel(f'{seniority.capitalize()} — avg chars')

plt.suptitle('Section Anatomy by Seniority and Period\nSWE, LinkedIn', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'section_anatomy_by_seniority.png')
plt.close()
print("  Saved section_anatomy_by_seniority.png")

print("\nDone.")
