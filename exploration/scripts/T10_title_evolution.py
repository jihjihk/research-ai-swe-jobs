#!/usr/bin/env python3
"""T10: Title taxonomy evolution — SWE LinkedIn postings, 2024 vs 2026."""

import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import re

BASE = Path("/home/jihgaboot/gabor/job-research")
DATA = BASE / "data/unified.parquet"
TECH_MATRIX = BASE / "exploration/artifacts/shared/swe_tech_matrix.parquet"
CLEANED = BASE / "exploration/artifacts/shared/swe_cleaned_text.parquet"
FIG_DIR = BASE / "exploration/figures/T10"
TAB_DIR = BASE / "exploration/tables/T10"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

FILTERS = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe = true"

con = duckdb.connect()

# ============================================================
# 0. Load base data
# ============================================================
print("Loading base data...")
base = con.sql(f"""
    SELECT uid, title_normalized, title, period, source, seniority_final, seniority_3level,
           is_aggregator, company_name_canonical, yoe_extracted,
           description_length, core_length
    FROM '{DATA}'
    WHERE {FILTERS}
""").fetchdf()

print(f"Total SWE rows: {len(base)}")
print(f"By period: {base.groupby('period').size().to_dict()}")

# Split by period for comparison
arshkon = base[base['source'] == 'kaggle_arshkon']
asaniczka = base[base['source'] == 'kaggle_asaniczka']
scraped = base[base['source'] == 'scraped']
p2024_01 = asaniczka
p2024_04 = arshkon
p2026 = scraped

# ============================================================
# 1. New and disappeared titles
# ============================================================
print("\n=== 1. New and disappeared titles ===")

# Get title counts by period
def title_counts(df, min_count=1):
    return df['title_normalized'].value_counts()

titles_2024_01 = title_counts(p2024_01)
titles_2024_04 = title_counts(p2024_04)
titles_2026 = title_counts(p2026)

# Combine 2024 titles (arshkon primary comparison)
all_2024_titles = set(titles_2024_01.index) | set(titles_2024_04.index)
titles_2026_set = set(titles_2026.index)

# New titles (in 2026, not in either 2024 source)
new_titles = titles_2026_set - all_2024_titles
new_titles_df = titles_2026[titles_2026.index.isin(new_titles)].sort_values(ascending=False)
print(f"New titles in 2026 (not in any 2024): {len(new_titles)}")
print(f"New titles with >=5 postings: {(new_titles_df >= 5).sum()}")
print("Top 30 new titles:")
print(new_titles_df.head(30).to_string())

# Disappeared titles (in arshkon 2024 with >=3, not in 2026)
# Use arshkon as baseline since it's closer in size to scraped
disappeared = set(titles_2024_04[titles_2024_04 >= 3].index) - titles_2026_set
disappeared_df = titles_2024_04[titles_2024_04.index.isin(disappeared)].sort_values(ascending=False)
print(f"\nDisappeared titles (in arshkon >=3, not in 2026): {len(disappeared)}")
print("Top 30 disappeared:")
print(disappeared_df.head(30).to_string())

# Save tables
new_titles_df.reset_index().rename(columns={'index': 'title', 'count': 'n_2026'}).head(100).to_csv(
    TAB_DIR / "new_titles_2026.csv", index=False)
disappeared_df.reset_index().rename(columns={'index': 'title', 'count': 'n_2024_04'}).head(100).to_csv(
    TAB_DIR / "disappeared_titles.csv", index=False)

# Seniority distribution for top titles in both periods
common_titles = titles_2024_04[titles_2024_04 >= 10].index.intersection(
    titles_2026[titles_2026 >= 10].index)
print(f"\nCommon high-frequency titles (>=10 in both arshkon & 2026): {len(common_titles)}")

seniority_shift = []
for t in common_titles:
    for pname, pdf in [('2024-04', p2024_04), ('2026-03', p2026)]:
        sub = pdf[pdf['title_normalized'] == t]
        dist = sub['seniority_3level'].value_counts(normalize=True)
        seniority_shift.append({
            'title': t,
            'period': pname,
            'n': len(sub),
            'pct_junior': dist.get('junior', 0),
            'pct_mid': dist.get('mid', 0),
            'pct_senior': dist.get('senior', 0),
            'pct_unknown': dist.get('unknown', 0),
        })
seniority_shift_df = pd.DataFrame(seniority_shift)
seniority_shift_df.to_csv(TAB_DIR / "seniority_shift_common_titles.csv", index=False)

# ============================================================
# 2. Title concentration
# ============================================================
print("\n=== 2. Title concentration ===")

for pname, pdf in [('2024-01', p2024_01), ('2024-04', p2024_04), ('2026-03', p2026)]:
    n = len(pdf)
    nunique = pdf['title_normalized'].nunique()
    per_1k = nunique / (n / 1000)
    top10_share = title_counts(pdf).head(10).sum() / n
    top50_share = title_counts(pdf).head(50).sum() / n
    hhi = ((title_counts(pdf) / n) ** 2).sum()
    print(f"{pname}: {n} postings, {nunique} unique titles, {per_1k:.1f} per 1K, "
          f"top10 share={top10_share:.1%}, top50 share={top50_share:.1%}, HHI={hhi:.6f}")

# Also do aggregator-excluded
print("\nAggregator-excluded:")
for pname, pdf in [('2024-01', p2024_01), ('2024-04', p2024_04), ('2026-03', p2026)]:
    pdf_na = pdf[~pdf['is_aggregator']]
    n = len(pdf_na)
    nunique = pdf_na['title_normalized'].nunique()
    per_1k = nunique / (n / 1000) if n > 0 else 0
    top10_share = title_counts(pdf_na).head(10).sum() / n if n > 0 else 0
    print(f"{pname}: {n} postings, {nunique} unique titles, {per_1k:.1f} per 1K, top10 share={top10_share:.1%}")

# ============================================================
# 3. AI/ML compound titles
# ============================================================
print("\n=== 3. AI/ML compound titles ===")

ai_pattern = re.compile(r'\b(ai|artificial intelligence|ml|machine learning|llm|'
                        r'large language|generative|gen\s?ai|agent|deep learning|'
                        r'nlp|natural language|computer vision|neural)\b', re.IGNORECASE)

for pname, pdf in [('2024-01', p2024_01), ('2024-04', p2024_04), ('2026-03', p2026)]:
    has_ai = pdf['title_normalized'].str.contains(ai_pattern, na=False)
    ai_share = has_ai.mean()
    n_ai = has_ai.sum()
    print(f"{pname}: {n_ai} AI/ML titles ({ai_share:.1%})")
    if n_ai > 0:
        ai_titles = pdf.loc[has_ai, 'title_normalized'].value_counts().head(15)
        print(f"  Top: {ai_titles.to_dict()}")

# Aggregator-excluded
print("\nAI titles, aggregator-excluded:")
for pname, pdf in [('2024-01', p2024_01), ('2024-04', p2024_04), ('2026-03', p2026)]:
    pdf_na = pdf[~pdf['is_aggregator']]
    has_ai = pdf_na['title_normalized'].str.contains(ai_pattern, na=False)
    ai_share = has_ai.mean()
    print(f"  {pname}: {has_ai.sum()} ({ai_share:.1%})")

# ============================================================
# 4. Title-to-content alignment (cosine similarity)
# ============================================================
print("\n=== 4. Title-to-content alignment ===")

# Load embeddings
emb_index = con.sql(f"""
    SELECT row_index, uid FROM 'exploration/artifacts/shared/swe_embedding_index.parquet'
""").fetchdf()
embeddings = np.load(BASE / "exploration/artifacts/shared/swe_embeddings.npy")

# Join to get period info
emb_meta = emb_index.merge(base[['uid', 'title_normalized', 'period', 'source']], on='uid', how='inner')

# Find top titles in both arshkon and scraped
common_top = []
for t in titles_2024_04.head(50).index:
    if t in titles_2026 and titles_2026[t] >= 10 and titles_2024_04[t] >= 10:
        common_top.append(t)
    if len(common_top) >= 15:
        break

print(f"Analyzing {len(common_top)} common titles for content drift")

from sklearn.metrics.pairwise import cosine_similarity

alignment_results = []
for title in common_top:
    rows_2024 = emb_meta[(emb_meta['title_normalized'] == title) & (emb_meta['source'] == 'kaggle_arshkon')]
    rows_2026 = emb_meta[(emb_meta['title_normalized'] == title) & (emb_meta['source'] == 'scraped')]

    if len(rows_2024) < 5 or len(rows_2026) < 5:
        continue

    idx_2024 = rows_2024['row_index'].values
    idx_2026 = rows_2026['row_index'].values

    # Compute mean embeddings for each period
    mean_2024 = embeddings[idx_2024].mean(axis=0, keepdims=True)
    mean_2026 = embeddings[idx_2026].mean(axis=0, keepdims=True)

    sim = cosine_similarity(mean_2024, mean_2026)[0][0]

    # Also compute within-period similarity as baseline
    if len(idx_2024) >= 10:
        half = len(idx_2024) // 2
        within_sim = cosine_similarity(
            embeddings[idx_2024[:half]].mean(axis=0, keepdims=True),
            embeddings[idx_2024[half:]].mean(axis=0, keepdims=True)
        )[0][0]
    else:
        within_sim = np.nan

    alignment_results.append({
        'title': title,
        'n_2024': len(rows_2024),
        'n_2026': len(rows_2026),
        'cross_period_sim': sim,
        'within_2024_sim': within_sim,
        'content_drift': within_sim - sim if not np.isnan(within_sim) else np.nan
    })

alignment_df = pd.DataFrame(alignment_results).sort_values('cross_period_sim')
print(alignment_df.to_string())
alignment_df.to_csv(TAB_DIR / "title_content_alignment.csv", index=False)

# ============================================================
# 5. Seniority-marker title shares
# ============================================================
print("\n=== 5. Seniority-marker title shares ===")

seniority_markers = {
    'junior': r'\bjunior\b',
    'associate': r'\bassociate\b',
    'senior': r'\bsenior\b',
    'staff': r'\bstaff\b',
    'principal': r'\bprincipal\b',
    'lead': r'\blead\b',
    'director': r'\bdirector\b',
    'architect': r'\barchitect\b',
    'founding': r'\bfounding\b',
    'intern': r'\bintern\b',
}

marker_results = []
for pname, pdf in [('2024-01', p2024_01), ('2024-04', p2024_04), ('2026-03', p2026)]:
    n = len(pdf)
    for marker, pattern in seniority_markers.items():
        count = pdf['title_normalized'].str.contains(pattern, na=False, flags=re.IGNORECASE).sum()
        marker_results.append({
            'period': pname,
            'marker': marker,
            'count': count,
            'share': count / n
        })

    # Also aggregator-excluded
    pdf_na = pdf[~pdf['is_aggregator']]
    n_na = len(pdf_na)
    for marker, pattern in seniority_markers.items():
        count = pdf_na['title_normalized'].str.contains(pattern, na=False, flags=re.IGNORECASE).sum()
        marker_results.append({
            'period': pname + '_noagg',
            'marker': marker,
            'count': count,
            'share': count / n_na if n_na > 0 else 0
        })

marker_df = pd.DataFrame(marker_results)
marker_pivot = marker_df[~marker_df['period'].str.contains('noagg')].pivot(
    index='marker', columns='period', values='share').round(4)
marker_pivot_noagg = marker_df[marker_df['period'].str.contains('noagg')].copy()
marker_pivot_noagg['period'] = marker_pivot_noagg['period'].str.replace('_noagg', '')
marker_pivot_noagg = marker_pivot_noagg.pivot(
    index='marker', columns='period', values='share').round(4)

print("All postings:")
print(marker_pivot.to_string())
print("\nAggregator-excluded:")
print(marker_pivot_noagg.to_string())

marker_df.to_csv(TAB_DIR / "seniority_marker_shares.csv", index=False)

# ============================================================
# 6. Emerging role categories
# ============================================================
print("\n=== 6. Emerging role categories ===")

# Categorize new titles by theme
theme_patterns = {
    'ai_ml': r'\b(ai|ml|machine learning|deep learning|llm|nlp|computer vision|generative|gen\s?ai|neural)\b',
    'platform_infra': r'\b(platform|infrastructure|sre|reliability|cloud|devops|systems)\b',
    'data': r'\b(data|analytics|etl|pipeline|warehouse)\b',
    'security': r'\b(security|cybersecurity|devsecops|appsec)\b',
    'mobile': r'\b(mobile|ios|android|flutter|react native)\b',
    'frontend': r'\b(frontend|front.end|ui|ux)\b',
    'backend': r'\b(backend|back.end|api|microservice)\b',
    'fullstack': r'\b(fullstack|full.stack)\b',
    'embedded': r'\b(embedded|firmware|iot|hardware)\b',
    'management': r'\b(manager|director|head|vp|chief)\b',
    'founding': r'\b(founding|co.founder|startup)\b',
    'agent': r'\b(agent|autonomous|agentic)\b',
}

new_high_freq = new_titles_df[new_titles_df >= 3]
print(f"Categorizing {len(new_high_freq)} new titles with >=3 postings")

theme_counts = {}
theme_examples = {}
for title in new_high_freq.index:
    matched = False
    for theme, pat in theme_patterns.items():
        if re.search(pat, title, re.IGNORECASE):
            theme_counts[theme] = theme_counts.get(theme, 0) + new_high_freq[title]
            if theme not in theme_examples:
                theme_examples[theme] = []
            theme_examples[theme].append((title, new_high_freq[title]))
            matched = True
            break  # first match wins
    if not matched:
        theme_counts['other'] = theme_counts.get('other', 0) + new_high_freq[title]

print("Theme distribution of new 2026 titles (by posting count):")
for theme, count in sorted(theme_counts.items(), key=lambda x: -x[1]):
    examples = theme_examples.get(theme, [])[:5]
    ex_str = '; '.join([f"{t} ({n})" for t, n in examples])
    print(f"  {theme}: {count} postings. Examples: {ex_str}")

# ============================================================
# FIGURES
# ============================================================
print("\n=== Generating figures ===")

# Fig 1: Seniority marker share change
fig, ax = plt.subplots(figsize=(10, 6))
markers_to_plot = ['junior', 'associate', 'senior', 'staff', 'principal', 'lead', 'founding']
plot_data = marker_df[
    (~marker_df['period'].str.contains('noagg')) &
    (marker_df['marker'].isin(markers_to_plot))
]
for marker in markers_to_plot:
    sub = plot_data[plot_data['marker'] == marker].sort_values('period')
    ax.plot(sub['period'], sub['share'] * 100, 'o-', label=marker, markersize=8)
ax.set_ylabel('Share of SWE postings (%)')
ax.set_xlabel('Period')
ax.set_title('Seniority Markers in SWE Job Titles Over Time')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "seniority_marker_trends.png", dpi=150, bbox_inches='tight')
plt.close()

# Fig 2: AI title share over time
fig, ax = plt.subplots(figsize=(8, 5))
ai_shares = []
for pname, pdf in [('2024-01', p2024_01), ('2024-04', p2024_04), ('2026-03', p2026)]:
    has_ai = pdf['title_normalized'].str.contains(ai_pattern, na=False)
    ai_shares.append({'period': pname, 'ai_share': has_ai.mean() * 100, 'variant': 'all'})
    pdf_na = pdf[~pdf['is_aggregator']]
    has_ai_na = pdf_na['title_normalized'].str.contains(ai_pattern, na=False)
    ai_shares.append({'period': pname, 'ai_share': has_ai_na.mean() * 100, 'variant': 'no_aggregator'})

ai_df = pd.DataFrame(ai_shares)
for variant in ['all', 'no_aggregator']:
    sub = ai_df[ai_df['variant'] == variant].sort_values('period')
    style = '-o' if variant == 'all' else '--s'
    ax.plot(sub['period'], sub['ai_share'], style, label=variant, markersize=8)
ax.set_ylabel('Share with AI/ML in title (%)')
ax.set_xlabel('Period')
ax.set_title('AI/ML Terms in SWE Job Titles')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "ai_title_share.png", dpi=150, bbox_inches='tight')
plt.close()

# Fig 3: Title concentration (unique titles per 1K)
fig, ax = plt.subplots(figsize=(8, 5))
conc_data = []
for pname, pdf in [('2024-01', p2024_01), ('2024-04', p2024_04), ('2026-03', p2026)]:
    n = len(pdf)
    nunique = pdf['title_normalized'].nunique()
    conc_data.append({'period': pname, 'unique_per_1k': nunique / (n/1000), 'variant': 'all'})
    pdf_na = pdf[~pdf['is_aggregator']]
    n_na = len(pdf_na)
    nunique_na = pdf_na['title_normalized'].nunique()
    conc_data.append({'period': pname, 'unique_per_1k': nunique_na / (n_na/1000), 'variant': 'no_aggregator'})
conc_df = pd.DataFrame(conc_data)
for variant in ['all', 'no_aggregator']:
    sub = conc_df[conc_df['variant'] == variant].sort_values('period')
    style = '-o' if variant == 'all' else '--s'
    ax.plot(sub['period'], sub['unique_per_1k'], style, label=variant, markersize=8)
ax.set_ylabel('Unique titles per 1,000 postings')
ax.set_xlabel('Period')
ax.set_title('Title Diversity Over Time')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "title_concentration.png", dpi=150, bbox_inches='tight')
plt.close()

# Fig 4: Content alignment scatter
if len(alignment_df) > 0:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(alignment_df)), alignment_df['cross_period_sim'], color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(alignment_df)))
    ax.set_yticklabels(alignment_df['title'], fontsize=8)
    ax.set_xlabel('Cosine Similarity (2024 vs 2026 description centroids)')
    ax.set_title('Title-to-Content Alignment: Same Title, Same Job?')
    ax.axvline(x=alignment_df['cross_period_sim'].median(), color='red', linestyle='--', alpha=0.5, label='median')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "content_alignment.png", dpi=150, bbox_inches='tight')
    plt.close()

print("\n=== T10 complete ===")
print(f"Figures: {list(FIG_DIR.glob('*.png'))}")
print(f"Tables: {list(TAB_DIR.glob('*.csv'))}")
