#!/usr/bin/env python3
"""
T13: Company-level patterns — Within-company vs between-company composition effects.

Steps:
1. Find companies appearing in both arshkon and scraped (using company_name_canonical)
2. Overlapping companies: seniority distributions, description lengths, AI keyword prevalence
3. Non-overlapping: what companies are new in 2026? What disappeared?
4. Size-band split where available
"""

import duckdb
import pandas as pd
import numpy as np
import re
import os
import sys
from pathlib import Path

# Paths
PARQUET = 'preprocessing/intermediate/stage8_final.parquet'
FIG_DIR = 'exploration/figures/T13'
TBL_DIR = 'exploration/tables/T13'

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

con = duckdb.connect()

# ------------------------------------------------------------------
# Step 1: Identify overlapping companies between arshkon and scraped
# ------------------------------------------------------------------
print("Step 1: Finding overlapping companies (arshkon + scraped)...")
sys.stdout.flush()

# Load SWE data with seniority patched
swe_df = con.sql(f"""
WITH swe_patched AS (
  SELECT *,
    CASE
      WHEN seniority_final_source IN ('title_keyword', 'title_manager')
        THEN seniority_final
      WHEN seniority_native IS NOT NULL AND seniority_native != ''
        THEN CASE seniority_native
               WHEN 'intern' THEN 'entry'
               WHEN 'executive' THEN 'director'
               ELSE seniority_native
             END
      WHEN seniority_final != 'unknown'
        THEN seniority_final
      ELSE 'unknown'
    END AS seniority_patched,
    CASE
      WHEN seniority_final_source IN ('title_keyword', 'title_manager')
        THEN CASE seniority_final
               WHEN 'entry' THEN 'junior'
               WHEN 'associate' THEN 'mid'
               WHEN 'mid-senior' THEN 'senior'
               WHEN 'director' THEN 'senior'
               ELSE 'unknown'
             END
      WHEN seniority_native IS NOT NULL AND seniority_native != ''
        THEN CASE seniority_native
               WHEN 'entry' THEN 'junior'
               WHEN 'intern' THEN 'junior'
               WHEN 'associate' THEN 'mid'
               WHEN 'mid-senior' THEN 'senior'
               WHEN 'director' THEN 'senior'
               WHEN 'executive' THEN 'senior'
               ELSE 'unknown'
             END
      WHEN seniority_final != 'unknown'
        THEN CASE seniority_final
               WHEN 'entry' THEN 'junior'
               WHEN 'associate' THEN 'mid'
               WHEN 'mid-senior' THEN 'senior'
               WHEN 'director' THEN 'senior'
               ELSE 'unknown'
             END
      ELSE 'unknown'
    END AS seniority_3level_patched,
    CASE
      WHEN seniority_final_source IN ('title_keyword', 'title_manager')
        THEN 'title_strong'
      WHEN seniority_native IS NOT NULL AND seniority_native != ''
        THEN 'native_backfill'
      WHEN seniority_final != 'unknown'
        THEN 'weak_signal'
      ELSE 'unknown'
    END AS seniority_patched_source
  FROM '{PARQUET}'
  WHERE source_platform = 'linkedin'
    AND is_english = true
    AND date_flag = 'ok'
)
SELECT uid, source, period, company_name_canonical, company_name_effective,
       seniority_patched, seniority_3level_patched,
       description, description_length, is_aggregator,
       company_size, company_size_category, company_industry
FROM swe_patched
WHERE is_swe = true
  AND source IN ('kaggle_arshkon', 'scraped')
""").df()

print(f"Loaded {len(swe_df)} SWE rows (arshkon + scraped)")
print(f"  arshkon: {(swe_df['source'] == 'kaggle_arshkon').sum()}")
print(f"  scraped: {(swe_df['source'] == 'scraped').sum()}")
sys.stdout.flush()

# Get company sets per source
arshkon_companies = set(swe_df[swe_df['source'] == 'kaggle_arshkon']['company_name_canonical'].dropna().unique())
scraped_companies = set(swe_df[swe_df['source'] == 'scraped']['company_name_canonical'].dropna().unique())

overlap = arshkon_companies & scraped_companies
arshkon_only = arshkon_companies - scraped_companies
scraped_only = scraped_companies - arshkon_companies

print(f"\nCompany overlap:")
print(f"  Arshkon unique companies: {len(arshkon_companies)}")
print(f"  Scraped unique companies: {len(scraped_companies)}")
print(f"  Overlap: {len(overlap)}")
print(f"  Arshkon-only: {len(arshkon_only)}")
print(f"  Scraped-only: {len(scraped_only)}")
sys.stdout.flush()

# Tag rows
swe_df['company_overlap'] = swe_df['company_name_canonical'].apply(
    lambda x: 'overlap' if x in overlap else ('arshkon_only' if x in arshkon_only else ('scraped_only' if x in scraped_only else 'unknown'))
)

# AI keyword pattern (moved here so overlap_rows gets has_ai)
AI_KEYWORDS = re.compile(
    r'\b(?:ai|artificial\s+intelligence|machine\s+learning|ml|deep\s+learning|'
    r'llm|large\s+language\s+model|genai|generative\s+ai|gpt|'
    r'copilot|github\s+copilot|ai[\s-]?(?:agent|tool|assistant|coding|powered|driven|enabled|native)|'
    r'prompt\s+engineering|rag|retrieval[\s-]augmented|'
    r'ai[\s-]?first|ai[\s-]?native|'
    r'langchain|openai|hugging\s*face|'
    r'chatgpt|claude|gemini|anthropic)\b',
    re.IGNORECASE
)

swe_df['has_ai'] = swe_df['description'].apply(lambda x: bool(AI_KEYWORDS.search(x)) if pd.notna(x) else False)

# How many rows in overlapping companies?
overlap_rows = swe_df[swe_df['company_overlap'] == 'overlap']
print(f"\nRows in overlapping companies: {len(overlap_rows)}")
print(f"  arshkon: {(overlap_rows['source'] == 'kaggle_arshkon').sum()}")
print(f"  scraped: {(overlap_rows['source'] == 'scraped').sum()}")
sys.stdout.flush()

# ------------------------------------------------------------------
# Step 2: Overlapping companies analysis
# ------------------------------------------------------------------
print("\nStep 2: Analyzing overlapping companies...")
sys.stdout.flush()

# 2a. Seniority distribution comparison for overlapping companies
print("\n2a. Seniority distributions in overlapping companies:")
for src in ['kaggle_arshkon', 'scraped']:
    subset = overlap_rows[overlap_rows['source'] == src]
    dist = subset['seniority_patched'].value_counts(normalize=True).sort_index()
    print(f"\n  {src}:")
    for k, v in dist.items():
        print(f"    {k}: {100*v:.1f}%  (n={subset[subset['seniority_patched']==k].shape[0]})")
sys.stdout.flush()

# 2b. Description length comparison
print("\n2b. Description length in overlapping companies:")
for src in ['kaggle_arshkon', 'scraped']:
    subset = overlap_rows[overlap_rows['source'] == src]
    print(f"  {src}: median={subset['description_length'].median():.0f}, mean={subset['description_length'].mean():.0f}")
sys.stdout.flush()

# 2c. AI keyword prevalence
print("\n2c. AI keyword prevalence in overlapping companies:")
for src in ['kaggle_arshkon', 'scraped']:
    subset = overlap_rows[overlap_rows['source'] == src]
    print(f"  {src}: {100*subset['has_ai'].mean():.1f}% (n={subset['has_ai'].sum()})")
sys.stdout.flush()

# 2d. Per-company within-company changes (top companies by posting count)
print("\n2d. Within-company changes for top overlapping companies:")

company_stats = []
for company in overlap:
    arshkon_sub = swe_df[(swe_df['company_name_canonical'] == company) & (swe_df['source'] == 'kaggle_arshkon')]
    scraped_sub = swe_df[(swe_df['company_name_canonical'] == company) & (swe_df['source'] == 'scraped')]
    
    n_arshkon = len(arshkon_sub)
    n_scraped = len(scraped_sub)
    
    if n_arshkon == 0 or n_scraped == 0:
        continue
    
    # Seniority: entry share
    entry_share_arshkon = (arshkon_sub['seniority_patched'] == 'entry').mean()
    entry_share_scraped = (scraped_sub['seniority_patched'] == 'entry').mean()
    
    # Description length
    desc_len_arshkon = arshkon_sub['description_length'].median()
    desc_len_scraped = scraped_sub['description_length'].median()
    
    # AI keywords
    ai_arshkon = arshkon_sub['has_ai'].mean()
    ai_scraped = scraped_sub['has_ai'].mean()
    
    # Junior share (3-level)
    junior_share_arshkon = (arshkon_sub['seniority_3level_patched'] == 'junior').mean()
    junior_share_scraped = (scraped_sub['seniority_3level_patched'] == 'junior').mean()
    
    # Unknown share
    unknown_arshkon = (arshkon_sub['seniority_patched'] == 'unknown').mean()
    unknown_scraped = (scraped_sub['seniority_patched'] == 'unknown').mean()
    
    company_stats.append({
        'company': company,
        'n_arshkon': n_arshkon,
        'n_scraped': n_scraped,
        'n_total': n_arshkon + n_scraped,
        'entry_share_arshkon': entry_share_arshkon,
        'entry_share_scraped': entry_share_scraped,
        'entry_share_delta': entry_share_scraped - entry_share_arshkon,
        'junior_share_arshkon': junior_share_arshkon,
        'junior_share_scraped': junior_share_scraped,
        'junior_share_delta': junior_share_scraped - junior_share_arshkon,
        'desc_len_arshkon': desc_len_arshkon,
        'desc_len_scraped': desc_len_scraped,
        'desc_len_ratio': desc_len_scraped / desc_len_arshkon if desc_len_arshkon > 0 else np.nan,
        'ai_arshkon': ai_arshkon,
        'ai_scraped': ai_scraped,
        'ai_delta': ai_scraped - ai_arshkon,
        'unknown_arshkon': unknown_arshkon,
        'unknown_scraped': unknown_scraped,
    })

company_df = pd.DataFrame(company_stats).sort_values('n_total', ascending=False)
company_df.to_csv(f'{TBL_DIR}/T13_company_level_changes.csv', index=False)

# Top 20 companies
print("\nTop 20 overlapping companies by total SWE postings:")
top20 = company_df.head(20)
for _, row in top20.iterrows():
    print(f"  {row['company']:<35} arshkon={row['n_arshkon']:>3}  scraped={row['n_scraped']:>3}  "
          f"entry: {100*row['entry_share_arshkon']:.0f}%->{100*row['entry_share_scraped']:.0f}%  "
          f"AI: {100*row['ai_arshkon']:.0f}%->{100*row['ai_scraped']:.0f}%  "
          f"desc_len: {row['desc_len_arshkon']:.0f}->{row['desc_len_scraped']:.0f}")
sys.stdout.flush()

# ------------------------------------------------------------------
# Step 3: Aggregate within-company vs between-company decomposition
# ------------------------------------------------------------------
print("\n\nStep 3: Within-company vs between-company decomposition...")
sys.stdout.flush()

# Weight by company size (posting count)
# Within-company change: average change among overlapping companies
# Between-company change: difference attributable to composition (new/departed companies)

# For entry share
# Overall entry share 2024 (arshkon) vs 2026 (scraped)
arshkon_all = swe_df[swe_df['source'] == 'kaggle_arshkon']
scraped_all = swe_df[swe_df['source'] == 'scraped']

overall_entry_arshkon = (arshkon_all['seniority_patched'] == 'entry').mean()
overall_entry_scraped = (scraped_all['seniority_patched'] == 'entry').mean()
overall_delta = overall_entry_scraped - overall_entry_arshkon

# Within-company (overlapping only)
arshkon_overlap = swe_df[(swe_df['source'] == 'kaggle_arshkon') & (swe_df['company_overlap'] == 'overlap')]
scraped_overlap = swe_df[(swe_df['source'] == 'scraped') & (swe_df['company_overlap'] == 'overlap')]

within_entry_arshkon = (arshkon_overlap['seniority_patched'] == 'entry').mean()
within_entry_scraped = (scraped_overlap['seniority_patched'] == 'entry').mean()
within_delta = within_entry_scraped - within_entry_arshkon

# Composition effect (difference between overall and within)
composition_delta = overall_delta - within_delta

print(f"\nEntry-level share decomposition:")
print(f"  Overall 2024 (arshkon): {100*overall_entry_arshkon:.1f}%")
print(f"  Overall 2026 (scraped): {100*overall_entry_scraped:.1f}%")
print(f"  Overall delta: {100*overall_delta:+.1f}pp")
print(f"  Within-company delta (overlap): {100*within_delta:+.1f}pp")
print(f"  Composition effect (residual): {100*composition_delta:+.1f}pp")

# Same for AI keyword prevalence
overall_ai_arshkon = arshkon_all['has_ai'].mean()
overall_ai_scraped = scraped_all['has_ai'].mean()
overall_ai_delta = overall_ai_scraped - overall_ai_arshkon

within_ai_arshkon = arshkon_overlap['has_ai'].mean()
within_ai_scraped = scraped_overlap['has_ai'].mean()
within_ai_delta = within_ai_scraped - within_ai_arshkon

composition_ai_delta = overall_ai_delta - within_ai_delta

print(f"\nAI keyword prevalence decomposition:")
print(f"  Overall 2024: {100*overall_ai_arshkon:.1f}%")
print(f"  Overall 2026: {100*overall_ai_scraped:.1f}%")
print(f"  Overall delta: {100*overall_ai_delta:+.1f}pp")
print(f"  Within-company delta: {100*within_ai_delta:+.1f}pp")
print(f"  Composition effect: {100*composition_ai_delta:+.1f}pp")

# Description length
overall_len_arshkon = arshkon_all['description_length'].median()
overall_len_scraped = scraped_all['description_length'].median()
within_len_arshkon = arshkon_overlap['description_length'].median()
within_len_scraped = scraped_overlap['description_length'].median()

print(f"\nDescription length decomposition (median):")
print(f"  Overall 2024: {overall_len_arshkon:.0f}")
print(f"  Overall 2026: {overall_len_scraped:.0f}")
print(f"  Overall ratio: {overall_len_scraped/overall_len_arshkon:.2f}x")
print(f"  Within-company 2024: {within_len_arshkon:.0f}")
print(f"  Within-company 2026: {within_len_scraped:.0f}")
print(f"  Within-company ratio: {within_len_scraped/within_len_arshkon:.2f}x")
sys.stdout.flush()

# ------------------------------------------------------------------
# Step 4: Non-overlapping companies
# ------------------------------------------------------------------
print("\n\nStep 4: Non-overlapping companies...")
sys.stdout.flush()

# Scraped-only companies (new in 2026)
scraped_only_rows = swe_df[(swe_df['source'] == 'scraped') & (swe_df['company_overlap'] == 'scraped_only')]
print(f"\nScraped-only (new 2026) companies: {len(scraped_only)}")
print(f"  Rows: {len(scraped_only_rows)}")

# Top scraped-only companies
scraped_only_top = scraped_only_rows.groupby('company_name_canonical').agg(
    n=('uid', 'count'),
    entry_share=('seniority_patched', lambda x: (x == 'entry').mean()),
    ai_rate=('has_ai', 'mean'),
    median_desc_len=('description_length', 'median')
).sort_values('n', ascending=False).head(20)
print(f"\nTop 20 scraped-only companies:")
print(scraped_only_top.to_string())

# Arshkon-only companies (disappeared by 2026)
arshkon_only_rows = swe_df[(swe_df['source'] == 'kaggle_arshkon') & (swe_df['company_overlap'] == 'arshkon_only')]
print(f"\n\nArshkon-only (not in 2026) companies: {len(arshkon_only)}")
print(f"  Rows: {len(arshkon_only_rows)}")

# Top arshkon-only companies
arshkon_only_top = arshkon_only_rows.groupby('company_name_canonical').agg(
    n=('uid', 'count'),
    entry_share=('seniority_patched', lambda x: (x == 'entry').mean()),
    ai_rate=('has_ai', 'mean'),
    median_desc_len=('description_length', 'median')
).sort_values('n', ascending=False).head(20)
print(f"\nTop 20 arshkon-only companies:")
print(arshkon_only_top.to_string())
sys.stdout.flush()

# Comparison: overlapping vs non-overlapping characteristics
print("\n\nCharacteristics comparison: overlap vs non-overlap")
groups = {
    'overlap_arshkon': swe_df[(swe_df['source'] == 'kaggle_arshkon') & (swe_df['company_overlap'] == 'overlap')],
    'overlap_scraped': swe_df[(swe_df['source'] == 'scraped') & (swe_df['company_overlap'] == 'overlap')],
    'arshkon_only': arshkon_only_rows,
    'scraped_only': scraped_only_rows,
}

for gname, gdf in groups.items():
    print(f"\n  {gname} (n={len(gdf)}):")
    print(f"    Entry share: {100*(gdf['seniority_patched'] == 'entry').mean():.1f}%")
    print(f"    Unknown share: {100*(gdf['seniority_patched'] == 'unknown').mean():.1f}%")
    print(f"    Median desc len: {gdf['description_length'].median():.0f}")
    print(f"    AI prevalence: {100*gdf['has_ai'].mean():.1f}%")
    print(f"    Aggregator rate: {100*gdf['is_aggregator'].mean():.1f}%")
sys.stdout.flush()

# ------------------------------------------------------------------
# Step 5: Size-band analysis where available
# ------------------------------------------------------------------
print("\n\nStep 5: Size-band analysis (arshkon only — only source with company_size)...")
sys.stdout.flush()

# company_size_category is from arshkon companion data
arshkon_with_size = swe_df[(swe_df['source'] == 'kaggle_arshkon') & (swe_df['company_size_category'].notna())]
print(f"Arshkon rows with company_size_category: {len(arshkon_with_size)} / {(swe_df['source'] == 'kaggle_arshkon').sum()}")
print(f"Size categories: {arshkon_with_size['company_size_category'].value_counts().to_dict()}")

# Check overlap companies by size
overlap_arshkon_with_size = arshkon_with_size[arshkon_with_size['company_overlap'] == 'overlap']
print(f"\nOverlap companies with size info: {overlap_arshkon_with_size['company_name_canonical'].nunique()}")
print(f"Size distribution of overlapping companies:")
size_dist = overlap_arshkon_with_size.groupby('company_size_category')['company_name_canonical'].nunique()
print(size_dist.to_string())

# Check company_size (numeric) distribution for overlap companies
if swe_df['company_size'].notna().sum() > 0:
    # Get company size from arshkon for overlap companies
    company_sizes = arshkon_with_size[arshkon_with_size['company_overlap'] == 'overlap'].groupby(
        'company_name_canonical')['company_size'].first().dropna()
    
    if len(company_sizes) > 0:
        # Define size bands
        def size_band(size):
            if pd.isna(size):
                return 'unknown'
            if size < 100:
                return '<100'
            elif size < 1000:
                return '100-999'
            elif size < 10000:
                return '1K-10K'
            else:
                return '10K+'
        
        company_size_map = company_sizes.apply(size_band).to_dict()
        
        # Apply to overlap rows
        overlap_rows_sized = overlap_rows.copy()
        overlap_rows_sized['size_band'] = overlap_rows_sized['company_name_canonical'].map(company_size_map)
        overlap_rows_sized = overlap_rows_sized[overlap_rows_sized['size_band'].notna() & (overlap_rows_sized['size_band'] != 'unknown')]
        
        print(f"\nOverlap rows with size band: {len(overlap_rows_sized)}")
        
        # Per size band: entry share change, AI change, desc length change
        size_band_stats = []
        for band in ['<100', '100-999', '1K-10K', '10K+']:
            band_rows = overlap_rows_sized[overlap_rows_sized['size_band'] == band]
            arshkon_band = band_rows[band_rows['source'] == 'kaggle_arshkon']
            scraped_band = band_rows[band_rows['source'] == 'scraped']
            
            n_companies = band_rows['company_name_canonical'].nunique()
            
            if len(arshkon_band) > 0 and len(scraped_band) > 0:
                size_band_stats.append({
                    'size_band': band,
                    'n_companies': n_companies,
                    'n_arshkon': len(arshkon_band),
                    'n_scraped': len(scraped_band),
                    'entry_arshkon': (arshkon_band['seniority_patched'] == 'entry').mean(),
                    'entry_scraped': (scraped_band['seniority_patched'] == 'entry').mean(),
                    'ai_arshkon': arshkon_band['has_ai'].mean(),
                    'ai_scraped': scraped_band['has_ai'].mean(),
                    'desc_len_arshkon': arshkon_band['description_length'].median(),
                    'desc_len_scraped': scraped_band['description_length'].median(),
                })
        
        if size_band_stats:
            sb_df = pd.DataFrame(size_band_stats)
            sb_df.to_csv(f'{TBL_DIR}/T13_size_band_analysis.csv', index=False)
            print("\nSize band analysis:")
            for _, row in sb_df.iterrows():
                print(f"  {row['size_band']:<10} companies={row['n_companies']:>3}  "
                      f"entry: {100*row['entry_arshkon']:.0f}%->{100*row['entry_scraped']:.0f}%  "
                      f"AI: {100*row['ai_arshkon']:.0f}%->{100*row['ai_scraped']:.0f}%  "
                      f"desc: {row['desc_len_arshkon']:.0f}->{row['desc_len_scraped']:.0f}")
sys.stdout.flush()

# ------------------------------------------------------------------
# Step 6: Figures
# ------------------------------------------------------------------
print("\n\nStep 6: Generating figures...")
sys.stdout.flush()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Figure 1: Seniority distribution comparison — overlapping vs all
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

seniority_order = ['entry', 'associate', 'mid-senior', 'director', 'unknown']
colors_sen = {'entry': '#FF6B6B', 'associate': '#4ECDC4', 'mid-senior': '#45B7D1', 
              'director': '#96CEB4', 'unknown': '#CCCCCC'}

# Left: Overlapping companies
ax = axes[0]
for idx, src in enumerate(['kaggle_arshkon', 'scraped']):
    subset = overlap_rows[overlap_rows['source'] == src]
    dist = subset['seniority_patched'].value_counts(normalize=True)
    dist = dist.reindex(seniority_order, fill_value=0)
    x = np.arange(len(seniority_order))
    width = 0.35
    offset = -width/2 if idx == 0 else width/2
    label = 'Arshkon 2024' if idx == 0 else 'Scraped 2026'
    bars = ax.bar(x + offset, dist.values * 100, width, label=label, 
                  alpha=0.8, color=['#4ECDC4', '#FF6B6B'][idx])

ax.set_xticks(x)
ax.set_xticklabels([s.title() for s in seniority_order], rotation=30, ha='right')
ax.set_ylabel('% of SWE Postings')
ax.set_title(f'Overlapping Companies (n={len(overlap)})', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Right: All companies
ax = axes[1]
for idx, src in enumerate(['kaggle_arshkon', 'scraped']):
    subset = swe_df[swe_df['source'] == src]
    dist = subset['seniority_patched'].value_counts(normalize=True)
    dist = dist.reindex(seniority_order, fill_value=0)
    x = np.arange(len(seniority_order))
    width = 0.35
    offset = -width/2 if idx == 0 else width/2
    label = 'Arshkon 2024' if idx == 0 else 'Scraped 2026'
    bars = ax.bar(x + offset, dist.values * 100, width, label=label,
                  alpha=0.8, color=['#4ECDC4', '#FF6B6B'][idx])

ax.set_xticks(x)
ax.set_xticklabels([s.title() for s in seniority_order], rotation=30, ha='right')
ax.set_ylabel('% of SWE Postings')
ax.set_title('All Companies', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

fig.suptitle('Seniority Distribution: Overlapping vs All Companies (SWE)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/T13_seniority_overlap_vs_all.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: T13_seniority_overlap_vs_all.png")

# Figure 2: Within-company AI keyword change scatter
fig, ax = plt.subplots(figsize=(8, 8))

# Only companies with >= 3 postings in each period
plot_df = company_df[(company_df['n_arshkon'] >= 3) & (company_df['n_scraped'] >= 3)]
print(f"\nCompanies with >=3 postings in both periods: {len(plot_df)}")

scatter = ax.scatter(plot_df['ai_arshkon'] * 100, plot_df['ai_scraped'] * 100, 
                     s=np.sqrt(plot_df['n_total']) * 10, alpha=0.5, c='#45B7D1', edgecolors='gray', linewidth=0.5)

# Diagonal line
ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1)
ax.set_xlabel('AI Keyword % in 2024 (arshkon)', fontsize=11)
ax.set_ylabel('AI Keyword % in 2026 (scraped)', fontsize=11)
ax.set_title(f'Within-Company AI Keyword Change\n(n={len(plot_df)} companies, >=3 postings each)', fontsize=13, fontweight='bold')
ax.set_xlim(-5, 105)
ax.set_ylim(-5, 105)
ax.grid(alpha=0.3)

# Annotate a few outliers
for _, row in plot_df.nlargest(5, 'n_total').iterrows():
    ax.annotate(row['company'][:20], (row['ai_arshkon']*100, row['ai_scraped']*100),
               fontsize=7, alpha=0.7, xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/T13_ai_keyword_within_company.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: T13_ai_keyword_within_company.png")

# Figure 3: Decomposition bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Entry share decomposition
ax = axes[0]
bars = ax.bar(['Overall\nDelta', 'Within-\nCompany', 'Composition\nEffect'],
              [100*overall_delta, 100*within_delta, 100*composition_delta],
              color=['#45B7D1', '#4ECDC4', '#FF6B6B'], alpha=0.8)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:+.1f}pp', xy=(bar.get_x() + bar.get_width()/2, height),
               xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Change in Entry-Level Share (pp)')
ax.set_title('Entry-Level Share', fontsize=12, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)

# AI keyword decomposition
ax = axes[1]
bars = ax.bar(['Overall\nDelta', 'Within-\nCompany', 'Composition\nEffect'],
              [100*overall_ai_delta, 100*within_ai_delta, 100*composition_ai_delta],
              color=['#45B7D1', '#4ECDC4', '#FF6B6B'], alpha=0.8)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:+.1f}pp', xy=(bar.get_x() + bar.get_width()/2, height),
               xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Change in AI Keyword % (pp)')
ax.set_title('AI Keyword Prevalence', fontsize=12, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)

# Description length decomposition
desc_overall_delta = overall_len_scraped / overall_len_arshkon - 1
desc_within_delta = within_len_scraped / within_len_arshkon - 1
desc_composition = desc_overall_delta - desc_within_delta

ax = axes[2]
bars = ax.bar(['Overall\nChange', 'Within-\nCompany', 'Composition\nEffect'],
              [100*desc_overall_delta, 100*desc_within_delta, 100*desc_composition],
              color=['#45B7D1', '#4ECDC4', '#FF6B6B'], alpha=0.8)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
               xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Change in Median Description Length (%)')
ax.set_title('Description Length', fontsize=12, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)

fig.suptitle('Within-Company vs Composition Effects (arshkon 2024 to scraped 2026, SWE)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/T13_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: T13_decomposition.png")

# Figure 4: New vs departed company characteristics
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Seniority distribution
ax = axes[0]
groups_plot = {
    'Overlap\n(arshkon)': overlap_rows[overlap_rows['source'] == 'kaggle_arshkon'],
    'Overlap\n(scraped)': overlap_rows[overlap_rows['source'] == 'scraped'],
    'Arshkon-\nonly': arshkon_only_rows,
    'Scraped-\nonly': scraped_only_rows,
}

x = np.arange(len(seniority_order))
width = 0.2
for i, (gname, gdf) in enumerate(groups_plot.items()):
    if len(gdf) == 0:
        continue
    dist = gdf['seniority_patched'].value_counts(normalize=True)
    dist = dist.reindex(seniority_order, fill_value=0)
    ax.bar(x + (i - 1.5) * width, dist.values * 100, width, label=gname, alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels([s.title() for s in seniority_order], rotation=30, ha='right')
ax.set_ylabel('% of Postings')
ax.set_title('Seniority by Company Group', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# Right: AI keyword prevalence
ax = axes[1]
ai_rates = []
labels = []
for gname, gdf in groups_plot.items():
    if len(gdf) > 0:
        ai_rates.append(gdf['has_ai'].mean() * 100)
        labels.append(gname)

bars = ax.bar(labels, ai_rates, color=['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4'], alpha=0.8)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
               xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('% of Postings with AI Keywords')
ax.set_title('AI Keyword Rate by Company Group', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

fig.suptitle('New vs Continuing vs Departed Companies (SWE)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/T13_new_vs_departed.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: T13_new_vs_departed.png")

# ------------------------------------------------------------------
# Save decomposition summary
# ------------------------------------------------------------------
decomposition = pd.DataFrame([
    {'metric': 'Entry-level share', 'overall_2024': overall_entry_arshkon, 'overall_2026': overall_entry_scraped,
     'overall_delta': overall_delta, 'within_company_delta': within_delta, 'composition_effect': composition_delta},
    {'metric': 'AI keyword prevalence', 'overall_2024': overall_ai_arshkon, 'overall_2026': overall_ai_scraped,
     'overall_delta': overall_ai_delta, 'within_company_delta': within_ai_delta, 'composition_effect': composition_ai_delta},
    {'metric': 'Median desc length', 'overall_2024': overall_len_arshkon, 'overall_2026': overall_len_scraped,
     'overall_delta': overall_len_scraped - overall_len_arshkon, 
     'within_company_delta': within_len_scraped - within_len_arshkon,
     'composition_effect': (overall_len_scraped - overall_len_arshkon) - (within_len_scraped - within_len_arshkon)},
])
decomposition.to_csv(f'{TBL_DIR}/T13_decomposition_summary.csv', index=False)

# Save new/departed company lists
scraped_only_top.to_csv(f'{TBL_DIR}/T13_scraped_only_top_companies.csv')
arshkon_only_top.to_csv(f'{TBL_DIR}/T13_arshkon_only_top_companies.csv')

print("\n\n=== FINAL SUMMARY ===")
print(f"Overlapping companies: {len(overlap)}")
print(f"Arshkon-only: {len(arshkon_only)}")
print(f"Scraped-only: {len(scraped_only)}")
print(f"\nOverlap rows: arshkon={len(arshkon_overlap)}, scraped={len(scraped_overlap)}")
print(f"Overlap as % of total: arshkon={100*len(arshkon_overlap)/len(arshkon_all):.1f}%, scraped={100*len(scraped_overlap)/len(scraped_all):.1f}%")
print(f"\nEntry share decomposition: overall={100*overall_delta:+.1f}pp, within={100*within_delta:+.1f}pp, composition={100*composition_delta:+.1f}pp")
print(f"AI keyword decomposition: overall={100*overall_ai_delta:+.1f}pp, within={100*within_ai_delta:+.1f}pp, composition={100*composition_ai_delta:+.1f}pp")

print("\nDone!")
