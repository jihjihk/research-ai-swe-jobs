#!/usr/bin/env python3
"""
T12: Requirements parsing — Extract structured data from job description requirements sections.

Extracts: YOE, education level, tech/tool count, soft skill / management language.
Analyzes by seniority_patched x period. Junior 2024 vs junior 2026 comparison.
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
FIG_DIR = 'exploration/figures/T12'
TBL_DIR = 'exploration/tables/T12'

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Step 0: Load SWE descriptions with seniority_patched via workaround
# ------------------------------------------------------------------
con = duckdb.connect()

print("Loading SWE descriptions with seniority workaround...")
sys.stdout.flush()

df = con.sql(f"""
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
SELECT uid, source, period, seniority_patched, seniority_3level_patched,
       description, description_length
FROM swe_patched
WHERE is_swe = true
""").df()

print(f"Loaded {len(df)} SWE rows")
print(f"By source: {df['source'].value_counts().to_dict()}")
print(f"By seniority_patched: {df['seniority_patched'].value_counts().to_dict()}")
sys.stdout.flush()

# ------------------------------------------------------------------
# Step 1: Requirements section extraction
# ------------------------------------------------------------------
print("\nStep 1: Extracting requirements sections...")
sys.stdout.flush()

# Pattern to identify requirements section headers
REQ_HEADERS = re.compile(
    r'(?:^|\n)\s*\**\s*'
    r'(?:'
    r'(?:basic\s+|minimum\s+|required\s+)?qualifications'
    r'|requirements'
    r'|what\s+you(?:\'ll|\s+will)\s+(?:need|bring)'
    r'|what\s+we(?:\'re|\s+are)\s+looking\s+for'
    r'|who\s+you\s+are'
    r'|must[\s-]have'
    r'|skills?\s+(?:and\s+)?(?:experience|requirements?)'
    r'|you\s+(?:should\s+)?have'
    r'|ideal\s+candidate'
    r'|about\s+you'
    r'|your\s+background'
    r'|key\s+(?:skills|requirements|qualifications)'
    r'|we\'re\s+looking\s+for'
    r'|you\s+bring'
    r'|what\s+you\s+(?:should\s+)?(?:know|have)'
    r')'
    r'\s*:?\s*\**',
    re.IGNORECASE
)

# Sections that typically follow requirements (to know where to stop)
NEXT_HEADERS = re.compile(
    r'(?:^|\n)\s*\**\s*'
    r'(?:'
    r'(?:preferred|nice[\s-]to[\s-]have|bonus|additional|plus|desired)\s+(?:qualifications|skills|experience)'
    r'|benefits'
    r'|compensation'
    r'|what\s+we\s+offer'
    r'|perks'
    r'|about\s+(?:us|the\s+(?:company|team|role))'
    r'|why\s+(?:join|work)'
    r'|our\s+(?:values|culture|mission)'
    r'|equal\s+(?:opportunity|employment)'
    r'|eeo'
    r'|diversity'
    r'|how\s+to\s+apply'
    r'|bonus\s+points'
    r'|additional\s+information'
    r'|responsibilities'
    r'|what\s+you(?:\'ll|\s+will)\s+(?:do|work\s+on)'
    r'|core\s+responsibilities'
    r')'
    r'\s*:?\s*\**',
    re.IGNORECASE
)

def extract_requirements_section(text):
    """Extract the requirements/qualifications section from a job description."""
    if not text:
        return None
    
    # Find the start of requirements
    match = REQ_HEADERS.search(text)
    if not match:
        return None
    
    start = match.end()
    remaining = text[start:]
    
    # Find the next non-requirements header
    next_match = NEXT_HEADERS.search(remaining)
    if next_match:
        return remaining[:next_match.start()].strip()
    
    # If no next header, take up to 3000 chars or end of text
    return remaining[:3000].strip()


# ------------------------------------------------------------------
# Step 2: Feature extraction functions
# ------------------------------------------------------------------

# YOE extraction
YOE_PATTERNS = [
    # "5+ years of experience", "5+ YOE", "5+ years' experience"
    re.compile(r'(\d+)\+?\s*(?:years?|yrs?|yr)\s*(?:of\s+)?(?:experience|exp|professional)', re.IGNORECASE),
    # "5-7 years", "5 to 7 years"
    re.compile(r'(\d+)\s*[-–to]+\s*(\d+)\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp|professional)?', re.IGNORECASE),
    # "5+ YOE"
    re.compile(r'(\d+)\+?\s*YOE', re.IGNORECASE),
    # "at least 5 years"
    re.compile(r'(?:at\s+least|minimum(?:\s+of)?)\s+(\d+)\s*(?:years?|yrs?)', re.IGNORECASE),
    # "experience of 5+ years"
    re.compile(r'experience\s+(?:of\s+)?(\d+)\+?\s*(?:years?|yrs?)', re.IGNORECASE),
]

def extract_yoe(text):
    """Extract minimum years of experience from text. Returns int or None."""
    if not text:
        return None
    
    all_yoe = []
    for pat in YOE_PATTERNS:
        for m in pat.finditer(text):
            try:
                val = int(m.group(1))
                if 0 <= val <= 30:  # sanity check
                    all_yoe.append(val)
            except (ValueError, IndexError):
                pass
    
    if all_yoe:
        return min(all_yoe)  # Take minimum as the floor requirement
    return None

# Education extraction
EDU_PATTERNS = {
    'phd': re.compile(r'\b(?:ph\.?d|doctorate|doctoral)\b', re.IGNORECASE),
    'masters': re.compile(r'\b(?:master\'?s?|m\.?s\.?|m\.?eng\.?|mba|m\.?a\.?)\s*(?:degree|in\s+|or\b|\/)', re.IGNORECASE),
    'bachelors': re.compile(r'\b(?:bachelor\'?s?|b\.?s\.?|b\.?a\.?|b\.?eng\.?)\s*(?:degree|in\s+|or\b|\/|\s+(?:in|from))', re.IGNORECASE),
    'bachelors_alt': re.compile(r'\b(?:bachelor\'?s?|undergraduate)\s+degree\b', re.IGNORECASE),
    'degree_generic': re.compile(r'\bdegree\s+in\s+(?:computer|software|engineering|science|math|information|electrical)', re.IGNORECASE),
}

def extract_education(text):
    """Extract highest education level mentioned. Returns 'phd', 'masters', 'bachelors', or None."""
    if not text:
        return None
    
    if EDU_PATTERNS['phd'].search(text):
        return 'phd'
    if EDU_PATTERNS['masters'].search(text):
        return 'masters'
    if EDU_PATTERNS['bachelors'].search(text) or EDU_PATTERNS['bachelors_alt'].search(text):
        return 'bachelors'
    if EDU_PATTERNS['degree_generic'].search(text):
        return 'bachelors'
    return None

# Tech/tool extraction
TECH_KEYWORDS = [
    # Languages
    'python', 'java', 'javascript', 'typescript', 'c\+\+', 'c#', 'go', 'golang',
    'rust', 'ruby', 'scala', 'kotlin', 'swift', 'php', 'perl', 'r\b', 'matlab',
    'sql', 'html', 'css', 'bash', 'shell', 'powershell',
    # Frameworks
    'react', 'angular', 'vue', 'django', 'flask', 'spring', 'node\.?js', 'next\.?js',
    'express', '.net', 'rails', 'laravel', 'fastapi', 'graphql', 'rest(?:ful)?',
    # Cloud/Infra
    'aws', 'azure', 'gcp', 'google\s+cloud', 'terraform', 'kubernetes', 'k8s',
    'docker', 'jenkins', 'ci/?cd', 'ansible', 'puppet', 'chef',
    'cloudformation', 'helm', 'argocd', 'datadog', 'splunk', 'grafana',
    # Data
    'spark', 'kafka', 'hadoop', 'airflow', 'snowflake', 'databricks', 'dbt',
    'redshift', 'bigquery', 'elasticsearch', 'redis', 'mongodb', 'postgresql',
    'postgres', 'mysql', 'dynamodb', 'cassandra', 'neo4j', 'flink',
    # ML/AI
    'tensorflow', 'pytorch', 'scikit[\s-]learn', 'keras', 'hugging\s*face',
    'langchain', 'openai', 'llm', 'machine\s+learning', 'deep\s+learning',
    'nlp', 'computer\s+vision', 'mlops',
    # DevOps/Tools
    'git', 'jira', 'confluence', 'linux', 'unix', 'agile', 'scrum',
    'microservices', 'serverless', 'lambda',
]

TECH_PATTERN = re.compile(
    r'\b(?:' + '|'.join(TECH_KEYWORDS) + r')\b',
    re.IGNORECASE
)

def count_distinct_techs(text):
    """Count distinct technologies/tools mentioned in text."""
    if not text:
        return 0
    
    matches = set()
    for m in TECH_PATTERN.finditer(text):
        # Normalize match to lowercase for dedup
        normalized = m.group(0).lower().strip()
        # Merge some synonyms
        if normalized in ('golang',): normalized = 'go'
        if normalized in ('k8s',): normalized = 'kubernetes'
        if normalized in ('postgres',): normalized = 'postgresql'
        if normalized in ('node.js', 'nodejs'): normalized = 'node'
        if normalized in ('next.js', 'nextjs'): normalized = 'next'
        if normalized in ('google cloud',): normalized = 'gcp'
        matches.add(normalized)
    
    return len(matches)

# Soft skill / management language
SOFT_SKILL_TERMS = {
    'cross_functional': re.compile(r'\bcross[\s-]?functional\b', re.IGNORECASE),
    'stakeholder': re.compile(r'\bstakeholder\b', re.IGNORECASE),
    'ownership': re.compile(r'\b(?:ownership|take\s+ownership|end[\s-]to[\s-]end\s+ownership)\b', re.IGNORECASE),
    'end_to_end': re.compile(r'\bend[\s-]to[\s-]end\b', re.IGNORECASE),
    'mentoring': re.compile(r'\b(?:mentor(?:ing|ship)?|coach(?:ing)?)\b', re.IGNORECASE),
    'leadership': re.compile(r'\b(?:leader(?:ship)?|lead(?:ing)?\s+(?:a\s+)?team)\b', re.IGNORECASE),
    'communication': re.compile(r'\b(?:communication|communicat(?:e|ing)|written\s+and\s+verbal)\b', re.IGNORECASE),
    'collaboration': re.compile(r'\b(?:collaborat(?:e|ion|ing|ive)|teamwork|team\s+player)\b', re.IGNORECASE),
    'influence': re.compile(r'\b(?:influence|influencing)\b', re.IGNORECASE),
    'strategic': re.compile(r'\b(?:strategic|strategy)\b', re.IGNORECASE),
    'autonomy': re.compile(r'\b(?:autonom(?:y|ous(?:ly)?)|self[\s-]?directed|independently|self[\s-]?starter)\b', re.IGNORECASE),
}

# AI-related keywords (for T13 crossover but useful here too)
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

def extract_soft_skills(text):
    """Extract soft skill / management language flags from text."""
    if not text:
        return {}
    return {k: bool(v.search(text)) for k, v in SOFT_SKILL_TERMS.items()}

def has_ai_keywords(text):
    """Check if text mentions AI-related keywords."""
    if not text:
        return False
    return bool(AI_KEYWORDS.search(text))

# ------------------------------------------------------------------
# Step 3: Apply extraction to all rows
# ------------------------------------------------------------------
print("\nStep 2: Applying extraction to all SWE rows...")
sys.stdout.flush()

results = []
req_section_found = 0

for i, row in df.iterrows():
    desc = row['description'] if pd.notna(row['description']) else ''
    
    # Extract requirements section
    req_section = extract_requirements_section(desc)
    has_req = req_section is not None
    if has_req:
        req_section_found += 1
    
    # Use requirements section if found; otherwise use full description
    # For YOE and education, search full description as these may appear anywhere
    text_for_tech = req_section if req_section else desc
    
    yoe = extract_yoe(desc)  # Search full description for YOE
    edu = extract_education(desc)  # Search full description for education
    tech_count = count_distinct_techs(text_for_tech)  # Tech in requirements section preferred
    tech_count_full = count_distinct_techs(desc)  # Also get full-description count
    soft = extract_soft_skills(desc)  # Search full description for soft skills
    has_ai = has_ai_keywords(desc)
    
    r = {
        'uid': row['uid'],
        'source': row['source'],
        'period': row['period'],
        'seniority_patched': row['seniority_patched'],
        'seniority_3level_patched': row['seniority_3level_patched'],
        'description_length': row['description_length'],
        'has_req_section': has_req,
        'yoe_min': yoe,
        'education': edu,
        'tech_count': tech_count_full,  # Use full desc for comparability
        'has_ai_keywords': has_ai,
    }
    r.update({f'soft_{k}': v for k, v in soft.items()})
    results.append(r)
    
    if (i + 1) % 5000 == 0:
        print(f"  Processed {i+1}/{len(df)} rows...")
        sys.stdout.flush()

print(f"\nExtraction complete. Requirements section found: {req_section_found}/{len(df)} ({100*req_section_found/len(df):.1f}%)")
sys.stdout.flush()

rdf = pd.DataFrame(results)

# ------------------------------------------------------------------
# Step 4: Summary statistics by seniority_patched x period
# ------------------------------------------------------------------
print("\nStep 3: Computing summary statistics...")
sys.stdout.flush()

# Filter to non-unknown seniority for seniority-stratified analysis
rdf_known = rdf[rdf['seniority_patched'] != 'unknown'].copy()

# Map periods to labels
period_map = {'2024-01': 'asaniczka\n(Jan 2024)', '2024-04': 'arshkon\n(Apr 2024)', '2026-03': 'scraped\n(Mar 2026)'}
period_order = ['2024-01', '2024-04', '2026-03']
seniority_order = ['entry', 'associate', 'mid-senior', 'director']

# Compute per seniority x period
summary_rows = []
for sen in seniority_order:
    for per in period_order:
        mask = (rdf_known['seniority_patched'] == sen) & (rdf_known['period'] == per)
        subset = rdf_known[mask]
        n = len(subset)
        if n == 0:
            continue
        
        median_yoe = subset['yoe_min'].median()
        yoe_p25 = subset['yoe_min'].quantile(0.25) if subset['yoe_min'].notna().sum() > 0 else np.nan
        yoe_p75 = subset['yoe_min'].quantile(0.75) if subset['yoe_min'].notna().sum() > 0 else np.nan
        yoe_coverage = subset['yoe_min'].notna().mean()
        
        pct_bachelors = (subset['education'] == 'bachelors').mean()
        pct_masters = (subset['education'].isin(['masters', 'phd'])).mean()
        pct_phd = (subset['education'] == 'phd').mean()
        edu_coverage = subset['education'].notna().mean()
        
        median_tech = subset['tech_count'].median()
        mean_tech = subset['tech_count'].mean()
        
        # Normalize tech count by description length (per 1000 chars)
        subset_with_len = subset[subset['description_length'] > 0]
        if len(subset_with_len) > 0:
            tech_density = (subset_with_len['tech_count'] / (subset_with_len['description_length'] / 1000)).median()
        else:
            tech_density = np.nan
        
        pct_cross_func = subset['soft_cross_functional'].mean()
        pct_stakeholder = subset['soft_stakeholder'].mean()
        pct_ownership = subset['soft_ownership'].mean()
        pct_end_to_end = subset['soft_end_to_end'].mean()
        pct_mentoring = subset['soft_mentoring'].mean()
        pct_leadership = subset['soft_leadership'].mean()
        pct_communication = subset['soft_communication'].mean()
        pct_collaboration = subset['soft_collaboration'].mean()
        pct_autonomy = subset['soft_autonomy'].mean()
        pct_ai = subset['has_ai_keywords'].mean()
        
        median_desc_len = subset['description_length'].median()
        
        summary_rows.append({
            'seniority_patched': sen,
            'period': per,
            'n': n,
            'median_yoe': median_yoe,
            'yoe_p25': yoe_p25,
            'yoe_p75': yoe_p75,
            'yoe_coverage': yoe_coverage,
            'pct_bachelors_only': pct_bachelors,
            'pct_ms_or_phd': pct_masters,
            'pct_phd': pct_phd,
            'edu_coverage': edu_coverage,
            'median_tech_count': median_tech,
            'mean_tech_count': mean_tech,
            'tech_density_per_1k_chars': tech_density,
            'pct_cross_functional': pct_cross_func,
            'pct_stakeholder': pct_stakeholder,
            'pct_ownership': pct_ownership,
            'pct_end_to_end': pct_end_to_end,
            'pct_mentoring': pct_mentoring,
            'pct_leadership': pct_leadership,
            'pct_communication': pct_communication,
            'pct_collaboration': pct_collaboration,
            'pct_autonomy': pct_autonomy,
            'pct_ai_keywords': pct_ai,
            'median_desc_length': median_desc_len,
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(f'{TBL_DIR}/T12_seniority_period_summary.csv', index=False)
print(f"\nSaved summary table: {len(summary)} rows")
print(summary[['seniority_patched', 'period', 'n', 'median_yoe', 'pct_ms_or_phd', 
               'median_tech_count', 'pct_cross_functional', 'pct_stakeholder',
               'pct_ownership', 'pct_end_to_end', 'pct_ai_keywords']].to_string(index=False))
sys.stdout.flush()

# ------------------------------------------------------------------
# Step 5: Junior 2024 vs Junior 2026 comparison
# ------------------------------------------------------------------
print("\n\nStep 4: Junior 2024 vs Junior 2026 comparison...")
sys.stdout.flush()

# Use 3level for larger samples
junior_24 = rdf[(rdf['seniority_3level_patched'] == 'junior') & (rdf['period'].isin(['2024-01', '2024-04']))]
junior_26 = rdf[(rdf['seniority_3level_patched'] == 'junior') & (rdf['period'] == '2026-03')]

# Also by source for 5-level
entry_arshkon = rdf[(rdf['seniority_patched'] == 'entry') & (rdf['source'] == 'kaggle_arshkon')]
entry_scraped = rdf[(rdf['seniority_patched'] == 'entry') & (rdf['source'] == 'scraped')]

print(f"\nJunior (3-level) 2024: {len(junior_24)}")
print(f"Junior (3-level) 2026: {len(junior_26)}")
print(f"Entry (5-level) arshkon: {len(entry_arshkon)}")
print(f"Entry (5-level) scraped: {len(entry_scraped)}")

# Comparison function
def compare_groups(g1, g2, label1, label2):
    metrics = {}
    for col, desc in [
        ('yoe_min', 'Median YOE'),
        ('tech_count', 'Median tech count'),
    ]:
        v1 = g1[col].median()
        v2 = g2[col].median()
        metrics[desc] = (v1, v2)
    
    for col, desc in [
        ('soft_cross_functional', '% cross-functional'),
        ('soft_stakeholder', '% stakeholder'),
        ('soft_ownership', '% ownership'),
        ('soft_end_to_end', '% end-to-end'),
        ('soft_mentoring', '% mentoring'),
        ('soft_leadership', '% leadership'),
        ('soft_communication', '% communication'),
        ('soft_collaboration', '% collaboration'),
        ('soft_autonomy', '% autonomy/self-directed'),
        ('has_ai_keywords', '% AI keywords'),
    ]:
        v1 = g1[col].mean()
        v2 = g2[col].mean()
        metrics[desc] = (v1, v2)
    
    # Education
    metrics['% MS/PhD'] = (
        g1['education'].isin(['masters', 'phd']).mean(),
        g2['education'].isin(['masters', 'phd']).mean()
    )
    
    # Normalized tech density
    g1_wl = g1[g1['description_length'] > 0]
    g2_wl = g2[g2['description_length'] > 0]
    metrics['Tech density (per 1k chars)'] = (
        (g1_wl['tech_count'] / (g1_wl['description_length'] / 1000)).median() if len(g1_wl) > 0 else np.nan,
        (g2_wl['tech_count'] / (g2_wl['description_length'] / 1000)).median() if len(g2_wl) > 0 else np.nan,
    )
    
    print(f"\n{'Metric':<35} {label1:>12} {label2:>12} {'Delta':>10}")
    print("-" * 72)
    for desc, (v1, v2) in metrics.items():
        if 'Median' in desc or 'density' in desc:
            delta = f"{v2 - v1:+.1f}" if pd.notna(v1) and pd.notna(v2) else "N/A"
            print(f"{desc:<35} {v1:>12.1f} {v2:>12.1f} {delta:>10}")
        else:
            delta = f"{100*(v2-v1):+.1f}pp" if pd.notna(v1) and pd.notna(v2) else "N/A"
            print(f"{desc:<35} {100*v1:>11.1f}% {100*v2:>11.1f}% {delta:>10}")
    
    return metrics

print("\n--- Junior (3-level) 2024 vs 2026 ---")
jr_metrics = compare_groups(junior_24, junior_26, 'Jr 2024', 'Jr 2026')

print("\n--- Entry (5-level) arshkon vs scraped ---")
entry_metrics = compare_groups(entry_arshkon, entry_scraped, 'Arshkon', 'Scraped')
sys.stdout.flush()

# Save junior comparison as CSV
jr_comparison = []
for desc, (v1, v2) in jr_metrics.items():
    jr_comparison.append({
        'metric': desc,
        'junior_2024': v1,
        'junior_2026': v2,
        'delta': v2 - v1 if pd.notna(v1) and pd.notna(v2) else np.nan
    })
pd.DataFrame(jr_comparison).to_csv(f'{TBL_DIR}/T12_junior_comparison.csv', index=False)

entry_comparison = []
for desc, (v1, v2) in entry_metrics.items():
    entry_comparison.append({
        'metric': desc,
        'entry_arshkon': v1,
        'entry_scraped': v2,
        'delta': v2 - v1 if pd.notna(v1) and pd.notna(v2) else np.nan
    })
pd.DataFrame(entry_comparison).to_csv(f'{TBL_DIR}/T12_entry_comparison.csv', index=False)

# ------------------------------------------------------------------
# Step 6: Also compute for mid-senior to see if changes are SWE-wide
# ------------------------------------------------------------------
print("\n\n--- Mid-Senior 2024 vs 2026 (control comparison) ---")
midsen_24 = rdf[(rdf['seniority_patched'] == 'mid-senior') & (rdf['period'].isin(['2024-01', '2024-04']))]
midsen_26 = rdf[(rdf['seniority_patched'] == 'mid-senior') & (rdf['period'] == '2026-03')]
print(f"Mid-senior 2024: {len(midsen_24)}, 2026: {len(midsen_26)}")
ms_metrics = compare_groups(midsen_24, midsen_26, 'MS 2024', 'MS 2026')
sys.stdout.flush()

# ------------------------------------------------------------------
# Step 7: Figures
# ------------------------------------------------------------------
print("\n\nStep 5: Generating figures...")
sys.stdout.flush()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Figure 1: YOE by seniority x period
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

for idx, sen in enumerate(['entry', 'associate', 'mid-senior']):
    ax = axes[idx]
    data_by_period = []
    labels = []
    for per in period_order:
        subset = rdf_known[(rdf_known['seniority_patched'] == sen) & (rdf_known['period'] == per)]
        yoe_vals = subset['yoe_min'].dropna()
        if len(yoe_vals) > 5:
            data_by_period.append(yoe_vals.values)
            source_label = {'2024-01': 'asaniczka\n(Jan 24)', '2024-04': 'arshkon\n(Apr 24)', '2026-03': 'scraped\n(Mar 26)'}[per]
            labels.append(f"{source_label}\nn={len(yoe_vals)}")
    
    if data_by_period:
        bp = ax.boxplot(data_by_period, labels=labels, patch_artist=True, 
                       widths=0.5, showfliers=False)
        colors = ['#4ECDC4', '#45B7D1', '#FF6B6B']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax.set_title(f'{sen.title()}', fontsize=13, fontweight='bold')
    ax.set_ylabel('Min YOE Stated' if idx == 0 else '')
    ax.set_ylim(-0.5, 15)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Minimum Years of Experience by Seniority and Period (SWE)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/T12_yoe_by_seniority_period.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: T12_yoe_by_seniority_period.png")

# Figure 2: Soft skill prevalence comparison for junior roles
fig, ax = plt.subplots(figsize=(12, 6))

soft_cols = ['soft_cross_functional', 'soft_stakeholder', 'soft_ownership', 
             'soft_end_to_end', 'soft_mentoring', 'soft_leadership',
             'soft_communication', 'soft_collaboration', 'soft_autonomy']
soft_labels = ['Cross-functional', 'Stakeholder', 'Ownership', 'End-to-end',
               'Mentoring', 'Leadership', 'Communication', 'Collaboration', 'Autonomy']

# Compare junior 2024 vs 2026
x = np.arange(len(soft_labels))
width = 0.35

jr24_vals = [junior_24[c].mean() * 100 for c in soft_cols]
jr26_vals = [junior_26[c].mean() * 100 for c in soft_cols]

bars1 = ax.bar(x - width/2, jr24_vals, width, label=f'Junior 2024 (n={len(junior_24)})', color='#4ECDC4', alpha=0.8)
bars2 = ax.bar(x + width/2, jr26_vals, width, label=f'Junior 2026 (n={len(junior_26)})', color='#FF6B6B', alpha=0.8)

ax.set_ylabel('% of Postings Mentioning', fontsize=11)
ax.set_title('Soft Skill / Management Language in Junior SWE Roles: 2024 vs 2026', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(soft_labels, rotation=30, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 1:
            ax.annotate(f'{height:.0f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/T12_soft_skills_junior_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: T12_soft_skills_junior_comparison.png")

# Figure 3: Tech count + AI keywords by seniority x period
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Tech density (normalized by description length)
ax = axes[0]
for i, sen in enumerate(['entry', 'associate', 'mid-senior']):
    densities = []
    periods_used = []
    for per in period_order:
        subset = rdf_known[(rdf_known['seniority_patched'] == sen) & (rdf_known['period'] == per)]
        subset_wl = subset[subset['description_length'] > 0]
        if len(subset_wl) > 5:
            density = (subset_wl['tech_count'] / (subset_wl['description_length'] / 1000)).median()
            densities.append(density)
            periods_used.append(per)
    if densities:
        ax.plot(range(len(densities)), densities, 'o-', label=sen.title(), markersize=8, linewidth=2)

ax.set_xticks(range(len(period_order)))
ax.set_xticklabels(['Jan 2024', 'Apr 2024', 'Mar 2026'])
ax.set_ylabel('Median Tech Keywords per 1K Chars')
ax.set_title('Tech Density (Length-Normalized)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# AI keyword prevalence
ax = axes[1]
for i, sen in enumerate(['entry', 'associate', 'mid-senior']):
    ai_rates = []
    for per in period_order:
        subset = rdf_known[(rdf_known['seniority_patched'] == sen) & (rdf_known['period'] == per)]
        if len(subset) > 5:
            ai_rates.append(subset['has_ai_keywords'].mean() * 100)
    if ai_rates:
        ax.plot(range(len(ai_rates)), ai_rates, 'o-', label=sen.title(), markersize=8, linewidth=2)

ax.set_xticks(range(len(period_order)))
ax.set_xticklabels(['Jan 2024', 'Apr 2024', 'Mar 2026'])
ax.set_ylabel('% of Postings Mentioning AI')
ax.set_title('AI Keyword Prevalence', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

fig.suptitle('Technology Requirements by Seniority and Period (SWE)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/T12_tech_and_ai_by_seniority.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: T12_tech_and_ai_by_seniority.png")

# Figure 4: Scope inflation heatmap — delta (2026 - 2024) for key metrics
fig, ax = plt.subplots(figsize=(10, 5))

metrics_for_heatmap = [
    ('Median YOE', 'yoe_min', 'median'),
    ('% MS/PhD', 'education', 'ms_pct'),
    ('Tech density', 'tech_count', 'density'),
    ('% Cross-functional', 'soft_cross_functional', 'pct'),
    ('% Stakeholder', 'soft_stakeholder', 'pct'),
    ('% Ownership', 'soft_ownership', 'pct'),
    ('% End-to-end', 'soft_end_to_end', 'pct'),
    ('% Mentoring', 'soft_mentoring', 'pct'),
    ('% Leadership', 'soft_leadership', 'pct'),
    ('% AI keywords', 'has_ai_keywords', 'pct'),
]

delta_data = []
sen_labels = ['entry', 'associate', 'mid-senior']
for sen in sen_labels:
    row_data = []
    # Compare average of 2024 periods vs 2026
    mask_24 = (rdf_known['seniority_patched'] == sen) & (rdf_known['period'].isin(['2024-01', '2024-04']))
    mask_26 = (rdf_known['seniority_patched'] == sen) & (rdf_known['period'] == '2026-03')
    g24 = rdf_known[mask_24]
    g26 = rdf_known[mask_26]
    
    for label, col, agg_type in metrics_for_heatmap:
        if agg_type == 'median':
            v24 = g24[col].median()
            v26 = g26[col].median()
            delta = v26 - v24 if pd.notna(v24) and pd.notna(v26) else np.nan
        elif agg_type == 'ms_pct':
            v24 = g24['education'].isin(['masters', 'phd']).mean() * 100
            v26 = g26['education'].isin(['masters', 'phd']).mean() * 100
            delta = v26 - v24
        elif agg_type == 'density':
            g24_wl = g24[g24['description_length'] > 0]
            g26_wl = g26[g26['description_length'] > 0]
            v24 = (g24_wl['tech_count'] / (g24_wl['description_length'] / 1000)).median() if len(g24_wl) > 5 else np.nan
            v26 = (g26_wl['tech_count'] / (g26_wl['description_length'] / 1000)).median() if len(g26_wl) > 5 else np.nan
            delta = v26 - v24 if pd.notna(v24) and pd.notna(v26) else np.nan
        else:  # pct
            v24 = g24[col].mean() * 100 if len(g24) > 0 else np.nan
            v26 = g26[col].mean() * 100 if len(g26) > 0 else np.nan
            delta = v26 - v24 if pd.notna(v24) and pd.notna(v26) else np.nan
        row_data.append(delta)
    delta_data.append(row_data)

delta_array = np.array(delta_data)
metric_labels = [m[0] for m in metrics_for_heatmap]

im = ax.imshow(delta_array, cmap='RdYlGn_r', aspect='auto', vmin=-15, vmax=15)
ax.set_xticks(range(len(metric_labels)))
ax.set_xticklabels(metric_labels, rotation=40, ha='right', fontsize=9)
ax.set_yticks(range(len(sen_labels)))
ax.set_yticklabels([s.title() for s in sen_labels], fontsize=11)

# Annotate cells
for i in range(len(sen_labels)):
    for j in range(len(metric_labels)):
        val = delta_array[i, j]
        if pd.notna(val):
            text = f"{val:+.1f}"
            color = 'white' if abs(val) > 10 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=9, color=color, fontweight='bold')

plt.colorbar(im, ax=ax, label='Change (2026 - 2024, pp or units)')
ax.set_title('Scope Inflation: 2024 to 2026 Changes by Seniority (SWE)\nPositive = higher requirements in 2026', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/T12_scope_inflation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: T12_scope_inflation_heatmap.png")

# ------------------------------------------------------------------
# Final summary stats for report
# ------------------------------------------------------------------
print("\n\n=== REPORT SUMMARY ===")
print(f"Total SWE rows analyzed: {len(rdf)}")
print(f"Requirements section found: {req_section_found} ({100*req_section_found/len(rdf):.1f}%)")
print(f"YOE extractable: {rdf['yoe_min'].notna().sum()} ({100*rdf['yoe_min'].notna().mean():.1f}%)")
print(f"Education extractable: {rdf['education'].notna().sum()} ({100*rdf['education'].notna().mean():.1f}%)")

# Entry-level scope inflation summary
print("\n--- Entry-level scope inflation summary ---")
if len(entry_arshkon) > 0 and len(entry_scraped) > 0:
    print(f"Entry arshkon median YOE: {entry_arshkon['yoe_min'].median()}")
    print(f"Entry scraped median YOE: {entry_scraped['yoe_min'].median()}")
    print(f"Entry arshkon % AI: {100*entry_arshkon['has_ai_keywords'].mean():.1f}%")
    print(f"Entry scraped % AI: {100*entry_scraped['has_ai_keywords'].mean():.1f}%")
    print(f"Entry arshkon % cross-functional: {100*entry_arshkon['soft_cross_functional'].mean():.1f}%")
    print(f"Entry scraped % cross-functional: {100*entry_scraped['soft_cross_functional'].mean():.1f}%")
    print(f"Entry arshkon % ownership: {100*entry_arshkon['soft_ownership'].mean():.1f}%")
    print(f"Entry scraped % ownership: {100*entry_scraped['soft_ownership'].mean():.1f}%")
    print(f"Entry arshkon median tech count: {entry_arshkon['tech_count'].median()}")
    print(f"Entry scraped median tech count: {entry_scraped['tech_count'].median()}")

# Also save raw extraction data for downstream use
rdf.to_csv(f'{TBL_DIR}/T12_extracted_features.csv', index=False)
print(f"\nSaved extracted features: {TBL_DIR}/T12_extracted_features.csv")

# Sensitivity: repeat junior comparison using seniority_final (unpatched)
print("\n\n--- SENSITIVITY: Using original seniority_final ---")
# Re-load with seniority_final
df_final = con.sql(f"""
SELECT uid, source, period, seniority_final, seniority_3level, description_length
FROM '{PARQUET}'
WHERE source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
  AND is_swe = true
""").df()

# Merge with extraction results
merged = rdf.merge(df_final[['uid', 'seniority_final', 'seniority_3level']], on='uid', how='left')

jr_final_24 = merged[(merged['seniority_3level'] == 'junior') & (merged['period'].isin(['2024-01', '2024-04']))]
jr_final_26 = merged[(merged['seniority_3level'] == 'junior') & (merged['period'] == '2026-03')]
print(f"Junior (seniority_final) 2024: {len(jr_final_24)}, 2026: {len(jr_final_26)}")
if len(jr_final_24) > 5 and len(jr_final_26) > 5:
    compare_groups(jr_final_24, jr_final_26, 'Jr24(fin)', 'Jr26(fin)')

print("\nDone!")
