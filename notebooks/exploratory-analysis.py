# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exploratory Analysis: AI-Driven Restructuring of Junior SWE Labor
#
# This notebook explores four datasets to validate our core hypotheses:
# - **Kaggle LinkedIn postings (2023–2024):** ~124K postings with titles, descriptions, seniority, skills
# - **LinkedIn scraped data (March 2026):** 486 SWE + 181 non-SWE postings from our daily scraper
# - **FRED JOLTS:** Official job openings for Professional & Business Services, Information sector, and Total Nonfarm
#
# ---
#
# ## Research Questions Addressed
# | RQ | Question | What we look for here |
# |-----|---------|----------------------|
# | RQ1 | Disappearing vs. redefined? | Seniority distribution shifts, description length/complexity changes |
# | RQ2 | Which skills migrated? | Skill prevalence in junior vs. senior postings, cross-period comparison |
# | RQ3 | Structural break late 2025? | JOLTS time series for regime shifts |
# | RQ4 | SWE-specific or broader? | Compare SWE vs. non-SWE in scraped data + JOLTS |

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import re
import textwrap
import warnings
from IPython.display import display, HTML
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.metrics import classification_report, confusion_matrix
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 100

DATA = '../data/'
print('Setup complete.')

# %% [markdown]
# ---
# ## 1. Load & Clean Data

# %% [markdown]
# ### 1a. Kaggle LinkedIn (2023–2024)

# %%
# Load Kaggle LinkedIn postings — large file, use relevant columns only
KAGGLE_COLS = [
    'job_id', 'company_name', 'title', 'description', 'location',
    'formatted_experience_level', 'formatted_work_type', 'skills_desc',
    'original_listed_time', 'listed_time', 'max_salary', 'med_salary',
    'min_salary', 'pay_period', 'remote_allowed', 'views', 'applies'
]

kaggle = pd.read_csv(
    f'{DATA}kaggle-linkedin-jobs-2023-2024/postings.csv',
    usecols=KAGGLE_COLS,
    low_memory=False
)

# Parse dates
kaggle['listed_date'] = pd.to_datetime(kaggle['listed_time'], unit='ms', errors='coerce')
kaggle['listed_month'] = kaggle['listed_date'].dt.to_period('M')

# Standardize seniority
kaggle['seniority'] = kaggle['formatted_experience_level'].str.strip().str.lower()

print(f'Kaggle postings loaded: {len(kaggle):,} rows')
print(f'Date range: {kaggle["listed_date"].min()} to {kaggle["listed_date"].max()}')
print(f'\nSeniority distribution:')
print(kaggle['seniority'].value_counts())

# %% [markdown]
# ### 1b. Filter to SWE-related postings

# %%
# Define SWE-related title patterns
SWE_PATTERN = r'(?i)\b(software\s*(engineer|developer|dev)|swe|full[- ]?stack|front[- ]?end|back[- ]?end|web\s*developer|mobile\s*developer|devops|platform\s*engineer|data\s*engineer|ml\s*engineer|machine\s*learning\s*engineer|site\s*reliability)\b'

# Control occupations (non-AI-exposed) for DiD
CONTROL_PATTERN = r'(?i)\b(civil\s*engineer|mechanical\s*engineer|nurse|registered\s*nurse|nursing|electrical\s*engineer|chemical\s*engineer)\b'

kaggle['is_swe'] = kaggle['title'].str.contains(SWE_PATTERN, na=False)
kaggle['is_control'] = kaggle['title'].str.contains(CONTROL_PATTERN, na=False)

swe = kaggle[kaggle['is_swe']].copy()
control = kaggle[kaggle['is_control']].copy()

print(f'SWE postings: {len(swe):,} ({len(swe)/len(kaggle)*100:.1f}% of total)')
print(f'Control postings: {len(control):,} ({len(control)/len(kaggle)*100:.1f}% of total)')
print(f'\nSWE seniority breakdown:')
print(swe['seniority'].value_counts())

# %% [markdown]
# ### 1c. LinkedIn Scraped Data (March 2026)
#
# Our daily scraper collected 486 SWE jobs and 181 non-SWE jobs from LinkedIn on 2026-03-05.

# %%
# Load today's scraped SWE and non-SWE data
scraped_swe = pd.read_csv(f'{DATA}scraped/2026-03-05_swe_jobs.csv', low_memory=False)
scraped_non_swe = pd.read_csv(f'{DATA}scraped/2026-03-05_non_swe_jobs.csv', low_memory=False)

# Combine for full scraped dataset
scraped = pd.concat([scraped_swe, scraped_non_swe], ignore_index=True)

# LinkedIn's original label
scraped['seniority_linkedin'] = scraped['job_level'].str.strip().str.lower()
scraped['listed_date'] = pd.to_datetime(scraped['date_posted'], errors='coerce')
scraped['is_swe'] = scraped.index < len(scraped_swe)  # SWE file came first

print(f'Scraped data loaded: {len(scraped):,} total rows')
print(f'  SWE postings: {scraped["is_swe"].sum():,}')
print(f'  Non-SWE (control): {(~scraped["is_swe"]).sum():,}')
print(f'\nLinkedIn seniority labels (raw):')
print(scraped[scraped['is_swe']]['seniority_linkedin'].value_counts())
print(f'\n"not applicable" rate: {(scraped[scraped["is_swe"]]["seniority_linkedin"] == "not applicable").mean():.0%}')


# %% [markdown]
# ### 1d. Impute Seniority for Unlabeled Postings
#
# 27% of scraped SWE postings have `job_level = "not applicable"`. We build a rule-based
# classifier using **title keywords** (primary) and **description experience requirements** (fallback):
#
# | Signal | Source | Maps to |
# |--------|--------|---------|
# | Senior, Sr., Staff, Principal, Lead | Title | mid-senior level |
# | III, 3, IV, 4+ | Title | mid-senior level |
# | II, 2 | Title | associate |
# | Junior, Jr., New Grad, Early Career, I, 1 | Title | entry level |
# | Intern | Title | internship |
# | Manager, Director, VP, Head of | Title | director |
# | 0–2 years required | Description | entry level |
# | 3–4 years required | Description | associate |
# | 5+ years required | Description | mid-senior level |
#
# We apply this to **all** scraped postings and validate against LinkedIn's own labels where available.

# %%
def impute_seniority(title, description):
    """Classify seniority from title keywords and description experience requirements.
    
    Priority: title keywords > description years > unknown.
    """
    title = str(title).lower().strip()
    desc = str(description).lower()
    
    # --- Title-based rules (highest priority) ---
    # Internship
    if re.search(r'\b(intern|internship)\b', title):
        return 'internship'
    
    # Director+
    if re.search(r'\b(director|vp|vice\s*president|head\s+of|chief)\b', title):
        return 'director'
    
    # Senior / Staff / Principal / Lead / Architect
    if re.search(r'\b(senior|sr\.?|staff|principal|distinguished|lead\b|architect)\b', title):
        return 'mid-senior level'
    
    # Roman numeral / numeric levels in title
    # III, IV, 3, 4+ → mid-senior
    if re.search(r'\biii\b|\biv\b|\b[3-9]\b', title) and re.search(r'engineer|developer|swe', title):
        return 'mid-senior level'
    # II, 2 → associate
    if re.search(r'\bii\b|\b2\b', title) and re.search(r'engineer|developer|swe', title):
        return 'associate'
    # I, 1 → entry (but careful: "I" is also a pronoun, require context)
    if re.search(r'(?:engineer|developer|swe)\s+i\b|\b1\b.*(?:engineer|developer)', title):
        return 'entry level'
    
    # Junior / New Grad / Early Career
    if re.search(r'\b(junior|jr\.?|new\s*grad|entry[- ]?level|early\s*career|associate)\b', title):
        return 'entry level'
    
    # --- Description-based fallback (extract minimum years required) ---
    # Find all "X+ years of experience" patterns and use the minimum mentioned
    year_matches = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|professional|relevant|related|work)', desc)
    reasonable = [int(y) for y in year_matches if 0 < int(y) <= 20]
    if reasonable:
        min_years = min(reasonable)
        if min_years <= 2:
            return 'entry level'
        elif min_years <= 4:
            return 'associate'
        else:
            return 'mid-senior level'
    
    # Broader year pattern (just "X years" without "experience")
    year_matches_broad = re.findall(r'(\d+)\+?\s*years?', desc)
    reasonable_broad = [int(y) for y in year_matches_broad if 0 < int(y) <= 20]
    if reasonable_broad:
        min_years = min(reasonable_broad)
        if min_years <= 2:
            return 'entry level'
        elif min_years <= 4:
            return 'associate'
        else:
            return 'mid-senior level'
    
    return 'unknown'

# Apply to all scraped data
scraped['seniority_imputed'] = scraped.apply(
    lambda row: impute_seniority(row['title'], row['description']), axis=1
)

# Use LinkedIn label when available, imputed for "not applicable"
scraped['seniority'] = scraped['seniority_linkedin'].where(
    scraped['seniority_linkedin'] != 'not applicable',
    scraped['seniority_imputed']
)

swe_2026 = scraped[scraped['is_swe']].copy()
control_2026 = scraped[~scraped['is_swe']].copy()

print('=== Seniority imputation results (SWE only) ===')
print(f'\nBefore (LinkedIn labels):')
print(scraped[scraped['is_swe']]['seniority_linkedin'].value_counts().to_string())
print(f'\nImputed values for "not applicable" postings:')
na_mask = scraped['is_swe'] & (scraped['seniority_linkedin'] == 'not applicable')
print(scraped.loc[na_mask, 'seniority_imputed'].value_counts().to_string())
print(f'\nAfter (combined):')
print(swe_2026['seniority'].value_counts().to_string())
print(f'\nRemaining unknown: {(swe_2026["seniority"] == "unknown").sum()} ({(swe_2026["seniority"] == "unknown").mean():.0%})')

# %%
# Validate: how well does our imputed seniority match LinkedIn's own label?
# Use labeled postings (where LinkedIn gave a real seniority) as ground truth

labeled = scraped[
    scraped['is_swe'] & 
    (scraped['seniority_linkedin'] != 'not applicable')
].copy()

labeled['imputed'] = labeled.apply(
    lambda row: impute_seniority(row['title'], row['description']), axis=1
)

# Map to 3-level for cleaner comparison
def to_3level(s):
    if s in ('entry level', 'internship'):
        return 'junior'
    elif s in ('associate',):
        return 'mid'
    elif s in ('mid-senior level', 'director', 'executive'):
        return 'senior'
    return 'other'

labeled['truth_3'] = labeled['seniority_linkedin'].map(to_3level)
labeled['imputed_3'] = labeled['imputed'].map(to_3level)

# Confusion matrix
valid = labeled[labeled['imputed_3'] != 'other']  # exclude unknowns

print(f'Validation set: {len(labeled)} labeled SWE postings')
print(f'Classifier produced a label for: {len(valid)}/{len(labeled)} ({len(valid)/len(labeled):.0%})')
print()

if len(valid) > 10:
    print('=== Confusion Matrix (rows=LinkedIn truth, cols=imputed) ===')
    labels = ['junior', 'mid', 'senior']
    cm = confusion_matrix(valid['truth_3'], valid['imputed_3'], labels=labels)
    cm_df = pd.DataFrame(cm, index=[f'True: {l}' for l in labels], columns=[f'Pred: {l}' for l in labels])
    print(cm_df.to_string())
    print()
    print('=== Classification Report ===')
    print(classification_report(valid['truth_3'], valid['imputed_3'], labels=labels, zero_division=0))
    
    # Overall accuracy
    accuracy = (valid['truth_3'] == valid['imputed_3']).mean()
    print(f'Overall accuracy: {accuracy:.1%}')
    
    # Where does it disagree?
    disagree = valid[valid['truth_3'] != valid['imputed_3']]
    if len(disagree) > 0:
        print(f'\nDisagreements ({len(disagree)}):')
        for _, row in disagree.head(10).iterrows():
            print(f'  Title: "{row["title"][:60]}" | LinkedIn: {row["seniority_linkedin"]} | Imputed: {row["imputed"]}')

# %% [markdown]
# ### 1d. FRED JOLTS Data (via API)

# %%
# Download JOLTS series from FRED
# Key series:
#   JTSJOL      - Total Nonfarm job openings (SA)
#   JTS540099JOL - Professional & Business Services job openings (SA)
#   JTU5100JOL  - Information sector job openings (NSA)
#   JTS7200JOL  - Accommodation & Food Services (SA) — low-AI-exposure control
#   JTS9000JOL  - Government job openings (SA) — another control

FRED_SERIES = {
    'JTSJOL': 'Total Nonfarm',
    'JTS540099JOL': 'Professional & Business Services',
    'JTU5100JOL': 'Information',
    'JTS510099JOL': 'Information (SA)',
    'JTS7200JOL': 'Accommodation & Food Services',
    'JTS9000JOL': 'Government',
    'JTS3000JOL': 'Manufacturing',
    'JTS6000JOL': 'Education & Health Services',
    'JTS4000JOL': 'Trade, Transportation & Utilities',
}

def fetch_fred_csv(series_id):
    """Fetch a FRED series as CSV (no API key needed for CSV download)."""
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    try:
        df = pd.read_csv(url, parse_dates=['DATE'])
        df.columns = ['date', 'value']
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df.dropna()
    except Exception as e:
        print(f'  Could not fetch {series_id}: {e}')
        return pd.DataFrame()

jolts = {}
for sid, name in FRED_SERIES.items():
    print(f'Fetching {sid} ({name})...')
    jolts[sid] = fetch_fred_csv(sid)
    if not jolts[sid].empty:
        print(f'  Got {len(jolts[sid])} obs: {jolts[sid]["date"].min().date()} to {jolts[sid]["date"].max().date()}')

# Combine into single DataFrame
jolts_combined = pd.DataFrame()
for sid, name in FRED_SERIES.items():
    if not jolts[sid].empty:
        tmp = jolts[sid].copy()
        tmp['series'] = name
        tmp['series_id'] = sid
        jolts_combined = pd.concat([jolts_combined, tmp], ignore_index=True)

if not jolts_combined.empty:
    print(f'\nJOLTS combined: {len(jolts_combined):,} rows, {jolts_combined["series"].nunique()} series')
else:
    print('\nJOLTS data could not be fetched (FRED may be blocked in this environment).')
    print('To use JOLTS data, download CSVs manually from fred.stlouisfed.org and place them in the data/ folder.')
    print('Series needed: ' + ', '.join(FRED_SERIES.keys()))

# %% [markdown]
# ---
# ## 2. RQ1: Are Junior SWE Roles Disappearing or Being Redefined?
#
# We look for:
# - Shifts in seniority distribution across the Kaggle dataset timeline
# - Whether posting descriptions for "junior" roles are getting longer/more complex
# - Comparison of seniority mix between 2023–24 (Kaggle) and March 2026 (scraped)

# %%
# 2a. Seniority distribution: Kaggle 2024 vs Scraped 2026
# Core test: has the junior share declined?

# Kaggle SWE seniority (excluding NaN)
kaggle_dist = swe['seniority'].value_counts(normalize=True).rename('Kaggle 2024')

# Scraped 2026 SWE seniority
scraped_dist = swe_2026['seniority'].value_counts(normalize=True).rename('Scraped 2026')

# Combine and order
compare = pd.concat([kaggle_dist, scraped_dist], axis=1).fillna(0)
key_levels = ['entry level', 'associate', 'mid-senior level', 'director', 'executive', 'internship', 'not applicable']
compare = compare.reindex([l for l in key_levels if l in compare.index])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart comparison
compare.plot.barh(ax=ax1, color=['#4C72B0', '#DD8452'])
ax1.set_title('SWE Seniority Distribution: 2024 vs. 2026', fontweight='bold')
ax1.set_xlabel('Share of SWE Postings')
ax1.set_ylabel('')

# Delta chart — change in share
compare['delta'] = compare['Scraped 2026'] - compare['Kaggle 2024']
colors = ['#55A868' if x > 0 else '#C44E52' for x in compare['delta']]
compare['delta'].plot.barh(ax=ax2, color=colors)
ax2.set_title('Change in Seniority Share (2026 − 2024)', fontweight='bold')
ax2.set_xlabel('Δ Share')
ax2.axvline(0, color='black', linewidth=0.8)
ax2.set_ylabel('')

plt.tight_layout()
plt.savefig('fig1_seniority_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print('\nCross-period comparison:')
print(compare.round(3).to_string())

# Statistical test
# Build contingency table for top seniority levels
test_levels = ['entry level', 'mid-senior level', 'associate']
kaggle_counts = swe['seniority'].value_counts().reindex(test_levels, fill_value=0)
scraped_counts = swe_2026['seniority'].value_counts().reindex(test_levels, fill_value=0)
contingency = pd.DataFrame({'Kaggle 2024': kaggle_counts, 'Scraped 2026': scraped_counts})
chi2, p, dof, expected = chi2_contingency(contingency.T)
print(f'\nChi-squared test (entry/associate/mid-senior): χ²={chi2:.1f}, p={p:.4f}')

# %%
# 2b. Description complexity — proxy for scope inflation
# Compare 2024 vs 2026: are junior descriptions getting longer?

swe['desc_word_count'] = swe['description'].str.split().str.len()
swe_2026['desc_word_count'] = swe_2026['description'].str.split().str.len()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Box plot: 2024 description lengths by seniority
plot_data_2024 = swe[swe['seniority'].isin(['entry level', 'associate', 'mid-senior level'])].copy()
plot_data_2024['period'] = 'Kaggle 2024'
plot_data_2026 = swe_2026[swe_2026['seniority'].isin(['entry level', 'associate', 'mid-senior level', 'not applicable'])].copy()
plot_data_2026['period'] = 'Scraped 2026'

# Panel 1: 2024 vs 2026 description lengths by seniority
combined_plot = pd.concat([plot_data_2024, plot_data_2026], ignore_index=True)
sns.boxplot(data=combined_plot, x='seniority', y='desc_word_count', hue='period',
            order=['entry level', 'associate', 'mid-senior level'],
            showfliers=False, ax=ax1)
ax1.set_title('Description Length by Seniority: 2024 vs. 2026', fontweight='bold')
ax1.set_ylabel('Word count')
ax1.set_xlabel('')
ax1.tick_params(axis='x', rotation=15)

# Panel 2: Distribution of description lengths (entry-level only)
junior_2024 = swe[swe['seniority'] == 'entry level']['desc_word_count'].dropna()
junior_2026 = swe_2026[swe_2026['seniority'] == 'entry level']['desc_word_count'].dropna()
ax2.hist(junior_2024, bins=30, alpha=0.6, label=f'2024 (n={len(junior_2024)}, med={junior_2024.median():.0f})', color='#4C72B0', density=True)
ax2.hist(junior_2026, bins=30, alpha=0.6, label=f'2026 (n={len(junior_2026)}, med={junior_2026.median():.0f})', color='#DD8452', density=True)
ax2.set_title('Entry-Level SWE Description Length Distribution', fontweight='bold')
ax2.set_xlabel('Word count')
ax2.set_ylabel('Density')
ax2.legend()

plt.tight_layout()
plt.savefig('fig2_description_complexity.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary stats
print('Description word count (median) by seniority and period:')
for level in ['entry level', 'associate', 'mid-senior level']:
    med_2024 = swe[swe['seniority'] == level]['desc_word_count'].median()
    med_2026 = swe_2026[swe_2026['seniority'] == level]['desc_word_count'].median()
    n_2024 = len(swe[swe['seniority'] == level])
    n_2026 = len(swe_2026[swe_2026['seniority'] == level])
    print(f'  {level}: 2024={med_2024:.0f} (n={n_2024}), 2026={med_2026:.0f} (n={n_2026}), Δ={med_2026-med_2024:+.0f}')

# Mann-Whitney test for entry-level
if len(junior_2026) >= 5:
    stat, p = mannwhitneyu(junior_2024, junior_2026, alternative='two-sided')
    print(f'\nMann-Whitney U test (entry-level word count 2024 vs 2026): U={stat:.0f}, p={p:.4f}')

# %% [markdown]
# ### 2c. Entry-Level Job Descriptions: 2024 vs. 2026 Side-by-Side
#
# Qualitative comparison of what employers are asking for in entry-level SWE postings.
# We sample descriptions from both periods and display key requirements, skills,
# and language differences.

# %%
# 2c. Side-by-side entry-level descriptions: 2024 vs 2026

def extract_requirements(desc):
    """Pull out key bullets/requirements from a job description."""
    desc = str(desc)
    # Try to find requirements/qualifications section
    sections = re.split(r'(?i)(requirements?|qualifications?|what you.?ll need|must have|skills?)', desc)
    if len(sections) > 1:
        # Take the section after the first match
        req_text = sections[2] if len(sections) > 2 else sections[1]
        # Truncate at next section header
        req_text = re.split(r'(?i)(nice to have|preferred|benefits|about us|what we offer)', req_text)[0]
        return req_text.strip()[:800]
    return desc[:800]

def format_posting(row, period, desc_col='description'):
    """Format a single posting as HTML."""
    title = row.get('title', 'N/A')
    company = row.get('company_name', row.get('company', 'N/A'))
    seniority = row.get('seniority', 'N/A')
    desc = str(row[desc_col])
    reqs = extract_requirements(desc)
    # Clean up markdown artifacts
    reqs = re.sub(r'[#*_`]', '', reqs)
    reqs = re.sub(r'\n{3,}', '\n\n', reqs)
    reqs = reqs[:600]
    
    return f"""
    <div style="padding:10px; margin:5px; background:#f8f9fa; border-radius:8px; font-size:12px; overflow:hidden;">
        <b style="font-size:13px;">{title}</b><br>
        <i>{company}</i> · <span style="color:#666;">{seniority}</span> · {period}<br>
        <hr style="margin:5px 0;">
        <pre style="white-space:pre-wrap; font-family:inherit; font-size:11px; max-height:300px; overflow-y:auto;">{reqs}</pre>
    </div>"""

# Sample entry-level postings from each period
np.random.seed(42)
junior_2024_sample = swe[swe['seniority'] == 'entry level'].sample(min(5, len(swe[swe['seniority'] == 'entry level'])))
junior_2026_sample = swe_2026[swe_2026['seniority'] == 'entry level'].sample(min(5, len(swe_2026[swe_2026['seniority'] == 'entry level'])))

html = '<h3>Entry-Level SWE Postings: 2024 vs 2026</h3>'
html += '<div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">'
html += '<div><h4 style="text-align:center;">Kaggle 2024</h4>'
for _, row in junior_2024_sample.iterrows():
    html += format_posting(row, '2024')
html += '</div>'
html += '<div><h4 style="text-align:center;">Scraped 2026</h4>'
for _, row in junior_2026_sample.iterrows():
    html += format_posting(row, '2026')
html += '</div></div>'

display(HTML(html))

# %%
# 2d. Entry-level requirements comparison — quantitative
# What are employers asking for in junior roles now vs. 2024?

REQUIREMENT_PATTERNS = {
    # Experience
    '0-1 years exp': r'(?i)\b[01]\+?\s*years?\b',
    '2-3 years exp': r'(?i)\b[23]\+?\s*years?\b',
    '5+ years exp': r'(?i)\b[5-9]\+?\s*years?\b',
    # Degrees
    'BS/BA required': r'(?i)\b(bachelor|b\.?s\.?|b\.?a\.?)\b',
    'MS/PhD preferred': r'(?i)\b(master|m\.?s\.?|ph\.?d|graduate\s*degree)\b',
    # Technical
    'Cloud (AWS/GCP/Azure)': r'(?i)\b(aws|gcp|azure|google\s*cloud)\b',
    'Docker/Kubernetes': r'(?i)\b(docker|kubernetes|k8s)\b',
    'CI/CD': r'(?i)\b(ci/?cd|continuous\s*(integration|delivery))\b',
    'System design': r'(?i)\b(system\s*design|distributed\s*system|architecture)\b',
    'Microservices': r'(?i)\b(microservice|micro[- ]service)\b',
    # AI-era
    'AI/ML/LLM': r'(?i)\b(machine\s*learning|ml\b|llm|ai[- ]?(tool|powered|assisted)|copilot|chatgpt|claude|cursor|generative\s*ai)\b',
    'Prompt engineering': r'(?i)\b(prompt\s*engineer|prompting)\b',
    # Soft skills / scope
    'Cross-functional collab': r'(?i)\b(cross[- ]?functional|stakeholder|product\s*team)\b',
    'Mentorship/leadership': r'(?i)\b(mentor|lead\s*(a|the)\s*team|coach)\b',
    'End-to-end ownership': r'(?i)\b(end[- ]?to[- ]?end|full\s*ownership|own\s*(the|a)\s*(feature|product))\b',
    'Code review': r'(?i)\b(code\s*review|pull\s*request|pr\s*review)\b',
}

entry_2024 = swe[swe['seniority'] == 'entry level']
entry_2026 = swe_2026[swe_2026['seniority'] == 'entry level']

results = []
for name, pattern in REQUIREMENT_PATTERNS.items():
    r2024 = entry_2024['description'].str.contains(pattern, na=False).mean()
    r2026 = entry_2026['description'].str.contains(pattern, na=False).mean()
    results.append({'Requirement': name, '2024': r2024, '2026': r2026, 'Δ': r2026 - r2024})

req_df = pd.DataFrame(results).sort_values('Δ', ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(req_df))
width = 0.35
ax.barh(x + width/2, req_df['2024'], width, label='Entry-Level 2024', color='#4C72B0', alpha=0.8)
ax.barh(x - width/2, req_df['2026'], width, label='Entry-Level 2026', color='#DD8452', alpha=0.8)
ax.set_yticks(x)
ax.set_yticklabels(req_df['Requirement'])
ax.set_xlabel('Share of entry-level postings')
ax.set_title('What Employers Require in Entry-Level SWE Postings: 2024 vs. 2026', fontweight='bold')
ax.legend()
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

plt.tight_layout()
plt.savefig('fig_entry_level_requirements.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'\nEntry-level requirement comparison (n_2024={len(entry_2024)}, n_2026={len(entry_2026)}):')
print(req_df.to_string(index=False, float_format='{:.1%}'.format))

# %% [markdown]
# ---
# ## 3. RQ2: Skill Migration — Which Competencies Moved from Senior to Junior?
#
# We define target skills that our lit review identifies as traditionally senior-level,
# and measure their prevalence in junior vs. senior SWE postings.

# %%
# 3a. Define skill/competency keywords to track
SKILL_KEYWORDS = {
    # Traditionally senior skills (migration candidates)
    'System Design': r'(?i)\b(system\s*design|distributed\s*systems?|architecture)\b',
    'CI/CD': r'(?i)\b(ci/?cd|continuous\s*(integration|deployment|delivery)|jenkins|github\s*actions)\b',
    'Cross-functional': r'(?i)\b(cross[- ]?functional|stakeholder|product\s*team|collaborate\s*with\s*(product|design|business))\b',
    'Mentorship': r'(?i)\b(mentor|coach|lead\s*(a|the)\s*team|guide\s*(junior|team))\b',
    'End-to-end ownership': r'(?i)\b(end[- ]?to[- ]?end|full\s*ownership|own\s*(the|a)\s*(feature|product|service))\b',
    'Technical leadership': r'(?i)\b(tech\s*lead|technical\s*leadership|lead\s*engineer|principal)\b',
    'Code review': r'(?i)\b(code\s*review|pull\s*request|pr\s*review|review\s*code)\b',
    
    # AI-era skills (new entrants)
    'AI/ML tools': r'(?i)\b(copilot|chatgpt|claude|cursor|ai[- ]?(assisted|augmented|powered)|llm|large\s*language\s*model|prompt\s*engineer|generative\s*ai|genai)\b',
    'Prompt engineering': r'(?i)\b(prompt\s*engineer|prompt\s*design|prompting)\b',
    
    # Baseline skills (should be stable)
    'Python': r'(?i)\bpython\b',
    'JavaScript': r'(?i)\b(javascript|typescript|react|angular|vue)\b',
    'Cloud': r'(?i)\b(aws|azure|gcp|google\s*cloud|cloud\s*(native|infrastructure))\b',
    'Docker/K8s': r'(?i)\b(docker|kubernetes|k8s|container)\b',
    'SQL/Database': r'(?i)\b(sql|postgres|mysql|mongodb|database|data\s*model)\b',
    'Agile': r'(?i)\b(agile|scrum|sprint|kanban|jira)\b',
}

# Scan BOTH datasets
for skill_name, pattern in SKILL_KEYWORDS.items():
    swe[f'has_{skill_name}'] = swe['description'].str.contains(pattern, na=False).astype(int)
    swe_2026[f'has_{skill_name}'] = swe_2026['description'].str.contains(pattern, na=False).astype(int)

skill_cols = [c for c in swe.columns if c.startswith('has_')]

print('Skill prevalence comparison (2024 vs 2026):')
print(f'{"Skill":<25} {"2024":>8} {"2026":>8} {"Delta":>8}')
print('-' * 51)
for col in sorted(skill_cols, key=lambda c: swe[c].mean(), reverse=True):
    name = col.replace('has_', '')
    r2024 = swe[col].mean()
    r2026 = swe_2026[col].mean()
    delta = r2026 - r2024
    marker = '***' if abs(delta) > 0.05 else ''
    print(f'{name:<25} {r2024:>7.1%} {r2026:>7.1%} {delta:>+7.1%} {marker}')

# %%
# 3b. Skill prevalence by seniority — junior vs senior, 2024 vs 2026
junior_mask = swe['seniority'].isin(['entry level', 'associate'])
senior_mask = swe['seniority'].isin(['mid-senior level', 'director', 'executive'])
junior_mask_2026 = swe_2026['seniority'].isin(['entry level', 'associate'])
senior_mask_2026 = swe_2026['seniority'].isin(['mid-senior level', 'director', 'executive'])

junior_2024_skills = swe.loc[junior_mask, skill_cols].mean().rename('Junior 2024')
senior_2024_skills = swe.loc[senior_mask, skill_cols].mean().rename('Senior 2024')
junior_2026_skills = swe_2026.loc[junior_mask_2026, skill_cols].mean().rename('Junior 2026')
senior_2026_skills = swe_2026.loc[senior_mask_2026, skill_cols].mean().rename('Senior 2026')

skill_compare = pd.concat([junior_2024_skills, senior_2024_skills, junior_2026_skills, senior_2026_skills], axis=1)
skill_compare.index = skill_compare.index.str.replace('has_', '')

# Sort by the gap between junior 2026 and junior 2024 (migration signal)
skill_compare['junior_shift'] = skill_compare['Junior 2026'] - skill_compare['Junior 2024']
skill_compare = skill_compare.sort_values('junior_shift', ascending=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Panel 1: Junior vs Senior in 2024
x = np.arange(len(skill_compare))
width = 0.35
ax1.barh(x + width/2, skill_compare['Junior 2024'], width, label='Junior 2024', color='#4C72B0')
ax1.barh(x - width/2, skill_compare['Senior 2024'], width, label='Senior 2024', color='#DD8452')
ax1.set_yticks(x)
ax1.set_yticklabels(skill_compare.index)
ax1.set_xlabel('Prevalence')
ax1.set_title('Skill Prevalence: Junior vs. Senior (2024)', fontweight='bold')
ax1.legend()

# Panel 2: Junior 2024 vs Junior 2026 (migration over time)
ax2.barh(x + width/2, skill_compare['Junior 2024'], width, label='Junior 2024', color='#4C72B0', alpha=0.6)
ax2.barh(x - width/2, skill_compare['Junior 2026'], width, label='Junior 2026', color='#55A868')
ax2.set_yticks(x)
ax2.set_yticklabels(skill_compare.index)
ax2.set_xlabel('Prevalence')
ax2.set_title('Junior SWE Skill Shift: 2024 → 2026', fontweight='bold')
ax2.legend()

plt.tight_layout()
plt.savefig('fig3_skill_migration.png', dpi=150, bbox_inches='tight')
plt.show()

print('\nJunior skill shift (2024 → 2026):')
print(skill_compare[['Junior 2024', 'Junior 2026', 'junior_shift']].round(3).to_string())

# %%
# 3c. Skill prevalence heatmap: seniority × period
# Shows which skills are appearing more in junior 2026 postings vs junior 2024

heatmap_data = pd.DataFrame({
    'Junior 2024': swe.loc[junior_mask, skill_cols].mean(),
    'Senior 2024': swe.loc[senior_mask, skill_cols].mean(),
    'Junior 2026': swe_2026.loc[junior_mask_2026, skill_cols].mean(),
    'Senior 2026': swe_2026.loc[senior_mask_2026, skill_cols].mean(),
})
heatmap_data.index = heatmap_data.index.str.replace('has_', '')
heatmap_data = heatmap_data.sort_values('Junior 2026', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt='.0%', cmap='YlOrRd', ax=ax,
            linewidths=0.5, vmin=0, vmax=0.6)
ax.set_title('Skill Prevalence by Seniority & Period', fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('')

plt.tight_layout()
plt.savefig('fig4_skill_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# Key finding: which skills grew most in junior postings?
print('\nBiggest junior skill gains (2024 → 2026):')
gains = (heatmap_data['Junior 2026'] - heatmap_data['Junior 2024']).sort_values(ascending=False)
print(gains.round(3).head(8).to_string())

# %% [markdown]
# ---
# ## 4. RQ3 & RQ4: Structural Break & SWE-Specificity
#
# Using FRED JOLTS to look for:
# - Regime shifts in knowledge-work job openings
# - Whether AI-exposed sectors diverge from controls

# %%
# 4a. JOLTS: Job openings in knowledge-work vs. control sectors
# AI-exposed: Professional & Business Services, Information
# Controls: Manufacturing, Accommodation & Food, Government, Education & Health

if not jolts_combined.empty:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot absolute levels
    ai_exposed = ['Professional & Business Services', 'Information', 'Information (SA)']
    controls = ['Manufacturing', 'Accommodation & Food Services', 'Government', 
                'Education & Health Services', 'Trade, Transportation & Utilities']

    for series_name in ai_exposed:
        subset = jolts_combined[jolts_combined['series'] == series_name]
        if not subset.empty:
            ax1.plot(subset['date'], subset['value'], label=series_name, linewidth=2)
    
    ax1.axvline(pd.Timestamp('2025-12-01'), color='red', linestyle='--', alpha=0.7, label='Agent deployment (Dec 2025)')
    ax1.axvline(pd.Timestamp('2022-11-01'), color='gray', linestyle=':', alpha=0.5, label='ChatGPT launch (Nov 2022)')
    ax1.set_title('JOLTS Job Openings: AI-Exposed Sectors (thousands)', fontweight='bold')
    ax1.set_ylabel('Job Openings (thousands)')
    ax1.legend(fontsize=9)
    ax1.set_xlim(pd.Timestamp('2019-01-01'), jolts_combined['date'].max())

    for series_name in controls:
        subset = jolts_combined[jolts_combined['series'] == series_name]
        if not subset.empty:
            ax2.plot(subset['date'], subset['value'], label=series_name, linewidth=1.5, alpha=0.8)

    ax2.axvline(pd.Timestamp('2025-12-01'), color='red', linestyle='--', alpha=0.7, label='Agent deployment')
    ax2.axvline(pd.Timestamp('2022-11-01'), color='gray', linestyle=':', alpha=0.5, label='ChatGPT launch')
    ax2.set_title('JOLTS Job Openings: Control Sectors (thousands)', fontweight='bold')
    ax2.set_ylabel('Job Openings (thousands)')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('fig5_jolts_sectors.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print('JOLTS data not available — FRED may be blocked in this environment.')
    print('Download manually from: https://fred.stlouisfed.org/series/JTS540099JOL')

# %%
# 4b. JOLTS: Indexed to 100 at Jan 2020 for clean comparison
if not jolts_combined.empty:
    fig, ax = plt.subplots(figsize=(14, 7))
    
    focus_series = ['Professional & Business Services', 'Information (SA)', 
                    'Manufacturing', 'Education & Health Services', 'Total Nonfarm']
    
    for series_name in focus_series:
        subset = jolts_combined[jolts_combined['series'] == series_name].copy()
        if subset.empty:
            continue
        subset = subset.sort_values('date')
        # Index to Jan 2020
        base = subset.loc[subset['date'] >= '2020-01-01'].iloc[0]['value'] if len(subset[subset['date'] >= '2020-01-01']) > 0 else subset.iloc[0]['value']
        subset['indexed'] = (subset['value'] / base) * 100
        subset = subset[subset['date'] >= '2020-01-01']
        
        lw = 2.5 if 'Professional' in series_name or 'Information' in series_name else 1.5
        ls = '-' if 'Professional' in series_name or 'Information' in series_name else '--'
        ax.plot(subset['date'], subset['indexed'], label=series_name, linewidth=lw, linestyle=ls)
    
    ax.axhline(100, color='black', linestyle='-', alpha=0.2)
    ax.axvline(pd.Timestamp('2025-12-01'), color='red', linestyle='--', alpha=0.7, label='Agent deployment')
    ax.axvline(pd.Timestamp('2022-11-01'), color='gray', linestyle=':', alpha=0.5, label='ChatGPT launch')
    ax.set_title('JOLTS Job Openings Indexed to Jan 2020 = 100', fontweight='bold')
    ax.set_ylabel('Index (Jan 2020 = 100)')
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('fig6_jolts_indexed.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print('Skipping JOLTS indexed chart — data not available.')

# %% [markdown]
# ---
# ## 5. "Ghost Jobs" & Seniority Mismatch Detection
#
# Operationalizing Akanegbu (2026): postings titled "junior" but with requirements
# that read as mid-level or senior.

# %%
# 5a. Detect seniority mismatch — junior titles with senior requirements
# Compare ghost job rates: 2024 vs 2026

SENIOR_SIGNALS = [
    r'(?i)\b(5|6|7|8|9|10)\+?\s*years?',                    # 5+ years experience
    r'(?i)\b(system\s*design|architecture|distributed)\b',   # System design
    r'(?i)\b(lead|mentor|manage\s*(a|the)\s*team)\b',        # Leadership
    r'(?i)\b(end[- ]?to[- ]?end\s*own)\b',                  # Full ownership
    r'(?i)\b(principal|staff|senior)\b',                     # Explicit senior language
]

def compute_ghost_rate(df, label):
    junior = df[df['seniority'] == 'entry level'].copy()
    junior['senior_signal_count'] = 0
    for pattern in SENIOR_SIGNALS:
        junior['senior_signal_count'] += junior['description'].str.contains(pattern, na=False).astype(int)
    junior['is_ghost_job'] = junior['senior_signal_count'] >= 2
    rate = junior['is_ghost_job'].mean()
    n_ghost = junior['is_ghost_job'].sum()
    print(f'{label}: ghost rate = {rate:.1%} ({n_ghost}/{len(junior)})')
    return junior

junior_2024 = compute_ghost_rate(swe, 'Kaggle 2024')
junior_2026 = compute_ghost_rate(swe_2026, 'Scraped 2026')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Senior signal distribution comparison
ax1.hist(junior_2024['senior_signal_count'], bins=range(0, 6), alpha=0.6, 
         label=f'2024 (n={len(junior_2024)})', color='#4C72B0', density=True, align='left')
ax1.hist(junior_2026['senior_signal_count'], bins=range(0, 6), alpha=0.6,
         label=f'2026 (n={len(junior_2026)})', color='#DD8452', density=True, align='left')
ax1.set_title('Senior Signals in Entry-Level SWE Postings', fontweight='bold')
ax1.set_xlabel('Number of senior-level signals detected')
ax1.set_ylabel('Density')
ax1.legend()

# Panel 2: Ghost job rate comparison
ghost_rates = pd.DataFrame({
    'Period': ['Kaggle 2024', 'Scraped 2026'],
    'Ghost Rate': [junior_2024['is_ghost_job'].mean(), junior_2026['is_ghost_job'].mean()],
    'n': [len(junior_2024), len(junior_2026)]
})
bars = ax2.bar(ghost_rates['Period'], ghost_rates['Ghost Rate'], color=['#4C72B0', '#DD8452'])
ax2.set_title('"Ghost Job" Rate in Entry-Level SWE', fontweight='bold')
ax2.set_ylabel('Share with 2+ senior signals')
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
for bar, n in zip(bars, ghost_rates['n']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'n={n}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('fig10_ghost_jobs.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 6. AI Skill Mentions — Emergence Timeline

# %%
# 6a. AI/ML tool mentions — the key signal for RQ3
# Compare 2024 vs 2026: how much have AI/LLM mentions grown?

AI_PATTERN = r'(?i)\b(copilot|chatgpt|claude|cursor|ai[- ]?(assisted|augmented|powered)|llm|large\s*language|prompt\s*engineer|generative\s*ai|genai|ai\s*agent|ai\s*coding|ai\s*tool)\b'

kaggle['has_ai_mention'] = kaggle['description'].str.contains(AI_PATTERN, na=False).astype(int)
swe['has_ai_broad'] = swe['description'].str.contains(AI_PATTERN, na=False).astype(int)
swe_2026['has_ai_broad'] = swe_2026['description'].str.contains(AI_PATTERN, na=False).astype(int)
scraped['has_ai_broad'] = scraped['description'].str.contains(AI_PATTERN, na=False).astype(int)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Overall AI mention rate comparison
ai_rates = pd.DataFrame({
    'Category': ['All postings\n2024', 'SWE\n2024', 'All postings\n2026', 'SWE\n2026'],
    'AI Rate': [
        kaggle['has_ai_mention'].mean(),
        swe['has_ai_broad'].mean(),
        scraped['has_ai_broad'].mean(),
        swe_2026['has_ai_broad'].mean(),
    ],
    'Color': ['#4C72B0', '#4C72B0', '#DD8452', '#DD8452'],
    'Hatch': ['', '//', '', '//'],
})
bars = ax1.bar(ai_rates['Category'], ai_rates['AI Rate'], color=ai_rates['Color'])
ax1.set_title('AI/LLM Tool Mention Rate in Job Postings', fontweight='bold')
ax1.set_ylabel('Share of postings')
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
for bar, rate in zip(bars, ai_rates['AI Rate']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{rate:.1%}', ha='center', fontsize=10)

# Panel 2: AI mention rate by seniority (2026 scraped)
ai_by_seniority_2024 = swe.groupby('seniority')['has_ai_broad'].mean().rename('2024')
ai_by_seniority_2026 = swe_2026.groupby('seniority')['has_ai_broad'].mean().rename('2026')
ai_seniority = pd.concat([ai_by_seniority_2024, ai_by_seniority_2026], axis=1).fillna(0)
levels = ['entry level', 'associate', 'mid-senior level', 'director']
ai_seniority = ai_seniority.reindex([l for l in levels if l in ai_seniority.index])

ai_seniority.plot.bar(ax=ax2, color=['#4C72B0', '#DD8452'])
ax2.set_title('AI/LLM Mentions by Seniority Level', fontweight='bold')
ax2.set_ylabel('Share of postings')
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax2.tick_params(axis='x', rotation=15)
ax2.legend()

plt.tight_layout()
plt.savefig('fig11_ai_mentions.png', dpi=150, bbox_inches='tight')
plt.show()

# Print specific AI tool mentions in 2026
print('\nSpecific AI tool mentions in 2026 SWE postings:')
for tool, pattern in [
    ('Copilot', r'(?i)\bcopilot\b'),
    ('ChatGPT', r'(?i)\bchatgpt\b'),
    ('Claude', r'(?i)\bclaude\b'),
    ('Cursor', r'(?i)\bcursor\b'),
    ('LLM/GenAI', r'(?i)\b(llm|large\s*language|generative\s*ai|genai)\b'),
    ('AI agent', r'(?i)\bai\s*agent\b'),
    ('Prompt engineering', r'(?i)\bprompt\s*engineer\b'),
]:
    rate = swe_2026['description'].str.contains(pattern, na=False).mean()
    count = swe_2026['description'].str.contains(pattern, na=False).sum()
    print(f'  {tool}: {rate:.1%} ({count} postings)')

# %% [markdown]
# ---
# ## 7. Summary Statistics Table

# %%
# 7a. Summary statistics for the paper
print('='*70)
print('SUMMARY STATISTICS')
print('='*70)

print(f'\n--- Kaggle LinkedIn (2023–2024) ---')
print(f'Total postings: {len(kaggle):,}')
print(f'SWE postings: {len(swe):,} ({len(swe)/len(kaggle)*100:.1f}%)')
print(f'Control postings: {len(control):,}')
print(f'Date range: {kaggle["listed_date"].min().date()} to {kaggle["listed_date"].max().date()}')

print(f'\n--- Scraped LinkedIn (March 2026) ---')
print(f'Total postings: {len(scraped):,}')
print(f'SWE postings: {len(swe_2026):,}')
print(f'Non-SWE (control): {len(control_2026):,}')
print(f'Source: daily scraper (python-jobspy)')

print(f'\n--- SWE Description Stats (2024 vs 2026) ---')
for level in ['entry level', 'associate', 'mid-senior level']:
    sub_24 = swe[swe['seniority'] == level]
    sub_26 = swe_2026[swe_2026['seniority'] == level]
    med_24 = sub_24['desc_word_count'].median() if len(sub_24) > 0 else 0
    med_26 = sub_26['desc_word_count'].median() if len(sub_26) > 0 else 0
    print(f'  {level}: 2024 n={len(sub_24):,} med_words={med_24:.0f} | '
          f'2026 n={len(sub_26):,} med_words={med_26:.0f}')

print(f'\n--- Ghost Jobs (entry-level with 2+ senior signals) ---')
print(f'  2024: {junior_2024["is_ghost_job"].mean():.1%} ({junior_2024["is_ghost_job"].sum()}/{len(junior_2024)})')
print(f'  2026: {junior_2026["is_ghost_job"].mean():.1%} ({junior_2026["is_ghost_job"].sum()}/{len(junior_2026)})')

print(f'\n--- AI/LLM Tool Mentions ---')
print(f'  All 2024: {kaggle["has_ai_mention"].mean():.1%}')
print(f'  SWE 2024: {swe["has_ai_broad"].mean():.1%}')
print(f'  SWE 2026: {swe_2026["has_ai_broad"].mean():.1%}')


# %% [markdown]
# ---
# ## 8. Preliminary Observations & Next Steps
#
# ### Key findings from this EDA
#
# **RQ1 (Disappearing vs. redefined?):**
# - Compare seniority distributions between Kaggle 2024 and scraped 2026 data
# - Track whether entry-level share is shrinking and "not applicable" is growing
# - Description lengths may indicate scope inflation in junior roles
#
# **RQ2 (Skill migration):**
# - Cross-period skill prevalence shows which traditionally-senior skills now appear in junior postings
# - AI/LLM tool mentions provide direct measurement of the technology shift
# - Heatmap reveals the specific skills migrating downward
#
# **RQ3 (Structural break?):**
# - AI tool mention rates between 2024 and 2026 quantify the shift
# - JOLTS data for sector-level trends
#
# **RQ4 (SWE-specific or broader?):**
# - Non-SWE scraped control group enables within-period comparison
#
# ### What we still need for the full paper
# - Longer time series of posting-level text (more daily scrapes, Common Crawl 2020–2022)
# - Formal ITS / Bai-Perron breakpoint detection (needs more post-break months)
# - Embedding-based seniority classifier (train on clean 2020–22 labels, apply forward)
# - BERTopic for emergent skill discovery
# - Synthetic control method for RQ4
# - Indeed data to supplement LinkedIn (scraper supports it, not yet run)
