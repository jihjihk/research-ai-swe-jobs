# Analysis Plan: Pre-Processing, Validation & Hypothesis Testing

Date: 2026-03-18
Last updated: 2026-03-18
Status: Draft — ready for review before implementation

---

## Context

This is the master reference for the full analysis pipeline: from raw data to publication-ready results. It covers pre-processing (Stages 1-8), data validation (Stage 9), exploratory analysis (Stage 10), formal hypothesis testing (Stage 11), and statistical verification (Stage 12).

The current implementation state (`harmonize.py`) handles basic schema unification and job_url dedup. This document specifies everything needed for publication-quality rigor, including all validation checks a reviewer at a major venue would expect.

**Research questions addressed:** RQ1 (junior roles disappearing/redefined?), RQ2 (skill migration?), RQ3 (structural break?), RQ4 (SWE-specific?), RQ5 (training implications), RQ6 (senior archetype shift?), RQ7 (historical comparison). See `docs/research-design-h1-h3.md` for full RQ specifications.

---

## Data inventory (corrected after investigation)

### What we actually have

| | Kaggle | Scraped |
|---|---|---|
| **Source** | `arshkon/linkedin-job-postings` (Kaggle) | Our daily scraper (LinkedIn + Indeed) |
| **Total rows** | 123,849 | ~100,739 SWE-file + ~86,214 non-SWE-file (across 14 days) |
| **Actual date range** | **April 2024 only** (99.9% April; 1,815 rows from March; 19 rows from Dec 2023-Feb 2024) | March 5-18, 2026 |
| **SWE postings** | ~3,466 (expanded SWE regex) | ~14,391 unique (title regex match across all days) |
| **Dedup status** | 0 duplicate job_ids | 0 duplicate job_urls (scraper dedup working) |
| **Platform** | LinkedIn only | LinkedIn (60K) + Indeed (41K) |
| **Seniority labels** | 66.5% labeled, 33.5% null | LinkedIn: 100% labeled (28% "not applicable"); Indeed: 0% labeled |
| **Salary coverage** | 27.9% of SWE | LinkedIn: 4%; Indeed: 76% |
| **Skills (structured)** | `skills_desc` 98% null; BUT `job_skills.csv` companion has 35 coarse categories for 98% of SWE | Not structured; extract from description |
| **Company metadata** | Available via companion files (industry, size, employee count) | `company_industry`, `company_num_employees` fields present |

### Critical corrections to prior assumptions

**1. The Kaggle dataset is a single cross-sectional snapshot from April 2024, not a multi-year panel.**

Investigation confirms: the [arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) dataset (v13, last updated August 2024) contains 123,849 postings. The underlying [scraper](https://github.com/ArshKA/LinkedIn-Job-Scraper) runs as a "continuous stream" that captures whatever is currently live on LinkedIn. LinkedIn only displays active/recent postings — so the dataset is a snapshot of jobs visible on LinkedIn around April 5-20, 2024, not a historical archive of all jobs posted during 2023-2024. The "2023-2024" title refers to the project timeframe, not the data's temporal span.

Date evidence:
- `listed_time`: 99.9% of rows fall between April 5-20, 2024 (scrape window)
- `original_listed_time`: 98.5% are April 2024, 1.5% are March 2024, 19 rows are Dec 2023-Feb 2024 (these are long-lived postings still active when scraped)
- Zero rows from 2023 in `listed_time`; only 3 rows from 2023 in `original_listed_time`

**This means:**
- We cannot do within-Kaggle trend analysis (no monthly time series)
- Same-month comparison (March vs. March) is impossible — effectively zero March SWE postings in Kaggle
- The comparison is April 2024 vs. March 2026, a 23-month gap with a 1-month seasonal offset
- All prior documentation describing this as "2023-2024 data" must be corrected
- The 123,849 rows IS the full dataset (matches Kaggle's description of "124,000+ postings") — we are not missing data

**Potential supplementary dataset:** [asaniczka/1-3m-linkedin-jobs-and-skills-2024](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) contains 1.3M LinkedIn postings from 2024 with augmented skills data (~2GB). If this dataset has broader temporal coverage (multiple months of 2024), it could give us the within-2024 trend analysis we're missing. It would also increase our baseline SWE sample from ~3,134 to potentially 30-40K+. **Decision needed:** Should we download and evaluate this dataset as a supplement or replacement for the arshkon dataset?

**2. The `_swe_jobs.csv` files are split by query tier, not by title pattern.** The scraper runs 28 queries across 3 tiers (SWE, adjacent, control). Results from Tier 1 SWE queries go to `_swe_jobs.csv`; Tier 2+3 go to `_non_swe_jobs.csv`. Within the SWE file, 98.1% of titles match the SWE regex — the 1.9% leakage is LinkedIn/Indeed returning non-SWE results for SWE queries. The non-SWE file contains a mix of adjacent roles (9.2%), control occupations (17.5%), and other results (73.2%).

**3. Description lengths differ significantly between datasets.** Kaggle SWE median: 3,242 chars. Scraped SWE (LinkedIn): 5,036 chars. This 55% difference could be (a) real scope inflation, (b) different scraping methods capturing different content, or (c) boilerplate differences. **This is a major confounder for RQ1 description complexity analysis and must be investigated before any length-based conclusions.**

**4. Indeed and LinkedIn have complementary but asymmetric data.** Indeed provides salary (76%) but no seniority (0%). LinkedIn provides seniority (100%) but almost no salary (4%). Combining them without accounting for this creates compositional artifacts.

**5. Aggregators are present in BOTH datasets.** Kaggle top SWE companies include DataAnnotation (168), Dice (35), Apex Systems (29), TEKsystems (15), Insight Global (14). Scraped top includes Lensa (16), Jobs via Dice (29). The aggregator problem is not limited to scraped data.

**6. Company concentration differs.** Kaggle SWE top-5 companies = 12.6% of postings. Scraped SWE top-5 = 30.2%. The scraped data is more concentrated, which could bias skill/seniority distributions if those companies have unusual posting patterns.

### Kaggle companion files (available, currently unused)

| File | Rows | Content | Joinable? |
|---|---|---|---|
| `companies/companies.csv` | ~22MB | company_id, name, description, company_size, state, country, city | Yes via `company_id` in postings |
| `companies/employee_counts.csv` | employee_count, follower_count per company | Yes via `company_id` |
| `companies/company_industries.csv` | Industry associations per company | Yes via `company_id` |
| `jobs/job_industries.csv` | industry_id per job_id | Yes — covers 99.3% of SWE postings |
| `jobs/job_skills.csv` | skill_abr per job_id (35 coarse categories like IT, ENG, SALE) | Yes — covers 98.3% of SWE postings; too coarse for skill migration analysis |
| `jobs/salaries.csv` | Structured salary data per job_id | Yes — may fill some salary gaps |
| `jobs/benefits.csv` | Benefits per job_id | Yes |
| `mappings/industries.csv` | 422 industry_id → industry_name mappings | Lookup table |
| `mappings/skills.csv` | 35 skill_abr → skill_name mappings | Lookup table |

**Action:** Join `job_industries` + `companies` to populate `company_industry` and `company_size` for Kaggle rows. The `job_skills` are too coarse (35 categories) for our skill migration analysis but useful for broad occupation validation.

---

## Pipeline overview

```
Raw Data (Kaggle CSV + daily scraped CSVs)
  │
  ├─ Phase 1: Data Preparation
  │   ├─ Stage 1: Ingest & Schema Unification
  │   ├─ Stage 2: Aggregator / Staffing Company Handling
  │   ├─ Stage 3: Boilerplate Removal
  │   ├─ Stage 4: Deduplication (within-dataset + cross-dataset)
  │   ├─ Stage 5: Classification (SWE detection + seniority imputation)
  │   ├─ Stage 6: Field Normalization & Validation
  │   ├─ Stage 7: Temporal Alignment
  │   └─ Stage 8: Quality Flags & Filtering
  │        ▼
  │   Analysis-Ready Dataset (unified.parquet)
  │
  ├─ Phase 2: Data Validation
  │   └─ Stage 9: Representativeness, bias, classifiers, distributions (13 checks)
  │
  ├─ Phase 3: Spot-checks (manual, interspersed)
  │
  ├─ Phase 4: Exploration & Discovery
  │   └─ Stage 10: Embeddings, topics, corpus comparison, skill extraction (11 analyses)
  │
  ├─ Phase 5: Formal Analysis
  │   └─ Stage 11: Hypothesis testing for RQ1-RQ7 + robustness protocol
  │
  └─ Phase 6: Statistical Verification
      └─ Stage 12: Power, placebos, controls, alternative explanations, sensitivity (7 checks)
           ▼
      Publication-Ready Results
```

Each preprocessing stage (1-8) produces logged counts (rows in → rows out, rows flagged) so we can report the full data funnel in our methodology section.

---

## Stage 1: Ingest & schema unification

**What exists:** `harmonize.py` already maps both datasets to a 20-column unified schema.

**What needs to change:**

### 1a. SWE_PATTERN consistency

The harmonizer's `SWE_PATTERN` is missing terms that the scraper uses. The scraper includes `ai\s*engineer`, `llm\s*engineer`, `agent\s*engineer`, `applied\s*ai\s*engineer`, `prompt\s*engineer`, `founding\s*engineer`, `member\s*of\s*technical\s*staff`, `product\s*engineer`. The harmonizer does not.

**Action:** Use a single canonical `SWE_PATTERN` defined in one place, imported by both scraper and harmonizer. The expanded pattern (from the scraper) should be the canonical one.

### 1b. Additional columns needed

Add these columns during ingest (they feed downstream stages):

| Column | Purpose | Source |
|--------|---------|--------|
| `source_platform` | linkedin / indeed / kaggle | Split from `source` |
| `salary_source` | employer / platform-imputed / missing | Scraped data has this; Kaggle needs heuristic |
| `description_length` | Character count of clean description | Computed |
| `posting_age_days` | Days between date_posted and scrape_date | Computed |
| `title_normalized` | Lowercased, stripped of level indicators | For dedup |

### 1c. Kaggle companion file joins

Current harmonizer sets `company_industry = None` and `company_size = None` for Kaggle data. Investigation confirmed that companion files exist and are joinable:
- `jobs/job_industries.csv` → join via `job_id` → covers 99.3% of SWE postings
- `companies/companies.csv` → join via `company_id` → provides `company_size`, `state`, `city`
- `companies/employee_counts.csv` → join via `company_id` → provides `employee_count`
- `mappings/industries.csv` → lookup table (422 industry_id → industry_name)

**Action:** Join these during ingest to populate `company_industry` and `company_size` for Kaggle rows. This enables industry-controlled and company-size-controlled comparisons in Stage 9g and Stage 12.

---

## Stage 2: Aggregator / staffing company handling

**Problem identified:** 9% of scraped data comes from aggregators (Lensa: 16 postings, Jobs via Dice: 29, Actalent, TalentAlly, etc.). These create two problems:

1. **Boilerplate contamination**: Lensa prepends ~150 words of self-description ("Lensa is a career site that helps job seekers...") before the actual job content. Dice wraps postings in a "job summary" template.

2. **Employer attribution**: The real employer is buried inside the description (e.g., Lensa posting for Amazon, Dice posting for Raytheon). Company-level analysis would group these under the aggregator rather than the actual employer.

### 2a. Identify aggregators

Maintain a curated list of known aggregator/staffing company names:

```python
AGGREGATORS = {
    'Lensa', 'Jobs via Dice', 'Dice', 'Jobot', 'CyberCoders',
    'Hired', 'ZipRecruiter', 'Actalent', 'TalentAlly', 'Randstad',
    'Robert Half', 'TEKsystems', 'Kforce', 'Insight Global',
    'Motion Recruitment', 'Harnham', 'Jack & Jill',
}
```

**Action:** Flag rows with `is_aggregator = True`. Do NOT remove them by default — instead, run sensitivity analyses with and without aggregator postings.

### 2b. Extract real employer from aggregator descriptions

For Lensa: the actual employer name and job content follows their standard boilerplate. Pattern: strip everything before the first paragraph break after "Lensa partners with DirectEmployers..." or similar.

For Dice: the real content follows "job summary:" and the employer is typically referenced as "Our client is [Company]".

**Action:** Write aggregator-specific boilerplate strippers. Extract `real_employer` where possible. Log extraction success rate.

### 2c. Spot-check requirement

Manually review 20-30 aggregator postings after stripping to verify:
- Boilerplate fully removed
- Real employer correctly identified
- Job content preserved intact

**Decision point:** If aggregator postings are mostly duplicates of direct postings from the same employer (i.e., Amazon posts directly AND Lensa reposts it), they should be deduplicated against the direct posting. If they surface unique jobs not posted directly, keep them.

---

## Stage 3: Boilerplate removal

**Problem:** Job descriptions contain large blocks of repeated text that inflate description length metrics and text similarity scores:
- Company "About Us" sections (repeat across all postings from that company)
- EEO/diversity statements ("We are an equal opportunity employer...")
- Benefits sections ("We offer competitive salary, 401(k)...")
- Application instructions ("To apply, visit...")
- Salary/location appendices (Lensa appends multi-location salary tables)

### 3a. Section-based removal

Most job descriptions follow a loose template:

```
[About the company]      ← boilerplate (repeats per company)
[About the role]         ← KEEP
[Responsibilities]       ← KEEP
[Requirements]           ← KEEP
[Nice-to-haves]          ← KEEP
[Benefits/compensation]  ← boilerplate (repeats per company)
[EEO statement]          ← boilerplate (near-identical across all companies)
[Application info]       ← boilerplate
```

**Action:** Build a section classifier that identifies and tags each section. Keep role/responsibilities/requirements/nice-to-haves. Strip or tag the rest.

**Approach — regex-based section splitting:**

```python
SECTION_HEADERS = [
    (r'(?i)(about\s+(us|the\s+company|our\s+company))', 'about_company'),
    (r'(?i)(about\s+the\s+(role|position|job|opportunity))', 'about_role'),
    (r'(?i)(responsibilities|what\s+you.?ll\s+do|your\s+role|the\s+role)', 'responsibilities'),
    (r'(?i)(requirements?|qualifications?|what\s+you.?ll\s+need|must\s+have|minimum)', 'requirements'),
    (r'(?i)(nice\s+to\s+have|preferred|bonus|plus)', 'nice_to_have'),
    (r'(?i)(benefits?|perks|what\s+we\s+offer|compensation|we\s+offer)', 'benefits'),
    (r'(?i)(equal\s+opportunity|eeo|diversity|we\s+are\s+an?\s+equal)', 'eeo'),
    (r'(?i)(how\s+to\s+apply|to\s+apply|application)', 'application'),
]
```

Store both `description_full` (original) and `description_core` (role + responsibilities + requirements + nice-to-haves only). Use `description_core` for all text analysis; use `description_full` only for section-aware analyses.

### 3b. EEO statement fingerprinting

EEO statements are nearly identical across companies. Build a small set of EEO fingerprints (5-10 common variants) and strip matching paragraphs.

### 3c. Intra-company boilerplate detection

For companies with 3+ postings, compute paragraph-level hashes. Any paragraph appearing in >80% of a company's postings is company boilerplate. Strip it.

**This is important for:** description length analysis (RQ1 scope inflation), text embedding quality, and any NLP that uses full descriptions.

### 3d. Spot-check requirement

After boilerplate removal, manually review 30-50 postings (stratified by company size and source) to verify:
- No actual job requirements were stripped
- All major boilerplate categories are caught
- Description length reduction is consistent (not erratic)

Report: median description length before vs. after boilerplate removal, by source.

---

## Stage 4: Deduplication

**Why this is critical:** Lightcast reports deduplicating up to 80% of raw scraped postings. Without rigorous dedup, volume counts are meaningless, and text analyses are biased toward frequently-reposted jobs.

### 4a. Within-dataset exact dedup

**Scraped data:**
- Already deduplicated by `job_url` within each daily CSV
- Cross-day dedup via `_seen_job_ids.json` (182K IDs)
- **Gap:** Same job posted with different IDs on LinkedIn vs. Indeed. Same role posted by employer directly AND by aggregator. Different locations for the same role at the same company.

**Kaggle data:**
- Dedup status unknown (provenance undocumented)
- Has `job_id` field — check uniqueness

**Action:** Apply exact dedup on `(title_normalized, company_name_normalized, location)` within a 60-day rolling window (industry standard per Lightcast). Log dedup counts.

### 4b. Near-duplicate detection

Same job with slightly different titles ("Software Engineer" vs. "Software Engineer - Remote"), different formatting, or description edits.

**Tiered approach (per Abdelaal et al. 2024, F1=0.94):**

1. **Title similarity**: RapidFuzz `token_set_ratio` ≥ 85 on normalized titles
2. **Company match**: RapidFuzz `token_set_ratio` ≥ 85 on company names (handles "JPMC" / "JPMorgan" / "J.P. Morgan Chase")
3. **Description similarity**: If title+company match, check description cosine similarity ≥ 0.70 (TF-IDF or sentence embeddings)

**Candidates are near-duplicates if:** title match AND company match AND (same location OR description similarity ≥ 0.70).

**Keep rule:** Keep the posting with the most complete fields (non-null salary, seniority label, etc.). If tied, keep the earliest posting.

### 4c. Multi-location posting handling

**Observed:** 38.6% of scraped rows have duplicate title+company combos, often differing only in location. These represent one role posted in multiple cities (e.g., "Software Engineer @ Google" in SF, NYC, Seattle).

**Options:**
- **Option A (default):** Keep all location variants. They represent distinct labor demand in each metro. This is standard in the literature (Hershbein & Kahn 2018 count each posting-location pair).
- **Option B (sensitivity):** Collapse to one row per unique (title, company, description) regardless of location. Report results under both options.

**Action:** Flag `is_multi_location = True` for postings sharing (title, company, description_hash) across 2+ locations. Run sensitivity analyses both ways.

### 4d. LinkedIn "Reposted" handling

Scraped data may include postings marked "Reposted X days ago" by LinkedIn. These are employer-refreshed listings that could be months old. The `date_posted` field may reflect the repost date, not the original date.

**Action:** If the scraped data captures a "reposted" indicator, flag it. For temporal analysis, note that reposted jobs inflate recent-period counts.

### 4e. Dedup reporting

Report a dedup funnel table in the methodology section:

```
                    Kaggle      Scraped     Total
Raw rows            123,849     ~187,000    ~311,000
SWE title match     3,466       14,391      17,857
After exact dedup   X           X           X
After near-dedup    X           X           X
After aggregator    X           X           X
  dedup
Final SWE           X           X           X
Final Control       X           X           X
```

---

## Stage 5: Classification

### 5a. SWE detection — multi-tier approach

**Problem with regex-only:** A regex on titles is fast and transparent but fundamentally brittle. It fails on novel titles, misses titles with non-standard phrasing ("Software Development Engineer", Amazon's standard SWE title — 16+ false negatives per day), and can't handle ambiguous cases where the title alone is insufficient ("Engineer" at a software company vs. "Engineer" at a construction company). The regex approach also drifts over time as new title conventions emerge (e.g., "AI Agent Developer" in 2026 didn't exist in 2024).

**Our classification needs:** We need a binary SWE/non-SWE classifier that:
- Handles the long tail of title variations
- Works consistently across both time periods (April 2024 and March 2026)
- Is transparent and reproducible (reviewers need to understand it)
- Provides confidence scores (not just binary labels) for sensitivity analysis
- Can also classify into SWE / SWE-adjacent / control / other for the DiD design

**Proposed: 3-tier classification pipeline**

```
Tier 1: Regex (fast, deterministic, handles obvious cases)
  ↓ unmatched or low-confidence titles
Tier 2: Embedding similarity (handles the long tail)
  ↓ still ambiguous
Tier 3: Description-based classification (resolves edge cases)
  ↓
Final label + confidence score
```

#### Tier 1: Improved regex (handles ~85% of postings)

The regex serves as the fast, deterministic first pass. Most SWE titles are unambiguous and a well-designed regex catches them.

```python
SWE_INCLUDE = re.compile(
    r'(?i)\b('
    r'software\s*(engineer|developer|dev|development\s*engineer)|'
    r'swe\b|full[- ]?stack|front[- ]?end\s*(engineer|developer)|'
    r'back[- ]?end\s*(engineer|developer)|'
    r'web\s*(developer|engineer)|mobile\s*(developer|engineer)|'
    r'devops\s*(engineer)?|platform\s*engineer|'
    r'data\s*engineer|ml\s*engineer|machine\s*learning\s*engineer|'
    r'site\s*reliability\s*(engineer)?|'
    r'ai\s*engineer|llm\s*engineer|agent\s*engineer|'
    r'applied\s*(ai|ml)\s*engineer|prompt\s*engineer|'
    r'infrastructure\s*engineer|cloud\s*engineer|'
    r'founding\s*engineer|member\s*of\s*technical\s*staff|'
    r'product\s*engineer|systems?\s*engineer'
    r')\b'
)

SWE_EXCLUDE = re.compile(
    r'(?i)\b('
    r'sales\s*engineer|support\s*engineer|field\s*(service|engineer)|'
    r'customer\s*(success|support)\s*engineer|'
    r'solutions?\s*(architect|engineer)|'
    r'systems?\s*administrator|'
    r'civil\s*engineer|mechanical\s*engineer|electrical\s*engineer|'
    r'chemical\s*engineer|industrial\s*engineer|'
    r'audio\s*engineer|recording\s*engineer|sound\s*engineer|'
    r'network\s*engineer|hardware\s*engineer'
    r')\b'
)
```

Regex output: `swe_regex = True / False / excluded`. Titles that match `SWE_INCLUDE` and NOT `SWE_EXCLUDE` are classified SWE with high confidence. Excluded titles are classified non-SWE with high confidence. Everything else goes to Tier 2.

#### Tier 2: Embedding similarity to SOC reference titles (handles ~10% more)

**Concept:** Encode all job titles using a sentence transformer, then measure cosine similarity to a curated set of SWE reference titles drawn from O\*NET SOC codes.

**SOC codes for SWE (from O\*NET/BLS):**
- 15-1252: Software Developers (primary)
- 15-1253: Software Quality Assurance Analysts and Testers
- 15-1254: Web Developers
- 15-1255: Web and Digital Interface Designers (borderline)
- 15-1211: Computer Systems Analysts (borderline)
- 15-1256: Software Quality Assurance Analysts (overlap with 15-1253)
- 15-1299: Computer Occupations, All Other

**Reference title set:** For each SOC code, O\*NET provides "sample reported titles" — the actual titles real workers use. For 15-1252 alone: "Application Developer", "Application Integration Engineer", "Developer", "DevOps Engineer", "Infrastructure Engineer", "Software Architect", "Software Developer", "Software Development Engineer", "Software Engineer", "Systems Engineer". Combine these across all SWE-relevant SOC codes to build a reference set of ~50-80 canonical SWE titles.

**Implementation:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('TechWolf/JobBERT-v2')  # job-domain fine-tuned, 1024d

# Encode reference titles (once, cached)
swe_refs = [
    "Software Engineer", "Software Developer", "Software Development Engineer",
    "Full Stack Developer", "Frontend Engineer", "Backend Engineer",
    "DevOps Engineer", "Data Engineer", "ML Engineer", "AI Engineer",
    "Site Reliability Engineer", "Platform Engineer", "Cloud Engineer",
    "Mobile Developer", "Infrastructure Engineer", "Software Architect",
    # ... expand from O*NET sample titles for SOC 15-1252 through 15-1256
]
ref_embeddings = model.encode(swe_refs)

# For each unresolved title from Tier 1:
title_embedding = model.encode(title)
similarities = cosine_similarity([title_embedding], ref_embeddings)[0]
max_similarity = similarities.max()

# Classification:
# > 0.70: SWE (high confidence)
# 0.50 - 0.70: SWE-probable (send to Tier 3 for confirmation)
# < 0.50: not SWE
```

**Why this works better than regex:**
- "Software Development Engineer" → high similarity to "Software Developer" and "Software Engineer" (catches the Amazon false negative)
- "Applied AI Research Scientist" → moderate similarity (correctly ambiguous — Tier 3 resolves)
- "Mechanical Integration Leader" → low similarity (correctly excluded)
- Robust to novel title phrasings because it measures semantic similarity, not lexical patterns
- The reference set is grounded in an authoritative taxonomy (O\*NET), not researcher intuition

**Threshold calibration:** Run Tier 2 on the titles where we already have regex labels (high-confidence SWE and high-confidence non-SWE from Tier 1). Use this to calibrate the similarity thresholds — find the threshold that maximizes agreement with regex on unambiguous cases. Then apply that threshold to ambiguous cases.

#### Tier 3: Description-based classification (resolves the remaining ~5%)

For titles still ambiguous after Tier 2 (similarity 0.50-0.70, or contradictory signals), use the job description to make the final call.

**Option A: Zero-shot NLI classifier (no training needed)**

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

result = classifier(
    description_core[:512],  # truncate to first 512 tokens
    candidate_labels=["software engineering job", "non-software engineering job"],
    hypothesis_template="This is a {}."
)
# result['scores'] gives probability for each label
```

**Pros:** No training data needed. Works out of the box. Reproducible.
**Cons:** Slow (~0.5s per posting on CPU). May be overconfident on boilerplate-heavy descriptions.

**Option B: SetFit few-shot classifier (small training set, fast inference)**

```python
from setfit import SetFitModel, SetFitTrainer

# Train on ~50-100 labeled examples (25 SWE, 25 non-SWE, 50 ambiguous)
model = SetFitModel.from_pretrained("TechWolf/JobBERT-v2")
trainer = SetFitTrainer(model=model, train_dataset=labeled_set)
trainer.train()

# Inference is fast (same speed as sentence transformer encoding)
predictions = model.predict(descriptions)
```

**Pros:** Needs only 50-100 labeled examples to reach ~90%+ accuracy. Fast inference after training. Learns from description content, not just title.
**Cons:** Requires a labeled training set (but we need one anyway for validation).

**Option C: SOC code assignment via `occupationcoder` (external tool)**

The [occupationcoder](https://github.com/aeturrell/occupationcoder) package (Bank of England) assigns SOC codes given title + description + sector. We could assign SOC codes to all postings and define SWE as any posting with SOC 15-12XX codes.

**Pros:** Grounded in an official taxonomy. Assigns fine-grained codes we can use for other analyses.
**Cons:** Uses UK SOC 2010 (not US SOC 2018). Bag-of-words approach, less accurate than embeddings. Would need adaptation for US occupational codes.

**Recommendation:** Use Option B (SetFit) for Tier 3. It requires minimal labeled data, inference is fast, and it learns from descriptions — which is exactly what we need for ambiguous titles. The labeled examples we create for validation (Stage 9e) can double as training data.

#### Tier output and confidence scoring

Every posting gets three fields:

| Field | Values | Source |
|---|---|---|
| `is_swe` | True / False | Final binary label |
| `swe_confidence` | 0.0 - 1.0 | Confidence score |
| `swe_classification_tier` | regex / embedding / description | Which tier made the call |

Confidence scoring:
- Tier 1 regex match (include, no exclude): `confidence = 0.95`
- Tier 1 regex match (exclude): `confidence = 0.95` (confident non-SWE)
- Tier 2 embedding similarity > 0.70: `confidence = similarity_score`
- Tier 2 embedding similarity < 0.50: `confidence = 1.0 - similarity_score`
- Tier 3 SetFit/NLI: `confidence = model_probability`

**Sensitivity analysis:** Run core analyses at confidence thresholds of 0.50 (inclusive, more postings), 0.70 (moderate), and 0.90 (strict, fewer postings). If findings hold across all thresholds, the classification isn't driving the results.

#### SWE-adjacent vs. control taxonomy

The same 3-tier approach classifies into a broader taxonomy needed for DiD (RQ4):

| Category | SOC codes | Examples |
|---|---|---|
| **SWE** (treatment) | 15-1252, 15-1253, 15-1254 | Software Engineer, QA Engineer, Web Developer |
| **SWE-adjacent** (AI-exposed tech) | 15-1211, 15-1221, 15-1231, 15-1241, 15-1299 | Data Scientist, Sysadmin, Data Analyst, DBA |
| **Control** (low AI-exposure) | 17-2051, 17-2141, 29-1141, 13-2011 | Civil Engineer, Mechanical Engineer, Registered Nurse, Accountant |
| **Other** | Everything else | |

For the embedding approach, build separate reference title sets for each category. A posting's category is the one with the highest similarity. This replaces the current approach of having separate regex patterns for SWE and control.

#### Validation protocol for the classifier

1. **Gold-standard annotation:** Sample 200 postings stratified by classification tier and confidence level. Oversample from the ambiguous zone (confidence 0.40-0.70). Have 2 annotators label SWE / SWE-adjacent / control / other. Compute Cohen's kappa.

2. **Per-tier accuracy:** Report precision/recall/F1 for each tier separately. If Tier 1 (regex) has 98% precision but Tier 3 (description) has 75% precision, that's important context.

3. **Cross-period consistency:** Run the classifier on both datasets. If the SWE detection rate (SWE postings / total postings) differs dramatically between Kaggle and scraped, investigate whether that's a real labor market change or a classification artifact. Compare against known SWE rates from BLS OES (SOC 15-1252 as % of all occupations).

4. **Error analysis:** For every false positive and false negative in the gold-standard sample, categorize the error type:
   - Title ambiguity (e.g., "Engineer" without qualifier)
   - Novel title (e.g., "AI Agent Orchestrator" — not in reference set)
   - Aggregator confusion (e.g., staffing company title differs from actual role)
   - Description mismatch (title says SWE but description is project management)

5. **Comparison with SOC-based approach:** As a robustness check, run `occupationcoder` or a similar SOC mapper on a sample and compare its SWE classification against ours. Agreement rate establishes external validity.

#### Implementation notes

- **Speed:** Tier 1 (regex) processes all ~200K postings in seconds. Tier 2 (embedding with JobBERT-v2) processes ~5K titles per minute on CPU. Tier 3 (SetFit) processes ~3K descriptions per minute on CPU. Total pipeline: ~15-20 minutes for full dataset.
- **Reproducibility:** Pin model versions (e.g., `TechWolf/JobBERT-v2` from HuggingFace, specific commit hash). Cache all embeddings. Log the reference title sets. This ensures re-running the pipeline produces identical results.
- **The reference title set is a researcher decision.** Document it exhaustively. It should be grounded in O\*NET sample titles, supplemented with titles observed in our data that were manually verified as SWE. Publish it in the appendix.

### 5b. Seniority imputation

**Current approach:** Rule-based classifier using title keywords (primary) + description years-of-experience (fallback). Validated at ~80% accuracy against LinkedIn's own labels on the labeled subset.

**What needs to happen for rigor:**

1. **Apply uniformly:** The same `impute_seniority()` function must run on both Kaggle and scraped data. Currently, the notebook applies it only to scraped data and uses Kaggle's native `formatted_experience_level` directly. This creates a systematic difference — we'd be comparing LinkedIn's classifier (April 2024) against our classifier (March 2026).

2. **Gold-standard validation:** Sample 300-500 postings stratified by (source, predicted seniority, title ambiguity). Have 2 annotators label independently. Compute Cohen's kappa. Evaluate per-class precision/recall against this gold standard.

3. **Report imputation rate:** What fraction of each dataset required imputation vs. had a native label? If the imputation rate differs between periods (e.g., 27% in 2026 vs. X% in Kaggle), that itself is a measurement artifact that could drive apparent seniority shifts.

4. **Three-level bucketing:** For cross-period comparison, use the 3-level scheme (junior / mid / senior) rather than the 6-level scheme. Finer granularity amplifies classifier noise.

**Decision point:** Should we use our `impute_seniority()` for ALL postings (ignoring native labels), or use native labels where available and impute only where missing? The research docs recommend the former (retroactive classification — apply one classifier uniformly). This is the Lightcast approach.

**Recommendation:** Use our imputer as the canonical seniority variable. Keep native labels as a separate column for validation and sensitivity analysis.

**Future upgrade path:** The same 3-tier approach used for SWE detection (regex → embedding → description classifier) could be applied to seniority. Tier 2 would measure similarity to reference titles per seniority level (O\*NET provides "Job Zone" ratings that map to seniority). Tier 3 would use a SetFit classifier trained on LinkedIn's native labels. This is lower priority than improving SWE detection because (a) our rule-based imputer already validates at ~80% against LinkedIn labels and (b) the 3-level bucketing (junior/mid/senior) is coarse enough to absorb most misclassification noise. But if seniority classification accuracy becomes a reviewer concern, this is the path forward.

### 5c. Control occupation detection

**Current approach:** `CONTROL_PATTERN` regex for civil/mechanical/electrical/chemical engineers and nurses.

**Risk:** Control occupations may have very different LinkedIn posting rates than SWE. The DiD design (RQ4) requires adequate coverage in both datasets.

**Action:** After classification, report control occupation counts by source. If any control group has <100 postings in either period, it's too thin for DiD and should be flagged.

---

## Stage 6: Field normalization & validation

### 6a. Company name standardization

**Problem:** Same company appears as "JPMorgan Chase", "JPMC", "J.P. Morgan Chase & Co.", "JP Morgan".

**Action:** RapidFuzz `token_set_ratio` with threshold ~85 to group variants. Build a lookup table of canonical company names. Needed for firm-level analysis and intra-company boilerplate detection (Stage 3).

### 6b. Location normalization

**Problem:** Locations appear as "San Francisco, CA", "San Francisco Bay Area", "SF Bay Area", "San Francisco, California, United States". Remote jobs have inconsistent location strings.

**Action:** Normalize to `(metro area, state, country)` tuples. Map metro areas. Flag remote postings separately (don't conflate remote with any geographic metro).

### 6c. Salary validation (low priority)

**Note:** Salary is NOT a primary analysis variable for this study. Our core RQs focus on seniority distributions, skill migration, and structural breaks — none of which depend on salary data. Salary analysis is secondary and conditional on data availability.

**Problem:**
- Kaggle salary: 27.9% coverage for SWE (mostly annual, some hourly)
- Scraped LinkedIn: 4% coverage (near-useless)
- Scraped Indeed: 76% coverage (but these are platform-imputed estimates in many cases, not employer-provided)
- Platform-imputed wages inflated apparent coverage by ~520% per Hazell & Taska (2023)

**Action (if salary is used at all):**
1. Flag `salary_source` = employer vs. platform-imputed vs. missing. The scraped data has a `salary_source` field — use it.
2. Winsorize or flag salary outliers: annual salary outside $20K-$1M for SWE roles is suspicious.
3. Do NOT impute missing salaries. Report salary coverage rate per period and treat any salary analysis as conditional on observability with explicit MNAR caveat (Azar et al. 2022, Hazell & Taska 2023).
4. If salary analysis is attempted, restrict to Indeed-only (76% coverage) or employer-provided-only subsets, and caveat heavily.

### 6d. Date validation

**Problem:**
- Kaggle `listed_time` range: April 5-20, 2024 (scrape window). `original_listed_time`: 98.5% April, 1.5% March, 19 rows Dec 2023-Feb 2024 (long-lived postings still active when scraped).
- Scraped `date_posted` may be null or reflect repost dates.

**Action:** Validate all dates fall within expected ranges (Kaggle: 2024-03-24 to 2024-04-20; scraped: 2026-03-01 to 2026-03-31). Flag and investigate out-of-range dates. For the 19 Kaggle rows with pre-March `original_listed_time`, keep them but flag — they are persistent postings, not representative of their original posting date.

### 6e. Description language detection

**Assumption:** All postings are English-language US jobs. But scraped data from LinkedIn may include non-English postings or non-US locations that slipped through geographic filters.

**Action:** Run a lightweight language detector (e.g., `langdetect`) on descriptions. Flag non-English postings. Report the rate and exclude from text analysis.

---

## Stage 7: Temporal alignment

### 7a. The date range problem (CRITICAL)

**The Kaggle data is a single-month snapshot (April 2024), not a multi-year dataset.** Same-month comparison (March vs. March) is impossible. Our actual comparison is:

| | Kaggle | Scraped |
|---|---|---|
| Date range | April 2024 | March 2026 |
| Gap | 23 months |
| Seasonal offset | April vs. March (1 month) |
| SWE sample | ~3,466 | ~14,391 unique |

**Primary comparison:** April 2024 vs. March 2026. The 1-month seasonal offset (March vs. April) is small but must be disclosed. Both are spring hiring months. The 23-month gap is the feature we're studying, not a confounder.

**Seasonal concern:** If we find differences, a reviewer will ask: "Is this just April-to-March seasonality?" Mitigation: compare BLS/JOLTS seasonal patterns for these months and show that SWE-specific metrics don't typically shift between March and April. Also run the same comparison on control occupations — if they don't show the same shift, it's not seasonality.

**Action:** Create a `period` column with values `2024-04` and `2026-03`. Document the actual date range prominently. Correct all prior documentation that says "2023-2024".

### 7b. Proportions over counts

All cross-period metrics must be expressed as shares:
- Junior share = junior SWE postings / total SWE postings
- Skill prevalence = postings mentioning skill X / total SWE postings at seniority level Y
- Never compare raw counts between periods with different sample sizes

The Kaggle SWE sample (~3,466) is ~4.2x smaller than the scraped SWE sample (~14,391). This asymmetry affects confidence intervals but not point estimates when using proportions.

### 7c. Within-period stability (scraped data only)

We have 14 days of scraped data. Check whether the daily distributions are stable:
- Does the seniority mix change day-to-day?
- Does the SWE/non-SWE ratio change?
- Are there day-of-week effects?

If daily distributions are stable, we can pool all 14 days confidently. If they fluctuate, we should report the variance and consider date-level fixed effects.

---

## Stage 8: Quality flags & filtering

### 8a. Ghost job detection

Per CRS Report IF12977 (2025), 18-27% of US online job listings are estimated to be ghost jobs. The rate is higher for entry-level tech.

**Heuristic flags:**
- Entry-level title + 5+ years experience required in description
- Entry-level title + salary > $180K (75th percentile for mid-senior SWE)
- Posting age > 90 days (if computable)
- No application URL or instructions

We need to research more characteristics of ghost job postings to apply better filtering.

**Action:** Flag `ghost_job_risk = low | medium | high`. Do not remove — run sensitivity analyses with and without high-risk flagged postings.

### 8b. Minimum description quality

- Flag postings with `description_core` < 50 words after boilerplate removal (likely incomplete or template-only)
- Flag postings where description is entirely boilerplate (no core job content detected)

### 8c. Data provenance flags

For every row, record:
- `preprocessing_version`: Pipeline version (for reproducibility)
- `dedup_method`: How it survived dedup (first seen / kept by completeness / etc.)
- `seniority_source`: native_label | imputed_title | imputed_description | unknown
- `is_aggregator`: Whether the posting came from a staffing/aggregator company
- `boilerplate_removed`: Whether boilerplate stripping was applied

---

## Stage 9: Data validation

This is the core validation battery. Every check here answers a question a reviewer would ask. Results feed into the methodology section and determine whether downstream analyses are defensible.

The validation stage runs AFTER preprocessing (Stages 1-8) is complete, on the cleaned unified dataset. Some checks may trigger iteration on earlier stages (e.g., if representativeness is poor, revisit dedup thresholds or geographic filtering).

### 9a. Representativeness: Do our scraped data look like the real labor market?

**Why a reviewer asks:** "Your data is whatever LinkedIn's algorithm gave you. How do you know it reflects actual job openings?"

**Benchmark sources:**
- **JOLTS** (via FRED): Total job openings, Professional & Business Services, Information sector. Free CSV download. Establishes macro-level plausibility.
- **BLS OES** (Occupational Employment & Wage Statistics): Occupation × metro employment counts and wages. The gold standard for occupation-level validation.
- **Revelio Labs** (already ingested): SOC-15 aggregate hiring, openings, salary trends.

**Tests to run:**

| Test | What it measures | Acceptable threshold | Reference |
|---|---|---|---|
| **Occupation share correlation** | Pearson r between our occupation distribution and OES employment shares | r > 0.80 (Hershbein & Kahn got 0.84-0.98) | Hershbein & Kahn (2018) |
| **Industry share comparison** | Our SWE industry distribution vs. OES SWE industry distribution | Within 5pp per industry | OECD (2024) |
| **Geographic share correlation** | Our state-level SWE counts vs. OES state-level SWE employment | r > 0.80 | Hershbein & Kahn (2018) |
| **Dissimilarity index** | Duncan index between our occupation distribution and OES | Report and discuss (no hard cutoff) | OECD (2024) |
| **Posting volume vs. JOLTS** | Time-series correlation between our daily posting counts and JOLTS information sector openings | r > 0.60 (Turrell et al. got 0.65-0.95) | Turrell et al. (2019) |
| **Wage distribution check** | KS test comparing our salary distribution against OES wage percentiles for SOC 15-1252 (Software Developers) | p > 0.05 (non-rejection = compatible) | Hazell & Taska (2023) |

**Implementation:**
1. Pull BLS OES data for SOC 15-1252 (Software Developers) and SOC 15-1256 (Software Quality Assurance): employment by state, median wage, industry concentration.
2. Pull JOLTS information sector series from FRED.
3. Compute our distributions on the cleaned unified dataset.
4. Run correlations and report in a representativeness table.

**Do this separately for Kaggle and scraped datasets.** They have different selection mechanisms and may have different representativeness profiles.

### 9b. Cross-dataset comparability: Are Kaggle and scraped data measuring the same thing?

**Why a reviewer asks:** "You're comparing two datasets with unknown/different collection methods. Any difference you find could be an artifact of the data, not the labor market."

This is the single most important validation for our study. We must demonstrate that differences between Kaggle (April 2024) and scraped (March 2026) data reflect real labor market changes, not measurement artifacts.

**Tests to run:**

| Test | What it measures | What we hope to see | What would be concerning |
|---|---|---|---|
| **Description length distribution** | KS test on `description_core` character counts | p > 0.05 OR explainable difference | Large difference that could be scraping artifact |
| **Company overlap** | Jaccard similarity of company names across datasets | > 0.20 for top-100 companies | Complete non-overlap would suggest different market segments |
| **Geographic distribution** | Chi-squared test on state-level shares | Similar state rankings, proportional differences | Completely different geographic profiles |
| **Seniority distribution (native labels)** | Compare LinkedIn's native seniority label distributions | Similar distributions (entry/mid/senior proportions within 5pp) | Massive shifts that could indicate LinkedIn changed its labeling |
| **Industry distribution** | Chi-squared test on industry shares (after joining Kaggle companion data) | Similar industry mix | One dataset dominated by an industry absent from the other |
| **Title vocabulary overlap** | Jaccard similarity of unique job titles | High overlap for common titles | New title categories in 2026 not present in 2024 (expected but should be quantified) |
| **Company size distribution** | KS test on employee counts | Similar distributions | One dataset overrepresenting small/large companies |

**Key confounders to investigate:**

**LinkedIn platform changes (2024 → 2026):** Did LinkedIn change its job posting display, categorization, search algorithm, or seniority labeling between April 2024 and March 2026? Platform changes create artificial differences that look like labor market changes. Check LinkedIn's published changelog, engineering blog, and press releases for relevant product updates.

**Indeed vs. LinkedIn composition effect:** The scraped data is 60% LinkedIn + 40% Indeed. The Kaggle data is 100% LinkedIn. Any Kaggle-vs-scraped difference could partly reflect Indeed-vs-LinkedIn differences, not temporal changes. **Mitigation:** Run all cross-period comparisons on the LinkedIn-only subset of scraped data first, then check whether including Indeed changes results.

**Scraper query design effect:** Our scraper runs specific queries ("software engineer", "full stack engineer", etc.) in specific cities. The Kaggle dataset was collected with unknown queries and geographic scope. Different query strategies surface different jobs even on the same day. **Mitigation:** Document this as a limitation. Compare the title distributions to see if one dataset captures roles the other misses.

### 9c. Missing data audit

**Why a reviewer asks:** "With 76% salary missing in Kaggle and 96% missing from LinkedIn scrapes, any salary-based conclusion is drawn from a tiny, non-random subset."

**Produce a missing data table:**

| Field | Kaggle (n=~3,466 SWE) | Scraped LinkedIn (n=~10K SWE) | Scraped Indeed (n=~4K SWE) |
|---|---|---|---|
| Title | % | % | % |
| Description | % | % | % |
| Company name | % | % | % |
| Location | % | % | % |
| Seniority (native) | % | % | % |
| Salary (any) | % | % | % |
| Skills (structured) | % | % | % |
| Industry | % | % | % |
| Company size | % | % | % |
| Date posted | % | % | % |

**Missingness mechanism analysis (for salary):**

Per Hazell & Taska (2023), salary missingness is almost certainly MNAR (Missing Not At Random). Run these diagnostics:

1. **`missingno` heatmap**: `msno.heatmap(df)` — reveals whether salary missingness correlates with seniority, company size, industry. If correlations are strong, missingness is at least MAR.
2. **Salary missingness by seniority**: Compute the salary disclosure rate per seniority level. If entry-level discloses at a different rate than senior, salary-based seniority comparisons are biased.
3. **Salary missingness by company size**: Large companies may disclose differently than small ones.
4. **Salary missingness by platform**: Indeed (76%) vs. LinkedIn (4%) is a massive platform effect. Platform-provided salary estimates must be distinguished from employer-provided salaries.

**Decision:** Treat salary as a secondary analysis variable, not a primary one. All core analyses (seniority shifts, skill migration, structural break) should NOT depend on salary. Salary analysis is conditional on observability, with explicit caveat citing Hazell & Taska.

### 9d. Selection bias diagnostics

**Why a reviewer asks:** "Your data only captures jobs posted online, through specific queries, on specific platforms. How do you know your findings generalize?"

**The five selection mechanisms in our data (from research docs):**

1. **Platform selection**: LinkedIn overrepresents BA+ professional jobs. For SWE roles specifically, coverage is high (>80% posted online per Carnevale et al. 2014). But control occupations (nursing, civil engineering) have lower online posting rates — this confounds the DiD design.

2. **Algorithm selection**: LinkedIn's ranking algorithm optimizes for engagement, not representativeness. Promoted (paid) posts get 3-5x visibility. Our scraper captures what the algorithm surfaces, not a random sample.

3. **Scraper selection**: Our 28 queries × 20 cities × 25 results/query design creates deterministic gaps:
   - Max 25 results per query-city combo means we miss long-tail postings
   - 20 cities misses smaller metros entirely
   - Query tier design means roles that don't match any query are excluded

4. **Employer selection**: Staffing companies (Lensa, Dice, etc.) inflate some companies' representation. DataAnnotation has 168 postings in the Kaggle SWE set (5.4%) — likely an outlier.

5. **Temporal selection (volatility bias)**: Daily scraping oversamples longer-lived postings (Foerderer 2023). A job open for 60 days is 60x more likely to appear in any daily scrape than a 1-day posting. This biases toward hard-to-fill roles.

**Tests to run:**

| Test | What it checks | How |
|---|---|---|
| **Covariate balance (ASMD)** | Whether scraped data distributions match BLS benchmarks | Meta's `balance` package. ASMD < 0.1 is acceptable per Stuart et al. (2013) |
| **Company size distribution vs. BLS** | Whether we over-represent large firms | Compare our company size distribution against Census SUSB (Statistics of U.S. Businesses) for NAICS 5112 (Software Publishers) |
| **Geographic coverage map** | Whether our 20-city design biases results | Plot SWE postings by metro. Compare against OES metro-level SWE employment |
| **Query saturation check** | Whether 25 results/query is enough | For key queries, re-scrape with higher limits (50, 100) and compare distributions |
| **Posting duration analysis** | Whether we oversample long-lived postings | If `date_posted` is available, compute posting duration distribution. Compare against Foerderer (2023) benchmarks |

**Covariate balance protocol (from research docs):**

```python
from balance import Sample
sample = Sample.from_frame(scraped_df[['seniority', 'company_size', 'industry', 'state']])
target = Sample.from_frame(bls_benchmark_df[['seniority', 'company_size', 'industry', 'state']])
adjusted = sample.set_target(target).adjust()
# Reports ASMD per covariate — threshold: ASMD < 0.1
```

If ASMD > 0.1 for key covariates, apply inverse-probability-of-selection weighting (IPSW) to reweight our sample toward the BLS benchmark. Report results both with and without reweighting.

### 9e. Classifier validation

**Why a reviewer asks:** "Your entire study depends on correctly classifying jobs as SWE and correctly imputing seniority. How accurate are these classifiers?"

**SWE detection validation:**

1. Sample 100 postings from each dataset: 50 classified as SWE (true positive candidates) + 50 classified as non-SWE (false negative candidates, selected from titles containing "engineer", "developer", "software", "tech" that didn't match the pattern).
2. Human annotator labels each as SWE / not-SWE.
3. Report precision, recall, F1.
4. **Known issues to check:**
   - Does "Software Development Engineer" match? (It's in non-SWE files currently — appears 16 times in March 18 non-SWE file. This is a false negative.)
   - Does "Product Engineer" match in both datasets consistently?
   - Does "Language Engineer, Artificial General Intelligence" count as SWE? (Appears 33 times in non-SWE file.)

**Seniority classifier validation:**

Per the research docs, create a gold-standard validation set:

1. Sample 300-500 postings stratified by: source (Kaggle vs. scraped), predicted seniority (entry/mid/senior), and title ambiguity (clear titles like "Senior Engineer" vs. ambiguous titles like "Software Engineer").
2. Have 2 annotators independently label each posting's seniority from title + description.
3. Compute inter-rater reliability: Cohen's kappa ≥ 0.67 tentatively acceptable, ≥ 0.80 good.
4. Adjudicate disagreements.
5. Evaluate our classifier against this gold standard. Report per-class precision/recall/F1.
6. **Critical check:** Run the same classifier on Kaggle data where LinkedIn native labels exist. Compare our imputation against LinkedIn's labels. If our classifier agrees with LinkedIn at a different rate for different seniority levels, that differential error biases cross-period comparisons.

**Classifier temporal stability:**

Our seniority classifier was designed from 2026 posting conventions. It may perform differently on 2024 data if title conventions changed. **Test:** Compute per-class accuracy on Kaggle (where native labels are available) and on scraped LinkedIn (where native labels are available). If accuracy differs significantly between periods, the classifier introduces a temporal artifact.

### 9f. Distribution comparisons for key analysis variables

**Why a reviewer asks:** "Before I believe your cross-period findings, show me the raw distributions. Are you comparing normal distributions? Skewed? Bimodal?"

For each key variable, produce distribution plots (histograms or KDEs) side by side for Kaggle vs. scraped, and run formal distribution comparison tests:

| Variable | Test | Why |
|---|---|---|
| Description length (chars) | KS test + QQ plot | RQ1 scope inflation proxy — must rule out scraping artifact |
| Description length after boilerplate removal | KS test + QQ plot | The apples-to-apples version |
| Seniority distribution | Chi-squared test | RQ1 core metric |
| Salary (where available) | KS test | Wage trends |
| Company size | KS test | Composition control |
| Word count of requirements section only | KS test | More targeted scope inflation measure than full description |
| Number of distinct skills mentioned | KS test | Skill breadth index |
| Years of experience required | KS test | Direct seniority requirement measure |
| Remote work rate | Proportion test | Compositional difference |

**Interpretation framework:** A statistically significant difference is NOT automatically evidence of labor market change. It could also indicate:
- Scraping method differences
- Platform changes
- Company composition differences
- Seasonal variation

For each significant difference, attempt to decompose it: how much is explained by composition (different companies, industries, geographies) vs. within-composition change?

### 9g. Compositional analysis: Is the comparison apples-to-apples?

**Why a reviewer asks:** "Maybe the seniority distribution shifted not because junior jobs disappeared, but because your 2026 scraper happened to capture a different set of companies than the 2024 Kaggle dataset."

**Decomposition approach:**

1. **Company overlap analysis:** Identify companies appearing in both datasets. For the overlapping set, compare seniority distributions. If the shift holds within the same companies, it's not a composition effect.

2. **Industry-controlled comparison:** Compare seniority distributions within matched industries (e.g., SWE postings in "Technology/Information" only, excluding healthcare SWE, finance SWE, etc.).

3. **Geography-controlled comparison:** Compare within matched metros (e.g., San Francisco SWE only, NYC SWE only).

4. **Company-size-controlled comparison:** Compare within matched size bands (e.g., large companies >10K employees only).

5. **Oaxaca-Blinder decomposition** (if warranted): Formally decompose the cross-period difference in any outcome (seniority share, skill prevalence) into:
   - A composition effect (different mix of companies/industries/geographies)
   - A within-composition effect (same companies posting differently)
   This is the gold standard in labor economics for separating "who's hiring" from "what they're hiring for."

### 9h. Company concentration analysis and normalization

**Why a reviewer asks:** "Your scraped data top-5 companies account for 30% of SWE postings. How do you know your findings aren't driven by the hiring patterns of a handful of large employers?"

**The problem:** Company concentration differs between datasets (Kaggle top-5 = 12.6%, scraped top-5 = 30.2%). A company like DataAnnotation (168 Kaggle SWE postings, 5.4%) may have unusual seniority distributions, skill requirements, or description styles that skew aggregate metrics. If one dataset is dominated by a few companies that the other doesn't have, cross-period differences may reflect company composition, not labor market change.

**Diagnostic steps:**

1. **Compute concentration metrics per dataset:**
   - Herfindahl-Hirschman Index (HHI): Sum of squared posting-share per company. HHI > 0.15 = moderately concentrated, > 0.25 = highly concentrated.
   - Top-5 / top-10 / top-20 company share of total SWE postings
   - Gini coefficient of company posting counts

2. **Identify dominant companies and audit them:**
   - For any company with >3% of SWE postings in either dataset, manually review 5-10 postings to check:
     - Are these real SWE jobs or annotation/crowdwork (e.g., DataAnnotation)?
     - Do they use standardized templates that inflate similarity?
     - Is the seniority distribution unusual?
   - Flag companies that are functionally aggregators or crowdwork platforms even if not in the AGGREGATORS list

3. **Within-company vs. between-company decomposition:**
   - For companies appearing in BOTH datasets, compare their seniority distributions across periods. If the shift holds within the same companies, it's not a composition effect.
   - Compute the cross-period seniority shift (a) on the full sample, (b) on the overlapping-companies-only sample, and (c) on the non-overlapping sample. If (b) shows the same shift as (a), company composition is not driving the finding.

4. **Company-capped analysis (sensitivity):**
   - Cap each company at N postings (e.g., N = 10 or N = median company count) to prevent any single company from dominating.
   - Re-run key analyses on the capped sample. If findings hold, they're not driven by a few prolific posters.

5. **Company-level fixed effects (for regression analyses):**
   - Include company fixed effects in any regression model. This absorbs all between-company variation and isolates within-company changes over time.
   - Only works for companies appearing in both periods — report the overlap rate.

6. **Exclusion sensitivity tests:**
   - Re-run analyses excluding the top-5 companies from each dataset
   - Re-run excluding all aggregators/staffing companies
   - Re-run excluding DataAnnotation specifically (suspected crowdwork platform)

**Reporting:** Include a company concentration table in the methodology section:

| Metric | Kaggle SWE | Scraped SWE |
|---|---|---|
| Unique companies | X | X |
| Top-1 company share | X% | X% |
| Top-5 share | 12.6% | 30.2% |
| Top-10 share | X% | X% |
| HHI | X | X |
| Company overlap (Jaccard) | X | |
| Companies in both datasets | X | |

### 9i. Temporal stability and seasonality checks

**Why a reviewer asks:** "With only two snapshots 23 months apart, how do you separate genuine structural change from normal seasonal or cyclical variation?"

**Checks:**

1. **JOLTS seasonal pattern:** Plot JOLTS information sector openings by month. Show that March-to-April variation is small relative to the cross-year change we observe. If BLS data shows a typical March-April seasonal swing of ±5% but we observe a 20% shift in junior share, seasonality alone cannot explain it.

2. **Within-period stability (scraped data):** We have 14 daily scrapes. Compute daily seniority distributions and test for day-to-day stability. If the distribution is stable within our 2-week window, it's unlikely that a 1-month seasonal offset (March vs. April) drives our findings.

3. **Kaggle within-snapshot variation:** The Kaggle data covers ~4 weeks (March 24 - April 20). Check whether postings from early vs. late in this window differ. If they don't, within-month variation is negligible.

4. **External triangulation:** Compare our findings against Revelio Labs trends (SOC 15 hiring, openings, salary) which have monthly resolution across 2021-2026. Do Revelio trends show a gradual decline or a discrete break? Does the slope accelerate around late 2025?

### 9j. Power analysis: Do we have enough data?

**Why a reviewer asks:** "With only ~3,466 Kaggle SWE postings, do you have statistical power to detect a meaningful difference?"

**Key power calculations needed:**

| Analysis | Effect size to detect | Sample sizes | Estimated power |
|---|---|---|---|
| Junior share change (chi-squared) | 5pp shift (e.g., 12% → 7%) | 3,466 vs. 14,391 | Compute |
| Description length change (Mann-Whitney) | Cohen's d = 0.2 (small) | 385 entry-level Kaggle vs. N entry-level scraped | Compute |
| Skill prevalence change (proportion test) | 5pp shift in skill mention rate | Same | Compute |
| DiD (SWE vs. control) | Interaction effect | SWE: ~17K, Control: ~1K-3K | Compute — control sample may be binding constraint |

**Kaggle entry-level SWE is the binding constraint:** Only 385 Kaggle SWE postings are labeled "entry level" (native). After imputation, this might rise to 500-700, but it's still small. Compute the minimum detectable effect size given this sample.

**Control occupation sample size:** On March 18, the non-SWE file had 874 control-pattern matches (nurses, civil/mechanical/electrical/chemical engineers). Across 14 days, we may have ~3,000-5,000 unique control postings. But the Kaggle control count depends on the raw Kaggle data (not just SWE-filtered). We need to check this.

### 9k. Robustness pre-registration: What specifications will we test?

**Why a reviewer asks:** "You could have tried 50 different specifications and reported the one that worked. How do I know you didn't?"

Per the research docs, define the specification space BEFORE looking at results:

**SWE definition variants:**
1. Narrow: Current `SWE_PATTERN` (core SWE titles only)
2. Broad: Add "data scientist", "data analyst", "product engineer"
3. Excluding adjacent: Remove "data engineer", "ML engineer" (these may have different dynamics)

**Seniority classification variants:**
1. Our imputer applied to all rows (recommended default)
2. LinkedIn native labels where available, imputed where missing
3. Description-only classifier (ignore titles)

**Dedup variants:**
1. Strict: Exact match on (title, company, location)
2. Standard: Near-dedup with similarity ≥ 0.70
3. Loose: Near-dedup with similarity ≥ 0.50

**Sample variants:**
1. Full sample
2. LinkedIn only (excludes Indeed composition effect)
3. Excluding aggregator postings
4. Top-10 metros only (more comparable geographic scope)
5. Excluding top-5 most common companies (reduces concentration bias)

**Key findings must hold across all defensible specifications.** Use the `specification_curve` package to visualize this. If a finding is fragile (holds under some specifications but not others), it is reported as suggestive, not conclusive.

### 9l. Placebo and falsification tests (pre-registration)

**Note:** This section defines the placebo tests. Stage 12b-12c executes them after Stage 11 produces results.

**Why a reviewer asks:** "Maybe your method finds 'structural change' in any two snapshots, regardless of whether anything actually changed."

**Placebos to pre-register:**

1. **Control occupation placebo:** Run the same seniority-shift analysis on control occupations (civil engineering, nursing, mechanical engineering). If they show the same "structural change" as SWE, the finding is confounded by macro trends or measurement artifacts, not AI-specific restructuring.

2. **Within-Kaggle time-split placebo:** Split Kaggle into early April (before Apr 12, n=30,101) vs. late April (Apr 12+, n=93,748). SWE counts: 743 vs. 2,723. Run the same analysis across the two halves. If we find a "shift" within a 2-week window, our method is detecting noise.

3. **Shuffled-label test:** Randomly permute the dataset labels (Kaggle vs. scraped) and re-run the analysis 10,000 times. The observed effect size should exceed 95% of permuted effect sizes.

4. **Within-scraped week-over-week:** Split 14 days into week 1 (Mar 5-11) vs. week 2 (Mar 12-18). Run the same analyses. Expect null results.

4. **Null-effect occupations:** Identify occupations with no theoretical reason to be affected by AI coding agents (e.g., registered nurses, civil engineers). Run the full analysis pipeline on these as negative controls. Expect null results.

### 9m. Bias threat summary table

Produce a consolidated table for the methodology section:

| Bias | Direction | Magnitude estimate | Mitigation | Residual risk |
|---|---|---|---|---|
| Platform selection (LinkedIn overrepresents tech/professional) | Favors SWE coverage; underrepresents control occupations | ~11pp for tech occupations (Hershbein & Kahn) | Post-stratification against OES | Low for SWE; moderate for controls |
| Algorithm selection (promoted posts) | Unknown direction | Unknown | Cannot correct; acknowledge | Moderate |
| Scraper query design (25 results × 20 cities) | Misses long-tail postings | Unknown | Query saturation check | Moderate |
| Aggregator contamination | Inflates some company counts; adds boilerplate | 9% of scraped, ~15% of Kaggle SWE | Flag and sensitivity analysis | Low after flagging |
| Temporal selection (volatility bias) | Oversamples long-lived postings | 60:1 for 60-day vs. 1-day postings (Foerderer) | Report duration distribution; consider IPW | Moderate |
| Kaggle provenance unknown | Could bias anything | Unknown | Treat as stated limitation | High (irreducible) |
| Ghost jobs | Inflates entry-level tech postings | 18-27% of all postings (CRS 2025) | Flag and sensitivity analysis | Moderate |
| Salary missingness (MNAR) | Biases salary-based analyses | 72% missing in Kaggle, 96% missing in LinkedIn scrape | Drop-and-flag (Azar et al. 2022) | High for salary outcomes; N/A for non-salary |
| Platform changes (2024 → 2026) | Could create artificial differences | Unknown | Investigate LinkedIn changelog; run LinkedIn-only comparison | Moderate (irreducible) |
| Company composition shift | Could drive apparent seniority shift | Unknown until tested | Oaxaca-Blinder decomposition; within-company comparison | Low after decomposition |
| Seasonal offset (April vs. March) | Could inflate/deflate metrics | Typically small for adjacent months | JOLTS seasonal comparison | Low |

---

## Spot-check protocol

Manual review is non-negotiable for publication quality. The following spot-checks should be conducted after the pipeline runs:

| Check | Sample size | Stratification | What to verify |
|-------|-------------|----------------|----------------|
| SWE classification | 50 per dataset | True positives + false negatives | Pattern catches real SWE, excludes non-SWE |
| Seniority imputation | 300-500 total | Source × predicted level × ambiguity | Per-class precision/recall vs. human labels |
| Boilerplate removal | 30-50 | Company size × source | Core content preserved, boilerplate stripped |
| Aggregator handling | 20-30 | By aggregator company | Real employer extracted, boilerplate stripped |
| Near-dedup | 30 pairs | True positives + false positives | Duplicates correctly identified |
| Ghost job flags | 30 flagged | By risk level | Flags are sensible |

---

## Implementation order

The stages have dependencies. Recommended implementation sequence:

```
Phase 1: Data preparation
  1. Schema unification (Stage 1)            ← update harmonize.py, join Kaggle companions
     ↓
  2. Aggregator handling (Stage 2)            ← new module
     ↓
  3. Company name standardization (6a)        ← needed for boilerplate detection
     ↓
  4. Boilerplate removal (Stage 3)            ← new module
     ↓
  5. Deduplication (Stage 4)                  ← new module, depends on clean text
     ↓
  6. Classification (Stage 5)                 ← update existing, apply uniformly
     ↓
  7. Field normalization (Stage 6b-6e)        ← new logic in harmonize.py
     ↓
  8. Temporal alignment (Stage 7)             ← computed columns
     ↓
  9. Quality flags (Stage 8)                  ← new module

Phase 2: Validation (after preprocessing produces unified.parquet)
  10. Representativeness checks (9a)           ← pull BLS/JOLTS benchmarks
  11. Cross-dataset comparability (9b)         ← distribution comparisons
  12. Missing data audit (9c)                  ← missingno diagnostics
  13. Selection bias diagnostics (9d)          ← balance package, coverage analysis
  14. Classifier validation (9e)               ← gold-standard annotation
  15. Distribution comparisons (9f)            ← KS tests, histograms
  16. Compositional analysis (9g)              ← company overlap, Oaxaca-Blinder
  17. Company concentration (9h)               ← HHI, capping, within-company analysis
  18. Temporal stability (9i)                  ← daily variance, JOLTS seasonality
  19. Power analysis (9j)                      ← sample size calculations
  20. Robustness specification space (9k)      ← define before looking at results
  21. Placebo/falsification tests (9l)         ← control occupation null tests

Phase 3: Spot-checks (manual, interspersed with phases 1-2)
  21. Spot-check protocol (as defined above)

Phase 4: Exploration & Discovery (Stage 10 — detailed below)
  22. Raw data inspection & manual review (10a)
  23. Embedding space exploration (10b)
  24. Corpus comparison — Fightin' Words (10c)
  25. Topic discovery — BERTopic + BERTrend (10d)
  26. Structured skill extraction (10e)
  27. Seniority boundary analysis (10f)
  28. Requirements section parsing (10g)
  29. Temporal drift measurement (10h)
  30. Company-level patterns (10i)
  31. Ghost job and anomaly profiling (10j)
  32. Cross-occupation comparison (10k)

Phase 5: Formal Analysis (Stage 11)
  33. RQ1 seniority shift + content convergence + scope inflation (11a)
  34. RQ2 skill prevalence shifts + Fightin' Words + co-occurrence networks (11b)
  35. RQ3 external breakpoint detection + permutation magnitude test (11c)
  36. RQ4 DiD + synthetic control + embedding drift comparison (11d)
  37. RQ6 management vs AI-orchestration keyword shift + BERTopic (11e)
  38. RQ5/RQ7 qualitative synthesis + historical comparison (11f)
  39. Specification curves + multiple testing corrections + bootstrap CIs (11g)

Phase 6: Statistical Verification (Stage 12 — runs after Stage 11 produces results)
  40. Power analysis for every primary test (12a)
  41. Within-period placebos: early/late Kaggle, week 1/2 scraped, random splits (12b)
  42. Control group verification: per-occupation + pooled + AI-exposure gradient (12c)
  43. Alternative explanation tests: platform, composition, scraping, seasonality, ghost (12d)
  44. Sensitivity matrix: seniority × SWE def × dedup × platform × company × geography (12e)
  45. Effect size calibration against external benchmarks (12f)
  46. Reproducibility protocol: seeds, model versions, data hashes (12g)
```

**Parallelizable within Phase 1:** Stages 2, 3, and 6a can be developed in parallel. Stage 4 depends on 2 and 3.

**Parallelizable within Phase 2:** Most validation checks (9a-9i) are independent of each other and can run in parallel once unified.parquet exists. Exceptions: 9j (specification space) should be defined before 9k (placebo tests), and 9g (compositional analysis) benefits from 9b (comparability) results.

**Iteration between phases:** Validation results may trigger re-runs of preprocessing. For example, if representativeness checks (9a) reveal severe geographic bias, we may revisit Stage 4 (dedup) or Stage 6b (location normalization). Build the pipeline to be re-runnable end-to-end.

---

## Stage 10: Exploration & discovery

This phase builds intuition about the data before formal hypothesis testing. Every exploration here is designed to either (a) generate visualizations and tables that go directly into the paper, (b) surface unexpected patterns that refine our research questions, or (c) validate that the preprocessed data behaves as expected before committing to expensive analyses.

**Key tool choices for this stage:**

| Tool | What it does | Why we use it here |
|---|---|---|
| [**JobBERT-v2**](https://huggingface.co/TechWolf/JobBERT-v2) | Sentence transformer fine-tuned on 5.5M job title-skill pairs (MPNet base, 1024d). Trained by TechWolf, published IEEE ACCESS 2025. | Domain-specific embeddings outperform general-purpose models for job posting similarity, clustering, and retrieval. Use this instead of `all-MiniLM-L6-v2` for all job-domain embedding tasks. |
| [**BERTopic**](https://bertopic.com/) | Neural topic modeling (embedding → UMAP → HDBSCAN → c-TF-IDF). Supports dynamic topic modeling over time. | Discovers emergent skill clusters we didn't think to look for. Handles semantic meaning, not just keywords. |
| [**BERTrend**](https://github.com/rte-france/BERTrend) | Runs BERTopic per time slice, merges across windows, classifies topics as noise / weak signal / strong signal. | Detects genuinely new topics (e.g., AI-orchestration skills) that emerge between 2024 and 2026. Standard BERTopic assumes a fixed topic set — BERTrend relaxes this. |
| [**Fightin' Words**](https://github.com/Wigder/fightin_words) | Log-odds-ratio with Dirichlet prior for pairwise corpus comparison. Produces both effect size and z-score per word. | The statistically rigorous way to answer "what words distinguish corpus A from corpus B?" Handles corpus size imbalance (our datasets are 3K vs. 14K). |
| [**Scattertext**](https://github.com/JasonKessler/scattertext) | Interactive HTML visualization of distinguishing terms between two corpora. | Produces publication-quality figures and lets us visually inspect what drives corpus differences. |
| [**KeyBERT**](https://github.com/MaartenGr/KeyBERT) | Keyword/keyphrase extraction using BERT embeddings + cosine similarity. | Extracts the most representative terms from each posting or group of postings without a predefined dictionary. |
| [**ESCO Skill Extractor**](https://github.com/nestauk/ojd_daps_skills) | Maps free-text skill mentions to the ESCO/Lightcast Open Skills taxonomy. | Structured skill extraction grounded in an international standard. Gives us a controlled vocabulary instead of raw keywords. |
| [**Nesta OJD Skills Library**](https://github.com/nestauk/ojd_daps_skills) | End-to-end pipeline: extract skill phrases from job ads, map to ESCO or Lightcast taxonomy. | Built specifically for job ad analysis. Handles the full pipeline from raw text to structured skill tags. |

### 10a. Raw data inspection & manual review

**Before any automated analysis, look at the data with human eyes.** This is non-negotiable — automated methods can produce plausible-looking results from garbage data.

**Actions:**

1. **Random sample review (50 per segment):**
   - 50 Kaggle SWE postings (stratified by seniority: 15 entry, 15 mid-senior, 10 associate, 10 other)
   - 50 scraped SWE postings (same stratification)
   - 50 non-SWE postings (mix of adjacent and control)
   - For each: read the full description. Note: Does the seniority label match the content? Are the skills realistic? Is the description a real job or boilerplate/spam? Does the company name match what's described?

2. **Extreme value inspection:**
   - Shortest 20 descriptions (what's in them? Template stubs?)
   - Longest 20 descriptions (are they multiple jobs concatenated? Boilerplate-heavy?)
   - Postings with the most skills mentioned (skill inflation or genuinely complex roles?)
   - Entry-level postings requiring 5+ years (ghost jobs or misclassified seniority?)

3. **Side-by-side comparison (10 matched pairs):**
   - Find 10 companies appearing in both Kaggle and scraped data. For each, pull one SWE posting from each period. Compare: How has the description changed? Same structure? Different requirements? Longer? More AI mentions?
   - This is qualitative evidence that anchors the quantitative findings. Quote specific examples in the paper.

4. **Output format:** A markdown file with annotated examples. Tag each with observations: `[boilerplate]`, `[ghost-job?]`, `[skill-inflation]`, `[good-example]`, `[misclassified]`.

### 10b. Embedding space exploration

**Goal:** Visualize how job postings cluster in semantic space and whether clusters align with our seniority/occupation categories.

**Implementation:**

1. **Embed all SWE postings** using [JobBERT-v2](https://huggingface.co/TechWolf/JobBERT-v2) (5.5M job-domain training pairs, 1024d). This model was specifically trained on job title + skills pairs — it captures occupational semantics that general-purpose models miss. For example, it knows that "Software Engineer" and "Software Development Engineer" are near-synonyms (cosine sim ~0.87), while general SBERT models may score them lower.

   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer("TechWolf/JobBERT-v2")

   # Embed title + first 200 words of description_core
   texts = (df['title'] + ' ' + df['description_core'].str[:800]).tolist()
   embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
   ```

2. **UMAP projection to 2D:**
   ```python
   import umap
   reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine')
   coords = reducer.fit_transform(embeddings)
   ```

3. **Visualization layers** (produce multiple views of the same embedding space):
   - Color by **seniority** (junior/mid/senior) — do seniority levels form distinct clusters, or do they overlap? Overlap = the seniority boundary is fuzzy in practice.
   - Color by **period** (2024 vs. 2026) — do the two periods occupy different regions, or are they interleaved? Separation = real distributional shift. Interleaving = similar posting content across periods.
   - Color by **company** (top-10 companies highlighted) — do individual companies form tight clusters (standardized templates) or spread across the space?
   - Color by **topic** (from BERTopic, step 10d) — visual confirmation that topic clusters are semantically coherent.

4. **Quantitative embedding analysis:**
   - **Junior-senior centroid distance:** Compute the cosine distance between the average junior embedding and the average senior embedding, separately for each period. If this distance shrank between 2024 and 2026, junior roles are converging toward senior roles in content (RQ1 evidence).
   - **Within-seniority variance:** Compute the average pairwise cosine distance within each seniority level. If junior postings have higher variance in 2026 than 2024, the category is becoming more heterogeneous (redefinition, not disappearance).
   - **Cross-period drift per seniority:** For each seniority level, compute the cosine distance between the 2024 centroid and the 2026 centroid. If junior roles drifted more than senior roles, that's directional evidence for RQ1.

### 10c. Corpus comparison — Fightin' Words & Scattertext

**Goal:** Identify the specific words and phrases that statistically distinguish different groups (junior vs. senior, 2024 vs. 2026, SWE vs. control).

**Comparisons to run (6 total):**

| Comparison | What it tests | RQ |
|---|---|---|
| Junior 2024 vs. Junior 2026 | How junior role language changed | RQ1 |
| Senior 2024 vs. Senior 2026 | How senior role language changed | RQ6 |
| Junior 2024 vs. Senior 2024 | Baseline junior-senior gap | RQ1, RQ2 |
| Junior 2026 vs. Senior 2026 | Current junior-senior gap | RQ1, RQ2 |
| Junior 2026 vs. Senior 2024 | Are 2026 juniors linguistically similar to 2024 seniors? | RQ1 (redefinition hypothesis) |
| SWE 2026 vs. Control 2026 | SWE-specific vs. general labor market language | RQ4 |

**Implementation:**

1. **Fightin' Words** (quantitative):
   ```python
   from fightin_words import FWExtractor
   fw = FWExtractor(ngram_range=(1, 3), min_df=5)
   results = fw.fit_transform(corpus_a, corpus_b)
   # Returns (word, log-odds-ratio, z-score) for every n-gram
   # Sort by |z-score| to find the most distinguishing terms
   ```
   Use `description_core` (boilerplate-stripped). Run on unigrams, bigrams, and trigrams separately. Filter to |z-score| > 3.0 for statistical significance after Bonferroni correction.

2. **Scattertext** (visual):
   ```python
   import scattertext as st
   corpus = st.CorpusFromPandas(df, category_col='period',
                                 text_col='description_core',
                                 nlp=nlp).build()
   html = st.produce_scattertext_explorer(corpus,
       category='2026-03', category_name='March 2026',
       not_category_name='April 2024')
   open('scattertext_2024_vs_2026.html', 'w').write(html)
   ```
   Produces an interactive HTML plot. Each dot is a word. Position = relative frequency in each corpus. Hover for context. This is both an analysis tool and a publication figure.

3. **What to look for:**
   - **AI/LLM terms emerging:** Do "LLM", "AI agent", "prompt engineering", "RAG", "vector database" appear in 2026 but not 2024?
   - **Management terms declining in senior roles:** Do "mentorship", "coaching", "team leadership", "hiring" shift downward in 2026 senior postings? (RQ6)
   - **Senior terms appearing in junior roles:** Do "system design", "architecture", "end-to-end", "cross-functional" appear more in 2026 junior postings than 2024? (RQ1 skill migration)
   - **Years-of-experience inflation:** Do "3+ years", "5+ years" appear more frequently in 2026 entry-level postings? (ghost job / scope inflation signal)

### 10d. Topic discovery — BERTopic & BERTrend

**Goal:** Discover latent topic structure and track which topics are emerging, stable, or declining between periods. This catches skill categories and role archetypes that our predefined keyword lists miss.

**Implementation:**

1. **Base BERTopic on full SWE corpus (both periods combined):**

   ```python
   from bertopic import BERTopic
   from sentence_transformers import SentenceTransformer

   embedding_model = SentenceTransformer("TechWolf/JobBERT-v2")
   topic_model = BERTopic(
       embedding_model=embedding_model,
       min_topic_size=30,          # ~0.2% of SWE corpus
       nr_topics="auto",
       verbose=True
   )
   topics, probs = topic_model.fit_transform(descriptions)
   ```

   **Critical:** Use `description_core` (boilerplate-stripped). EEO statements and benefits sections are the #1 source of spurious topics. Our research docs flag this: "Remove EEO statements, benefits sections, company boilerplate. This is the single highest-impact step."

2. **Dynamic topic modeling (topics over time):**
   ```python
   topics_over_time = topic_model.topics_over_time(
       descriptions, timestamps=dates, nr_bins=6  # e.g., monthly or bimonthly
   )
   topic_model.visualize_topics_over_time(topics_over_time)
   ```
   This shows which topics are growing and shrinking. With only 2 time points (April 2024, March 2026), the resolution is limited — but the signal we're looking for (new AI topics, declining management topics) should be detectable even with 2 bins.

3. **BERTrend for emerging signal detection:**

   BERTrend goes beyond standard BERTopic by classifying each topic as **noise**, **weak signal**, or **strong signal** based on popularity trends. This is designed for exactly our use case: detecting whether AI-orchestration skills are a weak signal in 2024 that became a strong signal by 2026.

   ```python
   from bertrend import BERTrend
   # Process data in time-sliced batches
   bertrend = BERTrend(embedding_model="TechWolf/JobBERT-v2")
   bertrend.fit(docs_by_period)  # dict of {period: [docs]}
   signals = bertrend.get_signals()  # weak, strong, noise classification
   ```

4. **Topic-level analysis for each RQ:**

   | RQ | What to look for in topics | Expected signal |
   |---|---|---|
   | RQ1 | Topics that appear in senior postings in 2024 but migrate to junior postings in 2026 | Topics like "system design", "architecture decisions" |
   | RQ2 | The temporal order in which skill-topics appear in junior postings | Sequence: cloud → CI/CD → system design → AI tools |
   | RQ6 | Topics in senior postings: management-heavy in 2024, AI-heavy in 2026 | "Team leadership, hiring" declining; "AI integration, agent orchestration" emerging |
   | RQ4 | Whether the same topic shifts appear in control occupations | They should NOT (if they do, it's macro confounding) |

5. **Guided BERTopic** (semi-supervised):

   BERTopic supports [guided topic modeling](https://maartengr.github.io/BERTopic/getting_started/guided/guided.html) where you provide seed topics. Use our RQ-derived skill categories as seeds:
   ```python
   seed_topic_list = [
       ["system design", "architecture", "distributed systems", "scalability"],
       ["AI", "LLM", "prompt engineering", "RAG", "agent", "copilot"],
       ["mentorship", "coaching", "team lead", "hiring", "performance review"],
       ["CI/CD", "deployment", "infrastructure", "DevOps", "kubernetes"],
       ["testing", "QA", "quality assurance", "test automation"],
   ]
   topic_model = BERTopic(seed_topic_list=seed_topic_list, ...)
   ```
   This nudges the model to discover topics aligned with our hypotheses while still allowing it to find unexpected topics.

### 10e. Structured skill extraction

**Goal:** Extract structured, taxonomy-mapped skills from free-text descriptions. This goes far beyond keyword matching — it handles synonyms, multi-word phrases, and maps to a standard vocabulary.

**Why this matters:** Our current skill analysis uses a hand-coded list of 16 keywords. This misses skills we didn't think of, conflates different uses of the same word, and can't handle synonyms ("k8s" = "Kubernetes" = "container orchestration").

**Implementation — two complementary approaches:**

1. **Nesta OJD Skills Library** (taxonomy-mapped):
   ```python
   from ojd_daps_skills.pipeline.skill_ner import SkillNER
   from ojd_daps_skills.pipeline.skill_match import SkillMapper

   ner = SkillNER()
   mapper = SkillMapper(taxonomy="esco")  # or "lightcast"

   skills_raw = ner.extract(description_core)
   skills_mapped = mapper.map(skills_raw)
   # Returns: [{"skill": "Kubernetes", "esco_code": "S4.8.1", "confidence": 0.92}, ...]
   ```

   Maps to ESCO (13,890 skills) or Lightcast Open Skills taxonomy. Gives us a controlled vocabulary for cross-period comparison — "Docker" and "containerization" both map to the same ESCO skill code.

2. **KeyBERT** (unsupervised, catches what taxonomies miss):
   ```python
   from keybert import KeyBERT
   kw_model = KeyBERT(model="TechWolf/JobBERT-v2")

   keywords = kw_model.extract_keywords(
       description_core,
       keyphrase_ngram_range=(1, 3),
       stop_words='english',
       top_n=15
   )
   # Returns: [("distributed systems", 0.72), ("react native", 0.68), ...]
   ```

   Using JobBERT-v2 as the backbone means keyphrases are scored by relevance to the job domain, not just general text statistics. This catches emerging terms (e.g., "agentic workflow", "vibe coding") that aren't in any taxonomy yet.

3. **Skill prevalence analysis (replaces hand-coded keyword lists):**
   - For each ESCO-mapped skill, compute prevalence by seniority × period
   - Identify skills with the largest prevalence change between periods
   - Identify skills that migrated from senior-only to junior+senior
   - Produce the task migration map (RQ2) grounded in a standard taxonomy rather than researcher-selected keywords

### 10f. Seniority boundary analysis

**Goal:** Understand where the junior/senior boundary actually lies in the data, and whether it moved between periods.

**This addresses the core of RQ1:** Are junior roles being redefined? If the content of 2026 "junior" postings looks like 2024 "mid-senior" postings, that's direct evidence of redefinition.

**Implementation:**

1. **Embedding-based seniority boundary:**
   - Train a simple logistic regression on JobBERT-v2 embeddings to predict seniority (junior vs. senior) using 2024 data only
   - Apply this classifier to 2026 data. If 2026 "junior" postings are classified as "senior" by the 2024 model at a higher rate than 2024 "junior" postings, the boundary has shifted. This is the "content convergence" metric from the validation plan.
   - Report: % of 2026 junior postings that the 2024-trained model classifies as senior (the "redefinition rate")

2. **Decision boundary visualization:**
   - In the UMAP space from 10b, draw the decision boundary of the 2024-trained seniority classifier
   - Overlay 2026 junior postings. How many fall on the "senior" side of the 2024 boundary?
   - This is a powerful visual for the paper: it literally shows junior postings crossing into senior territory.

3. **Feature importance for the boundary:**
   - What words/skills most strongly predict "senior" vs. "junior" in the 2024 model?
   - Which of those features are now present in 2026 junior postings?
   - Use SHAP values or logistic regression coefficients for interpretability.

### 10g. Requirements section parsing

**Goal:** Extract structured data from the requirements/qualifications section of job descriptions, rather than relying on the full description text.

**Why this is separate from full-text analysis:** The requirements section is the most decision-relevant part of a job posting for applicants and for our research. "Scope inflation" (RQ1) should be measured by what's required, not what's in the "About Us" section.

**Implementation:**

1. **Section extraction** (from Stage 3 boilerplate removal):
   - Parse `description_core` into sections: responsibilities, requirements/qualifications, nice-to-haves
   - Analyze requirements sections separately from full descriptions

2. **Structured requirement extraction:**
   - **Years of experience:** Extract all "X+ years" patterns. Compute min, max, and median years required per seniority level × period. Test for inflation.
   - **Education requirements:** Extract degree mentions (BS, MS, PhD). Compute degree distribution per seniority × period.
   - **Technology requirements:** Count distinct technologies mentioned in requirements (not just description). Are junior roles in 2026 requiring more technologies than in 2024?
   - **Soft skill requirements:** Extract management/leadership/communication mentions from requirements specifically. Are junior roles now requiring "cross-functional collaboration" and "stakeholder management"?

3. **Requirement count as scope metric:**
   - Count the number of bullet points or distinct requirements per posting
   - Compare this count across seniority × period
   - This is a more targeted "scope inflation" metric than description word count, which is confounded by boilerplate

### 10h. Temporal drift measurement

**Goal:** Quantify how much the posting landscape changed between April 2024 and March 2026, and characterize the direction of change.

**Implementation:**

1. **Corpus-level embedding drift:**
   ```python
   # Compute centroid embeddings for each period
   centroid_2024 = embeddings_2024.mean(axis=0)
   centroid_2026 = embeddings_2026.mean(axis=0)
   drift = 1 - cosine_similarity([centroid_2024], [centroid_2026])[0][0]
   ```
   Do this overall, and per seniority level. Report as a table.

2. **Vocabulary drift (JSD):**
   - Build unigram frequency distributions for each period
   - Compute Jensen-Shannon divergence between them
   - Do this for the full SWE corpus, and separately for junior-only and senior-only
   - JSD for junior postings vs. JSD for senior postings: if junior postings changed more than senior, that's RQ1 evidence

3. **Nearest-neighbor stability:**
   - For each posting in the overlap set (companies appearing in both periods), find its k=10 nearest neighbors
   - What fraction of neighbors are from the same period vs. the other period?
   - If 2026 junior postings' nearest neighbors are mostly 2024 senior postings, that's strong evidence of content convergence

4. **Keyword emergence/disappearance:**
   - Terms appearing in 2026 at >1% prevalence but absent from 2024 (or <0.1%): these are emerging requirements
   - Terms appearing in 2024 at >1% prevalence but absent from 2026: these are declining requirements
   - Do this separately for junior and senior postings

### 10i. Company-level patterns

**Goal:** Understand whether observed shifts are driven by within-company changes (same companies posting differently) or between-company changes (different companies dominating).

**Implementation:**

1. **Company overlap set analysis:**
   - Identify companies appearing in both Kaggle and scraped datasets
   - For overlapping companies: compare their seniority distributions, skill mentions, description length across periods
   - This is the within-company change signal — not confounded by composition

2. **Company archetypes (via clustering):**
   - Cluster companies by their posting profiles (average embedding, seniority mix, skill distribution)
   - Are there distinct "types" of SWE employers? (e.g., FAANG-style, startup-style, consulting-style, government)
   - Did the relative share of these archetypes change between periods?

3. **Firm-size effects:**
   - Split by company size bands (1-50, 50-500, 500-5000, 5000+)
   - Run all key metrics (seniority distribution, skill prevalence, description length) within each size band
   - Does scope inflation show equally in startups and large enterprises, or is it concentrated?

### 10j. Ghost job and anomaly profiling

**Goal:** Characterize the ghost job phenomenon and understand whether it biases our findings.

**Implementation:**

1. **Ghost job feature analysis:**
   - For postings flagged as ghost-risk in Stage 8, profile them: Which companies? Which seniority levels? Which geographies? How do their descriptions differ from non-ghost postings?
   - Do ghost jobs have systematically different skill requirements (more inflated)?

2. **Anomaly detection:**
   - Use isolation forest or DBSCAN on the embedding space to identify outlier postings
   - Manually review the outliers: spam, duplicate templates, non-English, non-US, misclassified occupation?
   - Report the anomaly rate and exclude from analysis with sensitivity check

### 10k. Cross-occupation comparison

**Goal:** Establish that observed changes are SWE-specific, not part of a broader labor market trend (RQ4 groundwork).

**Implementation:**

1. **Run the same exploration on control occupations:**
   - Embedding space, Fightin' Words, topic modeling, skill extraction — all on civil engineering, nursing, mechanical engineering postings
   - The key question: do control occupations show the same patterns (AI skill emergence, scope inflation, seniority compression)?
   - If they do, our SWE findings are confounded. If they don't, we have DiD evidence.

2. **SWE-adjacent occupation analysis:**
   - Data scientist, product manager, UX designer — these are AI-exposed but not SWE
   - Do they show similar restructuring patterns? This tests whether the effect is specific to coding or broader to tech

3. **Cross-occupation embedding distance:**
   - How far apart are SWE, SWE-adjacent, and control postings in embedding space?
   - Is the distance between SWE and SWE-adjacent shrinking (role convergence)?

### Exploration outputs

The exploration phase produces:

| Output | Type | Used in |
|---|---|---|
| Annotated raw sample | Markdown file | Qualitative examples for paper |
| UMAP embedding plots (4 color schemes) | PNG + interactive HTML | Paper figures |
| Scattertext comparisons (6 pairs) | Interactive HTML | Paper figures, appendix |
| Fightin' Words tables (6 comparisons) | CSV + sorted tables | Paper tables |
| BERTopic model + topic list | Saved model + CSV | RQ2 analysis input |
| BERTrend signal report | CSV (topic × signal strength) | RQ2, RQ3 analysis input |
| ESCO-mapped skill prevalence table | Parquet | RQ2 task migration map |
| KeyBERT emerging terms list | CSV | Appendix |
| Seniority boundary classifier | Saved model | RQ1 redefinition rate |
| Requirements section structured data | Parquet (years, degree, tech count) | RQ1, RQ2 |
| Company-level metric table | Parquet | Compositional analysis |
| Ghost job profile | Markdown + CSV | Methodology section |

---

## Stage 11: Formal hypothesis testing

This is the core analysis that produces the paper's findings. Each RQ gets a primary test (the simplest credible method that answers the question), a stronger test (ML/NLP-powered, higher power or richer signal), and robustness checks. Every test is paired with the specific alternative finding that would falsify our hypothesis.

**Design constraint:** We have two cross-sections (April 2024, March 2026), not a panel. We cannot track individual postings or companies over time. All analyses are cross-sectional comparisons of distributions, proportions, and text features across periods. We frame findings as "consistent with" structural change, not as causal proof.

---

### 11a. RQ1: Are junior SWE roles disappearing or being redefined?

**What we need to show:** Either (a) the share of junior SWE postings declined between 2024 and 2026, or (b) the content of junior postings in 2026 resembles senior postings from 2024, or (c) both.

**Null hypothesis:** The junior share and the content of junior postings are statistically indistinguishable across the two periods, after controlling for composition.

#### Test 1: Seniority distribution shift (primary)

The simplest credible test. Compare the proportion of entry-level / associate / mid-senior SWE postings between April 2024 and March 2026.

```
H₀: P(junior | SWE, 2024) = P(junior | SWE, 2026)
H₁: P(junior | SWE, 2024) ≠ P(junior | SWE, 2026)
```

**Method:** Chi-squared test of homogeneity on the 3×2 contingency table (3 seniority levels × 2 periods). Report Cramér's V for effect size.

**Implementation:**
```python
from scipy.stats import chi2_contingency
contingency = pd.crosstab(df['period'], df['seniority_3level'])
chi2, p, dof, expected = chi2_contingency(contingency)
cramers_v = np.sqrt(chi2 / (len(df) * (min(contingency.shape) - 1)))
```

**Sufficient evidence:** p < 0.05 with Cramér's V > 0.05 (small-but-meaningful effect). Report the actual shift in junior share (percentage points) with bootstrap 95% CI.

**What would falsify:** If the junior share is unchanged or increased, the "disappearing" narrative is wrong. If the distribution shifted but only because a different mix of companies was captured, it's a composition artifact (checked in 9g/9h).

#### Test 2: Content convergence — the redefinition test (stronger)

This separates title inflation (cosmetic relabeling) from genuine content change (what the job actually requires). A 2026 "Junior SWE" posting that reads like a 2024 "Mid-Senior SWE" posting is a redefined role, not a disappeared one.

**Method:** Train a logistic regression seniority classifier on 2024 JobBERT-v2 embeddings. Apply it to 2026 postings. Measure the "redefinition rate": the fraction of 2026 entry-level postings that the 2024-trained model predicts as mid-senior.

```python
from sklearn.linear_model import LogisticRegression

# Train on 2024 data (where seniority labels are known)
clf = LogisticRegression(max_iter=1000)
clf.fit(embeddings_2024, seniority_2024)  # labels: junior / mid / senior

# Predict on 2026 entry-level postings
junior_2026 = df_2026[df_2026['seniority'] == 'entry level']
preds = clf.predict(embeddings_junior_2026)
redefinition_rate = (preds != 'junior').mean()
```

**Sufficient evidence:** If the redefinition rate for 2026 junior postings is significantly higher than the "misclassification" rate on 2024 junior postings (the baseline error rate), the content has shifted. Compare with a permutation test: shuffle the period labels 1000 times and recompute. The observed redefinition rate should exceed 95% of permuted rates.

**Alternative method (model-free):** Compute the cosine similarity between each 2026 junior posting's embedding and the 2024 junior centroid vs. the 2024 senior centroid. If more 2026 junior postings are closer to the senior centroid than to the junior centroid (compared to 2024 junior postings), that's convergence. Test with a two-sample t-test on the similarity ratio.

#### Test 3: Scope inflation metrics

**Methods (run all, report as a table):**

| Metric | Test | Interpretation |
|---|---|---|
| Description length (requirements section only) | Mann-Whitney U | Longer requirements = more demanded of juniors |
| Distinct skill count per posting | Mann-Whitney U | More skills = broader scope |
| Years of experience required (median) | Mann-Whitney U | Higher YoE = inflated requirements |
| Skill Breadth Index (ESCO-mapped distinct skills) | Mann-Whitney U | Taxonomy-grounded scope measure |
| Senior keyword infiltration rate | Proportion test (z-test) | "system design", "architecture" appearing in junior postings |

Each test is run on junior postings only, comparing 2024 vs. 2026. Use Benjamini-Hochberg FDR correction across the battery. Report effect sizes (Cohen's d or rank-biserial correlation) alongside p-values.

#### Test 4: Controlled comparison (OLS regression)

Control for composition to isolate within-composition change:

```
SkillBreadth_i = β₀ + β₁(Period2026) + β₂(CompanySize) + β₃(Industry)
                + β₄(Metro) + β₅(IsRemote) + ε_i
```

Run on junior SWE postings only. β₁ is the period effect after controlling for composition. Use HC3 robust standard errors. Cluster by company if enough firms appear in both periods.

**Sufficient evidence for RQ1 overall:** At least 2 of the 4 tests pointing in the same direction (seniority shift + content convergence, or seniority shift + scope inflation, etc.). A single test alone is not conclusive given our data limitations.

---

### 11b. RQ2: Which competencies migrated from senior to junior postings, and in what order?

**What we need to show:** Specific skills that were predominantly senior-associated in 2024 appear at significantly higher rates in junior postings by 2026. The chronological ordering of this migration is secondary (limited by having only 2 time points) but we can establish which skills have migrated furthest.

**Null hypothesis:** The skill profile of junior postings is unchanged between periods. No individual skill shows a statistically significant increase in junior prevalence.

#### Test 1: Skill prevalence shift (primary)

For each ESCO-mapped skill (from 10e), compute its prevalence in junior postings in each period. Test for significant changes.

```
H₀: P(skill_k | junior, 2024) = P(skill_k | junior, 2026)   for each skill k
H₁: P(skill_k | junior, 2024) ≠ P(skill_k | junior, 2026)   for at least some k
```

**Method:** Two-proportion z-test for each skill. Apply Benjamini-Hochberg FDR at q = 0.05. Report adjusted p-values and the absolute prevalence change (Δ percentage points).

**Visualization:** Skill migration heatmap — rows = skills (sorted by Δ), columns = (junior 2024, senior 2024, junior 2026, senior 2026). Color = prevalence. The visual pattern of skills "sliding" from senior-only to junior+senior is the core RQ2 figure.

#### Test 2: Fightin' Words for junior vocabulary shift

Run Fightin' Words (log-odds-ratio with Dirichlet prior) on:
- Junior 2024 vs. Junior 2026 (what changed in junior postings?)
- Junior 2026 vs. Senior 2024 (do 2026 juniors sound like 2024 seniors?)

Words with high positive log-odds in the second comparison AND high positive log-odds in the first comparison are "migrated" terms: they're newly associated with junior roles and they sound like what senior roles used to require.

**Implementation:**
```python
from fightin_words import FWExtractor
fw = FWExtractor(ngram_range=(1, 3), min_df=5)
results = fw.fit_transform(junior_2026_texts, senior_2024_texts)
# Sort by z-score; positive = overrepresented in junior 2026 vs senior 2024
```

#### Test 3: Skill co-occurrence network shift

Build a skill co-occurrence graph for junior postings in each period. Nodes = ESCO skills, edges = co-occurrence within the same posting, edge weights = PMI (pointwise mutual information).

**What to measure:**
- New edges in 2026 that didn't exist in 2024 (new skill combinations emerging in junior roles)
- Skills that gained centrality (degree, betweenness) — these are skills that became "hub" requirements
- Community structure changes — did junior skill clusters reorganize?

**Implementation:**
```python
import networkx as nx
from sklearn.metrics import mutual_info_score

G_2024 = build_cooccurrence_graph(junior_skills_2024, min_pmi=1.0)
G_2026 = build_cooccurrence_graph(junior_skills_2026, min_pmi=1.0)

new_edges = set(G_2026.edges()) - set(G_2024.edges())
centrality_change = {skill: nx.betweenness_centrality(G_2026).get(skill, 0)
                            - nx.betweenness_centrality(G_2024).get(skill, 0)
                     for skill in all_skills}
```

#### Test 4: Embedding trajectory analysis

For each skill term (e.g., "system design"), compute its average context embedding when it appears in junior postings vs. senior postings, in each period.

**Migration metric:** For skill k, compute:
```
convergence_k = cos_sim(junior_context_2026_k, senior_context_2024_k)
              - cos_sim(junior_context_2024_k, senior_context_2024_k)
```

If convergence_k > 0, the way junior postings talk about skill k in 2026 is more similar to how senior postings talked about it in 2024. This captures whether skills are migrating in meaning (ownership vs. exposure), not just frequency.

**Sufficient evidence for RQ2:** A set of 5-15 skills showing statistically significant prevalence increases in junior postings (FDR-corrected), with directional confirmation from Fightin' Words and co-occurrence network analysis. The task migration map (which skills, ranked by magnitude of shift) is the deliverable.

---

### 11c. RQ3: Did the junior SWE market experience a structural break?

**What we need to show:** The cross-sectional differences between 2024 and 2026 are larger than what gradual trend extrapolation would predict. Ideally: evidence of a discrete level shift rather than smooth drift.

**Constraint:** With only 2 time points, we cannot run time-series methods (Bai-Perron, ITS) on our own data. These require a monthly time series. Our options:

1. Use external time-series data (Revelio Labs, JOLTS) for breakpoint detection, and show that our cross-sectional findings are consistent with the identified break.
2. Acquire the 1.3M Kaggle dataset — if it spans multiple months of 2024, we get within-2024 variation.
3. Frame our contribution as documenting the magnitude of change between 2024 and 2026, with the break-detection as supplementary evidence from external data.

#### Test 1: External time series analysis (Revelio + JOLTS)

We already have Revelio Labs data (SOC 15 hiring, openings, salary, employment, 2021-2026) and JOLTS data. Run structural break detection on these external series.

**Method:** Bai-Perron endogenous breakpoint detection on the Revelio SOC-15 monthly job openings series.

```python
import ruptures as rpt

# Revelio monthly SWE job openings, 2021-2026
signal = revelio_soc15_openings.values
algo = rpt.Pelt(model="rbf", min_size=3).fit(signal)
breakpoints = algo.predict(pen=10)
# Returns indices of detected break dates
```

**Workflow:**
1. UDmax test: any breaks at all in the Revelio SWE openings series?
2. If yes, estimate break date(s) with confidence intervals
3. Run the same on control occupation series (nursing, civil engineering). If they show the same break, it's macro, not SWE-specific.

**Confirmatory:** Chow test at hypothesized break date (December 2025, when production coding agents deployed):
```python
from statsmodels.stats.diagnostic import breaks_cusumolsresid
# Or manual F-test splitting series at Dec 2025
```

#### Test 2: Magnitude-of-change test

Even without a time series, we can test whether the observed 2024-2026 difference is larger than what random variation would produce.

**Method:** Permutation test. Pool all SWE postings from both periods. Randomly assign them to "2024" and "2026" groups (maintaining the original group sizes). Recompute the seniority shift metric (junior share change) 10,000 times. The observed shift should exceed 95% of permuted shifts.

```python
observed_shift = junior_share_2026 - junior_share_2024
permuted_shifts = []
for _ in range(10000):
    shuffled = np.random.permutation(all_seniority_labels)
    perm_2024 = shuffled[:n_2024]
    perm_2026 = shuffled[n_2024:]
    permuted_shifts.append(perm_2026_junior_share - perm_2024_junior_share)
p_value = (np.abs(permuted_shifts) >= np.abs(observed_shift)).mean()
```

This doesn't prove a break per se, but it proves the difference is not noise.

#### Test 3: Multivariate change detection (BOCPD on external data)

Run Bayesian Online Change Point Detection simultaneously on multiple Revelio/JOLTS series: SWE openings + hiring rate + salary trend + average skill count. Multivariate detection has higher power because it pools coordinated signals.

```python
from bayesian_changepoint_detection import online_changepoint_detection
from bayesian_changepoint_detection.hazard_functions import constant_hazard
from bayesian_changepoint_detection.likelihoods import MultivariateGaussian

# Stack multiple time series into a matrix
features = np.column_stack([swe_openings, hiring_rate, avg_skill_count])
R, maxes = online_changepoint_detection(features, constant_hazard(250),
                                         MultivariateGaussian())
```

**Sufficient evidence for RQ3:** A break detected in external time series (Revelio/JOLTS) near the hypothesized date (late 2025), combined with a permutation test showing our cross-sectional difference exceeds random variation. If no external break is detected, we frame RQ3 as inconclusive and focus on the magnitude of cross-sectional change.

---

### 11d. RQ4: Is this shift specific to SWE or part of a broader labor market trend?

**What we need to show:** The patterns observed in SWE postings (seniority shift, skill migration, scope inflation) do NOT appear in control occupations, or appear at significantly smaller magnitude.

**Null hypothesis:** The change in junior share (or skill breadth, or description length) is the same for SWE and control occupations.

#### Test 1: Difference-in-Differences (primary)

The workhorse causal inference design. Compare the change in junior share between SWE (treatment) and non-AI-exposed occupations (control) across the two periods.

```
Y_i = α + β₁(SWE) + β₂(Post2026) + β₃(SWE × Post2026) + γX + ε
```

β₃ is the treatment effect: the SWE-specific change beyond what control occupations experienced.

**Implementation:**
```python
import statsmodels.formula.api as smf

model = smf.ols(
    'junior_indicator ~ is_swe * is_post2026 + company_size + is_remote + C(industry)',
    data=df_swe_and_control
).fit(cov_type='HC3')
# β₃ = model.params['is_swe:is_post2026']
```

**Run on multiple outcomes:** junior share, skill breadth index, description length, AI-keyword prevalence. Each DiD produces a separate β₃ estimate. Use FDR correction across outcomes.

**Critical assumption:** Parallel pre-trends. With only 2 time points, we cannot directly test this. Mitigation:
- Use external data (Revelio) to show SWE and control occupations were on parallel trajectories before 2024
- Report sensitivity of β₃ to different control occupation sets

**Control occupation selection:** Use Felten et al. (2023) AI exposure scores. Select occupations in the bottom quartile of AI exposure: civil engineering (SOC 17-2051), mechanical engineering (17-2141), registered nursing (29-1141), accounting (13-2011). These are our Tier 3 scraper queries. Verify adequate sample sizes in both periods before running.

#### Test 2: Synthetic control (stronger)

Instead of hand-picking controls, construct a weighted combination of all non-SWE occupations that best matches pre-treatment SWE trends (using Revelio data where we have monthly resolution).

```python
from SyntheticControlMethods import Synth

# Y = junior share, monthly, by occupation
synth = Synth(df, outcome='junior_share', unit='occupation',
              time='month', treatment='SWE', treatment_period='2025-12')
synth.fit()
synth.plot(['original', 'pointwise', 'cumulative'])
```

The gap between the actual SWE junior share and the synthetic control's predicted junior share IS the treatment effect.

**Robustness:** Run placebo synthetic controls for each donor occupation (in-space placebo). If many placebos show gaps as large as SWE, the finding is not significant. Compute a pseudo p-value: (rank of SWE gap among all gaps) / (number of donor occupations).

**Data requirement:** Monthly time series per occupation. This requires the Revelio data or the 1.3M Kaggle dataset (if it has monthly resolution). Cannot be done with our 2-snapshot data alone.

#### Test 3: Cross-occupation embedding analysis

Using the full embedding space from 10b, measure whether SWE postings drifted more than control postings between periods.

```python
# Centroid drift per occupation
for occupation in ['SWE', 'civil_eng', 'nursing', 'mech_eng']:
    centroid_2024 = embeddings[occ == occupation & period == 2024].mean(axis=0)
    centroid_2026 = embeddings[occ == occupation & period == 2026].mean(axis=0)
    drift = 1 - cosine_similarity([centroid_2024], [centroid_2026])[0][0]
    print(f"{occupation}: drift = {drift:.4f}")
```

If SWE drift >> control drift, the change is SWE-specific. Test with a bootstrap: resample postings within each occupation-period, recompute drift, build confidence intervals. If the SWE drift CI doesn't overlap with control drift CIs, the difference is significant.

**Sufficient evidence for RQ4:** A positive, significant β₃ in the DiD regression, confirmed by directional consistency in the synthetic control (if feasible) and the embedding drift comparison. If β₃ is null, the shift is macro (affects all occupations equally) and we report that finding instead.

---

### 11e. RQ6: Are senior SWE roles shedding management and gaining AI-orchestration?

**What we need to show:** Senior SWE postings in 2026 mention management/people-leadership skills less frequently and AI/orchestration skills more frequently than in 2024.

**Null hypothesis:** The management keyword frequency and AI-orchestration keyword frequency in senior SWE postings are unchanged between periods.

#### Test 1: Keyword category prevalence shift (primary)

Define two keyword categories grounded in the research design:

**Management category:** mentorship, coaching, hiring, team lead, team leadership, performance review, people management, direct reports, staff development, career development, 1:1, one-on-one

**AI-orchestration category:** AI agent, LLM, large language model, prompt engineering, RAG, retrieval augmented, model evaluation, AI integration, copilot, AI orchestration, AI-assisted, agent framework, vector database, agentic

For each category, compute prevalence (fraction of senior SWE postings mentioning at least one keyword) in each period. Test with two-proportion z-test.

**Archetype Shift Index:** Compute the ratio (AI-orchestration prevalence) / (management prevalence) for each period. This single number captures the directional shift. Test whether it increased with a bootstrap CI.

#### Test 2: Fightin' Words on senior postings

Run Fightin' Words comparing senior 2024 vs. senior 2026 descriptions. The top terms distinguish the two periods without requiring predefined keyword lists. This is a check on our keyword categories: if our chosen keywords rank highly in the Fightin' Words output, our categories are well-constructed. If other terms rank higher, we may be missing important dimensions of change.

#### Test 3: BERTopic on senior postings only

Run BERTopic separately on senior SWE postings (both periods combined). Examine whether:
- Management-themed topics are more prevalent in 2024 than 2026
- AI/technical-themed topics are more prevalent in 2026 than 2024
- New topics exist in 2026 that don't appear in 2024 (emerging senior archetype)

Use `topics_over_time()` with 2 bins (2024, 2026) to quantify topic prevalence shift.

#### Test 4: Controlled regression

```
MgmtKeywordCount_i = β₀ + β₁(Post2026) + β₂(CompanySize) + β₃(Industry) + ε
AIKeywordCount_i   = β₀ + β₁(Post2026) + β₂(CompanySize) + β₃(Industry) + ε
```

Run on senior SWE postings only. β₁ should be negative for management and positive for AI-orchestration. HC3 standard errors.

**Sufficient evidence for RQ6:** Significant decline in management keyword prevalence AND significant increase in AI-orchestration keyword prevalence, confirmed by Fightin' Words and BERTopic showing the same directional shift. The Archetype Shift Index increasing between periods, with a bootstrap CI excluding zero change.

---

### 11f. RQ5: Training implications & RQ7: Historical platform comparison

These are synthesis RQs, not hypothesis tests. They draw on the empirical outputs of RQ1-4 and RQ6.

#### RQ5: Training implications

**Method:** Structured qualitative synthesis.

1. From RQ2's task migration map, identify the top 5-10 skills that migrated to junior roles. These are the skills that training programs must front-load.
2. From RQ1's scope inflation analysis, quantify how much more is now expected of entry-level candidates. This frames the training gap.
3. From RQ6's archetype shift, identify the new senior competencies (AI orchestration) that the training pipeline must eventually develop.
4. Cross-validate against documented cross-profession parallels:
   - Radiology: when AI diagnostic tools deployed, how did residency programs adapt?
   - Accounting: when automated bookkeeping arrived, how did entry requirements change?
   - Aviation: how did autopilot change the pilot training pipeline?
5. Derive prescriptive recommendations, each traceable to a specific empirical finding.

**Output:** The "AI Supervision Residency" framework (Appendix B in research design), grounded in empirical evidence rather than speculation.

#### RQ7: Historical platform comparison

**Method:** Descriptive comparison using O\*NET occupation definitions across time.

1. Pull O\*NET "detailed work activities" and "technology skills" for SOC 15-1252 (Software Developers) across available O\*NET versions (2010, 2015, 2019, 2024).
2. For each version, identify the modal "senior SWE" skill profile.
3. Construct a decade-by-decade summary of the senior SWE archetype:
   - **Mainframe era (1960s-80s):** Hardware optimization, batch processing
   - **PC/C era (1980s-90s):** Systems programming, memory management
   - **Web/Java era (2000s):** Architecture, design patterns, team management
   - **Mobile/cloud era (2010s):** Distributed systems, cross-functional coordination
   - **AI era (2025+):** AI orchestration, system design, less people management
4. Quantify the magnitude of each transition where data exists (O\*NET task change rates). Is the AI-era shift larger or smaller than prior transitions?

**This is contextualization, not causal identification.** It places our RQ1-RQ6 findings in the longer arc of computing history.

---

### 11g. Robustness & multiple testing protocol

This section applies to ALL RQs. Without it, a reviewer can dismiss any individual finding as fragile or cherry-picked.

#### Specification curve analysis

Run every primary test under all defensible specifications (defined in Stage 9k). The specification space:

| Dimension | Variants | Count |
|---|---|---|
| SWE definition | Narrow, standard, broad | 3 |
| Seniority classifier | Our imputer, native labels, description-only | 3 |
| Dedup threshold | Strict (exact), standard (0.70), loose (0.50) | 3 |
| Sample scope | Full, LinkedIn-only, excl. aggregators, top-10 metros | 4 |
| Company capping | None, cap at 10 per company, cap at median | 3 |

Total: 3 × 3 × 3 × 4 × 3 = 324 specifications per test.

```python
import specification_curve as sc
sco = sc.SpecificationCurve(
    df, y_endog='junior_share',
    x_exog='is_post2026',
    controls=[['company_size'], ['industry'], ['is_remote'], ['metro']],
)
sco.fit()
sco.plot()
```

**Rule:** A finding is "robust" if it is significant (p < 0.05) in > 80% of specifications. If 50-80%, it is "suggestive." If < 50%, it is "fragile" and not reported as a finding.

#### Multiple testing corrections

| Scope | Method | Rationale |
|---|---|---|
| Within a single RQ (e.g., testing 15 skills in RQ2) | Benjamini-Hochberg FDR at q = 0.05 | Controls false discovery rate |
| Across RQs (e.g., RQ1 + RQ2 + RQ4 + RQ6 primary tests) | Holm-Bonferroni | More conservative for the study's headline findings |
| Specification curve | Report the fraction significant, not individual p-values | The curve itself accounts for multiplicity |

#### Bootstrap confidence intervals

For every key estimate (junior share change, skill prevalence change, Archetype Shift Index, DiD coefficient), report bootstrap 95% CIs alongside the point estimate. Use 2000 bootstrap resamples.

```python
from scipy.stats import bootstrap
result = bootstrap((junior_share_data,), np.mean, n_resamples=2000,
                   confidence_level=0.95, method='BCa')
```

BCa (bias-corrected and accelerated) intervals are preferred over percentile intervals for skewed distributions.

#### Placebo and falsification tests (from 9l)

| Test | Expected result | What it proves |
|---|---|---|
| Control occupation seniority shift | Null (no shift) | SWE shift is occupation-specific |
| Random time-split within Kaggle | Null | Our method doesn't find "change" in noise |
| Permuted period labels | Observed > 95th percentile of permuted | The difference exceeds random variation |
| SWE-adjacent (data scientist, PM) | Intermediate or null | Calibrates AI-exposure gradient |

#### Effect size reporting

Every test reports both statistical significance AND practical significance:

| Measure | When to use | "Small" | "Medium" | "Large" |
|---|---|---|---|---|
| Cramér's V | Chi-squared tests | 0.10 | 0.30 | 0.50 |
| Cohen's d | Continuous comparisons | 0.20 | 0.50 | 0.80 |
| Rank-biserial r | Mann-Whitney U | 0.10 | 0.30 | 0.50 |
| Log-odds ratio | Fightin' Words | 0.5 | 1.0 | 2.0 |
| Prevalence Δ (pp) | Skill prevalence shifts | 3pp | 5pp | 10pp |

A finding that is statistically significant (p < 0.05) but practically trivial (effect size below "small") is noted but not emphasized.

---

### Analysis outputs by RQ

| RQ | Primary table/figure | What it shows |
|---|---|---|
| **RQ1** | Table: Seniority distribution shift with chi-squared and Cramér's V | Junior share declined by X pp |
| **RQ1** | Figure: UMAP with 2024 seniority boundary, 2026 junior postings overlaid | Visual evidence of content convergence |
| **RQ1** | Table: Scope inflation metrics (5 measures, FDR-corrected) | Entry-level requirements inflated |
| **RQ2** | Figure: Skill migration heatmap (prevalence by seniority × period) | Which skills migrated |
| **RQ2** | Table: Top 15 migrated skills with Δ prevalence and FDR-adjusted p | Statistical proof of migration |
| **RQ2** | Figure: Skill co-occurrence network (2024 vs. 2026 junior) | Structural reorganization of junior skill requirements |
| **RQ3** | Figure: Revelio SWE openings with Bai-Perron breakpoints | External evidence of structural break |
| **RQ3** | Table: Permutation test results for cross-sectional magnitude | Our observed difference exceeds random variation |
| **RQ4** | Table: DiD regression coefficients for 4+ outcomes | SWE-specific effects isolated |
| **RQ4** | Figure: Embedding drift by occupation with bootstrap CIs | Visual evidence of SWE-specific change |
| **RQ6** | Table: Management vs. AI-orchestration keyword prevalence shift | Senior archetype changing |
| **RQ6** | Figure: Archetype Shift Index with bootstrap CI | Directional measure of transformation |
| **RQ7** | Table: Senior SWE archetype by computing era | Historical contextualization |
| **All** | Figure: Specification curve for each primary finding | Robustness across 324 specifications |
| **All** | Table: Placebo test results | Falsification evidence |

---

## Stage 12: Statistical verification

This stage systematically challenges every finding from Stage 11. The goal is to show that our results are not artifacts of sample composition, measurement error, random noise, or macro trends. A reviewer should be able to read this section and conclude: "They tried hard to break their own findings and couldn't."

Stage 12 runs AFTER the primary analysis (Stage 11) produces initial results. We design the verification battery now, but execute it only once we have findings to verify.

---

### 12a. Power analysis and sample size verification

**Why this comes first:** If we lack statistical power to detect a plausible effect, a null result is uninformative and a significant result may be inflated (winner's curse). We must confirm we have adequate power BEFORE interpreting results.

**Actual sample sizes (from data investigation):**

| Group | Kaggle (April 2024) | Scraped (March 2026) |
|---|---|---|
| SWE total | 3,466 | ~14,391 |
| SWE entry-level (native label) | ~385 | ~1,920 |
| SWE entry-level (after imputation, estimated) | ~600 | ~2,500 |
| SWE mid-senior | ~1,770 | ~7,300 |
| Control total | 10,766 | 17,246 |
| Nursing (largest control) | 6,082 | 5,360 |
| Civil eng | 157 | 547 |
| Mechanical eng | 265 | 1,388 |
| Accountant | 1,317 | 2,139 |

**Power calculations for each primary test:**

#### RQ1: Junior share shift (chi-squared)

```python
from statsmodels.stats.power import GofChisquarePower
power_analysis = GofChisquarePower()

# Minimum detectable effect: shift of 5 percentage points in junior share
# e.g., 12.3% → 7.3% or 12.3% → 17.3%
# n1 = 3,466 (Kaggle SWE), n2 = 14,391 (scraped SWE)
# Effect size w for chi-squared ≈ 0.05 for a 5pp shift in one cell of 3×2 table
mde = power_analysis.solve_power(effect_size=None, nobs=3466, alpha=0.05, power=0.80, n_bins=3)
print(f"Minimum detectable effect size (w): {mde:.4f}")
```

**Expected:** With n=3,466 in the smaller group, we have >80% power to detect a Cramér's V of ~0.05 (a 3-5pp shift in junior share). This is adequate for our hypothesized effect.

#### RQ1: Scope inflation (Mann-Whitney on description length)

```python
from statsmodels.stats.power import TTestIndPower
power = TTestIndPower()

# Kaggle entry-level SWE: ~600 (after imputation)
# Scraped entry-level SWE: ~2,500
# What's the minimum detectable Cohen's d?
mde = power.solve_power(effect_size=None, nobs1=600, ratio=2500/600,
                         alpha=0.05, power=0.80, alternative='two-sided')
print(f"Minimum detectable Cohen's d: {mde:.3f}")
```

**Expected:** With 600 vs. 2,500, we can detect Cohen's d ≈ 0.14 (a small effect). This is adequate — if scope inflation is real, the effect should be at least "small" by conventional standards.

#### RQ4: DiD interaction term

```python
# DiD requires adequate samples in all 4 cells:
# SWE × 2024: 3,466    SWE × 2026: 14,391
# Control × 2024: 10,766    Control × 2026: 17,246
# The binding constraint is the smallest cell: SWE × 2024 (3,466)
# With 3,466 in the smallest cell, detect interaction effect of ~0.05 SD
```

**Power table to report in methodology:**

| Test | Smaller group n | Minimum detectable effect | Adequate? |
|---|---|---|---|
| Junior share shift (chi-sq) | 3,466 | Cramér's V ≈ 0.05 (3-5pp shift) | Yes |
| Description length (M-W) | 600 entry-level | Cohen's d ≈ 0.14 | Yes |
| Skill prevalence per skill (z-test) | 600 entry-level | 4pp prevalence shift | Marginal — some rare skills may be underpowered |
| DiD interaction | 3,466 SWE × 2024 | β₃ ≈ 0.05 SD | Yes |
| Embedding drift (bootstrap) | 600 entry-level | cos distance ≈ 0.02 | Yes |
| Archetype Shift Index (bootstrap) | ~1,770 senior × 2024 | Ratio change ≈ 0.10 | Yes |

**Decision rule:** If a test is underpowered (power < 0.60 for the hypothesized effect), do NOT run it as a primary test. Report it as exploratory or aggregate to a coarser level (e.g., combine rare skills into skill clusters).

---

### 12b. Within-period placebo tests

**The core logic:** If our method detects a "change" between two halves of the same period (where no real change occurred), it's measuring noise, not signal. Every finding must pass this sanity check.

#### Placebo 1: Early vs. late Kaggle (within April 2024)

The Kaggle data spans ~4 weeks (April 5-20, 2024). Split into two halves:
- Early: before April 12 (30,101 rows, 743 SWE)
- Late: April 12+ (93,748 rows, 2,723 SWE)

Run the SAME tests from Stage 11 on this split:

| Test | Expected result | If NOT null |
|---|---|---|
| Junior share shift (chi-sq) | p > 0.05 (no shift) | Our method detects noise |
| Description length (M-W) | p > 0.05 | Scraping artifacts within Kaggle |
| Skill prevalence shifts (FDR) | 0 significant skills | Too many false positives |
| Content convergence (redefinition rate) | Rate ≈ baseline error rate | Classifier is unstable |

**The early/late split also controls for scraping order effects.** If the Kaggle scraper captured different types of jobs on different days (e.g., scraping tech companies first, then healthcare), the split would reveal this artifact.

**Sample size concern:** 743 SWE postings in the early split is small. If tests are underpowered at this sample size, compute the minimum detectable effect and report: "Our placebo test would have detected an effect of size X with 80% power. We observed null results, consistent with no within-period change, though small effects cannot be ruled out."

#### Placebo 2: Day-to-day variation in scraped data

We have 14 daily scrapes. Split into two arbitrary halves:
- Week 1: March 5-11 (7 days)
- Week 2: March 12-18 (7 days)

Run the same comparison between these two weeks. We expect NO significant differences — if we find them, it means daily variation is a concern and our results need date-level controls.

**Stronger version:** Run the full test battery on every possible 7-day vs. 7-day split (C(14,7) = 3,432 splits). Compute the distribution of test statistics under these permutations. Our cross-period test statistic should be far outside this null distribution.

```python
from itertools import combinations
import numpy as np

daily_indices = list(range(14))  # days 0-13
null_distribution = []
for combo in combinations(daily_indices, 7):
    week_a = postings_from_days(combo)
    week_b = postings_from_days(set(daily_indices) - set(combo))
    stat = compute_junior_share_diff(week_a, week_b)
    null_distribution.append(stat)

# Our cross-period statistic should exceed 99% of this null
cross_period_stat = compute_junior_share_diff(kaggle_swe, scraped_swe)
p_value = (np.abs(null_distribution) >= np.abs(cross_period_stat)).mean()
```

#### Placebo 3: Random subsample equivalence

Randomly split each period's SWE postings into two halves (A and B). Compare A₂₀₂₄ vs. B₂₀₂₄ and A₂₀₂₆ vs. B₂₀₂₆. Neither comparison should yield significant results. If they do, our sample has internal heterogeneity that inflates test statistics.

Run 100 random splits and report the fraction producing p < 0.05 for each test. Under the null, this fraction should be ~5%. If it's much higher, something is wrong (e.g., batch effects from daily scraping, company clustering, geographic concentration).

---

### 12c. Control group verification

**The logic:** Our primary findings claim SWE roles changed between 2024 and 2026. If control occupations (nursing, civil engineering, accounting) show the SAME changes, the finding is not SWE-specific — it's a platform change, a macro trend, or a scraping artifact.

#### Control group 1: Low-AI-exposure occupations (primary controls)

Run every Stage 11 test on control occupations:

| Test | SWE result | Control expected | If control matches SWE |
|---|---|---|---|
| Junior share shift | Significant decline | No change | Macro trend, not AI-driven |
| Scope inflation (desc. length, skill count) | Significant increase | No change | Scraping artifact |
| Content convergence | High redefinition rate | Low/baseline rate | Classifier drift |
| AI keyword emergence | Significant increase | No change (or much smaller) | Expected — this IS the differentiator |

**Per-occupation controls (run separately, then pool):**

| Occupation | Kaggle n | Scraped n | Adequate for DiD? |
|---|---|---|---|
| Nursing | 6,082 | 5,360 | Yes — large sample, strong control |
| Accountant | 1,317 | 2,139 | Yes |
| Electrical eng | 377 | 1,435 | Marginal |
| Financial analyst | 372 | 1,417 | Marginal |
| Mechanical eng | 265 | 1,388 | Marginal |
| Civil eng | 157 | 547 | Underpowered — report but caveat |
| Chemical eng | 27 | 34 | No — exclude from analysis |

**Pooled control:** Combine all control occupations (10,766 Kaggle + 17,246 scraped). This gives the best-powered DiD. Then run per-occupation to show results are consistent across controls.

**AI-exposure gradient test:** Order occupations by Felten et al. (2023) AI-exposure scores. If our effects scale with AI exposure (largest for SWE, intermediate for data scientists, small for accountants, null for nurses), that's strong evidence of an AI-driven mechanism.

```python
# Compute effect magnitude by occupation, plot against AI-exposure score
effects = {}
for occ in ['SWE', 'data_scientist', 'product_manager', 'accountant',
            'financial_analyst', 'mech_eng', 'nursing']:
    effects[occ] = compute_junior_share_shift(occ)

ai_exposure = get_felten_scores(effects.keys())
correlation = np.corrcoef(list(ai_exposure.values()),
                          list(effects.values()))[0,1]
# If correlation > 0.5 and significant, the effect scales with AI exposure
```

#### Control group 2: SWE-adjacent occupations (dose-response)

SWE-adjacent roles (data scientist, product manager, UX designer, QA engineer) are partially AI-exposed. If they show intermediate-sized effects (smaller than SWE, larger than controls), that's a dose-response pattern consistent with AI exposure being the mechanism.

| Occupation | AI exposure (Felten) | Expected effect |
|---|---|---|
| SWE | High | Large |
| Data scientist | High | Large (similar to SWE) |
| Product manager | Medium-high | Medium |
| UX designer | Medium | Small-medium |
| QA engineer | Medium | Medium |
| Accountant | Low | Small/null |
| Civil engineer | Low | Null |
| Nursing | Very low | Null |

**If the gradient is confirmed:** This is a powerful finding. It turns RQ4 from a binary question ("SWE-specific or not?") into a continuous relationship: the magnitude of restructuring scales with AI exposure.

#### Control group 3: Same-company cross-occupation

For companies appearing in both SWE and non-SWE postings within the same period, compare whether the company's SWE postings changed differently than its non-SWE postings. This holds employer-level factors constant (company culture, HR practices, scraping effects) and isolates the occupation-level effect.

```python
# Companies with both SWE and control postings in both periods
overlap_companies = (set(swe_2024['company']) & set(swe_2026['company'])
                    & set(ctrl_2024['company']) & set(ctrl_2026['company']))

for company in overlap_companies:
    swe_shift = junior_share(swe_2026[company]) - junior_share(swe_2024[company])
    ctrl_shift = junior_share(ctrl_2026[company]) - junior_share(ctrl_2024[company])
    within_company_did = swe_shift - ctrl_shift
```

This is the most demanding test: it asks whether the SAME EMPLOYER changed its SWE hiring differently than its non-SWE hiring.

---

### 12d. Alternative explanation tests

Each test here addresses a specific alternative explanation for our findings. If any alternative survives, we must either refute it or acknowledge it as a limitation.

#### Alternative 1: "It's just LinkedIn's algorithm/UI changing"

**Threat:** LinkedIn may have changed how it displays, ranks, or categorizes job postings between 2024 and 2026. A change in LinkedIn's seniority labeling algorithm would directly cause the seniority distribution shift we observe.

**Test:** Compare the distribution of LinkedIn's native `job_level` labels between periods (for LinkedIn-only data). If "not applicable" rate changed, or if LinkedIn relabeled categories, the shift is partly artificial.

Also check: did the ratio of seniority labels change for NON-SWE postings? If LinkedIn changed its labeling, it would affect all occupations equally. SWE-specific changes cannot be explained by platform changes.

```python
# Compare "not applicable" rate across periods and occupations
na_rate_swe_2024 = (swe_2024['seniority_native'] == 'not applicable').mean()
na_rate_swe_2026 = (swe_2026['seniority_native'] == 'not applicable').mean()
na_rate_ctrl_2024 = (ctrl_2024['seniority_native'] == 'not applicable').mean()
na_rate_ctrl_2026 = (ctrl_2026['seniority_native'] == 'not applicable').mean()
# If SWE NA rate changed but control didn't → NOT a platform change
```

#### Alternative 2: "It's a composition effect (different companies)"

**Threat:** The 2026 sample might capture a different mix of companies (more FAANG, fewer startups, or vice versa) with inherently different seniority distributions.

**Test:** The Oaxaca-Blinder decomposition from 9g. Additionally:

```python
# Within overlapping companies: does the shift hold?
shared_companies = set(swe_2024['company_normalized']) & set(swe_2026['company_normalized'])
within_shift = (
    swe_2026[swe_2026['company_normalized'].isin(shared_companies)]['seniority']
    .value_counts(normalize=True) -
    swe_2024[swe_2024['company_normalized'].isin(shared_companies)]['seniority']
    .value_counts(normalize=True)
)
# If the shift holds within shared companies, it's not composition
```

**Report:** "X% of the observed shift is explained by composition changes; Y% remains after controlling for company overlap."

#### Alternative 3: "It's the scraping methodology"

**Threat:** Our scraper captures different postings than the Kaggle scraper. Query design, pagination depth, geographic scope, and anti-bot handling all differ. The observed differences may reflect what each scraper captures, not what the labor market looks like.

**Tests:**
1. **Indeed-only vs. LinkedIn-only:** Run the analysis separately on each platform within the scraped data. If both platforms show the same SWE shift, it's unlikely to be a single-platform scraping artifact.
2. **Geographic subset test:** Restrict both datasets to the same top-5 states (CA, TX, WA, VA, NY — present in both). If the shift holds in matched geography, geographic coverage differences aren't driving it.
3. **Company-matched test:** Use only companies appearing in both datasets. This is the strongest control for scraping methodology — the same companies, captured by different scrapers.

#### Alternative 4: "It's seasonal (April vs. March)"

**Threat:** April and March have different hiring patterns. Spring budget cycles, new-year planning, etc.

**Test:** Pull JOLTS monthly data for the information sector. Compare March vs. April historically (2019-2024). If the March-April difference in job openings is typically <3%, seasonality cannot explain a 5+pp shift in junior share.

```python
jolts_info = jolts[jolts['series'] == 'Information']
jolts_info['month'] = jolts_info['date'].dt.month
march = jolts_info[jolts_info['month'] == 3]['value']
april = jolts_info[jolts_info['month'] == 4]['value']
seasonal_diff = ((april.values - march.values) / march.values).mean()
print(f"Average March→April seasonal change: {seasonal_diff:.1%}")
```

#### Alternative 5: "Ghost jobs are driving the results"

**Threat:** If ghost job prevalence increased between 2024 and 2026 (CRS estimates 12.5% in 2023 → ~20% in 2025), and ghost jobs disproportionately inflate entry-level requirements, our "scope inflation" finding could partly reflect ghost job growth rather than genuine restructuring.

**Test:** Run the full analysis excluding postings flagged as high ghost-job risk (from Stage 8a). If findings hold after exclusion, ghost jobs aren't driving them. Report results both with and without ghost-flagged postings.

---

### 12e. Sensitivity analyses

Each sensitivity analysis asks: "If we made a different reasonable methodological choice, would we reach the same conclusion?"

#### Seniority classifier sensitivity

Run all primary tests under 3 classification schemes:
1. Our rule-based imputer (default)
2. LinkedIn native labels where available, imputed only where missing
3. SetFit description-based classifier (from Stage 5a)

Report a table: do all 3 classifiers produce the same directional finding? If yes, the finding is robust to classifier choice. If not, identify which postings are classified differently and investigate.

#### SWE definition sensitivity

Run under 3 SWE definitions:
1. Narrow: core SWE titles only (excluding data engineer, ML engineer)
2. Standard: our canonical SWE_PATTERN (default)
3. Broad: include data scientist, QA engineer, product engineer

If the finding holds only under the broad definition, it's driven by non-core SWE roles and needs nuancing.

#### Deduplication sensitivity

Run under 3 dedup levels:
1. No near-dedup (keep all postings with unique URLs)
2. Standard near-dedup (cosine ≥ 0.70)
3. Aggressive dedup (cosine ≥ 0.50, collapses multi-location)

#### Platform sensitivity

Run on:
1. LinkedIn only (both periods on the same platform)
2. Full sample (LinkedIn + Indeed for 2026)

If findings differ between LinkedIn-only and full-sample, the Indeed inclusion is driving partial results. LinkedIn-only is the more conservative and defensible comparison.

#### Company concentration sensitivity

Run with:
1. Full sample
2. Capped at 10 postings per company
3. Excluding top-5 companies from each dataset
4. Excluding DataAnnotation specifically (crowdwork platform in Kaggle)

#### Geographic sensitivity

Run on:
1. Full sample
2. Top-5 metros only (matched across datasets)
3. Excluding remote postings

#### Produce a consolidated sensitivity table

| Finding | Default | Alt seniority | Alt SWE def | LinkedIn-only | Cap 10/co | Top-5 metros | Verdict |
|---|---|---|---|---|---|---|---|
| Junior share ↓ | p=X, Δ=Y | p=X, Δ=Y | ... | ... | ... | ... | Robust/Suggestive/Fragile |
| Scope inflation | ... | ... | ... | ... | ... | ... | ... |
| Skill migration (top skill) | ... | ... | ... | ... | ... | ... | ... |
| DiD β₃ | ... | ... | ... | ... | ... | ... | ... |
| Archetype shift | ... | ... | ... | ... | ... | ... | ... |

**Verdict rules:**
- **Robust:** Significant (p < 0.05) and same direction in ≥ 5 of 6 sensitivity variants
- **Suggestive:** Significant and same direction in 3-4 of 6 variants
- **Fragile:** Significant in < 3 variants → not reported as a finding

---

### 12f. Effect size calibration

**Why this is separate from significance testing:** A statistically significant result with a tiny effect size is practically meaningless. Conversely, a large effect size with p = 0.06 is still informative. We report both, but calibrate what effect sizes mean in context.

#### External benchmarks for calibration

| Metric | Our observed Δ | Context benchmark | Interpretation |
|---|---|---|---|
| Junior share change | X pp | Hershbein & Kahn (2018): junior share dropped ~5pp during Great Recession | If our Δ is similar magnitude, it's comparable to a recession-level shock |
| Description length change | X words | Deming & Kahn (2018): skill requirements increased ~20% 2007-2017 | This gives a decade-scale baseline for scope inflation |
| AI keyword emergence | X pp | Acemoglu et al. (2022): AI-exposed occupations saw 15-20% task reallocation | Calibrates what "meaningful" AI adoption looks like |
| Embedding drift (cosine) | X | Need to establish: what's the typical within-period drift? | If cross-period drift >> within-period drift, the change is real |

#### Practical significance thresholds

For this study, we define practically significant as:
- **Junior share change:** ≥ 3 percentage points (roughly the MDE given our power)
- **Skill prevalence change:** ≥ 5 percentage points for any individual skill
- **Description length change:** ≥ 15% increase in requirements section
- **Redefinition rate:** ≥ 10 percentage points above the baseline misclassification rate
- **DiD interaction:** ≥ 0.10 SD (small-to-medium effect)
- **Archetype Shift Index change:** ≥ 20% relative increase

These thresholds are set before looking at results (pre-registered). Findings that cross both statistical and practical significance thresholds are "strong evidence." Findings that are statistically significant but below practical thresholds are "detectable but modest."

---

### 12g. Reproducibility protocol

**Why this matters:** Computational social science papers are notoriously difficult to replicate. Pin every random seed, model version, and data transform.

1. **Random seeds:** Set `np.random.seed(42)` and `torch.manual_seed(42)` at the start of every analysis script. Report the seed.
2. **Model versioning:** Pin all HuggingFace model versions by commit hash (not just model name). E.g., `TechWolf/JobBERT-v2` at commit `abc123`.
3. **Data versioning:** Hash the unified.parquet file (SHA-256) and report in methodology. Any re-run must produce the same hash after preprocessing.
4. **Pipeline code:** All preprocessing and analysis code in version-controlled scripts. No manual steps between preprocessing and results.
5. **Intermediate outputs:** Cache and version all embeddings, topic models, and classifier predictions. A reviewer should be able to start from cached embeddings and reproduce all results without re-running the embedding step.

---

### Verification outputs

| Output | Type | Paper section |
|---|---|---|
| Power analysis table | Table | §3.5 Statistical power |
| Within-period placebo results (3 tests) | Table | §4.4 Placebo tests |
| Control group results (per occupation + pooled) | Table + figure | §4.3 Control analysis |
| AI-exposure gradient plot | Figure | §4.3 (if gradient confirmed) |
| Alternative explanation tests (5 tests) | Table | §4.4 or §5 Discussion |
| Sensitivity analysis matrix | Table (consolidated) | §4.5 Sensitivity |
| Specification curve plots (per finding) | Figures | §4.5 or Appendix |
| Effect size calibration table | Table | §4.2 alongside main results |
| Reproducibility metadata | Appendix | Appendix C |

---

## How validation feeds into the methodology section

The validation battery produces the following tables/figures for the paper:

| Output | Section | Content |
|---|---|---|
| **Table: Data sources and coverage** | §3.1 Data | Date ranges, sample sizes, platform, selection mechanism |
| **Table: Missing data rates** | §3.1 Data | Per-field missingness by dataset (from 9c) |
| **Table: Representativeness** | §3.2 Validation | Our distribution vs. OES, with correlations and dissimilarity indices (from 9a) |
| **Table: Classifier performance** | §3.3 Classification | Per-class precision/recall/F1 for SWE detection and seniority imputation (from 9e) |
| **Table: Cross-dataset comparability** | §3.2 Validation | Distribution comparison tests for key variables (from 9f) |
| **Table: Bias threat summary** | §3.4 Limitations | The bias table from 9l |
| **Figure: Specification curve** | §4 Results | Effect stability across all defensible specifications (from 9j) |
| **Table: Placebo tests** | §4 Results | Null results on control occupations and random splits (from 9k) |
| **Table: Data funnel** | §3.1 Data | Raw → cleaned → final counts (from Stage 4f) |

---

## Output specification

The pipeline produces:

### Primary output: `data/unified.parquet`

All columns from current schema, plus:
- `source_platform`, `period`, `posting_age_days`
- `description_core` (boilerplate-stripped)
- `title_normalized`, `company_name_normalized`, `location_normalized`
- `seniority_imputed` (our classifier, applied to all rows)
- `seniority_native` (LinkedIn's label where available)
- `salary_source`, `salary_validated` (bool)
- `is_swe`, `is_control`, `is_aggregator`, `is_multi_location`
- `ghost_job_risk`, `description_quality_flag`
- `seniority_source`, `dedup_method`, `preprocessing_version`

### Quality report: `data/quality_report.json`

```json
{
  "pipeline_version": "1.0",
  "run_date": "2026-03-18",
  "funnel": {
    "kaggle_raw": 3382601,
    "kaggle_after_dedup": "...",
    "scraped_raw": "...",
    "scraped_after_dedup": "...",
    "final_swe": "...",
    "final_control": "..."
  },
  "classification_rates": {
    "swe_rate_kaggle": "...",
    "swe_rate_scraped": "...",
    "seniority_imputation_rate_kaggle": "...",
    "seniority_imputation_rate_scraped": "..."
  },
  "missing_data": {
    "salary_missing_kaggle": "...",
    "salary_missing_scraped": "...",
    "description_missing": "...",
    "seniority_unknown_after_imputation": "..."
  },
  "aggregator_stats": {
    "total_flagged": "...",
    "real_employer_extracted": "..."
  },
  "boilerplate_stats": {
    "median_chars_removed": "...",
    "median_length_before": "...",
    "median_length_after": "..."
  }
}
```

### Preprocessing log: `data/preprocessing_log.txt`

Human-readable log of every stage: rows in, rows out, rows flagged, decisions made. This feeds directly into the methodology section writeup.

---

## Open questions and decisions needed

### Critical (blocks analysis design)

1. **The Kaggle date range problem.** Our "cross-period comparison" is April 2024 vs. March 2026, not "2023-2024 vs 2026". This is still a valid comparison (23-month gap during a period of rapid AI deployment), but it changes the framing. Options:
   - (a) Proceed with April 2024 vs. March 2026 and adjust all documentation
   - (b) Download the [1.3M LinkedIn Jobs & Skills (2024)](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) dataset — 10x more data, may have broader temporal coverage, has augmented skills
   - (c) Both — use the 1.3M dataset as primary baseline, keep arshkon for comparison
   - **Recommendation:** Option (c). Download and evaluate the 1.3M dataset. If it covers multiple months of 2024, it becomes our primary baseline. The arshkon dataset becomes a validation cross-reference.

2. **Indeed inclusion.** The scraped data is 40% Indeed. Including Indeed gives us more data (especially salary data) but introduces a platform composition effect that Kaggle doesn't have. Options:
   - (a) Use LinkedIn-only scraped data for primary analysis, Indeed for sensitivity
   - (b) Use combined LinkedIn+Indeed, with a `source_platform` control variable
   - (c) Run everything both ways

3. **Description length discrepancy.** Kaggle SWE median 3,242 chars vs. scraped 5,036 chars is a 55% gap. Before attributing any description-length finding to "scope inflation," we must determine whether this is a scraping artifact. Possible investigation:
   - Scrape a few current LinkedIn postings manually and compare against our scraper output
   - Check whether Kaggle truncated descriptions (look for suspiciously round max lengths)
   - Compare boilerplate-stripped lengths to see if the gap narrows

### Important (affects methodology section)

4. **Uniform seniority classification vs. native-when-available.** Recommendation: use our imputer as canonical, keep native labels for validation. But this means we're deliberately ignoring LinkedIn's own labels for 66.5% of Kaggle data where they're available.

5. **"Software Development Engineer" is a false negative.** This title appears 16+ times in the non-SWE file but is clearly a SWE role (it's Amazon's standard SWE title). The expanded SWE_PATTERN should catch it, but verify.

6. **Kaggle companion file join.** The companion files have industry (99.3% coverage) and company data. Joining them is straightforward and gives us industry-controlled comparisons. Recommend doing this.

7. **Multi-location postings.** Keep all variants as default (literature standard). Run sensitivity analysis collapsing to unique (title, company, description).

8. **Aggregator postings.** Keep and flag. Present results both with and without.

### Nice-to-have

9. **DataAnnotation dominance in Kaggle.** 168 postings = 5.4% of Kaggle SWE. This company provides AI training data annotation work — arguably not traditional SWE. Should it be excluded? Or flagged and sensitivity-tested?

10. **Query saturation test.** Re-scrape one query-city combo with 50 and 100 result limits to test whether 25 is a binding constraint.

11. **Boilerplate removal approach.** Start regex, upgrade to LLM if needed.

12. **Near-dedup threshold calibration.** Ideal is to label 50-100 candidate pairs and empirically tune the similarity threshold. Pragmatic default: use literature standard (0.70 cosine).
