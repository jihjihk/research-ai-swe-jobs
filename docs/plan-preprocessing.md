# Pre-Processing Pipeline

Date: 2026-03-19
Status: Draft — ready for review before implementation

This document covers Stages 1-8: transforming raw data into the analysis-ready `unified.parquet` dataset. For validation and exploration, see `plan-exploration.md`. For hypothesis testing, see `plan-analysis.md`.

---

## Context

This document specifies the preprocessing pipeline (Stages 1-8) that transforms raw data into the analysis-ready `unified.parquet` dataset. The current implementation (`harmonize.py`) handles basic schema unification and URL dedup. This document specifies the full pipeline needed for publication quality.

**Project state (as of 2026-03-19):** The scraper has been running continuously since March 5, 2026. We are currently building the preprocessing and analysis pipeline while data collection continues. The scraper will run through April 2026 and beyond, which means:
- **Seasonality is not a blocking concern.** Once April 2026 data is collected, we can compare April 2024 (Kaggle) vs. April 2026 (scraped) — same month, 2-year gap. The March 2026 data serves as pipeline development material and will supplement the April-to-April primary comparison.
- **The pipeline must be ready before the "important" data starts arriving.** April 2026 is our target comparison window. All preprocessing, validation, and analysis code should be tested and working on March data before then.

**Research questions:** RQ1-RQ7 as defined in `docs/research-design-h1-h3.md`.

---

## Data inventory (corrected after investigation)

### What we actually have

| | Kaggle (arshkon) | Kaggle (asaniczka) — TO DOWNLOAD | Scraped |
|---|---|---|---|
| **Source** | `arshkon/linkedin-job-postings` | `asaniczka/1-3m-linkedin-jobs-and-skills-2024` | Our daily scraper |
| **Total rows** | 123,849 | ~1.3M | ~100K SWE-file + ~86K non-SWE-file (14 days) |
| **Date range** | **April 2024 only** | January 2024 (need to verify span) | March 5-18, 2026 (ongoing) |
| **SWE postings** | ~3,466 | ~30-40K estimated | ~14,391 unique |
| **Platform** | LinkedIn only | LinkedIn only | LinkedIn (60K) + Indeed (41K) |
| **Seniority** | 66.5% labeled | Has `job_level` field | LI: 100% labeled; Indeed: 0% |
| **Skills** | 98% null in main; companion has 35 coarse categories | Augmented skills in `job_skills.csv` | Extract from description |
| **Descriptions** | Yes (full text) | Has `job_summary.csv` (need to verify if full descriptions) | Yes (full text) |
| **Size** | 493MB | ~2GB (compressed) | ~584MB |

**Primary analysis platform decision:** LinkedIn only. Indeed data is used for sensitivity analyses only (see resolved decisions below).

### The 1.3M dataset (asaniczka) — download and evaluate

The [asaniczka/1-3m-linkedin-jobs-and-skills-2024](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) dataset contains 3 CSVs:
- `linkedin_job_postings.csv` — main postings file with `job_title`, `company`, `job_location`, `job_level`, `job_type`, `first_seen`, `job_link`
- `job_skills.csv` — augmented skills per posting (structured, mapped)
- `job_summary.csv` — may contain descriptions or summaries (need to verify after download)

**Why this matters:** At 1.3M rows (vs. 124K for arshkon), this gives us ~10x more baseline data. If it spans multiple months of 2024, we get within-2024 trend analysis. Its augmented skills data could replace or supplement our ESCO-based extraction for the baseline period.

**Evaluation steps after download:**
1. Check actual date range — does `first_seen` span multiple months of 2024?
2. Check whether `job_summary.csv` has full descriptions or just summaries
3. Check field overlap with arshkon dataset — are they from the same scraper?
4. Count SWE postings using our expanded SWE_PATTERN
5. Compare seniority distributions and company overlap with arshkon
6. If it subsumes the arshkon dataset (same or broader coverage, same period), use asaniczka as primary baseline and arshkon as validation cross-reference
7. If they cover different time windows, combine them for broader temporal coverage

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

**3. Description lengths differ significantly between datasets.** Kaggle SWE median: 3,242 chars. Scraped SWE (LinkedIn): 5,036 chars. This 55% difference could be (a) real scope inflation, (b) different scraping methods capturing different content, or (c) boilerplate differences. **This is a major confounder for RQ1 description complexity analysis.**

**Investigation plan for description length discrepancy (must complete before any length-based analysis):**
1. **Matched-pair comparison (10 pairs, LLM-assisted):** Find 10 companies appearing in both datasets. For each, pull one SWE posting from each period. LLM compares side-by-side: "Is the 2026 description genuinely longer, or does it include more boilerplate/formatting? What content differences exist?" Human reviews the LLM's comparison summaries.
2. **Boilerplate-stripped comparison:** After Stage 3 (boilerplate removal), re-compare `description_core` lengths. If the gap narrows substantially, the difference was boilerplate, not content.
3. **Check Kaggle truncation:** Examine the max description length distribution in Kaggle data. If there's a suspicious cliff at a specific character count (e.g., 5,000 or 10,000 chars), the Kaggle scraper may have truncated.
4. **Check formatting differences:** Kaggle descriptions may be pre-stripped of HTML/markdown. Our scraper captures raw markdown. After stripping markdown, the lengths may converge.
5. **Cross-reference with asaniczka dataset:** If the 1.3M dataset has descriptions, compare their lengths against the arshkon dataset for the same time period. If they differ, it's scraper-dependent.

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

### Lessons from Kaggle community preprocessing

Research into how other Kaggle projects preprocess these LinkedIn datasets reveals that community preprocessing is **extremely basic** compared to what publication-quality research requires:

- **Standard Kaggle approach:** Drop duplicates on job_id, handle missing values (drop or fill), drop irrelevant columns, basic type conversion. No boilerplate removal, no near-dedup, no aggregator handling, no cross-dataset validation.
- **No one does boilerplate stripping.** EEO statements, "About Us" sections, and benefits boilerplate are left in descriptions. This inflates word counts and corrupts topic models. Our pipeline addresses this in Stage 3.
- **No one validates seniority labels.** Projects use LinkedIn's native `job_level` field as-is, including the 23-33% "not applicable" entries. Our pipeline applies uniform imputation (Stage 5b).
- **Skills are typically keyword-counted, not taxonomy-mapped.** Most projects count occurrences of hand-picked keywords ("Python", "AWS"). Our pipeline uses ESCO-mapped extraction (Stage 10e) for structured, comparable skill analysis.
- **Dedup is always exact.** No near-duplicate detection. Given that Lightcast reports up to 80% raw duplication, this is a significant gap that inflates sample sizes.

**Implication:** Our preprocessing pipeline is substantially more rigorous than anything in the Kaggle ecosystem. This is appropriate for a research paper but means we cannot rely on community code for preprocessing — we build it ourselves.

---

## Pipeline overview

```
Raw Data (Kaggle CSV + daily scraped CSVs)
  │
  ├─ Stage 1: Ingest & Schema Unification
  ├─ Stage 2: Aggregator / Staffing Company Handling
  ├─ Stage 3: Boilerplate Removal
  ├─ Stage 4: Deduplication (within-dataset + cross-dataset)
  ├─ Stage 5: Classification (SWE detection + seniority imputation)
  ├─ Stage 6: Field Normalization & Validation
  ├─ Stage 7: Temporal Alignment
  ├─ Stage 8: Quality Flags & Filtering
  │
  ▼
Analysis-Ready Dataset (unified.parquet)
  + quality_report.json
  + preprocessing_log.txt
```

Each stage produces logged counts (rows in → rows out, rows flagged) so we can report the full data funnel in our methodology section.

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

### 2c. Validation (3-tier — see review protocol)

- **Tier 1:** Flag all AGGREGATORS list matches automatically.
- **Tier 2:** LLM reviews 100 aggregator postings — extracts real employer name, verifies boilerplate was stripped, checks if job content is preserved. Prompt: "Given this aggregator posting, what is the real employer? Was the job content preserved after boilerplate removal? Any issues?"
- **Tier 3:** Human reviews 20 items the LLM flagged as problematic + 10 random "clean" items.

**Decision point:** If aggregator postings are mostly duplicates of direct postings from the same employer (i.e., Amazon posts directly AND Lensa reposts it), they should be deduplicated against the direct posting. If they surface unique jobs not posted directly, keep them. The LLM can help identify this by comparing aggregator postings against direct postings from the same employer.

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

### 3d. Validation (3-tier — see review protocol)

- **Tier 1:** Compute description length before/after removal. Flag any posting where >80% of content was stripped (possible over-removal) or <10% was stripped (possible under-removal).
- **Tier 2:** LLM reviews 200 post-removal descriptions. Prompt: "Compare the original and stripped description. Was any actual job requirement removed? Was boilerplate left in? Rate: over-stripped / under-stripped / correct." Focus on postings near the over/under-removal thresholds.
- **Tier 3:** Human reviews 30 items LLM flagged as over- or under-stripped + 10 random "correct" items.

Report: median description length before vs. after boilerplate removal, by source. Also report the LLM-assessed accuracy rate on the 200-item sample.

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

#### Validation protocol for the classifier (3-tier — see review protocol)

1. **Gold-standard annotation (LLM pre-labeled + human corrected):**
   - Sample 500 postings stratified by classification tier and confidence level. Oversample from the ambiguous zone (confidence 0.40-0.70).
   - **Tier 2 (LLM):** Claude Sonnet pre-labels each posting: SWE / SWE-adjacent / control / other, with reasoning. Prompt includes title, first 500 chars of description, and company name.
   - **Tier 3 (Human):** Annotator reviews LLM labels, correcting disagreements. This is 5-10x faster than labeling from scratch.
   - Compute Cohen's kappa between LLM and human. If kappa > 0.80, LLM is a reliable second annotator.

2. **Per-tier accuracy:** Report precision/recall/F1 for each classification tier separately. If Tier 1 (regex) has 98% precision but Tier 3 (description) has 75% precision, that's important context.

3. **Cross-period consistency:** Run the classifier on both datasets. If the SWE detection rate differs dramatically, investigate via LLM: have it review 100 borderline postings from each period and explain why they're SWE or not. Check for systematic period-specific patterns.

4. **Error analysis (LLM-assisted):** For every false positive and false negative in the gold-standard sample, LLM categorizes the error type:
   - Title ambiguity (e.g., "Engineer" without qualifier)
   - Novel title (e.g., "AI Agent Orchestrator" — not in reference set)
   - Aggregator confusion (e.g., staffing company title differs from actual role)
   - Description mismatch (title says SWE but description is project management)
   Human reviews the LLM's error categorizations for the most consequential errors.

5. **Comparison with SOC-based approach:** As a robustness check, run `occupationcoder` or a similar SOC mapper on a sample and compare its SWE classification against ours. Agreement rate establishes external validity.

#### Implementation notes

- **Speed:** Tier 1 (regex) processes all ~200K postings in seconds. Tier 2 (embedding with JobBERT-v2) processes ~5K titles per minute on CPU. Tier 3 (SetFit) processes ~3K descriptions per minute on CPU. Total pipeline: ~15-20 minutes for full dataset.
- **Reproducibility:** Pin model versions (e.g., `TechWolf/JobBERT-v2` from HuggingFace, specific commit hash). Cache all embeddings. Log the reference title sets. This ensures re-running the pipeline produces identical results.
- **The reference title set is a researcher decision.** Document it exhaustively. It should be grounded in O\*NET sample titles, supplemented with titles observed in our data that were manually verified as SWE. Publish it in the appendix.

### 5b. Seniority imputation

**Current approach:** Rule-based classifier using title keywords (primary) + description years-of-experience (fallback). Validated at ~80% accuracy against LinkedIn's own labels on the labeled subset.

**What needs to happen for rigor:**

1. **Apply uniformly:** The same `impute_seniority()` function must run on both Kaggle and scraped data. Currently, the notebook applies it only to scraped data and uses Kaggle's native `formatted_experience_level` directly. This creates a systematic difference — we'd be comparing LinkedIn's classifier (April 2024) against our classifier (March 2026).

2. **Gold-standard validation (3-tier — see review protocol):** Sample 500 postings stratified by (source, predicted seniority, title ambiguity). LLM pre-labels seniority from title + description with reasoning. Human corrects LLM labels. Compute kappa between LLM and human, then evaluate our imputer against the corrected gold standard. Report per-class precision/recall.

3. **Report imputation rate:** What fraction of each dataset required imputation vs. had a native label? If the imputation rate differs between periods (e.g., 27% in 2026 vs. X% in Kaggle), that itself is a measurement artifact that could drive apparent seniority shifts.

4. **Three-level bucketing:** For cross-period comparison, use the 3-level scheme (junior / mid / senior) rather than the 6-level scheme. Finer granularity amplifies classifier noise.

**Decision point:** Should we use our `impute_seniority()` for ALL postings (ignoring native labels), or use native labels where available and impute only where missing? The research docs recommend the former (retroactive classification — apply one classifier uniformly). This is the Lightcast approach.

**Recommendation:** Use our imputer as the canonical seniority variable. Keep native labels as a separate column for validation and sensitivity analysis.

**Future upgrade path:** The same 3-tier approach used for SWE detection (regex → embedding → description classifier) could be applied to seniority. Tier 2 would measure similarity to reference titles per seniority level (O\*NET provides "Job Zone" ratings that map to seniority). Tier 3 would use a SetFit classifier trained on LinkedIn's native labels. This is lower priority than improving SWE detection because (a) our rule-based imputer already validates at ~80% against LinkedIn labels and (b) the 3-level bucketing (junior/mid/senior) is coarse enough to absorb most misclassification noise. But if seniority classification accuracy becomes a reviewer concern, this is the path forward.

### 5c. Control occupation detection

**Current approach:** `CONTROL_PATTERN` regex for civil/mechanical/electrical/chemical engineers and nurses.

**Risk:** Control occupations may have very different LinkedIn posting rates than SWE. The DiD design (RQ4) requires adequate coverage in both datasets.

**Action:** After classification, report control occupation counts by source. If any control group has <100 postings in either period, it's too thin for DiD and should be flagged.

### 5d. Embedding model validation

**Why:** The plan uses [JobBERT-v2](https://huggingface.co/TechWolf/JobBERT-v2) as the default embedding model for title classification (Stage 5a Tier 2), exploration (Stage 10), and analysis (Stage 11). Before committing to it, we should validate that it outperforms general-purpose alternatives on OUR data.

**Critical finding from research:** JobBERT-v2 has a **max sequence length of 64 tokens** — it is designed for job titles, NOT full descriptions. For description-level embeddings (topic modeling, content convergence, drift measurement), we need a different model or a dual-model approach.

**Candidate models to benchmark:**

| Model | Dimension | Max tokens | Domain | Notes |
|---|---|---|---|---|
| `TechWolf/JobBERT-v2` | 1024 | 64 | Job titles + skills | Fine-tuned on 5.5M job title pairs. Best for title-level tasks. |
| `TechWolf/JobBERT-v3` | 1024 | 64 | Multilingual job titles | 21M training pairs. Multilingual. May be overkill for English-only. |
| `sentence-transformers/all-mpnet-base-v2` | 768 | 384 | General purpose | Strong general-purpose baseline. Handles full descriptions. |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 | General purpose | Fast, smaller. Good for rapid iteration. |
| `intfloat/e5-large-v2` | 1024 | 512 | General purpose | Top MTEB scores. Handles long descriptions. |
| `BAAI/bge-large-en-v1.5` | 1024 | 512 | General purpose | Competitive with e5-large. |

**Benchmark protocol:**

1. **Title classification task:** Sample 200 titles with known SWE/non-SWE labels (from gold-standard annotation). Embed with each model. Run k-NN (k=5) against the O\*NET reference title set. Measure precision@5 and recall@5. JobBERT-v2 should win here.

2. **Seniority separation task:** Embed 500 postings with known seniority labels. Compute silhouette score for junior/mid/senior clusters. The model that best separates seniority levels in embedding space is best for our content convergence analysis.

3. **Description similarity task:** Take 50 known duplicate pairs (from near-dedup) and 50 random non-duplicate pairs. Embed full descriptions with each model. Compute AUC for duplicate detection. Models with 64-token limits will perform poorly here — confirming we need a description-capable model.

**Expected outcome — dual-model approach:**
- **JobBERT-v2** for all title-level tasks: SWE classification (Tier 2), title similarity, title clustering
- **all-mpnet-base-v2** or **e5-large-v2** for all description-level tasks: topic modeling, content convergence, skill extraction context, embedding drift

Cache both sets of embeddings. Use the appropriate model for each downstream task.

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

**Primary comparison:** April 2024 vs. April 2026 (once April 2026 scraping completes). Same month, 2-year gap. No seasonal offset.

**Current state:** March 2026 data is available now for pipeline development and testing. April 2026 data will be the primary comparison window. March 2026 data supplements the analysis and provides a secondary comparison.

**Seasonality is not a blocking concern.** The scraper runs continuously. By late April 2026, we'll have a full month of April data for the primary same-month comparison. March 2026 data is used for:
- Building and testing the full preprocessing + analysis pipeline
- Secondary comparison (April 2024 vs. March 2026, 1-month seasonal offset)
- Within-2026 stability check (March vs. April scraped data should show similar distributions)

**Action:** Create a `period` column. For the primary analysis, use `2024-04` (Kaggle) vs. `2026-04` (scraped April). For secondary analysis, include `2026-03`. Correct all prior documentation that says "2023-2024".

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

---

## Review protocol: 3-tier validation strategy

Every quality check in this pipeline uses a 3-tier approach that maximizes coverage while keeping human effort manageable. This replaces pure manual spot-checking with a scalable LLM-assisted pipeline.

### The 3 tiers

```
Tier 1: Rule-based validation (automated, full dataset)
  ↓ flagged items
Tier 2: LLM review (automated, hundreds to thousands of items)
  ↓ items LLM flags as problematic or uncertain
Tier 3: Human review (small sample, only what matters)
```

**Tier 1 — Rule-based (runs on every row):** Deterministic checks that catch obvious problems. Example: description < 50 chars, salary < $10K, title matches SWE_EXCLUDE, seniority label contradicts years-of-experience. These are fast and catch the bulk of issues.

**Tier 2 — LLM review (runs on hundreds-thousands of items):** For checks that require judgment (e.g., "Is this description real job content or mostly boilerplate?", "Does this seniority label match the description?"), use Claude via the CLI. This scales the "human eye" to cover far more data than manual review allows.

**Implementation — Claude Code CLI:**
```bash
# Single-item review
claude -p "Given this job posting title and description, classify:
1. Is this a software engineering role? (yes/no/borderline)
2. What seniority level? (entry/mid/senior/unknown)
3. Is this a ghost job? (yes/no/maybe — check if requirements are unrealistic for stated level)
4. Quality issues? (boilerplate-only/spam/non-english/duplicate-template/none)

Title: ${TITLE}
Description (first 500 chars): ${DESC}

Respond as JSON." --output-format json

# Batch processing with rate limiting
for id in $(cat sample_ids.txt); do
    claude -p "$(build_prompt $id)" --output-format json >> results.jsonl
    sleep 0.5  # rate limiting
done
```

**Rate limiting considerations:** Anthropic Max plan ($200/month). Haiku is ~$0.25/M input tokens. A 500-char description ≈ 150 tokens. 1,000 reviews ≈ 150K tokens ≈ $0.04. Even 10,000 reviews costs < $0.50. Rate limit is the constraint, not cost — add 0.5-1s delay between calls.

**Tier 3 — Human review (targeted):** Only review items that Tier 2 flagged as problematic or uncertain. Also review a random sample of Tier 2's "clean" outputs (10-20 items) to validate the LLM's judgment. If the LLM's error rate on the random sample exceeds 10%, expand human review scope.

### Spot-check table (with tier assignments)

| Check | Tier 1 (rule-based) | Tier 2 (LLM, sample size) | Tier 3 (human) |
|-------|---|---|---|
| **SWE classification** | Regex match/exclude flags | 500 ambiguous titles — LLM classifies SWE/non-SWE with reasoning | 50 items LLM flagged as borderline + 20 random "confident" items |
| **Seniority imputation** | Title keyword match + years-of-experience extraction | 500 postings — LLM reads title+description, assigns seniority | 50 items LLM disagreed with our imputer + 20 random agreements |
| **Boilerplate removal** | Section header regex; paragraph hash dedup | 200 post-removal descriptions — LLM checks "was any real job content stripped?" | 30 items LLM flagged as potentially over-stripped |
| **Aggregator handling** | Company name in AGGREGATORS list | 100 aggregator postings — LLM extracts real employer name from description | 20 items to verify LLM's employer extraction |
| **Near-dedup** | Exact match on (title, company, location) | 200 candidate pairs — LLM judges "same job or different?" | 30 pairs LLM was uncertain about |
| **Ghost job flags** | Entry-level + 5yr+ experience OR salary > $150K | 300 entry-level postings — LLM rates "realistic requirements for entry-level?" | 30 items where LLM and rules disagreed |
| **Description quality** | Length < 50 chars; language detection | 200 short/suspicious descriptions — LLM reads and tags quality | 20 items to validate LLM tags |
| **Gold-standard annotation** | N/A | LLM labels 500 postings (SWE, seniority, skills) as initial annotations | Human annotator corrects LLM labels; compute kappa between LLM and human |

### Gold-standard annotation with LLM pre-labeling

The gold-standard annotation set (needed for classifier validation, Stage 9e) uses LLM pre-labeling to dramatically reduce human effort:

1. **Sample 500 postings** stratified by source, predicted seniority, and classification confidence.
2. **LLM pre-labels** each posting for: SWE/non-SWE, seniority level, top 5 skills, ghost job risk, quality flags. Use Claude Sonnet for higher accuracy on this critical task.
3. **Human annotator reviews LLM labels**, correcting where they disagree. This is 5-10x faster than labeling from scratch.
4. **Compute inter-rater reliability** between LLM and human (Cohen's kappa). If kappa > 0.80, the LLM is reliable enough to serve as a second annotator. If kappa < 0.60, expand human review.
5. **Adjudicate disagreements** to produce the final gold standard.

**Critical warning (Ashwin et al. 2025):** LLM coding errors are NOT random — they correlate with text characteristics. Always validate against human labels before trusting LLM annotations at scale. The kappa check in step 4 is non-negotiable.

---

---

## Implementation order

Stages have dependencies. Recommended sequence:

```
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
   ↓
10. Spot-checks (3-tier: rules → LLM → human, after pipeline runs)
```

**Parallelizable:** Stages 2, 3, and 6a can be developed in parallel. Stage 4 depends on 2 and 3.

**Iteration:** Validation results (see `plan-exploration.md`, Stage 9) may trigger re-runs of preprocessing. Build the pipeline to be re-runnable end-to-end.

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

## Resolved decisions

These questions were open; decisions are now recorded here for reference.

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | **Kaggle date range** | Download and evaluate the 1.3M asaniczka dataset. Use it as primary baseline if it has broader coverage. Keep arshkon for cross-reference. | 10x more data, augmented skills, potentially broader temporal span. |
| 2 | **Indeed inclusion** | **LinkedIn-only for primary analysis.** Indeed for sensitivity only. | Eliminates platform composition artifact. Kaggle is LinkedIn-only, so primary comparison should be LinkedIn-to-LinkedIn. |
| 3 | **Description length discrepancy** | Investigate through LLM-assisted matched-pair comparison + automated checks (see investigation plan in data inventory section). Do not draw length-based conclusions until resolved. | 55% gap could be scraping artifact. |
| 4 | **Seniority classification** | Use our imputer as canonical for all rows. Keep native labels as separate column for validation/sensitivity. | Retroactive classification (Lightcast approach). Avoids comparing two different classifiers across periods. |
| 5 | **"Software Development Engineer"** | Expanded SWE_PATTERN catches it. Verify during Stage 5 implementation. | Amazon's standard SWE title; confirmed false negative in current pattern. |
| 6 | **Kaggle companion files** | Join during ingest (Stage 1c). | Confirmed joinable. 99.3% industry coverage, company size/location. |
| 7 | **Multi-location postings** | Keep all variants as default. Sensitivity-test collapsing. | Literature standard (Hershbein & Kahn 2018). |
| 8 | **Aggregator postings** | Keep and flag. Sensitivity-test excluding. | 9% of scraped, ~15% of Kaggle SWE. Too high to ignore, too low to remove by default. |
| 9 | **DataAnnotation dominance** | Flag and sensitivity-test excluding. LLM reviews 20 DataAnnotation postings to determine if they're real SWE jobs or data annotation/crowdwork. Human verifies 5 of the LLM's assessments. | 168 postings = 5.4% of Kaggle SWE. May be crowdwork, not traditional SWE. |
| 10 | **Boilerplate removal** | Start with regex. Upgrade to LLM if spot-checks show poor accuracy. | Regex is fast, reproducible, and sufficient for most patterns. |
| 11 | **Near-dedup threshold** | Use literature standard (0.70 cosine). Calibrate on labeled pairs if time permits. | Abdelaal et al. 2024 achieved F1=0.94 with similar thresholds. |
| 12 | **Embedding model** | Dual-model approach: JobBERT-v2 for titles, general-purpose model (all-mpnet-base-v2 or e5-large-v2) for descriptions. Validate with benchmark in Stage 5d. | JobBERT-v2 has 64-token limit — great for titles, cannot handle descriptions. |
