# Pre-Processing Pipeline v3: LLM-Augmented

Date: 2026-03-21
Status: Draft — ready for review before implementation
Supersedes: v2 (2026-03-20), v1 (2026-03-19)

This document covers the preprocessing pipeline that transforms raw data into the analysis-ready `unified.parquet` and `unified_observations.parquet` datasets. For validation and exploration, see `plan-exploration.md`. For hypothesis testing, see `plan-analysis.md`.

## What changed from v2

The v2 pipeline assumed a single data schema across sources and relied on the scraper harmonizer for schema unification. The v3 redesign corrects several data inventory errors and restructures the pipeline around verified source schemas:

- **Research questions updated:** RQ1-RQ4 as defined in `docs/1-research-design.md` (employer-side restructuring, task/requirement migration, employer-requirement/worker-usage divergence, mechanisms). Replaces the old RQ1-RQ7 from the retired `docs/research-design-h1-h3.md`.
- **Data sources narrowed to three:** Kaggle arshkon, Kaggle asaniczka, and scraped current-format files. No YC, no Apify.
- **Legacy scraped data dropped:** Mar 5-18 data used 25 results/query and lacked search metadata columns (`search_query`, `query_tier`, `search_metro_id`, etc.). Stage 1 should load all matching scraped dates in the current 41-column format and skip incompatible legacy files.
- **Stage 1 rewritten:** Pipeline owns all schema unification across three different source schemas. It ingests approved rows equivalently, does not filter by occupation class, and does NOT rely on `scraper/harmonize.py`.
- **Two outputs:** `unified.parquet` (canonical postings, one row per unique posting) + `unified_observations.parquet` (daily panel, one row per posting per scrape_date). Keep both because posting-level and observation-level analyses have different units of observation.
- **Asaniczka limitations documented:** Only two seniority levels present (Mid senior: 17,045; Associate: 1,124). No entry-level postings exist in this dataset, which limits its use as a historical baseline for junior-share analysis (RQ1), but the source still contributes SWE-adjacent and control-occupation rows.

The LLM augmentation stages (9-12) are architecturally unchanged from v2. Prompt design, seniority classification rationale, and the 3-tier review protocol carry forward.

---

## Context

**Project state (as of 2026-03-21):** The scraper upgraded to 100 results/query and the new 41-column format on March 20, 2026. The v1 preprocessing pipeline produced an initial `data/unified.parquet` (1.2M rows, 53 columns) from the old data. This v3 pipeline will rebuild from scratch using only the three verified sources below.

**Key constraint:** Seasonality is not a blocking concern. Once April 2026 data is collected, we can compare April 2024 (Kaggle arshkon) vs. April 2026 (scraped) — same month, 2-year gap. March 2026 data serves as pipeline development material and supplements the primary comparison.

**Research questions:** RQ1-RQ4 as defined in `docs/1-research-design.md`:
- RQ1: Employer-side restructuring (junior share/volume, junior scope inflation, senior role redefinition, source/metro heterogeneity)
- RQ2: Task and requirement migration (which requirements moved down/shifted)
- RQ3: Employer-requirement / worker-usage divergence
- RQ4: Mechanisms (interview-based, qualitative — not addressed by this pipeline)

---

## Data inventory

### Source 1: Kaggle arshkon

| Field | Value |
|---|---|
| **Path** | `data/kaggle-linkedin-jobs-2023-2024/postings.csv` |
| **Total rows** | 123,849 |
| **Date range** | April 2024 (single-month snapshot; the "2023-2024" title refers to the project timeframe) |
| **Platform** | LinkedIn only |
| **SWE postings** | ~3,466 (~2.1% title match rate among 165K unique titles) |
| **ID column** | `job_id` (integer) |
| **Title** | `title` |
| **Description** | `description` (inline, full text) |
| **Seniority** | `formatted_experience_level` ("Mid-Senior level", "Entry level", "Associate", "Director", "Internship", "Executive", "Not Applicable"); 66.5% populated |
| **Date** | `listed_time` (epoch milliseconds) |
| **Company** | `company_name`, `company_id` |
| **Location** | `location` |
| **Companion files** | `jobs/job_industries.csv` (99.3% coverage via `job_id`), `companies/companies.csv` (`company_size`, `state`, `city` via `company_id`), `companies/employee_counts.csv` (`employee_count` via `company_id`), `mappings/industries.csv` (422 industry_id to industry_name) |

### Source 2: Kaggle asaniczka

| Field | Value |
|---|---|
| **Path** | `data/kaggle-asaniczka-1.3m/` |
| **Total rows** | ~1.35M |
| **Date range** | January 12-17, 2024 |
| **Platform** | LinkedIn only |
| **SWE postings** | 18,169 US SWE matches (~1.3% match rate among US postings) |
| **ID column** | `job_link` (URL string) |
| **Title** | `job_title` |
| **Description** | NOT inline. Descriptions in separate `job_summary.csv` — join on `job_link`. 96.2% coverage. |
| **Seniority** | `job_level` — only TWO values present: "Mid senior" (17,045 SWE matches) and "Associate" (1,124 SWE matches). NO entry-level postings exist. |
| **Date** | `first_seen` (YYYY-MM-DD format) |
| **Company** | `company` |
| **Location** | `job_location` |
| **Salary** | None |
| **Company metadata** | None (no company size, no industry) |
| **Skills** | `job_skills.csv` (join on `job_link`) |
| **Filtering required** | `search_country == "United States"` |
| **Other columns** | `search_city`, `search_country`, `search_position`, `job_type` |

**Critical limitation for RQ1:** This dataset contains zero entry-level postings. It cannot serve as a historical baseline for junior posting share or junior scope inflation. It is useful for mid-senior and associate-level content analysis only (RQ2 requirement migration within those levels).

### Source 3: Scraped current-format files

| Field | Value |
|---|---|
| **Path** | `data/scraped/YYYY-MM-DD_{swe,non_swe}_jobs.csv` |
| **Total rows** | ~3,680 SWE rows/day + ~30,888 non-SWE rows/day |
| **Date range** | March 20, 2026 onward (ongoing) |
| **Platform** | LinkedIn + Indeed |
| **ID column** | `id` (string) |
| **Title** | `title` |
| **Description** | `description` (inline, full text) |
| **Columns** | 41 columns total |
| **Search metadata** | `search_query`, `query_tier`, `search_metro_id`, `search_metro_name`, `search_metro_region`, `search_location` |
| **Scrape metadata** | `scrape_date`, `site` (linkedin/indeed) |
| **Query design** | 100 results/query, 26 metros |
| **Cross-day overlap** | ~40% of IDs reappear across days |

**Platform-specific field availability (scraped):**

| Field | LinkedIn | Indeed |
|---|---|---|
| `job_level` | 100% | 0% |
| `company_industry` | 100% | 0% |
| `description` | 100% | 100% |
| `date_posted` | 2.8% | 100% |
| `company_num_employees` | 0% | 91% |

**File structure note:** The `_swe_jobs.csv` and `_non_swe_jobs.csv` split reflects query tier, not title pattern. SWE files contain Tier 1 SWE query results; non-SWE files contain Tier 2+3 results. Scraper-level dedup already removes duplicate IDs within each day.

### Dropped: old scraped data (Mar 5-18)

The scraper ran from March 5-18 with 25 results/query and an older CSV format lacking `search_query`, `query_tier`, `search_metro_id`, `search_metro_name`, `search_metro_region`, and `search_location` columns. This data is excluded from the v3 pipeline. The upgrade to 100 results/query on March 20 means the new data has 4x the coverage per query and includes the search metadata needed for metro-level analysis.

### Cross-source notes

1. **Description lengths differ by ~55% between Kaggle and scraped** (Kaggle SWE median: 3,242 chars; scraped SWE LinkedIn: 5,036 chars). Could be real scope inflation, scraping differences, or boilerplate. Investigation plan unchanged from v1.
2. **Aggregators present across sources.** Kaggle: DataAnnotation (168), Dice (35), Apex Systems (29). Scraped: Lensa, Jobs via Dice. DataAnnotation alone is 5.4% of Kaggle SWE postings.
3. **Primary analysis platform:** LinkedIn only. Indeed data is used for sensitivity analyses only. Both Kaggle sources are LinkedIn-only, so LinkedIn-only analysis provides the cleanest cross-period comparison.

---

## Pipeline architecture

```
Raw Data (3 sources: arshkon CSV, asaniczka CSV+joins, daily scraped CSVs)
  |
  +-- Stage 1: Ingest & Schema Unification          [REWRITTEN for v3]
  |     +-- 1a: Arshkon ingest + companion joins
  |     +-- 1b: Asaniczka ingest + description/skills joins
  |     +-- 1c: Scraped ingest (all current-format files, LinkedIn + Indeed)
  |     +-- 1d: Schema unification to canonical columns
  +-- Stage 2: Aggregator / Staffing Handling        [UNCHANGED from v1]
  +-- Stage 3: Rule-Based Boilerplate Removal        [UNCHANGED from v1]
  +-- Stage 4: Deduplication                         [UNCHANGED from v1]
  +-- Stage 5: Rule-Based Classification             [UNCHANGED from v1]
  +-- Stage 6: Field Normalization                   [UNCHANGED from v1]
  +-- Stage 7: Temporal Alignment                    [UNCHANGED from v1]
  +-- Stage 8: Quality Flags                         [UNCHANGED from v1]
  |
  v
intermediate/stage8_final.parquet (rule-based pipeline output)
  |
  +-- Stage 9: LLM Pre-Filtering                    [UNCHANGED from v2]
  +-- Stage 10: LLM Classification (single call)    [UNCHANGED from v2]
  +-- Stage 11: LLM Response Integration             [UNCHANGED from v2]
  +-- Stage 12: Three-Way Validation                 [UNCHANGED from v2]
  |
  v
data/unified.parquet          (canonical postings: one row per unique posting)
data/unified_observations.parquet  (daily panel: one row per posting per scrape_date)
  + data/quality_report.json
  + data/preprocessing_log.txt
```

**Design principle:** The rule-based pipeline (Stages 1-8) runs first and produces complete rule-based outputs. The LLM layer (Stages 9-12) runs on top, adding LLM-derived columns alongside rule-based columns. This means:
- The pipeline works end-to-end without any LLM calls (rule-based fallback)
- LLM outputs are additive — they never overwrite rule-based columns
- We can run ablation studies comparing rule-based vs. LLM vs. both
- If an LLM call fails for a posting, the rule-based values remain

**Two output files:**
- `unified.parquet`: One row per unique posting. For Kaggle sources, each posting appears once. For scraped data, each unique `id` appears once with its first-seen metadata. This is the primary analysis file.
- `unified_observations.parquet`: One row per posting per scrape_date. Only meaningful for scraped data (Kaggle sources appear once). Tracks when postings appear/disappear from search results. Supports posting-duration analysis and daily-panel sensitivity checks.

Each stage produces logged counts (rows in, rows out, rows flagged) for the methodology section.

---

## Stage 1: Ingest and schema unification (rewritten)

This is the major structural change in v3. Each source has a different schema, different ID format, different description storage, and different field availability. Stage 1 handles each source independently, then unifies to canonical columns.

Stage 1 is a source-agnostic ingest layer, not an occupation classifier. It should keep approved rows from each source, normalize schema and provenance, and leave occupation classification to Stage 5.

### Stage 1a: Arshkon ingest

**Input:** `data/kaggle-linkedin-jobs-2023-2024/postings.csv` + companion files.

**Steps:**
1. Load `postings.csv`. Apply the source-specific joins and ingest normalization only; do not filter historical rows by occupation class.
2. Join `jobs/job_industries.csv` on `job_id` to get `industry_id`.
3. Join `mappings/industries.csv` on `industry_id` to get `industry_name`.
4. Join `companies/companies.csv` on `company_id` to get `company_size`, `state`, `city`.
5. Join `companies/employee_counts.csv` on `company_id` to get `employee_count`.
6. Convert `listed_time` from epoch milliseconds to date.
7. Preserve `formatted_experience_level` as `seniority_raw`.
8. Map `formatted_experience_level` to canonical seniority in `seniority_native`: "Entry level" -> "entry", "Associate" -> "associate", "Mid-Senior level" -> "mid-senior", "Director" -> "director", "Internship" -> "intern", "Executive" -> "executive", "Not Applicable" -> null. Unmapped values should remain null in `seniority_native`, not passed through as raw strings.

**Output columns mapped:**
- `uid` <- `"arshkon_" + str(job_id)`
- `source` <- `"kaggle_arshkon"`
- `source_platform` <- `"linkedin"`
- `title` <- `title`
- `description_raw` <- original `description`
- `description` <- normalized working copy of `description`
- `company_name` <- `company_name`
- `location` <- `location`
- `date_posted` <- converted from `listed_time`
- `seniority_raw` <- original `formatted_experience_level`
- `seniority_native` <- mapped from `formatted_experience_level`
- `company_industry` <- from companion join
- `company_size` <- numeric `employee_count`
- `company_size_raw` <- `employee_count`
- `company_size_category` <- companion `company_size`
- `scrape_date` <- null (not applicable)
- `site` <- `"linkedin"`
- `search_query`, `query_tier`, `search_metro_id`, `search_metro_name`, `search_metro_region`, `search_location` <- all null

### Stage 1b: Asaniczka ingest

**Input:** `data/kaggle-asaniczka-1.3m/` main file + `job_summary.csv` + `job_skills.csv`.

**Steps:**
1. Load main postings file.
2. Filter to `search_country == "United States"`.
3. Apply source-specific ingest normalization only; do not filter by occupation class.
4. Join `job_summary.csv` on `job_link` to get descriptions. Log the 3.8% of postings without descriptions. These remain in the dataset with null descriptions.
5. Join `job_skills.csv` on `job_link` to get skills (stored as a separate column for later analysis).
6. Preserve `job_level` as `seniority_raw`.
7. Map `job_level` to canonical seniority in `seniority_native`: "Mid senior" -> "mid-senior", "Associate" -> "associate". All other values -> null.

**Output columns mapped:**
- `uid` <- `"asaniczka_" + sha256(job_link)[:16]` (URL is too long for an ID; hash it)
- `source` <- `"kaggle_asaniczka"`
- `source_platform` <- `"linkedin"`
- `title` <- `job_title`
- `description_raw` <- from `job_summary.csv` join (null if no match)
- `description` <- normalized working copy of `description_raw`
- `company_name` <- `company`
- `location` <- `job_location`
- `date_posted` <- `first_seen` (already YYYY-MM-DD)
- `seniority_raw` <- original `job_level`
- `seniority_native` <- mapped from `job_level`
- `company_industry` <- null (not available)
- `company_size` <- null (not available)
- `scrape_date` <- null
- `site` <- `"linkedin"`
- `search_query` <- `search_position` (closest equivalent)
- `query_tier` <- null
- `search_metro_id` <- null
- `search_metro_name` <- `search_city`
- `search_metro_region` <- null
- `search_location` <- null
- `asaniczka_skills` <- from `job_skills.csv` join (supplementary column)

**Note on missing entry-level data:** The absence of entry-level postings in asaniczka is a data characteristic, not a filtering error. The original dataset simply does not contain postings labeled below "Associate." This must be documented in the methodology and accounted for when interpreting junior-share trends (RQ1). Arshkon is the only historical source with entry-level postings, but both historical Kaggle sources remain useful for later occupation classification and control analyses.

### Stage 1c: Scraped ingest (current-format files)

**Input:** `data/scraped/YYYY-MM-DD_{swe,non_swe}_jobs.csv` files in the current 41-column format.

**Steps:**
1. Glob for all CSV files matching the date pattern. Do not hard-code a month boundary.
2. Skip YC files and skip incompatible legacy scraped files that do not match the current 41-column schema.
3. Load each valid file. The `scrape_date` is extracted from the filename.
4. Both `_swe_jobs.csv` and `_non_swe_jobs.csv` are loaded (the file split reflects query tier, not SWE classification — Stage 5 handles classification).
5. Preserve `job_level` as `seniority_raw`.
6. Map `job_level` to canonical seniority in `seniority_native`: "entry level" -> "entry", "mid-senior level" -> "mid-senior", "associate" -> "associate", "director" -> "director". Unmapped values should remain null. For Indeed rows (`site == "indeed"`), `job_level` is expected to be null.
7. For the daily panel output (`unified_observations.parquet`), keep all rows including cross-day duplicates.
8. For the canonical output (`unified.parquet`), deduplicate by `id`: keep the first occurrence of each unique ID with its earliest `scrape_date`.

**Output columns mapped:**
- `uid` <- `site + "_" + id`
- `source` <- `"scraped"`
- `source_platform` <- `site` ("linkedin" or "indeed")
- `title` <- `title`
- `description_raw` <- original `description`
- `description` <- normalized working copy of `description_raw`
- `company_name` <- `company`
- `location` <- `location`
- `date_posted` <- `date_posted` (sparse for LinkedIn: 2.8%)
- `seniority_raw` <- original `job_level`
- `seniority_native` <- mapped from `job_level` (null for Indeed)
- `company_industry` <- `company_industry` (LinkedIn only)
- `company_size` <- parsed numeric `company_num_employees` (Indeed only)
- `company_size_raw` <- original `company_num_employees`
- `scrape_date` <- from filename
- `site` <- `site`
- `search_query` <- `search_query`
- `query_tier` <- `query_tier`
- `search_metro_id` <- `search_metro_id`
- `search_metro_name` <- `search_metro_name`
- `search_metro_region` <- `search_metro_region`
- `search_location` <- `search_location`

### Stage 1d: Schema unification

Concatenate outputs from 1a, 1b, and 1c into a single dataframe with canonical columns. Verify:
- All rows have a non-null `uid`
- All rows have a non-null `source`
- All rows have a non-null `title`
- Log null rates for key ingest fields by source, including `description`, `description_raw`, `seniority_raw`, and `seniority_native`
- Log description-join coverage and seniority-mapping coverage by source

**Memory note:** Use pyarrow for all reads. Process asaniczka in chunks (1.35M rows). Never load the full asaniczka dataset into pandas at once.

---

## Stages 2-8: Rule-based pipeline

These stages are implemented and working from v1. Summary:

| Stage | Module | Purpose |
|---|---|---|
| 1 | `stage1_ingest.py` | Schema unification (rewritten for v3 — see above) |
| 2 | `stage2_aggregators.py` | Aggregator flagging, real employer extraction |
| 3 | `stage3_boilerplate.py` | Section-based boilerplate removal (regex) |
| 4 | `stage4_dedup.py` | Exact + near-duplicate detection |
| 5 | `stage5_classification.py` | 3-tier occupation classification (SWE / SWE-adjacent / control; regex + LLM lookup + embedding), multi-signal seniority |
| 6 | `stage678_normalize_temporal_flags.py` | Location/date normalization, language detection |
| 7 | (same file) | Period assignment, posting age computation |
| 8 | (same file) | Ghost job heuristics, description quality flags |

**Rule-based columns produced (preserved as ablation baselines):**
- `is_swe`, `is_swe_adjacent`, `swe_confidence`, `swe_classification_tier`
- `seniority_imputed`, `seniority_source`, `seniority_confidence`, `seniority_3level`
- `description_core` (boilerplate-stripped)
- `ghost_job_risk`

---

## Stage 9: LLM pre-filtering

### Goal

Reduce the number of LLM calls from the full posting count to ~10-50K that actually need LLM review. We can afford thousands of LLM calls, but not hundreds of thousands.

### Pre-filter rules

Apply these filters sequentially. A posting is **skipped** (no LLM call) if it matches any skip condition:

**1. Obvious non-SWE (skip LLM entirely):**
Titles matching `SWE_EXCLUDE` AND not matching `SWE_INCLUDE`. These are civil engineers, mechanical engineers, sales engineers, etc. Classify as non-SWE with high confidence. Rule-based classification is sufficient.

**2. Non-English postings (skip LLM entirely):**
Detected by `langdetect` in Stage 6e. Exclude from both LLM and analysis.

**3. Duplicate descriptions (cache hit):**
After dedup (Stage 4), many surviving postings share identical descriptions (same role at different locations, or reposted). Cache LLM responses by `sha256(description_text)`. Identical descriptions get the same classification without a second call.

**4. High-confidence rule-based agreement (skip if rules are clear):**
If all of the following are true, skip LLM:
- SWE classification: `swe_classification_tier == "regex"` (high-confidence regex match)
- Seniority: `seniority_source` starts with `"title_"` (clear signal from title keywords or level numbers)
- Ghost job: `ghost_job_risk == "low"` AND no title-seniority contradiction

These postings have unambiguous rule-based labels. LLM adds no value.

**Postings that MUST go to the LLM:**
- Any posting where `swe_classification_tier` is `"embedding_adjacent"` or `"unresolved"` (ambiguous SWE classification)
- Any posting where `seniority_source` is `"description_yoe"`, `"description_language"`, or `"unknown"` (no clear title signal for seniority)
- Any posting flagged `ghost_job_risk == "high"` (validate the high-risk flag)
- All postings go through LLM for boilerplate removal (the rule-based remover misses front-matter boilerplate in ~20-25% of cases)

**Revised estimate:** After pre-filtering, the LLM will process approximately:
- ~10-50K postings for the full 4-task bundle (SWE + seniority + boilerplate + ghost)
- Additional postings for boilerplate-only LLM processing (where SWE/seniority are already clear but boilerplate removal needs improvement)

In practice, we should profile the first 100 calls, then extrapolate volume and cost before running the full batch.

### Description dedup for caching

Before sending to the LLM, compute `sha256(description_text)` for all candidate postings. Group by hash. For each unique hash, send only one LLM call and apply the result to all postings sharing that hash. This is the single biggest volume reducer.

---

## Stage 10: LLM classification (single bundled call)

### Architecture

One LLM call per posting performs all four tasks simultaneously on the **full job description** (not truncated). This avoids the truncation bias that plagued the v1 LLM validation (which used only the first 300-500 characters).

### Model selection

- **Primary:** GPT-5.4 mini via the CLI:
  ```bash
  codex exec --full-auto --config model=gpt-5.4-mini "<prompt>" --skip-git-repo-check
  ```
- **Alternative:** Claude Haiku via `claude -p` (Max plan, $200/month) if Codex doesn't work or has reliability issues.
- **Validation model:** GPT-5.4 (full) for the three-way comparison (Stage 12).

### Prompt design

```
You are a labor economics research assistant classifying job postings.
Perform ALL FOUR tasks below on this job posting. Return ONLY valid JSON.

TASK 1 — SWE CLASSIFICATION
Classify this role into exactly one category:
- "SWE": The role's primary function is writing, designing, or maintaining
  software. Includes: software engineers, full-stack developers,
  frontend/backend engineers, mobile developers, ML engineers, data engineers
  who primarily write code, DevOps engineers whose description emphasizes
  writing code for infrastructure. Test: does this person spend most of their
  time producing or maintaining code?
- "SWE_ADJACENT": Technical roles that involve some code but where coding is
  not the primary function. Includes: data analysts who write SQL/Python,
  DevOps focused on operations rather than code, QA/SDET roles, technical
  program managers, solutions architects. Test: this person uses code as a
  tool but their primary output is not software.
- "NOT_SWE": Roles where software development is not a meaningful part of
  the job. Includes: hardware engineers, civil/mechanical/electrical
  engineers, sales engineers, support engineers, project managers,
  non-technical roles. Also includes roles with misleading titles (e.g.,
  "Systems Engineer - Train Control" at a transit agency).

Edge cases:
- Firmware engineers: SWE if primarily writing firmware code, SWE_ADJACENT if
  primarily hardware integration
- "Systems Engineer": depends entirely on description
- Data engineers: SWE if building data pipelines in code, SWE_ADJACENT if
  managing/analyzing data

TASK 2 — SENIORITY CLASSIFICATION
Look ONLY for explicit seniority signals in the title and description:
- "junior", "jr", "intern", "new grad", "entry-level", "early career" -> "entry"
- "associate", "I", "1" (as a level code) -> "associate"
- "senior", "sr", "II", "2", "staff", "principal", "lead", "architect" -> "mid-senior"
- "director", "VP", "head of", "chief" -> "director"
- No clear signal -> "unknown"

IMPORTANT: Do NOT infer seniority from:
- Responsibilities, tech stack complexity, or team size
- Years-of-experience requirements (companies inflate YOE)
- Company reputation or typical leveling
When in doubt, classify as "unknown". We want high precision, not high recall.

TASK 3 — BOILERPLATE IDENTIFICATION
Identify which parts of the description are boilerplate vs. core job content.
Boilerplate includes: company overview/About Us, EEO/diversity statements,
benefits/compensation sections, application instructions, recruiter platform
framing (e.g., "This is a job that [name] is recruiting for..."), corporate
mission/values statements.
Core job content includes: role description, responsibilities, requirements,
qualifications, nice-to-haves, tech stack.

Return the core job content ONLY. Rules:
- Do NOT paraphrase. Return exact text from the original description.
- Preserve the original formatting and line breaks.
- If the entire description is core content, return it unchanged.
- If the entire description is boilerplate, return an empty string.

TASK 4 — GHOST JOB ASSESSMENT
Assess whether this posting's requirements are realistic for its stated level:
- "realistic": Requirements match the stated or apparent seniority level
- "inflated": Requirements are significantly higher than what the stated
  level would normally demand (e.g., entry-level title asking for 5+ years,
  or a junior role requiring expertise in 10+ technologies)
- "ghost_likely": Strong signals this is not a genuine open position
  (impossibly broad requirements, contradictory signals, copy-paste template
  with no specific details)

If the seniority is unclear, assess based on what a reasonable interpretation
of the role would require.

---

TITLE: {title}
COMPANY: {company}
DESCRIPTION:
{full_description}

---

Respond with this exact JSON structure:
{
  "swe_classification": "SWE" | "SWE_ADJACENT" | "NOT_SWE",
  "seniority": "entry" | "associate" | "mid-senior" | "director" | "unknown",
  "description_core": "<exact text from original, boilerplate removed>",
  "ghost_assessment": "realistic" | "inflated" | "ghost_likely"
}
```

### Seniority classification — design rationale

This is the most important design change from v1. The LLM validation showed that inferring seniority from responsibilities produces garbage (87/100 classified as mid-senior when given truncated descriptions). The new approach:

**What the LLM does:** Looks for explicit seniority signals only — title keywords, level codes, explicit "entry-level" or "senior" language. Maps to the enum. Defaults to "unknown" when ambiguous.

**What the LLM does NOT do:** Infer seniority from responsibilities, tech stack complexity, team size, YOE requirements, or company reputation.

**Why:** The research goal is to understand how companies label and frame their roles at different seniority levels. We need clean seniority labels that reflect the company's intent, not reverse-engineered "true" seniority. Later analysis examines how skills/requirements differ by seniority — for that, we need labels that aren't derived from the very signals we're analyzing.

**Seniority enum mapping:**
- junior / intern / new grad / entry-level -> entry
- associate / I / 1 -> associate
- senior / sr / II / 2 / staff / principal / lead -> mid-senior
- director / VP / head of -> director
- No clear signal -> unknown

YOE mentions do NOT drive the enum. An "entry-level" posting asking for "3+ years" is still classified as entry, and separately flagged as potentially inflated in ghost job assessment.

### Three seniority ablations

Preserve all three seniority signals for analysis:

| Column | Source | Primary use |
|---|---|---|
| `seniority_llm` | LLM classification (new, primary) | Main analysis variable |
| `seniority_imputed` | Rule-based classifier (existing) | Ablation baseline |
| `seniority_native` | Canonically mapped native/source-provided label | Cross-validation |
| `seniority_raw` | Original source label before mapping | Mapping audit / refinement |

### Boilerplate removal — verbatim constraint

The LLM must return exact substrings of the original description. After receiving the response, programmatically verify:

1. Split the LLM's `description_core` into sentences.
2. Check that each sentence appears verbatim in the original `description` (allow minor whitespace differences).
3. If >10% of sentences fail the verbatim check, fall back to the rule-based `description_core` for that posting.
4. Log the fallback rate.

**Two boilerplate columns for ablation:**

| Column | Source |
|---|---|
| `description_core` | Rule-based removal (existing, Stage 3) |
| `description_core_llm` | LLM-based removal (new, Stage 10) |

Analysis will run on both and compare results.

### Improving the rule-based seniority classifier

Apply these improvements from the LLM validation report (implemented in Stage 5 regardless of LLM augmentation):

1. **YOE cross-check column:** Regex for "X+ years", "X-Y years experience" patterns. Extract the minimum YOE mentioned. Compare against imputed seniority level and flag contradictions (e.g., entry-level title + 5+ YOE requirement). This feeds ghost job detection, not seniority assignment.

2. **Entry-level strict filter:** For entry-level-specific analysis, filter to postings where the title contains explicit junior signals: "Junior", "Entry", "New Grad", "I" (as a level), "Intern", "Associate". This reduces noise from mislabeled postings.

---

## Stage 11: LLM response integration

### Caching

- **Cache key:** `sha256(description_text)`
- **Cache storage:** SQLite database at `preprocessing/cache/llm_responses.db`
  - Schema: `(hash TEXT PRIMARY KEY, model TEXT, prompt_version TEXT, response_json TEXT, timestamp TEXT, tokens_used INTEGER)`
- **On re-run:** Check cache first. Only call LLM for uncached descriptions.
- **Cache invalidation:** If the prompt changes (tracked by `prompt_version`), re-run all cached entries with the new prompt. Keep old entries for comparison.

### Robustness

- **Rate limiting:** Respect API limits. Configurable delay between calls (default: 0.5s for Codex, 1.0s for Claude).
- **Retry:** Exponential backoff on transient errors (rate limits, timeouts). Max 3 retries per call.
- **Parse validation:** Every response must parse as valid JSON with exactly the four expected fields and valid enum values. Malformed responses are logged to `preprocessing/logs/llm_errors.jsonl` and excluded from the final data. The rule-based values remain for these postings.
- **Profiling run:** Before the full batch, run the first 100 calls and report:
  - Mean/P95 latency per call
  - Error rate
  - Mean tokens per request (input + output)
  - Estimated total cost for the full batch
  - Any systematic parse failures
  - Review 10 random responses manually for quality

### Memory constraint

31GB RAM limit. Continue using pyarrow chunked I/O:
- Process LLM calls in batches of 1,000 descriptions
- Write results incrementally to `preprocessing/intermediate/llm_responses.parquet`
- Final merge with `stage8_final.parquet` is a streaming join on row index

### Integration into unified.parquet

After all LLM calls complete, merge LLM-derived columns into the final output:

```python
# For each row in stage8_final.parquet:
# If LLM response exists for this description_hash:
row["swe_classification_llm"] = response["swe_classification"]
row["seniority_llm"] = response["seniority"]
row["description_core_llm"] = response["description_core"]
row["ghost_assessment_llm"] = response["ghost_assessment"]
# Else: these columns remain null (rule-based values are always present)
```

---

## Stage 12: Three-way validation

### Goal

For each of the four tasks, compare three classifiers to pick the best approach, identify biases, and quantify agreement.

### Classifiers compared

| | Classifier A | Classifier B | Classifier C |
|---|---|---|---|
| SWE classification | Rule-based (regex + embedding) | GPT-5.4 mini | GPT-5.4 (full) |
| Seniority | Rule-based (multi-signal) | GPT-5.4 mini | GPT-5.4 (full) |
| Boilerplate | Rule-based (section headers) | GPT-5.4 mini | GPT-5.4 (full) |
| Ghost job | Rule-based (heuristic flags) | GPT-5.4 mini | GPT-5.4 (full) |

### Sample design

100-200 postings per task, stratified by:
- **Source dataset:** Kaggle arshkon, Kaggle asaniczka, scraped LinkedIn, scraped Indeed
- **Ambiguity level:** Oversample from the ambiguous zone (e.g., embedding similarity 0.50-0.85 for SWE, seniority_source = "unknown" for seniority)
- **Classification disagreement:** Oversample cases where rule-based and GPT-5.4 mini disagree

### Report format

For each task:

1. **Agreement matrix:** 3x3 (or larger) contingency table showing pairwise agreement counts.
2. **Cohen's kappa:** Between each pair (A-B, A-C, B-C). Kappa > 0.80 = strong agreement; 0.60-0.80 = moderate; < 0.60 = weak.
3. **Categorized disagreements:** For every disagreement, categorize the error type:
   - Definitional disagreement (different but defensible interpretations)
   - Genuine misclassification by one classifier
   - Edge case that needs a policy decision
4. **Disagreement examples:** 5-10 representative examples per category with full context.

### Decision criteria

- If GPT-5.4 mini and GPT-5.4 (full) agree > 95% of the time, use mini for production (cheaper).
- If one LLM systematically outperforms on a specific task, use that LLM for that task.
- Where rule-based and LLM agree, the classification is high-confidence.
- Where they disagree, the LLM response is preferred for seniority and ghost job (where rules are weakest). Rule-based is preferred for SWE classification (where regex has 95%+ precision on unambiguous cases).

### Using LLM findings to improve rules

After the validation:
- Extract new regex patterns from LLM disagreements (new `SWE_EXCLUDE` terms, new aggregator names)
- Add new entries to the `AGGREGATORS` list based on LLM-identified staffing companies
- Document every rule change with the LLM evidence that motivated it
- Re-run the rule-based pipeline with updated rules and measure improvement

---

## Output specification

### Primary output: `data/unified.parquet`

One row per unique posting across all sources.

**Core columns:**
- `uid`, `source`, `source_platform`, `site`
- `title`, `title_normalized`, `company_name`, `company_name_normalized`
- `location`, `location_normalized`, `date_posted`, `description`, `description_raw`, `description_length`
- `seniority_raw`, `seniority_native`, `company_industry`, `company_size`, `company_size_raw`, `company_size_category`
- `search_query`, `query_tier`, `search_metro_id`, `search_metro_name`, `search_metro_region`, `search_location`
- `scrape_date` (first seen, for scraped data; null for Kaggle)

**Rule-based classification columns:**
- `is_swe`, `is_swe_adjacent`, `is_control`, `swe_confidence`, `swe_classification_tier`
- `seniority_imputed`, `seniority_source`, `seniority_confidence`, `seniority_final`, `seniority_3level`
- `seniority_cross_check`
- `is_aggregator`, `real_employer`, `is_multi_location`
- `description_core` (rule-based boilerplate removal)
- `posting_age_days`, `period`
- `ghost_job_risk`
- `description_quality_flag`, `dedup_method`, `preprocessing_version`
- `is_english`, `description_hash`

**LLM-derived columns:**

| Column | Type | Values | Source |
|---|---|---|---|
| `swe_classification_llm` | string | SWE / SWE_ADJACENT / NOT_SWE / null | LLM Stage 10 |
| `seniority_llm` | string | entry / associate / mid-senior / director / unknown / null | LLM Stage 10 |
| `description_core_llm` | string | Verbatim-extracted core content / null | LLM Stage 10 |
| `ghost_assessment_llm` | string | realistic / inflated / ghost_likely / null | LLM Stage 10 |
| `llm_model` | string | Model name that produced the classification / null | LLM Stage 10 |
| `llm_prompt_version` | string | Prompt version hash / null | LLM Stage 10 |
| `yoe_extracted` | float | Minimum years-of-experience mentioned in description / null | Stage 5 improvement |
| `yoe_seniority_contradiction` | bool | True if YOE contradicts seniority label | Stage 5 improvement |

**Null values in LLM columns** mean that posting was not sent to the LLM (pre-filtered out with high-confidence rule-based classification) or the LLM call failed. In either case, the rule-based columns provide the classification.

**Asaniczka-specific column:**
- `asaniczka_skills` — skills from `job_skills.csv` join (null for other sources)

### Secondary output: `data/unified_observations.parquet`

One row per posting per scrape_date. Columns:
- `uid`, `scrape_date`, `source`, `source_platform`
- All other columns from `unified.parquet` (denormalized for query convenience)

For Kaggle sources, each posting has one observation row (scrape_date = null). For scraped data, a posting that appears on 5 different scrape dates has 5 rows. This supports:
- Posting duration analysis (how long postings stay active)
- Daily composition snapshots
- Sensitivity analysis: canonical postings vs. daily observations

Do not replace this with an array-of-dates column inside `unified.parquet`. The daily panel is a different unit of observation and is much easier to query, aggregate, and join when each appearance is a row. If storage becomes a concern, the compact alternative is a thin `uid`/`scrape_date` appearances table, not a list-valued column in the canonical posting file.

### Analysis variable selection guide

| Analysis need | Primary variable | Ablation variables |
|---|---|---|
| Is this a SWE role? | `is_swe` (rule-based, high precision) | `swe_classification_llm` |
| What seniority level? | `seniority_llm` (where available), else `seniority_imputed` | `seniority_native`, `seniority_imputed` |
| Clean description text | `description_core_llm` (where available), else `description_core` | `description_core`, `description` (full) |
| Is this a ghost job? | `ghost_assessment_llm` (where available), else `ghost_job_risk` | `ghost_job_risk` |

### Quality report: `data/quality_report.json`

```json
{
  "pipeline_version": "3.0",
  "run_date": "2026-03-21",
  "funnel": {
    "arshkon_raw": "...",
    "arshkon_swe": "...",
    "asaniczka_raw": "...",
    "asaniczka_us_filtered": "...",
    "asaniczka_swe": "...",
    "asaniczka_description_join_rate": "...",
    "scraped_raw": "...",
    "scraped_linkedin": "...",
    "scraped_indeed": "...",
    "scraped_after_crossday_dedup": "...",
    "final_swe": "...",
    "final_control": "..."
  },
  "llm_stats": {
    "total_unique_descriptions": "...",
    "descriptions_sent_to_llm": "...",
    "descriptions_pre_filtered": "...",
    "llm_calls_made": "...",
    "cache_hits": "...",
    "parse_failures": "...",
    "verbatim_check_failures": "...",
    "mean_latency_ms": "...",
    "total_tokens": "...",
    "estimated_cost": "..."
  },
  "classification_rates": {
    "swe_rate_rules": "...",
    "swe_rate_llm": "...",
    "seniority_unknown_rate_rules": "...",
    "seniority_unknown_rate_llm": "..."
  },
  "validation": {
    "swe_kappa_rules_vs_mini": "...",
    "swe_kappa_mini_vs_full": "...",
    "seniority_kappa_rules_vs_mini": "...",
    "seniority_kappa_mini_vs_full": "..."
  },
  "boilerplate_stats": {
    "median_chars_removed_rules": "...",
    "median_chars_removed_llm": "...",
    "verbatim_check_pass_rate": "..."
  },
  "source_specific": {
    "arshkon_seniority_distribution": "...",
    "asaniczka_seniority_distribution": "...",
    "scraped_linkedin_seniority_distribution": "...",
    "scraped_indeed_seniority_null_rate": "..."
  }
}
```

### Preprocessing log: `data/preprocessing_log.txt`

Human-readable log of every stage: rows in, rows out, rows flagged, decisions made. This feeds directly into the methodology section writeup.

---

## Sensitivity analyses

These are baked into the pipeline design, not afterthoughts:

| Sensitivity check | What it tests | How |
|---|---|---|
| **LinkedIn-only estimates** | Platform composition artifact | Filter to `source_platform == "linkedin"` for all analyses |
| **LinkedIn + Indeed pooled** | Whether Indeed changes the story | Pool both platforms, control for `source_platform` |
| **Dedup sensitivity** | Whether dedup threshold matters | Run analyses at 0.60, 0.70, 0.80 cosine thresholds |
| **Canonical vs. daily observations** | Whether repost-weighting matters | Compare `unified.parquet` results to `unified_observations.parquet` results |
| **Metro-balanced subsamples** | Whether metro composition drives results | Resample to equal metro representation |
| **Exclude aggregators** | Whether staffing firms distort signals | Filter to `is_aggregator == False` |
| **Arshkon-only historical baseline** | Whether asaniczka's missing entry-level biases trends | Run RQ1 junior-share analysis using only arshkon as historical baseline |
| **Rule-based vs. LLM classification** | Whether classification method drives results | Run key analyses with both `seniority_imputed` and `seniority_llm` |

---

## Implementation order

```
Phase 1: Rule-based pipeline rebuild
  1. Stage 1 rewrite (3 source schemas)           NEW for v3
     - 1a: Arshkon ingest + companion joins
     - 1b: Asaniczka ingest + description/skills joins
     - 1c: Scraped ingest (all current-format files)
     - 1d: Schema unification
     |
  2. Stages 2-8 (re-run on new Stage 1 output)    Existing code, new input
     - Verify all stages handle new uid format
     - Verify all stages handle null fields from asaniczka
     |
  Output: intermediate/stage8_final.parquet

Phase 2: LLM augmentation
  3. Stage 9: Pre-filtering script                 depends on: stage8_final.parquet
     - Compute description hashes
     - Apply skip conditions
     - Output: candidate list with description hashes
     |
  4. Profiling run (100 calls)                     depends on: Stage 9
     - Test both Codex and Claude CLI
     - Measure latency, cost, parse reliability
     - Manual review of 10 responses
     - Decision: which model to use
     |
  5. Stage 10: Full LLM batch                      depends on: profiling run
     - Run all candidate descriptions through chosen model
     - Cache responses in SQLite
     - Log errors separately
     |
  6. Stage 11: Integration                         depends on: Stage 10
     - Merge LLM columns into stage8_final.parquet
     - Run verbatim checks on boilerplate removal
     - Produce unified.parquet + unified_observations.parquet
     |
  7. Stage 12: Three-way validation                depends on: Stage 11
     - Sample stratified postings
     - Run GPT-5.4 (full) on the validation sample
     - Compute agreement matrices and kappa
     - Produce validation report

Phase 3: Rule improvement
  8. Extract patterns from LLM disagreements
  9. Update SWE_EXCLUDE, AGGREGATORS, boilerplate regexes
  10. Re-run Stages 1-8 with updated rules
  11. Re-run Stages 9-12 to measure improvement
```

**Critical path:** The Stage 1 rewrite (step 1) is the gate for everything else. The profiling run (step 4) is the gate for the full LLM batch.

---

## Review protocol: 3-tier validation

Every quality check uses a 3-tier approach.

```
Tier 1: Rule-based validation (automated, full dataset)
  | flagged items
Tier 2: LLM review (automated, hundreds to thousands of items)
  | items LLM flags as problematic or uncertain
Tier 3: Human review (small sample, only what matters)
```

**Tier 1:** Deterministic checks on every row. Example: description < 50 chars, title-seniority contradictions.

**Tier 2:** LLM reviews flagged items. With the v3 pipeline, the Stage 10 LLM call itself serves as Tier 2 for classification tasks. Additional Tier 2 reviews for edge cases use the same CLI interface.

**Tier 3:** Human reviews items where Tier 2 was uncertain or flagged issues. Also reviews a random sample of Tier 2's "clean" outputs (10-20 items) to validate LLM judgment. If LLM error rate on the random sample exceeds 10%, expand human review scope.

### Spot-check table

| Check | Tier 1 | Tier 2 (LLM) | Tier 3 (human) |
|---|---|---|---|
| **SWE classification** | Regex match/exclude | Stage 10 LLM classification on full description | 50 disagreements + 20 random agreements |
| **Seniority** | Title keyword match + YOE extraction | Stage 10 LLM explicit-signals-only classification | 50 disagreements + 20 random agreements |
| **Boilerplate** | Section header regex; paragraph-level filtering | Stage 10 LLM full-text boilerplate identification | 30 verbatim-check failures + 10 random passes |
| **Ghost job** | Entry-level + 5yr+ experience heuristic | Stage 10 LLM ghost assessment | 30 rule-LLM disagreements |
| **Aggregator** | Company name in AGGREGATORS list | 100 aggregator postings — LLM extracts real employer | 20 verification items |

### Gold-standard annotation

Sample 500 postings stratified by source, predicted seniority, and classification confidence. LLM pre-labels all four tasks. Human corrects LLM labels. Compute inter-rater kappa. If kappa > 0.80, LLM is reliable as a second annotator. If kappa < 0.60, expand human review.

**Warning (Ashwin et al. 2025):** LLM coding errors are NOT random — they correlate with text characteristics. Always validate against human labels before trusting LLM annotations at scale.

---

## Resolved decisions

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | **Historical baseline sources** | Use both Kaggle datasets. Arshkon is the primary historical baseline for entry-level analysis (RQ1). Asaniczka provides mid-senior/associate content for RQ2. | Asaniczka has zero entry-level postings — cannot support junior-share analysis. Arshkon has full seniority distribution. |
| 2 | **Indeed inclusion** | LinkedIn-only for primary analysis. Indeed for sensitivity only. | Both Kaggle sources are LinkedIn-only. LinkedIn-only analysis provides cleanest cross-period comparison. |
| 3 | **Description length discrepancy** | Investigate through LLM-assisted matched-pair comparison + automated checks. Do not draw length-based conclusions until resolved. | 55% gap could be scraping artifact. |
| 4 | **Seniority classification** | Three ablations: LLM-classified (primary), rule-based imputed, LinkedIn native. | Addresses v1 validation failure (32% agreement). LLM uses explicit signals only. |
| 5 | **SWE classification approach** | Rule-based remains primary (high precision on unambiguous cases). LLM adds coverage for ambiguous cases. | v1 validation showed 64% raw agreement but 83% when collapsing SWE+adjacent. The boundary is the problem, not the core classification. |
| 6 | **Kaggle arshkon companion files** | Join during ingest (Stage 1a). | Confirmed joinable. 99.3% industry coverage. |
| 7 | **Multi-location postings** | Keep all variants as default. Sensitivity-test collapsing. | Literature standard (Hershbein & Kahn 2018). |
| 8 | **Aggregator postings** | Keep and flag. Sensitivity-test excluding. | Present across all sources. |
| 9 | **DataAnnotation dominance** | Flag and sensitivity-test excluding. | 168 postings = 5.4% of Kaggle SWE. May be crowdwork. |
| 10 | **Boilerplate removal** | Rule-based + LLM as dual columns for ablation. | v1 regex works for tail-end boilerplate but misses front-matter. LLM handles both. |
| 11 | **Near-dedup threshold** | Literature standard (0.70 cosine). | Abdelaal et al. 2024, F1=0.94. |
| 12 | **Embedding model** | Dual-model: JobBERT-v2 for titles, general-purpose for descriptions. | JobBERT-v2 has 64-token limit. |
| 13 | **LLM model for classification** | GPT-5.4 mini primary, GPT-5.4 full for validation. Claude Haiku as fallback. | Cost-effective for thousands of calls. Validation uses the stronger model. |
| 14 | **LLM boilerplate constraint** | Verbatim extraction only, no paraphrasing. Programmatic check. | Ensures the LLM doesn't introduce artifacts into the text we analyze. |
| 15 | **Seniority from YOE** | YOE is extracted but NOT used for seniority assignment. Used only for contradiction flagging and ghost job detection. | Companies inflate YOE requirements. Using YOE to assign seniority would bake that inflation into the labels. |
| 16 | **Old scraped data (Mar 5-18)** | Skip incompatible legacy scraped files, but do not hard-code a month/date ceiling for current-format files. | Old format lacked search metadata columns. Current-format scraped data should be loaded for any available date once it matches the 41-column schema. |
| 17 | **Schema unification ownership** | Pipeline Stage 1 handles all three schemas independently. Does NOT rely on `scraper/harmonize.py`. | Each source has a different schema. Centralizing in the pipeline ensures reproducibility and makes schema differences explicit. |
| 18 | **Asaniczka description join** | Left join `job_summary.csv` on `job_link`. Postings without descriptions (3.8%) remain with null description. Preserve `description_raw` separately from the normalized working `description` column. | Dropping them would bias against postings that lost their descriptions. Keeping a raw text column preserves source fidelity while later stages operate on a normalized working copy. |
| 19 | **Two output files** | `unified.parquet` (canonical) + `unified_observations.parquet` (daily panel). | Canonical postings are the primary unit of analysis. Daily observations support duration analysis and sensitivity checks per the research design, and row-wise observations are much easier to query than list-valued appearance columns. |
| 20 | **Asaniczka ID format** | Hash `job_link` URL to create a fixed-length ID: `"asaniczka_" + sha256(job_link)[:16]`. | Raw URLs are too long and unwieldy as IDs. Hash preserves uniqueness. |
| 21 | **Raw vs mapped seniority** | Preserve the original source label in `seniority_raw` and store the canonical mapping in `seniority_native`. Unmapped values should stay null in `seniority_native`. | This keeps Stage 1 faithful to source data while preserving a normalized cross-source field for downstream deduplication, cross-checking, and validation. |
