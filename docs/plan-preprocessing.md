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
  |     +-- derives `company_name_effective` = `real_employer` else `company_name`
  +-- Stage 3: Rule-Based Boilerplate Removal        [UNCHANGED from v1]
  +-- Stage 4: Company Canonicalization + Dedup      [UPDATED key-first dedup]
  |     +-- builds `company_name_canonical` from `company_name_effective`
  +-- Stage 5: Rule-Based Classification             [UNCHANGED from v1]
  +-- Stage 6: Field Normalization                   [row-preserving]
  +-- Stage 7: Temporal Alignment                    [row-preserving]
  +-- Stage 8: Quality Flags                         [row-preserving]
  |
  v
intermediate/stage8_final.parquet (rule-based pipeline output)
  |
  +-- Stage 9: LLM Extraction + Cleaned Text        [control cohort + extraction + row-preserving integration]
  +-- Stage 10: LLM Classification + Final Merge    [classification + row-preserving posting artifact]
  +-- Stage 12: Three-Way Validation                 [UNCHANGED from v2]
  |
  v
data/unified.parquet          (canonical postings: one row per unique posting)
data/unified_observations.parquet  (daily panel: one row per posting per scrape_date)
  + data/quality_report.json
  + data/preprocessing_log.txt
```

**Design principle:** The rule-based pipeline (Stages 1-8) runs first and produces the baseline corpus plus fallback labels. The LLM layer (Stages 9-12) is a separate augmentation layer that can be validated independently; the baseline output remains usable without it, but the intended production analysis dataset adds LLM-derived columns alongside rule-based columns once the LLM stages are trusted. This means:
- Stages 1-8 remain necessary because they define the baseline corpus, cache keys, and fallback labels
- The default production pipeline does not stop at Stage 8; it continues through Stages 9-10 before writing `unified.parquet`
- LLM outputs are additive — they do not erase the rule-based columns, which remain for ablations and failure fallback
- If an LLM call fails for a posting, the rule-based values remain and the LLM columns stay null

**Two output files:**
- `unified.parquet`: One row per unique posting. For Kaggle sources, each posting appears once. For scraped data, each unique `id` appears once with its first-seen metadata. This is the primary analysis file.
- `unified_observations.parquet`: One row per posting per scrape_date. Only meaningful for scraped data (Kaggle sources appear once). Tracks when postings appear/disappear from search results. Supports posting-duration analysis and daily-panel sensitivity checks.

Each stage produces logged counts (rows in, rows out, rows flagged) for the methodology section. Reader-facing parquet and text outputs should be written to sibling temp files and atomically promoted only after the write completes.

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
| 2 | `stage2_aggregators.py` | Aggregator flagging, real employer extraction, `company_name_effective` derivation |
| 3 | `stage3_boilerplate.py` | Section-based boilerplate removal (regex) |
| 4 | `stage4_dedup.py` | Company canonicalization plus posting-level dedup for canonical postings: exact IDs, exact opening duplicates, same-location near-duplicates, multi-location flagging |
| 5 | `stage5_classification.py` | 3-tier occupation classification (SWE / SWE-adjacent / control; regex + LLM lookup + embedding), multi-signal seniority |
| 6 | `stage678_normalize_temporal_flags.py` | Location parsing, remote inference into `is_remote_inferred`, metro enrichment |
| 7 | (same file) | Period assignment, posting age computation, scrape week |
| 8 | (same file) | Date/language quality flags, description hash, ghost-job heuristics, description quality flags, provenance |

**Rule-based columns produced (preserved as ablation baselines):**
- `is_swe`, `is_swe_adjacent`, `swe_confidence`, `swe_classification_tier`
- `seniority_imputed`, `seniority_source`, `seniority_confidence`, `seniority_final`, `seniority_3level`
- `description_core` (boilerplate-stripped)
- `ghost_job_risk`

### Stage 4: Deduplication boundary

Stage 4 is the posting-level dedup boundary for the canonical dataset. It owns:
- canonicalization of `company_name_effective` into `company_name_canonical` for opening-level matching
- exact duplicate removal by unified posting ID
- exact opening dedup using normalized company, title, location, and matching description support
- same-location near-duplicate resolution as a narrower fallback when description support agrees
- `is_multi_location` flagging for postings that share the same canonical opening across multiple locations

Stage 4 consumes the Stage 2 `company_name_effective` field. Stage 2 owns aggregator detection and `real_employer` extraction; Stage 4 owns canonicalizing that effective employer label for dedup.

Stage 4 does **not** own daily observation dedup beyond the Stage 1 scraped canonicalization boundary, aggregator detection, occupation classification, or later analytical sample definition.

Current implementation note: Stage 4 now uses a key-first, description-supported dedup design rather than the older cosine-threshold framing. Description is a supporting signal inside company/title/location candidate sets, not a standalone dedup key.

---

### Stage 6: Field normalization contract

Stage 6 is row-preserving. It consumes Stage 5 rows one-for-one and must not add,
drop, merge, or reorder postings.

Stage 6 owns:
- parsing `location` into `city_extracted`, `state_normalized`, and `country_extracted`
- inferring `is_remote_inferred` from remote markers in the location string
- deriving posting-location `metro_area` plus `metro_source` and `metro_confidence`

Stage 6 input assumptions:
- `location` is the raw normalized location string from Stage 1
- `is_remote` is the Stage 1 normalized source-provided remote flag
- `search_metro_*` remains query metadata from Stage 1, not posting geography

Stage 6 output semantics:
- `is_remote` remains the source-provided normalized flag from Stage 1
- `is_remote_inferred` is a separate boolean set from location-text cues only
- `metro_area` is a posting-location metro label aligned to the study metro frame
  when Stage 6 can infer one
- `metro_source` records how the metro was assigned (`search_metro`,
  `manual_alias`, `city_state_lookup`, `city_state_reference`, or `unresolved`)
- `metro_confidence` is a coarse confidence tier for the metro assignment
- location parsing is best-effort row-level enrichment, not a dedup or sample-definition step

If `preprocessing/reference/metro_city_state_lookup.parquet` is present, Stage 6
uses it as an offline fallback for unresolved US city/state pairs. Regenerate the
reference files with:

```bash
./.venv/bin/python preprocessing/scripts/build_metro_city_state_reference.py
```

Stage 6 does not own:
- rewriting `is_remote`
- canonicalizing company/title/location for dedup
- occupation classification or analytical sample selection

### Stage 7: Temporal alignment contract

Stage 7 is row-preserving. It adds time-derived fields but does not filter rows or
change the unit of observation.

Stage 7 owns:
- `period`
- `posting_age_days`
- `scrape_week`

Stage 7 output semantics:
- `period` is source-driven for historical Kaggle rows and derived from `scrape_date`
  for scraped rows
- `posting_age_days` is populated here, not in Stage 1; it is mainly meaningful for
  scraped rows that have both `scrape_date` and `date_posted`
- `scrape_week` is the ISO week of `scrape_date`

Stage 7 does not own:
- date-quality judgments beyond producing time-derived fields
- observation expansion into the daily panel

### Stage 8: Quality, utility, and provenance contract

Stage 8 is row-preserving. It adds quality flags, LLM-support utility fields, and
row-level provenance markers.

Stage 8 owns:
- `date_flag`
- `is_english`
- `description_hash`
- `ghost_job_risk`
- `description_quality_flag`
- `preprocessing_version`, `dedup_method`, `boilerplate_removed`

Stage 8 output semantics:
- `date_flag` is a row-level validation summary for `scrape_date` and `date_posted`
- `description_hash` is the stable raw-description lineage hash; LLM caching now uses task-specific `input_hash` values
- `ghost_job_risk` is a rule-based heuristic derived from Stage 5 fields:
  canonical `seniority_final` (entry-like only), `yoe_extracted`, and
  `yoe_seniority_contradiction`
- `ghost_job_risk` values are `low`, `medium`, or `high`

Stage 8 does not own:
- row filtering or analytical exclusion decisions
- recomputing classification labels
- redefining canonical postings or daily observations

---

## Stage 9: Cohort Selection, Extraction, and Cleaned-Text Integration

### Goal

Define the LLM analysis universe, choose the deterministic control cohort, run extraction only, validate extraction responses, and materialize the posting-level cleaned-description contract that Stage 10 will classify against.

### Fixed default analysis universe

The default LLM routing universe is:
- `source_platform == "linkedin"`
- `is_english == True`
- raw `description` is present

Within that universe, the default extraction corpus includes:
- all Stage 5 `is_swe == True` rows
- all Stage 5 `is_swe_adjacent == True` rows
- selected control-cohort rows only

Rows outside that universe stay in the dataset, but they are not part of the default LLM routing path.

This explicitly excludes:
- non-English rows
- rows with null/empty raw `description`
- Indeed rows in the default production path
- control rows that were not selected into the deterministic control cohort
- unresolved Stage 5 rows as a default LLM-recovery target

Unresolved-row recovery is moved to a separate audit/sensitivity workflow, not the core pipeline.

### Short-description hard skip

If raw `description` is present but has fewer than 15 words:
- do not send the row to extraction
- set `description_core_llm = ''`
- record the skip reason on the posting row
- let Stage 10 inherit that exclusion when it builds classification candidates

### Control cohort selection

Controls are selected before any extraction calls are planned.

Recommended algorithm:
1. Build the eligible control pool at the extraction call unit using the same base text rules as extraction.
2. Create `control_bucket`:
   - scraped rows: `scraped|YYYY-WW`
   - historical rows: `source|period`
3. Match the total selected controls to eligible SWE counts by bucket, redistributing shortfall to buckets with spare capacity.
4. Rank controls deterministically inside each bucket using a stable pseudo-random score derived from `control_bucket` and `extraction_input_hash`.
5. Select the lowest-score controls up to each bucket target.

The selected cohort must be stable across reruns and monotone as the eligible SWE corpus grows.

### Required Stage 9 outputs

- `preprocessing/intermediate/stage9_control_cohort.parquet`
- `preprocessing/intermediate/stage9_llm_extraction_candidates.parquet`
- `preprocessing/intermediate/stage9_llm_extraction_results.parquet`
- `preprocessing/intermediate/stage9_llm_cleaned.parquet`

`stage9_llm_cleaned.parquet` is row-preserving and posting-level. It should carry the stable cleaned-text contract used later by Stage 10, especially:
- `description_core_llm`
- `selected_for_control_cohort`
- short-description skip state needed by Stage 10 classification routing

## Stage 10: Classification Routing, Execution, and Final Integration

### Architecture

Stage 10 consumes `stage9_llm_cleaned.parquet`, builds the classification candidate set, executes classification per unique `classification_input_hash`, and writes the canonical posting-level LLM artifact.

The classification universe is:
- `source_platform == "linkedin"`
- `is_english == True`
- not excluded by the Stage 9 short-description rule
- technical corpus plus `selected_for_control_cohort == True`

Default classification skip logic preserves the current production behavior. Skip LLM classification when all of the following hold:
- `swe_classification_tier` is one of `regex`, `embedding_high`, or `title_lookup_llm`
- `seniority_source` starts with `title_`
- `ghost_job_risk == "low"`

The critical redesign change is the classifier input order:
- `description_core_llm` when non-empty
- else `description_core`
- else raw `description`

Stage 10 produces:
- `preprocessing/intermediate/stage10_llm_classification_results.parquet`
- `preprocessing/intermediate/stage10_llm_integrated.parquet`

This makes Stage 10 the canonical posting-level artifact. Stage 11 is compatibility-only if it exists at all.

### Model selection

- **Codex path:** always pin to GPT-5.4 mini via:
  ```bash
  codex exec --full-auto --config model=gpt-5.4-mini "<prompt>" --skip-git-repo-check
  ```
- **Claude path:** always pin to Haiku via:
  ```bash
  claude -p "<prompt>" --model haiku --output-format json
  ```
- **Validation model:** GPT-5.4 (full) for the three-way comparison (Stage 12).

Enabled engines are a runtime choice:
- `--engines codex,claude` enables both engines
- `--engines claude` means Claude-only
- `--engine-tiers codex=full,claude=non_intrusive` assigns Codex to full-utilization mode and Claude to conservative slot-budget mode
- When multiple enabled engines are currently available, tasks are assigned across them from the shared queue
- Once a task is assigned to an engine, retries stay on that same engine; the runtime does not fall back to a different model for that task

### Prompt design

Stage 10 maintains the classification prompt family only:
- SWE classification
- seniority
- ghost assessment
- YOE extraction

```
Classification prompt:
- `swe_classification`
- `seniority`
- `ghost_assessment`
- `yoe_min_years`

The classification prompt keeps seniority and YOE explicitly separated:
- `seniority` uses only title/description seniority language and must not be
  inferred from YOE.
- `yoe_min_years` is a cross-check field. It extracts the binding YOE floor for
  the role from explicit YOE mentions only and may return `null`.
```

### Seniority classification — design rationale

This is the most important design change from v1. The LLM validation showed that inferring seniority from responsibilities produces garbage (87/100 classified as mid-senior when given truncated descriptions). The new approach:

**What the LLM does:** Looks for explicit seniority signals only — title keywords, level codes, and explicit role-label language about the posting itself. Maps to the enum. Defaults to "unknown" when ambiguous.

**What the LLM does NOT do:** Infer seniority from responsibilities, tech stack complexity, team size, YOE requirements, or company reputation.

**Why:** The research goal is to understand how companies label and frame their roles at different seniority levels. We need clean seniority labels that reflect the company's intent, not reverse-engineered "true" seniority. Later analysis examines how skills/requirements differ by seniority — for that, we need labels that aren't derived from the very signals we're analyzing.

**Seniority enum mapping:**
- junior / intern / new grad / entry-level -> entry
- associate / I / 1 -> associate
- senior / sr / II / 2 / staff / principal / lead -> mid-senior
- director / VP / head of -> director
- No clear signal -> unknown

YOE mentions do NOT drive the enum. An "entry-level" posting asking for "3+ years" is still classified as entry, and separately flagged as potentially inflated in ghost job assessment.

### LLM YOE cross-check field

The classification prompt also returns `yoe_min_years`, a nullable LLM
cross-check column.

Design intent:
- This is not a seniority input. It exists only for ablation / cross-validation
  against the Stage 5 rule-based YOE extractor.
- The LLM returns the binding YOE floor for the role using explicit YOE
  mentions only.
- If the posting gives multiple acceptable qualification paths, return the
  lowest path-level YOE floor.
- Tool/framework/domain-specific YOE counts if it is the only YOE mention on a
  path, or if it is higher than the general-role YOE on that path.
- If no relevant explicit YOE exists, return `null`.

### Three seniority ablations

Preserve all three seniority signals for analysis:

| Column | Source | Primary use |
|---|---|---|
| `seniority_llm` | LLM classification (new, primary) | Main analysis variable |
| `seniority_imputed` | Rule-based classifier (existing) | Ablation baseline |
| `seniority_native` | Canonically mapped native/source-provided label | Cross-validation |
| `seniority_raw` | Original source label before mapping | Mapping audit / refinement |

### Boilerplate removal — cleaned-text contract

Stage 9 returns extraction responses, validates them locally, and reconstructs `description_core_llm` on the posting row before Stage 10 classification begins.

Validation happens locally:

1. Confirm every returned ID exists and is unique.
2. Confirm `uncertain_unit_ids` is a subset of valid IDs.
3. If `task_status == "cannot_complete"` or reconstruction is empty / obviously malformed, fall back to rule-based `description_core`.
4. Log the fallback rate and the rate of `cannot_complete` responses.

**Two boilerplate columns for ablation:**

| Column | Source |
|---|---|
| `description_core` | Rule-based removal (existing, Stage 3) |
| `description_core_llm` | LLM-based removal reconstructed locally from Stage 9 extraction output |

Analysis will run on both and compare results.

### Improving the rule-based seniority classifier

Apply these improvements from the LLM validation report (implemented in Stage 5 regardless of LLM augmentation):

1. **YOE cross-check columns:** Extract YOE from raw `description`, not `description_core`, using layered candidate rules rather than one narrow regex. Segment raw text into lightweight section-aware clauses, tag candidates with richer role/section metadata, resolve a posting-level primary YOE into `yoe_extracted`, preserve lower/upper bounds in `yoe_min_extracted` and `yoe_max_extracted`, store `yoe_match_count`, `yoe_resolution_rule`, and `yoe_all_mentions_json` for auditability, and then compare the resolved primary YOE against canonical `seniority_final` to flag contradictions. `seniority_imputed` remains available for ablation and QA, but `yoe_seniority_contradiction` follows the canonical resolved seniority used downstream. This feeds ghost job detection, not seniority assignment. Fixed YOE regression cases now live in `tests/test_stage5_yoe_extractor.py`.

2. **Entry-level strict filter:** For entry-level-specific analysis, filter to postings where the title contains explicit junior signals: "Junior", "Entry", "New Grad", "I" (as a level), "Intern", "Associate". This reduces noise from mislabeled postings.

---

## Stage 11: Compatibility Alias Only

### Caching

- **Cache key:** task-specific `input_hash`
- extraction: `sha256(title, company_name, raw description)`
- classification: `sha256(title, company_name, classifier input)`
- **Cache storage:** SQLite database at `preprocessing/cache/llm_responses.db`
  - Schema: one row per `(input_hash, task_name, prompt_version)` with `model`, `response_json`, `timestamp`, and `tokens_used`
- **On re-run:** Check cache first. Only call LLM for uncached descriptions.
- **Cache invalidation:** If the prompt changes (tracked by `prompt_version`), re-run all cached entries with the new prompt. Keep old entries for comparison.
- **Commit behavior:** Successful task responses are committed immediately to SQLite, so reruns resume from cache even if Stage 10 was interrupted before writing its final parquet.

### Robustness

- **Retry:** Each subprocess call still retries internally up to 3 times on malformed output / timeout paths. If the task still fails, the outer engine runtime waits 1 minute and retries the same task on the same engine rather than falling back to another engine.
- **Quota handling:** Quota state is provider-scoped, not global. Full-utilization engines pause for `--quota-wait-hours` after quota / rate-limit failures. Non-intrusive engines follow midnight-aligned five-hour slots in the configured timezone: `00:00-05:00` runs until quota exhaustion, `05:00-10:00` allows 2,000 calls, and later slots allow 1,000 calls each. Quota hits on a non-intrusive engine pause that engine until the current slot ends.
- **Parse validation:** Every response must parse as valid task-specific JSON with valid enum values or valid extraction IDs/status. Malformed responses are logged to `preprocessing/logs/llm_errors.jsonl` and excluded from the final data. The rule-based values remain for these postings.
- **Profiling run:** Before the full batch, run a small stratified subset first, then the first 100 calls in profile mode, and report:
  - Mean/P95 latency per call
  - Error rate
  - Mean tokens per request (input + output)
  - Estimated total cost for the full batch
  - Any systematic parse failures
  - Review 10 random responses manually for quality
  - Include at least a few one-unit descriptions and a few multi-unit descriptions in the profile set

### Operational restart notes

For a production rerun from the current Stage 8 artifact:

1. Run Stage 9:
   ```bash
   /usr/bin/time -v ./.venv/bin/python preprocessing/scripts/stage9_llm_prefilter.py \
     --engines codex,claude \
     --engine-tiers codex=full,claude=non_intrusive \
     --quota-wait-hours 5 \
     --max-workers 30
   ```

2. Run Stage 10 directly with the same engine controls:
   ```bash
   /usr/bin/time -v ./.venv/bin/python preprocessing/scripts/stage10_llm_classify.py \
     --engines codex,claude \
     --engine-tiers codex=full,claude=non_intrusive \
     --quota-wait-hours 5 \
     --max-workers 30
   ```

3. If you keep the optional compatibility alias, treat it as transitional only. The architectural handoff is Stage 10, not Stage 11.

Operational behavior worth remembering:
- Successful Stage 10 task responses are committed to SQLite immediately after each task finishes.
- The durable checkpoint is `preprocessing/cache/llm_responses.db`, not the parquet outputs.
- If Stage 10 is interrupted, rerun the same Stage 10 command; completed tasks will be loaded from cache and skipped.
- Cache reuse keys on `(input_hash, task_name, prompt_version)`.

### Memory constraint

31GB RAM limit. Continue using pyarrow chunked I/O:
- Process cache lookups / output assembly in batches of 1,000 descriptions
- Use SQLite as the durable incremental checkpoint during Stage 10
- Write the Stage 10 results/integrated parquets after the cached/fresh task set is complete
- Any compatibility alias should be a row-preserving copy of the Stage 10 integrated output, not a new integration boundary

### Integration into unified.parquet

After Stage 10 completes, the posting-level integrated parquet is the direct input to final output generation:

```python
# For each row in stage10_llm_integrated.parquet:
row["swe_classification_llm"] = class_response["swe_classification"]
row["seniority_llm"] = class_response["seniority"]
row["ghost_assessment_llm"] = class_response["ghost_assessment"]
row["yoe_min_years_llm"] = class_response["yoe_min_years"]
# description_core_llm and extraction diagnostics were already integrated in Stage 9
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
- `title`, `title_normalized`, `company_name`, `company_name_effective`, `company_name_canonical`, `company_name_canonical_method`
- `company_name_normalized` (legacy ingest-level normalized copy; not the canonical dedup key)
- `location`, `work_type`, `is_remote`, `date_posted`, `description`, `description_raw`, `description_length`
- `seniority_raw`, `seniority_native`, `company_industry`, `company_size`, `company_size_raw`, `company_size_category`
- `search_query`, `query_tier`, `search_metro_id`, `search_metro_name`, `search_metro_region`, `search_location`
- `scrape_date` (first seen, for scraped data; null for Kaggle)

**Rule-based classification columns:**
- `is_swe`, `is_swe_adjacent`, `is_control`, `swe_confidence`, `swe_classification_tier`
- `seniority_imputed`, `seniority_source`, `seniority_confidence`, `seniority_final`, `seniority_3level`
  `seniority_final` is canonical; `seniority_3level` is a derived helper bucket
- `seniority_cross_check`
- `is_aggregator`, `real_employer`, `is_multi_location`
- `city_extracted`, `state_normalized`, `country_extracted`, `is_remote_inferred`
- `metro_area`, `metro_source`, `metro_confidence`
- `description_core` (rule-based boilerplate removal)
- `posting_age_days`, `period`, `scrape_week`
- `ghost_job_risk`
- `description_quality_flag`, `date_flag`, `dedup_method`, `preprocessing_version`, `boilerplate_removed`
- `is_english`, `description_hash`

`is_remote` is the normalized source-provided remote flag from Stage 1.
`is_remote_inferred` is a separate Stage 6 boolean derived from the location string.
`search_metro_*` remains query metadata from ingest.
`metro_area` is the inferred posting-location metro label from Stage 6.

`posting_age_days` is populated in Stage 7 and is mainly meaningful for scraped rows
with both `scrape_date` and `date_posted`.

**LLM-derived columns:**

| Column | Type | Values | Source |
|---|---|---|---|
| `swe_classification_llm` | string | SWE / SWE_ADJACENT / NOT_SWE / null | LLM Stage 10 |
| `seniority_llm` | string | entry / associate / mid-senior / director / unknown / null | LLM Stage 10 |
| `description_core_llm` | string | Core content reconstructed locally from Stage 10 unit IDs / null | LLM Stage 10 |
| `ghost_assessment_llm` | string | realistic / inflated / ghost_likely / null | LLM Stage 10 |
| `yoe_min_years_llm` | int | Binding LLM YOE floor for the role / null | LLM Stage 10 |
| `llm_extraction_status` | string | ok / cannot_complete / null | LLM Stage 10 |
| `llm_extraction_unit_ids` | string | JSON list of boilerplate unit IDs / null | LLM Stage 10 |
| `llm_extraction_uncertain_unit_ids` | string | JSON list of uncertain unit IDs / null | LLM Stage 10 |
| `llm_model` | string | Model name that produced the classification / null | LLM Stage 10 |
| `llm_prompt_version` | string | Prompt version hash / null | LLM Stage 10 |
| `yoe_extracted` | float | Resolved primary years-of-experience requirement from raw description / null | Stage 5 improvement |
| `yoe_min_extracted` | float | Minimum valid accepted YOE mention / null | Stage 5 improvement |
| `yoe_max_extracted` | float | Maximum valid accepted YOE bound / null | Stage 5 improvement |
| `yoe_match_count` | int | Number of accepted YOE mentions used for resolution | Stage 5 improvement |
| `yoe_resolution_rule` | string | Rule used to choose `yoe_extracted` | Stage 5 improvement |
| `yoe_all_mentions_json` | string | Compact JSON audit trail of YOE candidate mentions | Stage 5 improvement |
| `yoe_seniority_contradiction` | bool | True if resolved primary YOE contradicts canonical `seniority_final` | Stage 5 improvement |

`seniority_source` is a controlled rule-based provenance field with values:
`title_keyword`, `title_level_number`, `description_explicit`, or `unknown`.
YOE extraction remains a cross-check only and does not create its own seniority
source state. `yoe_min_years_llm` is also cross-validation only and must not
drive seniority assignment.

**Null values in LLM columns** mean that the posting was not routed to that specific LLM task by Stage 9 or that the corresponding LLM call failed. In either case, the rule-based columns remain available as fallback / ablation columns.

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
    "unit_validation_failures": "...",
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
    "unit_reconstruction_pass_rate": "...",
    "cannot_complete_rate": "..."
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
| **Dedup sensitivity** | Whether Stage 4 dedup assumptions matter | Compare alternative Stage 4 matching regimes (for example stricter vs looser title similarity or description-support requirements) |
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
     - Select control cohort
     - Apply short-description skips
     - Output: extraction candidates, extraction results, and cleaned posting table
     |
  4. Stage 10: Full classification batch           depends on: Stage 9
     - Run classification against cleaned-description-first inputs
     - Cache responses in SQLite
     - Write classification results and final posting-level integrated artifact
     |
  5. Stage 12: Three-way validation                depends on: Stage 10
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

**Critical path:** The Stage 1 rewrite (step 1) is the gate for everything else. The small profiling subset and 100-call profile run (step 4) are the gate for the full LLM batch.

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
| **Boilerplate** | Section header regex; paragraph-level filtering | Stage 10 LLM unit-ID boilerplate selection | 30 unit-validation failures / uncertain cases + 10 random passes |
| **Ghost job** | Entry-level + 5yr+ experience heuristic | Stage 10 LLM ghost assessment | 30 rule-LLM disagreements |
| **Aggregator** | Company name in AGGREGATORS list | 100 aggregator postings — LLM extracts real employer | 20 verification items |

### Gold-standard annotation

Sample 500 postings stratified by source, predicted seniority, and classification confidence. LLM pre-labels all four tasks. Human corrects LLM labels. Compute inter-rater kappa. If kappa > 0.80, LLM is reliable as a second annotator. If kappa < 0.60, expand human review.

**Warning (Ashwin et al. 2025):** LLM coding errors are NOT random — they correlate with text characteristics. Always validate against human labels before trusting LLM annotations at scale.

