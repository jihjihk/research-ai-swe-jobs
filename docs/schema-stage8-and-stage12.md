# Schema Reference: Stage 8 (Current) and Stage 12 (Planned)

Date: 2026-03-23

## Overview

This document describes the column schema available for exploration **right now** from Stage 8, and the additional columns expected after the LLM augmentation pipeline (Stages 9-12) runs at scale.

**Current state:** The pipeline runs through Stage 8, producing `preprocessing/intermediate/stage8_final.parquet` (76 columns, ~1.22M rows). The LLM stages (9-12) are implemented but have not run at scale due to quota constraints. The existing `data/unified.parquet` (67 columns, ~1.06M rows) is from an earlier pipeline run and is **stale** relative to the current stage8 artifact.

**Practical consequence for exploration:** Work from `stage8_final.parquet` directly. It has 9 columns that were dropped in the stale unified export (including metro, seniority detail, and company canonical fields) and ~162K more rows from additional scraped days.

### Artifact summary

| Artifact | Rows | Columns | Status | Use for exploration? |
|---|---|---|---|---|
| `preprocessing/intermediate/stage8_final.parquet` | 1,220,245 | 76 | Current | **Yes — primary** |
| `data/unified.parquet` | 1,058,397 | 67 | Stale (2026-03-21 run) | No — use stage8 |
| `data/unified_observations.parquet` | 1,064,831 | 67 | Stale (2026-03-21 run) | No — use stage8 + stage1_observations |
| `preprocessing/intermediate/stage1_observations.parquet` | current | — | Current | For daily panel joins |

### Source composition (stage8_final)

| Source | Platform | Rows | Period | SWE rows |
|---|---|---|---|---|
| kaggle_asaniczka | linkedin | 1,061,202 | 2024-01 | 22,492 |
| kaggle_arshkon | linkedin | 118,589 | 2024-04 | 4,441 |
| scraped | linkedin | 26,884 | 2026-03 | — |
| scraped | indeed | 13,570 | 2026-03 | — |
| **Total** | | **1,220,245** | | **32,257** |

---

## Column Reference

### Status legend

- **Verified**: Column exists in the current stage8_final artifact and values confirmed by query
- **Code-confirmed**: Column defined in pipeline code, artifact not yet produced at scale
- **Plan-only**: Column described in plan docs but not yet implemented

---

### 1. Identity and Provenance

| Column | Type | Origin | Meaning | Status |
|---|---|---|---|---|
| `uid` | VARCHAR | Stage 1 | Globally unique ID: `source_prefix + id`. E.g., `arshkon_12345`, `linkedin_67890` | Verified |
| `job_id` | VARCHAR | Stage 1 | Source-native ID (job_id for arshkon, hashed job_link for asaniczka, site+id for scraped) | Verified |
| `source` | VARCHAR | Stage 1 | Dataset origin: `kaggle_arshkon`, `kaggle_asaniczka`, `scraped` | Verified |
| `source_platform` | VARCHAR | Stage 1 | Posting platform: `linkedin` or `indeed` | Verified |
| `site` | VARCHAR | Stage 1 | Same as source_platform for scraped; `linkedin` for Kaggle sources | Verified |

**Use `uid`** as the primary key. Use `source` for dataset-level stratification. Use `source_platform` to filter to LinkedIn-only analysis.

---

### 2. Job Content

| Column | Type | Origin | Meaning | Status |
|---|---|---|---|---|
| `title` | VARCHAR | Stage 1 | Original job title as posted | Verified |
| `title_normalized` | VARCHAR | Stage 1 | Lowercased, whitespace-normalized title | Verified |
| `description_raw` | VARCHAR | Stage 1 | Original description text, never modified after ingest | Verified |
| `description` | VARCHAR | Stage 1 | Working copy of description (whitespace-normalized) | Verified |
| `description_length` | BIGINT | Stage 1 | Character count of `description` | Verified |
| `description_core` | VARCHAR | Stage 3 | Rule-based boilerplate removal (regex section matching). ~44% accuracy. | Verified |
| `core_length` | DOUBLE | Stage 3 | Character count of `description_core` | Verified |
| `boilerplate_flag` | VARCHAR | Stage 3 | Boilerplate detection result from Stage 3 | Verified |
| `description_core_llm` | VARCHAR | Stage 11 | LLM-based boilerplate removal. Reconstructed from sentence-unit IDs. Falls back to `description_core` on LLM failure. | Code-confirmed |

**Recommended usage order for descriptions:**
1. **Now (Stage 8):** Use `description_core` for text analysis despite low accuracy. Use `description` when boilerplate matters less (e.g., keyword presence) or when `description_core` is null.
2. **After Stage 12:** Use `description_core_llm` as primary. Keep `description_core` for ablation.

---

### 3. Company

| Column | Type | Origin | Meaning | Status |
|---|---|---|---|---|
| `company_name` | VARCHAR | Stage 1 | Company name as listed on the posting | Verified |
| `company_name_normalized` | VARCHAR | Stage 1 | Lowercased, whitespace-normalized company name | Verified |
| `is_aggregator` | BOOLEAN | Stage 2 | True if posting is from a staffing agency or aggregator (e.g., Dice, Lensa, DataAnnotation) | Verified |
| `real_employer` | VARCHAR | Stage 2 | Extracted actual employer name when aggregator detected; null otherwise | Verified |
| `company_name_effective` | VARCHAR | Stage 2 | `real_employer` if aggregator, else `company_name`. **Only in stage8, not in stale unified.** | Verified (stage8 only) |
| `company_name_canonical` | VARCHAR | Stage 4 | Normalized version of `company_name_effective` for dedup matching. **Only in stage8.** | Verified (stage8 only) |
| `company_name_canonical_method` | VARCHAR | Stage 4 | How canonicalization was done. **Only in stage8.** | Verified (stage8 only) |
| `company_industry` | VARCHAR | Stage 1 | Industry label. Arshkon: from companion join. Scraped LinkedIn: 100%. Asaniczka/Indeed: null. | Verified |
| `company_size` | DOUBLE | Stage 1 | Numeric employee count. Arshkon: from companion. Indeed: 91%. Others: null. | Verified |
| `company_size_raw` | VARCHAR | Stage 1 | Original size string before parsing | Verified |
| `company_size_category` | VARCHAR | Stage 1 | Categorical size bucket (arshkon companion data only) | Verified |
| `company_id_kaggle` | DOUBLE | Stage 1 | Arshkon company_id for companion joins; null for other sources | Verified |

**Recommended usage order for company:**
1. Use `company_name_effective` (stage8) for company-level analysis — it resolves aggregators.
2. Use `company_name_canonical` (stage8) for grouping across spelling variants.
3. Filter `is_aggregator == True` rows for sensitivity analysis.

---

### 4. Seniority (Complex — Read Carefully)

There are **10 seniority-related columns** in stage8, reflecting multiple classification attempts and resolution strategies. This section explains each and provides a clear recommendation.

#### Column inventory

| Column | Type | Origin | Values | Meaning | Status |
|---|---|---|---|---|---|
| `seniority_raw` | VARCHAR | Stage 1 | Verbatim source strings (e.g., "Mid-Senior level", "Mid senior") | Original label before any mapping. For auditing the mapping. | Verified |
| `seniority_native` | VARCHAR | Stage 1 | `entry`, `associate`, `mid-senior`, `director`, `intern`, `executive`, null | Platform-provided label mapped to canonical enum. High quality where available. | Verified |
| `seniority_imputed` | VARCHAR | Stage 5 | `entry`, `associate`, `mid-senior`, `director`, `unknown` | Rule-based classifier using title keywords and description patterns. 80% unknown. | Verified |
| `seniority_source` | VARCHAR | Stage 5 | `unknown`, `title_keyword`, `weak_title_associate`, `title_manager`, `description_explicit`, `weak_title_level` | Which signal drove `seniority_imputed` | Verified |
| `seniority_confidence` | DOUBLE | Stage 5 | 0.0-1.0 | Confidence of `seniority_imputed` | Verified |
| `seniority_final` | VARCHAR | Stage 5 | `entry`, `associate`, `mid-senior`, `director`, `unknown` | **Best available 5-level label.** Merges `seniority_imputed` with `seniority_native` backfill. | Verified |
| `seniority_final_source` | VARCHAR | Stage 5 | `title_keyword`, `native_backfill`, `weak_title_associate`, `title_manager`, `description_explicit`, `weak_title_level`, `title_prior`, `unknown` | How `seniority_final` was resolved. **Only in stage8.** | Verified (stage8 only) |
| `seniority_final_confidence` | DOUBLE | Stage 5 | 0.0-1.0 | Confidence of `seniority_final`. **Only in stage8.** | Verified (stage8 only) |
| `seniority_3level` | VARCHAR | Stage 5 | `junior`, `mid`, `senior`, `unknown` | Coarse 3-level collapse derived from `seniority_final`: entry->junior, associate->mid, mid-senior->senior, director->senior. | Verified |
| `seniority_cross_check` | VARCHAR | Stage 5 | — | Cross-validation diagnostic between imputed and native | Verified |
| `yoe_extracted` | DOUBLE | Stage 5 | Numeric or null | Minimum years-of-experience parsed from description text | Verified |
| `yoe_seniority_contradiction` | BOOLEAN | Stage 5 | true/false | True when YOE and seniority level contradict (e.g., entry + 5 YOE) | Verified |
| `seniority_llm` | VARCHAR | Stage 11 | `entry`, `associate`, `mid-senior`, `director`, `unknown` | LLM-classified seniority from explicit title/description signals only. Primary analysis variable post-LLM. | Code-confirmed |

#### How `seniority_final` is resolved (Stage 5 logic)

```
1. Start with seniority_imputed (rule-based from title/description)
2. If imputed != 'unknown': use it (source = title_keyword, etc.)
3. If imputed == 'unknown' AND seniority_native is available:
   backfill from native (source = 'native_backfill')
4. Apply title_prior heuristic for remaining edge cases
5. Result: seniority_final with seniority_final_source
```

This reduces the unknown rate from 80% (imputed alone) to 72.5%.

#### Coverage by source

| Source | seniority_native coverage | seniority_final != unknown |
|---|---|---|
| kaggle_arshkon | 75.7% | ~54.6% of arshkon SWE |
| kaggle_asaniczka | 100% (but only mid-senior/associate) | ~71.2% of asaniczka SWE |
| scraped | 56.8% (LinkedIn only; Indeed = 0%) | ~54.8% of scraped SWE |

**Critical gap:** asaniczka has zero entry-level labels. All entry-level historical baseline comes from arshkon (~89 entry-level SWE postings).

#### Recommended usage for seniority

**Right now (Stage 8):**
1. **`seniority_final`** — Use as the primary seniority variable. It gives the best coverage by merging rule-based and platform-native labels.
2. **`seniority_final_source`** (stage8 only) — Filter or stratify by this to understand where the label came from. `native_backfill` rows rely on platform labels; `title_keyword` rows use title pattern matching.
3. **`seniority_3level`** — Use for coarse analyses needing larger cell sizes: junior/mid/senior.
4. **`seniority_native`** — Use for high-confidence platform-label-only analysis (100% precision but incomplete coverage).
5. **`seniority_imputed`** — Avoid unless you specifically need the rule-only baseline.

**After Stage 12:**
1. **`seniority_llm`** — Primary analysis variable. High-precision explicit-signal-only classification.
2. **`seniority_final`** — Ablation baseline / fallback when `seniority_llm` is null.
3. **`seniority_native`** — Cross-validation anchor.

#### Current seniority distribution (stage8_final, all rows)

| seniority_final | Count | % |
|---|---|---|
| unknown | 885,372 | 72.6% |
| mid-senior | 253,395 | 20.8% |
| associate | 50,429 | 4.1% |
| director | 18,960 | 1.6% |
| entry | 12,089 | 1.0% |

---

### 5. SWE / Occupation Classification

| Column | Type | Origin | Values | Meaning | Status |
|---|---|---|---|---|---|
| `is_swe` | BOOLEAN | Stage 5 | true/false | Primary SWE flag. True if the role's primary function is writing/maintaining software. | Verified |
| `is_swe_adjacent` | BOOLEAN | Stage 5 | true/false | SWE-adjacent: technical roles involving some code but not primarily software development. | Verified |
| `is_control` | BOOLEAN | Stage 5 | true/false | Control occupation group for cross-occupation comparisons. | Verified |
| `swe_confidence` | DOUBLE | Stage 5 | 0.0-1.0 | Classification confidence score | Verified |
| `swe_classification_tier` | VARCHAR | Stage 5 | `regex`, `embedding_high`, `embedding_llm`, `embedding_adjacent` | Which classification method fired | Verified |
| `swe_classification_llm` | VARCHAR | Stage 11 | `SWE`, `SWE_ADJACENT`, `NOT_SWE` | LLM-based SWE classification | Code-confirmed |

**Recommended usage:**
1. **Now:** `is_swe == True` for the SWE sample. 32,257 rows in stage8. Classification is 85% regex-based, 13% embedding+LLM, 2% embedding-only.
2. **After Stage 12:** Use `swe_classification_llm` as primary; keep `is_swe` for ablation.
3. `is_swe_adjacent` rows (9,007) are useful for RQ2 (task migration) cross-occupation comparisons.
4. `is_control` rows (131,841) are the non-technical control group.

---

### 6. Geography and Location

| Column | Type | Origin | Meaning | Status |
|---|---|---|---|---|
| `location` | VARCHAR | Stage 1 | Raw location string from posting | Verified |
| `location_normalized` | VARCHAR | Stage 1 | Whitespace-normalized location | Verified |
| `city_extracted` | VARCHAR | Stage 6 | Parsed city from location string | Verified |
| `state_normalized` | VARCHAR | Stage 6 | Parsed and normalized state abbreviation | Verified |
| `country_extracted` | VARCHAR | Stage 6 | Parsed country | Verified |
| `is_remote` | BOOLEAN | Stage 1 | Source-provided remote flag (normalized) | Verified |
| `is_remote_inferred` | BOOLEAN | Stage 6 | Inferred from location text cues (e.g., "Remote", "Anywhere"). **Only in stage8.** | Verified (stage8 only) |
| `metro_area` | VARCHAR | Stage 6 | Posting-location metro aligned to study metro frame. **Only in stage8.** | Verified (stage8 only) |
| `metro_source` | VARCHAR | Stage 6 | How metro was assigned: `search_metro`, `manual_alias`, `city_state_lookup`, `unresolved`. **Only in stage8.** | Verified (stage8 only) |
| `metro_confidence` | VARCHAR | Stage 6 | Coarse confidence for metro assignment. **Only in stage8.** | Verified (stage8 only) |
| `search_query` | VARCHAR | Stage 1 | Scraper search query (scraped only; asaniczka: `search_position`) | Verified |
| `query_tier` | VARCHAR | Stage 1 | Query tier from scraper design (scraped only) | Verified |
| `search_metro_id` | VARCHAR | Stage 1 | Metro ID from scraper config (scraped only) | Verified |
| `search_metro_name` | VARCHAR | Stage 1 | Metro name from scraper search (scraped only; asaniczka: `search_city`) | Verified |
| `search_metro_region` | VARCHAR | Stage 1 | Region from scraper config (scraped only) | Verified |
| `search_location` | VARCHAR | Stage 1 | Full search location string (scraped only) | Verified |

**Recommended usage:**
1. **`metro_area`** (stage8 only) for geographic analysis — it normalizes posting locations to the 26-metro study frame.
2. **`state_normalized`** for state-level analysis.
3. **`search_metro_name`** is query metadata (where we searched), not posting location. Don't use it as posting geography.
4. **`is_remote`** is the source-provided flag; **`is_remote_inferred`** (stage8 only) is a separate boolean derived from regex on `location` (matches "remote", "anywhere", "work from home", "wfh").

---

### 7. Time and Temporal

| Column | Type | Origin | Meaning | Status |
|---|---|---|---|---|
| `date_posted` | VARCHAR | Stage 1 | Date the posting was listed. Arshkon: from epoch conversion. Asaniczka: `first_seen`. Scraped LinkedIn: 2.8% populated. Scraped Indeed: 100%. | Verified |
| `scrape_date` | VARCHAR | Stage 1 | Date the posting was observed by scraper. Null for Kaggle sources. | Verified |
| `period` | VARCHAR | Stage 7 | Coarse time period: `2024-01` (asaniczka), `2024-04` (arshkon), `2026-03` (scraped). Source-driven for Kaggle, scrape_date-derived for scraped. | Verified |
| `posting_age_days` | DOUBLE | Stage 7 | Days between `date_posted` and `scrape_date`. Mainly meaningful for scraped rows with both dates. | Verified |
| `scrape_week` | DOUBLE | Stage 7 | ISO week of `scrape_date`. Null for Kaggle. | Verified |

**Recommended usage:**
1. **`period`** for cross-era comparison (2024-01 vs 2024-04 vs 2026-03). This is the primary temporal stratifier.
2. **`date_posted`** for within-period date variation where available.
3. **`posting_age_days`** for posting-duration analysis (scraped rows with `date_posted` only).

---

### 8. Quality Flags

| Column | Type | Origin | Values | Meaning | Status |
|---|---|---|---|---|---|
| `date_flag` | VARCHAR | Stage 8 | `ok`, `date_posted_out_of_range` | Date validation summary | Verified |
| `is_english` | BOOLEAN | Stage 8 | true/false | Language detection flag | Verified |
| `description_hash` | VARCHAR | Stage 8 | SHA-256 hex | Hash of raw `description` (not `description_core`). Used as LLM cache key and dedup signal. | Verified |
| `ghost_job_risk` | VARCHAR | Stage 8 | `low`, `medium`, `high` | Rule-based ghost-job heuristic. Only entry-level `seniority_final` rows can score above `low` (medium: YOE >= 3 or contradiction; high: YOE >= 5). | Verified |
| `description_quality_flag` | VARCHAR | Stage 8 | `ok`, `too_short`, `empty` | Description quality: `empty` if null/blank, `too_short` if < 50 chars, else `ok`. Based on `description_core`. | Verified |
| `ghost_assessment_llm` | VARCHAR | Stage 11 | `realistic`, `inflated`, `ghost_likely` | LLM-based ghost assessment | Code-confirmed |

**Recommended usage:**
1. Default filter: `is_english == True AND date_flag == 'ok'`
2. `ghost_job_risk` is very conservative (only 359 non-low out of 1.22M). Useful for flagging, not for exclusion.
3. After Stage 12: `ghost_assessment_llm` will be a richer signal.

---

### 9. Pipeline Metadata

| Column | Type | Origin | Meaning | Status |
|---|---|---|---|---|
| `preprocessing_version` | VARCHAR | Stage 8 | Pipeline version string | Verified |
| `dedup_method` | VARCHAR | Stage 8 | How this row survived dedup | Verified |
| `boilerplate_removed` | BOOLEAN | Stage 8 | Whether boilerplate removal was applied | Verified |
| `is_multi_location` | BOOLEAN | Stage 4 | True if this posting shares a canonical opening across multiple locations | Verified |
| `work_type` | VARCHAR | Stage 1 | Job type field from source | Verified |
| `job_url` | VARCHAR | Stage 1 | Posting URL where available | Verified |
| `skills_raw` | VARCHAR | Stage 1 | Skills field from source | Verified |
| `asaniczka_skills` | VARCHAR | Stage 1 | Asaniczka-specific skills from companion join | Verified |

---

## Columns Available Only After LLM Stages (Stage 12 Target)

These columns will be added by Stage 11 when the LLM pipeline runs at scale. They are additive — all existing rule-based columns are preserved. Stage 9 routing columns (`needs_llm_classification`, `needs_llm_extraction`, `llm_candidate`, `llm_route_group`, `llm_skip_reason`, `llm_classification_reason`) are used internally and **dropped** before Stage 11 writes its output.

### Primary LLM analysis columns

| Column | Type | Origin | Values | Meaning | Status |
|---|---|---|---|---|---|
| `swe_classification_llm` | VARCHAR | Stage 11 | `SWE`, `SWE_ADJACENT`, `NOT_SWE` | LLM-based occupation classification. Null for rows where rule-based confidence was already high. | Code-confirmed |
| `seniority_llm` | VARCHAR | Stage 11 | `entry`, `associate`, `mid-senior`, `director`, `unknown` | LLM seniority from explicit title/description signals only. Primary analysis variable post-LLM. | Code-confirmed |
| `ghost_assessment_llm` | VARCHAR | Stage 11 | `realistic`, `inflated`, `ghost_likely` | LLM ghost-job assessment. Richer than rule-based `ghost_job_risk`. | Code-confirmed |
| `description_core_llm` | VARCHAR | Stage 11 | Text | LLM-based boilerplate removal via sentence-unit selection. Null on LLM failure; `description_core` remains as fallback. | Code-confirmed |

### LLM extraction diagnostics

| Column | Type | Origin | Meaning | Status |
|---|---|---|---|---|
| `llm_extraction_status` | VARCHAR | Stage 11 | `ok` or `cannot_complete` | Code-confirmed |
| `llm_extraction_validated` | BOOLEAN | Stage 11 | True only if extraction passed all validation checks | Code-confirmed |
| `llm_extraction_unit_ids` | VARCHAR | Stage 11 | JSON array of boilerplate unit IDs | Code-confirmed |
| `llm_extraction_uncertain_unit_ids` | VARCHAR | Stage 11 | JSON array of uncertain unit IDs | Code-confirmed |
| `llm_extraction_reason` | VARCHAR | Stage 11 | Validation failure reason (Stage 11 side) | Code-confirmed |
| `llm_extraction_model_reason` | VARCHAR | Stage 11 | LLM's own short reason phrase | Code-confirmed |
| `llm_extraction_units_count` | INT64 | Stage 11 | Total sentence-unit count for the description | Code-confirmed |
| `llm_extraction_single_unit` | BOOLEAN | Stage 11 | True if description had only one unit (extraction not attempted) | Code-confirmed |
| `llm_extraction_drop_ratio` | DOUBLE | Stage 11 | Fraction of characters dropped as boilerplate | Code-confirmed |

### LLM provenance

| Column | Type | Origin | Meaning | Status |
|---|---|---|---|---|
| `llm_model_classification` | VARCHAR | Stage 11 | Model name used for classification call | Code-confirmed |
| `llm_model_extraction` | VARCHAR | Stage 11 | Model name used for extraction call | Code-confirmed |
| `llm_prompt_version_classification` | VARCHAR | Stage 11 | Prompt template hash for classification | Code-confirmed |
| `llm_prompt_version_extraction` | VARCHAR | Stage 11 | Prompt template hash for extraction | Code-confirmed |

**Design principle:** LLM columns are null for rows that were not routed or where the LLM call failed. In those cases, the rule-based columns serve as fallback. Both are preserved for ablation studies. Stage 11 joins LLM results back to all posting rows by `description_hash`, so identical descriptions share the same LLM outputs.

---

## Recommended Columns for Immediate Exploration (Stage 8)

Use `preprocessing/intermediate/stage8_final.parquet` as the exploration dataset. Apply these default filters:

```sql
WHERE source_platform = 'linkedin'    -- LinkedIn-only for cross-period comparability
  AND is_english = true               -- Drop non-English
  AND date_flag = 'ok'                -- Drop date-flagged rows
```

### Core analysis columns

| Analysis need | Primary column | Fallback | Notes |
|---|---|---|---|
| **SWE sample** | `is_swe` | — | 32,257 rows. `is_swe_adjacent` for broader tech sample. |
| **Seniority** | `seniority_final` | `seniority_native` | 72.5% unknown. Filter to non-unknown for seniority-stratified analysis. Use `seniority_final_source` to understand provenance. |
| **Seniority (coarse)** | `seniority_3level` | — | junior/mid/senior/unknown. Larger cells. |
| **Time period** | `period` | `date_posted` | Three periods: 2024-01, 2024-04, 2026-03. |
| **Description text** | `description_core` | `description` | ~44% accuracy on boilerplate removal. Use `description` when accuracy matters more. |
| **Company** | `company_name_effective` | `company_name` | Resolves aggregators. Only in stage8. |
| **Company (dedup)** | `company_name_canonical` | `company_name_effective` | For grouping across spelling variants. Only in stage8. |
| **Geography** | `metro_area` | `state_normalized` | Metro aligned to 26-metro study frame. Only in stage8. |
| **Remote** | `is_remote_inferred` | `is_remote` | Combines source flag + text inference. Only in stage8. |
| **Ghost/inflation** | `ghost_job_risk` | — | Very conservative. Pair with `yoe_extracted` and `yoe_seniority_contradiction` for custom checks. |
| **Aggregator filter** | `is_aggregator` | — | 55,120 aggregator rows. Consider excluding for sensitivity. |

### Important caveats for exploration

1. **Seniority is the weakest link.** 72.5% unknown rate means seniority-stratified analyses use <28% of data. This will improve substantially with `seniority_llm` after Stage 10 runs.

2. **Entry-level historical baseline is thin.** Only arshkon has entry-level labels, yielding ~89 entry-level SWE postings. Asaniczka has zero. This is the binding constraint for RQ1 junior-share analysis.

3. **Boilerplate removal is noisy.** `description_core` has ~44% accuracy. For keyword/pattern analyses, using raw `description` may be preferable. `description_core_llm` will be a major improvement.

4. **stage8 has columns that unified drops.** The 9 columns only in stage8 are important for exploration: `company_name_effective`, `company_name_canonical`, `company_name_canonical_method`, `seniority_final_source`, `seniority_final_confidence`, `is_remote_inferred`, `metro_area`, `metro_source`, `metro_confidence`. Use stage8 directly.

5. **Row counts will grow.** Each additional day of scraping adds ~34K rows before dedup. Re-run the pipeline from Stage 1 after syncing new scraped data.
