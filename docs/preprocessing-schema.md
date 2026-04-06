# Preprocessing Schema Reference

Last updated: 2026-03-31

Complete column reference for the preprocessing pipeline. For architecture, operations, and development practices, see [`preprocessing-guide.md`](preprocessing-guide.md).

---

## How to Use This Document

**Primary analysis file:** `data/unified.parquet` (~99 columns, ~1.40M rows). This is the Stage 11 final output containing all rule-based columns plus LLM columns from Stages 9-10.

**Current LLM coverage status (as of 2026-04-05):**
- Stage 9 (`description_core_llm`): Available for Kaggle SWE rows (~24K labeled). Scraped data is all deferred (no budget allocated).
- Stage 10 (`seniority_llm`, `swe_classification_llm`, `ghost_assessment_llm`, `yoe_min_years_llm`): No budget allocated — all null. Use rule-based columns (`seniority_final`, `is_swe`, `ghost_job_risk`).

---

## Column Availability by Stage

This table shows when each column category first becomes available:

| Category | First available | Columns | Key additions |
|---|---|---|---|
| Identity & provenance | Stage 1 | 5 | `uid`, `source`, `source_platform` |
| Raw job content | Stage 1 | 6 | `title`, `description`, `description_raw` |
| Company (basic) | Stage 1 | 5 | `company_name`, `company_industry`, `company_size` |
| Aggregator handling | Stage 2 | 3 | `is_aggregator`, `real_employer`, `company_name_effective` |
| Boilerplate removal | Stage 3 | 3 | `description_core`, `core_length`, `boilerplate_flag` |
| Company canonicalization | Stage 4 | 2 | `company_name_canonical`, `company_name_canonical_method` |
| Multi-location flag | Stage 4 | 1 | `is_multi_location` |
| SWE classification | Stage 5 | 5 | `is_swe`, `is_swe_adjacent`, `is_control`, `swe_confidence`, `swe_classification_tier` |
| Seniority (rule-based) | Stage 5 | 10 | `seniority_final`, `seniority_3level`, etc. |
| YOE extraction | Stage 5 | 7 | `yoe_extracted`, `yoe_min_extracted`, `yoe_seniority_contradiction`, etc. |
| Location parsing | Stage 6 | 6 | `city_extracted`, `state_normalized`, `metro_area`, `is_remote_inferred` |
| Temporal derivations | Stage 7 | 3 | `period`, `posting_age_days`, `scrape_week` |
| Quality flags | Stage 8 | 5 | `date_flag`, `is_english`, `ghost_job_risk`, `description_quality_flag` |
| Pipeline metadata | Stage 8 | 3 | `preprocessing_version`, `dedup_method`, `boilerplate_removed` |
| LLM cleaned text | Stage 9 | 2 | `description_core_llm`, `selected_for_control_cohort` |
| LLM coverage tracking | Stage 9-10 | 2 | `llm_extraction_coverage`, `llm_classification_coverage` |
| LLM extraction diagnostics | Stage 9 | 9 | `llm_extraction_status`, `llm_extraction_drop_ratio`, etc. |
| LLM classification | Stage 10 | 4 | `seniority_llm`, `swe_classification_llm`, `ghost_assessment_llm`, `yoe_min_years_llm` |
| LLM provenance | Stage 9-10 | 4 | `llm_model_*`, `llm_prompt_version_*` |

**Total at Stage 8:** ~80 columns, 1,395,790 rows.
**Total at Stage 11 / unified.parquet:** 99 columns, 1,395,790 rows (all Stage 8 columns preserved + LLM additions).

---

## What to Use at Each Processing Milestone

### At Stage 8+ (rule-based baseline)

Work from `data/unified.parquet` (Stage 11 output, includes all rule-based + available LLM columns).

| Analysis need | Primary column | Fallback | Notes |
|---|---|---|---|
| SWE sample | `is_swe` | — | ~33K rows. `is_swe_adjacent` for broader tech sample. |
| Seniority | `seniority_final` | `seniority_native` | 71.8% unknown. Filter to non-unknown for stratified analysis. |
| Seniority (coarse) | `seniority_3level` | — | junior/mid/senior/unknown |
| Time period | `period` | `date_posted` | Three periods: 2024-01, 2024-04, 2026-03 |
| Description text | `description_core` | `description` | ~44% accuracy on boilerplate removal |
| Company | `company_name_effective` | `company_name` | Resolves aggregators |
| Company (grouped) | `company_name_canonical` | `company_name_effective` | For grouping across spelling variants |
| Geography | `metro_area` | `state_normalized` | 26-metro study frame |
| Remote status | `is_remote_inferred` | `is_remote` | Combines source flag + text inference |
| Ghost/inflation | `ghost_job_risk` | — | Very conservative; pair with `yoe_extracted` |
| Aggregator filter | `is_aggregator` | — | 73K aggregator rows |

**Default filters for most analyses:**

```sql
WHERE source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
```

### At Stage 9 (extraction complete, classification pending)

Work from `preprocessing/intermediate/stage9_llm_cleaned.parquet`. Same row count as Stage 8.

New columns available:
- `description_core_llm`: LLM-cleaned description. Use as primary text for analysis. Falls back to `description_core` when null/empty.
- `selected_for_control_cohort`: Whether this row is in the deterministic control group for LLM analysis.

### At Stage 10 (LLM classification complete)

Work from `preprocessing/intermediate/stage10_llm_integrated.parquet` or `data/unified.parquet`.

| Analysis need | Primary column | Fallback / ablation |
|---|---|---|
| SWE sample | `swe_classification_llm` (for routed rows) | `is_swe` |
| Seniority | `seniority_llm` | `seniority_final`, `seniority_native` |
| Clean text | `description_core_llm` | `description_core`, `description` |
| Ghost / inflation | `ghost_assessment_llm` | `ghost_job_risk` |

LLM columns are null for rows not routed to LLM classification. In those cases, use the rule-based columns as fallback.

---

## Source Composition

| Source | Platform | Rows | Period | SWE rows |
|---|---|---|---|---|
| kaggle_asaniczka | linkedin | ~1,012K | 2024-01 | 23,213 |
| kaggle_arshkon | linkedin | ~118K | 2024-04 | 5,019 |
| scraped | linkedin | ~146K | 2026-03 | 24,095 |
| scraped | indeed | varies | 2026-03 | varies |
| **Total** | | **~1,396K** | | **~59K** |

Note: Filtered counts (LinkedIn, English, date_flag=ok) shown for SWE. Total rows include all platforms and languages.

---

## Detailed Column Reference

### 1. Identity and Provenance

| Column | Type | Stage | Meaning |
|---|---|---|---|
| `uid` | VARCHAR | 1 | Globally unique ID: `source_prefix + id`. E.g., `arshkon_12345`, `asaniczka_70c0f7742486e930`, `linkedin_67890`. **Primary key.** |
| `job_id` | VARCHAR | 1 | Source-native ID. `job_id` for arshkon, hashed `job_link` for asaniczka, `site + "_" + id` for scraped. |
| `source` | VARCHAR | 1 | Dataset origin: `kaggle_arshkon`, `kaggle_asaniczka`, `scraped`. Use for dataset-level stratification. |
| `source_platform` | VARCHAR | 1 | Posting platform: `linkedin` or `indeed`. |
| `site` | VARCHAR | 1 | Same as `source_platform` for scraped; `linkedin` for Kaggle sources. |

---

### 2. Job Content

| Column | Type | Stage | Meaning |
|---|---|---|---|
| `title` | VARCHAR | 1 | Original job title as posted. |
| `title_normalized` | VARCHAR | 1 | Lowercased, whitespace-normalized. Level indicators and remote/hybrid suffixes stripped. |
| `description_raw` | VARCHAR | 1 | Original description text, never modified after ingest. Immutable provenance copy. |
| `description` | VARCHAR | 1 | Working copy: whitespace-normalized, Unicode-cleaned. Used by most downstream stages. |
| `description_length` | BIGINT | 1 | Character count of `description`. |
| `description_core` | VARCHAR | 3 | Rule-based boilerplate removal via section header detection. ~44% accuracy. |
| `core_length` | DOUBLE | 3 | Character count of `description_core`. |
| `boilerplate_flag` | VARCHAR | 3 | Boilerplate detection result. |
| `description_core_llm` | VARCHAR | 9 | LLM-based boilerplate removal. Reconstructed from validated extraction output. Empty string for short-description hard skips (< 15 words). Use `description_core` as fallback when null/empty. **Primary text column after Stage 9.** |

**Recommended text column priority:**
1. `description_core_llm` (after Stage 9) — highest quality boilerplate removal
2. `description_core` (Stage 3+) — rule-based, ~44% accuracy
3. `description` (Stage 1+) — full text including boilerplate

---

### 3. Company

| Column | Type | Stage | Meaning |
|---|---|---|---|
| `company_name` | VARCHAR | 1 | Company name as listed on the posting. |
| `company_name_normalized` | VARCHAR | 1 | Lowercased, whitespace-normalized. |
| `is_aggregator` | BOOL | 2 | True if posting is from a staffing agency (Dice, Lensa, Robert Half, etc.). |
| `real_employer` | VARCHAR | 2 | Extracted actual employer name when aggregator detected. Null otherwise. |
| `company_name_effective` | VARCHAR | 2 | `real_employer` if aggregator, else `company_name`. **Use for company-level analysis.** |
| `company_name_canonical` | VARCHAR | 4 | Normalized version of `company_name_effective` for dedup. **Use for grouping across spelling variants.** |
| `company_name_canonical_method` | VARCHAR | 4 | How canonicalization was done: `passthrough`, `exact_normalized`, `fuzzy`, `alias`. |
| `company_industry` | VARCHAR | 1 | Industry label. Arshkon: from companion join. Scraped LinkedIn: 100%. Asaniczka/Indeed: null. |
| `company_size` | DOUBLE | 1 | Numeric employee count. Arshkon: from companion. Indeed: 91%. Others: null. |
| `company_size_raw` | VARCHAR | 1 | Original size string before parsing (e.g., "10,001-50,000"). |
| `company_size_category` | VARCHAR | 1 | Categorical size bucket (arshkon companion data only). |
| `company_id_kaggle` | DOUBLE | 1 | Arshkon `company_id` for companion joins. Null for other sources. |

---

### 4. Seniority

There are **10 seniority-related columns** at Stage 8, reflecting multiple classification attempts and resolution strategies.

#### Seniority enum values

All seniority columns use a canonical 5-level enum: `entry`, `associate`, `mid-senior`, `director`, `unknown`.

The coarse 3-level mapping (`seniority_3level`): entry → junior, associate → mid, mid-senior → senior, director → senior.

#### Column inventory

| Column | Type | Stage | Values | Meaning |
|---|---|---|---|---|
| `seniority_raw` | VARCHAR | 1 | Verbatim source strings | Original label before mapping. For auditing. |
| `seniority_native` | VARCHAR | 1 | 5-level enum or null | Platform-provided label mapped to canonical enum. High quality where available. |
| `seniority_imputed` | VARCHAR | 5 | 5-level enum | Rule-based classifier from title keywords + description patterns. 80% unknown. |
| `seniority_source` | VARCHAR | 5 | `unknown`, `title_keyword`, `weak_title_associate`, `title_manager`, `description_explicit`, `weak_title_level` | Signal that drove `seniority_imputed`. |
| `seniority_confidence` | DOUBLE | 5 | 0.0-1.0 | Confidence of `seniority_imputed`. |
| `seniority_final` | VARCHAR | 5 | 5-level enum | **Best available rule-based label.** Merges imputed + native backfill. |
| `seniority_final_source` | VARCHAR | 5 | `title_keyword`, `native_backfill`, `weak_title_associate`, `title_manager`, `description_explicit`, `weak_title_level`, `title_prior`, `unknown` | How `seniority_final` was resolved. |
| `seniority_final_confidence` | DOUBLE | 5 | 0.0-1.0 | Confidence of `seniority_final`. |
| `seniority_3level` | VARCHAR | 5 | `junior`, `mid`, `senior`, `unknown` | Coarse 3-level collapse. |
| `seniority_cross_check` | VARCHAR | 5 | — | Cross-validation diagnostic between imputed and native. |
| `seniority_llm` | VARCHAR | 10 | 5-level enum | **LLM-classified seniority from explicit signals only.** Primary analysis variable after Stage 10. |

#### How `seniority_final` is resolved (Stage 5)

```
1. Start with seniority_imputed (rule-based from title/description)
2. If imputed != 'unknown': use it (source = title_keyword, etc.)
3. If imputed == 'unknown' AND seniority_native is available:
   backfill from native (source = 'native_backfill')
4. Apply title_prior heuristic for remaining edge cases
5. Result: seniority_final with seniority_final_source
```

This reduces the unknown rate from 80.4% (imputed alone) to 71.8%.

#### Coverage by source (SWE rows)

| Source | seniority_native coverage | seniority_final != unknown |
|---|---|---|
| kaggle_arshkon | 68.9% | 81.6% |
| kaggle_asaniczka | 100% (only mid-senior/associate) | 100% |
| scraped | 67.2% | 87.8% |

**Critical gap:** Asaniczka has zero entry-level labels. All entry-level historical baseline comes from arshkon (~89 entry-level SWE postings).

#### Seniority distribution (Stage 8, all rows)

| seniority_final | Count | % |
|---|---|---|
| unknown | 873,804 | 71.8% |
| mid-senior | 260,021 | 21.4% |
| associate | 51,363 | 4.2% |
| director | 18,956 | 1.6% |
| entry | 13,155 | 1.1% |

#### Recommended seniority usage

**At Stage 8:**
1. `seniority_final` — primary variable, best coverage
2. `seniority_final_source` — stratify by this to understand label provenance
3. `seniority_3level` — for coarse analyses needing larger cell sizes
4. `seniority_native` — high-confidence platform-label-only analysis

**After Stage 10:**
1. `seniority_llm` — primary variable, explicit-signal-only classification
2. `seniority_final` — ablation baseline / fallback when `seniority_llm` is null
3. `seniority_native` — cross-validation anchor

#### LLM seniority design rationale

The LLM classifier looks for **explicit seniority signals only** — title keywords, level codes, and role-label language. It does NOT infer seniority from responsibilities, tech stack complexity, team size, or YOE requirements. This is by design: the research analyzes how requirements differ by seniority level, so labels must not be derived from the signals being analyzed.

---

### 5. SWE / Occupation Classification

| Column | Type | Stage | Values | Meaning |
|---|---|---|---|---|
| `is_swe` | BOOL | 5 | true/false | Primary SWE flag. True if the role's primary function is writing/maintaining software. |
| `is_swe_adjacent` | BOOL | 5 | true/false | Technical roles involving some code but not primarily software development. |
| `is_control` | BOOL | 5 | true/false | Control occupation group for cross-occupation comparisons. |
| `swe_confidence` | DOUBLE | 5 | 0.0-1.0 | Classification confidence score. |
| `swe_classification_tier` | VARCHAR | 5 | `regex`, `embedding_high`, `title_lookup_llm`, `embedding_adjacent` | Which method fired. |
| `swe_classification_llm` | VARCHAR | 10 | `SWE`, `SWE_ADJACENT`, `NOT_SWE` | LLM-based classification. Null for unrouted rows. |

**Classification tiers (Stage 5):**
- **Tier 1 (regex):** Pattern matching on title (software engineer, full-stack, DevOps, etc.). Highest precision.
- **Tier 2 (title_lookup_llm):** Curated title lookup with LLM-trained thresholds (>= 0.85 = SWE). Some elevated false-positive rate.
- **Tier 3 (embedding):** Embedding similarity. `embedding_high` for SWE, `embedding_adjacent` for SWE-adjacent.

**Exclusion rules:** Sales engineer, field service engineer, civil/mechanical/hardware engineer are excluded from SWE.

**Sample sizes (LinkedIn, English, date_flag=ok):** SWE: ~52K | SWE-adjacent: ~15K | Control: varies

---

### 6. Years of Experience (YOE)

| Column | Type | Stage | Meaning |
|---|---|---|---|
| `yoe_extracted` | DOUBLE | 5 | Primary resolved YOE floor from rule-based parser. Parsed from raw `description`. |
| `yoe_min_extracted` | DOUBLE | 5 | Minimum valid YOE mention across all accepted candidates. |
| `yoe_max_extracted` | DOUBLE | 5 | Maximum valid YOE bound across all accepted candidates. |
| `yoe_match_count` | SMALLINT | 5 | Count of accepted YOE mention candidates. |
| `yoe_resolution_rule` | VARCHAR | 5 | Which rule selected `yoe_extracted` from candidates. |
| `yoe_all_mentions_json` | VARCHAR | 5 | JSON audit trail of all candidate mentions, flags, and reject reasons. |
| `yoe_seniority_contradiction` | BOOL | 5 | True when YOE and seniority contradict (e.g., entry-level + 5 YOE). Feeds `ghost_job_risk`. |
| `yoe_min_years_llm` | INT64 | 10 | LLM-extracted binding YOE floor. **Cross-check column only** — does not drive seniority. |

The rule-based YOE extractor uses clause-aware section segmentation with multiple candidate tagging and a resolver hierarchy. `yoe_min_years_llm` is a separate LLM cross-check that exists only for ablation against the rule-based extractor.

---

### 7. Geography and Location

| Column | Type | Stage | Meaning |
|---|---|---|---|
| `location` | VARCHAR | 1 | Raw location string from posting. |
| `location_normalized` | VARCHAR | 1 | Whitespace-normalized. |
| `city_extracted` | VARCHAR | 6 | Parsed city from location string. |
| `state_normalized` | VARCHAR | 6 | Parsed and normalized state abbreviation. |
| `country_extracted` | VARCHAR | 6 | Parsed country. |
| `is_remote` | BOOL | 1 | Source-provided remote flag (normalized). |
| `is_remote_inferred` | BOOL | 6 | Inferred from location text cues ("Remote", "Anywhere", "WFH"). |
| `metro_area` | VARCHAR | 6 | Posting location aligned to 26-metro study frame. |
| `metro_source` | VARCHAR | 6 | How metro was assigned: `search_metro`, `manual_alias`, `city_state_lookup`, `unresolved`. |
| `metro_confidence` | VARCHAR | 6 | Coarse confidence for metro assignment. |
| `search_query` | VARCHAR | 1 | Scraper search query (scraped only; asaniczka: `search_position`). |
| `query_tier` | VARCHAR | 1 | Query tier from scraper design (scraped only). |
| `search_metro_id` | VARCHAR | 1 | Metro ID from scraper config (scraped only). |
| `search_metro_name` | VARCHAR | 1 | Metro name from scraper search (scraped only). **Query metadata, not posting location.** |
| `search_metro_region` | VARCHAR | 1 | Region from scraper config (scraped only). |
| `search_location` | VARCHAR | 1 | Full search location string (scraped only). |

**Key distinction:** `search_metro_name` is where we searched, not where the posting is located. Use `metro_area` for geographic analysis.

---

### 8. Time and Temporal

| Column | Type | Stage | Meaning |
|---|---|---|---|
| `date_posted` | VARCHAR | 1 | Date the posting was listed. Arshkon: epoch conversion. Asaniczka: `first_seen`. Scraped LinkedIn: 2.8% populated. Scraped Indeed: 100%. |
| `scrape_date` | VARCHAR | 1 | Date the posting was observed by scraper. Null for Kaggle sources. |
| `period` | VARCHAR | 7 | Coarse time period: `2024-01` (asaniczka), `2024-04` (arshkon), `2026-03` (scraped). **Primary temporal stratifier.** |
| `posting_age_days` | DOUBLE | 7 | Days between `date_posted` and `scrape_date`. Mainly meaningful for scraped rows with both dates. |
| `scrape_week` | DOUBLE | 7 | ISO week of `scrape_date`. Null for Kaggle. |

---

### 9. Quality Flags and Metadata

| Column | Type | Stage | Values | Meaning |
|---|---|---|---|---|
| `date_flag` | VARCHAR | 8 | `ok`, `date_posted_out_of_range` | Date validation. Parseable dates with a 2020 floor, no future-date enforcement. |
| `is_english` | BOOL | 8 | true/false | Language detection via langdetect. |
| `description_hash` | VARCHAR | 8 | SHA-256 hex | Hash of raw `description`. Provenance/lineage field. LLM caching uses task-specific `input_hash` instead. |
| `ghost_job_risk` | VARCHAR | 8 | `low`, `medium`, `high` | Rule-based ghost-job heuristic. Only entry-level rows can score above `low`. Medium: YOE >= 3 or contradiction. High: YOE >= 5. Very conservative (519 non-low out of 1.22M). |
| `description_quality_flag` | VARCHAR | 8 | `ok`, `too_short`, `empty` | Based on `description_core`. Empty if null/blank, too_short if < 50 chars. |
| `preprocessing_version` | VARCHAR | 8 | Version string | Pipeline version marker. |
| `dedup_method` | VARCHAR | 8 | — | How this row survived dedup. |
| `boilerplate_removed` | BOOL | 8 | true/false | Whether boilerplate removal was applied. |
| `is_multi_location` | BOOL | 4 | true/false | True if this posting shares a canonical opening across multiple locations. |
| `work_type` | VARCHAR | 1 | — | Job type field from source. |
| `job_url` | VARCHAR | 1 | — | Posting URL where available. |
| `skills_raw` | VARCHAR | 1 | — | Skills field from source. |
| `asaniczka_skills` | VARCHAR | 1 | — | Asaniczka-specific skills from companion join. |
| `ghost_assessment_llm` | VARCHAR | 10 | `realistic`, `inflated`, `ghost_likely` | LLM ghost-job assessment. Richer signal than rule-based. |

---

### 10. LLM Columns (Stages 9-10)

These columns are null for rows not routed to LLM processing. Rule-based columns serve as fallback.

#### Primary LLM analysis columns

| Column | Type | Stage | Values | Meaning |
|---|---|---|---|---|
| `swe_classification_llm` | VARCHAR | 10 | `SWE`, `SWE_ADJACENT`, `NOT_SWE` | LLM occupation classification. Null for rows where rule-based confidence was high. |
| `seniority_llm` | VARCHAR | 10 | 5-level enum | LLM seniority from explicit signals only. **Primary seniority variable post-LLM.** |
| `ghost_assessment_llm` | VARCHAR | 10 | `realistic`, `inflated`, `ghost_likely` | LLM ghost-job assessment. |
| `yoe_min_years_llm` | INT64 | 10 | Numeric or null | LLM-extracted YOE floor. Cross-check only. |
| `description_core_llm` | VARCHAR | 9 | Text | LLM-cleaned description. Empty string for short-description skips. |
| `selected_for_control_cohort` | BOOL | 9 | true/false | Deterministic control-cohort flag. |
| `llm_extraction_coverage` | VARCHAR | 9 | `labeled`, `deferred`, `not_routed`, `skipped_short` | Stage 9 LLM coverage status. **Filter to `labeled` when using `description_core_llm`.** |
| `llm_classification_coverage` | VARCHAR | 10 | `labeled`, `deferred`, `not_routed`, `skipped_short`, `rule_sufficient` | Stage 10 LLM coverage status. **Filter to `labeled` when using `seniority_llm`, `swe_classification_llm`, `ghost_assessment_llm`, or `yoe_min_years_llm`.** |

#### LLM extraction diagnostics

| Column | Type | Stage | Meaning |
|---|---|---|---|
| `llm_extraction_status` | VARCHAR | 9 | `ok` or `cannot_complete`. |
| `llm_extraction_validated` | BOOL | 9 | True only if extraction passed all validation checks. |
| `llm_extraction_unit_ids` | VARCHAR | 9 | JSON array of boilerplate unit IDs identified for removal. |
| `llm_extraction_uncertain_unit_ids` | VARCHAR | 9 | JSON array of uncertain unit IDs. |
| `llm_extraction_reason` | VARCHAR | 9 | Validation failure reason. |
| `llm_extraction_model_reason` | VARCHAR | 9 | LLM's own short reason phrase. |
| `llm_extraction_units_count` | INT64 | 9 | Total sentence-unit count for the description. |
| `llm_extraction_single_unit` | BOOL | 9 | True if description had only one unit (cannot extract). |
| `llm_extraction_drop_ratio` | DOUBLE | 9 | Fraction of characters dropped as boilerplate. |

#### LLM provenance

| Column | Type | Stage | Meaning |
|---|---|---|---|
| `llm_model_classification` | VARCHAR | 10 | Model used for classification (e.g., `gpt-5.4-mini`, `haiku`). |
| `llm_model_extraction` | VARCHAR | 9 | Model used for extraction. |
| `llm_prompt_version_classification` | VARCHAR | 10 | SHA-256 hash of classification prompt template. |
| `llm_prompt_version_extraction` | VARCHAR | 9 | SHA-256 hash of extraction prompt template. |

#### LLM routing rules

**Extraction (Stage 9):** Routes rows that are LinkedIn, English, have a raw description, and are SWE/SWE-adjacent/selected-control. Hard-skips descriptions under 15 words.

**Classification (Stage 10):** Skips LLM classification when all hold:
- `swe_classification_tier` in {`regex`, `embedding_high`, `title_lookup_llm`}
- `seniority_source` starts with `title_`
- `ghost_job_risk == "low"`

For skipped rows, LLM columns remain null and rule-based columns serve as the analysis values.

#### Budget-Constrained LLM Processing

Stages 9 and 10 require an explicit `--llm-budget` parameter (no default). This caps the number of **new** LLM calls per run across all data sources (Kaggle and scraped alike).

**Category split (default 40/30/30):**
Budget is split across three categories via `--llm-budget-split swe,swe_adjacent,control`:
- SWE: 40% (primary study target)
- SWE-adjacent: 30%
- Control: 30%

If a category has fewer uncached rows than its share, the surplus cascades to the other categories proportionally to their shares.

**Daily water-filling:**
Within each category, budget is distributed across `scrape_date` (YYYY-MM-DD) buckets using water-filling — the least-covered days get budget first. This prevents temporal bias at daily granularity. Within-day selection is deterministic (SHA-256 hash) for reproducibility.

**Coverage tracking:**
- `llm_extraction_coverage` (Stage 9) and `llm_classification_coverage` (Stage 10) track whether each row has LLM results.
- Values: `labeled` (has results), `deferred` (eligible but budget-capped), `not_routed` (not eligible), `skipped_short` (< 15 words), `rule_sufficient` (Stage 10 only, rules confident enough).
- **Downstream analyses using LLM columns must filter to `*_coverage == 'labeled'`.**

**Historical skew (accepted):**
Early scrape dates were processed with the old "label everything" behavior and have dense LLM coverage. Water-filling naturally routes new budget to less-covered (recent) days. Older days retain their dense coverage — the baseline skew is permanent and accepted.

**Incremental runs:**
Each run adds to the cache. Re-running with a higher budget selects a deterministic superset. Running with budget=0 is valid (uses only existing cache, no new calls).

---

## Source-Specific Field Availability

Not all columns are populated across all sources. Key gaps:

| Field | arshkon | asaniczka | scraped (LinkedIn) | scraped (Indeed) |
|---|---|---|---|---|
| `description` | 100% | 96.2% (join) | 100% | 100% |
| `seniority_native` | 66.5% | 100% | 100% | 0% |
| `date_posted` | 100% | 100% | 2.8% | 100% |
| `company_industry` | 99.3% (join) | 0% | 100% | 0% |
| `company_size` | via companion | 0% | 0% | 91% |
| Entry-level labels | Yes (~385 SWE) | **None** | Yes | N/A (no seniority) |
| `search_query` | 0% | `search_position` | 100% | 100% |
| `skills_raw` | available | 0% | available | available |
| `asaniczka_skills` | 0% | from companion | 0% | 0% |

---

## Important Caveats

1. **Seniority is the weakest Stage 8 link.** 71.8% unknown rate means seniority-stratified analyses use < 29% of rows. This improves with `seniority_llm`.

2. **Entry-level historical baseline is thin.** Only arshkon has entry-level labels, yielding ~89 entry-level SWE postings. This is the binding constraint for RQ1 junior-share analysis.

3. **Boilerplate removal is noisy.** `description_core` has ~44% accuracy. For keyword/pattern analyses, raw `description` may be preferable. `description_core_llm` is the intended upgrade.

4. **unified.parquet contains all columns.** As of the current pipeline run, `data/unified.parquet` includes all Stage 8 columns plus LLM additions. There is no need to work from intermediate stage files.

5. **Ghost detection is conservative.** Only 519 non-low `ghost_job_risk` rows out of 1.22M. Use `ghost_assessment_llm` for a richer signal.

6. **Row counts grow daily.** Each day of scraping adds ~34K rows before dedup. Re-run from Stage 1 after syncing new data.

7. **`description_hash` is a lineage field.** It hashes the raw description for provenance. LLM caching uses task-specific `input_hash` values, not `description_hash`.
