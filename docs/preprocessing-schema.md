# Preprocessing Schema Reference

Last updated: 2026-04-10

Complete column reference for the preprocessing pipeline. For architecture, operations, and development practices, see [`preprocessing-guide.md`](preprocessing-guide.md).

---

## How to Use This Document

**Primary analysis file:** `data/unified.parquet`. This is the final analysis output containing rule-based columns plus LLM columns from Stages 9-10. Query the current artifact for row and column counts instead of relying on documented totals.

**LLM column usage (updated 2026-04-10):**
- `description_core_llm`: **The only cleaned-text column** and the required input for every text-dependent analysis. Check `llm_extraction_coverage` for coverage by source. Raw `description` is the only acceptable fallback and only for analyses that are insensitive to boilerplate (e.g., binary keyword presence); text-sensitive work must filter to `llm_extraction_coverage = 'labeled'`.
- **Seniority — use `seniority_final`.** This is the single primary seniority column. Stage 5 fills it from high-confidence title keywords; Stage 10 overwrites it with the LLM result for rows the router sent to the LLM. `seniority_final_source` records which path produced the value (`title_keyword`, `title_manager`, `llm`, or `unknown`). See Section 4 for details.
- `ghost_assessment_llm`: Primary ghost indicator (`realistic`/`inflated`/`ghost_likely`). Richer than rule-based `ghost_job_risk`. Use `ghost_job_risk` as fallback.
- `swe_classification_llm`, `yoe_min_years_llm`: Cross-check columns.

Always check `llm_extraction_coverage` and `llm_classification_coverage` to confirm which rows have LLM results. For raw Stage 9/Stage 10 LLM columns (including `description_core_llm`), filter to `labeled`. Seniority is the exception: `seniority_final` is already the combined best-available column and should be used directly without filtering by coverage.

Balanced-sample claims apply only to `selected_for_llm_frame = true`. Supplemental cache rows can extend the usable LLM set, but they are not part of the balanced core frame.

---

## Column Availability by Stage

This table shows when each column category first becomes available:

| Category | First available | Columns | Key additions |
|---|---|---|---|
| Identity & provenance | Stage 1 | 5 | `uid`, `source`, `source_platform` |
| Raw job content | Stage 1 | 6 | `title`, `description`, `description_raw` |
| Company (basic) | Stage 1 | 5 | `company_name`, `company_industry`, `company_size` |
| Aggregator handling | Stage 2 | 3 | `is_aggregator`, `real_employer`, `company_name_effective` |
| Company canonicalization | Stage 4 | 2 | `company_name_canonical`, `company_name_canonical_method` |
| Multi-location flag | Stage 4 | 1 | `is_multi_location` |
| SWE classification | Stage 5 | 5 | `is_swe`, `is_swe_adjacent`, `is_control`, `swe_confidence`, `swe_classification_tier` |
| Seniority | Stage 1 + 5 + 10 | 4 | `seniority_native` (Stage 1), `seniority_final`, `seniority_final_source`, `seniority_3level` |
| YOE extraction | Stage 5 | 7 | `yoe_extracted`, `yoe_min_extracted`, `yoe_seniority_contradiction`, etc. |
| Location parsing | Stage 6 | 6 | `city_extracted`, `state_normalized`, `metro_area`, `is_remote_inferred` |
| Temporal derivations | Stage 7 | 3 | `period`, `posting_age_days`, `scrape_week` |
| Quality flags | Stage 8 | 5 | `date_flag`, `is_english`, `ghost_job_risk`, `description_quality_flag` |
| Pipeline metadata | Stage 8 | 2 | `preprocessing_version`, `dedup_method` |
| LLM frame + cleaned text | Stage 9 | 5 | `description_core_llm`, `selected_for_llm_frame`, `selection_date_bin`, `selected_for_control_cohort`, `llm_extraction_sample_tier` |
| LLM coverage tracking | Stage 9-10 | 4 | `llm_extraction_coverage`, `llm_extraction_resolution`, `llm_classification_coverage`, `llm_classification_resolution` |
| LLM classification | Stage 10 | 4 | `swe_classification_llm`, `ghost_assessment_llm`, `yoe_min_years_llm`, `llm_classification_sample_tier` (LLM seniority writes back to `seniority_final`; there is no separate `seniority_llm` column) |

Row and column counts change as scraped data grows. Query the current artifacts before analysis. The 2026-04-10 removal of the rule-based boilerplate stage dropped `description_core`, `core_length`, `boilerplate_flag`, and `boilerplate_removed`.

---

## What to Use at Each Processing Milestone

### At Stage 8+ (rule-based baseline)

Work from `data/unified.parquet` (the final output with all rule-based and available LLM columns).

| Analysis need | Primary column | Ablation / fallback | Notes |
|---|---|---|---|
| SWE sample | `is_swe` | `swe_classification_llm` where labeled | `is_swe_adjacent` for broader tech sample. |
| Seniority | `seniority_final` | `seniority_native` (arshkon-only diagnostic), YOE-based proxy | `seniority_final` is the combined high-confidence rule + LLM column. Always validate entry-level findings with the YOE-based proxy. |
| Seniority (coarse) | `seniority_3level` | — | junior/mid/senior/unknown |
| Time period | `period` | `date_posted`, `scrape_date` | Query current `period` values and source date ranges before temporal analysis. |
| Description text | `description_core_llm` | raw `description` (only when LLM cleaned text is unavailable and the analysis is boilerplate-insensitive) | No rule-based cleaned text exists. `description_core` was retired on 2026-04-10. |
| Company | `company_name_effective` | `company_name` | Resolves aggregators |
| Company (grouped) | `company_name_canonical` | `company_name_effective` | For grouping across spelling variants |
| Geography | `metro_area` | `state_normalized` | 26-metro study frame |
| Remote status | `is_remote_inferred` | `is_remote` | Combines source flag + text inference |
| Ghost/inflation | `ghost_assessment_llm` | `ghost_job_risk` | Rule-based is conservative. LLM is richer. |
| Aggregator filter | `is_aggregator` | — | Aggregator rows |

**Default filters for most analyses:**

```sql
WHERE source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
```

### At Stage 9 (extraction complete, classification pending)

Work from `preprocessing/intermediate/stage9_llm_cleaned.parquet`. Same row count as Stage 8.

New columns available:
- `description_core_llm`: LLM-cleaned description. Use as primary text for analysis. When null/empty, raw `description` is the only fallback — and only when the analysis is not boilerplate-sensitive.
- `selected_for_llm_frame`: Whether this row is in the deterministic Stage 9 selected core frame.
- `selection_date_bin`: Source-specific balancing date for the selected core frame.
- `selected_for_control_cohort`: Compatibility-only mirror for legacy consumers; equivalent to `selected_for_llm_frame & is_control`.

### At Stage 10 (LLM classification complete)

Work from `preprocessing/intermediate/stage10_llm_integrated.parquet` or `data/unified.parquet`.

| Analysis need | Primary column | Fallback / ablation |
|---|---|---|
| SWE sample | `swe_classification_llm` (for routed rows) | `is_swe` |
| Seniority | `seniority_final` | `seniority_native` (arshkon-only), YOE-based proxy |
| Clean text | `description_core_llm` | raw `description` (boilerplate-insensitive analyses only) |
| Ghost / inflation | `ghost_assessment_llm` | `ghost_job_risk` |

LLM columns are null for rows the LLM did not process. **Seniority is the exception:** `seniority_final` is always populated (from a strong rule, from the LLM, or `'unknown'`) and should be used directly without coverage filtering. For `swe_classification_llm` and `ghost_assessment_llm`, filter on `llm_classification_coverage = 'labeled'` and fall back to the corresponding rule-based columns (`is_swe`, `ghost_job_risk`) where the LLM was not called. For `description_core_llm`, filter on `llm_extraction_coverage = 'labeled'`; there is no rule-based cleaned-text fallback — analyses that need cleaned text for unlabeled rows must either restrict the sample or accept raw `description`.

---

## Source Composition

| Source | Platform | Temporal role | Notes |
|---|---|---|---|
| kaggle_asaniczka | linkedin | Historical snapshot | Large volume; no native entry labels. |
| kaggle_arshkon | linkedin | Historical snapshot | Has native entry-level labels. |
| scraped | linkedin | Growing current window | Primary current comparison source. |
| scraped | indeed | Growing current window | Sensitivity analyses only. |

Query `data/unified.parquet` for current row counts, date ranges, and SWE rows before analysis.

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
| `description_core_llm` | VARCHAR | 9 | LLM-based boilerplate removal. Reconstructed from validated extraction output. Empty string for short-description hard skips (< 15 words). **The only cleaned-text column in the pipeline.** Filter on `llm_extraction_coverage = 'labeled'` to identify rows where it is populated. |

**Recommended text column priority:**
1. `description_core_llm` (where `llm_extraction_coverage = 'labeled'`) — the only boilerplate-removed text column.
2. `description` (raw) — full text including boilerplate. Acceptable only for analyses that are insensitive to boilerplate (e.g., binary keyword presence) or when falling back because LLM coverage is absent.

The rule-based `description_core` column (and its `core_length` / `boilerplate_flag` siblings) was retired on 2026-04-10 because the regex-based extractor performed at ~44% accuracy and was causing downstream agents to mix rule-based and LLM text.

**Usage guidance:** For binary keyword presence (does the posting mention X?), raw `description` is acceptable and may improve recall. For density/frequency metrics (mentions per 1K chars), embeddings, topic modeling, and corpus comparison, use `description_core_llm` — restrict the sample to rows where it is labeled rather than backfilling with raw text.

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

There are **4 seniority columns**. The schema was simplified on 2026-04-10 — `seniority_final` is now the single primary column, combining high-confidence rule-based labels with LLM-classified labels into one materialized value. Previous columns `seniority_raw`, `seniority_imputed`, `seniority_source`, `seniority_confidence`, `seniority_final_confidence`, `seniority_cross_check`, and `seniority_llm` were removed; their roles are now either subsumed by `seniority_final` or no longer needed.

#### Seniority enum values

All seniority columns use a canonical 5-level enum: `entry`, `associate`, `mid-senior`, `director`, `unknown`.

The coarse 3-level mapping (`seniority_3level`): entry → junior, associate → mid, mid-senior → senior, director → senior.

#### Column inventory

| Column | Type | Stage | Values | Meaning |
|---|---|---|---|---|
| `seniority_native` | VARCHAR | 1 | 5-level enum or null | Platform-provided label mapped to canonical enum. **Diagnostic only** — used as a label-independence check and as the arshkon-only baseline for entry-level analyses. Not the primary analysis column. Asaniczka: only `mid-senior`/`associate`; Indeed: null. |
| `seniority_final` | VARCHAR | 5/10 | 5-level enum | **Primary seniority column.** Stage 5 sets it from high-confidence title keywords; Stage 10 overwrites it with the LLM result for rows the router sent to the LLM. `'unknown'` for rows where neither signal fired. |
| `seniority_final_source` | VARCHAR | 5/10 | `title_keyword`, `title_manager`, `llm`, `unknown` | How `seniority_final` was resolved. Use this to filter for rule-only or LLM-only subsets when needed. |
| `seniority_3level` | VARCHAR | 5/10 | `junior`, `mid`, `senior`, `unknown` | Coarse 3-level collapse of `seniority_final`. Convenience column for stratification. |

#### How `seniority_final` is resolved

**Stage 5 — rule-based pass.** Stage 5 inspects the title for explicit strong-seniority keywords (junior, senior, lead, principal, staff, director, vp, etc.) and explicit manager indicators. If a strong rule fires, `seniority_final` is set to that level and `seniority_final_source` is set to `title_keyword` or `title_manager`. Otherwise `seniority_final = 'unknown'` and `seniority_final_source = 'unknown'`. Stage 5 deliberately does not consult `seniority_native`, weak title patterns, or description text — those signals were judged unreliable for the primary analysis variable and were removed during the 2026-04-10 simplification.

**Stage 10 — LLM pass.** The Stage 10 router sends a row to the LLM when ALL hold:
- `swe_classification_tier ∈ {regex, embedding_high, title_lookup_llm}` (strong SWE classification)
- `seniority_final = 'unknown'` (Stage 5 found no strong rule)
- `ghost_job_risk = 'low'`

For routed rows (`llm_classification_coverage = 'labeled'`), the LLM result overwrites `seniority_final` and `seniority_final_source` is set to `'llm'`. For rows the router skipped because Stage 5 already produced a strong-rule label (`llm_classification_coverage = 'rule_sufficient'`), `seniority_final` keeps its Stage 5 value. For rows outside the LLM frame, `seniority_final` stays as Stage 5 wrote it (often `'unknown'`).

**Implication for `seniority_final_source = 'llm'`:** the LLM may return any of the 5 enum values, including `'unknown'` when no explicit signal is present in the description. A row with `seniority_final_source = 'llm'` AND `seniority_final = 'unknown'` means the LLM was called and could not determine seniority — this is correct, expected behavior, not a defect. See "LLM seniority design rationale" below.

#### Recommended seniority usage

Use `seniority_final` as the primary seniority column for any seniority-stratified analysis. Do not filter by `llm_classification_coverage` first — `seniority_final` already incorporates both the LLM and rule-based halves.

For label-independence validation (required for any entry-level finding), use:

1. **YOE-based proxy:** share of postings with `yoe_extracted ≤ 2` by period, plus the YOE distribution by period. This is the strongest fully label-independent check, since it does not depend on any seniority labeler.
2. **`seniority_native` (arshkon-only):** the platform's own label, available as a sanity check for entry-level baselines on arshkon. **Do not pool asaniczka into a `seniority_native`-based comparison** — asaniczka has zero native entry-level labels and would dilute the entry rate to near-zero.

If `seniority_final` and the YOE-based proxy disagree on the direction of an entry-level trend, do not pick a side without investigating. Possible explanations include: real market change, differential native-label quality across snapshots, shifts in employer labeling explicitness, compositional change in the unknown pool, or instrument noise. Report disagreement honestly — material disagreement is itself a finding, not a problem to hide.

#### Asaniczka caveat

Asaniczka has zero native entry-level labels — its `seniority_native` distribution contains only `mid-senior` and `associate`. Under the new schema, asaniczka rows can still receive entry-level labels in `seniority_final` via the LLM (for rows that route to Stage 10). However, `seniority_native` cannot detect entry-level postings in asaniczka by construction. For any sanity check that uses `seniority_native`, use arshkon only.

#### LLM seniority design rationale

The LLM classifier looks for **explicit seniority signals only** — title keywords, level codes, and role-label language. It does NOT infer seniority from responsibilities, tech stack complexity, team size, or YOE requirements. This is by design: the research analyzes how requirements differ by seniority level, so labels must not be derived from the signals being analyzed. A consequence is that the LLM classifies many routed rows as `'unknown'` when explicit signals are absent — this is correct behavior, not a defect.

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
| `period` | VARCHAR | 7 | Coarse source/window bucket derived in Stage 7. Query current distinct values before temporal analysis; scraped values evolve as new data is ingested. |
| `posting_age_days` | DOUBLE | 7 | Days between `date_posted` and `scrape_date`. Mainly meaningful for scraped rows with both dates. |
| `scrape_week` | DOUBLE | 7 | ISO week of `scrape_date`. Null for Kaggle. |

---

### 9. Quality Flags and Metadata

| Column | Type | Stage | Values | Meaning |
|---|---|---|---|---|
| `date_flag` | VARCHAR | 8 | `ok`, `date_posted_out_of_range` | Date validation. Parseable dates with a 2020 floor, no future-date enforcement. |
| `is_english` | BOOL | 8 | true/false | Language detection via langdetect. |
| `description_hash` | VARCHAR | 8 | SHA-256 hex | Hash of raw `description`. Provenance/lineage field. LLM caching uses task-specific `input_hash` instead. |
| `ghost_job_risk` | VARCHAR | 8 | `low`, `medium`, `high` | Rule-based ghost-job heuristic. Only entry-level rows can score above `low`. Medium: YOE >= 3 or contradiction. High: YOE >= 5. Conservative fallback for `ghost_assessment_llm`. |
| `description_quality_flag` | VARCHAR | 8 | `ok`, `too_short`, `empty` | Based on the raw `description`. Empty if null/blank, too_short if < 50 chars. |
| `preprocessing_version` | VARCHAR | 8 | Version string | Pipeline version marker. |
| `dedup_method` | VARCHAR | 8 | — | How this row survived dedup. |
| `is_multi_location` | BOOL | 4 | true/false | True on the surviving representative of a collapsed multi-location group (rows sharing `company_name_canonical + title + description_hash` across 2+ distinct locations). Stage 4 keeps the lowest-`uid` row as the representative and drops the others. The representative's `location` is overwritten to `"multi-location"` and `search_metro_name`/`search_metro_id`/`search_metro_region`/`search_location` are cleared to null so Stage 6 cannot re-attribute the row to a single metro. Representatives end up with `metro_area = NULL` after Stage 6 and are naturally excluded from per-metro rollups — this is correct behavior because the posting does not belong to any single metro. |
| `work_type` | VARCHAR | 1 | — | Job type field from source. |
| `job_url` | VARCHAR | 1 | — | Posting URL where available. |
| `skills_raw` | VARCHAR | 1 | — | Skills field from source. |
| `asaniczka_skills` | VARCHAR | 1 | — | Asaniczka-specific skills from companion join. |
| `ghost_assessment_llm` | VARCHAR | 10 | `realistic`, `inflated`, `ghost_likely` | LLM ghost-job assessment. Richer signal than rule-based. |

---

### 10. LLM Columns (Stages 9-10)

These columns are null for rows not routed to LLM processing. Rule-based columns serve as fallback. **Seniority is the exception:** LLM seniority writes back to `seniority_final` (see Section 4) — there is no separate `seniority_llm` column.

#### LLM analysis columns

| Column | Type | Stage | Values | Meaning |
|---|---|---|---|---|
| `swe_classification_llm` | VARCHAR | 10 | `SWE`, `SWE_ADJACENT`, `NOT_SWE` | LLM occupation classification. Null for rows where rule-based confidence was high. |
| `ghost_assessment_llm` | VARCHAR | 10 | `realistic`, `inflated`, `ghost_likely` | LLM ghost-job assessment. |
| `yoe_min_years_llm` | INT64 | 10 | Numeric or null | LLM-extracted YOE floor. Cross-check only. |
| `description_core_llm` | VARCHAR | 9 | Text | LLM-cleaned description. Empty string for short-description skips. |
| `selected_for_llm_frame` | BOOL | 9-10 | true/false | Deterministic core-frame flag propagated from Stage 9. Marks the sticky balanced core only. |
| `selected_for_control_cohort` | BOOL | 9-10 | true/false | Compatibility-only mirror for legacy consumers. Equivalent to `selected_for_llm_frame & is_control` when the legacy flag is still needed. |
| `llm_extraction_sample_tier` | VARCHAR | 9 | `core`, `supplemental_cache`, `none` | Stage 9 extraction sample tier. `core` rows belong to the sticky balanced frame. |
| `llm_classification_sample_tier` | VARCHAR | 10 | `core`, `supplemental_cache`, `none` | Stage 10 classification sample tier. May differ row-by-row from Stage 9. |
| `llm_extraction_coverage` | VARCHAR | 9 | `labeled`, `deferred`, `not_selected`, `skipped_short` | Stage 9 LLM coverage status. **Filter to `labeled` when using `description_core_llm`.** |
| `llm_extraction_resolution` | VARCHAR | 9 | `cached_llm`, `fresh_llm`, `deferred`, `not_selected`, `skipped_short` | Stage 9 resolution method. Separates cache hits from fresh calls. |
| `llm_classification_coverage` | VARCHAR | 10 | `labeled`, `deferred`, `not_selected`, `skipped_short`, `rule_sufficient` | Stage 10 LLM coverage status. **Filter to `labeled` when using raw LLM columns; include `rule_sufficient` only if you explicitly report that best-available Stage 10 choice.** |
| `llm_classification_resolution` | VARCHAR | 10 | `cached_llm`, `fresh_llm`, `rule_sufficient`, `deferred`, `not_selected`, `skipped_short` | Stage 10 resolution method. Keeps rule-based resolution separate from cached and fresh LLM calls. |

#### LLM routing rules

**Extraction (Stage 9):** Routes rows that are LinkedIn, English, have a raw description, and are in the Stage 9 selected core frame. Hard-skips descriptions under 15 words.

**Classification (Stage 10):** Skips LLM classification when all hold:
- `swe_classification_tier` in {`regex`, `embedding_high`, `title_lookup_llm`}
- `seniority_final != 'unknown'` (Stage 5 already set a strong rule-based seniority)
- `ghost_job_risk == "low"`

For skipped rows (`llm_classification_coverage = 'rule_sufficient'`), `seniority_final` keeps its Stage 5 value, and the other LLM columns (`swe_classification_llm`, `ghost_assessment_llm`, `yoe_min_years_llm`) remain null with rule-based columns serving as the analysis values. For routed rows, the LLM seniority result overwrites `seniority_final` and `seniority_final_source = 'llm'` (see Section 4). Rows outside the inherited Stage 9 core frame are `not_selected`. A row may have usable Stage 9 text without Stage 10 classification, or vice versa.

#### Budget-Constrained LLM Processing

Stages 9 and 10 require an explicit `--llm-budget` parameter (no default). This caps the number of **new** LLM calls per run across all data sources (Kaggle and scraped alike).

**Selection frame vs budget:**
Stage 9 accepts an optional `--selection-target` for the core-frame size. When omitted, it defaults to `--llm-budget`. Stage 10 inherits the Stage 9 core frame and does not have a separate selection target. Supplemental cache rows can expand the usable LLM set, but they do not change the balanced core frame.

**Category split (default 40/30/30):**
Fresh-call budget is split across three categories via `--llm-budget-split swe,swe_adjacent,control`:
- SWE: 40% (primary study target)
- SWE-adjacent: 30%
- Control: 30%

If a category has fewer uncached rows than its share, the surplus cascades to the other categories proportionally to their shares.

**Fresh-call selection:**
Stage 9 first selects and persists the sticky core frame across `source × analysis_group × date_bin`. Stage 10 inherits that frame. For each task, cached selected rows are resolved without consuming budget; the fresh-call budget is split across unresolved selected rows by `analysis_group` via `split_budget_by_category`; each group then calls `select_fresh_call_tasks(...)`, which applies the same deterministic frame selector within that group. Supplemental cache rows can expand usable coverage, but they do not change the selected core frame or consume fresh-call budget.

**Coverage tracking:**
- `llm_extraction_coverage` (Stage 9) and `llm_classification_coverage` (Stage 10) track whether each row has LLM results.
- Values: `labeled` (has results), `deferred` (eligible but budget-capped), `not_selected` (not in the Stage 9/10 frame), `skipped_short` (< 15 words), `rule_sufficient` (Stage 10 only, rules confident enough).
- Raw LLM columns should be filtered to `*_coverage == 'labeled'`. If you include `rule_sufficient` in Stage 10 analysis, report that explicitly and keep it separate from `labeled`.

**Incremental runs:**
Each run adds to the cache. Re-running with a higher `selection_target` selects a deterministic superset core frame. Re-running with a higher `llm-budget` increases fresh-call usage within that same frame. Running with budget=0 is valid (uses only existing cache, no new calls). Stage 9 and Stage 10 caches are separate, so row-level coverage can differ across stages.

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
| Entry-level labels | Yes | **None** | Yes | N/A (no seniority) |
| `search_query` | 0% | `search_position` | 100% | 100% |
| `skills_raw` | available | 0% | available | available |
| `asaniczka_skills` | 0% | from companion | 0% | 0% |

---

## Important Caveats

1. **Seniority requires label-independent validation.** Use `seniority_final` as the primary seniority column — Stage 5 fills it from high-confidence title keywords and Stage 10 overwrites it with the LLM result for routed rows. The Stage 8 unknown rate is high because Stage 5 only fires on strong title keywords; the LLM closes most of the gap within the selected core frame. Always cross-check seniority-stratified findings against the label-independent YOE-based proxy. Differential native-label quality across data snapshots is a known risk; if any seniority-stratified finding disagrees with the YOE-based proxy on the direction of an entry-level trend, report the disagreement.

2. **Entry-level baselines are limited and source-dependent.** arshkon is the only 2024 source with native entry labels in `seniority_native`. Asaniczka has none. Use arshkon as the 2024 baseline for any sanity check that uses `seniority_native`. `seniority_final` and the YOE-based proxy can include asaniczka, since they do not depend on native labels — but asaniczka entry counts in `seniority_final` come entirely from the LLM (where it routed asaniczka rows) and may be small.

3. **Boilerplate removal is LLM-only.** `description_core_llm` is the sole cleaned-text column; there is no rule-based equivalent (the former `description_core` was retired on 2026-04-10). For text-sensitive analysis, filter to `llm_extraction_coverage = 'labeled'`. For binary keyword presence where boilerplate is irrelevant, raw `description` is acceptable.

4. **unified.parquet contains all columns.** As of the current pipeline run, `data/unified.parquet` includes all Stage 8 columns plus LLM additions. There is no need to work from intermediate stage files.

5. **Ghost detection is conservative.** Use `ghost_assessment_llm` for a richer signal and `ghost_job_risk` as a fallback.

6. **Row counts grow over time.** Re-run from Stage 1 after syncing new scraped data.

7. **`description_hash` is a lineage field.** It hashes the raw description for provenance. LLM caching uses task-specific `input_hash` values, not `description_hash`.
