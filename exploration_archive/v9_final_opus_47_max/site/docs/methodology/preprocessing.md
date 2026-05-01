# Preprocessing pipeline

## Why preprocessing exists at all

The three input collections do not share a schema. Those collections are the 2024 arshkon dataset (~5,000 LinkedIn postings from April 2024), the 2024 asaniczka dataset (~18,000 LinkedIn postings from January 2024), and our 2026 scrape (~45,000 LinkedIn and Indeed postings). They use different field names, cover different subsets of company metadata, and use different conventions for seniority labels and description formatting. An analyst who loaded them side by side would not be able to write a single query against the three of them.

Preprocessing turns all three into one analysis-ready table with:

- A unified 39-column schema keyed by a stable posting ID.
- Occupation, seniority, years-of-experience, and location classifications derived the same way across sources.
- A cleaned description text field with boilerplate (benefits language, EEO disclaimers, company overviews) removed, for any analysis that reads the posting text.
- Provenance flags so every row can be traced back to its original source and any apparent "change" can be separated from platform drift.

Without this step, cross-period comparison is meaningless: an apparent shift in seniority language could just be LinkedIn changing its label taxonomy between 2024 and 2026.

## Pipeline overview

Ten stages, organized in two layers:

```
Raw Data (3 sources)
  |
  +-- Stage 1: Ingest & Schema Unification
  |     +-- 1a: arshkon ingest + companion joins
  |     +-- 1b: asaniczka ingest + description/skills joins
  |     +-- 1c: 2026 scrape ingest (LinkedIn + Indeed)
  |     +-- 1d: Concatenation to canonical 39-column format
  +-- Stage 2: Aggregator / Staffing Handling
  +-- Stage 4: Company Canonicalization + Dedup
  +-- Stage 5: Occupation + Seniority Classification
  +-- Stage 6: Location Normalization       --+
  +-- Stage 7: Temporal Alignment             | (single script)
  +-- Stage 8: Quality Flags & Provenance   --+
  |
  V
  stage8_final.parquet   <-- rule-based baseline
  |
  +-- Stage 9: LLM Boilerplate Removal
  +-- Stage 10: LLM Classification + Final Integration
  |
  V
  data/unified.parquet            (one row per unique posting; full schema)
  data/unified_observations.parquet  (daily panel: posting x scrape_date)
  data/unified_core.parquet       (analysis-ready subset)
```

Stage 3 is deliberately absent. It existed in an earlier draft and was removed because it added no useful signal; the numbering is preserved for traceability.

## Two layers, different trust profiles

**Stages 1 through 8 are rule-based.** Deterministic, fast (roughly 30 minutes end-to-end), reproducible from the raw sources with no API calls. The output of stage 8 is already a usable corpus with regex-based and lookup-based labels, and this stage can be run on its own.

**Stages 9 and 10 use a language model** (GPT-5.4-mini by default, with Claude and OpenAI available as alternate engines). These stages take hours to days depending on API quotas. The rule-based columns from stages 1 through 8 are preserved alongside the LLM columns as fallbacks and as cache keys.

Only stage 4 reduces the row count (deduplication). Every other stage keeps the row count the same.

## What each stage does

| Stage | What it does | Why |
|---|---|---|
| **1 — Ingest** | Loads the three sources, applies source-specific joins (companion tables, description and skills joins, platform-specific fields), concatenates to a canonical 39-column layout. | Different sources carry different fields. Downstream code needs a single surface. |
| **2 — Aggregators** | Identifies staffing agencies and job aggregators (Dice, Lensa, Robert Half, and similar) by exact name and regex; extracts the real employer from the description text; writes an effective-company-name column. | Aggregators re-post other firms' jobs; if left in, they distort every company-level analysis. |
| **4 — Dedup** | The one row-reducing stage. Canonicalizes company names; removes exact duplicates, fuzzy near-duplicates (shared-token ratio of 85% or higher), and multi-location collapses of the same posting. | The same posting can appear multiple times across scrape runs or across city/state variants. The study needs one row per unique posting. |
| **5 — Classification** | Writes the software-engineering flag, the adjacent-role flag, and the control-occupation flag using a three-tier system (regex, curated title lookup, embedding similarity). Extracts years-of-experience from text with a clause-aware parser. Writes an immutable rule-based seniority label alongside its source. | First analytical boundary. Every downstream stratification depends on these flags. |
| **6-8 — Normalize** | One script covering location parsing (city, state, country, remote), temporal alignment (period, posting age), and quality flags (language detection, date validation, ghost-job heuristics). | The study needs comparable geography, time, and quality fields across three differently-shaped sources. |
| **9 — LLM extraction** | Selects a deterministic "core" frame of postings over source, analysis group, and date bin. Segments each description into numbered sentence-level units; the language model identifies which units are boilerplate and should be dropped. Produces the cleaned description column, the **only** cleaned-text field in the dataset. Descriptions under 15 words are skipped. | Boilerplate (benefits, EEO language, company overview) inflates length and adds noise. Any analysis that reads the posting text needs the boilerplate stripped. |
| **10 — LLM classification** | Reuses the same core frame. Sends each eligible posting to the language model for four tasks at once: software-engineering classification, seniority, a ghost-job assessment, and years-of-experience extraction. The LLM seniority result overwrites the rule-based seniority when not null. | Rule-based seniority misses 34 to 53% of rows. The LLM, by design, abstains on unclear cases too, producing a large "unknown" pool rather than fabricating labels. Within the frame it covers, the LLM's years-of-experience extraction is the primary signal the study uses. |

## What comes out

Four parquet files:

| File | Unit of observation | Purpose |
|---|---|---|
| `unified.parquet` | one row per unique posting | Full schema; used for audits and for rows outside the LLM frame |
| `unified_observations.parquet` | one row per posting per scrape date | Full-schema daily panel |
| **`unified_core.parquet`** | one row per unique posting | **Default analysis dataset.** Restricted to postings selected for the LLM frame; curated column set. |
| `unified_core_observations.parquet` | one row per posting per scrape date | Daily panel for the core subset |

### What is in `unified_core.parquet`

The columns are grouped by purpose:

| Group | Columns |
|---|---|
| Identity | `uid` |
| Source and time | `source`, `source_platform`, `period`, `date_posted`, `scrape_date` |
| Job content | `title`, `description`, `description_core_llm`, `description_length` |
| Company | `company_name`, `company_name_effective`, `company_name_canonical`, `is_aggregator`, `company_industry`, `company_size` |
| Occupation | `is_swe`, `is_swe_adjacent`, `is_control`, `analysis_group`, `swe_classification_tier`, `swe_classification_llm` |
| Seniority | `seniority_final`, `seniority_final_source`, `seniority_3level`, `seniority_rule`, `seniority_rule_source`, `seniority_native` |
| Years of experience | `yoe_extracted`, `yoe_min_years_llm` |
| Geography | `location`, `city_extracted`, `state_normalized`, `metro_area`, `is_remote_inferred`, `is_multi_location` |
| Quality | `is_english`, `date_flag`, `ghost_job_risk`, `ghost_assessment_llm` |
| LLM coverage | `llm_extraction_coverage`, `llm_classification_coverage` |

### Where the sources disagree

The three sources do not all carry the same fields. This table shows the coverage gap that motivates reporting pooled and arshkon-only magnitudes side by side:

| Field | arshkon | asaniczka | 2026 scrape (LinkedIn) |
|---|---|---|---|
| `description` | 100% | 96.2% (joined) | 100% |
| Native seniority label | 66.5% | **100%, but no 'entry' level** | 100% |
| Posted date | 100% | 100% | 2.8% |
| Company industry | 99.3% (joined) | 0% | 100% |
| Company size | joined | 0% | 0% |
| Entry-level native labels | yes | **none** | yes |

asaniczka carries no native entry-level labels at all. That is the single most important reason the study uses years-of-experience floors (J3 and S4) as its primary seniority definitions rather than LinkedIn's own labels, and why the thirteen-definition robustness panel exists: the study needs to know that findings do not depend on which definition you pick.

## The LLM stages: where the prompts live

The exact prompts sent in stages 9 and 10 are on the [LLM prompts page](llm-prompts.md). Prompts are versioned by a hash of their content; any edit forces the cache to re-resolve against the new version.

## Budget and coverage

Stages 9 and 10 take an explicit budget argument (there is no default). The budget splits across three categories in a 40/30/30 split (software-engineering, adjacent technical roles, control occupations). Per-row outcomes are tracked in a coverage column:

- `labeled` — the LLM returned a result.
- `deferred` — eligible for LLM routing but the budget was exhausted before this row.
- `not_selected` — the row was not in the stage-9 core frame.
- `skipped_short` — the description was under 15 words.

**For the 2026 scrape specifically, 56.9% of postings are labeled, 43.0% are not selected, and 0.02% are deferred.** Analyses sensitive to posting text must either restrict to labeled postings or report results separately on the labeled-versus-not-labeled split.

Every LLM response is cached in a local SQLite database keyed by the input hash, the task name, and the prompt version. Re-running with a larger budget adds fresh calls without re-charging the cache hits.

## A note on memory

The pipeline runs on a 31 GB machine. Every stage uses streamed chunked reads (200,000-row batches, dropping to 50,000 for the LLM stages) rather than loading the full parquet files into memory. This is a hard constraint, not a preference.
