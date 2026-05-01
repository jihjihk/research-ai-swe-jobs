# Preprocessing and schema.

The preprocessing pipeline exists because the raw inputs are heterogeneous: two historical Kaggle snapshots and a growing daily scrape. The pipeline normalizes, deduplicates, classifies, and enriches them into `data/unified.parquet` and `data/unified_observations.parquet`.

## Stage-by-stage summary

| Stage | What it does | Why it matters |
|---|---|---|
| 1 | Ingests the three sources and unifies them into a canonical schema. | Creates a single posting table and daily observation table that downstream tasks can compare. |
| 2 | Flags aggregators and staffing agencies, then derives a real-employer field. | Prevents staffing intermediaries from contaminating employer-level claims. |
| 4 | Canonicalizes company names and deduplicates openings. | Keeps company-level and corpus-level counts from being inflated by duplicates and spelling drift. |
| 5 | Classifies SWE status, seniority, and years of experience. | Produces the core analysis labels, including the conservative `seniority_final` field. |
| 6-8 | Normalizes location, temporal, language, and quality flags. | Makes geography, timing, and data quality comparable across sources. |
| 9 | Selects the LLM core frame and removes boilerplate to create `description_core_llm`. | Produces the only cleaned-text column and makes text-sensitive work coverage-explicit. |
| 10 | Runs LLM classification and merges results back into the full table. | Completes the combined seniority, SWE, ghost, and YOE cross-check layer. |
| final | Writes `data/unified.parquet` and `data/unified_observations.parquet`. | Produces the analysis-ready dataset used in the exploration. |

## Output data structure

One row in `data/unified.parquet` is one unique posting. The main column families are:

- identity and provenance
- raw and cleaned text
- company and aggregator fields
- SWE and seniority classification
- YOE extraction
- temporal, geography, and quality flags
- LLM-derived classification and coverage columns

The coverage columns matter. `selected_for_llm_frame` marks the sticky balanced core only. `llm_extraction_coverage` and `llm_classification_coverage` tell you whether a row was actually labeled. `description_core_llm` is the only cleaned text column, and `seniority_final` is the primary seniority column.

## LLM stages and prompts

<details>
<summary>Stage 9: boilerplate removal and cleaned text</summary>

The LLM is asked to identify boilerplate units in a segmented job description and return the cleaned posting text. The important output is `description_core_llm`.

Condensed prompt:

> Given a job posting split into sentence-like units, remove boilerplate and return the core posting text. Keep role content, requirements, and responsibilities. Drop company marketing, legal, and generic filler.

</details>

<details>
<summary>Stage 10: classification and integration</summary>

The LLM is asked to classify SWE status, resolve seniority when the rule-based layer cannot, assess ghost-job risk, and cross-check YOE. The explicit seniority signals are the only valid basis for the LLM seniority result.

Condensed prompt:

> Classify the posting's SWE status, seniority, ghost-job assessment, and minimum years of experience using the cleaned text and explicit signals only. Do not infer seniority from responsibilities, tech stack complexity, or YOE requirements.

</details>

## Budgeting and coverage caveats

Stages 9 and 10 require an explicit `--llm-budget`. That budget caps new LLM calls, so not every row receives LLM-derived columns. Stage 9 and Stage 10 use separate caches, which is why coverage can differ row by row. Findings built on LLM columns should always report the labeled count alongside the eligible count and should distinguish the sticky core from supplemental cache rows.
