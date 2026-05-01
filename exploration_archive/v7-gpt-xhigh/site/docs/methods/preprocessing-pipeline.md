# Preprocessing turns heterogeneous postings into a comparable analysis corpus.

<p class="lead">The raw inputs are two historical Kaggle snapshots plus a growing daily scrape. The preprocessing pipeline normalizes, deduplicates, classifies, enriches, and audits those inputs into the final posting-level and observation-level outputs used by exploration tasks.</p>

This page condenses the project preprocessing guide and schema reference. It is included here so a reader can understand how the upstream columns used in the findings were produced.

## Stage summary

| Stage | What it does | Why it matters |
|---|---|---|
| 1. Ingest and schema unification | Loads arshkon, asaniczka, and current scraped data; normalizes source fields into a canonical posting table and scraped daily observations. | Makes heterogeneous historical and current inputs comparable. |
| 2. Aggregator handling | Flags staffing agencies and job-board aggregators; extracts real employers when possible; derives effective company name. | Prevents aggregator rows from being mistaken for direct employer behavior. |
| 4. Company canonicalization and dedup | Canonicalizes company names and removes exact, near-duplicate, and multi-location duplicate postings. | Produces the unique-posting unit used for most analyses. Only this stage reduces row count. |
| 5. Occupation and seniority classification | Assigns SWE, SWE-adjacent, and control flags; sets strong-rule seniority labels; extracts YOE. | Defines analytic samples and the label-vs-YOE measurement boundary. |
| 6-8. Location, temporal, and quality flags | Parses locations and metros, derives period/date features, and adds language, date, ghost-risk, and description-quality flags. | Enables geography, temporal, and quality-filtered analyses without changing row count. |
| 9. LLM extraction and cleaned text | Selects a deterministic LLM frame and asks the LLM to identify boilerplate units to remove. Produces `description_core_llm`. | This is the only boilerplate-removed text column and the canonical input for text-sensitive analysis. |
| 10. LLM classification and integration | Routes eligible rows to LLM classification for SWE type, seniority, ghost assessment, and YOE cross-check; merges results back. | Adds richer classification where rules are insufficient while preserving rule-based fallback columns. |
| Final output | Writes `data/unified.parquet`, `data/unified_observations.parquet`, quality report, and preprocessing log. | Provides one row per unique posting and a daily posting-observation panel. |

Stage 3 is intentionally absent. Boilerplate removal happens only in Stage 9.

## Output row structure

The primary analysis file is `data/unified.parquet`, with one row per unique posting after deduplication. `data/unified_observations.parquet` is the daily panel with one row per posting x scrape date for scraped data.

Key column families:

| Family | Examples | Notes |
|---|---|---|
| Identity and provenance | `uid`, `source`, `source_platform`, `job_id` | Tracks origin and source/platform. |
| Raw job content | `title`, `description`, `description_raw` | Raw description remains available for binary, boilerplate-insensitive checks. |
| Company and aggregator | `company_name`, `company_name_effective`, `company_name_canonical`, `is_aggregator` | Supports direct-employer, common-company, and aggregator sensitivity checks. |
| Occupation | `is_swe`, `is_swe_adjacent`, `is_control`, `swe_classification_tier`, `swe_classification_llm` | Defines SWE, adjacent technical roles, and controls. |
| Seniority and YOE | `seniority_native`, `seniority_final`, `seniority_final_source`, `seniority_3level`, `yoe_extracted` | `seniority_final` is the combined best-available seniority label; YOE is a separate validation signal. |
| Location and time | `metro_area`, `state_normalized`, `is_remote_inferred`, `period`, `scrape_week` | Metro work excludes multi-location rows when a posting cannot be assigned to a single metro. |
| Quality and ghost-risk | `date_flag`, `is_english`, `ghost_job_risk`, `description_quality_flag`, `ghost_assessment_llm` | Supports analysis filters and conservative ghost-risk fallback. |
| LLM text and coverage | `description_core_llm`, `selected_for_llm_frame`, `llm_extraction_coverage`, `llm_classification_coverage` | Coverage columns gate which rows have valid LLM-derived content. |

## Seniority column rule

Use `seniority_final` as the primary label column. Stage 5 sets it from high-confidence title keywords or manager indicators. Stage 10 overwrites it for routed rows when the LLM finds explicit seniority signals. Rows without explicit evidence remain `unknown`.

Seniority must not be inferred from responsibility complexity, tech stack, or YOE because the research asks how those requirements differ by seniority. Every entry-level claim should therefore be checked against the YOE proxy (`yoe_extracted <= 2`) and the T30 J1-J4 panel.

## LLM extraction prompt

Stage 9 sends numbered description units to the model and asks it to return IDs to drop as non-core boilerplate. The condensed task is:

> Drop units that do not change what the worker does, must know, must have, must own, or must coordinate. Keep role summaries, responsibilities, requirements, preferred qualifications, tech stack, domain expertise, seniority/scope expectations, and real operational constraints. Drop company overview, mission, culture, compensation, benefits, EEO/legal boilerplate, application instructions, generic metadata, and generic work-model text.

Output JSON:

```json
{
  "task_status": "ok | cannot_complete",
  "boilerplate_unit_ids": [1, 2],
  "uncertain_unit_ids": [3],
  "reason": "short phrase"
}
```

<details>
<summary>Stage 9 prompt excerpt</summary>

```text
You are preparing job-posting text for labor-market research.
You will receive numbered extraction units. Return ONLY valid JSON.

Goal:
Drop units that do not change what the worker does, must know, must have, must own, or must coordinate.

KEEP core job content:
- role summary
- responsibilities and day-to-day work
- requirements and qualifications
- preferred qualifications
- tech stack, tools, systems, methods
- domain expertise
- seniority- or scope-relevant expectations
- operational constraints that affect the work itself

DROP non-core text:
- company overview, mission, values, culture, and employer-branding
- all salary, pay, compensation, bonus, equity, and pay-range text
- all benefits, perks, insurance, PTO, leave, retirement, and total-rewards text
- all EEO / equal-opportunity / accommodation / legal-policy boilerplate
- all application instructions, recruiter/platform framing, and candidate-journey text
- generic metadata such as requisition IDs, posted dates, labels, and location headers
- all remote, hybrid, on-site, and work-model text unless it encodes a real work constraint

Decision rule:
- Return IDs for units to DROP, not units to keep.
- Compensation, benefits, generic work-model text, and EEO/legal text are never core.
- If a unit mixes core content with salary, benefits, work-model, or EEO/legal text, put it in uncertain_unit_ids unless the whole unit is clearly non-core.
```
</details>

## LLM classification prompt

Stage 10 classifies routed rows on four tasks: SWE class, seniority, ghost-job assessment, and minimum YOE. The seniority task uses only explicit seniority signals.

Output JSON:

```json
{
  "swe_classification": "SWE | SWE_ADJACENT | NOT_SWE",
  "seniority": "entry | associate | mid-senior | director | unknown",
  "ghost_assessment": "realistic | inflated | ghost_likely",
  "yoe_min_years": 2
}
```

<details>
<summary>Stage 10 prompt excerpt</summary>

```text
TASK 1 - SWE CLASSIFICATION
Classify this role into exactly one category:
- SWE: primary function is writing, designing, or maintaining software.
- SWE_ADJACENT: technical role with some code where coding is not the primary output.
- NOT_SWE: software development is not a meaningful part of the job.

TASK 2 - SENIORITY CLASSIFICATION
Use ONLY explicit seniority signals from the title or description.
Strong signals:
- junior, jr, intern, new grad, entry-level, early career -> entry
- senior, sr, staff, principal, lead, architect -> mid-senior
- director, vp, head of, chief -> director

Weak company-specific signals:
- associate, analyst, consultant, fellow, partner
- numeric or Roman numeral levels such as I/II/III, 1/2/3, L3/L4/L5, E3/E4/E5

Use weak signals only when the posting itself makes the mapping explicit.
Do NOT infer seniority from years of experience, responsibilities, tech stack, scope, or company reputation.

TASK 3 - GHOST JOB ASSESSMENT
Assess whether requirements are realistic for stated level: realistic, inflated, or ghost_likely.

TASK 4 - YEARS OF EXPERIENCE EXTRACTION
Use explicit years-of-experience mentions only. Return the binding minimum YOE floor, or null if none exists. Do not use YOE to infer seniority.
```
</details>

## LLM budget and coverage caveat

Stages 9 and 10 require an explicit `--llm-budget`; there is no default. Fresh calls are split 40% SWE, 30% SWE-adjacent, and 30% control unless overridden. Cached rows can extend usable coverage, but they do not change the sticky balanced core frame.

Coverage columns must be used:

- `llm_extraction_coverage = 'labeled'` identifies rows with valid `description_core_llm`.
- `llm_classification_coverage = 'labeled'` identifies rows with raw Stage 10 LLM outputs.
- `selected_for_llm_frame = true` marks the deterministic core frame. Supplemental cache rows can be useful but are not the balanced-sample frame.

This matters for the exploration: scraped LLM-cleaned SWE coverage is about 31% in the reports. Text, archetype, semantic, and requirement-breadth findings should be read as labeled-subset evidence unless a task explicitly uses a raw, boilerplate-insensitive binary screen.
