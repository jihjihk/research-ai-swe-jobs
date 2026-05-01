# Preprocessing pipeline

Ten stages turn three raw data sources (two Kaggle snapshots, one daily scrape) into `data/unified.parquet` — the analysis-ready dataset. The first eight stages are deterministic rule-based processing; stages 9 and 10 call an LLM.

## Why preprocessing exists

The raw data has three incompatible shapes:

| Source | Period | Format | Gaps |
|---|---|---|---|
| Kaggle arshkon | 2024-04 | companion-join tables, HTML-stripped text | small SWE count |
| Kaggle asaniczka | 2024-01 | separate description join, HTML-stripped | zero native entry labels |
| Scraped (EC2) | 2026-03 onward | markdown-preserving, search-query metadata | growing daily |

The pipeline harmonizes them into a single 39-column canonical format so every research question can be answered on a shared corpus. LLM-cleaned text removes boilerplate (benefits, EEO, recruiter framing) that otherwise contaminates text-based findings.

## Stage-by-stage

| Stage | What | Why |
|---|---|---|
| 1 — Ingest | Load three sources, apply source-specific joins, concatenate into 39-column canonical schema | Single surface for every downstream stage |
| 2 — Aggregator handling | Flag staffing-agency postings (Dice, Lensa, Robert Half); extract `real_employer` when possible | Aggregators are a composition confound; `company_name_effective` resolves them |
| 4 — Dedup | Two-pass: compute keep/drop decisions from dedup keys, then stream filtered data | Only row-reducing stage; collapses exact, fuzzy, and multi-location duplicates |
| 5 — Classification | Three-tier SWE classifier (regex → title lookup → embedding); strong-rule seniority from title keywords; YOE extracted from description | First analytical classification boundary |
| 6-8 — Normalization / temporal / quality | Single script: location parsing, metro assignment, period tagging, language detection, date validation, ghost-job heuristics | Row-preserving enrichment |
| 9 — LLM extraction | Selects core frame deterministically; sends each posting to LLM to identify boilerplate units; reconstructs `description_core_llm` | Boilerplate removal at ~44% rule-based accuracy was unacceptable; LLM-only cleanup |
| 10 — LLM classification | Routes ambiguous rows to LLM for SWE / seniority / ghost / YOE classification; writes results back into `seniority_final` for routed rows | Disambiguation for rows where Stage 5 couldn't fire a strong rule |
| final | Read Stage 10 output, final column selection, write `data/unified.parquet` | Deliverable |

Stage 3 is intentionally absent — the original rule-based boilerplate removal was retired in April 2026 at ~44% accuracy.

## Data structure output

Primary artifact: `data/unified.parquet`, one row per unique posting. Secondary: `data/unified_observations.parquet`, one row per posting × scrape_date (daily panel for scraped data).

### Column families

| Family | Cols | Examples | Notes |
|---|---:|---|---|
| Identity | 5 | `uid`, `source`, `source_platform` | `uid` is primary key |
| Job content | 6 | `title`, `description`, `description_core_llm` | Use LLM-cleaned for text-sensitive analyses |
| Company | 8 | `company_name_effective`, `company_name_canonical`, `is_aggregator` | Use `_effective` for aggregator-resolved; `_canonical` for grouping |
| Seniority | 4 | `seniority_final`, `seniority_final_source`, `seniority_3level`, `seniority_native` | `seniority_final` is primary; `_native` is diagnostic-only |
| SWE classification | 6 | `is_swe`, `is_swe_adjacent`, `is_control`, `swe_classification_llm` | Study-sample flags |
| YOE | 7 | `yoe_extracted`, `yoe_seniority_contradiction`, `yoe_min_years_llm` | Rule-based primary; LLM is cross-check |
| Geography | 10 | `metro_area`, `state_normalized`, `is_remote_inferred` | **`is_remote_inferred` = 100% False — pipeline bug, blocks remote analysis** |
| Temporal | 3 | `period`, `posting_age_days`, `scrape_week` | |
| Quality | 5 | `date_flag`, `is_english`, `ghost_job_risk`, `description_quality_flag` | Default filters: `is_english = true AND date_flag = 'ok'` |
| LLM coverage | 4 | `llm_extraction_coverage`, `llm_classification_coverage`, `selected_for_llm_frame` | **Filter to `labeled` when using LLM-derived columns** |

### Coverage columns — critical for every analysis

- **`llm_extraction_coverage`** tracks whether `description_core_llm` is populated. Values: `labeled` / `deferred` / `not_selected` / `skipped_short`. Filter to `labeled` for any text-sensitive analysis.
- **`llm_classification_coverage`** tracks Stage 10 coverage. Values: `labeled` / `deferred` / `not_selected` / `skipped_short`. Every in-frame eligible row routes to the LLM; there is no rule-based shortcut.
- **`selected_for_llm_frame`** marks the deterministic sticky core frame — a balanced sample across source × analysis_group × date_bin. Balanced-sample claims apply only to this frame.

## LLM stages and prompts

Two LLM passes. Stage 9 removes boilerplate; Stage 10 classifies. Each has its own cache (SQLite), keyed by `(input_hash, task_name, prompt_version)`.

### Stage 9 — extraction prompt (boilerplate removal)

Input: posting split into numbered sentence-level units. Output: list of unit IDs to drop. The LLM returns IDs to drop (not keep) to reduce response-length variance.

<details>
<summary><strong>Show extraction prompt</strong></summary>

```
You are preparing job-posting text for labor-market research.
You will receive numbered extraction units. Return ONLY valid JSON.

Goal:
Drop units that do not change what the worker does, must know, must have,
must own, or must coordinate.

KEEP core job content:
- role summary
- responsibilities and day-to-day work
- requirements and qualifications
- preferred qualifications
- tech stack, tools, systems, methods
- domain expertise
- seniority- or scope-relevant expectations
- operational constraints that affect the work itself: travel frequency, shift
  coverage, on-call, clearance, contract length, or reporting line

DROP non-core text:
- company overview, mission, values, culture, and employer-branding
- all salary, pay, compensation, OTE, bonus, equity, and pay-range text
- all benefits, perks, insurance, PTO, leave, retirement, 401(k), wellness,
  tuition, and total-rewards text
- all EEO / equal-opportunity / anti-discrimination / accommodation /
  legal-policy boilerplate
- all application instructions, recruiter/platform framing, candidate-journey
- generic metadata such as requisition IDs, posted dates, labels, location
  headers
- all remote, hybrid, on-site, and work-model text unless it encodes a real
  work constraint like travel frequency, shift coverage, or clearance access

Decision rule:
- Return IDs for units to DROP, not units to keep.
- Compensation is never core.
- Benefits are never core.
- EEO/legal text is never core.
- Standalone headers inherit the type of their section.
- If a unit mixes core content with boilerplate, put it in uncertain_unit_ids.

TITLE: {title}
COMPANY: {company}

NUMBERED UNITS:
{numbered_units}

Respond with this exact JSON structure:
{
  "task_status": "ok" | "cannot_complete",
  "boilerplate_unit_ids": [1, 2],
  "uncertain_unit_ids": [3],
  "reason": "short phrase"
}
```

</details>

### Stage 10 — classification prompt (SWE, seniority, ghost, YOE)

Four tasks in one call. The seniority task has explicit rules against YOE-based or responsibility-based inference — labels must come from explicit title/description signals only so downstream analysis of "how requirements differ by seniority" is not circular.

<details>
<summary><strong>Show classification prompt</strong></summary>

```
You are a labor economics research assistant classifying job postings.
Perform the tasks below on this job posting. Return ONLY valid JSON.

TASK 1 - SWE CLASSIFICATION
Classify this role into exactly one category:
- "SWE": The role's primary function is writing, designing, or maintaining
  software. Includes software engineers, full-stack developers,
  frontend/backend engineers, mobile developers, ML engineers, data engineers
  who primarily write code, and DevOps engineers whose description emphasizes
  writing code for infrastructure.
- "SWE_ADJACENT": Technical roles that involve some code but where coding is
  not the primary function.
- "NOT_SWE": Roles where software development is not a meaningful part of the
  job.

TASK 2 - SENIORITY CLASSIFICATION
Use ONLY explicit seniority signals from the title or description.

Strong signals:
- "junior", "jr", "intern", "internship", "new grad", "graduate",
  "entry-level", "early career", "apprentice" -> "entry"
- "senior", "sr", "staff", "principal", "lead", "architect" -> "mid-senior"
- "director", "vp", "vice president", "head of", "chief" -> "director"

Weak company-specific signals:
- "associate", "analyst", "consultant", "fellow", "partner"
- numeric or Roman numeral levels such as I/II/III, 1/2/3, L3/L4/L5, E3/E4/E5

Rules for weak signals:
- Use them ONLY when the posting itself makes the mapping explicit.
- Do NOT assume company-specific numbering from general knowledge.
- If the only seniority evidence is a weak company-specific signal, return
  "unknown".

IMPORTANT:
- Do NOT infer seniority from years of experience, responsibilities, tech
  stack, scope, or company reputation.
- When in doubt, classify as "unknown".

TASK 3 - GHOST JOB ASSESSMENT
- "realistic": requirements match the stated or apparent seniority
- "inflated": requirements are significantly higher than stated level would
  normally demand
- "ghost_likely": strong signs this is not a genuine open position

TASK 4 - YEARS OF EXPERIENCE EXTRACTION
Extract `yoe_min_years`.
- Use explicit years-of-experience mentions only.
- For a single qualification path, return the binding YOE floor.
- If multiple acceptable qualification paths, return the lowest path-level.
- If a range is given, return the lower bound.
- Ignore title/level numbers, dates, salaries, addresses, clearance levels.
- Do not use YOE to infer seniority.

TITLE: {title}
COMPANY: {company}
DESCRIPTION:
{full_description}

Respond with this exact JSON structure:
{
  "swe_classification": "SWE" | "SWE_ADJACENT" | "NOT_SWE",
  "seniority": "entry" | "associate" | "mid-senior" | "director" | "unknown",
  "ghost_assessment": "realistic" | "inflated" | "ghost_likely",
  "yoe_min_years": <integer|null>
}
```

</details>

### Engines

| Engine | Model | Invocation | Role |
|---|---|---|---|
| Codex | `gpt-5.4-mini` | `codex exec --full-auto --config model=...` | Default |
| Claude | `haiku` | `claude -p ... --model haiku --output-format json` | Secondary |
| OpenAI | `gpt-5.4-nano` | Direct HTTPS `POST /v1/responses` | Tertiary |

## Budget and coverage caveats

**LLM stages are budget-capped.** Stages 9 and 10 require `--llm-budget N` (no default). The budget caps fresh LLM calls per run, split 40% SWE / 30% SWE-adjacent / 30% control by default.

**Sticky core frame.** Stage 9 selects a deterministic balanced sample across source × analysis_group × date_bin; Stage 10 inherits it. Supplemental cache rows can extend usable coverage but do not change the balanced frame.

**Coverage is incomplete.** 34,102 of 63,701 SWE rows carry `llm_extraction_coverage = 'labeled'`. The exploration reports every text-dependent headline on BOTH the full corpus and the labeled subset. V1.5 confirmed a 2-3× junior-share gap between labeled and unlabeled subsets — the LLM frame preferentially selects junior postings.

**Reporting restricted.** Balanced-sample claims (cross-period, cross-source balance) apply only to `selected_for_llm_frame = true`. Analyses outside this frame that use LLM-derived columns must report coverage per cell.

## Cross-references

- [Schema column reference](#) — see the full 90+ column definition in `docs/preprocessing-schema.md` checked into the repo.
- [Guide](#) — see `docs/preprocessing-guide.md` for operations, monitoring, and test patterns.
- [Sensitivity framework](sensitivity-framework.md) — how the exploration phase exercises these columns.
</content>
</invoke>