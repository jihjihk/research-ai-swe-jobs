# Preprocessing pipeline

This page is a condensed, reader-oriented description of the preprocessing pipeline that transforms raw job-posting CSVs into the analysis-ready `unified.parquet` used by every finding in this package. It is extracted from `docs/preprocessing-guide.md` and `docs/preprocessing-schema.md` (the canonical references in the repository) and is intended for a skeptical reader clicking through from a finding page. It is **not** operator documentation — for run commands, cache semantics, and failure modes, see the in-repo guide.

## Why preprocessing exists

Three problems force a real pipeline:

1. **Three heterogeneous sources** (arshkon, asaniczka, scraped) with different schemas, column names, date formats, seniority conventions, and description cleanup levels must merge into one comparable panel.
2. **Classification decisions** (is this SWE? what seniority? what occupation? what metro?) are load-bearing for every downstream finding and need both deterministic rules (for speed and reproducibility) and LLM labels (for the cases rules cannot handle).
3. **Boilerplate** (EEO text, compensation text, benefits, culture paragraphs) dominates raw description length and biases every text-sensitive metric — the pipeline has to produce a clean text column for analysis.

## Architecture: rule-based baseline + LLM augmentation

The pipeline has two layers:

1. **Rule-based baseline (Stages 1-8):** deterministic, fast (~30 min end-to-end), reproducible. Produces a usable corpus with rule-based classification labels for occupation, seniority, YOE, location, temporal, and quality fields.
2. **LLM augmentation (Stages 9-10):** adds higher-quality classification and cleaned text via LLM calls. Takes hours to days depending on corpus size and API quotas. Results are cached in SQLite for resumability.

Rule-based runs first and its outputs serve as both the classification fallback labels and the cache keys for the LLM layer.

## The 10 stages at a glance

```
Raw data (3 sources)
  │
  ├── Stage 1  — Ingest & schema unification
  │              (arshkon join + asaniczka join + scraped ingest → 39 canonical columns)
  ├── Stage 2  — Aggregator / staffing-agency handling
  │              (flag Dice, Lensa, Robert Half etc.; extract real employers)
  ├── Stage 4  — Company canonicalization + dedup
  │              (the ONLY row-reducing stage: exact/fuzzy/multi-location collapse)
  ├── Stage 5  — Occupation + seniority classification
  │              (regex + title lookup + embedding; YOE clause parser)
  ├── Stages 6-8 — Location normalize, temporal alignment, quality flags
  │              (metro assignment, period derivation, language detection, ghost heuristics)
  │
  ▼  preprocessing/intermediate/stage8_final.parquet   ← rule-based baseline
  │
  ├── Stage 9  — LLM extraction + cleaned text
  │              (core-frame selection; sentence-unit boilerplate removal
  │               → description_core_llm, the cleaned text column)
  ├── Stage 10 — LLM classification + final integration
  │              (SWE type, seniority, ghost, YOE cross-check for routed rows;
  │               LLM seniority overwrites seniority_final)
  │
  ▼  data/unified.parquet          (one row per unique posting)
  ▼  data/unified_observations.parquet   (daily panel)
```

**Stage 3 is intentionally absent** — the original stage 3 (rule-based boilerplate) was removed on 2026-04-10 because it didn't add useful signal. Boilerplate removal now lives **exclusively in Stage 9**.

## Stage-by-stage summary

### Stage 1 — Ingest & schema unification

Loads the three source datasets, applies source-specific joins and normalization, concatenates into a canonical 39-column format. Handles seniority mapping, text normalization, date parsing, company size parsing. Produces both the canonical posting table and a daily observations table for scraped data.

### Stage 2 — Aggregator handling

Identifies staffing agencies and job-board aggregators (Dice, Lensa, Robert Half, etc.) via exact name matching and regex patterns. Extracts the real employer from description text when possible. Derives `company_name_effective = real employer if aggregator, else original company name`. An exploration-phase finding (T08/T16) was that this flag *misses* entry-specialist intermediaries (SynergisticIT, WayUp, Leidos, Emonics) that drive ~15-20% of the 2026 entry pool — a follow-up flag is on the preprocessing action items list.

### Stage 4 — Company canonicalization + dedup

The only row-reducing stage. Uses a memory-safe two-pass design: Pass 1 loads only dedup-key columns to decide keep/drop; Pass 2 streams full data and filters to kept rows.

Dedup strategies:

- Exact `job_id` duplicates
- Exact opening duplicates (company + title + location + hash of raw description)
- Fuzzy near-duplicates (token_set_ratio ≥ 85%)
- **Multi-location collapse:** rows sharing `(company, title, description_hash)` across ≥ 2 normalized locations collapse to a single representative row with `location = 'multi-location'` and `is_multi_location = True`.

### Stage 5 — Occupation + seniority classification

The first analytical classification boundary. Assigns `is_swe`, `is_swe_adjacent`, `is_control` via a 3-tier system (regex → curated title lookup → embedding). Sets `seniority_final` from high-confidence title keywords (`title_keyword`, `title_manager`); rows with no strong rule match are left as `unknown` for Stage 10 to fill in via LLM. Extracts years-of-experience with a clause-aware parser.

### Stages 6-8 — Normalization, temporal, quality flags

A single script implementing three logical stages. Stage 6 parses locations into city/state/country, infers remote status, assigns metro areas. Stage 7 derives `period`, `posting_age_days`, `scrape_week`. Stage 8 adds language detection, date validation, ghost-job heuristics, description quality flags. All row-preserving.

### Stage 9 — LLM extraction + cleaned text

Defines the LLM analysis universe (LinkedIn, English, has description). Selects a deterministic sticky core over `source × analysis_group × date_bin` via a `selection_target` budget. Segments descriptions into sentence units and asks LLMs to identify **boilerplate units for removal**. Produces `description_core_llm` — the LLM-cleaned description, which is the **only** boilerplate-removed text in the pipeline and the canonical input for any text-sensitive analysis. Hard-skips descriptions under 15 words.

See **[The Stage 9 extraction prompt](#stage-9-extraction-prompt-verbatim)** below for the exact prompt text.

### Stage 10 — LLM classification + final integration

Reuses the Stage 9 core frame and routes eligible rows to LLM classification: SWE type, seniority, ghost-job assessment, YOE cross-check. **Skips** rows where the rule-based confidence is already high (strong SWE tier + Stage 5 strong-rule seniority + low ghost risk). For routed rows, the LLM seniority result overwrites `seniority_final` with `seniority_final_source = 'llm'`. Merges all LLM results back to the full posting table.

See **[The Stage 10 classification prompt](#stage-10-classification-prompt-verbatim)** below for the exact prompt text.

## Output schema highlights (the columns findings actually use)

| Column | Source | Used for |
|---|---|---|
| `uid` | Stage 1 | Row identity |
| `source`, `source_platform` | Stage 1 | Source provenance |
| `description`, `description_raw` | Stage 1 | Raw text (boilerplate-insensitive analyses only) |
| `description_core_llm` | Stage 9 | **The cleaned-text column for every text-sensitive analysis** |
| `is_swe`, `is_swe_adjacent`, `is_control` | Stage 5 | Occupation frames, DiD |
| `seniority_final`, `seniority_final_source` | Stages 5 + 10 | **Primary seniority column — always paired with YOE ≤ 2 proxy** |
| `seniority_3level` | — | Coarse junior/mid/senior/unknown |
| `yoe_extracted`, `yoe_min_extracted` | Stage 5 | YOE-based entry proxy |
| `metro_area`, `is_remote_inferred` | Stage 6 | 18-metro frame, remote-vs-metro comparison |
| `period`, `posting_age_days` | Stage 7 | Temporal slicing, within-scraped-window calibration |
| `is_english`, `date_flag`, `ghost_job_risk` | Stage 8 | Default filters |
| `llm_extraction_coverage`, `llm_classification_coverage` | Stages 9-10 | Binding constraint check (filter to `labeled`) |
| `is_aggregator`, `company_name_effective`, `company_name_canonical` | Stages 2, 4 | Aggregator sensitivity, within-company grouping |

**Default analytical filter:**

```sql
WHERE is_swe = TRUE
  AND source_platform = 'linkedin'
  AND is_english = TRUE
  AND date_flag = 'ok'
```

For text-sensitive analyses, additionally filter to `llm_extraction_coverage = 'labeled'`. See [Data sources](data-sources.md) for binding constraints on 2026 coverage.

---

## Stage 9 extraction prompt (verbatim)

This is the exact prompt sent to the LLM to identify boilerplate units for removal. Prompt version is SHA-256 hashed from the template text; any prompt edit invalidates the cache.

??? quote "Click to expand the full Stage 9 extraction prompt"

    ```
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
    - operational constraints that affect the work itself: travel frequency, shift coverage, on-call, clearance, contract length, or reporting line

    DROP non-core text:
    - company overview, mission, values, culture, and employer-branding
    - all salary, pay, compensation, OTE, bonus, equity, and pay-range text
    - all benefits, perks, insurance, PTO, leave, retirement, 401(k), wellness, tuition, and total-rewards text
    - all EEO / equal-opportunity / anti-discrimination / accommodation / legal-policy boilerplate such as
      "equal opportunity employer", "all qualified applicants will receive consideration", or protected-class language
    - all application instructions, recruiter/platform framing, and candidate-journey text
    - generic metadata such as requisition IDs, posted dates, labels, and location headers
    - all remote, hybrid, on-site, and work-model text, including in-office cadence, commute expectations,
      and flexibility language, unless it encodes a real work constraint like travel frequency,
      shift coverage, or clearance/facility access

    Decision rule:
    - Return IDs for units to DROP, not units to keep.
    - Compensation is never core.
    - Benefits are never core.
    - Generic remote, hybrid, on-site, and work-model text is never core by itself.
    - EEO/legal text is never core.
    - EEO/legal boilerplate is never core, even when it mentions disability, veteran status, or other protected classes.
    - Standalone headers inherit the type of their section. For example: Benefits, Benefits & Perks, Compensation,
      Salary Range, Work Model, Worker Category, EEO.
    - If a unit mixes core content with salary, benefits, work-model, or EEO/legal text,
      put it in uncertain_unit_ids unless the whole unit is clearly non-core and should be dropped.
    - Use only unit IDs that appear in the numbered units.
    - If the units are malformed or the task cannot be completed reliably, return
      "task_status": "cannot_complete" with empty ID lists.
    - Keep `reason` short.

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

    **Source:** `preprocessing/scripts/llm_shared.py:EXTRACTION_PROMPT_TEMPLATE`.

---

## Stage 10 classification prompt (verbatim)

This is the exact prompt sent to the LLM for SWE classification, seniority classification, ghost-job assessment, and YOE extraction, in a single call.

??? quote "Click to expand the full Stage 10 classification prompt"

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
      not the primary function. Includes QA/SDET roles, technical program
      managers, solutions architects, many data-science roles, and product roles
      for developer tooling where coding is useful but not the primary output.
    - "NOT_SWE": Roles where software development is not a meaningful part of the
      job. Includes non-technical roles, most IT support roles, hardware/product
      engineering roles focused on physical systems, and misleading "engineer"
      titles where the work is not software development.

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
    - Do NOT assume company-specific numbering or title ladders from general knowledge.
    - If the only seniority evidence is a weak company-specific signal, return "unknown".

    IMPORTANT:
    - Do NOT infer seniority from years of experience, responsibilities, tech stack,
      scope, or company reputation.
    - When in doubt, classify as "unknown".

    TASK 3 - GHOST JOB ASSESSMENT
    Assess whether this posting's requirements are realistic for its stated level:
    - "realistic": Requirements match the stated or apparent seniority level
    - "inflated": Requirements are significantly higher than the stated level would
      normally demand
    - "ghost_likely": Strong signs this is not a genuine open position

    TASK 4 - YEARS OF EXPERIENCE EXTRACTION
    Extract `yoe_min_years`.

    Rules:
    - Use explicit years-of-experience mentions only. Number words and digits both
      count.
    - For a single qualification path, return the binding YOE floor: use the
      highest relevant years-of-experience mention on that path, including
      preferred figures.
    - Tool/framework/domain-specific experience counts if it is the only YOE
      mention on that path, or if it is higher than the general-role YOE on that
      path.
    - If the posting gives multiple acceptable qualification paths, return the
      lowest path-level YOE floor.
    - If a range is given, return the lower bound.
    - Ignore title/job-level numbers, dates, salaries, addresses, clearance levels,
      and other numbers not tied to YOE.
    - If no relevant explicit YOE exists, return null.
    - Do not use YOE to infer seniority.

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
      "ghost_assessment": "realistic" | "inflated" | "ghost_likely",
      "yoe_min_years": <integer|null>
    }
    ```

    **Source:** `preprocessing/scripts/llm_shared.py:CLASSIFICATION_PROMPT_TEMPLATE`.

---

## Budgeting and coverage caveats

This is where preprocessing realities hit the analysis. Three coverage facts control which findings can run:

1. **Stage 9 LLM text coverage on scraped is 30.7% labeled.** This is the binding constraint on every text-sensitive 2026 analysis. Analyses on `description_core_llm` cap at ~12,500 2026 rows.
2. **Stage 10 LLM seniority coverage leaves 53% of scraped SWE as `seniority_final = 'unknown'`.** Denominator drift — "of known" entry shares drift from 61% → 47% between periods, structurally biasing any "of known" comparison.
3. **T09 archetype label coverage on scraped is 30.5%.** All within-archetype 2026-side claims inherit this constraint.

Both LLM stages use separate SQLite caches so cache reuse can be computed independently of fresh calls; prompt-version hashing ensures edits to prompt text invalidate the cache automatically.

**Preprocessing action items** (SYNTHESIS Section 17) to address these constraints in the next run:

- Raise Stage 9 selection target for scraped.
- Raise Stage 10 budget to reduce `seniority_final = 'unknown'` share.
- Add an entry-specialist intermediary flag to supplement `is_aggregator`.
- Apply the markdown-escape fix (`c\+\+`, `c\#`, `\.net`).
- Add a lightweight raw-text archetype classifier to broaden within-archetype coverage.

## Canonical references

- **[`docs/preprocessing-guide.md`](https://github.com/)** — the operator-oriented guide with run commands, cache semantics, and failure modes.
- **[`docs/preprocessing-schema.md`](https://github.com/)** — full column-by-column schema reference.
- **[`preprocessing/scripts/llm_shared.py`](https://github.com/)** — the source file for the two prompts above (`CLASSIFICATION_PROMPT_TEMPLATE` and `EXTRACTION_PROMPT_TEMPLATE`).
