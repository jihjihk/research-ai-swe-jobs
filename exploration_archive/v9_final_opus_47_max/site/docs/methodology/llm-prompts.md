# LLM prompts

This page shows the exact text sent to the language model for every LLM-mediated step in the preprocessing pipeline. Nothing here has been paraphrased or summarized. Each block below is what the model actually reads when it classifies or cleans a posting.

Two tasks use the language model: **classification** (stage 10) and **boilerplate extraction** (stage 9). Each task has a single prompt template; every field the model writes to in the dataset traces to one of the two templates below.

Every prompt is versioned by a SHA-256 hash of its content. Changing any character of a prompt changes the hash, which in turn changes the cache key, forcing every posting to be re-routed under the new version rather than served from the stale cache.

## Stage 10: classification

The classification prompt performs four tasks in a single call: it decides whether the posting is a software-engineering role, assigns a seniority band, rates whether the posting looks like a genuine open position, and extracts a years-of-experience floor from the text. The four outputs feed four separate columns in the dataset (`swe_classification_llm`, `seniority_final` when the LLM-routed path is taken, `ghost_assessment_llm`, and `yoe_min_years_llm`).

Source: `preprocessing/scripts/llm_shared.py`, line 52.

??? note "Full prompt text (click to expand)"

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

### What each task produces

| Column | Task | Enum / type |
|---|---|---|
| `swe_classification_llm` | Task 1 | `SWE` / `SWE_ADJACENT` / `NOT_SWE` |
| `seniority_final` (LLM-routed rows) | Task 2 | `entry` / `associate` / `mid-senior` / `director` / `unknown` |
| `ghost_assessment_llm` | Task 3 | `realistic` / `inflated` / `ghost_likely` |
| `yoe_min_years_llm` | Task 4 | int or null |

### A note on the seniority design

The seniority task in the prompt above explicitly tells the model *not* to infer seniority from responsibilities, tech stack, team size, or years of experience. This is deliberate. The study analyzes how requirements and responsibilities differ by seniority level; if the seniority label were itself derived from requirements or responsibilities, the finding "requirements differ by seniority" would collapse into a tautology.

The consequence is that the model returns "unknown" on 34 to 53% of software-engineering rows. This is correct behavior, not a defect. The model is refusing to guess when the posting does not give it an explicit title-level or career-stage signal. An audit confirmed the abstentions are genuinely ambiguous cases, not the model failing to classify clear ones.

## Stage 9: boilerplate extraction

The extraction prompt takes a posting description and returns the unit IDs that should be dropped from it: things like company-overview language, benefits text, EEO disclaimers, and application-process paragraphs. What is left after the drops is the cleaned description used in every text-sensitive analysis.

The description is pre-segmented into numbered sentence-level units before the prompt runs; the model reads those numbered units and returns a JSON list of IDs to discard. Descriptions under 15 words are not routed at all.

Source: `preprocessing/scripts/llm_shared.py`, line 138.

??? note "Full prompt text (click to expand)"

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
    - all EEO / equal-opportunity / anti-discrimination / accommodation / legal-policy boilerplate such as "equal opportunity employer", "all qualified applicants will receive consideration", or protected-class language
    - all application instructions, recruiter/platform framing, and candidate-journey text
    - generic metadata such as requisition IDs, posted dates, labels, and location headers
    - all remote, hybrid, on-site, and work-model text, including in-office cadence, commute expectations, and flexibility language, unless it encodes a real work constraint like travel frequency, shift coverage, or clearance/facility access

    Decision rule:
    - Return IDs for units to DROP, not units to keep.
    - Compensation is never core.
    - Benefits are never core.
    - Generic remote, hybrid, on-site, and work-model text is never core by itself.
    - EEO/legal text is never core.
    - EEO/legal boilerplate is never core, even when it mentions disability, veteran status, or other protected classes.
    - Standalone headers inherit the type of their section. For example: Benefits, Benefits & Perks, Compensation, Salary Range, Work Model, Worker Category, EEO.
    - If a unit mixes core content with salary, benefits, work-model, or EEO/legal text, put it in uncertain_unit_ids unless the whole unit is clearly non-core and should be dropped.
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

### What extraction produces

| Column | Source | Use |
|---|---|---|
| `description_core_llm` | Reassembled from non-dropped units | **The only cleaned-text column.** Required for text-sensitive analyses. |
| `llm_extraction_coverage` | Per-row routing outcome | `labeled` / `deferred` / `not_selected` / `skipped_short`. Filter to `labeled` when reading `description_core_llm`. |

Descriptions under 15 words are hard-skipped and never routed to the extraction prompt.

## Engine configuration

The study used three language-model providers, with Codex as the default and the others enabled for subsets of the run when Codex was rate-limited:

| Engine | Model | Invocation |
|---|---|---|
| Codex (default) | `gpt-5.4-mini` | `codex exec --full-auto --config model=gpt-5.4-mini` |
| Claude (optional) | `haiku` | `claude -p "<prompt>" --model haiku --output-format json` |
| OpenAI (optional) | `gpt-5.4-nano` | POST `/v1/responses` |

Model identifiers are pinned in code, not passed at call time. Claude and OpenAI were enabled for subsets of the run to accelerate coverage during quota pauses on the default engine.

## Prompt versioning

Prompt versions are content hashes, computed at import time:

```python
CLASSIFICATION_PROMPT_VERSION = sha256(CLASSIFICATION_PROMPT_TEMPLATE)
EXTRACTION_PROMPT_VERSION     = sha256(EXTRACTION_PROMPT_TEMPLATE)
PROMPT_BUNDLE_VERSION         = sha256(f"{CLASSIFICATION_PROMPT_VERSION}:{EXTRACTION_PROMPT_VERSION}")
```

The cache key is `(input_hash, task_name, prompt_version)`. Any edit to the prompt text (including whitespace) changes the hash and forces every row to be re-resolved under the new version. Cached results from the previous version remain on disk but are no longer served.

## Coverage caveats

Restating the points in [preprocessing](preprocessing.md) that matter for reading LLM-derived columns:

- On the 2026 scrape, the classification prompt labeled 56.9% of postings; 43.0% were not selected for routing (budget-capped) and 0.02% were deferred.
- The re-test on postings least likely to be LLM-authored shows that 80 to 130% of content deltas persist, so recruiter-side LLM writing is not a dominant mediator of the study's content findings.
- The `seniority_final` column is always populated (rule-based, LLM-based, or "unknown") and can be used directly without a coverage filter. Every other LLM-derived column (`swe_classification_llm`, `ghost_assessment_llm`, `yoe_min_years_llm`, `description_core_llm`) must be coverage-filtered before use.
