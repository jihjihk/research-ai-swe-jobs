# Research Review

Date: 2026-03-24

## Project in one view

- This project studies how software engineering job postings changed from 2024 to 2026 in the AI era.
- The main question is not whether AI already reduced employment directly.
- The main question is whether employers restructured roles first:
  - fewer junior openings
  - broader junior expectations
  - senior roles shifting toward AI-enabled orchestration, review, and systems responsibility
- The project is mixed-methods:
  - quantitative analysis of job postings
  - qualitative interviews to explain the mechanisms behind posting changes

## Core research questions

- RQ1: Did employer-side SWE requirements restructure across seniority levels?
- RQ2: Which responsibilities and requirements moved downward into junior roles, and how did senior roles change?
- RQ3: Did employer AI requirements rise faster than worker-side AI usage benchmarks?
- RQ4: What mechanisms explain the shift: real workflow change, screening inflation, template inflation, or anticipatory hiring?

## Data we have

- Historical LinkedIn benchmark data from 2024
- Current scraped postings from 2026
- Daily scrape infrastructure across major US metros
- Both canonical-posting and observation-level datasets
- Public worker-side AI usage benchmarks for comparison
- Interview artifacts prepared for qualitative follow-up

## Current pipeline state

- The project has a multi-stage preprocessing pipeline that unifies, deduplicates, classifies, and flags posting data.
- The current exploration base is the Stage 8 output, which is the most current analysis-ready artifact.
- The exported unified parquet files are usable for snapshots, but the current exploratory work relies on the fresher stage-level dataset.
- LLM augmentation stages are designed but not yet fully run at scale because of quota limits.
- Result:
  - the project is already analyzable now
  - some variables are still provisional rather than final

## What the pipeline is doing conceptually

- Ingests multiple historical and scraped sources into one schema
- Handles aggregators and staffing intermediaries
- Removes boilerplate as much as possible
- Canonicalizes company identities and deduplicates postings
- Classifies:
  - SWE vs adjacent vs control roles
  - seniority
  - quality flags and temporal fields
- Produces:
  - one canonical postings table
  - one observation-level table for repeated sightings over time

## Main design position

- The strongest paper is about employer-side restructuring, not direct labor-market causality.
- The study should be framed as posting-content and role-definition change.
- Break analysis and AI-release timing are supporting evidence, not the sole empirical backbone.
- The paper should not overclaim:
  - not a payroll study
  - not a direct employment-effects estimate
  - not proof that one model release caused one market shift on one date

## Literature position

- Prior work is strongest on:
  - employment effects
  - labor demand
  - task exposure
  - broad AI usage
- The main gap this project targets is:
  - how SWE roles themselves changed at the posting level
  - how junior and senior roles changed differently
  - whether posting requirements outpaced worker usage
  - who inside firms is responsible for these changes

## Methods position

- Best uses of methods in this project:
  - descriptive restructuring analysis for backbone facts
  - lexical contrast methods for identifying distinctive language changes
  - topic modeling for exploration and robustness, not headline claims
  - human-led thematic analysis for interview mechanisms
  - LLMs as annotation assistants, not replacements for qualitative judgment
- General methodological rule:
  - simple, interpretable measures should carry the main claims
  - more complex NLP and ML should support, validate, or deepen those claims

## Preliminary findings

### 1. Junior SWE share appears to have declined

- The strongest current result is that entry-level SWE share fell meaningfully between 2024 and 2026.
- The project currently interprets this as a narrowing of the junior rung.
- This result appears stronger within overlapping companies than in the raw cross-section.
- Control occupations move in the opposite direction, which argues against a simple economy-wide explanation.

### 2. Junior roles appear more complex, not simply more credential-heavy

- Junior postings look broader and more organizationally demanding.
- Growth is especially visible in language about:
  - cross-functional work
  - ownership
  - collaboration
  - end-to-end responsibility
- But the evidence does not support a simple story of rising formal credential demands.
- The current read is:
  - organizational complexity increased
  - formal credential inflation is weak or absent
- Best interpretation:
  - complexity inflation, not classic credential inflation

### 3. AI language is the clearest temporal shift

- AI-related language is the strongest textual marker separating 2024 from 2026 postings.
- Growth is visible in mentions of:
  - LLMs
  - GPT / Claude / Copilot / Cursor
  - agents
  - RAG
  - orchestration-style AI work
- The rise is visible across the corpus and is especially important for the project's employer-side AI requirement story.

### 4. Senior SWE roles appear to be shifting in archetype

- Senior postings seem to be moving away from classic people-management language.
- They appear to be shifting toward:
  - architecture
  - review
  - orchestration
  - AI-enabled leverage
  - broader technical system responsibility
- This is one of the most novel parts of the study because the literature says less about senior role rewriting than about generic exposure or labor demand.

### 5. Employer AI requirements appear to be catching up to worker-side AI usage

- The project compares employer-side AI requirement signals to worker-side usage benchmarks.
- The current reading is that the gap narrowed substantially from 2024 to 2026.
- Interpretation:
  - employers may be institutionalizing AI expectations rapidly
  - posting-level AI requirements are moving closer to actual reported worker usage
- This supports the argument for employer-side restructuring and anticipatory organizational change.

### 6. The paired and artifact evidence supports the quantitative story

- Paired same-company examples suggest 2026 postings are often:
  - longer
  - more AI-inflected
  - more organizationally expansive
- Interview artifacts are ready to test whether practitioners see those changes as:
  - real workflow shifts
  - inflated JDs
  - defensive screening
  - strategic signaling

## Main risks and caveats

### Seniority is still the biggest measurement problem

- The most important unresolved issue is seniority classification.
- A bug in the rule/native merge logic made one seniority variable unreliable enough to change the direction of the junior-share result under some definitions.
- The project currently relies on a patched seniority variable as the best available measure.
- This does not kill the current findings, but it is the biggest analytical risk still on the table.

### Additional limits

- One historical benchmark source contributes no true entry-level native labels, so the junior baseline depends heavily on the other historical source.
- Description length increased substantially, which complicates text comparisons.
- Boilerplate removal is imperfect.
- Some variables are not comparable across sources:
  - skills
  - industry
  - company size
  - work type in some cases
- Aggregator contamination remains a sensitivity issue.
- Posting data cannot directly identify employment effects.
- Posting requirements are not the same object as actual worker behavior.

## What is strongest right now

- Employer-side restructuring is a viable and defensible central claim.
- The junior-rung decline result is the strongest current substantive finding.
- AI requirement growth is clear and consistent.
- The senior archetype shift looks promising and likely publishable.
- Cross-occupation comparison strengthens the argument that this is not just a generic labor-market story.
- Within-company evidence strengthens the claim that observed changes are not only compositional.

## What still needs to happen

- Fix and rerun the seniority pipeline cleanly
- Run the remaining LLM augmentation stages at scale if quota allows
- Move from exploration to formal analysis on the current base
- Carry forward key sensitivity checks:
  - LinkedIn-only
  - aggregator exclusions
  - alternative seniority definitions
  - length-normalized text measures
- Use interviews to determine whether the observed changes reflect:
  - real workflow transformation
  - screening inflation
  - HR template inflation
  - anticipatory hiring behavior

## Bottom line for today

- The project is in good shape conceptually.
- The research question has narrowed to a stronger and more defensible contribution.
- The evidence so far supports a paper about employer-side restructuring of SWE roles in the AI era.
- The current best summary of the findings is:
  - junior SWE share fell
  - junior roles became more organizationally demanding
  - AI language surged
  - senior roles shifted toward orchestration and technical leverage
  - employer AI requirements moved closer to worker-side AI usage
- The main unresolved issue is still seniority measurement, but it looks fixable rather than fatal.
