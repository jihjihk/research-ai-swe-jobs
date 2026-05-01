# Post-Exploration Action Plan

Date: 2026-04-06

## Overview

This plan addresses structural improvements to the exploration pipeline identified after completing all 4 waves. Changes are organized by: (A) process improvements to the exploration plan, (B) specific task additions/modifications, (C) documentation updates. All changes assume that `seniority_llm`, `description_core_llm` for scraped data, and `ghost_assessment_llm` will become available and the exploration will be re-run from scratch.

---

## A. Process improvements

### A1. Verification + challenge agent after each analytical wave

**What:** After Wave 2 and Wave 3, before dispatching the next wave, run a single "Verification Agent" that:
1. Re-derives the top 3-5 headline numbers from that wave from scratch (independent code, not reading prior scripts)
2. For any keyword indicator introduced in that wave, validates pattern precision by sampling 50 matches stratified by period
3. Proposes at least one alternative explanation for each headline finding
4. Flags any finding that depends on a single methodological choice (seniority column, text source, sample definition)

**Where:** Add as "Agent V" in the dispatch blocks, running between Wave 2 and Wave 3 (call it "Gate 2 Verification") and between Wave 3 and Wave 4 ("Gate 3 Verification"). This combines the verification step (point 1) with the challenge round (point 4).

**Prompt guidance:** "Your job is adversarial quality assurance. Assume the prior wave's agents made mistakes. Re-derive their headline numbers independently. Look for measurement artifacts, sampling bias, and alternative explanations. If you reproduce a number within 5%, it's verified. If not, flag for investigation."

### A2. Sampling protocol in the analytical preamble

**What:** Add to the analytical preamble (section 1b of task reference):

> **Sampling protocol.** For any analysis based on a sample rather than the full dataset:
> - Document sample size, stratification method, and what was excluded
> - Report what fraction of each source/period/seniority group is represented
> - Prefer balanced period representation over proportional-to-population
> - For keyword pattern validation, stratify samples by period — pattern behavior may differ between 2024 and 2026 due to description style changes
> - If a finding could change with a different sample, test it with at least one alternative sample

### A3. Narrative alternatives evaluation in the orchestrator prompt

**What:** Add to the orchestrator prompt's "Steering the investigation" section:

> **Evaluate alternative framings at every gate.** The same data can support multiple narratives depending on what you emphasize. At each gate, explicitly consider at least two alternative framings of the strongest findings and explain why you prefer one over the other. Examples of framings that could apply to this data:
> - Expansion framing: "AI expanded the SWE skill surface" vs decline framing: "AI eliminated junior roles"
> - Market recomposition: "Different companies are hiring" vs firm restructuring: "Companies changed what they hire for"
> - Platform/template evolution: "How postings are written changed" vs real demand: "What employers want changed"
> - Domain shift: "The market moved to ML/AI" vs seniority shift: "Junior roles disappeared"
> 
> The paper's credibility comes from honestly weighing these alternatives against the evidence, not from picking the most dramatic framing.

---

## B. Task additions and modifications

### B1. H1 test (domain-driven junior decline decomposition)

**What:** Add as a step in T08 (distribution profiling):

> **Domain × seniority decomposition.** If T09 archetype labels are available (from `exploration/artifacts/shared/swe_archetype_labels.parquet`), compute entry share by domain archetype by period. Decompose the aggregate entry share change into:
> - **Within-domain component:** entry share change holding archetype constant
> - **Between-domain component:** change driven by the market shifting from low-skill-floor domains (Frontend) to high-skill-floor domains (ML/AI)
> If the between-domain component accounts for >50% of the aggregate decline, the junior decline is primarily a domain recomposition effect, not a within-domain elimination.

Also add the same cross-tab to T16's decomposition (the company panel already does within-company vs compositional; this adds domain as a third dimension).

### B2. LinkedIn platform artifact test

**What:** Add as a new step in T05 (cross-dataset comparability) or as a standalone focused task:

> **Platform labeling stability test.** For the top 20 SWE titles appearing in both arshkon and scraped:
> - Compare native seniority label distributions per title. If the same title has systematically different seniority labels across periods, this suggests platform relabeling.
> - For title×seniority cells that exist in both periods, compare YOE distributions. If a cell's YOE didn't change but its frequency shifted, that's a composition change. If YOE changed too, it's a content change.
> - Cross-validate with Indeed data: compute entry-level share using `seniority_imputed` on Indeed scraped rows. If Indeed shows similar patterns to LinkedIn, the LinkedIn platform artifact hypothesis weakens.
> - Check whether LinkedIn publicly documented seniority taxonomy changes between 2024 and 2026.

### B3. Length-normalized AI stack density check

**What:** Add as a note to T14 (technology ecosystems):

> **Length-normalized AI additive check.** The finding that AI-mentioning postings have 11.4 techs vs 7.3 could be partly an artifact of AI postings being longer (more text = more keyword matches). Compute tech density (techs per 1K chars) for AI-mentioning vs non-AI postings. If the density difference is smaller than the count difference, length is a confounder. Report both raw count and density.

### B4. Indeed sensitivity check

**What:** Add as a recommended sensitivity dimension in the analytical preamble:

> **(i) Indeed cross-platform validation.** For key findings (entry share, AI prevalence, description length), compute the same metric on Indeed scraped data (~14K rows, ~1K SWE). Indeed has no native seniority but has `seniority_imputed`. If Indeed patterns match LinkedIn, findings are more robust. If they diverge, the finding may be LinkedIn-specific.

### B5. ML/AI archetype entry-level rate

**What:** Add to T09 (archetype discovery) characterization step:

> For each archetype, report entry-level share of known seniority. This is critical for H1: if ML/AI has structurally lower entry share AND grew from 4% to 27%, the aggregate entry decline may be driven by domain composition, not within-domain elimination.

### B6. Update all tasks to prefer LLM columns when available

**What:** The task specs should be updated so that:
- Text analysis defaults to `description_core_llm` (check coverage; the column should be available for all sources after budget allocation)
- Seniority analysis defaults to `seniority_llm` as primary, with native/final/imputed as ablation variants
- Ghost analysis uses `ghost_assessment_llm` where available, with `ghost_job_risk` as fallback
- All tasks check column availability at runtime rather than assuming a specific coverage state

The core preamble already has much of this (the seniority ablation framework, the text quality guidance). The specific change: remove any language in task specs that says "use `seniority_final`" as a fixed instruction. Replace with "use the seniority ablation framework from the preamble."

### B7. Company size analysis (where feasible)

**What:** Add to T16 (company strategies) or T08 (distribution profiling):

> **Company size stratification (where data allows).** `company_size` is available for arshkon (99%) and Indeed (91%), but 0% for asaniczka and scraped LinkedIn. Within arshkon, stratify entry-level share, AI prevalence, and tech count by company size quartile. Do large companies show different patterns from small ones? Use posting volume per company as a proxy for company size in the cross-period analysis where `company_size` is unavailable.

---

## C. Documentation updates

### C1. Task reference: tasks to modify

| Task | Change | Section |
|------|--------|---------|
| T05 | Add platform labeling stability test (B2) | New step |
| T08 | Add domain × seniority decomposition (B1); add company size stratification (B7) | New steps |
| T09 | Add entry-level share per archetype (B5) | Characterization step |
| T14 | Add length-normalized AI additive check (B3) | New note |
| T16 | Add domain dimension to decomposition (B1) | Extend step 4 |

### C2. Task reference: preamble changes

| Section | Change |
|---------|--------|
| Core preamble | Update LLM column guidance to assume availability (B6) |
| Analytical preamble | Add sampling protocol (A2); add Indeed sensitivity dimension (B4) |

### C3. Orchestrator prompt changes

| Section | Change |
|---------|--------|
| Steering the investigation | Add narrative alternatives evaluation (A3) |
| Wave-by-wave guidance | Add verification agent between Wave 2→3 and Wave 3→4 (A1) |
| Gate evaluation | Add "consider alternative framings" to each gate's evaluation questions |

### C4. Update the seniority, text, and ghost column defaults

When `seniority_llm`, `description_core_llm` (all sources), and `ghost_assessment_llm` become available:

1. Core preamble: update "LLM classification columns may or may not be populated" to note they ARE populated
2. Remove all "if available" conditionals — they're now available
3. Shared preprocessing (Agent Prep): rebuild all artifacts with LLM-cleaned text for all sources
4. Seniority ablation: `seniority_llm` becomes the unconditional primary
5. Ghost analysis: `ghost_assessment_llm` becomes primary for T22

---

## Execution order

1. **Now:** Implement A2 (sampling protocol), A3 (narrative alternatives), B1-B5 task additions, C1-C3 doc updates — these are improvements regardless of LLM column availability
2. **After LLM budget runs:** Implement B6 (update all column defaults), C4 (column defaults), rebuild shared artifacts
3. **Re-run exploration from Wave 1.5:** With improved text, seniority, and ghost columns
4. **Add verification agents (A1):** Between Wave 2→3 and Wave 3→4 during the re-run
