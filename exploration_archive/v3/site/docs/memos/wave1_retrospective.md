# Wave 1 Retrospective: Lessons for Future Reruns

Date: 2026-04-05

## Changes made to exploration docs

### 1. Preamble split into core + analytical (task-reference-exploration.md)

**Before:** One monolithic preamble for all agents including text analysis hygiene and sensitivity framework.
**After:** Core preamble (short, for all waves) + analytical preamble (text hygiene, sensitivity framework, Wave 2+ only). Wave 1 agents get only the core preamble, saving context and reducing noise.

### 2. All hardcoded row counts removed

**Before:** Preamble, plan, and orchestrator prompt all contained specific row counts (e.g., "~33K SWE", "4.5K scraped SWE") that went stale as the scraper ran.
**After:** Agents are told to query the data for current counts. No specific numbers in any agent-facing doc.
**Files changed:** task-reference-exploration.md, plan-exploration.md, prompt-exploration-orchestrator.md

### 3. T01/T02 restructured to reduce overlap

**Before:** Both T01 and T02 computed missingness-by-source tables. Redundant work.
**After:** T01 = data profile + column coverage + constraint mapping. T02 = seniority comparability audit only (the asaniczka associate question). No duplicated missingness analysis.

### 4. Text column guidance revised: description_core_llm is primary

**Before:** "Use description_core uniformly for cross-period analysis."
**After:** Decision tree based on whether the analysis feeds into later stages:
- Text-dependent + feeds downstream: use description_core_llm (restricts to Kaggle rows, but much higher quality)
- Text-dependent standalone: use description_core_llm primary, description_core fallback
- Non-text analyses: all rows
**Rationale:** description_core retains substantial boilerplate garbage (~44% removal accuracy). description_core_llm is significantly better despite limiting row coverage.

### 5. Sensitivity dimension (d) updated

**Before:** Primary = description_core. Alt = description.
**After:** Primary = description_core_llm. Alt = description_core. Reflects the actual quality hierarchy.

### 6. Output conventions expanded

**Before:** Figures + tables + reports.
**After:** Added scripts (`exploration/scripts/TASK_ID_descriptive_name.py`) for reproducibility.

### 7. Report template made flexible

**Before:** Fixed template with mandatory sections including sensitivity checks.
**After:** "Use a structure appropriate to the task" with minimum required sections listed. Wave 2+ adds sensitivity checks section.

### 8. T06 elevated within-company decomposition

**Before:** Within-company seniority comparison was step 3 of 6, not highlighted.
**After:** Explicitly framed as "core analytical output" with instruction to compare within-company vs aggregate decline and state the direction of composition effects.

### 9. T07 reordered: feasibility table first

**Before:** External benchmarks first, power analysis second.
**After:** Feasibility table is "Part A (primary output, drives all downstream decisions)." External benchmarks are "Part B (useful context, not blocking)."

### 10. Shared preprocessing uses description_core_llm

**Before:** Agent Prep's cleaned text artifact used description_core for all rows.
**After:** Uses description_core_llm where available, falls back to description_core, tracks which source was used via a `text_source` column. This lets downstream agents filter to high-quality text rows when needed.

## What these changes mean for reruns

If Wave 1 is rerun with these changes:
- Agents A should produce a cleaner T01 (no redundant missingness table) and a more focused T02 (seniority comparability only)
- All agents will query actual data for counts, avoiding stale-number confusion
- Wave 2+ agents will get a shorter, more relevant preamble
- Text analyses will prioritize description_core_llm, producing higher-quality results on a smaller but cleaner sample
