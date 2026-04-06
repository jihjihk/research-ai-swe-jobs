# Wave 2 Retrospective: Lessons for Future Reruns

Date: 2026-04-05

## Changes implemented in task-reference-exploration.md

### Preamble changes

1. **Added `boilerplate` semantic category** — benefits, salary, compensation, culture, mission, etc. Prevents the "95 of 100 terms are uncategorized" problem from T12.

2. **Strengthened sensitivity investigation guidance** — "When a finding IS materially sensitive, investigate WHY." Agents should dig into mechanisms, not just report pass/fail.

3. **Added text source discipline requirement** — agents MUST use `description_core_llm` as primary, report `text_source` distribution, NEVER mix sources without explicit reporting. `description_core` is a sensitivity check, not the primary.

4. **Noted markdown formatting in instrument comparison** — section classifiers must handle both plain-text and markdown headers.

5. **Updated text column guidance to be coverage-aware** — agents should query `llm_extraction_coverage` to determine current coverage rather than assuming Kaggle-only. Anticipates LLM budget allocation for scraped data.

### Shared preprocessing changes

6. **Added within-2024 calibration table** as a shared artifact (step 6 in Agent Prep). Pre-computes calibration ratios for ~30 common metrics so Wave 2+ agents load it instead of recomputing independently.

7. **Added note about re-running after LLM budget allocation** — shared artifacts should be rebuilt when `description_core_llm` coverage changes.

### Task-level changes

8. **T09 sampling rebalanced** — balanced period representation (~2,700/period) instead of proportional, prefer arshkon over asaniczka for 2024, prefer `text_source = 'llm'`. Prevents asaniczka domination.

9. **T09 saves archetype labels** as shared artifact — `swe_archetype_labels.parquet` enables downstream tasks (T11, T16, T20) to stratify by domain.

10. **T11 adds management term deep dive** — must report top 10 trigger terms, define `management_strong` vs `management_broad` tiers. Validates the +31pp headline finding.

11. **T11 adds domain-stratified scope inflation** — tests whether scope inflation varies by T09 archetype (ML/AI vs Frontend vs Embedded vs Data). High-value analytical cross-reference.

12. **T12 sequenced after T13** — uses T13's section classifier to strip boilerplate sections before corpus comparison. Produces section-filtered Fightin' Words alongside full-text version.

13. **T12 relabeling diagnostic elevated** — framed as testing "downward task migration vs parallel skill expansion." The interpretation matters for the paper narrative.

14. **T15 convergence analysis improved** — trimmed centroids (outlier-robust), density-contour UMAP visualizations with movement arrows, domain-stratified convergence, publication-quality figures. Reframed as "structural map + convergence" rather than convergence-only.

## Lessons not requiring doc changes (behavioral)

- Agents should always check whether shared artifacts exist before recomputing (T14 may have recomputed the tech matrix)
- The relabeling diagnostic (Entry 2026 vs Mid-senior 2024) is underutilized — agents should connect it back to the scope inflation narrative
- Section anatomy (T13) is a prerequisite for clean text analysis — the sequencing change addresses this structurally
