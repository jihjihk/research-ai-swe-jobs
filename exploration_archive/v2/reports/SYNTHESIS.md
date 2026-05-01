# SYNTHESIS: Exploration handoff for analysis

## Bottom Line
The core descriptive story is stable: SWE postings become more junior-constrained and more AI/orchestration-heavy from 2024 to 2026, but the magnitude depends on seniority definitions, source mix, and benchmark choice. `seniority_final` is the right default seniority field, LinkedIn-only `is_english = true` and `date_flag = 'ok'` is the right cross-period frame, and `description_core` is the current default text field because `description_core_llm` is not yet present in stage 8.

## Recommended Analysis Sample
- Primary frame: `source_platform = 'linkedin'`, `is_english = true`, `date_flag = 'ok'`.
- RQ1/RQ2 main sample: `is_swe = true`, with `seniority_final` as the seniority field and `company_name_effective` / `company_name_canonical` for company work.
- Entry-level trend baseline: compare `2024-04` arshkon vs `2026-03` scraped; use `2024-01` as a wording/compositional check only because asaniczka has no native entry labels.
- Use `is_swe_adjacent` as the comparison group for task migration and RQ2 guardrails; use `is_control` for benchmark checks, not as a causal counterfactual.
- Use canonical postings for posting-level composition and daily observations only when a day-level trend is explicitly needed. Do not mix stages that change row cardinality.

## Seniority Recommendation
- Default: `seniority_final`.
- Validation/sensitivity: `seniority_native` and the high-confidence subset.
- Do not use 2024-01 asaniczka native labels for entry-level trend claims.
- `seniority_final` preserves the junior-share decline direction across variants, so the sign is stable even if the magnitude moves.

## Data Quality Verdicts By RQ
- RQ1: High confidence on direction, moderate confidence on magnitude. Junior share falls, but scope inflation is better captured by AI-tool language, tech density, and bundle shifts than by explicit YOE increases.
- RQ2: High confidence on direction. Requirements migrate toward AI tooling, systems, ownership, orchestration, and cross-functional language.
- RQ3: Moderate confidence and benchmark-sensitive. Employer AI requirements exceed Anthropic occupation-level usage, but still lag Stack Overflow developer self-reports.
- RQ4: Ready for interviews, not text-only inference. The artifact set now gives strong prompts for probing inflation, orchestration, and ghost-requirement mechanisms.

## Preliminary Findings
- RQ1: Junior share drops materially from 2024-04 to 2026-03; within-company change explains most of the overlap-firm movement, with composition still nontrivial.
- RQ1: Junior scope inflation shows up in normalized tech density and AI-tool language more than in higher explicit YOE.
- RQ1: Senior postings shift from people-management toward orchestration and technical coordination; the "new senior" share rises sharply by 2026-03.
- RQ2: AI/LLM tooling, retrieval, microservices, CI/CD, ownership, and stakeholder language are the clearest movers.
- RQ3: The employer-side AI signal is strongest in 2026-03 and is clearer for SWE than for adjacent or control groups, but the economy-wide benchmark is not flat.
- RQ4: Ghost-like patterns are mostly template saturation and company repetition, which is consistent with aspirational or cloned requirements rather than only real workflow change.

## Key Tensions To Resolve
- YOE decrease vs scope increase: juniors are not asking for more explicit YOE, but they are asking for denser and more AI-heavy scope.
- Field-wide vs SWE-specific: AI language rises outside SWE too, so the analysis must separate broad labor-market salience from SWE-specific restructuring.
- Within-company vs composition: within-firm change is real, but composition still matters, especially because overlap coverage is incomplete.
- Relabeling vs semantic drift: T20 and T24 do not support a clean relabeling story; 2026 juniors remain closer to historical juniors than to mid-senior postings.
- Ghost requirement prevalence vs exclusion: ghost-like patterns are common enough to diagnose, but not strong enough to justify hard filtering.

## Confounders And Sensitivity Requirements
- Description length grew substantially, and some of that growth is boilerplate, benefits, and legal/EEO text.
- Asaniczka has no entry-level native labels, so it cannot anchor the junior trend.
- Aggregator and staffing contamination matters, especially for company repetition and ghost-like patterns.
- Company capping changes junior-share estimates, so company-capped robustness checks should be reported alongside the raw series.
- Metro effects are real but weakly tied to the overall AI surge; geography should be treated as a moderator, not a single explanation.
- Length-normalized rates are required for keyword work; report binary any-mention rates and per-1,000-character rates together.

## Technology Evolution Summary
- The strongest growth is in AI/LLM tooling, RAG, agentic language, and adjacent workflow layers.
- Core languages, cloud stacks, and baseline infrastructure remain comparatively stable.
- Older frontend and legacy tooling declines, which supports a scope-shift story rather than simple stack replacement.

## Metro Heterogeneity Summary
- Geography is representative enough to support analysis, but the entry-share decline is not uniform across metros.
- Large tech metros show the steepest junior-share drops, while the correlation between AI surge and entry decline is weak.
- Metro should therefore be used as a moderator and sensitivity split, not as a single explanation for the trend.

## Senior Archetype Characterization
- Senior SWE language increasingly emphasizes orchestration, coordination, review, architecture, and AI-enabled leverage.
- Management language is sparse, and the management-to-orchestration ratio compresses over time.
- The rising "new senior" archetype is the best label for the 2026 direction of travel.

## Pipeline Issues Still Open
- `description_core_llm` is still absent from stage 8, so all text work remains on the rule-based layer.
- Stage 10-12 LLM augmentation is not yet a production-validated replacement for the current text layer.
- `ghost_job_risk` remains a diagnostic, not a filter.
- Stage 5 native backfill is fixed, but the analysis should still prefer `seniority_final` and keep native labels as validation only.

## Action Items For Analysis
- Build the main RQ1/RQ2 analyses on LinkedIn-only, English, date-ok SWE rows with `seniority_final`.
- Use `2024-04` arshkon as the entry-level baseline, with `2026-03` scraped as the comparison period.
- Report all keyword results with company-name stripping and length normalization.
- Treat `is_swe_adjacent`, `is_control`, aggregator rows, and company caps as robustness layers.
- Read the T25 artifacts first when preparing interview prompts, especially the inflated junior cards and the senior archetype chart.
