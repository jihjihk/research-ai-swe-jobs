# Exploration Task Index

| Task | Agent | Wave | Status | One-line finding |
|---|---|---|---|---|
| T01 | A | 1 | Warning | Core RQ1-RQ3 fields are usable, but `description_core_llm` is absent and 20 columns are >50% null in at least one source. |
| T02 | B | 1 | Warning | Recommend `seniority_final`; it agrees 96.4% with native labels on labeled SWE rows and preserves the junior-share decline direction, but `2024-01` native entry labels are structurally unusable. |
| T03 | A | 1 | Clean | Entry-level SWE sample size is adequate for the main comparison (`2024-04` arshkon `n=773`, `2026-03` scraped `n=561`), with LinkedIn-only analysis required. |
| T04 | B | 1 | Warning | Core SWE counting is usable and dual-flag violations are zero, but the boundary with SWE-adjacent roles remains noisy in the borderline audit. |
| T05 | C | 1 | Warning | The three LinkedIn SWE slices are comparable enough to analyze, but not interchangeable; description length, geography, seniority, and title mix all differ materially by source/period. |
| T06 | C | 1 | Warning | No company exceeds 10% of SWE postings, but concentration is meaningful and company-capping shifts junior-share estimates by about 0.8 pp. |
| T07 | D | 1 | Clean | Geographic representativeness against OEWS is very strong (`r=0.981` to `0.992`), while JOLTS is unavailable for this scrape window because the series ends in `2026-01`. |
| T08 | E | 2 | Clean | The `2024-04` to `2026-03` shifts are generally larger than within-2024 baseline variation, especially for length and AI-tool language. |
| T09 | E | 2 | Clean | All four seniority variants preserve the junior-share decline from `2024-04` to `2026-03`; the direction is stable even when effect size moves. |
| T10 | F | 2 | Clean | Fightin' Words was rerun on `description_core`-first text and the top terms are now reportable-category filtered, with `tech_stack`, `ai_tool`, `sys_design`, `credential`, and `method` dominating the selected results. |
| T11 | F | 2 | Clean | Temporal drift was rerun on SWE-only counters with matched seniority frames; the entry gate counts now match (`2024-04` `n=773`, `2026-03` `n=561`) and the emergent/disappearing lists are gate-safe. |
| T12 | G | 2 | Warning | Junior scope inflation shows up as higher AI-tool prevalence and higher normalized tech density, but not as higher explicit YOE; median junior YOE falls from `3` to `2`. |
| T13 | G | 2 | Clean | Within-company change explains most of the overlap-firm movement: entry share falls `9.1` pp, with `-7.2` pp from within-company change and `-1.9` pp from composition. |
| T14 | H | 2 | Warning | RQ3 is benchmark-sensitive: posting AI requirements exceed Anthropic occupation-level usage shares but remain below Stack Overflow developer adoption benchmarks. |
| T15 | H | 2 | Clean | `ghost_job_risk` is too sparse to drive exclusions (`514` non-low rows out of `1,156,381` LinkedIn-English-date-ok rows); treat it as an audit queue, not a filter. |
| T16 | H | 2 | Warning | AI language rises outside SWE too, but the junior-share decline is strongest in SWE; adjacent and control groups are benchmark checks, not flat nulls. |
| T17 | I | 3 | Clean | Technology shifts are concentrated in AI/LLM tooling and adjacent workflow layers, while core languages and cloud stacks remain relatively stable. |
| T18 | I | 3 | Warning | Description growth is mixed: requirements length rises, but benefits and legal/EEO sections also expand materially, especially in 2026 entry postings. |
| T19 | J | 3 | Clean | A distinct AI-augmented SWE archetype grows from `1.1%` to `12.2%`, with stronger 2026 co-occurrence bundles linking AI, ownership, and systems language. |
| T20 | J | 3 | Warning | The relabeling hypothesis is not supported; `2026` entry postings remain marginally closer to `2024` entry than to `2024` mid-senior postings. |
| T21 | K | 3 | Clean | Senior SWE shifts strongly toward orchestration: “new senior” rises to `48.3%` in `2026-03`, while “classic senior” falls to `47.5%`. |
| T22 | K | 3 | Warning | Metro heterogeneity is real but conditional on 17 metros with enough coverage, and metro-level AI-surge vs entry-decline correlation is weak. |
| T23 | L | 3 | Warning | Ghost-like patterns are driven mainly by template saturation and company repetition; AI-postings carry a higher ghost-like composite score but this remains a diagnostic, not an exclusion rule. |
| T24 | L | 3 | Warning | Embedding similarity does not support convergence: `2026` junior postings are slightly farther from `2024` mid-senior than `2024` juniors are. |

## Gate 1 Notes

- **Gate verdict:** No blockers. Wave 2 can proceed.
- **Seniority recommendation for Wave 2+:** Use `seniority_final` as the default seniority field. Use `seniority_native` only for validation/sensitivity. Do not use `2024-01` asaniczka native labels for entry-level trend claims.
- **Entry-level comparison frame:** Treat `2024-04` (arshkon) vs `2026-03` (scraped) as the meaningful entry-level baseline for RQ1.
- **Column exclusions for cross-period work:** Avoid source-specific search metadata and sparse fields as cross-period inputs: `asaniczka_skills`, `skills_raw`, `company_id_kaggle`, `company_size`, `company_size_raw`, `company_size_category`, `scrape_date`, `scrape_week`, `posting_age_days`, `search_query`, `search_location`, `search_metro_id`, `search_metro_name`, `search_metro_region`, `query_tier`, `work_type`.
- **Sparse-but-usable fields:** Treat `yoe_extracted`, `date_posted`, `real_employer`, and `company_industry` as source-limited and report coverage when used.
- **Text-analysis constraint:** `description_core_llm` is not present in stage 8. Wave 2 text work must fall back to `description` or `description_core` with explicit cleaning and length normalization.
- **Sample definition:** Keep Wave 2 on LinkedIn-only, `is_english = true`, `date_flag = 'ok'`, with source-aware interpretation and company-capped robustness checks where composition matters.

## Gate 2 Notes

- **Gate verdict:** Blockers resolved. `T10` and `T11` were rerun successfully with repaired scoping and filtering.
- **Why the rerun is acceptable:** `T10_category_summary.csv` is now dominated by reportable categories instead of generic noise, `T11_disappearing_terms.csv` no longer contains the earlier non-SWE artifacts, and the entry-level gate counts match the expected `n=773` and `n=561` values.
- **Wave 3 findings to carry forward:** Keep `seniority_final`; use `2024-04` vs `2026-03` for entry claims; treat within-company change as real but partial (`T13`); treat scope inflation as AI/tooling and tech-density heavy rather than YOE-heavy (`T12`); and treat RQ3 and cross-occupation comparisons as benchmark-sensitive (`T14`, `T16`).

## Gate 3 Notes

- **Gate verdict:** No blockers. Wave 4 can proceed.
- **Technology evolution:** `T17` shows the strongest growth in AI/LLM tooling, retrieval infrastructure, and adjacent workflow layers rather than in the core language/cloud backbone.
- **Description anatomy:** `T18` indicates 2026 description growth is partly substantive and partly administrative; requirements grow, but benefits and legal/EEO text also account for a large share of the added length.
- **Archetypes:** `T19` supports an “AI-augmented SWE” archetype that grows materially in 2026 and bundles AI language with ownership and systems cues.
- **Relabeling / convergence:** `T20` and `T24` both fail to support a strong relabeling or narrowing-gap story. Treat this as a negative finding to carry into synthesis rather than a blocker.
- **Senior archetype:** `T21` strongly supports the management-to-orchestration shift in senior SWE postings.
- **Metro heterogeneity:** `T22` suggests geographic moderation but not a simple “AI surge explains entry decline” metro pattern.
- **Ghost requirements:** `T23` strengthens the case for template repetition and company-level requirement cloning as the main ghost-like mechanism; keep it as a diagnostic sensitivity rather than a sample filter.
