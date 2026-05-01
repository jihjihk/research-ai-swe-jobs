# Gate 0 — Pre-Exploration Memo

Date: 2026-04-17
Author: Research advisor (Claude Opus 4.7)
Status: Pre-dispatch. No data queries yet; this memo captures priors and pre-commitments before any agent reads the parquet.

## Purpose of this memo

The orchestrator's job is to steer a discovery exploration whose findings will shape a substantive paper about SWE labor market restructuring. This memo records (a) what I am confident vs. uncertain about *before* the data speaks, and (b) the ablation discipline I am locking in now so it cannot drift under time pressure in later waves.

## The initial hypothesis (a starting point, not a conclusion)

`docs/1-research-design.md` hypothesizes three patterns over 2024→2026:
1. Junior scope inflation — narrowing junior rung, more senior-flavored requirements in entry postings
2. Senior archetype shift — from people-management to AI-enabled orchestration / review / architecture
3. Employer-requirement / worker-usage divergence — employer AI requirements outpace observed usage, consistent with anticipatory restructuring

These are hypotheses from before systematic analysis. Finding the *correct* narrative matters more than confirming this one. At each gate I will evaluate whether the evidence supports the initial framing, a modification, or an entirely different story. Plausible alternatives to weigh explicitly at every gate:

- **Expansion vs decline framing.** Did AI broaden the SWE skill surface (more asks per posting) or hollow out entry roles (fewer junior postings)? Same data, different emphasis.
- **Market recomposition vs firm restructuring.** Are aggregate changes driven by *different* companies posting in 2026 (composition), or by the *same* companies changing what they ask for (within-company)? T06 and T16 decompose this.
- **Platform evolution vs real demand.** Did LinkedIn templates or scraper formatting change how postings *read* without changing what employers *want*? Kaggle text is HTML-stripped; scraped text preserves markdown — an instrument difference, not a market change.
- **Domain shift vs seniority shift.** Did the market move toward ML/AI (which is structurally less junior-heavy) rather than entry roles disappearing? T09's archetype discovery and T08's domain×seniority decomposition both test this.
- **LLM-authored descriptions as a mechanism.** T29 explicitly tests whether recruiters using LLMs to draft JDs is unifying several "change" signals (length growth, tech density, scope vocabulary). If this hypothesis is supported, the paper must front-load it.

The paper's credibility will come from honest weighing, not from picking the most dramatic framing.

## What I am MOST confident about going into Wave 1

1. **The 56% description length growth is real and is the single biggest confound.** It is large, it is cross-source, and it biases every raw-count text metric toward showing "more" in 2026. Every text-dependent claim must be length-normalized (rate per 1K chars, binary presence) *and* section-decomposed (requirements vs boilerplate — T13 step 2–3).
2. **Asaniczka has zero native entry labels.** This is a structural property of that dataset, not a quality issue. Any `seniority_native`-based entry comparison that pools asaniczka is broken by construction. Arshkon is the only 2024 source with native entry; the YOE-based proxy is the only label-independent entry instrument.
3. **The three sources are three different instruments.** arshkon (curated Kaggle snapshot, April 2024, entry labels, industry join, ~124K), asaniczka (large Kaggle snapshot, Jan 2024, no entry, separate description join, ~1.35M), scraped (daily LinkedIn+Indeed, March 2026+, markdown-formatted descriptions, 26-metro search frame). Their "comparability" is dimension-dependent and will be task-specific.
4. **Company concentration will dominate aggregate findings unless controlled.** Prior-run history (per T06 spec) found ~23% of 2026 entry-labeled postings from 6 companies posting the same description 4-25 times each. This is not a rare pathology — it is the baseline shape of the data. Every corpus-level aggregate (term frequencies, topic models, co-occurrence networks, embedding centroids) needs company capping; every per-row metric needs the aggregator and entry-specialist filters.
5. **`description_core_llm` is the only cleaned-text column.** The rule-based `description_core` was retired (~44% accuracy caused downstream agents to mix cleaned and uncleaned text). LLM coverage is not universal and is gated by budget; text-sensitive analyses restrict to `llm_extraction_coverage = 'labeled'`, and we report coverage per cell.
6. **Seniority is a joint rule+LLM column.** `seniority_final` combines Stage 5 strong-keyword rules with Stage 10 LLM calls for routed rows. `seniority_final_source` tracks provenance. This is the label-based primary, but it is *not* sufficient on its own — T30's panel (J1–J4 junior, S1–S4 senior) is required for every seniority-stratified claim.

## What I am LEAST confident about going into Wave 1

1. **Whether J1 (`seniority_final = 'entry'`) is well-powered.** If the entry-only cells in arshkon and scraped are small (say, <300 per period at the LinkedIn SWE default filter), MDE will be too large to support seniority-stratified 2024→2026 comparisons on J1 alone. In that case J2 (entry+associate) becomes the Wave 2 primary, with J1 a sensitivity. T07 × T30 cross-tab settles this.
2. **How large the within-2024 arshkon-vs-asaniczka gap is on common metrics.** This is the noise floor. If it is large (SNR < 2 for most metrics), most 2024→2026 "changes" are not cleanly above instrument noise and the paper's headline claims shift toward the ones that *do* exceed the floor (likely AI-related surface vocabulary — which is also the most novel piece anyway).
3. **SWE classification stability across sources.** Stage 5 `swe_classification_tier = 'title_lookup_llm'` has elevated FP rate; ~9–10% of the SWE sample sits in this tier (historical prior). If tier composition changed between periods, sample composition changes too. T04 tells us whether the sample is stable enough to pool tiers or whether we must restrict to regex + embedding_high only.
4. **Whether the scraped window is long enough for a "2026" claim.** `period = 'scraped'` is a growing window; the actual date range at dispatch time is whatever the pipeline last produced. T19 will characterize within-window stability; until then, all "2026" claims carry a "this scraped window, not all of 2026" caveat.
5. **Whether the dominant natural structure is seniority, domain, or period.** T09's NMI on clusters × seniority/period/domain settles this. If domain NMI is 5–7× larger than seniority NMI (plausible prior based on spec hints), the paper's framing shifts toward *domain-led restructuring* with seniority-level effects conditional on domain — which is a fundamentally different story than pure "junior rung narrowing."
6. **Whether "scope inflation" is real content change, aggregator artifact, or LLM-authored recruiter text.** T13 isolates requirements from boilerplate; T06 separates aggregator patterns; T29 tests the LLM-authorship mechanism. Any one of these three could absorb a large share of the apparent "change." All three must clear before a scope-inflation headline is defensible.

## What would change my assessment of the project's direction

- **Scope-inflation collapses under T13 section decomposition.** If requirements-section length barely moved and the 56% growth is all boilerplate/benefits, the RQ1 junior-scope-inflation construct is weakened. Paper pivots toward domain recomposition (RQ1b) and/or the ghost/aspirational thread (RQ2b).
- **Control occupations show the same AI mention surge (T18).** If SWE-specificity fails, the "SWE restructuring" framing collapses to a field-wide template evolution story.
- **Company composition explains >50% of the entry-share decline (T16).** Then the paper's headline cannot be "employers reduced junior hiring" — it has to be "which companies post on LinkedIn changed, and that's what drove the aggregate."
- **T30 panel shows direction flips between J1 and J2.** If entry-only narrows but entry+associate widens, we have a relabeling story, not a scope-inflation story — and the paper reframes around *seniority-labeling evolution* as a primary finding.
- **T29 LLM-authorship signal is strong.** If LLM-style postings drive the length, AI-mention, and scope-vocabulary growth, a substantial share of Wave 2's "changes" are recruiter-tooling artifacts. The paper must either front-load this as a methodological contribution or re-run the headlines on low-LLM-score subsets.
- **T09 archetype NMI dominated by domain.** Then the "market organizes by seniority" prior is wrong, and the paper must treat seniority effects as conditional on domain rather than primary.

I am holding all of these open. None is yet evidence; they are conditions that would meaningfully change which story the paper tells.

## Pre-committed ablation dimensions (NON-NEGOTIABLE FOR WAVE 2+)

These are locked in now so that no Wave 2+ agent — under time pressure, deliverable fatigue, or optimistic interpretation — can quietly drop them. Every Wave 2+ task output *must* report its headline findings under these ablations. A finding that cannot be reported under these ablations is not yet a finding.

1. **T30 seniority panel.** Every seniority-stratified headline must be reported as a 4-row ablation:
   - Junior claims: J1 (`seniority_final = 'entry'`), J2 (`seniority_final IN ('entry','associate')`), J3 (`yoe_extracted <= 2`), J4 (`yoe_extracted <= 3`).
   - Senior claims: S1 (`seniority_final IN ('mid-senior','director')`), S2 (`seniority_final = 'director'`), S3 (title-keyword senior), S4 (`yoe_extracted >= 5`).
   - Robustness rule: 4-of-4 unanimous or 3-of-4 agreement in direction *and* effect-size spread within 30% = robust. Split or contradictory = the disagreement itself is the finding; investigate the mechanism rather than pick a side. Load `exploration/artifacts/shared/seniority_definition_panel.csv`; do not re-derive.

2. **Aggregator exclusion (sensitivity dimension a).** Every aggregate must be reported with and without `is_aggregator = true`. Aggregators have systematically different descriptions, seniority patterns, and template saturation.

3. **Company capping for corpus aggregates (sensitivity dimension b).** Cap at 20–50 postings per `company_name_canonical` as the primary specification for any analysis aggregating over a corpus (term frequencies, topic models, co-occurrence networks, embedding centroids). For per-row metrics (rates, distributions) or company-level analyses, capping is not appropriate; treat it as a sensitivity check or N/A.

4. **Within-2024 calibration (sensitivity dimension f).** Mandatory diagnostic for every 2024→2026 metric: also compute the arshkon-vs-asaniczka value of the same metric. SNR = cross-period effect / within-2024 effect. SNR < 2 → flagged as "not clearly above instrument noise."

5. **Semantic keyword precision check.** For any keyword indicator used in a prevalence, density, or rate metric, sample 50 matches stratified by period (25/25) and judge precision *semantically* — read the surrounding sentence, not the regex echo. Tautological precision claims ("the regex matches its own regex") are not permitted. Sub-patterns below 80% are dropped; the compound is rebuilt and re-run.

6. **Composite-score correlation check (matched deltas).** Every matched-delta claim using a composite score (authorship drift, ghost index, length composite) must report per-component × outcome correlations. If any component correlates r > 0.3 with the outcome, matching is confounded on that dimension — drop the component or report the matched result under ablated score versions.

7. **Prevalence citation transparency.** Every prevalence, density, share, or effect-size number must be cited with (a) the exact pattern or column definition, (b) the subset filter (sample, cap, aggregator exclusion, LLM-coverage restriction), and (c) the denominator ("of all" vs "of known-seniority" vs "of LLM-labeled"). Cross-task citations that combine numbers from different patterns or subsets into a single cell are prohibited.

8. **Text-source discipline.** `description_core_llm` (filtered to `llm_extraction_coverage = 'labeled'`) is the primary text column for text-dependent analyses. Raw `description` is acceptable only for analyses demonstrably insensitive to boilerplate phrasing (binary keyword presence). Mixing rows from the two text sources without reporting the split is prohibited. No legacy rule-based `description_core` content may be used — the column was retired.

## Data feasibility priors

Treat these as priors to be confirmed or replaced by Wave 1:

- **arshkon SWE entry count (T30 J1):** prior 300–500. If <300, J1 alone cannot support 2024→2026 comparisons without very large effect sizes.
- **scraped LinkedIn SWE entry count:** prior is unknown but likely larger than arshkon given the growing window. If <500, J1 is still thin; use J2.
- **`description_core_llm` coverage:** prior is LLM frame is sticky-balanced across sources, but Indeed is excluded and scraped Kaggle-pre-2026 bins may be thin. T01 step 5 reports actual coverage by source.
- **`seniority_final` unknown rate outside LLM frame:** prior is high for rows the Stage 5 title keyword rules miss and that weren't routed. Indeed SWE is particularly thin (no LLM routing). T03 step 1 reports this.
- **Multi-location representatives:** `is_multi_location = true` rows have `metro_area = NULL` and drop out of metro rollups naturally. Expect ~thousands in scraped; they must be reported separately, not imputed.

## Agent dispatch decisions for Wave 1

- **Model / effort.** Every sub-agent is dispatched at the highest available reasoning tier. On this Claude Code runtime, that is the `general-purpose` subagent on Claude with max effort. Do not downgrade for latency — Wave 1 feeds every subsequent gate.
- **Memory discipline reminder.** Every prompt explicitly reminds the agent of the 31 GB RAM limit, the DuckDB/pyarrow requirement, and the prohibition on loading the full parquet into pandas. Multiple agents reading the same parquet simultaneously could OOM if they all loaded into pandas.
- **Shared-output paths.** All reports go to `exploration/reports/T##.md`; tables to `exploration/tables/T##/`; figures to `exploration/figures/T##/`; scripts to `exploration/scripts/T##_*.py`. Shared artifacts go to `exploration/artifacts/shared/` with the exact filenames called out in the task specs (Wave 2 agents will load by path).

## What I expect to be asked at Gate 1

- "Which seniority definition is Wave 2 primary?" — decided by T30 × T07 cross-tab. Write into Wave 2 agent prompts; do not let it be re-litigated.
- "Is the SWE sample defensible?" — T04 answers. If tier composition is unstable, restrict to regex + embedding_high for the primary spec.
- "Is arshkon a defensible 2024 baseline on its own, or must we pool asaniczka?" — T02, T03, and T05 together. If asaniczka `associate` fails the T02 substitutability test, we stick to arshkon-only for any `seniority_native`-dependent sanity check and use YOE proxies for pooled comparisons.
- "What fraction of 2026 entry postings is concentrated in few employers?" — T06. If concentration is high, every entry-share headline must report the deconcentrated variant.
- "Which planned analyses are underpowered?" — T07 feasibility table. Underpowered analyses get deprioritized in Wave 2 rather than wasted effort.

Wave 1 dispatch follows.
