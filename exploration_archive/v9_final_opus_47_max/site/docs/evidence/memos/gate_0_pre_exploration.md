# Gate 0 — Pre-Exploration Research Memo

Date: 2026-04-20
Author: Orchestrator
Written: Before Wave 1 dispatch.

This memo captures the orchestrator's analytical state before any sub-agent has been dispatched. It exists so that future gate memos can be read against an honest record of what the orchestrator expected versus what the data actually showed.

---

## What we are studying

A longitudinal study of SWE posting content and composition across two 2024 LinkedIn snapshots (kaggle_arshkon, kaggle_asaniczka) and a growing 2026 scrape. The initial research design (docs/1-research-design.md) hypothesizes three linked narratives:

- **RQ1 Junior rung narrowing.** Fewer entry-level postings and inflated junior scope.
- **RQ2 Senior archetype shift.** From people-management toward AI-enabled orchestration, review, and systems responsibility.
- **RQ3 Employer-requirement / worker-usage divergence.** Postings demand AI faster than workers actually use it.
- RQ4 (interview-driven mechanisms) is out of scope for the computational exploration.

The orchestrator treats RQ1-RQ3 as *hypotheses*, not conclusions. The exploration's primary product is an honest picture of what the data shows and a paper narrative that the data actually supports.

## Prior beliefs (pre-Wave-1)

These are the orchestrator's priors about what the data will show. Wave 1-2 evidence should move them.

### On the headline narratives

- **Junior scope inflation is the most vulnerable claim.** Description length grew substantially between 2024 and 2026. If the growth is concentrated in boilerplate, marketing, and benefits sections rather than in requirements sections, most of the apparent "scope inflation" is an instrument/format artifact. T13's section-anatomy analysis is the single most important diagnostic for whether RQ1 survives contact with the data. My prior: moderate — half of the observed scope-expansion surface is real, half is length/boilerplate.
- **Senior archetype shift has a cleaner signal.** Even if length is confounded, the RELATIVE mix of management-language density and orchestration-language density within senior postings can move independently of total length. Also senior/junior are both affected by the same length confound, so *differential* senior-vs-junior patterns are more robust. My prior: moderate-to-high — some senior shift is real, but the "management → orchestration" framing may be too stylized.
- **Posting-usage divergence is the strongest bet but most dependent on external benchmarks.** Employers typically lead usage in any adoption curve; the direction of the gap is nearly a foregone conclusion. The contribution is in the MAGNITUDE and in cross-occupation heterogeneity (T23 + Wave 3.5 T32). My prior: strong on direction, weak on magnitude.

### On the data itself

- **Within-2024 cross-source variation will be substantial.** arshkon and asaniczka are different instruments (LinkedIn snapshots pulled differently, different coverage windows, and asaniczka has no native entry labels). The within-2024 Cohen's d for many metrics will not be zero. Whether it is 10% or 50% of the cross-period effect determines whether headline findings survive the SNR test.
- **Asaniczka `associate` is probably NOT a clean junior proxy.** The preprocessing schema notes asaniczka has only `mid-senior` and `associate`, and `associate` is likely a catch-all for "LinkedIn said not entry, not senior." The YOE distribution check in T02 will probably show asaniczka `associate` skewing older than arshkon `entry`. If the verdict is "not usable," the 2024 baseline for entry-level claims rests on arshkon alone for label variants and on YOE-based variants (J3/J4) for pooled 2024.
- **T30 junior-panel direction will probably NOT be unanimous.** The J1/J2 label variants pick up asaniczka's LLM-derived entry labels; J3/J4 are YOE-based and filter to `llm_classification_coverage = 'labeled'`. The known LLM-frame selection artifact (the selected core is non-random w.r.t. junior signal) probably pushes J3/J4 in a different direction than J1/J2 under a naive read. The orchestrator must not let agents paper over disagreement — material YOE-vs-label disagreement IS a finding.
- **Scraped 2026 is probably dominated by new entrants.** T06's returning-companies cohort is likely a minority of the 2026 firms. This means the full-corpus 2024-to-2026 deltas include a composition-shift component, not just within-firm rewriting. T16's decomposition and T37's returning-cohort sensitivity become the structural defense of every longitudinal claim.

### On what will surprise us

- **T09 archetypes will probably organize by tech domain more strongly than by seniority.** NMI(clusters, domain) is likely larger than NMI(clusters, seniority). If true, the paper's natural narrative axis is technology ecosystem restructuring, with seniority as a cross-cut — not the reverse.
- **The ML/AI archetype probably grew disproportionately in 2026.** If the AI-domain archetype is also structurally lower-entry-share than other archetypes, the aggregate junior decline is partly a between-domain recomposition effect (T08 step 7, T28). This reframes RQ1 mechanistically.
- **Recruiter-LLM authorship (T29) might explain a substantial fraction of content change.** If the low-LLM-authorship-score subset of 2026 postings shows muted headline effects, the "employer restructuring" story weakens toward "recruiter-tool-mediated JD inflation" — itself a paper-lead-material finding, but a different paper.
- **Cross-occupation DiD (T18, T32) may show substantial spillover.** If control occupations show similar AI-requirement and length trends, the "SWE-specific" story breaks down. This would still be a real paper, but a general-labor-market paper, not a SWE-restructuring paper.

### On what the paper's lead could be

I have not committed to a lead. Plausible leads the data could support, ranked by my prior probability:

1. **Employer-requirement / worker-usage divergence (RQ3 + T32 cross-occupation).** Strongest prior. If T32 generalizes, this becomes a general-AI-labor-market finding, possibly the paper's centerpiece.
2. **Domain-driven restructuring, not seniority-driven.** If T09 finds domain-dominant clusters and T28 finds between-domain recomposition drives aggregate junior decline, this reframes RQ1 into something more interpretable.
3. **Senior archetype shift.** Moderate prior, robust to length confounds if differential.
4. **Scope inflation / junior rung narrowing.** Lowest prior conditional on T13's section analysis breaking one way. If T13 shows requirements-section expansion, this holds; if it shows boilerplate expansion dominates, this collapses into a methodological finding.
5. **Recruiter-LLM authorship mediation.** Only the lead if T29 finds strong signal AND the low-score subset analyses make Wave 2 findings disappear. High-uncertainty but high-novelty.

## Confidence assessment

### Most confident

- Data exists and is queryable via DuckDB within the 31GB budget.
- Description length grew substantially between 2024 and 2026. The question is WHY, not whether.
- Asaniczka has no native entry labels; this is a binding constraint for any `seniority_native`-dependent analysis.
- Within-2024 calibration will be the load-bearing sensitivity dimension; the paper lives or dies on (cross-period effect) / (within-2024 effect) ratios.
- Some planned analyses will be infeasible under power; T07 will tell us which.

### Least confident

- Which T30 panel variant (J1-J4, S1-S4) is the primary workable slice for each comparison. Default is J3/S4 but could be overridden by T07 MDE.
- Whether scope inflation will look directional or bidirectional across panel variants.
- Whether T09 archetypes are method-robust (BERTopic vs NMF agreement).
- Whether the LLM-frame selection artifact materially biases text-based findings.
- Whether any "emergent senior role" (T21 + T34) is a real structural feature or a clustering artifact.
- Whether we'll hit OOM on the embedding step in Wave 1.5 Prep.

### What would update the plan

- **T01/T05 coverage gaps** → if a planned column has <50% coverage on a source used for comparison, the analysis category collapses or needs new methods.
- **T02 verdict that asaniczka `associate` is not a junior proxy** → label-based junior variants cannot pool 2024; J1 stays arshkon-only for label-based claims.
- **T06 finding most 2026 firms are new entrants** → the paper's main defensibility claim becomes T37's returning-cohort retention ratios; Wave 3.5 T37 gets promoted to paper-central rather than robustness-appendix.
- **T07 feasibility showing J3 underpowered** → promote J4 or J2 to primary for specific comparisons.
- **T30 showing asaniczka senior S4 shares diverge from arshkon within-2024 noise floor** → arshkon-only senior sensitivity becomes mandated, not optional.

## Pre-committed ablation dimensions

These are non-negotiable for every Wave 2+ agent. The orchestrator will enforce them in dispatch prompts and V1/V2 will audit adherence. This pre-commitment prevents ablation discipline from drifting under time pressure.

1. **T30 seniority panel.** Every seniority-stratified headline must report J1/J2/J3/J4 (junior claims) or S1/S2/S3/S4 (senior claims) as a 4-row table. J3/S4 are primary (YOE-based from `yoe_min_years_llm`); J1/J2/S1/S2 are label-based sensitivities. Lead claims require unanimous or 3-of-4 directional agreement; 2-of-4 or split panels require mechanistic investigation of the disagreement, not averaging. Load the canonical panel from `exploration/artifacts/shared/seniority_definition_panel.csv`.

2. **Aggregator exclusion (dim a).** Primary include-all; alternative exclude `is_aggregator = true`. Report both. T22 is the exception — it makes aggregator-vs-direct the primary axis.

3. **Company capping (dim b) for corpus aggregates.** 20-per-company as default cap; 50 where per-firm data density is required (T14, T35); 10 for per-company × per-title × period units (T31); not applicable for per-row metrics or company-level analyses. Document the chosen cap in task methods.

4. **Within-2024 calibration (dim f).** Mandatory for every 2024→2026 headline. SNR = (cross-period effect) / (within-2024 effect). Ratio < 2 must be flagged "not clearly above instrument noise."

5. **Semantic keyword precision (analytical preamble #6).** Any keyword pattern cited in a prevalence, density, or rate metric must be validated on a 50-row stratified semantic sample (25 per period), read in surrounding sentence context. Sub-pattern precision ≥ 80% required. Precision claims that only verify "regex matches its own regex" are tautological and disallowed. V1 will rebuild any tautologically-validated pattern.

6. **Composite-score correlation checks.** Any matched-delta analysis must report per-component × outcome correlations. Component r > 0.3 with outcome → matching is confounded; either drop the component or report under ablated score. No "X attenuates under matching" claim is defensible without this check.

7. **Length residualization.** Composites with length-correlated components (breadth counts, section shares, density metrics) must report a length-residualized version as primary: `composite_resid = composite_score − (b0 + b1 × log(description_cleaned_length))` via OLS on the relevant subset. Raw composite as sensitivity.

8. **Description text source (dim d).** Primary `description_core_llm` filtered to `llm_extraction_coverage = 'labeled'`; alternative raw `description`. Direction flips between the two → finding is boilerplate-driven and flagged.

9. **Source restriction (dim e).** Primary arshkon-vs-scraped; alternative pooled 2024 (arshkon + asaniczka). Report both where sample sizes allow.

10. **Prevalence citation transparency.** Every prevalence, density, share, or effect-size number must be cited with (a) exact pattern/column definition, (b) subset filter (sample, cap, aggregator exclusion, LLM-coverage restriction), (c) denominator ("of all" vs "of known-seniority" vs "of LLM-labeled"). Cross-task citations combining numbers from different patterns or subsets into one cell are prohibited.

## Alternative framings to evaluate explicitly at each gate

The same data can support multiple narratives. At each gate, the orchestrator will explicitly weigh at least two alternatives before committing to one.

| Dimension | Framing A (initial design) | Framing B (alternative) | What would favor B |
|---|---|---|---|
| Junior direction | Junior scope inflation (more required of entry roles) | Junior scope deflation / hiring-bar lowering | T33 showing requirements-section contraction correlates with lower YOE / credential asks |
| Structure axis | Seniority-driven restructuring | Technology/domain-driven restructuring | T09 NMI(clusters, domain) >> NMI(clusters, seniority); T28 between-domain component dominates |
| Firm dynamics | Within-firm restructuring | Market recomposition (different firms) | T06 returning-cohort minority share; T16 between-company component dominates; T37 retention ratios < 0.5 |
| Mechanism | Real demand change | Platform/tooling evolution (recruiter LLMs) | T29 low-authorship-score subset shows muted Wave 2 effects |
| Scope | SWE-specific restructuring | General AI-exposed occupation shift | T18 DiD near zero; T32 cross-occupation gap direction universal |

The Gate 2 and Gate 3 memos will carry a "Narrative evaluation" section that judges each of these five axes against the accumulated evidence.

## Dispatch plan

Wave 1 launches four agents in parallel:

- **Agent A** — T01 (data profile + coverage heatmap) + T02 (asaniczka `associate` as junior proxy).
- **Agent B** — T03 (`seniority_final` label audit) → **T30 (seniority definition ablation panel — canonical artifact consumed by every Wave 2+ task)** → T04 (SWE classification audit).
- **Agent C** — T05 (cross-dataset comparability) + T06 (company concentration + entry-specialist list + returning-companies cohort).
- **Agent D** — T07 (BLS/JOLTS benchmarks + power analysis with per-(comparison × T30 definition) MDE table).

All four agents get: core preamble, dispatch block, explicit 31GB RAM note, DuckDB/pyarrow-only instruction, task specs. Launched at max model capability.

After Wave 1 returns, the orchestrator writes Gate 1 (feasibility + T30 primary recommendation) and dispatches Agent Prep for Wave 1.5 shared preprocessing.

## Risks the orchestrator is watching

- **OOM on embedding step in Wave 1.5 Prep.** Mitigation: batches of 256, save partial state, local fallback in Wave 2 agents.
- **T30 panel disagreement being papered over by agents.** Mitigation: explicit dispatch instruction that YOE-vs-label disagreement IS a finding; V1 audits.
- **Keyword patterns cited with tautological precision.** Mitigation: pre-committed semantic sampling protocol; V1 rebuilds any tautological pattern.
- **Cross-task citation of prevalence numbers from different patterns/subsets as if comparable.** Mitigation: pre-committed transparency rule; V1/V2 audit flags any compound cite.
- **Agent N (Wave 4) importing orchestrator narrative rather than building from evidence.** Mitigation: Gate 3 memo must produce a ranked findings list BEFORE Agent N dispatches; Agent N reads the rank, does not re-decide it.

## What a good outcome looks like

By the end of Wave 4:

- A 1-2 sentence abstract draft grounded in strongest headline finding the data supports.
- A ranked findings list with evidence × novelty × narrative-value judgments.
- Explicit statement of which RQ1-RQ4 claims held up, which were reframed, which were demoted.
- T37 sampling-frame retention table and T38 hiring-selectivity correlation as the robustness appendix backbone.
- Methods recommendations: what worked, what didn't, what the analysis phase should do differently.
- Deferred-hypothesis inventory (T24): hypotheses the exploration could not test, with priority.
- A paper positioning recommendation (dataset/methods, substantive empirical, or mixed-methods) derived from the evidence, not pre-committed.

---

*Orchestrator note: this memo is a time-capsule. It will be read against Gate 1/2/3 memos to check whether the orchestrator's priors were updated honestly by the evidence or defended against it.*
