# Gate 0: Pre-Exploration Assessment

**Date:** 2026-04-09

## Data state snapshot

| Source | Platform | Total (Eng, ok) | SWE | SWE-adj | Control |
|---|---|---|---|---|---|
| kaggle_arshkon | LinkedIn | 117,645 | 5,019 | 2,475 | 8,445 |
| kaggle_asaniczka | LinkedIn | 1,012,193 | 23,213 | 11,068 | 112,080 |
| scraped | LinkedIn | 214,603 | 35,062 | 9,823 | 31,686 |
| scraped | Indeed | 110,726 | 9,100 | 2,545 | 20,616 |

Scraped LinkedIn SWE has grown to ~35K (up from ~24K in schema doc). This is good news for power.

## What I'm most confident about

1. **We have enough SWE volume for aggregate comparisons.** 5K arshkon + 35K scraped gives good power for overall SWE trends. Mid-senior comparisons are well-powered (arshkon 2,924 + scraped 26,805 via seniority_final).

2. **The cross-occupation design is strong.** With 8.4K arshkon control + 31.7K scraped control, plus SWE-adjacent groups, the DiD-style comparisons (T18) should have good statistical power. This is the validity backbone of the paper.

3. **Company and geographic data are sufficient for structural analysis.** 35K scraped SWE across 26 metros gives enough for metro-level and company-level patterns.

## What I'm most concerned about

1. **LLM coverage is severely limited for scraped data.** Only ~7.3K of 35K scraped SWE rows have LLM-extracted text (21%). LLM classification covers ~7.3K (labeled + rule_sufficient). This means most text analysis on scraped data falls back to `description_core` (44% accuracy boilerplate removal). This is the single biggest data quality concern.

2. **seniority_llm is extremely sparse.** Even among LLM-classified rows, ~85% are "unknown" because the classifier uses explicit signals only. Entry counts via seniority_llm: arshkon 84, asaniczka 39, scraped 180. These are far too thin for most seniority-stratified analyses. We'll rely heavily on seniority_final (which is much more populated but methodologically mixed) with seniority_llm as a sensitivity check rather than primary variable.

3. **Entry-level historical baseline remains the binding constraint.** Via seniority_final: arshkon has 848 entry SWE, scraped has 4,656. This is workable. But asaniczka's 129 entry-level rows are imputation artifacts. The direction of the entry-share trend hinges entirely on which seniority column and which 2024 source we use.

4. **The 2-year gap is both a strength and a weakness.** 2024 to 2026 gives us a large potential effect size to detect, but we cannot distinguish gradual from abrupt change, and many things change in two years besides AI tools.

## What would change my assessment

- **If seniority_final proves unreliable** (high disagreement with seniority_native, or if rule-based imputation systematically biases trends), the entry-level analysis becomes fragile. T02-T03 will tell us.
- **If within-2024 variability is large** (arshkon vs asaniczka differ more than expected on key metrics), then cross-period comparisons need much larger effects to be credible. T05 will tell us.
- **If the SWE classifier has high FPR** in specific tiers, the sample could be contaminated. T04 will tell us.
- **If the scraped data's limited LLM text coverage creates systematic bias** (e.g., only certain types of postings got LLM processing), all text-dependent analyses on scraped data are suspect.

## Initial assessment of RQ1-RQ4

- **RQ1 (restructuring):** Most testable. Entry share via seniority_final is well-powered. Scope inflation is measurable via description length, tech count, requirements complexity. But "junior scope inflation" specifically requires good entry-level samples, which are thin via seniority_llm.
- **RQ2 (task migration):** Requires clean text and good seniority labels simultaneously. The LLM text coverage gap for scraped data is the binding constraint.
- **RQ3 (divergence):** Depends on external benchmarks we haven't accessed yet. T07/T23 will determine feasibility.
- **RQ4 (mechanisms):** Qualitative, not constrained by this data. The exploration informs interview design.

## Pre-exploration hypotheses about what we'll find

- The scraped data is substantially larger and more recent. I expect description length growth, AI term proliferation, and possibly genuine seniority composition shift. But I also expect significant instrument-related differences that need careful calibration.
- I suspect the "junior scope inflation" narrative will be complicated by the entry-level measurement problem. The story may be more about seniority relabeling or domain composition shift than simple scope creep within entry-level roles.
- The strongest findings will likely come from aggregate text evolution (T12-T13) and technology ecosystem changes (T14), where we have the most statistical power and least measurement noise.

## Strategy for Wave 1

Wave 1 must answer: what analyses are actually feasible? The feasibility table from T07 and the seniority comparability assessment from T02-T03 are the most critical outputs. Everything downstream depends on knowing our constraints honestly.
