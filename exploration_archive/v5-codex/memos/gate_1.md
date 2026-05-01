# Gate 1 Research Memo

## What we learned

Wave 1 says the data can support a restructuring study, but not the simple version of the original junior-rung story. The primary LinkedIn frame is large: 1,186,281 rows and 59,972 SWE rows across arshkon, asaniczka, and scraped LinkedIn. Broad SWE, senior, pooled-2024, and company-panel analyses are statistically feasible. Entry-labeled analysis is the one fragile slice.

The central measurement finding is that `seniority_final` is high precision but low recall for junior roles. It finds a small explicit-entry subset, but misses many junior-like rows that appear in the YOE proxy. In T03, `seniority_final = entry` falls from 3.73% in arshkon to 2.18% in scraped 2026-04, while the label-independent `yoe_extracted <= 2` proxy rises from 14.98% to 16.97%. This is a direction-level disagreement, not just a magnitude difference.

Asaniczka cannot rescue junior power by treating `associate` as junior. T02 finds asaniczka `associate` is mixed and lower-mid leaning, with exact title overlap closer to arshkon `mid-senior` than arshkon `entry`. It should not be pooled into a junior bucket.

The strongest feasibility signals are outside the junior label itself. T05 finds description length growth above within-2024 calibration: mean description length rises from 3,306 chars in arshkon to 4,936 in scraped LinkedIn, with a cross-period effect about 2.6x the arshkon-vs-asaniczka gap. T06 finds AI-language growth and description-length growth have substantial within-company components in the overlap panel, while the YOE-based junior rise is mostly company composition.

## What surprised us

The native LinkedIn `entry` label is unstable. Arshkon native `entry` rows average 4.18 YOE and only 28.6% have YOE <= 2, while scraped LinkedIn native `entry` rows average 2.36 YOE and 81.0% have YOE <= 2. This makes native labels a diagnostic, not a credible cross-period truth source.

Company composition is already a first-order result. Top-50 employers account for 20.9% of arshkon, 30.6% of asaniczka, and 27.2% of scraped LinkedIn SWE rows. Most companies with at least five SWE postings have zero `seniority_final` entry roles. A market-level junior-share plot would hide that entry posting is a specialized employer behavior.

The data are geographically less worrying than expected. T07 reports BLS state-alignment correlations around 0.97 in available slices. This does not validate causal claims, but it reduces concern that the scrape frame is geographically bizarre.

## Evidence assessment

**Seniority measurement: moderate evidence, high importance.** T03 has large SWE samples and direct cross-tabs, but the entry-labeled cells are small. The finding that `seniority_final` is conservative is strong: final-entry rows have coherent low YOE, while unknown rates are high in scraped LinkedIn. The finding that junior share changed is weak under label-based measures because `seniority_final`, `seniority_native`, and YOE proxies disagree.

**Asaniczka associate comparability: strong evidence for non-use as junior.** T02 triangulates title overlap, title cues, YOE, and `seniority_final`. No single diagnostic is perfect, but they point the same way: asaniczka `associate` is not an entry proxy.

**SWE classification quality: moderate-to-strong evidence.** T04 finds 91.1% of SWE rows are regex-tier, no dual-flag violations, and uncertainty concentrated in the 7.3% `title_lookup_llm` tier and known boundary families. Downstream SWE findings are usable, with a required tier-exclusion sensitivity for boundary-sensitive results.

**Description length growth: strong evidence.** T05 shows a large sample, aggregator-insensitive effect, and cross-period change larger than within-2024 calibration. The interpretation remains open until T13 separates requirements sections from boilerplate, but the shift itself is credible.

**Company concentration and composition: strong evidence.** T06 uses direct concentration metrics and decomposition. The concentration facts survive aggregator exclusion. The decomposition is moderate rather than strong because AI mention and tech count are coarse raw-text proxies, but the within/between split is valuable for Wave 2 steering.

**Power and external benchmark: moderate evidence.** T07 gives a clear feasibility screen. Entry comparison is marginal at 175 vs 1,042 with MDE 0.229. All-SWE and senior comparisons are well-powered. BLS geography is reassuring context, but FRED/JOLTS was incomplete and should not be cited as finished.

## Narrative evaluation

**RQ1, employer-side restructuring: needs reframing.** The junior-share decline version is weakened. A strict explicit-entry label declines, but the YOE proxy rises, and the YOE-based rise appears mostly between-company composition. The stronger RQ1 framing is not "junior roles disappeared." It is: how did explicit seniority labeling, YOE floors, and employer composition jointly reshape the observable junior rung?

**RQ2, task and requirement migration: plausible but unproven.** Wave 1 does not test task migration directly. It shows text work is possible but coverage-limited, especially for scraped SWE cleaned text at 16.1% labeled. T13/T12/T11 should decide whether description length growth is requirements content or boilerplate.

**RQ3, employer-requirement / worker-usage divergence: still open.** Wave 1 only establishes that AI-language growth may be worth pursuing: T06's overlap-panel raw-text proxy rises from 13.8% to 53.1%, with about 71% of the change within-company. That is promising but not yet a validated AI-requirement measure.

**RQ4, mechanisms: strengthened as an interpretive need.** If posting shifts mix real work, relabeling, composition, and template inflation, interviews become more important, not less. The quantitative data alone may not distinguish aspirational requirements from actual hiring bars.

Two alternative framings now deserve equal consideration. First, a market recomposition framing: apparent junior change is about which companies appear in 2026, not broad within-firm junior elimination. Second, a posting-language evolution framing: employers and platforms may have changed how postings are written, with longer descriptions and more AI language, while seniority labels became less comparable. At Gate 1, I prefer a hybrid version of these alternatives over the original junior-decline narrative.

## Emerging narrative

The data are telling a more subtle story than the launch hypothesis. The most credible early narrative is: SWE postings changed visibly in text and AI-language content between 2024 and 2026, but the junior-rung story is instrument-sensitive and company-composition-driven. The paper should lead with restructuring in employer demand signals only after separating content change from platform/source artifacts and composition.

Right now, the strongest candidate lead is not junior share. It is the combination of measurable posting-language expansion, AI-language growth with a within-company component, and evidence that junior metrics depend sharply on whether "junior" means explicit label or low YOE.

## Research question evolution

Proposed RQ1 revision: How did employer-side SWE demand signals change from 2024 to 2026 across explicit seniority labels, YOE requirements, and company composition?

Proposed RQ2 revision: Which requirement and technology bundles expanded in SWE postings, and do these changes survive cleaned-text coverage limits, company concentration controls, and within-2024 calibration?

Keep RQ3 provisional: Do validated AI-related employer requirements outpace worker-usage benchmarks, and is the divergence within-company or driven by market recomposition?

Keep RQ4, with a sharper mechanism focus: How do practitioners and hiring-side actors distinguish real workflow change from relabeling, template inflation, and anticipatory signaling?

## Gaps and weaknesses

The biggest gap is cleaned-text coverage for scraped SWE. Only 16.1% of scraped SWE rows have labeled `description_core_llm`, so Wave 2 text claims must report coverage and avoid silently mixing raw and cleaned text.

The second gap is junior measurement. Entry-labeled cells are small and conservative; YOE proxies are broader but depend on parsed YOE availability. Any junior conclusion that does not show both will be fragile.

The third gap is benchmark incompleteness. T07 did not complete FRED/JOLTS, so hiring-cycle context remains unfinished.

The fourth gap is artifact separation. Description length growth is real as a measurement pattern, but we do not yet know whether it lives in requirements, benefits/legal boilerplate, formatting, or recruiter templates.

## Direction for next wave

Before Wave 2, Agent Prep should build shared artifacts but mark `text_source` clearly. Wave 2 agents must treat `text_source = 'llm'` as the primary text-sensitive subset and raw text as a recall-oriented or binary-presence supplement only.

Wave 2 should prioritize:

1. T13/T12 separation of requirements content from boilerplate and formatting.
2. T09/T15 method agreement on natural structure, with company caps by default.
3. T11/T14 requirement and technology expansion, company-capped and calibrated against within-2024 differences.
4. T08 distribution profiling that treats junior as dual-measured: explicit `seniority_final` and YOE proxy.
5. Sensitivity to excluding `title_lookup_llm` for SWE boundary-sensitive findings.

Wave 2 should deprioritize any claim that requires asaniczka `associate` as junior or treats `seniority_native` as pooled truth.

## Current paper positioning

If we stopped here, the best paper would be a dataset and measurement paper with early substantive evidence of posting-language change. To become a stronger empirical restructuring paper, Wave 2 needs to show that requirement or AI-language changes survive boilerplate controls, company capping, and within-2024 calibration. The paper should not currently lead with "junior roles declined"; it should lead with validated restructuring signals once Wave 2 identifies which ones are real.

