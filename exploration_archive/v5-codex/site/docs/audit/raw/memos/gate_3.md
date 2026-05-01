# Gate 3 Research Memo

## What we learned

Wave 3 confirms the Wave 2 pivot. The strongest story is no longer junior disappearance, senior management decline, or a clean employer-worker divergence. It is **differential densification under template drift**: SWE postings became more AI/LLM-centered and more multi-constraint, especially within returning companies, while the broader technical posting surface also became longer, more structured, and more scope-heavy. The evidence supports posting-language change and template drift, not AI-authorship attribution.

T16 shows company-level heterogeneity rather than one market-wide shift. Among 237 returning companies with at least three SWE postings in both arshkon and scraped LinkedIn, AI/scope/length movement is mostly within-company. On the primary overlap panel, AI/LLM share rises by 14.0 pp, scope share by 18.8 pp, cleaned-text length by 802 chars, requirement breadth by 1.39 categories, and stack depth by 0.47. T16 also proposed four company trajectories, but V2 did not reproduce the exact cluster split, so the typology should be treated as provisional until the clustering specification is frozen.

T18 is the key cross-occupation check. AI-tool language is the cleanest SWE-specific signal: in the cleaned-text frame, AI-tool share rises from 1.98% to 20.38% for SWE, from 2.45% to 18.14% for adjacent roles, and only from 1.25% to 1.41% for controls. But description length and scope also rise in controls, so the text-expansion story is not SWE-only. V2 did not finish the independent rederivation, so the exact percentages stay provisional even though the directional story is consistent.

T21 revises the senior-role story. Strict management does not decline: it is flat to slightly up. Strict orchestration rises sharply in senior postings, and senior clusters shift from a 97.6% people-manager cluster in 2024 to 78.5% in 2026, with tech-orchestrator rising to 15.3% and strategist to 6.3%. The right claim is AI/tool-orchestration growth around a stable management base, not management substitution. V2 did not independently finish the aggregate rederivation, so the exact cluster proportions remain provisional.

T22 and T23 discipline the AI interpretation. AI-posting language is more aspirational than non-AI language: hedge/firm ratio is 0.73 vs 0.52 in the section-filtered LLM core and 1.00 vs 0.80 in raw full text. But the employer-worker divergence claim is not robust: posting-side AI-tool rates are 30.3% in the primary core and 40.7% in raw sensitivity, and benchmark comparisons flip depending on whether the worker benchmark is 32.4%, 51%, 84%, or 99%. Both claims remain calibration-oriented until the independent recomputations finish.

## What surprised us

The cross-occupation result is not a simple SWE-specific shock. SWE shows the densest AI and requirement bundle, but adjacent roles move in the same direction on AI and controls also get longer and more scope-heavy. That makes platform/template evolution part of the core story, not a footnote.

The senior result contradicted the initial senior-archetype hypothesis. We expected management language to decline. Instead, strict management is stable or slightly higher, while AI/tool orchestration rises around it.

The ghost analysis did not find literal fake-job evidence as the main phenomenon. The signal is overloaded, aspirational, repeated templates, including direct employers. That is more subtle and more useful than a simple "ghost jobs increased" claim.

The divergence story weakened. AI language is clearly higher in postings, but the outpacing-worker-usage claim depends too much on benchmark choice and text source to headline.

## Evidence assessment

**AI/LLM technology and domain restructuring: strong.** Evidence accumulates across T09, T14, T16, T18, T19, T21, and T23. The exact rates vary by text source and indicator, but the direction survives company capping, aggregator exclusion, within-company checks, and cross-occupation comparison. The strongest version is narrow AI/LLM tooling and AI/tool orchestration, not broad AI keywords.

**Credential stacking and scope growth: strong with template caveat.** T11 and T16 support increased requirement breadth and stack depth, and T22 shows that some of this is aspirational/template-like. The interpretation should be "multi-constraint posting language expanded," not "hard requirements all increased."

**Posting-template evolution: strong.** T13 showed the structural shift; T18 showed length and scope also move outside SWE; T22 showed template saturation and aspiration. This is both a validity threat and a contribution.

**Company strategy recomposition: moderate-to-strong for decomposition, provisional for clusters.** T16 has a usable 237-company panel and V2 verified the overlap counts plus within-company AI/scope/length decomposition. V2 did not reproduce the exact four-cluster split, so use company clusters as exploratory labels only.

**Seniority and junior findings: moderate validity finding, weak substantive trend.** T20 confirms the ladder is measurable but uneven. The entry boundary is weakest and explicit entry remains conservative. T16/T17 show entry conclusions change with pooled 2024 and YOE proxy. Do not headline junior collapse.

**Senior role evolution: moderate.** T21 has a large senior frame and validated strict patterns. The AI/tool-orchestration shift is credible; the management-decline claim is contradicted.

**Employer-worker divergence: weak as a headline.** T23 should be a calibration section. Posting AI is real, but outpacing usage is benchmark-sensitive.

## Narrative evaluation

**RQ1: employer-side restructuring.** Reframed, not confirmed as originally written. The junior-share component is weakened; company strategy recomposition and domain/requirement densification are stronger. RQ1 should ask how employer posting strategies changed, not whether junior jobs simply declined.

**RQ2: task and requirement migration.** Strengthened if rewritten as requirement stacking, AI/tool-domain expansion, and aspirational/template bundling. The old "requirements moved downward" framing is too narrow.

**RQ3: employer-requirement / worker-usage divergence.** Weakened. The answer is not robust divergence; it is benchmark-sensitive elevated AI signaling. This should move from lead claim to robustness/calibration.

**RQ4: mechanisms.** Strengthened. The mechanisms now need to explain real AI/tool language, credential stacking, template drift, aspiration, and whether these are screening-relevant.

Alternative framing 1: **SWE-specific AI restructuring.** Supported for AI-tool language and AI/tool orchestration, but too narrow because adjacent roles also move and template drift is field-wide.

Alternative framing 2: **platform/template evolution.** Strongly supported for length, sectioning, and scope; insufficient alone because AI/LLM signals survive multiple checks and have within-company structure.

Preferred framing: **differential densification under template drift.** SWE sits at the densest end of a broader technical-posting template shift, with AI/LLM and scope/stacking expanding most strongly in SWE and returning companies.

## Emerging narrative

The paper's core argument should be:

Between 2024 and 2026, SWE postings reorganized around AI/LLM tooling and denser requirement bundles, but this happened through changing posting templates and heterogeneous company strategies rather than a uniform junior-role collapse. The strongest restructuring signal is within-company growth in AI/scope/stacking and an emergent AI/tool-orchestrator senior cluster, while seniority and employer-worker divergence claims require heavy qualification.

## Research question evolution

Recommended RQs for synthesis:

`RQ1: How did SWE employer posting strategies change between 2024 and 2026 across company trajectories, technology-domain archetypes, and posting-template structure?`

`RQ2: Which requirement categories became more densely bundled in SWE postings, and how much of that expansion reflects AI/LLM tooling, scope language, and aspirational/template phrasing?`

`RQ3: Which changes are SWE-specific versus part of a broader technical-posting template shift?`

`RQ4: How large is the posting-side AI signal relative to worker-usage benchmarks, and why is the divergence interpretation benchmark-sensitive?`

`RQ5: How do engineers and hiring-side actors explain whether AI/scope/stacking language is screened, aspirational, or template-driven?`

This promotes cross-occupation specificity into the quantitative core and demotes the unqualified worker-usage divergence claim.

## Ranked findings

1. **AI/LLM and technology-domain restructuring.** Strong evidence, high narrative value. Supported by T09, T14, T16, T18, T19, T21, and T23. Lead candidate.
2. **Credential stacking and scope densification.** Strong evidence, core narrative value, but must carry the aspiration/template caveat from T22.
3. **Posting-template and sectioning evolution.** Strong evidence and essential validity contribution. This is the methods backbone.
4. **Company-level recomposition.** Moderate-to-strong for within-company decomposition, high novelty. Useful as the actor-level mechanism tying aggregate changes to returning firms. The exact cluster typology is provisional.
5. **Senior AI/tool-orchestrator growth without management decline.** Moderate evidence, good corrective finding. Supports senior-role section but should not overclaim replacement.
6. **Junior measurement asymmetry.** Important validity finding, but weak as a substantive labor-market trend.
7. **Employer-worker divergence.** Interesting but weak as a lead because benchmark-sensitive.

## Gaps and weaknesses

LLM cleaned-text coverage remains a binding constraint, especially in scraped 2026 and outside SWE.

The ghost/aspiration frame is heuristic. It ranks postings and companies, but it does not prove what employers screen for.

Worker-usage benchmarks are not directly comparable to posting text. The paper needs to state that RQ3 compares different objects.

The company typology is useful but heuristic. It needs robustness checks before being treated as a formal classification.

The temporal design remains windowed, not continuous. No abstract should imply a smooth trend or causal event break.

## Direction for next wave

Wave 4 synthesis should lead with the revised narrative above and not restart from RQ1-RQ4 as originally written.

The synthesis agent should build the top figures/tables around:

1. AI/LLM technology-domain growth and co-occurrence network.
2. Requirement breadth / stack depth / scope densification, with T22 aspiration overlay. T22 numbers should be carried as plausible but not independently verified by V2.
3. Posting-template shift: sectioning, cleaned/raw length, and core-section coverage.
4. Company within-company decomposition. Use the strategy typology only as an exploratory support layer unless reverified.
5. Cross-occupation comparison showing SWE-specific AI growth but field-wide template drift. This needs analysis-phase verification because V2 did not complete the independent check.
6. Senior AI/tool-orchestrator cluster growth, with management-stability caveat. This also needs analysis-phase verification beyond the broad-management precision check.

Robustness checks needed in the analysis phase:

- Re-derive narrow AI/LLM, scope, and stack-depth indicators from a single validated dictionary.
- Run all core claims under arshkon-only vs pooled 2024, aggregator exclusion, company cap/weighting, and LLM-text vs raw sensitivity.
- Re-run domain-stratified stacking/convergence now that T09 archetypes exist.
- Spec-lock and re-run the T16 company clustering before treating cluster names or sizes as results.
- Independently re-run T18, T21, T22, and T23 headline numbers; V2 did not complete those checks.
- Validate ghost/aspiration scores with manual review and interviews.
- Keep `seniority_final` and YOE-proxy junior definitions side by side in every junior table.

## Current paper positioning

The best paper is now an empirical restructuring paper with a strong measurement-validity backbone. It should not market itself as proving AI caused junior job loss or employer-worker divergence. It should claim that AI-era SWE postings became more AI/LLM-centered, more multi-constraint, and more standardized/template-shaped, with heterogeneous company strategies and aspirational signaling mediating the interpretation.
