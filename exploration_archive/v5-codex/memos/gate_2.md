# Gate 2 Research Memo

## What we learned

Wave 2 changes the paper's center of gravity. The strongest emerging story is not junior disappearance or seniority convergence. It is domain and technology restructuring layered onto a changing posting template: SWE postings became more AI/LLM-saturated, more scope-heavy, and more multi-constraint, while the natural structure of postings is domain-first rather than seniority-first.

The most important single result is T09's latent structure finding. NMF `k=15` produced the first useful archetype map, and the full-corpus labels align far more with tech domain than with seniority. T09 estimated tech-domain NMI at 0.123; V1's independent rebuild put it at 0.115 while preserving the ordering. Seniority NMI is about 0.003, period about 0.028, source about 0.030, text source about 0.013, and aggregator status about 0.007. The AI / LLM workflows archetype is 91.5% 2026.

The strongest content result is T11's credential-stacking finding. On the primary company-capped LLM-text subset, requirement breadth, credential stack depth, scope count, and scope density all clear within-2024 calibration. Stack depth is especially strong: the cross-period signal-to-noise ratio is about 22.3-22.5. AI should be led with the narrower AI/LLM technology signal from T14 rather than a broad AI bucket. Tech count moves only modestly and should not be the headline.

The strongest validity result is T13's posting-template shift. Cleaned text length rises from 1,969 chars in arshkon to 2,644 in scraped, above the within-2024 gap. More importantly, docs with detected core sections rise from 1.6% in arshkon and 4.5% in asaniczka to about 61-62% in scraped. This means length and scope changes are partly about document form, not only employer demand.

The technology ecosystem result is robust. T14 finds AI-tool / LLM mention rates rise from 2.05% in arshkon to 17.19% in scraped, and aggregator exclusion barely changes the estimate. The network interpretation is also consistent: a stable cloud/frontend backbone gains a denser AI/LLM layer.

## What surprised us

The latent market structure is not seniority-first. This directly challenges the initial framing, which expected junior and senior roles to be the central axis. Seniority matters for measurement and interpretation, but it is not the main organizing structure of the posting space.

The title layer did not fragment. T10 finds unique titles per 1,000 postings are basically flat, while broad AI-related title share rises from 12.4% to 22.7%. That points to semantic re-anchoring, not a wholesale explosion of title categories.

The text evolution task is messier than expected. T12's cleaned full-text ranking is dominated by historical stack and credential artifacts, and the section-filtered comparison is too imbalanced to headline: 72 arshkon docs versus 3,359 scraped docs. The AI/workflow signal is real, but it emerges most clearly through raw sensitivity and topic modeling, not a clean before/after word list.

The junior-senior convergence story fails. T15 shows embedding-based junior-senior similarity changes are inside the within-2024 gap, while TF-IDF points to divergence rather than convergence. The semantic space tightens, but junior and senior roles do not collapse into each other.

## Evidence assessment

**Domain-first archetype structure: moderate-to-strong.** T09 uses a balanced LLM-text sample with company caps and a full-corpus NMF label artifact. The relative NMI comparison is clear, but the absolute NMI values are modest and V1 reproduced the tech-domain alignment at 0.115 rather than 0.123. BERTopic corroborates AI/LLM and mobile at a coarse level but collapses too much structure, so downstream domain claims should rely on NMF with method caveats.

**Credential stacking: strong.** T11 reports large calibration ratios for `credential_stack_depth` (about 22.3-22.5), `scope_density` (84.7), `scope_count` (14.0), and `requirement_breadth` (5.4). These survive company capping and aggregator exclusion. Raw text changes magnitude but not direction. Tech count, broad AI, and management language are weaker and should not lead. V1 confirms narrow AI/LLM indicators are clean, while broad management falls below the 80% precision bar.

**Posting-template shift: strong.** T13's sectioning change is large and stable across both scraped months. This is not a small parser artifact. It is a first-order measurement condition for all text claims. The remaining caveat is that the section parser misses unmarked historical structure, so "unclassified" should be interpreted as weakly structured text, not pure junk.

**AI/LLM technology ecosystem: strong.** T14's AI/LLM mention growth survives aggregator exclusion and within-2024 calibration. Specific changes such as `llm`, `generative_ai`, `agent`, and `claude` are well above the historical source gap. This is still posting language, not actual tool use.

**Open-ended text evolution: moderate but messy.** T12 supports a 2026 AI/workflow vocabulary cluster, including a BERTopic topic with 1,432 docs and 83.2% 2026 share. But the main cleaned log-odds surface is noisy and section-filtered evidence is imbalanced. Use T12 as triangulation, not a standalone lead.

**Title evolution: moderate.** Broad AI title share is robust, but title fragmentation is weak. Title-content similarity is coverage-limited. T10 should support a naming-layer section rather than define the paper.

**Semantic convergence: moderate negative evidence.** T15 is useful because it rules out a tempting story. Embeddings and TF-IDF disagree on magnitude, but both reject a clean convergence claim. The result should demote rather than disappear: it is evidence against an over-broad seniority-boundary narrative.

## Narrative evaluation

**RQ1, employer-side restructuring: needs substantial reframing.** The initial junior share/volume claim is no longer the best lead. The evidence points to a split between explicit entry labels and YOE-proxy junior-like rows, plus company composition and domain recomposition. RQ1 should ask how the observable junior rung changes under multiple instruments, not whether junior roles simply declined.

**RQ2, task and requirement migration: strengthened but reframed.** The stronger concept is credential stacking and scope expansion, not simple downward migration from senior to junior. AI and scope are added to requirement bundles; tech count alone is too weak. The junior-specific version must use both explicit-entry and YOE-proxy slices.

**RQ3, employer-requirement / worker-usage divergence: still promising but not established.** T14 and T12 provide a strong employer-side AI signal. They do not yet prove divergence from worker usage, and they do not distinguish real requirement from aspirational mention. This should be a Wave 3 focus, especially T22/T23.

**RQ4, mechanisms: more important.** Template evolution, AI vocabulary, and credential stacking create a strong reason for interviews. The mechanism question should ask whether these posting changes reflect actual work, recruiter templates, anticipatory AI signaling, or overscreening.

The best alternative framing is now **technology/domain recomposition plus credential stacking**. A second plausible framing is **platform/template evolution**, where changed document structure creates apparent scope inflation. I prefer the hybrid framing: template evolution is real and must discipline the methods, but it does not fully explain the AI/LLM network, AI archetype, and credential-stack changes.

## Emerging narrative

The data are telling a more publishable and more defensible story than the launch hypothesis: from 2024 to 2026, SWE postings reorganized around domain and technology bundles, especially an AI/LLM workflow layer, while employers bundled more kinds of requirements into postings. At the same time, the posting surface itself changed, with far more explicit sectioning and longer structured text. The paper should frame this as employer-side demand signaling under document-template change, not direct labor-market causality.

Current one-sentence abstract if forced: Between 2024 and 2026, SWE postings became more AI/LLM-centered and multi-constraint, but the strongest restructuring signal is domain and requirement-stack recomposition rather than a clean disappearance or convergence of junior roles.

## Research question evolution

Replace the original RQ1 lead with:

`RQ1: How did SWE posting structure and composition change from 2024 to 2026 across explicit seniority labels, YOE requirements, company composition, and technology-domain archetypes?`

Revise RQ2 to:

`RQ2: Which requirement categories and technology-domain bundles expanded in SWE postings, and do these changes survive cleaned-text, company-capping, and within-2024 calibration checks?`

Keep RQ3, but sharpen it:

`RQ3: Do validated employer-side AI/LLM requirement signals exceed worker-side AI usage benchmarks, and are those signals real requirements, aspirational/template language, or domain recomposition?`

Keep RQ4, with mechanisms focused on interpretation:

`RQ4: How do engineers and hiring-side actors explain the gap between posting language, actual work, and screening practices in AI-era SWE hiring?`

Add a candidate bridging question:

`RQ5 candidate: Did the document form of SWE postings change enough to alter how labor-demand signals should be measured?`

This may become a methods contribution rather than a standalone RQ.

## Gaps and weaknesses

The biggest weakness is text-source coverage. The shared LLM-text frame is 26,219 of 59,972 SWE LinkedIn rows, with scraped especially thin. Any text-sensitive claim needs coverage tables and raw-text sensitivity.

The second weakness is section comparability. T13 shows the document form changed, while T12 shows section-filtered comparisons are imbalanced. This limits claims about "requirements" unless later tasks validate section-level signals carefully.

The third weakness is seniority. `seniority_final` remains useful as a strict label subset, but junior prevalence should not be estimated from it alone. YOE proxy and explicit labels must travel together.

The fourth weakness is keyword validity. AI/LLM signals are strong, but keyword/regex indicators still need precision validation before becoming headline numbers. Management language is especially noisy and should be deemphasized.

## Direction for next wave

Wave 3 should be modified to pursue the Wave 2 discovery instead of blindly extending the original RQs.

For T16/T17, prioritize within-company versus composition decomposition using T09 archetype labels. The key question is whether AI/LLM and credential-stack growth is within returning companies or driven by new 2026 firms. Geography should be secondary unless clean metro cells appear.

For T18/T19, cross-occupation analysis becomes decisive. If control occupations show the same template/AI/sectioning pattern, the SWE-specific restructuring story weakens. If SWE shows stronger AI/LLM and credential-stack changes than controls, the paper gains credibility.

For T20/T21, demote simple seniority convergence. Focus instead on whether senior-role language shifts within domain archetypes, and whether explicit-entry versus YOE-junior discrepancies vary by domain.

For T22/T23, make ghost/aspirational requirement checks central. The strongest validity question is whether AI/LLM and credential-stack signals are real screening requirements or posting/template inflation.

Add a Wave 3 follow-up if capacity allows: use `swe_archetype_labels.parquet` to rerun domain-stratified requirement stacking and semantic convergence, because T11/T15 could not use labels during parallel execution.

## Current paper positioning

The best paper after Wave 2 is an empirical restructuring paper with a methods-validity backbone. The lead is not "AI killed junior jobs." The lead is: AI-era SWE postings show domain and requirement-stack recomposition, but interpreting those shifts requires accounting for changed posting form, company concentration, and seniority measurement.

The next wave needs to deliver three things to make this publishable: SWE-specificity against controls, within-company versus composition decomposition for AI/LLM and stacking signals, and a ghost/aspirational validity check for whether the AI/stacking language is screening-relevant.
