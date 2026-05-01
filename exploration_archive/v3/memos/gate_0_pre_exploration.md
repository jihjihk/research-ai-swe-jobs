# Gate 0: Pre-Exploration Assessment

Date: 2026-04-05

## What I'm most confident about

1. **The data exists and spans the right period.** ~33K SWE rows across three snapshots (Jan 2024, Apr 2024, Mar 2026) covers the AI coding tool adoption window. The core comparison — 2024 LinkedIn postings vs 2026 LinkedIn postings — is viable.

2. **Something changed in SWE postings between 2024 and 2026.** The 56% description length growth alone confirms structural change. The question is what changed, what's real, and what's artifact.

3. **We have enough mid-senior SWE data for robust analysis.** Arshkon has ~3,003 mid-senior SWE; scraped has ~3,459. These cells are well-powered for most comparisons.

4. **The sensitivity framework is well-designed.** Seven dimensions covering the major confounds. The within-2024 calibration (dimension f) is the most important — it sets the noise floor for everything.

## What I'm least confident about

1. **Entry-level analysis may be underpowered or misleading.** The headline RQ1 claim is about junior scope inflation, but our entry-level sample is thin: 830 arshkon entry vs 574 scraped entry. More concerning: arshkon's 385 native entry-level labels are our only historical baseline (asaniczka has zero native entry labels). If Wave 1 reveals that the 830 arshkon entry count includes many imputed labels of uncertain quality, the entry-level story gets even thinner.

2. **The YOE paradox is a serious red flag for the scope inflation narrative.** Entry-level median YOE *decreased* from 3.0 (arshkon 2024) to 2.0 (scraped 2026). This directly contradicts the "junior scope inflation" hypothesis. Either: (a) the entry-level composition changed between sources, (b) the YOE extractor behaves differently on longer/formatted descriptions, or (c) scope inflation isn't happening in the way we hypothesized. Any of these is important.

3. **Low company overlap (~18-25%) makes aggregate trends suspect.** If different companies are posting in 2024 vs 2026, aggregate changes in seniority shares, requirements, etc. could be entirely driven by compositional shifts rather than within-company restructuring. T16's within-company decomposition is critical.

4. **Instrument differences may dominate real signal.** Kaggle text is HTML-stripped/unformatted; scraped text preserves markdown formatting. This structural difference affects description length, readability, and potentially term frequencies. The within-2024 calibration is our main defense, but if within-2024 variability is large, our signal-to-noise may be poor.

5. **71.8% seniority unknown rate.** Seniority-stratified analyses use <29% of rows. The observed seniority distribution among known rows may not represent the full population. SWE-specific coverage is better (arshkon 81%, scraped 88%), but asaniczka's 100% "coverage" is almost all mid-senior/associate — useless for entry-level baseline.

## What would change my assessment of the project's direction

### Toward a stronger paper
- **If within-2024 calibration shows low noise** (arshkon and asaniczka agree on key metrics), then 2024-to-2026 differences become more credible. This would strengthen the empirical restructuring paper framing.
- **If T09 clustering reveals a clear, interpretable structure** that maps onto a novel organizing principle (not just seniority), this could become the paper's headline contribution.
- **If the AI tool adoption pattern shows a clear gradient** (appearing in specific role types before others, or in specific geographies first), that's a strong empirical finding about diffusion.

### Toward a weaker/different paper
- **If within-2024 variability exceeds cross-period differences**, the empirical restructuring story collapses and this becomes primarily a dataset/methods paper.
- **If entry-level analysis is truly infeasible** (underpowered, unreliable labels), the "junior scope inflation" RQ should be demoted and the paper should lead with a different finding (perhaps senior archetype shift or technology ecosystem restructuring).
- **If aggregator contamination explains most of the entry-level changes**, the "restructuring" narrative becomes "aggregator composition shift" — less novel but honest.
- **If control occupations show the same patterns as SWE** (T18), the "AI-driven SWE-specific" framing weakens and this becomes a broader labor market trend story.

### Toward an entirely different framing
- **If the most interesting pattern is something we never hypothesized** — e.g., technology ecosystem fragmentation, geographic divergence, company-type polarization — the paper should pivot to tell that story instead.
- **If the "ghost requirements" finding is strong** (T22 reveals widespread aspirational AI requirements), the most novel contribution might be about employer signaling rather than real restructuring. That's actually a more interesting and defensible claim.

## Hypotheses entering Wave 1

Ranked by what I expect the data to show (to be checked against what it actually shows):

1. **Description length growth is real but mostly boilerplate/benefits expansion**, not requirements expansion. (T13 will test this.)
2. **AI tool mentions surged, but mostly in generic/aspirational form**, not specific tooling. (T12, T23)
3. **Entry-level share declined modestly**, but the effect is sensitive to seniority operationalization and aggregator inclusion. (T08)
4. **Senior roles shifted toward technical orchestration language**, but the effect may be smaller than hypothesized. (T21)
5. **Company composition shift explains a substantial portion of aggregate trends.** (T06, T16)
6. **The within-2024 calibration will reveal moderate instrument noise** — enough to worry about but not enough to invalidate cross-period comparisons entirely. (T05, T08)

## Wave 1 expectations

Wave 1 is diagnostic. I expect it to tell us:
- Which analyses are feasible (T07 power analysis)
- Which columns are usable (T01, T02)
- Whether seniority labels are reliable enough (T03)
- Whether the SWE classification is trustworthy (T04)
- How much instrument noise we're dealing with (T05)
- Whether company concentration biases results (T06)
- Whether our data represents the broader market (T07)

The most important output is the feasibility table from T07. If entry-level analysis is underpowered, I'll redirect Wave 2 effort away from entry-specific tasks and toward analyses with better statistical support.
