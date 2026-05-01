# Gate 0 Pre-Exploration Memo

## Starting Position

This exploration begins with a plausible but unproven narrative: AI coding tools may have narrowed the junior SWE ladder, increased junior role scope, shifted senior roles toward orchestration and review, and opened a gap between employer requirements and worker usage. I am treating that as a launch hypothesis, not as the paper's destination. The first wave should tell us what the data can support before we let any substantive story harden.

The empirical object is employer-side posting language and posting composition across heterogeneous instruments: 2024 LinkedIn snapshots from two Kaggle sources and a 2026 daily scrape. That makes this a strong design for studying labor-demand signaling, but a weaker design for causal claims about employment, headcount, or actual work unless later evidence and interviews support that interpretation.

## Initial Confidence

Most confidence going into Wave 1:

- The project can credibly describe changes in SWE postings if source comparability, company concentration, and within-2024 calibration are handled as first-class evidence rather than appendix checks.
- The seniority panel can turn a fragile "entry-level" claim into a measurable construct, provided the panel exposes rather than hides disagreements between label-based and YOE-based definitions.
- Technology and text-discovery analyses are likely to surface real market structure beyond the original RQ1-RQ4 framing, especially if clusters organize by domain or employer type instead of seniority.

Least confidence going into Wave 1:

- A narrow `entry`-only junior story may be underpowered or instrument-dependent, especially because asaniczka has no native entry labels.
- Raw text comparisons may conflate real requirement change with description length growth, markdown/formatting differences, boilerplate, and possible recruiter use of LLM-written job descriptions.
- Senior role redefinition could be hard to distinguish from generic title inflation or template language unless Wave 2 and Wave 3 validate management, mentorship, orchestration, and strategic-scope patterns semantically.
- Employer-requirement / worker-usage divergence may depend more on external benchmark quality than on our posting data.

## What Would Change Direction

Wave 1 could move the project in several different directions:

- If T30 shows directional disagreement across J1-J4 or power is poor for all junior definitions, the paper should not lead with junior share or junior scope inflation. It should reframe toward market structure, domain recomposition, employer concentration, or text/requirement evolution.
- If T05 finds within-2024 source differences as large as 2024-to-2026 differences, cross-period claims must be framed as instrument-sensitive and should emphasize robust within-source or within-company patterns.
- If T06 finds that a small set of employers or intermediaries dominates entry-level rows, then the junior market story becomes about specialized entry pipelines and company composition, not broad market-wide restructuring.
- If T04 finds weak SWE classification quality around ML/data/DevOps boundary roles, then cross-occupation and technology-domain findings become more important than a clean SWE-only framing.
- If T07 shows that many planned seniority-stratified comparisons are underpowered, Wave 2 should prioritize all-SWE, company, domain, and text-structure analyses over thin level-specific claims.

## Pre-committed Ablation Dimensions

For Wave 2+ agents, the following sensitivity dimensions are non-negotiable and should be written into each prompt where applicable:

- **T30 seniority panel:** Every junior claim reports J1-J4; every senior claim reports S1-S4. Only unanimous or 3-of-4 directional agreement may support a lead claim. Split or contradictory panels are substantive findings to investigate.
- **Aggregator exclusion:** Every core finding must be reported with aggregators included and excluded, or explain why the dimension is not applicable.
- **Company capping:** Corpus aggregates, term frequencies, topic models, co-occurrence networks, and centroid-style metrics require a capped or otherwise de-concentrated specification; uncapped results are not sufficient for interpretation.
- **Within-2024 calibration:** Cross-period effects must be compared against arshkon-vs-asaniczka variability. Effects smaller than the within-2024 noise floor are not lead evidence.
- **Semantic keyword precision:** Any keyword pattern used for prevalence, density, or effect-size claims must be validated on a stratified semantic sample. Tautological regex self-checks do not count. Sub-patterns below 80% precision should be dropped or reported as exploratory.
- **Description text source:** Text-dependent analyses must use `description_core_llm` where `llm_extraction_coverage = 'labeled'` as the primary text source. Raw `description` is a sensitivity or a recall-oriented fallback only for boilerplate-insensitive binary presence checks.
- **SWE classification tier:** Key findings should test whether they survive excluding the elevated-risk `title_lookup_llm` tier.
- **LLM coverage transparency:** Analyses using LLM columns must report labeled counts, eligible counts, and whether estimates are core-frame or supplemental-cache based.
- **Composite-score correlation checks:** Any matched-delta analysis using a composite score must report component-by-outcome correlations before interpreting attenuation or sign changes. If any component has `r > 0.3` with the outcome, the matching interpretation is confounded unless ablated.

## Narrative Evaluation Before Evidence

The original RQ1-RQ4 narrative is plausible but fragile. RQ1 and RQ2 depend heavily on seniority measurement, source comparability, and whether requirement growth survives length and boilerplate controls. RQ3 depends on external benchmark validity. RQ4 remains valuable either way because interviews can adjudicate whether observed posting changes reflect real work, anticipatory hiring narratives, or template inflation.

Two alternative framings are already live:

- **Market recomposition framing:** Apparent restructuring may reflect which companies, domains, and intermediaries are posting in 2026 rather than a broad within-firm change in SWE work.
- **Posting-instrument evolution framing:** The strongest observed changes may be in how job descriptions are written, structured, and templated, not necessarily in the underlying hiring bar.

I do not prefer either alternative yet. Wave 1 should establish whether the dataset can separate them from the original seniority-restructuring frame.

## Direction for Wave 1

Wave 1 should prioritize feasibility and measurement discipline over substantive interpretation. The critical outputs are: seniority panel recommendation, feasibility table by seniority definition, source comparability assessment, company concentration profile, SWE classification audit, and explicit constraints for Wave 2. If those outputs are weak or contradictory, the right response is to reframe the exploration rather than force the initial RQs.

