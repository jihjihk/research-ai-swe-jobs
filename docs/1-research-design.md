# Research Design

This is the **canonical project design doc**. It defines the paper's claims, research questions, constructs, and empirical strategy. The intended reading order for active research docs is:

1. `docs/1-research-design.md`
2. `docs/2-interview-design-mechanisms.md`
3. `docs/3-literature-review.md`
4. `docs/4-literature-sources.md`
5. `docs/5-publication-targets-2026-2027.md`

## Core Claim

The strongest publishable claim this project can make is about **employer-side restructuring**, not direct employment effects. From 2023 to 2026, software engineering job postings appear to have changed in ways that:

- narrow the junior rung by reducing junior share and inflating junior scope
- redefine senior roles away from classic people-management language and toward AI-enabled orchestration, review, and systems responsibility
- move faster than observed workplace AI usage, consistent with **anticipatory restructuring**

This design treats break detection and causal timing as supporting evidence, not the sole backbone of the paper.

## Paper Scope

This project is:

- a longitudinal study of SWE posting content and posting composition
- a dataset paper plus a substantive empirical paper
- a mixed-methods account of how employers, workers, and hiring-side actors explain the changes

This project is not:

- a direct estimate of AI's effect on employment levels or wages
- a replication of Brynjolfsson et al. using payroll data
- a clean causal estimate that one specific model release changed the labor market on a single date

## Available Data Assets

### Data we have now

- Daily-scraped SWE postings across 26 US metros from LinkedIn, Indeed, and YC
- Historical LinkedIn benchmark data from Kaggle / Hugging Face for 2023-2024
- Public Anthropic Economic Index usage data (Handa et al. 2025; Appel et al. 2026) and the `observed exposure` measure from Massenkoff and McCrory (2026), which combines O\*NET tasks, Eloundou et al. feasibility scores, and Claude usage
- Public Revelio aggregate labor statistics
- Qualitative interviews we can field ourselves

### Data we do not currently have

- Lightcast parsed-skill microdata
- Revelio posting-level or worker-level microdata
- Payroll microdata comparable to ADP

The design below assumes only the data we actually control.

## Research Questions

### RQ1: Employer-side restructuring

How did employer-side SWE requirements restructure across seniority levels from 2023 to 2026? Note that our dataset is not exhaustive, but a representative scrape of Linkedin.

Primary focus:

- junior posting share and volume
- junior scope redfinition
- senior role redefinition
- source-specific and metro-specific heterogeneity

### RQ2: Task and requirement migration

Which requirements moved downward into junior postings, and which senior-role responsibilities shifted from management toward AI-enabled orchestration?

Primary focus:

- system design
- CI/CD and deployment ownership
- cross-functional coordination
- end-to-end ownership
- AI-tool proficiency
- mentorship / hiring / team-lead language

### RQ3: Employer-requirement / worker-usage divergence

Do employer-side AI requirements outpace observed workplace AI usage, consistent with anticipatory restructuring?

This is a comparison between two different objects:

- employer-side posting requirements
- worker-side observed AI usage or coverage

The contribution is the divergence itself, not a claim that the two measures are directly interchangeable.

### RQ4: Mechanisms

How do senior engineers, junior engineers, and hiring-side actors explain the restructuring of SWE postings?

This is where the mixed-methods design carries real weight. The interviews are not just color. They adjudicate whether observed posting changes reflect:

- real workflow change
- HR / recruiter template inflation
- hiring-market overscreening
- anticipation about where AI is going rather than what it does today

## Main Constructs

### Junior scope inflation

Entry-level or junior-labeled postings increasingly asking for requirements that historically clustered in mid-level or senior roles.

Candidate indicators:

- higher required years of experience within junior-tagged postings
- more system-design, ownership, and architecture language in junior postings
- higher junior-to-senior embedding similarity over time

### Senior archetype shift

Senior postings moving from people-management and team-development language toward review, architecture, AI-enabled leverage, and orchestration language.

Candidate indicators:

- decline in management keywords
- rise in orchestration / review / agent / evaluation language
- shifts in skill and task bundles within senior postings

A decline in management language is consistent with two different forces: AI-enabled leverage moving senior work toward review and orchestration, and structural delayering that compresses the management rung independent of AI (Gallup's 2026 span-of-control data shows average direct reports rising from `10.9` to `12.1` between 2024 and 2025). The posting text alone cannot separate these. The quantitative analysis should report the decline without attributing it solely to AI, and the interview study is responsible for disentangling which mechanism is operating where.

### Posting-usage divergence

The gap between employer-side AI requirements and observed AI usage in comparable occupation groups.

This should be framed as an **employer-requirement / worker-usage divergence index**, not a direct treatment effect.

### Ghost requirements

Requirements listed in postings that hiring-side actors describe as aspirational, template-driven, defensive, or not meaningfully screened in practice.

This is one of the most novel pieces of the project and should be validated qualitatively, not inferred from text alone.

Detailed measurement planning has been archived for now. This file stays focused on the paper design rather than implementation detail.

## Empirical Strategy

### 1. Descriptive restructuring first

Start with high-credibility descriptive facts:

- junior SWE share over time
- junior SWE volume over time
- senior archetype shift over time
- prevalence of migrated requirements
- source-by-source and metro-by-metro heterogeneity

This should be the backbone of the paper.

### 2. Paired historical comparison

Use the 2023-2024 LinkedIn benchmark plus the 2026 scraping pipeline to compare:

- junior postings then vs. now
- senior postings then vs. now
- change in requirement bundles, not just counts

This is much stronger than claiming a single abrupt treatment date.

### 3. Break analysis as supportive evidence

Use endogenous break detection and event-study style plots, but frame them carefully.

Candidate release windows to annotate:

- `2024-05-13` GPT-4o
- `2024-06-20` Claude 3.5 Sonnet
- `2025-05-22` Claude 4

The design should not assume one universal `Post-agent` date. The point is whether posting measures show discontinuities around the broader release era, not whether one exact date "caused" the break.

### 4. Comparative benchmarking, not overclaimed DiD

Brynjolfsson et al., Acemoglu et al., Hampole et al., and Massenkoff and McCrory (2026) provide useful external benchmarks:

- payroll employment by age group
- online vacancy composition
- task exposure and labor-demand effects
- worker-side `observed exposure` and young-worker job-finding rates

The Massenkoff and McCrory finding — a roughly `14%` drop in job-finding rates for 22-25 year olds in high-exposure occupations post-ChatGPT, without a matching aggregate unemployment rise — is especially useful as a downstream benchmark. Their worker-side signal and our employer-side posting-content signal are measuring different parts of the same phenomenon, and the paper should present the two as complementary rather than competing.

We should compare our findings to theirs, but not present our employer-side vacancy design as a direct replication of their employment or firm-level estimates.

### 5. Sensitivity analyses

- LinkedIn-only estimates
- LinkedIn + Indeed pooled estimates
- exclusion of aggregator-like employers
- metro-balanced subsamples
- columns with various levels of preprocessing
- dedupe and repost sensitivity
- measures using canonical postings vs. daily observations

## Outputs

### Main paper figures

1. Junior posting share and volume over time
2. Junior scope-inflation index over time
3. Senior archetype shift index over time
4. Junior-senior embedding similarity over time
5. Requirement migration heatmap by seniority and period
6. Employer-requirement / worker-usage divergence plot
7. Source-specific robustness plots
8. Annotated break-analysis plot with candidate release windows

### Main paper tables

1. Summary statistics by source, period, and seniority
2. Validation results for text measures
3. Regression estimates for junior scope inflation
4. Regression estimates for senior archetype shift
5. Sensitivity and robustness checks
6. Interview sample and mechanism summary table

## Interview Integration

The interview study should directly support RQ4 and partially validate RQ2-RQ3.

Specifically, interviews should test:

- whether AI requirements are real or ornamental
- who changed the JD and why
- whether managers believe AI changed the bar for juniors
- whether senior work has shifted from mentoring to review / orchestration
- whether observed posting changes reflect actual work or anticipatory narrative

The hiring-side cohort is essential here. Protocol details live in `docs/2-interview-design-mechanisms.md`.

## Threats to Validity and Mitigations

| Threat | Why it matters | Mitigation |
|---|---|---|
| Postings are not employment | Limits external validity relative to payroll papers | Be explicit that the outcome is labor-demand signaling, not headcount |
| Historical and current data come from different pipelines | Can induce artificial breaks | Report source-specific results and harmonization checks |
| AI release timing is diffuse | Weakens single-date causal claims | Use release windows and endogenous breaks, not one hard treatment date |
| Seniority is noisy | Titles are inflated | Use title + content classifier, then validate manually |
| AI keywords are noisy | Mention does not equal actual work | Build validated dictionaries and use interviews for adjudication |
| Ghost requirements are latent | Hard to infer from text alone | Treat as mixed-methods construct, not text-only fact |
| Senior archetype shift confounds AI-leverage with org delayering | A decline in management language may reflect structural flattening rather than AI-enabled leverage | Report descriptives without causal attribution; rely on interview probes to separate mechanisms |
| Metro sample is not the entire US | Limits generalization | Describe the frame as major US metros, not all US SWE jobs |
| Platform composition differs | LinkedIn and Indeed serve different slices | Show separate results before pooling |

## Novel Contribution

The most defensible novel contribution is a combination of three things:

1. **A new longitudinal SWE postings dataset** with transparent scraping, harmonization, dedupe, and canonical-posting outputs.
2. **A new measurement framework** for junior scope inflation, senior archetype shift, employer-requirement / worker-usage divergence, and ghost requirements.
3. **A mixed-methods mechanism account** that links employer-side restructuring to worker experience and hiring-side decision-making.

That is more credible and more original than a stronger-sounding but weaker causal claim.

## Best Conference Positioning

### Dataset / methodology venue

Lead with:

- dataset construction
- validation
- measurement framework
- benchmarking value for future labor-market and SE research

### Substantive labor / society / SE venue

Lead with:

- junior scope inflation
- senior archetype shift
- employer-requirement / worker-usage divergence
- mixed-methods mechanism evidence

### What to avoid in the abstract

- "AI caused a labor-market regime break"
- "We identify the employment effect of coding agents"
- "We replicate payroll-based or firm-level adoption studies"

### What to say instead

- "We document rapid restructuring in SWE postings"
- "We find the junior rung narrowing and senior roles shifting in content"
- "These employer-side changes appear larger and faster than observed AI usage benchmarks, and run in the same direction as the worker-side young-hire slowdown reported by Massenkoff and McCrory (2026)"
- "The senior shift is consistent with both AI-enabled leverage and concurrent org delayering; interviews adjudicate which mechanism dominates"
- "Interviews suggest a mix of real workflow change and anticipatory employer beliefs"
