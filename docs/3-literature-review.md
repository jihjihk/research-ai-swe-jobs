# Literature Review

This file summarizes the external literature most relevant to the project. It is not the canonical statement of the paper design. For the project's claims and methods, see `docs/1-research-design.md`.

## 1. What Prior Labor-Market Papers Actually Measure

The most relevant papers split along two axes: what they measure (tasks, vacancies, or workers) and which side of the market they observe (employer-side or worker-side). That split matters for our positioning — our project lives in the employer-side, posting-content slice, and reads as complementary to, not duplicative of, the worker-side and employment papers.

**Worker-side and employment-side evidence.**

- **Brynjolfsson et al. (2025)** use ADP payroll data to study realized employment, including age-group differences.
- **Massenkoff and McCrory (2026)** at Anthropic introduce an `observed exposure` measure that combines O*NET task definitions, Eloundou et al. theoretical feasibility scores, and aggregated Claude usage, with automation/augmentation weighting. They find that job-finding rates for workers aged 22-25 in high-exposure occupations dropped by roughly `14%` after late 2022, without a corresponding systematic rise in unemployment.
- Earlier Anthropic Economic Index reports (Handa et al. 2025; Appel et al. 2026 "Economic Primitives") provide the usage-side inputs to that measure: what tasks Claude is actually asked to do, separate from the theoretical-feasibility question.

**Employer-side and vacancy-side evidence.**

- **Acemoglu et al. (2022)** use Burning Glass online vacancies to study AI-related hiring and vacancy composition.
- **Hampole et al. (2025)** use Revelio data to study task exposure, labor demand, and firm-level offsetting effects.
- **Levanon et al. (2025)** at the Burning Glass Institute document structural shifts in entry-level hiring.

Our project primarily studies **employer-side restructuring visible in job postings**, which is closest in spirit to the online-vacancy literature. The Massenkoff and McCrory finding on the worker side is directly relevant: it suggests the same young-worker compression we observe upstream in posting content is also visible downstream in realized hiring.

## 2. What the Literature Already Establishes

### Early-career workers appear more exposed

Recent work suggests larger negative impacts for younger or entry-level workers in AI-exposed roles. Brynjolfsson et al. document age-heterogeneous employment effects in payroll data. Massenkoff and McCrory (2026) add a complementary worker-side finding: job-finding rates for 22-25 year olds in high-exposure occupations fell by roughly `14%` after late 2022, even though aggregate unemployment in exposed occupations did not systematically rise. More broadly, the literature increasingly argues that AI automates many "stepping-stone" tasks that historically helped juniors build expertise.

### Task-level substitution does not imply large aggregate employment collapse

Acemoglu, Hampole, and Anthropic collectively point to the same broad pattern:

- AI meaningfully overlaps with many task bundles
- employers may change hiring and task composition
- aggregate employment effects remain modest or mixed so far

That is exactly why employer-side restructuring is worth studying. Role definitions can move before aggregate employment fully does.

### AI usage is below theoretical exposure

Anthropic's exposure work is particularly useful because it distinguishes:

- what AI could theoretically do
- what workers appear to actually use it for
- a composite `observed exposure` that weights feasible tasks by actual usage and by whether that usage is automating or augmenting the task

That gap is large for our population. Massenkoff and McCrory (2026) report that while roughly `94%` of tasks in "Computer & Mathematical" occupations are rated theoretically feasible for LLMs, current Claude usage covers only about `33%` of those tasks, and the composite observed-exposure score for computer programmers comes in below the theoretical ceiling. That gap is what our `employer-requirement / worker-usage divergence` construct targets: employer-side AI requirements in SWE postings may track the theoretical ceiling while worker-side observed usage lags behind.

## 3. Concepts Most Relevant to This Project

### Junior scope inflation

The literature increasingly suggests that employers may compress entry-level pathways when AI can perform more foundational tasks. Our contribution is to measure whether junior SWE postings now resemble older mid-level or senior postings in requirement content.

### Senior archetype shift

Prior papers say less about how senior technical roles themselves are changing. This project focuses on whether senior SWE postings shift away from mentorship and people-management language toward orchestration, architecture, review, and AI-enabled leverage.

### Ghost requirements and anticipatory restructuring

The literature gives reasons to think employers may act on expectations rather than observed capability alone. But the mechanism is still underspecified. This project uses interviews to test whether AI-related requirements are:

- actually screened
- copied from templates
- driven by leadership signaling
- responses to current workflows
- anticipatory bets on future capability

## 4. Concurrent Industry Evidence

Two strands of industry data are useful as corroboration and as reminders of confounding forces, even though they are not academic sources.

### Aggregate tech demand is not collapsing

Rachitsky's March 2026 summary of TrueUp data reports roughly 67,000 open engineering roles and 7,300 open PM roles globally, with engineering volumes still growing. This is consistent with Massenkoff and McCrory's finding that no systematic unemployment rise has appeared in exposed occupations. It also reinforces the framing this project takes: the story is **compositional restructuring inside a healthy aggregate**, not headline job destruction.

### Org flattening is a parallel restructuring force

Gallup's 2026 span-of-control data shows the average number of direct reports per US manager rose from `10.9` in 2024 to `12.1` in 2025, with the mean driven upward by a rise in teams of 25 or more. Industry commentary (Musa 2026 at Levels.fyi) points to dramatic cases such as Amazon's explicit delayering program and Meta's roughly 50-to-1 applied-AI ratio.

This matters for our `senior archetype shift` construct. A decline in management language in senior SWE postings could be driven by two separable forces:

- AI-enabled leverage moving senior work toward review, orchestration, and agent supervision
- structural delayering that thins the management rung independent of AI

The quantitative analysis cannot cleanly separate these from posting text alone. The interview study has to carry that weight, and the protocol should probe delayering explicitly so we do not attribute purely organizational restructuring to AI.

## 5. Where the Gap Is

The existing literature is strong on:

- labor-demand or employment outcomes
- task exposure
- broad AI usage patterns

It is weaker on:

- how SWE role definitions changed at the posting level
- how those changes differ across junior and senior rungs
- whether posting changes outpace observed AI usage
- who inside firms is actually responsible for the change

That is the gap this project targets.

## 6. What This Project Adds

This project contributes:

1. A new longitudinal SWE postings dataset
2. Measures of junior scope inflation, senior archetype shift, and employer-requirement / worker-usage divergence
3. Mixed-methods evidence on the mechanisms behind inflated or changing requirements

## 7. What Not to Overclaim

This literature also clarifies the main limits of the project:

- employer-side posting data do not directly identify employment effects
- employer-side AI requirements are not the same as observed worker-side AI usage
- break timing is inherently fuzzy because model releases and organizational uptake are staggered

Those limits should be explicit in the paper rather than hidden.
