# Literature Review

This file summarizes the external literature most relevant to the project. It is not the canonical statement of the paper design. For the project's claims and methods, see `docs/1-research-design.md`.

## 1. What Prior Labor-Market Papers Actually Measure

The most relevant papers use different datasets and outcomes:

- **Brynjolfsson et al. (2025)** use ADP payroll data to study realized employment, including age-group differences.
- **Acemoglu et al. (2022)** use Burning Glass online vacancies to study AI-related hiring and vacancy composition.
- **Hampole et al. (2025)** use Revelio data to study task exposure, labor demand, and firm-level offsetting effects.
- **Anthropic (2026)** introduces occupation-level measures of theoretical exposure and observed AI usage based on Claude interactions.

This distinction matters. Our project primarily studies **employer-side restructuring visible in job postings**, which is closest in spirit to the online-vacancy literature, not the payroll-employment literature.

## 2. What the Literature Already Establishes

### Early-career workers appear more exposed

Recent work suggests larger negative impacts for younger or entry-level workers in AI-exposed roles. Brynjolfsson et al. document age-heterogeneous employment effects in payroll data. More broadly, the literature increasingly argues that AI automates many "stepping-stone" tasks that historically helped juniors build expertise.

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

That gap motivates our `employer-requirement / worker-usage divergence` construct.

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

## 4. Where the Gap Is

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

## 5. What This Project Adds

This project contributes:

1. A new longitudinal SWE postings dataset
2. Measures of junior scope inflation, senior archetype shift, and employer-requirement / worker-usage divergence
3. Mixed-methods evidence on the mechanisms behind inflated or changing requirements

## 6. What Not to Overclaim

This literature also clarifies the main limits of the project:

- employer-side posting data do not directly identify employment effects
- employer-side AI requirements are not the same as observed worker-side AI usage
- break timing is inherently fuzzy because model releases and organizational uptake are staggered

Those limits should be explicit in the paper rather than hidden.
