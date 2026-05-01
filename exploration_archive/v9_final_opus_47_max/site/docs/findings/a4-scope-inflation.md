# Every tier asks for broader skills than in 2024; senior roles gained more breadth than junior

!!! quote "Claim"
    Between pooled-2024 and 2026 data, the number of distinct skill or tool requirements per posting — residualised for posting length — rose by 1.58 items for J3 junior postings (postings asking for two or fewer years of experience — the study's primary junior definition) and by 2.61 items for S4 senior postings (postings asking for five or more years of experience). Within the same company, senior gained 1.97 items against junior's 1.43. A decomposition attributes 60 to 85% of the rise to broadening *within* each technical domain, not shifts *between* domains. One tempting alternative explanation — that firms are secretly lowering hiring bars by shrinking the requirements section — is formally rejected. Scope broadened because posting content changed, not because hiring got looser.

## Key figure

![Ranked effect sizes on junior and senior breadth metrics](../figures/T08/fig4_effect_ranking.png)

Effect sizes for breadth, tech-stack count, credential-stack depth, and AI-mention rate on J3 and S4 postings, ranked with confidence intervals. Every metric sits well above zero on both tiers; every metric is larger on senior than on junior.

## Evidence

- Breadth rose by 1.58 items for J3 and 2.61 for S4 overall. Within-company (where we can compare the same firm in both periods), S4 rose 1.97 against J3's 1.43.
- The share of postings listing five or more distinct credential categories rose by 16.9 pp for J3 (from 43% to 60%) and by 13.3 pp for S4 (from 52% to 66%).
- Decomposing the rise into within-domain broadening (an existing kind of role gaining more listed skills) and between-domain composition shift (the mix of roles tilting toward categories that are listed broader), within-domain dominates at 60 to 85% across every metric: breadth, tech count, scope density, credential-stack depth, AI-mention rate.
- For the J3 entry-share rise of +5.07 pp, within-domain accounts for +6.84 pp and between-domain for -0.02 pp. Compositional shifts in role mix do *not* explain the J3 rise.

## What gets ruled out: the "hidden hiring-bar lowering" story

There is a plausible alternative story: requirements sections look shorter, firms are quietly accepting weaker candidates, and the broader-scope appearance is just narrative expansion over the same actual bar. We tested it directly.

Proxy correlations for hiring-bar loosening come back uniformly weak. Spearman correlations (a rank-correlation where +1 means identical ordering, 0 means no relationship) have absolute value at most 0.28, and they flip sign across classifiers. On a hand-labelled sample of 50 postings, zero of 50 contained explicit loosening language. The section-share result that looked like shrinking requirements also flips sign under a simpler-regex classifier. The loosening story does not survive direct tests.

## Sensitivity verdict — strong

- A follow-up replication refit the length residualisation independently; the new residuals matched the original to a mean-absolute-difference of 0.001.
- The within-domain-dominant decomposition holds across the pooled panel, the arshkon-only panel, and the aggregator-excluded panel.
- The hiring-bar-loosening rejection extends from the J3 tier to the aggregate, not just to junior postings.

## Refinement from the pair-level work

Looking at pairs of postings from the same firm with the same job title, the breadth rise shrinks by 90% or more relative to the company-level rise. Scope inflation is happening *across titles within a firm*, not within the same title at the same firm. Firms broaden scope by re-titling and adding new senior-title archetypes (see [A5](a5-archetypes.md)), not by padding individual posting templates.

## Dig deeper

- The breadth and credential-stacking analysis: [source task](../evidence/tasks/T11.md).
- The within-domain decomposition: [source task](../evidence/tasks/T28.md).
- The hiring-bar-loosening rejection: [source task](../evidence/tasks/T33.md).
- The robustness replication: [verification notes](../evidence/verifications.md).
