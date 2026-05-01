# The software-engineering vs. control gap doesn't depend on which controls we picked

!!! quote "Claim"
    The SWE AI difference-in-differences estimate (the 2024-to-2026 change in SWE postings minus the change in control occupations, so a labour-market-wide shock would cancel out) is +13.10 percentage points under the cleaner version of our AI-mention pattern, with a 95% confidence interval of [+12.76, +13.45], and +14.04 pp under the primary pattern. The estimate holds within half a point of the +13 to +14 pp range under every alternative control we tried: dropping analysts, dropping nursing, restricting to manual-work controls, or dropping the title-lookup-LLM SWE tier.

## Why this matters

A difference-in-differences result is only as credible as its control group. If the answer changes when you change the controls, you have not actually measured a SWE-specific shift; you have measured an artifact of which occupations you picked to compare against. We ran the replication that tests this directly.

## Specification table

| Specification | SWE gap | Within 0.5 pp of +13 to +14? |
|---|---|---|
| Primary (cleaner AI pattern, default controls) | +13.10 pp | — |
| Primary (top-level AI pattern, default controls) | +14.04 pp | — |
| Drop data and financial analysts from controls | +13.18 pp | yes |
| Drop nursing (the largest control subgroup) | +13.02 pp | yes |
| Manual-work controls only | +13.44 pp | yes |
| Drop the title-lookup-LLM SWE tier (7% of SWE postings) | +13.19 pp | yes |

## Sensitivity verdict — strong

The specification swing across the alternatives is half a percentage point; the gap being measured is fourteen. That ratio matters. Dropping the 7% of SWE postings whose classification depended on the embedding-plus-LLM reconciliation step still produces +13.19 pp, so the gap is not an artifact of the harder-to-classify SWE margin.

## Dig deeper

- The difference-in-differences construction: [source task](../evidence/tasks/T18.md).
- The five-specification alternative-control replication: [verification notes](../evidence/verifications.md).
- The universality analysis that extends this from a binary SWE-vs-control gap to a continuous occupation spectrum: [A1](a1-cross-occupation-divergence.md).
