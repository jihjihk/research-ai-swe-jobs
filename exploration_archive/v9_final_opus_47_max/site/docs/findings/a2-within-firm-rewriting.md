# When firms add AI language to postings, they rewrite the same titles rather than invent new ones

!!! quote "Claim"
    Three independent panels of firms that posted in both 2024 and 2026 show the same thing. At the company level, AI-mention rates rose by 7.7 to 8.3 percentage points (a percentage point is the difference between 10% and 18%, not an 8% proportional rise). At the level of same-company, same-title pairs, the rise is 10 to 13 points. That the pair-level rise is *larger* than the company-level rise is the cleanest available test that firms are rewriting the same titles, not replacing old titles with new ones.

## Key figure

![Pair-level AI-mention drift](../figures/T31_drift_scatter.png)

Each point is a same-company, same-title pair with at least three postings in both 2024 and 2026. The x-axis is the change in AI-mention rate; the y-axis is the change in requirement breadth (the number of distinct skill or tool requirements per posting, residualised for posting length). The scatter leans right, so AI mentions rise. The y-axis is symmetric, so within the same title breadth does not rise. Scope broadening happens across titles, not within them.

## Evidence

Three panels, using our pattern-based AI-mention detector (which catches named AI tools like Copilot and Cursor, LLM-related terms, and AI-coding-assistant mentions):

| Panel | Metric | N | Rise |
|---|---|---|---|
| Arshkon-only company panel (min. 5 postings) | company-level AI-mention rate | 125 companies | +8.34 pp |
| Pooled company panel (min. 5 postings) | company-level AI-mention rate | 356 companies | +7.65 pp |
| Returning-firms cohort (2,109 firms posting in both years) | company-level AI-mention rate | 2,109 companies | +7.91 pp |
| Same-title pair panel (primary, arshkon-only, min. 3) | pair-level AI-mention rate | 23 pairs | +13.4 pp |
| Same-title pair panel (relaxed filter) | pair-level AI-mention rate | 37 pairs | +9.98 pp |
| Same-title pair panel (strict arshkon-only) | pair-level AI-mention rate | 12 pairs | +13.3 pp |

Company-level rise: +7.7 to +8.3 pp. Pair-level rise: +10 to +13 pp.

## Exemplar

The cleanest case is Microsoft's "Software Engineer II." Six postings in 2024, 35 in 2026, same title. The 2024 six do not mention Copilot, generative AI, or AI systems. The 2026 thirty-five do. Across that pair, the AI-mention rate jumps 40 percentage points. Parallel rewrites show up at Wells Fargo and Capital One on their own senior engineer titles.

## Sensitivity verdict — strong direction, range-reportable magnitude

A follow-up replication raised two flags worth naming.

The first flag: the original primary pair count of 23 does not reproduce exactly from the documented methodology. Replicating the pair construction gives 37 pairs at +9.98 pp under a relaxed filter, or 12 pairs at +13.3 pp under a strict source filter. The direction — pair-level rise exceeding company-level rise — holds across every reconstruction.

The second flag: under the cleaner version of our AI-mention pattern (the `v1_rebuilt` variant, see the [corrections page](corrections.md)), company-level magnitude drops slightly, to +7.5 to +7.8 pp. Direction unchanged.

One more note. Excluding recruiting aggregators tightens the pair-level signal from +13.4 to +16.5 pp. Direct employers are leading the rewriting; aggregators dilute it.

## Dig deeper

- The pair-level analysis: [source task](../evidence/tasks/T31.md).
- The company-level panels: [source task](../evidence/tasks/T16.md).
- The returning-firms cohort: [source task](../evidence/tasks/T37.md).
- The robustness replication and flags: [verification notes](../evidence/verifications.md).
- The interview exemplar set documenting the Microsoft SE II rewrite: stored under `exploration/artifacts/T25_interview/`.
