# AI-tool growth is SWE-specific, while length and scope are broader across roles.

## Claim
The best cross-occupation reading is porous boundary, not collapse: SWE changes are real, but some posting-template drift is broader than SWE.

T18 shows that AI-tool language is the cleanest SWE-specific signal relative to adjacent and control occupations. Length and scope, by contrast, move more like field-wide template drift. T19 then shows the time structure is windowed, with sharp jumps between the 2024 baseline and the scraped 2026 window. The direction is useful, but both T18 and T19 still need the analysis-phase recomputation noted in V2.

## Evidence
- AI-tool share rises from 1.98% to 20.38% for SWE, from 2.45% to 18.14% for adjacent roles, and only from 1.25% to 1.41% for controls.
- Boundary similarity stays high at about 0.80-0.83, which means the SWE-adjacent boundary is porous but not collapsed.
- Requirement breadth rises from 5.94 to 6.98, raw description length from 2,974 to 4,891, and AI-tool share from 2.56% to 20.30% across the source windows.
- Posting age clusters at one day in the scraped window, which looks more like scrape cadence than backlog age.

## Figures
![T18_parallel_trends.png](../assets/figures/T18/T18_parallel_trends.png)

![T18_boundary_similarity.png](../assets/figures/T18/T18_boundary_similarity.png)

![T19_source_window_rates.png](../assets/figures/T19/T19_source_window_rates.png)

## Sensitivity and caveats
- The control-group result keeps the AI-tool story from becoming a generic macro trend.
- Scope and length still rise in controls, so they should not be framed as SWE-only breaks.
- Remote status is unusable in the selected metro frame and should not be over-interpreted.
- Treat the T18/T19 magnitudes as directionally informative until the recomputation pass locks the exact aggregates.

## Raw trail
- [T18 report](../audit/raw/reports/T18.md)
- [T19 report](../audit/raw/reports/T19.md)
- [V2 verification](../audit/raw/reports/V2_verification.md)

## What this means
- Keep AI-tool language as the SWE-specific signal, and keep length/scope as a broader posting-surface shift.
- Do not model the data as a continuous time series; it is a set of windows and snapshots.
- Cross-occupation work is a specificity check, not a new lead claim.
