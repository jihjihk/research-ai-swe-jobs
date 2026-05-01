# Requirement stacking rose more clearly than raw tech breadth.

## Claim
The stronger content result is that 2026 postings ask for more categories at once, not simply more technologies.

T11 shows the change is concentrated in requirement breadth, credential stack depth, scope density, and AI mentions. `tech_count` moves only modestly and does not cleanly clear the 2024 calibration baseline, so the paper should not lead with a generic 'more tech' claim.

## Evidence
- On the primary company-capped LLM-text subset, `requirement_breadth` and `credential_stack_depth` both rise from 2024 to 2026.
- Scope density and AI mentions are the clearest increases.
- The junior-like explicit-entry slice and the YOE proxy both show the same direction, but the YOE slice is broader and more complex.
- T13 explains why part of the apparent growth is a form change: the postings are more sectioned in 2026.

## Figures
![T11_complexity_distributions.png](../assets/figures/T11/T11_complexity_distributions.png)

![T11_entry_complexity_comparison.png](../assets/figures/T11/T11_entry_complexity_comparison.png)

![T13_section_composition.png](../assets/figures/T13/T13_section_composition.png)

## Sensitivity and caveats
- Company capping keeps the direction of breadth, stack, and scope intact.
- Raw-text fallback nudges the magnitudes, but it does not reverse the story.
- Broad management is noisy and should stay out of the lead result.

## Raw trail
- [T11 report](../audit/raw/reports/T11.md)
- [T13 report](../audit/raw/reports/T13.md)
- [T08 report](../audit/raw/reports/T08.md)

## What this means
- The publishable claim is credential stacking around the SWE core, not raw tech inflation.
- AI and scope belong at the center of RQ2, not as side examples.
