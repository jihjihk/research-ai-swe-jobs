# Finding 2 — Within-company scope broadening is real

## Claim

**On the 240-company arshkon ∩ scraped overlap panel, length-residualized requirement-breadth rose +1.43 composite units within-company (102% of aggregate), 76% of companies broadened, 62% broadened by more than 1.0 unit — the pattern holds equally at junior and senior seniority levels.**

## Core numbers

| Metric | Value | Source |
|---|---:|---|
| Overlap-panel companies (arshkon ∩ scraped, ≥3 postings each period) | 240 | T16 |
| Length-residualized breadth Δ within-company | **+1.43 units** | T16 |
| Share of companies that broadened | **76 %** | T16 |
| Share that broadened by > 1.0 unit | 62 % | T16 |
| Within-company share of AI-strict rise | 102 % | T16 |
| Raw breadth Δ (not residualized) | +39 % | T11 |
| Length-content split of raw breadth | 71 % content / 29 % length | V1 |

The 102% is not a typo — length residualization removes a small between-component, leaving within slightly larger than the aggregate.

## Key figures

![Within vs between decomposition](../figures/T16/02_within_between_decomposition.png)
*T16 — Oaxaca decomposition of AI-strict and breadth-residualized on the 240-co panel. Within-company bars (blue) dominate; between-company bars (orange) are near zero.*

![Distribution of company-level breadth change](../figures/T16/03_breadth_change_distribution.png)
*T16 — 76% of the 240 overlap companies broadened; 62% broadened by more than 1.0 composite unit. Distribution is right-shifted, not bimodal.*

![Aggregator vs direct employer](../figures/T16/04_aggregator_vs_direct.png)
*T16 — Aggregators and direct employers show nearly identical AI+breadth trajectories (+14.1 vs +12.4pp). Aggregator exclusion is a sensitivity, not a driver.*

## Sensitivity grid (five variants)

Within-company share of the breadth rise:

| Variant | n cos | Within-co share |
|---|---:|---:|
| Primary panel | 240 | 102 % |
| No entry-specialist | 221 | 94 % |
| No aggregator | 213 | 112 % |
| Cap-50 postings per co per period | 240 | 108 % |
| Pooled-2024 baseline | 589 | 147 % |
| LLM-labeled subset only | 196 | 119 % |

The within-company signal survives every ablation.

## What it rules out

- **"Different companies arrived in 2026."** 74.5% of scraped companies are new entrants, but the overlap panel's 240 returning companies independently show the breadth rise.
- **"It's just that postings got longer."** Length-residualization (global OLS, a=6.498, b=0.00182) absorbs 29% of the raw rise. The residual 71% is a real increase in the number of distinct requirement types per posting.
- **"Junior scope inflation."** Breadth rose +39% at junior, +30% at senior — parallel. Not a junior-rung-narrowing story; it is a corpus-wide scope diversification.

## Tension to flag

**T11 breadth +39% rises vs T13 requirements-section −19% shrink.** Both are real. `requirement_breadth` counts distinct requirement types anywhere in the cleaned description. The formal requirements *section* itself contracted. The mechanism is **requirements types migrating into responsibilities, role_summary, and about_company**. Any reviewer will raise this — the paper must address it head-on.

## Limitations

- **Content split 71/29 (content/length).** 29% of the raw rise is length-driven. Always cite the length-residualized number as primary.
- **Composite-score correlation with length.** V1 found soft_skill_count r=0.363, org_scope_count r=0.399, management_STRICT r=0.351 with desc_cleaned_length. Residualization protocol applies to any composite.
- **Overlap panel is small.** 240 companies is enough for the decomposition to be statistically clear, but not for fine stratification by archetype × seniority × period.

## Links to raw

- [T11 — requirements complexity](../raw/wave-2/T11.md)
- [T13 — section anatomy](../raw/wave-2/T13.md)
- [T16 — company strategies](../raw/wave-3/T16.md)
- [V1 verification](../raw/verification/V1_verification.md) — length residualization protocol
</content>
</invoke>