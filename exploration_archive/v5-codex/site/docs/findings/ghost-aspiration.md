# Ghost-like language looks aspirational and template-heavy, not fake jobs.

## Claim
The ghost story is not that postings are fake. It is that some postings are overloaded, hedged, and templated in ways that make them look more aspirational than screening-tight.

T22 shows AI postings have a higher hedge-to-firm ratio than non-AI postings, and direct employers are slightly more ghost-like overall. The best reading is bundled scope plus aspirational wording, not literal fabricated roles. Use this as a mechanism hypothesis until the T22 recomputation is finished.

## Evidence
- On the section-filtered LLM core, AI postings have a hedge/firm ratio of about 0.73 versus 0.52 for non-AI postings.
- The raw-text sensitivity preserves the direction, though the magnitudes change.
- Direct employers are slightly more ghost-like overall; aggregators contribute a distinct form of template saturation.
- The validated management artifact demotes broad management to sensitivity only, which is a useful reminder that generic wording can be misleading.

## Figures
![T22_ai_aspiration_ratio.png](../assets/figures/T22/T22_ai_aspiration_ratio.png)

![T22_ghost_score_by_period_seniority.png](../assets/figures/T22/T22_ghost_score_by_period_seniority.png)

![T22_template_saturation_top_companies.png](../assets/figures/T22/T22_template_saturation_top_companies.png)

## Sensitivity and caveats
- The result is noisy enough that it should stay in the validity section, not the paper's headline.
- Broad management is a bad proxy for the same reason ghost language is tricky: generic words absorb too much noise.
- This is more a mechanism question than a proof of fake jobs.
- The exact hedge ratios still need the analysis-phase recomputation from V2.

## Raw trail
- [T22 report](../audit/raw/reports/T22.md)
- [Validated management patterns](../assets/artifacts/shared/validated_mgmt_patterns.json)

## What this means
- AI wording looks more aspirational than non-AI wording, but that is not the same as fake jobs.
- The right follow-up is qualitative: who writes the posting, how often is it templated, and what changes screening practice?
- Ghost language is a mechanism lens, not a headline result.
