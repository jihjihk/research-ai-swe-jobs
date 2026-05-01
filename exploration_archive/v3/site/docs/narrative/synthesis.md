# Full Synthesis

This page reproduces the consolidated exploration synthesis. For the canonical version, see [SYNTHESIS.md](../reports/SYNTHESIS.md) in the reports section.

## The paper's strongest story

Based on 23 exploration tasks and 4 gate reviews, the evidence supports this narrative:

**The SWE labor market restructured between 2024 and 2026 through three SWE-specific mechanisms operating in parallel:**

1. **AI competency requirements surged** (+24pp DiD vs control), validated as genuine hiring requirements (not aspirational). Requirements still lag developer usage but are converging rapidly. [Details](../findings/ai-requirements.md)

2. **Entry-level SWE posting share declined** while non-SWE junior share increased (DiD = -25pp). But AI adoption and entry-level changes are orthogonal at the firm level (r ~ 0) -- these are parallel market trends, not causally linked within organizations. Over half (57%) of aggregate change is compositional. [Details](../findings/junior-decline.md)

3. **The SWE domain landscape recomposed**: ML/AI engineering grew from 4% to 27% of postings while frontend/web contracted. Domain is 10x more structurally important than seniority in determining posting content. [Details](../findings/domain-recomposition.md)

## Supporting findings

- **AI is additive to technology stacks:** AI-mentioning postings require 11.4 techs vs 7.3 non-AI. Stack diversity increased from 6.2 to 8.3. [Details](../findings/technology-expansion.md)
- **YOE slot purification:** 5+ YOE entry-level postings dropped from 22.8% to 2.4%. The surviving junior roles are more genuinely junior.
- **Senior orchestration shift:** Director-level orchestration surged +46%. Management expanded everywhere, not migrated from senior to entry.
- **GenAI acceleration:** 8.3x between within-2024 and cross-period adoption rates.

## What the paper should NOT claim

- That AI caused junior elimination within firms (r ~ 0 at firm and metro level)
- That management scope inflation is dramatic (+4-10pp, not +31pp) or SWE-specific (field-wide)
- That junior and senior postings are semantically converging (fails calibration)
- That posting requirements outpace developer usage (they lag by ~34pp)
- That soft skills expansion is SWE-specific (SWE grew LESS than control)

See [What We Can Claim](claims.md) for the full enumeration, and [Corrections & Revisions](../findings/corrections.md) for how the narrative self-corrected.

## Data quality verdicts by research question

### RQ1: SWE labor demand restructuring

| Dimension | Quality | Notes |
|-----------|---------|-------|
| Junior share trend | Good (with caveats) | Direction robust on seniority_native; depends on operationalization |
| AI requirements | Strong | Binary keyword detection robust across all checks |
| Scope inflation | Moderate (corrected) | Management corrected from +31pp to +4-10pp |
| Domain recomposition | Strong | Method-robust clustering (ARI >= 0.996) |
| Within-firm vs compositional | Strong | 57% compositional from shift-share decomposition |

### RQ2: Technology ecosystem evolution

| Dimension | Quality |
|-----------|---------|
| Technology mention rates | Strong |
| Co-occurrence networks | Strong |
| Stack diversity | Strong |
| AI additive pattern | Strong |

### RQ3: Employer-requirement / worker-usage divergence

| Dimension | Quality | Notes |
|-----------|---------|-------|
| Posting-side AI rates | Strong | ~41% in 2026 |
| Usage-side benchmarks | Moderate | External surveys with different sampling frames |
| Divergence computation | Moderate | Direction clear (posting lags usage) |

### RQ4: Mechanisms (qualitative)

| Dimension | Quality |
|-----------|---------|
| Interview artifacts | Ready (T25) |
| Stimuli grounded in data | Strong |

## Next steps

See [the presentation](../presentation.md) slides 21-23 for the analysis sequencing plan, and [Limitations](../methodology/limitations.md) for the full risk inventory.
