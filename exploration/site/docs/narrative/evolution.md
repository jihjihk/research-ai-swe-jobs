# How the Story Evolved: From "Junior Scope Inflation" to "Parallel Restructuring"

The exploration narrative changed substantially across four waves and four gate memos. This page traces how each wave's findings reshaped the paper's framing.

## Pre-exploration (Gate 0): The hypothesis

**Starting narrative:** AI coding tools are causing junior scope inflation -- employers are demanding more from entry-level developers, effectively raising the bar for junior SWE roles.

**Key assumptions:**
- Seniority is the primary restructuring axis
- AI requirements are probably aspirational ("ghost requirements")
- Requirements likely outpace developer usage
- The changes are probably SWE-specific

**Confidence level:** Moderate. The YOE paradox (entry YOE decreased, contradicting scope inflation) was already flagged as a red flag.

## Wave 1 (Gate 1): "Quantity, not yet quality"

**What changed:** The data told a story about **quantity reduction, not scope inflation**. Established companies posted fewer entry-level SWE roles, and the remaining ones actually required less experience.

**Key discoveries:**
- Junior share declined 5.9-8.3pp (robust across 4 operationalizations)
- Within-company decline (-11.8pp) exceeded aggregate -- composition dampens, not inflates
- Entry-level YOE decreased (3.0 to 2.0) -- the "scope inflation" hypothesis was challenged
- Data was much larger than documented (52K SWE, not ~33K)

**Narrative shift:** From "junior scope inflation" to "junior slot elimination with entry-level standardization." The paper leads with quantity reduction; whether quality changed is deferred to Wave 2.

**Emerging RQ1b:** Within-firm vs between-firm decomposition identified as potential methods contribution.

## Wave 2 (Gate 2): "Purification plus scope expansion" -- the peak of the management story

**What changed:** Text analysis revealed both quantity and quality changes, producing the study's most dramatic (and later corrected) finding.

**Key discoveries:**
- Management indicators: 9.4% to 40.8% (+31pp) at entry level -- the headline number
- AI requirements: 3.9% to 27.5% at entry level
- Domain recomposition: ML/AI 4% to 27% (a finding not in the original design)
- AI is additive to stacks (11.4 vs 7.3 techs)
- Junior-senior semantic convergence appeared to hold
- Technology domain is 10x more important than seniority structurally

**Narrative shift:** From "slot elimination" to "simultaneous purification and redefinition." The proposed abstract led with the paradox: fewer juniors, lower YOE bars, but dramatically broader qualitative requirements.

**Risk identified:** The +31pp management figure was flagged as needing ghost forensics validation (T22). Cross-occupation specificity test (T18) identified as make-or-break.

## Wave 3 (Gate 3): The great correction -- three findings overturned

Wave 3 was the most consequential. It confirmed the strongest findings while overturning several others.

**What survived:**
- AI requirements surge: confirmed SWE-specific (DiD +24.4pp) and genuine (not aspirational)
- Junior share decline: confirmed SWE-specific (DiD -24.9pp) with control going opposite direction
- Domain recomposition: inherently SWE-specific, method-robust

**What was corrected:**
- Management indicator: +31pp became +4-10pp (measurement error in regex patterns)
- Management expansion: field-wide (DiD ~ 0), not SWE-specific
- Soft skills expansion: SWE grew LESS than control (DiD = -5.1pp)
- Junior-senior convergence: failed within-2024 calibration
- Requirements vs usage: inverted (requirements lag, not lead)
- Management migration: rejected (expanded at all levels, not migrated)

**New discoveries:**
- AI-entry orthogonality (r = -0.07 firm, r = -0.04 metro) -- the most important causal caveat
- 57% of aggregate change is compositional
- Senior orchestration surge (+46% at director)
- Associate collapsing toward entry

**Narrative shift:** From "paradoxical junior redefinition" to "parallel restructuring through domain recomposition and AI skill expansion, not through seniority-level task migration." The causal claim (AI drives junior decline) was explicitly abandoned.

## Wave 4 (Synthesis): The final framing

**Final narrative:** The SWE labor market restructured through three parallel, SWE-specific mechanisms: AI competency requirements surged, entry-level share declined (but orthogonally to AI within firms), and the domain landscape recomposed from frontend toward ML/AI. The corrections are presented as methodological rigor, not weakness.

**What the evolution demonstrates:**
1. **Iterative validation works.** The initial +31pp management finding would have been a published error without Wave 3's forensic analysis.
2. **Cross-occupation DiD is essential.** Without T18, the paper would have claimed SWE-specificity for field-wide trends.
3. **Within-firm decomposition changes the story.** The 57% compositional finding reframes "restructuring" as partly "recomposition of which companies are hiring."
4. **Negative findings are findings.** The orthogonality result (r ~ 0) is arguably the study's most important contribution to public discourse.

## Summary timeline

| Stage | Lead narrative | Key number | Status |
|-------|---------------|------------|--------|
| Gate 0 | Junior scope inflation | -- | Hypothesis |
| Gate 1 | Junior slot elimination | -11.8pp within-company | Partially confirmed |
| Gate 2 | Purification + scope expansion | +31pp management | Peak (later corrected) |
| Gate 3 | Parallel restructuring, not causal | +4-10pp (corrected); r = -0.07 | Final framing |
| Synthesis | Three parallel SWE-specific mechanisms | DiD: AI +24pp, Junior -25pp | Delivered |

## Full gate memos

- [Pre-Exploration (Gate 0)](../memos/gate_0_pre_exploration.md)
- [Gate 1: Data Foundation](../memos/gate_1.md)
- [Gate 2: Open Structural Discovery](../memos/gate_2.md)
- [Gate 3: Market Dynamics](../memos/gate_3.md)
