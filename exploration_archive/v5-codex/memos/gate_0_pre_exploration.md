# Gate 0 Pre-Exploration Note

## Initial Assessment

This note records the prior before Wave 1. It is not evidence from the data.

The initial RQ1-RQ4 narrative is plausible but fragile. The strongest paper would be an employer-side restructuring paper if Wave 1 establishes that seniority, SWE classification, and cross-source comparability are credible enough to support period comparisons. If those measurement checks are weak, the safer positioning shifts toward a dataset/methods and measurement paper, with substantive findings framed as exploratory signals rather than headline claims.

## Most Confident Going In

The dataset can support a careful LinkedIn-only historical comparison at the posting-content level, provided findings are interpreted as employer demand signals rather than employment effects. The schema is explicit about source, platform, seniority provenance, LLM coverage, and text quality, which gives the exploration enough metadata to separate many measurement artifacts from substantive changes.

I am also relatively confident that description length, company concentration, aggregator status, and cross-source composition will matter for interpretation. These are not just nuisances; they may become part of the story if apparent restructuring is driven by who posts, how platforms format postings, or which employers dominate entry-level rows.

## Least Confident Going In

Seniority is the highest-risk construct. `seniority_final` is the production column, but it combines strong title rules with LLM overwrite only for routed rows, while `seniority_native` is unusable as a pooled 2024 entry proxy because asaniczka has no native entry labels. The first binding question is whether `seniority_final`, the YOE-based proxy, and the arshkon-only native baseline agree on the direction of junior-share change. If they do not, the paper cannot lead with a simple junior-rung narrative until the mechanism of disagreement is understood.

Text-dependent scope findings are also at risk. `description_core_llm` is the only cleaned text source, and its coverage may be the binding constraint for topic models, requirement density, and embedding comparisons. Raw description comparisons may mostly measure formatting, boilerplate, or source pipeline differences rather than employer requirement change.

Cross-period inference is the third major risk. The comparison is not a clean panel: arshkon, asaniczka, and scraped data are different instruments. The within-2024 arshkon-vs-asaniczka calibration must set the noise floor. If 2024-to-2026 effects are smaller than within-2024 source differences, they should not be treated as substantive temporal shifts.

## What Would Redirect the Project

The project direction should change if Wave 1 finds any of the following:

- `seniority_final` and the YOE-based proxy disagree on the direction of the entry-level trend, especially if the disagreement is large or source-specific.
- Entry-level cells are too small or too concentrated in a few employers to support robust seniority-stratified claims.
- LLM cleaned-text coverage is too thin or imbalanced to support scope, topic, or embedding analyses without severe sample restriction.
- SWE classification has period-specific false positives, especially around ML, data, DevOps, or adjacent engineering roles.
- Within-2024 arshkon-vs-asaniczka variability is comparable to or larger than the proposed 2024-to-2026 changes for key outcomes.
- Company composition or aggregator concentration explains most apparent changes before any substantive role-content mechanism is visible.

## Initial Narrative Test

At Gate 0, the original narrative should be treated as a working hypothesis only:

- RQ1, employer-side restructuring: plausible, but depends on seniority validation and concentration checks.
- RQ2, task and requirement migration: plausible, but depends on cleaned-text coverage and within-2024 calibration.
- RQ3, employer-requirement / worker-usage divergence: conceptually promising, but likely weak until benchmark quality and AI-requirement measurement are validated.
- RQ4, mechanisms: important for interpretation, especially if quantitative evidence cannot distinguish real workflow change from template inflation or anticipatory signaling.

Two alternative framings should be kept live from the start. First, the story may be market recomposition rather than firm-level restructuring: different companies may be posting in 2026, and entry-level hiring may be concentrated among a narrow set of employers. Second, the story may be platform and description evolution rather than real demand change: source formatting, boilerplate, aggregator behavior, or posting templates may generate apparent scope inflation.

## Wave 1 Steering Priorities

Wave 1 should not try to prove a labor-market result. It should establish what the data can support:

1. Treat seniority validation as the central gate.
2. Make within-2024 calibration a baseline for all later effect sizes.
3. Identify binding sample-size and coverage constraints for seniority, text, geography, and company-level analysis.
4. Surface concentration and duplicate-template risks before Wave 2 topic or keyword work begins.
5. Use the T07 feasibility table to demote underpowered analyses early rather than spending Wave 2 effort on fragile comparisons.

