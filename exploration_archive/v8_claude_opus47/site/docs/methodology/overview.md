# Methodology overview

Five waves of parallel exploration, each followed by a research-director gate memo. Two adversarial verification rounds. Eighteen task reports, four gate memos, two verification reports.

## Wave structure

| Wave | Status | Agents | Purpose | Key outputs |
|---|---|---|---|---|
| 1 — Data foundation | complete | A-D | profile, seniority audit, SWE audit, comparability, concentration, feasibility | T01-T07, T30 |
| 1.5 — Preprocessing | complete | Prep | LLM-cleaned text, embeddings, tech matrix, calibration | shared artifacts |
| 2 — Structural discovery | complete | E-I | archetypes, requirements complexity, linguistic evolution, tech ecosystem, semantic similarity | T08-T15 |
| V1 — Gate 2 verification | complete | V1 | adversarial re-derivation of 6 Gate 2 claims | 5 verified, 1 corrected |
| 3 — Market dynamics | complete | J-M, O | company strategies, geography, DiD, temporal, boundaries, senior evolution, ghost forensics, employer-usage divergence, archetype stratification, LLM-authorship | T16-T23, T28-T29 |
| V2 — Gate 3 verification | complete | V2 | adversarial re-derivation of 8 Gate 3 claims | 5 verified, 2 flagged, 1 corrected |
| 4 — Synthesis | complete | N | new hypotheses, interview artifacts, synthesis document | T24-T26 |
| 5 — Presentation | complete | P | mkdocs-material site, MARP deck | this site |

## Gate discipline

Every wave's outputs are read by a research-director agent who writes a gate memo before the next wave dispatches. The memo:

- Summarizes confirmed, contradicted, and new findings.
- Flags surprises that contradict pre-exploration priors.
- Lists corrections that propagate forward.
- Sets the next wave's dispatch parameters.

This prevents agents from chaining on unverified findings from earlier waves.

## Verification rounds

**V1 (Gate 2 adversarial).** Re-derived six Wave 2 findings:

1. AI-mention pattern semantic precision (50-sample) — passed; refined (dropped `agent`, `mcp`).
2. Requirements-section shrink — passed.
3. `requirement_breadth` length correlation — corrected to 71% content / 29% length.
4. NMI domain/period/seniority — passed.
5. Within-LLM-frame J2 flip — passed.
6. Relabeling cosines (seniors changed more than juniors) — passed; "180× period vs seniority" claim corrected to ~1.2-1.5×.

**V2 (Gate 3 adversarial).** Re-derived eight Wave 3 findings. Verified 5; flagged 2 for method-sensitivity (T29 mentor retention, T28 broad-vs-strict attribution); corrected 1 (T28 ≥+10pp is broad-AI, not strict-AI).

## Non-negotiable ablations (pre-committed at Gate 0)

Locked in before any agent read the parquet:

1. Aggregator exclusion (sensitivity).
2. Entry-specialist exclusion (240 companies per T06).
3. Cap-50 per company per period (prevents single-company dominance).
4. Labeled-only LLM-frame vs full-corpus dual reporting.
5. Arshkon-only AND pooled-2024 baselines for any senior-side claim.
6. Length-residualization for any composite with component r > 0.3 against length.
7. Strict AI pattern primary; broad pattern as sensitivity.
8. Aggregator, specialist, cap-50 sensitivities on every within-company decomposition.
9. T30 seniority-definition panel reported on every seniority-stratified headline.
10. T29 authorship-subset sensitivity on every AI-content claim.

## Method sub-pages

- [Preprocessing pipeline](preprocessing.md) — ten stages, LLM prompts, coverage caveats.
- [Sensitivity framework](sensitivity-framework.md) — T30 panel, 9 sensitivity dimensions, SNR rule, semantic precision, composite-score correlation.
- [Limitations](limitations.md) — known pipeline bugs, macro context, recruiter-LLM mediation.
</content>
</invoke>