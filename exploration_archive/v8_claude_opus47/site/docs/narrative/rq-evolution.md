# Research question evolution

The pre-registered research questions were confirmed, inverted, or decomposed by the exploration. This page tracks the mapping from the original RQ set to the post-exploration set that the analysis phase will run.

## Original RQ set (from `docs/1-research-design.md`)

- **RQ1** — Employer-side restructuring: junior share / volume, scope inflation, senior role redefinition.
- **RQ2** — Task and requirement migration: which requirements moved between seniority levels.
- **RQ3** — Employer-requirement / worker-usage divergence: do posting AI requirements outpace actual usage?
- **RQ4** — Mechanisms: interview-based qualitative (reflexive thematic analysis).

## Post-exploration RQ set

| Original | Post-exploration successor | Verdict | Lead evidence |
|---|---|---|---|
| RQ1 | **RQ1a — SWE-specific AI-vocabulary rewriting** (LEAD) | Confirmed, reframed | T18 DiD 99%, T16 102% within |
| | **RQ1b — Within-company scope broadening** (supporting) | Confirmed | T16 +1.43 residualized, 76% of 240 cos |
| | **RQ1c — Senior archetype shift** (supporting) | Confirmed, senior-disproportionate | T21 mentor 1.46-1.73× mid-sr vs 1.07× entry |
| | **RQ1d — Seniority boundary sharpening** (NEW) | Confirmed | T20 +0.054/+0.084/+0.003; yoe-panel +0.134 |
| RQ2 | **RQ2 — Task/requirement migration** (retained, narrowed) | Diffuse cross-seniority, not migration | T11 breadth +34% vs +30%; T28 20/22 archetypes positive |
| RQ3 | **RQ3 — Employer under-specification of AI** | **Direction INVERTED** | T23 46.8% < 50/65/75/85% band |
| RQ4 | **RQ4 — Mechanisms, with quantitative priors** | Strengthened | T29 15-30% LLM-authorship mediation; T17 geographic null; T28 81% new-entrant |

## Key conceptual changes

- **"Junior scope inflation" → rejected as junior-specific.** Breadth rose equally at every seniority level (T11).
- **"Junior-senior convergence" → rejected.** Boundaries SHARPENED (T20).
- **"Anticipatory employer restructuring" → rejected.** Employers TRAIL workers (T23).
- **"Tech-giant intern pipeline expansion" → supporting methodological subplot** (T06, T16, T28). 81% of AI/ML-archetype growth is new-entrant-driven.
- **"Senior shift to management" → refined.** Senior shift is to **mentoring + orchestration + AI co-deployment**, not traditional people-management (T11 `manage` had 14% semantic precision; refined to `mentor|coach|hire|headcount|performance_review`).

## New findings not in any RQ

- **RQ1d — Seniority boundary sharpening** (T20). NEW.
- **RQ3 inversion as a standalone novelty** (T23). The direction flip is publishable on its own.
- **Systems_engineering as natural zero-AI control** (T28). +0.16 pp ai_strict; +7.6 pp senior mentor. Clean natural-experiment control for interviews.
- **Mgmt+orch+strat+AI sub-archetype** (T21). 97% 2026, n=860. Names a new senior role type for management-literature consumers.
- **Recruiter-LLM partial mediation** (T29). 15-30% of the rewrite is recruiter tooling. Establishes that the bulk remains real content change.

## Data quality verdict per RQ

| RQ | SAFE analyses | NEED CAVEATS | UNSAFE |
|---|---|---|---|
| RQ1a (AI rewriting) | DiD SWE vs control on ai_strict/tech_count/ai_tech_count; 102% within-co 240-co panel; 4-of-4 J panel AI rise | ai_broad 0.80 precision; rate citations must specify exact regex + subset + denominator | Conflating ai_broad and ai_strict |
| RQ1b (scope broadening) | Length-residualized breadth; composite-score correlation documented | Raw breadth; specialist+aggregator sensitivity mandatory | Raw requirement_breadth without length-residualization |
| RQ1c (senior archetype) | Mentor rate cross-seniority; V1-refined strict pattern; arshkon-only baseline | Mentor-rate magnitude method-dependent (1.46-1.73×) | Broad management including bare `manage` or `stakeholder` |
| RQ1d (boundary sharpening) | Per-boundary AUC, 5-fold CV; J3/S4 yoe-excluded panel (V2 exact) | Associate cell small — directional only | Per-archetype boundary fits with n<50 |
| RQ2 (narrowed) | Archetype-stratified Δ; within-company × within-archetype decomposition | Junior-senior gap claims require both baselines | "Juniors now look like 2024 seniors" claims — rejected |
| RQ3 (inverted) | Benchmark comparison 50/65/75/85%; V2 alternative framing | Worker-benchmarks self-reported, platform-biased, unit mismatch | Single-benchmark citation without 4-assumption sensitivity |
| RQ4 (mechanisms) | AI-strict attenuation 75-77%; new-entrant 81%; geographic null r=−0.11 | Mentor 72% vs 105% — method-sensitive, cite as 15-30% with uncertainty | Sharp citation of T29 retention % on mentor/breadth |

## Analysis-phase priorities

The analysis-phase pipeline should:

1. Run the RQ1a DiD framework on the full 63,701 SWE + 155,745 control corpus with bootstrap CI per metric.
2. Implement the T30 4-row ablation as a standard reporting unit.
3. Run the H_A cross-occupation extension (see [T24](../raw/wave-4/T24.md)): does the RQ3 inversion generalize beyond SWE?
4. Resolve mentor/breadth method-sensitivity with a labeled authorship-score calibration set.
5. Implement sampling-frame sensitivities (H_H) — restrict 2026 to returning-cos-only, recompute headlines.
</content>
</invoke>