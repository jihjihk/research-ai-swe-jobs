# Finding 3 — Senior role shift is senior-specific

## Claim

**Mid-senior mentor-binary rate rose 1.46-1.73× from 2024 to 2026 while entry rose only 1.07×; orchestration and strategic-scope densities approximately doubled at mid-senior; an emergent "management + orchestration + strategic + AI" sub-archetype (n=860, 97% of its members are 2026) now occupies 13% of the senior sample. The shift is senior-specific, not corpus-wide template expansion.**

## Core numbers

| Metric | Entry | Associate | Mid-senior | Director | Source |
|---|---:|---:|---:|---:|---|
| Mentor-binary ratio 2024 → 2026 | 1.07× | 0.61× | **1.46-1.73×** | 1.30× | T21 |
| Management-strict Δ (pp) | +1 | +4 | +14 | +9 | T21 |
| Orchestration-strict Δ (pp) | — | — | +21 | +28 | T21 |
| Strategic-strict Δ (pp) | — | — | +6 | +16 | T21 |
| AI-strict Δ (pp) | +9 | +11 | +14 | +23 | T21 |

Senior sub-archetype k-means clustering (k=4, standardized features):

| Cluster | n | % 2026 | AI | Orch | Mgmt | Mentor |
|---|---:|---:|---:|---:|---:|---:|
| baseline_ic | 4,795 | 70 % | — | — | — | — |
| mgmt_orch | 1,523 | 78 % | 0.00 | — | — | 0.96 |
| strat_scope | 1,052 | 76 % | — | — | — | — |
| **mgmt_orch_strat_ai** | **860** | **97 %** | **1.00** | **0.85** | 0.25 | 0.39 |

Staff-title share doubled: 2.6% → 6.3%. Staff-titled seniors carry +16pp mentor, +16pp management, +11pp strategic, +5pp AI vs non-staff seniors.

## Key figures

![Cross-seniority mentor rates](../figures/T21/T21_cross_seniority_mentor.png)
*T21 — Mentor-binary rate by seniority, 2024 vs 2026. Mid-senior bar pair widens; entry barely moves.*

![Emergent sub-archetypes](../figures/T21/T21_subarchetypes.png)
*T21 — Four-cluster k-means on standardized management/orchestration/strategic/AI density features. Cluster 4 (mgmt+orch+strat+AI) is 97% 2026 — a new senior role type.*

![Domain-stratified deltas](../figures/T21/T21_domain_deltas.png)
*T21 — Senior mentor Δ by archetype. ML/AI concentrated for AI and orchestration; frontend concentrated for management and mentorship; `systems_engineering` near-zero for AI (our control).*

## V1-refined semantic-precision patterns

The initial broad patterns failed semantic precision at 80% floor. V1 refined to:

- **people_management_strict** = `mentor|coach|hire|headcount|performance_review`
  - Dropped: bare `manage` (14% precision — "manage data", "manage state"), `stakeholder` (42%), `team_building` (10%).
- **technical_orchestration_strict** = `architecture review|code review|system design|technical direction|ai orchestration|workflow|pipeline|automation|evaluate|validate|quality gate|guardrails|prompt engineering|tool selection`
- **strategic_scope_strict** = `business impact|revenue|product strategy|roadmap|prioritization|resource allocation|budgeting|cross-functional alignment`

## What it rules out

- **Corpus-wide templating.** If the shift were template-wide, entry mentor rate would rise at the same ratio as mid-senior. It didn't (1.07× vs 1.46-1.73×).
- **Traditional "shift to people management."** Bare `manage` has 14% semantic precision in SWE JDs. The driving term is `mentor`, not `manage`. This is team-multiplier senior-IC work, not a ladder promotion.
- **"AI replaced senior work."** The emergent cluster with 100% AI coverage has 85% orchestration density and 39% mentor density. AI is bundled alongside human team multiplication, not substituting it.

## Sensitivities applied

- **Pattern precision** validated on 20-sample for every strict term; 100% precision on `mentor`.
- **Arshkon-only baseline** primary for senior-side (asaniczka has 0 native entry labels, inflating pooled-2024 senior share by 19pp).
- **Authorship-score bottom-40%** — mentor retention 72% under T29's primary score; 105% under V2's 3-feature score. Method-sensitive; paper cites "0-30% mediated" with explicit uncertainty.
- **T28 archetype stratification** — 20/22 archetypes show positive senior mentor Δ; 8 show Δ ≥ +10pp.

## Limitations

- **Magnitude is method-dependent.** T21 says 1.73×; V2 re-derivation says 1.46×. Direction verified; paper cites range not point.
- **Mentor-rate attenuation is the most LLM-sensitive of all findings.** T29 primary cut retains 72%; V2 3-feature retains 105%. Frame as 0-30% mediated.
- **Associate cell is small** (n=39-51 per period). Directional only; no precise ratio citable.

## Links to raw

- [T11 — management-pattern refinement](../raw/wave-2/T11.md)
- [T21 — senior role evolution](../raw/wave-3/T21.md)
- [T28 — archetype stratification](../raw/wave-3/T28.md)
- [T29 — LLM-authorship mediation](../raw/wave-3/T29.md)
- [V1 verification](../raw/verification/V1_verification.md) — pattern precision
- [V2 verification](../raw/verification/V2_verification.md) — mentor magnitude
</content>
</invoke>