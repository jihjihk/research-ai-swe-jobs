# Finding 3 — Senior roles specialized toward hands-on technical orchestration, not people management

**Co-headline. The cleanest within-corpus direct evidence of role redefinition in the whole exploration, and it is AI-linked.**

## Claim

Between 2024 and 2026, senior SWE roles specialized toward hands-on technical orchestration. Mid-senior technical-orchestration language rose **+98%**, director orchestration rose **+156%**, and the tech-lead sub-archetype doubled (**7.8% → 16.9%**), while director-level people-management density **fell 21%**. The AI × senior interaction is entirely in the orchestration profile, not people management. The mid-senior-to-director boundary blurred (the only boundary that did) because the `tech_count` coefficient flipped sign — 2026 directors mention *more* technologies than 2026 mid-senior postings, the opposite of the 2024 gradient.

## The numbers (T21)

Validated density per 1K chars, narrow object-noun-phrase patterns, 50-row stratified precision check per pattern, 100-row V2 audit at **100% precision**. Pattern bundle saved at `exploration/artifacts/shared/validated_mgmt_patterns.json`.

| Profile | Mid-senior 2024 → 2026 | Director 2024 → 2026 |
|---|---|---|
| People management (narrow, validated) | 0.186 → **0.232 (+25%)** | 0.228 → 0.181 (**−21%**) |
| **Technical orchestration** | 0.168 → **0.332 (+98%)** | 0.118 → **0.302 (+156%)** |
| Strategic scope | 0.045 → 0.053 (+17%) | 0.035 → 0.076 (+117%) |

**AI × senior interaction:**

- **2024:** AI-mentioning senior postings had identical profile density to non-AI.
- **2026:** AI-mentioning senior postings have **orchestration density 0.482 vs 0.274 non-AI (+76% uplift)**. V2 reproduces at +73%.
- People-management density is identical regardless of AI (0.230 vs 0.232, −1%).

**Sub-archetype clustering (k-means on z-scored densities):**

- **Tech-lead cluster: 7.8% → 16.9% (+9.1 pp)** — more than doubled.
- People-manager cluster: 14.5% → 14.7% (flat).
- Generic bucket: 70.2% → 59.8% (−10.4 pp).

**Cross-seniority pattern rules out a uniform-template explanation:**

- Entry people-management density: **−13%** (0.064 → 0.056).
- Mid-senior: **+25%**.
- Director: **−21%**.

Non-monotonic, so if LLM drafting tools were inserting mentoring language uniformly at every level, entry would have risen. It didn't.

## The director-recasting mechanism (T20)

- Mid-senior → director `tech_count` coefficient **flipped sign: −0.48 → +0.35**.
- Director `tech_count` rose **2.93 → 8.03 (+173%)** — the **largest per-cell shift** in the entire T20 feature-importance heatmap.
- Mid-senior ↔ director AUC **fell 0.677 → 0.616 (−0.061)**. Directors became structurally closer to mid-senior — because directors shed people-management and gained orchestration.

2024 directors had fewer tech mentions (people bosses); 2026 directors have more (tech orchestrators). This is the single largest feature-importance swing anywhere in the feature-importance work.

## Senior-title compression (T10/T21)

`senior` in raw titles **41.7% → 28.9% (−12.8 pp)**; `staff` rose only **+3.1 pp**. Within the 150-company overlap panel: mean senior_delta **−9.1 pp**, staff_delta **+1.6 pp**. **Staff absorbs only ~22% of the senior drop** — "staff is the new senior" is NOT supported. The classifier is stable (100% of senior- and staff-titled rows are Stage 5 rule matches; 99.5% classify as `seniority_final = 'mid-senior'` in both periods). Most-supported mechanism: **employer title-field template rewriting, likely under LLM drafting (T29).**

## Figures

![Density profile shift](../assets/figures/T21/density_profile_shift.png)
*T21 — density shift by seniority × profile. Mid-senior and director orchestration rose; director people-management fell.*

![AI × senior interaction](../assets/figures/T21/ai_senior_interaction.png)
*T21 — 2026 AI-mentioning senior postings have 76% higher orchestration density but identical people-management density vs non-AI.*

![Sub-archetype distribution](../assets/figures/T21/subarchetype_distribution.png)
*T21 — tech-lead sub-archetype doubled from 7.8% to 16.9%. People-manager flat.*

![Cross-seniority mgmt density](../assets/figures/T21/cross_seniority_mgmt.png)
*T21 — non-monotonic pattern rules out a uniform-template explanation.*

## Sensitivity checks this claim must survive

| Test | Requirement | Result |
|---|---|---|
| Pattern precision | ≥ 80% per pattern | 13/13 people + 14/14 orch + 9/9 strategic — PASS |
| 100-row audit | 100% precision | 100% — PASS (V2 Section 7) |
| Top-10 company exclusion | Direction and magnitude hold | +156% → **+120%** — PASS |
| Cross-seniority non-monotonic | Rules out uniform template | Entry −13%, mid-senior +25%, director −21% — PASS |
| Mentor style-correlation caveat | Cite caveat | Mentor sub-pattern is style-correlated (T29); orchestration is NOT — noted |

## Known reviewer attack surface

- **Director cells are thin** (99 rows in 2024, 112 in 2026). Report CIs; use T21 + T20 convergent evidence.
- **Mentor sub-pattern is style-correlated** (T29 authorship score r ≈ 0.09 with mentor density). Some fraction of the mid-senior people-management rise is template. The **orchestration rise is NOT style-correlated** (AI 0-7% attenuation).
- **V1 overturned T11's "manage −47%" finding.** The original T11 strict management pattern had 38-50% semantic precision. The T21 rebuilt pattern bundle is the only set of validated patterns for this work.

## Task citations

- **[T21 — Senior role evolution](../audit/reports/T21.md)** — primary. Pattern bundle, AI × senior interaction, sub-archetype clustering, senior-title compression.
- **[T20 — Seniority boundary clarity](../audit/reports/T20.md)** — the `tech_count` sign flip, feature-importance heatmap.
- **[T10 — Title taxonomy evolution](../audit/reports/T10.md)** — senior-in-title 41.7% → 28.9%.
- **[T29 — LLM authorship detection](../audit/reports/T29.md)** — mentor style correlation caveat.
- **[V2 verification (Gate 3)](../audit/verifications/V2_verification.md)** — Section 7 precision audit, Alt 2 top-10 exclusion.

## What this finding does NOT say

- It does NOT say "general management rose" — that T11 claim was overturned by V1. The narrow surviving claim is that people-management rose specifically at mid-senior, not as a corpus-wide template shift.
- It does NOT say "staff is the new senior" — staff absorbs only ~22% of the senior drop.
- It does NOT generalize to all senior roles — it is localized to the AI × senior interaction cell and the tech-lead sub-archetype.
- It does NOT say the director recasting is a fully confirmed real market shift — director cells are thin, and whether this is a real role change vs a template-rewriting artifact is open for the analysis phase.
