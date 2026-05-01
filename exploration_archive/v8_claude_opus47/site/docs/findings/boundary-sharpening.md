# Finding 5 — Seniority boundaries sharpened, not blurred

## Claim

**All three adjacent seniority boundaries gained AUC from 2024 to 2026 on a fixed 7-feature L2 logistic with 5-fold cross-validation. On a yoe-excluded junior-vs-senior panel, the AUC gained +0.134 — language-based seniority distinction became markedly sharper, not fuzzier, during the AI rewrite.**

## Core numbers

| Boundary | 2024 AUC | 2026 AUC | Δ | Source |
|---|---:|---:|---:|---|
| Associate vs entry | 0.621 | 0.675 | **+0.054** | T20 |
| Mid-senior vs associate | 0.701 | 0.785 | **+0.084** | T20 |
| Director vs mid-senior | 0.812 | 0.815 | +0.003 | T20 |
| J3 vs S4 (yoe-excluded panel) | 0.752 | **0.886** | **+0.134** | T20 |

V2 replicated the yoe-excluded +0.134 exactly. Per-domain boundary sharpening (mid-senior vs associate):

| Domain | 2024 AUC | 2026 AUC | Δ |
|---|---:|---:|---:|
| ML / AI | 0.598 | **0.703** | **+0.105** |
| Cloud / DevOps | 0.724 | 0.802 | +0.078 |
| Frontend | 0.715 | 0.781 | +0.066 |
| Backend | 0.698 | 0.759 | +0.061 |
| Data Eng | 0.681 | 0.712 | +0.031 |

ML/AI gained the most clarity — it was the worst-discriminated domain in 2024 and is now mid-pack in 2026.

## Key figures

![Per-boundary AUC change](../figures/T20/T20_boundary_auc.png)
*T20 — AUC by boundary and period. All three adjacent boundaries sharpened or held; yoe-excluded panel sharpened most (+0.134).*

![Boundary coefficient shifts](../figures/T20/T20_boundary_coefficients.png)
*T20 — Feature coefficients by boundary and period. Management density replaced tech-count as the #2 discriminator at the mid-senior-vs-associate boundary in 2026.*

![Domain-stratified boundary AUC](../figures/T20/T20_domain_boundary_auc.png)
*T20 — Per-domain per-boundary AUC. ML/AI moved from worst (0.598) to mid-pack (0.703).*

## What it rules out

- **"AI blurs junior/senior distinction."** The pre-exploration intuition that AI tool automation would narrow the skill gap between juniors and seniors is cleanly rejected. The gap widened.
- **"Seniority labels became noise."** If labels had degraded, AUC would fall. AUC rose. The 2026 labels distinguish senior from junior postings *better* than 2024 labels did.
- **"Corpus is becoming more uniform."** It is — slightly (T15 shows 2026 postings are more internally homogeneous within seniority level) — but the cross-seniority distance grew even faster.

## What happened at the mid-senior boundary

The model (L2 logistic, 7 standardized features) replaced one feature entirely:

- **2024 top discriminator:** technology count.
- **2026 top discriminator:** management density.
- Tech-count dropped from coefficient |0.48| to |0.29|. Management density rose from |0.22| to |0.42|.

Interpretation: in 2024 you could spot a mid-senior posting by the length of its tech list. In 2026 you spot it by its mentorship and orchestration language. The signal that carries seniority information shifted from stack breadth to team-multiplier content.

## Sensitivities applied

- **5-fold cross-validation** on every boundary × period cell.
- **No-aggregator sensitivity** — sharpens every boundary further (aggregator postings mix seniority).
- **Yoe-excluded panel** (J3 yoe ≤ 2 vs S4 yoe ≥ 5) — removes the possibility that AUC is driven by YOE extraction noise. Still +0.134.
- **Per-domain fits** (where n ≥ 50 per cell × period) — direction consistent across archetypes.

## Limitations

- **Associate cell** is small (n=39/51 per period). Associate-vs-entry boundary is directional only; no precise magnitude citable.
- **Feature set is fixed.** T20 uses a pre-committed 7-feature spec. Other feature sets could produce different AUC levels (not directions).
- **Director/mid-senior boundary** held flat (+0.003), not sharpened. Consistent with director postings already being highly distinguishable in 2024 (high baseline AUC).

## Links to raw

- [T20 — seniority boundary clarity](../raw/wave-3/T20.md)
- [V2 verification](../raw/verification/V2_verification.md) — yoe-panel replication
- [T21 — senior role evolution](../raw/wave-3/T21.md) — what the new senior content looks like
</content>
</invoke>