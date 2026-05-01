# SWE postings got rewritten — by the same firms, for the same titles

This is the evidence package for an exploration of U.S. software-engineering job postings from early 2024 through mid-2026. The headline: between those snapshots, the same firms rewrote the same job titles around AI tooling and platform-infrastructure orchestration, and the rewrite is specific to software engineering rather than a general labor-market drift.

## The three layers

This site answers three different questions, one per top-tab.

- **Overview** — what the exploration found, in a 23-slide deck and a short text summary.
- **Findings** — every major claim, its supporting evidence, its sensitivity verdict, and a link to the raw task report that produced it.
- **Methodology** — how we built the dataset, what LLM prompts produced which columns, what we can and cannot claim, and where each claim sits in the sensitivity framework.
- **Audit trail** — 34 task reports (T01 through T38), two adversarial verifications (V1, V2), four gate memos, and the full 850-line synthesis. Everything.

If you only have 90 seconds: open the slide deck on [Slide deck](presentation.md). If you have 10 minutes: read the [Executive summary](summary.md) and [A1](findings/a1-cross-occupation-divergence.md). If you want to audit: the [Audit trail](evidence/index.md) tab has everything.

## The six lead findings (Tier A)

|  | Finding | Magnitude | Verdict |
|---|---|---|---|
| A1 | Cross-occupation AI divergence is universal | 16/16 subgroups positive gap; SWE DiD +14 pp vs +0.17 pp control | strong, robust |
| A2 | Within-firm AI rewriting is same-title | pair-level +10 to +13 pp > company +7.7 to +8.3 pp | strong, range-report |
| A3 | Seniority boundaries sharpened, not blurred | TF-IDF 0.946 to 0.863; AUC +0.150 | strong, methods-convergent |
| A4 | Scope inflation universal, senior > junior | S4 +2.61 > J3 +1.58; 60-85% within-domain | strong |
| A5 | Two emergent senior archetypes | Applied-AI/LLM 15.6x; Data-Platform/DevOps 2.6x | strong |
| A6 | AI-DiD robust across control definitions | within 0.5 pp of +13-14 pp across 5 specs | strong |

## What we tested and rejected

- **Hiring-bar lowering** — REJECTED. Requirements-section contraction does not correlate with YOE, credential stack, tech count, or education asks (T33).
- **Legacy to AI substitution** — REJECTED. Disappearing 2024 titles map to 2026 neighbors at 3.6% AI rate, below the 14.4% market rate (T36).
- **Hiring-selectivity during the JOLTS trough** — REJECTED. Volume-up firms write longer JDs (T38).
- **LLM-authored JDs explain the AI rise** — REJECTED as dominant mediator. Content effects persist at 80 to 130% on the low-LLM-style subset (T29).
- **Sampling-frame artifact drives junior-share rise** — REJECTED. 13 of 15 headlines robust on the returning-firms cohort (T37).
- **Management language declining at senior tiers** — CORRECTED. Management was flat; the original claim was a measurement artifact from a 0.28-precision pattern (V1 + T21).

## The dataset

68,137 SWE LinkedIn rows under the default filter, drawn from three temporal snapshots:

- **arshkon** (Kaggle): 2024-04 snapshot, 4,691 rows.
- **asaniczka** (Kaggle): 2024-01 snapshot, 18,129 rows.
- **scraped** (LinkedIn, ongoing): 2026-03 onward, 45,317 rows.

The cross-source span is about 791 days (2.17 years) and encompasses roughly eight frontier-model releases.
