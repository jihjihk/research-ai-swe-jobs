# Executive summary

Between 2024 and 2026, software-engineering job postings underwent a real, same-firm, same-title content rewriting concentrated on AI tooling and platform-infrastructure orchestration. The rewriting is SWE-specific in magnitude, within-firm, and independent of recruiter-LLM authorship style, sampling-frame shifts, and hiring-market selectivity.

## One-paragraph version

SWE DiD against control occupations is +13.1 to +14.0 pp on AI-mention, 95% CI [+12.76, +14.37], while control drift is +0.17 pp. Same-company same-title pair-level AI drift is +10 to +13 pp, exceeding the company-level average of +7.7 to +8.3 pp — the rewriting is within-title, not title-recomposition. The employer-worker AI-requirement divergence is universal across 16 of 16 tested occupation subgroups, in both 2024 and 2026, with Spearman +0.92 rank concordance between worker adoption and employer codification. Seniority boundaries sharpened rather than blurred (TF-IDF 0.946 to 0.863; AUC +0.150 associate to mid-senior). Scope inflation is universal and senior-greater-than-junior within-domain. Two senior archetypes emerged: Applied-AI/LLM Engineer (15.6x growth, 94% in 2026, median YOE 6) and Data-Platform/DevOps Engineer (2.6x growth, +11.4 pp share).

## Three alternative explanations, all rejected

- **LLM-authorship mediation** — content effects persist at 80 to 130% on the low-LLM-quartile subset; only length growth is ~52% LLM-mediated (T29).
- **Hiring-bar lowering** — |ρ| ≤ 0.28 on all hiring-bar proxies; 0 of 50 sampled postings contain explicit loosening language (T33).
- **Hiring-selectivity during the JOLTS trough** — |r| < 0.11 on all content metrics; desc-length r = +0.20 is the opposite direction (T38).

## Three methods contributions

- **T30 multi-operationalization seniority panel** — 13 seniority definitions cross-tabbed against 6 analysis groups; 12 of 13 direction-consistent; junior-side 7 of 7 up, senior-side 5 of 6 down.
- **Within-2024 SNR calibration framework** — per-metric noise floors separate clean signals (AI-strict ratio 32.9; scope ratio 42.8) from near-noise seniority shares.
- **Pattern-validation methodology** — V1 exposed 0.28 precision on a widely-used broad management pattern and 0.55 on the strict version; T22 built and validated 7 primary patterns at 0.92+ precision.

## Corrections applied

- **Management language is flat, not falling.** The "management declining" claim in T11 came from a 0.28-precision pattern; under the V1-validated pattern, senior mgmt density is 0.039 to 0.038 (SNR 0.1). V2 independently replicated this flat direction.
- **T13 requirements-section shrink** is classifier-sensitive (direction flips under a simpler regex) and authorship-sensitive (near-disappears on the low-LLM subset). Demoted to a flagged finding.

## Data scope

- 68,137 SWE LinkedIn rows under `is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok'`.
- Three temporal snapshots: arshkon 2024-04 (4,691 rows), asaniczka 2024-01 (18,129), scraped 2026-03 onwards (45,317 and growing).
- Cross-source span about 791 days, covering roughly eight major frontier-model releases.

## Full synthesis

The 850-line SYNTHESIS.md that produced this summary is in [the Audit trail](evidence/synthesis.md).
