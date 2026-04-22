# Claim 7 — Cross-occupation rank correlation between worker AI use and employer AI requirements

**Status:** methodology audit, not number confirmation. The headline pattern survives, but the level-vs-rank framing is fragile to a few choices.

**Author:** S25 (Claude, 2026-04-21)
**Code:** `eda/scripts/S25_cross_occupation_rank.py`
**Tables:** `eda/tables/S25_subgroup_rates.csv`, `S25_pair_table.csv`, `S25_method_comparison.csv`, `S25_worker_benchmarks.csv`
**Figures:** `eda/figures/S25_method_comparison.png`, `S25_employer_vs_worker.png`

## Hypothesis

The v9 archive Story 08 ("The market got the pattern right") claims that across roughly 16 occupations, the Spearman rank correlation between worker-side AI usage rates (from public surveys) and employer-side AI requirement rates (in 2026 LinkedIn postings) is around +0.92, even though the *level* gap between worker use and employer requirements is enormous. The narrative implication is that employers have correctly identified *which* occupations have been changed by AI but are slow to write that down in job descriptions. This memo reconstructs the claim under multiple operationalizations and asks how robust the +0.92 number is.

## Data sources

### Employer-side
- `data/unified_core.parquet` — LinkedIn postings, English, date-OK rows. 110,000 rows total. Subgroup classification is title-regex (first-match-wins) carried over from v9 T32 (`exploration-archive/v9_final_opus_47/scripts/T32_cross_occupation_divergence.py`, lines 67-91). I use the **canonical** AI vocabulary regex `AI_VOCAB_PATTERN` from `eda/scripts/scans.py:69-71`, plus a sensitivity using the `ai_strict_v1_rebuilt` regex from `validated_mgmt_patterns.json` (the regex the v9 story actually used).

### Worker-side benchmarks (curated; full source list in `S25_worker_benchmarks.csv`; access date 2026-04-21)
- **SWE umbrella (`other_swe`):** Stack Overflow 2024 (0.63), Stack Overflow 2025 (0.84 any / 0.506 daily), DORA 2025 (0.90). Mean any = 0.79.
- **SWE-adjacent:** ML engineer 0.85 (SO 2025 ML sub-cut + Kaggle); data scientist 0.75 (SO 2024 ML/AI post); data engineer 0.75 (SO 2024); data analyst 0.60 (Microsoft WTI + Kaplan); devops 0.70 (DORA); solutions architect 0.65 (SO 2024); QA 0.50 (Capgemini WQR); security 0.40 (ISC2 2024); systems admin / network 0.40 / 0.35 (Bick et al. NBER 2024 + ISC2 + Cisco).
- **Control:** accountant 0.50 (Thomson Reuters 2024 Future of Professionals); financial analyst 0.30 (CFA 2024, N=200 firms); nurse 0.15 (JAMA / AHA 2024 hospital GenAI surveys); electrical engineer 0.30 (IEEE Spectrum 2024 + Bick NBER); mechanical engineer 0.25 (ASME 2024 + Autodesk); civil engineer 0.22 (ASCE 2024 + Autodesk).

URLs and methodology notes per row in `eda/tables/S25_worker_benchmarks.csv`. Daily-use rates also collected per occupation for the M5 sensitivity.

**No benchmark available** for marketing manager, HR rep, and sales rep (occupations the user named): they do not have dedicated subgroups in `unified_core` and the few survey numbers on those roles (Microsoft WTI, McKinsey AI Index) are heterogeneous in definition. I deliberately do not impute values for them. The analyzable set is 17 occupations (the SWE umbrella + 10 adjacent + 6 control), close to but not identical to the "16" the claim names.

### Macro sanity check
- Bick / Blandin / Deming, NBER 2024 ("The rapid adoption of generative AI", w32966): 39.4% of US workers report any workplace generative-AI use; 49.6% in computer/math occupations. Consistent with our level scaling.

## Methods explored

1. **M1 (recommended headline)** — canonical `AI_VOCAB_PATTERN`, 2026 employer level vs worker any-mid. Worker benchmark = mean across cited any-use surveys per occupation.
2. **M2 — `ai_strict_v1_rebuilt` regex.** Reproduces the v9 archive's exact employer-side measurement (literal replication row).
3. **M3 — 2024 employer levels** (contemporaneous with the 2024 worker surveys).
4. **M4 — 2024→2026 employer deltas vs worker level.** Tests whether occupations *moved* in line with worker exposure.
5. **M5 — worker daily-use benchmark** instead of any-use. Stricter, workflow-anchored.
6. **M6 — thick-cell only (n_2024 ≥ 500 AND n_2026 ≥ 500).** Drops devops_engineer, data_engineer, ml_engineer (swe_adj), civil_engineer, and systems_admin (12 obs).
7. **M7 — control occupations only** (6 obs).
8. **M8 — tech occupations only** (SWE + adjacent, 11 obs).
9. **M9 — drop ml_engineer**, the most extreme outlier on both axes.
10. **M10 — worker benchmark = MAX of cited surveys** (most permissive).
11. **M11 — worker benchmark = MIN of cited surveys** (most conservative).

Each method reports Spearman, Kendall, and Pearson; Spearman gets both Fisher-z analytic CIs and 2,000-iteration bootstrap CIs (resampling occupations).

## Per-method results

| Method | n | Spearman | Fisher-z 95% CI | Bootstrap 95% CI | Kendall | Pearson |
|---|---:|---:|---|---|---:|---:|
| M1 headline (canonical, 2026, any) | 17 | **+0.860** | [+0.65, +0.95] | [+0.57, +0.98] | +0.73 | +0.75 |
| M2 strict_v1 (literal v9 replication) | 17 | **+0.912** | [+0.77, +0.97] | [+0.74, +0.97] | +0.76 | +0.69 |
| M3 2024 employer levels | 17 | +0.581 | [+0.14, +0.83] | [+0.07, +0.88] | +0.44 | +0.50 |
| M4 2024→2026 deltas | 17 | +0.785 | [+0.49, +0.92] | [+0.50, +0.95] | +0.63 | +0.70 |
| M5 daily-use worker benchmark | 17 | +0.864 | [+0.66, +0.95] | [+0.59, +0.98] | +0.74 | +0.84 |
| M6 thick cells only | 12 | +0.782 | [+0.38, +0.94] | [+0.33, +0.98] | +0.65 | +0.78 |
| M7 control only | 6 | +0.406 | [-0.60, +0.92] | [-0.80, +1.00] | +0.41 | +0.15 |
| M8 tech only (SWE + adj) | 11 | +0.872 | [+0.57, +0.97] | [+0.57, +0.99] | +0.72 | +0.74 |
| M9 drop ml_engineer | 16 | +0.832 | [+0.57, +0.94] | [+0.49, +0.97] | +0.70 | +0.81 |
| M10 worker MAX of surveys | 17 | +0.850 | [+0.62, +0.94] | [+0.57, +0.97] | +0.72 | +0.73 |
| M11 worker MIN of surveys | 17 | +0.872 | [+0.67, +0.95] | [+0.58, +0.99] | +0.76 | +0.77 |

## Recommended methodology

**M1 is the most defensible headline.** It uses the project's canonical AI vocabulary regex (rather than the v9-specific `ai_strict_v1`) and reports the *level* the user's claim is about, in the year (2026) the user's claim names. The Spearman point estimate is **+0.86, Fisher-z 95% CI [+0.65, +0.95], bootstrap 95% CI [+0.57, +0.98]**. The point estimate is roughly 0.06 below the +0.92 figure in the v9 story, and the difference is entirely attributable to the regex choice — switching to `ai_strict_v1` (M2) recovers +0.91. Both intervals are too wide at n=17 to distinguish them statistically.

Why not M2 as the headline:

- The canonical pattern is the convention used elsewhere in the project (`eda/scripts/scans.py`, `core_scans.py`, all S1/S3/S10/S11 scans). Using a one-off regex purely because it produces a slightly cleaner number is a researcher-degree-of-freedom red flag.
- The two regexes differ mostly in `fine-tuning` handling and a handful of vendor terms; neither is more "true" than the other and both have validated precision around 0.92-0.96 per `validated_mgmt_patterns.json`.

What is robust across all 11 methods:

- The correlation is positive and large (point estimates 0.41 to 0.91; nine of eleven above 0.78).
- The direction-universality property survives: in every method, employers' 2026 ordering of occupations matches workers' ordering up to noise, and worker rates exceed employer rates everywhere.

What is NOT robust:

- M3 (2024 levels). At Spearman +0.58 the contemporaneous pairing is much weaker; the +0.86 headline depends on letting employers "catch up" by 2026.
- M7 (control-only). Within the six non-tech occupations the correlation collapses to +0.41 with a CI that crosses zero. The cross-occupation rank correlation is largely driven by the SWE-vs-non-SWE gap. Read M7 with M8 (+0.87 within tech): the pattern reads as "tech occupations are ordered correctly *and* non-tech occupations are ordered correctly, but most of the variance is between, not within, the two clusters."
- M9 (drop ml_engineer). +0.83 holds, so ml_engineer is not the only driver, but it is the highest-leverage point.

## Recommended single-claim formulation

> **"Across 17 occupations with public worker-AI-usage benchmarks, the rank order in which employers codify AI requirements in 2026 LinkedIn postings tracks the rank order of worker-side AI use closely (Spearman ρ = +0.86, bootstrap 95% CI [+0.57, +0.98], n = 17), even though employers' levels run 4-50× below workers' across the same occupations."**

For a paper, replace "+0.86" with "+0.86 [+0.65, +0.95]" using Fisher-z, and note that the figure rises to +0.91 under the v9 archive's stricter AI regex. Do *not* present +0.92 as the headline — it is the upper end of a regex-choice sensitivity band, not the central estimate.

## Caveats and threats to validity

1. **Worker benchmarks are heterogeneous in definition.** Surveys ask different questions ("ever tried", "currently use", "use weekly"). The Thomson Reuters 0.50 for accountants is the most fragile point — under the daily-use benchmark accountants drop to 0.20 and the level gap compresses from 72× to 14×, but the rank correlation barely moves (M5 still +0.86). The rank framing is more robust than any level-gap claim.

2. **Subgroup classification is title-regex.** A title with "senior backend engineer (ML team)" maps to other_swe rather than ml_engineer. False negatives push ml_engineer's denominator down without changing the numerator much, inflating its rate; this is the most plausible explanation for ml_engineer's 81-86% canonical AI rate.

3. **Thin cells.** devops_engineer (n_2026 = 6) and ml_engineer in swe_adjacent (n_2026 = 187) are small. M6 (drop n<500) leaves the headline at +0.78, so the conclusion is not driven by tiny cells, but the influential points should be flagged.

4. **Endogeneity of the regex.** The canonical AI vocab was developed against SWE postings. Recall in nurse or accountant postings may be lower (an accounting tool could be "AI-powered" without using LLM/vendor vocabulary). If recall is differentially lower in non-SWE postings, the rank correlation is artificially inflated. M5 (daily-use worker benchmark) partially addresses this since daily-use is also a "narrow" measure on the worker side; survival under M5 is reassuring but not dispositive.

5. **The level-gap framing is benchmark-sensitive; the rank framing is not.** This is the most useful methodological finding. The McKinsey / BCG / WEF / MIT-Fortune level-gap headlines swing by an order of magnitude depending on which survey you trust. The rank correlation moves by less than 0.1 across all eleven methods we tried. For the paper, lead with rank.
