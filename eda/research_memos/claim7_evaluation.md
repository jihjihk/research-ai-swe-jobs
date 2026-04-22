# Claim 7 evaluation — adversarial audit

**Status:** REVISE. The +0.86 headline survives narrow falsification tests but the framing in the previous memo overstates the within-cluster signal. A reformulated claim is publishable; the original is not.

**Author:** S25 evaluation pass (Claude Opus 4.7, 2026-04-21)
**Inputs:** `eda/research_memos/claim7_cross_occupation_rank.md`, `eda/scripts/S25_cross_occupation_rank.py`, `eda/tables/S25_pair_table.csv`, `eda/tables/S25_method_comparison.csv`
**New code:** `eda/scripts/S25_eval_claim7.py`
**New tables:** `S25_eval_permutation.csv`, `S25_eval_jackknife.csv`, `S25_eval_timestability.csv`, `S25_eval_extra_benchmarks.csv`, `S25_eval_blended.csv`
**New figures:** `S25_eval_permutation.png`, `S25_eval_jackknife.png`

---

## 1. Null-model comparison (n = 17)

Two nulls were run, each at 10,000 permutations (`S25_eval_claim7.py:46-72`).

| Null | What it preserves | Median ρ | 97.5%-ile ρ | p (right-tail vs ρ_obs = +0.860) |
|---|---|---:|---:|---:|
| Uniform shuffle | nothing | +0.00 | +0.49 | < 0.0001 |
| Two-cluster shuffle | tech-vs-non-tech level gap; randomizes within each cluster | **+0.56** | **+0.79** | **0.0044** |

The uniform null is irrelevant: of course +0.86 looks impressive against random pairings. The **two-cluster null is the actual question**: if you knew nothing more than "this occupation is tech vs non-tech" and randomly paired worker rates within each cluster, you would get a Spearman of ≈+0.56 just from the between-cluster gap. The observed +0.86 sits at the 99.6th percentile of that null — distinguishable, but not by orders of magnitude. **Roughly 65% of the +0.86 signal is the tech-vs-non-tech split; the remaining 25 points of correlation come from genuine within-cluster ordering.** That residual is real, but it is much smaller than the v9 archive's framing implied.

## 2. Within-cluster meaningfulness — jackknife

Per-occupation leave-one-out leverage in `S25_eval_jackknife.csv`; figure `S25_eval_jackknife.png`.

- **Full set (n=17):** ρ ranges +0.83 to +0.93 across all leave-one-out fits. No single occupation is load-bearing. Dropping nurse or ml_engineer pulls ρ down by ≈0.03; dropping accountant pushes it up by +0.07 (accountant is the worst-fit point).
- **Tech-only (n=11):** ρ ranges +0.83 to +0.91. Stable; ml_engineer is the highest-leverage point but the rest still produce ρ ≈ +0.83 without it.
- **Control-only (n=6):** ρ ranges **−0.05 to +0.97**. *Two single occupations control the entire result.* Drop accountant and ρ jumps to +0.97 (because accountant is the only "high-worker, low-employer" outlier among controls — Thomson Reuters reports 50% any-use but the accountant subgroup posts only 1.4% AI-mentioning postings). Drop nurse and ρ collapses to −0.05 (nurse anchors the low end on both axes — when it is gone there is no signal left).

**Verdict on M7:** the control set has effectively two informative points (nurse at the bottom, accountant as a contrarian) and four near-tied engineering occupations. The v9 story's "employers got the pattern right" claim has no traction below the tech/non-tech threshold.

## 3. Additional benchmarks searched

Checked Pew, Anthropic Economic Index, Microsoft WTI, Goldman Sachs, Indeed Hiring Lab, ADP, and Bick/Blandin/Deming. Curated additions in `S25_eval_extra_benchmarks.csv`:

- **Pew Research Sept 2025**: 21% of US workers use AI for some work, 28% among bachelor's+. Education split, not occupation-level. Corroboration anchor for `other_swe`.
- **Bick/Blandin/Deming NBER 2024 (w32966)**: computer/math 49.6% workplace any-use vs 23% all-occupation. Second `other_swe` anchor.
- **Anthropic Economic Index Mar 2026**: 35% of Claude.ai conversations are computer/math tasks. *Not* a worker-use rate (usage-share conditional on being a Claude user). Context only; not blended.
- **ADP People at Work 2025**: cross-occupation 43% frequent use, 20% near-daily. Macro anchor only.
- **Indeed Hiring Lab Sep 2025**: 45% of data & analytics postings, 15% marketing, 9% HR mention AI. *Employer*-side, so it cannot enter as a worker benchmark — but it independently corroborates the tech/non-tech ordering.
- **Microsoft WTI 2025**, **Goldman Sachs**: leadership-survey or theoretical-exposure metrics; no clean occupation-level worker-use rates.

Blending the comparable additions (Pew, Bick) into the worker-side mean: Spearman **+0.872, Fisher-z 95% CI [+0.67, +0.95], n=17** — essentially identical to the +0.860 headline. Tech-only rises to +0.918; control-only is unchanged. See `S25_eval_blended.csv`.

## 4. Time-stability (`S25_eval_timestability.csv`)

Worker benchmarks are 2024-vintage. The user's claim names 2026 employer levels.

| y-axis | Full ρ | Tech-only ρ | Control-only ρ |
|---|---:|---:|---:|
| 2024 employer levels (canonical) | +0.581 | +0.712 | +0.174 |
| 2026 employer levels (canonical) | **+0.860** | +0.872 | +0.406 |
| 2024→2026 delta (canonical) | +0.785 | +0.603 | +0.406 |
| 2024 employer levels (strict_v1) | +0.458 | +0.618 | −0.123 |
| 2026 employer levels (strict_v1) | +0.912 | +0.831 | +0.609 |
| 2024→2026 delta (strict_v1) | +0.898 | +0.785 | +0.609 |

Two important time-stability results:

- The rank correlation **is mostly a 2026 phenomenon**. In 2024, employers' AI-language ordering is only weakly aligned with worker use (full ρ = +0.58, tech-only +0.71). The +0.86 is the result of two years of differential rewriting.
- The **2024→2026 delta correlation is +0.78** under canonical regex and +0.90 under strict-v1: occupations that workers were already using AI in 2024 are the ones whose postings have been rewritten to mention AI by 2026. This is, methodologically, the *more interesting* claim — it ties the worker-side measure to a *change* on the employer side, which is harder to explain via static priors.

## 5. Comparable published rank correlations

- **Yale Budget Lab cross-metric AI-exposure** ([source](https://budgetlab.yale.edu/research/labor-market-ai-exposure-what-do-we-know)): cross-metric correlations among Felten-Raj-Seamans AIOE, Webb, Eloundou GPTs-are-GPTs etc. cluster in the **+0.7-0.9 band**. Our +0.86 sits at the high end, but with a different pairing: theoretical-ish worker-use vs realized employer-language. Cross-domain alignment should be *lower* than within-domain alignment, so +0.86 is informative.
- **Anthropic Labor-Market-Impacts** ([source](https://www.anthropic.com/research/labor-market-impacts)): "Spearman correlation of job exposure across resolutions is exceedingly high" — sets the ceiling.
- **GAO 2020 industry-tech-employment**: Pearson 0.30, Spearman 0.23 across 69 industries — a baseline for "weak alignment between two adjacent labor-market constructs."

**Anchor verdict:** +0.86 is impressive at face value but only modestly above the +0.7-0.9 published band of cross-measure AI-exposure correlations. Not boilerplate, not extraordinary. The two-cluster null result (§1) is the honest place to land.

## 6. Article framing test — Economist-style lede

**Cannot publish (the v9 implication):** "Across 16 occupations, employers have correctly identified which jobs AI is rewiring. Spearman ρ = +0.92 — almost perfect alignment between what workers do and what employers ask for." Defensible only under a regex the rest of the project does not use; would not survive a permutation test.

**Can publish (≈180 words):** Software engineers are four times more likely to face AI requirements in a job posting today than they were two years ago, and the change is not random. Across seventeen occupations with comparable worker surveys — from machine-learning engineers and devops engineers down through civil engineers, accountants, and registered nurses — the rank order in which 2026 LinkedIn postings codify AI tooling tracks the rank order of worker self-reported AI use closely (Spearman ρ = +0.86; bootstrap 95% CI +0.57 to +0.98). Employers' levels still run four to fifty times below workers'; they have not caught up to the workflow change, but the *ordering* is right. Most of that alignment is the obvious tech-versus-non-tech split: a permutation that preserves only that difference recovers ρ ≈ +0.56 by itself. The interesting residual is *motion*: occupations whose workers were already using AI in 2024 are the ones whose 2026 postings have been rewritten to require it (ρ on the change = +0.78). Within healthcare, finance, and the older engineering disciplines the ordering is essentially noise.

## Verdict

**REVISE.**

Why not KEEP: the original memo's recommended formulation says the rank order tracks "across 17 occupations." That is true on the page but misleading in interpretation: most of the variance is between two clusters, the within-control rank correlation is two outliers wide, and a single regex switch moves the headline from +0.86 to +0.92. A referee or sharp editor will catch the two-cluster issue.

Why not DROP: the +0.86 *is* distinguishable from the two-cluster null at p ≈ 0.004; the 2024→2026 delta correlation (+0.78) is a stronger and more honest causal-flavored claim than the level correlation; and the tech-only result (+0.87, jackknife-stable) is robust on its own.

**Recommended reformulation (use this in the paper):**

> Across seventeen occupations with comparable worker-AI-use benchmarks, the rank order in which 2026 LinkedIn postings codify AI tooling tracks the rank order of 2024 worker AI use (Spearman ρ = +0.86, bootstrap 95% CI [+0.57, +0.98], n = 17). Roughly two-thirds of that alignment is the tech-versus-non-tech gap (a within-cluster permutation null returns ρ ≈ +0.56); the residual reflects genuine within-tech ordering (tech-only ρ = +0.87, n = 11) but **not** within-control ordering (control-only ρ = +0.41 with CI crossing zero, n = 6). The stronger claim is on motion rather than levels: occupations whose workers were already using AI in 2024 are the ones whose postings have been rewritten by 2026 (Spearman ρ on the 2024→2026 employer delta = +0.78, n = 17).

Lead the paper with the *delta* result, support it with the *level* result, and explicitly disclaim the within-control null. Drop "+0.92" from any headline — it is a regex choice, not a finding.
