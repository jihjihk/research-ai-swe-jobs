# Sensitivity framework

Every headline claim on this site was stress-tested against at least one of nine robustness dimensions before being accepted. This page documents the nine dimensions, then walks through the specific robustness tables that sit behind the main findings.

## The nine sensitivity dimensions

Each dimension is a way a finding could be a statistical artifact rather than a real pattern. A claim's headline status depends on whether the finding survives the dimensions that apply to it.

| # | What it tests | Typical check |
|---|---|---|
| **a** | Does the finding depend on which 2024 dataset is used? | Computed arshkon-only, asaniczka-only, and pooled; compared |
| **b** | Does the finding depend on staffing agencies and aggregators being in the corpus? | Computed with aggregators excluded |
| **c** | Does the finding depend on a small number of high-volume firms? | No firm allowed to contribute more than 20 or 50 rows |
| **d** | Is the 2024-to-2026 move larger than the within-2024 noise? | Signal-to-noise calibration (see below) |
| **e** | Is the finding just longer descriptions? | Metrics residualized on log-length before comparison |
| **f** | Does the finding depend on the specific within-firm panel cutoff? | Compared across three panels with different volume thresholds |
| **g** | Does the finding depend on which sentence classifier splits "requirements" from "narrative"? | Two classifiers run, both reported |
| **h** | Does the finding depend on the LLM-labeled subset? | Compared labeled-only against labeled-plus-unlabeled |
| **i** | Does the finding depend on which AI-mention regex is used? | Audited v1_rebuilt pattern compared against top-level |

## Signal-to-noise calibration (dimension d)

The study uses two 2024 datasets (arshkon and asaniczka). By comparing them against each other, we can measure how much any given metric naturally varies within a single year. This is the "noise floor". When we then compute the 2024-to-2026 change, we can ask whether the cross-period move is larger than the within-2024 noise. That ratio is the signal-to-noise ratio (SNR).

An SNR of 1 means the cross-period move is the same size as within-year noise, which is not meaningful. An SNR of 10 means the cross-period move is ten times larger than year-internal variation, which is clean signal.

| Metric | Within-2024 effect | Cross-period effect | SNR | Verdict |
|---|---|---|---|---|
| AI-mention prevalence | +0.004 | +0.133 | **32.9** | clean signal |
| Scope-term prevalence | +0.005 | +0.210 | **42.8** | clean signal |
| Description length | +0.28 d | +0.50 d | 1.80 | near noise |
| Junior share (J3) | +4.75 pp | +5.04 pp | 1.06 | near noise |
| Senior share (S4) | +7.09 pp | +7.59 pp | 1.07 | near noise |
| Management-language (strict) | +0.038 | +0.080 | 2.07 | above noise |
| Management-language (broad) | +0.062 | +0.028 | 0.45 | **below noise** |
| Aggregate credential stack | noise | +0.20 | 0.59 | **below noise** |

What to take from this:

- **AI and scope are clean cross-period signals.** SNRs far above 10.
- **Seniority shares are near noise on an annualized scale.** The 2024-to-2026 change in junior and senior shares is only slightly larger than what the two 2024 datasets show against each other. The raw percentage-point deltas are still directional, but should not be annualized to a yearly rate.
- **The broad management pattern is below noise.** This corroborates the audit finding that the broad management pattern has only 28% precision and should not carry claims.
- **Aggregate credential stack is below noise at the corpus level.** Credential-stack claims must be reported per seniority tier (junior +16.9 pp, senior +13.3 pp), never aggregated.

## The thirteen-definition seniority panel (dimension a, applied to seniority)

Because asaniczka has no native entry-level labels and arshkon is small, the study tested thirteen different operationalizations of seniority across six analysis groups, 78 (definition, group) cells in total.

**Twelve of thirteen definitions are direction-consistent across periods:**

- **Junior side: all seven definitions move up.** J1, J2, J3, J4, J5, J6, and J3_rule (the composite of J3 with title-keyword matching) all show the junior share rising between 2024 and 2026.
- **Senior side: five of six definitions move down.** S1, S3, S4, S5, and S4_rule move the same way. The sixth, S2 (director-only, less than 1% of software engineering), is flat. S2 sits at the corpus's noise floor and has the low label precision discussed in [limitations](limitations.md).

**The primary pair used for headline claims** is J3 (postings asking for two or fewer years of experience, using the LLM's years-of-experience extraction) and S4 (postings asking for five or more). Pooled-2024 is the baseline; arshkon-only is reported as a co-primary for senior claims specifically.

**Primary magnitudes:**

- Junior (J3): +5.04 pp pooled, +1.19 pp arshkon-only. Minimum detectable effect is 2.5 pp.
- Senior (S4): -7.59 pp pooled, -1.78 pp arshkon-only. The arshkon-only magnitude is the conservative read; the pooled-versus-arshkon gap reflects asaniczka's asymmetric senior baseline.

## The returning-firms sensitivity table (dimension a and f combined)

The central robustness check. Fifteen headlines were computed on two samples side by side: the full corpus, and the 2,109 firms that posted in both 2024 and 2026. If a finding is driven by "the 2026 corpus just has different firms in it", restricting to returning firms should attenuate it.

| Headline | Full change | Returning-firms change | Ratio | Verdict |
|---|---|---|---|---|
| AI-mention prevalence | +9.72 pp | +8.36 pp | 0.86 | robust |
| Junior share, pooled | +5.05 pp | +6.17 pp | 1.22 | **amplifies** |
| Junior share, arshkon-only | +1.19 pp | +2.10 pp | 1.77 | **amplifies** |
| Senior share, pooled | -7.62 pp | -8.29 pp | 1.09 | **amplifies** |
| Senior share, arshkon-only | -1.94 pp | -3.06 pp | 1.58 | **amplifies** |
| Breadth (junior, length-residualized) | +1.56 | +1.09 | 0.70 | partially |
| Breadth (senior, length-residualized) | +2.60 | +2.55 | 0.98 | robust |
| Credential stack (junior) | +17.1 pp | +16.5 pp | 0.97 | robust |
| Credential stack (senior) | +13.4 pp | +13.7 pp | 1.03 | robust |
| Requirements-section share | -2.54 pp | -3.37 pp | 1.33 | robust |
| Description length (median) | +1,244 chars | +1,276 chars | 1.03 | robust |
| Scope-term prevalence | +23.26 pp | +22.41 pp | 0.96 | robust |
| Scope (broader definition) | +22.82 pp | +23.21 pp | 1.02 | robust |
| CI/CD at senior | +20.62 pp | +20.27 pp | 0.98 | robust |
| AI-oriented senior cluster | +10.19 pp | +9.18 pp | 0.90 | robust |

**Thirteen of fifteen headlines retain a ratio of at least 0.80 on the returning-firms cohort. None are driven by sampling-frame change.** The junior and senior share changes actually *intensify* when restricted to returning firms; they get larger, not smaller. The junior-breadth finding at 0.70 is the one partial-robustness case.

Bootstrap 95% confidence intervals on the returning-firms cohort: AI-mention [+7.61, +9.08]; junior share [+3.21, +9.86]; senior share [-11.97, -4.67]; scope [+19.94, +25.27]. All exclude zero.

## The hiring-selectivity null result

One alternative explanation for the content shift is that firms are becoming more selective (hiring less, demanding more). The data does not support this.

Pearson correlations between each firm's 2024-to-2026 posting-volume change and its 2024-to-2026 content change, on the 243-firm primary panel:

| Metric | Pearson r | 95% CI | p |
|---|---|---|---|
| Breadth (length-residualized) | -0.032 | [-0.157, +0.094] | 0.617 |
| AI-mention | -0.089 | [-0.212, +0.038] | 0.169 |
| Mentor-language on mid-senior | -0.072 | [-0.202, +0.061] | 0.287 |
| Description length (median) | **+0.203** | [+0.079, +0.321] | **0.0015** |
| Years-of-experience ask (median) | -0.008 | [-0.138, +0.122] | 0.899 |
| Scope (v1 pattern) | -0.033 | [-0.158, +0.093] | 0.605 |

Only description length is significantly correlated with volume change, and the direction is positive: firms posting *more* postings write *longer* descriptions, the opposite of what a selectivity story predicts. The pattern replicates on the 125-firm and 356-firm panels.

## The LLM-authorship re-test

A second alternative explanation: postings look different because recruiters are using language models to write them. The study built a diagnostic that flags postings likely to be LLM-authored (based on characteristic LLM vocabulary signatures) and re-ran the content effects on the quartile of postings *least* likely to be LLM-authored.

| Metric | Full sample | Low-LLM-quartile | % preserved |
|---|---|---|---|
| AI-mention | +0.131 | +0.115 | 88% |
| Tech-stack count | +2.06 | +1.99 | 97% |
| Scope density | +0.096 | +0.123 | 128% (stronger) |
| Credential stack (junior) | +16.9 pp | +16.6 pp | 97% |
| Requirement breadth (length-residualized) | +1.22 | +0.98 | 80% |
| Length growth | +1,130 chars | +583 chars | 52% (half LLM-mediated) |

Content effects hold at 80 to 130% of their full-sample magnitude. Length growth is different: roughly half of it is LLM-mediated. But the content findings are not driven by recruiter LLM use.

## Aggregator-exclusion checks (dimension b)

Staffing agencies and job aggregators can re-post the same role dozens of times and distort company-level analyses. Excluding them:

- **Within-company AI rewriting** moves by less than 20%; direction is preserved.
- **Same-title pair AI drift** *tightens* from +13.4 pp to +16.5 pp. Direct employers lead the rewriting, which is the opposite of what a "noise from aggregators" story would predict.
- **Ghost-likeness** concentrates at direct employers, not aggregators (a counter-intuitive finding).
- **Archetype-cluster metrics** shift less than 3% on aggregator exclusion.

## Length residualization (dimension e)

Any content metric whose correlation with log(description length) exceeds 0.3 is residualized on log-length before cross-period comparison. The model is a simple linear fit: metric ~ a + b · log(length), computed on the full corpus. Metrics that get residualized:

- Requirement breadth (r with log-length = 0.45)
- Credential-stack depth (r = 0.34)
- Scope count (r = 0.41)
- Broad management pattern (r = 0.39)

An independent replication fit the same residualization and produced residuals matching the original at a mean absolute difference of 0.001.

## Pattern validation

How the AI-mention and scope patterns themselves were validated is on the [pattern validation page](pattern-validation.md).
