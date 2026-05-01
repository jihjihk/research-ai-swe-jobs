# What we rejected

Five alternative explanations for the 2024-to-2026 SWE content shifts were tested quantitatively and rejected. These negative findings matter as much as the positives: each one is a leading prior hypothesis that a reader might otherwise treat as a plausible competing explanation.

## 1. Hidden hiring-bar lowering — rejected

**The prediction.** If the apparent shrink in junior requirements sections is really a signal of employers quietly accepting weaker candidates, we should see (a) that result reproduce across different ways of measuring the requirements section, (b) correlations between section shrink and other proxies of loosened hiring, and (c) explicit loosening language in postings at higher rates.

**The evidence against.**

- The section-shrink result is classifier-dependent. Under the original classifier, the aggregate period coefficient is -0.019 (shrink). Under an independently built simpler-regex classifier, it is +0.030 (growth). Both are statistically clean at p < 1e-13 with HC3-robust standard errors. They point in opposite directions.
- Every hiring-bar proxy we checked correlates with section shrink at absolute Spearman rank-correlation at most 0.28 (where +1 means identical ordering and 0 means no relationship), and the sign flips across classifiers.
- On the returning-firms cohort (356 firms with posting overlap in both 2024 and 2026), within-company change in requirements-share multiplied by within-company change in J3 (postings asking for two or fewer years of experience — the study's primary junior definition) share comes out to near zero under both classifiers.
- On a hand-labelled sample of 50 postings, zero of 50 contained explicit loosening language.
- One correlation comes back positive and significant: requirements-character change correlates with description-length change at r ≈ +0.35. That is consistent with postings getting more narratively expansive, not with hiring bars being quietly cut.

**The caveat.** The claim is about *hidden* hiring-bar lowering, the version that hides inside shorter requirements sections. We are not claiming hiring bars cannot be dropping in ways that do not show up in posting content.

Report: [source task](../evidence/tasks/T33.md).

## 2. Legacy stack substituted by AI-enabled roles — rejected

**The prediction.** If the disappearance of 2024 legacy-stack titles (Java, Drupal, PHP, .NET architects) is really being absorbed by AI-enabled 2026 roles, the 2026 postings that replace them should show elevated AI-mention rates.

**The evidence against.** Of 11 disappearing 2024 titles, 6 have identifiable 2026 neighbours by TF-IDF cosine similarity (a measure of shared vocabulary on weighted word frequencies) in the 0.30 to 0.59 range. The AI-mention rate on those 2026 neighbours is 3.6%, well below the overall 2026 market rate of 14.4%. The actual content drift shows SSIS, Unix scripting, Drupal, PHP, and Scala leaving, and Postgres, pgvector, CI/CD, microservices, Terraform, and ArgoCD arriving. Architect-to-engineer substitutions also drop the asked years of experience by two to four years. The substitution story that fits the data is legacy-stack to modern-stack, not legacy-to-AI.

**The caveat.** This is evidence about *content-matched neighbour roles*, not about every possible pathway a legacy engineer might take through the labour market.

Report: [source task](../evidence/tasks/T36.md).

## 3. Hiring-market selectivity during the hiring trough — rejected

**The prediction.** JOLTS (Job Openings and Labour Turnover Survey) Information-sector openings sat at 0.71x the 2023 average during the 2024 measurement window. If the remaining firms were unusually selective — writing richer job descriptions because they had the leverage of a tight market — we should see firms with bigger volume drops writing broader, more AI-heavy, and more senior-mentor-heavy postings.

**The evidence against.**

| Correlation with firm's volume change | Pearson r | p | Direction |
|---|---|---|---|
| Change in requirement breadth | -0.032 | 0.617 | null |
| Change in AI-mention rate | -0.089 | 0.169 | null |
| Change in senior-mentoring language | -0.072 | 0.287 | null |
| **Change in median description length** | **+0.203** | **0.0015** | **opposite** |
| Change in median YOE asked | -0.008 | 0.899 | null |
| Change in scope density | -0.033 | 0.605 | null |

Only one metric is statistically distinguishable from zero — description length — and it goes in the opposite direction of the theory. Firms whose volume rose, not firms whose volume dropped, wrote longer job descriptions. This holds across three panel definitions.

**The caveat.** The test is on firms' posting *content*, not on whether they are becoming selective in who they *hire*. Selectivity in candidate screening that does not show up in posting content is not measurable here.

Report: [source task](../evidence/tasks/T38.md).

## 4. LLM-authored JDs explain the content rise — rejected as a dominant mediator

**The prediction.** Recruiters are increasingly drafting job descriptions with LLM help. If that is what is driving the 2024-to-2026 rise in AI mentions, credential stacking, and scope breadth, the effects should collapse on the subset of postings that look *least* LLM-authored.

**The evidence against.**

| Metric | Full-corpus change | Low-LLM-style change | Percent preserved |
|---|---|---|---|
| AI-mention rate | +0.131 | +0.115 | 88% |
| Tech-stack count | +2.06 | +1.99 | 97% |
| Credential stack (J3) | +16.9 pp | +16.6 pp | 97% |
| Scope density | +0.096 | +0.123 | 128% (stronger) |
| Requirement breadth (length-residualised) | +1.22 | +0.98 | 80% |
| Length growth | +1,130 chars | +583 chars | 52% |

Content effects persist at 80 to 130% on the low-LLM-style subset. In one case — scope density — the effect is *larger* among less LLM-styled postings.

**The caveat.** The one mechanism that *is* partly LLM-mediated is length growth itself. Job descriptions are getting longer, and about half of that growth does appear to be LLM-introduced boilerplate. So "JDs are getting wordier because LLMs write wordier" is supported. "JDs are getting AI-focused because LLMs insert AI mentions" is not.

Report: [source task](../evidence/tasks/T29.md).

## 5. Sampling-frame artifact — rejected

**The prediction.** The 2026 scraped corpus comes from a different source than the 2024 Kaggle datasets. If the 2026 panel over-weights different kinds of firms (more tech-forward, more AI-heavy, bigger), the headline shifts could be composition-driven: real in the 2026 data, but not a change at any given firm.

**The evidence against.** On the returning-firms cohort (2,109 firms that posted SWE roles in both 2024 and 2026, together 55% of the 2026 posting volume and 25% of 2026 unique firms), 13 of 15 headline metrics reproduce the full-corpus direction robustly. Two actually *amplify* on the within-firm sample. The table below uses two codes: J3 (already introduced above) and S4 (postings asking for five or more years of experience — the study's primary senior definition).

| Metric | Full change | Returning-cohort change | Ratio | Verdict |
|---|---|---|---|---|
| AI-mention prevalence | +9.72 pp | +8.36 pp | 0.86 | robust |
| J3 entry share (pooled) | +5.05 pp | +6.17 pp | 1.22 | amplified |
| S4 senior share (pooled) | -7.62 pp | -8.29 pp | 1.09 | amplified |
| Breadth residualised (S4) | +2.60 | +2.55 | 0.98 | robust |
| Credential stack >= 5 (J3) | +17.1 pp | +16.5 pp | 0.97 | robust |
| Scope kitchen-sink | +22.82 pp | +23.21 pp | 1.02 | robust |
| CI/CD at S4 | +20.62 pp | +20.27 pp | 0.98 | robust |
| AI-oriented senior cluster | +10.19 pp | +9.18 pp | 0.90 | robust |

The J3 and S4 directions intensify on returning-only firms, a stronger test than the full corpus could deliver.

**The caveat.** One of fifteen metrics (J3 breadth-residualised) drops to a ratio of 0.70 on the returning cohort: partially robust, not fully. And the returning cohort is only 25% of 2026 unique firms, so this addresses whether the *same firms changed their content* without speaking to whether the overall population of posting firms is differently composed.

Report: [source task](../evidence/tasks/T37.md).

## What these rejections add up to

Rule out hidden hiring-bar lowering. Rule out legacy-to-AI substitution. Rule out selectivity-driven narrative expansion. Rule out LLM-authorship as a dominant driver. Rule out the sampling-frame composition artifact. What's left, as the live causal stories:

- the content shifts are same-firm, not composition-driven;
- the content shifts are demand-side, not screening-side;
- the content shifts are substantive, not boilerplate drift;
- AI is additive in the posting corpus (arriving alongside the legacy stack's disappearance, not via it), not substitutive;
- requirements sections are changing content, not quietly accepting weaker candidates.

This is what the interview round was designed to probe: the real within-firm rewriting documented in [A2](a2-within-firm-rewriting.md) and the new senior archetypes in [A5](a5-archetypes.md).
