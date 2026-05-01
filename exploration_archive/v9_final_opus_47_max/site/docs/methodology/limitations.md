# Limitations

Every study has gaps. This page lists the ones that a reader should have in mind before relying on any specific number on this site. Foundational limitations come first within each section; technical ones follow.

## Data

1. **The study has three snapshots, not a continuous time series.** The three datasets are the 2024 asaniczka release (January 2024), the 2024 arshkon release (April 2024), and our 2026 scrape (March-April 2026). There is no 2025 snapshot in between. This means no event study can be run on any specific model release date. The two-year gap spans roughly eight major model launches, and the data cannot separate their effects.

2. **The 2024 asaniczka dataset has no native entry-level labels.** LinkedIn's own "entry" tag was stripped before the dataset was published. Any claim that uses LinkedIn's native seniority labels has to fall back to arshkon alone (which is four times smaller). This is the single biggest reason the study uses years-of-experience floors as its primary seniority definitions.

3. **The 2026 scrape is query-stratified.** It was built by searching for specific roles rather than sweeping every posting on LinkedIn. Claims about *shares within* software engineering (the share of postings asking for 2+ years, say) are valid. Claims about software engineering as a share of *all jobs* on the platform are not.

4. **Posting-age information is almost entirely missing.** Fewer than 1% of rows carry a usable posting-age field, which rules out any analysis of how long postings stay up or how often they get re-listed.

5. **LinkedIn's industry taxonomy changed between 2024 and 2026.** Industry labels cannot be compared across periods without an external crosswalk to a stable classification like NAICS. No cross-period claim stratified by raw industry label is valid.

## Classification

6. **The control-occupation set is a judgement call.** The study tested four alternative definitions of "non-tech control occupations" and the headline results hold under all four, but an analysis-phase paper should pre-register a specific wider or narrower set before running the test again.

7. **ML Engineer roles are classified inconsistently across sources.** The software-engineering classifier flags 78% of ML-engineer postings in the arshkon dataset but only 59% in asaniczka, likely because of different title conventions. Any ML-specific claim must stratify by source.

8. **The director cell is too small and too noisy to carry claims on its own.** The LLM's label precision on director-level postings is only 22 to 27%, and directors are less than 1% of the corpus to begin with. The study uses the director code diagnostically only, never as a primary signal.

9. **The language-model seniority classifier abstains on 34 to 53% of software-engineering rows.** This is by design, not a defect: the prompt explicitly tells the model to return "unknown" on unclear cases rather than guess. But it leaves a large pool of rows with no LLM seniority label, and analyses that need a label for every row have to use the rule-based fallback.

## Method

10. **The same-title pair panel's exact row count could not be reproduced.** The original analysis reports 23 same-firm same-title pairs under specific filter settings; a follow-up replication under a relaxed filter produces 37 pairs (+9.98 percentage points AI rewriting), and under strict arshkon-only filtering produces 12 pairs (+13.3 pp). The direction is the same across all three; the magnitude is reported as a range of +10 to +13 pp.

11. **The within-company AI rewriting finding has a pattern-provenance caveat.** The report text claims to use the audited "v1_rebuilt" version of the AI-mention pattern, but the code actually runs the top-level pattern. Under the v1_rebuilt pattern the magnitudes drop by 10 to 15%. Direction is unchanged.

12. **The "disappearing titles" list is small-sample.** Only 2 to 11 titles dropped enough between periods to qualify, depending on the threshold. The legacy-substitution claim (new titles replacing old) is cited only as a qualified negative.

13. **The largest archetype cluster is heterogeneous.** A topic-modelling pass at six clusters bundles data-engineering, DevOps/SRE, and AI-lab data-contract work into one cell. A seven- or eight-cluster solution would likely split this. The cluster-quality score (silhouette 0.477 on a -1 to +1 scale) is mid-range, not clean separation.

14. **The requirements-section direction flips depending on which classifier you use.** Two different sentence-level classifiers that split postings into "requirements" versus "narrative" give opposite aggregate signs on whether the requirements share shrank. The site cites this as flagged; an analysis-phase paper should prefer the LLM-adjudicated classifier.

## LLM tooling

15. **Coverage on the 2026 scrape is 56.9%.** Forty-three percent of 2026 postings were not routed to the LLM (the run was budget-capped). Any claim that reads the LLM's output on the 2026 data must either restrict to the labeled subset or report the labeled-versus-unlabeled split.

16. **Recruiter-side LLM writing is present in the corpus but does not dominate the findings.** A diagnostic that flags postings likely written with LLM assistance shows that roughly 52% of the 2024-to-2026 length growth is LLM-mediated, but 80 to 130% of the content effects (AI mentions, scope terms, credential counts) persist even on the quartile of postings least likely to be LLM-authored. A more targeted test for whether the effect is senior-specific is deferred to a future phase.

## Benchmarks

17. **The worker-side AI-use benchmarks are 2024 vintage.** The employer-versus-worker divergence is computed against 2024 survey numbers. Later 2025 and 2026 vintages will show higher worker-side AI use and therefore a wider gap; the study's reported gap should be read as a 2024-benchmark-date estimate.

18. **Worker-side benchmarks use different definitions.** The three sources the study triangulates against (Stack Overflow, DORA, Anthropic) each define "AI use at work" differently. The study reports across four definitional bands; the direction is positive under all four, but the magnitude varies with the definition.

## Open questions the analysis phase should take up

Four questions the exploration surfaced but could not answer without data it did not have:

19. **Does within-firm AI rewriting correlate with firm-level digital maturity?** Requires external firm-maturity data the study did not collect.

20. **Is the concentration of "applied AI" senior roles in financial services real?** Seventeen percent of the main applied-AI cluster is financial services, which is plausibly driven by regulatory-AI adoption, but this needs regulatory-context interviews the study did not conduct.

21. **Are junior postings specifically hedging their AI asks (requiring some AI exposure but not committing)?** The data supports a qualitative read; a formal test is deferred to a later phase.

22. **Are senior individual contributors being asked to act as team multipliers?** Requires external hiring-panel data the study did not have.

## What the exploration deliberately did not do

The study is scoped to pattern-finding, not causal testing. Specifically:

- No new analysis was added in the final packaging phase. The site reports what the exploration already produced; no new statistical claims were introduced to fill gaps.
- No worker interviews were conducted. Interview artifacts exist as outputs (discussion guides, exemplar postings), but no transcripts were collected.
- No hypothesis was tested at a pre-registered significance level. Findings are descriptive and patterning-oriented. An analysis phase would pre-register tests for the four deferred questions above.
- No causal claim is made. Three alternative explanations for the main finding (that language models are writing the postings; that firms are lowering their hiring bars; that firms are becoming more selective) were ruled out at the quantitative level, but the positive causal story (that firms are genuinely restructuring work) awaits interview evidence to close the loop.
