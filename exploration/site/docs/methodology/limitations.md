# Limitations and Risks

## High-severity confounders

### Description length growth

The single largest effect in the data: 57-67% median growth from 2024 to 2026 (Cohen's d = 0.77). This inflates all text-based metrics including keyword counts, technology mentions, and scope indicators. **Mitigation:** Length-normalize all density metrics (per 1K characters). Use binary presence for keyword analyses.

### Asaniczka seniority gap

Asaniczka has zero native entry-level labels. Its 0.6% entry rate is a title-keyword imputation artifact. Including asaniczka in seniority trends creates false precision and potentially reverses the direction of entry-level findings. **Mitigation:** Exclude asaniczka from all seniority-stratified analyses. Use arshkon-only as 2024 entry baseline.

### Company composition

57% of aggregate change is compositional (T16). Only 18% of companies overlap between periods. Aggregate cross-period comparisons conflate within-firm changes with compositional shifts. **Mitigation:** Use within-company panel design (451 companies) as primary. Report aggregate as robustness.

### Text quality asymmetry

2024 Kaggle data has LLM-cleaned text (80-99% SWE coverage). 2026 scraped has 0% LLM-cleaned text. Rule-based cleaning retains more boilerplate in 2026, systematically inflating apparent text-based changes. The T22 analysis showed LLM vs rule-based text differs by ~15pp on management indicators. **Mitigation:** Use `description_core` uniformly for cross-period comparisons. When LLM-cleaned scraped text becomes available, re-run all text analyses.

## Moderate-severity confounders

### Field-wide posting evolution

Management, leadership, and soft skills expansion is NOT SWE-specific (T18 DiD ~ 0). Claiming these as AI-driven SWE restructuring would be incorrect. **Mitigation:** Use cross-occupation DiD to isolate SWE-specific effects. Only claim SWE-specificity for metrics with large positive DiD.

### Aggregator contamination

Aggregators are 12-27% of SWE postings by source. They have different seniority profiles and text patterns. **Mitigation:** Run all analyses with and without aggregators. Report both.

### Measurement error in keyword indicators

T22 showed "leading" was 99.4% adjective usage, not management. Broad pattern sets capture boilerplate rather than role content. **Mitigation:** Use validated/strict pattern sets only. Document pattern validation for any regex-based indicator.

### Seniority operationalization sensitivity

The entry-level trend direction depends on column choice. **Mitigation:** Use seniority_native as primary; seniority_final as robustness. Await seniority_llm for definitive resolution.

## Low-severity confounders

### SWE classification noise

4-6% false positive rate; QA/test engineer boundary unstable. **Mitigation:** Report sensitivity with regex-tier-only (highest precision).

### Geographic sampling design

Scraped data uses 26-metro targeted search; Kaggle is national. **Mitigation:** Use metro-level fixed effects. All 26 metros show same direction (T17).

### Scraper first-day backlog

1.6x normal volume on first scrape day (March 20). **Mitigation:** Mild; daily metric CVs < 10% (T19).

## Structural limitations

### Two-point comparison

The study compares 2024 to 2026. With only two time periods (plus a partial within-2024 sub-period), we cannot distinguish gradual trends from abrupt shifts. The GenAI acceleration estimate (8.3x) depends on a 3-point estimation that is inherently fragile.

### No causal identification

Even the cross-occupation DiD does not provide causal identification. It isolates SWE-specificity but cannot identify the mechanism. The orthogonality finding (r ~ 0 at firm and metro level) explicitly constrains causal claims about AI driving junior elimination.

### Posting-side only

The study measures employer demand (job postings), not worker outcomes (who gets hired, what they do). The gap between what postings request and what workers actually do is a well-documented phenomenon. The T22 aspiration analysis partially addresses this by showing AI requirements are genuine, but hiring outcomes remain unobserved.

### External benchmark quality for RQ3

The developer usage estimate (~75%) comes from external surveys (StackOverflow, JetBrains, GitHub) with self-selection bias and different sampling frames. The direction of the posting-usage gap is clear (posting lags usage), but the exact magnitude depends on benchmark choice.

### Survivor bias in overlap panel

The 451-company overlap panel is biased toward large, stable firms. Companies that entered or exited the market (82% of all companies) are excluded from within-firm analyses. The compositional finding (57%) is specific to this panel definition.

## Pending data improvements

### seniority_llm (highest priority)

Explicit-signal-only LLM seniority classification for all SWE rows. Will resolve the operationalization discrepancy, enable asaniczka to participate in seniority analyses, and definitively settle the entry-level trend direction.

### description_core_llm for scraped data (high priority)

LLM-quality boilerplate removal for 2026 scraped data. Will eliminate text quality asymmetry and enable reliable cross-period text density comparisons.

## Alternative narratives the data cannot rule out

1. **HR template modernization:** Management/leadership expansion may reflect changes in how postings are written, not changes in actual roles.
2. **Compositional dominance:** The "restructuring" is substantially about different companies entering the market (AI-native startups) and others exiting.
3. **Cyclical hiring:** Junior share decline could reflect macroeconomic hiring slowdown, not AI-driven restructuring.
4. **Time-lag causation:** The AI-entry link might exist with a lag our two-snapshot design cannot capture.
5. **Post-pandemic normalization:** Some changes may represent reversion from pandemic-era hiring patterns rather than AI-driven transformation.
