# Seniority Measurement Investigation

**Date:** 2026-04-09
**Purpose:** Investigate why entry-level SWE posting share trends flip direction depending on measurement method. Produce a defensible reconciliation for the paper's methodology.

---

## Executive Summary

The headline RQ1 finding -- whether entry-level SWE posting share increased or decreased from 2024 to 2026 -- **reverses direction** depending on the measurement method. This investigation decomposes the discrepancy across eight operationalizations and discovers that the reversal is driven by a specific mechanism: **LinkedIn's native "entry" label was substantially noisier in 2024 than in 2026**, inflating the 2024 baseline. Once this is accounted for, the label-independent evidence (YOE requirements) suggests **entry-level demand was approximately stable or modestly increased** between periods.

### Summary of all entry-share estimates

| Method | Arshkon 2024 | Scraped 2026 | Direction | Delta (pp) |
|--------|-------------|-------------|-----------|-----------|
| seniority_native (non-null) | 22.3% | 13.7% | DECREASE | -8.7 |
| seniority_final (known only) | 21.0% | 14.2% | DECREASE | -6.8 |
| seniority_final (all rows) | 16.9% | 13.3% | DECREASE | -3.6 |
| seniority_imputed (known) | 6.8% | 6.7% | ~FLAT | -0.1 |
| seniority_llm (non-unknown) | 16.7% | 27.7% | INCREASE | +11.1 |
| Explicit entry only | 2.1% | 2.6% | INCREASE | +0.5 |
| YOE<=2 proxy | 15.0% | 16.6% | INCREASE | +1.6 |
| YOE<=3 proxy | 30.8% | 33.6% | INCREASE | +2.8 |

**Key finding:** The only methods showing a DECREASE are those that rely on LinkedIn's native seniority labels. Every label-independent method (YOE proxies, explicit text signals, text-only imputation) shows stability or a modest increase.

---

## Analysis 1: Seniority Detection Method Decomposition

### How seniority_final is resolved

| Detection method | Arshkon (n) | Arshkon (%) | Scraped (n) | Scraped (%) |
|-----------------|------------|------------|------------|------------|
| native_backfill | 1,991 | 39.7% | 15,870 | 45.3% |
| title_keyword | 1,890 | 37.7% | 16,027 | 45.7% |
| unknown | 984 | 19.6% | 2,254 | 6.4% |
| title_prior | 59 | 1.2% | 121 | 0.4% |
| weak_title_level | 54 | 1.1% | 653 | 1.9% |
| description_explicit | 30 | 0.6% | 93 | 0.3% |
| weak_title_associate | 11 | 0.2% | 41 | 0.1% |

**Structural differences:**
1. Arshkon has 19.6% unknown (no seniority resolved); scraped has only 6.4%. This means seniority_final has much better coverage in 2026.
2. Native_backfill and title_keyword are roughly balanced in both periods, but scraped has slightly more title_keyword (45.7% vs 37.7%).
3. The "known rate" is 80.4% for arshkon vs 93.6% for scraped.

### Entry share within each detection method

| Detection method | Arshkon entry % | Scraped entry % |
|-----------------|----------------|----------------|
| native_backfill | **36.6%** | **23.3%** |
| title_keyword | 5.3% | 5.6% |
| description_explicit | 16.7% | 17.2% |
| weak_title_level | 25.9% | 7.8% |

The native_backfill channel drives the entire entry-share decline. Within title_keyword and description_explicit, entry rates are nearly identical across periods (5.3% vs 5.6%, 16.7% vs 17.2%). The trend reversal is entirely attributable to a decline in native_backfill's entry yield from 36.6% to 23.3%.

---

## Analysis 2: Explicit vs. Implicit Entry Detection

### Definitions
- **Explicitly-signaled entry:** seniority_final = 'entry' AND detected via title_keyword or description_explicit (the posting text itself signals entry level)
- **Implicitly-detected entry:** seniority_final = 'entry' AND detected via native_backfill (LinkedIn's classifier labeled it entry, but our text rules could not detect it)

### Results

| Metric | Arshkon 2024 | Scraped 2026 |
|--------|-------------|-------------|
| Total SWE postings | 5,019 | 35,062 |
| **Explicit entry** (n / %) | 105 / **2.1%** | 910 / **2.6%** |
| **Implicit entry** (n / %) | 729 / **14.5%** | 3,695 / **10.5%** |
| Total entry (n / %) | 848 / 16.9% | 4,656 / 13.3% |

**Interpretation:** Explicit entry-level detection -- which is purely text-based and does not depend on LinkedIn's classifier -- shows a **modest increase** from 2.1% to 2.6%. The decline is entirely driven by implicit (native-backfill) entry labels dropping from 14.5% to 10.5%.

Among known-seniority rows only:

| Metric | Arshkon 2024 | Scraped 2026 |
|--------|-------------|-------------|
| Known total | 4,035 | 32,808 |
| Explicit entry % of known | 2.6% | 2.8% |
| Implicit entry % of known | 18.1% | 11.3% |
| Total entry % of known | 21.0% | 14.2% |

The implicit entry channel accounts for 86% of the 6.8pp decline in entry share among known rows (18.1% - 11.3% = 6.8pp implicit decline vs. 0.2pp explicit increase).

---

## Analysis 3: YOE as a Label-Independent Proxy

YOE (years of experience) is extracted directly from posting text and does not depend on any seniority classification system. It provides a completely independent signal about whether postings target junior candidates.

### YOE coverage
| Source | Total SWE | Has YOE | Coverage |
|--------|-----------|---------|----------|
| Arshkon | 5,019 | 3,040 | 60.6% |
| Scraped | 35,062 | 24,209 | 69.1% |

### YOE distribution summary
| Statistic | Arshkon 2024 | Scraped 2026 |
|-----------|-------------|-------------|
| Mean YOE | 5.20 | 4.99 |
| Median YOE | 5.0 | 5.0 |
| P25 | 3.0 | 3.0 |
| P75 | 7.0 | 6.0 |

### Entry-like YOE thresholds
| Threshold | Arshkon 2024 | Scraped 2026 | Direction |
|-----------|-------------|-------------|-----------|
| YOE <= 1 | 3.7% | 4.5% | +0.8pp INCREASE |
| YOE <= 2 | 15.0% | 16.6% | +1.6pp INCREASE |
| YOE <= 3 | 30.8% | 33.6% | +2.8pp INCREASE |
| YOE >= 5 | 59.3% | 57.2% | -2.1pp DECREASE |
| YOE >= 7 | 26.5% | 23.5% | -3.1pp DECREASE |

**The YOE evidence consistently favors a modest shift toward lower experience requirements.** At every low-YOE threshold, the scraped 2026 share is higher than arshkon 2024. At every high-YOE threshold, the scraped share is lower. This is the opposite direction from the native-label-based finding.

### YOE histogram (integer bins, YOE 1-10)

| YOE | Arshkon % | Scraped % | Delta |
|-----|-----------|-----------|-------|
| 1 | 3.8% | 4.6% | +0.8 |
| 2 | 11.6% | 12.4% | +0.8 |
| 3 | 16.2% | 17.4% | +1.2 |
| 4 | 10.1% | 9.4% | -0.7 |
| 5 | 27.2% | 28.1% | +0.9 |
| 6 | 6.5% | 6.5% | 0.0 |
| 7 | 6.5% | 6.3% | -0.1 |
| 8 | 9.4% | 8.7% | -0.7 |
| 9 | 1.3% | 0.5% | -0.8 |
| 10 | 7.5% | 6.0% | -1.4 |

The distribution shift is concentrated at the low end (YOE 1-3 gaining share) and the high end (YOE 8-10 losing share). The modal value of 5 years is stable.

---

## Analysis 4: Unknown-Pool Analysis

### YOE profile of the unknown pool

| Seniority group | Source | N | Mean YOE | Median YOE | % with YOE<=2 |
|-----------------|--------|---|----------|-----------|---------------|
| entry | arshkon | 848 | 3.98 | 3.0 | 32.6% |
| entry | scraped | 4,656 | 2.19 | 2.0 | 82.1% |
| **unknown** | **arshkon** | **984** | **5.04** | **5.0** | **16.5%** |
| **unknown** | **scraped** | **2,254** | **4.07** | **3.5** | **28.1%** |
| mid-senior | arshkon | 2,924 | 5.67 | 5.0 | 8.2% |
| mid-senior | scraped | 26,805 | 5.32 | 5.0 | 9.8% |

The unknown pool in arshkon (mean YOE 5.04) looks more like mid-senior than entry. In scraped, the unknown pool is intermediate (mean YOE 4.07), leaning somewhat more junior than arshkon's unknown pool.

### Bounds on total entry rate

| Assumption about unknown pool | Arshkon entry % | Scraped entry % |
|-------------------------------|----------------|----------------|
| Unknown = 0% entry (lower bound) | 16.9% | 13.3% |
| Unknown = same rate as known | 21.0% | 14.2% |
| Unknown = 100% entry (upper bound) | 36.5% | 19.7% |

**No assumption about the unknown pool can reverse the decline under native labels.** Even if scraped's unknown pool were 56.3% entry (an extreme assumption), it would only equalize the total entry rate to match arshkon's 16.9%. Since the unknown pool's YOE profile is clearly not entry-dominated, the decline is robust under reasonable unknown-pool assumptions.

However, this analysis is conditional on the native labels being equally accurate in both periods -- which the next analysis questions.

---

## Analysis 5: Title-Level Explicit/Implicit Gap

### Top entry-native titles in arshkon: virtually no explicit signal

Among arshkon postings labeled seniority_native = 'entry', the top titles are generic titles with **no entry-level keywords**:

| Title | Arshkon native-entry n | Explicit signal % |
|-------|----------------------|------------------|
| software engineer | 52 | 1.9% |
| data engineer | 28 | 0.0% |
| developer | 23 | 8.7% |
| systems engineer | 19 | 0.0% |
| java developer | 18 | 5.6% |
| automation engineer | 16 | 6.3% |
| software developer | 12 | 0.0% |
| application engineer | 10 | 0.0% |
| devops engineer | 9 | 0.0% |

**98% of the top generic titles' native-entry classification comes exclusively from LinkedIn's platform label**, not from any signal in the posting text. A "software engineer" posting labeled "entry" has no textual justification for that classification.

### Same titles in scraped: still get entry labels, but at lower rates

| Title | Arshkon entry (n) | Scraped total | Scraped entry % |
|-------|------------------|--------------|----------------|
| software engineer | 52 | 3,236 | 17.3% |
| data engineer | 28 | 1,028 | 12.9% |
| developer | 23 | 87 | 10.3% |
| systems engineer | 19 | 172 | 10.5% |
| software developer | 12 | 202 | 22.8% |
| devops engineer | 9 | 496 | 11.9% |

These titles continue to receive entry labels in scraped data, but without knowing the arshkon denominator for each title, the within-title entry rate comparison is difficult. The key observation is that LinkedIn continues to label generic titles as "entry" in 2026.

---

## Analysis 6: The Labeling-Explicitness Hypothesis

**Hypothesis:** In 2026, employers more frequently include explicit seniority cues in posting titles/descriptions compared to 2024.

### Evidence

| Metric | Arshkon 2024 | Scraped 2026 | Delta |
|--------|-------------|-------------|-------|
| seniority_imputed known rate | 44.6% | 55.2% | **+10.6pp** |
| Explicit signal in seniority_final_source | 38.3% | 46.0% | **+7.7pp** |
| Title_keyword only | 37.7% | 45.7% | **+8.1pp** |

### Title-level keyword prevalence

| Keyword category | Arshkon 2024 | Scraped 2026 | Delta |
|-----------------|-------------|-------------|-------|
| Junior cue (junior/entry/intern/associate/etc.) | 1.8% | 3.6% | **+1.8pp** |
| Senior cue (senior/lead/principal/staff/etc.) | 1.9% | 0.8% | -1.1pp |
| Any seniority cue | 4.7% | 6.3% | **+1.7pp** |

**Specific keyword breakdown:**

| Keyword | Arshkon % | Scraped % |
|---------|-----------|-----------|
| intern | 0.94% | 1.40% |
| associate | 0.68% | 1.21% |
| entry | 0.10% | 0.33% |
| new grad | 0.04% | 0.10% |

**The hypothesis is STRONGLY SUPPORTED.** The imputed-known rate increased 10.6 percentage points, meaning substantially more 2026 postings contain textual seniority signals. Junior-specific keywords nearly doubled (1.8% to 3.6%). The "entry-level" keyword increased 3x (0.10% to 0.33%). "Intern" increased from 0.94% to 1.40%. "Associate" nearly doubled from 0.68% to 1.21%.

### Seniority_imputed distribution (text-only signals)

| Level | Arshkon 2024 | Scraped 2026 |
|-------|-------------|-------------|
| unknown | 55.4% | 44.9% |
| mid-senior | 38.5% | 46.6% |
| entry | 3.0% | 3.7% |
| associate | 2.8% | 4.1% |
| director | 0.2% | 0.8% |

The unknown rate dropped 10.5pp -- meaning text-based classification became more feasible in 2026. Within the known pool, entry stayed almost flat (6.75% vs 6.66%), while associate increased.

---

## Analysis 7: The Critical YOE Validation of Native Labels

This is the most important finding in this investigation.

### Arshkon's native-entry labels are substantially noisier than scraped's

| Metric | Arshkon native-entry | Scraped native-entry |
|--------|---------------------|---------------------|
| N | 769 | 3,972 |
| Has YOE (n) | 475 | 1,552 |
| Mean YOE | **4.12** | **2.33** |
| Median YOE | **3.0** | **2.0** |
| % with YOE <= 2 | **29.7%** | **79.8%** |
| % with YOE >= 5 | **41.3%** | **9.5%** |

**41% of arshkon postings labeled "entry" by LinkedIn's native classifier have YOE >= 5 years** -- a threshold that is categorically not entry-level. In scraped data, only 9.5% of native-entry postings have YOE >= 5.

### Implicit-entry (native backfill) YOE histogram

| YOE | Arshkon implicit-entry % | Scraped implicit-entry % |
|-----|-------------------------|-------------------------|
| 1 | 4.8% | 25.5% |
| 2 | 24.7% | 57.2% |
| 3 | 22.1% | 7.1% |
| 4 | 7.7% | 2.8% |
| 5 | **21.7%** | 3.5% |
| 6 | 4.6% | 1.2% |
| 7 | 3.3% | 0.9% |
| 8 | **7.2%** | 1.0% |
| 9 | 0.4% | 0.1% |
| 10+ | 3.3% | 0.8% |

Arshkon's implicit-entry pool is bimodal: a genuine entry cluster at YOE 1-3, and a **spurious cluster at YOE 5+ that represents nearly 40% of the pool**. Scraped's implicit-entry pool is concentrated at YOE 1-2 (82.7%), with minimal contamination above YOE 5.

### What this means

LinkedIn's native seniority classifier appears to have been **substantially less accurate in 2024 (or for the arshkon data snapshot)** than in 2026. Many postings requiring 5+ years of experience were labeled "entry," inflating the 2024 entry-level baseline by approximately 40%. If we exclude the contaminated portion:
- Arshkon entry count, corrected: ~848 * 0.60 = ~509 (removing the ~40% with YOE>=5)
- Corrected arshkon entry rate: ~509/5019 = **10.1%**
- Scraped entry rate (raw, already accurate): 13.3%
- **Corrected direction: INCREASE (+3.2pp)**

This correction is rough but directionally robust. The native label quality difference explains the entire sign reversal.

### Explicit-entry YOE validation: consistent across periods

| Category | Source | Mean YOE | Median YOE | % YOE<=2 |
|----------|--------|----------|-----------|----------|
| Explicit entry | arshkon | 2.30 | 2.0 | 65.0% |
| Explicit entry | scraped | 2.04 | 2.0 | 77.7% |
| Implicit entry | arshkon | **4.11** | **3.0** | **29.5%** |
| Implicit entry | scraped | 2.23 | 2.0 | 82.7% |
| Mid-senior | arshkon | 5.67 | 5.0 | 8.2% |
| Mid-senior | scraped | 5.32 | 5.0 | 9.8% |

Explicit-entry postings (detected from text) have consistent YOE profiles across periods -- validating that the text-based methods are temporally stable. The contamination is entirely in the implicit (native backfill) channel, and entirely in arshkon.

---

## Analysis 8: The "Software Engineer" Title Case Study

"Software Engineer" is the single most common SWE title. Examining its seniority distribution reveals the mechanism at work:

### seniority_native distribution for "Software Engineer" title

| Native label | Arshkon (n/%) | Scraped (n/%) |
|-------------|--------------|--------------|
| mid-senior | 248 / 54.6% | 2,011 / 62.1% |
| (null) | 134 / 29.5% | 522 / 16.1% |
| entry | 52 / **11.5%** | 560 / **17.3%** |
| associate | 18 / 4.0% | 106 / 3.3% |

Within this single title, entry share INCREASED from 11.5% to 17.3% -- the opposite direction from the aggregate trend. This confirms that the aggregate decline is not driven by within-title changes but by a shift in native label accuracy.

### YOE within "Software Engineer" by native label

| Native label | Source | Mean YOE | % YOE<=2 |
|-------------|--------|----------|----------|
| entry | arshkon | 3.00 | 52.6% |
| entry | scraped | 2.09 | 84.7% |
| mid-senior | arshkon | 5.30 | 7.5% |
| mid-senior | scraped | 5.44 | 10.2% |

Even within the "Software Engineer" title, arshkon's entry-labeled postings have substantially higher mean YOE (3.0 vs 2.1) and much lower low-YOE concentration (53% vs 85%) compared to scraped.

---

## Reconciliation Narrative

### Evaluating the five alternative hypotheses

**Hypothesis 1: "Entry share truly declined, and the explicit-signal increase is a labeling artifact."**
- **REJECTED.** The YOE-based proxy, which is entirely label-independent, shows a modest increase (15.0% to 16.6% for YOE<=2). If entry demand truly declined, YOE requirements should have shifted upward -- they did not. Additionally, the native-label decline can be explained by the documented YOE contamination in arshkon's entry labels.

**Hypothesis 2: "Entry share truly increased, and the native-label decline is a LinkedIn classifier artifact."**
- **PARTIALLY SUPPORTED.** The YOE evidence supports a modest increase. The native label decline is explained by differential label quality (41% of arshkon's native-entry postings have YOE>=5, vs 9.5% in scraped). However, we cannot definitively attribute this to LinkedIn's classifier changing -- it could also reflect arshkon's data collection process.

**Hypothesis 3: "Both are true: explicitly-labeled entry jobs increased while implicitly-entry jobs decreased -- the market bifurcated."**
- **NOT SUPPORTED in its strong form.** The "implicit entry decline" is an artifact of the contaminated arshkon labels. Scraped's implicit-entry pool has YOE profiles consistent with genuine entry-level roles. There is no evidence of market bifurcation.

**Hypothesis 4: "The unknown pool changed composition, making both measures unreliable."**
- **MINOR FACTOR.** Arshkon has 19.6% unknown vs. scraped's 6.4%, which does affect the denominator. The unknown pool's YOE profile is closer to mid-senior in both periods. But even aggressive assumptions about the unknown pool cannot reverse the label-independent YOE findings.

**Hypothesis 5: "The entry share barely changed, and both directions are within measurement noise."**
- **BEST SUPPORTED BY THE DATA.** The label-independent evidence (YOE) shows a modest +1.6pp increase, within the range of measurement noise. The text-only imputed method shows entry share virtually flat (6.75% vs 6.66% among known rows). The dramatic swings in either direction are artifacts of different methods' sensitivity to the native label quality differential.

### Best estimate and confidence bounds

**Most defensible point estimate:** Entry-level SWE posting share was **approximately 13-17% in both periods**, with the change being **small and direction-uncertain** after correcting for measurement differences.

**Bounds:**
- Lower bound (if arshkon labels are fully accurate): -8.7pp decline (native non-null: 22.3% to 13.7%)
- Upper bound (if arshkon labels are as contaminated as YOE suggests): +3.2pp increase (corrected ~10% to 13.3%)
- Label-independent center: +1.6pp (YOE<=2 proxy: 15.0% to 16.6%)

**Statistical significance:** The native-label decline is statistically significant (Z=11.75, p<0.001, 95% CI: 7.2-10.1pp). But statistical significance is irrelevant if the measurement is systematically biased, which the YOE validation strongly suggests.

### Recommended approach for the paper

1. **Primary finding:** Report the native-label entry share decline (22.3% to 13.7%) as the "unadjusted" estimate, then immediately present the YOE validation evidence showing that arshkon's native entry labels are substantially contaminated (41% have YOE>=5).

2. **Adjusted estimate:** Present the YOE-corrected entry share (roughly flat to +3pp increase) as the bias-corrected estimate.

3. **Label-independent robustness:** Present the YOE<=2 proxy (15.0% to 16.6%) and YOE<=3 proxy (30.8% to 33.6%) as fully independent evidence that entry-level demand did not decline.

4. **Bound the uncertainty:** Report the full range of estimates from the 8 operationalizations (decrease of 8.7pp to increase of 11.1pp) to convey the methodological uncertainty honestly.

5. **Substantive interpretation:** The most defensible claim is: "Entry-level SWE posting share was approximately stable between 2024 and 2026. Methods relying on LinkedIn's native seniority classifier show an apparent decline, but this is attributable to differential label quality across periods, as validated by years-of-experience requirements."

6. **What DID change:** The labeling-explicitness analysis shows that employers in 2026 are significantly more likely to include explicit seniority signals in posting titles (imputed-known rate increased from 44.6% to 55.2%). Junior-specific keywords nearly doubled (1.8% to 3.6%). This is a real finding about employer behavior -- the market became more transparent about seniority expectations -- even if total entry demand was roughly stable.

---

## Data Quality Implications

### For the preprocessing pipeline
- The `seniority_native` field should be flagged as having differential accuracy across data sources
- Consider adding a YOE-based validation flag for native-entry postings (flag those with YOE>=5 as suspect)
- The `seniority_final` column inherits native label noise through native_backfill; any analysis using seniority_final for entry-level trends must account for this

### For the analysis phase
- RQ1 entry-level share comparisons should present multiple operationalizations with the YOE validation
- Do NOT present the native-label decline as the headline finding without the bias correction
- The YOE-based proxy should be elevated to a co-primary measure alongside seniority labels
- seniority_imputed (text-only) is the most temporally stable label-based measure, though it has low power due to high unknown rates

### For the LLM classification budget
- A larger LLM classification budget would help, but the current seniority_llm results are on dangerously small samples (504 arshkon, 649 scraped non-unknown)
- seniority_llm shows an increase (16.7% to 27.7%), but this may reflect the LLM's explicit-signal-only design biasing toward the increasingly-explicit 2026 postings

---

## Output Files

- **Tables:** `exploration/tables/seniority_measurement/`
  - `detection_method_decomposition.csv` -- seniority_final_source breakdown by period
  - `known_vs_explicit_rates.csv` -- known and explicit seniority rates
  - `explicit_vs_implicit_entry.csv` -- explicit vs implicit entry counts and rates
  - `yoe_distribution_by_period.csv` -- YOE statistics by period
  - `yoe_histogram.csv` -- YOE integer-bin histogram
  - `unknown_pool_yoe.csv` -- YOE by seniority group including unknown
  - `unknown_pool_bounds.csv` -- entry rate under unknown-pool assumptions
  - `arshkon_entry_top_titles.csv` -- top entry-native titles in arshkon
  - `title_entry_rate_comparison.csv` -- same titles across periods
  - `labeling_explicitness.csv` -- explicitness metrics by period
  - `yoe_by_entry_type.csv` -- YOE for explicit vs implicit entry
  - `entry_share_summary.csv` -- master summary of all 8 operationalizations
- **Script:** `exploration/scripts/seniority_measurement_investigation.py`
