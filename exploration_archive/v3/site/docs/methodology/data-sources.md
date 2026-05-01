# Data Sources

## Overview

The study uses three LinkedIn job posting datasets spanning the AI coding tool adoption window (January 2024 to March 2026), plus a control sample of non-SWE occupations.

## Primary datasets

### Kaggle arshkon (April 2024)

| Property | Value |
|----------|-------|
| Period | April 2024 |
| Platform | LinkedIn |
| Total rows | ~124K |
| US SWE rows | 5,019 |
| Entry-level SWE | 769 (seniority_native) |
| Key strength | Native entry-level labels from LinkedIn API |
| Key gap | Small SWE count |

The arshkon dataset is the only source with reliable native entry-level labels. It serves as the **sole 2024 entry-level baseline**. Asaniczka has zero native entry-level labels and must be excluded from all seniority-stratified analyses.

### Kaggle asaniczka (January 2024)

| Property | Value |
|----------|-------|
| Period | January 2024 |
| Platform | LinkedIn |
| Total rows | ~1.35M |
| US SWE rows | 23,213 |
| Entry-level SWE | 0 (no native entry labels) |
| Key strength | Large volume for cross-sectional analyses |
| Key gap | No entry-level labels; 100% "coverage" is only mid-senior/associate |

Asaniczka provides volume for technology analysis, company-level patterns, and domain clustering, but **must be excluded from all seniority-stratified analyses** due to its complete absence of native entry-level labels.

### Scraped (March 2026)

| Property | Value |
|----------|-------|
| Period | March 20+ 2026 |
| Platform | LinkedIn (primary) + Indeed (sensitivity only) |
| Total rows | ~3.7K SWE/day |
| US SWE rows | 24,095 |
| Entry-level SWE | 3,255 (seniority_native) |
| Key strength | Fresh data with search metadata |
| Key gap | No LLM-cleaned text (0% description_core_llm coverage) |

The scraped dataset uses a 26-metro targeted search design. All 26 metros show the same directional changes. Geographic representativeness is excellent (r > 0.97 vs BLS OES).

## Control sample

~142K non-SWE postings (accountants, nurses, marketing managers, etc.) from the same sources and periods. Used for difference-in-differences to isolate SWE-specific changes from field-wide trends.

## SWE classification

SWE postings are identified using a multi-tier regex classifier on titles and descriptions:

- **False positive rate:** 4-6% (QA/test engineer boundary is the main source)
- **False negative rate:** < 0.5%
- Sensitivity analysis uses regex-tier-only (highest precision) as a robustness check

Full details: [T04: SWE Classification](../reports/T04.md)

## Company overlap panel

451 companies with 3+ SWE postings in both 2024 (pooling arshkon + asaniczka) and 2026. This panel supports within-firm analysis but is biased toward large firms (only 18% of companies overlap between periods). Contains ~22,929 postings (44% of all SWE).

## Text columns

| Use case | Column | Coverage | Notes |
|----------|--------|----------|-------|
| Binary keyword presence | `description` (full text) | 100% all sources | Best recall; includes boilerplate |
| Density/frequency metrics | `description_core_llm` | 99% arshkon, 81% asaniczka, 0% scraped | Preferred but creates asymmetry |
| Uniform cross-period text | `description_core` | 100% all sources | Lower quality (~44% boilerplate accuracy) but consistent |

**Critical asymmetry:** 2024 Kaggle data has LLM-cleaned text; 2026 scraped has 0%. This systematically inflates apparent text-based changes. All cross-period text comparisons should use `description_core` uniformly or report both columns.

## Seniority columns

| Column | When to use | Coverage |
|--------|------------|---------|
| `seniority_llm` | After Stage 10 (pending) | 0% (all null) |
| `seniority_native` | Trend estimation (primary) | arshkon 69%, scraped 83% |
| `seniority_final` | Cross-sectional (best coverage) | arshkon 81%, scraped 93% |
| `seniority_3level` | Coarse analyses | Same as seniority_final |
| `seniority_imputed` | Never as primary | Severe mid-senior bias |

**Critical rule:** Exclude asaniczka from ALL seniority-stratified analyses.

Full details: [T02: Seniority Comparability](../reports/T02.md) | [T03: Seniority Labels](../reports/T03.md)
