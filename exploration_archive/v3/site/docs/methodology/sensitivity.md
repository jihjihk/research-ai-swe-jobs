# Sensitivity Framework

Every finding in this study is reported with a standard set of robustness checks. This page documents the sensitivity framework and summarizes which findings survive which checks.

## Standard sensitivity dimensions

### (a) Aggregator exclusion

Aggregator postings (staffing agencies, recruiting platforms) constitute 12-27% of SWE postings by source. They have different seniority profiles (higher entry rates) and may use template language. All analyses are run with and without aggregators.

### (b) Company capping

Large companies can dominate the sample with near-identical postings. Company capping (max 10-20 postings per company) prevents template dominance. Entry-level findings show < 2pp sensitivity at any capping threshold.

### (c) Seniority operationalization

The entry-level trend direction depends on which seniority column is used. All seniority-stratified analyses report results using:

- `seniority_native` (primary -- cleanest labels)
- `seniority_final` (robustness -- best coverage)
- `seniority_3level` (coarse -- junior/mid/senior)

The discrepancy between these operationalizations is the most important specification sensitivity in the study.

### (f) Within-2024 calibration

Comparing arshkon (April 2024) vs asaniczka (January 2024) sets the noise floor. Any cross-period change that does not exceed within-2024 variability cannot be confidently attributed to real market change. This calibration revealed that description length, company overlap, geographic distribution, and title vocabulary show comparable within-2024 as cross-period variability (instrument artifacts). Seniority and AI requirements clearly exceed the noise floor.

### (g) SWE classification tier

The SWE classifier has a 4-6% false positive rate. Running analyses on regex-tier-only (highest precision, smaller sample) vs the full sample tests whether boundary cases affect findings.

## Additional sensitivity dimensions

### Text source sensitivity

Analyses using text features are run on both `description_core` (uniform, lower quality) and `description_core_llm` (higher quality, 2024 only). The 2024/2026 text quality asymmetry is the most important text-related confounder.

### Domain stratification

Given that domain NMI (0.175) is 10x seniority NMI (0.018), seniority-stratified analyses should also be domain-stratified. Pooled junior-vs-senior comparisons may mask domain-specific dynamics.

### Length normalization

Description length grew 57-67% (Cohen's d = 0.77 -- the single largest effect). All text density metrics must be normalized per 1K characters. Binary presence indicators are less affected.

## Sensitivity matrix: Which findings survive which checks

| Finding | (a) Aggregator | (b) Capping | (c) Seniority | (f) Calibration | (g) SWE tier |
|---------|---------------|-------------|---------------|-----------------|-------------|
| AI requirements surge (+24pp DiD) | Pass | Pass | N/A | Pass (5-17x) | Pass |
| Junior share decline (-25pp DiD) | Pass | Pass (< 2pp) | **Mixed** | Pass | Pass |
| Domain recomposition (ML/AI +22pp) | N/A | N/A | N/A | N/A | N/A |
| AI additive to stacks (11.4 vs 7.3) | Pass | Pass | N/A | Pass | Pass |
| YOE purification (22.8% to 2.4%) | Pass | Pass | N/A | N/A | Pass |
| Management indicator (+4-10pp corrected) | Pass | Pass | N/A | Pass | N/A |
| AI-entry orthogonality (r ~ 0) | Pass | Pass | Pass | N/A | Pass |
| 57% compositional | Pass | Pass | N/A | N/A | N/A |

**"Mixed" for junior decline seniority operationalization:** seniority_native and seniority_final both show decline. seniority_3level in the overlap panel appeared to show increase. This discrepancy is the highest-priority item for resolution (via seniority_llm).

## Findings that failed sensitivity

| Finding | Failed check | Consequence |
|---------|-------------|-------------|
| Management +31pp | Pattern validation (T22) | Corrected to +4-10pp |
| Soft skills expansion | Cross-occupation DiD (T18) | Demoted: not SWE-specific |
| Junior-senior convergence | Within-2024 calibration (T15) | Abandoned: below noise floor |
| Management migration | Senior decomposition (T21) | Rejected: expanded everywhere |

## Recommended formal robustness checks for analysis phase

Each core finding should have a formal robustness table:

1. **Junior share decline:** (a)(b)(c)(f)(g) + domain stratification -- the most specification-sensitive finding
2. **AI requirements surge:** (a)(b)(f)(g) -- already well-validated; formal tables needed
3. **Domain recomposition:** text source (LLM vs rule-based), method (BERTopic vs NMF), sample (full vs random)
4. **AI-entry orthogonality:** panel size threshold, seniority operationalization, with/without aggregators
5. **57% compositional:** panel definition, era pooling (separate vs pooled 2024)
