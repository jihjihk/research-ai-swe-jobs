# Sensitivity framework

Every headline finding in this package has passed a four-test robustness framework described in detail at [Findings → Robustness framework](../findings/06-robustness-framework.md). This page is a quick reference.

## The four tests

1. **Within-source calibration** (T05, T14) — arshkon vs asaniczka as independent 2024 halves. SNR threshold: **≥ 2**.
2. **Specification dependence** (T06, T16) — does the finding hold under both `seniority_final` AND a YOE ≤ 2 proxy? If direction flips across operationalizations, cite the specification dependence disclaimer.
3. **Macro-robustness ratio** (T19) — cross-period effect divided by within-scraped-window effect. Threshold: **≥ 10**.
4. **Authorship-style matching** (T29) — match on stylistic composite before computing content deltas. Cite attenuation, not sign flips.

## Mandatory sensitivity requirements per finding

| Finding | Must include |
|---|---|
| **RQ3 divergence** | Worker-rate sensitivity 50-85%; section-stratified hard-AI rate; narrow vs broad split (NOT combined); aggregator exclusion |
| **Senior orchestration** | Pattern precision tables (per-pattern); top-10-company exclusion for director; 50-row audit sample CSV; cross-seniority non-monotonic diagnostic |
| **Tool-stack restructuring** | Within-company Oaxaca decomposition; cluster stability across random seeds; 92% within-company at both ≥3 AND ≥5 thresholds; SWE-vs-adjacent DiD CI |
| **Credential-stack convergence** | 10/10 convergence under T28 pattern AND V2 6-category pattern; flip-count range [2-7] explicitly stated; style-matched delta |
| **Archetype pivot rate** | ≥3 and ≥5 thresholds; 30.5% scraped coverage caveat |
| **Length null** | T29 style matching under composite AND em-dash-only AND bullet-only; cite attenuation (23-62%) not the sign-flip |
| **Junior-share null** | Within-2024 SNR 0.33; arshkon-only entry flip; macro ratio 0.86×; denominator drift; specification-dependence |

## Things you should NEVER cite

From Gate 2 V1 and Gate 3 V2 narrowings:

- The specific **−411 char** style-matched flip (feature-set dependent, V2 narrowing 1).
- The **"7/10 sign flip"** number (pattern-dependent, V2 narrowing 2).
- The **108× ChatGPT** per-tool ratio (denominator near zero, V2 narrowing 3).
- **"SNR 925" combined with "5.15 → 28.63"** (cross-wired metric, V1 correction 1).
- **"Staff is the new senior"** (within-company rise of staff absorbs only ~22% of senior drop).
- **"Seniority levels blurred"** (3/4 boundaries sharpened, T20).

## Seniority validation rules

From T03, T06, T08 and verified across V1/V2:

- **Use `seniority_final` as primary, ALWAYS paired with the YOE ≤ 2 proxy.** Never rely on `seniority_native` pooled across sources.
- `seniority_final` and YOE ≤ 2 proxy have **< 10% row overlap** — they measure structurally different populations.
- On the 115-company overlap panel, the two operationalizations **disagree in direction** on within-company entry-share change (−3.2 pp vs +1.1 pp).
- Arshkon native `entry` rows have mean YOE **4.18** and **42.6% at YOE ≥ 5** — the 2024 native-label pool is majority non-entry by any reasonable YOE definition.
- **Asaniczka has zero native entry labels** — `seniority_native` is structurally unable to detect entry there.
- **53% of scraped SWE is `seniority_final = 'unknown'`** because Stage 10 LLM budget was capped. "Of known" denominators drift: 61% → 47% between periods.

**Practical implication:** every aggregate junior-share headline in the literature based on a single seniority operationalization is a specification-dependent claim.

## Precision-stratified pattern validation

From T11, T21, T22:

- Any keyword/regex indicator must have a **50-row stratified precision sample** per pattern, with a **≥ 80% threshold**.
- T11's original strict management pattern failed at 38-50% semantic precision — it looked like "100% precision" only because the test was tautological (matching rows that contained the pattern by construction).
- T21's rebuilt management pattern bundle has **13 people-management + 14 technical-orchestration + 9 strategic-scope patterns**, all ≥ 80% precision; several at 100%. Stored at `exploration/artifacts/shared/validated_mgmt_patterns.json`.
- The T21 bundle was **100% precision on a 100-row V2 audit** (50 per period).
- T11/T21/T22 dropped several patterns that failed: `stakeholder` (50%), `prioritization` (12%), `roadmap` (70%), `cross_functional_alignment` (70%), `guardrails` (62%), `coach` (60%), `ownership_bare` (60%), `vision` (74%).
