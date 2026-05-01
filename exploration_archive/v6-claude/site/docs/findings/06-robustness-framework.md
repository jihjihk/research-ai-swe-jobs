# Finding 6 — A novel robustness framework for longitudinal posting research (methodological contribution)

**The aggregate junior-share null *under this framework* is itself a methodological contribution.** It tells us which past junior-narrowing findings will not replicate.

## Claim

A four-test robustness framework for longitudinal posting research:

1. **Within-source calibration** — calibrate SNR using independent 2024 halves before cross-period comparison. Reject any metric with SNR < 2.
2. **Specification dependence diagnostics** — require every seniority-stratified finding to hold under `seniority_final` AND a label-independent YOE ≤ 2 proxy.
3. **Macro-robustness ratio** — compute cross-period effect divided by within-scraped-window effect. Reject if ratio < 10 (i.e. the effect is within noise).
4. **Authorship-style matching** — match 2024/2026 postings on stylistic features before computing content deltas. Cite attenuation, not sign flips.

Several past junior-narrowing and management-language findings in the literature fail at least one of these tests when computed on our data. The aggregate junior-share null we report is itself a methodological contribution.

## The four tests in detail

### Test 1 — Within-source calibration (T05, T14)

Use the two 2024 sources (arshkon and asaniczka) as independent same-period halves. Compute metric means on each half and the cross-period delta. **SNR = (cross-period delta) / (within-2024 delta)**. Threshold: ≥ 2.

Example results:

| Metric | Cross-period | Within-2024 | SNR | Verdict |
|---|---|---|---|---|
| AI narrow rate | +15.97 pp | 0.017 pp | **925** | PASS |
| AI broad rate (24-term) | +23.5 pp | 1.77 pp | **13.3** | PASS |
| `seniority_final` entry share | +2.1 pp | 6.4 pp | **0.33** | **FAIL** |
| `tech_density` | flat | flat | **0.7** | FAIL |
| `requirement_breadth` | +2.39 | 0.23 | **10.2** | PASS |

### Test 2 — Specification dependence (T06/T16)

`seniority_final` (Stage 5 rule + Stage 10 LLM) and a YOE ≤ 2 proxy have **<10% row overlap** — they measure structurally different populations. On the 115-company overlap panel:

| Operationalization | Within-company entry-share change |
|---|---|
| `seniority_final` | **−3.2 pp** |
| YOE ≤ 2 | **+1.1 pp** |

Direction disagreement. Any entry-share finding that holds on only one operationalization should be cited with the specification disclaimer.

Replicates at n=240 panel (T16): −0.032 vs +0.015.

### Test 3 — Macro-robustness ratio (T19)

JOLTS info-sector openings dropped 29% between windows — macro cooling could drive anything that looks like a junior-share change. Compute:

**macro ratio = (cross-period effect) / (within-scraped-window effect).**

Threshold: ≥ 10.

| Metric | Macro ratio | Verdict |
|---|---|---|
| AI broad | **24.7×** | PASS |
| AI narrow | **17.7×** | PASS |
| copilot | 8.1× | marginal |
| Entry share | **0.86×** | **FAIL — below noise floor** |

### Test 4 — Authorship-style matching (T29)

Train a composite authorship-style score on stylistic features (em-dash frequency, bullet density, paragraph structure). Match 2024/2026 postings on the composite before computing content deltas.

Expected behavior:

- **Real content changes attenuate little** under matching.
- **Style-driven changes attenuate substantially** (or sign-flip).

Actual attenuation on content metrics:

| Metric | Raw delta | Matched delta | Attenuation | Interpretation |
|---|---|---|---|---|
| AI broad | +23.5 pp | matched | **0-7%** | Real content |
| AI narrow | +16.0 pp | matched | **0%** | Real content |
| Orchestration density | +98% | matched | 0-7% | Real content |
| `requirement_breadth` | +35% | matched | **−62%** | Mostly style |
| `tech_count` | — | matched | **−23%** | Mostly real |
| Mentor sub-pattern | — | matched | style-correlated (r≈0.09) | Partly style |
| `char_len` | +17% | matched | feature-set dependent | Cite attenuation, not sign flip |

## Supporting diagnostics

- **Concentration prediction table (T06):** for a given finding, compute how concentrated the evidence is in a small number of companies; predict whether aggregator exclusion will change the sign. T06's prediction about entry-share was correct: excluding 7 entry-specialist intermediaries drops scraped entry by 2.1 pp.
- **Precision-stratified pattern validation (T11/T21/T22):** 50-row stratified sample per pattern with a ≥80% precision threshold. T11's strict management pattern failed at 38-50%. T21 rebuilt with 13+14+9 validated patterns at 100% precision on a 100-row V2 audit.

## Figures

![T29 unifying mechanism — what attenuates, what does not](../assets/figures/T29/fig4_unifying_mechanism.png)
*T29 — the authorship-style matching test. AI rise is content-real (0-7% attenuation); length and requirement-breadth are mostly style migration (23-62% attenuation). This figure is the visual proof of Test 4 above.*

## What this framework rules out in past literature

A straightforward application to the published posting-research corpus would flag:

1. Any "junior narrowing" finding that uses a single seniority operationalization.
2. Any length-based "scope inflation" finding without authorship-style matching.
3. Any "management language rose" finding without per-pattern precision validation.
4. Any cross-period posting-content finding without macro-robustness checking.

## Task citations

- **[T05](../audit/reports/T05.md), [T14](../audit/reports/T14.md)** — within-2024 SNR calibration.
- **[T06](../audit/reports/T06.md), [T16](../audit/reports/T16.md)** — specification dependence.
- **[T19](../audit/reports/T19.md)** — macro-robustness ratio.
- **[T29](../audit/reports/T29.md)** — authorship-style matching.
- **[T11](../audit/reports/T11.md), [T21](../audit/reports/T21.md), [T22](../audit/reports/T22.md)** — precision-stratified pattern validation.
- **[V1](../audit/verifications/V1_verification.md), [V2](../audit/verifications/V2_verification.md)** — cross-agent verification passes.

## What this contribution is and is not

- It **is** a testable, reproducible framework that can be applied to any other posting dataset.
- It **is** supported by concrete thresholds (SNR ≥ 2, macro ratio ≥ 10, pattern precision ≥ 80%) derived from this exploration.
- It is **not** a replacement for domain expertise — the pattern-validation threshold still requires human precision audits.
- It is **not** universally applicable — the within-source calibration test needs two independent same-period sources, which not every labor-market dataset has.
