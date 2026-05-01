# Gate 1 Research Memo

**Date:** 2026-04-09 (revised 2026-04-10 after seniority measurement deep-dive)
**Wave completed:** Wave 1 (T01-T07), Agents A-D, plus seniority measurement investigation
**Reports:** T01.md through T07.md, seniority_measurement_investigation.md

> **Revision note:** The original Gate 1 memo concluded that the entry-level decline was a robust finding anchored by the 90% within-company decomposition. After dispatching a deep-dive on seniority measurement (in response to a question about whether `seniority_llm` was meant to be combined with `seniority_final` rather than read alone), that conclusion no longer holds. The seniority investigation found that LinkedIn's native entry labels in arshkon are substantially contaminated (41% of native-entry rows have YOE >= 5 years, vs only 9.5% in scraped), and every label-independent method shows entry share is approximately stable or modestly increased. This memo has been rewritten to reflect the updated understanding. The original audit trail is preserved in the seniority investigation report and in the schema/preamble/orchestrator edits made on 2026-04-10.

---

## What we learned

### 1. The entry-share trend is approximately stable, not declining

The single most important discovery from Wave 1 is **not** that entry-level share declined. It is that the apparent decline depends entirely on whether you use measurement methods that inherit LinkedIn's native seniority labels. Once you control for differential native-label quality across data snapshots — using YOE as a label-independent proxy or constructing the combined best-available column — entry share is approximately stable.

| Method | Arshkon 2024 | Scraped 2026 | Direction | Notes |
|---|---|---|---|---|
| `seniority_native` (non-null) | 22.3% | 13.7% | DECREASE -8.7pp | Inherits arshkon contamination |
| `seniority_final` (known) | 21.0% | 14.2% | DECREASE -6.8pp | Inherits arshkon contamination via native_backfill |
| YOE <= 2 proxy | 15.0% | 16.6% | INCREASE +1.6pp | Label-independent |
| YOE <= 3 proxy | 30.8% | 33.6% | INCREASE +2.8pp | Label-independent |
| `seniority_imputed` (known, text-only) | 6.8% | 6.7% | FLAT | Text-only, uniform method |
| Explicit entry signals (text) | 2.1% | 2.6% | INCREASE +0.5pp | Text-only |

The smoking gun: **41% of arshkon's native-entry SWE postings have `yoe_extracted >= 5`**, vs only 9.5% in scraped. Arshkon's "entry" pool has a bimodal YOE distribution with a spurious cluster at 5+ years inflating the 2024 baseline by ~40%. This contaminated baseline drives the entire native-label-based decline. Within explicit-entry rows (where the title or description contains an entry signal), the YOE profile is consistent across periods (mean ~2.0-2.3, median 2.0).

### 2. Employer labeling explicitness increased substantially

A real and well-supported finding: 2026 employers are more likely to put explicit seniority signals in posting titles and descriptions.

- `seniority_imputed` known rate (text-only signal coverage): 44.6% → 55.2% (+10.6pp)
- Junior-specific title keywords: 1.8% → 3.6% (nearly doubled)
- "entry" keyword in title: 0.10% → 0.33% (3x)
- "intern" in title: 0.94% → 1.40%
- "associate" in title: 0.68% → 1.21%

This is a finding about **how employers communicate seniority**, not directly about the demand for junior workers. It's potentially novel and publishable as a methodological observation about the labor market data ecosystem.

### 3. The within-company decomposition needs reinterpretation

T06 found that among 84 overlap companies, native-label entry share fell from 28.7% to 12.4%, with 90% of the decline within-company. This was originally framed as the strongest evidence of real entry restructuring. With the new understanding, that interpretation is suspect:

- The decomposition was computed under `seniority_native` and `seniority_final`, which inherit arshkon's contaminated entry labels
- The "within-company decline" may largely be the same companies' arshkon postings being mislabeled entry-when-actually-mid-senior, then those same companies' scraped postings being labeled correctly
- The decomposition methodology is sound; the input labels are biased

**Action:** T16 (Wave 3) is now spec'd to re-run the decomposition under the combined best-available column AND under the YOE-based proxy. We will get the corrected version then. Do not cite the 90% within-company finding as a headline result until T16 confirms it under label-independent methods.

### 4. Description length growth (~57%) remains a strong, real finding

Calibration ratio from Agent Prep's table: 2.3x cross-period vs within-2024. This is well above the noise floor and is likely a real signal. T13 in Wave 2 should determine whether the growth is in requirements sections (real signal about job complexity) or in benefits/boilerplate sections (cosmetic).

### 5. The sample is geographically representative and cross-occupation comparisons are well-powered

T07 confirmed r=0.985 vs BLS OES, and the SWE-vs-control comparisons (T18) have ample power. These are the validity backbone for any cross-occupation framing.

---

## What surprised us

1. **The headline "junior decline" finding was a measurement artifact.** This is the most consequential surprise of Wave 1. Going in, we expected to confirm or moderately qualify the entry-level decline. Instead, every label-independent measure shows stability or modest increase.

2. **Arshkon native labels are systematically lower-quality than scraped native labels.** We did not anticipate that LinkedIn's classifier would have differential accuracy across snapshots. The bimodal YOE distribution within arshkon's native-entry pool (one cluster at 1-3 years, one spurious cluster at 5+) is striking.

3. **Employers became more explicit about seniority.** The +10.6pp jump in text-only seniority detection rate is a clean, well-powered finding that wasn't on our radar.

4. **The Stage 10 routing design was misframed in the schema.** The schema called `seniority_llm` "primary" without making clear that it's NULL by design for `rule_sufficient` rows. We (and the Wave 1 agents) read `seniority_llm` and `seniority_final` as competing options when they're halves of one variable. The schema, task reference, and orchestrator prompt have been rewritten to fix this for future runs.

5. **Aggregators are 46% entry in arshkon vs 18% direct.** Aggregator share fell from 27.3% (asaniczka) to 14.3% (scraped). This composition shift is itself a confound on any seniority-stratified comparison and needs aggregator-exclusion sensitivity in Wave 2.

---

## Evidence assessment

| Finding | Strength | Notes |
|---|---|---|
| Entry share approximately stable (label-independent) | **Strong** | YOE proxy is fully label-free; explicit signals confirm; multiple thresholds agree |
| Arshkon native-entry labels are contaminated | **Strong** | 41% YOE>=5 vs 9.5% in scraped is a large, clean diagnostic |
| Employer labeling explicitness increased | **Strong** | +10.6pp known-rate jump, replicated across multiple keyword categories |
| Description length grew ~57% | **Strong** | 2.3x calibration ratio, large effect |
| Native-label entry share declined | Real but **biased** | The decline is statistically real but reflects label noise, not market change |
| 90% within-company decline (T06) | **Reframed** | Methodologically sound but inherits the contaminated labels; defer to T16 re-run |
| SWE classifier quality (91% regex tier) | **Strong** | Sample is trustworthy |
| Geographic representativeness (r=0.985) | **Strong** | Sample frame is credible |

---

## Narrative evaluation

### RQ1 (employer-side restructuring): **Reframed**

The original RQ1 hypothesis was junior scope inflation and senior archetype shift. The data so far supports neither cleanly:
- **Junior share decline** is not supported once you correct for native-label quality. The data is consistent with stability or modest increase.
- **Junior scope inflation** has not yet been measured (Wave 2 task). The earlier YOE finding (entry median 3.0 in arshkon, 2.0 in scraped) is now explained: arshkon entry pool is contaminated with mid-senior rows; the "decline" in entry YOE is the contamination clearing.
- **Senior archetype shift** is also untested until Wave 2 (T21).

The most likely Wave 1 narrative for RQ1 is now: *"Entry-level posting share was approximately stable from 2024 to 2026. Methods that depend on LinkedIn's platform-provided seniority labels show an apparent decline, but this is attributable to differential native-label quality across data snapshots, validated by years-of-experience requirements. What did change is employers' explicit signaling of seniority in posting titles."*

### RQ2 (task migration): **Open**
Wave 2 task. No findings yet.

### RQ3 (employer-usage divergence): **Wave 3 task**
Awaiting T23 + benchmark data.

### RQ4 (mechanisms): **Not constrained by data**
Interview design will be informed by the new measurement findings — particularly the labeling-explicitness shift.

---

## Emerging narrative

The data is telling a different story than the initial hypothesis. The narrative is:

> Between 2024 and 2026, the SWE posting market grew substantially in volume but did not experience a meaningful shift in the entry-level share of postings. Findings that suggest otherwise are driven by differential quality of LinkedIn's platform seniority labels across data snapshots — a measurement artifact that label-independent (YOE-based) methods do not show. What did change: employers became significantly more likely to put explicit seniority signals in posting titles, postings grew ~57% longer, and (TBD in Wave 2) the content of postings appears to have shifted in technology mix and possibly in scope language. The paper's contribution shifts from "documenting the junior decline" to two distinct contributions: (1) a methodological warning about cross-temporal comparisons using platform seniority labels, validated by YOE-based independent evidence, and (2) an empirical characterization of what actually changed in posting content and structure during the AI coding tool adoption window.

This is more interesting than the original framing. "Nothing changed in junior share" plus "everything changed in how postings are written" is a richer story than "junior roles disappeared."

---

## Research question evolution

**No formal RQ changes yet** — those should wait for Wave 2 evidence on text content. But two reframings are emerging:

1. **RQ1 should be reframed from "did junior share decline?" to "what restructured?"** The entry-share question remains a sub-question, but it's no longer the headline. The headline becomes the broader content/structure restructuring.

2. **A new methodological RQ is emerging:** "How should longitudinal labor market posting studies measure seniority when platform classifiers have differential accuracy across snapshots?" This would be a contribution to the empirical methods literature.

3. **A new substantive sub-finding is emerging:** "Employer labeling explicitness about seniority increased significantly between 2024 and 2026." This wasn't in the original RQs at all and may belong as a standalone finding.

---

## Gaps and weaknesses

1. **T06's within-company decomposition uses contaminated inputs.** Will be addressed in T16 (Wave 3).

2. **The seniority_llm sample is small** (~9.6K labeled SWE rows, ~80% unknown within labeled). The combined column is functional but not high-resolution. An additional LLM budget allocation would help, especially for entry-level cells.

3. **Asaniczka's role as 2024 baseline depends on operationalization.** Native-dependent methods exclude it; combined column and YOE proxy include it. Wave 2 must report under both.

4. **JOLTS hiring-cycle confound (arshkon at trough) was not directly addressed by Wave 1.** It could affect mid-senior comparisons if the hiring mix shifts during a trough.

5. **We have not yet looked at posting content.** The biggest gap going into Wave 2.

---

## Direction for Wave 2

The seniority finding **does not change which Wave 2 tasks to run**, but it changes how the agents should frame and interpret entry-level results. Specific guidance for each agent:

**Agent E (T08 — distribution profiling):**
- Use the combined best-available column as primary; report YOE proxy alongside
- The "YOE paradox" task step has been replaced with a "native-label quality diagnostic" step (already updated in the task spec)
- Expect entry-share to be approximately stable; report any direction with the full ablation set

**Agent F (T09 — archetype discovery):**
- Sample stratification by seniority should use the combined column, not seniority_final alone
- The entry-share-by-archetype analysis needs the YOE proxy as a co-equal validator
- The "do clusters align with seniority?" question is now sharper: do they align with the combined column, with YOE-based junior status, or with neither?

**Agent G (T10/T11 — title evolution + requirements complexity):**
- The "credential stacking" question is now critical, since it's one of the candidate explanations for the labeling-explicitness shift
- Entry-level scope inflation analysis should use YOE-based entry definition (`yoe_extracted <= 2`) as the primary, with the combined column as alternative
- Pay attention to whether arshkon's entry-level requirements match its mid-senior requirements (which would directly explain the contaminated labels)

**Agent H (T12/T13 — text evolution):**
- The 57% length growth is confirmed real; T13 must determine which sections grew
- The Fightin' Words comparison should be the headline of Wave 2 — "what actually changed in posting content"
- The labeling-explicitness finding suggests boilerplate sections may have grown along with requirements; section anatomy is the diagnostic

**Agent I (T14/T15 — technology + semantic landscape):**
- The tech_count calibration ratio (24.8x) and AI keyword calibration ratio (-16x) suggest these are large real signals
- T14's tech ecosystem mapping is now likely the strongest substantive Wave 2 finding
- T15's semantic similarity analysis should also use the combined column for seniority, plus a YOE-based stratification

---

## Current paper positioning

**The paper positioning has shifted.** With the entry-share decline finding withdrawn, the strongest candidate positioning is now:

**Mixed dataset/methods + empirical paper.** Lead with two contributions:
1. **Methodological:** Cross-temporal comparisons using platform seniority labels are unreliable when label quality differs across snapshots. We present a measurement framework (combined LLM+rule routing column + YOE-based label-independent validation) and document how an apparent 8pp entry-share decline becomes approximately stable under the corrected methodology.
2. **Empirical:** What actually changed in SWE postings between 2024 and 2026, drawn from text evolution (T12/T13), technology mix (T14), and the labeling-explicitness shift.

This is a stronger and more honest contribution than the original "junior scope inflation" framing. It's also more publishable because the methodological warning has broader applicability (any longitudinal labor market study using platform classifiers).

**If we stopped here,** the paper would be: "We document that LinkedIn's platform seniority labels have differential accuracy across data snapshots, and propose a label-independent validation framework using YOE requirements. Applied to 2024 and 2026 SWE postings, our framework shows that the apparent 8pp decline in entry-level share under platform labels is largely a measurement artifact: entry-level demand was approximately stable. We also document that employers' explicit signaling of seniority increased substantially over the same period." Wave 2 will determine whether this is the final positioning or whether stronger substantive content findings (text/tech evolution) become the lead instead.
