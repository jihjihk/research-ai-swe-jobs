# Gate 1 Research Memo

Date: 2026-04-05
Wave: 1 (Data Foundation)
Tasks completed: T01-T07

---

## What we learned

### 1. The junior share decline is the strongest finding, and it's real.

All four seniority operationalizations show a 5.9-8.3pp decline in entry-level SWE share from arshkon 2024 (~20-22%) to scraped 2026 (~14-15%). The within-company analysis is even more compelling: among 110 companies appearing in both periods, entry share drops from 25.0% to 13.2% (-11.8pp). This within-company decline is *stronger* than the aggregate, meaning company composition effects actually *dampen* the signal. New 2026 entrants have slightly higher junior share (15.9%) than returning companies (13.2%). The decline is not a composition artifact — it's real restructuring within established companies.

This is robust to:
- Company capping (<2pp sensitivity at any threshold)
- Seniority operationalization (all 4 variants agree on direction and magnitude)
- Aggregator inclusion/exclusion (direction holds, though aggregators inflate absolute entry rates)

### 2. The YOE paradox complicates the "scope inflation" narrative.

Entry-level median YOE *decreased* from 3.0 (arshkon 2024) to 2.0 (scraped 2026). This directly contradicts the hypothesis that junior roles are requiring more experience. Two possible interpretations:
- **Slot elimination + genuine entry-level residual:** Companies are eliminating the "junior but experienced" positions (3+ YOE but entry-labeled) and keeping only truly entry-level slots (0-2 YOE). Fewer junior jobs, but the surviving ones are more genuinely junior.
- **Composition effect:** Different companies are posting entry-level jobs in 2026, and they happen to require less experience.

This distinction matters enormously for the paper framing. "Junior scope inflation" may be the wrong construct — "junior slot elimination with entry-level standardization" could be more accurate. Wave 2's T08 and T11 need to investigate this deeply.

### 3. Most cross-dataset differences are measurement artifacts, but seniority is the exception.

The within-2024 calibration (T05) reveals that description length, company overlap, geographic distribution, and title vocabulary all show comparable variability within-2024 (arshkon vs asaniczka) as across periods. These are instrument differences, not market changes. Seniority is the one metric where the cross-period signal clearly exceeds the within-2024 baseline — because asaniczka's seniority labels are broken, the calibration doesn't apply, but the arshkon-vs-scraped comparison uses comparable labeling mechanisms and shows a real shift.

### 4. The data is substantially larger than documented.

Scraped LinkedIn SWE = 24,095 (not 4,500). Scraped entry-level = 3,255 (not 574). Total SWE = 52,327. This dramatically improves power for nearly all analyses.

### 5. The infrastructure for cross-period comparison is solid despite constraints.

Power analysis confirms well-powered comparisons for entry-level (MDE d=0.109), senior (d=0.055), all SWE (d=0.044), and the company panel (d=0.073). Geographic representativeness is excellent (r>0.97 vs BLS OES). 18 metros are feasible at >=50 per period. The binding constraints are: no LLM-cleaned text for scraped, no company size data for cross-period, no remote work comparison, and asaniczka is unusable for seniority analysis.

---

## What surprised us

1. **Within-company decline (-11.8pp) exceeds aggregate (-5.9pp).** We expected composition effects to inflate the aggregate; they actually dampen it. New 2026 companies are more junior-friendly than returning ones. This strengthens the restructuring narrative considerably.

2. **Entry-level YOE decreased.** The "scope inflation" hypothesis predicts increasing YOE requirements for juniors. The opposite happened. Either scope inflation is wrong, or it's manifesting in dimensions other than YOE (e.g., technical breadth, organizational scope language).

3. **Description core_length grew more than raw description_length (67% vs 57%).** Boilerplate removal is less effective on longer scraped descriptions. The 44% accuracy of rule-based boilerplate removal is a bigger problem than anticipated for scraped data.

4. **The rule-based seniority imputer is nearly useless for non-mid-senior classes.** Entry accuracy: 21.2% (arshkon), 41.9% (scraped). Associate: 8.7-9.4%. Director: 15.5-21.1%. The imputer's value is almost entirely in returning "unknown" and triggering native backfill.

5. **74 native entry-level rows are overridden to mid-senior by title keywords.** Platform says entry, title says senior. These are either platform labeling errors or genuine mismatches that deserve investigation.

---

## Evidence assessment

| Finding | Evidence strength | Sample | Potential confounds | Survives calibration? |
|---------|-------------------|--------|--------------------|-----------------------|
| Junior share declined 5.9-8.3pp | **Strong** | 4,066 vs 22,469 known-seniority | Company composition, label provenance | Yes (within-company -11.8pp) |
| Within-company decline is stronger | **Strong** | 110 companies, 1,264 vs 4,970 postings | Overlap companies biased toward large firms | N/A (controls for composition) |
| YOE paradox (entry YOE decreased) | **Moderate** | 476 vs 1,066 entry with YOE | YOE extractor behavior on different text formats, company composition | Unknown (needs Wave 2) |
| Description length grew 57-67% | **Strong** (as artifact) | Full sample | None — this IS the artifact | Confirmed artifact (within-2024 KS=0.05 for core_length) |
| SWE classification 4-6% FP | **Moderate** | Manual audit of 50 borderline rows | Small audit sample | N/A |
| Aggregators inflate entry rates | **Strong** | 591-6,340 aggregator rows per source | None | Yes |
| Asaniczka associate != junior proxy | **Strong** | 2,014 asaniczka associate SWE | None | Yes (multiple signals agree) |

---

## Narrative evaluation

### RQ1: Junior scope inflation
**Status: PARTIALLY CONFIRMED, NEEDS REFRAMING.**

The junior *share* decline is confirmed and robust. But "scope inflation" as originally conceived — junior roles requiring more — is challenged by the YOE paradox. The evidence so far is more consistent with "junior slot elimination" than "junior scope inflation." Wave 2's text analysis (T11, T12, T13) will determine whether the remaining junior roles are actually asking for more (scope inflation) or whether the story is purely about quantity reduction.

**Proposed reframing for RQ1:** "How did employer-side SWE hiring restructure across seniority levels from 2024 to 2026?" — removing the presumption of "scope inflation" and letting Wave 2 determine whether the change is in quantity, in requirements, or both.

### RQ2: Task and requirement migration
**Status: CANNOT EVALUATE YET.** Needs Wave 2 text analysis. Infrastructure is in place.

### RQ3: Employer-requirement / worker-usage divergence
**Status: CANNOT EVALUATE YET.** Needs Wave 3.

### RQ4: Mechanisms (qualitative)
**Status: NOT PART OF EXPLORATION.** But the YOE paradox and within-company decline are strong candidates for interview probes.

### Is the initial narrative the best framing?
**Too early to tell, but the YOE paradox is a warning signal.** If Wave 2 shows that the remaining junior roles are NOT asking for more (i.e., scope inflation is not happening), then the paper's lead finding shifts from "junior scope inflation" to "junior hiring reduction within established companies." That's still a strong finding — possibly stronger, because it's a cleaner claim about labor demand — but it's a different story.

---

## Emerging narrative

The data tells a story about **quantity, not (yet) quality.** Established companies are posting fewer entry-level SWE roles in 2026 than 2024, and the remaining entry-level postings actually require *less* experience. The junior hiring reduction is real, within-company, and robust to specification choices. Whether this is accompanied by scope inflation (requiring more of juniors despite lower YOE bars) or by genuine role elimination remains the central question for Wave 2.

---

## Research question evolution

**Current RQ set (unchanged from initial design, but with caveats):**
- RQ1: Reframe from "junior scope inflation" to "seniority-level restructuring" — let Wave 2 determine whether it's scope inflation, slot elimination, or both
- RQ2: No change needed yet
- RQ3: No change needed yet
- RQ4: No change needed yet

**Potential new RQ emerging:** The within-company vs between-company decomposition could be a core contribution. RQ1b: "Is the seniority shift driven by within-firm restructuring or by changes in which firms are hiring?"

---

## Gaps and weaknesses

1. **No LLM-cleaned text for scraped data.** All cross-period text analysis must use rule-based `description_core` (~44% accuracy). This is the biggest quality constraint.
2. **Entry-level YOE paradox unexplained.** Could be real or artifact. T08 must investigate.
3. **Asaniczka is useless for seniority trends** but dominates the 2024 sample by size. Need to be careful about inadvertently including asaniczka in seniority-stratified analyses.
4. **Company overlap is 18% of companies.** The within-company panel (202 companies) is robust but biased toward large firms. Results from the panel may not generalize to smaller companies.
5. **No 2024-01 entry-level baseline.** We can't tell if the decline was already underway in Jan 2024 vs Apr 2024 because asaniczka has no entry labels. The two data points (Apr 2024, Mar 2026) could represent a gradual trend or an abrupt shift.

---

## Direction for Wave 1.5 and Wave 2

### Wave 1.5 (Shared Preprocessing)
Proceed as planned. Key decision: **use `description_core` uniformly for all sources** (not `description_core_llm`), since LLM-cleaned text is unavailable for scraped data. This ensures consistent text quality across periods.

### Wave 2 modifications

**T08 (Distribution profiling):** Add emphasis on:
- Deep investigation of the YOE paradox: break down by title, company type, aggregator status, seniority label provenance
- Test whether the YOE decrease is an artifact of different entry-level composition or a real requirement change

**T09 (Archetype discovery):** No modifications needed. This is the methods laboratory — let it run openly.

**T11 (Requirements complexity):** This is now the MOST CRITICAL Wave 2 task. It directly tests whether "scope inflation" is real beyond the junior share decline. If entry-level requirement complexity *increased* despite lower YOE, that's scope inflation. If it didn't, the paper leads with slot elimination.

**T12 (Text evolution):** Ensure within-2024 calibration is executed. Use `description_core` uniformly.

**T13 (Linguistic evolution):** Critical for interpreting the 57-67% length growth. The stacked section analysis (what grew: requirements vs boilerplate?) is essential.

**T14, T15:** No modifications.

---

## Current paper positioning

If we stopped after Wave 1, the best paper we could write is:

> "Using 52K SWE job postings from LinkedIn spanning 2024-2026, we document a robust 6-12 percentage point decline in entry-level posting share, concentrated within established companies. This within-firm restructuring is accompanied by a paradoxical decrease in entry-level experience requirements, suggesting employers are eliminating junior slots rather than inflating their scope."

This is a clean, credible descriptive finding. Wave 2 needs to deliver:
- Whether the remaining junior roles changed in content (scope inflation or not)
- What the dominant structural change is in the posting space (T09 archetypes)
- Whether senior roles shifted toward orchestration (T21 will test this in Wave 3)
- Whether these changes are SWE-specific or field-wide (T18 in Wave 3)

The paper's potential ceiling rises significantly if Wave 2 reveals that the content of postings changed alongside the composition — and rises even further if we can decompose the story into within-company vs between-company components with compelling detail.
