# Exploration Synthesis

**The one document the analysis agent reads first.**

Date: 2026-04-05
Exploration tasks: T01-T26 complete (23 reports, 4 gate memos)
Input data: `data/unified.parquet` (99 columns, 1,395,790 rows)

---

## 1. Data Quality Verdict by RQ

### RQ1: SWE labor demand restructuring

| Dimension | Quality | Notes |
|---|---|---|
| Junior share trend | **Good** (with caveats) | 5,019 arshkon + 24,095 scraped SWE. Entry-level trend direction is robust using seniority_native (arshkon-only 2024 baseline: 22.3% to 14.0%). CRITICAL CAVEAT: direction may reverse with seniority_3level in the overlap panel (T16 surprise). seniority_llm (planned) will resolve. |
| AI requirements | **Strong** | Binary keyword detection on full `description` is robust across all sensitivity checks. Calibration ratio 5-17x (T08). Not aspirational (T22). |
| Scope inflation | **Moderate** (corrected) | Management indicator corrected from +31pp to +4-10pp (T22). Must use strict/validated pattern set. Text quality asymmetry: 2024 has LLM-cleaned text, 2026 does not -- inflates apparent change. |
| Domain recomposition | **Strong** | T09 clustering is method-robust (ARI >= 0.996). ML/AI archetype 4% to 27% is the largest structural change. |
| Within-firm vs compositional | **Strong** | T16 shift-share decomposition: 57% compositional, 43% within-firm. 451-company overlap panel. |

### RQ2: Technology ecosystem evolution

| Dimension | Quality | Notes |
|---|---|---|
| Technology mention rates | **Strong** | 146-tech regex dictionary on full `description`. Within-2024 calibration validates AI signals (5-17x noise). |
| Co-occurrence networks | **Strong** | Phi-coefficient networks with Louvain clustering. 2024 and 2026 networks both have >460 edges. |
| Stack diversity | **Strong** | Mean techs/posting 6.2 to 8.3. Robust to aggregator exclusion and company capping. |
| AI additive pattern | **Strong** | AI-mentioning postings: 11.4 techs vs 7.3 non-AI. Clear and large. |

### RQ3: Employer-requirement / worker-usage divergence

| Dimension | Quality | Notes |
|---|---|---|
| Posting-side AI rates | **Strong** | Well-measured at ~41% (2026). |
| Usage-side benchmarks | **Moderate** | External surveys (StackOverflow, JetBrains, GitHub) with different sampling frames. 2026 estimate (75%) is extrapolated. |
| Divergence computation | **Moderate** | Direction is clear (posting lags usage). Exact gap depends on benchmark choice. |
| Aspiration assessment | **Strong** | T22 hedge-fraction method validated. AI is less hedged than traditional requirements. |

### RQ4: Mechanisms (qualitative)

| Dimension | Quality | Notes |
|---|---|---|
| Interview artifacts | **Ready** | T25 produced 6 artifacts with corrected findings. Three interview themes prioritized. |
| Stimuli grounded in data | **Strong** | JD pairs, company panel, and visualizations all derived from the parquet data. |

---

## 2. Recommended Analytical Samples

### Primary SWE sample

```sql
SELECT * FROM read_parquet('data/unified.parquet')
WHERE source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
  AND is_swe = true
-- Yields: ~52,327 rows (arshkon 5,019 + asaniczka 23,213 + scraped 24,095)
```

### Entry-level analysis (arshkon-only 2024 baseline)

```sql
-- 2024 entry: arshkon ONLY (asaniczka has zero native entry labels)
SELECT * FROM <primary_sample>
WHERE source = 'kaggle_arshkon' AND seniority_native = 'entry'
-- Yields: ~769 rows (seniority_native) or ~830 rows (seniority_final)

-- 2026 entry:
SELECT * FROM <primary_sample>
WHERE source = 'scraped' AND seniority_native = 'entry'
-- Yields: ~2,789 rows (seniority_native) or ~3,255 rows (seniority_final)
```

### Cross-occupation DiD sample

```sql
SELECT * FROM read_parquet('data/unified.parquet')
WHERE source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
  AND (is_swe = true OR is_swe_adjacent = true OR is_control = true)
-- Yields: ~214K rows (SWE ~52K + adjacent ~20K + control ~142K)
```

### Company overlap panel

```sql
-- Companies with >= 3 SWE in both eras (per T16 methodology)
-- Pooling 2024-01 + 2024-04 as "2024"; 2026-03 as "2026"
-- Yields: ~451 companies, ~22,929 postings (44% of all SWE)
```

### Text analysis recommendations

| Use case | Column | Coverage | Notes |
|---|---|---|---|
| Binary keyword presence | `description` (full text) | 100% all sources | Best recall; includes boilerplate |
| Density/frequency metrics | `description_core_llm` > `description_core` | LLM: 99% arshkon, 81% asaniczka, 0% scraped; Core: 100% all | LLM-cleaned preferred but creates asymmetry |
| Uniform cross-period text | `description_core` | 100% all sources | Lower quality (~44% boilerplate accuracy) but consistent |
| Embeddings (pre-computed) | all-MiniLM-L6-v2, 384-dim | Available for SWE sample | Used in T09, T15 |

---

## 3. Seniority Column Recommendation

### Ablation framework (in priority order)

| Priority | Column | When to use | Coverage | Notes |
|---|---|---|---|---|
| 1 | `seniority_llm` | After Stage 10 budget allocated | Currently 0% (all null) | Explicit-signal-only LLM classification. Will be the canonical column. |
| 2 | `seniority_native` | Trend estimation (primary) | arshkon 69%, asaniczka 100%*, scraped 83% | Cleanest platform labels. *Asaniczka has only mid-senior/associate. |
| 3 | `seniority_final` | Cross-sectional analysis (best coverage) | arshkon 81%, asaniczka 100%, scraped 93% | Merges imputed + native backfill. |
| 4 | `seniority_3level` | Coarse analyses | Same as seniority_final | junior/mid/senior/unknown. Associate standalone unusable (9% accuracy). |
| 5 | `seniority_imputed` | Never as primary | arshkon 44%, asaniczka 69%, scraped 54% | Severe mid-senior bias. Only 21-42% entry-level accuracy. |

### Critical rule: Exclude asaniczka from ALL seniority-stratified analyses

Asaniczka has zero native entry-level labels. Its `seniority_final` entry rate (0.6%) is a title-keyword imputation artifact. Including asaniczka in seniority trends creates false precision and potentially reverses the direction of entry-level findings.

### The operationalization discrepancy

The direction of the entry-level trend depends on column choice:
- `seniority_native` (arshkon-only 2024): 22.3% to 14.0% = **DECLINE** of 8.3pp
- `seniority_final` (arshkon 2024): 20.4% to 14.5% = **DECLINE** of 5.9pp
- `seniority_3level` in T16 overlap panel: appeared to show **INCREASE** (3.4% to 13.5%)

The T16 increase is likely driven by different operationalization and panel composition (pooled 2024 includes asaniczka). Resolution requires `seniority_llm`.

**Recommendation for analysis:** Use `seniority_native` as primary for entry-level trends (arshkon-only baseline). Report `seniority_final` as robustness. Flag the discrepancy explicitly and note that `seniority_llm` will resolve it.

---

## 4. Known Confounders

### Severity: HIGH

| Confounder | Description | Affected analyses | Mitigation |
|---|---|---|---|
| **Description length growth** | 57-67% median growth from 2024 to 2026. Cohen's d = 0.77 -- the single largest effect. | All text-based metrics | Length-normalize (per 1K chars). Use binary presence for keyword analyses. |
| **Asaniczka seniority gap** | Zero native entry-level labels. 0.6% entry rate is imputation artifact. | Any seniority-stratified analysis including asaniczka | Exclude asaniczka from seniority analyses. Use arshkon-only as 2024 entry baseline. |
| **Company composition** | 57% of aggregate change is compositional (T16). Only 18% company overlap between periods. | All aggregate cross-period comparisons | Use within-company panel design (451 companies) as primary. Report aggregate as robustness. |
| **Text quality asymmetry** | 2024 Kaggle data has LLM-cleaned text (80-99% SWE coverage). 2026 scraped has 0%. Rule-based cleaning retains more boilerplate in 2026. | All text-based indicators, especially management/scope patterns | Use `description_core` uniformly for cross-period comparisons. When LLM-cleaned scraped becomes available, re-run. |

### Severity: MODERATE

| Confounder | Description | Affected analyses | Mitigation |
|---|---|---|---|
| **Field-wide posting evolution** | Management, leadership, soft skills expansion is NOT SWE-specific (T18 DiD ~0). | Scope inflation claims | Use cross-occupation DiD to isolate SWE-specific effects. Only claim SWE-specificity for metrics with large positive DiD. |
| **Aggregator contamination** | Aggregators are 12-27% of SWE postings by source. Different seniority profiles (higher entry rates). | Seniority distribution, text patterns | Run all analyses with and without aggregators. Report both. |
| **Measurement error in keyword indicators** | T22 showed "leading" was 99.4% adjective usage, not management. Broad pattern sets capture boilerplate language. | Any regex-based indicator, especially management/scope | Use validated/strict pattern sets only. Document pattern validation. |
| **Seniority operationalization** | Entry-level trend direction depends on column choice (see Section 3). | RQ1 entry-level findings | Use seniority_native as primary; seniority_final as robustness. Await seniority_llm. |

### Severity: LOW

| Confounder | Description | Affected analyses | Mitigation |
|---|---|---|---|
| **SWE classification noise** | 4-6% false positive rate; QA/test engineer boundary unstable. | SWE sample definition | Report sensitivity with regex-tier-only (highest precision). |
| **Geographic sampling design** | Scraped data uses 26-metro targeted search; Kaggle is national. | Geographic comparisons | Use metro-level fixed effects. All 26 metros show same direction (T17). |
| **Scraper first-day backlog** | 1.6x normal volume on first scrape day (Mar 20). | Within-March temporal analyses | Mild; daily metric CVs < 10% (T19). |

---

## 5. Discovery Findings

### Confirmed (with caveats)

| Finding | Evidence | Key caveat |
|---|---|---|
| **AI requirements surged, SWE-specific** | 8% to 33% (broad) or 3.9% to 27.5% (entry). DiD vs control: +24.4pp. Validated as genuine, not aspirational (T22). | Binary keyword detection; specific-tool rates lower (LLM 10%, Copilot 4%, Claude 4%) |
| **Junior share declined, SWE-specific** | seniority_native: 22.3% to 14.0% (-8.3pp). DiD vs control: -24.9pp. Control junior share INCREASED. | Direction depends on seniority operationalization. Thin arshkon-only 2024 baseline (769 native entry). |
| **Domain recomposition** | ML/AI archetype 4% to 27% (+22pp). Frontend/Web 41% to 24% (-17pp). Domain NMI 10x seniority NMI. | Text source sensitivity (LLM vs rule-based): cosine similarity 0.88, not perfect. |
| **AI is additive to stacks** | AI-mentioning postings: 11.4 techs vs 7.3. Stack diversity increased 6.2 to 8.3. New 25-tech AI community emerged (T14). | Aggregators slightly inflate non-AI counts; pattern robust to exclusion. |
| **YOE slot purification** | 5+ YOE entry-level: 22.8% to 2.4%. Median entry YOE: 3.0 to 2.0. Confirmed across 5 independent decompositions (T08). | Not yet tested in cross-occupation DiD. |
| **GenAI accelerated 8.3x** | Within-2024 rate: +1.2pp/yr. Cross-period rate: +10.2pp/yr. Consistent with model release cascade. | 3-point estimation is fragile; acceleration ratio depends on within-2024 baseline quality. |

### Corrected

| Finding | Original claim | Corrected to | Source |
|---|---|---|---|
| **Management indicator** | +31pp at entry level (T11) | +4-10pp using validated patterns (T22) | "Leading" was 99.4% adjective; "cross-functional" was 84% collaboration; "leadership" was 77% boilerplate |
| **Soft skills expansion** | +16pp at entry level (T11) | SWE grew LESS than control (DiD = -5.1pp, T18) | Field-wide trend, not SWE-specific |

### Contradicted

| Hypothesis | Evidence against | Implication |
|---|---|---|
| **Management migrated from senior to entry** | Expanded at ALL levels (T21). Entry +31pp, mid-senior +27pp, director +7pp (narrow metric). Director density actually fell -23%. | Universal template expansion, not seniority-specific migration. |
| **Junior-senior semantic convergence** | Within-2024 calibration shift exceeds cross-period change (T15). Base-rate-adjusted NN analysis shows only +6.3pp excess. | Convergence claim not robust. Use specific requirement-level signals (T11) instead. |
| **Requirements outpace usage** | Requirements LAG usage: ~41% posting vs ~75% developer usage (T23). Gap narrowed from -45pp to -34pp. | Inverts RQ3 direction. Employers are catching up, not inflating beyond reality. |

### New findings (not in original design)

| Finding | Evidence | Significance |
|---|---|---|
| **AI-entry orthogonality** | Firm-level r = -0.07 (p=0.138, T16). Metro-level r = -0.04 (p=0.850, T17). | The paper CANNOT claim AI caused junior elimination at the firm level. These are parallel market trends. |
| **57% of aggregate change is compositional** | T16 shift-share decomposition. New entrants have 24.3% AI (vs 2.5% for exits). | More than half of what we observe reflects different companies posting, not same companies changing. |
| **Associate collapsing toward entry** | Relative position in feature space: 0.30 to 0.16 (T20). Feature profiles nearly identical. | De facto 3-tier seniority system emerging (junior, mid, senior). |
| **Director/mid-senior boundary blurred** | AUC dropped 0.75 to 0.64 (T20). Management density gap: 0.62 to 0.09. | Director increasingly indistinguishable from mid-senior on structured features. |
| **AI requirements genuine, not aspirational** | Hedge fraction reversed: AI was more aspirational in 2024 (40% vs 34%), less aspirational in 2026 (20% vs 30%). (T22) | AI listing requirements represent real hiring bars, strengthening AI surge finding. |

---

## 6. Posting Archetype Summary (T09)

14 BERTopic archetypes reduced from 92 fine-grained topics. Four macro-archetypes account for 81% of non-noise postings:

| Archetype | 2024 share | 2026 share | Change | Character |
|---|---|---|---|---|
| Frontend/Web | 41% | 24% | -17pp | JavaScript, React, web applications |
| Embedded/Systems | 26% | 19% | -7pp | Test, systems, hardware-adjacent |
| Data/Analytics | 22% | 21% | -1pp (stable) | Cloud, AWS, Azure, data pipelines |
| ML/AI Engineering | 4% | 27% | +22pp | AI, building, ML, models |

Seniority NMI = 0.018 (weakest signal). Domain NMI = 0.175 (strongest). A senior data engineer posting is more similar to a junior data engineer posting than to a senior frontend posting.

**Implication for analysis:** All seniority-stratified analyses should also be domain-stratified. Pooled junior-vs-senior comparisons mask domain-specific dynamics.

---

## 7. Technology Evolution Summary (T14)

- **61 technologies rising, 12 declining, 73 stable** (FDR-corrected).
- **Top risers:** CI/CD +16.3pp, Python +15.4pp, LLM +12.8pp, agent frameworks +10.3pp, generative AI +8.8pp, ML +8.3pp.
- **Top decliners:** Agile -4.8pp, SQL -3.8pp, HTML/CSS -3.6pp, Linux -2.1pp, .NET -1.8pp.
- **Python became majority:** 34.6% to 50.1%, the largest absolute gain among traditional technologies.
- **New AI ecosystem:** LangChain, RAG, vector DBs, agent frameworks emerged from near-zero.
- **AI is additive:** AI-mentioning postings require 11.4 techs vs 7.3 for non-AI.
- **Stack diversity:** 6.2 to 8.3 mean techs/posting (robust to all sensitivity checks).

---

## 8. Geographic Summary (T17)

- **All 26 metros show positive changes** on all five metrics (entry share, AI prevalence, org scope, description length, tech diversity). No counter-trend metro.
- **Tech hubs vs non-hubs:** No significant differences (all p > 0.08). Non-hubs change as much as hubs. The transformation is nationally uniform.
- **AI-entry correlation at metro level:** r = -0.039, p = 0.850 (null). Reinforces firm-level orthogonality finding.
- **CV of changes:** Description length most uniform (CV=0.24); entry share most variable (CV=0.40) but all positive.
- **Remote work:** 22.6% of 2026 scraped SWE postings are remote-tagged but invisible to metro analysis (no metro assignment). Cross-period remote comparison impossible (2024 = 0% by data artifact).

---

## 9. Senior Archetype Characterization (T21)

### Key finding: Management expanded everywhere; orchestration is the real senior shift

Management language did NOT migrate from senior to entry. It expanded at ALL levels:
- Entry: 9.5% to 40.8% (+31pp narrow metric; corrected to +4-10pp with validated patterns)
- Mid-senior: 23.1% to 49.9% (+27pp)
- Director: 47.8% to 54.5% (+7pp binary; but DENSITY fell -23%)

The distinctive senior change is **technical orchestration surging**: +16% mid-senior, +46% director. AI-mentioning senior postings have HIGHER orchestration and LOWER management density than non-AI senior postings. The gap widened from 2024 to 2026.

### Senior sub-archetypes (k=4 clustering)

| Cluster | Share | AI rate | Key trait |
|---|---|---|---|
| Low-touch Generalist | 58% (stable) | 12% | Standard senior SWE |
| People Manager | 22% (+2.5pp) | 11% | Team management, mentoring, hiring |
| Technical Orchestrator | 6% (+1.3pp) | 18% | Code review, system design, AI workflows (highest AI rate) |
| Strategic Leader | 14% (-3.3pp) | 13% | Stakeholder, roadmap, budget (declining) |

---

## 10. Ghost/Aspirational Assessment (T22)

### Management indicator: Measurement error corrected

The T11 headline (+31pp management at entry) was inflated by three problematic patterns:
- `\bleading\b`: 33% of 2026 entry postings matched, but 99.4% were adjective usage ("a leading company")
- `\bcross-functional\b`: 19% matched, 84% were collaboration not management
- `\bleadership\b`: 12% matched, 77% were company boilerplate

**Corrected rates (strict validated patterns):** 24.6-30.1% (2024) to 34.3% (2026) = +4-10pp.

### AI requirements: Genuine, not aspirational

| Period | AI hedge fraction | Non-AI hedge fraction |
|---|---|---|
| 2024-01 | 40.2% | 34.2% |
| 2024-04 | 30.9% | 21.9% |
| 2026-03 | **20.0%** | **30.0%** |

AI requirements REVERSED from more-aspirational-than-average (2024) to less-aspirational-than-average (2026). This is strong evidence that AI listing requirements are real hiring bars.

### Text quality asymmetry warning

LLM-cleaned text reduces 2024 management indicator rates by ~15pp compared to rule-based text. 2026 scraped data has no LLM-cleaned text. This asymmetry systematically inflates the apparent 2024-to-2026 change for ANY boilerplate-sensitive indicator. When LLM-cleaned 2026 text becomes available, the 2026 management rate will likely drop further.

---

## 11. New Hypotheses from T24

Eight new hypotheses generated, of which the most testable with existing data:

1. **H1: Domain recomposition drives junior decline.** ML/AI has lower entry share AND grew from 4% to 27%. The compositional shift alone could account for much of the aggregate decline. Test: within-domain entry share decomposition.

2. **H4: Two-speed market.** AI-native firms (new entrants) vs legacy transformation firms. Different companies drive the compositional (57%) and behavioral (43%) components.

3. **H7: GenAI as domain-boundary dissolving force.** AI-mentioning postings may show higher cross-domain similarity, blurring traditional archetype boundaries.

The orthogonality puzzle (AI and entry changes parallel but uncorrelated within firms) is the most important open question for both analysis and interviews.

---

## 12. Method Recommendations for Analysis Phase

### Recommended statistical methods by RQ

| RQ | Primary method | Secondary / robustness |
|---|---|---|
| RQ1 (restructuring) | Cross-occupation DiD (SWE vs control, by metric) | Within-firm panel FE; shift-share decomposition; domain-stratified entry share |
| RQ1b (within vs compositional) | Shift-share decomposition (T16 method) | Company-level regression with FE |
| RQ2 (technology) | Technology co-occurrence network analysis; stack diversity regression | NMF topic modeling; within-2024 calibration for all tech trends |
| RQ3 (divergence) | Benchmark comparison with confidence intervals | Aspiration-ratio within postings; specificity gradient (generic vs tool-specific) |
| RQ4 (mechanisms) | Reflexive thematic analysis on semi-structured interviews | JD-stimulated recall using T25 artifacts |

### Recommended sensitivity framework

Every finding should be reported with the following checks:
1. **(a) Aggregator exclusion** -- removes 12-27% of sample by source
2. **(b) Company capping** -- max 10-20 per company to prevent template dominance
3. **(c) Seniority operationalization** -- seniority_native vs seniority_final vs seniority_3level
4. **(f) Within-2024 calibration** -- arshkon vs asaniczka as noise floor
5. **(g) SWE classification tier** -- regex-only vs full sample

### Analysis sequencing

1. **First:** Run the domain-stratified entry share decomposition (H1). This may reframe the entire junior decline story.
2. **Second:** Reproduce T18 DiD with corrected management indicators. Confirm SWE-specificity of AI and entry findings with the strict pattern set.
3. **Third:** Formal robustness tables for each core finding (AI surge, junior decline, domain recomposition, AI additive, slot purification).
4. **Fourth:** Within-firm panel analysis with company FE for the 451 overlap companies.
5. **Fifth:** Geographic heterogeneity analysis with metro FE.

---

## 13. Sensitivity Requirements

### Findings requiring formal robustness checks

| Finding | Required sensitivities | Notes |
|---|---|---|
| Junior share decline | (a)(b)(c)(f)(g) + domain stratification | The most specification-sensitive finding. Must show robustness across seniority operationalizations. |
| AI requirements surge | (a)(b)(f)(g) | Already well-validated; formal tables needed for paper. |
| Domain recomposition | Text source (LLM vs rule-based), method (BERTopic vs NMF), sample (full vs random) | T09 showed 0.88 cosine similarity between LLM and rule-based cluster distributions. |
| Management indicator (corrected) | Pattern set specification (strict vs moderate), text source (description vs description_core) | Must use T22 validated patterns ONLY. |
| AI-entry orthogonality | Panel size threshold (min 3 vs 5 vs 10), seniority operationalization, with/without aggregators | A negative result -- must show it doesn't appear under any specification. |
| 57% compositional | Panel definition (overlap threshold), era pooling (separate 2024-01/2024-04 vs pooled) | Critical for framing; should be robust to reasonable alternatives. |

---

## 14. Pending Data Improvements

### seniority_llm (Stage 10 budget allocation)

**What it provides:** Explicit-signal-only seniority classification for all SWE rows. Will NOT infer seniority from responsibilities (which would circularly contaminate the analysis of how responsibilities differ by seniority).

**What changes when available:**
- Resolves the seniority operationalization discrepancy (Section 3)
- Provides a consistent classification across all three sources (including asaniczka)
- Enables asaniczka to participate in seniority-stratified analyses for the first time
- Expected to settle the entry-level trend direction definitively

**Priority:** HIGHEST. This is the binding constraint on RQ1.

### description_core_llm for scraped data (Stage 9 budget allocation)

**What it provides:** LLM-quality boilerplate removal for 2026 scraped data (currently 0% coverage).

**What changes when available:**
- Eliminates the text quality asymmetry between 2024 and 2026
- The 2026 management indicator rate will likely DROP further (T22 showed 15pp difference between LLM and rule-based text for 2024)
- All text density metrics become comparable across periods
- Enables reliable description_core_llm-based analyses on scraped data

**Priority:** HIGH. Second after seniority_llm.

---

## 15. Interview Priorities (RQ4)

### Priority 1: The orthogonality puzzle

**Prompt:** "AI adoption surged and junior hiring declined simultaneously at the market level, but these trends are completely uncorrelated within individual companies. Companies that adopted AI the most did NOT cut junior roles. What explains this?"

**Why this matters:** This finding challenges the dominant "AI replaces junior developers" narrative. Practitioner explanations will help distinguish between: (a) separate mechanisms operating in parallel, (b) market-level equilibrium effects, (c) time-lag causation our snapshots cannot capture, or (d) the trends being genuinely independent.

### Priority 2: Are AI requirements real hiring bars or signaling?

**Prompt:** "41% of SWE postings now mention AI requirements, and they're phrased more firmly than traditional requirements. But 75% of developers already use AI. Is listing 'LLM experience' on a posting a real filter, or is it signaling organizational modernity?"

**Why this matters:** T22 showed AI requirements are less hedged than traditional ones. But whether they function as actual hiring screens vs recruitment signaling is a behavioral question the data cannot answer.

### Priority 3: The domain recomposition experience

**Prompt:** "ML/AI engineering grew from 4% to 27% of SWE postings in two years. Frontend/Web contracted from 41% to 24%. How are you experiencing this shift? Are existing teams retraining? Are new hires arriving with different skills? Is the shift in posting labels matching a shift in actual work?"

**Why this matters:** The domain recomposition is the largest structural change in the data. Whether it represents genuine new work or cosmetic relabeling is a critical interpretation question.

---

## 16. The Paper's Strongest Story

Based on 23 exploration tasks and 4 gate reviews, the evidence supports this narrative:

**The SWE labor market restructured between 2024 and 2026 through three SWE-specific mechanisms operating in parallel:**

1. **AI competency requirements surged** (+24pp DiD vs control), validated as genuine hiring requirements (not aspirational). Requirements still lag developer usage but are converging rapidly.

2. **Entry-level SWE posting share declined** while non-SWE junior share increased (DiD = -25pp). But AI adoption and entry-level changes are orthogonal at the firm level (r ~ 0) -- these are parallel market trends, not causally linked within organizations. Over half (57%) of aggregate change is compositional.

3. **The SWE domain landscape recomposed**: ML/AI engineering grew from 4% to 27% of postings while frontend/web contracted. Domain is 10x more structurally important than seniority in determining posting content.

**What the paper should NOT claim:**
- That AI caused junior elimination within firms
- That management scope inflation is dramatic (+4-10pp, not +31pp) or SWE-specific (field-wide)
- That junior and senior postings are semantically converging (fails calibration)
- That posting requirements outpace developer usage (they lag)
- That soft skills expansion is SWE-specific (SWE grew LESS than control)

**What requires caveats:**
- Entry-level trend direction depends on seniority operationalization (planned LLM labels will resolve)
- Text quality asymmetry between 2024 (LLM-cleaned) and 2026 (rule-based) inflates text-based indicators
- 57% of aggregate change is compositional -- the "restructuring" is partly about which companies post, not just how they post
- The corrected management indicator (+4-10pp) still uses rule-based text for 2026; will likely drop further with LLM cleaning

---

## Appendix: Task Completion Summary

| Task | Wave | Key finding |
|---|---|---|
| T01 | 1 | 28 cols >50% null; LLM cols null; core analysis columns viable |
| T02 | 1 | Asaniczka associate NOT junior proxy; 830 arshkon entry-level baseline |
| T03 | 1 | Junior decline 5.9-8.3pp robust across 4 operationalizations |
| T04 | 1 | SWE classification 4-6% FP, <0.5% FN |
| T05 | 1 | Most cross-dataset diffs are artifacts; seniority is the exception |
| T06 | 1 | Within-company entry decline -11.8pp (stronger than aggregate) |
| T07 | 1 | Well-powered; r>0.97 geographic representativeness |
| T08 | 2 | YOE slot purification (5+ YOE entry: 22.8% to 2.4%); AI tech 5-17x noise |
| T09 | 2 | Domain > seniority (10x); ML/AI 4% to 27% |
| T10 | 2 | AI titles 4x; Staff tripled; titles stable in meaning |
| T11 | 2 | Scope inflation real but management CORRECTED by T22 to +4-10pp |
| T12 | 2 | AI dominant text signal; most 2026-distinctive terms are boilerplate |
| T13 | 2 | 93.9% of length growth is core content, not boilerplate |
| T14 | 2 | 61 rising techs; AI additive; new AI community; stack diversity +2.1 |
| T15 | 2 | Convergence fails calibration; 7/9 findings representation-robust |
| T16 | 3 | 57% compositional; AI-entry null (r=-0.07); 4 company clusters |
| T17 | 3 | All metros same direction; hub/non-hub null; AI-entry null at metro |
| T18 | 3 | SWE-SPECIFIC CONFIRMED: AI DiD +24pp, junior DiD -25pp; mgmt/soft field-wide |
| T19 | 3 | GenAI accelerated 8.3x; within-March stable (CV<10%) |
| T20 | 3 | Boundaries asymmetric: entry/assoc sharp, dir/midsen blurred; associate collapsing toward entry |
| T21 | 3 | Management migration REJECTED; expanded everywhere; orchestration +46% at director |
| T22 | 3 | Management indicator CORRECTED +4-10pp; AI NOT aspirational (20% vs 30% hedge) |
| T23 | 3 | Requirements LAG usage (~41% vs ~75%); gap narrowing; AI-as-domain may overshoot |
| T24 | 4 | 8 new hypotheses; H1 (domain drives junior decline) most testable |
| T25 | 4 | 6 interview artifacts; 3 priority themes |
| T26 | 4 | This synthesis |
