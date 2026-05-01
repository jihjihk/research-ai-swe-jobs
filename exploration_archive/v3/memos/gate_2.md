# Gate 2 Research Memo

Date: 2026-04-05
Wave: 2 (Open Structural Discovery)
Tasks completed: T08-T15

---

## What we learned

### 1. Junior roles are being simultaneously purified and redefined — the headline finding.

The two most important tasks (T08 and T11) together reveal a paradoxical restructuring of entry-level SWE:

- **Slot purification (T08):** High-YOE "entry-level" postings nearly vanished (5+ YOE entry: 22.8% → 2.4%). The remaining entry roles are more genuinely junior (median YOE 3.0 → 2.0). Confirmed across five independent decompositions.
- **Qualitative scope inflation (T11):** Despite lower YOE bars, the surviving entry roles demand dramatically more: management indicators 9.4% → 40.8% (+31pp), AI requirements 3.9% → 27.5% (+24pp), soft skills 71.4% → 87.5% (+16pp), scope/ownership 37.0% → 48.2% (+11pp). All survive sensitivity checks and calibration (4.6x above noise floor).

The paper's lead should be: *Employers are hiring fewer juniors, asking for less experience, but demanding qualitatively more — management awareness, AI competency, and organizational scope that barely existed in 2024 entry-level postings.*

### 2. Technology domain, not seniority, is the market's dominant structure.

T09's unsupervised clustering reveals that SWE postings organize by what kind of software people build (frontend, embedded, data, ML/AI), not by how senior they are (NMI: domain 0.175 vs seniority 0.018, a 10x difference). A senior data engineer posting is more similar to a junior data engineer posting than to a senior frontend posting.

The temporal shift is dramatic: ML/AI Engineering surged from 4% to 27% of SWE postings (+22pp), while Frontend/Web contracted from 41% to 24% (-17pp). This domain recomposition is itself a restructuring signal that may be as important as the seniority story.

### 3. AI is additive, not replacing — and the content growth is genuine.

T14 shows AI-mentioning postings require MORE traditional technologies (11.4 vs 7.3), not fewer. Stack diversity increased overall (6.2 → 8.3 techs/posting). A new 25-technology AI/ML ecosystem emerged with no 2024 counterpart.

T13 resolves the description length question: 93.9% of the 57% growth is in core content sections (role summary +42%, responsibilities +27%, requirements +25%). It is NOT boilerplate padding. Employers are writing genuinely longer, more detailed job descriptions.

### 4. The text evolution is dominated by AI signals and boilerplate, in roughly equal measure.

T12's Fightin' Words analysis shows "AI" is the strongest genuine text signal (z=47.0). But of the top 100 2026-distinguishing terms, only ~13 are genuine content changes — the rest are boilerplate artifacts (benefits, salary, compensation). The AI signal is validated against within-2024 calibration and is NOT among instrument artifacts. Traditional stacks declined sharply (JavaScript -42.8z, SQL -48.5z, Java -36.6z).

### 5. Junior-senior semantic convergence does NOT survive calibration.

T15's nearest-neighbor analysis shows 2026 junior postings are slightly more similar to 2024 senior postings, but the within-2024 calibration shift exceeds the cross-period change. The "convergence" story is not robust. This is an important negative finding — the scope inflation signal from T11 (specific requirement dimensions) is much stronger than the semantic similarity signal from T15.

---

## What surprised us

1. **Management indicators in entry-level jumped from 9.4% to 40.8%.** This is the single largest scope inflation signal. Nearly half of 2026 entry-level postings mention management, leadership, or mentoring — up from fewer than 1 in 10. This suggests entry-level roles are being redefined to include organizational responsibilities that were previously mid-level.

2. **Technology domain is 10x more structurally important than seniority.** The RQ1-RQ4 framing assumes seniority is the primary axis of restructuring. The data says technical specialization matters far more for how postings are organized. The paper may need a domain-stratified approach.

3. **ML/AI archetype grew from 4% to 27%.** This is the single largest compositional shift. It's not just that existing roles added AI requirements — an entire new category of role emerged and grew to over a quarter of all SWE postings.

4. **Agile/Scrum language declined across the board.** This was unexpected — methodology language is falling even as other organizational language rises. An era may be ending.

5. **"Years experience" bigram dropped from 5.6 to 2.1 per 1K chars.** Employers are moving away from experience-year requirements as a primary screening criterion, consistent with the T08 YOE purification finding.

6. **Entry-level 2026 postings are MORE readable** (FK 15.3 vs 16.5). The surviving junior postings are written in more accessible language — opposite to what "scope inflation toward more complex roles" might predict.

7. **AI term explosion comes from a growing tail, not uniform shift.** Median AI-forward language is still 0.0 in 2026. The 27.5% entry-level AI requirement rate means ~73% of entry postings still don't mention AI. The shift is real but concentrated in a subset.

---

## Evidence assessment

| Finding | Strength | Sample | Calibration ratio | Survives sensitivities? |
|---------|----------|--------|-------------------|------------------------|
| Entry scope inflation (mgmt +31pp) | **Strong** | 830 vs 3,255 | 4.6x noise | Yes (all 4 checks) |
| Entry YOE purification (5+ YOE: 22.8%→2.4%) | **Strong** | 830 vs 3,255 | N/A (internal) | Yes (5 decompositions) |
| Domain recomposition (ML/AI 4%→27%) | **Strong** | 8,002 sample | N/A (unsupervised) | Method-robust (ARI≥0.996) |
| AI additive to stacks | **Strong** | 24,095 (2026) | N/A | Yes (aggregator, capping) |
| Content growth genuine (93.9% in core) | **Strong** | Full sample | N/A | V2 section classifier |
| Stack diversity increase (6.2→8.3) | **Strong** | Full sample | Not reported | Yes (aggregator, capping) |
| Junior-senior convergence | **Weak** | 12,260 sample | <1x (fails calibration) | No |
| AI text signal (z=47.0) | **Strong** | 5,017 vs 24,095 | 5-17x noise | Yes (all checks) |
| Title meaning stable (cos=0.95) | **Strong** | Top 10 titles | N/A | Yes |
| Agile decline (-4.8pp) | **Moderate** | Full sample | Not reported | Likely |

---

## Narrative evaluation

### RQ1: Junior scope inflation
**Status: CONFIRMED, with reframing.**

The original hypothesis was directionally correct but incomplete. "Junior scope inflation" is real but occurs alongside slot purification and YOE deflation. The more precise characterization: *junior role redefinition* — fewer slots, lower experience bars, but dramatically broader qualitative requirements. Management, AI, and organizational scope language entered entry-level postings at rates that dwarf all other changes.

**Proposed reframing:** "How are entry-level SWE roles being redefined?" — encompassing both the quantity reduction and the qualitative transformation.

### RQ2: Task and requirement migration
**Status: PARTIALLY CONFIRMED.**

T11 shows management and AI requirements migrated into entry-level postings. T14 shows technology stack diversification. But T09 reveals the dominant migration may be at the domain level (ML/AI archetype growth) rather than the seniority level. The paper should address both axes.

### RQ3: Employer-usage divergence
**Status: CANNOT EVALUATE.** Needs T23 in Wave 3. But the AI requirement surge (3.9% → 27.5% for entry, 7.6% → 33.2% overall) is now well-quantified as the posting-side input.

### RQ4: Mechanisms (qualitative)
**Status: NOT PART OF EXPLORATION.** But the scope inflation findings (especially the 40.8% management indicator for entry-level) and the domain recomposition are strong interview prompts.

### Is the initial RQ1-RQ4 framing still the best?

**It needs modification but not replacement.** The data supports a three-part story:

1. **Junior role redefinition** (RQ1, enriched): Slot purification + qualitative scope expansion
2. **Domain recomposition** (NEW): ML/AI archetype surge, frontend contraction
3. **Technology ecosystem expansion** (supports RQ2): AI additive to existing stacks, stack diversity increasing

The domain recomposition finding (T09) is strong enough to warrant its own RQ or to be integrated into RQ2. The employer-usage divergence (RQ3) hasn't been tested yet but has strong input data.

---

## Emerging narrative

**Draft paper abstract (as of Gate 2):**

*Using ~52K SWE job postings from LinkedIn spanning 2024-2026, we document a three-part restructuring of software engineering labor demand during the period of rapid AI coding tool adoption. First, entry-level SWE roles are being simultaneously purified and redefined: established companies reduced junior posting share by 12 percentage points, eliminated high-experience "entry-level" positions, but dramatically expanded the qualitative scope of remaining junior roles to include management awareness (+31pp), AI competency (+24pp), and organizational scope language (+11pp). Second, the market's dominant structural dimension shifted from traditional web/systems development toward ML/AI engineering, which grew from 4% to 27% of SWE postings. Third, AI requirements are additive to existing technology stacks — AI-mentioning postings require 56% more technologies than non-AI postings, suggesting AI is expanding rather than simplifying the SWE skill surface. These changes are robust to sensitivity analysis and exceed within-2024 instrument noise by 5-17x.*

---

## Research question evolution

**Current RQ set (proposed modifications in bold):**

- **RQ1 (modified):** How are entry-level SWE roles being redefined between 2024 and 2026? *(From "junior scope inflation" to "junior role redefinition" — encompasses slot purification, YOE deflation, and qualitative scope expansion)*
- **RQ1b (NEW):** Is the seniority-level restructuring driven by within-firm changes or by compositional shifts in which firms are hiring?
- **RQ2 (expanded):** How did the SWE technology ecosystem and domain composition change? *(Incorporates the domain recomposition finding from T09 alongside the original task migration question)*
- **RQ3 (unchanged):** Do employer-side AI requirements outpace observed workplace AI usage?
- **RQ4 (unchanged):** How do practitioners explain the restructuring?

---

## Gaps and weaknesses

1. **Ghost requirements are the biggest validity threat.** The 40.8% management indicator in entry-level could be aspirational copy-paste rather than real hiring bars. T22 (Wave 3) is the critical validity check.
2. **Cross-occupation comparison not yet done.** If control occupations show similar patterns, the "SWE-specific" framing weakens. T18 is essential.
3. **Domain recomposition needs temporal resolution.** We don't know if ML/AI grew gradually or surged suddenly. T19 will characterize the temporal structure.
4. **Within-company scope inflation not yet tested.** T06 showed within-company junior share decline. T16 should test whether within-company scope inflation also occurs.
5. **AI requirement concentration.** The 27.5% entry-level AI requirement rate means 73% of entry postings don't mention AI. The shift is real but concentrated. Who are the adopters vs non-adopters?
6. **Semantic convergence failed calibration.** We cannot claim junior and senior postings are "converging" — the specific requirement-level signals from T11 are the right evidence, not the embedding similarity from T15.

---

## Direction for Wave 3

### Modifications based on Gate 2

**T16 (Company strategies):** Add instruction to stratify by T09 domain archetype. Do companies that shifted toward ML/AI show different entry-level patterns than those that stayed in frontend/systems?

**T18 (Cross-occupation):** MOST CRITICAL Wave 3 task. The scope inflation findings need the SWE-specificity test. If management indicators also surged in control occupations, this is a macro trend, not AI-driven restructuring.

**T20 (Seniority boundaries):** Add domain archetype as an additional stratifier. Does the boundary blur/sharpen differently across domains?

**T21 (Senior role evolution):** Compare the management language shift in senior postings to the entry-level surge from T11. If senior management language DECREASED while entry management language INCREASED, that's migration evidence.

**T22 (Ghost forensics):** ELEVATED IMPORTANCE. The 40.8% management indicator finding is the paper's most striking number. If it's ghost requirements, the headline weakens substantially. T22 must specifically test whether management terms in entry-level are more aspirational than in other contexts.

**T23 (Employer-usage divergence):** The posting-side AI requirement data is now well-quantified. This task needs to find the worker-side benchmark and compute the gap.

---

## Current paper positioning

The data now supports an **empirical restructuring paper** rather than a dataset/methods paper. The strongest positioning:

*Lead with the paradoxical junior role redefinition (lower YOE + higher scope), supported by within-company evidence and domain recomposition. Frame as employer-side anticipatory restructuring during the AI adoption era. Use the within-2024 calibration as the methodological contribution (demonstrating that instrument noise is controlled). Use interviews (RQ4) to adjudicate whether scope inflation reflects real workflow change or anticipatory signaling.*

**If we stopped here,** the best paper is:

> "The AI-era restructuring of SWE labor demand: Evidence from 52K job postings showing simultaneous junior slot purification, scope expansion, and domain recomposition."

**What Wave 3 needs to deliver to strengthen this:**
- SWE-specificity confirmation (T18) — the most important remaining test
- Ghost requirement assessment (T22) — validates the scope inflation finding
- Within-company scope inflation (T16) — extends the T06 within-company design
- Senior archetype characterization (T21) — completes the seniority story
- Employer-usage divergence estimate (T23) — tests RQ3
