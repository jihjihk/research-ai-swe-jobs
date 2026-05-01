# Gate 1 Research Memo

Date: 2026-04-15
Wave: 1 (Data Foundation, T01-T07)
Reports read: T01, T02, T03, T04, T05, T06, T07

---

## What we learned

1. **The data is large enough in aggregate but thin at the seams that matter for RQ1.** SWE LinkedIn: arshkon 4,691 / asaniczka 18,129 / scraped 40,881. Power is comfortable for all-SWE and senior cross-period comparisons (MDE_binary 1.2-3.3 pp). It is **underpowered for arshkon-only entry** (MDE 11.3 pp on n=175 vs 1,275) and only **marginal for pooled-2024 entry** (MDE 8.2 pp on n=379 vs 1,275). The observed entry-share changes are comparable in size to the MDE, which means any RQ1 entry-specific claim will be on shaky statistical footing from the start.

2. **Every junior-share metric rises aggregate 2024 → 2026 — but the metrics do not agree with each other, and the overlap panel flips direction.** This is the single most important Wave 1 finding.

   | Operationalization | 2024 | 2026 | Direction (aggregate) | Overlap-panel within-company Δ |
   |---|---|---|---|---|
   | `seniority_final = 'entry'` (of known) | 2.70% | 6.70% | ↑ ~2.5× | **−3.2 pp** |
   | YOE ≤ 2 (of YOE-known) | 11.22% | 17.19% | ↑ ~1.5× | **+1.1 pp** |
   | `seniority_native = 'entry'` arshkon | 15.73% | — | — (broken baseline) | — |

   - In the **aggregate**, both metrics rise, but at different magnitudes.
   - On the **arshkon↔scraped overlap panel (115 companies with ≥5 SWE in both)**, the within-company component flips direction between the two operationalizations: `seniority_final` says same companies post *fewer* entry, YOE ≤ 2 says same companies post *slightly more*. The aggregate rise is driven primarily by **between-company composition** (new entrants, non-overlap companies) under both metrics.
   - The `seniority_native` arshkon-only baseline is unusable: arshkon native='entry' rows have **mean YOE 4.18, 26% have YOE ≥ 5**, vs scraped native='entry' mean YOE 2.34. LinkedIn's own native entry label drifted between 2024 and 2026, and pooling it across periods creates a spurious signal. The cleanest diagnostic here is that 84% of arshkon native='entry' rows become `seniority_final = 'unknown'` — the Stage-10 LLM is refusing to confirm them as entry, correctly.

3. **Entry-level posting is a specialist activity in every source, and a small number of companies drive the entry pool in the scraped window.** Among companies with ≥5 SWE, the share with **zero** entry-labeled postings is 79% (arshkon), 89% (asaniczka), 79% (scraped) under `seniority_final`; 53-58% under the YOE proxy. In 2026, **Walmart drives 73.8% of its own YOE ≤ 2 rate** and ~3.6% of the entire scraped YOE ≤ 2 pool; Google drives 52.4% of its own rate and ~8.2% of the pool. **TikTok + ByteDance alone = 10.4% of the scraped entry-final pool.** Excluding Walmart and Google single-handedly moves the direction of the "juniors rising" YOE story. Stage 4's multi-location collapse is working — the prior run's "6 companies × identical descriptions" artifact is gone — but 24 cross-company hash collisions remain (top collision: 6 companies sharing one description), likely aggregator relabeling.

4. **AI mention rate is the only aggregate metric that clearly survives within-2024 calibration.** T05 SNR table:
   - AI mention rate: SNR **925** → robust.
   - Description length: SNR 2.29 → marginal (confounded by Kaggle HTML strip vs scraped markdown).
   - YOE ≤ 2 share: SNR 0.98 → below threshold.
   - Mean YOE: SNR 0.80 → below threshold.
   - `seniority_final` entry share: SNR **0.33** → **below threshold — pooling 2024 sources is not safe for the entry-share comparison.**

   Indeed cross-validation (separate platform) agrees with LinkedIn on entry share to within 0.1 pp. This **rules out LinkedIn platform relabeling** as the source of the scraped-2026 entry signal, but leaves the 2024 asaniczka label-coverage gap as the dominant confound.

5. **The macro backdrop is cooling, not booming.** JOLTS information-sector openings averaged 136k in our 2024 window vs 97k in our 2026 window — a **29% decline**. BLS state-level SWE employment correlates with our state-level SWE posting counts at **r = 0.97** (far above the 0.80 target), so our geographic frame is validated. Any cross-period story must land against a contracting hiring market — an "entry share rising" finding could be consistent with employers shifting posting mix toward cheaper roles while total volume falls, not with entry-level expansion.

6. **`seniority_final` is defensible as the primary seniority column for the rest of the exploration**, with hard caveats. LLM routing-error rate on weak-marker titles is <5%; kappa vs `seniority_native` is 0.45 arshkon / 0.65 scraped (the arshkon gap is a feature, not a bug — it reflects real native-label noise on arshkon that the LLM correctly rejects). The downstream rule: always report the YOE proxy alongside, always show both "of all" and "of known-seniority" denominators, never use `seniority_native` for 2024 entry baselines. **This recommendation frees downstream agents from re-litigating seniority quality — T03 is the canonical reference.**

---

## What surprised us

- **Arshkon — not asaniczka — has the noisy native entry label.** I expected asaniczka (which has zero native entry labels) to be the problematic 2024 source. It turns out arshkon's native `entry` label over-captures mid-level: mean YOE 4.18, 26% at YOE ≥ 5. This inverts the implied hierarchy of the two 2024 sources for seniority diagnostics and is a real instrument finding, not a pipeline artifact. The Stage-10 LLM's "unknown" landing for 84% of arshkon native='entry' is correct, not a failure.

- **Asaniczka `associate` is not a usable junior proxy.** 88% of `seniority_final`-labeled asaniczka `associate` rows land in mid-senior, not entry. Top-title Jaccard with asaniczka's own mid-senior (0.23) is higher than with arshkon entry (0.14). Asaniczka's level mapping appears to skew more senior than arshkon's — T02 suggests this may be LinkedIn level-mapping drift between asaniczka's 2024-01 capture and arshkon's 2024-04 capture, which is a micro-signal worth raising to the preprocessing owner.

- **Scraped has WORSE LLM text coverage than 2024 historical**: 30.7% `llm_extraction_coverage = 'labeled'` vs 94-100% for arshkon/asaniczka. The Stage-9 balanced core frame was sized for a smaller scraped window and is being outgrown as the scraper runs. **Text-based analyses are 2026-side capped at ~12.5k SWE rows, not 40.9k.** This is counterintuitive and is the binding constraint for T08, T09, T12, T13, T14, T15.

- **53% of scraped `seniority_final = 'unknown'`**, so the scraped known-seniority SWE frame is ~19k, not 41k. Stage 10 budget-capped routing is the proximate cause.

- **Entry-share direction flip on the same 115-company overlap panel** between `seniority_final` and the YOE proxy is not sampling noise — it's the two labels measuring genuinely different things on the same rows. This single finding reframes what kind of paper this can be (see Narrative Evaluation).

- **QA Engineer's SWE share jumped from 9% (asaniczka) to 37% (scraped)** in the classifier. Two possible explanations: (1) real migration of QA roles toward automation/test-engineering overlap with SWE, or (2) classifier drift. Either is a finding worth keeping — a 28-point jump should not be invisible.

- **Stage 4's multi-location collapse cleaned up the prior run's duplicate-template artifact.** The "23% of 2026 entry pool from 6 companies posting identical descriptions 4-25 times each" is gone. This is a real pipeline improvement and means Wave 2 can trust the per-row count as a per-posting count, not a per-multi-location-fanout count.

- **BLS correlation r = 0.97**, not the ~0.85 I expected. The sample frame is better than I thought.

---

## Evidence assessment

Per the framework in the orchestrator brief:

| Finding | Sample | Evidence strength | Calibration survives? | Sensitivity survives? |
|---|---|---|---|---|
| AI mention rate rose 2024 → 2026 | Full SWE | **Strong** | Yes (SNR 925) | Not yet tested (Wave 2) |
| Entry share rose in aggregate (`seniority_final`) | Pooled 2024 n=379, 2026 n=1,275 | **Weak → moderate** | **No** (SNR 0.33) | Not yet tested |
| Entry share rose in aggregate (YOE ≤ 2) | Full SWE with YOE | **Weak** | **No** (SNR 0.98) | Not yet tested |
| Same-company within-panel entry (seniority_final) fell | 115 companies | **Moderate** | N/A (within-company) | Flips under YOE proxy |
| Same-company within-panel YOE ≤ 2 roughly flat | 115 companies | **Moderate** | N/A | Flips under seniority_final |
| Between-company composition drives most aggregate movement | 115-company overlap panel + non-overlap | **Moderate** | — | Stable across both operationalizations |
| Entry-posting is a specialist activity (~80% of companies post zero entry) | All SWE sources | **Strong** | Applies cross-source → yes | Stable |
| Top-company concentration drives entry pool (Walmart + TikTok/ByteDance ~14%) | Scraped SWE | **Strong** | N/A | Drives direction flips when excluded |
| Arshkon `seniority_native = 'entry'` is noisy (mean YOE 4.18) | arshkon SWE | **Strong** | — | — |
| JOLTS info sector −29% (macro cooling) | External | **Strong** | — | External source |
| QA Engineer SWE-share drift 9% → 37% | Classifier | **Moderate** | — | Could be real or drift |

**The only Wave 1 finding with strong-and-surviving status on cross-period interpretation is the AI mention rate rise.** Every RQ1-adjacent junior-share finding is either below the SNR threshold, direction-unstable across operationalizations, or concentration-driven.

---

## Narrative evaluation

**The initial RQ1 narrative is contradicted at the aggregate level and specification-dependent at the within-company level.**

The research design explicitly hypothesizes that "the junior rung" is narrowing — that junior share *falls* and junior scope *inflates*. Wave 1 evidence:

- **Junior share direction:** The aggregate rose, not fell. In the overlap panel under `seniority_final` it fell; under YOE ≤ 2 it stayed flat.
- **Statistical support:** Below MDE for the entry-specific comparison and below the SNR threshold for distinguishing from within-2024 noise.
- **Scope inflation (junior-specific):** Not yet measured — that's Wave 2 (T11, T13). But the concentration prediction table warns that the scope indicator is **HIGH risk / direction-unstable** without company capping and keyword validation.

**Status of each original RQ:**

- **RQ1 — Junior share decline:** **Contradicted (weakly) at the aggregate, unstable at the within-company level.** The paper cannot lead with "junior share fell" as a headline. It may still be able to say "entry-level posting concentration rose and the entry pool became more employer-concentrated," which is a different story.
- **RQ1 — Junior scope inflation:** **Unmeasured — deferred to Wave 2.** Must be tested under the concentration prediction table's recommended handling (cap 50/company, validate keywords, YOE-proxy stratification, exclude Walmart/Google as sensitivity).
- **RQ1 — Senior redefinition:** **Unmeasured — deferred to Wave 2/3 (T21).** Well-powered (senior n is large).
- **RQ2 — Task/requirement migration:** **Unmeasured — deferred.** Feasibility is good at the all-SWE level but shaky at the entry level.
- **RQ3 — Employer/worker AI divergence:** **Unmeasured — deferred.** The AI mention rate SNR of 925 is the single strongest Wave 1 signal, so RQ3 is the best-positioned original RQ going into Wave 2.
- **RQ4 — Interview mechanisms:** Out of scope for exploration.

### Alternative narratives to evaluate at Gate 2

Two alternatives are now as plausible as the original RQ1 framing:

**Alternative A — "Instrument-dominated apparent change" (null-leaning):**
The apparent cross-period rise in entry share is driven by (a) LinkedIn's native entry label drifting between arshkon's 2024-04 capture and the scraped 2026 capture, (b) the Stage-10 LLM cleaning up the 2024 baseline more aggressively than the 2026 window because arshkon native labels were noisier, (c) between-company composition as new high-entry employers (Walmart, TikTok/ByteDance, aggregator relabeling) entered the scraped pool, and (d) a cooling labor market with fewer senior postings raising the denominator-driven entry share without an absolute entry increase. Under this reading, the RQ1 junior story is largely instrument and composition, and the paper's headline must move elsewhere.

**Alternative B — "Restructured market composition, not restructured roles":**
The entry pool in 2026 is more concentrated (driven by a handful of new-entrant mega-employers) and AI mentions exploded (SNR 925), but within-company behavior on the overlap panel is roughly flat. The paper's story becomes "who is hiring SWE entry-level changed, not what they're asking of them." This is compatible with most of the Wave 1 evidence and survives the specification-dependence of the `seniority_final` ↔ YOE proxy gap (the gap itself becomes a methodological finding).

**Original RQ1-RQ4 framing — "Junior rung narrowing, senior redefining, requirements outpacing usage":**
Only RQ3 (AI requirement-usage divergence) has strong Wave 1 signal. The junior narrowing has the **opposite** aggregate direction from the hypothesis. The senior redefinition story is untested but well-powered.

**My current preference:** Alternative B is the strongest candidate if the scope/management/AI-orchestration analyses in Wave 2/3 produce meaningful within-domain findings. If Wave 2 finds that within-company scope language is flat or falls (consistent with the `seniority_final` overlap-panel entry decline), we are looking at a market-composition paper, not a role-redefinition paper. If Wave 2 finds within-company scope rising concurrent with AI explosion, the story becomes tighter and closer to a modified RQ1. I want both alternatives to be possible until Wave 2 reports.

---

## Emerging narrative

**Right now, the most defensible one-sentence version of this paper is:**

*"Between 2024 and 2026, SWE job postings show a robust AI requirement explosion, an unstable aggregate entry-share signal that depends on the seniority operationalization used, and — on the same companies appearing in both periods — a flat-to-slightly-declining within-company entry share with most aggregate movement coming from between-company composition shifts, all against a cooling (−29% JOLTS info openings) hiring backdrop."*

That sentence is not inspiring, but it is what the data currently supports. If Wave 2 surfaces a clean within-company domain recomposition result (archetype clusters show growth in entry-poor ML/AI and decline in entry-rich Frontend/Embedded), or a clean senior-archetype shift inside the overlap panel, the story tightens into one of those framings. If Wave 2 shows no additional structure, this may turn into a **methods/dataset paper** whose contribution is "how instrument noise and specification-dependence threaten naive cross-period posting analyses."

---

## Research question evolution

Changes proposed to the RQ set going into Wave 2:

**RQ1 — split into two sub-questions.**
- RQ1a (revised): "How does junior SWE posting share and concentration change between 2024 and 2026, and is the change dominated by within-company hiring behavior or by between-company composition?" — this moves the decomposition from a sensitivity check into the primary question.
- RQ1b (revised): "Does within-company junior scope language (requirements breadth, management/orchestration terms, tech/org scope density) change between periods on the overlap panel?" — this restricts the "scope inflation" question to the cleanest identification available.

**RQ2 — keep, but flag feasibility risk.** Task and requirement migration between seniority levels is feasible at the mid-senior↔entry boundary under the pooled-2024 sample, but the entry n is thin. Wave 2 should answer it at the all-SWE level first.

**RQ3 — elevate.** The AI requirement explosion (SNR 925) is the only robust cross-period signal in Wave 1. RQ3 should be treated as potentially the **lead RQ** of the paper until Wave 2 tests whether RQ1's revised version produces a stronger finding.

**New RQ5 (proposed, provisional) — "Specification dependence of cross-period junior metrics."** The `seniority_final` ↔ YOE-proxy direction flip on the same companies is itself a finding — potentially a publishable methodological contribution for labor-market posting research. I'll reassess after Wave 2 whether this stands alone or folds into RQ1a.

**New RQ6 (proposed, provisional) — "Is the entry pool concentrating on specialist employers?"** Entry posting is already specialist (79% of ≥5-SWE companies post zero entry in scraped); Wave 2/3 should test whether this concentration is growing or stable. If growing, it's a clean structural finding compatible with Alternative B.

---

## Gaps and weaknesses

1. **Entry-level cross-period is underpowered.** Any RQ1 entry-specific claim lives in a 8-11 pp MDE neighborhood, which is larger than the plausible real effect. Wave 2 must either (a) avoid headline entry-specific claims, (b) pool differently (e.g., three-level coarse seniority), or (c) explicitly frame RQ1 as "junior-share signal below detection threshold" as a finding.

2. **2026 text coverage is the binding constraint for every text-based analysis.** Only ~12.5k scraped SWE rows have `llm_extraction_coverage = 'labeled'`. The partner should be told to raise the Stage-9 selection target for the next LLM budget run — this is the single highest-leverage action before a follow-up wave.

3. **Entry-level fine structure is blocked by thin 2024 samples.** Even the pooled 2024 entry pool (n=379) is smaller than its 2026 counterpart (n=1,275), and within-subgroup splits (by metro, company size, industry, domain archetype) will run out of cells fast.

4. **We cannot do industry-stratified cross-period pooling** because asaniczka has 0% industry coverage. Any industry finding is arshkon-vs-scraped only.

5. **The 24 cross-company hash collisions need a preprocessing-owner ticket.** They're small in volume (~top 6 companies sharing one description) but suggest residual aggregator relabeling that Stage 4 doesn't catch. Not a Gate 1 blocker, but worth raising.

6. **We still don't know whether the entry-poster specialist class is growing or stable.** T06 established the snapshot but didn't trend it. Wave 3 T16/T17 can close this gap.

7. **No interview or qualitative signal yet (by design).** RQ4 is out of scope for exploration.

---

## Direction for next wave

**Dispatch Agent Prep (Wave 1.5) immediately.** The shared artifact build is the gate for Wave 2. Two constraints to pass through to the Prep agent:

- The cleaned-text artifact should **only include `text_source = 'llm'` rows for boilerplate-sensitive analyses**, and must record the `text_source` distribution in the README. Downstream tasks that need recall (binary keyword presence) can use raw text; everything else (embeddings, term freq, topic models) must filter to `llm`.
- The calibration table should include AI mention rate (expected SNR 925), description length (2.29), YOE ≤ 2 (0.98), mean YOE (0.80), `seniority_final` entry share (0.33), and at least 15-20 more common metrics. Downstream agents will consult this instead of recomputing.

**Wave 2 dispatch modifications (to pass to agents via task prompts):**

- **Agent E (T08):** Read T06's `concentration_predictions.csv` first. Apply the cap 50/company recommendation to term-frequency and scope-inflation analyses by default. Always report `seniority_final` AND YOE ≤ 2 for entry-share changes. Explicitly test Alternative A (instrument-dominated) and Alternative B (composition-dominated) as competing explanations for any aggregate change observed.
- **Agent F (T09):** The dominant-structure NMI check is now **critical** for Gate 2. If archetypes organize by tech domain (frontend/backend/data/ML-AI) and ML-AI grew while entry-poor domains shrank, we have direct evidence for Alternative B. Label the archetypes descriptively and save cluster labels to `shared/swe_archetype_labels.parquet` so T08, T11, T16, T20, T28 can all stratify.
- **Agent G (T10, T11):** T11's domain-stratified scope inflation now depends on T09's archetype labels. For the management indicator, validate ≥80% precision on a 50-row sample stratified by period — prior runs found 3-5× inflation from generic patterns.
- **Agent H (T13, T12):** T13 runs first. The section-anatomy step is the critical test of whether the 56% length growth is in requirements (real) or benefits/about-company (boilerplate). T12 depends on it.
- **Agent I (T14, T15):** T15's convergence analysis must run the within-2024 calibration as a first-class check, not an appendix. If asaniczka→arshkon similarity shift > arshkon→scraped shift, the convergence signal does not survive calibration.

**Tell every Wave 2 agent:** the RQ1 junior story is under threat; we are not defending it, we are testing it. Alternatives A and B are live. Report findings that support any of the three framings with the same rigor.

---

## Current paper positioning

**If we stopped here and wrote a paper today, it would be a "data + methods" paper** with three contributions:

1. **A longitudinal SWE postings dataset** with transparent preprocessing, three independent sources, and a demonstrated LLM-budget-aware frame. BLS r=0.97 gives external validity for the geographic frame.
2. **A robustness framework** showing how within-source cross-source calibration (arshkon vs asaniczka) produces SNR diagnostics that kill most aggregate RQ1-style findings. The `seniority_final` ↔ YOE-proxy direction flip on an overlap panel would be the methodological headline — "specification dependence is a first-class threat to validity for longitudinal posting research."
3. **A single robust empirical finding — the AI requirement explosion** — reported against the macro backdrop of a cooling JOLTS info-sector labor market, framed as an employer-side signal that is decoupled from the direction of overall hiring.

**What Wave 2 needs to deliver to strengthen this:**

- **A within-company scope result** on the 115-company overlap panel. If scope language rose within the same companies (cap-50 per company, keyword-validated), we recover a weakened RQ1b.
- **A domain archetype structure** from T09. If ML/AI is growing and entry-poor, between-domain composition explains the aggregate junior story and Alternative B becomes the lead.
- **A senior archetype shift result** from T21 within the overlap panel — well-powered and uncontaminated by the entry-sample problems.
- **Section anatomy of the length growth** from T13. If length grew in benefits/about-company rather than requirements, almost every scope-inflation finding has to be length-normalized and re-reported.

If Wave 2 delivers any two of these, we have a substantive empirical paper. If it delivers none, the dataset+methods framing is the honest positioning.

---

## Decisions going into Wave 1.5 and Wave 2

1. **`seniority_final` is the primary seniority column.** Downstream uses it without re-litigating. T03 is the canonical reference.
2. **Every entry-related finding must report `seniority_final` AND YOE ≤ 2 side by side.** If they disagree in direction or magnitude, the disagreement is itself the finding to investigate.
3. **The entry-level cross-period pooling is pooled-2024 vs scraped.** Arshkon-only entry is underpowered and should be a sensitivity check, not the primary.
4. **Text analyses are capped at ~12.5k scraped SWE rows** (the `llm_extraction_coverage = 'labeled'` subset). Raw-text backfill is only acceptable for binary keyword presence.
5. **`seniority_native = 'entry'` in arshkon is NOT a usable 2024 baseline.** T03 established this; it should not appear as a primary anywhere in Wave 2.
6. **Asaniczka `associate` is NOT a usable junior proxy.** Drop it from any RQ1 analysis.
7. **Capping 50 postings per `company_name_canonical` is the default for corpus-level term frequency, topic models, co-occurrence, and scope indicator analyses**, per T06's concentration prediction table.
8. **The scraped entry pool has ~10-14% coming from Walmart+TikTok+ByteDance.** Every entry-level aggregate must report the exclude-top-3 sensitivity.
9. **Indeed cross-validation is a first-class sensitivity**, not optional — the Wave 1 finding that Indeed matches LinkedIn within 0.1 pp on entry share is a real constraint on "LinkedIn-specific artifact" explanations.

Gate 1 clears. Proceeding to Wave 1.5 (Agent Prep).
