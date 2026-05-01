# Gate 1 — Post-Wave-1 Research Memo

Date: 2026-04-20
Author: Orchestrator
Written: After Wave 1 (T01–T07 + T30) completes, before Wave 1.5 (Prep) and Wave 2 dispatch.

This memo synthesizes Wave 1 findings, confirms the feasibility envelope for Wave 2+, and sets the seniority-panel primary that every downstream task will use.

---

## What we learned

### 1. The data foundation is queryable but constrained

**68,137 SWE LinkedIn rows** under the default filter (arshkon 4,691 + asaniczka 18,129 + scraped 45,317). Scraped window is 2026-03-20 through 2026-04-17 and still growing. T01/T07 confirm we have enough rows for the headline comparisons at the right T30 definitions. Indeed is excluded from the LLM frame, so `yoe_min_years_llm` is null and `seniority_final` carries only Stage 5 strong-rule labels there — Indeed is a sensitivity axis, not a primary.

### 2. The T30 panel is load-bearing and defensible

**12 of 13 T30 definitions are direction-consistent** (7 of 7 junior-side UP; 5 of 6 senior-side DOWN; S2 director-only is flat — noise). This is the strongest possible prior that seniority direction is not an operationalization artifact.

Primary recommendation: **J3 (`yoe_min_years_llm ≤ 2`) for junior claims and S4 (`yoe_min_years_llm ≥ 5`) for senior claims** under a pooled-2024 baseline. J1/J2 and S1/S2 are label sensitivities. J4/S5 are generosity checks.

**Feasibility envelope (T07):** J3 + pooled-2024 has MDE ≈ 4.3 pp at p=0.5; S4 + pooled-2024 has MDE ≈ 1.7 pp. **Underpowered and sensitivity-only:** J1 on arshkon-only, S2 (director-only) at all baselines. All-SWE comparisons have MDE ≈ 1.1 pp — tightly bounded.

### 3. The labels themselves carry substantial instrument noise

**Platform relabeling drift (T05):** For the top 20 SWE titles shared between arshkon and scraped, `seniority_native` mid-senior share rises +15 to +40 pp on the SAME titles while YOE means stay flat. This is LinkedIn changing label policy, not the market changing. Any `seniority_native`-based trend carries this artifact.

**Senior-side asaniczka asymmetry (T30):** S4 share is +7.1 pp higher on asaniczka than arshkon within 2024. This is comparable to the cross-period S4 effect size (−7.6 pp pooled). Arshkon-only S4 drops to −1.8 pp (not significant at 80% power).

**Implication.** `seniority_native` is effectively unusable for temporal claims. `seniority_final` (rule + LLM) is defensible (T03 kappa 0.66 scraped / 0.45 arshkon against native, though note that arshkon-native itself drifts against the LLM reference). **YOE-based variants (J3/S4) are the label-independent primary.**

### 4. The junior-share rise is between-company; the AI-mention rise is within-company

**This is the single most consequential Wave 1 finding for the paper's narrative.** T06's overlap-panel decomposition (n=125 companies with ≥5 SWE in both arshkon and scraped):

| Metric | Aggregate Δ | Within-company | Between-company |
|---|---|---|---|
| J3 entry share | +5 pp | ≈ 0 | +5 pp |
| AI-mention binary | +39 pp | +31 pp | +8 pp |
| Mean description length | +1,750 chars | ~half | ~half |

The junior-share rise as employer-side restructuring is weakened. Different firms are posting SWE roles in 2026 than in 2024, and those different firms have a higher entry-share. AI-mention rise survives firmly as within-firm rewriting.

### 5. Entry posting is a minority activity

**58-63% of companies with ≥5 SWE posts have ZERO J3 entry rows** (under J1: 77-87% zero). Entry-share is driven by a specialized subset of employers, not a market-wide baseline. Any entry-specialist-weighted finding needs T06's 206-company `entry_specialist_employers.csv` applied as an exclusion sensitivity.

### 6. Macro context: 2026 is in a hiring trough

JOLTS Information openings are at 0.71× of the 2023 average (T07, via FRED). Total nonfarm is at 0.76×. Information is contracting faster than overall. **All cross-period claims must be framed as shares, not volumes.** Volume-based claims conflate demand with the hiring cycle — this is why Wave 3.5 T38 (hiring-selectivity × scope-broadening) exists.

---

## What surprised us

1. **Composition dominance of the junior finding.** Going in, the orchestrator's prior was that some of the junior change would be composition. The between-company ≈ +5 pp / within-company ≈ 0 pp decomposition is cleaner and more extreme than expected. The paper's junior framing must be rebuilt around composition.

2. **AI within-company signal is stronger than expected.** +31 pp within-company is a very large same-firm rewriting signal. This is the cleanest "employer restructuring" evidence in Wave 1 and already supports the Wave 2 AI-rewriting story.

3. **Direction consistency across 12 of 13 T30 definitions.** Wave 0 prior was that YOE-vs-label disagreement would be common. Instead, J3 / J4 / J1 / J2 / J5 / J6 / J3_rule all move UP; S1 / S3 / S4 / S5 / S4_rule all move DOWN. S2 director-only is flat (noise). Magnitude varies; direction is robust. This is a green light on the seniority direction claim across the panel.

4. **Magnitude fragility for senior-side claims.** Despite direction consistency, the magnitude difference between pooled-2024 S4 (−7.6 pp) and arshkon-only S4 (−1.8 pp) is large enough that senior claims need dual reporting. This was anticipated but the size of the gap is materially larger than a "minor sensitivity."

5. **LLM conservatism.** LLM abstains on 34-53% of SWE rows it was called on. This is by-design (the LLM looks for explicit seniority signal), but it means `seniority_final = 'unknown'` share is larger than many downstream agents may expect. **Every label-based claim must report the unknown-share denominator.**

6. **ML Engineer SWE-classification disagreement across 2024 sources.** Arshkon and asaniczka differ by 19.6 pp on whether "ML Engineer" counts as SWE (78% vs 59%). Any ML-engineer-specific analysis must stratify by source. The T09 / T28 archetype work should handle this naturally by clustering on content, but the orchestrator will flag it in every Wave 2+ prompt touching ML/AI.

7. **Industry taxonomy drift.** LinkedIn changed its industry label system between 2024 and 2026 (T07 flagged a +13.1 pp "Software Development" vs −13.2 pp "IT Services and IT Consulting" symmetric swap plus multi-label industry strings appearing only in scraped). Industry-stratified analyses at raw-label level are unreliable. If Wave 2 / Wave 3 wants industry heterogeneity, it must use a cross-taxonomy mapping.

---

## Evidence assessment

| Claim | Strength | Sample | Confounds | Calibration |
|---|---|---|---|---|
| J3 entry share rises +5 pp pooled 2024 → scraped | **Strong (direction)** | 48,452 labeled | Sampling frame (between-co); LLM-frame selection | Within-2024 noise: J3 arshkon vs asaniczka effect size ~2-3 pp → SNR ≈ 2 (borderline) |
| Junior rise is between-company not within-company (n=125 overlap) | **Strong** | 125 companies | Overlap panel is 14% of firms but 55% of 2026 volume | Direct decomposition; robust to J1/J2/J3/J4 panel |
| S4 senior share falls −7.6 pp pooled, −1.8 pp arshkon-only | **Mixed magnitude, consistent direction across S1/S3/S4/S5; S2 flat** | Pooled n≈15,000; arshkon n≈3,500 | Asaniczka senior-asymmetry within 2024 (+7.1 pp) persists under LLM YOE | SNR < 2 on arshkon-only |
| AI-mention binary +39 pp aggregate, +31 pp within-co | **Strong** | 48,452 labeled | Recruiter-LLM authorship mediation (T29 will test); keyword precision (V1 will test) | Within-2024 noise < 3 pp on AI-strict → SNR ~10 |
| Description length +1,750 chars | **Strong but uninterpretable alone** | Full corpus | Boilerplate vs requirements split (T13 will decide) | Within-2024 difference ~400 chars → SNR ≈ 4 |
| `seniority_native` drifts +15-40 pp on same titles | **Strong** | Top-20 shared titles | Confirms platform relabeling | YOE stable → supports artifact interpretation |
| 2,109 returning companies = 55% of 2026 postings | **Descriptive** | Full corpus | — | — |
| BLS OES correlation r = 0.94 log-log (pooled 2024) | **Strong** | 50 states | — | — |

## Seniority panel — canonical primary for Wave 2+

**Junior claims:** Report J1 / J2 / J3 / J4 as a 4-row table. J3 primary. Unanimous or 3-of-4 directional agreement for lead claims. Pooled-2024 baseline (asaniczka + arshkon) unless the specific comparison's MDE requires arshkon-only.

**Senior claims:** Report S1 / S2 / S3 / S4 as a 4-row table. S4 primary. **Both pooled-2024 AND arshkon-only magnitudes must be cited for senior-side claims** — the asaniczka asymmetry makes arshkon-only effectively co-primary, not just a sensitivity. S2 (director-only) is underpowered; expect flat / null results and interpret accordingly.

**Panel artifact:** `exploration/artifacts/shared/seniority_definition_panel.csv` (78 rows, schema documented in T30 spec). All Wave 2+ tasks load this file.

---

## Narrative evaluation

### RQ1 — Junior rung narrowing / scope inflation

**Status: needs reframing.** The junior-share direction (UP at aggregate, not DOWN) is already counter to the initial RQ1 phrasing that hypothesized "reduces junior share." Moreover, the aggregate UP is *between-company* composition, not within-firm change. The "employer-side restructuring" framing of RQ1 weakens.

**Proposed reframe:** "How does the COMPOSITION of SWE hiring change between 2024 and 2026? Is the entry-share change driven by new firms entering the hiring market, by existing firms restructuring, or both?" This becomes a composition-aware question rather than a within-firm claim. The Wave 2 agents can keep hunting for within-firm junior-scope inflation (T11/T12/T13 section-level; T16/T28 archetype-stratified), but the leading finding is already composition.

**Scope inflation (separate claim):** Whether junior postings ASK for more things in 2026 than in 2024 is still open — description length grew, but whether growth is in requirements or boilerplate is T13's call. Do not presuppose the answer. If T13 shows boilerplate-dominant growth, "scope inflation" collapses; if requirements-dominant, it survives conditionally on calibration.

### RQ2 — Task and requirement migration

**Status: testable but high-burden.** AI-mention rise is strong and within-company, which supports the "requirements migrating into postings" framing. Whether requirements migrate DOWNWARD across seniority (the RQ2 specific claim) is Wave 2 territory — T08 junior/senior content comparison and T12/T15 relabeling diagnostics.

**Caveat.** The J3-based primary is label-independent, so "migration into junior postings" means "migration into YOE ≤ 2 postings" — this is a cleaner formulation than "migration into entry-labeled postings" and should be adopted.

### RQ3 — Employer-requirement / worker-usage divergence

**Status: on track, likely strong.** AI-mention +39 pp aggregate / +31 pp within-co far exceeds the worker-usage benchmark movement in any reasonable external benchmark. The direction of divergence is near-guaranteed; the MAGNITUDE and CROSS-OCCUPATION generality (T32) are the paper-lead-material questions.

### RQ4 — Mechanisms (interviews)

Out of scope for computational exploration. Wave 1 provides no update.

---

## Emerging narrative

**Current leading framing candidate (revised from pre-exploration):**

> Between 2024 and 2026, the software-engineering hiring market recomposed. The set of firms posting SWE roles shifted — new entrants disproportionately post entry-level roles, while AI-mention language rose sharply WITHIN returning firms' postings. The aggregate "junior rise" is a composition story, but the "AI rewriting" is a same-firm content story. This decoupling itself is the finding: employer-side restructuring exists, but it is about WHAT employers write about (AI-enabled work), not about the level-mix they hire for.

**Alternative framings to evaluate at Gate 2:**

1. **Within-firm restructuring framing** — if Wave 2 T16 and T28 find strong within-company, within-domain scope changes, the paper can still center employer-side restructuring (just not junior-share).
2. **Technology ecosystem restructuring** — if T09 archetypes are domain-dominant and T14/T35 show crystallizing tech ecosystems, the paper re-leads with technology evolution.
3. **Cross-occupation AI-signal divergence** — if T18 / T32 show SWE-adjacent and control occupations follow the same AI-mention pattern at smaller scale, the contribution generalizes to a labor-market divergence finding.
4. **Recruiter-LLM authorship mediation** — if T29 finds strong authorship signal and the low-LLM-score subset shows muted headline effects, the paper becomes about tooling-mediated content change.
5. **Hiring-trough selectivity** — if T38 finds volume-down companies have the broadest JDs, scope-broadening is partly macro-mediated selectivity, not pure demand change.

The orchestrator is not committing. The Gate 2 memo will weigh them empirically.

---

## Research question evolution

Changes since Gate 0:

- **RQ1.** Reframe from "junior share reduces" to "the composition of SWE hiring shifts and content changes within firms." The specific sub-claim "junior share reduces" is contradicted at the aggregate (it went UP). The specific sub-claim "junior scope inflates" is still testable but shifts to T11 / T13 / T28 as the primary carriers.
- **RQ2.** Retain, but reframe "migration into junior postings" as "migration into YOE ≤ 2 postings" (label-independent).
- **RQ3.** Retain; likely becomes the paper's lead if T32 cross-occupation extension holds.
- **RQ4.** No change.

**Candidate new RQ (post-Wave-1, proposed):**

> **RQ1-revised:** How much of the 2024→2026 change in SWE posting composition is driven by firms entering/exiting the hiring market, vs existing firms changing what they post? And what is the content signature of each component?

Decision on whether to promote this revision: deferred to Gate 2, when Wave 2 T16 and T28 give evidence on within-vs-between at the archetype level.

---

## Gaps and weaknesses

1. **Wave 1 cannot say whether description length growth is boilerplate or content.** T13 (Wave 2) is the decider.
2. **Wave 1 cannot say whether T09 archetypes organize by domain, seniority, or firm type.** T09 (Wave 2) is the decider.
3. **ML Engineer cross-source SWE-classification gap is documented but not mitigated.** Wave 2 ML/AI-relevant tasks must source-stratify or re-classify.
4. **Industry-label drift is documented but not bridged.** Wave 2/3 industry analyses must construct a cross-taxonomy mapping or drop industry heterogeneity.
5. **Asaniczka senior asymmetry is documented but mechanism unclear.** Is asaniczka more senior-skewed because it's a different LinkedIn sampling, or because the LLM pipeline treats its text differently? V1 can investigate in Gate 2.
6. **Scraped `not_selected` = 43%** means text-sensitive and LLM-column analyses lose almost half the scraped rows. Wave 2 text-heavy tasks (T12, T13, T15) must explicitly report labeled-vs-not split.

---

## Direction for Wave 1.5 and Wave 2

### Wave 1.5 (Agent Prep) — mechanical

Standard spec. The one non-mechanical decision: **use the pooled-2024 baseline for the within-2024 calibration table** per T07 and T30 feasibility. Include arshkon-only as a secondary column for senior-side metrics.

### Wave 2 — analytical priorities

1. **T08 (distribution profiling)** — use J3 primary, pooled-2024 baseline; report arshkon-only senior magnitude alongside; stratify all metrics by `is_entry_specialist` (from T06) to separate specialist-firm behavior from the general market.
2. **T09 (archetype discovery)** — this is the strategic pivot point. If clusters are domain-dominant, we reframe the paper; if seniority-dominant, we don't. Methods-comparison (BERTopic vs NMF) is standard but orient the interpretation around the composition-shift finding from T06.
3. **T11 / T13 (complexity + linguistic)** — T13 section anatomy is the single most important Wave 2 decider for scope-inflation claims. T11's section-decomposition of requirement_breadth must be length-residualized (Gate 0 pre-commit) and section-stratified.
4. **T12 (text evolution)** — must run section-filtered (depends on T13 classifier). The pair "Entry 2026 vs Mid-senior 2024" comparison is the cleanest relabeling diagnostic; the YOE-band version (YOE≤2 2026 vs YOE≥5 2024) is label-independent.
5. **T14 / T15 (tech + semantic)** — apply company capping at 20-50 per T06 recommendation; ML-engineer stratification caveat.
6. **T16-setup note (informational, Wave 3):** the arshkon∩scraped overlap panel of 125 companies is our cleanest longitudinal sample. T31 will tighten it to same-co × same-title. The orchestrator will ensure Wave 3 dispatch prompts reference the T06 returning-companies cohort as a co-primary frame.

### Orchestrator decisions to pre-commit into Wave 2 dispatch prompts

- **Primary seniority = J3 / S4 with pooled-2024 baseline; arshkon-only co-primary for senior claims.**
- **Every seniority-stratified finding reports the 4-row T30 panel.**
- **Every cross-period finding reports within-2024 calibration SNR.**
- **Every text-based finding on scraped data reports labeled-vs-not split (Wave 2 dim h).**
- **Entry-specialist subset exclusion is a mandatory sensitivity** for any aggregate entry-share claim (in addition to the standard T30 panel).
- **ML Engineer source-stratification is required** for any ML/AI-specific claim on 2024 data.
- **Industry label drift forbids industry-stratified 2024→2026 claims at raw-label level.**
- **Seniority_native-based temporal claims are forbidden** except as a within-arshkon diagnostic.

---

## Current paper positioning

If we stopped here, the strongest paper we could write is a *methods + documented composition-shift* paper:

- **Dataset contribution:** the 68k-row longitudinal SWE frame with LLM-enriched seniority, YOE, and text columns.
- **Methods contribution:** the T30 multi-operationalization panel, the within-vs-between decomposition framework from T06, the sampling-frame sensitivity approach from T37.
- **Empirical contribution:** "The aggregate junior-share rise 2024 → 2026 is driven by market recomposition, not employer restructuring. The AI-mention rise is within-company and robust." — strong but not a blockbuster.

What Wave 2 needs to deliver to strengthen:

- **T13 confirms the description-length growth is content, not boilerplate.** Then scope claims survive.
- **T09 produces coherent and method-robust archetypes.** Then domain-stratified narratives are defensible.
- **T12 / T15 find a within-YOE-band content evolution signal.** Then "what entry roles ask for" is a content claim, not a composition claim.
- **T11 shows credential stacking rose within the YOE ≤ 2 subset.** Then scope inflation has a concrete quantitative carrier.

If Wave 2 delivers all four, the paper moves from moderate to strong-empirical. If Wave 2 comes back flat on all four, the paper leads with composition-shift + AI-rewriting (the Wave 1 findings) and becomes a more modest but still publishable contribution.

---

*Orchestrator note: Wave 2 dispatches next, inheriting the pre-commits above. Gate 2 will evaluate what T08-T15 found against the composition-shift priors established here.*
