# Gate 2 — Post-Wave-2 + V1 Research Memo

Date: 2026-04-20
Author: Orchestrator
Written: After Wave 2 (T08-T15) and V1 verification complete; before Wave 3 dispatch.

This memo consolidates Wave 2's structural-discovery findings with V1's adversarial verification corrections. It is the first gate where the paper's narrative substantively evolves and is the primary input to the Wave 3 dispatch prompts.

---

## What we learned

### 1. The dominant axis is technology domain, not seniority

**T09 verified by V1 at NMI ratio 8.88× (title_archetype vs seniority_3level).** BERTopic and NMF agree ordinally: title > tech > period > seniority. The paper's natural narrative axis has shifted from the pre-exploration RQ1 framing ("junior rung narrowing + senior archetype shift") to a **technology-ecosystem restructuring** framing with seniority as a cross-cut.

Biggest archetype growth: ML/LLM cluster (+4.8 pp, 1.65% → 6.47%). New-in-2026: AI-agent orchestration archetype (10× growth, 2 → 45 postings — thin but directionally real). Biggest shrinkage: DoD/clearance cluster (−2.1 pp). Generic Java backend −2.0 pp.

### 2. Seniority boundaries SHARPENED, not blurred

**T15 verified by V1 at TF-IDF junior-senior cosine 0.950 → 0.871.** Both embedding and TF-IDF representations agree; direction and magnitude match Wave 2. Nearest-neighbor analysis: 2026 junior pulls 2024-junior neighbors at 2.79× base rate; 2024-senior neighbors at or below base rate. **The RQ1 core subclaim "junior roles being relabeled as senior" is falsified.** T12's label-based AND YOE-based relabeling diagnostics independently agree: 2026 entry ≠ 2024 senior. This is the cleanest negative result of Wave 2.

Additionally, 2026 groups are MORE tightly clustered within-tier than 2024 groups (within-group cosine 0.504 → 0.535 for junior; 0.513 → 0.562 for senior) — stylistic homogenization within each tier.

### 3. Scope inflation is universal, not junior-specific

**T11 verified by V1: S4 breadth +2.60 > J3 breadth +1.56, length-residualized.** V1 re-fit the residualization formula from scratch; residuals match Wave 2 within 0.02. Within-company decomposition (V1 Phase A): S4 within-company breadth +1.97 vs J3 within-company +1.43. **Senior-side scope inflation is larger than junior-side, and the senior rise survives within-company restriction — it is not an artifact of firm composition shifts.**

The initial "junior scope inflation" framing of RQ1 is weakened. The correct framing is "SWE scope broadened across all seniority levels, with senior postings inflating more in absolute terms."

### 4. Length growth is driven by responsibilities and boilerplate — the requirements question is now narrower and classifier-sensitive

**T13 verified by V1 exactly:** requirements-section share −3.5 pp (23.6% → 20.1%); responsibilities +49%; benefits +89%; legal +80%. For J3 specifically, requirements chars dropped −5.4% absolute (1,057 → 1,000).

**V1 caveat (important):** the J3 requirements-shrank result is **classifier-sensitive**. V1 built an independent simpler regex classifier and got J3 requirements UP +88%. The direction flips. Both classifiers are imperfect (T13's handles markdown + fused text; V1's simpler regex may over-credit content as "requirements" when headers are missing). Neither is ground truth.

**Gate 2 ruling:** the "junior requirements chars shrank" claim is DEMOTED from strong to flagged/moderate. What survives cleanly: **responsibilities +49% and benefits +89% are direction-robust across classifiers and above noise**. That's enough for the boilerplate-and-responsibilities-led length growth story without betting on the junior-requirements-specific direction.

### 5. The AI-rewriting signal is robust and period-dominant

**T12 AI-term acceleration verified by V1 exactly** (RAG 75.3×, multimodal 31.1×, MCP 28.8×, multi-agent 23.6×). **T14 phi-crystallization verified**: `pinecone × weaviate` 0.00 → 0.71; `rag × llm` 0.20 → 0.51. The AI rise is:

- **Real and large.** T08 AI-strict +13.3 pp, SNR 32.9 (far above noise). V1-validated AI-strict pattern has 0.86 semantic precision.
- **Content-driven, not boilerplate-driven.** Section-filtered FW (T12) shows the AI terms are concentrated in requirements + responsibilities, not in benefits/legal. Tension with T13's J3-requirements-shrank claim is resolved by Tension B (see below) — AI tokens densified into smaller junior requirement sections.
- **Period-dominant, not relabeling.** Label-based AND YOE-based relabeling diagnostics from T12 agree: 2026 entry content ≠ 2024 senior content.
- **Within-company driven** for the returning-cohort (T06: +31 pp of +39 pp aggregate is within-firm).

### 6. Technology stacks expanded; AI is additive, not substitutive

**T14 verified:** AI-mentioning postings carry 11.35 techs vs 6.81 for non-AI; length-normalized density 2.79 vs 1.88 per 1K chars. The richer-stack-with-AI pattern survives length normalization at ~70% magnitude. Median techs per posting: J3 4 → 6; S4 5 → 7. CI/CD is the single largest S4 riser at +20.6 pp — bigger than any AI term. Orchestration work at senior tier is real.

### 7. The within-vs-between decomposition is subsample-specific

**V1 resolution of Tension A.** Gate 1 cited T06's finding of ~0 within-company J3 rise (arshkon-only n=125 panel). T08 found +3.5 to +6.4 pp within-company J3 rise (pooled / asaniczka / min-N panels). V1 confirmed both independently: arshkon-only n=125 within-company = −0.22 pp (near zero); pooled min-5 n=356 = +3.95 pp; asaniczka min-5 n=254 = +8.24 pp; pooled min-10 n=160 = +4.87 pp.

**These measure different things.** arshkon-only within-company is the cleanest test (arshkon firms are the most comparable across periods). Pooled within-company pulls in asaniczka, which has the LLM-frame selection artifact and the asaniczka entry-label gap. V1 recommends **pooled-min-5 as the defensible primary for within-company claims, with arshkon-min-5 as co-primary**. The aggregate J3 rise of +5 pp is a mix of within-firm rewriting (on the pooled panel) AND composition shift (arshkon new-entrants). The composition-only framing from Gate 1 was overstated.

### 8. Management density fell on length-normalized measures — but the measure itself is suspect

**T11 finding:** mgmt_strong_density −0.02, mgmt_broad_density −0.04 per 1K chars. Counter-intuitive vs the RQ2 "senior shift away from management" narrative.

**V1 correction (critical for Gate 3):** the management patterns have poor semantic precision.
- `mgmt_broad` (including lead / team / stakeholder / coordinate / manage): V1 precision **0.28** — 72% of matches are FP. This is unusable.
- `mgmt_strict` (mentor / coach / hire / headcount / performance_review / direct_reports): V1 precision **0.55**. Worst sub-patterns: `hire` 0.07 (captures "contract-to-hire", "how we hire"), `performance_review` 0.25 (captures "code review").

**The management density finding is demoted from a substantive claim to a measurement issue.** V1 has rebuilt `mgmt_strict_v1_rebuilt` requiring mentor+(junior|engineers|team|peers) co-occurrence. Wave 3.5 T22 must validate the rebuilt pattern before primary use. **The RQ2 senior-role shift (management → orchestration) is now UNTESTABLE at the requested rigor until T22 validates the rebuilt pattern.**

### 9. Credential stacking aggregate is below noise — only per-seniority claims hold

**V1 flag:** T11's credential_stack_depth aggregate +0.20 has SNR 0.59 (below within-2024 calibration noise). The per-seniority claims are still above noise: J3 stack depth share ≥ 5 categories +16.9 pp (43% → 60%); S4 +13.3 pp (52% → 66%). **Wave 3+ must cite per-seniority credential-stacking, not aggregate.**

---

## What surprised us

1. **T09 NMI ratio is more extreme than the orchestrator's Gate 0 prior.** Expected the paper's axis might pivot; got an 8.88× ratio — a definitive pivot. The domain-dominant structure is not a subtle preference; it's the structure.

2. **Seniority boundary SHARPENING is counter-narrative.** The initial RQ1 framing had a "blurring" subclaim baked in. Sharpening is the opposite result. This is a cleaner negative result than most papers achieve and is worth foregrounding — seniority is NOT becoming ambiguous; it is becoming more distinctively assigned.

3. **Senior scope inflation > junior scope inflation.** This was not a hypothesized direction. In a world where AI is supposed to be "raising the bar for juniors," we see the bar rising MORE for seniors. Within-company (V1 verified) confirms this is real firm behavior, not composition shift.

4. **J3 junior requirements section may have shrunk in absolute chars** (T13) — but V1 showed this flips under a different classifier. The finding that SURVIVES is: junior requirement sections densified (more tech/AI tokens per char) even as total length went elsewhere. The J3 copilot rate (4.6%) is HIGHER than the S4 copilot rate (4.1%) — AI tools appear at junior level at least as much as at senior.

5. **"Senior" title-marker share dropped 12 pp** (T10) while "staff" doubled and unmarked titles rose. Title-level seniority is drifting away from content-level seniority. The labels and the work are decoupling.

6. **CI/CD is the single largest S4 riser at +20.6 pp** — bigger than any AI term. The senior-work content shift has a CI/CD backbone, not just AI. This is new and could be a secondary paper claim.

7. **Financial-services firms dominate 2026 outlier AI-rich postings** (T11 top-1%: JPM, Citi, GEICO, Visa, Wells, Solera). AI adoption in SWE postings is not just tech-industry; finance is writing the most AI-saturated JDs.

8. **Management density fall is a measurement failure, not a substantive finding.** V1's 0.28 precision on mgmt_broad means Wave 2's "management down" claim is an artifact of matching generic collaboration terms. This is an important null / measurement result.

9. **MCP contamination in 2024 baselines.** "MCP = Microsoft Certified Professional" in 2024 postings contaminates any MCP-growth claim using `ai_broad`. Fix: use `ai_strict` for MCP claims or drop MCP from broad-baseline comparisons.

---

## Evidence assessment

| Claim | Strength | Sample | Calibration (SNR) | V1 verification | Qualifications |
|---|---|---|---|---|---|
| Domain-axis dominance (T09 NMI 8.88×) | **Strong** | 8,000 balanced | — | ✅ verified | Title-archetype regex is reasonable proxy; NMF confirms ordinally |
| Seniority boundary sharpening (T15) | **Strong** | ~5,500 stratified | SNR ~1.8 | ✅ verified | Cross-period change only ~1.8× within-2024; real but not dominant |
| Scope inflation universal, senior > junior (T11 H6) | **Strong** | Full corpus | SNR > 2 (seniority-wise) | ✅ verified within-company | Within-company S4 +1.97 > J3 +1.43 is the defensible claim |
| AI-rewriting +13.3 pp, period-dominant (T08/T12) | **Strong** | Full corpus | SNR 32.9 | ✅ verified | AI-strict pattern 0.86 precision; MCP via ai_broad contaminates 2024 baseline |
| AI-term acceleration (RAG 75×, etc.) (T12/H5) | **Strong** | Full corpus | — | ✅ verified | Token-rate methodology replicated exactly |
| Tech stacks expanded, AI additive (T14) | **Strong** | Full corpus | survives density | ✅ verified | 70% magnitude under density normalization |
| Length growth dominated by boilerplate (T13) | **Moderate** | Full corpus | SNR 4.87 benefits | ✅ partial | Benefits +89% and responsibilities +49% are direction-robust. **Junior-requirements-shrank is classifier-sensitive — demote.** |
| Within-company AI rewriting +31 pp (T06) | **Strong** | n=125 arshkon overlap | SNR 32.9 | ✅ verified | Cleanest within-firm AI signal |
| Within-company J3 rise (T08 +3.5-6.4 pp) | **Moderate** | Multiple panels | SNR < 2 | ✅ verified (both panels) | Panel-specific; use pooled-min-5 primary with arshkon-min-5 co-primary |
| CI/CD S4 riser +20.6 pp (T14/T15) | **Moderate** | Full corpus | — | Not independently re-derived | Worth flagging for Wave 3 T21 amplification |
| Management density fell (T11) | **Weak (measurement failure)** | Full corpus | — | ❌ V1 flagged | mgmt_broad precision 0.28, mgmt_strict 0.55 — untestable until T22 rebuilt-pattern validation |
| Credential stacking aggregate +0.20 | **Below noise** | Full corpus | SNR 0.59 | ❌ V1 flagged | Drop aggregate; keep per-seniority J3 +16.9 pp / S4 +13.3 pp |
| Junior requirements chars shrank (T13) | **Flagged** | J3 n=4,578 | — | ⚠️ classifier-sensitive | Direction flips under alt classifier; demote to "densified, direction uncertain" |

---

## Seniority panel — status unchanged from Gate 1

Primary remains J3 / S4 with pooled-2024 baseline; arshkon-only co-primary for senior claims. T30 4-row panel mandatory for every seniority-stratified finding. V1 audit confirms Wave 2 mostly followed this; a few T12/T13 cells used arshkon-only without the pooled sibling — Wave 3 agents must include both.

T08's T30 panels are clean. T11 reported J1/J2/J3/J4 and S1/S2/S3/S4 for breadth and stack depth. T15 used seniority_3level (ok given the Wave 2 scope). **Wave 3 agents must continue reporting 4-row panels.**

---

## Narrative evaluation

### RQ1 — Junior rung narrowing / scope inflation

**Status: substantially reframed.**

- "Junior share reduces" — **contradicted**. J3 share rose +5 pp aggregate. Direction UP.
- "Junior scope inflates" — **partially supported, but not junior-specific**. Scope rose more for senior than junior (T11 H6, verified by V1 within-company).
- "Junior roles relabeled as senior" — **contradicted** (T12 + T15). Boundaries sharpened, not blurred. 2026 entry ≠ 2024 senior.
- "Within-firm employer restructuring" — **partially supported**. Within-firm AI rewriting is +31 pp (strong). Within-firm J3 rise is subsample-dependent. Full within-firm junior scope inflation is weaker than expected.

**Proposed RQ1-revised:** "How did SWE posting content restructure between 2024 and 2026 across seniority and technology domain? Which changes are within-firm rewriting vs composition shifts?" This preserves the empirical framing without committing to a direction.

### RQ2 — Task and requirement migration

**Status: UNTESTABLE at current rigor pending T22/Wave-3.5 pattern rebuilds.**

The core RQ2 claim — specific requirements (system design, CI/CD, AI tools, mentorship) migrating across seniority levels — requires validated semantic patterns. V1 found the management patterns failed (0.28 broad / 0.55 strict). Scope terms at 0.89 and AI-strict at 0.86 are usable. CI/CD is captured reasonably.

- "AI requirements migrating to junior" — **contradicted** by the copilot rate: J3 4.6% > S4 4.1%. AI is at junior level at LEAST as much as senior. But this is "no migration needed; AI was there already at junior" or "AI is at junior because junior postings are where AI enters the hiring funnel."
- "CI/CD migrating" — **directionally supported**. S4 +20.6 pp rise is the cleanest senior-specific tech rise. Whether it migrated FROM senior to junior or independently rose at all levels is Wave 3 T20 territory.
- "Mentorship migrating" — **depends on T22-validated patterns**. Current mentor-binary precision (V1 rebuilt) is 0.78 borderline. Wave 3 T21 and Wave 3.5 T34 carry this.

**Proposed RQ2-revised:** "Which specific requirement categories moved between seniority levels 2024→2026, and which moved universally across levels?"

### RQ3 — Employer-requirement / worker-usage divergence

**Status: on track; Wave 3 T23 + Wave 3.5 T32 are the primary tests.**

The AI-strict +13.3 pp prevalence rise (SNR 32.9) is the employer-side side of the RQ3 measurement. The Wave 2 findings don't touch worker-side usage directly. T23 and T32 carry this forward. Gate 3 will evaluate.

### RQ4 — Mechanisms

Out of scope for computational exploration.

---

## Emerging narrative

**Current leading framing candidate (revised at Gate 2):**

> Between 2024 and 2026, software-engineering job postings reorganized along technology-domain and content axes rather than along seniority. The largest structural shift is the emergence of an ML/LLM/AI-agent archetype within the posting space; the largest content shift is the rise of AI-tooling and platform-infrastructure language (workflows, pipelines, observability, orchestration) in requirements sections across all seniority levels. Seniority boundaries *sharpened* rather than blurred, and scope broadened universally — not as a flow from senior to junior, but as a level-up across the ladder. Length growth is concentrated in responsibilities and boilerplate (benefits/legal) rather than in requirements. Employer-side restructuring is real, with strong within-firm AI-rewriting; the aggregate junior-share rise is a mix of within-firm rewriting and market recomposition.

**Plausible paper leads, ranked by V1-adjusted evidence × novelty × narrative value:**

1. **Technology-domain restructuring with AI-archetype emergence** (T09 + T12 + T14 + T35 pending). Strong evidence, high novelty vs prior labor literature framing SWE as single occupation. Core paper axis.

2. **Universal scope broadening with sharpened seniority boundaries — not junior relabeling** (T11 + T15 + T12). Strong evidence, high novelty (counter to prior "junior scope inflation" narratives including our own). Core paper second section.

3. **Within-firm AI rewriting at scale** (T06 +31 pp, T08 confirmation, T16 pending). Strong evidence. Decomposes RQ1's "employer restructuring" into within-firm content change (strong) + between-firm composition (partial).

4. **Employer-requirement / worker-usage divergence** (T23 + T32 pending). TBD — Wave 3 task. Could promote to lead if cross-occupation universal.

5. **Boilerplate-led length growth with responsibilities expansion** (T13 confirmed ex-junior-requirements claim). Supporting, methodological contribution to how longitudinal posting studies handle length.

6. **Measurement finding: management patterns require semantic validation in longitudinal posting studies** (V1 → T22 rebuilt). Supporting, methods contribution.

---

## Research question evolution

Proposed Gate-2 revisions (deferred final commitment to Gate 3):

- **RQ1-revised:** "How did SWE posting content and composition restructure 2024→2026 across seniority and technology domain? What share is within-firm rewriting vs market recomposition?"
- **RQ1a (new):** "Do seniority boundaries blur, sharpen, or shift in content between periods?" — answer: sharpened (T15). Keep as an explicit finding, not just a subclaim.
- **RQ1b (new):** "Is 2024→2026 scope change within-seniority or cross-seniority flow?" — answer: within-seniority (at all levels), not cross-seniority flow. Senior > junior.
- **RQ2-revised:** "Which specific requirement categories (AI tools, CI/CD, system design, mentorship, scope) moved between seniority levels 2024→2026, and which moved universally?"
- **RQ3:** unchanged — Wave 3 decides.

**New candidate RQ (post-Wave-2, induced):**

- **RQ-candidate T:** "Does the 2024→2026 SWE posting restructuring decompose into a technology-ecosystem reorganization (ML/LLM archetype emergence, platform infrastructure rise, legacy decline) with seniority as a cross-cut rather than the primary axis?"

Gate 3 will finalize.

---

## Gaps and weaknesses

1. **Management patterns failed semantic precision.** Wave 3 T21 and Wave 3.5 T22 + T34 carry the senior-role-shift story and must use V1's rebuilt patterns. The RQ2 mentorship migration question cannot be answered cleanly until T22 validates.

2. **H3 junior-requirements-chars direction is classifier-sensitive.** Wave 3.5 T33 (hiring-bar regression on requirements-section share) must use T13's classifier AND run a sensitivity with an alternative classifier. If the regression's conclusion flips, the classifier choice becomes a material caveat.

3. **Between-co vs within-co subsample dependence.** Pooled-min-5 is the V1-recommended primary but this is asaniczka-dominated. Wave 3 T16 and Wave 3.5 T31 must explicitly distinguish arshkon-only and pooled within-company estimates.

4. **MCP contamination in ai_broad for 2024 baselines.** Not a huge issue — most AI-broad claims use 2026 denominators. But Wave 3 T22 must note this when citing MCP-specific growth.

5. **Cross-task citation transparency.** V1 flagged a couple of T08/T12 citations combining arshkon-only with pooled denominators into the same claim. Not material yet, but Wave 3 must not compound the error.

6. **The T10 disappearing-titles list is thin** (n=2 strict, n=11 at arshkon≥5). Wave 3.5 T36 substitution map will have limited scope.

7. **T09 archetype labels are 8,000 of 48,223 labeled rows (17%).** T28 and T34 project via nearest-centroid from embeddings — acceptable but introduces noise. Wave 3+ should note coverage when citing archetype-stratified claims.

---

## Direction for Wave 3

Based on Gate 2 evidence, Wave 3 dispatch prompts carry the following guidance (in addition to Gate 0 + Gate 1 pre-commits):

### Gate 2 pre-commits (ADDITIVE — do not remove Gate 0/1 rules)

1. **Load V1-validated patterns from `exploration/artifacts/shared/validated_mgmt_patterns.json` as the DEFAULT.** Do NOT re-derive management/AI/scope patterns. Use `mgmt_strict_v1_rebuilt` for any management density claim. Use `ai_strict` (note: fine-tuning sub-pattern has 0.47 precision in 2024 — restrict to LLM-adjacent context or drop from 2024-baseline ratios). For MCP specifically, use `ai_strict` only (drop from `ai_broad`).

2. **Management claims are provisional until T22 validates rebuilt patterns.** Wave 3 T21 must report management/orchestration/strategic density using V1's rebuilt pattern AND flag the 80% semantic threshold as unmet if T21's own validation doesn't reach it. No "management fell" or "orchestration rose" claim is a lead finding until Wave 3.5 T22 validates.

3. **Within-company decomposition reporting:** report both arshkon-only and pooled-min-5 magnitudes for any within-vs-between claim. They measure different things and the reader needs both.

4. **Classifier sensitivity for section-based claims:** Wave 3 T18 and Wave 3.5 T33 must report findings under T13's classifier AND an alternative (simpler regex or alternative labels set). If direction flips, the claim is demoted.

5. **Credential stacking aggregate is dropped.** Cite only per-seniority J3 +16.9 pp and S4 +13.3 pp.

6. **Scope inflation framing is "universal, senior > junior" not "junior-specific."** Wave 3 T16/T28 should decompose scope/breadth changes within and between archetypes under this framing.

7. **Seniority boundary sharpening is an established finding.** T20 (boundary classifier) should treat this as the prior to test, and report whether supervised-feature discriminability corroborates T15's unsupervised finding.

### Task-level Wave 3 guidance

**T16 (company strategies):** explicitly report arshkon-only vs pooled within-company estimates for each metric. The T06 vs T08 tension must not resurface silently. Save the overlap panel and company change vectors to `exploration/tables/T16/` for Wave 3.5 T31/T37/T38 handoff.

**T17 (geographic):** apply domain-archetype stratification (T09 labels available). Check whether tech-hub vs non-tech-hub metros show different archetype mixes.

**T18 (cross-occupation boundaries):** this is the RQ3 universality gate. If SWE-adjacent and control show the same length growth and AI-mention pattern at reduced magnitude, the paper's novelty shrinks. Use V1-validated AI-strict pattern. Run DiD formally.

**T19 (temporal patterns):** straightforward. Ensure within-2024 calibration + annotated timeline.

**T20 (seniority boundary classifier):** load `T11_posting_features.parquet`. Prior: sharpened. Test whether logistic-regression AUC on features also shows sharpening. Report Δgap per feature, attribution by side (senior vs junior). Continuous YOE × period interaction is the cleanest complement to T15.

**T21 (senior role evolution):** MUST use V1-validated patterns. Report both the original T11 management patterns AND the V1-rebuilt `mgmt_strict_v1_rebuilt` sub-pattern. Cluster senior postings by language profile; save cluster assignments to `exploration/tables/T21/cluster_assignments.csv` for Wave 3.5 T34.

**T22 (ghost forensics):** aggregator-stratified as its primary axis. MUST produce `exploration/artifacts/shared/validated_mgmt_patterns.json` with measured precision — V1 has already shipped a v1 version; T22 can extend or refine but must not delete. Validate V1's 3 rebuilt patterns semantically.

**T23 (employer-usage divergence):** use V1-validated AI-strict. Report under 50%/65%/75%/85% worker-usage assumption bands. Produce the divergence chart.

**T28 (domain-stratified scope changes):** load T09 archetype labels. Decompose J3 share change AND S4 share change within-archetype vs between-archetype. If between-archetype dominates, the Wave 2 scope inflation story is partly domain-composition.

**T29 (LLM-authorship detection):** exploratory but potentially paper-relevant. If signal, the Wave 2 findings may be partly recruiter-tool mediated. Use T13 readability parquet.

### Wave 3 dispatch ordering

All 5 Wave 3 agents dispatch in parallel. No inter-dependencies during Wave 3 itself (T28 depends on T09 labels which already exist).

---

## Current paper positioning

**If we stopped here, the strongest paper is a hybrid empirical/methods paper** with:

- **Empirical contribution:** SWE posting content reorganized around technology domain (ML/LLM archetype emergence) and universal scope broadening, with sharpened — not blurred — seniority boundaries. AI-tooling and platform-infrastructure language drive the content shift. Within-firm AI-rewriting at +31 pp is the strongest single quantitative finding. Composition-shift is partially real but smaller than Gate 1 suggested.

- **Methods contribution:** the T30 multi-operationalization panel (labeled-vs-YOE-vs-title-keyword), within-2024 calibration SNR framework, composite-score correlation check, length residualization protocol, and adversarial semantic-precision validation (V1). Together these define how to run a longitudinal posting study without the usual failure modes. The management-pattern-validation finding (0.28 broad / 0.55 strict) is directly publishable as a warning to prior literature.

- **Dataset contribution:** 68k-row longitudinal SWE frame with LLM-enriched columns and validated patterns.

**What Wave 3 needs to deliver to strengthen the lead:**

- **T18 cross-occupation:** if control occupations do NOT show the same AI-mention rise, RQ3 becomes a SWE-specific labor-market finding and potentially the paper's lead alongside T09. If they DO show it at similar magnitude, the paper stays SWE-centric but frames AI as a general labor-market shift.

- **T20 corroborates T15 sharpening:** if structured-feature discriminability also sharpened, the boundary-sharpening finding has two independent methods agreeing. Strong.

- **T16 within-firm rewriting at archetype-adjusted unit:** confirms or qualifies the +31 pp within-firm AI rewriting.

- **T22 validates V1 rebuilt patterns** and extends to orchestration/mentorship patterns. If T22 delivers clean patterns, RQ2 becomes testable at Wave 3.5 T34.

- **T28 within-archetype vs between-archetype scope decomposition:** tells us whether scope inflation is a within-domain phenomenon or a market-recomposition effect from domain mix shifting.

- **T29 LLM-authorship:** if the low-score subset shows muted Wave 2 effects, a significant fraction of apparent content change is recruiter-tool-mediated — which is itself paper-lead-material.

---

*Orchestrator note: Wave 3 dispatches immediately after this memo. Wave 3 + Wave 3.5 + V2 together form the pre-synthesis evidence body. Gate 3 is the unified post-synthesis-input memo that Agent N (SYNTHESIS.md) reads as primary input.*
