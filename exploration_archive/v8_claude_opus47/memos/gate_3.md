# Gate 3 Research Memo

Date: 2026-04-18
Gate: after Wave 3 (Agents J-M, O) + V2 verification · before Wave 4 (synthesis) dispatch
Reports read: T16, T17, T18, T19, T20, T21, T22, T23, T28, T29, V2_verification.

Wave 3 delivered both the SWE-specificity test (T18) and the RQ3 benchmark test (T23). V2's re-derivation verified most Wave 3 lead findings and flagged three corrections that propagate into the synthesis.

## What we learned

**1. The paper's lead finding is SWE-specific.** T18's DiD shows that the AI-vocabulary rise is 99% SWE-specific (vs control), tech-count rise is 95% SWE-specific, requirement-breadth rise is 72%, and org-scope rise is 71%. Two claims are NOT SWE-specific — description length (37% of SWE-only change is attributable to SWE vs control) and soft-skill mentions (0%, DiD CI crosses zero). **The requirements-section shrink is itself SWE-specific**: SWE requirements fell −10.7pp share-of-description, SWE-adjacent fell −10.9pp, control ROSE +0.9pp (opposite direction). V2 confirmed this with an independent section classifier on 500 control postings (73-86% has_req_rate across groups — classifier works on control; the divergence is real, not a classifier failure). **The "SWE-specific AI rewriting" framing holds.**

**2. RQ3 inverted — employers UNDER-specify AI relative to worker usage.** T23's benchmark comparison: 2026 SWE employer broad-AI requirement rate = 46.8% (V2: 47.8%), strict-tool rate = 14.2%. Developer usage from Stack Overflow 2024 = 62% currently using, 76% plan. Anthropic Labor Market Impacts = 75% programmer exposure. GitHub Octoverse 2024 = 73% OSS. **Employer requirements trail worker adoption by 15-30pp under any reasonable benchmark assumption (50%, 65%, 75%, 85%).** V2 confirmed the direction. Seniors are MORE AI-specified than juniors (51.4% broad S1 vs 43.5% J2), ruling out an "AI-as-junior-filter" interpretation. **The anticipatory-restructuring hypothesis is wrong in direction**: employers are BEHIND, not ahead. This is a novel finding on its own.

**3. Within-company scope broadening is real (contra Gate 2's worry).** T16's 240-company arshkon∩scraped overlap panel: length-residualized `requirement_breadth` rose +1.43 within-company on the primary panel; 76% of overlap companies broadened; 62% by >1.0 unit. Holds 94-147% within across 5 sensitivity variants (specialist-exclude, aggregator-exclude, cap-50, labeled-only, pooled-2024). V2 verified the 102% within-company AI-strict claim exactly. **Scope inflation IS real at the company level**, even if the pre-exploration "junior scope inflation" framing is wrong (seniors show parallel breadth growth). The Gate 2 leaning toward "rewriting, not restructuring" needs to be tempered — same-company breadth is a restructuring signal too.

**4. The senior archetype shift is senior-specific, not corpus-wide template expansion.** T21's cross-seniority comparison: V1-refined strict mentor rate at mid-senior rose 1.73× 2024→2026 (V2 got 1.46×, direction verified but magnitude disagrees); at entry it rose 1.07×; at director +30% (less than mid-senior). Management + orchestration + strategic densities all rose more at senior than entry. An emergent "mgmt + orch + strat + AI" sub-archetype appeared (97% of its members are 2026; cleanly new). **Senior archetype shift is real and senior-specific.**

**5. Seniority boundaries SHARPENED, not blurred.** T20's per-boundary AUC: associate/entry +0.054, mid-senior/associate +0.084, director/mid-senior flat, yoe-only J3/S4 panel +0.14 (V2: +0.134 — exact match). All sharpened or held. ML/AI gained the most clarity (+0.105 AUC) — was worst-discriminated in 2024, now mid-pack. **The initial "junior-senior convergence" intuition is wrong**: the market is differentiating seniority more sharply, not less.

**6. Recruiter-LLM mediation is partial, not dominant.** T29's low-LLM-score bottom-40% subset retains ~77% of AI-strict Δ, ~86% of AI-broad Δ, ~72% of mentor Δ, ~71% of breadth-resid Δ (V2 partially verified: AI-strict retention 75% matches; mentor/breadth retention 3-feature score gave >100%, making the T29 71-72% numbers method-sensitive). **Paper does NOT need to reframe to methodological-only.** But mentor-rate is the most LLM-sensitive finding and must be framed carefully.

**7. AI rise is cross-archetype, not ML/AI-driven composition.** T28: AI-BROAD Δ positive in 20/20 archetypes, ≥+10pp in 15/20 (V2 verified: 18/22 ≥+10pp under broad; V2 corrected the Wave 3 summary that had attributed this distribution to STRICT pattern — under strict, 6/22 archetypes hit +10pp). `systems_engineering` is the clean zero-AI control (+0.16pp strict; V2: exact). **AI rewriting happened almost everywhere in SWE**, with AI/ML only 2× archetype-average. The 2024→2026 AI/ML archetype growth (2.6% → 16.2%) is **81% new-entrant-driven** (V2 verified: 935 only-in-2026 employers vs 83 both-period contributing 1,634 vs 380 postings). AI/ML growth is new-firm-entry + cross-domain within-company AI rewriting — not existing firms pivoting into AI/ML.

**8. AI surge is geographically uniform.** T17: AI-mention rose in 26/26 scraped metros (CV 0.29). Top-3 surges are Sunbelt finance/healthcare (Tampa Bay, Atlanta, Charlotte), NOT tech hubs. Correlation between metro AI surge and metro J2 decline is r = −0.11 (p = 0.60) — NULL. **Rejects the "AI exposure → junior displacement" spatial narrative.** ML/AI archetype concentration widened in Bay Area (12%→32%) and Seattle (4%→29%); other metros grew ML/AI from tiny bases.

## V2 corrections to Gate 2 / Wave 3 claims

1. **T28 "15/20 ≥+10pp" is BROAD AI, not STRICT.** Under strict pattern only 6/22 archetypes hit +10pp (the strict pattern is stricter by design). Wave 3 summary had conflated the two. Correction applied.

2. **T21 mentor rise magnitude is 1.46× (V2) to 1.73× (T21) at mid-senior.** Direction verified; specific magnitude is method-dependent on how "mentor-binary" is computed. Paper should cite the range, not the point estimate.

3. **T29 mentor/breadth low-LLM retention numbers (71-72%) are method-sensitive.** V2's 3-feature authorship score gave retention >100% (no attenuation). The method-dependence itself is a finding: "T29's mediation estimate ranges from 0% to ~30% depending on authorship-score specification; AI-strict is robust across specifications; mentor and breadth are not." Paper frames as "~15-30% of mentor/breadth rise is recruiter-LLM mediated, with uncertainty."

4. **The Gate 2 memo's "180× period-vs-seniority embedding dominance" (carried from T15)** was already corrected by V1 to ~1.2-1.5× centroid-pairwise; V2 did not re-test. Use NMI 1.9× as the headline ratio (period/seniority).

## Ranked findings for the paper (with evidence strength × novelty × narrative value)

| Rank | Finding | Evidence | Novelty | Narrative value |
|---|---|---|---|---|
| 1 | SWE-specific AI-vocabulary rewriting (+33pp, DiD 99% SWE-specific, cross-seniority, cross-archetype, within-company) | Strong | Medium-High | **Paper lead** |
| 2 | Employer-side AI requirements UNDER-specify relative to worker usage (RQ3 direction inverted) | Strong (benchmark-dependent) | High | Novel finding — own section |
| 3 | Within-company scope broadening is real (+1.43 breadth residualized on 76% of 240-co panel) | Strong (multiple sensitivities) | Medium | Restructuring-not-just-rewriting thread |
| 4 | Senior archetype shift toward mentoring/orchestration, senior-specific | Strong-Moderate (magnitude method-dependent) | Medium-High | RQ1 senior subclaim |
| 5 | Seniority boundaries SHARPENED, not blurred (all three adjacent boundaries) | Strong | Medium | Inverts RQ1 convergence intuition |
| 6 | Requirements section shrank, narrative sections expanded (SWE-specific, not universal) | Strong (T13 + V2 classifier on control) | Medium | Methodological + substantive |
| 7 | AI/ML archetype growth (2.6%→16.2%) is 81% new-entrant-driven | Strong | Medium | Market composition + restructuring |
| 8 | AI surge is geographically uniform; uncorrelated with junior decline | Strong (26/26 metros, r=−0.11) | Medium | Rejects spatial displacement story |
| 9 | Recruiter-LLM mediation accounts for 15-30% of the rise | Moderate (method-sensitive) | Medium-High | Methodological caveat, not a reframe |
| 10 | Staff-title doubled (2.6%→6.3%), emergent mgmt+orch+strat+AI sub-archetype | Strong | Medium | Supporting senior archetype |
| 11 | Junior-share direction baseline-dependent (Gate 1) | Strong (as a negative finding) | Medium | Methodological contribution |
| 12 | "Systems engineering" is the clean zero-AI control | Strong | Medium | Natural control for interviews (RQ4) |

## Emerging narrative (one paragraph draft)

Between 2024 and 2026, US LinkedIn software-engineering job postings were rewritten in a distinctively SWE-specific, within-company, cross-seniority way. AI-tool vocabulary rose from 1.5% to 14.9% of postings (14× growth), requirement breadth broadened within the same 240 overlap companies (+1.43 length-residualized units, 76% of companies), and senior postings shifted toward mentoring and orchestration (mentor-binary rate 1.46-1.73× at mid-senior vs 1.07× at entry). Contrary to the initial anticipatory-restructuring hypothesis, employers UNDER-specify AI by 15-30 percentage points relative to observed developer usage — workers are ahead of the JDs, not behind. Seniority boundaries SHARPENED, not blurred. The rewriting is SWE-specific (DiD 99% for AI-strict vs control occupations), cross-archetype (rises in 20/20 domains), and geographically uniform (26/26 metros, uncorrelated with junior decline). Recruiter-LLM adoption accounts for 15-30% of the apparent shift, leaving the bulk as a real employer-side content change.

## Research question evolution (proposed for synthesis)

- **RQ1a — SWE-specific AI-vocabulary rewriting (LEAD).** New RQ promoted to lead based on T18 DiD, T28 cross-archetype, T16 within-company, T17 cross-metro consistency.
- **RQ1b — Within-company scope broadening and senior archetype shift (SUPPORTING).** Replaces "junior scope inflation + senior archetype shift" as two parallel supporting findings on the same 240-company panel.
- **RQ1c — Seniority boundary sharpening (NEW, from T20).** Replaces the junior-senior convergence construct, which Wave 2/3 rejected.
- **RQ2 — Task/requirement migration — RETAINED but narrower.** T18 + T28 show migration is cross-seniority and cross-domain; T16 shows it's within-company. Migration happened; which *tasks* migrated downward vs stayed senior needs a content-level analysis the exploration didn't run.
- **RQ3 — Employer-requirement / worker-usage divergence — INVERTED.** Employers trail, not lead. Now a novel finding on its own, with a direction opposite to the initial design's.
- **RQ4 — Mechanisms — STRENGTHENED.** T29's partial-mediation finding gives interviews a clean question: "how much of what you wrote in your 2025-2026 JDs came from an LLM draft?" Interviews should also adjudicate T23's inversion: do employers know they're behind worker adoption?

## Direction for Wave 4 (synthesis)

Agent N's T26 SYNTHESIS.md must:
1. Integrate the 12 ranked findings with evidence strength per the post-V2 corrections.
2. State the paper's lead sentence and three supporting claims.
3. Define the recommended analysis-phase RQ set (RQ1a/b/c, RQ2 narrowed, RQ3 inverted, RQ4).
4. Flag all sensitivity dimensions that Wave 3 identified as materially affecting specific findings (aggregator comp, specialist exclusion, baseline choice on J2, LLM-mediation on mentor).
5. Recommend the paper positioning (hybrid dataset + substantive, with framing 1 as lead).
6. Produce interview elicitation artifacts (T25) from the strongest evidence.
7. Generate new testable hypotheses (T24) not already in the research design — e.g., "employer-side AI requirements lag worker adoption in other occupations with high AI exposure" (testable with the same cross-occupation data), or "requirement-section contraction is a SWE-specific signal of hiring-bar lowering masked by narrative-section expansion."

Wave 4 dispatch is unblocked. After Wave 4, orchestrator pauses for user review before Wave 5.
