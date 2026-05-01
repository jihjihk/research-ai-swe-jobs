# Gate 3 — Unified Post-Synthesis-Input Research Memo

Date: 2026-04-20
Author: Orchestrator
Written: After Wave 3 (T16-T23, T28, T29), Wave 3.5 (T31-T38), and V2 verification complete; before Wave 4 dispatch (Agent N).

**Gate 3 is the single document Agent N reads as primary input to SYNTHESIS.md.** It consolidates the full pre-synthesis evidence body: Wave 2 discoveries + Wave 3 market dynamics + Wave 3.5 induced hypothesis tests + V1/V2 verification corrections.

---

## What we learned (synthesized across Waves 2, 3, 3.5)

### 1. The paper's narrative axis is AI-driven SWE content restructuring — within-firm, SWE-specific, universal across seniority

The strongest cross-task synthesis: between 2024 and 2026, SWE postings underwent a real, same-firm, same-title, content rewriting that is specific to SWE (and to a lesser extent SWE-adjacent tech roles), independent of firm composition shifts, recruiter-LLM authorship style, or hiring-market cycle selectivity. The AI signal carries this story; seniority and length effects are secondary.

Three independent panels converge on within-company AI rewriting: T16 company-level +7.7-8.3 pp; T37 returning-cohort +7.91 pp; T31 pair-level (same-company × same-title) +13.4 pp (V2 replicates +10 to +13 depending on pair-sample construction). **The pair-level drift EXCEEDING company-level drift is the cleanest test we have that within-firm rewriting is same-title, not title-recomposition.**

The SWE-DiD against control occupations (T18 +14.02 pp; V2 confirmed robust to alternative control definitions including dropping data/financial analysts, dropping nursing, restricting to manual-work controls, and excluding title_lookup_llm SWE tier) preserves the paper's novelty: this is not a general labor-market phenomenon. Control occupations show +0.17 pp drift.

The cross-occupation divergence (T23 SWE + T32 extension) is universal across 16/16 tested subgroups in both 2024 and 2026. Spearman(AI-exposure, gap) = +0.71. Ranking of occupations by employer AI rate matches ranking by worker AI rate (Spearman +0.92); the gap is in level, not rank.

### 2. Seniority boundaries sharpened, not blurred — falsifying a core RQ1 subclaim

T15 (unsupervised, TF-IDF centroid) and T20 (supervised, logistic regression AUC) independently agree that seniority boundaries SHARPENED between 2024 and 2026. T20 AUC sharpening: +0.093 entry↔associate, +0.150 associate↔mid-senior, −0.022 mid-senior↔director (directors softened slightly, possibly tied to T21 "senior Applied-AI" cluster having ~2× director share). T15 TF-IDF junior-vs-senior cosine 0.950 → 0.871 (V1 verified).

T12's relabeling diagnostics — label-based (J3 vs S4) AND YOE-based (YOE≤2 vs YOE≥5) — agree: 2026 entry postings ≠ 2024 senior postings. **The original RQ1 hypothesis of "junior roles being redefined as senior" is falsified.**

### 3. Scope inflation is universal, senior > junior, within-domain — not junior-specific

T11 (verified by V1): length-residualized `requirement_breadth` rose +2.61 for S4 senior, +1.58 for J3 junior. Within-company (V1 verified): S4 +1.97 > J3 +1.43. Scope inflation is a universal SWE-ladder phenomenon, with senior rising more. T28 decomposed the scope rise as 60-85% WITHIN-domain, not between-domain composition — when SWE hiring shifted toward ML/LLM archetypes, those archetypes themselves carry higher scope. T33 tested and REJECTED the hidden-hiring-bar-lowering hypothesis (H_B): requirements-section contraction does NOT correlate with lower YOE, credential stack, tech count, or education asks. The scope broadening is demand-side content change, not screening relaxation.

### 4. Two emergent senior archetypes — "Senior Applied-AI/LLM Engineer" and "Senior Data-Platform/DevOps Engineer"

T34 profiled T21's candidate senior clusters. **Cluster 0 ("AI-oriented"):** 15.6× growth (144 → 2,251), 94% rows 2026, 28%+ titled "AI Engineer" (likely >45% true share), T09 archetype cross-tab shows `models/systems/llm` over-represented 6.75× and `systems/agent/workflows` 5.65×. Distinguishing bigrams: `claude code`, `rag pipelines`, `github copilot claude`, `langchain llamaindex`, `augmented generation rag`, `guardrails model`. Median YOE 6.0 (one year ABOVE the other senior clusters — AI-oriented roles ask MORE experience, consistent with sharpened-seniority). Industry mix: Software Development 45% + Financial Services 17%. 1,163 distinct firms, HHI 38.6 — genuinely distributed. Concentrates 30% in T16 `ai_forward_scope_inflator` employer cluster.

**Cluster 1 ("Tech-orchestration non-AI"):** 2.6× growth (2,034 → 5,258), +11.4 pp share. T09 cross-tab: `pipelines/sql/etl` 2.62×, `kubernetes/terraform/cicd` 2.11×. Titles: 13% data engineer + 11% DevOps/SRE + 20% senior engineer. Bundles DE + DevOps/SRE + AI-lab data-contract work.

CI/CD is the single largest S4 tech-mention riser at +20.6 pp (T14) — larger than any AI term. Orchestration is the defining senior-tier content shift.

### 5. Management language is flat, not falling — correcting a Wave 2 measurement failure

T11's "management density fell" finding was a measurement failure: V1 found the management pattern had 0.55 precision (strict) and 0.28 precision (broad). T21 applied V1's rebuilt `mgmt_strict_v1_rebuilt` pattern (requires mentor co-occurrence with junior/engineer/team/peer — V1 and T22 validated to ≥0.98 precision) and found senior mgmt density FLAT (0.039 → 0.038, SNR 0.1). V2 confirmed. **The RQ2 "senior shift away from management" claim collapses; the correct claim is "senior shift toward orchestration" (T21 +0.67/1K chars, SNR 5.6).**

This is a methods-contribution finding on its own — longitudinal posting studies need semantic pattern validation or they confabulate shifts in generic-collaboration language as "management declining."

### 6. Length growth is boilerplate-led, requirements-section direction is classifier-sensitive

T13 reported benefits +89%, responsibilities +49%, legal +80% — length growth is NOT requirements-led. T13 claimed J3 junior requirements chars shrank −5%, but V1 showed this direction flips under a simpler-regex classifier. T33 confirmed classifier sensitivity extends to AGGREGATE share shift (T13: −0.019 coef; simple-regex: +0.030 coef). T29 showed T13 requirements-share decline near-disappears on low-LLM-authorship subset. **Gate 3 ruling: "junior requirements shrank" is demoted to a flagged, classifier+authorship-sensitive finding.** What survives: length growth is boilerplate-led (benefits/legal/about-company), not requirements-led; requirements content changed (AI/infra/orchestration tokens) but its share of the full description is direction-uncertain.

### 7. Ecosystem crystallization is real but modest; legacy substitution is stack-modernization not AI-ification

T35 found Louvain modularity rose +0.029 (0.632 → 0.662), stable across 10 seeds (SD 0.0019-0.0005). The LLM-vendor cluster was ALREADY coherent in 2024 pooled — T14's pair-level phi spikes (pinecone×weaviate 0 → 0.71) happened WITHIN a pre-existing cluster. Genuine 2026 novelty: the AI dev-tools sub-cluster (copilot/cursor/chatgpt/claude_tool/codex) split off with ≥4 of 5 members having no 2024 above-threshold edges. Observability merged into devops/platform.

T36 attempted to map legacy-title substitutions to neighboring 2026 roles. **The "legacy → AI" interpretation is REJECTED.** 2024 disappearing titles (Java/Drupal/PHP/.NET architect / specialist roles) map to 2026 neighbors that average only 3.6% AI-strict — below the market rate of 14.4%. The substitution is LEGACY → MODERN STACK (Postgres + CI/CD + microservices + Terraform), not LEGACY → AI. The paper's AI story is concentrated in ML/LLM archetype growth + within-firm same-title rewriting, not in legacy-role replacement.

### 8. Sampling-frame-mediated composition shifts are partial, not dominant — paper claims survive restriction

T37 restricted Wave 2/3 headlines to the returning-companies cohort (2,109 firms, 55% of 2026 postings). **13 of 15 headlines robust at retention ratio ≥ 0.80; 0 sampling-frame-driven.** One partial (J3 breadth_resid at 0.70). J3 and S4 directions INTENSIFY on returning-only (not attenuate as Gate 1 hypothesized). Within-company AI-strict Δ on returning cohort +7.91 pp, matching T16 estimates within 0.45 pp. **The paper's headline claims do NOT need a sampling-frame caveat as a first-order qualification.**

T38 tested and REJECTED the hiring-selectivity hypothesis (H_N). All |r| < 0.11 on selectivity-predicted metrics (breadth, AI-strict, scope, mentor, YOE). Only significant correlation (desc_length, r = +0.20 on arshkon_min3) is POSITIVE — opposite of selectivity. Volume-UP firms write LONGER JDs. Content shifts are NOT concentrated in volume-contracting firms. **The JOLTS 2026 hiring trough is a real backdrop but NOT the mechanism.** The paper can claim demand-side content shift independent of hiring-volume cycle.

T29 rejected the LLM-authorship mediation hypothesis. Content effects persist at 80-130% of full-corpus magnitude on the low-LLM-style-score subset. **Length growth is ~half LLM-mediated** (boilerplate verbosity), but AI-mention, credential stacking, scope inflation, CI/CD rise are real independent of authorship style.

### 9. Geographic pattern: AI rise is UNIFORM, not tech-hub driven; ML/LLM archetype is DIFFUSING

T17 found the AI rise is geographically uniform (SNR ≈ 10, every metro +4.7 to +14.5 pp). Tech-hub premium < 2 pp. AI-rise leaders: Atlanta, Tampa, Miami, Salt Lake City (not SF or Seattle). The ML/LLM archetype is growing faster in non-hubs (Minneapolis +13 pp, Tampa +11.4 pp) than in Seattle (+8.8 pp). J3 entry-share rise, in contrast, is geographically HETEROGENEOUS (Detroit −5 pp to San Diego +15 pp) and metro-level J3 SNR < 2 on 13 of 18 calibrated metros. **AI and junior-share changes are geographically decoupled (r = −0.22).**

### 10. Aggregator-vs-direct-employer behavior inverts the usual ghost-posting prior

T22 found ghost-likeness (kitchen-sink score, aspiration ratio, YOE-scope mismatch) concentrates at DIRECT employers, NOT aggregators. Aggregators post cleaner JDs — this is counter to the typical "aggregators are spammy" narrative and is a useful methodological finding for future posting studies. GitHub Copilot appears in only 0.10% of postings (vs ~33% regular-use industry benchmark); employers do not formalize even the most-adopted AI tool in JD text. RAG at 5.2% (58× rise from 2024's 0.09%) is the leading tool-specific employer rise.

---

## What surprised us (post-Wave-3.5)

1. **Pair-level AI drift EXCEEDS company-level drift** (T31 +13.4 pp > T16 +7.7-8.3 pp). We expected pair-level to be tighter (less within-company title noise), hence smaller. Actually it's larger — meaning within-company same-title rewriting is MORE pronounced than the cross-title company average captures. Companies are rewriting specific roles (e.g., Microsoft "Software Engineer II") even more than their aggregate title mix would suggest.

2. **Two emergent senior archetypes, not one.** Going in, we expected to find at most one — the "AI-enabled orchestration" senior role RQ1 hypothesized. Instead T34 confirmed two: an Applied-AI/LLM Engineer (clearly AI-domain-native) AND a Data-Platform/DevOps Engineer (infrastructure-orchestration-focused). The second is as significant quantitatively (+11.4 pp share shift) as the first.

3. **AI-oriented senior roles ask MORE experience, not less.** Median YOE 6 vs 5 for other senior clusters. 2× director share. This corroborates the sharpened-seniority finding: AI-era senior work is compressing into a more explicit high-experience bar, not democratizing.

4. **Management language is flat, not falling — a measurement fix flipped a Wave 2 claim.** The "senior shift from management" RQ2 subclaim was a 0.28-precision-pattern artifact. Under valid patterns it collapses. The TRUE senior shift is toward orchestration.

5. **Cross-occupation divergence UNIVERSAL across 16/16 subgroups in both periods.** We expected some subgroups to show no gap or inverted gap. Even nurses (AI-usage ~10-15% per benchmarks) have employer ai_strict at 0.00% — the gap is universal.

6. **Hiring-selectivity REJECTED — opposite direction from prediction.** Volume-UP firms write LONGER JDs. JOLTS trough is background, not mechanism.

7. **New-entrant firms are LESS junior-heavy than returning firms** (T16: 12.8% vs 15.3%). Between-company J3 rise is EXIT-driven, not entry-driven. 4,395 firms exited the panel with lower-than-average J3 share.

8. **Legacy substitution is stack-modernization, not AI-ification.** 2024 architect/specialist titles map to 2026 engineer titles with AI mention rates BELOW market. The paper's AI story does not run through title-substitution.

9. **T13 + T33 classifier sensitivity extends to aggregate**, not just juniors. Requirements-section direction flips with classifier choice at the full-corpus level. This is a meaningful methods caveat for longitudinal posting studies.

10. **Tech ecosystem crystallization is modest (+0.029 modularity), not dramatic.** The LLM-vendor ecosystem was already coherent in 2024 pooled. Real novelty: AI dev-tools sub-cluster (Copilot, Cursor, Claude-tool) SPLIT OFF in 2026 — tool use is semantically distinct from systems engineering.

---

## Evidence assessment — ranked findings

Ordered by (evidence strength) × (novelty) × (narrative value). Each finding carries a strength rating, sample, and V1/V2 verification status.

### Tier A — Lead-candidate findings (strong, novel, core to the paper)

**A1. Cross-occupation employer-vs-worker AI divergence is universal and SWE-specific in magnitude.** Evidence: **strong** (T23 + T32; V2 confirmed DiD robust across 4 alt control definitions). 16/16 subgroups positive gap 2024 AND 2026 under any-AI benchmark. Spearman(exposure, gap) +0.71. SWE DiD +14.02 pp [+13.67, +14.37] vs control +0.17. **This is the paper's strongest single quantitative claim.** Figure candidate: T32's cross-occupation divergence chart.

**A2. Within-firm AI content rewriting is real, same-title, and clean of composition.** Evidence: **strong** (three independent panels converge: T16 +7.7-8.3 pp company-level, T37 +7.91 pp returning-cohort, T31 +10-13.4 pp pair-level). Pair-level EXCEEDS company-level → same-title rewriting beyond title-composition. V1/V2 verified. Exemplar: Microsoft "Software Engineer II" rewrite at n=35 with Copilot/GenAI/AI-Systems citations.

**A3. Seniority boundaries sharpened, not blurred.** Evidence: **strong** (T15 TF-IDF 0.950 → 0.871; T20 AUC +0.150 associate↔mid-senior; T12 label-based AND YOE-based relabeling diagnostics agree: 2026 entry ≠ 2024 senior). **Falsifies a core RQ1 subclaim.** Negative results of this cleanness are publishable.

**A4. Scope inflation is universal, senior > junior, within-domain — not junior-specific relabeling.** Evidence: **strong** (T11 S4 breadth +2.61 > J3 +1.58; T28 60-85% within-domain decomposition; T33 rejects hiring-bar-lowering mechanism). V1/V2 verified within-company S4 +1.97 > J3 +1.43.

**A5. Two emergent senior archetypes with content-grounded role names.** Evidence: **strong** (T34 cluster 0 Applied-AI/LLM Engineer 15.6× growth, 94% 2026; cluster 1 Tech-orchestration/DevOps 2.6× growth). Title-regex + T09 archetype cross-tab + content exemplars confirm role coherence. V2 confirmed 14/20 cluster 0 title sample explicitly AI/ML.

**A6. AI-DiD cross-occupation robustness.** Evidence: **strong** (V2 Phase E). SWE DiD survives dropping data/financial analysts, dropping nurse, restricting to manual-work controls, dropping title_lookup_llm SWE tier. All within 0.5 pp of +13-14 pp. SWE-specificity is robust to classification-tier and control-group choices.

### Tier B — Strong supporting findings

**B1. AI-mention acceleration is 2.80× above within-2024 noise — the cleanest rate signal.** Evidence: **strong** (T19 + calibration_table.csv). Only AI (ratio 2.80) and scope (1.54) clear the > 1 threshold. Seniority annualized rates are dominated by within-2024 noise; report raw 2024→2026 pp deltas (T30 panel), not annualized seniority rates.

**B2. Management language is flat, not falling — a measurement correction.** Evidence: **strong** (V1 exposed 0.28 precision on mgmt_broad; T21 applied V1-rebuilt pattern and found senior mgmt density FLAT; V2 verified). Corrects T11 claim. Publishable as a methods-caveat on longitudinal posting-content studies.

**B3. Hiring-bar-lowering hypothesis REJECTED.** Evidence: **strong** (T33 classifier-sensitive magnitude, but hiring-bar-proxy correlations uniformly |ρ| ≤ 0.28, within-company cross-metric ≈ 0, 0/50 narrative-content samples contain explicit requirement-loosening language).

**B4. Sampling-frame-mediated composition NOT dominant — 13/15 headlines robust on returning cohort.** Evidence: **strong** (T37). Paper can proceed without a sampling-frame caveat as first-order qualification. J3 and S4 directions INTENSIFY on returning-only.

**B5. Hiring-selectivity hypothesis REJECTED.** Evidence: **strong** (T38). Content shifts NOT concentrated in volume-contracting firms; volume-UP firms write longer JDs. JOLTS trough is real but NOT the mechanism.

**B6. LLM-authorship NOT dominant mediator.** Evidence: **strong** (T29). Content effects persist at 80-130% on low-authorship-score subset. Length growth ~half LLM-mediated (boilerplate); AI-mention, credential stacking, scope inflation, CI/CD rise are real.

**B7. AI dev-tools sub-cluster SPLIT OFF in 2026 as semantically distinct from LLM engineering.** Evidence: **moderate-strong** (T35 Louvain stability confirmed). Copilot, Cursor, ChatGPT-tool, Claude-tool, Codex with ≥4/5 having no 2024 above-threshold edges. Tool use vs systems engineering are separate dimensions in the 2026 tech taxonomy.

**B8. CI/CD is the single largest S4 riser (+20.6 pp), bigger than any AI term.** Evidence: **moderate** (T14 + T15). Orchestration is the content backbone of senior-tier shift.

### Tier C — Supporting / methodological contributions

**C1. Technology ecosystem modularity rose modestly (+0.029).** T35. Real but not dramatic. LLM-vendor ecosystem was already coherent in 2024 pooled; 2026 added members. The TRUE novelty is AI dev-tools sub-cluster (B7).

**C2. Geographic AI rise is UNIFORM; ML/LLM archetype is DIFFUSING to non-hubs.** T17. AI-rise leaders are Atlanta/Tampa/Miami/SLC, not SF/Seattle. Metro-level J3 and AI changes are decoupled (r = −0.22).

**C3. Aggregators post CLEANER JDs than direct employers.** T22. Counter-prior; useful methodological finding for posting-study scholarship.

**C4. Copilot at 0.10% of postings.** T23. Employers do NOT formalize even the most-adopted AI tool. Explains part of the employer-worker divergence: under-specification in JDs.

**C5. Legacy substitution is LEGACY → MODERN STACK, not LEGACY → AI.** T36 rejects the legacy-to-AI-ification framing. Stack modernization dominates.

**C6. The T30 multi-operationalization seniority panel framework.** Methods contribution: 12 of 13 definitions direction-consistent across the exploration. Junior-side UP under all 7 junior definitions; senior-side DOWN under 5 of 6 senior definitions (S2 director-only flat). Paper should publish the panel as a reusable tool.

**C7. Within-2024 calibration SNR framework.** Methods contribution: per-metric SNR (cross-period effect / within-2024 effect). AI-strict SNR 32.9; scope 42.8; seniority shares near/below noise for annualized rates (use raw pp deltas instead).

**C8. V1 pattern-validation methodology exposed 0.28/0.55 precision on widely-used management patterns.** Methods contribution. Longitudinal posting studies need semantic (not tautological) precision validation.

### Tier D — Flagged / demoted findings

**D1. "Junior requirements section chars shrank" (T13).** Classifier-sensitive (direction flips under simple-regex), authorship-sensitive (near-disappears on low-LLM subset). **Demoted to qualified claim**: "junior descriptions densified on AI/tech tokens; net requirements-section direction is classifier-uncertain."

**D2. "Aggregate credential stack depth rose."** Below within-2024 noise (SNR 0.59). Cite only per-seniority claims (J3 +16.9 pp, S4 +13.3 pp) — do NOT cite aggregate.

**D3. MCP in ai_broad pattern for 2024 baselines.** "Microsoft Certified Professional" contamination. Use ai_strict for MCP-specific growth citations.

**D4. T16/T23 pattern-provenance mismatch.** V2 flagged that T16 and T23 numbers match top-level `ai_strict` (0.86 precision) rather than V1-rebuilt `ai_strict_v1_rebuilt` (0.96 precision). Under V1-rebuilt, magnitudes drop ~10-15% but direction unchanged. Gate 3 note for SYNTHESIS.md: cite ai_strict with its 0.86 precision or re-derive under V1-rebuilt for cleaner numbers. Since direction holds, the paper's qualitative story is unaffected.

**D5. T31 pair-count panel-dependent.** V2 replicates +10-13 pp on different pair constructions (n=12 strict arshkon-only, n=23 primary, n=37 relaxed). Direction (pair > company) robust; exact magnitude range-report.

---

## Seniority panel — final verdict

T30's 12-of-13 direction-consistency held through Wave 3.5. Primary remains J3 / S4 with pooled-2024 baseline; arshkon-only co-primary for senior claims. The asaniczka senior-side asymmetry (+7.1 pp within-2024 on S4) is real and persistent; senior claims MUST cite both pooled (−7.6 pp) and arshkon-only (−1.8 pp) magnitudes. S2 director-only is flat (noise) and should NOT appear as a lead claim.

T20's supervised AUC corroborates T15's unsupervised TF-IDF sharpening: entry↔associate AUC +0.093, associate↔mid-senior AUC +0.150, mid-senior↔director softened −0.022 (driven by director share moving toward the AI-oriented senior cluster T34 identified).

**Panel-consistent SYNTHESIS.md claims:**
- J3 (YOE ≤ 2) entry-share +5.0 pp pooled (+1.2 pp arshkon-only) — direction UP robust across J1/J2/J3/J4.
- S4 (YOE ≥ 5) senior-share −7.6 pp pooled (−1.8 pp arshkon-only) — direction DOWN robust across S1/S3/S4/S5; S2 flat.
- J3 scope metrics (breadth_resid, stack depth, tech count) rose significantly — direction robust.
- S4 scope metrics rose MORE than J3 — direction robust, magnitude robust.

---

## Narrative evaluation — RQ1-RQ4 status

**RQ1 — Employer-side restructuring (junior rung narrowing, scope inflation, senior redefinition).**
- "Junior share reduces" — **contradicted** (J3 rose +5 pp).
- "Junior scope inflates" — **partially supported but not junior-specific** (T11/T28: universal, senior > junior).
- "Junior roles relabeled as senior" — **contradicted** (T15 + T20 + T12 YOE-based relabeling diagnostic).
- "Senior redefinition toward AI orchestration" — **supported, with emergent-role specificity** (T34 cluster 0 Applied-AI, cluster 1 Data-Platform/DevOps).
- "Senior shift AWAY from management" — **contradicted** under V1-validated pattern (T21 flat); corrected narrative is "toward orchestration, not away from management."

**Revised RQ1 claim (for paper):** "SWE postings restructured around technology domain and AI-enabled orchestration content. Seniority boundaries sharpened; scope broadened universally with senior-tier larger shifts; an emergent Applied-AI/LLM Engineer archetype grew 15.6×."

**RQ2 — Task and requirement migration.**
- "Requirements migrate downward across seniority" — **contradicted in part.** T15 junior-senior divergence + T20 sharpening + Gate 2 copilot rate (J3 4.6% ≈ S4 4.1%) show AI is at both levels rather than migrating. The correct framing is "AI requirements rose universally across seniority; orchestration content rose more at senior."
- "Migration into YOE ≤ 2 postings" — **supported with qualification** (T28 within-domain J3 scope rose; but so did S4 scope, by more).

**Revised RQ2 claim:** "AI tools appear at junior and senior postings at comparable rates; orchestration and platform-infrastructure content concentrated at senior postings. There is no downward migration; there is universal scope broadening with senior-tier larger magnitudes."

**RQ3 — Employer-requirement / worker-usage divergence.**
- **STRONGLY SUPPORTED.** T23 + T32: 16/16 subgroups positive gap in 2024 AND 2026, Spearman(exposure, gap) +0.71, SWE DiD +14 pp vs control +0.17 pp. **This is the paper's potential lead claim.**

**RQ3 survives intact and strengthens to "universal across AI-exposed occupations."**

**RQ4 — Mechanisms (interviews).** Out of scope for computational exploration. T29 + T33 + T38 pre-cleared alternative explanations (LLM-authorship mediation, hiring-bar-lowering, hiring-selectivity) by REJECTING them at the quantitative level. RQ4 interviews can focus on: how senior engineers experience the orchestration shift; how hiring-side actors explain the SWE-specific AI concentration; why employers under-specify tool asks (Copilot 0.10%); whether "Applied-AI/LLM Engineer" is experienced as a new role or a rebranding.

---

## Emerging narrative (final pre-synthesis form)

**Proposed paper abstract (2-3 sentences):**

> Between 2024 and 2026, software-engineering job postings underwent a real, same-firm, same-title content rewriting concentrated on AI tooling and platform-infrastructure orchestration. The rewriting is SWE-specific (DiD +14 pp against control occupations; universal across 16/16 tested subgroups), within-firm (same-company × same-title pair-level AI drift +10-13 pp exceeds company-level +8 pp), and independent of recruiter-LLM authorship style, sampling-frame shifts, and hiring-market selectivity. Seniority boundaries sharpened rather than blurred, scope broadened universally with senior-tier shifts larger than junior, and a new Applied-AI/LLM Engineer archetype grew 15.6× — a structural feature of the 2026 SWE hiring market rather than a junior-level relabeling or composition effect.

**Alternative framings still worth weighing at SYNTHESIS:**
1. **Dataset/methods lead:** the T30 panel, the SNR calibration framework, the pattern-validation methodology (V1's 0.28 precision exposure), and the composition-decomposition framework are independently publishable as a methods paper.
2. **Cross-occupation employer-worker divergence lead:** if the paper emphasizes the universality (16/16 subgroups) rather than the within-firm SWE-specific story, it becomes a general labor-market paper on anticipatory AI-hiring signaling.
3. **SWE-restructuring empirical lead (preferred):** the above abstract.

The orchestrator recommends positioning #3 with #1 as methods contribution in the middle third of the paper.

---

## Research question evolution (final)

**RQ1 → revised:** How did SWE posting content restructure 2024→2026 across seniority, technology domain, and within-vs-between-firm dimensions? What is the content signature of within-firm rewriting?

**RQ1a (new):** Do seniority boundaries blur, sharpen, or stay stable? Answer: **sharpened** (T15 + T20).

**RQ1b (new):** Is scope change within-seniority, cross-seniority flow, or universal? Answer: **universal, senior > junior** (T11 + T28).

**RQ1c (new):** Did new senior archetypes emerge? Answer: **yes, two — Applied-AI/LLM Engineer + Data-Platform/DevOps Engineer** (T34).

**RQ2 → revised:** Which requirement categories moved between seniority levels, and which moved universally? Answer: **AI tools universal, not downward; orchestration concentrated at senior; no management exodus (management flat under valid patterns).**

**RQ3 → affirmed and extended:** Does employer AI-requirement rate diverge from worker AI-usage rate? Is this pattern SWE-specific or universal across AI-exposed occupations? Answer: **divergence universal across 16/16 subgroups; SWE DiD +14 pp vs control +0.17 pp.**

**RQ4:** unchanged (interview-based).

**New post-3.5 RQ candidate:**

**RQ5 (induced):** Does within-firm same-title rewriting explain the bulk of 2024→2026 SWE content shifts, after controlling for composition, sampling-frame, authorship, and hiring selectivity? Answer: **yes** (T16 + T31 + T37 + T38 + T29).

---

## Gaps and weaknesses

1. **Pattern-provenance mismatch in T16/T23** (V2 D4). Numbers cite top-level ai_strict (0.86 precision) not V1-rebuilt (0.96). Direction holds; magnitude drops ~10-15% under rebuilt. **Mitigation in SYNTHESIS.md:** footnote the pattern-precision trade-off; cite the direction as primary claim.

2. **T31 pair-count range** (V2 D5). +10-13 pp across constructions. **Mitigation:** range-report in SYNTHESIS.md.

3. **Requirements-section Δ classifier-sensitive** (T13 + T33 + V2 confirmed). Direction flips under alternative classifier. **Mitigation:** cite as a flagged finding; emphasize that length growth is boilerplate-led is direction-robust while requirements-section change is classifier-dependent.

4. **T36 thin disappearing-titles list** (n=2-11). Legacy-substitution story is small-sample. **Mitigation:** cite as a negative / qualified result — "we find no evidence that legacy-role substitution carries AI-ification."

5. **Control-occupation definition sensitivity.** V2 tested 4 alt definitions, all held. Analysis-phase can pre-register a narrower / wider definition.

6. **Asaniczka-only 2024 senior-baseline** is +7.1 pp higher on S4 than arshkon-only. Persistent across LLM-YOE. **Mitigation:** always cite both pooled and arshkon-only magnitudes for senior claims.

7. **Industry taxonomy drift** (LinkedIn 2024→2026). No cross-period industry-stratified claim is valid at raw-label level. Analysis-phase needs a cross-taxonomy mapping.

8. **`posting_age_days` coverage is 0.9%.** Posting-lifecycle analysis is not feasible with current data. Analysis-phase either drops this or requires a different data source.

9. **Benchmark sensitivity for T32/T23.** Worker-AI-usage benchmarks are heterogeneous (survey → "tried ever"; DORA → "daily"; Anthropic → task coverage). T32 reported under 4 bands; direction robust across all. But magnitudes vary substantially. **Mitigation:** paper should present the RANGE, not a point estimate.

10. **ML Engineer cross-source classification gap** (T04: arshkon 78% SWE vs asaniczka 59%). Mitigation is source-stratification which Wave 3+ applied. Persists as an analysis-phase caveat.

---

## Direction for Wave 4 — Agent N's SYNTHESIS.md production

Agent N is about to execute T24 (hypothesis consolidation) + T25 (interview artifacts) + T26 (SYNTHESIS.md). This memo is the primary input. Specific guidance:

### T24 — Hypothesis consolidation

**Directly tested in Wave 3.5:**
- H_A (cross-occupation divergence): **STRONGLY SUPPORTED.** T32 universal across 16/16 subgroups.
- H_B (hidden hiring-bar): **REJECTED.** T33 classifier-sensitive direction, no correlation with YOE/stack/tech/education.
- H_C (emergent senior archetype): **STRONGLY SUPPORTED.** T34 two clusters with content-grounded names.
- H_H (sampling-frame artifact): **REJECTED.** T37 13/15 headlines robust.
- H_K (ecosystem crystallization): **PARTIALLY SUPPORTED.** T35 modularity +0.029 modest; AI dev-tools sub-cluster split off is the clearer novelty.
- H_L (legacy → AI substitution): **REJECTED.** T36 legacy → modern stack, not AI.
- H_M (same-co same-title drift): **SUPPORTED.** T31 pair-level > company-level (direction robust).
- H_N (hiring-selectivity × scope): **REJECTED.** T38 no correlation; positive if any.

**Deferred to analysis phase (from T24 planned list):**
- H_D (senior IC-as-team-multiplier) — analysis-phase, requires within-firm panel regression.
- H_E (same-co J1 drop + J3 rise regime shift) — analysis-phase, requires formal break analysis.
- H_F (Sunbelt AI surge catchup) — T17 found geographic uniformity; H_F is partially absorbed but formal catchup-timing needs event-study.
- H_G (staff-title redistribution) — T10 found "staff" doubled, "senior" dropped 12 pp; analysis-phase can formalize.
- H_I (AI as coordination signal) — needs mechanism-test design.
- H_J (recruiter-LLM senior bias) — T29 rejected the general-mediation hypothesis; H_J specifically targets senior postings.

**New post-3.5 hypothesis candidates:**
- H_O: "Within-firm AI rewriting intensity correlates with firm's 2024 digital-transformation maturity" — requires external firm-maturity data.
- H_P: "Applied-AI Engineer archetype concentrates in financial services due to regulatory-compliance AI adoption" — T34 found 17% financial services over-representation.

### T25 — Interview artifacts

Strongest exemplars (drawn from existing reports):
- Microsoft "Software Engineer II" pair-level rewrite with Copilot/GenAI/AI-Systems citations (T31).
- Wells Fargo, Capital One top-20 AI-drift exemplars (T31).
- 3-5 entry-level ghost postings with ≥3 senior scope terms (T22).
- Applied-AI/LLM Engineer cluster 0 exemplars (T34; 20 samples). These 20 are the single most important interview prompt — show real posting text to senior engineers, ask "is this a real job?"
- T23 cross-occupation divergence chart.
- T32 cross-occupation divergence figure (paper-lead candidate).
- T15 junior-senior centroid movement figure (falsifies relabeling hypothesis).

### T26 — SYNTHESIS.md

Build on this Gate 3 memo. Ranked findings above = Tier A core + Tier B supporting + Tier C methods + Tier D flagged. Robustness appendix = T37 retention-ratio table + T38 selectivity null + T29 low-LLM-subset re-test + V1/V2 magnitude corrections + T30 panel direction-consistency. Method recommendations for analysis phase = pattern-validation protocol, T30 panel, SNR calibration, within-vs-between decomposition, pair-level drift construction.

**Paper positioning recommendation (for Agent N):** hybrid empirical/methods paper with the following chapter structure:
1. Introduction framed around the employer-worker AI-requirement divergence (T23 + T32).
2. Dataset + methods (T30 panel, SNR, pattern validation, within-vs-between).
3. Empirical section 1: Technology-domain restructuring and Applied-AI Engineer emergence.
4. Empirical section 2: Universal scope broadening with sharpened seniority boundaries.
5. Empirical section 3: Within-firm AI rewriting at same-title level (pair-level exemplars).
6. Robustness: sampling-frame, authorship, selectivity, classifier, panel variants.
7. Alternative explanations adjudicated (hiring-bar-lowering REJECTED, legacy-to-AI REJECTED, hiring-selectivity REJECTED).
8. Discussion: what this implies for "AI restructures the SWE labor market" claims; what interviews should investigate next (RQ4).

---

## Current paper positioning — final Gate 3 recommendation

**Strongest paper:** a hybrid empirical + methods paper with the cross-occupation AI-divergence finding (A1) and the within-firm same-title rewriting (A2) as co-lead claims, supported by the sharpened seniority boundaries (A3) and the Applied-AI Engineer archetype emergence (A5). The methods contributions (T30 panel, SNR calibration, pattern validation, within-vs-between) are a clean 25-30% of the paper's contribution and independently publishable as methods-venue content.

**Competing positioning #2:** substantive labor-market paper focused narrowly on cross-occupation divergence (A1 + A6). Simpler narrative; would lose the within-firm and archetype findings to appendix. Not recommended.

**Competing positioning #3:** technology-ecosystem restructuring paper (T09 + T14 + T34 + T35). Technology-domain axis as primary. Methodologically sound but less novelty vs prior labor papers. Not recommended as lead, but absorbed into empirical section 1.

The negative / null findings (H_B rejected, H_L rejected, H_M/H_N rejected, H_H rejected, T29 rejected, management flat) are a major strength of the paper — they rule out the leading alternative explanations for the AI rise. SYNTHESIS.md must foreground these as "we tested and rejected X, Y, Z" rather than hiding them.

---

*Orchestrator note: Agent N's SYNTHESIS.md is the paper's analytical backbone; this memo is its primary input. Gate 3 is done.*
