# Gate 3 Research Memo

Date: 2026-04-15
Wave: 3 (Market Dynamics & Cross-cutting Patterns, T16-T23 + T28-T29)
Reports read: T16, T17, T18, T19, T20, T21, T22, T23, T28, T29
Gate 2 + V1 corrections: applied throughout.

Gate 3 is the gate where the paper's lead finding inverts from the original hypothesis. This memo is deliberately re-opinionated — not a Gate 2 update, a new read.

---

## What we learned

### 1. The RQ3 divergence points the OPPOSITE direction from the original hypothesis

The original RQ3 asked: "do employer-side AI requirements outpace worker-side AI usage, consistent with anticipatory restructuring?" The assumption was that employers would demand AI skills faster than workers had adopted them, producing a ghost-requirement layer.

**T23's answer: no. Workers are far ahead of employers.** Using Stack Overflow 2025 Developer Survey (80.8% professional any-use of AI tools) as the worker benchmark:

- Broad AI posting rate 28.6% vs worker 80.8% → **−52.2 pp gap (ratio 0.35)**
- Narrow AI posting rate 34.6% vs worker 80.8% → **−46.2 pp gap (ratio 0.43)**
- AI-as-tool posting rate 6.87% vs worker 80.8% → **−73.9 pp gap (ratio 0.09)**
- **Copilot: posting 3.77% vs worker 54.9% → ratio 0.07 (worker 14.5× posting)**
- **ChatGPT: posting 0.61% vs worker 66.0% → ratio 0.01 (worker 108× posting)**
- Claude Code: posting 3.37% vs worker 33.0% → ratio 0.10 (worker ~10× posting)

The direction holds under every plausible worker-rate assumption (50-85% sensitivity range). Flipping it requires assuming worker adoption is below 29% (broad), 35% (narrow), or 7% (tool), which every benchmark we found contradicts.

**When AI is mentioned in postings, it is overwhelmingly in `responsibilities` and `role_summary`, not in `requirements`.** Only 21-24% of AI mentions land in the requirements section, vs 34-39% for non-AI tech (a −13 to −16 pp shortfall). **Hard-AI-requirement rate (AI × requirements section) is 6.0% in 2026.** So when employers name AI skills at all, they typically describe them as work context ("you'll build agent workflows", "work with our RAG stack") rather than as gating requirements ("must have 2+ years LangChain experience"). The divergence against workers is even larger when restricted to hard requirements: 6.0% vs 80.8% = −74.8 pp gap, 13.5× worker-to-posting ratio.

**This is not a weakening of the paper's lead finding — it is a stronger and more interesting one.** The new narrative is: *AI tool adoption by workers has outpaced employer naming of those tools in job postings by roughly an order of magnitude. Postings are a LAGGING indicator of actual workplace AI adoption, not a leading one. The employer-side description layer has not caught up to workplace practice.* This contradicts the popular "employers demand impossible AI skills they don't actually need" framing AND the anticipatory restructuring framing. It suggests instead that posting language is slow to update — consistent with T29's template-migration mechanism.

### 2. The AI explosion is real, broad-based, and NOT SWE-specific

Wave 3 validated the Gate 2 lead at four different levels:

- **Within-company (T16):** AI prevalence change in the overlap panel is **92% within-company** (+0.2103 of +0.2291 total). Same companies are naming AI more, not new companies bringing it in. This rules out the composition-only reading.
- **Within-metro (T17):** AI rose in **every one of 18 qualifying metros** (+0.070 to +0.303; mean +0.19, std 0.06). 2× the 2024 baseline even in the smallest-surge metro.
- **Across occupation groups (T18):** The DiD AI-broad effect is +29.6 pp (SWE vs control) and +27.5 pp (SWE-adjacent vs control). **SWE-vs-adjacent DiD is a trivial +2.1 pp.** The AI explosion is information-tech wide, not SWE-specific. **Network engineer AI prevalence quadrupled 4.2% → 16.6% with zero SWE-reclassification** — the cleanest single demonstration of pure AI-vocabulary spillover into a structurally non-SWE role.
- **Above macro noise (T19):** Cross-period-to-within-scraped-window ratios are AI broad **24.7×**, AI narrow 17.7×, copilot 8.1×, desc length 13.7× — far above the ratio-10 threshold for "not confounded by JOLTS cooling." Acceleration ratios (cross-period / within-2024 annualized): AI broad 3.24×, AI narrow 3.09×, copilot 6.67×, claude_tool 90.9×.

Combined with T14's tech network modularity rise (0.56 → 0.66, two new AI co-occurrence communities), the AI explosion is the **single most over-determined finding in the exploration.** Every sensitivity, every restriction, every representation points the same direction at similar magnitudes.

**Framing implication:** the paper is no longer about "SWE restructuring" — it is about "information-tech posting restructuring around AI tools and frameworks." This broadening strengthens the empirical case (more occupations in scope, clearer parallel-trends story) even as it reframes the audience.

### 3. The senior redefinition is a technical-orchestration story — and it is MUCH stronger than Gate 2 realized

Gate 2 said "only `mentor` is clean from the senior language analysis." V1's correction was that the strict management pattern had 38-50% semantic precision. T21 (Agent L) then **rebuilt the patterns from scratch with object-noun-phrase requirements and 50-row stratified precision validation**, and the senior redefinition story came back far stronger than before:

**Validated density by profile × seniority (2024 → 2026, per 1K chars):**

| Profile | mid-senior | director |
|---|---|---|
| People management (narrow, validated) | 0.186 → **0.232 (+25%)** | 0.228 → 0.181 (**−21%**) |
| Technical orchestration | 0.168 → **0.332 (+98%)** | 0.118 → **0.302 (+156%)** |
| Strategic scope | 0.045 → 0.053 (+17%) | 0.035 → 0.076 (+117%) |

The technical-orchestration rise at director level is **+156%**, the largest single cross-period change in the entire exploration after the per-tool AI rises. Mid-senior tech orchestration +98%. Tech-lead sub-archetype cluster **more than doubled** (7.8% → 16.9%); people-manager cluster stayed flat (14.5% → 14.7%).

**The AI × senior interaction is entirely in the orchestration profile**, and this is the cleanest direct link in the whole exploration:
- 2024: AI-mentioning senior postings had identical profile density to non-AI.
- 2026: AI-mentioning senior postings have **+76% orchestration density** over non-AI, with **identical** people-management density.

So AI-mentioning 2026 senior postings are tech-leads/orchestrators, not people managers. Senior postings with AI are the load-bearing evidence for "senior roles redefined around AI-enabled leverage."

**Cross-seniority pattern rules out a field-wide template-shift explanation:**
- Entry people-mgmt: **−13%**
- Mid-senior people-mgmt: **+25%**
- Director people-mgmt: **−21%**

Non-monotonic. If LLM drafting tools were inserting mentoring language uniformly at every level, entry would have risen. The rise is concentrated at mid-senior and is specific, not templated. (But note — T29 found that the mentor sub-pattern specifically IS correlated with LLM-style scoring. The non-monotonic level pattern and the T29 style correlation are in mild tension; the honest reading is that narrow people-management rose at mid-senior but some fraction of the mentor component is style-driven. The orchestration rise is far less style-correlated.)

**The T21 rebuild effectively overturns Gate 2 correction 2.** Gate 2 wrote "only `mentor` is clean; drop the `manage` −47% claim." The T21 rebuild finds that narrow object-noun-phrase management patterns (manage_team, manage_people, lead_team, team_building, direct_reports, performance_review, one_on_one, code_review, system_design, architecture_review) DO validate at ≥80% precision. Gate 2 correction 2 should be re-corrected: **T11's aggregation failed precision; the narrow-validated rebuild works and shows management language RISING at mid-senior, NOT falling overall.**

### 4. Seniority boundaries SHARPENED, not blurred — except at the top of the ladder

T20's L2 logistic classifier (8 structured features, 5-fold stratified CV, SWE LinkedIn `seniority_final != 'unknown'`) produces AUC-by-period for each seniority boundary:

| Boundary | AUC 2024 | AUC 2026 | Δ |
|---|---|---|---|
| entry ↔ associate | 0.626 | 0.719 | **+0.093 (sharpened)** |
| entry ↔ mid-senior | 0.836 | 0.876 | **+0.040 (sharpened)** |
| associate ↔ mid-senior | 0.691 | 0.791 | **+0.100 (sharpened)** |
| mid-senior ↔ director | 0.677 | 0.616 | **−0.061 (BLURRED)** |

**Three of four boundaries sharpened.** The only blurred boundary is at the top. This is the opposite of the convergence-blurring hypothesis in RQ1 — and the T15 "slight divergence below SNR" null is now directional under the feature-based classifier. **Seniority signal is clearer in 2026 than in 2024** for the junior-through-mid-senior portion of the ladder.

The director-level blur is a new finding Gate 2 did not anticipate. `tech_count` coefficient at mid-senior → director **flipped sign** (−0.48 → +0.35): 2024 directors had the fewest tech mentions (classic people bosses), 2026 directors have the most. Combined with the +156% orchestration rise in director postings and the −21% people-management decline, **directors are being recast from "people boss" to "hands-on tech-lead / orchestrator."** This is where the "people management" role signal is thinning out — at the director level, not at mid-senior.

**Domain-stratified boundary (T20 step 6):** all six top archetypes sharpened on entry↔mid-senior. Largest gains are AI-heavy (**LLM/GenAI +0.153, DevOps +0.117**); smallest is JS frontend (+0.024). Consistent with the reading that AI-heavy stacks are reorganizing around more-differentiated seniority signals, not fewer.

### 5. Credential-stack depth CONVERGES within archetypes — entry stacks MORE than mid-senior in 2026

T28's within-archetype comparison (junior vs mid-senior credential stack depth) produces what may be the single most counterintuitive finding of Wave 3:

- **Credential stack gap converges in all 10 large archetypes.**
- **Gap flips sign in 7 of 10** — entry postings in 2026 ask for more credential categories than mid-senior postings in the same archetype.

Magnitudes (per-archetype entry-minus-mid-senior credential stack delta, 2024 → 2026):
- Data engineering −0.88
- Java enterprise −0.74
- AWS cloud SDE −0.68
- JS frontend −0.68
- Defense/cleared −0.62
- LLM/GenAI −0.60
- .NET −0.59
- Agile generalist −0.58
- Embedded −0.35
- DevOps −0.30

**The T15 corpus-level null on convergence was hiding this clean within-domain pattern.** It is the narrow RQ1b scope-inflation claim in its strongest surviving form: *within the same tech domain, entry postings in 2026 ask for more distinct credential categories than mid-senior postings in the same domain.*

**Caveat:** T29's style-matched test (see Finding 7) finds that the requirement-breadth rise is roughly 1/3 real and 2/3 style-driven. Credential stack depth, a narrower metric than breadth, is less style-correlated (r ≈ 0.09 with the authorship score) but some attenuation is still expected. Wave 4 T24 should propose the analysis-phase pre-specified version of this finding.

**Requirement-breadth gap:** converges in 5/10, diverges in 5/10. **LLM/GenAI is the clearest divergence** — its junior/senior gap on requirement_breadth widens +1.4 and tech_count widens +2.0. So within the fastest-growing domain, juniors and seniors are becoming MORE differentiated, not more similar.

### 6. Length growth is entirely a style artifact; the tech stack restructuring is not

T29 promoted from exploratory to primary and delivered the single sharpest methodological finding of the exploration:

**Style-matched unifying test** (for each 2026 posting, find the 2024 posting with the closest LLM-authorship score, then compute the delta on paired postings):

| Metric | Full delta | Style-matched delta | Attenuation |
|---|---|---|---|
| any_ai_broad | +0.195 | +0.190 | **2%** |
| any_ai_narrow | +0.367 | +0.370 | **0%** |
| tech_count | +1.36 | +1.04 | 23% |
| scope_density | +0.078 | +0.065 | 17% |
| credential_stack_depth | −0.51 | −0.76 | strengthens |
| requirement_breadth | +0.92 | +0.35 | **62%** |
| **char_len** | **+377** | **−411** | **209% (FLIPS)** |

**Length growth flips direction under style matching.** Within the same authorship class, 2026 postings are SHORTER than 2024 postings. **The entire T13 length-growth finding is style migration.** Period.

Requirement breadth: roughly 1/3 real, 2/3 style. Scope density: 83% real. AI prevalence: fully real (0-2% attenuation). Credential stack depth: strengthens under matching (an even cleaner real signal).

T29 authorship score median shifted +0.33 std (−0.139 → +0.192). 88.7% of 2026 postings score above the 2024 median; 3.9% fall below 2024 p25. Variance does NOT drop — everyone shifted, nobody became more spread out. Raw-text sensitivity cuts the median shift roughly in half (to ~+0.07), so ~50% of the cleaned-text authorship delta is instrument (Kaggle HTML-stripped vs scraped markdown), but the direction holds. Top feature movers: bullet density 0.40 → 6.33 /1k (16× but partly format), em-dash density 0.094 → 0.187 /1k (+98%), LLM-tell density 0.24 → 0.34 /1k (+41%).

**T29 verdict: supported.** LLM drafting tool adoption is a real mechanism and unifies T13 (length growth) and T15 (within-group homogenization) with the tech-stack restructuring (which is *not* style-driven) — but in a split way. The AI explosion is real labor-market signal AND recruiter-tool adoption is a real phenomenon adding length and some vocabulary. Both are true; they are not rivals.

### 7. The "uniform within-archetype entry rise" holds under the YOE proxy

T09 reported that junior share rose in 9 of 10 large archetypes. Gate 2 read this as evidence for Alt A (instrument-driven) because uniform within-group movement is a signature of measurement change. **T28 then re-ran the same analysis under the YOE ≤ 2 proxy** — which is label-independent — and the uniform rise held: 16/16 large archetypes rise under `seniority_final`, **15/16 rise under YOE ≤ 2 of all**, 13/13 rise under YOE ≤ 2 of known. (Only Azure data platform falls under YOE ≤ 2 of all, −0.58 pp.)

**Because YOE does not depend on any seniority classifier, the pure-instrument-drift version of Alt A weakens.** If the rise were purely Stage-10 LLM labeling drift, the YOE proxy would not show a matching pattern.

**But it does not fully weaken Alt A**, because the junior rise is still dominated by between-company composition within domains (T16 found per-company entry decomposition shows LLM/GenAI within-company +0.148, Data eng +0.144, Java −0.054, Defense −0.027 — NOT uniform at the within-company level), and T19 quantitatively confirms that the cross-period entry-share effect is smaller than within-scraped-window drift (ratio 0.86×). The junior-share finding is still weak and specification-dependent at the "is anything real happening" level.

**Revised Alt A verdict:** the junior-share rise is dominated by (a) instrument/denominator drift, (b) intermediary concentration, (c) between-company composition within domains, and (d) a small residual category-wide shift of roughly the magnitude of T28's credential-stack within-archetype convergence. The paper cannot cite the junior-share change as an aggregate finding; it CAN cite the within-archetype credential-stack convergence as a narrow content finding.

### 8. The AI market is bifurcated: tool-stack adopters (existing employers) + new-entrant wave

T16 k-means on 240 overlap-panel companies produced 4 clusters. Cluster 3 — **46 companies, 19% of the panel** — is the clean "tool-stack adopter":
- ΔAI any **+0.523**
- Δ description length **+1,149 chars**
- Δ scope **+0.328**
- Entry share flat under both operationalizations (+0.001 seniority_final / −0.002 YOE ≤ 2)

These are NOT tech-native companies. The most tech-native member is Adobe. The cluster is dominated by **consulting / system-integrator / enterprise-software employers** (AT&T, Deloitte, Amex, Aditi Consulting, Aveva). This is the strongest direct within-company evidence for the Alt C reading.

Meanwhile, T28 found that the **LLM/GenAI archetype is 68% new entrants** (1,036 new / 138 returning / 478 dropped). Only 138 of 1,174 2026 LLM/GenAI companies were in the 2024 LLM/GenAI cluster. Top 2026 LLM/GenAI employers include Anthropic, Microsoft AI, Intel, Alignerr, Harvey, Cognizant.

**The market is bifurcated:** some existing companies pivoted hard (T16's 46-company tool-stack adopters) AND a much larger wave of new employers appeared (T28's LLM/GenAI new-entrant cluster). The paper should present both mechanisms side by side.

Also notable: **74.6% of overlap-panel companies pivoted their dominant archetype between 2024 and 2026** (73.2% at ≥5 per period, median total-variation distance 0.629). Companies restructured *what kind* of SWE roles they post at high frequency. This is the within-company analog of the market recomposition story, and it is a new finding Gate 2 did not anticipate.

### 9. Metro-level pattern is consistent with Alt C (tool-stack uniformity)

T17 tested whether metros with larger AI surges show larger entry-share shifts. **Pearson r = −0.283, p = 0.255 (n = 18 qualifying metros).** Spearman ρ = −0.441 (p = 0.067). Under aggregator exclusion r = −0.382, p = 0.13. Under 50/company cap r = −0.296. Not significant at p < 0.05 under any specification.

**This is consistent with Alt C's "the AI surge is metro-uniform and uncorrelated with entry-share noise"** — if the junior-share story were real, metros with bigger shocks should show bigger effects. They don't, because the AI surge hit every metro at similar magnitude and the entry-share signal is noise.

**Austin stands out:** JS frontend archetype share collapsed **25.3% → 9.1%** while LLM/GenAI rose **3.2% → 15.9%**. Sharpest single metro × archetype shift in the dataset. Worth interview probing.

---

## What surprised us

1. **The RQ3 direction inverted.** The paper's lead-finding framing must move from "employer anticipation" to "posting lag." Workers are 10-15× ahead of postings on tool naming. The 6.0% hard-AI-requirements rate in 2026 means most postings STILL don't gate on AI skills even while naming them in responsibilities. This is a stronger and more novel finding than the original RQ3; it is also the OPPOSITE of the popular "employers demand impossible AI skills" narrative.

2. **Senior redefinition is stronger than Gate 2 realized, not weaker.** T11's precision failure was in the aggregation, not the concept. The narrow rebuild shows mid-senior people-management rose +25% AND technical orchestration rose +98% (mid-senior) / +156% (director). Tech-lead sub-archetype doubled. This is close to the strength of the AI explosion as a content finding.

3. **Seniority boundaries SHARPENED at three of four levels.** The only blur is at director level, and the mechanism is that directors are being recast as hands-on tech-leads (tech_count coefficient flipped sign). The junior-through-mid-senior portion of the ladder got more distinguishable in 2026, not less.

4. **Entry postings in 2026 ask for MORE credential categories than mid-senior postings in the same archetype** — and this holds in 7/10 large archetypes. The T15 corpus-level null on convergence was hiding this clean within-domain pattern. This is the narrow RQ1b surviving the whole Wave 2/3 pipeline with the strongest effect size.

5. **74.6% of overlap-panel companies pivoted their dominant archetype in two years.** Not "companies changed a little"; companies meaningfully reconfigured what kind of SWE roles they post. The within-company restructuring is in archetype, not in seniority.

6. **The tool-stack adopter cluster is consulting/SI/enterprise-software companies, not tech-native firms.** Deloitte, AT&T, Amex, Aditi, Aveva. Adobe is the most tech-native member. The companies that pivoted hardest toward AI vocabulary are the ones that need to signal AI capability to clients, not the ones actually building AI.

7. **LLM/GenAI is 68% new entrants**, not existing-company pivot. Combined with the tool-stack adopter cluster, the AI market is bifurcated — two distinct mechanisms producing the AI growth.

8. **Network Engineer AI prevalence quadrupled 4.2% → 16.6% with zero SWE-reclassification.** The cleanest single demonstration in the exploration that AI vocabulary has spilled into structurally non-SWE roles.

9. **The entry-share cross-period effect (−0.006) is literally smaller than within-scraped-window drift (−0.007).** T19's macro-robustness ratio for entry share is 0.86× — below 1, below the 10-threshold. The entry signal is not just weak, it is smaller than the noise floor.

10. **Length growth flips sign under style matching** (+377 full → −411 matched). Within the same authorship class, 2026 postings are shorter than 2024 postings. The T13 length-growth headline is entirely a style artifact.

---

## Evidence assessment

| Finding | Sample | Evidence strength | Calibration survives? | Style-filter survives? |
|---|---|---|---|---|
| **AI tool/framework explosion (lead)** | Full SWE + adjacent + control | **Strong** | Yes (SNR 44-326, 24× macro ratio) | **Yes (0-2% attenuation)** |
| AI explosion is info-tech-wide (not SWE-specific) | T18 DiD | **Strong** | Yes (+27.5 vs +29.6 pp, 95% CIs tight) | — |
| AI is 92% within-company | T16 overlap panel | **Strong** | — | — |
| Tech network modularity rose (0.56 → 0.66) | T14 capped corpus | **Strong** | Stable across caps 20/50 | — |
| Senior technical orchestration rise (mid-senior +98%, director +156%) | T21 validated | **Strong** | — | Moderate — some style correlation on mentor sub-pattern |
| AI × senior interaction localized to orchestration | T21 | **Strong** | — | — |
| Mid-senior narrow people-management rose (+25%) | T21 validated | **Moderate-to-strong** | — | Mentor sub-component style-correlated |
| Director recasting (orchestration +156%, people-mgmt −21%, tech_count coefficient flipped) | T20+T21 | **Strong** | — | — |
| Boundaries sharpened at 3/4 levels | T20 classifier | **Strong** | — | — |
| Director boundary blur | T20 | **Moderate** | — | — |
| Within-archetype credential-stack convergence (entry stacks more than mid-senior in 7/10) | T28 | **Strong** | — | Strengthens under matching |
| Requirement breadth rise (1/3 real, 2/3 style) | T11 + T29 | **Weak (real residual)** | Yes (SNR 10.2) | **62% attenuation** |
| **Scope density corrected +64.7% (65% real, 35% pattern-dependent)** | T11 + T22 rebuild | **Moderate** | — | 17% attenuation |
| Length growth in responsibilities/role_summary | T13 | **Contradicted** by T29 style test | — | **209% attenuation (flips sign)** |
| **RQ3 divergence direction inverted (worker > employer by 10-15×)** | T23 | **Strong** | Robust across 50-85% worker rate | — |
| Tool-stack adopter cluster (46 overlap companies, consulting/SI-dominated) | T16 | **Moderate-strong** | — | — |
| 74.6% archetype pivot rate | T16 | **Strong** | — | — |
| LLM/GenAI 68% new entrants | T28 | **Strong** | — | — |
| Austin JS-frontend → LLM/GenAI shift | T17 | **Moderate** | Single-metro example | — |
| Uniform within-archetype entry rise under YOE proxy | T28 | **Moderate** | Label-independent | — |
| **Aggregate junior share rise as headline** | Wave 2/3 aggregate | **Contradicted** | No (SNR 0.17, macro ratio 0.86×) | Fails every robustness check |
| Seniority convergence (blurring hypothesis) | T20 | **Contradicted** (opposite direction) | — | — |
| Original "employer anticipation / ghost AI requirements" framing | T23 | **Contradicted** | — | — |

---

## Narrative evaluation

### Original RQ1-RQ4 framing — post-Wave-3 status

- **RQ1 junior narrowing (junior share decline):** **Contradicted.** Aggregate share rise dissolves under instrument calibration, macro cooling, and within-company decomposition. Below the macro noise floor on T19's ratio.
- **RQ1 junior scope inflation:** **Partially supported in narrow form.** T28's within-archetype credential-stack convergence is the clean remnant. Entry postings in 2026 ask for more distinct credential categories than mid-senior postings in the same archetype, in 7 of 10 large archetypes. Survives the style-match attenuation (strengthens, in fact).
- **RQ1 senior archetype shift toward orchestration:** **Strongly supported and much stronger than Gate 2 realized.** +98% mid-senior orchestration, +156% director orchestration, tech-lead sub-archetype doubled, AI × senior interaction localized to orchestration.
- **RQ2 requirement migration between seniority levels:** **Narrowly supported via T28's within-archetype credential convergence.** Not as cross-level task migration per se, but as "entry postings stack more credentials within the same domain."
- **RQ3 employer/worker AI divergence:** **Direction inverted.** Workers (SO Developer Survey 2025, 80.8%) are 10-15× ahead of employer posting-side tool naming. The employer-side AI explosion is real in magnitude (5× rise, SNR 925/326/etc.) but still lags worker adoption. The paper's lead finding must be reframed as "posting lag" not "employer anticipation."
- **RQ4 interview mechanisms:** Out of scope for exploration, but the new RQ3 framing generates a sharper interview probe set: "why is your posting still written as if AI were a preferred skill when your team uses it daily?"

### Alternative narratives — post-Wave-3 status

- **Alt A (instrument-dominated apparent junior-share change):** **Confirmed for the junior-share headline**, but weakened by T28's finding that the uniform within-archetype rise holds under the label-independent YOE proxy. The cleanest reading is a mix: instrument drift + between-company intermediary concentration + a small real category-wide shift.
- **Alt B (ML/AI eats frontend composition mechanism):** **Decisively contradicted.** Kitagawa decomposition: 85-113% within-domain, 2-10% between-domain. And frontend didn't shrink.
- **Alt C (tool-stack restructuring + template homogenization):** **Confirmed as the strongest available framing.** T14 modularity rise, T16 tool-stack adopter cluster, T18 DiD AI SWE-vs-adjacent = +2.1 pp (broad field-wide pattern), T19 macro ratio 24.7× (robust above macro), T28 LLM/GenAI +15.6 pp, T29 authorship-style shift (mechanism).
- **Alt D (new, from T21/T20/T28):** **"The senior tier specialized toward hands-on orchestration."** Mid-senior orchestration +98%, director orchestration +156%, director people-management −21%, director tech_count coefficient flipped sign, tech-lead cluster doubled, within-archetype credential-stack gap flipped sign in 7/10 archetypes. This is a genuinely new framing unhypothesized by the original research design.

### Ranked by strength of current evidence (best at top)

1. **AI tool/framework explosion at employer side, field-wide, 92% within-company, and BELOW worker adoption (RQ3 inverted).** Over-determined across T14/T16/T17/T18/T19/T23/T29. Lead finding.
2. **Senior redefinition toward hands-on technical orchestration.** Co-headline. Mid-senior +98%, director +156%, AI × senior localized to orchestration, tech-lead cluster doubled.
3. **Tool-stack adopter company cluster + LLM/GenAI new-entrant wave (bifurcated AI market).** T16 + T28. Explains the within-company AI surge mechanism.
4. **Seniority boundaries SHARPENED at 3 of 4 levels, blurred at director only.** T20. Contradicts the convergence hypothesis; the director blur is a new finding.
5. **Within-archetype credential-stack convergence (entry stacks more than mid-senior in same domain, 7/10 archetypes).** T28. Narrow RQ1b surviving.
6. **74.6% of overlap-panel companies pivoted dominant archetype in two years.** T16. Within-company market recomposition finding.
7. **Template/style migration as mechanism (T29 supported).** Length growth is fully style-driven; breadth is 2/3 style; AI explosion is 0% style. Mechanism candidate unifying T13/T15 with the non-style-driven findings.
8. **Junior-share "change" is instrument + concentration + style artifact** (macro ratio 0.86×, T19). Null result, but a rigorous one — RQ5 methodological contribution.
9. **Network Engineer AI spillover (4.2 → 16.6%) as single-cell demonstration of AI vocabulary leaking into structurally non-SWE roles.**
10. **The scope density rise is 65% real (corrected +64.7%)**, pattern-validated.

### Dropped / contradicted claims (cannot appear in the paper)

- "Junior share declined" (RQ1 as written)
- "Seniority levels blurred / converged"
- "ML/AI eats frontend"
- "Employers demand AI faster than workers adopt it" (original RQ3 as written — direction inverted)
- "Length growth reflects scope inflation" (T29 refutes: it's style)
- "Management language rose generally" (T11 aggregation overturned by V1 and T21 rebuild — the clean finding is narrow mid-senior orchestration, not general management)

---

## Emerging narrative (updated two-sentence version)

*"Between 2024 and 2026, information-tech job postings (SWE and adjacent) underwent a structural AI tool and framework explosion that restructured the employer-side technology co-occurrence graph and concurrently redefined senior roles around hands-on technical orchestration — but worker-side AI tool adoption (80.8% per Stack Overflow 2025) outpaced employer-side posting language by an order of magnitude, so the employer-side AI explosion is better read as the posting layer catching up to already-normalized workplace practice than as anticipatory hiring-bar restructuring. The junior-narrowing story, length-growth story, and seniority-convergence story do not survive calibration, macro robustness, or style-matching — the real within-domain junior finding is that entry postings now stack more credential categories than mid-senior postings in the same archetype."*

---

## Research question evolution (proposed for Wave 4 synthesis)

- **RQ3 (elevated to lead, REFRAMED):** How does the growth in employer-side AI tool naming in job postings compare to the rate of worker-side AI tool adoption, and what does the gap pattern reveal about posting-language update lag?
- **RQ1 (reframed as RQ1a + RQ1b):**
  - **RQ1a:** How does within-company and within-archetype technology-stack composition change in SWE postings between 2024 and 2026? (Replaces junior-share narrowing with stack-restructuring.)
  - **RQ1b (narrow surviving scope claim):** Do entry-level postings in 2026 stack more distinct credential categories than mid-senior postings in the same tech domain? Does this narrow convergence hold across the overlap panel and under style-matched analysis?
- **RQ2 (reframed):** How does senior-role language shift across the mid-senior / director boundary between 2024 and 2026, with specific attention to technical orchestration, people management, and AI × seniority interaction? (Replaces task-migration with senior specialization.)
- **RQ4 (unchanged):** Mechanisms — interview-based. New interview probes: posting-vs-practice gap (RQ3), archetype pivot decisions (T16 74.6% rate), LLM drafting adoption (T29 style shift), tool-stack adopter cluster motivation (is it client-signaling in consulting?).
- **RQ5 (new methodological contribution, from Gate 1-3):** Specification dependence is a first-class threat to validity for longitudinal posting research. `seniority_final` ↔ YOE proxy have <10% row overlap, `seniority_native` is temporally unstable, and macro-cooling confounds cross-period claims. The paper should present a robustness framework contribution: within-source calibration, macro-robustness ratios, authorship-style matching.
- **RQ6 (new, from Wave 1-3):** Entry-level SWE posting is a specialist-employer activity. 79% of companies with ≥5 SWE post zero entry in the scraped window; top-10 entry-specialist employers (many staffing/college-jobsite intermediaries) drive ~15-20% of the entry pool. Single-operationalization analyses without concentration handling produce false trend claims.

---

## Gaps and weaknesses

1. **The mentor sub-pattern is style-correlated.** T29 found the `mentor` rise correlates with LLM-authorship score. T21 showed management-narrow at mid-senior rose +25% and mentor is a component. The honest read is that some fraction of the mentor rise is template, not real. Wave 4 T24 should pre-specify an analysis-phase test of narrow management using style-matched controls.

2. **T21's cross-seniority non-monotonic pattern partially conflicts with T29's style-migration mechanism.** If LLM drafting were uniform, we'd expect management language to rise at every level; T21 finds it fell at entry and director but rose at mid-senior. Either (a) template rollout is seniority-aware, (b) the non-monotonic pattern is real role-redefinition, or (c) the measurement is partly noisy. This is a tension to flag in Wave 4.

3. **Worker benchmark is Stack Overflow Developer Survey**, which has well-known self-selection bias. The 50-85% sensitivity range covers the plausible uncertainty, and direction holds across it, but a paper reviewer will press on this. Wave 4 T23's benchmark sensitivity table is the mitigation.

4. **T16's tool-stack adopter cluster (n=46) is small.** k-means on 240 companies with 46 in the AI-heavy cluster is suggestive, not conclusive. The analysis-phase version should test with alternative clustering methods (HDBSCAN, Gaussian mixture) and report cluster stability.

5. **T28's 30.5% 2026 archetype coverage is the binding limit** on all within-archetype claims. The credential-stack convergence finding is therefore computed on a sample that is 2026-side text-bound, and the 2026 rows are non-random (they are the rows the Stage-9 LLM budget was spent on). This should be flagged in the SYNTHESIS and in any paper draft.

6. **T23's hard-AI-requirement rate (6.0%) depends on T13's section classifier** being accurate for AI-mentioning postings. V1 verified the classifier's overall precision but did not audit it specifically for AI-relevant rows. Wave 4 should flag this as a method caveat.

7. **The "director recasting" finding is based on a thin sample.** Directors are a small fraction of SWE postings. The tech_count coefficient sign flip is real, but CIs should be reported in the paper draft.

8. **T17 has 18 qualifying metros at ≥50 SWE per period, 8 at ≥100.** The Austin JS-frontend → LLM/GenAI finding is a single-metro example, not a statistical pattern. Use it as illustration, not as a central claim.

9. **T29's authorship score is a ~50% instrument + ~50% content composite.** Raw-text sensitivity halved the shift. Wave 4 should report both the cleaned-text and raw-text distributions and explicitly frame T29 as a partial-attribution mechanism, not a full attribution.

---

## Direction for V2 verification and Wave 4

**Dispatch Agent V2 now** before Wave 4 synthesis. V2 should:
1. Independently re-derive the T23 divergence numbers. This is the paper's lead. A computation error here would be catastrophic. Re-compute copilot / chatgpt / claude_code posting rates from scratch on the LLM-labeled subset; re-pull the Stack Overflow 2025 worker rates and cross-check with the T23 report.
2. Independently re-derive the T16 decomposition — specifically the 92% within-company AI prevalence finding. This is the second-most-important Wave 3 number.
3. Independently re-derive the T21 validated density numbers for technical orchestration at mid-senior and director. The +98% and +156% numbers drive the senior co-headline and must be replicable.
4. Sample-validate 50 rows each from the validated orchestration pattern and the narrow people-management pattern. Confirm ≥80% precision. (T21 said they passed but did not publish precision-per-pattern tables.)
5. Independently re-derive the T28 credential-stack-gap flip direction under both `seniority_final` and YOE ≤ 2. Check that 7/10 flip is robust to different thresholds and different credential-category definitions.
6. Test the T29 style-match protocol under an alternative matching criterion (e.g., match on raw-text authorship score rather than cleaned-text, or match within archetype). Does the length-growth sign flip survive?
7. Validate T18's DiD confidence intervals — the SWE-vs-adjacent +2.1 pp is the load-bearing number for "not SWE-specific." If that CI crosses zero under alternative standard error assumptions, the framing reverts to "SWE-plus-adjacent distinct from control."

**Wave 4 dispatch (Agent N — synthesis):**
- Read all 10 Wave 3 reports, Gates 2 and 3 memos, V1 and V2 verifications.
- Strongest-supported lead: RQ3 inverted ("posting lag behind worker adoption") + senior orchestration co-headline.
- Secondary contributions: Alt C tool-stack restructuring, T28 within-archetype credential convergence, T16 tool-stack adopter cluster, 74.6% archetype pivot rate, senior title compression as template rewriting.
- Methodological contributions: within-2024 calibration, specification-dependence framework, T29 style-matched unifying test, T06 concentration prediction table.
- Interview artifacts (T25) should be built from: inverted RQ3 divergence chart, T21 senior orchestration density chart, T16 company typology example pair, T28 within-archetype credential convergence chart, T29 authorship-score shift chart.
- SYNTHESIS.md (T26) should structure around the four core findings: RQ3 inverted, senior orchestration, tool-stack restructuring, archetype pivot — NOT around the original RQ1-RQ4.

---

## Current paper positioning

### Lead claim (one sentence, for the abstract)

*"Between 2024 and 2026, information-tech job postings underwent a structural AI tool and framework explosion that restructured the employer-side technology co-occurrence graph and recast senior roles around hands-on technical orchestration — but worker-side tool adoption was already an order of magnitude ahead, so the posting-language change is better read as a lagging catch-up to normalized workplace practice than as anticipatory employer demand."*

### Paper contributions (ranked)

1. **A longitudinal information-tech postings dataset** across SWE, SWE-adjacent, and control occupations, with transparent preprocessing, LLM-budget-aware frame, BLS geographic validation (r=0.97), and a three-snapshot cross-period window calibrated against within-2024 cross-source variability.
2. **A rigorous robustness framework for posting-based labor research** — within-source calibration, specification-dependence diagnostics, concentration prediction, macro-robustness ratios, authorship-style matching. The junior-narrowing null under this framework is itself a methodological contribution.
3. **The RQ3 inversion as empirical headline:** Workers outpace posting-side tool naming by 10-15× even after a 5-fold employer-side explosion. This contradicts both the popular "employers demand impossible AI skills" framing and the anticipatory-restructuring hypothesis. Posting language is a lagging indicator.
4. **The senior technical-orchestration shift as co-headline:** Mid-senior +98%, director +156%, tech-lead sub-archetype doubled, AI × senior interaction localized entirely to orchestration. Not a management-to-orchestration pivot — a specialization of the senior tier toward hands-on tech-lead work, with director-level people-management decline as the clearest single signal.
5. **The tool-stack-adopter + new-entrant bifurcation** of the AI market: 46 existing consulting/SI/enterprise-software employers pivoted (T16), concurrently with a ~1,000-company new-entrant LLM/GenAI wave (T28). Two distinct mechanisms produce the aggregate AI explosion.
6. **The 74.6% archetype-pivot rate** among overlap-panel companies. Within-company restructuring in archetype, not in seniority.
7. **T28's within-archetype credential-stack convergence** (entry stacks more than mid-senior in 7/10 large archetypes) — the narrow sliver of junior-scope-inflation that survives every robustness check, and that T15's corpus-level null was hiding.
8. **T29's LLM-authorship-score style-match as a new diagnostic** for longitudinal posting analyses. The length-growth sign flip under style matching is a clean demonstration of why authorship-drift controls are essential for 2024-era-vs-2026-era posting comparisons.

### Paper framing for the venue decision

Wave 3 has shifted the paper toward a **substantive labor-economics / information-science paper** rather than a pure methods/dataset paper. The empirical findings are strong enough and surprising enough to carry the lead:
- "Posting language lags worker adoption by an order of magnitude"
- "Senior roles specialized toward orchestration, not managed more"
- "The junior-rung-narrowing story is instrument"

The dataset and methods are supporting contributions, not the headline. That said, the robustness framework (calibration + specification-dependence + macro-ratio + style-matching) is a publishable methods contribution in its own right and could become a standalone methods section or a companion piece.

---

## Decisions going into V2 verification and Wave 4

1. **Lead RQ is RQ3 inverted:** employer-side AI naming lags worker adoption by an order of magnitude.
2. **Co-headline is senior technical-orchestration specialization** (T21: +98% mid-senior, +156% director).
3. **Dropped claims:** junior narrowing, seniority convergence, length-as-scope-inflation, employer anticipation of AI requirements.
4. **New core findings to cite:** T28 within-archetype credential convergence, T16 tool-stack adopter cluster, T16 74.6% archetype pivot, T29 authorship style shift, T19 macro-robustness ratios, T18 DiD SWE-vs-adjacent trivial.
5. **Paper shifts from "SWE restructuring" to "information-tech restructuring"** per T18.
6. **Gate 2 correction 2 is itself re-corrected:** the T21 rebuild finds narrow object-noun-phrase management rose at mid-senior. T11's aggregation failed precision, but narrow rebuilds work.
7. **Gate 3 clears** conditional on V2 re-deriving the top 3 Wave 3 numbers (T23 divergence, T16 92% within-company AI, T21 +156% director orchestration).
