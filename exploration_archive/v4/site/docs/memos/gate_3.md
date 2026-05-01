# Gate 3 Research Memo

**Date:** 2026-04-10 (V2 verification addendum added 2026-04-10)
**Wave completed:** Wave 3 (T16-T23, T28, T29), Agents J/K/L/M/O, plus V2 verification
**Reports:** T16.md through T23.md, T28.md, T29.md, V2_verification.md

---

## V2 verification addendum (read first)

V2 was dispatched after the body of this memo was written. Its findings refine several specific claims while strengthening the overall narrative. Read this addendum first.

### What V2 verified

- **AI/ML domain growth +11pp.** Already verified by V1 from two proxies; T28 corroborates. Robust.
- **Senior tier reorganization (people-management collapse, mentoring + tech-orchestration growth).** All converging evidence holds.
- **T16's 87% between-company entry decomposition.** Replicates EXACTLY under T16's convention (n=206 arshkon∩scraped panel): within +0.82pp, between +5.68pp. Combined-column -0.27pp also exact.
- **T18's tech-cluster-wide finding.** Cleanly verified.
- **T22's 10× hedge:firm ratio for AI requirements.** Verified at 10.61 (2024) → 11.59 (2026); global 2.03 → 2.69. Structural temporal stability holds.
- **T23's RQ3 inversion.** Direct-only any_ai 14.5%→55.8%; gap vs SO-central (~80%) ≈ 24pp. Workers ahead of employers; employers catching up at 4-10× the relative rate. The inversion is solid.
- **T29's LLM-authorship rejection — directionally.** In low-LLM Q1 subset: length +38% (vs +44% full corpus), AI rate +197% (vs +237% full). Hypothesis stays rejected for length and AI growth.
- **"AI Engineer" within-title content evolution.** Verified at 14× → 16× volume, 0% → 45% `agentic`, pytorch share falling. Paper-figure quality.
- **Mid-senior credential vocabulary stripping.** Already V1-verified; remains solid.

### What V2 changed

1. **The tech-count growth claim is reduced from +34-60% to ~16-27% mean (3→4 median).** This is the most consequential V2 finding. T11 (+34%), T14 (+60% median), and T19 (~flat) are all arithmetically correct under their own conventions, but neither extreme is right. T11/T14 used the broken `swe_tech_matrix.parquet` whose issues extend beyond c++/c#; T19's safe-15-tech list is too narrow and is also suppressed by text-source composition. V2's independent 39-tech detector on raw `description` gives **mean +15.8%, median 3→4 (+33%)**; with description_hash dedup, **+27% mean**. The growth is real but smaller than headlined. **Wave 4 should report ~16-27% mean growth, median 3→4, NOT T11's +34% or T14's +60%.**

2. **T16's 87% between-company finding is FRAGILE under alternative panel definitions.** The 87% replicates exactly under T16's specific convention (n=206 arshkon∩scraped, min ≥3 SWE in both periods). But:
   - Under min ≥5 SWE in both periods: within drops to ~0.5% (extreme between-company)
   - Under pooled-2024 panel (arshkon + asaniczka): within moves to 47-50% (much more balanced)
   The qualitative finding ("most of the entry rise is between-company composition, not within-company employer pivot") holds across panel definitions, but the specific 87% number is convention-dependent. **Wave 4 should report the 87% with explicit "arshkon-only panel convention" caveat** rather than as a clean headline. The right framing is something like: "between-company composition explains a substantial majority (50-87% depending on panel definition) of the entry-share rise; within-company change explains the remainder."

3. **T18's tech-cluster-wide framing is STRENGTHENED, not weakened.** V2 finds SWE-adjacent AI growth at **~105% of SWE (30.4pp vs 29.0pp), not 83%**. SWE-adjacent occupations show essentially the same restructuring as SWE, possibly slightly larger. **The "tech-cluster-wide (and roughly equal-magnitude in adjacent)" framing should be stated more strongly in Wave 4.**

4. **T29's LLM-authorship rejection has one nuance.** The rejection holds cleanly for length growth (+38% Q1 vs +44% full) and AI mention growth (+197% vs +237%), but **credential stacking is partially attenuated in the low-LLM subset** (+29% Q1 vs +149% full corpus). The "every Wave 2 headline strengthens in Q1" framing from T29 is slightly overstated. The reframing for Wave 4: "the content findings broadly persist when LLM-style postings are excluded — confirming the changes are not authoring-tool artifacts — but credential stacking is partially attenuated, suggesting writing style and credential listing are correlated but distinct mechanisms."

5. **Cross-task pattern inconsistency:** T22's strict_mentor pattern is 47-65% broader than T11's. Absolute mention rates differ by that margin; growth rates are consistent (~2.5×). **Wave 4 should standardize on T22's validated patterns from `exploration/artifacts/shared/validated_mgmt_patterns.json` and re-state any T11/T21 absolute-level claims using the standardized definitions.**

### What V2 left unchanged

The narrative skeleton from the body of this memo holds:

- Three-axis content restructuring (scope inflation, senior reorganization, AI/ML domain growth) is real, robust to confounders, and within-company/within-domain
- Senior tier shift toward IC + mentoring + technical orchestration with new vocabulary
- Entry-share rise is composition-driven, not within-company pivot (specific magnitude convention-sensitive)
- Pattern is tech-cluster-wide, not SWE-specific (and per V2, the parity is even tighter than originally reported)
- AI requirements are structurally aspirational AND below worker usage; RQ3 is inverted from anticipatory to lagged catch-up
- LLM authorship is not the mechanism (with the credential-stacking nuance noted above)
- AI/ML "less junior-heavy" was a labeling artifact (already withdrawn)

### Recommendations for Wave 4 synthesis

After V2:

1. **Drop the +34% / +60% tech-count headlines.** Use ~16-27% mean / 3→4 median. This is still substantively a meaningful growth — just not as dramatic as Wave 2 reported.
2. **Report the 87% between-company finding with the panel-convention caveat.** Frame as "between-company composition explains the majority (50-87% depending on panel definition)" rather than as a clean point estimate.
3. **Strengthen the tech-cluster-wide framing.** SWE-adjacent is roughly equal to SWE, not 83% — the restructuring affects technical occupations as a class, not specifically SWE.
4. **Standardize on T22's validated patterns.** Re-state T11 / T21 absolute levels using the canonical definitions; growth rates are roughly consistent so directional claims hold.
5. **Soften the T29 unifying-mechanism test.** The hypothesis is rejected for length and AI growth, partially attenuated for credential stacking. Report the credential nuance honestly.
6. **The senior IC + mentoring + technical orchestration finding remains the cleanest paper-quality narrative finding.** No V2 changes here.
7. **The T16 91% concentration finding (most companies post zero entry roles) is unaffected by V2** — it's a structural fact about which companies post entry-level SWE jobs, not a cross-period comparison. Report it as-is.

The body of this memo remains useful for context but the V2 reconciliation above should override any specific claims that conflict.

---

## What we learned

Wave 3 was the deepest gate of the exploration. It refined or significantly reshaped almost every Wave 2 headline:

### 1. Content/scope inflation is real, universal across domains, and survives every confounder we tested

This is the most important Wave 3 finding because it closes off the alternative explanations we'd been worrying about:

- **Within-domain everywhere (T28).** The aggregate scope inflation is overwhelmingly within-domain. Between-domain (AI/ML composition) component is **≤0.13pp** under the primary operationalization vs an aggregate entry-share rise of +8-10pp. Scope inflation rises in **every archetype**: breadth +17-50%, tech count +4-107%, scope count +41-153%, credential stack ≥7 ratio 1.5× to 64×.
- **Legacy stacks drove the headline jump, not AI/ML.** ServiceNow 64×, Spring/Java 23×, .NET 18× drove the T11 5.4× aggregate credential-stack-7 jump. AI/ML was already at 25.7% credential-stack-7 in 2024 (highest baseline) and only rose to 38.0% — the SMALLEST relative increase (1.5×). The original "AI/ML restructuring drove scope inflation" intuition is wrong.
- **NOT explained by recruiter LLM authorship (T29 cleanly rejected).** In the low-LLM-score (Q1) subset, every Wave 2 headline persists or strengthens — credential stack ≥7 jump is +785.7% in Q1 vs +445.9% in the full corpus. Cross-posting variance is unchanged 2024→2026, contradicting the "LLMs uniformize" prediction. Microsoft is the highest-scoring company in BOTH periods, showing the score measures company writing style not LLM authorship.
- **Half explained by text-source confound for length** (V1 found +26% LLM-only vs +72% aggregate). But the credential stack jump survives this in full (T29 confirmed under text-source-controlled subset). Length growth shrinks under text-source control; credential growth does not.

The accumulating effect is that the scope-inflation finding is robust to: domain composition, recruiter tooling, text-source confound, company concentration, native-label contamination. **Whatever caused it is real labor market signal**, not measurement noise from any of the candidate sources.

### 2. The senior tier reorganized along three axes — IC + mentoring + technical orchestration

This is the strongest paper-quality narrative finding. T11/V1/T21/T28 collectively support it from four independent angles:

- **People-management collapsed:** -58% density (T21); formal People-Manager cluster -70% relative; all explicit people-manager terms (`direct report`, `performance review`, `people manager`, `manage/lead a team of`) declining (V1 verified)
- **Mentoring nearly doubled in any-share** (11%→22%, T21). T28 found this is **the single most robust cross-domain directional finding** — mentoring grew +1.9 to +29.4pp in EVERY archetype, with strict people-manager terms near-zero universally (17/20 archetypes)
- **Technical orchestration grew the most:** +47% density, biggest single shift. Driven by a new vocabulary that didn't exist in 2024: `agentic` 415×, `prompt engineering` 20×, `guardrails` 15×, `multi-agent` 40×
- **Domain heterogeneity (T21):** Mentoring grew at Frontend, Backend, Cloud — but DECLINED in AI/ML, where the senior shift is purely toward technical orchestration. Backend/Enterprise is the only domain where people-management density actually rose
- **Director-specific finding (T20+T21):** Directors in 2026 collapse to the mid-senior profile on people-management, mentoring, AI. Only strategic scope distinguishes directors. The director↔mid-senior boundary is the only one that meaningfully blurred (AUC 0.751→0.686, -0.065). This isn't a market-wide blur; it's a specific story about the director tier merging into "senior IC with strategic scope" rather than remaining a people-management track.
- **The AI-mentioning senior profile flipped (T21):** 2024 AI-mentioning seniors were mentor-heavy (research-mentor-like); 2026 are tech-orchestration-heavy (IC-builder-like). Same group, completely different positioning.

### 3. The entry-share rise is real but composition-driven (NOT employer pivot)

T16's within/between decomposition reframes the entry story for the third time:

- **YOE entry rise: +6.50pp aggregate, 87% between-company (+5.68pp), only 13% within-company (+0.82pp).**
- **Combined-column within-company Δentry is actually slightly negative (-0.27pp).**
- **2026 reweights toward companies that were ALREADY entry-heavy** (defense contractors, large new-grad-program runners), NOT employers pivoting toward new-grad hiring.
- The same companies are not changing their behavior; the market mix is shifting under them.

This is a completely different story from "junior scope inflation" — it's "junior posting concentration shift toward formal new-grad-program employers." Defense contractors are unexpectedly over-represented (SpaceX 65%, Northrop, Raytheon, Peraton, Booz Allen, Leidos).

**Coupled finding:** 90.4% of scraped companies with ≥5 SWE postings have ZERO entry rows under the combined column; 49.3% under the YOE proxy. Entry-level posting is a specialized activity at a small set of new-grad-program employers, not a market-wide function.

**The earlier "AI/ML is structurally less junior-heavy" finding from T09 was wrong (T28 debunked).** Under the YOE proxy, AI/ML 2026 entry share is 16.0% vs 17.2% rest-of-market — essentially identical. The combined-column appearance was a routing-coverage artifact (ML Engineer-style titles bypass the Stage 5 rule router). The entry-share-by-domain story doesn't have a meaningful AI/ML signal.

### 4. The pattern is tech-cluster-wide, not SWE-specific

T18's cross-occupation analysis was the validity backbone, and it weakens the SWE-specific framing significantly:

- SWE AI rate +22.9pp; SWE-adjacent +19.0pp (**83% of SWE magnitude**); control +1.2pp
- DiD SWE−control = +21.7pp (strong); DiD SWE−adjacent = only +3.9pp (weak)
- **Length growth is LARGER in adjacent than SWE** (+629 vs +527 chars under text-source control)
- **Scope language growth is also LARGER in adjacent than SWE** (+26.3pp vs +19.0pp)
- **SWE↔control CONVERGED** (+0.079) — driven by shared length expansion, shared inclusive-language shifts, cross-occupation scope vocabulary diffusion. All white-collar postings drift toward shared longer/inclusive style.
- **SWE↔adjacent slightly DIVERGED** (-0.022) — opposite of the boundary-blurring hypothesis
- "AI Engineer" title evolved cleanly within-title from PyTorch/ML role (0% agentic, 22% pytorch) to LLM-agent role (38% agentic, 66% llm-stack). 14× volume growth.

**Implication for the paper:** The framing must be **"tech-cluster-wide restructuring"** (SWE + SWE-adjacent both showing the same patterns at similar magnitudes), not "SWE-specific." This is broader and arguably more important. The restructuring affects software engineering and adjacent technical occupations in roughly equal measure.

### 5. AI requirements are STRUCTURALLY aspirational AND below worker usage

T22 + T23 produced two findings that together restructure RQ3:

**T22:** Within 80-char windows around AI terms, the hedge:firm marker ratio is **~10:1 in BOTH 2024 and 2026**, vs a global baseline of ~1.5:1. **AI requirements have always been ~10× more hedged than non-AI requirements.** What changed is that AI mention prevalence quintupled (14.3%→51.2%), so the aggregate aspiration ratio rose as a composition effect.

**T23:** The original RQ3 hypothesis was "employer requirements outpace worker AI usage = anticipatory restructuring." T23 finds the opposite:
- Employer AI requirement rate: 14.3% → 51.2% (+36.9pp, 3.6×)
- Direct-only: 11.2% → 52.9% (4.7×)
- StackOverflow developer AI usage estimate: ~80%
- **Employer requirements are STILL BELOW worker usage by ~27pp**
- But employers are growing 4-10× faster than workers in relative terms

**The narrative correction:** Not "employers anticipating AI" but **"employers were slow in 2024 (50pp lag) and are racing to catch up; still below the line but closing fast."** Workers adopted AI tools first via developer-driven adoption; employers are belatedly catching up in JD requirements. This **inverts the original RQ3 hypothesis** — anticipatory restructuring becomes employer catch-up.

The combined T22+T23 picture: AI requirements are hedge-heavy ("nice to have", "exposure to", "preferred") even when present, AND they are below worker actual usage, AND they are growing fast. So "aspirational AI requirements" is real but the directional story is opposite to what we hypothesized.

### 6. Methodology refinements with downstream impact

- **`mcp` pattern was contaminated** (T22) — "MCP" matches "Microsoft Certified Professional." Removed from `ai_tool` patterns. Wave 2 task that used it (probably T08's `desc_contains_mcp` claim) needs re-statement.
- **Naive `agent` pattern is contaminated** (V1) — `agentic` is the high-precision token. Wave 2 tasks that used bare `agent` are inflated.
- **C++ and C# regex bugs** in `swe_tech_matrix.parquet` mean any Wave 2/3 finding using those columns is wrong. C++ actually grew +21%, C# is flat (V1 corrected).
- **Tech-count growth is now contested.** T11 reported +34%, T14 reported +60% median, T19 reported nearly flat (2.49→2.73 with safe 15-tech list). These cannot all be right. Most likely explanation: T11/T14 used the broken tech matrix (which has c++/c# wrong but also potentially other issues), while T19 used a simpler hand-validated list. **V2 should reconcile this.**
- **Validated mgmt/scope patterns saved at `exploration/artifacts/shared/validated_mgmt_patterns.json`** by T22 — Wave 4 synthesis tasks should use these instead of constructing their own.

---

## What surprised us

1. **Mentoring DECLINED in AI/ML** while it grew in every other domain (T21+T28). The senior shift looks different in AI/ML (purely tech-orchestration) than in non-AI domains (tech-orchestration + mentoring). This is unexpectedly heterogeneous.

2. **The director↔mid-senior boundary blurred specifically**, not because of broad seniority blurring but because directors stopped being a separate people-management track. The structure of the seniority hierarchy changed at the top, not throughout.

3. **Defense contractors are massively over-represented in 2026 entry-level postings** (SpaceX 65%, Northrop, Raytheon, Peraton, Booz Allen, Leidos). Combined with the 87% between-company entry rise, this means a substantial part of the visible "entry-share rise" is the scraper picking up more defense-contractor postings (or defense contractors maintaining strong new-grad programs while other employers pulled back).

4. **2024 AI-mentioning seniors were mentor-heavy; 2026 are tech-orchestration-heavy** (T21). Same group, profile flipped. This is the sharpest single Wave 3 finding — early AI experts were positioned as researchers/mentors; current AI experts are positioned as IC builders.

5. **Length growth and scope growth are LARGER in SWE-adjacent than SWE** (T18). The "SWE restructuring" framing cannot survive this. The pattern is broader.

6. **AI requirements are 10× more hedged than non-AI requirements** AND have always been (T22). This is a structural fact about how employers signal AI demand, not a temporal shift. But it inflates the aggregate aspiration ratio as AI mentions grow.

7. **The original RQ3 anticipatory-restructuring hypothesis is INVERTED** (T23). Workers adopted AI first; employers are catching up. -27pp gap in employer-vs-worker direction.

8. **AI/ML "structurally less junior-heavy" was a labeling artifact** (T28). One of T09's striking findings was wrong. The combined-column was systematically under-representing AI/ML entry rows because ML-Engineer-style titles bypass the Stage 5 rule router.

9. **Tech count growth is contested across tasks.** T11 +34%, T14 +60% median, T19 flat. Three Wave 2/3 tasks disagree on a metric they all measured. The V1 c++/c# fix isn't sufficient to explain the discrepancy.

10. **AI/ML is becoming LESS concentrated, not more** (T28). Top-20 concentration in AI/ML fell from 31% to 22%. AI/ML is diffusing into mainstream tech employers' stacks (Microsoft, JPMC, GM, Uber posting AI/ML roles), not consolidating into AI specialists.

11. **Direct employers are MORE ghost-like than aggregators** on content measures (T22). The naive intuition is wrong. Aggregators have entry-share skew but their content is less aspirational.

12. **The hedge-vs-firm "ghost" pattern around AI terms is structural, not temporal** (T22). It's just that more postings mention AI now.

---

## Evidence assessment

| Finding | Strength | Notes |
|---|---|---|
| Scope inflation (credential stack 5.4×) is robust to LLM authorship, domain composition, text source | **Very strong** | Multiple independent confounders ruled out |
| Senior mentoring growth (cross-domain robust) | **Very strong** | T28 confirmed in every archetype with the strict detector |
| Technical-orchestration growth in senior tier | **Very strong** | T21 — biggest single senior-tier shift, new vocabulary clean signal |
| AI/ML domain growth +11pp | **Very strong** | Verified by V1 from two independent proxies; replicated by T28 within-corpus |
| Tech-cluster-wide AI restructuring (not SWE-specific) | **Very strong** | T18 DiD; SWE-adjacent at 83% of SWE magnitude |
| Entry rise is 87% between-company | **Strong** | T16 decomposition under multiple operationalizations |
| AI requirements 10× more hedged than non-AI | **Strong** | T22 with validated patterns; structural across periods |
| Employer AI requirements still below worker usage | **Strong** | T23, with appropriate benchmark uncertainty bands |
| Director↔mid-senior boundary blur | **Moderate** | T20 single-boundary finding; small effect (-0.065 AUC) |
| AI-mentioning senior profile flipped (mentor → tech-orch) | **Strong** | T21, clean within-cohort comparison |
| LLM authorship hypothesis (T29) | **Cleanly rejected** | Headlines persist or strengthen in low-LLM subset; variance unchanged |
| AI/ML "less junior-heavy" (from T09) | **Withdrawn** | T28 showed it was a routing-coverage artifact |
| Tech count growth (T11 +34%, T14 +60%) | **Contested** | T19 found nearly flat; needs V2 reconciliation |
| Senior people-management decline | **Very strong** | T11, V1, T21 all converge; multiple strict-detector validations |

---

## Narrative evaluation

### RQ1 (employer-side restructuring): **Reframed and confirmed in a richer form**

- **Junior scope inflation:** CONFIRMED across multiple operationalizations and within every domain archetype. Robust to all confounders we tested. Specifically: longer postings, more requirement categories simultaneously demanded, more technologies, more org-scope language. NOT a single mechanism — not driven by AI/ML composition (T28), not driven by recruiter tooling (T29). The original framing is supported.
- **Senior archetype shift:** CONFIRMED but reframed. People-management collapsed; mentoring + technical orchestration grew. Within AI/ML, the shift is purely tech-orchestration; in other domains, mentoring grew alongside. Director tier merged into mid-senior on every dimension except strategic scope.
- **Junior share decline:** REJECTED for the third time. Under correct measurement, the entry-share rise is real but the rise is 87% between-company composition, not within-company employer pivot. The original "junior rung narrowing" framing is wrong on the within-company axis but correct on the visible aggregate. Reframe to: "the visible 2024-2026 entry rise is composition-driven (concentration of new-grad-program employers), with no within-company pivot toward junior hiring."

### RQ2 (task and requirement migration): **Strongly supported in a specific form**

T12's Fightin' Words analysis + T22's validated patterns + T28's domain stratification + T21's senior deep-dive collectively show a clean migration story:
- **Senior responsibilities migrated FROM people-management TO technical orchestration and mentoring.** Not "downward" to juniors; sideways within the senior tier.
- **AI vocabulary is a NEW category** that didn't exist in 2024 (`agentic`, `prompt engineering`, `guardrails`, `multi-agent`, `langchain`, `mcp` after fixing) — concentrated in senior tier.
- **Mid-senior credential vocabulary was actively stripped** (`qualifications`, `degree`, `bachelor`, `required`, `requirements`). Senior postings became less credential-focused and more outcome-focused.

This RQ is supported but the migration directions are different from what we originally hypothesized. The original hypothesis was downward migration (senior responsibilities into junior postings); the actual finding is sideways migration within the senior tier.

### RQ3 (employer-requirement / worker-usage divergence): **INVERTED**

The original hypothesis was "employer AI requirements outpace worker AI usage = anticipatory restructuring." T23 found the opposite: employers are STILL BELOW worker usage by ~27pp, but employers are growing 4-10× faster than workers. The "divergence" is real but the direction is opposite. Reframe RQ3 to: "employer AI requirements lag worker AI usage but are catching up rapidly; the gap is closing, not opening."

T22 adds a complementary finding: AI requirements are structurally aspirational (10× hedge ratio), so even the employer growth that does exist is hedged language ("nice to have", "preferred", "exposure to"), not firm requirements ("must have", "minimum"). The combined story: workers adopted AI tools first; employers are racing to catch up via JD signaling, with the AI-side of the catch-up overwhelmingly framed as aspirational rather than firm requirements.

### RQ4 (mechanisms): **Several Wave 3 findings inform interview design**

The interview protocol should now ask about:
- The mentoring growth (is it real "I now mentor more juniors at work" or template language?)
- The director merge (have your director responsibilities changed?)
- The aspirational AI requirements (do you screen on "preferred" AI experience?)
- The composition shift (do you maintain a new-grad program? did you start one? did you stop?)
- The scope inflation (can a 2025 entry hire actually do all the things the JD lists?)
- The lagged employer catch-up (when did your team start including AI tools in the JD?)

---

## Updated emerging narrative

The narrative is consolidated and well-supported. Here is the version after Wave 3:

> Between 2024 and 2026, the **tech-cluster occupational space** (software engineering and adjacent technical roles, both showing the same patterns at 83-100% of each other's magnitude) restructured along three independent axes:
>
> **First, posting content expanded.** The same companies, in every tech domain, started demanding more requirements simultaneously. The share of postings that simultaneously demand all 7 categories of requirements (tech, soft skills, scope, education, YOE, management, AI) jumped 5.4× (3.8% → 20.5%). Postings grew longer (+88% in the substantive sections, half of which is real content expansion under text-source control), more technologies are listed, scope language nearly doubled. **This expansion is universal across tech domains** — legacy stacks (ServiceNow 64×, Spring/Java 23×, .NET 18×) actually drove the credential-stack jump more than the new AI/ML domain. It is **not** driven by AI/ML composition shift, **not** driven by recruiter LLM authorship, **not** explained by text-source confound, **not** driven by company concentration. We tested all four candidate confounders cleanly. Whatever caused the content expansion is real labor market signal.
>
> **Second, the senior tier reorganized along three dimensions.** Explicit people-management language collapsed (-58% density; the formal People-Manager cluster shrank by 70% relative). Mentoring nearly doubled in any-share (11% → 22%) and is the most robust cross-domain directional finding in the exploration — it grew in every archetype with strict people-manager terms near-zero universally. Technical orchestration grew the most (+47% density, the largest single shift), driven by a new vocabulary that didn't exist in 2024: `agentic` 415×, `prompt engineering` 20×, `guardrails` 15×, `multi-agent` 40×. Within AI/ML specifically, the senior shift is purely toward technical orchestration (mentoring DECLINED in AI/ML); in other domains, mentoring grew alongside. The director tier merged into mid-senior on every dimension except strategic scope — directors stopped being a separate people-management track. The 2024 cohort of AI-mentioning seniors were mentor-heavy (research-mentor profile); the 2026 cohort are technical-orchestration-heavy (IC-builder profile) — same group, completely different positioning.
>
> **Third, entry-level posting share rose modestly but the rise is 87% between-company composition, not within-company pivot.** The visible entry rise (+6.5pp under YOE proxy) is overwhelmingly explained by 2026 reweighting toward employers that have always run formal new-grad programs (Google, Walmart, Qualcomm, SpaceX, Microsoft, Amazon, Cisco, Visa) plus a defense-contractor cluster that is unexpectedly over-represented (SpaceX, Northrop, Raytheon, Peraton, Booz Allen, Leidos). Within-company, the same firms are not pivoting toward junior hiring — combined-column within-company Δentry is -0.27pp (slightly negative); YOE within-company is +0.82pp (small). Entry posting is a specialized activity at a small set of companies: 90.4% of scraped companies with ≥5 SWE postings have ZERO entry rows under the combined column.
>
> **What did not happen:**
>
> - **The story is NOT SWE-specific.** SWE-adjacent occupations show the same AI mention growth at 83% of SWE magnitude, larger length growth, and larger scope growth. The pattern is tech-cluster-wide. Frame as restructuring of the technical-occupation space, not specifically of software engineering.
> - **There is NO seniority convergence at the semantic or feature level** (T15 + T20). Junior and senior postings did not become more similar to each other in either text-content space or in structured-feature space. The original "junior scope inflation = juniors becoming like seniors" intuition is wrong. Junior postings became more demanding within the junior tier; they did not become indistinguishable from senior postings.
> - **Employer AI requirements do NOT outpace worker AI usage.** They lag by ~27pp but are growing 4-10× faster than worker usage. The original RQ3 anticipatory-restructuring hypothesis is INVERTED — workers adopted first, employers are catching up.
> - **AI requirements are structurally aspirational** (10× hedge ratio) — but this has always been so, it's not a 2024-2026 shift. The shift is in AI-mention prevalence (5×), which inflates the aggregate aspiration ratio as a composition effect.
> - **Recruiter LLM authorship does NOT explain the content findings** (T29 cleanly rejected). All Wave 2 headlines persist or strengthen in the low-LLM subset.

This narrative is sharper, more honest, and significantly more publishable than the original RQ1-RQ4 framing.

---

## Research question evolution

**RQ1 — Employer-side restructuring.** Reframe to:
> "How did employer-side technical posting requirements restructure across seniority levels and domains from 2024 to 2026? What is the relative contribution of within-company content change vs between-company composition shift?"

The within/between distinction is now central. The content change is overwhelmingly within-company; the entry-share change is overwhelmingly between-company. These are different mechanisms and should be reported separately.

**RQ2 — Task and requirement migration.** Reframe to:
> "Which senior responsibilities migrated within the senior tier (from people-management to technical orchestration and mentoring), and what new vocabulary emerged for AI-orchestration roles that did not exist in 2024?"

The original "downward migration into junior postings" framing is gone. The actual migration is sideways within senior.

**RQ3 — Employer-requirement / worker-usage divergence.** **INVERT** to:
> "How fast did employer AI requirements catch up to worker AI usage between 2024 and 2026, and how much of the requirement growth is hedged-aspirational versus firm-required?"

The data does not support anticipatory restructuring; it supports lagged catch-up. The aspirational nature of AI requirements is a separate finding that complements the catch-up story.

**RQ4 — Mechanisms.** Largely unchanged. The interview protocol should be updated to ask about the specific Wave 3 findings (mentoring growth, director merge, lagged catch-up, between-company composition).

**NEW potential RQ5 — Cross-occupation generalization.** The T18 finding that the pattern is tech-cluster-wide rather than SWE-specific could become a separate research question:
> "How does the AI-driven content restructuring of SWE postings compare to the same restructuring in adjacent technical occupations (data engineering, ML engineering, network engineering, data science)? Is the SWE pattern a leading indicator, a lagging indicator, or contemporaneous?"

This might or might not become a separate RQ in the paper; it could also be an extended sensitivity analysis under RQ1.

---

## Gaps and weaknesses

1. **Tech-count growth is contested across tasks** (T11 +34%, T14 +60%, T19 nearly flat). This is the most important open methodological question. V2 should reconcile.

2. **The c++/c# tech matrix bug is documented but not fixed.** Wave 2/3 tasks that depend on the matrix may have residual errors beyond just c++/c#. The engineer handoff doc covers the dedup issue but not the regex; Wave 4 / future runs should regenerate the matrix or at least cross-check tech counts.

3. **The tech-count discrepancy may indicate the broader tech matrix has issues beyond c++/c#** — specifically, double-counting (one technology firing under multiple regex variants), or systematic bias in which technologies the matrix covers vs the safe-15 list. This is worth a brief V2 investigation.

4. **The "AI requirements are still below worker usage" finding (T23) depends on benchmark quality.** StackOverflow developer survey numbers may have selection bias (developers who answer surveys are more likely to use AI). The benchmark sensitivity in T23 partially addresses this, but Wave 4 should be careful about how strongly to claim the gap.

5. **The 87% between-company finding (T16) depends on the overlap panel composition.** The 220-company overlap panel is small and may not be representative of the full 2024-2026 employer set. V2 should spot-check whether the finding holds under alternative panel definitions.

6. **The director↔mid-senior boundary blur is based on small samples** (directors are <2% of SWE postings). The finding is interesting but should be reported with appropriate uncertainty.

7. **JOLTS hiring-cycle confound (Gate 1)** was not directly addressed in Wave 3. Arshkon was at a hiring trough; if the 2026 scraping period is a different point in the cycle, some content patterns may reflect cycle position rather than secular change. T19's rate-of-change estimation is the closest we have but is limited by the discrete-snapshot data structure.

8. **The "AI requirements are structurally aspirational" finding (T22) is based on hedge:firm marker ratios** which are themselves keyword indicators. Pattern precision was validated to ≥90% (good) but the broader question of "what counts as aspirational language vs neutral language" depends on the pattern definitions.

---

## Direction for Wave 4 (synthesis)

Wave 4 (Agent N, tasks T24-T26) is the integration phase. After Gate 3 + V2:

- **T24 (hypothesis generation):** Should formalize the new RQ structure (see "Research question evolution" above) and propose 5-10 follow-up hypotheses. Specifically interesting: the 2024 → 2026 AI-senior profile flip (mentor → IC builder), the defense-contractor over-representation in entry, the mentoring-grew-but-declined-in-AI/ML domain heterogeneity.
- **T25 (interview elicitation artifacts):** Update interview prompts to probe the Wave 3 findings — mentoring growth, director merge, lagged catch-up, composition vs within-company.
- **T26 (synthesis):** Write `exploration/reports/SYNTHESIS.md`. The structure should follow the consolidated narrative above, with explicit handling of: what's confirmed under multiple methods, what was reframed from the original RQ1-RQ4, what was rejected, what remains uncertain. The synthesis should be the document the analysis-phase agent reads first.

**Before dispatching Wave 4, V2 should resolve:**
1. The tech-count discrepancy (most important)
2. T16's 87% decomposition under alternative panel definitions
3. T18's DiD numbers under alternative SWE-adjacent definitions
4. T22's hedge:firm ratio independent re-derivation
5. T23's StackOverflow benchmark numbers spot-check
6. T29's low-LLM subset Wave 2 headline persistence (the cleanest Wave 3 negative result; deserves an adversarial check)

---

## Current paper positioning

The positioning has solidified into a single coherent option:

**Empirical paper with methods contributions, framed as tech-cluster (not SWE-specific) restructuring.**

Lead with the substantive empirical finding: between 2024 and 2026, the tech-cluster occupational space restructured around AI/ML, with three specific within-company content changes (scope inflation universal across domains, senior shift toward IC + mentoring + technical orchestration, new AI-orchestration vocabulary) and one between-company composition change (entry posting concentrated at formal new-grad-program employers, especially defense contractors). The original "junior scope inflation + senior management shift + employer-usage divergence" framing is partially confirmed and significantly refined.

Methods contributions:
1. **Cross-temporal seniority measurement.** Native platform labels have differential accuracy across snapshots; we present a routing-based combined column validated by label-independent YOE proxy. Apparent +21pp entry decline becomes approximately stable under correct measurement.
2. **Within-vs-between decomposition for posting metrics.** Aggregate trends conflate within-company employer behavior with between-company composition shift. We decompose entry-share, AI-mention, length, and tech-count changes and find they have different mechanisms (87% between-company for entry vs 91% within-company for AI mentions).
3. **Cross-occupation validation.** The SWE-specific framing is too narrow — adjacent technical occupations show 83-100% of the same patterns. The restructuring is tech-cluster-wide, not SWE-specific.
4. **Confounder ruling.** Recruiter LLM authorship does not explain the content findings (cleanly rejected); domain composition does not explain it (within-domain dominates everywhere); text-source confound explains some but not most of the length growth.

Sub-finding methodological warnings:
- LinkedIn platform seniority labels have differential accuracy across snapshots
- Default boilerplate strippers do not catch HR-process language inside requirements sections
- Naive keyword indicators (`hire`, bare `agent`, `mcp`) inflate findings by 3-5×
- Tech matrix regex patterns with special characters near word boundaries fail silently
- The duplicate-template scraper artifact (Affirm/Canonical/Epic posting the same description 14-25 times) inflates concentration metrics in small entry pools
- Cross-period text comparisons are sensitive to text-source composition (LLM-cleaned vs rule-cleaned)

If we stopped here, the paper would already have a strong narrative and four credible empirical findings plus four methods contributions. Wave 4 will primarily formalize the synthesis and produce the interview artifacts; V2 will validate the most consequential Wave 3 numbers before synthesis.
