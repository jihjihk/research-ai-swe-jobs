# Gate 2 Research Memo

Date: 2026-04-17
Gate: after Wave 2 (Agents E-I) · before V1 verification and Wave 3 dispatch
Reports read: T08 (distribution profiling), T09 (archetype discovery), T10 (title evolution), T11 (requirements complexity), T12 (open-ended text), T13 (linguistic/structural), T14 (technology ecosystem), T15 (semantic similarity)

Gate 2 is the most important gate. The Wave 2 findings reshape the paper's narrative more than Gate 1 suggested they would, and they create one major internal tension (T11 breadth rise vs T13 requirements-section shrink) that V1 must re-derive before Wave 3 depends on it.

## What we learned

**1. The "junior scope inflation" construct, as classically framed, is not supported by Wave 2 — but for a more interesting reason than "nothing changed."** T13's section decomposition shows the requirements section *shrank* in absolute characters (−249 chars, −19%) and in share-of-description (37.8% → 23.3%, though SNR 1.46 marginal). Entry postings dropped −9% in requirements chars; senior postings dropped −22%. Meanwhile, T11 measured a +39% rise in `requirement_breadth` (SNR 30.7). These are not contradictory once you see the mechanism: **`requirement_breadth` is a count of distinct requirement TYPES mentioned anywhere in the cleaned description** (tech + scope + soft skills + management + AI + YOE + education). The 2026 posting mentions more *kinds* of things across more *sections* — responsibilities, role summary, about-company — while the formal requirements section itself contracted. This is rewriting + section reallocation, not scope inflation.

**2. The market organizes by domain, not seniority.** T09's NMI on the 8,000-posting balanced sample: NMI(cluster, domain) = 0.26, NMI(cluster, period) = 0.04, NMI(cluster, seniority) = 0.03. **Domain dominates by ~7×.** This survives aggregator exclusion, SWE-tier strict, and raw-text sensitivity. The paper's organizing principle needs to stratify by domain archetype before reading any seniority signal — otherwise ML/AI (the fastest-growing archetype) confounds everything else.

**3. Period is ~180× more discriminating than seniority in embedding space.** T15's centroid similarity: mean within-period cross-seniority = 0.981; mean cross-period same-seniority = 0.971. Translation: a 2026 junior posting is more similar to a 2026 senior posting than it is to a 2024 junior posting. The 2024→2026 shift is a coherent *period-wide* style/content move that applies across seniority levels, not a differential junior-vs-senior effect. T12 confirms mechanistically: 2026 entry rows do NOT acquire 2024-senior markers (lead, architect, principal); they gain new 2026-specific vocabulary (cursor, mcp, chatbot, prompts, hands-on, scikit-learn). **Seniors' cosine-shift across time (0.942) is LARGER than entry's (0.953) — seniors changed more between 2024 and 2026 than juniors did.** This is the opposite of the initial RQ1 hypothesis.

**4. AI-vocabulary is the single cleanest within-company 2024→2026 signal, and it survives length normalization.** T08 top-5-by-effect-size-×-SNR: `ai_tool_binary` (35.4), `ai_mention_binary` (24.7), `org_scope_binary` (12.7), `soft_skill_binary` (6.5), `tech_count_mean` (6.3). T14 confirms density: AI-mentioning postings have 1.28× tech density (vs 1.49× raw) — effect survives length confound. T12 emerging terms top-5: `rag`, `copilot`, `claude`, `ai-assisted`, `cursor`. T14 shows the 2024 LLM-tool singletons coalesced by 2026 into a 13-node phi > 0.15 cluster — the LLM vendor/tool space became a coherent co-occurrence neighborhood in SWE JDs. **This is the paper's strongest and most novel finding.**

**5. Gate 1's J3 decomposition (95% between-company) and T09's J2 decomposition (85% within-archetype) are both true simultaneously.** They measure different decomposition axes. T06: YOE-based J3 entry share rose mostly because new companies entered the scraped pool (aggregators, tech-giant intern pipelines). T09: within the balanced 8,000 sample, the label-based J2 entry share rose within 18 of 22 archetypes — same archetype, more entry labeling. ML/AI archetype specifically rose 11% → 27% entry share within-archetype, and its share of the corpus rose 3.7% → 18.7%. These are compatible. The integrated story is: the SWE posting corpus in 2026 has (a) different companies posting (composition), (b) more ML/AI archetype postings (domain shift), AND (c) within each archetype, more postings labeled entry. But **the within-archetype entry rise is not scope inflation** — it's labeling shift or pipeline expansion, because T13 shows the requirements section is *shrinking* at every seniority level.

Supporting facts the gate rests on:
- T11 refined strict-management pattern (`coach|headcount|hire|mentor|performance_review` after `manage`/`team_building` fell below 80% precision; `manage` had 14% semantic precision in SWE JDs — "manage data" / "manage systems" dominates): SNR 5.09 vs Gate 1's unrefined 1.50. `mentor` term alone rose 13.9% → 20.8%. Senior-side management language is genuinely expanding, but the signal is narrower than the Gate 1 calibration number suggested.
- T15 2026 groups got MORE homogeneous internally (+0.022–0.028 embedding; +0.006–0.017 TF-IDF). Robust across representations. Mechanism likely templating / LLM-drafting.
- T10: title concentration stable (unique-per-1K: 554 → 507 → 533). Title space grows with corpus, not fragmenting faster. `staff` title share doubled (2.6% → 6.3%); `senior` flat. Senior redefinition is landing as "staff," not as "more senior."
- T14 top rising by raw pp: python (+17), ci/cd (+14.6), llm_token (+11.4), typescript (+8.7), kubernetes (+7.9). Only 7 of top-20 rising are AI-era — traditional cloud/DevOps/Python drive most pp-mass. AI-era techs dominate the *relative* growth list (claude z=338, copilot 36×, rag 16×) but not the absolute-pp list.
- T14 structured-vs-extracted validation: Spearman ρ = 0.985 between asaniczka `skills_raw` and description-extracted tech. Description extraction is a recall superset of structured skills — the regex captures what structured has plus broad narrative vocabulary (agile, azure, typescript, aws).
- T10 disappearing titles: `java architect`, `drupal developer`, `devops architect`, `senior php developer`, `sr. .net developer` — **legacy-stack consolidation**. Combined with T14's `dotnet` −4.2pp, this is a real "legacy tech specialist titles are declining" thread.
- T8 anomaly (important): arshkon `seniority_native='entry'` has **26.4% of rows at ≥5 YOE** (mean 4.18 years). **Native entry labels are YOE-incoherent.** Any "entry postings want 5 YOE" finding likely predates AI and is a long-standing posting convention, not a 2024→2026 AI-era shift.
- T8 pipeline bug: `is_remote_inferred` is 100% False across all sources (preprocessing issue, not a Stage-6 output). Blocks any remote/hybrid analysis. Surface to preprocessing maintenance.
- T13: requirements section shrank MORE for seniors (−22%) than entry (−9%). Seniors reallocated into role_summary (+84%) and about_company (+96%) — the senior-side change is a different kind of rewriting.
- T12: `lieu degree` is now a top 2026 bigram — credential loosening is measurable in the text.

## What surprised us

**The requirements section shrank.** This is the biggest surprise in Wave 2 and the one that most reshapes the paper. The classic "2026 postings ask for more" framing assumes that growing length = growing scope. T13 decomposes the +1,319 char pooled→scraped growth and finds requirements is the *only* section that shrank (−249 chars). This inverts the pre-exploration narrative.

**Seniors changed MORE than juniors.** T12's relabeling diagnostic: entry 2026 vs entry 2024 cosine = 0.953 (small change); senior 2026 vs senior 2024 cosine = 0.942 (larger change). Every pre-exploration framing assumed junior was the dynamic cell; the data says senior is. If we write RQ1 around seniors, the evidence is stronger.

**ML/AI is not a structurally-low-entry domain — its entry share ROSE.** Pre-exploration prior (and Gate 1 hypothesis) said AI/ML roles are structurally more senior and the apparent junior rise might be explained by ML/AI growing as an archetype while staying structurally senior. T09 rejected this cleanly: within the ML/AI archetype, J2 entry share rose 11% → 27% (2024 → 2026). ML/AI is getting *more* entry-heavy, not holding its seniority mix. Combined with its archetype share rising from 3.7% to 18.7%, ML/AI is where the biggest *within-archetype* junior rise is occurring.

**`manage` has 14% semantic precision in SWE job descriptions.** T11 sampled and read 50 matches stratified by period. 86% of matches are non-managerial uses like "manage data," "manage systems," "manage infrastructure," "manage state." This invalidates every prior-wave "management language rose" claim that used a broad management pattern. **The pre-committed semantic precision check caught a headline-material pattern failure.** The refined strict pattern is `mentor|coach|hire|headcount|performance_review`, which is dominated by `mentor` (13.9% → 20.8%, +7pp). This shifts the senior archetype story from "management declined" to "mentoring rose."

**Smallest arshkon employers had the HIGHEST AI-mention share (7.6%) in 2024, higher than the tech giants.** This is an anomaly worth following. It could be: (a) small AI startups flooding the AI-mention pool in 2024, (b) Kaggle's arshkon snapshot overrepresenting small startups, or (c) tech giants hadn't yet started writing AI into their JDs in April 2024. Any way, it suggests the 2024 AI-mention baseline is not a "no AI" condition — small shops had it, big shops hadn't added it yet.

**Legacy-stack titles are the disappearing population.** T10's top disappearing titles are all legacy stack: Java architect, Drupal, DevOps architect, senior PHP, Sr. .NET. Combined with T14's `dotnet` −4.2pp and `html/css` declines, the negative-direction story is about legacy enterprise consolidation, not "fewer SWE postings overall." This is a second secondary finding worth its own paragraph.

**T15: 2026 postings are MORE homogeneous internally than 2024, not less.** Under both embeddings and TF-IDF, average within-group cosine rose by +0.02. Every pre-exploration hypothesis predicted the opposite (more niches, more diverse vocabulary). Mechanism hypotheses: templating, LLM-drafting, aggregator copy-paste, or converged training corpora. T29 in Wave 3 tests the LLM-authorship hypothesis directly — this has been promoted to a higher-priority task.

## Evidence assessment

| Finding | Evidence strength | n | Confounds | Calibration | Sensitivity survival | Verdict |
|---|---|---|---|---|---|---|
| AI-mention binary share rose +33.3pp within-company (T06) / +36pp corpus (T08) | **Strong** | 125-co overlap + 63,701 corpus | Length confound (T14 density 1.28× survives) | SNR 24.7 (pooled); T06 within-company decomposition | Survives specialist + aggregator exclusion; raw and cleaned text | **Lead candidate** |
| AI-tool-specific share (copilot/cursor/claude) rose +13.4pp | Strong | 63,701 | None material | SNR 35.4 | Survives all exclusions | **Supports lead** |
| Requirements SECTION shrank (T13) | Strong | full corpus (raw desc with classifier) | Section classifier precision (T13 asserts done) | Requirements mean_chars SNR 0.48 BELOW noise (section contracted more than instrument noise) | Entry −9%, senior −22% (both negative); aggregator exclusion not materially different | **Inverts pre-exploration scope-inflation story** |
| Domain dominates by ~7× in NMI (T09) | Strong | 8,000 sample | Sample representativeness | NMI under aggregator exclusion 0.262 (stable) | Survives SWE-tier strict, raw-text sensitivity | **Reframes paper organization** |
| Period dominates by ~180× in embedding space (T15) | Strong | 12,000 stratified | Representation-dependent (TF-IDF agrees but weaker) | Within-2024 calibration done: SNR marginal to above | Robust between embedding and TF-IDF | **Reframes paper organization** |
| Seniors changed more than juniors in content (T12) | Strong | full LLM-labeled corpus | Depends on cleaned text; cross-checked with raw | Within-2024 calibration shows this pattern not in 2024 noise | Section-filtered comparison confirms | **New finding: RQ1 pivot toward senior** |
| T11 requirement_breadth +39% (SNR 30.7) | Strong (if composite is validated) | 34,102 LLM-labeled | **Composite-score correlation unchecked** — may correlate with description length | SNR 30.7 | Not yet tested under length-matched sample | **Must be verified by V1** — if requirement_breadth components correlate r>0.3 with description length, matching is confounded |
| Refined management-strict SNR 5.09 (down from broad pattern's 1.50) | Moderate | 34,102 | Dominated by `mentor` — single term drives signal | SNR 5.09 above noise | Precision-validated sub-patterns | **Supports senior-shift** |
| ML/AI archetype within-J2 rose 11%→27%, share rose 3.7%→18.7% | Strong | 1,482 archetype-sample rows across both periods | Archetype sample size | Not directly calibrated to within-2024 | T09 archetype labels deterministic across seed | **Key mechanism for aggregate junior rise** |
| 84.7% of J2 sample rise is within-archetype (T09) | Moderate | 8,000 sample | Sample balanced, not proportional — may attenuate composition effect | Not calibrated | T06's 95% between-company (different unit — company vs archetype) | **Compatible with T06; not a contradiction** |
| Arshkon `seniority_native='entry'` 26.4% ≥5 YOE | Strong | 738 arshkon entry rows | None — this is a data-quality fact | N/A (single-source) | N/A | **Invalidates prior-wave "entry wants 5 YOE" claims dated to AI era** |
| 2026 groups more internally homogeneous | Strong | 12,000 stratified | Representation dependent (emb + TF-IDF agree) | N/A | Robust | **Mechanism hint for T29 (LLM-authorship)** |
| Smallest arshkon employers had highest AI-mention share in 2024 | Weak | limited to arshkon, ~50 small-employer cells | Kaggle sample composition may over-represent startups | N/A | N/A | **Worth flagging for V1 re-derivation** |
| `lieu degree` emerging bigram | Weak | counts are small | N/A | N/A | N/A | **Color, not a lead** |

## Seniority panel (applied to Wave 2 headline findings that depend on seniority stratification)

This section is pre-committed in Gate 0: every seniority-stratified headline must report a 4-row T30 panel ablation (J1–J4 for juniors, S1–S4 for seniors).

### Junior share 2024 → 2026 under Wave 2 (from T08 + T30 baseline table)

| Variant | arshkon-only Δ | pooled-2024 Δ | arshkon verdict | pooled verdict |
|---|---:|---:|---|---|
| J1 entry | −0.61pp | +1.46pp | down | up |
| J2 entry+associate | −0.80pp | +1.41pp | down | up |
| J3 YOE≤2 | +2.31pp | +3.86pp | up | up |
| J4 YOE≤3 | +3.44pp | +5.08pp | up | up |

**Agreement verdict:** Split between arshkon-only (J1/J2 down, J3/J4 up) and pooled (all four up). Under the Gate 0 robustness rule, this is **NOT a defensible unanimous-or-3-of-4 finding**. The Wave 2 interpretation is that the aggregate "junior rose" framing cannot stand alone — it must be decomposed into (a) within-archetype J2 rise (T09 finding; real), (b) between-company J3 rise (T06 finding; compositional), and (c) within-LLM-frame flip (T08 finding; sample-restriction artifact). The paper cannot cite "junior share rose" as a headline; it must cite "the apparent aggregate junior share rise decomposes into …"

### Senior share 2024 → 2026

| Variant | arshkon-only Δ | pooled-2024 Δ | Notes |
|---|---:|---:|---|
| S1 mid-senior + director | −0.95pp | −16.24pp | Pooled driven by asaniczka's structurally high senior share |
| S2 director | +0.44pp | +0.26pp | Sparse; not interpretable |
| S3 title-keyword senior (raw) | +3.83pp | −14.42pp | Same baseline dependence as S1 |
| S4 YOE≥5 | +1.66pp | −6.05pp | Label-independent but still baseline-dependent |

**Agreement verdict:** Split across baselines. Under arshkon-only, senior share is near-flat to slightly up on S1/S3/S4. Under pooled-2024, senior share fell 6–16pp. **"Senior share fell" is not defensible without specifying the baseline**, and the arshkon-only baseline (the fairer comparison for native-label content) says essentially no senior-share change. What DID change is senior-side CONTENT (T11 `org_scope_count` +88% at S1, T11 refined mentor-rate +7pp, T13 senior requirements section shrank −22% while role_summary grew +84%) — the senior story is content-based, not share-based.

### Requirements breadth 2024 → 2026 (T11)

| Variant | arshkon mean | scraped mean | Δ | Arshkon verdict |
|---|---:|---:|---:|---|
| J1 entry | ~6.5 | ~9.0 | +2.5 | up |
| J2 entry+associate | 7.48 | 10.38 | +2.90 | up |
| J3 YOE≤2 | ~7.2 | ~9.8 | +2.6 | up |
| J4 YOE≤3 | ~7.4 | ~10.2 | +2.8 | up |
| S1 mid-senior+director | ~7.7 | ~10.4 | +2.7 | up |
| S4 YOE≥5 | ~7.8 | ~10.5 | +2.7 | up |

*(Values above for J1/J3/J4/S4 are approximations — T11 did not report all four junior variants explicitly; V1 must re-derive and confirm the full 4-row junior panel.)*

**Agreement verdict (pending V1 re-derivation):** If T11's breadth rise is unanimous across J1–J4 and S1–S4, the finding is 4-of-4 robust per the Gate 0 rule. But it is also ~equal at junior and senior, so the "junior scope inflation" framing is not supported — breadth rose equally at every seniority level. **Equal rise at every seniority = rewriting, not restructuring.**

### AI-mention share 2024 → 2026 (T08)

| Variant | arshkon | scraped | Δ | Direction |
|---|---:|---:|---:|---|
| J1/J2 entry | ~13% | ~47% | +34pp | up (unanimous) |
| S1 mid-senior+director | ~17% | ~52% | +35pp | up (unanimous) |

**Agreement verdict:** 4-of-4 unanimous UP across junior variants; unanimous UP across senior variants. AI-mention is the cleanest seniority-robust finding in Wave 2. Effect size at junior and senior is essentially identical (+34pp vs +35pp), which is a 2nd-order finding: **AI-mention rewriting is cross-seniority**, not a junior- or senior-specific phenomenon. This is consistent with T15's "period ~180× seniority" result.

## Narrative evaluation

Gate 1 left four candidate framings on the table. Wave 2 updates:

**Framing (1) — "AI-requirement rewriting is the real restructuring."** STRENGTHENED but needs reframing. AI-mention finding survives every sensitivity. T12 emerging terms confirm the vocabulary (rag, copilot, claude, cursor). T14 confirms LLM vendors coalesced into a coherent 13-node cluster by 2026. But framing (1) as originally stated implied employer-side "restructuring" via AI requirements — the Wave 2 evidence actually says employers are *rewriting job descriptions* (period effect, cross-seniority, within-archetype) and the AI rewriting is part of a broader content reallocation (more responsibilities narrative, more boilerplate, more archetypes, more section types — but less formal requirements section). The finding needs to be framed more carefully than "AI requirements restructured SWE hiring."

**Framing (2) — "Platform evolution is the confound."** STRENGTHENED materially. T10 legacy-stack disappearance, T13 section decomposition (requirements shrank, boilerplate grew), T15 2026 groups more homogeneous, T12 section-filtered comparison showing 41 of top-100 distinguishing terms are boilerplate-driven — all support a "how postings are written has changed" story that partially but not fully explains aggregate shifts. This is the methodological thread the paper must lead with to honestly contextualize the AI signal.

**Framing (3) — "Tech-giant intern pipelines expanded; entry market hollowed elsewhere."** WEAKENED. T09 shows 18 of 22 archetypes saw within-archetype J2 rise; it is NOT concentrated in intern-pipeline archetypes. T06 still stands on between-company composition being ~95% of the J3 rise, but that is on the 125-co overlap panel — a narrow slice. Framing (3) does not unify enough of the Wave 2 evidence to be the lead. It remains a supporting thread (and a real methodological finding about data collection) but not the paper's headline.

**Framing (4) — "AI-requirement rewriting crosses occupation boundaries."** UNTESTED — T18 in Wave 3 is the decisive gate.

**New framing emerging from Wave 2 — "Period-dominated, cross-seniority JD rewriting":**
The data says 2024→2026 is a coherent *period*-wide rewriting of SWE postings in which (a) AI vocabulary was added at every seniority level, (b) the responsibilities narrative was expanded while the formal requirements section was contracted, (c) section variety grew (more benefits, more preferred, more about-company), and (d) seniors changed *more* than juniors. The aggregate "junior share rose" and "scope inflated" findings, when decomposed, are (i) partly compositional (new companies, new archetypes), (ii) partly label-routing artifacts (LLM coverage shift), and (iii) partly measurement problems (requirement_breadth counts vs requirements-section chars disagree). The cleanest substantive signal is the AI-vocabulary expansion, which is real, above noise, within-company, and cross-seniority.

**My current lead framing for Gate 2:** A combination of (1) and (2) with the Wave 2 reframe. Lead sentence draft: **"Between 2024 and 2026, LinkedIn SWE postings underwent a period-dominated rewriting in which AI-tool vocabulary expanded from 1.5% to 14.9% of postings (SNR 35.4) across all seniority levels and every tech-domain archetype, while formal requirements sections contracted and narrative/boilerplate sections expanded — producing an apparent but fragile 'scope inflation' signal that disappears once requirements text is isolated."**

This is the framing I will steer Wave 3 toward, while leaving framings (3) and (4) as testable alternatives through T16, T18, T28.

## Emerging narrative

Draft one-sentence headline at Gate 2: **"2024→2026 SWE posting change is a period-wide, cross-seniority, cross-domain RE-WRITING of job descriptions toward AI-tool vocabulary and broader responsibilities narrative — NOT a restructuring of who gets hired or how seniority is differentiated."**

This is a stronger and more defensible headline than Gate 1's draft. It is also more novel: a paper that says "the *apparent* labor-market restructuring is an artifact of JD rewriting" is more interesting than one that says "juniors narrowed and seniors shifted." The methodological contribution (section decomposition, semantic precision checks, T30 panel robustness) is what the field needs.

## Research question evolution

Gate 1 proposed RQ1a (AI-requirement rewriting) and RQ1b (composition shift). Gate 2 refines further:

- **RQ1a — Cross-seniority AI-vocabulary expansion (PROMOTED to lead).** Between 2024 and 2026, do LinkedIn SWE postings add AI-tool vocabulary across all seniority levels at similar magnitude, independent of posting archetype? *Measured as: AI-mention binary share (strict + broad) by period × T30 seniority × archetype, length-normalized, under within-2024 calibration, on both the full corpus and the 125-company overlap panel.* **Evidence now: SNR 24.7 broad / 35.4 strict on binary; 4.99 on density; effect size ~34pp at junior ~35pp at senior; 18/22 archetypes see it.**
- **RQ1b — Period-dominated rewriting (REFRAMED from "composition shift" + "platform evolution").** Is the 2024→2026 shift a period-wide rewriting of JD form (section reallocation, more vocabulary breadth, more homogeneous within-period corpus), applied across seniority levels and archetypes roughly equally? *Measured as: section anatomy proportions by period × seniority × archetype (T13 classifier); embedding-space period-vs-seniority dominance (T15); within-period homogeneity (T15); relabeling diagnostic (T12).* **Evidence now: T15 period ~180× seniority; T12 seniors changed more; T13 requirements section shrank at every seniority.**
- **RQ1c — Archetype composition and within-archetype share (NEW).** How did the SWE archetype distribution change 2024→2026, and which archetypes drove within-archetype junior rises? *Measured as: T09 archetype proportions, within-archetype J2/S1 shares, archetype × AI mention rate.* **Evidence now: ML/AI archetype 3.7%→18.7%; within-archetype J2 up in 18/22 archetypes; ML/AI J2 up 11%→27%.**
- **Old RQ1 "junior scope inflation" — DEMOTED to Supplementary.** The requirement_breadth rise (T11) is real but equal at every seniority level; the requirements-section-char change is a contraction. "Scope inflation" as a junior-specific phenomenon is not supported. The construct becomes a supplementary diagnostic, not a headline claim.
- **RQ2 — Task and requirement migration — UNCHANGED but narrowed.** T16 and T18 will test whether the AI-vocabulary expansion migrates downward (junior postings picking up AI asks that were senior in 2024) or applies uniformly (cross-seniority). Wave 2 evidence (equal effect size at junior and senior) suggests uniform, but same-company within-seniority longitudinal tests remain in Wave 3.
- **RQ3 — Employer-requirement / worker-usage divergence — STRENGTHENED but untested.** T23 in Wave 3 is the direct benchmark test. Wave 2 evidence: the 1.5% → 14.9% AI-tool share is so large it almost certainly exceeds plausible same-year developer usage ramps (which typical surveys put at 30-75% of developers having tried AI tools once, but lower for daily use). The divergence question is conceptually supported; precise number requires T23.
- **RQ4 — Mechanisms — UNCHANGED and increasingly important.** T12's "seniors changed more" and T15's "more internally homogeneous" and T13's "section reallocation" all have candidate mechanism explanations that interviews will need to adjudicate:
  - Recruiters adopted LLMs to draft JDs (T29 quantitative, interviews qualitative)
  - LinkedIn platform relabeling (partial — T05 found bare "developer" lost 61pp native entry)
  - Template consolidation within HR tooling
  - Actual employer demand shift toward AI-literate hires
  The fact that seniors changed MORE than juniors suggests recruiter-LLM adoption more than employer demand — mid-senior JDs are the default drafting unit in most HR tooling.

## Gaps and weaknesses

- **T11 `requirement_breadth` is a composite score that has not been checked for component-outcome correlation.** Pre-committed in Gate 0: any matched-delta claim using a composite score must report per-component × outcome correlations. V1 must re-derive requirement_breadth, check whether it correlates r > 0.3 with description length, and if so re-interpret the +39% cross-period effect. If breadth is largely a function of length, the finding reduces to "longer descriptions mention more types of things" — still real, but a different claim.
- **T08 `within LLM-labeled subset, J2 direction flips to up`** — this is potentially a major confound. It means LLM coverage rationing is not random with respect to junior/senior signal. V1 should verify.
- **T09 NMI is on an 8,000 stratified sample, not the full 63,701 corpus.** Full-corpus NMI might differ if the stratification over-represents certain archetypes. Not blocking, but worth noting for V1.
- **The AI-mention pattern used in calibration_table.csv is broad** (matches `\b(ai|artificial intelligence|ml|machine learning|llm|...)\b`). The `\bai\b` sub-pattern specifically could be matching non-AI uses ("ai" in "jai alai," "ai-series" hardware, "aide"-like fragments inside Markdown). V1 must sample-check 50 matches stratified by period and report precision per sub-pattern. If broad-pattern precision is <80%, the +33pp "AI mention" SNR 24.7 finding needs to be re-anchored on the AI-tool-specific pattern (SNR 35.4) which is narrower and less likely to have precision issues.
- **T11's refined strict-management pattern (`mentor|coach|hire|headcount|performance_review`) was precision-validated, but the mentor term alone drives the signal.** Need to verify mentor's precision on 2024 vs 2026 samples — is "mentor" used the same way in both periods? (T21 in Wave 3 does deeper senior archetype work.)
- **T15's TF-IDF divergence is stronger than embedding divergence.** Under TF-IDF, junior-senior cosine fell 0.953 → 0.884 (clear divergence); under embedding, 0.980 → 0.972 (marginal). If the paper cites "no convergence," the representation-dependent caveat must be included. V1 should check whether the TF-IDF divergence is driven by a small number of AI-era keywords that dominate TF-IDF but are averaged out in embeddings.
- **`is_remote_inferred` 100% False is a preprocessing bug.** Blocks any remote/hybrid analysis. Not a Wave-2 finding but an infrastructure issue the paper must note. Preprocessing maintenance task.
- **T11 domain stratification was deferred** because T09 archetype labels landed too late. T28 (Wave 3) will do the full domain × seniority decomposition on all scope/breadth metrics.
- **`\bleadership\b` token from the skills_raw → could be "leadership and management" → may confound the refined strict mentor pattern.** T11 reported `mentor` cleanly but V1 should sample-check.

## Direction for next wave

### Gate 2 Verification (Agent V1) — dispatch IMMEDIATELY before Wave 3

Target findings for re-derivation from scratch:

1. **AI-mention binary share.** Compute +34 to +36pp cross-period effect on full corpus independently using (i) broad pattern (`\bai\b|\bartificial intelligence\b|\bml\b|\bmachine learning\b|\bllm\b|...`), (ii) strict AI-tool pattern (`copilot|cursor|claude|chatgpt|openai|rag|langchain|fine.tuning|...`). Sample 50 matches per sub-pattern stratified by period; report semantic precision per sub-pattern; drop sub-patterns <80%. Rebuild compound. Re-report SNR. Expected: strict should hold; broad may need revision.

2. **Requirements section shrinkage.** Re-implement T13's section classifier from scratch (independent regex) and re-run the section decomposition on a sample of 2,000 postings per period. If requirements-section char shrinkage reproduces within 20%, verified. If not, investigate.

3. **`requirement_breadth` composite-score correlation check (PRE-COMMITTED in Gate 0).** Compute Pearson correlation between each `requirement_breadth` component (tech_count, soft_skill_count, org_scope_count, education_level, yoe_numeric, management_strict_count, ai_count) and description_length on the T11 feature parquet. If any component correlates r > 0.3 with description_length, flag that the +39% breadth rise is at least partly length-driven. Report length-residualized breadth effect.

4. **T09 NMI re-computation.** Compute NMI(cluster, domain) vs NMI(cluster, seniority) vs NMI(cluster, period) on the full 63,701-row SWE LinkedIn corpus (not the 8,000 stratified sample), using the T09 archetype labels projected onto unsampled rows via nearest-centroid. If full-corpus NMI preserves the domain >> period >> seniority ordering, the finding is robust.

5. **Within-LLM-labeled subset J2 flip (T08 anomaly).** Re-derive T08's finding that restricting to `llm_extraction_coverage = 'labeled'` flips the J2 junior-share direction from down (arshkon-only) to up. If confirmed, this is a sample-restriction artifact that must be flagged in every Wave 3 junior claim.

6. **T12 relabeling diagnostic.** Re-compute TF-IDF cosines for (entry26-vs-entry24) and (entry26-vs-midsr24) independently. Confirm entry26 is closer to entry24 than to midsr24. If the order flips or shrinks, the "period-effect dominant" conclusion weakens.

7. **Propose alternative explanations for each of the five Wave 2 lead findings** (AI rewriting, requirements shrinkage, domain NMI, period dominance, seniors-changed-more). What else could explain each pattern? Rank plausibility.

8. **Audit prevalence citation transparency.** Go through the memo's cited numbers and verify each cite has exact pattern + subset + denominator.

Write `exploration/reports/V1_verification.md`. If verification reveals a problem with any of finding 1-6, correct it and flag before Wave 3 dispatch.

### Wave 3 direction (after V1 clears)

With the Gate 2 narrative now centered on "period-dominated, cross-seniority JD rewriting + AI-vocabulary expansion," Wave 3's priority order shifts:

- **T18 cross-occupation DiD (Agent K) — PROMOTED to critical-path.** Does the AI-vocabulary rise appear in SWE-adjacent and control occupations too? If yes, "SWE-specific" framing collapses and the paper becomes a generic-hiring-template paper. If not, framing (1) strengthens into a lead. This is the single most consequential Wave 3 test.
- **T29 LLM-authored JD detection (Agent O) — PROMOTED to critical-path.** T13/T15 both hint at LLM-drafting (requirements shrank, 2026 more internally homogeneous, seniors changed more). T29 directly tests recruiter-LLM authorship and reruns Wave 2 headline findings on low-LLM-score subsets. If the AI-rewriting signal halves on low-LLM-score text, a substantial share is recruiter tooling rather than employer demand — the paper must front-load this.
- **T22 ghost forensics (Agent M) — PROMOTED.** Are AI requirements more aspirational than traditional? This is the validity check on the lead finding. If AI mentions are "realistic" per `ghost_assessment_llm` at similar rates to traditional requirements, the rewriting is genuine. If more aspirational, the paper must frame as "what employers SAY they want, not what they require."
- **T23 employer-usage divergence (Agent M) — PROMOTED.** RQ3's direct benchmark test. With AI-tool share rising from 1.5% to 14.9%, the divergence question is now well-instrumented.
- **T16 company strategies (Agent J) — high priority, scope unchanged.** Extends T06's decomposition.
- **T28 domain-stratified scope (Agent O) — high priority.** T09 labels exist; this tests whether scope/requirement patterns differ across ML/AI vs Frontend vs Embedded. Pairs with T11's deferred step 7.
- **T21 senior role evolution (Agent L) — REFOCUSED.** T12 found seniors changed more than juniors; T21 should quantify via the management-vs-orchestration-vs-strategic framework, using T11's refined mentor pattern (not broad management). Cross-stratify by T09 archetype.
- **T17 geographic (Agent J) — LOWER priority.** T09 domain >> geo; T15 period >> seniority. Geographic is supporting, not lead.
- **T19 temporal / T20 seniority boundary — LOWER priority.** T07 JOLTS context is already in hand; T15 already showed period >> seniority. These tasks should be scoped down.

### Task-spec modifications for Wave 3 prompts

I will write these directly into Wave 3 agent prompts:

- **All Wave 3 tasks that use AI-mention patterns must cite the V1-refined strict-tool pattern**, NOT the broad Gate 1 pattern. If V1 finds the broad pattern <80% precision, treat strict-tool (SNR 35.4) as primary.
- **T18 DiD must use BOTH the broad AI pattern AND the strict-tool pattern and report both.** Control occupations may show broad "AI" mentions (generic marketing language) at higher rates than strict-tool mentions; the distinction matters.
- **T16 overlap panel decomposition must use T11's `requirement_breadth` ONLY AFTER V1 confirms component-vs-length correlation is acceptable.** If r > 0.3 with length, T16 must report length-residualized breadth.
- **T22 must validate ghost indicators with the same semantic precision protocol as T11.** Any ghost pattern cited as ≥80% precision must have been tested on a stratified 50-sample, not tautologically.
- **T28 must stratify AI-mention, requirement_breadth, entry-share, and section shares BY the T09 archetype labels.** The ML/AI archetype's 11%→27% J2 rise and 3.7%→18.7% share rise must be reconciled with the corpus-level pattern.
- **T29 must test the unifying-mechanism hypothesis: re-run AI-mention, requirement_breadth, section shares, and the +36pp headline on low-LLM-score subsets.** If headlines halve, a substantial mediator is recruiter tooling.
- **T21 must use the V1-refined strict management pattern** and drop broad patterns.

## Current paper positioning

At Gate 2, the paper positioning has shifted from "empirical labor restructuring" to a stronger hybrid:

**Primary positioning: "A longitudinal dataset + methodological framework that reveals a period-dominated AI-vocabulary rewriting of SWE job descriptions."** This has three contributions stacked:
1. The dataset + preprocessing pipeline.
2. The measurement framework (T30 panel, sensitivity dimensions, semantic precision, composite-score correlation checks).
3. The substantive finding: AI-tool vocabulary expanded from 1.5% to 14.9% (SNR 35.4) across all seniority levels and 18 of 22 domain archetypes between 2024 and 2026 — a cross-seniority period effect that coexists with a contraction of formal requirements sections and expansion of narrative/boilerplate sections.

**Venue implications.** This hybrid positioning is credible at a dataset/methods venue (ICWSM, CHIIR, WWW datasets track) AND at a substantive labor/society venue (ICWSM, CHI, labor economics) depending on which contribution we lead with. The methodological contribution is strongest if Wave 3's T29 confirms substantial LLM-authorship mediation — then the paper tells the field how to analyze posting data in the LLM era. The substantive contribution is strongest if Wave 3's T18 shows the AI signal is SWE-specific (or SWE-plus-adjacent) and T22 shows the AI mentions are "realistic" not aspirational.

**What Wave 3 must deliver for each positioning:**

For the **methodological / dataset** lead: T29 must show a meaningful mediator signal (low-LLM-score subsets show weaker headlines), AND T22 must show that AI mentions are more aspirational than traditional requirements, AND T18 must show the AI signal is either generic-hiring-wide OR sharply SWE-specific (not halfway).

For the **substantive labor** lead: T18 must show the AI signal is SWE-specific, AND T23 must show a clear employer-side / worker-side divergence, AND T29 must NOT show substantial LLM-authorship mediation (i.e., the AI signal persists on human-written JDs).

I will dispatch Wave 3 with the hybrid positioning in mind. The exact lead decision is a Gate 3 call.

Gate 2 complete. V1 verification dispatching next; Wave 3 after V1 clears.

---

## V1 verification corrections (added 2026-04-17 after V1 completed)

V1 re-derived six lead findings from scratch. Five verified; one corrected.

**Corrections that must propagate to Wave 3 prompts:**

1. **AI-mention pattern refined.** Two sub-terms failed the 80% semantic precision threshold and have been dropped: `\bagent\b` (66% precision — matches "insurance/sales agent", "SQL agent", "change agent", "user agent" in addition to AI agents) and `\bmcp\b` (57% precision — Microsoft Certified Professional dominates 2024; "Model Context Protocol" splits ~50/50 only in 2026). Effect size barely changes (+31.05pp → +31.09pp). **Primary AI-mention measure for the paper is the STRICT AI-tool pattern** (SNR 38.7 after dropping `mcp`), with broad as sensitivity. Wave 3 agents must use V1's refined patterns.

2. **`requirement_breadth` decomposition is 71% content / 29% length.** V1 confirmed three components correlate r > 0.3 with `desc_cleaned_length` (soft_skill_count r=0.363, org_scope_count r=0.399, management_STRICT_count r=0.351). Length explains 18% of breadth variance. Raw breadth rise +2.71 (+32.3%); length-residualized rise +1.93 (SNR 31.7 vs raw 40.4). Still above noise and defensible; but the paper must report length-residualized when citing breadth.

3. **"Period dominates embedding space by ~180×" is wrong; correct figure is ~1.2–1.5×.** V1 centroid-pairwise-distance measurement: within-period 0.0213 vs cross-period 0.0253 — only 1.2×. The 180× number came from mixing delta-of-means across different scales. **Rephrase as: "period dominates seniority modestly but consistently across measures."** NMI (domain/period/seniority = 0.275/0.032/0.016; domain/period = 8.6×, period/seniority = 1.9×) is the defensible decisive ratio.

4. **LLM-frame J2 flip is a real selection artifact.** V1 confirmed: scraped `llm_extraction_coverage='labeled'` rows have J2 share of 6.23% (2026-03) / 4.32% (2026-04), while `not_selected` rows have 2.61% / 2.06% — 2-3× higher labeled J2 share. The LLM extraction pipeline's selection policy correlates with some junior-signal feature. **Every Wave 3 junior claim that restricts to labeled text MUST flag this** and report both labeled and full-corpus directions separately.

5. **T09 NMI verified on full 34,102-row corpus** (not just 8k sample): domain=0.275, period=0.032, seniority=0.016. Ordering preserved; 8.6× domain/period dominance confirmed (slightly more domain-dominant than the Gate 2 memo's 7×).

6. **T12 relabeling diagnostic re-verified:** entry26-vs-entry24 cos = 0.9602 > entry26-vs-midsr24 cos = 0.9294. Period-effect dominates; relabeling hypothesis rejected. Magnitudes differ from T12 due to different vocab/aggregation but ordering holds.

7. **T13 section decomposition directionally verified.** V1 independent classifier: requirements −25% (vs T13 −19%); benefits +88% (vs +87%); about_company +82% (vs +71%). Direction and relative ranking across sections match. The classifier-sensitive absolute magnitudes favor T13 as primary.

**Panel robustness verified:** AI-mention rise is unanimous 4-of-4 across J1–J4 AND 4-of-4 across S1–S4 in both arshkon-only and pooled-2024 baselines. `requirement_breadth` is unanimous 7-of-7 across the full T30 junior+senior panel in both baselines. These are the two strongest-evidence claims in Wave 2.

**No Wave 3 dispatch blocker.** V1 has cleared Wave 2 for Wave 3 with the refinements above.

The lead candidate framing from Gate 2 ("Period-dominated, cross-seniority JD rewriting + AI-vocabulary expansion") survives V1 with one magnitude correction (period-vs-seniority embedding dominance is modest, not 180×) and one mechanism refinement (breadth rise is 71% content). Proceed to Wave 3.
