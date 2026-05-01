# SYNTHESIS — 2024→2026 SWE Labor Market Exploration

**Author:** Agent N (Wave 4 synthesis) · **Date:** 2026-04-17
**Upstream:** Gate 0 memo, T01–T23, T28–T30, V1/V2 verification, Gate 1/2/3 memos.
**Downstream:** analysis-phase pipeline + paper draft (consumed first by any agent who writes the paper).

This document is the single synthesis the analysis phase reads first. It contains
the **lead finding**, the **three supporting claims**, the **recommended analytical
samples** for each claim, the **confounders and their mitigations**, the **ranked
findings with evidence strength**, the **new hypotheses** for follow-on analysis,
and the **method recommendations** that every analysis-phase task must respect.

---

## Executive summary

Between 2024 and 2026, US LinkedIn software-engineering job postings underwent a
**SWE-specific, cross-seniority, within-company rewriting toward AI-tool vocabulary**.
Three facts establish this pattern. First, AI-tool mentions in SWE postings rose from
1.5% to 14.9% (strict pattern, SNR 35.4; broad pattern 7.2% → 46.8%, SNR 24.7); 99%
of that strict-pattern rise is SWE-specific under a difference-in-differences comparison
to occupation controls (T18), and 102% of the strict-pattern rise on the 240-company
arshkon∩scraped overlap panel is a within-company shift, not a composition effect
(T16). Second, the cross-domain surface-vocabulary expansion coexists with a senior-
disproportionate role-content shift: mid-senior mentor-binary rate rose 1.46–1.73×
(vs entry 1.07×; T21), orchestration and strategic-scope densities both approximately
doubled at mid-senior (T21), and an emergent "management + orchestration + strategic +
AI" sub-archetype (97% of its members are 2026, n=860) now occupies 13% of the senior
sample (T21 cluster 2). Third, this employer-side rewriting **under-specifies AI
relative to observed developer usage**: broad employer AI-requirement rate 46.8% vs
Stack Overflow 2024 worker usage 62–76% vs GitHub Octoverse 73% vs Anthropic 2025
exposure 75% — the gap is robust across all four usage assumptions in the 50/65/75/85%
band, and seniors are more AI-specified than juniors (51.4% S1 vs 43.5% J2), ruling
out an "AI-as-junior-filter" interpretation (T23). The originally hypothesized
direction (anticipatory employer over-specification) is **inverted**.

The exploration also reframes two pre-registered constructs. **Junior "scope inflation"
is not supported** as a junior-specific phenomenon: requirement-breadth rose equally
at junior and senior (+34% vs +30%; T11), and the formal requirements *section*
actually contracted (−19% chars corpus-wide; SWE −10.7pp share vs control +0.9pp;
T13, T18). **Seniority convergence is rejected**: all three adjacent boundaries
sharpened in 2026, with a +0.134 AUC gain on a yoe-excluded junior-vs-senior panel
(T20). The 2024→2026 junior-share direction is **baseline-contingent** (J1/J2 down
under arshkon-only, up under pooled-2024) and decomposes into: between-company
composition (T06's 95% between-company J3 rise; 81% of AI/ML archetype growth comes
from employers only-in-2026 per T28) plus LLM-label-routing selection (T29 +1.14 SDs
authorship-score shift; V1-confirmed 2-3× labeled-vs-not-labeled J2-share gap). The
paper should not cite "junior share rose" as an unqualified headline.

The **recommended paper positioning is hybrid**: a dataset + measurement contribution
anchoring a substantive labor finding. The substantive lead is "SWE-specific AI-
vocabulary rewriting," not "junior scope inflation." Framing (1) — SWE-specific
AI rewriting — is the lead; framing (3) — tech-giant intern pipeline expansion —
is a methodological subplot; framing (2) — pure platform/boilerplate evolution —
is partially supported (requirements-section contraction, 2026-more-homogeneous
corpus, recruiter-LLM mediation ~15–30% of the rise per T29) but does not dominate.

---

## Paper's lead sentence (draft)

> "Between 2024 and 2026, US LinkedIn software-engineering job postings added AI-tool
> vocabulary at a rate 99% specific to SWE versus control occupations (DiD, T18) and
> 102% attributable to the same 240 companies rewriting their own postings (T16),
> while employers simultaneously *under-specified* AI by 15–30 percentage points
> relative to the most conservative developer-usage benchmark (T23) — a cross-
> seniority, cross-archetype, geographically uniform rewrite that coexists with a
> senior-disproportionate role-content shift toward mentoring and orchestration
> (T21) and sharpening rather than blurring of seniority boundaries (T20)."

---

## Three supporting claims (one sentence each)

1. **Within-company scope broadening is real (T16).** On the 240-company arshkon∩scraped
   overlap panel, length-residualized requirement-breadth rose +1.43 composite units
   within-company (102% of aggregate; stable 94–147% within across five sensitivities),
   76% of companies broadened, 62% broadened by >1.0 unit — the "same companies are
   asking for more kinds of things" pattern holds even after length residualization
   and equally at junior and senior seniority levels.
2. **Senior-role content shift is senior-specific, not corpus-wide templating (T21).**
   Mid-senior mentor-binary rate ratio is 1.46–1.73× 2024→2026, entry is 1.07×, associate
   fell to 0.61× — the mentor rise is concentrated at mid-senior; an emergent
   "management + orchestration + strategic + AI" sub-archetype (n=860, 97% of its
   members are 2026) and a doubled staff-title share (2.6% → 6.3%) indicate a new
   senior role type rather than proportional template expansion.
3. **Employer posting-level AI requirements systematically under-specify worker AI
   usage (T23).** 2026 SWE broad-AI requirement rate is 46.8%, below every plausible
   developer-usage assumption (50, 65, 75, 85%; the closest match is an unusually
   conservative 50% at which S1 broad-AI 51.4% just crosses); under central 75%
   usage the gap is −28 pp for broad and −61 pp for specific tools, inverting the
   pre-registered direction (anticipatory over-specification).

---

## Research question evolution (original → post-exploration)

| Original RQ | Post-exploration successor | Verdict | Lead evidence |
|---|---|---|---|
| RQ1 (junior scope inflation + senior redefinition) | **RQ1a — SWE-specific AI-vocabulary rewriting** (LEAD) | **Confirmed, reframed** | T18 DiD 99% SWE-specific ai_strict; T16 102% within-company |
|  | **RQ1b — Within-company scope broadening** (SUPPORTING) | **Confirmed** | T16 breadth-resid +1.43, 76% of 240 cos |
|  | **RQ1c — Senior archetype shift** (SUPPORTING) | **Confirmed, senior-disproportionate** | T21 mentor 1.46–1.73× mid-senior vs 1.07× entry; cluster 2 97%-2026 |
|  | **RQ1d — Seniority boundary sharpening** (NEW) | **Confirmed** | T20 all three adjacent boundaries +0.054/+0.084/+0.003 AUC; J3/S4 yoe panel +0.134 |
| RQ2 (task migration across seniority) | **RQ2 — Task/requirement migration** (RETAINED, narrowed) | **Diffuse cross-seniority, not migration** | T11 breadth +34% vs +30% junior vs senior; T28 20/22 archetypes positive |
| RQ3 (employer over-specification) | **RQ3 — Employer under-specification of AI** | **Direction INVERTED** | T23 46.8% < 50/65/75/85% band; robust across benchmarks |
| RQ4 (mechanisms — interviews) | **RQ4 — Mechanisms, with quantitative priors** | **Strengthened** | T29 15–30% LLM-authorship mediation; T17 geographic null; T28 81% new-entrant |

**Key conceptual changes:**
- "Junior scope inflation" → rejected as junior-specific; breadth rose equally at every seniority level (T11).
- "Junior-senior convergence" → rejected; boundaries SHARPENED (T20).
- "Anticipatory employer restructuring" → rejected; employers TRAIL workers (T23).
- "Tech-giant intern pipeline expansion" → supporting methodological subplot (T06, T16, T28); 81% of AI/ML-archetype growth is new-entrant-driven.
- "Senior shift to management" → refined; senior shift is to **mentoring + orchestration + AI co-deployment**, not traditional people-management (T11 `manage` had 14% semantic precision; refined to `mentor|coach|hire|headcount|performance_review`).

---

## Data quality verdict per RQ

| RQ | Analyses SAFE | Analyses NEEDING CAVEATS | Analyses UNSAFE |
|---|---|---|---|
| RQ1a (AI rewriting) | DiD SWE vs control on ai_strict / tech_count / ai_tech_count (T18 bootstrap CI clear of zero); within-company 102% on 240-co panel (T16); 4-of-4 J1-J4 panel unanimous AI rise (V1) | ai_broad pattern (0.80 precision per V1 — report strict as primary, broad as sensitivity); any rate cited must specify exact regex + subset + denominator (V1 §4) | Broadly conflating ai_broad and ai_strict (V2 corrected T28 ≥+10pp attribution) |
| RQ1b (scope broadening) | Length-residualized breadth on 240-co overlap (T16); composite-score correlation documented (V1.3 reports soft/scope/mgmt r>0.3 with length) | Raw breadth numbers (report residualized as primary; V1 §3 rule); specialist and aggregator sensitivity mandatory (both survive, T16 §3) | Raw requirement_breadth without length-residualization (71% content / 29% length per V1) |
| RQ1c (senior archetype) | Mentor rate cross-seniority (T21 §7); strict management pattern = `mentor|coach|hire|headcount|performance_review` (V1-refined); arshkon-only baseline on senior | Mentor-rate magnitude method-dependent (V2.7 1.46× vs T21 1.73× mid-senior); precision of `mentor` senior-context is 100% on 20-sample (V2) | Broad management pattern including bare `manage` (14% precision T11) or `stakeholder` (42% precision T21) |
| RQ1d (boundary sharpening) | Per-boundary AUC with T20 7-feature spec (5-fold CV, L2 logistic); J3/S4 yoe-excluded panel (AUC +0.134, V2 reproduced +0.134 exact) | Associate cell n=39/51 → directional only; class-balance sensitivity on associate_vs_entry | Under-cell-threshold per-archetype boundary fits (most < n=50 per boundary × period) |
| RQ2 (migration narrowed) | Archetype-stratified AI/breadth/mentor Δ (T28 22-archetype projection); within-company × within-archetype decomposition (T16 §8) | Junior-senior gap closing claims require report of both arshkon-only and pooled baselines (senior-side asaniczka artifact) | Any "juniors now look like 2024 seniors" claim — T12 relabeling rejected, T20 boundaries sharpened |
| RQ3 (under-specification) | Benchmark comparison across 50/65/75/85% band (T23); alternative framings (V2.2: 63.2% "currently using" gives gap −49.3pp at ai_tool vs broader +28pp at ai_broad) | Worker-benchmarks are self-reported and platform-biased; unit mismatch (postings ≠ workers); the paper must report the qualitative gap direction, not a precise pp | Single-benchmark citation without the 4-assumption sensitivity |
| RQ4 (mechanisms) | AI-strict attenuation 75–77% under low-LLM authorship cut (T29 + V2 agree); new-entrant decomposition AI/ML 81% (T28, V2 exact); geographic null r=−0.11 (T17 + V2 alternative control) | Mentor-rate attenuation 72% (T29) vs 105% (V2 3-feature) — method-sensitive; cite as "15–30% mediated with uncertainty"; T29's 2024 baseline is already post-ChatGPT (lower bound on attenuation) | Sharp citation of "71%" / "72%" T29 retention on mentor/breadth (V2 flagged method-sensitive) |

---

## Recommended analytical samples for the analysis phase

**Default SWE filter.** `source_platform='linkedin' AND is_english=true AND
date_flag='ok' AND is_swe=true`. n = 63,701 (arshkon 4,691; asaniczka 18,129;
scraped 2026-03 19,777; scraped 2026-04 21,104).

**Text-dependent analyses.** Additional filter `text_source='llm'` via
`swe_cleaned_text.parquet` (34,102 rows); always report per-cell coverage.

### Per analysis type

| Analysis | Sample | Required filters | Report under |
|---|---|---|---|
| AI-mention prevalence | All is_swe LinkedIn rows | Strict V1-refined primary; broad as sensitivity | 2024-01 (asaniczka) / 2024-04 (arshkon) / 2026-03 / 2026-04; arshkon-only AND pooled-2024 |
| Within-company decomposition | 240-co arshkon∩scraped overlap panel | ≥3 per co in both periods; specialist and aggregator sensitivity | Cap-50, no-specialist, no-aggregator, pooled-2024 (589 cos), labeled-only (196 cos) |
| Junior-share | T30 panel J1/J2/J3/J4 | Primary J2; J3 for YOE-based; J1 for label-strict; J4 for generous YOE | arshkon-only AND pooled-2024; full corpus AND `llm_extraction_coverage='labeled'` (report flip) |
| Senior-share / senior content | T30 panel S1/S2/S3/S4 | Primary S1 (mid-senior+director); S2 (director) sparse | arshkon-only primary; pooled-2024 as sensitivity (asaniczka senior-inflation) |
| Requirement breadth | T11 feature parquet, LLM-frame only (n=34,102) | Length-residualized (global OLS); raw as sensitivity | J2, J3, S1, S4 panels; aggregator and specialist exclusion |
| Section anatomy | T13 classifier, LLM-frame + raw fallback | Section-by-section chars AND share | SWE + adjacent + control (T18 shows divergent direction) |
| Senior archetype / mentor | T21 spec, LLM-frame only | V1-refined mgmt-strict pattern; bare `\bmentor\w*` for cross-seniority | Seniority × period × archetype (T28 archetype labels projected) |
| Cross-occupation DiD | Full LinkedIn, group = swe / swe_adjacent / control | Raw text for cross-group comparability (T18 §1); aggregator-excl sensitivity | Bootstrap 400 reps, 95% CI per metric |
| Employer-usage divergence | All-SWE, J2, J3, S1, S4 (T23 spec) | V1-refined strict as primary; broad as sensitivity; ai_tool (strict ∖ codex) for specific-tool slice | 4-band worker-usage sensitivity (50/65/75/85%) |
| Cluster discovery / archetype projection | Shared `swe_archetype_labels.parquet` (n=8,000); nearest-centroid projection to 34,102 | T09 labels; BERTopic seed-retry logic | Per-archetype AI strict and broad; n ≥ 50 per period cell |
| Authorship mediation | Full 63,701 corpus; low-40% by within-period score | Primary cut within-period; global cut as stress test | AI-strict retention, mentor-all, mentor-senior, breadth_resid |

---

## Seniority validation summary

`seniority_final` is defensible (Cohen's κ vs the Stage-5 title rules: arshkon 0.40,
scraped 0.49; T03). No dual-flag violations; YOE ordering is correct. BUT:

- **LLM-frame selection artifact (V1.5 confirmed):** scraped 2026
  `llm_extraction_coverage='labeled'` rows have J2 share 4.3–6.2% while not-selected
  rows have 2.1–2.6%. LLM-labeling preferentially selects junior postings; analyses
  restricted to the labeled frame systematically overstate junior direction by +1–2pp.
  Every junior claim must report BOTH labeled and full-corpus directions.

- **arshkon native entry has 26.4% ≥5 YOE (T08).** Native entry labels are YOE-
  incoherent on arshkon; any "entry postings want 5 YOE" finding likely predates AI
  and is a long-standing posting convention, not an AI-era shift.

- **"Graduate"/"New College Graduate" under-detected (T03):** ~1,000 scraped SWE
  rows have clear junior titles but `seniority_final='unknown'`. Preprocessing fix
  recommended; in current data J1 is under-counted by ~2.4pp on scraped.

- **`title_normalized` strips level indicators (T30):** J5/S3 title-keyword matches
  must operate on raw `title`, not `title_normalized`.

- **Seniors changed more than juniors 2024→2026 in content (T12 + V1.6 verified):**
  cos(entry26, entry24) = 0.9602 > cos(entry26, midsr24) = 0.9294 → period-effect
  dominates; relabeling rejected; seniors' cosine shift (0.9421) is LARGER than
  entry's (0.9602).

---

## Known confounders with severity assessment

| Confounder | Severity | Mitigation | Residual risk |
|---|---|---|---|
| **Description length growth** (+35% median cleaned, +56% raw) | **High** | T13 section decomposition shows requirements SHRANK while narrative grew; V1.3 length-residualization on requirement_breadth (71% content / 29% length); T14 density reports (AI-mention 1.28× density over non-AI, not just longer) | Some composite scores still length-correlated; cite residualized primary |
| **Asaniczka 0 native entry labels** | **High** | Arshkon-only baseline primary on senior-side; pooled as sensitivity; YOE-based proxies (J3/J4) label-independent | Senior-share direction depends entirely on baseline choice; paper MUST name the baseline |
| **Aggregator share doubled** (9.2% → 16.6%) | Medium | Aggregator-exclusion is pre-committed sensitivity (Gate 0); T16 shows aggregator and direct trajectories are nearly identical on AI+breadth (+14.1 vs +12.4pp); T18 aggregator exclusion sharpens DiD (+10% on ai_broad) | Aggregator-excluded should always be reported as sensitivity |
| **Company composition shift** (74.5% new entrants in scraped) | **High** | T06 decomposition: 95% between-company on J3; T16 within-company 102% on AI-strict; T28 81% of AI/ML growth is new-entrant-driven | Any longitudinal corpus-level claim needs returning-cos-only sensitivity |
| **Field-wide vs SWE-specific uncertainty** | Medium | T18 DiD: ai_strict 99% SWE-specific; tech_count 95%; desc_len 37% (macro); soft_skills 0% (macro) | Paper can safely cite SWE-specific for ai_strict/tech/breadth; CANNOT for length or soft skills |
| **LLM-frame selection artifact (V1.5)** | **High** (for junior claims) | Every junior claim reports full-corpus AND labeled subset | Junior-direction is fundamentally baseline-contingent AND frame-contingent |
| **Recruiter-LLM mediation (T29)** | Medium | Report low-LLM subset as sensitivity; AI-strict retains 75–77% across score variants; mentor retention 72% (T29) vs 105% (V2); cite as 15–30% mediated with uncertainty | Mentor-rate is the most LLM-sensitive finding; cite with explicit caveat |
| **2026 = JOLTS hiring low** (Info-sector openings 91K = 0.66× 2023 avg) | Medium | All claims framed as share-of-SWE, not volume; T07 JOLTS context in every metro / temporal analysis | Volume-shift language prohibited in paper |
| **Platform taxonomy drift** ("IT Services and IT Consulting" → "Software Development" +17pp relabel; bare "developer" lost 61pp native entry) | **High** | T05 shared-title stability check (14/20 common titles lost native entry); T10 legacy-stack title consolidation; Indeed cross-validation matches LinkedIn on J1/J3 levels | LinkedIn native-label semantics drifted; any cross-period native-label comparison must flag |
| **`is_remote_inferred` 100% False** | Pipeline bug | Confirmed T17; no remote/hybrid analysis is feasible until preprocessing fix | Blocks any remote-work claim |
| **Description length as confound in authorship score** (r=0.44 with score) | Medium | Length-residualized breadth r drops to 0.16 from 0.31; structural req_header presence has zero correlation | Authorship score cannot separate prose from length |
| **AI-broad pattern 0.80 precision** | Medium | V1 refinement dropped `agent` (66%) and `mcp` (57%); residual broad pattern includes bare `ai`/`ml` which pull AMB in narrow contexts | Strict pattern (SNR 38.7) is primary; broad is sensitivity |

---

## Discovery findings organized

### Confirmed hypotheses (with confidence)

1. **AI-vocabulary rewriting is SWE-specific (T18, V2 verified).** 99% of the ai_strict
   rise is SWE-specific under DiD; tech_count 95%; requirement_breadth 72%. Control
   postings do not adopt named AI tools (0.002 → 0.002). Bootstrap 95% CIs clear of
   zero. **Confidence: high.**
2. **Within-company scope broadening (T16 + V2 exact reproduction).** Length-residualized
   +1.43 breadth; 102% within-company on AI-strict; 102% within on breadth_resid;
   76% of 240 cos positive. **Confidence: high.**
3. **Senior archetype shift is senior-specific (T21 + V2 direction verified).** Mid-
   senior mentor 1.46-1.73× vs entry 1.07×; mgmt+orch+strat+AI sub-archetype 97%-2026
   n=860; 8-domain stratification (T28) shows senior mentor Δ +4 to +26pp across 20
   archetypes. **Confidence: high (direction); moderate (magnitude — report as range).**
4. **Seniority boundaries sharpened (T20 + V2 within 0.01-0.02 AUC).** All three
   adjacent boundaries +0.054/+0.084/+0.003; J3/S4 yoe-excluded panel +0.134 AUC
   (V2: +0.134 exact). ML/AI gained most clarity (+0.105). **Confidence: high.**
5. **AI rise is cross-archetype (T28 + V2 corrected).** AI-BROAD positive in 22/22,
   ≥+10pp in 18/22; AI-STRICT positive in 21/22, ≥+5pp in 16/22, ≥+10pp in 6/22;
   systems_engineering is the clean zero-AI-strict control (+0.16pp, V2 exact).
   **Confidence: high.**
6. **AI rise is geographically uniform (T17 + V2 spot-check).** 26/26 metros positive;
   CV 0.29; r(ΔAI, ΔJ2) = −0.11 null; Sunbelt leaders (Tampa Bay/Atlanta/Charlotte)
   not tech hubs. **Confidence: high.**
7. **New-entrant-driven AI/ML growth (T28 + V2 exact).** 81% of 2026 AI/ML volume
   from employers only-in-2026 (935 cos); 23% from both-period (83 cos). **Confidence:
   high.**
8. **Requirements-section contracted SWE-specifically (T13 + T18 + V2 non-SWE
   validation).** SWE requirements-section share −10.7pp; adjacent −10.9pp; control
   +0.9pp (opposite direction). V2 verified on 500 non-SWE postings that the classifier
   works on control (has_req_rate 73-86%). **Confidence: high.**

### Contradicted hypotheses (with evidence)

1. **Junior scope inflation** (original RQ1 junior-specific). Breadth rose equally at
   junior and senior (J2 +39% vs S1 +34%; T11); scope inflation is NOT junior-specific.
2. **Junior-senior convergence** (original RQ1 embedded claim). All boundaries
   SHARPENED, not blurred (T20); the 2026 junior-vs-senior language gap is larger
   than 2024 by ~14 AUC points on the yoe-excluded panel.
3. **Anticipatory employer over-specification of AI** (original RQ3). Direction
   INVERTED: employers under-specify by 15–30 pp vs worker-usage benchmarks (T23).
4. **Junior share rose unambiguously** (default pre-exploration reading). BASELINE-
   CONTINGENT: arshkon-only J1/J2 DOWN −0.6 pp, J3/J4 UP; pooled-2024 all UP (T30);
   SNR < 1 on every junior metric (T05); LLM-frame flip confirmed (V1.5).
5. **Management language rose in senior postings** (under broad pattern). Broad
   `manage` has 14% SWE semantic precision (T11); signal dominated by non-managerial
   uses. Refined to `mentor|coach|hire|headcount|performance_review` → mentor is the
   single driving term and is senior-disproportionate.
6. **AI postings are more aspirational than non-AI** (junior-filter hypothesis).
   Modest at sentence-level (matched-share +0.24), NOT elevated at posting-level
   (LLM ghost risk-ratio 0.98 for ai_strict, 0.84 for ai_broad vs non-AI); reframe
   as "emerging-demand framing" not "aspirational padding" (T22).
7. **Period ~180× seniority in embedding space** (Gate 2 overstatement). V1 re-derived
   as ~1.2× centroid-pairwise; NMI period/seniority ratio is 1.9× (V1 §4). Use NMI
   8.6× domain/period as the defensible ratio.

### New discoveries (with novelty assessment)

1. **T23 RQ3 inversion** (NOVEL, high). The 2026 employer ai_broad rate (46.8%) is
   below the 50% floor of the plausible worker-usage range. Pre-registered direction
   was opposite. **This is a publishable finding on its own.**
2. **T21 "mgmt+orch+strat+AI" sub-archetype** (NOVEL, medium-high). 97% of cluster 2
   members are 2026 (n=860; vs n=30 in 2024 = noise). Staff-title profile concentrates
   the same bundle. Names a new senior role type for the management-literature.
3. **T20 seniority boundary SHARPENING** (novel, medium). All three adjacent boundaries
   gained AUC; ML/AI gained the most (+0.105). Management-density replaced tech-count
   as the #2 discriminator at the mid-senior boundary in 2026.
4. **T28 AI-rise cross-archetype + `systems_engineering` zero-AI control** (novel,
   medium). AI-STRICT rose in 21/22 archetypes; systems_engineering is +0.16pp — a
   clean natural-experiment control for interviews (RQ4).
5. **T17 geographic null** (novel, medium). r(ΔAI, ΔJ2) = −0.11 rejects the spatial
   "AI exposure → junior displacement" narrative. Tampa Bay/Atlanta/Charlotte lead
   AI surge, not SF Bay/Seattle.
6. **T29 partial-mediation by recruiter-LLM** (novel, medium-high). Authorship-score
   shift 1.14 SDs; AI-strict Δ retains 75–77% under primary cut; mentor/breadth
   method-sensitive. Establishes that 15–30% of the rewrite is recruiter tooling —
   a substantive methodological contribution.

### Unresolved tensions

1. **T11 breadth +39% vs T13 requirements-section −19%.** Compatible but must be
   explained explicitly in the paper. Breadth counts TYPES across the whole cleaned
   description; requirements-section counts CHARS in one section. The mechanism is
   "types migrating from requirements into responsibilities + role_summary +
   about_company."
2. **T06 95% between-company J3 vs T09 84% within-archetype J2.** Different operational
   variables (YOE-based vs label-based). The aggregate "junior rose" decomposes into
   (a) new-entrant firms (tech-giant intern pipelines) and (b) within-archetype label
   drift — compatible.
3. **T12 seniors-changed-more vs T20 ML/AI-sharpened-most.** Seniors changed more in
   absolute cosine; ML/AI gained the most boundary clarity because its 2024 junior-
   senior overlap was worst. Both true; separate claims.
4. **T29 mentor retention 72% vs V2 3-feature 105%.** Method-sensitive. The defensible
   claim is "method-dependent attenuation 0–30% on mentor/breadth; AI-strict is
   robust across specifications at 75–77%."

---

## Posting archetype summary (T09 + T28)

22 archetypes discovered via BERTopic (k_min_topic=30, seed=20260417) on 8,000 LLM-
cleaned stratified sample; projected to full 34,102 LLM corpus via nearest-centroid
(T28). Primary NMI axis: **domain** (0.275) >> period (0.032) >> seniority (0.016).
Domain/period ratio 8.6× (V1 full-corpus verified).

**Top 10 archetypes by 2024→2026 volume and share change:**

| Archetype | 2024 share | 2026 share | Δpp | AI-strict Δ | Key mechanism |
|---|---:|---:|---:|---:|---|
| ai_ml_engineering | 2.6% | 16.2% | **+13.6** | +26.4pp | **81% new-entrant driven** (T28 §8) |
| backend_platform | 3.5% | 7.0% | +3.5 | +10.2pp | Content-dense growth |
| cloud_devops | 7.6% | 10.9% | +3.3 | +5.8pp | AI tooling in CI/CD contexts |
| data_engineering | 7.7% | 9.2% | +1.6 | +5.4pp | Modest share growth |
| generic_software_engineer | 15.6% | 15.4% | −0.2 | +3.0pp | Largest single driver of J2 within-archetype rise |
| frontend_react | 6.8% | 6.7% | −0.1 | +8.6pp | Stable |
| java_spring_backend | 6.6% | 4.8% | −1.8 | +5.7pp | Shrinking slightly |
| network_security_linux | 5.1% | 3.5% | −1.6 | +2.1pp | Declining |
| customer_project_engineer | 5.9% | 3.1% | −2.8 | +3.5pp | Declining |
| systems_engineering | 11.0% | 3.4% | **−7.6** | **+0.16pp** | **Zero-AI natural control** |

---

## Technology evolution summary (T14)

- **Rising (top 5 by raw pp):** python +17.0, ci/cd +14.6, llm_token +11.4, typescript +8.7, kubernetes +7.9. 7 of top 20 rising are AI-era; 13 are traditional cloud/DevOps/Python.
- **Declining (top 5):** dotnet −4.2, html −3.6, css −3.2, agile −2.7, sql −2.5. Length-normalized, agile/sql/javascript are null; dotnet/jquery/php are genuine stack-rotation.
- **New coalescence (2026 LLM-vendor cluster, n=13):** `llm_token, rag, copilot, claude, langchain, langgraph, openai, gemini, mcp, anthropic, cursor, fine_tuning, agent_framework` at phi > 0.15. In 2024 these were singletons; by 2026 they form the single most internally-cohesive community. Adjusted Rand Index 0.545 between 2024 and 2026 on 89-tech shared panel.
- **Structured-vs-extracted validation (asaniczka skills_raw vs description regex):** Spearman ρ = 0.985 across 107 techs. Description extraction is a recall superset.
- **Stack-diversity**: tech_count mean 5.4 → 7.0 (+1.6) but density/1k fell from 5.04 → 3.47. The raw-count rise is partly length-driven; AI-mention density 1.28× (length-normalized) is real.

---

## Geographic heterogeneity summary (T17)

- **26 metros well-powered** (≥50 SWE/period in both pooled-2024 and scraped).
- **AI surge uniform:** 26/26 metros positive; mean Δ_ai_strict +12.8pp; CV 0.29.
  Tech-hub mean +13.1pp; non-hub mean +12.7pp (essentially equal).
- **Top 3 AI surges:** Tampa Bay +20.0pp, Atlanta +18.6pp, Charlotte +17.2pp
  (financial-services / healthcare / aerospace Sunbelt metros — NOT tech hubs).
- **Null correlation:** r(Δ_ai_strict, Δ_entry_j2) = −0.108 (p = 0.60);
  Spearman −0.143. **Spatial "AI exposure → junior displacement" story is
  rejected.**
- **Domain concentration:** SF Bay 2026 ML-archetype share 32% (up from 12%);
  Seattle 29% (up from 4%). Other metros grew ML from tiny bases (Atlanta 17%,
  Raleigh-Durham 21%). The ML-archetype concentration widened in tech hubs.
- **`is_remote_inferred` = 100% False** pipeline bug precludes remote analysis.

---

## Senior archetype characterization (T21)

- **Patterns (V1 precision-verified):**
  - `people_management_strict` = `mentor|coach|hire|headcount|performance_review`
  - `technical_orchestration_strict` = `architecture review|code review|system design|
    technical direction|ai orchestration|workflow|pipeline|automation|evaluate|
    validate|quality gate|guardrails|prompt engineering|tool selection`
  - `strategic_scope_strict` = `business impact|revenue|product strategy|roadmap|
    prioritization|resource allocation|budgeting|cross-functional alignment`
  - `stakeholder` (42% precision) and `agent` (44% precision) dropped from strict;
    `team_building` (10% precision) dropped entirely.

- **S1 (mid-senior + director) binary-share Δ 2024→2026:**
  - mgmt +14pp (mid-senior 19.8% → 33.7%) / +9pp at director
  - orch-strict +21pp at mid-senior / +28pp at director
  - strat-strict +6pp mid-senior / +16pp director
  - strat-broad +12pp mid-senior / +18pp director
  - AI-strict +14pp mid-senior / +23pp director

- **Cross-seniority mentor ratio (the central senior-shift evidence):**
  entry 1.07×, associate 0.61×, mid-senior 1.46–1.73× (T21 / V2), director 1.30×.
  **Senior-disproportionate — rejects corpus-wide templating.**

- **Emergent sub-archetypes (k-means k=4, senior-only, standardized features):**
  - baseline_ic (n=4,795, 70% 2026)
  - mgmt_orch (n=1,523, 78% 2026, mentor 96%, AI 0%)
  - strat_scope (n=1,052, 76% 2026)
  - **mgmt_orch_strat_ai (n=860, 97% 2026, AI 100%, orch 0.85, mgmt 0.25, mentor 0.39)**

- **Staff-title profile:** 2026 staff-titled seniors have +16pp mgmt, +11pp strat,
  +16pp mentor, +5pp AI vs non-staff seniors. Staff-title share doubled 2.6% → 6.3%.

- **Domain stratification:** shift is ML/AI-concentrated for AI and orchestration,
  frontend-concentrated for management/mentorship, systems-engineering near-zero.

---

## Ghost / aspirational prevalence (T22)

- **Validated patterns (all ≥80% precision):** `validated_mgmt_patterns.json`.
- **AI-sentence-level aspiration:** pooled ratio 1.75× (2024) → 3.20× (2026);
  matched-share delta +0.24 in 2026.
- **Posting-level LLM ghost rubric:** AI-mentioning postings NOT elevated. RR(ai_strict)
  = 0.98; RR(ai_broad) = 0.89. No reframe to "aspirational padding."
- **Conclusion:** AI-specific requirements are introduced with hedging (`exposure to`,
  `familiarity with`) — emerging-skill demand framing, not a step-change hard
  requirement. Postings remain realistic as whole documents.

---

## New hypotheses (from T24) — ranked for analysis-phase priority

| # | Hypothesis | Test with existing data | Priority |
|---:|---|---|---:|
| 1 | H_A — Cross-occupation employer under-specification of AI | Stratify is_swe_adjacent + is_control by ONET; benchmark vs occupation-specific usage | **1** |
| 2 | H_B — Requirement-section contraction as hidden hiring-bar signal | Regress requirements-share on period × seniority × archetype; company-level correlation with J3 rise | 2 |
| 3 | H_C — "AI-enabled tech lead" as emergent senior role type | Within T21 cluster 2: title distribution + within-company trajectory | 3 |
| 4 | H_D — Senior-specific mentor rise is team-multiplier IC, not manager-ladder | Classify title as IC vs manager; mentor rate stratification | 4 |
| 5 | H_E — Same-company J1 drop + J3 rise = labeling-regime shift | Joint seniority_native × yoe distribution on 240-co panel | 5 |
| 6 | H_H — 2026 corpus samples different firm population | Restrict 2026 to returning-cos-only; recompute every headline | 6 |
| 7 | H_I — AI as coordination signal, not skill demand | Cross-reference with firm-level AI patents / earnings-call AI mentions | 7 |
| 8 | H_F — Sunbelt/tech-hub convergence on AI baseline | Per-metro 2026 level convergence check | 8 |
| 9 | H_G — Staff-title doubling as senior-tier internal redistribution | Within-co: 2024 mid-senior count vs 2026 staff count | 9 |
| 10 | H_J — Recruiter-LLM adoption senior-biased by HR tooling defaults | T29 score × seniority stratification | 10 |

See `exploration/reports/T24.md` for full statements and motivating evidence.

---

## Method recommendations for the analysis phase (MANDATORY)

1. **T30 seniority panel.** Every seniority-stratified headline must report the 4-row
   ablation (J1/J2/J3/J4 for junior claims, S1/S2/S3/S4 for senior). Load
   `exploration/artifacts/shared/seniority_definition_panel.csv`; do not re-derive.
2. **V1-refined AI patterns.** Strict pattern primary (SNR 38.7):
   `\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|
   langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|
   huggingface)\b`. Broad as sensitivity (dropped `agent_bare`, `mcp`). Always
   cite by name.
3. **Length-residualization on any composite with length-correlating components.**
   V1.3 found soft_skill_count r=0.363, org_scope_count r=0.399, management_STRICT
   r=0.351 with desc_cleaned_length. Primary metric for breadth is residualized
   (a=6.498, b=0.00182 global OLS fit). Cite raw as sensitivity only.
4. **Semantic precision sampling for every new pattern.** Sample 50 matches per
   sub-term stratified 25/25 by period; apply 80% precision floor; drop failing
   sub-terms and re-run compound. `manage` (14%), `stakeholder` (42%), `agent` (44%),
   `mcp` (57%), `team_building` (10%) are all known fails — do not use unqualified.
5. **DiD vs control for every SWE-specific claim.** Any claim "SWE is experiencing X"
   must test against is_swe_adjacent AND is_control. Use T18's framework. Bootstrap
   CI per metric.
6. **Section-decomposed text analyses.** T13 classifier is the primary tool; V2
   verified it works on non-SWE at 73-86% has_req_rate. Any prose-level finding
   must be cross-checked at the section-level.
7. **Report BOTH full-corpus AND LLM-labeled subsets.** The J2 flip (V1.5) and the
   30% scraped LLM-frame coverage make every junior/text claim require dual reporting.
8. **Full-corpus vs 240-co overlap sensitivity.** For any longitudinal claim, report
   the within-company decomposition on the 240-co panel AND the full-corpus
   aggregate. If they disagree by >30%, the aggregate is composition-dominated.
9. **Arshkon-only baseline for senior-side findings.** Asaniczka's missing native
   entry labels inflate pooled-2024 senior share by 19pp. Arshkon-only is the fairer
   comparison for native-label content; pooled-2024 as sensitivity.
10. **T29 authorship-subset sensitivity for any AI-content claim.** Report headline
    Δ on full corpus AND on low-40% within-period subset. Mentor/breadth findings
    carry an explicit "method-sensitive 0-30% mediation" caveat.

---

## Sensitivity requirements per finding

| Finding | Required sensitivities | Status |
|---|---|---|
| AI-strict rewriting cross-period | Specialist-excl, aggregator-excl, LLM-frame, arshkon-only AND pooled, cap-50, broad-pattern-as-alt | All reported (T16 / T18 / T28 / V1 / V2) |
| Within-company breadth +1.43 | No-specialist (221 cos), no-aggregator (213), pooled (589), cap-50, labeled-only (196) | 94-147% within across all five (T16 §3) |
| Senior mentor 1.46-1.73× | Cross-seniority panel; arshkon-only vs pooled; pattern precision 100% on 20-sample | Done (T21 / V2 §1.7) |
| Seniority boundary sharpening | Primary + no-aggregator sensitivity (increases sharpening); per-archetype (where n≥50) | Done (T20 + V2 replicate) |
| Employer-usage divergence | 4-band usage 50/65/75/85%; strict AND broad; J2/J3/S1/S4 panels; alternative-usage framing ("currently using 63%") | All done (T23 + V2.2) |
| LLM-authorship mediation | Within-period vs global cut; 5-feature vs 3-feature score (V2); per-metric attenuation | Done (T29 + V2.5) |
| Cross-occupation DiD | Aggregator-excl, tier-strict; binary vs count metric | Done (T18 + V2.1) |
| Junior-share baseline-contingency | 4-row panel J1/J2/J3/J4 × 2 baselines (arshkon-only, pooled); LLM-frame vs full | Done (T30 + V1.5) |
| AI/ML 81% new-entrant | Exact employer-count decomposition; aggregator-contamination check | Done (T28 + V2.4) |
| Geographic AI uniformity | No-aggregator, no-specialist sensitivities; arshkon-vs-scraped additional panel | Done (T17 §7) |

---

## Interview priorities (T25 handoff)

RQ4 interviews should adjudicate five concrete questions, each with a dedicated artifact
in `exploration/artifacts/T25_interview/`:

1. **Scope realism:** Would a 2024 or 2026 entry-level candidate realistically meet the
   stated scope in `inflated_junior_jds.md`? Four ghost-flagged entry-labeled JDs are
   attached; probe hiring-manager intent vs recruiter boilerplate.
2. **Content change vs stylistic change:** What in the 2026 version of each paired JD
   (in `paired_jds_over_time.md`) reflects real hiring-requirement change vs
   LLM-augmented template? Four same-company 2024/2026 pairs on the 240-co overlap
   panel.
3. **Employer-usage gap mechanism:** Why do SWE employers mention AI tools in 46.8%
   of JDs when developers use them at 62-75% rates (`employer_usage_divergence.png`)?
   Four candidate mechanisms: JD template lag, implicit assumption, hiring-manager
   belief that AI is "picked up on the job," or AI as an internal filtering criterion
   not stated in public postings.
4. **Senior role redefinition:** Is the mgmt+orch+strat+AI senior profile
   (`senior_archetype_chart.png`) a new role or a relabeling of existing seniors?
   Probe with systems-engineering as the natural-experiment control (AI-strict
   +0.16pp; senior mentor Δ +7.6pp).
5. **LLM-authorship:** When drafting a JD today vs 3 years ago, what is your pipeline?
   What percentage of JD language came from an LLM draft vs a copy-edit of an existing
   template? This probes T29's 15-30% mediation estimate qualitatively.

Sample sizes for interviews: 10-15 hiring managers + 5 recruiters + 3 HR-tooling
operators, stratified by archetype (AI/ML, generic SWE, systems-engineering
control) and by firm size (T06 entry-specialist vs. direct employer).

---

## Recommended paper positioning

**Primary framing:** *hybrid dataset + substantive empirical labor paper with
"SWE-specific AI rewriting" as the substantive lead.*

Three contributions stacked:

1. **A longitudinal SWE JD dataset + preprocessing pipeline** harmonizing three
   sources (arshkon, asaniczka, scraped) with 63,701 LinkedIn SWE rows and 155,745
   control-occupation rows.
2. **A measurement framework** (T30 panel, sensitivity dimensions, semantic
   precision protocol, composite-score correlation check, LLM-authorship mediation
   test) usable by future researchers working on posting data in the LLM era.
3. **Three substantive findings** in order of evidence strength:
   - SWE-specific AI-vocabulary rewriting (99% DiD) is cross-seniority, cross-
     archetype (20/22 positive strict), within-company (102%), geographically
     uniform (26/26 metros);
   - RQ3 direction inversion: employers under-specify AI by 15-30 pp vs workers;
   - Senior-disproportionate role shift toward mentoring + orchestration +
     AI-deployment, with an emergent mgmt+orch+strat+AI sub-archetype.

**Venue implications.**
- **ICWSM / WWW / CSCW**: the methodological + hybrid positioning lands.
- **Labor economics (ILR Review, JOLE, BE Journal)**: the RQ3 inversion alone is
  publishable; the extended findings support the substantive contribution.
- **CHI / Management Science**: the emergent senior role type (H_C / T21 cluster 2)
  could anchor a specific paper on tech-lead work.
- **Nature Human Behaviour / PNAS**: the cross-occupation extension (H_A) could
  elevate the paper if implemented.

**Recommended lead venue for initial submission:** ICWSM dataset & methods track
with a separate labor-economics follow-up. The RQ3 inversion is novel enough to
anchor a standalone short paper or a dedicated section.

**Lead sentence draft** (from earlier in document):
> "Between 2024 and 2026, US LinkedIn software-engineering job postings added AI-tool
> vocabulary at a rate 99% specific to SWE versus control occupations (DiD), 102%
> attributable to the same 240 companies rewriting their own postings, while
> employers simultaneously under-specified AI by 15–30 percentage points relative
> to the most conservative developer-usage benchmark — a cross-seniority, cross-
> archetype, geographically uniform rewrite that coexists with a senior-
> disproportionate role-content shift toward mentoring and orchestration and
> sharpening rather than blurring of seniority boundaries."

---

## Appendix — reference artifacts

### Shared artifacts (under `exploration/artifacts/shared/`)
- `seniority_definition_panel.csv` (T30) — 44 rows of junior/senior definitions × sources.
- `entry_specialist_employers.csv` (T06) — 240 flagged specialist cos.
- `swe_cleaned_text.parquet` (Prep) — LLM-cleaned text, 34,102 rows.
- `swe_embeddings.npy` + `swe_embedding_index.parquet` — MiniLM 384-d, 34,102 rows.
- `swe_tech_matrix.parquet` — 107 boolean tech cols, 63,701 rows.
- `swe_archetype_labels.parquet` (T09) — 22 archetypes on 8,000 sample.
- `company_stoplist.txt` — token-level canonical company stoplist.
- `asaniczka_structured_skills.parquet` — 107-tech baseline.
- `calibration_table.csv` — AI-mention + breadth benchmarks by source.
- `validated_mgmt_patterns.json` (T22) — 8 patterns, all ≥80% precision.

### Wave 4 outputs (this agent)
- `exploration/reports/T24.md` — 10 new hypotheses ranked.
- `exploration/artifacts/T25_interview/` — 6 interview elicitation artifacts + README.
- `exploration/reports/SYNTHESIS.md` — this document.

### Gate memos
- `exploration/memos/gate_0_pre_exploration.md`
- `exploration/memos/gate_1.md`
- `exploration/memos/gate_2.md`
- `exploration/memos/gate_3.md`

### Verification reports
- `exploration/reports/V1_verification.md` — Gate 2 adversarial (5/6 verified, 1 corrected).
- `exploration/reports/V2_verification.md` — Gate 3 adversarial (5/8 verified, 2 flagged, 1 corrected).

### Task reports
- T01–T07 (Wave 1, data foundation)
- T08–T15 (Wave 2, structural discovery)
- T16–T23 (Wave 3, market dynamics)
- T28–T30 (cross-wave dependencies)

**End SYNTHESIS.md. Analysis phase may begin.**
