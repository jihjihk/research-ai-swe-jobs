# Gate 1 Research Memo

Date: 2026-04-17
Gate: after Wave 1 (Agents A-D) · before Wave 1.5 dispatch
Reports read: T01 (profile), T02 (asaniczka), T03 (seniority audit), T30 (seniority panel), T04 (SWE audit), T05 (comparability), T06 (concentration), T07 (feasibility)

## What we learned

Wave 1 converged on four load-bearing facts that reshape the rest of the exploration. None of them were in the Gate 0 priors list verbatim, and each is more informative than a feasibility check.

**1. The within-2024 cross-source variability is LARGER than the 2024→2026 cross-period effect on every junior share metric.** T05's SNR table is the single most consequential number in Wave 1: J1 SNR 0.19, J2 SNR 0.24, J3 SNR 0.43 — all well below the materiality threshold of 2. Arshkon-only-vs-scraped J1 is essentially flat (+3.73% → +3.12%, a tiny decline); pooled-2024-vs-scraped J1 looks positive (+1.66% → +3.12%) but that's because asaniczka (which has zero native entry labels) drags pooled-2024 down. **Any "junior share rose in 2026" headline is baseline-contingent.** T30's "all junior defs go up" pattern is a pooled-2024 artifact, not a defendable directional finding.

**2. The J3/J4 YOE-based junior rise is ~95% between-company compositional.** T06's decomposition on the 125-company arshkon∩scraped overlap panel: aggregate J3 rises +7.2pp, of which +6.9pp is between-company (different employers entering/exiting) and only +0.4pp is within-company (same employers lowering their YOE floor). Excluding the 240 entry-specialist employers shrinks the full-corpus J3 gap by 67% (+2.21pp → +0.72pp). Meanwhile J1 (label-based) moves slightly *negative* within the same overlap panel (−0.65pp within-company). **The "entry share went up" story is an artifact of which companies the scraper caught in 2026 — tech-giant intern pipelines (Google J3=52%, Walmart J3=74%, Microsoft J3=39%) that are largely absent from the Kaggle 2024 snapshots.**

**3. Exactly one aggregate metric is unambiguously a within-company 2024→2026 shift: AI-mention prevalence.** T06's same decomposition shows AI-mention share jumped +36.5pp in the overlap panel, of which +33.3pp (91%) is *within-company* and only +3.2pp is between. This is the employer-side rewriting signal the research design was looking for, and it is the one aggregate that survives composition, calibration, and concentration checks. Every other "change" at the aggregate level either flips sign under T30 panel variants (junior share), halves under aggregator exclusion (senior share), or collapses under specialist exclusion (J3 entry share). AI-mention does not.

**4. 2026-03/04 is a genuinely soft labor market, not a boom.** T07's JOLTS Information-sector series: Feb-2026 openings = 91K = 0.66× of 2023 average, 0.74× of 2024 average — the lowest info-sector openings reading since the post-COVID 2020 trough. Any framing that says "employers are hiring more X in 2026" needs to be share-of-SWE, not volume. This is the first macroeconomic fact the paper cannot ignore; it changes the narrative choice between "AI reshaped hiring" and "employers used an AI-era hiring slowdown to restructure their asks."

Supporting facts the waves build on:
- `seniority_final` is defensible (κ arshkon 0.40, scraped 0.49; zero dual-flag violations; YOE ordering correct); primary slices are **J2 (entry+associate)** and **S1 (mid-senior+director)**. Power is fine for every population-share comparison. Binding power constraint is within-slice content on S2 (director-only, n=13–295) and J1/J2 under arshkon-only.
- `title_normalized` strips level indicators, so J5/S3 must operate on raw `title` — T30 caught this and corrected.
- "Graduate"/"New College Graduate" is a missed entry signal in Stage 5 rules (~1,000 scraped SWE rows have clear junior titles but `seniority_final = 'unknown'`). Preprocessing fix recommended, but not blocking.
- Indeed cross-validation on scraped (J1 = 3.05%, J3 = 18.1%) matches LinkedIn scraped (J1 = 3.12%, J3 = 17.2%) within 1pp — the "LinkedIn-specific-artifact" hypothesis is weakened at the end-of-pipeline level.
- Geographic representativeness vs BLS OEWS: Pearson r = 0.96–0.98 across sources. Data is nationally representative at the top-10 state level; 20 small states are zero in scraped by 26-metro design.
- Aggregator share nearly doubled between arshkon (9.2%) and scraped (16.6%). Every corpus-level text aggregate must report aggregator-excluded as a sensitivity.
- 74.5% of scraped companies are new entrants (no 2024 match). New entrants are aggregator-heavier (20% vs 14%) and slightly more junior-labeled on J1.

## What surprised us

**The direction of the junior change depends on arithmetic, not on the market.** Under arshkon-only-vs-scraped, J1 moves flat. Under pooled-2024-vs-scraped, J1 moves up. Under aggregator exclusion, it moves the same way. Under specialist exclusion, the J3 half of the story shrinks 67%. The pre-exploration prior that "the junior direction will be clear and the only question is calibration" was wrong — the junior direction is *inherently baseline-contingent in this data*, and the paper has to say so.

**Tech-giant intern pipelines, not aggregators, are the invisible entry-specialist class.** Of T06's 240 flagged entry-specialist companies, only 12 (5%) are actual aggregators. Eleven tech-giant employers (Google, Amazon, AWS, Walmart, Microsoft, TikTok, Intel, NVIDIA, Apple, IBM, Qualcomm) carry 2,486 of the 5,144 flagged-specialist postings (48%). Walmart in scraped has mean YOE 2.2, J3 share 73.8%, and 0% J1 entry — it is effectively running a new-grad pipeline that no seniority label surfaces. Google J3 = 52%, J1 = 0.9%. **TikTok alone has 52% J1 entry** — it is the one tech giant where the labels genuinely say "entry."

**"Developer" lost 61 pp of native entry share** between arshkon and scraped as a single platform-taxonomy move — not a content change. LinkedIn has shifted how it labels the bare "developer" title. Combined with the industry taxonomy drift ("IT Services and IT Consulting" 2024 → "Software Development" 2026, a +17pp swap that is mostly relabeling), this points to **LinkedIn platform evolution as a first-order confound**. The Indeed cross-validation rescues us for entry-share *levels*, but not for native-label *stability* across periods.

**The senior-side within-2024 gap (−19pp arshkon vs asaniczka on S1) is larger than the cross-period effect under some baselines.** Asaniczka SWE is 63.5% mid-senior+director; arshkon SWE is 44.3%. Scraped is 43.3%. Pooled-2024-vs-scraped says the senior share fell 16pp; arshkon-only-vs-scraped says it fell 1pp. Same data, different stories, driven entirely by the pooling choice. The senior-side RQ1 claim has to lead with arshkon-only and report pooled as a sensitivity, because asaniczka's native-label distribution is structurally incompatible with arshkon's.

**Stage 4's multi-location dedup is validated.** Only 16 residual within-source duplicate (company × title × hash) triples remain in scraped, max group size 2. The prior-run finding that "~23% of the 2026 entry-labeled pool came from six companies posting the same description 4–25 times" is resolved — the new concentration story is tech-giant intern pipelines, not bulk-template employers.

## Evidence assessment

| Finding | Evidence strength | n | Confounds | Calibration | Sensitivity survival |
|---|---|---|---|---|---|
| AI-mention rise is within-company (+33pp on 125-co overlap) | **Strong** | 125 cos × period | Description length growth (need density check in Wave 2) | Not yet computed within-2024; T06 only did 2024-vs-2026 | Only the overlap panel tested; corpus-level not yet decomposed |
| Junior share direction is baseline-contingent | **Strong (as a negative finding)** | J1–J4 × (arshkon-only vs pooled) | None — the pattern is itself the finding | T05 SNR < 1 across J1/J2/J3 | Survives all four junior variants |
| J3 rise is 95% between-company composition | **Strong** | 125-co overlap panel | Requires arshkon∩scraped overlap; small panel | Not computed within-2024 | Specialist exclusion also supports (67% attenuation) |
| Senior share fell (pooled-2024) | Moderate | 13,595 vs 17,713 | Asaniczka native-label artifact dominates; within-2024 gap −19pp | SNR ~1 (pooled calibration) | Halves under aggregator exclusion (−16pp → −12pp); halves under arshkon-only baseline (−16pp → −1pp) |
| Senior share fell (arshkon-only) | Weak | 2,077 vs 17,713 | Small 2024 n | Arshkon vs asaniczka already discards 19pp of variability | Barely moves (−1pp) |
| Description length grew 64% at median | Strong | 4,691+18,129 vs 40,881 | Instrument change (HTML-stripped Kaggle vs markdown scraped) | Within-2024 length at fixed seniority ~13% gap | Definitional: T13 must section-decompose |
| Scraped companies 74.5% new entrants | Strong | 7,940 scraped companies | None for the composition fact; interpretation caveat | N/A | N/A — definitional |
| `is_swe` flag is defensible | Strong | 1.2M rows | QA/Test borderline (~20–25% FP on borderline); boundary blurring at runtime | Stable across periods | FN ≤ 0.15% |
| Indeed matches LinkedIn on J1/J3 levels | Moderate | 11,860 Indeed rows (rule-only, 54% unknown) | Only tests levels, not label stability over time | Strong tie-break for LinkedIn-artifact hypothesis | Suggests entry-share level is real across platforms |
| Geographic correlation with BLS r=0.96–0.98 | Strong | 50 states | 20 small states zero in scraped by design | N/A | Log-Pearson 0.82, Spearman 0.86 more honest |

## Seniority panel (applied to the two most important findings)

This section is the pre-committed ablation for Wave 1's findings that depend on seniority stratification. Junior-share and entry-share direction claims must clear the T30 panel (loaded from `exploration/artifacts/shared/seniority_definition_panel.csv`). Senior-share claims likewise.

### Junior share 2024 → 2026 (pooled-2024 baseline, primary reading)

| Variant | Pooled 2024 share | Scraped 2026 share | Δ (pp) | n2024 | n2026 | Direction | Notes |
|---|---:|---:|---:|---:|---:|---|---|
| J1 `entry` | 1.66% | 3.12% | +1.46 | 379 | 1,275 | up | Composition-dependent (see arshkon-only below) |
| J2 `entry`+`associate` | 1.83% | 3.25% | +1.41 | 418 | 1,327 | up | Near-identical to J1 — associate is tiny |
| J3 `YOE ≤ 2` | 7.74% | 11.60% | +3.86 | 1,766 | 4,742 | up | 95% between-company in overlap panel |
| J4 `YOE ≤ 3` | 17.68% | 22.77% | +5.08 | 4,035 | 9,307 | up | Between-company dominant |

**Agreement verdict (pooled-2024 baseline): 4 of 4 UP.** Direction unanimous. Magnitude spread is 2.8× (1.4pp vs 5.0pp), which fails the 30% effect-size-spread test of the Gate 0 robustness rule. J3/J4 pick up the YOE-driven composition story; J1/J2 are label-bound.

### Junior share 2024 → 2026 (arshkon-only baseline — parallel reading)

| Variant | Arshkon 2024 | Scraped 2026 | Δ (pp) | n2024 | n2026 | Direction |
|---|---:|---:|---:|---:|---:|---|
| J1 | 3.73% | 3.12% | −0.61 | 175 | 1,275 | down |
| J2 | 4.05% | 3.25% | −0.80 | 190 | 1,327 | down |
| J3 | 9.29% | 11.60% | +2.31 | 436 | 4,742 | up |
| J4 | 19.33% | 22.77% | +3.44 | 907 | 9,307 | up |

**Agreement verdict (arshkon-only baseline): SPLIT — J1/J2 DOWN, J3/J4 UP.** The label-based and YOE-based junior definitions disagree on direction when the baseline is arshkon-only. Mechanism: arshkon has richer native entry labels, so its label-based entry share is artificially high relative to a 2026 scraped window that has 40% `seniority_final = 'unknown'` from LLM-budget rationing. The YOE-based definitions are label-independent and pick up composition instead.

**Under the Gate 0 robustness rule, "junior share rose 2024→2026" is NOT a defensible headline.** Neither pooled-2024 (spread > 30%) nor arshkon-only (split direction) clears the bar.

### Senior share 2024 → 2026

| Variant | Arshkon 2024 | Pooled-2024 | Scraped 2026 | Δ arshkon (pp) | Δ pooled (pp) |
|---|---:|---:|---:|---:|---:|
| S1 `mid-senior`+`director` | 44.28% | 59.57% | 43.33% | −0.95 | −16.24 |
| S2 `director` | 0.28% | 0.46% | 0.72% | +0.44 | +0.26 |
| S3 title-kw senior (raw title) | 38.46% | 56.71% | 42.29% | +3.83 | −14.42 |
| S4 `YOE ≥ 5` | 37.33% | 45.04% | 38.99% | +1.66 | −6.05 |

**Agreement verdict: SPLIT across baselines.** Arshkon-only-vs-scraped shows senior share near-flat or slightly up on S1/S3/S4 (−1 to +4pp). Pooled-2024-vs-scraped shows senior share fell −6 to −16pp. The direction disagreement is *entirely driven by asaniczka's missing native entry labels* (asaniczka is 63.5% S1 vs arshkon's 44.3%). **Under the Gate 0 robustness rule, "senior share fell 2024→2026" is not defensible without specifying the baseline, and the arshkon-only baseline (which is the fairer comparison for native-label content) shows essentially no senior-share change.**

## Narrative evaluation

The initial RQ1–RQ4 framing was:
- RQ1: Employer-side restructuring — junior narrowing + scope inflation + senior redefinition
- RQ2: Task/requirement migration across seniority levels
- RQ3: Employer-requirement / worker-usage divergence
- RQ4: Mechanisms (interview-based)

Gate 1 verdict on each:

**RQ1 — WEAKENED / NEEDS REFRAMING.** The "junior narrowing" component is not defensible from label-based evidence — the direction flips between arshkon-only and pooled-2024 baselines, and the within-2024 SNR is below 1. The YOE-based junior rise (J3 +3.9pp pooled) is 95% between-company compositional (T06 decomposition) — it is *which employers post* more than *what they ask for*. The "senior redefinition" component requires the arshkon-only baseline (because of asaniczka's native-label artifact), and under that baseline the senior share is near-flat — so the story has to move from "senior postings shrank" to "senior content changed" (which is a Wave 2 task, not a Wave 1 conclusion).

**RQ2 — DEFERRED.** Wave 2 (T11, T12, T14) tests requirement migration directly.

**RQ3 — STRENGTHENED.** T06's within-company +33.3pp AI-mention finding is the single cleanest signal in Wave 1. It survives composition, concentration, and calibration in a way no other aggregate does. This is the strongest candidate to become the paper's lead. The *divergence* half of RQ3 still depends on Wave 3's T23 benchmark work.

**RQ4 — UNCHANGED.** Not yet testable quantitatively.

### Alternative framings to weigh

Four framings the evidence could support — I am NOT picking one yet; Wave 2 must decide.

1. **"Employers rewriting JDs toward AI skills is the real restructuring; junior/senior shifts are composition."** Lead: the +33pp within-company AI-mention finding. Supporting: junior direction is baseline-contingent (not real); 75% new entrants + aggregator doubling + tech-giant intern pipelines explain most apparent shifts. Strongest fit to T06 + the cleanest 2024→2026 within-company delta. Changes the paper from a "workforce restructuring" paper to an "employer-side AI template" paper. **Current leading candidate.**

2. **"Platform evolution is the confound, the story is how much signal survives."** Lead: the bare-title "developer" losing 61pp native entry share, the industry taxonomy drift, the description length doubling, the markdown/HTML instrument change. Frames the paper as a dataset/methods contribution about measurement challenges in longitudinal posting studies; empirical findings ride alongside. Best fit if Wave 2 finds that section-decomposed content change is small once boilerplate is removed.

3. **"Tech-giant intern pipelines expanded; the entry market hollowed elsewhere."** Lead: Google/Walmart/Microsoft J3 shares of 40-74%; 240 specialists carry 65% of J3 pool; excluding them attenuates effect by 67%. Reframes RQ1 from "junior compression" to "junior concentration." Risk: depends on whether tech-giant scraping coverage was worse in 2024 Kaggle snapshots (likely yes), which makes this at least partly an instrument story.

4. **"AI-requirement rewriting crosses occupation boundaries."** Wait for Wave 3 T18. If control occupations also show +30pp AI mentions within-company, the SWE-specific framing collapses; if they don't, framing (1) strengthens. Either way, the story has to clear the SWE-specificity bar.

My current preference: framing (1) with framing (3) as supporting. Framing (2) is where the paper goes if Wave 2 blows up the scope-inflation story cleanly. I need Wave 2 evidence before committing.

## Emerging narrative

Draft one-sentence headline as of Gate 1: **"Between 2024 and 2026, the single cleanest within-company shift in US LinkedIn SWE postings is a +33-percentage-point rise in AI-tool and AI-requirement mentions; apparent shifts in junior share, senior share, and description length are largely compositional (different employers posting) or instrument-driven (platform relabeling, description-format change), and cannot be read as a workforce restructuring on their own."**

This is a weaker headline than the initial "AI-era employer-side restructuring" framing — but it is *defensible from Wave 1 evidence* and survives the pre-committed ablations. Wave 2 will test whether this picture holds or strengthens.

## Research question evolution

Proposed changes at Gate 1 (tentative — Wave 2 confirms or revises):

- **RQ1 is split into RQ1a and RQ1b.**
  - **RQ1a — Employer-side AI-requirement rewriting (PROMOTED to lead candidate).** Between 2024 and 2026, do the same companies rewrite their SWE postings to ask for AI-tool and AI-domain skills? *Measured as: within-company change in AI-mention prevalence on the 125-company arshkon∩scraped overlap panel, length-normalized, under both broad and strict AI keyword patterns, with within-2024 calibration.*
  - **RQ1b — Market composition shift (REFRAMED from "junior narrowing").** Between 2024 and 2026, did the composition of LinkedIn SWE posters shift toward aggregators, tech-giant intern pipelines, and new entrants? *Measured as: aggregator share, new-entrant share, entry-specialist share, and between-company decomposition of J3/J4.*
  - Old RQ1 "senior redefinition" persists as a Wave 2 content question, not a share question.
- **RQ2 is unchanged, but its primary lens is now within-company.** T16 (Wave 3) is the decisive test — same companies, different asks.
- **RQ3 is promoted toward a possible lead.** If AI-mention shift passes Wave 2 and Wave 3 tests, it becomes the paper's organizing empirical finding.
- **RQ4 is unchanged.**

A potential new RQ1c emerging from the evidence: "How much of apparent 2024→2026 SWE posting change is platform evolution (LinkedIn relabeling, taxonomy drift, format change) rather than labor-market signal?" — worth adding if Wave 2 T13's section decomposition confirms that length growth is boilerplate-driven.

## Gaps and weaknesses

- **We have not yet length-normalized AI-mention rates or computed density.** T06's +36pp is *binary prevalence*; the +33pp within-company rise could weaken if the 64% length growth diluted mentions per 1K chars. Wave 2 must verify.
- **We have not tested whether the same AI-mention rise appears in control occupations (T18).** The SWE-specificity of the strongest Wave 1 signal is untested.
- **The arshkon∩scraped overlap panel (125 companies) is 33% of arshkon rows and 18% of scraped rows.** Small panel. Decomposition results should be read against the larger 589-company pooled-2024∩scraped panel as a robustness.
- **"Graduate"/"new grad" under-detection** depresses scraped J1 by ~2.4pp (~1,000 rows out of 40,881). The official cross-period junior-share numbers slightly understate the 2026 junior signal, which would make J1 pooled-vs-scraped +1.46pp closer to +3pp — but arshkon-vs-scraped would also shift. Preprocessing fix recommended.
- **Tech-giant scraping coverage is probably worse in 2024 Kaggle than in 2026 scraped.** If the Kaggle snapshots systematically missed Amazon/Google/Microsoft intern postings that exist in the scraped window, the "tech-giant intern pipelines expanded" finding is partly an instrument artifact. No immediate way to test this without reverse-engineering Kaggle's scraping methodology; the best we can do is arshkon-only analysis and flagging.
- **Scraped `description_core_llm` coverage is 30.5%** (12,534 labeled SWE rows). Every Wave 2 text-dependent analysis must report within-labeled-frame counts and flag thin cells. Where labeled coverage is not enough, the analysis is restricted to the labeled subset, not backfilled with raw text.
- **Indeed cross-validation is at the end-of-pipeline only.** It matches LinkedIn on entry-share levels but cannot diagnose whether LinkedIn native-label semantics drifted over time. The best "is this platform artifact" test available is the arshkon∩scraped shared-title stability check (T05 §9), and that already shows 14/20 titles lost native entry share between 2024 and 2026 — real platform drift.

## Direction for next wave

### Wave 1.5 (Agent Prep) — dispatch immediately

Before Wave 2, Agent Prep must build the shared artifacts so five Wave 2 agents can pull from one source of truth. Key requirements driven by Wave 1 findings:

- **Cleaned text artifact must filter to `llm_extraction_coverage = 'labeled'`** and record `text_source = 'llm'` for rows that use `description_core_llm` and `text_source = 'raw'` for fallback rows. Report the split by source and period. Wave 2 text-sensitive tasks restrict to `text_source = 'llm'`.
- **Tech matrix must handle scraped markdown escapes** (`C\+\+`, `C\#`, `\.net`) — strip backslash escapes from text before regex match. The sanity-check step 7 of the spec is mandatory; Wave 2 must not launch with a tech matrix that undercounts `c++` in scraped.
- **Calibration table must include AI-mention prevalence as a named metric** so Wave 2 can benchmark the headline finding against other metrics without recomputing arshkon-vs-asaniczka.
- **Company stoplist must be token-level and lowercased**, and cover all `company_name_canonical` values. Downstream corpus comparisons need this before tokenization.
- **Asaniczka `skills_raw` parsing** is the only structured-skills baseline we have; T14 depends on it. Lowercase, split on comma, deduplicate.

### Wave 2 direction

Given Wave 1's evidence, Wave 2 should be reweighted:

- **T09 archetype discovery is now the single most important Wave 2 task.** If the dominant cluster structure is by domain (Frontend / ML/AI / Embedded / Backend / Data) and ML/AI is both growing in share AND structurally less junior-heavy, framing (1) or framing (3) becomes clear. If the dominant structure is by seniority, framing (2) is more likely. Tell Agent F to compute NMI of clusters × (seniority, period, tech-domain, industry) and report which dimension dominates.

- **T13 must do the section decomposition early and hand off to T12.** The 64% length growth must be decomposed before Wave 2 makes any corpus-level claim. If the growth is in benefits/about-company sections, the boilerplate-driven hypothesis (framing 2) is strong; if it is in requirements sections, framing (1) strengthens.

- **T11 scope-inflation metric must be length-normalized AND reported under every T30 panel variant.** Agent G must report tech_count per 1K chars, scope terms per 1K chars, credential stack depth per 1K chars — no raw counts. Primary junior is J2; report J1/J3/J4 as a 4-row ablation for every headline.

- **T08 distribution profiling must test the "AI-mention within-company +33pp" finding on the full corpus with length normalization.** If it holds under density (AI mentions per 1K chars of cleaned text) and within the arshkon-only baseline, it is the paper's lead. If it halves under density, framing (1) weakens.

- **T14 must report AI-mention prevalence under broad, strict, and AI-tool-specific patterns with semantic precision checks.** Broad patterns like `\bai\b` match generic phrases; strict patterns (specific tools: copilot, cursor, claude, gpt, rag, llm, prompt engineering) are the paper's backbone. Sample 50 matches per pattern, stratified by period, and report precision before citing any prevalence.

- **T15 embedding-based similarity should flag the domain-vs-seniority structure explicitly.** If embeddings show domain clusters dominate (Frontend / ML/AI / Embedded) and seniority is a weak axis within each domain, the paper's methodology reframes around domain-stratified analysis.

### Task-spec modifications for Wave 2 prompts

I will write these modifications directly into Wave 2 agent prompts rather than editing the task reference file:

- **All Wave 2 agents must load `exploration/artifacts/shared/entry_specialist_employers.csv` and report headlines with AND without the 240 companies excluded.**
- **All Wave 2 cross-period findings must cite the T07 feasibility verdict: "well-powered at population-share level; binding constraint is within-slice content under J1/J2 arshkon-only and under S2 everywhere."** Agents should not attempt within-director content analysis.
- **All Wave 2 findings must report SNR = |cross-period effect| / |within-2024 effect| alongside the primary number.** If SNR < 2, flag as "not clearly above instrument noise."
- **All Wave 2 junior claims must report J1/J2/J3/J4 as a 4-row table and conclude with an agreement verdict.** Under the Gate 0 pre-commitment, only unanimous or 3-of-4 agreement may be cited as a lead claim. Split/contradictory results require mechanistic discussion.
- **Senior-side findings must lead with arshkon-only and report pooled-2024 as a separate row.** Because of asaniczka's native-label structure, pooling is methodologically costly on the senior side.
- **Text density metrics use `description_core_llm` with `llm_extraction_coverage = 'labeled'`.** Raw `description` is only acceptable for binary keyword presence (recall-oriented) and must report the split.

## Current paper positioning

If we stopped at Gate 1, the strongest paper is a **dataset + methods contribution** that leads with the observation that **2024→2026 "change" findings in job-posting data are dominated by compositional and platform-evolution confounds, and that the one cross-period signal that survives full decomposition is within-company AI-mention rewriting**. This is a credible ICWSM / labor-economics methods venue paper and has a single clean headline finding.

To upgrade this to a **substantive empirical labor paper**, Wave 2 must deliver:
1. AI-mention rewriting confirmed at density (per 1K chars) not just binary prevalence — and confirmed in section-decomposed requirements text, not boilerplate.
2. A defensible content-based senior archetype shift (management → orchestration) within arshkon∩scraped overlap companies — independent of share shifts, which Wave 1 showed are baseline-contingent.
3. Domain-stratified story from T09 that explains which SWE subpopulation is driving the AI-requirement rewrite.
4. Continued SWE-specificity of the AI signal through T18 in Wave 3 — if control occupations also show it, the paper becomes a generic-hiring-template paper, which is less publishable but still honest.

If Wave 2 fails on any of those four tests, the paper positions as the methods contribution with an honest "this is what we can and cannot say" framing. That is still a publishable paper — just a different one.

Gate 1 complete. Dispatching Wave 1.5 (Agent Prep) next.
