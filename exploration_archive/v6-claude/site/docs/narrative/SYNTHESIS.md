# SYNTHESIS — Exploration handoff for the analysis phase

Agent N (Wave 4) · 2026-04-15

This is the single document the analysis agent should read first. It consolidates Wave 1-3 findings (26 task reports, 4 gate memos, 2 verification passes) into (a) the four core findings, (b) data quality and recommended samples, (c) confirmed / contradicted / reframed claims, and (d) what the paper should and should not say. All corrections from V1 (Gate 2) and V2 (Gate 3) are applied throughout.

---

## 1. Executive summary

Between 2024 and 2026, information-tech job postings — SWE and SWE-adjacent — underwent a structural AI tool and framework explosion that restructured the employer-side technology co-occurrence graph, redefined senior roles around hands-on technical orchestration, and reshaped what 74.6% of the overlap-panel companies post as their dominant archetype. The AI explosion is real, 92% within-company, field-wide across information work (not SWE-specific), and survives every sensitivity check we can apply. The senior redefinition is a clean specialization story: mid-senior technical-orchestration language rose +98%, director orchestration +156%, and the tech-lead sub-archetype doubled (7.8% → 16.9%). These two findings are the paper's load-bearing empirical contributions.

The paper's lead RQ3 narrative inverts. The original hypothesis — that employer AI requirements run ahead of worker adoption — is contradicted. Stack Overflow 2025 reports 80.8% of professional developers use AI tools in their work, while the 2026 SWE broad-AI posting rate is 28.6% and the hard-AI-requirement rate (AI mentioned in the requirements section) is just 6.0%. The posting-language gap is negative by 52 pp on broad AI and ~75 pp on hard requirements. Posting language is a lagging indicator of workplace AI adoption, not a leading one. This contradicts both the popular "employers demand impossible AI skills" framing and the anticipatory-restructuring framing in the original research design.

Three things originally framed as findings do not survive. The aggregate junior-narrowing story fails calibration, macro-robustness, and within-company decomposition (the cross-period entry-share effect is literally smaller than within-scraped-window drift — T19 ratio 0.86×). The seniority-convergence hypothesis is contradicted: three of four boundaries SHARPENED on a structured classifier (T20), with the director-level blur explained by directors being recast as hands-on tech leads. The length-growth-as-scope-inflation reading collapses under T29's style-matched analysis — length growth is mostly recruiter-LLM drafting style migration, not content expansion. What survives RQ1 is a narrow within-archetype credential-stack convergence: entry and mid-senior credential breadth gaps close in all 10 large archetypes and flip sign in 2-7 archetypes depending on pattern definition.

---

## 2. The four core findings

Each is reported with evidence strength, the specific numbers, the sensitivity survival state, and the task-report citations.

### Finding 1 — Posting language LAGS worker AI adoption by roughly an order of magnitude (RQ3 inverted)

This is the paper's lead empirical claim. The direction is the opposite of the original RQ3 hypothesis.

**Numbers (2026 SWE LinkedIn, default filters):**

| Measure | 2026 posting | Worker rate | Gap | Ratio |
|---|---|---|---|---|
| Broad AI (24-term union) | **28.6%** | 80.8% (SO 2025 pro any-use) | **−52.2 pp** | 0.35 |
| Narrow AI (LLM-labeled subset) | 34.6% | 80.8% | −46.2 pp | 0.43 |
| AI-as-tool (copilot/cursor/claude/chatgpt) | 6.9% | 80.8% | −73.9 pp | 0.08 |
| **Hard AI requirement** (AI × requirements section) | **6.0%** | 80.8% | **−74.8 pp** | 0.07 |

**Sensitivity / robustness:**
- Direction holds across the 50-85% worker-rate sensitivity range (V2 re-verified).
- Posting rates reproduce exactly under independent V2 re-derivation (within 1 pp on every metric).
- Per-tool figures are approximate (SO 2025 publishes per-tool shares only among the "AI agent user" subset; T23 multiplied by 80.8% any-use to get unconditional rates). **Cite the broad AI ratio 0.35 as the lead number**; soften per-tool claims to "~10-15× worker-to-posting ratio for the major AI coding tools." Do NOT cite the 108× ChatGPT ratio (denominator near zero, unstable). (Gate 3 correction 3.)
- Temporal: posting-side broad AI grew 5.6× (5.13% → 28.6%), worker-side grew 1.28× (63.2% → 80.8%). Posting-side is growing faster in ratio, but from a much lower base and still well below.
- Calibration: T14 SNR 13.3 on broad union, T05 SNR 925 on narrow pattern. Both above threshold. (V1 correction 1: cite the narrow and broad cells separately — do NOT combine.)
- T19 macro-robustness ratio 24.7× (cross-period / within-scraped-window) → well above noise floor, JOLTS cooling not confounding.

**Section location of AI in postings (T22/T23):** Only 21-24% of AI mentions land in the `requirements` section vs 34-39% for non-AI tech (lift −13 to −16 pp). AI lives in `responsibilities` + `role_summary`, and its share in `preferred` sections TRIPLED (2.4% → 7.9%). Employers describe AI as *what you'll do* far more than as *what you must have*.

**Known reviewer attack surface:** Stack Overflow self-selection bias. Mitigation: sensitivity range 50-85% and cross-checks against Anthropic, Accenture, GitClear, SO blog — all put pro-dev AI any-use in the 75-84% range. Direction holds at the lowest plausible worker rate (50%).

**Evidence strength: Strong.** Citations: T23 (headline), T14 (SNR), T19 (macro ratio), T22 (section test), V2 section 2 (re-derivation).

### Finding 2 — Senior roles specialized toward hands-on technical orchestration, not people management

The cleanest within-corpus direct evidence of role redefinition in the whole exploration, and it is AI-linked.

**Validated density (per 1K chars, T21 rebuilt from scratch with object-noun-phrase patterns, 50-row stratified precision check per pattern, bundle at `exploration/artifacts/shared/validated_mgmt_patterns.json`):**

| Profile | Mid-senior 2024 → 2026 | Director 2024 → 2026 |
|---|---|---|
| People management (narrow, validated) | 0.186 → **0.232 (+25%)** | 0.228 → 0.181 (**−21%**) |
| Technical orchestration | 0.168 → **0.332 (+98%)** | 0.118 → **0.302 (+156%)** |
| Strategic scope | 0.045 → 0.053 (+17%) | 0.035 → 0.076 (+117%) |

**AI × senior interaction (T21):**
- 2024: AI-mentioning senior postings had identical profile density to non-AI.
- 2026: AI-mentioning senior postings have **orchestration density 0.482 vs 0.274 non-AI** (+76% uplift — V2 reproduces at +73%). People-management density is identical (0.230 vs 0.232, −1%).
- **So AI-mentioning 2026 senior postings are tech-leads / orchestrators, not people managers.**

**Sub-archetype clustering (k-means on z-scored densities):**
- Tech-lead cluster: **7.8% → 16.9% (+9.1 pp)** — more than doubled.
- People-manager cluster: 14.5% → 14.7% (flat).
- Generic bucket: 70.2% → 59.8% (−10.4 pp).

**Cross-seniority pattern rules out a uniform-template explanation:**
- Entry people-management density: **−13%** (0.064 → 0.056).
- Mid-senior: **+25%**.
- Director: **−21%**.

Non-monotonic, so if LLM drafting tools were inserting mentoring language uniformly at every level, entry would have risen. It didn't.

**Director recasting — largest single feature-importance swing (T20):**
- Mid-senior → director `tech_count` coefficient flipped sign: **−0.48 → +0.35**. 2024 directors had fewer tech mentions (people bosses); 2026 directors have more (tech orchestrators).
- Director `tech_count` rose 2.93 → 8.03 (+173%), the largest per-cell shift in the T20 heatmap.
- Mid-senior↔director AUC FELL 0.677 → 0.616 (−0.061). Directors became structurally closer to mid-senior — because directors shed people-management and gained orchestration.

**Robustness:**
- Validated patterns precision-audited at **100% on a 100-row V2 audit** (50 per period). Bulletproof.
- T21 primary numbers reproduce to 4 decimal places (V2 sections 7 and 3).
- Director +156% attenuates to **+120%** after excluding the top 10 AI-heavy director companies — direction and magnitude hold (V2 Alt 2).
- Mentor sub-pattern is style-correlated (T29), so some fraction of the mid-senior people-management rise is template. The orchestration rise is NOT style-correlated (AI 0-7% attenuation).

**Senior-title compression (T10/T21):** `senior` in raw titles 41.7% → 28.9% (−12.8 pp); `staff` rose only +3.1 pp. Within 150-company overlap panel: mean senior_delta −9.1 pp, staff_delta +1.6 pp. Staff absorbs only ~22% of the senior drop. Classifier is stable (100% of senior- and staff-titled rows are Stage 5 rule matches; 99.5% classify as `seniority_final='mid-senior'` in both periods). Most-supported mechanism: employer title-field template rewriting, likely under LLM drafting (T29).

**Evidence strength: Strong** (co-headline). Citations: T21 (primary), T20 (boundary/feature), T29 (style correlation caveat on mentor), V2 section 7 (precision audit), V2 Alt 2 (top-10 exclusion).

### Finding 3 — Tool-stack restructuring is field-wide, 92% within-company, and bifurcated (existing-employer tool-stack adopters + new-entrant LLM/GenAI wave)

**Field-wide, not SWE-specific (T18 DiD, pooled-2024 vs 2026-scraped):**
- SWE vs control broad AI DiD: **+29.6 pp, 95% CI [28.9, 30.4]** (V2: +29.73, CI [28.99, 30.46]).
- Adjacent vs control: +27.5 pp.
- **SWE vs adjacent: +2.08 pp, CI [0.77, 3.38]** — does NOT cross zero but is modest; SWE and adjacent moved together against control. (V2 Alt 5.)
- SWE↔adjacent TF-IDF cosine did NOT blur (0.862 → 0.808, slight sharpening).
- Cleanest single spillover case: **network_engineer 4.2% → 16.6% broad AI with zero SWE reclassification.**

**92% within-company (T16 overlap panel, n=240, ≥3 SWE per period):**
- Panel AI any rose 4.02% → 26.93% (+22.91 pp), within-company component +21.03 pp (**92%**, V2 reproduces at 89.7% under a slightly different weighting scheme).
- Among 186 panel companies with zero AI in 2024, within-company AI rose +0.2293 by 2026 (drove almost all of the panel rise). Alternative "driven by pre-existing AI-forward companies" ruled out. (V2 Alt 3.)
- Above macro noise: T19 cross-period / within-scraped-window ratio **24.7×** on broad AI. Macro cooling (JOLTS info openings −29%) is not confounding.
- Above style migration: T29 authorship-style matched delta attenuates AI broad only **0-7%** under any matching specification (V1 composite 2%, V2 simplified composite 7%). The AI explosion is real content, not recruiter-tool artifact.

**Tool-stack adopter cluster (T16, k-means on panel change vectors):**
- **46 of 240 overlap-panel companies (19%)** cluster as "tool-stack adopters": ΔAI any **+0.523**, Δdesc length +1,149 chars, Δscope +0.328, entry share flat under both operationalizations.
- Cluster reproduces across seeds at n=50 with ΔAI +0.510 in V2 re-derivation.
- Composition: consulting / system-integrator / enterprise software employers dominate (AT&T, Deloitte, American Express, Aditi Consulting, Aveva, Adobe, Macquarie Group). **Adobe is the most tech-native member. No FAANG.** The companies rewriting their templates toward AI vocabulary are the downstream adopters that need to signal AI capability, not the ones building AI systems.

**LLM/GenAI new-entrant wave (T28):**
- 2024 LLM/GenAI cluster: 616 companies. 2026 cluster: 1,174 companies.
- **68.2% of 2026 LLM/GenAI volume comes from new-in-2026 companies.** Only 138 companies overlap.
- Top 2026 LLM/GenAI employers: Anthropic, Microsoft AI, Intel, Alignerr, Harvey, LinkedIn, Intuit, Cognizant.
- LLM/GenAI also shows a widening junior/senior gap on requirement_breadth (+1.4) and tech_count (+2.0) — the one archetype where juniors and seniors are pulling APART on content breadth, not converging.

**Archetype pivot (T16):**
- **74.6% of overlap-panel companies changed their dominant archetype between 2024 and 2026** (V2: 71.7% on a slightly different denominator — within 3 pp).
- Holds at 73.2% when restricted to ≥5 labeled rows per period.
- Median total-variation distance across period archetype distributions: 0.629. Companies reconfigured *what kind* of SWE roles they post at high rates.

**AI vocabulary spread into non-AI archetypes (T28):**
- JS frontend broad AI +14.7 pp, .NET +13.6, Java enterprise +11.6, DevOps +10.2, Data engineering +9.2.
- Defense/cleared and Embedded lag at +1.8 and +2.7 (clearance / domain barriers).

**Evidence strength: Strong.** Citations: T14, T16, T17, T18, T19, T28, T29 attenuation, V2 sections 3-4 and Alt 3.

### Finding 4 — Aggregate junior-narrowing does not survive; a narrow within-archetype credential-stack convergence does

The original RQ1 junior-narrowing framing is contradicted. The aggregate entry-share "rise" fails calibration, macro robustness, within-company decomposition, and is smaller than within-scraped-window drift. What survives is a specific within-archetype content finding.

**The aggregate junior-share story fails:**
- Within-2024 SNR on `seniority_final` entry share: **0.33** (T05, far below 2.0 threshold). Not safe to pool 2024.
- Arshkon-only baseline: entry share FLIPS direction 7.72% → 6.70% (−1.0 pp). Excluding 7 entry-specialist intermediaries: −2.1 pp. (T08.)
- Within-company overlap panel: direction depends on operationalization. `seniority_final` within −0.032, YOE ≤ 2 within +0.015 (T06/T16). The two metrics have <10% row overlap — they measure different populations.
- **T19 macro-robustness ratio for entry share: 0.86×** — the cross-period effect is literally smaller than within-scraped-window drift. Below the noise floor.
- Denominator drift: 53% of scraped SWE is `seniority_final = 'unknown'` (Stage 10 budget-capped). The "of known" comparison is structurally biased.
- JOLTS info-sector openings dropped 29% between windows — macro cooling is the dominant backdrop.

**What DOES survive — within-archetype credential-stack convergence (T28):**
- Credential-stack gap between entry and mid-senior **converges in all 10 large archetypes** under T28's definition.
- Under an independently defined 6-category credential stack (V2 re-derivation) 10/10 converge at similar magnitudes.
- **Gap flips sign in 2-7 archetypes depending on credential pattern definition** (T28: 7/10; V2: 2/10). **Do not cite "7/10 flip" as a headline.** The robust claim is convergence in all 10. (Gate 3 correction 2.)
- Some style attenuation expected (T29 credential_stack correlation r ≈ 0.09 with authorship score, low), but the credential stack delta STRENGTHENS under T29's matching rather than attenuating.

**Within-archetype entry rise under the label-independent YOE proxy (T28):**
- 16/16 large archetypes rise under `seniority_final`.
- 15/16 rise under YOE ≤ 2 of all.
- 13/13 rise under YOE ≤ 2 of known.
- This weakens the pure-instrument reading of Alt A (Stage 10 LLM label drift alone cannot produce the uniform rise under a classifier-independent proxy) but does not save the aggregate junior-narrowing claim because the rise is still within the within-scraped-window noise floor on T19.

**Evidence strength: Strong (null); Strong on convergence direction, Moderate on the specific flip count.** Citations: T05, T08, T19, T28 (primary), V2 section 6 (convergence verification), Gate 3 correction 2.

---

## 3. Data quality verdict per research question

| RQ (reframed) | Safe analyses | Needs caveat | Unsafe |
|---|---|---|---|
| **RQ3 (lead) — posting-vs-worker AI divergence** | Broad AI rate by period; section-stratified hard-AI rate; temporal posting-side rate; within-2024 calibration; worker-rate sensitivity band | Per-tool unconditional worker rates (extrapolation), SO self-selection bias | Citing 108× ChatGPT ratio; citing a single aggregate AI rate without the narrow/broad split |
| **RQ1/RQ2 (reframed) — senior orchestration + within-archetype credential convergence** | T21 validated patterns on mid-senior and director; AI × senior interaction; tech-lead sub-archetype share; T28 within-archetype credential-stack gap direction; senior-title compression under within-company analysis | Director cells small (99/112); mentor sub-pattern style-correlated (T29); credential-stack flip count definition-dependent | Aggregate junior-share rise as a headline; seniority convergence as a hypothesis; general management rose; length-as-scope-inflation |
| **RQ3 co-lead — tool-stack restructuring, 92% within-company** | T16 overlap-panel decomposition; T18 DiD; T14 modularity rise; T19 macro ratios; T29 style attenuation; archetype pivot rate | Tool-stack adopter cluster n=46 is small; T16 pivot rate coverage-limited on scraped side (30.5% archetype coverage); SWE-vs-adjacent CI is modest | ML/AI-eats-frontend composition; AI SWE-specific framing |
| **Methodological contributions** | Within-2024 calibration (T05, T14); specification dependence (`seniority_final` ↔ YOE proxy, T06/T16); macro robustness (T19); style-matching diagnostic (T29); concentration prediction (T06); authorship composite sensitivity (V2) | T29 length-flip is feature-set dependent; authorship score correlates with `char_len` r=0.59 | Single-operationalization headline entry-share claims |

---

## 4. Recommended analytical samples

| Analysis type | Frame | Filters | Notes |
|---|---|---|---|
| AI prevalence (broad, binary) | `swe_tech_matrix.parquet`, 63,701 rows | default SWE LinkedIn filters, binary on 24-term union | Boilerplate-insensitive; no text coverage cap |
| AI prevalence (narrow, density, section-stratified) | `unified.parquet` + `description_core_llm` | default filters + `llm_extraction_coverage='labeled'` | 2026 side capped at ~12,500 rows |
| Senior density profiles | LLM-labeled senior rows | `seniority_final IN ('mid-senior','director')` + `llm_extraction_coverage='labeled'` | Use `validated_mgmt_patterns.json` (T21 bundle) |
| Within-company decomposition | Overlap panel, ≥3 SWE per period | pooled 2024 vs scraped 2026 | n=240 (≥3) or 125 (≥5); company-level metric |
| Cross-occupation DiD | `is_swe` / `is_swe_adjacent` / `is_control` groups | default filters, binary keyword on raw description | SWE-vs-control is the primary contrast |
| Within-archetype analysis | T09 archetype labels (`swe_archetype_labels.parquet`) | default filters; `archetype >= 0` | **2026 coverage 30.5% — binding constraint** |
| Per-metro | 18 metros at ≥50 SWE per period | non-remote, non-multi-location | n under-powers some Spearman/Pearson tests |
| Authorship style | T29 labeled subset | `llm_extraction_coverage='labeled'` | Bullet density is instrument-confounded; report cleaned and raw |
| Per-tool divergence | Full tech matrix | 2026 scraped broad filter | Cite broad AI (0.35 ratio) as lead; soften per-tool |

**Default SWE frame everywhere:**
```
WHERE is_swe = TRUE
  AND source_platform = 'linkedin'
  AND is_english = TRUE
  AND date_flag = 'ok'
```

**For any seniority-stratified finding, report three operationalizations side by side:**
1. `seniority_final` (primary)
2. YOE ≤ 2 proxy (label-independent validator)
3. `seniority_native` arshkon-only (diagnostic only — asaniczka has no native entry labels; never pool)

---

## 5. Seniority validation summary

**Use `seniority_final` as primary, ALWAYS paired with the YOE ≤ 2 proxy. Never rely on `seniority_native` pooled across sources.**

Key facts:
- `seniority_final` ↔ YOE ≤ 2 proxy have **<10% row overlap** — they measure structurally different populations (T06).
- On the 115-company overlap panel, the two operationalizations **disagree in direction** on within-company entry-share change: `seniority_final` says −3.2 pp, YOE ≤ 2 says +1.1 pp (T06). Same finding replicates at n=240 panel (T16): −0.032 vs +0.015.
- Arshkon native `entry` rows have mean YOE 4.18 and **42.6% at YOE ≥ 5** — the 2024 native-label pool is majority non-entry by any reasonable YOE definition (T08 correction of Gate 1's 26%).
- 84% of arshkon native `entry` rows are landed as `seniority_final = 'unknown'` by Stage 10 — the LLM is correctly refusing to confirm them as entry (T03).
- Asaniczka has ZERO native entry labels — `seniority_native` is structurally unable to detect entry in asaniczka.
- Asaniczka `associate` is NOT a junior proxy (88% lands in mid-senior under `seniority_final`) (T02).
- 53% of scraped SWE is `seniority_final = 'unknown'` because Stage 10 LLM budget was capped. "Of known" denominators drift: 61% → 47% between periods, producing denominator bias for any "of known" comparison (T08).

**Practical implication:** every aggregate junior-share headline in the literature based on a single seniority operationalization is a specification-dependent claim. The paper should include a specification-dependence framework as a methodological contribution (RQ5) and require all entry-share findings to survive both `seniority_final` and YOE ≤ 2 operationalizations. Failure of the `seniority_final` entry-share rise under the YOE proxy is itself a finding.

---

## 6. Known confounders (with severity)

| Confounder | Severity | What it affects | Mitigation |
|---|---|---|---|
| **Length growth is mostly style migration** | **High** | Any raw length / density metric comparing 2024 to 2026 | T29 style matching, T13 section-anatomy decomposition; report attenuation |
| **Kaggle unformatted vs scraped markdown (instrument)** | High | Bullet density, em-dash, paragraph structure | Raw-text sensitivity (T29 halves authorship shift to +0.07); use raw for binary presence only |
| **Asaniczka zero native entry labels** | High | Any pooled-2024 entry metric using `seniority_native` | Arshkon-only baseline for native; pool only on `seniority_final` + YOE proxy |
| **Aggregator + entry-specialist intermediary contamination** | Moderate | Entry-share aggregate, company concentration, top-term frequency | `is_aggregator` exclusion sensitivity; add a companion "entry-specialist intermediary" flag in preprocessing (see action items) |
| **Company composition shift (new-entrant wave)** | Moderate | Between- vs within-company decomposition; LLM/GenAI archetype | Kitagawa decomposition on overlap panel (T16); explicit new-entrant bucket |
| **SWE-vs-field-wide (T18)** | Low | SWE-specificity framing | Cite T18 DiD; reframe paper from SWE to information-tech |
| **JOLTS macro cooling (−29% info-sector openings)** | Moderate | Any volume / share metric; entry share | T19 macro-robustness ratio (cross-period / within-scraped-window); ratio ≥ 10 required |
| **LLM budget coverage gap (scraped 30.7% labeled)** | Moderate | All text-based analyses on 2026 side | Cap at ~12,500 2026 rows; raise Stage 9 target going forward |
| **T09 archetype 30.5% scraped coverage** | Moderate | All within-archetype 2026-side claims | Flag explicitly; consider re-running T09 after coverage raise |
| **Stack Overflow self-selection** | Low (bounded) | RQ3 worker benchmark | Sensitivity 50-85% range; direction holds at floor |
| **Director cells thin** (99 / 112) | Moderate | Director-specific claims | Report CIs; use T21 + T20 convergent evidence |
| **T28 credential-stack pattern dependence** | Moderate | Entry-vs-mid-senior flip count | Cite convergence direction only (10/10); do not cite 7/10 flip |
| **Markdown-escape bug in scraped text** | Low (to fix) | Under-counts `c++`, `c#`, `.NET` in 2026 | Preprocessing fix pending (see action items) |

---

## 7. Discovery findings organized

### Confirmed (survived every check)

- **AI tool/framework explosion** — SNR 925 (narrow)/13.3 (broad); 92% within-company; 24.7× macro ratio; 0-7% style attenuation. Structural: tech network modularity 0.56 → 0.66, two new first-class communities (LLM/RAG, AI-tools triad) (T14).
- **Senior technical orchestration shift** — +98% mid-senior, +156% director; 100% pattern precision on a 100-row audit; holds at +120% after excluding top-10 AI companies; AI × senior interaction localized to orchestration, not people management (T21).
- **AI explosion is field-wide, not SWE-specific** — T18 DiD SWE-vs-adjacent +2.1 pp [0.77, 3.38]; network engineer 4.2% → 16.6% with zero reclassification.
- **Length grew in responsibilities + role_summary, NOT requirements** — 52% responsibilities, 37% role_summary, 29% preferred, <1% benefits/legal, FLAT requirements (T13).
- **Within-company archetype pivot is pervasive** — 74.6% of overlap-panel companies (T16).
- **Entry-posting is a specialist activity** — 79% of companies with ≥5 SWE post zero entry in each source; a small number of intermediaries drive the entry pool (T06/T08).
- **Tech network modularity ROSE** — against the intuitive "blurring" prior, 2026 has more modular tech bundles with new first-class AI communities (T14).
- **BLS geographic validation** r=0.97 (T07).

### Contradicted (cannot appear in the paper)

- "Junior share declined" (RQ1 aggregate) — macro ratio 0.86×, below noise floor.
- "Seniority levels blurred / converged" — 3/4 boundaries sharpened (T20); T15 null not divergence.
- "ML/AI eats frontend" composition mechanism — within-domain dominates between-domain 85-113% (T28); frontend did not shrink.
- "Employers demand AI faster than workers adopt it" (original RQ3) — direction inverted.
- "Length growth reflects scope inflation" — T29 attenuation 23-62% on content metrics; length growth is mostly style migration.
- "Management language rose generally" — T11 aggregation overturned (V1 precision fail). The narrow surviving claim is "narrow people-management rose at mid-senior specifically" (T21).
- "ML/AI is entry-poor" (Alt B premise) — T09 junior share in ML/AI comparable to other archetypes.
- "Staff is the new senior" — only ~22% of the senior-title drop is absorbed by staff rises within companies (T21).

### Reframed

- **RQ3 → posting lag:** employer-side AI naming lags worker adoption by ~order of magnitude. Worker rate 80.8%, posting rate 28.6% (broad), hard-requirement rate 6.0%.
- **RQ1 → senior orchestration specialization:** not a general senior archetype shift; a specific rise in technical-orchestration language at mid-senior and especially at director level, with the director cell losing people-management and gaining tech orchestration.
- **RQ1 → narrow within-archetype credential convergence:** the surviving RQ1b claim is that entry postings catch up to mid-senior postings on credential-stack breadth within the same tech domain. Direction robust in 10/10 archetypes; flip-count definition-dependent.
- **RQ2 → senior archetype shift by archetype:** the senior redefinition is largely localized to the AI × senior interaction cell (T21) and widens specifically in the LLM/GenAI archetype (T28), rather than being a corpus-wide task-migration.

### New discoveries

1. **74.6% archetype pivot rate** in 2 years on the overlap panel (T16).
2. **LLM/GenAI is 68% new entrants**, not existing-employer pivot (T28).
3. **Tool-stack adopter cluster is consulting/SI/enterprise, not FAANG** — Adobe most tech-native, cluster dominated by AT&T/Deloitte/Amex/Aditi (T16).
4. **Boundary sharpening at 3/4 levels**; director-level blur driven by `tech_count` coefficient flipping sign (T20).
5. **Tech-lead sub-archetype more than doubled** (7.8% → 16.9%); people-manager sub-archetype flat (T21).
6. **AI × senior interaction is entirely in the orchestration profile** — identical people-management density regardless of AI (T21).
7. **Entry-specialist intermediaries are invisible to `is_aggregator`** — SynergisticIT, WayUp, Leidos, Emonics etc. drive ~15-20% of the entry pool with 87% aggregator-flagged rate only in the narrow T08 set (T08/T06).
8. **Markdown-escape tokenization bug** — `c\+\+`, `c\#`, `\.net` silently dropped in scraped text (T12/T14).
9. **2026 postings more internally homogeneous** — template homogenization, not role blurring (T15).
10. **AI vocabulary spread into non-AI archetypes** — JS frontend +14.7 pp, .NET +13.6, Java +11.6 (T28).
11. **Network engineer AI prevalence quadrupled** (4.2% → 16.6%) with zero SWE reclassification — cleanest single spillover case (T18).
12. **Austin's JS frontend share collapsed 25.3% → 9.1%** while LLM/GenAI rose 3.2% → 15.9% in the same metro (T17).
13. **T29 authorship-style shift**: 88.7% of 2026 postings score above the 2024 median; 3.9% fall below 2024 p25 (T29).
14. **Section anatomy**: AI mentions land 21-24% in `requirements` vs 34-39% for non-AI tech (T22).

---

## 8. Posting archetype summary (T09 + T16 + T28)

**Structure:** BERTopic on 7,730 SWE rows, company-capped at 50, balanced across three period groups. The SWE posting market's dominant organizing principle is tech domain, not seniority. NMI(cluster, tech_domain_proxy)=0.412 vs NMI(cluster, seniority_3level)=0.015 — tech domain is 27× the seniority signal.

**20 BERTopic topics (4 are employer-template artifacts — Amazon AWS boilerplate, Amazon program boilerplate, generic JS, generic Python).** Usable archetype labels cover 53.5% of SWE LinkedIn rows; 99.9% on arshkon, 94% on asaniczka, **30.7% on scraped** (binding constraint). Labels saved at `exploration/artifacts/shared/swe_archetype_labels.parquet`.

**Archetype mass movement, full corpus (T09):**
- **LLM / GenAI / ML engineering: 4.9% → 20.5% (+15.6 pp)** — largest mover.
- **Defense / cleared: 26.2% → 11.4% (−14.8 pp)** — partly an arshkon-composition artifact.
- DevOps / SRE / platform, JS frontend, Java enterprise, Data engineering, Embedded, .NET, Python backend, iOS, Android, QA, GPU/CUDA (small and declining — the Gate 2 correction).

**Within-archetype findings (T16 + T28):**
- Within-company entry share is NOT uniform across archetypes under `seniority_final`: LLM/GenAI +0.148, Data +0.144, but Java −0.054, Defense −0.027, Agile-generalist −0.185. The T09 uniform rise was between-company.
- Every large archetype rose on requirement breadth and broad AI mention.
- Credential-stack gap converged in 10/10 large archetypes.
- LLM/GenAI is the outlier on requirement-breadth (senior-junior gap WIDENS +1.4) and tech_count (+2.0).
- **74.6% archetype pivot rate on the overlap panel** (T16); 73.2% at ≥5 per period.

---

## 9. Technology evolution summary (T14)

**Tech network modularity ROSE** 0.56 → 0.66 (Louvain on phi-weighted co-occurrence graph, cap-50 per company). 12 → 15 communities. Two new first-class AI communities:
- **LLM/RAG applications cluster (17 techs):** langchain, langgraph, rag, openai_api, claude_api, agents_framework, mcp, fine_tuning, embedding, pytorch, scikit_learn, pandas, numpy, ml, nlp, llm, tensorflow.
- **AI-tools triad:** copilot, cursor_tool, claude_tool.

**Per-tool SNRs (within-2024 calibrated):**

| Tech | 2024 → 2026 | Δ pp | SNR | Notes |
|---|---|---|---|---|
| claude_tool | 0.01% → 3.37% | +3.35 | **326** | |
| agents_framework | 0.61% → 12.70% | +12.08 | 140 | 2024 FP ~10% per V1; true ~+11.4 |
| embedding | 0.15% → 2.82% | +2.67 | 123 | |
| copilot | 0.06% → 3.77% | +3.71 | 44 | |
| langchain | 0.11% → 3.10% | +2.99 | 36 | |
| AI narrow (T05) | 2.81% → 18.78% | +15.97 | **925** | Cite with narrow context |
| AI broad (T14 24-term union) | 5.15% → 28.63% | +23.5 | 13.3 | Cite with broad context |

**IMPORTANT (V1 correction 1):** Gate 2 cross-wired the "SNR 925" figure with the broad 5.15 → 28.63 rates. Those numbers are from different metrics and tasks. ALWAYS cite the two cells separately:
- "Narrow AI keyword rate (T05, LIKE '%ai%'): 2.81% → 18.78%, SNR 925."
- "Broad AI prevalence (T14, 24-term union): 5.15% → 28.63%, SNR 13.3."

**Stack breadth rose median 4.0 → 6.0 across all SWE postings.** AI-mentioning postings are denser in tech per 1K chars (2.20 vs 1.60 non-AI) — AI is ADDED to stacks, not replacing. Asaniczka structured skills vs regex tech matrix Spearman ρ=0.947 — validation.

---

## 10. Geographic heterogeneity summary (T17)

**18 metros qualify at n ≥ 50 SWE per period.** Pooled 2024 (arshkon) → scraped 2026.

**AI surge is broadly uniform.** 18 of 18 metros rose between +0.07 and +0.30 broad AI (mean +0.19). The smallest-surge metro (Detroit +0.07) is still 2× the 2024 baseline. Seattle +0.30, SF Bay +0.28 lead; LA +0.11, Detroit +0.07 lag.

**AI surge ↔ entry-share correlation:** metro-level Pearson r = −0.283 (p = 0.255), Spearman ρ = −0.441 (p = 0.067). Not significant. Consistent with Alt C "AI surge is metro-uniform and uncorrelated with entry-share noise."

**Archetype geography is real:**
- **SF Bay Area LLM/GenAI share 15.7% → 35.6%** (+19.9 pp).
- **Austin JS frontend 25.3% → 9.1%, LLM/GenAI 3.2% → 15.9%.** Sharpest single metro × archetype shift in the dataset.
- Washington DC SCI-cleared share nearly quadrupled (3.2% → 11.8%).
- **Defense archetype is concentrated in LA (index 1.87), Denver (1.75), Detroit (1.70), not DC.** Counterintuitive — DC is SCI-cleared, not defense per se.

**Remote vs metro-assigned pools identical on AI any** (0.273 vs 0.274). Remote is not a selection hiding place for AI surge.

---

## 11. Senior archetype characterization (T21)

See Finding 2. Summary pointers:
- **T21 rebuilt pattern bundle**: `exploration/artifacts/shared/validated_mgmt_patterns.json`.
- 13 people-management patterns (all ≥80% precision, several at 100%).
- 14 technical-orchestration patterns (all kept; `guardrails` dropped at 62%).
- 9 strategic-scope patterns (dropped: bare `stakeholder` 50%, `prioritization` 12%, `roadmap` 70%, `cross_functional_alignment` 70%).
- **The three highest-n strategic candidates all FAILED precision.** Any strategic-language finding that depends on them is a measurement artifact.
- **T22's scope-pattern rebuild** (independent of T21) also drops `ownership_bare` (60%) and `vision` (74%), keeps `ownership_qualified` (100%), `end_to_end`, `cross_functional`, `strategy`, `roadmap`, `drive_impact`, `initiative`, `autonomous` (86%).
- Treat T21's drops of `stakeholder` and `roadmap` as a STRATEGIC-sense ruling; T22's keeps of them are in a SCOPE sense. The profiles measure different semantic targets even though patterns overlap.

**Corrected scope_density** (validated patterns only, per 1K chars): **+64.7% (kept set) / +74.5% (pessimistic after dropping T21-flagged patterns)** vs T11's original +85% claim. (Gate 2 correction 3, confirmed by T22.)

---

## 12. Ghost / aspirational prevalence (T22)

**Aspiration-heavy share of SWE postings doubled:** 13.0% → 25.9%. Aspiration/firm cue ratio rose 0.61 → 1.00. Consistent across seniority, holds under aggregator exclusion.

**Kitchen-sink score** (tech_count × scope_count, kept patterns): 4.39 → 10.59 (**+141%, 2.4×**). Entry 2.95 → 6.12 (2.1×); mid-senior 5.29 → 13.50 (2.6×).

**AI section-location test (T22/T23) — the RQ3 mechanism:**
- **2024: AI mentions 23.6% in `requirements` vs non-AI tech 39.4% (lift −15.8 pp).**
- **2026: AI 21.0% vs non-AI 34.3% (lift −13.3 pp).**
- AI `preferred` section share TRIPLED (2.4% → 7.9%).
- Interpretation: employers describe AI as *what you'll do* far more than as *what you must have*. The RQ3 "hard AI requirement rate" of 6.0% comes from this restriction.

**Credential impossibility is rare (<0.3%) and mostly YOE-extractor artifacts.** Drop as an indicator.

**Aggregators are LESS ghost-like than direct employers at corpus level** (kitchen-sink 8.66 vs 10.99 in 2026). Their signature is high-tech-short-text, not high-scope-high-tech.

**18 of 20 top ghost-entry postings are real AI/ML grad roles at big tech** (Visa, GM, Adobe, TikTok). They are not artifacts.

**Corrected scope_density: +64.7% per 1K chars** (kept patterns) — see section 11.

---

## 13. New hypotheses for the analysis phase (full list in T24.md)

1. **Posting-language update lag is a quantifiable, multi-year phenomenon.** Operationalize as "time elapsed between worker-side adoption crossing X and posting-side prevalence crossing Y." Test on pre/post Copilot release, ChatGPT release, Claude Code release.
2. **Director-level role reconfiguration is the largest single role-redefinition signal in the corpus.** Test with cluster-robust SEs on director-only cells; supplement with LinkedIn-self-title analysis where available.
3. **The tool-stack adopter cluster is motivated by client-signaling, not internal workflow change.** Test by comparing the cluster's description content against their client-facing marketing language; interview probe.
4. **The 74.6% archetype pivot rate reflects within-company portfolio reconfiguration, not posting instability.** Test against 2023/2024 intra-year snapshots if obtainable; alternative: compare pivot rate for same-company pairs in adjacent months vs across the 2-year window.
5. **Within-archetype credential-stack convergence is driven by entry postings stacking AI/tool credentials as a new category.** Test by decomposing the credential-stack gap into category components and showing the AI-category explains most of the convergence.
6. **Entry-specialist intermediaries are a growing structural component of the entry pool.** Test with a dedicated intermediary flag and time-series over longer windows.
7. **Specification dependence of junior metrics is a publishable methodological contribution.** Test by replicating the `seniority_final` ↔ YOE proxy direction flip across independent posting datasets (e.g., Burning Glass, Lightcast, Indeed).
8. **Recruiter-LLM drafting adoption is the mechanism behind length growth, template homogenization, and mentor-language rise.** Test with a corpus-level authorship classifier calibrated on labeled pre-2023 and post-2024 samples; test whether top-LLM-score companies correlate with known recruiter-tool adopters.
9. **The LLM/GenAI archetype is a separate labor market segment, not a continuation of the ML/AI archetype.** Test by comparing the 2024 ML/AI companies vs the 2026 LLM/GenAI companies on all relevant features; 138/1,174 overlap supports this.
10. **Director-level orchestration specialization is AI-driven.** Test via interaction term in a regression of orchestration density on AI mention × seniority, with company fixed effects.

---

## 14. Method recommendations

| Method | Use for | Why |
|---|---|---|
| **Cluster-robust standard errors** (company clusters) | Any posting-level regression | Within-company correlation is non-trivial (92% of AI change is within-company) |
| **Company fixed effects on the overlap panel** | Within-company AI/orch/credential-stack claims | Isolates within-company change; the T16 panel decomposition is the reference |
| **Oaxaca-Blinder decomposition** (symmetric weights) | Aggregate change decomposition | Already done informally in T16/T28; formalize with bootstrap SEs |
| **Kitagawa decomposition** (within × between × interaction) | Entry-share and content metric decomposition | T28 reference |
| **DiD with parallel-trends checks** | Cross-occupation AI spread | T18 reference; direction robust; report CIs |
| **Logistic regression for boundary classification** (stratified CV, L2) | Seniority boundary clarity | T20 reference; 5-fold stratified CV, AUC by boundary |
| **Style-matched nearest-neighbor matching** | Any text-metric longitudinal comparison | T29 reference; AI survives, length does not |
| **Within-2024 calibration** (arshkon vs asaniczka) | SNR for every cross-period metric | T05/T14 reference; ratio ≥ 2 threshold |
| **Macro-robustness ratio** (cross-period / within-scraped-window) | JOLTS-cooling confound check | T19 reference; ratio ≥ 10 threshold |
| **Precision-stratified pattern validation** | Any keyword indicator | 50-row stratified sample per pattern; ≥80% precision threshold (T11/T21/T22) |
| **Concentration prediction table** | Per-finding sensitivity planning | T06 reference |
| **Bootstrap CIs on small cells** | Director, per-metro findings | Director n=99/112, 18 qualifying metros |

**Tools and samples:** Use DuckDB and pyarrow for all queries. Use `description_core_llm` for text-sensitive analyses with `llm_extraction_coverage='labeled'` filter. Use raw `description` for binary keyword presence only. Load `validated_mgmt_patterns.json` for senior-language patterns. Load `swe_tech_matrix.parquet` for boilerplate-insensitive binary keyword presence across the full corpus.

---

## 15. Sensitivity requirements — MANDATORY robustness for paper draft

Every headline finding must have the sensitivity checks listed here in the paper draft or the paper is one reviewer query away from a revision.

| Finding | Must include |
|---|---|
| RQ3 divergence | Worker-rate sensitivity 50-85%; section-stratified hard-AI-rate; narrow vs broad split (not combined); aggregator exclusion |
| Senior orchestration | Pattern precision tables (per-pattern); top-10-company exclusion for director; 50-row audit sample CSV; cross-seniority non-monotonic diagnostic |
| Tool-stack restructuring | Within-company Oaxaca decomposition; cluster stability across random seeds; 92% within-company at ≥3 AND ≥5 thresholds; SWE-vs-adjacent DiD CI |
| Credential-stack convergence | 10/10 convergence under T28 pattern AND V2 6-category pattern; flip-count range [2-7] explicitly stated; style-matched delta |
| Archetype pivot rate | ≥3 and ≥5 thresholds; 30.5% scraped coverage caveat |
| Length null | T29 style matching under composite AND em-dash-only AND bullet-only; cite attenuation (23-62%) not the sign-flip |
| Junior-share null | Within-2024 SNR 0.33; arshkon-only entry flip; macro ratio 0.86×; denominator drift; specification-dependence |

**Do not cite:**
- The specific −411 char style-matched flip (feature-set dependent, V2 narrowing 1).
- The "7/10 sign flip" number (pattern-dependent, V2 narrowing 2).
- The 108× ChatGPT per-tool ratio (denominator near zero, V2 narrowing 3).
- "SNR 925" combined with "5.15 → 28.63" (cross-wired metric, V1 correction 1).
- "Staff is the new senior" (within-company rise of staff absorbs only ~22% of senior drop).
- "Seniority levels blurred" (3/4 boundaries sharpened).

---

## 16. Interview priorities (RQ4 qualitative phase)

Five probes, ranked by yield. Supported by the artifacts in `exploration/artifacts/interview/`.

1. **The RQ3 gap.** "Your company uses AI tools daily but the job posting barely mentions them by name. Why?" Target: hiring managers at tool-stack adopter companies (Adobe, Deloitte, Amex). Expected responses: (a) HR templates lag (b) tool-neutrality deliberate (c) "we expect developers to figure it out."
2. **Senior orchestration recasting.** "Did your director-level role actually become more hands-on technical since 2024, or is that just how you write the posting now?" Target: director+ ICs and their recruiters. Expected responses: (a) real role change (b) template rewriting (c) staff-ladder adoption as a response to flatter senior distinction.
3. **Archetype pivot motivation.** "Between 2024 and 2026, your company changed what kind of SWE you post for — why?" Target: overlap-panel companies with high TVD (Deloitte, AT&T, Amex). Expected responses: (a) client demand shift (b) AI skill-signaling (c) portfolio reorganization.
4. **LLM drafting adoption.** "Are you using ChatGPT or Claude or a recruiter-LLM to draft postings? When did you start?" Target: T29 top-LLM-score 2026 companies (Alignerr, Intuit, LinkedIn, Intel, Harvey). Expected responses: (a) yes, adopted 2024-2025 (b) company policy disallows (c) used for first draft, not final.
5. **Worker-posting mismatch awareness.** "Stack Overflow reports 80% of professional developers use AI tools in their work. What percent of your current SWE postings mention AI tools by name?" Target: anyone at a company with the mismatch. Expected responses: (a) "much lower" (b) "didn't realize the gap" (c) policy reasons.

Supplementary: Austin JS-frontend-collapse interviews; LLM/GenAI new-entrant CEO interviews.

---

## 17. Preprocessing owner action items

1. **Markdown-escape fix (Gate 2 item 9)** — apply `re.sub(r"\\([+\-#.&_()\[\]\{\}!*])", r"\1", text)` in the cleaned-text pipeline. `c\+\+`, `c\#`, `\.net` are silently dropped in 2026 scraped rows. Not a blocker but biases any "legacy language decline" finding.
2. **Entry-specialist intermediary flag** — add a new companion to `is_aggregator` for entry-specialist staffing firms and college-jobsite intermediaries (SynergisticIT, WayUp, Jobs via Dice, Lensa, Emonics, Leidos, IBM). None are caught by the current `is_aggregator` flag; they drive ~15-20% of the 2026 entry pool.
3. **Stage 9 LLM text coverage** — the 30.7% scraped-side `llm_extraction_coverage = 'labeled'` rate is the binding constraint on every text-based analysis. Raise the Stage 9 selection target for the next LLM budget run.
4. **Stage 10 seniority coverage** — 53% of scraped SWE is `seniority_final = 'unknown'`. Raising Stage 10 budget would fix the denominator drift problem that drives the junior-share specification dependence.
5. **Archetype classifier on raw 2026 text** — the 30.5% scraped coverage on T09 labels is a T28-binding constraint. A lightweight classifier that assigns archetype on raw text (not just LLM-cleaned text) would broaden within-archetype analysis to the full 2026 corpus.
6. **Track `seniority_final_source` distribution over time** — T21's finding that 100% of senior-titled rows are Stage 5 rule matches is a useful audit but will not survive a rule version change silently. Pin the rule version to the artifact.
7. **24 cross-company hash collisions** (Gate 1 item) — canonicalizer follow-up for residual aggregator relabeling.

---

## 18. What the paper should and should not claim

### The paper CAN claim (ranked):

1. *"Between 2024 and 2026, information-tech job postings (SWE and SWE-adjacent) underwent a structural AI tool and framework explosion that restructured the employer-side technology co-occurrence graph, while worker-side AI tool adoption — already at ~80% in Stack Overflow 2025 — outpaced employer-side posting language by roughly an order of magnitude."* (Lead finding.)
2. *"Senior SWE roles specialized toward hands-on technical orchestration: mid-senior tech-orchestration language rose 98%, director orchestration rose 156%, and the tech-lead sub-archetype doubled, while director-level people-management density fell 21%. The AI × senior interaction is entirely in the orchestration profile."* (Co-headline.)
3. *"The AI posting-side rise is 92% within-company on a 240-company overlap panel, not driven by new entrants. An independently identifiable 19% tool-stack adopter cluster — dominated by consulting, system-integrator, and enterprise-software companies, not FAANG — pivoted hard on AI vocabulary without moving its seniority mix."*
4. *"74.6% of overlap-panel companies pivoted their dominant posting archetype within two years. The LLM/GenAI archetype's 15.6 pp rise is 68% driven by new entrants, not existing-company pivoting. The AI market is bifurcated between existing-company tool-stack adoption and a new-entrant wave."*
5. *"Across information-tech work, the posting-side AI explosion is NOT SWE-specific. A SWE-vs-adjacent difference-in-differences is only +2.1 pp [0.77, 3.38] compared to +29.6 pp [28.99, 30.46] for SWE-vs-control. Network Engineer AI prevalence quadrupled (4.2% → 16.6%) with zero SWE reclassification."*
6. *"Within each of 10 large archetypes, the entry-vs-mid-senior credential-stack gap converged; in 2-7 archetypes it flipped sign depending on which credential pattern set is used. This is the narrow surviving form of the original RQ1 junior-scope-inflation claim."*
7. *"Seniority boundaries SHARPENED at 3 of 4 levels between 2024 and 2026 on a structured 8-feature classifier. The only boundary that blurred is mid-senior ↔ director, driven by the `tech_count` coefficient flipping sign — 2026 directors mention more technologies than 2026 mid-senior postings, the opposite of the 2024 gradient."*
8. *"A novel robustness framework for longitudinal posting research: within-source calibration (arshkon vs asaniczka), specification dependence diagnostics (`seniority_final` ↔ YOE proxy), macro-robustness ratios (cross-period / within-scraped-window), and authorship-style matching (T29) as a diagnostic for recruiter-LLM drafting confounds. The aggregate junior-share null under this framework is itself a methodological contribution."*

### The paper CANNOT claim:

- "Junior share declined" / "the junior rung is narrowing" (aggregate).
- "Seniority levels blurred" / "roles are converging" (three of four sharpened).
- "Employers demand AI faster than workers adopt it" (direction inverted).
- "Length growth reflects scope inflation" (style migration).
- "General management language rose" (narrow rebuild; task management dominated T11).
- "Staff is the new senior" (rise absorbs only ~22% of senior-title drop).
- "ML/AI eats frontend" (within-domain dominates).
- Any single-operationalization entry-share headline without the specification-dependence disclaimer.
- "Senior postings are gated on AI" (only 6.0% of 2026 SWE postings have a hard AI requirement).
- "108× worker-to-posting ratio for ChatGPT" (denominator near zero).
- "Length flipped direction under style matching to −411 chars" (feature-set dependent).
- "Credential-stack flipped in 7/10 archetypes" (pattern-dependent; cite 10/10 convergence).

### What is still open for Wave 5 / analysis phase

- Whether the director-level recasting is a real market shift or a template-rewriting artifact (thin cells + T29 mentor correlation).
- Whether the per-metro Austin JS-frontend collapse is a single-city shock or a generalizable pattern.
- Whether the LLM/GenAI new-entrant wave sustains or is a cyclical 2025-2026 startup surge.
- Whether the hard-AI-requirement rate (6.0%) shifts when the section classifier is audited specifically for AI-relevant rows (Gate 3 method caveat).

---

## Appendix — cross-references to gate memos and reports

- **Gate 0 (pre-exploration):** `exploration/memos/gate_0_pre_exploration.md`
- **Gate 1 (post-Wave-1):** `exploration/memos/gate_1.md`
- **Gate 2 (post-Wave-2):** `exploration/memos/gate_2.md` + `gate_2_corrections.md`
- **Gate 3 (post-Wave-3):** `exploration/memos/gate_3.md` + `gate_3_corrections.md`
- **V1 verification:** `exploration/reports/V1_verification.md` (5/5 headlines reproduced; 6 corrections applied)
- **V2 verification:** `exploration/reports/V2_verification.md` (8/10 PASS; 2 spec-dependent narrowings applied)
- **Task reports (all 26):** `exploration/reports/T01.md` through `T29.md`
- **Full state table:** `exploration/reports/INDEX.md`
- **Shared artifacts:** `exploration/artifacts/shared/` (cleaned text, embeddings, tech matrix, archetype labels, validated mgmt patterns, calibration table, company stoplist)
- **Interview artifacts:** `exploration/artifacts/interview/` (T25 output — 5 PNG + READMEs)
- **T24 — Hypothesis generation:** `exploration/reports/T24.md`
- **T25 — Interview elicitation artifacts README:** `exploration/reports/T25.md`

End of SYNTHESIS.md.
