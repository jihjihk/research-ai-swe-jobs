# SYNTHESIS — The paper's analytical backbone

**Owner:** Agent N (Wave 4, T26)
**Date:** 2026-04-20
**Primary input:** `exploration/memos/gate_3.md` (unified post-synthesis-input memo)
**Adversarial verifications:** `V1_verification.md` (Gate 2), `V2_verification.md` (Gate 3)
**Task reports consumed:** T01-T07, T30, T08-T15, T16-T23, T28-T29, T31-T38 (34 task reports)

This document is the consolidated analytical backbone for the analysis phase. It is written assuming Agent P (Wave 5) reads only this document to produce the presentation, and the analysis-phase authors read it to specify formal RQs, power analyses, and paper figures.

The Gate 3 memo has done the narrative-decision work. This document faithfully executes that narrative as a single complete integration, with every claim tied to task ID and V1/V2 verification line.

---

## 1. Executive summary (≤400 words)

**Lead claim (combined A1 + A2):** Between 2024 and 2026, software-engineering job postings underwent a real, same-firm, same-title content rewriting concentrated on AI tooling and platform-infrastructure orchestration. The rewriting is SWE-specific in magnitude (DiD SWE−control +13.1 to +14.0 pp on AI-mention, 95% CI [+12.76, +14.37]), within-firm (same-company × same-title pair-level AI drift +10 to +13 pp exceeds company-level +7.7-8.3 pp), and independent of recruiter-LLM authorship style (T29), sampling-frame shifts (T37), and hiring-market selectivity (T38). The employer-worker AI-requirement divergence is universal across 16/16 tested occupation subgroups in both 2024 and 2026 (Spearman +0.92 rank concordance with worker adoption), with SWE-concentrated magnitude but control-occupation 72× ratio gaps (T32).

**Three supporting claims.** Seniority boundaries sharpened, not blurred (T15 TF-IDF 0.946→0.863; T20 AUC +0.150 associate↔mid-senior; T12 label-based and YOE-based relabeling diagnostics agree — 2026 entry ≠ 2024 senior). Scope inflation is universal and senior>junior within-domain (T11 S4 +2.61 > J3 +1.58 length-residualized breadth; T28 within-domain decomposition 60-85%; T33 rejects hidden hiring-bar lowering). Two emergent senior archetypes — Applied-AI/LLM Engineer (15.6× growth, 94% 2026, median YOE 6; T34 cluster 0) and Data-Platform/DevOps Engineer (2.6× growth, +11.4 pp share; T34 cluster 1).

**RQ evolution.** RQ1 reframed: seniority boundaries sharpened; scope broadened universally with senior-tier shifts larger than junior; Applied-AI/LLM Engineer emerged. RQ2 reframed: AI tools appear at junior and senior rates comparably; orchestration concentrates at senior; no downward migration. RQ3 affirmed and extended: universal cross-occupation divergence with SWE-concentrated magnitude. RQ4 unchanged (interview-based), with T29 + T33 + T38 pre-clearing three alternative explanations (LLM-authorship, hiring-bar-lowering, hiring-selectivity).

**Method recommendations.** (i) T30 multi-operationalization seniority panel. (ii) Within-2024 SNR calibration framework. (iii) V1 pattern-validation methodology (exposed mgmt_broad 0.28 precision). (iv) Within-vs-between decomposition + pair-level drift construction.

**Caveats.** Management-decline claim was a measurement artifact; flat under V1-validated pattern (T21 mid-senior 0.039→0.038). T13 junior requirements-chars shrank is classifier-sensitive (demoted). T16/T23 pattern-label mismatch (numbers match top-level ai_strict 0.86 precision; direction robust under v1_rebuilt 0.96). T31 pair-count range-reportable (+10 to +13 pp across constructions).

---

## 2. Paper positioning recommendation

**Recommended positioning (Gate 3):** Hybrid empirical + methods paper with A1 (cross-occupation AI-divergence) and A2 (within-firm same-title rewriting) as co-lead claims, supported by A3 (sharpened seniority) and A5 (Applied-AI archetype). Methods contributions (T30 panel, SNR calibration, pattern validation, within-vs-between) constitute 25-30% of the paper and are independently publishable as methods-venue content.

**Chapter structure:**

1. Introduction framed around employer-worker AI-requirement divergence (T23 + T32).
2. Dataset + methods (T30 panel, SNR calibration, pattern validation, within-vs-between).
3. Empirical section 1 — technology-domain restructuring and Applied-AI Engineer emergence.
4. Empirical section 2 — universal scope broadening with sharpened seniority boundaries.
5. Empirical section 3 — within-firm AI rewriting at same-title level (pair-level exemplars).
6. Robustness — sampling-frame (T37), authorship (T29), selectivity (T38), classifier (T33), panel variants (V1 + V2).
7. Alternative explanations adjudicated (hiring-bar-lowering REJECTED, legacy-to-AI REJECTED, hiring-selectivity REJECTED, LLM-authorship mediation REJECTED).
8. Discussion — what this implies for "AI restructures SWE labor market" claims; what interviews (RQ4) should investigate next.

**Alternative framings considered at Gate 3:**

- **Framing #2 (competing):** substantive labor-market paper focused narrowly on cross-occupation divergence (A1 + A6). Simpler; loses within-firm and archetype findings to appendix. **Not recommended.**
- **Framing #3 (competing):** technology-ecosystem restructuring paper (T09 + T14 + T34 + T35). Less novelty vs prior labor papers. **Not recommended as lead; absorbed into empirical section 1.**

The negative/null findings (H_B/H_H/H_L/H_M/H_N hypothesis rejections, management flat under V1) are a major strength — they rule out leading alternative explanations for the AI rise. SYNTHESIS.md foregrounds these as "we tested and rejected X, Y, Z" rather than hiding them in appendix.

---

## 3. Data quality verdict per RQ

### Data corpus facts

- **68,137 SWE LinkedIn rows** under default filter (`is_swe = TRUE AND source_platform = 'linkedin' AND is_english = TRUE AND date_flag = 'ok'`): arshkon 4,691 + asaniczka 18,129 + scraped 45,317.
- **Three temporal snapshots.** asaniczka 2024-01-12 to 2024-01-17 (6 days); arshkon 2024-04-05 to 2024-04-20 (15 days); scraped 2026-03-20 to 2026-04-18+ (30+ days, growing).
- **Cross-source span** arshkon→scraped midpoint = ~791 days (2.17 years), encompassing ~8 major frontier-model releases (GPT-4o, Claude 3.5 Sonnet, o1, DeepSeek-V3, GPT-4.5, Claude 3.6 Sonnet, Claude 4 Opus, Gemini 2.5 Pro).
- **LLM labeled coverage:** 2024 sources 99.9%; scraped 56.9% (`not_selected` 43.0%, `deferred` 0.02%).

### RQ1 — Employer-side restructuring

- **Safe:** T30 panel J3/S4 YOE-based primaries (12/13 direction-consistency); pooled-2024 baseline + arshkon-only co-primary for senior claims; T15/T20 seniority-boundary sharpening; T11 within-company S4>J3 scope inflation.
- **Needs caveat:** senior share magnitude requires dual reporting (pooled −7.6 pp AND arshkon-only −1.8 pp) per T30 asaniczka asymmetry.
- **Unsafe:** `seniority_native` temporal claims (T05 platform-relabeling drift +15-40 pp on same titles with flat YOE). T11 original mgmt patterns (V1 precision 0.28/0.55 fail).

### RQ2 — Task and requirement migration

- **Safe:** AI-mention universality across YOE (T11 copilot J3 4.6% ≈ S4 4.1%; T20 AI × YOE × period null p=0.66); orchestration concentration at senior (T21 mid-senior orch +0.67/1K chars, SNR 5.6); CI/CD S4 +20.6 pp (T14).
- **Needs caveat:** T13 requirements-section share direction is classifier-sensitive (T33 confirmed aggregate direction flip); cite only as "narrative expansion dominating over requirements contraction" (r=+0.35 req_chars × desc_len).
- **Unsafe:** "requirements migrated downward" (contradicted by T11 and T20); "management migrated downward" (contradicted by T21 V1-rebuilt flat at all tiers).

### RQ3 — Employer-worker AI divergence

- **Safe:** T18 DiD SWE vs control +14.02 pp (V2 robust across 4 alt control defs); T23 employer 1.03%→10.61% / worker 63-90% divergence; T32 16/16 subgroup universality and Spearman +0.92 worker-employer rank concordance.
- **Needs caveat:** benchmark heterogeneity (SO vs DORA vs Anthropic use different "AI use" definitions); T32 reports under 4 bands with direction universally positive across all four; magnitude varies.
- **T16/T23 pattern-label mismatch caveat (V2 D4):** report-text says v1_rebuilt; code uses top-level ai_strict (0.86 precision). Numbers match top-level pattern. Under v1_rebuilt, ratio is 18.6× (0.75%→13.93%) rather than 10.3× — direction robust, magnitude larger under v1_rebuilt.

### RQ4 — Mechanisms (interviews)

- **Pre-cleared alt explanations** (the exploration has computational-quantitative rejections of three alt mechanisms):
  - **LLM-authorship mediation REJECTED** (T29): content effects persist at 80-130% on low-LLM-quartile subset; length growth ~52% LLM-mediated.
  - **Hiring-bar-lowering REJECTED** (T33): |ρ|≤0.28 all hiring-bar proxies; 0/50 loosening language; classifier-sensitive direction.
  - **Hiring-selectivity REJECTED** (T38): |r|<0.11 all content metrics; desc-length r=+0.20 is OPPOSITE direction; volume-UP firms write longer JDs.

---

## 4. Recommended analytical samples

Per analysis type, primary samples and their specifications:

### Sample 1 — Full SWE LinkedIn frame (descriptive + prevalence claims)

- **Rows:** 68,137. Filter: default (`is_swe ∧ source_platform='linkedin' ∧ is_english ∧ date_flag='ok'`).
- **Pattern/column reference:** `exploration/artifacts/shared/validated_mgmt_patterns.json` (7 patterns at ≥0.92 precision).
- **Usage:** aggregate prevalence (T08, T11, T12, T13, T14), within-2024 SNR calibration (`exploration/artifacts/shared/calibration_table.csv`), cross-period raw pp deltas.
- **Caveats:** scraped n=45K is query-stratified by scraper design; within-SWE share claims are safe; raw SWE-share comparisons cross-period are NOT valid.

### Sample 2 — T06 returning-companies cohort (between-vs-within decomposition)

- **Rows:** 13,437 (2024) + 24,927 (2026) = 38,364 rows. **2,109 companies** present in 2024 (arshkon OR asaniczka) AND 2026 (scraped); 55% of 2026 postings; 25% of 2026 unique firms.
- **Artifact:** `exploration/artifacts/shared/returning_companies_cohort.csv`.
- **Usage:** Sampling-frame sensitivity (T37), within-between decomposition (T37 + T16), within-company AI-strict +7.91 pp corroborates T16 arshkon_min5 +8.34 pp and pooled_min5 +7.65 pp.
- **T37 verdict:** 13 of 15 Gate 2/3 headlines retain ratio ≥0.80 on returning cohort; 0 of 15 are sampling-frame-driven. J3/S4 directions INTENSIFY on returning-only (not attenuate). Caveat: H_d J3 breadth_resid retains 0.70 (partially robust).

### Sample 3 — T16 overlap panels (within-company longitudinal)

- **Rows:** 3 panels with different thresholds. `exploration/tables/T16/overlap_panel.csv`, 724 rows × 9 cols.
  - `arshkon_min3`: arshkon n≥3 AND scraped n≥3 → n=243 companies.
  - `arshkon_min5`: arshkon n≥5 AND scraped n≥5 → n=125 companies.
  - `pooled_min5`: pooled (arshkon+asaniczka) n≥5 AND scraped n≥5 → n=356 companies.
- **Within-company AI-strict rewriting:** arshkon_min5 +8.34 pp; pooled_min5 +7.65 pp; T37 returning-cohort +7.91 pp. Gate 3 consolidated range: **+7.7 to +8.3 pp** (V1-rebuilt pattern drops to +7.5-7.8 pp per V2 flag D4).
- **Caveats:** pooled panel is asaniczka-dominated (asaniczka 4× arshkon); asaniczka has LLM-frame selection artifact. Report both arshkon-only and pooled magnitudes (V1 Tension A resolution).

### Sample 4 — T31 same-company × same-title pair panel

- **Rows:** 23 pairs (arshkon_min3, n≥3 per pair-period) / 37 pairs (V2 replicates / V2 strict filter n=12).
- **Pattern:** `ai_strict_v1_rebuilt` (per T31 code, correctly using V1-rebuilt).
- **Usage:** within-firm same-title AI drift. Mean pair-level AI Δ +13.4 pp (T31 primary) / +9.98 to +13.3 pp (V2 replications across filters).
- **Gate 3 citation:** range-report +10 to +13 pp; direction (pair > company-level) robust across all V2 reconstructions.

### Sample 5 — T30 LLM-frame labeled-only subset

- **Rows:** 48,277 under `llm_classification_coverage='labeled'`. YOE-based J3/S4 primaries require this sample.
- **Usage:** per-seniority T11 / T20 / T28 analyses; panel-consistent direction claims.
- **Caveat:** scraped 2026 is 56.9% labeled; 43% not_selected. Text-sensitive analyses must report labeled-vs-not split (Gate 1 pre-commit 4; compliance partial per V2 audit).

### Sample 6 — T09 archetype panel

- **Rows:** 8,000 T09-labeled + 40,223 nearest-centroid-projected + 19,914 unassigned (raw-text scraped). Artifact: `exploration/artifacts/shared/swe_archetype_labels.parquet` + `exploration/tables/T28/T28_corpus_with_archetype.parquet`.
- **Usage:** archetype-stratified analyses (T28, T34, T35).
- **Caveat:** 17% direct label + 83% projected labels. Over-representation ratios are within the covered subset; noise-outlier cluster (46.8% of BERTopic output) is "stack-agnostic senior SWE".

### Sample 7 — T22 + T34 interview exemplar sample

- **Rows:** T22 top-20 ghost entry + T34 cluster-0 20 exemplars = 40 postings (plus extensions).
- **Usage:** RQ4 interview artifacts (T25).

---

## 5. Seniority validation summary

### T03 + T30 T30 multi-operationalization panel

13 seniority definitions × 6 analysis groups = 78-row canonical panel at `exploration/artifacts/shared/seniority_definition_panel.csv`. **12 of 13 direction-consistent** across periods:

- **Junior side:** 7/7 definitions (J1, J2, J3, J4, J5, J6, J3_rule) move UP.
- **Senior side:** 5/6 definitions (S1, S3, S4, S5, S4_rule) move DOWN; S2 (director-only, n<1% of SWE) is flat (noise floor 0.15 pp × 2 ≈ 0.3 pp).
- **Primary (Gate 1 pre-commit):** J3 (YOE LLM ≤ 2) + S4 (YOE LLM ≥ 5), pooled-2024 baseline with arshkon-only co-primary for senior.
- **Primary magnitudes:** J3 +5.04 pp pooled / +1.19 pp arshkon-only (MDE 2.5 pp); S4 −7.59 pp pooled / −1.78 pp arshkon-only.

### T03 seniority_final defensibility

- `seniority_final` kappa vs LLM reference: 0.66 scraped / 0.45 arshkon (arshkon `seniority_native` drifts from LLM reference).
- LLM abstains on 34-53% of SWE rows it is called on (by design — LLM looks for explicit seniority signal).
- Director cell accuracy 22-27% per T03 audit; do NOT use S2 as primary.
- YOE-based J3/S4 are label-independent; preferred for cross-period.

### V1 LLM-frame audit

- Arshkon `seniority_native='entry'` is NOT clean (median YOE=4, 45% ≥5 YOE); T08 flagged.
- Use `seniority_final` for label-based sensitivities; use YOE-based J3/S4 as primary.
- `seniority_native` is UNUSABLE for temporal claims (T05 platform-relabeling drift).

### T20 supervised AUC corroboration

- entry ↔ associate: AUC +0.093; associate ↔ mid-senior: AUC +0.150 (cleanest signal); mid-senior ↔ director: −0.022 (softened).
- Corroborates T15 TF-IDF junior↔senior cosine divergence (0.946 → 0.863; V1 verified exactly).
- Two independent methods agree: **seniority boundaries SHARPENED, not blurred.**

### Wave 3.5 pair-level results under T30 panel

- T34 cluster 0 share robust across S1/S2/S3/S4 panels at 10-14× rise each (V2 Phase C).
- T20 associate↔mid-senior sharpening robust (+0.146 in V2 replication); entry↔associate and mid-senior↔director are sample-thin, direction-only.

**Verdict:** Seniority panel is load-bearing and passes three independent validations (T30 direction consistency, T20 supervised corroboration of T15, V2 panel-robustness audit).

---

## 6. Known confounders with severity + mitigation

### 6.1 Description length growth (moderate severity)

- **Finding:** mean length 3,307 (arshkon) / 3,881 (asaniczka) / 4,893 (scraped) = +~1,250 chars median, +1,750 chars mean. Within-2024 d=0.28, cross-period d=0.50 (SNR 1.80).
- **Mitigation:** length residualization (T11 Phase D) + responsibility/benefits/legal section decomposition (T13). Within-company + within-pair controls (T31 + T37) confirm length growth is ~half LLM-mediated (T29) but not artifact-dominated for content deltas.
- **Cite:** T08, T11 residualization; V1 Phase D (re-fit residuals match T11 at MAD 0.001).

### 6.2 Asaniczka label gap (moderate severity, senior-side only)

- **Finding:** S4 asaniczka 72.1% vs arshkon 65.0% = +7.1 pp within-2024 gap; comparable to cross-period S4 −7.6 pp pooled.
- **Mitigation:** Senior claims MUST cite both pooled (−7.6 pp) and arshkon-only (−1.8 pp) magnitudes (Gate 1 pre-commit; T30 primary rule).
- **Cite:** T30 §5 asaniczka-asymmetry audit.

### 6.3 Aggregator contamination (low-moderate severity)

- **Finding:** Aggregator share is 9.2-17.0% across periods; T22 found aggregators post CLEANER JDs (not ghost-inflated). T16 found aggregator-exclusion moves within-co AI by <20%.
- **Mitigation:** aggregator-exclusion sensitivity at task level. T31 aggregator-exclusion TIGHTENS AI-drift signal (+13.4 → +16.5 pp) — direct employers lead AI rewriting.
- **Cite:** T16 §6, T22 aggregator_vs_direct.csv, T31 sensitivity_aggregator.csv.

### 6.4 Company composition shift (moderate-high severity)

- **Finding:** 4,395 firms exited the 2024→2026 panel with lower-than-average J3 share (T16). New-entrant firms are LESS junior-heavy than returning (12.8% vs 15.3%). Between-company J3 rise is EXIT-driven, not entry-driven.
- **Mitigation:** T06 returning-companies cohort (n=2,109); T16 overlap panels (n=125-356); T37 sensitivity confirms 13/15 headlines robust on returning-cohort. Between-vs-within decomposition reported for every major metric.
- **Cite:** T06, T08, T16, T37.

### 6.5 Field-wide vs SWE-specific (high severity before T18)

- **Finding:** T18 DiD SWE−CTL +14.02 pp [+13.67, +14.37], V2 confirmed robust to 4 alt control defs (drop analysts, drop nurse, manual-work only, drop title_lookup_llm SWE tier). SWE-specific; not a general labor-market phenomenon.
- **Mitigation:** DiD framework adopted; T32 extends to 16 occupation subgroups. Control AI-rise is +0.17 pp → SWE AI-rise is 98.8% SWE-specific by DiD.
- **Cite:** T18, T32, V2 Phase E.

### 6.6 LLM-frame selection (moderate severity)

- **Finding:** scraped 2026 `not_selected` = 43% (no LLM labels); YOE-based J3/S4 require labeled subset.
- **Mitigation:** T29 low-LLM-quartile re-test (content effects persist 80-130%); labeled-vs-not split per Wave 2 dim h (Gate 1 pre-commit 4).
- **Cite:** T29 authorship_scores.csv; V2 pre-commit 4 audit (partial compliance).

### 6.7 Recruiter-LLM authorship mediation (rejected)

- **Finding:** Signature vocabulary density rose (Cohen d=0.22, SNR 4.8; 57.6% → 75.0% of postings). Content effects persist at 80-130% on low-LLM-quartile subset; length growth is ~52% LLM-mediated.
- **Mitigation:** T29 clean-signal framework. Only signature vocabulary density passes within-2024 calibration; sentence-length and em-dash features are format-confounded.
- **Cite:** T29 §3-5.

### 6.8 Macro hiring-context (JOLTS 2026 trough) (rejected as mechanism)

- **Finding:** JOLTS Information openings at 0.71× 2023 average (T07). T38 tested hiring-selectivity × content: |r|<0.11 across all metrics; desc-length r=+0.20 OPPOSITE direction (volume-UP firms write longer JDs).
- **Mitigation:** T38 formal null result; cite macro context as backdrop, not mechanism.
- **Cite:** T07, T38.

### 6.9 Platform taxonomy drift (high severity for industry)

- **Finding:** LinkedIn changed industry-label schema between 2024 and 2026 (Gate 1). `seniority_native` also drifted (+15-40 pp on same titles with flat YOE per T05).
- **Mitigation:** Forbid raw-label industry trends (Gate 1 pre-commit 7); use `seniority_final` for label-based but prefer YOE-based J3/S4 (Gate 1 pre-commit 8).
- **Cite:** T05, T01, T07.

### 6.10 T37 sampling-frame sensitivity (primary defense)

- **Finding:** 13 of 15 headlines retain ratio ≥ 0.80 on T06 returning-companies cohort (2,109 firms; 55% of 2026 postings). 0/15 sampling-frame-driven. J3/S4 directions INTENSIFY on returning-only.
- **Primary defense:** the returning-cohort robustness is the robustness-appendix centerpiece for RQ1 and RQ2 claims.
- **Cite:** T37.

### 6.11 Pattern-provenance mismatch (V2 flag D4)

- **Finding:** T16 and T23 report-text claims v1_rebuilt pattern (0.96 precision); code uses top-level ai_strict (0.86 precision). Numbers match top-level pattern exactly; under v1_rebuilt, magnitudes drop ~10-15% but direction unchanged.
- **Mitigation for SYNTHESIS.md:** cite pattern explicitly. Report within-co AI range as "+7.7 to +8.3 pp under top-level ai_strict pattern (V1 0.86 precision)" or "+7.5 to +7.8 pp under v1_rebuilt (0.96 precision)". Direction holds under both.
- **Cite:** V2 §3a.

### 6.12 T31 pair-count range (V2 flag D5)

- **Finding:** V2 could not reproduce T31's exact n=23; replicates n=37 (+9.98 pp) or n=12 under strict arshkon-only filter (+13.3 pp). Direction (pair > company-level) robust across all V2 reconstructions.
- **Mitigation for SYNTHESIS.md:** cite as "+10 to +13 pp range depending on panel filter; direction (pair > T16 company-level) robust".
- **Cite:** V2 §1 table, §7.

---

## 7. Ranked findings — the paper's backbone

### Tier A — Lead-candidate findings (strong, novel, core to the paper)

**A1. Cross-occupation employer-worker AI-codification divergence is universal and SWE-specific in magnitude.**

- **Claim:** Across 16 tested occupation subgroups spanning SWE, SWE-adjacent, and control, employer AI-mention rates are universally below worker AI-usage rates in BOTH 2024 AND 2026. SWE DiD magnitude is +14.02 pp [+13.67, +14.37]; control drift is +0.17 pp. Spearman(worker_mid, employer_2026) = +0.92: employers rank occupations identically to workers, at ~10-30% of worker-adoption levels.
- **Evidence:** T18 (DiD SWE=+14.19, control=+0.17, adjacent=+10.84); T23 (employer ai_strict 1.03%→10.61%, worker 63-90%); T32 (16/16 subgroups positive gap, Spearman +0.71 exposure×gap).
- **Sensitivity verdict:** **strong + robust.** V2 Phase E: SWE DiD survives dropping data/financial analysts, dropping nurse, restricting to manual-work controls, dropping title_lookup_llm SWE tier (all within 0.5 pp of +13-14 pp). T32 universality replicates from saved output; direction robust across 50/65/75/85% worker-usage bands.
- **Figure candidate:** `exploration/figures/T32_cross_occupation_divergence.png` (+ SVG) — **PAPER-LEAD.**

**A2. Within-firm AI content rewriting is real, same-title, and clean of composition.**

- **Claim:** Three independent panels converge on within-firm AI rewriting. T16 company-level (n=125-356): +7.65 to +8.34 pp. T37 returning-cohort (n=2,109): +7.91 pp (within 0.45 pp of T16). T31 pair-level same-company × same-title (n=23-37): +10 to +13.4 pp. Pair-level EXCEEDING company-level is the cleanest test that within-firm rewriting is same-title, not title-recomposition.
- **Evidence:** T16 (arshkon_min5 +8.34, pooled_min5 +7.65); T37 (within-co +7.91, consolidated range +7.7 to +8.3 pp); T31 (arshkon_min3 +13.4 reported / V2 +9.98 to +13.3 across filters; aggregator-excluded +16.5 pp).
- **Sensitivity verdict:** **strong direction + range-reportable magnitude.** V1 resolved T06-vs-T08 tension (both panels correct at different subsamples). V2 D5 flag: T31 pair count n=23 does not exactly reproduce; direction robust across all V2 reconstructions. V2 D4: under v1_rebuilt pattern, within-co Δ drops to +7.5-7.8 pp; direction unchanged.
- **Exemplar:** Microsoft "Software Engineer II" pair-level rewrite n=35 at +40 pp AI-strict (T31 top-20 exemplars; detailed in T25 artifact 1).

**A3. Seniority boundaries sharpened, not blurred.**

- **Claim:** Junior and senior SWE postings became MORE distinct between 2024 and 2026 under both unsupervised and supervised methods. T15 TF-IDF junior↔senior centroid cosine: 0.946 → 0.863 (diverging, Δ=−0.083). T20 associate↔mid-senior supervised AUC: 0.743 → 0.893 (+0.150). T20 continuous YOE×period interaction: +0.273 items/YOE-year on requirement_breadth (p<1e-44). T12 label-based AND YOE-based relabeling diagnostics agree: 2026 entry ≠ 2024 senior.
- **Evidence:** T15, T20, T12.
- **Sensitivity verdict:** **strong + methods-convergent.** V1 verified T15 direction exactly (0.950 → 0.871; Δ +0.004 / +0.008 vs T15 report). V2 confirmed associate↔mid-senior +0.146 (vs T20 +0.150) and flagged entry↔assoc / midsen↔dir as sample-thin (direction-only). T20 AUC sharpening survives YOE-drop sensitivity (+0.194 entry↔associate, +0.147 associate↔mid-senior without YOE feature).
- **Falsifies:** pre-exploration RQ1 "junior roles relabeled as senior" and "seniority boundaries blurring" priors.
- **Figure candidate:** `exploration/figures/T20/auc_by_boundary.png`; `exploration/figures/T15/similarity_heatmap_tfidf.png`.

**A4. Scope inflation is universal, senior > junior, within-domain — not junior-specific relabeling.**

- **Claim:** Length-residualized requirement_breadth rose +1.58 for J3 junior and +2.61 for S4 senior between pooled-2024 and scraped-2026. Within-company (V1 verified): S4 +1.97 > J3 +1.43. T28 decomposed scope rise as 60-85% WITHIN-domain, not between-domain. T33 tested and REJECTED the hidden hiring-bar lowering hypothesis (H_B).
- **Evidence:** T11 (J3 breadth_resid +1.58, S4 +2.61, within-company S4 +1.97 > J3 +1.43); T28 (within-domain +1.27 = 61% of total Δ +2.07); T33 (hiring-bar proxies |ρ|≤0.28, 0/50 loosening language).
- **Sensitivity verdict:** **strong.** V1 re-fit residualization independently; residuals match T11 at MAD 0.001. T28 within-dominant decomposition robust across pooled / arshkon-only / non-aggregator panels. T33 dual-classifier rejection; within-company Δ(req_share) × Δ(J3 share) ≈ 0.
- **Refines Gate 2 finding:** scope inflation is cross-title within-company (per T31 pair-level breadth shrinks 90%+ vs T16 co-level), NOT within-title. Companies broaden scope via re-titling and adding new senior-title archetypes.

**A5. Two emergent senior archetypes with content-grounded names.**

- **Claim:** T34 profiled T21's 5-cluster senior-cohort partition. **Cluster 0 (Senior Applied-AI / LLM Engineer):** 15.6× growth (144→2,251), 94% 2026, 28% "AI Engineer" title (true share ≥45% per manual check), median YOE 6.0 (+1 vs cluster 1), T09 archetype cross-tab models/systems/llm 6.75× over-representation + systems/agent/workflows 5.65×, distinguishing bigrams `claude code` `rag pipelines` `github copilot claude` `langchain llamaindex`, industry Software Development 44.6% + Financial Services 16.5%, 1,163 distinct firms at HHI 38.6. **Cluster 1 (Senior Data-Platform / DevOps Engineer):** 2.6× growth (2,034→5,258), +11.4 pp share, orch_density 2.27/1K chars (highest across clusters), T09 archetype pipelines/sql/etl 2.62× + kubernetes/terraform/cicd 2.11×.
- **Evidence:** T21 (5-cluster k-means, silhouette 0.477); T34 (cluster profiling, title distribution, T09 cross-tab, content exemplars, distinguishing n-grams).
- **Sensitivity verdict:** **strong for cluster 0; moderate for cluster 1.** V2 Phase B: 14/20 cluster-0 sampled titles are explicit AI/ML. V2 Phase C: cluster 0 share robust across S1/S2/S3/S4 T30 panel variants (10-14× rise each). Both clusters robust to aggregator exclusion (<3% metric shift). Cluster 1 is visibly a DE+DevOps bundle; a k=6-7 clustering would likely split.
- **Refinement for analysis phase:** pre-register k via BIC / gap-statistic; expect cluster 1 to split into DE and DevOps.

**A6. AI-DiD cross-occupation robustness across alt control definitions.**

- **Claim:** SWE DiD +13.10 pp [+12.76, +13.45] under V1-rebuilt pattern / +14.04 pp under top-level pattern, robust under 5 variants: drop analyst titles from control; drop nurse (largest control subgroup); manual-work controls only; drop title_lookup_llm SWE tier.
- **Evidence:** V2 Phase E (5 alt DiD specs, all within 0.5 pp of +13-14 pp).
- **Sensitivity verdict:** **strong.** Drop-title_lookup_llm tier (7% of SWE) leaves DiD at +13.19 pp. SWE-classifier-tier drift is NOT the explanation for the +14 pp DiD.

---

### Tier B — Strong supporting findings

**B1. AI-mention acceleration 2.80× above within-2024 noise.** Evidence: T19 + `calibration_table.csv`. Only AI (ratio 2.80) and scope (1.54) clear the >1 threshold under annualized rates. Seniority annualized rates dominated by within-2024 noise; use raw pp deltas (T30 panel) per T19 methodological note.

**B2. Management language is flat, not falling — measurement correction.** Evidence: V1 exposed mgmt_broad 0.28 precision and mgmt_strict 0.55 precision. T21 applied V1-rebuilt pattern: senior mgmt_rebuilt density 0.039 → 0.038 (Δ≈0, within-2024 SNR 0.1). V2 confirmed 0.034 → 0.034 replication. Corrects T11 "management fell" claim (which was a precision-0.28 pattern artifact). Publishable methods-caveat on longitudinal posting-content studies.

**B3. Hiring-bar-lowering REJECTED.** Evidence: T33. Classifier-sensitive magnitude (T13 −1.9 pp vs simple +3.0 pp HC3-robust period coefs, both p<1e-13). Hiring-bar-proxy correlations uniformly |ρ|≤0.28, sign-flipping across classifiers. Within-company Δ(req_share) × Δ(J3 share) ≈ 0 under both classifiers. Narrative 50-sample: 0 of 50 contain explicit loosening language. Δ(req_chars) × Δ(desc_length) r≈+0.35 supports "narrative expansion dominating" over "hiring bar lowering."

**B4. 13 of 15 headlines robust on returning-companies cohort.** Evidence: T37. Returning cohort (n=2,109; 55% of 2026 postings) retains ratio ≥0.80 on 13 of 15 tested headlines (V2 table shows 14/15). 0/15 sampling-frame-driven. J3 +5.0→+6.2 pp and S4 −7.6→−8.3 pp INTENSIFY on returning-only. Within-co AI +7.91 pp matches T16 range within 0.45 pp. H_d J3 breadth_resid at 0.70 is the one partial-robustness case; flagged.

**B5. Hiring-selectivity REJECTED.** Evidence: T38. Pearson |r|<0.11 on all selectivity-hypothesis metrics (breadth, ai_strict, scope, mentor, yoe) on arshkon_min3 n=243 primary panel. Desc-length r=+0.20 (p=0.0015) is OPPOSITE direction of selectivity; strengthens to r=+0.29 on non-aggregator + non-giant subset. Seniority-stratified Spearman ai_strict r=+0.30 J3 / +0.20 S4 (p<0.01) — also POSITIVE, volume-UP firms AI-rewrite more. Robust across 3 panels.

**B6. LLM-authorship NOT dominant mediator.** Evidence: T29. Low-LLM-quartile subset preserves 80-130% of Wave 2 content effects (AI-binary Δ +0.115 vs +0.131; tech_count +1.99 vs +2.06; credential J3 +16.6 pp vs +17.0 pp). Length growth IS ~half LLM-mediated (+583 vs +1,130 chars, 52%) — partial mechanism for boilerplate. T13 requirements-share decline near-disappears on low-LLM (−0.025 → −0.002). Only signature vocabulary density passes within-2024 calibration (SNR 4.8).

**B7. AI dev-tools sub-cluster SPLIT OFF in 2026.** Evidence: T35. Louvain community detection: `copilot + cursor_tool + chatgpt + claude_tool + codex` form an isolated 5-tech community with ≥4 of 5 members having no 2024 above-threshold edges. This is the cleanest quasi-new cluster in the 2026 tech taxonomy — semantically distinct from LLM engineering (langchain / llamaindex / rag / vector_database sit in the LLM-vendor cluster).

**B8. CI/CD is the single largest S4 riser (+20.6 pp), bigger than any AI term.** Evidence: T14 + T15. Orchestration (CI/CD, pipelines, workflows) is the content backbone of the senior-tier shift. T21 orch_density rose +0.67/1K chars at mid-senior (SNR 5.6), +0.44/1K at director. Observability merged into devops_platform cluster (T35).

---

### Tier C — Supporting + methodological contributions

**C1. Technology ecosystem modularity rose modestly +0.029.** T35. 0.632 → 0.662 across 10 Louvain seeds (SD <0.002). LLM-vendor cluster pre-existed 2024 (pooled arshkon+asaniczka); 2026 added 4 members (mcp, gemini, chroma, claude_api) and split off the ai_dev_tools sub-cluster.

**C2. Geographic AI rise is UNIFORM; ML/LLM diffusing to non-hubs.** T17. Median SNR ~10, all metros +4.7 to +14.5 pp. AI-rise leaders: Atlanta, Tampa, Miami, SLC (NOT SF/Seattle). J3 rise is geographically HETEROGENEOUS (−5 pp Detroit to +15 pp San Diego); r(AI-rise, J3-rise) = −0.22. AI and junior-share changes are geographically decoupled.

**C3. Aggregators post CLEANER JDs than direct employers.** T22. Counter-prior. Kitchen-sink score direct 18.1 vs aggregator 13.7 (2026). Aspiration share direct 25.3% vs aggregator 19.6%. YOE-scope mismatch concentrates at direct employers. Useful methodological finding for posting-study scholarship.

**C4. Copilot at 0.10% of postings.** T23. GitHub Copilot has ~33% regular-use industry benchmark and 4.7M paid subs; appears in 0.10% of SWE postings. Employers do NOT formalize even the most-adopted AI tool. Explains part of the employer-worker divergence: under-specification in JDs.

**C5. Legacy substitution is LEGACY → MODERN STACK, not LEGACY → AI.** T36. 6 of 11 disappearing 2024 titles map to 2026 neighbors via TF-IDF cosine [0.30, 0.59]. 2026 neighbors' ai_strict rate 3.6% (below 14.4% market rate). Content drift: SSIS/Unix-scripts/Drupal/PHP/Scala → Postgres/pgvector/CI-CD/microservices/Terraform/ArgoCD. 2-4 year YOE drop on architect → engineer substitutions. Title-inflation reversal, not AI-ification.

**C6. T30 multi-operationalization panel framework.** Methods contribution. 12/13 definitions direction-consistent; junior-side 7/7 UP, senior-side 5/6 DOWN, S2 director-only flat. Paper should publish the panel as reusable tool for longitudinal posting studies.

**C7. Within-2024 SNR calibration framework.** Methods contribution. Per-metric SNR ratio = (cross-period effect) / (within-2024 noise). AI-strict SNR 32.9; scope-terms 42.8. Seniority shares near/below noise for annualized rates (use raw pp deltas). Distinguishes clean signals from instrument artifacts.

**C8. V1 pattern-validation methodology exposed 0.28/0.55 precision on widely-used management patterns.** Methods contribution. `mgmt_broad` 0.28 (all 4 broad tokens below 0.80); `mgmt_strict` 0.55 (hire 0.07, performance_review 0.25 worst subterms). V1-rebuilt mgmt_strict_v1_rebuilt re-validated at 0.98-1.00 precision by T22. Longitudinal posting studies need semantic (not tautological) precision validation.

---

### Tier D — Flagged / demoted findings

**D1. "Junior requirements section chars shrank" (T13).** Classifier-sensitive (direction flips under simple-regex; V1 H3 alt-classifier gave +88% reqs delta). Authorship-sensitive (T29: near-disappears on low-LLM subset, −0.025 → −0.002). T33 confirmed classifier-sensitivity extends to AGGREGATE level (not just J3). **Demoted to qualified claim:** "junior descriptions densified on AI/tech tokens; net requirements-section direction is classifier-uncertain."

**D2. "Aggregate credential stack depth rose."** T11 aggregate +0.20; V1 within-2024 noise 0.34, SNR 0.59 — below noise. Cite only per-seniority claims (J3 +16.9 pp, S4 +13.3 pp); do NOT cite aggregate.

**D3. MCP in ai_broad pattern for 2024 baselines.** "Microsoft Certified Professional" contamination in 2024 asaniczka. Use ai_strict for MCP-specific growth citations; drop MCP from ai_broad for 2024-baseline comparisons.

**D4. T16/T23 pattern-provenance mismatch.** V2 flagged that T16 and T23 numbers match top-level `ai_strict` pattern (0.86 precision) rather than `ai_strict_v1_rebuilt` (0.96 precision). Under V1-rebuilt, magnitudes drop ~10-15%; direction unchanged. SYNTHESIS.md cites explicitly; analysis-phase pre-register pattern.

**D5. T31 pair-count panel-dependent.** V2 replicates +10-13 pp across panel constructions (n=12 strict arshkon-source-only / n=23 primary / n=37 relaxed). Direction (pair > T16 co-level) robust; exact magnitude range-report.

---

## 8. Discovery findings organized

### Confirmed (Wave 2/3 priors that survived)

- AI-mention rise is the cleanest content signal (T08 + T12 + T14 + T19 + T23).
- Scope broadened universally with senior-tier larger shifts (T11 + T28).
- Tech-stack expanded additively with AI (T14: AI postings 11.35 techs vs non-AI 6.81, density 48% higher after length-normalization).
- Junior and senior postings are dissimilar, not converging (T15 + T20).
- Boilerplate-led length growth (benefits +89%, responsibilities +49%, legal +80%; T13).
- CI/CD S4 riser +20.6 pp (T14).
- Financial-services dominance in AI-rich outliers (T11 top-1%).

### Contradicted (Wave 2/3 priors that were falsified)

- "Junior share reduces" — J3 share rose +5 pp (T30, T08).
- "Junior roles relabeled as senior" — boundaries sharpened (T15, T20, T12 diagnostics).
- "Management declining" — measurement artifact; flat under V1-validated pattern (T11 → T21 correction).
- "Hiring bar lowered via requirements contraction" — REJECTED (T33, 0/50 loosening language).
- "Legacy titles substituted by AI-enabled roles" — REJECTED (T36, 2026 neighbors 3.6% ai_strict below market).
- "Recruiter-LLM writing explains the AI rise" — REJECTED (T29 low-LLM retains 80-130% of content effects).
- "Hiring-trough selectivity explains content shifts" — REJECTED (T38 |r|<0.11).
- "Sampling-frame artifact drives J3 rise" — REJECTED (T37 13-14/15 headlines robust).
- "Within-firm AI rewriting is title-recomposition" — REFINED: pair-level EXCEEDS co-level (T31), same-title rewriting dominates.

### New discoveries (post-Wave-3.5)

- **Two emergent senior archetypes, not one** (T34 cluster 0 + cluster 1).
- **Cross-occupation employer-worker divergence is universal** (T32 16/16 subgroups, Spearman +0.92).
- **Accountant 72× ratio gap larger than SWE 7× gap** (T32).
- **Nurse ai_strict = 0.00% on 6,801 postings** (T32).
- **AI-oriented senior roles ask MORE experience (YOE 6 vs 5)** (T34) — counter to a "AI lowers the bar" prior.
- **AI dev-tools sub-cluster split off** (T35 — 4/5 members new-in-2026).
- **New-entrant firms are LESS junior-heavy than returning firms** (T16: 12.8% vs 15.3%). Between-firm J3 rise is EXIT-driven.
- **S4 senior within-firm decline EXCEEDS aggregate** (T37: −9.31 pp within-co vs −8.29 pp aggregate on returning-cohort). Parallel "exit-driven senior decline" to T16 "exit-driven junior rise".

### Unresolved tensions (see §11 for analysis-phase priorities)

- T31 pair-count methodology reproducibility (V2 D5).
- Pattern-label provenance mismatch (V2 D4).
- T34 cluster 1 heterogeneity (DE + DevOps + AI-data-contract bundle).
- T22 hedging-near-AI: J3-specific fraction not quantified (proposed H_Q in T24).

---

## 9. Posting archetype summary (T09 + T28 + T34)

**T09 headline:** DOMINANT-AXIS is technology/role-type, NOT seniority. BERTopic NMI(clusters, title_archetype) = 0.216 vs NMI(clusters, seniority_3level) = 0.025. **Ratio 8.8×.** NMF k=12 ARI=0.998 (stable). BERTopic seed-unstable; primary labels saved at `exploration/artifacts/shared/swe_archetype_labels.parquet` (8,000 rows) with noise preserved as -1.

**Biggest archetype share gainers (T09):**
- ML/LLM archetype: +4.8 pp (1.65% → 6.47% of 8,000-row sample).
- DevOps/Kubernetes: +2.2 pp.
- AI-agent-orchestration: new-in-2026 (2 → 45 rows, 10× growth).

**Biggest shrinkers:**
- Generic Java/Spring/microservices: −2.0 pp.
- DoD clearance/MBSE: −2.1 pp.

**T28 archetype domain buckets** (15 domains covering 48,223 labeled + projected rows):

| Domain | % 2024 | % 2026 | Δ pp |
|---|---|---|---|
| **ai_ml** | 2.6 | **12.8** | **+10.2** |
| cloud_devops | 11.5 | 15.0 | +3.5 |
| data_eng | 3.6 | 5.5 | +2.0 |
| backend | 18.5 | 21.5 | +3.0 |
| generic_swe | 19.9 | 17.8 | −2.1 |
| **clearance_defense** | 12.6 | 6.8 | **−5.8** |
| **dotnet_legacy** | 3.8 | 1.2 | −2.6 |
| other | 9.6 | 4.5 | −5.1 |

**Key T28 decomposition:** Scope inflation is 60-85% WITHIN-domain across 5 metrics (breadth_resid, tech_count, scope_density, credential_stack_depth, ai_binary). J3 entry-share rise (+5.07 pp) is +6.84 pp within-domain, −0.02 pp between-domain; domain recomposition does NOT explain the J3 rise.

**T34 emergent senior archetype profiles:**

- **Cluster 0 (Senior Applied-AI / LLM Engineer):** 15.6× growth, 94% 2026, distinguishing n-grams `rag pipelines`, `claude code`, `github copilot claude`, `langchain llamaindex`; industry Software Development 44.6% + Financial Services 16.5%; HHI 38.6 across 1,163 firms; median YOE 6.0; director share 1.9%; concentrates 30% in T16 `ai_forward_scope_inflator` employer cluster.
- **Cluster 1 (Senior Data-Platform / DevOps Engineer):** 2.6× growth, 72% 2026, +11.4 pp share; titles 13% data engineer + 11% DevOps/SRE + 20% senior engineer; T09 pipelines/sql/etl 2.62× + kubernetes/terraform/cicd 2.11× over-represented; concentrates in T16 `traditional_hold` clusters at 24-29%.

---

## 10. Technology evolution summary (T14 + T35 + T36)

**T14 findings:**
- AI-family tech crystallization: phi(pinecone, weaviate) = 0 → 0.71; phi(rag, llm) = 0.19 → 0.49. V1 verified at 0.72 and 0.49 respectively.
- 60 rising vs 12 declining techs under within-2024 calibration.
- Stack diversity rising: J3 median 4 → 6 techs; S4 median 5 → 7.
- CI/CD is the single largest S4 tech riser: +20.6 pp (bigger than any AI term).
- AI-strict postings have 48% higher tech density after length-normalization (additive, not substitutive).

**T35 ecosystem crystallization:**
- Modularity +0.029 (0.632 → 0.662), robust across 10 Louvain seeds (SD<0.002).
- 9 communities in 2024 → 10 in 2026 pooled.
- LLM/AI cluster PRE-EXISTED 2024 (n=16 techs at phi>0.15 on pooled arshkon+asaniczka). 2026 added 4 members (mcp, gemini, chroma, claude_api) and SPLIT OFF the ai_dev_tools sub-cluster.
- **AI dev-tools sub-cluster:** copilot + cursor_tool + chatgpt + claude_tool + codex. 4 of 5 members had no 2024 above-threshold edges. Quasi-new semantically distinct cluster.
- **Observability merged into devops_platform** (Datadog, New Relic, PagerDuty, Grafana, Prometheus, Splunk all joined the 2026 devops_platform cluster).
- **Data-engineering fragmented** into modern warehouse stack (Snowflake, dbt, Airflow, Databricks, BigQuery, Spark, Flink) and web-backend stack (PHP, Laravel, MongoDB, PostgreSQL, Redis).

**T36 legacy substitution map (n=6 credibly matched):**
- 6 of 11 wider-list disappearing titles match 2026 neighbors at cosine [0.30, 0.59].
- Substitution pattern: database developer → database engineer; devops architect → devops engineer (−4 YOE); java architect → java developer (−3 YOE); drupal developer → web developer; scala developer → big data engineer.
- **AI-strict rate at 2026 neighbors: mean 3.6%** — BELOW the 14.4% 2026 market rate.
- **Content drift:** SSIS + Unix scripts + Drupal + PHP + Scala LEAVE. Postgres + pgvector + CI/CD + microservices + Terraform + ArgoCD ARRIVE.
- **H_L (legacy → AI-enabled substitution) is REJECTED.** Substitution is legacy-stack → modern-stack, NOT legacy-stack → AI.
- Title-inflation reversal: architect titles substitute into engineer titles with 2-4 year YOE drop.

---

## 11. Geographic heterogeneity summary (T17)

**Headlines:**
- **AI rise is geographically UNIFORM.** 26 metros with ≥50 postings per era; median SNR ~10; every metro +4.7 to +14.5 pp on AI-strict. Tech-hub premium < 2 pp.
- **AI-rise leaders:** Atlanta, Tampa, Miami, Salt Lake City (NOT SF Bay Area or Seattle).
- **ML/LLM archetype is DIFFUSING to non-hubs.** Minneapolis +13 pp > Seattle +8.8 pp on ML/LLM share.
- **J3 rise is geographically HETEROGENEOUS.** −5.1 pp Detroit to +14.9 pp San Diego; SNR<2 on 13 of 18 calibrated metros.
- **AI and junior-share changes are decoupled** at the metro level: r(AI Δ, J3 Δ) = −0.22.
- **Remote:** 19.9% of 2026 scraped (0% 2024 per scraper design); remote AI rate 11.1% barely above metro 10.5%.
- **Aggregator-exclusion and cap-20 move metro deltas <1 pp.**

**Implication for paper:** the AI-restructuring story is NOT a tech-hub phenomenon. It is a field-wide, geography-general phenomenon with ML/LLM diffusion favoring non-hub metros. This adds a dimension to A1 (cross-occupation universality) and A3 (seniority sharpening): the restructuring is also geography-universal.

---

## 12. Senior archetype characterization (T21 + T34)

**T21 5-cluster k-means on senior SWE cohort** (mid-senior + director, n=33,693):

| Cluster | 2024 share | 2026 share | Δ pp | Description |
|---|---|---|---|---|
| Low-profile generic | 62.0% | 38.4% | **−23.6** | Short, low on all density signals (was default 2024 senior posting) |
| Tech-orchestration (non-AI) | 14.9% | 26.2% | **+11.4** | CI/CD, pipelines, workflows, K8s (T34 cluster 1) |
| AI-oriented | 1.1% | 11.2% | **+10.2** | LLM, RAG, agentic, orch (T34 cluster 0) |
| People-management | 3.4% | 5.3% | +1.9 | Mentor + direct reports (V1-validated pattern) |
| Strategic-language | 18.7% | 18.8% | +0.1 | Stakeholder/roadmap (pattern precision 0.32, diagnostic only) |

**Key senior-tier findings:**
- **Management is FLAT at all tiers under V1-validated pattern.** T21 mgmt_rebuilt density: mid-senior 0.039 → 0.038, director 0.031 → 0.026. Junior-side 0.000 → 0.004 (near-zero). V2 confirmed 0.034 → 0.034 on independent replication.
- **Orchestration rose significantly.** Mid-senior orch_density 0.423 → 1.095 (+0.67 / 1K chars, SNR 5.6). Director orch_density 0.396 → 0.833 (+0.44).
- **AI-strict rose sharply at senior.** Mid-senior 1.1% → 11.8% (+10.7 pp). Director 1.1% → 14.8% (+13.7 pp, 13.4× growth).
- **AI × senior interaction:** within senior cohort, AI-mentioning postings have 2× higher orch_density AND LOWER mgmt density than non-AI senior. AI-senior roles are MORE technical-orchestration-heavy, NOT more management-heavy.

**T34 cluster 0 (Applied-AI / LLM Engineer) content signature:**
- Median YOE 6.0 years (+1 vs cluster 1).
- Director share 1.9% (2× cluster 1's 1.0%).
- 28.2% titled "AI Engineer" at face value (true share ≥45%); 17.8% senior engineer; 8.0% staff engineer; 6.0% tech lead.
- Industry: Software Development 44.6%, Financial Services 16.5%, IT Services 13.6%.
- Geography: SF Bay Area 26%, DFW 11.4%, Seattle 11.1%, NYC, Boston, Atlanta.
- Distinguishing bigrams: `rag pipelines` (+5.04), `claude code` (+5.52), `github copilot claude` (+5.02), `langchain llamaindex` (+4.86), `augmented generation rag` (+4.72).
- Recurring asks (20 exemplars): LLM/RAG/prompt 12/20; pipelines 13/20; agentic 8/20; orchestration 7/20; system design 7/20.

**T34 cluster 1 (Data-Platform / DevOps Engineer) content signature:**
- Median YOE 5.0 years.
- Titles: 13% data engineer + 11% DevOps/SRE + 20% senior engineer.
- Industry: Software Development 43.1%, IT Services 14.5%, Financial Services 11.9%. Staffing/Recruiting 6.0%.
- Distinguishing bigrams: `training benchmarking`, `annotation validation`, `dask spark`, `benchmarking pipelines`, `automation-first`.
- Visibly a BUNDLE of data-engineering + DevOps/SRE + AI-lab data-contract work. A k=6-7 clustering would likely split.

---

## 13. Ghost/aspirational prevalence (T22)

**T22 findings on 48,634-row SWE LinkedIn corpus:**
- **Kitchen-sink score** (tech_count × scope_kitchen_count) rose 2.6× at aggregate (6.72 → 17.36 mean; median 0 → 8).
- **YOE-scope mismatch for J3 (yoe≤2):** 2.88% (2024) → 7.45% (2026) = 2.6× rise. Absolute rate is still minority (<10% of J3 postings).
- **Aspirational language NEAR AI tokens:** 50.3% near-AI aspiration vs 28.2% far-from-AI in 2026 — AI asks are 1.8× more hedging-adjacent than non-AI text.
- **LLM ghost-assessment is FLAT cross-period:** 6.1% (2024) → 5.3% (2026). The LLM-adjudicated ghost rate did NOT rise; only measurable breadth rose.
- **Aggregators post CLEANER JDs than direct employers** (T22 aggregator_vs_direct.csv):
  - Kitchen-sink score 2026: direct 18.12 vs aggregator 13.72.
  - Aspiration share: direct 25.3% vs aggregator 19.6%.
  - YOE-scope mismatch: direct 1.12% vs aggregator 0.34%.
  - **Ghost-likeness concentrates at DIRECT employers, not aggregators** — counter to the typical "aggregators are spammy" prior.
- **Template saturation is RARE.** Only 1 firm (Syms Strategic Group LLC) at company-period mean pairwise cosine >0.80 across 400 company-period cells.
- **Top-20 ghost-entry postings are AI-driven.** Visa new-grad × 4 with LLM fine-tuning / GPT integration asks; DataVisor Software Engineer (AI); PNNL Senior Software Engineer III with 36 tech_count / 9 scope_kitchen_count; PayPal Software Engineer Routing Platform with architect-level asks at YOE 1.

**Pattern validation pipeline** (`exploration/artifacts/shared/validated_mgmt_patterns.json`, T22 re-validated + extended):

| Pattern | T22 precision (n=50) | V2 precision (n=30) | Recommendation |
|---|---|---|---|
| ai_strict_v1_rebuilt | 0.96 | — | PRIMARY |
| mgmt_strict_v1_rebuilt | 0.98 | — | PRIMARY |
| scope_v1_rebuilt | 1.00 | — | PRIMARY |
| aspiration_hedging | 0.92 | **1.00** | PRIMARY |
| firm_requirement | 1.00 | **1.00** | PRIMARY |
| scope_kitchen_sink | 0.96 | — | PRIMARY |
| senior_scope_terms | ≥0.89 by construction | — | PRIMARY |

All 7 primary patterns have `semantic_precision_measured: true`. V2 independently confirmed aspiration_hedging and firm_requirement at 1.00 / 1.00 on fresh stratified 30-row samples.

---

## 14. Robustness appendix (centerpiece of paper's defense)

### T37 sampling-frame sensitivity table

Primary robustness-appendix centerpiece. 15 tested headlines retained on returning-companies cohort (n=2,109):

| Metric | Full Δ | Returning Δ | Ratio | Verdict |
|---|---|---|---|---|
| H_a AI-strict prevalence | +9.72 pp | +8.36 pp | 0.86 | robust |
| H_b J3 entry share (pooled) | +5.05 pp | **+6.17 pp** | 1.22 | robust (amplified) |
| H_b-alt J3 entry (arshkon-only) | +1.19 pp | +2.10 pp | 1.77 | robust |
| H_c S4 senior share (pooled) | −7.62 pp | **−8.29 pp** | 1.09 | robust (amplified) |
| H_c-alt S4 senior (arshkon-only) | −1.94 pp | −3.06 pp | 1.58 | robust |
| H_d-J3 Breadth residualized (J3) | +1.56 | +1.09 | **0.70** | **partially robust (only one)** |
| H_d-S4 Breadth residualized (S4) | +2.60 | +2.55 | 0.98 | robust |
| H_e-J3 Credential stack ≥5 (J3) | +17.1 pp | +16.5 pp | 0.97 | robust |
| H_e-S4 Credential stack ≥5 (S4) | +13.4 pp | +13.7 pp | 1.03 | robust |
| H_f T13 requirements-share | −2.54 pp | −3.37 pp | 1.33 | robust |
| H_g Description length median | +1,244 chars | +1,276 chars | 1.03 | robust |
| H_h Scope term prevalence (V1) | +23.26 pp | +22.41 pp | 0.96 | robust |
| H_h-alt Scope kitchen-sink | +22.82 pp | +23.21 pp | 1.02 | robust |
| H_i CI/CD tech at S4 | +20.62 pp | +20.27 pp | 0.98 | robust |
| H_j AI-oriented senior cluster | +10.19 pp | +9.18 pp | 0.90 | robust |

**Bootstrap 95% CIs on returning cohort:** AI-strict [+7.61, +9.08]; J3 [+3.21, +9.86]; S4 [−11.97, −4.67]; Scope [+19.94, +25.27]. All exclude zero.

### T38 hiring-selectivity null

Pearson correlations on arshkon_min3 n=243 primary panel:

| Metric | Pearson r | 95% CI | p |
|---|---|---|---|
| breadth_resid Δ | −0.032 | [−0.157, +0.094] | 0.617 |
| ai_strict Δ | −0.089 | [−0.212, +0.038] | 0.169 |
| mentor_on_S1 Δ | −0.072 | [−0.202, +0.061] | 0.287 |
| **desc_len_median Δ** | **+0.203** | [+0.079, +0.321] | **0.0015** |
| yoe_llm_median Δ | −0.008 | [−0.138, +0.122] | 0.899 |
| scope_v1 Δ | −0.033 | [−0.158, +0.093] | 0.605 |

Only desc_length is significant; direction is POSITIVE (opposite of selectivity). Panel-robust across arshkon_min5 n=125 and pooled_min5 n=356.

### T29 low-LLM-quartile re-test

Content deltas preserved on bottom quartile by signature vocabulary density:

| Metric | Full corpus | Low-LLM quartile | % preserved |
|---|---|---|---|
| AI-binary Δ | +0.131 | +0.115 | 88% |
| Tech count Δ | +2.06 | +1.99 | 97% |
| Scope density Δ | +0.096 | +0.123 | **128%** (stronger) |
| Credential stack ≥5 (J3) Δ | +16.9 pp | +16.6 pp | 97% |
| Requirement breadth_resid Δ | +1.22 | +0.98 | 80% |
| Length growth | +1,130 chars | +583 chars | **52%** (half LLM-mediated) |

### Aggregator-exclusion sensitivities (T06, T16, T31)

- T16 within-company AI: aggregator-excluded moves by <20% (not re-reported here; direction-preserving).
- T31 pair-level AI drift: aggregator-excluded TIGHTENS signal from +13.4 → +16.5 pp (direct employers lead AI rewriting).
- T22 ghost-likeness concentrates at direct employers, not aggregators.
- T34 cluster 0 and cluster 1 metrics shift <3% on aggregator-exclusion.

### Within-2024 calibration SNRs

Shared artifact: `exploration/artifacts/shared/calibration_table.csv`. Selected values:

| Metric | Within-2024 effect (d or %) | Cross-period effect | SNR ratio | Verdict |
|---|---|---|---|---|
| ai_mention_strict | +0.004 | +0.133 | **32.9** | **above_noise** |
| scope_term_rate | +0.005 | +0.210 | **42.8** | **above_noise** |
| description_length_mean | +0.28 (d) | +0.50 (d) | 1.80 | near_noise |
| J3 share | +4.75 pp | +5.04 pp | 1.06 | near_noise |
| S4 share | +7.09 pp | +7.59 pp | 1.07 | near_noise |
| mgmt_strict prevalence | +0.038 | +0.080 | 2.07 | above_noise |
| mgmt_broad prevalence | +0.062 | +0.028 | 0.45 | below_noise |
| Aggregate credential stack | noise | +0.20 (T11) | 0.59 | below_noise |

**Implication:** AI and scope are above noise as cross-period signals; seniority shares are near-noise under annualized rates (use raw pp deltas per T19/T30). Aggregate credential stacking and mgmt_broad are below-noise (cite per-seniority or rebuilt pattern).

### V1 + V2 corrections

**V1 pattern-validation corrections applied:**
- `mgmt_broad` (0.28 precision): RETIRED. Used in T11 → corrected in T21.
- `mgmt_strict` original (0.55): replaced with `mgmt_strict_v1_rebuilt` (1.00 precision on T21 + T22 re-validation).
- `ai_strict` fine-tuning sub-pattern 0.47 precision in 2024: restricted to LLM-adjacent context in `ai_strict_v1_rebuilt`.
- `ai_broad` MCP: dropped for 2024-baseline comparisons (Microsoft Certified Professional contamination).

**V2 corrections applied:**
- T16 / T23 pattern-provenance: cite explicitly, range-report.
- T31 pair-count: range-report +10 to +13 pp across panel constructions.
- T37 headline count: 14/15 robust per saved table (report-text said 13/15).
- T20 entry↔associate and mid-senior↔director AUC: demote to "direction only, power-limited".

---

## 15. Hypothesis status table (every hypothesis, exploration + planned)

| Hypothesis | Source | Verdict | Evidence | Analysis-phase action |
|---|---|---|---|---|
| RQ1 "junior share reduces" | Pre-exploration | Contradicted | T08 +5 pp J3 rise; T30 7/7 junior UP | Retire |
| RQ1 "junior scope inflates" | Pre-exploration | Partial — universal, senior>junior | T11, T28 | Reframe |
| RQ1 "junior relabeled as senior" | Pre-exploration | Contradicted | T15, T20, T12 | Retire |
| RQ1 "senior redefinition toward AI" | Pre-exploration | Supported — emergent-role-specific | T21, T34 | Extend |
| RQ1a "seniority boundaries blur vs sharpen" | Gate 1 induced | **Sharpen** supported | T15, T20 | Formal test pre-register |
| RQ1b "scope change within vs cross-seniority" | Gate 2 induced | Within-seniority, senior>junior | T11, T28 | Formal test pre-register |
| RQ1c "emergent senior archetypes" | Gate 2 induced | **2 archetypes supported** | T34 | Cluster-selection sensitivity |
| RQ2 "requirements migrate downward" | Pre-exploration | Contradicted | T11, T20 AI×YOE null | Retire |
| RQ2 "migration into YOE≤2" | Pre-exploration | Supported with qualification | T28 | Extend with archetype controls |
| RQ3 "employer under-codifies AI" | Pre-exploration | **STRONGLY SUPPORTED, universal 16/16** | T23, T32 | Lead-claim status |
| RQ3 "SWE-specific magnitude" | Pre-exploration | Supported | T18, V2 Phase E | Lead-claim status |
| RQ5 (induced) "within-firm rewriting explains SWE content shifts" | Gate 3 induced | **SUPPORTED** | T16+T31+T37+T38+T29 convergence | Lead-claim status |
| H_A cross-occupation divergence | T24-planned | **STRONGLY SUPPORTED** | T32 16/16 | Extend to 2025 benchmarks |
| H_B hidden hiring-bar lowering | T24-planned | **REJECTED** | T33 | Archive |
| H_C emergent senior archetype | T24-planned | **SUPPORTED — 2 clusters** | T34 | Cluster-selection sensitivity |
| H_D senior-IC-as-team-multiplier | T24-planned | Deferred | — | HIGH priority; external hiring panel |
| H_E same-co J1 drop + J3 rise regime shift | T24-planned | Deferred | T19 absorbed partially | LOW priority; requires longer panel |
| H_F Sunbelt AI surge catchup | T24-planned | Partially absorbed | T17 uniform | LOW priority |
| H_G staff-title redistribution | T24-planned | Deferred | T10 + T36 supporting | MEDIUM priority |
| H_H sampling-frame artifact | T24-planned | **REJECTED** | T37 | Archive |
| H_I AI as coordination signal | T24-planned | Deferred | — | MEDIUM priority; interview-carrier |
| H_J recruiter-LLM senior bias | T24-planned | Deferred; T29 general-case rejected | — | MEDIUM priority |
| H_K ecosystem crystallization | Wave 3.5 | **PARTIALLY SUPPORTED** | T35 modest modularity; sub-cluster split | Extend methodology |
| H_L legacy → AI substitution | Wave 3.5 | **REJECTED** | T36 | Archive |
| H_M same-co same-title drift | Wave 3.5 | **SUPPORTED** | T31 | Formal pair panel pre-register |
| H_N hiring-selectivity × scope | Wave 3.5 | **REJECTED** | T38 | Archive |
| H_O digital-maturity × AI-rewriting | Post-3.5 (T24) | Proposed | T34 + T16 implicit | HIGH priority; external maturity data |
| H_P Applied-AI in financial-services-compliance | Post-3.5 (T24) | Proposed | T34 + T11 indirect | MEDIUM priority |
| H_Q J3 hedging-near-AI | Post-3.5 (T24) | Proposed | T22 implicit | MEDIUM priority; data exists |

---

## 16. Method recommendations for analysis phase

### 16.1 Pattern validation protocol (V1 + T22)

- For every content pattern used in claims, measure semantic precision on a stratified 30-50 row sample (pre-period + post-period).
- Thresholds: PRIMARY at ≥0.80 precision; DIAGNOSTIC at ≥0.60; FAIL below 0.60.
- Report per-sub-pattern precisions (hire 0.07, performance_review 0.25, etc.).
- Artifact format: `validated_mgmt_patterns.json` schema (pattern / precision / sub_pattern_precisions / by_period_precision / fp_classes / recommendation).

### 16.2 T30 multi-operationalization seniority panel

- Report 13-definition panel (label-based J1/J2/S1/S2/S3; YOE-based J3/J4/S4/S5; title-keyword J5/S3; rule-based J3_rule/S4_rule) for every seniority-stratified finding.
- Primary: J3 + S4 pooled-2024 baseline with arshkon-only co-primary for senior.
- Mandate: cite both pooled and arshkon-only magnitudes for senior claims (asaniczka asymmetry).

### 16.3 Within-2024 SNR calibration

- For every cross-period claim, report within-2024 noise floor (arshkon vs asaniczka on same metric).
- Classify metric as above_noise (SNR>2), near_noise (1<SNR<2), or below_noise (SNR<1).
- Below-noise findings cannot be cited as point estimates; only directionally.

### 16.4 Within-vs-between decomposition

- For every aggregate delta, report within-company and between-company components on returning-company panels (T06 returning-cohort, T16 overlap panels).
- For pair-level analyses (T31): report panel construction explicitly (n≥2 vs n≥3 per pair-period; arshkon-only vs pooled vs relaxed filters).

### 16.5 Classifier sensitivity for section-based claims

- Dual-classifier reporting for all requirements-section / responsibilities-section / boilerplate claims.
- If direction flips (T33 aggregate; V1 H3 J3), demote to "flagged" or "classifier-uncertain".

### 16.6 Length residualization

- For any composite breadth/stacking metric with r(metric, log_length) > 0.3, residualize before cross-period comparison.
- Fit: `y ~ a + b × log(description_length)` on full corpus; report residual.
- T11 Phase D established: requirement_breadth (r=0.43), credential_stack (r=0.34), scope_count (r=0.41), mgmt_broad (r=0.39) all require residualization.

### 16.7 Benchmark-band robustness (T32)

- For worker-vs-employer divergence claims, report under 4 bands (50/65/75/85% worker-usage).
- Direction-robust across all bands is required for lead citation.

### 16.8 Aggregator-exclusion sensitivity

- Apply as primary sensitivity for ALL prevalence claims (Gate 2 pre-commit).
- Direction-preserving + magnitude shift <20% is the robustness threshold.

---

## 17. Sensitivity requirements for analysis phase (gaps exploration didn't fully close)

### 17.1 Industry-taxonomy crosswalk

- LinkedIn industry-label schema changed between 2024 and 2026; raw-label trends invalid.
- Analysis-phase pre-req: construct company-level sector mapping via external data (e.g., NAICS via company name + website). Cross-industry claims require this crosswalk.

### 17.2 T31 pair-panel pre-registration

- V2 could not reproduce T31's exact n=23; V2 replicates n=37 (+9.98 pp) or n=12 (+13.3 pp) under alt filters.
- Analysis-phase pre-register: threshold (n≥3 vs n≥2 per pair-period), source-restriction (arshkon-only vs pooled), panel-intersection (arshkon_min3 vs arshkon_min5 vs pooled_min5).
- Report range explicitly.

### 17.3 Pattern-provenance re-derivation

- T16 / T23 numbers match top-level ai_strict; report-text says v1_rebuilt.
- Analysis-phase: re-derive under v1_rebuilt for cleaner precision; expect magnitude drop ~10-15%.

### 17.4 Cluster-selection robustness (T34)

- k=5 in T21 is ad-hoc. Silhouette 0.477 is mid-range.
- Analysis-phase: test k=4-7 via BIC or gap-statistic; expect cluster 1 (Tech-orch + DevOps + AI-data-contract bundle) to split.

### 17.5 LLM-adjudicated section classifier

- T33 classifier sensitivity unresolved (T13 vs simple-regex flip).
- Analysis-phase: LLM-adjudicate 50-row sample per classifier to triangulate.

### 17.6 Senior-specific LLM-authorship mediation (H_J)

- T29 rejected general mediation; senior subset not specifically tested.
- Analysis-phase: re-run T29 authorship-score analysis on senior cohort only.

### 17.7 Posting-lifecycle / posting-age

- `posting_age_days` at 0.9% coverage (T19). Lifecycle analysis infeasible.
- Analysis-phase: drop lifecycle claims OR use external data (Indeed date_posted or expanded scrape).

### 17.8 Worker-AI benchmark vintage

- T32 benchmarks are 2024-vintage; 2025/2026-vintage will show higher worker rates.
- Analysis-phase: update benchmarks (SO 2026, DORA 2026) when available; expect widened gap.

### 17.9 External firm-maturity data (H_O)

- T16 employer-cluster labels are endogenous (derived from posting behavior).
- Analysis-phase: join external firm-maturity indices (Apptopia, BuiltWith, Dealroom AI-startup flags) for H_O test.

### 17.10 Event-study temporal granularity (H_E)

- Three snapshots with ~89-day within-2024 window cannot support event-study.
- Analysis-phase: sustained scraping across 2025-2027 for continuous panel.

---

## 18. Interview priorities (RQ4 mechanism elicitation)

### T25 artifact priorities (reading order for interviewees)

1. **Artifact 3 — Applied-AI/LLM Engineer archetype exemplars.** Single most important interview prompt. Show senior engineers real 2026 postings and ask "is this a real job?"
2. **Artifact 1 — Microsoft "Software Engineer II" pair-level rewrite.** Probes same-title rewriting mechanism. Hiring-manager informants first.
3. **Artifact 4 — Senior cluster transitions (orch vs mgmt).** Probes mechanism of the senior-IC shift from low-profile generic to orchestration-heavy.
4. **Artifact 5 — Cross-occupation divergence figure.** Probes employer-codification mechanism across SWE / ML Engineer / accountant / nurse informants.
5. **Artifact 8 — Ghost entry scope-inflation exemplars.** Probes hedging-near-AI mechanism; new-grad JDs with LLM asks.
6. **Artifact 9 — Hiring-bar contraction without bar-lowering exemplars.** Probes narrative-reallocation mechanism; challenges interpretation of requirements-section shrink.
7. **Artifact 6 — Seniority-boundary sharpening.** Probes counter-intuitive finding (AI-era ladder hardened, not flattened).
8. **Artifact 7 — Junior-share + AI-timeline.** Probes temporal attribution to model releases.
9. **Artifact 2 — Wells Fargo / Capital One exemplars.** Probes financial-services-specific mechanism.

### Informant stratification

- **Hiring managers + recruiters** at returning firms (T06 cohort): artifacts 1, 2, 8, 9.
- **Senior IC engineers** at AI-forward firms (T16 ai_forward_scope_inflator cluster): artifacts 3, 4, 6.
- **Junior / early-career engineers**: artifacts 6, 7, 8.
- **Cross-occupation informants** (ML Engineer, data scientist, accountant, nurse): artifact 5.
- **HR / JD-writing informants** (possibly with LLM JD-drafting tool adoption): artifacts 1, 9.

### Mechanism-relevant Wave 3.5 findings to foreground in interviews

1. **T22 ghost forensics** (artifact 8): what fraction of entry-level AI asks are genuine filters vs. wish-list signals?
2. **T33 hiring-bar rejection** (artifact 9): why did requirements-section shrink without lowering explicit hiring bars?
3. **T34 emergent Applied-AI role** (artifact 3): is this a real new role, a rebranding, or a skill-stacking of existing ML Engineer?
4. **T29 LLM-authorship rejection** (artifacts 1, 9): how much of the 2026 JD content-shift is recruiter-LLM-tooling-mediated?

---

## 19. Paper figures candidate list

Suggested top 5-7 figures for paper. All source paths under `exploration/figures/`.

### Paper-lead figure candidates (pick one)

1. **T32 cross-occupation divergence** — `T32_cross_occupation_divergence.png`. Single page showing employer-2024 + employer-2026 bars vs worker-benchmark diamonds across 16 subgroups. Visual: diamonds uniformly above bars. **Strongest single-figure argument for universality.**
2. **T23 within-SWE divergence chart** — `T23_divergence_chart.png`. Single-SWE version with more benchmark detail (SO/DORA/Anthropic/McKinsey). Alternative paper-lead.

### Supporting figures (4-5 recommended)

3. **T15 junior-senior divergence TF-IDF heatmap** — `T15/similarity_heatmap_tfidf.png`. 8-cell period × seniority cosine heatmap. Shows junior↔senior diverging.
4. **T20 boundary AUC comparison** — `T20/auc_by_boundary.png`. 4-bar pair (3 adjacent-level + 1 aggregate) AUC 2024 vs 2026. Sharpening visible.
5. **T34 Applied-AI archetype title distribution** — `T34/title_distribution_bars.png`. 2-cluster comparison showing 28%+ AI Engineer in cluster 0 vs 5% in cluster 1.
6. **T31 pair-level drift scatter** — `T31_drift_scatter.png`. AI Δ × breadth Δ scatter for n=23-37 pairs. Shows AI concentrated right, breadth symmetric. Paired with top-20 exemplars (Microsoft, Wells Fargo).
7. **T08 seniority ranked-change plot** — `T08/fig4_effect_ranking.png`. Effect sizes on J3/S4 across panels with CIs. Supports T30 direction consistency.

### Appendix-grade figures

- T35 network side-by-side (Louvain communities 2024 vs 2026): `T35/network_sidebyside.png`.
- T21 cluster shares: `T21/cluster_shares.png`.
- T19 timeline with model releases: `T19/fig1_timeline.png` (needs annotation upgrade).

---

## 20. Closing reconciliation

The exploration phase closes with 6 Tier A lead-candidate findings, 8 Tier B strong supporting findings, 8 Tier C methods contributions, and 5 Tier D flagged/demoted findings. Three alternative explanations for the AI rise were formally REJECTED (hiring-bar-lowering, legacy-to-AI substitution, hiring-selectivity-response); two were REJECTED as dominant mediators (LLM-authorship, sampling-frame artifact). The positive findings (A1-A6) carry the paper's narrative; the rejections carry its defense.

The strongest single novelty signal is the cross-occupation employer-worker AI-codification divergence (A1: 16/16 subgroups direction-universal 2024 AND 2026; Spearman +0.92 worker-employer rank concordance). The cleanest within-firm restructuring signal is pair-level same-title AI rewriting (A2: +10-13 pp pair-level > +7.7-8.3 pp company-level, demonstrating rewriting is same-title not title-recomposition). The cleanest negative result falsifying a pre-exploration prior is seniority-boundary sharpening (A3). The most publishable methods contribution is pattern-validation (exposed mgmt_broad 0.28 precision as an artifact of widely-used patterns) combined with the T30 multi-operationalization seniority panel.

The analysis phase inherits 5 deferred hypotheses from the T24-planned list (H_D highest priority; H_I/H_J medium; H_E/H_F/H_G low) plus 3 new post-3.5 hypotheses (H_O digital-maturity × AI-rewriting, H_P Applied-AI in financial-services, H_Q J3 hedging-near-AI). Analysis-phase priorities are (i) re-derive T16/T23 under v1_rebuilt pattern, (ii) pre-register pair-panel construction for T31-style tests, (iii) formalize H_D senior-IC-as-team-multiplier with external hiring panel data, (iv) test H_O with external firm-maturity data, (v) design RQ4 interview protocol using T25 artifacts.

This SYNTHESIS.md is the paper's analytical backbone and the primary input to Agent P (Wave 5) for the presentation.

*T26 ends. Wave 5 dispatch follows.*
