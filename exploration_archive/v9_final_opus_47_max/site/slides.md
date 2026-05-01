---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Inter', 'Helvetica', sans-serif;
    padding: 50px 70px;
    font-size: 24px;
    background: #fbfbfb;
  }
  section.lead {
    text-align: center;
  }
  h1 {
    color: #1a237e;
    font-size: 34px;
    line-height: 1.2;
  }
  h2 {
    color: #283593;
    font-size: 28px;
  }
  h3 {
    color: #303f9f;
  }
  strong { color: #d84315; }
  em { color: #1565c0; font-style: normal; }
  section.sectionslide {
    background: #1a237e;
    color: #fff;
    text-align: center;
  }
  section.sectionslide h1 { color: #fff; font-size: 44px; }
  section.sectionslide p { color: #c5cae9; font-size: 22px; }
  table { font-size: 20px; }
  .big { font-size: 48px; font-weight: 700; color: #d84315; }
  .sub { font-size: 18px; color: #555; }
  footer { font-size: 14px; color: #999; }
---

<!-- _class: lead sectionslide -->

# Between 2024 and 2026, the same firms rewrote the same SWE titles around AI

## An evidence-layered exploration of 68,137 LinkedIn postings

<p class="sub">34 task reports &middot; 9 sensitivity dimensions &middot; 2 adversarial verifications &middot; 5 rejected alternatives</p>

---

# If AI tooling rose in software-engineering postings, we could not tell what caused it

- Between 2024 and 2026, AI tooling appears more often in SWE JDs. That much was clear from the raw corpus.
- But five rival explanations were all plausible: recruiter-LLM authorship, hiring-bar lowering, hiring-selectivity during the JOLTS trough, legacy-to-AI role substitution, and sampling-frame composition.
- Prior labor-market studies had not resolved these. Without resolving them, no cleanly-identified claim about the SWE labor market is possible.
- **This exploration tests and rejects all five alternatives, isolates within-firm same-title rewriting, and documents two emergent senior archetypes.**

---

# We built one dataset, three panels, and a ten-stage preprocessing pipeline

- **68,137** SWE LinkedIn rows under `is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok'`.
- **Three panels**: arshkon 2024-04 (4,691), asaniczka 2024-01 (18,129), scraped 2026-03+ (45,317 and growing).
- **~791-day span** covering ~8 major frontier-model releases.
- **Ten-stage pipeline**: eight deterministic + two LLM stages. LLM produces `description_core_llm`, `seniority_final` (where routed), `yoe_min_years_llm`, `ghost_assessment_llm`, `swe_classification_llm`.
- **Seven analytical samples**: full frame, returning-cohort (2,109 firms), 3 overlap panels, pair-level panel (T31), labeled-only subset, archetype panel, exemplar sample.

---

<!-- _class: lead -->

## Headline 1 of 6

---

# Across 16 occupation subgroups, workers use AI far more than employers code for it, in both 2024 and 2026

![bg right:50% w:95%](figures/T32_cross_occupation_divergence.png)

- **Universal gap**: 16 of 16 tested subgroups show workers above employers, in both periods.
- **Spearman +0.92**: employers rank occupations identically to workers, at 10 to 30% of worker levels.
- **SWE DiD**: +14.02 pp [+13.67, +14.37] vs control +0.17 pp.
- **72x ratio gap for accountants**; 0.00% employer rate for nurses on 6,801 postings.

<p class="sub">X-axis: 16 subgroups. Y-axis: 2026 AI-mention rate. Bars = employer posting rate; diamonds = worker-use benchmark.</p>

---

# At the same firm, at the same title, AI mentions rose more than the firm's average

- **T16 company-level**: within-co AI +7.7 to +8.3 pp across three overlap panels (n=125-356 firms).
- **T37 returning-cohort** (n=2,109 firms): within-co AI +7.91 pp — matches T16 within 0.45 pp.
- **T31 same-company same-title pair level** (n=23 arshkon_min3): AI drift **+10 to +13 pp**.
- **Pair > company** is the cleanest test that rewriting is same-title, not title-recomposition.
- **Exemplar**: Microsoft "Software Engineer II", n=35 in 2026 vs n=6 in 2024 — AI-strict delta +40 pp, citing Copilot / Generative AI / AI Systems responsibilities.

---

# Junior and senior postings became more distinct, not less — seniority boundaries sharpened

![bg right:55% w:95%](figures/T20/auc_by_boundary.png)

- **T15 TF-IDF junior-senior centroid cosine**: 0.946 → 0.863 (diverging, Δ = -0.083).
- **T20 supervised AUC**: associate ↔ mid-senior **+0.150** (cleanest signal).
- **T12 relabeling diagnostics**: label-based AND YOE-based agree — 2026 entry ≠ 2024 senior.
- **Falsifies** the pre-exploration prior "junior roles relabeled as senior."

<p class="sub">Four adjacent-boundary AUC pairs (2024 vs 2026). The associate-to-mid-senior bar rises from 0.743 to 0.893.</p>

---

# Scope broadened across all seniority, but senior more than junior — within-domain

- **Length-residualized requirement breadth**: J3 +1.58, S4 **+2.61**.
- **Within-company**: S4 +1.97 > J3 +1.43.
- **T28 decomposition**: 60 to 85% of the scope rise is **within-domain**, not between-domain composition.
- **T33 rejects** hidden-hiring-bar lowering: |ρ| <= 0.28 on all proxies; 0 of 50 postings sampled contain explicit loosening language.
- Scope broadening is demand-side content change, not screening relaxation.

---

# Two emergent senior archetypes with content-grounded names

![bg right:45% w:95%](figures/T34/title_distribution_bars.png)

**Cluster 0 — Senior Applied-AI / LLM Engineer**
- 15.6x growth (144 → 2,251), 94% in 2026.
- Median YOE **6** (+1 vs other senior clusters).
- Distinguishing bigrams: *claude code*, *rag pipelines*, *langchain llamaindex*, *augmented generation rag*.

**Cluster 1 — Senior Data-Platform / DevOps Engineer**
- 2.6x growth; +11.4 pp senior share.
- `pipelines/sql/etl` 2.62x over-rep; `kubernetes/terraform/cicd` 2.11x over-rep.

<p class="sub">AI-senior roles ask MORE experience, contradicting the "AI lowers the bar" prior.</p>

---

<!-- _class: lead sectionslide -->

# What we tested and rejected

## Five alternative explanations, ruled out computationally

---

# LLM-authored JDs were not the dominant mediator

- Hypothesis: recruiters adopted LLMs to write JDs, inflating AI-mention rates.
- **Test** (T29): re-measure content effects on the bottom quartile by signature-vocabulary density.
- **Result**: AI-binary +0.115 vs full +0.131 (**88% preserved**); tech count +1.99 vs +2.06 (**97%**); credential J3 +16.6 pp vs +16.9 pp (**97%**).
- **Length growth IS half-LLM-mediated** (+583 vs +1,130 chars, **52%**) — boilerplate mechanism.
- But content signal is real; LLM-authorship is not the story.

---

# Hiring-bar lowering rejected — requirements did not correlate with easing any bar

| Metric | Hiring-bar-lowering predicts | Observed |
|---|---|---|
| Requirements-section Δ | Large shrink | Classifier-sensitive (direction flips) |
| YOE ask Δ | Declining | Null (|ρ| <= 0.09) |
| Credential stack Δ | Declining | **Rising** (+13 to +17 pp) |
| Tech-count Δ | Declining | **Rising** (+2.06) |
| Explicit loosening language | Present | **0 of 50 samples** |

- Δ(req_chars) × Δ(desc_length) r ≈ +0.35 supports "narrative expansion" over "hiring bar lowering."

---

# Legacy-to-AI substitution rejected — legacy titles mapped to modern stacks, not AI roles

- T36 attempted to map disappearing 2024 titles (Java/Drupal/PHP/.NET architect) to 2026 neighbors via TF-IDF cosine.
- 6 of 11 credibly matched at cosine [0.30, 0.59].
- **2026 neighbors' AI-strict rate: 3.6%** — below the 14.4% market rate.
- Content drift: SSIS / Unix-scripts / Drupal / PHP / Scala leave; Postgres / pgvector / CI-CD / microservices / Terraform / ArgoCD arrive.
- Substitution is legacy-stack-to-modern-stack, not legacy-to-AI.

---

# Sampling-frame artifact rejected — 13 of 15 headlines robust on returning firms

- T37 restricted every Wave 2/3 headline to the 2,109-firm **returning cohort** (55% of 2026 postings).
- **13 of 15 retain retention ratio >= 0.80.**
- **0 of 15 are sampling-frame-driven.**
- J3 and S4 directions **INTENSIFY** on returning-only:
  - J3 pooled: +5.05 pp → +6.17 pp (ratio 1.22).
  - S4 pooled: -7.62 pp → -8.29 pp (ratio 1.09).
- Within-co AI-strict +7.91 pp on the returning cohort matches T16 range within 0.45 pp.

---

# Management language was flat — the Wave 2 "decline" was a 0.28-precision-pattern artifact

| Report | Pattern | Precision | Claim |
|---|---|---|---|
| T11 (Wave 2) | `mgmt_broad` | **0.28** | "management density fell" |
| T11 (Wave 2) | `mgmt_strict` | 0.55 | "management density fell" |
| T21 (V1-corrected) | `mgmt_strict_v1_rebuilt` | **0.98 to 1.00** | **"management density FLAT"** |

- T21 mgmt_rebuilt density: mid-senior 0.039 to 0.038; director 0.031 to 0.026.
- V2 independently replicated: 0.034 to 0.034.
- **Publishable methods-caveat on longitudinal posting studies.**

---

<!-- _class: lead sectionslide -->

# Methods contributions

## Three frameworks the analysis phase inherits

---

# T30 multi-operationalization seniority panel — 12 of 13 definitions direction-consistent

- **13 seniority definitions** x **6 analysis groups** = 78 rows in the canonical panel.
- **Junior side: 7 of 7 definitions UP.** J1/J2/J3/J4/J5/J6/J3_rule.
- **Senior side: 5 of 6 definitions DOWN.** S1/S3/S4/S5/S4_rule; S2 (director-only, <1% of SWE) flat at noise.
- **Primary**: J3 (YOE LLM <= 2) + S4 (YOE LLM >= 5), pooled-2024 baseline, arshkon-only co-primary for senior (asaniczka asymmetry).
- Reusable for other longitudinal posting studies; shipped as `seniority_definition_panel.csv`.

---

# V1 pattern validation exposed a 0.28-precision artifact that a generation of papers may share

- Protocol: stratified 50-row sample per pattern; manual TP/FP labeling by span meaning; per-sub-pattern and per-period precisions.
- **Threshold**: >= 0.80 precision for PRIMARY use; >= 0.60 for DIAGNOSTIC; < 0.60 FAIL.
- `mgmt_broad` precision **0.28** (all 4 broad tokens fail).
- `mgmt_strict` precision **0.55** (hire 0.07, performance_review 0.25 worst sub-patterns).
- `ai_broad` had direct within-2024 contamination (`MCP` = Microsoft Certified Professional).
- **T22 extended the pattern set to 7 validated patterns at 0.92+ precision.** Shipped as `validated_mgmt_patterns.json`.

---

# Within-2024 SNR calibration separates content signals from instrument artifacts

| Metric | Within-2024 noise | Cross-period effect | SNR | Verdict |
|---|---|---|---|---|
| ai_mention_strict | +0.004 | +0.133 | **32.9** | above_noise |
| scope_term_rate | +0.005 | +0.210 | **42.8** | above_noise |
| J3 share | +4.75 pp | +5.04 pp | 1.06 | near_noise |
| S4 share | +7.09 pp | +7.59 pp | 1.07 | near_noise |
| mgmt_broad prevalence | +0.062 | +0.028 | 0.45 | **below_noise** |

- AI and scope are clean cross-period signals; seniority shares are near-noise for annualized rates — use raw pp deltas.
- `mgmt_broad` below noise corroborates V1's pattern-precision failure.

---

# Within-vs-between decomposition + returning-cohort sensitivity are the robustness centerpieces

- **Returning-cohort** (2,109 firms, 55% of 2026 postings): 13 of 15 headlines ratio >= 0.80.
- **T28 within-domain vs between-domain**: 60 to 85% of scope rise is WITHIN-domain across 5 metrics.
- **J3 entry-share rise** is +6.84 pp within-domain vs -0.02 pp between-domain — **not** domain recomposition.
- **S4 within-firm decline** (-9.31 pp) EXCEEDS aggregate (-8.29 pp) — parallel "exit-driven senior decline" to T16's "exit-driven junior rise."

---

<!-- _class: lead sectionslide -->

# What the evidence means

---

# Paper positioning — hybrid empirical + methods with A1 and A2 as co-lead claims

- **Empirical lead 1 (A1)**: cross-occupation employer-worker AI-codification divergence is universal, SWE-specific in magnitude.
- **Empirical lead 2 (A2)**: within-firm same-title rewriting is real, +10 to +13 pp pair-level, exceeding company-level.
- **Supporting claims**: sharpened seniority (A3), universal scope inflation senior>junior (A4), two emergent senior archetypes (A5), AI-DiD robust across control defs (A6).
- **Methods contributions** (25 to 30% of paper): T30 seniority panel, pattern validation, SNR calibration, within-vs-between decomposition.

---

# RQ1 reframes; RQ3 strengthens; RQ2 narrows; RQ4 interview priorities are pre-cleared

- **RQ1 revised**: SWE restructured around technology domain + AI-enabled orchestration. Seniority **sharpened**; scope broadened universally with senior-tier larger shifts. Applied-AI/LLM Engineer archetype grew 15.6x.
- **RQ2 revised**: AI appears at junior AND senior comparably; no downward migration; universal broadening.
- **RQ3 extended**: cross-occupation universality (16/16 subgroups); SWE DiD +14 vs control +0.17.
- **RQ4**: three alt mechanisms (LLM-authorship, hiring-bar, selectivity) computationally rejected → interviews can focus on: same-title rewriting experience, Applied-AI role reality vs rebranding, senior-IC shift to orchestration.

---

# Nine interview artifacts designed for three informant strata

| Artifact | For whom | Mechanism probe |
|---|---|---|
| 3. Applied-AI / LLM Engineer exemplars | Senior IC engineers | Is this a real new role or rebranding? |
| 1. Microsoft "SE II" pair rewrite | Hiring managers, recruiters | What drove the same-title rewrite? |
| 4. Senior cluster transitions | Senior IC engineers | Orch vs mgmt mechanism in lived work? |
| 5. Cross-occupation divergence | Cross-occupation informants | Why under-specify AI in JDs? |
| 8. Ghost entry exemplars | Early-career engineers | Wish-list vs real filter? |
| 9. Hiring-bar contraction | Hiring managers | Narrative reallocation mechanism? |

<p class="sub">Artifacts 2, 6, 7 add financial-services context, seniority-sharpening, and AI-timeline prompts.</p>

---

# Six deferred hypotheses set the analysis-phase roadmap

| ID | Hypothesis | Priority | Needs |
|---|---|---|---|
| H_D | Senior-IC as team-multiplier | **HIGH** | External firm hiring-panel data |
| H_O | Digital-maturity x AI-rewriting | **HIGH** | External firm-maturity index |
| H_P | Applied-AI in financial-services | MEDIUM | Regulatory context + interviews |
| H_Q | J3 hedging-near-AI | MEDIUM | Data exists; formal test |
| H_I | AI as coordination signal | MEDIUM | Mechanism-test design |
| H_J | Recruiter-LLM senior bias | MEDIUM | Re-run T29 on senior cohort |

---

<!-- _class: lead sectionslide -->

# Closing

---

# This exploration is an argument for cleaning the instrument before measuring the shift

- Pre-exploration RQ1 ("junior share reduces", "junior relabeled as senior", "management declining") was **contradicted** by its own data — but only after fixing a 0.28-precision pattern and building a 13-definition seniority panel.
- The **rejections are the strength**: hiring-bar, legacy-to-AI, selectivity, LLM-authorship, sampling-frame, management-decline — five rival stories eliminated leaves a clean empirical claim.
- The **within-firm same-title rewriting** finding is cleanly identified because everything else has been controlled or ruled out.
- Measurement fidelity is upstream of every longitudinal content claim. Studies that skip pattern validation will confabulate shifts in the direction of whatever semantic neighbors dominate in the second period.
- **What the data says**: the same firms, at the same titles, rewrote SWE postings around AI tooling and platform-infrastructure orchestration between 2024 and 2026. Why they did is RQ4.
