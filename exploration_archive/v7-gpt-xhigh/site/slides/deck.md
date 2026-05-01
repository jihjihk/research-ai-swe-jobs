---
marp: true
theme: default
paginate: true
title: AI-era skill-surface expansion in technical job postings
description: Wave 5 T27 evidence package slide deck
style: |
  :root {
    font-family: Inter, Arial, sans-serif;
    color: #1f2523;
  }
  section {
    background: #fbfcfb;
    color: #1f2523;
    padding: 54px 64px;
  }
  h1 {
    font-size: 34px;
    line-height: 1.14;
    letter-spacing: 0;
    color: #16201d;
  }
  h2 {
    font-size: 24px;
    color: #0b7f75;
    letter-spacing: 0;
  }
  p, li {
    font-size: 22px;
    line-height: 1.34;
  }
  table {
    font-size: 19px;
  }
  th {
    background: #edf2f0;
  }
  strong {
    color: #0b7f75;
  }
  .small {
    font-size: 17px;
    color: #5c6661;
  }
  .kicker {
    color: #5c6661;
    font-size: 24px;
  }
  .claim {
    border-left: 5px solid #0b7f75;
    padding-left: 20px;
    margin-top: 28px;
  }
  .warn {
    border-left: 5px solid #b4332f;
    padding-left: 20px;
    margin-top: 28px;
  }
  .twocol {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 28px;
    align-items: start;
  }
  .center-img img {
    max-height: 410px;
    object-fit: contain;
  }
  footer {
    color: #5c6661;
    font-size: 15px;
  }
---

# The exploration changed the paper from junior collapse to skill-surface expansion.

<p class="kicker">Wave 5 Task T27 evidence package</p>

The strongest paper is now about how AI/tool/platform/workflow language diffused through software-producing technical job ads.

<!-- _footer: "Evidence boundary: SYNTHESIS, Gate 3" -->

---

# The evidence supports posting-content restructuring, not employment effects.

| We observe | We do not observe |
|---|---|
| Job-ad language | Filled jobs |
| Employer labels and stated YOE | Actual worker seniority |
| AI/tool/platform mentions | Screened hiring bars |
| Company, metro, and occupation patterns | Causal impact of AI on employment |

<!-- _footer: "See: Methods / What we can and cannot claim" -->

---

# The initial junior-collapse frame weakened under the seniority panel.

T30 found J1-J4 junior definitions rising from pooled 2024 to scraped 2026, while T16 found common-company entry labels down but low-YOE floors up.

<div class="warn">
The result is construct disagreement, not a clean junior disappearance story.
</div>

<!-- _footer: "Evidence: T30, T16, Gate 1, Gate 3" -->

---

# Returning companies sharply expanded AI and tool language.

| Metric in common companies | 2024 | 2026 | Change |
|---|---:|---:|---:|
| Broad AI prevalence | 3.74% | 23.23% | +19.49 pp |
| AI-tool strict prevalence | 2.23% | 19.80% | +17.57 pp |
| Tech count | 5.45 | 7.19 | +1.74 |
| Requirement breadth, LLM text | 7.34 | 9.25 | +1.91 |

<!-- _footer: "Evidence: T16. Details: findings/within-company-ai-tool-expansion/" -->

---

# Most returning-company AI/tool change happened within firms.

![Common-company decomposition](assets/figures/T16/common_company_decomposition.png)

<!-- _footer: "Evidence: T16 common-company decomposition" -->

---

# New 2026 entrants do not explain away the expansion.

| Scraped 2026 company type | Broad AI | AI-tool strict | Requirement breadth |
|---|---:|---:|---:|
| New entrants | 25.44% | 21.79% | 9.39 |
| Returning firms | 22.28% | 18.90% | 9.38 |

<p class="small">New entrants are somewhat more AI-heavy, but returning firms also moved sharply and have nearly identical LLM-text requirement breadth.</p>

<!-- _footer: "Evidence: T16 new entrant profile" -->

---

# The expansion is SWE-amplified, not SWE-only.

| Metric | SWE change | Adjacent change | Control change |
|---|---:|---:|---:|
| Broad AI | +27.2 pp | +27.0 pp | +2.6 pp |
| AI-tool | +21.8 pp | +19.1 pp | +0.8 pp |
| Bounded tech count | +2.36 | +2.31 | +0.14 |
| Requirement breadth | +2.56 | +2.93 | +0.70 |

<!-- _footer: "Evidence: T18. Details: findings/adjacent-parallel-expansion/" -->

---

# Adjacent roles absorbed much of the same AI/platform surface.

![AI gradient by occupation](assets/figures/T18/ai_gradient_by_occupation.png)

<!-- _footer: "Evidence: T18 occupation gradient" -->

---

# AI/tool expansion reached every eligible metro.

| Metric | Mean change | Minimum | Positive metros |
|---|---:|---:|---:|
| Broad AI prevalence | +17.6 pp | +8.7 pp | 26 / 26 |
| AI-tool strict prevalence | +16.1 pp | +6.7 pp | 26 / 26 |
| Requirement breadth | +2.14 | +0.53 | 26 / 26 |
| Mean tech count | +2.00 | +0.31 | 26 / 26 |

<!-- _footer: "Evidence: T17. Details: findings/geographic-diffusion/" -->

---

# Geography looks like diffusion, not a tech-hub-only shock.

![Metro heatmap](assets/figures/T17/metro_metric_delta_heatmap.png)

<!-- _footer: "Evidence: T17 metro metric deltas" -->

---

# Requirement breadth rose across junior and senior definitions.

| Definition | Requirement breadth change |
|---|---:|
| J1 entry label | +1.74 |
| J2 entry/associate | +1.84 |
| J3 YOE <= 2 | +1.75 |
| J4 YOE <= 3 | +2.08 |
| S1 mid-senior/director | +2.40 |
| S4 YOE >= 5 | +2.61 |

<!-- _footer: "Evidence: T11, V1. Details: findings/requirement-tech-breadth/" -->

---

# The technology surface expanded around AI, Python, APIs, and platform work.

| Technology | 2024 | 2026 | Change |
|---|---:|---:|---:|
| CI/CD | 15.4% | 33.6% | +18.2 pp |
| Python | 32.3% | 49.4% | +17.0 pp |
| API design | 13.0% | 27.4% | +14.4 pp |
| Observability | 1.9% | 13.9% | +12.0 pp |
| LLM | 1.0% | 13.0% | +12.0 pp |

<!-- _footer: "Evidence: T14, V1" -->

---

# AI is integrating into existing stacks rather than replacing them.

![AI integration stack](assets/figures/T14/ai_integration_stack.png)

<p class="small">T14 cautions that AI-tool rows are longer, so raw count advantages should not be read as density growth.</p>

<!-- _footer: "Evidence: T14 AI integration stack" -->

---

# Open-text changes point to workflow, tooling, ownership, exposure, and AI agents.

T12's primary LLM-cleaned, non-aggregator, company-capped comparison surfaces 2026-heavy terms such as:

`workflows`, `pipelines`, `familiarity`, `hands-on`, `tooling`, `observability`, `ownership`, `ai/ml`, `agent`, `llm`, `rag`.

<!-- _footer: "Evidence: T12 open-ended text evolution" -->

---

# The text signal is not only benefits, legal text, or company boilerplate.

T13 finds cleaned-text growth concentrated in role summary, responsibilities, requirements, and unclassified job content. Benefits/about/legal contribute only about +1 character in the cleaned-text primary.

<div class="claim">
Raw scraped descriptions contain substantial boilerplate, which is why LLM-cleaned text remains the primary source for text-sensitive claims.
</div>

<!-- _footer: "Evidence: T13, V1" -->

---

# Domain and technology structure dominate seniority structure.

| Structure test | Domain/archetype | Seniority |
|---|---:|---:|
| T09 NMI | 0.205 | 0.0069 |
| T15 embedding eta-squared | 0.111 | 0.007 |
| T15 TF-IDF/SVD eta-squared | 0.132 | 0.007 |

<!-- _footer: "Evidence: T09, T15, V1. Details: findings/domain-technology-structure/" -->

---

# Generic junior-senior semantic convergence was rejected.

| Representation | Arshkon 2024 | Scraped 2026 | Shift | Calibration verdict |
|---|---:|---:|---:|---|
| Embedding centroid similarity | 0.954 | 0.959 | +0.005 | Below within-2024 +0.022 |
| TF-IDF/SVD centroid similarity | 0.884 | 0.842 | -0.042 | Reject |

<!-- _footer: "Evidence: T15, V1" -->

---

# Entry labels and low-YOE floors are different labor-market signals.

In scraped 2026 low-YOE rows (`yoe_extracted <= 2`):

| Seniority label | Share |
|---|---:|
| Unknown | 72.4% |
| Mid-senior | 21.5% |
| Entry | 5.5% |

<!-- _footer: "Evidence: T08, T16, T30. Details: findings/seniority-label-yoe-divergence/" -->

---

# Senior roles broadened, but management-decline is unsupported.

T11 finds senior requirement breadth rising under S1 and S4. Strong management/mentorship indicators rise, while direct reports, performance reviews, hiring, and headcount remain rare.

<div class="claim">
The supported senior story is broader complexity plus mentorship and coordination, not a clean shift away from management.
</div>

<!-- _footer: "Evidence: T11. T21 senior deep dive is missing." -->

---

# Several attractive claims remain outside the evidence.

| Missing task | Claim it blocks |
|---|---|
| T19 | Temporal-rate stability and annualized flow claims |
| T20/T21 | Formal seniority-boundary and senior-role redefinition claims |
| T22 | Ghost, aspirational, template, and screened-force claims |
| T23 | Employer-worker usage divergence claims |
| V2 | Independent verification of T16/T17/T18 |

<!-- _footer: "Evidence boundary: SYNTHESIS, Gate 3, INDEX" -->

---

# The claims we can make are posting-language claims.

Use:

<div class="claim">
Postings broadened around AI/tool/platform/workflow requirements across software-producing technical work.
</div>

Avoid:

<div class="warn">
AI eliminated junior SWE jobs, AI requirements are screened hiring bars, or employer requirements outpace worker usage.
</div>

<!-- _footer: "Details: methods/can-and-cannot-claim/" -->

---

# The next analysis should estimate expansion formally and recover requirement force.

High-value next steps:

1. Company fixed-effect models for AI/tool, tech breadth, and requirement breadth.
2. Cross-occupation DiD models with alternative adjacent-role definitions.
3. Metro diffusion models with domain-mix controls.
4. T22-style force classification for required, preferred, aspirational, template, and branding language.

<!-- _footer: "Evidence: SYNTHESIS recommended analysis priorities" -->

---

# The constructive next step is to test mechanisms, not revive the old headline.

The publishable contribution is a transparent longitudinal posting dataset plus a measurement framework showing how AI-era requirements diffused through technical job ads while seniority labels became unreliable anchors.

<!-- _footer: "Evidence: SYNTHESIS honest paper positioning" -->
