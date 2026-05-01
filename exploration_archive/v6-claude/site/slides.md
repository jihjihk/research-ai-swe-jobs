---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 26px;
    padding: 50px 70px;
  }
  h1 {
    font-size: 38px;
    color: #1a1a1a;
    line-height: 1.2;
  }
  h2 {
    font-size: 30px;
    color: #1a1a1a;
  }
  section.title h1 {
    font-size: 54px;
  }
  section.section h1 {
    font-size: 48px;
    text-align: center;
    margin-top: 30%;
  }
  section.big h1 {
    font-size: 44px;
    text-align: center;
    margin-top: 25%;
  }
  img {
    max-height: 480px;
    display: block;
    margin: 0 auto;
  }
  .small { font-size: 20px; color: #555; }
  .cite { font-size: 18px; color: #888; font-style: italic; }
  table { font-size: 22px; }
  em { color: #c0392b; }
---

<!-- _class: title -->

# What the 2024-2026 SWE labor market actually did

### A four-finding tour of 63,701 job postings, 26 tasks, and one inverted research question

Agent P · Wave 5 synthesis · 2026-04-15

---

# Most people assume employers demand AI skills faster than workers adopt them. The data says the opposite.

- **Worker side (Stack Overflow 2025):** 80.8% of professional developers use AI tools in their daily work.
- **Employer side (2026 SWE LinkedIn postings):** 28.6% mention AI at all, and only **6.0%** list it as a hard requirement.
- **Gap:** −52 percentage points on broad AI. Worker-to-posting ratio **0.35**.

<span class="cite">T23 Section 2 · T14 broad-union · V2 re-derivation Section 2</span>

---

# This talk has one lead finding, one co-headline, one structural finding, and one null result

1. **Posting language *lags* worker AI adoption by ~10×.** The paper's RQ3 direction inverts.
2. **Senior roles specialized toward hands-on technical orchestration, not people management.** Clean and AI-linked.
3. **The AI tool-stack restructuring is field-wide, 92% within-company, and bifurcated** into existing-employer adopters and a new-entrant LLM/GenAI wave.
4. **The junior-narrowing story does not survive robustness checks.** Length growth is style migration. Seniority boundaries *sharpened*.

<span class="cite">Each of the next 20 slides is one point from this list.</span>

---

<!-- _class: section -->

# Part 1 — The data frame

---

# Three posting sources combine into one comparable 2024-vs-2026 panel

| Source | Role | Platform | SWE rows |
|---|---|---|---|
| Kaggle arshkon | 2024 historical snapshot | LinkedIn | 4,691 |
| Kaggle asaniczka | 2024 historical snapshot | LinkedIn | 18,129 |
| Scraped (2026-03-20 onwards) | 2026 growing window | LinkedIn + Indeed | 40,881 |

Default frame: **`is_swe = TRUE ∧ source_platform = 'linkedin' ∧ is_english ∧ date_flag = 'ok'`**. Total SWE: **63,701**.

<span class="cite">T01 · T05 · T07 · default filter set documented in SYNTHESIS Section 4.</span>

---

# A 10-stage preprocessing pipeline turns 3 raw sources into 1 analysis dataset

Rule-based stages (1-8) run in ~30 minutes deterministically; LLM stages (9-10) run for hours and produce the cleaned-text column and final seniority labels.

![w:900](assets/figures/T08/description_length_by_period.png)

The figure shows raw description length by period — a 17% increase that later gets decomposed into what is real and what is style migration.

<span class="cite">See Methodology → Preprocessing pipeline for the full stage reference plus the two LLM prompts verbatim.</span>

---

# The sample frame validates against BLS at r=0.97 across 18 metros

<span class="cite">T07 / T17</span>

- **Pearson r = 0.97** between BLS 15-1252 software-developer employment and our sample metro counts.
- 18 metros qualify at n ≥ 50 SWE per period. 2024 AI rate was near-zero everywhere; 2026 rose in **all 18 metros** (+0.07 Detroit to +0.30 Seattle).
- The sample is not metro-biased toward AI hotspots — AI rose *everywhere* technical work is posted.

---

<!-- _class: section -->

# Part 2 — The AI explosion is real, structural, and field-wide

---

# The strongest signal in the entire corpus is a narrow AI keyword rate that rose from 2.81% to 18.78% at SNR 925

<span class="cite">T05 · T14 · V1 correction 1</span>

| Metric | 2024 → 2026 | SNR |
|---|---|---|
| **AI narrow** (LIKE '%ai%') | **2.81% → 18.78%** | **925** |
| AI broad (24-term union) | 5.15% → 28.63% | 13.3 |
| claude_tool | 0.01% → 3.37% | 326 |
| copilot | 0.06% → 3.77% | 44 |
| langchain | 0.11% → 3.10% | 36 |

**Cite narrow and broad cells separately — they measure different things.** Within-2024 calibration rules out that this is noise.

---

# The technology co-occurrence network became *more* modular, not less, with two new AI communities

![w:950](assets/figures/T14/cooccurrence_network_2024_2026.png)

Network modularity **0.56 → 0.66** (Louvain). 12 → 15 communities. Two are new: a **LLM/RAG application cluster** (17 techs: langchain, langgraph, rag, openai_api, mcp...) and an **AI-tools triad** (copilot, cursor, claude_tool). AI does not blur the tech stack — it adds structured sub-regions.

<span class="cite">T14 Section 5</span>

---

# A difference-in-differences shows the AI explosion is field-wide, not SWE-specific

![w:900](assets/figures/T18/ai_adoption_gradient.png)

- **SWE vs control:** +29.6 pp broad AI, CI [28.9, 30.4]
- **Adjacent vs control:** +27.5 pp
- **SWE vs adjacent:** **+2.1 pp**, CI [0.77, 3.38] — doesn't cross zero, but modest.

The cleanest single case: network engineer 4.2% → 16.6% with zero SWE reclassification. AI is not eating SWE; it is eating information-tech.

<span class="cite">T18 · V2 Alt 5</span>

---

# The AI rise is 92% within-company on the 240-company overlap panel, and survives against JOLTS macro noise at 24.7×

<span class="cite">T16 · T19 · T29</span>

- **Overlap panel broad AI:** 4.02% → 26.93% (+22.91 pp), **21.03 pp within-company (92%)**.
- **Macro-robustness ratio** (cross-period / within-scraped-window): **24.7×** — above the 10× noise floor by a factor of 2.5.
- **Authorship-style matched** attenuation: **0–7%** — the rise is real content, not recruiter-LLM drafting artifact.

Among the 186 panel companies with zero AI in 2024, within-company AI rose +0.229 by 2026. Existing employers retooled — it isn't a new-entrant composition effect.

---

<!-- _class: section -->

# Part 3 — But workers are ~10× ahead of postings

---

# Workers adopted AI tools at ~81% while 2026 postings mention AI at 29% and list it as a requirement at 6%

![w:900](assets/interview/inverted_rq3_divergence.png)

The figure shows the divergence at three layers: broad AI mention (28.6%), narrow LLM-labeled (34.6%), and hard AI requirement (6.0%), against the Stack Overflow 2025 professional-developer any-use rate (80.8%). The ratio on broad AI is **0.35** — postings are at roughly one-third of worker rates.

<span class="cite">T23 · SO Developer Survey 2025 · hard-requirement subset from T22 section test</span>

---

# The inversion survives every sensitivity check from a 50% worker rate floor up to 85%

![w:900](assets/figures/T23/divergence_sensitivity.png)

Even at the lowest plausible Stack Overflow worker rate (50% — more pessimistic than Anthropic, Accenture, GitClear, SO blog cross-checks), the gap is still −21 pp. **Direction robust.** The broad-AI worker-to-posting ratio never exceeds 0.56 across the full sensitivity band.

<span class="cite">T23 · V2 Section 2 re-derivation</span>

---

# Employers describe AI as what you'll *do*, not as what you must *have* — its share in `preferred` sections tripled

<span class="cite">T22 Section 3 · T23 ghost cross-check</span>

- Only **21-24%** of AI mentions land in the `requirements` section, vs **34-39%** for non-AI tech (lift −13 to −16 pp).
- AI's share in the `preferred` section **tripled**: 2.4% → 7.9%.
- **Hard AI requirement rate: 6.0%** in 2026 SWE postings. That's a 13.5× worker-to-posting gap at the requirement layer.

**Reframe:** postings are a *lagging* indicator of workplace AI adoption. Employer HR templates update slower than developer toolchains.

---

<!-- _class: section -->

# Part 4 — Senior roles specialized toward hands-on technical orchestration

---

# Mid-senior technical-orchestration density rose 98% and director orchestration rose 156% while director people-management fell 21%

![w:850](assets/figures/T21/density_profile_shift.png)

Density per 1K chars, validated patterns only (100% precision on a 100-row audit):

| | Mid-senior 2024 → 2026 | Director 2024 → 2026 |
|---|---|---|
| People mgmt | 0.186 → 0.232 (+25%) | 0.228 → 0.181 (**−21%**) |
| **Technical orchestration** | 0.168 → **0.332 (+98%)** | 0.118 → **0.302 (+156%)** |

<span class="cite">T21 Section 2 · V2 Section 7 precision audit</span>

---

# The AI × senior interaction is entirely in the orchestration profile — people-management density is identical regardless of AI

![w:900](assets/figures/T21/ai_senior_interaction.png)

- **2024:** AI-mentioning senior postings had identical profile density to non-AI senior postings.
- **2026:** AI-mentioning senior postings have **orch density 0.482 vs 0.274 non-AI** (+76% uplift).
- People-management density: 0.230 vs 0.232 — **identical, sign of zero**.

AI-mentioning 2026 senior postings are tech-leads and orchestrators, not people managers.

<span class="cite">T21 · V2 re-derivation at +73%</span>

---

# Directors were recast: the `tech_count` coefficient flipped sign and directors now mention *more* technologies than mid-senior postings

<span class="cite">T20 feature-importance Section 4</span>

- Mid-senior → director `tech_count` coefficient: **−0.48 → +0.35** (sign flip).
- Director `tech_count` rose **2.93 → 8.03 (+173%)** — the largest per-cell shift in the whole feature heatmap.
- The tech-lead sub-archetype doubled: **7.8% → 16.9%** (people-manager cluster flat at 14.5% → 14.7%).
- 2024 directors were people bosses. 2026 directors are tech orchestrators.

This is the single cleanest within-corpus direct evidence of a role redefinition, and it is localized, AI-linked, and survives top-10-company exclusion (+120% vs +156%).

---

<!-- _class: section -->

# Part 5 — The tool-stack restructuring is bifurcated

---

# A 19% tool-stack adopter cluster — consulting and system integrators, no FAANG — rewrote their templates toward AI vocabulary

![w:900](assets/figures/T16/cluster_heatmap_k4.png)

<span class="cite">T16 Section 4 · V2 Alt 3</span>

- **46 of 240** overlap-panel companies cluster as tool-stack adopters: ΔAI +0.523, Δdesc length +1,149 chars, Δscope +0.328, entry share flat.
- Dominant members: **AT&T, Deloitte, American Express, Aditi Consulting, Adobe, Macquarie.** Adobe is the most tech-native member. **No FAANG.**
- These are downstream adopters signaling AI capability — not the companies *building* AI systems.

---

# 74.6% of overlap-panel companies changed their dominant posting archetype in two years

![w:900](assets/figures/T16/archetype_tvd_hist.png)

- **Archetype pivot rate: 74.6%** (73.2% at ≥5 labeled rows per period).
- Median total-variation distance between a company's 2024 and 2026 archetype distributions: **0.629**.
- Companies reconfigured *what kind* of SWE roles they post at astonishingly high rates.

<span class="cite">T16 Section 6 · 240-company panel</span>

---

# The LLM/GenAI archetype is 68% new entrants — a bifurcation with the existing-employer tool-stack adopter story

<span class="cite">T28 Section 5</span>

- 2024 LLM/GenAI cluster: **616 companies**. 2026: **1,174 companies**. Only **138** overlap.
- **68.2% of 2026 LLM/GenAI posting volume comes from new-in-2026 companies.**
- Top 2026 LLM/GenAI employers: Anthropic, Microsoft AI, Intel, Alignerr, Harvey, LinkedIn, Intuit, Cognizant.
- Within the LLM/GenAI archetype alone, the junior-senior `requirement_breadth` gap *widens* by 1.4 and `tech_count` by 2.0 — the only archetype where juniors and seniors *diverge* on breadth.

The AI market is bifurcated: existing-employer tool-stack adoption PLUS a new-entrant LLM/GenAI wave.

---

<!-- _class: section -->

# Part 6 — What does NOT survive

---

# The aggregate junior-narrowing story fails every robustness check we can apply

<span class="cite">T05 · T08 · T19 · V2</span>

- **Within-2024 SNR** on `seniority_final` entry share: **0.33** — far below the 2.0 threshold. It is not safe to pool 2024.
- **Arshkon-only baseline:** entry share FLIPS direction 7.72% → 6.70% (−1.0 pp).
- **Macro-robustness ratio:** **0.86×** — the cross-period effect is *literally smaller* than within-scraped-window drift.
- **Specification dependence:** `seniority_final` and YOE ≤ 2 proxy disagree in direction on within-company change, and have <10% row overlap — they measure different populations.

The aggregate junior-share rise is below the noise floor. The narrow surviving claim is within-archetype credential-stack convergence in 10/10 large archetypes (flip count 2-7, pattern-dependent).

---

# Seniority boundaries SHARPENED on three of four levels — the only blur is explained by directors being recast as tech leads

<span class="cite">T20 Section 2</span>

| Boundary | 2024 AUC → 2026 AUC |
|---|---|
| Entry ↔ mid-senior | 0.836 → **0.876** (+0.040) |
| Associate ↔ mid-senior | 0.691 → **0.791** (+0.100) |
| Entry ↔ associate | 0.626 → **0.719** (+0.093) |
| Mid-senior ↔ director | 0.677 → **0.616** (−0.061) ← the only blur |

The mid-senior/director blur is the `tech_count` sign flip from slide 24. The "seniority levels blurred" framing from earlier gates is contradicted.

---

# Length growth is mostly recruiter-LLM drafting style migration, not content expansion

![w:850](assets/figures/T29/fig1_score_distribution.png)

- **88.7%** of 2026 postings score above the 2024 median on the T29 authorship-style composite.
- Style-matched attenuation of length-growth effects: **23-62%** on content metrics.
- But the **AI rise** attenuates only **0-7%** — so the content story (AI, orchestration, tool-stack restructuring) is real, while the length story is mostly style migration.

**Length growth is NOT scope inflation.** It is recruiter tool adoption. (Gate 3 correction applied — we cite attenuation, not the feature-set-dependent sign flip.)

<span class="cite">T29 · Gate 3 narrowing 1</span>

---

<!-- _class: section -->

# What this work contributes

---

# The paper has two load-bearing empirical findings and one methodological contribution

**Empirical:**

1. **Posting language lags worker AI adoption by ~10× at the broad layer and ~13× at the hard-requirement layer.** Job postings are a lagging indicator of workplace AI adoption, not an anticipating one. This inverts the popular "employers demand impossible AI skills" framing.
2. **Senior roles specialized toward hands-on technical orchestration, not people management.** Directors became tech leads — the cleanest role redefinition signal in the corpus, and it is AI-linked.

**Methodological:**

3. **A four-test robustness framework** for longitudinal posting research: within-source calibration, specification dependence, macro-robustness ratios, and authorship-style matching. The aggregate junior-share null *under this framework* is itself a contribution — it tells us which past junior-narrowing findings will not replicate.

---

# The work means: we should stop reading job postings as a mirror of actual workplace practice

<span class="cite">Closing thought</span>

- Companies are using AI daily, and their postings trail by years.
- Job postings are a *template artifact* as much as a signal: 19% of companies rewrote their templates toward AI vocabulary, 75% pivoted their dominant archetype, and 89% of 2026 postings score above the 2024 recruiter-tool-drafting median.
- Labor-market research that treats postings as a direct measurement instrument will systematically miss the lead-lag structure.
- **Next:** a longer post-2026 window to measure the posting update lag as a quantifiable, multi-year phenomenon, and qualitative interviews at the 46 tool-stack adopter companies to test the client-signaling hypothesis.

Thank you.

---

<!-- _class: small -->

## Citations and reproducibility

- **Primary input:** `exploration/reports/SYNTHESIS.md`
- **Task reports:** `exploration/reports/T01.md` through `T29.md` (26 tasks)
- **Gate memos:** `exploration/memos/gate_0` through `gate_3_corrections`
- **Verifications:** `V1_verification.md` (Gate 2), `V2_verification.md` (Gate 3)
- **Validated pattern bundle:** `exploration/artifacts/shared/validated_mgmt_patterns.json` (100% precision, 100-row audit)
- **Figures used in this deck:** T08, T14, T16, T18, T20, T21, T22, T23, T29, interview artifacts.

Every claim on every slide has a direct citation to a task report. Every task report is in the Audit Trail tab of the accompanying mkdocs site.
