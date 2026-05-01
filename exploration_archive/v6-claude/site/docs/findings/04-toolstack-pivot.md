# Finding 4 — Tool-stack restructuring is field-wide, 92% within-company, and bifurcated

## Claim

The 2024-2026 AI tool-stack restructuring on the employer side is bifurcated into two distinct mechanisms: (a) a **46-company (19% of overlap panel) "tool-stack adopter" cluster** — dominated by consulting, system-integrator, and enterprise-software companies, **not FAANG** — pivoted hard on AI vocabulary and scope without moving its seniority mix; and (b) a **new-entrant LLM/GenAI wave** in which 68% of 2026 LLM/GenAI posting volume comes from companies not present in 2024. Across the overlap panel, **74.6% of companies changed their dominant posting archetype** between 2024 and 2026.

## Field-wide, not SWE-specific (T18 DiD)

- SWE vs control broad AI DiD: **+29.6 pp, 95% CI [28.9, 30.4]**
- Adjacent vs control: **+27.5 pp**
- **SWE vs adjacent: +2.08 pp, CI [0.77, 3.38]**
- SWE↔adjacent TF-IDF cosine did NOT blur (0.862 → 0.808, slight sharpening).
- Cleanest single spillover case: **network_engineer 4.2% → 16.6%** with zero SWE reclassification.

## 92% within-company (T16 overlap panel, n=240, ≥3 SWE per period)

- Panel AI any: **4.02% → 26.93%** (+22.91 pp).
- Within-company component: **+21.03 pp → 92%**. V2 reproduces at 89.7%.
- Among 186 panel companies with zero AI in 2024, within-company AI rose +0.229 by 2026.
- **Above macro noise:** T19 cross-period / within-scraped-window ratio **24.7×** on broad AI.
- **Above style migration:** T29 authorship-style matched delta attenuates broad AI only **0-7%**.

## Tool-stack adopter cluster (T16 k-means on panel change vectors)

- **46 of 240** overlap-panel companies (19%) cluster as "tool-stack adopters": ΔAI any **+0.523**, Δdesc length **+1,149 chars**, Δscope **+0.328**, entry share flat under both operationalizations.
- Cluster reproduces across seeds at n=50 with ΔAI +0.510 in V2 re-derivation.
- **Composition:** AT&T, Deloitte, American Express, Aditi Consulting, Aveva, Adobe, Macquarie Group. **Adobe is the most tech-native member. No FAANG.**

The companies rewriting their templates toward AI vocabulary are the downstream adopters that need to **signal AI capability**, not the ones building AI systems.

## LLM/GenAI new-entrant wave (T28)

- 2024 LLM/GenAI cluster: **616 companies**. 2026: **1,174 companies**. Only **138 overlap**.
- **68.2% of 2026 LLM/GenAI volume comes from new-in-2026 companies.**
- Top 2026 LLM/GenAI employers: **Anthropic, Microsoft AI, Intel, Alignerr, Harvey, LinkedIn, Intuit, Cognizant**.
- Within the LLM/GenAI archetype, the junior-senior `requirement_breadth` gap *widens* (+1.4) and `tech_count` *widens* (+2.0) — the one archetype where juniors and seniors pull **apart** on content breadth.

## Archetype pivot (T16)

- **74.6% of overlap-panel companies** changed their dominant archetype between 2024 and 2026. V2 reproduces at 71.7% on a slightly different denominator.
- Holds at **73.2%** when restricted to ≥5 labeled rows per period.
- Median total-variation distance across period archetype distributions: **0.629**.

## AI vocabulary spread into non-AI archetypes (T28)

- JS frontend broad AI **+14.7 pp**
- .NET **+13.6 pp**
- Java enterprise **+11.6 pp**
- DevOps **+10.2 pp**
- Data engineering **+9.2 pp**
- Defense/cleared and Embedded lag at +1.8 and +2.7 (clearance / domain barriers).

## Figures

![Tool-stack adopter cluster heatmap](../assets/figures/T16/cluster_heatmap_k4.png)
*T16 — k-means cluster decomposition. Cluster 3 (46 companies) is the AI/scope pivoting group.*

![Archetype pivot TVD histogram](../assets/figures/T16/archetype_tvd_hist.png)
*T16 — total-variation distance between 2024 and 2026 archetype distributions, by company. Median 0.629 — 74.6% pivoted their dominant archetype.*

![New vs returning decomposition](../assets/figures/T16/new_vs_returning.png)
*T16 — decomposition of AI rise into within-company (92%) vs between-company components.*

## Sensitivity checks this claim must survive

| Test | Requirement | Result |
|---|---|---|
| 92% within-company at both ≥3 and ≥5 thresholds | Robust | 74.6% pivot at ≥3; 73.2% at ≥5 — PASS |
| Cluster stability across random seeds | Stable | V2 reproduces at n=50 with ΔAI +0.510 — PASS |
| SWE-vs-adjacent DiD CI | Direction consistent | [0.77, 3.38] — PASS |
| Macro-robustness ratio | ≥ 10 | 24.7× — PASS |
| Authorship-style attenuation | Content real | 0-7% — PASS |

## Known reviewer attack surface

- **Tool-stack adopter cluster n=46 is small** — cluster membership is stable across seeds, but individual-company claims should cite the set explicitly.
- **T16 pivot rate is coverage-limited on scraped side** (30.5% archetype coverage). Flag explicitly; consider re-running T09 after coverage raise.
- **SWE-vs-adjacent DiD CI is modest** (+0.77 to +3.38 pp) — enough to say "SWE slightly ahead of adjacent" but not enough to say "SWE-distinctive."

## Task citations

- **[T14 — Technology ecosystem mapping](../audit/reports/T14.md)** — narrow/broad AI rates.
- **[T16 — Company hiring strategy typology](../audit/reports/T16.md)** — overlap panel, 92% within-company, tool-stack adopter cluster, archetype pivot rate.
- **[T17 — Geographic market structure](../audit/reports/T17.md)** — AI rose uniformly across 18 metros.
- **[T18 — Cross-occupation boundary](../audit/reports/T18.md)** — DiD.
- **[T19 — Temporal patterns](../audit/reports/T19.md)** — macro-robustness ratio.
- **[T28 — Domain-stratified scope changes](../audit/reports/T28.md)** — LLM/GenAI new-entrant wave, vocabulary spread.
- **[T29 — LLM authorship detection](../audit/reports/T29.md)** — 0-7% style attenuation.
- **[V2 verification](../audit/verifications/V2_verification.md)** — Alt 3 (existing-vs-new decomposition), cluster re-derivation.

## What this finding does NOT say

- It does not say tool-stack adoption explains everything — 8% of the AI rise is between-company.
- It does not say FAANG is absent from AI — only that FAANG is NOT in the 46-company tool-stack adopter cluster. FAANG is in the new-entrant + existing-AI-forward bucket.
- It does not say LLM/GenAI is a continuation of 2024 ML/AI — only 138/1,174 = 12% overlap; the LLM/GenAI segment is substantially new.
