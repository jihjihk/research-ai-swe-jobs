# Finding 1 — The AI explosion is real, structural, and field-wide

## Claim

Between 2024 and 2026, information-technology job postings underwent a structural AI tool and framework explosion that restructured the employer-side technology co-occurrence graph, is 92% within-company on the 240-company overlap panel, is NOT SWE-specific (a DiD SWE-vs-adjacent difference is only +2.1 pp against +29.6 pp SWE-vs-control), and survives every sensitivity check we can apply.

## The numbers

**Narrow and broad cells cited separately** (V1 Gate 2 correction 1 — these are from different metrics and must not be combined):

| Metric | 2024 → 2026 | SNR | Citation |
|---|---|---|---|
| **AI narrow** (`LIKE '%ai%'` on text) | **2.81% → 18.78%** | **925** | T05 |
| **AI broad** (24-term union, regex) | **5.15% → 28.63%** | **13.3** | T14 |
| claude_tool | 0.01% → 3.37% | 326 | T14 |
| copilot | 0.06% → 3.77% | 44 | T14 |
| langchain | 0.11% → 3.10% | 36 | T14 |
| embedding | 0.15% → 2.82% | 123 | T14 |
| agents_framework | 0.61% → 12.70% | 140 | T14 (2024 FP ~10%, true delta ~+11.4 pp) |

**Network structure (T14, Louvain modularity on phi-weighted tech co-occurrence):**

- Tech network modularity **0.56 → 0.66**.
- 12 → 15 communities.
- Two new first-class AI communities: an **LLM/RAG applications cluster** (17 techs: langchain, langgraph, rag, openai_api, claude_api, agents_framework, mcp, fine_tuning, embedding, pytorch, scikit_learn, pandas, numpy, ml, nlp, llm, tensorflow) and an **AI-tools triad** (copilot, cursor_tool, claude_tool).

**Within-company (T16 overlap panel, n=240, ≥3 SWE per period):**

- Panel AI any: **4.02% → 26.93%** (+22.91 pp).
- **Within-company component: +21.03 pp → 92%.** V2 reproduces at 89.7% under a slightly different weighting scheme.
- Among 186 panel companies with zero AI in 2024, within-company AI rose +0.229 by 2026.

**Field-wide (T18 DiD, pooled-2024 vs 2026-scraped):**

- SWE vs control broad AI DiD: **+29.6 pp**, 95% CI [28.9, 30.4].
- Adjacent vs control: **+27.5 pp**.
- **SWE vs adjacent: +2.08 pp**, CI [0.77, 3.38] — modest and does not cross zero.
- Cleanest single spillover case: **network_engineer 4.2% → 16.6%** with zero SWE reclassification.

## Figures

![Technology co-occurrence network comparison](../assets/figures/T14/cooccurrence_network_2024_2026.png)
*T14 — 2024 vs 2026 tech co-occurrence graph with community coloring. Modularity rose 0.56 → 0.66.*

![DiD AI adoption gradient](../assets/figures/T18/ai_adoption_gradient.png)
*T18 — AI adoption gradient by occupation group. SWE and adjacent move together against control.*

## Sensitivity checks this claim must survive

| Test | Requirement | Result |
|---|---|---|
| Within-2024 SNR | ≥ 2 | 13.3 (broad), 925 (narrow) — PASS |
| Macro-robustness ratio | ≥ 10 | 24.7× on broad AI — PASS |
| Authorship-style attenuation | No sign flip | 0-7% on broad, 0% on narrow — PASS |
| SWE-vs-adjacent DiD CI | Direction consistent | +2.1 pp [0.77, 3.38] — PASS |
| Reproducibility | Within 1 pp | V1 Gate 2 verification — PASS |

## Known reviewer attack surface

- **2024 agents_framework baseline has ~10% false positives** (ambiguous pattern matching). The per-tool SNR 140 holds because the 2026 rise dwarfs the baseline; true delta is ~+11.4 pp rather than +12.1. (V1 correction 4.)
- **JOLTS macro cooling:** info-sector openings dropped 29% between windows. Addressed with the T19 macro-robustness ratio (24.7×).
- **Markdown escape bug** (`c\+\+`, `c\#`, `\.net`) under-counts legacy languages in 2026 scraped text. Preprocessing fix pending. Does not affect AI metrics.

## Task citations

- **[T05 — Cross-dataset comparability](../audit/reports/T05.md)** — narrow AI SNR 925, within-2024 calibration table.
- **[T14 — Technology ecosystem mapping](../audit/reports/T14.md)** — broad 24-term union, modularity, per-tool SNRs, community decomposition.
- **[T16 — Company hiring strategy typology](../audit/reports/T16.md)** — 92% within-company decomposition.
- **[T18 — Cross-occupation boundary](../audit/reports/T18.md)** — DiD numbers.
- **[T19 — Temporal patterns](../audit/reports/T19.md)** — macro-robustness ratio 24.7× on broad AI.
- **[T29 — LLM authorship detection](../audit/reports/T29.md)** — 0-7% style attenuation.
- **[V1 verification (Gate 2)](../audit/verifications/V1_verification.md)** — 5/5 headline numbers reproduced; narrow/broad split correction applied.
- **[V2 verification (Gate 3)](../audit/verifications/V2_verification.md)** — 92% within-company reproduced at 89.7%.

## What this finding does NOT say

- It does not say AI has "eaten SWE" — the SWE-vs-adjacent DiD is only +2.1 pp. The AI rise is a field-wide information-technology phenomenon.
- It does not mean every archetype rose — GPU/CUDA fell 4.7% → 3.4%. The correct phrasing is **"every large archetype rose"** or 9/10 large archetypes.
- It does not say the rise is driven by a few new AI-forward companies — the 92% within-company decomposition rules that out.
