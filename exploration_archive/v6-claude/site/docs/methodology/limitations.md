# Limitations and open questions

## Known confounders (ranked by severity, from SYNTHESIS Section 6)

| Confounder | Severity | Affects | Mitigation |
|---|---|---|---|
| **Length growth is mostly style migration** | **High** | Any raw length / density metric comparing 2024 to 2026 | T29 style matching, T13 section-anatomy decomposition; report attenuation |
| **Kaggle unformatted vs scraped markdown (instrument)** | High | Bullet density, em-dash, paragraph structure | Raw-text sensitivity (T29 halves authorship shift to +0.07); use raw for binary presence only |
| **Asaniczka zero native entry labels** | High | Any pooled-2024 entry metric using `seniority_native` | Arshkon-only baseline for native; pool only on `seniority_final` + YOE proxy |
| **Aggregator + entry-specialist intermediary contamination** | Moderate | Entry-share aggregate, company concentration, top-term frequency | `is_aggregator` exclusion sensitivity; add companion "entry-specialist intermediary" flag in preprocessing (action item) |
| **Company composition shift (new-entrant wave)** | Moderate | Between- vs within-company decomposition; LLM/GenAI archetype | Kitagawa decomposition on overlap panel (T16); explicit new-entrant bucket |
| **SWE-vs-field-wide (T18)** | Low | SWE-specificity framing | Cite T18 DiD; reframe paper from SWE to information-tech |
| **JOLTS macro cooling (−29% info-sector openings)** | Moderate | Any volume / share metric; entry share | T19 macro-robustness ratio; threshold ≥ 10 |
| **LLM budget coverage gap (scraped 30.7% labeled)** | Moderate | All text-based analyses on 2026 side | Cap at ~12,500 2026 rows; raise Stage 9 target going forward |
| **T09 archetype 30.5% scraped coverage** | Moderate | All within-archetype 2026-side claims | Flag explicitly; consider re-running T09 after coverage raise |
| **Stack Overflow self-selection** | Low (bounded) | RQ3 worker benchmark | Sensitivity 50-85% range; direction holds at floor |
| **Director cells thin** (99 / 112) | Moderate | Director-specific claims | Report CIs; use T21 + T20 convergent evidence |
| **T28 credential-stack pattern dependence** | Moderate | Entry-vs-mid-senior flip count | Cite convergence direction only (10/10); do not cite 7/10 flip |
| **Markdown-escape bug in scraped text** | Low (to fix) | Under-counts `c++`, `c#`, `.NET` in 2026 | Preprocessing fix pending |

## Open questions for the analysis phase

From SYNTHESIS Section 18:

1. **Is director-level recasting a real market shift or a template-rewriting artifact?** Thin cells (99/112) plus T29 mentor style-correlation make this non-trivial. Test: cluster-robust SEs on director-only cells; supplement with LinkedIn self-title analysis where available.
2. **Is the per-metro Austin JS-frontend collapse a single-city shock or a generalizable pattern?** Needs longer window.
3. **Is the LLM/GenAI new-entrant wave a sustained structural shift or a cyclical 2025-2026 startup surge?** Needs time-series extension.
4. **Does the hard-AI-requirement rate (6.0%) shift when the section classifier is audited specifically for AI-relevant rows?** Open method caveat from Gate 3.

## Preprocessing owner action items

From SYNTHESIS Section 17:

1. **Markdown-escape fix** — apply `re.sub(r"\\([+\-#.&_()\[\]\{\}!*])", r"\1", text)` in the cleaned-text pipeline. `c\+\+`, `c\#`, `\.net` silently dropped in 2026 scraped rows. Biases any "legacy language decline" finding.
2. **Entry-specialist intermediary flag** — add a companion to `is_aggregator` for SynergisticIT, WayUp, Jobs via Dice, Lensa, Emonics, Leidos, IBM. None are caught by `is_aggregator`; they drive ~15-20% of the 2026 entry pool.
3. **Stage 9 LLM text coverage raise** — 30.7% scraped-side labeled is the binding constraint on every text-based analysis.
4. **Stage 10 seniority coverage raise** — 53% of scraped SWE is `unknown`; fixes denominator drift.
5. **Archetype classifier on raw 2026 text** — 30.5% coverage is binding on within-archetype analysis; a lightweight raw-text classifier would broaden it.
6. **Track `seniority_final_source` distribution over time** — useful audit, will not survive a rule-version change silently.
7. **24 cross-company hash collisions** — canonicalizer follow-up for residual aggregator relabeling.

## What is NOT resolved in the exploration

The exploration was scoped to Wave 1-4 findings + sensitivity checks + synthesis, NOT to:

- Formal pre-registration of analysis-phase hypotheses (scheduled for Wave 5 / analysis phase; see T24 for 10 candidate hypotheses with formal-test specs).
- Qualitative interview execution (scheduled for Wave 6; see T25 for prepared stimuli).
- Final paper drafting (scheduled for Wave 7).
- Per-metro sub-study of Austin JS-frontend collapse, SF Bay LLM/GenAI concentration, Washington DC SCI-cleared cases.
