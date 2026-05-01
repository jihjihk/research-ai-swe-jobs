# Exploration Tracking Index

Last updated: 2026-04-17 (Gate 2 memo written; V1 verification dispatching)

**Gate 2 takeaways (see `exploration/memos/gate_2.md`):**
- Requirements section SHRANK −19% absolute chars (T13); responsibilities + benefits + role_summary + about_company expanded. Scope-inflation hypothesis INVERTED.
- Domain dominates NMI by ~7× (T09: domain=0.26, period=0.04, seniority=0.03). Period dominates embedding space by ~180× over seniority (T15).
- AI-mention signal SURVIVES every sensitivity: specialist + aggregator exclusion; SNR 24.7 broad / 35.4 strict; density SNR 4.99. Effect size ~34pp junior ~35pp senior — cross-seniority, not differential.
- Seniors changed MORE than juniors 2024→2026 in content (T12: cos 0.942 vs 0.953); seniors reallocated from requirements (−22%) to role_summary (+84%) and about_company (+96%).
- T11 refined strict-mgmt: `mentor|coach|hire|headcount|performance_review` after `manage` sub-pattern failed semantic precision (14% in SWE JDs). SNR 5.09, dominated by `mentor` (13.9%→20.8%).
- J2 junior share DIRECTION STILL BASELINE-DEPENDENT (arshkon-only down, pooled-2024 up) — Gate 1 finding confirmed.
- Emerging lead narrative: "Period-dominated, cross-seniority JD RE-WRITING toward AI vocabulary + section reallocation" — NOT restructuring of who gets hired.
- V1 MUST re-derive: (1) AI-mention semantic precision on 50-sample, (2) requirements-section shrink, (3) requirement_breadth component-length correlation, (4) NMI on full corpus, (5) within-LLM-frame J2 flip, (6) relabeling cosines.

**Gate 1 takeaways (see `exploration/memos/gate_1.md`):**
- Within-2024 SNR < 1 on every junior metric; junior-share direction depends on baseline choice.
- J3 YOE-based entry rise is 95% between-company compositional (not within-company scope change).
- AI-mention prevalence rose +33.3pp within-company on the 125-company arshkon∩scraped overlap panel — the only aggregate that survives composition, calibration, and concentration checks cleanly.
- 2026-03/04 is a JOLTS Info-sector hiring low (91K openings = 0.66× of 2023 avg).
- Primary seniority slices for Wave 2: **J2** (`seniority_final IN ('entry','associate')`) junior, **S1** (`seniority_final IN ('mid-senior','director')`) senior. Robustness: J3 (`yoe≤2`), S4 (`yoe≥5`).
- Senior-side findings MUST lead with arshkon-only baseline (asaniczka's missing native entry labels inflate pooled-2024 senior share).

## Wave-by-wave status

| Wave | Status | Agents | Dispatched | Completed |
|---|---|---|---|---|
| 1 — Data Foundation | complete | A, B, C, D | 2026-04-17 | 2026-04-17 |
| 1.5 — Shared Preprocessing | complete | Prep | 2026-04-17 | 2026-04-17 |
| 2 — Structural Discovery | complete | E, F, G, H, I | 2026-04-17 | 2026-04-17 |
| V1 — Gate 2 Verification | complete | V1 | 2026-04-17 | 2026-04-17 |
| 3 — Market Dynamics | complete | J, K, L, M, O | 2026-04-17 | 2026-04-18 |
| V2 — Gate 3 Verification | complete | V2 | 2026-04-17 | 2026-04-18 |
| 4 — Synthesis | complete | N | 2026-04-17 | 2026-04-17 |
| 5 — Presentation | pending | P | — | — |

## Task-level tracking

| Task | Agent | Wave | Status | Report | Key outputs |
|---|---|---|---|---|---|
| T01 Data profile & column coverage | A | 1 | complete | `exploration/reports/T01.md` | coverage heatmap |
| T02 Asaniczka associate as junior proxy | A | 1 | complete | `exploration/reports/T02.md` | verdict: NOT usable |
| T03 Seniority label audit | B | 1 | complete | `exploration/reports/T03.md` | kappa 0.40/0.49; defensible |
| T30 Seniority definition ablation panel | B | 1 | complete | `exploration/reports/T30.md` | `seniority_definition_panel.csv` |
| T04 SWE classification audit | B | 1 | complete | `exploration/reports/T04.md` | `is_swe` defensible |
| T05 Cross-dataset comparability | C | 1 | complete | `exploration/reports/T05.md` | tables/T05, figures/T05 |
| T06 Company concentration | C | 1 | complete | `exploration/reports/T06.md` | `entry_specialist_employers.csv` + prediction table |
| T07 Benchmarks & power analysis | D | 1 | complete | `exploration/reports/T07.md` | `feasibility_table.csv` (146 rows) |
| Shared preprocessing | Prep | 1.5 | complete | `exploration/artifacts/shared/README.md` | 9 artifacts, 127MB; AI-mention SNR 24.7 binary / 35.4 tool |
| T08 Distribution profiling | E | 2 | pending | — | — |
| T09 Archetype discovery | F | 2 | pending | — | `swe_archetype_labels.parquet` |
| T10 Title taxonomy evolution | G | 2 | complete | `exploration/reports/T10.md` | 15,021 truly-new titles; TF-IDF cosine 0.83-0.95; legacy-stack titles disappearing |
| T11 Requirements complexity | G | 2 | complete | `exploration/reports/T11.md` | `T11_posting_features.parquet`; requirement_breadth +34% (SNR 34.7); refined mgmt-strict SNR 5.09 |
| T13 Linguistic evolution | H | 2 | complete | `exploration/reports/T13.md` | section classifier; benefits +94% / resp +86% / requirements −8% pooled→scraped; requirements BELOW NOISE (SNR 0.21) |
| T12 Open-ended text evolution | H | 2 | complete | `exploration/reports/T12.md` | top emerging: rag/copilot/claude/ai-assisted/cursor; 60/100 terms "genuine" (in both full+section-filtered); relabeling diagnostic: period-effect dominant (entry26-entry24 cos 0.953 > entry26-midsr24 cos 0.938) |
| T14 Technology ecosystem | I | 2 | complete | `exploration/reports/T14.md` | tech co-occurrence graph; 7/20 top rising techs AI-era; 2026 LLM-vendor cluster (13 nodes); struct-vs-extract Spearman ρ=0.985 |
| T15 Semantic similarity | I | 2 | complete | `exploration/reports/T15.md` | period ~180× dominant over seniority; no robust convergence; 2026 postings more homogeneous internally |
| V1 Verification | V1 | 2.5 | complete | `exploration/reports/V1_verification.md` | 5 verified / 1 corrected (180×→1.2×); `agent_bare`+`mcp` precision dropped; breadth 71% content/29% length; LLM-frame J2 flip confirmed |
| T16 Company strategies | J | 3 | complete | `exploration/reports/T16.md` | 5 clusters (k=5, sil 0.188); AI-strict 102% within-co; breadth_resid +1.43 within-co; 76% cos broadened |
| T17 Geographic structure | J | 3 | complete | `exploration/reports/T17.md` | 26 well-powered metros; AI surge uniform (CV 0.29, all +); r(Δai,Δj2)=−0.11 null; top-3 AI: Tampa Bay/Atlanta/Charlotte |
| T18 Cross-occupation boundaries | K | 3 | complete | `exploration/reports/T18.md` | DiD SWE-specific for AI/tech/scope/breadth (99%,82%,95%,71%,72% of SWE-only); macro for desc_len (37%) and soft_skills (0%); req-section shrink SWE -10.7pp / adj -10.9pp / ctrl +0.9pp (NOT universal); ML-eng/DS overtake SWE on AI-mention |
| T19 Temporal patterns | K | 3 | complete | `exploration/reports/T19.md` | AI-strict acceleration 3.81×, AI-broad 2.75×; desc_len within-2024 sign-flips cross-period; within-scraped CV 0.04-0.14 (stable); `posting_age_days` degenerate; Sunday broad-AI 0.58 vs weekday 0.48-0.52 |
| T20 Seniority boundary clarity | L | 3 | complete | `exploration/reports/T20.md` | all boundaries sharpened (mid-sr/assoc +0.084, J3/S4 panel +0.14); ML/AI gained most clarity (+0.105) but cloud/frontend sharpest; scope+mgmt replaced tech as #2 discriminator at mid-sr line |
| T21 Senior role evolution | L | 3 | complete | `exploration/reports/T21.md` | mentor rose 1.73× at mid-sr vs 1.07× at entry (senior-specific, not corpus-wide); mgmt+orch+strat+AI sub-archetype 97% 2026; staff-title bundles mgmt+orch+strat+AI; shift ML/AI-concentrated for AI/orch, frontend for mgmt; systems is control outlier |
| T22 Ghost forensics | M | 3 | complete | `exploration/reports/T22.md` | `validated_mgmt_patterns.json`; AI matched-share delta +0.24 (2026), LLM ghost RR=0.98; paper NOT reframed to padding, add emerging-demand qualifier |
| T23 Employer-usage divergence | M | 3 | complete | `exploration/reports/T23.md` | RQ3 INVERTED: broad-AI req 46.8% < usage 75%; robust across 50–85% assumptions; employer UNDER-specifies |
| T28 Domain-stratified scope | O | 3 | complete | `exploration/reports/T28.md` | 22-archetype projection; AI-STRICT rise broad (18/20 >+5pp) but 2x at AI/ML; J2 within-archetype 84.3% of +3.32pp on full corpus; AI/ML 81% new entrants; systems_engineering = AI-STRICT zero control |
| T29 LLM-authorship detection | O | 3 | complete | `exploration/reports/T29.md` | score shift +1.14 SDs 2024→2026; low40-within-period subset retains **71-86%** of Wave 2 headlines (AI-strict 77%, mentor-senior 72%, breadth_resid 71%); PARTIAL support — mechanism present but not dominant |
| V2 Verification | V2 | 3.5 | complete | `exploration/reports/V2_verification.md` | 5 verified / 2 flagged / 1 corrected; T28 ≥+10pp is BROAD not STRICT; T29 retention method-sensitive; AI precision control 95% |
| T24 Hypothesis generation | N | 4 | complete | `exploration/reports/T24.md` | 10 new hypotheses ranked (H_A–H_J); H_A cross-occupation under-specification is priority 1 |
| T25 Interview artifacts | N | 4 | complete | `exploration/artifacts/T25_interview/README.md` | 4 ghost-JDs + 4 same-co pairs + 4 PNG charts (junior-trend, senior-archetype, employer-usage, AI-gradient) |
| T26 Synthesis | N | 4 | complete | `exploration/reports/SYNTHESIS.md` | lead sentence draft; RQ1a lead + RQ1b/c supporting + RQ1d NEW + RQ3 inverted; analysis-phase samples + method rules + sensitivity requirements |
| T27 Presentation package | P | 5 | pending | — | MARP + mkdocs site |

## Gate memos

- `exploration/memos/gate_0_pre_exploration.md` — pre-dispatch assessment, pre-committed ablation dimensions
- `exploration/memos/gate_1.md` — (pending)
- `exploration/memos/gate_2.md` — (pending)
- `exploration/memos/gate_3.md` — post-Wave-3 + V2 synthesis; RQ1a LEAD (SWE-specific AI rewriting); RQ3 INVERTED
- `exploration/memos/gate_4_handoff.md` — (pending)

## Shared artifacts (pending Wave 1 + Wave 1.5)

- `exploration/artifacts/shared/seniority_definition_panel.csv` — from T30 (junior J1–J6, senior S1–S5)
- `exploration/artifacts/shared/entry_specialist_employers.csv` — from T06
- `exploration/artifacts/shared/swe_cleaned_text.parquet` — from Agent Prep
- `exploration/artifacts/shared/swe_embeddings.npy` + `swe_embedding_index.parquet` — from Agent Prep
- `exploration/artifacts/shared/swe_tech_matrix.parquet` — from Agent Prep
- `exploration/artifacts/shared/company_stoplist.txt` — from Agent Prep
- `exploration/artifacts/shared/asaniczka_structured_skills.parquet` — from Agent Prep
- `exploration/artifacts/shared/calibration_table.csv` — from Agent Prep
- `exploration/artifacts/shared/tech_matrix_sanity.csv` — from Agent Prep
- `exploration/artifacts/shared/swe_archetype_labels.parquet` — from T09 (Wave 2)
- `exploration/artifacts/shared/validated_mgmt_patterns.json` — from T22 (Wave 3)
