# Exploration Report Index

Last updated: 2026-04-16

This index tracks task execution, report availability, key artifacts, and gate-level interpretation decisions. It should be updated after each wave and verification gate.

| Wave | Agent | Tasks | Status | Report(s) | Key artifacts | Gate notes |
|---|---|---:|---|---|---|---|
| 0 | Advisor | Gate 0 | Complete | [gate_0_pre_exploration.md](../memos/gate_0_pre_exploration.md) | Exploration directory scaffold | Pre-committed ablation discipline before Wave 1 |
| 1 | A | T01, T02 | Complete | [T01.md](T01.md), [T02.md](T02.md) | Coverage heatmap, coverage CSV | Cleaned-text coverage is the main text constraint; asaniczka native `associate` is not a junior proxy |
| 1 | B | T03, T30, T04 | Complete | [T03.md](T03.md), [T30.md](T30.md), [T04.md](T04.md) | `seniority_definition_panel.csv`, overlap heatmaps | `seniority_final` defensible but conservative; J1-J4 aggregate up; S1/S3/S4 down while S2 up |
| 1 | C | T05, T06 | Complete | [T05.md](T05.md), [T06.md](T06.md) | `entry_specialist_employers.csv` | Dataset instruments differ; company composition and entry-specialist employers are first-order confounds |
| 1 | D | T07 | Complete | [T07.md](T07.md) | `feasibility_summary.csv` | All-SWE/S1/S4 powered; J1 thin; J3/J4 powered low-YOE validators; geography benchmark strong |
| 1 | Advisor | Gate 1 | Complete | [gate_1.md](../memos/gate_1.md) | Wave 2 seniority and sensitivity guidance | Initial junior-decline narrative weakened; Wave 2 should investigate boundary/composition mechanisms |
| 1.5 | Prep | Shared preprocessing | Complete | [shared README](../artifacts/shared/README.md) | Shared text, embeddings, tech matrix, stoplist, structured skills, calibration table | Built sequentially with 4GB DuckDB cap and CPU embedding batches; embeddings complete for LLM-text rows |
| 1.5 | Advisor | Gate 1.5 | Complete | [gate_1_5.md](../memos/gate_1_5.md) | Mechanical artifact check | Shared artifacts pass; Wave 2 should be staged/memory-capped after OOM |
| 2 | E | T08 | Complete | [T08.md](T08.md) | Distribution/anomaly tables, seniority sensitivities, domain decomposition, four figures | Junior direction is source-sensitive; low-YOE and entry labels diverge; AI/tech/length changes are largest calibrated shifts |
| 2 | F | T09 | Complete | [T09.md](T09.md) | `swe_archetype_labels.parquet`, methods comparison, archetype tables, UMAP/PCA plots | NMF-8 chosen for shared labels; clusters align mainly with tech/domain, not seniority; labels cover LLM-cleaned rows only |
| 2 | G | T10, T11 | Complete | [T10.md](T10.md), [T11.md](T11.md) | Title taxonomy, requirement-complexity features, T30 panel effects, validation samples | AI/data hybrid titles and requirement breadth rise robustly; scope-density growth is stronger for low-YOE than J1 entry labels |
| 2 | H | T13, T12 | Complete | [T13.md](T13.md), [T12.md](T12.md) | T13 section anatomy/readability/tone tables; T12 log-odds term tables and sensitivities | Cleaned-text growth is not mostly boilerplate; 2026-heavy text shifts toward workflows, platforms, exposure credentials, systems scope, and AI tools |
| 2 | I | T14, T15 | Complete | [T14.md](T14.md), [T15.md](T15.md) | Tech ecosystem networks, structured-skill validation, semantic centroid/NN tables, UMAP/PCA maps | Tech expansion is broad and clustered around AI/platform/devops; generic semantic convergence is rejected after calibration |
| 2 | Advisor | Gate 2 | Complete | [gate_2.md](../memos/gate_2.md) | Wave 3 steering guidance | Lead story reframed to AI/platform/workflow skill-surface expansion and seniority-boundary ambiguity |
| 2V | V1 | Gate 2 verification | Complete | [V1_verification.md](V1_verification.md) | `exploration/tables/V1/`, `exploration/scripts/V1_verify_wave2.py` | Main Wave 2 story verified; cite broad vs narrow AI, LLM subset, row-share denominator, and ambiguous-token caveats carefully |
| 3 | J | T16, T17 | Complete | [T16.md](T16.md), [T17.md](T17.md) | Company strategy/decomposition tables; metro change tables and figures | Within returning companies, AI/tool growth is mostly within-firm; entry-label and low-YOE trends diverge; AI/tool rise is geographically broad |
| 3 | K | T18, T19 | Partial; T19 failed after retry | [T18.md](T18.md), T19 missing | Cross-occupation outputs; no temporal report | T18 completed; original K stalled before T19 and the T19-only retry produced no report, so temporal-rate evidence is a Gate 3 gap |
| 3 | L | T20, T21 | Failed after retry | Missing | Empty T20 directories only | Both original and T20-only retry stalled before scripts/reports; seniority-boundary modeling and senior-role deep dive remain Gate 3 gaps |
| 3 | M | T22, T23 | Failed after retry | Missing | Empty output directories only | Both original and narrowed retry stalled before producing scripts/reports; ghost/aspirational validity and RQ3 divergence remain major Gate 3 gaps |
| 3 | O | T28, T29 | Not started | Pending | Domain-stratified scope and authorship outputs | Deferred; T28 remains high-value targeted recovery, T29 lower priority |
| 3 | Advisor | Gate 3 | Complete | [gate_3.md](../memos/gate_3.md) | Wave 3 evidence synthesis and gap assessment | Strongest story is technical-work skill-surface expansion; T19/T20/T22/T23 gaps constrain claims |
| 3V | V2 | Gate 3 verification | Failed after retry | Missing | Empty V2 directory only | Both full and narrow verification attempts stalled before scripts/reports; T16/T17/T18 remain unverified beyond advisor review |
| 4 | N | T24, T25, T26 | Complete | [T24.md](T24.md), [T25.md](T25.md), [T26.md](T26.md), [SYNTHESIS.md](SYNTHESIS.md) | Hypotheses, interview artifacts, synthesis handoff | Strongest paper is AI-era skill-surface expansion across software-producing technical work; failed-task areas demoted to explicit gaps |
| 5 | P | T27 | Complete | [T27.md](T27.md) | `exploration/site/site/`, MARP HTML/PDF, hosted on port 8081 | Evidence package built; 8080 was occupied, so site is served at `http://100.127.245.121:8081/` |
