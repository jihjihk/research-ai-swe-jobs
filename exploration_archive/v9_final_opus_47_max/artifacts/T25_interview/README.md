# T25 — Interview Elicitation Artifacts for RQ4

**Owner:** Agent N (Wave 4)
**Date:** 2026-04-20
**Purpose:** Data-grounded artifacts for RQ4 qualitative interviews (reflexive thematic analysis of mechanisms behind the 2024→2026 SWE posting restructuring).

This directory contains 9 artifacts spanning three categories:

1. **Content exemplars** (artifacts 1-4) — real posting text drawn from the data, to probe interviewees' lived experience of the content shifts the exploration documented.
2. **Visual / chart artifacts** (artifacts 5-7) — paper-lead candidate figures from T15/T20/T23/T32 that communicate the headline findings to interviewees.
3. **Inflated / ghost postings** (artifacts 8-9) — exemplars of posting-level ghost-likeness and entry-level scope inflation, for prompts on hiring-side authorship and mechanism.

---

## Artifacts and interview-question mapping

| # | Artifact | File | Source | Interview question (RQ4 mechanism probe) |
|---|---|---|---|---|
| 1 | Microsoft "Software Engineer II" pair-level rewrite | `01_microsoft_swe_ii_pair_rewrite.md` | T31 | "Did your role description change meaningfully between 2024 and 2026 at your firm? What drove the rewrite? (engineering priorities, recruiter LLM tools, compliance, hiring-team restructuring?)" |
| 2 | Wells Fargo & Capital One AI-drift exemplars | `02_wellsfargo_capitalone_drift.md` | T31 | "How did your team experience the transition from 'senior software engineer' to 'AI-enabled senior engineer' language in postings? Did the actual work shift, or just the description?" |
| 3 | Applied-AI / LLM Engineer archetype exemplars | `03_applied_ai_engineer_archetype.md` | T34 cluster 0 | "Is this a real job or a rebranding? If your employer posts this kind of role, what does the 'AI Engineer' day-to-day actually look like vs a 'Software Engineer'? Is seniority actually higher?" |
| 4 | Senior cluster transition exemplars (orch vs management) | `04_senior_cluster_transitions.md` | T21 + T34 | "The data shows senior SWE postings shifting from 'management language' toward 'orchestration language' (pipelines, workflows, system design, AI). Does this match your experience of senior IC work?" |
| 5 | Cross-occupation divergence figure | `05_cross_occupation_divergence.md` | T32 (figure reference) | "Why might employers list AI skills in only 10% of SWE postings when 80%+ of developers use AI daily? What's the gap between 'required skills' and 'tools engineers actually use'?" |
| 6 | Seniority-boundary sharpening figure | `06_seniority_boundary_sharpening.md` | T15 + T20 (figure reference) | "Between 2024 and 2026, entry and senior SWE roles became MORE distinct, not more blurred. Do you see this in hiring? Did entry-level hiring standards harden rather than soften?" |
| 7 | Junior-share and AI-timeline chart | `07_junior_share_ai_timeline.md` | T08 + T19 (figure reference) | "How did your hiring pipeline change as AI models matured between 2024 and 2026? Do you associate any specific model release (GPT-4o, Claude 3.5, Claude 4) with posting-language shifts?" |
| 8 | Ghost entry exemplars (senior scope in new-grad JDs) | `08_ghost_entry_scope_inflation.md` | T22 + T33 | "These are entry-level postings asking for LLM fine-tuning, GPT integration, architect-level system design. Do hiring managers write these with serious intent, or as wish-list 'nice-to-haves'? What fraction of applicants are expected to meet all of these?" |
| 9 | Hidden hiring-bar exemplars (requirements-contraction without bar-lowering) | `09_hiring_bar_contraction_exemplars.md` | T33 | "The requirements section shrank in many 2026 postings, but YOE asks, credential stack, and education asks did NOT drop. What changed about requirements-section writing? (Responsibilities migration? LLM-authored JD structure? Explicit de-emphasis of formal quals?)" |

---

## Usage notes for interview protocol

- **Show artifacts, not tables.** Interviewees respond best to real posting text (artifacts 1, 2, 3, 4, 8, 9) and to labeled charts (artifacts 5, 6, 7). Raw numerical tables bury the signal.
- **Ask about the MECHANISM, not the number.** RQ4 seeks "why" and "how" — the artifacts are prompts for causal stories, not arithmetic verification.
- **Stratify informants.** The artifacts are designed for three informant types:
  - Hiring managers / recruiters at returning firms (artifacts 1, 2, 9).
  - Senior IC engineers at AI-forward firms (artifacts 3, 4).
  - Junior / early-career engineers (artifacts 6, 7, 8).
  - Cross-occupation workers (e.g., data scientists, ML engineers at AI labs): artifact 5.
- **Ghost exemplars require care.** Artifact 8 shows real new-grad postings asking for LLM fine-tuning. Interviewees may recognize specific employers (Visa, DataVisor, PayPal); handle attributions sensitively per the interview protocol's anonymization rules.

---

## Artifact provenance

All artifacts derive from exploration outputs:
- Pair-level exemplars (artifacts 1, 2): `exploration/tables/T31/top20_ai.csv` + `data/unified.parquet` (description text joinable via uid).
- Archetype exemplars (artifact 3): `exploration/tables/T34/content_exemplars_cluster0.csv`.
- Senior cluster exemplars (artifact 4): `exploration/tables/T21/cluster_assignments.csv` + T34 content_exemplars.
- Cross-occupation divergence figure (artifact 5): `exploration/figures/T32_cross_occupation_divergence.png` (also SVG).
- Seniority-boundary figures (artifact 6): `exploration/figures/T20/auc_by_boundary.png`, `exploration/figures/T15/similarity_heatmap_tfidf.png`.
- Junior-share + AI timeline (artifact 7): `exploration/figures/T08/fig3_seniority_final.png`, `exploration/figures/T19/fig1_timeline.png`.
- Ghost entry (artifact 8): `exploration/tables/T22/top20_ghost_entry.csv`.
- Hiring-bar exemplars (artifact 9): `exploration/tables/T33/narrative_sample_50.csv`.

---

## Limitations

- The pair-level exemplars use representative 2026 postings selected for AI-mention density. Interviewees' direct access to the 2024 counterparts requires reading the T31 report's narrative (n=35 Microsoft "software engineer ii" 2026 postings cite Copilot / Generative AI / AI Systems; 2024 counterparts at n=6 do not).
- The artifacts are computed artifacts — interview protocol should additionally reserve unstructured prompts for informants to raise content shifts NOT surfaced by the quantitative exploration (e.g., informal AI use, non-posted hiring channels).
- Industry composition varies across artifacts; financial services and Software Development dominate the AI-rich examples (per T11 top-1% and T34 cluster 0). Interview sample should intentionally include under-represented sectors (healthcare, education, government) to surface mechanism heterogeneity.
