# Wave 4 Synthesis Handoff

## Scope And Evidence Boundary

This synthesis uses existing exploration reports and memos only. No new data analysis, preprocessing stages, parquet inspection, or LLM calls were run.

Gate 3 is the primary strategic context. T19, T20/T21, T22/T23, and V2 failed or are missing; their topics are treated as explicit gaps, not inferred evidence.

## Emerging Paper Narrative

Between 2024 and 2026, SWE and software-adjacent technical postings did not show a robust collapse of junior demand. The stronger evidence is AI-era skill-surface expansion: postings broadened around AI/tool/platform/workflow requirements across returning companies, metros, and adjacent technical occupations, while seniority labels and YOE floors diverged as measurement signals.

This should be framed as employer-side posting-content restructuring, not employment effects or a clean causal estimate of AI on jobs.

## Top Findings

| Rank | Finding | Evidence strength | Why it matters |
|---:|---|---|---|
| 1 | Within-company AI/tool expansion | Strong | T16: common companies moved broad AI 3.74% -> 23.23% and AI-tool 2.23% -> 19.80%, with 91-92% of AI/tool change decomposed as within-company. This is the lead mechanism result. |
| 2 | SWE-amplified, adjacent-parallel expansion | Strong | T18: SWE and adjacent technical roles move far more than controls. This reframes the paper away from SWE-only restructuring. |
| 3 | Geographic diffusion | Strong | T17: all 26 eligible metros increased broad-AI and AI-tool prevalence. The shift is not only Bay Area/Seattle/NYC. |
| 4 | Requirement and tech breadth expansion across seniority | Strong to moderate | T11/T14/V1: breadth rises across J1-J4 and S1/S4. Caveat: many text measures are LLM-cleaned subset claims. |
| 5 | Seniority label-vs-YOE divergence | Moderate, high conceptual value | T08/T16/T30/T15: explicit entry labels and low-YOE floors identify different rows and sometimes move in opposite directions. |
| 6 | Domain/technology dominates seniority in posting structure | Strong within LLM subset | T09/T15/V1: archetype/domain explains much more representation structure than seniority. |
| 7 | Senior roles broaden, but management-decline story is unsupported | Moderate | T11: senior breadth and mentorship/coordination rise; direct-report management remains rare. T21 missing prevents a lead senior-archetype claim. |
| 8 | Signal is not only boilerplate, but requirement force is unresolved | Moderate for not-only-boilerplate, missing for force | T13/V1/T16 reduce simple boilerplate/new-company explanations. T22 missing means screened versus aspirational requirements remain unknown. |

## Recommended RQ Evolution

**RQ1:** How much did AI/tool/platform and requirement-breadth language expand in SWE postings from 2024 to 2026, and does that expansion survive company, source, geography, and occupation controls?

**RQ2:** Is the expansion uniquely SWE-specific, or is it shared across software-adjacent technical occupations relative to controls?

**RQ3:** How do seniority labels, YOE floors, and requirement profiles diverge, and what does that imply for measuring junior and senior labor-market restructuring?

**RQ4:** Which parts of AI/tool/platform language are screened requirements, preferred skills, aspirational language, employer branding, or templates?

**RQ5:** Do employer-side AI/tool requirements outpace worker-side AI usage benchmarks? This should remain a hypothesis until T23-equivalent evidence exists.

## What Changed From The Original Framing

The original RQ1 emphasis on junior-share decline is weakened or contradicted. Aggregate and geography/cross-occupation panels often show junior definitions rising; common-company panels split J1/J2 labels downward and J3/J4 low-YOE upward. The stronger junior result is boundary ambiguity plus scope expansion.

The original RQ2 downward-migration frame is too narrow. Requirement breadth rises for junior and senior definitions, and domain/technology structure dominates seniority. The better framing is broad skill-surface expansion, with targeted migration as a secondary analysis.

The original senior-role redefinition claim needs demotion. Wave 2 supports senior complexity, mentorship, coordination, and technical breadth; it does not support a clean decline in management language. T21 failed, so senior archetype claims should be cautious.

The original employer-worker divergence RQ remains important but untested. Posting-side AI/tool evidence is strong, but T23 failed and requirement force is unknown.

## Method Recommendations

**What worked:**

- T30 seniority panel discipline: J1/J3 and S1/S4 distinctions prevented false junior/senior conclusions.
- Within-2024 calibration: essential for separating source noise from cross-period movement.
- Common-company decomposition: T16 turned AI/tool expansion into a credible mechanism result.
- Cross-occupation control design: T18 clarified that the paper is about technical-work expansion, not SWE uniqueness.
- LLM-cleaned text with raw sensitivity: T13/T12 avoided raw boilerplate traps.
- V1 verification: independent checks caught construct drift and citation risks.
- Company caps and aggregator exclusions: necessary for text, topic, and entry-share claims.

**What did not work or should be avoided:**

- Single junior-share estimates.
- Asaniczka native `associate` as a junior proxy.
- Current normalized-title S3 as a senior-title prevalence measure.
- Raw title/new-title counts as clean temporal evidence.
- Raw-description topic modeling without boilerplate controls.
- Generic junior-senior semantic convergence as a lead claim.
- Remote-work analysis until the all-zero 2026 remote field is audited.
- Broad multi-task agents after OOM/stalls; use smaller single-task recovery scripts.

## Claims Table

| Lead claim | Support | Caveats | Missing checks |
|---|---|---|---|
| Returning companies sharply expanded AI/tool posting language. | T16 common-company AI/tool decomposition; V1 verified Wave 2 AI prevalence. | Company canonicalization; broad versus narrow AI; binary mentions not force. | V2 verification; T22 requirement-force validation. |
| Expansion is SWE-amplified but adjacent-parallel. | T18 SWE/adjacent/control DiD. | Occupation classification and adjacent heterogeneity. | V2 verification; boundary-specific follow-up for data/ML/data engineering. |
| AI/tool diffusion is geographically broad. | T17 all 26 eligible metros positive. | Metro-known frame; source and domain composition; multi-location exclusions. | V2 verification; stricter metro threshold sensitivity in formal models. |
| Requirement and tech breadth rose across seniority. | T11/T14/V1; J1-J4 and S1/S4 all up. | LLM-cleaned subset for requirement breadth; length confounds; regex measures. | Expanded LLM coverage or careful subset language; T22 force validation. |
| Junior collapse is not supported as a lead fact. | T30, T08, T17, T18; T16 split by construct. | J1/J2 source-sensitive; J3/J4 are YOE constructs; unknown seniority large. | T19 temporal stability; T20 boundary modeling. |
| Seniority labels and YOE floors diverge. | T08 low-YOE mostly unknown/mid-senior; T16 J1 down/J3 up; T15 nearest-neighbor split. | Source and company composition; title code under-labeling. | T20 recovery; deeper unknown-pool profiling. |
| Senior roles broaden rather than shed management. | T11 senior breadth and mentorship/coordination findings. | Management categories are regex-derived; director S2 sparse. | T21 senior-role deep dive; manual validation. |
| Some posting language may be ghost/aspirational/template. | Plausible threat from V1/T11 caveats; not direct evidence. | T13/V1/T16 argue against all-boilerplate, but force is unknown. | T22 or interview evidence required. |
| Employer-worker divergence is plausible. | Posting-side numerator is strong. | No external usage comparison. | T23-equivalent benchmark analysis required. |

## Recommended Analysis-Phase Priorities

1. Estimate company fixed-effect models for broad AI, AI-tool, tech breadth, and requirement breadth.
2. Estimate cross-occupation DiD models for SWE, SWE-adjacent, and controls, with alternative adjacent group definitions.
3. Build a compact metro diffusion figure and model with metro fixed effects or domain-mix controls.
4. Formalize requirement-breadth models with length controls, section filtering, company caps, aggregator exclusion, and LLM-coverage disclosure.
5. Treat seniority as a measurement chapter: J1 versus J3/J4, S1 versus S4, unknown-seniority profiling, and label-vs-YOE divergence.
6. Recover a small T22-style requirement-force analysis before claiming hiring-bar changes.
7. Run a lightweight T19 stability check before making rate or flow claims about the 2026 scraped window.
8. Attempt T23 only after tool-specific posting measures and external usage benchmarks are cleanly aligned.
9. Expand or explicitly bracket scraped LLM-cleaned coverage before final semantic/archetype claims.

## Explicit Gaps From Failed Tasks

- **T19 missing:** no temporal-rate stability, scraped-window consistency, posting-age, backlog, or flow evidence. Avoid annualized rate claims.
- **T20 missing:** no seniority-boundary classifier, associate-distance model, or formal low-YOE unknown profiling.
- **T21 missing:** no senior-role deep dive. Do not lead with senior archetype redefinition beyond T11-supported complexity/mentorship evidence.
- **T22 missing:** no ghost/aspirational/template or screened-requirement force analysis.
- **T23 missing:** no employer-requirement / worker-usage divergence evidence.
- **T28/T29 not run:** no full domain-stratified decomposition and no LLM-authorship artifact test.
- **V2 missing:** Wave 3 T16/T17/T18 are synthesized by Gate 3 but lack independent verification comparable to V1.

## Suggested Figures And Tables

1. Within-company decomposition of broad AI, AI-tool, tech breadth, requirement breadth, J1, and J3.
2. Cross-occupation DiD slope plot or table for SWE, adjacent technical roles, and controls.
3. Metro heatmap or ranked slope plot showing all-metro AI/tool increases.
4. Requirement and technology breadth by T30 seniority definitions, with LLM coverage note.
5. J1/J2 versus J3/J4 seniority-boundary panel, including unknown-seniority composition.
6. Domain/archetype versus seniority structure figure from T09/T15.
7. AI/Python/platform technology network change from T14.
8. Claims-and-caveats table separating posting-language claims from hiring-bar claims.
9. Interview mechanism table mapping screened, preferred, aspirational, template, and adoption pathways.

## Honest Paper Positioning

The strongest paper right now is an empirical posting-content restructuring paper with a measurement contribution. It can credibly claim that AI/tool/platform/workflow language expanded across software-producing technical work and that this expansion appears within returning firms, across metros, and across SWE plus adjacent occupations.

It should not claim that AI eliminated junior SWE roles, that postings measure employment, that requirements are necessarily screened hiring bars, or that employer AI requirements outpace worker usage. Those stronger claims require the missing T19/T20/T22/T23-style evidence and interview validation.

The defensible contribution is narrower but still publishable: a transparent longitudinal posting dataset and a careful measurement framework showing how AI-era requirements diffused through technical job ads while seniority labels became unreliable anchors for interpreting junior and senior restructuring.

