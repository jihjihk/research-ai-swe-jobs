# Interview elicitation artifacts

Produced by T25 (Agent N, Wave 4) · 2026-04-15
Script: `exploration/scripts/T25_interview_artifacts.py`

Five artifacts that support the RQ4 qualitative-phase data-prompted interviews. Each artifact is designed to be a discussion stimulus — a one-look image that encodes a non-obvious finding from the exploration that the interviewee can be asked to explain or react to.

## Artifacts

| File | What it shows | Source | Interview probe |
|---|---|---|---|
| `inverted_rq3_divergence.png` | Worker AI adoption (SO 2025) vs SWE posting AI rates, per-tool and aggregate, with worker-rate sensitivity band. | T23 + T22 + V2 re-derivation | "Your company uses AI tools daily but the job posting barely names them — why?" |
| `senior_orchestration_shift.png` | T21 people-mgmt, tech-orch, and strategic density by period × seniority. AI × senior interaction for 2026. | T21 validated patterns | "Did your director-level role actually become more hands-on technical, or is that just how you write the posting now?" |
| `archetype_pivot.png` | 74.6% archetype-pivot headline + illustrative paired pivots for tool-stack adopter companies. | T16 overlap panel | "Your company changed what kind of SWE you post for between 2024 and 2026 — what actually changed?" |
| `authorship_style_shift.png` | T29 LLM-authorship score distribution 2024 → 2026. Histogram + cumulative distribution. | T29 | "Are you using LLM drafting tools to write postings? When did you start?" |
| `paired_company_jds.csv` + `.md` | Raw JD pairs (2024 vs 2026) from 5 tool-stack adopter companies (Adobe, AT&T, Deloitte, American Express, Aditi Consulting). | `data/unified.parquet` / `description_core_llm` | "Between 2024 and 2026 your posting added AI language — what actually changed in the work?" |

## Per-artifact details

### 1. `inverted_rq3_divergence.png`

**Data:** Left panel — bar chart with worker any-use 80.8% (SO 2025), posting broad AI 28.6%, posting narrow AI 34.6%, posting AI-as-tool 6.9%, hard AI requirement 6.0% (AI × requirements section). Blue shaded sensitivity band 50-85% over the worker rate. Right panel — per-tool comparison for Copilot, Claude Code, ChatGPT with posting rate (2026) vs worker rate (SO 2025 extrapolated).

**Interpretation.** The paper's lead finding. RQ3 inverted: worker-side AI adoption is ~order of magnitude ahead of employer-side posting language. The hard-requirement rate (AI mentioned specifically in the requirements section) is 6.0% — employers describe AI as work context, not as hiring bars.

**Caveats applied.**
- Worker-rate sensitivity band (50-85%) shows direction holds across the full plausible range (per V2 re-verification).
- Per-tool worker rates are extrapolations from the SO 2025 "AI agent user" subset multiplied by 80.8% any-use — per Gate 3 narrowing 3, they are approximate ("~10-15× worker-to-posting ratio"). Do NOT present the per-tool panel as exact.

**Primary probe.** "Your company uses AI tools daily but the job posting barely mentions them — why?" Target: hiring managers at tool-stack adopter companies. Expected responses: template lag, tool-neutrality, "we expect developers to figure it out," legal/IP reasons.

### 2. `senior_orchestration_shift.png`

**Data:** Left panel — grouped bars (people-mgmt, tech-orchestration, strategic) × 4 columns (mid-senior 2024/2026, director 2024/2026). Annotated with +98% mid-senior orch and +156% director orch. Right panel — 2026 AI × senior interaction: no-AI vs AI-mentioning senior postings, showing the orchestration density nearly doubles (+76%) while people-management is identical.

**Interpretation.** Senior SWE roles specialized toward hands-on technical orchestration. Mid-senior tech-orch +98%, director tech-orch +156%, director people-management −21%. The AI × senior interaction is ENTIRELY in the orchestration profile — AI-mentioning senior postings are tech leads, not people managers.

**Caveats applied.**
- Director cell is small (99 / 112). Direction is robust (V2 precision audit 100% on a 100-row sample; holds at +120% excluding top-10 AI-heavy directors).
- Mentor sub-pattern (part of people-management) is style-correlated with T29 authorship score. The orchestration rise is NOT style-correlated.
- Validated patterns at `exploration/artifacts/shared/validated_mgmt_patterns.json`.

**Primary probe.** "Did your director-level role actually become more hands-on technical since 2024, or is that just how you write the posting now?" Target: director+ ICs and their recruiters.

### 3. `archetype_pivot.png`

**Data:** Left panel — headline bar chart: 179 of 240 overlap-panel companies pivoted dominant archetype (74.6%). Right panel — illustrative company pivots table for AT&T (Java enterprise → LLM/GenAI), Adobe, American Express, Deloitte, Aditi Consulting.

**Interpretation.** In two years, 74.6% of same-company panel entries changed their dominant T09 archetype. Median total-variation distance is 0.629 — half the companies redistribute over 63% of their archetype mass. Companies are reconfiguring what *kind* of SWE they hire for at high rates.

**Caveats applied.**
- Holds at 73.2% under ≥5-per-period threshold (V2 re-derived 71.7% with slightly different denominator).
- 30.5% scraped-side archetype label coverage — the 2026 side is a non-random subset of the 2026 postings. Pivot rate may overstate true reshuffling if Stage 9 favored certain titles.

**Primary probe.** "Your company changed what kind of SWE you post for between 2024 and 2026 — why?" Target: overlap-panel companies with high TVD (Deloitte, AT&T, Amex).

### 4. `authorship_style_shift.png`

**Data:** Left panel — overlapping histograms of T29 authorship score for 2024 and 2026, with median lines labeled. Right panel — cumulative distributions showing the translation upward without compression.

**Interpretation.** T29 authorship score shifted +0.33 std between periods. 88.7% of 2026 postings score above the 2024 median; only 3.9% fall below 2024 p25. The distribution TRANSLATED upward — variance did not decrease, indicating whole-corpus style migration rather than within-group compression onto a template. Candidate mechanism: recruiter-LLM drafting tool adoption.

**Caveats applied.**
- Bullet density is a major feature and is part instrument difference (Kaggle HTML-stripped vs scraped markdown). Raw-text sensitivity halves the shift to ~+0.07.
- Per Gate 3 narrowing 1, do NOT present the style-matched char_len "flip to −411." Present the attenuation story: length growth is mostly style migration (23-62% attenuation on tech_count, scope_density, requirement_breadth under ANY matching spec).
- AI explosion survives style matching: 0-7% attenuation. The AI finding is NOT an authorship artifact.

**Primary probe.** "Are you using LLM drafting tools (ChatGPT, Claude, a recruiter tool) to write postings? When did you start?" Target: T29 top-LLM-score 2026 companies (Alignerr, Intuit, LinkedIn, Intel, Harvey, Microsoft AI, Walmart).

### 5. `paired_company_jds.csv` + `paired_company_jds.md`

**Data:** Five same-company paired job descriptions (2024 arshkon/asaniczka + 2026 scraped) from tool-stack adopter cluster companies. For each company, the longest available `description_core_llm` in each period. CSV for programmatic use; Markdown for human reading in interview sessions.

Companies included:
- AT&T
- Adobe
- American Express
- Aditi Consulting
- Deloitte

**Interpretation.** Same-company, same-title-family paired JDs show the tool-stack adopter pattern at the text level. The reader can see exactly what 2024 asked for vs what 2026 asks for — typically with AI vocabulary added, longer descriptions, and often different titles.

**Caveats applied.**
- 2024 source is kaggle_arshkon or kaggle_asaniczka (HTML-stripped text); 2026 source is scraped (preserves markdown). The length difference is partly instrument.
- 2024 and 2026 postings in the same company may not be for the same specific role — companies have multiple SWE postings, and we selected the longest per period. This is interview stimulus material, not a formal content-similarity analysis.
- Adobe has the most comparable pair (same pipeline, actively hiring in both periods).

**Primary probe.** "Between 2024 and 2026 your posting added AI language — what actually changed in the work?" Target: hiring managers who recognize the company's posting templates.

## Usage notes

- All PNGs are 150 dpi, figure-size optimized for presentation (13×6 or 13×5.5).
- All artifacts reflect Gate 3 narrowings: broad-AI ratio as lead, per-tool softened, convergence direction over flip count, attenuation over sign-flip.
- Source tables are at `exploration/tables/T{16,21,22,23,29}/`.
- For the formal analysis phase, re-derive the numbers from source tables or shared artifacts rather than from these figures.

## Related outputs

- `exploration/reports/T25.md` — meta-report documenting artifact provenance and interview-use context.
- `exploration/reports/T24.md` — new hypotheses for pre-registration, including the interview-probe targets.
- `exploration/reports/SYNTHESIS.md` — consolidated exploration handoff.

End of README.
