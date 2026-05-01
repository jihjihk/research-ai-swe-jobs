# SWE Labor Market 2024-2026 — Findings Package

This is the findings package from the Wave 1-4 exploration of what happened to software-engineering job postings between 2024 and 2026. It is organized in three layers, one for each kind of reader.

**Start here (tabs at the top):**

- **Slides** — a 24-slide talk. The narrative in 10 minutes. Read the slide titles in sequence for a complete summary of what we found.
- **Findings** — one page per claim. Each states the claim, shows the evidence, cites the task reports, and summarizes the sensitivity checks a skeptic would ask about.
- **Methodology** — how the data was built (three sources, 10-stage pipeline, LLM prompts), the robustness framework, and the limitations.
- **Narrative** — what the paper can and cannot say, how the story changed across four gate reviews, and the full 500-line synthesis.
- **Audit trail** — every task report (T01-T29), every gate memo, both verification passes, the full task inventory. Use this when you want to check a number.

---

## The four core findings in one paragraph

Between 2024 and 2026, information-technology job postings underwent a structural AI-tool and framework explosion that was **field-wide**, **92% within-company**, and concentrated in consulting and system-integrator firms rather than FAANG. Senior roles specialized toward **hands-on technical orchestration** — mid-senior orchestration density rose 98%, director orchestration rose 156%, and the tech-lead sub-archetype more than doubled. But here is the surprise that inverts the original research question: **worker-side AI adoption (~81%) leads employer-side AI posting language (~29%) by roughly an order of magnitude.** Posting language is a *lagging* indicator of workplace AI adoption, not a leading one. Meanwhile three things framed as findings earlier in the exploration did not survive robustness checks: aggregate junior-share narrowing is smaller than within-scraped-window drift, seniority boundaries *sharpened* on three of four boundaries, and length growth is mostly recruiter-LLM drafting style migration.

---

## Embedded slide deck

<iframe
  src="presentation.html"
  width="100%"
  height="680"
  style="border: 1px solid #ccc; border-radius: 4px;"
  title="SWE labor market 2024-2026 — slides">
</iframe>

If the slides do not render, open them directly: [presentation.html](presentation.html).

---

## How to cite a finding

Every page in the Findings tab cites task reports by ID, e.g. `(T23, Section 2)`. Follow the audit-trail tab to read the full report for that task; follow the verification passes (V1 Gate 2, V2 Gate 3) to see exactly which numbers were re-derived and which corrections were applied.

## Timeline

- **Wave 1** — data foundation (T01-T07). Calibration, SNR thresholds, panel power.
- **Wave 2** — structural discovery (T08-T15). Archetypes, tech ecosystem, title evolution. V1 verification (Gate 2).
- **Wave 3** — market dynamics (T16-T23, T28-T29). DiD, geography, senior roles, divergence, LLM drafting. V2 verification (Gate 3).
- **Wave 4** — synthesis (T24-T26). Hypothesis generation, interview artifacts, SYNTHESIS.md.
- **Wave 5** — this package (T27).
