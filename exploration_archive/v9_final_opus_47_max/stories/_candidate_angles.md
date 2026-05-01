---
title: "Candidate angles beyond the published collection"
date: 2026-04-21
status: reference
---

# Candidate angles — inventory of theories surfaced in the new-angle scans

This is the consolidated inventory of what the five new-angle scanning investigators surfaced across academic, consulting, tech-commentator, HR/recruiting and policy/government discourse. Each angle has a source, a quantitative claim, and a note on testability against the 68,000-row SWE-LinkedIn posting dataset. Angles marked "PICKED" were drafted into published pieces; others remain available for a second round.

Full investigator briefs with URLs are preserved in `exploration/stories/_logs/newangles_0N_*.md`.

---

## Picked (folded into pieces 01-10)

- **Entry-level collapse thesis** (Brynjolfsson, Amodei, SignalFire) — PIECE 01.
- **Copilot ubiquity thesis** (Nadella, Stack Overflow, DORA, McKinsey) — PIECE 02.
- **AI-hub concentration thesis** (Brookings, CBRE, PitchBook) — PIECE 03.
- **Regulated-industry lag thesis** (Deloitte, McKinsey, Gartner) — PIECE 04.
- **Pattern-validation absence in published literature** (Deming-Kahn, Acemoglu, Lightcast, Hansen as counter-example) — PIECE 05.
- **Five easy-explanation alternatives for the AI rise** (Fortune, Bersin, SHRM, HR Dive, Indeed Hiring Lab, Stanford DEL) — PIECE 06.
- **AI-democratises-engineering thesis** (Karpathy, Huang, Altman, Welsh) — PIECE 07.
- **Employer-worker AI perception gap** (McKinsey Superagency, BCG, WEF, MIT/Fortune) — PIECE 08.
- **Forward-Deployed Engineer surge** (Flex.ai, Indeed, bloomberry) — PIECE 09.
- **Mid-career squeeze / "seniority rollercoaster"** (Willison, Hosseini-Lichtinger) — PIECE 10 (rebutted with redirected rung).

---

## High-interest unpicked — publishable with one more data query

### A. "Platform engineering eats DevOps"
**Source:** Gartner 2025 Platform Engineering Report (55% of enterprise engineering orgs will have internal developer platforms by 2026, rising to 80%). Also Accenture Tech Vision 2025.
**Testable:** Yes. T09 archetype data shows cloud_devops +3.5 pp between 2024 and 2026, and T34 cluster 1 ("Senior Data-Platform / DevOps Engineer") is the second emergent senior archetype. A piece could compute the share of postings with "platform engineer" in the title against "devops engineer" / "SRE" across the period and test Gartner's prediction.
**Headline candidate:** "Platform engineering, but for whom?"
**Why it's worth running:** Gartner's 55%→80% is a prediction for employer behaviour; the posting data can report what actually happened in the first half of the window.

### B. "AI-first CEO memo cohort"
**Source:** Shopify (Lütke, April 2025), Amazon (Jassy, 2025), Klarna (40% workforce cut), Meta (Zuckerberg AI-first memo). Multiple CEOs have issued public "AI-first" memos.
**Testable:** Yes, but labour-intensive — requires manual labelling of which firms' CEOs have issued such memos, then comparing within-firm AI-rewriting (T31 pair-level drift) for memo-cohort vs non-memo firms.
**Headline candidate:** "Did the memo matter?"
**Why:** A clean quasi-natural-experiment — did firms whose CEOs committed publicly to AI-first development actually write more AI into their engineering postings?

### C. "The AI-builder vs AI-user geography split"
**Source:** Technical-commentator scan (angle #3). The FDE / Applied-AI "builder" roles may stay hub-locked while copilot-user roles diffuse.
**Testable:** Yes. Cross-tabulate the T09 ML/LLM archetype vs non-AI archetype by metro, within the already-computed T17 heatmap.
**Headline candidate:** "Two geographies, not one."
**Why:** Sharpens piece 03; would be an addendum rather than a new piece.

### D. "The Lightcast vs Korn Ferry AI-premium contradiction"
**Source:** Lightcast "Beyond the Buzz" (+28% AI-skill premium); Korn Ferry 2025 (AI premium only 5-15%).
**Testable:** NO without salary data. We have no compensation field. Skip — analysis-phase only.

### E. "Ghost-job epidemic"
**Source:** ResumeUp.AI / CNBC / Greenhouse (27.4% of US LinkedIn listings allegedly ghost; 40% of tech firms admit posting without intent to hire).
**Testable:** Partial — our T22 ghost-forensics work used LLM adjudication and kitchen-sink scoring. The 27% ghost rate is not directly reproducible in our frame, but we can compute posting-lifetime (`posting_age_days`) for the subset where coverage exists (0.9%).
**Headline candidate:** "The posting that nobody fills."
**Why:** Could extend piece 02 (postings-are-not-workflow theme).

### F. "Domain verticalization killing generic full-stack"
**Source:** Gartner 2026 Predictions.
**Testable:** Yes via T10 title taxonomy (full-stack title share 2024 vs 2026) and T28 cross-domain archetype share change.
**Headline candidate:** "The death of the full-stack generalist."
**Why:** Piece would test a widely-made prediction against posting-level evidence.

### G. "Title collapse of prompt engineer, rise of RAG engineer"
**Source:** Technical-commentator scan (angle #4). Prompt-engineer title said to have collapsed ~40% while RAG/LangChain/vector-DB tokens surge.
**Testable:** Yes with a simple regex.
**Headline candidate:** "The title that unstuck."
**Why:** Short, sharp; could be a piece 11 if a quick data query runs.

### H. "Frontend contracts faster than backend"
**Source:** Technical-commentator scan (angle #5).
**Testable:** Yes via T28 archetype rows (frontend_fullstack +2.7 pp per T10, but the piece would test whether specific frontend technologies decline vs backend).
**Headline candidate:** "Where AI hit the front-end first."

### I. "Federal Tech Force $130-200k AI fellows"
**Source:** OPM Tech Force (15 Dec 2025) — 1,000 federal AI fellows.
**Testable:** Partial — our corpus is LinkedIn-based; federal direct-hire postings may be under-represented.
**Headline candidate:** "The state catches up."
**Why:** Orthogonal new angle but thin on data.

### J. "H-1B wage-weighted lottery and SWE wages"
**Source:** USCIS H-1B rule effective 27 Feb 2026. Wage-weighted lottery replaces random selection.
**Testable:** Partial — requires salary data we don't have.

### K. "GAO cleared-workforce shortage"
**Source:** 70,000 cleared IT vacancies; 63,934 federal cyber staff. GAO report.
**Testable:** Partial — we have `has_clearance` flag (T28 clearance_defense archetype fell 12.6%→6.8%).
**Headline candidate:** "The clearance gap shrinks."

### L. "Babina et al.: AI firms flatten hierarchy"
**Source:** JFE 151 (2024) — AI firms hire 34% more junior-level workers; junior share rises in AI-heavy firms.
**Testable:** Yes via T16 AI-forward-scope-inflator cluster cross-tab with junior share.
**Why:** Directly supports piece 01. Could be folded in as corroborating mechanism rather than a standalone piece.

### M. "Humlum & Vestergaard Denmark study: 3% time saved, 8.4% new tasks"
**Source:** NBER 33777.
**Testable:** No (Denmark, not US; individual-worker-level, not posting-level).
**Background only.**

### N. "Task-concentration offset" (Hampole et al. NBER 33509)
**Source:** NBER 33509.
**Testable:** Theoretical — requires task-decomposition of postings.
**Analysis-phase only.**

### O. "Microsoft AI cluster: GitHub Copilot Workspace, Cursor, Claude Code as distinct sub-tools"
**Source:** T35 AI dev-tools sub-cluster finding.
**Testable:** Yes — already partially covered in pieces 02 and 07. Could expand to a separate methods-piece on how the tool-ecosystem has bifurcated (vendor-LLM for language, dev-tool-LLM for coding).

### P. "The Bachelor's-required rebound and the skills-based-hiring myth"
**Source:** Indeed January 2026 (Bachelor's requirement rebounded to 19.3%); Burning Glass Institute 2024 ("skills-based hiring affects <1 in 700 hires").
**Testable:** Yes via our `education_level` field in T11 posting features.
**Headline candidate:** "The skills-based hiring movement, quietly shelved."

### Q. "Returnship programs at IBM/LinkedIn/Goldman/JPM/Cisco for post-parental-leave SWE"
**Source:** SHRM, Techneeds.
**Testable:** Partial — we can compute age-cohort proxies from YOE but not from parental-leave signals.
**Too niche alone.**

### R. "FedFedsNote: NO evidence AI adoption reduces hiring"
**Source:** Fed FEDS Note (27 Mar 2026) — firm-level regression finds +0.082*** coefficient for AI adoption predicting hiring; the opposite direction of the Amodei-style story.
**Testable:** Yes via our T38 hiring-selectivity analysis, which found a similar result (volume-up firms write longer JDs).
**Why:** Already folded into piece 06; could be emphasized further.

### S. "$200k compensation cliff in AI hiring" (Korn Ferry 2025)
**Source:** Korn Ferry 2025 — AI-role postings above $200k take 114 days to fill vs 52 days for lower-paid roles.
**Testable:** No (no salary data).
**Analysis-phase only.**

### T. "St. Louis Fed: 0.57 correlation between unemployment change and AI adoption intensity by occupation"
**Source:** Fed St. Louis August 2025.
**Testable:** Yes, but at the occupation level, which extends beyond SWE.
**Piece candidate:** "What the Fed thinks we're missing about AI unemployment."

---

## Lower-priority or unactionable without more data

- T40: AI-skills wage premium 25%→56% jump (PwC) — needs salary.
- T41: McKinsey "juniors 7-10% slower with AI" — individual-productivity claim, not posting-level.
- T42: UK wage compression (Klein Teeselink SSRN 5516798) — UK-specific.
- T43: Mid-career returnship programs — niche.
- T44: Dice 17.7% AI premium — salary required.

---

## How to use this index

A second round of pieces could be commissioned quickly from angles A, B, C, G, F and L without needing new data beyond what our existing tables support. Angles D, S, J, and similar salary-dependent claims would require paired external compensation data (Levels.fyi, Dice, Korn Ferry).

For a seven-piece short-list of the next batch, the orchestrator's priority would be:

1. **Angle A** — "Platform engineering, but for whom?" — Gartner vs actual
2. **Angle C** — AI-builder vs AI-user geography split (addendum to piece 03)
3. **Angle F** — "The death of the full-stack generalist"
4. **Angle G** — "The title that unstuck" (prompt engineer → RAG engineer)
5. **Angle L** — Babina et al. hierarchical flattening (corroborates piece 01; could be folded in or separate)
6. **Angle P** — The skills-based-hiring myth (Bachelor's rebound)
7. **Angle B** — "Did the memo matter?" — only if manual CEO-memo labelling is tolerable

Angles K, R, and T could be expanded only if adjacent-occupation or federal-hiring lenses become priorities.
