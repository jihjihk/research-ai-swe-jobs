# Session Summary: AI-Driven Restructuring of Junior SWE Labor

## What We Started With

Three files comprising a research design for studying how AI coding agents are restructuring junior software engineer roles:

- **research-design-posting-restructuring.md** — Revised research questions, measurement strategy, and mixed-methods empirical design centered on posting-side restructuring
- **research-review.md** — Literature review covering the flipped pyramid, supervision paradox, strategic abstraction in "AI literacy," psychological framing effects, and the GenAI divide
- **sources.txt** — Eight annotated references (Brynjolfsson, Shen & Tamkin, Akanegbu, Chandar, Levanon et al., etc.)

## What We Produced

### 1. Validation Plan (`validation-plan.md`)

Mapped each RQ to concrete observables in job posting data, pairing simple baseline tests with more powerful ML approaches:

| RQ | Simple Test | ML Approach |
|----|------------|-------------|
| RQ1 (disappearing vs. redefined) | Volume counts by seniority | Fine-tuned seniority classifier (SetFit/BERT) to detect content-level redefinition |
| RQ2 (skill migration) | Keyword prevalence curves | BERTopic + skill embedding trajectories |
| RQ3 (structural break) | Bai-Perron on univariate series | Multivariate Bayesian changepoint detection (BOCPD) |
| RQ4 (SWE-specific) | Hand-picked DiD controls | Synthetic control method |
| RQ5 (training) | Qualitative synthesis | N/A — conditional on RQ1–4 |

### 2. Data Access Guide (`data-access-and-prompts.md`)

Since Lightcast is not accessible, we documented:

- **Public datasets** ranked by utility: Kaggle LinkedIn (posting text), Indeed Hiring Lab (aggregates), JOLTS/FRED (macro controls), JobHop (career trajectories), Stack Overflow & GitHub Archive (behavioral proxies)
- **Scraping strategy** using JobSpy (open source) for live data + Common Crawl for historical (2020–2023)
- **Four LLM prompts** ready to use: dataset discovery, scraping pipeline builder, posting classifier (extracts seniority, skills, ghost job flag), and task migration analysis

### 3. Exploratory Notebook (`exploratory-analysis.ipynb`)

Executed end-to-end across three datasets the user provided:

**Data sources ingested:**
- Kaggle LinkedIn postings (2023–2024) — full posting text with seniority, skills, descriptions
- LinkedIn scraped sample (March 2026) — 456 recent postings for temporal comparison
- Revelio Labs public data (2021–2026) — aggregate hiring, attrition, job openings, salaries, employment, layoffs by SOC code
- FRED JOLTS (configured but requires local execution — blocked in sandbox)

**Analyses performed (9 figures generated):**

1. **Seniority distribution shifts** — stacked area chart over time + cross-period comparison (2023–24 vs. 2026)
2. **Description complexity** — median word count by seniority over time + box plot comparison (scope inflation proxy)
3. **Skill prevalence: junior vs. senior** — 16 skills tracked; system design, CI/CD, mentorship, cloud show largest junior-senior gaps (migration candidates)
4. **Task migration map** — monthly skill prevalence in junior postings for 8 key skills
5. **Revelio 4-panel dashboard** — job openings, hiring rate, salary, employment for SOC 15 (Computer & Math) vs. controls (Architecture & Engineering, Healthcare, Management, Business & Financial). Computer & Math postings visibly declining steeper than controls.
6. **Indexed hiring rates** — SOC 15 vs. controls, indexed to first observation
7. **Layoff context** — WARN Act notifications over time
8. **Ghost job detection** — operationalized Akanegbu (2026)'s concept; entry-level SWE postings with 2+ senior signals flagged as ghost jobs
9. **AI skill emergence timeline** — AI/LLM tool mentions in all postings vs. SWE-only, plus breakdown by seniority

## Key Early Signals From the Data

- Revelio shows Computer & Math (SOC 15) job postings declining from ~3M to ~1.5M active postings between 2022 and 2025, steeper than control occupations
- Skill prevalence gaps between junior and senior SWE postings are substantial for system design, mentorship, Docker/K8s, and cross-functional collaboration — these are the migration candidates for RQ2
- AI/LLM tool mentions in SWE postings are still low in the 2023–24 Kaggle data (pre-agent era), establishing a useful baseline

## What's Still Needed

- **Pre-2023 posting text** at scale (Common Crawl parsing or Revelio academic access) for fuller historical coverage
- **More post-break months** to strengthen any breakpoint claims
- **Embedding-based seniority classifier** trained on the 2023-2024 benchmark dataset
- **BERTopic** for emergent skill discovery beyond our keyword dictionary
- **Hiring-side interview cohort** to validate JD authorship, screening, and ghost requirements
- **FRED JOLTS data** — download CSVs locally from fred.stlouisfed.org for the JOLTS charts
