# SWE Labor Market Research — Project Instructions

Last updated: 2026-04-06 

## Purpose

Research project studying how AI coding agents are restructuring SWE roles. The core comparison is historical LinkedIn postings from 2024 versus daily-scraped postings from 2026.

### Research Questions

- RQ1: Employer-side restructuring — junior share/volume, scope inflation, senior role redefinition
- RQ2: Task and requirement migration — which requirements moved between seniority levels
- RQ3: Employer-requirement / worker-usage divergence — do posting AI requirements outpace actual usage
- RQ4: Mechanisms — interview-based qualitative (reflexive thematic analysis)

### Team

Two-person team:
- Partner owns scraper/infrastructure
- You own preprocessing, analysis, and related documentation in the analysis lane

## Repo Structure

```text
├── AGENTS.md              ← you are here
├── docs/                  ← research docs, plans, guides
├── preprocessing/         ← pipeline code, intermediate artifacts, logs
├── scraper/               ← daily scraping pipeline (EC2)
├── research/              ← academic writing workspace
├── data/                  ← final outputs (unified.parquet, etc.)
├── tests/                 ← pytest suite for preprocessing
└── notebooks/             ← exploration notebooks
```

## Work Areas

### 1. Preprocessing & analysis (`preprocessing/`, `data/`, `notebooks/`, `tests/`)

Pipeline that transforms raw job-posting data into analysis-ready datasets.

- **Guide:** [`docs/preprocessing-guide.md`](docs/preprocessing-guide.md) — architecture, operations, development, stage ownership rules
- **Schema:** [`docs/preprocessing-schema.md`](docs/preprocessing-schema.md) — column definitions and stage availability
- **Testing:** [`docs/testing/preprocessing-test-strategy.md`](docs/testing/preprocessing-test-strategy.md)
- **Outputs:** `data/unified.parquet`, `data/unified_observations.parquet`
- **LLM budget (REQUIRED):** Stages 9 and 10 require `--llm-budget N` (no default). The budget caps all new LLM calls across all data sources, split 40% SWE / 30% SWE-adjacent / 30% control by default. See the "Budget-Constrained LLM Processing" section in `docs/preprocessing-schema.md`.
- **Backup:** After a full pipeline run, back up outputs + LLM cache to S3: `python preprocessing/scripts/backup_to_s3.py` (or `--backup` flag on `run_pipeline.py`)
- Do not touch: `scraper/`, research writing files

### 2. Exploration & validation

Exploratory analysis on pipeline outputs to validate data quality and surface research insights.

- **Task reference:** [`docs/task-reference-exploration.md`](docs/task-reference-exploration.md) — shared preamble, 26 task specs + 2 verification agents
- **Orchestrator:** [`docs/prompt-exploration-orchestrator.md`](docs/prompt-exploration-orchestrator.md) — dispatch, gate logic, wave guidance
- **Synthesis:** `exploration/reports/SYNTHESIS.md` — consolidated findings from T01-T26 (THE handoff document for analysis phase)
- **Retrospectives:** `exploration/memos/wave*_retrospective.md` — lessons learned per wave; `exploration/memos/post_exploration_action_plan.md` — improvements for next re-run

### 3. Analysis

Formal hypothesis testing and robustness checks for RQ1-RQ3.

- **Plan:** [`docs/plan-analysis.md`](docs/plan-analysis.md) — Stages 15-16 spec

### 4. Scraper & infrastructure (`scraper/`)

Daily scraping pipeline running on EC2.

- **Docs:** [`docs/infrastructure-setup.md`](docs/infrastructure-setup.md), [`docs/data-sources-and-prompts.md`](docs/data-sources-and-prompts.md)
- **Code:** `scraper/scrape_linkedin_swe.py`, `scraper/harmonize.py`, `scraper/run_daily.sh`
- Do not touch: `preprocessing/`, `notebooks/`, research writing files

### 5. Research writing (`research/`, `docs/1-*.md` through `docs/6-*.md`)

Academic writing, research design, literature review, interview protocol, methods, paper drafts.

- **Instructions:** [`research/AMPLIFY.md`](research/AMPLIFY.md)
- **Canonical docs:** `docs/1-research-design.md` through `docs/6-methods-learning.md`
- Do not touch: `preprocessing/`, `scraper/`, `notebooks/`

## Global Rules

- Read [`docs/1-research-design.md`](docs/1-research-design.md) first for any research-related work. It defines RQ1-RQ4 and the empirical strategy.
- Use DuckDB for parquet/CSV inspection via the repo virtualenv (`./.venv/bin/python`). Avoid inline Python for data inspection.
- 31 GB RAM limit. Use pyarrow chunked I/O for pipeline code. See `docs/preprocessing-guide.md` for memory patterns.
- Do not overwrite large data artifacts unless the task requires it.
- After completing work, update the relevant documentation if status, known issues, or priorities changed.

## Data Sources

| Source | Period | Platform | Key strength | Key gap |
|---|---|---|---|---|
| Kaggle arshkon | Apr 2024 | LinkedIn | Entry-level labels | Small SWE count |
| Kaggle asaniczka | Jan 2024 | LinkedIn | Large volume | No entry-level labels |
| Scraped | Mar 2026+ | LinkedIn + Indeed | Fresh data, search metadata | Growing daily |

- Primary analysis platform: LinkedIn only. Indeed: sensitivity analyses only.
- Do not use: YC data, Apify data, old scraped format (Mar 5-18).
- Sync fresh data: `aws s3 sync s3://swe-labor-research/scraped/ data/scraped/`

See [`docs/preprocessing-guide.md`](docs/preprocessing-guide.md) for detailed source schemas and ingestion behavior.
