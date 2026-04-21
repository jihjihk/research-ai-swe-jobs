# SWE Labor Market Research — Project Instructions


## Purpose

Research project studying how AI coding agents are restructuring software-engineering roles, comparing historical LinkedIn postings from 2024 against daily-scraped postings from 2026. The emerging picture is that the same firms are rewriting the same roles toward AI-tooling and platform-infrastructure content, with seniority boundaries sharpening rather than blurring and the ladder narrowing at senior rather than entry levels; interviews extend the employer-side evidence with mechanism accounts of who rewrote the job descriptions and what changed about the work.

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

### 1. Preprocessing (`preprocessing/`, `data/`, `tests/`)

Pipeline that transforms raw job-posting data into analysis-ready datasets.

- **Guide:** [`docs/preprocessing-guide.md`](docs/preprocessing-guide.md) — architecture, operations, development, stage ownership rules
- **Schema:** [`docs/preprocessing-schema.md`](docs/preprocessing-schema.md) — column definitions and stage availability
- **Testing:** pytest suite in `tests/`
- **Outputs:** `data/unified.parquet`, `data/unified_observations.parquet`
- **LLM budget (REQUIRED):** Stages 9 and 10 require `--llm-budget N` (no default). The budget caps all new LLM calls across all data sources, split 40% SWE / 30% SWE-adjacent / 30% control by default. See the "Budget-Constrained LLM Processing" section in `docs/preprocessing-schema.md`.
- **Backup:** After a full pipeline run, back up outputs + LLM cache to S3: `python preprocessing/scripts/backup_to_s3.py` (or `--backup` flag on `run_pipeline.py`)
- Do not touch: `scraper/`, research writing files

### 2. Exploration & validation

Exploratory analysis on pipeline outputs to validate data quality and surface research insights.

- **Task reference:** [`docs/task-reference-exploration.md`](docs/task-reference-exploration.md) — shared preamble, task specs (T01–T38), agent dispatch blocks, V1/V2 verification specs.
- **Orchestrator:** [`docs/prompt-exploration-orchestrator.md`](docs/prompt-exploration-orchestrator.md) — dispatch, gate logic, wave guidance.
- **Outputs:** generated under `exploration/` for each run; stale outputs from prior runs are archived and not treated as canonical.
- **Current status (2026-04-20):** exploration phase complete through Wave 5 (T27). SYNTHESIS.md + 9 interview artifacts + 3-layer evidence site done.
  - **Paper's analytical backbone:** `exploration/reports/SYNTHESIS.md` (850 lines). Tier A + B + C + D rankings; 29-row hypothesis table; robustness appendix.
  - **Presentable artifact:** `exploration/site/` serves on tailnet at `http://100.127.245.121:8080` (27-slide MARP deck + mkdocs-material site + raw audit trail). Python HTTP server on port 8080 (background, PID tracked in `/tmp/site_server.log`).
  - **Hypotheses queued for analysis phase:** H_D (highest), H_O, H_P, H_Q; + 5 medium/low-priority from T24.

### 3. Analysis

Formal hypothesis testing and robustness checks for RQ1-RQ3. Formal analysis plan is pending.

**Navigation index — where existing work lives, where to look for new questions:**

- **Latest full exploration:** `exploration-archive/v9_final_opus_47/` — the 8-wave orchestrator run. Start with `reports/SYNTHESIS.md` (paper backbone), `reports/INDEX.md` (task catalog T01-T38), and `memos/gate_{0,1,2,3}.md` (narrative evolution). If a later version exists (v10, v11, …), prefer it.
- **Targeted follow-up analysis:** `eda/` — open-ended, hypothesis-driven notebooks that extend the exploration. Current entry point: `eda/reports/open_ended_v2.md` and `eda/notebooks/findings_consolidated_2026-04-21.ipynb`. See `eda/README.md` for structure.
- **Primary data for new queries:** `data/unified_core.parquet` — 193 MB, ~110k rows, 42 columns, covers the columns most analyses need. Start here. Switch to `data/unified.parquet` (7.4 GB) only when you need columns not in the core.
- **Data schema:** `docs/preprocessing-schema.md` — column definitions, stage availability, source-specific gaps, enum values. Read before writing any SQL.
- **Research design:** `docs/1-research-design.md` through `docs/6-methods-learning.md` — RQs, constructs, interview protocol, literature, publication targets, methods notes.
- **RAM:** 31 GB limit. Use DuckDB / pyarrow — never `pd.read_parquet` on the full `unified.parquet`.

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
| Kaggle arshkon | Historical snapshot | LinkedIn | Entry-level labels | Small SWE count |
| Kaggle asaniczka | Historical snapshot | LinkedIn | Large volume | No entry-level labels |
| Scraped | Growing current window | LinkedIn + Indeed | Fresh data, search metadata | Growing daily |

- Primary analysis platform: LinkedIn only. Indeed: sensitivity analyses only.
- Do not use: YC data, Apify data, old scraped format.
- Sync fresh data: `aws s3 sync s3://swe-labor-research/scraped/ data/scraped/`

See [`docs/preprocessing-guide.md`](docs/preprocessing-guide.md) for detailed source schemas and ingestion behavior.
