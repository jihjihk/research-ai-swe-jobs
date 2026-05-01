# SWE Labor Market Research — Project Instructions

## Purpose

The SWE job market is changing quickly due to AI. Our goal is to quantify and describe how the role is changing. The pace of change and the lack of clarity can be harmful for both employers and job seekers. We suggest what new skills and requirements are emerging and what a better structure looks like for hiring.

## Repo Structure

```text
├── AGENTS.md              ← you are here
├── CLAUDE.md              ← Claude-specific addendum
├── README.md              ← public-facing project README
├── scraper/               ← daily scraping pipeline (EC2)
├── preprocessing/         ← pipeline code, intermediate artifacts, logs
├── data/                  ← final outputs (unified.parquet, unified_core.parquet, etc.)
├── tests/                 ← pytest suite for preprocessing
├── figures/               ← scripts/notebooks producing paper figures and tables
├── paper/                 ← LaTeX manuscript workspace
├── exploration_archive/   ← frozen archival material from earlier exploration phase
├── eda_archive/           ← frozen EDA notebooks from earlier exploration phase
├── keys/                  ← scraper SSH keys (gitignored)
├── pytest.ini             ← pytest configuration
└── requirements-test.txt  ← test dependencies
```

## Work Areas

### 1. Scraper (`scraper/`)

Daily scraping pipeline running on EC2.

- **Docs:** [`scraper/infrastructure-setup.md`](scraper/infrastructure-setup.md), [`scraper/ALERTING.md`](scraper/ALERTING.md)
- **Code:** `scraper/scrape_linkedin_swe.py`, `scraper/harmonize.py`, `scraper/run_daily.sh`
- Do not touch: `preprocessing/`, `figures/`, `paper/`

### 2. Preprocessing (`preprocessing/`, `data/`, `tests/`)

Pipeline that transforms raw job-posting data into analysis-ready datasets.

- **Guide:** [`preprocessing/preprocessing-guide.md`](preprocessing/preprocessing-guide.md) — architecture, operations, development, stage ownership rules
- **Schema:** [`preprocessing/preprocessing-schema.md`](preprocessing/preprocessing-schema.md) — column definitions and stage availability
- **Testing:** pytest suite in `tests/`
- **Outputs:** `data/unified.parquet`, `data/unified_observations.parquet`, `data/unified_core.parquet`, `data/unified_core_observations.parquet`. `data/unified_core.parquet` is the integration handoff to downstream analysis (figures, paper).
- **LLM budget (REQUIRED):** Stages 9 and 10 require `--llm-budget N` (no default). The budget caps all new LLM calls across all data sources, split 70% combined SWE (Stage 5 `is_swe OR is_swe_adjacent`) / 30% control by default. See the "Budget-Constrained LLM Processing" section in `preprocessing/preprocessing-schema.md`.
- **Backup:** After a full pipeline run, back up outputs + LLM cache to S3: `python preprocessing/scripts/backup_to_s3.py` (or `--backup` flag on `run_pipeline.py`)
- Do not touch: `scraper/`, `figures/`, `paper/`

### 3. Figures (`figures/`)

Scripts and notebooks that produce the figures and tables for the paper. Reads from `data/unified_core.parquet`. 

### 4. Paper (`paper/`)

LaTeX manuscript workspace for the publication itself.


### 5. Legacy Archives
`exploration_archive/` and `eda_archive/` are frozen archival material from the earlier exploration phase. Do not consult or modify unless explicitly instructed.


## Global Rules

- Use DuckDB for parquet/CSV inspection via the repo virtualenv (`./.venv/bin/python`). Avoid inline Python for data inspection.
- 31 GB RAM limit. Use pyarrow chunked I/O for pipeline code. See `preprocessing/preprocessing-guide.md` for memory patterns.
- Do not overwrite large data artifacts unless the task requires it.
- After completing work, update the relevant documentation if status, known issues, or priorities changed.

## Writing style

Prose for findings, stories, and methodology pages should read like *The Economist* — data-driven investigative journalism where numbers are carried by sentences rather than scanned off tables, caveats sit next to the claims they qualify, sentence length varies, and the writer commits where evidence is strong and hedges precisely where it is weak. Tonal anchors, to read before writing:

- Anthropic Economic Index, ["Learning curves"](https://www.anthropic.com/research/economic-index-march-2026-report) — defines a construct (automation vs. augmentation patterns in Claude usage), names the intuitive reading, and lands a counterintuitive fact about tenured users.
- Indeed Hiring Lab, Cory Stahle, ["How Employers Are Talking About AI in Job Postings"](https://www.hiringlab.org/2025/10/28/how-employers-are-talking-about-ai-in-job-postings/) — dense posting-share statistics carried by prose, with methodological caveats surfaced inline.
- Derek Thompson, ["The Evidence That AI Is Destroying Jobs For Young People Just Got Stronger"](https://www.derekthompson.org/p/the-evidence-that-ai-is-destroying) — steelmans the aggregate-null position, then names what it fails to explain; hard commit next to precise hedge.

## Data Sources

| Source | Period | Platform | Key strength | Key gap |
|---|---|---|---|---|
| Kaggle arshkon | Historical snapshot | LinkedIn | Entry-level labels | Small SWE count |
| Kaggle asaniczka | Historical snapshot | LinkedIn | Large volume | No entry-level labels |
| Scraped | Growing current window | LinkedIn | Fresh data, search metadata | Growing daily |

- Indeed and YC scraper code paths exist in `scraper/` as legacy infrastructure; their data is not used in current analysis.
- Sync fresh data: `aws s3 sync s3://swe-labor-research/scraped/ data/scraped/`

See [`preprocessing/preprocessing-guide.md`](preprocessing/preprocessing-guide.md) for detailed source schemas and ingestion behavior.
