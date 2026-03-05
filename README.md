# The AI Restructuring of Junior SWE Labor

Research project studying how AI coding agents are restructuring junior software engineer roles, using job postings data.

## Project Structure

```
research/
├── README.md
├── setup.sh                       # One-command setup for new machines
├── alerts.conf                    # Alert channel configuration
│
├── scraper/                       # Scraping pipeline
│   ├── scrape_linkedin_swe.py     # Main scraper — LinkedIn + Indeed
│   ├── harmonize.py               # Unifies all data sources into one schema
│   ├── send_alert.py              # Alert dispatcher (email, Slack, Discord, ntfy, file)
│   └── run_daily.sh               # Cron wrapper (lock file, retries, log rotation)
│
├── notebooks/                     # Analysis
│   └── exploratory-analysis.ipynb # EDA notebook (Kaggle, Revelio, scraped data)
│
├── docs/                          # Research documents
│   ├── research-design-h1-h3.md   # Research questions & empirical strategy
│   ├── research-review.md         # Literature review
│   ├── validation-plan.md         # ML/stats validation approaches per RQ
│   ├── data-access-and-prompts.md # Data sources & LLM prompts
│   ├── session-summary.md         # Prior analysis session notes
│   └── sources.txt                # Reference sources
│
├── data/                          # (gitignored)
│   ├── unified.parquet            # Harmonized dataset (all sources, analysis-ready)
│   ├── scraped/                   # Daily scraper output
│   │   ├── YYYY-MM-DD_swe_jobs.csv
│   │   ├── YYYY-MM-DD_non_swe_jobs.csv
│   │   └── _seen_job_ids.json     # Dedup index
│   ├── kaggle-linkedin-jobs-2023-2024/
│   └── revelio/
│
└── logs/                          # Scraper logs (auto-rotated, 30 days)
```

## Quick Start

### On a new machine

```bash
git clone <repo-url> research
cd research
chmod +x setup.sh scraper/run_daily.sh
./setup.sh
```

This will:
1. Detect Python (>= 3.10 required)
2. Create a virtual environment (`.venv/`)
3. Install dependencies (`python-jobspy`, `pandas`, `pyarrow`)
4. Run a test scrape to verify everything works
5. Offer to install the daily cron job

### Manual setup (if you prefer)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install python-jobspy pandas pyarrow
mkdir -p data/scraped logs

# Test
python3 scraper/scrape_linkedin_swe.py --test

# Install cron (runs daily at 6 AM)
chmod +x scraper/run_daily.sh
(crontab -l 2>/dev/null; echo "0 6 * * * $(pwd)/scraper/run_daily.sh >> $(pwd)/logs/cron.log 2>&1") | crontab -
```

## Scraper Usage

### Running manually

```bash
# Test (1 query, 1 city, 5 results per site — ~1 minute)
python3 scraper/scrape_linkedin_swe.py --test

# Quick run (4 queries x 10 cities — ~30 min for both sites)
python3 scraper/scrape_linkedin_swe.py --quick

# Full run (12 queries x 20 cities — ~2-3 hours for both sites)
python3 scraper/scrape_linkedin_swe.py

# Single site only
python3 scraper/scrape_linkedin_swe.py --sites linkedin
python3 scraper/scrape_linkedin_swe.py --sites indeed

# More results per search
python3 scraper/scrape_linkedin_swe.py --results 50

# Catch up after a missed day
python3 scraper/scrape_linkedin_swe.py --hours-old 48

# Skip harmonization (just scrape)
python3 scraper/scrape_linkedin_swe.py --no-harmonize
```

### Via the cron wrapper

```bash
./scraper/run_daily.sh              # Default full run
./scraper/run_daily.sh --quick      # Quick mode (~15 min)
./scraper/run_daily.sh --catchup    # 48-hour lookback
```

The wrapper also accepts `--sites` and `--no-harmonize`:

```bash
./scraper/run_daily.sh --quick --sites indeed   # Quick, Indeed only
./scraper/run_daily.sh --sites linkedin         # Full, LinkedIn only
./scraper/run_daily.sh --catchup                # 48h lookback, both sites
```

The wrapper adds:
- **Lock file** — prevents overlapping runs; auto-clears stale locks (> 6 hours)
- **Retries** — up to 3 attempts with exponential backoff (5 min, 10 min, 20 min)
- **Log rotation** — deletes logs older than 30 days
- **Post-run summary** — logs row count and accumulated file count
- **Alerts** — sends success/failure/warning notifications via configured channels

### Cron job management

```bash
# View current cron jobs
crontab -l

# Change schedule (e.g., run at 3 AM)
./setup.sh --cron-only --cron-hour 3

# Remove cron job
crontab -l | grep -v "run_daily.sh" | crontab -
```

## How It Works

### Scraping strategy

1. Scrapes **LinkedIn** (public pages, no login) and **Indeed** (public API)
2. Runs multiple search queries (`software engineer`, `data engineer`, etc.) across US cities
3. Splitting by city bypasses per-search result caps
4. Sites are scraped sequentially with separate rate limits per site
5. Only fetches postings from the last 24 hours (configurable via `--hours-old`)
6. After scraping, automatically harmonizes all data (Kaggle + daily scrapes) into `data/unified.parquet`

### Anti-detection

| Measure | LinkedIn | Indeed |
|---------|----------|-------|
| Request delay | 8–20s random | 5–12s random |
| Between-query pause | 15–30s | 10–20s |
| Between-site pause | 30–60s | 30–60s |
| User agent rotation | 5 browser UAs | 5 browser UAs |
| Failure backoff | 60s × failure count | 45s × failure count |
| Circuit breaker | 5 consecutive failures | 5 consecutive failures |

### Deduplication

- Each job gets a hash (LinkedIn job ID, or MD5 of title+company+location)
- Hashes persist in `_seen_job_ids.json` across runs
- Cross-query dedup within a run (by `job_url`)
- Index capped at 500K entries to prevent unbounded growth

### Output schema

Each daily CSV contains these columns (from `python-jobspy`):

| Column | Description |
|--------|-------------|
| `site` | Source site (linkedin, indeed) |
| `id` | Job ID |
| `title` | Job title |
| `company` | Company name |
| `location` | Job location |
| `date_posted` | When the job was posted |
| `description` | Full job description (markdown) |
| `job_level` | Seniority level (entry level, mid-senior level, etc.) |
| `job_type` | Full-time, part-time, contract, etc. |
| `is_remote` | Remote flag |
| `min_amount` / `max_amount` | Salary range (when listed) |
| `job_url` | Direct link to the posting |
| `company_industry` | Industry classification |
| `company_num_employees` | Company size |
| `skills` | Listed skills |
| `scrape_date` | Date this scrape ran |

### SWE filtering

Titles are matched against this regex (same as the research notebook):

```
software engineer|software developer|swe|full-stack|front-end|back-end|
web developer|mobile developer|devops|platform engineer|data engineer|
ml engineer|machine learning engineer|site reliability
```

Non-SWE results are saved separately (`*_non_swe_jobs.csv`) for use as DiD control occupations.

## Monitoring

### Check if scraper is running

```bash
# Check for lock file
ls -la .scraper.lock 2>/dev/null && echo "Running (PID: $(cat .scraper.lock))" || echo "Not running"
```

### Check recent results

```bash
# Latest scrape stats
tail -20 logs/scrape_$(date +%Y-%m-%d).log

# Accumulated data summary
echo "Total daily files: $(ls data/scraped/*_swe_jobs.csv 2>/dev/null | wc -l)"
echo "Total SWE rows: $(cat data/scraped/*_swe_jobs.csv 2>/dev/null | wc -l)"
echo "Disk usage: $(du -sh data/scraped/)"
```

### Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| 0 results for all queries | LinkedIn rate limiting / IP blocked | Wait 24h, or use a proxy (`--proxy` support in jobspy) |
| Circuit breaker triggered | Too many consecutive failures | Check logs; increase delays in config |
| Stale lock file | Previous run crashed | Auto-clears after 6h, or `rm .scraper.lock` |
| Missing cron output | Cron environment issue | Check `logs/cron.log`; ensure absolute paths |

## Alerts

The scraper sends alerts on success, failure, or suspiciously low job counts. Configure channels in `alerts.conf`.

### Supported channels

| Channel | Setup | Best for |
|---------|-------|----------|
| **File** (default) | Enabled by default, writes to `data/scraper_status.json` | Local agents (e.g., openclaw) watching for changes |
| **Email** | Set SMTP credentials in `alerts.conf` | Gmail, Outlook, any SMTP server |
| **Slack** | Create a [webhook URL](https://api.slack.com/messaging/webhooks) | Team notifications |
| **Discord** | Channel settings > Integrations > Webhooks | Personal notifications |
| **ntfy.sh** | Just pick a topic name, install [ntfy app](https://ntfy.sh) | Free push notifications to phone, no signup |
| **macOS** | Enable in `alerts.conf` | Desktop notifications (only works in GUI session) |

### File alert for local agents

Enabled by default. Writes JSON to `data/scraper_status.json` with `current` status and a rolling `history` (last 30 entries). Your openclaw agent or any local process can poll this file:

```json
{
  "current": {
    "status": "success",
    "message": "Collected 486 SWE jobs, 667 total (mode=default)",
    "timestamp": "2026-03-05T12:26:17",
    "details": { "swe_count": 486, "total_count": 667, "attempt": 1 }
  },
  "history": [...]
}
```

### Alert triggers

| Condition | Alert type |
|-----------|-----------|
| Scrape completes normally | `success` |
| Full run collects < 10 SWE jobs | `warning` (suspiciously low) |
| Scraper fails after 3 retries | `failure` |

## Research Context

This scraper feeds data into a study of how AI coding agents are restructuring junior SWE roles. See:

- `docs/research-design-h1-h3.md` — Research questions and empirical strategy
- `docs/validation-plan.md` — ML approaches for each research question
- `docs/session-summary.md` — Prior analysis results

### Research questions

1. Are junior SWE roles disappearing or being redefined?
2. Which competencies migrated from senior to junior postings, and when?
3. Was there a structural break in late 2025?
4. Is this SWE-specific or a broader labor market trend?
5. What should replace the broken junior training pipeline?

