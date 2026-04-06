# The AI Restructuring of the SWE Seniority Ladder

Research project studying how AI coding agents are restructuring software engineer roles across the entire seniority ladder. Uses a historical LinkedIn benchmark dataset (Kaggle / Hugging Face, 2023–2024) plus daily LinkedIn / Indeed / YC scraping in 2026+.

## Project Structure

```
research/
├── README.md
├── setup.sh                       # One-command setup for new machines
├── .env                           # Environment config (S3_BUCKET, SNS_TOPIC_ARN) — not in git
│
├── scraper/                       # Scraping pipeline
│   ├── scrape_linkedin_swe.py     # LinkedIn + Indeed scraper (python-jobspy)
│   ├── scrape_yc.py               # Y Combinator / Work at a Startup scraper
│   ├── harmonize.py               # Builds canonical postings + daily observations parquet files
│   ├── send_alert.py              # Daily summary + SNS alerting
│   └── run_daily.sh               # Cron wrapper (lock file, retries, log rotation)
│   └── queue_full_detached.sh     # Durable fallback wrapper for today's full rerun
│
├── notebooks/                     # Analysis
│   └── exploratory-analysis.ipynb # EDA notebook (Kaggle, Revelio, scraped data)
│
├── docs/                          # Research documents
│   ├── 1-research-design.md       # Canonical research design
│   ├── 2-interview-design-mechanisms.md # Canonical interview / mixed-methods design
│   ├── 3-literature-review.md     # Literature review
│   ├── 4-literature-sources.md    # Reference list
│   ├── 6-methods-learning.md      # Methodology learning guide from prior notes
│   ├── data-sources-and-prompts.md # Public data options & prompts
│   ├── infrastructure-setup.md    # EC2 & S3 infrastructure docs
│   ├── 5-publication-targets-2026-2027.md # Venue strategy and deadlines
│   └── archive/                   # Historical notes and superseded drafts
│
├── data/                          # (gitignored)
│   ├── unified.parquet            # Canonical postings table (global dedupe)
│   ├── unified_observations.parquet # Daily panel (one row per posting per scrape_date)
│   ├── scraped/                   # Daily scraper output
│   │   ├── YYYY-MM-DD_swe_jobs.csv       # LinkedIn/Indeed SWE-matched jobs
│   │   ├── YYYY-MM-DD_non_swe_jobs.csv   # LinkedIn/Indeed non-SWE jobs
│   │   ├── YYYY-MM-DD_yc_jobs.csv        # YC jobs (all roles)
│   │   ├── YYYY-MM-DD_manifest.json      # LinkedIn/Indeed run params
│   │   ├── YYYY-MM-DD_yc_manifest.json   # YC run params
│   │   ├── _seen_job_ids.json            # LinkedIn/Indeed dedup index
│   │   ├── _seen_yc_ids.json             # YC dedup index
│   │   └── _yc_scraper_state.json        # YC scan position tracker
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
3. Install dependencies (`python-jobspy`, `pandas`, `pyarrow`, `requests`, `beautifulsoup4`, `httpx`)
4. Run a test scrape to verify everything works
5. Offer to install the daily cron job

### Manual setup (if you prefer)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install python-jobspy pandas pyarrow requests beautifulsoup4 httpx
mkdir -p data/scraped logs

# Test
python3 scraper/scrape_linkedin_swe.py --test
python3 scraper/scrape_yc.py --test

# Install cron (runs daily at 6 AM)
chmod +x scraper/run_daily.sh
(crontab -l 2>/dev/null; echo "0 6 * * * $(pwd)/scraper/run_daily.sh >> $(pwd)/logs/cron.log 2>&1") | crontab -
```

## Data Sources

| Source | Scraper | Output | Scope |
|--------|---------|--------|-------|
| **LinkedIn** | `scrape_linkedin_swe.py` | `*_swe_jobs.csv`, `*_non_swe_jobs.csv` | SWE + adjacent + control roles across 26 US metro areas |
| **Indeed** | `scrape_linkedin_swe.py` | (same files) | Same queries / metros, primarily ablation + robustness |
| **YC (Work at a Startup)** | `scrape_yc.py` | `*_yc_jobs.csv` | All roles at YC-funded startups |
| **Kaggle** | (static download) | `kaggle-linkedin-jobs-2023-2024/` | Historical LinkedIn data (2023–2024) |

## Scraper Usage

### LinkedIn + Indeed

```bash
# Test (1 query, 1 city, 5 results per site — ~1 minute)
python3 scraper/scrape_linkedin_swe.py --test

# Quick run (4 queries x top 10 metros)
python3 scraper/scrape_linkedin_swe.py --quick

# Full run (28 queries x 26 metros, LinkedIn + Indeed in parallel)
python3 scraper/scrape_linkedin_swe.py

# Single site only
python3 scraper/scrape_linkedin_swe.py --sites linkedin
python3 scraper/scrape_linkedin_swe.py --sites indeed

# Run specific tiers only
python3 scraper/scrape_linkedin_swe.py --tiers swe            # SWE only
python3 scraper/scrape_linkedin_swe.py --tiers swe adjacent   # Skip control

# More / fewer results per query-metro pair
python3 scraper/scrape_linkedin_swe.py --results 50
python3 scraper/scrape_linkedin_swe.py --results 100

# Catch up after a missed day
python3 scraper/scrape_linkedin_swe.py --hours-old 48

# Skip harmonization (just scrape)
python3 scraper/scrape_linkedin_swe.py --no-harmonize

# Force sequential mode (debug / validation only)
python3 scraper/scrape_linkedin_swe.py --sequential

# Tighten resource guardrails on smaller machines
python3 scraper/scrape_linkedin_swe.py --request-timeout-sec 180 --memory-soft-limit-mb 8192 --memory-hard-limit-mb 10240
```

### YC (Work at a Startup)

```bash
# Test (scan ~70 job IDs — ~3 minutes)
python3 scraper/scrape_yc.py --test

# Full run (scan ~700+ IDs — ~30 minutes)
python3 scraper/scrape_yc.py

# Skip harmonization
python3 scraper/scrape_yc.py --no-harmonize
```

The YC scraper works by scanning sequential job IDs on workatastartup.com. Each active job page contains structured JSON-LD data (title, company, salary, location, description, YC batch). It tracks the highest scanned ID across runs so it only scans new ranges.

### Via the cron wrapper

```bash
./scraper/run_daily.sh              # Full run (LinkedIn + Indeed + YC)
./scraper/run_daily.sh --quick      # Quick mode (YC runs in test mode)
./scraper/run_daily.sh --catchup    # 48-hour lookback
./scraper/run_daily.sh --sites linkedin   # LinkedIn only (YC still runs)
```

The wrapper runs LinkedIn/Indeed first, then YC as a separate step. If YC fails, it doesn't affect the main scrape.

The wrapper adds:
- **Lock file** — prevents overlapping runs; auto-clears stale locks (> 6 hours)
- **Retries** — up to 3 attempts with exponential backoff (5 min, 10 min, 20 min)
- **Log rotation** — deletes logs older than 30 days
- **Post-run summary** — logs row counts plus both parquet outputs
- **S3 sync** — uploads daily CSVs, `unified.parquet`, `unified_observations.parquet`, and status JSON to S3 if `S3_BUCKET` is configured
- **Alerts** — sends daily summary email via AWS SNS, including repeated rate-limit warnings from the manifest

Resource guardrails:
- per-request timeout in the scraper (`--request-timeout-sec`, default `180`)
- RSS soft limit in the scraper (`--memory-soft-limit-mb`, default `50%` of RAM) triggers flush + GC
- RSS hard limit in the scraper (`--memory-hard-limit-mb`, default `65%` of RAM) aborts before the kernel OOM killer does
- wrapper-level wall-clock timeout via `SCRAPER_MAX_RUNTIME_SECONDS` (default `21600`, i.e. 6 hours)

### Durable reruns / detached execution

If you are running a one-off full rerun from an interactive shell and want it to survive the session:

```bash
# Queue a durable follow-up around an already-running foreground scrape
bash scraper/queue_full_detached.sh --wait-pid <SCRAPER_PID>

# Or just queue a detached full rerun immediately
bash scraper/queue_full_detached.sh
```

Behavior:
- if today's foreground scrape finishes successfully, the queued job regenerates both parquet outputs
- if the foreground scrape dies before producing today's manifest, the queued job starts a fresh detached `run_daily.sh --full`

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

### Query tiers (LinkedIn + Indeed)

Queries are organized into priority tiers. Work is shuffled across metros, but the tiering remains important for auditability and targeted reruns.

| Tier | Queries | Purpose |
|------|---------|---------|
| **swe** | `software engineer`, `full stack engineer`, `frontend engineer`, `backend engineer`, `devops engineer`, `data engineer`, `machine learning engineer`, `AI engineer`, `mobile developer`, `founding engineer` | Primary research focus |
| **adjacent** | `data scientist`, `data analyst`, `product manager`, `UX designer`, `QA engineer`, `security engineer`, `solutions engineer`, `technical program manager` | AI-exposed comparison group |
| **control** | `civil engineer`, `mechanical engineer`, `electrical engineer`, `chemical engineer`, `registered nurse`, `accountant`, `financial analyst`, `marketing manager`, `human resources`, `sales representative` | Non-AI-exposed group for DiD |

### SWE classification

Titles from LinkedIn/Indeed are matched against a regex to split into SWE vs non-SWE files:

```
software engineer|software developer|swe|full-stack|front-end|back-end|
web developer|mobile developer|devops|platform engineer|data engineer|
ml engineer|machine learning engineer|site reliability|ai engineer|
ai/ml engineer|llm engineer|agent engineer|applied ai engineer|
prompt engineer|infrastructure engineer|founding engineer|
member of technical staff|product engineer
```

The regex is broader than the search queries — it catches specific titles (e.g., "LLM engineer", "agent engineer") that appear in results from broader queries (e.g., "AI engineer").

Non-SWE results are saved separately (`*_non_swe_jobs.csv`) for use as DiD control occupations.

YC jobs are **not** split — all roles are saved in one file since the dataset is smaller and startup titles are often non-standard.

### Anti-detection

| Measure | LinkedIn | Indeed | YC |
|---------|----------|-------|----|
| Rate cap | 6 req/min | 8 req/min | n/a |
| Request delay | jobspy / site latency | jobspy / site latency | 1.5–3.5s |
| User agent rotation | 5 browser UAs | 5 browser UAs | 5 browser UAs |
| Failure backoff | 60s × failure count | 45s × failure count | n/a |
| Circuit breaker | 5 consecutive true failures | 5 consecutive true failures | 30 consecutive 404s |
| Rate-limit alerting | manifest + SNS warning | manifest + SNS warning | n/a |

Empty result sets are treated as valid zero-yield tasks, not request failures.

### Deduplication

**LinkedIn/Indeed daily CSVs:**
- Same-day output dedupe uses a hierarchy: native `id` -> canonicalized `job_url` -> stable content fingerprint
- Canonical URLs strip query parameters so tracking URLs collapse to one posting
- Daily reruns merge into today's CSVs instead of discarding earlier same-day data
- `_seen_job_ids.json` is retained for ever-seen bookkeeping, but it does not suppress today's snapshot output
- Index capped at 500K entries

**YC:**
- Tracks job IDs (numeric, from URL) in `_seen_yc_ids.json`
- Expired jobs (HTTP 302) are marked so they aren't re-checked
- `_yc_scraper_state.json` tracks the highest scanned ID to avoid re-scanning old ranges

### Output datasets

**Daily raw CSV snapshots**
- `data/scraped/YYYY-MM-DD_swe_jobs.csv`
- `data/scraped/YYYY-MM-DD_non_swe_jobs.csv`
- `data/scraped/YYYY-MM-DD_yc_jobs.csv`

**Canonical parquet**
- `data/unified.parquet`
- One row per globally unique posting
- Best analog to the Kaggle / Hugging Face LinkedIn postings corpus
- Includes quality / audit fields such as `date_posted_raw`, `date_posted_quality_flag`, `work_type_raw`, `opening_fingerprint`, `is_aggregator_posting`, and `aggregator_name`

**Daily observations parquet**
- `data/unified_observations.parquet`
- One row per posting per `scrape_date`
- Use this for survival / repost / duration analyses, not as the primary public benchmark table

### Output schema

**LinkedIn/Indeed** (`*_swe_jobs.csv` and `*_non_swe_jobs.csv`):

| Column | Description |
|--------|-------------|
| `site` | Source site (linkedin, indeed) |
| `id` | Job ID |
| `title` | Job title |
| `company` | Company name |
| `location` | Job location |
| `date_posted` | When the job was posted |
| `date_posted_raw` | Raw site-provided posting date |
| `date_posted_quality_flag` | `valid`, `missing`, `parse_failed`, `future`, or `stale_gt_60d` |
| `description` | Full job description (markdown) |
| `job_level` | Seniority level (entry level, mid-senior level, etc.) |
| `job_type` | Full-time, part-time, contract, etc. |
| `is_remote` | Remote flag |
| `min_amount` / `max_amount` | Salary range (when listed) |
| `job_url` | Direct link to the posting |
| `opening_fingerprint` | Approximate source-agnostic opening key for cross-source grouping |
| `is_aggregator_posting` | Whether the row looks like an intermediary / reposting site |
| `aggregator_name` | Normalized intermediary label when detected |
| `company_industry` | Industry classification |
| `company_num_employees` | Company size |
| `skills` | Listed skills |
| `scrape_date` | Date this scrape ran |

**YC** (`*_yc_jobs.csv`):

| Column | Description |
|--------|-------------|
| `source` | Always `yc_workatastartup` |
| `id` | Numeric job ID |
| `title` | Job title |
| `company` | Company name |
| `company_url` | Company website |
| `location` | Location(s), pipe-separated |
| `is_remote` | Remote flag |
| `date_posted` | ISO timestamp |
| `description` | Full description (plain text) |
| `job_type` | FULL_TIME, PART_TIME, etc. |
| `salary_currency` / `salary_min` / `salary_max` / `salary_unit` | Salary details |
| `job_url` | Link to workatastartup.com listing |
| `yc_batch` | YC batch (e.g., W24, S25) |
| `scrape_date` | Date this scrape ran |

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
tail -20 logs/yc_$(date +%Y-%m-%d).log

# Accumulated data summary
echo "Total daily files: $(ls data/scraped/*_swe_jobs.csv 2>/dev/null | wc -l)"
echo "Total SWE rows: $(cat data/scraped/*_swe_jobs.csv 2>/dev/null | wc -l)"
echo "Total YC files: $(ls data/scraped/*_yc_jobs.csv 2>/dev/null | wc -l)"
echo "Disk usage: $(du -sh data/scraped/)"
```

### Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| 0 results for all queries | LinkedIn rate limiting / IP blocked | Wait 24h, or use a proxy (`--proxy` support in jobspy) |
| Circuit breaker triggered | Too many consecutive failures | Check logs; increase delays in config |
| Stale lock file | Previous run crashed | Auto-clears after 6h, or `rm .scraper.lock` |
| Missing cron output | Cron environment issue | Check `logs/cron.log`; ensure absolute paths |
| YC scraper gets 0 jobs | Site structure changed | Check `logs/yc_*.log`; may need parser update |

## Alerts

After each daily run, the scraper sends a summary email via AWS SNS with:

- **Status**: success/failure/warning, retry count
- **Today's counts**: SWE jobs, control jobs, YC jobs, total
- **Trend**: comparison to yesterday and 7-day average
- **Data quality**: fill rates for key fields (seniority, location, company)
- **Sample postings**: 5 random job titles from today's scrape
- **Cumulative stats**: total days scraped, date range, unified dataset size
- **Errors**: recent error/warning lines from logs (if any)

A JSON status file is also written to `data/scraper_status.json` and synced to S3 after each run.

### Setup

SNS alerting requires two environment variables in `~/research/.env` on EC2:

```bash
S3_BUCKET=s3://swe-labor-research
SNS_TOPIC_ARN=arn:aws:sns:us-east-2:812064793967:scraper-alerts
```

To add a new email subscriber:

```bash
# On EC2:
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-2:812064793967:scraper-alerts \
  --protocol email \
  --notification-endpoint new-person@email.com
# They must click the confirmation link in their inbox
```

### Alert triggers

| Condition | Alert type |
|-----------|-----------|
| Scrape completes normally | `success` |
| Full run collects < 10 SWE jobs | `warning` (suspiciously low) |
| Scraper fails after 3 retries | `failure` |

### Test alerting

```bash
# Send a test email for a specific date
source .venv/bin/activate
python scraper/send_alert.py --status success --swe-count 100 --total-count 150 --attempt 1 --date 2026-03-17

# Send a test for today
python scraper/send_alert.py --status success --swe-count 0 --total-count 0 --attempt 1 --message "Test alert"
```

## Infrastructure

The scraper runs daily on EC2 with S3 for durable storage. See `docs/infrastructure-setup.md` for full details.

- **EC2:** t3.small in us-east-2, Amazon Linux 2023, cron at 6 AM UTC
- **S3:** `s3://swe-labor-research` — stores daily CSVs, manifests, and scraper status
- Local disk on EC2 is the staging area; S3 sync happens after each successful run

### SSH into EC2

```bash
ssh -i ~/path/to/scraper-key.pem ec2-user@<PUBLIC_IP>
```

To find the public IP: **AWS Console** → **EC2** (us-east-2) → **Instances** → `swe-scraper`

Once connected:

```bash
cd ~/research && source .env && source .venv/bin/activate

# Check cron job
crontab -l

# Check today's logs
tail -f logs/scrape_$(date +%Y-%m-%d).log

# Run scraper manually
./scraper/run_daily.sh

# Pull latest code
git pull origin main
```

### S3 data access

```bash
# List recent scrapes
aws s3 ls s3://swe-labor-research/scraped/

# Download all data locally
aws s3 sync s3://swe-labor-research/scraped/ data/scraped/

# Check scraper status
aws s3 cp s3://swe-labor-research/scraper_status.json - | python3 -m json.tool
```

## Accessing the data (collaborators)

All scraped data is synced daily to S3. You need AWS credentials with read access to the bucket.

### 1. Get credentials

Ask the project lead for an IAM access key scoped to `s3://swe-labor-research` (read-only). Then configure:

```bash
aws configure
# AWS Access Key ID: <your key>
# AWS Secret Access Key: <your secret>
# Default region: us-east-1
# Default output format: json
```

### 2. Download the data

```bash
# Download all scraped data
aws s3 sync s3://swe-labor-research/scraped/ data/scraped/

# Download a specific day
aws s3 cp s3://swe-labor-research/scraped/2026-03-17_swe_jobs.csv data/scraped/

# Just list what's available
aws s3 ls s3://swe-labor-research/scraped/ | tail -20
```

### 3. Build the unified dataset

After downloading, rebuild the analysis-ready parquet:

```bash
source .venv/bin/activate
python3 scraper/harmonize.py
# Outputs: data/unified.parquet
```

The unified parquet is also synced to S3 daily (`s3://swe-labor-research/unified.parquet`), but rebuilding locally ensures you get the latest harmonization logic.

## Checking scraper health

The scraper runs daily at 6 AM UTC on EC2. Here's how to check if it's working.

### From your local machine (via S3)

```bash
# Check the latest scraper status (updated after each run)
aws s3 cp s3://swe-labor-research/scraper_status.json - | python3 -m json.tool

# See the most recent files — are dates current?
aws s3 ls s3://swe-labor-research/scraped/ | tail -10

# Count how many days of data we have
aws s3 ls s3://swe-labor-research/scraped/ | grep "_swe_jobs.csv" | wc -l

# Check file sizes (if a CSV is suspiciously small, the scrape may have been rate-limited)
aws s3 ls s3://swe-labor-research/scraped/ --human-readable | grep "$(date +%Y-%m)" | head -20
```

### From EC2 (SSH in)

```bash
ssh -i ~/path/to/scraper-key.pem ec2-user@<PUBLIC_IP>
# Get the public IP: AWS Console → EC2 (us-east-2) → Instances → swe-scraper

cd ~/research && source .env && source .venv/bin/activate

# Is the scraper running right now?
ls -la .scraper.lock 2>/dev/null && echo "Running (PID: $(cat .scraper.lock))" || echo "Not running"

# Check today's log
tail -50 logs/scrape_$(date +%Y-%m-%d).log

# Check the cron job is still installed
crontab -l

# Quick data health check
echo "SWE files: $(ls data/scraped/*_swe_jobs.csv 2>/dev/null | wc -l)"
echo "Latest SWE file: $(ls -t data/scraped/*_swe_jobs.csv 2>/dev/null | head -1)"
echo "Disk usage: $(du -sh data/scraped/)"

# Row counts for the last 7 days
for f in $(ls -t data/scraped/*_swe_jobs.csv 2>/dev/null | head -7); do
  echo "$(basename $f): $(($(wc -l < $f) - 1)) rows"
done
```

### What to look for

| Signal | Healthy | Unhealthy |
|--------|---------|-----------|
| Latest file date | Today or yesterday | 2+ days old |
| SWE CSV row count | 200–2000+ rows | < 50 rows (rate-limited) or 0 |
| `scraper_status.json` status | `success` | `failure` or `warning` |
| Log file | Ends with "Done (exit code 0)" | Retries, errors, or missing |
| Lock file | Absent (not currently running) | Present for 6+ hours (stuck) |

### If something is wrong

1. **No recent files** → SSH into EC2, check `crontab -l` and logs
2. **Files exist but tiny** → LinkedIn is rate-limiting; check logs for "circuit breaker" or "0 results"
3. **Scraper stuck** → `rm .scraper.lock` on EC2 and re-run manually
4. **EC2 instance stopped** → Restart from AWS Console (us-east-2), IP will change

## Research context

This project studies how AI coding agents are restructuring SWE roles across the entire seniority ladder. See:

- `docs/1-research-design.md` — Canonical research questions and empirical strategy
- `docs/2-interview-design-mechanisms.md` — Canonical interview design for mechanism evidence
- `docs/3-literature-review.md` — Literature review
- `docs/4-literature-sources.md` — Reference list
- `docs/5-publication-targets-2026-2027.md` — Conference and venue strategy
- `docs/6-methods-learning.md` — Methodology learning guide tied to the March 6 notes
- `docs/infrastructure-setup.md` — Scraper infrastructure and operations
- `docs/archive/2026-03/analysis-validation-plan.md` — Archived measurement / validation plan

### Research questions

1. How did employer-side SWE requirements restructure across seniority levels from 2023 to 2026?
2. Which requirements moved downward into junior postings, and which senior-role responsibilities shifted toward AI-enabled orchestration?
3. Do employer-side AI requirements outpace observed workplace AI usage?
4. How do workers and hiring-side actors explain these changes?
