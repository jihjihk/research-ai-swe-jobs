# The AI Restructuring of the SWE Seniority Ladder

Research project studying how AI coding agents are restructuring software engineer roles across the entire seniority ladder — junior roles disappearing or absorbing senior requirements, senior roles shedding people-management and gaining AI-orchestration skills. Uses job postings data (Kaggle 2023–2024, daily LinkedIn/Indeed/YC scraping 2026+).

## Project Structure

```
research/
├── README.md
├── setup.sh                       # One-command setup for new machines
├── alerts.conf                    # Alert channel configuration
│
├── scraper/                       # Scraping pipeline
│   ├── scrape_linkedin_swe.py     # LinkedIn + Indeed scraper (python-jobspy)
│   ├── scrape_yc.py               # Y Combinator / Work at a Startup scraper
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
│   ├── ec2-setup.md               # EC2 & S3 infrastructure docs
│   ├── session-summary.md         # Prior analysis session notes
│   └── sources.txt                # Reference sources
│
├── data/                          # (gitignored)
│   ├── unified.parquet            # Harmonized dataset (all sources, analysis-ready)
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
3. Install dependencies (`python-jobspy`, `pandas`, `pyarrow`, `requests`, `beautifulsoup4`)
4. Run a test scrape to verify everything works
5. Offer to install the daily cron job

### Manual setup (if you prefer)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install python-jobspy pandas pyarrow requests beautifulsoup4
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
| **LinkedIn** | `scrape_linkedin_swe.py` | `*_swe_jobs.csv`, `*_non_swe_jobs.csv` | SWE + adjacent + control roles, 20 US cities |
| **Indeed** | `scrape_linkedin_swe.py` | (same files) | Same queries, same cities |
| **YC (Work at a Startup)** | `scrape_yc.py` | `*_yc_jobs.csv` | All roles at YC-funded startups |
| **Kaggle** | (static download) | `kaggle-linkedin-jobs-2023-2024/` | Historical LinkedIn data (2023–2024) |

## Scraper Usage

### LinkedIn + Indeed

```bash
# Test (1 query, 1 city, 5 results per site — ~1 minute)
python3 scraper/scrape_linkedin_swe.py --test

# Quick run (4 queries x 10 cities — ~30 min for both sites)
python3 scraper/scrape_linkedin_swe.py --quick

# Full run (10 SWE + 8 adjacent + 10 control queries x 20 cities — ~2-3 hours)
python3 scraper/scrape_linkedin_swe.py

# Single site only
python3 scraper/scrape_linkedin_swe.py --sites linkedin
python3 scraper/scrape_linkedin_swe.py --sites indeed

# Run specific tiers only
python3 scraper/scrape_linkedin_swe.py --tiers swe            # SWE only
python3 scraper/scrape_linkedin_swe.py --tiers swe adjacent   # Skip control

# More results per search
python3 scraper/scrape_linkedin_swe.py --results 50

# Catch up after a missed day
python3 scraper/scrape_linkedin_swe.py --hours-old 48

# Skip harmonization (just scrape)
python3 scraper/scrape_linkedin_swe.py --no-harmonize
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
- **Post-run summary** — logs row count and accumulated file count
- **S3 sync** — uploads results to S3 if `S3_BUCKET` is configured
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

### Query tiers (LinkedIn + Indeed)

Queries are organized into priority tiers. Tier 1 runs first so if the scraper gets rate-limited, the most important data is already collected.

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
| Request delay | 8–20s random | 5–12s random | 1.5–3.5s |
| Between-query pause | 15–30s | 10–20s | 3–6s (every 10 requests) |
| Between-site pause | 30–60s | 30–60s | n/a |
| User agent rotation | 5 browser UAs | 5 browser UAs | 5 browser UAs |
| Failure backoff | 60s × failure count | 45s × failure count | n/a |
| Circuit breaker | 5 consecutive failures | 5 consecutive failures | 30 consecutive 404s |

### Deduplication

**LinkedIn/Indeed:**
- Each job gets a hash (LinkedIn job ID, or MD5 of title+company+location)
- Hashes persist in `_seen_job_ids.json` across runs
- Cross-query dedup within a run (by `job_url`)
- Index capped at 500K entries

**YC:**
- Tracks job IDs (numeric, from URL) in `_seen_yc_ids.json`
- Expired jobs (HTTP 302) are marked so they aren't re-checked
- `_yc_scraper_state.json` tracks the highest scanned ID to avoid re-scanning old ranges

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

### Alert triggers

| Condition | Alert type |
|-----------|-----------|
| Scrape completes normally | `success` |
| Full run collects < 10 SWE jobs | `warning` (suspiciously low) |
| Scraper fails after 3 retries | `failure` |

## Infrastructure

The scraper runs daily on EC2 with S3 for durable storage. See `docs/ec2-setup.md` for full details.

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

The parquet file is not stored in S3 — it's rebuilt locally from the CSVs so you always get the latest harmonization logic.

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

- `docs/research-design-h1-h3.md` — Research questions and empirical strategy
- `docs/validation-plan.md` — ML approaches for each research question
- `docs/session-summary.md` — Prior analysis results

### Research questions

1. Are junior SWE roles disappearing or being redefined?
2. Which competencies migrated from senior to junior postings, and when?
3. Was there a structural break in late 2025?
4. Is this SWE-specific or a broader labor market trend?
5. What should replace the broken junior training pipeline?
6. Are senior SWE roles shedding management requirements and gaining AI-orchestration ones?
7. Does the current restructuring follow the pattern of prior platform shifts (mainframe → PC → web → mobile → AI)?
