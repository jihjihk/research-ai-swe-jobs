# Alerting system — `send_alert.py`

Reference for maintaining the daily scraper alert email.
If you change scraping logic (new sites, manifest fields, output files), update the parsers here to match.

## Design principles

1. **Memory-light.** Never load a full CSV into memory. Stream rows (reservoir sampling for samples, counters for quality). Use binary-mode line counting for row counts. Read cron.log by scanning backwards from EOF — only today's portion, not the whole file.
2. **Fast.** Read each log file at most once and pass the lines to every function that needs them. Compute shared values (7-day average, scrape stats) once. Early-exit when scanning for markers (e.g. Indeed missing-site check). Target: alerting adds <3s on top of the existing cumulative-stats scan.
3. **Comprehensive.** Surface everything a human needs to decide "is today's scrape healthy?" in one email, ordered by urgency: anomaly flags first, then site status, counts, stats, quality, errors last.

## Email sections (in order)

| # | Section | Source | Parser |
|---|---------|--------|--------|
| 1 | Header | CLI args (`--status`, `--attempt`, `--message`, `--date`) | — |
| 2 | Anomaly flags | CSVs + scrape log | `get_anomaly_flags()` |
| 3 | Per-site status | `{date}_manifest.json`, `scrape_{date}.log`, `yc_{date}.log` | `get_site_status()` |
| 4 | Today's counts | `data/scraped/{date}_*.csv` | `get_today_files()` |
| 5 | Trend | Past 7 days of `{date}_swe_jobs.csv` | `get_previous_day_count()`, `get_7day_average()` |
| 6 | Scrape stats | `{date}_manifest.json`, `scrape_{date}.log` | `get_scrape_stats()` |
| 7 | Data quality | `{date}_swe_jobs.csv` | `get_data_quality()` |
| 8 | Sample postings | `{date}_swe_jobs.csv` | `get_sample_titles()` |
| 9 | Cumulative | All `*_swe_jobs.csv` + `unified.parquet` | `get_cumulative_stats()` |
| 10 | Error details | `scrape_{date}.log`, `cron.log` | `get_traceback()` |

## Log patterns each parser depends on

When you change log output in the scrapers, these are the exact strings alerting looks for.

### `{date}_manifest.json` (produced by `scrape_linkedin_swe.py`)

The manifest is now the preferred source for:

- site-level completion and task totals
- exception counts
- suspected rate-limit counts
- run parameters (sites, metros, rates, tiers, workers)

Key fields currently consumed by alerting:

- `site_summary.{site}.tasks_total`
- `site_summary.{site}.tasks_completed`
- `site_summary.{site}.tasks_failed`
- `site_summary.{site}.tasks_rate_limited`
- `rate_limit_summary.total`

### `scrape_{date}.log` (produced by `scrape_linkedin_swe.py`)

| Pattern | Used by | Example |
|---------|---------|---------|
| `--- [SITE] Starting:` | sanity / fallback parsing | `--- [LINKEDIN] Starting: 28 queries x 26 metros = 728 tasks, workers=3, rate=6.0 req/min ---` |
| `[site] query='...' metro='...'` | progress / fallback parsing | `[linkedin] query='software engineer' metro='seattle' ...` |
| `raw=N new=M dups=K duration=...` | fallback scrape stats | `[linkedin] query='software engineer' ... raw=60 new=52 dups=8 duration=21.1s` |
| `Empty result for query=` | zero-yield task diagnostics | `Empty result for query='UX designer' metro='philadelphia' in 0.1s` |
| `Suspected rate limit for query=` | rate-limit diagnostics | `Suspected rate limit for query='...' metro='...'` |
| `Circuit breaker hit` | site degradation detection | `[indeed] Circuit breaker hit ...` |
| `[ERROR]` / `[WARNING]` | traceback / anomaly details | Any line containing these tags |

### `yc_{date}.log` (produced by `scrape_yc.py`)

| Pattern | Used by | Example |
|---------|---------|---------|
| `YC scraper starting` | `get_site_status` | `YC scraper starting — 2026-03-18` |
| `Done — N YC jobs collected` | `get_site_status` | `Done — 24 YC jobs collected` |
| `No jobs collected` | `get_site_status` | `[WARNING] No jobs collected` |

### `cron.log` (appended by `run_daily.sh` and Python stderr)

| Pattern | Used by | Example |
|---------|---------|---------|
| `Traceback (most recent call last):` | `get_traceback` | Standard Python traceback blocks |
| Date prefix `YYYY-MM-DD[T ]` | `_read_cron_log_for_date` | Used to scope extraction to today only |

### CSV files (`data/scraped/`)

| File | Used by |
|------|---------|
| `{date}_swe_jobs.csv` | `get_today_files`, `get_previous_day_count`, `get_7day_average`, `get_data_quality`, `get_sample_titles` |
| `{date}_non_swe_jobs.csv` | `get_today_files` |
| `{date}_yc_jobs.csv` | `get_today_files` |
| `*_swe_jobs.csv` (all) | `get_cumulative_stats` |
| `unified.parquet` | `get_cumulative_stats` |

## Anomaly flags

| Flag | Trigger | Threshold |
|------|---------|-----------|
| COUNT DROP | Today's `--swe-count` < 50% of 7-day avg | 50% |
| HIGH DEDUP | Dedup rate from scrape stats | >90% |
| MISSING SITE | A site ran yesterday but has no log/marker today | Indeed: `[INDEED] Starting` absent. YC: `yc_{date}.log` absent. |
| ZERO JOBS | `--status success` but `--swe-count 0` | Exact zero |
| RATE LIMITING | Manifest shows repeated suspected rate-limit tasks | `rate_limit_summary.total >= 3` |

## Per-site status values

- **OK** — manifest/logs show the site completed its task set without a breaker
- **PARTIAL** — site completed some tasks but also logged failures, rate limits, or a breaker
- **FAILED** — site started but never completed meaningfully
- **WARNING** — YC-specific: completed but collected 0 jobs

## Adding a new scrape site

1. Prefer adding task/site summary fields to the manifest; use log parsing only as fallback.
2. If the site writes its own log file (like YC does), add a read for it in `get_site_status()` and `get_traceback()`.
3. Add a missing-site check in `get_anomaly_flags()` if the site should run daily.
4. Add the CSV filename pattern to `get_today_files()`.

## Performance budget

The script runs once per day after the scrape completes. Current profile (~3s total):

| Component | Time | Notes |
|-----------|------|-------|
| `get_cumulative_stats` | ~1.5s | Reads all historical CSVs via binary line count. Pre-existing, scales with days scraped. |
| `get_7day_average` | ~0.1s | 7 binary line counts |
| `get_data_quality` | ~0.1s | Streams one CSV |
| Log parsing (all new functions) | ~0.05s | Single pass over scrape log (~500KB), backward scan of cron.log |
| Everything else | ~1.2s | Python startup, pyarrow import, SNS call |

The new alerting functions collectively add <0.1s. The dominant cost is the pre-existing `get_cumulative_stats` which will grow linearly with days scraped — consider caching that count in a file if it becomes a problem.
