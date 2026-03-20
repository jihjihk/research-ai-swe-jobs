# Parallel MSA scraper rewrite — code review guide

## Status note

This document describes the first parallel rewrite proposal. It is no longer fully current.

The live code now also changes:

- `scraper/harmonize.py`
- `scraper/run_daily.sh`
- `scraper/send_alert.py`

Key production updates since this review draft:

- metro coverage expanded to 26 US metros
- rates calibrated upward to 6 req/min (LinkedIn) and 8 req/min (Indeed)
- empty result sets no longer count as scraper failures
- manifest-backed site/task auditing and rate-limit alerting were added
- harmonization now produces two parquet outputs:
  - `data/unified.parquet` for globally deduped canonical postings
  - `data/unified_observations.parquet` for one row per posting per scrape date
- daily reruns merge into existing same-day CSVs instead of discarding earlier same-day rows

## Goal

Capture all US SWE job postings (March–April 2026) by finishing a full scrape within the cron window. The previous scraper never completed before the next cron fired.

## Previous state

- **20 ad-hoc cities** (e.g. "San Francisco, CA", "San Jose, CA" — overlapping, missing suburbs)
- **Sequential execution**: one query at a time, one site after the other
- **25 results per query** (jobspy default) — missed ~50% of available postings
- **28 queries × 20 cities × 2 sites = 1,120 combos at ~34s each = ~27 hours per run**
- Rate limiting was handled by random sleep delays between requests
- Dedup used a plain Python set (no threading concern since everything was sequential)

## What changed

All changes are in **one file**: `scraper/scrape_linkedin_swe.py`. No changes to `harmonize.py`, `run_daily.sh`, or `send_alert.py` — output contract (CSV filenames, manifest JSON, harmonize call) is preserved.

### 1. MSA locations replace city list (superseded)

Initial rewrite switched from 20 overlapping cities to MSA-based metro areas. Production code now uses 26 metros with site-specific location strings.

**Review for**: Do the location strings look right? Any metros missing that matter for the research?

### 2. Four new thread-safe utility classes (lines ~168–270)

| Class | Purpose |
|---|---|
| `TokenBucketRateLimiter` | Blocks workers until a token is available. Production settings are now LinkedIn 6 req/min, Indeed 8 req/min. |
| `ThreadSafeCircuitBreaker` | Shared per site. Any worker can trip it (5 consecutive failures), all workers check before each request. |
| `ThreadSafeDedup` | Wraps `seen_ids` set with a `threading.Lock`. `filter_new(df)` hashes, filters, updates atomically. |
| `ThreadSafeResultCollector` | Accumulates DataFrames from workers. Production code now streams partial CSVs and finalizes outputs without rebuilding one giant in-memory list. |

**Review for**: Lock granularity — are any critical sections too wide or too narrow? Is the flush-to-disk logic in `ThreadSafeResultCollector` safe if a worker crashes mid-write?

### 3. Parallel worker architecture (lines ~280–360)

```
run_scraper()
  └─ ThreadPoolExecutor(max_workers=2)     ← one thread per site, run simultaneously
      ├─ scrape_site_parallel("linkedin")
      │   └─ ThreadPoolExecutor(max_workers=3)  ← 3 workers pulling from a shared Queue
      └─ scrape_site_parallel("indeed")
          └─ ThreadPoolExecutor(max_workers=3)
```

- `site_worker()` is the inner loop: pull `(query, location)` from queue → `rate_limiter.acquire()` (blocks) → `scrape_batch()` → dedup → collect
- Work items are shuffled before queuing to spread load across metros
- The rate limiter is the actual throttle — worker count doesn't affect request rate, only concurrency of in-flight processing

**Review for**: Is the queue drain logic correct (check `work_queue.empty()` then `get_nowait()`)? Could a worker spin if the queue is empty but `task_done()` hasn't been called? Is `task_done()` even needed here (we don't call `queue.join()`)?

### 4. Updated defaults and new CLI args (lines ~455–475)

| Arg | Old | New |
|---|---|---|
| `--results` | 25 | 100 |
| `--workers` | n/a | 3 (per site) |
| `--sequential` | n/a | flag to fall back to old behavior |

### 5. Legacy sequential path preserved (lines ~370–420)

The old `scrape_site()` function is untouched and used when `--sequential` is passed. This is the escape hatch if parallel mode causes issues.

**Review for**: Is it worth keeping this code long-term, or should it be removed once parallel is validated?

### 6. Manifest additions (expanded in production)

The manifest now also includes task-level auditing summaries, rate-limit counts, merged preexisting-row counts, and current output stats.

**Review for**: Does this break any downstream consumers that read the manifest?

## Expected behavior

| Mode | Queries | Metros | Results/query | Sites | Wall time |
|---|---|---|---|---|---|
| `--test` | 1 | 1 | 5 | 2 parallel | ~15s |
| `--quick` | 4 | 10 | 100 | 2 parallel | depends on site latency |
| full | 28 | 26 | 100 | 2 parallel | ~2.0–2.3h target |
| `--sequential` | 28 | 26 | 100 | sequential | much slower; debug only |

Memory: worst case ~800MB in-flight across both sites before flush. Safe on 16GB.

Rate limits: calibrated to LinkedIn 6 req/min and Indeed 8 req/min. Repeated suspected rate limits now trigger SNS warnings.

Output files are identical in format: `{date}_swe_jobs.csv`, `{date}_non_swe_jobs.csv`, `{date}_manifest.json`.

## Files to look at

Files that now matter in production:

- **`scraper/scrape_linkedin_swe.py`**
- **`scraper/harmonize.py`**
- **`scraper/run_daily.sh`**
- **`scraper/send_alert.py`**

## Test run output

Test mode completed successfully: 10 jobs (5 per site) in ~15s, both sites ran in parallel, manifest includes new fields.
