#!/usr/bin/env python3
"""
Daily US SWE Job Scraper (LinkedIn + Indeed)
Scrapes public job postings for software engineering roles in the US.
Designed to run as a daily cron job.

Usage:
    python scrape_linkedin_swe.py                  # Full daily scrape (all sites)
    python scrape_linkedin_swe.py --test            # Test mode: 1 query, 1 city, 5 results
    python scrape_linkedin_swe.py --quick           # Quick mode: 4 queries, 10 cities (~15 min)
    python scrape_linkedin_swe.py --sites linkedin  # LinkedIn only
    python scrape_linkedin_swe.py --sites indeed    # Indeed only
    python scrape_linkedin_swe.py --results 50      # Custom results per query
    python scrape_linkedin_swe.py --hours-old 48    # Look back 48 hours
"""

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import time
from datetime import datetime, date
from pathlib import Path

import pandas as pd

from jobspy import scrape_jobs
from harmonize import harmonize_kaggle, harmonize_scraped

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent  # project root (one level up from scraper/)
DATA_DIR = BASE_DIR / "data" / "scraped"
LOG_DIR = BASE_DIR / "logs"
DEDUP_FILE = DATA_DIR / "_seen_job_ids.json"

# SWE title regex (matches the notebook's SWE_PATTERN)
SWE_PATTERN = re.compile(
    r'(?i)\b(software\s*(engineer|developer|dev)|swe|full[- ]?stack|front[- ]?end|'
    r'back[- ]?end|web\s*developer|mobile\s*developer|devops|platform\s*engineer|'
    r'data\s*engineer|ml\s*engineer|machine\s*learning\s*engineer|site\s*reliability)\b'
)

# Search queries — multiple queries to maximize coverage
SEARCH_QUERIES = [
    "software engineer",
    "software developer",
    "full stack engineer",
    "frontend engineer",
    "backend engineer",
    "devops engineer",
    "platform engineer",
    "data engineer",
    "ML engineer",
    "site reliability engineer",
    "mobile developer",
    "web developer",
]

# US cities to split searches (bypasses per-search result caps)
US_LOCATIONS = [
    "San Francisco, CA",
    "New York, NY",
    "Seattle, WA",
    "Austin, TX",
    "Boston, MA",
    "Los Angeles, CA",
    "Chicago, IL",
    "Denver, CO",
    "Atlanta, GA",
    "San Jose, CA",
    "Washington, DC",
    "Portland, OR",
    "Dallas, TX",
    "Miami, FL",
    "Phoenix, AZ",
    "Minneapolis, MN",
    "Philadelphia, PA",
    "Raleigh, NC",
    "San Diego, CA",
    "Detroit, MI",
]

# Per-site configuration
SITE_CONFIG = {
    "linkedin": {
        "min_delay": 8,
        "max_delay": 20,
        "between_query_delay": (15, 30),
        "max_consecutive_failures": 5,
        "failure_backoff": 60,
        "extra_kwargs": {
            "linkedin_fetch_description": True,
        },
    },
    "indeed": {
        "min_delay": 5,
        "max_delay": 12,
        "between_query_delay": (10, 20),
        "max_consecutive_failures": 5,
        "failure_backoff": 45,
        "extra_kwargs": {},
    },
}

ALL_SITES = list(SITE_CONFIG.keys())

# User agent rotation
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
]


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"scrape_{date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def load_seen_ids() -> set:
    if DEDUP_FILE.exists():
        with open(DEDUP_FILE) as f:
            data = json.load(f)
        return set(data.get("ids", []))
    return set()


def save_seen_ids(seen: set):
    DEDUP_FILE.parent.mkdir(parents=True, exist_ok=True)
    ids = list(seen)
    if len(ids) > 500_000:
        ids = ids[-500_000:]
    with open(DEDUP_FILE, "w") as f:
        json.dump({"ids": ids, "updated": datetime.now().isoformat()}, f)


def make_job_hash(row) -> str:
    """Create a dedup key from job ID or title+company combo."""
    if pd.notna(row.get("id")):
        return str(row["id"])
    key = f"{row.get('title', '')}|{row.get('company', '')}|{row.get('location', '')}"
    return hashlib.md5(key.encode()).hexdigest()


def is_swe_role(title: str) -> bool:
    if not isinstance(title, str):
        return False
    return bool(SWE_PATTERN.search(title))


def scrape_batch(site: str, search_term: str, location: str,
                 results_wanted: int, hours_old: int, logger) -> pd.DataFrame:
    """Run a single scrape query against one site."""
    ua = random.choice(USER_AGENTS)
    config = SITE_CONFIG[site]
    try:
        df = scrape_jobs(
            site_name=[site],
            search_term=search_term,
            location=location,
            results_wanted=results_wanted,
            hours_old=hours_old,
            country_indeed="usa",
            description_format="markdown",
            user_agent=ua,
            verbose=0,
            **config["extra_kwargs"],
        )
        return df
    except Exception as e:
        logger.warning(f"[{site}] Scrape failed for '{search_term}' in {location}: {e}")
        return pd.DataFrame()


def scrape_site(site: str, queries: list, locations: list,
                results_per: int, hours_old: int,
                seen_ids: set, logger) -> list:
    """Run all query×location combinations for one site. Returns list of DataFrames."""
    config = SITE_CONFIG[site]
    total = len(queries) * len(locations)
    all_jobs = []
    consecutive_failures = 0
    requests_done = 0

    logger.info(f"--- [{site.upper()}] Starting: {len(queries)} queries x "
                f"{len(locations)} locations = {total} requests ---")

    for qi, query in enumerate(queries):
        for loc in locations:
            requests_done += 1
            progress = f"[{site}:{requests_done}/{total}]"

            if consecutive_failures >= config["max_consecutive_failures"]:
                logger.error(f"{progress} Circuit breaker hit ({consecutive_failures} "
                             f"consecutive failures). Stopping {site}.")
                return all_jobs

            logger.info(f"{progress} '{query}' in {loc}...")

            df = scrape_batch(site, query, loc, results_per, hours_old, logger)

            if df.empty:
                consecutive_failures += 1
                backoff = config["failure_backoff"] * consecutive_failures
                logger.info(f"  No results (failures: {consecutive_failures}), "
                            f"backing off {backoff}s")
                time.sleep(backoff)
                continue

            consecutive_failures = 0
            logger.info(f"  Got {len(df)} raw results")

            # Deduplicate
            df["_hash"] = df.apply(make_job_hash, axis=1)
            new_mask = ~df["_hash"].isin(seen_ids)
            df_new = df[new_mask].copy()
            seen_ids.update(df["_hash"].tolist())
            logger.info(f"  After dedup: {len(df_new)} new "
                        f"({len(df) - len(df_new)} duplicates)")

            if not df_new.empty:
                all_jobs.append(df_new)

            delay = random.uniform(config["min_delay"], config["max_delay"])
            logger.info(f"  Waiting {delay:.1f}s...")
            time.sleep(delay)

        if consecutive_failures >= config["max_consecutive_failures"]:
            return all_jobs

        # Pause between query terms
        if qi < len(queries) - 1:
            lo, hi = config["between_query_delay"]
            pause = random.uniform(lo, hi)
            logger.info(f"  [{site}] Next query in {pause:.0f}s...")
            time.sleep(pause)

    return all_jobs


def run_scraper(args):
    logger = setup_logging()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    sites = args.sites
    logger.info(f"=== Starting scrape for {today} | sites: {', '.join(sites)} ===")

    # Load dedup index
    seen_ids = load_seen_ids()
    logger.info(f"Dedup index: {len(seen_ids):,} previously seen job IDs")

    # Determine queries and locations
    if args.test:
        queries = SEARCH_QUERIES[:1]
        locations = US_LOCATIONS[:1]
        results_per = 5
        logger.info("TEST MODE: 1 query, 1 location, 5 results")
    elif args.quick:
        queries = SEARCH_QUERIES[:4]
        locations = US_LOCATIONS[:10]
        results_per = args.results
        logger.info("QUICK MODE: 4 queries, 10 cities")
    else:
        queries = SEARCH_QUERIES
        locations = US_LOCATIONS
        results_per = args.results

    hours_old = args.hours_old

    # Scrape each site sequentially (separate rate limit domains)
    all_jobs = []
    for site in sites:
        if site not in SITE_CONFIG:
            logger.warning(f"Unknown site '{site}', skipping. "
                           f"Available: {', '.join(ALL_SITES)}")
            continue

        site_jobs = scrape_site(
            site, queries, locations, results_per, hours_old, seen_ids, logger
        )
        if site_jobs:
            all_jobs.extend(site_jobs)
            logger.info(f"[{site.upper()}] Collected {sum(len(df) for df in site_jobs)} "
                        f"new jobs across {len(site_jobs)} batches")

        # Pause between sites
        if site != sites[-1]:
            pause = random.uniform(30, 60)
            logger.info(f"=== Switching sites, waiting {pause:.0f}s ===")
            time.sleep(pause)

    # Combine results
    if not all_jobs:
        logger.warning("No jobs collected. Exiting.")
        save_seen_ids(seen_ids)
        return

    combined = pd.concat(all_jobs, ignore_index=True)
    combined.drop(columns=["_hash"], inplace=True, errors="ignore")

    # Final dedup (cross-query and cross-site)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["job_url"], keep="first")
    logger.info(f"Cross-query/site dedup: {before} -> {len(combined)} jobs")

    # Filter to SWE roles
    swe_mask = combined["title"].apply(is_swe_role)
    swe_jobs = combined[swe_mask].copy()
    non_swe = combined[~swe_mask].copy()
    logger.info(f"SWE filter: {len(swe_jobs)} SWE roles, "
                f"{len(non_swe)} non-SWE (saved separately)")

    # Add metadata
    for df in [swe_jobs, non_swe]:
        df["scrape_date"] = today

    # Save run manifest (full config for reproducibility)
    manifest = {
        "scrape_date": today,
        "sites": sites,
        "queries": queries,
        "locations": locations,
        "results_per_query": results_per,
        "hours_old": hours_old,
        "mode": "test" if args.test else ("quick" if args.quick else "full"),
        "swe_count": len(swe_jobs),
        "non_swe_count": len(non_swe),
        "total_raw": len(combined),
        "dedup_index_size": len(seen_ids),
        "sites_collected": combined["site"].value_counts().to_dict() if "site" in combined.columns else {},
        "timestamp": datetime.now().isoformat(),
    }
    manifest_file = DATA_DIR / f"{today}_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved run manifest to {manifest_file}")

    # Save daily CSVs
    swe_file = DATA_DIR / f"{today}_swe_jobs.csv"
    swe_jobs.to_csv(swe_file, index=False)
    logger.info(f"Saved {len(swe_jobs)} SWE jobs to {swe_file}")

    if len(non_swe) > 0:
        non_swe_file = DATA_DIR / f"{today}_non_swe_jobs.csv"
        non_swe.to_csv(non_swe_file, index=False)
        logger.info(f"Saved {len(non_swe)} non-SWE jobs to {non_swe_file}")

    # Save dedup index
    save_seen_ids(seen_ids)

    # --- Harmonize into unified dataset ---
    if not args.no_harmonize:
        logger.info("=== Harmonizing into unified dataset ===")
        try:
            unified_path = BASE_DIR / "data" / "unified.parquet"
            parts = []

            kaggle_path = BASE_DIR / "data" / "kaggle-linkedin-jobs-2023-2024" / "postings.csv"
            if kaggle_path.exists():
                parts.append(harmonize_kaggle(str(kaggle_path)))

            parts.append(harmonize_scraped(str(DATA_DIR)))

            if parts:
                unified = pd.concat(parts, ignore_index=True)
                unified = unified.drop_duplicates(subset=["job_id"], keep="first")
                unified.to_parquet(unified_path, index=False)
                swe_count = unified["is_swe"].sum()
                logger.info(f"Unified dataset: {len(unified):,} total, "
                            f"{swe_count:,} SWE -> {unified_path}")
        except Exception as e:
            logger.warning(f"Harmonization failed (non-fatal): {e}")

    # Summary stats
    logger.info("=== Summary ===")
    logger.info(f"Total unique jobs collected: {len(combined)}")
    logger.info(f"SWE jobs: {len(swe_jobs)}")
    if len(swe_jobs) > 0:
        # Per-site breakdown
        if "site" in swe_jobs.columns:
            logger.info("By site:")
            for site_name, count in swe_jobs["site"].value_counts().items():
                logger.info(f"  {site_name}: {count}")
        logger.info("Seniority distribution:")
        seniority_col = "job_level" if "job_level" in swe_jobs.columns else "seniority"
        if seniority_col in swe_jobs.columns:
            for level, count in swe_jobs[seniority_col].value_counts().items():
                logger.info(f"  {level}: {count}")
        logger.info("Top locations:")
        if "location" in swe_jobs.columns:
            for loc, count in swe_jobs["location"].value_counts().head(10).items():
                logger.info(f"  {loc}: {count}")
        logger.info("Top companies:")
        company_col = "company" if "company" in swe_jobs.columns else "company_name"
        if company_col in swe_jobs.columns:
            for co, count in swe_jobs[company_col].value_counts().head(10).items():
                logger.info(f"  {co}: {count}")

    logger.info("=== Done ===")


def main():
    parser = argparse.ArgumentParser(
        description="Daily US SWE job scraper (LinkedIn + Indeed)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 1 query, 1 location, 5 results")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 4 queries, 10 cities (~15 min per site)")
    parser.add_argument("--sites", nargs="+", default=ALL_SITES,
                        choices=ALL_SITES,
                        help=f"Sites to scrape (default: all). Choices: {', '.join(ALL_SITES)}")
    parser.add_argument("--results", type=int, default=25,
                        help="Results per query-location combo (default: 25)")
    parser.add_argument("--hours-old", type=int, default=24,
                        help="Only fetch jobs posted in the last N hours (default: 24)")
    parser.add_argument("--no-harmonize", action="store_true",
                        help="Skip harmonization step after scraping")
    args = parser.parse_args()
    run_scraper(args)


if __name__ == "__main__":
    main()
