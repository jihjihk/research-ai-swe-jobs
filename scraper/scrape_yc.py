#!/usr/bin/env python3
"""
scrape_yc.py — Scrape all job listings from Y Combinator's Work at a Startup.

Strategy:
  1. Fetch the /companies listing page to get seed job IDs (server-rendered HTML)
  2. Scan recent job IDs (the ID space is sequential) to discover active listings
  3. For each active job (HTTP 200), extract structured JSON-LD data

Each job page at workatastartup.com/jobs/{id} that returns 200 contains a
schema.org/JobPosting JSON-LD block with title, company, salary, location,
description, and posting date.

Usage:
    python3 scraper/scrape_yc.py              # Full scrape
    python3 scraper/scrape_yc.py --test       # Test mode (small ID range)
    python3 scraper/scrape_yc.py --no-harmonize
"""

import argparse
import csv
import html as html_lib
import json
import logging
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install requests beautifulsoup4")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "scraped"
LOG_DIR = BASE_DIR / "logs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
today = datetime.now().strftime("%Y-%m-%d")
logger = logging.getLogger("scrape_yc")
logger.setLevel(logging.INFO)

fh = logging.FileHandler(LOG_DIR / f"yc_{today}.log")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = "https://www.workatastartup.com"

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0",
]

# Rate limiting
MIN_DELAY = 1.5
MAX_DELAY = 3.5
BATCH_PAUSE = 10  # Pause every N requests

# Dedup
SEEN_IDS_FILE = DATA_DIR / "_seen_yc_ids.json"
# Track the highest ID we've scanned so we don't re-scan old ranges
STATE_FILE = DATA_DIR / "_yc_scraper_state.json"

# How far above the highest known ID to scan
SCAN_AHEAD = 200
# How far back from the lowest seed ID to scan (catch jobs we missed)
SCAN_BACK = 500
# How far around the current seed window to rescan even if state has drifted
SEED_RESCAN_BACK = 50
SEED_RESCAN_AHEAD = 20

# Circuit breaker: stop scanning if too many consecutive 404s
MAX_CONSECUTIVE_404 = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_session():
    """Create a requests session with random UA."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return session


def load_seen_ids():
    """Load previously seen job IDs."""
    if SEEN_IDS_FILE.exists():
        try:
            with open(SEEN_IDS_FILE) as f:
                data = json.load(f)
            if len(data) > 200_000:
                data = dict(list(data.items())[-100_000:])
            return data
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_seen_ids(seen):
    with open(SEEN_IDS_FILE, "w") as f:
        json.dump(seen, f)


def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"highest_scanned_id": 0}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def polite_sleep():
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


def clean_html(html_str):
    """Convert HTML description to plain text."""
    if not html_str:
        return ""
    soup = BeautifulSoup(html_str, "html.parser")
    return soup.get_text(separator="\n", strip=True)[:5000]


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def parse_inertia_props(html_text):
    """Extract Inertia.js props from the data-page attribute."""
    match = re.search(r'data-page="([^"]+)"', html_text)
    if not match:
        return None
    try:
        decoded = html_lib.unescape(match.group(1))
        return json.loads(decoded).get("props", {})
    except (json.JSONDecodeError, AttributeError):
        return None


def get_seed_job_ids(session):
    """Get initial job IDs from the /jobs listing page (Inertia.js data-page)."""
    # Try /jobs first (has job listings), fall back to homepage
    for url in [f"{BASE_URL}/jobs", BASE_URL]:
        try:
            resp = session.get(url, timeout=30, allow_redirects=True)
            resp.raise_for_status()

            # Try Inertia.js data-page attribute (current site structure)
            props = parse_inertia_props(resp.text)
            if props and "jobs" in props:
                ids = [int(j["id"]) for j in props["jobs"] if "id" in j]
                if ids:
                    logger.info(f"Seed job IDs from {url}: {len(ids)} (Inertia.js)")
                    return sorted(ids)

            # Fallback: regex for /jobs/{id} links in HTML
            ids = [int(x) for x in set(re.findall(r'/jobs/(\d+)', resp.text))]
            if ids:
                logger.info(f"Seed job IDs from {url}: {len(ids)} (regex)")
                return sorted(ids)

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch seed IDs from {url}: {e}")
            continue

    logger.warning("Could not get seed IDs from any page")
    return []


def _parse_job_inertia(props, job_id, url):
    """Extract job data from Inertia.js props (current site structure)."""
    job_data = props.get("job", {})
    company = props.get("company", {})
    if not job_data or not job_data.get("title"):
        return None

    location = job_data.get("location", "")
    is_remote = bool(job_data.get("remote") or "remote" in location.lower())

    # Parse salary range string like "$120K - $180K"
    salary_min = ""
    salary_max = ""
    salary_currency = ""
    salary_range = job_data.get("salaryRange", "")
    if salary_range:
        salary_currency = "USD" if "$" in salary_range else ""
        amounts = re.findall(r'[\d,.]+', salary_range.replace("K", "000").replace("k", "000"))
        if len(amounts) >= 2:
            salary_min = amounts[0].replace(",", "")
            salary_max = amounts[1].replace(",", "")
        elif len(amounts) == 1:
            salary_min = amounts[0].replace(",", "")

    # Extract YC batch from company data
    yc_batch = company.get("batch", "")
    if not yc_batch:
        batch_match = re.search(r'\b([SWFX]\d{2})\b', company.get("name", ""))
        if batch_match:
            yc_batch = batch_match.group(1)

    return {
        "source": "yc_workatastartup",
        "id": str(job_id),
        "title": job_data.get("title", ""),
        "company": company.get("name", job_data.get("companyName", "")),
        "company_url": company.get("website", ""),
        "location": location,
        "is_remote": is_remote,
        "date_posted": job_data.get("createdAt", job_data.get("postedAt", "")),
        "description": clean_html(job_data.get("descriptionHtml", "")),
        "job_type": job_data.get("jobType", job_data.get("type", "")),
        "salary_currency": salary_currency,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_unit": "YEAR",
        "job_url": f"{BASE_URL}/jobs/{job_id}",
        "yc_batch": yc_batch,
        "scrape_date": today,
    }


def _parse_job_jsonld(resp_text, job_id, url):
    """Extract job data from JSON-LD script tag (legacy site structure)."""
    soup = BeautifulSoup(resp_text, "html.parser")
    script = soup.find("script", type="application/ld+json")
    if not script or not script.string:
        return None

    data = json.loads(script.string)
    if data.get("@type") != "JobPosting":
        return None

    org = data.get("hiringOrganization", {})
    salary = data.get("baseSalary", {})
    salary_val = salary.get("value", {}) if isinstance(salary, dict) else {}
    locations = data.get("jobLocation", [])
    if isinstance(locations, dict):
        locations = [locations]

    loc_parts = []
    for loc in locations:
        addr = loc.get("address", {})
        city = addr.get("addressLocality", "")
        state = addr.get("addressRegion", "")
        if city and state:
            loc_parts.append(f"{city}, {state}")
        elif city:
            loc_parts.append(city)
        elif state:
            loc_parts.append(state)

    is_remote = data.get("jobLocationType", "") == "TELECOMMUTE"
    if is_remote:
        loc_parts.append("Remote")

    yc_batch = ""
    batch_match = re.search(r'\(([SWFX]\d{2})\)', resp_text)
    if batch_match:
        yc_batch = batch_match.group(1)

    return {
        "source": "yc_workatastartup",
        "id": str(job_id),
        "title": data.get("title", ""),
        "company": org.get("name", ""),
        "company_url": org.get("sameAs", ""),
        "location": " | ".join(loc_parts),
        "is_remote": is_remote,
        "date_posted": data.get("datePosted", ""),
        "description": clean_html(data.get("description", "")),
        "job_type": data.get("employmentType", ""),
        "salary_currency": salary.get("currency", "") if isinstance(salary, dict) else "",
        "salary_min": salary_val.get("minValue", "") if isinstance(salary_val, dict) else "",
        "salary_max": salary_val.get("maxValue", "") if isinstance(salary_val, dict) else "",
        "salary_unit": salary_val.get("unitText", "") if isinstance(salary_val, dict) else "",
        "job_url": data.get("url", url),
        "yc_batch": yc_batch,
        "scrape_date": today,
    }


def fetch_job(session, job_id):
    """Fetch a single job page. Returns (status, job_dict_or_None).

    Tries Inertia.js data-page first (current site), falls back to JSON-LD (legacy).
    - 200 with job data → active job, returns parsed data
    - 302 → expired/closed job (redirects to company page)
    - 404 → ID doesn't exist
    """
    url = f"{BASE_URL}/jobs/{job_id}"
    try:
        resp = session.get(url, timeout=20, allow_redirects=False)

        if resp.status_code == 302:
            return 302, None

        if resp.status_code == 404:
            return 404, None

        if resp.status_code != 200:
            return resp.status_code, None

        # Try Inertia.js data-page first (current site structure)
        props = parse_inertia_props(resp.text)
        if props and "job" in props:
            job = _parse_job_inertia(props, job_id, url)
            if job:
                return 200, job

        # Fallback to JSON-LD (legacy structure)
        job = _parse_job_jsonld(resp.text, job_id, url)
        if job:
            return 200, job

        return 200, None

    except json.JSONDecodeError:
        logger.warning(f"Bad JSON for job {job_id}")
        return 200, None
    except requests.RequestException as e:
        logger.warning(f"Request failed for job {job_id}: {e}")
        return -1, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_scraper(args):
    logger.info("=" * 60)
    logger.info(f"YC scraper starting — {today}")
    logger.info(f"Mode: {'test' if args.test else 'full'}")

    session = get_session()
    seen_ids = load_seen_ids()
    state = load_state()
    all_jobs = []

    # Step 1: Get seed IDs from the listing page
    seed_ids = get_seed_job_ids(session)
    polite_sleep()

    # Step 2: Determine scan range
    if seed_ids:
        max_seed = max(seed_ids)
        min_seed = min(seed_ids)
    else:
        max_seed = state.get("highest_scanned_id", 91500)
        min_seed = max_seed - 500

    prev_highest = state.get("highest_scanned_id", 0)

    if args.test:
        # Test: scan a small range around the seeds
        scan_start = max_seed - 50
        scan_end = max_seed + 20
    else:
        # Full:
        # 1. Always rescan around the current seed window.
        # 2. If state has progressed beyond the visible seeds, continue scanning forward
        #    from a recent trailing window near prev_highest.
        seed_window_start = max(min_seed - SCAN_BACK, min_seed - SEED_RESCAN_BACK, 1)
        seed_window_end = max_seed + SCAN_AHEAD
        if prev_highest > seed_window_end:
            scan_start = max(prev_highest - SCAN_BACK, 1)
            scan_end = max(prev_highest + SCAN_AHEAD, seed_window_end)
            logger.warning(
                "Saved YC state is ahead of current seed IDs; scanning around the saved "
                f"frontier instead of the lower seed window (prev_highest={prev_highest}, "
                f"seed_max={max_seed})"
            )
        else:
            scan_start = seed_window_start
            scan_end = max(seed_window_end, prev_highest + SCAN_AHEAD)

    scan_start = max(int(scan_start), 1)
    scan_end = max(int(scan_end), scan_start)

    logger.info(f"Scan range: {scan_start} — {scan_end} ({scan_end - scan_start + 1} IDs)")
    logger.info(f"Previously seen: {len(seen_ids)} jobs")
    logger.info(
        f"Seed window: min_seed={min_seed}, max_seed={max_seed}, "
        f"prev_highest={prev_highest}"
    )

    # Step 3: Scan the ID range
    consecutive_404 = 0
    scanned = 0
    skipped = 0
    fetched = 0

    for job_id in range(scan_start, scan_end + 1):
        str_id = str(job_id)

        # Skip already-seen active jobs
        if str_id in seen_ids:
            skipped += 1
            continue

        status, job = fetch_job(session, job_id)
        scanned += 1

        if status == 200 and job:
            all_jobs.append(job)
            seen_ids[str_id] = today
            consecutive_404 = 0
            fetched += 1
            if fetched % 10 == 0:
                logger.info(f"  Fetched {fetched} jobs so far (scanned {scanned})...")
        elif status == 302:
            # Expired job — mark as seen so we don't re-check
            seen_ids[str_id] = f"{today}_expired"
            consecutive_404 = 0
        elif status == 404:
            consecutive_404 += 1
            if consecutive_404 >= MAX_CONSECUTIVE_404:
                logger.info(f"Hit {MAX_CONSECUTIVE_404} consecutive 404s at ID {job_id}, stopping scan")
                break
        else:
            consecutive_404 = 0

        # Rate limiting
        if scanned % BATCH_PAUSE == 0:
            time.sleep(random.uniform(3, 6))
        else:
            polite_sleep()

    logger.info(f"Scan complete: {scanned} checked, {fetched} new jobs, {skipped} skipped (already seen)")

    # Update state
    state["highest_scanned_id"] = max(scan_end, state.get("highest_scanned_id", 0))
    state["last_run"] = today
    state["last_run_jobs"] = fetched
    save_state(state)

    # Save results
    if all_jobs:
        output_file = DATA_DIR / f"{today}_yc_jobs.csv"

        fieldnames = [
            "source", "id", "title", "company", "company_url", "location",
            "is_remote", "date_posted", "description", "job_type",
            "salary_currency", "salary_min", "salary_max", "salary_unit",
            "job_url", "yc_batch", "scrape_date",
        ]

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_jobs)

        logger.info(f"Saved {len(all_jobs)} jobs to {output_file}")

        # Manifest
        manifest = {
            "scrape_date": today,
            "source": "yc_workatastartup",
            "mode": "test" if args.test else "full",
            "seed_min_id": min_seed,
            "seed_max_id": max_seed,
            "prev_highest_scanned_id": prev_highest,
            "scan_range": [scan_start, scan_end],
            "ids_scanned": scanned,
            "ids_skipped": skipped,
            "jobs_collected": len(all_jobs),
            "dedup_index_size": len(seen_ids),
            "timestamp": datetime.now().isoformat(),
        }
        manifest_file = DATA_DIR / f"{today}_yc_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest: {manifest_file}")
    else:
        logger.warning("No jobs collected")

    save_seen_ids(seen_ids)

    # Harmonize
    if not args.no_harmonize:
        harmonize_script = Path(__file__).parent / "harmonize.py"
        if harmonize_script.exists():
            logger.info("Running harmonize.py...")
            try:
                subprocess.run(
                    [sys.executable, str(harmonize_script)],
                    check=True, capture_output=True, text=True,
                )
                logger.info("Harmonization complete")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Harmonization failed: {e.stderr[:200]}")

    logger.info(f"Done — {len(all_jobs)} YC jobs collected")
    return len(all_jobs)


def main():
    parser = argparse.ArgumentParser(description="Scrape YC Work at a Startup jobs")
    parser.add_argument("--test", action="store_true",
                        help="Test mode (small ID range, ~70 IDs)")
    parser.add_argument("--no-harmonize", action="store_true",
                        help="Skip harmonization step")
    args = parser.parse_args()

    count = run_scraper(args)
    sys.exit(0 if count > 0 else 1)


if __name__ == "__main__":
    main()
