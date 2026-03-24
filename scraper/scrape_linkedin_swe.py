#!/usr/bin/env python3
"""
Daily US SWE job scraper (LinkedIn + Indeed).

Scrapes public job postings for software engineering and comparison roles in
major US metro areas. Designed to run as a daily cron job.

Usage:
    python scrape_linkedin_swe.py                  # Full daily scrape (all sites)
    python scrape_linkedin_swe.py --test          # Test mode: 1 query, 1 metro, 5 results
    python scrape_linkedin_swe.py --quick         # Quick mode: 4 queries, 10 metros
    python scrape_linkedin_swe.py --sites indeed  # Indeed only
    python scrape_linkedin_swe.py --results 50    # Custom results per query
    python scrape_linkedin_swe.py --hours-old 48  # Look back 48 hours
    python scrape_linkedin_swe.py --sequential    # Disable cross-site and per-site parallelism
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import queue
import random
import re
import shutil
import sys
import tempfile
import threading
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from jobspy import scrape_jobs
from harmonize import harmonize_scraped

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "scraped"
LOG_DIR = BASE_DIR / "logs"
DEDUP_FILE = DATA_DIR / "_seen_job_ids.json"

SWE_PATTERN = re.compile(
    r"(?i)\b(software\s*(engineer|developer|dev)|swe|full[- ]?stack|front[- ]?end|"
    r"back[- ]?end|web\s*developer|mobile\s*developer|devops|platform\s*engineer|"
    r"data\s*engineer|ml\s*engineer|machine\s*learning\s*engineer|site\s*reliability|"
    r"ai\s*engineer|ai[/ ]ml\s*engineer|llm\s*engineer|agent\s*engineer|"
    r"applied\s*ai\s*engineer|prompt\s*engineer|infrastructure\s*engineer|"
    r"founding\s*engineer|member\s*of\s*technical\s*staff|product\s*engineer)\b"
)

QUERY_TIERS = {
    "swe": [
        "software engineer",
        "full stack engineer",
        "frontend engineer",
        "backend engineer",
        "devops engineer",
        "data engineer",
        "machine learning engineer",
        "AI engineer",
        "mobile developer",
        "founding engineer",
    ],
    "adjacent": [
        "data scientist",
        "data analyst",
        "product manager",
        "UX designer",
        "QA engineer",
        "security engineer",
        "solutions engineer",
        "technical program manager",
    ],
    "control": [
        "civil engineer",
        "mechanical engineer",
        "electrical engineer",
        "chemical engineer",
        "registered nurse",
        "accountant",
        "financial analyst",
        "marketing manager",
        "human resources",
        "sales representative",
    ],
}

QUERY_TO_TIER = {
    query: tier
    for tier, queries in QUERY_TIERS.items()
    for query in queries
}

METRO_AREAS = [
    {
        "metro_id": "sf_bay",
        "name": "San Francisco Bay Area",
        "region": "west",
        "priority": 1,
        "locations": {
            "linkedin": "San Francisco Bay Area",
            "indeed": "San Francisco Bay Area, CA",
        },
    },
    {
        "metro_id": "new_york",
        "name": "New York City Metro",
        "region": "northeast",
        "priority": 2,
        "locations": {
            "linkedin": "New York City Metropolitan Area",
            "indeed": "New York, NY",
        },
    },
    {
        "metro_id": "seattle",
        "name": "Seattle Metro",
        "region": "west",
        "priority": 3,
        "locations": {
            "linkedin": "Seattle Metro Area, WA",
            "indeed": "Seattle, WA",
        },
    },
    {
        "metro_id": "los_angeles",
        "name": "Los Angeles Metro",
        "region": "west",
        "priority": 4,
        "locations": {
            "linkedin": "Los Angeles Metropolitan Area",
            "indeed": "Los Angeles, CA",
        },
    },
    {
        "metro_id": "boston",
        "name": "Boston Metro",
        "region": "northeast",
        "priority": 5,
        "locations": {
            "linkedin": "Boston, MA",
            "indeed": "Boston, MA",
        },
    },
    {
        "metro_id": "dc",
        "name": "Washington DC Metro",
        "region": "mid_atlantic",
        "priority": 6,
        "locations": {
            "linkedin": "Washington DC Metro Area",
            "indeed": "Washington, DC",
        },
    },
    {
        "metro_id": "austin",
        "name": "Austin Metro",
        "region": "south",
        "priority": 7,
        "locations": {
            "linkedin": "Austin, Texas Metropolitan Area",
            "indeed": "Austin, TX",
        },
    },
    {
        "metro_id": "dallas_fort_worth",
        "name": "Dallas-Fort Worth Metro",
        "region": "south",
        "priority": 8,
        "locations": {
            "linkedin": "Dallas-Fort Worth Metroplex",
            "indeed": "Dallas-Fort Worth, TX",
        },
    },
    {
        "metro_id": "chicago",
        "name": "Chicago Metro",
        "region": "midwest",
        "priority": 9,
        "locations": {
            "linkedin": "Chicago, IL",
            "indeed": "Chicago, IL",
        },
    },
    {
        "metro_id": "atlanta",
        "name": "Atlanta Metro",
        "region": "south",
        "priority": 10,
        "locations": {
            "linkedin": "Atlanta Metropolitan Area, GA",
            "indeed": "Atlanta, GA",
        },
    },
    {
        "metro_id": "raleigh_durham",
        "name": "Raleigh-Durham-Chapel Hill",
        "region": "south",
        "priority": 11,
        "locations": {
            "linkedin": "Raleigh-Durham-Chapel Hill Area",
            "indeed": "Raleigh-Durham, NC",
        },
    },
    {
        "metro_id": "san_diego",
        "name": "San Diego Metro",
        "region": "west",
        "priority": 12,
        "locations": {
            "linkedin": "San Diego Metropolitan Area",
            "indeed": "San Diego, CA",
        },
    },
    {
        "metro_id": "philadelphia",
        "name": "Philadelphia Metro",
        "region": "mid_atlantic",
        "priority": 13,
        "locations": {
            "linkedin": "Greater Philadelphia",
            "indeed": "Philadelphia, PA",
        },
    },
    {
        "metro_id": "denver",
        "name": "Denver Metro",
        "region": "west",
        "priority": 14,
        "locations": {
            "linkedin": "Denver, CO",
            "indeed": "Denver, CO",
        },
    },
    {
        "metro_id": "houston",
        "name": "Houston Metro",
        "region": "south",
        "priority": 15,
        "locations": {
            "linkedin": "Houston Metropolitan Area",
            "indeed": "Houston, TX",
        },
    },
    {
        "metro_id": "minneapolis_st_paul",
        "name": "Minneapolis-Saint Paul",
        "region": "midwest",
        "priority": 16,
        "locations": {
            "linkedin": "Minneapolis-Saint Paul, MN",
            "indeed": "Minneapolis, MN",
        },
    },
    {
        "metro_id": "phoenix",
        "name": "Phoenix Metro",
        "region": "west",
        "priority": 17,
        "locations": {
            "linkedin": "Phoenix Metropolitan Area, AZ",
            "indeed": "Phoenix, AZ",
        },
    },
    {
        "metro_id": "miami",
        "name": "Miami Metro",
        "region": "south",
        "priority": 18,
        "locations": {
            "linkedin": "Miami-Fort Lauderdale Area",
            "indeed": "Miami, FL",
        },
    },
    {
        "metro_id": "portland",
        "name": "Portland Metro",
        "region": "west",
        "priority": 19,
        "locations": {
            "linkedin": "Portland, Oregon Metropolitan Area",
            "indeed": "Portland, OR",
        },
    },
    {
        "metro_id": "detroit",
        "name": "Detroit Metro",
        "region": "midwest",
        "priority": 20,
        "locations": {
            "linkedin": "Detroit Metropolitan Area",
            "indeed": "Detroit, MI",
        },
    },
    {
        "metro_id": "charlotte",
        "name": "Charlotte Metro",
        "region": "south",
        "priority": 21,
        "locations": {
            "linkedin": "Charlotte Metro",
            "indeed": "Charlotte, NC",
        },
    },
    {
        "metro_id": "pittsburgh",
        "name": "Pittsburgh Metro",
        "region": "northeast",
        "priority": 22,
        "locations": {
            "linkedin": "Greater Pittsburgh Region",
            "indeed": "Pittsburgh, PA",
        },
    },
    {
        "metro_id": "salt_lake_city",
        "name": "Salt Lake City Metro",
        "region": "west",
        "priority": 23,
        "locations": {
            "linkedin": "Salt Lake City Metropolitan Area",
            "indeed": "Salt Lake City, UT",
        },
    },
    {
        "metro_id": "columbus",
        "name": "Columbus Metro",
        "region": "midwest",
        "priority": 24,
        "locations": {
            "linkedin": "Columbus, Ohio Metropolitan Area",
            "indeed": "Columbus, OH",
        },
    },
    {
        "metro_id": "nashville",
        "name": "Nashville Metro",
        "region": "south",
        "priority": 25,
        "locations": {
            "linkedin": "Nashville Metropolitan Area",
            "indeed": "Nashville, TN",
        },
    },
    {
        "metro_id": "tampa",
        "name": "Tampa Bay Metro",
        "region": "south",
        "priority": 26,
        "locations": {
            "linkedin": "Tampa Bay Area",
            "indeed": "Tampa, FL",
        },
    },
]

QUICK_METRO_COUNT = 10

OUTPUT_COLUMN_ORDER = [
    "id",
    "site",
    "job_url",
    "job_url_direct",
    "title",
    "company",
    "location",
    "date_posted",
    "job_type",
    "salary_source",
    "interval",
    "min_amount",
    "max_amount",
    "currency",
    "is_remote",
    "job_level",
    "job_function",
    "listing_type",
    "emails",
    "description",
    "company_industry",
    "company_url",
    "company_logo",
    "company_url_direct",
    "company_addresses",
    "company_num_employees",
    "company_revenue",
    "company_description",
    "skills",
    "experience_range",
    "company_rating",
    "company_reviews_count",
    "vacancy_count",
    "work_from_home_type",
    "search_query",
    "query_tier",
    "search_metro_id",
    "search_metro_name",
    "search_metro_region",
    "search_location",
    "scrape_date",
]

SITE_CONFIG = {
    "linkedin": {
        "max_consecutive_failures": 5,
        "failure_backoff": 60,
        "rate_per_second": 0.1,  # 6 req/min
        "extra_kwargs": {
            "linkedin_fetch_description": True,
        },
    },
    "indeed": {
        "max_consecutive_failures": 5,
        "failure_backoff": 45,
        "rate_per_second": 8 / 60,  # 8 req/min
        "extra_kwargs": {},
    },
}

ALL_SITES = list(SITE_CONFIG.keys())
DEFAULT_REQUEST_TIMEOUT_SEC = 180
DEFAULT_MEMORY_SOFT_LIMIT_RATIO = 0.50
DEFAULT_MEMORY_HARD_LIMIT_RATIO = 0.65

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScrapeOutcome:
    df: pd.DataFrame
    had_exception: bool
    error_message: str
    duration_sec: float
    is_rate_limited: bool


@dataclass(frozen=True)
class ResourceLimits:
    request_timeout_sec: int
    memory_soft_limit_bytes: int
    memory_hard_limit_bytes: int


# ---------------------------------------------------------------------------
# Thread-safe utility classes
# ---------------------------------------------------------------------------

class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, rate: float, burst: int = 1):
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            time.sleep(0.5)


class ThreadSafeCircuitBreaker:
    """Shared per-site circuit breaker. Only request failures trip it."""

    def __init__(self, threshold: int = 5):
        self._threshold = threshold
        self._consecutive_failures = 0
        self._tripped = False
        self._lock = threading.Lock()

    @property
    def is_tripped(self) -> bool:
        with self._lock:
            return self._tripped

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._consecutive_failures

    def record_success(self):
        with self._lock:
            self._consecutive_failures = 0

    def record_failure(self) -> bool:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._threshold:
                self._tripped = True
                return True
            return False


class ThreadSafeDedup:
    """Thread-safe dedup index for cross-query and cross-site jobs."""

    def __init__(self, initial: set[str]):
        self._seen = set(initial)
        self._lock = threading.Lock()

    def filter_new(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["_hash"] = df.apply(make_job_hash, axis=1)
        with self._lock:
            new_mask = ~df["_hash"].isin(self._seen)
            df_new = df[new_mask].copy()
            self._seen.update(df_new["_hash"].tolist())
        return df_new

    @property
    def seen_set(self) -> set[str]:
        with self._lock:
            return set(self._seen)


class ThreadSafeResultCollector:
    """Buffers deduped job batches and flushes them to partial CSVs on disk."""

    FLUSH_THRESHOLD_BYTES = 64 * 1024 * 1024  # 64 MB

    def __init__(self, site: str, temp_dir: Path):
        self._site = site
        self._temp_dir = temp_dir
        self._frames: list[pd.DataFrame] = []
        self._partial_files: list[Path] = []
        self._accumulated_bytes = 0
        self._part_index = 0
        self._lock = threading.Lock()

    def add(self, df: pd.DataFrame):
        if df.empty:
            return
        nbytes = int(df.memory_usage(deep=True).sum())
        with self._lock:
            self._frames.append(df)
            self._accumulated_bytes += nbytes
            if self._accumulated_bytes >= self.FLUSH_THRESHOLD_BYTES:
                self._flush_locked()

    def finalize(self):
        with self._lock:
            self._flush_locked()

    def iter_frames(self, chunksize: int = 2000):
        self.finalize()
        with self._lock:
            partial_files = list(self._partial_files)
            self._partial_files.clear()

        for path in partial_files:
            try:
                for chunk in pd.read_csv(path, low_memory=False, chunksize=chunksize):
                    yield chunk
            finally:
                path.unlink(missing_ok=True)

    def _flush_locked(self):
        if not self._frames:
            return

        frames = [frame for frame in self._frames if not frame.empty]
        if not frames:
            self._frames.clear()
            self._accumulated_bytes = 0
            return
        combined = pd.concat(frames, ignore_index=True)
        final_path = self._temp_dir / f"{self._site}_partial_{self._part_index:05d}.csv"
        temp_path = final_path.with_suffix(".tmp")
        combined.to_csv(temp_path, index=False)
        os.replace(temp_path, final_path)

        self._partial_files.append(final_path)
        self._frames.clear()
        self._accumulated_bytes = 0
        self._part_index += 1


class ThreadSafeTaskAudit:
    """One manifest row per (site, query, metro)."""

    def __init__(self, sites: list[str], queries: list[str], metros: list[dict]):
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str, str], dict] = {}

        for site in sites:
            for query in queries:
                for metro in metros:
                    key = (site, query, metro["metro_id"])
                    self._records[key] = {
                        "site": site,
                        "query": query,
                        "query_tier": QUERY_TO_TIER.get(query, "other"),
                        "metro_id": metro["metro_id"],
                        "metro_name": metro["name"],
                        "metro_priority": metro["priority"],
                        "metro_region": metro["region"],
                        "location": metro["locations"][site],
                        "status": "pending",
                        "raw_count": 0,
                        "new_count": 0,
                        "duration_sec": 0.0,
                        "had_exception": False,
                        "is_rate_limited": False,
                        "error_message": "",
                        "completed_at": "",
                    }

    def record_result(
        self,
        site: str,
        query: str,
        metro: dict,
        *,
        status: str,
        raw_count: int,
        new_count: int,
        duration_sec: float,
        had_exception: bool,
        is_rate_limited: bool = False,
        error_message: str = "",
    ):
        key = (site, query, metro["metro_id"])
        with self._lock:
            record = self._records[key]
            record.update(
                {
                    "status": status,
                    "raw_count": raw_count,
                    "new_count": new_count,
                    "duration_sec": round(duration_sec, 2),
                    "had_exception": had_exception,
                    "is_rate_limited": is_rate_limited,
                    "error_message": error_message[:500],
                    "completed_at": datetime.now().isoformat(),
                }
            )

    def finalize_site(self, site: str, circuit_breaker_tripped: bool):
        with self._lock:
            for record in self._records.values():
                if record["site"] != site or record["status"] != "pending":
                    continue
                record["status"] = (
                    "skipped_circuit_breaker" if circuit_breaker_tripped else "not_attempted"
                )
                record["completed_at"] = datetime.now().isoformat()

    def manifest_rows(self) -> list[dict]:
        with self._lock:
            rows = [dict(record) for record in self._records.values()]
        rows.sort(
            key=lambda row: (
                row["site"],
                row["metro_priority"],
                row["query_tier"],
                row["query"],
            )
        )
        return rows

    def site_summary(self) -> dict:
        with self._lock:
            rows = [dict(record) for record in self._records.values()]

        summary: dict[str, dict] = {}
        for row in rows:
            site = row["site"]
            site_summary = summary.setdefault(
                site,
                {
                    "tasks_total": 0,
                    "tasks_attempted": 0,
                    "tasks_ok": 0,
                    "tasks_empty": 0,
                    "tasks_exception": 0,
                    "tasks_rate_limited": 0,
                    "tasks_skipped": 0,
                    "tasks_not_attempted": 0,
                    "raw_count": 0,
                    "new_count": 0,
                    "duration_sec": 0.0,
                    "circuit_breaker_tripped": False,
                },
            )

            site_summary["tasks_total"] += 1
            site_summary["raw_count"] += row["raw_count"]
            site_summary["new_count"] += row["new_count"]
            site_summary["duration_sec"] += row["duration_sec"]

            if row["status"] == "ok":
                site_summary["tasks_attempted"] += 1
                site_summary["tasks_ok"] += 1
            elif row["status"] == "empty":
                site_summary["tasks_attempted"] += 1
                site_summary["tasks_empty"] += 1
            elif row["status"] == "exception":
                site_summary["tasks_attempted"] += 1
                site_summary["tasks_exception"] += 1
                if row.get("is_rate_limited"):
                    site_summary["tasks_rate_limited"] += 1
            elif row["status"] == "skipped_circuit_breaker":
                site_summary["tasks_skipped"] += 1
                site_summary["circuit_breaker_tripped"] = True
            elif row["status"] == "not_attempted":
                site_summary["tasks_not_attempted"] += 1

        for site_summary in summary.values():
            site_summary["duration_sec"] = round(site_summary["duration_sec"], 2)
        return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def load_seen_ids() -> set[str]:
    if DEDUP_FILE.exists():
        with open(DEDUP_FILE) as handle:
            data = json.load(handle)
        return set(data.get("ids", []))
    return set()


def save_seen_ids(seen: set[str]):
    DEDUP_FILE.parent.mkdir(parents=True, exist_ok=True)
    ids = list(seen)
    if len(ids) > 500_000:
        ids = ids[-500_000:]
    with open(DEDUP_FILE, "w") as handle:
        json.dump({"ids": ids, "updated": datetime.now().isoformat()}, handle)


def make_job_hash(row) -> str:
    job_id = row.get("id")
    if pd.notna(job_id):
        job_id = str(job_id).strip()
        if job_id:
            return job_id
    job_url = canonicalize_job_url(row.get("job_url", ""))
    if job_url:
        return job_url
    return make_content_fingerprint(row)


def make_output_dedup_keys(df: pd.DataFrame) -> pd.Series:
    if "id" in df.columns:
        keys = df["id"].fillna("").astype(str)
    else:
        keys = pd.Series([""] * len(df), index=df.index, dtype="object")

    missing = keys.eq("")
    if "job_url" in df.columns:
        canonical_urls = df["job_url"].fillna("").astype(str).map(canonicalize_job_url)
        keys.loc[missing] = canonical_urls.loc[missing]
        missing = keys.eq("")

    if "id" in df.columns:
        native_ids = df["id"].fillna("").astype(str)
        usable_ids = native_ids.ne("")
        keys.loc[usable_ids] = native_ids.loc[usable_ids]
        missing = keys.eq("")

    if missing.any():
        fallback = df.loc[missing].apply(make_content_fingerprint, axis=1)
        keys.loc[missing] = fallback
    return keys


def is_swe_role(title: str) -> bool:
    if not isinstance(title, str):
        return False
    return bool(SWE_PATTERN.search(title))


def canonicalize_job_url(raw_url) -> str:
    if not isinstance(raw_url, str):
        return ""
    raw_url = raw_url.strip()
    if not raw_url:
        return ""
    try:
        parts = urlsplit(raw_url)
    except ValueError:
        return raw_url
    query = ""
    host = parts.netloc.lower()
    if "indeed.com" in host:
        stable_params = [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=False)
            if key in {"jk", "vjk"}
        ]
        if stable_params:
            query = urlencode(sorted(stable_params))
    return urlunsplit((parts.scheme.lower(), host, parts.path, query, ""))


def normalize_text_fragment(value, *, limit: int | None = None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    if limit is not None:
        text = text[:limit]
    return text


def make_content_fingerprint(row) -> str:
    parts = [
        normalize_text_fragment(row.get("site")),
        normalize_text_fragment(row.get("title")),
        normalize_text_fragment(row.get("company") or row.get("company_name")),
        normalize_text_fragment(row.get("location")),
        normalize_text_fragment(row.get("date_posted")),
        normalize_text_fragment(row.get("min_amount") or row.get("min_salary")),
        normalize_text_fragment(row.get("max_amount") or row.get("max_salary")),
        normalize_text_fragment(row.get("description"), limit=500),
    ]
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def _read_memtotal_bytes() -> int:
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def get_process_rss_bytes() -> int:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def format_bytes(nbytes: int) -> str:
    if nbytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(nbytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024


def resolve_resource_limits(args) -> ResourceLimits:
    total_memory_bytes = _read_memtotal_bytes()

    if args.memory_soft_limit_mb is not None:
        soft_limit = args.memory_soft_limit_mb * 1024 * 1024
    else:
        soft_limit = int(total_memory_bytes * DEFAULT_MEMORY_SOFT_LIMIT_RATIO) if total_memory_bytes else 0

    if args.memory_hard_limit_mb is not None:
        hard_limit = args.memory_hard_limit_mb * 1024 * 1024
    else:
        hard_limit = int(total_memory_bytes * DEFAULT_MEMORY_HARD_LIMIT_RATIO) if total_memory_bytes else 0

    if soft_limit and hard_limit and hard_limit <= soft_limit:
        raise ValueError("memory hard limit must be greater than memory soft limit")

    return ResourceLimits(
        request_timeout_sec=args.request_timeout_sec,
        memory_soft_limit_bytes=soft_limit,
        memory_hard_limit_bytes=hard_limit,
    )


def enforce_memory_budget(
    collector: "ThreadSafeResultCollector",
    limits: ResourceLimits,
    logger,
    site: str,
    reason: str,
):
    rss_before = get_process_rss_bytes()
    soft_limit = limits.memory_soft_limit_bytes
    hard_limit = limits.memory_hard_limit_bytes

    if soft_limit and rss_before >= soft_limit:
        logger.warning(
            f"[{site}] RSS {format_bytes(rss_before)} exceeded soft limit "
            f"{format_bytes(soft_limit)} during {reason}; flushing collector and running GC"
        )
        collector.finalize()
        gc.collect()

    rss_after = get_process_rss_bytes()
    if hard_limit and rss_after >= hard_limit:
        raise MemoryError(
            f"[{site}] RSS {format_bytes(rss_after)} exceeded hard limit "
            f"{format_bytes(hard_limit)} during {reason}"
        )


def is_rate_limit_error(message: str) -> bool:
    message = (message or "").lower()
    patterns = [
        "429",
        "too many requests",
        "rate limit",
        "rate-limited",
        "throttl",
        "temporarily blocked",
        "captcha",
        "access denied",
    ]
    return any(pattern in message for pattern in patterns)


def get_selected_metros(args) -> list[dict]:
    metros = sorted(METRO_AREAS, key=lambda metro: metro["priority"])
    if args.test:
        return metros[:1]
    if args.quick:
        return metros[:QUICK_METRO_COUNT]
    return metros


def serialize_metros(metros: list[dict], sites: list[str]) -> list[dict]:
    serialized = []
    for metro in metros:
        serialized.append(
            {
                "metro_id": metro["metro_id"],
                "name": metro["name"],
                "region": metro["region"],
                "priority": metro["priority"],
                "locations": {
                    site: metro["locations"][site]
                    for site in sites
                    if site in metro["locations"]
                },
            }
        )
    return serialized


def append_batch_metadata(df: pd.DataFrame, site: str, query: str, metro: dict) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["site"] = site
    df["search_query"] = query
    df["query_tier"] = QUERY_TO_TIER.get(query, "other")
    df["search_metro_id"] = metro["metro_id"]
    df["search_metro_name"] = metro["name"]
    df["search_metro_region"] = metro["region"]
    df["search_location"] = metro["locations"][site]
    return df


def determine_queries(args, logger) -> tuple[list[str], int]:
    if args.test:
        logger.info("TEST MODE: 1 query, 1 metro, 5 results")
        return QUERY_TIERS["swe"][:1], 5
    if args.quick:
        logger.info(f"QUICK MODE: 4 queries, {QUICK_METRO_COUNT} metros")
        return QUERY_TIERS["swe"][:4], args.results

    queries: list[str] = []
    for tier in args.tiers:
        queries.extend(QUERY_TIERS[tier])
    logger.info(f"Tiers: {', '.join(args.tiers)} ({len(queries)} queries)")
    return queries, args.results


def iter_existing_daily_frames(today: str):
    existing_paths = [
        DATA_DIR / f"{today}_swe_jobs.csv",
        DATA_DIR / f"{today}_non_swe_jobs.csv",
    ]
    for path in existing_paths:
        if not path.exists():
            continue
        for chunk in pd.read_csv(path, low_memory=False, chunksize=2000):
            yield path, chunk


def append_dataframe_csv(path: Path, df: pd.DataFrame, header_written: bool) -> bool:
    if df.empty:
        return header_written
    df = normalize_output_frame_schema(df)
    df.to_csv(path, mode="a" if header_written else "w", header=not header_written, index=False)
    return True


def normalize_output_frame_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.copy()
    frame.rename(
        columns={
            "metro_id": "search_metro_id",
            "metro_name": "search_metro_name",
            "metro_region": "search_metro_region",
        },
        inplace=True,
    )
    extras = [col for col in frame.columns if col not in OUTPUT_COLUMN_ORDER]
    return frame.reindex(columns=OUTPUT_COLUMN_ORDER + extras)


def finalize_daily_outputs(
    collectors: dict[str, ThreadSafeResultCollector],
    today: str,
    logger,
) -> dict:
    swe_temp = DATA_DIR / f".{today}_swe_jobs.tmp.csv"
    non_swe_temp = DATA_DIR / f".{today}_non_swe_jobs.tmp.csv"
    swe_file = DATA_DIR / f"{today}_swe_jobs.csv"
    non_swe_file = DATA_DIR / f"{today}_non_swe_jobs.csv"

    for path in (swe_temp, non_swe_temp):
        path.unlink(missing_ok=True)

    seen_output_keys: set[str] = set()
    swe_header_written = False
    non_swe_header_written = False
    site_counts: Counter = Counter()
    swe_site_counts: Counter = Counter()
    seniority_counts: Counter = Counter()
    location_counts: Counter = Counter()
    company_counts: Counter = Counter()

    total_unique_jobs = 0
    swe_count = 0
    non_swe_count = 0
    cross_output_duplicates = 0
    preexisting_rows_merged = 0

    def ingest_frame(frame: pd.DataFrame, *, default_site: str | None = None):
        nonlocal swe_header_written, non_swe_header_written
        nonlocal total_unique_jobs, swe_count, non_swe_count
        nonlocal cross_output_duplicates, preexisting_rows_merged

        if frame.empty:
            return

        dedup_keys = make_output_dedup_keys(frame)
        keep_mask = ~dedup_keys.isin(seen_output_keys)
        cross_output_duplicates += int((~keep_mask).sum())

        if not keep_mask.any():
            return

        kept_keys = dedup_keys[keep_mask]
        seen_output_keys.update(kept_keys.tolist())

        frame = frame[keep_mask].copy()
        frame.drop(columns=["_hash"], inplace=True, errors="ignore")
        frame["scrape_date"] = today

        total_unique_jobs += len(frame)
        if "site" in frame.columns:
            site_counts.update(frame["site"].fillna("unknown").astype(str).tolist())
        elif default_site:
            site_counts.update([default_site] * len(frame))

        swe_mask = frame["title"].apply(is_swe_role)
        swe_batch = frame[swe_mask].copy()
        non_swe_batch = frame[~swe_mask].copy()

        if not swe_batch.empty:
            swe_header_written = append_dataframe_csv(swe_temp, swe_batch, swe_header_written)
            swe_count += len(swe_batch)
            if "site" in swe_batch.columns:
                swe_site_counts.update(swe_batch["site"].fillna("unknown").astype(str).tolist())
            seniority_col = "job_level" if "job_level" in swe_batch.columns else "seniority"
            if seniority_col in swe_batch.columns:
                seniority_counts.update(swe_batch[seniority_col].fillna("unknown").astype(str).tolist())
            if "location" in swe_batch.columns:
                location_counts.update(swe_batch["location"].fillna("unknown").astype(str).tolist())
            company_col = "company" if "company" in swe_batch.columns else "company_name"
            if company_col in swe_batch.columns:
                company_counts.update(swe_batch[company_col].fillna("unknown").astype(str).tolist())

        if not non_swe_batch.empty:
            non_swe_header_written = append_dataframe_csv(
                non_swe_temp, non_swe_batch, non_swe_header_written
            )
            non_swe_count += len(non_swe_batch)

    for path, frame in iter_existing_daily_frames(today):
        preexisting_rows_merged += len(frame)
        ingest_frame(frame)

    for site in collectors:
        for frame in collectors[site].iter_frames():
            ingest_frame(frame, default_site=site)

    if swe_header_written:
        os.replace(swe_temp, swe_file)
        logger.info(f"Saved {swe_count} SWE jobs to {swe_file}")
    else:
        swe_temp.unlink(missing_ok=True)

    if non_swe_header_written:
        os.replace(non_swe_temp, non_swe_file)
        logger.info(f"Saved {non_swe_count} non-SWE jobs to {non_swe_file}")
    else:
        non_swe_temp.unlink(missing_ok=True)

    return {
        "swe_file": swe_file if swe_header_written else None,
        "non_swe_file": non_swe_file if non_swe_header_written else None,
        "swe_count": swe_count,
        "non_swe_count": non_swe_count,
        "total_unique_jobs": total_unique_jobs,
        "cross_output_duplicates": cross_output_duplicates,
        "preexisting_rows_merged": preexisting_rows_merged,
        "sites_collected": dict(site_counts),
        "swe_sites": dict(swe_site_counts),
        "seniority_top": seniority_counts.most_common(10),
        "locations_top": location_counts.most_common(10),
        "companies_top": company_counts.most_common(10),
    }


def scrape_batch(
    site: str,
    search_term: str,
    metro: dict,
    results_wanted: int,
    hours_old: int,
    resource_limits: ResourceLimits,
    logger,
) -> ScrapeOutcome:
    ua = random.choice(USER_AGENTS)
    location = metro["locations"][site]
    config = SITE_CONFIG[site]
    start = time.monotonic()

    try:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            scrape_jobs,
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
        try:
            df = future.result(timeout=resource_limits.request_timeout_sec)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"request timed out after {resource_limits.request_timeout_sec}s"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if df is None:
            df = pd.DataFrame()
        return ScrapeOutcome(
            df=append_batch_metadata(df, site, search_term, metro),
            had_exception=False,
            error_message="",
            duration_sec=time.monotonic() - start,
            is_rate_limited=False,
        )
    except Exception as exc:
        error_message = str(exc)
        logger.warning(
            f"[{site}] Scrape failed for query='{search_term}' metro='{metro['metro_id']}' "
            f"location='{location}': {error_message}"
        )
        return ScrapeOutcome(
            df=pd.DataFrame(),
            had_exception=True,
            error_message=error_message,
            duration_sec=time.monotonic() - start,
            is_rate_limited=is_rate_limit_error(error_message),
        )


# ---------------------------------------------------------------------------
# Scraping workers
# ---------------------------------------------------------------------------

def site_worker(
    work_queue: queue.Queue,
    site: str,
    results_per: int,
    hours_old: int,
    rate_limiter: TokenBucketRateLimiter,
    circuit_breaker: ThreadSafeCircuitBreaker,
    dedup: ThreadSafeDedup,
    collector: ThreadSafeResultCollector,
    resource_limits: ResourceLimits,
    audit: ThreadSafeTaskAudit,
    logger,
) -> dict:
    stats = {"requests": 0, "raw": 0, "new": 0, "empty": 0, "exceptions": 0, "rate_limited": 0}

    while True:
        if circuit_breaker.is_tripped:
            break

        try:
            query, metro = work_queue.get_nowait()
        except queue.Empty:
            break

        enforce_memory_budget(collector, resource_limits, logger, site, "pre-request")

        rate_limiter.acquire()
        if circuit_breaker.is_tripped:
            work_queue.put((query, metro))
            break

        location = metro["locations"][site]
        stats["requests"] += 1
        logger.info(
            f"[{site}] query='{query}' metro='{metro['metro_id']}' "
            f"location='{location}' remaining~{work_queue.qsize()}"
        )

        outcome = scrape_batch(
            site,
            query,
            metro,
            results_per,
            hours_old,
            resource_limits,
            logger,
        )

        if outcome.had_exception:
            stats["exceptions"] += 1
            if outcome.is_rate_limited:
                stats["rate_limited"] += 1
            just_tripped = circuit_breaker.record_failure()
            audit.record_result(
                site,
                query,
                metro,
                status="exception",
                raw_count=0,
                new_count=0,
                duration_sec=outcome.duration_sec,
                had_exception=True,
                is_rate_limited=outcome.is_rate_limited,
                error_message=outcome.error_message,
            )
            if outcome.is_rate_limited:
                logger.warning(
                    f"[{site}] Suspected rate limit for query='{query}' metro='{metro['metro_id']}'"
                )
            if just_tripped:
                logger.error(
                    f"[{site}] Circuit breaker tripped after "
                    f"{circuit_breaker.failure_count} consecutive request failures."
                )
            else:
                backoff = SITE_CONFIG[site]["failure_backoff"] * circuit_breaker.failure_count
                logger.info(
                    f"[{site}] Request failure count={circuit_breaker.failure_count}, "
                    f"backing off {backoff}s"
                )
                time.sleep(backoff)
            continue

        circuit_breaker.record_success()
        raw_count = len(outcome.df)
        stats["raw"] += raw_count

        if raw_count == 0:
            stats["empty"] += 1
            audit.record_result(
                site,
                query,
                metro,
                status="empty",
                raw_count=0,
                new_count=0,
                duration_sec=outcome.duration_sec,
                had_exception=False,
                is_rate_limited=False,
            )
            logger.info(
                f"[{site}] Empty result for query='{query}' metro='{metro['metro_id']}' "
                f"in {outcome.duration_sec:.1f}s"
            )
            continue

        df_new = dedup.filter_new(outcome.df)
        new_count = len(df_new)
        dup_count = raw_count - new_count
        stats["new"] += new_count

        audit.record_result(
            site,
            query,
            metro,
            status="ok",
            raw_count=raw_count,
            new_count=new_count,
            duration_sec=outcome.duration_sec,
            had_exception=False,
            is_rate_limited=False,
        )

        logger.info(
            f"[{site}] query='{query}' metro='{metro['metro_id']}' "
            f"raw={raw_count} new={new_count} dups={dup_count} "
            f"duration={outcome.duration_sec:.1f}s"
        )

        if not df_new.empty:
            collector.add(df_new.drop(columns=["_hash"], errors="ignore"))
            enforce_memory_budget(collector, resource_limits, logger, site, "post-collect")

    return stats


def scrape_site(
    site: str,
    queries: list[str],
    metros: list[dict],
    results_per: int,
    hours_old: int,
    num_workers: int,
    dedup: ThreadSafeDedup,
    resource_limits: ResourceLimits,
    audit: ThreadSafeTaskAudit,
    temp_dir: Path,
    logger,
) -> tuple[ThreadSafeResultCollector, dict]:
    config = SITE_CONFIG[site]
    rate_limiter = TokenBucketRateLimiter(rate=config["rate_per_second"])
    circuit_breaker = ThreadSafeCircuitBreaker(threshold=config["max_consecutive_failures"])
    collector = ThreadSafeResultCollector(site, temp_dir)

    work_items = [(query, metro) for metro in metros for query in queries]
    random.shuffle(work_items)
    work_queue: queue.Queue = queue.Queue()
    for item in work_items:
        work_queue.put(item)

    start = time.monotonic()
    logger.info(
        f"--- [{site.upper()}] Starting: {len(queries)} queries x {len(metros)} metros = "
        f"{len(work_items)} tasks, workers={num_workers}, "
        f"rate={config['rate_per_second'] * 60:.1f} req/min ---"
    )

    worker_stats: list[dict] = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                site_worker,
                work_queue,
                site,
                results_per,
                hours_old,
                rate_limiter,
                circuit_breaker,
                dedup,
                collector,
                resource_limits,
                audit,
                logger,
            )
            for _ in range(num_workers)
        ]
        for future in as_completed(futures):
            worker_stats.append(future.result())

    collector.finalize()
    audit.finalize_site(site, circuit_breaker.is_tripped)

    summary = {
        "requests": sum(stats["requests"] for stats in worker_stats),
        "raw": sum(stats["raw"] for stats in worker_stats),
        "new": sum(stats["new"] for stats in worker_stats),
        "empty": sum(stats["empty"] for stats in worker_stats),
        "exceptions": sum(stats["exceptions"] for stats in worker_stats),
        "rate_limited": sum(stats["rate_limited"] for stats in worker_stats),
        "circuit_breaker_tripped": circuit_breaker.is_tripped,
        "duration_sec": round(time.monotonic() - start, 2),
        "workers": num_workers,
    }

    logger.info(
        f"--- [{site.upper()}] Summary: attempted={summary['requests']}/{len(work_items)} "
        f"raw={summary['raw']} new={summary['new']} empty={summary['empty']} "
        f"exceptions={summary['exceptions']} rate_limited={summary['rate_limited']} "
        f"circuit_breaker={summary['circuit_breaker_tripped']} "
        f"duration={summary['duration_sec']:.1f}s ---"
    )

    return collector, summary


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_scraper(args):
    logger = setup_logging()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    sites = [site for site in args.sites if site in SITE_CONFIG]
    logger.info(f"=== Starting scrape for {today} | sites: {', '.join(sites)} ===")

    if not sites:
        logger.error("No valid sites selected.")
        return

    historical_seen_ids = load_seen_ids()
    logger.info(
        f"Historical dedup index: {len(historical_seen_ids):,} previously seen job IDs "
        f"(used for ever-seen stats, not to suppress today's snapshot)"
    )

    queries, results_per = determine_queries(args, logger)
    metros = get_selected_metros(args)
    hours_old = args.hours_old
    resource_limits = resolve_resource_limits(args)
    logger.info(
        "Resource limits: "
        f"request_timeout={resource_limits.request_timeout_sec}s "
        f"rss_soft={format_bytes(resource_limits.memory_soft_limit_bytes)} "
        f"rss_hard={format_bytes(resource_limits.memory_hard_limit_bytes)}"
    )
    dedup = ThreadSafeDedup(set())
    audit = ThreadSafeTaskAudit(sites, queries, metros)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"scrape_{today}_", dir=str(DATA_DIR)))

    site_run_summaries: dict[str, dict] = {}
    collectors: dict[str, ThreadSafeResultCollector] = {}

    try:
        if args.sequential:
            for site in sites:
                collector, summary = scrape_site(
                    site,
                    queries,
                    metros,
                    results_per,
                    hours_old,
                    1,
                    dedup,
                    resource_limits,
                    audit,
                    temp_dir,
                    logger,
                )
                collectors[site] = collector
                site_run_summaries[site] = summary
        else:
            with ThreadPoolExecutor(max_workers=len(sites)) as executor:
                future_to_site = {
                    executor.submit(
                        scrape_site,
                        site,
                        queries,
                        metros,
                        results_per,
                        hours_old,
                        args.workers,
                        dedup,
                        resource_limits,
                        audit,
                        temp_dir,
                        logger,
                    ): site
                    for site in sites
                }
                for future in as_completed(future_to_site):
                    site = future_to_site[future]
                    collector, summary = future.result()
                    collectors[site] = collector
                    site_run_summaries[site] = summary

        output_stats = finalize_daily_outputs(collectors, today, logger)
        if output_stats["preexisting_rows_merged"] > 0:
            logger.info(
                f"Merged {output_stats['preexisting_rows_merged']} rows from existing "
                f"{today} CSVs before writing the refreshed daily snapshot"
            )
        seen_ids = historical_seen_ids | dedup.seen_set
        save_seen_ids(seen_ids)
        site_summary = audit.site_summary()
        total_rate_limited = sum(
            summary.get("tasks_rate_limited", 0) for summary in site_summary.values()
        )

        manifest = {
            "manifest_version": 2,
            "scrape_date": today,
            "sites": sites,
            "queries": queries,
            "tiers": args.tiers if not args.test and not args.quick else ["swe"],
            "location_type": "msa",
            "locations": {
                site: [metro["locations"][site] for metro in metros]
                for site in sites
            },
            "metros": serialize_metros(metros, sites),
            "results_per_query": results_per,
            "hours_old": hours_old,
            "mode": "test" if args.test else ("quick" if args.quick else "full"),
            "parallel": not args.sequential,
            "workers_per_site": args.workers if not args.sequential else 1,
            "request_timeout_sec": resource_limits.request_timeout_sec,
            "memory_soft_limit_mb": round(resource_limits.memory_soft_limit_bytes / (1024 * 1024), 2)
            if resource_limits.memory_soft_limit_bytes
            else None,
            "memory_hard_limit_mb": round(resource_limits.memory_hard_limit_bytes / (1024 * 1024), 2)
            if resource_limits.memory_hard_limit_bytes
            else None,
            "rate_limits": {
                site: SITE_CONFIG[site]["rate_per_second"]
                for site in sites
            },
            "swe_count": output_stats["swe_count"],
            "non_swe_count": output_stats["non_swe_count"],
            "total_unique_jobs": output_stats["total_unique_jobs"],
            "cross_output_duplicates": output_stats["cross_output_duplicates"],
            "preexisting_rows_merged": output_stats["preexisting_rows_merged"],
            "historical_output_dedup_applied": False,
            "dedup_index_size_before": len(historical_seen_ids),
            "dedup_index_size": len(seen_ids),
            "sites_collected": output_stats["sites_collected"],
            "site_run_summaries": site_run_summaries,
            "site_summary": site_summary,
            "rate_limit_summary": {
                "total": total_rate_limited,
                "by_site": {
                    site: summary.get("tasks_rate_limited", 0)
                    for site, summary in site_summary.items()
                },
            },
            "task_results": audit.manifest_rows(),
            "timestamp": datetime.now().isoformat(),
        }

        manifest_file = DATA_DIR / f"{today}_manifest.json"
        with open(manifest_file, "w") as handle:
            json.dump(manifest, handle, indent=2, default=str)
        logger.info(f"Saved run manifest to {manifest_file}")

        if output_stats["total_unique_jobs"] == 0:
            logger.warning("No jobs collected after dedup.")

        if not args.no_harmonize:
            logger.info("=== Harmonizing into unified dataset ===")
            try:
                unified_path = BASE_DIR / "data" / "unified.parquet"
                observations_path = BASE_DIR / "data" / "unified_observations.parquet"
                harmonize_scraped(
                    str(DATA_DIR),
                    output_path=unified_path,
                    observations_output_path=observations_path,
                )
                logger.info(f"Unified dataset saved to {unified_path}")
                logger.info(f"Daily observations saved to {observations_path}")
            except Exception as exc:
                logger.warning(f"Harmonization failed (non-fatal): {exc}")

        logger.info("=== Summary ===")
        logger.info(f"Total unique jobs collected: {output_stats['total_unique_jobs']}")
        logger.info(f"SWE jobs: {output_stats['swe_count']}")
        logger.info(f"Non-SWE jobs: {output_stats['non_swe_count']}")
        if output_stats["swe_sites"]:
            logger.info("By site:")
            for site_name, count in output_stats["swe_sites"].items():
                logger.info(f"  {site_name}: {count}")
        if output_stats["seniority_top"]:
            logger.info("Seniority distribution:")
            for level, count in output_stats["seniority_top"]:
                logger.info(f"  {level}: {count}")
        if output_stats["locations_top"]:
            logger.info("Top locations:")
            for location, count in output_stats["locations_top"]:
                logger.info(f"  {location}: {count}")
        if output_stats["companies_top"]:
            logger.info("Top companies:")
            for company, count in output_stats["companies_top"]:
                logger.info(f"  {company}: {count}")

        logger.info("=== Done ===")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Daily US SWE job scraper (LinkedIn + Indeed)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: 1 query, 1 metro, 5 results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 4 queries, 10 metros",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=ALL_SITES,
        choices=ALL_SITES,
        help=f"Sites to scrape (default: all). Choices: {', '.join(ALL_SITES)}",
    )
    parser.add_argument(
        "--results",
        type=int,
        default=100,
        help="Results per query-location combo (default: 100)",
    )
    parser.add_argument(
        "--hours-old",
        type=int,
        default=24,
        help="Only fetch jobs posted in the last N hours (default: 24)",
    )
    parser.add_argument(
        "--no-harmonize",
        action="store_true",
        help="Skip harmonization step after scraping",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["swe", "adjacent", "control"],
        choices=list(QUERY_TIERS.keys()),
        help="Query tiers to run (default: all).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Disable cross-site and per-site parallelism",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Worker threads per site in parallel mode (default: 3)",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SEC,
        help="Per-request timeout for jobspy calls in seconds (default: 180)",
    )
    parser.add_argument(
        "--memory-soft-limit-mb",
        type=int,
        default=None,
        help="Soft RSS limit in MB; above this the scraper flushes to disk and runs GC (default: 50%% of system RAM)",
    )
    parser.add_argument(
        "--memory-hard-limit-mb",
        type=int,
        default=None,
        help="Hard RSS limit in MB; above this the scraper aborts before the kernel OOM killer does (default: 65%% of system RAM)",
    )
    args = parser.parse_args()
    run_scraper(args)


if __name__ == "__main__":
    main()
