#!/usr/bin/env python3
"""
Send daily scraper summary via AWS SNS.

Usage (called by run_daily.sh):
    python send_alert.py --status success --swe-count 150 --total-count 200 --attempt 1
    python send_alert.py --status failure --message "Scraper failed" --attempt 3

Requires SNS_TOPIC_ARN in environment (set via .env).
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SCRAPED_DIR = DATA_DIR / "scraped"
LOG_DIR = PROJECT_DIR / "logs"

# Load .env if it exists (so SNS_TOPIC_ARN is available even without sourcing)
ENV_FILE = PROJECT_DIR / ".env"
if ENV_FILE.exists():
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


def get_today_files(today: str) -> dict:
    """Find today's scraped CSV files and return paths + row counts."""
    files = {}
    for pattern in [f"{today}_swe_jobs.csv", f"{today}_non_swe_jobs.csv", f"{today}_yc_jobs.csv"]:
        path = SCRAPED_DIR / pattern
        if path.exists():
            with open(path) as f:
                reader = csv.reader(f)
                header = next(reader, None)
                rows = sum(1 for _ in reader)
            files[pattern] = {"path": path, "rows": rows, "columns": len(header) if header else 0}
    return files


def _count_csv_lines(path: Path) -> int:
    """Count data rows in a CSV file (excludes header). Fast line counting."""
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def load_manifest(today: str) -> dict:
    """Load today's scraper manifest if present."""
    path = SCRAPED_DIR / f"{today}_manifest.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def get_previous_day_count(today: str):
    """Get yesterday's SWE job count for comparison."""
    yesterday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    path = SCRAPED_DIR / f"{yesterday}_swe_jobs.csv"
    if path.exists():
        return _count_csv_lines(path)
    return None


def get_7day_average(today: str):
    """Get 7-day average SWE job count."""
    counts = []
    for i in range(1, 8):
        day = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=i)).strftime("%Y-%m-%d")
        path = SCRAPED_DIR / f"{day}_swe_jobs.csv"
        if path.exists():
            counts.append(_count_csv_lines(path))
    return round(sum(counts) / len(counts), 1) if counts else None


def get_sample_titles(today: str, n: int = 5):
    """Get random sample of job titles from today's SWE file via reservoir sampling."""
    path = SCRAPED_DIR / f"{today}_swe_jobs.csv"
    if not path.exists():
        return []
    reservoir = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i < n:
                reservoir.append(r)
            else:
                j = random.randint(0, i)
                if j < n:
                    reservoir[j] = r
    titles = []
    for r in reservoir:
        title = r.get("title", r.get("job_title", "Unknown"))
        company = r.get("company", r.get("company_name", ""))
        loc = r.get("location", "")
        line = f"  {title}"
        if company:
            line += f" @ {company}"
        if loc:
            line += f" ({loc})"
        titles.append(line)
    return titles


def get_data_quality(today: str) -> dict:
    """Check null rates for key fields in today's SWE file. Streams rows."""
    path = SCRAPED_DIR / f"{today}_swe_jobs.csv"
    if not path.exists():
        return {}

    key_fields = ["seniority", "location", "company", "company_name", "salary", "skills_raw"]
    total = 0
    filled_counts = {}

    with open(path) as f:
        reader = csv.DictReader(f)
        # Filter to fields that actually exist in the header
        active_fields = [fld for fld in key_fields if fld in (reader.fieldnames or [])]
        if not active_fields:
            return {}
        for fld in active_fields:
            filled_counts[fld] = 0
        for row in reader:
            total += 1
            for fld in active_fields:
                if row.get(fld, "").strip():
                    filled_counts[fld] += 1

    if total == 0:
        return {}
    return {fld: f"{round(cnt / total * 100)}% filled" for fld, cnt in filled_counts.items()}


def get_cumulative_stats() -> dict:
    """Stats on total accumulated data."""
    stats = {}
    swe_files = sorted(SCRAPED_DIR.glob("*_swe_jobs.csv"))
    if swe_files:
        stats["days_scraped"] = len(swe_files)
        stats["first_date"] = swe_files[0].name[:10]
        stats["last_date"] = swe_files[-1].name[:10]
        total_rows = 0
        for f in swe_files:
            total_rows += _count_csv_lines(f)
        stats["total_swe_rows"] = total_rows

    unified = DATA_DIR / "unified.parquet"
    if unified.exists():
        size_mb = unified.stat().st_size / (1024 * 1024)
        stats["unified_size"] = f"{size_mb:.1f} MB"
        try:
            import pyarrow.parquet as pq
            meta = pq.read_metadata(unified)
            stats["unified_rows"] = meta.num_rows
        except Exception:
            pass

    return stats


def _read_log_lines(path: Path) -> list[str]:
    """Read a log file, returning lines or empty list if missing."""
    if path.exists():
        with open(path) as f:
            return f.readlines()
    return []


def _read_cron_log_for_date(today: str) -> list[str]:
    """Read only today's portion of cron.log by scanning backwards from EOF."""
    path = LOG_DIR / "cron.log"
    if not path.exists():
        return []

    # Read backwards in chunks to find where today's entries start.
    # Avoids loading the full (and ever-growing) file.
    file_size = path.stat().st_size
    if file_size == 0:
        return []

    chunk_size = 512 * 1024  # 512KB chunks — a day's cron output is typically <100KB
    today_lines = []
    found_today = False
    found_earlier_date = False

    with open(path, "rb") as f:
        offset = max(0, file_size - chunk_size)
        while True:
            f.seek(offset)
            raw = f.read(min(chunk_size, file_size - offset))
            text = raw.decode("utf-8", errors="replace")
            lines = text.splitlines(keepends=True)

            # If we didn't read from the start, the first line may be partial
            if offset > 0:
                lines = lines[1:]

            # Process lines in reverse to find the boundary
            for line in reversed(lines):
                dm = re.match(r"(\d{4}-\d{2}-\d{2})[T ]", line)
                if dm:
                    line_date = dm.group(1)
                    if line_date == today:
                        found_today = True
                    elif found_today:
                        # We've gone past today into an earlier date
                        found_earlier_date = True
                        break
                # Keep lines that are either today-dated or undated (tracebacks)
                # while we're in the today zone
                if found_today and not found_earlier_date:
                    today_lines.append(line)
                elif not found_today:
                    # Still scanning backwards from EOF, hasn't hit today yet
                    # (could be undated lines at the very end)
                    today_lines.append(line)

            if found_earlier_date or offset == 0:
                break
            offset = max(0, offset - chunk_size)

    today_lines.reverse()
    return today_lines


def get_traceback(today: str, scrape_lines: list[str]) -> list[str]:
    """Extract tracebacks and error/warning lines from today's logs."""
    sections = []

    cron_lines = _read_cron_log_for_date(today)

    for log_name, lines in [
        (f"scrape_{today}.log", scrape_lines),
        ("cron.log", cron_lines),
    ]:
        if not lines:
            continue

        # Extract traceback blocks
        tracebacks = []
        current_tb = []
        in_tb = False
        for line in lines:
            if "Traceback (most recent call last):" in line:
                in_tb = True
                current_tb = [line.rstrip()]
            elif in_tb:
                current_tb.append(line.rstrip())
                # A traceback ends with a line that starts with a non-space
                # character (the exception line) after at least one indented line
                if len(current_tb) > 2 and line[0:1] not in (" ", "\t", ""):
                    tracebacks.append(current_tb)
                    current_tb = []
                    in_tb = False
        if current_tb:
            tracebacks.append(current_tb)

        # Keep last 3 tracebacks, truncated to 20 lines each
        for tb in tracebacks[-3:]:
            truncated = tb[:20]
            if len(tb) > 20:
                truncated.append(f"  ... ({len(tb) - 20} more lines)")
            sections.append(f"  [{log_name}]")
            sections.extend(f"  {l}" for l in truncated)
            sections.append("")

    # Collect [ERROR] and [WARNING] lines from scrape log (already in memory)
    error_lines = []
    for line in scrape_lines:
        if "[ERROR]" in line or "[WARNING]" in line:
            error_lines.append(line.strip()[:150])
    if error_lines:
        sections.append("  Log errors/warnings:")
        for e in error_lines[-10:]:
            sections.append(f"  {e}")
        if len(error_lines) > 10:
            sections.append(f"  ... ({len(error_lines) - 10} more)")

    return sections


def get_site_status(
    today: str, scrape_lines: list[str], manifest: Optional[dict] = None
) -> list[str]:
    """Get per-site status from the manifest when available, else fall back to logs."""
    if manifest and manifest.get("site_summary"):
        results = []
        for site, info in sorted(manifest["site_summary"].items()):
            attempted = info.get("tasks_attempted", 0)
            total = info.get("tasks_total", 0)
            empty = info.get("tasks_empty", 0)
            exceptions = info.get("tasks_exception", 0)
            rate_limited = info.get("tasks_rate_limited", 0)
            skipped = info.get("tasks_skipped", 0) + info.get("tasks_not_attempted", 0)
            new_count = info.get("new_count", 0)

            if attempted == 0 and total > 0:
                status = "FAILED"
            elif info.get("circuit_breaker_tripped") or exceptions > 0 or skipped > 0:
                status = "PARTIAL"
            else:
                status = "OK"

            results.append(
                f"  {site.upper()}: {status}"
                f" | attempted {attempted}/{total}"
                f" | new {new_count:,}"
                f" | empty {empty}"
                f" | exceptions {exceptions}"
                f" | rate-limited {rate_limited}"
                f" | skipped {skipped}"
            )
        return results

    yc_lines = _read_log_lines(LOG_DIR / f"yc_{today}.log")
    sites = {}

    for line in scrape_lines:
        m = re.search(r"--- \[(\w+)\] Starting:", line)
        if m:
            site = m.group(1).upper()
            sites[site] = {"started": True, "result": None, "circuit_breaker": False}

        m = re.search(
            r"--- \[(\w+)\] Summary: attempted=(\d+)/(\d+) raw=(\d+) new=(\d+) "
            r"empty=(\d+) exceptions=(\d+) circuit_breaker=(True|False)",
            line,
        )
        if m:
            site = m.group(1).upper()
            attempted = int(m.group(2))
            total = int(m.group(3))
            new_count = int(m.group(5))
            empty = int(m.group(6))
            exceptions = int(m.group(7))
            circuit_breaker = m.group(8) == "True"
            status = "OK"
            if attempted == 0 and total > 0:
                status = "FAILED"
            elif circuit_breaker or exceptions > 0:
                status = "PARTIAL"
            if site in sites:
                sites[site]["result"] = (
                    f"{status} ({new_count:,} jobs, attempted {attempted}/{total}, "
                    f"empty {empty}, exceptions {exceptions})"
                )
                sites[site]["circuit_breaker"] = circuit_breaker

    for line in yc_lines:
        if "YC scraper starting" in line:
            sites["YC"] = {"started": True, "result": None, "circuit_breaker": False}
        m = re.search(r"Done — (\d+) YC jobs collected", line)
        if m:
            count = int(m.group(1))
            sites.setdefault("YC", {"started": True, "result": None, "circuit_breaker": False})
            sites["YC"]["result"] = f"OK ({count:,} jobs)" if count > 0 else "OK (0 jobs)"
        if "No jobs collected" in line:
            sites.setdefault("YC", {"started": True, "result": None, "circuit_breaker": False})
            sites["YC"]["result"] = "WARNING (0 jobs)"

    results = []
    for site, info in sites.items():
        status = info["result"] or "FAILED (no completion)"
        if info["circuit_breaker"] and info["result"]:
            status = status.replace("OK", "PARTIAL")
        results.append(f"  {site}: {status}")

    return results


def get_scrape_stats(scrape_lines: list[str], manifest: Optional[dict] = None) -> dict:
    """Get scrape progress from the manifest when available, else fall back to logs."""
    if manifest and manifest.get("task_results"):
        task_results = manifest["task_results"]
        attempted = sum(
            1 for row in task_results if row.get("status") in {"ok", "empty", "exception"}
        )
        total = len(task_results)
        raw_count = sum(int(row.get("raw_count", 0)) for row in task_results)
        new_count = sum(int(row.get("new_count", 0)) for row in task_results)
        dupes = max(raw_count - new_count, 0)
        errors = sum(1 for row in task_results if row.get("had_exception"))
        empty = sum(1 for row in task_results if row.get("status") == "empty")
        skipped = sum(
            1
            for row in task_results
            if row.get("status") in {"skipped_circuit_breaker", "not_attempted"}
        )
        rate_limited = sum(1 for row in task_results if row.get("is_rate_limited"))
        dedup_rate = round(dupes / raw_count * 100) if raw_count > 0 else 0
        return {
            "requests_done": attempted,
            "requests_total": total,
            "raw": raw_count,
            "new": new_count,
            "dupes": dupes,
            "dedup_rate": dedup_rate,
            "errors": errors,
            "empty": empty,
            "skipped": skipped,
            "rate_limited": rate_limited,
        }

    if not scrape_lines:
        return {}

    max_current = 0
    max_total = 0
    total_new = 0
    total_dupes = 0
    failed_requests = 0

    for line in scrape_lines:
        m = re.search(r"\[\w+:(\d+)/(\d+)\]", line)
        if m:
            current = int(m.group(1))
            total = int(m.group(2))
            if current > max_current:
                max_current = current
            if total > max_total:
                max_total = total

        m = re.search(r"After dedup: (\d+) new \((\d+) duplicates\)", line)
        if m:
            total_new += int(m.group(1))
            total_dupes += int(m.group(2))

        if "Scrape failed for" in line:
            failed_requests += 1

    if max_total == 0:
        return {}

    total_seen = total_new + total_dupes
    dedup_rate = round(total_dupes / total_seen * 100) if total_seen > 0 else 0

    return {
        "requests_done": max_current,
        "requests_total": max_total,
        "raw": total_seen,
        "new": total_new,
        "dupes": total_dupes,
        "dedup_rate": dedup_rate,
        "errors": failed_requests,
        "empty": 0,
        "skipped": 0,
        "rate_limited": 0,
    }


def get_anomaly_flags(
    today: str, swe_count: int, status: str, scrape_stats: dict,
    avg_7d: Optional[float], scrape_lines: list[str], manifest: Optional[dict] = None,
) -> list[str]:
    """Check for anomalous conditions worth flagging."""
    flags = []

    if avg_7d and avg_7d > 0 and swe_count < avg_7d * 0.5:
        flags.append(
            f"COUNT DROP: {swe_count:,} SWE jobs is <50% of 7-day avg ({avg_7d:,.0f})"
        )

    dedup_rate = scrape_stats.get("dedup_rate", 0)
    if dedup_rate > 90:
        flags.append(
            f"HIGH DEDUP: {dedup_rate}% duplicates — queries may be stale"
        )

    yesterday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_manifest = load_manifest(yesterday)

    if manifest and manifest.get("site_summary"):
        for site, info in sorted(manifest["site_summary"].items()):
            if info.get("circuit_breaker_tripped"):
                flags.append(f"CIRCUIT BREAKER: {site.upper()} skipped remaining tasks")
            if info.get("tasks_exception", 0) > 0:
                flags.append(
                    f"REQUEST FAILURES: {site.upper()} had {info['tasks_exception']} failed tasks"
                )
        rate_limit_total = manifest.get("rate_limit_summary", {}).get("total", 0)
        if rate_limit_total >= 3:
            flags.append(f"RATE LIMITING: {rate_limit_total} suspected rate-limited tasks")

        todays_sites = set(manifest["site_summary"].keys())
        y_sites = set(yesterday_manifest.get("site_summary", {}).keys())
        if "indeed" in y_sites and "indeed" not in todays_sites:
            flags.append("MISSING SITE: Indeed ran yesterday but not today")
    else:
        t_has_indeed = any("[INDEED] Starting" in l for l in scrape_lines)
        if not t_has_indeed:
            y_scrape = LOG_DIR / f"scrape_{yesterday}.log"
            if y_scrape.exists():
                with open(y_scrape) as f:
                    for line in f:
                        if "[INDEED] Starting" in line:
                            flags.append("MISSING SITE: Indeed ran yesterday but not today")
                            break

    if (LOG_DIR / f"yc_{yesterday}.log").exists() and not (LOG_DIR / f"yc_{today}.log").exists():
        flags.append("MISSING SITE: YC log exists for yesterday but not today")

    if status == "success" and swe_count == 0:
        flags.append("ZERO JOBS: Status is success but SWE count is 0")

    return flags


def build_summary(args, today: str) -> str:
    """Build the full summary email body."""
    lines = []
    status_icon = {"success": "OK", "failure": "FAILED", "warning": "WARNING"}[args.status]

    # Header
    lines.append(f"=== SWE Job Scraper Daily Report ===")
    lines.append(f"Date: {today}")
    lines.append(f"Status: {status_icon}")
    lines.append(f"Attempts: {args.attempt}")
    if args.message:
        lines.append(f"Note: {args.message}")
    lines.append("")

    # Load scrape log once — shared by multiple functions
    scrape_lines = _read_log_lines(LOG_DIR / f"scrape_{today}.log")
    manifest = load_manifest(today)

    # Pre-compute values shared across sections
    scrape_stats = get_scrape_stats(scrape_lines, manifest)
    avg_7d = get_7day_average(today)

    # Anomaly flags (top of email for visibility)
    anomaly_flags = get_anomaly_flags(
        today, args.swe_count, args.status, scrape_stats, avg_7d, scrape_lines, manifest,
    )
    if anomaly_flags:
        lines.append("--- Anomaly Flags ---")
        for flag in anomaly_flags:
            lines.append(f"  !! {flag}")
        lines.append("")

    # Per-site status
    site_status = get_site_status(today, scrape_lines, manifest)
    if site_status:
        lines.append("--- Per-Site Status ---")
        lines.extend(site_status)
        lines.append("")

    # Today's counts
    lines.append("--- Today's Scrape ---")
    files = get_today_files(today)
    if files:
        for name, info in files.items():
            label = name.replace(f"{today}_", "").replace(".csv", "")
            lines.append(f"  {label}: {info['rows']:,} jobs")
        lines.append(f"  Total: {sum(f['rows'] for f in files.values()):,} jobs")
    else:
        lines.append("  No files found for today")
    lines.append("")

    # Trend comparison
    lines.append("--- Trend ---")
    today_swe = files.get(f"{today}_swe_jobs.csv", {}).get("rows", 0) if files else 0
    yesterday_count = get_previous_day_count(today)
    if yesterday_count is not None:
        delta = today_swe - yesterday_count
        sign = "+" if delta >= 0 else ""
        lines.append(f"  vs yesterday: {sign}{delta} ({yesterday_count:,} -> {today_swe:,})")
    if avg_7d is not None:
        lines.append(f"  7-day avg: {avg_7d:,.0f} SWE jobs/day")
    if yesterday_count is None and avg_7d is None:
        lines.append("  No prior data for comparison")
    lines.append("")

    # Scrape stats
    if scrape_stats:
        lines.append("--- Scrape Stats ---")
        lines.append(
            f"  Requests: {scrape_stats['requests_done']:,}/{scrape_stats['requests_total']:,}"
            f" | Raw: {scrape_stats['raw']:,}"
            f" | New: {scrape_stats['new']:,}"
            f" | Dupes: {scrape_stats['dupes']:,} ({scrape_stats['dedup_rate']}%)"
            f" | Empty: {scrape_stats['empty']}"
            f" | Errors: {scrape_stats['errors']}"
            f" | Skipped: {scrape_stats['skipped']}"
            f" | Rate-limited: {scrape_stats['rate_limited']}"
        )
        lines.append("")

    # Data quality
    quality = get_data_quality(today)
    if quality:
        lines.append("--- Data Quality ---")
        for field, pct in quality.items():
            lines.append(f"  {field}: {pct}")
        lines.append("")

    # Sample titles
    samples = get_sample_titles(today)
    if samples:
        lines.append("--- Sample Postings ---")
        for s in samples:
            lines.append(s)
        lines.append("")

    # Cumulative stats
    cum = get_cumulative_stats()
    if cum:
        lines.append("--- Cumulative ---")
        if "days_scraped" in cum:
            lines.append(f"  Days scraped: {cum['days_scraped']} ({cum['first_date']} to {cum['last_date']})")
        if "total_swe_rows" in cum:
            lines.append(f"  Total SWE rows (all CSVs): {cum['total_swe_rows']:,}")
        if "unified_rows" in cum:
            lines.append(f"  Unified parquet: {cum['unified_rows']:,} rows ({cum.get('unified_size', '?')})")
        elif "unified_size" in cum:
            lines.append(f"  Unified parquet: {cum['unified_size']}")
        lines.append("")

    # Error details
    error_details = get_traceback(today, scrape_lines)
    if error_details:
        lines.append("--- Error Details ---")
        lines.extend(error_details)
        lines.append("")

    return "\n".join(lines)


def send_sns(topic_arn: str, subject: str, message: str):
    """Publish to SNS topic."""
    import boto3
    region = topic_arn.split(":")[3]
    client = boto3.client("sns", region_name=region)
    client.publish(
        TopicArn=topic_arn,
        Subject=subject[:100],
        Message=message,
    )
    print(f"  [sns] Sent to {topic_arn}")


def send_file_alert(status: str, message: str, swe_count: int, total_count: int):
    """Write status to JSON file (for S3 sync / dashboard)."""
    path = DATA_DIR / "scraper_status.json"
    payload = {
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "swe_count": swe_count,
        "total_count": total_count,
    }

    history = []
    if path.exists():
        try:
            with open(path) as f:
                existing = json.load(f)
            history = existing.get("history", [])
        except (json.JSONDecodeError, KeyError):
            pass

    history.append(payload)
    history = history[-30:]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"current": payload, "history": history}, f, indent=2)
    print(f"  [file] Written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Send scraper daily summary")
    parser.add_argument("--status", required=True, choices=["success", "failure", "warning"])
    parser.add_argument("--message", default="")
    parser.add_argument("--swe-count", type=int, default=0)
    parser.add_argument("--total-count", type=int, default=0)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--date", default=None, help="Report date (YYYY-MM-DD). Defaults to today.")
    args = parser.parse_args()

    today = args.date or datetime.now().strftime("%Y-%m-%d")
    subject = f"[SWE Scraper] {args.status.upper()} - {today} - {args.swe_count} SWE jobs"
    summary = build_summary(args, today)

    print(f"Sending alert: {args.status}")
    print(summary)

    # Always write status file
    send_file_alert(args.status, args.message, args.swe_count, args.total_count)

    # Send via SNS if configured
    topic_arn = os.environ.get("SNS_TOPIC_ARN", "")
    if topic_arn:
        try:
            send_sns(topic_arn, subject, summary)
        except Exception as e:
            print(f"  [sns] Failed: {e}", file=sys.stderr)
    else:
        print("  [sns] SNS_TOPIC_ARN not set, skipping")


if __name__ == "__main__":
    main()
