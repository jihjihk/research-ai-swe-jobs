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
import sys
from datetime import datetime, timedelta
from pathlib import Path

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


def get_previous_day_count(today: str):
    """Get yesterday's SWE job count for comparison."""
    yesterday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    path = SCRAPED_DIR / f"{yesterday}_swe_jobs.csv"
    if path.exists():
        with open(path) as f:
            return sum(1 for _ in f) - 1  # subtract header
    return None


def get_7day_average(today: str):
    """Get 7-day average SWE job count."""
    counts = []
    for i in range(1, 8):
        day = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=i)).strftime("%Y-%m-%d")
        path = SCRAPED_DIR / f"{day}_swe_jobs.csv"
        if path.exists():
            with open(path) as f:
                counts.append(sum(1 for _ in f) - 1)
    return round(sum(counts) / len(counts), 1) if counts else None


def get_sample_titles(today: str, n: int = 5):
    """Get random sample of job titles from today's SWE file."""
    path = SCRAPED_DIR / f"{today}_swe_jobs.csv"
    if not path.exists():
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return []
    sample = random.sample(rows, min(n, len(rows)))
    titles = []
    for r in sample:
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
    """Check null rates for key fields in today's SWE file."""
    path = SCRAPED_DIR / f"{today}_swe_jobs.csv"
    if not path.exists():
        return {}
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}

    key_fields = ["seniority", "location", "company", "company_name", "salary", "skills_raw"]
    quality = {}
    for field in key_fields:
        if field in rows[0]:
            total = len(rows)
            filled = sum(1 for r in rows if r.get(field, "").strip())
            quality[field] = f"{round(filled / total * 100)}% filled"
    return quality


def get_cumulative_stats() -> dict:
    """Stats on total accumulated data."""
    stats = {}
    swe_files = sorted(SCRAPED_DIR.glob("*_swe_jobs.csv"))
    if swe_files:
        stats["days_scraped"] = len(swe_files)
        stats["first_date"] = swe_files[0].name[:10]
        stats["last_date"] = swe_files[-1].name[:10]
        # Total SWE rows across all files
        total_rows = 0
        for f in swe_files:
            with open(f) as fh:
                total_rows += sum(1 for _ in fh) - 1  # subtract header
        stats["total_swe_rows"] = total_rows

    unified = DATA_DIR / "unified.parquet"
    if unified.exists():
        size_mb = unified.stat().st_size / (1024 * 1024)
        stats["unified_size"] = f"{size_mb:.1f} MB"
        # Try to get row count from parquet metadata (fast, no full read)
        try:
            import pyarrow.parquet as pq
            meta = pq.read_metadata(unified)
            stats["unified_rows"] = meta.num_rows
        except Exception:
            pass

    return stats


def get_recent_errors(today: str, n: int = 5):
    """Get last N error/warning lines from today's log."""
    if not LOG_DIR.exists():
        return []
    # Find today's log file
    log_files = sorted(LOG_DIR.glob(f"*{today}*.log"), reverse=True)
    if not log_files:
        log_files = sorted(LOG_DIR.glob("*.log"), reverse=True)
    if not log_files:
        return []

    errors = []
    with open(log_files[0]) as f:
        for line in f:
            lower = line.lower()
            if any(kw in lower for kw in ["error", "warn", "fail", "exception", "traceback"]):
                errors.append(line.strip()[:120])
    return errors[-n:]


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
    avg_7d = get_7day_average(today)
    if yesterday_count is not None:
        delta = today_swe - yesterday_count
        sign = "+" if delta >= 0 else ""
        lines.append(f"  vs yesterday: {sign}{delta} ({yesterday_count:,} -> {today_swe:,})")
    if avg_7d is not None:
        lines.append(f"  7-day avg: {avg_7d:,.0f} SWE jobs/day")
    if yesterday_count is None and avg_7d is None:
        lines.append("  No prior data for comparison")
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

    # Errors
    errors = get_recent_errors(today)
    if errors:
        lines.append("--- Recent Errors/Warnings ---")
        for e in errors:
            lines.append(f"  {e}")
        lines.append("")

    return "\n".join(lines)


def send_sns(topic_arn: str, subject: str, message: str):
    """Publish to SNS topic."""
    import boto3
    client = boto3.client("sns")
    client.publish(
        TopicArn=topic_arn,
        Subject=subject[:100],  # SNS subject limit
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
