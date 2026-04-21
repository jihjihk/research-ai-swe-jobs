#!/usr/bin/env python3
"""
Back up final pipeline outputs and LLM cache to S3 with timestamped paths.

Uploads to: s3://swe-labor-research/backups/<YYYY-MM-DD_HHMMSS>/
    unified.parquet
    unified_observations.parquet
    unified_core.parquet
    unified_core_observations.parquet
    quality_report.json
    preprocessing_log.txt
    llm_responses.db

Usage:
    python preprocessing/scripts/backup_to_s3.py            # Back up now
    python preprocessing/scripts/backup_to_s3.py --dry-run  # Show what would be uploaded
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "preprocessing" / "cache"

S3_BUCKET = "s3://swe-labor-research"
S3_BACKUP_PREFIX = "backups"

# Files to back up: (local_path, s3_filename)
BACKUP_MANIFEST = [
    (DATA_DIR / "unified.parquet", "unified.parquet"),
    (DATA_DIR / "unified_observations.parquet", "unified_observations.parquet"),
    (DATA_DIR / "unified_core.parquet", "unified_core.parquet"),
    (DATA_DIR / "unified_core_observations.parquet", "unified_core_observations.parquet"),
    (DATA_DIR / "quality_report.json", "quality_report.json"),
    (DATA_DIR / "preprocessing_log.txt", "preprocessing_log.txt"),
    (CACHE_DIR / "llm_responses.db", "llm_responses.db"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def s3_cp(local_path: Path, s3_uri: str) -> bool:
    """Upload a single file to S3. Returns True on success."""
    result = subprocess.run(
        ["aws", "s3", "cp", str(local_path), s3_uri, "--quiet"],
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min per file (unified.parquet is ~6 GB)
    )
    if result.returncode != 0:
        log.error(f"  FAILED: {result.stderr.strip()}")
        return False
    return True


def run_backup(dry_run: bool = False) -> bool:
    """Upload all files in the manifest to a timestamped S3 prefix."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    s3_prefix = f"{S3_BUCKET}/{S3_BACKUP_PREFIX}/{timestamp}"

    log.info(f"Backup destination: {s3_prefix}/")

    # Check which files exist
    to_upload = []
    for local_path, s3_name in BACKUP_MANIFEST:
        if local_path.exists():
            size_mb = local_path.stat().st_size / 1e6
            to_upload.append((local_path, s3_name, size_mb))
            log.info(f"  Will upload: {local_path.name} ({size_mb:.1f} MB)")
        else:
            log.warning(f"  Skipping (not found): {local_path}")

    if not to_upload:
        log.error("No files found to back up.")
        return False

    if dry_run:
        log.info("Dry run -- no files uploaded.")
        return True

    # Upload
    failures = 0
    for local_path, s3_name, size_mb in to_upload:
        s3_uri = f"{s3_prefix}/{s3_name}"
        log.info(f"  Uploading {local_path.name} ({size_mb:.1f} MB) -> {s3_uri}")
        if not s3_cp(local_path, s3_uri):
            failures += 1

    if failures:
        log.error(f"Backup completed with {failures} failure(s).")
        return False

    log.info(f"Backup complete: {len(to_upload)} files -> {s3_prefix}/")
    return True


def main():
    parser = argparse.ArgumentParser(description="Back up pipeline outputs to S3")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be uploaded without uploading")
    args = parser.parse_args()

    success = run_backup(dry_run=args.dry_run)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
