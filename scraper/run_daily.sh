#!/usr/bin/env bash
#
# run_daily.sh — Cron-safe wrapper for the US SWE job scraper (LinkedIn + Indeed).
#
# Features:
#   - Lock file prevents overlapping runs
#   - Retries on failure (up to 3 attempts with backoff)
#   - Stale lock detection (auto-clears locks older than 6 hours)
#   - Log rotation (keeps last 30 days)
#   - Exit code reporting
#
# Usage:
#   ./run_daily.sh                    # Normal daily run (all sites)
#   ./run_daily.sh --catchup          # 48-hour lookback (missed a day)
#   ./run_daily.sh --full             # Full run (all 12 queries x 20 cities)
#   ./run_daily.sh --quick            # Quick run (4 queries x 10 cities)
#   ./run_daily.sh --sites linkedin   # LinkedIn only
#   ./run_daily.sh --sites indeed     # Indeed only
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOCK_FILE="$PROJECT_DIR/.scraper.lock"
LOCK_MAX_AGE_SECONDS=21600  # 6 hours — stale lock threshold (2 sites take longer)
MAX_RETRIES=3
RETRY_DELAY_BASE=300  # 5 minutes, doubles each retry
LOG_RETENTION_DAYS=30
S3_BUCKET="${S3_BUCKET:-}"  # set to "s3://your-bucket" to enable S3 sync

# Determine Python binary
if [[ -f "$PROJECT_DIR/.venv/bin/python3" ]]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python3"
elif command -v python3 &>/dev/null; then
    PYTHON="$(command -v python3)"
else
    echo "$(date -Iseconds) [FATAL] python3 not found" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
SCRAPER_ARGS=""
MODE="default"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --catchup)
            SCRAPER_ARGS="$SCRAPER_ARGS --hours-old 48"
            MODE="catchup"
            shift ;;
        --full)
            SCRAPER_ARGS="$SCRAPER_ARGS --results 25"
            MODE="full"
            shift ;;
        --quick)
            SCRAPER_ARGS="$SCRAPER_ARGS --quick --results 25"
            MODE="quick"
            shift ;;
        --sites)
            SCRAPER_ARGS="$SCRAPER_ARGS --sites $2"
            shift 2 ;;
        --no-harmonize)
            SCRAPER_ARGS="$SCRAPER_ARGS --no-harmonize"
            shift ;;
        *)
            SCRAPER_ARGS="$SCRAPER_ARGS $1"
            shift ;;
    esac
done

log() {
    echo "$(date -Iseconds) [run_daily] $*"
}

# ---------------------------------------------------------------------------
# Lock file management
# ---------------------------------------------------------------------------
acquire_lock() {
    if [[ -f "$LOCK_FILE" ]]; then
        lock_pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
        lock_age=0

        # Check lock age
        if [[ -f "$LOCK_FILE" ]]; then
            if command -v stat &>/dev/null; then
                # macOS and Linux compatible
                if [[ "$(uname)" == "Darwin" ]]; then
                    lock_created=$(stat -f %m "$LOCK_FILE" 2>/dev/null || echo 0)
                else
                    lock_created=$(stat -c %Y "$LOCK_FILE" 2>/dev/null || echo 0)
                fi
                now=$(date +%s)
                lock_age=$((now - lock_created))
            fi
        fi

        # Check if the locking process is still alive
        if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
            if [[ $lock_age -gt $LOCK_MAX_AGE_SECONDS ]]; then
                log "WARN: Stale lock (PID $lock_pid, age ${lock_age}s > ${LOCK_MAX_AGE_SECONDS}s). Clearing."
                rm -f "$LOCK_FILE"
            else
                log "ERROR: Another instance running (PID $lock_pid, age ${lock_age}s). Exiting."
                exit 0
            fi
        else
            log "WARN: Stale lock file (PID $lock_pid not running). Clearing."
            rm -f "$LOCK_FILE"
        fi
    fi

    echo $$ > "$LOCK_FILE"
    trap 'rm -f "$LOCK_FILE"' EXIT INT TERM
}

# ---------------------------------------------------------------------------
# Log rotation
# ---------------------------------------------------------------------------
rotate_logs() {
    if [[ -d "$PROJECT_DIR/logs" ]]; then
        find "$PROJECT_DIR/logs" -name "*.log" -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null || true
        deleted=$(find "$PROJECT_DIR/logs" -name "*.log" -mtime +$LOG_RETENTION_DAYS 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$deleted" -gt 0 ]]; then
            log "Rotated $deleted old log file(s)"
        fi
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
acquire_lock
rotate_logs

log "Starting scraper (mode=$MODE, python=$PYTHON)"
log "Args: $SCRAPER_ARGS"

attempt=0
exit_code=1

while [[ $attempt -lt $MAX_RETRIES ]]; do
    attempt=$((attempt + 1))
    log "Attempt $attempt/$MAX_RETRIES..."

    if $PYTHON "$SCRIPT_DIR/scrape_linkedin_swe.py" $SCRAPER_ARGS; then
        exit_code=0
        log "Scrape succeeded on attempt $attempt"
        break
    else
        exit_code=$?
        log "Scrape failed (exit code $exit_code) on attempt $attempt"

        if [[ $attempt -lt $MAX_RETRIES ]]; then
            delay=$((RETRY_DELAY_BASE * (2 ** (attempt - 1))))
            log "Retrying in ${delay}s..."
            sleep $delay
        fi
    fi
done

# ---------------------------------------------------------------------------
# Post-run summary & alerts
# ---------------------------------------------------------------------------
TODAY=$(date +%Y-%m-%d)
SWE_FILE="$PROJECT_DIR/data/scraped/${TODAY}_swe_jobs.csv"
ALERT_SCRIPT="$SCRIPT_DIR/send_alert.py"  # send_alert.py is in scraper/

swe_count=0
total_count=0

if [[ -f "$SWE_FILE" ]]; then
    swe_count=$(($(wc -l < "$SWE_FILE") - 1))  # subtract header
    file_size=$(du -h "$SWE_FILE" | cut -f1)
    log "Output: $SWE_FILE ($swe_count rows, $file_size)"
else
    log "WARN: No output file for today ($SWE_FILE)"
fi

# Count non-SWE too for total
NON_SWE_FILE="$PROJECT_DIR/data/scraped/${TODAY}_non_swe_jobs.csv"
if [[ -f "$NON_SWE_FILE" ]]; then
    non_swe_count=$(($(wc -l < "$NON_SWE_FILE") - 1))
    total_count=$((swe_count + non_swe_count))
else
    total_count=$swe_count
fi

# Count total accumulated data
total_csvs=$(ls "$PROJECT_DIR/data/scraped/"*_swe_jobs.csv 2>/dev/null | wc -l | tr -d ' ')
log "Total daily files accumulated: $total_csvs"

# Check unified dataset
UNIFIED_FILE="$PROJECT_DIR/data/unified.parquet"
if [[ -f "$UNIFIED_FILE" ]]; then
    unified_size=$(du -h "$UNIFIED_FILE" | cut -f1)
    log "Unified dataset: $UNIFIED_FILE ($unified_size)"
else
    log "WARN: Unified dataset not found at $UNIFIED_FILE"
fi

# ---------------------------------------------------------------------------
# Send alerts
# ---------------------------------------------------------------------------
send_alert() {
    local status="$1"
    local message="$2"
    if [[ -f "$ALERT_SCRIPT" ]]; then
        $PYTHON "$ALERT_SCRIPT" \
            --status "$status" \
            --message "$message" \
            --swe-count "$swe_count" \
            --total-count "$total_count" \
            --attempt "$attempt" \
            2>&1 | while read -r line; do log "  $line"; done
    fi
}

if [[ $exit_code -eq 0 ]]; then
    if [[ $swe_count -lt 10 ]] && [[ "$MODE" != "quick" ]] && [[ "$MODE" != "default" || $swe_count -gt 0 ]]; then
        # Suspiciously low count for a full run
        log "WARNING: Only $swe_count SWE jobs (expected more)"
        send_alert "warning" "Low yield: only $swe_count SWE jobs collected (mode=$MODE)"
    else
        log "Success: $swe_count SWE jobs collected"
        send_alert "success" "Collected $swe_count SWE jobs, $total_count total (mode=$MODE)"
    fi
else
    log "FAILED after $MAX_RETRIES attempts"
    send_alert "failure" "Scraper failed after $MAX_RETRIES attempts (mode=$MODE, exit=$exit_code)"
fi

# ---------------------------------------------------------------------------
# S3 sync (if configured)
# ---------------------------------------------------------------------------
if [[ -n "$S3_BUCKET" ]] && [[ $exit_code -eq 0 ]]; then
    log "Syncing data to $S3_BUCKET..."
    if command -v aws &>/dev/null; then
        aws s3 sync "$PROJECT_DIR/data/scraped/" "$S3_BUCKET/scraped/" --quiet 2>&1 | while read -r line; do log "  $line"; done
        [[ -f "$PROJECT_DIR/data/scraper_status.json" ]] && aws s3 cp "$PROJECT_DIR/data/scraper_status.json" "$S3_BUCKET/scraper_status.json" --quiet
        log "S3 sync complete"
    else
        log "WARN: aws CLI not found, skipping S3 sync"
    fi
fi

log "Done (exit code $exit_code)"
exit $exit_code
