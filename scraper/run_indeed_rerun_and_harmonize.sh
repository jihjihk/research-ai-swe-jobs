#!/usr/bin/env bash
#
# run_indeed_rerun_and_harmonize.sh
# Wait for the current wrapper run to finish, rerun Indeed with the latest
# scraper logic, then regenerate both parquet outputs with the latest harmonizer.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

WAIT_PID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wait-pid)
            WAIT_PID="${2:-}"
            shift 2 ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 1 ;;
    esac
done

if [[ -f "$PROJECT_DIR/.venv/bin/python3" ]]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python3"
else
    PYTHON="$(command -v python3)"
fi

LOCK_FILE="$PROJECT_DIR/.scraper.lock"

log() {
    echo "$(date -Iseconds) [indeed_rerun] $*"
}

if [[ -n "${WAIT_PID:-}" ]] && kill -0 "$WAIT_PID" 2>/dev/null; then
    log "Waiting for wrapper PID $WAIT_PID to finish"
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 60
    done
    log "Wrapper PID $WAIT_PID exited"
fi

if [[ -f "$LOCK_FILE" ]]; then
    lock_pid=$(cat "$LOCK_FILE" 2>/dev/null || true)
    if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
        log "Waiting for lock holder PID $lock_pid to release $LOCK_FILE"
        while kill -0 "$lock_pid" 2>/dev/null; do
            sleep 30
        done
    fi
    rm -f "$LOCK_FILE"
    log "Cleared stale lock file if present"
fi

log "Starting Indeed rerun"
timeout --signal=TERM --kill-after=300 14400 \
    "$PYTHON" "$SCRIPT_DIR/scrape_linkedin_swe.py" \
    --sites indeed \
    --no-harmonize \
    --request-timeout-sec 180 \
    --memory-soft-limit-mb 8192 \
    --memory-hard-limit-mb 10240

log "Indeed rerun completed; rebuilding parquet outputs"
timeout --signal=TERM --kill-after=300 7200 \
    "$PYTHON" "$SCRIPT_DIR/harmonize.py" \
    --output "$PROJECT_DIR/data/unified.parquet" \
    --observations-output "$PROJECT_DIR/data/unified_observations.parquet"

log "Parquet rebuild completed"
