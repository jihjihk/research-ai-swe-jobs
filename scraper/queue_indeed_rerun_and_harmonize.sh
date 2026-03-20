#!/usr/bin/env bash
#
# queue_indeed_rerun_and_harmonize.sh
# - Wait for the current full wrapper run to finish (if a PID is provided)
# - Rerun Indeed only for today's window with the latest scraper code
# - Rebuild canonical + observations parquet outputs with the latest harmonizer
#
# This is meant for durable, detached recovery work that should survive the
# current Codex session.
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

mkdir -p "$PROJECT_DIR/logs"

timestamp="$(date +%Y%m%d_%H%M%S)"
launcher_log="$PROJECT_DIR/logs/queue_indeed_rerun_${timestamp}.log"
run_log="$PROJECT_DIR/logs/indeed_rerun_${timestamp}.log"

export PROJECT_DIR SCRIPT_DIR PYTHON WAIT_PID

nohup bash -lc '
set -euo pipefail

log() {
    echo "$(date -Iseconds) [queue_indeed_rerun] $*" | tee -a "'"$launcher_log"'"
}

LOCK_FILE="$PROJECT_DIR/.scraper.lock"

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

log "Starting detached Indeed rerun"
timeout --signal=TERM --kill-after=300 14400 \
    "$PYTHON" "$SCRIPT_DIR/scrape_linkedin_swe.py" \
    --sites indeed \
    --no-harmonize \
    --request-timeout-sec 180 \
    --memory-soft-limit-mb 8192 \
    --memory-hard-limit-mb 10240 \
    >> "'"$run_log"'" 2>&1

log "Indeed rerun completed; rebuilding parquet outputs"
timeout --signal=TERM --kill-after=300 7200 \
    "$PYTHON" "$SCRIPT_DIR/harmonize.py" \
    --output "$PROJECT_DIR/data/unified.parquet" \
    --observations-output "$PROJECT_DIR/data/unified_observations.parquet" \
    >> "'"$run_log"'" 2>&1

log "Parquet rebuild completed"
' >/dev/null 2>&1 &

pid=$!
echo "Queued detached Indeed rerun + harmonize job: PID $pid"
echo "Launcher log: $launcher_log"
echo "Run log: $run_log"
