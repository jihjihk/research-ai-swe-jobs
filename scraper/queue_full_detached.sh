#!/usr/bin/env bash
#
# queue_full_detached.sh
# - If a foreground scrape is currently running, wait for it to finish.
# - If today's manifest exists afterward, regenerate both parquet outputs.
# - Otherwise start a fresh detached full rerun.
#
# This lets us preserve in-flight progress while still making the pipeline
# durable if the interactive session disappears.
#
# Usage:
#   ./scraper/queue_full_detached.sh --wait-pid 15247
#   ./scraper/queue_full_detached.sh
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
launcher_log="$PROJECT_DIR/logs/queue_full_detached_${timestamp}.log"
run_log="$PROJECT_DIR/logs/detached_full_${timestamp}.log"

export PROJECT_DIR SCRIPT_DIR PYTHON WAIT_PID

nohup bash -lc '
set -euo pipefail

log() {
    echo "$(date -Iseconds) [queue_full_detached] $*" | tee -a "'"$launcher_log"'"
}

TODAY=$(date +%Y-%m-%d)
MANIFEST_FILE="$PROJECT_DIR/data/scraped/${TODAY}_manifest.json"
UNIFIED_FILE="$PROJECT_DIR/data/unified.parquet"
OBSERVATIONS_FILE="$PROJECT_DIR/data/unified_observations.parquet"

if [[ -n "${WAIT_PID:-}" ]] && kill -0 "$WAIT_PID" 2>/dev/null; then
    log "Waiting for existing scrape PID $WAIT_PID to finish"
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 60
    done
    log "Existing scrape PID $WAIT_PID exited"
fi

if [[ -f "$MANIFEST_FILE" ]]; then
    log "Found today'\''s manifest at $MANIFEST_FILE; regenerating parquet outputs only"
    "$PYTHON" "$SCRIPT_DIR/harmonize.py" \
        --output "$UNIFIED_FILE" \
        --observations-output "$OBSERVATIONS_FILE" \
        >> "'"$run_log"'" 2>&1
    log "Harmonization complete"
else
    log "No manifest for today; starting detached full rerun"
    bash "$SCRIPT_DIR/run_daily.sh" --full >> "'"$run_log"'" 2>&1
    log "Detached full rerun completed"
fi
' >/dev/null 2>&1 &

pid=$!
echo "Queued detached recovery/finalization job: PID $pid"
echo "Launcher log: $launcher_log"
echo "Run log: $run_log"
