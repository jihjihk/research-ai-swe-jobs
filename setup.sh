#!/usr/bin/env bash
#
# setup.sh — One-command setup for the LinkedIn SWE job scraper.
# Run this on any new machine to install dependencies, verify the
# scraper works, and optionally install the daily cron job.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh              # Interactive setup
#   ./setup.sh --no-cron    # Skip cron job installation
#   ./setup.sh --cron-only  # Only install cron job (deps already set up)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { printf "\033[1;34m[INFO]\033[0m  %s\n" "$*"; }
ok()    { printf "\033[1;32m[OK]\033[0m    %s\n" "$*"; }
warn()  { printf "\033[1;33m[WARN]\033[0m  %s\n" "$*"; }
fail()  { printf "\033[1;31m[FAIL]\033[0m  %s\n" "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
SKIP_CRON=false
CRON_ONLY=false
CRON_HOUR="6"
CRON_MINUTE="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cron)   SKIP_CRON=true; shift ;;
        --cron-only) CRON_ONLY=true; shift ;;
        --cron-hour) CRON_HOUR="$2"; shift 2 ;;
        *) warn "Unknown arg: $1"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 1: Detect Python
# ---------------------------------------------------------------------------
if ! $CRON_ONLY; then

info "Detecting Python..."
PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [[ "$major" -ge 3 && "$minor" -ge 10 ]]; then
            PYTHON="$(command -v "$candidate")"
            ok "Found $candidate ($ver) at $PYTHON"
            break
        else
            warn "$candidate is version $ver (need >= 3.10)"
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    fail "Python >= 3.10 not found. Install it first:
    macOS:   brew install python@3.12
    Ubuntu:  sudo apt install python3.12 python3.12-venv
    Fedora:  sudo dnf install python3.12"
fi

# ---------------------------------------------------------------------------
# Step 2: Create virtual environment (if not already in one)
# ---------------------------------------------------------------------------
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    info "Creating virtual environment..."
    if [[ ! -d ".venv" ]]; then
        "$PYTHON" -m venv .venv
        ok "Created .venv"
    else
        ok ".venv already exists"
    fi
    source .venv/bin/activate
    PYTHON="$(command -v python3)"
    ok "Activated .venv (python: $PYTHON)"
else
    ok "Already in a virtual environment: $VIRTUAL_ENV"
fi

# ---------------------------------------------------------------------------
# Step 3: Install dependencies
# ---------------------------------------------------------------------------
info "Installing dependencies..."
pip install --upgrade pip -q
pip install python-jobspy pandas pyarrow -q
ok "Dependencies installed"

# Verify import
"$PYTHON" -c "from jobspy import scrape_jobs; import pandas; print('Imports OK')" \
    || fail "Failed to import required packages"

# ---------------------------------------------------------------------------
# Step 4: Create directories
# ---------------------------------------------------------------------------
info "Creating directories..."
mkdir -p data/scraped logs
ok "data/scraped/ and logs/ ready"

# ---------------------------------------------------------------------------
# Step 5: Test scrape
# ---------------------------------------------------------------------------
info "Running test scrape (1 query, 1 city, 5 results)..."
"$PYTHON" scraper/scrape_linkedin_swe.py --test
if [[ $? -eq 0 ]]; then
    ok "Test scrape succeeded"
    CSV_COUNT=$(ls data/scraped/*.csv 2>/dev/null | wc -l | tr -d ' ')
    ok "Found $CSV_COUNT CSV file(s) in data/scraped/"
else
    fail "Test scrape failed — check logs/ for details"
fi

fi  # end !CRON_ONLY

# ---------------------------------------------------------------------------
# Step 6: Install cron job
# ---------------------------------------------------------------------------
if $SKIP_CRON; then
    info "Skipping cron job installation (--no-cron)"
else
    # Resolve paths for cron
    VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"
    if [[ ! -f "$VENV_PYTHON" ]]; then
        VENV_PYTHON="$(command -v python3)"
        warn "No .venv found, using system python: $VENV_PYTHON"
    fi
    RUN_SCRIPT="$SCRIPT_DIR/scraper/run_daily.sh"

    if [[ ! -f "$RUN_SCRIPT" ]]; then
        fail "run_daily.sh not found at $RUN_SCRIPT"
    fi
    chmod +x "$RUN_SCRIPT"

    CRON_LINE="$CRON_MINUTE $CRON_HOUR * * * $RUN_SCRIPT >> $SCRIPT_DIR/logs/cron.log 2>&1"

    info "Proposed cron job (runs daily at ${CRON_HOUR}:$(printf '%02d' "$CRON_MINUTE")):"
    echo "  $CRON_LINE"
    echo ""

    # Check if already installed
    if crontab -l 2>/dev/null | grep -qF "run_daily.sh"; then
        warn "A cron job for run_daily.sh already exists:"
        crontab -l 2>/dev/null | grep "run_daily.sh"
        echo ""
        read -rp "Replace it? [y/N] " yn
        if [[ "$yn" =~ ^[Yy] ]]; then
            crontab -l 2>/dev/null | grep -vF "run_daily.sh" | { cat; echo "$CRON_LINE"; } | crontab -
            ok "Cron job replaced"
        else
            info "Keeping existing cron job"
        fi
    else
        read -rp "Install this cron job? [y/N] " yn
        if [[ "$yn" =~ ^[Yy] ]]; then
            (crontab -l 2>/dev/null || true; echo "$CRON_LINE") | crontab -
            ok "Cron job installed"
        else
            info "Skipped. To install manually, run:"
            echo "  (crontab -l 2>/dev/null; echo '$CRON_LINE') | crontab -"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
ok "Setup complete!"
echo ""
echo "  Quick commands:"
echo "    ./scraper/run_daily.sh              # Run a full daily scrape now"
echo "    python3 scraper/scrape_linkedin_swe.py --test   # Quick test (5 results)"
echo "    crontab -l                  # Verify cron job"
echo "    ls data/scraped/            # Check collected data"
echo "    ls logs/                    # Check logs"
echo ""
