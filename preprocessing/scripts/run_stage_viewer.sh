#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PORT="${STAGE_VIEWER_PORT:-8501}"

cd "${ROOT_DIR}"

exec ./.venv/bin/streamlit run \
  preprocessing/viewer/stage_viewer.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT}" \
  --server.headless true
