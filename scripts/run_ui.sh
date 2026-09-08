#!/usr/bin/env bash
# Start the Streamlit UI.
#
#   ./scripts/run_ui.sh
#   PORT=8502 ./scripts/run_ui.sh
#
# The backend must already be running -- start it with ./scripts/run_api.sh.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PORT="${PORT:-8501}"
API_BASE="${ASHEN_API_BASE:-http://127.0.0.1:8000}"

if [ -x ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    echo "No .venv found. See scripts/run_api.sh for setup."
    exit 1
fi

# Warn rather than fail: the UI shows a "Backend unreachable" panel with
# recovery instructions, which is more useful than refusing to start.
if curl -sf -m 3 "${API_BASE}/health" > /dev/null 2>&1; then
    echo "Backend reachable at ${API_BASE}"
else
    echo "Backend not reachable at ${API_BASE}"
    echo "  Start it in another terminal:  ./scripts/run_api.sh"
    echo "  The UI will still open and will say the backend is down."
    echo
fi

echo "UI       http://localhost:${PORT}"
echo

exec "$PYTHON" -m streamlit run src/ui/app.py --server.port "$PORT"
