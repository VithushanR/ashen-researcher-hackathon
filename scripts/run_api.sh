#!/usr/bin/env bash
# Start the FastAPI backend.
#
#   ./scripts/run_api.sh          default port 8000
#   PORT=8080 ./scripts/run_api.sh
#
# Run from anywhere; the script locates the repo root itself.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PORT="${PORT:-8000}"

# Prefer the project venv over whatever python is on PATH, so this behaves the
# same whether or not the venv is activated in the caller's shell.
if [ -x ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"   # Windows layout, via Git Bash
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    echo "No .venv found. Create one first:"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate    # or .venv/Scripts/activate on Windows"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo "API      http://127.0.0.1:${PORT}"
echo "Docs     http://127.0.0.1:${PORT}/docs"
echo "Health   http://127.0.0.1:${PORT}/health"
echo

exec "$PYTHON" -m uvicorn src.api.main:app --reload --port "$PORT"
