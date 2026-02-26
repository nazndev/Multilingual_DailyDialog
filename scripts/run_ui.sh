#!/usr/bin/env bash
# Run Streamlit UI from repo root. Uses venv if present, installs deps, then launches.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then set -a; source .env; set +a; fi

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

pip install -q -r requirements.txt
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
echo "Starting UI at http://localhost:8501 (Ctrl+C to stop)"
exec streamlit run ui/app.py
