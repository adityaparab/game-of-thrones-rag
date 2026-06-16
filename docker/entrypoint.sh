#!/usr/bin/env bash
# Single image, two web apps. Pick one with APP=api|streamlit (default: api).
# $PORT is provided by Railway; falls back to a sane default for local runs.
set -euo pipefail

APP="${APP:-api}"

case "$APP" in
  api)
    exec uv run --no-sync uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}"
    ;;
  streamlit)
    exec uv run --no-sync streamlit run app_streamlit.py \
      --server.address 0.0.0.0 --server.port "${PORT:-8501}"
    ;;
  *)
    echo "Unknown APP='$APP' (expected 'api' or 'streamlit')" >&2
    exit 1
    ;;
esac
