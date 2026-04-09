#!/usr/bin/env bash
set -euo pipefail

if [ -d .venv ]; then
  source .venv/bin/activate
fi

(uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload) &
BACK_PID=$!

cleanup() {
  kill "$BACK_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
