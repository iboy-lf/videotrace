#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDEOTRACE_ROOT:-/lavender/VideoTrace}"
PID_FILE="$ROOT/outputs_runtime/qwen35/server.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No Qwen3.5 pid file found."
  exit 0
fi

pid="$(cat "$PID_FILE")"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  for _ in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
fi
rm -f "$PID_FILE"
echo "Qwen3.5 server stopped."
