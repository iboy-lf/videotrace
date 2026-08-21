#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDEOTRACE_ROOT:-/lavender/VideoTrace}"
HOST="${VIDEOTRACE_WEB_HOST:-127.0.0.1}"
PORT="${VIDEOTRACE_WEB_PORT:-7860}"
RUN_DIR="$ROOT/outputs_runtime/web"
PID_FILE="$RUN_DIR/server.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No managed VideoTrace Web PID file exists at $PID_FILE"
  exit 0
fi

pid="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
  echo "Refusing to stop: invalid PID file content at $PID_FILE" >&2
  exit 1
fi
if ! kill -0 "$pid" 2>/dev/null; then
  echo "Managed VideoTrace Web PID $pid is no longer running."
  rm -f -- "$PID_FILE"
  exit 0
fi

cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
expected_root="$(readlink -f "$ROOT")"
if [[ "$cwd" != "$expected_root" || "$cmdline" != *"scripts/run_web.py"* || "$cmdline" != *"--port $PORT"* ]]; then
  echo "Refusing to stop PID $pid: it is not the managed VideoTrace Web process." >&2
  echo "cwd=$cwd cmdline=$cmdline" >&2
  exit 1
fi

if [[ "${VIDEOTRACE_REQUIRE_IDLE_STOP:-1}" == "1" ]]; then
  python_bin="${VIDEOTRACE_ENV:-/linyuanping/miniconda3/envs/guide2play-qwen35}/bin/python"
  "$python_bin" - "$HOST" "$PORT" <<'PY'
import json
import sys
import urllib.request

host, port = sys.argv[1], int(sys.argv[2])
try:
    with urllib.request.urlopen(f"http://{host}:{port}/api/health", timeout=3) as response:
        payload = json.load(response)
except Exception as exc:
    raise SystemExit(f"Refusing idle-checked stop because health could not be read: {exc}")
counts = dict(payload.get("jobs", {}).get("counts") or {})
if int(counts.get("queued", 0)) or int(counts.get("running", 0)):
    raise SystemExit(f"Refusing to stop VideoTrace while jobs are active: {counts}")
PY
fi

kill -TERM "$pid"
for _ in $(seq 1 30); do
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f -- "$PID_FILE"
    echo "Stopped managed VideoTrace Web process PID $pid."
    exit 0
  fi
  sleep 1
done

echo "VideoTrace Web PID $pid did not exit after SIGTERM; no SIGKILL was sent." >&2
exit 1
