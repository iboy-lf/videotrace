#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDEOTRACE_ROOT:-/lavender/VideoTrace}"
ENV_ROOT="${VIDEOTRACE_ENV:-/linyuanping/miniconda3/envs/guide2play-qwen35}"
MODEL="${QWEN35_MODEL:-/lavender/models/Qwen3.5-9B}"
HOST="${QWEN35_HOST:-127.0.0.1}"
PORT="${QWEN35_PORT:-8000}"
GPU="${QWEN35_GPU:-0}"
RUN_DIR="$ROOT/outputs_runtime/qwen35"
PID_FILE="$RUN_DIR/server.pid"
LOG_FILE="$RUN_DIR/server.log"

mkdir -p "$RUN_DIR"
if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo "Qwen3.5 server is already healthy at http://$HOST:$PORT"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "A server process is already running with pid=$old_pid; inspect $LOG_FILE"
    exit 1
  fi
  rm -f "$PID_FILE"
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="${HF_HOME:-/lavender/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

nohup "$ENV_ROOT/bin/transformers" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --reasoning off \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --log-level info \
  >"$LOG_FILE" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"

for _ in $(seq 1 180); do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "Qwen3.5 server ready: http://$HOST:$PORT (pid=$pid)"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Qwen3.5 server exited during startup. Last log lines:"
    tail -80 "$LOG_FILE" || true
    exit 1
  fi
  sleep 2
done

echo "Timed out waiting for Qwen3.5 server. Last log lines:"
tail -80 "$LOG_FILE" || true
exit 1
