#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDEOTRACE_ROOT:-/lavender/VideoTrace}"
ENV_ROOT="${VIDEOTRACE_ENV:-/linyuanping/miniconda3/envs/guide2play-qwen35}"
HOST="${VIDEOTRACE_WEB_HOST:-127.0.0.1}"
PORT="${VIDEOTRACE_WEB_PORT:-7860}"
RUN_DIR="$ROOT/outputs_runtime/web"
PID_FILE="$RUN_DIR/server.pid"
LOG_FILE="$RUN_DIR/server.log"

mkdir -p "$RUN_DIR"
export PYTHONPATH="$ROOT/scripts/remote/python_compat:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export VIDEOTRACE_CONFIG="$ROOT/configs/iboy_qwen35.yaml"
cd "$ROOT"

source_sha="$("$ENV_ROOT/bin/python" - <<'PY'
from pathlib import Path
from videomemo.eval.reproducibility import source_fingerprint

print(source_fingerprint(Path.cwd()))
PY
)"

health_status() {
  "$ENV_ROOT/bin/python" - "$HOST" "$PORT" "$ROOT" "$source_sha" <<'PY'
import json
from pathlib import Path
import sys
import urllib.request

host, port, expected_root, expected_source = sys.argv[1:]
try:
    with urllib.request.urlopen(f"http://{host}:{port}/api/health", timeout=3) as response:
        payload = json.load(response)
except Exception:
    raise SystemExit(1)
root_matches = Path(str(payload.get("root", ""))).resolve() == Path(expected_root).resolve()
matches = (
    bool(payload.get("ok"))
    and int(payload.get("product_version", 0)) >= 3
    and root_matches
    and str(payload.get("source_sha256", "")) == expected_source
)
raise SystemExit(0 if matches else 2)
PY
}

health_code=0
health_status || health_code=$?
if [[ "$health_code" == "0" ]]; then
  echo "VideoTrace Web already healthy at http://$HOST:$PORT"
  exit 0
fi
if [[ "$health_code" == "2" ]]; then
  if [[ "${VIDEOTRACE_RESTART_STALE:-0}" != "1" ]]; then
    echo "A stale or foreign service is listening at http://$HOST:$PORT; refusing to reuse it." >&2
    exit 1
  fi
  VIDEOTRACE_REQUIRE_IDLE_STOP=1 bash "$ROOT/scripts/remote/stop_web_service.sh"
fi

selector_args=(
  --wait-seconds "${VIDEOTRACE_GPU_WAIT_SECONDS:-1800}"
  --stable-checks 3
  --stable-interval-seconds 5
  --audit-log "$RUN_DIR/gpu_selection_audit.json"
)
if [[ -n "${QWEN35_GPU:-}" || -n "${SIGLIP_GPU:-}" ]]; then
  if [[ -z "${QWEN35_GPU:-}" || -z "${SIGLIP_GPU:-}" ]]; then
    echo "QWEN35_GPU and SIGLIP_GPU must be provided together." >&2
    exit 1
  fi
  selector_args+=(--preferred-qwen-index "$QWEN35_GPU" --preferred-siglip-index "$SIGLIP_GPU")
fi
gpu_pair="$("$ENV_ROOT/bin/python" "$ROOT/scripts/remote/select_gpus.py" "${selector_args[@]}")"
export CUDA_VISIBLE_DEVICES="$gpu_pair"
export VIDEOTRACE_PHYSICAL_GPUS="$gpu_pair"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "VideoTrace Web GPUs: physical=$gpu_pair, Qwen=cuda:0, SigLIP=cuda:1"

nohup "$ENV_ROOT/bin/python" scripts/run_web.py \
  --host "$HOST" \
  --port "$PORT" \
  --config "$ROOT/configs/iboy_qwen35.yaml" \
  --latest-pack "$ROOT/outputs/iboy_qwen35/cola_review/knowledge_pack.json" \
  >"$LOG_FILE" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE.tmp"
mv -f "$PID_FILE.tmp" "$PID_FILE"

for _ in $(seq 1 180); do
  if health_status; then
    echo "VideoTrace Web ready: http://$HOST:$PORT (pid=$pid)"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "VideoTrace Web exited during startup; recent log:"
    tail -100 "$LOG_FILE" || true
    exit 1
  fi
  sleep 2
done

echo "Timed out waiting for VideoTrace Web; recent log:"
tail -100 "$LOG_FILE" || true
exit 1
