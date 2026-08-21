#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDEOTRACE_ROOT:-/lavender/VideoTrace}"
ENV_ROOT="${VIDEOTRACE_ENV:-/linyuanping/miniconda3/envs/guide2play-qwen35}"
BASE_URL="${VIDEOTRACE_WEB_URL:-http://127.0.0.1:7860}"
VIDEO="${1:-}"
QUERY="${2:-这个视频主要讲了什么？请给出带时间戳的证据。}"

cd "$ROOT"
args=(
  scripts/browser_e2e.py
  --base-url "$BASE_URL"
  --query "$QUERY"
  --output outputs/reports/browser_e2e.json
)
if [[ -n "$VIDEO" ]]; then
  args+=(--video "$VIDEO")
fi
"$ENV_ROOT/bin/python" "${args[@]}"
