#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDEOTRACE_ROOT:-/lavender/VideoTrace}"
ENV_ROOT="${VIDEOTRACE_ENV:-/linyuanping/miniconda3/envs/guide2play-qwen35}"
HOST="${VIDEOTRACE_WEB_HOST:-127.0.0.1}"
PORT="${VIDEOTRACE_WEB_PORT:-7860}"

export PYTHONPATH="$ROOT/scripts/remote/python_compat:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export VIDEOTRACE_CONFIG="$ROOT/configs/iboy_qwen35.yaml"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$ROOT"
exec bash "$ROOT/scripts/remote/start_web_service.sh"
