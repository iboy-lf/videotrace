#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDEOTRACE_ROOT:-/lavender/VideoTrace}"
ENV_ROOT="${VIDEOTRACE_ENV:-/linyuanping/miniconda3/envs/guide2play-qwen35}"
PY="$ENV_ROOT/bin/python"
RUN_DIR="$ROOT/outputs_runtime/revalidation"

export PYTHONPATH="$ROOT/scripts/remote/python_compat:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$ROOT"
mkdir -p "$RUN_DIR"

gpu_pair="$("$PY" "$ROOT/scripts/remote/select_gpus.py" \
  --wait-seconds "${VIDEOTRACE_GPU_WAIT_SECONDS:-0}" \
  --audit-log "$RUN_DIR/gpu_selection_error_analysis.json")"
export CUDA_VISIBLE_DEVICES="$gpu_pair"
export VIDEOTRACE_PHYSICAL_GPUS="$gpu_pair"
echo "VideoTrace regression GPUs: physical=$gpu_pair, Qwen=cuda:0, SigLIP=cuda:1"

exec "$PY" scripts/run_regression_suite.py "$@"
