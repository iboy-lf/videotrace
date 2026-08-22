#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDEOTRACE_ROOT:-/lavender/VideoTrace}"
ENV_ROOT="${VIDEOTRACE_ENV:-/linyuanping/miniconda3/envs/guide2play-qwen35}"
PY="$ENV_ROOT/bin/python"
export PYTHONPATH="$ROOT/scripts/remote/python_compat:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$ROOT"
gpu_pair="$("$PY" scripts/remote/select_gpus.py --wait-seconds "${VIDEOTRACE_GPU_WAIT_SECONDS:-1800}" --stable-checks 3 --stable-interval-seconds 5 --audit-log outputs/reports/gpu_selection_dpo_sweep.json)"
export CUDA_VISIBLE_DEVICES="$gpu_pair"
export VIDEOTRACE_PHYSICAL_GPUS="$gpu_pair"
echo "VideoTrace DPO sweep GPUs: physical=$gpu_pair, logical=cuda:0,cuda:1"
exec "$PY" scripts/run_dpo_sweep.py "$@"
