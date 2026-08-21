#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDEOTRACE_ROOT:-/lavender/VideoTrace}"
ENV_ROOT="${VIDEOTRACE_ENV:-/linyuanping/miniconda3/envs/guide2play-qwen35}"
VIDEO="${1:-$ROOT/data/raw/cola_review.mp4}"
QUERY="${2:-这个视频的整体流程是什么？请概括开场、分国家试喝和最后盲测三个阶段并给出时间戳。}"

export PYTHONPATH="$ROOT/scripts/remote/python_compat:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${QWEN35_GPU:-}" && -n "${SIGLIP_GPU:-}" ]]; then
  gpu_pair="$("$ENV_ROOT/bin/python" "$ROOT/scripts/remote/select_gpus.py" \
    --wait-seconds "${VIDEOTRACE_GPU_WAIT_SECONDS:-0}" \
    --audit-log "$ROOT/outputs/reports/gpu_selection_canonical.json" \
    --preferred-qwen-index "$QWEN35_GPU" \
    --preferred-siglip-index "$SIGLIP_GPU")"
else
  gpu_pair="$("$ENV_ROOT/bin/python" "$ROOT/scripts/remote/select_gpus.py" \
    --wait-seconds "${VIDEOTRACE_GPU_WAIT_SECONDS:-0}" \
    --audit-log "$ROOT/outputs/reports/gpu_selection_canonical.json")"
fi
export CUDA_VISIBLE_DEVICES="$gpu_pair"
export VIDEOTRACE_PHYSICAL_GPUS="$gpu_pair"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "VideoTrace GPUs: physical=$gpu_pair, Qwen=cuda:0, SigLIP=cuda:1"
cd "$ROOT"
"$ENV_ROOT/bin/python" -m videomemo.cli \
  "$VIDEO" \
  --query "$QUERY" \
  --config "$ROOT/configs/iboy_qwen35.yaml"
