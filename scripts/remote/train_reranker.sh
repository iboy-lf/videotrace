#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDEOTRACE_ROOT:-/lavender/VideoTrace}"
ENV_ROOT="${VIDEOTRACE_ENV:-/linyuanping/miniconda3/envs/guide2play-qwen35}"
SEGMENT_SECONDS="${VIDEOTRACE_RERANKER_SEGMENT_SECONDS:-5}"
DATASET="${VIDEOTRACE_RERANKER_DATASET:-$ROOT/outputs_train/reranker_dev_${SEGMENT_SECONDS}s.jsonl}"
MODEL="${VIDEOTRACE_RERANKER_MODEL:-$ROOT/outputs/models/neural_reranker.pt}"
METRICS="${VIDEOTRACE_RERANKER_METRICS:-$ROOT/outputs/models/neural_reranker_metrics.json}"
MODEL_CARD="${VIDEOTRACE_RERANKER_MODEL_CARD:-$ROOT/outputs/models/neural_reranker_model_card.json}"

export PYTHONPATH="$ROOT/scripts/remote/python_compat:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${QWEN35_GPU:-}" && -n "${SIGLIP_GPU:-}" ]]; then
  gpu_pair="$("$ENV_ROOT/bin/python" "$ROOT/scripts/remote/select_gpus.py" \
    --wait-seconds "${VIDEOTRACE_GPU_WAIT_SECONDS:-0}" \
    --preferred-qwen-index "$QWEN35_GPU" \
    --preferred-siglip-index "$SIGLIP_GPU")"
else
  gpu_pair="$("$ENV_ROOT/bin/python" "$ROOT/scripts/remote/select_gpus.py" \
    --wait-seconds "${VIDEOTRACE_GPU_WAIT_SECONDS:-0}")"
fi
export CUDA_VISIBLE_DEVICES="$gpu_pair"
export VIDEOTRACE_PHYSICAL_GPUS="$gpu_pair"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "VideoTrace reranker data GPUs: physical=$gpu_pair, Qwen=cuda:0, SigLIP=cuda:1"

cd "$ROOT"
"$ENV_ROOT/bin/python" scripts/build_reranker_dataset.py \
  --supervision "$ROOT/data/supervision/reranker_annotations.json" \
  --config "$ROOT/configs/iboy_qwen35.yaml" \
  --split dev \
  --segment-seconds "$SEGMENT_SECONDS" \
  --output "$DATASET"

# The reranker is a tiny feature MLP; CPU training avoids holding a GPU after
# Qwen/SigLIP supervision generation has completed.
CUDA_VISIBLE_DEVICES="" "$ENV_ROOT/bin/python" scripts/tune_reranker.py \
  "$DATASET" \
  --model "$MODEL" \
  --metrics "$METRICS" \
  --model-card "$MODEL_CARD" \
  --device cpu
