#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDEOTRACE_ROOT:-/lavender/VideoTrace}"
ENV_ROOT="${VIDEOTRACE_ENV:-/linyuanping/miniconda3/envs/guide2play-qwen35}"
export PYTHONPATH="$ROOT/scripts/remote/python_compat:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"
"$ENV_ROOT/bin/python" scripts/build_reranker_model_card.py \
  --dataset outputs_train/reranker_dev_5s.jsonl \
  --model outputs/models/neural_reranker.pt \
  --metrics outputs/models/neural_reranker_metrics.json \
  --output outputs/models/neural_reranker_model_card.json

"$ENV_ROOT/bin/python" scripts/validate_interview_package.py \
  --knowledge-pack outputs/iboy_qwen35/cola_review/knowledge_pack.json \
  --checkpoint outputs/models/neural_reranker.pt \
  --metrics outputs/models/neural_reranker_metrics.json \
  --dataset outputs_train/reranker_dev_5s.jsonl \
  --dataset-summary outputs_train/reranker_dev_5s.summary.json \
  --model-card outputs/models/neural_reranker_model_card.json \
  --output outputs/interview_readiness.json

"$ENV_ROOT/bin/python" scripts/build_artifact_manifest.py \
  --output outputs/reports/artifact_manifest.json

"$ENV_ROOT/bin/python" scripts/validate_delivery_package.py \
  --manifest outputs/reports/artifact_manifest.json \
  --output outputs/reports/delivery_readiness.json
