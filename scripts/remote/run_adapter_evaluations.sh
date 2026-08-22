#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDEOTRACE_ROOT:-/lavender/VideoTrace}"
ENV_ROOT="${VIDEOTRACE_ENV:-/linyuanping/miniconda3/envs/guide2play-qwen35}"
PY="$ENV_ROOT/bin/python"
PACK="${VIDEOTRACE_CANONICAL_PACK:-$ROOT/outputs/iboy_qwen35/cola_review/knowledge_pack.json}"
MODEL="${VIDEOTRACE_QWEN_MODEL:-/lavender/models/Qwen3.5-9B}"
RUN_DIR="$ROOT/outputs_runtime/revalidation"

export PYTHONPATH="$ROOT/scripts/remote/python_compat:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$ROOT"
mkdir -p "$RUN_DIR" "$ROOT/outputs/reports"

gpu_pair="$("$PY" "$ROOT/scripts/remote/select_gpus.py" \
  --wait-seconds "${VIDEOTRACE_GPU_WAIT_SECONDS:-0}" \
  --audit-log "$RUN_DIR/gpu_selection_adapter.json")"
adapter_gpu="${gpu_pair%%,*}"
export CUDA_VISIBLE_DEVICES="$adapter_gpu"
export VIDEOTRACE_PHYSICAL_GPUS="$adapter_gpu"
echo "VideoTrace adapter evaluation GPU: physical=$adapter_gpu, runtime=cuda:0"

evaluate() {
  local label="$1"
  shift
  echo "  evaluating $label"
  "$PY" scripts/evaluate_qwen35_adapter.py "$@" >"$RUN_DIR/$label.log"
}

sft_baseline="outputs/reports/qwen35_sft_eval_baseline.json"
sft_adapter="outputs/reports/qwen35_sft_eval_adapter.json"
sft_comparison="outputs/reports/qwen35_sft_eval.json"
dpo_baseline="outputs/reports/qwen35_dpo_eval_baseline.json"
dpo_adapter="outputs/reports/qwen35_dpo_eval_adapter.json"
dpo_comparison="outputs/reports/qwen35_dpo_eval.json"

# The baseline is independent of the candidate adapter. Evaluate it once, then
# copy the exact bytes so both candidate comparisons are bound to the same
# baseline generation rather than two stochastic or time-separated runs.
evaluate qwen35_baseline \
  --variant baseline \
  --pack "$PACK" \
  --model "$MODEL" \
  --device cuda:0 \
  --output "$sft_baseline"
cp -- "$sft_baseline" "$dpo_baseline"

evaluate qwen35_sft_adapter \
  --variant adapter \
  --pack "$PACK" \
  --model "$MODEL" \
  --adapter outputs/models/qwen35_sft_adapter \
  --candidate-id qwen35_sft \
  --device cuda:0 \
  --output "$sft_adapter"
evaluate qwen35_sft_compare \
  --variant compare \
  --input-baseline "$sft_baseline" \
  --input-adapter "$sft_adapter" \
  --output "$sft_comparison"

evaluate qwen35_dpo_adapter \
  --variant adapter \
  --pack "$PACK" \
  --model "$MODEL" \
  --adapter outputs/models/qwen35_dpo_adapter \
  --candidate-id qwen35_dpo \
  --device cuda:0 \
  --output "$dpo_adapter"
evaluate qwen35_dpo_compare \
  --variant compare \
  --input-baseline "$dpo_baseline" \
  --input-adapter "$dpo_adapter" \
  --output "$dpo_comparison"

echo "  selecting the hash-bound product adapter"
"$PY" scripts/select_best_qwen35_adapter.py >"$RUN_DIR/select_best_adapter.log"
"$PY" - <<'PYEOF'
import json
from pathlib import Path

registry = json.loads(Path("outputs/models/best_adapter.json").read_text(encoding="utf-8"))
print("selected adapter:", registry["selected_candidate_id"])
for candidate_id, candidate in registry["candidates"].items():
    print(
        f"  {candidate_id}: product={candidate['product_gate']['passed']} "
        f"preference={candidate['preference_gate']['passed']}"
    )
PYEOF
