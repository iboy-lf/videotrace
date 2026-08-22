#!/usr/bin/env bash
# Regenerate every piece of evidence that is bound to the product source tree.
#
# Why this exists: src/videomemo/eval/reproducibility.py:source_fingerprint
# hashes src/, scripts/, configs/, tests/, pyproject.toml and README.md into
# every machine report. Editing any of them invalidates the delivery checks
# until the GPU-dependent artifacts are produced again. This script runs that
# regeneration in dependency order so the six remote checks are not re-run
# by hand and in the wrong sequence.
#
# GPU safety is inherited, not reimplemented: every stage goes through
# scripts/remote/select_gpus.py, which requires consecutive stable probes of
# genuinely idle cards and never touches another user's processes. If no safe
# pair appears within the wait window the script stops with a non-zero exit
# instead of forcing its way onto a busy card.
#
# Usage (on the GPU host):
#   bash scripts/remote/revalidate_all.sh --dry-run       # check preconditions, touch nothing
#   bash scripts/remote/revalidate_all.sh                 # fail fast if GPUs are busy
#   bash scripts/remote/revalidate_all.sh --wait 3600     # wait up to an hour for a safe pair
#   bash scripts/remote/revalidate_all.sh --skip-browser  # everything except the E2E stage
#   bash scripts/remote/revalidate_all.sh --keep-web      # leave the web service resident
#
# The web service is stopped when the run finishes unless --keep-web is given:
# this is a shared host, and a resident 9B model should not hold a card after an
# unattended run.

set -euo pipefail

ROOT="${VIDEOTRACE_ROOT:-/lavender/VideoTrace}"
ENV_ROOT="${VIDEOTRACE_ENV:-/linyuanping/miniconda3/envs/guide2play-qwen35}"
PY="$ENV_ROOT/bin/python"
WAIT_SECONDS=0
SKIP_BROWSER=0
KEEP_WEB=0
DRY_RUN=0
LOG_DIR="$ROOT/outputs/reports"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait) WAIT_SECONDS="${2:?--wait needs a value in seconds}"; shift 2 ;;
    --skip-browser) SKIP_BROWSER=1; shift ;;
    --keep-web) KEEP_WEB=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

export VIDEOTRACE_GPU_WAIT_SECONDS="$WAIT_SECONDS"
export PYTHONPATH="$ROOT/scripts/remote/python_compat:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$ROOT"
mkdir -p "$LOG_DIR"

step() { printf '\n=== [%s] %s\n' "$(date -u +%H:%M:%SZ)" "$1"; }
fail() { printf '\n!!! FAILED at: %s\n' "$1" >&2; exit 1; }
# In dry-run every mutating stage is announced but not executed, so the whole
# plan can be checked on a busy host without consuming a single GPU second.
run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then printf '    (dry-run) would run: %s\n' "$*"; return 0; fi
  "$@"
}

step "0/7 preflight: interpreter, models, source fingerprint"
[[ -x "$PY" ]] || fail "interpreter missing: $PY"
"$PY" - <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, "src")
from videomemo.eval.reproducibility import source_fingerprint
print("source_sha256:", source_fingerprint(Path(".")))
PYEOF

step "1/7 test suite"
export VIDEOTRACE_SKIP_DOCUMENTATION_CONSISTENCY=1
run bash scripts/remote/run_tests.sh || fail "remote test suite"
unset VIDEOTRACE_SKIP_DOCUMENTATION_CONSISTENCY

step "2/7 canonical knowledge pack (Qwen3.5-9B + SigLIP2)"
run bash scripts/remote/run_qwen35_demo.sh || fail "canonical demo"

step "3/7 adapter evaluation and hash-bound admission"
run bash scripts/remote/run_adapter_evaluations.sh || fail "adapter evaluation/admission"

step "4/7 frozen regression suite"
run bash scripts/remote/run_regression_suite.sh || fail "frozen regression suite"
run "$PY" scripts/run_failure_recovery_demo.py >/dev/null || fail "failure recovery demo"

step "5/7 cold/warm runtime profile"
run bash scripts/remote/run_profile_runtime.sh || fail "runtime profile"

if [[ "$SKIP_BROWSER" -eq 1 ]]; then
  step "6/7 browser E2E -- SKIPPED by request"
  echo "NOTE: browser_e2e.json keeps its previous source fingerprint, so"
  echo "      validate_delivery_package.py will still report that check red."
else
  step "6/7 resident web service + browser E2E"
  run bash scripts/remote/start_web_service.sh || fail "web service start"
  run bash scripts/remote/run_browser_e2e.sh "$ROOT/data/raw/cola_review.mp4" || fail "browser E2E"
  if [[ "$KEEP_WEB" -eq 1 ]]; then
    echo "    web service left resident by request (--keep-web)"
  else
    echo "    stopping the web service so it does not hold a card unattended"
    run bash scripts/remote/stop_web_service.sh || echo "    WARNING: stop_web_service.sh failed; check the PID manually"
  fi
fi

step "7/7 rebuild interview artifacts, manifest and re-check delivery on the host"
run "$PY" scripts/analyze_dpo_length_bias.py >/dev/null || fail "length-bias diagnostic"
if [[ -f outputs/reports/dpo_sweep.json ]]; then
  run "$PY" scripts/validate_dpo_sweep.py >/dev/null || fail "DPO sweep validation"
fi
run "$PY" scripts/build_reranker_model_card.py \
  --dataset outputs_train/reranker_dev_5s.jsonl \
  --model outputs/models/neural_reranker.pt \
  --metrics outputs/models/neural_reranker_metrics.json \
  --output outputs/models/neural_reranker_model_card.json >/dev/null || fail "reranker model card"
run "$PY" scripts/validate_interview_package.py \
  --knowledge-pack outputs/iboy_qwen35/cola_review/knowledge_pack.json \
  --checkpoint outputs/models/neural_reranker.pt \
  --metrics outputs/models/neural_reranker_metrics.json \
  --dataset outputs_train/reranker_dev_5s.jsonl \
  --dataset-summary outputs_train/reranker_dev_5s.summary.json \
  --model-card outputs/models/neural_reranker_model_card.json \
  --output outputs/interview_readiness.json >/dev/null || fail "interview package"
run "$PY" scripts/build_artifact_manifest.py >/dev/null || fail "artifact manifest"
run "$PY" scripts/validate_delivery_package.py >/dev/null 2>&1 || true
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '
=== dry-run complete: preconditions checked, nothing was modified.
'
  exit 0
fi
"$PY" - <<'PYEOF'
import json
from pathlib import Path
report = json.loads(Path("outputs/reports/delivery_readiness.json").read_text(encoding="utf-8"))
print(f"delivery: {report['checks_passed']}/{report['checks_total']}")
for failure in report.get("failures", []):
    print("  still red:", failure)
PYEOF

printf '\n=== done. Copy the regenerated artifacts back, then run locally:\n'
printf '    python scripts/build_artifact_manifest.py\n'
printf '    python scripts/validate_delivery_package.py\n'
printf '    python scripts/validate_documentation_consistency.py --strict\n'
