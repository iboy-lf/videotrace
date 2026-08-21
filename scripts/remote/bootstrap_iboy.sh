#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDEOTRACE_ROOT:-/lavender/VideoTrace}"
ENV_ROOT="${VIDEOTRACE_ENV:-/linyuanping/miniconda3/envs/guide2play-qwen35}"
PYTHON="$ENV_ROOT/bin/python"
export PYTHONPATH="$ROOT/scripts/remote/python_compat:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

test -x "$PYTHON"
test -f "$ROOT/pyproject.toml"
test -f /lavender/models/Qwen3.5-9B/config.json
test -f /lavender/models/siglip2-large-patch16-256/config.json

"$PYTHON" -m pip install --no-deps --no-build-isolation -e "$ROOT"

"$PYTHON" - <<'PY'
import cv2
import sklearn
import torch
import transformers
import videomemo

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("opencv", cv2.__version__)
print("sklearn", sklearn.__version__)
print("videomemo", videomemo.__file__)
PY
