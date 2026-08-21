#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDEOTRACE_ROOT:-/lavender/VideoTrace}"
ENV_ROOT="${VIDEOTRACE_ENV:-/linyuanping/miniconda3/envs/guide2play-qwen35}"
export PYTHONPATH="$ROOT/scripts/remote/python_compat:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export VIDEOTRACE_REUSE_SAMPLE=1

select_test_python() {
  if [[ -n "${VIDEOTRACE_TEST_PYTHON:-}" ]]; then
    printf '%s\n' "$VIDEOTRACE_TEST_PYTHON"
    return
  fi
  local candidate
  for candidate in \
    "$ENV_ROOT/bin/python" \
    /linyuanping/miniconda3/envs/wyf_vm/bin/python \
    /usr/bin/python; do
    [[ -x "$candidate" ]] || continue
    if "$candidate" - <<'PY' >/dev/null 2>&1
import tempfile
from pathlib import Path
import cv2
import pytest
import sklearn

with tempfile.TemporaryDirectory(prefix="videotrace-test-python-") as directory:
    target = Path(directory) / "probe.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 4, (64, 64))
    usable = writer.isOpened()
    writer.release()
    if not usable:
        raise SystemExit(1)
PY
    then
      printf '%s\n' "$candidate"
      return
    fi
  done
  return 1
}

TEST_PYTHON="$(select_test_python)" || {
  echo "No existing Python interpreter has pytest, sklearn and a usable MP4 encoder." >&2
  exit 1
}

NODE_BIN="${VIDEOTRACE_NODE:-}"
if [[ -z "$NODE_BIN" ]]; then
  NODE_BIN="$(command -v node || true)"
fi
if [[ -z "$NODE_BIN" && -x /usr/local/nvm/versions/node/v18.20.3/bin/node ]]; then
  NODE_BIN=/usr/local/nvm/versions/node/v18.20.3/bin/node
fi

cd "$ROOT"
"$ENV_ROOT/bin/python" -m compileall -q src scripts tests
echo "VideoTrace test Python: $TEST_PYTHON"
"$TEST_PYTHON" -m pytest -q
if [[ -n "$NODE_BIN" ]]; then
  "$NODE_BIN" --check src/videomemo/web/static/app.js
  "$NODE_BIN" --check src/videomemo/web/static/playback.js
  "$NODE_BIN" --check src/videomemo/web/static/technical.js
  "$NODE_BIN" --check src/videomemo/web/static/job_status.js
  "$NODE_BIN" tests/js/playback.test.cjs
  "$NODE_BIN" tests/js/technical.test.cjs
  "$NODE_BIN" tests/js/job_status.test.cjs
else
  echo "Node.js unavailable; JS checks were not run." >&2
  exit 1
fi
