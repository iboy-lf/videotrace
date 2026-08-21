from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_agent_runtime_import_does_not_require_pipeline_dependencies() -> None:
    env = dict(os.environ)
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    script = (
        "import sys; "
        "sys.modules['sklearn'] = None; "
        "from videomemo.agent import AgentToolRegistry; "
        "assert AgentToolRegistry.__name__ == 'AgentToolRegistry'"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

