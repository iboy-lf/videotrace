from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PROBE = r"""
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
"""


def _candidate_pythons() -> list[Path]:
    configured = os.environ.get("VIDEOTRACE_TEST_PYTHON", "").strip()
    env_root = Path(
        os.environ.get(
            "VIDEOTRACE_ENV",
            "/linyuanping/miniconda3/envs/guide2play-qwen35",
        )
    )
    raw = [
        Path(configured) if configured else None,
        Path(sys.executable),
        env_root / "bin" / "python",
        Path("/linyuanping/miniconda3/envs/wyf_vm/bin/python"),
        Path("/usr/bin/python"),
    ]
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in raw:
        if candidate is None:
            continue
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


def _usable(candidate: Path, env: dict[str, str]) -> bool:
    if not candidate.is_file():
        return False
    try:
        result = subprocess.run(
            [str(candidate), "-c", PROBE],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def main() -> int:
    env = dict(os.environ)
    paths = [str(ROOT / "scripts" / "remote" / "python_compat"), str(ROOT / "src")]
    existing = env.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    for candidate in _candidate_pythons():
        if not _usable(candidate, env):
            continue
        command = [str(candidate), "-m", "pytest", "-q", *sys.argv[1:]]
        if Path(sys.executable).resolve() == candidate.resolve():
            return subprocess.call(command, cwd=ROOT, env=env)
        os.execve(str(candidate), command, env)
    print(
        "No existing Python interpreter has pytest, sklearn and a usable MP4 encoder.",
        file=sys.stderr,
    )
    return 1


raise SystemExit(main())
