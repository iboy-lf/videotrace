from __future__ import annotations

from pathlib import Path
import sys
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def ensure_src_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    compat = ROOT / "scripts" / "remote" / "python_compat"
    if compat.is_dir() and str(compat) not in sys.path:
        sys.path.insert(0, str(compat))
        sitecustomize = compat / "sitecustomize.py"
        if sitecustomize.exists():
            spec = importlib.util.spec_from_file_location("videotrace_sitecustomize", sitecustomize)
            module = importlib.util.module_from_spec(spec) if spec and spec.loader else None
            if module and spec and spec.loader:
                spec.loader.exec_module(module)
