from __future__ import annotations

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from videomemo.web import run_server


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="videomemo-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7860, type=int)
    parser.add_argument("--latest-pack", default=None, help="knowledge_pack.json to show on the initial page")
    parser.add_argument(
        "--config",
        default=None,
        help="runtime config; defaults to configs/iboy_qwen35.yaml when available",
    )
    args = parser.parse_args()
    if args.latest_pack:
        os.environ["VIDEOTRACE_LATEST_PACK"] = str(Path(args.latest_pack).resolve())
    config = Path(args.config).resolve() if args.config else ROOT / "configs" / "iboy_qwen35.yaml"
    if config.exists():
        os.environ["VIDEOTRACE_CONFIG"] = str(config)
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
