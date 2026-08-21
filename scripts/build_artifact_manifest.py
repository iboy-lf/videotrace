from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_path

ensure_src_path()

from videomemo.eval.artifact_manifest import write_artifact_manifest


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-build-artifact-manifest")
    parser.add_argument("--output", default="outputs/reports/artifact_manifest.json")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    manifest = write_artifact_manifest(ROOT, args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not manifest["complete"] and not args.allow_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
