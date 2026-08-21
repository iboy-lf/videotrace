from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_path

ensure_src_path()

from videomemo.eval.delivery_readiness import validate_delivery_package


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-validate-delivery-package")
    parser.add_argument("--manifest", default="outputs/reports/artifact_manifest.json")
    parser.add_argument("--output", default="outputs/reports/delivery_readiness.json")
    args = parser.parse_args()
    report = validate_delivery_package(ROOT, args.manifest)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
