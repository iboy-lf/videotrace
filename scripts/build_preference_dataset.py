from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_path

ensure_src_path()

from videomemo.training.preference_data import build_grounded_preference_dataset


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-build-grounded-preference")
    parser.add_argument("--sft-dataset", default="data/sft/grounded_qa.jsonl")
    parser.add_argument("--annotations", default="data/preference/preference_annotations.json")
    parser.add_argument("--output", default="data/preference/grounded_dpo.jsonl")
    args = parser.parse_args()
    summary = build_grounded_preference_dataset(
        _rooted(args.sft_dataset),
        _rooted(args.annotations),
        _rooted(args.output),
        project_root=ROOT,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _rooted(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
