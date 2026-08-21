from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_path

ensure_src_path()

from videomemo.training.sft_data import load_sft_records, summarize_sft_dataset


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-validate-sft-dataset")
    parser.add_argument("--dataset", default="data/sft/grounded_qa.jsonl")
    parser.add_argument("--summary", default="data/sft/grounded_qa.summary.json")
    args = parser.parse_args()
    dataset = _rooted(args.dataset)
    summary_path = _rooted(args.summary)
    summary = summarize_sft_dataset(load_sft_records(dataset), dataset, project_root=ROOT)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["validation"]["valid"] else 1)


def _rooted(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
