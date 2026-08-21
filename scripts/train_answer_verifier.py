from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_path

ensure_src_path()

from videomemo.training.answer_verifier import (
    train_answer_verifier,
    write_answer_verifier_dataset,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-train-answer-verifier")
    parser.add_argument("--preference", default="data/preference/grounded_dpo.jsonl")
    parser.add_argument("--dataset", default="data/verifier/answer_verifier.jsonl")
    parser.add_argument("--summary", default="data/verifier/answer_verifier.summary.json")
    parser.add_argument("--checkpoint", default="outputs/models/answer_verifier.pkl")
    parser.add_argument("--metrics", default="outputs/models/answer_verifier_metrics.json")
    parser.add_argument("--model-card", default="outputs/models/answer_verifier_model_card.json")
    args = parser.parse_args()

    summary = write_answer_verifier_dataset(
        _rooted(args.preference),
        _rooted(args.dataset),
        _rooted(args.summary),
        ROOT,
    )
    result = train_answer_verifier(
        _rooted(args.dataset),
        _rooted(args.checkpoint),
        _rooted(args.metrics),
        _rooted(args.model_card),
        ROOT,
    )
    print(json.dumps({"dataset": summary, "training": result.dump()}, ensure_ascii=False, indent=2))


def _rooted(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
