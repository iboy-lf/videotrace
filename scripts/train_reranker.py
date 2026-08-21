from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from videomemo.eval.reproducibility import source_fingerprint
from videomemo.reranker import train_reranker
from videomemo.reranker.model_card import build_reranker_model_card


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-train-reranker")
    parser.add_argument("dataset", nargs="?", default="outputs_train/reranker_dev_5s.jsonl")
    parser.add_argument("--model", default="outputs/models/neural_reranker.pt")
    parser.add_argument("--metrics", default="outputs/models/neural_reranker_metrics.json")
    parser.add_argument("--model-card", default="outputs/models/neural_reranker_model_card.json")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-test-split",
        action="store_true",
        help="explicit diagnostic override; do not use for interview model fitting",
    )
    args = parser.parse_args()

    result = train_reranker(
        args.dataset,
        args.model,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        device=args.device,
        allow_test_split=args.allow_test_split,
    )
    payload = result.dump()
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    card = build_reranker_model_card(
        args.dataset,
        args.model,
        args.metrics,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        source_sha256=source_fingerprint(ROOT),
    )
    card_path = Path(args.model_card)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
