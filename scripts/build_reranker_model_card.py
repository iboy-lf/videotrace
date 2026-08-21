from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_path

ensure_src_path()

from videomemo.eval.reproducibility import source_fingerprint
from videomemo.reranker.model_card import build_reranker_model_card


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-build-reranker-model-card")
    parser.add_argument("--dataset", default="outputs_train/reranker_dev_5s.jsonl")
    parser.add_argument("--model", default="outputs/models/neural_reranker.pt")
    parser.add_argument("--metrics", default="outputs/models/neural_reranker_metrics.json")
    parser.add_argument("--output", default="outputs/models/neural_reranker_model_card.json")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=32)
    args = parser.parse_args()

    card = build_reranker_model_card(
        args.dataset,
        args.model,
        args.metrics,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        source_sha256=source_fingerprint(ROOT),
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(card, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
