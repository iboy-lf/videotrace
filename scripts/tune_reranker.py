from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from videomemo.eval.reproducibility import source_fingerprint
from videomemo.reranker import train_reranker
from videomemo.reranker.model_card import build_reranker_model_card


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-tune-reranker")
    parser.add_argument("dataset", nargs="?", default="outputs_train/reranker_dev_5s.jsonl")
    parser.add_argument("--model", default="outputs/models/neural_reranker.pt")
    parser.add_argument("--metrics", default="outputs/models/neural_reranker_metrics.json")
    parser.add_argument("--model-card", default="outputs/models/neural_reranker_model_card.json")
    parser.add_argument("--report", default="outputs/models/neural_reranker_tuning.json")
    parser.add_argument("--candidate-dir", default="outputs/models/reranker_tuning")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=[5e-4, 1e-3, 2e-3, 5e-3],
    )
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    candidate_dir = Path(args.candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for hidden_dim in args.hidden_dims:
        for learning_rate in args.learning_rates:
            tag = f"h{hidden_dim}_lr{learning_rate:g}"
            checkpoint = candidate_dir / f"{tag}.pt"
            result = train_reranker(
                args.dataset,
                str(checkpoint),
                epochs=args.epochs,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                eval_fraction=args.eval_fraction,
                seed=args.seed,
                device=args.device,
            )
            payload = result.dump()
            payload.update({"hidden_dim": hidden_dim, "learning_rate": learning_rate})
            candidates.append(payload)

    selected = max(
        candidates,
        key=lambda item: (
            _metric(item.get("blended_pairwise_accuracy")),
            _metric(item.get("pairwise_accuracy")),
            -_metric(item.get("eval_loss"), default=1e9),
            -int(item["hidden_dim"]),
        ),
    )
    target = Path(args.model)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected["checkpoint_path"], target)
    metrics = dict(selected)
    metrics["checkpoint_path"] = str(target)
    metrics["selection_policy"] = (
        "maximize blended held-out pairwise accuracy, then neural pairwise accuracy, "
        "then minimize eval loss and model size"
    )
    metrics["search_space"] = {
        "hidden_dims": args.hidden_dims,
        "learning_rates": args.learning_rates,
        "epochs": args.epochs,
        "eval_fraction": args.eval_fraction,
        "seed": args.seed,
    }
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "selected": metrics,
        "candidates": sorted(
            candidates,
            key=lambda item: (
                -_metric(item.get("blended_pairwise_accuracy")),
                -_metric(item.get("pairwise_accuracy")),
                _metric(item.get("eval_loss"), default=1e9),
            ),
        ),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    card = build_reranker_model_card(
        args.dataset,
        str(target),
        str(metrics_path),
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        hidden_dim=int(selected["hidden_dim"]),
        source_sha256=source_fingerprint(ROOT),
    )
    card["tuning"] = {
        "report_path": str(report_path),
        "num_candidates": len(candidates),
        "selection_policy": metrics["selection_policy"],
        "search_space": metrics["search_space"],
    }
    card_path = Path(args.model_card)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _metric(value: object, default: float = -1.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
