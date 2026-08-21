from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_path

ensure_src_path()

from videomemo.eval.interview_readiness import validate_interview_package


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-validate-interview-package")
    parser.add_argument(
        "--knowledge-pack",
        default="outputs/iboy_qwen35/cola_review/knowledge_pack.json",
    )
    parser.add_argument("--checkpoint", default="outputs/models/neural_reranker.pt")
    parser.add_argument("--metrics", default="outputs/models/neural_reranker_metrics.json")
    parser.add_argument("--dataset", default="outputs_train/reranker_dev_5s.jsonl")
    parser.add_argument(
        "--dataset-summary",
        default="outputs_train/reranker_dev_5s.summary.json",
    )
    parser.add_argument(
        "--model-card",
        default="outputs/models/neural_reranker_model_card.json",
    )
    parser.add_argument("--output", default="outputs/interview_readiness.json")
    args = parser.parse_args()

    report = validate_interview_package(
        ROOT,
        Path(args.knowledge_pack),
        Path(args.checkpoint),
        Path(args.metrics),
        Path(args.dataset),
        Path(args.dataset_summary),
        Path(args.model_card),
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
