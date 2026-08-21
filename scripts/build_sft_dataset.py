from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from videomemo.training.sft_data import build_grounded_sft_dataset


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-build-grounded-sft")
    parser.add_argument("--annotations", default="data/supervision/reranker_annotations.json")
    parser.add_argument("--cola-pack", default="outputs/cola_review_qwen35/knowledge_pack.json")
    parser.add_argument("--output", default="data/sft/grounded_qa.jsonl")
    args = parser.parse_args()
    cola_pack = ROOT / args.cola_pack
    if not cola_pack.exists() and args.cola_pack == "outputs/cola_review_qwen35/knowledge_pack.json":
        remote_pack = ROOT / "outputs" / "iboy_qwen35" / "cola_review" / "knowledge_pack.json"
        if remote_pack.exists():
            cola_pack = remote_pack
    summary = build_grounded_sft_dataset(
        ROOT / args.annotations,
        ROOT / args.output,
        cola_pack_path=cola_pack,
        project_root=ROOT,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
