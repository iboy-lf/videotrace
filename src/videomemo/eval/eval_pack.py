from __future__ import annotations

from pathlib import Path
import json

from ..config import VideoMemoConfig
from ..pipeline import VideoMemoPipeline
from .metrics import evaluate_pack


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(prog='videomemo-eval')
    parser.add_argument('video')
    parser.add_argument('--query', default='Summarize the video and give evidence.')
    parser.add_argument('--output', default='outputs_eval')
    args = parser.parse_args()

    cfg = VideoMemoConfig(output_dir=args.output)
    pack = VideoMemoPipeline(cfg).run(args.video, args.query)
    result = evaluate_pack(pack, args.query.split())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'eval.json').write_text(json.dumps(result.dump(), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result.dump(), ensure_ascii=False))


if __name__ == '__main__':
    main()
