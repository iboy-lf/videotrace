from __future__ import annotations

from pathlib import Path
import json
from _bootstrap import ensure_src_path

ensure_src_path()

from videomemo.config import VideoMemoConfig
from videomemo.pipeline import VideoMemoPipeline
from videomemo.eval import evaluate_harness_run, evaluate_pack


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(prog='videomemo-eval-script')
    parser.add_argument('video')
    parser.add_argument('--query', default='总结这个视频，并给出带时间戳的证据。')
    parser.add_argument('--output', default='outputs_eval')
    args = parser.parse_args()

    cfg = VideoMemoConfig(output_dir=args.output)
    pack = VideoMemoPipeline(cfg).run(args.video, args.query)
    result = evaluate_pack(pack, args.query.split())
    harness_result = evaluate_harness_run(pack.metadata.get("agent_run", {}))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "pack": result.dump(),
        "harness": harness_result.dump(),
    }
    (out / 'eval.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
