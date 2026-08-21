from __future__ import annotations

from pathlib import Path
import json
from _bootstrap import ensure_src_path

ensure_src_path()

from videomemo.config import VideoMemoConfig
from videomemo.pipeline import VideoMemoPipeline
from videomemo.scorer import write_scorer_dataset
from videomemo.planner.train_data import write_planner_dataset, write_ranker_dataset


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(prog='videomemo-train-data')
    parser.add_argument('video')
    parser.add_argument('--query', default='What happens in the video?')
    parser.add_argument('--output', default='outputs_train')
    args = parser.parse_args()

    cfg = VideoMemoConfig(output_dir=args.output)
    pack = VideoMemoPipeline(cfg).run(args.video, args.query)
    ranked = pack.metadata.get('ranked_segments', [])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    planner_path = write_planner_dataset(args.query, ranked, str(out / 'planner.json'))
    ranker_path = write_ranker_dataset(args.query, ranked, str(out / 'ranker.json'))
    scorer_path = write_scorer_dataset(args.query, pack.segments, str(out / 'scorer.json'))
    print(json.dumps({'planner': planner_path, 'ranker': ranker_path, 'scorer': scorer_path}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
