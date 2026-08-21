from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
from _bootstrap import ensure_src_path

ensure_src_path()

from videomemo.config import VideoMemoConfig
from videomemo.ingest.video import sample_segment_text, split_video_into_segments
from videomemo.memory import PersistentMemoryStore, VideoMemoryStore


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="videomemo-build-memory")
    parser.add_argument("video")
    parser.add_argument("--output", default=str(ROOT / "outputs_memory" / "memories.jsonl"))
    args = parser.parse_args()

    cfg = VideoMemoConfig()
    _, segments = split_video_into_segments(args.video, cfg.segment_seconds, use_scene_cut=cfg.use_scene_cut)
    segments = sample_segment_text(args.video, segments)
    video_id = Path(args.video).stem
    store = VideoMemoryStore.from_segments(segments, video_id=video_id, video_path=args.video)
    count = PersistentMemoryStore(args.output).upsert(store)
    print(json.dumps({"memory_path": args.output, "records_written": count}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
