from __future__ import annotations

import argparse
from pathlib import Path

from .config import VideoMemoConfig
from .pipeline import VideoMemoPipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog='videomemo')
    parser.add_argument('video', help='path to a video file')
    parser.add_argument('--query', default='总结这个视频，并给出带时间戳的证据。')
    parser.add_argument('--config', default=None)
    parser.add_argument('--llm-backend', default=None, choices=['template', 'openai_compatible', 'qwen_vl_api', 'qwen35_local'])
    parser.add_argument('--llm-base-url', default=None)
    parser.add_argument('--llm-model', default=None)
    parser.add_argument('--llm-api-key', default=None)
    parser.add_argument('--segment-understanding-backend', default=None, choices=['baseline', 'qwen_vl_api', 'qwen35_local', 'none'])
    parser.add_argument('--segment-understanding-base-url', default=None)
    parser.add_argument('--segment-understanding-model', default=None)
    parser.add_argument('--vlm-backend', default=None, choices=['baseline', 'clip', 'siglip', 'none'])
    parser.add_argument('--vlm-model', default=None)
    parser.add_argument('--vlm-device', default=None)
    args = parser.parse_args()

    cfg = VideoMemoConfig.load(args.config)
    if args.llm_backend:
        cfg.llm_backend = args.llm_backend
    if args.llm_base_url:
        cfg.llm_base_url = args.llm_base_url
    if args.llm_model:
        cfg.llm_model = args.llm_model
    if args.llm_api_key:
        cfg.llm_api_key = args.llm_api_key
    if args.segment_understanding_backend:
        cfg.segment_understanding_backend = args.segment_understanding_backend
    if args.segment_understanding_base_url:
        cfg.segment_understanding_base_url = args.segment_understanding_base_url
    if args.segment_understanding_model:
        cfg.segment_understanding_model = args.segment_understanding_model
    if args.vlm_backend:
        cfg.vlm_backend = args.vlm_backend
    if args.vlm_model:
        cfg.vlm_model_name = args.vlm_model
    if args.vlm_device:
        cfg.vlm_device = args.vlm_device
    pipeline = VideoMemoPipeline(cfg)
    out = pipeline.run_and_export(args.video, query=args.query)
    print(out)


if __name__ == '__main__':
    main()
