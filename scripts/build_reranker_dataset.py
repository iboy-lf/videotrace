from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from videomemo.config import VideoMemoConfig
from videomemo.index.simple_index import build_segment_index, rank_segments
from videomemo.ingest.asr import build_asr_backend, enrich_segments_with_asr
from videomemo.ingest.video import sample_segment_text, split_video_into_segments
from videomemo.pipeline import VideoMemoPipeline
from videomemo.reranker import build_reranker_features
from videomemo.reranker.supervision import load_reranker_supervision
from videomemo.scorer import SegmentScorer
from videomemo.vlm import build_segment_analyzer, build_vlm_embedder
from videomemo.vlm.scoring import attach_vlm_scores


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-build-reranker-dataset")
    parser.add_argument(
        "--supervision",
        default="data/supervision/reranker_annotations.json",
        help="manually labeled development supervision",
    )
    parser.add_argument("--config", default="configs/iboy_qwen35.yaml")
    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=5,
        help="supervision window in seconds; the canonical task-local dataset uses 5-second windows",
    )
    parser.add_argument("--output", default="outputs_train/reranker_dev_5s.jsonl")
    parser.add_argument(
        "--split",
        default="dev",
        choices=["all", "dev"],
        help="training supervision is development-only",
    )
    args = parser.parse_args()

    cfg = VideoMemoConfig.load(args.config)
    if args.segment_seconds <= 0:
        raise SystemExit("--segment-seconds must be positive")
    cfg.segment_seconds = args.segment_seconds
    cases = load_reranker_supervision(args.supervision)
    if args.split != "all":
        cases = [case for case in cases if case.split == args.split]
    by_video: dict[str, list] = {}
    for case in cases:
        by_video.setdefault(case.video_path, []).append(case)

    analyzer = build_segment_analyzer(
        cfg.segment_understanding_backend,
        cfg.segment_understanding_base_url or cfg.llm_base_url,
        cfg.segment_understanding_model or cfg.llm_model,
        cfg.segment_understanding_api_key or cfg.llm_api_key,
        cfg.segment_understanding_cache_dir,
        cfg.segment_understanding_frames,
        cfg.segment_understanding_timeout_sec,
        cfg.segment_understanding_fail_open,
        device=cfg.segment_understanding_device,
        dtype=cfg.segment_understanding_dtype,
        max_new_tokens=cfg.segment_understanding_max_new_tokens,
    )
    asr_backend = build_asr_backend(
        cfg.asr_backend,
        model=cfg.asr_model,
        device=cfg.asr_device,
        compute_type=cfg.asr_compute_type,
        language=cfg.asr_language,
    )
    embedder = build_vlm_embedder(
        cfg.vlm_backend,
        cfg.vlm_model_name,
        cfg.vlm_cache_dir,
        cfg.vlm_num_frames,
        cfg.vlm_device or None,
    )
    scorer = SegmentScorer()
    rows = []
    reports = []

    for video_path, video_cases in by_video.items():
        asset, segments = split_video_into_segments(video_path, cfg.segment_seconds, use_scene_cut=cfg.use_scene_cut)
        segments = sample_segment_text(video_path, segments)
        asr_report = enrich_segments_with_asr(video_path, segments, asr_backend, cfg.asr_cache_dir, cfg.asr_fail_open)
        understanding_report = analyzer.enrich(video_path, segments)
        for case in video_cases:
            attach_vlm_scores(video_path, case.query, segments, embedder)
            index = build_segment_index(segments)
            retrieved = rank_segments(case.query, index, max(1, len(segments)))
            retrieval_by_id = {item["segment_id"]: item for item in retrieved}
            segments = scorer.rank(case.query, segments)
            for segment in segments:
                segment.retrieval_score = float(retrieval_by_id.get(segment.segment_id, {}).get("score", 0.0))
                segment.scorer_score = float(segment.score)
            VideoMemoPipeline._attach_rank_scores(segments)
            for segment in segments:
                segment.score = (
                    cfg.retrieval_weight * segment.retrieval_rank_score
                    + cfg.scorer_weight * segment.scorer_rank_score
                    + cfg.vlm_weight * segment.vlm_rank_score
                )
                label = _label_segment(segment.start_sec, segment.end_sec, case.gold_spans)
                rows.append(
                    {
                        "group_id": case.case_id,
                        "case_id": case.case_id,
                        "video_id": case.video_id,
                        "split": case.split,
                        "query": case.query,
                        "segment_id": segment.segment_id,
                        "start_sec": segment.start_sec,
                        "end_sec": segment.end_sec,
                        "label": label,
                        "features": build_reranker_features(case.query, segment, asset.duration_sec),
                    }
                )
        reports.append(
            {
                "video_path": video_path,
                "num_cases": len(video_cases),
                "num_segments": len(segments),
                "asr": asr_report,
                "segment_understanding": {
                    "backend": understanding_report.get("backend"),
                    "num_generated": understanding_report.get("num_generated", 0),
                    "num_cache_hits": understanding_report.get("num_cache_hits", 0),
                    "num_fallbacks": understanding_report.get("num_fallbacks", 0),
                },
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "output": str(output),
        "num_rows": len(rows),
        "num_groups": len({row["group_id"] for row in rows}),
        "num_positive": sum(row["label"] > 0.5 for row in rows),
        "split": args.split,
        "reports": reports,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _label_segment(start_sec: float, end_sec: float, gold_spans: list[dict]) -> float:
    if not gold_spans:
        return 0.0
    best = max((_temporal_iou(start_sec, end_sec, span) for span in gold_spans), default=0.0)
    return 1.0 if best > 0.0 else 0.0


def _temporal_iou(start_sec: float, end_sec: float, span: dict) -> float:
    gold_start = float(span.get("start_sec", 0.0))
    gold_end = float(span.get("end_sec", gold_start))
    intersection = max(0.0, min(end_sec, gold_end) - max(start_sec, gold_start))
    union = max(end_sec, gold_end) - min(start_sec, gold_start)
    return intersection / union if union > 0.0 else 0.0


if __name__ == "__main__":
    main()
