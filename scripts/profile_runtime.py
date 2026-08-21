from __future__ import annotations

"""Capture stable cold/warm/task-local runtime evidence without benchmark sprawl."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from videomemo.config import VideoMemoConfig
from videomemo.eval.reproducibility import file_sha256, runtime_environment, source_fingerprint
from videomemo.pipeline import VideoMemoPipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-profile-runtime")
    parser.add_argument("--video", default="data/raw/cola_review.mp4")
    parser.add_argument("--query", default="这个视频的整体流程是什么？请概括开场、分国家试喝和最后盲测三个阶段并给出时间戳。")
    parser.add_argument("--config", default="configs/iboy_qwen35.yaml")
    parser.add_argument("--output", default="outputs/reports/performance_report.json")
    args = parser.parse_args()

    video = _rooted(args.video)
    config = VideoMemoConfig.load(str(_rooted(args.config)))
    # The first pipeline captures model-load/cold-cache cost; the second uses
    # resident runtimes and persisted segment/SigLIP features.
    cold_started = time.perf_counter()
    cold_pipeline = VideoMemoPipeline(config)
    cold_pack = cold_pipeline.run(str(video), query=args.query)
    cold_elapsed = time.perf_counter() - cold_started
    warm_started = time.perf_counter()
    warm_pack = cold_pipeline.run(str(video), query=args.query)
    warm_elapsed = time.perf_counter() - warm_started

    report = {
        "schema_version": "videotrace-performance-report-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "single real cola task profile; no public benchmark or ranking claim",
        "video_path": str(video),
        "video_sha256": file_sha256(video),
        "source_sha256": source_fingerprint(ROOT),
        "runtime": runtime_environment(),
        "model_residency": {
            "cold_pipeline_construction_and_run_seconds": round(cold_elapsed, 3),
            "warm_cache_hit_run_seconds": round(warm_elapsed, 3),
            "speedup": round(cold_elapsed / max(warm_elapsed, 1e-6), 3),
            "same_pipeline_instance": True,
            "gpu_requests": "serial queue / one resident pipeline per config",
        },
        "cold_pack_performance": cold_pack.metadata.get("performance", {}),
        "warm_pack_performance": warm_pack.metadata.get("performance", {}),
        "cache_reuse": {
            "cold": cold_pack.metadata.get("performance", {}).get("cache", {}),
            "warm": warm_pack.metadata.get("performance", {}).get("cache", {}),
            "siglip_index_reuse": bool(
                warm_pack.metadata.get("performance", {}).get("cache", {}).get("vlm", {}).get("hits", 0)
            ),
        },
        "precision": {
            "bf16": "used by the remote Qwen3.5/SigLIP product path",
            "int8_or_int4": "not enabled; remote bitsandbytes is CPU-only and failed the explicit 4bit preflight",
            "correctness_check": {
                "cold_verified": bool(cold_pack.metadata.get("agent_run", {}).get("verified")),
                "warm_verified": bool(warm_pack.metadata.get("agent_run", {}).get("verified")),
                "answer_timestamp_binding_preserved": not bool(
                    warm_pack.metadata.get("agent_run", {}).get("verification", {}).get("unmatched_timestamp_refs")
                ),
            },
        },
        "training_profile_reference": {
            "source": "outputs/models/qwen35_sft_metrics.json",
            "note": "one-step real LoRA training profile, not a serving benchmark",
        },
    }
    output = _rooted(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _rooted(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
