from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from _bootstrap import ensure_src_path

ensure_src_path()


TIMESTAMP_RE = re.compile(r"timestamp=([0-9]+(?:\.[0-9]+)?-[0-9]+(?:\.[0-9]+)?)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-validate-outputs")
    parser.add_argument("path", help="knowledge_pack.json")
    parser.add_argument("--video", default=None, help="override the video path for a knowledge pack")
    args = parser.parse_args()

    path = Path(args.path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    report = _validate_pack(payload, Path(args.video).resolve() if args.video else None)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


def _validate_pack(payload: dict, video_override: Path | None) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    duration = _number(payload.get("duration_sec"), 0.0)
    video_path = video_override or Path(str(payload.get("video_path", "")))
    if not video_path.exists():
        warnings.append(f"video not available for local validation: {video_path}")
    else:
        video_hash = _sha256(video_path)
    segments = list(payload.get("segments", []))
    timeline = list(payload.get("timeline", []))
    clips = list(payload.get("clips", []))
    if not segments:
        errors.append("knowledge pack has no segments")
    for label, items in (("segments", segments), ("timeline", timeline), ("clips", clips)):
        for index, item in enumerate(items):
            start = _number(item.get("start_sec"), -1.0)
            end = _number(item.get("end_sec"), -1.0)
            if start < 0 or end <= start:
                errors.append(f"{label}[{index}] has invalid bounds {start}-{end}")
            if duration > 0 and end > duration + 0.5:
                errors.append(f"{label}[{index}] ends after video duration: {end}>{duration}")
    for index, clip in enumerate(clips):
        if clip.get("playback_mode") != "source_video_window":
            errors.append(f"clips[{index}] is not bound to the source-video playback path")

    answer = str(payload.get("answer", ""))
    if "问题：" not in answer or "结论：" not in answer:
        errors.append("answer is missing canonical question/conclusion structure")
    agent_run = payload.get("metadata", {}).get("agent_run", {})
    evidence_tags = set(str(value) for value in agent_run.get("context", {}).get("evidence_tags", []))
    refs = TIMESTAMP_RE.findall(answer)
    unbound = [ref for ref in refs if f"timestamp={ref}" not in evidence_tags]
    if refs and unbound:
        errors.append(f"answer contains unbound timestamps: {unbound}")
    if not refs:
        errors.append("answer contains no timestamp references")
    performance = payload.get("metadata", {}).get("performance", {})
    stages = performance.get("stage_seconds", {})
    if _number(stages.get("total"), 0.0) <= 0:
        errors.append("performance.stage_seconds.total is not positive")
    report = {
        "valid": not errors,
        "path": str(video_path),
        "duration_sec": duration,
        "num_segments": len(segments),
        "num_timeline_items": len(timeline),
        "num_clips": len(clips),
        "timestamp_refs": refs,
        "unbound_timestamp_refs": unbound,
        "warnings": warnings,
        "errors": errors,
    }
    if video_path.exists():
        report["video_sha256"] = video_hash
    return report


def _number(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
