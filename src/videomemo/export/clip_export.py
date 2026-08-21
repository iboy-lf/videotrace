from __future__ import annotations

from pathlib import Path

from ..models import KnowledgePack


def export_clip_files(pack: KnowledgePack, output_dir: str) -> list[dict]:
    """Attach source-video windows instead of browser-incompatible transcodes."""
    del output_dir
    source = str(Path(pack.video_path).resolve())
    duration = float(pack.duration_sec)
    updated: list[dict] = []
    for clip in pack.clips:
        start = max(0.0, float(clip["start_sec"]))
        end = min(duration, float(clip["end_sec"]))
        if end <= start:
            continue
        item = dict(clip)
        item.update(
            {
                "start_sec": start,
                "end_sec": end,
                "file": source,
                "playback_mode": "source_video_window",
            }
        )
        updated.append(item)
    pack.clips = updated
    return updated
