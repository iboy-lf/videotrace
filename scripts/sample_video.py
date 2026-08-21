from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess

import cv2
import numpy as np


SCENES = [
    {
        "start_sec": 0.0,
        "end_sec": 20.0,
        "title": "VideoTrace: long-video multimodal agent",
        "subtitle": "Turn one MP4 into an evidence-backed knowledge pack",
        "detail": "timeline + grounded QA + replayable clips",
    },
    {
        "start_sec": 20.0,
        "end_sec": 40.0,
        "title": "Multimodal retrieval",
        "subtitle": "Find timestamped evidence with text and keyframes",
        "detail": "sparse retrieval + SigLIP + score fusion",
    },
    {
        "start_sec": 40.0,
        "end_sec": 60.0,
        "title": "Agent planning and evidence verification",
        "subtitle": "retrieve -> assess -> synthesize -> verify",
        "detail": "tool trace + context budget + grounded report",
    },
    {
        "start_sec": 60.0,
        "end_sec": 80.0,
        "title": "Trainable modules and evaluation",
        "subtitle": "export ranker data and train an ML scorer",
        "detail": "training signals + future VLM reranker",
    },
]


def make_demo_video(path: str, duration_sec: int = 80, fps: int = 10, size: tuple[int, int] = (640, 360)) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fallback = out.parent / "sample.mp4"
    if os.environ.get("VIDEOTRACE_REUSE_SAMPLE") == "1" and fallback.exists() and fallback.resolve() != out.resolve():
        shutil.copyfile(fallback, out)
        return str(out)
    temp = out.with_name(f"{out.stem}.writing{out.suffix}")
    writer = None
    for codec in ('mp4v', 'avc1', 'H264', 'MJPG'):
        candidate = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*codec), fps, size)
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()
    if writer is None:
        if _make_demo_video_with_ffmpeg(temp, duration_sec=duration_sec, fps=fps, size=size):
            temp.replace(out)
            return str(out)
        if fallback.exists() and fallback.resolve() != out.resolve():
            shutil.copyfile(fallback, out)
            return str(out)
        raise RuntimeError("No usable OpenCV video encoder was found")
    total_frames = duration_sec * fps
    for i in range(total_frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        scene = min(len(SCENES) - 1, i // (fps * 20))
        scene_info = SCENES[scene]
        palettes = [(40, 80, 120), (130, 60, 80), (70, 140, 70), (150, 130, 40)]
        base = palettes[scene]
        color = tuple(min(255, base[c] + (i * (3 + c)) % 50) for c in range(3))
        frame[:, :] = color
        cv2.putText(frame, f'VideoTrace Demo | scene={scene} | {i//fps:02d}s', (32, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.putText(frame, scene_info['title'], (32, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.putText(frame, scene_info['subtitle'], (32, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2)
        cv2.putText(frame, scene_info['detail'], (32, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1)
        progress = int((i % (fps * 20)) / max(1, fps * 20 - 1) * (size[0] - 64))
        cv2.rectangle(frame, (32, 300), (32 + progress, 310), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    probe = cv2.VideoCapture(str(temp))
    valid = probe.isOpened() and int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
    probe.release()
    if not valid:
        temp.unlink(missing_ok=True)
        if fallback.exists() and fallback.resolve() != out.resolve():
            shutil.copyfile(fallback, out)
            return str(out)
        raise RuntimeError("Demo video encoder produced an unreadable file")
    temp.replace(out)
    return str(out)


def _make_demo_video_with_ffmpeg(
    output: Path,
    duration_sec: int,
    fps: int,
    size: tuple[int, int],
) -> bool:
    """Portable fallback for CI images whose OpenCV lacks an encoder.

    Some headless Linux builds can decode MP4 through OpenCV but expose no
    usable ``VideoWriter`` codec.  Reusing an already-installed ffmpeg binary
    keeps the smoke test hermetic without installing packages or writing
    outside the requested output directory.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        explicit = str(os.environ.get("VIDEOTRACE_FFMPEG", "")).strip()
        if explicit and Path(explicit).is_file():
            ffmpeg = explicit
    if not ffmpeg:
        for candidate in (
            Path("/linyuanping/miniconda3/envs/lhvln/bin/ffmpeg"),
            Path("/linyuanping/miniconda3/envs/fozo/bin/ffmpeg"),
            Path("/linyuanping/miniconda3/envs/zerosiam/bin/ffmpeg"),
        ):
            if candidate.is_file():
                ffmpeg = str(candidate)
                break
    if not ffmpeg:
        return False
    output.unlink(missing_ok=True)
    width, height = size
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={width}x{height}:rate={max(1, int(fps))}",
        "-t",
        str(max(1, int(duration_sec))),
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=max(30, int(duration_sec) * 4),
        )
    except (OSError, subprocess.SubprocessError):
        output.unlink(missing_ok=True)
        return False
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size <= 0:
        output.unlink(missing_ok=True)
        return False
    probe = cv2.VideoCapture(str(output))
    valid = probe.isOpened() and int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
    probe.release()
    if not valid:
        output.unlink(missing_ok=True)
    return valid


if __name__ == '__main__':
    print(make_demo_video('data/raw/sample.mp4'))
