from __future__ import annotations

from typing import List

import cv2
import numpy as np


def sample_frames(video_path: str, start_sec: float, end_sec: float, num_frames: int = 4) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened() or end_sec <= start_sec:
        return []
    frames: list[np.ndarray] = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = frame_count / fps if fps > 0 else end_sec - start_sec
    times = _pick_evidence_times(video_path, start_sec, end_sec, duration, num_frames)
    for sec in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000.0)
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return _dedupe_frames(frames)


def _pick_evidence_times(video_path: str, start_sec: float, end_sec: float, duration: float, num_frames: int) -> list[float]:
    if num_frames <= 1:
        return [(start_sec + end_sec) / 2.0]
    span = max(0.01, end_sec - start_sec)
    base = [start_sec + span * 0.06, start_sec + span * 0.5, max(start_sec, end_sec - span * 0.06)]
    change_point = _find_change_point(video_path, start_sec, end_sec)
    if change_point is not None:
        base.insert(1, change_point)
    if num_frames <= 3:
        return sorted(base)[:num_frames]
    extra = np.linspace(start_sec, end_sec, num=max(1, num_frames - len(base)), endpoint=False).tolist()
    merged = sorted({round(sec, 3) for sec in base + extra})
    return merged[:num_frames]


def _find_change_point(video_path: str, start_sec: float, end_sec: float) -> float | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    times = np.linspace(start_sec, end_sec, num=5, endpoint=False)
    prev_gray = None
    best_time = None
    best_diff = 0.0
    for sec in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
            if diff > best_diff:
                best_diff = diff
                best_time = float(sec)
        prev_gray = gray
    cap.release()
    return best_time


def _dedupe_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    if not frames:
        return frames
    unique: list[np.ndarray] = []
    signatures: set[int] = set()
    for frame in frames:
        signature = int(np.mean(frame))
        if signature in signatures:
            continue
        signatures.add(signature)
        unique.append(frame)
    return unique
