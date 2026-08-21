from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import hashlib

import cv2
import numpy as np

from ..models import Segment


@dataclass
class VideoAsset:
    path: str
    duration_sec: float
    fps: float
    frame_count: int
    width: int = 0
    height: int = 0
    scene_count: int = 0
    keyframe_count: int = 0


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]


def _hash_frames(frames: List[np.ndarray]) -> str:
    digest = hashlib.sha1()
    for frame in frames:
        small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        digest.update(str(small.shape).encode("ascii"))
        digest.update(small.tobytes())
    return digest.hexdigest()[:16] if frames else _hash_text("empty")


def probe_video(path: str) -> VideoAsset:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = frame_count / fps if fps else 0.0
    cap.release()
    return VideoAsset(path=path, duration_sec=duration_sec, fps=fps, frame_count=frame_count, width=width, height=height)


def _compute_scene_boundaries(path: str, sample_every_sec: float = 2.0, diff_threshold: float = 22.0) -> List[float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [0.0]
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        cap.release()
        return [0.0]
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    boundaries = [0.0]
    prev_gray = None
    sample_idx = 0
    while True:
        sec = sample_idx * sample_every_sec
        if sec >= duration:
            break
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
            if diff >= diff_threshold:
                boundaries.append(round(sec, 2))
        prev_gray = gray
        sample_idx += 1
    boundaries.append(round(duration, 2))
    cap.release()
    cleaned = sorted(set(boundaries))
    return cleaned


def split_video_into_segments(path: str, segment_seconds: int = 30, use_scene_cut: bool = False) -> Tuple[VideoAsset, List[Segment]]:
    asset = probe_video(path)
    segments: List[Segment] = []
    if asset.duration_sec <= 0:
        segments.append(Segment(segment_id='seg-0000', start_sec=0.0, end_sec=0.0))
        return asset, segments

    if use_scene_cut:
        boundaries = _compute_scene_boundaries(path)
        if len(boundaries) >= 2:
            for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                if end <= start:
                    continue
                segments.append(Segment(segment_id=f'seg-{idx:04d}', start_sec=round(start, 2), end_sec=round(end, 2)))
        else:
            use_scene_cut = False

    if not use_scene_cut:
        idx = 0
        start = 0.0
        while start < asset.duration_sec:
            end = min(asset.duration_sec, start + segment_seconds)
            seg_id = f'seg-{idx:04d}'
            segments.append(Segment(segment_id=seg_id, start_sec=round(start, 2), end_sec=round(end, 2)))
            idx += 1
            start = end
    return asset, segments


def _sample_keyframes(video_path: str, start_sec: float, end_sec: float, num_frames: int = 3) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    if end_sec <= start_sec:
        cap.release()
        return []
    frames = []
    times = np.linspace(start_sec, end_sec, num=max(1, num_frames), endpoint=False)
    for sec in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def _visual_stats(frames: List[np.ndarray]) -> tuple[float, float, float, str]:
    if not frames:
        return 0.0, 0.0, 0.0, _hash_text('empty')
    brightness_vals = []
    contrast_vals = []
    motion_vals = []
    prev = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_vals.append(float(np.mean(gray)))
        contrast_vals.append(float(np.std(gray)))
        if prev is not None:
            motion_vals.append(float(np.mean(cv2.absdiff(gray, prev))))
        prev = gray
    brightness_mean = float(np.mean(brightness_vals))
    contrast_std = float(np.mean(contrast_vals))
    motion_score = float(np.mean(motion_vals)) if motion_vals else 0.0
    signature = _hash_text(f"{brightness_mean:.3f}:{contrast_std:.3f}:{motion_score:.3f}")
    return brightness_mean, contrast_std, motion_score, signature


def _describe_visual_segment(seg: Segment, duration_sec: float) -> str:
    position = _segment_position(seg.start_sec, seg.end_sec, duration_sec)
    motion = _motion_label(seg.motion_score)
    brightness = _brightness_label(seg.brightness_mean)
    contrast = _contrast_label(seg.contrast_std)
    return (
        f"这是视频的{position}片段，时间范围 {seg.start_sec:.1f}s 到 {seg.end_sec:.1f}s。"
        f"当前仅记录不依赖文件名的低层视觉统计：画面整体{brightness}，"
        f"画面变化{motion}，帧内对比度{contrast}。"
        "尚未运行语义视觉理解器，因此不对人物、物体、动作或画面文字作推断。"
    )


def _segment_position(start_sec: float, end_sec: float, duration_sec: float) -> str:
    mid = (start_sec + end_sec) / 2.0
    if duration_sec <= 0:
        return "当前"
    ratio = mid / duration_sec
    if ratio < 0.25:
        return "开头"
    if ratio < 0.55:
        return "中前段"
    if ratio < 0.80:
        return "中后段"
    return "结尾"


def _brightness_label(value: float) -> str:
    if value <= 0:
        return "亮度未知"
    if value < 70:
        return "偏暗"
    if value < 145:
        return "亮度适中"
    return "偏亮"


def _motion_label(value: float) -> str:
    if value < 3:
        return "较小，可能是静态讲解或稳定镜头"
    if value < 12:
        return "中等，可能包含人物动作、切换或镜头移动"
    return "明显，可能包含快速切换、运动画面或较多画面变化"


def _contrast_label(value: float) -> str:
    if value < 4:
        return "较稳定"
    if value < 14:
        return "适中"
    return "较强"


def sample_segment_text(video_path: str, segments: List[Segment]) -> List[Segment]:
    enriched = []
    asset = probe_video(video_path)
    for seg in segments:
        seg.ocr_text = ""
        seg.asr_text = ""
        keyframes = _sample_keyframes(video_path, seg.start_sec, seg.end_sec, num_frames=3)
        seg.frame_count = len(keyframes)
        seg.frame_hash = _hash_frames(keyframes)
        brightness_mean, contrast_std, motion_score, signature = _visual_stats(keyframes)
        seg.brightness_mean = brightness_mean
        seg.contrast_std = contrast_std
        seg.motion_score = motion_score
        seg.visual_signature = signature
        seg.text = _describe_visual_segment(seg, asset.duration_sec)
        seg.caption = seg.text
        seg.understanding_backend = "baseline"
        seg.evidence = [f"timestamp={seg.start_sec:.1f}-{seg.end_sec:.1f}"]
        enriched.append(seg)
    return enriched
