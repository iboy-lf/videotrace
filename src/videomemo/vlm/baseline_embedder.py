from __future__ import annotations

import hashlib

import cv2
import numpy as np

from ..models import Segment
from ..utils.text import tokenize_text
from .base import SegmentEmbedding
from .cache import EmbeddingCache
from .frame_sampler import sample_frames


class BaselineMultimodalEmbedder:
    """Deterministic local multimodal embedder used when no VLM weights are available."""

    backend = "baseline_multimodal"

    def __init__(self, cache_dir: str = "outputs_cache/vlm", dim: int = 128, num_frames: int = 4):
        self.cache = EmbeddingCache(cache_dir)
        self.dim = dim
        self.num_frames = num_frames

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        for token in tokenize_text(text):
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        return _normalize(vec)

    def embed_segment(self, video_path: str, segment: Segment) -> SegmentEmbedding:
        text_hash = hashlib.sha256(segment.searchable_text().encode("utf-8")).hexdigest()
        cache_key = self.cache.key(
            "segment-content-v1",
            self.backend,
            segment.segment_id,
            segment.start_sec,
            segment.end_sec,
            segment.frame_hash,
            text_hash,
            self.num_frames,
            self.dim,
        )
        legacy_key = self.cache.key(
            self.backend,
            video_path,
            segment.segment_id,
            segment.start_sec,
            segment.end_sec,
            segment.frame_hash,
        )
        cached = self.cache.load_or_migrate(cache_key, (legacy_key,))
        if cached is not None:
            return SegmentEmbedding(segment.segment_id, cached, self.backend, {"cache": "hit"})

        text_vec = self.embed_text(segment.searchable_text())
        visual_vec = self._visual_vector(video_path, segment)
        vector = _normalize(0.58 * text_vec + 0.42 * visual_vec)
        self.cache.save(cache_key, vector)
        return SegmentEmbedding(
            segment.segment_id,
            vector,
            self.backend,
            {"cache": "miss", "num_frames": self.num_frames, "dim": self.dim},
        )

    def _visual_vector(self, video_path: str, segment: Segment) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        frames = sample_frames(video_path, segment.start_sec, segment.end_sec, self.num_frames)
        if not frames:
            return vec
        hist_parts = []
        for frame in frames:
            small = cv2.resize(frame, (32, 32))
            for channel in range(3):
                hist = cv2.calcHist([small], [channel], None, [16], [0, 256]).reshape(-1)
                hist_parts.append(hist)
        hist_vec = np.concatenate(hist_parts).astype("float32")
        length = min(self.dim, hist_vec.shape[0])
        vec[:length] = hist_vec[:length]
        stats = np.array(
            [segment.brightness_mean, segment.contrast_std, segment.motion_score],
            dtype="float32",
        )
        vec[-3:] = stats
        return _normalize(vec)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return vector.astype("float32")
    return (vector / norm).astype("float32")
