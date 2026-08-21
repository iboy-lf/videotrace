from __future__ import annotations

from pathlib import Path

import numpy as np

from ..models import Segment
from .base import SegmentEmbedding
from .cache import EmbeddingCache
from .frame_sampler import sample_frames


class FrozenCLIPEmbedder:
    backend = "frozen_clip"

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        cache_dir: str = "outputs_cache/vlm",
        num_frames: int = 4,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.cache = EmbeddingCache(cache_dir)
        self.num_frames = num_frames
        self.device = device
        self._model = None
        self._processor = None

    def embed_text(self, text: str) -> np.ndarray:
        self._load()
        import torch

        self.device = self.device or _model_device(self._model)
        if self.device is None:
            raise RuntimeError("CLIP runtime has no resolved input device")
        inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            vec = self._model.get_text_features(**inputs)[0].detach().cpu().numpy().astype("float32")
        return _normalize(vec)

    def embed_segment(self, video_path: str, segment: Segment) -> SegmentEmbedding:
        cache_key = self.cache.key(
            "segment-content-v1",
            self.backend,
            self.model_name,
            segment.segment_id,
            segment.start_sec,
            segment.end_sec,
            segment.frame_hash,
            self.num_frames,
        )
        legacy_key = self.cache.key(
            self.backend,
            self.model_name,
            video_path,
            segment.segment_id,
            segment.start_sec,
            segment.end_sec,
            segment.frame_hash,
        )
        cached = self.cache.load_or_migrate(cache_key, (legacy_key,))
        if cached is not None:
            return SegmentEmbedding(segment.segment_id, cached, self.backend, {"cache": "hit", "model_name": self.model_name})
        self._load()
        import torch
        from PIL import Image

        frames = sample_frames(video_path, segment.start_sec, segment.end_sec, self.num_frames)
        if not frames:
            raise ValueError(f"No frames sampled for segment {segment.segment_id}")
        images = [Image.fromarray(frame) for frame in frames]
        self.device = self.device or _model_device(self._model)
        if self.device is None:
            raise RuntimeError("CLIP runtime has no resolved input device")
        inputs = self._processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs).detach().cpu().numpy().astype("float32")
        vector = _normalize(image_features.mean(axis=0))
        self.cache.save(cache_key, vector)
        return SegmentEmbedding(
            segment.segment_id,
            vector,
            self.backend,
            {"cache": "miss", "model_name": self.model_name, "num_frames": len(frames)},
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        local_only = Path(self.model_name).exists()
        self._processor = CLIPProcessor.from_pretrained(self.model_name, local_files_only=local_only)
        self._model = CLIPModel.from_pretrained(self.model_name, local_files_only=local_only).to(self.device)
        self._model.eval()


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return vector.astype("float32")
    return (vector / norm).astype("float32")


def _model_device(model):
    if model is None:
        return None
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None
