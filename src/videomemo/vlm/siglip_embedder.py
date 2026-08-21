from __future__ import annotations

from pathlib import Path

import numpy as np

from ..models import Segment
from .base import SegmentEmbedding
from .cache import EmbeddingCache
from .frame_sampler import sample_frames


_MODEL_CACHE: dict[tuple[str, str], tuple[object, object]] = {}


class FrozenSigLIPEmbedder:
    backend = "frozen_siglip"

    def __init__(
        self,
        model_name: str,
        cache_dir: str = "outputs_cache/vlm",
        num_frames: int = 4,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.cache = EmbeddingCache(cache_dir)
        self.num_frames = num_frames
        self.device = device or None
        self._model = None
        self._processor = None

    def embed_text(self, text: str) -> np.ndarray:
        self._load()
        import torch

        self.device = self.device or _model_device(self._model)
        if self.device is None:
            raise RuntimeError("SigLIP runtime has no resolved input device")
        inputs = self._processor(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(self.device)
        with torch.inference_mode():
            output = self._model.get_text_features(**inputs)
        features = _feature_tensor(output)
        return _normalize(features[0].detach().float().cpu().numpy())

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
            return SegmentEmbedding(
                segment.segment_id,
                cached,
                self.backend,
                {"cache": "hit", "model_name": self.model_name},
            )

        self._load()
        import torch
        from PIL import Image

        frames = sample_frames(video_path, segment.start_sec, segment.end_sec, self.num_frames)
        if not frames:
            raise ValueError(f"No frames sampled for segment {segment.segment_id}")
        images = [Image.fromarray(frame[:, :, ::-1]) for frame in frames]
        self.device = self.device or _model_device(self._model)
        if self.device is None:
            raise RuntimeError("SigLIP runtime has no resolved input device")
        inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self._model.get_image_features(**inputs)
        features = _feature_tensor(output)
        vector = _normalize(features.detach().float().cpu().numpy().mean(axis=0))
        self.cache.save(cache_key, vector)
        return SegmentEmbedding(
            segment.segment_id,
            vector,
            self.backend,
            {
                "cache": "miss",
                "model_name": self.model_name,
                "num_frames": len(frames),
                "device": self.device,
            },
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_key = (self.model_name, self.device)
        if cache_key in _MODEL_CACHE:
            self._processor, self._model = _MODEL_CACHE[cache_key]
            return
        local_only = Path(self.model_name).exists()
        dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            local_files_only=local_only,
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            local_files_only=local_only,
            trust_remote_code=True,
            dtype=dtype,
        ).to(self.device)
        self._model.eval()
        _MODEL_CACHE[cache_key] = (self._processor, self._model)


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype("float32")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return vector
    return vector / norm


def _feature_tensor(output):
    import torch

    if isinstance(output, torch.Tensor):
        return output
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            return value
    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden.mean(dim=1)
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        value = output[0]
        return value.mean(dim=1) if value.ndim == 3 else value
    raise TypeError(f"Unsupported feature output type: {type(output).__name__}")


def _model_device(model):
    if model is None:
        return None
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None
