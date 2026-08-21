from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..models import Segment


@dataclass
class SegmentEmbedding:
    segment_id: str
    vector: np.ndarray
    backend: str
    metadata: dict

    def dump(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "backend": self.backend,
            "dim": int(self.vector.shape[0]),
            "metadata": self.metadata,
        }


class VLMEmbedder(Protocol):
    backend: str

    def embed_text(self, text: str) -> np.ndarray:
        ...

    def embed_segment(self, video_path: str, segment: Segment) -> SegmentEmbedding:
        ...
