from __future__ import annotations

import numpy as np

from ..models import Segment
from .base import VLMEmbedder


def attach_vlm_scores(
    video_path: str,
    query: str,
    segments: list[Segment],
    embedder: VLMEmbedder | None,
    persist_index: bool = False,
    index_dir: str = "outputs/indexes",
) -> dict:
    if embedder is None:
        return {"enabled": False, "backend": "none", "segments": []}
    query_vec = embedder.embed_text(query)
    records = []
    index_records = []
    for segment in segments:
        segment_embedding = embedder.embed_segment(video_path, segment)
        segment.vlm_score = cosine(query_vec, segment_embedding.vector)
        records.append(
            {
                **segment_embedding.dump(),
                "vlm_score": segment.vlm_score,
            }
        )
        index_records.append(
            {
                "segment_id": segment.segment_id,
                "start_sec": segment.start_sec,
                "end_sec": segment.end_sec,
                "vector": segment_embedding.vector,
            }
        )
    index_artifact = {"enabled": False}
    if persist_index:
        from pathlib import Path

        from ..index.dense_index import persist_segment_index

        index_artifact = persist_segment_index(
            index_dir, Path(video_path).stem, embedder.backend, index_records
        )
    return {
        "enabled": True,
        "backend": embedder.backend,
        "segments": records,
        "persistent_index": index_artifact,
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return 0.0
    return float((a @ b.T) / denom)
