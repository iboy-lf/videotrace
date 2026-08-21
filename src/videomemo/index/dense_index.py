from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class DenseIndexArtifact:
    backend: str
    storage_backend: str
    vector_path: str
    metadata_path: str
    num_vectors: int
    dim: int

    def dump(self) -> dict:
        return self.__dict__


class NumpyDenseIndex:
    """A small persistent cosine index with a FAISS-compatible contract."""

    storage_backend = "numpy_flat_cosine"

    def __init__(self, vectors: np.ndarray, records: list[dict]):
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2:
            raise ValueError("dense index vectors must be a 2D matrix")
        if matrix.shape[0] != len(records):
            raise ValueError("dense index vector and metadata counts do not match")
        self.vectors = _normalize_rows(matrix)
        self.records = list(records)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        if not len(self.records) or int(top_k) <= 0:
            return []
        query = np.asarray(query_vector, dtype="float32")
        if query.ndim != 1:
            raise ValueError("dense index query vector must be one-dimensional")
        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"dense index query dim {query.shape[0]} does not match index dim {self.vectors.shape[1]}"
            )
        query = _normalize(query)
        scores = self.vectors @ query
        order = np.argsort(-scores)[: max(0, int(top_k))]
        return [
            {**self.records[int(index)], "dense_score": float(scores[int(index)])}
            for index in order
        ]

    def save(self, directory: str, name: str, backend: str) -> DenseIndexArtifact:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        safe_name = _safe_name(name)
        vector_path = root / f"{safe_name}.npy"
        metadata_path = root / f"{safe_name}.json"
        np.save(vector_path, self.vectors.astype("float32"))
        metadata_path.write_text(
            json.dumps(
                {
                    "backend": backend,
                    "storage_backend": self.storage_backend,
                    "num_vectors": int(self.vectors.shape[0]),
                    "dim": int(self.vectors.shape[1]) if self.vectors.size else 0,
                    "records": self.records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return DenseIndexArtifact(
            backend=backend,
            storage_backend=self.storage_backend,
            vector_path=str(vector_path),
            metadata_path=str(metadata_path),
            num_vectors=int(self.vectors.shape[0]),
            dim=int(self.vectors.shape[1]) if self.vectors.size else 0,
        )

    @classmethod
    def load(cls, metadata_path: str) -> "NumpyDenseIndex":
        meta_path = Path(metadata_path)
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        storage_backend = payload.get("storage_backend")
        if storage_backend not in {None, cls.storage_backend}:
            raise ValueError(f"unsupported dense index storage backend: {storage_backend}")
        vector_path = meta_path.with_suffix(".npy")
        index = cls(np.load(vector_path), list(payload.get("records", [])))
        expected_dim = int(payload.get("dim", index.vectors.shape[1]))
        expected_count = int(payload.get("num_vectors", len(index.records)))
        if index.vectors.shape[1] != expected_dim or len(index.records) != expected_count:
            raise ValueError("dense index metadata does not match stored vectors")
        return index


def persist_segment_index(
    directory: str,
    video_id: str,
    backend: str,
    embeddings: Iterable[dict],
) -> dict:
    vectors: list[np.ndarray] = []
    records: list[dict] = []
    for item in embeddings:
        vector = item.get("vector")
        if vector is None:
            continue
        vectors.append(np.asarray(vector, dtype="float32"))
        records.append(
            {
                "segment_id": str(item.get("segment_id", "")),
                "start_sec": float(item.get("start_sec", 0.0)),
                "end_sec": float(item.get("end_sec", 0.0)),
            }
        )
    if not vectors:
        return {"enabled": False, "reason": "no segment embeddings"}
    index = NumpyDenseIndex(np.stack(vectors), records)
    artifact = index.save(directory, video_id, backend)
    return {"enabled": True, **artifact.dump()}


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= 1e-9 else vector / norm


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-9, 1.0, norms)
    return matrix / norms


def _safe_name(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._-")
    return f"{cleaned[:64] or 'video'}-{digest}"
