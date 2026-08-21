from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np


class EmbeddingCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key(self, *parts: object) -> str:
        raw = json.dumps([str(part) for part in parts], ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def load(self, key: str) -> np.ndarray | None:
        path = self.cache_dir / f"{key}.npy"
        if not path.exists():
            return None
        return np.load(path)

    def load_or_migrate(self, key: str, legacy_keys: tuple[str, ...] = ()) -> np.ndarray | None:
        cached = self.load(key)
        if cached is not None:
            return cached
        for legacy_key in legacy_keys:
            if legacy_key == key:
                continue
            cached = self.load(legacy_key)
            if cached is not None:
                self.save(key, cached)
                return cached
        return None

    def save(self, key: str, vector: np.ndarray) -> Path:
        path = self.cache_dir / f"{key}.npy"
        np.save(path, vector.astype("float32"))
        return path
