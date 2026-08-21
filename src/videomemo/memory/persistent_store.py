from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from ..utils.text import keyword_set
from .store import MemoryRecord, VideoMemoryStore


@dataclass
class PersistentMemoryStore:
    path: str

    def load_records(self) -> list[MemoryRecord]:
        target = Path(self.path)
        if not target.exists():
            return []
        records: list[MemoryRecord] = []
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            records.append(MemoryRecord(**data))
        return records

    def upsert(self, store: VideoMemoryStore) -> int:
        target = Path(self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = {record.memory_id: record for record in self.load_records()}
        for record in store.records:
            existing[record.memory_id] = record
        lines = [json.dumps(record.dump(), ensure_ascii=False) for record in existing.values()]
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return len(store.records)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q = keyword_set(query)
        scored = []
        for record in self.load_records():
            text_overlap = len(q & keyword_set(record.text))
            keyword_overlap = len(q & set(record.keywords))
            score = text_overlap + 1.5 * keyword_overlap + 0.08 * record.importance + 0.05 * record.salience
            scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                **record.dump(),
                "score": float(score),
            }
            for score, record in scored[:top_k]
            if score > 0
        ]
