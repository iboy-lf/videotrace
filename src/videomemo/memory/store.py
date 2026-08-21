from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List
import json

from ..models import Segment
from ..utils.text import keyword_set, top_keywords


@dataclass
class MemoryRecord:
    memory_id: str
    kind: str
    text: str
    source_segment_id: str
    start_sec: float
    end_sec: float
    importance: float = 0.0
    salience: float = 0.0
    keywords: list[str] = field(default_factory=list)
    video_id: str = ""
    video_path: str = ""

    def dump(self) -> dict:
        return self.__dict__


@dataclass
class VideoMemoryStore:
    records: List[MemoryRecord] = field(default_factory=list)

    @classmethod
    def from_segments(cls, segments: Iterable[Segment], video_id: str = "", video_path: str = "") -> "VideoMemoryStore":
        store = cls()
        for seg in segments:
            text = seg.searchable_text()
            if not text:
                continue
            prefix = f"{video_id}:" if video_id else ""
            keywords = top_keywords(text, limit=8)
            salience = min(1.0, 0.35 * (len(text) / 180.0) + 0.35 * (seg.motion_score / 30.0) + 0.30 * float(seg.score))
            importance = float(seg.score) + 0.02 * float(seg.motion_score) + salience
            store.records.append(
                MemoryRecord(
                    memory_id=f"{prefix}mem-{seg.segment_id}",
                    kind="episodic",
                    text=text,
                    source_segment_id=seg.segment_id,
                    start_sec=seg.start_sec,
                    end_sec=seg.end_sec,
                    importance=importance,
                    salience=salience,
                    keywords=keywords,
                    video_id=video_id,
                    video_path=video_path,
                )
            )
            if keywords:
                store.records.append(
                    MemoryRecord(
                        memory_id=f"{prefix}sem-{seg.segment_id}",
                        kind="semantic",
                        text=f"片段主题关键词：{', '.join(keywords)}",
                        source_segment_id=seg.segment_id,
                        start_sec=seg.start_sec,
                        end_sec=seg.end_sec,
                        importance=0.5 * importance,
                        salience=salience,
                        keywords=keywords,
                        video_id=video_id,
                        video_path=video_path,
                    )
                )
        return store

    def search(self, query: str, top_k: int = 3, kind: str | None = None) -> list[dict]:
        q = keyword_set(query)
        scored = []
        for record in self.records:
            if kind and record.kind != kind:
                continue
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

    def dump(self) -> list[dict]:
        return [record.dump() for record in self.records]

    def save(self, path: str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target
