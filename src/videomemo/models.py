from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Segment:
    segment_id: str
    start_sec: float
    end_sec: float
    text: str = ""
    ocr_text: str = ""
    asr_text: str = ""
    frame_count: int = 0
    frame_hash: str = ""
    brightness_mean: float = 0.0
    contrast_std: float = 0.0
    motion_score: float = 0.0
    visual_signature: str = ""
    caption: str = ""
    entities: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    scene: str = ""
    understanding_confidence: float = 0.0
    understanding_backend: str = "baseline"
    retrieval_score: float = 0.0
    vlm_score: float = 0.0
    scorer_score: float = 0.0
    reranker_score: float = 0.0
    retrieval_rank_score: float = 0.0
    vlm_rank_score: float = 0.0
    scorer_rank_score: float = 0.0
    score: float = 0.0
    evidence: List[str] = field(default_factory=list)

    def searchable_text(self) -> str:
        """Return the aligned text channels used by retrieval and memory."""
        return " ".join(
            value.strip()
            for value in (self.text, self.ocr_text, self.asr_text)
            if value and value.strip()
        ).strip()


@dataclass
class KnowledgePack:
    video_path: str
    duration_sec: float
    segments: List[Segment]
    summary: str
    answer: str
    timeline: List[dict]
    clips: List[dict]
    metadata: dict = field(default_factory=dict)

    def dump(self) -> dict:
        return {
            "video_path": self.video_path,
            "duration_sec": self.duration_sec,
            "segments": [segment.__dict__ for segment in self.segments],
            "summary": self.summary,
            "answer": self.answer,
            "timeline": self.timeline,
            "clips": self.clips,
            "metadata": self.metadata,
        }
