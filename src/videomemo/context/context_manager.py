from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from ..utils.text import keyword_set, split_sentences


@dataclass
class ContextBudget:
    max_chars: int = 1800
    per_segment_chars: int = 360
    min_segments: int = 2


@dataclass
class ContextItem:
    segment_id: str
    start_sec: float
    end_sec: float
    score: float
    retrieval_score: float
    scorer_score: float
    vlm_score: float
    retrieval_rank_score: float
    scorer_rank_score: float
    vlm_rank_score: float
    understanding_confidence: float
    selection_reason: str
    preserved_fields: list[str]
    text: str
    compressed: bool = False
    compression_reason: str = ""
    # Keep the structured VLM facts beside the compressed text.  The text is
    # still the primary prompt channel, but the verifier must not lose an
    # entity (for example ``眼罩``) merely because query-aware compression
    # dropped the sentence that mentioned it.
    caption: str = ""
    ocr_text: str = ""
    entities: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    scene: str = ""

    def dump(self) -> dict:
        return self.__dict__


@dataclass
class ContextWindow:
    query: str
    budget: ContextBudget
    items: List[ContextItem] = field(default_factory=list)
    dropped_segment_ids: list[str] = field(default_factory=list)
    used_chars: int = 0

    def evidence_tags(self) -> list[str]:
        return [f"timestamp={item.start_sec:.1f}-{item.end_sec:.1f}" for item in self.items]

    def dump(self) -> dict:
        return {
            "query": self.query,
            "budget": self.budget.__dict__,
            "items": [item.dump() for item in self.items],
            "dropped_segment_ids": self.dropped_segment_ids,
            "used_chars": self.used_chars,
        }


class ContextManager:
    """Builds a compact, evidence-preserving context for the agent runtime."""

    def __init__(self, budget: ContextBudget | None = None):
        self.budget = budget or ContextBudget()

    def build(self, query: str, ranked_segments: Iterable[dict]) -> ContextWindow:
        window = ContextWindow(query=query, budget=self.budget)
        for seg in ranked_segments:
            text = str(seg.get("text", "")).strip()
            compressed_text, reason = self._compress_text(query, text, self.budget.per_segment_chars)
            item = ContextItem(
                segment_id=str(seg["segment_id"]),
                start_sec=float(seg["start_sec"]),
                end_sec=float(seg["end_sec"]),
                score=float(seg.get("score", 0.0)),
                retrieval_score=float(seg.get("retrieval_score", 0.0)),
                scorer_score=float(seg.get("scorer_score", 0.0)),
                vlm_score=float(seg.get("vlm_score", 0.0)),
                retrieval_rank_score=float(seg.get("retrieval_rank_score", 0.0)),
                scorer_rank_score=float(seg.get("scorer_rank_score", 0.0)),
                vlm_rank_score=float(seg.get("vlm_rank_score", 0.0)),
                understanding_confidence=float(seg.get("understanding_confidence", 0.0)),
                selection_reason=str(seg.get("selection_reason", "")),
                preserved_fields=[
                    "segment_id",
                    "timestamp",
                    "score",
                    "evidence_text",
                    "caption",
                    "ocr_text",
                    "entities",
                    "actions",
                    "scene",
                ],
                text=compressed_text,
                compressed=len(compressed_text) < len(text),
                compression_reason=reason,
                caption=str(seg.get("caption", "") or "").strip(),
                ocr_text=str(seg.get("ocr_text", "") or "").strip(),
                entities=_string_list(seg.get("entities")),
                actions=_string_list(seg.get("actions")),
                scene=str(seg.get("scene", "") or "").strip(),
            )
            projected = window.used_chars + len(item.text)
            must_keep = len(window.items) < self.budget.min_segments
            if must_keep or projected <= self.budget.max_chars:
                window.items.append(item)
                window.used_chars += len(item.text)
            else:
                window.dropped_segment_ids.append(str(seg["segment_id"]))
        return window

    @staticmethod
    def _compress_text(query: str, text: str, limit: int) -> tuple[str, str]:
        if len(text) <= limit:
            return text, "within_budget"
        q = keyword_set(query)
        sentences = split_sentences(text)
        if sentences and q:
            ranked = sorted(
                sentences,
                key=lambda sentence: (len(q & keyword_set(sentence)), len(sentence)),
                reverse=True,
            )
            kept: list[str] = []
            used = 0
            for sentence in ranked:
                projected = used + len(sentence)
                if projected <= limit:
                    kept.append(sentence)
                    used = projected
            if kept:
                return " ".join(kept), "query_aware_sentence_selection"
        head = max(0, int(limit * 0.72))
        tail = max(0, limit - head - 18)
        compact = f"{text[:head].rstrip()} ... {text[-tail:].lstrip()}" if tail else text[:limit].rstrip()
        return compact, "head_tail_fallback"


def _string_list(value: object) -> list[str]:
    """Normalize optional structured fields without trusting model output."""
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    result: list[str] = []
    for raw in values:
        text = str(raw or "").strip()
        if text and text not in result:
            result.append(text[:160])
    return result[:24]
