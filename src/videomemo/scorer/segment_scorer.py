from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..models import Segment
from ..utils.text import informative_keyword_set, keyword_set


@dataclass
class ScoreExample:
    query: str
    segment_id: str
    label: float
    features: dict


class SegmentScorer:
    def score(self, query: str, segment: Segment) -> float:
        q = informative_keyword_set(query)
        t = keyword_set(segment.searchable_text().lower())
        lexical = len(q & t)
        visual = 0.01 * segment.motion_score + 0.001 * segment.brightness_mean + 0.001 * segment.contrast_std
        temporal = 1.0 / (1.0 + segment.start_sec)
        return float(lexical + visual + temporal)

    def rank(self, query: str, segments: List[Segment]) -> List[Segment]:
        ranked = sorted(segments, key=lambda seg: self.score(query, seg), reverse=True)
        for seg in ranked:
            seg.score = self.score(query, seg)
        return ranked


def build_score_examples(query: str, segments: List[Segment], positive_k: int = 1) -> List[ScoreExample]:
    if not segments:
        return []
    scorer = SegmentScorer()
    ranked = scorer.rank(query, segments)
    positives = ranked[:positive_k]
    negatives = ranked[positive_k:]
    examples = []
    for pos in positives:
        examples.append(
            ScoreExample(
                query=query,
                segment_id=pos.segment_id,
                label=1.0,
                features=_segment_features(pos),
            )
        )
    for neg in negatives:
        examples.append(
            ScoreExample(
                query=query,
                segment_id=neg.segment_id,
                label=0.0,
                features=_segment_features(neg),
            )
        )
    return examples


def _segment_features(segment: Segment) -> dict:
    return {
        'start_sec': segment.start_sec,
        'end_sec': segment.end_sec,
        'frame_count': segment.frame_count,
        'brightness_mean': segment.brightness_mean,
        'contrast_std': segment.contrast_std,
        'motion_score': segment.motion_score,
        'visual_signature': segment.visual_signature,
        'text_len': len(segment.text),
        'ocr_len': len(segment.ocr_text),
        'asr_len': len(segment.asr_text),
    }
