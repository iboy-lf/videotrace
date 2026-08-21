from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List

from ..models import KnowledgePack
from ..utils.text import informative_keyword_set


@dataclass
class EvalResult:
    retrieval_hit_rate: float
    evidence_coverage: float
    timeline_coverage: float
    clip_overlap: float
    score: float

    def dump(self) -> dict:
        return self.__dict__


@dataclass
class RetrievalEvalResult:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    mean_temporal_iou: float

    def dump(self) -> dict:
        return self.__dict__


def compute_overlap_score(pred_spans: List[dict], gold_spans: List[dict]) -> float:
    if not pred_spans or not gold_spans:
        return 0.0
    covered = 0
    for gold in gold_spans:
        gold_start = float(gold["start_sec"])
        gold_end = float(gold["end_sec"])
        gold_len = max(1e-9, gold_end - gold_start)
        best_cover = 0.0
        for pred in pred_spans:
            pred_start = float(pred["start_sec"])
            pred_end = float(pred["end_sec"])
            inter = max(0.0, min(gold_end, pred_end) - max(gold_start, pred_start))
            best_cover = max(best_cover, inter / gold_len)
        if best_cover >= 0.5:
            covered += 1
    return covered / max(1, len(gold_spans))


def evaluate_temporal_retrieval(
    pred_spans: List[dict],
    gold_spans: List[dict],
    top_k: int | None = None,
    relevance_threshold: float = 0.3,
) -> RetrievalEvalResult:
    predictions = list(pred_spans[:top_k] if top_k else pred_spans)
    if not predictions or not gold_spans:
        return RetrievalEvalResult(0.0, 0.0, 0.0, 0.0, 0.0)

    relevances = [max(_temporal_iou(pred, gold) for gold in gold_spans) for pred in predictions]
    relevant_flags = [score >= relevance_threshold for score in relevances]
    precision_at_k = sum(relevant_flags) / max(1, len(predictions))

    covered_gold = 0
    best_gold_ious: list[float] = []
    for gold in gold_spans:
        best_iou = max(_temporal_iou(pred, gold) for pred in predictions)
        best_gold_ious.append(best_iou)
        if best_iou >= relevance_threshold:
            covered_gold += 1
    recall_at_k = covered_gold / max(1, len(gold_spans))

    first_relevant = next((idx for idx, flag in enumerate(relevant_flags, start=1) if flag), None)
    mrr = 1.0 / first_relevant if first_relevant else 0.0
    dcg = sum(score / math.log2(rank + 1) for rank, score in enumerate(relevances, start=1))
    ideal = [1.0] * min(len(gold_spans), len(predictions))
    idcg = sum(score / math.log2(rank + 1) for rank, score in enumerate(ideal, start=1))
    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0
    mean_temporal_iou = sum(best_gold_ious) / max(1, len(best_gold_ious))
    return RetrievalEvalResult(
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        mrr=mrr,
        ndcg_at_k=ndcg_at_k,
        mean_temporal_iou=mean_temporal_iou,
    )


def _temporal_iou(a: dict, b: dict) -> float:
    a_start = float(a["start_sec"])
    a_end = float(a["end_sec"])
    b_start = float(b["start_sec"])
    b_end = float(b["end_sec"])
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0


def _expand_query_terms(query_terms: List[str]) -> list[str]:
    return sorted(informative_keyword_set(" ".join(query_terms).lower()))


def evaluate_pack(pack: KnowledgePack, query_terms: List[str] | None = None) -> EvalResult:
    query_terms = _expand_query_terms([t.lower() for t in (query_terms or []) if t])
    segments = pack.segments or []
    if not segments:
        return EvalResult(0.0, 0.0, 0.0, 0.0, 0.0)

    selected = pack.metadata.get("ranked_segments") or pack.timeline or []
    hit = 0
    for item in selected:
        text = str(item.get("text", "")).lower()
        if query_terms and any(term in text for term in query_terms):
            hit += 1
    retrieval_hit_rate = hit / max(1, len(selected))

    evidence_coverage = float(
        pack.metadata.get("agent_run", {}).get("verification", {}).get("coverage", 0.0)
    )
    timeline_coverage = len(pack.timeline) / len(segments)
    clip_overlap = min(len(pack.clips), len(pack.timeline)) / max(1, len(pack.timeline))
    score = 0.35 * retrieval_hit_rate + 0.25 * evidence_coverage + 0.20 * timeline_coverage + 0.20 * clip_overlap
    return EvalResult(
        retrieval_hit_rate=retrieval_hit_rate,
        evidence_coverage=evidence_coverage,
        timeline_coverage=timeline_coverage,
        clip_overlap=clip_overlap,
        score=score,
    )
