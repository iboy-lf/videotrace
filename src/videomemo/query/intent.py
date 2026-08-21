from __future__ import annotations

from dataclasses import dataclass, field
import re

from ..utils.text import informative_keyword_set


@dataclass
class QueryIntent:
    kind: str = "locate"
    coverage_mode: str = "local"
    stage_hints: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    rationale: str = ""
    time_anchor_sec: float | None = None

    def dump(self) -> dict:
        return {
            "kind": self.kind,
            "coverage_mode": self.coverage_mode,
            "stage_hints": list(self.stage_hints),
            "terms": list(self.terms),
            "rationale": self.rationale,
            "time_anchor_sec": self.time_anchor_sec,
        }


def classify_query(query: str) -> QueryIntent:
    value = str(query or "").strip().lower()
    terms = sorted(informative_keyword_set(query))
    opening = _contains(value, ("开场", "开头", "开始", "一开始", "opening", "intro"))
    middle = _contains(value, ("中段", "中间", "过程", "各国", "分国家", "试喝", "middle"))
    ending = _contains(value, ("最后", "结尾", "末尾", "盲测", "ending", "final"))
    stage_hints = [name for name, hit in (("opening", opening), ("middle", middle), ("ending", ending)) if hit]
    time_anchor_sec = _time_anchor(value)
    overview_markers = (
        "整体",
        "流程",
        "主要讲",
        "概括",
        "总结",
        "全过程",
        "阶段",
        "依次",
        "从头到尾",
        "从开场到",
        "从开始到",
        "到最后",
        "overview",
        "process",
    )
    is_overview = len(stage_hints) >= 2 or any(marker in value for marker in overview_markers)
    if is_overview:
        rationale = "query asks for a global summary or multiple temporal stages"
        return QueryIntent("overview", "distributed", stage_hints, terms, rationale, time_anchor_sec)
    if any(marker in value for marker in ("比较", "区别", "差异", "差距", "对比", "compare", "difference")):
        multi_span = any(
            marker in value
            for marker in ("分别", "两次", "多个", "各国", "各自", "前后", "不同国家", "multi-span")
        )
        return QueryIntent(
            "comparison",
            "multi_span" if multi_span else ("stage_local" if stage_hints else "local"),
            stage_hints,
            terms,
            "query compares multiple entities or moments",
            time_anchor_sec,
        )
    if any(marker in value for marker in ("多少", "几个", "数量", "count")):
        return QueryIntent("count", "multi_span", stage_hints, terms, "query asks for evidence across multiple moments", time_anchor_sec)
    if stage_hints:
        return QueryIntent("locate", "stage_local", stage_hints, terms, "query names a temporal stage", time_anchor_sec)
    if time_anchor_sec is not None:
        return QueryIntent("locate", "time_local", stage_hints, terms, "query names an explicit time anchor", time_anchor_sec)
    return QueryIntent("locate", "local", stage_hints, terms, "query is best answered by local evidence", time_anchor_sec)


def select_ranked_segments(
    candidates: list[dict],
    top_k: int,
    duration_sec: float,
    intent: QueryIntent,
    enabled: bool = True,
    min_segments: int = 3,
) -> list[dict]:
    """Select evidence with temporal coverage for overview queries.

    Local questions retain pure relevance ranking. Overview questions use a bounded
    greedy selector that trades a small amount of score for coverage of early,
    middle, and late portions of the video.
    """
    if not candidates or top_k <= 0:
        return []
    if intent.coverage_mode == "stage_local":
        return _select_stage_local(pool=candidates, top_k=top_k, duration=duration_sec, intent=intent)
    if intent.coverage_mode == "time_local" and intent.time_anchor_sec is not None:
        return _select_time_local(pool=candidates, top_k=top_k, anchor=intent.time_anchor_sec)
    if not enabled or intent.coverage_mode == "local" or top_k < 2:
        return [dict(item, selection_reason="relevance_top_k") for item in candidates[:top_k]]

    pool = [dict(item) for item in candidates]
    duration = max(float(duration_sec), 1.0)
    score_values = [float(item.get("score", 0.0)) for item in pool]
    low, high = min(score_values), max(score_values)

    def normalized_score(item: dict) -> float:
        if high - low <= 1e-9:
            return 0.5
        return (float(item.get("score", 0.0)) - low) / (high - low)

    remaining = list(pool)
    if intent.coverage_mode == "multi_span":
        remaining.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        first = remaining.pop(0)
        first["selection_reason"] = "relevance_anchor"
        selected = [first]
        while remaining and len(selected) < top_k:
            def multi_span_utility(item: dict) -> float:
                relevance = normalized_score(item)
                novelty = min(
                    abs(_midpoint(item) - _midpoint(other)) / duration
                    for other in selected
                )
                return 0.88 * relevance + 0.12 * novelty

            chosen = max(remaining, key=multi_span_utility)
            chosen["selection_reason"] = "relevance_plus_temporal_novelty"
            selected.append(chosen)
            remaining.remove(chosen)
        selected.sort(key=lambda item: (float(item.get("start_sec", 0.0)), -float(item.get("score", 0.0))))
        return selected[:top_k]

    targets = _temporal_targets(intent, top_k, min_segments)
    selected: list[dict] = []
    for target, label in targets:
        if not remaining:
            break
        stage_pool = [item for item in remaining if _in_stage(item, duration, label)]
        eligible = stage_pool or remaining
        chosen = max(
            eligible,
            key=lambda item: 0.10 * normalized_score(item)
            - 0.90 * _temporal_distance(item, duration, label, target),
        )
        chosen["selection_reason"] = f"temporal_coverage:{label}"
        selected.append(chosen)
        remaining.remove(chosen)

    while remaining and len(selected) < top_k:
        def utility(item: dict) -> float:
            relevance = normalized_score(item)
            novelty = min(
                abs(_midpoint(item) - _midpoint(other)) / duration
                for other in selected
            ) if selected else 1.0
            return 0.78 * relevance + 0.22 * novelty

        chosen = max(remaining, key=utility)
        chosen["selection_reason"] = "relevance_plus_temporal_novelty"
        selected.append(chosen)
        remaining.remove(chosen)

    selected.sort(key=lambda item: (float(item.get("start_sec", 0.0)), -float(item.get("score", 0.0))))
    return selected[:top_k]


def _temporal_targets(intent: QueryIntent, top_k: int, min_segments: int) -> list[tuple[float, str]]:
    labels = list(intent.stage_hints)
    if not labels:
        labels = ["opening", "middle", "ending"]
    targets = {"opening": 0.08, "middle": 0.50, "ending": 0.92}
    result = [(targets[label], label) for label in labels if label in targets]
    if len(result) < min(min_segments, top_k):
        for label, target in (("opening", 0.08), ("middle", 0.50), ("ending", 0.92)):
            if label not in {name for _, name in result}:
                result.append((target, label))
            if len(result) >= min(min_segments, top_k):
                break
    return result[:top_k]


def _contains(value: str, markers: tuple[str, ...]) -> bool:
    return any(marker in value for marker in markers)


def _time_anchor(value: str) -> float | None:
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:秒|s)(?:左右|附近|前后)?", value)
    return float(match.group(1)) if match else None


def _select_stage_local(pool: list[dict], top_k: int, duration: float, intent: QueryIntent) -> list[dict]:
    labels = set(intent.stage_hints)
    eligible = [item for item in pool if any(_in_stage(item, max(duration, 1.0), label) for label in labels)]
    if not eligible:
        eligible = list(pool)
    eligible.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    selected = [dict(item, selection_reason=f"stage_local:{next(iter(labels), 'local')}") for item in eligible[:top_k]]
    selected.sort(key=lambda item: (float(item.get("start_sec", 0.0)), -float(item.get("score", 0.0))))
    return selected


def _select_time_local(pool: list[dict], top_k: int, anchor: float) -> list[dict]:
    ranked = sorted(
        pool,
        key=lambda item: (
            abs(((float(item.get("start_sec", 0.0)) + float(item.get("end_sec", 0.0))) / 2.0) - anchor),
            -float(item.get("score", 0.0)),
        ),
    )
    selected = [dict(item, selection_reason="time_local") for item in ranked[:top_k]]
    selected.sort(key=lambda item: (float(item.get("start_sec", 0.0)), -float(item.get("score", 0.0))))
    return selected


def _midpoint(item: dict) -> float:
    return (float(item.get("start_sec", 0.0)) + float(item.get("end_sec", 0.0))) / 2.0


def _temporal_distance(item: dict, duration: float, label: str, target: float) -> float:
    duration = max(duration, 1.0)
    if label == "opening":
        return max(float(item.get("start_sec", 0.0)), 0.0) / duration
    if label == "ending":
        return max(duration - float(item.get("end_sec", 0.0)), 0.0) / duration
    return abs(_midpoint(item) / duration - target)


def _in_stage(item: dict, duration: float, label: str) -> bool:
    ratio = _midpoint(item) / max(duration, 1.0)
    if label == "opening":
        return ratio <= 0.25
    if label == "ending":
        return ratio >= 0.75
    return 0.25 < ratio < 0.75
