from __future__ import annotations

from dataclasses import dataclass

from ..utils.text import informative_keyword_set, keyword_set


@dataclass
class EvidenceDecision:
    sufficient: bool
    reason: str
    query_terms: list[str]
    max_query_coverage: float
    max_vlm_score: float
    best_segment_id: str
    support_term_count: int = 0
    matched_query_terms: list[str] | None = None
    structured_match_count: int = 0
    structured_matched_concepts: list[str] | None = None

    def dump(self) -> dict:
        return self.__dict__


def assess_evidence_sufficiency(
    query: str,
    context: dict,
    min_query_coverage: float = 0.18,
    min_vlm_score: float = 0.16,
    use_semantic_score: bool = True,
    intent_kind: str = "locate",
    coverage_mode: str = "local",
) -> dict:
    items = list(context.get("items", []))
    if not items:
        return EvidenceDecision(False, "no evidence segments", [], 0.0, 0.0, "").dump()

    query_terms = sorted(informative_keyword_set(query))
    if intent_kind == "overview" or coverage_mode == "distributed":
        return EvidenceDecision(
            True,
            "overview query uses temporal coverage evidence",
            query_terms,
            0.0,
            max(float(item.get("vlm_score", 0.0)) for item in items),
            str(items[0].get("segment_id", "")),
            0,
            [],
        ).dump()
    if len(query_terms) <= 1:
        return EvidenceDecision(
            True,
            "query is broad; evidence inspection is allowed",
            query_terms,
            0.0,
            max(float(item.get("vlm_score", 0.0)) for item in items),
            str(items[0].get("segment_id", "")),
        ).dump()

    best_segment_id = ""
    max_coverage = 0.0
    max_vlm_score = 0.0
    max_support_count = 0
    matched_query_terms: list[str] = []
    query_term_set = set(query_terms)
    for item in items:
        text_terms = keyword_set(str(item.get("text", "")))
        matched_terms = sorted(query_term_set & text_terms)
        coverage = len(matched_terms) / max(1, len(query_term_set))
        vlm_score = float(item.get("vlm_score", 0.0))
        if coverage > max_coverage or (coverage == max_coverage and vlm_score > max_vlm_score):
            max_coverage = coverage
            best_segment_id = str(item.get("segment_id", ""))
            max_support_count = len(matched_terms)
            matched_query_terms = matched_terms
        max_vlm_score = max(max_vlm_score, vlm_score)

    # A stage-local question often asks for several concrete entities or
    # attributes (e.g. "佩戴了什么，桌上有什么").  The compressed text and
    # the lexical bigram coverage can undercount such a query, especially when
    # the VLM put the facts in ``entities``/``actions``.  Aggregate those
    # structured facts across the selected stage windows before falling back
    # to the legacy single-window gate.  This remains conservative: at least
    # two independently matched concepts are required, so an unrelated
    # low-confidence window is still rejected.  We retain the ordinary lexical
    # metrics in the decision so reports remain interpretable.
    if coverage_mode == "stage_local":
        structured = _structured_stage_support(query, items)
        if structured["sufficient"]:
            return EvidenceDecision(
                True,
                (
                    "stage-local structured evidence: "
                    f"{structured['match_count']} concepts across "
                    f"{structured['item_count']} windows"
                ),
                query_terms,
                max_coverage,
                max_vlm_score,
                structured["best_segment_id"] or best_segment_id,
                max_support_count,
                matched_query_terms,
                structured["match_count"],
                structured["matched_concepts"],
            ).dump()

    # Chinese bigram recall can look numerically small for a long question
    # even when several concrete concepts are present in one evidence window.
    lexical_support = max_coverage >= min_query_coverage or max_support_count >= 2
    semantic_support = use_semantic_score and max_vlm_score >= min_vlm_score
    sufficient = lexical_support or semantic_support
    if lexical_support:
        reason = f"lexical evidence coverage={max_coverage:.3f}"
    elif semantic_support:
        reason = f"semantic evidence score={max_vlm_score:.3f}"
    else:
        reason = (
            f"insufficient support: coverage={max_coverage:.3f}<{min_query_coverage:.3f}, "
            f"vlm={max_vlm_score:.3f}<{min_vlm_score:.3f}"
        )
    return EvidenceDecision(
        sufficient=sufficient,
        reason=reason,
        query_terms=query_terms,
        max_query_coverage=max_coverage,
        max_vlm_score=max_vlm_score,
        best_segment_id=best_segment_id,
        support_term_count=max_support_count,
        matched_query_terms=matched_query_terms,
    ).dump()


# Concept aliases are intentionally small and product-facing rather than a
# general Chinese ontology.  They make the gate robust to common VLM wording
# variants while keeping unsupported questions behind the normal threshold.
_CONCEPT_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("wearing_or_blindfold", ("佩戴", "戴着", "戴了", "眼罩", "蒙眼")),
    ("blind_test", ("盲测", "盲猜", "猜测", "品尝")),
    ("table_or_surface", ("桌上", "桌子", "桌面", "桌前")),
    ("drink_or_cola", ("饮料", "饮品", "可乐", "饮料瓶", "饮料罐", "吸管")),
    ("display_or_present", ("展示", "举起", "拿起", "介绍")),
    ("label_or_ingredients", ("配料", "配料表", "标签", "产品信息")),
)


def _structured_stage_support(query: str, items: list[dict]) -> dict:
    query_text = str(query or "").lower()
    matched_concepts: list[str] = []
    best_segment_id = ""
    best_item_matches: list[str] = []
    item_count = 0
    for item in items:
        structured_values = (
            item.get("caption"),
            item.get("ocr_text"),
            item.get("scene"),
            item.get("entities"),
            item.get("actions"),
        )
        has_structured = any(
            bool(value)
            for value in structured_values
        )
        structured_text = " ".join(
            [
                str(item.get("caption", "") or ""),
                str(item.get("ocr_text", "") or ""),
                str(item.get("scene", "") or ""),
                " ".join(str(value) for value in (item.get("entities") or [])),
                " ".join(str(value) for value in (item.get("actions") or [])),
                str(item.get("text", "") or ""),
            ]
        ).lower()
        if not structured_text.strip() or not has_structured:
            continue
        item_count += 1
        item_matches: list[str] = []
        for concept, aliases in _CONCEPT_ALIASES:
            query_has_concept = any(alias in query_text for alias in aliases)
            evidence_has_concept = any(alias in structured_text for alias in aliases)
            if query_has_concept and evidence_has_concept:
                item_matches.append(concept)
                if concept not in matched_concepts:
                    matched_concepts.append(concept)
        if len(item_matches) > len(best_item_matches):
            best_item_matches = item_matches
            best_segment_id = str(item.get("segment_id", ""))
    return {
        "sufficient": len(matched_concepts) >= 2 and item_count > 0,
        "match_count": len(matched_concepts),
        "matched_concepts": matched_concepts,
        "item_count": item_count,
        "best_segment_id": best_segment_id,
    }
