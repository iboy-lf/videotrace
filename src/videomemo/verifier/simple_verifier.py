from __future__ import annotations

import math
import re
from typing import Iterable, List, Sequence, Tuple


_TIMESTAMP_PATTERN = re.compile(r"timestamp=([0-9]+(?:\.[0-9]+)?-[0-9]+(?:\.[0-9]+)?)")
_CJK_PATTERN = re.compile(r"[\u3400-\u9fff]+")
_ASCII_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]+")
_REFUSAL_MARKERS = (
    "证据不足",
    "无法确认",
    "不能确认",
    "无法回答",
    "不足以支持",
    "未提供足以支持",
    "未提供足够证据",
)
_GENERIC_UNITS = {
    "视频",
    "画面",
    "展示",
    "可以",
    "看到",
    "出现",
    "片段",
    "证据",
    "结论",
    "随后",
    "进行",
    "一个",
    "一名",
    "男子",
    "女子",
    "镜头",
    "时间",
}


def _timestamp_key(value: str) -> tuple[float, float] | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)", str(value or ""))
    if not match:
        return None
    return (round(float(match.group(1)), 3), round(float(match.group(2)), 3))


def inspect_answer_grounding(
    answer: str,
    evidence: Sequence[str | dict],
    evidence_items: Sequence[dict] | None = None,
    *,
    claim_support_threshold: float = 0.22,
) -> dict:
    evidence_tags, normalised_items = _normalise_evidence(evidence, evidence_items)
    if not evidence_tags:
        return {
            "ok": False,
            "reason": "missing evidence",
            "matched_evidence": [],
            "missing_evidence": [],
            "coverage": 0.0,
            "timestamp_refs": [],
            "matched_timestamp_refs": [],
            "unmatched_timestamp_refs": [],
            "claim_support_checked": bool(normalised_items),
            "claim_support_ok": False,
            "claim_support_coverage": 0.0,
            "claim_checks": [],
            "unsupported_claims": [],
        }
    matched = [item for item in evidence_tags if item in answer]
    missing = [item for item in evidence_tags if item not in answer]
    timestamp_refs = _TIMESTAMP_PATTERN.findall(answer)
    evidence_timestamp_keys = {
        key
        for item in evidence_tags
        if (key := _timestamp_key(item.replace("timestamp=", ""))) is not None
    }
    matched_timestamp_refs = [
        ref for ref in timestamp_refs if _timestamp_key(ref) in evidence_timestamp_keys
    ]
    unmatched_timestamp_refs = [
        ref for ref in timestamp_refs if _timestamp_key(ref) not in evidence_timestamp_keys
    ]
    coverage = len(matched) / max(1, len(evidence_tags))
    has_structure = _has_reasoning_structure(answer)
    claim_report = _inspect_claim_support(
        answer,
        normalised_items,
        threshold=claim_support_threshold,
    )
    claim_gate = not claim_report["checked"] or claim_report["ok"]
    ok = (
        len(answer.strip()) >= 10
        and has_structure
        and coverage > 0.0
        and bool(timestamp_refs)
        and not unmatched_timestamp_refs
        and claim_gate
    )
    if not ok:
        if len(answer.strip()) < 10:
            reason = "answer too short"
        elif not has_structure:
            reason = "missing reasoning structure"
        elif not timestamp_refs:
            reason = "answer missing timestamp reference"
        elif unmatched_timestamp_refs:
            reason = "answer contains unbound timestamp reference"
        elif claim_report["checked"] and not claim_report["ok"]:
            reason = "answer contains claim not supported by its timestamp evidence"
        else:
            reason = "answer missing evidence reference"
    else:
        suffix = (
            f"; claim_support={claim_report['coverage']:.2f}"
            if claim_report["checked"]
            else ""
        )
        reason = f"evidence attached; coverage={coverage:.2f}{suffix}"
    return {
        "ok": ok,
        "reason": reason,
        "matched_evidence": matched,
        "missing_evidence": missing,
        "coverage": coverage,
        "timestamp_refs": timestamp_refs,
        "matched_timestamp_refs": matched_timestamp_refs,
        "unmatched_timestamp_refs": unmatched_timestamp_refs,
        "claim_support_checked": claim_report["checked"],
        "claim_support_ok": claim_report["ok"],
        "claim_support_coverage": claim_report["coverage"],
        "claim_checks": claim_report["checks"],
        "unsupported_claims": claim_report["unsupported"],
    }


def verify_answer(
    answer: str,
    evidence: List[str | dict],
    evidence_items: Sequence[dict] | None = None,
) -> Tuple[bool, str]:
    result = inspect_answer_grounding(answer, evidence, evidence_items=evidence_items)
    if not result["ok"]:
        return False, str(result["reason"])
    tags, _ = _normalise_evidence(evidence, evidence_items)
    if any("timestamp=" not in item for item in tags):
        return False, "evidence missing timestamps"
    return True, str(result["reason"])


def _normalise_evidence(
    evidence: Sequence[str | dict],
    evidence_items: Sequence[dict] | None,
) -> tuple[list[str], list[dict]]:
    supplied_items = list(evidence_items or [])
    tags: list[str] = []
    inferred_items: list[dict] = []
    for item in evidence:
        if isinstance(item, dict):
            inferred_items.append(dict(item))
            try:
                tags.append(f"timestamp={float(item['start_sec']):.1f}-{float(item['end_sec']):.1f}")
            except (KeyError, TypeError, ValueError):
                continue
        else:
            tags.append(str(item))
    items = supplied_items or inferred_items
    normalised_items = []
    for item in items:
        try:
            start = float(item.get("start_sec"))
            end = float(item.get("end_sec"))
        except (AttributeError, TypeError, ValueError):
            continue
        if start < 0 or end <= start:
            continue
        text = str(
            item.get("text")
            or item.get("evidence_text")
            or item.get("summary")
            or ""
        ).strip()
        normalised_items.append(
            {
                "timestamp": f"{start:.1f}-{end:.1f}",
                "timestamp_key": (round(start, 3), round(end, 3)),
                "text": text,
            }
        )
    return list(dict.fromkeys(tags)), normalised_items


def _inspect_claim_support(answer: str, evidence_items: Sequence[dict], threshold: float) -> dict:
    if not evidence_items:
        return {"checked": False, "ok": True, "coverage": 1.0, "checks": [], "unsupported": []}
    by_timestamp = {item["timestamp_key"]: item for item in evidence_items if item.get("text")}
    if not by_timestamp:
        return {"checked": False, "ok": True, "coverage": 1.0, "checks": [], "unsupported": []}
    checks: list[dict] = []
    refusal = any(marker in answer for marker in _REFUSAL_MARKERS)
    for line in answer.splitlines():
        refs = _TIMESTAMP_PATTERN.findall(line)
        if not refs:
            continue
        cleaned = _clean_claim_line(line)
        clauses = _claim_clauses(cleaned)
        if not clauses and refusal:
            continue
        for ref in refs:
            evidence_item = by_timestamp.get(_timestamp_key(ref))
            if evidence_item is None:
                continue
            for clause in clauses:
                if any(marker in clause for marker in _REFUSAL_MARKERS):
                    continue
                score, overlaps = _support_score(clause, str(evidence_item["text"]))
                supported = bool(score >= threshold)
                checks.append(
                    {
                        "timestamp": ref,
                        "claim": clause,
                        "evidence_excerpt": str(evidence_item["text"])[:240],
                        "support_score": round(score, 6),
                        "overlap_units": overlaps[:12],
                        "supported": supported,
                    }
                )
    if not checks:
        return {
            "checked": True,
            "ok": refusal,
            "coverage": 1.0 if refusal else 0.0,
            "checks": [],
            "unsupported": [] if refusal else [{"claim": "no timestamp-bound factual claim detected"}],
        }
    supported_count = sum(bool(item["supported"]) for item in checks)
    unsupported = [item for item in checks if not item["supported"]]
    coverage = supported_count / len(checks)
    return {
        "checked": True,
        "ok": not unsupported,
        "coverage": coverage,
        "checks": checks,
        "unsupported": unsupported,
    }


def _clean_claim_line(line: str) -> str:
    value = _TIMESTAMP_PATTERN.sub("", line)
    value = re.sub(r"\(\s*\)", "", value)
    value = re.sub(r"^\s*[-*]\s*", "", value)
    value = re.sub(r"^\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?\s*[：:]?", "", value)
    value = re.sub(r"^(问题|结论|总体结论|证据)\s*[：:]", "", value)
    return value.strip(" ：:。.;；")


def _claim_clauses(value: str) -> list[str]:
    clauses = []
    for item in re.split(r"[，,；;。]+", value):
        clause = item.strip(" -：:。.;；")
        if len(re.sub(r"\s+", "", clause)) < 4:
            continue
        if clause in {"时间戳证据", "候选证据"}:
            continue
        clauses.append(clause)
    return clauses


def _support_score(claim: str, evidence: str) -> tuple[float, list[str]]:
    normal_claim = _normalise_text(claim)
    normal_evidence = _normalise_text(evidence)
    if len(normal_claim) >= 4 and normal_claim in normal_evidence:
        return 1.0, [normal_claim[:32]]
    claim_units = _semantic_units(claim)
    evidence_units = _semantic_units(evidence)
    if not claim_units or not evidence_units:
        return 0.0, []
    overlaps = sorted(claim_units & evidence_units, key=lambda item: (-len(item), item))
    required = max(1, math.ceil(len(claim_units) * 0.18))
    if len(overlaps) < required:
        return len(overlaps) / max(1, len(claim_units)), overlaps
    return len(overlaps) / max(1, len(claim_units)), overlaps


def _semantic_units(value: str) -> set[str]:
    units = {token.lower() for token in _ASCII_PATTERN.findall(value) if len(token) >= 2}
    for sequence in _CJK_PATTERN.findall(value):
        if len(sequence) == 1:
            continue
        if len(sequence) == 2:
            units.add(sequence)
            continue
        units.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return {unit for unit in units if unit not in _GENERIC_UNITS and not unit.isdigit()}


def _normalise_text(value: str) -> str:
    return re.sub(r"[^\u3400-\u9fffa-zA-Z0-9]+", "", value).lower()


def _has_reasoning_structure(answer: str) -> bool:
    answer_l = answer.lower()
    has_legacy_trace = "retrieve:" in answer_l and "synthesize:" in answer_l
    has_agent_answer = (
        ("问题：" in answer and "结论：" in answer)
        or ("用户问题：" in answer and "总体结论：" in answer)
        or ("question:" in answer_l and "answer:" in answer_l)
    )
    return has_legacy_trace or has_agent_answer
