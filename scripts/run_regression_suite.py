from __future__ import annotations

"""Run a small, frozen product regression and emit actionable error classes."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from videomemo.config import VideoMemoConfig
from videomemo.eval.reproducibility import file_sha256, source_fingerprint
from videomemo.pipeline import VideoMemoPipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-run-regression-suite")
    parser.add_argument("--cases", default="data/regression_cases.json")
    parser.add_argument("--config", default="configs/iboy_qwen35.yaml")
    parser.add_argument(
        "--adapter",
        default="",
        help="optional explicit hash-checked adapter path for an isolated experiment; does not alter the Web registry",
    )
    parser.add_argument("--output", default="outputs/reports/error_analysis.json")
    args = parser.parse_args()

    case_path = _rooted(args.cases)
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    video_path = _rooted(str(payload["video_path"]))
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    config = VideoMemoConfig.load(str(_rooted(args.config)))
    if args.adapter:
        adapter = _rooted(args.adapter)
        if not (adapter / "adapter_config.json").is_file() or not (
            adapter / "adapter_model.safetensors"
        ).is_file():
            raise FileNotFoundError(f"adapter is incomplete: {adapter}")
        config.llm_adapter_path = str(adapter)
    pipeline = VideoMemoPipeline(config)
    results = []
    for case in payload.get("cases", []):
        pack = pipeline.run(str(video_path), query=str(case["query"]))
        results.append(_evaluate_case(case, pack))

    counts: dict[str, int] = {}
    for result in results:
        category = result["primary_error_category"]
        counts[category] = counts.get(category, 0) + 1
    report = {
        "schema_version": "videotrace-error-analysis-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "frozen task-local cola regression; not a public benchmark",
        "case_manifest": str(case_path),
        "case_manifest_sha256": file_sha256(case_path),
        "video_path": str(video_path),
        "video_sha256": file_sha256(video_path),
        "source_sha256": source_fingerprint(ROOT),
        "resolved_config": config.dump(),
        "num_cases": len(results),
        "num_passed": sum(result["passed"] for result in results),
        "error_category_counts": counts,
        "error_taxonomy": {
            "retrieval_error": "No selected evidence window overlaps a required gold span.",
            "visual_understanding_error": "The right window was selected but required visible facts are absent from its structured description.",
            "temporal_coverage_error": "A global-process question missed one or more opening/middle/ending spans.",
            "generation_error": "Evidence is present but the answer omits required facts or fails to refuse unsupported content.",
            "claim_support_error": "A timestamp is valid but one or more factual clauses are not supported by that window's evidence text.",
            "verifier_miss": "The verifier accepted an answer that still violates the case-level expectation.",
            "none": "All task-local evidence, generation and verification checks passed."
        },
        "cases": results,
    }
    output = _rooted(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _evaluate_case(case: dict, pack) -> dict:
    timeline = list(pack.timeline)
    answer = str(pack.answer or "")
    answer_body = "\n".join(answer.splitlines()[1:])
    expected = str(case.get("expected_behavior", "answer"))
    gold_spans = [tuple(map(float, span)) for span in case.get("gold_spans", [])]
    selected_spans = [
        (float(item.get("start_sec", 0.0)), float(item.get("end_sec", 0.0)))
        for item in timeline
    ]
    overlap_by_gold = [any(_overlap(gold, selected) for selected in selected_spans) for gold in gold_spans]
    retrieval_ok = all(overlap_by_gold) if gold_spans else True
    temporal_ok = retrieval_ok if case.get("case_type") == "global_process" else True
    evidence_items = [
        item
        for item in timeline
        if not gold_spans or any(
            _overlap(
                gold,
                (float(item.get("start_sec", 0.0)), float(item.get("end_sec", 0.0))),
            )
            for gold in gold_spans
        )
    ]
    evidence_text = "\n".join(
        str(item.get("text", ""))
        for item in evidence_items
    )
    expected_keywords = [str(item) for item in case.get("expected_keywords", [])]
    if expected == "abstain":
        visual_ok = True
        generation_ok = any(marker in answer_body for marker in ("证据不足", "无法确认", "不能确认"))
        generation_ok = generation_ok and not any(
            assertion in answer_body for assertion in case.get("forbidden_assertions", [])
        )
    else:
        if case.get("case_type") == "global_process":
            # Global questions are judged by stage coverage plus distributed
            # visual support.  A local caption need not literally repeat a
            # high-level abstraction such as "盲测", but every expected visual
            # concept still needs support somewhere in the selected windows.
            # This keeps the visual-understanding error class meaningful.
            visual_ok = bool(evidence_items) and all(
                _global_visual_support(keyword, evidence_items)
                for keyword in expected_keywords
            )
        else:
            visual_ok = all(keyword in evidence_text for keyword in expected_keywords)
        generation_ok = all(keyword in answer for keyword in expected_keywords)
    agent = dict(pack.metadata.get("agent_run") or {})
    verification = dict(agent.get("verification") or {})
    claim_support_ok = bool(verification.get("claim_support_ok", True))
    verifier_ok = (
        bool(agent.get("verified"))
        and not verification.get("unmatched_timestamp_refs")
        and claim_support_ok
    )
    primary = "none"
    if not retrieval_ok:
        primary = "retrieval_error"
    elif not temporal_ok:
        primary = "temporal_coverage_error"
    elif not visual_ok:
        primary = "visual_understanding_error"
    elif not generation_ok and verifier_ok:
        primary = "verifier_miss"
    elif not generation_ok:
        primary = "generation_error"
    elif not claim_support_ok:
        primary = "claim_support_error"
    elif not verifier_ok:
        primary = "generation_error"
    return {
        "case_id": case.get("case_id"),
        "case_type": case.get("case_type"),
        "query": case.get("query"),
        "expected_behavior": expected,
        "gold_spans": case.get("gold_spans", []),
        "selected_spans": [list(span) for span in selected_spans],
        "selected_segment_ids": [str(item.get("segment_id", "")) for item in timeline],
        "retrieval_ok": retrieval_ok,
        "visual_understanding_ok": visual_ok,
        "temporal_coverage_ok": temporal_ok,
        "generation_ok": generation_ok,
        "verifier_ok": verifier_ok,
        "verification_coverage": float(verification.get("coverage", 0.0)),
        "claim_support_ok": claim_support_ok,
        "claim_support_coverage": float(verification.get("claim_support_coverage", 0.0)),
        "unsupported_claims": list(verification.get("unsupported_claims") or []),
        "grounding_decision": agent.get("grounding_decision", {}),
        "primary_error_category": primary,
        "passed": primary == "none",
        "answer": answer,
        "timeline": timeline,
    }


def _global_visual_support(keyword: str, items: list[dict]) -> bool:
    """Check a global visual concept across distributed evidence windows."""
    needle = str(keyword or "").strip().lower()
    if not needle:
        return True
    aliases = {
        "盲测": ("盲测", "盲猜", "眼罩", "蒙眼", "猜测", "品尝"),
        "展示": ("展示", "举起", "拿起", "介绍", "陈列"),
    }.get(needle, (needle,))
    for item in items:
        text = " ".join(
            [
                str(item.get("text", "") or ""),
                str(item.get("caption", "") or ""),
                str(item.get("ocr_text", "") or ""),
                " ".join(str(value) for value in (item.get("entities") or [])),
                " ".join(str(value) for value in (item.get("actions") or [])),
            ]
        ).lower()
        if any(alias in text for alias in aliases):
            return True
    return False


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def _rooted(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
