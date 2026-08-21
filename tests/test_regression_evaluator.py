from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("videotrace_regression_script", ROOT / "scripts" / "run_regression_suite.py")
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_global_process_evaluator_uses_stage_coverage_not_literal_visual_abstraction():
    case = {
        "case_id": "global",
        "case_type": "global_process",
        "query": "概括开场、中段和最后盲测。",
        "expected_behavior": "answer",
        "gold_spans": [[0.0, 20.0], [180.0, 320.0], [400.0, 416.2]],
        "expected_keywords": ["展示", "盲测"],
    }
    pack = SimpleNamespace(
        timeline=[
            {"segment_id": "opening", "start_sec": 0.0, "end_sec": 20.0, "text": "男子介绍多罐可乐。"},
            {"segment_id": "middle", "start_sec": 200.0, "end_sec": 220.0, "text": "男子倒入杯中并试喝。"},
            {
                "segment_id": "ending",
                "start_sec": 400.0,
                "end_sec": 416.2,
                "text": "男子戴眼罩，桌上摆着饮料。",
                "entities": ["眼罩", "饮料", "桌子"],
            },
        ],
        answer=(
            "问题：整体流程？\n结论：开场展示产品，中段试喝，最后进行盲测。\n"
            "- 0.0-20.0：展示产品 (timestamp=0.0-20.0)\n"
            "- 200.0-220.0：试喝 (timestamp=200.0-220.0)\n"
            "- 400.0-416.2：盲测 (timestamp=400.0-416.2)"
        ),
        metadata={
            "agent_run": {
                "verified": True,
                "verification": {"coverage": 1.0, "unmatched_timestamp_refs": []},
                "grounding_decision": {"sufficient": True},
            }
        },
    )
    result = MODULE._evaluate_case(case, pack)
    assert result["temporal_coverage_ok"]
    assert result["visual_understanding_ok"]
    assert result["generation_ok"]
    assert result["primary_error_category"] == "none"


def test_global_process_evaluator_still_detects_missing_stage():
    case = {
        "case_id": "global-missing-ending",
        "case_type": "global_process",
        "query": "概括开场、中段和结尾。",
        "expected_behavior": "answer",
        "gold_spans": [[0.0, 20.0], [180.0, 320.0], [400.0, 416.2]],
        "expected_keywords": ["展示", "盲测"],
    }
    pack = SimpleNamespace(
        timeline=[
            {"segment_id": "opening", "start_sec": 0.0, "end_sec": 20.0, "text": "展示可乐。"},
            {"segment_id": "middle", "start_sec": 200.0, "end_sec": 220.0, "text": "试喝可乐。"},
        ],
        answer="问题：流程？\n结论：展示后盲测。\n- 0.0-20.0：展示 (timestamp=0.0-20.0)",
        metadata={"agent_run": {"verified": True, "verification": {"unmatched_timestamp_refs": []}}},
    )
    result = MODULE._evaluate_case(case, pack)
    assert not result["retrieval_ok"]
    assert result["primary_error_category"] == "retrieval_error"


def test_global_process_evaluator_keeps_visual_error_when_stage_facts_are_absent():
    case = {
        "case_id": "global-visual-miss",
        "case_type": "global_process",
        "query": "概括开场、中段和最后盲测。",
        "expected_behavior": "answer",
        "gold_spans": [[0.0, 20.0], [180.0, 320.0], [400.0, 416.2]],
        "expected_keywords": ["展示", "盲测"],
    }
    pack = SimpleNamespace(
        timeline=[
            {"segment_id": "opening", "start_sec": 0.0, "end_sec": 20.0, "text": "室内人物说话。"},
            {"segment_id": "middle", "start_sec": 200.0, "end_sec": 220.0, "text": "室内人物说话。"},
            {"segment_id": "ending", "start_sec": 400.0, "end_sec": 416.2, "text": "室内人物说话。"},
        ],
        answer=(
            "问题：流程？\n结论：开场展示产品，中段试喝，最后进行盲测。\n"
            "- 0.0-20.0：证据 (timestamp=0.0-20.0)\n"
            "- 200.0-220.0：证据 (timestamp=200.0-220.0)\n"
            "- 400.0-416.2：证据 (timestamp=400.0-416.2)"
        ),
        metadata={"agent_run": {"verified": True, "verification": {"unmatched_timestamp_refs": []}}},
    )
    result = MODULE._evaluate_case(case, pack)
    assert result["retrieval_ok"]
    assert not result["visual_understanding_ok"]
    assert result["primary_error_category"] == "visual_understanding_error"
