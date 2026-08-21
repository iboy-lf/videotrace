from __future__ import annotations

import pytest

from videomemo.agent import (
    AgentHarness,
    AgentToolRegistry,
    ToolCircuitBreaker,
    ToolCircuitOpenError,
    ToolContractError,
    ToolSpec,
)


def _spec(**kwargs) -> ToolSpec:
    defaults = {
        "name": "lookup",
        "description": "test tool",
        "input_keys": ["query"],
        "output_keys": ["value"],
        "input_schema": {"query": "str"},
    }
    defaults.update(kwargs)
    return ToolSpec(**defaults)


def test_tool_registry_validates_input_contract_and_records_error():
    registry = AgentToolRegistry()
    registry.register(_spec(), lambda query: {"value": query})
    with pytest.raises(ToolContractError, match="missing required inputs"):
        registry.call("lookup")
    record = registry.trace()[-1]
    assert record["status"] == "error"
    assert record["error_code"] == "invalid_input"
    assert record["attempts"] == 0
    assert record["attempt_trace"][0]["status"] == "blocked"
    assert record["attempt_trace"][0]["error_code"] == "invalid_input"
    assert record["call_id"]
    assert record["started_at_utc"]
    assert record["finished_at_utc"]


def test_tool_registry_retries_transient_failure_once():
    calls = 0

    def handler(query):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary")
        return {"value": query.upper()}

    registry = AgentToolRegistry(default_max_attempts=1)
    registry.register(_spec(max_attempts=2, retry_backoff_sec=0.0), handler)
    assert registry.call("lookup", query="ok") == {"value": "OK"}
    record = registry.trace()[-1]
    assert calls == 2
    assert record["attempts"] == 2
    assert record["response"]["metadata"]["attempts"] == 2
    assert [attempt["status"] for attempt in record["attempt_trace"]] == ["error", "success"]
    assert record["attempt_trace"][0]["error_code"] == "execution_error"
    assert record["attempt_trace"][0]["will_retry"] is True
    assert record["attempt_trace"][1]["ok"] is True


def test_tool_registry_circuit_breaker_blocks_after_consecutive_failures():
    calls = 0

    def handler(query):
        nonlocal calls
        calls += 1
        raise RuntimeError("down")

    breaker = ToolCircuitBreaker(failure_threshold=2, recovery_sec=60.0)
    registry = AgentToolRegistry(circuit_breaker=breaker)
    registry.register(_spec(), handler)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="down"):
            registry.call("lookup", query="x")
    with pytest.raises(ToolCircuitOpenError):
        registry.call("lookup", query="x")
    assert calls == 2
    assert registry.trace()[-1]["error_code"] == "circuit_open"
    assert registry.trace()[-1]["attempt_trace"][0]["status"] == "blocked"
    assert breaker.status("lookup")["state"] == "open"


def test_tool_registry_rejects_invalid_output_contract():
    registry = AgentToolRegistry()
    registry.register(_spec(), lambda query: {"other": query})
    with pytest.raises(ToolContractError, match="output missing fields"):
        registry.call("lookup", query="x")
    assert registry.trace()[-1]["error_code"] == "invalid_output"


def _register_plan_tools(harness: AgentHarness, failing_action: str = "") -> None:
    def handler(action, output):
        def run(**_kwargs):
            if action == failing_action:
                raise RuntimeError(f"{action} unavailable")
            return output

        return run

    candidates = [
        {
            "segment_id": "seg-0001",
            "start_sec": 1.0,
            "end_sec": 2.0,
            "text": "evidence",
            "score": 1.0,
        }
    ]
    context = {
        "items": candidates,
        "evidence_tags": ["timestamp=1.0-2.0"],
        "used_chars": 8,
        "dropped_segment_ids": [],
    }
    decision = {
        "sufficient": True,
        "reason": "enough evidence",
        "max_query_coverage": 1.0,
        "max_vlm_score": 1.0,
    }
    tools = [
        ("retrieve_segments", ["query"], candidates),
        ("build_context", ["query", "candidates"], context),
        ("assess_evidence", ["query", "context"], decision),
        ("search_memory", ["query"], []),
        (
            "synthesize_answer",
            ["query", "context", "memory_hits", "grounding_decision"],
            "结论：有证据的回答\n- 1.0s-2.0s：evidence",
        ),
        (
            "verify_answer",
            ["query", "answer", "evidence_tags", "evidence_items", "grounding_decision"],
            {"ok": True, "reason": "verified", "coverage": 1.0},
        ),
    ]
    for name, input_keys, output in tools:
        harness.register_tool(
            ToolSpec(
                name=name,
                description=f"test {name}",
                input_keys=input_keys,
                output_keys=[],
            ),
            handler(name, output),
        )


def test_plan_execute_fails_closed_with_trace_when_retrieval_crashes():
    harness = AgentHarness()
    _register_plan_tools(harness, failing_action="retrieve_segments")

    run = harness.run_plan_execute("发生了什么？")

    assert run.status == "safe_refusal"
    assert run.degraded is True
    assert run.verified is False
    assert "安全" not in run.answer
    assert "不输出事实性回答" in run.answer
    assert run.verification["safe_refusal"] is True
    assert run.recovery_events[0]["action"] == "retrieve_segments"
    assert run.recovery_events[0]["policy"] == "fail_closed_safe_refusal"
    assert run.tool_trace[-1]["ok"] is False
    assert run.tool_trace[-1]["error_code"] == "execution_error"


def test_plan_execute_continues_without_optional_memory_and_records_recovery():
    harness = AgentHarness()
    _register_plan_tools(harness, failing_action="search_memory")

    run = harness.run_plan_execute("发生了什么？")

    assert run.status == "completed_with_recovery"
    assert run.degraded is True
    assert run.verified is True
    assert len(run.tool_trace) == 6
    assert run.tool_trace[3]["name"] == "search_memory"
    assert run.tool_trace[3]["ok"] is False
    assert run.recovery_events == run.safeguards["recovery_events"]
    assert run.recovery_events[0]["policy"] == "continue_without_memory"


def test_plan_execute_replaces_unverified_generation_when_verifier_crashes():
    harness = AgentHarness()
    _register_plan_tools(harness, failing_action="verify_answer")

    run = harness.run_plan_execute("发生了什么？")

    assert run.status == "safe_refusal"
    assert run.verified is False
    assert "有证据的回答" not in run.answer
    assert run.verification["failure_action"] == "verify_answer"
    assert run.tool_trace[-1]["name"] == "verify_answer"
    assert run.tool_trace[-1]["ok"] is False
