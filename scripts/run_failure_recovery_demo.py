from __future__ import annotations

"""Deterministic现场演示 of retry, circuit breaking and controlled fallback."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from videomemo.agent import AgentToolRegistry, ToolCircuitBreaker, ToolSpec
from videomemo.agent.tool_registry import ToolCircuitOpenError


def main() -> None:
    attempts = {"count": 0}
    breaker = ToolCircuitBreaker(failure_threshold=2, recovery_sec=3600)
    registry = AgentToolRegistry(default_max_attempts=2, circuit_breaker=breaker)

    def flaky_retrieval(query: str) -> dict:
        attempts["count"] += 1
        # Two registry calls each get two bounded attempts.  The fourth
        # failure trips the per-tool circuit; the third call must degrade
        # without invoking the handler again.
        if attempts["count"] <= 4:
            raise TimeoutError("SigLIP index timeout (simulated)")
        return {"status": "ok", "segment_id": "seg-0000", "query": query}

    registry.register(
        ToolSpec(
            name="retrieve_segments",
            description="simulated retrieval for recovery demo",
            input_keys=["query"],
            output_keys=["status", "segment_id", "query"],
            input_schema={"query": "str"},
            max_attempts=2,
        ),
        flaky_retrieval,
    )
    events: list[dict] = []
    for call_index in range(1, 4):
        try:
            result = registry.call("retrieve_segments", query="最后盲测在哪里？")
            events.append({"call": call_index, "outcome": "recovered", "result": result})
        except ToolCircuitOpenError as exc:
            events.append({"call": call_index, "outcome": "controlled_fallback", "error": str(exc)})
            break
        except Exception as exc:
            events.append({"call": call_index, "outcome": "retry_exhausted", "error": f"{type(exc).__name__}: {exc}"})

    report = {
        "schema_version": "videotrace-agent-failure-recovery-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": "retrieval timeout -> bounded retry -> circuit open -> controlled fallback",
        "simulated_failure_isolated": True,
        "attempts": attempts["count"],
        "events": events,
        "tool_trace": registry.trace(),
        "safeguards": registry.safeguards(),
        "final_action": "return a user-visible evidence-insufficient state; do not fabricate an answer",
        "passed": (
            any(event["outcome"] == "retry_exhausted" for event in events)
            and any(event["outcome"] == "controlled_fallback" for event in events)
            and any(record["error_code"] == "circuit_open" and record["attempts"] == 0 for record in registry.trace())
            and any(record["error_code"] == "execution_error" and record["attempts"] == 2 for record in registry.trace())
        ),
    }
    output = ROOT / "outputs" / "reports" / "agent_failure_recovery.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
