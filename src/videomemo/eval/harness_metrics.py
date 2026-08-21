from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HarnessEvalResult:
    plan_coverage: float
    tool_success_rate: float
    step_order_score: float
    evidence_binding_score: float
    answer_structure_score: float
    verification_score: float
    trace_completeness: float
    score: float

    def dump(self) -> dict:
        return self.__dict__


def evaluate_harness_run(agent_run: dict) -> HarnessEvalResult:
    plan = agent_run.get("plan", [])
    tool_trace = agent_run.get("tool_trace", [])
    verification = agent_run.get("verification", {})
    answer = str(agent_run.get("answer", "") or "")

    expected = [
        "retrieve_segments",
        "build_context",
        "assess_evidence",
        "search_memory",
        "synthesize_answer",
        "verify_answer",
    ]
    observed_actions = [str(step.get("action", "")) for step in plan]
    plan_coverage = len([action for action in expected if action in observed_actions]) / max(1, len(expected))

    tool_success_rate = (
        sum(1 for call in tool_trace if call.get("ok", True)) / len(tool_trace)
        if tool_trace
        else 0.0
    )
    step_order_score = _score_step_order(observed_actions, expected)
    evidence_binding_score = float(verification.get("coverage", 0.0))
    verification_score = 1.0 if agent_run.get("verified") else 0.0
    answer_structure_score = _score_answer_structure(answer)
    trace_completeness = _score_trace_completeness(plan, tool_trace)

    score = (
        0.18 * plan_coverage
        + 0.18 * tool_success_rate
        + 0.12 * step_order_score
        + 0.22 * evidence_binding_score
        + 0.12 * answer_structure_score
        + 0.10 * verification_score
        + 0.08 * trace_completeness
    )
    return HarnessEvalResult(
        plan_coverage=plan_coverage,
        tool_success_rate=tool_success_rate,
        step_order_score=step_order_score,
        evidence_binding_score=evidence_binding_score,
        answer_structure_score=answer_structure_score,
        verification_score=verification_score,
        trace_completeness=trace_completeness,
        score=score,
    )


def _score_step_order(observed: list[str], expected: list[str]) -> float:
    if not observed:
        return 0.0
    index = 0
    matches = 0
    for action in observed:
        while index < len(expected) and expected[index] != action:
            index += 1
        if index >= len(expected):
            break
        matches += 1
        index += 1
    return matches / max(1, len(expected))


def _score_answer_structure(answer: str) -> float:
    if not answer.strip():
        return 0.0
    has_question = "问题：" in answer or "用户问题：" in answer
    has_conclusion = "结论：" in answer or "总体结论：" in answer
    has_timestamp = "timestamp=" in answer
    evidence_lines = sum(1 for line in answer.splitlines() if line.strip().startswith("- "))
    score = 0.0
    score += 0.35 if has_question else 0.0
    score += 0.35 if has_conclusion else 0.0
    score += 0.20 if has_timestamp else 0.0
    score += 0.10 if evidence_lines >= 2 else 0.0
    return min(1.0, score)


def _score_trace_completeness(plan: list[dict], tool_trace: list[dict]) -> float:
    if not plan:
        return 0.0
    plan_actions = {str(step.get("action", "")) for step in plan}
    trace_actions = {str(call.get("name", "")) for call in tool_trace}
    return len(plan_actions & trace_actions) / max(1, len(plan_actions))
