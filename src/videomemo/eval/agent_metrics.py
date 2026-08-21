from __future__ import annotations

from dataclasses import dataclass

from ..models import KnowledgePack


@dataclass
class AgentEvalResult:
    verified: float
    evidence_reference_coverage: float
    context_keep_rate: float
    memory_hit_rate: float
    tool_success_rate: float
    tool_call_count: int
    score: float

    def dump(self) -> dict:
        return self.__dict__


def evaluate_agent_run(pack: KnowledgePack) -> AgentEvalResult:
    agent_run = pack.metadata.get("agent_run", {})
    context = agent_run.get("context", {})
    items = context.get("items", [])
    dropped = context.get("dropped_segment_ids", [])
    memory_hits = agent_run.get("memory_hits", [])
    tool_trace = agent_run.get("tool_trace", [])
    evidence_reference_coverage = float(agent_run.get("verification", {}).get("coverage", 0.0))
    if evidence_reference_coverage == 0.0 and agent_run.get("verified"):
        evidence_reference_coverage = 1.0

    total_context_candidates = len(items) + len(dropped)
    context_keep_rate = len(items) / max(1, total_context_candidates)
    memory_hit_rate = min(1.0, len(memory_hits) / 3.0)
    tool_success_rate = (
        sum(1 for call in tool_trace if call.get("ok", True)) / len(tool_trace)
        if tool_trace
        else 0.0
    )
    verified = 1.0 if agent_run.get("verified") else 0.0
    score = (
        0.25 * verified
        + 0.25 * evidence_reference_coverage
        + 0.20 * context_keep_rate
        + 0.20 * tool_success_rate
        + 0.10 * memory_hit_rate
    )
    return AgentEvalResult(
        verified=verified,
        evidence_reference_coverage=evidence_reference_coverage,
        context_keep_rate=context_keep_rate,
        memory_hit_rate=memory_hit_rate,
        tool_success_rate=tool_success_rate,
        tool_call_count=len(tool_trace),
        score=score,
    )
