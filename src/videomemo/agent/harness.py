from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .tool_registry import AgentToolRegistry, ToolCircuitBreaker, ToolSpec


@dataclass
class HarnessStep:
    step: int
    action: str
    observation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def dump(self) -> dict:
        return self.__dict__


@dataclass
class HarnessRun:
    mode: str
    answer: str
    plan: list[dict]
    context: dict
    grounding_decision: dict
    memory_hits: list[dict]
    verified: bool
    verification_reason: str
    verification: dict = field(default_factory=dict)
    tool_trace: list[dict] = field(default_factory=list)
    safeguards: dict = field(default_factory=dict)
    status: str = "completed"
    degraded: bool = False
    recovery_events: list[dict] = field(default_factory=list)

    def dump(self) -> dict:
        return self.__dict__


class AgentHarness:
    """VideoTrace 的任务编排层：注册工具、执行计划、收集 trace 和守卫结果。"""

    def __init__(
        self,
        max_steps: int = 6,
        tool_max_attempts: int = 1,
        tool_retry_backoff_sec: float = 0.0,
        circuit_breaker: ToolCircuitBreaker | None = None,
    ):
        self.max_steps = max_steps
        self.registry = AgentToolRegistry(
            default_max_attempts=tool_max_attempts,
            default_retry_backoff_sec=tool_retry_backoff_sec,
            circuit_breaker=circuit_breaker,
        )

    def register_tool(self, spec: ToolSpec, handler: Callable[..., Any]) -> None:
        self.registry.register(spec, handler)

    def call(self, name: str, **kwargs: Any) -> Any:
        return self.registry.call(name, **kwargs)

    def trace(self) -> list[dict]:
        return self.registry.trace()

    def run_plan_execute(self, query: str) -> HarnessRun:
        plan = [
            HarnessStep(step=1, action="retrieve_segments", observation="先找和问题最相关的视频片段。"),
            HarnessStep(step=2, action="build_context", observation="把候选片段压缩成可放入模型上下文的证据窗口。"),
            HarnessStep(step=3, action="assess_evidence", observation="判断证据是否足以回答，必要时拒答。"),
            HarnessStep(step=4, action="search_memory", observation="从历史片段记忆里补充可复用事实。"),
            HarnessStep(step=5, action="synthesize_answer", observation="基于证据生成回答和知识包摘要。"),
            HarnessStep(step=6, action="verify_answer", observation="检查回答是否带时间戳证据。"),
        ]
        candidates: list[dict] = []
        context: dict = {}
        grounding_decision: dict = {}
        memory_hits: list[dict] = []
        recovery_events: list[dict] = []

        try:
            candidates = self.call("retrieve_segments", query=query)
        except Exception as exc:
            return self._safe_refusal_run(
                query=query,
                plan=plan,
                action="retrieve_segments",
                exc=exc,
                context=context,
                grounding_decision=grounding_decision,
                memory_hits=memory_hits,
                recovery_events=recovery_events,
            )
        try:
            context = self.call("build_context", query=query, candidates=candidates)
        except Exception as exc:
            return self._safe_refusal_run(
                query=query,
                plan=plan,
                action="build_context",
                exc=exc,
                context=context,
                grounding_decision=grounding_decision,
                memory_hits=memory_hits,
                recovery_events=recovery_events,
            )
        try:
            grounding_decision = self.call("assess_evidence", query=query, context=context)
        except Exception as exc:
            return self._safe_refusal_run(
                query=query,
                plan=plan,
                action="assess_evidence",
                exc=exc,
                context=context,
                grounding_decision=grounding_decision,
                memory_hits=memory_hits,
                recovery_events=recovery_events,
            )
        context["grounding_decision"] = grounding_decision
        try:
            memory_hits = self.call("search_memory", query=query)
        except Exception as exc:
            memory_hits = []
            recovery_events.append(
                self._recovery_event(
                    "search_memory",
                    exc,
                    policy="continue_without_memory",
                    message="视频记忆工具异常，已仅使用当前视频证据继续执行。",
                )
            )
        try:
            answer = self.call(
                "synthesize_answer",
                query=query,
                context=context,
                memory_hits=memory_hits,
                grounding_decision=grounding_decision,
            )
        except Exception as exc:
            return self._safe_refusal_run(
                query=query,
                plan=plan,
                action="synthesize_answer",
                exc=exc,
                context=context,
                grounding_decision=grounding_decision,
                memory_hits=memory_hits,
                recovery_events=recovery_events,
            )
        try:
            verified = self.call(
                "verify_answer",
                query=query,
                answer=answer,
                evidence_tags=context.get("evidence_tags", []),
                evidence_items=context.get("items", []),
                grounding_decision=grounding_decision,
            )
        except Exception as exc:
            return self._safe_refusal_run(
                query=query,
                plan=plan,
                action="verify_answer",
                exc=exc,
                context=context,
                grounding_decision=grounding_decision,
                memory_hits=memory_hits,
                recovery_events=recovery_events,
            )
        return HarnessRun(
            mode="plan_execute",
            answer=answer,
            plan=[step.dump() for step in plan],
            context=context,
            grounding_decision=grounding_decision,
            memory_hits=memory_hits,
            verified=bool(verified["ok"]),
            verification_reason=str(verified["reason"]),
            verification=verified,
            tool_trace=self.trace(),
            safeguards=self._plan_execute_safeguards(recovery_events),
            status="completed_with_recovery" if recovery_events else "completed",
            degraded=bool(recovery_events),
            recovery_events=recovery_events,
        )

    def _safe_refusal_run(
        self,
        *,
        query: str,
        plan: list[HarnessStep],
        action: str,
        exc: Exception,
        context: dict,
        grounding_decision: dict,
        memory_hits: list[dict],
        recovery_events: list[dict],
    ) -> HarnessRun:
        labels = {
            "retrieve_segments": "检索候选片段",
            "build_context": "构建证据上下文",
            "assess_evidence": "判断证据充分性",
            "search_memory": "检索视频记忆",
            "synthesize_answer": "生成证据回答",
            "verify_answer": "校验回答证据",
        }
        event = self._recovery_event(
            action,
            exc,
            policy="fail_closed_safe_refusal",
            message=f"{labels.get(action, action)}异常，已停止事实性生成并安全拒答。",
        )
        events = [*recovery_events, event]
        safe_context = dict(context or {})
        safe_context.setdefault("items", [])
        safe_context.setdefault("evidence_tags", [])
        safe_context.setdefault("used_chars", 0)
        safe_context.setdefault("dropped_segment_ids", [])
        decision = dict(grounding_decision or {})
        if not decision:
            decision = {
                "sufficient": False,
                "reason": "Agent 关键工具异常，证据链未完整建立。",
                "max_query_coverage": 0.0,
                "max_vlm_score": 0.0,
            }
        decision["execution_blocked"] = True
        decision["failure_action"] = action
        safe_context["grounding_decision"] = decision
        error_code = str(event.get("error_code") or "execution_error")
        verification = {
            "ok": False,
            "reason": f"safe refusal after {action} failure",
            "coverage": 0.0,
            "timestamp_refs": [],
            "matched_timestamp_refs": [],
            "matched_evidence": [],
            "unmatched_timestamp_refs": [],
            "claim_support_ok": True,
            "claim_support_coverage": 1.0,
            "unsupported_claims": [],
            "calibrated_verifier": {
                "enabled": False,
                "passed": True,
                "reason": "generation blocked before a factual answer could be accepted",
            },
            "calibrated_verifier_ok": True,
            "safe_refusal": True,
            "failure_action": action,
            "error_code": error_code,
        }
        answer = (
            f"问题：{query}\n"
            f"结论：当前在“{labels.get(action, action)}”阶段发生工具异常。"
            "为避免在证据链不完整时产生幻觉，本次不输出事实性回答，请稍后重试。"
        )
        return HarnessRun(
            mode="plan_execute",
            answer=answer,
            plan=[step.dump() for step in plan],
            context=safe_context,
            grounding_decision=decision,
            memory_hits=list(memory_hits or []),
            verified=False,
            verification_reason=str(verification["reason"]),
            verification=verification,
            tool_trace=self.trace(),
            safeguards=self._plan_execute_safeguards(events),
            status="safe_refusal",
            degraded=True,
            recovery_events=events,
        )

    def _recovery_event(
        self,
        action: str,
        exc: Exception,
        *,
        policy: str,
        message: str,
    ) -> dict:
        record = next(
            (item for item in reversed(self.trace()) if str(item.get("name")) == action),
            {},
        )
        error_code = str(record.get("error_code") or "execution_error")
        return {
            "action": action,
            "policy": policy,
            "message": message,
            "error_code": error_code,
            "error_type": type(exc).__name__,
            "attempts": int(record.get("attempts") or 0),
            "circuit_state": str(record.get("circuit_state") or "closed"),
            "retryable": error_code not in {"invalid_input", "invalid_output"},
        }

    def _plan_execute_safeguards(self, recovery_events: list[dict]) -> dict:
        return {
            "max_steps": self.max_steps,
            "path_oscillation_control": "Plan-Execute 模式固定工具顺序，每个工具只执行一次，避免来回检索震荡。",
            "context_policy": "优先保留 segment_id、timestamp、score、evidence_text，超预算时只丢低分片段。",
            "degradation_policy": "关键证据工具失败时 fail-closed 安全拒答；视频记忆失败时记录 trace 并仅依赖当前视频证据继续。",
            "recovery_events": list(recovery_events),
            "tool_runtime": self.registry.safeguards(),
        }

    def run_react_like(self, query: str) -> HarnessRun:
        visited: set[tuple[str, str]] = set()
        plan: list[dict[str, Any]] = []
        candidates: list[dict] = []
        context: dict = {}
        grounding_decision: dict = {}
        memory_hits: list[dict] = []
        memory_searched = False
        answer = ""
        verified = {"ok": False, "reason": "not verified"}

        for step_idx in range(1, self.max_steps + 1):
            if not candidates:
                action = "retrieve_segments"
                key = (action, query)
                if key in visited:
                    break
                visited.add(key)
                candidates = self.call(action, query=query)
                plan.append({"step": step_idx, "action": action, "observation": f"取回 {len(candidates)} 个候选片段。"})
                continue
            if not context:
                action = "build_context"
                key = (action, ",".join(item["segment_id"] for item in candidates))
                if key in visited:
                    break
                visited.add(key)
                context = self.call(action, query=query, candidates=candidates)
                plan.append({"step": step_idx, "action": action, "observation": f"上下文使用 {context['used_chars']} 个字符。"})
                continue
            if not grounding_decision:
                action = "assess_evidence"
                key = (action, query)
                if key in visited:
                    break
                visited.add(key)
                grounding_decision = self.call(action, query=query, context=context)
                context["grounding_decision"] = grounding_decision
                plan.append({"step": step_idx, "action": action, "observation": grounding_decision["reason"]})
                continue
            if not memory_searched:
                action = "search_memory"
                key = (action, query)
                if key in visited:
                    break
                visited.add(key)
                memory_hits = self.call(action, query=query)
                memory_searched = True
                plan.append({"step": step_idx, "action": action, "observation": f"命中 {len(memory_hits)} 条记忆。"})
                continue
            if not answer:
                answer = self.call(
                    "synthesize_answer",
                    query=query,
                    context=context,
                    memory_hits=memory_hits,
                    grounding_decision=grounding_decision,
                )
                plan.append({"step": step_idx, "action": "synthesize_answer", "observation": "生成带证据回答。"})
                continue
            verified = self.call(
                "verify_answer",
                query=query,
                answer=answer,
                evidence_tags=context["evidence_tags"],
                evidence_items=context.get("items", []),
                grounding_decision=grounding_decision,
            )
            plan.append({"step": step_idx, "action": "verify_answer", "observation": verified["reason"]})
            break

        return HarnessRun(
            mode="react_like",
            answer=answer,
            plan=plan,
            context=context,
            grounding_decision=grounding_decision,
            memory_hits=memory_hits,
            verified=bool(verified["ok"]),
            verification_reason=str(verified["reason"]),
            verification=verified,
            tool_trace=self.trace(),
            safeguards={
                "max_steps": self.max_steps,
                "path_oscillation_control": "记录 action+input 签名，重复路径直接停止。",
                "context_policy": "每轮只把压缩后的证据窗口交给后续步骤。",
                "tool_runtime": self.registry.safeguards(),
            },
        )
