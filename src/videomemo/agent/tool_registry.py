from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict
import time
import uuid

from .tool_protocol import ToolResponse


_TYPE_CHECKS: dict[str, Any] = {
    "any": object,
    "str": str,
    "dict": dict,
    "list": list,
    "bool": bool,
    "number": (int, float),
}


class ToolContractError(ValueError):
    pass


class ToolCircuitOpenError(RuntimeError):
    pass


@dataclass
class ToolSpec:
    name: str
    description: str
    input_keys: list[str]
    output_keys: list[str]
    input_schema: dict[str, str] = field(default_factory=dict)
    allow_extra_inputs: bool = False
    max_attempts: int | None = None
    retry_backoff_sec: float | None = None

    def dump(self) -> dict:
        return self.__dict__


@dataclass
class ToolCallRecord:
    name: str
    inputs: dict
    output: Any
    latency_ms: float
    ok: bool = True
    error: str = ""
    status: str = "success"
    error_code: str = ""
    attempts: int = 1
    response: dict = field(default_factory=dict)
    circuit_state: str = "closed"
    call_id: str = ""
    started_at_utc: str = ""
    finished_at_utc: str = ""
    attempt_trace: list[dict] = field(default_factory=list)

    def dump(self) -> dict:
        return self.__dict__


@dataclass
class _RegisteredTool:
    spec: ToolSpec
    handler: Callable[..., Any]


class ToolCircuitBreaker:
    """Per-tool consecutive-failure circuit breaker with timed recovery."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_sec: float = 60.0,
        enabled: bool = True,
    ):
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_sec = max(0.0, float(recovery_sec))
        self.enabled = bool(enabled)
        self._failure_counts: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    def allow(self, name: str) -> bool:
        if not self.enabled or name not in self._opened_at:
            return True
        if time.monotonic() - self._opened_at[name] >= self.recovery_sec:
            self.reset(name)
            return True
        return False

    def record_success(self, name: str) -> None:
        self.reset(name)

    def record_failure(self, name: str) -> None:
        if not self.enabled:
            return
        count = self._failure_counts.get(name, 0) + 1
        self._failure_counts[name] = count
        if count >= self.failure_threshold:
            self._opened_at[name] = time.monotonic()

    def reset(self, name: str) -> None:
        self._failure_counts[name] = 0
        self._opened_at.pop(name, None)

    def status(self, name: str) -> dict:
        is_open = name in self._opened_at and not self.allow(name)
        result = {
            "state": "open" if is_open else "closed",
            "failure_count": self._failure_counts.get(name, 0),
        }
        if is_open:
            elapsed = time.monotonic() - self._opened_at[name]
            result["recover_in_sec"] = round(max(0.0, self.recovery_sec - elapsed), 3)
        return result

    def all_statuses(self) -> dict[str, dict]:
        names = sorted(set(self._failure_counts) | set(self._opened_at))
        return {name: self.status(name) for name in names}


class AgentToolRegistry:
    def __init__(
        self,
        default_max_attempts: int = 1,
        default_retry_backoff_sec: float = 0.0,
        circuit_breaker: ToolCircuitBreaker | None = None,
    ):
        self._tools: Dict[str, _RegisteredTool] = {}
        self.calls: list[ToolCallRecord] = []
        self.default_max_attempts = max(1, int(default_max_attempts))
        self.default_retry_backoff_sec = max(0.0, float(default_retry_backoff_sec))
        self.circuit_breaker = circuit_breaker or ToolCircuitBreaker()

    def register(self, spec: ToolSpec, handler: Callable[..., Any]) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._validate_spec(spec)
        self._tools[spec.name] = _RegisteredTool(spec=spec, handler=handler)

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        tool = self._tools[name]
        started = time.perf_counter()
        call_id = uuid.uuid4().hex
        started_at_utc = datetime.now(timezone.utc).isoformat()
        attempt_trace: list[dict] = []
        try:
            self._validate_inputs(tool.spec, kwargs)
        except ToolContractError as exc:
            response = ToolResponse.error("invalid_input", str(exc))
            attempt_trace.append(
                {
                    "attempt": 0,
                    "status": "blocked",
                    "ok": False,
                    "error_code": "invalid_input",
                    "error": str(exc),
                    "latency_ms": 0.0,
                    "will_retry": False,
                    "backoff_sec": 0.0,
                }
            )
            self._record(
                name,
                kwargs,
                response,
                started,
                attempts=0,
                call_id=call_id,
                started_at_utc=started_at_utc,
                attempt_trace=attempt_trace,
            )
            raise

        if not self.circuit_breaker.allow(name):
            message = f"Tool circuit is open: {name}"
            response = ToolResponse.error("circuit_open", message)
            attempt_trace.append(
                {
                    "attempt": 0,
                    "status": "blocked",
                    "ok": False,
                    "error_code": "circuit_open",
                    "error": message,
                    "latency_ms": 0.0,
                    "will_retry": False,
                    "backoff_sec": 0.0,
                }
            )
            self._record(
                name,
                kwargs,
                response,
                started,
                attempts=0,
                call_id=call_id,
                started_at_utc=started_at_utc,
                attempt_trace=attempt_trace,
            )
            raise ToolCircuitOpenError(message)

        max_attempts = max(1, int(tool.spec.max_attempts or self.default_max_attempts))
        backoff = (
            self.default_retry_backoff_sec
            if tool.spec.retry_backoff_sec is None
            else max(0.0, float(tool.spec.retry_backoff_sec))
        )
        last_error: Exception | None = None
        attempts_used = 0
        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            attempt_started = time.perf_counter()
            try:
                output = tool.handler(**kwargs)
                self._validate_output(tool.spec, output)
                response = ToolResponse.success(
                    output,
                    message=f"{name} completed",
                    attempts=attempt,
                )
                self.circuit_breaker.record_success(name)
                attempt_trace.append(
                    {
                        "attempt": attempt,
                        "status": "success",
                        "ok": True,
                        "error_code": "",
                        "error": "",
                        "latency_ms": (time.perf_counter() - attempt_started) * 1000.0,
                        "will_retry": False,
                        "backoff_sec": 0.0,
                    }
                )
                self._record(
                    name,
                    kwargs,
                    response,
                    started,
                    attempts=attempt,
                    call_id=call_id,
                    started_at_utc=started_at_utc,
                    attempt_trace=attempt_trace,
                )
                return output
            except ToolContractError as exc:
                last_error = exc
                attempt_trace.append(
                    {
                        "attempt": attempt,
                        "status": "error",
                        "ok": False,
                        "error_code": "invalid_output",
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_ms": (time.perf_counter() - attempt_started) * 1000.0,
                        "will_retry": False,
                        "backoff_sec": 0.0,
                    }
                )
                break
            except Exception as exc:
                last_error = exc
                will_retry = attempt < max_attempts
                backoff_sec = backoff * attempt if will_retry else 0.0
                attempt_trace.append(
                    {
                        "attempt": attempt,
                        "status": "error",
                        "ok": False,
                        "error_code": "execution_error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_ms": (time.perf_counter() - attempt_started) * 1000.0,
                        "will_retry": will_retry,
                        "backoff_sec": backoff_sec,
                    }
                )
                if backoff_sec > 0:
                    time.sleep(backoff_sec)

        assert last_error is not None
        self.circuit_breaker.record_failure(name)
        error_code = "invalid_output" if isinstance(last_error, ToolContractError) else "execution_error"
        response = ToolResponse.error(
            error_code,
            f"{type(last_error).__name__}: {last_error}",
            attempts=attempts_used,
        )
        self._record(
            name,
            kwargs,
            response,
            started,
            attempts=attempts_used,
            call_id=call_id,
            started_at_utc=started_at_utc,
            attempt_trace=attempt_trace,
        )
        raise last_error

    def specs(self) -> list[dict]:
        return [tool.spec.dump() for tool in self._tools.values()]

    def trace(self) -> list[dict]:
        return [call.dump() for call in self.calls]

    def safeguards(self) -> dict:
        return {
            "default_max_attempts": self.default_max_attempts,
            "default_retry_backoff_sec": self.default_retry_backoff_sec,
            "circuit_breakers": self.circuit_breaker.all_statuses(),
        }

    @staticmethod
    def _validate_spec(spec: ToolSpec) -> None:
        unknown = set(spec.input_schema) - set(spec.input_keys)
        if unknown:
            raise ValueError(f"Tool {spec.name} schema has unknown inputs: {sorted(unknown)}")
        invalid_types = {value for value in spec.input_schema.values() if value not in _TYPE_CHECKS}
        if invalid_types:
            raise ValueError(f"Tool {spec.name} has unsupported schema types: {sorted(invalid_types)}")

    @staticmethod
    def _validate_inputs(spec: ToolSpec, inputs: dict[str, Any]) -> None:
        missing = [key for key in spec.input_keys if key not in inputs]
        if missing:
            raise ToolContractError(f"Tool {spec.name} missing required inputs: {missing}")
        extra = sorted(set(inputs) - set(spec.input_keys))
        if extra and not spec.allow_extra_inputs:
            raise ToolContractError(f"Tool {spec.name} received unexpected inputs: {extra}")
        for key, type_name in spec.input_schema.items():
            value = inputs.get(key)
            expected = _TYPE_CHECKS[type_name]
            valid = isinstance(value, expected)
            if type_name == "number" and isinstance(value, bool):
                valid = False
            if not valid:
                raise ToolContractError(
                    f"Tool {spec.name} input {key!r} must be {type_name}, got {type(value).__name__}"
                )

    @staticmethod
    def _validate_output(spec: ToolSpec, output: Any) -> None:
        if not spec.output_keys:
            return
        if isinstance(output, dict):
            missing = [key for key in spec.output_keys if key not in output]
        elif isinstance(output, list):
            missing = []
            for item in output:
                if isinstance(item, dict):
                    missing.extend(key for key in spec.output_keys if key not in item)
            missing = sorted(set(missing))
        elif len(spec.output_keys) == 1:
            missing = []
        else:
            missing = list(spec.output_keys)
        if missing:
            raise ToolContractError(f"Tool {spec.name} output missing fields: {missing}")

    def _record(
        self,
        name: str,
        inputs: dict,
        response: ToolResponse,
        started: float,
        attempts: int,
        *,
        call_id: str,
        started_at_utc: str,
        attempt_trace: list[dict],
    ) -> None:
        self.calls.append(
            ToolCallRecord(
                name=name,
                inputs=inputs,
                output=response.data,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                ok=response.ok,
                error=response.message if not response.ok else "",
                status=response.status.value,
                error_code=response.error_code,
                attempts=attempts,
                response=response.dump(),
                circuit_state=self.circuit_breaker.status(name)["state"],
                call_id=call_id,
                started_at_utc=started_at_utc,
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
                attempt_trace=list(attempt_trace),
            )
        )
