from __future__ import annotations

from .harness import AgentHarness, HarnessRun
from .tool_registry import AgentToolRegistry


AgentRun = HarnessRun


class VideoMemoAgentRuntime(AgentHarness):
    def __init__(self, registry: AgentToolRegistry, max_steps: int = 6):
        super().__init__(max_steps=max_steps)
        self.registry = registry

    def register_tool(self, spec, handler):
        self.registry.register(spec, handler)

    def call(self, name: str, **kwargs):
        return self.registry.call(name, **kwargs)

    def trace(self) -> list[dict]:
        return self.registry.trace()
