from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    backend: str

    def generate_answer(self, query: str, context: dict, memory_hits: list[dict]) -> str:
        ...
