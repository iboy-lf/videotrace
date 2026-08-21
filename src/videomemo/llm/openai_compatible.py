from __future__ import annotations

import json
import os
from urllib import request

from .prompts import SYSTEM_PROMPT, build_user_prompt


class OpenAICompatibleClient:
    backend = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_sec: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout_sec = timeout_sec

    def generate_answer(self, query: str, context: dict, memory_hits: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(query, context, memory_hits)},
            ],
            "temperature": 0.2,
        }
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with request.urlopen(req, timeout=self.timeout_sec) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]
