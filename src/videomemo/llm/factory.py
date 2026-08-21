from __future__ import annotations

from .openai_compatible import OpenAICompatibleClient
from .qwen35_local import Qwen35LocalClient
from .qwen_vl_api import QwenVLAPIClient
from .template_llm import TemplateLLMClient


def build_llm_client(
    backend: str,
    base_url: str,
    model: str,
    api_key: str | None = None,
    adapter_path: str = "",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_new_tokens: int = 1000,
    num_frames_per_segment: int = 2,
):
    if backend == "template":
        return TemplateLLMClient()
    if backend == "openai_compatible":
        return OpenAICompatibleClient(base_url=base_url, model=model, api_key=api_key)
    if backend == "qwen_vl_api":
        return QwenVLAPIClient(base_url=base_url, model=model, api_key=api_key)
    if backend == "qwen35_local":
        return Qwen35LocalClient(
            model_path=model,
            adapter_path=adapter_path,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            num_frames_per_segment=num_frames_per_segment,
        )
    raise ValueError(f"Unknown LLM backend: {backend}")
