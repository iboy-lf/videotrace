from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

from PIL import Image

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .qwen_vl_api import QwenVLAPIClient
from ..vlm.frame_sampler import sample_frames


_RUNTIMES: dict[tuple[str, str, str, str], "Qwen35LocalRuntime"] = {}


class Qwen35LocalRuntime:
    def __init__(
        self,
        model_path: str,
        adapter_path: str = "",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.model = None
        self.processor = None
        self.input_device = None
        self.adapter_loaded = False

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float = 0.2,
        top_p: float = 0.8,
        top_k: int = 20,
        use_adapter: bool = True,
    ) -> str:
        self._load()
        import torch

        if use_adapter:
            self._ensure_adapter_loaded()

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        target_device = self.input_device or self._model_device()
        if target_device is None:
            raise RuntimeError("Qwen runtime has no resolved input device after model load")
        inputs = inputs.to(target_device)
        prompt_len = int(inputs["input_ids"].shape[1])
        do_sample = temperature > 0.0
        generation_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "top_k": int(top_k),
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(temperature=float(temperature), top_p=float(top_p))
        adapter_context = nullcontext()
        if not use_adapter and self.adapter_loaded and hasattr(self.model, "disable_adapter"):
            adapter_context = self.model.disable_adapter()
        with adapter_context, torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        generated = output_ids[:, prompt_len:]
        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }.get(self.dtype, torch.bfloat16)
        local_only = Path(self.model_path).exists()
        load_kwargs: dict[str, Any] = {
            "local_files_only": local_only,
            "trust_remote_code": True,
            "dtype": dtype,
            "attn_implementation": self.attn_implementation,
            "low_cpu_mem_usage": True,
        }
        if self.device == "auto":
            load_kwargs["device_map"] = "auto"
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=local_only,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, **load_kwargs)
        self._ensure_adapter_loaded()
        if self.device != "auto":
            self.model = self.model.to(self.device)
            self.input_device = self.device
        else:
            self.input_device = next(self.model.parameters()).device
        self.model.eval()

    def _model_device(self):
        if self.input_device is not None:
            return self.input_device
        if self.model is None:
            return None
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return None

    def configure_adapter(self, adapter_path: str) -> None:
        value = str(adapter_path or "").strip()
        if not value:
            return
        resolved = str(Path(value).expanduser().resolve())
        if self.adapter_path and str(Path(self.adapter_path).expanduser().resolve()) != resolved:
            raise RuntimeError(
                "Qwen runtime already owns a different adapter; isolate the model/device configuration"
            )
        self.adapter_path = resolved
        if self.model is not None:
            self._ensure_adapter_loaded()

    def _ensure_adapter_loaded(self) -> None:
        if self.adapter_loaded or not self.adapter_path or self.model is None:
            return
        adapter = Path(self.adapter_path).expanduser()
        if not (adapter.is_dir() and (adapter / "adapter_config.json").exists()):
            raise RuntimeError(f"证据回答 adapter 不完整：{adapter}")
        try:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter), is_trainable=False)
            self.model.eval()
            self.adapter_loaded = True
        except Exception as exc:
            raise RuntimeError(f"无法加载证据回答 adapter：{adapter}: {type(exc).__name__}: {exc}") from exc


def get_qwen35_runtime(
    model_path: str,
    adapter_path: str = "",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
) -> Qwen35LocalRuntime:
    key = (model_path, device, dtype, attn_implementation)
    if key not in _RUNTIMES:
        _RUNTIMES[key] = Qwen35LocalRuntime(
            model_path=model_path,
            adapter_path="",
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
    runtime = _RUNTIMES[key]
    runtime.configure_adapter(adapter_path)
    return runtime


class Qwen35LocalClient:
    backend = "qwen35_local"

    def __init__(
        self,
        model_path: str,
        adapter_path: str = "",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 1000,
        num_frames_per_segment: int = 2,
    ):
        self.runtime = get_qwen35_runtime(model_path, adapter_path=adapter_path, device=device, dtype=dtype)
        self.use_adapter = bool(str(adapter_path or "").strip())
        self.max_new_tokens = int(max_new_tokens)
        self.num_frames_per_segment = max(1, int(num_frames_per_segment))

    def generate_answer(self, query: str, context: dict, memory_hits: list[dict]) -> str:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    build_user_prompt(query, context, memory_hits)
                    + "\n请同时核对下面按片段分组的关键帧。最终优先输出紧凑 JSON："
                    '{"conclusion":"...","evidence":[{"timestamp":"start-end","text":"..."}]}。'
                    "不要输出 Markdown；timestamp 必须复制候选片段范围。"
                ),
            }
        ]
        video_path = str(context.get("video_path", ""))
        items = list(context.get("items", []))
        for index, item in enumerate(items[:4], start=1):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"片段 {index}: segment_id={item['segment_id']} "
                        f"timestamp={float(item['start_sec']):.1f}-{float(item['end_sec']):.1f}"
                    ),
                }
            )
            frames = sample_frames(
                video_path,
                float(item["start_sec"]),
                float(item["end_sec"]),
                self.num_frames_per_segment,
            )
            for frame in frames:
                content.append({"type": "image", "image": Image.fromarray(frame[:, :, ::-1])})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        raw = self.runtime.generate(
            messages,
            max_new_tokens=self.max_new_tokens,
            # Product answers are hash-bound artifacts. Greedy decoding keeps
            # repeated runs reproducible; data diversity belongs in SFT/DPO
            # construction rather than serving-time sampling.
            temperature=0.0,
            top_p=0.8,
            use_adapter=self.use_adapter,
        )
        return QwenVLAPIClient._format_answer(raw, query, items)
