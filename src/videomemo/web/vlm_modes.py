from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import os

from ..config import VideoMemoConfig


@dataclass(frozen=True)
class VLMMode:
    mode_id: str
    label: str
    description: str
    overrides: dict[str, object]

    def public(self) -> dict:
        return {
            "id": self.mode_id,
            "label": self.label,
            "description": self.description,
        }


def available_vlm_modes(config: VideoMemoConfig) -> list[VLMMode]:
    """Return only executable, server-approved visual modes.

    Discovery is deliberately narrow: it can resolve the configured weights or
    known Qwen3.5/SigLIP2 directories under trusted model roots, but it never
    accepts a browser-provided model path.
    """

    qwen_path = _resolve_model(
        config.segment_understanding_model or config.llm_model,
        ("Qwen3.5-9B", "Qwen3.5*9B*", "*Qwen3.5*9B*"),
    )
    siglip_path = _resolve_model(
        config.vlm_model_name if config.vlm_backend == "siglip" else "",
        ("siglip2-large-patch16-256", "*siglip2*large*256*", "*SigLIP2*large*256*"),
    )
    # These product modes map to the in-process qwen35_local runtime. An
    # arbitrary configured HTTP endpoint is not enough to advertise them.
    qwen_ready = bool(qwen_path)
    siglip_ready = bool(siglip_path)
    reranker_ready = config.reranker_backend != "neural" or _path_exists(config.reranker_model_path)

    if not reranker_ready:
        return []

    modes: list[VLMMode] = []
    if qwen_ready and siglip_ready:
        modes.append(
            VLMMode(
                mode_id="auto_best",
                label="自动最佳（Qwen3.5 + SigLIP2）",
                description="Qwen3.5 逐片段视频理解，结合 SigLIP2 视觉检索增强。",
                overrides={
                    "segment_understanding_backend": "qwen35_local",
                    "segment_understanding_model": qwen_path or config.segment_understanding_model,
                    "segment_understanding_fail_open": False,
                    "vlm_backend": "siglip",
                    "vlm_model_name": siglip_path,
                    "llm_backend": "qwen35_local",
                    "llm_model": qwen_path or config.llm_model,
                    "retrieval_weight": 0.42,
                    "scorer_weight": 0.18,
                    "vlm_weight": 0.40,
                    "persist_dense_index": True,
                },
            )
        )
    if qwen_ready:
        modes.append(
            VLMMode(
                mode_id="qwen35_video",
                label="Qwen3.5 视频理解",
                description="使用 Qwen3.5 生成片段描述与 OCR，再进行稀疏检索和证据回答。",
                overrides={
                    "segment_understanding_backend": "qwen35_local",
                    "segment_understanding_model": qwen_path or config.segment_understanding_model,
                    "segment_understanding_fail_open": False,
                    "vlm_backend": "baseline",
                    "vlm_model_name": "",
                    "llm_backend": "qwen35_local",
                    "llm_model": qwen_path or config.llm_model,
                    "retrieval_weight": 0.68,
                    "scorer_weight": 0.32,
                    "vlm_weight": 0.0,
                    "persist_dense_index": False,
                },
            )
        )
    if qwen_ready and siglip_ready:
        modes.append(
            VLMMode(
                mode_id="siglip_retrieval",
                label="SigLIP2 检索增强",
                description="以 SigLIP2 图文检索定位片段，由固定 Qwen3.5 回答链路生成证据答案。",
                overrides={
                    "segment_understanding_backend": "baseline",
                    "segment_understanding_model": "",
                    "segment_understanding_fail_open": True,
                    "vlm_backend": "siglip",
                    "vlm_model_name": siglip_path,
                    "llm_backend": "qwen35_local",
                    "llm_model": qwen_path or config.llm_model,
                    "retrieval_weight": 0.35,
                    "scorer_weight": 0.15,
                    "vlm_weight": 0.50,
                    "persist_dense_index": True,
                },
            )
        )
    return modes


def apply_vlm_mode(config: VideoMemoConfig, mode_id: str) -> tuple[VideoMemoConfig, VLMMode]:
    modes = {mode.mode_id: mode for mode in available_vlm_modes(config)}
    selected = modes.get(mode_id)
    if selected is None:
        raise ValueError(f"不可用或不受支持的视觉模式：{mode_id}")
    updated = replace(config, **selected.overrides)
    updated.selected_vlm_mode = selected.mode_id
    updated.selected_vlm_mode_label = selected.label
    return updated, selected


def capability_payload(config: VideoMemoConfig) -> dict:
    modes = available_vlm_modes(config)
    default_mode = "auto_best" if any(mode.mode_id == "auto_best" for mode in modes) else (
        modes[0].mode_id if modes else ""
    )
    if modes:
        message = "远端视觉算力已连接，可以上传视频并开始分析。"
        state = "ready"
    else:
        message = "当前页面可预览视频，但尚未连接可执行的远端视觉算力。请使用远端一键启动入口。"
        state = "unavailable"
    return {
        "analysis_available": bool(modes),
        "state": state,
        "message": message,
        "default_mode": default_mode,
        "vlm_modes": [mode.public() for mode in modes],
        "queue": {"policy": "serial", "max_active_gpu_jobs": 1},
    }


def _resolve_model(configured: str, patterns: tuple[str, ...]) -> str:
    configured = str(configured or "").strip()
    if configured and Path(configured).expanduser().exists():
        return str(Path(configured).expanduser().resolve())
    roots = os.environ.get("VIDEOTRACE_MODEL_ROOTS", "/lavender/models").split(os.pathsep)
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if not root.is_dir():
            continue
        for pattern in patterns:
            for candidate in sorted(root.glob(pattern)):
                if candidate.is_dir():
                    return str(candidate.resolve())
    return ""


def _path_exists(path: str) -> bool:
    candidate = Path(str(path or "")).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.exists()
