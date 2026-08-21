from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class VideoMemoConfig:
    segment_seconds: int = 30
    sample_fps: float = 1.0
    top_k: int = 5
    clip_margin_sec: int = 3
    output_dir: str = "outputs"
    scorer_model_path: str = "outputs/scorer_model.pkl"
    export_clips: bool = True
    export_html: bool = True
    use_ocr: bool = True
    use_scene_cut: bool = False
    min_query_overlap: int = 1
    enable_cache: bool = False
    max_context_chars: int = 1800
    per_segment_context_chars: int = 360
    max_agent_steps: int = 6
    tool_retry_max_attempts: int = 2
    tool_retry_backoff_sec: float = 0.2
    tool_circuit_failure_threshold: int = 3
    tool_circuit_recovery_sec: float = 60.0
    segment_understanding_backend: str = "baseline"
    segment_understanding_base_url: str = ""
    segment_understanding_model: str = ""
    segment_understanding_api_key: str = ""
    segment_understanding_cache_dir: str = "outputs_cache/segment_understanding"
    segment_understanding_frames: int = 4
    segment_understanding_timeout_sec: int = 180
    segment_understanding_fail_open: bool = True
    segment_understanding_device: str = "cuda:0"
    segment_understanding_dtype: str = "bfloat16"
    segment_understanding_max_new_tokens: int = 800
    asr_backend: str = "none"
    asr_model: str = ""
    asr_device: str = "cuda"
    asr_compute_type: str = "float16"
    asr_language: str = "zh"
    asr_cache_dir: str = "outputs_cache/asr"
    asr_fail_open: bool = True
    vlm_backend: str = "baseline"
    vlm_model_name: str = "openai/clip-vit-base-patch32"
    vlm_cache_dir: str = "outputs_cache/vlm"
    vlm_num_frames: int = 4
    vlm_device: str = ""
    retrieval_weight: float = 0.45
    scorer_weight: float = 0.35
    vlm_weight: float = 0.20
    persist_dense_index: bool = False
    dense_index_dir: str = "outputs/indexes"
    reranker_backend: str = "none"
    reranker_model_path: str = "outputs/models/neural_reranker.pt"
    reranker_device: str = "cpu"
    reranker_weight: float = 0.35
    llm_backend: str = "template"
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "local-model"
    llm_adapter_path: str = ""
    llm_api_key: str = ""
    llm_device: str = "cuda:0"
    llm_dtype: str = "bfloat16"
    llm_max_new_tokens: int = 1000
    llm_num_frames_per_segment: int = 2
    persistent_memory_enabled: bool = False
    persistent_memory_path: str = "outputs_memory/memories.jsonl"
    abstain_enabled: bool = True
    abstain_min_query_coverage: float = 0.18
    abstain_min_vlm_score: float = 0.16
    answer_verifier_backend: str = "deterministic"
    answer_verifier_model_path: str = "outputs/models/answer_verifier.pkl"
    answer_verifier_threshold: float = -1.0
    answer_verifier_fail_open: bool = True
    temporal_coverage_enabled: bool = True
    overview_min_segments: int = 3
    selected_vlm_mode: str = "auto_best"
    selected_vlm_mode_label: str = "自动最佳（Qwen3.5 + SigLIP2）"

    @classmethod
    def load(cls, path: Optional[str] = None) -> "VideoMemoConfig":
        if not path:
            return cls()
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**data)

    def dump(self) -> dict:
        """Return a JSON-serializable snapshot for reproducible output artifacts."""
        return dict(self.__dict__)
