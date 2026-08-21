from __future__ import annotations

from .baseline_embedder import BaselineMultimodalEmbedder
from .clip_embedder import FrozenCLIPEmbedder
from .siglip_embedder import FrozenSigLIPEmbedder


def build_vlm_embedder(
    backend: str,
    model_name: str,
    cache_dir: str,
    num_frames: int,
    device: str | None = None,
):
    if backend == "clip":
        return FrozenCLIPEmbedder(
            model_name=model_name,
            cache_dir=cache_dir,
            num_frames=num_frames,
            device=device,
        )
    if backend == "siglip":
        return FrozenSigLIPEmbedder(
            model_name=model_name,
            cache_dir=cache_dir,
            num_frames=num_frames,
            device=device,
        )
    if backend == "baseline":
        return BaselineMultimodalEmbedder(cache_dir=cache_dir, num_frames=num_frames)
    if backend in {"none", ""}:
        return None
    raise ValueError(f"Unknown VLM backend: {backend}")
