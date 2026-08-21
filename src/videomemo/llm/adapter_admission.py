from __future__ import annotations

"""Resolve only project-whitelisted post-training adapters with hash admission."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from ..eval.reproducibility import source_fingerprint


BEST_ADAPTER_SCHEMA_VERSION = "videotrace-best-adapter-registry-v1"


@dataclass(frozen=True)
class AdapterCandidateSpec:
    candidate_id: str
    method: str
    adapter_relative: str
    metrics_relative: str
    model_card_relative: str


ADAPTER_CANDIDATES: dict[str, AdapterCandidateSpec] = {
    "qwen35_sft": AdapterCandidateSpec(
        candidate_id="qwen35_sft",
        method="sft",
        adapter_relative="outputs/models/qwen35_sft_adapter",
        metrics_relative="outputs/models/qwen35_sft_metrics.json",
        model_card_relative="outputs/models/qwen35_sft_model_card.json",
    ),
    "qwen35_dpo": AdapterCandidateSpec(
        candidate_id="qwen35_dpo",
        method="dpo",
        adapter_relative="outputs/models/qwen35_dpo_adapter",
        metrics_relative="outputs/models/qwen35_dpo_metrics.json",
        model_card_relative="outputs/models/qwen35_dpo_model_card.json",
    ),
}


def resolve_validated_adapter(project_root: str | Path) -> str:
    root = Path(project_root).resolve()
    registry_path = root / "outputs" / "models" / "best_adapter.json"
    if registry_path.is_file():
        registry = _load_json(registry_path)
        resolved, _ = _resolve_registry_candidate(root, registry)
        return str(resolved) if resolved is not None else ""
    legacy = _resolve_legacy_sft(root)
    return str(legacy) if legacy is not None else ""


def adapter_admission_metadata(project_root: str | Path) -> dict:
    root = Path(project_root).resolve()
    registry_path = root / "outputs" / "models" / "best_adapter.json"
    if registry_path.is_file():
        registry = _load_json(registry_path)
        resolved, selected_id = _resolve_registry_candidate(root, registry)
        if resolved is None or not selected_id:
            return {"enabled": False, "path": "", "validated_for_web": False}
        entry = dict(registry.get("candidates", {}).get(selected_id) or {})
        return {
            "enabled": True,
            "path": str(resolved),
            "validated_for_web": True,
            "candidate_id": selected_id,
            "method": entry.get("method", ""),
            "adapter_sha256": entry.get("adapter_sha256", ""),
            "evaluation_sha256": entry.get("evaluation_sha256", ""),
            "pack_sha256": entry.get("pack_sha256", ""),
            "video_sha256": entry.get("video_sha256", ""),
            "decision": registry.get("decision", ""),
            "registry_sha256": _sha256(registry_path),
            "fallback_used": selected_id != str(registry.get("selected_candidate_id") or ""),
        }

    resolved = _resolve_legacy_sft(root)
    if resolved is None:
        return {"enabled": False, "path": "", "validated_for_web": False}
    metrics = _load_json(root / ADAPTER_CANDIDATES["qwen35_sft"].metrics_relative)
    admission = dict(metrics.get("adapter_admission") or {})
    return {
        "enabled": True,
        "path": str(resolved),
        "validated_for_web": True,
        "candidate_id": "qwen35_sft",
        "method": "sft",
        "adapter_sha256": admission.get("adapter_sha256", ""),
        "evaluation_sha256": admission.get("evaluation_sha256", ""),
        "pack_sha256": admission.get("pack_sha256", ""),
        "video_sha256": admission.get("video_sha256", ""),
        "decision": admission.get("decision", "legacy SFT admission"),
        "fallback_used": False,
    }


def candidate_artifact_paths(project_root: str | Path, candidate_id: str) -> dict[str, Path]:
    root = Path(project_root).resolve()
    spec = ADAPTER_CANDIDATES.get(candidate_id)
    if spec is None:
        raise ValueError(f"adapter candidate is not whitelisted: {candidate_id}")
    adapter = (root / spec.adapter_relative).resolve()
    return {
        "adapter": adapter,
        "weights": adapter / "adapter_model.safetensors",
        "config": adapter / "adapter_config.json",
        "metrics": (root / spec.metrics_relative).resolve(),
        "model_card": (root / spec.model_card_relative).resolve(),
    }


def _resolve_registry_candidate(root: Path, registry: dict) -> tuple[Path | None, str]:
    if registry.get("schema_version") != BEST_ADAPTER_SCHEMA_VERSION:
        return None, ""
    if not bool(registry.get("validated_for_web")):
        return None, ""
    if str(registry.get("source_sha256") or "") != source_fingerprint(root):
        return None, ""
    selected = str(registry.get("selected_candidate_id") or "")
    order = [selected, *[str(item) for item in list(registry.get("fallback_order") or [])]]
    entries = dict(registry.get("candidates") or {})
    for candidate_id in dict.fromkeys(item for item in order if item):
        spec = ADAPTER_CANDIDATES.get(candidate_id)
        entry = entries.get(candidate_id)
        if spec is None or not isinstance(entry, dict):
            continue
        resolved = _validate_registry_entry(root, spec, entry)
        if resolved is not None:
            return resolved, candidate_id
    return None, ""


def _validate_registry_entry(root: Path, spec: AdapterCandidateSpec, entry: dict) -> Path | None:
    if not bool(entry.get("validated_for_web")) or entry.get("method") != spec.method:
        return None
    paths = candidate_artifact_paths(root, spec.candidate_id)
    if not all(paths[name].is_file() for name in ("weights", "config", "metrics", "model_card")):
        return None
    expected_hashes = {
        "adapter_sha256": _sha256(paths["weights"]),
        "adapter_config_sha256": _sha256(paths["config"]),
        "metrics_sha256": _sha256(paths["metrics"]),
        "model_card_sha256": _sha256(paths["model_card"]),
    }
    if any(str(entry.get(key) or "").lower() != value for key, value in expected_hashes.items()):
        return None
    evaluation_path = _resolve_report_path(root, str(entry.get("evaluation_report") or ""))
    expected_eval_sha = str(entry.get("evaluation_sha256") or "").lower()
    if not expected_eval_sha or not evaluation_path.is_file() or _sha256(evaluation_path) != expected_eval_sha:
        return None
    if str(entry.get("source_sha256") or "") != source_fingerprint(root):
        return None
    return paths["adapter"].resolve()


def _resolve_legacy_sft(root: Path) -> Path | None:
    paths = candidate_artifact_paths(root, "qwen35_sft")
    if not all(paths[name].is_file() for name in ("weights", "config", "metrics")):
        return None
    metrics = _load_json(paths["metrics"])
    admission = metrics.get("adapter_admission")
    if not metrics.get("validated_for_web") or not isinstance(admission, dict):
        return None
    if _sha256(paths["weights"]) != str(admission.get("adapter_sha256") or "").lower():
        return None
    expected_eval_sha = str(admission.get("evaluation_sha256") or "").lower()
    report_path = _resolve_report_path(root, str(admission.get("evaluation_report") or ""))
    if expected_eval_sha and (not report_path.is_file() or _sha256(report_path) != expected_eval_sha):
        return None
    return paths["adapter"].resolve()


def _resolve_report_path(root: Path, value: str) -> Path:
    if not value.strip():
        return Path()
    supplied = Path(value).expanduser()
    if supplied.is_file():
        return supplied.resolve()
    candidates = (
        root / "outputs" / "reports" / "adapter_admissions" / supplied.name,
        root / "outputs" / "reports" / supplied.name,
    )
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), candidates[0].resolve())


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
