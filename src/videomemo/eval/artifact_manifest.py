from __future__ import annotations

"""Stable, hash-addressed inventory for the interview delivery package."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

from .reproducibility import file_sha256, source_fingerprint


ARTIFACT_MANIFEST_SCHEMA = "videotrace-artifact-manifest-v1"

DEFAULT_ARTIFACT_PATHS: dict[str, str] = {
    "input_video": "data/raw/cola_review.mp4",
    "canonical_pack": "outputs/iboy_qwen35/cola_review/knowledge_pack.json",
    "canonical_export_manifest": "outputs/iboy_qwen35/cola_review/manifest.json",
    "reranker_checkpoint": "outputs/models/neural_reranker.pt",
    "reranker_metrics": "outputs/models/neural_reranker_metrics.json",
    "reranker_model_card": "outputs/models/neural_reranker_model_card.json",
    "reranker_dataset": "outputs_train/reranker_dev_5s.jsonl",
    "reranker_dataset_summary": "outputs_train/reranker_dev_5s.summary.json",
    "answer_verifier_dataset": "data/verifier/answer_verifier.jsonl",
    "answer_verifier_dataset_summary": "data/verifier/answer_verifier.summary.json",
    "answer_verifier_checkpoint": "outputs/models/answer_verifier.pkl",
    "answer_verifier_metrics": "outputs/models/answer_verifier_metrics.json",
    "answer_verifier_model_card": "outputs/models/answer_verifier_model_card.json",
    "sft_dataset": "data/sft/grounded_qa.jsonl",
    "sft_dataset_summary": "data/sft/grounded_qa.summary.json",
    "sft_adapter_weights": "outputs/models/qwen35_sft_adapter/adapter_model.safetensors",
    "sft_adapter_config": "outputs/models/qwen35_sft_adapter/adapter_config.json",
    "sft_optimizer_state": "outputs/models/qwen35_sft_adapter/optimizer.pt",
    "sft_rng_state": "outputs/models/qwen35_sft_adapter/rng_state.pt",
    "sft_trainer_state": "outputs/models/qwen35_sft_adapter/trainer_state.json",
    "sft_checkpoint_manifest": "outputs/models/qwen35_sft_adapter/checkpoint_manifest.json",
    "sft_metrics": "outputs/models/qwen35_sft_metrics.json",
    "sft_model_card": "outputs/models/qwen35_sft_model_card.json",
    "sft_resume_validation": "outputs/reports/qwen35_sft_resume_validation.json",
    "sft_resume_checkpoint_manifest": "outputs/runs/latest/sft_resume_validation/checkpoint_manifest.json",
    "preference_annotations": "data/preference/preference_annotations.json",
    "preference_dataset": "data/preference/grounded_dpo.jsonl",
    "preference_dataset_summary": "data/preference/grounded_dpo.summary.json",
    "dpo_adapter_weights": "outputs/models/qwen35_dpo_adapter/adapter_model.safetensors",
    "dpo_adapter_config": "outputs/models/qwen35_dpo_adapter/adapter_config.json",
    "dpo_optimizer_state": "outputs/models/qwen35_dpo_adapter/optimizer.pt",
    "dpo_rng_state": "outputs/models/qwen35_dpo_adapter/rng_state.pt",
    "dpo_trainer_state": "outputs/models/qwen35_dpo_adapter/trainer_state.json",
    "dpo_checkpoint_manifest": "outputs/models/qwen35_dpo_adapter/checkpoint_manifest.json",
    "dpo_metrics": "outputs/models/qwen35_dpo_metrics.json",
    "dpo_model_card": "outputs/models/qwen35_dpo_model_card.json",
    "dpo_reference_logprobs": "outputs/models/qwen35_dpo_reference_logprobs.json",
    "dpo_resume_validation": "outputs/reports/qwen35_dpo_resume_validation.json",
    "dpo_resume_checkpoint_manifest": "outputs/runs/latest/dpo_resume_validation/checkpoint_manifest.json",
    "best_adapter_registry": "outputs/models/best_adapter.json",
    "sft_adapter_evaluation": "outputs/reports/qwen35_sft_eval.json",
    "dpo_adapter_evaluation": "outputs/reports/qwen35_dpo_eval.json",
    "adapter_evaluation": "outputs/reports/qwen35_adapter_eval.json",
    "adapter_evaluation_baseline": "outputs/reports/qwen35_adapter_eval_baseline.json",
    "adapter_evaluation_adapter": "outputs/reports/qwen35_adapter_eval_adapter.json",
    "error_analysis": "outputs/reports/error_analysis.json",
    "performance_report": "outputs/reports/performance_report.json",
    "agent_failure_recovery": "outputs/reports/agent_failure_recovery.json",
    "gpu_selection_canonical": "outputs/reports/gpu_selection_canonical.json",
    "gpu_selection_dpo": "outputs/reports/gpu_selection_dpo.json",
    "browser_e2e": "outputs/reports/browser_e2e.json",
}


def build_artifact_manifest(
    root: str | Path,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> dict:
    root = Path(root).resolve()
    configured = dict(DEFAULT_ARTIFACT_PATHS if artifact_paths is None else artifact_paths)
    artifacts: dict[str, dict] = {}
    missing: list[str] = []
    for name, value in configured.items():
        path = _rooted(root, value)
        if not path.is_file():
            missing.append(name)
            continue
        artifacts[name] = _artifact_entry(root, path)

    admission_history: list[dict] = []
    history_dir = root / "outputs" / "reports" / "adapter_admissions"
    if history_dir.exists():
        admission_history = [
            _artifact_entry(root, path)
            for path in sorted(history_dir.glob("*.json"))
            if path.is_file()
        ]

    payloads = {
        name: _load_json(_rooted(root, configured[name]))
        for name in (
            "canonical_pack",
            "sft_metrics",
            "sft_resume_validation",
            "dpo_metrics",
            "dpo_resume_validation",
            "best_adapter_registry",
            "adapter_evaluation",
            "error_analysis",
            "performance_report",
            "agent_failure_recovery",
            "gpu_selection_canonical",
            "browser_e2e",
            "answer_verifier_metrics",
        )
        if name in configured and _rooted(root, configured[name]).is_file()
    }
    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_fingerprint(root),
        "complete": not missing,
        "missing_required": missing,
        "artifacts": artifacts,
        "adapter_admission_history": admission_history,
        "evidence_summary": _evidence_summary(payloads),
    }


def write_artifact_manifest(
    root: str | Path,
    output: str | Path,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> dict:
    root = Path(root).resolve()
    output_path = _rooted(root, output)
    manifest = build_artifact_manifest(root, artifact_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    # newline="\n" keeps the digest of this tracked report identical on Windows
    # and Linux; the delivery validator hashes these bytes.
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temp.replace(output_path)
    return manifest


def validate_manifest_hashes(root: str | Path, manifest: dict) -> dict:
    root = Path(root).resolve()
    mismatches: list[dict] = []
    for name, entry in dict(manifest.get("artifacts") or {}).items():
        path = _rooted(root, str(entry.get("path") or ""))
        current = file_sha256(path) if path.is_file() else ""
        expected = str(entry.get("sha256") or "")
        if current != expected or (path.is_file() and path.stat().st_size != int(entry.get("size_bytes", -1))):
            mismatches.append(
                {
                    "name": name,
                    "path": str(path),
                    "expected_sha256": expected,
                    "current_sha256": current,
                }
            )
    for entry in list(manifest.get("adapter_admission_history") or []):
        path = _rooted(root, str(entry.get("path") or ""))
        current = file_sha256(path) if path.is_file() else ""
        if current != str(entry.get("sha256") or ""):
            mismatches.append(
                {
                    "name": "adapter_admission_history",
                    "path": str(path),
                    "expected_sha256": entry.get("sha256", ""),
                    "current_sha256": current,
                }
            )
    return {"valid": not mismatches, "mismatches": mismatches}


def _artifact_entry(root: Path, path: Path) -> dict:
    try:
        display = path.resolve().relative_to(root).as_posix()
    except ValueError:
        display = str(path.resolve())
    return {
        "path": display,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _evidence_summary(payloads: Mapping[str, dict]) -> dict:
    pack = payloads.get("canonical_pack", {})
    metadata = dict(pack.get("metadata") or {})
    agent = dict(metadata.get("agent_run") or {})
    metrics = payloads.get("sft_metrics", {})
    dpo_metrics = payloads.get("dpo_metrics", {})
    sft_resume = payloads.get("sft_resume_validation", {})
    dpo_resume = payloads.get("dpo_resume_validation", {})
    registry = payloads.get("best_adapter_registry", {})
    admission = dict(metrics.get("adapter_admission") or {})
    evaluation = payloads.get("adapter_evaluation", {})
    regression = payloads.get("error_analysis", {})
    performance = payloads.get("performance_report", {})
    recovery = payloads.get("agent_failure_recovery", {})
    gpu = payloads.get("gpu_selection_canonical", {})
    browser = payloads.get("browser_e2e", {})
    answer_verifier = payloads.get("answer_verifier_metrics", {})
    selected_id = str(registry.get("selected_candidate_id") or "")
    selected_entry = dict((registry.get("candidates") or {}).get(selected_id) or {})
    return {
        "canonical": {
            "source_sha256": metadata.get("source_sha256", ""),
            "video_sha256": metadata.get("video_sha256", ""),
            "verified": bool(agent.get("verified")),
            "coverage": float(agent.get("verification", {}).get("coverage", 0.0)),
            "vlm_mode": metadata.get("vlm_mode", {}),
            "llm_adapter": metadata.get("llm_adapter", {}),
            "physical_gpu_ids": metadata.get("environment", {}).get("physical_gpu_ids", ""),
        },
        "post_training": {
            "sft_status": metrics.get("status", ""),
            "sft_steps": metrics.get("steps", 0),
            "sft_counts": metrics.get("counts", {}),
            "sft_gradient_payload_sha256": metrics.get("gradient_payload_sha256", ""),
            "sft_validated_for_web": bool(metrics.get("validated_for_web")),
            "sft_adapter_sha256": admission.get("adapter_sha256", ""),
            "sft_evaluation_sha256": admission.get("evaluation_sha256", ""),
            "selected_candidate_id": selected_id,
            "selected_method": selected_entry.get("method", ""),
            "selected_validated_for_web": bool(selected_entry.get("validated_for_web")),
            "selected_adapter_sha256": selected_entry.get("adapter_sha256", ""),
            "selected_evaluation_sha256": selected_entry.get("evaluation_sha256", ""),
            "evaluation_passed": bool(evaluation.get("comparison", {}).get("validated_for_web")),
            "fallback_order": registry.get("fallback_order", []),
            "dpo_status": dpo_metrics.get("status", ""),
            "dpo_steps": dpo_metrics.get("steps", 0),
            "dpo_dev_reward_margin": dpo_metrics.get("evaluations", {}).get("dev", {}).get("mean_reward_margin"),
            "dpo_frozen_reward_margin": dpo_metrics.get("evaluations", {}).get("frozen_test", {}).get("mean_reward_margin"),
            "sft_resume": {
                "resumed_from_step": sft_resume.get("resumed_from_step", 0),
                "steps_this_run": sft_resume.get("steps_this_run", 0),
                "final_step": sft_resume.get("steps", 0),
            },
            "dpo_resume": {
                "resumed_from_step": dpo_resume.get("resumed_from_step", 0),
                "steps_this_run": dpo_resume.get("steps_this_run", 0),
                "final_step": dpo_resume.get("steps", 0),
            },
        },
        "regression": {
            "num_cases": regression.get("num_cases", 0),
            "num_passed": regression.get("num_passed", 0),
            "error_category_counts": regression.get("error_category_counts", {}),
        },
        "performance": {
            "cold_seconds": performance.get("model_residency", {}).get(
                "cold_pipeline_construction_and_run_seconds"
            ),
            "warm_seconds": performance.get("model_residency", {}).get("warm_cache_hit_run_seconds"),
            "speedup": performance.get("model_residency", {}).get("speedup"),
            "correctness": performance.get("precision", {}).get("correctness_check", {}),
        },
        "agent_failure_recovery": {
            "passed": bool(recovery.get("passed")),
            "attempts": recovery.get("attempts", 0),
            "final_action": recovery.get("final_action", ""),
        },
        "gpu_selection": {
            "status": gpu.get("status", ""),
            "selected_physical_gpu_ids": gpu.get("selected_physical_gpu_ids", []),
            "stable_checks": gpu.get("required_stable_checks", 0),
        },
        "browser_e2e": {
            "valid": bool(browser.get("valid")),
            "checks": browser.get("checks", {}),
            "job_id": browser.get("job_id", ""),
            "durable_job": bool(browser.get("job", {}).get("persistence", {}).get("durable")),
            "elapsed_sec": browser.get("job", {}).get("elapsed_sec"),
            "elapsed_stable": bool(browser.get("checks", {}).get("completed_elapsed_is_frozen")),
        },
        "answer_verifier": {
            "status": answer_verifier.get("status", ""),
            "validated_for_product": bool(answer_verifier.get("validated_for_product")),
            "model_format": answer_verifier.get("model_format", ""),
            "checkpoint_sha256": answer_verifier.get("checkpoint_sha256", ""),
            "threshold": answer_verifier.get("threshold"),
            "counts": answer_verifier.get("counts", {}),
            "dev": answer_verifier.get("evaluations", {}).get("dev", {}),
            "frozen_test": answer_verifier.get("evaluations", {}).get("frozen_test", {}),
        },
    }


def _rooted(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
