from __future__ import annotations

"""Cross-check product, training, Agent and runtime evidence as one package."""

import json
from pathlib import Path
from typing import Mapping

from ..llm.adapter_admission import (
    adapter_admission_metadata,
    candidate_artifact_paths,
    resolve_validated_adapter,
)
from ..training.preference_data import (
    load_preference_records,
    preference_gradient_payload_sha256,
    validate_preference_records,
)
from ..training.sft_data import gradient_payload_sha256, load_sft_records, validate_sft_records
from ..training.answer_verifier import validate_answer_verifier_rows
from ..verifier.calibrated import CHECKPOINT_SCHEMA_VERSION, FEATURE_CONTRACT_VERSION, FEATURE_NAMES
from .artifact_manifest import DEFAULT_ARTIFACT_PATHS, validate_manifest_hashes
from .interview_readiness import validate_interview_package
from .reproducibility import canonical_pack_sha256, file_sha256, source_fingerprint


def validate_delivery_package(
    root: str | Path,
    manifest_path: str | Path = "outputs/reports/artifact_manifest.json",
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> dict:
    root = Path(root).resolve()
    configured = dict(DEFAULT_ARTIFACT_PATHS)
    if artifact_paths:
        configured.update(artifact_paths)
    paths = {name: _rooted(root, value) for name, value in configured.items()}
    paths["artifact_manifest"] = _rooted(root, manifest_path)
    checks: list[dict] = []

    def add(name: str, passed: bool, evidence: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    missing = [name for name, path in paths.items() if not path.is_file()]
    add("required_delivery_artifacts_exist", not missing, {"missing": missing})
    if missing:
        return _report(checks, paths)

    manifest = _load(paths["artifact_manifest"])
    current_source = source_fingerprint(root)
    hash_report = validate_manifest_hashes(root, manifest)
    add(
        "artifact_manifest_is_complete_and_current",
        bool(manifest.get("complete"))
        and not manifest.get("missing_required")
        and manifest.get("source_sha256") == current_source
        and hash_report["valid"],
        {
            "manifest_source": manifest.get("source_sha256", ""),
            "current_source": current_source,
            "missing": manifest.get("missing_required", []),
            "hash_mismatches": hash_report["mismatches"],
        },
    )

    core = validate_interview_package(
        root,
        paths["canonical_pack"],
        paths["reranker_checkpoint"],
        paths["reranker_metrics"],
        paths["reranker_dataset"],
        paths["reranker_dataset_summary"],
        paths["reranker_model_card"],
    )
    for check in core.get("checks", []):
        add(f"core::{check['name']}", bool(check.get("passed")), check.get("evidence"))

    pack = _load(paths["canonical_pack"])
    metadata = dict(pack.get("metadata") or {})
    agent = dict(metadata.get("agent_run") or {})
    pack_sha = canonical_pack_sha256(paths["canonical_pack"])
    pack_file_sha = file_sha256(paths["canonical_pack"])
    video_sha = file_sha256(paths["input_video"])
    registry = _load(paths["best_adapter_registry"])
    selected_id = str(registry.get("selected_candidate_id") or "")
    try:
        selected_paths = candidate_artifact_paths(root, selected_id)
    except ValueError:
        selected_paths = {}
    selected_weights = selected_paths.get("weights")
    adapter_sha = file_sha256(selected_weights) if selected_weights and selected_weights.is_file() else ""
    registry_sha = file_sha256(paths["best_adapter_registry"])
    pack_adapter = dict(metadata.get("llm_adapter") or {})
    resolved_adapter = resolve_validated_adapter(root)
    admission_metadata = adapter_admission_metadata(root)
    add(
        "canonical_uses_hash_bound_admitted_adapter",
        bool(pack_adapter.get("enabled"))
        and bool(pack_adapter.get("validated_for_web"))
        and selected_id in {"qwen35_sft", "qwen35_dpo"}
        and pack_adapter.get("candidate_id") == selected_id
        and pack_adapter.get("adapter_sha256") == adapter_sha
        and pack_adapter.get("registry_sha256") == registry_sha
        and admission_metadata.get("candidate_id") == selected_id
        and bool(resolved_adapter),
        {
            "pack_adapter": pack_adapter,
            "current_adapter_sha256": adapter_sha,
            "registry_sha256": registry_sha,
            "selected_candidate_id": selected_id,
            "resolved_adapter": resolved_adapter,
            "admission_metadata": admission_metadata,
        },
    )
    history_hashes = {
        str(entry.get("sha256") or "")
        for entry in list(manifest.get("adapter_admission_history") or [])
    }
    add(
        "canonical_load_admission_is_immutable",
        bool(pack_adapter.get("evaluation_sha256"))
        and pack_adapter.get("evaluation_sha256") in history_hashes,
        {
            "pack_evaluation_sha256": pack_adapter.get("evaluation_sha256", ""),
            "preserved_evaluation_hashes": sorted(history_hashes),
        },
    )

    records = load_sft_records(paths["sft_dataset"])
    split_report = validate_sft_records(records, project_root=root)
    metrics = _load(paths["sft_metrics"])
    card = _load(paths["sft_model_card"])
    summary = _load(paths["sft_dataset_summary"])
    counts = {
        "train": sum(record.split == "train" for record in records),
        "dev": sum(record.split == "dev" for record in records),
        "test": sum(record.split == "test" for record in records),
    }
    cola_train_dev = [
        record.record_id
        for record in records
        if "cola" in record.video_id.lower() and record.split in {"train", "dev"}
    ]
    frozen_outside_test = [record.record_id for record in records if record.frozen_test and record.split != "test"]
    add(
        "sft_split_isolation_and_frozen_cola_policy",
        split_report["valid"]
        and counts == {"train": 7, "dev": 4, "test": 1}
        and not cola_train_dev
        and not frozen_outside_test
        and summary.get("cola_video_in_train_or_dev") is False,
        {
            "counts": counts,
            "validation": split_report,
            "cola_train_dev": cola_train_dev,
            "frozen_outside_test": frozen_outside_test,
        },
    )
    dataset_sha = file_sha256(paths["sft_dataset"])
    optimizer_sha = gradient_payload_sha256(records)
    metrics_counts = dict(metrics.get("counts") or {})
    add(
        "sft_dataset_and_optimizer_payload_hashes_match",
        metrics.get("dataset_sha256") == dataset_sha
        and metrics.get("current_dataset_sha256", metrics.get("dataset_sha256")) == dataset_sha
        and summary.get("dataset_sha256") == dataset_sha
        and metrics.get("gradient_payload_sha256") == optimizer_sha
        and summary.get("gradient_payload_sha256") == optimizer_sha
        and card.get("data", {}).get("dataset_sha256") == dataset_sha
        and card.get("data", {}).get("gradient_payload_sha256") == optimizer_sha
        and metrics_counts
        == {"train": counts["train"], "dev": counts["dev"], "frozen_test": counts["test"]},
        {
            "dataset_sha256": dataset_sha,
            "optimizer_payload_sha256": optimizer_sha,
            "metrics_dataset_sha256": metrics.get("dataset_sha256", ""),
            "summary_dataset_sha256": summary.get("dataset_sha256", ""),
            "metrics_gradient_payload_sha256": metrics.get("gradient_payload_sha256", ""),
            "summary_gradient_payload_sha256": summary.get("gradient_payload_sha256", ""),
        },
    )
    add(
        "real_lora_training_run_is_recorded",
        metrics.get("status") == "completed"
        and int(metrics.get("steps", 0)) >= 1
        and float(metrics.get("train_loss_last", -1)) >= 0
        and float(metrics.get("dev_loss", -1)) >= 0
        and float(metrics.get("peak_cuda_memory_mib", 0)) > 0
        and bool(metrics.get("physical_gpu_ids"))
        and bool(metrics.get("resume_supported")),
        {
            key: metrics.get(key)
            for key in (
                "status",
                "steps",
                "train_loss_last",
                "dev_loss",
                "tokens_per_second",
                "peak_cuda_memory_mib",
                "physical_gpu_ids",
                "resume_supported",
            )
        },
    )
    sft_checkpoint = _checkpoint_recovery_evidence(paths, "sft", metrics, card)
    add(
        "sft_checkpoint_is_hash_committed_and_resume_safe",
        sft_checkpoint["valid"],
        sft_checkpoint,
    )
    sft_training_source = _training_source_sha256(metrics, card)
    sft_resume = _actual_resume_evidence(paths, "sft", metrics, sft_training_source)
    add(
        "sft_checkpoint_was_actually_resumed_for_an_optimizer_step",
        sft_resume["valid"],
        sft_resume,
    )

    preference_records = load_preference_records(paths["preference_dataset"])
    preference_validation = validate_preference_records(
        preference_records,
        project_root=root,
        source_records=records,
    )
    preference_summary = _load(paths["preference_dataset_summary"])
    preference_counts = {
        "train": sum(record.split == "train" for record in preference_records),
        "dev": sum(record.split == "dev" for record in preference_records),
        "test": sum(record.split == "test" for record in preference_records),
    }
    preference_sha = file_sha256(paths["preference_dataset"])
    preference_gradient_sha = preference_gradient_payload_sha256(preference_records)
    add(
        "preference_data_is_explicit_group_isolated_and_frozen",
        preference_validation["valid"]
        and preference_counts == {"train": 7, "dev": 4, "test": 1}
        and preference_summary.get("dataset_sha256") == preference_sha
        and preference_summary.get("gradient_payload_sha256") == preference_gradient_sha
        and preference_summary.get("cola_video_in_train_or_dev") is False
        and preference_summary.get("negative_type_counts")
        == {
            "wrong_timestamp": 4,
            "missing_timestamp": 1,
            "hallucinated_detail": 2,
            "unsupported_overclaim": 5,
        },
        {
            "counts": preference_counts,
            "negative_type_counts": preference_summary.get("negative_type_counts", {}),
            "dataset_sha256": preference_sha,
            "gradient_payload_sha256": preference_gradient_sha,
            "validation": preference_validation,
        },
    )

    dpo_metrics = _load(paths["dpo_metrics"])
    dpo_card = _load(paths["dpo_model_card"])
    reference_logprobs = _load(paths["dpo_reference_logprobs"])
    reference_sha = file_sha256(paths["dpo_reference_logprobs"])
    sft_adapter_sha = file_sha256(paths["sft_adapter_weights"])
    dpo_dev = dict(dpo_metrics.get("evaluations", {}).get("dev") or {})
    dpo_frozen = dict(dpo_metrics.get("evaluations", {}).get("frozen_test") or {})
    add(
        "real_dpo_run_is_hash_bound_recoverable_and_improves_preferences",
        dpo_metrics.get("status") == "completed"
        and int(dpo_metrics.get("steps", 0)) >= 1
        and dpo_metrics.get("dataset_sha256") == preference_sha
        and dpo_metrics.get("gradient_payload_sha256") == preference_gradient_sha
        and dpo_metrics.get("initial_adapter_sha256") == sft_adapter_sha
        and dpo_metrics.get("reference_logprobs", {}).get("sha256") == reference_sha
        and reference_logprobs.get("initial_adapter_sha256") == sft_adapter_sha
        and reference_logprobs.get("dataset_sha256") == preference_sha
        and bool(dpo_metrics.get("reference_logprobs", {}).get("frozen_before_optimizer"))
        and float(dpo_metrics.get("train_loss_last", -1)) >= 0
        and float(dpo_metrics.get("peak_cuda_memory_mib", 0)) > 0
        and float(dpo_dev.get("mean_reward_margin", 0)) > 0
        and float(dpo_frozen.get("mean_reward_margin", 0)) > 0
        and float(dpo_dev.get("reward_preference_accuracy", 0)) >= 1.0
        and float(dpo_frozen.get("reward_preference_accuracy", 0)) >= 1.0
        and bool(dpo_metrics.get("resume_supported"))
        and dpo_card.get("data", {}).get("frozen_test_excluded_from_gradients") is True,
        {
            "steps": dpo_metrics.get("steps"),
            "train_loss_last": dpo_metrics.get("train_loss_last"),
            "tokens_per_second": dpo_metrics.get("tokens_per_second"),
            "peak_cuda_memory_mib": dpo_metrics.get("peak_cuda_memory_mib"),
            "physical_gpu_ids": dpo_metrics.get("physical_gpu_ids"),
            "dev": dpo_dev,
            "frozen_test": dpo_frozen,
            "reference_sha256": reference_sha,
        },
    )
    dpo_checkpoint = _checkpoint_recovery_evidence(paths, "dpo", dpo_metrics, dpo_card)
    add(
        "dpo_checkpoint_is_hash_committed_and_resume_safe",
        dpo_checkpoint["valid"],
        dpo_checkpoint,
    )
    dpo_training_source = _training_source_sha256(dpo_metrics, dpo_card)
    dpo_resume = _actual_resume_evidence(paths, "dpo", dpo_metrics, dpo_training_source)
    add(
        "dpo_checkpoint_was_actually_resumed_for_an_optimizer_step",
        dpo_resume["valid"],
        dpo_resume,
    )

    evaluation = _load(paths["adapter_evaluation"])
    baseline_eval = _load(paths["adapter_evaluation_baseline"])
    adapter_eval = _load(paths["adapter_evaluation_adapter"])
    comparison = dict(evaluation.get("comparison") or {})
    admission = dict(registry.get("candidates", {}).get(selected_id) or {})
    evaluation_sha = file_sha256(paths["adapter_evaluation"])
    same_pack = all(
        payload.get("pack_sha256") == pack_sha
        for payload in (baseline_eval, adapter_eval, evaluation.get("baseline", {}), evaluation.get("adapter", {}))
    )
    add(
        "adapter_evaluation_and_product_admission_are_bound",
        evaluation.get("status") == "completed"
        and bool(comparison.get("validated_for_web"))
        and bool(comparison.get("adapter_verified"))
        and bool(comparison.get("adapter_timestamp_binding_ok"))
        and bool(comparison.get("adapter_claim_support_ok"))
        and evaluation.get("candidate_id") == selected_id
        and adapter_eval.get("candidate_id") == selected_id
        and same_pack
        and admission.get("pack_sha256") == pack_sha
        and admission.get("video_sha256") == video_sha
        and admission.get("adapter_sha256") == adapter_sha
        and admission.get("evaluation_sha256") == evaluation_sha
        and admission.get("source_sha256") == current_source
        and registry.get("source_sha256") == current_source
        and bool(registry.get("validated_for_web"))
        and resolve_validated_adapter(root) == str(selected_paths.get("adapter", "")),
        {
            "comparison": comparison,
            "same_pack": same_pack,
            "pack_sha256": pack_sha,
            "pack_file_sha256": pack_file_sha,
            "evaluation_sha256": evaluation_sha,
            "admission": admission,
        },
    )

    source_bound = {
        "canonical_pack": metadata.get("source_sha256", ""),
        "adapter_eval_baseline": baseline_eval.get("source_sha256", ""),
        "adapter_eval_adapter": adapter_eval.get("source_sha256", ""),
        "error_analysis": _load(paths["error_analysis"]).get("source_sha256", ""),
        "performance_report": _load(paths["performance_report"]).get("source_sha256", ""),
        "sft_eval_baseline": _load(paths["sft_adapter_evaluation"]).get("baseline", {}).get("source_sha256", ""),
        "sft_eval_candidate": _load(paths["sft_adapter_evaluation"]).get("adapter", {}).get("source_sha256", ""),
        "dpo_eval_baseline": _load(paths["dpo_adapter_evaluation"]).get("baseline", {}).get("source_sha256", ""),
        "dpo_eval_candidate": _load(paths["dpo_adapter_evaluation"]).get("adapter", {}).get("source_sha256", ""),
    }
    add(
        "source_bound_runtime_artifacts_match_final_snapshot",
        all(value == current_source for value in source_bound.values()),
        {"current": current_source, "artifacts": source_bound},
    )

    regression = _load(paths["error_analysis"])
    error_counts = dict(regression.get("error_category_counts") or {})
    add(
        "frozen_task_regression_passes_all_error_classes",
        int(regression.get("num_cases", 0)) == 5
        and int(regression.get("num_passed", -1)) == 5
        and error_counts == {"none": 5}
        and all(bool(case.get("passed")) for case in regression.get("cases", []))
        and all(bool(case.get("claim_support_ok")) for case in regression.get("cases", []))
        and regression.get("video_sha256") == video_sha,
        {
            "num_cases": regression.get("num_cases"),
            "num_passed": regression.get("num_passed"),
            "error_category_counts": error_counts,
        },
    )

    performance = _load(paths["performance_report"])
    residency = dict(performance.get("model_residency") or {})
    correctness = dict(performance.get("precision", {}).get("correctness_check") or {})
    cold_seconds = _float(residency.get("cold_pipeline_construction_and_run_seconds"), 0.0)
    warm_seconds = _float(residency.get("warm_cache_hit_run_seconds"), 0.0)
    warm_cache = dict(performance.get("cache_reuse", {}).get("warm") or {})
    add(
        "runtime_profile_has_speed_memory_cache_and_correctness_evidence",
        performance.get("video_sha256") == video_sha
        and cold_seconds > warm_seconds > 0
        and _float(residency.get("speedup"), 0.0) > 1.0
        and bool(residency.get("same_pipeline_instance"))
        and all(bool(value) for value in correctness.values())
        and int(warm_cache.get("segment_understanding", {}).get("hits", 0)) > 0
        and int(warm_cache.get("vlm", {}).get("hits", 0)) > 0
        and _peak_memory(performance) > 0
        and "cpu-only" in str(performance.get("precision", {}).get("int8_or_int4", "")).lower(),
        {
            "model_residency": residency,
            "correctness": correctness,
            "warm_cache": warm_cache,
            "peak_cuda_memory_mib": _peak_memory(performance),
            "precision": performance.get("precision", {}),
        },
    )

    recovery = _load(paths["agent_failure_recovery"])
    outcomes = [str(event.get("outcome")) for event in recovery.get("events", [])]
    trace_codes = [str(item.get("error_code")) for item in recovery.get("tool_trace", [])]
    add(
        "agent_failure_recovery_is_bounded_and_non_hallucinatory",
        bool(recovery.get("passed"))
        and int(recovery.get("attempts", 0)) == 4
        and "retry_exhausted" in outcomes
        and "controlled_fallback" in outcomes
        and "circuit_open" in trace_codes
        and "do not fabricate" in str(recovery.get("final_action", "")),
        {
            "attempts": recovery.get("attempts"),
            "outcomes": outcomes,
            "trace_error_codes": trace_codes,
            "final_action": recovery.get("final_action", ""),
        },
    )

    gpu = _load(paths["gpu_selection_canonical"])
    selected = [int(value) for value in gpu.get("selected_physical_gpu_ids", [])]
    required = int(gpu.get("required_stable_checks", 0))
    probes = list(gpu.get("probes") or [])
    stable_probes = probes[-required:] if required > 0 else []
    thresholds = dict(gpu.get("thresholds") or {})
    gpu_safe = bool(stable_probes) and all(
        probe.get("selected_pair") == selected
        and all(
            not state.get("compute_pids")
            and int(state.get("used_mib", 10**9)) <= int(thresholds.get("max_used_mib", -1))
            and int(state.get("utilization_pct", 10**9)) <= int(thresholds.get("max_utilization_pct", -1))
            for state in probe.get("states", [])
            if int(state.get("index", -1)) in selected
        )
        and sum(int(state.get("index", -1)) in selected for state in probe.get("states", [])) == len(selected)
        for probe in stable_probes
    )
    recorded_pair = ",".join(str(value) for value in selected)
    add(
        "gpu_selection_has_three_non_interfering_stable_probes",
        gpu.get("status") == "selected"
        and len(selected) == 2
        and required >= 3
        and gpu_safe
        and metadata.get("environment", {}).get("physical_gpu_ids") == recorded_pair
        and "never terminates" in str(gpu.get("non_interference_policy", "")),
        {
            "selected": selected,
            "required_stable_checks": required,
            "stable_probe_count": len(stable_probes),
            "gpu_safe": gpu_safe,
            "pack_physical_gpu_ids": metadata.get("environment", {}).get("physical_gpu_ids", ""),
            "sft_training_physical_gpu_ids": metrics.get("physical_gpu_ids", ""),
            "note": "canonical inference and historical SFT training are independently audited runs",
        },
    )
    dpo_gpu = _load(paths["gpu_selection_dpo"])
    dpo_selected = [int(value) for value in dpo_gpu.get("selected_physical_gpu_ids", [])]
    dpo_required = int(dpo_gpu.get("required_stable_checks", 0))
    dpo_probes = list(dpo_gpu.get("probes") or [])
    dpo_stable = dpo_probes[-dpo_required:] if dpo_required > 0 else []
    dpo_thresholds = dict(dpo_gpu.get("thresholds") or {})
    dpo_safe = bool(dpo_stable) and all(
        probe.get("selected_pair") == dpo_selected
        and all(
            not state.get("compute_pids")
            and int(state.get("used_mib", 10**9)) <= int(dpo_thresholds.get("max_used_mib", -1))
            and int(state.get("utilization_pct", 10**9)) <= int(
                dpo_thresholds.get("max_utilization_pct", -1)
            )
            for state in probe.get("states", [])
            if int(state.get("index", -1)) in dpo_selected
        )
        for probe in dpo_stable
    )
    dpo_training_gpu = str(dpo_metrics.get("physical_gpu_ids") or "")
    dpo_binding = _dpo_gpu_binding_evidence(dpo_metrics, dpo_selected)
    add(
        "dpo_gpu_was_stably_selected_without_interrupting_existing_processes",
        dpo_gpu.get("status") == "selected"
        and len(dpo_selected) == 2
        and dpo_required >= 3
        and dpo_safe
        and dpo_binding["valid"]
        and "never terminates" in str(dpo_gpu.get("non_interference_policy", "")),
        {
            "selected_pair": dpo_selected,
            "stable_probe_count": len(dpo_stable),
            "gpu_safe": dpo_safe,
            "training_physical_gpu_ids": dpo_training_gpu,
            "cuda_visible_devices": dpo_metrics.get("cuda_visible_devices", ""),
            "model_parallel": dpo_metrics.get("model_parallel", {}),
            "binding": dpo_binding,
        },
    )
    verifier_rows = [
        json.loads(line)
        for line in paths["answer_verifier_dataset"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    verifier_validation = validate_answer_verifier_rows(verifier_rows)
    verifier_summary = _load(paths["answer_verifier_dataset_summary"])
    verifier_metrics = _load(paths["answer_verifier_metrics"])
    verifier_card = _load(paths["answer_verifier_model_card"])
    verifier_checkpoint_sha = file_sha256(paths["answer_verifier_checkpoint"])
    verifier_train_ids = {
        str(row.get("example_id"))
        for row in verifier_rows
        if str(row.get("split")) == "train"
    }
    verifier_frozen_ids = {
        str(row.get("example_id"))
        for row in verifier_rows
        if str(row.get("split")) == "test"
    }
    add(
        "calibrated_answer_verifier_is_split_isolated_and_product_validated",
        verifier_validation.get("valid") is True
        and verifier_validation.get("counts") == {"train": 14, "dev": 8, "test": 2}
        and verifier_train_ids.isdisjoint(verifier_frozen_ids)
        and verifier_summary.get("frozen_test_excluded_from_gradient") is True
        and verifier_metrics.get("frozen_test_excluded_from_gradient") is True
        and verifier_metrics.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
        and verifier_metrics.get("feature_contract") == FEATURE_CONTRACT_VERSION
        and verifier_metrics.get("feature_names") == FEATURE_NAMES
        and verifier_metrics.get("checkpoint_sha256") == verifier_checkpoint_sha
        and verifier_card.get("checkpoint", {}).get("sha256") == verifier_checkpoint_sha
        and verifier_metrics.get("validated_for_product") is True
        and float(verifier_metrics.get("evaluations", {}).get("dev", {}).get("safe_recall", 0.0)) >= 1.0
        and float(verifier_metrics.get("evaluations", {}).get("frozen_test", {}).get("pairwise_accuracy", 0.0)) >= 1.0,
        {
            "validation": verifier_validation,
            "counts": verifier_metrics.get("counts", {}),
            "checkpoint_sha256": verifier_checkpoint_sha,
            "dev": verifier_metrics.get("evaluations", {}).get("dev", {}),
            "frozen_test": verifier_metrics.get("evaluations", {}).get("frozen_test", {}),
            "policy": verifier_card.get("intended_use", ""),
        },
    )
    pack_verifier = dict(metadata.get("answer_verifier") or {})
    run_verifier = dict(agent.get("verification", {}).get("calibrated_verifier") or {})
    add(
        "canonical_uses_portable_calibrated_answer_verifier",
        pack_verifier.get("backend") == "calibrated"
        and pack_verifier.get("loaded") is True
        and pack_verifier.get("model_format") == "portable-numpy-logistic-v1"
        and pack_verifier.get("checkpoint_sha256") == verifier_checkpoint_sha
        and float(pack_verifier.get("threshold", -1.0)) == float(verifier_metrics.get("threshold", -2.0))
        and run_verifier.get("enabled") is True
        and run_verifier.get("passed") is True
        and run_verifier.get("checkpoint_sha256") == verifier_checkpoint_sha,
        {
            "pack": pack_verifier,
            "agent_run": run_verifier,
            "artifact_checkpoint_sha256": verifier_checkpoint_sha,
        },
    )
    browser = _load(paths["browser_e2e"])
    browser_checks = dict(browser.get("checks") or {})
    job_events = list(browser.get("job", {}).get("events") or [])
    event_phases = [str(event.get("phase") or "") for event in job_events]
    required_browser_checks = {
        "core_controls_visible",
        "requested_mode_available",
        "desktop_no_horizontal_overflow",
        "immediate_blob_preview",
        "upload_switched_to_remote_media",
        "answer_evidence_timeline_rendered",
        "evidence_auto_pause_releases_window",
        "continue_from_current_plays",
        "timeline_continues_without_window",
        "http_range_206",
        "mobile_no_horizontal_overflow",
        "completed_elapsed_is_frozen",
        "completed_elapsed_matches_execution_window",
        "console_clean",
    }
    add(
        "browser_e2e_and_durable_job_observability_pass",
        bool(browser.get("valid"))
        and browser.get("source_sha256") == current_source
        and browser.get("video_sha256") == video_sha
        and required_browser_checks.issubset(browser_checks)
        and all(bool(browser_checks[name]) for name in required_browser_checks)
        and browser.get("job", {}).get("status") == "completed"
        and bool(browser.get("job", {}).get("persistence", {}).get("durable"))
        and {"queued", "checking_resources", "loading_models", "analyzing", "exporting", "completed"}.issubset(
            set(event_phases)
        )
        and not browser.get("console_errors")
        and not browser.get("console_warnings")
        and not browser.get("page_errors"),
        {
            "valid": browser.get("valid"),
            "checks": browser_checks,
            "job_id": browser.get("job_id", ""),
            "job_persistence": browser.get("job", {}).get("persistence", {}),
            "event_phases": event_phases,
            "range": browser.get("range", {}),
        },
    )
    return _report(checks, paths)


def _report(checks: list[dict], paths: Mapping[str, Path]) -> dict:
    failures = [check["name"] for check in checks if not check["passed"]]
    return {
        "valid": not failures,
        "checks_passed": len(checks) - len(failures),
        "checks_total": len(checks),
        "failures": failures,
        "checks": checks,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def _rooted(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dpo_gpu_binding_evidence(metrics: dict, selected_physical_ids: list[int]) -> dict:
    """Validate the recorded DPO GPU binding for single- and multi-GPU runs.

    ``CUDA_VISIBLE_DEVICES`` and ``VIDEOTRACE_PHYSICAL_GPUS`` record physical
    ids, while Accelerate/Transformers reports trainable parameter placement
    using logical ``cuda:0..N`` ids after visibility remapping.  Treating every
    run as the legacy single-GPU ``cuda:0`` path rejects a real model-parallel
    checkpoint and encourages falsifying the audit metadata.
    """

    physical_ids = _gpu_id_list(metrics.get("physical_gpu_ids"))
    visible_ids = _gpu_id_list(metrics.get("cuda_visible_devices"))
    model_parallel = dict(metrics.get("model_parallel") or {})
    active = bool(model_parallel.get("active"))
    trainable_devices = sorted(
        {str(value) for value in model_parallel.get("trainable_parameter_devices", [])}
    )
    if active:
        expected_devices = [f"cuda:{index}" for index in range(len(selected_physical_ids))]
        valid = (
            len(selected_physical_ids) >= 2
            and physical_ids == selected_physical_ids
            and visible_ids == selected_physical_ids
            and trainable_devices == expected_devices
        )
        mode = "model_parallel"
    else:
        valid = (
            len(physical_ids) == 1
            and physical_ids == visible_ids
            and physical_ids[0] in selected_physical_ids
            and not trainable_devices
        )
        mode = "single_gpu"
    return {
        "valid": valid,
        "mode": mode,
        "selected_physical_gpu_ids": selected_physical_ids,
        "recorded_physical_gpu_ids": physical_ids,
        "cuda_visible_devices": visible_ids,
        "trainable_parameter_devices": trainable_devices,
    }


def _gpu_id_list(value: object) -> list[int]:
    if isinstance(value, (list, tuple)):
        tokens = list(value)
    else:
        tokens = str(value or "").split(",")
    try:
        return [int(str(token).strip()) for token in tokens if str(token).strip()]
    except ValueError:
        return []


def _peak_memory(performance: dict) -> float:
    devices = performance.get("warm_pack_performance", {}).get("cuda_memory", {}).get("devices", [])
    return max((_float(item.get("peak_allocated_mib"), 0.0) for item in devices), default=0.0)


def _checkpoint_recovery_evidence(
    paths: Mapping[str, Path],
    prefix: str,
    metrics: dict,
    model_card: dict,
) -> dict:
    manifest_path = paths[f"{prefix}_checkpoint_manifest"]
    trainer_state_path = paths[f"{prefix}_trainer_state"]
    checkpoint_dir = manifest_path.parent
    manifest = _load(manifest_path)
    trainer_state = _load(trainer_state_path)
    expected_contract = str(metrics.get("checkpoint_contract_sha256") or "")
    manifest_files = dict(manifest.get("files") or {})
    mismatches = []
    for name, expected_sha in manifest_files.items():
        path = checkpoint_dir / name
        current_sha = file_sha256(path) if path.is_file() else ""
        if current_sha != expected_sha:
            mismatches.append(
                {
                    "name": name,
                    "expected_sha256": expected_sha,
                    "current_sha256": current_sha,
                }
            )
    required_names = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng_state.pt",
        "trainer_state.json",
    }
    recorded_files = set(str(item) for item in metrics.get("checkpoint_files", []))
    card = dict(model_card.get("checkpoint_recovery") or {})
    valid = (
        bool(expected_contract)
        and manifest.get("contract_sha256") == expected_contract
        and trainer_state.get("contract", {}).get("contract_sha256") == expected_contract
        and int(manifest.get("global_step", -1)) == int(metrics.get("steps", -2))
        and int(trainer_state.get("global_step", -1)) == int(metrics.get("steps", -2))
        and metrics.get("checkpoint_manifest_sha256") == file_sha256(manifest_path)
        and card.get("contract_sha256") == expected_contract
        and card.get("manifest_sha256") == file_sha256(manifest_path)
        and card.get("resume_supported") is True
        and required_names.issubset(manifest_files)
        and recorded_files == set(manifest_files)
        and set(card.get("files") or []) == recorded_files
        and not mismatches
    )
    return {
        "valid": valid,
        "method": prefix,
        "global_step": metrics.get("steps"),
        "contract_sha256": expected_contract,
        "manifest_sha256": file_sha256(manifest_path),
        "required_files": sorted(required_names),
        "recorded_files": sorted(recorded_files),
        "hash_mismatches": mismatches,
    }


def _actual_resume_evidence(
    paths: Mapping[str, Path],
    prefix: str,
    canonical_metrics: dict,
    expected_training_source: str,
) -> dict:
    report = _load(paths[f"{prefix}_resume_validation"])
    manifest_path = paths[f"{prefix}_resume_checkpoint_manifest"]
    manifest = _load(manifest_path)
    checkpoint_dir = manifest_path.parent
    manifest_files = dict(manifest.get("files") or {})
    mismatches = []
    for name, expected_sha in manifest_files.items():
        path = checkpoint_dir / name
        current_sha = file_sha256(path) if path.is_file() else ""
        if current_sha != expected_sha:
            mismatches.append(
                {
                    "name": name,
                    "expected_sha256": expected_sha,
                    "current_sha256": current_sha,
                }
            )
    required_names = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng_state.pt",
        "trainer_state.json",
    }
    canonical_step = int(canonical_metrics.get("steps", 0))
    resumed_from_step = int(report.get("resumed_from_step", -1))
    final_step = int(report.get("steps", -1))
    report_files = set(str(item) for item in report.get("checkpoint_files", []))
    expected_resume_suffix = f"outputs/models/qwen35_{prefix}_adapter"
    actual_resume_path = str(report.get("resumed_from_checkpoint") or "").replace("\\", "/")
    expected_output_suffix = f"outputs/runs/latest/{prefix}_resume_validation"
    actual_output_path = str(report.get("adapter_path") or "").replace("\\", "/")
    report_training_source = str(
        report.get("training_source_sha256")
        or report.get("source_sha256")
        or ""
    )
    same_reference = True
    if prefix == "dpo":
        same_reference = (
            report.get("initial_adapter_sha256") == canonical_metrics.get("initial_adapter_sha256")
            and report.get("reference_logprobs", {}).get("sha256")
            == canonical_metrics.get("reference_logprobs", {}).get("sha256")
        )
    valid = (
        report.get("status") == "completed"
        and bool(expected_training_source)
        and report_training_source == expected_training_source
        and canonical_step >= 1
        and resumed_from_step == canonical_step
        and final_step > resumed_from_step
        and int(report.get("steps_this_run", 0)) >= 1
        and report.get("checkpoint_contract_sha256")
        == canonical_metrics.get("checkpoint_contract_sha256")
        and report.get("dataset_sha256") == canonical_metrics.get("dataset_sha256")
        and report.get("gradient_payload_sha256")
        == canonical_metrics.get("gradient_payload_sha256")
        and bool(report.get("physical_gpu_ids"))
        and actual_resume_path.endswith(expected_resume_suffix)
        and actual_output_path.endswith(expected_output_suffix)
        and report.get("checkpoint_manifest_sha256") == file_sha256(manifest_path)
        and manifest.get("contract_sha256") == report.get("checkpoint_contract_sha256")
        and int(manifest.get("global_step", -1)) == final_step
        and required_names.issubset(manifest_files)
        and report_files == set(manifest_files)
        and same_reference
        and not mismatches
    )
    return {
        "valid": valid,
        "method": prefix,
        "canonical_step": canonical_step,
        "resumed_from_step": resumed_from_step,
        "steps_this_run": report.get("steps_this_run", 0),
        "final_step": final_step,
        "training_source_sha256": report_training_source,
        "expected_training_source_sha256": expected_training_source,
        "physical_gpu_ids": report.get("physical_gpu_ids", ""),
        "resume_checkpoint": actual_resume_path,
        "output_checkpoint": actual_output_path,
        "manifest_sha256": file_sha256(manifest_path),
        "required_files": sorted(required_names),
        "hash_mismatches": mismatches,
        "same_reference_chain": same_reference,
    }


def _training_source_sha256(metrics: dict, model_card: dict) -> str:
    reproducibility = dict(model_card.get("reproducibility") or {})
    return str(
        metrics.get("training_source_sha256")
        or model_card.get("training_source_sha256")
        or reproducibility.get("training_source_sha256")
        or reproducibility.get("source_sha256")
        or metrics.get("source_sha256")
        or ""
    )
