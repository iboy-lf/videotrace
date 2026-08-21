from __future__ import annotations

"""Select the product adapter from fixed SFT/DPO candidates and bind its hashes."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from videomemo.eval.reproducibility import file_sha256, source_fingerprint
from videomemo.llm.adapter_admission import (
    ADAPTER_CANDIDATES,
    BEST_ADAPTER_SCHEMA_VERSION,
    candidate_artifact_paths,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-select-best-qwen35-adapter")
    parser.add_argument("--sft-evaluation", default="outputs/reports/qwen35_sft_eval.json")
    parser.add_argument("--dpo-evaluation", default="outputs/reports/qwen35_dpo_eval.json")
    parser.add_argument("--output", default="outputs/models/best_adapter.json")
    parser.add_argument("--selected-evaluation", default="outputs/reports/qwen35_adapter_eval.json")
    parser.add_argument(
        "--selected-baseline-evaluation",
        default="outputs/reports/qwen35_adapter_eval_baseline.json",
    )
    parser.add_argument("--selected-candidate-evaluation", default="outputs/reports/qwen35_adapter_eval_adapter.json")
    args = parser.parse_args()

    current_source = source_fingerprint(ROOT)
    inputs = {
        "qwen35_sft": _rooted(args.sft_evaluation),
        "qwen35_dpo": _rooted(args.dpo_evaluation),
    }
    candidates = {
        candidate_id: _candidate_entry(candidate_id, evaluation_path, current_source)
        for candidate_id, evaluation_path in inputs.items()
    }
    sft_ok = bool(candidates["qwen35_sft"]["validated_for_web"])
    dpo_ok = bool(candidates["qwen35_dpo"]["validated_for_web"])
    dpo_preference_ok = bool(candidates["qwen35_dpo"].get("preference_gate", {}).get("passed"))
    if dpo_ok and dpo_preference_ok:
        selected = "qwen35_dpo"
        decision = (
            "DPO selected: frozen product regression passed and the real preference run improved "
            "reference-relative margins on dev and frozen cola; SFT remains the hash-validated fallback."
        )
    elif sft_ok:
        selected = "qwen35_sft"
        decision = "SFT selected: DPO did not pass every product/preference gate; keep the admitted safe fallback."
    else:
        raise SystemExit("no post-training adapter passed the fixed product admission gates")

    fallback_order = [candidate_id for candidate_id in ("qwen35_sft",) if candidate_id != selected]
    selected_entry = candidates[selected]
    selected_evaluation_path = inputs[selected]
    selected_evaluation = _read(selected_evaluation_path)
    selected_baseline_path = _rooted(args.selected_baseline_evaluation)
    selected_variant_path = _rooted(args.selected_candidate_evaluation)
    selected_report_path = _rooted(args.selected_evaluation)
    _publish_selected_reports(
        selected_evaluation_path,
        selected_evaluation,
        selected_report_path,
        selected_baseline_path,
        selected_variant_path,
    )

    registry = {
        "schema_version": BEST_ADAPTER_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validated_for_web": True,
        "source_sha256": current_source,
        "selected_candidate_id": selected,
        "fallback_order": fallback_order,
        "decision": decision,
        "selection_policy": {
            "browser_adapter_selection": False,
            "candidate_paths": "fixed server whitelist",
            "product_gate": "grounding, timestamp binding and baseline coverage non-regression on the frozen cola pack",
            "dpo_gate": "completed real LoRA step plus positive dev/frozen reference-relative reward margins",
            "tie_break": "prefer DPO only when both product and preference gates pass; otherwise keep SFT",
        },
        "candidates": candidates,
        "selected_compatibility_reports": {
            "comparison": str(selected_report_path),
            "baseline": str(selected_baseline_path),
            "candidate": str(selected_variant_path),
        },
    }
    output = _rooted(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    registry_sha = file_sha256(temp)
    _bind_pack_metadata(selected_evaluation, selected, selected_entry, registry_sha, output)
    temp.replace(output)
    print(
        json.dumps(
            {
                "ok": True,
                "selected_candidate_id": selected,
                "fallback_order": fallback_order,
                "registry": str(output),
                "registry_sha256": registry_sha,
                "decision": decision,
                "candidates": candidates,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _candidate_entry(candidate_id: str, evaluation_path: Path, current_source: str) -> dict:
    if candidate_id not in ADAPTER_CANDIDATES:
        raise RuntimeError(f"candidate is not whitelisted: {candidate_id}")
    paths = candidate_artifact_paths(ROOT, candidate_id)
    missing = [name for name in ("weights", "config", "metrics", "model_card") if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError(f"{candidate_id} missing artifacts: {missing}")
    _bind_candidate_provenance(paths, current_source)
    evaluation = _read(evaluation_path)
    comparison = dict(evaluation.get("comparison") or {})
    adapter_eval = dict(evaluation.get("adapter") or {})
    baseline_eval = dict(evaluation.get("baseline") or {})
    source_match = (
        adapter_eval.get("source_sha256") == current_source
        and baseline_eval.get("source_sha256") == current_source
    )
    same_pack = bool(adapter_eval.get("pack_sha256")) and (
        adapter_eval.get("pack_sha256") == baseline_eval.get("pack_sha256")
    )
    same_video = bool(adapter_eval.get("video_sha256")) and (
        adapter_eval.get("video_sha256") == baseline_eval.get("video_sha256")
    )
    candidate_identity = adapter_eval.get("candidate_id") == candidate_id
    product_passed = bool(
        evaluation.get("status") == "completed"
        and comparison.get("validated_for_web")
        and comparison.get("adapter_verified")
        and comparison.get("adapter_timestamp_binding_ok")
        and comparison.get("adapter_claim_support_ok")
        and comparison.get("coverage_non_regression")
        and source_match
        and same_pack
        and same_video
        and candidate_identity
    )
    metrics = _read(paths["metrics"])
    preference_gate = _preference_gate(candidate_id, metrics, paths)
    immutable = _preserve_evaluation(candidate_id, evaluation_path)
    return {
        "candidate_id": candidate_id,
        "method": ADAPTER_CANDIDATES[candidate_id].method,
        "validated_for_web": product_passed and bool(preference_gate["passed"]),
        "product_gate": {
            "passed": product_passed,
            "source_match": source_match,
            "same_pack": same_pack,
            "same_video": same_video,
            "candidate_identity": candidate_identity,
            "comparison": comparison,
        },
        "preference_gate": preference_gate,
        "adapter_path": ADAPTER_CANDIDATES[candidate_id].adapter_relative,
        "adapter_sha256": file_sha256(paths["weights"]),
        "adapter_config_sha256": file_sha256(paths["config"]),
        "metrics_sha256": file_sha256(paths["metrics"]),
        "model_card_sha256": file_sha256(paths["model_card"]),
        "evaluation_report": str(immutable),
        "evaluation_latest_report": str(evaluation_path),
        "evaluation_sha256": file_sha256(evaluation_path),
        "pack_sha256": adapter_eval.get("pack_sha256", ""),
        "video_sha256": adapter_eval.get("video_sha256", ""),
        "source_sha256": current_source,
        "training_source_sha256": metrics.get("training_source_sha256", ""),
        "admission_source_sha256": metrics.get("admission_source_sha256", ""),
        "dataset_sha256": metrics.get("dataset_sha256", ""),
        "gradient_payload_sha256": metrics.get("gradient_payload_sha256", ""),
    }


def _bind_candidate_provenance(paths: dict[str, Path], current_source: str) -> None:
    """Separate immutable training provenance from the current product admission."""
    metrics = _read(paths["metrics"])
    card = _read(paths["model_card"])
    reproducibility = dict(card.get("reproducibility") or {})
    training_source = str(
        metrics.get("training_source_sha256")
        or card.get("training_source_sha256")
        or reproducibility.get("training_source_sha256")
        or reproducibility.get("source_sha256")
        or metrics.get("source_sha256")
        or ""
    )
    if not training_source:
        raise RuntimeError("candidate training source fingerprint is missing")
    metrics["training_source_sha256"] = training_source
    metrics["source_sha256"] = current_source
    metrics["admission_source_sha256"] = current_source
    card["training_source_sha256"] = training_source
    card["source_sha256"] = current_source
    card["admission_source_sha256"] = current_source
    reproducibility["source_sha256"] = training_source
    reproducibility["training_source_sha256"] = training_source
    reproducibility["admission_source_sha256"] = current_source
    card["reproducibility"] = reproducibility
    _atomic_json(paths["metrics"], metrics)
    _atomic_json(paths["model_card"], card)


def _preference_gate(candidate_id: str, metrics: dict, paths: dict[str, Path]) -> dict:
    if candidate_id == "qwen35_sft":
        passed = bool(
            metrics.get("status") == "completed"
            and int(metrics.get("steps", 0)) >= 1
            and float(metrics.get("peak_cuda_memory_mib", 0)) > 0
            and bool(metrics.get("resume_supported"))
        )
        return {
            "passed": passed,
            "kind": "SFT training provenance",
            "steps": metrics.get("steps", 0),
            "peak_cuda_memory_mib": metrics.get("peak_cuda_memory_mib", 0),
        }

    sft_paths = candidate_artifact_paths(ROOT, "qwen35_sft")
    evaluations = dict(metrics.get("evaluations") or {})
    dev = dict(evaluations.get("dev") or {})
    frozen = dict(evaluations.get("frozen_test") or {})
    sft_sha = file_sha256(sft_paths["weights"]) if sft_paths["weights"].is_file() else ""
    passed = bool(
        metrics.get("status") == "completed"
        and int(metrics.get("steps", 0)) >= 1
        and metrics.get("initial_adapter_sha256") == sft_sha
        and float(metrics.get("peak_cuda_memory_mib", 0)) > 0
        and float(dev.get("mean_reward_margin", 0)) > 0
        and float(frozen.get("mean_reward_margin", 0)) > 0
        and float(dev.get("reward_preference_accuracy", 0)) >= 1.0
        and float(frozen.get("reward_preference_accuracy", 0)) >= 1.0
        and metrics.get("counts") == {"train": 7, "dev": 4, "frozen_test": 1}
        and bool(metrics.get("reference_logprobs", {}).get("frozen_before_optimizer"))
        and bool(metrics.get("resume_supported"))
    )
    return {
        "passed": passed,
        "kind": "DPO reference-relative preference and provenance gate",
        "steps": metrics.get("steps", 0),
        "initial_sft_hash_matches": metrics.get("initial_adapter_sha256") == sft_sha,
        "dev_mean_reward_margin": dev.get("mean_reward_margin", 0),
        "dev_reward_preference_accuracy": dev.get("reward_preference_accuracy", 0),
        "frozen_mean_reward_margin": frozen.get("mean_reward_margin", 0),
        "frozen_reward_preference_accuracy": frozen.get("reward_preference_accuracy", 0),
        "peak_cuda_memory_mib": metrics.get("peak_cuda_memory_mib", 0),
    }


def _preserve_evaluation(candidate_id: str, path: Path) -> Path:
    digest = file_sha256(path)
    target = ROOT / "outputs" / "reports" / "adapter_admissions" / f"{candidate_id}_eval_{digest}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha256(target) != digest:
            raise RuntimeError(f"immutable evaluation hash mismatch: {target}")
        return target.resolve()
    temp = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(path, temp)
    if file_sha256(temp) != digest:
        temp.unlink(missing_ok=True)
        raise RuntimeError("evaluation changed while preserving immutable copy")
    temp.replace(target)
    return target.resolve()


def _bind_pack_metadata(
    evaluation: dict,
    selected: str,
    entry: dict,
    registry_sha: str,
    registry_path: Path,
) -> None:
    pack_value = str(evaluation.get("adapter", {}).get("pack_path") or "").strip()
    pack_path = Path(pack_value).expanduser() if pack_value else Path()
    if not pack_path.is_file():
        candidates = (
            ROOT / "outputs" / "iboy_qwen35" / "cola_review" / "knowledge_pack.json",
            ROOT / "outputs" / "cola_review_qwen35" / "knowledge_pack.json",
        )
        pack_path = next((path for path in candidates if path.is_file()), pack_path)
    if not pack_path.is_file():
        raise FileNotFoundError(f"evaluated canonical pack not found: {pack_value}")
    payload = _read(pack_path)
    metadata = dict(payload.get("metadata") or {})
    metadata["llm_adapter"] = {
        "enabled": True,
        "validated_for_web": True,
        "candidate_id": selected,
        "method": entry.get("method", ""),
        "path": str(candidate_artifact_paths(ROOT, selected)["adapter"]),
        "adapter_sha256": entry.get("adapter_sha256", ""),
        "evaluation_sha256": entry.get("evaluation_sha256", ""),
        "pack_sha256": entry.get("pack_sha256", ""),
        "video_sha256": entry.get("video_sha256", ""),
        "registry_path": str(registry_path),
        "registry_sha256": registry_sha,
    }
    payload["metadata"] = metadata
    _atomic_json(pack_path, payload)


def _copy_exact(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temp)
    temp.replace(target)


def _publish_selected_reports(
    source: Path,
    evaluation: dict,
    comparison_target: Path,
    baseline_target: Path,
    candidate_target: Path,
) -> None:
    """Publish one internally consistent compatibility-report triplet.

    Delivery validation consumes the generic comparison, baseline and adapter
    paths.  Copying only the comparison and adapter can leave a baseline from a
    previous source snapshot (or no baseline at all), even though the selected
    candidate evaluation itself is valid.
    """

    baseline = dict(evaluation.get("baseline") or {})
    candidate = dict(evaluation.get("adapter") or {})
    if not baseline or not candidate:
        raise ValueError("selected evaluation must contain baseline and adapter variants")
    _copy_exact(source, comparison_target)
    _atomic_json(baseline_target, baseline)
    _atomic_json(candidate_target, candidate)


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
