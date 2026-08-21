from __future__ import annotations

"""Write the explicit, hash-bound product admission for a Qwen SFT adapter."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from videomemo.eval.reproducibility import file_sha256, source_fingerprint
from videomemo.training.sft_data import gradient_payload_sha256, load_sft_records


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-admit-qwen35-adapter")
    parser.add_argument("--evaluation", default="outputs/reports/qwen35_adapter_eval.json")
    parser.add_argument("--metrics", default="outputs/models/qwen35_sft_metrics.json")
    parser.add_argument("--model-card", default="outputs/models/qwen35_sft_model_card.json")
    parser.add_argument("--adapter", default="outputs/models/qwen35_sft_adapter")
    args = parser.parse_args()

    evaluation_path = _rooted(args.evaluation)
    metrics_path = _rooted(args.metrics)
    card_path = _rooted(args.model_card)
    adapter_path = _rooted(args.adapter)
    evaluation = _read(evaluation_path)
    decision = dict(evaluation.get("comparison") or {})
    if evaluation.get("status") != "completed":
        raise SystemExit("adapter evaluation is not completed")
    if not bool(decision.get("validated_for_web")):
        raise SystemExit(
            "adapter failed frozen admission gates: "
            + str(decision.get("decision_reason", "unknown reason"))
        )
    if not (adapter_path / "adapter_config.json").exists():
        raise SystemExit(f"adapter_config.json not found: {adapter_path}")

    metrics = _read(metrics_path)
    card = _read(card_path)
    current_source = source_fingerprint(ROOT)
    adapter_hash = file_sha256(adapter_path / "adapter_model.safetensors")
    evaluation_hash = file_sha256(evaluation_path)
    immutable_evaluation = _preserve_evaluation(evaluation_path, evaluation_hash)
    now = datetime.now(timezone.utc).isoformat()
    admission = {
        "validated_for_web": True,
        "admitted_at_utc": now,
        "evaluation_report": str(immutable_evaluation),
        "evaluation_latest_report": str(evaluation_path),
        "evaluation_sha256": evaluation_hash,
        "pack_sha256": evaluation.get("adapter", {}).get("pack_sha256", ""),
        "video_sha256": evaluation.get("adapter", {}).get("video_sha256", ""),
        "adapter_sha256": adapter_hash,
        "source_sha256": current_source,
        "decision": decision.get("decision_reason", "adapter passed frozen evidence and non-regression gates"),
        "gates": {
            "adapter_verified": bool(decision.get("adapter_verified")),
            "timestamp_binding_ok": bool(decision.get("adapter_timestamp_binding_ok")),
            "coverage_non_regression": bool(decision.get("coverage_non_regression")),
            "baseline_coverage": float(decision.get("baseline_coverage", 0.0)),
            "adapter_coverage": float(decision.get("adapter_coverage", 0.0)),
        },
    }
    training_source = str(
        metrics.get("training_source_sha256")
        or card.get("training_source_sha256")
        or card.get("reproducibility", {}).get("training_source_sha256")
        or card.get("reproducibility", {}).get("source_sha256")
        or metrics.get("source_sha256")
        or ""
    )
    dataset_value = str(metrics.get("dataset_path") or "data/sft/grounded_qa.jsonl")
    dataset_path = Path(dataset_value).expanduser()
    if not dataset_path.exists():
        dataset_path = ROOT / "data" / "sft" / "grounded_qa.jsonl"
    if dataset_path.exists():
        current_dataset_sha = file_sha256(dataset_path)
        current_gradient_sha = gradient_payload_sha256(load_sft_records(dataset_path))
        metrics["current_dataset_sha256"] = current_dataset_sha
        metrics["gradient_payload_sha256"] = current_gradient_sha
        metrics["dataset_artifact_matches_training"] = (
            current_dataset_sha == str(metrics.get("dataset_sha256") or "").lower()
        )
        card.setdefault("data", {})["current_dataset_sha256"] = current_dataset_sha
        card["data"]["gradient_payload_sha256"] = current_gradient_sha
        card["data"]["artifact_matches_training"] = metrics["dataset_artifact_matches_training"]
    metrics["validated_for_web"] = True
    metrics["adapter_admission"] = admission
    metrics["training_source_sha256"] = training_source
    # Keep the original training snapshot for provenance, while binding the
    # current product admission to the current source snapshot.
    metrics["source_sha256"] = current_source
    metrics["admission_source_sha256"] = current_source
    card["source_sha256"] = current_source
    card["training_source_sha256"] = training_source
    card["admission_source_sha256"] = current_source
    card.setdefault("reproducibility", {})["training_source_sha256"] = training_source
    card["reproducibility"]["admission_source_sha256"] = current_source
    card["validated_for_web"] = True
    card["adapter_admission"] = admission
    card.setdefault("artifacts", {})["adapter_sha256"] = adapter_hash
    card["artifacts"]["evaluation_sha256"] = evaluation_hash
    pending_gate = "Evaluate adapter against the frozen cola cases before enabling it as the Web default."
    admitted_limit = "Admission is task-local: it passed the frozen cola evidence/timestamp regression, not a public benchmark."
    limitations = [
        str(item)
        for item in list(card.get("limitations") or [])
        if pending_gate not in str(item)
    ]
    card["limitations"] = list(dict.fromkeys([*limitations, admitted_limit]))
    if str(metrics.get("hyperparameters", {}).get("quantization")) == "none":
        card["training_method"] = (
            "BF16 LoRA supervised fine-tuning with frozen Qwen3.5 base weights; "
            "the 4-bit QLoRA preflight was rejected because the installed bitsandbytes build is CPU-only."
        )
    _atomic_write(metrics_path, metrics)
    _atomic_write(card_path, card)
    _bind_pack_metadata(evaluation, admission, adapter_path)
    print(json.dumps({"ok": True, "metrics": str(metrics_path), "model_card": str(card_path), "admission": admission}, ensure_ascii=False, indent=2))


def _rooted(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _preserve_evaluation(path: Path, digest: str) -> Path:
    """Keep the exact admission input addressable after the latest report changes."""
    target = path.parent / "adapter_admissions" / f"qwen35_adapter_eval_{digest}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha256(target) != digest:
            raise RuntimeError(f"immutable admission artifact hash mismatch: {target}")
        return target.resolve()
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_bytes(path.read_bytes())
    if file_sha256(temp) != digest:
        temp.unlink(missing_ok=True)
        raise RuntimeError("evaluation changed while preserving admission artifact")
    temp.replace(target)
    return target.resolve()


def _bind_pack_metadata(evaluation: dict, admission: dict, adapter_path: Path) -> None:
    """Write the final admission pointer into the evaluated pack.

    ``canonical_pack_sha256`` deliberately excludes ``metadata.llm_adapter``;
    therefore this post-evaluation binding does not invalidate the evaluation
    identity or create a self-referential hash loop.  The raw pack file hash is
    still captured by the artifact manifest after this update.
    """
    pack_value = str(evaluation.get("adapter", {}).get("pack_path") or "").strip()
    if not pack_value:
        return
    pack_path = Path(pack_value).expanduser()
    if not pack_path.exists():
        candidates = (ROOT / "outputs" / "iboy_qwen35" / "cola_review" / "knowledge_pack.json",)
        pack_path = next((candidate for candidate in candidates if candidate.exists()), pack_path)
    if not pack_path.is_file():
        raise FileNotFoundError(f"evaluated canonical pack not found: {pack_value}")
    try:
        payload = json.loads(pack_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load evaluated canonical pack: {pack_path}") from exc
    metadata = dict(payload.get("metadata") or {})
    bound = dict(admission)
    bound.update(
        {
            "enabled": True,
            "path": str(adapter_path.resolve()),
            "validated_for_web": True,
        }
    )
    metadata["llm_adapter"] = bound
    payload["metadata"] = metadata
    temp = pack_path.with_suffix(pack_path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(pack_path)


if __name__ == "__main__":
    main()
