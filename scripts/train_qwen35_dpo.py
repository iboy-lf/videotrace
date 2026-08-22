from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from videomemo.eval.reproducibility import file_sha256, source_fingerprint
from videomemo.training.dpo_objective import dpo_statistics
from videomemo.training.preference_data import (
    PreferenceRecord,
    load_preference_records,
    preference_gradient_payload_sha256,
    validate_preference_records,
)
from videomemo.training.sft_data import load_sft_records


SYSTEM_PROMPT = (
    "你是 VideoTrace 的证据约束视频问答模型。只能根据候选时间证据回答；"
    "每个事实必须绑定候选时间范围。证据不足时明确拒答，不得补写视频中没有的信息。"
)
REFERENCE_SCHEMA_VERSION = "videotrace-qwen35-dpo-reference-logprobs-v1"


@dataclass(frozen=True)
class EncodedAnswer:
    input_ids: list[int]
    labels: list[int]


@dataclass(frozen=True)
class EncodedPreference:
    pair_id: str
    split: str
    negative_type: str
    chosen: EncodedAnswer
    rejected: EncodedAnswer


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-train-qwen35-dpo")
    parser.add_argument("--config", default="configs/qwen35_dpo.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--initial-adapter", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--model-card-path", default=None)
    parser.add_argument("--reference-logprobs-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--skip-frozen-eval",
        action="store_true",
        help="seal the frozen test during hyperparameter selection",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = _load_config(_rooted(args.config))
    overrides = {
        "dataset_path": args.dataset,
        "model_path": args.model,
        "initial_adapter_path": args.initial_adapter,
        "output_dir": args.output_dir,
        "metrics_path": args.metrics_path,
        "model_card_path": args.model_card_path,
        "reference_logprobs_path": args.reference_logprobs_path,
        "device": args.device,
        "resume_from_checkpoint": args.resume_from_checkpoint,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.num_train_epochs is not None:
        config["num_train_epochs"] = args.num_train_epochs
    if args.beta is not None:
        config["beta"] = args.beta
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.seed is not None:
        config["seed"] = args.seed
    if args.skip_frozen_eval:
        config["evaluate_frozen_test"] = False
    config = _resolve_config_paths(config)

    records = load_preference_records(config["dataset_path"])
    source_records = load_sft_records(config["source_sft_path"])
    validation = validate_preference_records(records, project_root=ROOT, source_records=source_records)
    if not validation["valid"]:
        raise SystemExit("DPO preference dataset validation failed: " + "; ".join(validation["errors"]))
    train_records = [record for record in records if record.split == "train"]
    dev_records = [record for record in records if record.split == "dev"]
    frozen_test_records = [record for record in records if record.split == "test"]
    if any(record.frozen_test for record in train_records + dev_records):
        raise SystemExit("frozen test preference pair leaked into optimizer split")
    if not train_records or not dev_records or not frozen_test_records:
        raise SystemExit("DPO requires train, dev and frozen test preference pairs")

    initial_adapter = Path(config["initial_adapter_path"])
    _require_adapter(initial_adapter, "initial admitted SFT adapter")
    output_dir = Path(config["output_dir"])
    metrics_path = Path(config["metrics_path"])
    model_card_path = Path(config["model_card_path"])
    reference_path = Path(config["reference_logprobs_path"])
    resume_path_value = str(config.get("resume_from_checkpoint") or "").strip()
    resume_path = Path(resume_path_value) if resume_path_value else None
    if resume_path is not None:
        _require_adapter(resume_path, "DPO resume checkpoint")
    if output_dir.exists() and (output_dir / "adapter_config.json").exists() and resume_path is None:
        if not args.force:
            raise SystemExit(f"DPO adapter already exists; use --force or --resume-from-checkpoint: {output_dir}")
        _archive_existing(output_dir, metrics_path, model_card_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    model_card_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 43))
    _seed_everything(seed)

    if args.dry_run:
        report = _dry_run(config, records, train_records, dev_records, frozen_test_records, validation)
        _atomic_json(metrics_path, report)
        _atomic_json(model_card_path, _model_card(report))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    report = _train(
        config,
        records,
        train_records,
        dev_records,
        frozen_test_records,
        validation,
        output_dir,
        initial_adapter,
        resume_path,
        reference_path,
    )
    _atomic_json(metrics_path, report)
    _atomic_json(model_card_path, _model_card(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _train(
    config: dict,
    records: list[PreferenceRecord],
    train_records: list[PreferenceRecord],
    dev_records: list[PreferenceRecord],
    frozen_test_records: list[PreferenceRecord],
    validation: dict,
    output_dir: Path,
    initial_adapter: Path,
    resume_path: Path | None,
    reference_path: Path,
) -> dict:
    import torch
    from transformers import AutoProcessor

    model_path = str(config["model_path"])
    device = str(config.get("device", "cuda:0"))
    dtype_name = str(config.get("dtype", "bfloat16"))
    dtype = _torch_dtype(torch, dtype_name)
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("DPO training requested CUDA but torch.cuda.is_available() is false")
    if str(config.get("quantization", "none")) == "4bit":
        raise RuntimeError(
            "4-bit DPO is disabled for this project environment: the installed bitsandbytes build is CPU-only"
        )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    max_length = int(config.get("max_length", 768))
    encoded = _encode_records(tokenizer, records, max_length)
    encoded_by_id = {item.pair_id: item for item in encoded}
    dataset_sha = file_sha256(Path(config["dataset_path"]))
    gradient_sha = preference_gradient_payload_sha256(records)
    initial_adapter_sha = file_sha256(initial_adapter / "adapter_model.safetensors")

    reference_cache = _load_valid_reference_cache(
        reference_path,
        dataset_sha=dataset_sha,
        gradient_sha=gradient_sha,
        initial_adapter_sha=initial_adapter_sha,
        model_path=model_path,
        max_length=max_length,
        pair_ids=[record.pair_id for record in records],
    ) if bool(config.get("reuse_reference_logprobs", True)) else None

    policy_path = resume_path or initial_adapter
    model = None
    if reference_cache is None:
        model = _load_model(config, initial_adapter, trainable=True, torch=torch)
        reference_cache = _precompute_reference_logprobs(
            model,
            encoded,
            torch=torch,
            dtype=dtype,
            dataset_sha=dataset_sha,
            gradient_sha=gradient_sha,
            initial_adapter=initial_adapter,
            initial_adapter_sha=initial_adapter_sha,
            model_path=model_path,
            max_length=max_length,
        )
        _atomic_json(reference_path, reference_cache)
        if policy_path.resolve() != initial_adapter.resolve():
            del model
            model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if model is None:
        model = _load_model(config, policy_path, trainable=True, torch=torch)
    reference_sha = file_sha256(reference_path)
    reference_rows = dict(reference_cache.get("records") or {})
    checkpoint_contract = _training_contract(
        config,
        dataset_sha=dataset_sha,
        gradient_sha=gradient_sha,
        reference_sha=reference_sha,
        initial_adapter_sha=initial_adapter_sha,
    )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("DPO policy exposes no trainable LoRA parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config.get("learning_rate", 5e-5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    resume_state = _load_resume_state(
        resume_path,
        optimizer,
        torch,
        contract=checkpoint_contract,
    )
    optimizer_state_offload = bool(config.get("optimizer_state_offload", True))
    if optimizer_state_offload:
        _move_optimizer_state(optimizer, torch.device("cpu"), torch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    global_step = int(resume_state.get("global_step", 0))
    resumed_from_step = global_step
    start_epoch = int(resume_state.get("next_epoch_index", 0))
    start_pair_index = int(resume_state.get("next_pair_index", 0))
    beta = float(config.get("beta", 0.1))
    grad_accum = max(1, int(config.get("gradient_accumulation_steps", 4)))
    epochs = max(1, int(config.get("num_train_epochs", 1)))
    max_steps = int(config.get("max_steps", 0) or 0)
    save_every = max(0, int(config.get("save_every_steps", 1)))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    history: list[dict] = list(resume_state.get("history") or [])
    trained_pair_ids: list[str] = list(resume_state.get("trained_pair_ids") or [])
    trained_tokens = int(resume_state.get("trained_tokens", 0))
    steps_this_run = 0
    tokens_this_run = 0
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # DPO needs deterministic policy/reference scores. Eval mode disables
    # adapter dropout while gradients on LoRA weights remain enabled.
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    final_next_epoch = start_epoch
    final_next_pair = start_pair_index

    for epoch_index in range(start_epoch, epochs):
        ordered = [encoded_by_id[record.pair_id] for record in train_records]
        random.Random(int(config.get("seed", 43)) + epoch_index).shuffle(ordered)
        offset = start_pair_index if epoch_index == start_epoch else 0
        while offset < len(ordered):
            if max_steps > 0 and global_step >= max_steps:
                break
            group = ordered[offset : offset + grad_accum]
            optimizer.zero_grad(set_to_none=True)
            group_stats: list[dict] = []
            for example in group:
                reference = dict(reference_rows[example.pair_id])
                with torch.no_grad():
                    policy_chosen = _sequence_logp(model, example.chosen, torch, dtype)
                    policy_rejected = _sequence_logp(model, example.rejected, torch, dtype)
                stats = dpo_statistics(
                    float(policy_chosen.detach().cpu()),
                    float(policy_rejected.detach().cpu()),
                    float(reference["chosen_logp"]),
                    float(reference["rejected_logp"]),
                    beta=beta,
                )
                # The no-grad scoring pass still creates large temporary CUDA
                # blocks.  A resumed optimizer has a different allocator
                # history from a fresh run, so release those blocks before
                # constructing the gradient-bearing forwards.
                del policy_chosen, policy_rejected
                if optimizer_state_offload and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                scale = 1.0 / len(group)
                chosen_logp = _sequence_logp(model, example.chosen, torch, dtype)
                (chosen_logp * float(stats["chosen_logp_gradient"]) * scale).backward()
                del chosen_logp
                rejected_logp = _sequence_logp(model, example.rejected, torch, dtype)
                (rejected_logp * float(stats["rejected_logp_gradient"]) * scale).backward()
                del rejected_logp
                if optimizer_state_offload and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                group_stats.append({"pair_id": example.pair_id, "negative_type": example.negative_type, **stats})
                trained_pair_ids.append(example.pair_id)
                pair_tokens = sum(label != -100 for label in example.chosen.labels)
                pair_tokens += sum(label != -100 for label in example.rejected.labels)
                trained_tokens += pair_tokens
                tokens_this_run += pair_tokens

            gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm).detach().cpu())
            if optimizer_state_offload:
                _move_optimizer_state_to_parameters(optimizer, torch)
            optimizer.step()
            if optimizer_state_offload:
                _move_optimizer_state(optimizer, torch.device("cpu"), torch)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            global_step += 1
            steps_this_run += 1
            next_offset = offset + len(group)
            next_epoch = epoch_index
            next_pair = next_offset
            if next_offset >= len(ordered):
                next_epoch = epoch_index + 1
                next_pair = 0
            final_next_epoch = next_epoch
            final_next_pair = next_pair
            history.append(
                {
                    "step": global_step,
                    "epoch": epoch_index + 1,
                    "pair_ids": [item["pair_id"] for item in group_stats],
                    "negative_types": [item["negative_type"] for item in group_stats],
                    "loss": round(_mean(item["loss"] for item in group_stats), 8),
                    "reward_margin": round(_mean(item["reward_margin"] for item in group_stats), 8),
                    "chosen_reward": round(_mean(item["chosen_reward"] for item in group_stats), 8),
                    "rejected_reward": round(_mean(item["rejected_reward"] for item in group_stats), 8),
                    "policy_preference_accuracy": round(
                        _mean(float(item["policy_preference_correct"]) for item in group_stats), 6
                    ),
                    "gradient_norm": round(gradient_norm, 6),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if save_every and global_step % save_every == 0:
                _save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    output_dir,
                    global_step=global_step,
                    next_epoch_index=next_epoch,
                    next_pair_index=next_pair,
                    history=history,
                    trained_pair_ids=trained_pair_ids,
                    trained_tokens=trained_tokens,
                    contract=checkpoint_contract,
                    torch=torch,
                )
            offset = next_offset
        if max_steps > 0 and global_step >= max_steps:
            break
        start_pair_index = 0

    elapsed = max(1e-6, time.perf_counter() - started)
    evaluations = {
        "train": _evaluate_preferences(model, [encoded_by_id[item.pair_id] for item in train_records], reference_rows, beta, torch, dtype),
        "dev": _evaluate_preferences(model, [encoded_by_id[item.pair_id] for item in dev_records], reference_rows, beta, torch, dtype),
    }
    frozen_evaluated = bool(config.get("evaluate_frozen_test", True))
    if frozen_evaluated:
        evaluations["frozen_test"] = _evaluate_preferences(
            model,
            [encoded_by_id[item.pair_id] for item in frozen_test_records],
            reference_rows,
            beta,
            torch,
            dtype,
        )
    else:
        evaluations["frozen_test"] = {
            "skipped": True,
            "reason": "sealed during hyperparameter selection; evaluate only after selecting on dev",
            "num_pairs": len(frozen_test_records),
        }
    _save_checkpoint(
        model,
        tokenizer,
        optimizer,
        output_dir,
        global_step=global_step,
        next_epoch_index=final_next_epoch,
        next_pair_index=final_next_pair,
        history=history,
        trained_pair_ids=trained_pair_ids,
        trained_tokens=trained_tokens,
        contract=checkpoint_contract,
        torch=torch,
    )
    peak_memory = round(torch.cuda.max_memory_allocated() / (1024**2), 2) if torch.cuda.is_available() else 0.0
    parameter_devices = sorted({str(parameter.device) for parameter in trainable_parameters})
    return {
        "schema_version": "videotrace-qwen35-dpo-metrics-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "training_method": "single-policy LoRA DPO with frozen precomputed SFT reference log-probabilities",
        "model_path": model_path,
        "adapter_path": str(output_dir.resolve()),
        "initial_adapter_path": str(initial_adapter.resolve()),
        "initial_adapter_sha256": initial_adapter_sha,
        "dataset_path": str(Path(config["dataset_path"]).resolve()),
        "dataset_sha256": dataset_sha,
        "gradient_payload_sha256": gradient_sha,
        "source_sha256": source_fingerprint(ROOT),
        "training_source_sha256": source_fingerprint(ROOT),
        "split_validation": validation,
        "counts": {
            "train": len(train_records),
            "dev": len(dev_records),
            "frozen_test": len(frozen_test_records),
        },
        "negative_type_counts": validation.get("negative_type_counts", {}),
        "reference_logprobs": {
            "path": str(reference_path.resolve()),
            "sha256": reference_sha,
            "schema_version": reference_cache.get("schema_version"),
            "num_pairs": len(reference_rows),
            "frozen_before_optimizer": True,
        },
        "hyperparameters": {
            key: value
            for key, value in config.items()
            if key not in {"model_path", "dataset_path", "initial_adapter_path", "resume_from_checkpoint"}
        },
        "history": history,
        "steps": global_step,
        "steps_this_run": steps_this_run,
        "resumed_from_checkpoint": str(resume_path.resolve()) if resume_path else "",
        "resumed_from_step": resumed_from_step,
        "checkpoint_contract_sha256": checkpoint_contract["contract_sha256"],
        "checkpoint_manifest_sha256": file_sha256(output_dir / "checkpoint_manifest.json"),
        "checkpoint_files": sorted(
            path.name for path in output_dir.iterdir() if path.is_file() and path.name != "checkpoint_manifest.json"
        ),
        "train_loss_last": history[-1]["loss"] if history else None,
        "trained_pair_ids": trained_pair_ids,
        "train_tokens": trained_tokens,
        "train_tokens_this_run": tokens_this_run,
        "tokens_per_second": round(tokens_this_run / elapsed, 3),
        "elapsed_seconds": round(elapsed, 3),
        "evaluations": evaluations,
        "frozen_test_evaluated": frozen_evaluated,
        "peak_cuda_memory_mib": peak_memory,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "physical_gpu_ids": os.environ.get("VIDEOTRACE_PHYSICAL_GPUS", ""),
        "resume_supported": True,
        "optimizer_state_offload": optimizer_state_offload,
        "model_parallel": {
            "requested": bool(config.get("model_parallel", False)),
            "active": len(parameter_devices) > 1,
            "trainable_parameter_devices": parameter_devices,
            "max_memory_gib_per_visible_gpu": int(config.get("model_parallel_max_memory_gib", 22)),
        },
        "failure_recovery": (
            "checkpoint_manifest.json commits adapter/tokenizer/optimizer/RNG/trainer-state hashes at optimizer-step boundaries; "
            "resume validates reference, dataset, SFT adapter, source and optimizer-affecting config"
        ),
        "frozen_test_policy": "cola preference pairs are evaluation-only and never passed to optimizer/backward",
    }


def _dry_run(config, records, train_records, dev_records, frozen_test_records, validation):
    initial_adapter = Path(config["initial_adapter_path"])
    return {
        "schema_version": "videotrace-qwen35-dpo-metrics-v1",
        "status": "dry_run",
        "training_method": "single-policy LoRA DPO with frozen precomputed SFT reference log-probabilities",
        "model_path": str(config["model_path"]),
        "initial_adapter_path": str(initial_adapter.resolve()),
        "initial_adapter_sha256": file_sha256(initial_adapter / "adapter_model.safetensors"),
        "dataset_path": str(Path(config["dataset_path"]).resolve()),
        "dataset_sha256": file_sha256(Path(config["dataset_path"])),
        "gradient_payload_sha256": preference_gradient_payload_sha256(records),
        "source_sha256": source_fingerprint(ROOT),
        "training_source_sha256": source_fingerprint(ROOT),
        "split_validation": validation,
        "counts": {
            "train": len(train_records),
            "dev": len(dev_records),
            "frozen_test": len(frozen_test_records),
        },
        "negative_type_counts": validation.get("negative_type_counts", {}),
        "gradient_update_pairs": [record.pair_id for record in train_records],
        "frozen_test_pairs": [record.pair_id for record in frozen_test_records],
        "reference_contract": (
            "reference log-probabilities are computed from the admitted SFT adapter before the first optimizer step "
            "and hash-bound to dataset, tokenizer length, base model and SFT adapter weights"
        ),
    }


def _model_card(metrics: dict) -> dict:
    evaluations = dict(metrics.get("evaluations") or {})
    return {
        "schema_version": "videotrace-qwen35-dpo-model-card-v1",
        "model_name": "VideoTrace Qwen3.5 evidence-preference LoRA adapter",
        "base_model": metrics.get("model_path"),
        "initial_adapter": metrics.get("initial_adapter_path"),
        "adapter_path": metrics.get("adapter_path", "outputs/models/qwen35_dpo_adapter"),
        "intended_use": (
            "Prefer timestamp-correct, evidence-supported Chinese video answers and explicit abstention over "
            "wrong timestamps, missing timestamps, hallucinated details and unsupported overclaims."
        ),
        "training_method": metrics.get("training_method"),
        "reference_policy": {
            "kind": "frozen admitted SFT adapter",
            "adapter_sha256": metrics.get("initial_adapter_sha256"),
            "logprob_artifact": metrics.get("reference_logprobs", {}),
            "memory_design": "precomputed reference log-probs avoid co-resident policy/reference 9B models",
        },
        "data": {
            "dataset": metrics.get("dataset_path"),
            "dataset_sha256": metrics.get("dataset_sha256"),
            "gradient_payload_sha256": metrics.get("gradient_payload_sha256"),
            "counts": metrics.get("counts", {}),
            "negative_type_counts": metrics.get("negative_type_counts", {}),
            "train_dev_test_isolation": metrics.get("split_validation", {}).get("valid", False),
            "frozen_test_excluded_from_gradients": True,
        },
        "task_local_evaluation": evaluations,
        "checkpoint_recovery": {
            "resume_supported": metrics.get("resume_supported", False),
            "contract_sha256": metrics.get("checkpoint_contract_sha256", ""),
            "manifest_sha256": metrics.get("checkpoint_manifest_sha256", ""),
            "files": metrics.get("checkpoint_files", []),
            "resumed_from_step": metrics.get("resumed_from_step", 0),
            "policy": (
                "The checkpoint commit manifest binds adapter/tokenizer/optimizer/RNG/trainer state and "
                "the frozen reference, data, SFT initialization, source and optimizer contract."
            ),
        },
        "limitations": [
            "Small manually authored task-local preference set; no public benchmark or broad alignment claim.",
            "One short LoRA update demonstrates a real, recoverable DPO path but is not large-scale preference training.",
            "Visual recognition remains provided by the Qwen3.5/SigLIP2 inference stack; DPO targets answer behavior.",
            "Product use still requires frozen-pack comparison and hash-bound best-adapter admission.",
            "The installed bitsandbytes build is CPU-only, so the verified run uses BF16 LoRA rather than claiming QLoRA.",
            (
                "Frozen-test evaluation was sealed during hyperparameter selection."
                if metrics.get("frozen_test_evaluated") is False
                else "Frozen-test results were read only for the final selected run."
            ),
        ],
        "metrics_path": "outputs/models/qwen35_dpo_metrics.json",
        "reproducibility": {
            "source_sha256": metrics.get("source_sha256"),
            "training_source_sha256": metrics.get("training_source_sha256", metrics.get("source_sha256")),
            "cuda_visible_devices": metrics.get("cuda_visible_devices", ""),
            "physical_gpu_ids": metrics.get("physical_gpu_ids", ""),
            "resume_supported": metrics.get("resume_supported", False),
        },
    }


def _encode_records(tokenizer, records: list[PreferenceRecord], max_length: int) -> list[EncodedPreference]:
    encoded: list[EncodedPreference] = []
    for record in records:
        prompt_text = tokenizer.apply_chat_template(
            _messages(record, None), tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        if len(prompt_ids) >= max_length - 4:
            raise ValueError(f"DPO prompt exceeds max_length for {record.pair_id}")
        chosen = _encode_answer(tokenizer, record, record.chosen, prompt_ids, max_length)
        rejected = _encode_answer(tokenizer, record, record.rejected, prompt_ids, max_length)
        encoded.append(
            EncodedPreference(
                pair_id=record.pair_id,
                split=record.split,
                negative_type=record.negative_type,
                chosen=chosen,
                rejected=rejected,
            )
        )
    return encoded


def _encode_answer(tokenizer, record, answer: str, prompt_ids: list[int], max_length: int) -> EncodedAnswer:
    full_text = tokenizer.apply_chat_template(
        _messages(record, answer), tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length).input_ids
    if len(prompt_ids) >= len(full_ids):
        raise ValueError(f"DPO response was truncated or missing for {record.pair_id}")
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids) :]
    if not any(label != -100 for label in labels):
        raise ValueError(f"DPO response has no trainable tokens for {record.pair_id}")
    return EncodedAnswer(input_ids=full_ids, labels=labels)


def _messages(record: PreferenceRecord, answer: str | None) -> list[dict]:
    evidence_lines = [
        f"[{float(item['start_sec']):.1f}-{float(item['end_sec']):.1f}] {item.get('text', '')}"
        for item in record.evidence
    ]
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "（没有足够的候选证据）"
    user = f"问题：{record.query}\n候选时间证据：\n{evidence_text}\n请输出带时间戳的中文回答。"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def _load_model(config: dict, adapter_path: Path, *, trainable: bool, torch):
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    dtype = _torch_dtype(torch, str(config.get("dtype", "bfloat16")))
    device = str(config.get("device", "cuda:0"))
    device_map, max_memory = _model_device_map(config, torch)
    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "dtype": dtype,
        "device_map": device_map,
    }
    if max_memory:
        load_kwargs["max_memory"] = max_memory
    base = AutoModelForImageTextToText.from_pretrained(str(config["model_path"]), **load_kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=trainable)
    if bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def _precompute_reference_logprobs(
    model,
    encoded: list[EncodedPreference],
    *,
    torch,
    dtype,
    dataset_sha: str,
    gradient_sha: str,
    initial_adapter: Path,
    initial_adapter_sha: str,
    model_path: str,
    max_length: int,
) -> dict:
    model.eval()
    rows = {}
    with torch.no_grad():
        for example in encoded:
            chosen = _sequence_logp(model, example.chosen, torch, dtype)
            rejected = _sequence_logp(model, example.rejected, torch, dtype)
            rows[example.pair_id] = {
                "split": example.split,
                "negative_type": example.negative_type,
                "chosen_logp": float(chosen.detach().cpu()),
                "rejected_logp": float(rejected.detach().cpu()),
                "chosen_tokens": sum(label != -100 for label in example.chosen.labels),
                "rejected_tokens": sum(label != -100 for label in example.rejected.labels),
            }
    model.eval()
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": model_path,
        "initial_adapter_path": str(initial_adapter.resolve()),
        "initial_adapter_sha256": initial_adapter_sha,
        "dataset_sha256": dataset_sha,
        "gradient_payload_sha256": gradient_sha,
        "max_length": max_length,
        "records": rows,
    }


def _load_valid_reference_cache(
    path: Path,
    *,
    dataset_sha: str,
    gradient_sha: str,
    initial_adapter_sha: str,
    model_path: str,
    max_length: int,
    pair_ids: list[str],
) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    expected = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "dataset_sha256": dataset_sha,
        "gradient_payload_sha256": gradient_sha,
        "initial_adapter_sha256": initial_adapter_sha,
        "model_path": model_path,
        "max_length": max_length,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    rows = dict(payload.get("records") or {})
    if sorted(rows) != sorted(pair_ids):
        return None
    if any("chosen_logp" not in row or "rejected_logp" not in row for row in rows.values()):
        return None
    return payload


def _sequence_logp(model, answer: EncodedAnswer, torch, dtype):
    input_ids = torch.tensor([answer.input_ids], dtype=torch.long, device=model.device)
    labels = torch.tensor([answer.labels], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids)
    use_amp = torch.cuda.is_available() and dtype in {torch.bfloat16, torch.float16}
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        mask = shifted_labels.ne(-100)
        safe_labels = shifted_labels.masked_fill(~mask, 0)
        selected = shifted_logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logps = selected - shifted_logits.logsumexp(dim=-1)
        return (token_logps * mask).sum(dim=-1).squeeze(0)


def _evaluate_preferences(model, examples, reference_rows, beta, torch, dtype) -> dict:
    model.eval()
    rows = []
    with torch.no_grad():
        for example in examples:
            chosen = float(_sequence_logp(model, example.chosen, torch, dtype).detach().cpu())
            rejected = float(_sequence_logp(model, example.rejected, torch, dtype).detach().cpu())
            reference = reference_rows[example.pair_id]
            stats = dpo_statistics(
                chosen,
                rejected,
                float(reference["chosen_logp"]),
                float(reference["rejected_logp"]),
                beta=beta,
            )
            chosen_tokens = sum(label != -100 for label in example.chosen.labels)
            rejected_tokens = sum(label != -100 for label in example.rejected.labels)
            policy_chosen_per_token = chosen / max(1, chosen_tokens)
            policy_rejected_per_token = rejected / max(1, rejected_tokens)
            reference_chosen_per_token = float(reference["chosen_logp"]) / max(1, chosen_tokens)
            reference_rejected_per_token = float(reference["rejected_logp"]) / max(1, rejected_tokens)
            rows.append(
                {
                    "pair_id": example.pair_id,
                    "negative_type": example.negative_type,
                    "chosen_tokens": chosen_tokens,
                    "rejected_tokens": rejected_tokens,
                    "policy_preference_correct_per_token": policy_chosen_per_token
                    > policy_rejected_per_token,
                    "reference_preference_correct_per_token": reference_chosen_per_token
                    > reference_rejected_per_token,
                    **stats,
                }
            )
    model.eval()
    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_type.setdefault(str(row["negative_type"]), []).append(row)
    return {
        "num_pairs": len(rows),
        "mean_loss": round(_mean(item["loss"] for item in rows), 8),
        "mean_reward_margin": round(_mean(item["reward_margin"] for item in rows), 8),
        "mean_chosen_reward": round(_mean(item["chosen_reward"] for item in rows), 8),
        "mean_rejected_reward": round(_mean(item["rejected_reward"] for item in rows), 8),
        "mean_abs_implicit_reward": round(
            _mean(
                (abs(item["chosen_reward"]) + abs(item["rejected_reward"])) / 2.0
                for item in rows
            ),
            8,
        ),
        "reward_preference_accuracy": round(
            _mean(float(item["reward_preference_correct"]) for item in rows), 6
        ),
        "policy_preference_accuracy": round(
            _mean(float(item["policy_preference_correct"]) for item in rows), 6
        ),
        "reference_preference_accuracy": round(
            _mean(float(item["reference_preference_correct"]) for item in rows), 6
        ),
        "policy_preference_accuracy_per_token": round(
            _mean(float(item["policy_preference_correct_per_token"]) for item in rows), 6
        ),
        "reference_preference_accuracy_per_token": round(
            _mean(float(item["reference_preference_correct_per_token"]) for item in rows), 6
        ),
        "policy_flip_count_vs_reference": sum(
            bool(item["policy_preference_correct"])
            != bool(item["reference_preference_correct"])
            for item in rows
        ),
        "by_negative_type": {
            negative_type: {
                "num_pairs": len(items),
                "mean_reward_margin": round(_mean(item["reward_margin"] for item in items), 8),
                "policy_preference_accuracy": round(
                    _mean(float(item["policy_preference_correct"]) for item in items), 6
                ),
            }
            for negative_type, items in sorted(by_type.items())
        },
        "pairs": [
            {
                "pair_id": item["pair_id"],
                "negative_type": item["negative_type"],
                "loss": round(float(item["loss"]), 8),
                "reward_margin": round(float(item["reward_margin"]), 8),
                "policy_preference_correct": bool(item["policy_preference_correct"]),
                "reference_preference_correct": bool(item["reference_preference_correct"]),
                "policy_preference_correct_per_token": bool(
                    item["policy_preference_correct_per_token"]
                ),
                "reference_preference_correct_per_token": bool(
                    item["reference_preference_correct_per_token"]
                ),
                "chosen_tokens": int(item["chosen_tokens"]),
                "rejected_tokens": int(item["rejected_tokens"]),
            }
            for item in rows
        ],
    }


def _save_checkpoint(
    model,
    tokenizer,
    optimizer,
    output_dir: Path,
    *,
    global_step: int,
    next_epoch_index: int,
    next_pair_index: int,
    history: list[dict],
    trained_pair_ids: list[str],
    trained_tokens: int,
    contract: dict,
    torch,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _atomic_torch_save(optimizer.state_dict(), output_dir / "optimizer.pt", torch)
    _atomic_torch_save(_rng_state(torch), output_dir / "rng_state.pt", torch)
    _atomic_json(
        output_dir / "trainer_state.json",
        {
            "schema_version": "videotrace-dpo-trainer-state-v2",
            "global_step": global_step,
            "next_epoch_index": next_epoch_index,
            "next_pair_index": next_pair_index,
            "history": list(history),
            "trained_pair_ids": list(trained_pair_ids),
            "trained_tokens": int(trained_tokens),
            "contract": contract,
        },
    )
    files = {
        path.name: file_sha256(path)
        for path in sorted(output_dir.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "checkpoint_manifest.json"
    }
    _atomic_json(
        output_dir / "checkpoint_manifest.json",
        {
            "schema_version": "videotrace-dpo-checkpoint-manifest-v1",
            "global_step": int(global_step),
            "contract_sha256": contract["contract_sha256"],
            "files": files,
        },
    )


def _load_resume_state(
    resume_path: Path | None,
    optimizer,
    torch,
    *,
    contract: dict,
) -> dict:
    if resume_path is None:
        return {}
    _validate_resume_checkpoint(resume_path, contract)
    state = json.loads((resume_path / "trainer_state.json").read_text(encoding="utf-8"))
    optimizer.load_state_dict(
        torch.load(resume_path / "optimizer.pt", map_location="cpu", weights_only=False)
    )
    rng = torch.load(resume_path / "rng_state.pt", map_location="cpu", weights_only=False)
    if "python" in rng:
        random.setstate(rng["python"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda"):
        torch.cuda.set_rng_state_all(rng["cuda"])
    return state


def _validate_resume_checkpoint(resume_path: Path, contract: dict) -> None:
    resume_path = Path(resume_path).resolve()
    required = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng_state.pt",
        "trainer_state.json",
        "checkpoint_manifest.json",
    ]
    missing = [name for name in required if not (resume_path / name).is_file()]
    if missing:
        raise RuntimeError(
            f"DPO resume checkpoint is incomplete: {resume_path}; missing {', '.join(missing)}"
        )
    manifest = json.loads((resume_path / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    state = json.loads((resume_path / "trainer_state.json").read_text(encoding="utf-8"))
    if manifest.get("contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError(
            "DPO resume checkpoint provenance mismatch: "
            f"expected {contract.get('contract_sha256')}, checkpoint {manifest.get('contract_sha256')}"
        )
    if state.get("contract", {}).get("contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("DPO resume trainer state contract does not match current training inputs")
    mismatches = []
    for name, expected in dict(manifest.get("files") or {}).items():
        path = resume_path / name
        current = file_sha256(path) if path.is_file() else ""
        if current != expected:
            mismatches.append(name)
    if mismatches:
        raise RuntimeError(
            "DPO resume checkpoint hash mismatch; refusing partial/corrupt state: "
            + ", ".join(sorted(mismatches))
        )


def _training_contract(
    config: dict,
    *,
    dataset_sha: str,
    gradient_sha: str,
    reference_sha: str,
    initial_adapter_sha: str,
) -> dict:
    payload = {
        "schema_version": "videotrace-dpo-resume-contract-v1",
        "dataset_sha256": dataset_sha,
        "gradient_payload_sha256": gradient_sha,
        "reference_logprobs_sha256": reference_sha,
        "initial_adapter_sha256": initial_adapter_sha,
        "source_sha256": source_fingerprint(ROOT),
        "model_path": str(config["model_path"]).replace("\\", "/"),
        "seed": int(config.get("seed", 43)),
        "dtype": str(config.get("dtype", "bfloat16")),
        "quantization": str(config.get("quantization", "none")),
        "max_length": int(config.get("max_length", 768)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 4)),
        "learning_rate": float(config.get("learning_rate", 5e-5)),
        "weight_decay": float(config.get("weight_decay", 0.01)),
        "beta": float(config.get("beta", 0.1)),
        "max_grad_norm": float(config.get("max_grad_norm", 1.0)),
        "optimizer_state_offload": bool(config.get("optimizer_state_offload", True)),
        "model_parallel": bool(config.get("model_parallel", False)),
        "model_parallel_max_memory_gib": int(config.get("model_parallel_max_memory_gib", 22)),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "contract_sha256": hashlib.sha256(encoded).hexdigest()}


def _rng_state(torch) -> dict:
    payload = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _move_optimizer_state(optimizer, device, torch) -> None:
    """Move Adam state without changing values or optimizer parameter bindings.

    Qwen3.5-9B nearly fills a 24 GiB card during long-sequence DPO forwards.
    A resumed AdamW eagerly restores its two FP32 moment tensors on the GPU,
    unlike a fresh run where they are only materialized at ``step()``. Keeping
    those small LoRA-only states on CPU between steps makes fresh and resumed
    runs obey the same forward-memory envelope.
    """

    def move(value):
        if torch.is_tensor(value):
            return value.to(device=device, non_blocking=False)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    for parameter, state in list(optimizer.state.items()):
        optimizer.state[parameter] = move(state)


def _move_optimizer_state_to_parameters(optimizer, torch) -> None:
    for parameter, state in list(optimizer.state.items()):
        def move(value):
            if torch.is_tensor(value):
                return value.to(device=parameter.device, non_blocking=False)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, list):
                return [move(item) for item in value]
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            return value

        optimizer.state[parameter] = move(state)


def _model_device_map(config: dict, torch):
    if not torch.cuda.is_available():
        return {"": "cpu"}, None
    if bool(config.get("model_parallel", False)) and torch.cuda.device_count() >= 2:
        limit = max(1, int(config.get("model_parallel_max_memory_gib", 22)))
        return "balanced", {index: f"{limit}GiB" for index in range(torch.cuda.device_count())}
    return {"": str(config.get("device", "cuda:0"))}, None


def _atomic_torch_save(value, path: Path, torch) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _require_adapter(path: Path, label: str) -> None:
    missing = [name for name in ("adapter_config.json", "adapter_model.safetensors") if not (path / name).is_file()]
    if missing:
        raise SystemExit(f"{label} is incomplete at {path}: missing {', '.join(missing)}")


def _archive_existing(output_dir: Path, metrics_path: Path, model_card_path: Path) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = output_dir.parent / "history" / f"qwen35_dpo_{timestamp}"
    archive.mkdir(parents=True, exist_ok=False)
    if output_dir.exists():
        shutil.move(str(output_dir), str(archive / output_dir.name))
    for path in (metrics_path, model_card_path):
        if path.exists():
            shutil.move(str(path), str(archive / path.name))


def _torch_dtype(torch, value: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(value, torch.bfloat16)


def _load_config(path: Path) -> dict:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _resolve_config_paths(config: dict) -> dict:
    result = dict(config)
    for key in (
        "dataset_path",
        "source_sft_path",
        "initial_adapter_path",
        "output_dir",
        "metrics_path",
        "model_card_path",
        "reference_logprobs_path",
        "resume_from_checkpoint",
    ):
        value = str(result.get(key, "") or "")
        if value and not Path(value).is_absolute():
            result[key] = str((ROOT / value).resolve())
    return result


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _mean(values) -> float:
    rows = list(values)
    return sum(rows) / max(1, len(rows))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
