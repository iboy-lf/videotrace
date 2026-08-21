from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
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
from videomemo.training.sft_data import gradient_payload_sha256, load_sft_records, validate_sft_records


SYSTEM_PROMPT = (
    "你是 VideoTrace 的证据约束视频问答模型。只能根据候选时间证据回答；"
    "每个事实必须绑定候选时间范围。证据不足时明确拒答，不得补写视频中没有的信息。"
)


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-train-qwen35-sft")
    parser.add_argument("--config", default="configs/qwen35_sft.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--model-card-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quantization", choices=["none", "4bit"], default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = _load_config(ROOT / args.config)
    if args.dataset:
        config["dataset_path"] = args.dataset
    if args.model:
        config["model_path"] = args.model
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.metrics_path:
        config["metrics_path"] = args.metrics_path
    if args.model_card_path:
        config["model_card_path"] = args.model_card_path
    if args.device:
        config["device"] = args.device
    if args.quantization:
        config["quantization"] = args.quantization
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.num_train_epochs is not None:
        config["num_train_epochs"] = args.num_train_epochs
    if args.resume_from_checkpoint is not None:
        config["resume_from_checkpoint"] = args.resume_from_checkpoint
    config = _resolve_config_paths(config)
    records = load_sft_records(config["dataset_path"])
    validation = validate_sft_records(records, project_root=ROOT)
    if not validation["valid"]:
        raise SystemExit("SFT dataset validation failed: " + "; ".join(validation["errors"]))
    train_records = [record for record in records if record.split == "train"]
    dev_records = [record for record in records if record.split == "dev"]
    frozen_test_records = [record for record in records if record.split == "test"]
    if any(record.frozen_test for record in train_records + dev_records):
        raise SystemExit("frozen test record leaked into gradient-update split")
    if not train_records or not dev_records or not frozen_test_records:
        raise SystemExit("SFT requires train, dev, and frozen test records")

    resume_value = str(config.get("resume_from_checkpoint") or "").strip()
    resume_path = Path(resume_value) if resume_value else None
    contract = _training_contract(config, records)
    if resume_path is not None:
        _validate_resume_checkpoint(resume_path, contract)

    output_dir = Path(config["output_dir"])
    metrics_path = Path(config["metrics_path"])
    model_card_path = Path(config["model_card_path"])
    if (
        output_dir.exists()
        and (output_dir / "adapter_config.json").exists()
        and not args.force
        and resume_path is None
    ):
        raise SystemExit(f"adapter already exists; use --force or resume_from_checkpoint: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    model_card_path.parent.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 42))
    _seed_everything(seed)
    if args.dry_run:
        report = _dry_run(config, records, train_records, dev_records, frozen_test_records, validation)
        metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        model_card_path.write_text(json.dumps(_model_card(report), ensure_ascii=False, indent=2), encoding="utf-8")
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
        contract,
        resume_path,
    )
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    model_card_path.write_text(json.dumps(_model_card(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _train(
    config,
    records,
    train_records,
    dev_records,
    frozen_test_records,
    validation,
    output_dir,
    contract,
    resume_path,
):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    model_path = str(config["model_path"])
    device = str(config.get("device", "cuda:0"))
    dtype_name = str(config.get("dtype", "bfloat16"))
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(dtype_name, torch.bfloat16)
    quantization = str(config.get("quantization", "none"))
    if quantization == "4bit" and not torch.cuda.is_available():
        raise RuntimeError("4bit QLoRA requires CUDA; use quantization=none for CPU dry runs")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if quantization == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = {"": device}
    else:
        load_kwargs["dtype"] = dtype
        load_kwargs["device_map"] = {"": device} if torch.cuda.is_available() else {"": "cpu"}
    model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
    if quantization == "4bit":
        model = prepare_model_for_kbit_training(model)
    target_modules = list(config.get("lora_target_modules", ["q_proj", "v_proj"]))
    available_targets = _available_target_modules(model, target_modules)
    if not available_targets:
        raise RuntimeError(f"none of the requested LoRA target modules exist: {target_modules}")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=available_targets,
        bias="none",
    )
    if resume_path is not None:
        model = PeftModel.from_pretrained(model, str(resume_path), is_trainable=True)
    else:
        model = get_peft_model(model, peft_config)
    if bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.train()
    model.print_trainable_parameters()

    max_length = int(config.get("max_length", 768))
    train_encoded = _encode_records(tokenizer, train_records, max_length)
    dev_encoded = _encode_records(tokenizer, dev_records, max_length)
    batch_size = max(1, int(config.get("per_device_batch_size", 1)))
    grad_accum = max(1, int(config.get("gradient_accumulation_steps", 1)))
    epochs = max(1, int(config.get("num_train_epochs", 1)))
    max_steps = int(config.get("max_steps", 0) or 0)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.get("learning_rate", 2e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    total_updates = max(1, math.ceil(len(train_encoded) / batch_size / grad_accum) * epochs)
    if max_steps > 0:
        total_updates = min(total_updates, max_steps)
    warmup_updates = int(total_updates * float(config.get("warmup_ratio", 0.05)))
    history: list[dict] = []
    update_step = 0
    start_epoch = 0
    start_offset = 0
    resumed_from_step = 0
    if resume_path is not None:
        resume_state = _load_resume_state(
            resume_path,
            optimizer,
            torch,
            contract=contract,
        )
        history = list(resume_state.get("history") or [])
        update_step = int(resume_state.get("global_step", 0))
        resumed_from_step = update_step
        start_epoch = int(resume_state.get("next_epoch_index", 0))
        start_offset = int(resume_state.get("next_batch_offset", 0))
    next_epoch_index = start_epoch
    next_batch_offset = start_offset
    updates_this_run = 0
    train_tokens_this_run = 0
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)

    accumulation_count = 0
    for epoch in range(start_epoch, epochs):
        epoch_examples = _ordered_examples(train_encoded, int(config.get("seed", 42)), epoch)
        offset_start = start_offset if epoch == start_epoch else 0
        for offset in range(offset_start, len(epoch_examples), batch_size):
            if max_steps > 0 and update_step >= max_steps:
                break
            batch_examples = epoch_examples[offset : offset + batch_size]
            train_tokens_this_run += sum(
                sum(label != -100 for label in example.labels) for example in batch_examples
            )
            batch = _collate(batch_examples, tokenizer, torch)
            batch = {key: value.to(model.device) for key, value in batch.items()}
            use_amp = torch.cuda.is_available() and dtype in {torch.bfloat16, torch.float16}
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                loss = model(**batch).loss / grad_accum
            loss.backward()
            accumulation_count += 1
            is_epoch_end = offset + batch_size >= len(epoch_examples)
            if accumulation_count < grad_accum and not is_epoch_end:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_count = 0
            update_step += 1
            updates_this_run += 1
            next_epoch_index, next_batch_offset = _next_position(
                epoch, offset, len(epoch_examples), batch_size
            )
            current_lr = float(optimizer.param_groups[0]["lr"])
            history.append({"step": update_step, "epoch": epoch + 1, "loss": float(loss.detach().cpu()) * grad_accum, "lr": current_lr})
            if int(config.get("save_every_steps", 20)) > 0 and update_step % int(config.get("save_every_steps", 20)) == 0:
                _save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    output_dir,
                    step=update_step,
                    next_epoch_index=next_epoch_index,
                    next_batch_offset=next_batch_offset,
                    history=history,
                    contract=contract,
                    torch=torch,
                )
            if max_steps > 0 and update_step >= max_steps:
                break
        if max_steps > 0 and update_step >= max_steps:
            break

    dev_loss = _evaluate(model, dev_encoded, tokenizer, batch_size, torch, dtype)
    _save_checkpoint(
        model,
        tokenizer,
        optimizer,
        output_dir,
        step=update_step,
        next_epoch_index=next_epoch_index,
        next_batch_offset=next_batch_offset,
        history=history,
        contract=contract,
        torch=torch,
    )
    elapsed = max(1e-6, time.perf_counter() - started)
    peak_memory = round(torch.cuda.max_memory_allocated() / (1024**2), 2) if torch.cuda.is_available() else 0.0
    report = {
        "schema_version": "videotrace-qwen35-sft-metrics-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "model_path": model_path,
        "adapter_path": str(output_dir.resolve()),
        "dataset_path": str(Path(config["dataset_path"]).resolve()),
        "dataset_sha256": file_sha256(Path(config["dataset_path"])),
        "gradient_payload_sha256": gradient_payload_sha256(records),
        "source_sha256": source_fingerprint(ROOT),
        "training_source_sha256": source_fingerprint(ROOT),
        "split_validation": validation,
        "counts": {"train": len(train_records), "dev": len(dev_records), "frozen_test": len(frozen_test_records)},
        "hyperparameters": {key: value for key, value in config.items() if key not in {"model_path", "dataset_path"}},
        "lora_target_modules": available_targets,
        "history": history,
        "train_loss_last": history[-1]["loss"] if history else None,
        "dev_loss": dev_loss,
        "steps": update_step,
        "steps_this_run": updates_this_run,
        "resumed_from_checkpoint": str(resume_path.resolve()) if resume_path else "",
        "resumed_from_step": resumed_from_step,
        "checkpoint_contract_sha256": contract["contract_sha256"],
        "checkpoint_manifest_sha256": file_sha256(output_dir / "checkpoint_manifest.json"),
        "checkpoint_files": sorted(
            path.name for path in output_dir.iterdir() if path.is_file() and path.name != "checkpoint_manifest.json"
        ),
        "train_tokens": train_tokens_this_run,
        "tokens_per_second": round(train_tokens_this_run / elapsed, 3),
        "elapsed_seconds": round(elapsed, 3),
        "peak_cuda_memory_mib": peak_memory,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "physical_gpu_ids": os.environ.get("VIDEOTRACE_PHYSICAL_GPUS", ""),
        "resume_supported": True,
        "failure_recovery": (
            "checkpoint saves adapter, tokenizer, optimizer.pt, rng_state.pt and trainer_state.json; "
            "checkpoint_manifest.json is the commit marker and resume validates dataset/source/config hashes"
        ),
        "frozen_test_policy": "test records are evaluated only and never passed to the optimizer",
    }
    return report


def _dry_run(config, records, train_records, dev_records, frozen_test_records, validation):
    contract = _training_contract(config, records)
    return {
        "schema_version": "videotrace-qwen35-sft-metrics-v1",
        "status": "dry_run",
        "model_path": str(config["model_path"]),
        "dataset_path": str(Path(config["dataset_path"]).resolve()),
        "dataset_sha256": file_sha256(Path(config["dataset_path"])),
        "gradient_payload_sha256": gradient_payload_sha256(records),
        "source_sha256": source_fingerprint(ROOT),
        "training_source_sha256": source_fingerprint(ROOT),
        "split_validation": validation,
        "counts": {"train": len(train_records), "dev": len(dev_records), "frozen_test": len(frozen_test_records)},
        "gradient_update_records": [record.record_id for record in train_records],
        "frozen_test_records": [record.record_id for record in frozen_test_records],
        "checkpoint_contract": contract,
        "adapter_default_contract": "Web resolves outputs/models/qwen35_sft_adapter when adapter_config.json exists",
    }


def _model_card(metrics: dict) -> dict:
    return {
        "schema_version": "videotrace-qwen35-sft-model-card-v1",
        "model_name": "VideoTrace Qwen3.5 evidence-grounded LoRA adapter",
        "base_model": metrics.get("model_path"),
        "adapter_path": metrics.get("adapter_path", "outputs/models/qwen35_sft_adapter"),
        "intended_use": "Generate Chinese answers whose claims are bound to candidate video timestamps.",
        "training_method": "LoRA/QLoRA-compatible supervised fine-tuning; frozen Qwen3.5 base weights.",
        "data": {
            "dataset": metrics.get("dataset_path"),
            "dataset_sha256": metrics.get("dataset_sha256"),
            "gradient_payload_sha256": metrics.get("gradient_payload_sha256"),
            "train_dev_test_isolation": metrics.get("split_validation", {}).get("valid", False),
            "frozen_test_excluded_from_gradients": True,
            "cola_policy": "cola review is frozen test-only and never enters train/dev updates",
        },
        "checkpoint_recovery": {
            "resume_supported": metrics.get("resume_supported", False),
            "contract_sha256": metrics.get("checkpoint_contract_sha256", ""),
            "manifest_sha256": metrics.get("checkpoint_manifest_sha256", ""),
            "files": metrics.get("checkpoint_files", []),
            "resumed_from_step": metrics.get("resumed_from_step", 0),
            "policy": (
                "The commit manifest binds adapter/tokenizer/optimizer/RNG/trainer state; "
                "dataset, gradient payload, source and optimizer-affecting config must match before resume."
            ),
        },
        "limitations": [
            "Small, manually verified task dataset; not a broad benchmark claim.",
            "Text-grounded SFT teaches evidence formatting and abstention; visual recognition remains provided by Qwen3.5/SigLIP2 inference.",
            "Evaluate adapter against the frozen cola cases before enabling it as the Web default.",
        ],
        "metrics_path": "outputs/models/qwen35_sft_metrics.json",
        "reproducibility": {
            "source_sha256": metrics.get("source_sha256"),
            "training_source_sha256": metrics.get("training_source_sha256", metrics.get("source_sha256")),
            "cuda_visible_devices": metrics.get("cuda_visible_devices", ""),
            "physical_gpu_ids": metrics.get("physical_gpu_ids", ""),
            "resume_supported": metrics.get("resume_supported", False),
        },
    }


def _encode_records(tokenizer, records, max_length: int) -> list[EncodedExample]:
    encoded: list[EncodedExample] = []
    for record in records:
        prompt_text = tokenizer.apply_chat_template(
            _messages(record, include_answer=False),
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            _messages(record, include_answer=True),
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length).input_ids
        if len(prompt_ids) >= len(full_ids):
            raise ValueError(f"answer was truncated or missing for {record.record_id}")
        if len(prompt_ids) >= max_length - 4:
            raise ValueError(f"prompt exceeds max_length for {record.record_id}")
        labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids) :]
        encoded.append(EncodedExample(input_ids=full_ids, labels=labels))
    return encoded


def _messages(record, include_answer: bool) -> list[dict]:
    evidence_lines = []
    for item in record.evidence:
        evidence_lines.append(
            f"[{float(item['start_sec']):.1f}-{float(item['end_sec']):.1f}] {item.get('text', '')}"
        )
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "（没有足够的候选证据）"
    user = f"问题：{record.query}\n候选时间证据：\n{evidence_text}\n请输出带时间戳的中文回答。"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    if include_answer:
        messages.append({"role": "assistant", "content": record.answer})
    return messages


def _collate(examples, tokenizer, torch):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    width = max(len(example.input_ids) for example in examples)
    input_ids = []
    labels = []
    attention = []
    for example in examples:
        padding = width - len(example.input_ids)
        input_ids.append(example.input_ids + [pad_id] * padding)
        labels.append(example.labels + [-100] * padding)
        attention.append([1] * len(example.input_ids) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def _evaluate(model, examples, tokenizer, batch_size, torch, dtype):
    model.eval()
    losses = []
    with torch.no_grad():
        for offset in range(0, len(examples), batch_size):
            batch = _collate(examples[offset : offset + batch_size], tokenizer, torch)
            batch = {key: value.to(model.device) for key, value in batch.items()}
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
    model.train()
    return round(sum(losses) / max(1, len(losses)), 6)


def _save_checkpoint(
    model,
    tokenizer,
    optimizer,
    output_dir: Path,
    *,
    step: int,
    next_epoch_index: int,
    next_batch_offset: int,
    history: list[dict],
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
            "schema_version": "videotrace-sft-trainer-state-v2",
            "global_step": int(step),
            "next_epoch_index": int(next_epoch_index),
            "next_batch_offset": int(next_batch_offset),
            "history": list(history),
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
            "schema_version": "videotrace-sft-checkpoint-manifest-v1",
            "global_step": int(step),
            "contract_sha256": contract["contract_sha256"],
            "files": files,
        },
    )


def _load_resume_state(resume_path: Path, optimizer, torch, *, contract: dict) -> dict:
    _validate_resume_checkpoint(resume_path, contract)
    state = json.loads((resume_path / "trainer_state.json").read_text(encoding="utf-8"))
    optimizer.load_state_dict(torch.load(resume_path / "optimizer.pt", map_location="cpu", weights_only=False))
    rng_path = resume_path / "rng_state.pt"
    if rng_path.is_file():
        rng = torch.load(rng_path, map_location="cpu", weights_only=False)
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
        raise SystemExit(
            f"SFT resume checkpoint is incomplete: {resume_path}; missing {', '.join(missing)}"
        )
    manifest = json.loads((resume_path / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    state = json.loads((resume_path / "trainer_state.json").read_text(encoding="utf-8"))
    if manifest.get("contract_sha256") != contract.get("contract_sha256"):
        raise SystemExit(
            "SFT resume checkpoint provenance mismatch: "
            f"expected {contract.get('contract_sha256')}, checkpoint {manifest.get('contract_sha256')}"
        )
    if state.get("contract", {}).get("contract_sha256") != contract.get("contract_sha256"):
        raise SystemExit("SFT resume trainer state contract does not match current dataset/source/config")
    mismatches = []
    for name, expected in dict(manifest.get("files") or {}).items():
        path = resume_path / name
        current = file_sha256(path) if path.is_file() else ""
        if current != expected:
            mismatches.append(name)
    if mismatches:
        raise SystemExit(
            "SFT resume checkpoint hash mismatch; refusing partial/corrupt state: "
            + ", ".join(sorted(mismatches))
        )


def _training_contract(config: dict, records) -> dict:
    payload = {
        "schema_version": "videotrace-sft-resume-contract-v1",
        "dataset_sha256": file_sha256(Path(config["dataset_path"])),
        "gradient_payload_sha256": gradient_payload_sha256(records),
        "source_sha256": source_fingerprint(ROOT),
        "model_path": str(config["model_path"]).replace("\\", "/"),
        "seed": int(config.get("seed", 42)),
        "dtype": str(config.get("dtype", "bfloat16")),
        "quantization": str(config.get("quantization", "none")),
        "max_length": int(config.get("max_length", 768)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 1)),
        "per_device_batch_size": int(config.get("per_device_batch_size", 1)),
        "learning_rate": float(config.get("learning_rate", 2e-4)),
        "weight_decay": float(config.get("weight_decay", 0.01)),
        "lora_r": int(config.get("lora_r", 16)),
        "lora_alpha": int(config.get("lora_alpha", 32)),
        "lora_dropout": float(config.get("lora_dropout", 0.05)),
        "lora_target_modules": list(config.get("lora_target_modules", ["q_proj", "v_proj"])),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "contract_sha256": hashlib.sha256(encoded).hexdigest()}


def _ordered_examples(examples: list[EncodedExample], seed: int, epoch: int) -> list[EncodedExample]:
    indices = list(range(len(examples)))
    random.Random(seed + epoch).shuffle(indices)
    return [examples[index] for index in indices]


def _next_position(epoch: int, offset: int, total: int, batch_size: int) -> tuple[int, int]:
    next_offset = offset + batch_size
    if next_offset >= total:
        return epoch + 1, 0
    return epoch, next_offset


def _rng_state(torch) -> dict:
    payload = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _atomic_torch_save(value, path: Path, torch) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _available_target_modules(model, requested):
    names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    return [module for module in requested if module in names]


def _load_config(path: Path) -> dict:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _resolve_config_paths(config: dict) -> dict:
    result = dict(config)
    for key in ["dataset_path", "output_dir", "metrics_path", "model_card_path", "resume_from_checkpoint"]:
        value = str(result.get(key, "") or "")
        if value and not Path(value).is_absolute():
            result[key] = str((ROOT / value).resolve())
    return result


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
