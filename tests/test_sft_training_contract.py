from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import train_qwen35_sft as trainer
from videomemo.eval.reproducibility import file_sha256


ROOT = Path(__file__).resolve().parents[1]


def test_sft_dry_run_records_frozen_split_and_resume_contract(tmp_path):
    config = {
        "model_path": "/lavender/models/Qwen3.5-9B",
        "dataset_path": str(ROOT / "data/sft/grounded_qa.jsonl"),
        "output_dir": str(tmp_path / "adapter"),
        "metrics_path": str(tmp_path / "metrics.json"),
        "model_card_path": str(tmp_path / "model_card.json"),
        "device": "cuda:0",
        "dtype": "bfloat16",
        "quantization": "none",
        "max_steps": 1,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_qwen35_sft.py",
            "--config",
            str(config_path),
            "--metrics-path",
            str(tmp_path / "metrics.json"),
            "--model-card-path",
            str(tmp_path / "model_card.json"),
            "--num-train-epochs",
            "2",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "dry_run"
    assert metrics["counts"] == {"train": 7, "dev": 4, "frozen_test": 1}
    assert len(metrics["gradient_update_records"]) == 7
    assert metrics["frozen_test_records"] == ["cola-review:frozen-overview-test"]
    contract = metrics["checkpoint_contract"]
    assert contract["schema_version"] == "videotrace-sft-resume-contract-v1"
    assert contract["dataset_sha256"] == file_sha256(ROOT / "data/sft/grounded_qa.jsonl")
    assert len(contract["contract_sha256"]) == 64


def test_sft_resume_checkpoint_rejects_incomplete_or_wrong_provenance(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    contract = {"contract_sha256": "a" * 64}
    with pytest.raises(SystemExit, match="incomplete"):
        trainer._validate_resume_checkpoint(checkpoint, contract)

    _write_checkpoint(checkpoint, contract)
    trainer._validate_resume_checkpoint(checkpoint, contract)
    (checkpoint / "adapter_model.safetensors").unlink()
    with pytest.raises(SystemExit, match="adapter_model.safetensors"):
        trainer._validate_resume_checkpoint(checkpoint, contract)
    _write_checkpoint(checkpoint, contract)
    with pytest.raises(SystemExit, match="provenance mismatch"):
        trainer._validate_resume_checkpoint(checkpoint, {"contract_sha256": "b" * 64})


def test_sft_resume_checkpoint_rejects_hash_corruption(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    contract = {"contract_sha256": "c" * 64}
    _write_checkpoint(checkpoint, contract)
    (checkpoint / "optimizer.pt").write_bytes(b"corrupt")
    with pytest.raises(SystemExit, match="hash mismatch"):
        trainer._validate_resume_checkpoint(checkpoint, contract)


def test_sft_epoch_order_and_resume_position_are_deterministic():
    examples = [trainer.EncodedExample([index], [index]) for index in range(8)]
    first = [item.input_ids[0] for item in trainer._ordered_examples(examples, 42, 0)]
    repeated = [item.input_ids[0] for item in trainer._ordered_examples(examples, 42, 0)]
    second_epoch = [item.input_ids[0] for item in trainer._ordered_examples(examples, 42, 1)]
    assert first == repeated
    assert first != second_epoch
    assert trainer._next_position(0, 2, total=8, batch_size=2) == (0, 4)
    assert trainer._next_position(0, 6, total=8, batch_size=2) == (1, 0)


def _write_checkpoint(checkpoint: Path, contract: dict) -> None:
    files = {
        "adapter_config.json": b"{}",
        "adapter_model.safetensors": b"weights",
        "optimizer.pt": b"optimizer",
        "rng_state.pt": b"rng",
    }
    for name, content in files.items():
        (checkpoint / name).write_bytes(content)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 1, "contract": contract}),
        encoding="utf-8",
    )
    hashes = {
        path.name: file_sha256(path)
        for path in checkpoint.iterdir()
        if path.is_file()
    }
    (checkpoint / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "videotrace-sft-checkpoint-manifest-v1",
                "contract_sha256": contract["contract_sha256"],
                "files": hashes,
            }
        ),
        encoding="utf-8",
    )
