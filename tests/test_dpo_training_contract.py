from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import train_qwen35_dpo as trainer
from videomemo.eval.reproducibility import file_sha256
from videomemo.training.dpo_objective import dpo_statistics


ROOT = Path(__file__).resolve().parents[1]


def test_dpo_objective_has_standard_initial_loss_and_gradient_signs():
    stats = dpo_statistics(-10.0, -12.0, -10.0, -12.0, beta=0.1)
    assert stats["loss"] == pytest.approx(math.log(2.0))
    assert stats["reward_margin"] == pytest.approx(0.0)
    assert stats["chosen_logp_gradient"] == pytest.approx(-0.05)
    assert stats["rejected_logp_gradient"] == pytest.approx(0.05)


def test_dpo_objective_rewards_policy_margin_over_frozen_reference():
    improved = dpo_statistics(-9.0, -13.0, -10.0, -12.0, beta=0.1)
    regressed = dpo_statistics(-11.0, -11.0, -10.0, -12.0, beta=0.1)
    assert improved["loss"] < math.log(2.0)
    assert improved["reward_margin"] > 0
    assert improved["reward_preference_correct"] is True
    assert regressed["loss"] > math.log(2.0)
    assert regressed["reward_margin"] < 0


def test_dpo_objective_rejects_non_positive_beta():
    with pytest.raises(ValueError, match="beta"):
        dpo_statistics(0, 0, 0, 0, beta=0)


def test_dpo_dry_run_binds_sft_reference_and_frozen_split(tmp_path):
    # The SFT adapter weights are not redistributed with the repository, so a
    # fresh clone cannot run this contract check. Skip explicitly rather than
    # failing: a red suite for a missing, deliberately-absent artifact would
    # train the reader to ignore the suite.
    sft_adapter = ROOT / "outputs/models/qwen35_sft_adapter"
    if not (sft_adapter / "adapter_model.safetensors").is_file():
        pytest.skip(
            "requires outputs/models/qwen35_sft_adapter weights, which are not "
            "redistributed; see docs/REVALIDATION.md"
        )
    config = {
        "model_path": "/lavender/models/Qwen3.5-9B",
        "initial_adapter_path": str(ROOT / "outputs/models/qwen35_sft_adapter"),
        "dataset_path": str(ROOT / "data/preference/grounded_dpo.jsonl"),
        "source_sft_path": str(ROOT / "data/sft/grounded_qa.jsonl"),
        "output_dir": str(tmp_path / "adapter"),
        "metrics_path": str(tmp_path / "metrics.json"),
        "model_card_path": str(tmp_path / "model_card.json"),
        "reference_logprobs_path": str(tmp_path / "reference.json"),
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
            "scripts/train_qwen35_dpo.py",
            "--config",
            str(config_path),
            "--metrics-path",
            str(tmp_path / "metrics.json"),
            "--model-card-path",
            str(tmp_path / "model_card.json"),
            "--reference-logprobs-path",
            str(tmp_path / "reference.json"),
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
    card = json.loads((tmp_path / "model_card.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "dry_run"
    assert metrics["counts"] == {"train": 7, "dev": 4, "frozen_test": 1}
    assert len(metrics["gradient_update_pairs"]) == 7
    assert metrics["frozen_test_pairs"] == ["cola-review-frozen-overview:wrong-timestamp"]
    assert metrics["initial_adapter_sha256"]
    assert card["data"]["frozen_test_excluded_from_gradients"] is True


def test_dpo_resume_contract_binds_reference_data_source_and_optimizer_settings():
    contract = trainer._training_contract(
        {
            "model_path": "/lavender/models/Qwen3.5-9B",
            "seed": 43,
            "dtype": "bfloat16",
            "quantization": "none",
            "max_length": 768,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "beta": 0.1,
            "max_grad_norm": 1.0,
            "optimizer_state_offload": True,
            "model_parallel": True,
            "model_parallel_max_memory_gib": 22,
        },
        dataset_sha="a" * 64,
        gradient_sha="b" * 64,
        reference_sha="c" * 64,
        initial_adapter_sha="d" * 64,
    )
    assert contract["schema_version"] == "videotrace-dpo-resume-contract-v1"
    assert contract["reference_logprobs_sha256"] == "c" * 64
    assert contract["initial_adapter_sha256"] == "d" * 64
    assert contract["optimizer_state_offload"] is True
    assert contract["model_parallel"] is True
    assert len(contract["contract_sha256"]) == 64


def test_dpo_optimizer_state_can_be_offloaded_and_restored_without_value_changes():
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    parameter.grad = torch.tensor([0.5])
    optimizer.step()
    expected = {
        key: value.detach().cpu().clone()
        for key, value in optimizer.state[parameter].items()
        if torch.is_tensor(value)
    }
    trainer._move_optimizer_state(optimizer, torch.device("cpu"), torch)
    assert all(
        not value.is_cuda
        for value in optimizer.state[parameter].values()
        if torch.is_tensor(value)
    )
    trainer._move_optimizer_state(optimizer, parameter.device, torch)
    for key, value in expected.items():
        assert torch.equal(optimizer.state[parameter][key].cpu(), value)
    trainer._move_optimizer_state_to_parameters(optimizer, torch)
    assert all(
        value.device == parameter.device
        for value in optimizer.state[parameter].values()
        if torch.is_tensor(value)
    )


def test_dpo_device_map_uses_balanced_visible_gpus_when_requested():
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

    class FakeTorch:
        cuda = FakeCuda()

    device_map, max_memory = trainer._model_device_map(
        {"model_parallel": True, "model_parallel_max_memory_gib": 21},
        FakeTorch(),
    )
    assert device_map == "balanced"
    assert max_memory == {0: "21GiB", 1: "21GiB"}


def test_dpo_resume_checkpoint_rejects_partial_provenance_or_corruption(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    contract = {"contract_sha256": "e" * 64}
    with pytest.raises(RuntimeError, match="incomplete"):
        trainer._validate_resume_checkpoint(checkpoint, contract)

    _write_dpo_checkpoint(checkpoint, contract)
    trainer._validate_resume_checkpoint(checkpoint, contract)
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        trainer._validate_resume_checkpoint(checkpoint, {"contract_sha256": "f" * 64})

    (checkpoint / "rng_state.pt").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        trainer._validate_resume_checkpoint(checkpoint, contract)


def _write_dpo_checkpoint(checkpoint: Path, contract: dict) -> None:
    for name, content in {
        "adapter_config.json": b"{}",
        "adapter_model.safetensors": b"weights",
        "optimizer.pt": b"optimizer",
        "rng_state.pt": b"rng",
    }.items():
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
                "schema_version": "videotrace-dpo-checkpoint-manifest-v1",
                "contract_sha256": contract["contract_sha256"],
                "files": hashes,
            }
        ),
        encoding="utf-8",
    )
