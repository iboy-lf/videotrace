from __future__ import annotations

import json
from pathlib import Path

from videomemo.eval.delivery_readiness import (
    _actual_resume_evidence,
    _checkpoint_recovery_evidence,
    _dpo_gpu_binding_evidence,
    _training_source_sha256,
)
from videomemo.eval.reproducibility import file_sha256


def test_delivery_checkpoint_recovery_requires_commit_manifest_and_matching_card(tmp_path):
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    contract_sha = "a" * 64
    for name, value in {
        "adapter_config.json": b"{}",
        "adapter_model.safetensors": b"weights",
        "optimizer.pt": b"optimizer",
        "rng_state.pt": b"rng",
    }.items():
        (checkpoint / name).write_bytes(value)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 1,
                "contract": {"contract_sha256": contract_sha},
            }
        ),
        encoding="utf-8",
    )
    file_hashes = {
        path.name: file_sha256(path)
        for path in checkpoint.iterdir()
        if path.is_file()
    }
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "global_step": 1,
                "contract_sha256": contract_sha,
                "files": file_hashes,
            }
        ),
        encoding="utf-8",
    )
    metrics = {
        "steps": 1,
        "checkpoint_contract_sha256": contract_sha,
        "checkpoint_manifest_sha256": file_sha256(manifest_path),
        "checkpoint_files": sorted(file_hashes),
    }
    card = {
        "checkpoint_recovery": {
            "resume_supported": True,
            "contract_sha256": contract_sha,
            "manifest_sha256": file_sha256(manifest_path),
            "files": sorted(file_hashes),
        }
    }
    paths = {
        "sft_checkpoint_manifest": manifest_path,
        "sft_trainer_state": checkpoint / "trainer_state.json",
    }
    report = _checkpoint_recovery_evidence(paths, "sft", metrics, card)
    assert report["valid"] is True
    assert report["hash_mismatches"] == []

    (checkpoint / "optimizer.pt").write_bytes(b"corrupt")
    report = _checkpoint_recovery_evidence(paths, "sft", metrics, card)
    assert report["valid"] is False
    assert [item["name"] for item in report["hash_mismatches"]] == ["optimizer.pt"]


def test_delivery_requires_an_actual_hash_validated_resume_step(tmp_path):
    checkpoint = tmp_path / "outputs" / "runs" / "latest" / "sft_resume_validation"
    checkpoint.mkdir(parents=True)
    contract_sha = "b" * 64
    for name, value in {
        "adapter_config.json": b"{}",
        "adapter_model.safetensors": b"weights",
        "optimizer.pt": b"optimizer",
        "rng_state.pt": b"rng",
        "trainer_state.json": b"{}",
    }.items():
        (checkpoint / name).write_bytes(value)
    file_hashes = {
        path.name: file_sha256(path)
        for path in checkpoint.iterdir()
        if path.is_file()
    }
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "global_step": 2,
                "contract_sha256": contract_sha,
                "files": file_hashes,
            }
        ),
        encoding="utf-8",
    )
    training_source_sha = "c" * 64
    admission_source_sha = "f" * 64
    resume_report = tmp_path / "outputs" / "reports" / "qwen35_sft_resume_validation.json"
    resume_report.parent.mkdir(parents=True)
    resume_report.write_text(
        json.dumps(
            {
                "status": "completed",
                "source_sha256": training_source_sha,
                "training_source_sha256": training_source_sha,
                "adapter_path": str(checkpoint),
                "resumed_from_checkpoint": str(tmp_path / "outputs/models/qwen35_sft_adapter"),
                "resumed_from_step": 1,
                "steps_this_run": 1,
                "steps": 2,
                "physical_gpu_ids": "2,3",
                "checkpoint_contract_sha256": contract_sha,
                "checkpoint_manifest_sha256": file_sha256(manifest_path),
                "checkpoint_files": sorted(file_hashes),
                "dataset_sha256": "d" * 64,
                "gradient_payload_sha256": "e" * 64,
            }
        ),
        encoding="utf-8",
    )
    paths = {
        "sft_resume_validation": resume_report,
        "sft_resume_checkpoint_manifest": manifest_path,
    }
    canonical = {
        "steps": 1,
        "source_sha256": admission_source_sha,
        "training_source_sha256": training_source_sha,
        "checkpoint_contract_sha256": contract_sha,
        "dataset_sha256": "d" * 64,
        "gradient_payload_sha256": "e" * 64,
    }
    expected_training_source = _training_source_sha256(canonical, {})
    report = _actual_resume_evidence(paths, "sft", canonical, expected_training_source)
    assert report["valid"] is True
    assert report["training_source_sha256"] == training_source_sha
    assert report["expected_training_source_sha256"] == training_source_sha
    assert report["resumed_from_step"] == 1
    assert report["final_step"] == 2

    (checkpoint / "optimizer.pt").write_bytes(b"corrupt")
    assert _actual_resume_evidence(paths, "sft", canonical, expected_training_source)["valid"] is False


def test_delivery_accepts_hash_audited_dpo_model_parallel_gpu_binding():
    report = _dpo_gpu_binding_evidence(
        {
            "physical_gpu_ids": "2,3",
            "cuda_visible_devices": "2,3",
            "model_parallel": {
                "active": True,
                "trainable_parameter_devices": ["cuda:0", "cuda:1"],
            },
        },
        [2, 3],
    )
    assert report["valid"] is True
    assert report["mode"] == "model_parallel"


def test_delivery_rejects_model_parallel_gpu_audit_mismatch():
    report = _dpo_gpu_binding_evidence(
        {
            "physical_gpu_ids": "2,3",
            "cuda_visible_devices": "2,3",
            "model_parallel": {
                "active": True,
                "trainable_parameter_devices": ["cuda:0"],
            },
        },
        [2, 3],
    )
    assert report["valid"] is False


def test_delivery_accepts_legacy_single_gpu_binding_from_a_safe_pair():
    report = _dpo_gpu_binding_evidence(
        {
            "physical_gpu_ids": "2",
            "cuda_visible_devices": "2",
            "model_parallel": {"active": False, "trainable_parameter_devices": []},
        },
        [2, 3],
    )
    assert report["valid"] is True
    assert report["mode"] == "single_gpu"
