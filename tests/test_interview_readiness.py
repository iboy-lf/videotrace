from __future__ import annotations

import json
from pathlib import Path

from videomemo.eval.interview_readiness import validate_interview_package
from videomemo.eval.reproducibility import file_sha256, source_fingerprint
from videomemo.reranker.model_card import build_reranker_model_card
from videomemo.reranker.neural import FEATURE_NAMES


def _features(value: float) -> dict[str, float]:
    return {name: value for name in FEATURE_NAMES}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_model_card_records_hashes_contract_and_baselines(tmp_path):
    rows = []
    for index in range(5):
        group_id = f"group-{index}"
        rows.extend(
            [
                {
                    "group_id": group_id,
                    "video_id": "dev-video",
                    "split": "dev",
                    "label": 1.0,
                    "features": _features(0.9),
                },
                {
                    "group_id": group_id,
                    "video_id": "dev-video",
                    "split": "dev",
                    "label": 0.0,
                    "features": _features(0.1),
                },
            ]
        )
    dataset = tmp_path / "rows.jsonl"
    checkpoint = tmp_path / "model.pt"
    metrics = tmp_path / "metrics.json"
    _write_rows(dataset, rows)
    checkpoint.write_bytes(b"checkpoint")
    _write_json(
        metrics,
        {
            "pairwise_accuracy": 0.75,
            "base_pairwise_accuracy": 0.5,
            "recommended_blend_weight": 0.5,
            "blended_pairwise_accuracy": 0.8,
            "train_loss": 0.4,
            "eval_loss": 0.8,
            "best_epoch": 12,
            "hidden_dim": 16,
        },
    )

    card = build_reranker_model_card(
        str(dataset),
        str(checkpoint),
        str(metrics),
        source_sha256="source",
    )
    assert card["architecture"]["feature_contract_valid"]
    assert card["architecture"]["hidden_dims"] == [16, 8]
    assert card["artifacts"]["checkpoint_sha256"] == file_sha256(checkpoint)
    assert card["artifacts"]["dataset_sha256"] == file_sha256(dataset)
    assert card["data_contract"]["contains_test_rows"] is False
    assert card["evaluation"]["pairwise_eval_pairs"] > 0
    assert card["evaluation"]["pairwise_baselines"]["base_fusion_score"] == 1.0


def test_interview_package_validator_checks_end_to_end_contract(tmp_path):
    root = tmp_path / "project"
    for relative, value in (
        ("src/module.py", "VALUE = 1\n"),
        ("scripts/tool.py", "print('ok')\n"),
        ("configs/demo.yaml", "top_k: 3\n"),
        ("pyproject.toml", "[project]\nname='demo'\nversion='0.1'\n"),
        ("README.md", "# Demo\n"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    video = root / "data" / "raw" / "final.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video")
    artifact_dir = root / "outputs" / "demo"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "demo.html").write_text("<html></html>", encoding="utf-8")

    source_sha = source_fingerprint(root)
    pack = {
        "video_path": str(video),
        "duration_sec": 100.0,
        "segments": [{"segment_id": f"s{i}"} for i in range(3)],
        "answer": (
            "问题：流程？\n结论：有证据。 "
            "(timestamp=0.0-10.0) (timestamp=45.0-55.0) (timestamp=90.0-100.0)"
        ),
        "timeline": [
            {"start_sec": 0.0, "end_sec": 10.0, "selection_reason": "temporal_coverage:opening"},
            {"start_sec": 45.0, "end_sec": 55.0, "selection_reason": "temporal_coverage:middle"},
            {"start_sec": 90.0, "end_sec": 100.0, "selection_reason": "temporal_coverage:ending"},
        ],
        "clips": [
            {"file": str(video), "playback_mode": "source_video_window"}
            for index in range(3)
        ],
        "metadata": {
            "source_sha256": source_sha,
            "video_sha256": file_sha256(video),
            "segment_understanding": {"backend": "qwen35_local"},
            "vlm": {
                "backend": "frozen_siglip",
                "persistent_index": {"enabled": True, "num_vectors": 3},
            },
            "reranker": {"backend": "neural"},
            "llm_backend": "qwen35_local",
            "environment": {
                "python": "3.11",
                "platform": "test",
                "packages": {"torch": "2.0", "transformers": "4.0"},
                "physical_gpu_ids": "0,1",
            },
            "agent_run": {
                "verified": True,
                "verification": {"coverage": 1.0},
                "tool_trace": [{"name": "retrieve"}],
                "context": {
                    "evidence_tags": [
                        "timestamp=0.0-10.0",
                        "timestamp=45.0-55.0",
                        "timestamp=90.0-100.0",
                    ]
                },
            },
        },
    }
    knowledge_pack = artifact_dir / "knowledge_pack.json"
    _write_json(knowledge_pack, pack)

    rows = []
    for group_id in ("eval-a", "eval-b"):
        rows.extend(
            [
                {
                    "group_id": group_id,
                    "video_id": "dev-video",
                    "split": "dev",
                    "label": 1.0,
                    "features": _features(0.8),
                },
                {
                    "group_id": group_id,
                    "video_id": "dev-video",
                    "split": "dev",
                    "label": 0.0,
                    "features": _features(0.2),
                },
            ]
        )
    dataset = root / "outputs_train" / "rows.jsonl"
    summary = root / "outputs_train" / "rows.summary.json"
    checkpoint = root / "outputs" / "models" / "model.pt"
    metrics = root / "outputs" / "models" / "metrics.json"
    model_card = root / "outputs" / "models" / "model_card.json"
    _write_rows(dataset, rows)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    _write_json(
        metrics,
        {
            "num_rows": 4,
            "num_groups": 2,
            "num_positive": 2,
            "num_pairwise_eval_groups": 2,
            "pairwise_accuracy": 1.0,
            "blended_pairwise_accuracy": 1.0,
            "hidden_dim": 16,
        },
    )
    _write_json(summary, {"num_rows": 4, "num_groups": 2, "num_positive": 2})
    _write_json(
        model_card,
        {
            "architecture": {
                "feature_names": FEATURE_NAMES,
                "feature_contract_valid": True,
                "hidden_dims": [16, 8],
            },
            "artifacts": {
                "checkpoint_sha256": file_sha256(checkpoint),
                "dataset_sha256": file_sha256(dataset),
                "metrics_sha256": file_sha256(metrics),
            },
            "data_contract": {"contains_test_rows": False},
            "evaluation": {
                "base_fusion_pairwise_accuracy": 0.5,
                "pairwise_baselines": {"base_fusion_score": 0.5},
            },
            "source_sha256": source_sha,
        },
    )

    report = validate_interview_package(
        root,
        knowledge_pack,
        checkpoint,
        metrics,
        dataset,
        summary,
        model_card,
    )
    assert report["valid"], report["failures"]
    assert report["checks_passed"] == report["checks_total"]

    rows[0]["split"] = "test"
    _write_rows(dataset, rows)
    leaked = validate_interview_package(
        root,
        knowledge_pack,
        checkpoint,
        metrics,
        dataset,
        summary,
        model_card,
    )
    assert "training_split_has_no_test_leakage" in leaked["failures"]
    assert "training_artifact_hashes_match" in leaked["failures"]
