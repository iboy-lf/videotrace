from __future__ import annotations

from pathlib import Path
import json

import pytest

from videomemo.training.sft_data import (
    SFTRecord,
    build_grounded_sft_dataset,
    gradient_payload_sha256,
    load_sft_records,
    validate_sft_records,
)


ROOT = Path(__file__).resolve().parents[1]


def test_grounded_sft_dataset_has_group_isolated_frozen_cola_test(tmp_path):
    output = tmp_path / "grounded.jsonl"
    summary = build_grounded_sft_dataset(
        ROOT / "data/supervision/reranker_annotations.json",
        output,
        cola_pack_path=ROOT / "outputs/cola_review_qwen35/knowledge_pack.json",
        project_root=ROOT,
    )
    assert summary["validation"]["valid"] is True
    assert summary["counts"] == {"train": 7, "dev": 4, "test": 1}
    rows = load_sft_records(output)
    assert any(row.frozen_test for row in rows)
    assert all("cola" not in row.video_id.lower() or row.split == "test" for row in rows)
    assert sum(row.expected_behavior == "abstain" for row in rows) == 5
    assert summary["gradient_payload_sha256"] == gradient_payload_sha256(rows)


def test_sft_validation_rejects_cross_split_and_leakage():
    rows = [
        SFTRecord("a", "same", "raw/a.mp4", "train", "q", tuple(), "a", "abstain", "test"),
        SFTRecord("b", "same", "raw/a.mp4", "dev", "q", tuple(), "a", "abstain", "test"),
        SFTRecord("c", "cola-review-frozen-test", "raw/cola_review.mp4", "train", "q", tuple(), "a", "answer", "test"),
    ]
    report = validate_sft_records(rows)
    assert report["valid"] is False
    assert any("crosses splits" in error for error in report["errors"])
    assert any("frozen cola" in error for error in report["errors"])


def test_sft_jsonl_schema_is_stable(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "videotrace-grounded-sft-v1",
                "record_id": "r1",
                "video_id": "v1",
                "video_path": "raw/v.mp4",
                "split": "train",
                "query": "问题",
                "evidence": [],
                "answer": "问题：问题\n结论：证据不足",
                "expected_behavior": "abstain",
                "provenance": "unit-test",
                "frozen_test": False,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_sft_records(path)
    assert rows[0].record_id == "r1"
    assert rows[0].expected_behavior == "abstain"


def test_gradient_payload_hash_uses_only_train_records():
    train = SFTRecord(
        "train-a", "video-a", "raw/a.mp4", "train", "q", tuple(), "a", "abstain", "unit"
    )
    dev = SFTRecord(
        "dev-a", "video-b", "raw/b.mp4", "dev", "q", tuple(), "a", "abstain", "unit"
    )
    frozen = SFTRecord(
        "test-a", "cola-review-frozen-test", "raw/c.mp4", "test", "q", tuple(), "a", "abstain", "unit", True
    )
    baseline = gradient_payload_sha256([train, dev, frozen])
    changed_dev = SFTRecord(
        "dev-a", "video-b", "raw/b.mp4", "dev", "changed", tuple(), "changed", "abstain", "unit"
    )
    assert gradient_payload_sha256([train, changed_dev, frozen]) == baseline
    changed_train = SFTRecord(
        "train-a", "video-a", "raw/a.mp4", "train", "changed", tuple(), "a", "abstain", "unit"
    )
    assert gradient_payload_sha256([changed_train, dev, frozen]) != baseline
