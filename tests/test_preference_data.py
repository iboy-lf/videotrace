from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from videomemo.training.preference_data import (
    PreferenceRecord,
    build_grounded_preference_dataset,
    load_preference_records,
    preference_gradient_payload_sha256,
    validate_preference_records,
)


ROOT = Path(__file__).resolve().parents[1]


def test_preference_dataset_uses_explicit_negatives_and_freezes_cola(tmp_path):
    output = tmp_path / "grounded_dpo.jsonl"
    summary = build_grounded_preference_dataset(
        ROOT / "data/sft/grounded_qa.jsonl",
        ROOT / "data/preference/preference_annotations.json",
        output,
        project_root=ROOT,
    )
    rows = load_preference_records(output)
    assert summary["validation"]["valid"] is True
    assert summary["counts"] == {"train": 7, "dev": 4, "test": 1}
    assert summary["negative_type_counts"] == {
        "wrong_timestamp": 4,
        "missing_timestamp": 1,
        "hallucinated_detail": 2,
        "unsupported_overclaim": 5,
    }
    assert summary["cola_video_in_train_or_dev"] is False
    assert [row.pair_id for row in rows if row.frozen_test] == [
        "cola-review-frozen-overview:wrong-timestamp"
    ]
    assert all(row.chosen != row.rejected for row in rows)
    assert summary["gradient_payload_sha256"] == preference_gradient_payload_sha256(rows)


def test_preference_validation_rejects_leakage_duplicates_and_bad_negative_contract():
    base = PreferenceRecord(
        pair_id="pair-a",
        source_record_id="source-a",
        video_id="video-a",
        video_path="raw/a.mp4",
        split="train",
        query="问题",
        evidence=({"start_sec": 1.0, "end_sec": 2.0, "text": "证据"},),
        chosen="结论：证据支持。(timestamp=1.0-2.0)",
        rejected="结论：错误时间。(timestamp=8.0-9.0)",
        expected_behavior="answer",
        negative_type="wrong_timestamp",
        rationale="错误时间窗",
        provenance="unit-test",
    )
    rows = [
        base,
        replace(base, pair_id="pair-a", video_id="cola-review-frozen-test", split="dev"),
        replace(
            base,
            pair_id="pair-c",
            video_id="video-c",
            split="test",
            rejected="结论：仍是正确时间。(timestamp=1.0-2.0)",
        ),
    ]
    report = validate_preference_records(rows)
    assert report["valid"] is False
    assert any("duplicate pair_id" in error for error in report["errors"])
    assert any("frozen cola" in error for error in report["errors"])
    assert any("wrong_timestamp negative" in error for error in report["errors"])


def test_preference_gradient_hash_excludes_dev_and_frozen_test():
    train = _record("train", "train-pair", "video-train")
    dev = _record("dev", "dev-pair", "video-dev")
    test = replace(
        _record("test", "test-pair", "cola-review-frozen-test"),
        frozen_test=True,
    )
    baseline = preference_gradient_payload_sha256([train, dev, test])
    assert preference_gradient_payload_sha256(
        [train, replace(dev, rejected="changed dev"), replace(test, rejected="changed test")]
    ) == baseline
    assert preference_gradient_payload_sha256(
        [replace(train, rejected="changed train"), dev, test]
    ) != baseline


def test_preference_builder_rejects_unknown_sft_source(tmp_path):
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        """{
  "schema_version": "videotrace-preference-annotations-v1",
  "pairs": [{
    "pair_id": "unknown",
    "source_record_id": "missing-record",
    "negative_type": "wrong_timestamp",
    "rejected": "timestamp=1.0-2.0",
    "rationale": "unit",
    "provenance": "unit"
  }]
}
""",
        encoding="utf-8",
    )
    try:
        build_grounded_preference_dataset(
            ROOT / "data/sft/grounded_qa.jsonl",
            annotations,
            tmp_path / "out.jsonl",
            project_root=ROOT,
        )
    except ValueError as exc:
        assert "unknown SFT record" in str(exc)
    else:
        raise AssertionError("unknown source record must be rejected")


def _record(split: str, pair_id: str, video_id: str) -> PreferenceRecord:
    return PreferenceRecord(
        pair_id=pair_id,
        source_record_id=pair_id + ":source",
        video_id=video_id,
        video_path="raw/example.mp4",
        split=split,
        query="问题",
        evidence=({"start_sec": 1.0, "end_sec": 2.0, "text": "证据"},),
        chosen="结论：正确。(timestamp=1.0-2.0)",
        rejected="结论：错误。(timestamp=8.0-9.0)",
        expected_behavior="answer",
        negative_type="wrong_timestamp",
        rationale="unit",
        provenance="unit",
    )
