from __future__ import annotations

import json
from pathlib import Path
import pickle

import pytest

from videomemo.pipeline import VideoMemoPipeline
from videomemo.training.answer_verifier import (
    build_answer_verifier_rows,
    train_answer_verifier,
    validate_answer_verifier_rows,
    write_answer_verifier_dataset,
)
from videomemo.training.preference_data import load_preference_records
from videomemo.verifier.calibrated import CalibratedAnswerVerifier, PORTABLE_MODEL_FORMAT


ROOT = Path(__file__).resolve().parents[1]
PREFERENCE_PATH = ROOT / "data" / "preference" / "grounded_dpo.jsonl"


def _train(tmp_path: Path) -> tuple[CalibratedAnswerVerifier, dict]:
    dataset = tmp_path / "answer_verifier.jsonl"
    summary = tmp_path / "answer_verifier.summary.json"
    checkpoint = tmp_path / "answer_verifier.pkl"
    metrics = tmp_path / "answer_verifier_metrics.json"
    card = tmp_path / "answer_verifier_model_card.json"
    write_answer_verifier_dataset(PREFERENCE_PATH, dataset, summary, ROOT)
    train_answer_verifier(dataset, checkpoint, metrics, card, ROOT)
    return CalibratedAnswerVerifier(str(checkpoint)), json.loads(metrics.read_text(encoding="utf-8"))


def test_answer_verifier_dataset_is_split_isolated() -> None:
    records = load_preference_records(str(PREFERENCE_PATH))
    rows = build_answer_verifier_rows(records)
    report = validate_answer_verifier_rows(rows)
    assert report["valid"] is True
    assert report["counts"] == {"train": 14, "dev": 8, "test": 2}
    assert report["frozen_test_rows"] == 2
    train_ids = {row["pair_id"] for row in rows if row["split"] == "train"}
    assert "cola-review-frozen-overview:wrong-timestamp" not in train_ids


def test_answer_verifier_training_preserves_frozen_test(tmp_path: Path) -> None:
    _, metrics = _train(tmp_path)
    assert b"\r\n" not in (tmp_path / "answer_verifier.jsonl").read_bytes()
    assert metrics["frozen_test_excluded_from_gradient"] is True
    assert metrics["counts"] == {"train": 14, "dev": 8, "test": 2}
    assert metrics["evaluations"]["dev"]["safe_recall"] == 1.0
    assert metrics["evaluations"]["frozen_test"]["pairwise_accuracy"] == 1.0
    assert metrics["validated_for_product"] is True


def test_answer_verifier_checkpoint_is_portable_and_sklearn_free(tmp_path: Path) -> None:
    verifier, metrics = _train(tmp_path)
    checkpoint = tmp_path / "answer_verifier.pkl"
    payload = pickle.loads(checkpoint.read_bytes())
    assert payload["model_format"] == PORTABLE_MODEL_FORMAT
    assert "model" not in payload
    assert len(payload["model_parameters"]["coef"]) == len(payload["feature_names"])
    assert verifier.metadata()["model_format"] == PORTABLE_MODEL_FORMAT
    assert metrics["model_format"] == PORTABLE_MODEL_FORMAT


def test_calibrated_verifier_ranks_supported_answer_above_hallucination(tmp_path: Path) -> None:
    verifier, _ = _train(tmp_path)
    records = load_preference_records(str(PREFERENCE_PATH))
    record = next(item for item in records if item.negative_type == "hallucinated_detail")
    evidence = [dict(item) for item in record.evidence]
    grounding = {"sufficient": True}
    chosen = verifier.assess(record.query, record.chosen, evidence, grounding)
    rejected = verifier.assess(record.query, record.rejected, evidence, grounding)
    assert chosen["passed"] is True
    assert chosen["safe_probability"] > rejected["safe_probability"]


class _AlwaysReject:
    def assess(self, *args, **kwargs):
        return {"enabled": True, "passed": False, "safe_probability": 0.01, "threshold": 0.5}


def test_pipeline_calibrated_verifier_can_only_veto() -> None:
    evidence = [{"start_sec": 0.0, "end_sec": 10.0, "text": "男子展示可乐并介绍产品。"}]
    answer = "问题：发生了什么？\n结论：男子展示可乐。(timestamp=0.0-10.0)"
    result = VideoMemoPipeline._verify_payload(
        answer,
        ["timestamp=0.0-10.0"],
        {"sufficient": True},
        evidence_items=evidence,
        query="发生了什么？",
        calibrated_verifier=_AlwaysReject(),
    )
    assert result["ok"] is False
    assert result["calibrated_verifier_ok"] is False
    assert "calibrated verifier rejected" in result["reason"]


def test_pipeline_never_lets_calibrated_model_override_hard_failure(tmp_path: Path) -> None:
    verifier, _ = _train(tmp_path)
    evidence = [{"start_sec": 0.0, "end_sec": 10.0, "text": "男子展示可乐并介绍产品。"}]
    result = VideoMemoPipeline._verify_payload(
        "问题：发生了什么？\n结论：男子展示可乐。(timestamp=20.0-30.0)",
        ["timestamp=0.0-10.0"],
        {"sufficient": True},
        evidence_items=evidence,
        query="发生了什么？",
        calibrated_verifier=verifier,
    )
    assert result["ok"] is False
    assert result["unmatched_timestamp_refs"] == ["20.0-30.0"]
    assert result["calibrated_verifier"]["enabled"] is False
