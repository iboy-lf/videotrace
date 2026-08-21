from __future__ import annotations

import json

import pytest

from scripts import select_best_qwen35_adapter as selector


def test_selected_adapter_publishes_comparison_baseline_and_candidate(tmp_path):
    evaluation = {
        "status": "completed",
        "baseline": {"variant": "baseline", "source_sha256": "source"},
        "adapter": {"variant": "adapter", "candidate_id": "qwen35_dpo"},
    }
    source = tmp_path / "dpo_eval.json"
    source.write_text(json.dumps(evaluation), encoding="utf-8")
    comparison = tmp_path / "selected.json"
    baseline = tmp_path / "selected_baseline.json"
    candidate = tmp_path / "selected_adapter.json"

    selector._publish_selected_reports(source, evaluation, comparison, baseline, candidate)

    assert comparison.read_bytes() == source.read_bytes()
    assert json.loads(baseline.read_text(encoding="utf-8")) == evaluation["baseline"]
    assert json.loads(candidate.read_text(encoding="utf-8")) == evaluation["adapter"]


def test_selected_adapter_rejects_incomplete_evaluation(tmp_path):
    source = tmp_path / "invalid.json"
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="baseline and adapter"):
        selector._publish_selected_reports(
            source,
            {},
            tmp_path / "comparison.json",
            tmp_path / "baseline.json",
            tmp_path / "adapter.json",
        )


def test_candidate_provenance_keeps_training_source_and_updates_admission_source(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    card_path = tmp_path / "card.json"
    metrics_path.write_text(json.dumps({"source_sha256": "training"}), encoding="utf-8")
    card_path.write_text(
        json.dumps({"reproducibility": {"source_sha256": "training"}}),
        encoding="utf-8",
    )

    selector._bind_candidate_provenance(
        {"metrics": metrics_path, "model_card": card_path},
        "product",
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert metrics["training_source_sha256"] == "training"
    assert metrics["source_sha256"] == "product"
    assert metrics["admission_source_sha256"] == "product"
    assert card["training_source_sha256"] == "training"
    assert card["reproducibility"]["source_sha256"] == "training"
    assert card["reproducibility"]["admission_source_sha256"] == "product"
