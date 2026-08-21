import json

import pytest

import numpy as np

from videomemo.reranker.neural import (
    _load_rows,
    _mixed_label_groups,
    _select_blend_weight,
    _split_rows_by_group,
)


def test_reranker_dataset_rows_preserve_split_contract(tmp_path):
    path = tmp_path / "rows.jsonl"
    row = {"group_id": "g", "label": 1.0, "split": "dev", "features": {}}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert _load_rows(str(path))[0]["split"] == "dev"


def test_test_split_guard_is_declared_in_training_api():
    import inspect

    from videomemo.reranker import train_reranker

    signature = inspect.signature(train_reranker)
    assert signature.parameters["allow_test_split"].default is False


def test_group_split_prefers_pairwise_eval_and_retains_pairwise_train_group():
    rows = []
    for group_id in ("a", "b", "c"):
        rows.extend(
            [
                {"group_id": group_id, "label": 1.0},
                {"group_id": group_id, "label": 0.0},
            ]
        )
    rows.extend(
        [
            {"group_id": "negative-only", "label": 0.0},
            {"group_id": "positive-only", "label": 1.0},
        ]
    )
    train_rows, eval_rows = _split_rows_by_group(rows, eval_fraction=0.4, seed=42)
    assert len(_mixed_label_groups(eval_rows)) == 2
    assert len(_mixed_label_groups(train_rows)) == 1


def test_blend_weight_selection_prefers_smallest_weight_at_best_accuracy():
    rows = [
        {"group_id": "g", "label": 1.0},
        {"group_id": "g", "label": 0.0},
    ]
    weight, accuracy = _select_blend_weight(
        np.asarray([0.4, 0.6], dtype="float32"),
        np.asarray([0.9, 0.1], dtype="float32"),
        rows,
    )
    assert accuracy == 1.0
    assert weight == 0.25
