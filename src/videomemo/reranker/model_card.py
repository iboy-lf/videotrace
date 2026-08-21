from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .neural import FEATURE_NAMES, _load_rows, _mixed_label_groups, _split_rows_by_group


PAIRWISE_BASELINES = (
    "base_fusion_score",
    "query_coverage",
    "retrieval_rank_score",
    "scorer_rank_score",
    "vlm_rank_score",
)


def build_reranker_model_card(
    dataset_path: str,
    checkpoint_path: str,
    metrics_path: str,
    *,
    eval_fraction: float = 0.2,
    seed: int = 42,
    hidden_dim: int | None = None,
    source_sha256: str = "",
) -> dict:
    dataset = Path(dataset_path)
    checkpoint = Path(checkpoint_path)
    metrics_file = Path(metrics_path)
    rows = _load_rows(str(dataset))
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    resolved_hidden_dim = int(metrics.get("hidden_dim", hidden_dim or 32))
    train_rows, eval_rows = _split_rows_by_group(rows, eval_fraction, seed)
    feature_contract_ok = all(
        list(row.get("features", {})) == FEATURE_NAMES
        for row in rows
    )
    baselines = {
        name: _pairwise_signal_accuracy(eval_rows, name)
        for name in PAIRWISE_BASELINES
    }
    neural_accuracy = metrics.get("pairwise_accuracy")
    base_accuracy = metrics.get("base_pairwise_accuracy", baselines["base_fusion_score"])
    blended_accuracy = metrics.get("blended_pairwise_accuracy", neural_accuracy)
    lift = None
    if blended_accuracy is not None and base_accuracy is not None:
        lift = float(blended_accuracy) - float(base_accuracy)

    return {
        "schema_version": "1.0",
        "model_name": "VideoTrace query-segment neural reranker",
        "model_type": "feature_mlp_binary_relevance",
        "architecture": {
            "input_dim": len(FEATURE_NAMES),
            "hidden_dims": [resolved_hidden_dim, max(8, resolved_hidden_dim // 2)],
            "output_dim": 1,
            "feature_names": list(FEATURE_NAMES),
            "feature_contract_valid": feature_contract_ok,
        },
        "artifacts": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": _sha256(checkpoint),
            "dataset_path": str(dataset),
            "dataset_sha256": _sha256(dataset),
            "metrics_path": str(metrics_file),
            "metrics_sha256": _sha256(metrics_file),
        },
        "data_contract": {
            "splits": sorted({str(row.get("split", "")) for row in rows}),
            "video_ids": sorted({str(row.get("video_id", "")) for row in rows}),
            "num_rows": len(rows),
            "num_groups": len({str(row.get("group_id", "")) for row in rows}),
            "num_positive": sum(float(row.get("label", 0.0)) > 0.5 for row in rows),
            "train_groups": sorted({str(row["group_id"]) for row in train_rows}),
            "eval_groups": sorted({str(row["group_id"]) for row in eval_rows}),
            "pairwise_eval_groups": _mixed_label_groups(eval_rows),
            "eval_fraction": float(eval_fraction),
            "seed": int(seed),
            "contains_test_rows": any(
                str(row.get("split", "")).lower() == "test" for row in rows
            ),
        },
        "evaluation": {
            "neural_pairwise_accuracy": neural_accuracy,
            "base_fusion_pairwise_accuracy": base_accuracy,
            "recommended_blend_weight": metrics.get("recommended_blend_weight"),
            "blended_pairwise_accuracy": blended_accuracy,
            "pairwise_eval_pairs": _pairwise_pair_count(eval_rows),
            "pairwise_baselines": baselines,
            "lift_over_base_fusion": lift,
            "train_loss": metrics.get("train_loss"),
            "eval_loss": metrics.get("eval_loss"),
            "best_epoch": metrics.get("best_epoch"),
        },
        "source_sha256": source_sha256,
        "intended_use": "Blend query-segment relevance with sparse, scorer, and SigLIP signals before temporal evidence selection.",
        "limitations": [
            "Dev supervision is small and spans two videos, so metrics are contract evidence rather than a broad quality claim.",
            "The foundation vision-language models remain frozen.",
            "The frozen cola test video is excluded from fitting.",
        ],
    }


def _pairwise_signal_accuracy(rows: list[dict], feature_name: str) -> float | None:
    by_group: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        score = float(row.get("features", {}).get(feature_name, 0.0))
        by_group.setdefault(str(row.get("group_id", "")), []).append(
            (score, float(row.get("label", 0.0)))
        )
    correct = 0
    total = 0
    for items in by_group.values():
        positives = [score for score, label in items if label > 0.5]
        negatives = [score for score, label in items if label <= 0.5]
        for positive in positives:
            for negative in negatives:
                correct += int(positive > negative)
                total += 1
    return correct / total if total else None


def _pairwise_pair_count(rows: list[dict]) -> int:
    by_group: dict[str, list[float]] = {}
    for row in rows:
        by_group.setdefault(str(row.get("group_id", "")), []).append(
            float(row.get("label", 0.0))
        )
    return sum(
        sum(label > 0.5 for label in labels) * sum(label <= 0.5 for label in labels)
        for labels in by_group.values()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
