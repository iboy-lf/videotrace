from __future__ import annotations

"""Data and training contract for the calibrated answer verifier."""

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..eval.reproducibility import file_sha256, source_fingerprint
from ..verifier.calibrated import (
    CHECKPOINT_SCHEMA_VERSION,
    FEATURE_CONTRACT_VERSION,
    FEATURE_NAMES,
    PORTABLE_MODEL_FORMAT,
    extract_answer_verifier_features,
    vectorize_answer_verifier_features,
)
from .preference_data import PreferenceRecord, load_preference_records, validate_preference_records


DATASET_SCHEMA_VERSION = "videotrace-answer-verifier-dataset-v1"


@dataclass
class AnswerVerifierTrainResult:
    checkpoint_path: str
    metrics_path: str
    model_card_path: str
    num_rows: int
    train_rows: int
    dev_rows: int
    frozen_test_rows: int
    threshold: float
    dev_accuracy: float
    frozen_test_accuracy: float

    def dump(self) -> dict:
        return self.__dict__


def build_answer_verifier_rows(records: Iterable[PreferenceRecord]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        evidence = [dict(item) for item in record.evidence]
        grounding = {"sufficient": bool(evidence)}
        for variant, label, answer, negative_type in (
            ("chosen", 1, record.chosen, ""),
            ("rejected", 0, record.rejected, record.negative_type),
        ):
            report_features = extract_answer_verifier_features(
                record.query,
                answer,
                evidence,
                grounding,
            )
            rows.append(
                {
                    "schema_version": DATASET_SCHEMA_VERSION,
                    "example_id": f"{record.pair_id}:{variant}",
                    "pair_id": record.pair_id,
                    "source_record_id": record.source_record_id,
                    "split": record.split,
                    "frozen_test": bool(record.frozen_test),
                    "variant": variant,
                    "label": int(label),
                    "negative_type": negative_type,
                    "query": record.query,
                    "answer": answer,
                    "evidence": evidence,
                    "grounding_decision": grounding,
                    "features": report_features,
                }
            )
    return rows


def validate_answer_verifier_rows(rows: list[dict]) -> dict:
    errors: list[str] = []
    by_pair: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("schema_version") != DATASET_SCHEMA_VERSION:
            errors.append(f"unsupported schema: {row.get('example_id')}")
        if row.get("variant") not in {"chosen", "rejected"}:
            errors.append(f"invalid variant: {row.get('example_id')}")
        if row.get("label") not in {0, 1}:
            errors.append(f"invalid label: {row.get('example_id')}")
        if set(row.get("features", {}).keys()) != set(FEATURE_NAMES):
            errors.append(f"feature contract mismatch: {row.get('example_id')}")
        if row.get("frozen_test") != (str(row.get("split")) == "test"):
            errors.append(f"frozen test flag mismatch: {row.get('example_id')}")
        by_pair.setdefault(str(row.get("pair_id")), []).append(row)

    for pair_id, pair_rows in by_pair.items():
        variants = {str(row.get("variant")) for row in pair_rows}
        splits = {str(row.get("split")) for row in pair_rows}
        if variants != {"chosen", "rejected"}:
            errors.append(f"pair must contain chosen and rejected: {pair_id}")
        if len(splits) != 1:
            errors.append(f"pair crosses splits: {pair_id}")

    counts = {
        split: sum(str(row.get("split")) == split for row in rows)
        for split in ("train", "dev", "test")
    }
    return {
        "valid": not errors,
        "errors": errors,
        "num_rows": len(rows),
        "num_pairs": len(by_pair),
        "counts": counts,
        "frozen_test_rows": sum(bool(row.get("frozen_test")) for row in rows),
    }


def write_answer_verifier_dataset(
    preference_path: str | Path,
    dataset_path: str | Path,
    summary_path: str | Path,
    project_root: str | Path,
) -> dict:
    root = Path(project_root).resolve()
    records = load_preference_records(str(preference_path))
    preference_report = validate_preference_records(records, project_root=root)
    if not preference_report["valid"]:
        raise ValueError(f"preference dataset is invalid: {preference_report['errors']}")
    rows = build_answer_verifier_rows(records)
    validation = validate_answer_verifier_rows(rows)
    if not validation["valid"]:
        raise ValueError(f"answer verifier dataset is invalid: {validation['errors']}")

    dataset_target = Path(dataset_path)
    summary_target = Path(summary_path)
    dataset_target.parent.mkdir(parents=True, exist_ok=True)
    summary_target.parent.mkdir(parents=True, exist_ok=True)
    # Pin LF bytes explicitly so the dataset SHA is identical on Windows and
    # Linux.  This is part of the reproducibility contract, not presentation
    # formatting: Path.write_text() otherwise applies the host newline policy.
    with dataset_target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n")
    train_rows = [row for row in rows if row["split"] == "train"]
    gradient_hash = _gradient_payload_sha256(train_rows)
    summary = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_path": _display_path(root, dataset_target),
        "dataset_sha256": file_sha256(dataset_target),
        "preference_dataset_sha256": file_sha256(Path(preference_path)),
        "source_sha256": source_fingerprint(root),
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "feature_names": FEATURE_NAMES,
        "gradient_payload_sha256": gradient_hash,
        "counts": validation["counts"],
        "num_pairs": validation["num_pairs"],
        "frozen_test_rows": validation["frozen_test_rows"],
        "frozen_test_excluded_from_gradient": True,
        "validation": validation,
        "provenance": "Derived from manually authored grounded DPO pairs; frozen cola pair is evaluation-only.",
    }
    summary_target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def train_answer_verifier(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    metrics_path: str | Path,
    model_card_path: str | Path,
    project_root: str | Path,
) -> AnswerVerifierTrainResult:
    root = Path(project_root).resolve()
    rows = _load_rows(Path(dataset_path))
    validation = validate_answer_verifier_rows(rows)
    if not validation["valid"]:
        raise ValueError(f"answer verifier dataset is invalid: {validation['errors']}")
    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if any(row.get("frozen_test") for row in train_rows):
        raise ValueError("frozen test rows must not enter verifier training")
    if {row["label"] for row in train_rows} != {0, 1}:
        raise ValueError("verifier training needs both labels")

    x_train, y_train = _matrix(train_rows)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    dev_prob = _probabilities(model, dev_rows)
    threshold, threshold_report = _select_threshold(dev_rows, dev_prob)
    dev_metrics = _evaluate(dev_rows, dev_prob, threshold)
    test_prob = _probabilities(model, test_rows)
    test_metrics = _evaluate(test_rows, test_prob, threshold)
    train_metrics = _evaluate(train_rows, _probabilities(model, train_rows), threshold)

    checkpoint_target = Path(checkpoint_path)
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    scaler = model.named_steps["scale"]
    classifier = model.named_steps["classifier"]
    classes = [int(value) for value in classifier.classes_.tolist()]
    if classes != [0, 1] or classifier.coef_.shape != (1, len(FEATURE_NAMES)):
        raise ValueError(
            "answer verifier export requires binary LogisticRegression with one coefficient row"
        )
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "feature_names": FEATURE_NAMES,
        "model_format": PORTABLE_MODEL_FORMAT,
        "threshold": threshold,
        "model_parameters": {
            "mean": [float(value) for value in scaler.mean_.tolist()],
            "scale": [float(value) for value in scaler.scale_.tolist()],
            "coef": [float(value) for value in classifier.coef_[0].tolist()],
            "intercept": float(classifier.intercept_[0]),
            "classes": classes,
        },
        "training": {
            "source_sha256": source_fingerprint(root),
            "dataset_sha256": file_sha256(Path(dataset_path)),
            "gradient_payload_sha256": _gradient_payload_sha256(train_rows),
            "train_example_ids": [row["example_id"] for row in train_rows],
            "dev_example_ids": [row["example_id"] for row in dev_rows],
            "frozen_test_example_ids": [row["example_id"] for row in test_rows],
            "threshold_selection": threshold_report,
            "train": train_metrics,
            "dev": dev_metrics,
            "frozen_test": test_metrics,
            "policy": "hard deterministic failures remain failures; model only vetoes a deterministic pass or certifies a safe abstention",
        },
    }
    checkpoint_target.write_bytes(pickle.dumps(payload, protocol=4))
    checkpoint_sha = file_sha256(checkpoint_target)
    metrics = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": "completed",
        "source_sha256": source_fingerprint(root),
        "dataset_sha256": file_sha256(Path(dataset_path)),
        "gradient_payload_sha256": _gradient_payload_sha256(train_rows),
        "checkpoint_sha256": checkpoint_sha,
        "model_format": PORTABLE_MODEL_FORMAT,
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "feature_names": FEATURE_NAMES,
        "threshold": threshold,
        "counts": validation["counts"],
        "frozen_test_excluded_from_gradient": True,
        "evaluations": {"train": train_metrics, "dev": dev_metrics, "frozen_test": test_metrics},
        "threshold_selection": threshold_report,
        "validated_for_product": bool(dev_metrics["safe_recall"] >= 1.0 and test_metrics["pairwise_accuracy"] >= 0.5),
    }
    metrics_target = Path(metrics_path)
    metrics_target.parent.mkdir(parents=True, exist_ok=True)
    metrics_target.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model_card = {
        "schema_version": "videotrace-answer-verifier-model-card-v1",
        "model_type": "scikit-learn LogisticRegression with StandardScaler",
        "runtime_model_format": PORTABLE_MODEL_FORMAT,
        "training_runtime": _training_runtime(),
        "intended_use": "Conservative post-generation safety veto for timestamp-grounded answers.",
        "not_intended_for": "Replacing hard timestamp binding, visual understanding or open-ended entailment.",
        "source_sha256": source_fingerprint(root),
        "dataset": {
            "path": _display_path(root, Path(dataset_path)),
            "sha256": file_sha256(Path(dataset_path)),
            "counts": validation["counts"],
            "frozen_test_excluded_from_gradient": True,
            "gradient_payload_sha256": _gradient_payload_sha256(train_rows),
        },
        "checkpoint": {
            "path": _display_path(root, checkpoint_target),
            "sha256": checkpoint_sha,
            "threshold": threshold,
            "feature_contract": FEATURE_CONTRACT_VERSION,
            "feature_names": FEATURE_NAMES,
        },
        "metrics": metrics["evaluations"],
        "limitations": [
            "Small task-local preference set; not a general entailment benchmark.",
            "The deterministic verifier remains the first safety gate.",
            "A model veto is conservative and does not add evidence or repair timestamps.",
        ],
    }
    card_target = Path(model_card_path)
    card_target.parent.mkdir(parents=True, exist_ok=True)
    card_target.write_text(json.dumps(model_card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return AnswerVerifierTrainResult(
        checkpoint_path=str(checkpoint_target),
        metrics_path=str(metrics_target),
        model_card_path=str(card_target),
        num_rows=len(rows),
        train_rows=len(train_rows),
        dev_rows=len(dev_rows),
        frozen_test_rows=len(test_rows),
        threshold=threshold,
        dev_accuracy=float(dev_metrics["accuracy"]),
        frozen_test_accuracy=float(test_metrics["accuracy"]),
    )


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([vectorize_answer_verifier_features(row["features"]) for row in rows]),
        np.asarray([int(row["label"]) for row in rows], dtype="int64"),
    )


def _probabilities(model, rows: list[dict]) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype="float32")
    return model.predict_proba(np.stack([vectorize_answer_verifier_features(row["features"]) for row in rows]))[:, 1]


def _select_threshold(rows: list[dict], probabilities: np.ndarray) -> tuple[float, dict]:
    if not rows:
        return 0.5, {"source": "default_no_dev", "threshold": 0.5}
    candidates = sorted({0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, *[float(v) for v in probabilities]})
    scored = []
    labels = np.asarray([int(row["label"]) for row in rows])
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        safe_recall = _recall(labels, predictions, 1)
        unsafe_recall = _recall(labels, predictions, 0)
        scored.append((safe_recall >= 1.0, unsafe_recall, safe_recall, -threshold, threshold))
    best = max(scored)
    threshold = float(best[-1])
    return threshold, {
        "source": "dev_safe_recall_then_unsafe_recall",
        "threshold": threshold,
        "safe_recall": best[2],
        "unsafe_recall": best[1],
    }


def _evaluate(rows: list[dict], probabilities: np.ndarray, threshold: float) -> dict:
    if not rows:
        return {
            "rows": 0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "f1": 0.0,
            "roc_auc": None,
            "safe_recall": 0.0,
            "unsafe_recall": 0.0,
            "pairwise_accuracy": 0.0,
        }
    labels = np.asarray([int(row["label"]) for row in rows])
    predictions = (probabilities >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        auc = None
    return {
        "rows": len(rows),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": auc,
        "safe_recall": _recall(labels, predictions, 1),
        "unsafe_recall": _recall(labels, predictions, 0),
        "pairwise_accuracy": _pairwise_accuracy(rows, probabilities),
    }


def _pairwise_accuracy(rows: list[dict], probabilities: np.ndarray) -> float:
    by_pair: dict[str, list[tuple[int, float]]] = {}
    for row, probability in zip(rows, probabilities):
        by_pair.setdefault(str(row["pair_id"]), []).append((int(row["label"]), float(probability)))
    comparisons = []
    for values in by_pair.values():
        chosen = [probability for label, probability in values if label == 1]
        rejected = [probability for label, probability in values if label == 0]
        comparisons.extend(pos > neg for pos in chosen for neg in rejected)
    return float(sum(comparisons) / len(comparisons)) if comparisons else 0.0


def _recall(labels: np.ndarray, predictions: np.ndarray, target: int) -> float:
    mask = labels == target
    return float(np.sum(predictions[mask] == target) / max(1, np.sum(mask)))


def _gradient_payload_sha256(rows: list[dict]) -> str:
    payload = [
        {
            "example_id": row["example_id"],
            "label": row["label"],
            "features": row["features"],
        }
        for row in sorted(rows, key=lambda item: str(item["example_id"]))
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _training_runtime() -> dict:
    versions: dict[str, str] = {}
    for package in ("numpy", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {"python": sys.version.split()[0], "packages": versions}
