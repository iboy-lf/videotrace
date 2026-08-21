from __future__ import annotations

"""A small calibrated safety layer for evidence-grounded answers.

The deterministic verifier remains authoritative for timestamp and claim
binding.  This model consumes only auditable scalar features and may veto an
otherwise valid answer; it never repairs an invalid timestamp or invents
missing evidence.
"""

import hashlib
import math
from pathlib import Path
import pickle
from typing import Mapping, Sequence

import numpy as np

from ..utils.text import informative_keyword_set, keyword_set
from .simple_verifier import inspect_answer_grounding


CHECKPOINT_SCHEMA_VERSION = "videotrace-calibrated-answer-verifier-v2"
FEATURE_CONTRACT_VERSION = "answer-verifier-features-v1"
PORTABLE_MODEL_FORMAT = "portable-numpy-logistic-v1"
FEATURE_NAMES = [
    "has_refusal",
    "grounding_sufficient",
    "evidence_count_scaled",
    "answer_length_log",
    "timestamp_count_scaled",
    "matched_timestamp_ratio",
    "unmatched_timestamp_ratio",
    "claim_support_coverage",
    "unsupported_claim_ratio",
    "answer_evidence_coverage",
    "query_answer_coverage",
    "decision_alignment",
    "unsupported_overclaim",
    "deterministic_ok",
]

REFUSAL_MARKERS = (
    "证据不足",
    "无法确认",
    "不能确认",
    "无法回答",
    "不足以支持",
    "未提供足够证据",
)


def is_refusal(answer: str) -> bool:
    return any(marker in str(answer or "") for marker in REFUSAL_MARKERS)


def extract_answer_verifier_features(
    query: str,
    answer: str,
    evidence_items: Sequence[dict] | None,
    grounding_decision: Mapping[str, object] | None,
    deterministic_report: Mapping[str, object] | None = None,
) -> dict[str, float]:
    evidence = [dict(item) for item in (evidence_items or [])]
    decision = dict(grounding_decision or {})
    report = dict(
        deterministic_report
        or inspect_answer_grounding(answer, evidence, evidence_items=evidence)
    )
    refusal = is_refusal(answer)
    sufficient = bool(decision.get("sufficient"))
    timestamp_refs = list(report.get("timestamp_refs") or [])
    matched_refs = list(report.get("matched_timestamp_refs") or [])
    unmatched_refs = list(report.get("unmatched_timestamp_refs") or [])
    claim_checks = list(report.get("claim_checks") or [])
    unsupported = list(report.get("unsupported_claims") or [])
    answer_terms = informative_keyword_set(str(answer or ""))
    query_terms = informative_keyword_set(str(query or ""))
    evidence_terms: set[str] = set()
    for item in evidence:
        evidence_terms.update(keyword_set(_evidence_text(item)))

    raw = {
        "has_refusal": float(refusal),
        "grounding_sufficient": float(sufficient),
        "evidence_count_scaled": min(1.0, len(evidence) / 5.0),
        "answer_length_log": min(1.0, math.log1p(len(str(answer or ""))) / 8.0),
        "timestamp_count_scaled": min(1.0, len(timestamp_refs) / 5.0),
        "matched_timestamp_ratio": len(matched_refs) / max(1, len(timestamp_refs)),
        "unmatched_timestamp_ratio": len(unmatched_refs) / max(1, len(timestamp_refs)),
        "claim_support_coverage": float(report.get("claim_support_coverage", 0.0)),
        "unsupported_claim_ratio": len(unsupported) / max(1, len(claim_checks)),
        "answer_evidence_coverage": len(answer_terms & evidence_terms) / max(1, len(answer_terms)),
        "query_answer_coverage": len(query_terms & answer_terms) / max(1, len(query_terms)),
        "decision_alignment": float(refusal != sufficient),
        "unsupported_overclaim": float(not sufficient and not refusal),
        "deterministic_ok": float(bool(report.get("ok"))),
    }
    # Keep the training artifact byte-stable across supported Python/libm
    # builds.  The raw values occasionally differ beyond the 12th decimal on
    # Windows and Linux even though the feature is semantically identical.
    return {name: round(float(raw[name]), 12) for name in FEATURE_NAMES}


def vectorize_answer_verifier_features(features: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(features.get(name, 0.0)) for name in FEATURE_NAMES], dtype="float32")


class CalibratedAnswerVerifier:
    def __init__(self, checkpoint_path: str, threshold: float | None = None):
        self.checkpoint_path = str(Path(checkpoint_path))
        payload = pickle.loads(Path(checkpoint_path).read_bytes())
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("answer verifier checkpoint schema is unsupported")
        checkpoint_features = list(payload.get("feature_names") or [])
        if checkpoint_features != FEATURE_NAMES:
            raise ValueError("answer verifier checkpoint feature contract does not match runtime")
        self.model_format = str(payload.get("model_format") or "legacy-sklearn-pipeline")
        self.model_parameters = dict(payload.get("model_parameters") or {})
        self.model = payload.get("model")
        if self.model_format == PORTABLE_MODEL_FORMAT:
            _validate_portable_parameters(self.model_parameters)
        elif self.model is None:
            raise ValueError("answer verifier checkpoint has no executable model")
        checkpoint_threshold = float(payload.get("threshold", 0.5))
        self.threshold = checkpoint_threshold if threshold is None or threshold < 0 else float(threshold)
        self.training = dict(payload.get("training") or {})
        self.checkpoint_sha256 = _file_sha256(Path(checkpoint_path))

    def assess(
        self,
        query: str,
        answer: str,
        evidence_items: Sequence[dict] | None,
        grounding_decision: Mapping[str, object] | None,
        deterministic_report: Mapping[str, object] | None = None,
    ) -> dict:
        features = extract_answer_verifier_features(
            query,
            answer,
            evidence_items,
            grounding_decision,
            deterministic_report,
        )
        vector = vectorize_answer_verifier_features(features).reshape(1, -1)
        if self.model_format == PORTABLE_MODEL_FORMAT:
            probability = _portable_safe_probability(vector[0], self.model_parameters)
        else:
            probabilities = self.model.predict_proba(vector)[0]
            classes = list(getattr(self.model, "classes_", [0, 1]))
            positive_index = classes.index(1) if 1 in classes else len(probabilities) - 1
            probability = float(probabilities[positive_index])
        return {
            "enabled": True,
            "passed": probability >= self.threshold,
            "safe_probability": round(probability, 6),
            "threshold": round(self.threshold, 6),
            "feature_contract": FEATURE_CONTRACT_VERSION,
            "features": {name: round(float(features[name]), 6) for name in FEATURE_NAMES},
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    def metadata(self) -> dict:
        return {
            "backend": "calibrated",
            "loaded": True,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "threshold": self.threshold,
            "feature_contract": FEATURE_CONTRACT_VERSION,
            "model_format": self.model_format,
            "training": self.training,
            "policy": "deterministic hard rules first; calibrated model may veto but never override an invalid claim",
        }


def _evidence_text(item: Mapping[str, object]) -> str:
    return " ".join(
        value
        for value in (
            str(item.get("text") or item.get("evidence_text") or item.get("summary") or ""),
            str(item.get("caption") or ""),
            str(item.get("ocr_text") or ""),
            " ".join(str(value) for value in (item.get("entities") or [])),
            " ".join(str(value) for value in (item.get("actions") or [])),
        )
        if value
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_portable_parameters(parameters: Mapping[str, object]) -> None:
    expected = len(FEATURE_NAMES)
    for name in ("mean", "scale", "coef"):
        values = list(parameters.get(name) or [])
        if len(values) != expected:
            raise ValueError(f"portable answer verifier {name} has {len(values)} values; expected {expected}")
    classes = [int(value) for value in list(parameters.get("classes") or [])]
    if classes != [0, 1]:
        raise ValueError(f"portable answer verifier requires binary classes [0, 1], got {classes}")
    float(parameters.get("intercept", 0.0))


def _portable_safe_probability(vector: np.ndarray, parameters: Mapping[str, object]) -> float:
    mean = np.asarray(parameters["mean"], dtype="float64")
    scale = np.asarray(parameters["scale"], dtype="float64")
    coefficient = np.asarray(parameters["coef"], dtype="float64")
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    standardized = (np.asarray(vector, dtype="float64") - mean) / safe_scale
    logit = float(np.dot(standardized, coefficient) + float(parameters["intercept"]))
    if logit >= 0.0:
        return float(1.0 / (1.0 + math.exp(-logit)))
    exp_logit = math.exp(logit)
    return float(exp_logit / (1.0 + exp_logit))
