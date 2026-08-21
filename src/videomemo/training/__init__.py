"""Reproducible, product-facing post-training utilities."""

from .sft_data import (
    SFT_SCHEMA_VERSION,
    build_grounded_sft_dataset,
    gradient_payload_sha256,
    load_sft_records,
    summarize_sft_dataset,
    validate_sft_records,
)
from .preference_data import (
    NEGATIVE_TYPES,
    PREFERENCE_ANNOTATION_SCHEMA_VERSION,
    PREFERENCE_SCHEMA_VERSION,
    PreferenceRecord,
    build_grounded_preference_dataset,
    load_preference_records,
    preference_gradient_payload_sha256,
    summarize_preference_dataset,
    validate_preference_records,
)
from .dpo_objective import dpo_statistics
from .answer_verifier import (
    DATASET_SCHEMA_VERSION as ANSWER_VERIFIER_DATASET_SCHEMA_VERSION,
    build_answer_verifier_rows,
    train_answer_verifier,
    validate_answer_verifier_rows,
    write_answer_verifier_dataset,
)

__all__ = [
    "SFT_SCHEMA_VERSION",
    "build_grounded_sft_dataset",
    "gradient_payload_sha256",
    "load_sft_records",
    "summarize_sft_dataset",
    "validate_sft_records",
    "NEGATIVE_TYPES",
    "PREFERENCE_ANNOTATION_SCHEMA_VERSION",
    "PREFERENCE_SCHEMA_VERSION",
    "PreferenceRecord",
    "build_grounded_preference_dataset",
    "load_preference_records",
    "preference_gradient_payload_sha256",
    "summarize_preference_dataset",
    "validate_preference_records",
    "dpo_statistics",
    "ANSWER_VERIFIER_DATASET_SCHEMA_VERSION",
    "build_answer_verifier_rows",
    "train_answer_verifier",
    "validate_answer_verifier_rows",
    "write_answer_verifier_dataset",
]
