from __future__ import annotations

import json
import re
from pathlib import Path

from ..reranker.neural import FEATURE_NAMES
from .reproducibility import file_sha256, source_fingerprint


TIMESTAMP_RE = re.compile(r"timestamp=([0-9]+(?:\.[0-9]+)?-[0-9]+(?:\.[0-9]+)?)")


def validate_interview_package(
    root: Path,
    knowledge_pack_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    dataset_path: Path,
    dataset_summary_path: Path,
    model_card_path: Path,
) -> dict:
    root = root.resolve()
    paths = {
        "knowledge_pack": knowledge_pack_path.resolve(),
        "checkpoint": checkpoint_path.resolve(),
        "metrics": metrics_path.resolve(),
        "dataset": dataset_path.resolve(),
        "dataset_summary": dataset_summary_path.resolve(),
        "model_card": model_card_path.resolve(),
    }
    checks: list[dict] = []

    def add(name: str, passed: bool, evidence: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    missing = [name for name, path in paths.items() if not path.exists()]
    add("required_artifacts_exist", not missing, {"missing": missing})
    if missing:
        return _report(checks, paths)

    pack = _load_json(paths["knowledge_pack"])
    metrics = _load_json(paths["metrics"])
    summary = _load_json(paths["dataset_summary"])
    model_card = _load_json(paths["model_card"])
    rows = [
        json.loads(line)
        for line in paths["dataset"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metadata = pack.get("metadata", {})
    video_path = Path(str(pack.get("video_path", "")))
    if not video_path.is_absolute():
        video_path = root / video_path
    video_path = video_path.resolve()
    add("input_video_exists", video_path.exists(), str(video_path))

    expected_source_sha = source_fingerprint(root)
    add(
        "source_fingerprint_matches",
        metadata.get("source_sha256") == expected_source_sha,
        {
            "artifact": metadata.get("source_sha256", ""),
            "current": expected_source_sha,
        },
    )
    actual_video_sha = file_sha256(video_path) if video_path.exists() else ""
    add(
        "video_fingerprint_matches",
        bool(actual_video_sha) and metadata.get("video_sha256") == actual_video_sha,
        {
            "artifact": metadata.get("video_sha256", ""),
            "current": actual_video_sha,
        },
    )

    stack = {
        "segment_understanding": metadata.get("segment_understanding", {}).get("backend"),
        "vlm": metadata.get("vlm", {}).get("backend"),
        "reranker": metadata.get("reranker", {}).get("backend"),
        "llm": metadata.get("llm_backend"),
    }
    add(
        "interview_stack_is_active",
        stack == {
            "segment_understanding": "qwen35_local",
            "vlm": "frozen_siglip",
            "reranker": "neural",
            "llm": "qwen35_local",
        },
        stack,
    )
    environment = metadata.get("environment", {})
    packages = environment.get("packages", {})
    add(
        "runtime_environment_is_recorded",
        bool(environment.get("python"))
        and bool(environment.get("platform"))
        and bool(packages.get("torch"))
        and bool(packages.get("transformers"))
        and bool(environment.get("physical_gpu_ids")),
        environment,
    )

    agent_run = metadata.get("agent_run", {})
    coverage = float(agent_run.get("verification", {}).get("coverage", 0.0))
    add(
        "agent_answer_is_grounded",
        bool(agent_run.get("verified")) and coverage >= 0.999,
        {
            "verified": bool(agent_run.get("verified")),
            "coverage": coverage,
            "tool_calls": len(agent_run.get("tool_trace", [])),
        },
    )
    evidence_tags = set(agent_run.get("context", {}).get("evidence_tags", []))
    timestamp_refs = TIMESTAMP_RE.findall(str(pack.get("answer", "")))
    unbound = [ref for ref in timestamp_refs if f"timestamp={ref}" not in evidence_tags]
    add(
        "answer_timestamps_are_bound",
        len(timestamp_refs) >= 3 and not unbound,
        {"timestamp_refs": timestamp_refs, "unbound": unbound},
    )

    timeline = list(pack.get("timeline", []))
    duration = float(pack.get("duration_sec", 0.0))
    reasons = {str(item.get("selection_reason", "")) for item in timeline}
    opening = any(
        float(item.get("start_sec", duration)) <= 1.0
        and item.get("selection_reason") == "temporal_coverage:opening"
        for item in timeline
    )
    ending = any(
        duration - float(item.get("end_sec", 0.0)) <= 1.0
        and item.get("selection_reason") == "temporal_coverage:ending"
        for item in timeline
    )
    middle = "temporal_coverage:middle" in reasons
    add(
        "overview_temporal_coverage_is_complete",
        len(timeline) >= 3 and opening and middle and ending,
        {
            "selected": [
                [item.get("start_sec"), item.get("end_sec"), item.get("selection_reason")]
                for item in timeline
            ]
        },
    )

    dense_index = metadata.get("vlm", {}).get("persistent_index", {})
    add(
        "dense_index_is_persisted",
        bool(dense_index.get("enabled"))
        and int(dense_index.get("num_vectors", 0)) == len(pack.get("segments", [])),
        dense_index,
    )
    clip_paths = []
    source_window_flags = []
    for clip in pack.get("clips", []):
        path = Path(str(clip.get("file", "")))
        if not path.is_absolute():
            path = root / path
        clip_paths.append(path.resolve())
        source_window_flags.append(clip.get("playback_mode") == "source_video_window")
    demo_path = paths["knowledge_pack"].parent / "demo.html"
    add(
        "replayable_media_exists",
        bool(clip_paths)
        and demo_path.exists()
        and all(path.exists() for path in clip_paths)
        and all(source_window_flags)
        and all(path == video_path for path in clip_paths),
        {
            "demo": str(demo_path),
            "clips": [str(path) for path in clip_paths],
            "playback_mode": "source_video_window",
        },
    )

    splits = {str(row.get("split", "")).lower() for row in rows}
    final_video_id = _normalize_id(video_path.stem)
    training_video_ids = {_normalize_id(str(row.get("video_id", ""))) for row in rows}
    add(
        "training_split_has_no_test_leakage",
        splits == {"dev"}
        and final_video_id not in training_video_ids
        and not bool(model_card.get("data_contract", {}).get("contains_test_rows")),
        {
            "splits": sorted(splits),
            "training_video_ids": sorted(training_video_ids),
            "final_video_id": final_video_id,
        },
    )
    feature_contract = all(
        list(row.get("features", {}).keys()) == FEATURE_NAMES for row in rows
    )
    expected_hidden_dim = int(metrics.get("hidden_dim", 0))
    card_hidden_dims = model_card.get("architecture", {}).get("hidden_dims", [])
    add(
        "reranker_feature_contract_matches",
        feature_contract
        and model_card.get("architecture", {}).get("feature_names") == FEATURE_NAMES
        and bool(model_card.get("architecture", {}).get("feature_contract_valid"))
        and bool(card_hidden_dims)
        and int(card_hidden_dims[0]) == expected_hidden_dim,
        {
            "feature_names": FEATURE_NAMES,
            "metrics_hidden_dim": expected_hidden_dim,
            "model_card_hidden_dims": card_hidden_dims,
        },
    )

    row_count = len(rows)
    group_count = len({str(row.get("group_id", "")) for row in rows})
    positive_count = sum(float(row.get("label", 0.0)) > 0.5 for row in rows)
    counts_match = (
        int(metrics.get("num_rows", -1)) == row_count
        and int(metrics.get("num_groups", -1)) == group_count
        and int(metrics.get("num_positive", -1)) == positive_count
        and int(summary.get("num_rows", -1)) == row_count
        and int(summary.get("num_groups", -1)) == group_count
        and int(summary.get("num_positive", -1)) == positive_count
    )
    add(
        "training_counts_are_consistent",
        counts_match,
        {"rows": row_count, "groups": group_count, "positive": positive_count},
    )
    neural_accuracy = _optional_float(metrics.get("pairwise_accuracy"))
    blended_accuracy = _optional_float(metrics.get("blended_pairwise_accuracy"))
    base_accuracy = _optional_float(
        model_card.get("evaluation", {}).get("base_fusion_pairwise_accuracy")
    )
    add(
        "reranker_has_held_out_pairwise_signal",
        neural_accuracy is not None
        and blended_accuracy is not None
        and base_accuracy is not None
        and int(metrics.get("num_pairwise_eval_groups", 0)) >= 2
        and blended_accuracy >= base_accuracy,
        {
            "neural_pairwise_accuracy": neural_accuracy,
            "blended_pairwise_accuracy": blended_accuracy,
            "base_fusion_pairwise_accuracy": base_accuracy,
            "recommended_blend_weight": metrics.get("recommended_blend_weight"),
            "pairwise_eval_groups": metrics.get("num_pairwise_eval_groups"),
        },
    )
    card_artifacts = model_card.get("artifacts", {})
    add(
        "training_artifact_hashes_match",
        card_artifacts.get("checkpoint_sha256") == file_sha256(paths["checkpoint"])
        and card_artifacts.get("dataset_sha256") == file_sha256(paths["dataset"])
        and card_artifacts.get("metrics_sha256") == file_sha256(paths["metrics"]),
        {
            "checkpoint_sha256": file_sha256(paths["checkpoint"]),
            "dataset_sha256": file_sha256(paths["dataset"]),
            "metrics_sha256": file_sha256(paths["metrics"]),
        },
    )
    add(
        "model_card_matches_source_snapshot",
        model_card.get("source_sha256") == expected_source_sha,
        {
            "model_card": model_card.get("source_sha256", ""),
            "current": expected_source_sha,
        },
    )
    return _report(checks, paths)


def _report(checks: list[dict], paths: dict[str, Path]) -> dict:
    failures = [check["name"] for check in checks if not check["passed"]]
    return {
        "valid": not failures,
        "checks_passed": len(checks) - len(failures),
        "checks_total": len(checks),
        "failures": failures,
        "checks": checks,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
