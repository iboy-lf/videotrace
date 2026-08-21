from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
from typing import Iterable


SFT_SCHEMA_VERSION = "videotrace-grounded-sft-v1"
_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


@dataclass(frozen=True)
class SFTRecord:
    record_id: str
    video_id: str
    video_path: str
    split: str
    query: str
    evidence: tuple[dict, ...]
    answer: str
    expected_behavior: str
    provenance: str
    frozen_test: bool = False

    def dump(self) -> dict:
        return {
            "schema_version": SFT_SCHEMA_VERSION,
            "record_id": self.record_id,
            "video_id": self.video_id,
            "video_path": self.video_path,
            "split": self.split,
            "query": self.query,
            "evidence": [dict(item) for item in self.evidence],
            "answer": self.answer,
            "expected_behavior": self.expected_behavior,
            "provenance": self.provenance,
            "frozen_test": self.frozen_test,
        }


def build_grounded_sft_dataset(
    annotations_path: str | Path,
    output_path: str | Path,
    *,
    cola_pack_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict:
    """Build clean, group-isolated evidence-grounded SFT records.

    The manually verified non-cola videos are split by video group: SafeDroid
    is train, Yoga is dev, and the frozen cola knowledge pack is test-only.
    No cola record can enter train/dev; this is checked again by validation.
    """

    annotations_path = Path(annotations_path).resolve()
    root = Path(project_root).resolve() if project_root else annotations_path.parents[1]
    annotation_payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    cases = list(annotation_payload.get("cases", []))
    sources = {str(item.get("video_id")): item for item in annotation_payload.get("sources", [])}
    records: list[SFTRecord] = []

    for case in cases:
        video_id = str(case.get("video_id", "")).strip()
        source = sources.get(video_id, {})
        source_file = str(source.get("file") or Path(str(case.get("video_path", ""))).name)
        split = "train" if video_id == "safedroid-demo" else "dev" if video_id == "yoga-action" else "dev"
        relative_video = _safe_relative_video(source_file, root)
        expected = str(case.get("expected_behavior") or "answer").strip().lower()
        spans = _normalise_spans(case.get("gold_spans", []))
        focus = str(case.get("expected_focus") or "标注内容").strip()
        keywords = [str(item).strip() for item in case.get("gold_keywords", []) if str(item).strip()]
        forbidden = [str(item).strip() for item in case.get("forbidden_keywords", []) if str(item).strip()]
        evidence = tuple(
            {
                "start_sec": start,
                "end_sec": end,
                "text": f"人工核验焦点：{focus}" + (f"；关键词：{'、'.join(keywords)}" if keywords else ""),
                "source": "verified_annotation",
            }
            for start, end in spans
        )
        if expected == "abstain":
            answer = _abstention_answer(str(case.get("query", "")), forbidden)
            evidence = tuple()
        else:
            answer = _grounded_answer(str(case.get("query", "")), evidence)
        records.append(
            SFTRecord(
                record_id=f"{video_id}:{case.get('case_id', len(records))}",
                video_id=video_id,
                video_path=relative_video,
                split=split,
                query=str(case.get("query", "")).strip(),
                evidence=evidence,
                answer=answer,
                expected_behavior=expected,
                provenance="data/supervision/reranker_annotations.json; human-verified development annotation",
            )
        )

    if cola_pack_path:
        records.extend(_cola_test_records(resolve_frozen_cola_pack(cola_pack_path, root), root))

    report = validate_sft_records(records, project_root=root)
    if not report["valid"]:
        raise ValueError("invalid SFT dataset: " + "; ".join(report["errors"]))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.dump(), ensure_ascii=False, sort_keys=True) + "\n")
    summary = summarize_sft_dataset(records, output_path, project_root=root)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def summarize_sft_dataset(
    records: Iterable[SFTRecord],
    dataset_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict:
    rows = list(records)
    path = Path(dataset_path).resolve()
    report = validate_sft_records(rows, project_root=project_root)
    return {
        "schema_version": SFT_SCHEMA_VERSION,
        "dataset_path": str(path),
        "dataset_sha256": _sha256(path),
        "gradient_payload_sha256": gradient_payload_sha256(rows),
        "counts": dict(Counter(record.split for record in rows)),
        "behavior_counts": dict(Counter(record.expected_behavior for record in rows)),
        "video_groups": sorted({record.video_id for record in rows}),
        "cola_video_in_train_or_dev": any(
            "cola" in record.video_id.lower() and record.split in {"train", "dev"}
            for record in rows
        ),
        "validation": report,
        "provenance": "Manually verified evidence annotations plus frozen canonical cola knowledge pack for test-only regression.",
    }


def gradient_payload_sha256(records: Iterable[SFTRecord]) -> str:
    """Hash only train-split fields that can contribute optimizer gradients.

    Artifact provenance may contain a Windows or iboy absolute path.  That
    should remain auditable, but it must not obscure whether the query,
    evidence and answer seen by the optimizer were identical.  Dev and frozen
    test records are deliberately excluded: changing evaluation provenance
    must not change the identity of the gradient-update payload.
    """
    payload = [
        {
            "record_id": record.record_id,
            "query": record.query,
            "evidence": [dict(item) for item in record.evidence],
            "answer": record.answer,
        }
        for record in records
        if record.split == "train"
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_frozen_cola_pack(
    requested: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve the canonical cola pack across local and iboy layouts.

    The code sync intentionally excludes large output directories, so a clean
    checkout on iboy may have only ``outputs/iboy_qwen35/cola_review`` while a
    local checkout keeps the shorter ``outputs/cola_review_qwen35`` path.  The
    resolver accepts the local default as a logical alias, but never replaces a
    caller-supplied non-canonical path silently.
    """
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    default_relative = Path("outputs/cola_review_qwen35/knowledge_pack.json")
    requested_path = Path(requested) if requested is not None else root / default_relative
    if not requested_path.is_absolute():
        requested_path = root / requested_path
    requested_path = requested_path.resolve()
    if requested_path.exists():
        return requested_path

    try:
        requested_relative = requested_path.relative_to(root)
    except ValueError:
        requested_relative = None
    is_default_alias = requested is None or (
        requested_relative is not None
        and requested_relative.as_posix() == default_relative.as_posix()
    )
    if is_default_alias:
        for candidate in (
            root / "outputs" / "cola_review_qwen35" / "knowledge_pack.json",
            root / "outputs" / "iboy_qwen35" / "cola_review" / "knowledge_pack.json",
            root / "outputs" / "runs" / "latest" / "knowledge_pack.json",
        ):
            if candidate.exists():
                return candidate.resolve()
    checked = [str(requested_path)]
    checked.extend(
        str(root / relative)
        for relative in (
            Path("outputs/cola_review_qwen35/knowledge_pack.json"),
            Path("outputs/iboy_qwen35/cola_review/knowledge_pack.json"),
            Path("outputs/runs/latest/knowledge_pack.json"),
        )
    )
    raise FileNotFoundError("frozen cola pack not found; checked: " + ", ".join(dict.fromkeys(checked)))


def load_sft_records(path: str | Path) -> list[SFTRecord]:
    records: list[SFTRecord] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            records.append(
                SFTRecord(
                    record_id=str(payload["record_id"]),
                    video_id=str(payload["video_id"]),
                    video_path=str(payload["video_path"]),
                    split=str(payload["split"]),
                    query=str(payload["query"]),
                    evidence=tuple(dict(item) for item in payload.get("evidence", [])),
                    answer=str(payload["answer"]),
                    expected_behavior=str(payload["expected_behavior"]),
                    provenance=str(payload.get("provenance", "")),
                    frozen_test=bool(payload.get("frozen_test", False)),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid SFT JSONL row {line_number}: {exc}") from exc
    return records


def validate_sft_records(records: Iterable[SFTRecord], project_root: str | Path | None = None) -> dict:
    rows = list(records)
    root = Path(project_root).resolve() if project_root else None
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    split_groups: dict[str, set[str]] = {}
    for index, record in enumerate(rows):
        if record.record_id in seen:
            errors.append(f"duplicate record_id: {record.record_id}")
        seen.add(record.record_id)
        if record.split not in {"train", "dev", "test"}:
            errors.append(f"row {index} has invalid split {record.split!r}")
        if not record.query.strip() or not record.answer.strip():
            errors.append(f"row {index} has empty query or answer")
        if record.expected_behavior not in {"answer", "abstain"}:
            errors.append(f"row {index} has invalid expected_behavior")
        split_groups.setdefault(record.video_id, set()).add(record.split)
        if "cola" in record.video_id.lower() and record.split in {"train", "dev"}:
            errors.append(f"frozen cola video leaked into {record.split}: {record.record_id}")
        if record.expected_behavior == "abstain" and record.evidence:
            errors.append(f"abstention row contains evidence: {record.record_id}")
        for evidence in record.evidence:
            start = _number(evidence.get("start_sec"), -1)
            end = _number(evidence.get("end_sec"), -1)
            if start < 0 or end <= start:
                errors.append(f"invalid evidence bounds in {record.record_id}")
            if not str(evidence.get("text", "")).strip():
                errors.append(f"empty evidence text in {record.record_id}")
        if root is not None:
            video_path = (root / "data" / record.video_path).resolve()
            if not _within(video_path, (root / "data").resolve()):
                errors.append(f"video path escapes data root: {record.record_id}")
            elif not video_path.exists() and record.split != "test":
                warnings.append(f"video not present locally: {record.video_path}")

    # Group isolation: a source video may belong to one split only.
    for video_id, splits in split_groups.items():
        if len(splits) > 1:
            errors.append(f"video group crosses splits: {video_id} -> {sorted(splits)}")
    if not any(record.split == "train" for record in rows):
        errors.append("dataset has no train rows")
    if not any(record.split == "dev" for record in rows):
        errors.append("dataset has no dev rows")
    if not any(record.split == "test" for record in rows):
        errors.append("dataset has no frozen test rows")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "num_records": len(rows),
        "counts": dict(Counter(record.split for record in rows)),
        "video_groups": {video_id: sorted(splits) for video_id, splits in sorted(split_groups.items())},
    }


def _cola_test_records(pack_path: Path, root: Path) -> list[SFTRecord]:
    if not pack_path.exists():
        raise FileNotFoundError(f"frozen cola pack not found: {pack_path}")
    payload = json.loads(pack_path.read_text(encoding="utf-8"))
    video_path = Path(str(payload.get("video_path", "")))
    timeline = list(payload.get("timeline", []))
    if not timeline:
        raise ValueError("cola pack has no timeline for frozen test records")
    query = str(payload.get("metadata", {}).get("query") or "这个视频的整体流程是什么？请给出带时间戳的证据。")
    selected = timeline[: min(3, len(timeline))]
    evidence = tuple(
        {
            "start_sec": _number(item.get("start_sec"), 0.0),
            "end_sec": _number(item.get("end_sec"), 0.0),
            "text": str(item.get("text") or "").strip()[:300],
            "source": "canonical_knowledge_pack",
        }
        for item in selected
        if _number(item.get("end_sec"), 0.0) > _number(item.get("start_sec"), 0.0)
    )
    relative = _safe_relative_video(video_path.name, root)
    return [
        SFTRecord(
            record_id="cola-review:frozen-overview-test",
            video_id="cola-review-frozen-test",
            video_path=relative,
            split="test",
            query=query,
            evidence=evidence,
            answer=_grounded_answer(query, evidence),
            expected_behavior="answer",
            provenance=f"Frozen canonical result: {pack_path.as_posix()}; never used for gradient updates.",
            frozen_test=True,
        )
    ]


def _grounded_answer(query: str, evidence: Iterable[dict]) -> str:
    lines = [f"问题：{query}", "结论：仅根据给定时间证据回答，不补充证据之外的细节。"]
    for item in evidence:
        start = float(item["start_sec"])
        end = float(item["end_sec"])
        text = str(item.get("text") or "证据片段").strip()
        timestamp = f"{start:.1f}-{end:.1f}"
        lines.append(f"- {timestamp}：{text} (timestamp={timestamp})")
    return "\n".join(lines)


def _abstention_answer(query: str, forbidden: Iterable[str]) -> str:
    detail = "、".join(item for item in forbidden if item) or "所问内容"
    return f"问题：{query}\n结论：证据不足，无法确认视频中存在{detail}。"


def _normalise_spans(spans: object) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for item in spans if isinstance(spans, list) else []:
        if not isinstance(item, dict):
            continue
        start = _number(item.get("start_sec"), -1)
        end = _number(item.get("end_sec"), -1)
        if start >= 0 and end > start:
            pair = (round(start, 3), round(end, 3))
            if pair not in result:
                result.append(pair)
    return result


def _safe_relative_video(filename: str, root: Path) -> str:
    name = Path(filename).name
    if Path(name).suffix.lower() not in _VIDEO_SUFFIXES:
        raise ValueError(f"unsupported video file in SFT record: {filename}")
    candidate = root / "data" / "raw" / name
    return candidate.relative_to(root / "data").as_posix()


def _number(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _within(target: Path, root: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
