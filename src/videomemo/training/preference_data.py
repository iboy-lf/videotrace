from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

from .sft_data import SFTRecord, load_sft_records


PREFERENCE_SCHEMA_VERSION = "videotrace-grounded-preference-v1"
PREFERENCE_ANNOTATION_SCHEMA_VERSION = "videotrace-preference-annotations-v1"
NEGATIVE_TYPES = {
    "wrong_timestamp",
    "missing_timestamp",
    "hallucinated_detail",
    "unsupported_overclaim",
}
_TIMESTAMP_PATTERN = re.compile(
    r"(?:timestamp\s*=\s*)?(?P<start>\d+(?:\.\d+)?)\s*[-–—]\s*(?P<end>\d+(?:\.\d+)?)"
)


@dataclass(frozen=True)
class PreferenceRecord:
    pair_id: str
    source_record_id: str
    video_id: str
    video_path: str
    split: str
    query: str
    evidence: tuple[dict, ...]
    chosen: str
    rejected: str
    expected_behavior: str
    negative_type: str
    rationale: str
    provenance: str
    frozen_test: bool = False

    def dump(self) -> dict:
        return {
            "schema_version": PREFERENCE_SCHEMA_VERSION,
            "pair_id": self.pair_id,
            "source_record_id": self.source_record_id,
            "video_id": self.video_id,
            "video_path": self.video_path,
            "split": self.split,
            "query": self.query,
            "evidence": [dict(item) for item in self.evidence],
            "chosen": self.chosen,
            "rejected": self.rejected,
            "expected_behavior": self.expected_behavior,
            "negative_type": self.negative_type,
            "rationale": self.rationale,
            "provenance": self.provenance,
            "frozen_test": self.frozen_test,
        }


def build_grounded_preference_dataset(
    sft_dataset_path: str | Path,
    annotations_path: str | Path,
    output_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict:
    """Join verified SFT positives with explicitly authored preference negatives.

    The annotation file contains only human-authored rejected answers and their
    error taxonomy.  Chosen answers, evidence, splits and the frozen-test flag
    are copied from the validated SFT source so the preference dataset cannot
    silently relabel the cola video or move a video group between splits.
    """

    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    sft_path = _rooted(root, sft_dataset_path)
    annotation_path = _rooted(root, annotations_path)
    output = _rooted(root, output_path)
    sft_records = load_sft_records(sft_path)
    source_by_id = {record.record_id: record for record in sft_records}
    if len(source_by_id) != len(sft_records):
        raise ValueError("SFT source contains duplicate record_id values")

    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    if annotations.get("schema_version") != PREFERENCE_ANNOTATION_SCHEMA_VERSION:
        raise ValueError(
            "unsupported preference annotation schema: "
            + str(annotations.get("schema_version", "missing"))
        )
    annotation_rows = list(annotations.get("pairs") or [])
    records: list[PreferenceRecord] = []
    for index, annotation in enumerate(annotation_rows):
        source_id = str(annotation.get("source_record_id") or "").strip()
        source = source_by_id.get(source_id)
        if source is None:
            raise ValueError(f"preference annotation row {index} references unknown SFT record: {source_id}")
        pair_id = str(annotation.get("pair_id") or f"{source_id}:preference-{index}").strip()
        provenance = "; ".join(
            item
            for item in (
                str(annotation.get("provenance") or "").strip(),
                f"chosen copied from {sft_path.as_posix()}#{source_id}",
            )
            if item
        )
        records.append(
            PreferenceRecord(
                pair_id=pair_id,
                source_record_id=source_id,
                video_id=source.video_id,
                video_path=source.video_path,
                split=source.split,
                query=source.query,
                evidence=source.evidence,
                chosen=source.answer,
                rejected=str(annotation.get("rejected") or "").strip(),
                expected_behavior=source.expected_behavior,
                negative_type=str(annotation.get("negative_type") or "").strip(),
                rationale=str(annotation.get("rationale") or "").strip(),
                provenance=provenance,
                frozen_test=source.frozen_test,
            )
        )

    report = validate_preference_records(records, project_root=root, source_records=sft_records)
    if not report["valid"]:
        raise ValueError("invalid preference dataset: " + "; ".join(report["errors"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.dump(), ensure_ascii=False, sort_keys=True) + "\n")
    summary = summarize_preference_dataset(
        records,
        output,
        project_root=root,
        source_dataset_path=sft_path,
        annotations_path=annotation_path,
    )
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def load_preference_records(path: str | Path) -> list[PreferenceRecord]:
    records: list[PreferenceRecord] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if payload.get("schema_version") != PREFERENCE_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported schema_version {payload.get('schema_version', 'missing')!r}"
                )
            records.append(
                PreferenceRecord(
                    pair_id=str(payload["pair_id"]),
                    source_record_id=str(payload["source_record_id"]),
                    video_id=str(payload["video_id"]),
                    video_path=str(payload["video_path"]),
                    split=str(payload["split"]),
                    query=str(payload["query"]),
                    evidence=tuple(dict(item) for item in payload.get("evidence", [])),
                    chosen=str(payload["chosen"]),
                    rejected=str(payload["rejected"]),
                    expected_behavior=str(payload["expected_behavior"]),
                    negative_type=str(payload["negative_type"]),
                    rationale=str(payload.get("rationale", "")),
                    provenance=str(payload.get("provenance", "")),
                    frozen_test=bool(payload.get("frozen_test", False)),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid preference JSONL row {line_number}: {exc}") from exc
    return records


def validate_preference_records(
    records: Iterable[PreferenceRecord],
    *,
    project_root: str | Path | None = None,
    source_records: Iterable[SFTRecord] | None = None,
) -> dict:
    rows = list(records)
    root = Path(project_root).resolve() if project_root else None
    source_by_id = {record.record_id: record for record in source_records or []}
    errors: list[str] = []
    warnings: list[str] = []
    pair_ids: set[str] = set()
    payloads: set[str] = set()
    split_groups: dict[str, set[str]] = {}

    for index, record in enumerate(rows):
        if not record.pair_id.strip():
            errors.append(f"row {index} has empty pair_id")
        elif record.pair_id in pair_ids:
            errors.append(f"duplicate pair_id: {record.pair_id}")
        pair_ids.add(record.pair_id)
        if record.split not in {"train", "dev", "test"}:
            errors.append(f"row {index} has invalid split {record.split!r}")
        split_groups.setdefault(record.video_id, set()).add(record.split)
        if "cola" in record.video_id.lower() and record.split in {"train", "dev"}:
            errors.append(f"frozen cola video leaked into {record.split}: {record.pair_id}")
        if record.frozen_test and record.split != "test":
            errors.append(f"frozen preference pair is not test-only: {record.pair_id}")
        if not record.query.strip() or not record.chosen.strip() or not record.rejected.strip():
            errors.append(f"row {index} has empty query/chosen/rejected")
        if _normalise_text(record.chosen) == _normalise_text(record.rejected):
            errors.append(f"chosen and rejected are identical: {record.pair_id}")
        if record.negative_type not in NEGATIVE_TYPES:
            errors.append(f"row {index} has invalid negative_type {record.negative_type!r}")
        if not record.rationale.strip() or not record.provenance.strip():
            errors.append(f"row {index} lacks rationale or provenance")
        if record.expected_behavior not in {"answer", "abstain"}:
            errors.append(f"row {index} has invalid expected_behavior")

        fingerprint = _pair_fingerprint(record)
        if fingerprint in payloads:
            errors.append(f"duplicate preference payload: {record.pair_id}")
        payloads.add(fingerprint)

        evidence_ranges = _evidence_ranges(record.evidence)
        chosen_ranges = _timestamp_ranges(record.chosen)
        rejected_ranges = _timestamp_ranges(record.rejected)
        if record.expected_behavior == "answer":
            if not evidence_ranges:
                errors.append(f"answer pair has no evidence: {record.pair_id}")
            if not chosen_ranges or not all(_range_supported(item, evidence_ranges) for item in chosen_ranges):
                errors.append(f"chosen answer is not timestamp-bound to evidence: {record.pair_id}")
        else:
            if record.evidence:
                errors.append(f"abstention pair contains evidence: {record.pair_id}")
            if "证据不足" not in record.chosen:
                errors.append(f"chosen abstention lacks explicit refusal: {record.pair_id}")

        if record.negative_type == "wrong_timestamp":
            if not rejected_ranges or all(_range_supported(item, evidence_ranges) for item in rejected_ranges):
                errors.append(f"wrong_timestamp negative has no unsupported range: {record.pair_id}")
        elif record.negative_type == "missing_timestamp":
            if rejected_ranges:
                errors.append(f"missing_timestamp negative still contains a range: {record.pair_id}")
        elif record.negative_type == "hallucinated_detail":
            if record.expected_behavior != "answer" or not rejected_ranges:
                errors.append(f"hallucinated_detail must be a timestamped answer pair: {record.pair_id}")
        elif record.negative_type == "unsupported_overclaim":
            if record.expected_behavior != "abstain":
                errors.append(f"unsupported_overclaim must contrast an abstention: {record.pair_id}")
            if "证据不足" in record.rejected:
                errors.append(f"unsupported_overclaim rejected answer still abstains: {record.pair_id}")

        source = source_by_id.get(record.source_record_id)
        if source_records is not None:
            if source is None:
                errors.append(f"source SFT record is missing: {record.source_record_id}")
            elif not _matches_source(record, source):
                errors.append(f"preference fields drifted from SFT source: {record.pair_id}")

        if root is not None:
            video_path = (root / "data" / record.video_path).resolve()
            if not _within(video_path, (root / "data").resolve()):
                errors.append(f"video path escapes data root: {record.pair_id}")
            elif not video_path.exists() and record.split != "test":
                warnings.append(f"video not present locally: {record.video_path}")

    for video_id, splits in split_groups.items():
        if len(splits) > 1:
            errors.append(f"video group crosses splits: {video_id} -> {sorted(splits)}")
    for split in ("train", "dev", "test"):
        if not any(record.split == split for record in rows):
            errors.append(f"dataset has no {split} preference pairs")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "num_records": len(rows),
        "counts": dict(Counter(record.split for record in rows)),
        "negative_type_counts": dict(Counter(record.negative_type for record in rows)),
        "video_groups": {video_id: sorted(splits) for video_id, splits in sorted(split_groups.items())},
    }


def summarize_preference_dataset(
    records: Iterable[PreferenceRecord],
    dataset_path: str | Path,
    *,
    project_root: str | Path | None = None,
    source_dataset_path: str | Path | None = None,
    annotations_path: str | Path | None = None,
) -> dict:
    rows = list(records)
    dataset = Path(dataset_path).resolve()
    report = validate_preference_records(rows, project_root=project_root)
    return {
        "schema_version": PREFERENCE_SCHEMA_VERSION,
        "dataset_path": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "gradient_payload_sha256": preference_gradient_payload_sha256(rows),
        "counts": dict(Counter(record.split for record in rows)),
        "negative_type_counts": dict(Counter(record.negative_type for record in rows)),
        "video_groups": sorted({record.video_id for record in rows}),
        "frozen_test_pair_ids": [record.pair_id for record in rows if record.frozen_test],
        "cola_video_in_train_or_dev": any(
            "cola" in record.video_id.lower() and record.split in {"train", "dev"}
            for record in rows
        ),
        "source_dataset_sha256": _sha256(Path(source_dataset_path)) if source_dataset_path else "",
        "annotations_sha256": _sha256(Path(annotations_path)) if annotations_path else "",
        "validation": report,
        "provenance": (
            "Chosen answers are copied from the verified grounded SFT dataset; rejected answers are "
            "explicitly authored and categorized in data/preference/preference_annotations.json."
        ),
    }


def preference_gradient_payload_sha256(records: Iterable[PreferenceRecord]) -> str:
    payload = [
        {
            "pair_id": record.pair_id,
            "query": record.query,
            "evidence": [dict(item) for item in record.evidence],
            "chosen": record.chosen,
            "rejected": record.rejected,
            "negative_type": record.negative_type,
        }
        for record in records
        if record.split == "train"
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _matches_source(record: PreferenceRecord, source: SFTRecord) -> bool:
    return (
        record.video_id == source.video_id
        and record.video_path == source.video_path
        and record.split == source.split
        and record.query == source.query
        and record.evidence == source.evidence
        and record.chosen == source.answer
        and record.expected_behavior == source.expected_behavior
        and record.frozen_test == source.frozen_test
    )


def _pair_fingerprint(record: PreferenceRecord) -> str:
    payload = {
        "query": _normalise_text(record.query),
        "evidence": [dict(item) for item in record.evidence],
        "chosen": _normalise_text(record.chosen),
        "rejected": _normalise_text(record.rejected),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evidence_ranges(evidence: Iterable[dict]) -> list[tuple[float, float]]:
    result = []
    for item in evidence:
        try:
            start = float(item.get("start_sec"))
            end = float(item.get("end_sec"))
        except (TypeError, ValueError):
            continue
        if start >= 0 and end > start:
            result.append((start, end))
    return result


def _timestamp_ranges(text: str) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for match in _TIMESTAMP_PATTERN.finditer(text):
        start = float(match.group("start"))
        end = float(match.group("end"))
        if end > start and (start, end) not in result:
            result.append((start, end))
    return result


def _range_supported(candidate: tuple[float, float], evidence: Iterable[tuple[float, float]]) -> bool:
    return any(abs(candidate[0] - item[0]) <= 0.11 and abs(candidate[1] - item[1]) <= 0.11 for item in evidence)


def _normalise_text(value: str) -> str:
    return " ".join(value.split()).strip().lower()


def _within(target: Path, root: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


def _rooted(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
