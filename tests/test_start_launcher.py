from __future__ import annotations

import hashlib
import json

import scripts.start as start


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_one_click_launcher_prefers_current_canonical_pack(monkeypatch, tmp_path):
    root = tmp_path
    payload = b"canonical-video"
    video = root / "data" / "raw" / "cola_review.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(payload)
    canonical = root / "outputs" / "iboy_qwen35" / "cola_review" / "knowledge_pack.json"
    legacy = root / "outputs" / "cola_review_qwen35" / "knowledge_pack.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(
        json.dumps(
            {
                "video_path": "/lavender/VideoTrace/data/raw/cola_review.mp4",
                "metadata": {"video_sha256": _sha256(payload)},
            }
        ),
        encoding="utf-8",
    )
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"video_path": str(video)}), encoding="utf-8")

    monkeypatch.setattr(start, "ROOT", root)
    assert start._preferred_pack() == (canonical, True)


def test_one_click_launcher_serves_canonical_evidence_when_video_hash_mismatches(monkeypatch, tmp_path):
    root = tmp_path
    video = root / "data" / "raw" / "cola_review.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"different-video")
    canonical = root / "outputs" / "iboy_qwen35" / "cola_review" / "knowledge_pack.json"
    legacy = root / "outputs" / "cola_review_qwen35" / "knowledge_pack.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(
        json.dumps(
            {
                "video_path": "/lavender/VideoTrace/data/raw/cola_review.mp4",
                "metadata": {"video_sha256": _sha256(b"expected-video")},
            }
        ),
        encoding="utf-8",
    )
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"video_path": str(video)}), encoding="utf-8")

    monkeypatch.setattr(start, "ROOT", root)
    # The canonical pack is still served -- read-only, evidence cannot be
    # replayed -- but it is never downgraded to the stale legacy result.
    assert start._preferred_pack() == (canonical, False)


def test_one_click_launcher_never_falls_back_to_baseline_sample(monkeypatch, tmp_path):
    root = tmp_path
    video = root / "data" / "raw" / "sample.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    sample = root / "outputs" / "sample" / "knowledge_pack.json"
    sample.parent.mkdir(parents=True)
    sample.write_text(json.dumps({"video_path": str(video)}), encoding="utf-8")

    monkeypatch.setattr(start, "ROOT", root)
    assert start._preferred_pack() == (None, False)


def test_one_click_launcher_serves_evidence_when_video_file_is_absent(monkeypatch, tmp_path):
    """A fresh clone has no source video; the verified evidence must still open."""

    root = tmp_path
    canonical = root / "outputs" / "iboy_qwen35" / "cola_review" / "knowledge_pack.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(
        json.dumps(
            {
                "video_path": "/lavender/VideoTrace/data/raw/cola_review.mp4",
                "metadata": {"video_sha256": _sha256(b"canonical-video")},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(start, "ROOT", root)
    assert start._preferred_pack() == (canonical, False)
