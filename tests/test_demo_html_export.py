from __future__ import annotations

import json

from videomemo.export.demo_html import export_demo_html
from videomemo.models import KnowledgePack


def _pack(video_path: str = "/remote/VideoTrace/data/raw/cola_review.mp4") -> KnowledgePack:
    timeline = [
        {"start_sec": 0.0, "end_sec": 20.0, "text": "开场展示多款可乐 场景：室内"},
        {"start_sec": 200.0, "end_sec": 220.0, "text": "倒入杯中试喝"},
    ]
    return KnowledgePack(
        video_path=video_path,
        duration_sec=416.2,
        segments=[],
        summary="摘要",
        answer="问题：视频讲了什么？\n结论：横评 17 款可乐。",
        timeline=timeline,
        clips=[],
        metadata={"query": "视频讲了什么？", "agent_run": {"verified": True, "verification": {"coverage": 1.0}}},
    )


def test_demo_html_is_written_with_lf_so_its_digest_is_platform_stable(tmp_path):
    """The manifest hashes these bytes; CRLF on Windows would change the digest."""

    path = export_demo_html(_pack(), str(tmp_path))
    assert b"\r\n" not in path.read_bytes()


def test_demo_html_degrades_to_evidence_only_when_the_video_cannot_load(tmp_path):
    """A clone has no source video, so the standalone demo must say so."""

    html = export_demo_html(_pack(), str(tmp_path)).read_text(encoding="utf-8")

    # The notice exists but starts hidden: it is revealed by the video's own
    # error event, so a working video never shows it.
    assert 'id="mediaMissing"' in html
    assert 'class="mediaMissing" hidden' in html
    assert 'video.addEventListener("error"' in html
    # Evidence buttons must be disabled rather than silently doing nothing.
    assert "button.disabled = true" in html


def test_demo_html_still_binds_every_evidence_button_to_its_timestamp(tmp_path):
    html = export_demo_html(_pack(), str(tmp_path)).read_text(encoding="utf-8")

    assert html.count('class="evidence"') == 2
    assert 'data-start="0.000"' in html
    assert 'data-end="20.000"' in html
    assert 'data-start="200.000"' in html
    # The timeline payload the click handler indexes into must match.
    payload = html.split("const timeline = ", 1)[1].split(";\n", 1)[0]
    assert len(json.loads(payload)) == 2
