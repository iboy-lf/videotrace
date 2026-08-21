import json
from pathlib import Path

from videomemo.pipeline import VideoMemoPipeline
from videomemo.config import VideoMemoConfig
from videomemo.eval.reproducibility import file_sha256
from videomemo.export.simple_export import export_manifest
from videomemo.models import KnowledgePack
from videomemo.web.server import _artifact_url
from scripts.sample_video import make_demo_video


def test_pipeline_smoke(tmp_path):
    video_path = make_demo_video(str(tmp_path / "test_sample.mp4"), duration_sec=8, fps=4)
    cfg = VideoMemoConfig(
        segment_seconds=4,
        top_k=2,
        output_dir=str(tmp_path / "outputs"),
        scorer_model_path=str(tmp_path / "outputs" / "missing.pkl"),
    )
    pipeline = VideoMemoPipeline(cfg)
    pack = pipeline.run(video_path, query="这个视频讲了什么？")
    assert pack.summary
    assert pack.metadata["agent_run"]["verified"]
    assert len(pack.metadata["agent_run"]["tool_trace"]) == 6
    assert all("test_sample" not in segment.text for segment in pack.segments)
    assert pack.metadata["score_fusion"]["normalization"] == "per_query_min_max"
    assert pack.metadata["video_sha256"] == file_sha256(Path(video_path))
    assert pack.metadata["environment"]["python"]
    assert pack.metadata["performance"]["stage_seconds"]["total"] > 0
    assert all(0.0 <= segment.score <= 1.0 for segment in pack.segments)


def test_manifest_exports_video_sha256(tmp_path):
    pack = KnowledgePack(
        video_path="demo.mp4",
        duration_sec=1.0,
        segments=[],
        summary="summary",
        answer="answer",
        timeline=[],
        clips=[],
        metadata={"video_sha256": "video-fingerprint"},
    )
    manifest_path = export_manifest(pack, str(tmp_path / "artifact"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["video_sha256"] == "video-fingerprint"


def test_artifact_url_keeps_static_report_relative_assets_under_workspace():
    root = Path(__file__).resolve().parents[1]
    demo_path = root / "outputs" / "cola_review_qwen35" / "demo.html"
    assert _artifact_url(demo_path) == "/artifact/outputs/cola_review_qwen35/demo.html"
