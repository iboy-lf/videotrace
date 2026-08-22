from __future__ import annotations

from pathlib import Path
import json
import hashlib
import os
import shutil
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request

import pytest

from videomemo.config import VideoMemoConfig
from videomemo.export.clip_export import export_clip_files
from videomemo.models import KnowledgePack
from videomemo.web import server
from videomemo.web.server import VideoMemoWebHandler
from videomemo.web.vlm_modes import apply_vlm_mode, available_vlm_modes, capability_payload


ROOT = Path(__file__).resolve().parents[1]


def _find_node_binary() -> str | None:
    """Find Node even on managed hosts where it is not exported in PATH."""
    configured = os.environ.get("VIDEOTRACE_NODE", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if _is_executable(candidate):
            return str(candidate)
    path_node = shutil.which("node")
    if path_node and _is_executable(Path(path_node)):
        return str(path_node)

    # Only now fall back to scanning known runtime locations. iboy's VS
    # Code/Cursor runtimes are stable, user-owned binaries and do not require
    # installing anything into the environment. On a CI runner these paths can
    # exist while belonging to another user, so an unreadable directory must be
    # skipped rather than aborting discovery.
    candidates: list[Path] = []
    for root in (Path("/root/.vscode-server"), Path("/root/.cursor-server")):
        try:
            if not root.exists():
                continue
            candidates.extend(sorted(root.glob("**/server/node"), reverse=True))
            candidates.extend(sorted(root.glob("**/node"), reverse=True))
        except OSError:
            continue
    candidates.append(Path("/usr/local/nvm/versions/node/v18.20.3/bin/node"))
    for candidate in candidates:
        if _is_executable(candidate):
            return str(candidate)
    return None


def _is_executable(candidate: Path) -> bool:
    try:
        return candidate.is_file() and os.access(candidate, os.X_OK)
    except OSError:
        return False


def test_user_web_exposes_only_visual_mode_and_keeps_core_controls_visible():
    html = (ROOT / "src" / "videomemo" / "web" / "static" / "index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "src" / "videomemo" / "web" / "static" / "app.js").read_text(encoding="utf-8")

    assert '<section id="analysisControls" class="analysisBar">' in html
    assert 'id="queryControls"' in html
    assert 'id="vlmMode"' in html
    assert 'id="jobMeta"' in html
    assert 'id="retryBtn"' in html
    assert html.count("<select") == 1
    assert "segmentBackend" not in html
    assert "llmBackend" not in html
    assert "reranker" not in html.lower()
    assert "dtype" not in html.lower()
    assert 'id="device"' not in html.lower()
    assert "body: JSON.stringify({ video_id: videoId, query, vlm_mode: vlmMode })" in javascript
    assert "无证据拒答" in javascript
    assert r'class=\"inlineCitation\"' in javascript
    assert ".inlineCitation.active" in javascript
    assert '$("analysisControls").hidden' not in javascript
    assert '$("queryControls").hidden' not in javascript


def test_playback_controller_behavior_contract_runs_in_node():
    node = _find_node_binary()
    if not node:
        pytest.skip("Node.js is unavailable on this host")
    result = subprocess.run(
        [node, str(ROOT / "tests" / "js" / "playback.test.cjs")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_technical_summary_matches_selected_visual_mode_in_node():
    node = _find_node_binary()
    if not node:
        pytest.skip("Node.js is unavailable on this host")
    result = subprocess.run(
        [node, str(ROOT / "tests" / "js" / "technical.test.cjs")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_server_discovers_only_real_whitelisted_vlm_modes(monkeypatch, tmp_path):
    qwen = tmp_path / "models" / "Qwen3.5-9B"
    siglip = tmp_path / "models" / "siglip2-large-patch16-256"
    checkpoint = tmp_path / "neural_reranker.pt"
    qwen.mkdir(parents=True)
    siglip.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv("VIDEOTRACE_MODEL_ROOTS", str(tmp_path / "models"))
    config = VideoMemoConfig(
        segment_understanding_model=str(qwen),
        llm_model=str(qwen),
        vlm_backend="siglip",
        vlm_model_name=str(siglip),
        reranker_backend="neural",
        reranker_model_path=str(checkpoint),
    )

    modes = available_vlm_modes(config)
    assert [mode.mode_id for mode in modes] == ["auto_best", "qwen35_video", "siglip_retrieval"]
    assert capability_payload(config)["default_mode"] == "auto_best"
    updated, selected = apply_vlm_mode(config, "siglip_retrieval")
    assert selected.label == "SigLIP2 检索增强"
    assert updated.selected_vlm_mode == "siglip_retrieval"
    assert updated.vlm_backend == "siglip"
    assert updated.segment_understanding_backend == "baseline"
    with pytest.raises(ValueError):
        apply_vlm_mode(config, "../../arbitrary-model")


def test_missing_remote_weights_do_not_hide_upload_or_invent_modes(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEOTRACE_MODEL_ROOTS", str(tmp_path / "missing"))
    config = VideoMemoConfig(
        segment_understanding_model=str(tmp_path / "missing" / "Qwen3.5-9B"),
        llm_model=str(tmp_path / "missing" / "Qwen3.5-9B"),
        vlm_backend="siglip",
        vlm_model_name=str(tmp_path / "missing" / "siglip2-large-patch16-256"),
    )
    capability = capability_payload(config)
    assert capability["analysis_available"] is False
    assert capability["vlm_modes"] == []
    html = (ROOT / "src" / "videomemo" / "web" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="videoUpload"' in html
    assert 'id="runBtn"' in html


def test_health_payload_binds_service_to_current_source(monkeypatch, tmp_path):
    from videomemo.eval.reproducibility import source_fingerprint

    source = tmp_path / "src" / "service.py"
    source.parent.mkdir(parents=True)
    source.write_text("service = 'current'\n", encoding="utf-8")
    server._service_source_sha256.cache_clear()
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "_runtime_defaults", lambda: {"stack": "test"})
    monkeypatch.setattr(server, "_capabilities", lambda: {"state": "ready"})
    monkeypatch.setattr(server, "_job_service_status", lambda: {"counts": {}})

    health = server._health_payload()

    assert health["root"] == str(tmp_path)
    assert health["source_sha256"] == source_fingerprint(tmp_path)
    assert health["service"] == {"state": "ready"}


def test_gpu_runtime_check_allows_own_service_pid_but_rejects_external_pid(monkeypatch):
    own_pid = 4321
    gpu_rows = "0, GPU-0, 19000, 0\n1, GPU-1, 2000, 0\n"

    def run(command, **_kwargs):
        if any(str(value).startswith("--query-gpu=") for value in command):
            return gpu_rows
        return f"GPU-0, {own_pid}, python, 19000\nGPU-1, {own_pid}, python, 2000\n"

    monkeypatch.setattr(server.os, "getpid", lambda: own_pid)
    monkeypatch.setattr(server.subprocess, "check_output", run)
    assert server._gpu_snapshot(["0", "1"])["safe"] is True

    def run_with_external(command, **_kwargs):
        if any(str(value).startswith("--query-gpu=") for value in command):
            return gpu_rows
        return f"GPU-0, {own_pid}, python, 19000\nGPU-1, 9876, other, 2000\n"

    monkeypatch.setattr(server.subprocess, "check_output", run_with_external)
    report = server._gpu_snapshot(["0", "1"])
    assert report["safe"] is False
    assert report["gpus"][1]["other_compute_pids"] == [9876]


def test_web_adapter_requires_explicit_frozen_eval_admission(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    adapter = tmp_path / "outputs" / "models" / "qwen35_sft_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"validated-adapter")
    evaluation = tmp_path / "outputs" / "reports" / "qwen35_adapter_eval.json"
    evaluation.parent.mkdir(parents=True)
    evaluation.write_text('{"status":"completed"}', encoding="utf-8")
    metrics = tmp_path / "outputs" / "models" / "qwen35_sft_metrics.json"

    config = server._prepare_runtime_config(VideoMemoConfig())
    assert config.llm_adapter_path == ""

    admission = {
        "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "evaluation_report": str(evaluation),
        "evaluation_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
    }
    metrics.write_text(
        json.dumps({"validated_for_web": False, "adapter_admission": admission}),
        encoding="utf-8",
    )
    config = server._prepare_runtime_config(VideoMemoConfig())
    assert config.llm_adapter_path == ""

    metrics.write_text(
        json.dumps({"validated_for_web": True, "adapter_admission": admission}),
        encoding="utf-8",
    )
    config = server._prepare_runtime_config(VideoMemoConfig())
    assert config.llm_adapter_path == str(adapter.resolve())

    weights.write_bytes(b"tampered")
    config = server._prepare_runtime_config(VideoMemoConfig())
    assert config.llm_adapter_path == ""


def test_best_adapter_registry_prefers_dpo_and_hash_falls_back_to_sft(monkeypatch, tmp_path):
    from videomemo.eval.reproducibility import file_sha256, source_fingerprint
    from videomemo.llm.adapter_admission import BEST_ADAPTER_SCHEMA_VERSION

    monkeypatch.setattr(server, "ROOT", tmp_path)
    for relative in ("src/core.py", "scripts/tool.py", "configs/demo.yaml", "tests/test_demo.py"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relative, encoding="utf-8")
    (tmp_path / "README.md").write_text("# test", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'", encoding="utf-8")
    source_sha = source_fingerprint(tmp_path)
    entries = {}
    for candidate_id, method, folder in (
        ("qwen35_sft", "sft", "qwen35_sft_adapter"),
        ("qwen35_dpo", "dpo", "qwen35_dpo_adapter"),
    ):
        adapter = tmp_path / "outputs" / "models" / folder
        adapter.mkdir(parents=True)
        weights = adapter / "adapter_model.safetensors"
        config_path = adapter / "adapter_config.json"
        weights.write_bytes(candidate_id.encode("utf-8"))
        config_path.write_text("{}", encoding="utf-8")
        metrics = tmp_path / "outputs" / "models" / f"{candidate_id}_metrics.json"
        card = tmp_path / "outputs" / "models" / f"{candidate_id}_model_card.json"
        metrics.write_text("{}", encoding="utf-8")
        card.write_text("{}", encoding="utf-8")
        report = tmp_path / "outputs" / "reports" / "adapter_admissions" / f"{candidate_id}.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text('{"status":"completed"}', encoding="utf-8")
        entries[candidate_id] = {
            "candidate_id": candidate_id,
            "method": method,
            "validated_for_web": True,
            "adapter_sha256": file_sha256(weights),
            "adapter_config_sha256": file_sha256(config_path),
            "metrics_sha256": file_sha256(metrics),
            "model_card_sha256": file_sha256(card),
            "evaluation_report": str(report),
            "evaluation_sha256": file_sha256(report),
            "source_sha256": source_sha,
        }
    registry = tmp_path / "outputs" / "models" / "best_adapter.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": BEST_ADAPTER_SCHEMA_VERSION,
                "validated_for_web": True,
                "source_sha256": source_sha,
                "selected_candidate_id": "qwen35_dpo",
                "fallback_order": ["qwen35_sft"],
                "candidates": entries,
            }
        ),
        encoding="utf-8",
    )

    config = server._prepare_runtime_config(VideoMemoConfig())
    assert config.llm_adapter_path == str(
        (tmp_path / "outputs/models/qwen35_dpo_adapter").resolve()
    )

    (tmp_path / "outputs/models/qwen35_dpo_adapter/adapter_model.safetensors").write_bytes(b"tampered")
    config = server._prepare_runtime_config(VideoMemoConfig())
    assert config.llm_adapter_path == str(
        (tmp_path / "outputs/models/qwen35_sft_adapter").resolve()
    )

def test_analysis_progress_reports_liveness_without_reaching_export_range():
    assert server._analysis_progress(0) == 35
    assert server._analysis_progress(20) == 41
    assert server._analysis_progress(1000) == 85


def test_web_upload_is_project_scoped_and_media_range_returns_206(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), VideoMemoWebHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_port}"
    try:
        request = urllib.request.Request(
            f"{base}/api/upload",
            data=b"0123456789abcdef",
            method="POST",
            headers={"Content-Type": "application/octet-stream", "X-Filename": "../demo.mp4"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            uploaded = json.loads(response.read().decode("utf-8"))
        assert uploaded["ok"] is True
        assert uploaded["video_id"].startswith("uploads/")
        target = server._resolve_video_id(uploaded["video_id"])
        assert target is not None
        assert target.is_relative_to(tmp_path / "data")

        range_request = urllib.request.Request(
            base + uploaded["video_url"],
            headers={"Range": "bytes=2-7"},
        )
        with urllib.request.urlopen(range_request, timeout=5) as response:
            assert response.status == 206
            assert response.headers["Content-Range"].startswith("bytes 2-7/")
            assert response.read() == b"234567"

        outside = tmp_path.parent / "outside.mp4"
        outside.write_bytes(b"outside")
        bad_url = f"{base}/media?path={urllib.parse.quote(str(outside.resolve()), safe='')}"
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(bad_url, timeout=5)
        assert error.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_web_multipart_upload_streams_exact_bytes_without_cgi(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), VideoMemoWebHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_port}"
    boundary = "VideoTraceBoundary7MA4YWxkTrZu0gW"
    payload = b"\x00video\r\nline\nend\xff"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="note"\r\n\r\n',
            b"ignored\r\n",
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="file"; filename="../demo.mp4"\r\n',
            b"Content-Type: video/mp4\r\n\r\n",
            payload,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    try:
        request = urllib.request.Request(
            f"{base}/api/upload",
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            uploaded = json.loads(response.read().decode("utf-8"))
        target = server._resolve_video_id(uploaded["video_id"])
        assert uploaded["ok"] is True
        assert uploaded["filename"] == "demo.mp4"
        assert uploaded["size_bytes"] == len(payload)
        assert target is not None
        assert target.read_bytes() == payload
        assert list((tmp_path / "data" / "uploads").glob("*.uploading")) == []

        invalid_body = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="note"\r\n\r\n',
                b"no file\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        invalid_request = urllib.request.Request(
            f"{base}/api/upload",
            data=invalid_body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(invalid_request, timeout=5)
        assert error.value.code == 400
        assert list((tmp_path / "data" / "uploads").glob("*.uploading")) == []
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    source = (ROOT / "src" / "videomemo" / "web" / "server.py").read_text(encoding="utf-8")
    assert "import cgi" not in source
    assert "FieldStorage" not in source


def test_web_clip_urls_point_to_original_video_time_windows(monkeypatch, tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    pack = {
        "video_path": str(video),
        "duration_sec": 30.0,
        "timeline": [],
        "clips": [
            {
                "segment_id": "seg-0000",
                "start_sec": 4.0,
                "end_sec": 12.0,
                "file": str(tmp_path / "broken-mp4v.mp4"),
            }
        ],
        "metadata": {},
    }
    monkeypatch.setattr(server, "ROOT", tmp_path)
    handler = object.__new__(VideoMemoWebHandler)
    response = handler._pack_to_response(pack, tmp_path / "demo.html")

    assert response["clips"][0]["playback_url"] == response["video_url"]
    assert response["clips"][0]["url"].endswith("#t=4.000,12.000")
    assert "broken-mp4v" not in response["clips"][0]["url"]
    assert response["product"]["model_selection_locked"] is True
    assert response["product"]["evidence_playback"] == "source_video_window"


def test_web_maps_remote_canonical_video_to_hash_identical_local_source(monkeypatch, tmp_path):
    payload = b"same-video-bytes"
    local_video = tmp_path / "data" / "raw" / "cola_review.mp4"
    local_video.parent.mkdir(parents=True)
    local_video.write_bytes(payload)
    remote_path = "/lavender/VideoTrace/data/raw/cola_review.mp4"
    pack = {
        "video_path": remote_path,
        "duration_sec": 30.0,
        "timeline": [],
        "clips": [{"segment_id": "seg-0000", "start_sec": 4.0, "end_sec": 12.0}],
        "metadata": {"video_sha256": hashlib.sha256(payload).hexdigest()},
    }
    monkeypatch.setattr(server, "ROOT", tmp_path)
    handler = object.__new__(VideoMemoWebHandler)

    response = handler._pack_to_response(pack, tmp_path / "demo.html")

    assert response["video_path"] == str(local_video.resolve())
    assert response["video_id"] == "raw/cola_review.mp4"
    assert response["media_ready"] is True
    assert response["metadata"]["artifact_video_path"] == remote_path
    assert response["metadata"]["video_path_remapped"] is True
    assert urllib.parse.quote(str(local_video.resolve()), safe="") in response["video_url"]
    assert response["clips"][0]["playback_url"] == response["video_url"]


def test_clip_export_preserves_source_codec_and_audio(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source-video")
    pack = KnowledgePack(
        video_path=str(video),
        duration_sec=30.0,
        segments=[],
        summary="summary",
        answer="answer",
        timeline=[],
        clips=[{"segment_id": "seg-0000", "start_sec": 2.0, "end_sec": 8.0, "score": 1.0}],
    )

    clips = export_clip_files(pack, str(tmp_path / "artifact"))

    assert clips[0]["file"] == str(video.resolve())
    assert clips[0]["playback_mode"] == "source_video_window"
    assert not (tmp_path / "artifact" / "clips").exists()


def test_node_discovery_survives_unreadable_runtime_directories(monkeypatch):
    """A CI runner has /root/.vscode-server owned by another user.

    Globbing it raised PermissionError and aborted the whole test module. The
    scan must skip unreadable roots instead.
    """

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.delenv("VIDEOTRACE_NODE", raising=False)

    def _boom(self, _pattern):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "glob", _boom)
    monkeypatch.setattr(Path, "is_file", lambda self: False)

    assert _find_node_binary() is None


def test_node_discovery_prefers_path_and_skips_scanning(monkeypatch):
    """When node is on PATH the fallback scan must never run."""

    monkeypatch.delenv("VIDEOTRACE_NODE", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/node")
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(os, "access", lambda _path, _mode: True)

    def _fail(self, _pattern):  # pragma: no cover - must not be reached
        raise AssertionError("fallback scan ran even though PATH resolved node")

    monkeypatch.setattr(Path, "glob", _fail)

    assert _find_node_binary() == "/usr/bin/node"
