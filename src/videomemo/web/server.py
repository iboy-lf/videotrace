from __future__ import annotations

from email.parser import BytesHeaderParser
from email.policy import default as EMAIL_POLICY
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse
from functools import lru_cache
import json
import mimetypes
import os
import queue
import re
import subprocess
import threading
import traceback
import time
import uuid

from ..config import VideoMemoConfig
from ..eval.reproducibility import source_fingerprint
from ..media_resolver import resolve_pack_video
from ..pipeline import VideoMemoPipeline
from .vlm_modes import apply_vlm_mode, capability_payload
from .job_store import job_store_dir, load_jobs, persist_job
from ..llm.adapter_admission import resolve_validated_adapter


ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = Path(__file__).resolve().parent / "static"
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
MULTIPART_HEADER_BYTES = 64 * 1024
MULTIPART_LINE_BYTES = 1024 * 1024

_PIPELINES: dict[str, VideoMemoPipeline] = {}
_PIPELINE_LOCK = threading.Lock()
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_JOB_QUEUE: queue.Queue[str] = queue.Queue()
_WORKER_LOCK = threading.Lock()
_WORKER_STARTED = False
_JOB_STATE_LOCK = threading.Lock()
_JOB_STATE_ROOT = ""
_JOB_LOAD_WARNINGS: list[str] = []
_ACTIVE_JOB_ID = ""


class MultipartUploadError(ValueError):
    """Malformed multipart requests that should be reported as client errors."""


class _ContentLengthReader:
    """Prevent multipart parsing from consuming bytes past this HTTP request."""

    def __init__(self, source, content_length: int):
        self.source = source
        self.remaining = max(0, int(content_length))

    def read(self, size: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""
        amount = self.remaining if size is None or size < 0 else min(self.remaining, size)
        chunk = self.source.read(amount)
        self.remaining -= len(chunk)
        return chunk

    def readline(self, size: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""
        amount = self.remaining if size is None or size < 0 else min(self.remaining, size)
        line = self.source.readline(amount)
        self.remaining -= len(line)
        return line


class VideoMemoWebHandler(SimpleHTTPRequestHandler):
    server_version = "VideoTraceWeb/0.3"

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            return self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if parsed.path.startswith("/static/"):
            target = STATIC_DIR / parsed.path.replace("/static/", "", 1)
            return self._send_file(target)
        if parsed.path == "/api/latest":
            return self._send_json(self._load_latest())
        if parsed.path == "/api/capabilities":
            return self._send_json(_capabilities())
        if parsed.path == "/api/health":
            return self._send_json(_health_payload())
        if parsed.path == "/api/jobs":
            query_values = parse_qs(parsed.query)
            raw_limit = query_values.get("limit", ["20"])[0]
            try:
                limit = max(1, min(100, int(raw_limit)))
            except (TypeError, ValueError):
                limit = 20
            return self._send_json({"ok": True, "jobs": _list_jobs(limit), "service": _job_service_status()})
        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            job = _job_public(job_id)
            if job is None:
                return self._send_json({"ok": False, "message": "任务不存在。"}, status=404)
            return self._send_json(job)
        if parsed.path.startswith("/artifact/"):
            relative_path = unquote(parsed.path.replace("/artifact/", "", 1))
            return self._send_media(str((ROOT / relative_path).resolve()), allow_artifact=True)
        if parsed.path == "/media":
            query_values = parse_qs(parsed.query)
            media_path = unquote(query_values.get("path", [""])[0])
            return self._send_media(media_path)
        self.send_error(404)

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/static/"):
            target = STATIC_DIR / parsed.path.replace("/static/", "", 1)
            return self._send_file(target, head_only=True)
        if parsed.path == "/media":
            query_values = parse_qs(parsed.query)
            media_path = unquote(query_values.get("path", [""])[0])
            return self._send_media(media_path, head_only=True)
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/upload":
                return self._handle_upload()
            if parsed.path in {"/api/jobs", "/api/run"}:
                return self._handle_job_submission()
            self.send_error(404)
        except Exception as exc:
            return self._send_json(
                {
                    "ready": False,
                    "ok": False,
                    "message": f"服务端处理失败：{type(exc).__name__}: {exc}",
                },
                status=500,
            )

    def _handle_job_submission(self) -> None:
        length = _content_length(self.headers.get("Content-Length", ""), maximum=64 * 1024)
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        video_id = str(payload.get("video_id") or "").strip()
        query_text = str(payload.get("query") or "").strip()
        mode_id = str(payload.get("vlm_mode") or "").strip()
        if not video_id:
            return self._send_json({"ok": False, "message": "请先上传或选择视频。"}, status=400)
        if not query_text:
            return self._send_json({"ok": False, "message": "问题不能为空。"}, status=400)
        if len(query_text) > 4000:
            return self._send_json({"ok": False, "message": "问题过长，请控制在 4000 字以内。"}, status=400)
        target = _resolve_video_id(video_id)
        if target is None:
            return self._send_json({"ok": False, "message": "视频 ID 无效或已失效。"}, status=400)
        capabilities = _capabilities()
        allowed_modes = {item["id"] for item in capabilities["vlm_modes"]}
        selected_mode = mode_id or capabilities.get("default_mode", "")
        if selected_mode not in allowed_modes:
            return self._send_json(
                {"ok": False, "message": "所选视觉模式当前不可用，请刷新算力状态。"},
                status=503,
            )
        return self._send_json(_submit_job(target, query_text, selected_mode), status=202)

    def _handle_upload(self) -> None:
        upload_dir = ROOT / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        maximum = _max_upload_bytes()
        try:
            content_length = _content_length(
                self.headers.get("Content-Length", ""),
                maximum=maximum + 8 * 1024 * 1024,
            )
        except ValueError as exc:
            status = 413 if "超过" in str(exc) else 400
            return self._send_json({"ok": False, "message": str(exc)}, status=status)
        content_type = self.headers.get("Content-Type", "")
        filename = ""
        partial: Path | None = None
        written = 0

        if content_type.startswith("multipart/form-data"):
            partial = upload_dir / f".{uuid.uuid4().hex}.uploading"
            try:
                filename, written = _receive_multipart_file(
                    self.rfile,
                    content_type=content_type,
                    content_length=content_length,
                    target=partial,
                    maximum=maximum,
                )
            except MultipartUploadError as exc:
                partial.unlink(missing_ok=True)
                return self._send_json({"ok": False, "message": str(exc)}, status=400)
            except ValueError as exc:
                partial.unlink(missing_ok=True)
                return self._send_json({"ok": False, "message": str(exc)}, status=413)
        else:
            filename = unquote(self.headers.get("X-Filename", "")).strip()

        safe_name = _safe_upload_name(filename)
        if not safe_name:
            if partial is not None:
                partial.unlink(missing_ok=True)
            return self._send_json({"ok": False, "message": "仅支持常见视频格式。"}, status=400)

        unique_name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:10]}-{safe_name}"
        target = (upload_dir / unique_name).resolve()
        if not _is_within(target, upload_dir.resolve()):
            if partial is not None:
                partial.unlink(missing_ok=True)
            return self._send_json({"ok": False, "message": "上传文件名无效。"}, status=400)
        if partial is None:
            partial = target.with_suffix(target.suffix + ".uploading")
            try:
                written = _copy_limited(
                    self.rfile,
                    partial,
                    maximum,
                    expected_bytes=content_length,
                )
            except ValueError as exc:
                partial.unlink(missing_ok=True)
                return self._send_json({"ok": False, "message": str(exc)}, status=413)
        if written <= 0:
            partial.unlink(missing_ok=True)
            return self._send_json({"ok": False, "message": "上传文件为空。"}, status=400)
        os.replace(partial, target)

        return self._send_json(
            {
                "ok": True,
                "video_id": _video_id(target),
                "video_url": _media_url(str(target)),
                "filename": safe_name,
                "size_bytes": written,
            }
        )

    def _load_latest(self) -> dict:
        configured_pack = os.environ.get("VIDEOTRACE_LATEST_PACK", "").strip()
        pack_path = Path(configured_pack).expanduser().resolve() if configured_pack else _preferred_pack_path()
        if not _is_within(pack_path, ROOT.resolve()):
            return {
                "ready": False,
                "message": "VIDEOTRACE_LATEST_PACK 必须位于项目目录内。",
                "defaults": _runtime_defaults(),
                "product": _product_payload(),
            }
        demo_path = pack_path.parent / "demo.html"
        if not pack_path.exists():
            sample_video = _preferred_sample_video()
            return {
                "ready": False,
                "sample_video": str(sample_video) if sample_video else "",
                "sample_video_id": _video_id(sample_video) if sample_video else "",
                "sample_video_url": _media_url(str(sample_video)) if sample_video else "",
                "sample_query": "这个视频主要讲了什么？请给出带时间戳的证据。",
                "defaults": _runtime_defaults(),
                "product": _product_payload(),
                "message": "还没有生成结果，可以上传视频开始分析。",
            }
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        return self._pack_to_response(pack, demo_path)

    def _pack_to_response(self, pack: dict, demo_path: Path, retrieval_eval: dict | None = None) -> dict:
        metadata = pack.get("metadata", {})
        agent_run = metadata.get("agent_run", {})
        clips = pack.get("clips", [])
        artifact_video_path = str(pack.get("video_path", ""))
        resolved_video = resolve_pack_video(pack, ROOT)
        video_path = str(resolved_video) if resolved_video else ""
        video_remapped = bool(video_path and video_path != artifact_video_path)
        return {
            "ready": True,
            "video_path": video_path or artifact_video_path,
            "video_id": _video_id(Path(video_path)) if video_path else "",
            "video_url": _media_url(video_path),
            "duration_sec": pack.get("duration_sec", 0.0),
            "video_title": Path(video_path or artifact_video_path).stem,
            "media_ready": bool(video_path),
            "summary": pack.get("summary"),
            "answer": pack.get("answer"),
            "timeline": pack.get("timeline", []),
            "clips": [
                {
                    **clip,
                    "url": _time_window_url(video_path, clip.get("start_sec", 0.0), clip.get("end_sec", 0.0)),
                    "playback_url": _media_url(video_path),
                }
                for clip in clips
            ],
            "agent": agent_run,
            "metadata": {
                "query": metadata.get("query"),
                "vlm_mode": metadata.get("vlm_mode", {}),
                "vlm": metadata.get("vlm", {}),
                "segment_understanding": metadata.get("segment_understanding", {}),
                "asr": metadata.get("asr", {}),
                "score_fusion": metadata.get("score_fusion", {}),
                "query_intent": metadata.get("query_intent", {}),
                "retrieval_selection": metadata.get("retrieval_selection", {}),
                "reranker": metadata.get("reranker", {}),
                "performance": metadata.get("performance", {}),
                "index_stats": metadata.get("index_stats", {}),
                "llm_backend": metadata.get("llm_backend"),
                "llm_adapter": metadata.get("llm_adapter", {}),
                "scorer_mode": metadata.get("scorer_mode"),
                "persistent_memory": metadata.get("persistent_memory", {}),
                "source_sha256": metadata.get("source_sha256", ""),
                "video_sha256": metadata.get("video_sha256", ""),
                "environment": metadata.get("environment", {}),
                "deployment": metadata.get("deployment", {}),
                "artifact_video_path": artifact_video_path,
                "video_path_remapped": video_remapped,
                "tool_safeguards": agent_run.get("safeguards", {}),
            },
            "eval": {
                "retrieval": retrieval_eval,
                "agent_score": _agent_score_from_json(agent_run),
            },
            "files": {
                "demo": str(demo_path),
                "demo_url": _artifact_url(demo_path),
                "knowledge_pack": str(demo_path.parent / "knowledge_pack.json"),
                "report": str(demo_path.parent / "report.md"),
            },
            "defaults": _runtime_defaults(),
            "product": _product_payload(),
        }

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str | None = None, head_only: bool = False) -> None:
        target = path.resolve()
        if not target.exists() or not _is_within(target, STATIC_DIR.resolve()):
            self.send_error(404)
            return
        guessed_type = content_type or mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if target.suffix in {".js", ".css", ".mjs"} and "charset" not in guessed_type:
            guessed_type = f"{guessed_type}; charset=utf-8"
        content = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", guessed_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if not head_only:
            self.wfile.write(content)

    def _send_media(self, path: str, allow_artifact: bool = False, head_only: bool = False) -> None:
        if not path:
            self.send_error(404)
            return
        target = Path(path).resolve()
        allowed = target.exists() and _is_within(target, ROOT.resolve())
        if not allow_artifact:
            allowed = allowed and target.suffix.lower() in VIDEO_SUFFIXES
        if not allowed or not target.is_file():
            self.send_error(404)
            return
        file_size = target.stat().st_size
        if file_size <= 0:
            self.send_error(404)
            return
        range_header = self.headers.get("Range", "")
        start = 0
        end = file_size - 1
        status = 200
        if range_header.startswith("bytes="):
            try:
                raw_range = range_header[6:].split(",", 1)[0].strip()
                left, right = raw_range.split("-", 1)
                if left:
                    start = int(left)
                    end = int(right) if right else file_size - 1
                else:
                    suffix_length = int(right)
                    if suffix_length <= 0:
                        raise ValueError("invalid suffix range")
                    start = max(0, file_size - suffix_length)
                if start < 0 or start >= file_size or end < start:
                    raise ValueError("invalid byte range")
                end = min(end, file_size - 1)
                status = 206
            except (TypeError, ValueError):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
        content_length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        if head_only:
            return
        with target.open("rb") as stream:
            stream.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    break
                remaining -= len(chunk)


def run_server(host: str = "127.0.0.1", port: int = 7860) -> ThreadingHTTPServer:
    _ensure_jobs_loaded()
    _ensure_worker()
    httpd = ThreadingHTTPServer((host, port), VideoMemoWebHandler)
    print(f"VideoTrace Web: http://{host}:{port}", flush=True)
    httpd.serve_forever()
    return httpd


def _submit_job(video_path: Path, query_text: str, mode_id: str) -> dict:
    _ensure_jobs_loaded()
    _ensure_worker()
    job_id = uuid.uuid4().hex
    now = time.time()
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "phase": "queued",
            "progress": 0,
            "message": "任务已进入 GPU 串行队列。",
            "created_at": now,
            "updated_at": now,
            "video_path": str(video_path),
            "query": query_text,
            "vlm_mode": mode_id,
            "result": None,
            "error": "",
            "gpu_safety": {},
            "error_code": "",
            "retryable": False,
            "phase_started_at": now,
            "started_at": None,
            "completed_at": None,
            "worker_pid": os.getpid(),
            "events": [
                {
                    "at": now,
                    "status": "queued",
                    "phase": "queued",
                    "progress": 0,
                    "message": "任务已进入 GPU 串行队列。",
                }
            ],
        }
        snapshot = dict(_JOBS[job_id])
    _persist_job_snapshot(snapshot)
    _JOB_QUEUE.put(job_id)
    return _job_public(job_id) or {}


def _ensure_worker() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        threading.Thread(target=_job_worker, name="videotrace-gpu-worker", daemon=True).start()
        _WORKER_STARTED = True


def _job_worker() -> None:
    global _ACTIVE_JOB_ID
    while True:
        job_id = _JOB_QUEUE.get()
        _ACTIVE_JOB_ID = job_id
        try:
            _execute_job(job_id)
        finally:
            _ACTIVE_JOB_ID = ""
            _JOB_QUEUE.task_done()


def _execute_job(job_id: str) -> None:
    job = _job_internal(job_id)
    if job is None:
        return
    try:
        _update_job(job_id, status="running", phase="checking_resources", progress=5, message="正在复核 GPU 安全状态。")
        gpu_report = _stable_gpu_safety_check()
        _update_job(job_id, gpu_safety=gpu_report)
        if not gpu_report.get("safe", False):
            raise RuntimeError(gpu_report.get("message") or "所选 GPU 当前不再空闲，任务未启动。")

        base_config = _prepare_runtime_config(_load_runtime_config())
        selected_config, selected_mode = apply_vlm_mode(base_config, job["vlm_mode"])
        _update_job(job_id, phase="loading_models", progress=15, message=f"正在复用或加载 {selected_mode.label}。")
        with _PIPELINE_LOCK:
            pipeline = _pipeline_for_config(selected_config)
            _update_job(job_id, phase="analyzing", progress=35, message="正在切片、检索并生成证据回答。")
            heartbeat_stop = threading.Event()
            heartbeat = threading.Thread(
                target=_analysis_heartbeat,
                args=(job_id, heartbeat_stop),
                name=f"videotrace-job-heartbeat-{job_id[:8]}",
                daemon=True,
            )
            heartbeat.start()
            try:
                demo_path = pipeline.run_and_export(job["video_path"], query=job["query"])
            finally:
                heartbeat_stop.set()
                heartbeat.join(timeout=1.0)
        _update_job(job_id, phase="exporting", progress=90, message="正在校验并整理知识包。")
        pack_path = demo_path.parent / "knowledge_pack.json"
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        handler = object.__new__(VideoMemoWebHandler)
        result = handler._pack_to_response(pack, demo_path)
        result["job"] = {"job_id": job_id, "vlm_mode": selected_mode.public(), "gpu_safety": gpu_report}
        _update_job(
            job_id,
            status="completed",
            phase="completed",
            progress=100,
            message="分析完成。",
            result=result,
        )
    except Exception as exc:
        traceback.print_exc()
        _update_job(
            job_id,
            status="failed",
            phase="failed",
            progress=100,
            message="分析未完成。",
            error=f"{type(exc).__name__}: {exc}",
            error_code=_job_error_code(exc),
            retryable=True,
        )


def _job_internal(job_id: str) -> dict | None:
    _ensure_jobs_loaded()
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def _job_public(job_id: str) -> dict | None:
    _ensure_jobs_loaded()
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        public = _public_job_locked(job, include_result=True)
        return public


def _public_job_locked(job: dict, include_result: bool) -> dict:
        hidden = {"video_path", "query"}
        if not include_result:
            hidden.add("result")
        public = {key: value for key, value in job.items() if key not in hidden}
        public["ok"] = job["status"] != "failed"
        public["poll_url"] = f"/api/jobs/{job['job_id']}"
        public["elapsed_sec"] = round(_job_elapsed_seconds(job), 1)
        queued = sorted(
            (item for item in _JOBS.values() if item.get("status") == "queued"),
            key=lambda item: float(item.get("created_at", 0.0)),
        )
        public["queue_position"] = next(
            (index for index, item in enumerate(queued, start=1) if item.get("job_id") == job.get("job_id")),
            0,
        )
        public["persistence"] = {
            "durable": True,
            "restored": bool(job.get("restored_at")),
        }
        return public


def _job_elapsed_seconds(job: dict, *, now: float | None = None) -> float:
    """Return queue or execution time without letting terminal jobs keep aging."""

    def timestamp(name: str) -> float | None:
        value = job.get(name)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    status = str(job.get("status") or "")
    created_at = timestamp("created_at")
    started_at = timestamp("started_at")
    start = created_at if status == "queued" else (started_at or created_at)
    if start is None:
        return 0.0

    if status in {"completed", "failed"}:
        end = timestamp("completed_at") or timestamp("updated_at") or start
    else:
        end = time.time() if now is None else float(now)
    return max(0.0, end - start)


def _update_job(job_id: str, **values) -> None:
    _ensure_jobs_loaded()
    snapshot = None
    with _JOBS_LOCK:
        if job_id not in _JOBS:
            return
        job = _JOBS[job_id]
        now = time.time()
        previous_status = job.get("status")
        previous_phase = job.get("phase")
        job.update(values)
        job["updated_at"] = now
        job["worker_pid"] = os.getpid()
        if previous_status != job.get("status") and job.get("status") == "running" and not job.get("started_at"):
            job["started_at"] = now
        if previous_phase != job.get("phase"):
            job["phase_started_at"] = now
        if job.get("status") in {"completed", "failed"}:
            job["completed_at"] = job.get("completed_at") or now
        if previous_status != job.get("status") or previous_phase != job.get("phase"):
            event = {
                "at": now,
                "status": job.get("status"),
                "phase": job.get("phase"),
                "progress": job.get("progress"),
                "message": job.get("message", ""),
            }
            if job.get("error_code"):
                event["error_code"] = job["error_code"]
            job["events"] = [*list(job.get("events") or []), event][-64:]
        snapshot = dict(job)
    if snapshot is not None:
        _persist_job_snapshot(snapshot)


def _ensure_jobs_loaded() -> None:
    global _JOB_STATE_ROOT, _JOB_LOAD_WARNINGS
    current_root = str(ROOT.resolve())
    if _JOB_STATE_ROOT == current_root:
        return
    with _JOB_STATE_LOCK:
        if _JOB_STATE_ROOT == current_root:
            return
        loaded, warnings = load_jobs(ROOT)
        now = time.time()
        recovered: list[dict] = []
        for job in loaded.values():
            if job.get("status") in {"queued", "running"}:
                job.update(
                    {
                        "status": "failed",
                        "phase": "failed",
                        "progress": 100,
                        "message": "服务重启时任务仍未完成，已安全终止；请重新提交。",
                        "error": "service_restarted: previous worker stopped before producing a completed result",
                        "error_code": "service_restarted",
                        "retryable": True,
                        "updated_at": now,
                        "completed_at": now,
                        "restored_at": now,
                    }
                )
                job["events"] = [
                    *list(job.get("events") or []),
                    {
                        "at": now,
                        "status": "failed",
                        "phase": "failed",
                        "progress": 100,
                        "message": job["message"],
                        "error_code": "service_restarted",
                    },
                ][-64:]
                recovered.append(dict(job))
            else:
                job["restored_at"] = now
        with _JOBS_LOCK:
            _JOBS.clear()
            _JOBS.update(loaded)
        _JOB_LOAD_WARNINGS = warnings
        _JOB_STATE_ROOT = current_root
        for job in recovered:
            _persist_job_snapshot(job)


def _persist_job_snapshot(job: dict) -> None:
    try:
        persist_job(ROOT, job)
    except (OSError, TypeError, ValueError) as exc:
        print(f"VideoTrace job persistence warning: {type(exc).__name__}: {exc}", flush=True)


def _list_jobs(limit: int = 20) -> list[dict]:
    _ensure_jobs_loaded()
    with _JOBS_LOCK:
        ordered = sorted(_JOBS.values(), key=lambda item: float(item.get("created_at", 0.0)), reverse=True)
        return [_public_job_locked(job, include_result=False) for job in ordered[: max(1, int(limit))]]


def _job_service_status() -> dict:
    _ensure_jobs_loaded()
    with _JOBS_LOCK:
        counts = {
            status: sum(1 for job in _JOBS.values() if job.get("status") == status)
            for status in ("queued", "running", "completed", "failed")
        }
    store = job_store_dir(ROOT)
    return {
        "policy": "serial",
        "active_job_id": _ACTIVE_JOB_ID,
        "counts": counts,
        "persistence": {
            "enabled": True,
            "project_scoped": True,
            "directory": str(store.relative_to(ROOT.resolve())),
            "load_warnings": list(_JOB_LOAD_WARNINGS),
        },
    }


def _job_error_code(exc: Exception) -> str:
    message = str(exc).lower()
    if "gpu" in message and ("占用" in str(exc) or "空闲" in str(exc) or "compute" in message):
        return "gpu_unavailable"
    if isinstance(exc, ValueError) and "视觉模式" in str(exc):
        return "invalid_vlm_mode"
    return "execution_error"


def _analysis_heartbeat(job_id: str, stop: threading.Event) -> None:
    started = time.monotonic()
    while not stop.wait(5.0):
        job = _job_internal(job_id)
        if job is None or job.get("status") != "running" or job.get("phase") != "analyzing":
            return
        elapsed = max(0, int(time.monotonic() - started))
        _update_job(
            job_id,
            progress=_analysis_progress(elapsed),
            message=f"正在切片、检索并生成证据回答（本阶段已运行 {elapsed} 秒）。",
        )


def _analysis_progress(elapsed_seconds: int | float) -> int:
    # Time-based liveness only: never reaches the exporting/completed range.
    return min(85, 35 + max(0, int(float(elapsed_seconds))) // 10 * 3)


def _capabilities() -> dict:
    payload = capability_payload(_prepare_runtime_config(_load_runtime_config()))
    payload["queue"].update(_job_service_status())
    payload["upload"] = {
        "max_bytes": _max_upload_bytes(),
        "extensions": sorted(VIDEO_SUFFIXES),
        "destination": "project_upload_directory",
    }
    return payload


def _health_payload() -> dict:
    return {
        "ok": True,
        "product_version": 3,
        "root": str(ROOT),
        "source_sha256": _service_source_sha256(str(ROOT.resolve())),
        "defaults": _runtime_defaults(),
        "service": _capabilities(),
        "jobs": _job_service_status(),
    }


@lru_cache(maxsize=4)
def _service_source_sha256(root: str) -> str:
    """Hash one deployed source tree once per process, not once per poll."""

    return source_fingerprint(Path(root))


def _product_payload() -> dict:
    capabilities = _capabilities()
    return {
        "stack": "Qwen3.5 + SigLIP2 + neural reranker",
        "model_selection_locked": True,
        "evidence_playback": "source_video_window",
        "analysis_enabled": capabilities["analysis_available"],
        "service_state": capabilities["state"],
        "available_vlm_modes": capabilities["vlm_modes"],
    }


def _media_url(path: str) -> str:
    if not path:
        return ""
    return "/media?path=" + quote(str(Path(path).resolve()), safe="")


def _time_window_url(path: str, start_sec: object, end_sec: object) -> str:
    if not path:
        return ""
    start = max(0.0, float(start_sec or 0.0))
    end = max(start, float(end_sec or start))
    return f"{_media_url(path)}#t={start:.3f},{end:.3f}"


def _artifact_url(path: Path) -> str:
    target = path.resolve()
    try:
        relative = target.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return ""
    return "/artifact/" + quote(relative, safe="/")


def _safe_upload_name(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        return ""
    stem = Path(filename).stem or "video"
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("._-") or "video"
    return f"{stem[:80]}{suffix}"


def _video_id(path: Path | None) -> str:
    if path is None:
        return ""
    target = Path(path).resolve()
    data_root = (ROOT / "data").resolve()
    try:
        return target.relative_to(data_root).as_posix()
    except ValueError:
        return ""


def _resolve_video_id(video_id: str) -> Path | None:
    normalized = str(video_id or "").replace("\\", "/").lstrip("/")
    target = (ROOT / "data" / normalized).resolve()
    data_root = (ROOT / "data").resolve()
    if not target.is_file() or not _is_within(target, data_root) or target.suffix.lower() not in VIDEO_SUFFIXES:
        return None
    return target


def _is_within(target: Path, root: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


def _agent_score_from_json(agent_run: dict) -> dict:
    verification = agent_run.get("verification", {})
    tool_trace = agent_run.get("tool_trace", [])
    return {
        "verified": 1.0 if agent_run.get("verified") else 0.0,
        "evidence_reference_coverage": float(verification.get("coverage", 0.0)),
        "tool_success_rate": (
            sum(1 for call in tool_trace if call.get("ok", True)) / len(tool_trace)
            if tool_trace
            else 0.0
        ),
        "tool_call_count": len(tool_trace),
    }


def _load_runtime_config() -> VideoMemoConfig:
    config_path = os.environ.get("VIDEOTRACE_CONFIG", "").strip()
    return VideoMemoConfig.load(config_path or None)


def _prepare_runtime_config(config: VideoMemoConfig) -> VideoMemoConfig:
    config.output_dir = str((ROOT / "outputs" / "runs" / "latest").resolve())
    for field_name in (
        "scorer_model_path",
        "segment_understanding_cache_dir",
        "asr_cache_dir",
        "vlm_cache_dir",
        "dense_index_dir",
        "reranker_model_path",
        "persistent_memory_path",
    ):
        value = str(getattr(config, field_name, "") or "")
        if value and not Path(value).is_absolute():
            setattr(config, field_name, str((ROOT / value).resolve()))
    # CLI and Web share the same hash-bound frozen-evaluation admission gate.
    if not config.llm_adapter_path:
        config.llm_adapter_path = resolve_validated_adapter(ROOT)
    return config


def _runtime_defaults() -> dict:
    config = _load_runtime_config()
    return {
        "segment_understanding_backend": config.segment_understanding_backend,
        "vlm_backend": config.vlm_backend,
        "llm_backend": config.llm_backend,
        "stack_label": "Qwen3.5 + SigLIP2 + neural reranker",
        "model_selection_locked": True,
    }


def _preferred_pack_path() -> Path:
    candidates = (
        ROOT / "outputs" / "iboy_qwen35" / "cola_review" / "knowledge_pack.json",
        ROOT / "outputs" / "cola_review_qwen35" / "knowledge_pack.json",
        ROOT / "outputs" / "sample" / "knowledge_pack.json",
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[-1])


def _preferred_sample_video() -> Path | None:
    candidates = (
        ROOT / "data" / "raw" / "cola_review.mp4",
        ROOT / "data" / "raw" / "sample.mp4",
    )
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _pipeline_for_config(config: VideoMemoConfig) -> VideoMemoPipeline:
    key = json.dumps(config.dump(), ensure_ascii=True, sort_keys=True, default=str)
    if key not in _PIPELINES:
        _PIPELINES[key] = VideoMemoPipeline(config)
    return _PIPELINES[key]


def _max_upload_bytes() -> int:
    raw = os.environ.get("VIDEOTRACE_MAX_UPLOAD_MIB", "2048")
    try:
        value = max(1, int(raw))
    except ValueError:
        value = 2048
    return value * 1024 * 1024


def _content_length(raw: str, maximum: int) -> int:
    try:
        length = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("请求缺少有效的 Content-Length。") from exc
    if length <= 0:
        raise ValueError("请求内容为空。")
    if length > maximum:
        raise ValueError("请求内容超过服务限制。")
    return length


def _receive_multipart_file(
    source,
    *,
    content_type: str,
    content_length: int,
    target: Path,
    maximum: int,
) -> tuple[str, int]:
    boundary = _multipart_boundary(content_type)
    delimiter = b"--" + boundary
    reader = _ContentLengthReader(source, content_length)
    first_line = reader.readline(MULTIPART_HEADER_BYTES + 1)
    if _trim_multipart_line(first_line) != delimiter:
        raise MultipartUploadError("multipart 起始边界无效。")

    filename = ""
    written = 0
    found_file = False
    final_boundary = False
    while not final_boundary:
        headers = _read_multipart_headers(reader)
        disposition = headers.get_content_disposition()
        field_name = str(headers.get_param("name", header="content-disposition") or "")
        part_filename = str(headers.get_filename() or "")
        is_file = disposition == "form-data" and field_name == "file" and bool(part_filename)
        if part_filename and not is_file:
            raise MultipartUploadError("multipart 仅允许名为 file 的视频文件字段。")
        if is_file and found_file:
            raise MultipartUploadError("一次请求只能上传一个视频文件。")

        if is_file:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                written, final_boundary = _copy_multipart_part(
                    reader,
                    delimiter,
                    output,
                    maximum=maximum,
                )
            filename = part_filename
            found_file = True
        else:
            _, final_boundary = _copy_multipart_part(
                reader,
                delimiter,
                None,
                maximum=maximum,
            )

    if not found_file:
        raise MultipartUploadError("请选择有效的视频文件。")
    return filename, written


def _multipart_boundary(content_type: str) -> bytes:
    try:
        message = BytesHeaderParser(policy=EMAIL_POLICY).parsebytes(
            f"Content-Type: {content_type}\r\n\r\n".encode("latin-1")
        )
    except (UnicodeEncodeError, ValueError) as exc:
        raise MultipartUploadError("multipart Content-Type 无效。") from exc
    boundary = str(message.get_boundary() or "")
    if not boundary or len(boundary) > 200 or "\r" in boundary or "\n" in boundary:
        raise MultipartUploadError("multipart 请求缺少有效 boundary。")
    try:
        return boundary.encode("ascii")
    except UnicodeEncodeError as exc:
        raise MultipartUploadError("multipart boundary 必须为 ASCII。") from exc


def _read_multipart_headers(reader: _ContentLengthReader):
    raw = bytearray()
    while True:
        line = reader.readline(MULTIPART_HEADER_BYTES + 1)
        if not line:
            raise MultipartUploadError("multipart 请求在文件头结束前中断。")
        if len(line) > MULTIPART_HEADER_BYTES or not line.endswith(b"\n"):
            raise MultipartUploadError("multipart 文件头过长。")
        if line in {b"\r\n", b"\n"}:
            break
        raw.extend(line)
        if len(raw) > MULTIPART_HEADER_BYTES:
            raise MultipartUploadError("multipart 文件头超过服务限制。")
    try:
        return BytesHeaderParser(policy=EMAIL_POLICY).parsebytes(bytes(raw) + b"\r\n")
    except ValueError as exc:
        raise MultipartUploadError("multipart 文件头无法解析。") from exc


def _copy_multipart_part(
    reader: _ContentLengthReader,
    delimiter: bytes,
    output,
    *,
    maximum: int,
) -> tuple[int, bool]:
    pending = b""
    written = 0
    while True:
        line = reader.readline(MULTIPART_LINE_BYTES)
        if not line:
            raise MultipartUploadError("multipart 请求在结束边界前中断。")
        marker = _trim_multipart_line(line)
        if pending.endswith(b"\n") and marker in {delimiter, delimiter + b"--"}:
            payload = pending
            if payload.endswith(b"\r\n"):
                payload = payload[:-2]
            elif payload.endswith(b"\n"):
                payload = payload[:-1]
            written = _write_multipart_chunk(output, payload, written, maximum)
            return written, marker.endswith(b"--")
        if pending:
            written = _write_multipart_chunk(output, pending, written, maximum)
        pending = line


def _write_multipart_chunk(output, chunk: bytes, written: int, maximum: int) -> int:
    if output is None:
        return written
    total = written + len(chunk)
    if total > maximum:
        raise ValueError(f"视频超过上传上限（{maximum // (1024 * 1024)} MiB）。")
    if chunk:
        output.write(chunk)
    return total


def _trim_multipart_line(line: bytes) -> bytes:
    if line.endswith(b"\r\n"):
        return line[:-2]
    if line.endswith(b"\n"):
        return line[:-1]
    return line


def _copy_limited(source, target: Path, maximum: int, expected_bytes: int | None = None) -> int:
    written = 0
    with target.open("wb") as output:
        while True:
            if expected_bytes is not None and written >= expected_bytes:
                break
            read_size = min(1024 * 1024, maximum - written + 1)
            if expected_bytes is not None:
                read_size = min(read_size, expected_bytes - written)
            chunk = source.read(read_size)
            if not chunk:
                break
            written += len(chunk)
            if written > maximum:
                raise ValueError(f"视频超过上传上限（{maximum // (1024 * 1024)} MiB）。")
            output.write(chunk)
    if expected_bytes is not None and written != expected_bytes:
        raise ValueError("上传内容不完整。")
    return written


def _stable_gpu_safety_check(checks: int = 3, interval_sec: float = 1.0) -> dict:
    physical = [value.strip() for value in os.environ.get("VIDEOTRACE_PHYSICAL_GPUS", "").split(",") if value.strip()]
    if not physical:
        return {
            "safe": True,
            "enforced": False,
            "message": "未配置物理 GPU 绑定；当前按本地/CPU 服务运行。",
            "snapshots": [],
        }
    snapshots = []
    for index in range(max(1, checks)):
        snapshot = _gpu_snapshot(physical)
        snapshots.append(snapshot)
        if not snapshot.get("safe"):
            return {
                "safe": False,
                "enforced": True,
                "physical_gpu_ids": physical,
                "message": snapshot.get("message", "GPU 已被其他进程占用。"),
                "snapshots": snapshots,
            }
        if index + 1 < checks:
            time.sleep(max(0.1, interval_sec))
    return {
        "safe": True,
        "enforced": True,
        "physical_gpu_ids": physical,
        "message": "连续 GPU 安全探测通过，未发现其他计算进程。",
        "snapshots": snapshots,
    }


def _gpu_snapshot(physical: list[str]) -> dict:
    try:
        gpu_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        process_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"safe": False, "message": f"GPU 状态探测失败：{type(exc).__name__}: {exc}", "gpus": []}

    rows: dict[str, dict] = {}
    uuid_to_index: dict[str, str] = {}
    for raw_line in gpu_output.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 4:
            continue
        rows[parts[0]] = {
            "index": int(parts[0]),
            "uuid": parts[1],
            "memory_used_mib": int(parts[2]),
            "utilization_pct": int(parts[3]),
            "other_compute_pids": [],
        }
        uuid_to_index[parts[1]] = parts[0]
    current_pid = os.getpid()
    for raw_line in process_output.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 2:
            continue
        gpu_index = uuid_to_index.get(parts[0])
        if gpu_index not in rows:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        if pid != current_pid:
            rows[gpu_index]["other_compute_pids"].append(pid)
    selected = [rows.get(index) for index in physical]
    if any(item is None for item in selected):
        return {"safe": False, "message": "配置的物理 GPU 编号不存在。", "gpus": list(rows.values())}
    conflicts = [item for item in selected if item and item["other_compute_pids"]]
    return {
        "safe": not conflicts,
        "message": "GPU 空闲。" if not conflicts else f"检测到其他 GPU 计算进程：{conflicts}",
        "gpus": selected,
        "observed_at": time.time(),
    }
