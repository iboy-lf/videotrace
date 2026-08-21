from __future__ import annotations

import json
import queue
import time

from videomemo.web import server
from videomemo.web.job_store import load_jobs, persist_job


def _reset_runtime(monkeypatch, root) -> None:
    monkeypatch.setattr(server, "ROOT", root)
    monkeypatch.setattr(server, "_JOB_STATE_ROOT", "")
    monkeypatch.setattr(server, "_JOB_LOAD_WARNINGS", [])
    monkeypatch.setattr(server, "_JOBS", {})
    monkeypatch.setattr(server, "_JOB_QUEUE", queue.Queue())
    monkeypatch.setattr(server, "_WORKER_STARTED", False)
    monkeypatch.setattr(server, "_ACTIVE_JOB_ID", "")


def test_completed_job_is_restored_and_inflight_job_becomes_retryable_failure(monkeypatch, tmp_path):
    now = time.time() - 10
    completed = {
        "job_id": "a" * 32,
        "status": "completed",
        "phase": "completed",
        "progress": 100,
        "message": "分析完成。",
        "created_at": now,
        "updated_at": now,
        "video_path": str(tmp_path / "data/raw/demo.mp4"),
        "query": "demo",
        "vlm_mode": "auto_best",
        "result": {"ready": True, "answer": "ok"},
        "error": "",
        "events": [],
    }
    running = {
        **completed,
        "job_id": "b" * 32,
        "status": "running",
        "phase": "analyzing",
        "progress": 53,
        "result": None,
    }
    persist_job(tmp_path, completed)
    persist_job(tmp_path, running)
    _reset_runtime(monkeypatch, tmp_path)

    restored = server._job_public(completed["job_id"])
    interrupted = server._job_public(running["job_id"])

    assert restored is not None and restored["result"]["answer"] == "ok"
    assert restored["persistence"] == {"durable": True, "restored": True}
    assert interrupted is not None
    assert interrupted["status"] == "failed"
    assert interrupted["error_code"] == "service_restarted"
    assert interrupted["retryable"] is True
    disk_jobs, warnings = load_jobs(tmp_path)
    assert not warnings
    assert disk_jobs[running["job_id"]]["status"] == "failed"


def test_queue_positions_phase_events_and_public_list_are_durable(monkeypatch, tmp_path):
    data = tmp_path / "data" / "uploads"
    data.mkdir(parents=True)
    video = data / "demo.mp4"
    video.write_bytes(b"video")
    _reset_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "_ensure_worker", lambda: None)

    first = server._submit_job(video, "question one", "auto_best")
    second = server._submit_job(video, "question two", "auto_best")
    assert first["queue_position"] == 1
    assert second["queue_position"] == 2

    server._update_job(
        first["job_id"],
        status="running",
        phase="checking_resources",
        progress=5,
        message="checking",
    )
    assert server._job_public(second["job_id"])["queue_position"] == 1
    listing = server._list_jobs(10)
    assert len(listing) == 2
    assert all("query" not in item and "video_path" not in item and "result" not in item for item in listing)
    first_internal = server._job_internal(first["job_id"])
    assert [event["phase"] for event in first_internal["events"]] == ["queued", "checking_resources"]

    stored = json.loads(
        (tmp_path / "outputs" / "runs" / "latest" / "jobs" / f"{first['job_id']}.json").read_text(encoding="utf-8")
    )
    assert stored["job"]["phase"] == "checking_resources"
    status = server._job_service_status()
    assert status["counts"] == {"queued": 1, "running": 1, "completed": 0, "failed": 0}
    assert status["persistence"]["project_scoped"] is True


def test_completed_job_elapsed_is_execution_time_and_stops_aging(monkeypatch, tmp_path):
    data = tmp_path / "data" / "uploads"
    data.mkdir(parents=True)
    video = data / "demo.mp4"
    video.write_bytes(b"video")
    _reset_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "_ensure_worker", lambda: None)
    clock = [100.0]
    monkeypatch.setattr(server.time, "time", lambda: clock[0])

    submitted = server._submit_job(video, "question", "auto_best")
    clock[0] = 105.0
    server._update_job(
        submitted["job_id"],
        status="running",
        phase="analyzing",
        progress=50,
        message="running",
    )
    clock[0] = 112.0
    server._update_job(
        submitted["job_id"],
        status="completed",
        phase="completed",
        progress=100,
        message="done",
    )
    clock[0] = 10_000.0

    completed = server._job_public(submitted["job_id"])

    assert completed is not None
    assert completed["elapsed_sec"] == 7.0
    assert completed["started_at"] == 105.0
    assert completed["completed_at"] == 112.0


def test_job_store_rejects_path_outside_project(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEOTRACE_JOB_STORE", str(tmp_path.parent / "outside-jobs"))
    job = {"job_id": "c" * 32}
    try:
        persist_job(tmp_path, job)
    except ValueError as exc:
        assert "项目目录内" in str(exc)
    else:
        raise AssertionError("outside job store should be rejected")
