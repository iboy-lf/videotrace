from __future__ import annotations

from pathlib import Path
import json
import os
import re
import time
import uuid


JOB_STORE_SCHEMA_VERSION = "videotrace-web-job-v1"
_SAFE_JOB_ID = re.compile(r"^[0-9A-Za-z_-]{8,64}$")


def job_store_dir(root: Path) -> Path:
    """Return a project-scoped directory for durable Web job records."""

    project_root = Path(root).resolve()
    configured = str(os.environ.get("VIDEOTRACE_JOB_STORE", "outputs/runs/latest/jobs") or "").strip()
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    target = candidate.resolve()
    try:
        target.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("VIDEOTRACE_JOB_STORE 必须位于项目目录内。") from exc
    return target


def persist_job(root: Path, job: dict) -> Path:
    """Atomically persist one serializable job without exposing partial JSON."""

    job_id = str(job.get("job_id") or "")
    if not _SAFE_JOB_ID.fullmatch(job_id):
        raise ValueError("invalid job id")
    directory = job_store_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{job_id}.json"
    temporary = directory / f".{job_id}.{uuid.uuid4().hex}.tmp"
    payload = {
        "schema_version": JOB_STORE_SCHEMA_VERSION,
        "persisted_at": time.time(),
        "job": job,
    }
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def load_jobs(root: Path, limit: int = 200) -> tuple[dict[str, dict], list[str]]:
    """Load recent valid jobs; malformed files are reported and ignored."""

    directory = job_store_dir(root)
    if not directory.is_dir():
        return {}, []
    jobs: dict[str, dict] = {}
    warnings: list[str] = []
    candidates = sorted(
        directory.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[: max(1, int(limit))]
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != JOB_STORE_SCHEMA_VERSION:
                raise ValueError("unsupported schema")
            job = payload.get("job")
            if not isinstance(job, dict):
                raise ValueError("missing job payload")
            job_id = str(job.get("job_id") or "")
            if not _SAFE_JOB_ID.fullmatch(job_id) or path.stem != job_id:
                raise ValueError("job id mismatch")
            jobs[job_id] = job
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            warnings.append(f"{path.name}: {type(exc).__name__}: {exc}")
    return jobs, warnings
