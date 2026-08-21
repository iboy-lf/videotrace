from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
import hashlib
import re


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def resolve_pack_video(pack: Mapping[str, object], root: Path) -> Path | None:
    """Resolve a knowledge pack's source video inside the current checkout.

    Canonical packs produced on ``iboy`` intentionally retain their remote
    absolute ``video_path`` for provenance.  A local read-only launch must not
    rewrite that immutable artifact, so this resolver maps it to a project
    video only when the recorded SHA-256 proves that the bytes are identical.
    """

    project_root = root.resolve()
    declared = str(pack.get("video_path") or "").strip()
    metadata = pack.get("metadata")
    expected_sha = ""
    if isinstance(metadata, Mapping):
        expected_sha = str(metadata.get("video_sha256") or "").strip().lower()
    if not SHA256_PATTERN.fullmatch(expected_sha):
        expected_sha = ""

    direct = _declared_path(declared, project_root)
    if _is_project_video(direct, project_root) and _matches_expected_sha(direct, expected_sha):
        return direct

    # A remote absolute path may not exist on the presentation laptop.  Only
    # inspect the project-owned raw-video directory, trying the same basename
    # first and accepting a candidate solely on a hash match.
    if not expected_sha:
        return None
    raw_root = (project_root / "data" / "raw").resolve()
    candidates: list[Path] = []
    declared_name = Path(declared).name if declared else ""
    if declared_name:
        candidates.append(raw_root / declared_name)
    if raw_root.is_dir():
        candidates.extend(
            path
            for path in sorted(raw_root.rglob("*"))
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )

    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved).casefold()
        if key in seen:
            continue
        seen.add(key)
        if _is_project_video(resolved, project_root) and _matches_expected_sha(resolved, expected_sha):
            return resolved
    return None


def pack_has_playable_video(pack_path: Path, root: Path) -> bool:
    try:
        import json

        payload = json.loads(pack_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return False
    return isinstance(payload, Mapping) and resolve_pack_video(payload, root) is not None


def _declared_path(value: str, project_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _is_project_video(path: Path | None, project_root: Path) -> bool:
    if path is None or not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
        return False
    try:
        path.relative_to(project_root)
        return True
    except ValueError:
        return False


def _matches_expected_sha(path: Path, expected_sha: str) -> bool:
    if not expected_sha:
        return True
    stat = path.stat()
    return _cached_file_sha256(str(path), stat.st_size, stat.st_mtime_ns) == expected_sha


@lru_cache(maxsize=32)
def _cached_file_sha256(path: str, size: int, mtime_ns: int) -> str:
    # ``size`` and ``mtime_ns`` are cache-key integrity guards.
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
