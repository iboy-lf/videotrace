from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


def build_run_metadata(config, manifest_path: str, split: str | None = None) -> dict:
    root = Path(__file__).resolve().parents[3]
    config_payload = asdict(config)
    manifest = Path(manifest_path).resolve()
    source_sha256 = source_fingerprint(root)
    manifest_sha256 = file_sha256(manifest)
    signature_payload = {
        "source_sha256": source_sha256,
        "manifest_sha256": manifest_sha256,
        "split": split or "all",
        "config": config_payload,
    }
    run_signature = hashlib.sha256(
        json.dumps(signature_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_signature": run_signature,
        "source_sha256": source_sha256,
        "manifest_sha256": manifest_sha256,
        "manifest_path": str(manifest),
        "split": split or "all",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "cuda": _cuda_snapshot(),
        "config": config_payload,
    }


def source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    roots = [root / "src", root / "scripts", root / "configs", root / "tests"]
    extra = [root / "pyproject.toml", root / "README.md"]
    files: list[Path] = []
    for directory in roots:
        if directory.exists():
            files.extend(
                path
                for path in directory.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
            )
    files.extend(path for path in extra if path.exists())
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_pack_sha256(path: Path) -> str:
    """Hash the stable knowledge-pack content independently of admission metadata.

    The Web admission record is embedded in ``metadata.llm_adapter``.  Hashing
    that field would create a self-referential loop: writing a new admission
    changes the pack hash which invalidates the admission itself.  We therefore
    bind evaluations to the canonical JSON payload with that one dynamic field
    removed.  The raw file SHA remains available from ``file_sha256`` for
    artifact-manifest integrity checks.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            metadata = dict(metadata)
            metadata.pop("llm_adapter", None)
            payload = dict(payload)
            payload["metadata"] = metadata
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "cuda": _cuda_snapshot(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "physical_gpu_ids": os.environ.get("VIDEOTRACE_PHYSICAL_GPUS", ""),
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ["numpy", "opencv-python", "opencv-python-headless", "scikit-learn", "torch", "transformers"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _cuda_snapshot() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False, "device_count": 0, "devices": []}
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_mib": round(props.total_memory / (1024**2), 1),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
        return {"available": True, "device_count": len(devices), "devices": devices}
    except Exception as exc:
        return {"available": False, "device_count": 0, "devices": [], "error": f"{type(exc).__name__}: {exc}"}
