from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


_VIDEO_CONTENT_HASH_CACHE: dict[tuple[str, int, int], str] = {}


@dataclass
class TranscriptSpan:
    start_sec: float
    end_sec: float
    text: str

    def dump(self) -> dict:
        return {
            "start_sec": round(float(self.start_sec), 3),
            "end_sec": round(float(self.end_sec), 3),
            "text": self.text,
        }


class DisabledASR:
    backend = "none"

    def transcribe(self, video_path: str) -> list[TranscriptSpan]:
        return []


class SidecarASR:
    backend = "sidecar"

    def transcribe(self, video_path: str) -> list[TranscriptSpan]:
        sidecar = _find_sidecar(Path(video_path))
        if sidecar is None:
            return []
        if sidecar.suffix.lower() == ".json":
            return _parse_json_sidecar(sidecar)
        return _parse_caption_sidecar(sidecar)


class FasterWhisperASR:
    backend = "faster_whisper"
    _models: dict[tuple[str, str, str], Any] = {}

    def __init__(self, model: str, device: str, compute_type: str, language: str):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language or None

    def transcribe(self, video_path: str) -> list[TranscriptSpan]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "asr_backend=faster_whisper requires the optional faster-whisper package"
            ) from exc
        key = (self.model_name, self.device, self.compute_type)
        model = self._models.get(key)
        if model is None:
            model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            self._models[key] = model
        segments, _info = model.transcribe(
            video_path,
            language=self.language,
            vad_filter=True,
            beam_size=1,
        )
        result: list[TranscriptSpan] = []
        for item in segments:
            text = str(getattr(item, "text", "") or "").strip()
            start = float(getattr(item, "start", 0.0) or 0.0)
            end = float(getattr(item, "end", start) or start)
            if text and end > start:
                result.append(TranscriptSpan(start, end, text))
        return result


def build_asr_backend(
    backend: str,
    model: str = "",
    device: str = "cuda",
    compute_type: str = "float16",
    language: str = "zh",
) -> DisabledASR | SidecarASR | FasterWhisperASR:
    normalized = (backend or "none").strip().lower()
    if normalized in {"", "none", "disabled"}:
        return DisabledASR()
    if normalized == "sidecar":
        return SidecarASR()
    if normalized == "auto":
        return SidecarASR() if model == "" else FasterWhisperASR(model, device, compute_type, language)
    if normalized in {"faster_whisper", "faster-whisper"}:
        if not model:
            raise ValueError("asr_model is required for asr_backend=faster_whisper")
        return FasterWhisperASR(model, device, compute_type, language)
    raise ValueError(f"Unknown ASR backend: {backend}")


def enrich_segments_with_asr(
    video_path: str,
    segments: list,
    backend: DisabledASR | SidecarASR | FasterWhisperASR,
    cache_dir: str,
    fail_open: bool = True,
) -> dict:
    for segment in segments:
        segment.asr_text = ""
    if backend.backend == "none":
        return {
            "enabled": False,
            "backend": "none",
            "num_spans": 0,
            "num_segments_with_asr": 0,
            "cache": "disabled",
        }

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{_cache_key(video_path, backend)}.json"
    cache_status = "miss"
    try:
        spans = _load_spans(cache_path)
        if spans is not None:
            cache_status = "hit"
        else:
            spans = backend.transcribe(video_path)
            _save_spans(cache_path, spans)
        for segment in segments:
            matching = [
                span.text
                for span in spans
                if span.end_sec > float(segment.start_sec)
                and span.start_sec < float(segment.end_sec)
            ]
            segment.asr_text = " ".join(_dedupe_text(matching))[:1600]
        return {
            "enabled": bool(spans),
            "backend": backend.backend,
            "num_spans": len(spans),
            "num_segments_with_asr": sum(bool(segment.asr_text) for segment in segments),
            "cache": cache_status,
            "cache_path": str(cache_path),
        }
    except Exception as exc:
        if not fail_open:
            raise
        return {
            "enabled": False,
            "backend": backend.backend,
            "num_spans": 0,
            "num_segments_with_asr": 0,
            "cache": cache_status,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _cache_key(video_path: str, backend: Any) -> str:
    path = Path(video_path)
    try:
        stat = path.stat()
        stat_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        signature = _VIDEO_CONTENT_HASH_CACHE.get(stat_key, "")
        if not signature:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            signature = digest.hexdigest()
            _VIDEO_CONTENT_HASH_CACHE[stat_key] = signature
    except OSError:
        signature = str(path)
    sidecar_signature = ""
    if getattr(backend, "backend", "") == "sidecar":
        sidecar = _find_sidecar(path)
        if sidecar is not None:
            try:
                sidecar_signature = hashlib.sha256(sidecar.read_bytes()).hexdigest()
            except OSError:
                sidecar_signature = str(sidecar)
    raw = json.dumps(
        [
            signature,
            backend.backend,
            getattr(backend, "model_name", ""),
            getattr(backend, "language", ""),
            getattr(backend, "compute_type", ""),
            sidecar_signature,
        ],
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_spans(path: Path) -> list[TranscriptSpan] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return [TranscriptSpan(float(item["start_sec"]), float(item["end_sec"]), str(item["text"])) for item in data]


def _save_spans(path: Path, spans: list[TranscriptSpan]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps([span.dump() for span in spans], ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _find_sidecar(video_path: Path) -> Path | None:
    for suffix in (".srt", ".vtt", ".json"):
        candidate = video_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _parse_json_sidecar(path: Path) -> list[TranscriptSpan]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_items = data.get("segments", data) if isinstance(data, dict) else data
    result = []
    for item in raw_items if isinstance(raw_items, list) else []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        start = item.get("start_sec", item.get("start", 0.0))
        end = item.get("end_sec", item.get("end", start))
        if float(end) > float(start):
            result.append(TranscriptSpan(float(start), float(end), text))
    return sorted(result, key=lambda item: item.start_sec)


def _parse_caption_sidecar(path: Path) -> list[TranscriptSpan]:
    raw = path.read_text(encoding="utf-8-sig", errors="ignore").replace("\r", "")
    blocks = re.split(r"\n\s*\n", raw)
    result = []
    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        timing = next((line for line in lines if "-->" in line), "")
        if not timing:
            continue
        left, right = [part.strip().split(" ", 1)[0] for part in timing.split("-->", 1)]
        text_lines = [line for line in lines[lines.index(timing) + 1 :] if not line.startswith("NOTE")]
        text = " ".join(text_lines).strip()
        start, end = _caption_time(left), _caption_time(right)
        if text and end > start:
            result.append(TranscriptSpan(start, end, re.sub(r"<[^>]+>", "", text)))
    return sorted(result, key=lambda item: item.start_sec)


def _caption_time(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600.0 + float(minutes) * 60.0 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60.0 + float(seconds)
    return float(parts[0])


def _dedupe_text(items: list[str]) -> list[str]:
    result = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in result:
            result.append(value)
    return result
