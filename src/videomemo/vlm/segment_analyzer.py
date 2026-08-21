from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from pathlib import Path
from urllib import request

import cv2
from PIL import Image

from ..models import Segment
from .frame_sampler import sample_frames


PROMPT_VERSION = "qwen35-segment-v1"


class BaselineSegmentAnalyzer:
    backend = "baseline"

    def enrich(self, video_path: str, segments: list[Segment]) -> dict:
        return {
            "enabled": False,
            "backend": self.backend,
            "num_segments": len(segments),
            "records": [
                {
                    "segment_id": segment.segment_id,
                    "status": "baseline_only",
                    "confidence": segment.understanding_confidence,
                }
                for segment in segments
            ],
        }


class QwenVLAPISegmentAnalyzer:
    backend = "qwen_vl_api"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        cache_dir: str = "outputs_cache/segment_understanding",
        num_frames: int = 4,
        timeout_sec: int = 180,
        fail_open: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_frames = max(1, int(num_frames))
        self.timeout_sec = int(timeout_sec)
        self.fail_open = bool(fail_open)

    def enrich(self, video_path: str, segments: list[Segment]) -> dict:
        records: list[dict] = []
        for segment in segments:
            key = self._cache_key(video_path, segment)
            cached = self._load_cache(key)
            if cached is None:
                cached = self._load_cache(self._legacy_cache_key(video_path, segment))
                if cached is not None:
                    self._save_cache(key, cached)
            try:
                analysis = _normalize_analysis(cached) if cached else self._analyze(video_path, segment)
                if cached is None:
                    self._save_cache(key, analysis)
                elif analysis != cached:
                    # Repair legacy cache entries in place.  This is a
                    # reversible, content-addressed update and avoids sending
                    # the same segment back through a large VLM merely because
                    # an older parser stored a JSON-ish response as ``summary``.
                    self._save_cache(key, analysis)
                self._apply(segment, analysis)
                records.append(
                    {
                        "segment_id": segment.segment_id,
                        "status": "cache_hit" if cached else "generated",
                        "confidence": segment.understanding_confidence,
                        "scene": segment.scene,
                        "entities": segment.entities,
                        "actions": segment.actions,
                    }
                )
            except Exception as exc:
                if not self.fail_open:
                    raise
                records.append(
                    {
                        "segment_id": segment.segment_id,
                        "status": "fallback",
                        "error": f"{type(exc).__name__}: {exc}",
                        "confidence": segment.understanding_confidence,
                    }
                )
        return {
            "enabled": True,
            "backend": self.backend,
            "model": self.model,
            "num_segments": len(segments),
            "num_generated": sum(record["status"] == "generated" for record in records),
            "num_cache_hits": sum(record["status"] == "cache_hit" for record in records),
            "num_fallbacks": sum(record["status"] == "fallback" for record in records),
            "records": records,
        }

    def _analyze(self, video_path: str, segment: Segment) -> dict:
        frames = sample_frames(
            video_path,
            segment.start_sec,
            segment.end_sec,
            self.num_frames,
        )
        if not frames:
            raise ValueError(f"no frames sampled for {segment.segment_id}")

        content: list[dict] = [
            {
                "type": "text",
                "text": self._instruction(segment, len(frames)),
            }
        ]
        for index, frame in enumerate(frames, start=1):
            content.append(
                {
                    "type": "text",
                    "text": f"按时间顺序排列的关键帧 {index}/{len(frames)}：",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _frame_data_url(frame)},
                }
            )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是严格的视频片段标注器。只描述关键帧中可见的内容，"
                        "不根据文件名或问题背景猜测。看不清的内容必须留空或写入 uncertainties。"
                    ),
                },
                {"role": "user", "content": content},
            ],
            "max_tokens": 900,
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with request.urlopen(req, timeout=self.timeout_sec) as response:
            result = json.loads(response.read().decode("utf-8"))
        raw = result["choices"][0]["message"]["content"]
        return _parse_analysis(raw)

    @staticmethod
    def _instruction(segment: Segment, num_frames: int) -> str:
        return (
            f"请分析时间窗 {segment.start_sec:.1f}-{segment.end_sec:.1f} 秒的 {num_frames} 张关键帧。\n"
            "输出一个 JSON 对象，不要输出 Markdown，字段必须是：\n"
            '{"summary":"一句可检索的客观描述","ocr_text":"可确认的画面文字，无法确认则为空字符串",'
            '"entities":["人物或物体"],"actions":["可见动作或状态变化"],'
            '"scene":"场景类型","confidence":0.0,"uncertainties":["不能确认的内容"]}\n'
            "summary 必须覆盖主体、动作、场景和明显变化；不得使用可能、似乎、大概来替代证据不足。"
        )

    def _cache_key(self, video_path: str, segment: Segment) -> str:
        raw = json.dumps(
            [
                "content-addressed-v1",
                PROMPT_VERSION,
                self.model,
                segment.segment_id,
                segment.start_sec,
                segment.end_sec,
                segment.frame_hash,
                self.num_frames,
            ],
            ensure_ascii=False,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _legacy_cache_key(self, video_path: str, segment: Segment) -> str:
        path = Path(video_path)
        try:
            stat = path.stat()
            file_signature = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            file_signature = str(path)
        raw = json.dumps(
            [
                PROMPT_VERSION,
                self.model,
                file_signature,
                segment.segment_id,
                segment.start_sec,
                segment.end_sec,
                segment.frame_hash,
                self.num_frames,
            ],
            ensure_ascii=False,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_cache(self, key: str) -> dict | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _save_cache(self, key: str, analysis: dict) -> None:
        path = self.cache_dir / f"{key}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def _apply(self, segment: Segment, analysis: dict) -> None:
        summary = str(analysis.get("summary") or "").strip()
        ocr_text = str(analysis.get("ocr_text") or "").strip()
        entities = _string_list(analysis.get("entities"))
        actions = _string_list(analysis.get("actions"))
        scene = str(analysis.get("scene") or "").strip()
        confidence = _confidence(analysis.get("confidence"))

        searchable = [summary]
        if scene:
            searchable.append(f"场景：{scene}")
        if entities:
            searchable.append(f"人物与物体：{'、'.join(entities)}")
        if actions:
            searchable.append(f"动作与变化：{'、'.join(actions)}")
        segment.caption = summary
        segment.text = " ".join(part for part in searchable if part).strip() or segment.text
        segment.ocr_text = ocr_text
        segment.entities = entities
        segment.actions = actions
        segment.scene = scene
        segment.understanding_confidence = confidence
        segment.understanding_backend = self.backend


class Qwen35LocalSegmentAnalyzer(QwenVLAPISegmentAnalyzer):
    backend = "qwen35_local"

    def __init__(
        self,
        model: str,
        cache_dir: str,
        num_frames: int,
        timeout_sec: int,
        fail_open: bool,
        device: str,
        dtype: str,
        max_new_tokens: int,
    ):
        super().__init__(
            base_url="local",
            model=model,
            api_key="EMPTY",
            cache_dir=cache_dir,
            num_frames=num_frames,
            timeout_sec=timeout_sec,
            fail_open=fail_open,
        )
        from ..llm.qwen35_local import get_qwen35_runtime

        self.runtime = get_qwen35_runtime(model, device=device, dtype=dtype)
        self.max_new_tokens = int(max_new_tokens)

    def _analyze(self, video_path: str, segment: Segment) -> dict:
        frames = sample_frames(
            video_path,
            segment.start_sec,
            segment.end_sec,
            self.num_frames,
        )
        if not frames:
            raise ValueError(f"no frames sampled for {segment.segment_id}")
        content: list[dict] = [
            {
                "type": "text",
                "text": self._instruction(segment, len(frames)),
            }
        ]
        for index, frame in enumerate(frames, start=1):
            content.append(
                {
                    "type": "text",
                    "text": f"按时间顺序排列的关键帧 {index}/{len(frames)}：",
                }
            )
            content.append({"type": "image", "image": Image.fromarray(frame[:, :, ::-1])})
        raw = self.runtime.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "你是严格的视频片段标注器。只描述关键帧中可见的内容，"
                        "不根据文件名或问题背景猜测。看不清的内容必须留空或写入 uncertainties。"
                    ),
                },
                {"role": "user", "content": content},
            ],
            max_new_tokens=self.max_new_tokens,
            temperature=0.2,
            top_p=0.8,
            use_adapter=False,
        )
        return _parse_analysis(raw)


def build_segment_analyzer(
    backend: str,
    base_url: str,
    model: str,
    api_key: str,
    cache_dir: str,
    num_frames: int,
    timeout_sec: int,
    fail_open: bool,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_new_tokens: int = 800,
):
    normalized = (backend or "baseline").strip().lower()
    if normalized in {"baseline", "none", ""}:
        return BaselineSegmentAnalyzer()
    if normalized == "qwen_vl_api":
        if not base_url:
            raise ValueError("segment_understanding_base_url is required for qwen_vl_api")
        if not model:
            raise ValueError("segment_understanding_model is required for qwen_vl_api")
        return QwenVLAPISegmentAnalyzer(
            base_url=base_url,
            model=model,
            api_key=api_key,
            cache_dir=cache_dir,
            num_frames=num_frames,
            timeout_sec=timeout_sec,
            fail_open=fail_open,
        )
    if normalized == "qwen35_local":
        if not model:
            raise ValueError("segment_understanding_model is required for qwen35_local")
        return Qwen35LocalSegmentAnalyzer(
            model=model,
            cache_dir=cache_dir,
            num_frames=num_frames,
            timeout_sec=timeout_sec,
            fail_open=fail_open,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"Unknown segment understanding backend: {backend}")


def _frame_data_url(frame) -> str:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise ValueError("failed to encode sampled frame")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _parse_analysis(raw: object) -> dict:
    if isinstance(raw, list):
        text = "\n".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in raw)
    else:
        text = str(raw or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return _normalize_analysis(data)
        except json.JSONDecodeError:
            pass
    if not text:
        raise ValueError("empty model response")
    return _normalize_analysis({
        "summary": text[:1200],
        "ocr_text": "",
        "entities": [],
        "actions": [],
        "scene": "",
        "confidence": 0.35,
        "uncertainties": ["model response was not valid JSON"],
    })


def _normalize_analysis(value: object) -> dict:
    """Normalize current and legacy VLM records into the segment schema.

    A few older Qwen responses were saved as ``{"summary": "{...}"}`` when
    the model emitted a malformed OCR list.  We salvage the valid fields
    rather than exposing the raw JSON blob as user-facing evidence.
    """
    data = dict(value) if isinstance(value, dict) else {"summary": str(value or "")}
    summary = str(data.get("summary") or "").strip()
    nested = _extract_jsonish_fields(summary) if summary.startswith("{") else {}
    if nested:
        merged = dict(nested)
        for key in ("ocr_text", "entities", "actions", "scene", "confidence", "uncertainties"):
            current = data.get(key)
            # ``0.35`` is the historical fallback used when the old parser
            # failed to decode the nested response; let a salvaged confidence
            # replace that sentinel while preserving genuinely supplied low
            # confidence values.
            is_legacy_confidence = key == "confidence" and _confidence(current) == 0.35
            if current not in (None, "", [], ()) and not is_legacy_confidence:
                merged[key] = current
        data = merged
        summary = str(data.get("summary") or "").strip()
    return {
        "summary": summary[:1200],
        "ocr_text": str(data.get("ocr_text") or "").strip()[:2400],
        "entities": _string_list(data.get("entities")),
        "actions": _string_list(data.get("actions")),
        "scene": str(data.get("scene") or "").strip()[:240],
        "confidence": _confidence(data.get("confidence")),
        "uncertainties": _string_list(data.get("uncertainties")),
    }


def _extract_jsonish_fields(text: str) -> dict:
    """Best-effort extraction for a mostly-JSON model response.

    This intentionally only returns fields with unambiguous delimiters; it
    never invents entities or actions from arbitrary prose.
    """
    result: dict = {}
    summary_match = re.search(r'"summary"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.DOTALL)
    if summary_match:
        result["summary"] = _unescape_json_string(summary_match.group(1))
    ocr_match = re.search(r'"ocr_text"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.DOTALL)
    if ocr_match:
        result["ocr_text"] = _unescape_json_string(ocr_match.group(1))
    for field in ("entities", "actions", "uncertainties"):
        match = re.search(rf'"{field}"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
        if match:
            result[field] = [
                _unescape_json_string(item)
                for item in re.findall(r'"((?:\\.|[^"\\])*)"', match.group(1), flags=re.DOTALL)
            ]
    scene_match = re.search(r'"scene"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.DOTALL)
    if scene_match:
        result["scene"] = _unescape_json_string(scene_match.group(1))
    confidence_match = re.search(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if confidence_match:
        result["confidence"] = float(confidence_match.group(1))
    return result if result.get("summary") else {}


def _unescape_json_string(value: str) -> str:
    try:
        return str(json.loads('"' + value + '"'))
    except (json.JSONDecodeError, TypeError):
        return value.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    result: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text[:160])
    return result[:20]


def _confidence(value: object) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
