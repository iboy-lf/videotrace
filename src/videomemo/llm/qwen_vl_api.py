from __future__ import annotations

import base64
import json
import os
import re
from urllib import request

import cv2


def _is_continuation(text: str) -> bool:
    value = str(text or "").strip()
    return value.startswith(("结论：", "观察：", "描述：", "视频在", "画面中", "画面显示")) or len(value) >= 18


def _is_noise_line(text: str) -> bool:
    value = str(text or "").strip().lower()
    return value in {"证据", "证据如下", "候选片段", "时间戳证据", "evidence"}


def _looks_like_module_heading(text: str) -> bool:
    value = str(text or "").strip().lower()
    if len(value) > 72 or any(marker in value for marker in ("视频", "画面", "展示了", "包含", "秒")):
        return False
    markers = ("模块", "retrieval", "planning", "verification", "training", "evaluation", "scorer")
    return any(marker in value for marker in markers)


def _clean_display_text(value: object) -> str:
    return re.sub(r"[*_`#]", "", str(value or "")).strip()


def _timestamp_key(value: object) -> tuple[float, float] | None:
    match = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\s*s?\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*s?",
        str(value or ""),
    )
    if not match:
        return None
    return (round(float(match.group(1)), 3), round(float(match.group(2)), 3))


def _is_generic_evidence_text(value: object) -> bool:
    text = _clean_display_text(value)
    if not text:
        return True
    normalized = re.sub(r"[：:]", "", text).strip().lower()
    return normalized in {
        "证据",
        "证据如下",
        "时间戳证据",
        "时间戳",
        "evidence",
        "timestamp",
    }


class QwenVLAPIClient:
    backend = "qwen_vl_api"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_sec: int = 120,
    ):
        self.base_url = (base_url or os.environ.get("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.model = model or os.environ.get("DASHSCOPE_MODEL") or "qwen3.5-flash"
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.timeout_sec = timeout_sec

    def generate_answer(self, query: str, context: dict, memory_hits: list[dict]) -> str:
        items = list(context.get("items", []))
        content = [
            {
                "type": "text",
                "text": self._build_instruction(query, context),
            }
        ]
        for idx, item in enumerate(items[:4], start=1):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"片段{idx}：segment_id={item['segment_id']} "
                        f"timestamp={item['start_sec']:.1f}-{item['end_sec']:.1f} "
                        f"综合相关性={float(item.get('score', 0.0)):.4f}。"
                        "请先观察这组帧，再给出该片段的真实画面描述。"
                    ),
                }
            )
            for image_url in _segment_frame_data_urls(
                context.get("video_path", ""),
                float(item["start_sec"]),
                float(item["end_sec"]),
                num_frames=3,
            ):
                content.append({"type": "image_url", "image_url": {"url": image_url}})

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是 VideoTrace 的视频理解模型。只能根据所见画面回答。"
                        "不要根据文件名猜测。不要编造。"
                        "如果证据不足，直接写证据不足。"
                    ),
                },
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "temperature": 0.1,
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
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return f"问题：{query}\n结论：API 调用失败：{type(exc).__name__}: {exc}"
        raw = result["choices"][0]["message"]["content"]
        return self._format_answer(raw, query, items)

    @staticmethod
    def _build_instruction(query: str, context: dict) -> str:
        lines = [
            "你是 VideoTrace 的视频理解 Agent。用户只上传了一个 MP4，下面给你若干时间片段的关键帧。",
            "请直接根据图片内容回答，不要根据文件名猜测；看不清就写证据不足。",
            "每条结论必须带 timestamp=start-end 格式的时间戳。",
            "使用中文，结构清晰，重点说画面里真实出现的人、物体、动作、文字、场景和变化，不要发散。",
            "禁止使用“可能、似乎、大概、也许、推测”等词。若不能确定，直接写证据不足。",
            "",
            f"用户问题：{query}",
            "",
            "输出要求：",
            "1. 优先输出紧凑 JSON，不要输出 Markdown：{\"conclusion\":\"总体结论\",\"evidence\":[{\"timestamp\":\"start-end\",\"text\":\"画面观察\"}]}。",
            "2. evidence 中每个 timestamp 只能出现一次，必须完全复制候选片段的时间范围。",
            "3. 如果无法确认，conclusion 写证据不足，evidence 置空。",
            "4. 不要写“可能/似乎/大概”，除非证据不足。",
            "",
            "候选片段：",
        ]
        for item in context.get("items", []):
            lines.append(
                f"- segment_id={item['segment_id']} timestamp={item['start_sec']:.1f}-{item['end_sec']:.1f}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_answer(raw: str, query: str, items: list[dict]) -> str:
        text = str(raw or "").strip()
        if not text:
            return f"问题：{query}\n结论：证据不足"
        if text.startswith("{") and text.endswith("}"):
            try:
                data = json.loads(text)
            except Exception:
                data = None
            if isinstance(data, dict):
                answer = str(
                    data.get("conclusion") or data.get("answer") or data.get("summary") or ""
                ).strip()
                evidence = data.get("evidence") or []
                return QwenVLAPIClient._render_canonical_answer(query, answer, evidence, items)
        sections = QwenVLAPIClient._parse_answer_sections(text)
        if sections["conclusion"] or sections["evidence"] or sections["memory"]:
            return QwenVLAPIClient._render_canonical_answer(
                query=query,
                conclusion=sections["conclusion"],
                evidence=sections["evidence"],
                fallback_items=items,
                memory=sections["memory"],
                question=sections["question"] or query,
            )
        return QwenVLAPIClient._render_canonical_answer(query, text, [], items)

    @staticmethod
    def _parse_answer_sections(text: str) -> dict:
        question = ""
        conclusion = ""
        evidence: list[dict] = []
        memory: list[dict] = []
        section = "evidence"
        pending_heading = ""

        def append_evidence(item: dict) -> None:
            """Merge repeated timestamp bullets emitted by chat models."""
            timestamp = str(item.get("timestamp", "")).strip()
            key = _timestamp_key(timestamp)
            existing = next(
                (
                    candidate
                    for candidate in evidence
                    if key is not None and _timestamp_key(candidate.get("timestamp")) == key
                ),
                None,
            )
            if existing is None:
                evidence.append(item)
                return
            existing["text"] = QwenVLAPIClient._join_evidence_text(
                str(existing.get("text", "")),
                str(item.get("text", "")),
            )
            if item.get("score") is not None and existing.get("score") is None:
                existing["score"] = item["score"]

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            plain = re.sub(r"[*_`#]", "", line).strip()
            if plain.startswith(("问题：", "用户问题：")):
                question = plain.split("：", 1)[1].strip()
                continue
            if plain.startswith(("结论：", "总体结论：", "回答：")):
                value = plain.split("：", 1)[1].strip()
                if value and _looks_like_module_heading(value):
                    pending_heading = value
                if not conclusion or not _looks_like_module_heading(conclusion):
                    conclusion = value
                section = "evidence"
                continue
            if plain.startswith(("证据：", "证据如下：", "依据：")):
                section = "evidence"
                continue
            if plain.startswith(("记忆补充：", "记忆：")):
                section = "memory"
                continue
            if re.match(r"^[-*•]\s+", plain):
                item = QwenVLAPIClient._parse_bullet(plain[2:].strip(), section)
                if item:
                    if section == "memory":
                        memory.append(item)
                    elif item.get("timestamp"):
                        if pending_heading and _is_generic_evidence_text(item.get("text")):
                            item["text"] = QwenVLAPIClient._join_evidence_text(
                                pending_heading,
                                str(item.get("text", "")),
                            )
                        elif pending_heading:
                            item["text"] = QwenVLAPIClient._join_evidence_text(
                                pending_heading,
                                str(item.get("text", "")),
                            )
                        append_evidence(item)
                        pending_heading = ""
                    elif evidence and _is_continuation(str(item.get("text", ""))) and not _looks_like_module_heading(str(item.get("text", ""))):
                        evidence[-1]["text"] = QwenVLAPIClient._join_evidence_text(
                            str(evidence[-1].get("text", "")),
                            str(item.get("text", "")),
                        )
                    elif _looks_like_module_heading(str(item.get("text", ""))):
                        pending_heading = str(item.get("text", "")).strip()
                    elif not conclusion:
                        conclusion = str(item.get("text", ""))
                    else:
                        pending_heading = str(item.get("text", "")).strip()
                continue
            if "timestamp=" in plain:
                item = QwenVLAPIClient._parse_bullet(plain, section)
                if item:
                    if section == "memory":
                        memory.append(item)
                    elif item.get("timestamp"):
                        if pending_heading and _is_generic_evidence_text(item.get("text")):
                            item["text"] = QwenVLAPIClient._join_evidence_text(
                                pending_heading,
                                str(item.get("text", "")),
                            )
                        elif pending_heading:
                            item["text"] = QwenVLAPIClient._join_evidence_text(
                                pending_heading,
                                str(item.get("text", "")),
                            )
                        append_evidence(item)
                        pending_heading = ""
                    elif evidence and _is_continuation(str(item.get("text", ""))) and not _looks_like_module_heading(str(item.get("text", ""))):
                        evidence[-1]["text"] = QwenVLAPIClient._join_evidence_text(
                            str(evidence[-1].get("text", "")),
                            str(item.get("text", "")),
                        )
                    elif _looks_like_module_heading(str(item.get("text", ""))):
                        pending_heading = str(item.get("text", "")).strip()
                    else:
                        pending_heading = str(item.get("text", "")).strip()
                continue
            if _is_noise_line(plain):
                continue
            if _looks_like_module_heading(plain):
                pending_heading = plain
            elif not conclusion and section == "evidence":
                conclusion = plain
            elif evidence and section == "evidence" and _is_continuation(plain):
                evidence[-1]["text"] = QwenVLAPIClient._join_evidence_text(
                    str(evidence[-1].get("text", "")), plain
                )
            elif section == "evidence":
                pending_heading = plain
        if _looks_like_module_heading(conclusion) and evidence:
            if conclusion not in str(evidence[0].get("text", "")):
                evidence[0]["text"] = QwenVLAPIClient._join_evidence_text(
                    conclusion,
                    str(evidence[0].get("text", "")),
                )
            conclusion = "视频中包含多个可定位模块，具体见下面的时间戳证据。"
        return {
            "question": question,
            "conclusion": conclusion,
            "evidence": QwenVLAPIClient._dedupe_items(evidence),
            "memory": QwenVLAPIClient._dedupe_items(memory),
        }

    @staticmethod
    def _parse_bullet(text: str, section: str) -> dict | None:
        cleaned = text.strip()
        if not cleaned:
            return None
        timestamp_match = re.search(r"timestamp=([0-9.]+-[0-9.]+)", cleaned)
        range_match = re.search(r"([0-9.]+\s*s?\s*-\s*[0-9.]+\s*s?)", cleaned)
        score_match = re.search(r"(?:score|memory_score)=([0-9.]+)", cleaned)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
        elif range_match:
            timestamp = re.sub(r"\s*s\s*", "", range_match.group(1))
        else:
            timestamp = ""
        body = re.sub(r"\s*\(timestamp=[^)]+\)\s*$", "", cleaned)
        body = re.sub(r"\s*\((?:memory_)?score=[^)]+\)\s*$", "", body)
        body = re.sub(r"^timestamp=([0-9.]+-[0-9.]+)\s*[：:]\s*", "", body)
        body = re.sub(
            r"^([0-9.]+\s*s?\s*-\s*[0-9.]+\s*s?)\s*[：:]\s*",
            "",
            body,
        )
        body = re.sub(r"^时间戳证据\s*[：:]?\s*", "", body, flags=re.IGNORECASE)
        body = re.sub(r"\s*timestamp=[0-9.]+-[0-9.]+", "", body, flags=re.IGNORECASE)
        body = re.sub(r"^(?:结论|观察|描述)\s*[：:]\s*", "", body)
        body = re.sub(r"^来自\s+([^：:]+)\s*[：:]\s*", r"\1：", body)
        body = body.strip()
        if not body and not timestamp:
            return None
        item = {
            "timestamp": timestamp,
            "text": body,
        }
        if score_match:
            item["score"] = score_match.group(1)
        if section == "memory":
            source_match = re.match(r"^([^：:]+)\s*[：:]\s*(.+)$", body)
            if source_match:
                item["source"] = source_match.group(1).strip()
                item["text"] = source_match.group(2).strip()
        return item

    @staticmethod
    def _join_evidence_text(*parts: str) -> str:
        cleaned: list[str] = []
        for part in parts:
            value = str(part or "").strip(" ：:;；")
            if not value or value in cleaned:
                continue
            cleaned.append(value)
        return "；".join(cleaned)

    @staticmethod
    def _dedupe_items(items: list[dict]) -> list[dict]:
        seen: dict[tuple[float, float] | tuple[str, str], dict] = {}
        deduped: list[dict] = []
        for item in items:
            timestamp = str(item.get("timestamp", "")).strip()
            timestamp_key = _timestamp_key(timestamp)
            key = timestamp_key if timestamp_key is not None else (
                timestamp,
                str(item.get("text", "")).strip(),
            )
            if key in seen:
                if timestamp_key is not None:
                    seen[key]["text"] = QwenVLAPIClient._join_evidence_text(
                        str(seen[key].get("text", "")),
                        str(item.get("text", "")),
                    )
                continue
            seen[key] = item
            deduped.append(item)
        return deduped

    @staticmethod
    def _render_canonical_answer(
        query: str,
        conclusion: str,
        evidence: list[dict],
        fallback_items: list[dict],
        memory: list[dict] | None = None,
        question: str | None = None,
    ) -> str:
        question_text = question or query
        conclusion_text = _clean_display_text(conclusion) if conclusion else "证据不足"
        lines = [f"问题：{question_text}", f"结论：{conclusion_text}"]
        if _is_abstention(conclusion_text):
            rendered_evidence = []
        else:
            rendered_evidence = (
                QwenVLAPIClient._enrich_evidence(evidence, fallback_items)
                if evidence
                else QwenVLAPIClient._fallback_evidence(fallback_items)
            )
            rendered_evidence = QwenVLAPIClient._complete_evidence(
                rendered_evidence,
                fallback_items,
            )
        for item in rendered_evidence[:5]:
            timestamp = str(item.get("timestamp") or "").strip()
            text = _clean_display_text(item.get("text") or item.get("observation") or "")
            if not timestamp and item.get("start_sec") is not None and item.get("end_sec") is not None:
                timestamp = f"{float(item['start_sec']):.1f}-{float(item['end_sec']):.1f}"
            if not text:
                text = "证据不足"
            if timestamp:
                lines.append(f"- {timestamp}：{text} (timestamp={timestamp})")
            else:
                lines.append(f"- {text}")
        if memory:
            lines.append("记忆补充：")
            for item in memory[:3]:
                source = str(item.get("source") or item.get("memory_id") or "memory").strip()
                text = _clean_display_text(item.get("text") or "")
                if text:
                    lines.append(f"- 来自 {source}：{text}")
        return "\n".join(lines)

    @staticmethod
    def _enrich_evidence(evidence: list[dict], fallback_items: list[dict]) -> list[dict]:
        """Bind sparse model bullets back to the structured segment evidence."""
        enriched: list[dict] = []
        for raw_item in evidence:
            item = dict(raw_item)
            timestamp = str(item.get("timestamp") or "").strip()
            if not timestamp and item.get("start_sec") is not None and item.get("end_sec") is not None:
                timestamp = f"{float(item['start_sec']):.1f}-{float(item['end_sec']):.1f}"
                item["timestamp"] = timestamp
            key = _timestamp_key(timestamp)
            fallback = None
            if key is not None:
                for candidate in fallback_items:
                    candidate_key = _timestamp_key(
                        f"{candidate.get('start_sec', '')}-{candidate.get('end_sec', '')}"
                    )
                    if candidate_key == key:
                        fallback = candidate
                        break
            model_text = _clean_display_text(item.get("text") or item.get("observation") or "")
            fallback_text = QwenVLAPIClient._fallback_text(fallback) if fallback else ""
            if _is_generic_evidence_text(model_text):
                model_text = fallback_text or model_text
            elif fallback_text and len(model_text) < 12:
                model_text = QwenVLAPIClient._join_evidence_text(model_text, fallback_text)
            item["text"] = model_text or "证据不足"
            enriched.append(item)
        return QwenVLAPIClient._dedupe_items(enriched)

    @staticmethod
    def _fallback_text(item: dict | None) -> str:
        if not item:
            return ""
        parts = [
            str(item.get("caption") or "").strip(),
            str(item.get("ocr_text") or "").strip(),
            str(item.get("text") or "").strip(),
        ]
        scene = str(item.get("scene") or "").strip()
        actions = [str(value).strip() for value in item.get("actions", []) if str(value).strip()]
        entities = [str(value).strip() for value in item.get("entities", []) if str(value).strip()]
        if scene:
            parts.append(f"场景：{scene}")
        if entities:
            parts.append(f"人物与物体：{'、'.join(entities)}")
        if actions:
            parts.append(f"动作与变化：{'、'.join(actions)}")
        compact: list[str] = []
        for part in parts:
            clean = _clean_display_text(part)
            if clean and clean not in compact:
                compact.append(clean)
        return "；".join(compact)[:260]

    @staticmethod
    def _fallback_evidence(items: list[dict]) -> list[dict]:
        evidence: list[dict] = []
        for item in items[:4]:
            evidence.append(
                {
                    "timestamp": f"{float(item['start_sec']):.1f}-{float(item['end_sec']):.1f}",
                    "text": str(item.get("text", ""))[:160].strip() or "证据不足",
                }
            )
        return evidence

    @staticmethod
    def _complete_evidence(evidence: list[dict], fallback_items: list[dict]) -> list[dict]:
        """Append selected windows omitted by an otherwise valid answer.

        The model still owns the conclusion and the wording it generated.
        Missing evidence is copied only from the already selected structured
        context, improving timestamp coverage without inventing claims.
        """

        completed = QwenVLAPIClient._dedupe_items([dict(item) for item in evidence])
        existing = {
            key
            for key in (_timestamp_key(item.get("timestamp")) for item in completed)
            if key is not None
        }
        for item in fallback_items[:5]:
            timestamp = f"{float(item['start_sec']):.1f}-{float(item['end_sec']):.1f}"
            key = _timestamp_key(timestamp)
            if key is None or key in existing:
                continue
            completed.append(
                {
                    "timestamp": timestamp,
                    "text": QwenVLAPIClient._fallback_text(item) or "证据不足",
                }
            )
            existing.add(key)
        return completed


def _is_abstention(value: object) -> bool:
    text = str(value or "")
    return any(marker in text for marker in ("证据不足", "无法确认", "不能确认", "无法判断"))


def _segment_frame_data_urls(video_path: str, start_sec: float, end_sec: float, num_frames: int) -> list[str]:
    if not video_path:
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened() or end_sec <= start_sec:
        return []
    urls: list[str] = []
    if num_frames <= 1:
        times = [(start_sec + end_sec) / 2.0]
    else:
        step = (end_sec - start_sec) / float(num_frames + 1)
        times = [start_sec + step * (idx + 1) for idx in range(num_frames)]
    for sec in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 76])
        if not ok:
            continue
        b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        urls.append(f"data:image/jpeg;base64,{b64}")
    cap.release()
    return urls
