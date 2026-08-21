from __future__ import annotations


SYSTEM_PROMPT = """你是 VideoTrace 的视频证据问答模块。请严格遵守：
1. 只能使用给定上下文和记忆里的信息回答。
2. 每条关键结论必须带 timestamp=start-end 形式的时间戳证据。
3. 如果证据不足，明确说“证据不足”，不要编造。
4. 不要使用“可能、似乎、大概、也许、推测”等不确定措辞。
5. 使用中文回答，结构清晰，不要输出无关解释。
6. 优先输出紧凑 JSON，不要输出 Markdown：
   {"conclusion":"一句总体结论","evidence":[{"timestamp":"start-end","text":"可见证据"}]}
7. evidence 中每个 timestamp 只能出现一次，只能使用候选片段给出的时间范围；无法确认时 evidence 置空。
"""


def build_user_prompt(query: str, context: dict, memory_hits: list[dict]) -> str:
    lines = [f"用户问题：{query}", "", "证据上下文："]
    for item in context.get("items", []):
        structured = []
        if item.get("entities"):
            structured.append("实体=" + "、".join(str(value) for value in item["entities"]))
        if item.get("actions"):
            structured.append("动作=" + "、".join(str(value) for value in item["actions"]))
        if item.get("scene"):
            structured.append("场景=" + str(item["scene"]))
        if item.get("ocr_text"):
            structured.append("OCR=" + str(item["ocr_text"]))
        structured_suffix = (" " + " ".join(structured)) if structured else ""
        lines.append(
            f"- segment_id={item['segment_id']} "
            f"timestamp={item['start_sec']:.1f}-{item['end_sec']:.1f} "
            f"score={item['score']:.4f} text={item['text']}{structured_suffix}"
        )
    lines.extend(["", "记忆命中："])
    for memory in memory_hits:
        lines.append(
            f"- memory_id={memory['memory_id']} "
            f"source={memory['source_segment_id']} text={memory['text']}"
        )
    lines.extend(
        [
            "",
            "请生成最终回答。优先输出紧凑 JSON，字段只能是 conclusion 和 evidence；",
            "每个 evidence 必须包含 timestamp 和 text，timestamp 必须完全复制候选片段的范围。",
        ]
    )
    return "\n".join(lines)
