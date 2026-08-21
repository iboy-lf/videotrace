from __future__ import annotations


class TemplateLLMClient:
    backend = "template"

    def generate_answer(self, query: str, context: dict, memory_hits: list[dict]) -> str:
        lines = [f"问题：{query}", "结论：系统按时间戳整理出最相关的视频证据，主要内容如下："]
        for item in context.get("items", [])[:3]:
            lines.append(
                f"- {item['start_sec']:.1f}s-{item['end_sec']:.1f}s：{item['text']} "
                f"(timestamp={item['start_sec']:.1f}-{item['end_sec']:.1f})"
            )
        if memory_hits:
            lines.append("记忆补充：")
            for memory in memory_hits[:2]:
                lines.append(
                    f"- 来自 {memory['source_segment_id']}：{memory['text']}"
                )
        dropped = context.get("dropped_segment_ids", [])
        if dropped:
            lines.append(f"上下文预算有限，部分低优先级片段未进入最终上下文：{', '.join(dropped)}")
        return "\n".join(lines)
