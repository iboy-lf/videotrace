from __future__ import annotations

import json
from pathlib import Path

from ..models import KnowledgePack


def export_pack(pack: KnowledgePack, output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / 'knowledge_pack.json'
    md_path = out / 'report.md'

    json_path.write_text(json.dumps(pack.dump(), ensure_ascii=False, indent=2), encoding='utf-8')

    agent_run = pack.metadata.get("agent_run", {})
    lines = [
        '# VideoTrace 项目报告',
        '',
        '输入：单个 MP4 视频',
        f'视频时长：`{pack.duration_sec:.2f}s`',
        '',
        '## 摘要',
        pack.summary,
        '',
        '## Agent 回答',
        pack.answer,
        '',
        '## 技术栈运行快照',
        f"- 片段理解：{pack.metadata.get('segment_understanding', {}).get('backend', 'none')}",
        f"- 语音对齐：{pack.metadata.get('asr', {}).get('backend', 'none')}",
        f"- 多模态向量：{pack.metadata.get('vlm', {}).get('backend', 'none')}",
        f"- 回答生成：{pack.metadata.get('llm_backend', 'unknown')}",
        f"- 查询意图：{pack.metadata.get('query_intent', {}).get('kind', 'unknown')}",
        f"- 证据选择：{pack.metadata.get('retrieval_selection', {}).get('strategy', 'unknown')}",
        '',
        '## 证据时间线',
    ]
    for item in pack.timeline:
        lines.append(f"- {item['start_sec']:.1f}s-{item['end_sec']:.1f}s: {item['text']}")
    lines += ['', '## 推荐片段']
    for clip in pack.clips:
        lines.append(f"- {clip['start_sec']:.1f}s-{clip['end_sec']:.1f}s score={clip['score']:.2f}")

    lines += ['', '## Agent 执行轨迹']
    for step in agent_run.get("plan", []):
        step_id = step.get("step", "-")
        action = step.get("action", "")
        why = step.get("why") or step.get("observation", "")
        lines.append(f"- Step {step_id}: `{action}` - {why}")

    lines += ['', '## 上下文窗口']
    context = agent_run.get("context", {})
    lines.append(f"- 使用字符数：{context.get('used_chars', 0)}")
    dropped = context.get("dropped_segment_ids", [])
    lines.append(f"- 被裁掉的低优先级片段：{', '.join(dropped) if dropped else '无'}")
    for item in context.get("items", []):
        lines.append(f"- {item['segment_id']} {item['start_sec']:.1f}s-{item['end_sec']:.1f}s: {item['text']}")

    lines += ['', '## 元数据']
    lines.append(json.dumps(pack.metadata, ensure_ascii=False, indent=2))
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    return json_path


def export_manifest(pack: KnowledgePack, output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    manifest = {
        'video_path': pack.video_path,
        'duration_sec': pack.duration_sec,
        'num_segments': len(pack.segments),
        'num_clips': len(pack.clips),
        'summary': pack.summary,
        'agent_mode': pack.metadata.get('agent_run', {}).get('mode'),
        'scorer_mode': pack.metadata.get('scorer_mode'),
        'vlm_backend': pack.metadata.get('vlm', {}).get('backend'),
        'llm_backend': pack.metadata.get('llm_backend'),
        'asr_backend': pack.metadata.get('asr', {}).get('backend'),
        'query_intent': pack.metadata.get('query_intent', {}),
        'retrieval_strategy': pack.metadata.get('retrieval_selection', {}).get('strategy'),
        'source_sha256': pack.metadata.get('source_sha256'),
        'video_sha256': pack.metadata.get('video_sha256'),
        'environment': pack.metadata.get('environment', {}),
        'performance': pack.metadata.get('performance', {}),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest_path
