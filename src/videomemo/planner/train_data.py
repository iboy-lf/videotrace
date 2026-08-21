from __future__ import annotations

from pathlib import Path
import json

from .simple_planner import build_planner_example, build_ranker_pairs


def write_planner_dataset(query: str, ranked_segments: list[dict], out_path: str) -> str:
    payload = build_planner_example(query, ranked_segments)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return str(path)


def write_ranker_dataset(query: str, ranked_segments: list[dict], out_path: str) -> str:
    payload = build_ranker_pairs(query, ranked_segments)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return str(path)
