from __future__ import annotations

from pathlib import Path
import json

from .segment_scorer import build_score_examples


def write_scorer_dataset(query: str, segments: list, out_path: str) -> str:
    examples = build_score_examples(query, segments)
    payload = [ex.__dict__ for ex in examples]
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return str(path)
