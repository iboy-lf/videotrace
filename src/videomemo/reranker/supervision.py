from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class RerankerSupervisionCase:
    case_id: str
    video_path: str
    query: str
    video_id: str
    split: str = "dev"
    gold_spans: list[dict] = field(default_factory=list)


def load_reranker_supervision(path: str) -> list[RerankerSupervisionCase]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", []) if isinstance(payload, dict) else payload
    cases: list[RerankerSupervisionCase] = []
    for index, item in enumerate(raw_cases):
        video_path = Path(str(item["video_path"]))
        if not video_path.is_absolute():
            video_path = (source.parent / video_path).resolve()
        cases.append(
            RerankerSupervisionCase(
                case_id=str(item.get("case_id") or f"supervision-{index:04d}"),
                video_path=str(video_path),
                query=str(item["query"]),
                video_id=str(item.get("video_id") or video_path.stem),
                split=str(item.get("split") or "dev").lower(),
                gold_spans=list(item.get("gold_spans", [])),
            )
        )
    return cases
