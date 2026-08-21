from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class PlanStep:
    action: str
    target: str
    rationale: str


class SimplePlanner:
    def plan(self, query: str, ranked_segments: List[dict]) -> List[PlanStep]:
        if not ranked_segments:
            return [PlanStep(action='inspect', target='full_video', rationale='No candidate segments found')]
        steps = []
        for seg in ranked_segments[:3]:
            steps.append(
                PlanStep(
                    action='retrieve',
                    target=seg['segment_id'],
                    rationale=f"match score={seg['score']:.4f}",
                )
            )
        steps.append(PlanStep(action='synthesize', target='knowledge_pack', rationale='Aggregate evidence into answer'))
        return steps


class PlannerDatasetRecord:
    def __init__(self, query: str, candidates: List[dict], target: str):
        self.query = query
        self.candidates = candidates
        self.target = target


def build_planner_example(query: str, ranked_segments: List[dict]) -> dict:
    return {
        'query': query,
        'candidates': ranked_segments,
        'target_plan': [
            {'action': 'retrieve', 'target': seg['segment_id'], 'rank': i}
            for i, seg in enumerate(ranked_segments[:3])
        ] + [{'action': 'synthesize', 'target': 'knowledge_pack'}],
    }


def build_ranker_pairs(query: str, ranked_segments: List[dict]) -> list[dict]:
    pairs = []
    if not ranked_segments:
        return pairs
    best = ranked_segments[0]
    for neg in ranked_segments[1:]:
        pairs.append(
            {
                'query': query,
                'positive': best,
                'negative': neg,
            }
        )
    return pairs
