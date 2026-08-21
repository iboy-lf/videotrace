from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..models import Segment
from ..utils.text import informative_keyword_set, keyword_set, tokenize_text


def build_segment_index(segments: List[Segment]) -> dict:
    texts = [seg.searchable_text() for seg in segments]
    vectorizer = TfidfVectorizer(tokenizer=tokenize_text, token_pattern=None, lowercase=False, min_df=1)
    matrix = vectorizer.fit_transform(texts) if texts else None
    return {
        'segments': segments,
        'texts': texts,
        'vectorizer': vectorizer,
        'matrix': matrix,
    }


def score_segment(query: str, index: dict, item_idx: int) -> float:
    matrix = index['matrix']
    if matrix is None:
        return 0.0
    vectorizer = index['vectorizer']
    query_vec = vectorizer.transform([query])
    sim = float((matrix[item_idx] @ query_vec.T).toarray()[0][0])
    seg = index['segments'][item_idx]
    time_bias = 1.0 / (1.0 + seg.start_sec)
    lexical = _lexical_overlap(query, index['texts'][item_idx])
    visual = 0.0008 * seg.brightness_mean + 0.003 * seg.motion_score + 0.001 * seg.contrast_std
    return 0.68 * sim + 0.20 * lexical + 0.08 * visual + 0.04 * time_bias


def rank_segments(query: str, index: dict, top_k: int = 5, use_mmr: bool = True, mmr_lambda: float = 0.72) -> List[dict]:
    scored = []
    for i, seg in enumerate(index['segments']):
        scored.append(
            {
                'item_idx': i,
                'segment_id': seg.segment_id,
                'start_sec': seg.start_sec,
                'end_sec': seg.end_sec,
                'text': index['texts'][i],
                'score': score_segment(query, index, i),
                'retrieval_signals': {
                    'lexical_overlap': _lexical_overlap(query, index['texts'][i]),
                    'motion_score': seg.motion_score,
                    'visual_signature': seg.visual_signature,
                    'vlm_score': getattr(seg, "vlm_score", 0.0),
                },
            }
        )
    scored.sort(key=lambda x: x['score'], reverse=True)
    if use_mmr and index.get('matrix') is not None:
        scored = _mmr_select(scored, index['matrix'], top_k=top_k, mmr_lambda=mmr_lambda)
    else:
        scored = scored[:top_k]
    for item in scored:
        item.pop('item_idx', None)
    return scored


def index_statistics(index: dict) -> dict:
    matrix = index.get('matrix')
    if matrix is None:
        return {'num_segments': 0, 'num_terms': 0}
    return {
        'num_segments': int(matrix.shape[0]),
        'num_terms': int(matrix.shape[1]),
        'retriever': 'hybrid_tfidf_lexical_visual_mmr',
    }


def _lexical_overlap(query: str, text: str) -> float:
    q = informative_keyword_set(query)
    t = keyword_set(text)
    if not q:
        return 0.0
    return len(q & t) / max(1, len(q))


def _mmr_select(candidates: list[dict], matrix, top_k: int, mmr_lambda: float) -> list[dict]:
    if not candidates:
        return []
    selected: list[dict] = []
    remaining = list(candidates)
    dense = matrix.toarray()
    norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-9
    normalized = dense / norms

    while remaining and len(selected) < top_k:
        best_item = None
        best_value = -1e9
        for item in remaining:
            relevance = float(item['score'])
            idx = int(item['item_idx'])
            diversity_penalty = 0.0
            if selected:
                sims = [
                    float(normalized[idx] @ normalized[int(prev['item_idx'])].T)
                    for prev in selected
                ]
                diversity_penalty = max(sims)
            value = mmr_lambda * relevance - (1.0 - mmr_lambda) * diversity_penalty
            if value > best_value:
                best_value = value
                best_item = item
        selected.append(best_item)
        remaining = [item for item in remaining if item is not best_item]
    return selected
