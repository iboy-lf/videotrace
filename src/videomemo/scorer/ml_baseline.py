from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
import json
import pickle

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ..models import Segment
from .segment_scorer import _segment_features


FEATURE_NAMES = [
    'start_sec',
    'duration_sec',
    'frame_count',
    'brightness_mean',
    'contrast_std',
    'motion_score',
    'text_len',
    'ocr_len',
    'asr_len',
]


@dataclass
class TrainResult:
    num_examples: int
    num_positive: int
    average_precision: float
    roc_auc: float | None
    model_path: str
    feature_names: list[str]

    def dump(self) -> dict:
        return self.__dict__


def _feature_vector(example: dict) -> list[float]:
    features = example.get('features', {})
    start = float(features.get('start_sec', 0.0))
    end = float(features.get('end_sec', start))
    return [
        start,
        max(0.0, end - start),
        float(features.get('frame_count', 0.0)),
        float(features.get('brightness_mean', 0.0)),
        float(features.get('contrast_std', 0.0)),
        float(features.get('motion_score', 0.0)),
        float(features.get('text_len', 0.0)),
        float(features.get('ocr_len', 0.0)),
        float(features.get('asr_len', 0.0)),
    ]


def load_examples(paths: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        for item in data:
            rows.append(_feature_vector(item))
            labels.append(float(item.get('label', 0.0)))
    if not rows:
        raise ValueError('no scorer examples found')
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def train_scorer(example_paths: list[str], model_path: str) -> TrainResult:
    x, y = load_examples(example_paths)
    if len(set(y.tolist())) < 2:
        raise ValueError('need both positive and negative examples for scorer training')

    counts = np.bincount(y.astype(int))
    can_split = len(y) >= 8 and len(counts) >= 2 and int(np.min(counts)) >= 2
    if can_split:
        x_train, x_eval, y_train, y_eval = train_test_split(
            x,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )
    else:
        x_train, x_eval, y_train, y_eval = x, x, y, y
    model = GradientBoostingClassifier(random_state=42)
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_eval)[:, 1]
    ap = float(average_precision_score(y_eval, prob))
    try:
        auc = float(roc_auc_score(y_eval, prob))
    except ValueError:
        auc = None

    out = Path(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as f:
        pickle.dump({'model': model, 'feature_names': FEATURE_NAMES}, f)

    return TrainResult(
        num_examples=int(len(y)),
        num_positive=int(np.sum(y)),
        average_precision=ap,
        roc_auc=auc,
        model_path=str(out),
        feature_names=FEATURE_NAMES,
    )


class SklearnSegmentScorer:
    def __init__(self, model_path: str):
        payload = pickle.loads(Path(model_path).read_bytes())
        self.model = payload['model']
        self.feature_names = payload['feature_names']

    def score(self, segment: Segment) -> float:
        example = {'features': _segment_features(segment)}
        vector = _feature_vector(example)
        expected = int(getattr(self.model, 'n_features_in_', len(vector)))
        x = np.asarray([vector[:expected]], dtype=np.float32)
        return float(self.model.predict_proba(x)[0, 1])

    def rank(self, segments: List[Segment]) -> List[Segment]:
        ranked = sorted(segments, key=self.score, reverse=True)
        for seg in ranked:
            seg.score = self.score(seg)
        return ranked
