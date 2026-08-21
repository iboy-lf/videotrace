from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np

from ..models import Segment
from ..utils.text import informative_keyword_set, keyword_set


FEATURE_NAMES = [
    "query_coverage",
    "retrieval_score",
    "scorer_score",
    "vlm_score",
    "retrieval_rank_score",
    "scorer_rank_score",
    "vlm_rank_score",
    "base_fusion_score",
    "understanding_confidence",
    "temporal_position",
    "duration_ratio",
    "text_length_log",
]


@dataclass
class RerankerTrainResult:
    checkpoint_path: str
    num_rows: int
    num_groups: int
    num_positive: int
    num_train_groups: int
    num_eval_groups: int
    num_pairwise_eval_groups: int
    num_pairwise_eval_pairs: int
    best_epoch: int
    train_loss: float
    eval_loss: float | None
    pairwise_accuracy: float | None
    base_pairwise_accuracy: float | None
    recommended_blend_weight: float | None
    blended_pairwise_accuracy: float | None
    device: str

    def dump(self) -> dict:
        return self.__dict__


def build_reranker_features(query: str, segment: Segment, duration_sec: float) -> dict[str, float]:
    query_terms = informative_keyword_set(query)
    text_terms = keyword_set(segment.searchable_text())
    coverage = len(query_terms & text_terms) / max(1, len(query_terms))
    duration = max(float(duration_sec), 1.0)
    midpoint = (float(segment.start_sec) + float(segment.end_sec)) / 2.0
    segment_duration = max(0.0, float(segment.end_sec) - float(segment.start_sec))
    return {
        "query_coverage": float(coverage),
        "retrieval_score": float(segment.retrieval_score),
        "scorer_score": float(segment.scorer_score),
        "vlm_score": float(segment.vlm_score),
        "retrieval_rank_score": float(segment.retrieval_rank_score),
        "scorer_rank_score": float(segment.scorer_rank_score),
        "vlm_rank_score": float(segment.vlm_rank_score),
        "base_fusion_score": float(segment.score),
        "understanding_confidence": float(segment.understanding_confidence),
        "temporal_position": midpoint / duration,
        "duration_ratio": segment_duration / duration,
        "text_length_log": float(np.log1p(len(segment.searchable_text())) / 8.0),
    }


def vectorize_features(features: dict[str, float]) -> np.ndarray:
    return np.asarray([float(features.get(name, 0.0)) for name in FEATURE_NAMES], dtype="float32")


class NeuralSegmentReranker:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_features = list(payload.get("feature_names", []))
        if checkpoint_features and checkpoint_features != FEATURE_NAMES:
            raise ValueError(
                "reranker checkpoint feature contract does not match the current runtime"
            )
        self.device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.mean = np.asarray(payload["mean"], dtype="float32")
        self.std = np.asarray(payload["std"], dtype="float32")
        if self.mean.shape != (len(FEATURE_NAMES),) or self.std.shape != (len(FEATURE_NAMES),):
            raise ValueError("reranker checkpoint normalization statistics have invalid dimensions")
        self.std = np.where(self.std <= 1e-6, 1.0, self.std).astype("float32")
        self.model = _build_model(len(FEATURE_NAMES), int(payload.get("hidden_dim", 32)))
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()
        self.checkpoint_path = checkpoint_path
        training = payload.get("training", {})
        self.recommended_blend_weight = float(training.get("recommended_blend_weight", 0.35))

    def score(self, query: str, segment: Segment, duration_sec: float) -> float:
        import torch

        vector = vectorize_features(build_reranker_features(query, segment, duration_sec))
        normalized = (vector - self.mean) / self.std
        tensor = torch.from_numpy(normalized).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            return float(torch.sigmoid(self.model(tensor)).item())

    def rank(self, query: str, segments: list[Segment], duration_sec: float) -> list[Segment]:
        for segment in segments:
            segment.reranker_score = self.score(query, segment, duration_sec)
        return sorted(segments, key=lambda segment: segment.reranker_score, reverse=True)


def train_reranker(
    dataset_path: str,
    checkpoint_path: str,
    epochs: int = 80,
    hidden_dim: int = 32,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    eval_fraction: float = 0.2,
    seed: int = 42,
    device: str = "auto",
    allow_test_split: bool = False,
) -> RerankerTrainResult:
    import torch
    from torch import nn

    rows = _load_rows(dataset_path)
    if not rows:
        raise ValueError("reranker dataset is empty")
    test_rows = [row for row in rows if str(row.get("split", "")).lower() == "test"]
    if test_rows and not allow_test_split:
        raise ValueError(
            "reranker dataset contains frozen test rows; rebuild with --split dev or pass an explicit override"
        )
    positives = sum(float(row["label"]) > 0.5 for row in rows)
    if positives == 0 or positives == len(rows):
        raise ValueError("reranker training requires both positive and negative rows")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_device = _resolve_device(device, torch)

    groups = sorted({str(row["group_id"]) for row in rows})
    train_rows, eval_rows = _split_rows_by_group(rows, eval_fraction, seed)
    if not train_rows:
        train_rows, eval_rows = rows, []

    if not _has_both_labels(train_rows):
        train_rows, eval_rows = rows, []
    train_groups = sorted({str(row["group_id"]) for row in train_rows})
    eval_groups = sorted({str(row["group_id"]) for row in eval_rows})
    pairwise_eval_groups = _mixed_label_groups(eval_rows)

    x_train, y_train = _matrix(train_rows)
    x_eval, y_eval = _matrix(eval_rows) if eval_rows else (None, None)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std <= 1e-6, 1.0, std).astype("float32")
    x_train = ((x_train - mean) / std).astype("float32")
    if x_eval is not None:
        x_eval = ((x_eval - mean) / std).astype("float32")

    model = _build_model(len(FEATURE_NAMES), hidden_dim).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_pos = max(1, int(y_train.sum()))
    num_neg = max(1, int(len(y_train) - y_train.sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([num_neg / num_pos], device=torch_device))
    x_train_tensor = torch.from_numpy(x_train).to(torch_device)
    y_train_tensor = torch.from_numpy(y_train).to(torch_device).unsqueeze(1)
    x_eval_tensor = torch.from_numpy(x_eval).to(torch_device) if x_eval is not None else None
    y_eval_tensor = torch.from_numpy(y_eval).to(torch_device).unsqueeze(1) if y_eval is not None else None

    best_state = None
    best_epoch = 0
    best_eval = float("inf")
    best_train = float("inf")
    last_train = float("inf")
    patience = 12
    stale = 0
    train_group_ids = [str(row["group_id"]) for row in train_rows]
    for epoch in range(1, max(1, int(epochs)) + 1):
        model.train()
        logits = model(x_train_tensor)
        loss = criterion(logits, y_train_tensor)
        pair_loss = _pairwise_loss(logits, y_train_tensor, train_group_ids, torch)
        total_loss = loss + 0.25 * pair_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        last_train = float(total_loss.detach().cpu())

        model.eval()
        with torch.inference_mode():
            if x_eval_tensor is not None:
                current_eval = float(criterion(model(x_eval_tensor), y_eval_tensor).detach().cpu())
            else:
                current_eval = last_train
        if current_eval < best_eval - 1e-5:
            best_eval = current_eval
            best_epoch = epoch
            best_train = last_train
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    pairwise_accuracy = None
    base_pairwise_accuracy = None
    recommended_blend_weight = None
    blended_pairwise_accuracy = None
    num_pairwise_eval_pairs = _pairwise_pair_count(eval_rows)
    if eval_rows:
        neural_scores = _model_scores(model, x_eval, torch_device, torch)
        pairwise_accuracy = _pairwise_accuracy_from_scores(neural_scores, eval_rows)
        base_scores = np.asarray(
            [float(row.get("features", {}).get("base_fusion_score", 0.0)) for row in eval_rows],
            dtype="float32",
        )
        base_pairwise_accuracy = _pairwise_accuracy_from_scores(base_scores, eval_rows)
        recommended_blend_weight, blended_pairwise_accuracy = _select_blend_weight(
            base_scores,
            neural_scores,
            eval_rows,
        )

    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "feature_names": FEATURE_NAMES,
            "mean": mean.astype("float32"),
            "std": std.astype("float32"),
            "hidden_dim": int(hidden_dim),
            "training": {
                "dataset_path": str(Path(dataset_path).resolve()),
                "best_epoch": best_epoch,
                "train_loss": best_train,
                "eval_loss": best_eval if eval_rows else None,
                "pairwise_accuracy": pairwise_accuracy,
                "base_pairwise_accuracy": base_pairwise_accuracy,
                "recommended_blend_weight": recommended_blend_weight,
                "blended_pairwise_accuracy": blended_pairwise_accuracy,
                "num_pairwise_eval_pairs": num_pairwise_eval_pairs,
                "seed": seed,
                "train_groups": train_groups,
                "eval_groups": eval_groups,
                "pairwise_eval_groups": pairwise_eval_groups,
            },
        },
        target,
    )
    return RerankerTrainResult(
        checkpoint_path=str(target),
        num_rows=len(rows),
        num_groups=len(groups),
        num_positive=positives,
        num_train_groups=len(train_groups),
        num_eval_groups=len(eval_groups),
        num_pairwise_eval_groups=len(pairwise_eval_groups),
        num_pairwise_eval_pairs=num_pairwise_eval_pairs,
        best_epoch=best_epoch,
        train_loss=best_train,
        eval_loss=best_eval if eval_rows else None,
        pairwise_accuracy=pairwise_accuracy,
        base_pairwise_accuracy=base_pairwise_accuracy,
        recommended_blend_weight=recommended_blend_weight,
        blended_pairwise_accuracy=blended_pairwise_accuracy,
        device=str(torch_device),
    )


def _build_model(input_dim: int, hidden_dim: int):
    from torch import nn

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(0.10),
        nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
        nn.GELU(),
        nn.Linear(max(8, hidden_dim // 2), 1),
    )


def _load_rows(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([vectorize_features(row["features"]) for row in rows])
    y = np.asarray([float(row["label"]) for row in rows], dtype="float32")
    return x, y


def _resolve_device(device: str, torch):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _pairwise_loss(logits, labels, group_ids: list[str], torch):
    losses = []
    for group_id in sorted(set(group_ids)):
        indices = [index for index, value in enumerate(group_ids) if value == group_id]
        group_logits = logits[indices]
        group_labels = labels[indices, 0]
        positive = group_logits[group_labels > 0.5]
        negative = group_logits[group_labels <= 0.5]
        if not len(positive) or not len(negative):
            continue
        diff = positive[:, None, 0] - negative[None, :, 0]
        losses.append(torch.nn.functional.softplus(-diff).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def _has_both_labels(rows: list[dict]) -> bool:
    labels = {float(row["label"]) > 0.5 for row in rows}
    return labels == {False, True}


def _mixed_label_groups(rows: list[dict]) -> list[str]:
    labels_by_group: dict[str, set[bool]] = {}
    for row in rows:
        labels_by_group.setdefault(str(row["group_id"]), set()).add(float(row["label"]) > 0.5)
    return sorted(group_id for group_id, labels in labels_by_group.items() if labels == {False, True})


def _split_rows_by_group(
    rows: list[dict],
    eval_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    groups = sorted({str(row["group_id"]) for row in rows})
    eval_count = int(round(len(groups) * max(0.0, min(0.5, eval_fraction))))
    if eval_count <= 0 or len(groups) <= 1:
        return list(rows), []

    mixed = _mixed_label_groups(rows)
    mixed_set = set(mixed)
    single_label = [group_id for group_id in groups if group_id not in mixed_set]
    rng = random.Random(seed)
    rng.shuffle(mixed)
    rng.shuffle(single_label)

    # Prefer pairwise-evaluable groups while retaining one mixed group for
    # pairwise training whenever the supervision contains more than one.
    candidates = mixed[:-1] + single_label if len(mixed) > 1 else single_label
    if len(candidates) < eval_count:
        candidates += [group_id for group_id in mixed if group_id not in candidates]
    eval_group_ids = set(candidates[:eval_count])
    train_rows = [row for row in rows if str(row["group_id"]) not in eval_group_ids]
    eval_rows = [row for row in rows if str(row["group_id"]) in eval_group_ids]
    return train_rows, eval_rows


def _pairwise_accuracy(model, x_eval: np.ndarray, rows: list[dict], device, torch) -> float | None:
    if x_eval is None:
        return None
    scores = _model_scores(model, x_eval, device, torch)
    return _pairwise_accuracy_from_scores(scores, rows)


def _model_scores(model, matrix: np.ndarray, device, torch) -> np.ndarray:
    with torch.inference_mode():
        logits = model(torch.from_numpy(matrix).to(device))
        return torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)


def _pairwise_accuracy_from_scores(scores: np.ndarray, rows: list[dict]) -> float | None:
    by_group: dict[str, list[tuple[float, float]]] = {}
    for row, score in zip(rows, scores):
        by_group.setdefault(str(row["group_id"]), []).append((float(score), float(row["label"])))
    correct = 0
    total = 0
    for items in by_group.values():
        positives = [score for score, label in items if label > 0.5]
        negatives = [score for score, label in items if label <= 0.5]
        for pos in positives:
            for neg in negatives:
                correct += int(pos > neg)
                total += 1
    return correct / total if total else None


def _pairwise_pair_count(rows: list[dict]) -> int:
    labels_by_group: dict[str, list[float]] = {}
    for row in rows:
        labels_by_group.setdefault(str(row["group_id"]), []).append(float(row["label"]))
    return sum(
        sum(label > 0.5 for label in labels) * sum(label <= 0.5 for label in labels)
        for labels in labels_by_group.values()
    )


def _select_blend_weight(
    base_scores: np.ndarray,
    neural_scores: np.ndarray,
    rows: list[dict],
) -> tuple[float | None, float | None]:
    candidates = [index / 20.0 for index in range(21)]
    scored = []
    for weight in candidates:
        blended = (1.0 - weight) * base_scores + weight * neural_scores
        accuracy = _pairwise_accuracy_from_scores(blended, rows)
        if accuracy is not None:
            scored.append((accuracy, weight))
    if not scored:
        return None, None
    best_accuracy = max(accuracy for accuracy, _ in scored)
    best_weight = min(weight for accuracy, weight in scored if accuracy == best_accuracy)
    return best_weight, best_accuracy
