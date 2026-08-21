from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from videomemo.index.dense_index import NumpyDenseIndex
from videomemo.ingest.asr import SidecarASR, enrich_segments_with_asr
from videomemo.models import Segment
from videomemo.query.intent import QueryIntent, classify_query, select_ranked_segments
from videomemo.reranker.neural import FEATURE_NAMES, build_reranker_features, vectorize_features
from videomemo.scorer.ml_baseline import SklearnSegmentScorer


class _EightFeatureModel:
    n_features_in_ = 8

    def predict_proba(self, matrix):
        assert matrix.shape == (1, 8)
        return np.asarray([[0.25, 0.75]], dtype="float32")


def _candidate(segment_id: str, start: float, score: float) -> dict:
    return {
        "segment_id": segment_id,
        "start_sec": start,
        "end_sec": start + 20.0,
        "score": score,
    }


def test_overview_selection_keeps_opening_middle_and_ending():
    candidates = [
        _candidate("late", 380.0, 1.00),
        _candidate("middle", 190.0, 0.95),
        _candidate("early", 0.0, 0.20),
        _candidate("middle-2", 220.0, 0.90),
    ]
    selected = select_ranked_segments(
        candidates,
        top_k=3,
        duration_sec=420.0,
        intent=QueryIntent(kind="overview", coverage_mode="distributed"),
    )
    assert [item["segment_id"] for item in selected] == ["early", "middle", "late"]
    assert {item["selection_reason"] for item in selected} == {
        "temporal_coverage:opening",
        "temporal_coverage:middle",
        "temporal_coverage:ending",
    }


def test_overview_opening_prefers_video_boundary_on_416_second_video():
    candidates = [
        _candidate("opening-near-eight-percent", 40.0, 0.90),
        _candidate("middle", 200.0, 0.95),
        _candidate("ending", 400.0, 0.92),
        _candidate("true-opening", 0.0, 0.82),
    ]
    selected = select_ranked_segments(
        candidates,
        top_k=3,
        duration_sec=416.2,
        intent=QueryIntent(
            kind="overview",
            coverage_mode="distributed",
            stage_hints=["opening", "middle", "ending"],
        ),
    )
    assert [item["segment_id"] for item in selected] == ["true-opening", "middle", "ending"]


def test_multi_span_selection_initializes_candidate_pool():
    candidates = [
        _candidate("first", 40.0, 0.95),
        _candidate("second", 240.0, 0.90),
        _candidate("third", 360.0, 0.80),
    ]
    selected = select_ranked_segments(
        candidates,
        top_k=2,
        duration_sec=416.2,
        intent=QueryIntent(kind="count", coverage_mode="multi_span"),
    )
    assert len(selected) == 2
    assert {item["segment_id"] for item in selected} == {"first", "second"}


def test_local_query_keeps_relevance_order_and_comparison_is_not_forced_global():
    intent = classify_query("百事可乐与可口可乐的差异是什么？")
    assert intent.kind == "comparison"
    assert intent.coverage_mode == "local"
    candidates = [_candidate("best", 200.0, 0.9), _candidate("next", 0.0, 0.8)]
    selected = select_ranked_segments(candidates, 2, 420.0, intent)
    assert [item["segment_id"] for item in selected] == ["best", "next"]
    assert all(item["selection_reason"] == "relevance_top_k" for item in selected)


def test_query_classifier_does_not_treat_generic_from_as_overview():
    intent = classify_query("从视频中找到墨西哥可乐出现的位置")
    assert intent.kind == "locate"
    assert intent.coverage_mode == "local"


def test_stage_hint_constrains_local_retrieval_to_requested_part_of_video():
    intent = classify_query("开场主持人展示了哪些可乐？")
    assert intent.coverage_mode == "stage_local"
    candidates = [_candidate("late", 340.0, 1.0), _candidate("opening", 0.0, 0.2)]
    selected = select_ranked_segments(candidates, 1, 416.2, intent)
    assert [item["segment_id"] for item in selected] == ["opening"]


def test_explicit_seconds_constrain_local_retrieval():
    intent = classify_query("300 秒左右展示了什么？")
    assert intent.coverage_mode == "time_local"
    assert intent.time_anchor_sec == 300.0
    candidates = [_candidate("far", 0.0, 1.0), _candidate("near", 300.0, 0.2)]
    selected = select_ranked_segments(candidates, 1, 416.2, intent)
    assert [item["segment_id"] for item in selected] == ["near"]


def test_sidecar_asr_aligns_srt_uses_cache_and_invalidates_on_change(tmp_path):
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")
    sidecar = video.with_suffix(".srt")
    sidecar.write_text(
        "1\n00:00:00,500 --> 00:00:03,000\n开场介绍\n\n"
        "2\n00:00:04,000 --> 00:00:07,000\n开始试喝\n",
        encoding="utf-8",
    )
    segments = [Segment("s0", 0.0, 4.0), Segment("s1", 4.0, 8.0)]
    backend = SidecarASR()

    first = enrich_segments_with_asr(str(video), segments, backend, str(tmp_path / "cache"))
    assert first["cache"] == "miss"
    assert segments[0].asr_text == "开场介绍"
    assert segments[1].asr_text == "开始试喝"

    second = enrich_segments_with_asr(str(video), segments, backend, str(tmp_path / "cache"))
    assert second["cache"] == "hit"

    sidecar.write_text(
        "1\n00:00:00,500 --> 00:00:03,000\n更新后的开场\n",
        encoding="utf-8",
    )
    third = enrich_segments_with_asr(str(video), segments, backend, str(tmp_path / "cache"))
    assert third["cache"] == "miss"
    assert segments[0].asr_text == "更新后的开场"
    assert third["cache_path"] != first["cache_path"]


def test_sidecar_asr_parses_json_segments(tmp_path):
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")
    video.with_suffix(".json").write_text(
        json.dumps({"segments": [{"start": 1.0, "end": 2.5, "text": "字幕"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    spans = SidecarASR().transcribe(str(video))
    assert [span.dump() for span in spans] == [
        {"start_sec": 1.0, "end_sec": 2.5, "text": "字幕"}
    ]


def test_dense_index_round_trip_and_dimension_validation(tmp_path):
    index = NumpyDenseIndex(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
        [{"segment_id": "a"}, {"segment_id": "b"}],
    )
    artifact = index.save(str(tmp_path), "demo/video", "siglip")
    restored = NumpyDenseIndex.load(artifact.metadata_path)
    result = restored.search(np.asarray([0.9, 0.1], dtype="float32"), top_k=1)
    assert result[0]["segment_id"] == "a"
    assert result[0]["dense_score"] > 0.9
    with pytest.raises(ValueError, match="query dim"):
        restored.search(np.asarray([1.0, 0.0, 0.0], dtype="float32"))


def test_reranker_features_include_asr_search_text_and_stable_vector_contract():
    segment = Segment(
        "s0",
        20.0,
        40.0,
        text="画面展示饮料",
        asr_text="主持人说墨西哥可乐",
        retrieval_score=0.4,
        scorer_score=0.3,
        vlm_score=0.8,
        score=0.7,
    )
    features = build_reranker_features("墨西哥可乐", segment, duration_sec=100.0)
    vector = vectorize_features(features)
    assert features["query_coverage"] == 1.0
    assert vector.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(vector).all()


def test_old_eight_feature_scorer_checkpoint_remains_loadable(tmp_path):
    checkpoint = tmp_path / "old.pkl"
    checkpoint.write_bytes(
        pickle.dumps({"model": _EightFeatureModel(), "feature_names": [f"f{i}" for i in range(8)]})
    )
    scorer = SklearnSegmentScorer(str(checkpoint))
    assert scorer.score(Segment("s0", 0.0, 10.0, asr_text="new channel")) == pytest.approx(0.75)
