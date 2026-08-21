from __future__ import annotations

from pathlib import Path

from videomemo.models import Segment
from videomemo.llm.qwen35_local import Qwen35LocalClient, _RUNTIMES, get_qwen35_runtime
from videomemo.vlm.cache import EmbeddingCache
from videomemo.vlm.segment_analyzer import QwenVLAPISegmentAnalyzer
from videomemo.vlm.siglip_embedder import _model_device


def _segment() -> Segment:
    return Segment(
        segment_id="seg-0001",
        start_sec=0.0,
        end_sec=20.0,
        frame_hash="same-frame-content",
        text="同一视频内容",
    )


def test_qwen_content_cache_key_is_independent_of_upload_path(tmp_path):
    analyzer = QwenVLAPISegmentAnalyzer(
        "http://127.0.0.1:1",
        "/model",
        cache_dir=str(tmp_path),
    )
    segment = _segment()
    assert analyzer._cache_key("/a/first.mp4", segment) == analyzer._cache_key("/b/renamed.mp4", segment)
    assert analyzer._legacy_cache_key("/a/first.mp4", segment) != analyzer._legacy_cache_key(
        "/b/renamed.mp4", segment
    )


def test_embedding_cache_migrates_legacy_path_key(tmp_path):
    cache = EmbeddingCache(str(tmp_path))
    legacy = cache.key("legacy", "/a/first.mp4")
    primary = cache.key("content-addressed", "same-frame-content")
    import numpy as np

    cache.save(legacy, np.array([1.0, 2.0], dtype="float32"))
    restored = cache.load_or_migrate(primary, (legacy,))
    assert restored is not None
    assert restored.tolist() == [1.0, 2.0]
    assert cache.load(primary) is not None


def test_model_device_falls_back_to_parameters():
    class Parameter:
        device = "cuda:1"

    class Model:
        def parameters(self):
            yield Parameter()

    assert _model_device(Model()) == "cuda:1"


def test_qwen_base_and_adapter_clients_share_one_runtime(tmp_path):
    _RUNTIMES.clear()
    model = str(tmp_path / "model")
    adapter = str(tmp_path / "adapter")
    base_runtime = get_qwen35_runtime(model, device="cuda:0")
    adapter_runtime = get_qwen35_runtime(model, adapter_path=adapter, device="cuda:0")
    assert adapter_runtime is base_runtime
    assert adapter_runtime.adapter_path == str((tmp_path / "adapter").resolve())

    base_client = Qwen35LocalClient(model, device="cuda:0")
    adapter_client = Qwen35LocalClient(model, adapter_path=adapter, device="cuda:0")
    assert base_client.runtime is adapter_client.runtime
    assert base_client.use_adapter is False
    assert adapter_client.use_adapter is True


def test_qwen_product_answer_uses_deterministic_decoding():
    captured = {}

    class FakeRuntime:
        def generate(self, messages, **kwargs):
            captured.update(kwargs)
            return '{"conclusion":"证据不足","evidence":[]}'

    client = object.__new__(Qwen35LocalClient)
    client.runtime = FakeRuntime()
    client.use_adapter = True
    client.max_new_tokens = 32
    client.num_frames_per_segment = 1
    client.generate_answer("问题", {"video_path": "", "items": []}, [])

    assert captured["temperature"] == 0.0
    assert captured["use_adapter"] is True
