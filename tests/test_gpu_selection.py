from scripts.remote.select_gpus import GPUState, select_pair, select_safe_pair


def _gpu(index: int, free: int, used: int = 1, util: int = 0, pids=()) -> GPUState:
    return GPUState(
        index=index,
        uuid=f"GPU-{index}",
        total_mib=24576,
        used_mib=used,
        free_mib=free,
        utilization_pct=util,
        compute_pids=tuple(pids),
    )


def test_memory_only_selector_keeps_distinct_devices():
    assert select_pair([(0, 24000), (1, 8000)], 20000, 6000) == (0, 1)


def test_safe_selector_rejects_compute_process_even_with_free_memory():
    states = [_gpu(0, 24000, pids=(1234,)), _gpu(1, 24000), _gpu(2, 8000)]
    assert select_safe_pair(states, 20000, 6000, 512, 5) == (1, 2)


def test_safe_selector_rejects_busy_utilization_and_memory():
    states = [_gpu(0, 24000, util=90), _gpu(1, 23000, used=800), _gpu(2, 22000), _gpu(3, 7000)]
    assert select_safe_pair(states, 20000, 6000, 512, 5) == (2, 3)


def test_preferred_pair_is_validated_instead_of_trusted():
    states = [_gpu(0, 24000, pids=(9,)), _gpu(1, 24000), _gpu(2, 24000)]
    assert select_safe_pair(states, 20000, 6000, 512, 5, 0, 1) is None
    assert select_safe_pair(states, 20000, 6000, 512, 5, 1, 2) == (1, 2)
