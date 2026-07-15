from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cskcache.logging import init_logger
from cskcache.v1.compute.gate import CSKProbeAccumulator
from cskcache.v1.core import cache_engine as cache_engine_module
from cskcache.v1.core.cache_engine import CSKCacheEngine
from cskcache.v1.core.config import CSKCacheConfig
from cskcache.v1.metadata import CSKCacheEntry
from cskcache.v1.storage.storage_manager import StorageManager


class _RecordingGPUConnector:
    def __init__(self) -> None:
        self.loaded: list[tuple[str, int, int]] = []

    def bind_kv_caches(self, kv_caches) -> None:
        return

    def set_model(self, model) -> None:
        return

    def to_gpu(
        self, entry, plan, block_ids, trace=None, prefetch_stream=None
    ) -> tuple[int, int, int]:
        self.loaded.append((entry.cache_id, plan.start, plan.end))
        layers = len(entry.kv_by_layer)
        return layers, layers, 0

    def reuse_slice(self, *args, **kwargs):
        raise AssertionError("not used")

    def gather(self, *args, **kwargs):
        raise AssertionError("not used")


def test_owned_logger_writes_cskcache_prefix() -> None:
    stream = io.StringIO()
    with redirect_stderr(stream):
        logger = init_logger("cskcache.tests.owned_logger")
        logger.info("visible event req_id=r1")
    output = stream.getvalue()
    assert "CSKCache INFO: visible event req_id=r1" in output


def test_reuse_lifecycle_logs_once_and_includes_worker_completion() -> None:
    entry = CSKCacheEntry(
        cache_id="doc-coauthoring",
        source_start=0,
        source_end=4,
        token_ids=[10, 11, 12, 13],
        kv_by_layer={"layer0": (torch.zeros(4, 1), torch.ones(4, 1))},
    )
    storage = StorageManager()
    storage.put(entry)
    gpu = _RecordingGPUConnector()
    engine = CSKCacheEngine(
        CSKCacheConfig(), storage, block_size=4, gpu_connector=gpu
    )
    prompt = [1, 2, *entry.token_ids, 99]
    signal = {
        "cskcache": {
            "cache_id": entry.cache_id,
            "target_start": 2,
            "target_end": 6,
        }
    }

    stream = io.StringIO()
    handler = cache_engine_module.logger.handlers[0]
    original_stream = handler.stream
    handler.setStream(stream)
    try:
        assert engine.get_num_new_matched_tokens("r1", prompt, 0, signal) == (
            0,
            False,
        )
        assert engine.cap_prefill_before_reuse("r1", prompt, 0, 32, signal) == 2
        assert engine.get_boundary_reuse_load_tokens("r1", prompt, 2) == 4
        engine.update_reuse_after_alloc("r1", ([0, 1],), 4)
        requests, _, _, _ = engine.build_meta({"r1": 0})
        engine.load(requests)
        engine.on_finished(["r1"])
    finally:
        handler.setStream(original_stream)

    output = stream.getvalue()
    assert output.count("reuse signal accepted req_id=r1") == 1
    assert "prompt_tokens=7 prefix_frontier=0 entries=1" in output
    assert "target=[2,6) skill_tokens=4" in output
    assert "gap_from_frontier=2 gap_from_previous_entry=2" in output
    assert "tokens_after_skill=1" in output
    assert "cache_id=doc-coauthoring entry=1/1" in output
    assert "load dispatched req_id=r1 cache_id=doc-coauthoring" in output
    assert "KV load completed req_id=r1 cache_id=doc-coauthoring" in output
    assert "source=[0,4) target=[2,6) rope_delta=2 tokens=4" in output
    assert "expected_layers=1 scattered_layers=1 skipped_layers=0" in output
    assert "scheduler_state_remaining=0" in output
    assert gpu.loaded == [("doc-coauthoring", 2, 6)]


def test_probe_layer_coverage_mismatch_fails_closed() -> None:
    entry = CSKCacheEntry(
        cache_id="two-layer-skill",
        source_start=0,
        source_end=2,
        token_ids=[10, 11],
        kv_by_layer={
            "layer0": (torch.ones(2, 1), torch.ones(2, 1)),
            "layer1": (torch.ones(2, 1), torch.ones(2, 1)),
        },
    )
    storage = StorageManager()
    storage.put(entry)
    engine = CSKCacheEngine(CSKCacheConfig(), storage, block_size=4)
    accumulator = CSKProbeAccumulator(
        req_id="r-missing-layer",
        cache_id=entry.cache_id,
        tau=0.15,
        gate_metric="max",
    )
    tensor = torch.ones(2, 1)
    accumulator.add_layer(
        "layer0",
        reuse_key=tensor,
        reuse_value=tensor,
        recompute_key=tensor,
        recompute_value=tensor,
    )
    engine._probe_accumulators[accumulator.req_id] = accumulator

    try:
        engine.decide_probes()
    except RuntimeError as exc:
        assert "layer coverage mismatch" in str(exc)
        assert "expected_layers=2 captured_layers=1 missing_layers=1" in str(exc)
    else:
        raise AssertionError("Incomplete probe layer coverage must fail closed")
    assert accumulator.req_id not in engine._probe_accumulators


if __name__ == "__main__":
    test_owned_logger_writes_cskcache_prefix()
    test_reuse_lifecycle_logs_once_and_includes_worker_completion()
    test_probe_layer_coverage_mismatch_fails_closed()
    print("CSKCache logging tests passed")
