from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cskcache.v1.core.cache_engine import CSKCacheEngine
from cskcache.v1.core.config import CSKCacheConfig
from cskcache.v1.kv_transfer.gpu_connector import VLLMPagedGPUConnector
from cskcache.v1.metadata import CSKCacheEntry, CSKCacheMode, CSKLoadPlan, CSKReqMeta
from cskcache.v1.storage.storage_manager import StorageManager

BLOCK_SIZE = 4
NUM_BLOCKS = 2
HEADS = 2
DIM = 3


@contextmanager
def _raises(error_type: type[Exception], message: str) -> Iterator[None]:
    try:
        yield
    except error_type as exc:
        assert message in str(exc)
    else:
        raise AssertionError(f"Expected {error_type.__name__}: {message}")


class _RecordingConnector(VLLMPagedGPUConnector):
    def __init__(self, block_size: int) -> None:
        super().__init__(block_size)
        self.models: list[object | None] = []

    def set_model(self, model: object | None) -> None:
        self.models.append(model)
        super().set_model(model)


def _entry(length: int = 4, device: torch.device | str = "cpu") -> CSKCacheEntry:
    key = torch.arange(
        length * HEADS * DIM, dtype=torch.float32, device=device
    ).reshape(length, HEADS, DIM)
    value = key + 500.0
    return CSKCacheEntry(
        cache_id="skill",
        source_start=0,
        source_end=length,
        token_ids=list(range(length)),
        kv_by_layer={"layer0": (key, value)},
    )


def _packed_cache() -> torch.Tensor:
    # vLLM packed layout: [2 (K/V), num_blocks, block_size, heads, dim].
    return torch.zeros(2, NUM_BLOCKS, BLOCK_SIZE, HEADS, DIM)


def test_connector_scatter_gather_roundtrip() -> None:
    conn = VLLMPagedGPUConnector(BLOCK_SIZE)
    cache = _packed_cache()
    conn.bind_kv_caches({"layer0": cache})
    conn.set_model(None)  # start==0 target -> no RoPE needed
    entry = _entry(4)
    plan = CSKLoadPlan(
        req_id="r1",
        cache_id="skill",
        mode=CSKCacheMode.REUSE,
        start=0,
        end=4,
        token_ids=tuple(entry.token_ids),
        source_offset=0,
    )
    assert conn.to_gpu(entry, plan, block_ids=[0]) == (1, 1, 0)
    gk, gv = conn.gather(cache, [0], 0, 4)
    ek, ev = entry.kv_by_layer["layer0"]
    assert torch.equal(gk, ek) and torch.equal(gv, ev)


class _StreamRecordingConnector(VLLMPagedGPUConnector):
    def __init__(self, block_size: int) -> None:
        super().__init__(block_size)
        self.received_streams: list[object] = []

    def to_gpu(self, entry, plan, block_ids, trace=None, prefetch_stream=None):
        self.received_streams.append(prefetch_stream)
        return super().to_gpu(
            entry, plan, block_ids, trace=trace, prefetch_stream=prefetch_stream
        )


def test_engine_load_passes_no_stream_when_gpu_prefetch_disabled() -> None:
    """gpu_prefetch_enabled defaults to False: load() must call to_gpu()
    with prefetch_stream=None, reproducing the original sequential path."""
    storage = StorageManager()
    entry = _entry(4)
    storage.put(entry)
    conn = _StreamRecordingConnector(BLOCK_SIZE)
    eng = CSKCacheEngine(
        CSKCacheConfig(), storage, block_size=BLOCK_SIZE, gpu_connector=conn
    )
    eng.register_kv_caches({"layer0": _packed_cache()})
    plan = CSKLoadPlan(
        req_id="r1",
        cache_id="skill",
        mode=CSKCacheMode.REUSE,
        start=0,
        end=4,
        token_ids=tuple(entry.token_ids),
        source_offset=0,
    )

    eng.load([CSKReqMeta(plan=plan, block_ids=([0],))], model=None)

    assert conn.received_streams == [None]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_engine_load_passes_real_stream_when_gpu_prefetch_enabled() -> None:
    device = torch.device("cuda:0")
    storage = StorageManager()
    entry = _entry(4, device=device)
    storage.put(entry)
    conn = _StreamRecordingConnector(BLOCK_SIZE)
    eng = CSKCacheEngine(
        CSKCacheConfig(gpu_prefetch_enabled=True),
        storage,
        block_size=BLOCK_SIZE,
        gpu_connector=conn,
    )
    eng.register_kv_caches(
        {"layer0": torch.zeros(2, NUM_BLOCKS, BLOCK_SIZE, HEADS, DIM, device=device)}
    )
    plan = CSKLoadPlan(
        req_id="r1",
        cache_id="skill",
        mode=CSKCacheMode.REUSE,
        start=0,
        end=4,
        token_ids=tuple(entry.token_ids),
        source_offset=0,
    )

    eng.load([CSKReqMeta(plan=plan, block_ids=([0],))], model=None)

    assert len(conn.received_streams) == 1
    assert isinstance(conn.received_streams[0], torch.cuda.Stream)
    # Same connector, second load: the stream is created once and reused.
    eng.load([CSKReqMeta(plan=plan, block_ids=([0],))], model=None)
    assert conn.received_streams[0] is conn.received_streams[1]


def test_engine_load_uses_connector() -> None:
    storage = StorageManager()
    entry = _entry(4)
    storage.put(entry)
    conn = VLLMPagedGPUConnector(BLOCK_SIZE)
    eng = CSKCacheEngine(
        CSKCacheConfig(), storage, block_size=BLOCK_SIZE, gpu_connector=conn
    )
    cache = _packed_cache()
    eng.register_kv_caches({"layer0": cache})
    plan = CSKLoadPlan(
        req_id="r1",
        cache_id="skill",
        mode=CSKCacheMode.REUSE,
        start=0,
        end=4,
        token_ids=tuple(entry.token_ids),
        source_offset=0,
    )
    eng.load([CSKReqMeta(plan=plan, block_ids=([0],))], model=None)
    gk, gv = conn.gather(cache, [0], 0, 4)
    ek, ev = entry.kv_by_layer["layer0"]
    assert torch.equal(gk, ek) and torch.equal(gv, ev)


def test_engine_kv_only_load_preserves_registered_model() -> None:
    storage = StorageManager()
    entry = _entry(4)
    storage.put(entry)
    conn = _RecordingConnector(BLOCK_SIZE)
    eng = CSKCacheEngine(
        CSKCacheConfig(), storage, block_size=BLOCK_SIZE, gpu_connector=conn
    )
    cache = _packed_cache()
    eng.register_kv_caches({"layer0": cache})
    model = torch.nn.Module()
    eng.register_model(model)
    plan = CSKLoadPlan(
        req_id="r1",
        cache_id="skill",
        mode=CSKCacheMode.REUSE,
        start=0,
        end=4,
        token_ids=tuple(entry.token_ids),
        source_offset=0,
    )

    # This matches vLLM's kv_connector_no_forward path: no model is present in
    # ForwardContext, but the model registered during worker init must survive.
    eng.load([CSKReqMeta(plan=plan, block_ids=([0],))], model=None)

    assert conn.models == [model]


def test_connector_rejects_missing_layer_before_scatter() -> None:
    conn = VLLMPagedGPUConnector(BLOCK_SIZE)
    cache = _packed_cache()
    conn.bind_kv_caches({"different-layer": cache})
    entry = _entry(4)
    plan = CSKLoadPlan(
        req_id="r-missing",
        cache_id="skill",
        mode=CSKCacheMode.REUSE,
        start=0,
        end=4,
        token_ids=tuple(entry.token_ids),
        source_offset=0,
    )

    with _raises(RuntimeError, "missing_layers=1"):
        conn.to_gpu(entry, plan, block_ids=[0])
    assert torch.count_nonzero(cache) == 0


def _multi_layer_entry(
    num_layers: int, length: int = 4, device: torch.device | str = "cpu"
) -> CSKCacheEntry:
    kv_by_layer = {}
    for layer_index in range(num_layers):
        base = layer_index * 1000.0
        key = base + torch.arange(
            length * HEADS * DIM, dtype=torch.float32, device=device
        ).reshape(length, HEADS, DIM)
        value = key + 500.0
        kv_by_layer[f"layer{layer_index}"] = (key, value)
    return CSKCacheEntry(
        cache_id="skill",
        source_start=0,
        source_end=length,
        token_ids=list(range(length)),
        kv_by_layer=kv_by_layer,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_to_gpu_pipelined_path_matches_sequential_path() -> None:
    """The opt-in prefetch_stream path must be a pure speed change: scatter
    the exact same multi-layer entry both ways and require byte-identical
    paged-cache contents and identical (expected, scattered, skipped)
    counts. This is the correctness contract the whole pipelining change
    stands or falls on.
    """
    device = torch.device("cuda:0")
    num_layers = 6
    entry = _multi_layer_entry(num_layers, length=4, device=device)
    plan = CSKLoadPlan(
        req_id="r-pipeline",
        cache_id="skill",
        mode=CSKCacheMode.REUSE,
        start=0,
        end=4,
        token_ids=tuple(entry.token_ids),
        source_offset=0,
    )

    def fresh_caches() -> dict[str, torch.Tensor]:
        return {
            f"layer{i}": torch.zeros(2, NUM_BLOCKS, BLOCK_SIZE, HEADS, DIM, device=device)
            for i in range(num_layers)
        }

    sequential_caches = fresh_caches()
    conn_sequential = VLLMPagedGPUConnector(BLOCK_SIZE)
    conn_sequential.bind_kv_caches(sequential_caches)
    conn_sequential.set_model(None)
    sequential_counts = conn_sequential.to_gpu(entry, plan, block_ids=[0])

    pipelined_caches = fresh_caches()
    conn_pipelined = VLLMPagedGPUConnector(BLOCK_SIZE)
    conn_pipelined.bind_kv_caches(pipelined_caches)
    conn_pipelined.set_model(None)
    prefetch_stream = torch.cuda.Stream(device=device)
    pipelined_counts = conn_pipelined.to_gpu(
        entry, plan, block_ids=[0], prefetch_stream=prefetch_stream
    )
    torch.cuda.synchronize(device)

    assert pipelined_counts == sequential_counts == (num_layers, num_layers, 0)
    for layer_name in sequential_caches:
        assert torch.equal(sequential_caches[layer_name], pipelined_caches[layer_name])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL KV-TRANSFER TESTS PASSED")
