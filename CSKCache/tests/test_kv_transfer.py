from __future__ import annotations

import sys
from pathlib import Path

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


def _entry(length: int = 4) -> CSKCacheEntry:
    key = torch.arange(length * HEADS * DIM, dtype=torch.float32).reshape(length, HEADS, DIM)
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
    conn.to_gpu(entry, plan, block_ids=[0])
    gk, gv = conn.gather(cache, [0], 0, 4)
    ek, ev = entry.kv_by_layer["layer0"]
    assert torch.equal(gk, ek) and torch.equal(gv, ev)


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL KV-TRANSFER TESTS PASSED")
