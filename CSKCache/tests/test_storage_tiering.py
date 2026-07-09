from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cskcache.v1.metadata import CSKCacheEntry
from cskcache.v1.registry import CSKCacheRegistry
from cskcache.v1.storage.abstract_backend import entry_nbytes
from cskcache.v1.storage.local_cpu_backend import LocalCPUBackend
from cskcache.v1.storage.local_disk_backend import LocalDiskBackend
from cskcache.v1.storage.storage_manager import StorageManager


def _make_entry(cache_id: str, length: int = 4) -> CSKCacheEntry:
    num_kv_heads, head_dim = 2, 3
    key = torch.arange(length * num_kv_heads * head_dim, dtype=torch.float32)
    key = key.reshape(length, num_kv_heads, head_dim)
    value = key + 1000.0
    return CSKCacheEntry(
        cache_id=cache_id,
        source_start=0,
        source_end=length,
        token_ids=list(range(length)),
        kv_by_layer={"layer0": (key, value), "layer1": (key + 1, value + 1)},
    )


def test_cpu_only_default_behaves_like_registry() -> None:
    mgr = StorageManager()
    e = _make_entry("a")
    mgr.put(e)
    assert mgr.contains("a")
    got = mgr.get("a")
    assert got is not None and got.cache_id == "a"
    assert mgr.size_bytes("disk") == 0
    assert set(mgr.keys()) == {"a"}


def test_disk_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        disk = LocalDiskBackend(tmp)
        e = _make_entry("skill-x", length=5)
        disk.put(e)
        assert disk.contains("skill-x")
        back = disk.get("skill-x")
        assert back is not None
        assert back.token_ids == e.token_ids
        for layer in e.kv_by_layer:
            k0, v0 = e.kv_by_layer[layer]
            k1, v1 = back.kv_by_layer[layer]
            assert torch.equal(k0, k1) and torch.equal(v0, v1)


def test_disk_index_persists_across_reopen() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        LocalDiskBackend(tmp).put(_make_entry("persist"))
        reopened = LocalDiskBackend(tmp)
        assert reopened.contains("persist")
        assert reopened.get("persist") is not None


def test_spill_to_disk_and_promote_back() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        one = entry_nbytes(_make_entry("size-probe"))
        # Budget for ~1.5 entries: the second put must evict the first (LRU).
        mgr = StorageManager(
            cpu_backend=LocalCPUBackend(),
            disk_backend=LocalDiskBackend(tmp),
            cpu_max_bytes=int(one * 1.5),
        )
        mgr.put(_make_entry("first"))
        mgr.put(_make_entry("second"))
        # first is LRU -> spilled to disk; second stays hot.
        assert mgr.size_bytes("cpu") <= int(one * 1.5)
        assert "first" in mgr._disk.keys()
        assert mgr._cpu.contains("second")
        assert not mgr._cpu.contains("first")
        # get() must still find first (from disk) and promote it.
        got = mgr.get("first")
        assert got is not None and got.cache_id == "first"
        assert mgr._cpu.contains("first")


def test_lru_eviction_order() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        one = entry_nbytes(_make_entry("size-probe"))
        mgr = StorageManager(
            cpu_backend=LocalCPUBackend(),
            disk_backend=LocalDiskBackend(tmp),
            cpu_max_bytes=int(one * 2.5),
        )
        mgr.put(_make_entry("a"))
        mgr.put(_make_entry("b"))
        mgr.put(_make_entry("c"))
        mgr.get("a")  # touch a -> b is now the coldest
        mgr.put(_make_entry("d"))  # over budget -> evict coldest (b)
        assert not mgr._cpu.contains("b")
        assert mgr._cpu.contains("a")
        assert mgr.contains("b")  # still retrievable from disk


def test_registry_shim_uses_storage_manager() -> None:
    reg = CSKCacheRegistry()
    reg.put(_make_entry("r1"))
    assert "r1" in reg
    assert reg.get("r1") is not None
    assert {e.cache_id for e in reg.entries()} == {"r1"}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL STORAGE TESTS PASSED")
