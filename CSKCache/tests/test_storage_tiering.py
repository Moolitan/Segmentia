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


def _print_test(name: str, purpose: str) -> None:
    print(f"\n=== {name} ===")
    print(f"purpose: {purpose}")


def _print_tiers(mgr: StorageManager, label: str) -> None:
    cpu_keys = sorted(mgr._cpu.keys())
    disk_keys = sorted(mgr._disk.keys()) if mgr._disk is not None else []
    print(f"{label}: cpu_keys={cpu_keys}, disk_keys={disk_keys}")


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
    """
    Verifies the behavior of the CPU-only (in-memory) mode.
    """
    _print_test(
        "test_cpu_only_default_behaves_like_registry",
        "Store and retrieve one entry using the default CPU-only backend.",
    )
    mgr = StorageManager()
    e = _make_entry("a")
    mgr.put(e)
    print(
        f"stored cache_id={e.cache_id!r}, tokens={len(e.token_ids)}, "
        f"layers={len(e.kv_by_layer)}, cpu_bytes={mgr.size_bytes('cpu')}"
    )
    _print_tiers(mgr, "after put")
    assert mgr.contains("a")
    got = mgr.get("a")
    assert got is not None and got.cache_id == "a"
    assert mgr.size_bytes("disk") == 0
    assert set(mgr.keys()) == {"a"}
    print(f"retrieved cache_id={got.cache_id!r}; disk_bytes=0")


def test_disk_roundtrip() -> None:
    """
    Writes a `CSKCacheEntry` to a `.pt` file, reloads it, and verifies that the K/V tensors and token IDs are identical across all layers.
    """
    _print_test(
        "test_disk_roundtrip",
        "Write .pt/.json files and compare every reloaded K/V tensor.",
    )
    with tempfile.TemporaryDirectory() as tmp:
        disk = LocalDiskBackend(tmp)
        e = _make_entry("skill-x", length=5)
        first_key, _ = e.kv_by_layer["layer0"]
        print(
            f"entry cache_id={e.cache_id!r}, tokens={len(e.token_ids)}, "
            f"layers={list(e.kv_by_layer)}, kv_shape={tuple(first_key.shape)}, "
            f"dtype={first_key.dtype}"
        )
        disk.put(e)
        files = sorted(path.name for path in Path(tmp).iterdir())
        print(f"disk_dir={tmp}")
        print(f"created_files={files}, payload_bytes={disk.size_bytes()}")
        assert disk.contains("skill-x")
        back = disk.get("skill-x")
        assert back is not None
        assert back.token_ids == e.token_ids
        for layer in e.kv_by_layer:
            k0, v0 = e.kv_by_layer[layer]
            k1, v1 = back.kv_by_layer[layer]
            assert torch.equal(k0, k1) and torch.equal(v0, v1)
            print(
                f"roundtrip layer={layer}: key_equal={torch.equal(k0, k1)}, "
                f"value_equal={torch.equal(v0, v1)}"
            )
        print(f"token_ids_equal={back.token_ids == e.token_ids}")


def test_disk_index_persists_across_reopen() -> None:
    """
    Recreates the `LocalDiskBackend` and verifies that the index can be rebuilt via the `.json` sidecar to load the existing cache.
    """
    _print_test(
        "test_disk_index_persists_across_reopen",
        "Rebuild the cache_id index from JSON sidecars after reopening.",
    )
    with tempfile.TemporaryDirectory() as tmp:
        original = LocalDiskBackend(tmp)
        original.put(_make_entry("persist"))
        print(f"before reopen: keys={sorted(original.keys())}")
        reopened = LocalDiskBackend(tmp)
        print(f"after reopen: keys={sorted(reopened.keys())}")
        assert reopened.contains("persist")
        loaded = reopened.get("persist")
        assert loaded is not None
        print(
            f"reloaded cache_id={loaded.cache_id!r}, tokens={len(loaded.token_ids)}, "
            f"layers={len(loaded.kv_by_layer)}"
        )


def test_spill_to_disk_and_promote_back() -> None:
    """
    Spills the LRU entry to disk when the CPU budget is exceeded, then loads it back to the CPU upon access.
    """
    _print_test(
        "test_spill_to_disk_and_promote_back",
        "Spill the CPU LRU entry to disk, then promote it on access.",
    )
    with tempfile.TemporaryDirectory() as tmp:
        one = entry_nbytes(_make_entry("size-probe"))
        budget = int(one * 1.5)
        # Budget for ~1.5 entries: the second put must evict the first (LRU).
        mgr = StorageManager(
            cpu_backend=LocalCPUBackend(),
            disk_backend=LocalDiskBackend(tmp),
            cpu_max_bytes=budget,
        )
        print(f"entry_bytes={one}, cpu_budget={budget}")
        mgr.put(_make_entry("first"))
        _print_tiers(mgr, "after put first")
        mgr.put(_make_entry("second"))
        _print_tiers(mgr, "after put second and spill")
        # first is LRU -> spilled to disk; second stays hot.
        assert mgr.size_bytes("cpu") <= budget
        assert "first" in mgr._disk.keys()
        assert mgr._cpu.contains("second")
        assert not mgr._cpu.contains("first")
        # get() must still find first (from disk) and promote it.
        got = mgr.get("first")
        assert got is not None and got.cache_id == "first"
        assert mgr._cpu.contains("first")
        _print_tiers(mgr, "after get first and promote")


def test_lru_eviction_order() -> None:
    """
    Verifies the LRU eviction order based on data temperature (hot/cold).
    """
    _print_test(
        "test_lru_eviction_order",
        "Touch a, then verify b becomes the coldest entry and spills.",
    )
    with tempfile.TemporaryDirectory() as tmp:
        one = entry_nbytes(_make_entry("size-probe"))
        budget = int(one * 2.5)
        mgr = StorageManager(
            cpu_backend=LocalCPUBackend(),
            disk_backend=LocalDiskBackend(tmp),
            cpu_max_bytes=budget,
        )
        print(f"entry_bytes={one}, cpu_budget={budget}")
        mgr.put(_make_entry("a"))
        mgr.put(_make_entry("b"))
        mgr.put(_make_entry("c"))
        _print_tiers(mgr, "after put a,b,c")
        mgr.get("a")  # touch a -> b is now the coldest
        _print_tiers(mgr, "after touch a")
        mgr.put(_make_entry("d"))  # over budget -> evict coldest (b)
        _print_tiers(mgr, "after put d and spill coldest")
        assert not mgr._cpu.contains("b")
        assert mgr._cpu.contains("a")
        assert mgr.contains("b")  # still retrievable from disk
        print("verified: b left CPU but remains retrievable from disk")


def test_registry_shim_uses_storage_manager() -> None:
    """
    Verifies compatibility between the legacy registry interface and the new `StorageManager`.
    """
    _print_test(
        "test_registry_shim_uses_storage_manager",
        "Exercise the legacy registry API through StorageManager.",
    )
    reg = CSKCacheRegistry()
    reg.put(_make_entry("r1"))
    print(f"registry keys after put={sorted(e.cache_id for e in reg.entries())}")
    assert "r1" in reg
    loaded = reg.get("r1")
    assert loaded is not None
    assert {e.cache_id for e in reg.entries()} == {"r1"}
    print(f"registry get returned cache_id={loaded.cache_id!r}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL STORAGE TESTS PASSED")
