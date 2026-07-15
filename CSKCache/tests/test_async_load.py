from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cskcache.v1.async_load import PrefetchRegistry, submit_disk_prefetch
from cskcache.v1.async_load.disk_prefetch import PrefetchHandle
from cskcache.v1.async_load.gpu_prefetch import submit_gpu_prefetch
from cskcache.v1.metadata import CSKCacheEntry
from cskcache.v1.storage.storage_manager import StorageManager

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def _entry(cache_id: str = "skill") -> CSKCacheEntry:
    key = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2)
    return CSKCacheEntry(
        cache_id=cache_id,
        source_start=0,
        source_end=4,
        token_ids=[10, 11, 12, 13],
        kv_by_layer={"layer0": (key, key + 100)},
    )


class _RecordingStorage:
    """Fake backend that records the exact args storage.get() was called with."""

    def __init__(self, entry: CSKCacheEntry | None) -> None:
        self._entry = entry
        self.calls: list[tuple[str, object]] = []

    def get(self, cache_id: str, trace: object = None) -> CSKCacheEntry | None:
        self.calls.append((cache_id, trace))
        return self._entry


class _GatedStorage:
    """Fake backend whose get() blocks until the test releases it -- makes
    'not ready yet' assertions deterministic instead of timing-dependent."""

    def __init__(self, entry: CSKCacheEntry, release: threading.Event) -> None:
        self._entry = entry
        self._release = release

    def get(self, cache_id: str, trace: object = None) -> CSKCacheEntry:
        self._release.wait()
        return self._entry


class _FailingStorage:
    def get(self, cache_id: str, trace: object = None) -> CSKCacheEntry:
        raise RuntimeError("boom")


def test_submit_disk_prefetch_returns_entry_from_real_storage_manager() -> None:
    """End-to-end against the real StorageManager, not just a fake."""
    entry = _entry()
    storage = StorageManager()
    storage.put(entry)

    handle = submit_disk_prefetch(storage, "skill")
    result = handle.result(timeout=5)

    assert result is not None and result.cache_id == "skill"


def test_submit_disk_prefetch_forwards_cache_id_and_trace() -> None:
    entry = _entry()
    storage = _RecordingStorage(entry)
    sentinel_trace = object()

    handle = submit_disk_prefetch(storage, "skill", trace=sentinel_trace)
    handle.result(timeout=5)

    assert storage.calls == [("skill", sentinel_trace)]


def test_prefetch_handle_blocks_until_background_work_completes() -> None:
    release = threading.Event()
    entry = _entry()
    storage = _GatedStorage(entry, release)

    handle = submit_disk_prefetch(storage, "skill")
    assert not handle.is_ready()

    release.set()
    result = handle.result(timeout=5)

    assert result is entry
    assert handle.is_ready()


def test_prefetch_handle_result_reraises_background_exception() -> None:
    handle = submit_disk_prefetch(_FailingStorage(), "skill")
    with pytest.raises(RuntimeError, match="boom"):
        handle.result(timeout=5)


def test_prefetch_handle_returns_none_for_missing_cache_id() -> None:
    storage = StorageManager()
    handle = submit_disk_prefetch(storage, "does-not-exist")
    assert handle.result(timeout=5) is None


def test_prefetch_registry_get_or_submit_dedups_same_key() -> None:
    registry = PrefetchRegistry()
    submit_calls = []

    def submit() -> PrefetchHandle:
        submit_calls.append(1)
        return submit_disk_prefetch(_RecordingStorage(_entry()), "skill")

    first = registry.get_or_submit(("r1", "skill"), submit)
    second = registry.get_or_submit(("r1", "skill"), submit)

    assert first is second
    assert len(submit_calls) == 1


def test_prefetch_registry_different_keys_submit_independently() -> None:
    registry = PrefetchRegistry()
    submit_calls = []

    def make_submit():
        def submit() -> PrefetchHandle:
            submit_calls.append(1)
            return submit_disk_prefetch(_RecordingStorage(_entry()), "skill")

        return submit

    handle_a = registry.get_or_submit(("r1", "skill"), make_submit())
    handle_b = registry.get_or_submit(("r2", "skill"), make_submit())

    assert handle_a is not handle_b
    assert len(submit_calls) == 2


def test_prefetch_registry_pop_removes_entry_and_is_one_shot() -> None:
    registry = PrefetchRegistry()
    handle = registry.get_or_submit(
        ("r1", "skill"), lambda: submit_disk_prefetch(_RecordingStorage(_entry()), "skill")
    )

    popped = registry.pop(("r1", "skill"))
    assert popped is handle
    assert registry.pop(("r1", "skill")) is None


def test_prefetch_registry_pop_missing_key_returns_none() -> None:
    registry = PrefetchRegistry()
    assert registry.pop(("no-such-req", "skill")) is None


@requires_cuda
def test_submit_gpu_prefetch_returns_correct_value() -> None:
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)

    def compute() -> torch.Tensor:
        return torch.arange(8, device=device, dtype=torch.float32) * 2.0

    handle = submit_gpu_prefetch(compute, stream)
    result = handle.result()
    torch.cuda.synchronize(device)

    expected = torch.arange(8, device=device, dtype=torch.float32) * 2.0
    assert torch.equal(result, expected)


@requires_cuda
def test_gpu_prefetch_handle_result_orders_default_stream_after_prefetch() -> None:
    """result() must make the *current* stream wait on the prefetch's event
    (a real cross-stream ordering dependency, not a CPU-blocking call): a
    trivial op queued on the default stream right after result(), with no
    manual synchronize() in between, still has to wait its turn behind the
    prefetch and therefore sees the fully-written tensor once the device
    catches up."""
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)

    def compute() -> torch.Tensor:
        base = torch.ones(2048, 2048, device=device, dtype=torch.float32)
        return base @ base

    handle = submit_gpu_prefetch(compute, stream)
    result = handle.result()
    doubled = result * 2  # queued on the default stream, ordered after result()
    torch.cuda.synchronize(device)

    assert handle._event.query()  # the completion event did fire by now
    expected = torch.full((2048, 2048), 2048.0, device=device)
    assert torch.equal(result, expected)
    assert torch.equal(doubled, expected * 2)


@requires_cuda
def test_submit_gpu_prefetch_handles_on_different_streams_are_independent() -> None:
    device = torch.device("cuda:0")
    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)

    handle_a = submit_gpu_prefetch(
        lambda: torch.full((4,), 1.0, device=device), stream_a
    )
    handle_b = submit_gpu_prefetch(
        lambda: torch.full((4,), 2.0, device=device), stream_b
    )

    result_a = handle_a.result()
    result_b = handle_b.result()
    torch.cuda.synchronize(device)

    assert torch.equal(result_a, torch.full((4,), 1.0, device=device))
    assert torch.equal(result_b, torch.full((4,), 2.0, device=device))
