from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/wsh/vllm")

from cskcache.integration.vllm.v1_adapter import CSKCacheConnectorV1Impl
from cskcache.v1.core.cache_engine import CSKCacheEngine
from cskcache.v1.core.config import CSKCacheConfig
from cskcache.v1.metadata import CSKCacheEntry
from cskcache.v1.storage.storage_manager import StorageManager


class _Request:
    """Small stand-in for vLLM Request.

    The production path receives kv_transfer_params from the OpenAI request and
    stores it on vllm.v1.request.Request. These tests exercise only CSKCache's
    scheduler-side reuse-signal logic, so a tiny object with the same attributes is
    enough and avoids starting a vLLM server.
    """

    def __init__(self, token_ids: list[int], cskcache: dict | None) -> None:
        self.request_id = "req-reuse-signal"
        self.prompt_token_ids = token_ids
        self.all_token_ids = token_ids
        self.kv_transfer_params = (
            {"cskcache": cskcache} if cskcache is not None else None
        )


def _make_entry(cache_id: str = "skill-demo") -> CSKCacheEntry:
    """Build one cached skill entry with deterministic token IDs."""

    length = 6
    key = torch.zeros(length, 1, 2)
    value = torch.ones(length, 1, 2)
    return CSKCacheEntry(
        cache_id=cache_id,
        source_start=0,
        source_end=length,
        token_ids=[10, 11, 12, 13, 14, 15],
        kv_by_layer={"layer.0": (key, value)},
    )


def _make_impl(entry: CSKCacheEntry) -> CSKCacheConnectorV1Impl:
    return _make_impl_many([entry])


def _make_impl_many(entries: list[CSKCacheEntry]) -> CSKCacheConnectorV1Impl:
    """Construct only the adapter fields needed by reuse scheduling tests."""

    storage = StorageManager()
    for entry in entries:
        storage.put(entry)
    impl = CSKCacheConnectorV1Impl.__new__(CSKCacheConnectorV1Impl)
    impl._engine = CSKCacheEngine(CSKCacheConfig(), storage, block_size=4)
    return impl


def test_explicit_span_reuse_signal_caps_then_loads_mid_prompt() -> None:
    """Explicit span path for [prefix][skill][trailing].

    The reuse signal says the current skill is [2, 8). Since the scheduler starts
    at token 0, CSKCache should first cap normal prefill to 2 tokens, then
    report a 6-token in-process load at the span boundary.
    """

    entry = _make_entry()
    impl = _make_impl(entry)
    request = _Request(
        [1, 2, *entry.token_ids, 90, 91],
        {
            "enabled": True,
            "cache_id": entry.cache_id,
            "target_start": 2,
            "target_end": 8,
        },
    )

    matched, load_async = impl.get_num_new_matched_tokens(request, 0)
    assert (matched, load_async) == (0, False)
    assert impl.cap_prefill_before_reuse(request, 0, 32) == 2
    assert impl.get_boundary_reuse_load_tokens(request, 2) == entry.length
    plan = impl._engine._plans[request.request_id]
    assert (plan.cache_id, plan.start, plan.end, plan.source_offset) == (
        entry.cache_id,
        2,
        8,
        0,
    )
    assert plan.token_ids == tuple(entry.token_ids)


def test_disabled_reuse_signal_does_not_fallback_to_matching() -> None:
    """enabled=false is an explicit opt-out, not an absent reuse signal.

    The prompt contains the cached tokens, but CSKCache does not scan prompt
    tokens for production reuse. Explicit disable simply keeps normal prefill.
    """

    entry = _make_entry()
    impl = _make_impl(entry)
    request = _Request([1, 2, *entry.token_ids], {"enabled": False})

    matched, load_async = impl.get_num_new_matched_tokens(request, 0)
    assert (matched, load_async) == (0, False)
    assert not impl._engine._plans
    assert not impl._engine._pending_reuses


def test_reuse_signal_stale_token_slice_still_builds_plan() -> None:
    """Reuse signals are control inputs, not prompt token validators.

    The target span length and bounds must be valid, but CSKCache does not
    compare the prompt token slice against the cached entry token IDs.
    """

    entry = _make_entry()
    impl = _make_impl(entry)
    request = _Request(
        [1, 2, 10, 11, 999, 13, 14, 15],
        {
            "enabled": True,
            "cache_id": entry.cache_id,
            "target_start": 2,
            "target_end": 8,
        },
    )

    matched, _ = impl.get_num_new_matched_tokens(request, 0)
    assert matched == 0
    assert impl.cap_prefill_before_reuse(request, 0, 32) == 2
    assert impl.get_boundary_reuse_load_tokens(request, 2) == entry.length
    assert impl._engine._plans[request.request_id].token_ids == (
        10,
        11,
        999,
        13,
        14,
        15,
    )


def test_multiple_reuse_entries_advance_through_adapter_hooks() -> None:
    first = _make_entry("first")
    second = _make_entry("second")
    request = _Request(
        [1, *first.token_ids, 90, 91, *second.token_ids, 99],
        {
            "operation": "reuse",
            "entries": [
                {"cache_id": "first", "target_start": 1, "target_end": 7},
                {"cache_id": "second", "target_start": 9, "target_end": 15},
            ],
        },
    )
    impl = _make_impl_many([first, second])

    assert impl.get_num_new_matched_tokens(request, 0) == (0, False)
    assert impl.cap_prefill_before_reuse(request, 0, 32) == 1
    assert impl.get_boundary_reuse_load_tokens(request, 1) == 6
    first_plan = impl._engine._plans[request.request_id]
    assert (first_plan.cache_id, first_plan.start, first_plan.end) == (
        "first",
        1,
        7,
    )
    impl._engine.update_reuse_after_alloc(
        request.request_id, ([0],), num_external_tokens=6
    )
    impl._engine.build_meta({request.request_id: 0})

    assert impl.cap_prefill_before_reuse(request, 7, 32) == 2
    assert impl.get_boundary_reuse_load_tokens(request, 9) == 6
    second_plan = impl._engine._plans[request.request_id]
    assert (second_plan.cache_id, second_plan.start, second_plan.end) == (
        "second",
        9,
        15,
    )


if __name__ == "__main__":
    test_explicit_span_reuse_signal_caps_then_loads_mid_prompt()
    # test_disabled_reuse_signal_does_not_fallback_to_matching()
    # test_reuse_signal_stale_token_slice_still_builds_plan()
    print("CSKCache request reuse signal tests passed")
