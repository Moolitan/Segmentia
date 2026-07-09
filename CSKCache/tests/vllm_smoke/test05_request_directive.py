from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/wsh/vllm")

from cskcache.integration.vllm.v1_adapter import CSKCacheConnectorV1Impl
from cskcache.v1.matcher import SegmentCatalog
from cskcache.v1.metadata import CSKCacheEntry
from cskcache.v1.registry import CSKCacheRegistry


class _Config:
    """Minimal connector config used by the adapter methods under test."""

    probe_enabled = False


class _Request:
    """Small stand-in for vLLM Request.

    The production path receives kv_transfer_params from the OpenAI request and
    stores it on vllm.v1.request.Request. These tests exercise only CSKCache's
    scheduler-side directive logic, so a tiny object with the same attributes is
    enough and avoids starting a vLLM server.
    """

    def __init__(self, token_ids: list[int], cskcache: dict | None) -> None:
        self.request_id = "req-directive"
        self.prompt_token_ids = token_ids
        self.all_token_ids = token_ids
        self.kv_transfer_params = (
            {"cskcache": cskcache} if cskcache is not None else None
        )


def _make_entry() -> CSKCacheEntry:
    """Build one cached skill entry with deterministic token identity."""

    length = 6
    key = torch.zeros(length, 1, 2)
    value = torch.ones(length, 1, 2)
    return CSKCacheEntry(
        cache_id="skill-demo",
        source_start=0,
        source_end=length,
        token_ids=[10, 11, 12, 13, 14, 15],
        kv_by_layer={"layer.0": (key, value)},
    )


def _make_impl(entry: CSKCacheEntry) -> CSKCacheConnectorV1Impl:
    """Construct only the adapter fields needed by directive scheduling tests."""

    registry = CSKCacheRegistry()
    registry.put(entry)
    impl = CSKCacheConnectorV1Impl.__new__(CSKCacheConnectorV1Impl)
    impl._config = _Config()
    impl._registry = registry
    impl._catalog = SegmentCatalog.from_entries(registry.entries())
    impl._plans = {}
    impl._pending_boundaries = {}
    impl._directive_boundaries = set()
    impl._probe_states = {}
    impl._allocated_blocks = {}
    impl._probe_accumulators = {}
    impl._block_size = 4
    return impl


def test_explicit_span_directive_caps_then_loads_mid_prompt() -> None:
    """Explicit span path for [prefix][skill][trailing].

    The directive says the current skill is [2, 8). Since the scheduler starts
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
    assert impl.cap_num_new_tokens(request, 0, 32) == 2
    assert impl.get_inprocess_load_tokens(request, 2) == entry.length
    plan = impl._plans[request.request_id]
    assert (plan.cache_id, plan.start, plan.end, plan.source_offset) == (
        entry.cache_id,
        2,
        8,
        0,
    )
    assert plan.token_ids == tuple(entry.token_ids)


def test_suffix_before_trailing_directive_resolves_span() -> None:
    """Suffix-before-trailing path for chat-template tail tokens.

    The prompt ends with two non-skill tokens. CSKCache derives target_end by
    subtracting trailing_token_count, then derives target_start from the cached
    entry length. It still verifies exact token equality before accepting it.
    """

    entry = _make_entry()
    impl = _make_impl(entry)
    request = _Request(
        [1, 2, *entry.token_ids, 90, 91],
        {
            "enabled": True,
            "cache_id": entry.cache_id,
            "placement": "suffix_before_trailing",
            "trailing_token_count": 2,
        },
    )

    matched, _ = impl.get_num_new_matched_tokens(request, 0)
    assert matched == 0
    assert impl.cap_num_new_tokens(request, 0, 32) == 2
    assert impl.get_inprocess_load_tokens(request, 2) == entry.length
    assert impl._plans[request.request_id].start == 2


def test_disabled_directive_does_not_fallback_to_matching() -> None:
    """enabled=false is an explicit opt-out, not an absent directive.

    The prompt contains the cached tokens, so fallback matching would find a
    hit. This test guards the contract that explicit disable suppresses that
    fallback for the request.
    """

    entry = _make_entry()
    impl = _make_impl(entry)
    request = _Request([1, 2, *entry.token_ids], {"enabled": False})

    matched, load_async = impl.get_num_new_matched_tokens(request, 0)
    assert (matched, load_async) == (0, False)
    assert not impl._plans
    assert not impl._pending_boundaries


def test_directive_token_mismatch_fails_closed() -> None:
    """A directive with stale offsets must not silently fall back.

    Wrong token identity at the requested span would scatter unrelated K/V into
    the model cache. The adapter should raise immediately so the caller fixes
    its span computation or tokenizer/template assumptions.
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

    try:
        impl.get_num_new_matched_tokens(request, 0)
    except RuntimeError as exc:
        assert "token mismatch" in str(exc)
    else:
        raise AssertionError("directive token mismatch should fail")


if __name__ == "__main__":
    test_explicit_span_directive_caps_then_loads_mid_prompt()
    test_suffix_before_trailing_directive_resolves_span()
    test_disabled_directive_does_not_fallback_to_matching()
    test_directive_token_mismatch_fails_closed()
    print("CSKCache request directive tests passed")
