from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cskcache.v1.core.cache_engine import CSKCacheEngine
from cskcache.v1.core.config import CSKCacheConfig
from cskcache.v1.core.probe_state import CSKReuseStage
from cskcache.v1.compute.gate import CSKProbeDecision, CSKProbeMetrics
from cskcache.v1.metadata import CSKCacheEntry
from cskcache.v1.storage.storage_manager import StorageManager


@contextmanager
def _raises(error_type: type[Exception], message: str) -> Iterator[None]:
    try:
        yield
    except error_type as exc:
        assert message in str(exc)
    else:
        raise AssertionError(f"Expected {error_type.__name__}: {message}")


def _make_entry(cache_id: str, token_ids: list[int]) -> CSKCacheEntry:
    length = len(token_ids)
    key = torch.zeros(length, 2, 3)
    return CSKCacheEntry(
        cache_id=cache_id,
        source_start=0,
        source_end=length,
        token_ids=list(token_ids),
        kv_by_layer={"layer0": (key, key + 1)},
    )


def _engine(entry: CSKCacheEntry, **cfg) -> CSKCacheEngine:
    return _engine_many([entry], **cfg)


def _engine_many(entries: list[CSKCacheEntry], **cfg) -> CSKCacheEngine:
    storage = StorageManager()
    for entry in entries:
        storage.put(entry)

    # LocalCPUBackend.put(entry) 里面是按 entry.cache_id 存的
    # print(f"STORAGE KEYS: storage.keys()={storage.keys()}")
    # print(f"STORAGE GET: storage.get('{entry.cache_id}')={storage.get(entry.cache_id)}")
    # print(f"CPU BYTES: storage.size_bytes('cpu')={storage.size_bytes('cpu')}")
    return CSKCacheEngine(CSKCacheConfig(**cfg), storage, block_size=16)


def test_engine_importable_without_vllm() -> None:
    """Verify the core engine can be imported without importing vLLM.

    Purpose: the CSKCache core layer should stay vLLM-agnostic. It should only
    own plain scheduling, lookup, storage, load-plan, and probe-state logic. If
    importing the engine pulls in vLLM, the package boundary is too heavy for
    offline unit tests, analysis tools, and non-vLLM integrations.
    """
    # The engine and everything it imports must not drag in vLLM.
    assert "vllm" not in sys.modules, "cache engine import chain pulled in vLLM"


def test_direct_reuse_ready_at_frontier() -> None:
    """Verify direct KV reuse when the cached skill starts at the frontier.

    This is the simplest direct-reuse case: the request has computed zero
    tokens, and the reuse signal says the cached skill span is exactly [0, 4).
    The engine should tell vLLM that the next four tokens can be supplied by
    CSKCache as external KV and should create a load plan for that span.
    """
    skill = [10, 11, 12, 13]
    eng = _engine(_make_entry("skill", skill))
    signal = {
        "cskcache": {
            "cache_id": "skill",
            "target_start": 0,
            "target_end": 4,
        }
    }
    n, load_async = eng.get_num_new_matched_tokens("r1", skill, 0, signal)
    assert n == 4 and load_async is False
    eng.update_reuse_after_alloc("r1", ([0, 1],), num_external_tokens=4)
    requests, probes, saves, _ = eng.build_meta({"r1": 0})
    assert len(requests) == 1 and not probes and not saves
    assert requests[0].plan.cache_id == "skill"
    assert requests[0].plan.start == 0 and requests[0].plan.end == 4
 

def test_reuse_signal_middle_injection() -> None:
    """Verify reuse-signal-based middle-span reuse.

    The prompt layout is [prefix][skill][trailing], so the reusable skill is not
    at the beginning of the request. The signal identifies the current skill
    span as [3, 7). The engine should first let normal prefill run only up to
    the skill start, hold scheduling at the boundary, load the four skill
    tokens in-process, and then allow trailing tokens to continue normally.

    Purpose: vLLM's connector interface is naturally prefix/frontier oriented,
    but CSKCache must support injecting a skill span in the middle of a prompt.
    """
    skill = [10, 11, 12, 13]
    prompt = [1, 2, 3] + skill + [99, 99]  # [prefix][skill][trailing]
    eng = _engine(_make_entry("skill", skill))
    signal = {
        "cskcache": {
            "cache_id": "skill",
            "target_start": 3,
            "target_end": 7,
        }
    }
    # At the frontier 0 the span is ahead -> recorded as a boundary, 0 matched now.
    n, _ = eng.get_num_new_matched_tokens("r1", prompt, 0, signal)
    assert n == 0
    # Prefix chunk is capped to stop exactly at span start (3).
    assert eng.cap_prefill_before_reuse("r1", prompt, 0, 10, signal) == 3
    # At the boundary, the chunk is held (0) so the next pass loads in-process.
    assert eng.cap_prefill_before_reuse("r1", prompt, 3, 10, signal) == 0
    assert eng.get_boundary_reuse_load_tokens("r1", prompt, 3) == (4, True)
    eng.update_reuse_after_alloc("r1", ([0, 1],), num_external_tokens=4)
    requests, probes, saves, _ = eng.build_meta({"r1": 0})
    assert len(requests) == 1
    assert not probes and not saves
    assert requests[0].plan.start == 3 and requests[0].plan.end == 7


def test_multiple_reuse_entries_load_in_prompt_order() -> None:
    """Load two request-local entries with normal prefill between their spans."""

    first = _make_entry("first", [10, 11, 12, 13])
    second = _make_entry("second", [20, 21, 22])
    prompt = [1, 2, *first.token_ids, 90, 91, *second.token_ids, 99]
    eng = _engine_many([first, second])
    signal = {
        "cskcache": {
            "operation": "reuse",
            # Deliberately reversed: the core must order entries by target span.
            "entries": [
                {
                    "cache_id": "second",
                    "target_start": 8,
                    "target_end": 11,
                },
                {
                    "cache_id": "first",
                    "target_start": 2,
                    "target_end": 6,
                },
            ],
        }
    }

    matched, _ = eng.get_num_new_matched_tokens("r1", prompt, 0, signal)
    assert matched == 0
    assert eng.cap_prefill_before_reuse("r1", prompt, 0, 20, signal) == 2
    assert eng.get_boundary_reuse_load_tokens("r1", prompt, 2) == (4, True)
    eng.update_reuse_after_alloc("r1", ([0],), num_external_tokens=4)
    requests, _, _, _ = eng.build_meta({"r1": 0})
    assert [(item.plan.cache_id, item.plan.start, item.plan.end) for item in requests] == [
        ("first", 2, 6)
    ]

    assert eng.cap_prefill_before_reuse("r1", prompt, 6, 20, signal) == 2
    assert eng.get_boundary_reuse_load_tokens("r1", prompt, 8) == (3, True)
    eng.update_reuse_after_alloc("r1", ([0],), num_external_tokens=3)
    requests, _, _, _ = eng.build_meta({"r1": 0})
    assert [(item.plan.cache_id, item.plan.start, item.plan.end) for item in requests] == [
        ("second", 8, 11)
    ]
    assert eng.cap_prefill_before_reuse("r1", prompt, 11, 20, signal) == 20

    eng.on_finished(["r1"])
    assert "r1" not in eng._reuse_spans


def test_multiple_reuse_entries_reject_ambiguous_or_invalid_spans() -> None:
    first = _make_entry("first", [10, 11, 12, 13])
    second = _make_entry("second", [20, 21, 22])
    eng = _engine_many([first, second])
    prompt = [0] * 12

    mixed = {
        "cskcache": {
            "cache_id": "first",
            "target_start": 0,
            "target_end": 4,
            "entries": [],
        }
    }
    with _raises(ValueError, "both cache_id and entries"):
        eng.get_num_new_matched_tokens("mixed", prompt, 0, mixed)

    overlapping = {
        "cskcache": {
            "entries": [
                {"cache_id": "first", "target_start": 2, "target_end": 6},
                {"cache_id": "second", "target_start": 5, "target_end": 8},
            ]
        }
    }
    with _raises(ValueError, "must not overlap"):
        eng.get_num_new_matched_tokens("overlap", prompt, 0, overlapping)

    out_of_bounds = {
        "cskcache": {
            "entries": [
                {"cache_id": "first", "target_start": 9, "target_end": 13}
            ]
        }
    }
    with _raises(RuntimeError, "invalid target_start"):
        eng.get_num_new_matched_tokens("bounds", prompt, 0, out_of_bounds)


def test_reuse_signal_disabled_suppresses_reuse() -> None:
    """Verify an explicit disabled reuse signal suppresses all reuse.

    Even though the request tokens exactly match the cached skill, an explicit
    {"cskcache": {"enabled": False}} signal must return zero matched tokens
    and must not create a load plan.

    Purpose: enabled=False means "do not use CSKCache for this request", not
    "fall back to full-prompt token matching". CSKCache never scans the prompt
    for production reuse.
    """
    skill = [10, 11, 12, 13]
    eng = _engine(_make_entry("skill", skill))
    signal = {"cskcache": {"enabled": False}}
    n, _ = eng.get_num_new_matched_tokens("r1", skill, 0, signal)
    assert n == 0
    assert "r1" not in eng._plans


def test_no_reuse_signal_does_not_scan_prompt() -> None:
    """Verify absent reuse signal means normal prefill, even on token match."""
    skill = [10, 11, 12, 13]
    eng = _engine(_make_entry("skill", skill))
    n, _ = eng.get_num_new_matched_tokens("r1", skill, 0)
    assert n == 0
    assert "r1" not in eng._plans


def test_probe_fsm_pass_then_load() -> None:
    """Verify the probe-gated success path.

    The skill span is [2, 8). As soon as the frontier reaches 2 (the span
    start), the whole span is bulk-preloaded from cache in one shot, without
    advancing the frontier. vLLM then really prefills the probe span [2, 4)
    so the worker can compare recomputed KV against cached KV -- this real
    forward naturally overwrites whatever the bulk preload just scattered at
    [2, 4). When the worker reports that the probe passed, only the frontier
    needs confirming out to the end (8); [4, 8) is already resident from the
    bulk preload and needs no second load.

    Purpose: this covers the quality-protected fast path. CSKCache first
    checks a small real recompute probe; if cached KV looks compatible, the
    rest of the skill never needed a second trip to the cache.
    """
    skill = [10, 11, 12, 13, 14, 15]
    prompt = [1, 2] + skill  # skill starts at frontier-reachable offset 2
    eng = _engine(
        _make_entry("skill", skill),
        probe_enabled=True,
        probe_tokens=2,
        anchor_tokens=4,
    )
    signal = {
        "cskcache": {
            "cache_id": "skill",
            "target_start": 2,
            "target_end": 8,
        }
    }
    # Prefix up to span start (2).
    assert eng.cap_prefill_before_reuse("r1", prompt, 0, 10, signal) == 2
    state = eng._reuse_states["r1"]
    assert state.stage == CSKReuseStage.LOADING
    # Frontier is at the span start but the bulk preload has not dispatched
    # yet, so real prefill stays capped at 0 until it does.
    assert eng.cap_prefill_before_reuse("r1", prompt, 2, 10, signal) == 0

    # Bulk preload: the whole [2,8) span, scattered without advancing frontier.
    tokens, advance = eng.get_boundary_reuse_load_tokens("r1", prompt, 2)
    assert (tokens, advance) == (6, False)
    assert state.stage == CSKReuseStage.PROBING
    eng.update_reuse_after_alloc("r1", ([0],), num_external_tokens=6)
    requests, _, _, _ = eng.build_meta({"r1": 0})
    assert len(requests) == 1
    assert requests[0].plan.start == 2 and requests[0].plan.end == 8
    assert requests[0].plan.requires_scatter is True
    assert state.stage == CSKReuseStage.PROBING  # bulk preload alone isn't "done"

    # Now vLLM really prefills the probe prefix [2,4).
    assert eng.cap_prefill_before_reuse("r1", prompt, 2, 10, signal) == 2
    assert state.pending_capture is True
    # The probe span is normal prefill, so vLLM allocates blocks for it
    # (num_external_tokens=0 since nothing new is loaded here).
    eng.update_reuse_after_alloc("r1", ([0],), num_external_tokens=0)
    _, probes, saves, _ = eng.build_meta({"r1": 2})
    assert len(probes) == 1 and probes[0].start == 2 and probes[0].end == 4
    assert not saves
    assert state.stage == CSKReuseStage.GATING
    # Worker says the cached KV matches -> nothing left to recompute.
    metrics = CSKProbeMetrics(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "max", ())
    passed = CSKProbeDecision("r1", "skill", True, 0.15, metrics)
    eng.on_worker_decisions([passed])
    assert state.stage == CSKReuseStage.READY
    # Confirm-only: [4,8) was already scattered by the bulk preload.
    tokens, advance = eng.get_boundary_reuse_load_tokens("r1", prompt, 4)
    assert (tokens, advance) == (4, True)
    eng.update_reuse_after_alloc("r1", ([0],), num_external_tokens=4)
    requests, _, saves, _ = eng.build_meta({"r1": 0})
    assert requests[0].plan.start == 4 and requests[0].plan.source_offset == 2
    assert requests[0].plan.requires_scatter is False
    assert not saves
    assert state.stage == CSKReuseStage.DONE


def test_probe_fsm_fail_needs_recompute() -> None:
    """Verify the probe-gated fallback path.

    This follows the same setup as the passing-probe test, but the worker
    reports that the probe failed. The engine must not trust the bulk-preloaded
    prefix. Instead, it should enter RECOMPUTING and require additional real
    prefill up to anchor_end (2 + 4 = 6); because the probe already reached 4,
    the next capped chunk should be two tokens. That real prefill overwrites
    the bulk-preloaded [4, 6); [6, 8) still needs no second load.

    Purpose: this covers the safety path for cases where cached KV appears too
    different from real recompute KV. The engine responds by recomputing more
    of the skill (bounded to anchor_tokens) before trusting the rest.
    """
    skill = [10, 11, 12, 13, 14, 15]
    prompt = [1, 2] + skill
    eng = _engine(
        _make_entry("skill", skill),
        probe_enabled=True,
        probe_tokens=2,
        anchor_tokens=4,
    )
    signal = {
        "cskcache": {
            "cache_id": "skill",
            "target_start": 2,
            "target_end": 8,
        }
    }
    eng.cap_prefill_before_reuse("r1", prompt, 0, 10, signal)
    eng.cap_prefill_before_reuse("r1", prompt, 2, 10, signal)  # still LOADING, 0
    eng.get_boundary_reuse_load_tokens("r1", prompt, 2)  # dispatch bulk preload
    eng.update_reuse_after_alloc("r1", ([0],), num_external_tokens=6)
    eng.build_meta({"r1": 0})
    eng.cap_prefill_before_reuse("r1", prompt, 2, 10, signal)  # probe prefix
    eng.update_reuse_after_alloc("r1", ([0],), num_external_tokens=0)
    eng.build_meta({"r1": 2})
    metrics = CSKProbeMetrics(1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, "max", ())
    failed = CSKProbeDecision("r1", "skill", False, 0.15, metrics)
    eng.on_worker_decisions([failed])
    assert eng._reuse_states["r1"].stage == CSKReuseStage.RECOMPUTING
    # Recompute up to anchor_end (2 + 4 = 6).
    assert eng.cap_prefill_before_reuse("r1", prompt, 4, 10, signal) == 2


def test_probe_fsm_advances_to_second_reuse_entry() -> None:
    first = _make_entry("first", [10, 11, 12, 13])
    second = _make_entry("second", [20, 21, 22, 23])
    prompt = [1, *first.token_ids, 90, *second.token_ids]
    eng = _engine_many(
        [first, second],
        probe_enabled=True,
        probe_tokens=1,
        anchor_tokens=2,
    )
    signal = {
        "cskcache": {
            "entries": [
                {"cache_id": "first", "target_start": 1, "target_end": 5},
                {"cache_id": "second", "target_start": 6, "target_end": 10},
            ]
        }
    }

    assert eng.cap_prefill_before_reuse("r1", prompt, 0, 20, signal) == 1
    assert eng.cap_prefill_before_reuse("r1", prompt, 1, 20, signal) == 0  # LOADING
    tokens, advance = eng.get_boundary_reuse_load_tokens("r1", prompt, 1)
    assert (tokens, advance) == (4, False)
    eng.update_reuse_after_alloc("r1", ([0],), num_external_tokens=4)
    eng.build_meta({"r1": 0})

    assert eng.cap_prefill_before_reuse("r1", prompt, 1, 20, signal) == 1
    eng.update_reuse_after_alloc("r1", ([0],), num_external_tokens=0)
    _, probes, _, _ = eng.build_meta({"r1": 1})
    assert [(probe.cache_id, probe.start, probe.end) for probe in probes] == [
        ("first", 1, 2)
    ]
    metrics = CSKProbeMetrics(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "max", ())
    eng.on_worker_decisions(
        [CSKProbeDecision("r1", "first", True, 0.15, metrics)]
    )
    assert eng.get_boundary_reuse_load_tokens("r1", prompt, 2) == (3, True)
    eng.update_reuse_after_alloc("r1", ([0],), num_external_tokens=3)
    requests, _, _, _ = eng.build_meta({"r1": 0})
    assert requests[0].plan.cache_id == "first"

    assert eng.cap_prefill_before_reuse("r1", prompt, 5, 20, signal) == 1
    assert eng.cap_prefill_before_reuse("r1", prompt, 6, 20, signal) == 0  # LOADING
    assert eng._reuse_states["r1"].cache_id == "second"


if __name__ == "__main__":

    print("RUNNING ENGINE TESTS")
    print("---------------------- test 1: test engine importable without vLLM ------------------------------")
    test_engine_importable_without_vllm()
    print("PASS test_engine_importable_without_vllm")

    print("---------------------- test 2: test direct reuse ready at frontier ------------------------------")
    test_direct_reuse_ready_at_frontier()
    print("PASS test_direct_reuse_ready_at_frontier")
    
    print("---------------------- test 3: test reuse signal middle injection ------------------------------")
    test_reuse_signal_middle_injection()
    print("PASS test_reuse_signal_middle_injection")
    
    print("---------------------- test 4: test reuse signal disabled suppresses reuse ------------------------------")
    test_reuse_signal_disabled_suppresses_reuse()
    print("PASS test_reuse_signal_disabled_suppresses_reuse")
    
    print("---------------------- test 5: test probe FSM pass then load ------------------------------")
    test_probe_fsm_pass_then_load()
    print("PASS test_probe_fsm_pass_then_load")
    
    print("---------------------- test 6: test probe FSM fail needs anchor ------------------------------")
    test_probe_fsm_fail_needs_anchor()
    print("PASS test_probe_fsm_fail_needs_anchor")

    print("ALL ENGINE TESTS PASSED")
