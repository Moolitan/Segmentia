from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cskcache.v1.core.cache_engine import CSKCacheEngine
from cskcache.v1.core.config import CSKCacheConfig
from cskcache.v1.core.probe_state import CSKProbePhase
from cskcache.v1.compute.gate import CSKProbeDecision, CSKProbeMetrics
from cskcache.v1.metadata import CSKCacheEntry
from cskcache.v1.storage.storage_manager import StorageManager


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
    storage = StorageManager()
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
    print(f"n={n}, load_async={load_async}")
    assert n == 4 and load_async is False
    eng.update_state_after_alloc("r1", ([0, 1],), num_external_tokens=4)
    requests, probes = eng.build_meta({"r1": 0})
    print(f"requests={requests}")
    print(f"probes={probes}")
    assert len(requests) == 1 and not probes
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
    n, _ = eng.get_num_new_matched_tokens("r1", prompt, 2, signal)
    print(f"n={n}")
    assert n == 0
    # Prefix chunk is capped to stop exactly at span start (3).
    assert eng.cap_num_new_tokens("r1", prompt, 0, 10, signal) == 3
    # At the boundary, the chunk is held (0) so the next pass loads in-process.
    assert eng.cap_num_new_tokens("r1", prompt, 3, 10, signal) == 0
    assert eng.get_inprocess_load_tokens("r1", prompt, 3) == 4
    eng.update_state_after_alloc("r1", ([0, 1],), num_external_tokens=4)
    requests, probes = eng.build_meta({"r1": 0})
    assert len(requests) == 1
    assert requests[0].plan.start == 3 and requests[0].plan.end == 7


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

    The skill span is [2, 8), with probe_tokens=2. The engine should prefill the
    prefix [0, 2), then prefill the probe span [2, 4) normally so the worker can
    compare recomputed KV against cached KV. When the worker reports that the
    probe passed, the state machine should load only the remaining tail [4, 8).

    Purpose: this covers the quality-protected fast path. CSKCache first checks
    a small real recompute probe; if cached KV looks compatible, it skips the
    rest of the skill by loading the tail from offline KV.
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
    assert eng.cap_num_new_tokens("r1", prompt, 0, 10, signal) == 2
    # Probe prefix: only probe_tokens (2) allowed [2,4).
    assert eng.cap_num_new_tokens("r1", prompt, 2, 10, signal) == 2
    state = eng._probe_states["r1"]
    assert state.pending_capture == "probe"
    # The probe span is normal prefill, so vLLM allocates blocks for it
    # (num_external_tokens=0 since nothing is loaded yet).
    eng.update_state_after_alloc("r1", ([0],), num_external_tokens=0)
    # Scheduling the probe chunk emits a probe capture and moves to WAIT_PROBE.
    _, probes = eng.build_meta({"r1": 2})
    assert len(probes) == 1 and probes[0].start == 2 and probes[0].end == 4
    assert state.phase == CSKProbePhase.WAIT_PROBE
    # Worker says the cached KV matches -> load the tail from probe_end (4).
    metrics = CSKProbeMetrics(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "max", ())
    passed = CSKProbeDecision("r1", "skill", True, 0.15, metrics)
    eng.on_worker_decisions([passed])
    assert state.phase == CSKProbePhase.NEED_LOAD
    assert eng.get_inprocess_load_tokens("r1", prompt, 4) == 4  # [4,8)
    eng.update_state_after_alloc("r1", ([0],), num_external_tokens=4)
    requests, _ = eng.build_meta({"r1": 0})
    assert requests[0].plan.start == 4 and requests[0].plan.source_offset == 2


def test_probe_fsm_fail_needs_anchor() -> None:
    """Verify the probe-gated fallback path.

    This follows the same setup as the passing-probe test, but the worker
    reports that the probe failed. The engine must not blindly load the tail.
    Instead, it should enter NEED_ANCHOR and require additional normal prefill
    up to anchor_end. With start=2 and anchor_tokens=4, anchor_end is 6; because
    the probe already reached 4, the next capped chunk should be two tokens.

    Purpose: this covers the safety path for cases where cached KV appears too
    different from real recompute KV. The engine responds by recomputing more of
    the skill before considering reuse.
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
    eng.cap_num_new_tokens("r1", prompt, 0, 10, signal)
    eng.cap_num_new_tokens("r1", prompt, 2, 10, signal)
    eng.update_state_after_alloc("r1", ([0],), num_external_tokens=0)
    eng.build_meta({"r1": 2})
    metrics = CSKProbeMetrics(1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, "max", ())
    failed = CSKProbeDecision("r1", "skill", False, 0.15, metrics)
    eng.on_worker_decisions([failed])
    assert eng._probe_states["r1"].phase == CSKProbePhase.NEED_ANCHOR
    # Anchor recompute up to anchor_end (2 + 4 = 6).
    assert eng.cap_num_new_tokens("r1", prompt, 4, 10, signal) == 2


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
    
    # print("---------------------- test 4: test reuse signal disabled suppresses reuse ------------------------------")
    # test_reuse_signal_disabled_suppresses_reuse()
    # print("PASS test_reuse_signal_disabled_suppresses_reuse")
    
    # print("---------------------- test 5: test probe FSM pass then load ------------------------------")
    # test_probe_fsm_pass_then_load()
    # print("PASS test_probe_fsm_pass_then_load")
    
    # print("---------------------- test 6: test probe FSM fail needs anchor ------------------------------")
    # test_probe_fsm_fail_needs_anchor()
    # print("PASS test_probe_fsm_fail_needs_anchor")

    # print("ALL ENGINE TESTS PASSED")
