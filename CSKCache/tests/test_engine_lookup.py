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
    return CSKCacheEngine(CSKCacheConfig(**cfg), storage, block_size=16)


def test_engine_importable_without_vllm() -> None:
    # The engine and everything it imports must not drag in vLLM.
    assert "vllm" not in sys.modules, "cache engine import chain pulled in vLLM"


def test_direct_reuse_ready_at_frontier() -> None:
    skill = [10, 11, 12, 13]
    eng = _engine(_make_entry("skill", skill))
    n, load_async = eng.get_num_new_matched_tokens("r1", skill, 0)
    assert n == 4 and load_async is False
    eng.update_state_after_alloc("r1", ([0, 1],), num_external_tokens=4)
    requests, probes = eng.build_meta({"r1": 0})
    assert len(requests) == 1 and not probes
    assert requests[0].plan.cache_id == "skill"
    assert requests[0].plan.start == 0 and requests[0].plan.end == 4


def test_directive_middle_injection() -> None:
    skill = [10, 11, 12, 13]
    prompt = [1, 2, 3] + skill + [99, 99]  # [prefix][skill][trailing]
    eng = _engine(_make_entry("skill", skill))
    directive = {
        "cskcache": {
            "cache_id": "skill",
            "placement": "explicit_span",
            "target_start": 3,
            "target_end": 7,
        }
    }
    # At the frontier 0 the span is ahead -> recorded as a boundary, 0 matched now.
    n, _ = eng.get_num_new_matched_tokens("r1", prompt, 0, directive)
    assert n == 0
    # Prefix chunk is capped to stop exactly at span start (3).
    assert eng.cap_num_new_tokens("r1", prompt, 0, 10, directive) == 3
    # At the boundary, the chunk is held (0) so the next pass loads in-process.
    assert eng.cap_num_new_tokens("r1", prompt, 3, 10, directive) == 0
    assert eng.get_inprocess_load_tokens("r1", prompt, 3) == 4
    eng.update_state_after_alloc("r1", ([0, 1],), num_external_tokens=4)
    requests, probes = eng.build_meta({"r1": 0})
    assert len(requests) == 1
    assert requests[0].plan.start == 3 and requests[0].plan.end == 7


def test_directive_disabled_suppresses_reuse() -> None:
    skill = [10, 11, 12, 13]
    eng = _engine(_make_entry("skill", skill))
    directive = {"cskcache": {"enabled": False}}
    n, _ = eng.get_num_new_matched_tokens("r1", skill, 0, directive)
    assert n == 0
    assert "r1" not in eng._plans


def test_probe_fsm_pass_then_load() -> None:
    skill = [10, 11, 12, 13, 14, 15]
    prompt = [1, 2] + skill  # skill starts at frontier-reachable offset 2
    eng = _engine(
        _make_entry("skill", skill),
        probe_enabled=True,
        probe_tokens=2,
        anchor_tokens=4,
    )
    directive = {
        "cskcache": {
            "cache_id": "skill",
            "placement": "explicit_span",
            "target_start": 2,
            "target_end": 8,
        }
    }
    # Prefix up to span start (2).
    assert eng.cap_num_new_tokens("r1", prompt, 0, 10, directive) == 2
    # Probe prefix: only probe_tokens (2) allowed [2,4).
    assert eng.cap_num_new_tokens("r1", prompt, 2, 10, directive) == 2
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
    skill = [10, 11, 12, 13, 14, 15]
    prompt = [1, 2] + skill
    eng = _engine(
        _make_entry("skill", skill),
        probe_enabled=True,
        probe_tokens=2,
        anchor_tokens=4,
    )
    directive = {
        "cskcache": {
            "cache_id": "skill",
            "placement": "explicit_span",
            "target_start": 2,
            "target_end": 8,
        }
    }
    eng.cap_num_new_tokens("r1", prompt, 0, 10, directive)
    eng.cap_num_new_tokens("r1", prompt, 2, 10, directive)
    eng.update_state_after_alloc("r1", ([0],), num_external_tokens=0)
    eng.build_meta({"r1": 2})
    metrics = CSKProbeMetrics(1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, "max", ())
    failed = CSKProbeDecision("r1", "skill", False, 0.15, metrics)
    eng.on_worker_decisions([failed])
    assert eng._probe_states["r1"].phase == CSKProbePhase.NEED_ANCHOR
    # Anchor recompute up to anchor_end (2 + 4 = 6).
    assert eng.cap_num_new_tokens("r1", prompt, 4, 10, directive) == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL ENGINE TESTS PASSED")
