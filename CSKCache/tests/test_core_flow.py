from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cskcache.v1.compute.gate import CSKProbeAccumulator
from cskcache.v1.compute.reuse import prepare_reuse_slice
from cskcache.v1.matcher import SegmentCatalog, find_best_reuse
from cskcache.v1.metadata import CSKCacheEntry
from cskcache.v1.registry import CSKCacheRegistry
from cskcache.v1.slot_ops import gather_span, scatter_span


def _make_entry() -> CSKCacheEntry:
    length = 6
    num_kv_heads = 2
    head_dim = 3
    key = torch.arange(length * num_kv_heads * head_dim, dtype=torch.float32)
    key = key.reshape(length, num_kv_heads, head_dim)
    value = key + 1000
    return CSKCacheEntry(
        cache_id="skill-demo",
        source_start=0,
        source_end=length,
        token_ids=[10, 11, 12, 13, 14, 15],
        kv_by_layer={"layer.0": (key, value)},
    )


def test_match_slice_scatter_and_gather_roundtrip() -> None:
    entry = _make_entry()
    registry = CSKCacheRegistry()
    registry.put(entry)

    prompt_tokens = [1, 2, *entry.token_ids, 99]
    catalog = SegmentCatalog.from_entries(registry.entries())

    upcoming = find_best_reuse(catalog, prompt_tokens, num_computed_tokens=0)
    assert upcoming is not None
    assert (upcoming.cache_id, upcoming.start, upcoming.end) == ("skill-demo", 2, 8)

    ready = find_best_reuse(catalog, prompt_tokens, num_computed_tokens=2)
    assert ready is not None
    assert ready.start == 2

    key_slice, value_slice = prepare_reuse_slice(
        entry,
        layer_name="layer.0",
        source_offset=2,
        length=3,
        target_start=2,
        rope=None,
        device="cpu",
    )
    assert torch.equal(key_slice, entry.kv_by_layer["layer.0"][0][2:5])
    assert torch.equal(value_slice, entry.kv_by_layer["layer.0"][1][2:5])

    block_size = 4
    kv_cache = torch.zeros(2, 3, block_size, 2, 3)
    block_ids = [2, 0, 1]
    scatter_span(
        kv_cache,
        block_ids,
        start=2,
        end=5,
        block_size=block_size,
        key=key_slice,
        value=value_slice,
    )
    gathered_key, gathered_value = gather_span(
        kv_cache,
        block_ids,
        start=2,
        end=5,
        block_size=block_size,
    )
    assert torch.equal(gathered_key, key_slice)
    assert torch.equal(gathered_value, value_slice)


def test_probe_gate_pass_and_fail_metrics() -> None:
    reuse = torch.ones(4, 2, 3)

    passing = CSKProbeAccumulator(
        req_id="req-pass",
        cache_id="skill-demo",
        tau=0.1,
        gate_metric="max",
    )
    passing.add_layer(
        "layer.0",
        reuse_key=reuse,
        reuse_value=reuse,
        recompute_key=reuse.clone(),
        recompute_value=reuse.clone(),
    )
    pass_decision = passing.decide()
    assert pass_decision.passed
    assert pass_decision.metrics.num_layers == 1
    assert pass_decision.metrics.gate_value < 1e-5

    failing = CSKProbeAccumulator(
        req_id="req-fail",
        cache_id="skill-demo",
        tau=0.1,
        gate_metric="max",
    )
    failing.add_layer(
        "layer.0",
        reuse_key=reuse,
        reuse_value=reuse,
        recompute_key=-reuse,
        recompute_value=-reuse,
    )
    fail_decision = failing.decide()
    assert not fail_decision.passed
    assert fail_decision.metrics.k_max_layer > 1.9
    assert fail_decision.metrics.v_max_layer > 1.9


if __name__ == "__main__":
    test_match_slice_scatter_and_gather_roundtrip()
    test_probe_gate_pass_and_fail_metrics()
    print("CSKCache core-flow tests passed")
