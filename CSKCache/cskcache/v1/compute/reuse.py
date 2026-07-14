from __future__ import annotations

import torch

from cskcache.profiling import LoadTrace, NullLoadTrace
from cskcache.v1.metadata import CSKCacheEntry
from cskcache.v1.rope import rerotate_k_for_target_positions


def prepare_reuse_slice(
    entry: CSKCacheEntry,
    *,
    layer_name: str,
    source_offset: int,
    length: int,
    target_start: int,
    rope: object | None,
    device: torch.device | str,
    trace: LoadTrace | NullLoadTrace | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source_offset < 0:
        raise ValueError(f"CSK source_offset must be non-negative: {source_offset}")
    if length < 0:
        raise ValueError(f"CSK reuse length must be non-negative: {length}")
    source_end = source_offset + length
    if source_end > entry.length:
        raise ValueError(
            "CSK reuse slice exceeds entry length: "
            f"source_end={source_end}, entry_length={entry.length}"
        )

    source_key, source_value = entry.kv_by_layer[layer_name]
    stage_trace = trace if trace is not None else NullLoadTrace()
    with stage_trace.cuda_stage("key_h2d", device):
        key = source_key[source_offset:source_end].to(device)
    with stage_trace.cuda_stage("value_h2d", device):
        value = source_value[source_offset:source_end].to(device)
    source_start = entry.source_start + source_offset
    if source_start != target_start:
        with stage_trace.cuda_stage("rope", device):
            key = rerotate_k_for_target_positions(
                key,
                source_start=source_start,
                target_start=target_start,
                rope=rope,
            )
    return key, value
