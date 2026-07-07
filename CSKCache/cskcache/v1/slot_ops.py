from __future__ import annotations

import torch


def split_kv_cache(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if kv_cache.shape[0] == 2:
        return kv_cache.unbind(0)
    if kv_cache.shape[1] == 2:
        return kv_cache.unbind(1)
    raise ValueError(
        "CSKCache expects KV cache with a 2-way K/V dimension at axis 0 or 1; "
        f"got shape={tuple(kv_cache.shape)}"
    )


def flatten_cache(cache: torch.Tensor) -> torch.Tensor:
    if cache.dim() != 4:
        raise ValueError(
            "CSKCache currently supports KV layout "
            "[num_blocks, block_size, num_kv_heads, head_dim]; "
            f"got shape={tuple(cache.shape)}"
        )
    return cache.flatten(0, 1)


def slots_for_span(
    block_ids: list[int],
    start: int,
    end: int,
    block_size: int,
) -> torch.Tensor:
    slots = []
    for pos in range(start, end):
        logical_block = pos // block_size
        block_offset = pos % block_size
        slots.append(block_ids[logical_block] * block_size + block_offset)
    return torch.tensor(slots, dtype=torch.long)


def gather_span(
    kv_cache: torch.Tensor,
    block_ids: list[int],
    start: int,
    end: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_cache, value_cache = split_kv_cache(kv_cache)
    key_flat = flatten_cache(key_cache)
    value_flat = flatten_cache(value_cache)
    slots = slots_for_span(block_ids, start, end, block_size).to(key_flat.device)
    return key_flat.index_select(0, slots), value_flat.index_select(0, slots)


def scatter_span(
    kv_cache: torch.Tensor,
    block_ids: list[int],
    start: int,
    end: int,
    block_size: int,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    key_cache, value_cache = split_kv_cache(kv_cache)
    key_flat = flatten_cache(key_cache)
    value_flat = flatten_cache(value_cache)
    slots = slots_for_span(block_ids, start, end, block_size).to(key_flat.device)
    key_flat.index_copy_(0, slots, key.to(key_flat.dtype))
    value_flat.index_copy_(0, slots, value.to(value_flat.dtype))

