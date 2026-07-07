from __future__ import annotations

import torch


def split_kv_cache(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a packed K/V tensor into two separate tensors.

    Many implementations store Key and Value in a single tensor, using a
    size-2 dimension to tell them apart. That "2-way" dimension may live on
    axis 0 or axis 1, so we check both.
    """
    if kv_cache.shape[0] == 2:
        # K/V dimension is on axis 0; unbind along it into (key, value)
        return kv_cache.unbind(0)
    if kv_cache.shape[1] == 2:
        # K/V dimension is on axis 1
        return kv_cache.unbind(1)
    # Neither axis is 2, so the K/V layout can't be recognized -> error out
    raise ValueError(
        "CSKCache expects KV cache with a 2-way K/V dimension at axis 0 or 1; "
        f"got shape={tuple(kv_cache.shape)}"
    )


def flatten_cache(cache: torch.Tensor) -> torch.Tensor:
    """Merge the two leading dims so every token gets a globally unique "slot" index.

    Input  layout: [num_blocks, block_size, num_kv_heads, head_dim]
    Output layout: [num_blocks * block_size, num_kv_heads, head_dim]

    After merging, any token can be addressed with a single 1-D index, with no
    need to reason about "which block, which offset within the block."
    """
    if cache.dim() != 4:
        # Only the 4-D paged layout is supported; bail out otherwise
        raise ValueError(
            "CSKCache currently supports KV layout "
            "[num_blocks, block_size, num_kv_heads, head_dim]; "
            f"got shape={tuple(cache.shape)}"
        )
    # Collapse dims 0 and 1 (num_blocks and block_size)
    return cache.flatten(0, 1)


def slots_for_span(
    block_ids: list[int],
    start: int,
    end: int,
    block_size: int,
) -> torch.Tensor:
    """Core address translation: map the logical position range [start, end)
    to physical slot indices in the flattened cache.

    block_ids[i] is the physical block number backing "the i-th logical block"
    of this sequence (physical block numbers are usually non-contiguous).
    """
    slots = []
    for pos in range(start, end):
        logical_block = pos // block_size   # which logical block this position falls in
        block_offset = pos % block_size     # offset within that block
        # look up the physical block number -> multiply by block_size for the
        # block's starting slot -> add the in-block offset
        slots.append(block_ids[logical_block] * block_size + block_offset)
    # These indices line up exactly with the row numbers after flatten_cache
    return torch.tensor(slots, dtype=torch.long)


def gather_span(
    kv_cache: torch.Tensor,
    block_ids: list[int],
    start: int,
    end: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read op: fetch the Key and Value of every token in the range [start, end)."""
    key_cache, value_cache = split_kv_cache(kv_cache)   # split K/V
    key_flat = flatten_cache(key_cache)                 # flatten each
    value_flat = flatten_cache(value_cache)
    # compute physical slots and move them onto the cache's device
    slots = slots_for_span(block_ids, start, end, block_size).to(key_flat.device)
    # pick out the corresponding rows along axis 0
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
    """Write op (inverse of gather_span): copy the given key/value back into their slots, in place."""
    key_cache, value_cache = split_kv_cache(kv_cache)   # split K/V
    key_flat = flatten_cache(key_cache)                 # flatten each
    value_flat = flatten_cache(value_cache)
    # compute physical slots (also moved onto the cache's device)
    slots = slots_for_span(block_ids, start, end, block_size).to(key_flat.device)
    # index_copy_ writes in place; .to(dtype) keeps types consistent with the cache.
    # The trailing-underscore methods mutate in place, so this function returns None.
    key_flat.index_copy_(0, slots, key.to(key_flat.dtype))
    value_flat.index_copy_(0, slots, value.to(value_flat.dtype))