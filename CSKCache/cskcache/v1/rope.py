from __future__ import annotations

from typing import Any

import torch


def find_rotary_embedding(model) -> object | None:
    for module in model.modules():
        if module.__class__.__name__.endswith("RotaryEmbedding"):
            return module
        if hasattr(module, "cos_sin_cache") or hasattr(module, "cos_cached"):
            return module
    return None


def _get_rope_attr(rope: object, name: str) -> Any:
    if not hasattr(rope, name):
        raise TypeError(f"CSKCache rotary embedding is missing required attr {name!r}")
    return getattr(rope, name)


def _load_apply_rotary_emb():
    try:
        from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
    except ImportError as exc:
        raise RuntimeError(
            "CSKCache RoPE correction requires vLLM rotary embedding helpers"
        ) from exc
    return ApplyRotaryEmb


def rerotate_k_for_target_positions(
    key: torch.Tensor,
    *,
    source_start: int,
    target_start: int,
    rope: object | None,
) -> torch.Tensor:
    """Move cached RoPE-applied keys from source positions to target positions.

    A cached key for token i has already been rotated at
    `source_start + i`. Reusing it at `target_start + i` requires one extra
    relative rotation by `target_start - source_start`. Values are not rotated.
    """

    if source_start == target_start:
        return key
    if rope is None:
        raise ValueError(
            "CSKCache RoPE correction requires a rotary embedding instance when "
            "source and target positions differ"
        )

    rotary_dim = int(_get_rope_attr(rope, "rotary_dim"))
    head_dim = key.shape[-1]
    if rotary_dim > head_dim:
        raise ValueError(
            f"CSKCache rotary_dim={rotary_dim} exceeds key head_dim={head_dim}"
        )

    match_cache = _get_rope_attr(rope, "_match_cos_sin_cache_dtype")
    if not callable(match_cache):
        raise TypeError(
            "CSKCache rotary embedding attr '_match_cos_sin_cache_dtype' is not callable"
        )

    delta = target_start - source_start
    abs_delta = abs(delta)
    cos_sin_cache = match_cache(key)
    if abs_delta >= cos_sin_cache.shape[0]:
        raise ValueError(
            "CSKCache RoPE correction offset exceeds rotary cache length: "
            f"offset={abs_delta}, cache_len={cos_sin_cache.shape[0]}"
        )

    positions = torch.full(
        (key.shape[0],), abs_delta, dtype=torch.long, device=key.device
    )
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    if delta < 0:
        sin = -sin

    apply_rotary_emb = _load_apply_rotary_emb()
    key_rot = key[..., :rotary_dim]
    key_pass = key[..., rotary_dim:]
    key_rot = apply_rotary_emb.forward_static(
        key_rot, cos, sin, bool(_get_rope_attr(rope, "is_neox_style"))
    )
    if key_pass.numel() == 0:
        return key_rot
    return torch.cat((key_rot, key_pass), dim=-1)
