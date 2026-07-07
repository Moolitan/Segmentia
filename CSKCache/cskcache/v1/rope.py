from __future__ import annotations

import torch


def find_rotary_embedding(model) -> object | None:
    for module in model.modules():
        if module.__class__.__name__.endswith("RotaryEmbedding"):
            return module
        if hasattr(module, "cos_sin_cache") or hasattr(module, "cos_cached"):
            return module
    return None


def rerotate_k_for_target_positions(
    key: torch.Tensor,
    *,
    source_start: int,
    target_start: int,
    rope: object | None,
) -> torch.Tensor:
    """Placeholder for CSK key RoPE correction.

    The old experiment path already has a concrete vLLM-specific implementation
    in `/home/wsh/vllm/vllm/v1/context_segment_cache/rope.py`.  CSKCache keeps
    this function as the stable package boundary; wiring the exact rotary
    implementation is the next parity step.
    """

    if source_start == target_start:
        return key
    raise NotImplementedError(
        "CSKCache ROPE_REUSE with different source/target positions needs the "
        "rotary correction ported from the old vLLM context_segment_cache rope.py"
    )

