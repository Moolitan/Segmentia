#!/usr/bin/env python3
"""Pure-Torch reference for materialized and shared Segmentia attention."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class AttentionResult:
    output: torch.Tensor
    lse: torch.Tensor


def _validate_heads(query: torch.Tensor, key: torch.Tensor) -> int:
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("query and key must have shape [tokens, heads, dim]")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions differ")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads")
    return query.shape[1] // key.shape[1]


def expand_kv_heads(tensor: torch.Tensor, query_heads: int) -> torch.Tensor:
    """Expand GQA KV heads to query heads without changing token order."""
    if tensor.ndim != 3:
        raise ValueError("KV tensor must have shape [tokens, kv_heads, dim]")
    kv_heads = tensor.shape[1]
    if query_heads % kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")
    return tensor.repeat_interleave(query_heads // kv_heads, dim=1)


def rope_delta(
    tensor: torch.Tensor,
    delta: int,
    *,
    rotary_dim: int | None = None,
    theta: float = 1_000_000.0,
    neox_style: bool = True,
) -> torch.Tensor:
    """Apply a constant RoPE position delta to post-RoPE Q or K.

    Qwen3 uses NeoX-style pairing. Supporting GPT-J pairing here makes the
    reference explicit and lets tests catch layout assumptions.
    """
    if tensor.ndim != 3:
        raise ValueError("RoPE input must have shape [tokens, heads, dim]")
    head_dim = tensor.shape[-1]
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive, even, and <= head_dim")

    work = tensor.to(torch.float32)
    rotated = work[..., :rotary_dim]
    passed = work[..., rotary_dim:]
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=work.device)
            / rotary_dim
        )
    )
    angle = float(delta) * inv_freq
    cos = angle.cos().view(1, 1, -1)
    sin = angle.sin().view(1, 1, -1)
    if neox_style:
        first, second = rotated.chunk(2, dim=-1)
        rotated = torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )
    else:
        first, second = rotated[..., ::2], rotated[..., 1::2]
        rotated = torch.stack(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        ).flatten(-2)
    return torch.cat((rotated, passed), dim=-1).to(tensor.dtype)


def _attention_from_logits(
    logits: torch.Tensor, value: torch.Tensor
) -> AttentionResult:
    if logits.shape[-1] == 0:
        raise ValueError("attention segment cannot be empty")
    logits_fp32 = logits.to(torch.float32)
    lse = torch.logsumexp(logits_fp32, dim=-1)
    probs = torch.softmax(logits_fp32, dim=-1)
    output = torch.einsum("qhk,khd->qhd", probs, value.to(torch.float32))
    return AttentionResult(output=output, lse=lse)


def merge_attention_states(
    first: AttentionResult, second: AttentionResult
) -> AttentionResult:
    if first.output.shape != second.output.shape or first.lse.shape != second.lse.shape:
        raise ValueError("partial attention state shapes differ")
    merged_lse = torch.logaddexp(first.lse, second.lse)
    first_weight = torch.exp(first.lse - merged_lse).unsqueeze(-1)
    second_weight = torch.exp(second.lse - merged_lse).unsqueeze(-1)
    output = first.output * first_weight + second.output * second_weight
    return AttentionResult(output=output, lse=merged_lse)


def materialized_attention(
    query: torch.Tensor,
    source_key: torch.Tensor,
    shared_value: torch.Tensor,
    offset: torch.Tensor,
    position_delta: int,
    *,
    private_key: torch.Tensor | None = None,
    private_value: torch.Tensor | None = None,
    rotary_dim: int | None = None,
    theta: float = 1_000_000.0,
) -> AttentionResult:
    """Current path: rotate and correct every shared K, then attend once."""
    _validate_heads(query, source_key)
    if offset.shape != source_key.shape[1:]:
        raise ValueError("offset must have shape [kv_heads, head_dim]")
    corrected_key = rope_delta(
        source_key,
        position_delta,
        rotary_dim=rotary_dim,
        theta=theta,
    ) + offset.to(source_key.dtype).unsqueeze(0)
    keys = corrected_key
    values = shared_value
    if private_key is not None or private_value is not None:
        if private_key is None or private_value is None:
            raise ValueError("private key and value must be supplied together")
        keys = torch.cat((private_key, corrected_key), dim=0)
        values = torch.cat((private_value, shared_value), dim=0)
    expanded_key = expand_kv_heads(keys, query.shape[1])
    expanded_value = expand_kv_heads(values, query.shape[1])
    scale = 1.0 / math.sqrt(query.shape[-1])
    logits = torch.einsum(
        "qhd,khd->qhk", query.to(torch.float32), expanded_key.to(torch.float32)
    ) * scale
    return _attention_from_logits(logits, expanded_value)


def shared_attention(
    query: torch.Tensor,
    source_key: torch.Tensor,
    shared_value: torch.Tensor,
    offset: torch.Tensor,
    position_delta: int,
    *,
    private_key: torch.Tensor | None = None,
    private_value: torch.Tensor | None = None,
    rotary_dim: int | None = None,
    theta: float = 1_000_000.0,
) -> AttentionResult:
    """No-copy path: inverse-rotate Q and convert K offset to segment bias."""
    _validate_heads(query, source_key)
    if offset.shape != source_key.shape[1:]:
        raise ValueError("offset must have shape [kv_heads, head_dim]")
    query_heads = query.shape[1]
    scale = 1.0 / math.sqrt(query.shape[-1])
    shared_query = rope_delta(
        query,
        -position_delta,
        rotary_dim=rotary_dim,
        theta=theta,
    )
    expanded_source_key = expand_kv_heads(source_key, query_heads)
    expanded_shared_value = expand_kv_heads(shared_value, query_heads)
    expanded_offset = offset.to(source_key.dtype).repeat_interleave(
        query_heads // offset.shape[0], dim=0
    )
    shared_logits = torch.einsum(
        "qhd,khd->qhk",
        shared_query.to(torch.float32),
        expanded_source_key.to(torch.float32),
    ) * scale
    segment_bias = torch.einsum(
        "qhd,hd->qh", query.to(torch.float32), expanded_offset.to(torch.float32)
    ) * scale
    shared_state = _attention_from_logits(
        shared_logits + segment_bias.unsqueeze(-1), expanded_shared_value
    )

    if private_key is None and private_value is None:
        return shared_state
    if private_key is None or private_value is None:
        raise ValueError("private key and value must be supplied together")
    expanded_private_key = expand_kv_heads(private_key, query_heads)
    expanded_private_value = expand_kv_heads(private_value, query_heads)
    private_logits = torch.einsum(
        "qhd,khd->qhk",
        query.to(torch.float32),
        expanded_private_key.to(torch.float32),
    ) * scale
    private_state = _attention_from_logits(private_logits, expanded_private_value)
    return merge_attention_states(private_state, shared_state)


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual = actual.to(torch.float32)
    expected = expected.to(torch.float32)
    diff = actual - expected
    denominator = torch.linalg.vector_norm(expected).clamp_min(1e-12)
    flat_actual = actual.flatten()
    flat_expected = expected.flatten()
    cosine = torch.nn.functional.cosine_similarity(
        flat_actual.unsqueeze(0), flat_expected.unsqueeze(0)
    ).item()
    return {
        "max_abs": float(diff.abs().max().item()),
        "relative_l2": float((torch.linalg.vector_norm(diff) / denominator).item()),
        "cosine": float(cosine),
    }
