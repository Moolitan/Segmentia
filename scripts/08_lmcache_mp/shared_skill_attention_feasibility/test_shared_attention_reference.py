from __future__ import annotations

import pytest
import torch

from shared_attention_reference import (
    error_metrics,
    materialized_attention,
    rope_delta,
    shared_attention,
)


def tensors(dtype: torch.dtype = torch.float32):
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(5, 40, 128, generator=generator, dtype=dtype)
    k = torch.randn(19, 8, 128, generator=generator, dtype=dtype)
    v = torch.randn(19, 8, 128, generator=generator, dtype=dtype)
    mu = torch.randn(8, 128, generator=generator, dtype=torch.float32) * 0.03
    private_k = torch.randn(11, 8, 128, generator=generator, dtype=dtype)
    private_v = torch.randn(11, 8, 128, generator=generator, dtype=dtype)
    return q, k, v, mu, private_k, private_v


@pytest.mark.parametrize("delta", [-4096, -37, 0, 73, 8192])
@pytest.mark.parametrize("with_private", [False, True])
def test_shared_form_matches_materialized_fp32(delta: int, with_private: bool):
    q, k, v, mu, private_k, private_v = tensors()
    kwargs = (
        {"private_key": private_k, "private_value": private_v}
        if with_private
        else {}
    )
    materialized = materialized_attention(q, k, v, mu, delta, **kwargs)
    shared = shared_attention(q, k, v, mu, delta, **kwargs)
    output_error = error_metrics(shared.output, materialized.output)
    lse_error = error_metrics(shared.lse, materialized.lse)
    assert output_error["max_abs"] <= 1e-5
    assert output_error["relative_l2"] <= 1e-6
    assert lse_error["max_abs"] <= 1e-5


@pytest.mark.parametrize("delta", [-913, 0, 2003])
def test_shared_form_stays_close_with_bfloat16_inputs(delta: int):
    q, k, v, mu, private_k, private_v = tensors(torch.bfloat16)
    materialized = materialized_attention(
        q,
        k,
        v,
        mu,
        delta,
        private_key=private_k,
        private_value=private_v,
    )
    shared = shared_attention(
        q,
        k,
        v,
        mu,
        delta,
        private_key=private_k,
        private_value=private_v,
    )
    metrics = error_metrics(shared.output, materialized.output)
    assert metrics["relative_l2"] <= 1e-2
    assert metrics["cosine"] >= 0.9999


def test_rope_delta_is_invertible_fp32():
    q, *_ = tensors()
    restored = rope_delta(rope_delta(q, 173), -173)
    metrics = error_metrics(restored, q)
    assert metrics["max_abs"] <= 1e-5


def test_rejects_incompatible_gqa_heads():
    q, k, v, mu, *_ = tensors()
    with pytest.raises(ValueError, match="divisible"):
        shared_attention(q[:, :39], k, v, mu, 0)
