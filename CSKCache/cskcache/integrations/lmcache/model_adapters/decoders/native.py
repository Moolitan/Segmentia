"""Delegation for decoder families already supported natively by LMCache."""

from __future__ import annotations

from typing import Any

from lmcache.v1.compute.models.utils import infer_model_from_vllm


class NativeLMCacheDecoderBuilder:
    def build(self, decoder_model: Any, blender: Any) -> Any:
        return infer_model_from_vllm(
            decoder_model,
            blender=blender,
            enable_sparse=False,
        )
