"""Mistral text-decoder adaptation for CSKCache correction."""

from __future__ import annotations

from typing import Any

from lmcache.v1.compute.models.llama import LMCLlamaModel


class MistralDecoderBuilder:
    """Use LMCache's generic Llama-shaped auxiliary execution primitive."""

    def build(self, decoder_model: Any, blender: Any) -> Any:
        return LMCLlamaModel(
            decoder_model,
            blender,
            enable_sparse=False,
        )
