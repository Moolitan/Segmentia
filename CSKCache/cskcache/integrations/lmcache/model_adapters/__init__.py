"""Default CSKCache model-adapter composition."""

from __future__ import annotations

from .decoders import (
    MistralDecoderBuilder,
    NativeLMCacheDecoderBuilder,
)
from .registry import ModelAdapterRegistry
from .wrappers import PixtralTextDecoderResolver


def build_default_model_adapter_registry() -> ModelAdapterRegistry:
    registry = ModelAdapterRegistry()
    registry.register_wrapper(
        "PixtralForConditionalGeneration",
        PixtralTextDecoderResolver(),
    )
    native = NativeLMCacheDecoderBuilder()
    for architecture in (
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
    ):
        registry.register_decoder(architecture, native)
    registry.register_decoder(
        "MistralForCausalLM",
        MistralDecoderBuilder(),
    )
    return registry


MODEL_ADAPTERS = build_default_model_adapter_registry()

__all__ = [
    "MODEL_ADAPTERS",
    "ModelAdapterRegistry",
    "build_default_model_adapter_registry",
]
