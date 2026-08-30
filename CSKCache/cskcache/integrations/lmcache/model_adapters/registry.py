"""Composable wrapper and decoder registries for CSKCache."""

from __future__ import annotations

from typing import Any

from .base import (
    DecoderBuilder,
    LayerwiseModelBinding,
    WrapperResolver,
)


class ModelAdapterRegistry:
    """Resolve wrappers first, then dispatch the resulting decoder family."""

    def __init__(self) -> None:
        self._wrappers: dict[str, WrapperResolver] = {}
        self._decoders: dict[str, DecoderBuilder] = {}

    def register_wrapper(
        self,
        architecture: str,
        resolver: WrapperResolver,
    ) -> None:
        self._register(self._wrappers, architecture, resolver, "wrapper")

    def register_decoder(
        self,
        architecture: str,
        builder: DecoderBuilder,
    ) -> None:
        self._register(self._decoders, architecture, builder, "decoder")

    def bind(self, outer_model: Any, blender: Any) -> LayerwiseModelBinding:
        outer_architecture = type(outer_model).__name__
        resolver = self._wrappers.get(outer_architecture)
        if resolver is None:
            decoder_model = outer_model
            text_only = True
        else:
            decoder_model = resolver.resolve(outer_model)
            text_only = bool(resolver.text_only)

        decoder_architecture = type(decoder_model).__name__
        builder = self._decoders.get(decoder_architecture)
        if builder is None:
            supported = ", ".join(sorted(self._decoders)) or "<none>"
            raise NotImplementedError(
                "CSKCache has no layerwise decoder adapter for "
                f"{decoder_architecture}; supported decoders: {supported}"
            )
        layerwise_model = builder.build(decoder_model, blender)
        if getattr(layerwise_model, "vllm_model", None) is not decoder_model:
            raise RuntimeError(
                "CSKCache decoder adapter bound a different model instance"
            )
        model_body = getattr(decoder_model, "model", None)
        layers = getattr(model_body, "layers", None)
        if layers is None:
            raise AttributeError(
                f"{decoder_architecture} does not expose model.layers"
            )
        num_layers = len(layers)
        if num_layers <= 0:
            raise ValueError(f"{decoder_architecture} has no decoder layers")
        return LayerwiseModelBinding(
            outer_model=outer_model,
            decoder_model=decoder_model,
            layerwise_model=layerwise_model,
            outer_architecture=outer_architecture,
            decoder_architecture=decoder_architecture,
            num_layers=num_layers,
            text_only=text_only,
        )

    @staticmethod
    def _register(
        registry: dict[str, Any],
        architecture: str,
        adapter: Any,
        kind: str,
    ) -> None:
        architecture = architecture.strip()
        if not architecture:
            raise ValueError(f"{kind} architecture must be non-empty")
        if architecture in registry:
            raise ValueError(
                f"{kind} adapter is already registered: {architecture}"
            )
        registry[architecture] = adapter
