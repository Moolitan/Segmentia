"""Framework boundary types for CSKCache layerwise model adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LayerwiseModelBinding:
    """One validated mapping from a vLLM wrapper to its text decoder."""

    outer_model: Any
    decoder_model: Any
    layerwise_model: Any
    outer_architecture: str
    decoder_architecture: str
    num_layers: int
    text_only: bool


class WrapperResolver(Protocol):
    """Resolve a framework wrapper to the decoder that owns language KV."""

    text_only: bool

    def resolve(self, outer_model: Any) -> Any: ...


class DecoderBuilder(Protocol):
    """Build the layerwise auxiliary model for one decoder family."""

    def build(self, decoder_model: Any, blender: Any) -> Any: ...
