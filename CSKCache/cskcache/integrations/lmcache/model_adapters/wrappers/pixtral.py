"""Text-only decoder resolution for vLLM Pixtral wrappers."""

from __future__ import annotations

from typing import Any


class PixtralTextDecoderResolver:
    """Select the Mistral language decoder; never adapt the vision tower."""

    text_only = True

    def resolve(self, outer_model: Any) -> Any:
        decoder = getattr(outer_model, "language_model", None)
        if decoder is None:
            raise AttributeError(
                "Pixtral wrapper does not expose language_model"
            )
        architecture = type(decoder).__name__
        if architecture != "MistralForCausalLM":
            raise NotImplementedError(
                "Pixtral language_model must be MistralForCausalLM, "
                f"found {architecture}"
            )
        return decoder
