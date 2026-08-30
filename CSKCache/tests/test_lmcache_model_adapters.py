"""CPU-only tests for CSKCache's vLLM model adaptation boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cskcache.integrations.lmcache.model_adapters import (
    build_default_model_adapter_registry,
)
from cskcache.integrations.lmcache.model_adapters.decoders.mistral import (
    MistralDecoderBuilder,
)
from cskcache.integrations.lmcache.model_adapters.registry import (
    ModelAdapterRegistry,
)
from cskcache.integrations.lmcache.model_adapters.wrappers.pixtral import (
    PixtralTextDecoderResolver,
)


class _Builder:
    def __init__(self, *, bind_input: bool = True) -> None:
        self.bind_input = bind_input

    def build(self, decoder_model, blender):
        bound = decoder_model if self.bind_input else object()
        return SimpleNamespace(vllm_model=bound, blender=blender)


def _model(architecture: str, *, layers: int = 2):
    model_type = type(architecture, (), {})
    value = model_type()
    value.model = SimpleNamespace(layers=[object() for _ in range(layers)])
    return value


def test_registry_binds_direct_decoder_and_validates_layer_count() -> None:
    decoder = _model("DirectDecoder", layers=3)
    registry = ModelAdapterRegistry()
    registry.register_decoder("DirectDecoder", _Builder())

    binding = registry.bind(decoder, blender="blend")

    assert binding.outer_model is decoder
    assert binding.decoder_model is decoder
    assert binding.layerwise_model.vllm_model is decoder
    assert binding.num_layers == 3
    assert binding.text_only is True


def test_default_registry_resolves_pixtral_to_mistral_decoder(monkeypatch) -> None:
    decoder = _model("MistralForCausalLM", layers=4)
    wrapper_type = type("PixtralForConditionalGeneration", (), {})
    wrapper = wrapper_type()
    wrapper.language_model = decoder
    monkeypatch.setattr(MistralDecoderBuilder, "build", _Builder().build)

    binding = build_default_model_adapter_registry().bind(wrapper, object())

    assert binding.outer_model is wrapper
    assert binding.decoder_model is decoder
    assert binding.outer_architecture == "PixtralForConditionalGeneration"
    assert binding.decoder_architecture == "MistralForCausalLM"
    assert binding.num_layers == 4
    assert binding.text_only is True


def test_registry_fails_closed_for_unsupported_or_misbound_decoder() -> None:
    with pytest.raises(NotImplementedError, match="UnsupportedDecoder"):
        ModelAdapterRegistry().bind(_model("UnsupportedDecoder"), object())

    registry = ModelAdapterRegistry()
    registry.register_decoder("DirectDecoder", _Builder(bind_input=False))
    with pytest.raises(RuntimeError, match="different model instance"):
        registry.bind(_model("DirectDecoder"), object())


def test_pixtral_resolver_rejects_missing_or_wrong_decoder() -> None:
    resolver = PixtralTextDecoderResolver()
    with pytest.raises(AttributeError, match="language_model"):
        resolver.resolve(SimpleNamespace())
    with pytest.raises(NotImplementedError, match="MistralForCausalLM"):
        resolver.resolve(SimpleNamespace(language_model=_model("Qwen3ForCausalLM")))
