# CSKCache model adapters

This package owns the model-specific boundary between vLLM model objects and
the layerwise auxiliary execution primitives borrowed from LMCache.

Resolution has two independent stages:

1. A wrapper resolver selects the text decoder that owns the language KV
   cache. For example, Pixtral resolves to its `language_model`.
2. A decoder builder constructs the layerwise auxiliary model for that decoder
   family. For example, Mistral uses the generic Llama-shaped primitive.

To support another multimodal wrapper, add a module below `wrappers/` and
register it in `build_default_model_adapter_registry()`. To support another
decoder family, add a module below `decoders/` and register its builder.
Neither extension requires a change to `worker.py`.

Every binding fails closed unless:

- the wrapper resolves to the expected decoder architecture;
- the decoder builder binds the exact resolved model instance;
- the decoder exposes a non-empty `model.layers` collection.

LMCache's global model registry is intentionally not modified by this package.
