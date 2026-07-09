from __future__ import annotations

from typing import Any

from cskcache.v1.core.config import CSKCacheConfig

# Backward-compatible alias: the config type now lives in the vLLM-agnostic core
# so it can be built from a dict/env without importing vLLM. Callers that read
# ``config.probe_tokens`` etc. keep working because the field names are unchanged.
CSKCacheVllmConfig = CSKCacheConfig


def get_extra_config(vllm_config: Any) -> dict[str, Any]:
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return {}
    return getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}


def load_vllm_config(vllm_config: Any) -> CSKCacheConfig:
    """Build the CSKCache config from a vLLM config object.

    Thin wrapper over :meth:`CSKCacheConfig.from_vllm`; kept as the integration
    entry point so the adapter's import path is stable.
    """

    return CSKCacheConfig.from_vllm(vllm_config)
