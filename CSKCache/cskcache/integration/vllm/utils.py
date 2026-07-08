from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CSKCacheVllmConfig:
    kv_dir: Path | None
    probe_enabled: bool
    probe_tokens: int
    anchor_tokens: int
    probe_tau: float
    gate_metric: str


def _get_bool(extra: dict[str, Any], key: str, default: bool) -> bool:
    value = extra.get(key)
    if value is None:
        env_key = key.upper().replace(".", "_")
        value = os.environ.get(env_key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _get_int(extra: dict[str, Any], key: str, default: int) -> int:
    value = extra.get(key)
    if value is None:
        env_key = key.upper().replace(".", "_")
        value = os.environ.get(env_key)
    return default if value is None else int(value)


def _get_float(extra: dict[str, Any], key: str, default: float) -> float:
    value = extra.get(key)
    if value is None:
        env_key = key.upper().replace(".", "_")
        value = os.environ.get(env_key)
    return default if value is None else float(value)


def get_extra_config(vllm_config: Any) -> dict[str, Any]:
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return {}
    return getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}


def load_vllm_config(vllm_config: Any) -> CSKCacheVllmConfig:
    extra = get_extra_config(vllm_config)
    kv_dir_raw = extra.get("cskcache.kv_dir") or os.environ.get("CSKCACHE_KV_DIR")
    probe_tokens = _get_int(extra, "cskcache.probe_tokens", 4)
    anchor_tokens = _get_int(extra, "cskcache.anchor_tokens", 32)
    if probe_tokens <= 0:
        raise ValueError("cskcache.probe_tokens must be positive")
    if anchor_tokens < probe_tokens:
        raise ValueError("cskcache.anchor_tokens must be >= cskcache.probe_tokens")
    return CSKCacheVllmConfig(
        kv_dir=Path(str(kv_dir_raw)) if kv_dir_raw else None,
        probe_enabled=_get_bool(extra, "cskcache.probe_enabled", False),
        probe_tokens=probe_tokens,
        anchor_tokens=anchor_tokens,
        probe_tau=_get_float(extra, "cskcache.probe_tau", 0.15),
        gate_metric=str(
            extra.get("cskcache.gate_metric")
            or os.environ.get("CSKCACHE_GATE_METRIC")
            or "max"
        ),
    )
