from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CSKCacheVllmConfig:
    kv_dir: Path | None


def get_extra_config(vllm_config: Any) -> dict[str, Any]:
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return {}
    return getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}


def load_vllm_config(vllm_config: Any) -> CSKCacheVllmConfig:
    extra = get_extra_config(vllm_config)
    kv_dir_raw = extra.get("cskcache.kv_dir") or os.environ.get("CSKCACHE_KV_DIR")
    return CSKCacheVllmConfig(
        kv_dir=Path(str(kv_dir_raw)) if kv_dir_raw else None,
    )
