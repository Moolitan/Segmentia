from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cskcache.v1.matcher import SegmentCatalog


@dataclass(frozen=True)
class CSKCacheVllmConfig:
    catalog_file: Path | None
    kv_dir: Path | None


def get_extra_config(vllm_config: Any) -> dict[str, Any]:
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return {}
    return getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}


def load_vllm_config(vllm_config: Any) -> CSKCacheVllmConfig:
    extra = get_extra_config(vllm_config)
    catalog_raw = extra.get("cskcache.catalog_file") or os.environ.get(
        "CSKCACHE_CATALOG_FILE"
    )
    kv_dir_raw = extra.get("cskcache.kv_dir") or os.environ.get("CSKCACHE_KV_DIR")
    return CSKCacheVllmConfig(
        catalog_file=Path(str(catalog_raw)) if catalog_raw else None,
        kv_dir=Path(str(kv_dir_raw)) if kv_dir_raw else None,
    )


def load_segment_catalog(config: CSKCacheVllmConfig) -> SegmentCatalog:
    if config.catalog_file is None:
        return SegmentCatalog([])
    return SegmentCatalog.from_json_file(config.catalog_file)

