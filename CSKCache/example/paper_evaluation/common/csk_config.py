"""Build fail-closed LMCache extra config from an offline CSKCache Catalog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def catalog_path(pool_root: Path, model_id: str, backend: str) -> Path:
    leaf = "raw" if backend == "raw_block" else "layer_files"
    return pool_root / model_id / leaf / "catalog.json"


def require_catalog_skills(path: Path, skill_names: set[str]) -> None:
    """Fail before server startup when an offline cache object is missing."""

    if not path.is_file():
        raise FileNotFoundError(f"offline Catalog does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    available = {
        str(item.get("skill_name"))
        for item in payload.get("objects", [])
        if item.get("skill_name")
    }
    missing = sorted(skill_names - available)
    if missing:
        raise RuntimeError(
            f"offline Catalog {path} is missing Skills: {', '.join(missing)}; "
            "build these KV objects with example/offline_skill_kv first"
        )


def build_extra_config(
    *,
    pool_root: Path,
    model_id: str,
    backend: str,
    chunk_tokens: int,
    storage_layout: str,
    host_layout: str,
    execution_order: str,
    correction_strategy: str,
    calibration_tokens: int,
    calibration_ratio: float | None,
    correction_alpha: float,
    minimum_full_recompute_tokens: int,
    minimum_reuse_tokens: int,
    io_engine: str = "io_uring",
    use_odirect: bool = True,
    queue_depth: int = 64,
    catalog_override: Path | None = None,
) -> dict[str, Any]:
    path = catalog_override or catalog_path(pool_root, model_id, backend)
    if not path.is_file():
        raise FileNotFoundError(f"offline Catalog does not exist: {path}")
    result: dict[str, Any] = {
        "csk_t0_prefetch": True,
        "external_control_enabled": True,
        "exact_save_kv_2td": True,
        "cskcache_metadata_path": str(path),
        "csk_storage_backend": backend,
        "csk_chunk_size_tokens": chunk_tokens,
        "csk_storage_layout": storage_layout,
        "csk_host_layout": host_layout,
        "csk_execution_order": execution_order,
        "csk_prefetch_handle_ttl_seconds": 60.0,
        "csk_correction_strategy": correction_strategy,
        "csk_minimum_full_recompute_tokens": minimum_full_recompute_tokens,
        "csk_calibration_tokens": calibration_tokens,
        "csk_calibration_ratio": calibration_ratio,
        "csk_minimum_reuse_tokens": minimum_reuse_tokens,
        "csk_correction_alpha": correction_alpha,
    }
    if backend == "local_disk":
        return result
    if backend != "raw_block":
        raise ValueError(f"unsupported storage backend: {backend}")
    catalog = json.loads(path.read_text(encoding="utf-8"))
    containers = catalog.get("containers") or []
    if len(containers) != 1:
        raise ValueError("raw-block Catalog must describe exactly one container")
    container = containers[0]
    result.update(
        {
            "storage_plugin.raw_block.module_path": (
                "lmcache.v1.storage_backend.plugins.rust_raw_block_backend"
            ),
            "storage_plugin.raw_block.class_name": "RustRawBlockBackend",
            "rust_raw_block.device_path": container["raw_file_path"],
            "rust_raw_block.capacity_bytes": container["capacity_bytes"],
            "rust_raw_block.block_align": container["alignment_bytes"],
            "rust_raw_block.header_bytes": container["header_bytes"],
            "rust_raw_block.slot_bytes": 128 * 1024**2,
            "rust_raw_block.use_odirect": use_odirect,
            "rust_raw_block.enable_zero_copy": True,
            "rust_raw_block.meta_total_bytes": 64 * 1024**2,
            "rust_raw_block.meta_magic": "CSKRAW01",
            "rust_raw_block.meta_version": container["container_format_version"],
            "rust_raw_block.meta_enable_periodic": False,
            "rust_raw_block.load_checkpoint_on_init": True,
            "rust_raw_block.io_engine": io_engine,
            "rust_raw_block.iouring_queue_depth": queue_depth,
        }
    )
    return result
