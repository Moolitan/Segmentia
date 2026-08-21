"""Validate interactive ``config.py`` and derive runtime configuration."""

from __future__ import annotations

import json
from pathlib import Path
import shlex

import config as cfg


def backend_root() -> Path:
    leaf = "raw" if cfg.STORAGE_BACKEND == "raw_block" else "layer_files"
    return cfg.POOL_ROOT / cfg.POOL_MODEL_DIR / leaf


def catalog_path() -> Path:
    return backend_root() / "catalog.json"


def validate_config() -> None:
    if cfg.MODE not in {"cskcache", "no_reuse"}:
        raise ValueError("MODE must be cskcache or no_reuse")
    if cfg.STORAGE_BACKEND not in {"raw_block", "local_disk"}:
        raise ValueError("STORAGE_BACKEND must be raw_block or local_disk")
    if cfg.SKILLS and cfg.COLLECTION is not None:
        raise ValueError("configure SKILLS or COLLECTION, not both")
    if len(set(cfg.SKILLS)) != len(cfg.SKILLS):
        raise ValueError("SKILLS contains duplicates")
    if cfg.HOST_LAYOUT not in {"full_layer", "chunk_major"}:
        raise ValueError("HOST_LAYOUT must be full_layer or chunk_major")
    if cfg.EXECUTION_ORDER not in {"h2d_first", "compute_first"}:
        raise ValueError("EXECUTION_ORDER must be h2d_first or compute_first")
    if cfg.STORAGE_BACKEND == "raw_block" and cfg.HOST_LAYOUT != "full_layer":
        raise ValueError("the current raw-block backend requires full_layer")
    for value, name in (
        (cfg.HOST_CHUNK_TOKENS, "HOST_CHUNK_TOKENS"),
        (cfg.MINIMUM_FULL_RECOMPUTE_TOKENS, "MINIMUM_FULL_RECOMPUTE_TOKENS"),
        (cfg.CALIBRATION_TOKENS, "CALIBRATION_TOKENS"),
        (cfg.MINIMUM_REUSE_TOKENS, "MINIMUM_REUSE_TOKENS"),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")


def lmcache_extra_config() -> dict[str, object]:
    if cfg.MODE == "no_reuse":
        return {}
    metadata_path = catalog_path()
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"offline CSKCache Catalog does not exist: {metadata_path}"
        )
    result: dict[str, object] = {
        "csk_t0_prefetch": True,
        "external_control_enabled": True,
        "exact_save_kv_2td": True,
        "cskcache_metadata_path": str(metadata_path),
        "csk_storage_backend": cfg.STORAGE_BACKEND,
        "csk_host_layout": cfg.HOST_LAYOUT,
        "csk_host_chunk_tokens": cfg.HOST_CHUNK_TOKENS,
        "csk_execution_order": cfg.EXECUTION_ORDER,
        "csk_prefetch_handle_ttl_seconds": cfg.PREFETCH_HANDLE_TTL_SECONDS,
        "csk_minimum_full_recompute_tokens": cfg.MINIMUM_FULL_RECOMPUTE_TOKENS,
        "csk_calibration_tokens": cfg.CALIBRATION_TOKENS,
        "csk_minimum_reuse_tokens": cfg.MINIMUM_REUSE_TOKENS,
        "csk_correction_alpha": cfg.CORRECTION_ALPHA,
    }
    if cfg.STORAGE_BACKEND == "raw_block":
        catalog = json.loads(metadata_path.read_text(encoding="utf-8"))
        containers = catalog.get("containers") or []
        if len(containers) != 1:
            raise ValueError("raw_block requires exactly one Catalog container")
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
                "rust_raw_block.slot_bytes": cfg.RAW_SLOT_BYTES,
                "rust_raw_block.use_odirect": True,
                "rust_raw_block.enable_zero_copy": True,
                "rust_raw_block.meta_total_bytes": cfg.RAW_METADATA_BYTES,
                "rust_raw_block.meta_magic": cfg.RAW_METADATA_MAGIC,
                "rust_raw_block.meta_version": container["container_format_version"],
                "rust_raw_block.meta_enable_periodic": False,
                "rust_raw_block.load_checkpoint_on_init": True,
                "rust_raw_block.io_engine": cfg.RAW_IO_ENGINE,
                "rust_raw_block.iouring_queue_depth": cfg.RAW_QUEUE_DEPTH,
            }
        )
    return result


def shell_environment() -> dict[str, str]:
    validate_config()
    connector = (
        {
            "kv_connector": "CSKCacheConnectorV1",
            "kv_connector_module_path": "cskcache.integrations.vllm.connector",
            "kv_role": "kv_both",
        }
        if cfg.MODE == "cskcache"
        else {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
    )
    return {
        "INTERACTIVE_MODE": cfg.MODE,
        "INTERACTIVE_STORAGE_BACKEND": cfg.STORAGE_BACKEND,
        "INTERACTIVE_POOL_DIR": str(backend_root()),
        "INTERACTIVE_WORKSPACE": str(cfg.WORKSPACE),
        "VLLM_MODEL_PATH": str(cfg.MODEL_PATH),
        "VLLM_SERVED_NAME": cfg.SERVED_MODEL,
        "VLLM_PORT": str(cfg.PORT),
        "VLLM_API_KEY": cfg.API_KEY,
        "VLLM_GPU_UTIL": str(cfg.GPU_MEMORY_UTILIZATION),
        "VLLM_MAX_MODEL_LEN": str(cfg.MAX_MODEL_LEN),
        "VLLM_TOOL_CALL_PARSER": cfg.TOOL_CALL_PARSER,
        "VLLM_REASONING_PARSER": cfg.REASONING_PARSER,
        "LMCACHE_EXTRA_CONFIG": json.dumps(
            lmcache_extra_config(), separators=(",", ":")
        ),
        "VLLM_KV_TRANSFER_CONFIG": json.dumps(connector, separators=(",", ":")),
        "LMCACHE_CHUNK_SIZE": str(cfg.HOST_CHUNK_TOKENS),
        "LMCACHE_MAX_LOCAL_CPU_SIZE": str(cfg.LMCACHE_MAX_LOCAL_CPU_SIZE_GB),
        "LMCACHE_MAX_LOCAL_DISK_SIZE": str(cfg.LMCACHE_MAX_LOCAL_DISK_SIZE_GB),
        "LMCACHE_STORAGE_PLUGINS": (
            "raw_block"
            if cfg.MODE == "cskcache" and cfg.STORAGE_BACKEND == "raw_block"
            else ""
        ),
        "LMCACHE_LOCAL_DISK": (
            str(backend_root())
            if cfg.MODE == "cskcache" and cfg.STORAGE_BACKEND == "local_disk"
            else ""
        ),
        "CSKCACHE_PROFILE": "1" if cfg.PROFILE else "0",
        "CSKCACHE_FINE_TIMELINE": "1" if cfg.FINE_TIMELINE else "0",
        "CSKCACHE_DISABLE_VISUALIZER": "1" if cfg.DISABLE_VISUALIZER else "0",
    }


if __name__ == "__main__":
    for name, value in shell_environment().items():
        print(f"export {name}={shlex.quote(value)}")
