"""Validate offline ``config.py`` and export it to the shell launcher."""

from __future__ import annotations

import shlex

import config as cfg


def validate_config() -> None:
    if cfg.STORAGE_BACKEND not in {"raw_block", "local_disk"}:
        raise ValueError("STORAGE_BACKEND must be raw_block or local_disk")
    if cfg.CHUNK_SIZE_TOKENS <= 0:
        raise ValueError("CHUNK_SIZE_TOKENS must be positive")
    if cfg.STORAGE_LAYOUT not in {
        "chunk_single_layer",
        "packed_chunks_single_layer",
    }:
        raise ValueError("the current offline encoder requires a single-layer layout")
    if cfg.SKILLS and cfg.COLLECTION is not None:
        raise ValueError("configure SKILLS or COLLECTION, not both")
    if not cfg.SKILLS and cfg.COLLECTION is None:
        raise ValueError("configure at least one Skill or one collection")
    if len(set(cfg.SKILLS)) != len(cfg.SKILLS):
        raise ValueError("SKILLS contains duplicates")
    if cfg.PORT <= 0 or cfg.EXPECTED_LAYERS <= 0:
        raise ValueError("PORT and EXPECTED_LAYERS must be positive")
    if not 0 < cfg.GPU_MEMORY_UTILIZATION <= 1:
        raise ValueError("GPU_MEMORY_UTILIZATION must be in (0, 1]")


def shell_environment() -> dict[str, str]:
    validate_config()
    return {
        "OFFLINE_STORAGE_BACKEND": cfg.STORAGE_BACKEND,
        "CSKCACHE_CHUNK_SIZE_TOKENS": str(cfg.CHUNK_SIZE_TOKENS),
        "CSKCACHE_STORAGE_LAYOUT": cfg.STORAGE_LAYOUT,
        "OFFLINE_SKILLS": "\n".join(cfg.SKILLS),
        "OFFLINE_COLLECTION": cfg.COLLECTION or "",
        "OFFLINE_EXCLUDED_SKILLS": "\n".join(cfg.EXCLUDED_SKILLS),
        "OFFLINE_OVERWRITE": "1" if cfg.OVERWRITE else "0",
        "OFFLINE_DRY_RUN": "1" if cfg.DRY_RUN else "0",
        "OFFLINE_SKILLS_DIR": str(cfg.SKILLS_DIR),
        "VLLM_MODEL_PATH": str(cfg.MODEL_PATH),
        "VLLM_SERVED_NAME": cfg.SERVED_MODEL,
        "SKILL_SAVE_POOL_ROOT": str(cfg.POOL_ROOT),
        "SKILL_POOL_MODEL_DIR_NAME": cfg.POOL_MODEL_DIR,
        "VLLM_PORT": str(cfg.PORT),
        "VLLM_API_KEY": cfg.API_KEY,
        "VLLM_GPU_UTIL": str(cfg.GPU_MEMORY_UTILIZATION),
        "VLLM_MAX_MODEL_LEN": str(cfg.MAX_MODEL_LEN),
        "CSKCACHE_MODEL_NUM_LAYERS": str(cfg.EXPECTED_LAYERS),
        "VLLM_READY_MAX_ATTEMPTS": str(cfg.READINESS_ATTEMPTS),
        "VLLM_READY_INTERVAL": str(cfg.READINESS_INTERVAL_SECONDS),
        "VLLM_SHUTDOWN_TIMEOUT": str(cfg.SHUTDOWN_TIMEOUT_SECONDS),
        "CSKCACHE_RAW_CAPACITY_BYTES": str(cfg.RAW_CAPACITY_BYTES),
        "CSKCACHE_RAW_SLOT_BYTES": str(cfg.RAW_SLOT_BYTES),
        "CSKCACHE_RAW_CONTAINER_ID": cfg.RAW_CONTAINER_ID,
    }


if __name__ == "__main__":
    for name, value in shell_environment().items():
        print(f"export {name}={shlex.quote(value)}")
